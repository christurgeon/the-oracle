"""Mark-to-market portfolio snapshot for `/status`.

Pure observability: never touches ``RiskManager._bankroll`` or any sizing
inputs. The cost-basis bankroll remains authoritative for Kelly sizing,
exposure caps, and drawdown halts (see ``python/risk/manager.py``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from python.common import Side
from python.strategies.weather_parsing import _parse_dollar_price

if TYPE_CHECKING:
    from python.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioMtm:
    cash: float
    position_mtm: float
    equity_mtm: float
    n_positions_fetched: int
    as_of_utc: datetime
    degraded: bool
    breakdown: tuple[tuple[str, Side, float, float, float], ...] = field(default=())
    # (ticker, side, qty_abs, bid_for_side, contribution_dollars)


def _parse_optional_decimal(value: object) -> float | None:
    """Parse a Kalshi-style decimal field; return None on missing/unparseable."""
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def position_mtm_contribution(
    qty_fp: float, yes_bid: float | None, yes_ask: float | None
) -> tuple[Side, float, float, float]:
    """Pure-math seam. Returns ``(side, qty_abs, px, contribution_dollars)``.

    ``Side.NONE`` is returned for ``qty_fp == 0``; callers should skip those
    rows from any displayed breakdown.

    A legitimate ``yes_bid == 0.0`` yields zero contribution — conservative
    liquidation matches Kalshi's UI behavior on a no-bid market. Missing /
    unparseable prices also yield zero.
    """
    if qty_fp == 0.0:
        return Side.NONE, 0.0, 0.0, 0.0

    side = Side.YES if qty_fp > 0 else Side.NO
    qty_abs = abs(qty_fp)

    if side == Side.YES:
        px = yes_bid if yes_bid is not None else 0.0
    else:
        px = (1.0 - yes_ask) if yes_ask is not None else 0.0

    px = max(0.0, min(1.0, px))
    return side, qty_abs, px, qty_abs * px


def _unwrap_market(resp: dict[str, Any]) -> dict[str, Any]:
    # Kalshi v2 /markets/{ticker} wraps the payload as {"market": {...}};
    # mirrors engine.py and fetch_shadow_settlements.py.
    inner = resp.get("market")
    return inner if isinstance(inner, dict) else resp


def _quote_prices(market: dict[str, Any]) -> tuple[float | None, float | None]:
    if market.get("status", "active") == "active":
        return (
            _parse_dollar_price(market.get("yes_bid_dollars")),
            _parse_dollar_price(market.get("yes_ask_dollars")),
        )
    last_price = _parse_dollar_price(market.get("last_price_dollars"))
    return last_price, last_price


async def compute_portfolio_mtm(
    kalshi_client: KalshiClient,
    *,
    market_concurrency: int = 2,
) -> PortfolioMtm | None:
    """Snapshot of current mark-to-market portfolio value.

    Returns ``None`` on catastrophic failure (balance or positions fetch
    raises). Returns ``degraded=True`` when ``iter_all_positions`` was
    truncated OR any per-market fetch failed; partial MTM still rendered.
    Never raises.

    ``market_concurrency`` is intentionally low (default 2) to keep added
    rate-limit pressure well clear of Kalshi's ~100-req/10s sliding window
    even when a strategy scan tick is in flight.
    """
    try:
        balance_resp, positions_resp = await asyncio.gather(
            kalshi_client.get_balance(),
            kalshi_client.iter_all_positions(settlement_status="unsettled"),
        )
    except Exception:
        logger.warning(
            "compute_portfolio_mtm: balance/positions fetch failed", exc_info=True
        )
        return None

    cash_cents = _parse_optional_decimal(balance_resp.get("balance"))
    cash = (cash_cents or 0.0) / 100.0
    positions, truncated = positions_resp
    degraded = bool(truncated)

    open_positions: list[tuple[dict[str, Any], float]] = []
    for p in positions:
        qty = _parse_optional_decimal(p.get("position_fp"))
        if qty is None or qty == 0.0 or not p.get("ticker"):
            continue
        open_positions.append((p, qty))

    sem = asyncio.Semaphore(market_concurrency)

    async def fetch_one(
        pos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with sem:
            try:
                return pos, await kalshi_client.get_market(pos["ticker"])
            except Exception:
                logger.debug(
                    "compute_portfolio_mtm: get_market(%s) failed",
                    pos.get("ticker"),
                    exc_info=True,
                )
                return pos, None

    results = await asyncio.gather(*(fetch_one(p) for p, _ in open_positions))

    qty_by_ticker = {p["ticker"]: q for p, q in open_positions}
    breakdown: list[tuple[str, Side, float, float, float]] = []
    position_mtm = 0.0

    for pos, mkt in results:
        if mkt is None:
            degraded = True
            continue
        market = _unwrap_market(mkt)
        yes_bid, yes_ask = _quote_prices(market)
        side, qty_abs, px, contribution = position_mtm_contribution(
            qty_by_ticker[pos["ticker"]], yes_bid, yes_ask
        )
        if side == Side.NONE:
            continue
        position_mtm += contribution
        breakdown.append((pos["ticker"], side, qty_abs, px, contribution))

    return PortfolioMtm(
        cash=cash,
        position_mtm=position_mtm,
        equity_mtm=cash + position_mtm,
        n_positions_fetched=len(open_positions),
        as_of_utc=datetime.now(UTC),
        degraded=degraded,
        breakdown=tuple(breakdown),
    )
