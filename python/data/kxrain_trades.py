"""Pure aggregation of KXRAIN trade prints into the 30-minute window summary.

Hermetic — no client, no I/O. Sorts by parsed timestamp: ``created_time``'s
variable sub-second precision breaks a raw string sort."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "TradePrint",
    "TradeWindow",
    "MAX_PRINTS",
    "summarise_trade_window",
    "to_dollar_price",
    "to_epoch",
    "to_fp_float",
]

# Caps "prints" so a trades row stays under PIPE_BUF; aggregate fields still
# cover the whole page.
MAX_PRINTS = 12

_TAKER_SIDES = frozenset({"yes", "no"})


def to_dollar_price(val: object) -> float | None:
    """Parse a Kalshi ``*_dollars`` field to a probability in [0, 1]."""
    if val is None:
        return None
    try:
        price = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if price < 0.0 or price > 1.0:
        return None
    return price


def to_fp_float(val: object) -> float | None:
    """Parse a Kalshi ``*_fp`` field to float. Kept fractional: sub-1 counts
    like ``"0.37"`` truncate to 0 and vanish under ``int(float(...))``."""
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def to_epoch(created_time: object) -> float | None:
    if not isinstance(created_time, str) or not created_time:
        return None
    try:
        return datetime.fromisoformat(created_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class TradePrint:
    """One taker print, coerced."""

    ts: str
    taker_side: str
    yes_price: float
    no_price: float
    count: float


@dataclass(frozen=True)
class TradeWindow:
    """Aggregates over one ``/markets/trades`` page."""

    n_trades_total: int
    n_trades_no_side: int
    no_volume: float
    yes_volume: float
    vwap_no: float | None
    last_no_trade_ts: str | None
    last_no_trade_price: float | None
    prints: tuple[TradePrint, ...]
    prints_truncated: bool
    page_truncated: bool
    parse_failures: int
    # Oldest print on the page, not in "prints" (they diverge once max_prints
    # bites) — records the page's real span since it can be under 30m.
    oldest_print_ts: str | None = None
    oldest_print_epoch: float | None = None

    @property
    def has_recent_no_trade(self) -> bool:
        return self.n_trades_no_side > 0


def summarise_trade_window(
    payload: dict[str, Any] | None, *, max_prints: int = MAX_PRINTS
) -> TradeWindow:
    """Reduce a ``/markets/trades`` payload to the window summary. Never raises.

    A malformed row increments ``parse_failures`` and is excluded rather than
    aborting the page. An empty page is a valid "not eligible" answer,
    distinct from a failed fetch (the caller passes ``None`` for that)."""
    rows = (payload or {}).get("trades") or []
    cursor = (payload or {}).get("cursor") or ""

    parsed: list[tuple[float, TradePrint]] = []
    failures = 0
    for row in rows:
        if not isinstance(row, dict):
            failures += 1
            continue
        side = str(row.get("taker_side") or "").strip().lower()
        yes_price = to_dollar_price(row.get("yes_price_dollars"))
        no_price = to_dollar_price(row.get("no_price_dollars"))
        count = to_fp_float(row.get("count_fp"))
        epoch = to_epoch(row.get("created_time"))
        if (
            side not in _TAKER_SIDES
            or yes_price is None
            or no_price is None
            or count is None
            or count <= 0.0
            or epoch is None
        ):
            failures += 1
            continue
        parsed.append(
            (
                epoch,
                TradePrint(
                    ts=str(row["created_time"]),
                    taker_side=side,
                    yes_price=yes_price,
                    no_price=no_price,
                    count=count,
                ),
            )
        )

    if failures:
        logger.debug("kxrain trade window: %d/%d prints unparseable", failures, len(rows))

    parsed.sort(key=lambda item: item[0], reverse=True)
    ordered = [p for _, p in parsed]
    oldest_epoch, oldest_print = parsed[-1] if parsed else (None, None)

    no_prints = [p for p in ordered if p.taker_side == "no"]
    no_volume = sum(p.count for p in no_prints)
    yes_volume = sum(p.count for p in ordered if p.taker_side == "yes")

    vwap_no: float | None = None
    if no_volume > 0:
        vwap_no = sum(p.no_price * p.count for p in no_prints) / no_volume

    newest_no = no_prints[0] if no_prints else None

    return TradeWindow(
        n_trades_total=len(rows),
        n_trades_no_side=len(no_prints),
        no_volume=no_volume,
        yes_volume=yes_volume,
        vwap_no=vwap_no,
        last_no_trade_ts=newest_no.ts if newest_no else None,
        last_no_trade_price=newest_no.no_price if newest_no else None,
        prints=tuple(ordered[:max_prints]),
        prints_truncated=len(ordered) > max_prints,
        page_truncated=bool(cursor),
        parse_failures=failures,
        oldest_print_ts=oldest_print.ts if oldest_print is not None else None,
        oldest_print_epoch=oldest_epoch,
    )
