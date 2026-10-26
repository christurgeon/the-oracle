"""Reconstruct shadow-mode positions from the trade log.

``_reconcile_from_kalshi`` queries the exchange for authoritative
positions at startup in live/paper mode; shadow mode has no exchange to
query, and before this the risk manager started every restart with an
empty position book — silently lifting the ``TOTAL_EXPOSURE_CAP`` and
re-firing trades that had been blocked for hours.

The trade log is the source of truth:
  * ``{"event": "fill", ...}`` rows (or legacy rows with no ``event``
    key) add to the position's dollars.
  * ``{"event": "settlement", ...}`` rows drop the position.
  * A 24h ET date safety net keeps entries whose ticker date is today or
    yesterday; anything older is assumed settled-but-unlogged from an
    unclean shutdown prior to settlement-row logging.

CLI::

    uv run python -m python.execution.shadow_broker \
        --log state/trades.jsonl --as-of 2026-04-22
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

from python.common import ExecutionMode, Side, et_today
from python.execution.trade_log import TradeLogEvent, is_pnl_relevant_row
from python.utils.jsonl import iter_jsonl
from python.utils.ticker_dates import parse_ticker_date

logger = logging.getLogger(__name__)

__all__ = ["ShadowBroker", "ShadowPosition"]


@dataclass(frozen=True)
class ShadowPosition:
    """Reconstructed open position from the trade log."""

    position_key: str
    market_ticker: str
    market_id: str
    side: Side
    dollars: float
    strategy_name: str
    close_time: str
    ticker_date: date | None
    order_id: str


class ShadowBroker:
    """Replay ``trades.jsonl`` to derive open positions."""

    def __init__(
        self,
        *,
        trade_log_path: Path,
        configured_bankroll: float,
        today: date | None = None,
    ) -> None:
        self._trade_log_path = trade_log_path
        self._bankroll = configured_bankroll
        self._today = today if today is not None else et_today()

    @property
    def bankroll(self) -> float:
        return self._bankroll

    def replay(self) -> list[ShadowPosition]:
        """Return the currently-open positions implied by the trade log."""
        positions: dict[str, ShadowPosition] = {}
        unparseable_tickers: set[str] = set()
        skipped_non_shadow = 0

        for entry in iter_jsonl(self._trade_log_path):
            # Missing ``mode`` predates the field — treat as shadow.
            if entry.get("mode") in (ExecutionMode.LIVE, ExecutionMode.PAPER):
                skipped_non_shadow += 1
                continue

            event = entry.get("event", TradeLogEvent.FILL.value)
            ticker = entry.get("market_ticker", "")
            if not ticker:
                continue
            key = entry.get("position_key") or entry.get("market_id") or ticker

            if event == TradeLogEvent.SETTLEMENT.value:
                positions.pop(key, None)
                continue

            if event in (
                TradeLogEvent.CASH_OUT.value,
                TradeLogEvent.CASH_OUT_FORCE_APPLIED.value,
            ):
                # Full closes pop the position (mirroring SETTLEMENT); partial
                # closes reduce dollars by the offset. force_applied is the
                # operator-tool variant — same adjustment as an auto cash_out.
                close_type = str(entry.get("close_type", ""))
                if close_type == "full":
                    positions.pop(key, None)
                elif close_type == "partial":
                    existing = positions.get(key)
                    if existing is not None:
                        offset = _coerce_float(entry.get("cost_basis_offset_dollars")) or 0.0
                        positions[key] = replace(
                            existing,
                            dollars=max(0.0, existing.dollars - offset),
                        )
                continue

            if event == TradeLogEvent.CASH_OUT_REVERTED.value:
                # Reverses a prior cash_out / force_applied for the same position.
                # Partial reverts re-add the offset dollars. Full reverts can't be
                # reconstructed from the row alone (the prior position was popped,
                # and the original cost basis lived on it) — replay skips them;
                # the operator runbook covers manual recovery.
                close_type = str(entry.get("close_type", ""))
                if close_type == "partial":
                    existing = positions.get(key)
                    if existing is not None:
                        offset = _coerce_float(entry.get("cost_basis_offset_dollars")) or 0.0
                        positions[key] = replace(
                            existing,
                            dollars=existing.dollars + offset,
                        )
                continue

            if not is_pnl_relevant_row(entry):
                continue

            dollars = _coerce_float(entry.get("dollars"))
            if dollars is None or dollars <= 0:
                continue

            ticker_date = parse_ticker_date(ticker)
            if ticker_date is None and ticker not in unparseable_tickers:
                logger.warning(
                    "ShadowBroker: ticker %s has no parseable date — "
                    "date safety net will not apply",
                    ticker,
                )
                unparseable_tickers.add(ticker)

            close_time = str((entry.get("metadata") or {}).get("close_time", ""))
            existing = positions.get(key)
            if existing is None:
                positions[key] = ShadowPosition(
                    position_key=key,
                    market_ticker=ticker,
                    market_id=entry.get("market_id", "") or "",
                    side=_coerce_side(entry.get("side", "")),
                    dollars=dollars,
                    strategy_name=entry.get("strategy", "") or "",
                    close_time=close_time,
                    ticker_date=ticker_date,
                    order_id=entry.get("order_id", "") or "",
                )
            else:
                # Partial-fill rows add dollars; preserve the first
                # close_time we saw because follow-on rows may omit
                # metadata entirely.
                positions[key] = replace(
                    existing,
                    dollars=existing.dollars + dollars,
                    close_time=existing.close_time or close_time,
                )

        if skipped_non_shadow > 0:
            logger.warning(
                "ShadowBroker: skipped %d live/paper fill row(s) — mode flip "
                "detected in trade log; those positions are not resurrected as "
                "shadow exposure",
                skipped_non_shadow,
            )

        return [p for p in positions.values() if self._keep(p)]

    def _keep(self, pos: ShadowPosition) -> bool:
        if pos.ticker_date is None:
            # Only explicit settlement rows can close an unparseable ticker;
            # the date safety net doesn't apply.
            return True
        # Weather markets settle mid-morning ET the day after the ticker
        # date — keep today and yesterday.
        return pos.ticker_date >= self._today - timedelta(days=1)


def _coerce_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_side(val: object) -> Side:
    try:
        return Side(str(val).upper()) if val else Side.YES
    except ValueError:
        return Side.YES


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m python.execution.shadow_broker",
        description="Dry-run ShadowBroker against a trades.jsonl file.",
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to trades.jsonl")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Treat this ET date as 'today' (YYYY-MM-DD). Defaults to current ET date.",
    )
    parser.add_argument(
        "--bankroll", type=float, default=0.0, help="Configured bankroll (display only)"
    )
    args = parser.parse_args(argv)

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    broker = ShadowBroker(
        trade_log_path=args.log, configured_bankroll=args.bankroll, today=as_of
    )
    positions = broker.replay()

    if not positions:
        print("(no open positions)")
        return 0

    print(f"Open shadow positions as of {broker._today.isoformat()} (ET):")
    total = 0.0
    for pos in sorted(positions, key=lambda p: p.position_key):
        total += pos.dollars
        print(
            f"  {pos.position_key:<40s} side={pos.side.value:<3s} "
            f"${pos.dollars:>8.2f} date={pos.ticker_date} close={pos.close_time or '-'}"
        )
    print(f"\nTotal exposure: ${total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
