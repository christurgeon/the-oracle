"""Manual-close / cash-out cost-basis helper.

Used by FillsReconciler._apply_manual_close and _apply_cash_out. Reads
state/trades.jsonl and aggregates entry FILL rows for one
(position_key, market_ticker).

No cache. No lock-state assertion. Manual closes are sub-weekly events;
a full file scan per close is trivially cheap. Caller MUST hold
trade_log_lock during the scan.

`restore_fills_from_log` in engine.py keeps its own inline aggregation —
it's display-only, has no fees field, and doesn't share this code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from python.execution.trade_log import (
    TRADE_LOG_EVENT_CASH_OUT,
    TRADE_LOG_EVENT_CASH_OUT_FORCE_APPLIED,
    TRADE_LOG_EVENT_CASH_OUT_REVERTED,
    is_pnl_relevant_row,
)


@dataclass(frozen=True)
class CostBasis:
    """Aggregated entry-side stats for one (position_key, market_ticker).

    Fees are tracked separately from total_dollars — collapsing them would
    cause realized-P&L double-counting via fee-on-both-sides.
    """

    total_dollars: float
    total_contracts: int
    total_fees: float
    last_strategy: str | None
    n_fills: int


def cost_basis_for_position(
    trade_log_path: Path,
    *,
    position_key: str,
    market_ticker: str,
) -> CostBasis:
    """Aggregate entry FILL rows for one position, NET of any prior cash-out offsets.

    Two-section algorithm:
    1. FILL pass — sums dollars/contracts/fees over rows where is_pnl_relevant_row
       is True AND row's position_key matches (legacy fallback: market_ticker).
    2. Cash-out adjustment pass — for rows with event in
       {cash_out, cash_out_force_applied, cash_out_reverted}, subtract (cash_out
       and cash_out_force_applied) or add back (cash_out_reverted) the
       `cost_basis_offset_*` fields. This pass BYPASSES is_pnl_relevant_row
       (all three event types return False from that gate).

    Returns the EFFECTIVE remaining cost basis.
    """
    total_dollars = 0.0
    total_contracts = 0
    total_fees = 0.0
    last_strategy: str | None = None
    n_fills = 0

    if not trade_log_path.exists():
        return CostBasis(0.0, 0, 0.0, None, 0)

    with trade_log_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_pk = row.get("position_key")
            row_ticker = row.get("market_ticker")
            # Fall back to market_ticker only for legacy rows that predate position_key.
            if row_pk is not None:
                if row_pk != position_key:
                    continue
            else:
                if row_ticker != market_ticker:
                    continue

            event = str(row.get("event", "fill"))

            if is_pnl_relevant_row(row):
                try:
                    price = float(row.get("fill_price", 0))
                    contracts = int(row.get("contracts", 0))
                    fees = float(row.get("fees_from_api", 0))
                except (TypeError, ValueError):
                    continue
                if contracts <= 0:
                    continue
                total_dollars += price * contracts
                total_contracts += contracts
                total_fees += fees
                n_fills += 1
                strat = row.get("metadata", {}).get("strategy")
                if strat:
                    last_strategy = str(strat)
                continue

            # Cash-out adjustment (bypasses is_pnl_relevant_row).
            if event in (TRADE_LOG_EVENT_CASH_OUT, TRADE_LOG_EVENT_CASH_OUT_FORCE_APPLIED):
                sign = -1
            elif event == TRADE_LOG_EVENT_CASH_OUT_REVERTED:
                sign = +1
            else:
                continue

            try:
                offset_dollars = float(row.get("cost_basis_offset_dollars", 0))
                offset_contracts = int(row.get("contracts_offset", 0))
                offset_fees = float(row.get("cost_basis_offset_fees", 0))
            except (TypeError, ValueError):
                continue
            total_dollars += sign * offset_dollars
            total_contracts += sign * offset_contracts
            total_fees += sign * offset_fees

    return CostBasis(
        total_dollars=total_dollars,
        total_contracts=total_contracts,
        total_fees=total_fees,
        last_strategy=last_strategy,
        n_fills=n_fills,
    )


@dataclass(frozen=True)
class PartialBasis:
    """Result of allocating a CostBasis between offset and residual legs.

    Proportional per-contract on a fungible pool — no FIFO/LIFO.
    `offset_*` fields are rounded to 4dp; `residual_*` are computed by
    UNROUNDED subtraction so `offset + residual == total` to the cent.
    """

    offset_dollars: float
    offset_fees: float
    residual_dollars: float
    residual_fees: float


def partial_basis(cb: CostBasis, n_offset: int) -> PartialBasis:
    """Allocate cost basis proportionally for a partial close.

    Raises:
        ValueError: if n_offset <= 0 or n_offset > cb.total_contracts.
    """
    if n_offset <= 0:
        raise ValueError(f"n_offset must be positive, got {n_offset}")
    if n_offset > cb.total_contracts:
        raise ValueError(
            f"n_offset={n_offset} > cb.total_contracts={cb.total_contracts}"
        )
    if n_offset == cb.total_contracts:
        return PartialBasis(cb.total_dollars, cb.total_fees, 0.0, 0.0)
    ratio = n_offset / cb.total_contracts
    offset_dollars = round(cb.total_dollars * ratio, 4)
    offset_fees = round(cb.total_fees * ratio, 4)
    residual_dollars = cb.total_dollars - offset_dollars
    residual_fees = cb.total_fees - offset_fees
    return PartialBasis(offset_dollars, offset_fees, residual_dollars, residual_fees)
