"""Central row-filter helpers for state/trades.jsonl.

This module is the schema-evolution gate. Every consumer iterating
trades.jsonl MUST use is_pnl_relevant_row() instead of writing its own
event == 'fill' check.

P&L-relevance map:
  - FILL: True (contributes to cost basis + bankroll)
  - PLACEMENT: False (side-channel for strategy attribution, no P&L)
  - SETTLEMENT: False (P&L already applied via close_position at settle time)
  - MANUAL_CLOSE: False (P&L applied via close_position; row is forensic only,
                  and including it would double-count cost basis as a buy at
                  the close price)
  - CASH_OUT: False (P&L applied via close_position / realize_partial at apply
                  time; row is forensic only)
  - CASH_OUT_REVERTED: False (forensic-only marker for retracted cash-out)
  - CASH_OUT_FORCE_APPLIED: False (forensic-only marker for operator override)
  - CANCEL: False (0-fill terminal status; cost basis and bankroll unchanged
                  from pre-placement state. Consumed by per_day_event_cap
                  hydration to release the (city,var,date) slot a matching
                  placement row would otherwise hold.)

Canonical field semantics (every writer of a fill-bearing row MUST agree):
  - is_taker_fills / is_maker_fills: SUM OF CONTRACTS filled as taker / maker
    (NOT the count of fills). Maps directly to fee/exposure math. An unclassified
    is_taker=None fill counts toward NEITHER (is True / is False identity, not
    truthiness). Both the engine path (engine.py) and all reconciler synthetic-row
    builders (fills_reconciler.py _build_synthetic_row / _apply_manual_close /
    _apply_cash_out) use this semantic, so a downstream consumer reads the same
    meaning regardless of which writer produced the row.
"""

from __future__ import annotations

from enum import StrEnum

TRADE_LOG_EVENT_FILL = "fill"
TRADE_LOG_EVENT_PLACEMENT = "placement"
TRADE_LOG_EVENT_SETTLEMENT = "settlement"
TRADE_LOG_EVENT_MANUAL_CLOSE = "manual_close"
TRADE_LOG_EVENT_CASH_OUT = "cash_out"
TRADE_LOG_EVENT_CASH_OUT_REVERTED = "cash_out_reverted"
TRADE_LOG_EVENT_CASH_OUT_FORCE_APPLIED = "cash_out_force_applied"
TRADE_LOG_EVENT_CANCEL = "cancel"

TRADE_LOG_EVENT_KNOWN: frozenset[str] = frozenset(
    {
        TRADE_LOG_EVENT_FILL,
        TRADE_LOG_EVENT_PLACEMENT,
        TRADE_LOG_EVENT_SETTLEMENT,
        TRADE_LOG_EVENT_MANUAL_CLOSE,
        TRADE_LOG_EVENT_CASH_OUT,
        TRADE_LOG_EVENT_CASH_OUT_REVERTED,
        TRADE_LOG_EVENT_CASH_OUT_FORCE_APPLIED,
        TRADE_LOG_EVENT_CANCEL,
    }
)


class TradeLogEvent(StrEnum):
    """Event discriminator for trades.jsonl rows.

    PLACEMENT: written immediately on successful place_order.
    FILL: written when a fill is observed (engine path or reconciler path).
    SETTLEMENT: written when a market settles.
    MANUAL_CLOSE: written when FillsReconciler detects a sell fill the bot
                  didn't originate (user closed via Kalshi UI).
    CASH_OUT: written when FillsReconciler observes a Kalshi cash-out fill
              (forensic-only; P&L applied via close_position / realize_partial
              at apply time).
    CASH_OUT_REVERTED: forensic-only marker emitted when a previously observed
                       cash-out is retracted by Kalshi.
    CASH_OUT_FORCE_APPLIED: forensic-only marker emitted when an operator
                            forces application of a stale cash-out candidate.
    CANCEL: written when a placed order reaches a terminal-no-fill status
            (CANCELLED/EXPIRED) with fill_count == 0. Used by
            per_day_event_cap hydration to release the (city,var,date) slot
            that the matching placement row would otherwise hold.
    """

    PLACEMENT = TRADE_LOG_EVENT_PLACEMENT
    FILL = TRADE_LOG_EVENT_FILL
    SETTLEMENT = TRADE_LOG_EVENT_SETTLEMENT
    MANUAL_CLOSE = TRADE_LOG_EVENT_MANUAL_CLOSE
    CASH_OUT = TRADE_LOG_EVENT_CASH_OUT
    CASH_OUT_REVERTED = TRADE_LOG_EVENT_CASH_OUT_REVERTED
    CASH_OUT_FORCE_APPLIED = TRADE_LOG_EVENT_CASH_OUT_FORCE_APPLIED
    CANCEL = TRADE_LOG_EVENT_CANCEL


def is_pnl_relevant_row(row: dict[str, object]) -> bool:
    """True iff the row contributes to realised P&L attribution.

    Returns False for every non-fill event, including any future event type
    the consumer doesn't yet recognise. Centralising this filter is the
    schema-evolution contract for trades.jsonl: every consumer iterating the
    file MUST use this helper instead of its own `event == 'fill'` check.

    The default-to-FILL behaviour preserves backward compatibility with rows
    written before the event discriminator existed.
    """
    return str(row.get("event", TRADE_LOG_EVENT_FILL)) == TRADE_LOG_EVENT_FILL


def is_known_event(row: dict[str, object]) -> bool:
    """True iff the row's event type is recognised by the current code.

    Strict consumers (e.g., a future audit tool that wants to fail-loud
    on unknown rows) can use this to detect rows from a future schema.
    Most consumers should use is_pnl_relevant_row() instead.
    """
    return str(row.get("event", TRADE_LOG_EVENT_FILL)) in TRADE_LOG_EVENT_KNOWN
