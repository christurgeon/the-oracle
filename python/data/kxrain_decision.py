"""Pure evaluation of the pre-registered KXRAIN "buy NO" eligibility rule.

Hermetic: no client, no I/O, no strategy imports and no import-time
``assert`` — it logs a decision, it never models the weather."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SKIP_REASONS",
    "DecisionOutcome",
    "evaluate_decision",
]

# "market_absent" is never emitted by evaluate_decision(); it labels the
# census row for a ticker that vanished from /markets before entry.
SKIP_REASONS: frozenset[str] = frozenset(
    {
        "out_of_window",
        "no_price",
        "price_band",
        "already_entered",
        "trades_unavailable",
        "no_recent_no_trade",
        "market_absent",
    }
)


@dataclass(frozen=True)
class DecisionOutcome:
    """One evaluated check against the pre-registered rule.

    ``eligible=None`` means UNKNOWN (trades fetch failed) — never coerce to ``False``."""

    eligible: bool | None
    skip_reason: str | None
    in_price_band: bool
    no_ask: float | None
    no_trade_recent: bool | None
    yes_bid_ge_040: bool
    assumed_entry_price: float | None


def evaluate_decision(
    *,
    yes_bid: float | None,
    in_window: bool,
    no_trades_30m: int | None,
    already_entered: bool,
    yes_bid_min: float = 0.10,
    yes_bid_max: float = 0.90,
    secondary_cut: float = 0.40,
) -> DecisionOutcome:
    """Evaluate one decision check. Pure, keyword-only, never raises.

    ``in_window`` must already encode the settlement-date and local-time
    checks; this function does not recompute them."""
    if yes_bid is None or not math.isfinite(yes_bid):
        no_ask: float | None = None
        in_price_band = False
        yes_bid_ge_040 = False
    else:
        no_ask = 1.0 - yes_bid
        in_price_band = yes_bid_min <= yes_bid < yes_bid_max
        yes_bid_ge_040 = yes_bid >= secondary_cut

    no_trade_recent: bool | None = None if no_trades_30m is None else no_trades_30m > 0

    eligible: bool | None
    skip_reason: str | None

    # out_of_window first: it never fetches trades, so ranking it after
    # trades_unavailable would hide a real trades-fetch outage.
    if not in_window:
        eligible, skip_reason = False, "out_of_window"
    elif no_ask is None:
        eligible, skip_reason = False, "no_price"
    elif not in_price_band:
        eligible, skip_reason = False, "price_band"
    elif already_entered:
        eligible, skip_reason = False, "already_entered"
    elif no_trades_30m is None:
        eligible, skip_reason = None, "trades_unavailable"
    elif no_trades_30m <= 0:
        eligible, skip_reason = False, "no_recent_no_trade"
    else:
        eligible, skip_reason = True, None

    assumed_entry_price = no_ask if eligible is True else None

    return DecisionOutcome(
        eligible=eligible,
        skip_reason=skip_reason,
        in_price_band=in_price_band,
        no_ask=no_ask,
        no_trade_recent=no_trade_recent,
        yes_bid_ge_040=yes_bid_ge_040,
        assumed_entry_price=assumed_entry_price,
    )
