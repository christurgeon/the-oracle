"""Canonical outcome derivation for weather threshold markets.

These rules MUST match the resolution logic Kalshi uses to settle the market so
that offline isotonic fitting (scripts/fit_isotonic.py) and the live
calibration monitor agree on ground truth. Keep this module as the single
source of truth; all other code must import from here.
"""

from __future__ import annotations


def derive_outcome_yes(
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
    actual_temp: float,
) -> bool | None:
    """Return the YES-side binary outcome for a threshold market.

    Returns None if the market's strike parameters are inconsistent with
    ``strike_type`` (caller should skip such records).

    Kalshi settlement convention (verified empirically on the
    post-pilot corpus):

      - greater: YES if ``actual_temp > floor_strike`` (STRICT).
                 Kalshi rules_primary text says "greater than X°" verbatim
                 (verified on KXHIGHNY-26MAY16-T83 via Kalshi API: "is
                 greater than 83°, then the market resolves to Yes").
                 Matches production ``EnsembleForecast.fraction_above``
                 (strict ``>``). Brackets partition cleanly: e.g., for one
                 event T67(less) | B60.5(between) ... | B66.5(between, {66,67})
                 | T67(greater, ≥68) — observed=67 is in B66.5, NOT in T67.
      - less:    YES if ``actual_temp <  cap_strike`` (strict; matches
                 production ``fraction_below = 1 - P(t >= cap)``).
      - between: YES if ``floor_strike <= actual_temp <= cap_strike``
                 (CLOSED, right-INCLUSIVE). An earlier version used
                 right-exclusive ``< cap_strike``, which mis-labelled
                 every ``actual_temp == cap_strike`` case as NO. Truth
                 audit: 42 of 42 settled markets with ``t == cap`` paid
                 YES on Kalshi; the buggy code reported NO on all 42.
                 Production ``EnsembleForecast.fraction_between`` already
                 uses CLOSED semantics (``floor <= t <= cap``), so this
                 keeps the model probability and binary outcome
                 contract-consistent.
    """
    if strike_type == "greater":
        if floor_strike is None:
            return None
        return actual_temp > floor_strike
    if strike_type == "less":
        if cap_strike is None:
            return None
        return actual_temp < cap_strike
    if strike_type == "between":
        if floor_strike is None or cap_strike is None:
            return None
        return floor_strike <= actual_temp <= cap_strike
    return None


def derive_outcome_for_side(
    side: str,
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
    actual_temp: float,
) -> bool | None:
    """Return the side-denominated outcome (True = our side won).

    ``side`` must be "YES" or "NO".
    """
    yes_outcome = derive_outcome_yes(strike_type, floor_strike, cap_strike, actual_temp)
    if yes_outcome is None:
        return None
    if side == "YES":
        return yes_outcome
    if side == "NO":
        return not yes_outcome
    return None
