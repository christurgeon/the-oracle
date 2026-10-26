"""Predicted distance from forecast mean to nearest NWS-rounding boundary.

Evidence: docs/research/2026-05-28-boundary-aware-phase0-backtest.md.
Pure function: zero I/O, deterministic, importable from any layer.
"""

from __future__ import annotations

import math

from python.common import NWS_ROUND_HALF_F


def boundary_distance(
    forecast_mean_f: float,
    floor_strike: float | None,
    cap_strike: float | None,
    strike_type: str,
) -> float:
    """Distance (°F) from forecast mean to the nearest NWS-rounding boundary
    that flips bracket resolution.

    Per the production NWS-rounding convention used by
    ``python.strategies.weather_evaluation``:

    - ``between [floor, cap]`` — YES if NWS-rounded temp in {floor, …, cap},
      i.e., actual T ∈ [floor − 0.5, cap + 0.5). Rounding boundaries:
      ``{floor − 0.5, cap + 0.5}``.
    - ``greater`` (e.g., "high > 53°F") — YES if NWS-rounded temp ≥ floor + 1,
      i.e., T ≥ floor + 0.5. Rounding boundary: ``floor + 0.5``.
    - ``less`` (e.g., "low < 64°F") — YES if NWS-rounded temp ≤ cap − 1,
      i.e., T < cap − 0.5. Rounding boundary: ``cap − 0.5``.

    Returns ``math.nan`` if the strike geometry is incompatible with the
    strike_type (e.g., greater with floor=None). Callers MUST check for NaN
    before using the result in a comparison.
    """
    if strike_type == "between":
        if floor_strike is None or cap_strike is None:
            return math.nan
        b_lo = float(floor_strike) - NWS_ROUND_HALF_F
        b_hi = float(cap_strike) + NWS_ROUND_HALF_F
        if forecast_mean_f <= b_lo:
            return b_lo - forecast_mean_f
        if forecast_mean_f >= b_hi:
            return forecast_mean_f - b_hi
        return 0.0  # mean inside YES rounding region

    if strike_type == "greater":
        if floor_strike is None:
            return math.nan
        b = float(floor_strike) + NWS_ROUND_HALF_F
        return abs(forecast_mean_f - b)

    if strike_type == "less":
        if cap_strike is None:
            return math.nan
        b = float(cap_strike) - NWS_ROUND_HALF_F
        return abs(forecast_mean_f - b)

    return math.nan
