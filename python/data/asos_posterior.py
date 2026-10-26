"""Hermetic ASOS intraday posterior math.

A daily HIGH is the max of the diurnal curve. Given observed max-so-far ``M``
and a climatological remaining-gain distribution (truncated Student-t at 0,
since gain >= 0), produce P(daily_max in bracket). No I/O; imports limited to
stdlib + scipy + the ``python.common`` leaf enums (same spirit as emos.py: no
runtime-heavy or I/O deps).
"""

from __future__ import annotations

from scipy import stats

from python.common import Direction

NWS_ROUND_HALF_F = 0.5  # mirror python/common.py


def prob_daily_max_at_least(
    s: float,
    M: float,  # noqa: N803
    mu_gain: float,
    sigma_gain: float,
    nu: float,
) -> float:
    """P(daily_max >= s) under a left-truncated (at 0) location-scale Student-t
    remaining-gain posterior. daily_max = M + gain, gain ~ t_nu(mu_gain,
    sigma_gain) truncated to gain >= 0.
    """
    if sigma_gain <= 0:
        return 1.0 if s <= M + max(mu_gain, 0.0) else 0.0
    if s <= M:
        return 1.0
    sf = stats.t.sf
    z_s = (s - M - mu_gain) / sigma_gain
    z_0 = (0.0 - mu_gain) / sigma_gain
    denom = sf(z_0, df=nu)
    if denom <= 0:
        return 0.0
    return float(sf(z_s, df=nu) / denom)


def bracket_probability(
    strike_type: Direction,
    *,
    floor: float | None,
    cap: float | None,
    M: float,  # noqa: N803
    mu_gain: float,
    sigma_gain: float,
    nu: float,
) -> float:
    """P(bracket settles YES) for a HIGH market, over the NWS integer-rounding
    window — mirrors the logic of python/data/weather.py _apply_nws_window.
    """
    h = NWS_ROUND_HALF_F

    def pg(s: float) -> float:
        return prob_daily_max_at_least(s=s, M=M, mu_gain=mu_gain, sigma_gain=sigma_gain, nu=nu)

    if strike_type == Direction.ABOVE:
        if floor is None:
            raise ValueError("floor required for ABOVE")
        return pg(floor + h)
    if strike_type == Direction.BELOW:
        if cap is None:
            raise ValueError("cap required for BELOW")
        return 1.0 - pg(cap - h)
    if strike_type == Direction.BETWEEN:
        if floor is None or cap is None:
            raise ValueError("floor and cap required for BETWEEN")
        return max(0.0, pg(floor - h) - pg(cap + h))
    raise ValueError(f"unknown strike_type {strike_type!r}")
