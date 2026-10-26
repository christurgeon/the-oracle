"""Tests for python/data/asos_posterior.py — Task 1.1, 1.2 (Phase 1).

Run with: uv run pytest tests/test_asos_posterior.py -v
"""
from __future__ import annotations

from python.common import Direction
from python.data.asos_posterior import bracket_probability, prob_daily_max_at_least

# ---------------------------------------------------------------------------
# Task 1.1 — truncated-Student-t survival
# ---------------------------------------------------------------------------


def test_prob_at_or_below_M_is_one():  # noqa: N802
    p = prob_daily_max_at_least(s=80.0, M=85.0, mu_gain=2.0, sigma_gain=1.0, nu=5)
    assert p == 1.0


def test_prob_decreases_with_higher_strike():
    p_low = prob_daily_max_at_least(s=86.0, M=85.0, mu_gain=2.0, sigma_gain=1.0, nu=5)
    p_high = prob_daily_max_at_least(s=90.0, M=85.0, mu_gain=2.0, sigma_gain=1.0, nu=5)
    assert 0.0 < p_high < p_low < 1.0


def test_student_t_fatter_tail_than_gaussian():
    far = dict(s=92.0, M=85.0, mu_gain=2.0, sigma_gain=1.0)
    p_fat = prob_daily_max_at_least(**far, nu=4)
    p_thin = prob_daily_max_at_least(**far, nu=50)
    assert p_fat > p_thin


# ---------------------------------------------------------------------------
# Task 1.2 — bracket_probability over the NWS-rounding window
# ---------------------------------------------------------------------------

_KW = dict(M=85.0, mu_gain=2.0, sigma_gain=1.0, nu=5)


def test_between_uses_half_degree_window():
    p = bracket_probability(Direction.BETWEEN, floor=87.0, cap=88.0, **_KW)
    expect = prob_daily_max_at_least(s=86.5, **_KW) - prob_daily_max_at_least(s=88.5, **_KW)
    assert abs(p - expect) < 1e-9


def test_above_uses_floor_plus_half():
    p = bracket_probability(Direction.ABOVE, floor=87.0, cap=None, **_KW)
    assert abs(p - prob_daily_max_at_least(s=87.5, **_KW)) < 1e-9


def test_below_uses_cap_minus_half():
    p = bracket_probability(Direction.BELOW, floor=None, cap=90.0, **_KW)
    assert abs(p - (1.0 - prob_daily_max_at_least(s=89.5, **_KW))) < 1e-9


def test_probability_in_unit_interval():
    for d, f, c in [
        (Direction.BETWEEN, 86.0, 87.0),
        (Direction.ABOVE, 88.0, None),
        (Direction.BELOW, None, 84.0),
    ]:
        p = bracket_probability(d, floor=f, cap=c, **_KW)
        assert 0.0 <= p <= 1.0
