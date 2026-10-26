"""Unit tests for boundary_distance."""

import math

import pytest

from python.analytics.boundary_distance import boundary_distance

# ──────────────────────────────────────────────────────────────────
# BETWEEN
# ──────────────────────────────────────────────────────────────────


def test_between_mean_well_below_returns_distance_to_lower_boundary():
    # bracket [57, 58], mean = 55.0; lower rounding boundary = 56.5
    assert boundary_distance(55.0, 57, 58, "between") == pytest.approx(1.5)


def test_between_mean_well_above_returns_distance_to_upper_boundary():
    # bracket [57, 58], mean = 60.0; upper rounding boundary = 58.5
    assert boundary_distance(60.0, 57, 58, "between") == pytest.approx(1.5)


def test_between_mean_at_lower_boundary_returns_zero():
    # bracket [57, 58], mean exactly at floor - 0.5 = 56.5
    assert boundary_distance(56.5, 57, 58, "between") == pytest.approx(0.0)


def test_between_mean_inside_yes_rounding_region_returns_zero():
    # bracket [57, 58], mean = 57.4 (inside the YES-rounding region [56.5, 58.5))
    assert boundary_distance(57.4, 57, 58, "between") == pytest.approx(0.0)


def test_between_mean_just_outside_upper_boundary_returns_small_positive():
    # bracket [57, 58], mean = 58.6; distance to 58.5 = 0.1
    assert boundary_distance(58.6, 57, 58, "between") == pytest.approx(0.1)


# ──────────────────────────────────────────────────────────────────
# GREATER ("high > 53°F" → YES iff T ≥ floor + 0.5)
# ──────────────────────────────────────────────────────────────────


def test_greater_mean_above_boundary_returns_distance():
    # "high > 53", boundary = 53.5, mean = 56.0; distance = 2.5
    assert boundary_distance(56.0, 53, None, "greater") == pytest.approx(2.5)


def test_greater_mean_below_boundary_returns_distance():
    # "high > 53", boundary = 53.5, mean = 50.0; distance = 3.5
    assert boundary_distance(50.0, 53, None, "greater") == pytest.approx(3.5)


def test_greater_mean_at_boundary_returns_zero():
    # "high > 53", boundary = 53.5, mean = 53.5; distance = 0
    assert boundary_distance(53.5, 53, None, "greater") == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────
# LESS ("low < 64°F" → YES iff T < cap - 0.5)
# ──────────────────────────────────────────────────────────────────


def test_less_mean_below_boundary_returns_distance():
    # "low < 64", boundary = 63.5, mean = 60.0; distance = 3.5
    assert boundary_distance(60.0, None, 64, "less") == pytest.approx(3.5)


def test_less_mean_above_boundary_returns_distance():
    # "low < 64", boundary = 63.5, mean = 70.0; distance = 6.5
    assert boundary_distance(70.0, None, 64, "less") == pytest.approx(6.5)


# ──────────────────────────────────────────────────────────────────
# Half-integer strikes (defensive; Kalshi today uses integer strikes only,
# but the formula must continue to work if that changes)
# ──────────────────────────────────────────────────────────────────


def test_between_with_half_integer_strikes():
    # bracket [57.5, 58.5], boundaries at {57.0, 59.0}; mean 56.5 → dist = 0.5
    assert boundary_distance(56.5, 57.5, 58.5, "between") == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────
# Invalid input — returns NaN
# ──────────────────────────────────────────────────────────────────


def test_unknown_strike_type_returns_nan():
    result = boundary_distance(57.0, 57, 58, "weird")
    assert math.isnan(result)


def test_greater_without_floor_returns_nan():
    result = boundary_distance(57.0, None, 58, "greater")
    assert math.isnan(result)


def test_less_without_cap_returns_nan():
    result = boundary_distance(57.0, 57, None, "less")
    assert math.isnan(result)


def test_between_without_strikes_returns_nan():
    assert math.isnan(boundary_distance(57.0, None, 58, "between"))
    assert math.isnan(boundary_distance(57.0, 57, None, "between"))
