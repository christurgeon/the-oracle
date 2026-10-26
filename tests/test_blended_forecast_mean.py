"""Unit tests for weather.blended_forecast_mean (boundary-filter mean source)."""

from __future__ import annotations

import math
from datetime import date

import pytest

from python.data.weather import EnsembleForecast, blended_forecast_mean


def _fc(temps: list[float]) -> EnsembleForecast:
    return EnsembleForecast(city="MIA", forecast_date=date(2026, 5, 31), temperatures=temps)


def test_single_model_returns_its_mean():
    fcs = {"gfs_seamless": _fc([70.0, 71.0, 72.0])}
    assert blended_forecast_mean(fcs, {"gfs_seamless": 1.0}) == pytest.approx(71.0)


def test_weight_renormalization_matches_probability_path():
    # gfs mean 70, ecmwf mean 80; weights 0.75/0.25 → 0.75*70 + 0.25*80 = 72.5
    fcs = {"gfs_seamless": _fc([70.0]), "ecmwf_ifs025": _fc([80.0])}
    out = blended_forecast_mean(fcs, {"gfs_seamless": 0.75, "ecmwf_ifs025": 0.25})
    assert out == pytest.approx(72.5)


def test_missing_model_weight_is_renormalized_away():
    # weights name a model not present → renormalize over present only.
    fcs = {"gfs_seamless": _fc([60.0]), "ecmwf_ifs025": _fc([70.0])}
    weights = {"gfs_seamless": 1.0, "ecmwf_ifs025": 1.0, "icon": 5.0}  # icon absent
    # equal weight over the two present → 65.0
    assert blended_forecast_mean(fcs, weights) == pytest.approx(65.0)


def test_zero_weight_falls_back_to_equal_weight():
    fcs = {"gfs_seamless": _fc([60.0]), "ecmwf_ifs025": _fc([70.0])}
    assert blended_forecast_mean(fcs, {"gfs_seamless": 0.0, "ecmwf_ifs025": 0.0}) == pytest.approx(
        65.0
    )


def test_weights_absent_entirely_falls_back_to_equal_weight():
    fcs = {"gfs_seamless": _fc([60.0]), "ecmwf_ifs025": _fc([70.0])}
    assert blended_forecast_mean(fcs, {}) == pytest.approx(65.0)


def test_empty_forecasts_returns_nan():
    assert math.isnan(blended_forecast_mean({}, {"gfs_seamless": 1.0}))
