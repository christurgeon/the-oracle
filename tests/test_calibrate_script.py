"""Regression tests for `scripts/calibrate.py`.

Locks in the producer/consumer shape contract for `forecast_lookup`:
- Key is a 5-tuple `(snapshot_date, city, forecast_date, daily_variable, model)`.
- Consumer site filters to `model="gfs_seamless"` (production runtime is GFS-only).
- Missing `model` field on any forecast record fails fast.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.calibrate import build_forecast_outcome_pairs


def _make_market() -> dict:
    return {
        "ticker": "KXHIGHNY-26MAY01-T70",
        "series_ticker": "KXHIGHNY",
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "strike_type": "greater",
        "floor_strike": 70.0,
        "cap_strike": None,
        "yes_bid_dollars": 0.40,
        "yes_ask_dollars": 0.45,
        "expiration_ts": int(datetime(2026, 5, 1, 18, 0, 0, tzinfo=UTC).timestamp()),
        "_snapshot_date": "2026-04-30",
    }


def _make_settlement() -> dict:
    return {
        "city": "NYC",
        "observation_date": "2026-05-01",
        "high_f": 75.0,
        "low_f": 60.0,
        "is_preliminary": False,
        "issued_at": "2026-05-02T06:00:00+00:00",
        "product_id": "test-product",
    }


def _make_forecast(model: str | None, temperatures: list[float]) -> dict:
    fc: dict = {
        "city": "NYC",
        "forecast_date": "2026-05-01",
        "daily_variable": "temperature_2m_max",
        "temperatures": temperatures,
        "_snapshot_date": "2026-04-30",
    }
    if model is not None:
        fc["model"] = model
    return fc


def test_gfs_selected_when_both_models_present_at_same_key() -> None:
    """Regression: both gfs_seamless and ecmwf_ifs025 at the same
    (snapshot_date, city, forecast_date, daily_variable) — GFS must win.
    Pre-fix this was last-writer-wins (ECMWF) because the lookup key omitted model.
    """
    gfs_temps = [72.0] * 30  # all members > 70 → model_prob = 1.0
    ecmwf_temps = [60.0] * 50  # all members < 70 → model_prob = 0.0

    forecasts = [
        _make_forecast("gfs_seamless", gfs_temps),
        _make_forecast("ecmwf_ifs025", ecmwf_temps),
    ]
    markets = [_make_market()]
    settlements = [_make_settlement()]

    pairs, _stats = build_forecast_outcome_pairs(markets, forecasts, settlements)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.ensemble_temperatures == gfs_temps
    assert pair.model_prob == 1.0  # would be 0.0 under the old bug


def test_missing_model_field_raises_value_error() -> None:
    """Forecasts must declare their model. No silent default — schema regression
    must fail loudly per `pipeline_implicit_conventions_audit` rule.
    """
    bad_forecast = _make_forecast(None, [70.0] * 30)
    markets = [_make_market()]
    settlements = [_make_settlement()]

    with pytest.raises(ValueError, match="missing required 'model' field"):
        build_forecast_outcome_pairs(markets, [bad_forecast], settlements)


def test_compute_model_prob_uses_nws_rounded_window() -> None:
    """Regression: scripts/calibrate.py must use ±0.5°F NWS-rounded bracket
    math (compute_blended_probability), not raw ensemble.fraction_above.

    Setup: 26 members evenly split — 13 at 70.2°F, 13 at 71.0°F. Floor = 70.0.
    Member count must clear the script's skipped_low_members gate (≥25,
    scripts/calibrate.py:412).
    - Raw fraction_above(70.0) → 26/26 = 1.0 (both groups exceed 70.0).
    - NWS-shifted Direction.ABOVE uses T >= 70.5 → 13/26 = 0.5
      (the 70.2 group fails the shifted threshold).
    Pre-2026-05-22 calibrate.py returned 1.0; post-fix returns 0.5.
    """
    market = _make_market()  # floor_strike = 70.0, strike_type = "greater"
    forecasts = [_make_forecast("gfs_seamless", [70.2] * 13 + [71.0] * 13)]
    settlements = [_make_settlement()]

    pairs, _stats = build_forecast_outcome_pairs([market], forecasts, settlements)

    assert len(pairs) == 1
    assert pairs[0].model_prob == pytest.approx(0.5, abs=0.01)
