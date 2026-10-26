"""Tests for GFS bias correction (python/data/bias_correction.py)."""

from __future__ import annotations

import json
import math
import tempfile
from datetime import date
from pathlib import Path

import pytest

from python.data.bias_correction import (
    BiasEntry,
    BiasTable,
    apply_bias_correction,
)
from python.data.weather import EnsembleForecast

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_forecast(
    city: str = "NYC",
    forecast_date: date | None = None,
    temperatures: list[float] | None = None,
) -> EnsembleForecast:
    return EnsembleForecast(
        city=city,
        forecast_date=forecast_date or date(2026, 7, 15),
        temperatures=temperatures or [70.0 + i for i in range(31)],
    )


def _make_bias_json(
    city: str = "NYC",
    variable: str = "high",
    overall_bias: float = 1.5,
    month_biases: dict[str, float] | None = None,
    count: int = 60,
    std_bias: float = 2.0,
) -> dict:
    """Build a minimal bias correction JSON structure."""
    def _entry(bias: float) -> dict:
        return {
            "count": count,
            "mean_bias_f": bias,
            "mae_f": abs(bias) + 0.5,
            "rmse_f": abs(bias) + 1.0,
            "std_bias_f": std_bias,
        }

    by_month = {}
    if month_biases:
        for m, b in month_biases.items():
            by_month[m] = _entry(b)

    return {
        city: {
            variable: {
                "overall": _entry(overall_bias),
                "by_month": by_month,
            }
        }
    }


def _write_bias_table(data: dict) -> BiasTable:
    """Write bias JSON to a temp file and load a BiasTable from it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    return BiasTable(path=path)


# ── BiasEntry ────────────────────────────────────────────────────────────────


class TestBiasEntry:
    def test_snr_calculation(self):
        entry = BiasEntry(count=60, mean_bias_f=2.0, mae_f=2.5, rmse_f=3.0, std_bias_f=2.0)
        expected_se = 2.0 / math.sqrt(60)
        expected_snr = 2.0 / expected_se
        assert abs(entry.snr - expected_snr) < 0.01

    def test_snr_zero_std(self):
        entry = BiasEntry(count=60, mean_bias_f=2.0, mae_f=2.5, rmse_f=3.0, std_bias_f=0.0)
        assert entry.snr == 0.0

    def test_snr_zero_count(self):
        entry = BiasEntry(count=0, mean_bias_f=2.0, mae_f=2.5, rmse_f=3.0, std_bias_f=2.0)
        assert entry.snr == 0.0


# ── BiasTable ────────────────────────────────────────────────────────────────


class TestBiasTable:
    def test_loads_valid_json(self):
        data = _make_bias_json("NYC", "high", overall_bias=1.0)
        table = _write_bias_table(data)
        assert table.loaded

    def test_missing_file_returns_empty(self):
        table = BiasTable(path=Path("/nonexistent/path.json"))
        assert not table.loaded

    def test_get_bias_monthly(self):
        data = _make_bias_json("NYC", "high", overall_bias=1.0, month_biases={"7": 2.5})
        table = _write_bias_table(data)
        entry = table.get_bias("NYC", "high", month=7)
        assert entry is not None
        assert entry.mean_bias_f == 2.5

    def test_get_bias_falls_back_to_overall(self):
        data = _make_bias_json("NYC", "high", overall_bias=1.0)
        table = _write_bias_table(data)
        # Month 3 not in by_month, should fall back to overall
        entry = table.get_bias("NYC", "high", month=3)
        assert entry is not None
        assert entry.mean_bias_f == 1.0

    def test_get_bias_missing_city(self):
        data = _make_bias_json("NYC", "high", overall_bias=1.0)
        table = _write_bias_table(data)
        assert table.get_bias("LAX", "high", month=1) is None

    def test_get_bias_missing_variable(self):
        data = _make_bias_json("NYC", "high", overall_bias=1.0)
        table = _write_bias_table(data)
        assert table.get_bias("NYC", "low", month=1) is None

    def test_daily_variable_string_mapping(self):
        """DailyVariable enum value strings map to the correct bias key."""
        data = _make_bias_json("NYC", "high", overall_bias=1.5)
        table = _write_bias_table(data)
        # Using the string value of DailyVariable.TEMP_MAX
        entry = table.get_bias("NYC", "temperature_2m_max", month=1)
        assert entry is not None
        assert entry.mean_bias_f == 1.5

    def test_daily_variable_low_mapping(self):
        data = _make_bias_json("MIA", "low", overall_bias=2.0)
        table = _write_bias_table(data)
        entry = table.get_bias("MIA", "temperature_2m_min", month=7)
        assert entry is not None
        assert entry.mean_bias_f == 2.0


# ── apply_bias_correction ────────────────────────────────────────────────────


class TestApplyBiasCorrection:
    def test_positive_bias_shifts_temperatures_down(self):
        """GFS warm bias (+2°F) → correction subtracts from temperatures."""
        data = _make_bias_json("NYC", "high", overall_bias=2.0, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 3, 15))

        corrected, correction_f = apply_bias_correction(
            forecast, table, "high", shrinkage=0.5,
        )

        assert correction_f == pytest.approx(-1.0)  # -2.0 * 0.5
        for orig, corr in zip(forecast.temperatures, corrected.temperatures):
            assert corr == pytest.approx(orig - 1.0)

    def test_negative_bias_shifts_temperatures_up(self):
        """GFS cool bias (-1.8°F) → correction adds to temperatures."""
        data = _make_bias_json("DEN", "low", overall_bias=-1.8, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("DEN", date(2026, 12, 15))

        corrected, correction_f = apply_bias_correction(
            forecast, table, "low", shrinkage=0.5,
        )

        assert correction_f == pytest.approx(0.9)  # -(-1.8) * 0.5
        for orig, corr in zip(forecast.temperatures, corrected.temperatures):
            assert corr == pytest.approx(orig + 0.9)

    def test_shrinkage_applied(self):
        data = _make_bias_json("NYC", "high", overall_bias=4.0, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 1, 15))

        corrected_30, corr_30 = apply_bias_correction(
            forecast, table, "high", shrinkage=0.3,
        )
        corrected_50, corr_50 = apply_bias_correction(
            forecast, table, "high", shrinkage=0.5,
        )

        assert corr_30 == pytest.approx(-1.2)  # -4.0 * 0.3
        assert corr_50 == pytest.approx(-2.0)  # -4.0 * 0.5

    def test_cap_enforced(self):
        data = _make_bias_json("NYC", "high", overall_bias=20.0, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 1, 15))

        _, correction_f = apply_bias_correction(
            forecast, table, "high", shrinkage=1.0, max_correction_f=5.0,
        )

        assert abs(correction_f) <= 5.0

    def test_missing_city_returns_original(self):
        data = _make_bias_json("NYC", "high", overall_bias=2.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("LAX", date(2026, 1, 15))

        corrected, correction_f = apply_bias_correction(
            forecast, table, "high",
        )

        assert correction_f == 0.0
        assert corrected.temperatures == forecast.temperatures

    def test_snr_gate_skips_insignificant_bias(self):
        """Bias with low SNR should not be applied."""
        # With count=60, std=2.0, SE = 2.0/sqrt(60) ≈ 0.258
        # mean_bias=0.2 → SNR = 0.2/0.258 ≈ 0.77 < 2.0 → skip
        data = _make_bias_json("LAX", "high", overall_bias=0.2, std_bias=2.0, count=60)
        table = _write_bias_table(data)
        forecast = _make_forecast("LAX", date(2026, 6, 15))

        corrected, correction_f = apply_bias_correction(
            forecast, table, "high",
        )

        assert correction_f == 0.0
        assert corrected.temperatures == forecast.temperatures

    def test_snr_gate_keeps_significant_bias(self):
        """Bias with high SNR should be applied."""
        # With count=60, std=2.0, SE = 0.258
        # mean_bias=2.0 → SNR = 2.0/0.258 ≈ 7.75 > 2.0 → apply
        data = _make_bias_json("MIA", "low", overall_bias=2.0, std_bias=2.0, count=60)
        table = _write_bias_table(data)
        forecast = _make_forecast("MIA", date(2026, 7, 15))

        _, correction_f = apply_bias_correction(
            forecast, table, "low", shrinkage=0.3,
        )

        assert correction_f == pytest.approx(-0.6)  # -2.0 * 0.3

    def test_preserves_ensemble_spread(self):
        """Correction shifts mean but preserves std dev."""
        data = _make_bias_json("NYC", "high", overall_bias=3.0, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 2, 15))

        corrected, _ = apply_bias_correction(
            forecast, table, "high", shrinkage=0.5,
        )

        assert corrected.std_dev == pytest.approx(forecast.std_dev, abs=0.001)

    def test_probability_shifts_expected_direction(self):
        """Warm bias correction should reduce fraction_above for the same threshold."""
        # GFS is +3°F warm for NYC HIGH → after correction, fewer members exceed threshold
        data = _make_bias_json("NYC", "high", overall_bias=3.0, std_bias=1.0)
        table = _write_bias_table(data)
        temps = [float(t) for t in range(65, 96)]  # 65-95°F, 31 members
        forecast = _make_forecast("NYC", date(2026, 7, 15), temperatures=temps)

        corrected, _ = apply_bias_correction(
            forecast, table, "high", shrinkage=0.5,
        )

        threshold = 80.0
        raw_above = forecast.fraction_above(threshold)
        corrected_above = corrected.fraction_above(threshold)
        # Warm bias correction → fewer members above threshold
        assert corrected_above < raw_above

    def test_probability_shifts_for_cool_bias(self):
        """Cool bias correction should increase fraction_above for the same threshold."""
        data = _make_bias_json("DEN", "low", overall_bias=-3.0, std_bias=1.0)
        table = _write_bias_table(data)
        temps = [float(t) for t in range(20, 51)]  # 20-50°F, 31 members
        forecast = _make_forecast("DEN", date(2026, 12, 15), temperatures=temps)

        corrected, _ = apply_bias_correction(
            forecast, table, "low", shrinkage=0.5,
        )

        threshold = 35.0
        raw_above = forecast.fraction_above(threshold)
        corrected_above = corrected.fraction_above(threshold)
        # Cool bias correction → more members above threshold
        assert corrected_above > raw_above

    def test_frozen_forecast_not_mutated(self):
        """Original forecast remains unchanged after correction."""
        data = _make_bias_json("NYC", "high", overall_bias=2.0, std_bias=1.0)
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 1, 15))
        original_temps = list(forecast.temperatures)

        apply_bias_correction(forecast, table, "high", shrinkage=0.5)

        assert forecast.temperatures == original_temps

    def test_uses_monthly_bias_over_overall(self):
        """Monthly bias should be used when available, not overall."""
        data = _make_bias_json(
            "NYC", "high",
            overall_bias=1.0,
            month_biases={"7": 3.0},
            std_bias=1.0,
        )
        table = _write_bias_table(data)
        forecast = _make_forecast("NYC", date(2026, 7, 15))

        _, correction_f = apply_bias_correction(
            forecast, table, "high", shrinkage=0.5,
        )

        # Should use month 7 bias (3.0), not overall (1.0)
        assert correction_f == pytest.approx(-1.5)  # -3.0 * 0.5


# ── apply_bias_correction immutability invariant ─────────────────────────────


class TestApplyBiasCorrectionImmutability:
    """Pin the contract: apply_bias_correction returns a new EnsembleForecast
    whose temperatures is a freshly-constructed list. EMOS shadow's
    corrected-members reconstruction (python/data/emos.py) silently relies
    on this invariant.
    """

    def test_returned_temperatures_is_a_new_list_object(self):
        """Returned forecast.temperatures must not be the same list object as
        the input forecast.temperatures (no aliasing)."""
        # Use a high-SNR bias so a correction is definitely applied
        # (SE = 1.0/sqrt(60) ≈ 0.129, SNR = 2.0/0.129 ≈ 15.5 >> 2.0)
        data = _make_bias_json(
            "NYC", "high", overall_bias=2.0, std_bias=1.0, count=60,
        )
        table = _write_bias_table(data)
        fc = _make_forecast("NYC", date(2026, 5, 5))

        corrected, correction_f = apply_bias_correction(
            fc, table, "high", shrinkage=0.3,
        )

        # Sanity: ensure correction was actually applied (not the no-op path)
        assert correction_f != 0.0, "test setup error: correction was skipped"
        assert corrected.temperatures is not fc.temperatures, (
            "apply_bias_correction must return a new list (not alias the input)"
        )

    def test_input_temperatures_unchanged_after_call(self):
        """Input forecast.temperatures must equal its pre-call snapshot after
        apply_bias_correction returns (no mutation)."""
        data = _make_bias_json(
            "NYC", "high", overall_bias=2.0, std_bias=1.0, count=60,
        )
        table = _write_bias_table(data)
        fc = _make_forecast("NYC", date(2026, 5, 5))
        snapshot_pre = list(fc.temperatures)  # capture pre-call

        corrected, correction_f = apply_bias_correction(
            fc, table, "high", shrinkage=0.3,
        )

        # Sanity: ensure correction was actually applied (not the no-op path)
        assert correction_f != 0.0, "test setup error: correction was skipped"
        assert fc.temperatures == snapshot_pre, (
            "apply_bias_correction must not mutate the input forecast's "
            "temperatures"
        )
