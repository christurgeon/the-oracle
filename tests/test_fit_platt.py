"""Fitter tests for the per-type Platt calibration artifact.

End-to-end: synthesize a minimal set of snapshots (forecasts +
markets + settlements), invoke the fitter in-process, and verify the
produced artifacts load cleanly under :class:`PerTypePlattCalibrator`
and :class:`PlattCalibrator`.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

pytest.importorskip("sklearn", reason="ML extras not installed (uv sync --extra ml)")
pytest.importorskip("numpy", reason="numpy not installed")

import numpy as np  # noqa: E402

from python.data.calibration import (  # noqa: E402
    PerTypePlattCalibrator,
    PlattCalibrator,
)
from scripts._calibration_common import build_pairs, load_snapshot_data  # noqa: E402
from scripts.fit_platt import (  # noqa: E402
    _fit_per_type,
    _fit_platt,
    _pairs_to_arrays,
    _sigmoid_predict,
)

# ---------------------------------------------------------------------------
# Synthetic snapshot builder
# ---------------------------------------------------------------------------


def _write_day_snapshot(
    root: Path,
    day: date,
    *,
    city: str,
    variable: str,
    ensemble: list[float],
    actual_temp: float,
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
    ticker_suffix: str,
) -> None:
    """Write one day's forecasts/markets/settlements matching the loader schema."""
    d = root / day.isoformat()
    d.mkdir(parents=True, exist_ok=True)

    (d / "weather_forecasts.json").write_text(
        json.dumps(
            [
                {
                    "city": city,
                    "forecast_date": day.isoformat(),
                    "daily_variable": variable,
                    "model": "gfs_seamless",
                    "temperatures": ensemble,
                }
            ]
        )
    )

    # Ticker format that _parse_market_date understands: KXHIGHNY-26APR14-<suffix>
    year_str = f"{day.year % 100:02d}"
    month_str = day.strftime("%b").upper()[:3]
    day_str = f"{day.day:02d}"
    ticker = f"KXHIGHNY-{year_str}{month_str}{day_str}-{ticker_suffix}"
    (d / "markets.json").write_text(
        json.dumps(
            [
                {
                    "ticker": ticker,
                    "city": city,
                    "daily_variable": variable,
                    "strike_type": strike_type,
                    "floor_strike": floor_strike,
                    "cap_strike": cap_strike,
                }
            ]
        )
    )

    settle_field = "high_f" if variable == "temperature_2m_max" else "low_f"
    (d / "settlements.json").write_text(
        json.dumps(
            [
                {
                    "city": city,
                    "observation_date": day.isoformat(),
                    "is_preliminary": False,
                    settle_field: actual_temp,
                }
            ]
        )
    )


def _seed_snapshots(root: Path, n_days: int = 250, seed: int = 7) -> None:
    """Build a multi-day synthetic snapshot dir with realistic label structure."""
    rng = random.Random(seed)
    start = date(2026, 5, 1)
    # 25 ensemble members (above the ``len(members) >= 25`` gate).
    for i in range(n_days):
        day = start + timedelta(days=i)
        mean_temp = 75.0 + rng.uniform(-8.0, 8.0)
        members = [mean_temp + rng.gauss(0.0, 3.0) for _ in range(25)]
        actual = mean_temp + rng.gauss(0.0, 2.5)

        # Rotate through the three strike types so each is well-represented.
        rot = i % 3
        if rot == 0:
            strike_type = "greater"
            floor_strike = mean_temp + rng.uniform(-5, 5)
            cap_strike = None
            suffix = "greater"
        elif rot == 1:
            strike_type = "less"
            floor_strike = None
            cap_strike = mean_temp + rng.uniform(-5, 5)
            suffix = "less"
        else:
            strike_type = "between"
            lo = mean_temp - 4.0
            hi = mean_temp + 4.0
            floor_strike = lo
            cap_strike = hi
            suffix = "between"

        _write_day_snapshot(
            root,
            day,
            city="NYC",
            variable="temperature_2m_max",
            ensemble=members,
            actual_temp=actual,
            strike_type=strike_type,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            ticker_suffix=suffix,
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSigmoidPredict:
    def test_zero_input(self) -> None:
        out = _sigmoid_predict(np.array([0.5]), 0.0, 0.0)
        assert out[0] == pytest.approx(0.5)

    def test_large_positive_does_not_overflow(self) -> None:
        # Stability: z=1000 must map to ~1 without an overflow warning.
        out = _sigmoid_predict(np.array([1.0]), 999.0, 1.0)
        assert 0.99 < out[0] <= 1.0

    def test_large_negative_does_not_overflow(self) -> None:
        out = _sigmoid_predict(np.array([0.0]), -1000.0, 1.0)
        assert 0.0 <= out[0] < 0.01

    def test_matches_runtime_calibrator(self, tmp_path: Path) -> None:
        # Runtime _apply_platt and training _sigmoid_predict must agree.
        from python.data.calibration import _apply_platt

        for raw in (0.0, 0.25, 0.5, 0.75, 1.0):
            fit_out = _sigmoid_predict(np.array([raw]), -0.5, 2.0)[0]
            runtime_out = _apply_platt(raw, -0.5, 2.0)
            # _apply_platt clamps to [1e-6, 1-1e-6]; loosen tolerance
            # enough that clamping doesn't make us fail at extreme raw.
            assert abs(float(fit_out) - runtime_out) < 1e-4


class TestFitPlattDegenerate:
    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError, match="single-class"):
            _fit_platt(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            _fit_platt(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# End-to-end: synthetic snapshots → fit → load via runtime calibrator
# ---------------------------------------------------------------------------


class TestFitPlattEndToEnd:
    @pytest.fixture
    def fit_inputs(self, tmp_path: Path):
        """Build snapshots and return (pairs, raw, y, strike_types)."""
        snapshots = tmp_path / "snapshots"
        _seed_snapshots(snapshots, n_days=330)  # 110 per strike type
        forecasts, settlements, markets = load_snapshot_data(snapshots)
        pairs = build_pairs(
            forecasts,
            settlements,
            markets,
            use_spread_inflation=False,
            spread_inflation_factor_fn=None,
        )
        assert len(pairs) > 200, f"only {len(pairs)} pairs — fixture too small"
        raw, y, strike_types = _pairs_to_arrays(pairs)
        return pairs, raw, y, strike_types

    def test_global_fit_produces_loadable_artifact(self, fit_inputs, tmp_path: Path) -> None:
        _, raw, y, _ = fit_inputs
        a, b = _fit_platt(raw, y)
        # Coefficients must be finite for the runtime loader to accept them.
        assert np.isfinite(a) and np.isfinite(b)

        payload = {
            "schema_version": 1,
            "fit_date": "2026-04-21",
            "a": round(a, 6),
            "b": round(b, 6),
            "n_pairs": int(raw.size),
        }
        out = tmp_path / "platt.json"
        out.write_text(json.dumps(payload))

        cal = PlattCalibrator(path=out)
        assert cal.is_loaded
        assert cal.fallback_reason is None
        # Runtime output matches a hand-computed sigmoid within the
        # clamp epsilon.
        from python.data.calibration import _apply_platt

        expected = _apply_platt(0.42, a, b)
        assert cal.calibrate(0.42) == pytest.approx(expected)

    def test_per_type_fit_produces_loadable_artifact(self, fit_inputs, tmp_path: Path) -> None:
        _, raw, y, strike_types = fit_inputs
        per_type, warnings = _fit_per_type(
            raw,
            y,
            strike_types,
            min_per_type=50,
            cross_validate=False,
        )
        # Exactly the three canonical strike types.
        assert set(per_type.keys()) == {"greater", "less", "between"}
        for stype, entry in per_type.items():
            assert np.isfinite(entry["a"]), f"{stype}: non-finite a"
            assert np.isfinite(entry["b"]), f"{stype}: non-finite b"
            assert entry["n"] > 0

        payload = {
            "schema_version": 2,
            "fit_date": "2026-04-21",
            "per_type_warnings": warnings,
            "per_type": per_type,
        }
        out = tmp_path / "platt_per_strike_type.json"
        out.write_text(json.dumps(payload))

        cal = PerTypePlattCalibrator(path=out)
        assert cal.is_loaded, f"failed to load: {cal.fallback_reason}"
        # All three strike types produce finite probabilities.
        for stype in ("greater", "less", "between"):
            p = cal.calibrate(0.4, strike_type=stype)
            assert 0.0 < p < 1.0

    def test_legacy_forecast_without_model_field(self, tmp_path: Path) -> None:
        """load_snapshot_data must accept legacy records lacking a 'model' key.

        Such records existed before the 'model' field was added to the snapshot
        schema; they should be stored under the default key "gfs_seamless" so
        downstream fitters include them without any special-casing.
        """
        snapshots = tmp_path / "snapshots"
        day = date(2026, 3, 14)
        d = snapshots / day.isoformat()
        d.mkdir(parents=True, exist_ok=True)

        # Legacy record: no "model" key at all.
        (d / "weather_forecasts.json").write_text(
            json.dumps(
                [
                    {
                        "city": "NYC",
                        "forecast_date": day.isoformat(),
                        "daily_variable": "temperature_2m_max",
                        # deliberately omitting "model"
                        "temperatures": [72.0] * 31,
                    }
                ]
            )
        )
        (d / "markets.json").write_text(json.dumps([]))
        (d / "settlements.json").write_text(json.dumps([]))

        forecasts, _, _ = load_snapshot_data(snapshots)

        assert ("NYC", "2026-03-14", "temperature_2m_max", "gfs_seamless") in forecasts, (
            "Legacy forecast record (no 'model' field) must be keyed under 'gfs_seamless'"
        )

    def test_artifact_rejected_by_runtime_if_key_renamed(self, fit_inputs, tmp_path: Path) -> None:
        """B1 regression: renaming a per-type key must trigger identity fallback,
        never silent acceptance."""
        _, raw, y, strike_types = fit_inputs
        per_type, _ = _fit_per_type(raw, y, strike_types, min_per_type=50, cross_validate=False)
        # Rename "greater" to "above" — matches the Direction enum value but
        # is the wrong canonical label for this pipeline.
        per_type["above"] = per_type.pop("greater")
        payload = {"schema_version": 2, "per_type": per_type}

        out = tmp_path / "pt.json"
        out.write_text(json.dumps(payload))

        cal = PerTypePlattCalibrator(path=out)
        assert not cal.is_loaded
        assert "strike-type-keys" in (cal.fallback_reason or "")


# ---------------------------------------------------------------------------
# Preflight-wiring tests (Task 4)
#
# Mirror the T3a–T3i tests in tests/test_fit_isotonic.py. These exercise the
# input-contract preflight + abort policy + flag plumbing introduced when
# ``fit_platt.py`` was wired to ``scripts/_calibration_input_contract``. The
# heavy LogisticRegression fit is **never** invoked — every test stubs the
# production-mirrored pair-builder so that ``main()`` either aborts during
# the preflight or exits at the existing ``len(pairs) < 100`` floor.
#
# Helpers are private (leading underscore) — pytest does not collect them.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _empty_snapshot_dir(tmp_path: Path) -> Path:
    """Build an empty (but well-formed) snapshots tree."""
    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))
    return tmp_path / "snapshots"


def _write_calendar(path: Path, models: dict[str, str] | None = None) -> Path:
    """Write a calendar JSON to ``path`` and return it."""
    if models is None:
        models = {"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00+00:00",
                "method": "auto",
                "models": {name: {"first_seen_date": d} for name, d in models.items()},
            }
        )
    )
    return path


def _write_bias(path: Path) -> Path:
    """Write a minimal but ``loaded == True`` bias table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # BiasTable.loaded == bool(self._data); a non-empty dict suffices.
    path.write_text(json.dumps({"NYC": {"high": {"by_month": {}}}}))
    return path


def _make_report(
    *,
    total: int = 200,
    full: int = 200,
    partial: int = 0,
    empty: int = 0,
    single_model_era: int = 0,
    exempt: int = 0,
):
    from scripts._calibration_input_contract import InputContractReport

    return InputContractReport(
        n_triples_total=total,
        n_triples_full=full,
        n_triples_partial=partial,
        n_triples_empty=empty,
        n_triples_single_model_era=single_model_era,
        n_triples_exempt=exempt,
        n_settlements_final=total,
        n_settlements_dropped_preliminary=0,
    )


def _stub_pair_builder(monkeypatch, pairs=None) -> dict:
    """Stub ``build_production_mirrored_pairs`` to yield the given pairs."""
    import scripts._calibration_common as cc

    called = {"helper": False, "kwargs": None}

    def fake_helper(**kwargs):
        called["helper"] = True
        called["kwargs"] = kwargs
        return iter(pairs or [])

    monkeypatch.setattr(cc, "build_production_mirrored_pairs", fake_helper)
    return called


def _stub_preflight(monkeypatch, report) -> dict:
    """Stub ``assert_snapshot_input_contract`` to return ``report``."""
    import scripts._calibration_input_contract as ic

    called = {"preflight": False, "kwargs": None}

    def fake(*args, **kwargs):
        called["preflight"] = True
        called["kwargs"] = kwargs
        return report

    monkeypatch.setattr(ic, "assert_snapshot_input_contract", fake)
    return called


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fit_platt.py", *args])


# ---------------------------------------------------------------------------
# T4a — calendar missing
# ---------------------------------------------------------------------------


def test_t4a_calendar_missing_raises(tmp_path: Path, monkeypatch) -> None:
    """If the data-availability calendar is missing, abort with a clear message."""
    import scripts._calibration_common as _cc
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_cc, "_PROJECT_ROOT", tmp_path)
    # Calendar deliberately not written.

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="calendar"):
        fit_pl.main()


# ---------------------------------------------------------------------------
# T4b — partial-coverage abort + flag honor
# ---------------------------------------------------------------------------


def test_t4b_partial_coverage_aborts_without_flag(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, partial=10))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--allow-partial-model-coverage"):
        fit_pl.main()


def test_t4b_partial_coverage_flag_proceeds(tmp_path: Path, monkeypatch) -> None:
    """With --allow-partial-model-coverage the fitter proceeds past preflight."""
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, partial=10))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--allow-partial-model-coverage",
    )

    # The empty pair-builder stub yields zero pairs; the existing
    # ``len(pairs) < 100`` floor calls ``sys.exit(1)`` afterwards.
    with pytest.raises(SystemExit):
        fit_pl.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T4c — empty-cells abort + flag honor
# ---------------------------------------------------------------------------


def test_t4c_empty_cells_aborts_without_flag(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, empty=10))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--allow-empty-cells"):
        fit_pl.main()


def test_t4c_empty_cells_flag_proceeds(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, empty=10))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--allow-empty-cells",
    )

    with pytest.raises(SystemExit):
        fit_pl.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T4d — bias-table missing without/with override
# ---------------------------------------------------------------------------


def test_t4d_missing_bias_table_aborts_without_override(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    bias_path = tmp_path / "missing_bias.json"  # NOT written

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias_path),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--no-bias-correction"):
        fit_pl.main()


def test_t4d_missing_bias_table_no_bias_correction_proceeds_and_records(
    tmp_path: Path, monkeypatch
) -> None:
    """--no-bias-correction lets the fitter proceed; metadata records the override."""
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    bias_path = tmp_path / "missing_bias.json"  # NOT written

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    # Stub the pair-builder to return a balanced corpus across all three
    # strike types so both the global fit and the per-type fit converge.
    # Each strike type needs both classes (0 and 1) present.
    pairs = []
    for st in ("greater", "less", "between"):
        # 60 negatives at low raw, 60 positives at high raw — well-separated.
        for _ in range(60):
            pairs.append((0.3, 0, "NYC", "temperature_2m_max", st, "2026-04-15"))
        for _ in range(60):
            pairs.append((0.7, 1, "NYC", "temperature_2m_max", st, "2026-04-15"))
    _stub_pair_builder(monkeypatch, pairs=pairs)

    out_global = tmp_path / "platt.json"
    out_per_type = tmp_path / "platt_pt.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(out_global),
        "--output-per-type",
        str(out_per_type),
        "--bias-table",
        str(bias_path),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--min-per-type",
        "10",
    )

    fit_pl.main()
    artifact = json.loads(out_global.read_text())
    assert artifact["bias_correction_applied"] is False
    artifact_pt = json.loads(out_per_type.read_text())
    assert artifact_pt["bias_correction_applied"] is False


# ---------------------------------------------------------------------------
# T4e — config-load failure / explicit --ensemble-weights bypass
# ---------------------------------------------------------------------------


def test_t4e_missing_ensemble_weights_in_config_aborts(tmp_path: Path, monkeypatch) -> None:
    """If config has no strategies.weather.ensemble_weights and no CLI override → abort."""
    import scripts._calibration_common as _cc
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")

    # Synthesise a config.yaml at the test's _PROJECT_ROOT that lacks the key.
    cfg_dir = tmp_path / "python"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text("strategies:\n  weather:\n    other_key: 1\n")

    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_cc, "_PROJECT_ROOT", tmp_path)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
    )

    with pytest.raises(
        CalibratorInputContractError,
        match="strategies.weather.ensemble_weights",
    ):
        fit_pl.main()


def test_t4e_explicit_ensemble_weights_bypass_config(tmp_path: Path, monkeypatch) -> None:
    """--ensemble-weights overrides the config-load and proceeds."""
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")

    # Config still missing the key — the CLI arg should win.
    cfg_dir = tmp_path / "python"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text("strategies:\n  weather:\n    other_key: 1\n")

    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(SystemExit):
        fit_pl.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T4f — --write-pair-provenance happy path (post-Task 5)
# ---------------------------------------------------------------------------


def test_t4f_write_pair_provenance_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    """Post-Task 5 the sibling generator exists; --write-pair-provenance writes
    one JSONL row per yielded pair with schema_version=1 and the expected fields.
    """
    import scripts._calibration_common as cc
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    # Sibling generator must exist after Task 5.
    assert hasattr(cc, "build_production_mirrored_pairs_with_provenance")

    # Stub the sibling generator with a balanced corpus across all three
    # strike types so the per-type fit converges (each type needs both classes).
    sibling_pairs = []
    for st in ("greater", "less", "between"):
        for i in range(60):
            sibling_pairs.append(
                (
                    0.3,
                    0,
                    "NYC",
                    "temperature_2m_max",
                    st,
                    "2026-04-15",
                    ("gfs_seamless",),
                    False,
                    f"KXHIGHNY-26APR15-{st}-N{i}",
                )
            )
            sibling_pairs.append(
                (
                    0.7,
                    1,
                    "NYC",
                    "temperature_2m_max",
                    st,
                    "2026-04-15",
                    ("gfs_seamless", "ecmwf_ifs025"),
                    True,
                    f"KXHIGHNY-26APR15-{st}-P{i}",
                )
            )

    def fake_sibling(**kwargs):
        return iter(sibling_pairs)

    monkeypatch.setattr(cc, "build_production_mirrored_pairs_with_provenance", fake_sibling)

    out_provenance = tmp_path / "provenance.jsonl"
    out_global = tmp_path / "platt.json"
    out_per_type = tmp_path / "platt_pt.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(out_global),
        "--output-per-type",
        str(out_per_type),
        "--bias-table",
        str(tmp_path / "missing_bias.json"),  # use --no-bias-correction below
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--write-pair-provenance",
        str(out_provenance),
        "--min-per-type",
        "10",
    )

    fit_pl.main()

    # JSONL was written with one row per yielded pair.
    assert out_provenance.exists()
    rows = [json.loads(line) for line in out_provenance.read_text().splitlines() if line.strip()]
    assert len(rows) == len(sibling_pairs)
    for row in rows:
        assert row["schema_version"] == 1
        assert row["city"] == "NYC"
        assert row["daily_variable"] == "temperature_2m_max"
        assert row["strike_type"] in {"greater", "less", "between"}
        assert "raw_prob" in row
        assert "outcome" in row
        assert "models_used" in row
        assert isinstance(row["models_used"], list)  # JSON has no tuple type
        assert "bias_correction_applied" in row
        assert isinstance(row["bias_correction_applied"], bool)
        assert "spread_inflation_factor" in row
        assert "ensemble_weights_at_fit" in row
        assert "ticker" in row
        assert row["ticker"].startswith("KXHIGHNY-26APR15-")


# ---------------------------------------------------------------------------
# T4f1 — --write-pair-provenance truncates a stale target on re-run
# ---------------------------------------------------------------------------


def test_t4f1_write_pair_provenance_truncates_stale_target(tmp_path: Path, monkeypatch) -> None:
    """A pre-existing target file must be truncated before the new run starts.

    Regression: record_pair_provenance uses O_APPEND for atomic concurrent
    writes. Without truncation at the start of the loop, re-running the
    fitter against the same target accumulates on top of the prior run's
    output, producing duplicate rows and confusing the operator.
    """
    import scripts._calibration_common as cc
    import scripts.fit_platt as fit_pl

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    sibling_pairs = []
    for st in ("greater", "less", "between"):
        for i in range(60):
            sibling_pairs.append(
                (0.3, 0, "NYC", "temperature_2m_max", st, "2026-04-15",
                 ("gfs_seamless",), False, f"KXHIGHNY-26APR15-{st}-N{i}")
            )
            sibling_pairs.append(
                (0.7, 1, "NYC", "temperature_2m_max", st, "2026-04-15",
                 ("gfs_seamless", "ecmwf_ifs025"), True, f"KXHIGHNY-26APR15-{st}-P{i}")
            )

    monkeypatch.setattr(
        cc,
        "build_production_mirrored_pairs_with_provenance",
        lambda **_kw: iter(sibling_pairs),
    )

    out_provenance = tmp_path / "provenance.jsonl"
    out_provenance.write_text('{"schema_version":1,"ticker":"STALE-FROM-PREVIOUS-RUN"}\n')

    _set_argv(
        monkeypatch,
        "--snapshots", str(snapshots),
        "--output-global", str(tmp_path / "platt.json"),
        "--output-per-type", str(tmp_path / "platt_pt.json"),
        "--bias-table", str(tmp_path / "missing_bias.json"),
        "--ensemble-weights", json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--write-pair-provenance", str(out_provenance),
        "--min-per-type", "10",
    )

    fit_pl.main()

    rows = [json.loads(line) for line in out_provenance.read_text().splitlines() if line.strip()]
    assert len(rows) == len(sibling_pairs), (
        f"expected exactly {len(sibling_pairs)} rows; got {len(rows)} "
        "(stale rows from prior run leaked through)"
    )
    assert not any(r.get("ticker") == "STALE-FROM-PREVIOUS-RUN" for r in rows)


# ---------------------------------------------------------------------------
# T4g — sample-size guard
# ---------------------------------------------------------------------------


def test_t4g_sample_size_guard_aborts_below_100(tmp_path: Path, monkeypatch) -> None:
    """If full + single_model_era < 100 the preflight aborts before fitting."""
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    # 50 full + 40 single = 90 < 100 → abort.
    _stub_preflight(
        monkeypatch,
        _make_report(total=90, full=50, single_model_era=40),
    )
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="≥100|>=100|100"):
        fit_pl.main()


# ---------------------------------------------------------------------------
# T4h — n_triples_total == 0 abort
# ---------------------------------------------------------------------------


def test_t4h_total_zero_aborts(tmp_path: Path, monkeypatch) -> None:
    """If the preflight reports n_triples_total == 0 the fitter aborts."""
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=0, full=0))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="n_triples_total"):
        fit_pl.main()


# ---------------------------------------------------------------------------
# T4i — all-exempt abort
# ---------------------------------------------------------------------------


def test_t4i_all_exempt_aborts(tmp_path: Path, monkeypatch) -> None:
    """If every triple is calendar-exempt the fitter aborts."""
    import scripts.fit_platt as fit_pl
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_pl, "_PROJECT_ROOT", tmp_path)

    # n_triples_exempt == n_triples_total > 0 → abort with "exempt" message.
    _stub_preflight(monkeypatch, _make_report(total=200, full=0, exempt=200))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output-global",
        str(tmp_path / "platt.json"),
        "--output-per-type",
        str(tmp_path / "platt_pt.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="exempt"):
        fit_pl.main()
