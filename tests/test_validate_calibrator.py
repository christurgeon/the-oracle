"""Tests for the held-out-by-date pre-deploy validation gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _write_calibrator(path: Path, breakpoints, values):
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fit_date": "2026-05-01",
                "n_pairs": 1000,
                "brier_raw": 0.130,
                "brier_calibrated": 0.099,
                "improvement_pct": 23.8,
                "breakpoints": breakpoints,
                "values": values,
            }
        )
    )


def test_validate_calibrator_reports_holdout_size_and_max_value(tmp_path, monkeypatch):
    """Structural smoke test: holdout split + plateau/output-range checks."""
    from datetime import date, timedelta

    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    _write_calibrator(cand_path, [0.0, 0.5, 1.0], [0.05, 0.5, 0.95])
    _write_calibrator(base_path, [0.0, 1.0], [0.092, 0.456])

    train_dates = [date(2026, 4, 1) - timedelta(days=i) for i in range(20)]
    holdout_dates = [date(2026, 4, 28) - timedelta(days=i) for i in range(10)]

    train_pairs = []
    for d in train_dates:
        train_pairs += [(0.5, 1, "NYC", "temperature_2m_max", "between", d) for _ in range(5)] + [
            (0.5, 0, "NYC", "temperature_2m_max", "between", d) for _ in range(5)
        ]
    holdout_pairs = []
    for d in holdout_dates:
        holdout_pairs += [
            (0.05, 0, "NYC", "temperature_2m_max", "between", d) for _ in range(10)
        ] + [(0.95, 1, "NYC", "temperature_2m_max", "between", d) for _ in range(10)]
    all_pairs = train_pairs + holdout_pairs

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: all_pairs)

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.05,
        min_calibration_brier_lift=-1.0,
    )
    assert "n_holdout_pairs" in report
    assert report["n_holdout_pairs"] == 200
    assert report["n_train_pairs"] == 200
    assert report["max_calibrated_value"] == 0.95
    assert "plateau_too_wide" not in report.get("fail_reasons", [])
    assert "output_range_capped" not in report.get("fail_reasons", [])


def _synthetic_pairs(n_train: int = 200, n_holdout: int = 200) -> list:
    from datetime import date, timedelta

    pairs = []
    for i in range(n_train):
        d = date(2026, 4, 1) - timedelta(days=i % 20)
        pairs.append((0.5, i % 2, "NYC", "temperature_2m_max", "between", d))
    for i in range(n_holdout):
        d = date(2026, 4, 28) - timedelta(days=i % 10)
        pairs.append((0.5, i % 2, "NYC", "temperature_2m_max", "between", d))
    return pairs


def test_validate_calibrator_fails_on_wide_plateau(tmp_path, monkeypatch):
    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    _write_calibrator(cand_path, [0.0, 0.1, 0.5, 1.0], [0.05, 0.05, 0.05, 0.95])
    _write_calibrator(base_path, [0.0, 1.0], [0.10, 0.90])

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: _synthetic_pairs())

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.05,
        min_calibration_brier_lift=-1.0,
    )
    assert report["pass"] is False
    assert "plateau_too_wide" in report["fail_reasons"]


def test_validate_calibrator_fails_on_capped_output(tmp_path, monkeypatch):
    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    _write_calibrator(cand_path, [0.0, 1.0], [0.05, 0.45])
    _write_calibrator(base_path, [0.0, 1.0], [0.05, 0.95])

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: _synthetic_pairs())

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.5,
        min_calibration_brier_lift=-1.0,
    )
    assert "output_range_capped" in report["fail_reasons"]


def test_validate_calibrator_fails_on_insufficient_holdout(tmp_path, monkeypatch):
    """Holdout slice below the 150-pair threshold short-circuits to fail."""
    from datetime import date, timedelta

    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    _write_calibrator(cand_path, [0.0, 1.0], [0.05, 0.95])
    _write_calibrator(base_path, [0.0, 1.0], [0.05, 0.95])

    pairs = [
        (0.5, i % 2, "NYC", "temperature_2m_max", "between", date(2026, 4, 28) - timedelta(days=i))
        for i in range(5)
    ]
    train_pairs = [
        (0.5, i % 2, "NYC", "temperature_2m_max", "between", date(2026, 4, 1) - timedelta(days=i))
        for i in range(20)
    ]
    all_pairs = train_pairs + pairs

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: all_pairs)

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.5,
        min_calibration_brier_lift=-1.0,
    )
    assert report["pass"] is False
    assert "insufficient_holdout_pairs" in report["fail_reasons"]


def test_validate_calibrator_advisory_mode_demotes_structural_failures(tmp_path, monkeypatch):
    """In advisory mode, plateau/output failures move to advisory_warnings,
    not fail_reasons. The pass boolean depends only on the Brier lift gate."""
    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    # Wide plateau AND capped output — would fail strict mode on both.
    _write_calibrator(cand_path, [0.0, 0.6, 1.0], [0.05, 0.05, 0.45])
    _write_calibrator(base_path, [0.0, 1.0], [0.10, 0.90])

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: _synthetic_pairs())

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.05,
        min_calibration_brier_lift=-1.0,  # disable lift gate
        strict_structural_checks=False,
    )
    assert report["pass"] is True
    assert "plateau_too_wide" not in report["fail_reasons"]
    assert "output_range_capped" not in report["fail_reasons"]
    assert "plateau_too_wide" in report["advisory_warnings"]
    assert "output_range_capped" in report["advisory_warnings"]


def test_validate_calibrator_advisory_mode_still_gates_on_lift_failure(tmp_path, monkeypatch):
    """Advisory mode demotes structural checks but the Brier-lift gate
    remains load-bearing. A candidate that fails the lift threshold must
    still return pass=False even when structural failures are advisory."""
    from scripts.validate_calibrator import validate_calibrator_artifact

    cand_path = tmp_path / "candidate.json"
    base_path = tmp_path / "baseline.json"
    # Well-shaped candidate (no structural advisories) but a high lift
    # threshold the synthetic data cannot clear.
    _write_calibrator(cand_path, [0.0, 0.5, 1.0], [0.05, 0.5, 0.95])
    _write_calibrator(base_path, [0.0, 1.0], [0.05, 0.95])

    import scripts.validate_calibrator as mod

    monkeypatch.setattr(mod, "_load_pairs_with_dates", lambda **kw: _synthetic_pairs())

    report = validate_calibrator_artifact(
        candidate_path=cand_path,
        baseline_path=base_path,
        snapshots_dir=tmp_path,
        bias_table_path=None,
        ensemble_weights={"gfs_seamless": 1.0},
        holdout_days=14,
        max_plateau_width=0.5,
        min_calibration_brier_lift=10.0,  # impossibly high -> lift gate fails
        strict_structural_checks=False,  # advisory mode
    )
    assert report["pass"] is False
    assert "calibration_no_lift_on_holdout" in report["fail_reasons"]


# ---------------------------------------------------------------------------
# Preflight-wiring tests
# ---------------------------------------------------------------------------


def _empty_snap_dir(tmp_path: Path) -> Path:
    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))
    return tmp_path / "snapshots"


def _write_calendar(path: Path, models: dict[str, str] | None = None) -> Path:
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _stub_preflight_check(monkeypatch, report) -> dict:
    """Stub ``assert_snapshot_input_contract`` to return ``report``."""
    import scripts._calibration_input_contract as ic

    called = {"preflight": False}

    def fake(*args, **kwargs):
        called["preflight"] = True
        return report

    monkeypatch.setattr(ic, "assert_snapshot_input_contract", fake)
    return called


def _stub_validator(monkeypatch) -> dict:
    """Stub ``validate_calibrator_artifact`` so heavy math is skipped."""
    import scripts.validate_calibrator as mod

    called = {"called": False}

    def fake(**kwargs):
        called["called"] = True
        return {"pass": True, "fail_reasons": [], "advisory_warnings": []}

    monkeypatch.setattr(mod, "validate_calibrator_artifact", fake)
    return called


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_calibrator.py", *args])


def _write_dummy_calibrator(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fit_date": "2026-05-01",
                "n_pairs": 1000,
                "brier_raw": 0.130,
                "brier_calibrated": 0.099,
                "improvement_pct": 23.8,
                "breakpoints": [0.0, 0.5, 1.0],
                "values": [0.05, 0.5, 0.95],
            }
        )
    )
    return path


def test_t11_validate_calibrator_rejects_partial_holdout(tmp_path, monkeypatch):
    """Validator rejects partial-coverage corpus; --allow-partial-model-coverage bypasses."""
    import scripts.validate_calibrator as vc
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snap_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(vc, "_PROJECT_ROOT", tmp_path)

    cand_path = _write_dummy_calibrator(tmp_path / "candidate.json")
    base_path = _write_dummy_calibrator(tmp_path / "baseline.json")

    _stub_preflight_check(monkeypatch, _make_report(total=200, full=190, partial=10))
    _stub_validator(monkeypatch)

    _set_argv(
        monkeypatch,
        "--candidate",
        str(cand_path),
        "--baseline",
        str(base_path),
        "--snapshots",
        str(snapshots),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )
    with pytest.raises(CalibratorInputContractError, match="--allow-partial-model-coverage"):
        vc.main()

    _set_argv(
        monkeypatch,
        "--candidate",
        str(cand_path),
        "--baseline",
        str(base_path),
        "--snapshots",
        str(snapshots),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--allow-partial-model-coverage",
    )
    rc = vc.main()
    assert rc == 0


def test_t_vc_2_calendar_missing_aborts(tmp_path, monkeypatch):
    """T-vc-2: calendar missing → CalibratorInputContractError."""
    import scripts._calibration_common as cc
    import scripts.validate_calibrator as vc
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snap_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    # Calendar deliberately not written.
    monkeypatch.setattr(vc, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cc, "_PROJECT_ROOT", tmp_path)

    cand_path = _write_dummy_calibrator(tmp_path / "candidate.json")
    base_path = _write_dummy_calibrator(tmp_path / "baseline.json")

    _set_argv(
        monkeypatch,
        "--candidate",
        str(cand_path),
        "--baseline",
        str(base_path),
        "--snapshots",
        str(snapshots),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )
    with pytest.raises(CalibratorInputContractError, match="calendar"):
        vc.main()


def test_t_vc_3_bias_table_missing_without_no_bias_correction_aborts(tmp_path, monkeypatch):
    """T-vc-3: bias-table missing without --no-bias-correction → abort."""
    import scripts.validate_calibrator as vc
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snap_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(vc, "_PROJECT_ROOT", tmp_path)

    cand_path = _write_dummy_calibrator(tmp_path / "candidate.json")
    base_path = _write_dummy_calibrator(tmp_path / "baseline.json")
    bias_path = tmp_path / "missing_bias.json"  # NOT written

    _set_argv(
        monkeypatch,
        "--candidate",
        str(cand_path),
        "--baseline",
        str(base_path),
        "--snapshots",
        str(snapshots),
        "--bias-table",
        str(bias_path),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )
    with pytest.raises(CalibratorInputContractError, match="--no-bias-correction"):
        vc.main()
