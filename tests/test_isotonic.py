"""Tests for isotonic calibration loader + fallback chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.data.calibration import IsotonicCalibrator, _validate_curve


def _write_model(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "isotonic.json"
    path.write_text(json.dumps(payload))
    return path


class TestIsotonicCalibratorLoad:
    def test_missing_file_falls_back_to_identity(self, tmp_path: Path) -> None:
        cal = IsotonicCalibrator(path=tmp_path / "does-not-exist.json")
        assert cal.is_loaded is False
        assert cal.fallback_reason is not None
        assert cal.fallback_reason.startswith("file-missing:")
        # Identity mapping — the raw probability is returned unchanged.
        assert cal.calibrate(0.42) == 0.42

    def test_parse_error_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "isotonic.json"
        path.write_text("{not-json")
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is False
        assert cal.fallback_reason is not None
        assert cal.fallback_reason.startswith("parse-error:")
        assert cal.calibrate(0.5) == 0.5

    def test_schema_error_falls_back(self, tmp_path: Path) -> None:
        # Missing required keys.
        path = _write_model(tmp_path, {"fit_date": "2026-04-16"})
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is False
        assert cal.fallback_reason is not None
        assert cal.fallback_reason.startswith("schema-error:")

    def test_non_monotonic_breakpoints_rejected(self, tmp_path: Path) -> None:
        path = _write_model(
            tmp_path,
            {"breakpoints": [0.1, 0.05, 0.9], "values": [0.0, 0.2, 1.0]},
        )
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is False
        assert "non-monotonic-breakpoints" in (cal.fallback_reason or "")

    def test_non_monotonic_values_rejected(self, tmp_path: Path) -> None:
        path = _write_model(
            tmp_path,
            {"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 0.8, 0.4]},
        )
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is False
        assert "non-monotonic-values" in (cal.fallback_reason or "")

    def test_length_mismatch_rejected(self, tmp_path: Path) -> None:
        path = _write_model(
            tmp_path,
            {"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 1.0]},
        )
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is False
        assert "length-mismatch" in (cal.fallback_reason or "")

    def test_well_formed_loads(self, tmp_path: Path) -> None:
        path = _write_model(
            tmp_path,
            {
                "schema_version": 2,
                "fit_date": "2026-04-16",
                "brier_raw": 0.14,
                "brier_calibrated": 0.10,
                "breakpoints": [0.0, 0.5, 1.0],
                "values": [0.0, 0.25, 1.0],
            },
        )
        cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is True
        assert cal.fallback_reason is None
        assert cal.metadata["schema_version"] == 2

    def test_nested_metadata_fit_date_is_logged(self, tmp_path: Path, caplog) -> None:
        """The favorite_bias g(mid) map nests ``fit_date`` under ``metadata``.

        Reading only the top level logged "fit ?" for the live-money calibrator,
        so the startup log could not say which vintage of the map was serving.
        """
        path = _write_model(
            tmp_path,
            {
                "breakpoints": [0.0, 0.5, 1.0],
                "values": [0.0, 0.25, 1.0],
                "metadata": {"fit_date": "2026-07-04"},
            },
        )
        with caplog.at_level("INFO", logger="python.data.calibration"):
            cal = IsotonicCalibrator(path=path)
        assert cal.is_loaded is True
        assert "fit 2026-07-04" in caplog.text
        assert "fit ?" not in caplog.text

    def test_top_level_fit_date_preferred_over_metadata(self, tmp_path: Path, caplog) -> None:
        path = _write_model(
            tmp_path,
            {
                "fit_date": "2026-04-16",
                "breakpoints": [0.0, 0.5, 1.0],
                "values": [0.0, 0.25, 1.0],
                "metadata": {"fit_date": "2020-01-01"},
            },
        )
        with caplog.at_level("INFO", logger="python.data.calibration"):
            IsotonicCalibrator(path=path)
        assert "fit 2026-04-16" in caplog.text

    def test_artifact_hash_populated_on_load(self, tmp_path: Path) -> None:
        path = _write_model(
            tmp_path,
            {"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 0.25, 1.0]},
        )
        cal = IsotonicCalibrator(path=path)
        assert cal.artifact_hash is not None
        assert len(cal.artifact_hash) == 64  # sha256 hex
        # Stable across loads of the same file
        assert IsotonicCalibrator(path=path).artifact_hash == cal.artifact_hash

    def test_artifact_hash_differs_across_artifacts(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        a.write_text(
            json.dumps({"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 0.25, 1.0]})
        )
        b = tmp_path / "b.json"
        b.write_text(
            json.dumps({"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 0.30, 1.0]})
        )
        assert IsotonicCalibrator(path=a).artifact_hash != IsotonicCalibrator(path=b).artifact_hash

    def test_artifact_hash_none_on_fallback(self, tmp_path: Path) -> None:
        cal = IsotonicCalibrator(path=tmp_path / "missing.json")
        assert cal.is_loaded is False
        assert cal.artifact_hash is None


class TestIsotonicCalibrate:
    def _simple_cal(self, tmp_path: Path) -> IsotonicCalibrator:
        path = _write_model(
            tmp_path,
            {
                "breakpoints": [0.0, 0.5, 1.0],
                "values": [0.0, 0.25, 1.0],
            },
        )
        return IsotonicCalibrator(path=path)

    def test_exact_breakpoints(self, tmp_path: Path) -> None:
        cal = self._simple_cal(tmp_path)
        assert cal.calibrate(0.0) == pytest.approx(0.0)
        assert cal.calibrate(0.5) == pytest.approx(0.25)
        assert cal.calibrate(1.0) == pytest.approx(1.0)

    def test_linear_interpolation(self, tmp_path: Path) -> None:
        cal = self._simple_cal(tmp_path)
        # Halfway between (0.0, 0.0) and (0.5, 0.25) → 0.125.
        assert cal.calibrate(0.25) == pytest.approx(0.125)
        # Halfway between (0.5, 0.25) and (1.0, 1.0) → 0.625.
        assert cal.calibrate(0.75) == pytest.approx(0.625)

    def test_out_of_range_clamps(self, tmp_path: Path) -> None:
        cal = self._simple_cal(tmp_path)
        assert cal.calibrate(-0.5) == pytest.approx(0.0)
        assert cal.calibrate(1.5) == pytest.approx(1.0)


class TestValidateCurve:
    def test_accepts_well_formed(self) -> None:
        assert _validate_curve([0.0, 0.5, 1.0], [0.0, 0.3, 1.0]) is None

    def test_rejects_empty(self) -> None:
        assert _validate_curve([], []) == "empty"

    def test_rejects_single_breakpoint(self) -> None:
        assert _validate_curve([0.5], [0.5]) == "too-few-breakpoints"


def test_fit_isotonic_default_invokes_production_helper(tmp_path, monkeypatch):
    """Without flags, fit_isotonic.py defaults to production-mirrored builder."""
    import json
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))

    bias_path = tmp_path / "bias.json"
    bias_path.write_text(json.dumps({"corrections": {"NYC": {}}}))

    called = {"helper": False, "legacy": False}

    def fake_helper(**kwargs):
        called["helper"] = True
        return iter([])

    def fake_legacy(*args, **kwargs):
        called["legacy"] = True
        return []

    import scripts._calibration_common as cc
    import scripts._calibration_input_contract as ic
    import scripts.fit_isotonic as fit_iso
    monkeypatch.setattr(cc, "build_production_mirrored_pairs", fake_helper)
    monkeypatch.setattr(fit_iso, "_build_pairs", fake_legacy)

    # Stub the preflight so this test exercises only the dispatch path —
    # see Task 3 (input-contract preflight) for the full coverage tests in
    # tests/test_fit_isotonic.py.
    def fake_preflight(*args, **kwargs):
        return ic.InputContractReport(
            n_triples_total=200,
            n_triples_full=200,
            n_triples_partial=0,
            n_triples_empty=0,
            n_triples_single_model_era=0,
            n_triples_exempt=0,
            n_settlements_final=200,
            n_settlements_dropped_preliminary=0,
        )

    monkeypatch.setattr(ic, "assert_snapshot_input_contract", fake_preflight)

    monkeypatch.setattr(
        sys, "argv",
        [
            "fit_isotonic.py",
            "--snapshots", str(tmp_path / "snapshots"),
            "--output", str(tmp_path / "out.json"),
            "--bias-table", str(bias_path),
        ],
    )

    with pytest.raises(SystemExit):
        fit_iso.main()

    assert called["helper"] is True, "default should call production-mirrored helper"
    assert called["legacy"] is False


def test_fit_isotonic_rejects_no_spread_inflation_in_default_mode(tmp_path, monkeypatch):
    """--no-spread-inflation requires --legacy-gfs-only."""
    import sys

    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)

    import scripts.fit_isotonic as fit_iso

    monkeypatch.setattr(
        sys, "argv",
        [
            "fit_isotonic.py",
            "--snapshots", str(tmp_path / "snapshots"),
            "--output", str(tmp_path / "out.json"),
            "--no-spread-inflation",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        fit_iso.main()
    # argparse.error -> exit code 2
    assert exc_info.value.code == 2


def test_fit_isotonic_legacy_flag_uses_old_pair_builder(tmp_path, monkeypatch):
    """--legacy-gfs-only flag opts out and uses _build_pairs."""
    import json
    import sys

    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))

    called = {"helper": False, "legacy": False}

    def fake_helper(**kwargs):
        called["helper"] = True
        return iter([])

    def fake_legacy(*args, **kwargs):
        called["legacy"] = True
        return []

    import scripts._calibration_common as cc
    import scripts.fit_isotonic as fit_iso
    monkeypatch.setattr(cc, "build_production_mirrored_pairs", fake_helper)
    monkeypatch.setattr(fit_iso, "_build_pairs", fake_legacy)

    monkeypatch.setattr(
        sys, "argv",
        ["fit_isotonic.py",
         "--snapshots", str(tmp_path / "snapshots"),
         "--output", str(tmp_path / "out.json"),
         "--legacy-gfs-only"],
    )

    with pytest.raises(SystemExit):
        fit_iso.main()

    assert called["legacy"] is True
    assert called["helper"] is False
