"""Tests for the unified calibration layer.

Covers:
  * IdentityCalibrator — pass-through
  * IsotonicCalibrator — back-compat via shim + new path
  * PlattCalibrator — sigmoid, overflow-safe, clamp
  * PerTypePlattCalibrator — per-key fits, load-time key assertion,
    hot-path refusal on unknown strike_type
  * load_calibrator — legacy flag resolution, explicit override,
    shadow namespace, identity fallback when artifact is missing
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from python.data.calibration import (
    STRIKE_TYPES,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PerTypePlattCalibrator,
    PlattCalibrator,
    _apply_platt,
    _stable_sigmoid,
    build_calibrator,
    load_calibrator,
)


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# Protocol / identity
# ---------------------------------------------------------------------------


class TestIdentityCalibrator:
    def test_is_loaded_and_kind(self) -> None:
        cal = IdentityCalibrator()
        assert cal.is_loaded is True
        assert cal.fallback_reason is None
        assert cal.kind == "identity"

    def test_passthrough(self) -> None:
        cal = IdentityCalibrator()
        assert cal.calibrate(0.0) == 0.0
        assert cal.calibrate(0.5) == 0.5
        assert cal.calibrate(1.0) == 1.0
        # strike_type must be accepted for protocol compatibility.
        assert cal.calibrate(0.7, strike_type="greater") == 0.7

    def test_satisfies_protocol(self) -> None:
        assert isinstance(IdentityCalibrator(), Calibrator)


# ---------------------------------------------------------------------------
# Sigmoid / platt helpers
# ---------------------------------------------------------------------------


class TestStableSigmoid:
    def test_zero(self) -> None:
        assert _stable_sigmoid(0.0) == pytest.approx(0.5)

    def test_positive(self) -> None:
        assert _stable_sigmoid(2.0) == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))

    def test_negative(self) -> None:
        assert _stable_sigmoid(-2.0) == pytest.approx(math.exp(-2.0) / (1.0 + math.exp(-2.0)))

    def test_large_positive_does_not_overflow(self) -> None:
        # Straight math.exp(+1000) overflows; the stable form does not.
        assert _stable_sigmoid(1000.0) == pytest.approx(1.0)

    def test_large_negative_does_not_overflow(self) -> None:
        assert _stable_sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-300)


class TestApplyPlatt:
    def test_identity_parameters(self) -> None:
        # a=0, b=1: sigmoid(raw). Not literally identity, just a sanity check.
        out = _apply_platt(0.5, 0.0, 1.0)
        assert out == pytest.approx(1.0 / (1.0 + math.exp(-0.5)))

    def test_extreme_coefficients_clamped(self) -> None:
        # a=-50, b=100, raw=0 → z = -50 → ~0 but clamped above 0
        out = _apply_platt(0.0, -50.0, 100.0)
        assert 0 < out < 1
        # a=50, b=-100, raw=1 → z = -50 → ~0 but clamped above 0
        out = _apply_platt(1.0, 50.0, -100.0)
        assert 0 < out < 1

    def test_non_finite_raw_safe(self) -> None:
        assert _apply_platt(float("nan"), 0.0, 1.0) == 0.5


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class TestPlattCalibrator:
    def test_missing_file_identity_fallback(self, tmp_path: Path) -> None:
        cal = PlattCalibrator(path=tmp_path / "does-not-exist.json")
        assert cal.is_loaded is False
        assert (cal.fallback_reason or "").startswith("file-missing:")
        # Identity pass-through.
        assert cal.calibrate(0.42) == 0.42

    def test_parse_error_identity(self, tmp_path: Path) -> None:
        p = tmp_path / "platt.json"
        p.write_text("{not-json")
        cal = PlattCalibrator(path=p)
        assert cal.is_loaded is False
        assert (cal.fallback_reason or "").startswith("parse-error:")

    def test_schema_missing_key(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "platt.json", {"a": 0.0})
        cal = PlattCalibrator(path=p)
        assert cal.is_loaded is False
        assert (cal.fallback_reason or "").startswith("schema-error:")

    def test_non_finite_rejected(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "platt.json", {"a": float("inf"), "b": 1.0})
        cal = PlattCalibrator(path=p)
        assert cal.is_loaded is False
        assert (cal.fallback_reason or "").startswith("validation-error:")

    def test_well_formed_loads(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "platt.json",
            {"schema_version": 1, "fit_date": "2026-04-21", "a": -0.5, "b": 2.0},
        )
        cal = PlattCalibrator(path=p)
        assert cal.is_loaded is True
        assert cal.coefficients == (-0.5, 2.0)
        expected = _apply_platt(0.6, -0.5, 2.0)
        assert cal.calibrate(0.6) == pytest.approx(expected)

    def test_strike_type_ignored(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "platt.json", {"a": 0.0, "b": 1.0})
        cal = PlattCalibrator(path=p)
        v1 = cal.calibrate(0.7)
        v2 = cal.calibrate(0.7, strike_type="greater")
        v3 = cal.calibrate(0.7, strike_type="between")
        assert v1 == v2 == v3


# ---------------------------------------------------------------------------
# PerTypePlattCalibrator
# ---------------------------------------------------------------------------


def _per_type_payload(
    greater: tuple[float, float] = (-0.1, 1.1),
    less: tuple[float, float] = (-0.2, 1.0),
    between: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fit_date": "2026-04-21",
        "per_type": {
            "greater": {"a": greater[0], "b": greater[1], "n": 800},
            "less": {"a": less[0], "b": less[1], "n": 600},
            "between": {"a": between[0], "b": between[1], "n": 300},
        },
    }


class TestPerTypePlattCalibrator:
    def test_missing_file_identity(self, tmp_path: Path) -> None:
        cal = PerTypePlattCalibrator(path=tmp_path / "nope.json")
        assert cal.is_loaded is False
        assert cal.calibrate(0.5, strike_type="greater") == 0.5

    def test_missing_per_type_key(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "pt.json", {"schema_version": 1})
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded is False
        assert "missing-per-type" in (cal.fallback_reason or "")

    def test_wrong_strike_type_keys_rejected(self, tmp_path: Path) -> None:
        payload = _per_type_payload()
        # Rename one key — load must refuse rather than silently fall back.
        payload["per_type"]["above"] = payload["per_type"].pop("greater")
        p = _write(tmp_path / "pt.json", payload)
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded is False
        assert "strike-type-keys" in (cal.fallback_reason or "")

    def test_extra_keys_rejected(self, tmp_path: Path) -> None:
        payload = _per_type_payload()
        payload["per_type"]["bonus"] = {"a": 0.0, "b": 1.0, "n": 10}
        p = _write(tmp_path / "pt.json", payload)
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded is False

    def test_non_finite_coefficients_rejected(self, tmp_path: Path) -> None:
        payload = _per_type_payload(greater=(float("nan"), 1.0))
        p = _write(tmp_path / "pt.json", payload)
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded is False

    def test_well_formed_all_three_types(self, tmp_path: Path) -> None:
        payload = _per_type_payload()
        p = _write(tmp_path / "pt.json", payload)
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded is True
        # Each strike type produces the Platt output for its own coefficients.
        for stype in STRIKE_TYPES:
            a = payload["per_type"][stype]["a"]
            b = payload["per_type"][stype]["b"]
            expected = _apply_platt(0.55, a, b)
            assert cal.calibrate(0.55, strike_type=stype) == pytest.approx(expected)

    def test_unknown_strike_type_raises(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "pt.json", _per_type_payload())
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded
        with pytest.raises(ValueError, match="strike_type"):
            cal.calibrate(0.5, strike_type="above")

    def test_missing_strike_type_raises(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "pt.json", _per_type_payload())
        cal = PerTypePlattCalibrator(path=p)
        assert cal.is_loaded
        with pytest.raises(ValueError, match="strike_type"):
            cal.calibrate(0.5)


# ---------------------------------------------------------------------------
# IsotonicCalibrator widened signature
# ---------------------------------------------------------------------------


class TestIsotonicStrikeTypeKwarg:
    def test_accepts_and_ignores_strike_type(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path / "iso.json",
            {"breakpoints": [0.0, 0.5, 1.0], "values": [0.0, 0.25, 1.0]},
        )
        cal = IsotonicCalibrator(path=p)
        assert cal.calibrate(0.25) == pytest.approx(0.125)
        # Same output regardless of strike_type.
        for stype in ("greater", "less", "between", None, "anything"):
            assert cal.calibrate(0.25, strike_type=stype) == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# load_calibrator / build_calibrator
# ---------------------------------------------------------------------------


class TestBuildCalibrator:
    def test_none_returns_identity(self) -> None:
        assert isinstance(build_calibrator(None), IdentityCalibrator)
        assert isinstance(build_calibrator("none"), IdentityCalibrator)
        assert isinstance(build_calibrator("identity"), IdentityCalibrator)

    def test_unknown_kind_identity(self, caplog: pytest.LogCaptureFixture) -> None:
        cal = build_calibrator("mystery")
        assert isinstance(cal, IdentityCalibrator)


class TestLoadCalibrator:
    def test_no_config_returns_identity(self) -> None:
        cal = load_calibrator({})
        assert isinstance(cal, IdentityCalibrator)

    def test_explicit_calibration_type_selects_platt(self, tmp_path: Path) -> None:
        platt = _write(tmp_path / "p.json", {"a": 0.0, "b": 1.0})
        cal = load_calibrator(
            {"calibration_type": "platt"},
            platt_path=platt,
        )
        assert isinstance(cal, PlattCalibrator)

    def test_per_type_platt_loaded(self, tmp_path: Path) -> None:
        pt = _write(tmp_path / "pt.json", _per_type_payload())
        cal = load_calibrator(
            {"calibration_type": "per_type_platt"},
            per_type_platt_path=pt,
        )
        assert isinstance(cal, PerTypePlattCalibrator)
        assert cal.is_loaded

    def test_missing_artifact_falls_back_to_identity(self, tmp_path: Path) -> None:
        cal = load_calibrator(
            {"calibration_type": "platt"},
            platt_path=tmp_path / "absent.json",
        )
        # load_calibrator degrades to Identity when the real calibrator
        # fails to load cleanly, so signal generation keeps running.
        assert isinstance(cal, IdentityCalibrator)

    def test_shadow_namespace_separate(self, tmp_path: Path) -> None:
        iso = _write(
            tmp_path / "iso.json",
            {"breakpoints": [0.0, 1.0], "values": [0.0, 1.0]},
        )
        # Config selects isotonic for live, but shadow is unset.
        cfg = {"calibration_type": "isotonic"}
        shadow = load_calibrator(cfg, key="shadow_calibrator", isotonic_path=iso)
        assert isinstance(shadow, IdentityCalibrator)

    def test_shadow_namespace_explicit(self, tmp_path: Path) -> None:
        platt = _write(tmp_path / "p.json", {"a": 0.0, "b": 1.0})
        cfg = {"calibration_type": "none", "shadow_calibrator": "platt"}
        shadow = load_calibrator(cfg, key="shadow_calibrator", platt_path=platt)
        assert isinstance(shadow, PlattCalibrator)
        assert shadow.is_loaded

    def test_none_kind_disables(self, tmp_path: Path) -> None:
        cal = load_calibrator({"calibration_type": "none"})
        assert isinstance(cal, IdentityCalibrator)
