"""Unit tests for python.analytics.isotonic_fit.

The headline test is U-12 (bit-identical extraction): the new hermetic
fit_isotonic_pairs() and the refactored legacy fit_isotonic() must both
agree with the pre-refactor frozen baseline at
tests/analytics/fixtures/isotonic_legacy_baseline.json.

The baseline survives legacy-code deletion — it is the actual regression
gate, not the parallel-fit tautology of the prior plan revision.

Baseline regeneration: if a sklearn version bump or numerical change
invalidates the baseline, regenerate by running the same generator that
produced it. The generator is _make_synthetic_pairs(n=200, seed=42) +
sklearn IsotonicRegression with out_of_bounds="clip", y_min=0.0,
y_max=1.0, exactly as fit_isotonic_model does today. Capture
model.X_thresholds_.tolist() / y_thresholds_.tolist() for both
anchor_endpoints=True and anchor_endpoints=False; write to
tests/analytics/fixtures/isotonic_legacy_baseline.json with the same
two-key structure. Verify before commit that all U-12 tests pass against
the new baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on sys.path so we can import scripts.* helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.analytics.isotonic_fit import (  # noqa: E402
    default_anchor_weight,
    fit_isotonic_model,
    fit_isotonic_pairs,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_BASELINE_PATH = _FIXTURES_DIR / "isotonic_legacy_baseline.json"


def _make_synthetic_pairs(n: int = 200, seed: int = 42) -> np.ndarray:
    """Canonical synthetic corpus — must match the Step 2 fixture generator."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=n)
    true_p = 1.0 / (1.0 + np.exp(-8.0 * (raw - 0.5)))
    outcomes = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.float64)
    return np.column_stack([raw, outcomes])


def _load_baseline(label: str) -> tuple[list[float], list[float]]:
    payload = json.loads(_BASELINE_PATH.read_text())
    block = payload[label]
    return block["X_thresholds"], block["y_thresholds"]


def test_u12_pairs_match_frozen_baseline_anchored():
    """U-12: fit_isotonic_pairs matches the pre-refactor legacy thresholds (anchored)."""
    pairs = _make_synthetic_pairs(n=200, seed=42)
    bp, vl = fit_isotonic_pairs(pairs, anchor_endpoints=True, anchor_weight=None)
    expected_bp, expected_vl = _load_baseline("n200_anchored")
    assert bp == expected_bp
    assert vl == expected_vl


def test_u12_model_match_frozen_baseline_anchored():
    """U-12: fit_isotonic_model.X_thresholds_ matches the pre-refactor legacy thresholds."""
    pairs = _make_synthetic_pairs(n=200, seed=42)
    model = fit_isotonic_model(pairs, anchor_endpoints=True, anchor_weight=None)
    expected_bp, expected_vl = _load_baseline("n200_anchored")
    assert model.X_thresholds_.tolist() == expected_bp
    assert model.y_thresholds_.tolist() == expected_vl


def test_u12_legacy_wrapper_match_frozen_baseline():
    """U-12: the refactored scripts.fit_isotonic.fit_isotonic wrapper matches baseline.

    This guards against the wrapper drifting from the hermetic primitive
    during future cleanups.
    """
    from scripts.fit_isotonic import fit_isotonic as legacy_fit_isotonic

    pairs = _make_synthetic_pairs(n=200, seed=42)
    features = pairs[:, 0]
    targets = pairs[:, 1]
    model = legacy_fit_isotonic(features, targets, anchor_endpoints=True, anchor_weight=None)
    expected_bp, expected_vl = _load_baseline("n200_anchored")
    assert model.X_thresholds_.tolist() == expected_bp
    assert model.y_thresholds_.tolist() == expected_vl


def test_u12b_no_anchor_match_frozen_baseline():
    """U-12b: anchor_endpoints=False path also matches the pre-refactor baseline."""
    pairs = _make_synthetic_pairs(n=200, seed=42)
    bp, vl = fit_isotonic_pairs(pairs, anchor_endpoints=False)
    expected_bp, expected_vl = _load_baseline("n200_no_anchor")
    assert bp == expected_bp
    assert vl == expected_vl


def test_u13_default_anchor_weight_direct():
    """U-13a: default_anchor_weight matches the documented formula at canonical sizes."""
    assert default_anchor_weight(0) == 5
    assert default_anchor_weight(200) == 5
    assert default_anchor_weight(999) == 5
    assert default_anchor_weight(1000) == 5
    assert default_anchor_weight(1001) == 5  # 1001 // 200 == 5
    assert default_anchor_weight(2000) == 10
    assert default_anchor_weight(5000) == 25


def test_u13_anchor_weight_composed():
    """U-13b: anchor_weight=None and explicit value agree when both resolve to same weight."""
    pairs_400 = _make_synthetic_pairs(n=400, seed=1)
    bp_default, _ = fit_isotonic_pairs(pairs_400, anchor_weight=None)
    bp_explicit_5, _ = fit_isotonic_pairs(pairs_400, anchor_weight=5)
    assert bp_default == bp_explicit_5  # default_anchor_weight(400) == 5

    pairs_2000 = _make_synthetic_pairs(n=2000, seed=2)
    bp_default_2k, _ = fit_isotonic_pairs(pairs_2000, anchor_weight=None)
    bp_explicit_10, _ = fit_isotonic_pairs(pairs_2000, anchor_weight=10)
    assert bp_default_2k == bp_explicit_10  # default_anchor_weight(2000) == 10


def test_u14_edge_empty():
    """U-14a: empty pairs raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        fit_isotonic_pairs(np.empty((0, 2)))


def test_u14_edge_wrong_shape():
    """U-14b: wrong-shape pairs raises ValueError."""
    with pytest.raises(ValueError, match="shape"):
        fit_isotonic_pairs(np.zeros((10, 3)))


def test_u14_edge_non_finite():
    """U-14c: non-finite values in pairs raises ValueError."""
    pairs = _make_synthetic_pairs(n=10, seed=99)
    pairs[3, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fit_isotonic_pairs(pairs)


def test_u14_edge_all_zeros():
    """U-14d: all-zero outcomes produce a degenerate-but-valid curve."""
    pairs = np.column_stack([
        np.linspace(0.0, 1.0, 100),
        np.zeros(100),
    ])
    bp, vals = fit_isotonic_pairs(pairs, anchor_endpoints=False)
    assert len(bp) >= 1
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_u14_edge_all_ones():
    """U-14e: all-one outcomes produce a degenerate-but-valid curve."""
    pairs = np.column_stack([
        np.linspace(0.0, 1.0, 100),
        np.ones(100),
    ])
    bp, vals = fit_isotonic_pairs(pairs, anchor_endpoints=False)
    assert len(bp) >= 1
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_u14_edge_float32_input_matches_float64():
    """U-14f: float32 input is coerced to float64 and produces valid output.

    The hermetic module casts inputs to float64 internally; this test
    documents that contract by verifying:
    1. float32 input does not crash or raise.
    2. The resulting thresholds are bit-identical to a fit on the SAME data
       values (pairs_f32 round-tripped through float32→float64), proving
       the internal cast is the only dtype conversion happening.
    """
    pairs_f64 = _make_synthetic_pairs(n=200, seed=42)
    pairs_f32 = pairs_f64.astype(np.float32)
    # Round-trip: treat f32 as the source of truth, cast to f64 explicitly.
    pairs_f32_as_f64 = pairs_f32.astype(np.float64)

    bp32, vl32 = fit_isotonic_pairs(pairs_f32, anchor_endpoints=True)
    bp32_as_f64, vl32_as_f64 = fit_isotonic_pairs(pairs_f32_as_f64, anchor_endpoints=True)

    # The internal cast to float64 must produce the same result as an
    # explicit external cast — no hidden double-conversion or silent clamp.
    assert bp32 == bp32_as_f64
    assert vl32 == vl32_as_f64

    # The output must be a valid monotone calibration curve.
    assert len(bp32) >= 1
    assert all(0.0 <= v <= 1.0 for v in vl32)
    assert all(vl32[i] <= vl32[i + 1] for i in range(len(vl32) - 1))


def test_isotonic_fit_not_imported_by_runtime_modules():
    """Structural: runtime modules MUST NOT pull in isotonic_fit (sklearn).

    Enforces the hermetic-module contract documented in the isotonic_fit
    module docstring. If a future PR adds `from python.analytics.isotonic_fit
    import ...` to any runtime path (calibration, strategy, orchestrator),
    this test fires and the contract is restored before merge.
    """
    import importlib
    import sys

    # Drop any cached harness/fit imports from earlier tests in this run
    # so we get a clean view of what each runtime module pulls in.
    for cached in [k for k in list(sys.modules) if k.startswith("python.analytics.isotonic_fit")]:
        del sys.modules[cached]

    runtime_modules = [
        "python.data.calibration",
        "python.strategies.weather_edge",
        "python.strategies.weather_evaluation",
        "python.orchestrator",
    ]
    for mod_name in runtime_modules:
        importlib.import_module(mod_name)
        assert "python.analytics.isotonic_fit" not in sys.modules, (
            f"{mod_name} imported python.analytics.isotonic_fit — "
            "sklearn would become a runtime dependency. See the module "
            "docstring for the offline-only contract."
        )
