"""Tests for ``scripts/validate_plateau_bounds.py``.

Focus: the F1.4 held-out reliability gate (`held_out_reliability_check`) and
its CLI surface. Plateau geometry is exercised by the existing
``tests/test_audit_tripwires.py`` and ``tests/test_calibrate_script.py``;
these tests don't re-cover that.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.validate_plateau_bounds import (
    BOOTSTRAP_SEED,
    FLOOR_DEFEAT_EPSILON,
    MAX_PLATEAU_DEVIATION,
    MIN_HELD_OUT_N,
    cluster_bootstrap_ci,
    floor_coupling_check,
    held_out_reliability_check,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "validate_plateau_bounds.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_pairs(
    *,
    n: int,
    yes_rate: float,
    raw_prob: float,
    base_date: date = date(2026, 5, 10),
    pairs_per_day: int = 3,
) -> list[tuple[float, int, date]]:
    """Build ``n`` synthetic pairs with the requested empirical YES rate.

    Pairs are spread across multiple forecast_dates (``pairs_per_day`` each)
    so cluster bootstrap has more than one cluster to resample.
    """
    pairs: list[tuple[float, int, date]] = []
    n_yes_remaining = int(round(yes_rate * n))
    for i in range(n):
        day_offset = i // pairs_per_day
        d = base_date + timedelta(days=day_offset)
        outcome = 1 if n_yes_remaining > 0 else 0
        if n_yes_remaining > 0:
            n_yes_remaining -= 1
        pairs.append((raw_prob, outcome, d))
    return pairs


def _isotonic_artifact_text(
    *,
    breakpoints: list[float],
    values: list[float],
    fit_date: str = "2026-05-09",
) -> str:
    """Build a minimal isotonic.json string for CLI tests."""
    return json.dumps(
        {
            "schema_version": 3,
            "fit_date": fit_date,
            "input_pipeline": "production_mirrored",
            "bias_correction_applied": True,
            "ensemble_weights": {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            "n_pairs": 4020,
            "breakpoints": breakpoints,
            "values": values,
        }
    )


def _live_artifact_text() -> str:
    """Mirror the geometry of the 2026-05-12 live artifact: one dangerous
    plateau in [0.649, 0.967] with value 0.321678. Only the plateau-relevant
    breakpoints are required for the geometry walker.
    """
    return _isotonic_artifact_text(
        breakpoints=[0.0, 0.5, 0.648667, 0.966667, 1.0],
        values=[0.05, 0.10, 0.321678, 0.321678, 0.95],
    )


# ---------------------------------------------------------------------------
# Test 1 — gate passes when empirical agrees with plateau value
# ---------------------------------------------------------------------------


def test_held_out_check_passes_when_empirical_matches_plateau() -> None:
    """Empirical YES ≈ plateau value AND n >= MIN_HELD_OUT_N → status='ok'."""
    plateau_value = 0.32
    pairs = _synthetic_pairs(n=45, yes_rate=0.30, raw_prob=0.80)

    result = held_out_reliability_check(
        pairs=pairs,
        low=0.65,
        high=0.967,
        plateau_value=plateau_value,
        held_out_after=date(2026, 5, 9),
    )

    assert result["status"] == "ok", result["message"]
    assert result["n"] == 45
    assert result["deviation"] < MAX_PLATEAU_DEVIATION
    assert "OK" in result["message"]


def test_held_out_check_passes_via_cli_with_synthetic_artifact(tmp_path: Path) -> None:
    """End-to-end: validator with skip-flag exits 0 on the live-shaped artifact.

    Snapshot integration is exercised by the verification run on the live
    artifact (see Phase-2 verification in the follow-up doc); this CLI test
    confirms the plateau detection + skip path still works.
    """
    iso = tmp_path / "isotonic.json"
    iso.write_text(_live_artifact_text())

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--isotonic",
            str(iso),
            "--skip-held-out-check",
        ],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "block_raw_prob_range: [0.649, 0.967]" in completed.stdout
    assert "SKIPPED" in completed.stderr


# ---------------------------------------------------------------------------
# Test 2 — gate fails when empirical diverges by > MAX_PLATEAU_DEVIATION
# ---------------------------------------------------------------------------


def test_held_out_check_fails_when_empirical_diverges() -> None:
    """Empirical YES way above plateau by > 0.10 AND n >= 30 → status='fail'.

    Mirrors the Phase-1 finding shape: artifact says plateau ~0.32, but the
    held-out window shows ~0.73. If we ever again accumulate n>=30 in a band
    where the empirical rate disagrees, the gate must surface it.
    """
    plateau_value = 0.32
    pairs = _synthetic_pairs(n=45, yes_rate=0.73, raw_prob=0.80)

    result = held_out_reliability_check(
        pairs=pairs,
        low=0.65,
        high=0.967,
        plateau_value=plateau_value,
        held_out_after=date(2026, 5, 9),
    )

    assert result["status"] == "fail", result["message"]
    assert result["n"] == 45
    assert result["deviation"] >= MAX_PLATEAU_DEVIATION
    assert "FAIL" in result["message"]
    # Sanity: message should call out the deviation explicitly.
    assert f"{result['deviation']:.4f}" in result["message"]


def test_held_out_check_fails_when_empirical_far_below_plateau() -> None:
    """Symmetric: empirical << plateau by > 0.10 also fails.

    Catches the inverse pathology (suppression earned MORE than the artifact
    claims; bound could in principle be tightened).
    """
    plateau_value = 0.50
    pairs = _synthetic_pairs(n=45, yes_rate=0.20, raw_prob=0.80)

    result = held_out_reliability_check(
        pairs=pairs,
        low=0.65,
        high=0.967,
        plateau_value=plateau_value,
        held_out_after=date(2026, 5, 9),
    )

    assert result["status"] == "fail"
    assert result["deviation"] >= MAX_PLATEAU_DEVIATION


# ---------------------------------------------------------------------------
# Test 3 — insufficient n warns, does not fail
# ---------------------------------------------------------------------------


def test_held_out_check_insufficient_n_warns_not_fails() -> None:
    """n < MIN_HELD_OUT_N → status='warn_insufficient_n' regardless of deviation.

    Don't block refits during corpus accumulation — this is exactly the
    current state (n=15 in band as of 2026-05-19).
    """
    plateau_value = 0.32
    # n=10, large divergence: would FAIL with enough data; must WARN here.
    pairs = _synthetic_pairs(n=10, yes_rate=0.80, raw_prob=0.80)

    result = held_out_reliability_check(
        pairs=pairs,
        low=0.65,
        high=0.967,
        plateau_value=plateau_value,
        held_out_after=date(2026, 5, 9),
    )

    assert result["status"] == "warn_insufficient_n", result["message"]
    assert result["n"] == 10
    assert "WARN" in result["message"]


def test_held_out_check_zero_pairs_warns_not_fails() -> None:
    """n == 0 → status='warn_insufficient_n'. The validator must not exit
    non-zero just because no held-out data has accumulated yet."""
    result = held_out_reliability_check(
        pairs=[],
        low=0.65,
        high=0.967,
        plateau_value=0.32,
        held_out_after=date(2026, 5, 9),
    )
    assert result["status"] == "warn_insufficient_n"
    assert result["n"] == 0


# ---------------------------------------------------------------------------
# Test 4 — cluster bootstrap resamples on forecast_date, not rows
# ---------------------------------------------------------------------------


def test_held_out_check_cluster_bootstrap_not_row() -> None:
    """Resampling unit MUST be forecast_date (F1.2).

    Falsification harness: build two datasets that have identical (raw_prob,
    outcome) row-pools but very different forecast_date clustering. Row
    bootstrap would give identical CIs; cluster bootstrap must differ.
    """
    # Dataset A: 30 wins spread across 30 forecast_dates (high cluster diversity).
    pairs_a = [(0.80, 1, date(2026, 5, 10) + timedelta(days=i)) for i in range(30)]
    # Dataset B: 30 wins concentrated on 1 forecast_date (zero diversity).
    pairs_b = [(0.80, 1, date(2026, 5, 10)) for _ in range(30)]

    lo_a, hi_a = cluster_bootstrap_ci(pairs_a, n_boot=2000, seed=BOOTSTRAP_SEED)
    lo_b, hi_b = cluster_bootstrap_ci(pairs_b, n_boot=2000, seed=BOOTSTRAP_SEED)

    # All-1 outcomes => mean is always 1.0 regardless of resampling — this
    # confirms by construction the CI is degenerate on outcome=1 data.
    assert lo_a == hi_a == 1.0
    assert lo_b == hi_b == 1.0

    # Sharper falsification: mix outcomes so the CI is non-degenerate. With
    # cluster bootstrap, n_clusters=1 collapses CI to a single value (resampling
    # the same cluster gives the same mean); n_clusters=k>1 gives a wider CI.
    mixed_a: list[tuple[float, int, date]] = []
    for i in range(30):
        # 30 dates, 1 row each, alternating outcomes → CI non-trivial.
        mixed_a.append((0.80, i % 2, date(2026, 5, 10) + timedelta(days=i)))

    mixed_b: list[tuple[float, int, date]] = []
    for i in range(30):
        # 1 date, 30 rows, same outcome alternation → CI collapses to single mean.
        mixed_b.append((0.80, i % 2, date(2026, 5, 10)))

    lo_a2, hi_a2 = cluster_bootstrap_ci(mixed_a, n_boot=2000, seed=BOOTSTRAP_SEED)
    lo_b2, hi_b2 = cluster_bootstrap_ci(mixed_b, n_boot=2000, seed=BOOTSTRAP_SEED)

    width_a = hi_a2 - lo_a2
    width_b = hi_b2 - lo_b2
    # Cluster on date: B has 1 cluster, so resampled mean is constant → CI width ≈ 0.
    # A has 30 clusters with mixed outcomes, so CI width is large.
    assert width_b == pytest.approx(0.0, abs=1e-9), (
        f"single-cluster data should collapse cluster-bootstrap CI to a point; "
        f"got width={width_b} (lo={lo_b2}, hi={hi_b2}). If this fails, the "
        "resampling unit may have regressed to per-row instead of per-date."
    )
    assert width_a > 0.10, (
        f"multi-cluster data should give a non-trivial CI; got width={width_a}. "
        "If this fails, the cluster bootstrap is not actually varying across clusters."
    )


# ---------------------------------------------------------------------------
# Threshold sanity (cheap guardrails — fail-loud if config drifts)
# ---------------------------------------------------------------------------


def test_thresholds_match_spec() -> None:
    """Spec (F1.4 in 2026-05-19-adversarial-review-open-items.md):
    n >= 30 AND |empirical - plateau| < 0.10."""
    assert MIN_HELD_OUT_N == 30
    assert MAX_PLATEAU_DEVIATION == 0.10


# ---------------------------------------------------------------------------
# Floor / curve-bottom coupling gate (added 2026-05-20 after a live floor-bypass incident)
#
# When ``artifact["values"][0]`` (the calibrated probability at raw=0) creeps
# above the live ``min_calibrated_prob`` floor, the floor stops biting and
# trades that were intended to be suppressed start firing. One promote
# shipped an artifact with ``values[0]=0.1153`` against a ``0.10`` floor and
# four long-shot YES trades fired before the revert. This gate
# fails the promote when ``values[0] > min_calibrated_prob - epsilon``.
# ---------------------------------------------------------------------------


def test_floor_coupling_passes_on_live_artifact() -> None:
    """Live ``data/calibration/isotonic.json`` (``values[0]=0.07967``) must pass
    against the configured floor ``0.10`` with the default epsilon (0.01)."""
    result = floor_coupling_check(
        values_zero=0.07967,
        min_calibrated_prob=0.10,
        epsilon=FLOOR_DEFEAT_EPSILON,
    )
    assert result["status"] == "ok", result["message"]
    assert result["values_zero"] == 0.07967
    assert result["threshold"] == pytest.approx(0.10 - FLOOR_DEFEAT_EPSILON)


def test_floor_coupling_fails_on_bad_artifact() -> None:
    """Bad ``data/calibration/isotonic.proposed_2026-05-19.json`` (``values[0]=
    0.115261``) must fail against the configured floor ``0.10``."""
    result = floor_coupling_check(
        values_zero=0.115261,
        min_calibrated_prob=0.10,
        epsilon=FLOOR_DEFEAT_EPSILON,
    )
    assert result["status"] == "fail", result["message"]
    assert "0.115261" in result["message"]
    assert "0.10" in result["message"] or "0.100" in result["message"]
    # Message must point the operator at a remediation.
    assert "min_calibrated_prob" in result["message"]


def test_floor_coupling_passes_when_values_zero_equals_threshold() -> None:
    """Strict ``>`` comparison: ``values[0] == floor - epsilon`` is the boundary
    and must be admitted (we only reject curves that strictly exceed it)."""
    result = floor_coupling_check(
        values_zero=0.09,
        min_calibrated_prob=0.10,
        epsilon=FLOOR_DEFEAT_EPSILON,
    )
    assert result["status"] == "ok"


def test_floor_coupling_disabled_when_floor_is_none() -> None:
    """``min_calibrated_prob: null`` in config disables the floor entirely; the
    gate must be a no-op (``status='skipped'``) in that case so the absence of
    a configured floor doesn't masquerade as a passing check."""
    result = floor_coupling_check(
        values_zero=0.5,
        min_calibrated_prob=None,
        epsilon=FLOOR_DEFEAT_EPSILON,
    )
    assert result["status"] == "skipped"


def test_floor_coupling_default_epsilon_is_one_percent() -> None:
    """Spec: epsilon = 0.01. Keep the threshold visible to operators."""
    assert FLOOR_DEFEAT_EPSILON == 0.01


def test_floor_coupling_cli_fails_on_bad_artifact(tmp_path: Path) -> None:
    """End-to-end: the bad artifact (``values[0]=0.115261``) drives the CLI to
    exit code 5 with a clear message naming the floor and the curve bottom."""
    bad_path = _PROJECT_ROOT / "data" / "calibration" / "isotonic.proposed_2026-05-19.json"
    config_path = _PROJECT_ROOT / "python" / "config.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--isotonic",
            str(bad_path),
            "--config",
            str(config_path),
            "--skip-held-out-check",
        ],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert completed.returncode == 5, (
        f"expected exit 5 (floor defeated); got {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
    combined = completed.stdout + completed.stderr
    assert "values[0]" in combined or "curve-bottom" in combined.lower()
    assert "min_calibrated_prob" in combined


def test_floor_coupling_cli_passes_on_live_artifact(tmp_path: Path) -> None:
    """End-to-end: the reverted live artifact (``values[0]=0.07967``) must NOT
    trip the floor-coupling gate. Held-out check skipped so this CLI run does
    not depend on snapshots being present in tmp_path."""
    good_path = _PROJECT_ROOT / "data" / "calibration" / "isotonic.json"
    config_path = _PROJECT_ROOT / "python" / "config.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--isotonic",
            str(good_path),
            "--config",
            str(config_path),
            "--skip-held-out-check",
        ],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    # Exit 0 is the "plateau found + skip held-out" happy path; exit 5 would
    # signal a regression of this gate.
    assert completed.returncode != 5, (
        f"floor-coupling gate spuriously failed on the live artifact\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
    )
