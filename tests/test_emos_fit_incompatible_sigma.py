"""Tests for the fit-time σ/bracket-width incompatibility gate.

The gate detects EMOS fits whose predictive σ at high ens_std (`ens_std_p99`)
exceeds a multiple of the trading bracket width. When `σ_emos / bracket_width`
is large, the predicted Gaussian is essentially uniform across any single
bracket and calibration is structurally meaningless at that geometry. This
class of failure was missed by the per-cell promotion gate on 2026-05-11
(LAX/high promoted under `unstable_d` override, rolled back 2026-05-18 after
live ROI=-39.6%). See the original prompt's structural derivation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from python.analytics.emos_fit import (
    EmosFit,
    bootstrap_cell,
    sigma_emos_at,
    sigma_to_bracket_ratio,
)

# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #


def test_sigma_emos_at_matches_lax_high_known_value():
    """LAX/high coefficients (2026-05-11 fit) must reproduce
    the structurally-incompatible σ_emos ≈ 5.76°F at ens_std_p99 = 2.123°F.

    This is the canonical failure mode that motivated the gate:
    σ_emos = √(c + d · ens_std²) = √(0.770 + 7.188 · 2.123²) ≈ 5.76°F.
    With 1°F Kalshi brackets that is ~5.76x bracket width.
    """
    c = 0.7702674542818296
    d = 7.187684592929092
    ens_std_p99 = 2.122645496505716
    sigma = sigma_emos_at(c=c, d=d, ens_std=ens_std_p99)
    assert math.isclose(sigma, math.sqrt(c + d * ens_std_p99**2))
    assert 5.74 < sigma < 5.78


def test_sigma_emos_at_handles_d_zero():
    """Floor cell (d=0) → σ_emos = √c, independent of ens_std."""
    # DEN/low: c=17.999, d=0
    sigma = sigma_emos_at(c=17.999138727585066, d=0.0, ens_std=4.288)
    assert math.isclose(sigma, math.sqrt(17.999138727585066), abs_tol=1e-9)


def test_sigma_emos_at_rejects_negative_c_plus_d_eps():
    """The σ²-floor (1e-12) is preserved; negative arguments raise."""
    with pytest.raises(ValueError, match="non-negative"):
        sigma_emos_at(c=-1.0, d=0.0, ens_std=1.0)


def test_sigma_to_bracket_ratio_scaling():
    """`σ / bracket_width` must scale inversely with bracket_width.

    σ_emos = √(1 + 4·1²) = √5 ≈ 2.236.
    bracket_width 1.0 → 2.236; 2.0 → 1.118; 0.5 → 4.472.
    """
    ratio_1 = sigma_to_bracket_ratio(c=1.0, d=4.0, ens_std=1.0, bracket_width=1.0)
    ratio_2 = sigma_to_bracket_ratio(c=1.0, d=4.0, ens_std=1.0, bracket_width=2.0)
    ratio_half = sigma_to_bracket_ratio(c=1.0, d=4.0, ens_std=1.0, bracket_width=0.5)
    assert math.isclose(ratio_1, math.sqrt(5.0), abs_tol=1e-9)
    assert math.isclose(ratio_2, math.sqrt(5.0) / 2.0, abs_tol=1e-9)
    assert math.isclose(ratio_half, math.sqrt(5.0) * 2.0, abs_tol=1e-9)


def test_sigma_to_bracket_ratio_rejects_nonpositive_bracket_width():
    with pytest.raises(ValueError, match="bracket_width"):
        sigma_to_bracket_ratio(c=1.0, d=1.0, ens_std=1.0, bracket_width=0.0)


# --------------------------------------------------------------------------- #
# incompatible_sigma bootstrap warning
# --------------------------------------------------------------------------- #


def _fake_fit_factory(samples):
    """Return a fake fit_cell whose i-th call returns samples[i]."""
    call_idx = {"i": 0}

    def fake_fit(*_a, **_k):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = samples[i % len(samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    return fake_fit


def test_bootstrap_cell_emits_incompatible_sigma_above_threshold(monkeypatch):
    """When σ_emos(p99) / bracket_width > threshold, `incompatible_sigma` fires.

    Fake fit returns c≈0.77, d≈7.19 (LAX/high-like). ens_std_p99 will be ~2.12
    by construction of the input array. σ_emos ≈ 5.76 → ratio 5.76 at
    bracket_width=1.0, well above the default threshold of 2.0.
    """
    from python.analytics import emos_fit as _module

    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.full(400, 0.77),
            np.full(400, 7.19),
        ]
    )
    monkeypatch.setattr(_module, "fit_cell", _fake_fit_factory(samples))

    # Input ens_std distribution whose p99 is ~2.12 (LAX/high-like).
    rng = np.random.default_rng(0)
    ens_std = np.abs(rng.normal(1.5, 0.3, size=400)) + 0.1
    ens_std = np.clip(ens_std, 0.1, 2.12)

    ci = bootstrap_cell(
        np.zeros(400),
        ens_std,
        np.zeros(400),
        n_iter=400,
        seed=42,
        bracket_width=1.0,
        incompatible_sigma_threshold=2.0,
    )

    assert "incompatible_sigma" in ci["ci_warnings"]


def test_bootstrap_cell_no_incompatible_sigma_below_threshold(monkeypatch):
    """A well-behaved fit (c≈0.5, d≈1.0, ens_std_p99≈2) produces σ≈√4.5≈2.12.
    At bracket_width=1.0 ratio ≈2.12 which is just over default threshold 2.0;
    raise threshold to 3.0 → warning must NOT fire.
    """
    from python.analytics import emos_fit as _module

    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.full(400, 0.5),
            np.full(400, 1.0),
        ]
    )
    monkeypatch.setattr(_module, "fit_cell", _fake_fit_factory(samples))

    rng = np.random.default_rng(1)
    ens_std = np.abs(rng.normal(1.0, 0.3, size=400)) + 0.1
    ens_std = np.clip(ens_std, 0.1, 2.0)

    ci = bootstrap_cell(
        np.zeros(400),
        ens_std,
        np.zeros(400),
        n_iter=400,
        seed=7,
        bracket_width=1.0,
        incompatible_sigma_threshold=3.0,
    )

    assert "incompatible_sigma" not in ci["ci_warnings"]


def test_bootstrap_cell_incompatible_sigma_respects_bracket_width(monkeypatch):
    """Intraday strategies trade wider brackets (e.g., 2°F or 5°F). The same
    σ_emos ≈ 5.76 that fails for 1°F brackets is acceptable at 5°F brackets
    (ratio ~1.15). Gate must not over-fire at wider bracket widths.
    """
    from python.analytics import emos_fit as _module

    # LAX/high-like fit.
    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.full(400, 0.77),
            np.full(400, 7.19),
        ]
    )
    monkeypatch.setattr(_module, "fit_cell", _fake_fit_factory(samples))

    rng = np.random.default_rng(0)
    ens_std = np.abs(rng.normal(1.5, 0.3, size=400)) + 0.1
    ens_std = np.clip(ens_std, 0.1, 2.12)

    # bracket_width=5 → ratio ≈ 5.76 / 5 ≈ 1.15, below default 2.0 threshold.
    ci_wide = bootstrap_cell(
        np.zeros(400),
        ens_std,
        np.zeros(400),
        n_iter=400,
        seed=42,
        bracket_width=5.0,
        incompatible_sigma_threshold=2.0,
    )
    assert "incompatible_sigma" not in ci_wide["ci_warnings"]


def test_bootstrap_cell_default_bracket_width_is_one_degree(monkeypatch):
    """Default bracket_width must be 1.0 (KXHIGH/KXLOWT geometry).
    Caller can omit the param and get current daily-strategy semantics.
    """
    from python.analytics import emos_fit as _module

    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.full(400, 0.77),
            np.full(400, 7.19),
        ]
    )
    monkeypatch.setattr(_module, "fit_cell", _fake_fit_factory(samples))

    rng = np.random.default_rng(0)
    ens_std = np.abs(rng.normal(1.5, 0.3, size=400)) + 0.1
    ens_std = np.clip(ens_std, 0.1, 2.12)

    # No bracket_width passed — must default to 1.0 and trigger warning.
    ci = bootstrap_cell(
        np.zeros(400),
        ens_std,
        np.zeros(400),
        n_iter=400,
        seed=42,
        incompatible_sigma_threshold=2.0,
    )
    assert "incompatible_sigma" in ci["ci_warnings"]


def test_bootstrap_cell_returns_sigma_diagnostics_fields(monkeypatch):
    """The bootstrap envelope must expose `sigma_emos_at_p50`,
    `sigma_emos_at_p99`, and `sigma_to_bracket_ratio_p99` so the diagnostics
    file can persist them per cell.
    """
    from python.analytics import emos_fit as _module

    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.full(400, 0.77),
            np.full(400, 7.19),
        ]
    )
    monkeypatch.setattr(_module, "fit_cell", _fake_fit_factory(samples))

    rng = np.random.default_rng(0)
    ens_std = np.abs(rng.normal(1.5, 0.3, size=400)) + 0.1
    ens_std = np.clip(ens_std, 0.1, 2.12)

    ci = bootstrap_cell(
        np.zeros(400),
        ens_std,
        np.zeros(400),
        n_iter=400,
        seed=42,
        bracket_width=1.0,
    )

    assert "sigma_emos_at_p50" in ci
    assert "sigma_emos_at_p99" in ci
    assert "sigma_to_bracket_ratio_p99" in ci
    # Numeric sanity — these are real numbers, not None.
    assert math.isfinite(ci["sigma_emos_at_p50"])
    assert math.isfinite(ci["sigma_emos_at_p99"])
    assert math.isfinite(ci["sigma_to_bracket_ratio_p99"])
    # Ratio at bracket_width=1.0 equals σ_emos(p99).
    assert math.isclose(
        ci["sigma_to_bracket_ratio_p99"], ci["sigma_emos_at_p99"], rel_tol=1e-9
    )


# --------------------------------------------------------------------------- #
# Promotion gate rejects incompatible_sigma
# --------------------------------------------------------------------------- #


def test_promotion_gate_rejects_incompatible_sigma_via_new_warnings():
    """`_stage_b_decision` (and `decide_cell`) reject the candidate when the
    `incompatible_sigma` warning is newly introduced (caught by the standard
    `has_new_warnings` path).
    """
    from scripts.compare_emos_coefficients import _stage_b_decision, decide_cell

    decision_stage_b = _stage_b_decision(
        n=100,
        n_min_iso=10,
        bss_vs_iso_point=0.20,
        bss_vs_iso_ci_lower=0.10,
        has_new_warnings=True,
        cand_warnings={"incompatible_sigma"},
    )
    assert decision_stage_b == "BLOCKED_WARNINGS"

    decision = decide_cell(
        n=100,
        n_min=15,
        bss_point=0.20,
        bss_ci_lower=0.10,
        mu_drift=0.5,
        mu_drift_max=1.0,
        has_new_warnings=True,
        cand_warnings={"incompatible_sigma"},
    )
    assert decision == "KEEP_PROD"


def test_promotion_gate_rejects_incompatible_sigma_even_when_prod_also_has_it():
    """`incompatible_sigma` is non-overridable: even if prod and candidate
    both carry it (so `has_new_warnings = False`), the candidate is rejected.

    This blocks a regression where re-fitting an already-incompatible cell
    would re-pass the gate because the warning is no longer "new".
    """
    from scripts.compare_emos_coefficients import _stage_b_decision, decide_cell

    decision_stage_b = _stage_b_decision(
        n=100,
        n_min_iso=10,
        bss_vs_iso_point=0.20,
        bss_vs_iso_ci_lower=0.10,
        has_new_warnings=False,  # warning present in BOTH — not "new"
        cand_warnings={"incompatible_sigma"},
    )
    assert decision_stage_b == "BLOCKED_WARNINGS"

    decision = decide_cell(
        n=100,
        n_min=15,
        bss_point=0.20,
        bss_ci_lower=0.10,
        mu_drift=0.5,
        mu_drift_max=1.0,
        has_new_warnings=False,
        cand_warnings={"incompatible_sigma"},
    )
    assert decision == "KEEP_PROD"


def test_promotion_gate_admits_unstable_d_when_prod_has_it_too():
    """Sanity check: `unstable_d` is overridable (was the LAX/high mechanism).
    When both prod and candidate carry it (so has_new_warnings=False) and all
    other gates pass, the candidate must still be PROMOTE-eligible. Confirms
    we did not over-tighten the gate.
    """
    from scripts.compare_emos_coefficients import _stage_b_decision, decide_cell

    decision_stage_b = _stage_b_decision(
        n=100,
        n_min_iso=10,
        bss_vs_iso_point=0.20,
        bss_vs_iso_ci_lower=0.10,
        has_new_warnings=False,
        cand_warnings={"unstable_d"},  # not in _NON_OVERRIDABLE_WARNINGS
    )
    assert decision_stage_b == "STAGE_B_ELIGIBLE"

    decision = decide_cell(
        n=100,
        n_min=15,
        bss_point=0.20,
        bss_ci_lower=0.10,
        mu_drift=0.5,
        mu_drift_max=1.0,
        has_new_warnings=False,
        cand_warnings={"unstable_d"},
    )
    assert decision == "PROMOTE"
