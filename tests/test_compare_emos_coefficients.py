"""Tests for scripts.compare_emos_coefficients."""
from __future__ import annotations

import math

import pytest


class TestResolveOutcome:
    def test_between_observed_in_range_returns_1(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=68.0, cap=72.0, observed=70.0) == 1

    def test_between_observed_at_floor_returns_1(self):
        # Kalshi convention: floor <= obs <= cap → inclusive on both ends.
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=68.0, cap=72.0, observed=68.0) == 1

    def test_between_observed_at_cap_returns_1(self):
        # Kalshi convention: between is CLOSED right-inclusive (42/42 audit
        # 2026-05-19, see python/analytics/outcomes.py). Prior to the
        # outcome-bug-fix this test asserted 0, which encoded the bug.
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=68.0, cap=72.0, observed=72.0) == 1

    def test_between_observed_below_floor_returns_0(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=68.0, cap=72.0, observed=67.0) == 0

    # Canonical bug-example replays (audit 2026-05-19; all three settled YES on Kalshi).
    def test_lax_high_may17_b70_5_settles_yes(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=70.0, cap=71.0, observed=71.0) == 1

    def test_aus_low_may16_b68_5_settles_yes(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=68.0, cap=69.0, observed=69.0) == 1

    def test_ny_high_may15_b65_5_settles_yes(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("between", floor=65.0, cap=66.0, observed=66.0) == 1

    def test_greater_observed_above_returns_1(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("greater", floor=70.0, cap=None, observed=72.0) == 1

    def test_greater_observed_at_floor_returns_0(self):
        # Kalshi rule: t > floor (STRICT). Verified 2026-05-19 via Kalshi
        # API rules_primary ("greater than X°") and empirical settlement
        # of KXHIGHMIA-26MAY08-T93 (NO with actual=93).
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("greater", floor=70.0, cap=None, observed=70.0) == 0

    def test_greater_observed_below_returns_0(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("greater", floor=70.0, cap=None, observed=69.0) == 0

    def test_less_observed_below_returns_1(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("less", floor=None, cap=70.0, observed=69.0) == 1

    def test_less_observed_at_cap_returns_0(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("less", floor=None, cap=70.0, observed=70.0) == 0

    def test_less_observed_above_returns_0(self):
        from scripts.compare_emos_coefficients import resolve_outcome
        assert resolve_outcome("less", floor=None, cap=70.0, observed=71.0) == 0


class TestBootstrapBSSCI:
    def test_identical_arrays_yield_zero_ci_with_zero_width(self):
        # If candidate == prod, BSS = 0 on every resample → CI is [0, 0].
        from scripts.compare_emos_coefficients import bootstrap_bss_ci
        sq_err = [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # 50 rows
        lo, hi = bootstrap_bss_ci(
            sq_err_prod=sq_err, sq_err_cand=sq_err, n_iters=500, seed=42
        )
        assert math.isclose(lo, 0.0, abs_tol=1e-12)
        assert math.isclose(hi, 0.0, abs_tol=1e-12)

    def test_uniformly_better_candidate_yields_positive_lower_bound(self):
        # Candidate errors are uniformly half of prod errors → BSS ≈ 0.75 (since (0.5*err)^2 = 0.25*err^2).
        from scripts.compare_emos_coefficients import bootstrap_bss_ci
        sq_err_prod = [0.2, 0.3, 0.4, 0.5, 0.6] * 10
        sq_err_cand = [e * 0.25 for e in sq_err_prod]  # candidate squared-error is 25% of prod
        lo, hi = bootstrap_bss_ci(
            sq_err_prod=sq_err_prod, sq_err_cand=sq_err_cand, n_iters=500, seed=42
        )
        # BSS = 1 - 0.25 = 0.75; bootstrap CI should bracket it with positive lower bound.
        assert lo > 0.5
        assert hi < 0.95

    def test_seeded_reproducible(self):
        from scripts.compare_emos_coefficients import bootstrap_bss_ci
        # Use enough variance that resampling actually produces different means.
        sq_err_prod = [0.1, 0.2, 0.3, 0.4, 0.5] * 4
        sq_err_cand = [0.05, 0.15, 0.25, 0.35, 0.45] * 4
        ci1 = bootstrap_bss_ci(sq_err_prod, sq_err_cand, n_iters=500, seed=123)
        ci2 = bootstrap_bss_ci(sq_err_prod, sq_err_cand, n_iters=500, seed=123)
        assert ci1 == ci2


class TestMuShiftDrift:
    def test_identical_coeffs_yield_zero_drift(self):
        from scripts.compare_emos_coefficients import compute_mu_shift_drift
        ens_means = [60.0, 65.0, 70.0, 75.0, 80.0]
        drift = compute_mu_shift_drift(
            ens_means,
            prod_a=1.0, prod_b=1.0,
            cand_a=1.0, cand_b=1.0,
        )
        assert math.isclose(drift, 0.0, abs_tol=1e-12)

    def test_a_shift_only(self):
        # μ_shift_prod = 0 + (1-1)·mean = 0; μ_shift_cand = 2 + (1-1)·mean = 2 → |drift| = 2.
        from scripts.compare_emos_coefficients import compute_mu_shift_drift
        ens_means = [60.0, 70.0, 80.0]
        drift = compute_mu_shift_drift(
            ens_means,
            prod_a=0.0, prod_b=1.0,
            cand_a=2.0, cand_b=1.0,
        )
        assert math.isclose(drift, 2.0, abs_tol=1e-9)

    def test_b_shift_at_typical_mean(self):
        # prod: μ_shift = 0 + (1.0-1)·70 = 0
        # cand: μ_shift = 0 + (0.9-1)·70 = -7 → median(|0 - (-7)|) = 7
        from scripts.compare_emos_coefficients import compute_mu_shift_drift
        ens_means = [60.0, 70.0, 80.0]
        drift = compute_mu_shift_drift(
            ens_means,
            prod_a=0.0, prod_b=1.0,
            cand_a=0.0, cand_b=0.9,
        )
        # Per-row drifts: |0 - (-6)|=6, |0 - (-7)|=7, |0 - (-8)|=8. Median = 7.
        assert math.isclose(drift, 7.0, abs_tol=1e-9)


class TestDecideCell:
    def test_promote_when_all_pass(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "PROMOTE"

    def test_keep_prod_when_bss_negative(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=-0.01, bss_ci_lower=-0.10,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "KEEP_PROD"

    def test_keep_prod_when_ci_lower_too_negative(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.20,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "KEEP_PROD"

    def test_keep_prod_when_mu_drift_exceeds_bound(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=1.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "KEEP_PROD"

    def test_keep_prod_when_new_warnings(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=True,
        )
        assert decision == "KEEP_PROD"

    def test_insufficient_n_when_below_min(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=10, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "INSUFFICIENT_N"


class TestSquaredErrorsPerCell:
    def test_groups_rows_by_cell_and_computes_squared_errors(self):
        """A handful of synthetic settled rows should produce per-cell squared-error lists."""
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_squared_errors_per_cell

        coeffs_table = {
            "MIA/high": EmosCoefficients(
                a=0.0, b=1.0, c=1.0, d=0.0,
                n_train=100, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
            ),
        }
        # Row 1: MIA/high between 68-72 obs=70 → prob ≈ 0.9544, outcome=1 → sq_err ≈ 0.00208
        # Row 2: MIA/high between 68-72 obs=65 → outcome=0; same prob → sq_err ≈ 0.9109
        rows = [
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 68.0, "cap": 72.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 1.0,
                "observed": 70.0,
            },
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 68.0, "cap": 72.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 1.0,
                "observed": 65.0,
            },
        ]
        result = compute_squared_errors_per_cell(rows, coeffs_table)
        assert "MIA/high" in result
        assert len(result["MIA/high"]) == 2
        # Both rows have the same prob (≈0.9544); outcomes 1 and 0.
        assert math.isclose(result["MIA/high"][0], (1 - 0.9544997) ** 2, rel_tol=1e-3)
        assert math.isclose(result["MIA/high"][1], (0 - 0.9544997) ** 2, rel_tol=1e-3)

    def test_skips_cells_not_in_coeffs_table(self):
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_squared_errors_per_cell

        coeffs_table = {
            "MIA/high": EmosCoefficients(
                a=0.0, b=1.0, c=1.0, d=0.0,
                n_train=100, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
            ),
        }
        rows = [
            {  # Coefficients exist
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 68.0, "cap": 72.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 1.0, "observed": 70.0,
            },
            {  # No coefficients for SEA/low — skip silently
                "city": "SEA", "var": "low",
                "strike_type": "less", "floor": None, "cap": 50.0,
                "ens_mean_raw": 48.0, "ens_std_raw": 2.0, "observed": 49.0,
            },
        ]
        result = compute_squared_errors_per_cell(rows, coeffs_table)
        assert list(result.keys()) == ["MIA/high"]
        assert len(result["MIA/high"]) == 1


class TestBriefIntegration:
    """Sanity test: identical prod vs candidate coefficients should yield BSS=0 exactly."""

    def test_identical_coeffs_yield_zero_bss(self):
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_squared_errors_per_cell

        c = EmosCoefficients(
            a=1.0, b=0.95, c=2.0, d=0.5,
            n_train=100, low_n=False, ci_warnings=[],
            leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
        )
        coeffs_table = {"NYC/low": c}
        rows = [
            {"city": "NYC", "var": "low", "strike_type": "less",
             "floor": None, "cap": 45.0,
             "ens_mean_raw": 42.0 + i, "ens_std_raw": 1.5, "observed": 43.0 + i}
            for i in range(10)
        ]
        sq_prod = compute_squared_errors_per_cell(rows, coeffs_table)["NYC/low"]
        sq_cand = compute_squared_errors_per_cell(rows, coeffs_table)["NYC/low"]
        # When identical, mean(sq_cand) == mean(sq_prod), so BSS = 0.
        assert sq_prod == sq_cand
        m_prod = sum(sq_prod) / len(sq_prod)
        m_cand = sum(sq_cand) / len(sq_cand)
        bss = 1 - m_cand / m_prod if m_prod > 0 else 0.0
        assert math.isclose(bss, 0.0, abs_tol=1e-12)


@pytest.fixture
def tmp_forecasts_raw(tmp_path):
    """Write a tiny forecasts_raw.jsonl fixture for join testing."""
    import json
    path = tmp_path / "forecasts_raw.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({
            "scan_id": "scan-1",
            "forecast_date": "2026-05-04",
            "city": "MIA", "var": "high",
            "gfs_ens_mean_raw_ddof1": 85.0,
            "gfs_ens_std_raw_ddof1": 1.2,
        }) + "\n")
        f.write(json.dumps({
            "scan_id": "scan-2",
            "forecast_date": "2026-05-05",
            "city": "NYC", "var": "low",
            "gfs_ens_mean_raw_ddof1": 52.0,
            "gfs_ens_std_raw_ddof1": 2.1,
        }) + "\n")
    return path


class TestForecastsRawIndex:
    def test_builds_index_keyed_by_city_var_date_scan(self, tmp_forecasts_raw):
        from scripts.compare_emos_coefficients import load_forecasts_raw_index
        idx = load_forecasts_raw_index(tmp_forecasts_raw)
        assert idx[("MIA", "high", "2026-05-04", "scan-1")] == (85.0, 1.2)
        assert idx[("NYC", "low", "2026-05-05", "scan-2")] == (52.0, 2.1)
        assert len(idx) == 2


class TestIsIsotonicBaselineRow:
    """The Stage B filter must include pre-cutover rows where
    ``live_calibrator_kind`` is absent (the field was added
    on 2026-05-11; before that, isotonic was always the live calibrator on
    any row that reached the EMOS shadow block)."""

    def test_explicit_isotonic_kind_returns_true(self):
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row(
            {"live_calibrator_kind": "isotonic", "forecast_date": "2026-05-12"}
        ) is True

    def test_explicit_isotonic_fallback_kind_returns_true(self):
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row(
            {"live_calibrator_kind": "isotonic_fallback", "forecast_date": "2026-05-12"}
        ) is True

    def test_emos_raw_returns_false(self):
        # EMOS was live; including would double-count candidate EMOS against itself.
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row(
            {"live_calibrator_kind": "emos_raw", "forecast_date": "2026-05-12"}
        ) is False
        # Even pre-cutover, an explicit kind is honored.
        assert _is_isotonic_baseline_row(
            {"live_calibrator_kind": "emos_raw", "forecast_date": "2026-05-10"}
        ) is False

    def test_identity_kind_returns_false(self):
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row(
            {"live_calibrator_kind": "identity", "forecast_date": "2026-05-10"}
        ) is False

    def test_missing_kind_pre_cutover_returns_true(self):
        # The bug we're fixing: pre-cutover rows have kind absent because the
        # field hadn't been added yet, but isotonic was the live calibrator.
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row({"forecast_date": "2026-05-10"}) is True

    def test_missing_kind_at_cutover_returns_false(self):
        # Conservatively dropped — could be from a pre-deploy scan that day
        # (isotonic fired) or a schema bug post-deploy (unknown). Handful of
        # rows; not worth misclassifying.
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row({"forecast_date": "2026-05-11"}) is False

    def test_missing_kind_post_cutover_returns_false(self):
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row({"forecast_date": "2026-05-12"}) is False

    def test_missing_kind_and_missing_date_returns_false(self):
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row({}) is False
        assert _is_isotonic_baseline_row({"forecast_date": None}) is False

    def test_missing_kind_with_non_iso_date_returns_false(self):
        # ``dt.date.fromisoformat`` raises on non-ISO strings; the helper
        # catches and returns False rather than silently lex-comparing wrong.
        from scripts.compare_emos_coefficients import _is_isotonic_baseline_row
        assert _is_isotonic_baseline_row({"forecast_date": "26MAY10"}) is False
        assert _is_isotonic_baseline_row({"forecast_date": "2026-05-10T12:00:00"}) is False


class TestLogLossesPerCell:
    """Per-cell mean-NLL accumulator — analogous to compute_squared_errors_per_cell
    but for the logarithmic scoring rule (Kelly-aligned proper score).

    Numerical safety: emos_probability can return exact 0 or 1 in tails,
    which would crash on log(0). The implementation must clip to
    [eps, 1-eps] before log."""

    def test_groups_rows_by_cell_and_computes_log_losses(self):
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_log_losses_per_cell

        coeffs_table = {
            "MIA/high": EmosCoefficients(
                a=0.0, b=1.0, c=1.0, d=0.0,
                n_train=100, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
            ),
        }
        rows = [
            # prob ≈ 0.9544, outcome=1 → -log(0.9544) ≈ 0.0466
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 68.0, "cap": 72.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 1.0,
                "observed": 70.0,
            },
            # prob ≈ 0.9544, outcome=0 → -log(1-0.9544) ≈ 3.087
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 68.0, "cap": 72.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 1.0,
                "observed": 65.0,
            },
        ]
        result = compute_log_losses_per_cell(rows, coeffs_table)
        assert "MIA/high" in result
        assert len(result["MIA/high"]) == 2
        assert math.isclose(result["MIA/high"][0], -math.log(0.9544997), rel_tol=1e-3)
        assert math.isclose(result["MIA/high"][1], -math.log(1 - 0.9544997), rel_tol=1e-3)

    def test_clips_extreme_probabilities_to_avoid_log_zero(self):
        """A coefficient set producing p=1.0 on a NO outcome must not crash."""
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_log_losses_per_cell

        # Tiny sigma + strike far from mean → p saturates near 1.0.
        coeffs_table = {
            "MIA/high": EmosCoefficients(
                a=0.0, b=1.0, c=1e-6, d=0.0,
                n_train=100, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=5.0,
            ),
        }
        rows = [
            {
                "city": "MIA", "var": "high",
                "strike_type": "less", "floor": None, "cap": 100.0,
                "ens_mean_raw": 70.0, "ens_std_raw": 0.001,
                "observed": 110.0,  # outcome=0 even though model says ~1.0
            },
        ]
        result = compute_log_losses_per_cell(rows, coeffs_table)
        nll = result["MIA/high"][0]
        assert math.isfinite(nll)
        # Should be a large finite penalty (≈ -log(eps)).
        assert nll > 10.0

    def test_nan_probability_raises_loudly(self, monkeypatch):
        """Audit-2026-05-23 (code-auditor HIGH #2): NaN must raise, not be
        absorbed by the clip. Forces emos_probability to return NaN and
        confirms ValueError, so a future upstream regression cannot silently
        mask itself as a large log-loss."""
        from python.data import emos as emos_mod
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import compute_log_losses_per_cell

        def _nan_prob(*_args, **_kwargs):
            return (float("nan"), 0.0, 0.0)

        monkeypatch.setattr(
            "scripts.compare_emos_coefficients.emos_probability", _nan_prob
        )
        _ = emos_mod  # silence unused import lint; the patch keys off scripts module
        coeffs_table = {
            "MIA/high": EmosCoefficients(
                a=0.0, b=1.0, c=1.0, d=0.0,
                n_train=10, low_n=False, ci_warnings=[],
                leads_pooled=[0], by_lead=None, ens_std_p99=2.0,
            ),
        }
        rows = [{
            "city": "MIA", "var": "high",
            "strike_type": "between", "floor": 68.0, "cap": 72.0,
            "ens_mean_raw": 70.0, "ens_std_raw": 1.0,
            "observed": 70.0,
        }]
        with pytest.raises(ValueError, match="non-finite"):
            compute_log_losses_per_cell(rows, coeffs_table)


class TestDecideCellLogSkillFloorBoundary:
    """Pin the floor-boundary semantics: gate is `<`, not `<=`, so a
    LogSkill CI lower bound exactly equal to the floor is admitted.
    Default floor lowered to -0.02 (2026-05-23 calibration audit)."""

    def test_log_skill_ci_lower_exactly_at_default_floor_is_admitted(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
            log_skill_point=0.01,
            log_skill_ci_lower=-0.02,  # exactly at default floor
        )
        assert decision == "PROMOTE"

    def test_log_skill_ci_lower_one_epsilon_below_floor_rejects(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
            log_skill_point=0.01,
            log_skill_ci_lower=-0.0201,
        )
        assert decision == "KEEP_PROD"


class TestBootstrapLogSkillCI:
    """Paired bootstrap on LogSkill = 1 - mean(NLL_cand) / mean(NLL_baseline).
    Pairs prod/cand at the row level so the comparison is well-defined."""

    def test_identical_arrays_yield_zero_ci(self):
        from scripts.compare_emos_coefficients import bootstrap_log_skill_ci
        nll = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
        lo, hi = bootstrap_log_skill_ci(
            nll_prod=nll, nll_cand=nll, n_iters=500, seed=42
        )
        assert math.isclose(lo, 0.0, abs_tol=1e-12)
        assert math.isclose(hi, 0.0, abs_tol=1e-12)

    def test_uniformly_better_candidate_yields_positive_lower_bound(self):
        from scripts.compare_emos_coefficients import bootstrap_log_skill_ci
        nll_prod = [0.5, 0.6, 0.7, 0.8, 0.9] * 10
        nll_cand = [n * 0.5 for n in nll_prod]  # halved NLL → LogSkill ≈ 0.5
        lo, hi = bootstrap_log_skill_ci(
            nll_prod=nll_prod, nll_cand=nll_cand, n_iters=500, seed=42
        )
        assert lo > 0.4
        assert hi < 0.6

    def test_worse_candidate_yields_negative_upper_bound(self):
        from scripts.compare_emos_coefficients import bootstrap_log_skill_ci
        nll_prod = [0.5, 0.6, 0.7, 0.8, 0.9] * 10
        nll_cand = [n * 2.0 for n in nll_prod]  # doubled NLL → LogSkill ≈ -1.0
        lo, hi = bootstrap_log_skill_ci(
            nll_prod=nll_prod, nll_cand=nll_cand, n_iters=500, seed=42
        )
        assert hi < -0.5

    def test_seeded_reproducible(self):
        from scripts.compare_emos_coefficients import bootstrap_log_skill_ci
        nll_prod = [0.3, 0.4, 0.5, 0.6, 0.7] * 4
        nll_cand = [0.2, 0.3, 0.4, 0.5, 0.6] * 4
        ci1 = bootstrap_log_skill_ci(nll_prod, nll_cand, n_iters=500, seed=99)
        ci2 = bootstrap_log_skill_ci(nll_prod, nll_cand, n_iters=500, seed=99)
        assert ci1 == ci2

    def test_unequal_lengths_raises(self):
        from scripts.compare_emos_coefficients import bootstrap_log_skill_ci
        with pytest.raises(ValueError):
            bootstrap_log_skill_ci([0.1, 0.2], [0.3], n_iters=10, seed=0)


class TestDecideCellWithLogSkill:
    """Promotion gate must require BOTH BSS and LogSkill CI lower bounds to pass.
    LogSkill catches Kelly-aligned mis-calibration (esp. tail bias) that
    squared-error BSS can under-weight."""

    def test_promote_when_both_bss_and_log_skill_pass(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
            log_skill_point=0.04,
            log_skill_ci_lower=-0.02,
        )
        assert decision == "PROMOTE"

    def test_keep_prod_when_log_skill_negative_despite_positive_bss(self):
        """The exact failure mode log-score is being added to catch: candidate
        is squared-error competitive but log-loss worse (asymmetric mis-cal)."""
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
            log_skill_point=-0.01,
            log_skill_ci_lower=-0.10,
        )
        assert decision == "KEEP_PROD"

    def test_keep_prod_when_log_skill_ci_too_negative(self):
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
            log_skill_point=0.04,
            log_skill_ci_lower=-0.20,
        )
        assert decision == "KEEP_PROD"

    def test_backward_compat_when_log_skill_args_omitted(self):
        """Pre-existing call sites that don't pass log_skill_* must still
        receive a BSS-only decision (and the existing test class above proves
        this on all the legacy cases)."""
        from scripts.compare_emos_coefficients import decide_cell
        decision = decide_cell(
            n=20, n_min=15,
            bss_point=0.05, bss_ci_lower=-0.02,
            mu_drift=0.5, mu_drift_max=1.0,
            has_new_warnings=False,
        )
        assert decision == "PROMOTE"


class TestBuildStageBRowRegression:
    """Regression test for the empty-Stage-B bug. Pre-cutover audit rows have
    ``live_calibrator_kind`` absent because the field was added
    on 2026-05-11. The original Stage B filter required kind ∈
    {isotonic, isotonic_fallback} and silently dropped every pre-cutover row.

    Without this test, a future revert to that filter passes all the pure
    ``_is_isotonic_baseline_row`` unit tests and silently re-introduces the
    bug — Stage B section becomes empty even with thousands of valid rows."""

    def test_stage_b_populated_when_only_pre_cutover_rows_exist(self):
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import _build_stage_b_row

        cand_table = {
            "MIA/high": EmosCoefficients(
                a=4.0, b=1.0, c=0.5, d=1.0,
                n_train=145, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=2.0,
            ),
        }
        # Three pre-cutover rows. ``live_calibrator_kind`` is absent — under
        # the OLD filter all three would be dropped and the Stage B row would
        # be None. Under the fixed helper they are recognized as isotonic
        # baseline rows because forecast_date < 2026-05-11.
        settled_rows = [
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 88.0, "cap": 92.0,
                "ens_mean_raw": 90.0, "ens_std_raw": 1.0,
                "observed": 89.0,
                "calibrated_prob": 0.7,
                "forecast_date": "2026-05-10",
                # live_calibrator_kind intentionally absent
            },
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 90.0, "cap": 94.0,
                "ens_mean_raw": 91.0, "ens_std_raw": 1.0,
                "observed": 92.0,
                "calibrated_prob": 0.55,
                "forecast_date": "2026-05-09",
            },
            {
                "city": "MIA", "var": "high",
                "strike_type": "greater", "floor": 88.0, "cap": None,
                "ens_mean_raw": 90.0, "ens_std_raw": 1.5,
                "observed": 91.0,
                "calibrated_prob": 0.8,
                "forecast_date": "2026-05-08",
            },
        ]

        row = _build_stage_b_row(
            "MIA/high",
            settled_rows,
            cand_table,
            cand_warnings=set(),
            bootstrap_iters=200,
            seed=42,
        )

        # Under the buggy pre-fix filter this would be None.
        assert row is not None, "Stage B row should be populated for pre-cutover isotonic rows"
        assert row["cell"] == "MIA/high"
        assert row["n"] == 3
        # The cell will register INSUFFICIENT_N because n_min_iso defaults to 10,
        # but the row itself exists — that's the regression we're pinning.
        assert row["stage_b"] == "INSUFFICIENT_N"

    def test_stage_b_excludes_emos_raw_rows(self):
        """A row with live_calibrator_kind=emos_raw must NOT contribute to the
        Stage B isotonic baseline — that would double-count candidate EMOS
        against its own output."""
        from python.data.emos import EmosCoefficients
        from scripts.compare_emos_coefficients import _build_stage_b_row

        cand_table = {
            "MIA/high": EmosCoefficients(
                a=4.0, b=1.0, c=0.5, d=1.0,
                n_train=145, low_n=False, ci_warnings=[],
                leads_pooled=[0, 1, 2], by_lead=None, ens_std_p99=2.0,
            ),
        }
        settled_rows = [
            {
                "city": "MIA", "var": "high",
                "strike_type": "between", "floor": 88.0, "cap": 92.0,
                "ens_mean_raw": 90.0, "ens_std_raw": 1.0,
                "observed": 89.0,
                "calibrated_prob": 0.7,
                "forecast_date": "2026-05-12",
                "live_calibrator_kind": "emos_raw",
            },
        ]
        row = _build_stage_b_row(
            "MIA/high",
            settled_rows,
            cand_table,
            cand_warnings=set(),
            bootstrap_iters=200,
            seed=42,
        )
        # No isotonic rows → None
        assert row is None
