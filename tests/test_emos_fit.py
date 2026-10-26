"""Unit tests for python/analytics/emos_fit.py (pure functions)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from python.analytics.emos_fit import EmosFit, crps_gaussian, cv_fold_iter, fit_cell, skip_rule


def test_crps_gaussian_matches_numerical_integration():
    """Closed-form CRPS must match the integral definition.

    CRPS(F, y) = ∫ (F(x) - 1{x >= y})² dx where F is the predictive CDF.
    """
    obs, mu, sigma = 12.3, 10.0, 2.5

    def integrand(x: float) -> float:
        cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
        indicator = 1.0 if x >= obs else 0.0
        return (cdf - indicator) ** 2

    expected, _ = integrate.quad(integrand, mu - 20 * sigma, mu + 20 * sigma)
    actual = crps_gaussian(np.array([obs]), np.array([mu]), np.array([sigma]))[0]

    assert abs(actual - expected) < 1e-6, f"closed-form {actual} vs numerical {expected}"


def test_crps_gaussian_handles_arrays():
    obs = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.0, 2.0, 3.0])  # perfect forecasts
    sigma = np.array([1.0, 1.0, 1.0])

    crps = crps_gaussian(obs, mu, sigma)

    # Perfect forecast: z=0, so CRPS = sigma * (0 + 2*phi(0) - 1/sqrt(pi))
    # = sigma * (2/sqrt(2*pi) - 1/sqrt(pi))
    expected = 1.0 * (2.0 / math.sqrt(2 * math.pi) - 1.0 / math.sqrt(math.pi))
    np.testing.assert_allclose(crps, expected, atol=1e-9)


def test_crps_gaussian_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def _synthesize(rng, n, *, a=0.5, b=0.98, c=4.0, d=2.0):
    """Generate (ens_mean, ens_std, obs) from a known (a, b, c, d)."""
    ens_mean = rng.normal(60.0, 10.0, size=n)
    ens_std = np.abs(rng.normal(2.0, 0.6, size=n)) + 0.5
    mu = a + b * ens_mean
    sigma = np.sqrt(c + d * ens_std**2)
    obs = rng.normal(mu, sigma)
    return ens_mean, ens_std, obs


def test_fit_cell_recovers_known_parameters():
    rng = np.random.default_rng(42)
    ens_mean, ens_std, obs = _synthesize(rng, n=400, a=0.5, b=0.98, c=4.0, d=2.0)

    fit = fit_cell(ens_mean, ens_std, obs)

    assert isinstance(fit, EmosFit)
    assert abs(fit.a - 0.5) < 3.0  # a has high uncertainty due to aliasing with b*ens_mean
    assert abs(fit.b - 0.98) < 0.1
    assert abs(fit.c - 4.0) < 2.0  # relax slightly to allow optimizer noise
    assert abs(fit.d - 2.0) < 0.6


def test_fit_cell_bounds_enforced():
    """Even pathological data must not push parameters outside their bounds."""
    rng = np.random.default_rng(7)
    n = 200
    ens_mean = rng.normal(60.0, 10.0, size=n)
    ens_std = np.full(n, 0.5)
    # Obs that pushes b way too high if unbounded
    obs = ens_mean * 5.0 + rng.normal(0.0, 1.0, size=n)

    fit = fit_cell(ens_mean, ens_std, obs)

    assert 0.3 <= fit.b <= 1.5
    assert fit.c >= 1e-6
    assert fit.d >= 0.0


def test_fit_cell_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        fit_cell(np.zeros(3), np.ones(3), np.zeros(2))


def test_fit_cell_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="finite"):
        fit_cell(
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_fit_cell_rejects_nonpositive_ens_std():
    with pytest.raises(ValueError, match="ens_std"):
        fit_cell(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_cv_fold_iter_yields_5_folds_by_default():
    forecast_dates = np.arange(50)
    folds = list(cv_fold_iter(forecast_dates))
    assert len(folds) == 5


def test_cv_fold_iter_train_strictly_before_validate():
    """Forward chaining: every train index must precede every validation index."""
    forecast_dates = np.array([f"2026-04-{d:02d}" for d in range(1, 51)])
    for train_idx, val_idx in cv_fold_iter(forecast_dates):
        assert train_idx.max() < val_idx.min(), (
            f"train max {train_idx.max()} must be < val min {val_idx.min()}"
        )


def test_cv_fold_iter_expanding_train_window():
    """Each subsequent fold's train set must be a superset of the previous."""
    forecast_dates = np.arange(50)
    folds = list(cv_fold_iter(forecast_dates))
    for i in range(1, len(folds)):
        prev_train = set(folds[i - 1][0].tolist())
        cur_train = set(folds[i][0].tolist())
        assert prev_train.issubset(cur_train)


def test_cv_fold_iter_handles_repeated_forecast_dates():
    """Multiple rows per forecast_date (different leads) must stay together in a fold."""
    forecast_dates = np.repeat(np.arange(20), 3)  # 20 dates × 3 leads = 60 rows
    for train_idx, val_idx in cv_fold_iter(forecast_dates):
        train_dates = set(forecast_dates[train_idx].tolist())
        val_dates = set(forecast_dates[val_idx].tolist())
        assert train_dates.isdisjoint(val_dates)


def test_bootstrap_cell_percentile_math_matches_numpy():
    """The percentile extraction logic must match numpy.percentile exactly.

    We patch the per-iteration fit to return deterministic samples so we can
    verify the percentile math (not the optimizer).
    """
    from python.analytics import emos_fit as _module
    from python.analytics.emos_fit import bootstrap_cell

    # Make every call to fit_cell return a known sequence of params.
    fake_samples = np.array([[float(i), 1.0 + 0.01 * i, 1.0, 1.0] for i in range(400)])
    call_idx = {"i": 0}

    def fake_fit(ens_mean, ens_std, obs):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = fake_samples[i % len(fake_samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    original = _module.fit_cell
    _module.fit_cell = fake_fit
    try:
        ci = bootstrap_cell(
            np.zeros(50),
            np.ones(50),
            np.zeros(50),
            n_iter=400,
            seed=42,
        )
    finally:
        _module.fit_cell = original

    expected_a = {
        "p2.5": float(np.percentile(fake_samples[:, 0], 2.5)),
        "p50": float(np.percentile(fake_samples[:, 0], 50)),
        "p97.5": float(np.percentile(fake_samples[:, 0], 97.5)),
    }
    np.testing.assert_allclose(
        [ci["a"]["p2.5"], ci["a"]["p50"], ci["a"]["p97.5"]],
        [expected_a["p2.5"], expected_a["p50"], expected_a["p97.5"]],
        atol=1e-9,
    )


def test_bootstrap_cell_warnings_unstable_b():
    """If 95% CI width of b exceeds 0.4, emit unstable_b warning."""
    from python.analytics import emos_fit as _module
    from python.analytics.emos_fit import bootstrap_cell

    samples = np.column_stack(
        [
            np.zeros(400),
            np.linspace(0.5, 1.5, 400),  # width ≈ 1.0 > 0.4
            np.ones(400),
            np.ones(400),
        ]
    )
    call_idx = {"i": 0}

    def fake_fit(*_args, **_kwargs):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = samples[i % len(samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    original = _module.fit_cell
    _module.fit_cell = fake_fit
    try:
        ci = bootstrap_cell(
            np.zeros(50),
            np.ones(50),
            np.zeros(50),
            n_iter=400,
            seed=1,
        )
    finally:
        _module.fit_cell = original

    assert "unstable_b" in ci["ci_warnings"]


def test_bootstrap_cell_warnings_unstable_d():
    """If 95% CI width of d > 4.0 (absolute), emit unstable_d. d>0 must NOT trigger d_at_zero."""
    from python.analytics import emos_fit as _module
    from python.analytics.emos_fit import bootstrap_cell

    # d ranges roughly 1.0 .. 6.5 -> p50 ~3.75, width ~5.4 (> 4.0). p50 well above 1e-6.
    samples = np.column_stack(
        [
            np.zeros(400),
            np.ones(400),
            np.ones(400),
            np.linspace(1.0, 6.5, 400),
        ]
    )
    call_idx = {"i": 0}

    def fake_fit(*_a, **_k):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = samples[i % len(samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    original = _module.fit_cell
    _module.fit_cell = fake_fit
    try:
        ci = bootstrap_cell(np.zeros(50), np.ones(50), np.zeros(50), n_iter=400, seed=2)
    finally:
        _module.fit_cell = original

    assert "unstable_d" in ci["ci_warnings"]
    assert "d_at_zero" not in ci["ci_warnings"]


def test_bootstrap_cell_warnings_d_at_zero():
    """If d_p50 <= 1e-6, emit d_at_zero (independent of width)."""
    from python.analytics import emos_fit as _module
    from python.analytics.emos_fit import bootstrap_cell

    # All-zero d -> p50=0; width=0 so unstable_d must NOT fire.
    samples = np.column_stack([np.zeros(400), np.ones(400), np.ones(400), np.zeros(400)])
    call_idx = {"i": 0}

    def fake_fit(*_a, **_k):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = samples[i % len(samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    original = _module.fit_cell
    _module.fit_cell = fake_fit
    try:
        ci = bootstrap_cell(np.zeros(50), np.ones(50), np.zeros(50), n_iter=400, seed=3)
    finally:
        _module.fit_cell = original

    assert "d_at_zero" in ci["ci_warnings"]
    assert "unstable_d" not in ci["ci_warnings"]


def test_bootstrap_cell_warnings_d_at_zero_and_unstable_d_can_coexist():
    """A cell with d_p50 ~= 0 but a few large bootstrap draws (width > 4) gets both flags."""
    from python.analytics import emos_fit as _module
    from python.analytics.emos_fit import bootstrap_cell

    # 380 zeros then 20 large values -> p50=0, p97.5 ~10, width > 4.
    d_col = np.concatenate([np.zeros(380), np.linspace(8.0, 10.0, 20)])
    samples = np.column_stack([np.zeros(400), np.ones(400), np.ones(400), d_col])
    call_idx = {"i": 0}

    def fake_fit(*_a, **_k):
        i = call_idx["i"]
        call_idx["i"] += 1
        a, b, c, d = samples[i % len(samples)]
        return EmosFit(a=a, b=b, c=c, d=d)

    original = _module.fit_cell
    _module.fit_cell = fake_fit
    try:
        ci = bootstrap_cell(np.zeros(50), np.ones(50), np.zeros(50), n_iter=400, seed=4)
    finally:
        _module.fit_cell = original

    assert "d_at_zero" in ci["ci_warnings"]
    assert "unstable_d" in ci["ci_warnings"]


def test_synthetic_bracket_brier_returns_finite_floats():
    from python.analytics.emos_fit import synthetic_bracket_brier

    rng = np.random.default_rng(0)
    n = 100
    ens_mean = rng.normal(60.0, 10.0, size=n)
    ens_std = np.abs(rng.normal(2.0, 0.5, size=n)) + 0.5
    obs = ens_mean + rng.normal(0.0, 2.0, size=n)
    fit = EmosFit(a=0.0, b=1.0, c=1.0, d=1.0)

    brier_emos, brier_raw = synthetic_bracket_brier(ens_mean, ens_std, obs, fit)

    assert math.isfinite(brier_emos)
    assert math.isfinite(brier_raw)
    assert 0.0 <= brier_emos <= 1.0
    assert 0.0 <= brier_raw <= 1.0


def test_synthetic_bracket_brier_outcomes_mix_zero_and_one():
    """The 5-offset grid must produce both contains-obs and doesn't-contain outcomes.

    Verified by checking that a perfectly-confident predictor (sigma -> 0)
    centered exactly at obs gets non-trivial Brier — i.e., it loses on the
    offset brackets where outcome=0.

    Passes `half_width=2.5` explicitly: the test's purpose is to validate
    the GRID-OF-OFFSETS math, not the default geometry. At 1°F (the new
    Tier 2 D default), the off-center offsets are far enough from obs that
    a degenerate σ→0 EMOS correctly predicts P=0 / outcome=0 → contribution=0,
    so the same scenario yields Brier_emos == 0 and would no longer witness
    mixed outcomes. A wider half_width (2.5°F) keeps the off-center brackets
    close enough to obs that σ→0's P=0 prediction misses (P=0, outcome=1),
    which is the failure mode this test was designed to catch.
    """
    from python.analytics.emos_fit import synthetic_bracket_brier

    n = 50
    ens_mean = np.zeros(n)
    ens_std = np.full(n, 1.0)
    obs = np.zeros(n)  # obs perfectly equals ens_mean
    # A degenerate fit collapsing sigma to ~0 around mu=ens_mean.
    fit = EmosFit(a=0.0, b=1.0, c=1e-6, d=1e-6)

    brier_emos, brier_raw = synthetic_bracket_brier(
        ens_mean, ens_std, obs, fit, half_width=2.5
    )

    # If the test ever measured only obs-centered brackets, brier_emos would be
    # ~0 (collapsed mass at obs always predicts P=1, outcome=1). Mixed grid
    # forces non-trivial loss.
    assert brier_emos > 0.05


def test_synthetic_bracket_brier_default_half_width_one_degree():
    """Default half_width is 0.5 → 1°F brackets (KXHIGH/KXLOWT geometry).

    The old 2.5° default scored EMOS fits on 5°F-wide brackets, flattering
    cells with large `d` (σ_emos ≫ 1°F looks fine on 5°F brackets but is
    structurally meaningless at the trading geometry).
    """
    from python.analytics import emos_fit
    from python.analytics.emos_fit import synthetic_bracket_brier

    # Constant witness — both the module-internal and re-exported references
    # must read 0.5 so callers using either path get 1°F geometry by default.
    assert emos_fit._DEFAULT_HALF_WIDTH == 0.5

    # Behavioural witness: a call without `half_width=` must equal a call
    # passing `half_width=0.5` on the same inputs (i.e., the default is 0.5).
    # We use a non-degenerate fit so the equality is over non-trivial Brier
    # values, not a degenerate 0 == 0.
    rng = np.random.default_rng(7)
    n = 80
    ens_mean = rng.normal(60.0, 8.0, size=n)
    ens_std = np.abs(rng.normal(2.0, 0.5, size=n)) + 0.5
    obs = ens_mean + rng.normal(0.0, 1.5, size=n)
    fit = EmosFit(a=0.5, b=0.98, c=1.0, d=0.5)

    default_emos, default_raw = synthetic_bracket_brier(ens_mean, ens_std, obs, fit)
    explicit_1f_emos, explicit_1f_raw = synthetic_bracket_brier(
        ens_mean, ens_std, obs, fit, half_width=0.5
    )

    assert default_emos == pytest.approx(explicit_1f_emos, abs=1e-12)
    assert default_raw == pytest.approx(explicit_1f_raw, abs=1e-12)


def test_synthetic_bracket_brier_respects_half_width_param():
    """Passing half_width=2.5 reproduces the prior 5°F-default behaviour.

    Back-compat witness for intraday strategies (or any caller) that need
    wider-bracket scoring. The result must differ from the 1°F default —
    otherwise the parameter is dead.
    """
    from python.analytics.emos_fit import synthetic_bracket_brier

    # Frozen-input fixture so the 2.5° result is a stable record of the
    # legacy default behaviour. If a future refactor changes the Brier math,
    # this assertion will trip first; the 1°F-vs-2.5° comparison below
    # remains the load-bearing "parameter is wired" check.
    rng = np.random.default_rng(7)
    n = 80
    ens_mean = rng.normal(60.0, 8.0, size=n)
    ens_std = np.abs(rng.normal(2.0, 0.5, size=n)) + 0.5
    obs = ens_mean + rng.normal(0.0, 1.5, size=n)
    fit = EmosFit(a=0.5, b=0.98, c=1.0, d=0.5)

    default_emos, default_raw = synthetic_bracket_brier(ens_mean, ens_std, obs, fit)
    wide_emos, wide_raw = synthetic_bracket_brier(
        ens_mean, ens_std, obs, fit, half_width=2.5
    )

    # Numerical fixture for the 2.5° default: pre-Tier-2-D production behaviour.
    # Generated from the same seed/inputs above with `half_width=2.5`. Locks
    # the legacy semantic so back-compat callers (e.g., intraday strategies)
    # can reproduce historical diagnostics by passing the wider half_width.
    assert wide_emos == pytest.approx(0.1372398344, abs=1e-6)
    assert wide_raw == pytest.approx(0.1433372835, abs=1e-6)

    # The 1°F default must differ from the 2.5° result — otherwise the
    # `half_width` parameter would be dead code.
    assert not math.isclose(default_emos, wide_emos, abs_tol=1e-4)
    assert not math.isclose(default_raw, wide_raw, abs_tol=1e-4)


@pytest.mark.parametrize(
    "n,expected",
    [
        (12, (True, False, "n=12 below floor")),
        (29, (True, False, "n=29 below floor")),
        (30, (False, True, None)),
        (45, (False, True, None)),
        (59, (False, True, None)),
        (60, (False, False, None)),
        (200, (False, False, None)),
    ],
)
def test_skip_rule_thresholds(n, expected):
    assert skip_rule(n) == expected


def _make_synthetic_cell_df(rng, n_dates=50, leads=(0, 1, 2)):
    rows = []
    for i in range(n_dates):
        date = pd.Timestamp("2026-03-01") + pd.Timedelta(days=i)
        for lead in leads:
            ens_mean = float(rng.normal(60.0, 10.0))
            ens_std = float(abs(rng.normal(2.0, 0.5)) + 0.5)
            obs = ens_mean + 0.5 + 0.98 * 0.0 + float(rng.normal(0.0, ens_std))
            rows.append(
                {
                    "forecast_date": date,
                    "lead": lead,
                    "ens_mean": ens_mean,
                    "ens_std": ens_std,
                    "obs": obs,
                }
            )
    return pd.DataFrame(rows)


def test_fit_one_cell_skipped_when_n_below_floor():
    from python.analytics.emos_fit import fit_one_cell

    rng = np.random.default_rng(0)
    df = _make_synthetic_cell_df(rng, n_dates=4, leads=(0, 1, 2))  # n=12

    coef, diag = fit_one_cell(df, seed=42)

    assert coef == {"skipped": True, "reason": "n=12 below floor"}
    assert diag["n_train"] == 12


def test_fit_one_cell_full_output_for_normal_cell():
    from python.analytics.emos_fit import fit_one_cell

    rng = np.random.default_rng(0)
    df = _make_synthetic_cell_df(rng, n_dates=50, leads=(0, 1, 2))  # n=150

    coef, diag = fit_one_cell(df, seed=42, bootstrap_iters=50)

    assert coef["leads_pooled"] == [0, 1, 2]
    assert coef["low_n"] is False
    assert "ci_warnings" in coef
    assert all(k in coef for k in ("a", "b", "c", "d", "n_train"))
    assert all(
        k in diag
        for k in (
            "n_train",
            "n_per_lead",
            "in_sample_crps",
            "oos_crps_a1",
            "oos_crps_a2",
            "oos_brier_emos",
            "oos_brier_raw",
            "fold_crps",
            "bootstrap",
            "ci_warnings",
        )
    )
    assert diag["n_per_lead"] == {0: 50, 1: 50, 2: 50}

    # σ-diagnostic plumbing must survive the bootstrap.pop(...) lift in
    # fit_one_cell. A regression would silently zero out the persisted
    # coefficients.diagnostics.json σ block on the next refit.
    for key in (
        "sigma_emos_at_p50",
        "sigma_emos_at_p99",
        "sigma_to_bracket_ratio_p99",
        "synthetic_brier_half_width",
        "bracket_width",
        "incompatible_sigma_threshold",
    ):
        assert key in diag, f"missing top-level diag key: {key}"
    assert isinstance(diag["sigma_emos_at_p50"], float)
    assert isinstance(diag["sigma_emos_at_p99"], float)
    assert diag["sigma_emos_at_p99"] >= diag["sigma_emos_at_p50"]
    assert diag["sigma_to_bracket_ratio_p99"] == pytest.approx(
        diag["sigma_emos_at_p99"] / diag["bracket_width"], rel=1e-9
    )
    # σ keys must NOT remain inside the bootstrap envelope after pop().
    bootstrap = diag["bootstrap"]
    for popped in (
        "sigma_emos_at_p50",
        "sigma_emos_at_p99",
        "sigma_to_bracket_ratio_p99",
        "ci_warnings",
    ):
        assert popped not in bootstrap, f"{popped} should have been popped from bootstrap"


def test_fit_one_cell_threads_non_default_bracket_width():
    """fit_one_cell must thread a non-default bracket_width through to
    bootstrap_cell + diagnostics. Intraday strategies with wider bracket
    geometry would otherwise silently get the 1°F σ-gate."""
    from python.analytics.emos_fit import fit_one_cell

    rng = np.random.default_rng(0)
    df = _make_synthetic_cell_df(rng, n_dates=50, leads=(0, 1, 2))  # n=150

    coef_1f, diag_1f = fit_one_cell(df, seed=42, bootstrap_iters=50, bracket_width=1.0)
    coef_5f, diag_5f = fit_one_cell(df, seed=42, bootstrap_iters=50, bracket_width=5.0)

    # bracket_width is persisted to top-level diag verbatim.
    assert diag_1f["bracket_width"] == 1.0
    assert diag_5f["bracket_width"] == 5.0

    # σ_p99 is bracket-width-independent (function of (c, d, ens_std) only),
    # so the ratio must scale exactly 5× between the two widths.
    assert diag_1f["sigma_emos_at_p99"] == pytest.approx(diag_5f["sigma_emos_at_p99"], rel=1e-12)
    assert diag_1f["sigma_to_bracket_ratio_p99"] == pytest.approx(
        5.0 * diag_5f["sigma_to_bracket_ratio_p99"], rel=1e-9
    )

    # Non-default bracket_width does not change the underlying fit (a, b, c, d).
    # bracket_width is only consumed by bootstrap_cell's σ-ratio computation;
    # fit_cell does not see it. Same inputs → same optimizer trajectory →
    # bit-identical result.
    for key in ("a", "b", "c", "d"):
        assert coef_1f[key] == pytest.approx(coef_5f[key], rel=1e-12), (
            f"fit value {key} drifted between bracket widths; "
            "bracket_width must not influence the L-BFGS fit"
        )
