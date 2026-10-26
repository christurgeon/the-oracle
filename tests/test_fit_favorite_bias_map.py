# tests/test_fit_favorite_bias_map.py
from __future__ import annotations

import math

from scripts.fit_favorite_bias_map import (
    Block,
    _block_p,
    _expit,
    _logit,
    apply_lift_guard,
    build_artifact,
    cap_values,
    estimate_tau2,
    fit_logit_trend,
    interp_curve,
    logit_stats,
    monotone_project,
    pav_isotonic,
)


def _pairs(xs, ys, days=None):
    days = days or [f"2026-01-{i + 1:02d}" for i in range(len(xs))]
    return list(zip(xs, ys, days))


def _block(mid, y, n_days, obs_per_day=1):
    return Block(
        [(mid, float(y), f"2026-02-{d + 1:02d}") for d in range(n_days) for _ in range(obs_per_day)]
    )


# --- PAV -------------------------------------------------------------------


def test_pav_is_monotone():
    blocks = pav_isotonic(_pairs([0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 1.0, 0.0, 1.0, 1.0]))
    vals = [b.value for b in blocks]
    assert vals == sorted(vals)


def test_pav_unique_breakpoints_and_real_counts():
    """Duplicate x-values must yield strictly increasing block mean mids,
    no phantom zero-count blocks, and counts summing to n."""
    xs = [0.40, 0.40, 0.40, 0.55, 0.55, 0.55, 0.55, 0.70, 0.70, 0.70]
    ys = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    blocks = pav_isotonic(_pairs(xs, ys))
    mids = [b.mean_mid for b in blocks]
    assert mids == sorted(set(mids))
    assert all(b.n_obs > 0 for b in blocks)
    assert sum(b.n_obs for b in blocks) == len(xs)


# --- logit helpers ---------------------------------------------------------


def test_logit_expit_roundtrip_and_clamping():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert math.isclose(_expit(_logit(p)), p, rel_tol=1e-9)
    # 0 and 1 must stay finite rather than producing +/-inf
    assert math.isfinite(_logit(0.0)) and math.isfinite(_logit(1.0))


def test_block_p_continuity_correction():
    """A block that went 5-for-5 must not have an infinite logit."""
    assert 0.5 < _block_p(5, 5) < 1.0
    assert math.isfinite(_logit(_block_p(5, 5)))


def test_logit_stats_single_day_block_gets_a_large_se():
    obs, se = logit_stats(_block(0.5, 1.0, n_days=1, obs_per_day=20), b=200)
    assert math.isfinite(obs)
    assert se > 1.0, se  # one day of evidence must not read as well-measured


def test_logit_stats_degenerate_block_does_not_report_zero_se():
    """Regression: the bootstrap collapses on an all-wins block, reporting SE ~ 0
    and making 3 observations look perfectly measured."""
    _obs, se = logit_stats(_block(0.6, 1.0, n_days=3), b=500)
    assert se > 1.0, se


def test_logit_stats_degenerate_block_with_uneven_days():
    """The degeneracy test must key on OUTCOMES, not on the bootstrap magnitude.

    With uneven obs-per-day the bootstrap does not collapse to exactly 0 — it
    wanders via the continuity correction's denominator — so a magnitude test
    misses it. This is the shape that produced a 0-for-486 block with a bootstrap
    SE of 0.043 against a true 1.42.
    """
    members = [(0.6, 1.0, f"d{d}") for d in range(3) for _ in range(d + 1)]
    _obs, se = logit_stats(Block(members), b=500)
    assert se > 1.0, se


def test_logit_stats_uses_parametric_for_a_lopsided_large_block():
    """A 0-for-N block is degenerate no matter how many observations it has."""
    members = [(0.02, 0.0, f"d{i % 40}") for i in range(120)]
    _obs, se = logit_stats(Block(members), b=500)
    assert se > 1.0, se


# --- trend and shrinkage ---------------------------------------------------


def test_fit_logit_trend_recovers_known_beta():
    """Blocks generated from logit(p) = 0 + 2*logit(mid) must recover beta ~ 2."""
    blocks, stats = [], []
    for mid in (0.2, 0.35, 0.5, 0.65, 0.8):
        p = _expit(2.0 * _logit(mid))
        wins = round(p * 100)
        members = [(mid, 1.0 if i < wins else 0.0, f"2026-03-{i % 28 + 1:02d}") for i in range(100)]
        b = Block(members)
        blocks.append(b)
        stats.append((_logit(_block_p(wins, 100)), 0.05))
    _alpha, beta = fit_logit_trend(blocks, stats, min_days=1)
    assert 1.7 < beta < 2.3, beta


def test_fit_logit_trend_defaults_to_identity_odds_without_evidence():
    """Too few trend-eligible blocks -> identity in odds, never an invented slope."""
    b = _block(0.5, 1.0, n_days=1)
    alpha, beta = fit_logit_trend([b], [(0.0, 2.5)])
    assert (alpha, beta) == (0.0, 1.0)


def test_fit_logit_trend_rejects_a_negative_slope():
    """A decreasing trend would invert the map; fall back to identity in odds."""
    blocks = [_block(m, 1.0, n_days=5) for m in (0.2, 0.4, 0.6, 0.8)]
    # win rate falling as mid rises -> negative beta
    stats = [(2.0, 0.1), (1.0, 0.1), (-1.0, 0.1), (-2.0, 0.1)]
    assert fit_logit_trend(blocks, stats, min_days=1) == (0.0, 1.0)


def test_estimate_tau2_is_zero_when_underdetermined():
    b = _block(0.5, 1.0, n_days=10)
    tau2, q, df = estimate_tau2([b], [(0.0, 0.1)], (0.0, 1.0), min_days=1)
    assert (tau2, q, df) == (0.0, 0.0, 0)


def test_estimate_tau2_dl_arithmetic():
    """Hand-computed DL with the meta-regression (p=2) denominator.

    k=4 blocks, se=0.5 so w=4, residuals +/-1 about a zero trend.
      Q     = sum(r^2/se^2) = 4 * (1/0.25) = 16
      df    = k - 2 = 2
      trace = tr[(X'WX)^-1 X'W^2X] = w * p = 4 * 2 = 8   (equal weights)
      denom = sum(w) - trace = 16 - 8 = 8
      tau2  = (16 - 2) / 8 = 1.75

    Expected values are hard-coded, NOT re-derived from the implementation's own
    denominator expression — otherwise the test would lock in whichever form the
    code happens to use (it previously used the intercept-only p=1 denominator,
    which this case would not have caught).
    """
    blocks = [_block(m, 1.0, n_days=5) for m in (0.3, 0.5, 0.7, 0.9)]
    stats = [(r, 0.5) for r in (1.0, -1.0, 1.0, -1.0)]
    tau2, q, df = estimate_tau2(blocks, stats, (0.0, 0.0), min_days=1)
    assert df == 2
    assert math.isclose(q, 16.0, rel_tol=1e-9)
    assert math.isclose(tau2, 1.75, rel_tol=1e-9), tau2


def test_estimate_tau2_k3_unequal_ses():
    """k=3 (df=1) with unequal SEs still produces a finite non-negative tau2."""
    blocks = [_block(m, 1.0, n_days=5) for m in (0.3, 0.6, 0.9)]
    stats = [(1.0, 0.2), (-0.5, 0.4), (0.8, 0.7)]
    tau2, q, df = estimate_tau2(blocks, stats, (0.0, 0.0), min_days=1)
    assert df == 1
    assert q > 0
    assert tau2 >= 0.0 and math.isfinite(tau2)


def test_estimate_tau2_is_zero_when_blocks_agree_with_trend():
    blocks = [_block(m, 1.0, n_days=5) for m in (0.3, 0.5, 0.7, 0.9)]
    stats = [(0.0, 0.5)] * 4  # zero residuals -> Q = 0 < df
    tau2, _q, _df = estimate_tau2(blocks, stats, (0.0, 0.0), min_days=1)
    assert tau2 == 0.0


def _synth_block(mid, win_rate, n_days, obs_per_day=4, tag="x"):
    """A block at ``mid`` with the given win rate spread over ``n_days`` days."""
    members = []
    total = n_days * obs_per_day
    wins = round(win_rate * total)
    for i in range(total):
        members.append((mid, 1.0 if i < wins else 0.0, f"{tag}-{i % n_days:03d}"))
    return Block(members)


def test_thin_block_is_pulled_further_toward_the_trend_than_a_fat_one():
    """The empirical-Bayes weight must rise with the evidence behind a block.

    Needs enough trend-eligible blocks, with dispersion the 2-parameter trend
    cannot absorb, for ``tau2`` to be estimable at all — with tau2 == 0 every
    block is pinned to the trend and the weights are trivially equal.
    """
    # 100 days each so the parametric SE floor is small enough that a genuine
    # zigzag around the logit-linear trend survives as between-block variance.
    fat = [
        _synth_block(0.20, 0.14, 100, obs_per_day=1, tag="a"),
        _synth_block(0.30, 0.45, 100, obs_per_day=1, tag="b"),
        _synth_block(0.45, 0.38, 100, obs_per_day=1, tag="c"),
        _synth_block(0.55, 0.74, 100, obs_per_day=1, tag="d"),
        _synth_block(0.70, 0.66, 100, obs_per_day=1, tag="e"),
        _synth_block(0.80, 0.94, 100, obs_per_day=1, tag="f"),
    ]
    thin = _synth_block(0.60, 1.0, 3, obs_per_day=1, tag="g")
    blocks = fat[:4] + [thin] + fat[4:]  # keep mids increasing
    _bp, _vals, info = apply_lift_guard(blocks, min_trend_days=5)

    assert info["tau2_logit"] > 0, "test is vacuous unless tau2 is estimable"
    w_thin = info["blocks"][4]["shrink_weight"]
    w_fat = info["blocks"][3]["shrink_weight"]
    assert w_thin < w_fat, (w_thin, w_fat)


def test_guard_output_is_monotone_and_in_probability_range():
    """Regression: a linear lift trend extrapolated past 1.0 and the calibrator's
    own validation does not check range, so values must be verified here."""
    blocks = pav_isotonic(
        _pairs(
            [0.02, 0.10, 0.30, 0.50, 0.70, 0.90, 0.98] * 6,
            [0, 0, 0, 1, 1, 1, 1] * 6,
            [f"2026-06-{i % 28 + 1:02d}" for i in range(42)],
        )
    )
    _bp, vals, _info = apply_lift_guard(blocks, min_trend_days=1)
    assert vals == sorted(vals)
    assert all(0.0 < v < 1.0 for v in vals), vals


# --- monotone projection ---------------------------------------------------


def test_monotone_project_preserves_length_and_orders():
    out = monotone_project([0.2, 0.9, 0.4, 0.5], [1.0, 1.0, 1.0, 1.0])
    assert len(out) == 4
    assert out == sorted(out)


def test_monotone_project_averages_rather_than_only_raising():
    """Regression: cumulative max only ever pushed values up, inflating the curve
    where evidence was weakest. A weighted projection must pull them together."""
    out = monotone_project([0.8, 0.2], [1.0, 1.0])
    assert out == [0.5, 0.5]
    # and it must respect weight: a heavy low block drags the merged value down
    heavy = monotone_project([0.8, 0.2], [1.0, 9.0])
    assert heavy[0] < 0.5


# --- artifact --------------------------------------------------------------


def test_cap_clips_high_values_in_band():
    bps, vals, idx = cap_values([0.65, 0.95], [0.92, 0.99], cap=0.85, mid_ceiling=0.75)
    assert vals[bps.index(0.65)] == 0.85  # in band -> capped
    assert vals[bps.index(0.95)] == 0.99  # above the ceiling -> untouched
    # A knot IS required here: interpolating from the capped 0.65 knot to the
    # uncapped 0.95 one still crosses the cap before reaching the ceiling.
    assert idx == 1 and bps[idx] == 0.75
    assert interp_curve(bps, vals, 0.74) <= 0.85 + 1e-12


def test_cap_holds_under_interpolation_not_just_at_knots():
    """Regression: with the last in-band knot below mid_ceiling and the next one
    above it uncapped, prices just under the ceiling interpolated past the cap."""
    bps, vals, idx = cap_values([0.665, 0.756], [0.823, 0.899], cap=0.85, mid_ceiling=0.75)
    assert idx == 1 and bps[idx] == 0.75, "a knot must be pinned at the ceiling"
    assert vals == sorted(vals), vals
    for x in (0.70, 0.7499, 0.75):
        assert interp_curve(bps, vals, x) <= 0.85 + 1e-12, x


def test_cap_boundary_branches():
    """The four non-insert branches must each return a curve honouring the cap."""
    # ceiling below the first knot: the flat left clamp must still be capped
    bps, vals, idx = cap_values([0.80, 0.90], [0.92, 0.99], cap=0.85, mid_ceiling=0.75)
    assert idx is None
    assert interp_curve(bps, vals, 0.70) <= 0.85

    # ceiling above the last knot: everything is in band
    bps, vals, idx = cap_values([0.30, 0.60], [0.92, 0.99], cap=0.85, mid_ceiling=0.75)
    assert idx is None and vals == [0.85, 0.85]

    # ceiling exactly on an existing knot: no duplicate breakpoint
    bps, vals, idx = cap_values([0.60, 0.75, 0.90], [0.5, 0.9, 0.95], cap=0.85, mid_ceiling=0.75)
    assert idx is None and bps.count(0.75) == 1 and vals[1] == 0.85

    # curve already entirely below the cap: unchanged, no knot
    bps, vals, idx = cap_values([0.60, 0.90], [0.5, 0.7], cap=0.85, mid_ceiling=0.75)
    assert idx is None and vals == [0.5, 0.7] and len(bps) == 2


def test_cap_and_projection_reject_bad_input():
    for fn, args in (
        (cap_values, ([0.1, 0.2], [0.3])),
        (monotone_project, ([0.1, 0.2], [1.0])),
        (cap_values, ([], [])),
    ):
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted invalid input")


def test_artifact_schema_and_metadata():
    blocks = [_block(0.40, 1.0, n_days=4), _block(0.60, 1.0, n_days=3)]
    info = {"trend_alpha": 0.0, "trend_beta": 1.4, "tau2_logit": 0.02, "blocks": [{}, {}]}
    art = build_artifact([0.4, 0.6], [0.5, 0.8], blocks, info)
    assert art["breakpoints"] == [0.4, 0.6]
    assert art["values"] == [0.5, 0.8]
    assert art["metadata"]["feature"] == "mid"
    assert art["metadata"]["n_per_bucket"] == [4, 3]
    assert art["metadata"]["n_days_per_bucket"] == [4, 3]
    assert art["metadata"]["guard_version"] == 2
    assert art["metadata"]["fit_info"]["trend_beta"] == 1.4
    assert "fit_date" in art["metadata"]


def test_artifact_metadata_stays_aligned_with_a_synthetic_knot():
    """Regression: the inserted cap knot is not a block, so without a placeholder
    every per-block entry after it mis-attributes by one."""
    blocks = [_block(0.40, 1.0, n_days=4), _block(0.90, 1.0, n_days=3)]
    info = {
        "trend_alpha": 0.0,
        "trend_beta": 1.4,
        "tau2_logit": 0.0,
        "blocks": [{"a": 1}, {"b": 2}],
    }
    art = build_artifact([0.4, 0.75, 0.9], [0.5, 0.85, 0.9], blocks, info, synthetic_index=1)
    md = art["metadata"]
    for key in ("n_per_bucket", "n_days_per_bucket"):
        assert len(md[key]) == len(art["breakpoints"]), key
        assert md[key][1] is None, "synthetic knot must be a placeholder, not a real count"
    assert len(md["fit_info"]["blocks"]) == len(art["breakpoints"])
    assert md["fit_info"]["blocks"][1] == {"synthetic_cap_knot": True}
    assert md["fit_info"]["blocks"][2] == {"b": 2}  # real block not shifted
    assert md["synthetic_knot_index"] == 1
