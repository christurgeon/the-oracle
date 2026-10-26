"""Offline isotonic fitter for the favorite-bias price->P(YES) map g(mid).

Fits on FULL greater/less lead-1 (mid, yes_outcome) pairs from the snapshot
corpus, applies the guards below, and writes the IsotonicCalibrator JSON schema
atomically. `--out` is required; pass the live path only to replace the live map.

Guard v2 (2026-08-29) corrects three v1 defects that let the map assert lift it
could not support:

1. Effective n is the block's distinct forecast_date count, not its raw obs
   count — markets sharing a date share one weather draw.
2. Breakpoints and shrink targets are the block's MEAN mid, not its right edge,
   which had moved values away from the block's own average price.
3. Blocks borrow strength from each other. PAV pools only downward violations,
   so a spuriously-high monotone block was accepted verbatim; each block's win
   rate is now shrunk toward a fitted trend by an empirical-Bayes weight.

The trend is `logit(g) = alpha + beta*logit(mid)`, the power-odds model from the
favourite-longshot literature (beta > 1 IS the bias). A linear lift-vs-mid trend
was tried first and rejected: it cannot satisfy g->0 and g->1 at the ends, and
extrapolated to g > 1 — which `_validate_curve` does not catch, since it checks
monotonicity but not range.

Shrinkage is continuous rather than a pass/fail gate on the lift CI. The gate was
seed-dependent, let a block clearing zero by +0.002 assert its full point
estimate, and was silently undone afterwards by the monotonicity repair.

Monotonicity uses a weighted isotonic projection, not a cumulative max, which
could only push values up.

CAVEAT: tau2 is estimated from few blocks and has little power — a tau2 of 0
means every block is pinned to the trend and the artifact is a 2-parameter curve.
Check `q_over_df` and `degenerate_pure_trend` in the output before trusting it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from python.analytics.outcomes import derive_outcome_yes
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date
from scripts._calibration_common import load_snapshot_data

OUT = Path("data/calibration/favorite_bias_isotonic.json")

# Guard v2 parameters.
# Bootstrap resamples for each block's day-clustered lift standard error. A
# standard error converges far faster than the tail quantile the earlier
# pass/fail gate needed (that required B=50000 to be seed-stable), so 4000 is
# ample here and keeps the fit under ~10s.
LIFT_BOOTSTRAP_B = 4_000
# Floor on the estimated standard error. Deep-longshot blocks are degenerate
# (every outcome is a loss and mid barely varies), so their bootstrap SE is
# ~0 and an unfloored inverse-variance weight would let them dominate.
SE_FLOOR = 0.005
# A block with fewer than this many outcomes on its minority side is degenerate:
# its bootstrap SE reflects the continuity correction's denominator, not outcome
# variability, and understates the truth by up to ~30x. Gate on the OUTCOME
# counts, not on the bootstrap's magnitude — a degenerate block only bootstraps
# to exactly 0 when every day carries the same number of observations.
DEGENERATE_OUTCOME_FLOOR = 5
# A block needs this many distinct forecast_dates to help *estimate* the trend.
# It is still shrunk toward the trend regardless of how thin it is.
MIN_TREND_DAYS = 5
# Clamp for the logit transform, so a 0% or 100% block stays on a finite scale.
LOGIT_EPS = 1e-6
CAP = 0.85
MID_CEILING = 0.75

# g(mid) is the LIVE favorite_bias calibrator. Its corpus is FROZEN to the
# cities the pilot actually trades. New Kalshi cities are collected (snapshots
# now capture them and parse_ticker_city_var resolves them) but MUST NOT enter
# this refit before their Phase B per-city validation — otherwise the map and
# the validation share data (circularity).
# Adding a city here is a deliberate Phase B step, gated on calibration sign-off.
_CORPUS_CITIES: frozenset[str] = frozenset(
    {"NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "BOS", "PHIL"}
)


def build_pairs(snapshots_dir: Path) -> list[tuple[float, int, str]]:
    """Full greater/less lead-1 (mid, yes_outcome, forecast_date).

    One obs per (ticker, latest snapshot). ``forecast_date`` is the clustering
    key for the guard-v2 shrinkage: every market resolving on the same date
    settles against one weather draw, so they are not independent observations.
    """
    return [(mid, y, day) for mid, _ask, y, day in build_rows(snapshots_dir)]


def build_rows(snapshots_dir: Path) -> list[tuple[float, float, int, str]]:
    """As ``build_pairs`` but also carrying the yes-ask.

    Validation needs the executable price, not just the mid, to score a map on
    what it would actually have paid. Fitting and validation share this one
    loader so their populations cannot silently diverge.
    """
    _f, settle, markets = load_snapshot_data(snapshots_dir)
    # Deduplicate: keep the latest snapshot for each ticker (highest snapshot_ts).
    best: dict[str, tuple[float, dict]] = {}  # ticker -> (snapshot_ts, market)
    for m in markets:
        t = m.get("ticker", "")
        if not (t.startswith("KXHIGH") or t.startswith("KXLOWT")):
            continue
        if m.get("strike_type") not in ("greater", "less"):
            continue
        fd = parse_ticker_date(t)
        ts = m.get("snapshot_ts")
        if fd is None or ts is None:
            continue
        # Lead-1 gate: snapshot was taken exactly 1 day before the forecast date.
        snap_date = datetime.fromtimestamp(ts, tz=UTC).date()
        if (fd - snap_date).days != 1:
            continue
        if t not in best or ts > best[t][0]:
            best[t] = (ts, m)

    rows: list[tuple[float, float, int, str]] = []
    for t, (_ts, m) in best.items():
        fd = parse_ticker_date(t)
        if fd is None:
            continue
        # Derive city and var from the ticker so the fitter is consistent with
        # the live strategy (_parse_market also uses parse_ticker_city_var).
        # Snapshot market dicts carry a `city` field, but using it here would
        # create a silent inconsistency: the fitter and the strategy would use
        # different city-resolution paths.
        cv = parse_ticker_city_var(t)
        if cv is None:
            continue
        city, var = cv
        if city not in _CORPUS_CITIES:
            continue  # new cities are collection-only until Phase B (spec §7)
        st = settle.get((city, fd.isoformat()))
        if st is None:
            continue
        temp = st.get("high_f") if var == "high" else st.get("low_f")
        if temp is None:
            continue
        yo = derive_outcome_yes(
            m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike"), temp
        )
        if yo is None:
            continue
        ya = m.get("yes_ask_dollars")
        yb = m.get("yes_bid_dollars")
        # Require a genuine two-sided quote: zero-bid markets yield a fictitious mid.
        if ya is None or yb is None or not (0 < yb < ya < 1):
            continue
        rows.append(((ya + yb) / 2.0, float(ya), 1 if yo else 0, fd.isoformat()))

    return rows


class Block:
    """One PAV block: a contiguous run of mids sharing a single fitted value."""

    __slots__ = ("members", "value")

    def __init__(self, members: list[tuple[float, float, str]]) -> None:
        self.members = members  # (mid, y, forecast_date)
        self.value = sum(m[1] for m in members) / len(members)

    @property
    def n_obs(self) -> int:
        return len(self.members)

    @property
    def n_days(self) -> int:
        return len({m[2] for m in self.members})

    @property
    def mean_mid(self) -> float:
        return sum(m[0] for m in self.members) / len(self.members)

    @property
    def lift(self) -> float:
        return self.value - self.mean_mid


def pav_isotonic(pairs: list[tuple[float, int, str]]) -> list[Block]:
    """Pool-adjacent-violators with tie pre-aggregation, returning member-carrying blocks.

    Ties on x are pre-aggregated so each unique mid contributes a single block
    before PAV runs. Merges concatenate member lists, so each returned block can
    report its own raw count, distinct-day count, mean mid, and bootstrap CI —
    all of which guard v2 needs and the v1 (value, count, x) triple could not
    supply.
    """
    ordered = sorted(pairs, key=lambda p: p[0])

    # Pre-aggregate tied mids into unique-x blocks.
    blocks: list[Block] = []
    for mid, y, day in ordered:
        if blocks and blocks[-1].members[-1][0] == mid:
            blocks[-1].members.append((mid, float(y), day))
            blocks[-1].value = sum(m[1] for m in blocks[-1].members) / len(blocks[-1].members)
        else:
            blocks.append(Block([(mid, float(y), day)]))

    # Merge adjacent non-monotone blocks until the value sequence is non-decreasing.
    merged = True
    while merged:
        merged = False
        out: list[Block] = []
        for b in blocks:
            if out and out[-1].value > b.value:
                out[-1] = Block(out[-1].members + b.members)
                merged = True
            else:
                out.append(b)
        blocks = out

    return blocks


def _logit(p: float) -> float:
    p = min(max(p, LOGIT_EPS), 1.0 - LOGIT_EPS)
    return math.log(p / (1.0 - p))


def _expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _block_p(wins: float, n: float) -> float:
    """Win rate with a +0.5/+1 continuity correction.

    A block that went 5-for-5 has an observed rate of 1.0, whose logit is
    infinite; the correction keeps every block on a finite scale without
    special-casing the extremes.
    """
    return (wins + 0.5) / (n + 1.0)


def logit_stats(block: Block, *, b: int = LIFT_BOOTSTRAP_B) -> tuple[float, float]:
    """Block win rate on the logit scale, plus its day-clustered standard error.

    The bootstrap resamples whole forecast_dates with replacement, mirroring the
    day-clustered CIs used everywhere else in this repo: markets resolving on one
    date settle against one weather draw and are not independent draws.

    A parametric binomial SE takes over on **outcome-degenerate** blocks — those
    with fewer than ``DEGENERATE_OUTCOME_FLOOR`` outcomes on their minority side.
    There the bootstrap is not measuring outcome variability at all: a 0-for-486
    block resamples to an SE of 0.043 against a true 1.42, understating by 33x,
    because the only thing that moves between resamples is the continuity
    correction's denominator.

    Testing the bootstrap's *magnitude* instead does not work: it hits exactly 0
    only when every day carries the same number of observations, so uneven blocks
    slip through. Applying ``max(bootstrap, parametric)`` everywhere is the other
    failure — it binds on most blocks and drives between-block variance to 0.

    Effective n is the distinct day count throughout, for both the continuity
    correction and the variance.
    """
    by_day: dict[str, list[float]] = {}
    for _mid, y, day in block.members:
        by_day.setdefault(day, []).append(y)
    wins_total = sum(m[1] for m in block.members)
    obs = _logit(_block_p(wins_total, float(block.n_obs)))
    # Parametric fallback is expressed entirely on the effective (day) scale.
    p_eff = _block_p(wins_total * block.n_days / block.n_obs, float(block.n_days))
    parametric = math.sqrt(1.0 / max(block.n_days * p_eff * (1.0 - p_eff), 1e-9))

    days = list(by_day)
    if len(days) < 2:
        return obs, max(parametric, SE_FLOOR)

    rng = random.Random(20260829)
    draws: list[float] = []
    for _ in range(b):
        wins = 0.0
        n = 0
        for _ in range(len(days)):
            ys = by_day[days[rng.randrange(len(days))]]
            wins += sum(ys)
            n += len(ys)
        if n:
            draws.append(_logit(_block_p(wins, float(n))))
    if len(draws) < 2:
        return obs, max(parametric, SE_FLOOR)
    mean = sum(draws) / len(draws)
    var = sum((x - mean) ** 2 for x in draws) / len(draws)
    boot = var**0.5
    minority = min(wins_total, block.n_obs - wins_total)
    if minority < DEGENERATE_OUTCOME_FLOOR:
        return obs, max(parametric, boot, SE_FLOOR)
    return obs, max(boot, SE_FLOOR)


def fit_logit_trend(
    blocks: list[Block],
    stats: list[tuple[float, float]],
    *,
    min_days: int = MIN_TREND_DAYS,
) -> tuple[float, float]:
    """Weighted least squares of ``logit(win rate)`` on ``logit(mean mid)``.

    Returns (alpha, beta) for ``logit(g) = alpha + beta * logit(mid)``. Weights
    are distinct-day counts, not inverse variance: deep-longshot blocks are
    near-degenerate and inverse-variance weighting would hand them the leverage
    over a trend that must also serve the favourite region.

    Blocks thinner than ``min_days`` are still shrunk toward the trend but do not
    help estimate it. Falls back to identity-in-odds (0, 1) rather than emit a
    non-positive or non-finite beta, which would invert or flatten the curve.
    """
    pts = [
        (_logit(b.mean_mid), obs, float(b.n_days))
        for b, (obs, _se) in zip(blocks, stats)
        if b.n_days >= min_days
    ]
    if len(pts) < 2:
        return (0.0, 1.0)  # not enough structure to claim a trend: identity in odds
    sw = sum(w for _, _, w in pts)
    mx = sum(w * x for x, _, w in pts) / sw
    my = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    beta = sxy / sxx if sxx > 0 else 1.0
    if not math.isfinite(beta) or beta <= 0.0:
        return (0.0, 1.0)
    return (my - beta * mx, beta)


def estimate_tau2(
    blocks: list[Block],
    stats: list[tuple[float, float]],
    trend: tuple[float, float],
    *,
    min_days: int = MIN_TREND_DAYS,
) -> tuple[float, float, int]:
    """DerSimonian-Laird between-block variance around the trend, on the logit scale.

    Returns ``(tau2, Q, df)``. ``tau2`` is the dispersion the trend does not
    explain; the shrinkage weight is ``tau2 / (tau2 + se^2)``, so tau2=0 pins
    every block to the trend.

    Inverse-variance weighted, df = k - 2 for the two trend parameters. The
    earlier unweighted ``mean(r^2) - mean(se^2)`` let a single wide block dominate
    the sampling term.

    ``Q < df`` means blocks scatter LESS than their own SEs claim — evidence the
    SEs are inflated, not that the trend is explanatory. Surface it: this
    estimator has little power at the block counts this corpus produces, so
    tau2=0 is weak evidence of homogeneity.
    """
    resid = [
        (_logit(b.mean_mid), obs - (trend[0] + trend[1] * _logit(b.mean_mid)), se)
        for b, (obs, se) in zip(blocks, stats)
        if b.n_days >= min_days
    ]
    k = len(resid)
    if k < 3:
        return 0.0, 0.0, 0
    ws = [1.0 / (se**2) for _, _, se in resid]
    q = sum(w * r**2 for (_, r, _), w in zip(resid, ws))
    df = k - 2

    # Meta-regression denominator: S1 - tr[(X'WX)^-1 X'W^2 X] with X = [1, x].
    # The intercept-only form (sw - sw2/sw) belongs to a random-effects MEAN and
    # is inconsistent with a df of k-2; on this corpus it inflates the
    # denominator ~1.55x and so under-estimates tau2 by ~35%.
    xs = [x for x, _, _ in resid]
    a11 = sum(ws)
    a12 = sum(w * x for w, x in zip(ws, xs))
    a22 = sum(w * x * x for w, x in zip(ws, xs))
    b11 = sum(w * w for w in ws)
    b12 = sum(w * w * x for w, x in zip(ws, xs))
    b22 = sum(w * w * x * x for w, x in zip(ws, xs))
    det = a11 * a22 - a12 * a12
    if det <= 0:
        return 0.0, q, df
    trace = (a22 * b11 - 2.0 * a12 * b12 + a11 * b22) / det
    denom = a11 - trace
    tau2 = 0.0 if denom <= 0 else max(0.0, (q - df) / denom)
    return tau2, q, df


def monotone_project(values: list[float], weights: list[float]) -> list[float]:
    """Weighted isotonic (PAV) projection onto the non-decreasing cone.

    Used instead of a cumulative max: cumulative max only ever raises values, so
    it systematically inflated the curve wherever the evidence was weakest. A
    weighted projection moves violators toward each other, respecting how much
    data each block carries.
    """
    if len(values) != len(weights):
        raise ValueError(f"length mismatch: {len(values)} values, {len(weights)} weights")
    # (value, total weight, number of original entries covered)
    blocks: list[list[float]] = [[v, w, 1] for v, w in zip(values, weights)]
    merged = True
    while merged:
        merged = False
        out: list[list[float]] = []
        for blk in blocks:
            if out and out[-1][0] > blk[0]:
                w = out[-1][1] + blk[1]
                out[-1] = [
                    (out[-1][0] * out[-1][1] + blk[0] * blk[1]) / w if w > 0 else blk[0],
                    w,
                    out[-1][2] + blk[2],
                ]
                merged = True
            else:
                out.append(list(blk))
        blocks = out

    expanded: list[float] = []
    for value, _w, count in blocks:
        expanded.extend([value] * int(count))
    return expanded


def apply_lift_guard(
    blocks: list[Block],
    *,
    min_trend_days: int = MIN_TREND_DAYS,
) -> tuple[list[float], list[float], dict]:
    """Guard v2: shrink each block's lift toward the fitted lift-vs-mid trend.

    Returns (breakpoints, values, diagnostics). Breakpoints are block mean mids.

    For each block, the empirical-Bayes weight ``w = tau2 / (tau2 + se^2)`` decides
    how much of its own measured lift it keeps versus the trend's prediction:
    a well-measured block (small se) keeps most of its own signal, a thin one is
    pulled onto the trend. Blocks with no estimable clustered SE take the trend
    outright. The result is then projected back onto the monotone cone.
    """
    if not blocks:
        raise ValueError("apply_lift_guard requires at least one block")
    stats = [logit_stats(b) for b in blocks]
    trend = fit_logit_trend(blocks, stats, min_days=min_trend_days)
    tau2, q, df = estimate_tau2(blocks, stats, trend, min_days=min_trend_days)

    raw: list[float] = []
    diags: list[dict] = []
    for blk, (obs, se) in zip(blocks, stats):
        z_trend = trend[0] + trend[1] * _logit(blk.mean_mid)
        w = 0.0 if tau2 <= 0.0 else tau2 / (tau2 + se**2)
        g = _expit(z_trend + w * (obs - z_trend))
        raw.append(g)
        diags.append(
            {
                "n_obs": blk.n_obs,
                "n_days": blk.n_days,
                "p_raw": round(_expit(obs), 5),
                "p_trend": round(_expit(z_trend), 5),
                "p_shrunk": round(g, 5),
                "lift_raw": round(blk.lift, 5),
                "lift_shrunk": round(g - blk.mean_mid, 5),
                "se_logit": round(se, 5),
                "shrink_weight": round(w, 4),
            }
        )

    breakpoints = [b.mean_mid for b in blocks]
    values = monotone_project(raw, [float(b.n_obs) for b in blocks])
    # How concentrated is Q? A single degenerate block can supply nearly all of
    # it and manufacture a tau2 that has nothing to do with the traded band.
    contribs = sorted(
        (
            ((obs - (trend[0] + trend[1] * _logit(b.mean_mid))) ** 2) / se**2
            for b, (obs, se) in zip(blocks, stats)
            if b.n_days >= min_trend_days
        ),
        reverse=True,
    )
    max_q_share = (contribs[0] / q) if contribs and q > 0 else 0.0

    mean_w = sum(d["shrink_weight"] for d in diags) / len(diags)
    fit_info = {
        "trend_alpha": round(trend[0], 6),
        "trend_beta": round(trend[1], 6),
        "tau2_logit": round(tau2, 6),
        "heterogeneity_q": round(q, 4),
        "heterogeneity_df": df,
        # Q/df < 1 means the blocks scatter LESS than their SEs claim, i.e. the
        # SEs are inflated — a tau2 of 0 then says little about homogeneity.
        "q_over_df": round(q / df, 4) if df > 0 else None,
        # Share of Q from the single largest contributor. High values mean the
        # heterogeneity is one block, not a property of the curve.
        "max_single_block_q_share": round(max_q_share, 4),
        "mean_shrink_weight": round(mean_w, 4),
        # When this is true the artifact is expit(alpha + beta*logit(mid)) and
        # nothing else — a two-parameter curve, not an isotonic map. It is a
        # materially different object and must not pass unremarked.
        "degenerate_pure_trend": mean_w == 0.0,
        "blocks": diags,
    }
    return breakpoints, values, fit_info


def interp_curve(breakpoints: list[float], values: list[float], x: float) -> float:
    """Evaluate the piecewise-linear curve exactly as ``IsotonicCalibrator`` does.

    Linear between breakpoints, flat clamp outside them. Shared with the offline
    validator so the two cannot drift apart.
    """
    if x <= breakpoints[0]:
        return values[0]
    if x >= breakpoints[-1]:
        return values[-1]
    lo, hi = 0, len(breakpoints) - 1
    while lo < hi - 1:
        m = (lo + hi) // 2
        if breakpoints[m] <= x:
            lo = m
        else:
            hi = m
    span = breakpoints[hi] - breakpoints[lo]
    if span <= 0:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (x - breakpoints[lo]) / span


def cap_values(
    breakpoints: list[float],
    values: list[float],
    cap: float = CAP,
    mid_ceiling: float = MID_CEILING,
) -> tuple[list[float], list[float], int | None]:
    """Cap ``g <= cap`` across the whole band ``mid <= mid_ceiling``.

    Returns ``(breakpoints, values, synthetic_index)``, where ``synthetic_index``
    is the position of the inserted knot or None. Callers holding per-block
    metadata MUST splice a placeholder at that index, or their arrays silently
    misalign against ``breakpoints`` from the knot onward.

    Clamping only the breakpoint values is not enough: the runtime interpolates
    between knots, so if the last in-band knot sits below ``mid_ceiling`` and the
    next one is uncapped, prices just under the ceiling interpolate straight
    through the cap (measured: ``calibrate(0.7499) == 0.893`` against a 0.85 cap).
    A synthetic knot at ``mid_ceiling`` pins the curve at the boundary.
    """
    if len(breakpoints) != len(values):
        raise ValueError(f"length mismatch: {len(breakpoints)} breakpoints, {len(values)} values")
    if not breakpoints:
        raise ValueError("cap_values requires at least one breakpoint")
    capped = [min(v, cap) if bp <= mid_ceiling else v for bp, v in zip(breakpoints, values)]

    # Left of the first knot the curve is a flat clamp at values[0], so a ceiling
    # at or below breakpoints[0] still needs that first value capped.
    if mid_ceiling <= breakpoints[0]:
        capped[0] = min(capped[0], cap)
        return list(breakpoints), capped, None
    if mid_ceiling >= breakpoints[-1] or any(abs(bp - mid_ceiling) < 1e-12 for bp in breakpoints):
        return list(breakpoints), capped, None

    at_ceiling = interp_curve(breakpoints, values, mid_ceiling)
    if at_ceiling <= cap:
        # Curve is already under the cap at the ceiling, and it is non-decreasing,
        # so everything below is too. Inserting a knot would change nothing.
        return list(breakpoints), capped, None

    idx = next(i for i, bp in enumerate(breakpoints) if bp > mid_ceiling)
    out_bp = breakpoints[:idx] + [mid_ceiling] + breakpoints[idx:]
    out_v = capped[:idx] + [cap] + capped[idx:]
    return out_bp, out_v, idx


def build_artifact(
    breakpoints: list[float],
    values: list[float],
    blocks: list[Block],
    fit_info: dict,
    synthetic_index: int | None = None,
) -> dict:
    # Every per-block array stays index-aligned with `breakpoints`. The synthetic
    # cap knot is not a block, so it gets an explicit placeholder rather than
    # shifting every later entry by one.
    n_obs: list[int | None] = [b.n_obs for b in blocks]
    n_days: list[int | None] = [b.n_days for b in blocks]
    per_block: list[dict | None] = list(fit_info["blocks"])
    if synthetic_index is not None:
        n_obs.insert(synthetic_index, None)
        n_days.insert(synthetic_index, None)
        per_block.insert(synthetic_index, {"synthetic_cap_knot": True})
    fit_info = {**fit_info, "blocks": per_block}

    return {
        "breakpoints": breakpoints,
        "values": values,
        "metadata": {
            "feature": "mid",
            "fit_date": datetime.now(UTC).date().isoformat(),
            "synthetic_knot_index": synthetic_index,
            "n_per_bucket": n_obs,
            "n_days_per_bucket": n_days,
            "fit_info": fit_info,
            "scope": "greater_less_lead1_fullpop",
            "guard_version": 2,
            "guards": {
                "shrinkage": "empirical_bayes_toward_lift_vs_mid_trend",
                "lift_bootstrap_b": LIFT_BOOTSTRAP_B,
                "se_floor": SE_FLOOR,
                "min_trend_days": MIN_TREND_DAYS,
                "monotone_step": "weighted_isotonic_projection",
                "cap": CAP,
                "mid_ceiling": MID_CEILING,
                "breakpoint_location": "block_mean_mid",
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "output artifact path. REQUIRED — pass the live calibrator path "
            f"({OUT}) only when you intend to replace the map the live strategy "
            "serves; write a candidate path first and inspect it."
        ),
    )
    args = ap.parse_args()
    out: Path = args.out

    pairs = build_pairs(Path("data/snapshots"))
    print(f"pairs: {len(pairs)}  days: {len({p[2] for p in pairs})}")
    blocks = pav_isotonic(pairs)
    bp, vals, fit_info = apply_lift_guard(blocks)
    bp, vals, synthetic_index = cap_values(bp, vals)

    # `_validate_curve` checks monotonicity only, so an out-of-range value would
    # load cleanly and reach Kelly as a probability outside [0, 1].
    assert bp == sorted(bp), "breakpoints must be non-decreasing"
    assert vals == sorted(vals), "values must be non-decreasing"
    assert all(0.0 < v < 1.0 for v in vals), (
        f"values must be probabilities: got [{min(vals):.4f}, {max(vals):.4f}]"
    )
    spread = vals[-1] - vals[0]
    assert spread > 0.1, f"degenerate curve (spread={spread:.3f})"

    art = build_artifact(bp, vals, blocks, fit_info, synthetic_index)
    assert len(art["metadata"]["n_per_bucket"]) == len(bp), "metadata must align with breakpoints"

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(art, fh, indent=2)

        # Validate BEFORE publishing: a curve that fails here would otherwise land
        # on the live path and silently identity-fall-back.
        from python.data.calibration import IsotonicCalibrator

        probe = IsotonicCalibrator(Path(tmp))
        if not probe.is_loaded:
            raise AssertionError(f"artifact failed to load, not published: {probe.fallback_reason}")
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    print(
        f"wrote {out}: {len(bp)} breakpoints, spread={spread:.3f}, "
        f"beta={fit_info['trend_beta']:.3f}, tau2={fit_info['tau2_logit']:.4f}, "
        f"Q/df={fit_info['q_over_df']}, mean shrink weight="
        f"{fit_info['mean_shrink_weight']:.3f}"
    )
    if fit_info["degenerate_pure_trend"]:
        print(
            "  NOTE: tau2=0 — every block pinned to the trend. This artifact is\n"
            "  expit(alpha + beta*logit(mid)), a 2-parameter curve, not an isotonic map."
        )
    if fit_info["max_single_block_q_share"] > 0.5:
        print(
            f"  WARNING: {fit_info['max_single_block_q_share']:.1%} of Q comes from a"
            " SINGLE block.\n  The heterogeneity driving tau2 is one block, not the curve"
            " — do not trust tau2."
        )


if __name__ == "__main__":
    main()
