"""ASOS continuous-vs-sampled high gap measurement probe.

Builds an LST-framed continuous-high proxy from 1-min ASOS, validates it
against real CLI settlements, decomposes the gap vs the hourly-sampled max,
simulates the economic effect via asos_posterior, and emits a go/no-go verdict.

No runtime change. Pure helpers are unit-tested; I/O glue runs in main().
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path so the script works standalone (same pattern as
# scripts/build_asos_climatology.py and scripts/build_emos_corpus.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.common import Direction  # noqa: E402
from python.data.asos_posterior import bracket_probability  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Re-declared offsets; equality to python/data/asos.py is enforced by the frame-parity test in
# tests/test_research_asos_continuous_gap.py.
CITY_STD_OFFSET_HOURS: dict[str, int] = {
    "NYC": -5, "CHI": -6, "LAX": -8, "MIA": -5, "AUS": -6,
}


def daily_max_by_lst_day(
    rows: list[tuple[datetime, float]], offset_hours: int
) -> dict[date, float]:
    """Max tmpf per LOCAL-STANDARD-TIME climate day.

    `rows` are (tz-aware UTC datetime, tmpf). The LST day uses a FIXED offset
    (never DST), bit-identical to python/data/asos.py CITY_TZ framing.
    """
    tz = timezone(timedelta(hours=offset_hours))
    out: dict[date, float] = {}
    for dt, t in rows:
        d = dt.astimezone(tz).date()
        if d not in out or t > out[d]:
            out[d] = t
    return out


def decompose_gap(cli_high: float, h1min: float, m_hourly: float) -> dict[str, float]:
    """Split the total understatement into recoverable vs irreducible parts.

    between_sample (lever A, a faster feed could recover) = h1min - m_hourly
    quant_residual (lever B, irreducible by cadence)      = cli_high - h1min
    total                                                  = cli_high - m_hourly
    """
    return {
        "between_sample": h1min - m_hourly,
        "quant_residual": cli_high - h1min,
        "total": cli_high - m_hourly,
    }


def bracket_prob_shift(
    strike_type: Direction,
    floor: float | None,
    cap: float | None,
    M: float,  # noqa: N803
    mu_gain: float,
    sigma_gain: float,
    nu: float,
    m_correction: float,
) -> tuple[float, float, float]:
    """Probability of a bracket under raw vs gap-corrected M.

    `bracket_probability` (asos_posterior.py:43) takes a Direction enum and
    KEYWORD-ONLY floor/cap/M/mu_gain/sigma_gain/nu. Raising M by `m_correction`
    (the conditional gap) shifts the posterior up. Returns (base, corrected, delta).
    NOTE: simulates the bound via runtime M; the production fix corrects the
    climatology build TARGET, not runtime M.
    """
    base = bracket_probability(
        strike_type, floor=floor, cap=cap, M=M,
        mu_gain=mu_gain, sigma_gain=sigma_gain, nu=nu,
    )
    corrected = bracket_probability(
        strike_type, floor=floor, cap=cap, M=M + m_correction,
        mu_gain=mu_gain, sigma_gain=sigma_gain, nu=nu,
    )
    return base, corrected, corrected - base


def _percentile(sorted_xs: list[float], pct: float) -> float:
    if not sorted_xs:
        return float("nan")
    k = max(0, min(len(sorted_xs) - 1, int(round(pct / 100.0 * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def day_clustered_bootstrap_ci(
    day_values: list[float],
    n_resamples: int = 500,
    seed: int = 0,
    lo_pct: float = 2.5,
    hi_pct: float = 97.5,
) -> tuple[float, float]:
    """95% CI on the MEAN of per-city-day G values via a day-level bootstrap.

    Each element is already one independent city-day (one G scalar), so resampling
    the list with replacement IS day-clustered. Deterministic given `seed`.
    """
    if not day_values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(day_values)
    means = []
    for _ in range(n_resamples):
        sample = [day_values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return (_percentile(means, lo_pct), _percentile(means, hi_pct))


def loo_month_stable(
    month_values: list[tuple[int, float]],
    threshold: float = 0.5,
) -> bool:
    """True iff dropping any single calendar month does not flip sign(mean - threshold).

    Needs >= 2 distinct months to be assessable; returns False otherwise (an
    unassessable / underpowered case is treated as not-stable).
    """
    months = {m for m, _v in month_values}
    if len(months) < 2 or not month_values:
        return False
    full = statistics.fmean(v for _m, v in month_values)
    full_sign = (full - threshold) >= 0
    for drop in months:
        kept = [v for m, v in month_values if m != drop]
        if not kept:
            return False
        if ((statistics.fmean(kept) - threshold) >= 0) != full_sign:
            return False
    return True


ONEMIN_DIR = REPO / "data" / "research" / "asos_1min"
HOURLY_DIR = REPO / "data" / "research" / "asos"
SNAPSHOTS = REPO / "data" / "snapshots"


def load_obs_csv(path: Path) -> list[tuple[datetime, float]]:
    """Read a station CSV (station,valid,tmpf) into (UTC datetime, tmpf)."""
    rows: list[tuple[datetime, float]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            raw = (r.get("tmpf") or "").strip()
            if raw in ("", "M", "T"):
                continue
            try:
                t = float(raw)
            except ValueError:
                continue
            v = r["valid"].strip().replace("T", " ")
            try:
                dt = datetime.strptime(v[:16], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except ValueError:
                continue
            rows.append((dt, t))
    return rows


def load_cli_highs() -> dict[tuple[str, date], float]:
    """{(city, observation_date): high_f} from FINAL (non-preliminary) CLI rows.

    Keeps the row with the LATEST snapshot_ts per (city, date) so a re-collected
    final never loses to an earlier one (guards the "later overwrites = final"
    assumption when two non-preliminary rows exist).
    """
    best: dict[tuple[str, date], tuple[float, float]] = {}  # key -> (snapshot_ts, high)
    for sf in sorted(SNAPSHOTS.glob("**/settlements.json")):
        for rec in json.loads(sf.read_text()):
            if rec.get("is_preliminary"):
                continue
            high = rec.get("high_f")
            if high is None:
                continue
            try:
                d = date.fromisoformat(rec["observation_date"])
            except (KeyError, ValueError):
                continue
            key = (rec["city"], d)
            ts = float(rec.get("snapshot_ts", 0.0))
            if key not in best or ts >= best[key][0]:
                best[key] = (ts, float(high))
    return {k: v[1] for k, v in best.items()}


def diurnal_amplitude_by_lst_day(
    rows: list[tuple[datetime, float]], offset_hours: int
) -> dict[date, float]:
    """max - min tmpf per LST day; a temp-only clear-day proxy (bigger = clearer)."""
    tz = timezone(timedelta(hours=offset_hours))
    hi: dict[date, float] = {}
    lo: dict[date, float] = {}
    for dt, t in rows:
        d = dt.astimezone(tz).date()
        if d not in hi or t > hi[d]:
            hi[d] = t
        if d not in lo or t < lo[d]:
            lo[d] = t
    return {d: hi[d] - lo[d] for d in hi}


def _summ(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 3),
        "median": round(statistics.median(xs), 3),
        "p95": round(_percentile(s, 95.0), 3),
        "stdev": round(statistics.pstdev(xs), 3) if len(xs) > 1 else 0.0,
    }


def _proxy_valid(q_cli: list[float], h1min_overlap: list[float]) -> bool:
    """Not mean-only: a warm-day-conditional bias must not pass (spec §3.3)."""
    if not q_cli or len(q_cli) != len(h1min_overlap):
        return False
    if abs(statistics.fmean(q_cli)) > 1.0:
        return False
    med = statistics.median(h1min_overlap)
    warm = [q for q, h in zip(q_cli, h1min_overlap, strict=True) if h >= med]
    cold = [q for q, h in zip(q_cli, h1min_overlap, strict=True) if h < med]
    for half in (warm, cold):
        if half and abs(statistics.fmean(half)) > 1.5:
            return False
    return True


def analyze_city(
    city: str,
    onemin_rows: list[tuple[datetime, float]],
    hourly_rows: list[tuple[datetime, float]],
    cli: dict[tuple[str, date], float],
) -> dict:
    """Build the headline settlement-bias G, decomposition, and CLI validation."""
    off = CITY_STD_OFFSET_HOURS[city]
    has_1min = bool(onemin_rows)
    h1min = daily_max_by_lst_day(onemin_rows, off) if has_1min else {}
    # M_hourly ALWAYS from the hourly corpus -> bit-identical to the climatology's
    # day_extremum (NOT a downsampled 1-min series). This is the live-seen max.
    m_hourly = daily_max_by_lst_day(hourly_rows, off)
    amp = diurnal_amplitude_by_lst_day(hourly_rows, off)

    # Headline: settlement bias G = CLI - M_hourly per CLI city-day.
    g_days: list[tuple[int, float]] = []   # (month, G)
    q_cli: list[float] = []
    h1min_overlap: list[float] = []
    quant_resid: list[float] = []          # q = CLI - h1min (lever B, overlap)
    between_overlap: list[float] = []      # b = h1min - m_hourly on the SAME overlap days
    g_overlap: list[float] = []            # G = CLI - m_hourly on the SAME overlap days
    for (c, d), high in cli.items():
        if c != city or d not in m_hourly:
            continue
        g_days.append((d.month, high - m_hourly[d]))
        if d in h1min:
            q_cli.append(high - h1min[d])
            h1min_overlap.append(h1min[d])
            quant_resid.append(high - h1min[d])
            between_overlap.append(h1min[d] - m_hourly[d])
            g_overlap.append(high - m_hourly[d])

    # Decomposition lever A (between-sample miss). Two cuts:
    #  - _2yr: powered on the full 1-min corpus (h1min - m_hourly, all days).
    #  - _overlap: on the SAME ~CLI-overlap days as G and lever B, so the additive
    #    identity holds: mean(between_overlap) + mean(quant_resid) == mean(G_overlap).
    between = [h1min[d] - m_hourly[d] for d in h1min if d in m_hourly]

    # Clear-day split of G by amplitude tercile (reporting only).
    amp_vals = sorted(a for a in (amp.get(d) for (c, d) in cli if c == city) if a is not None)
    clear_cut = amp_vals[int(0.66 * len(amp_vals))] if amp_vals else None
    g_clear = [
        high - m_hourly[d]
        for (c, d), high in cli.items()
        if c == city and d in m_hourly and clear_cut is not None
        and amp.get(d, -1.0) >= clear_cut
    ]

    g_vals = [g for _m, g in g_days]
    # G is whole-degree-quantized (CLI is integer; tmpf is reported in whole degF),
    # so G is a count of +0/+1/+2 events, not a smooth sub-degree offset.
    g_hist: dict[str, int] = {}
    for g in g_vals:
        k = str(int(round(g)))
        g_hist[k] = g_hist.get(k, 0) + 1
    return {
        "city": city,
        "has_1min": has_1min,
        "n_cli_days": len(g_days),
        "G_value_counts_intF": dict(sorted(g_hist.items())),
        "coverage": {
            "n_1min_rows": len(onemin_rows),
            "n_1min_days": len(h1min),
            "n_hourly_days": len(m_hourly),
            "n_overlap_days": len(g_overlap),
            "note": (
                "Sparse 1-min coverage (low n_1min_days vs n_hourly_days) inflates the "
                "1-min-derived stats and can drive leverA negative / PROXY_INVALID."
            ),
        },
        "G_settlement_bias": _summ(g_vals),
        "G_clear_days_top_tercile": _summ(g_clear),
        "decomposition": {
            "between_sample_leverA_2yr": _summ(between),
            "between_sample_leverA_overlap": _summ(between_overlap),
            "quant_residual_leverB_overlap": _summ(quant_resid),
            "G_overlap_for_identity": _summ(g_overlap),
            "caveat": (
                "Additive identity holds on the OVERLAP cut: "
                "mean(between_sample_leverA_overlap) + mean(quant_residual_leverB_overlap) "
                "== mean(G_overlap_for_identity) (NOT G_settlement_bias, which spans the "
                "larger all-CLI-days sample; the _2yr leverA is a different, larger "
                "day-sample and will NOT add up). Critically, the raw 1-min max can "
                "OVERSHOOT the official 5-min-averaged CLI high, so leverB "
                "(CLI - h1min) goes negative for some cells -- h1min is NOT bounded by "
                "settlement. The clean 'leverA recoverable / leverB irreducible floor' "
                "framing therefore does NOT hold; a faster real-time feed could read "
                "ABOVE the settlement value. Treat the headline G (CLI - hourly max), "
                "which is independent of 1-min, as the robust result."
            ),
        },
        "cli_validation": {
            "q_residual_cli_minus_h1min": _summ(q_cli),
            "proxy_valid": _proxy_valid(q_cli, h1min_overlap),
        },
        "_g_days": g_days,  # consumed by verdict_for_city, dropped before serialize
    }


def verdict_for_city(rec: dict, min_cli_days: int = 30) -> str:
    """GREEN / RED / UNDERPOWERED / PROXY_INVALID on the settlement bias (spec §3.6).

    Note: the economic boundary-shift confirmation (bracket_prob_shift over the
    trade cohort) is applied in the MEMO (Task 8), not here — this is the
    statistical core.
    """
    if rec["has_1min"] and not rec["cli_validation"]["proxy_valid"]:
        return "PROXY_INVALID"
    g_days: list[tuple[int, float]] = rec["_g_days"]
    if len(g_days) < min_cli_days:
        return "UNDERPOWERED"
    g_vals = [g for _m, g in g_days]
    mean_g = statistics.fmean(g_vals)
    lo, hi = day_clustered_bootstrap_ci(g_vals, n_resamples=1000, seed=1)
    if hi < 0.5:
        return "RED"
    if mean_g >= 0.5 and lo >= 0.3 and loo_month_stable(g_days, threshold=0.5):
        return "GREEN"
    return "UNDERPOWERED"  # CI straddles the threshold


def economic_gate(mean_g: float) -> list[dict]:
    """Reproducible bracket_prob_shift over a FIXED near-boundary grid (spec §3.6).

    Raises runtime M by `mean_g` -- an UPPER BOUND on the true benefit; the real
    fix lowers the climatology `mu_gain` by ~mean_g (sigma unchanged), a smaller
    effect. Purpose: show a mean-G correction moves near-boundary bracket
    probabilities by an economically meaningful amount. All params are explicit so
    the numbers regenerate exactly.
    """
    setups = [
        ("ABOVE floor=90, M=89.5", Direction.ABOVE, 90.0, None, 89.5, 0.25, 0.8),
        ("BETWEEN 88-89, M=88.5", Direction.BETWEEN, 88.0, 89.0, 88.5, 0.25, 0.8),
        ("BELOW cap=90, M=89.5", Direction.BELOW, None, 90.0, 89.5, 0.25, 0.8),
    ]
    out = []
    for label, st, floor, cap, m, mu, sg in setups:
        base, corr, delta = bracket_prob_shift(st, floor, cap, m, mu, sg, 5.0, mean_g)
        out.append(
            {
                "setup": label, "mu_gain": mu, "sigma_gain": sg, "nu": 5.0,
                "m_correction": round(mean_g, 3),
                "base": round(base, 3), "corrected": round(corr, 3),
                "delta": round(delta, 3),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "research" / "2026-06-10-asos-continuous-gap.json",
    )
    args = ap.parse_args(argv)

    cli = load_cli_highs()
    results = []
    for city in CITY_STD_OFFSET_HOURS:
        onemin = load_obs_csv(ONEMIN_DIR / f"k{city.lower()}_asos_1min.csv")
        hourly = load_obs_csv(HOURLY_DIR / f"k{city.lower()}_asos.csv")
        rec = analyze_city(city, onemin, hourly, cli)
        rec["verdict"] = verdict_for_city(rec)
        rec.pop("_g_days", None)  # drop the internal field before serialize
        results.append(rec)

    green = [r["city"] for r in results if r["verdict"] == "GREEN"]
    payload = {
        "script": "scripts/research_2026_06_10_asos_continuous_gap.py",
        "cli_days_loaded": len(cli),
        "decomposition_caveat": (
            "Per-city verdicts are independent; GREEN_cities lists only cities that "
            "individually cleared the gate. h1min lever split is a bound (see per-city)."
        ),
        "GREEN_cities": green,
        "economic_gate_upper_bound": {
            "note": (
                "bracket_prob_shift raising runtime M by mean(G) -- an UPPER bound; "
                "the real fix lowers climatology mu_gain by ~mean(G). Confirms a "
                "mean-G correction moves near-boundary bracket probabilities materially."
            ),
            "by_city": {
                r["city"]: economic_gate(r["G_settlement_bias"]["mean"])
                for r in results
                if r["verdict"] == "GREEN"
            },
        },
        "cities": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
