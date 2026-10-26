"""Refresh the NEW-CITY favorite-longshot analysis on the current corpus (2026-06-23).

Decides whether expanding the live ``favorite_bias`` strategy to Kalshi's 11 new
weather cities (ATL DAL DC HOU LV MIN NOLA OKC PHX SATX SFO) is any closer to
viable, vs the 2026-06-18 baseline (wrong-signed -0.069, ROI ~-26%, supply-starved).

Mandatory methodology (every prior false-positive in this repo violated one):
  1. Fill at the ASK, never mid/bid (order_fee taker).
  2. Lead = forecast_date - snapshot_calendar_day (lead-1 only).
  3. FINAL settlements only (load_snapshot_data enforces).
  4. ALL CIs day-clustered (block-bootstrap over forecast_date, >=2000 resamples).
  5. Live gate exactly: band [0.40,0.75), vol_fp>=200, {greater,less}, edge_floor 0.01,
     YES-only, one bet per (city,var,forecast_date) (latest slot per ticker).

The g(mid) map is fit ONLY on the mature-9 cities, so for new cities it is an
out-of-domain transfer — flagged explicitly in the ROI section.

Run: PYTHONPATH=. uv run python scripts/research/newcity_favorite_refresh.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from python.analytics.fees import fee_per_contract, order_fee
from python.data.calibration import IsotonicCalibrator
from scripts.research._fb_data import Cand, load_candidates

RNG = np.random.default_rng(20260623)
LIVE_MAP = Path("data/calibration/favorite_bias_isotonic.json")
OUT = Path("data/research/2026-06-23-newcity-favorite-refresh.json")

# --- live gate constants (verified vs python/config.yaml 2026-06-23) ---
BAND_LO, BAND_HI = 0.40, 0.75
EDGE_FLOOR = 0.01
MIN_VOL = 200
N_BOOT = 4000
LEAD = 1

NEW_CITIES = frozenset({"ATL", "DAL", "DC", "HOU", "LV", "MIN", "NOLA", "OKC", "PHX", "SATX", "SFO"})
MATURE_CITIES = frozenset({"NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "PHIL", "BOS"})
BUCKETS = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.75)]

# Prior 2026-06-18 baseline (for the change-vs-prior delta).
PRIOR = {
    "newcity_bias_overall": -0.069,
    "newcity_n_inband": 21,
    "newcity_roi": -0.26,
    "mature_bias_control": 0.124,
    "frac_extreme_outside_band": 0.628,
}


# ---------------------------------------------------------------------------
# P&L primitives
# ---------------------------------------------------------------------------
def leg_pnl(c: Cand, *, slip: float = 0.0) -> tuple[float, float]:
    """(net_pnl, cost) buying 1 YES at ask(+slip), taker fee. payout 1 if YES else 0."""
    a = min(0.99, c.ask + slip)
    fee = order_fee(a, 1, is_maker=False)
    cost = a + fee
    payout = 1.0 if c.yes else 0.0
    return payout - cost, cost


def wilson_lb(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def day_clustered_roi_ci(rows: list[Cand], *, slip: float = 0.0, n_boot: int = N_BOOT):
    """Block-bootstrap ROI over forecast DATES. Returns (lo, hi, point)."""
    if not rows:
        return 0.0, 0.0, 0.0
    by_day: dict = defaultdict(list)
    p_pt = w_pt = 0.0
    for c in rows:
        lp, cost = leg_pnl(c, slip=slip)
        by_day[c.fdate].append((lp, cost))
        p_pt += lp
        w_pt += cost
    point = p_pt / w_pt if w_pt else 0.0
    days = list(by_day.keys())
    rois = np.empty(n_boot)
    for b in range(n_boot):
        chosen = RNG.choice(len(days), size=len(days), replace=True)
        p = w = 0.0
        for di in chosen:
            for lp, cost in by_day[days[di]]:
                p += lp
                w += cost
        rois[b] = p / w if w else 0.0
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi), float(point)


def day_clustered_mean_ci(values_by_day: dict, *, n_boot: int = N_BOOT):
    """Block-bootstrap the day-pooled MEAN of a scalar (e.g. realized-YES-minus-mid).

    values_by_day maps forecast_date -> list of per-trade scalars. The statistic
    is the grand mean over all trades, resampled by clustering on day.
    Returns (lo, hi, point).
    """
    if not values_by_day:
        return 0.0, 0.0, 0.0
    days = list(values_by_day.keys())
    allv = [v for vs in values_by_day.values() for v in vs]
    point = float(np.mean(allv)) if allv else 0.0
    means = np.empty(n_boot)
    for b in range(n_boot):
        chosen = RNG.choice(len(days), size=len(days), replace=True)
        acc = []
        for di in chosen:
            acc.extend(values_by_day[days[di]])
        means[b] = float(np.mean(acc)) if acc else 0.0
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), point


# ---------------------------------------------------------------------------
# Gate + event de-dup
# ---------------------------------------------------------------------------
def in_band(c: Cand) -> bool:
    return c.stype in ("greater", "less") and (BAND_LO <= c.mid < BAND_HI)


def passes_full_gate(c: Cand, g) -> bool:
    if not in_band(c):
        return False
    if c.vol < MIN_VOL:
        return False
    p = g.calibrate(c.mid)
    fee = fee_per_contract(c.ask, is_maker=False)
    return (p - c.ask - fee) >= EDGE_FLOOR


def latest_per_event(rows: list[Cand]) -> list[Cand]:
    """One bet per (city,var,forecast_date): keep the latest-slot candidate."""
    best: dict[tuple, Cand] = {}
    for c in rows:
        k = (c.city, c.var, c.fdate)
        if k not in best or c.snap_ts > best[k].snap_ts:
            best[k] = c
    return sorted(best.values(), key=lambda c: (c.fdate, c.ticker))


# ---------------------------------------------------------------------------
def summarize_roi(rows: list[Cand], *, label: str, slip: float = 0.0) -> dict:
    n = len(rows)
    wins = sum(c.yes for c in rows)
    pnl = sum(leg_pnl(c, slip=slip)[0] for c in rows)
    lo, hi, point = day_clustered_roi_ci(rows, slip=slip)
    return {
        "label": label,
        "n": n,
        "n_events": len({(c.city, c.var, c.fdate) for c in rows}),
        "n_days": len({c.fdate for c in rows}),
        "win_rate": wins / n if n else 0.0,
        "win_wilson_lb": wilson_lb(wins, n),
        "roi": point,
        "roi_day_ci": [lo, hi],
        "net_per_contract_cents": (pnl / n * 100) if n else 0.0,
        "avg_ask": (sum(c.ask for c in rows) / n) if n else 0.0,
        "avg_mid": (sum(c.mid for c in rows) / n) if n else 0.0,
    }


def bias_by_bucket(rows_inband: list[Cand], *, label: str) -> dict:
    """realized-YES-minus-mid, overall + per bucket, day-clustered CI.
    rows_inband: one-per-event in-band greater/less rows (the cleanest model-free signal)."""
    out: dict = {"label": label, "n_inband": len(rows_inband)}
    # overall
    by_day = defaultdict(list)
    for c in rows_inband:
        by_day[c.fdate].append(c.yes - c.mid)
    lo, hi, pt = day_clustered_mean_ci(by_day)
    out["overall"] = {
        "n": len(rows_inband),
        "realized_yes": (sum(c.yes for c in rows_inband) / len(rows_inband)) if rows_inband else 0.0,
        "avg_mid": (sum(c.mid for c in rows_inband) / len(rows_inband)) if rows_inband else 0.0,
        "bias": pt,
        "bias_day_ci": [lo, hi],
    }
    out["buckets"] = []
    for blo, bhi in BUCKETS:
        sub = [c for c in rows_inband if blo <= c.mid < bhi]
        bd = defaultdict(list)
        for c in sub:
            bd[c.fdate].append(c.yes - c.mid)
        blo_ci, bhi_ci, bpt = day_clustered_mean_ci(bd)
        out["buckets"].append({
            "bucket": f"[{blo:.2f},{bhi:.2f})",
            "n": len(sub),
            "realized_yes": (sum(c.yes for c in sub) / len(sub)) if sub else None,
            "avg_mid": (sum(c.mid for c in sub) / len(sub)) if sub else None,
            "bias": bpt if sub else None,
            "bias_day_ci": [blo_ci, bhi_ci] if sub else None,
        })
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    g = IsotonicCalibrator(LIVE_MAP)
    assert g.is_loaded, getattr(g, "fallback_reason", "calibrator failed to load")

    # Load ALL cities at lead-1 (do NOT inherit the mature-9 corpus filter).
    cands_all = load_candidates(leads=(LEAD,), restrict_cities=None)
    new_c = [c for c in cands_all if c.city in NEW_CITIES]
    mat_c = [c for c in cands_all if c.city in MATURE_CITIES]

    # Restrict the mature control to the SAME forecast-date window as new cities.
    if new_c:
        nlo = min(c.fdate for c in new_c)
        nhi = max(c.fdate for c in new_c)
    else:
        nlo = nhi = None
    mat_window = [c for c in mat_c if nlo and nlo <= c.fdate <= nhi]

    out: dict = {
        "meta": {
            "generated": "2026-06-23",
            "lead": LEAD,
            "band": [BAND_LO, BAND_HI],
            "min_vol": MIN_VOL,
            "edge_floor": EDGE_FLOOR,
            "n_boot": N_BOOT,
            "new_city_window": [str(nlo), str(nhi)],
            "g_map": str(LIVE_MAP),
            "prior_baseline_2026_06_18": PRIOR,
            "n_new_lead1_candidates": len(new_c),
            "n_mature_lead1_candidates_in_window": len(mat_window),
        }
    }

    # one-per-event greater/less universe (latest slot)
    new_gl = latest_per_event([c for c in new_c if c.stype in ("greater", "less")])
    mat_gl = latest_per_event([c for c in mat_window if c.stype in ("greater", "less")])

    # ================================================================
    # A. SUPPLY CENSUS
    # ================================================================
    print("=" * 96)
    print("A. SUPPLY CENSUS — new-city greater/less universe at lead-1 (one per event)")
    print("=" * 96)

    def census(rows: list[Cand]) -> dict:
        n_total = len(rows)
        inband = [c for c in rows if in_band(c)]
        extreme_lo = [c for c in rows if c.mid < BAND_LO]
        extreme_hi = [c for c in rows if c.mid >= BAND_HI]
        inband_volok = [c for c in inband if c.vol >= MIN_VOL]
        return {
            "n_greater_less_events": n_total,
            "n_in_band": len(inband),
            "n_in_band_vol_ok": len(inband_volok),
            "n_extreme_below_0.40": len(extreme_lo),
            "n_extreme_at_or_above_0.75": len(extreme_hi),
            "pct_extreme_outside_band": ((len(extreme_lo) + len(extreme_hi)) / n_total) if n_total else 0.0,
            "pct_in_band": (len(inband) / n_total) if n_total else 0.0,
        }

    out["A_supply_census"] = {"new_cities": census(new_gl), "mature_control": census(mat_gl)}
    nc = out["A_supply_census"]["new_cities"]
    print(f"  NEW: {nc['n_greater_less_events']} g/l events | in-band {nc['n_in_band']} "
          f"({nc['pct_in_band']*100:.1f}%) | in-band&vol>=200 {nc['n_in_band_vol_ok']} | "
          f"extreme outside band {nc['pct_extreme_outside_band']*100:.1f}% "
          f"(below {nc['n_extreme_below_0.40']}, above {nc['n_extreme_at_or_above_0.75']})")
    print(f"  PRIOR 06-18 extreme-outside-band: {PRIOR['frac_extreme_outside_band']*100:.1f}%")

    # by-month breakdown (heat-lock onset)
    def by_month_census(rows: list[Cand]) -> dict:
        m: dict = defaultdict(lambda: {"total": 0, "inband": 0})
        for c in rows:
            key = f"{c.fdate.year}-{c.fdate.month:02d}"
            m[key]["total"] += 1
            if in_band(c):
                m[key]["inband"] += 1
        return {k: {**v, "pct_inband": (v["inband"] / v["total"]) if v["total"] else 0.0}
                for k, v in sorted(m.items())}
    out["A_supply_census"]["new_by_month"] = by_month_census(new_gl)
    print(f"  NEW by month (inband/total): "
          f"{ {k: f'{v[chr(105)+chr(110)+chr(98)+chr(97)+chr(110)+chr(100)]}/{v[chr(116)+chr(111)+chr(116)+chr(97)+chr(108)]}' for k,v in out['A_supply_census']['new_by_month'].items()} }")

    # per-city in-band breakdown
    pc = Counter()
    pc_ib = Counter()
    for c in new_gl:
        pc[c.city] += 1
        if in_band(c):
            pc_ib[c.city] += 1
    out["A_supply_census"]["new_per_city_inband"] = {
        city: {"g_l_events": pc[city], "in_band": pc_ib[city]} for city in sorted(pc)
    }

    # ================================================================
    # B. BIAS EXISTENCE TEST (model-free, cleanest signal)
    # ================================================================
    print("\n" + "=" * 96)
    print("B. BIAS TEST — realized-YES-minus-mid, in-band (one per event), day-clustered CI")
    print("=" * 96)
    new_inband = [c for c in new_gl if in_band(c)]
    mat_inband = [c for c in mat_gl if in_band(c)]
    out["B_bias"] = {
        "new_cities": bias_by_bucket(new_inband, label="new"),
        "mature_control": bias_by_bucket(mat_inband, label="mature"),
    }
    for lbl, blk in (("NEW", out["B_bias"]["new_cities"]), ("MATURE", out["B_bias"]["mature_control"])):
        o = blk["overall"]
        ci = o["bias_day_ci"]
        print(f"  {lbl}: n_inband={blk['n_inband']} realizedYES={o['realized_yes']*100:.1f}% "
              f"avgMid={o['avg_mid']*100:.1f}% BIAS={o['bias']*100:+.2f}pp "
              f"CI[{ci[0]*100:+.1f},{ci[1]*100:+.1f}]")
        for b in blk["buckets"]:
            if b["n"]:
                bc = b["bias_day_ci"]
                print(f"      {b['bucket']} n={b['n']:3d} realizedYES={b['realized_yes']*100:5.1f}% "
                      f"mid={b['avg_mid']*100:5.1f}% bias={b['bias']*100:+6.2f}pp "
                      f"CI[{bc[0]*100:+5.1f},{bc[1]*100:+5.1f}]")
            else:
                print(f"      {b['bucket']} n=0 (empty)")
    print(f"  PRIOR 06-18: new bias overall {PRIOR['newcity_bias_overall']*100:+.1f}pp (n={PRIOR['newcity_n_inband']}); "
          f"mature control {PRIOR['mature_bias_control']*100:+.1f}pp")

    # ================================================================
    # C. STRATEGY ROI (full live gate, g(mid), out-of-domain transfer)
    # ================================================================
    print("\n" + "=" * 96)
    print("C. STRATEGY ROI — full live gate + g(mid), one bet/event, fill @ ask, taker")
    print("=" * 96)
    new_traded = latest_per_event([c for c in new_c if passes_full_gate(c, g)])
    mat_traded = latest_per_event([c for c in mat_window if passes_full_gate(c, g)])
    out["C_roi"] = {
        "new_cities": summarize_roi(new_traded, label="new"),
        "mature_control": summarize_roi(mat_traded, label="mature"),
        "new_cities_slip2c": summarize_roi(new_traded, label="new_slip2c", slip=0.02),
        "note": "g(mid) is OUT-OF-DOMAIN for new cities (map fit on mature-9 only).",
    }
    for lbl, s in (("NEW", out["C_roi"]["new_cities"]), ("MATURE", out["C_roi"]["mature_control"])):
        ci = s["roi_day_ci"]
        print(f"  {lbl}: n={s['n']} ev={s['n_events']} d={s['n_days']} "
              f"win={s['win_rate']*100:.1f}% (wLB {s['win_wilson_lb']*100:.1f}%) "
              f"ROI={s['roi']*100:+.2f}% CI[{ci[0]*100:+.1f},{ci[1]*100:+.1f}] "
              f"net={s['net_per_contract_cents']:+.2f}c")
    s2 = out["C_roi"]["new_cities_slip2c"]
    print(f"  NEW +2c slip: ROI={s2['roi']*100:+.2f}% CI[{s2['roi_day_ci'][0]*100:+.1f},{s2['roi_day_ci'][1]*100:+.1f}]")
    print(f"  PRIOR 06-18 new ROI ~ {PRIOR['newcity_roi']*100:.0f}%")

    # Concentration / robustness on new-city traded set
    if new_traded:
        by_day = defaultdict(list)
        for c in new_traded:
            by_day[c.fdate].append(c)
        day_pnl = {d: sum(leg_pnl(c)[0] for c in r) for d, r in by_day.items()}
        best_day = max(day_pnl, key=day_pnl.get) if day_pnl else None
        s_drop = summarize_roi([c for c in new_traded if c.fdate != best_day], label="new_dropbest")
        # top-3 trades by pnl
        sorted_pnl = sorted((leg_pnl(c)[0] for c in new_traded), reverse=True)
        top3 = sum(sorted_pnl[:3])
        total_pnl = sum(sorted_pnl)
        out["C_roi"]["robustness"] = {
            "best_day": str(best_day),
            "drop_best_day": s_drop,
            "top3_pnl": top3,
            "total_pnl": total_pnl,
            "top3_share": (top3 / total_pnl) if total_pnl else None,
        }
        print(f"  ROBUST: drop best day ({best_day}) ROI={s_drop['roi']*100:+.2f}% "
              f"CI[{s_drop['roi_day_ci'][0]*100:+.1f},{s_drop['roi_day_ci'][1]*100:+.1f}]")

    # ================================================================
    # D. GATE STATUS
    # ================================================================
    print("\n" + "=" * 96)
    print("D. GATE STATUS vs the >=40 settled in-band floor + autumn pre-reg")
    print("=" * 96)
    n_inband_settled = len(new_inband)
    span_days = (nhi - nlo).days + 1 if nlo and nhi else 0
    inband_per_day = n_inband_settled / span_days if span_days else 0.0
    days_to_40 = ((40 - n_inband_settled) / inband_per_day) if inband_per_day > 0 else float("inf")
    # any bucket realized > mid?
    any_bucket_positive = any(
        (b["bias"] is not None and b["bias"] > 0) for b in out["B_bias"]["new_cities"]["buckets"]
    )
    out["D_gate"] = {
        "n_inband_settled": n_inband_settled,
        "floor_n": 40,
        "n_remaining_to_floor": max(0, 40 - n_inband_settled),
        "inband_arrivals_per_calendar_day": inband_per_day,
        "est_calendar_days_to_n40": days_to_40,
        "any_bucket_realized_gt_mid": any_bucket_positive,
        "pre_autumn": True,
        "autumn_reeval_window": "autumn 2026 (post heat-lock)",
    }
    print(f"  in-band settled n={n_inband_settled} (floor 40; remaining {max(0,40-n_inband_settled)})")
    print(f"  in-band arrival rate ~{inband_per_day:.2f}/day over {span_days} days -> "
          f"~{days_to_40:.0f} more calendar days to n>=40 at current summer rate")
    print(f"  any in-band bucket realized>mid: {any_bucket_positive}")
    print(f"  pre-autumn: True (re-eval gate window = autumn 2026 post heat-lock)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
