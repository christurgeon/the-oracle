"""Audit a SECOND favorite_bias at a DIFFERENT TIME HORIZON than the live lead-1.

Deliverables (every number 1-contract/leg, fill AT THE ASK, taker fee, day-clustered CI):
  A. Fresh lead-0 vs lead-1 vs lead-2 comparison on the CURRENT corpus.
  B. Incremental value of lead-0 over lead-1 (redundant-event vs incremental-event split).
  C. Ask-drift / execution cost (lead-0 decision -> next-day lead-1 slot, and -> settlement).
  D. Power analysis: how many more settled forecast-days to push the lower 95% day-clustered
     CI above the fee bar, at the OBSERVED point estimate. Sign-problem vs power-problem.

Hard rules (mirror the live favorite_bias gate exactly):
  * Fill at the ASK; edge = g(mid) - ask - fee_per_contract(ask, taker).
  * Lead = forecast_date - snapshot_calendar_day (NEVER time-to-expiry).
  * band [0.40, 0.75) left-incl right-excl; vol_fp >= 200; strike {greater,less};
    edge_floor 0.01; YES-only; one bet per (city,var,forecast_date) (latest slot per ticker).
  * FINAL settlements only (load_snapshot_data enforces).
  * CIs day-clustered (block-bootstrap over forecast_date, >=2000 resamples).

Run: PYTHONPATH=. uv run python scripts/research/fb_second_horizon.py
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
OUT = Path("data/research/2026-06-23-fb-second-horizon.json")

# --- live gate constants (verified against python/config.yaml 2026-06-23) ---
BAND_LO, BAND_HI = 0.40, 0.75
EDGE_FLOOR = 0.01
MIN_VOL = 200
N_BOOT = 4000


# ---------------------------------------------------------------------------
# P&L primitives
# ---------------------------------------------------------------------------
def leg_pnl(c: Cand, *, ask: float | None = None, slip: float = 0.0) -> tuple[float, float]:
    """(net_pnl, cost) buying 1 YES at ask(+slip), taker fee. payout 1 if YES else 0."""
    a = min(0.99, (c.ask if ask is None else ask) + slip)
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


def day_clustered_ci(rows: list[Cand], *, slip: float = 0.0, n_boot: int = N_BOOT):
    """Block-bootstrap ROI over forecast DATES. Returns (lo, hi, point, boot_rois)."""
    if not rows:
        return 0.0, 0.0, 0.0, np.array([])
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
    return float(lo), float(hi), float(point), rois


def summarize(rows: list[Cand], *, slip: float = 0.0, label: str = "") -> dict:
    n = len(rows)
    wins = sum(c.yes for c in rows)
    pnl = sum(leg_pnl(c, slip=slip)[0] for c in rows)
    wag = sum(leg_pnl(c, slip=slip)[1] for c in rows)
    lo, hi, point, _ = day_clustered_ci(rows, slip=slip)
    return {
        "label": label,
        "n": n,
        "n_events": len({(c.city, c.var, c.fdate) for c in rows}),
        "n_days": len({c.fdate for c in rows}),
        "win_rate": wins / n if n else 0.0,
        "win_wilson_lb": wilson_lb(wins, n),
        "roi": point,
        "roi_day_ci": [lo, hi],
        "pnl_per_contract": pnl,
        "net_per_contract_cents": (pnl / n * 100) if n else 0.0,
        "avg_ask": (sum(c.ask for c in rows) / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Live gate
# ---------------------------------------------------------------------------
def passes_gate(c: Cand, g, *, edge_floor: float = EDGE_FLOOR) -> bool:
    if c.stype not in ("greater", "less"):
        return False
    if not (BAND_LO <= c.mid < BAND_HI):
        return False
    if c.vol < MIN_VOL:
        return False
    p = g.calibrate(c.mid)
    fee = fee_per_contract(c.ask, is_maker=False)
    return (p - c.ask - fee) >= edge_floor


def latest_per_event(rows: list[Cand]) -> list[Cand]:
    """One bet per (city,var,forecast_date): keep the latest-slot candidate.
    Mirrors the live per-(city,var,fd) one-bet cap + 'last scan that sees it'."""
    best: dict[tuple, Cand] = {}
    for c in rows:
        k = (c.city, c.var, c.fdate)
        if k not in best or c.snap_ts > best[k].snap_ts:
            best[k] = c
    return sorted(best.values(), key=lambda c: (c.fdate, c.ticker))


def selected_for_lead(cands: list[Cand], g, lead: int, *, morning_only: bool = False) -> list[Cand]:
    """Apply the live gate to a given lead, one bet per event (latest slot).
    morning_only restricts lead-0 to slots {2,8} UTC (honest same-day morning decision,
    before the daytime high is realized)."""
    pool = [c for c in cands if c.lead == lead and c.stype in ("greater", "less")]
    if morning_only:
        pool = [c for c in pool if c.slot in (2, 8)]
    gated = [c for c in pool if passes_gate(c, g)]
    return latest_per_event(gated)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    g = IsotonicCalibrator(LIVE_MAP)
    assert g.is_loaded, getattr(g, "fallback_reason", "calibrator failed to load")

    # Load with no city restriction first to PROVE lead-2 has no venue across all cities.
    cands_all = load_candidates(leads=(0, 1, 2, 3), restrict_cities=None)
    lead_counts_allcities = dict(sorted(Counter(c.lead for c in cands_all).items()))

    # Corpus cities (the g-map's frozen cohort) for the actual gated comparison.
    cands = load_candidates(leads=(0, 1, 2, 3))
    lead_counts = dict(sorted(Counter(c.lead for c in cands).items()))

    out: dict = {
        "meta": {
            "corpus_dates": [str(min(c.fdate for c in cands)), str(max(c.fdate for c in cands))],
            "n_raw_candidates": len(cands),
            "lead_counts_corpus_cities": lead_counts,
            "lead_counts_all_cities": lead_counts_allcities,
            "edge_floor": EDGE_FLOOR,
            "band": [BAND_LO, BAND_HI],
            "min_vol": MIN_VOL,
        }
    }

    # ===================================================================
    # A. lead-0 vs lead-1 vs lead-2 (gated, one bet per event)
    # ===================================================================
    print("=" * 96)
    print("A. LEAD COMPARISON (live gate, one bet/event, fill @ ask, taker, day-clustered CI)")
    print("=" * 96)
    sel = {}
    sel["lead1"] = selected_for_lead(cands, g, 1)
    sel["lead0_all"] = selected_for_lead(cands, g, 0)
    sel["lead0_morning"] = selected_for_lead(cands, g, 0, morning_only=True)
    sel["lead2"] = selected_for_lead(cands, g, 2)

    out["A_lead_comparison"] = {}
    for name, rows in sel.items():
        s = summarize(rows, label=name)
        out["A_lead_comparison"][name] = s
        ci = s["roi_day_ci"]
        print(
            f"  {name:16s} n={s['n']:4d} ev={s['n_events']:4d} d={s['n_days']:3d} "
            f"win={s['win_rate'] * 100:5.1f}% (wLB {s['win_wilson_lb'] * 100:4.1f}%) "
            f"ROI={s['roi'] * 100:+6.2f}% CI[{ci[0] * 100:+5.1f},{ci[1] * 100:+5.1f}] "
            f"net={s['net_per_contract_cents']:+5.2f}c"
        )
    print(
        f"  lead-2 raw candidates across ALL cities/leads<=3: "
        f"{lead_counts_allcities.get(2, 0)} (NO-VENUE if 0)"
    )

    # ===================================================================
    # B. Incremental value of lead-0 over lead-1
    # ===================================================================
    print("\n" + "=" * 96)
    print("B. INCREMENTAL VALUE of lead-0 over lead-1 (event = city,var,forecast_date)")
    print("=" * 96)
    l1_events = {(c.city, c.var, c.fdate) for c in sel["lead1"]}
    for src_name, src_rows in (("lead0_all", sel["lead0_all"]), ("lead0_morning", sel["lead0_morning"])):
        redundant = [c for c in src_rows if (c.city, c.var, c.fdate) in l1_events]
        incremental = [c for c in src_rows if (c.city, c.var, c.fdate) not in l1_events]
        frac_redundant = len(redundant) / len(src_rows) if src_rows else 0.0
        s_inc = summarize(incremental, label=f"{src_name}_incremental")
        s_red = summarize(redundant, label=f"{src_name}_redundant")
        out[f"B_{src_name}"] = {
            "n_total": len(src_rows),
            "frac_redundant": frac_redundant,
            "incremental": s_inc,
            "redundant": s_red,
        }
        ci = s_inc["roi_day_ci"]
        print(f"  -- {src_name} --")
        print(
            f"     total buy-decisions: {len(src_rows)}  "
            f"redundant w/ lead-1: {len(redundant)} ({frac_redundant * 100:.0f}%)  "
            f"incremental: {len(incremental)}"
        )
        print(
            f"     INCREMENTAL-only: n={s_inc['n']} ev={s_inc['n_events']} d={s_inc['n_days']} "
            f"win={s_inc['win_rate'] * 100:.1f}% ROI={s_inc['roi'] * 100:+.2f}% "
            f"CI[{ci[0] * 100:+.1f},{ci[1] * 100:+.1f}] net={s_inc['net_per_contract_cents']:+.2f}c"
        )

    # ===================================================================
    # C. Ask-drift / execution cost for lead-0
    # ===================================================================
    print("\n" + "=" * 96)
    print("C. ASK-DRIFT for lead-0 candidates (decision slot -> later slots / settlement)")
    print("=" * 96)
    # For each lead-0 ticker, gather all its observed asks indexed by (lead, slot).
    by_ticker: dict[str, dict[tuple[int, int], Cand]] = defaultdict(dict)
    for c in cands:
        if c.stype in ("greater", "less"):
            by_ticker[c.ticker][(c.lead, c.slot)] = c

    # Drift from the lead-0 MORNING decision slot to the NEXT slot same day, and to last seen.
    # We define lead-0 morning decision = earliest of slots {2,8} that passes gate.
    drifts_next = []   # ask change to the immediately later same-day slot
    drifts_last = []   # ask change to the last slot we observe the ticker (closest to settle)
    settle_outcomes = []  # for the morning lead-0 decisions: did they win?
    for t, slotmap in by_ticker.items():
        # find earliest passing lead-0 morning observation
        l0_morning = sorted(
            (sl, c) for (ld, sl), c in slotmap.items() if ld == 0 and sl in (2, 8)
        )
        decision = None
        for sl, c in l0_morning:
            if passes_gate(c, g):
                decision = (sl, c)
                break
        if decision is None:
            continue
        d_slot, d_cand = decision
        # later same-day slots (lead 0): 8,14,20 after d_slot
        later_same = sorted(
            (sl, c) for (ld, sl), c in slotmap.items() if ld == 0 and sl > d_slot
        )
        if later_same:
            drifts_next.append((later_same[0][1].ask - d_cand.ask) * 100)
        # last observed ask for this ticker overall (any lead, latest snap_ts) = closest to close
        last_cand = max(slotmap.values(), key=lambda c: c.snap_ts)
        if last_cand.snap_ts > d_cand.snap_ts:
            drifts_last.append((last_cand.ask - d_cand.ask) * 100)
        settle_outcomes.append(d_cand.yes)

    def stat(arr):
        if not arr:
            return None
        a = np.array(arr)
        return {
            "n": len(a),
            "mean_c": float(a.mean()),
            "median_c": float(np.median(a)),
            "p25_c": float(np.percentile(a, 25)),
            "p75_c": float(np.percentile(a, 75)),
            "frac_positive": float((a > 0).mean()),
        }

    out["C_ask_drift"] = {
        "lead0_morning_decisions": len(settle_outcomes),
        "drift_to_next_same_day_slot": stat(drifts_next),
        "drift_to_last_observed_slot": stat(drifts_last),
        "morning_decision_win_rate": (
            sum(settle_outcomes) / len(settle_outcomes) if settle_outcomes else None
        ),
    }
    dn = out["C_ask_drift"]["drift_to_next_same_day_slot"]
    dl = out["C_ask_drift"]["drift_to_last_observed_slot"]
    print(f"  lead-0 morning (slot 2/8) decisions that pass gate: {len(settle_outcomes)}")
    if dn:
        print(
            f"  ask drift -> next same-day slot:  mean {dn['mean_c']:+.2f}c  "
            f"median {dn['median_c']:+.2f}c  frac>0 {dn['frac_positive'] * 100:.0f}%  (n={dn['n']})"
        )
    if dl:
        print(
            f"  ask drift -> last observed slot:  mean {dl['mean_c']:+.2f}c  "
            f"median {dl['median_c']:+.2f}c  frac>0 {dl['frac_positive'] * 100:.0f}%  (n={dl['n']})"
        )

    # ===================================================================
    # D. Power analysis
    # ===================================================================
    print("\n" + "=" * 96)
    print("D. POWER ANALYSIS — days needed to push lower 95% day-CI above 0 at observed point")
    print("=" * 96)
    out["D_power"] = {}
    # Settled FB-eligible events per day (gross arrival rate) for calendar-time estimate.
    # Use lead-1 selected events as the empirical FB-eligible arrival.
    days_l1 = len({c.fdate for c in sel["lead1"]})
    span_days = (max(c.fdate for c in cands) - min(c.fdate for c in cands)).days + 1
    eligible_per_day = len(sel["lead1"]) / days_l1 if days_l1 else 0.0

    for name in ("lead0_all", "lead0_morning"):
        rows = sel[name]
        if not rows:
            out["D_power"][name] = {"verdict": "no candidates"}
            continue
        # per-day P&L and wagered to build between-day distribution of ROI contributions.
        by_day = defaultdict(lambda: [0.0, 0.0])
        for c in rows:
            lp, cost = leg_pnl(c)
            by_day[c.fdate][0] += lp
            by_day[c.fdate][1] += cost
        day_pnl = np.array([v[0] for v in by_day.values()])
        day_wag = np.array([v[1] for v in by_day.values()])
        n_obs_days = len(day_pnl)
        point_roi = day_pnl.sum() / day_wag.sum() if day_wag.sum() else 0.0
        # mean per-day net pnl and its sd (the quantity whose mean we test vs 0).
        mean_day_pnl = day_pnl.mean()
        sd_day_pnl = day_pnl.std(ddof=1) if n_obs_days > 1 else 0.0
        # current lower CI from bootstrap
        lo, hi, _, _ = day_clustered_ci(rows)

        result = {
            "n_obs_days": int(n_obs_days),
            "point_roi": float(point_roi),
            "roi_day_ci": [float(lo), float(hi)],
            "mean_day_pnl_per_contract": float(mean_day_pnl),
            "sd_day_pnl": float(sd_day_pnl),
            "eligible_events_per_day_lead1": float(eligible_per_day),
        }

        if mean_day_pnl <= 0:
            result["verdict"] = "SIGN PROBLEM: mean per-day P&L <= 0; no n makes lower CI > 0"
            result["days_needed"] = None
        else:
            # Required days for a one-sided 95% mean test: mean - 1.645*sd/sqrt(N) > 0
            # => N > (1.645 * sd / mean)^2.  (Approximation; the bootstrap CI is two-sided
            # 95% so use 1.96 for the matching two-sided lower bound.)
            z = 1.96
            need = (z * sd_day_pnl / mean_day_pnl) ** 2 if mean_day_pnl > 0 else float("inf")
            result["days_needed_est"] = float(need)
            result["additional_days_needed"] = max(0.0, float(need - n_obs_days))
            # calendar time: how many incremental lead-0 decision-days at observed arrival
            l0_days_per_cal_day = n_obs_days / span_days if span_days else 0.0
            if l0_days_per_cal_day > 0:
                result["calendar_days_to_reach"] = float(
                    max(0.0, (need - n_obs_days)) / l0_days_per_cal_day
                )
            result["verdict"] = (
                "POINT EST POSITIVE — power-limited"
                if point_roi > 0
                else "marginal"
            )
        out["D_power"][name] = result
        print(f"  -- {name} --")
        print(
            f"     obs days={result['n_obs_days']}  point ROI={result['point_roi'] * 100:+.2f}%  "
            f"meanDayPnL/contract=${result['mean_day_pnl_per_contract']:+.3f}  "
            f"sd=${result['sd_day_pnl']:.3f}"
        )
        print(f"     current lower 95% day-CI: {result['roi_day_ci'][0] * 100:+.1f}%")
        print(f"     verdict: {result['verdict']}")
        if result.get("days_needed_est") is not None:
            print(
                f"     days needed (est): {result['days_needed_est']:.0f}  "
                f"additional: {result.get('additional_days_needed', 0):.0f}  "
                f"calendar days to reach: {result.get('calendar_days_to_reach', float('nan')):.0f}"
            )

    # ===================================================================
    # Robustness on any positive lead-0 horizon: drop best day + 2c slippage
    # ===================================================================
    print("\n" + "=" * 96)
    print("ROBUSTNESS — lead-0 (all & morning): drop best day, +2c slippage")
    print("=" * 96)
    out["robust"] = {}
    for name in ("lead0_all", "lead0_morning"):
        rows = sel[name]
        if not rows:
            continue
        by_day = defaultdict(list)
        for c in rows:
            by_day[c.fdate].append(c)
        day_pnl = {d: sum(leg_pnl(c)[0] for c in r) for d, r in by_day.items()}
        best_day = max(day_pnl, key=day_pnl.get)
        s_drop = summarize([c for c in rows if c.fdate != best_day], label=f"{name}_dropbest")
        s_slip = summarize(rows, slip=0.02, label=f"{name}_slip2c")
        out["robust"][name] = {"drop_best_day": s_drop, "slip2c": s_slip, "best_day": str(best_day)}
        print(
            f"  {name}: drop best day ({best_day}) ROI={s_drop['roi'] * 100:+.2f}% "
            f"CI[{s_drop['roi_day_ci'][0] * 100:+.1f},{s_drop['roi_day_ci'][1] * 100:+.1f}]  | "
            f"+2c slip ROI={s_slip['roi'] * 100:+.2f}% "
            f"CI[{s_slip['roi_day_ci'][0] * 100:+.1f},{s_slip['roi_day_ci'][1] * 100:+.1f}]"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
