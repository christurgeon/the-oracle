"""favorite_bias city / (city,var) exclusion analysis — should the live pilot
drop a persistently-negative cell?

Question (2026-06-15): favorite_bias is the only live order-placing strategy and
has NO city allowlist or exclusion (it trades every cell in its broad SERIES
tuple, including cells weather_edge never trades — notably BOS and DEN). The
2026-06-15 scale review found a weak-city cohort dragging an otherwise-real
+11.4% live edge. This script tests, on BOTH the live cohort and the backtest
corpus, which cells are *reliably* negative vs small-n noise, using a
PRE-REGISTERED rule, and quantifies the lift from excluding them.

Pre-registered exclusion rule (locked BEFORE per-cell inspection):
    EXCLUDE a cell IFF  n >= 15  AND  day-clustered ROI 95% CI upper bound < 0.
Both legs must hold; this is deliberately conservative (point-estimate-negative
is NOT enough — the whole CI must sit below zero).

Hard guardrails (project burned-history):
  * Realized P&L NET OF FEES, fills AT THE ASK. Never aggregate Brier.
  * Day-clustered CIs: block bootstrap over forecast DATES (intra-day outcomes
    correlate). percentile method, B=20000, fixed seed.
  * LIVE cohort: data/shadow_settlements.full.json filtered to
    strategy=favorite_bias AND settled (strategy attribution already resolved by
    fetch_shadow_settlements.py). pnl is Kalshi-realized, already net of fees;
    ROI on cost-basis = ``dollars``, the realised fill VWAP (NOT exec_price,
    which is the signal-time ask — see ``_cost_basis``).
  * BACKTEST corpus: scripts.research._fb_data (FINAL settlements only, lead from
    snapshot calendar day NEVER time-to-expiry) + the live entry rule; 1
    contract/leg, taker order_fee, fills at ask. ROI on ask+fee cost.
  * Live and backtest are NOT independent where their forecast-dates overlap —
    the script reports the overlap for every candidate cell so corroboration is
    not overclaimed.

Run: PYTHONPATH=. uv run python scripts/research/fb_city_exclusion.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.research._fb_data import latest_slot_per_ticker, load_candidates
from scripts.research.fb_improvements import apply_live_rule, leg_pnl_taker, live_g

RNG = np.random.default_rng(20260615)
LIVE = Path("data/shadow_settlements.full.json")
OUT = Path("data/research/2026-06-15-fb-city-exclusion.json")

N_MIN = 15  # pre-registered eligibility floor (per-cell)
B = 20000

# Cells weather_edge actually trades (the 7 tradeable cells, CLAUDE.md Current
# State). Used only to describe the "allowlist of weather-tradeable cities"
# option — NOT a recommendation input.
WEATHER_TRADEABLE_CITIES = frozenset({"NYC", "AUS", "SEA", "MIA", "LAX"})
# The scale review's (mislabelled) weak cohort.
SCALE_REVIEW_WEAK = frozenset({"PHIL", "BOS", "DEN"})


def _cost_basis(row: dict) -> float:
    """Dollars actually paid for a fill row.

    ``dollars`` is the realised cost basis and is preferred. ``exec_price`` is
    the signal-time ASK submitted as a limit, not the execution: on this corpus
    it disagrees with ``dollars`` on 39 of 187 favorite_bias settled rows and
    overstates total wagered by about 1.1%. The same defect was removed from
    python/analytics/attribution.py.
    """
    dollars = row.get("dollars")
    if dollars is not None:
        return float(dollars)
    contracts = int(row.get("contracts") or 0)
    for key in ("fill_price", "exec_price"):
        price = row.get(key)
        if price is not None:
            return float(price) * contracts
    return 0.0


# ---------------------------------------------------------------------------
# Trade record: a uniform (city, var, stype, fdate, pnl, wagered, won) tuple so
# the same aggregation works for live and backtest.
# ---------------------------------------------------------------------------
def load_live() -> list[dict]:
    rows = json.loads(LIVE.read_text())
    out = []
    for r in rows:
        if (r.get("strategy") or "") != "favorite_bias" or not r.get("settled"):
            continue
        md = r.get("metadata") or {}
        wag = _cost_basis(r)
        out.append(
            {
                "city": md.get("city"),
                "var": md.get("var") or md.get("market_type"),
                "stype": r.get("strike_type"),
                "fdate": md.get("forecast_date"),
                "pnl": float(r["pnl"]),  # net of fees (Kalshi-realized)
                "wag": wag,
                "won": bool(r["won"]),
            }
        )
    return out


def load_backtest() -> list[dict]:
    cands = load_candidates()
    g = live_g()
    l1_gl = [c for c in latest_slot_per_ticker(cands, lead=1) if c.stype in ("greater", "less")]
    sel = apply_live_rule(l1_gl, g)
    out = []
    for c in sel:
        pnl, cost = leg_pnl_taker(c)  # 1 contract, at ask, taker fee
        out.append(
            {
                "city": c.city,
                "var": c.var,
                "stype": c.stype,
                "fdate": c.fdate.isoformat(),
                "pnl": pnl,
                "wag": cost,
                "won": bool(c.yes),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation + day-clustered bootstrap
# ---------------------------------------------------------------------------
def agg(rows: list[dict]) -> dict:
    n = len(rows)
    pnl = sum(t["pnl"] for t in rows)
    wag = sum(t["wag"] for t in rows)
    w = sum(1 for t in rows if t["won"])
    return {
        "n": n,
        "wins": w,
        "win": (w / n if n else 0.0),
        "pnl": pnl,
        "wag": wag,
        "roi": (100 * pnl / wag if wag > 0 else 0.0),
        "days": len({t["fdate"] for t in rows}),
    }


def day_ci(rows: list[dict]) -> tuple[float | None, float | None]:
    """Block bootstrap over forecast dates; return (lo, hi) ROI% percentiles."""
    by = defaultdict(lambda: [0.0, 0.0])
    for t in rows:
        by[t["fdate"]][0] += t["pnl"]
        by[t["fdate"]][1] += t["wag"]
    days = list(by)
    K = len(days)
    if K < 2:
        return (None, None)
    rois = []
    idx = np.arange(K)
    for _ in range(B):
        chosen = RNG.choice(idx, size=K, replace=True)
        sp = sum(by[days[i]][0] for i in chosen)
        sw = sum(by[days[i]][1] for i in chosen)
        if sw > 0:
            rois.append(100 * sp / sw)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return (float(lo), float(hi))


def cell_row(label: str, rows: list[dict]) -> dict:
    a = agg(rows)
    lo, hi = day_ci(rows)
    eligible = a["n"] >= N_MIN
    reliably_neg = eligible and (hi is not None and hi < 0)
    return {
        "label": label,
        **a,
        "day_ci_lo": lo,
        "day_ci_hi": hi,
        "eligible_n>=15": eligible,
        "reliably_negative": reliably_neg,
    }


def group(rows: list[dict], keyfn) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for t in rows:
        g[keyfn(t)].append(t)
    return g


def fmt(row: dict) -> str:
    ci = (
        f"[{row['day_ci_lo']:+.0f},{row['day_ci_hi']:+.0f}]"
        if row["day_ci_lo"] is not None
        else "[--]"
    )
    tag = ""
    if row["reliably_negative"]:
        tag = "  *** RELIABLY NEGATIVE (exclude) ***"
    elif row["n"] >= N_MIN and row["roi"] < 0:
        tag = "  (n>=15 but CI includes 0 -> WATCH)"
    elif row["roi"] < 0:
        tag = "  (negative but n<15 -> noise/thin)"
    return (
        f"  {row['label']:11s} n={row['n']:3d} W={row['wins']:3d} "
        f"win={100 * row['win']:3.0f}% net=${row['pnl']:+7.2f} ROI={row['roi']:+7.1f}% "
        f"dayCI={ci:14s} K={row['days']:2d}{tag}"
    )


def section(title: str, rows: list[dict], keyfn, sort_by_roi=True) -> list[dict]:
    print(f"\n{title}")
    out = [cell_row(k, v) for k, v in group(rows, keyfn).items()]
    out.sort(key=lambda r: r["roi"] if sort_by_roi else r["label"])
    for r in out:
        print(fmt(r))
    return out


def exclude(rows: list[dict], pred) -> list[dict]:
    return [t for t in rows if not pred(t)]


def scenario(name: str, rows: list[dict], pred) -> dict:
    kept = exclude(rows, pred)
    dropped = [t for t in rows if pred(t)]
    a_all, a_kept, a_drop = agg(rows), agg(kept), agg(dropped)
    lo, hi = day_ci(kept)
    print(
        f"  {name:28s} drop n={a_drop['n']:2d} (${a_drop['pnl']:+6.2f}) "
        f"-> kept n={a_kept['n']:3d} ROI {a_all['roi']:+.1f}% -> {a_kept['roi']:+.1f}% "
        f"(dayCI [{lo:+.0f},{hi:+.0f}])  vol cost {a_drop['n']}/{a_all['n']}="
        f"{100 * a_drop['n'] / a_all['n']:.0f}%"
    )
    return {
        "scenario": name,
        "roi_before": a_all["roi"],
        "roi_after": a_kept["roi"],
        "n_before": a_all["n"],
        "n_after": a_kept["n"],
        "n_dropped": a_drop["n"],
        "pnl_dropped": a_drop["pnl"],
        "day_ci_after": [lo, hi],
        "vol_cost_pct": 100 * a_drop["n"] / a_all["n"] if a_all["n"] else 0,
    }


def gate_leg(rows: list[dict], weak: frozenset[str], pred_exclude=None) -> dict:
    """The scale-gate city-split leg: weak-cohort net P&L >= break-even (>=0)."""
    r = exclude(rows, pred_exclude) if pred_exclude else rows
    weak_rows = [t for t in r if t["city"] in weak]
    strong_rows = [t for t in r if t["city"] not in weak]
    aw, as_ = agg(weak_rows), agg(strong_rows)
    return {
        "weak_cities": sorted(weak),
        "weak_n": aw["n"],
        "weak_net": aw["pnl"],
        "weak_roi": aw["roi"],
        "weak_passes_BE": aw["pnl"] >= 0,
        "strong_n": as_["n"],
        "strong_net": as_["pnl"],
        "strong_roi": as_["roi"],
    }


def overlap_report(live: list[dict], bt: list[dict], pred, name: str) -> dict:
    ld = {t["fdate"] for t in live if pred(t)}
    bd = {t["fdate"] for t in bt if pred(t)}
    print(
        f"  {name}: live dates={len(ld)} backtest dates={len(bd)} "
        f"overlap={len(ld & bd)} live-only={len(ld - bd)} bt-only={len(bd - ld)}"
    )
    return {
        "cell": name,
        "live_dates": sorted(ld),
        "backtest_dates": sorted(bd),
        "overlap": sorted(ld & bd),
        "live_only": sorted(ld - bd),
        "backtest_only": sorted(bd - ld),
    }


def main():
    live = load_live()
    bt = load_backtest()
    out: dict = {"pre_registered_rule": "n>=15 AND day-clustered ROI 95% CI upper bound < 0"}

    for name, rows in (("LIVE", live), ("BACKTEST", bt)):
        a = agg(rows)
        lo, hi = day_ci(rows)
        print("=" * 100)
        print(
            f"{name} HEADLINE: n={a['n']} W={a['wins']} win={100 * a['win']:.1f}% "
            f"net=${a['pnl']:+.2f} wag=${a['wag']:.2f} ROI={a['roi']:+.1f}% "
            f"dayCI=[{lo:+.1f},{hi:+.1f}] over K={a['days']} forecast-date clusters"
        )
        out[f"{name.lower()}_headline"] = {**a, "day_ci": [lo, hi]}
        out[f"{name.lower()}_per_city"] = section(
            f"{name} per-CITY:", rows, lambda t: t["city"]
        )
        out[f"{name.lower()}_per_cell"] = section(
            f"{name} per-(CITY,VAR):", rows, lambda t: f"{t['city']}/{t['var']}"
        )
        out[f"{name.lower()}_per_strike"] = section(
            f"{name} per-STRIKE:", rows, lambda t: str(t["stype"]), sort_by_roi=False
        )
        out[f"{name.lower()}_per_var"] = section(
            f"{name} per-VAR:", rows, lambda t: str(t["var"]), sort_by_roi=False
        )

    # ---- Independence check for the candidate cells ----
    print("\n" + "=" * 100)
    print("INDEPENDENCE — do live & backtest measure the SAME forecast-dates? (overlap = NOT corroboration)")
    print("=" * 100)
    out["overlap"] = {
        "BOS/high": overlap_report(
            live, bt, lambda t: t["city"] == "BOS" and t["var"] == "high", "BOS/high"
        ),
        "BOS": overlap_report(live, bt, lambda t: t["city"] == "BOS", "BOS(all)"),
        "DEN": overlap_report(live, bt, lambda t: t["city"] == "DEN", "DEN(all)"),
        "CHI/low": overlap_report(
            live, bt, lambda t: t["city"] == "CHI" and t["var"] == "low", "CHI/low"
        ),
    }

    # ---- Lift scenarios (live cohort is the operative one for de-risking) ----
    print("\n" + "=" * 100)
    print("LIFT — exclusion scenarios on the LIVE cohort (aggregate ROI)")
    print("=" * 100)
    out["lift_live"] = [
        scenario("excl BOS/high", live, lambda t: t["city"] == "BOS" and t["var"] == "high"),
        scenario("excl BOS (city)", live, lambda t: t["city"] == "BOS"),
        scenario("excl {PHIL,BOS,DEN}", live, lambda t: t["city"] in SCALE_REVIEW_WEAK),
        scenario(
            "allowlist weather-cities",
            live,
            lambda t: t["city"] not in WEATHER_TRADEABLE_CITIES,
        ),
    ]
    print("\nLIFT — same scenarios on the BACKTEST corpus (sanity / power check)")
    out["lift_backtest"] = [
        scenario("excl BOS/high", bt, lambda t: t["city"] == "BOS" and t["var"] == "high"),
        scenario("excl BOS (city)", bt, lambda t: t["city"] == "BOS"),
        scenario("excl {PHIL,BOS,DEN}", bt, lambda t: t["city"] in SCALE_REVIEW_WEAK),
        scenario(
            "allowlist weather-cities",
            bt,
            lambda t: t["city"] not in WEATHER_TRADEABLE_CITIES,
        ),
    ]

    # ---- Scale-gate city-split leg ----
    print("\n" + "=" * 100)
    print("SCALE-GATE CITY-SPLIT LEG — weak-cohort net P&L (>=0 to pass) before/after BOS/high excl")
    print("=" * 100)
    bh = lambda t: t["city"] == "BOS" and t["var"] == "high"  # noqa: E731
    out["gate_leg"] = {
        "live_before": gate_leg(live, SCALE_REVIEW_WEAK),
        "live_after_excl_BOShigh": gate_leg(live, SCALE_REVIEW_WEAK, pred_exclude=bh),
    }
    for k, v in out["gate_leg"].items():
        print(
            f"  {k:26s} weak{v['weak_cities']} n={v['weak_n']} net=${v['weak_net']:+.2f} "
            f"({v['weak_roi']:+.1f}%) {'PASS' if v['weak_passes_BE'] else 'FAIL'} | "
            f"strong n={v['strong_n']} net=${v['strong_net']:+.2f} ({v['strong_roi']:+.1f}%)"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
