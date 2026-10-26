"""favorite_bias improvement research — measures every backtestable lever
against the committed snapshot corpus, fills AT THE ASK, fees taker, with
day-clustered (block-bootstrap over forecast dates) confidence intervals.

Guardrails honored:
  * Fills at the ask; fees via python.analytics.fees.order_fee (taker).
  * Outcomes via derive_outcome_yes (in _fb_data); FINAL settlements only.
  * Lead from snapshot calendar day, NEVER from time-to-expiry (the retracted
    look-ahead trap). All band/lead/window comparisons are paired or split-half
    walk-forward where a fit is involved.
  * greater/less only (the live scope). between/fade/arb are DEAD — not retested.
  * ROI is sizing-invariant: every P&L is reported at 1 contract/leg so the
    selection question is separated from the Kelly-sizing question.

Run: PYTHONPATH=. uv run python scripts/research/fb_improvements.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from python.analytics.fees import fee_per_contract, order_fee
from python.data.calibration import IsotonicCalibrator
from scripts.research._fb_data import (
    Cand,
    latest_slot_per_ticker,
    load_candidates,
)

RNG = np.random.default_rng(20260615)
LIVE_MAP = Path("data/calibration/favorite_bias_isotonic.json")
OUT = Path("data/research/2026-06-15-fb-improvements.json")

BAND_LO, BAND_HI = 0.40, 0.75
EDGE_FLOOR = 0.02
MIN_VOL = 200


# ---------------------------------------------------------------------------
# P&L primitives (1 contract/leg; fills at ask; taker fees)
# ---------------------------------------------------------------------------
def leg_pnl_taker(c: Cand, *, slip: float = 0.0) -> tuple[float, float]:
    """(net_pnl, cost) for buying 1 YES at the ask (+slip), taker fee."""
    ask = min(0.99, c.ask + slip)
    fee = order_fee(ask, 1, is_maker=False)
    cost = ask + fee
    payout = 1.0 if c.yes else 0.0
    return payout - cost, cost


def leg_pnl_maker(c: Cand, *, fill_price: float) -> tuple[float, float]:
    """(net_pnl, cost) for a MAKER fill at fill_price (rest inside spread),
    maker fee. Caller decides whether the order would have filled."""
    fee = order_fee(fill_price, 1, is_maker=True)
    cost = fill_price + fee
    payout = 1.0 if c.yes else 0.0
    return payout - cost, cost


def summarize(rows: list[Cand], *, slip: float = 0.0, maker_fill: float | None = None) -> dict:
    """Aggregate 1-contract P&L over rows. If maker_fill is a callable price fn,
    not used here; maker handled by dedicated functions."""
    pnl = wag = 0.0
    wins = 0
    legs = []
    for c in rows:
        lp, cost = leg_pnl_taker(c, slip=slip)
        pnl += lp
        wag += cost
        wins += c.yes
        legs.append(lp)
    n = len(rows)
    return {
        "n": n,
        "win": wins / n if n else 0.0,
        "pnl": pnl,
        "wagered": wag,
        "roi": pnl / wag if wag else 0.0,
        "legs": legs,
    }


def day_clustered_ci(
    rows: list[Cand], *, slip: float = 0.0, n_boot: int = 5000
) -> tuple[float, float]:
    """Block bootstrap over forecast DATES (intra-day outcomes correlate)."""
    if not rows:
        return (0.0, 0.0)
    by_day: dict = defaultdict(list)
    for c in rows:
        lp, cost = leg_pnl_taker(c, slip=slip)
        by_day[c.fdate].append((lp, cost))
    days = list(by_day.keys())
    rois = []
    for _ in range(n_boot):
        chosen = RNG.choice(len(days), size=len(days), replace=True)
        p = w = 0.0
        for di in chosen:
            for lp, cost in by_day[days[di]]:
                p += lp
                w += cost
        rois.append(p / w if w else 0.0)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return float(lo), float(hi)


def t_stat(rows: list[Cand], *, slip: float = 0.0) -> float:
    if len(rows) < 2:
        return 0.0
    legs = np.array([leg_pnl_taker(c, slip=slip)[0] for c in rows])
    sd = legs.std(ddof=1)
    return float(legs.mean() / (sd / math.sqrt(len(legs)))) if sd > 0 else 0.0


def n_days(rows: list[Cand]) -> int:
    return len({c.fdate for c in rows})


# ---------------------------------------------------------------------------
# g(mid) using the live artifact + walk-forward refit
# ---------------------------------------------------------------------------
def live_g():
    cal = IsotonicCalibrator(LIVE_MAP)
    assert cal.is_loaded, cal.fallback_reason
    return cal


def apply_live_rule(
    rows: list[Cand],
    g,
    *,
    band=(BAND_LO, BAND_HI),
    edge_floor=EDGE_FLOOR,
    min_vol=MIN_VOL,
) -> list[Cand]:
    """The live favorite_bias entry rule, taker: g(mid)-ask-fee >= edge_floor,
    inside band, vol gate, greater/less only."""
    out = []
    for c in rows:
        if c.stype not in ("greater", "less"):
            continue
        if not (band[0] <= c.mid < band[1]):
            continue
        if c.vol < min_vol:
            continue
        p = g.calibrate(c.mid)
        fee = fee_per_contract(c.ask, is_maker=False)
        if p - c.ask - fee >= edge_floor:
            out.append(c)
    return out


# ---- walk-forward isotonic refit (PAV) for OOS validation of g(mid) ----
def pav(xs, ys):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sy = [float(ys[i]) for i in order]
    blocks = []  # [val, weight, x]
    for x, y in zip(sx, sy):
        if blocks and blocks[-1][2] == x:
            b = blocks[-1]
            w = b[1] + 1
            blocks[-1] = [(b[0] * b[1] + y) / w, w, x]
        else:
            blocks.append([y, 1.0, x])
    merged = True
    while merged:
        merged = False
        out = []
        for b in blocks:
            if out and out[-1][0] > b[0]:
                w = out[-1][1] + b[1]
                out[-1] = [(out[-1][0] * out[-1][1] + b[0] * b[1]) / w, w, b[2]]
                merged = True
            else:
                out.append(list(b))
        blocks = out
    return [b[2] for b in blocks], [b[0] for b in blocks]


def interp(bps, vals, x):
    if not bps:
        return x
    if x <= bps[0]:
        return vals[0]
    if x >= bps[-1]:
        return vals[-1]
    for i in range(1, len(bps)):
        if x <= bps[i]:
            x0, x1 = bps[i - 1], bps[i]
            y0, y1 = vals[i - 1], vals[i]
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return vals[-1]


def walk_forward_refit(
    cands_gl: list[Cand],
    *,
    band=(BAND_LO, BAND_HI),
    edge_floor=EDGE_FLOOR,
    min_vol=MIN_VOL,
    burn_days=20,
    per_city=False,
) -> list[Cand]:
    """Expanding-window isotonic refit on (mid, yes), test strictly later days.
    Fully OOS (fit on past, apply to today). Optionally per-city maps."""
    dates = sorted({c.fdate for c in cands_gl})
    if len(dates) <= burn_days:
        return []
    start = dates[burn_days]
    sel = []
    for d in dates:
        if d < start:
            continue
        train = [c for c in cands_gl if c.fdate < d]
        if len(train) < 50:
            continue
        if per_city:
            maps = {}
            by_city = defaultdict(list)
            for c in train:
                by_city[c.city].append(c)
            for city, rows in by_city.items():
                if len(rows) >= 40:
                    maps[city] = pav([r.mid for r in rows], [r.yes for r in rows])
            gbps, gvals = pav([r.mid for r in train], [r.yes for r in train])
        else:
            gbps, gvals = pav([r.mid for r in train], [r.yes for r in train])
        for c in cands_gl:
            if c.fdate != d or c.vol < min_vol:
                continue
            if not (band[0] <= c.mid < band[1]):
                continue
            if per_city and c.city in maps:
                p = interp(maps[c.city][0], maps[c.city][1], c.mid)
            else:
                p = interp(gbps, gvals, c.mid)
            fee = fee_per_contract(c.ask, is_maker=False)
            if p - c.ask - fee >= edge_floor:
                sel.append(c)
    return sel


def line(label, res, *, slip=0.0, ci=None):
    s = (
        f"  {label:38s} n={res['n']:4d} d={n_days_from(res):>3} win={res['win'] * 100:5.1f}% "
        f"PnL/contract=${res['pnl']:7.2f} ROI={res['roi'] * 100:+6.2f}%"
    )
    if ci is not None:
        s += f"  dayCI=[{ci[0] * 100:+.1f},{ci[1] * 100:+.1f}]"
    return s


def n_days_from(res):
    return res.get("_d", "?")


def emit(label, rows, *, slip=0.0, want_ci=True):
    res = summarize(rows, slip=slip)
    res["_d"] = n_days(rows)
    ci = day_clustered_ci(rows, slip=slip) if (want_ci and rows) else None
    print(line(label, res, slip=slip, ci=ci))
    return {
        "label": label,
        "n": res["n"],
        "days": res["_d"],
        "win": res["win"],
        "pnl_per_contract": res["pnl"],
        "roi": res["roi"],
        "day_ci": list(ci) if ci else None,
        "t": t_stat(rows, slip=slip),
    }


def main():
    out: dict = {}
    cands = load_candidates()
    g = live_g()

    # Baseline universe: lead-1, latest slot per ticker, greater/less.
    l1 = latest_slot_per_ticker(cands, lead=1)
    l1_gl = [c for c in l1 if c.stype in ("greater", "less")]

    print("=" * 100)
    print("BASELINE — live rule (lead-1, latest slot, g/l, band[0.40,0.75), vol>=200, ef=0.02)")
    print("=" * 100)
    base_sel = apply_live_rule(l1_gl, g)
    out["baseline"] = emit("LIVE RULE (taker @ ask)", base_sel)
    out["baseline_slip1"] = emit("  + 1c slippage", base_sel, slip=0.01)
    out["baseline_slip2"] = emit("  + 2c slippage", base_sel, slip=0.02)

    # =====================================================================
    print("\n" + "=" * 100)
    print("LEVER 1 — FEES: band shift toward higher P (lower P(1-P) fee + more certain)")
    print("=" * 100)
    out["lever1_bands"] = []
    for band in [(0.40, 0.75), (0.45, 0.75), (0.50, 0.75), (0.55, 0.75),
                 (0.40, 0.65), (0.50, 0.70), (0.55, 0.70), (0.60, 0.75)]:
        sel = apply_live_rule(l1_gl, g, band=band)
        out["lever1_bands"].append(emit(f"band{band}", sel))

    # Fee accounting: avg fee paid per contract in current band vs shifted band.
    def avg_fee(rows):
        return sum(order_fee(c.ask, 1, is_maker=False) for c in rows) / len(rows) if rows else 0.0
    print(f"  [avg taker fee/contract]  current band: ${avg_fee(base_sel):.4f}")
    hi_band = apply_live_rule(l1_gl, g, band=(0.55, 0.75))
    print(f"  [avg taker fee/contract]  band(0.55,0.75): ${avg_fee(hi_band):.4f}")

    # =====================================================================
    print("\n" + "=" * 100)
    print("LEVER 1b — MAKER variant: rest inside spread at bid+1c, maker fee (4x cheaper).")
    print("  Modeled fill rule: order fills IFF realized... we cannot observe future")
    print("  trades, so we report TWO bounds: (a) optimistic = always fills at bid+1c,")
    print("  (b) the realized half-spread distribution to size the fill-vs-fee tradeoff.")
    print("=" * 100)
    # Spread distribution on the selected (traded) markets.
    spreads = [round((c.ask - c.bid) * 100) for c in base_sel]
    from collections import Counter
    print(f"  spread histogram (cents) on traded markets: {dict(sorted(Counter(spreads).items()))}")
    out["lever1b_spread_hist"] = dict(sorted(Counter(spreads).items()))
    # Maker optimistic: fill at bid+1c (or ask if spread<=1c -> taker anyway).
    mk_pnl = mk_wag = 0.0
    mk_wins = mk_n = 0
    for c in base_sel:
        spread_c = round((c.ask - c.bid) * 100)
        if spread_c <= 1:
            fill = c.ask
            fee = order_fee(fill, 1, is_maker=False)
        else:
            fill = c.bid + 0.01
            fee = order_fee(fill, 1, is_maker=True)
        cost = fill + fee
        payout = 1.0 if c.yes else 0.0
        mk_pnl += payout - cost
        mk_wag += cost
        mk_wins += c.yes
        mk_n += 1
    print(
        f"  MAKER-optimistic (100% fill @ bid+1c): n={mk_n} win={mk_wins / mk_n * 100:.1f}% "
        f"PnL/contract=${mk_pnl:.2f} ROI={mk_pnl / mk_wag * 100:+.2f}%  "
        f"(vs taker ROI {out['baseline']['roi'] * 100:+.2f}%)"
    )
    out["lever1b_maker_optimistic"] = {
        "n": mk_n,
        "win": mk_wins / mk_n if mk_n else 0,
        "pnl_per_contract": mk_pnl,
        "roi": mk_pnl / mk_wag if mk_wag else 0,
        "note": (
            "UPPER BOUND only — assumes 100% fill; real maker fill rate unknown "
            "(no future-trade data in snapshots)"
        ),
    }

    # =====================================================================
    print("\n" + "=" * 100)
    print("LEVER 3 — ENTRY TIME: paired slot14 vs slot20 on the SAME lead-1 tickers")
    print("=" * 100)
    # Pair tickers present at both slots; live rule applied with same g.
    slot_rows = {14: {}, 20: {}}
    for c in cands:
        if c.lead == 1 and c.stype in ("greater", "less") and c.slot in (14, 20):
            slot_rows[c.slot][c.ticker] = c
    common = set(slot_rows[14]) & set(slot_rows[20])
    paired14 = [slot_rows[14][t] for t in common]
    paired20 = [slot_rows[20][t] for t in common]
    sel14 = apply_live_rule(paired14, g)
    sel20 = apply_live_rule(paired20, g)
    print(f"  common lead-1 g/l tickers at both slots: {len(common)}")
    out["lever3_slot14"] = emit("slot 14 UTC (early)", sel14)
    out["lever3_slot20"] = emit("slot 20 UTC (late=live default)", sel20)
    # Asks on common in-band tickers: which slot offers a better ask?
    inb = [t for t in common if BAND_LO <= slot_rows[20][t].mid < BAND_HI]
    da = [slot_rows[20][t].ask - slot_rows[14][t].ask for t in inb]
    if da:
        print(
            f"  mean ask(slot20)-ask(slot14) over in-band common tickers: "
            f"{np.mean(da) * 100:+.2f}c (n={len(da)})  -> negative means slot20 ask cheaper"
        )
        out["lever3_mean_ask_diff_c"] = float(np.mean(da) * 100)

    # =====================================================================
    print("\n" + "=" * 100)
    print("LEVER 3b — LEAD selection: lead-0 vs lead-1 (verify config's lead-0=no-edge claim)")
    print("=" * 100)
    l0 = latest_slot_per_ticker(cands, lead=0)
    l0_gl = [c for c in l0 if c.stype in ("greater", "less")]
    sel_l0 = apply_live_rule(l0_gl, g)
    out["lever3b_lead0"] = emit("lead-0 (same-day) live rule", sel_l0)  # noqa: E501
    out["lever3b_lead1"] = emit("lead-1 (live)", base_sel)
    print("  NOTE: lead-2 has ZERO candidates in corpus (markets not open 2d before) "
          "— config 'no liquidity' CONFIRMED.")
    out["lever3b_lead2_n"] = len([c for c in cands if c.lead == 2])

    # =====================================================================
    print("\n" + "=" * 100)
    print("LEVER 4 — MODEL/SELECTION")
    print("=" * 100)
    print("  (a) Walk-forward isotonic REFIT (OOS, expanding window) vs live frozen map:")
    wf_pooled = walk_forward_refit(l1_gl)
    out["lever4_wf_pooled"] = emit("WF refit (pooled), live band/ef", wf_pooled)
    wf_city = walk_forward_refit(l1_gl, per_city=True)
    out["lever4_wf_percity"] = emit("WF refit (per-city maps)", wf_city)

    print("\n  (b) edge_floor sweep (live frozen map):")
    out["lever4_edge_floor"] = []
    for ef in (0.00, 0.01, 0.02, 0.03, 0.04, 0.06):
        sel = apply_live_rule(l1_gl, g, edge_floor=ef)
        out["lever4_edge_floor"].append(emit(f"edge_floor={ef:.2f}", sel))

    print("\n  (c) per-CITY realized P&L (live rule, frozen map):")
    out["lever4_per_city"] = []
    by_city = defaultdict(list)
    for c in base_sel:
        by_city[c.city].append(c)
    for city in sorted(by_city):
        out["lever4_per_city"].append(emit(f"city={city}", by_city[city], want_ci=False))

    print("\n  (d) per-VAR and per-STRIKE realized P&L:")
    out["lever4_per_var"] = []
    for key, attr in (("var", "var"), ("stype", "stype")):
        grp = defaultdict(list)
        for c in base_sel:
            grp[getattr(c, attr)].append(c)
        for gk in sorted(grp):
            out["lever4_per_var"].append(emit(f"{key}={gk}", grp[gk], want_ci=False))

    print("\n  (e) volume gate sweep:")
    out["lever4_vol"] = []
    for mv in (0, 100, 200, 400, 800):
        sel = apply_live_rule(l1_gl, g, min_vol=mv)
        out["lever4_vol"].append(emit(f"min_vol={mv}", sel))

    # =====================================================================
    print("\n" + "=" * 100)
    print("ROBUSTNESS — split-half walk-forward + remove best day")
    print("=" * 100)
    dts = sorted({c.fdate for c in base_sel})
    cut = dts[len(dts) // 2]
    out["robust_firsthalf"] = emit(
        f"first half (<{cut})", [c for c in base_sel if c.fdate < cut], want_ci=False
    )
    out["robust_secondhalf"] = emit(
        f"second half (>={cut})", [c for c in base_sel if c.fdate >= cut], want_ci=False
    )
    # remove best day
    by_day = defaultdict(list)
    for c in base_sel:
        by_day[c.fdate].append(c)
    day_pnl = {d: sum(leg_pnl_taker(c)[0] for c in rows) for d, rows in by_day.items()}
    best_day = max(day_pnl, key=day_pnl.get) if day_pnl else None
    if best_day:
        print(f"  best day = {best_day} (PnL/contract ${day_pnl[best_day]:.2f})")
        out["robust_drop_best_day"] = emit(
            "drop best day", [c for c in base_sel if c.fdate != best_day], want_ci=False
        )

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
