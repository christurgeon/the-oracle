"""Walk-forward validation of the favorite_bias g(mid) map against the live artifact.

Answers one question: does a refit map price the *traded band* better than the map
currently serving live money, on days it was never fitted on?

Method
------
Expanding-window walk-forward over forecast_date. For each fold the map is fitted
on all days strictly before the fold and scored on the fold's days, so no fold is
ever evaluated on its own data. Scored on the live traded population only
(``band_low <= mid < band_high``, read from config so this can't drift from the
strategy) — a map that is better on deep longshots we never touch is not better.

Baselines
---------
- ``market``  : g(mid) = mid. The null the strategy must beat to justify existing.
- ``live``    : the on-disk artifact serving live money today.
- ``refit``   : the candidate produced by scripts/fit_favorite_bias_map.py.

Metrics
-------
- log loss and Brier (calibration; lower is better)
- realized ROI trading each map's own signal AT THE ASK with taker fees, gated by
  the live ``edge_floor``
- the same P&L Kelly-WEIGHTED at the live sizing config. Per-contract ROI is
  nearly blind to a map change, because both maps select almost the same rows;
  the economic difference is in how large a bet each map's asserted edge buys.
- day-clustered bootstrap CI on the ROI difference vs the live map

Run: PYTHONPATH=. uv run python scripts/research/fb_map_walkforward.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from python.analytics.fees import fee_per_contract
from python.analytics.sizing import fractional_kelly
from python.data.calibration import IsotonicCalibrator
from python.settings import Settings
from scripts.fit_favorite_bias_map import (
    apply_lift_guard,
    build_rows,
    cap_values,
    interp_curve,
    pav_isotonic,
)

LIVE_MAP = Path("data/calibration/favorite_bias_isotonic.json")
N_FOLDS = 4
BOOTSTRAP_B = 4000
# Nominal bankroll for the Kelly-weighted comparison. Only the RATIO between the
# maps matters, so the absolute level is arbitrary.
BANKROLL = 550.0


def _fit(rows: list[tuple[float, float, int, str]]) -> tuple[list[float], list[float]]:
    blocks = pav_isotonic([(mid, y, day) for mid, _ask, y, day in rows])
    bps, vals, _info = apply_lift_guard(blocks)
    out_bp, out_v, _idx = cap_values(bps, vals)
    return out_bp, out_v


def _log_loss(p: float, y: int) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _roi(scored: list[tuple[float, float, int, str]], edge_floor: float) -> tuple[float, int]:
    """ROI of trading every row whose edge clears the floor, filled at the ask."""
    pnl = wagered = 0.0
    n = 0
    for p, ask, y, _day in scored:
        fee = fee_per_contract(ask)
        if p - ask - fee < edge_floor:
            continue
        n += 1
        wagered += ask
        pnl += (1.0 - ask - fee) if y else (-ask - fee)
    return (pnl / wagered if wagered else 0.0), n


def _kelly_pnl(
    scored: list[tuple[float, float, int, str]], edge_floor: float, cfg
) -> tuple[float, float]:
    """(P&L, capital deployed) at the live Kelly config.

    This is where two maps actually differ: they pick nearly the same rows, but a
    larger asserted edge buys a larger bet.

    ``max_contracts_per_trade`` matters and must not be omitted: it binds far more
    often on the map asserting the bigger edge, so leaving it out flatters that
    map. Applying it also makes the result depend on the absolute bankroll, which
    is the honest trade-off — scale-invariance and live-config fidelity cannot
    both hold.

    Still NOT modelled (so this is a map comparison, not a P&L replay):
    ``min_volume_fp``, ``excluded_city_vars``, and the per-day event cap, none of
    which the loader carries.
    """
    pnl = deployed = 0.0
    for p, ask, y, _day in scored:
        fee = fee_per_contract(ask)
        if p - ask - fee < edge_floor:
            continue
        size = fractional_kelly(p, ask, cfg.kelly_fraction, cfg.max_pct, BANKROLL, fee).dollars
        contracts = min(int(size / ask), cfg.max_contracts_per_trade)
        if contracts < 1:
            continue
        deployed += contracts * ask
        pnl += contracts * ((1.0 - ask - fee) if y else (-ask - fee))
    return pnl, deployed


def _kelly_diff_ci(
    a: list[tuple[float, float, int, str]],
    b: list[tuple[float, float, int, str]],
    edge_floor: float,
    cfg,
) -> tuple[float, float]:
    """Day-clustered bootstrap CI on Kelly P&L(a) - P&L(b).

    The CI is on Kelly P&L rather than per-contract ROI deliberately. ROI is a 0/1
    selection gate, and two candidate maps almost always select the SAME rows, so
    the ROI difference is identically zero in every resample — a CI of
    [+0.0%, +0.0%] that looks like a finding but is an artifact of the metric.
    The maps differ in bet SIZE, which is what this measures.
    """
    days = sorted({r[3] for r in a} | {r[3] for r in b})
    by_a: dict[str, list] = {d: [] for d in days}
    by_b: dict[str, list] = {d: [] for d in days}
    for r in a:
        by_a[r[3]].append(r)
    for r in b:
        by_b[r[3]].append(r)
    rng = random.Random(20260829)
    diffs = []
    for _ in range(BOOTSTRAP_B):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        pa, _ = _kelly_pnl([r for d in pick for r in by_a[d]], edge_floor, cfg)
        pb, _ = _kelly_pnl([r for d in pick for r in by_b[d]], edge_floor, cfg)
        diffs.append(pa - pb)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def main() -> None:
    cfg = Settings.load().strategies.favorite_bias
    band_low, band_high, edge_floor = cfg.band_low, cfg.band_high, cfg.edge_floor
    print(f"live band [{band_low}, {band_high}), edge_floor {edge_floor}")

    rows = build_rows(Path("data/snapshots"))
    days = sorted({r[3] for r in rows})
    live = IsotonicCalibrator(LIVE_MAP)
    assert live.is_loaded, f"live map failed to load: {live.fallback_reason}"

    scored: dict[str, list[tuple[float, float, int, str]]] = {"market": [], "live": [], "refit": []}
    fold_size = len(days) // N_FOLDS
    for k in range(1, N_FOLDS):
        train_days = set(days[: k * fold_size])
        test_days = set(days[k * fold_size : (k + 1) * fold_size])
        train = [r for r in rows if r[3] in train_days]
        test = [r for r in rows if r[3] in test_days and band_low <= r[0] < band_high]
        if not train or not test:
            continue
        bps, vals = _fit(train)
        for mid, ask, y, day in test:
            scored["market"].append((mid, ask, y, day))
            scored["live"].append((live.calibrate(mid), ask, y, day))
            scored["refit"].append((interp_curve(bps, vals, mid), ask, y, day))

    # The live artifact was fitted on its own corpus up to ``fit_date``. Folds
    # testing on earlier days therefore score it IN-sample while scoring the
    # refit out-of-sample, which flatters the incumbent. Report both the full
    # walk-forward and the strictly-fair slice after that date.
    live_fit_date = (live.metadata or {}).get("metadata", {}).get("fit_date")
    print(f"live map fit_date: {live_fit_date}")

    def _report(label: str, keep) -> None:
        sel = {k: [r for r in v if keep(r[3])] for k, v in scored.items()}
        n = len(sel["market"])
        if n == 0:
            print(f"\n{label}: no rows")
            return
        nd = len({r[3] for r in sel["market"]})
        print(f"\n{label}: {n} in-band rows across {nd} forecast days\n")
        hdr = f"{'map':>8} {'logloss':>9} {'brier':>8} {'traded':>7} {'ROI':>9}"
        print(f"{hdr} {'mean_edge':>10} {'deployed':>10} {'kellyP&L':>9}")
        for name in ("market", "live", "refit"):
            s = sel[name]
            ll = sum(_log_loss(p, y) for p, _a, y, _d in s) / len(s)
            br = sum((p - y) ** 2 for p, _a, y, _d in s) / len(s)
            roi, ntr = _roi(s, edge_floor)
            edges = [p - a - fee_per_contract(a) for p, a, _y, _d in s]
            traded_edges = [e for e in edges if e >= edge_floor]
            me = sum(traded_edges) / len(traded_edges) if traded_edges else 0.0
            kp, dep = _kelly_pnl(s, edge_floor, cfg)
            row = f"{name:>8} {ll:9.4f} {br:8.4f} {ntr:7d} {roi:+8.1%}"
            print(f"{row} {me:+10.4f} {dep:10.2f} {kp:+9.2f}")
        same = [r[0] for r in sel["refit"]] and _roi(sel["refit"], edge_floor)[1] == _roi(
            sel["live"], edge_floor
        )[1]
        if same:
            print("  (both maps select the same rows, so per-contract ROI cannot differ)")
        lo, hi = _kelly_diff_ci(sel["refit"], sel["live"], edge_floor, cfg)
        print(f"  kellyP&L(refit) - kellyP&L(live), day-clustered 95% CI: [{lo:+.2f}, {hi:+.2f}]")

    _report("FULL walk-forward (live map is IN-sample on early folds)", lambda d: True)
    if live_fit_date:
        _report(
            f"FAIR slice: test days after the live map's fit_date {live_fit_date}",
            lambda d: d > live_fit_date,
        )
    print("\nCI excludes 0 -> difference is real; includes 0 -> not distinguishable.")
    print(f"Kelly figures are at bankroll {BANKROLL:.0f} with the live sizing config.")


if __name__ == "__main__":
    main()
