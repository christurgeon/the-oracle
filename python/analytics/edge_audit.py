"""Hermetic edge-existence audit (Phase 0). Pure functions, stdlib + numpy;
also imports python.analytics.fees (itself pure-math, no I/O).

No file/network I/O lives here — the CLI driver (an unpublished research script)
owns all loading. Mirrors the hermetic style of emos_fit.py / boundary_distance.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from python.analytics.fees import fee_per_contract as _fee_per_contract


def outcome_yes(*, side: str, won: bool) -> int:
    """Realized YES outcome, calibrator-independent (fixed by settled temperature)."""
    side_u = side.upper()
    if side_u == "YES":
        return 1 if won else 0
    if side_u == "NO":
        return 0 if won else 1
    raise ValueError(f"unexpected side: {side!r}")


@dataclass(frozen=True)
class SettledTrade:
    ticker: str
    trading_day: str          # UTC YYYY-MM-DD from timestamp
    model_p_yes: float        # model's calibrated P(YES) (row.model_probability)
    market_p_yes: float       # market YES price at signal (row.mid_at_signal or market_price)
    claimed_edge: float
    outcome_yes: int
    pnl: float
    wagered: float            # row.dollars (recorded cost basis)
    side: str
    strike_type: str
    strategy: str


def trades_from_rows(rows: list[dict[str, Any]]) -> list[SettledTrade]:
    out: list[SettledTrade] = []
    for r in rows:
        if not r.get("settled"):
            continue
        if r.get("pnl") is None or r.get("dollars") is None:
            continue
        ts = str(r["timestamp"])
        market_p = r.get("mid_at_signal")
        if market_p is None:
            market_p = r.get("market_price")
        if market_p is None:
            continue
        out.append(
            SettledTrade(
                ticker=str(r["market_ticker"]),
                trading_day=ts[:10],
                model_p_yes=float(r["model_probability"]),
                market_p_yes=float(market_p),
                claimed_edge=float(
            (r.get("metadata") or {}).get("effective_edge") or r.get("edge", 0.0)
        ),
                outcome_yes=outcome_yes(side=str(r["side"]), won=bool(r["won"])),
                pnl=float(r["pnl"]),
                wagered=float(r["dollars"]),
                side=str(r["side"]),
                strike_type=str(r.get("strike_type", "")),
                strategy=str(r.get("strategy", "")),
            )
        )
    return out


def roi(pnl: list[float], wagered: list[float]) -> float:
    w = float(sum(wagered))
    return float(sum(pnl)) / w if w > 0 else 0.0


def day_clustered_bootstrap_ci(
    trading_days: list[str], pnl: list[float], wagered: list[float],
    *, b: int = 2000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float]:
    """95% percentile CI on ROI, resampling whole trading-days with replacement.

    Corrects the within-day correlation that per-trade resampling ignores.
    """
    rng = np.random.default_rng(seed)
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(trading_days):
        by_day.setdefault(d, []).append(i)
    day_keys = list(by_day.keys())
    pnl_a = np.asarray(pnl, dtype=float)
    wag_a = np.asarray(wagered, dtype=float)
    if not day_keys:
        return (0.0, 0.0)
    samples = np.empty(b, dtype=float)
    n_days = len(day_keys)
    for k in range(b):
        chosen = rng.integers(0, n_days, size=n_days)
        idx: list[int] = []
        for c in chosen:
            idx.extend(by_day[day_keys[c]])
        w = wag_a[idx].sum()
        samples[k] = pnl_a[idx].sum() / w if w > 0 else 0.0
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)


def stationary_block_bootstrap_ci(
    series: list[float], *, mean_block: int = 7, b: int = 2000,
    seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float]:
    """Politis-Romano stationary-block bootstrap CI on the mean of an ordered series.

    Geometric block lengths (mean `mean_block`) preserve day-to-day serial
    correlation that day-clustering alone misses.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    p = 1.0 / mean_block
    means = np.empty(b, dtype=float)
    for k in range(b):
        out: list[float] = []
        while len(out) < n:
            start = rng.integers(0, n)
            j = start
            out.append(x[j])
            while rng.random() > p and len(out) < n + mean_block:
                j = (j + 1) % n
                out.append(x[j])
        means[k] = float(np.mean(out[:n]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def edge_monotonicity(
    trades: list[SettledTrade], *, n_bins: int = 4, seed: int = 42
) -> dict[str, Any]:
    """T1: realized ROI by claimed-edge quantile bucket, with CIs.

    PASS = ROI monotone increasing across buckets AND top-bucket day-clustered
    CI lower bound > 0 AND top-bucket stationary-block CI lower bound > 0.
    """
    we = [t for t in trades if t.strategy == "weather_edge"]
    if len(we) < n_bins:
        return {"passes": False, "reason": "insufficient_n", "n": len(we), "buckets": []}
    we_sorted = sorted(we, key=lambda t: t.claimed_edge)
    edges = np.array_split(np.array(we_sorted, dtype=object), n_bins)
    buckets = []
    roi_points = []
    for bi, arr in enumerate(edges):
        bt = list(arr)
        r = roi([t.pnl for t in bt], [t.wagered for t in bt])
        dc_lo, dc_hi = day_clustered_bootstrap_ci(
            [t.trading_day for t in bt], [t.pnl for t in bt],
            [t.wagered for t in bt], seed=seed,
        )
        # Stationary-block CI on per-trading-day ROI series for this bucket.
        by_day: dict[str, tuple[float, float]] = {}
        for t in bt:
            p, w = by_day.get(t.trading_day, (0.0, 0.0))
            by_day[t.trading_day] = (p + t.pnl, w + t.wagered)
        day_roi_series = [
            sp / sw if sw > 0 else 0.0 for sp, sw in by_day.values()
        ]
        sb_lo, sb_hi = stationary_block_bootstrap_ci(day_roi_series, mean_block=7, seed=seed)
        roi_points.append(r)
        buckets.append({
            "bucket": bi, "n": len(bt),
            "edge_lo": min(t.claimed_edge for t in bt),
            "edge_hi": max(t.claimed_edge for t in bt),
            "roi": r,
            "ci_lo": min(dc_lo, sb_lo),        # more conservative of the two (backward-compat)
            "ci_hi": dc_hi,
            "ci_lo_day_clustered": dc_lo,
            "ci_lo_stationary_block": sb_lo,
        })
    monotone = all(roi_points[i] <= roi_points[i + 1] for i in range(len(roi_points) - 1))
    top = buckets[-1]
    top_dc_lo = top["ci_lo_day_clustered"]
    top_sb_lo = top["ci_lo_stationary_block"]
    return {
        "passes": bool(monotone and top_dc_lo > 0.0 and top_sb_lo > 0.0),
        "monotone_increasing": bool(monotone),
        "top_bucket_ci_lo": top_dc_lo,
        "top_bucket_ci_lo_stationary_block": top_sb_lo,
        "buckets": buckets,
        "n": len(we),
    }


_EPS = 1e-6


def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def _logscore(p: float, y: int) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def model_vs_market_scores(trades: list[SettledTrade], *, seed: int = 42) -> dict[str, Any]:
    """T2: is model_p_yes better calibrated than the market price on settled outcomes?

    A forecasting edge exists only if the model's Brier AND LogScore are lower
    than the market's, with the paired per-day score-difference CI excluding 0.
    """
    we = [t for t in trades if t.strategy == "weather_edge"]
    if not we:
        return {"model_beats_market": False, "reason": "insufficient_n", "n": 0}
    mb = float(np.mean([_brier(t.model_p_yes, t.outcome_yes) for t in we]))
    kb = float(np.mean([_brier(t.market_p_yes, t.outcome_yes) for t in we]))
    ml = float(np.mean([_logscore(t.model_p_yes, t.outcome_yes) for t in we]))
    kl = float(np.mean([_logscore(t.market_p_yes, t.outcome_yes) for t in we]))

    # Paired daily mean Brier difference (model - market); negative => model better.
    by_day_brier: dict[str, list[float]] = {}
    by_day_ls: dict[str, list[float]] = {}
    for t in we:
        db = _brier(t.model_p_yes, t.outcome_yes) - _brier(t.market_p_yes, t.outcome_yes)
        dl = _logscore(t.model_p_yes, t.outcome_yes) - _logscore(t.market_p_yes, t.outcome_yes)
        by_day_brier.setdefault(t.trading_day, []).append(db)
        by_day_ls.setdefault(t.trading_day, []).append(dl)
    daily_brier_diff = [float(np.mean(v)) for v in by_day_brier.values()]
    daily_logscore_diff = [float(np.mean(v)) for v in by_day_ls.values()]
    b_lo, b_hi = stationary_block_bootstrap_ci(daily_brier_diff, mean_block=7, seed=seed)
    ls_lo, ls_hi = stationary_block_bootstrap_ci(daily_logscore_diff, mean_block=7, seed=seed)
    return {
        "model_brier": mb, "market_brier": kb,
        "model_logscore": ml, "market_logscore": kl,
        "brier_diff_ci_lo": b_lo, "brier_diff_ci_hi": b_hi,
        "logscore_diff_ci_lo": ls_lo, "logscore_diff_ci_hi": ls_hi,
        "model_beats_market": bool(mb < kb and ml < kl and b_hi < 0.0 and ls_hi < 0.0),
        "n": len(we),
    }


@dataclass(frozen=True)
class ReplayInput:
    trading_day: str
    calibrated_p_yes: float   # prob produced by the candidate calibrator config
    market_p_yes: float       # market YES price at signal
    outcome_yes: int          # fixed settled outcome


def _kelly_contracts(
    prob_win: float, price: float, kelly_fraction: float, bankroll: float
) -> int:
    """Fractional-Kelly contract count for a binary contract bought at `price`.

    f* = (prob_win - price) / (1 - price); clamp to [0,1]; size = f*·kelly·bankroll.
    NOTE: Kelly denominator omits the fee (conservative upper-bound contract count).
    Uses base Kelly without production haircuts (same-day correlation, max_pct cap)
    so T3 replay is an upper bound on realized production size.
    """
    if price <= 0.0 or price >= 1.0:
        return 0
    edge_frac = (prob_win - price) / (1.0 - price)
    edge_frac = max(0.0, min(1.0, edge_frac))
    dollars = edge_frac * kelly_fraction * bankroll
    return int(dollars / price)


def replay_one(
    ri: ReplayInput, *, edge_threshold: float, kelly_fraction: float, bankroll: float
) -> dict[str, Any]:
    """Replay one decision under a candidate calibrator's prob, against the fixed outcome."""
    p = min(max(ri.calibrated_p_yes, 0.0), 1.0)
    mk = ri.market_p_yes
    # Side & gross edge from the calibrated prob vs the market YES price.
    if p >= mk:
        side, prob_win, price = "YES", p, mk
    else:
        side, prob_win, price = "NO", 1.0 - p, 1.0 - mk
    gross_edge = prob_win - price
    # fee_per_contract takes side-denominated price (not model_prob/market_price kwargs)
    fee = _fee_per_contract(price, is_maker=False)
    net_edge = gross_edge - fee
    if net_edge <= edge_threshold:
        return {"traded": False, "side": side, "contracts": 0, "pnl": 0.0,
                "wagered": 0.0, "trading_day": ri.trading_day}
    contracts = _kelly_contracts(prob_win, price, kelly_fraction, bankroll)
    if contracts <= 0:
        return {"traded": False, "side": side, "contracts": 0, "pnl": 0.0,
                "wagered": 0.0, "trading_day": ri.trading_day}
    won = (side == "YES" and ri.outcome_yes == 1) or (side == "NO" and ri.outcome_yes == 0)
    payoff_per = (1.0 - price) if won else (-price)
    pnl = contracts * payoff_per - contracts * fee
    wagered = contracts * price
    return {"traded": True, "side": side, "contracts": contracts,
            "pnl": float(pnl), "wagered": float(wagered), "trading_day": ri.trading_day}


def edge_attribution(trades: list[SettledTrade]) -> dict[str, Any]:
    """T4: localize realized P&L between the forecasting surface (between) and the
    behavioral surface (greater/less favorites).

    Scope: weather_edge only — matches T1/T2 scope so favorite_bias trades do not
    inflate the greater_less bucket.
    """

    def _agg(subset: list[SettledTrade]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "pnl": float(sum(t.pnl for t in subset)),
            "wagered": float(sum(t.wagered for t in subset)),
            "roi": roi([t.pnl for t in subset], [t.wagered for t in subset]),
        }

    # Filter to weather_edge only before splitting by strike_type.
    we = [t for t in trades if t.strategy == "weather_edge"]
    between = [t for t in we if t.strike_type == "between"]
    gl = [t for t in we if t.strike_type in ("greater", "less")]
    return {"between": _agg(between), "greater_less": _agg(gl)}
