"""Favorite-bias pilot report — one-stop monitoring command.

Two complementary views of the live favorite_bias pilot:

  1. **Calibration** (spec §6.4): joins ``state/favorite_bias_audit.jsonl``
     (decision=='buy') to settled outcomes and emits the §6.3(2)
     map-overstatement tripwire. ``calibration_monitor`` is weather_edge-only,
     so this is the named calibration consumer for favorite_bias.

     Outcomes come from the settled result in ``data/shadow_settlements.full.json``
     first, and from snapshot NWS-CLI settlements only for markets that file
     does not resolve. The snapshot schedule stopped on 2026-09-04, so a stale
     or missing settlements file freezes this section; the report prints the
     newest outcome date per source to make that visible.

     Scored on the LIVE-CONFIG book only — rows whose mid is outside the current
     band, or that were scored by a superseded g(mid) artifact, are excluded.
     Pooling them judges a book the bot no longer trades: on 2026-09-01 that
     mixed a retired sub-0.50 cohort (gap -0.112) with the live one (+0.098),
     cancelling to -0.002 and reporting "inverted" for a book that was in fact
     understating by ~10pp.

  2. **Settled P&L / win-rate / ROI + kill-criteria verdict**: reads
     ``data/shadow_settlements.full.json`` (attribution filtered to
     ``strategy=='favorite_bias'``) and evaluates the pre-registered pilot kill
     criteria (win rate <55% at n>=40; ROI <-10% at any n).

Run (cached): PYTHONPATH=. uv run python scripts/favorite_bias_report.py
Run (fresh):  uv run python scripts/favorite_bias_report.py --pull-vps --refresh
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from python.analytics.attribution import attribute_many, summarize  # noqa: E402
from python.analytics.edge_audit import day_clustered_bootstrap_ci  # noqa: E402
from python.analytics.fees import fee_per_contract  # noqa: E402
from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.data.calibration import IsotonicCalibrator  # noqa: E402
from python.settings import Settings  # noqa: E402
from python.utils.ticker_dates import parse_ticker_date  # noqa: E402
from scripts._calibration_common import load_snapshot_data  # noqa: E402
from scripts._vps import VPS_TRADES_PATH, pull_audit_files, require_vps_host  # noqa: E402


@dataclass(frozen=True)
class CalRow:
    """One settled calibration trial: a distinct market, deduped to one row."""

    g: float
    outcome: int
    mid: float
    ask: float
    forecast_date: str
    map_version: str | None
    ticker: str = ""
    outcome_source: str = "nws_snapshot"


# Sample gates for the calibration tripwire. BOTH must clear: measured flow is
# ~2.8 markets/day, so n=30 alone buys only ~11 independent day-clusters.
CAL_N_FLOOR = 30
CAL_DAY_FLOOR = 20
CAL_BOOTSTRAP_B = 4000
# Absolute floor for the material-overstatement threshold, used when the book is
# empty or its mean claimed edge is unusually thin.
CAL_MIN_THRESHOLD = 0.03

DEFAULT_AUDIT = _PROJECT_ROOT / "state" / "favorite_bias_audit.jsonl"
DEFAULT_TRADES = _PROJECT_ROOT / "data" / "vps_trades.jsonl"
DEFAULT_FULL = _PROJECT_ROOT / "data" / "shadow_settlements.full.json"
DEFAULT_AUDIT_DIR = _PROJECT_ROOT / "data" / "vps_audit"
DEFAULT_SNAPSHOTS = _PROJECT_ROOT / "data" / "snapshots"

STRATEGY_NAME = "favorite_bias"


# ---------------------------------------------------------------------------
# Calibration pure functions (unit-tested)
# ---------------------------------------------------------------------------


def realized_vs_g(rows: list[CalRow]) -> dict:
    """Realized-vs-predicted summary. Returns mean_g, realized, gap, n."""
    n = len(rows)
    if n == 0:
        return {"mean_g": 0.0, "realized": 0.0, "gap": 0.0, "n": 0}
    mean_g = sum(r.g for r in rows) / n
    realized = sum(r.outcome for r in rows) / n
    return {"mean_g": mean_g, "realized": realized, "gap": realized - mean_g, "n": n}


def _day_ci(
    rows: list[CalRow],
    stat: Callable[[list[CalRow]], float],
    *,
    b: int = CAL_BOOTSTRAP_B,
) -> tuple[float | None, float | None]:
    """Day-clustered block-bootstrap CI, resampling whole forecast_dates.

    Matches ``_roi_day_ci``: markets settling on one date share a weather draw.
    (On this book clustering barely moves the interval — the flow is ~2.8
    markets/day across 9 cities, so within-day correlation is weak — but the
    convention is the repo standard and holds if the book ever concentrates.)
    """
    by_day: dict[str, list[CalRow]] = {}
    for r in rows:
        by_day.setdefault(r.forecast_date, []).append(r)
    days = list(by_day)
    if len(days) < 2:
        return None, None
    rng = random.Random(20260901)
    vals: list[float] = []
    for _ in range(b):
        picked = [x for _ in range(len(days)) for x in by_day[days[rng.randrange(len(days))]]]
        if picked:
            vals.append(stat(picked))
    if len(vals) < b // 2:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def _gap_stat(rows: list[CalRow]) -> float:
    n = len(rows)
    return (sum(r.outcome for r in rows) - sum(r.g for r in rows)) / n if n else 0.0


def _net_edge_stat(rows: list[CalRow]) -> float:
    """realized - (ask + taker fee): does the book still beat its execution price."""
    n = len(rows)
    if not n:
        return 0.0
    return sum(r.outcome - r.ask - fee_per_contract(r.ask) for r in rows) / n


def material_gap_threshold(rows: list[CalRow], *, floor: float = CAL_MIN_THRESHOLD) -> float:
    """How far ``g`` must overstate before the book stops being worth trading.

    Anchored to the book, not a round number. ``g`` is used both to select
    (``edge = g - ask - fee >= edge_floor``) and to size (Kelly at ``p = g``), so
    an overstatement of delta shifts every claimed edge down by delta. The book
    goes to zero EV when delta reaches the MEAN CLAIMED EDGE — that is the
    demote condition, and it self-calibrates as ``edge_floor`` and book
    composition change.

    Do NOT anchor this to the fee+spread toll: the taker fee is already inside
    ``edge`` before ``edge_floor`` applies, so that would double-count it.
    """
    if not rows:
        return floor
    mean_edge = sum(r.g - r.ask - fee_per_contract(r.ask) for r in rows) / len(rows)
    return max(mean_edge, floor)


def tripwire_verdict(
    gap_ci_lo: float | None,
    gap_ci_hi: float | None,
    n: int,
    n_days: int,
    threshold: float,
    *,
    n_floor: int = CAL_N_FLOOR,
    day_floor: int = CAL_DAY_FLOOR,
) -> str:
    """§6.3(2) map-overstatement tripwire.

    Was a bare sign test (``WARN if gap < 0``) on a quantity whose day-clustered
    SD is ~0.028. That fires ~50% of the time when the map is *perfectly*
    calibrated — measured on this corpus it flipped verdict 12 times across an
    expanding window and 19 times on a rolling one, and it produced a false
    "demote the strategy" recommendation at gap=-0.0016. The same false alarm was
    already reported in docs/research/2026-06-12-new-strategy-search-dossier.md
    and never fixed; this is its second occurrence.

    The dual sample gate matters: the old ``n_floor=30`` docstring claimed n
    accrues at ~1/day, so 30 markets ~= 30 days. Measured flow is ~2.8/day, so
    n=30 buys only ~11 independent clusters. A day floor is what makes the
    clustered CI meaningful.

    Four-way, mirroring ``fill_timing_report``. Demote language attaches to WARN
    only; a degrading point estimate still surfaces as INCONCLUSIVE, so nothing
    is hidden. WARN is deliberately hard to reach — the economic failure mode is
    already covered by ``kill_criteria_verdict`` and the day-clustered ROI CI, so
    this redundant second detector is tuned quiet.
    """
    if n < n_floor or n_days < day_floor:
        return "INSUFFICIENT_N"
    if gap_ci_lo is None or gap_ci_hi is None:
        return "INCONCLUSIVE"
    if gap_ci_hi <= -threshold:
        return "WARN"
    if gap_ci_lo >= -threshold:
        return "OK"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Kill-criteria pure function (unit-tested)
# ---------------------------------------------------------------------------


def kill_criteria_verdict(
    win_rate_pct: float | None,
    roi_pct: float | None,
    n: int,
) -> str:
    """Pre-registered pilot kill criteria.

    See the favorite_bias bullet in the root CLAUDE.md for the kill criteria.

    Criteria: disable if (win rate < 55% AND n >= 40) OR (ROI < -10% at any n).

    Returns:
        "INSUFFICIENT_N" if n == 0, or n < 40 with no ROI breach.
        "DISABLE"        if a kill criterion is met.
        "OK"             otherwise.

    The explicit ``is not None`` guards are load-bearing: ``favorite_bias_pnl``
    returns ``None`` rates when there is no settled / wagered data, and
    ``None < -10.0`` would raise ``TypeError``. The ROI floor is checked BEFORE
    the n>=40 win-rate gate so a catastrophic small-n ROI trips immediately.
    """
    if n == 0:
        return "INSUFFICIENT_N"
    if roi_pct is not None and roi_pct < -10.0:
        return "DISABLE"
    if n >= 40:
        if win_rate_pct is not None and win_rate_pct < 55.0:
            return "DISABLE"
        return "OK"
    return "INSUFFICIENT_N"


# ---------------------------------------------------------------------------
# Settled P&L (reuses the attribution framework)
# ---------------------------------------------------------------------------


def _roi_day_ci(rows: list[dict], *, b: int = 4000,
                seed: int = 42) -> tuple[float | None, float | None, int]:
    """Day-clustered bootstrap 95% CI on settled ROI, plus the distinct-day count.

    Trades sharing a forecast-day share one weather draw, so they are NOT
    independent: a per-trade CI materially overstates precision. Resamples whole
    forecast-days with replacement. Returns (lo, hi, n_days) as ROI *percent*, or
    (None, None, n_days) with fewer than 2 days to resample.
    """
    by_day: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        pnl, wag = r.get("pnl"), r.get("dollars")
        if not isinstance(pnl, (int, float)) or not isinstance(wag, (int, float)) or wag <= 0:
            continue
        day = str(r.get("timestamp") or "")[:10]
        if not day:
            continue
        by_day.setdefault(day, []).append((float(pnl), float(wag)))

    days = list(by_day)
    if len(days) < 2:
        return (None, None, len(days))
    rng = np.random.default_rng(seed)
    rois: list[float] = []
    n_days = len(days)
    for _ in range(b):
        pnl_sum = wag_sum = 0.0
        for c in rng.integers(0, n_days, size=n_days):
            for pnl, wag in by_day[days[int(c)]]:
                pnl_sum += pnl
                wag_sum += wag
        if wag_sum > 0:
            rois.append(100.0 * pnl_sum / wag_sum)
    if len(rois) < b // 2:
        return (None, None, n_days)
    arr = np.asarray(rois)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), n_days)


def favorite_bias_pnl(
    settlements_path: Path,
    *,
    live_config: tuple[float, float] | None = None,
    map_version: str | None = None,
) -> dict[str, Any]:
    """Settled P&L / win-rate / ROI for favorite_bias rows, via attribution.

    Win rate / ROI are ``None`` when there is no settled / wagered data so the
    kill-criteria verdict can distinguish "no signal" from a real breach.

    ``roi_day_lo`` / ``roi_day_hi`` carry the day-clustered 95% CI. Read the ROI
    against that CI, not the point estimate alone: on 2026-07-19 the headline
    +8.3% (n=162) carried a CI of [-7.4%, +24.5%], i.e. not distinguishable from
    zero. See docs/research/2026-07-19-new-data-edge-audit.md.

    Pass ``live_config=(band_low, band_high)`` to judge only the book the
    CURRENT config would produce — pooling a retired band with the live one
    cancels an ROI +21.9% live cohort against a -38.9% retired one into a
    misleading "indistinguishable from zero". Omitting it reproduces the
    unfiltered (legacy) behavior.

    ``map_version`` additionally restricts to rows scored by a specific g(mid)
    artifact hash and is meaningful ONLY together with ``live_config`` —
    passing it alone raises ``ValueError`` rather than silently no-op'ing.
    Do NOT wire ``map_version`` into this function from ``main()``: the map
    hash rotates on every g(mid) refit, and doing so would blank this
    section's ``n`` (and the kill-criteria gate it feeds) for ~2 weeks after
    every refit, at ~2.7 trades/day. Band-only filtering is what the pilot's
    kill criteria should be evaluated against; the map filter belongs to the
    calibration tripwire section only.

    ``n_backfilled`` counts how many of the final ``n`` rows carry
    ``settlement_source == "nws_cli_backfill"`` (a row with no
    ``settlement_source`` field counts as ``"kalshi"``, matching the writer's
    own default in ``fetch_shadow_settlements.py``) — visibility into how many
    of the settled trades behind the headline number are Kalshi-confirmed vs.
    reconstructed by ``backfill_settlements_from_snapshots.py``.
    """
    if map_version is not None and live_config is None:
        raise ValueError("map_version requires live_config to be set")

    rows: list[dict] = []
    if settlements_path.exists():
        rows = json.loads(settlements_path.read_text())
    fb = [r for r in rows if (r.get("strategy") or "") == STRATEGY_NAME]

    n_excluded_no_mid = 0
    n_excluded_retired_config = 0
    if live_config is not None:
        band_low, band_high = live_config
        fb_settled = [r for r in fb if r.get("settled")]
        n_excluded_no_mid = sum(
            1 for r in fb_settled if (r.get("metadata") or {}).get("mid") is None
        )
        kept_settled = filter_live_config_settlement_rows(
            fb_settled, map_version=map_version, band_low=band_low, band_high=band_high
        )
        n_excluded_retired_config = len(fb_settled) - len(kept_settled) - n_excluded_no_mid
        fb = filter_live_config_settlement_rows(
            fb, map_version=map_version, band_low=band_low, band_high=band_high
        )

    s = summarize(attribute_many(fb))
    roi_lo, roi_hi, n_days = _roi_day_ci(fb)
    n_backfilled = sum(
        1 for r in fb if (r.get("settlement_source") or "kalshi") == "nws_cli_backfill"
    )
    return {
        "n": s.n,
        "wins": s.wins,
        "losses": s.losses,
        "win_rate_pct": round(s.win_rate_pct, 1) if s.n else None,
        # Recorded (realized) P&L — see attribution.AttributionSummary docs.
        "roi_pct": round(s.roi_recorded_pct, 1) if s.total_wagered > 0 else None,
        "roi_day_lo": round(roi_lo, 1) if roi_lo is not None else None,
        "roi_day_hi": round(roi_hi, 1) if roi_hi is not None else None,
        "n_days": n_days,
        "net_pnl": round(s.total_recorded_pnl, 2),
        "wagered": round(s.total_wagered, 2),
        "fees": round(s.total_fees, 2),
        "n_excluded_retired_config": n_excluded_retired_config,
        "n_excluded_no_mid": n_excluded_no_mid,
        "n_backfilled": n_backfilled,
    }


# ---------------------------------------------------------------------------
# Fill-timing cohort report (Phase 1 pre-registered kill test)
# ---------------------------------------------------------------------------

# Kill-test constants (spec 2026-06-30 §7). FIRST_TOUCH_TTL_S is the FIXED pre-feature
# first-touch window (the original 120s TTL); it deliberately does NOT track the current
# global order_ttl_seconds — it defines "would this have filled before the feature?".
FIRST_TOUCH_TTL_S = 120
KILL_MIN_PATIENT_MARKETS = 20    # distinct settled markets that NEEDED patience
KILL_MIN_PATIENT_DAYS = 10       # distinct forecast-days in the patient cohort
KILL_MAX_EDGE_GAP = -0.03        # patient realized-edge may be at most 3pp below immediate


def _fb_market_cohorts(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse settled favorite_bias fill-rows to ONE record per market, split into two
    DISJOINT cohorts: 'patient' (>=1 fill landed after FIRST_TOUCH_TTL_S) vs 'immediate'
    (all fills within the first-touch window). Market-level is load-bearing — contracts in
    one market share one settlement outcome, so counting contracts would re-introduce the
    n-inflation bug that load_g_outcome_pairs / tripwire_verdict already avoid.
    """
    by_ticker: dict[str, dict] = {}
    for r in rows:
        if (r.get("strategy") or "") != STRATEGY_NAME:
            continue
        if not r.get("settled") or r.get("won") is None:
            continue
        if r.get("seconds_since_placement") is None:
            continue  # legacy pre-instrumentation row — cannot classify
        tk = r.get("market_ticker", "")
        fd = parse_ticker_date(tk)
        if fd is None:
            continue  # unparseable ticker — drop (keeps day-keys aligned by construction)
        md = r.get("metadata") or {}
        m = by_ticker.get(tk)
        if m is None:
            g = md.get("g_of_mid")
            mid = md.get("mid")
            m = {"ticker": tk, "day": fd.isoformat(), "won": bool(r["won"]),
                 "g": float(g) if g is not None else None,
                 "mid": float(mid) if mid is not None else None,
                 "pnl": 0.0, "wagered": 0.0, "contracts": 0, "patient": False,
                 "fill_minus_ask": []}
            by_ticker[tk] = m
        m["pnl"] += float(r.get("pnl") or 0.0)
        m["wagered"] += float(r.get("dollars") or 0.0)
        m["contracts"] += int(r.get("contracts") or 0)
        if float(r["seconds_since_placement"]) > FIRST_TOUCH_TTL_S:
            m["patient"] = True
        fma = r.get("fill_price_minus_signal_ask")
        if fma is not None:
            m["fill_minus_ask"].append(float(fma))
    markets = list(by_ticker.values())
    return ([m for m in markets if not m["patient"]],
            [m for m in markets if m["patient"]])


def _cohort_edge(markets: list[dict]) -> float | None:
    """Realized-vs-model edge = market win-rate - mean g(mid), equal-weighted over markets."""
    if not markets:
        return None
    gs = [m["g"] for m in markets if m["g"] is not None]
    if not gs:
        return None
    win_rate = sum(1 for m in markets if m["won"]) / len(markets)
    return win_rate - (sum(gs) / len(gs))


def _cohort_summary(markets: list[dict]) -> dict[str, Any]:
    n = len(markets)
    if n == 0:
        return {"n_markets": 0, "n_days": 0, "n_contracts": 0, "win_rate": None,
                "mean_g": None, "mean_mid": None, "realized_edge": None,
                "roi": None, "roi_day_lo": None, "mean_fill_minus_ask": None}
    gs = [m["g"] for m in markets if m["g"] is not None]
    mids = [m["mid"] for m in markets if m["mid"] is not None]
    deltas = [d for m in markets for d in m["fill_minus_ask"]]
    pnl = sum(m["pnl"] for m in markets)
    wagered = sum(m["wagered"] for m in markets)
    day_keys = [m["day"] for m in markets]
    lo, _hi = day_clustered_bootstrap_ci(
        day_keys, [m["pnl"] for m in markets], [m["wagered"] for m in markets]
    )
    return {
        "n_markets": n,
        "n_days": len(set(day_keys)),
        "n_contracts": sum(m["contracts"] for m in markets),
        "win_rate": round(sum(1 for m in markets if m["won"]) / n, 4),
        "mean_g": round(sum(gs) / len(gs), 4) if gs else None,
        "mean_mid": round(sum(mids) / len(mids), 4) if mids else None,
        "realized_edge": round(_cohort_edge(markets), 4) if gs else None,
        "roi": round(pnl / wagered, 4) if wagered > 0 else None,  # advisory
        "roi_day_lo": round(lo, 4),                               # advisory
        "mean_fill_minus_ask": round(sum(deltas) / len(deltas), 4) if deltas else None,
    }


def _edge_gap_day_ci(immediate: list[dict], patient: list[dict], *,
                     b: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    """Day-clustered bootstrap 95% CI on (patient_edge - immediate_edge). Resamples whole
    forecast-days with replacement so within-day / within-market correlation is respected.
    Returns (None, None) if a cohort lacks g(mid) or there are <2 days to resample."""
    by_day: dict[str, dict[str, list[dict]]] = {}
    for m in immediate:
        by_day.setdefault(m["day"], {"imm": [], "pat": []})["imm"].append(m)
    for m in patient:
        by_day.setdefault(m["day"], {"imm": [], "pat": []})["pat"].append(m)
    days = list(by_day.keys())
    if len(days) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    gaps: list[float] = []
    n_days = len(days)
    for _ in range(b):
        chosen = rng.integers(0, n_days, size=n_days)
        imm_ms: list[dict] = []
        pat_ms: list[dict] = []
        for c in chosen:
            d = by_day[days[int(c)]]
            imm_ms.extend(d["imm"])
            pat_ms.extend(d["pat"])
        ie, pe = _cohort_edge(imm_ms), _cohort_edge(pat_ms)
        if ie is None or pe is None:
            continue  # a cohort came up empty in this draw — skip
        gaps.append(pe - ie)
    if len(gaps) < b // 2:  # too many degenerate draws to trust the CI
        return (None, None)
    arr = np.asarray(gaps)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def fill_timing_report(
    rows: list[dict],
    *,
    live_config: tuple[float, float] | None = None,
    map_version: str | None = None,
) -> dict[str, Any]:
    """Market-level first-touch vs patient cohort comparison for favorite_bias, with a
    day-clustered CI on the realized-edge gap (spec 2026-06-30 §7).

    EXPLORATORY directional check, NOT a confirmatory equivalence test: it resolves only
    gross differences, so 'NO_RED_FLAG' means 'no evidence patient fills are materially
    worse', never 'proven equivalent'. Verdicts: INSUFFICIENT_N / NO_GO / INCONCLUSIVE /
    NO_RED_FLAG. Legacy rows without seconds_since_placement are ignored.

    Pass ``live_config=(band_low, band_high)`` to restrict the comparison to the
    live-config book, matching ``favorite_bias_pnl``. Omitting it reproduces the
    unfiltered (legacy) behavior. ``map_version`` is meaningful only together
    with ``live_config`` (raises ``ValueError`` otherwise) — see
    ``favorite_bias_pnl``'s docstring for why ``main()`` must never wire the
    map filter into this function.
    """
    if map_version is not None and live_config is None:
        raise ValueError("map_version requires live_config to be set")
    if live_config is not None:
        band_low, band_high = live_config
        rows = filter_live_config_settlement_rows(
            rows, map_version=map_version, band_low=band_low, band_high=band_high
        )
    immediate, patient = _fb_market_cohorts(rows)
    imm = _cohort_summary(immediate)
    pat = _cohort_summary(patient)

    edge_gap = None
    if imm["realized_edge"] is not None and pat["realized_edge"] is not None:
        edge_gap = round(pat["realized_edge"] - imm["realized_edge"], 4)
    gap_lo, gap_hi = _edge_gap_day_ci(immediate, patient)

    if pat["n_markets"] < KILL_MIN_PATIENT_MARKETS or pat["n_days"] < KILL_MIN_PATIENT_DAYS:
        verdict = "INSUFFICIENT_N"
    elif gap_hi is None:
        verdict = "INCONCLUSIVE"          # cannot compute the gap CI yet
    elif gap_hi <= KILL_MAX_EDGE_GAP:
        verdict = "NO_GO"                 # even the optimistic end has patient >=3pp worse
    elif gap_lo is not None and gap_lo >= KILL_MAX_EDGE_GAP:
        verdict = "NO_RED_FLAG"           # CI excludes a materially-negative gap
    else:
        verdict = "INCONCLUSIVE"          # CI straddles -3pp — underpowered, keep gathering

    return {
        "immediate": imm, "patient": pat,
        "edge_gap": edge_gap, "edge_gap_ci": [gap_lo, gap_hi],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Audit calibration I/O helper
# ---------------------------------------------------------------------------


def load_settled_results(full_path: Path) -> tuple[dict[str, tuple[int, str]], set[str]]:
    """Read settled market results from ``shadow_settlements.full.json``.

    Returns ``(results, tickers_in_file)``. ``results`` maps ``market_ticker`` to
    ``(yes_outcome, settlement_source)``; ``market_result`` is the market's
    YES-side result whatever side the trade took. ``tickers_in_file`` is every
    ticker with a row, settled or not, so a caller can tell "pending" from
    "absent". A ticker whose rows disagree on the result is left unresolved.
    A missing or malformed file yields empty results.
    """
    try:
        rows = json.loads(full_path.read_text())
    except (OSError, ValueError):
        return {}, set()
    if not isinstance(rows, list):
        return {}, set()

    results: dict[str, tuple[int, str]] = {}
    tickers: set[str] = set()
    conflicted: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        ticker = r.get("market_ticker")
        if not ticker:
            continue
        tickers.add(ticker)
        result = r.get("market_result")
        if result not in ("yes", "no"):
            continue
        entry = (1 if result == "yes" else 0, r.get("settlement_source") or "kalshi")
        prior = results.get(ticker)
        if prior is not None and prior[0] != entry[0]:
            conflicted.add(ticker)
        results.setdefault(ticker, entry)
    for ticker in conflicted:
        del results[ticker]
    return results, tickers


def _earliest_buy_rows(audit_path: Path) -> dict[str, dict]:
    """One raw audit row per ``market_ticker``: its earliest-ts buy decision.

    DEDUP (load-bearing): the strategy re-audits the same open market every scan
    tick, so a single market emits hundreds of identical-outcome buy rows. Each
    distinct ``market_ticker`` is ONE independent calibration trial. We keep the
    EARLIEST-ts buy decision: the per-day cap + ``mark_traded`` mean the order
    fires on the first qualifying scan, and g drifts upward as a market
    converges, so keep-last would inflate mean_g and bias the gap toward a false
    WARN — the exact artifact this dedup removes.
    """
    if not audit_path.exists():
        return {}

    best: dict[str, tuple[str, dict]] = {}
    for line in audit_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue

        if r.get("decision") != "buy":
            continue
        ticker = r.get("market_ticker", "")
        if parse_ticker_date(ticker) is None:
            continue
        if r.get("g_of_mid") is None or r.get("mid") is None:
            continue

        # ISO-8601 ts with a uniform +00:00 suffix sorts lexically == chronologically.
        # The writer (_audit) always stamps datetime.now(UTC).isoformat(); guard
        # against a future/foreign row with null or absent ts so the `<` comparison
        # cannot crash and an empty ts cannot spuriously win keep-first. Such a row
        # is sorted LAST (sentinel "~", 0x7E > all ISO chars) — still counted if it
        # is the only row for its ticker, but never preferred over a real timestamp.
        ts = r.get("ts") or "~"
        prior = best.get(ticker)
        if prior is None or ts < prior[0]:
            best[ticker] = (ts, r)

    return {ticker: r for ticker, (_ts, r) in best.items()}


def _snapshot_outcome(r: dict, forecast_date: str, settlements: dict) -> int | None:
    st = settlements.get((r.get("city"), forecast_date))
    if st is None:
        return None
    temp = st.get("high_f") if r.get("var") == "high" else st.get("low_f")
    if temp is None:
        return None
    yo = derive_outcome_yes(
        r.get("strike_type"), r.get("floor_strike"), r.get("cap_strike"), temp
    )
    return None if yo is None else int(yo)


def load_buy_rows(
    audit_path: Path,
    settlements: dict,
    settled_results: dict[str, tuple[int, str]] | None = None,
) -> list[CalRow]:
    """Load one ``CalRow`` per settled market from the audit JSONL.

    Filters to decision=="buy" rows, deduped by ``_earliest_buy_rows``. The
    outcome is the market's settled result from ``settled_results`` (see
    ``load_settled_results``) when the ticker is there. Otherwise it is derived
    from a FINAL snapshot settlement and strike_type/floor_strike/cap_strike (as
    written by FavoriteBiasStrategy._audit); ``settlements`` is keyed
    (city_str, observation_date_iso_str) as returned by ``load_snapshot_data``
    (preliminary rows already dropped by the caller). A market neither source
    resolves is omitted.

    Rows are returned in (forecast_date, ticker) order so the seeded
    day-clustered bootstrap gives the same interval on every run.
    """
    settled_results = settled_results or {}
    rows: list[CalRow] = []
    for ticker, r in _earliest_buy_rows(audit_path).items():
        fd = parse_ticker_date(ticker)
        if fd is None:
            continue
        forecast_date = fd.isoformat()

        resolved = settled_results.get(ticker)
        if resolved is not None:
            outcome, source = resolved
        else:
            snapshot = _snapshot_outcome(r, forecast_date, settlements)
            if snapshot is None:
                continue
            outcome, source = snapshot, "nws_snapshot"

        mid = float(r["mid"])
        # Fall back to mid when the ask is absent so an older row still counts
        # for the gap; the net-edge line reports its own n and is the only
        # consumer that depends on a real ask.
        ask_val = r.get("yes_ask")
        rows.append(
            CalRow(
                g=float(r["g_of_mid"]),
                outcome=outcome,
                mid=mid,
                ask=float(ask_val) if ask_val is not None else mid,
                forecast_date=forecast_date,
                map_version=r.get("map_version"),
                ticker=ticker,
                outcome_source=source,
            )
        )

    rows.sort(key=lambda row: (row.forecast_date, row.ticker))
    return rows


def unresolved_buy_counts(
    audit_path: Path,
    resolved_tickers: set[str],
    tickers_in_settlements: set[str],
    *,
    map_version: str | None,
    band_low: float,
    band_high: float,
) -> dict[str, int]:
    """Count live-config audited buys that have no outcome, by reason.

    ``pending``: the ticker has a row in the settlements file but no result yet.
    ``not_in_settlements``: no row at all — the order never filled, or it filled
    after the file was last rebuilt. Such a market gets no outcome unless a
    snapshot covers it. Same band/map rule as ``filter_live_config``.
    """
    counts = {"pending": 0, "not_in_settlements": 0}
    for ticker, r in _earliest_buy_rows(audit_path).items():
        if ticker in resolved_tickers:
            continue
        if not (band_low <= float(r["mid"]) < band_high):
            continue
        if map_version is not None and r.get("map_version") != map_version:
            continue
        key = "pending" if ticker in tickers_in_settlements else "not_in_settlements"
        counts[key] += 1
    return counts


def outcome_source_summary(rows: list[CalRow]) -> dict[str, dict[str, Any]]:
    """Per outcome source: row count and newest ``forecast_date``."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        entry = out.setdefault(r.outcome_source, {"n": 0, "newest_forecast_date": ""})
        entry["n"] += 1
        entry["newest_forecast_date"] = max(entry["newest_forecast_date"], r.forecast_date)
    return out


def filter_live_config(
    rows: list[CalRow],
    *,
    map_version: str | None,
    band_low: float,
    band_high: float,
) -> list[CalRow]:
    """Keep only rows the CURRENT configuration would still produce.

    Without this the tripwire judges a book the bot no longer trades. Measured
    2026-09-01: of 247 rows, 117 (47%) came from the sub-0.50 band retired on
    2026-08-11, and 121 (49%) were scored by a superseded map. Those two cohorts
    have opposite gaps (-0.112 and +0.098, each with a CI excluding zero), so
    pooling them cancels to -0.002 and reports "inverted" for a book whose live
    half is understating by +10pp.

    Passing ``map_version=None`` (calibrator failed to load) disables the
    artifact filter; the band filter still applies. A row whose own
    ``map_version`` is absent is DROPPED when filtering is active — it cannot be
    confirmed as scored by the live map, and this gate exists to be strict.
    """
    out = [r for r in rows if band_low <= r.mid < band_high]
    if map_version is not None:
        out = [r for r in out if r.map_version == map_version]
    return out


def filter_live_config_settlement_rows(
    rows: list[dict],
    *,
    map_version: str | None,
    band_low: float,
    band_high: float,
) -> list[dict]:
    """Dict-schema sibling of ``filter_live_config`` for shadow_settlements.full.json rows.

    Keeps only rows the CURRENT band/map would still produce, reading
    ``metadata.mid`` / ``metadata.map_version`` (the settlement-row schema)
    instead of ``CalRow``'s flat fields. Without this, the settled P&L, kill
    criteria, and fill-timing sections pool a retired sub-0.50-band cohort
    (ROI -38.9%) with the live one (+21.9%), the same cancellation
    ``filter_live_config`` already guards against for the calibration
    section — see that function's docstring.

    A row with no ``metadata.mid`` is DROPPED — it cannot be confirmed
    in-band (this affects a small number of reconciler-synthetic fill rows
    with a thin ``metadata`` dict; callers that need to report *why* a row
    was excluded should count these separately — see ``favorite_bias_pnl``).
    As in ``filter_live_config``, ``map_version=None`` disables the artifact
    filter but the band filter still applies, and a row whose own
    ``map_version`` is absent is dropped while artifact filtering is active.
    """
    out = []
    for r in rows:
        meta = r.get("metadata") or {}
        mid = meta.get("mid")
        if mid is None or not (band_low <= mid < band_high):
            continue
        if map_version is not None and meta.get("map_version") != map_version:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# VPS pull / refresh (mirror scripts/shadow_dashboard.py)
# ---------------------------------------------------------------------------


def pull_from_vps(local_path: Path, audit_dir: Path) -> None:
    """SCP the trade log + diagnostic audit JSONLs (incl. favorite_bias_audit)."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", f"{require_vps_host()}:{VPS_TRADES_PATH}", str(local_path)]
    print("Pulling trades from VPS...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        sys.exit(1)
    n = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    print(f"  -> {n} trades saved to {local_path}")
    pull_audit_files(audit_dir)


def refresh_settlements(trades_path: Path) -> None:
    """Run fetch_shadow_settlements.py to rebuild shadow_settlements.full.json.

    No ``--output`` override: the subprocess writes the default
    ``data/shadow_settlements.json`` and derives ``.full.json`` from it, which
    is the path favorite_bias_pnl reads.
    """
    script = _PROJECT_ROOT / "scripts" / "fetch_shadow_settlements.py"
    cmd = ["uv", "run", "python", str(script), "--trades", str(trades_path)]
    print("Refreshing settlement data from Kalshi API...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Warning: settlement refresh failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Favorite-bias pilot report")
    p.add_argument("--pull-vps", action="store_true", help="SCP fresh trades + audit from VPS")
    p.add_argument(
        "--refresh", action="store_true", help="Re-fetch Kalshi settlements before reporting"
    )
    p.add_argument(
        "--audit",
        default=None,
        help="Path to favorite_bias_audit.jsonl. Defaults to state/ (or the pulled "
        "copy under --audit-dir when --pull-vps is set without an explicit --audit).",
    )
    p.add_argument(
        "--audit-dir", default=str(DEFAULT_AUDIT_DIR), help="Dest for pulled audit files"
    )
    p.add_argument("--trades", default=str(DEFAULT_TRADES), help="Local trade-log path")
    p.add_argument(
        "--settlements", default=str(DEFAULT_FULL), help="shadow_settlements.full.json path"
    )
    p.add_argument(
        "--snapshots", default=str(DEFAULT_SNAPSHOTS), help="Snapshots dir for settlements"
    )
    p.add_argument("--json", default=None, help="Write the report as JSON to this path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    trades_path = Path(args.trades)
    audit_dir = Path(args.audit_dir)

    if args.pull_vps:
        pull_from_vps(trades_path, audit_dir)
    if args.refresh:
        refresh_settlements(trades_path)

    # Resolve the audit path: explicit --audit wins; else the pulled copy when
    # --pull-vps was used; else the local state/ file.
    if args.audit is not None:
        audit_path = Path(args.audit)
    elif args.pull_vps:
        audit_path = audit_dir / "favorite_bias_audit.jsonl"
    else:
        audit_path = DEFAULT_AUDIT

    # --- Calibration (realized-vs-g + tripwire) ---
    _f, settlements, _m = load_snapshot_data(Path(args.snapshots))
    settlements_path = Path(args.settlements)
    settled_results, settlement_tickers = load_settled_results(settlements_path)
    all_rows = load_buy_rows(audit_path, settlements, settled_results)

    # Judge only the book the CURRENT config would produce. Read band + map
    # identity from live config/artifact so this cannot drift from the strategy.
    fb_cfg = Settings.load().strategies.favorite_bias
    live_cal = IsotonicCalibrator(Path(fb_cfg.isotonic_path))
    cal_rows = filter_live_config(
        all_rows,
        map_version=live_cal.artifact_hash,
        band_low=fb_cfg.band_low,
        band_high=fb_cfg.band_high,
    )

    cal = realized_vs_g(cal_rows)
    cal_days = len({r.forecast_date for r in cal_rows})
    gap_lo, gap_hi = _day_ci(cal_rows, _gap_stat)
    cal_threshold = material_gap_threshold(cal_rows)
    cal_verdict = tripwire_verdict(
        gap_lo, gap_hi, cal["n"], cal_days, cal_threshold
    )
    net_edge = _net_edge_stat(cal_rows)
    net_lo, net_hi = _day_ci(cal_rows, _net_edge_stat)
    cal_sources = outcome_source_summary(cal_rows)
    cal_unresolved = unresolved_buy_counts(
        audit_path,
        {r.ticker for r in all_rows},
        settlement_tickers,
        map_version=live_cal.artifact_hash,
        band_low=fb_cfg.band_low,
        band_high=fb_cfg.band_high,
    )
    settlements_mtime = (
        datetime.fromtimestamp(settlements_path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        if settlements_path.exists()
        else None
    )

    # --- Settled P&L + kill criteria ---
    pnl = favorite_bias_pnl(
        Path(args.settlements),
        live_config=(fb_cfg.band_low, fb_cfg.band_high),
    )
    kill = kill_criteria_verdict(pnl["win_rate_pct"], pnl["roi_pct"], pnl["n"])

    # --- Render ---
    print("=== FAVORITE_BIAS PILOT REPORT ===")
    def _ci(lo: float | None, hi: float | None) -> str:
        return f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else "[n/a]"

    print(
        f"calibration: n={cal['n']} days={cal_days} mean_g={cal['mean_g']:.3f} "
        f"realized={cal['realized']:.3f} gap={cal['gap']:+.4f} "
        f"day-CI {_ci(gap_lo, gap_hi)} -> {cal_verdict}"
    )
    print(
        f"  (live-config book only: mid in [{fb_cfg.band_low}, {fb_cfg.band_high}) and "
        f"map {str(live_cal.artifact_hash)[:8]}; {len(all_rows) - len(cal_rows)} of "
        f"{len(all_rows)} audited markets excluded as retired band/map.)"
    )
    print(
        f"  net edge vs execution price (realized - ask - taker fee): {net_edge:+.4f} "
        f"day-CI {_ci(net_lo, net_hi)}"
    )
    print(
        "  (net edge is an UPPER bound: audited buys that never filled are "
        "adversely selected. They are counted only where a snapshot outcome "
        "exists, so none after the last snapshot date.)"
    )
    print(
        "  (n = DISTINCT settled markets; differs from the P&L n below, which "
        "counts settled TRADES.)"
    )
    sources = ", ".join(
        f"{name} n={v['n']} newest {v['newest_forecast_date']}"
        for name, v in sorted(cal_sources.items())
    )
    print(
        f"  outcome sources: {sources or 'none'}; settlements file "
        f"{settlements_path} rebuilt {settlements_mtime or 'MISSING'}"
    )
    print(
        f"  no outcome yet: {cal_unresolved['pending']} pending settlement, "
        f"{cal_unresolved['not_in_settlements']} not in the settlements file "
        "(never filled, or filled after its last rebuild)."
    )
    if not settled_results:
        print(
            "WARN: the settlements file is missing or holds no settled result, so "
            "calibration uses snapshot outcomes only and is frozen at the last "
            "snapshot date. Run with --refresh, or pass --settlements."
        )
    if cal_verdict == "WARN":
        print(
            f"WARN: g(mid) overstates the win rate by more than {cal_threshold:.3f} "
            "(the book's mean claimed edge) with 95% confidence — at that size the "
            "book is no longer EV-positive. Operator should review; demote "
            "(strategies.favorite_bias.enabled: false) if confirmed."
        )
    elif cal_verdict == "INCONCLUSIVE":
        print(
            f"  INCONCLUSIVE: gap CI straddles the -{cal_threshold:.3f} material "
            "threshold. Not an alert — reported so a drifting point estimate stays visible."
        )

    wr = f"{pnl['win_rate_pct']:.1f}%" if pnl["win_rate_pct"] is not None else "n/a"
    roi = f"{pnl['roi_pct']:+.1f}%" if pnl["roi_pct"] is not None else "n/a"
    print(
        f"settled P&L: n={pnl['n']} W={pnl['wins']} L={pnl['losses']} "
        f"win_rate={wr} roi={roi} net=${pnl['net_pnl']:+.2f} "
        f"wagered=${pnl['wagered']:.2f} fees=${pnl['fees']:.2f}"
    )
    print(
        f"  (live-config book only: mid in [{fb_cfg.band_low}, {fb_cfg.band_high}); "
        f"{pnl['n_excluded_retired_config']} excluded as out-of-band, "
        f"{pnl['n_excluded_no_mid']} excluded as un-attributable — no recorded mid "
        "(e.g. reconciler-synthetic fills).)"
    )
    if pnl["n_backfilled"] > 0:
        print(
            f"  ({pnl['n_backfilled']} of {pnl['n']} settled rows are NWS-CLI "
            "backfilled, not Kalshi-confirmed.)"
        )
    if pnl["roi_day_lo"] is not None:
        sig = "EXCLUDES 0 (edge established)" if pnl["roi_day_lo"] > 0 else "INCLUDES 0"
        print(
            f"  ROI day-clustered 95% CI: [{pnl['roi_day_lo']:+.1f}%, "
            f"{pnl['roi_day_hi']:+.1f}%] over {pnl['n_days']} days -> {sig}"
        )
        if pnl["roi_day_lo"] <= 0:
            print(
                "  NOTE: ROI is not yet distinguishable from zero. No size-up is "
                "justified until this lower bound clears 0."
            )
    print(f"KILL CRITERIA -> {kill}")
    if kill == "DISABLE":
        print(
            "DISABLE: pilot kill criterion met (win rate <55% at n>=40, or ROI <-10%). "
            "Set strategies.favorite_bias.enabled: false and restart."
        )

    # --- Fill-timing cohorts (Phase 1 pre-registered kill test) ---
    settle_rows = json.loads(settlements_path.read_text()) if settlements_path.exists() else []
    ft = fill_timing_report(
        settle_rows,
        live_config=(fb_cfg.band_low, fb_cfg.band_high),
    )
    print("\n--- Fill-timing cohorts (Phase 1 kill test — MARKET-level, exploratory) ---")
    for name in ("immediate", "patient"):
        b = ft[name]
        print(
            f"  {name:9s}: markets={b['n_markets']} days={b['n_days']} "
            f"contracts={b['n_contracts']} "
            f"win={b['win_rate']} mean_g={b['mean_g']} realized_edge={b['realized_edge']} "
            f"roi={b['roi']} roi_day_lo={b['roi_day_lo']} mean_mid={b['mean_mid']} "
            f"fill_minus_ask={b['mean_fill_minus_ask']}"
        )
    print(
        f"  edge_gap(patient-immediate)={ft['edge_gap']} "
        f"day-clustered 95% CI={ft['edge_gap_ci']}  ->  VERDICT: {ft['verdict']}"
    )
    print("  (NO_RED_FLAG = no evidence patient is >=3pp worse; NOT proof of equivalence.)")

    if args.json:
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "calibration": {
                **cal,
                "n_days": cal_days,
                "gap_day_ci": [gap_lo, gap_hi],
                "material_threshold": cal_threshold,
                "net_edge_vs_ask": net_edge,
                "net_edge_day_ci": [net_lo, net_hi],
                "n_excluded_retired_config": len(all_rows) - len(cal_rows),
                "tripwire": cal_verdict,
                "outcome_sources": cal_sources,
                "n_unresolved": cal_unresolved,
                "settlements_file_mtime": settlements_mtime,
            },
            "pnl": {**pnl, "filter": {"band_low": fb_cfg.band_low, "band_high": fb_cfg.band_high}},
            "kill_criteria": {
                "verdict": kill,
                "win_rate_pct": pnl["win_rate_pct"],
                "roi_pct": pnl["roi_pct"],
                "n": pnl["n"],
            },
            "fill_timing": {
                **ft,
                "filter": {"band_low": fb_cfg.band_low, "band_high": fb_cfg.band_high},
            },
        }
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.json}")


if __name__ == "__main__":
    main()
