"""Daily trade report: settled P&L, pending scorecard, calibration gap.

Usage:
    uv run python scripts/daily_trade_report.py --help
    uv run python scripts/daily_trade_report.py --pull-vps --refresh
    uv run python scripts/daily_trade_report.py --date 2026-05-02
    uv run python scripts/daily_trade_report.py --json /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to sys.path so this script can run standalone via uv run
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.common import ET, DailyVariable  # noqa: E402
from python.data.phase0_depth_monitor import (  # noqa: E402
    PHASE0_DEPTH_SINK_FILENAME,
    PHASE0_ELIGIBLE_SERIES,
)
from python.data.weather import (  # noqa: E402
    GFS_DERIVED_SPREAD_INFLATION,
    GFS_DERIVED_SPREAD_INFLATION_DEFAULT,
)
from python.execution.trade_log import (  # noqa: E402
    TradeLogEvent,
    is_pnl_relevant_row,
)
from python.utils.jsonl import iter_jsonl  # noqa: E402
from python.utils.ticker_dates import parse_ticker_city_var  # noqa: E402

# Strategies whose trades belong to the weather model. The SETTLED P&L block,
# settled_per_cell (post_flip_monitor.sh T1/T2/T3), the pending GFS/ECMWF
# scorecard, and the calibration gap all default to this allowlist so the
# model-free favorite_bias (which trades the same Kalshi weather series on
# shared city/var keys) cannot contaminate them.
WEATHER_STRATEGY_NAMES: frozenset[str] = frozenset({"weather_edge"})

# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def compute_bracket_fraction(temps: list[float], floor: float, cap: float) -> float:
    """Fraction of ensemble members in [floor, cap] inclusive.

    Matches production EnsembleForecast.fraction_between exactly.
    """
    if not temps:
        return 0.0
    return sum(1 for t in temps if floor <= t <= cap) / len(temps)


def apply_spread_inflation(temps: list[float], factor: float) -> list[float]:
    """Inflate ensemble spread around the mean by *factor*.

    inflated = mean + factor * (temp - mean)

    Matches production EnsembleForecast.inflated(factor).
    """
    if not temps:
        return []
    if factor == 1.0:
        return list(temps)
    mean = sum(temps) / len(temps)
    return [mean + factor * (t - mean) for t in temps]


def assess_risk(
    gfs_raw: float,
    ecmwf_raw: float,
    gfs_mean: float,
    ecmwf_mean: float,
) -> str:
    """Classify risk level for a pending NO trade based on raw bracket fractions.

    Priority: SAFE WIN > SPLIT MODEL > DANGER > HIGH > Medium-High > Medium > Low.
    SPLIT MODEL fires when GFS/ECMWF ensemble means disagree by ≥ 3°F.
    DANGER fires when both models independently agree the bracket is likely.
    """
    if gfs_raw == 0.0 and ecmwf_raw == 0.0:
        return "SAFE WIN"
    if abs(gfs_mean - ecmwf_mean) >= 3.0:
        return "SPLIT MODEL"
    if gfs_raw >= 0.25 and ecmwf_raw >= 0.15:
        return "DANGER"
    if ecmwf_raw >= 0.30 or gfs_raw >= 0.35:
        return "HIGH"
    avg = (gfs_raw + ecmwf_raw) / 2
    if avg >= 0.15:
        return "Medium-High"
    if avg >= 0.08:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastData:
    city: str
    forecast_date: str
    var: str
    gfs_temps: list[float]
    ecmwf_temps: list[float]
    gfs_mean: float
    ecmwf_mean: float


@dataclass
class PendingRisk:
    ticker: str
    city: str
    var: str
    floor_strike: float
    cap_strike: float
    forecast_date: str
    model_p: float
    fill_no: float
    contracts: int
    dollars: float
    gfs_raw_frac: float
    ecmwf_raw_frac: float
    gfs_inflated_frac: float
    ecmwf_inflated_frac: float
    gfs_mean: float
    ecmwf_mean: float
    inflation_factor: float
    risk_level: str


@dataclass
class SettledTrade:
    ticker: str
    city: str
    var: str
    floor_strike: float | None
    cap_strike: float | None
    forecast_date: str
    side: str
    model_p: float
    fill_no: float
    contracts: int
    dollars: float
    pnl: float
    is_win: bool
    raw_ensemble_prob: float | None


@dataclass
class DailyPnL:
    trade_date: str
    wins: list[SettledTrade] = field(default_factory=list)
    losses: list[SettledTrade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.wins) + len(self.losses)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.wins + self.losses)

    @property
    def total_wagered(self) -> float:
        return sum(t.dollars for t in self.wins + self.losses)


# ---------------------------------------------------------------------------
# Snapshot / forecast I/O
# ---------------------------------------------------------------------------


def find_latest_snapshot_run(snapshots_dir: Path, target_date: date) -> Path | None:
    """Return path to the most recent run subdirectory for *target_date*.

    Run dirs are named numerically (e.g. "0500", "0900", "1400", "2000").
    Non-numeric entries are ignored. Returns None if no run dirs exist.
    """
    date_dir = snapshots_dir / target_date.isoformat()
    if not date_dir.exists():
        return None
    runs = sorted(d for d in date_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return runs[-1] if runs else None


def load_forecast_data(
    run_dir: Path,
    forecast_date: str,
    cities: set[str],
) -> dict[tuple[str, str], ForecastData]:
    """Load GFS + ECMWF ensemble member temps from a snapshot run directory.

    Returns a dict keyed by (city, "high"/"low").
    Only includes entries whose forecast_date matches and city is in *cities*.
    Missing file → returns {}.
    """
    fc_path = run_dir / "weather_forecasts.json"
    if not fc_path.exists():
        return {}

    with open(fc_path) as f:
        raw: list[dict[str, Any]] = json.load(f)

    gfs_by_key: dict[tuple[str, str], tuple[list[float], float]] = {}
    ecmwf_by_key: dict[tuple[str, str], tuple[list[float], float]] = {}

    for entry in raw:
        city = entry.get("city", "")
        if city not in cities:
            continue
        if entry.get("forecast_date") != forecast_date:
            continue
        daily_var = entry.get("daily_variable", "")
        var = "high" if "max" in daily_var else "low"
        model = entry.get("model", "")
        temps: list[float] = entry.get("temperatures", [])
        mean: float = entry.get("mean", sum(temps) / len(temps) if temps else 0.0)

        key = (city, var)
        if "gfs" in model.lower():
            gfs_by_key[key] = (temps, mean)
        elif "ecmwf" in model.lower():
            ecmwf_by_key[key] = (temps, mean)

    result: dict[tuple[str, str], ForecastData] = {}
    for key in set(gfs_by_key) | set(ecmwf_by_key):
        city, var = key
        gfs_temps, gfs_mean = gfs_by_key.get(key, ([], 0.0))
        ecmwf_temps, ecmwf_mean = ecmwf_by_key.get(key, ([], 0.0))
        result[key] = ForecastData(
            city=city,
            forecast_date=forecast_date,
            var=var,
            gfs_temps=gfs_temps,
            ecmwf_temps=ecmwf_temps,
            gfs_mean=gfs_mean,
            ecmwf_mean=ecmwf_mean,
        )

    return result


# ---------------------------------------------------------------------------
# Record loading + filtering
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load shadow_settlements.full.json. Returns list of trade dicts."""
    with open(path) as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def filter_records(
    records: list[dict[str, Any]],
    trade_date: str | None,
    mode: str,
) -> list[dict[str, Any]]:
    """Filter records by trade date (YYYY-MM-DD) and execution mode.

    trade_date=None keeps all dates. mode="all" keeps all modes.
    """
    result = []
    for r in records:
        ts = (r.get("timestamp") or "")[:10]
        if trade_date and ts != trade_date:
            continue
        m = r.get("mode", "")
        if mode != "all" and m != mode:
            continue
        result.append(r)
    return result


def filter_by_strategy(
    records: list[dict[str, Any]],
    allowed: frozenset[str] | None,
) -> list[dict[str, Any]]:
    """Keep rows whose strategy is in ``allowed``.

    ``allowed=None`` keeps every row. A row's strategy is read from the
    top-level ``strategy`` field; missing/empty maps to ``"unknown"`` (NOT
    weather_edge), so the weather allowlist correctly excludes legacy untagged
    rows. Strategy on settlement rows is reliable after the
    fetch_shadow_settlements build-layer enrichment.
    """
    if allowed is None:
        return list(records)
    return [r for r in records if (r.get("strategy") or "unknown") in allowed]


# ---------------------------------------------------------------------------
# Settled P&L
# ---------------------------------------------------------------------------


def _floor_cap(r: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract floor_strike/cap_strike.

    Pending trades store strikes ONLY in metadata; settled records may also
    have them at top-level. Metadata wins so pending rows are never missed.
    """
    meta = r.get("metadata") or {}
    meta_floor = meta.get("floor_strike")
    meta_cap = meta.get("cap_strike")
    floor = meta_floor if meta_floor is not None else r.get("floor_strike")
    cap = meta_cap if meta_cap is not None else r.get("cap_strike")
    return floor, cap


def _no_fill_price(r: dict[str, Any]) -> float:
    """Cost per NO contract: fill_price when present (live), else dollars/contracts (shadow)."""
    fill = r.get("fill_price")
    if fill is not None:
        return float(fill)
    contracts = int(r.get("contracts") or 1) or 1
    return float(r.get("dollars") or 0) / contracts


def build_settled_report(records: list[dict[str, Any]]) -> list[DailyPnL]:
    """Build per-day P&L from settled (non-pending) records.

    Pending records (market_result is None) are excluded.
    Days are sorted chronologically.
    """
    by_date: dict[str, DailyPnL] = {}

    for r in records:
        result = r.get("market_result")
        if result is None:
            continue

        ts = (r.get("timestamp") or "")[:10]
        meta = r.get("metadata") or {}
        side = r.get("side", "NO")
        is_win = (side == "NO" and result == "no") or (side == "YES" and result == "yes")
        floor, cap = _floor_cap(r)

        st = SettledTrade(
            ticker=r.get("market_ticker", ""),
            city=meta.get("city") or "",
            var=meta.get("market_type") or "",
            floor_strike=float(floor) if floor is not None else None,
            cap_strike=float(cap) if cap is not None else None,
            forecast_date=str(meta.get("forecast_date") or ""),
            side=side,
            model_p=float(r.get("model_probability") or 0),
            fill_no=_no_fill_price(r),
            contracts=int(r.get("contracts") or 0),
            dollars=float(r.get("dollars") or 0),
            pnl=float(r.get("pnl") or 0),
            is_win=is_win,
            raw_ensemble_prob=meta.get("raw_ensemble_prob"),
        )

        if ts not in by_date:
            by_date[ts] = DailyPnL(trade_date=ts)
        day = by_date[ts]
        (day.wins if is_win else day.losses).append(st)

    return [by_date[d] for d in sorted(by_date)]


def aggregate_per_cell(settled_days: list[DailyPnL]) -> dict[str, dict[str, int]]:
    """Aggregate win + loss counts per (city, var) across the report window.

    Consumed by post_flip_monitor.sh T3 to compute true per-cell loss rate
    without re-deriving the wins side from raw trade-log rows. Key shape
    matches operator-visible display and config convention (e.g.
    ``weather.emos_enabled_cells: [MIA/high, LAX/low]``). Dedup is upstream:
    ``state/trades.jsonl`` enforces one ``event="fill"`` per ``order_id`` via
    FillsReconciler, so each settled trade contributes exactly one count.

    IMPORTANT: ``main()`` feeds this ``settled_days`` built from the
    WEATHER-ONLY ``weather_filtered`` record set (scoped to
    ``WEATHER_STRATEGY_NAMES``), so ``settled_per_cell`` reflects weather_edge
    only. This is deliberate — other strategies (e.g. the model-free
    favorite_bias) trade the same Kalshi weather series on shared (city, var)
    keys, and including them would corrupt the weather pilot's
    post_flip_monitor.sh rollback triggers. Do not re-introduce non-weather
    rows here.
    """
    stats: dict[str, dict[str, int]] = {}
    for day in settled_days:
        for t in day.wins:
            key = f"{t.city}/{t.var}"
            stats.setdefault(key, {"n_wins": 0, "n_losses": 0})["n_wins"] += 1
        for t in day.losses:
            key = f"{t.city}/{t.var}"
            stats.setdefault(key, {"n_wins": 0, "n_losses": 0})["n_losses"] += 1
    return stats


# ---------------------------------------------------------------------------
# Calibration gap
# ---------------------------------------------------------------------------


def compute_calibration_gap(
    records: list[dict[str, Any]],
    *,
    strategy_allowlist: frozenset[str] | None = WEATHER_STRATEGY_NAMES,
) -> dict[str, Any]:
    """Compare model-predicted loss probability vs actual loss rate on settled records.

    model_probability is always p(YES). Loss probability = model_p for NO trades,
    1 - model_p for YES trades. Pending records (market_result is None) are excluded.
    Reconciler-backfilled rows (model_probability is None) are also excluded —
    they have no model prediction to calibrate against, and coercing to 0 via
    `or 0` would bias avg_model_loss_p downward, inflating the ratio. The count
    of excluded model-unattributed settled rows is returned as n_unattributed.

    ``strategy_allowlist`` (default weather strategies) scopes calibration to the
    weather model: without this filter, the model-free favorite_bias (which
    carries ``model_probability``=g(mid)) would contaminate the weather
    calibration_ratio once its trades settle. ``None`` disables the filter.
    ``n_strategy_filtered`` in the return counts rows dropped by the allowlist.
    Returns {} if no settled records exist.
    """
    settled_all = [r for r in records if r.get("market_result") is not None]
    n_before_strategy = len(settled_all)
    settled_all = filter_by_strategy(settled_all, strategy_allowlist)
    n_strategy_filtered = n_before_strategy - len(settled_all)
    if not settled_all:
        return {}

    settled = [r for r in settled_all if r.get("model_probability") is not None]
    n_unattributed = len(settled_all) - len(settled)
    if not settled:
        return {
            "n_settled": 0,
            "n_unattributed": n_unattributed,
            "n_strategy_filtered": n_strategy_filtered,
            "n_wins": 0,
            "n_losses": 0,
            "win_rate": None,
            "actual_loss_rate": None,
            "avg_model_loss_p": None,
            "calibration_ratio": None,
        }

    n = len(settled)
    losses = sum(
        1
        for r in settled
        if (r.get("side") == "NO" and r.get("market_result") == "yes")
        or (r.get("side") == "YES" and r.get("market_result") == "no")
    )
    wins = n - losses
    # model_probability is p(YES); convert to loss probability per side
    avg_model_loss_p = (
        sum(
            float(r["model_probability"])
            if r.get("side", "NO") == "NO"
            else 1.0 - float(r["model_probability"])
            for r in settled
        )
        / n
    )
    actual_loss_rate = losses / n

    return {
        "n_settled": n,
        "n_unattributed": n_unattributed,
        "n_strategy_filtered": n_strategy_filtered,
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": wins / n,
        "actual_loss_rate": actual_loss_rate,
        "avg_model_loss_p": avg_model_loss_p,
        "calibration_ratio": actual_loss_rate / avg_model_loss_p if avg_model_loss_p > 0 else None,
    }


# ---------------------------------------------------------------------------
# Per-strategy settled P&L (generic — automatic for every strategy)
# ---------------------------------------------------------------------------


def build_strategy_pnl_summary(
    records: list[dict[str, Any]],
    *,
    exclude: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-strategy settled P&L / win-rate / ROI via the attribution framework.

    Reuses ``python.analytics.attribution`` (settled-only; win/loss from the
    recorded outcome, NOT model_probability — so it is correct for the
    model-free favorite_bias). ``exclude`` drops strategies that have their own
    dedicated report section (defaults to the intraday EMOS HIGH strategy) to
    avoid double-reporting. Returns a name -> stats dict sorted by strategy name.
    """
    from python.analytics.attribution import attribute_many, by_strategy

    if exclude is None:
        exclude = frozenset({INTRADAY_STRATEGY_NAME})
    bs = by_strategy(attribute_many(records))
    return {
        name: {
            "n": s.n,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate_pct": round(s.win_rate_pct, 1),
            # Headline P&L/ROI use RECORDED (realized) P&L, not the decomposition
            # total — the latter diverges from settlement P&L by reconciliation_diff
            # and dropped reconciler-synthetic rows entirely before the recorded-only
            # attribution fix. recorded matches the `settled` block exactly.
            "roi_pct": round(s.roi_recorded_pct, 1),
            "net_pnl": round(s.total_recorded_pnl, 2),
            "wagered": round(s.total_wagered, 2),
            "fees": round(s.total_fees, 2),
        }
        for name, s in sorted(bs.items())
        if name not in exclude
    }


def _format_strategy_pnl_section(summary: dict[str, dict[str, Any]]) -> list[str]:
    """Render the generic per-strategy settled-P&L table (settlements-file view)."""
    lines = ["", "=== SETTLED P&L BY STRATEGY (settlements-file view) ==="]
    if not summary:
        lines.append("  (no settled trades for this date/mode)")
        return lines
    lines.append(
        f"{'Strategy':<18} {'n':>4} {'W':>4} {'L':>4} {'WR%':>6} "
        f"{'ROI%':>7} {'P&L':>9} {'Wagered':>9}"
    )
    lines.append("-" * 72)
    for name, s in summary.items():
        lines.append(
            f"{name:<18} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} "
            f"{s['win_rate_pct']:>6.1f} {s['roi_pct']:>7.1f} "
            f"${s['net_pnl']:>+8.2f} ${s['wagered']:>8.2f}"
        )
    return lines


# ---------------------------------------------------------------------------
# Boundary-distance filter — per-cell kill watch (Path 7a, spec §6.5)
# ---------------------------------------------------------------------------


def _boundary_cell_label(row: dict[str, Any]) -> str | None:
    """Reconstruct ``"<CITY>/<var>"`` from a shadow_settlements row.

    ``shadow_settlements.full.json`` rows carry ``metadata.city`` +
    ``metadata.market_type``; fall back to ``parse_ticker_city_var`` on
    ``market_ticker`` for reconciler synthetics or older rows that lack
    metadata. Returns ``None`` if neither path resolves.
    """
    md = row.get("metadata") or {}
    city = md.get("city") or row.get("city")
    var = md.get("market_type") or row.get("market_type")
    if city and var:
        return f"{city}/{var}"
    ticker = row.get("market_ticker")
    if isinstance(ticker, str) and ticker:
        cv = parse_ticker_city_var(ticker)
        if cv is not None:
            return f"{cv[0]}/{cv[1]}"
    return None


def _settled_boundary_rows(
    rows: list[dict[str, Any]], cell: str, *, since: str | None = None
) -> list[dict[str, Any]]:
    """Settled weather_edge rows for ``cell``.

    When ``since`` (YYYY-MM-DD) is given, keep only trades on/after that date.
    This is load-bearing for the §6.5 kill: the filter suppresses grazers
    going forward, but pre-enablement grazer trades stay in the settlements
    file forever — mixing them contaminates the realized-ROI CI. ``since``
    should be the date the cell was enabled on the box.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("strategy") not in WEATHER_STRATEGY_NAMES:
            continue
        if not r.get("settled"):
            continue
        if _boundary_cell_label(r) != cell:
            continue
        if since is not None and (r.get("timestamp") or "")[:10] < since:
            continue
        out.append(r)
    return out


def _day_clustered_roi_ci(
    cell_rows: list[dict[str, Any]], *, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Day-clustered bootstrap of realized ROI on the post-filter cohort.

    ROI = sum(pnl) / sum(dollars) * 100 — ``dollars`` is the canonical wager
    (matches ``DailyPnL.total_wagered``, NOT ``exec_price * contracts`` which
    is side-wrong for NO bets). Resamples trading days with replacement,
    carrying all same-day trades together. Mirrors the methodology in
    docs/research/2026-05-28-boundary-aware-phase0-backtest.md §1.
    Returns (point, ci95_lo, ci95_hi).
    """
    from collections import defaultdict

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in cell_rows:
        ts = (r.get("timestamp") or "")[:10]
        if ts:
            by_day[ts].append(r)
    days = sorted(by_day)
    if not days:
        return (0.0, 0.0, 0.0)

    def _roi(rs: list[dict[str, Any]]) -> float:
        pnl = sum(float(r.get("pnl") or 0.0) for r in rs)
        wag = sum(float(r.get("dollars") or 0.0) for r in rs)
        return (pnl / wag * 100.0) if wag > 0 else 0.0

    point = _roi(cell_rows)
    rng = random.Random(seed)
    rois: list[float] = []
    for _ in range(n_boot):
        sample_days = [rng.choice(days) for _ in range(len(days))]
        sample_rows = [r for d in sample_days for r in by_day[d]]
        rois.append(_roi(sample_rows))
    rois.sort()
    lo = rois[int(0.025 * len(rois))]
    hi = rois[min(int(0.975 * len(rois)), len(rois) - 1)]
    return (point, lo, hi)


def render_boundary_filter_section(
    rows: list[dict[str, Any]],
    configured_cells: dict[str, float],
    *,
    since: str | None = None,
) -> str:
    """Spec §6.5 auto-revert kill watch for cells in ``weather.boundary_filter_cells``.

    Renders per-cell n, days-active, realized ROI, day-clustered CI95, and a
    ``keep`` / ``WATCH`` verdict. ``configured_cells`` maps ``"<CITY>/<var>"`` →
    °F threshold (i.e. ``settings.strategies.weather.boundary_filter_cells``).
    Returns ``""`` when no cells are configured so the report stays clean
    pre-deployment. The auto-revert decision (CI lo < 0 AND point < incumbent)
    is the operator's call; this section provides the inputs.

    ``since`` (YYYY-MM-DD, the cell enablement date via ``--boundary-since``)
    scopes the CI to the post-enablement cohort. WITHOUT it the CI mixes
    historical pre-enablement grazer trades (which the filter now suppresses)
    and is NOT a valid §6.5 signal — the header flags this loudly.
    """
    if not configured_cells:
        return ""
    if since is None:
        scope_note = (
            "  cohort: ALL settled rows — NOT post-enablement; pass "
            "--boundary-since <enable-date> for a valid §6.5 signal"
        )
    else:
        scope_note = f"  cohort: trades on/after {since} (post-enablement)"
    lines = [
        "",
        "=== Boundary-distance filter — per-cell kill watch (spec §6.5) ===",
        scope_note,
        "",
    ]
    lines.append(
        f"{'cell':<14} {'thr_f':>6} {'n':>4} {'days':>5} {'roi':>9} "
        f"{'ci95_lo':>9} {'ci95_hi':>9}  action"
    )
    for cell, threshold in configured_cells.items():
        cell_rows = _settled_boundary_rows(rows, cell, since=since)
        n = len(cell_rows)
        if n < 10:
            lines.append(
                f"{cell:<14} {threshold:>6.2f} {n:>4} {'-':>5} {'-':>9} "
                f"{'-':>9} {'-':>9}  pending (n<10)"
            )
            continue
        point, lo, hi = _day_clustered_roi_ci(cell_rows)
        days_active = len({(r.get("timestamp") or "")[:10] for r in cell_rows})
        action = "keep" if lo >= 0 else "WATCH (CI lo<0) — see spec §6.5"
        lines.append(
            f"{cell:<14} {threshold:>6.2f} {n:>4} {days_active:>5} "
            f"{point:>+8.2f}% {lo:>+8.2f}% {hi:>+8.2f}%  {action}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pending trade scorecard
# ---------------------------------------------------------------------------

_RISK_ORDER = {
    "DANGER": 0,
    "HIGH": 1,
    "SPLIT MODEL": 2,
    "Medium-High": 3,
    "Medium": 4,
    "Low": 5,
    "SAFE WIN": 6,
}


def _inflation_factor(city: str, var: str) -> float:
    """Return the spread inflation factor for city/var, matching production."""
    var_key = DailyVariable.TEMP_MAX if var == "high" else DailyVariable.TEMP_MIN
    return GFS_DERIVED_SPREAD_INFLATION.get(city.upper(), {}).get(
        var_key, GFS_DERIVED_SPREAD_INFLATION_DEFAULT
    )


def build_pending_scorecard(
    records: list[dict[str, Any]],
    snapshots_dir: Path,
    report_date: date | None = None,
) -> list[PendingRisk]:
    """Build risk scorecard for all unsettled (pending) records.

    For each pending trade, loads the most recent available GFS + ECMWF
    forecast, computes raw and spread-inflated bracket fractions for both
    models, and classifies the risk level.

    Snapshot lookup order: report_date (freshest) → trade_date ± 1 day.
    Records with no matching snapshot are silently dropped.
    Results are sorted by risk (DANGER first, SAFE WIN last).

    Args:
        records: Trade records (may be pre-filtered by date/mode).
        snapshots_dir: Path to data/snapshots root.
        report_date: The date being reported (--date arg); used as primary
            snapshot lookup date to get the freshest available forecast.
    """
    from collections import defaultdict

    pending = [r for r in records if r.get("market_result") is None]
    if not pending:
        return []

    # Group by (forecast_date, trade_date) to batch snapshot lookups
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in pending:
        meta = r.get("metadata") or {}
        fd = str(meta.get("forecast_date") or "")
        td = (r.get("timestamp") or "")[:10]
        if fd:
            groups[(fd, td)].append(r)

    result: list[PendingRisk] = []

    for (forecast_date_str, trade_date_str), group in groups.items():
        cities = {str((r.get("metadata") or {}).get("city") or "") for r in group} - {""}

        snap_dir: Path | None = None
        try:
            trade_dt = date.fromisoformat(trade_date_str)
            # Try report_date first (freshest snapshot), then trade_date ± 1 day
            candidates: list[date] = []
            if report_date is not None:
                candidates.append(report_date)
            for delta in (0, -1, 1):
                d = trade_dt + timedelta(days=delta)
                if d not in candidates:
                    candidates.append(d)
            for d in candidates:
                found = find_latest_snapshot_run(snapshots_dir, d)
                if found is not None:
                    snap_dir = found
                    break
        except ValueError:
            continue

        if snap_dir is None:
            continue

        forecasts = load_forecast_data(snap_dir, forecast_date_str, cities)

        for r in group:
            meta = r.get("metadata") or {}
            city = str(meta.get("city") or "")
            var = str(meta.get("market_type") or "")
            floor_raw, cap_raw = _floor_cap(r)
            if floor_raw is None or cap_raw is None or not city or not var:
                continue

            floor = float(floor_raw)
            cap = float(cap_raw)
            fc = forecasts.get((city, var))
            if fc is None:
                continue

            factor = _inflation_factor(city, var)

            gfs_raw = compute_bracket_fraction(fc.gfs_temps, floor, cap)
            ecmwf_raw = compute_bracket_fraction(fc.ecmwf_temps, floor, cap)

            gfs_inf = compute_bracket_fraction(
                apply_spread_inflation(fc.gfs_temps, factor), floor, cap
            )
            ecmwf_inf = compute_bracket_fraction(
                apply_spread_inflation(fc.ecmwf_temps, factor), floor, cap
            )

            risk = assess_risk(gfs_raw, ecmwf_raw, fc.gfs_mean, fc.ecmwf_mean)

            result.append(
                PendingRisk(
                    ticker=r.get("market_ticker", ""),
                    city=city,
                    var=var,
                    floor_strike=floor,
                    cap_strike=cap,
                    forecast_date=forecast_date_str,
                    model_p=float(r.get("model_probability") or 0),
                    fill_no=_no_fill_price(r),
                    contracts=int(r.get("contracts") or 0),
                    dollars=float(r.get("dollars") or 0),
                    gfs_raw_frac=gfs_raw,
                    ecmwf_raw_frac=ecmwf_raw,
                    gfs_inflated_frac=gfs_inf,
                    ecmwf_inflated_frac=ecmwf_inf,
                    gfs_mean=fc.gfs_mean,
                    ecmwf_mean=fc.ecmwf_mean,
                    inflation_factor=factor,
                    risk_level=risk,
                )
            )

    result.sort(key=lambda x: _RISK_ORDER.get(x.risk_level, 99))
    return result


# ---------------------------------------------------------------------------
# Phase 0 passive depth-probing summary
#
# Consumes rows written by ``python/data/phase0_depth_monitor.py`` to
# ``state/phase0_depth_probes.jsonl``. The §6.4 ladder re-anchoring is
# pre-registered as qualitative bands ("consistently >= 50", "in [10, 25)"),
# so this section reports raw percentiles + an empty-book frequency and lets
# the operator pick the binding band manually (spec §12.1).
# ---------------------------------------------------------------------------


# Default lookback matches the spec's 2-week Phase 0 measurement window.
PHASE0_DEFAULT_LOOKBACK_DAYS: int = 14

# Spec §6.4(b) / §11.4: intraday depth distribution is bucketed by
# ET hour over the 08:00–20:00 window (inclusive). Hours outside this
# window are still counted in the lookback-wide totals (single-pass —
# avoids a second filter) but are excluded from the per-hour buckets.
PHASE0_HOUR_BUCKET_START_ET: int = 8
PHASE0_HOUR_BUCKET_END_ET: int = 20


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile of ``values`` at ``pct`` in [0, 100].

    ``None`` on empty input. Pure helper to avoid taking a runtime
    dependency on numpy/statistics here (statistics.quantiles exists but
    its inclusive method requires n >= 2; we want a single canonical
    fallback for n == 1).
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _iter_phase0_rows(
    samples_path: Path,
    *,
    cutoff_ts_unix: float | None,
) -> Iterator[dict[str, Any]]:
    """Yield Phase-0 rows from the sink file, optionally filtered by ``ts_unix``."""
    for row in iter_jsonl(samples_path):
        if row.get("sample_kind") != "phase0_intraday_depth":
            # Future-proofs against the sink being multiplexed with other
            # passive monitors that share the JSONL convention.
            continue
        if cutoff_ts_unix is not None:
            ts = row.get("ts_unix")
            if not isinstance(ts, (int, float)) or ts < cutoff_ts_unix:
                continue
        yield row


def summarize_phase0_depth(
    samples_path: Path,
    *,
    lookback_days: int = PHASE0_DEFAULT_LOOKBACK_DAYS,
    now_utc: datetime | None = None,
    cells: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate Phase-0 depth samples into per-cell percentile summaries.

    Returns a dict with shape::

        {
            "lookback_days": int,
            "n_samples": int,                       # total across cells
            "cells": {
                "MIA/high": {
                    "n_samples": int,
                    "empty_book_frac": float | None,
                    "yes_bid_size_p10": float | None,
                    "yes_bid_size_p25": float | None,
                    "yes_bid_size_p50": float | None,
                    "yes_bid_size_p75": float | None,
                    "yes_bid_size_p90": float | None,
                    "yes_ask_size_p10": float | None,
                    ... (p25/p50/p75/p90 for yes_ask_size)
                    "nbbo_min_size_p50": float | None,  # min(bid, ask) per row
                    "by_hour": {                         # spec §6.4(b)
                        8: {
                            "n": int,
                            "empty_book_frac": float | None,
                            "yes_bid_size_p10": float | None,
                            "yes_bid_size_p25": float | None,
                            "yes_bid_size_p50": float | None,
                            "yes_bid_size_p75": float | None,
                            "yes_bid_size_p90": float | None,
                            "yes_ask_size_p10": float | None,
                            ... (p25/p50/p75/p90 for yes_ask_size)
                        },
                        ...
                        20: {...},
                    },
                },
                "LAX/high": {...},
            },
        }

    Hours outside ``[PHASE0_HOUR_BUCKET_START_ET,
    PHASE0_HOUR_BUCKET_END_ET]`` (inclusive) are still counted in the
    lookback-wide top-level aggregates — single-pass keeps the cost
    bounded and matches the spec's "lookback-wide" framing for the
    headline NBBO/empty-book numbers — but are excluded from the
    ``by_hour`` buckets per spec §6.4(b).

    Convention for empty hour buckets: hours with zero samples are
    OMITTED from the ``by_hour`` dict entirely (so consumers can
    iterate over only the hours that actually have data).

    Missing or empty sink file → ``{"lookback_days": d, "n_samples": 0,
    "cells": {}}`` (consumer renders an "(no Phase 0 samples yet)" row).
    """
    cells_to_report = cells or [pair[0] for pair in PHASE0_ELIGIBLE_SERIES.values()]
    now = now_utc or datetime.now(UTC)
    cutoff = (now - timedelta(days=lookback_days)).timestamp()

    per_cell_bid: dict[str, list[float]] = {c: [] for c in cells_to_report}
    per_cell_ask: dict[str, list[float]] = {c: [] for c in cells_to_report}
    per_cell_min: dict[str, list[float]] = {c: [] for c in cells_to_report}
    per_cell_n: dict[str, int] = {c: 0 for c in cells_to_report}
    per_cell_empty: dict[str, int] = {c: 0 for c in cells_to_report}

    # By-hour buckets (ET hour ∈ [PHASE0_HOUR_BUCKET_START_ET,
    # PHASE0_HOUR_BUCKET_END_ET]).
    per_cell_hour_bid: dict[str, dict[int, list[float]]] = {c: {} for c in cells_to_report}
    per_cell_hour_ask: dict[str, dict[int, list[float]]] = {c: {} for c in cells_to_report}
    per_cell_hour_n: dict[str, dict[int, int]] = {c: {} for c in cells_to_report}
    per_cell_hour_empty: dict[str, dict[int, int]] = {c: {} for c in cells_to_report}

    n_total = 0
    for row in _iter_phase0_rows(samples_path, cutoff_ts_unix=cutoff):
        cell = row.get("cell")
        if not isinstance(cell, str) or cell not in per_cell_bid:
            continue
        n_total += 1
        per_cell_n[cell] += 1

        # ET-hour bucket eligibility — convert ts_unix → ET hour and
        # restrict to the spec window. Rows missing a usable timestamp
        # drop out of the by-hour buckets but still count in the
        # lookback-wide aggregates.
        ts_raw = row.get("ts_unix")
        et_hour: int | None = None
        if isinstance(ts_raw, (int, float)):
            et_hour = datetime.fromtimestamp(float(ts_raw), tz=ET).hour
        in_hour_window = (
            et_hour is not None
            and PHASE0_HOUR_BUCKET_START_ET <= et_hour <= PHASE0_HOUR_BUCKET_END_ET
        )

        if row.get("empty_book"):
            per_cell_empty[cell] += 1
            if in_hour_window and et_hour is not None:
                per_cell_hour_n[cell][et_hour] = per_cell_hour_n[cell].get(et_hour, 0) + 1
                per_cell_hour_empty[cell][et_hour] = per_cell_hour_empty[cell].get(et_hour, 0) + 1
            continue

        bid_size_raw = row.get("yes_bid_size")
        ask_size_raw = row.get("yes_ask_size")
        bid_size = float(bid_size_raw) if isinstance(bid_size_raw, (int, float)) else None
        ask_size = float(ask_size_raw) if isinstance(ask_size_raw, (int, float)) else None
        if bid_size is not None and bid_size >= 0:
            per_cell_bid[cell].append(bid_size)
        if ask_size is not None and ask_size >= 0:
            per_cell_ask[cell].append(ask_size)
        if bid_size is not None and ask_size is not None and bid_size >= 0 and ask_size >= 0:
            per_cell_min[cell].append(min(bid_size, ask_size))

        if in_hour_window and et_hour is not None:
            per_cell_hour_n[cell][et_hour] = per_cell_hour_n[cell].get(et_hour, 0) + 1
            if bid_size is not None and bid_size >= 0:
                per_cell_hour_bid[cell].setdefault(et_hour, []).append(bid_size)
            if ask_size is not None and ask_size >= 0:
                per_cell_hour_ask[cell].setdefault(et_hour, []).append(ask_size)

    summary_cells: dict[str, dict[str, Any]] = {}
    for cell in cells_to_report:
        n = per_cell_n[cell]
        bids = per_cell_bid[cell]
        asks = per_cell_ask[cell]
        mins = per_cell_min[cell]
        empty_frac: float | None = (per_cell_empty[cell] / n) if n else None

        by_hour: dict[int, dict[str, Any]] = {}
        for hour, hour_n in per_cell_hour_n[cell].items():
            if hour_n <= 0:
                continue
            hour_bids = per_cell_hour_bid[cell].get(hour, [])
            hour_asks = per_cell_hour_ask[cell].get(hour, [])
            hour_empty = per_cell_hour_empty[cell].get(hour, 0)
            by_hour[hour] = {
                "n": hour_n,
                "empty_book_frac": (hour_empty / hour_n) if hour_n else None,
                "yes_bid_size_p10": _percentile(hour_bids, 10),
                "yes_bid_size_p25": _percentile(hour_bids, 25),
                "yes_bid_size_p50": _percentile(hour_bids, 50),
                "yes_bid_size_p75": _percentile(hour_bids, 75),
                "yes_bid_size_p90": _percentile(hour_bids, 90),
                "yes_ask_size_p10": _percentile(hour_asks, 10),
                "yes_ask_size_p25": _percentile(hour_asks, 25),
                "yes_ask_size_p50": _percentile(hour_asks, 50),
                "yes_ask_size_p75": _percentile(hour_asks, 75),
                "yes_ask_size_p90": _percentile(hour_asks, 90),
            }

        summary_cells[cell] = {
            "n_samples": n,
            "empty_book_frac": empty_frac,
            "yes_bid_size_p10": _percentile(bids, 10),
            "yes_bid_size_p25": _percentile(bids, 25),
            "yes_bid_size_p50": _percentile(bids, 50),
            "yes_bid_size_p75": _percentile(bids, 75),
            "yes_bid_size_p90": _percentile(bids, 90),
            "yes_ask_size_p10": _percentile(asks, 10),
            "yes_ask_size_p25": _percentile(asks, 25),
            "yes_ask_size_p50": _percentile(asks, 50),
            "yes_ask_size_p75": _percentile(asks, 75),
            "yes_ask_size_p90": _percentile(asks, 90),
            "nbbo_min_size_p50": _percentile(mins, 50),
            "by_hour": by_hour,
        }

    return {
        "lookback_days": lookback_days,
        "n_samples": n_total,
        "cells": summary_cells,
    }


def _format_phase0_section(summary: dict[str, Any]) -> list[str]:
    """Render the Phase-0 depth section as report lines.

    Always returns at least a header so the report reader sees the
    section exists; the body collapses to a single "(no Phase 0 samples
    yet)" line when the sink is empty. Spec §11.4 + §6.4(b).
    """
    lines: list[str] = ["", "=== PHASE 0 DEPTH PROBING ==="]
    n_total = int(summary.get("n_samples", 0))
    lookback = int(summary.get("lookback_days", PHASE0_DEFAULT_LOOKBACK_DAYS))
    if n_total == 0:
        lines.append(f"  (no Phase 0 samples in last {lookback}d)")
        return lines

    lines.append(f"  Lookback: {lookback}d  |  Total samples: {n_total}")
    lines.append(
        f"  {'Cell':<10} {'n':>5} {'empty%':>7} "
        f"{'bidP10':>7} {'bidP25':>7} {'bidP50':>7} {'bidP90':>7} "
        f"{'askP10':>7} {'askP25':>7} {'askP50':>7} {'askP90':>7}"
    )
    lines.append("  " + "-" * 88)
    cells = summary.get("cells", {}) or {}

    def _fmt(v: Any, width: int = 7, fmt: str = ".0f") -> str:
        if v is None:
            return f"{'-':>{width}}"
        return f"{v:>{width}{fmt}}"

    for cell_name in sorted(cells.keys()):
        c = cells[cell_name]
        empty = c.get("empty_book_frac")
        empty_str = f"{empty * 100:>6.1f}%" if isinstance(empty, (int, float)) else "     -"
        lines.append(
            f"  {cell_name:<10} {c.get('n_samples', 0):>5d} {empty_str:>7}"
            f" {_fmt(c.get('yes_bid_size_p10'))} {_fmt(c.get('yes_bid_size_p25'))}"
            f" {_fmt(c.get('yes_bid_size_p50'))} {_fmt(c.get('yes_bid_size_p90'))}"
            f" {_fmt(c.get('yes_ask_size_p10'))} {_fmt(c.get('yes_ask_size_p25'))}"
            f" {_fmt(c.get('yes_ask_size_p50'))} {_fmt(c.get('yes_ask_size_p90'))}"
        )

    # Per-spec §6.4(b): surface the intraday distribution by ET hour.
    # Compact per-cell line: bid P50 / ask P50 at each hour with data,
    # within the 08:00–20:00 ET window. Hours with zero samples are
    # rendered as a dash.
    any_hour_data = any(bool(c.get("by_hour")) for c in cells.values())
    if any_hour_data:
        lines.append("")
        lines.append(
            f"  By ET hour (bid P50 / ask P50, "
            f"{PHASE0_HOUR_BUCKET_START_ET:02d}:00–"
            f"{PHASE0_HOUR_BUCKET_END_ET:02d}:00):"
        )
        for cell_name in sorted(cells.keys()):
            by_hour = cells[cell_name].get("by_hour", {}) or {}
            if not by_hour:
                lines.append(f"    {cell_name:<10} (no in-window samples)")
                continue

            # JSON round-trip turns int keys into strings — accept both.
            def _hour_key(h: int, _by_hour: dict[Any, Any] = by_hour) -> Any:
                if h in _by_hour:
                    return h
                if str(h) in _by_hour:
                    return str(h)
                return None

            parts: list[str] = []
            for hour in range(PHASE0_HOUR_BUCKET_START_ET, PHASE0_HOUR_BUCKET_END_ET + 1):
                key = _hour_key(hour)
                if key is None:
                    parts.append(f"{hour:02d}h:-/-")
                    continue
                bucket = by_hour[key]
                bid_p50 = bucket.get("yes_bid_size_p50")
                ask_p50 = bucket.get("yes_ask_size_p50")
                bid_str = f"{bid_p50:.0f}" if isinstance(bid_p50, (int, float)) else "-"
                ask_str = f"{ask_p50:.0f}" if isinstance(ask_p50, (int, float)) else "-"
                parts.append(f"{hour:02d}h:{bid_str}/{ask_str}")
            lines.append(f"    {cell_name:<10} " + "  ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Intraday EMOS HIGH strategy section (spec §9.2(c), §11.4, §8.4(b))
#
# Path-(b) chosen by v1.3 IMPORTANT #2: consec-loss runs are computed
# out-of-band by this daily-report scan instead of via an on_settle
# strategy hook (the Strategy ABC has no such hook).
#
# Schema constraints (cited verbatim from the spec):
#   - state/trades.jsonl is monolithic. No in-process rotation (OQ-3);
#     the Phase 0 sink rotates via ForecastsRawWriter but trades.jsonl
#     does not. A single-file scan with optional --pull-vps is enough.
#   - Settlement rows do NOT carry metadata["cell"]
#     (engine.py:_build_settlement_entry verified; v1.3 MINOR #2). Cell is derived
#     from the row's market_ticker via parse_ticker_city_var.
#   - Fill rows DO carry metadata, but win/loss outcome is on the
#     settlement row's `pnl`/`result`. We walk settlement rows here.
#   - is_pnl_relevant_row() is the canonical filter for fill rows
#     (CLAUDE.md mandate); we explicitly handle settlement rows because
#     this consumer needs P&L outcomes, mirroring the
#     python/execution/shadow_broker.py:80-98 pattern.
# ---------------------------------------------------------------------------

INTRADAY_STRATEGY_NAME: str = "intraday_emos_high"
INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS: int = 7
# Spec §8.4(b): permanent-disable trigger fires when LAX/high consec-loss
# runs exceed 5 in any 7-day rolling window. Surfaced as ALERT in the
# daily report; auto-disable is intentionally OPERATOR-driven (see
# spec §8.4 LAX extra-discipline clause).
LAX_PERMANENT_DISABLE_RUN_THRESHOLD: int = 5
LAX_PERMANENT_DISABLE_CELL: str = "LAX/high"

# Bootstrap parameters — v6 audit canonical values
# (from an unpublished research script).
# Held identical here so the daily-report ``ask_ci_lo_7d`` lines up
# bit-for-bit with v6 BSS methodology calls (spec §11.4, §8.2).
INTRADAY_BOOTSTRAP_B: int = 2000
INTRADAY_BOOTSTRAP_SEED: int = 20260514

# §8.5 re-audit trigger thresholds (mid-run). Surfaced informationally
# on the daily report so the operator sees armed triggers BEFORE the
# auto-pause fires. Auto-pause itself is operator-driven today (spec
# §8.5 — telemetry-only path picked in v1.3).
INTRADAY_TRIGGER_PNL_FLOOR: float = -200.0  # §8.5(3)
INTRADAY_TRIGGER_WIN_RATE_FLOOR: float = 0.35  # §8.5(4)
INTRADAY_TRIGGER_WIN_RATE_MIN_N: int = 20  # §8.5(4) gate
INTRADAY_TRIGGER_CI_STRADDLE_MIN_N: int = 30  # §8.5 / spec §11.4 implicit
INTRADAY_TRIGGER_CONSEC_LOSS_MAX: int = 5  # §8.5(consec) + §8.4(b)
INTRADAY_TRIGGER_MU_SHIFT_F: float = 5.0  # §8.5(2)(a)

# Scan-funnel artifact name (sibling to phase0_depth_probes.jsonl). Used
# by the §9.2(b) trade-count-cap utilisation read. Missing or empty file
# is non-fatal — the helper returns 0 and the report renders cleanly.
SCAN_FUNNEL_FILENAME: str = "scan_funnel.jsonl"


def _intraday_cell_for_row(row: dict[str, Any]) -> str | None:
    """Derive ``"<city>/<var>"`` from a settlement row's ``market_ticker``.

    Per v1.3 MINOR #2: settlement rows in ``state/trades.jsonl`` do NOT
    carry ``metadata["cell"]`` — only fill rows do. This helper recovers
    the cell label via M3's :func:`parse_ticker_city_var`, which handles
    both daily and intraday Kalshi ticker formats. Returns ``None`` for
    unparseable tickers (defensive — the helper's callers skip such rows).
    """
    ticker = row.get("market_ticker")
    if not isinstance(ticker, str) or not ticker:
        return None
    city_var = parse_ticker_city_var(ticker)
    if city_var is None:
        return None
    city, var = city_var
    return f"{city}/{var}"


def _iter_intraday_settlement_rows(
    trade_log_path: Path,
) -> Iterator[dict[str, Any]]:
    """Yield intraday-strategy settlement rows from ``state/trades.jsonl``.

    Filters applied (in order, mirroring shadow_broker.py:80-98):

      1. Strategy attribution: ``strategy == INTRADAY_STRATEGY_NAME``.
      2. Event discriminator: ``event == TradeLogEvent.SETTLEMENT.value``.
         is_pnl_relevant_row is the centralised filter for FILL rows
         (CLAUDE.md mandate); settlement rows are the explicit complement
         a consumer needs when reading outcomes — same idiom used in
         python/execution/shadow_broker.py:80-98 (settlement branch
         first, then is_pnl_relevant_row gate on the remainder).

    The yield is the raw row — caller is responsible for parsing
    timestamp / pnl / ticker.
    """
    for row in iter_jsonl(trade_log_path):
        if row.get("strategy") != INTRADAY_STRATEGY_NAME:
            continue
        event = str(row.get("event", TradeLogEvent.FILL.value))
        # v1.1 defensive: explicit cash_out skip. The is_pnl_relevant_row
        # fallthrough below already excludes these (cash_out is not a fill),
        # but the explicit check forestalls future regressions if the
        # fallthrough is refactored away — and documents intent at the
        # consumer site (cash-out rows are forensic-only, P&L applied via
        # close_position / realize_partial at apply time per spec §6.4).
        if event in (
            TradeLogEvent.CASH_OUT.value,
            TradeLogEvent.CASH_OUT_REVERTED.value,
            TradeLogEvent.CASH_OUT_FORCE_APPLIED.value,
        ):
            continue
        if event != TradeLogEvent.SETTLEMENT.value:
            # is_pnl_relevant_row is exercised on the complementary path
            # via the cross-consumer contract test
            # (tests/test_trade_log_consumers.py); for the intraday
            # daily-report section we want SETTLEMENT rows only.
            if is_pnl_relevant_row(row):
                # Skip fill rows defensively — they don't carry the
                # win/loss outcome we need for consec-loss accounting.
                continue
            continue
        yield row


def _parse_iso_ts(ts: Any) -> datetime | None:
    """Parse a row's ``timestamp`` field to an aware ``datetime``."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # ``fromisoformat`` accepts the engine's "+00:00" suffix natively.
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _is_loss(row: dict[str, Any]) -> bool:
    """Win/loss convention: ``pnl <= 0`` is a loss.

    Mirrors the existing daily-report convention used by
    :func:`build_settled_report`'s side/result mapping; settlement rows
    carry ``pnl`` directly so we don't need to re-derive side semantics.
    Zero P&L (push) is conservatively treated as a loss — consistent
    with the §8.4(b) LAX-extra-discipline framing where any non-win
    contributes to the rolling-window run.
    """
    pnl_raw = row.get("pnl")
    try:
        pnl = float(pnl_raw)
    except (TypeError, ValueError):
        # Non-numeric P&L → treat as loss defensively (worst-case for
        # the operator: surfaces the row in run-length accounting
        # rather than silently dropping it).
        return True
    return pnl <= 0.0


def _bootstrap_ci_lo(
    values: list[float],
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> float | None:
    """Return the ``alpha/2`` quantile of the bootstrap sum distribution.

    Mirrors ``_bootstrap_ci_by_row`` in the v5 walk-forward audit (an
    unpublished research script) exactly (B=2000, seed=20260514 by default
    in callers, same random.Random(seed) + ``rng.randrange(n)`` choice
    semantics) so the v6 BSS methodology used for §8.2 / §8.4(a)
    shadow-promotion stays bit-comparable to this telemetry surface
    (spec §11.4). ``n_resamples`` carries the v6 ``B`` (uppercase in
    the audit script; renamed here for ruff N803 compliance).

    Empty input → ``None`` (absence is NOT zero — daily-report consumers
    must distinguish "no trades in window" from "trades summed to 0").
    """
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    sums: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        sums.append(s)
    sums.sort()
    lo_idx = int((alpha / 2.0) * n_resamples)
    return float(sums[lo_idx])


def _build_intraday_order_id_to_meta(
    trade_log_path: Path,
) -> dict[str, dict[str, Any]]:
    """Single-pass join helper: fill rows → ``order_id -> (side, emos_prob_raw, ...)``.

    Settlement rows in ``state/trades.jsonl`` do NOT carry the
    calibration metadata we need for §11.4 calibration-ratio surfacing
    (engine.py:_build_settlement_entry only writes ``result``/``pnl``
    on top of base fields). Fill rows DO carry ``metadata.emos_prob_raw``.
    Join the two by ``order_id`` so a settlement row can recover
    ``emos_prob_raw`` and ``side`` for p(loss) computation.

    Filters on ``strategy == "intraday_emos_high"`` so the map is sized
    by intraday fills only.

    Returns ``{}`` when the file is missing / empty; callers treat that
    as "no calibration data available" and render the ratio as ``None``.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(trade_log_path):
        if row.get("strategy") != INTRADAY_STRATEGY_NAME:
            continue
        if str(row.get("event")) != TradeLogEvent.FILL.value:
            continue
        order_id = row.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            continue
        meta = row.get("metadata") or {}
        emos = meta.get("emos_prob_raw")
        side = row.get("side")
        out[order_id] = {
            "emos_prob_raw": float(emos) if isinstance(emos, (int, float)) else None,
            "side": side if isinstance(side, str) else None,
        }
    return out


def _p_loss_from_emos(side: str | None, emos_prob_raw: float | None) -> float | None:
    """Return the model's predicted loss probability for one fill.

    ``emos_prob_raw`` is the calibrated p(YES) — i.e., p(bracket fires)
    per §6.3 sign convention. For ``side="NO"`` the trade loses when the
    bracket fires (p(loss) = emos_prob_raw); for ``side="YES"`` the trade
    loses when it does NOT fire (p(loss) = 1 - emos_prob_raw).

    Returns ``None`` if either input is missing — caller skips the row
    from the average instead of biasing it.
    """
    if emos_prob_raw is None or side not in {"YES", "NO"}:
        return None
    if side == "NO":
        return emos_prob_raw
    return 1.0 - emos_prob_raw


def _read_intraday_scan_funnel_cap_total(
    funnel_path: Path,
    *,
    lookback_days: int,
    now_utc: datetime,
) -> int:
    """Sum ``n_after_trades_per_cell_per_day_cap`` over the lookback.

    Reads ``state/scan_funnel.jsonl`` (M4 funnel artifact), filters to
    rows where ``strategy == "intraday_emos_high"``, and sums the cap
    counter. Missing / empty file → ``0`` (no crash); each scan emits
    one row, so absence is meaningful "no scans observed yet".

    Note on per-cell attribution: the ``ScanFunnel`` struct
    (python/analytics/audit_log.py:ScanFunnel) holds a single global
    integer for cap utilisation per scan; per-cell breakdown is NOT
    persisted today. The §11.4 spec frames the surface as "per cell per
    day" but the available data is global per-scan only. We surface the
    GLOBAL total here; the ALERT (utilisation > 0) is enough to flag a
    §9.2(b) ``_select_one_per_event`` grouping-key drift regardless of
    which cell triggered it. Adding per-cell attribution would require
    extending ``scan_funnel.jsonl``'s schema — out of scope for M6.
    """
    cutoff_ts = (now_utc - timedelta(days=lookback_days)).timestamp()
    total = 0
    for row in iter_jsonl(funnel_path):
        if row.get("strategy") != INTRADAY_STRATEGY_NAME:
            continue
        # scan_ts is emitted as a unix timestamp by
        # IntradayEmosHighStrategy._emit_funnel via the ``extra`` dict.
        ts_raw = row.get("scan_ts")
        if isinstance(ts_raw, (int, float)) and float(ts_raw) < cutoff_ts:
            continue
        cap = row.get("n_after_trades_per_cell_per_day_cap")
        if isinstance(cap, (int, float)):
            total += int(cap)
    return total


def compute_intraday_consec_loss_runs(
    trade_log_path: Path,
    *,
    lookback_days: int = INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS,
    now_utc: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-cell consecutive-loss-run state from ``trades.jsonl``.

    Path-(b) implementation (spec §9.2(c) v1.3): walks every
    intraday-strategy settlement row chronologically and emits, per cell:

      * ``current_run_len`` — length of the trailing run of losses
        (0 if the most recent settlement was a win).
      * ``max_run_len_7d`` — the longest run of consecutive losses whose
        end timestamp lies inside the lookback window (default 7 days).
      * ``last_run_end_ts`` — ISO timestamp of the last loss that
        contributed to a run within the window, or ``None`` if no run.

    Filters per spec §9.2(c):
      * ``strategy == "intraday_emos_high"`` (binding decision #5)
      * ``event == "settlement"`` (P&L is on settlement rows)
      * cell derived via :func:`parse_ticker_city_var`
        (v1.3 MINOR #2 — settlement rows have no ``metadata["cell"]``)

    Empty / missing file → ``{}`` (caller renders the placeholder).
    Rows with unparseable timestamps or tickers are skipped silently.
    """
    now = now_utc or datetime.now(UTC)
    cutoff = now - timedelta(days=lookback_days)

    # Two-pass: collect + sort to be robust against
    # out-of-order JSONL appends (the engine writes settlement rows
    # under an asyncio lock so order is normally chronological, but
    # operator-edited or merged logs may not be).
    rows: list[tuple[datetime, str, bool]] = []
    for row in _iter_intraday_settlement_rows(trade_log_path):
        ts = _parse_iso_ts(row.get("timestamp"))
        if ts is None:
            continue
        cell = _intraday_cell_for_row(row)
        if cell is None:
            continue
        rows.append((ts, cell, _is_loss(row)))

    if not rows:
        return {}

    rows.sort(key=lambda r: r[0])

    # Per-cell run tracking. Two parallel counters:
    #   ``current_run_len`` — the actual contiguous tail of losses
    #       (window-agnostic): the operator's view of "where do I stand
    #       right now". Reset only by a win.
    #   ``window_run_len`` (scratch) — same counter but reset whenever
    #       a row crosses INTO the window from outside it (any pre-cutoff
    #       loss seeds the next in-window run at zero, NOT inflated by
    #       stale losses). This is the source of ``max_run_len_7d`` and
    #       drives the §8.4(b) "ANY 7-day rolling window" semantic.
    per_cell: dict[str, dict[str, Any]] = {}
    for ts, cell, is_loss in rows:
        st = per_cell.setdefault(
            cell,
            {
                "current_run_len": 0,
                "max_run_len_7d": 0,
                "last_run_end_ts": None,
                # Internal scratch — popped before return.
                "_window_run_len": 0,
            },
        )
        if is_loss:
            st["current_run_len"] = int(st["current_run_len"]) + 1
            if ts >= cutoff:
                st["_window_run_len"] = int(st["_window_run_len"]) + 1
                if st["_window_run_len"] > st["max_run_len_7d"]:
                    st["max_run_len_7d"] = int(st["_window_run_len"])
                    st["last_run_end_ts"] = ts.isoformat()
            else:
                # Pre-cutoff losses do NOT seed the in-window run.
                st["_window_run_len"] = 0
        else:
            # Win breaks both counters.
            st["current_run_len"] = 0
            st["_window_run_len"] = 0

    # Strip scratch fields.
    for st in per_cell.values():
        st.pop("_window_run_len", None)

    return per_cell


@dataclass
class IntradayCellStats:
    """Per-cell aggregate stats for the intraday daily-report section."""

    cell: str
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    # ``pnl_ask`` is the realized P&L computed against the ask fill (the
    # production fill price). Settlement rows (engine.py:_build_settlement_entry)
    # carry only a single ``pnl`` field — no ask/mid split —
    # so we route the on-disk ``pnl`` to ``pnl_ask`` and leave ``pnl_mid``
    # unpopulated. Mid-anchored P&L is a settlement-schema gap: it would
    # need ``mid_at_signal`` carried forward from the fill row through to
    # settlement to be computed honestly. Plumbing that through the engine
    # is out of scope for M6 — see spec §11.4 + this code-comment.
    pnl_ask: float = 0.0
    # Per-trade ``pnl_ask`` series for the bootstrap CI lo (fix #2).
    # NOT exposed on the dict envelope — bootstrap is computed in
    # ``summarize_intraday_strategy``.
    pnl_ask_series: list[float] = field(default_factory=list)
    # Per-fill p(loss) series for the calibration-ratio surface (fix #3).
    # Each entry is ``emos_prob_raw`` (for NO trades) or
    # ``1 - emos_prob_raw`` (for YES trades), per §6.3 sign convention.
    # Missing-meta fills drop out of the average — see
    # ``_p_loss_from_emos``.
    p_loss_series: list[float] = field(default_factory=list)
    current_run_len: int = 0
    max_run_len_7d: int = 0
    last_run_end_ts: str | None = None

    @property
    def win_rate(self) -> float | None:
        if self.n_trades == 0:
            return None
        return self.wins / self.n_trades

    @property
    def pnl_mid(self) -> float | None:
        """Mid-anchored P&L — settlement schema does not persist it.

        Returns ``None`` rather than ``0.0`` so consumers can distinguish
        "not measured" from "measured to zero". See ``pnl_ask`` doc.
        """
        return None

    @property
    def actual_loss_rate(self) -> float | None:
        if self.n_trades == 0:
            return None
        return self.losses / self.n_trades

    @property
    def avg_model_p_loss(self) -> float | None:
        """Average model-predicted p(loss) across settled fills.

        Returns ``None`` when no fill rows carried ``emos_prob_raw``
        (early-shadow days, or fills lacking the metadata field).
        """
        if not self.p_loss_series:
            return None
        return sum(self.p_loss_series) / len(self.p_loss_series)

    @property
    def calibration_ratio(self) -> float | None:
        """Spec §11.4: ``avg model p(loss) / actual loss rate``.

        Reads naturally as "ratio > 1 → model over-predicts losses
        (under-confident)"; "ratio < 1 → model under-predicts losses
        (over-confident)". Returns ``None`` when either average is
        unavailable or actual loss rate is 0 (no losses → undefined).
        """
        avg = self.avg_model_p_loss
        actual = self.actual_loss_rate
        if avg is None or actual is None or actual == 0.0:
            return None
        return avg / actual


def _evaluate_intraday_triggers(
    cell: str,
    st: IntradayCellStats,
    *,
    ask_ci_lo_7d: float | None,
) -> list[dict[str, Any]]:
    """Evaluate §8.5 mid-run re-audit triggers for one cell.

    Returns a list of trigger dicts:

      ``{"name": str, "current": Any, "threshold": Any, "fired": bool,
         "deferred": bool, "note": str | None}``

    Each trigger is informational on the daily report — auto-pause is
    operator-driven in v1.3 (spec §8.5, telemetry-only path). The
    μ-shift trigger (§8.5(2)(a)) is intentionally deferred: it requires
    a coefficients-history pipeline (delta vs the v6-walk-forward audit
    snapshot) that doesn't ship in M6. Recorded as a TODO row so the
    operator sees the trigger exists and knows what's pending. Honest
    deferral, NOT silent skip. See §8.5(2)(a) for the rationale and
    ``data/research/workstream1_gate_audit_v6_walkforward_coeffs.json``
    for the eventual data source.
    """
    triggers: list[dict[str, Any]] = []

    # §8.5(2)(a) — μ-shift > 5.0°F vs v6-walk-forward reference.
    # Deferred: needs an EMOS coefficient-history reader and a per-cell
    # delta computation. See spec §8.5(2)(a) and
    # data/research/workstream1_gate_audit_v6_walkforward_coeffs.json.
    triggers.append(
        {
            "name": "mu_shift_gt_5F",
            "current": None,
            "threshold": INTRADAY_TRIGGER_MU_SHIFT_F,
            "fired": False,
            "deferred": True,
            "note": (
                "TODO: needs EMOS coefficients-history reader; spec §8.5(2)(a). "
                "Source: data/research/workstream1_gate_audit_v6_walkforward_coeffs.json."
            ),
        }
    )

    # §8.5(3) — rolling 7-day P&L < -$200 on the cell.
    triggers.append(
        {
            "name": "pnl_7d_below_floor",
            "current": round(st.pnl_ask, 4),
            "threshold": INTRADAY_TRIGGER_PNL_FLOOR,
            "fired": st.pnl_ask < INTRADAY_TRIGGER_PNL_FLOOR,
            "deferred": False,
            "note": None,
        }
    )

    # §8.5(4) — rolling 7-day win rate < 35% with n ≥ 20 gate. Below the
    # n gate the trigger is intentionally NOT armed (spec §8.5(4) — too
    # noisy to fire on n<20).
    win_rate = st.win_rate
    win_rate_fired = (
        win_rate is not None
        and st.n_trades >= INTRADAY_TRIGGER_WIN_RATE_MIN_N
        and win_rate < INTRADAY_TRIGGER_WIN_RATE_FLOOR
    )
    triggers.append(
        {
            "name": "win_rate_7d_below_floor",
            "current": (round(win_rate, 4) if win_rate is not None else None),
            "threshold": INTRADAY_TRIGGER_WIN_RATE_FLOOR,
            "fired": win_rate_fired,
            "deferred": False,
            "note": (f"gated on n>={INTRADAY_TRIGGER_WIN_RATE_MIN_N} (observed n={st.n_trades})"),
        }
    )

    # §8.5 + §11.4 implicit — n ≥ 30 AND bootstrap CI lo straddles 0.
    # "Straddles 0" here = CI lo ≤ 0 (lower bound has not cleared zero
    # despite the sample size milestone). The CI ascends from
    # ``_bootstrap_ci_lo`` which is the 2.5%-quantile of the resampled
    # SUM; "straddles 0" is the v6-methodology pre-registered binding
    # criterion (see audit_same_day_gate_2026-05-14_v5_walkforward.py
    # decision rule).
    ci_fired = (
        st.n_trades >= INTRADAY_TRIGGER_CI_STRADDLE_MIN_N
        and ask_ci_lo_7d is not None
        and ask_ci_lo_7d <= 0.0
    )
    triggers.append(
        {
            "name": "ci_straddles_zero_at_n30",
            "current": (round(ask_ci_lo_7d, 4) if ask_ci_lo_7d is not None else None),
            "threshold": 0.0,
            "fired": ci_fired,
            "deferred": False,
            "note": (
                f"gated on n>={INTRADAY_TRIGGER_CI_STRADDLE_MIN_N} (observed n={st.n_trades})"
            ),
        }
    )

    # §8.5(consec) / §8.4(b) — consec-loss > 5. ``current_run_len`` is
    # set on ``IntradayCellStats`` by ``summarize_intraday_strategy``
    # from ``compute_intraday_consec_loss_runs``.
    triggers.append(
        {
            "name": "consec_loss_gt_threshold",
            "current": st.current_run_len,
            "threshold": INTRADAY_TRIGGER_CONSEC_LOSS_MAX,
            "fired": st.current_run_len > INTRADAY_TRIGGER_CONSEC_LOSS_MAX,
            "deferred": False,
            "note": None,
        }
    )

    # Cell-name embedded for downstream JSON consumers without needing
    # the parent map key.
    for t in triggers:
        t["cell"] = cell
    return triggers


def summarize_intraday_strategy(
    trade_log_path: Path,
    *,
    lookback_days: int = INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS,
    now_utc: datetime | None = None,
    scan_funnel_path: Path | None = None,
) -> dict[str, Any]:
    """Per-cell daily-report summary for the intraday EMOS HIGH strategy.

    Aggregates settlement rows in the lookback window into per-cell
    counts + P&L (Fix #1: ``pnl_ask`` / ``pnl_mid``), bootstrap CI lo
    (Fix #2), calibration ratio (Fix #3), §8.5 trigger flags (Fix #4),
    and §9.2(b) trade-count-cap utilisation (Fix #5). Consec-loss-run
    state comes from :func:`compute_intraday_consec_loss_runs`.

    The lookback bounds BOTH the aggregates AND the consec-loss-run
    window: §11.4 frames the section as "rolling 7-day" telemetry, so
    aggregating the entire log here would be inconsistent with that
    framing and would surface stale runs as "current".

    Returns ``{"lookback_days", "cells", "triggers", "cap_utilisation"}``.
    The ``cells`` map is empty when no in-window intraday settlement rows
    exist; callers render a single-line placeholder in that case.
    """
    now = now_utc or datetime.now(UTC)
    cutoff = now - timedelta(days=lookback_days)

    runs = compute_intraday_consec_loss_runs(
        trade_log_path, lookback_days=lookback_days, now_utc=now
    )

    # Fix #3 — single-pass fill-row map for emos_prob_raw join. Settlement
    # rows don't carry metadata (engine.py:_build_settlement_entry); fill
    # rows do. Join is by ``order_id`` (1:1 invariant per the dedup gate
    # documented in CLAUDE.md).
    fill_meta = _build_intraday_order_id_to_meta(trade_log_path)

    per_cell: dict[str, IntradayCellStats] = {}
    for row in _iter_intraday_settlement_rows(trade_log_path):
        ts = _parse_iso_ts(row.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        cell = _intraday_cell_for_row(row)
        if cell is None:
            continue
        st = per_cell.setdefault(cell, IntradayCellStats(cell=cell))
        st.n_trades += 1
        try:
            pnl = float(row.get("pnl") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        # Fix #1: settlement schema persists a single ``pnl`` value (no
        # ask/mid split). Route to ``pnl_ask``; ``pnl_mid`` stays None
        # (schema gap documented on the dataclass).
        st.pnl_ask += pnl
        st.pnl_ask_series.append(pnl)
        if pnl > 0.0:
            st.wins += 1
        else:
            st.losses += 1

        # Fix #3: per-fill p(loss). The settlement row's side comes from
        # the engine's base_log_fields (engine.py:_base_log_fields); the
        # ``emos_prob_raw`` comes from the joined fill row (settlement
        # rows don't carry it). For rows where the fill is missing (e.g.,
        # a reconciler-backfilled synthetic settlement), fall back to the
        # settlement row's own ``side`` and skip the emos lookup — that
        # row contributes nothing to the average.
        order_id = row.get("order_id") if isinstance(row.get("order_id"), str) else None
        meta = fill_meta.get(order_id) if order_id else None
        if meta is not None:
            side = meta.get("side") or row.get("side")
            p_loss = _p_loss_from_emos(side, meta.get("emos_prob_raw"))
            if p_loss is not None:
                st.p_loss_series.append(p_loss)

    for cell, st in per_cell.items():
        run_state = runs.get(cell, {})
        st.current_run_len = int(run_state.get("current_run_len", 0))
        st.max_run_len_7d = int(run_state.get("max_run_len_7d", 0))
        st.last_run_end_ts = run_state.get("last_run_end_ts")

    # Fix #5 — read scan_funnel.jsonl and sum the cap-counter over the
    # lookback window. Single global total because the funnel schema is
    # not per-cell (documented in ``_read_intraday_scan_funnel_cap_total``).
    cap_total = 0
    if scan_funnel_path is not None:
        cap_total = _read_intraday_scan_funnel_cap_total(
            scan_funnel_path, lookback_days=lookback_days, now_utc=now
        )

    cells_out: dict[str, dict[str, Any]] = {}
    triggers_out: dict[str, list[dict[str, Any]]] = {}
    for cell, st in per_cell.items():
        # Fix #2 — bootstrap ask CI lo per cell.
        ask_ci_lo = _bootstrap_ci_lo(
            st.pnl_ask_series,
            n_resamples=INTRADAY_BOOTSTRAP_B,
            seed=INTRADAY_BOOTSTRAP_SEED,
        )
        cells_out[cell] = {
            "n_trades": st.n_trades,
            "wins": st.wins,
            "losses": st.losses,
            "win_rate": st.win_rate,
            # Fix #1: legacy ``pnl`` key retained for backward compat with
            # existing consumers; ``pnl_ask`` is the canonical split.
            "pnl": round(st.pnl_ask, 4),
            "pnl_ask": round(st.pnl_ask, 4),
            # Fix #1: settlement-schema gap — see IntradayCellStats.pnl_mid.
            "pnl_mid": st.pnl_mid,
            # Fix #2.
            "ask_ci_lo_7d": (round(ask_ci_lo, 4) if ask_ci_lo is not None else None),
            # Fix #3.
            "avg_model_p_loss": (
                round(st.avg_model_p_loss, 4) if st.avg_model_p_loss is not None else None
            ),
            "actual_loss_rate": (
                round(st.actual_loss_rate, 4) if st.actual_loss_rate is not None else None
            ),
            "calibration_ratio": (
                round(st.calibration_ratio, 4) if st.calibration_ratio is not None else None
            ),
            "current_run_len": st.current_run_len,
            "max_run_len_7d": st.max_run_len_7d,
            "last_run_end_ts": st.last_run_end_ts,
        }
        # Fix #4 — per-cell trigger flags. Stored separately from the
        # ``cells`` dict so the JSON envelope can group triggers by cell
        # without inflating the cells map's keyspace.
        triggers_out[cell] = _evaluate_intraday_triggers(cell, st, ask_ci_lo_7d=ask_ci_lo)

    return {
        "lookback_days": lookback_days,
        "cells": cells_out,
        # Fix #4.
        "triggers": triggers_out,
        # Fix #5 — global cap-utilisation total (funnel schema is not
        # per-cell; see ``_read_intraday_scan_funnel_cap_total`` docstring).
        "trades_per_cell_cap_utilisation_7d": cap_total,
    }


def build_intraday_json(summary: dict[str, Any]) -> dict[str, Any]:
    """Render the intraday summary into the JSON-export envelope.

    Parallel to ``phase0_depth_probing``: a single top-level key with a
    ``cells`` map plus the §11.4 add-ons (triggers + cap utilisation
    from M6 fixes #4 and #5). Caller is :func:`build_json_output`.
    """
    return {
        "lookback_days": int(summary.get("lookback_days", INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS)),
        "cells": dict(summary.get("cells", {}) or {}),
        "triggers": dict(summary.get("triggers", {}) or {}),
        "trades_per_cell_cap_utilisation_7d": int(
            summary.get("trades_per_cell_cap_utilisation_7d") or 0
        ),
    }


def _lax_permanent_disable_armed(summary: dict[str, Any]) -> bool:
    """True iff §8.4(b) LAX permanent-disable trigger fires.

    Reads ``max_run_len_7d`` on ``LAX/high`` (case-sensitive — matches
    the cell label produced by :func:`_intraday_cell_for_row`) and
    compares against :data:`LAX_PERMANENT_DISABLE_RUN_THRESHOLD`.
    """
    cells = summary.get("cells", {}) or {}
    lax = cells.get(LAX_PERMANENT_DISABLE_CELL)
    if not isinstance(lax, dict):
        return False
    try:
        max_run = int(lax.get("max_run_len_7d", 0))
    except (TypeError, ValueError):
        return False
    return max_run > LAX_PERMANENT_DISABLE_RUN_THRESHOLD


def _format_intraday_section(summary: dict[str, Any]) -> list[str]:
    """Render the intraday-strategy section as report lines.

    Always emits a header so the report reader sees the section. The
    body collapses to ``(no intraday trades in last Nd)`` when no
    cells are populated. Spec §11.4 + §8.4(b).
    """
    lookback = int(summary.get("lookback_days", INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS))
    lines: list[str] = ["", "=== INTRADAY EMOS HIGH ==="]
    cells = summary.get("cells", {}) or {}
    cap_total = int(summary.get("trades_per_cell_cap_utilisation_7d") or 0)
    if not cells:
        lines.append(f"  (no intraday trades in last {lookback}d)")
        # Fix #5 — even with no settled trades, surface scan-funnel
        # cap-utilisation if the funnel observed cap hits. Defense-in-depth
        # for §9.2(b) ``_select_one_per_event`` grouping-key drift.
        if cap_total > 0:
            lines.append(
                f"  [!!] trades-per-cell-cap hit {cap_total} time(s) in "
                f"last {lookback}d — possible event-key drift — investigate."
            )
        return lines

    lines.append(f"  Lookback: {lookback}d  |  Strategy: {INTRADAY_STRATEGY_NAME}")
    # Fix #1: add ask P&L / mid P&L columns explicitly.
    # Fix #2: add 7d ask CI lo column.
    # Fix #3: add calibration-ratio column.
    lines.append(
        f"  {'Cell':<10} {'n':>4} {'W':>3} {'L':>3} {'win%':>6} "
        f"{'pnl_ask':>8} {'pnl_mid':>8} {'CI_lo7d':>9} {'cal_r':>6} "
        f"{'run_now':>8} {'run_max7d':>10}  last_loss_ts"
    )
    lines.append("  " + "-" * 110)

    for cell_name in sorted(cells.keys()):
        c = cells[cell_name]
        n = int(c.get("n_trades", 0))
        w = int(c.get("wins", 0))
        loss = int(c.get("losses", 0))
        wr = c.get("win_rate")
        wr_str = f"{wr * 100:>5.1f}%" if isinstance(wr, (int, float)) else "    -"
        # Fix #1 columns.
        pnl_ask = c.get("pnl_ask")
        pnl_ask_str = f"{pnl_ask:>+8.2f}" if isinstance(pnl_ask, (int, float)) else f"{'-':>8}"
        pnl_mid = c.get("pnl_mid")
        pnl_mid_str = f"{pnl_mid:>+8.2f}" if isinstance(pnl_mid, (int, float)) else f"{'-':>8}"
        # Fix #2 column.
        ci_lo = c.get("ask_ci_lo_7d")
        ci_str = f"{ci_lo:>+9.2f}" if isinstance(ci_lo, (int, float)) else f"{'-':>9}"
        # Fix #3 column.
        cal_r = c.get("calibration_ratio")
        cal_r_str = f"{cal_r:>5.2f}x" if isinstance(cal_r, (int, float)) else f"{'-':>6}"
        cur = int(c.get("current_run_len", 0))
        mx = int(c.get("max_run_len_7d", 0))
        last_ts = c.get("last_run_end_ts") or "-"
        lines.append(
            f"  {cell_name:<10} {n:>4d} {w:>3d} {loss:>3d} {wr_str:>6} "
            f"{pnl_ask_str} {pnl_mid_str} {ci_str} {cal_r_str} "
            f"{cur:>8d} {mx:>10d}  {last_ts}"
        )

    # Fix #4 — §8.5 trigger flags armed today, informational. Auto-pause
    # is operator-driven (spec §8.5; telemetry-only path picked in v1.3).
    triggers = summary.get("triggers", {}) or {}
    if triggers:
        lines.append("")
        lines.append("  §8.5 triggers armed today (informational; auto-pause is operator-driven):")
        lines.append(
            f"    {'Cell':<10} {'Trigger':<28} {'current':>10} {'threshold':>10} {'fired':>6}  note"
        )
        lines.append("    " + "-" * 88)
        for cell_name in sorted(triggers.keys()):
            for t in triggers[cell_name]:
                current = t.get("current")
                threshold = t.get("threshold")
                current_str = (
                    f"{current:>10.3g}"
                    if isinstance(current, (int, float))
                    else (f"{'TODO':>10}" if t.get("deferred") else f"{'-':>10}")
                )
                threshold_str = (
                    f"{threshold:>10.3g}" if isinstance(threshold, (int, float)) else f"{'-':>10}"
                )
                fired = bool(t.get("fired"))
                fired_str = "Y" if fired else ("d" if t.get("deferred") else "n")
                note = t.get("note") or ""
                lines.append(
                    f"    {cell_name:<10} {t.get('name', ''):<28} "
                    f"{current_str} {threshold_str} {fired_str:>6}  {note}"
                )

    # Fix #5 — §9.2(b) trade-count-cap utilisation surface. The funnel
    # field is global per scan (not per-cell), so we render one summary
    # line plus an ALERT when > 0 — that's the structural-drift signal
    # the §9.2(b) MINOR #2 honest framing names.
    lines.append("")
    lines.append(
        f"  §9.2(b) trades-per-cell-cap utilisation (last {lookback}d, "
        f"intraday strategy total): {cap_total}"
    )
    if cap_total > 0:
        lines.append(
            f"  [!!] trades-per-cell-cap hit {cap_total} time(s) — "
            f"possible event-key drift in _select_one_per_event — investigate."
        )

    # §8.4(b) LAX permanent-disable ALERT row — fires off the daily
    # report itself, not via Telegram (M6 is silent on alerts per
    # spec_alert_path_verification_rule.md).
    if _lax_permanent_disable_armed(summary):
        lines.append("")
        lines.append(
            f"  ** ALERT: {LAX_PERMANENT_DISABLE_CELL} permanent-disable trigger armed — "
            f"max_run_len_7d > {LAX_PERMANENT_DISABLE_RUN_THRESHOLD} per spec §8.4(b)."
        )
        lines.append(
            "         Remove 'LAX/high' from strategies.intraday_emos_high.enabled_cells "
            "in python/config.yaml and redeploy."
        )
    return lines


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    settled_days: list[DailyPnL],
    pending: list[PendingRisk],
    calibration: dict[str, Any],
    trade_date: str | None,
    *,
    phase0_summary: dict[str, Any] | None = None,
    intraday_summary: dict[str, Any] | None = None,
    cap_blocks_by_cell: dict[tuple[str, str], int] | None = None,
    strategy_pnl: dict[str, dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []

    header_date = trade_date or "ALL DATES"
    lines += [
        "=" * 72,
        f"  ORACLE DAILY TRADE REPORT  —  {header_date}",
        "=" * 72,
    ]

    # Settled P&L
    if settled_days:
        total_pnl = sum(d.total_pnl for d in settled_days)
        total_w = sum(len(d.wins) for d in settled_days)
        total_l = sum(len(d.losses) for d in settled_days)

        lines += [
            "",
            "=== SETTLED P&L ===",
            f"Overall: {total_w}W / {total_l}L  |  P&L: ${total_pnl:+.2f}",
            "",
        ]
        for day in settled_days:
            lines.append(
                f"  {day.trade_date}: {len(day.wins)}W/{len(day.losses)}L  "
                f"wagered=${day.total_wagered:.2f}  P&L=${day.total_pnl:+.2f}"
            )
            for t in day.losses:
                floor_str = (
                    f"{t.floor_strike:.0f}-{t.cap_strike:.0f}°"
                    if t.floor_strike is not None and t.cap_strike is not None
                    else "?"
                )
                raw_str = (
                    f"  raw={t.raw_ensemble_prob:.3f}" if t.raw_ensemble_prob is not None else ""
                )
                lines.append(
                    f"    LOSS  {t.city} {t.var} {floor_str} (for {t.forecast_date})  "
                    f"model={t.model_p:.1%}{raw_str}  P&L=${t.pnl:+.2f}"
                )
    else:
        lines += ["", "=== SETTLED P&L ===", "  (no settled trades for this date/mode)"]

    # Per-strategy settled P&L (generic — automatic for every strategy).
    if strategy_pnl is not None:
        lines += _format_strategy_pnl_section(strategy_pnl)

    # Pending scorecard
    if pending:
        lines += [
            "",
            "=== PENDING TRADE RISK SCORECARD ===",
            f"{'Trade':<27} {'model':>6} {'fill':>5} {'$':>6} "
            f"{'GFS%':>6} {'ECM%':>6} {'GFS_inf':>7} {'factor':>7}  Risk",
            "-" * 80,
        ]
        for p in pending:
            strike_str = (
                f"{p.floor_strike:.0f}-{p.cap_strike:.0f}°"
                if p.floor_strike is not None and p.cap_strike is not None
                else "?"
            )
            tag = f"{p.city} {p.var} {strike_str}"
            lines.append(
                f"{tag:<27} {p.model_p:>6.1%} {p.fill_no:>5.2f} {p.dollars:>6.2f} "
                f"{p.gfs_raw_frac:>6.1%} {p.ecmwf_raw_frac:>6.1%} "
                f"{p.gfs_inflated_frac:>7.1%} {p.inflation_factor:>6.1f}x  {p.risk_level}"
            )
        total_pending = sum(p.dollars for p in pending)
        lines.append(f"\n  Total pending: ${total_pending:.2f}")
    else:
        lines += ["", "=== PENDING TRADE RISK SCORECARD ===", "  (no pending trades)"]

    # Calibration gap
    if calibration:
        ratio = calibration.get("calibration_ratio")
        ratio_str = f"{ratio:.1f}x" if ratio is not None else "N/A"
        cal_mode = calibration.get("mode", "all")
        if cal_mode == "all":
            cal_header = "=== CALIBRATION GAP (all settled dates, all modes) ==="
        else:
            cal_header = f"=== CALIBRATION GAP (all settled dates, mode={cal_mode}) ==="
        n_unattributed = calibration.get("n_unattributed", 0)
        unattributed_note = f" ({n_unattributed} unattributed excluded)" if n_unattributed else ""
        win_rate = calibration.get("win_rate")
        avg_loss_p = calibration.get("avg_model_loss_p")
        actual_loss = calibration.get("actual_loss_rate")
        win_rate_str = f"{win_rate:.1%}" if win_rate is not None else "N/A"
        avg_loss_str = f"{avg_loss_p:.1%}" if avg_loss_p is not None else "N/A"
        actual_loss_str = f"{actual_loss:.1%}" if actual_loss is not None else "N/A"
        lines += [
            "",
            cal_header,
            f"  Settled: {calibration['n_settled']} trades{unattributed_note}  "
            f"{calibration['n_wins']}W / {calibration['n_losses']}L  "
            f"win rate: {win_rate_str}",
            f"  Model avg p(loss): {avg_loss_str}  |  "
            f"Actual loss rate: {actual_loss_str}  |  "
            f"Ratio: {ratio_str}",
        ]
    else:
        lines += ["", "=== CALIBRATION GAP ===", "  (no settled data)"]

    # Phase 0 passive depth-probing summary (intraday EMOS HIGH spec §6.4).
    # Always emitted: the empty-sink path renders a single "(no Phase 0
    # samples...)" line so the report reader sees the section exists.
    if phase0_summary is not None:
        lines += _format_phase0_section(phase0_summary)

    # Intraday EMOS HIGH per-cell section (spec §11.4). Path-(b) consec-loss
    # runs + §8.4(b) LAX permanent-disable ALERT live in here.
    if intraday_summary is not None:
        lines += _format_intraday_section(intraday_summary)

    # Per-day cap block counts (only emitted when --include-cap-blocks is set).
    if cap_blocks_by_cell is not None:
        lines += ["", "=== PER-DAY CAP BLOCKS (by forecast date) ==="]
        if cap_blocks_by_cell:
            for (city, var), count in sorted(cap_blocks_by_cell.items()):
                lines.append(f"  {city}/{var}: {count} blocked")
        else:
            lines.append("  (no cap-block rows for this date)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-day cap block counts
# ---------------------------------------------------------------------------


def load_cap_blocks_by_cell(
    cap_blocks_path: Path,
    report_date: str,
) -> dict[tuple[str, str], int]:
    """Return per-cell block counts for a specific forecast date.

    Reads ``state/per_day_cap_blocks.jsonl`` and counts rows whose
    ``event_key.forecast_date`` equals *report_date*. Groups by
    ``(city, var)`` — the same key used throughout the daily report.
    Filters by ``forecast_date``, NOT by the scan timestamp, so operators
    see which cells had cap-limited opportunity on the report's settlement
    day (not which scans ran that day).

    Tolerant of a missing file (returns empty dict) and corrupt JSONL
    lines (skips silently).
    """
    if not cap_blocks_path.exists():
        return {}
    counts: dict[tuple[str, str], int] = {}
    for row in iter_jsonl(cap_blocks_path):
        ek = row.get("event_key")
        if not isinstance(ek, dict):
            continue
        fdate = ek.get("forecast_date")
        if fdate != report_date:
            continue
        city = ek.get("city")
        var = ek.get("var")
        if not city or not var:
            continue
        key = (str(city), str(var))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def _settled_trade_json(t: SettledTrade) -> dict[str, Any]:
    """Serialize one settled trade for the JSON ``settled`` block.

    Shared by the ``wins`` and ``losses`` arrays so both sides stay in lockstep.
    """
    return {
        "ticker": t.ticker,
        "city": t.city,
        "var": t.var,
        "floor_strike": t.floor_strike,
        "cap_strike": t.cap_strike,
        "forecast_date": t.forecast_date,
        "model_p": t.model_p,
        "raw_ensemble_prob": t.raw_ensemble_prob,
        "fill_no": t.fill_no,
        "contracts": t.contracts,
        "dollars": t.dollars,
        "pnl": t.pnl,
    }


def build_json_output(
    trade_date: str | None,
    settled_days: list[DailyPnL],
    pending: list[PendingRisk],
    calibration: dict[str, Any],
    *,
    phase0_summary: dict[str, Any] | None = None,
    intraday_summary: dict[str, Any] | None = None,
    cap_blocks_by_cell: dict[tuple[str, str], int] | None = None,
    strategy_pnl: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "by_strategy": strategy_pnl,
        "phase0_depth_probing": phase0_summary,
        "intraday_emos_high": (
            build_intraday_json(intraday_summary) if intraday_summary is not None else None
        ),
        "per_cell": {
            f"{city}/{var}": {"per_day_cap_blocks": count}
            for (city, var), count in (cap_blocks_by_cell or {}).items()
        },
        "settled_per_cell": aggregate_per_cell(settled_days),
        "settled": [
            {
                "trade_date": day.trade_date,
                "n_wins": len(day.wins),
                "n_losses": len(day.losses),
                "total_pnl": round(day.total_pnl, 4),
                "total_wagered": round(day.total_wagered, 4),
                # Both sides are emitted so downstream consumers (e.g. the
                # post-mortem skill) can build full per-trade walkthroughs from
                # this file alone instead of re-deriving wins from the raw
                # settlements corpus.
                "wins": [_settled_trade_json(t) for t in day.wins],
                "losses": [_settled_trade_json(t) for t in day.losses],
            }
            for day in settled_days
        ],
        "pending": [
            {
                "ticker": p.ticker,
                "city": p.city,
                "var": p.var,
                "floor_strike": p.floor_strike,
                "cap_strike": p.cap_strike,
                "forecast_date": p.forecast_date,
                "model_p": p.model_p,
                "fill_no": p.fill_no,
                "contracts": p.contracts,
                "dollars": p.dollars,
                "gfs_raw_frac": round(p.gfs_raw_frac, 4),
                "ecmwf_raw_frac": round(p.ecmwf_raw_frac, 4),
                "gfs_inflated_frac": round(p.gfs_inflated_frac, 4),
                "ecmwf_inflated_frac": round(p.ecmwf_inflated_frac, 4),
                "gfs_mean": round(p.gfs_mean, 2),
                "ecmwf_mean": round(p.ecmwf_mean, 2),
                "inflation_factor": p.inflation_factor,
                "risk_level": p.risk_level,
            }
            for p in pending
        ],
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SETTLEMENTS = _REPO_ROOT / "data" / "shadow_settlements.full.json"
_DEFAULT_SNAPSHOTS = _REPO_ROOT / "data" / "snapshots"
_DEFAULT_PHASE0_SAMPLES = _REPO_ROOT / "state" / PHASE0_DEPTH_SINK_FILENAME
# Intraday consec-loss helper reads state/trades.jsonl (monolithic; see
# OQ-3). Operators run --pull-vps to refresh this file from the VPS.
_DEFAULT_TRADE_LOG = _REPO_ROOT / "state" / "trades.jsonl"
# Per-day cap audit stream — written by PerDayCapBlockWriter on every blocked
# candidate; read by the daily report for falsification-criterion monitoring.
_DEFAULT_CAP_BLOCKS = _REPO_ROOT / "state" / "per_day_cap_blocks.jsonl"
# Fix #5 — scan_funnel.jsonl is the M4 intraday-strategy funnel artifact;
# daily-report consumes it for the §9.2(b) cap-utilisation surface.
_DEFAULT_SCAN_FUNNEL = _REPO_ROOT / "state" / SCAN_FUNNEL_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle daily trade report: settled P&L, pending risk, calibration gap."
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=date.today().isoformat(),
        help="Trade date to report on (default: today)",
    )
    parser.add_argument(
        "--pull-vps",
        action="store_true",
        help="SCP latest trades.jsonl from VPS before reporting",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch settlements from Kalshi API",
    )
    parser.add_argument(
        "--boundary-since",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Enablement date for the boundary-distance filter kill watch (spec "
            "§6.5). Scopes the per-cell ROI CI to trades on/after this date so "
            "pre-enablement grazer trades don't contaminate the signal."
        ),
    )
    parser.add_argument(
        "--mode",
        default="live",
        choices=["live", "shadow", "paper", "all"],
        help="Execution mode filter (default: live)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help=(
            "Scope every section to a single strategy (e.g. weather_edge, "
            "favorite_bias). Default None: the weather sections (SETTLED P&L, "
            "settled_per_cell, pending scorecard, calibration gap) use the "
            "weather allowlist and the per-strategy table shows all strategies. "
            "NOTE: this scopes the settlements-file sections only — the "
            "dedicated INTRADAY EMOS HIGH section reads the trade log directly "
            "and is not affected. Missing-strategy rows map to 'unknown'."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write JSON report to this file path",
    )
    parser.add_argument(
        "--settlements",
        default=str(_DEFAULT_SETTLEMENTS),
        help=f"Path to shadow_settlements.full.json (default: {_DEFAULT_SETTLEMENTS})",
    )
    parser.add_argument(
        "--snapshots",
        default=str(_DEFAULT_SNAPSHOTS),
        help=f"Path to snapshots directory (default: {_DEFAULT_SNAPSHOTS})",
    )
    parser.add_argument(
        "--phase0-samples",
        default=str(_DEFAULT_PHASE0_SAMPLES),
        help=(f"Path to Phase 0 depth-probe JSONL sink (default: {_DEFAULT_PHASE0_SAMPLES})"),
    )
    parser.add_argument(
        "--phase0-lookback-days",
        type=int,
        default=PHASE0_DEFAULT_LOOKBACK_DAYS,
        help=(f"Phase 0 depth summary lookback in days (default: {PHASE0_DEFAULT_LOOKBACK_DAYS})"),
    )
    parser.add_argument(
        "--trade-log",
        default=str(_DEFAULT_TRADE_LOG),
        help=(
            f"Path to state/trades.jsonl for the intraday EMOS HIGH section "
            f"(default: {_DEFAULT_TRADE_LOG})"
        ),
    )
    parser.add_argument(
        "--intraday-lookback-days",
        type=int,
        default=INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS,
        help=(
            f"Intraday EMOS HIGH consec-loss / aggregate window in days "
            f"(default: {INTRADAY_CONSEC_LOSS_LOOKBACK_DAYS})"
        ),
    )
    parser.add_argument(
        "--scan-funnel",
        default=str(_DEFAULT_SCAN_FUNNEL),
        help=(
            f"Path to scan_funnel.jsonl for §9.2(b) cap-utilisation "
            f"surface (default: {_DEFAULT_SCAN_FUNNEL})"
        ),
    )
    parser.add_argument(
        "--cap-blocks",
        default=str(_DEFAULT_CAP_BLOCKS),
        help=(
            f"Path to per_day_cap_blocks.jsonl for per-cell block count "
            f"surface (default: {_DEFAULT_CAP_BLOCKS})"
        ),
    )
    parser.add_argument(
        "--include-cap-blocks",
        action="store_true",
        help="Include per-cell per-day cap block counts in the printed report",
    )
    args = parser.parse_args()

    if args.pull_vps:
        from scripts._vps import VPS_STATE_DIR, VPS_TRADES_PATH, require_vps_host

        vps_host = require_vps_host()

        local_trades = _REPO_ROOT / "data" / "vps_trades.jsonl"
        print(f"Pulling trades from VPS ({vps_host})...")
        subprocess.run(
            ["scp", f"{vps_host}:{VPS_TRADES_PATH}", str(local_trades)],
            check=True,
        )
        print(f"  -> {local_trades}")

        # Phase 0 depth-probe sink lives only on the VPS; without pulling
        # it, the daily-report Phase 0 summary renders empty throughout
        # the 2-week measurement window. Best-effort: rotated siblings
        # may be pruned beyond retention or absent on a fresh deploy, so
        # a missing-file scp must not abort the whole report run.
        local_phase0 = _REPO_ROOT / "state" / PHASE0_DEPTH_SINK_FILENAME
        local_phase0.parent.mkdir(parents=True, exist_ok=True)
        print(f"Pulling Phase 0 depth sink from VPS ({vps_host})...")
        _phase0_result = subprocess.run(
            [
                "scp",
                f"{vps_host}:{VPS_STATE_DIR}/{PHASE0_DEPTH_SINK_FILENAME}",
                str(local_phase0),
            ],
            capture_output=True,
            text=True,
        )
        if _phase0_result.returncode != 0:
            print(f"  !! skipped {PHASE0_DEPTH_SINK_FILENAME}: {_phase0_result.stderr.strip()}")
        else:
            print(f"  -> {local_phase0}")

    # Operator-local with --pull-vps lands trades at data/vps_trades.jsonl;
    # VPS-direct cron has live trades at state/trades.jsonl (the same path
    # --trade-log defaults to). Every downstream consumer that reads the
    # raw trade log (the Intraday EMOS HIGH section + the --refresh
    # subprocess) must follow this redirect, otherwise a --pull-vps run
    # on a laptop reads from a path that doesn't exist there.
    effective_trade_log = (
        _REPO_ROOT / "data" / "vps_trades.jsonl" if args.pull_vps else Path(args.trade_log)
    )

    if args.refresh:
        print("Refreshing settlements from Kalshi API...")
        refresh_trades_path = effective_trade_log
        # check=True so a refresh-subprocess failure surfaces loudly via
        # CalledProcessError instead of silently allowing the report to
        # proceed against a stale shadow_settlements.full.json.
        subprocess.run(
            [
                sys.executable,
                "scripts/fetch_shadow_settlements.py",
                "--trades",
                str(refresh_trades_path),
            ],
            cwd=_REPO_ROOT,
            check=True,
        )

    settlements_path = Path(args.settlements)
    snapshots_dir = Path(args.snapshots)

    if not settlements_path.exists():
        print(f"ERROR: {settlements_path} not found. Run with --refresh first.", flush=True)
        raise SystemExit(1)

    report_dt = date.fromisoformat(args.date)
    all_records = load_records(settlements_path)
    filtered = filter_records(all_records, args.date, args.mode)

    # The SETTLED P&L block, settled_per_cell (post_flip_monitor.sh T1/T2/T3),
    # and the pending GFS/ECMWF scorecard are weather-model-specific. Scope them
    # to the weather allowlist by default so the model-free favorite_bias (which
    # trades the same Kalshi weather series on shared city/var keys) cannot
    # contaminate the live pilot's rollback triggers. An explicit --strategy
    # overrides the allowlist to that single strategy.
    weather_scope = frozenset({args.strategy}) if args.strategy else WEATHER_STRATEGY_NAMES
    weather_filtered = filter_by_strategy(filtered, weather_scope)

    settled_days = build_settled_report(weather_filtered)
    pending = build_pending_scorecard(weather_filtered, snapshots_dir, report_date=report_dt)
    # Mode-only filter (NOT date+mode): calibration_ratio needs rolling N to
    # be statistically informative; restricting to args.date would collapse it
    # to a same-day ratio with N≈0-5 typical.
    mode_filtered_records = filter_records(all_records, trade_date=None, mode=args.mode)
    calibration = compute_calibration_gap(mode_filtered_records, strategy_allowlist=weather_scope)
    if calibration:
        calibration["mode"] = args.mode

    # Generic per-strategy settled-P&L (all strategies; intraday excluded — it
    # has its own section). Scoped by --strategy when provided.
    strategy_pnl_records = filter_by_strategy(
        filtered, frozenset({args.strategy}) if args.strategy else None
    )
    strategy_pnl = build_strategy_pnl_summary(strategy_pnl_records)

    # Phase 0 depth-probing summary (intraday EMOS HIGH spec §6.4 / §12.1).
    # ``summarize_phase0_depth`` tolerates a missing sink file: the early
    # days of Phase 0 will have no data and the function returns an empty
    # ``cells`` map plus ``n_samples=0`` so the report renders cleanly.
    phase0_summary = summarize_phase0_depth(
        Path(args.phase0_samples),
        lookback_days=args.phase0_lookback_days,
    )

    # Intraday EMOS HIGH per-cell section (spec §9.2(c) v1.3 path-(b)
    # + §11.4). Tolerant of a missing trade log: early-shadow days
    # will have no intraday rows and the helper returns an empty
    # cells map so the report renders a single-line placeholder.
    intraday_summary = summarize_intraday_strategy(
        effective_trade_log,
        lookback_days=args.intraday_lookback_days,
        scan_funnel_path=Path(args.scan_funnel),
    )

    # Per-day cap block counts per cell, filtered to the report date's
    # forecast_date dimension. Tolerant of a missing file (pre-deploy or
    # cap disabled via max_bets_per_day_per_event: null).
    cap_blocks = load_cap_blocks_by_cell(Path(args.cap_blocks), report_date=args.date)

    report = format_report(
        settled_days,
        pending,
        calibration,
        args.date,
        phase0_summary=phase0_summary,
        intraday_summary=intraday_summary,
        cap_blocks_by_cell=cap_blocks if args.include_cap_blocks else None,
        strategy_pnl=strategy_pnl,
    )
    print(report)

    # Boundary-distance filter kill watch (Path 7a, spec §6.5). Renders only
    # when at least one cell is enabled in weather.boundary_filter_cells; the
    # rolling mode-scoped cohort (not date-restricted) backs the per-cell CI.
    try:
        from python.settings import Settings

        _boundary_cells = Settings.load().strategies.weather.boundary_filter_cells
    except Exception as exc:  # noqa: BLE001 — telemetry section must never abort the report
        _boundary_cells = {}
        print(f"  (boundary-filter kill watch skipped: {exc})")
    boundary_section = render_boundary_filter_section(
        mode_filtered_records, _boundary_cells, since=args.boundary_since
    )
    if boundary_section:
        print(boundary_section)

    if args.json:
        output = build_json_output(
            args.date,
            settled_days,
            pending,
            calibration,
            phase0_summary=phase0_summary,
            intraday_summary=intraday_summary,
            cap_blocks_by_cell=cap_blocks if (cap_blocks and args.include_cap_blocks) else None,
            strategy_pnl=strategy_pnl,
        )
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON report written to {args.json}")


if __name__ == "__main__":
    main()
