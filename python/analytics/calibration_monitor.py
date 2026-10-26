"""Live calibration monitoring for the weather strategy.

Joins prediction-log records (written when a signal passes strategy gates)
with realized market outcomes to produce a rolling picture of calibration
quality. Emits Telegram alerts when the model starts drifting off its long-
run calibration baseline.

Outcome sources (selectable via ``CalibrationMonitorConfig.settlement_source``):

  - ``"kalshi"`` (default): joins ``data/shadow_settlements.full.json`` on
    ``market_ticker``. Authoritative — these are the same outcomes that
    appear on the trade dashboards (``shadow_dashboard.py``,
    ``daily_trade_report.py``) and that drive realized P&L. Unifying on this
    source means ``/brier`` and ``/profit`` agree on the same set of
    fundamental events. On a 248-market spot-check, the legacy NWS join
    disagreed with Kalshi on 45/248 (18%) of markets, 42 of which were a
    right-boundary ``between``-bracket settlement-rule mismatch (NWS-derived
    used ``floor <= temp < cap`` while Kalshi settles ``floor <= temp <= cap``),
    plus 3 stale-preliminary-NWS rows that never received a final-CLI
    update in our snapshot tree.
  - ``"nws"`` (legacy fallback): NWS Daily Climate Reports from
    ``data/snapshots/<date>/<run>/settlements.json``. Same source
    ``python/data/nws.py`` produces. Kept available as a fast-path lead
    indicator (NWS final-CLI lands ~9am next day; Kalshi-API settlement
    lands later) and so future operators can A/B compare without code
    changes. The Brier numbers feeding any kill decision should NOT use
    this path.

Three layers of independent observability:

  1. **Reliability bins** — bucket YES-probabilities into deciles, measure
     observed frequency per bin, attach a Wilson 95% CI. This is the canonical
     diagnostic for "does the 70% bin actually win 70%?".
  2. **Rolling Brier score** — 7d vs 30d score for overall skill. Brier
     decomposes into reliability + resolution - uncertainty (Murphy, 1973);
     we surface the scalar because that's what the alert thresholds key on.
  3. **Per-city-var segmentation** — we already know certain city/vars are
     untradeable (see ``excluded_city_vars`` in config.yaml). This surfaces
     the same finding at runtime so we notice regressions for the *tradeable*
     set before they show up in P&L.

Design notes:

- Kalshi settlement loading (default): reads
  ``data/shadow_settlements.full.json`` and emits a ``market_ticker → 0|1``
  outcome map. Records with ``market_result`` not in ``{"yes", "no"}`` are
  treated as unsettled (caller skips). No temperature is recovered along
  this path — ``SettledPair.actual_temp`` is ``None`` for Kalshi-source rows.
- NWS settlement loading (legacy fallback): keys on
  ``(city, observation_date)`` and selects the record with the latest
  ``(issued_at, product_id)`` tuple. ``is_preliminary`` is informational
  only; later issuance always subsumes earlier. See
  ``scripts/_calibration_common.prefer_latest_settlement`` for the canonical
  semantic. The same rule is replicated here (different record shape) rather
  than imported so the monitor stays dependency-free.
- Alerts are keyed on ``(code, severity)`` with a configurable cooldown, and
  escalation (warning → critical) bypasses the cooldown to avoid alert storms
  without muting genuine regime changes.
- Background loop is wrapped in per-tick ``try/except`` so a bad settlement
  file can never kill the monitor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from python.analytics.outcomes import derive_outcome_yes
from python.analytics.prediction_log import (
    EvalAuditRecordSource,
    PredictionLog,
    PredictionRecord,
)
from python.notifications.base import AlertLevel
from python.utils.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

# Minimum number of settled prediction-outcome pairs before the monitor emits
# any alerts. Below this the windowed Brier has too little signal to act on.
DEFAULT_WARMUP_PAIRS = 40

# Alert thresholds — configurable via CalibrationMonitorConfig but these are
# the defaults baked into the design.
DEFAULT_BRIER_WARN_MULT = 1.3  # 7d > 30d × 1.3 → warning
DEFAULT_BRIER_CRIT_MULT = 1.6  # 7d > 30d × 1.6 → critical
DEFAULT_BIN_DEVIATION = 0.15  # |observed - predicted| in any bin with n ≥ min
DEFAULT_BIN_DEVIATION_ENABLED = True  # master switch for the bin_deviation alert
DEFAULT_BIN_MIN_N = 20
DEFAULT_CITY_VAR_BRIER = 0.25
DEFAULT_CITY_VAR_MIN_N = 20
DEFAULT_COOLDOWN_SECONDS = 4 * 3600.0

_WINDOW_7D_DAYS = 7
_WINDOW_30D_DAYS = 30

# Decile bins — left-closed, right-exclusive, except the final bin which is
# right-closed so probability=1.0 has a home.
_BIN_EDGES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettledPair:
    """A prediction joined with its realized outcome, in YES-space."""

    timestamp: datetime  # signal generation time
    market_ticker: str
    city: str
    daily_variable: str  # "high" | "low"
    strike_type: str
    predicted_yes: float  # calibrated probability (== signal.model_probability)
    outcome_yes: int  # 1 if YES won, else 0
    raw_ensemble_prob: float | None  # pre-isotonic fraction, if weather
    observation_date: date
    # Only populated when ``settlement_source == "nws"`` and the NWS table
    # carries a temperature. Kept on the dataclass so reliability-by-
    # temperature diagnostics in downstream scripts continue to work against
    # the NWS path, but ``None`` is the rule on a live (Kalshi-default) monitor.
    actual_temp: float | None = None
    # Which calibrator produced ``predicted_yes`` on this row. ``"unknown"``
    # for legacy ``PredictionRecord`` rows written before schema v4 (so the
    # monitor never silently drops them when the 30d window straddles the
    # schema bump).
    live_calibrator_kind: str = "unknown"
    # Which settlement source produced ``outcome_yes`` — informational, used
    # by tests and diagnostics. Defaults to ``"nws"`` so existing call-sites
    # that construct ``SettledPair`` directly don't have to be touched.
    settlement_source: str = "nws"
    # Naive-persistence prediction: yesterday's observed extremum at the same
    # (city, var) station, evaluated against today's bracket. 1 if yesterday's
    # temp would have settled this market YES, 0 if NO, None if D-1 data is
    # unavailable (first day of series, missing station product, missing
    # variable). Pure baseline — short-horizon meteorological standard for
    # daily-extremum forecasting (Hamill 2012).
    persistence_yes: int | None = None


@dataclass(frozen=True)
class ReliabilityBin:
    """Observed vs predicted frequency for a decile bin."""

    lower: float
    upper: float
    n: int
    mean_predicted: float
    mean_observed: float
    wilson_low: float
    wilson_high: float

    @property
    def label(self) -> str:
        right = "]" if self.upper >= 1.0 else ")"
        return f"[{self.lower:.1f}-{self.upper:.1f}{right}"


@dataclass(frozen=True)
class CityVarBrier:
    city: str
    daily_variable: str
    n: int
    brier: float
    mean_predicted: float
    mean_observed: float
    # Which calibrator produced this bucket. Per-kind rows carry
    # ``"isotonic"`` / ``"emos_raw"`` / ``"isotonic_fallback"`` /
    # ``"identity"`` / ``"unknown"``; the aggregate (sum-over-kinds) row uses
    # the sentinel ``"all"`` so legacy consumers that key on (city, var)
    # still find a row.
    live_calibrator_kind: str = "all"
    # Naive-persistence baseline diagnostic. ``persistence_brier`` is the
    # Brier of the 0/1 yesterday-extremum-in-today's-bracket predictor over
    # the subset of pairs with D-1 data; ``bss_vs_persistence`` is
    # ``1 - brier / persistence_brier`` (None when ``persistence_brier == 0``).
    # ``n_persistence`` is the subset size and may be less than ``n``.
    persistence_brier: float | None = None
    bss_vs_persistence: float | None = None
    n_persistence: int = 0


@dataclass(frozen=True)
class CalibrationSnapshot:
    """A point-in-time view of calibration quality. Read by Telegram handlers."""

    generated_at: datetime
    n_pairs_total: int
    n_pairs_7d: int
    n_pairs_30d: int
    brier_7d: float | None
    brier_30d: float | None
    brier_all: float | None
    reliability_7d: list[ReliabilityBin]
    reliability_30d: list[ReliabilityBin]
    per_city_var_30d: list[CityVarBrier]
    isotonic_fit_date: str | None

    def summary_line(self) -> str:
        """One-liner suitable for the status header."""

        def _fmt(b: float | None) -> str:
            return f"{b:.3f}" if b is not None else "—"

        return (
            f"Brier 7d={_fmt(self.brier_7d)} "
            f"30d={_fmt(self.brier_30d)} "
            f"all={_fmt(self.brier_all)} "
            f"(n7={self.n_pairs_7d}, n30={self.n_pairs_30d})"
        )


@dataclass(frozen=True)
class DriftAlert:
    """An actionable regression event, ready to hand to the notifier."""

    code: str  # e.g. "brier_7d_vs_30d", "bin_deviation", "city_var_brier"
    severity: AlertLevel
    message: str
    detail: dict[str, Any]
    # Which strategy this alert concerns. ``None`` means infrastructure —
    # process, disk, pipeline integrity, a dead writer — and always dispatches.
    # Set by the PRODUCER: a consumer cannot infer it from an f-string code.
    owner: str | None = None
    # The substring of ``code`` that changes when the underlying finding becomes
    # new (a fit_date, an artifact path). ``None`` means NOT acknowledgeable, so
    # a new alert code is un-ackable until its author declares this. Never set
    # it on an alert whose condition is closeable only by fixing the thing.
    ack_key: str | None = None


@dataclass
class MonitorHealth:
    """Self-observability for the monitor loop."""

    last_tick_ts: datetime | None = None
    last_successful_tick_ts: datetime | None = None
    consecutive_errors: int = 0
    last_error: str | None = None
    total_ticks: int = 0
    total_errors: int = 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMonitorConfig:
    enabled: bool = False
    tick_interval_seconds: float = 300.0
    warmup_pairs: int = DEFAULT_WARMUP_PAIRS
    brier_warn_mult: float = DEFAULT_BRIER_WARN_MULT
    brier_crit_mult: float = DEFAULT_BRIER_CRIT_MULT
    bin_deviation_threshold: float = DEFAULT_BIN_DEVIATION
    # Master switch for the absolute-miscalibration ``bin_deviation`` alert. It
    # is the only drift sub-alert with no baseline-relative concept, so for a
    # model with a *known persistent* per-bin miss it re-fires every cooldown
    # forever. Set ``False`` to silence it without touching the relative
    # ``brier_7d_vs_30d`` drift alert or the ``city_var_brier`` economic alert.
    bin_deviation_enabled: bool = DEFAULT_BIN_DEVIATION_ENABLED
    bin_min_n: int = DEFAULT_BIN_MIN_N
    city_var_brier_threshold: float = DEFAULT_CITY_VAR_BRIER
    city_var_min_n: int = DEFAULT_CITY_VAR_MIN_N
    alert_cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    # Outcome source for the prediction-vs-outcome join. ``"kalshi"`` (default)
    # uses ``data/shadow_settlements.full.json`` so the monitor agrees with the
    # trade dashboards; ``"nws"`` is the legacy path joining NWS snapshots in
    # ``snapshots_dir`` (kept available as a fast-path lead indicator). Any
    # kill-relevant Brier number should be on the Kalshi path.
    settlement_source: str = "kalshi"


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI. Returns (0.0, 0.0) if n == 0.

    Chosen over the normal approximation because it behaves sensibly at the
    tails (observed 0/n or n/n), which is exactly where reliability bins tend
    to sit for weather markets.
    """
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_sq_error(
    pairs: Iterable[SettledPair],
    predictor: Callable[[SettledPair], float | None],
    *,
    skip_none: bool,
) -> tuple[float | None, int]:
    """Mean squared error of ``predictor(pair)`` vs ``pair.outcome_yes``.

    When ``skip_none`` is True, pairs whose ``predictor`` returns ``None`` are
    skipped entirely (contribute to neither numerator nor denominator). When
    False, ``None`` is not expected and a missing prediction surfaces as an
    error on subtraction. Returns ``(None, 0)`` when no pair contributes.
    """
    total = 0.0
    n = 0
    for p in pairs:
        value = predictor(p)
        if value is None:
            if skip_none:
                continue
            # Not expected when skip_none is False; mirror the original
            # behaviour of failing on the subtraction below.
            raise TypeError("predictor returned None but skip_none=False")
        diff = value - p.outcome_yes
        total += diff * diff
        n += 1
    if n == 0:
        return None, 0
    return total / n, n


def _brier_score(pairs: Iterable[SettledPair]) -> tuple[float | None, int]:
    """Mean squared error of predicted probability vs outcome."""
    return _mean_sq_error(pairs, lambda p: p.predicted_yes, skip_none=False)


def _persistence_brier(pairs: Iterable[SettledPair]) -> tuple[float | None, int]:
    """Brier score of the naive-persistence baseline.

    Persistence emits 0/1 deterministically (yesterday's bracket-membership),
    so each per-pair squared error is 0 or 1 and the mean is the error rate.
    Pairs with ``persistence_yes is None`` (missing D-1 data) are skipped —
    they contribute neither to numerator nor denominator. Returns
    ``(None, 0)`` when no pair carries persistence data.
    """
    return _mean_sq_error(pairs, lambda p: p.persistence_yes, skip_none=True)


def _bin_index(prob: float) -> int:
    """Return the decile bin index for a probability in [0, 1]."""
    if prob >= 1.0:
        return len(_BIN_EDGES) - 2
    if prob <= 0.0:
        return 0
    return min(int(prob * 10), len(_BIN_EDGES) - 2)


def _reliability_bins(pairs: Iterable[SettledPair]) -> list[ReliabilityBin]:
    """Decile reliability table with Wilson CIs. Empty bins are included so
    that formatters see a consistent 10-row shape."""
    buckets: list[list[SettledPair]] = [[] for _ in range(len(_BIN_EDGES) - 1)]
    for p in pairs:
        buckets[_bin_index(p.predicted_yes)].append(p)

    out: list[ReliabilityBin] = []
    for i, bucket in enumerate(buckets):
        lower = _BIN_EDGES[i]
        upper = _BIN_EDGES[i + 1]
        n = len(bucket)
        if n == 0:
            out.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    n=0,
                    mean_predicted=0.0,
                    mean_observed=0.0,
                    wilson_low=0.0,
                    wilson_high=0.0,
                )
            )
            continue
        mean_pred = sum(p.predicted_yes for p in bucket) / n
        wins = sum(p.outcome_yes for p in bucket)
        mean_obs = wins / n
        lo, hi = _wilson_interval(wins, n)
        out.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                n=n,
                mean_predicted=mean_pred,
                mean_observed=mean_obs,
                wilson_low=lo,
                wilson_high=hi,
            )
        )
    return out


def _per_city_var_brier(pairs: Iterable[SettledPair]) -> list[CityVarBrier]:
    """Brier score grouped by (city, daily_variable, live_calibrator_kind).

    The 30d window can mix isotonic-era and emos_raw-era rows for the same
    (city, var) after a calibrator cutover. Pooling those into a single Brier
    hides regression signals. We emit one row per (city, var, kind) AND an
    aggregate ``kind="all"`` row per (city, var) so legacy consumers that
    only key on (city, var) keep working.
    """
    pair_list = list(pairs)
    per_kind: dict[tuple[str, str, str], list[SettledPair]] = defaultdict(list)
    agg: dict[tuple[str, str], list[SettledPair]] = defaultdict(list)
    for p in pair_list:
        kind = p.live_calibrator_kind or "unknown"
        per_kind[(p.city, p.daily_variable, kind)].append(p)
        agg[(p.city, p.daily_variable)].append(p)

    # Defensive: pairs with the literal "all" sentinel would collide
    # with the aggregate-row key. Fold them into "unknown" BEFORE
    # iterating so we don't mutate per_kind during iteration. Writer
    # sites already constrain the value space; this is a hand-edited-
    # audit-row / corrupt-JSONL safety net.
    collisions = [
        (city, var, items)
        for (city, var, kind), items in per_kind.items()
        if kind == "all"
    ]
    if collisions:
        for city, var, items in collisions:
            per_kind.pop((city, var, "all"), None)
            per_kind[(city, var, "unknown")].extend(items)
        logger.warning(
            "calibration_monitor: %d pair(s) carried reserved kind='all' — "
            "re-bucketed under 'unknown'. Investigate writer source.",
            sum(len(items) for _, _, items in collisions),
        )

    out: list[CityVarBrier] = []

    def _row(
        city: str, var: str, items: list[SettledPair], kind: str
    ) -> CityVarBrier | None:
        brier, n = _brier_score(items)
        if brier is None:
            return None
        mean_pred = sum(p.predicted_yes for p in items) / n
        mean_obs = sum(p.outcome_yes for p in items) / n
        # BSS must be computed on a matched sample so numerator and
        # denominator are means over the same pairs — otherwise
        # selection bias (the persistence-available subset being
        # systematically easier or harder than the full n) contaminates
        # the ratio. We compute live Brier on the persistence-available
        # subset and compare against persistence Brier on that same
        # subset; the headline ``brier`` field keeps the full-n
        # semantics for everything else.
        matched = [p for p in items if p.persistence_yes is not None]
        pers_brier, n_pers = _persistence_brier(matched)
        live_brier_on_matched, _ = _brier_score(matched)
        if (
            pers_brier is None
            or live_brier_on_matched is None
            or pers_brier == 0
        ):
            # BSS undefined when the matched subset is empty or the
            # persistence baseline is perfect. ``persistence_brier == 0
            # AND live > 0`` is an interesting case (model strictly worse
            # than naive), but distinguishing it would require a tri-state
            # sentinel and a consumer that reads it; keep None for now and
            # let the daily report surface the raw {brier,
            # persistence_brier, n_persistence} triple when we wire it.
            bss = None
        else:
            bss = 1.0 - (live_brier_on_matched / pers_brier)
        return CityVarBrier(
            city=city,
            daily_variable=var,
            n=n,
            brier=brier,
            mean_predicted=mean_pred,
            mean_observed=mean_obs,
            live_calibrator_kind=kind,
            persistence_brier=pers_brier,
            bss_vs_persistence=bss,
            n_persistence=n_pers,
        )

    for (city, var, kind), items in per_kind.items():
        row = _row(city, var, items, kind)
        if row is not None:
            out.append(row)

    # Aggregate ``all`` rows — preserves backwards-compat for consumers
    # that only key on (city, var).
    for (city, var), items in agg.items():
        row = _row(city, var, items, "all")
        if row is not None:
            out.append(row)

    out.sort(key=lambda r: (-r.n, r.city, r.daily_variable, r.live_calibrator_kind))
    return out


# ---------------------------------------------------------------------------
# Settlement loading
# ---------------------------------------------------------------------------


@dataclass
class _SettlementEntry:
    city: str
    observation_date: date
    high_f: float | None
    low_f: float | None
    is_preliminary: bool
    issued_at: str = ""
    product_id: str = ""


def _load_settlements(
    snapshots_dir: Path, *, since: date | None = None
) -> dict[tuple[str, date], _SettlementEntry]:
    """Build the (city, observation_date) → settlement table.

    For each key, the record with the greatest ``(issued_at, product_id)``
    tuple wins. This matches ``scripts/_calibration_common.prefer_latest_settlement``.
    ``is_preliminary`` is tracked but NOT part of the merge key — NWS issues
    multiple products per observation date (morning partial, afternoon
    partial, next-morning final) and the latest issuance is always
    authoritative. Using ``is_preliminary`` as a tiebreak was the source of
    the AUS/DEN morning-partial bug (partial-day CLI misclassified as final
    by the regex, beating the real final next-morning report).

    ``since`` caps how far back the walk reads from disk. Day dirs whose
    name parses to a date < ``since`` are skipped without opening any
    files inside. Dirs whose name is not an ISO date (e.g. ``index.jsonl``)
    are not affected and continue to be filtered by the existing rules.
    """
    table: dict[tuple[str, date], _SettlementEntry] = {}
    try:
        day_dirs = sorted(snapshots_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return table

    for day_dir in day_dirs:
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        if since is not None:
            try:
                dir_date = date.fromisoformat(day_dir.name)
            except ValueError:
                # Non-date directory names fall through to the existing
                # file-presence checks; the bounded-window cap only applies
                # to date-named day dirs.
                pass
            else:
                if dir_date < since:
                    continue

        # Snapshots are either flat (older) or nested under run subdirs.
        flat = day_dir / "settlements.json"
        if flat.is_file():
            dirs_to_scan = [day_dir]
        else:
            dirs_to_scan = sorted(p for p in day_dir.iterdir() if p.is_dir())

        for d in dirs_to_scan:
            f = d / "settlements.json"
            try:
                records = json.loads(f.read_text())
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError):
                logger.warning("calibration_monitor: failed to read %s", f, exc_info=True)
                continue
            for rec in records:
                city = rec.get("city")
                obs_raw = rec.get("observation_date")
                if not city or not obs_raw:
                    continue
                try:
                    obs = date.fromisoformat(str(obs_raw))
                except ValueError:
                    continue
                entry = _SettlementEntry(
                    city=city,
                    observation_date=obs,
                    high_f=_as_float(rec.get("high_f")),
                    low_f=_as_float(rec.get("low_f")),
                    is_preliminary=bool(rec.get("is_preliminary", False)),
                    issued_at=str(rec.get("issued_at") or ""),
                    product_id=str(rec.get("product_id") or ""),
                )
                key = (city, obs)
                existing = table.get(key)
                # Canonical rule: max((issued_at, product_id)) wins. Mirrors
                # scripts._calibration_common.prefer_latest_settlement; the
                # record shape is different so we replicate the comparison
                # inline rather than round-trip through a dict.
                new_key = (entry.issued_at, entry.product_id)
                if existing is None or new_key > (
                    existing.issued_at,
                    existing.product_id,
                ):
                    table[key] = entry
    return table


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _load_kalshi_outcomes(path: Path) -> dict[str, int]:
    """Load Kalshi-authoritative outcomes from ``shadow_settlements.full.json``.

    Returns a ``market_ticker → 0|1`` map. Records are filtered to those that
    are settled with ``market_result`` in ``{"yes", "no"}``; everything else
    (unsettled, void, missing) is excluded — the caller decides what to do
    with prediction records whose ticker isn't in the map (treat as unsettled).

    When the same ticker appears multiple times (shadow appends one fill per
    scan; reconciler back-fills synthetic rows), all rows for a given ticker
    must agree on ``market_result`` because the market itself only settles
    once. If a disagreement is observed we log a warning and keep the FIRST
    seen outcome — the disagreement is a data-quality bug worth investigating
    upstream, not a silent corruption.

    Returns ``{}`` if the file is missing or unreadable — the monitor must
    continue ticking on a fresh deploy (no Kalshi file yet) rather than
    crashing.
    """
    table: dict[str, int] = {}
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return table
    except OSError:
        logger.warning(
            "calibration_monitor: failed to read Kalshi settlements %s",
            path,
            exc_info=True,
        )
        return table
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "calibration_monitor: Kalshi settlements %s is not valid JSON",
            path,
            exc_info=True,
        )
        return table
    if not isinstance(records, list):
        logger.warning(
            "calibration_monitor: Kalshi settlements %s top-level is not a list",
            path,
        )
        return table

    # File-format note: ``data/shadow_settlements.full.json`` is a LIST of
    # per-trade records (one per fill), not a dict keyed by ticker. The
    # producer is ``scripts/fetch_shadow_settlements.py`` (regenerates the
    # file wholesale on each run via ``write_text(json.dumps(results))`` — no
    # append, no history). Multiple records for the same ticker are
    # expected (multiple fills on one market); they should all carry the
    # same ``market_result`` because Kalshi settles a ticker as a whole.
    # Disagreement here = upstream data corruption; "first-seen wins" is a
    # defensive policy with a logged WARNING.
    n_conflicts = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ticker = rec.get("market_ticker")
        if not isinstance(ticker, str) or not ticker:
            continue
        # Treat ``settled == False`` as "not yet settled, exclude". Treat
        # ``market_result`` outside {"yes", "no"} (e.g. None for unsettled,
        # or "void") as exclude too.
        if not rec.get("settled", False):
            continue
        result = rec.get("market_result")
        if result == "yes":
            outcome = 1
        elif result == "no":
            outcome = 0
        else:
            continue
        prior = table.get(ticker)
        if prior is None:
            table[ticker] = outcome
        elif prior != outcome:
            n_conflicts += 1
    if n_conflicts:
        logger.warning(
            "calibration_monitor: Kalshi settlements file %s has %d ticker(s) "
            "with conflicting market_result rows; kept first-seen outcome. "
            "Investigate upstream (settlement-fetcher or reconciler).",
            path,
            n_conflicts,
        )
    return table


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _actual_temp_for(entry: _SettlementEntry, daily_variable: str) -> float | None:
    if daily_variable == "high":
        return entry.high_f
    if daily_variable == "low":
        return entry.low_f
    return None


def _persistence_yes_from_d1(
    rec: PredictionRecord,
    settlements: dict[tuple[str, date], _SettlementEntry],
) -> int | None:
    """Return the naive-persistence prediction (0/1) or None.

    Looks up yesterday's observed extremum at the same (city, var) and asks
    ``derive_outcome_yes(strike_type, floor, cap, yesterday_temp)``. Returns
    None when D-1 settlement is missing, the variable is missing on the D-1
    row, or strike metadata is inconsistent.
    """
    if rec.city is None or rec.daily_variable is None or rec.strike_type is None:
        return None
    if rec.expiration_date is None:
        return None
    d1 = rec.expiration_date - timedelta(days=1)
    entry = settlements.get((rec.city, d1))
    if entry is None:
        return None
    yesterday_temp = _actual_temp_for(entry, rec.daily_variable)
    if yesterday_temp is None:
        return None
    yes = derive_outcome_yes(
        strike_type=rec.strike_type,
        floor_strike=rec.floor_strike,
        cap_strike=rec.cap_strike,
        actual_temp=yesterday_temp,
    )
    if yes is None:
        return None
    return 1 if yes else 0


def _record_to_pair(
    rec: PredictionRecord,
    settlements: dict[tuple[str, date], _SettlementEntry],
) -> SettledPair | None:
    """Join a prediction record with its realized NWS-derived settlement.

    Returns ``None`` if:
      - the record is from a non-weather strategy (no city/var)
      - the expiration date isn't yet settled
      - strike metadata is missing/inconsistent for the strike type

    Note: this is the legacy NWS path. The Kalshi-authoritative path used by
    the default monitor configuration goes through ``_record_to_pair_kalshi``.
    """
    if rec.strategy != "weather_edge":
        return None
    if rec.city is None or rec.daily_variable is None or rec.strike_type is None:
        return None
    if rec.expiration_date is None:
        return None

    entry = settlements.get((rec.city, rec.expiration_date))
    if entry is None:
        return None

    actual = _actual_temp_for(entry, rec.daily_variable)
    if actual is None:
        return None

    yes = derive_outcome_yes(
        strike_type=rec.strike_type,
        floor_strike=rec.floor_strike,
        cap_strike=rec.cap_strike,
        actual_temp=actual,
    )
    if yes is None:
        return None

    return SettledPair(
        timestamp=rec.timestamp,
        market_ticker=rec.market_ticker,
        city=rec.city,
        daily_variable=rec.daily_variable,
        strike_type=rec.strike_type,
        predicted_yes=rec.calibrated_prob,
        outcome_yes=1 if yes else 0,
        raw_ensemble_prob=rec.raw_ensemble_prob,
        observation_date=rec.expiration_date,
        actual_temp=actual,
        # Legacy rows carry ``None`` for live_calibrator_kind; bucket as
        # "unknown" rather than silently dropping.
        live_calibrator_kind=(rec.live_calibrator_kind or "unknown"),
        settlement_source="nws",
        persistence_yes=_persistence_yes_from_d1(rec, settlements),
    )


def _record_to_pair_kalshi(
    rec: PredictionRecord,
    outcomes: dict[str, int],
    settlements: dict[tuple[str, date], _SettlementEntry] | None = None,
) -> SettledPair | None:
    """Join a prediction record with its Kalshi-authoritative outcome.

    Returns ``None`` if:
      - the record is from a non-weather strategy (no city/var/strike)
      - no Kalshi settlement is known for the ticker (market still open or
        not yet fetched by ``scripts/fetch_shadow_settlements.py``)

    Unlike the NWS path, this side does NOT need ``strike_type`` / strike
    bounds to derive the outcome — Kalshi has already done that. We still
    require those fields on the record so per-cell partitioning works.
    ``actual_temp`` is left ``None`` because Kalshi settlement data does
    not include the raw observed temperature.

    The optional ``settlements`` argument is used only to populate
    ``persistence_yes`` from the D-1 NWS extremum. Kalshi outcomes remain
    authoritative; this is a diagnostic side-channel.
    """
    if rec.strategy != "weather_edge":
        return None
    if rec.city is None or rec.daily_variable is None or rec.strike_type is None:
        return None
    if rec.expiration_date is None:
        return None

    outcome = outcomes.get(rec.market_ticker)
    if outcome is None:
        return None

    persistence_yes = (
        _persistence_yes_from_d1(rec, settlements) if settlements else None
    )

    return SettledPair(
        timestamp=rec.timestamp,
        market_ticker=rec.market_ticker,
        city=rec.city,
        daily_variable=rec.daily_variable,
        strike_type=rec.strike_type,
        predicted_yes=rec.calibrated_prob,
        outcome_yes=outcome,
        raw_ensemble_prob=rec.raw_ensemble_prob,
        observation_date=rec.expiration_date,
        actual_temp=None,
        live_calibrator_kind=(rec.live_calibrator_kind or "unknown"),
        settlement_source="kalshi",
        persistence_yes=persistence_yes,
    )


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------


# Ordered worst→best; index gives severity rank for escalation logic.
_SEVERITY_RANK: dict[AlertLevel, int] = {
    AlertLevel.INFO: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.ERROR: 2,
    AlertLevel.CRITICAL: 3,
}


@dataclass
class _AlertState:
    """Persisted per-code alert state."""

    severity: AlertLevel
    last_fired_ts: float


class _AlertGate:
    """Debounces alerts by (code, severity) with escalation bypass."""

    def __init__(self, cooldown_seconds: float) -> None:
        self._cooldown = cooldown_seconds
        self._state: dict[str, _AlertState] = {}

    def load_state(self, raw: dict[str, Any]) -> None:
        self._state.clear()
        for code, entry in (raw or {}).items():
            try:
                severity = AlertLevel(entry["severity"])
                ts = float(entry["last_fired_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if severity in _SEVERITY_RANK:
                self._state[code] = _AlertState(severity=severity, last_fired_ts=ts)

    def dump_state(self) -> dict[str, Any]:
        return {
            code: {"severity": s.severity.value, "last_fired_ts": s.last_fired_ts}
            for code, s in self._state.items()
        }

    def should_fire(self, code: str, severity: AlertLevel, now_ts: float) -> bool:
        severity = AlertLevel(severity)
        existing = self._state.get(code)
        if existing is None:
            return True
        if _SEVERITY_RANK[severity] > _SEVERITY_RANK[existing.severity]:
            return True
        return (now_ts - existing.last_fired_ts) >= self._cooldown

    def record(self, code: str, severity: AlertLevel, now_ts: float) -> None:
        self._state[code] = _AlertState(
            severity=AlertLevel(severity), last_fired_ts=now_ts
        )

    def clear(self, code: str) -> None:
        self._state.pop(code, None)


# ---------------------------------------------------------------------------
# Monitor service
# ---------------------------------------------------------------------------


class CalibrationMonitor:
    """Background service that refreshes calibration stats and emits alerts.

    Public surface (used by Telegram and orchestrator):

      - ``start()`` / ``stop()`` — lifecycle.
      - ``latest_snapshot`` property — most recent ``CalibrationSnapshot``; None
        until the first tick completes.
      - ``health`` property — ``MonitorHealth`` for self-observability.
      - ``force_refresh()`` — await a one-off refresh (used by tests and by
        ``/reliability`` on first use to avoid "monitor not warm" gaps).

    Alert emission goes through a user-provided async callback so the monitor
    stays decoupled from the notifier (and tests can inject a list.append).
    """

    def __init__(
        self,
        *,
        prediction_log: PredictionLog | EvalAuditRecordSource,
        snapshots_dir: Path | str,
        state_path: Path | str,
        config: CalibrationMonitorConfig,
        alert_callback: Any | None = None,  # async fn(DriftAlert) -> None
        isotonic_fit_date: str | None = None,
        kalshi_settlements_path: Path | str | None = None,
        corpus_stream_path: Path | str | None = None,
    ) -> None:
        self._log = prediction_log
        # Live stream file backing the corpus (eval_audit.jsonl when the
        # eval-audit source is wired). Enables the ``corpus_dark`` semantic
        # alert: pair count transitioning >0 → 0 while this file is FRESH
        # means the join/filter broke (schema drift, strategy-tag rename) —
        # a failure class a pure file-staleness tripwire cannot see. ``None``
        # (legacy predictions source) disables the check: that stream's
        # silence is ambiguous (no signals ≠ broken).
        self._corpus_stream_path = (
            Path(corpus_stream_path) if corpus_stream_path is not None else None
        )
        self._last_n_pairs_30d: int | None = None
        self._snapshots_dir = Path(snapshots_dir)
        self._state_path = Path(state_path)
        self._config = config
        self._alert_cb = alert_callback
        self._isotonic_fit_date = isotonic_fit_date
        # Kalshi-authoritative outcomes file. Required when
        # ``config.settlement_source == "kalshi"``; ignored otherwise. Passing
        # ``None`` keeps the legacy default path (``data/shadow_settlements.full.json``)
        # so callers that don't know about the new field stay working.
        self._kalshi_settlements_path = (
            Path(kalshi_settlements_path)
            if kalshi_settlements_path is not None
            else Path("data/shadow_settlements.full.json")
        )
        if config.settlement_source not in ("kalshi", "nws"):
            raise ValueError(
                f"CalibrationMonitorConfig.settlement_source={config.settlement_source!r} "
                "must be one of {'kalshi', 'nws'}."
            )

        self._gate = _AlertGate(cooldown_seconds=config.alert_cooldown_seconds)
        self._latest: CalibrationSnapshot | None = None
        self._health = MonitorHealth()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._refresh_lock = asyncio.Lock()
        # Captured at start() so _tick_sync (on a worker thread) can schedule
        # alert delivery back on the running loop.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Ring buffer of recent alerts, small, for diagnostics and /health.
        self._recent_alerts: deque[DriftAlert] = deque(maxlen=50)

        self._restore_state()

    # -- Properties ----------------------------------------------------------

    @property
    def latest_snapshot(self) -> CalibrationSnapshot | None:
        return self._latest

    @property
    def health(self) -> MonitorHealth:
        return self._health

    @property
    def recent_alerts(self) -> list[DriftAlert]:
        return list(self._recent_alerts)

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if not self._config.enabled:
            logger.info("calibration_monitor: disabled in config")
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="calibration-monitor")
        logger.info(
            "calibration_monitor: started (tick=%.0fs, warmup=%d)",
            self._config.tick_interval_seconds,
            self._config.warmup_pairs,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - cleanup
                pass
        self._task = None

    async def force_refresh(self) -> CalibrationSnapshot | None:
        """Run a single refresh outside the main loop. Safe to call anytime."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        await self._tick_safe()
        return self._latest

    # -- Background loop -----------------------------------------------------

    async def _run(self) -> None:
        interval = max(1.0, float(self._config.tick_interval_seconds))
        while not self._stop_event.is_set():
            await self._tick_safe()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def _tick_safe(self) -> None:
        """One tick wrapped so a bad settlement file can never kill the loop."""
        async with self._refresh_lock:
            self._health.total_ticks += 1
            self._health.last_tick_ts = datetime.now(UTC)
            try:
                await asyncio.to_thread(self._tick_sync)
                self._health.last_successful_tick_ts = self._health.last_tick_ts
                self._health.consecutive_errors = 0
                self._health.last_error = None
            except Exception as exc:  # noqa: BLE001 - logged + surfaced
                self._health.consecutive_errors += 1
                self._health.total_errors += 1
                self._health.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("calibration_monitor: tick failed")

    # -- Synchronous refresh (I/O-bound; offloaded to a worker thread) -------

    def _tick_sync(self) -> None:
        # NWS settlements are loaded on both paths now: the Kalshi path uses
        # them only as a D-1 side-channel to populate ``persistence_yes`` for
        # the per-cell BSS-vs-persistence diagnostic. Kalshi outcomes remain
        # authoritative for ``outcome_yes`` on that path. On a fresh deploy
        # with no NWS snapshots yet, ``_load_settlements`` returns ``{}`` and
        # ``persistence_yes`` is just ``None`` everywhere — no crash.
        #
        # I/O note: this walks the full snapshots tree (~60 day dirs today,
        # grows daily). Acceptable at current scale; the loader accepts a
        # ``since`` kwarg for a future bounded-window optimization once test
        # fixtures pin "now" or use recent dates.
        settlements = _load_settlements(self._snapshots_dir)
        if self._config.settlement_source == "kalshi":
            outcomes = _load_kalshi_outcomes(self._kalshi_settlements_path)
            pairs = self._build_pairs_kalshi(outcomes, settlements)
        else:
            pairs = self._build_pairs(settlements)

        now = datetime.now(UTC)
        cutoff_7d = now - timedelta(days=_WINDOW_7D_DAYS)
        cutoff_30d = now - timedelta(days=_WINDOW_30D_DAYS)

        pairs_7d = [p for p in pairs if p.timestamp >= cutoff_7d]
        pairs_30d = [p for p in pairs if p.timestamp >= cutoff_30d]

        brier_7d, n_7d = _brier_score(pairs_7d)
        brier_30d, n_30d = _brier_score(pairs_30d)
        brier_all, n_all = _brier_score(pairs)

        snapshot = CalibrationSnapshot(
            generated_at=now,
            n_pairs_total=n_all,
            n_pairs_7d=n_7d,
            n_pairs_30d=n_30d,
            brier_7d=brier_7d,
            brier_30d=brier_30d,
            brier_all=brier_all,
            reliability_7d=_reliability_bins(pairs_7d),
            reliability_30d=_reliability_bins(pairs_30d),
            per_city_var_30d=_per_city_var_brier(pairs_30d),
            isotonic_fit_date=self._isotonic_fit_date,
        )
        self._latest = snapshot

        # Corpus-liveness check — MUST run before the warmup early-return
        # (its trigger condition is exactly n → 0). Transition-based: fires
        # only when a previously nonzero 30d pair count collapses to zero
        # while the backing stream file is still fresh — i.e. rows keep
        # landing but none survive the strategy/schema join. A never-populated
        # corpus (fresh deploy, settlements not yet fetched) is not a
        # transition and stays silent.
        dark_alert = self._check_corpus_dark(n_30d, now)
        self._last_n_pairs_30d = n_30d
        if dark_alert is not None:
            self._recent_alerts.append(dark_alert)
            if self._alert_cb is not None:
                self._dispatch_alert(dark_alert)
            self._persist_state()

        if n_all < self._config.warmup_pairs:
            logger.debug(
                "calibration_monitor: warming up (%d/%d pairs)",
                n_all,
                self._config.warmup_pairs,
            )
            return

        alerts = self._evaluate_alerts(snapshot)
        if not alerts:
            return
        for alert in alerts:
            self._recent_alerts.append(alert)
        if self._alert_cb is not None:
            for alert in alerts:
                self._dispatch_alert(alert)
        # Persist only when alert state actually changed (gate recorded a fire).
        self._persist_state()

    def _dispatch_alert(self, alert: DriftAlert) -> None:
        # Called from a worker thread (via asyncio.to_thread), so hop back to
        # the main loop before awaiting the user-supplied callback.
        cb = self._alert_cb
        loop = self._loop
        if cb is None or loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._fire(cb, alert), loop=loop)
        )

    @staticmethod
    async def _fire(cb: Any, alert: DriftAlert) -> None:
        try:
            result = cb(alert)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.warning("calibration_monitor: alert callback raised", exc_info=True)

    # -- Pair construction ---------------------------------------------------

    def _latest_records(self) -> dict[str, PredictionRecord]:
        """Dedup the prediction log to one record per market_ticker.

        The prediction log appends on every scan tick that produces a passed
        signal, so the same market accumulates many records with an identical
        calibrated_prob and the same eventual outcome. Treating each as an
        independent Bernoulli observation inflates n, narrows Wilson CIs, and
        generates false-positive Brier alerts. We keep the latest-timestamp
        record so the most up-to-date forecast is what's scored.
        """
        latest: dict[str, PredictionRecord] = {}
        for rec in self._log.iter_all():
            prev = latest.get(rec.market_ticker)
            # >= so that on a timestamp tie (rare; only when two appends share a
            # microsecond), the later-appended record wins — matching "latest
            # timestamp wins" intent under append-order iteration.
            if prev is None or rec.timestamp >= prev.timestamp:
                latest[rec.market_ticker] = rec
        return latest

    def _build_pairs(
        self, settlements: dict[tuple[str, date], _SettlementEntry]
    ) -> list[SettledPair]:
        """Legacy NWS-derived path. See ``_build_pairs_kalshi`` for the default."""
        latest = self._latest_records()
        out: list[SettledPair] = []
        for rec in latest.values():
            pair = _record_to_pair(rec, settlements)
            if pair is not None:
                out.append(pair)
        logger.debug(
            "calibration_monitor: nws path, %d unique markets, %d settled pairs",
            len(latest),
            len(out),
        )
        return out

    def _build_pairs_kalshi(
        self,
        outcomes: dict[str, int],
        settlements: dict[tuple[str, date], _SettlementEntry] | None = None,
    ) -> list[SettledPair]:
        """Kalshi-authoritative pair builder (F8.1).

        Joins the deduped prediction stream against the ``market_ticker → 0|1``
        outcome map produced by ``_load_kalshi_outcomes``. Predictions whose
        ticker isn't yet in the outcome map (market still open, or settlement
        not yet fetched into the file) are skipped, NOT crashed — Brier
        denominator just shrinks by the unmatched count.

        ``settlements`` is consumed only for the D-1 persistence-baseline
        lookup; outcomes themselves remain Kalshi-authoritative.
        """
        latest = self._latest_records()
        out: list[SettledPair] = []
        for rec in latest.values():
            pair = _record_to_pair_kalshi(rec, outcomes, settlements)
            if pair is not None:
                out.append(pair)
        logger.debug(
            "calibration_monitor: kalshi path, %d unique markets, "
            "%d kalshi-settled tickers, %d settled pairs",
            len(latest),
            len(outcomes),
            len(out),
        )
        return out

    # -- Alert evaluation ----------------------------------------------------

    @staticmethod
    def _mixed_kind_30d_cells(snap: CalibrationSnapshot) -> set[tuple[str, str]]:
        """Return the set of (city, var) cells whose 30d window contains
        more than one ``live_calibrator_kind``.

        Implementation detail: ``per_city_var_30d`` carries per-kind rows
        plus an aggregate ``"all"`` row per cell. A cell is mixed-kind iff
        it has ≥ 2 per-kind (non-``"all"``) rows.
        """
        per_cell_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
        for r in snap.per_city_var_30d:
            if r.live_calibrator_kind == "all":
                continue
            per_cell_kinds[(r.city, r.daily_variable)].add(r.live_calibrator_kind)
        return {cell for cell, kinds in per_cell_kinds.items() if len(kinds) >= 2}

    # Freshness horizon for the corpus_dark check — matches the eval_audit
    # staleness tripwire's intuition (audit_tripwires): a scan pipeline that
    # is alive writes eval rows far more often than this.
    _CORPUS_STREAM_FRESH_SECONDS = 3 * 3600.0

    def _check_corpus_dark(self, n_30d: int, now: datetime) -> DriftAlert | None:
        """Semantic corpus-liveness alert (see __init__ docstring note).

        Fires WARNING when the 30d pair count transitions >0 → 0 while the
        backing stream file is fresh. Regression guard for the 2026-06-16
        incident class: the corpus went dark for 18 days with every
        file-freshness tripwire green.
        """
        if self._corpus_stream_path is None:
            return None
        if n_30d > 0 or not self._last_n_pairs_30d:
            return None
        try:
            mtime = self._corpus_stream_path.stat().st_mtime
        except OSError:
            return None  # stream missing/stale → the file tripwire owns it
        if now.timestamp() - mtime > self._CORPUS_STREAM_FRESH_SECONDS:
            return None
        code = "corpus_dark"
        now_ts = now.timestamp()
        if not self._gate.should_fire(code, AlertLevel.WARNING, now_ts):
            return None
        self._gate.record(code, AlertLevel.WARNING, now_ts)
        return DriftAlert(
            code=code,
            severity=AlertLevel.WARNING,
            message=(
                f"Calibration corpus went dark: 30d pairs "
                f"{self._last_n_pairs_30d} → 0 while "
                f"{self._corpus_stream_path.name} is still fresh. Either the "
                "monitor's strategy/schema join broke, or the settlements "
                "input (kalshi settlements file / NWS snapshots) went empty — "
                "check both sides."
            ),
            detail={
                "last_n_pairs_30d": self._last_n_pairs_30d,
                "stream": str(self._corpus_stream_path),
            },
            owner=None,
            ack_key=None,
        )

    def _evaluate_alerts(self, snap: CalibrationSnapshot) -> list[DriftAlert]:
        alerts: list[DriftAlert] = []
        now_ts = snap.generated_at.timestamp()
        cfg = self._config

        # -- Brier 7d vs 30d --------------------------------------------------
        # When any per-(city, var) cell has multiple calibrator kinds inside
        # the 30d window (e.g. a recent isotonic→EMOS cutover), the global
        # 7d/30d Brier ratio mixes apples with oranges and the alert becomes
        # uninterpretable. Suppress it in that case and log the suppression so
        # it's not silent.
        mixed_kind_cells = self._mixed_kind_30d_cells(snap)
        if (
            snap.brier_30d is not None
            and snap.brier_7d is not None
            and snap.brier_30d > 0
            and snap.n_pairs_7d >= cfg.bin_min_n
        ):
            ratio = snap.brier_7d / snap.brier_30d
            code = "brier_7d_vs_30d"
            severity: AlertLevel | None = None
            if ratio >= cfg.brier_crit_mult:
                severity = AlertLevel.CRITICAL
            elif ratio >= cfg.brier_warn_mult:
                severity = AlertLevel.WARNING

            if severity and mixed_kind_cells:
                # Cutover transient — don't fire, but make the suppression
                # visible so operators know the tripwire is intentionally
                # blind, not broken.
                logger.info(
                    "calibration_monitor: brier_7d_vs_30d suppressed "
                    "(suppression_reason=mixed_kind_30d) "
                    "ratio=%.2f cells=%s n_7d=%d n_30d=%d",
                    ratio,
                    sorted(mixed_kind_cells),
                    snap.n_pairs_7d,
                    snap.n_pairs_30d,
                )
            elif severity and self._gate.should_fire(code, severity, now_ts):
                self._gate.record(code, severity, now_ts)
                alerts.append(
                    DriftAlert(
                        code=code,
                        severity=severity,
                        message=(
                            f"Calibration drift: 7d Brier {snap.brier_7d:.3f} "
                            f"is {ratio:.2f}× 30d baseline {snap.brier_30d:.3f} "
                            f"(n7={snap.n_pairs_7d})"
                        ),
                        detail={
                            "brier_7d": snap.brier_7d,
                            "brier_30d": snap.brier_30d,
                            "ratio": ratio,
                            "n_7d": snap.n_pairs_7d,
                            "n_30d": snap.n_pairs_30d,
                        },
                        owner="weather_edge",
                        ack_key=None,
                    )
                )
            elif severity is None:
                # Recovery: clear any latched state so the next regression
                # fires cleanly instead of being suppressed.
                self._gate.clear(code)

        # -- Bin deviation ----------------------------------------------------
        # Absolute per-bin miscalibration detector (no baseline). When the
        # corpus belongs to a non-trading / shadow-only strategy, or carries a
        # known persistent low-p miss, this fires every cooldown forever with
        # nothing to action — so it's gated behind ``bin_deviation_enabled``.
        worst_bin: ReliabilityBin | None = None
        worst_deviation = 0.0
        for b in snap.reliability_30d:
            if b.n < cfg.bin_min_n:
                continue
            dev = abs(b.mean_observed - b.mean_predicted)
            if dev > worst_deviation:
                worst_bin = b
                worst_deviation = dev

        if (
            cfg.bin_deviation_enabled
            and worst_bin is not None
            and worst_deviation >= cfg.bin_deviation_threshold
        ):
            code = "bin_deviation"
            severity = (
                AlertLevel.CRITICAL
                if worst_deviation >= 2 * cfg.bin_deviation_threshold
                else AlertLevel.WARNING
            )
            if self._gate.should_fire(code, severity, now_ts):
                self._gate.record(code, severity, now_ts)
                alerts.append(
                    DriftAlert(
                        code=code,
                        severity=severity,
                        message=(
                            f"Reliability bin {worst_bin.label} is off: "
                            f"predicted {worst_bin.mean_predicted:.2f} vs "
                            f"observed {worst_bin.mean_observed:.2f} "
                            f"(n={worst_bin.n}, 95% CI "
                            f"[{worst_bin.wilson_low:.2f}, {worst_bin.wilson_high:.2f}])"
                        ),
                        detail={
                            "bin_lower": worst_bin.lower,
                            "bin_upper": worst_bin.upper,
                            "n": worst_bin.n,
                            "mean_predicted": worst_bin.mean_predicted,
                            "mean_observed": worst_bin.mean_observed,
                            "wilson_low": worst_bin.wilson_low,
                            "wilson_high": worst_bin.wilson_high,
                        },
                        owner="weather_edge",
                        ack_key=None,
                    )
                )

        # -- Per-city-var Brier ----------------------------------------------
        # Per-cell thresholds apply PER KIND, not on the aggregate ``all``
        # row. The aggregate exists for backwards-compat; firing the alert on
        # it would pool isotonic + EMOS together, masking which calibrator is
        # bleeding. Filter aggregate rows out of the alert iteration.
        bad_city_vars = [
            r
            for r in snap.per_city_var_30d
            if r.live_calibrator_kind != "all"
            and r.n >= cfg.city_var_min_n
            and r.brier >= cfg.city_var_brier_threshold
        ]
        if bad_city_vars:
            code = "city_var_brier"
            severity = AlertLevel.WARNING
            if self._gate.should_fire(code, severity, now_ts):
                self._gate.record(code, severity, now_ts)
                lines = [
                    f"{r.city}/{r.daily_variable} ({r.live_calibrator_kind}): "
                    f"Brier={r.brier:.3f} pred={r.mean_predicted:.2f} "
                    f"obs={r.mean_observed:.2f} n={r.n}"
                    for r in bad_city_vars
                ]
                alerts.append(
                    DriftAlert(
                        code=code,
                        severity=severity,
                        message=(
                            "Per-city-var Brier above threshold "
                            f"({cfg.city_var_brier_threshold:.2f}):\n" + "\n".join(lines)
                        ),
                        detail={
                            "city_vars": [
                                {
                                    "city": r.city,
                                    "daily_variable": r.daily_variable,
                                    "live_calibrator_kind": r.live_calibrator_kind,
                                    "n": r.n,
                                    "brier": r.brier,
                                    "mean_predicted": r.mean_predicted,
                                    "mean_observed": r.mean_observed,
                                }
                                for r in bad_city_vars
                            ]
                        },
                        owner="weather_edge",
                        ack_key=None,
                    )
                )

        return alerts

    # -- Persistence ---------------------------------------------------------

    def _restore_state(self) -> None:
        try:
            raw = json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "calibration_monitor: state file unreadable, starting fresh",
                exc_info=True,
            )
            return
        self._gate.load_state(raw.get("alerts") or {})
        last_n = raw.get("last_n_pairs_30d")
        self._last_n_pairs_30d = int(last_n) if last_n is not None else None
        health = raw.get("health") or {}
        self._health.total_ticks = int(health.get("total_ticks", 0))
        self._health.total_errors = int(health.get("total_errors", 0))

    def _persist_state(self) -> None:
        payload = {
            "alerts": self._gate.dump_state(),
            "last_n_pairs_30d": self._last_n_pairs_30d,
            "health": {
                "total_ticks": self._health.total_ticks,
                "total_errors": self._health.total_errors,
                "last_successful_tick_ts": (
                    self._health.last_successful_tick_ts.isoformat()
                    if self._health.last_successful_tick_ts
                    else None
                ),
                "last_error": self._health.last_error,
            },
            "snapshot_generated_at": (
                self._latest.generated_at.isoformat() if self._latest else None
            ),
        }
        atomic_write_text(self._state_path, json.dumps(payload, separators=(",", ":")))
