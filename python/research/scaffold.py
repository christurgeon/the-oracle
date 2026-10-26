"""Intraday research scaffold — hermetic substrate for offline workstreams.

This module provides the shared load → join → EMOS-recompute → outcome-derive →
optional gate-stack substrate consumed by serial offline research workstreams
(Gate audit, Lens 6 persistence baseline, Lens 1 market-implied distribution).
It wraps the production helpers `emos_probability`,
`_calibration_common.load_snapshot_data`, and
`outcomes.derive_outcome_yes` without re-implementing them, and mirrors the
production gate stack from `weather_evaluation.py` as a set of composable
pure-function gates.

Hermetic discipline (spec §2.1): this module MUST NOT be imported by — and
MUST NOT import from — `python.orchestrator`, `python.strategies.*`,
`python.risk.*`, `python.execution.*`, `python.notifications.*`,
`python.dashboard.*`, or `python.utils.*`. Allowed external imports are
limited to stdlib, numpy, scipy, `python.data.emos`,
`python.data.bias_correction`, `python.analytics.outcomes`,
`python.analytics.fees`, `python.analytics.bracket_metrics`, and
`scripts._calibration_common`.

"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import StrEnum
from math import sqrt
from pathlib import Path
from typing import Any, Literal

from python.analytics.outcomes import derive_outcome_yes
from python.data.bias_correction import BiasTable
from python.data.emos import EmosCoefficients, EmosTable, emos_probability
from scripts._calibration_common import (
    _CHI_SNAPSHOT_FIX_DATE,
    build_production_mirrored_pairs_with_provenance,
    load_snapshot_data,
    parse_market_date,
)


class SkipReason(StrEnum):
    """Categorical reasons rows are skipped during iteration (matches §5 precedence table).

    Emitted via the optional `on_skip` callback so workstreams can stratify skip
    distributions in their findings docs. See spec §4.6 + §5.
    """

    NO_FORECAST = "no_forecast"           # forecasts.json missing this (city, var, fc_date)
    NO_SETTLEMENT = "no_settlement"       # allow_preliminary=False and merge yields nothing
    NO_EMOS_COEFFS = "no_emos_coeffs"     # cell not in EmosTable; require_emos=True
    EMOS_OUT_OF_HORIZON = "emos_oot"      # lead > 2 days; EMOS won't compute
    MARKET_BAD_SHAPE = "market_bad_shape" # strike_type invalid or floor/cap missing
    SNAPSHOT_MISSING_FILE = "snapshot_missing_file"  # markets.json / weather_forecasts.json absent
    SNAPSHOT_PARSE_ERROR = "snapshot_parse_error"     # JSON decode failed
    CHI_PRE_FIX_DATE = "chi_pre_fix"      # known bad-coord period; matches _calibration_common
    FORECAST_THIN_MEMBERS = "thin_members"  # len(members) < 25 (production gate)
    SAME_DAY_SUPPRESSED = "same_day_suppressed"  # v1.1 (A1): allow_same_day_settlement=False


GateName = Literal[
    "excluded_cell_gate",
    "cal_floor",
    "plateau_filter",
    "raw_prob_gate",
    "strike_distance",
    "spread_filter",
    "threshold_raw_prob_clamp",
    "calibrator_disagreement",
    "edge_threshold_floor",
]


LiveCalibratorKind = Literal[
    "isotonic",
    "emos_raw",
    "isotonic_fallback",
    "identity",
]


# Single source of truth for canonical gate order. `apply_gate_stack` (M5)
# iterates this tuple and evaluates only the gates present in the caller's
# `gates` frozenset; first failure wins. Do NOT inline ordering literals
# anywhere else — production line-by-line alignment lives here per spec §4.5.
_canonical_gate_order: tuple[GateName, ...] = (
    "excluded_cell_gate",       # production GATE: excluded_city_vars (position #0)
    "cal_floor",                # production GATE: min_calibrated_prob floor
    "plateau_filter",           # production GATE: block_raw_prob_range plateau
    "raw_prob_gate",            # production GATE: between-only min raw prob
    "strike_distance",          # production GATE: threshold z-score floor
    "spread_filter",            # production GATE: max_spread
    "threshold_raw_prob_clamp", # production GATE: threshold raw-prob clamp
    "calibrator_disagreement",  # production GATE: calibrator disagreement
    "edge_threshold_floor",     # post-disagreement; production-order tail
)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One fully-joined (snapshot, market, forecast, settlement, posterior)
    record. See spec §3.2 for field semantics and §3.2.1 for the
    production-mirrored vs single-model raw-prob invariant table.
    """

    # --- Identity -------------------------------------------------------------
    snapshot_day: date
    run_id: str
    snapshot_ts: datetime
    city: str
    var: Literal["high", "low"]
    daily_variable: str
    forecast_date: date
    event_ticker: str
    market_ticker: str
    strike_type: Literal["greater", "less", "between"]
    floor_strike: float | None
    cap_strike: float | None

    # --- Forecast inputs (ddof=1 sample moments; see spec §6.3) --------------
    ens_mean: float
    ens_std: float
    n_members: int
    bracket_raw_prob_production_mirrored: float
    bracket_raw_prob_single_model: float

    # --- EMOS posterior -------------------------------------------------------
    emos_mu: float
    emos_sigma: float
    emos_prob_bracket: float | None
    emos_coeffs_hash: str
    emos_fit_date: str
    emos_lead: int
    live_calibrator_kind: LiveCalibratorKind

    # --- Market state ---------------------------------------------------------
    yes_bid: float | None
    yes_ask: float | None
    yes_bid_size: int | None
    yes_ask_size: int | None

    # --- Settlement -----------------------------------------------------------
    settled_temp_f: float | None
    settled_is_preliminary: bool | None
    outcome_yes: bool | None
    pending_settlement: bool

    # --- Gate-stack verdict (populated iff caller passed gates=) -------------
    rejection_reason: str | None
    rejection_payload: dict[str, float] | None

    # --- Lead / temporal ------------------------------------------------------
    lead_days: int


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Production gate constants. Defaults match python/config.yaml.
    Override per workstream if you want to vary one constant.
    Mirrors python/config.yaml fields exactly; no new knobs."""

    # Cal-floor. None = gate disabled.
    min_calibrated_prob: float | None = 0.10

    # Plateau filter. None = gate disabled.
    plateau_range: tuple[float, float] | None = (0.649, 0.967)

    # Raw-prob-between. Applies only to between markets.
    min_raw_prob_for_between: float = 0.02

    # Strike-distance. Applies only to threshold markets.
    # The scaffold does not have access to pooled_median / pooled_std
    # by default — workstreams must compute z-score outside and check
    # against this threshold themselves, OR opt out of strike_distance
    # in their `gates` frozenset.
    min_distance_from_strike_std: float = 0.0

    # Spread filter. Match production max_spread; the gate
    # rejects when market_spread > max_spread.
    max_spread: float = 0.10

    # Threshold raw-prob clamp. For greater/less markets,
    # rejects when raw < min OR raw > max. Hardcoded in production
    # at [0.05, 0.95]; exposed as config here for completeness.
    threshold_raw_prob_clamp: tuple[float, float] | None = (0.05, 0.95)

    # Edge-threshold floor (v1.1 A2). Off by default. When set, the
    # gate rejects rows where the entry-side edge against market_mid
    # falls below the floor. Workstreams that want entry-side
    # filtering enable this in `gates`; it is NOT in the default
    # frozenset because most analyses want to see sub-threshold
    # rows too.
    min_edge: float | None = None

    # (Excluded-cell gate has no config constants — it's a tag flip
    # driven by iter_evaluation_rows.excluded_cell_vars.)


@dataclass(frozen=True, slots=True)
class ScaffoldEnvelope:
    """Reproducibility envelope per spec §6.2. Workstreams serialize this
    into findings JSON and parquet headers; the three SHAs together pin the
    result of an `iter_evaluation_rows` invocation.
    """

    scaffold_version: str
    snapshots_dir_sha: str
    emos_artifact_hash: str
    emos_fit_date: str
    scaffold_module_sha: str
    lead_source: str
    gate_config_repr: str | None


def iter_evaluation_rows(
    snapshots_dir: Path,
    *,
    emos_table: EmosTable,
    ensemble_weights: dict[str, float] | None = None,
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
    bias_table: BiasTable | None = None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
    model: str = "gfs_seamless",
    cell_filter: frozenset[str] | None = None,
    var_filter: frozenset[str] | None = None,
    date_filter: Callable[[date], bool] | None = None,
    allow_same_day_settlement: bool = False,
    allow_preliminary: bool = False,
    require_emos: bool = False,
    excluded_cell_vars: frozenset[str] | None = None,
    lead_source: Literal[
        "utc_today", "snapshot_day_utc", "snapshot_day_local"
    ] = "snapshot_day_utc",
    gates: frozenset[GateName] | None = None,
    gate_config: GateConfig | None = None,
    on_skip: Callable[[SkipReason, Mapping[str, Any]], None] | None = None,
) -> Iterator[EvaluationRow]:
    """Stream `EvaluationRow` per (snapshot_day, run_id, market_ticker) triple.

    Gate evaluation (M5): when `gates is not None`, each emitted row is
    annotated with `(rejection_reason, rejection_payload)` from a single
    `apply_gate_stack` call using the row's production-mirrored raw prob
    (v1.1 A6) and the row's `live_calibrator_kind` tag as the caller's
    claim (v1.1 A8). Rejected rows are STILL yielded (spec §4.1
    "applied but yielded" semantic). The iterator passes
    `strike_abs_z=None`; workstreams that want strike-distance gating
    call `apply_gate_stack` themselves after the iterator emits.

    Forecast-per-snapshot fidelity (consumer-visible semantic):
    `load_snapshot_data` merges weather forecasts by
    `(city, forecast_date, model, daily_variable)` last-writer-wins
    (`scripts/_calibration_common.py:120`). Iterating multiple snapshot
    days that all carry the same forecast_date therefore yields the same
    `bracket_raw_prob_production_mirrored` and
    `bracket_raw_prob_single_model` for every snapshot day — those fields
    reflect the LATEST available forecast per fc_date, not the
    forecast that was live at each snapshot's `snapshot_ts`. Production-
    mirrored prob is also deduped at the ticker level inside
    `build_production_mirrored_pairs_with_provenance`, so the
    same ticker emitted across multiple snapshot days carries one
    identical raw prob.

    This is an inherited property of the production helpers, not a
    scaffold bug. It is the right semantic for daily-horizon refit
    analytics. Workstreams that depend on per-snapshot forecast fidelity
    (e.g. genuinely intraday signals at sub-day resolution) must load
    each run's `weather_forecasts.json` directly rather than relying on
    the iterator's merged view.

    Out-of-horizon identity-fallback asymmetry (deliberate, deferred
    decision for workstream-3): when `require_emos=False`,
    - **cell ABSENT from emos_table** → row emitted with
      `live_calibrator_kind="identity"`, `emos_prob_bracket =
      bracket_raw_prob_production_mirrored`, `emos_mu = ens_mean`,
      `emos_sigma = max(ens_std, 1e-6)`.
    - **cell PRESENT but lead > 2 (out of horizon)** → row SKIPPED
      with `SkipReason.EMOS_OUT_OF_HORIZON`, regardless of
      `require_emos`.

    Both behaviors are spec-compliant (§3.2 `emos_lead` docstring
    accepts either "emit with None" or "skip"). The chosen split costs
    Lens 1 coverage for cells-with-EMOS that happen to be 3+ days out,
    since those rows are dropped rather than emitted as identity-
    fallback. Confirm/change this asymmetry before workstream-3 (Lens 1
    market-implied distribution) dispatch; for workstream-1 (gate audit,
    short leads) and workstream-2 (Lens 6 with `require_emos=True`,
    skip is correct) it does not matter.
    """
    if cell_filter is not None and len(cell_filter) == 0:
        raise ValueError(
            "cell_filter=frozenset() excludes every cell. Use None for 'no filter'."
        )
    if var_filter is not None and len(var_filter) == 0:
        raise ValueError(
            "var_filter=frozenset() excludes every variable. Use None for 'no filter'."
        )

    _validate_filters(
        cell_filter=cell_filter,
        var_filter=var_filter,
        excluded_cell_vars=excluded_cell_vars,
        emos_table=emos_table,
    )

    forecasts, settlements, _markets_all = load_snapshot_data(snapshots_dir)

    production_mirrored_prob_by_ticker: dict[str, float] = {}
    if ensemble_weights is not None or bias_table is not None:
        effective_weights = ensemble_weights or {model: 1.0}
        for record in build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts,
            settlements=settlements,
            markets=_markets_all,
            bias_table=bias_table,
            ensemble_weights=effective_weights,
            spread_inflation_overrides=spread_inflation_overrides,
            bias_shrinkage=bias_shrinkage,
            bias_max_correction_f=bias_max_correction_f,
        ):
            raw_prob, _outcome, _city, _var, _strike, _fc_date, _models, _bc, ticker = record
            production_mirrored_prob_by_ticker[ticker] = float(raw_prob)

    effective_gate_config = gate_config or GateConfig()

    for snapshot_day, run_id, run_dir in _iter_snapshot_dirs(snapshots_dir):
        for mkt in _iter_markets_for_run(run_dir, on_skip):
            row = _build_row(
                mkt=mkt,
                snapshot_day=snapshot_day,
                run_id=run_id,
                forecasts=forecasts,
                settlements=settlements,
                emos_table=emos_table,
                model=model,
                cell_filter=cell_filter,
                var_filter=var_filter,
                date_filter=date_filter,
                allow_same_day_settlement=allow_same_day_settlement,
                allow_preliminary=allow_preliminary,
                require_emos=require_emos,
                lead_source=lead_source,
                production_mirrored_prob_by_ticker=production_mirrored_prob_by_ticker,
                ensemble_weights=ensemble_weights,
                bias_table=bias_table,
                on_skip=on_skip,
            )
            if row is None:
                continue
            if gates is not None:
                row = _annotate_gate_verdict(
                    row,
                    gates=gates,
                    gate_config=effective_gate_config,
                    excluded_cell_vars=excluded_cell_vars,
                )
            yield row


def _validate_filters(
    *,
    cell_filter: frozenset[str] | None,
    var_filter: frozenset[str] | None,
    excluded_cell_vars: frozenset[str] | None,
    emos_table: EmosTable,
) -> None:
    """Construction-time filter validation per spec §4.1 (A14 + B2).

    Raises `ValueError` with messages pinned to the spec so workstreams that
    typo a filter see a useful failure mode rather than a silent zero-match.
    """
    if excluded_cell_vars is not None:
        bad_keys = sorted(k for k in excluded_cell_vars if "/" in k)
        if bad_keys:
            raise ValueError(
                "excluded_cell_vars uses underscore form ('DEN_high') to mirror "
                f"production at weather_evaluation.py:591; got slash form: {bad_keys}. "
                "EMOS cells use slash form ('DEN/high'); only excluded_cell_vars "
                "uses underscore."
            )

    if cell_filter is not None and len(cell_filter) > 0:
        if not (cell_filter & set(emos_table.cells.keys())):
            raise ValueError(
                f"cell_filter={cell_filter!r} matches no cells in emos_table; "
                "did you mean MIA/high (lowercase)?"
            )

    if var_filter is not None and len(var_filter) > 0:
        if not (var_filter & {"high", "low"}):
            raise ValueError(
                f"var_filter={var_filter!r} matches no vars in emos_table; "
                "expected 'high' or 'low' (lowercase)."
            )


def _iter_snapshot_dirs(snapshots_dir: Path) -> Iterator[tuple[date, str, Path]]:
    """Walk both flat (`<date>/markets.json`) and nested (`<date>/<run_id>/`)
    layouts in `(snapshot_day asc, run_id asc)` order.

    For flat layout, `run_id = "flat"` (sentinel matching the
    `_calibration_common.py` build_emos_corpus convention).
    """
    if not snapshots_dir.exists():
        return
    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        try:
            snapshot_day = date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if (day_dir / "markets.json").exists():
            yield (snapshot_day, "flat", day_dir)
            continue
        for run_dir in sorted(day_dir.iterdir()):
            if run_dir.is_dir():
                yield (snapshot_day, run_dir.name, run_dir)


def _iter_markets_for_run(
    run_dir: Path,
    on_skip: Callable[[SkipReason, Mapping[str, Any]], None] | None,
) -> Iterator[dict[str, Any]]:
    """Yield market records from `<run_dir>/markets.json` sorted by ticker.

    Forecasts and settlements are loaded by the caller via
    `load_snapshot_data` (called once per `iter_evaluation_rows`
    invocation); this helper only reads `markets.json` to preserve
    per-run-dir provenance (`snapshot_day`, `run_id`) which the
    merged loader discards.
    """
    mkt_file = run_dir / "markets.json"
    if not mkt_file.exists():
        if on_skip is not None:
            on_skip(SkipReason.SNAPSHOT_MISSING_FILE, {"path": str(mkt_file)})
        return
    try:
        raw = json.loads(mkt_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if on_skip is not None:
            on_skip(
                SkipReason.SNAPSHOT_PARSE_ERROR,
                {"path": str(mkt_file), "error": str(exc)},
            )
        return
    yield from sorted(raw, key=lambda m: str(m.get("ticker", "")))


def _build_raw_prob_pair(
    forecast: dict[str, Any],
    *,
    floor_strike: float | None,
    cap_strike: float | None,
    strike_type: str,
    ticker: str,
    ensemble_weights: dict[str, float] | None,
    bias_table: BiasTable | None,
    production_mirrored_prob_by_ticker: Mapping[str, float],
) -> tuple[float, float, float, float, int]:
    """Return `(ens_mean_ddof1, ens_std_ddof1, prob_prod_mirrored, prob_gfs_single, n_members)`.

    Spec §6.3 / A9 — `ens_mean` / `ens_std` are recomputed `ddof=1`
    from `forecast["temperatures"]`. NEVER read the snapshot's persisted
    `mean` / `std_dev` (those are Open-Meteo `ddof=0`).

    Spec §3.2.1 — the production-mirrored column equals the GFS-single
    column when both `ensemble_weights` and `bias_table` are None;
    otherwise it is looked up from the pre-built map produced by
    `build_production_mirrored_pairs_with_provenance`.
    """
    members = list(forecast.get("temperatures") or forecast.get("ensemble_members") or [])
    n_members = len(members)
    if n_members == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    s = 0.0
    for m in members:
        s += float(m)
    ens_mean = s / n_members

    # Single-model GFS ddof=1 by design. This is the audit-side spread
    # used to feed the EMOS posterior (matches the EMOS-corpus
    # statistics.stdev convention). Production's strike-distance gate
    # pools GFS+ECMWF with ddof=0 via statistics.pstdev — a deliberate
    # downstream-multiplier asymmetry that only reaches effective_edge,
    # NOT emos_probability. See scripts/verify_intraday_replay.py
    # _classify_effective_edge for the empirical envelope (max
    # |Δ effective_edge| < 0.001 on the v6 corpus).
    if n_members >= 2:
        var_sum = 0.0
        for m in members:
            d = float(m) - ens_mean
            var_sum += d * d
        ens_std = sqrt(var_sum / (n_members - 1))
    else:
        ens_std = 0.0

    if strike_type == "greater":
        if floor_strike is None:
            prob_gfs_single = 0.0
        else:
            # Kalshi rule: t > floor (STRICT). Matches production
            # EnsembleForecast.fraction_above and derive_outcome_yes.
            prob_gfs_single = sum(1 for m in members if float(m) > floor_strike) / n_members
    elif strike_type == "less":
        if cap_strike is None:
            prob_gfs_single = 0.0
        else:
            prob_gfs_single = sum(1 for m in members if float(m) < cap_strike) / n_members
    else:  # between — CLOSED [floor, cap] to match Kalshi settlement
        # convention (right-inclusive, verified 2026-05-19) and production
        # EnsembleForecast.fraction_between. Outcome is derived via
        # python.analytics.outcomes.derive_outcome_yes.
        if floor_strike is None or cap_strike is None:
            prob_gfs_single = 0.0
        else:
            prob_gfs_single = sum(
                1 for m in members if floor_strike <= float(m) <= cap_strike
            ) / n_members

    if ensemble_weights is None and bias_table is None:
        prob_prod_mirrored = prob_gfs_single
    else:
        prob_prod_mirrored = production_mirrored_prob_by_ticker.get(ticker, prob_gfs_single)

    return ens_mean, ens_std, prob_prod_mirrored, prob_gfs_single, n_members


def _compute_lead_days(
    forecast_date: date,
    snapshot_day_utc: date,
    *,
    lead_source: str,
) -> int:
    """Pure scalar — `(forecast_date - <ref_day>).days`, unclamped (v1.2 B10).

    `snapshot_day_utc` (default): static, replay-deterministic.
    `utc_today`: production semantic from `weather_evaluation.evaluate_market_pure`;
        not bit-identically replayable.
    `snapshot_day_local`: ET-anchored; deferred to a later milestone.
    """
    if lead_source == "snapshot_day_utc":
        return (forecast_date - snapshot_day_utc).days
    if lead_source == "utc_today":
        return (forecast_date - datetime.now(UTC).date()).days
    if lead_source == "snapshot_day_local":
        raise NotImplementedError(
            "lead_source='snapshot_day_local' deferred — needs a local-tz "
            "anchor the scaffold does not yet plumb (M4 / M8 follow-up)."
        )
    raise ValueError(f"unknown lead_source: {lead_source!r}")


_VAR_LABEL_FROM_DAILY: dict[str, Literal["high", "low"]] = {
    "temperature_2m_max": "high",
    "temperature_2m_min": "low",
}


def _build_row(
    *,
    mkt: dict[str, Any],
    snapshot_day: date,
    run_id: str,
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    emos_table: EmosTable,
    model: str,
    cell_filter: frozenset[str] | None,
    var_filter: frozenset[str] | None,
    date_filter: Callable[[date], bool] | None,
    allow_same_day_settlement: bool,
    allow_preliminary: bool,
    require_emos: bool,
    lead_source: str,
    production_mirrored_prob_by_ticker: Mapping[str, float],
    ensemble_weights: dict[str, float] | None,
    bias_table: BiasTable | None,
    on_skip: Callable[[SkipReason, Mapping[str, Any]], None] | None,
) -> EvaluationRow | None:
    """Build a single `EvaluationRow` or return None on skip.

    Skip precedence follows spec §5 (rows 3-10). Rows 0-2 (file-level) are
    handled in `_iter_markets_for_run`. Pure-inclusion filters
    (`cell_filter`/`var_filter`/`date_filter`) skip silently — they are
    not in the §5 precedence table.
    """
    ticker = mkt.get("ticker")
    strike_type = mkt.get("strike_type")
    city = mkt.get("city")
    daily_variable = mkt.get("daily_variable")
    floor_strike = mkt.get("floor_strike")
    cap_strike = mkt.get("cap_strike")

    if not isinstance(ticker, str):
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": None,
                    "reason": "ticker_missing_or_not_string",
                },
            )
        return None

    fc_date = parse_market_date(ticker)

    if isinstance(city, str) and fc_date is not None and city == "CHI" \
            and fc_date < _CHI_SNAPSHOT_FIX_DATE:
        if on_skip is not None:
            on_skip(
                SkipReason.CHI_PRE_FIX_DATE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": None,
                    "forecast_date": str(fc_date),
                },
            )
        return None

    if strike_type not in ("greater", "less", "between"):
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "strike_type": strike_type,
                    "reason": "invalid_strike_type",
                },
            )
        return None

    # Strike↔floor/cap consistency. `_build_raw_prob_pair` returns 0.0 on
    # missing strike (graceful), but `recompute_emos_posterior` delegates to
    # `emos_probability` which raises ValueError on every shape mismatch
    # (`python/data/emos.py`). Spec §5 invariant: "scaffold never raises
    # on bad input data." Short-circuit to MARKET_BAD_SHAPE before EMOS runs.
    strike_shape_error: str | None = None
    if strike_type == "greater":
        if floor_strike is None:
            strike_shape_error = "greater_missing_floor_strike"
        elif cap_strike is not None:
            strike_shape_error = "greater_forbids_cap_strike"
    elif strike_type == "less":
        if cap_strike is None:
            strike_shape_error = "less_missing_cap_strike"
        elif floor_strike is not None:
            strike_shape_error = "less_forbids_floor_strike"
    else:  # between
        if floor_strike is None or cap_strike is None:
            strike_shape_error = "between_missing_strike"
    if strike_shape_error is not None:
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "strike_type": strike_type,
                    "floor_strike": floor_strike,
                    "cap_strike": cap_strike,
                    "reason": strike_shape_error,
                },
            )
        return None

    if not isinstance(city, str) or not isinstance(daily_variable, str):
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "daily_variable": daily_variable,
                    "reason": "city_or_daily_variable_missing",
                },
            )
        return None
    var_label = _VAR_LABEL_FROM_DAILY.get(daily_variable)
    if var_label is None:
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "daily_variable": daily_variable,
                    "reason": "unknown_daily_variable",
                },
            )
        return None
    if fc_date is None:
        if on_skip is not None:
            on_skip(
                SkipReason.MARKET_BAD_SHAPE,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "reason": "ticker_date_unparseable",
                },
            )
        return None

    # Pure inclusion predicates — silent skip (not in §5 precedence table).
    if var_filter is not None and var_label not in var_filter:
        return None
    cell_key = f"{city}/{var_label}"
    if cell_filter is not None and cell_key not in cell_filter:
        return None
    if date_filter is not None and not date_filter(fc_date):
        return None

    # Precedence #5: SAME_DAY_SUPPRESSED. Pinned to snapshot_day_utc regardless
    # of lead_source (spec §4.1 / v1.2 B8). Static-corpus determinism.
    if not allow_same_day_settlement and fc_date == snapshot_day:
        if on_skip is not None:
            on_skip(
                SkipReason.SAME_DAY_SUPPRESSED,
                {
                    "snapshot_day": str(snapshot_day),
                    "snapshot_day_utc": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "forecast_date": str(fc_date),
                },
            )
        return None

    floor_f: float | None = float(floor_strike) if floor_strike is not None else None
    cap_f: float | None = float(cap_strike) if cap_strike is not None else None

    fc = forecasts.get((city, str(fc_date), daily_variable, model))
    if fc is None:
        if on_skip is not None:
            on_skip(
                SkipReason.NO_FORECAST,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "forecast_date": str(fc_date),
                    "daily_variable": daily_variable,
                    "model": model,
                },
            )
        return None

    # Precedence #6: FORECAST_THIN_MEMBERS. Matches `_calibration_common.py`.
    members = fc.get("temperatures") or fc.get("ensemble_members") or []
    if len(members) < 25:
        if on_skip is not None:
            on_skip(
                SkipReason.FORECAST_THIN_MEMBERS,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "forecast_date": str(fc_date),
                    "n_members": len(members),
                },
            )
        return None

    ens_mean, ens_std, prob_prod_mirrored, prob_gfs_single, n_members = _build_raw_prob_pair(
        fc,
        floor_strike=floor_f,
        cap_strike=cap_f,
        strike_type=strike_type,
        ticker=ticker,
        ensemble_weights=ensemble_weights,
        bias_table=bias_table,
        production_mirrored_prob_by_ticker=production_mirrored_prob_by_ticker,
    )

    lead_days = _compute_lead_days(fc_date, snapshot_day, lead_source=lead_source)

    # Precedence #7 / #7' / #8: EMOS cell-presence BEFORE lead horizon (mirrors
    # EmosTable.get short-circuit, v1.2 B3).
    cell_present = cell_key in emos_table.cells
    if not cell_present:
        if require_emos:
            if on_skip is not None:
                on_skip(
                    SkipReason.NO_EMOS_COEFFS,
                    {
                        "snapshot_day": str(snapshot_day),
                        "run_id": run_id,
                        "market_ticker": ticker,
                        "city": city,
                        "var": var_label,
                        "forecast_date": str(fc_date),
                        "emos_artifact_hash": emos_table.artifact_hash,
                        "emos_fit_date": emos_table.fit_date,
                        "lead_days_requested": lead_days,
                    },
                )
            return None
        coeffs: EmosCoefficients | None = None
    else:
        coeffs = emos_table.get(city=city, var=var_label, lead=lead_days)
        if coeffs is None:
            # Cell IS present but lead out of horizon — precedence #8.
            # TODO(v2.0 walk-forward EMOS): EmosTable.get returns None in
            # three cases (emos.py): (a) cell absent — handled
            # above via `cell_present`; (b) lead<0 or lead>2; (c) cell has
            # `by_lead` and lead not in by_lead. Today's pooled artifact
            # makes (c) unreachable (every cell uses pooled coefficients),
            # so EMOS_OUT_OF_HORIZON is semantically accurate. When walk-
            # forward (per-lead) EMOS ships, case (c) becomes a real
            # case and EMOS_OUT_OF_HORIZON will silently mislabel
            # "this lead missing from this cell's by_lead map" as "lead
            # outside the documented horizon." Split the skip reason at
            # that point (e.g. EMOS_LEAD_NOT_FIT) before the artifact
            # change lands.
            if on_skip is not None:
                on_skip(
                    SkipReason.EMOS_OUT_OF_HORIZON,
                    {
                        "snapshot_day": str(snapshot_day),
                        "run_id": run_id,
                        "market_ticker": ticker,
                        "city": city,
                        "var": var_label,
                        "forecast_date": str(fc_date),
                        "emos_artifact_hash": emos_table.artifact_hash,
                        "emos_fit_date": emos_table.fit_date,
                        "lead_days_requested": lead_days,
                    },
                )
            return None

    if coeffs is None:
        emos_prob: float | None = prob_prod_mirrored
        emos_mu: float = ens_mean
        emos_sigma: float = ens_std if ens_std > 0.0 else 1e-6
        live_calibrator_kind: LiveCalibratorKind = "identity"
    else:
        emos_floor: float | None
        emos_cap: float | None
        if strike_type == "greater":
            emos_floor, emos_cap = floor_f, None
        elif strike_type == "less":
            emos_floor, emos_cap = None, cap_f
        else:
            emos_floor, emos_cap = floor_f, cap_f
        emos_mu, emos_sigma, prob_b = recompute_emos_posterior(
            coeffs=coeffs,
            ens_mean=ens_mean,
            ens_std=ens_std,
            strike_type=strike_type,
            floor=emos_floor,
            cap=emos_cap,
        )
        emos_prob = prob_b
        live_calibrator_kind = "emos_raw"

    settlement = settlements.get((city, str(fc_date)))
    settled_temp_f: float | None = None
    settled_is_preliminary: bool | None = None
    outcome_yes: bool | None = None
    pending_settlement = settlement is None
    if settlement is not None:
        field = "high_f" if var_label == "high" else "low_f"
        raw_t = settlement.get(field)
        if raw_t is not None:
            settled_temp_f = float(raw_t)
            settled_is_preliminary = bool(settlement.get("is_preliminary", False))
            outcome_yes = derive_outcome_yes(strike_type, floor_f, cap_f, settled_temp_f)
        else:
            # Settlement record present but the relevant temp field is missing.
            pending_settlement = True

    # Precedence #9 / #10: settlement missing or preliminary; allow_preliminary
    # gates emission.
    if pending_settlement and not allow_preliminary:
        if on_skip is not None:
            on_skip(
                SkipReason.NO_SETTLEMENT,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "forecast_date": str(fc_date),
                },
            )
        return None
    if (
        settled_is_preliminary is True
        and not allow_preliminary
    ):
        if on_skip is not None:
            on_skip(
                SkipReason.NO_SETTLEMENT,
                {
                    "snapshot_day": str(snapshot_day),
                    "run_id": run_id,
                    "market_ticker": ticker,
                    "city": city,
                    "var": var_label,
                    "forecast_date": str(fc_date),
                    "is_preliminary": True,
                },
            )
        return None

    snapshot_ts_raw = mkt.get("snapshot_ts")
    if isinstance(snapshot_ts_raw, int | float):
        snapshot_ts = datetime.fromtimestamp(float(snapshot_ts_raw), tz=UTC)
    else:
        snapshot_ts = datetime.combine(snapshot_day, datetime.min.time(), tzinfo=UTC)

    event_ticker = str(mkt.get("event_ticker") or ticker.rsplit("-", 1)[0])

    yes_bid_raw = mkt.get("yes_bid_dollars")
    yes_ask_raw = mkt.get("yes_ask_dollars")
    yes_bid = float(yes_bid_raw) if yes_bid_raw is not None else None
    yes_ask = float(yes_ask_raw) if yes_ask_raw is not None else None
    yes_bid_size_raw = mkt.get("yes_bid_size")
    yes_ask_size_raw = mkt.get("yes_ask_size")
    yes_bid_size = int(yes_bid_size_raw) if yes_bid_size_raw is not None else None
    yes_ask_size = int(yes_ask_size_raw) if yes_ask_size_raw is not None else None

    return EvaluationRow(
        snapshot_day=snapshot_day,
        run_id=run_id,
        snapshot_ts=snapshot_ts,
        city=city,
        var=var_label,
        daily_variable=daily_variable,
        forecast_date=fc_date,
        event_ticker=event_ticker,
        market_ticker=ticker,
        strike_type=strike_type,
        floor_strike=floor_f,
        cap_strike=cap_f,
        ens_mean=ens_mean,
        ens_std=ens_std,
        n_members=n_members,
        bracket_raw_prob_production_mirrored=prob_prod_mirrored,
        bracket_raw_prob_single_model=prob_gfs_single,
        emos_mu=emos_mu,
        emos_sigma=emos_sigma,
        emos_prob_bracket=emos_prob,
        emos_coeffs_hash=emos_table.artifact_hash or "",
        emos_fit_date=emos_table.fit_date or "",
        emos_lead=lead_days,
        live_calibrator_kind=live_calibrator_kind,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        yes_bid_size=yes_bid_size,
        yes_ask_size=yes_ask_size,
        settled_temp_f=settled_temp_f,
        settled_is_preliminary=settled_is_preliminary,
        outcome_yes=outcome_yes,
        pending_settlement=pending_settlement,
        rejection_reason=None,
        rejection_payload=None,
        lead_days=lead_days,
    )


def recompute_emos_posterior(
    *,
    coeffs: EmosCoefficients,
    ens_mean: float,
    ens_std: float,
    strike_type: Literal["greater", "less", "between"],
    floor: float | None,
    cap: float | None,
) -> tuple[float, float, float]:
    """Return `(mu, sigma, p_bracket)`.

    Thin wrapper over `python.data.emos.emos_probability` that ALSO returns
    the Gaussian moments. `emos_probability` returns only the bracket prob;
    the moments are useful diagnostics for Lens 1 (mu_emos input) and Lens
    6 (validating sliding-window EMOS state). Wraps, does not re-implement:
    `mu` and `sigma` are computed by the same expressions as
    `emos.py`, then the prob is delegated to `emos_probability`.

    Bit-identical to production: `tests/research/test_scaffold.py` asserts
    a numerical match to `state/eval_audit.jsonl` for EMOS-live cells.

    INPUT CONTRACT (v1.1, A9): `ens_mean` and `ens_std` MUST be ddof=1
    sample moments computed from the member list (matching
    `build_emos_corpus.py` and `weather_evaluation.evaluate_market_pure`). Callers that
    pass the snapshot's persisted ddof=0 `mean`/`std_dev` will get
    silently miscomputed posteriors. The scaffold's row construction
    always recomputes ddof=1 from `temperatures[]`; this helper trusts its
    caller.
    """
    mu = coeffs.a + coeffs.b * ens_mean
    sigma = sqrt(max(coeffs.c + coeffs.d * (ens_std ** 2), 1e-12))
    prob, _, _ = emos_probability(
        coeffs, ens_mean, ens_std, strike_type=strike_type, floor=floor, cap=cap,
    )
    return mu, sigma, prob


# --- Per-gate predicates --------------------------------------------------
#
# Each predicate mirrors the production trip condition at the
# corresponding line of python/strategies/weather_evaluation.py (see
# `_canonical_gate_order`). Signature is uniform so `apply_gate_stack`
# can dispatch by name; predicates ignore the kwargs they don't need.
# Returns `(rejection_reason, payload)` on trip; `(None, None)` on pass.


def _apply_excluded_cell_gate(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: excluded_city_vars — fires when the cell is
    in `excluded_city_vars`. Spec §4.5 row #0 (v1.2 B1).
    """
    if is_excluded_cell:
        return "EXCLUDED_CELL", {
            "raw_ensemble_prob": raw_ensemble_prob,
            "model_prob": model_prob,
        }
    return None, None


def _apply_cal_floor(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: cal_floor — `model_prob < min_calibrated_prob`."""
    if config.min_calibrated_prob is not None and model_prob < config.min_calibrated_prob:
        return "CAL_FLOOR_BLOCK", {
            "model_prob": model_prob,
            "min_calibrated_prob": float(config.min_calibrated_prob),
        }
    return None, None


def _apply_plateau(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: plateau_filter — inclusive-on-both-ends range trip."""
    rng = config.plateau_range
    if rng is not None and rng[0] <= raw_ensemble_prob <= rng[1]:
        return "PLATEAU_BLOCK", {
            "raw_ensemble_prob": raw_ensemble_prob,
            "plateau_lo": float(rng[0]),
            "plateau_hi": float(rng[1]),
        }
    return None, None


def _apply_raw_prob_gate(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: raw_prob_gate — between-only floor on raw prob."""
    if strike_type == "between" and raw_ensemble_prob < config.min_raw_prob_for_between:
        return "RAW_PROB_GATE", {
            "raw_ensemble_prob": raw_ensemble_prob,
            "min_raw_prob_for_between": float(config.min_raw_prob_for_between),
        }
    return None, None


def _apply_strike_distance(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: strike_distance — threshold-market z-score floor.

    v1.2 B5: `strike_abs_z=None` is a strict no-op (pass), NOT a skip.
    The scaffold's iterator can't compute pooled z-scores; workstreams
    that need this gate pass their own `strike_abs_z`.
    """
    if strike_abs_z is None:
        return None, None
    if strike_type not in ("greater", "less"):
        return None, None
    if strike_abs_z < config.min_distance_from_strike_std:
        return "STRIKE_DISTANCE", {
            "strike_abs_z": float(strike_abs_z),
            "min_distance_from_strike_std": float(config.min_distance_from_strike_std),
        }
    return None, None


def _apply_spread(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: spread_filter — `spread > max_spread`."""
    if market_spread is not None and market_spread > config.max_spread:
        return "SPREAD", {
            "market_spread": float(market_spread),
            "max_spread": float(config.max_spread),
        }
    return None, None


def _apply_threshold_raw_prob_clamp(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: threshold_raw_prob_clamp — `raw < 0.05 OR raw > 0.95`
    on threshold markets. Reuses `RAW_PROB_GATE` rejection reason (spec §4.5).
    """
    if strike_type not in ("greater", "less"):
        return None, None
    clamp = config.threshold_raw_prob_clamp
    if clamp is None:
        return None, None
    if raw_ensemble_prob < clamp[0] or raw_ensemble_prob > clamp[1]:
        return "RAW_PROB_GATE", {
            "raw_ensemble_prob": raw_ensemble_prob,
            "threshold_clamp_lo": float(clamp[0]),
            "threshold_clamp_hi": float(clamp[1]),
        }
    return None, None


def _apply_calibrator_disagreement(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Production GATE: calibrator_disagreement — `(raw-mid)*(model-mid) < 0`.

    v1.1 A8: the bypass on `live_calibrator_kind == "emos_raw"` is keyed
    to the CALLER'S argument, not the row's emitted tag. Callers gating
    an externally-computed isotonic arm on an emos_raw cell pass
    `live_calibrator_kind="isotonic"` and the gate fires normally.
    """
    if live_calibrator_kind == "emos_raw":
        return None, None
    if market_mid is None:
        return None, None
    if (raw_ensemble_prob - market_mid) * (model_prob - market_mid) < 0:
        return "CALIBRATOR_DISAGREEMENT", {
            "raw_ensemble_prob": raw_ensemble_prob,
            "model_prob": model_prob,
            "market_mid": float(market_mid),
        }
    return None, None


def _apply_edge_threshold_floor(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None,
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Entry-side floor (spec §4.5 row #8, v1.1 A2). Off by default.

    Production uses `raw_edge = abs(model_prob - exec_price)` in
    `weather_evaluation.evaluate_market_pure`; the scaffold mirrors that with
    `market_mid` as the entry-side reference price. The "entry side"
    is implicit: the bot enters whichever side has positive edge, so
    the magnitude of `(model_prob - market_mid)` is the edge.
    """
    if config.min_edge is None:
        return None, None
    if market_mid is None:
        return None, None
    edge = abs(model_prob - market_mid)
    if edge < config.min_edge:
        return "EDGE_BELOW_FLOOR", {
            "edge": edge,
            "min_edge": float(config.min_edge),
            "model_prob": model_prob,
            "market_mid": float(market_mid),
        }
    return None, None


# Name → predicate. The iteration order is bound by `_canonical_gate_order`,
# not by dict insertion; this map is just a dispatch table.
_GATE_PREDICATES: dict[
    GateName,
    Callable[..., tuple[str | None, dict[str, float] | None]],
] = {
    "excluded_cell_gate": _apply_excluded_cell_gate,
    "cal_floor": _apply_cal_floor,
    "plateau_filter": _apply_plateau,
    "raw_prob_gate": _apply_raw_prob_gate,
    "strike_distance": _apply_strike_distance,
    "spread_filter": _apply_spread,
    "threshold_raw_prob_clamp": _apply_threshold_raw_prob_clamp,
    "calibrator_disagreement": _apply_calibrator_disagreement,
    "edge_threshold_floor": _apply_edge_threshold_floor,
}


def apply_gate_stack(
    *,
    raw_ensemble_prob: float,
    model_prob: float,
    market_mid: float | None,
    market_spread: float | None,
    strike_type: Literal["greater", "less", "between"],
    is_excluded_cell: bool,
    live_calibrator_kind: LiveCalibratorKind,
    strike_abs_z: float | None = None,
    gates: frozenset[GateName],
    config: GateConfig,
) -> tuple[str | None, dict[str, float] | None]:
    """Return `(rejection_reason, rejection_payload)`; `(None, None)` on pass.

    Per-arm gate evaluator (v1.1 A4): gates ONE row's ONE posterior.
    Iterates `_canonical_gate_order`, evaluates only the gates in
    `gates`, first-failure-wins (spec §4.5).
    """
    for gate_name in _canonical_gate_order:
        if gate_name not in gates:
            continue
        predicate = _GATE_PREDICATES[gate_name]
        reason, payload = predicate(
            raw_ensemble_prob=raw_ensemble_prob,
            model_prob=model_prob,
            market_mid=market_mid,
            market_spread=market_spread,
            strike_type=strike_type,
            is_excluded_cell=is_excluded_cell,
            live_calibrator_kind=live_calibrator_kind,
            strike_abs_z=strike_abs_z,
            config=config,
        )
        if reason is not None:
            return reason, payload
    return None, None


def _annotate_gate_verdict(
    row: EvaluationRow,
    *,
    gates: frozenset[GateName],
    gate_config: GateConfig,
    excluded_cell_vars: frozenset[str] | None,
) -> EvaluationRow:
    """Apply the gate stack to an emitted row and return a new row with
    `(rejection_reason, rejection_payload)` populated. Row is yielded even
    when rejected (spec §4.1 "applied but yielded").

    The iterator passes `strike_abs_z=None` (spec M5: workstreams that
    want strike-distance gating call `apply_gate_stack` themselves with
    their own pooled z-score). `live_calibrator_kind` comes from the
    row's emitted tag — for in-iterator gating, the row's tag IS the
    caller's claim because the scaffold gates its own posterior.
    """
    # excluded_cell_vars uses underscore form ("DEN_high"); see
    # _validate_filters. EMOS cells use slash form ("DEN/high").
    cell_underscore = f"{row.city}_{row.var}"
    is_excluded = (
        excluded_cell_vars is not None and cell_underscore in excluded_cell_vars
    )

    market_mid: float | None
    market_spread: float | None
    if row.yes_bid is not None and row.yes_ask is not None:
        market_mid = (row.yes_bid + row.yes_ask) / 2.0
        market_spread = row.yes_ask - row.yes_bid
    else:
        market_mid = None
        market_spread = None

    reason, payload = apply_gate_stack(
        raw_ensemble_prob=row.bracket_raw_prob_production_mirrored,
        model_prob=row.emos_prob_bracket if row.emos_prob_bracket is not None else 0.0,
        market_mid=market_mid,
        market_spread=market_spread,
        strike_type=row.strike_type,
        is_excluded_cell=is_excluded,
        live_calibrator_kind=row.live_calibrator_kind,
        strike_abs_z=None,
        gates=gates,
        config=gate_config,
    )

    if reason is None and payload is None:
        return row

    return replace(row, rejection_reason=reason, rejection_payload=payload)


def _snapshots_dir_signature(snapshots_dir: Path) -> str:
    """sha256 of sorted `(relpath, size_bytes, mtime_ns)` tuples (spec §6.2).

    Metadata-tuple hash — matches the convention `EmosTable.artifact_signature`
    uses for hot-reload. O(metadata), not O(content). Workstreams that want
    content-hash reproducibility compute it themselves.
    """
    entries: list[tuple[str, int, int]] = []
    for path in sorted(snapshots_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rel = path.relative_to(snapshots_dir).as_posix()
            entries.append((rel, stat.st_size, stat.st_mtime_ns))
    serialized = "\n".join(f"{rel}\t{size}\t{mtime}" for rel, size, mtime in entries)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_envelope(
    snapshots_dir: Path,
    emos_table: EmosTable,
    gate_config: GateConfig | None = None,
    *,
    lead_source: Literal[
        "utc_today", "snapshot_day_utc", "snapshot_day_local"
    ] = "snapshot_day_utc",
) -> ScaffoldEnvelope:
    """Pure function. Build a `ScaffoldEnvelope` for reproducibility metadata.

    Spec §6.2 / v1.1 A5: three SHAs (scaffold module, EMOS artifact, snapshots
    dir signature) plus the gate-config repr and `lead_source` together pin
    the result of an `iter_evaluation_rows` invocation.

    v1.2 B6: when `gate_config is None`, `gate_config_repr` is `None`
    (serializes as JSON `null`) — signalling that gates were not applied.
    When non-None, the dataclass `__repr__` captures the constants.
    """
    scaffold_module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshots_dir_sha = _snapshots_dir_signature(snapshots_dir)
    gate_config_repr = None if gate_config is None else str(gate_config)
    return ScaffoldEnvelope(
        scaffold_version="1.3",
        snapshots_dir_sha=snapshots_dir_sha,
        emos_artifact_hash=emos_table.artifact_hash or "",
        emos_fit_date=emos_table.fit_date or "",
        scaffold_module_sha=scaffold_module_sha,
        lead_source=lead_source,
        gate_config_repr=gate_config_repr,
    )
