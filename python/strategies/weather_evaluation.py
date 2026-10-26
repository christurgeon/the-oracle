"""
Pure pipeline functions for the weather-edge strategy.

``evaluate_market_pure`` is the canonical per-market evaluation —
shared by the live ``WeatherEdgeStrategy`` and the backtest engine.
All rejection gates, sizing, and the EMOS shadow block live here.
Imported by ``weather_edge.py`` (which re-exports every public symbol
for back-compat with tests). ``python/analytics/backtest.py`` imports
directly from this module.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from datetime import UTC, date, datetime
from typing import Any

from python.analytics.audit_log import (
    AuditSinks,
    RejectionReason,
    ScanFunnel,
)
from python.analytics.boundary_distance import boundary_distance
from python.analytics.emos_fit import DEFAULT_BRACKET_WIDTH, sigma_emos_at
from python.analytics.fees import fee_per_contract as kalshi_fee
from python.analytics.sizing import SizingResult, dollars_to_contracts, fractional_kelly
from python.common import NWS_ROUND_HALF_F, DailyVariable, Direction, Side, et_today
from python.data.bias_correction import BiasTable, apply_bias_correction
from python.data.calibration import Calibrator
from python.data.emos import EmosTable, emos_probability
from python.data.weather import (
    EnsembleForecast,
    WeatherClient,
    blended_ensemble_median_std,
    blended_forecast_mean,
    compute_blended_probability,
    get_spread_inflation_factor,
)
from python.strategies.base import TradeSignal
from python.strategies.weather_audit import (
    _eval_audit_row,
    _log_emos_failure_once,
)
from python.strategies.weather_parsing import (
    _DIRECTION_TO_STRIKE_LABEL,
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    extract_market_price_info,
)

logger = logging.getLogger(__name__)


# NWS-rounded settle window [floor - 0.5, cap + 0.5), the width the σ-veto in
# emos_fit.bootstrap_cell evaluates against. Re-exported from
# emos_fit.DEFAULT_BRACKET_WIDTH under a series-agnostic name (KXHIGH and KXLOWT
# share the geometry) that the calibrator harness asserts on at import.
DAILY_WEATHER_BRACKET_WIDTH_DEG_F: float = DEFAULT_BRACKET_WIDTH

# Threshold-tail gate: skip threshold markets whose raw ensemble probability
# lands in the extreme tails [_THRESHOLD_TAIL_LO, _THRESHOLD_TAIL_HI].
_THRESHOLD_TAIL_LO: float = 0.05
_THRESHOLD_TAIL_HI: float = 0.95

# YES-side $0.15 floor: reject YES-side threshold trades priced below this midpoint.
_YES_SIDE_MIN_MIDPOINT: float = 0.15

# YES-side $0.10 floor for BETWEEN markets. The settled record for between-YES
# trades priced below $0.10 was 33 trades, 6.1% win rate, -58.7% ROI: the model
# systematically claims ~20% on 1°F brackets the market prices at ~1.5¢ and
# loses ~94% of the time. These all clear min_calibrated_prob (model_prob
# 0.14-0.26), so the floor block does not catch them. Lower than the threshold
# floor because the cheap-longshot bleed is unambiguous only below 10¢.
_BETWEEN_YES_SIDE_MIN_MIDPOINT: float = 0.10

# Bias-correction epsilon (°F): smaller corrections count as none.
_BIAS_CORRECTION_EPSILON_F: float = 0.01


@dataclass(frozen=True)
class _EvalConfig:
    """Loop-invariant config values for market evaluation."""

    max_spread: float
    spread_skill_sensitivity: float
    edge_threshold: float
    max_edge: float
    kelly_fraction: float
    max_pct: float
    max_take_pct: float
    depth_check_enabled: bool
    min_raw_prob_for_between: float = 0.02
    min_distance_from_strike_std: float = 0.0
    max_fill_slippage: float | None = None
    # Skip trades whose raw ensemble prob lands in this range. ``None`` disables.
    block_raw_prob_range: tuple[float, float] | None = None
    # Reject near-term (lead_days <= 2) markets where |model_prob - market_mid|
    # exceeds this threshold — catches stale GFS forecasts that disagree
    # wildly with the market on short-horizon trades. ``None`` = off.
    max_model_market_gap_near_term: float | None = None
    # Reject candidate signals when calibrated probability is below this floor.
    # ``None`` disables. Mode-aware: only set in live/paper modes via
    # generate_signals; shadow mode passes ``None`` so the bin remains observable.
    min_calibrated_prob: float | None = None
    # Reject EMOS-live candidates whose predictive σ_emos at the live ens_std
    # exceeds this multiple of the 1°F KXHIGH/KXLOWT bracket width. Mirrors
    # the fit-time ``emos_fit.DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD`` constant.
    # Set to a very large value to disable. Only consulted on rows where
    # ``live_calibrator_kind == "emos_raw"``.
    emos_runtime_sigma_threshold: float = 2.0
    # When False (shipped default), the gate computes ``nbm_veto_fired`` on
    # every between-strike candidate for the audit stream WITHOUT rejecting
    # the trade. When True the gate also rejects. ``nbm_fail_open`` controls
    # whether an NBM fetch failure (a missing cache entry) skips the gate
    # (True, default) or rejects with ``NBM_FETCH_FAILED`` (False). Both
    # flags are no-ops when the caller does not provide ``nbm_cache``.
    nbm_bracket_position_veto_enabled: bool = False
    nbm_fail_open: bool = True
    # Maps ``"<CITY>/<high|low>"`` → °F threshold (Path 7a). A candidate in a
    # listed cell is dropped when its predicted forecast-mean-to-nearest-
    # NWS-rounding-boundary distance is below the threshold. Empty (default)
    # = filter disabled for all cells.
    boundary_filter_cells: dict[str, float] = field(default_factory=dict)


def _parse_plateau_range(value: Any) -> tuple[float, float] | None:
    """Coerce ``block_raw_prob_range`` config input to a ``(lo, hi)`` tuple
    or ``None`` (disabled). Accepts a 2-element list/tuple; rejects ``null``,
    empty, and out-of-order bounds."""
    if value is None or value == [] or value == ():
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"block_raw_prob_range must be a 2-element list or null, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"block_raw_prob_range bounds must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]"
        )
    return (lo, hi)


def _nws_shifted_emos_bounds(
    strike_type: str,
    floor: float | None,
    cap: float | None,
) -> tuple[float | None, float | None]:
    """Shift (floor, cap) to the NWS integer-rounding settle window for emos_probability."""
    if strike_type == "between":
        if floor is None or cap is None:
            raise ValueError("between strike requires both floor and cap")
        return floor - NWS_ROUND_HALF_F, cap + NWS_ROUND_HALF_F
    if strike_type == "greater":
        if floor is None:
            raise ValueError("greater strike requires floor")
        return floor + NWS_ROUND_HALF_F, None
    if strike_type == "less":
        if cap is None:
            raise ValueError("less strike requires cap")
        return None, cap - NWS_ROUND_HALF_F
    raise ValueError(f"unknown strike_type: {strike_type!r}")


def _parse_min_calibrated_prob(value: Any) -> float | None:
    """Coerce ``min_calibrated_prob`` config input to a float in [0, 1] or None."""
    if value is None:
        return None
    floor = float(value)
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"min_calibrated_prob must satisfy 0 <= floor <= 1, got {floor}")
    return floor


@dataclass
class _MarketCandidate:
    """Intermediate result from evaluating a single market (before depth check)."""

    parsed: ParsedMarket
    market: dict[str, Any]  # raw Kalshi API dict (for metadata extraction)
    forecast: EnsembleForecast  # primary forecast (first available model)
    model_prob: float
    correction_f: float
    price_info: MarketPriceInfo
    raw_edge: float
    ensemble_confidence: float
    effective_edge: float
    sizing: SizingResult
    contracts: int
    exec_price: float = float("nan")  # executable YES-space price (ask for YES, bid for NO)
    per_model_probs: dict[str, float] | None = None  # per-model bracket probs
    ensemble_std_pooled: float | None = None  # pooled std across all models
    spread_inflation_factor: float = 1.0
    raw_ensemble_prob: float | None = None  # pre-calibration blended probability
    # GFS-only probability with the bias correction backed out — useful for
    # auditing whether the correction is improving or hurting calibration.
    # ``None`` when no bias correction was applied (avoids being confused
    # with the calibrated blended probability) or when GFS is absent.
    gfs_uncorrected_prob: float | None = None
    # |strike - pooled_median| / pooled_std — threshold markets only
    strike_abs_z: float | None = None
    # Pooled median across all models, snapshot pre-spread-inflation. Surfaces
    # to TradeSignal metadata so consumers can read it directly instead of
    # inverting ``ref_strike - strike_abs_z * ensemble_std_pooled``, which is
    # sign-ambiguous and undefined for BETWEEN markets.
    ensemble_median_pooled: float | None = None
    # Parallel-calibrator output, logged but never applied to sizing / edge.
    # ``None`` when no ``shadow_calibrator`` is configured.
    shadow_prob: float | None = None
    # EMOS shadow fields, populated only when emos_table is loaded and the
    # market is in-horizon. ``None`` means not computed for this row — never
    # confused with "computed and got 0".
    emos_prob_raw: float | None = None
    emos_prob_corrected: float | None = None
    emos_mu_raw: float | None = None
    emos_sigma_raw: float | None = None
    emos_mu_corr: float | None = None
    emos_sigma_corr: float | None = None
    emos_cdf_at_floor_raw: float | None = None
    emos_cdf_at_cap_raw: float | None = None
    emos_cdf_at_floor_corr: float | None = None
    emos_cdf_at_cap_corr: float | None = None
    emos_warnings: list[str] = field(default_factory=list)
    emos_lead: int | None = None
    emos_disabled_reason: str | None = None
    emos_coeffs_artifact_hash: str | None = None
    emos_coeffs_loaded_at: str | None = None
    isotonic_artifact_hash: str | None = None
    gfs_bias_correction_f: float | None = None
    # Per-cell EMOS dispatch (Stage A). One of:
    # "isotonic", "emos_raw", "isotonic_fallback", "identity".
    live_calibrator_kind: str = "isotonic"
    # NBM-bracket-position veto (shadow-only). ``None`` = gate not evaluated
    # (non-between markets, no NBM cache entry, or fetch failure under
    # fail-open). ``False`` = NBM agreed with the model's bracket position.
    # ``True`` = NBM disagreed — under
    # ``cfg.nbm_bracket_position_veto_enabled = True`` the candidate would
    # have been rejected with ``NBM_BRACKET_POSITION_VETO``; under the
    # shipped ``False`` the row reaches this candidate so the PASSED audit
    # row records the would-have-fired outcome.
    nbm_temperature: float | None = None
    nbm_veto_fired: bool | None = None


def _select_one_per_event(
    signals: list[TradeSignal],
    *,
    enabled: bool,
) -> tuple[list[TradeSignal], int]:
    """Cap weather signals to one per (city, var, forecast_date) event.

    Groups by (metadata["city"], metadata["market_type"], metadata["forecast_date"])
    and keeps the highest-effective_edge candidate in each group. Same-event
    losers are dropped; the survivor is returned with three populated metadata
    fields:

        event_candidates_count: int          -- size of the original group
        event_dropped_edges:    list[float]  -- post-boost effective_edges of
                                                losers, ordered descending by edge
        event_dropped_tickers:  list[str]    -- losers' tickers, same order as
                                                event_dropped_edges

    Tie-break (deterministic so replays reproduce):
        primary:    effective_edge   (descending)
        tiebreaker: confidence       (descending)
        tiebreaker: market_ticker    (ascending lexicographic)

    Note on bracket-boost: ``TradeSignal.effective_edge`` is *post-boost*
    (the boost is applied inside ``WeatherEdgeStrategy.generate_signals``
    before ``_build_signal`` constructs the TradeSignal).
    Two same-event candidates where one is bracket-boosted and the other
    is not will see the boosted candidate win even if its raw (pre-boost)
    edge was lower. That is intentional — the boost reflects higher
    structural signal quality from bracket-bid asymmetry.

    When ``enabled=False``, returns ``(signals, 0)`` unchanged. The metadata
    fields populated at construction time (count=1, [], []) remain in place.

    Returns ``(kept_signals, dropped_count)``. The returned list preserves
    the ordering of first-occurrence per group in the input list (so the
    caller's downstream logging stays stable).
    """
    if not enabled:
        return signals, 0

    groups: dict[tuple[str, str, str], list[TradeSignal]] = {}
    for sig in signals:
        key = (
            str(sig.metadata.get("city", "")),
            str(sig.metadata.get("market_type", "")),
            str(sig.metadata.get("forecast_date", "")),
        )
        groups.setdefault(key, []).append(sig)

    kept: list[TradeSignal] = []
    dropped_count = 0
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        ranked = sorted(
            group,
            key=lambda s: (-s.effective_edge, -s.confidence, s.market_ticker),
        )
        winner = ranked[0]
        losers = ranked[1:]
        dropped_count += len(losers)
        new_meta = dict(winner.metadata)
        new_meta["event_candidates_count"] = len(group)
        new_meta["event_dropped_edges"] = [s.effective_edge for s in losers]
        new_meta["event_dropped_tickers"] = [s.market_ticker for s in losers]
        kept.append(_replace(winner, metadata=new_meta))

    return kept, dropped_count


def _extract(sink_entry: dict[str, EnsembleForecast], model: str) -> list[float] | None:
    """Extract a model's member list from a nested sink entry, applying the
    "no intermediate empty-list" invariant from spec §5.3.

    Returns ``None`` when the model is absent, the temperatures list is
    empty, OR every member is non-finite. Returns a non-empty list of
    finite floats otherwise. This eliminates the empty-list state from
    the downstream row-build flow.
    """
    fc = sink_entry.get(model)
    if fc is None:
        return None
    finite = [m for m in fc.temperatures if math.isfinite(m)]
    return finite if finite else None


def _stats(members: list[float] | None) -> tuple[float | None, float | None]:
    """Compute per-model ddof=1 mean/std from a member list.

    ``None`` in, ``(None, None)`` out — used for absent-model rows.
    Single-member input returns ``(member, 0.0)`` to distinguish
    "degenerate-but-present" from "absent" in the row schema.

    One-pass arithmetic — do NOT swap for ``statistics.stdev`` (which is
    Fraction-based and ~35x slower) or ``EnsembleForecast.std_dev`` (which
    is ddof=0; the corpus convention is ddof=1, see ``build_emos_corpus.py``).
    """
    if members is None:
        return None, None
    n = len(members)
    if n < 2:
        return members[0], 0.0
    mean = sum(members) / n
    variance = sum((x - mean) ** 2 for x in members) / (n - 1)
    return mean, variance**0.5


def build_forecasts_raw_rows(
    *,
    scan_id: str,
    raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
    bias_corrections: dict[tuple[str, date, DailyVariable], float],
    utc_today: date,
) -> list[dict[str, Any]]:
    """Build forecasts_raw.jsonl rows from a captured raw-member snapshot.

    Pure (no I/O). Emits one row per fkey where at least one model has
    members. Each model contributes three fields (``{model}_members_raw``,
    ``{model}_ens_mean_raw_ddof1``, ``{model}_ens_std_raw_ddof1``); missing
    models surface as JSON nulls on those three. Rows with no member data
    for any model are dropped.

    Var serialization MUST match the corpus + eval_audit conventions:
      - corpus: scripts/build_emos_corpus.py:47-50 (TEMP_MAX -> "high",
        TEMP_MIN -> "low")
      - eval_audit: weather_audit._eval_audit_row ("var": parsed.temp_label)

    Lead uses ``utc_today`` (datetime.now(UTC).date() at the call site)
    NOT ``et_today()`` -- the corpus uses UTC ``snap_d``; ET-based leads
    drift one day on scans firing 19:00-23:59 ET.

    """
    rows: list[dict[str, Any]] = []
    for fkey, sink_entry in raw_member_sink.items():
        gfs_members = _extract(sink_entry, "gfs_seamless")
        ecmwf_members = _extract(sink_entry, "ecmwf_ifs025")
        if gfs_members is None and ecmwf_members is None:
            continue  # Case C: no model produced members → drop row
        gfs_mean, gfs_std = _stats(gfs_members)
        ecmwf_mean, ecmwf_std = _stats(ecmwf_members)
        city, fc_date, daily_var = fkey
        var_short = "high" if daily_var == DailyVariable.TEMP_MAX else "low"
        rows.append(
            {
                "scan_id": scan_id,
                "scan_ts": datetime.now(UTC).isoformat(),
                "city": city,
                "var": var_short,
                "forecast_date": fc_date.isoformat(),
                "lead": (fc_date - utc_today).days,
                "gfs_members_raw": gfs_members,
                "gfs_ens_mean_raw_ddof1": gfs_mean,
                "gfs_ens_std_raw_ddof1": gfs_std,
                "ecmwf_members_raw": ecmwf_members,
                "ecmwf_ens_mean_raw_ddof1": ecmwf_mean,
                "ecmwf_ens_std_raw_ddof1": ecmwf_std,
                "gfs_bias_correction_f": bias_corrections.get(fkey, 0.0),
            }
        )
    return rows


def postprocess_forecasts(
    forecast_cache: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
    bias_table: BiasTable | None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
    *,
    raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]
    | None = None,
) -> tuple[
    dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
    dict[tuple[str, date, DailyVariable], float],
    dict[tuple[str, date, DailyVariable], float],
    dict[tuple[str, date, DailyVariable], float],
]:
    """Apply bias correction + spread inflation + compute pooled median/std.

    Pure (no I/O). Mutates ``forecast_cache`` model dicts in place to keep
    memory footprint minimal, same as the original live implementation.

    ``raw_member_sink`` is an optional out-param dict. Pass ``None`` (or
    omit) to opt out — fully backward-compatible with all backtest paths.
    Pass an empty dict (``{}``) to opt in: the function deepcopies every
    available model in ``forecast_cache[fkey]`` at the very top, BEFORE
    bias correction runs, populating the dict as a two-level mapping
    ``{fkey: {model_name: EnsembleForecast}}`` mirroring the input cache
    shape (spec §5.2).

    The nested shape lets the EMOS shadow path keep reading GFS-only
    while ``build_forecasts_raw_rows`` reads both models for the
    forecasts_raw.jsonl side-stream. The per-model deepcopy preserves
    the immutability invariant pinned in scope-A4.

    Returns ``(forecast_cache, bias_corrections, raw_ensemble_stds,
    raw_ensemble_medians)`` — keyed by ``(city, forecast_date, daily_variable)``.
    Medians and stds are captured *before* spread inflation so
    ``spread_skill_sensitivity`` does not double-count the underdispersion
    correction and the strike-proximity gate scores against the same
    distribution the blended probability is computed from.
    """
    # Pre-bias-correction snapshot, taken first. Shape mirrors forecast_cache
    # (spec §5.2); the per-model deepcopy preserves the scope-A4 immutability invariant.
    if raw_member_sink is not None:
        for fkey, model_dict in forecast_cache.items():
            raw_member_sink[fkey] = {
                model: copy.deepcopy(forecast) for model, forecast in model_dict.items()
            }

    bias_corrections: dict[tuple[str, date, DailyVariable], float] = {}
    if bias_table is not None and bias_table.loaded:
        for fkey in list(forecast_cache.keys()):
            _city, _dt, _var = fkey
            model_dict = forecast_cache[fkey]
            if "gfs_seamless" in model_dict:
                corrected, correction_f = apply_bias_correction(
                    model_dict["gfs_seamless"],
                    bias_table,
                    _var,
                    shrinkage=bias_shrinkage,
                    max_correction_f=bias_max_correction_f,
                )
                model_dict["gfs_seamless"] = corrected
                bias_corrections[fkey] = correction_f
                if abs(correction_f) > _BIAS_CORRECTION_EPSILON_F:
                    logger.info(
                        "Bias correction for %s/%s/month=%d: %.2f°F",
                        _city,
                        _var,
                        _dt.month,
                        correction_f,
                    )

    raw_ensemble_stds: dict[tuple[str, date, DailyVariable], float] = {}
    raw_ensemble_medians: dict[tuple[str, date, DailyVariable], float] = {}
    for fkey in forecast_cache:
        median, std = blended_ensemble_median_std(forecast_cache[fkey])
        raw_ensemble_medians[fkey] = median
        raw_ensemble_stds[fkey] = std

    # GFS-derived factors applied to all models including ECMWF — intentional mismatch
    # pending EMOS, which absorbs spread calibration by construction.
    for fkey in forecast_cache:
        _city, _dt, _var = fkey
        factor = get_spread_inflation_factor(_city, _var, spread_inflation_overrides)
        if factor != 1.0:
            model_dict = forecast_cache[fkey]
            for model_name in list(model_dict.keys()):
                model_dict[model_name] = model_dict[model_name].inflated(factor)
            logger.debug(
                "Spread inflation for %s/%s: %.1fx",
                _city,
                _var,
                factor,
            )

    return forecast_cache, bias_corrections, raw_ensemble_stds, raw_ensemble_medians


@dataclass(frozen=True)
class _EmosShadow:
    """EMOS shadow-block output carried on every audit row and candidate.

    ``None`` means "not computed for this row" — never confused with a computed 0.
    """

    prob_raw: float | None = None
    prob_corrected: float | None = None
    mu_raw: float | None = None
    sigma_raw: float | None = None
    mu_corr: float | None = None
    sigma_corr: float | None = None
    cdf_at_floor_raw: float | None = None
    cdf_at_cap_raw: float | None = None
    cdf_at_floor_corr: float | None = None
    cdf_at_cap_corr: float | None = None
    warnings: tuple[str, ...] = ()
    lead: int | None = None
    disabled_reason: str | None = None
    coeffs_artifact_hash: str = "identity-fallback"
    coeffs_loaded_at: str | None = None

    @classmethod
    def initial(cls, emos_table: EmosTable | None) -> _EmosShadow:
        """State before (or without) the EMOS block running.

        ``disabled_reason``, the artifact hash and ``loaded_at`` are pinned
        per-evaluation here so the (hash, loaded_at) pair always describes a
        single coefficients snapshot. ``_build_signal`` MUST read them from the
        candidate, NOT from ``self.emos_table`` — a mid-scan hot reload would
        otherwise pair the pre-swap hash with the post-swap ``loaded_at``
        (R2-H9 invariant violation for signal-level metadata).
        """
        return cls(
            disabled_reason=(
                emos_table.fallback_reason if emos_table is not None else "table-not-loaded"
            ),
            coeffs_artifact_hash=(
                emos_table.artifact_hash
                if (emos_table is not None and emos_table.is_loaded and emos_table.artifact_hash)
                else "identity-fallback"
            ),
            coeffs_loaded_at=(
                emos_table.artifact_loaded_at
                if (emos_table is not None and emos_table.is_loaded)
                else None
            ),
        )

    def audit_fields(self) -> dict[str, Any]:
        """The ``emos_*`` kwargs shared by ``_eval_audit_row`` and ``_MarketCandidate``."""
        return {
            "emos_prob_raw": self.prob_raw,
            "emos_prob_corrected": self.prob_corrected,
            "emos_mu_raw": self.mu_raw,
            "emos_sigma_raw": self.sigma_raw,
            "emos_mu_corr": self.mu_corr,
            "emos_sigma_corr": self.sigma_corr,
            "emos_cdf_at_floor_raw": self.cdf_at_floor_raw,
            "emos_cdf_at_cap_raw": self.cdf_at_cap_raw,
            "emos_cdf_at_floor_corr": self.cdf_at_floor_corr,
            "emos_cdf_at_cap_corr": self.cdf_at_cap_corr,
            "emos_warnings": list(self.warnings),
            "emos_lead": self.lead,
            "emos_disabled_reason": self.disabled_reason,
            "emos_coeffs_artifact_hash": self.coeffs_artifact_hash,
            "emos_coeffs_loaded_at": self.coeffs_loaded_at,
        }


@dataclass(frozen=True)
class _NbmVeto:
    """Outcome of the NBM-bracket-position gate for one candidate.

    ``evaluated`` and ``fetch_failed`` are both ``False`` when the gate does
    not apply (non-between market, no cache, or no cache entry).
    """

    evaluated: bool = False
    fetch_failed: bool = False
    temperature: float | None = None
    veto_fired: bool | None = None
    inside: bool | None = None


def _compute_emos_shadow(
    initial: _EmosShadow,
    emos_table: EmosTable | None,
    parsed: ParsedMarket,
    strike_label: str,
    fkey: tuple[str, date, DailyVariable],
    raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] | None,
    bias_corrections: dict[tuple[str, date, DailyVariable], float],
    emos_log_dedup: dict[Any, Any] | None,
    scan_id: str | None,
) -> _EmosShadow:
    """EMOS shadow block — runs BEFORE every rejection gate.

    The analyzer sees plateau-blocked, min_raw_between-blocked,
    strike-proximity-blocked, threshold-tail-clipped, calibrator-disagreement
    and MODEL_MARKET_GAP_INTERIM rows with EMOS predictions populated. See spec
    Risk #1 (scope-C plateau-bound re-validation). Never raises on a model
    failure; each variant logs once and leaves its fields ``None``.
    """
    if emos_table is None or not emos_table.is_loaded:
        return initial

    emos_prob_raw: float | None = None
    emos_prob_corrected: float | None = None
    emos_mu_raw: float | None = None
    emos_sigma_raw: float | None = None
    emos_mu_corr: float | None = None
    emos_sigma_corr: float | None = None
    emos_cdf_at_floor_raw: float | None = None
    emos_cdf_at_cap_raw: float | None = None
    emos_cdf_at_floor_corr: float | None = None
    emos_cdf_at_cap_corr: float | None = None
    emos_warnings: list[str] = []

    # Lead MUST match corpus convention (UTC date, NOT ET — see
    # scripts/build_emos_corpus.py). ET drift would silently
    # bucket runtime lead-0 rows into corpus lead-1 cells.
    utc_today = datetime.now(UTC).date()
    lead = (parsed.forecast_date - utc_today).days
    emos_lead = lead

    if lead < 0:
        emos_warnings.append("lead_negative")
    elif lead > 2:
        emos_warnings.append("lead_out_of_range")
    else:
        try:
            coeffs = emos_table.get(city=parsed.city, var=parsed.temp_label, lead=lead)
        except KeyError:
            coeffs = None
            emos_warnings.append("var_shape_mismatch")

        if coeffs is None:
            emos_warnings.append("cell_skipped_or_missing")
        else:
            # Hoisted above both try blocks so neither variant can NameError on
            # the other's failure, and both see the same NaN/Inf-free member list
            # as build_forecasts_raw_rows (spec §9 R6).
            sink_entry = raw_member_sink.get(fkey, {}) if raw_member_sink is not None else {}
            gfs_members_finite = _extract(sink_entry, "gfs_seamless")
            shift = bias_corrections.get(fkey, 0.0)

            # Variant 1: raw GFS (corpus-conventional inputs).
            try:
                if gfs_members_finite is not None and len(gfs_members_finite) >= 2:
                    ens_mean_raw, ens_std_raw = _stats(gfs_members_finite)
                    # _stats returns (float | None, float | None); inside
                    # the len-guard the contract is float — narrow for mypy.
                    assert ens_mean_raw is not None and ens_std_raw is not None
                    p99 = emos_table.training_ens_std_p99(parsed.city, parsed.temp_label)
                    if p99 is not None and ens_std_raw > p99:
                        emos_warnings.append("ens_std_extrapolation")
                    emos_floor_raw, emos_cap_raw = _nws_shifted_emos_bounds(
                        strike_label, parsed.floor_strike, parsed.cap_strike
                    )
                    (
                        emos_prob_raw,
                        emos_cdf_at_floor_raw,
                        emos_cdf_at_cap_raw,
                    ) = emos_probability(
                        coeffs,
                        ens_mean_raw,
                        ens_std_raw,
                        strike_type=strike_label,
                        floor=emos_floor_raw,
                        cap=emos_cap_raw,
                    )
                    emos_mu_raw = coeffs.a + coeffs.b * ens_mean_raw
                    emos_sigma_raw = math.sqrt(max(coeffs.c + coeffs.d * ens_std_raw**2, 1e-12))
            except Exception as exc:
                _log_emos_failure_once(
                    emos_log_dedup,
                    scan_id,
                    parsed.city,
                    parsed.temp_label,
                    "emos_raw_failed",
                    exc,
                )

            # Variant 2: bias-corrected GFS, reconstructed from the
            # already-filtered list + shift (relies on the immutability
            # invariant pinned in scope-A4). No second _extract.
            try:
                if gfs_members_finite is not None and len(gfs_members_finite) >= 2:
                    corrected_members = [m + shift for m in gfs_members_finite]
                    ens_mean_corr, ens_std_corr = _stats(corrected_members)
                    assert ens_mean_corr is not None and ens_std_corr is not None
                    emos_floor_corr, emos_cap_corr = _nws_shifted_emos_bounds(
                        strike_label, parsed.floor_strike, parsed.cap_strike
                    )
                    (
                        emos_prob_corrected,
                        emos_cdf_at_floor_corr,
                        emos_cdf_at_cap_corr,
                    ) = emos_probability(
                        coeffs,
                        ens_mean_corr,
                        ens_std_corr,
                        strike_type=strike_label,
                        floor=emos_floor_corr,
                        cap=emos_cap_corr,
                    )
                    emos_mu_corr = coeffs.a + coeffs.b * ens_mean_corr
                    emos_sigma_corr = math.sqrt(max(coeffs.c + coeffs.d * ens_std_corr**2, 1e-12))
            except Exception as exc:
                _log_emos_failure_once(
                    emos_log_dedup,
                    scan_id,
                    parsed.city,
                    parsed.temp_label,
                    "emos_corr_failed",
                    exc,
                )

            emos_warnings.extend(coeffs.ci_warnings)
            if coeffs.low_n:
                emos_warnings.append("low_n")

    return _replace(
        initial,
        prob_raw=emos_prob_raw,
        prob_corrected=emos_prob_corrected,
        mu_raw=emos_mu_raw,
        sigma_raw=emos_sigma_raw,
        mu_corr=emos_mu_corr,
        sigma_corr=emos_sigma_corr,
        cdf_at_floor_raw=emos_cdf_at_floor_raw,
        cdf_at_cap_raw=emos_cdf_at_cap_raw,
        cdf_at_floor_corr=emos_cdf_at_floor_corr,
        cdf_at_cap_corr=emos_cdf_at_cap_corr,
        warnings=tuple(emos_warnings),
        lead=emos_lead,
    )


def _dispatch_live_calibrator(
    *,
    model_prob: float,
    emos_prob_raw: float | None,
    cell_label: str,
    emos_enabled_cells: frozenset[str] | set[str] | None,
    current_kind: str,
) -> tuple[float, str]:
    """Per-cell live EMOS dispatch (Stage A).

    When the cell is in ``emos_enabled_cells`` AND ``emos_prob_raw`` computed
    successfully, swap ``model_prob`` for ``emos_prob_raw``. Otherwise the
    isotonic-calibrated value (or identity passthrough) stands. A listed cell
    where EMOS did not compute (table not loaded, lead out of range,
    coefficients missing) is recorded as ``isotonic_fallback`` so monthly
    re-eval can spot it.
    """
    if emos_enabled_cells and cell_label in emos_enabled_cells:
        if emos_prob_raw is not None:
            return emos_prob_raw, "emos_raw"
        return model_prob, "isotonic_fallback"
    return model_prob, current_kind


def _emos_runtime_sigma_reject_extra(
    emos_table: EmosTable | None,
    parsed: ParsedMarket,
    emos_lead: int | None,
    pooled_ens_std: float,
    threshold: float,
) -> dict[str, Any] | None:
    """Runtime σ_emos / bracket_width gate (Tier 2 A).

    Returns the ``EMOS_SIGMA_INCOMPATIBLE`` reject ``extra`` dict, or ``None``
    when the candidate passes. The caller invokes it only on rows where the
    dispatcher swapped to EMOS (``live_calibrator_kind == "emos_raw"``).

    The fit-time σ-gate in ``emos_fit.bootstrap_cell`` catches new bad fits at
    the next refit; this is the only guard for the LAX/high-style failure mode
    (σ_emos≈5.76°F vs 1°F bracket → live ROI=−39.6%) on already-loaded
    coefficients. Bracket width is the KXHIGH/KXLOWT daily geometry; intraday
    strategies route through a separate module.

    Coefficients are re-fetched to read (c, d). A missing entry on the same
    scan is a logic bug the dispatcher should have prevented; it is surfaced as
    a rejection with ``runtime_ratio=None``.
    """
    sigma_coeffs = None
    if emos_table is not None and emos_table.is_loaded and emos_lead is not None:
        try:
            sigma_coeffs = emos_table.get(city=parsed.city, var=parsed.temp_label, lead=emos_lead)
        except KeyError:
            sigma_coeffs = None
    if sigma_coeffs is None:
        return {"runtime_ratio": None, "threshold": threshold}
    bracket_width = DAILY_WEATHER_BRACKET_WIDTH_DEG_F
    runtime_sigma = sigma_emos_at(c=sigma_coeffs.c, d=sigma_coeffs.d, ens_std=pooled_ens_std)
    runtime_ratio = runtime_sigma / bracket_width
    if runtime_ratio > threshold:
        logger.warning(
            "%s: EMOS_SIGMA_INCOMPATIBLE — σ_emos=%.3f°F / bracket=%.1f°F "
            "= ratio %.3f > threshold %.2f (cell=%s/%s, c=%.3f, d=%.3f, "
            "ens_std=%.3f)",
            parsed.ticker,
            runtime_sigma,
            bracket_width,
            runtime_ratio,
            threshold,
            parsed.city,
            parsed.temp_label,
            sigma_coeffs.c,
            sigma_coeffs.d,
            pooled_ens_std,
        )
        return {"runtime_ratio": round(runtime_ratio, 4), "threshold": threshold}
    return None


def _boundary_filter_distance(
    cfg: _EvalConfig,
    parsed: ParsedMarket,
    cell_label: str,
    strike_label: str,
    forecast_mean_f: float,
) -> tuple[float, float] | None:
    """Boundary-distance suppressive filter inputs (Path 7a).

    Evidence: ``docs/research/2026-05-28-boundary-aware-phase0-backtest.md``. Returns
    ``(d_boundary_f, threshold)`` when the filter evaluated the candidate, else
    ``None``. BETWEEN-only in v1 — the Phase 0 evidence and spec §11 scope the
    filter to between strikes (~97% of trades); greater/less pass through
    unfiltered even when the cell is enabled. A NaN distance (invalid geometry,
    e.g. between with a ``None`` strike) fails open, matching
    ``boundary_distance()``'s NaN contract. ``strike_label`` is the canonical
    "between"/"greater"/"less" string ``boundary_distance()`` expects.
    """
    threshold = cfg.boundary_filter_cells.get(cell_label)
    if threshold is None or parsed.strike_type != Direction.BETWEEN:
        return None
    d_boundary_f = boundary_distance(
        forecast_mean_f=forecast_mean_f,
        floor_strike=parsed.floor_strike,
        cap_strike=parsed.cap_strike,
        strike_type=strike_label,
    )
    if math.isnan(d_boundary_f):
        return None
    return d_boundary_f, threshold


def _strike_distance_gate(
    cfg: _EvalConfig,
    parsed: ParsedMarket,
    fkey: tuple[str, date, DailyVariable],
    raw_ensemble_medians: dict[tuple[str, date, DailyVariable], float],
    raw_ensemble_stds: dict[tuple[str, date, DailyVariable], float],
) -> tuple[float | None, bool]:
    """Strike-proximity gate for threshold markets.

    Returns ``(strike_abs_z, rejected)``. ``strike_abs_z`` is ``None`` when the
    gate does not apply or rejected before computing it (missing strike,
    degenerate ensemble).
    """
    k_strike = cfg.min_distance_from_strike_std
    if not (k_strike > 0.0 and parsed.strike_type in (Direction.ABOVE, Direction.BELOW)):
        return None, False
    pooled_median = raw_ensemble_medians.get(fkey, 0.0)
    pooled_std = raw_ensemble_stds.get(fkey, 0.0)
    ref_strike = parsed.floor_strike if parsed.strike_type == Direction.ABOVE else parsed.cap_strike
    if ref_strike is None:
        logger.warning(
            "%s: missing strike on threshold market — skipping",
            parsed.ticker,
        )
        return None, True
    if pooled_std <= 0.0:
        logger.warning(
            "%s: degenerate ensemble (std=%.4f) — skipping threshold market",
            parsed.ticker,
            pooled_std,
        )
        return None, True
    strike_abs_z = abs(ref_strike - pooled_median) / pooled_std
    if strike_abs_z < k_strike:
        logger.debug(
            "%s: strike %.1f vs pooled median %.2f — abs_z=%.2f < k=%.2f, skipping",
            parsed.ticker,
            ref_strike,
            pooled_median,
            strike_abs_z,
            k_strike,
        )
        return strike_abs_z, True
    return strike_abs_z, False


def _gfs_uncorrected_prob(
    model_dict: dict[str, EnsembleForecast],
    correction_f: float,
    parsed: ParsedMarket,
    weather_client: WeatherClient,
) -> float | None:
    """GFS-only probability with the bias correction backed out.

    Only meaningful when a non-trivial correction was applied; otherwise it
    would duplicate the already-logged blended probability and create the
    semantic confusion the rename was meant to avoid. ``None`` when no
    correction was applied or GFS is absent.
    """
    if abs(correction_f) > _BIAS_CORRECTION_EPSILON_F and "gfs_seamless" in model_dict:
        gfs_fc = model_dict["gfs_seamless"]
        uncorrected = _replace(
            gfs_fc,
            temperatures=[t - correction_f for t in gfs_fc.temperatures],
        )
        return weather_client.compute_model_probability(
            forecast=uncorrected,
            direction=parsed.strike_type,
            floor_strike=parsed.floor_strike,
            cap_strike=parsed.cap_strike,
        )
    return None


def _select_execution_side(
    model_prob: float, price_info: MarketPriceInfo
) -> tuple[float, float, bool]:
    """Pick the executable side: ``(exec_price, side_price, side_is_yes)``.

    ``exec_price`` is the executable YES-space price (ask for YES, bid for NO);
    ``side_price`` is the price actually paid for the chosen side.
    """
    if model_prob > price_info.midpoint:
        exec_price = price_info.yes_ask if price_info.yes_ask is not None else price_info.midpoint
        return exec_price, exec_price, True
    exec_price = price_info.yes_bid if price_info.yes_bid is not None else price_info.midpoint
    return exec_price, 1.0 - exec_price, False


def _evaluate_nbm_veto(
    parsed: ParsedMarket,
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] | None,
    side_is_yes: bool,
) -> _NbmVeto:
    """NBM-bracket-position veto (shadow-only).

    For between-strike candidates, compare the deterministic NBM forecast
    against the (floor, cap) bracket and the model's directional bet (YES =
    inside, NO = outside). The veto ``fires`` when NBM lands on the opposite
    side: YES + NBM outside, or NO + NBM inside.

    - ``cfg.nbm_bracket_position_veto_enabled = True``: the caller rejects with
      ``NBM_BRACKET_POSITION_VETO`` and the audit row records
      ``(nbm_temperature, nbm_veto_fired=True)``.
    - ``False`` (shipped): no rejection; the audit row still records the
      outcome so the shadow corpus accumulates without blocking trades.

    NBM fetch failures are surfaced by the caller as ``cache[key] = None``
    (``fetch_failed``); ``cfg.nbm_fail_open=True`` (shipped default) skips the
    gate cleanly, ``False`` rejects with ``NBM_FETCH_FAILED``. Absence of the
    key (or ``nbm_cache is None``) means the caller did not attempt a fetch —
    the gate is a no-op.

    Backtest evidence (2026-05-20 research §5, n=32): the disagree cohort had
    46.9% win-rate vs 68.4% for the rest; dropping it turned joined-window P&L
    from a loss to a gain. Audit-only until ≥80 shadow rows accumulate.
    """
    if (
        parsed.strike_type != Direction.BETWEEN
        or nbm_cache is None
        or parsed.floor_strike is None
        or parsed.cap_strike is None
    ):
        return _NbmVeto()
    nbm_key = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    if nbm_key not in nbm_cache:
        return _NbmVeto()
    cached_nbm = nbm_cache[nbm_key]
    if cached_nbm is None:
        return _NbmVeto(fetch_failed=True)
    temperature = float(cached_nbm)
    nbm_inside = parsed.floor_strike <= temperature <= parsed.cap_strike
    return _NbmVeto(
        evaluated=True,
        temperature=temperature,
        veto_fired=(not nbm_inside) if side_is_yes else nbm_inside,
        inside=nbm_inside,
    )


def _calibrator_disagreement_blocks(
    parsed: ParsedMarket,
    live_calibrator_kind: str,
    raw_ensemble_prob: float,
    model_prob: float,
    mid: float,
) -> bool:
    """CALIBRATOR_DISAGREEMENT gate: raw and calibrated on opposite sides of mid.

    Ties (raw == mid or cal == mid) are non-flips — the product is 0, not < 0 —
    so round-lot books don't false-block. Covers BETWEEN, which the
    strike-distance gate does not.

    The gate catches isotonic *plateau pathology* — the KXLOWTCHI-T53 loss had
    raw=0.82, isotonic-collapsed=0.20, so the bot bet against its own forecast.
    The premise is "the calibrator overcorrected; abort". That premise does NOT
    apply when EMOS is the deliberate live calibrator — EMOS structurally absorbs
    regime bias (e.g., +3.53°F μ-shift on MIA/high) and is *expected* to
    disagree with raw on the very cells we've cut over. Verification shows EMOS
    BSS gains of +58.6% on above-strike and +100% on below-strike for MIA/high —
    precisely the trades this gate would suppress. So the gate is skipped on
    emos_raw rows. The test is on the actual swap that happened
    (``live_calibrator_kind == "emos_raw"``), NOT on cell membership in
    ``emos_enabled_cells`` — isotonic_fallback rows (cell listed but EMOS didn't
    compute) still get gate protection because the isotonic plateau is back in
    play.

    On emos_raw rows the would-have-fired triple is logged at DEBUG so analysts
    can quantify the suppression impact by joining eval_audit on this signature.
    """
    disagrees = (raw_ensemble_prob - mid) * (model_prob - mid) < 0
    if live_calibrator_kind == "emos_raw":
        if disagrees:
            logger.debug(
                "%s: CALIBRATOR_DISAGREEMENT skipped (emos_raw); raw=%.3f vs emos=%.3f vs mid=%.3f",
                parsed.ticker,
                raw_ensemble_prob,
                model_prob,
                mid,
            )
        return False
    if disagrees:
        logger.debug(
            "%s: raw %.3f and calibrated %.3f on opposite sides of mid %.3f — skipping",
            parsed.ticker,
            raw_ensemble_prob,
            model_prob,
            mid,
        )
        return True
    return False


def _model_market_gap_extra(
    cfg: _EvalConfig,
    parsed: ParsedMarket,
    scan_today: date | None,
    model_prob: float,
    mid: float,
) -> dict[str, Any] | None:
    """Interim EMOS-gap safeguard (EMOS scope-B interim measure).

    Open-Meteo GFS ingestion lag (3–6 h) leaves near-term forecasts stale enough
    that the market may have priced in weather observations our ensemble missed.
    Blocks trades where |model_prob - market_mid| exceeds the configured
    threshold on lead_days <= 2 markets. Remove or relax once EMOS scope-B is
    live >= 30d and Open-Meteo lag < 1h. See config:
    ``max_model_market_gap_near_term``. Returns the reject ``extra`` dict, or
    ``None`` when the candidate passes.
    """
    if cfg.max_model_market_gap_near_term is None:
        return None
    _today = scan_today if scan_today is not None else et_today()
    lead_days = (parsed.forecast_date - _today).days
    raw_gap = abs(model_prob - mid)
    if lead_days <= 2 and raw_gap > cfg.max_model_market_gap_near_term:
        logger.warning(
            "%s: MODEL_MARKET_GAP_INTERIM — |model %.3f - mid %.3f| = %.3f > %.3f "
            "on lead_days=%d (near-term EMOS-gap gate)",
            parsed.ticker,
            model_prob,
            mid,
            raw_gap,
            cfg.max_model_market_gap_near_term,
            lead_days,
        )
        return {"lead_days": lead_days, "raw_gap": round(raw_gap, 4)}
    return None


def _ensemble_confidence(cfg: _EvalConfig, live_calibrator_kind: str, ensemble_std: float) -> float:
    """Spread-skill discount ``1 / (1 + k · σ)`` applied to raw edge.

    Tier 2 B: EMOS already widens its predictive uncertainty via
    σ = √(c + d · ens_std²), so the discount would double-count spread and is
    skipped on rows where the dispatcher actually swapped to EMOS
    (``live_calibrator_kind == "emos_raw"``). isotonic_fallback rows still get
    the discount — they're on the isotonic path and isotonic discards spread
    information.
    """
    if live_calibrator_kind == "emos_raw":
        return 1.0
    k = cfg.spread_skill_sensitivity
    return 1.0 / (1.0 + k * ensemble_std) if k > 0.0 else 1.0


def evaluate_market_pure(
    sc: ScanCandidate,
    forecast_cache: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
    bias_corrections: dict[tuple[str, date, DailyVariable], float],
    raw_ensemble_stds: dict[tuple[str, date, DailyVariable], float],
    raw_ensemble_medians: dict[tuple[str, date, DailyVariable], float],
    cfg: _EvalConfig,
    *,
    calibrator: Calibrator | None,
    shadow_calibrator: Calibrator | None = None,
    weights: dict[str, float],
    weather_client: WeatherClient,
    bankroll: float,
    excluded_city_vars: list[str] | None = None,
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
    emos_table: EmosTable | None = None,
    raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]
    | None = None,
    emos_log_dedup: dict[Any, Any] | None = None,
    emos_enabled_cells: frozenset[str] | set[str] | None = None,
    isotonic_artifact_hash: str | None = None,
    scan_id: str | None = None,
    sinks: AuditSinks | None = None,
    funnel: ScanFunnel | None = None,
    scan_today: date | None = None,
    n_existing_same_day_bets: int = 0,
    correlation_rho: float | None = None,
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] | None = None,
    strategy_name: str = "weather_edge",
) -> _MarketCandidate | None:
    """Synchronous, pure market evaluation shared by live strategy and backtest.

    Applies: excluded_city_vars → blended probability → isotonic calibration →
    min_raw_prob_for_between gate → strike-proximity gate → bias-correction
    metadata → spread filter → tail-probability gates ($0.05/$0.95 threshold
    clip, $0.15 YES floor) → ensemble confidence → executable price selection
    with inverted-side guards → fee-adjusted effective edge threshold & max_edge
    → Kelly sizing.

    Returns a ``_MarketCandidate`` (pre-depth-check) on pass, else ``None``.
    Orderbook depth adjustment remains the caller's responsibility (async in
    live; unused in backtest until snapshots carry full depth).
    """
    # Count every market that enters evaluation — including those rejected by
    # EXCLUDED_CITY_VAR — so the invariant holds:
    #   n_eval_enter == sum(all rejection counts) + n_signals_fired
    if funnel is not None:
        funnel.n_eval_enter += 1

    parsed = sc.parsed
    _scan_id = scan_id or ""
    _eval_emit = (
        sinks.eval_log.emit_eval if sinks is not None and sinks.eval_log is not None else None
    )

    # Bound up-front because the ``_reject`` closures read them by reference and
    # may fire before the assignment site (e.g. an early EXCLUDED_CITY_VAR
    # rejection needs ``fkey`` for the ``gfs_bias_correction_f`` lookup).
    shadow_prob: float | None = None  # parallel-calibrator output
    fkey: tuple[str, date, DailyVariable] | None = None
    # EMOS shadow state. Computed BEFORE every rejection gate; ``initial`` keeps
    # the table-derived defaults populated even when the block never runs.
    emos = _EmosShadow.initial(emos_table)
    # Per-cell dispatcher state (Stage A), surfaced on every audit row so
    # monthly per-cell re-evaluation can disambiguate which path the row took.
    live_calibrator_kind: str = "identity" if calibrator is None else "isotonic"
    # Boundary-distance filter audit state (Path 7a, #5 observability): set for
    # cells in cfg.boundary_filter_cells, surfaced on EVERY downstream audit row
    # (not just the block row) so the shadow window can verify the captured
    # forecast mean and d_boundary distribution.
    boundary_audit_d: float | None = None
    boundary_audit_mean: float | None = None
    # NBM-bracket-position veto (shadow-only). ``None`` on rows where the gate
    # was not evaluated (non-between markets, no cache, fail-open fetch failure).
    nbm_temperature: float | None = None
    nbm_veto_fired: bool | None = None
    price_info = sc.price_info
    strike_abs_z: float | None = None

    def _reject(reason: RejectionReason, **kwargs: Any) -> None:
        """Emit an eval_audit row and funnel-increment for a rejection."""
        if funnel is not None:
            funnel.increment(reason)
        if _eval_emit is None:
            return
        # Closure injection — rows that fired before a block ran carry nulls.
        kwargs.setdefault("shadow_prob", shadow_prob)
        for key, value in emos.audit_fields().items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("isotonic_artifact_hash", isotonic_artifact_hash)
        kwargs.setdefault(
            "gfs_bias_correction_f",
            bias_corrections.get(fkey, 0.0) if fkey is not None else None,
        )
        kwargs.setdefault("live_calibrator_kind", live_calibrator_kind)
        kwargs.setdefault("nbm_temperature", nbm_temperature)
        kwargs.setdefault("nbm_veto_fired", nbm_veto_fired)
        # Merge the boundary observability into ``extra`` without clobbering a
        # caller's keys (the BOUNDARY_FILTER_BLOCK row sets them explicitly).
        if boundary_audit_d is not None and boundary_audit_mean is not None:
            _bextra = dict(kwargs.get("extra") or {})
            _bextra.setdefault("d_boundary_f", round(boundary_audit_d, 4))
            _bextra.setdefault("forecast_mean_f", round(boundary_audit_mean, 4))
            kwargs["extra"] = _bextra
        _eval_emit(
            _eval_audit_row(
                scan_id=_scan_id,
                sc=sc,
                reason=reason,
                strategy=strategy_name,
                **kwargs,
            )
        )

    def _reject_scored(reason: RejectionReason, **kwargs: Any) -> None:
        """``_reject`` plus the scoring fields, read late so each row carries the
        current post-dispatch ``model_prob`` and post-NO_PRICE-gate ``price_info``."""
        kwargs.setdefault("price_info", price_info)
        kwargs.setdefault("raw_ens_prob", raw_ensemble_prob)
        kwargs.setdefault("calibrated_prob", model_prob)
        kwargs.setdefault("correction_f", correction_f)
        kwargs.setdefault("ens_std", ens_std_local)
        kwargs.setdefault("strike_abs_z", strike_abs_z)
        _reject(reason, **kwargs)

    if excluded_city_vars:
        temp_label = "high" if parsed.daily_variable == DailyVariable.TEMP_MAX else "low"
        city_var_key = f"{parsed.city}_{temp_label}"
        if city_var_key in excluded_city_vars:
            logger.debug(
                "%s: %s in excluded_city_vars — skipping",
                parsed.ticker,
                city_var_key,
            )
            _reject(RejectionReason.EXCLUDED_CITY_VAR, price_info=sc.price_info)
            return None

    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    model_dict = forecast_cache.get(fkey)
    if not model_dict:
        # A data-ingestion gap, not a rejection: counted in scan_funnel.jsonl but
        # kept out of the eval_audit stream.
        if funnel is not None:
            funnel.n_forecast_gap += 1
        return None

    model_prob, per_model_probs = compute_blended_probability(
        model_dict,
        weights,
        direction=parsed.strike_type,
        floor_strike=parsed.floor_strike,
        cap_strike=parsed.cap_strike,
    )

    raw_ensemble_prob = model_prob

    # Blended ensemble mean (°F) for the boundary-distance filter. Pinned beside
    # raw_ensemble_prob so the offline Phase 0 backtest and live trading
    # dispatch on identical numbers; ``blended_forecast_mean`` shares
    # compute_blended_probability's weight renormalization.
    blended_forecast_mean_f = blended_forecast_mean(model_dict, weights)

    # Canonical strike-type label (``greater`` / ``less`` / ``between``) — the
    # ``Direction`` enum values differ, so translate once for both single-curve
    # and per-type calibrators.
    strike_label = _DIRECTION_TO_STRIKE_LABEL[parsed.strike_type]
    # Shadow calibrator runs on the *pre-live-calibration* raw probability so
    # live and shadow see the same input.
    if shadow_calibrator is not None:
        try:
            shadow_prob = shadow_calibrator.calibrate(raw_ensemble_prob, strike_type=strike_label)
        except Exception:
            # Parallel logging must never crash the live path.
            logger.exception("%s: shadow calibrator raised", parsed.ticker)
            shadow_prob = None
    if calibrator is not None:
        model_prob = calibrator.calibrate(model_prob, strike_type=strike_label)

    ens_std_local = raw_ensemble_stds.get(fkey, 0.0)
    correction_f = bias_corrections.get(fkey, 0.0)
    cell_label = f"{parsed.city}/{parsed.temp_label}"

    emos = _compute_emos_shadow(
        emos,
        emos_table,
        parsed,
        strike_label,
        fkey,
        raw_member_sink,
        bias_corrections,
        emos_log_dedup,
        scan_id,
    )

    model_prob, live_calibrator_kind = _dispatch_live_calibrator(
        model_prob=model_prob,
        emos_prob_raw=emos.prob_raw,
        cell_label=cell_label,
        emos_enabled_cells=emos_enabled_cells,
        current_kind=live_calibrator_kind,
    )

    # Only EMOS-live rows: isotonic-native and isotonic_fallback rows are
    # untouched. The ``ens_std_extrapolation`` warning remains a diagnostic;
    # both may fire on the same row.
    if live_calibrator_kind == "emos_raw":
        sigma_extra = _emos_runtime_sigma_reject_extra(
            emos_table,
            parsed,
            emos.lead,
            ens_std_local,
            cfg.emos_runtime_sigma_threshold,
        )
        if sigma_extra is not None:
            _reject_scored(RejectionReason.EMOS_SIGMA_INCOMPATIBLE, extra=sigma_extra)
            return None

    # Calibrated-probability floor block (mode-aware: only active in live/paper).
    # The isotonic artifact has a flat floor at ~0.088/0.092 that absorbs distinct
    # raw inputs in [0.03, 0.12]; trades there have asymmetric small-win/big-loss
    # economics that the bot's win rate cannot overcome.
    if cfg.min_calibrated_prob is not None and model_prob < cfg.min_calibrated_prob:
        logger.debug(
            "%s: calibrated %.4f < floor %.3f — skipping",
            parsed.ticker,
            model_prob,
            cfg.min_calibrated_prob,
        )
        _reject_scored(RejectionReason.CAL_FLOOR_BLOCK)
        return None

    # Plateau-block: a flat stretch of the isotonic artifact gives every raw
    # probability in it the same calibrated value, so there is no discrimination.
    # The range is auto-derived per artifact (``null`` when none exists).
    plateau_range = cfg.block_raw_prob_range
    if plateau_range is not None and plateau_range[0] <= raw_ensemble_prob <= plateau_range[1]:
        logger.debug(
            "%s: raw %.3f in plateau range [%.2f, %.2f] — skipping",
            parsed.ticker,
            raw_ensemble_prob,
            plateau_range[0],
            plateau_range[1],
        )
        _reject_scored(RejectionReason.PLATEAU_BLOCK)
        return None

    # Boundary-distance filter. COHORT NOTE: fires AFTER min_calibrated_prob and
    # block_raw_prob_range, so low-prob grazers never reach it (spec §5 Phase
    # A.5 step-1 order).
    boundary = _boundary_filter_distance(
        cfg, parsed, cell_label, strike_label, blended_forecast_mean_f
    )
    if boundary is not None:
        d_boundary_f, boundary_threshold = boundary
        boundary_audit_d = d_boundary_f
        boundary_audit_mean = blended_forecast_mean_f
        if d_boundary_f < boundary_threshold:
            logger.debug(
                "%s: d_boundary %.3f°F < threshold %.3f°F for %s — skipping",
                parsed.ticker,
                d_boundary_f,
                boundary_threshold,
                cell_label,
            )
            _reject_scored(
                RejectionReason.BOUNDARY_FILTER_BLOCK,
                extra={
                    "d_boundary_f": round(d_boundary_f, 4),
                    "boundary_filter_threshold_f": round(boundary_threshold, 4),
                    "forecast_mean_f": round(blended_forecast_mean_f, 4),
                    "cell_label": cell_label,
                    # Self-contained replay state so an audit row alone can
                    # reproduce the gate decision.
                    "floor_strike": parsed.floor_strike,
                    "cap_strike": parsed.cap_strike,
                    "strike_type": strike_label,
                },
            )
            return None

    min_raw_between = cfg.min_raw_prob_for_between
    if parsed.strike_type == Direction.BETWEEN and raw_ensemble_prob < min_raw_between:
        logger.debug(
            "%s: raw_ensemble_prob %.4f < min %.3f for BETWEEN — skipping",
            parsed.ticker,
            raw_ensemble_prob,
            min_raw_between,
        )
        _reject_scored(RejectionReason.RAW_PROB_GATE)
        return None

    primary_forecast = next(iter(model_dict.values()))

    strike_abs_z, strike_rejected = _strike_distance_gate(
        cfg, parsed, fkey, raw_ensemble_medians, raw_ensemble_stds
    )
    if strike_rejected:
        _reject_scored(RejectionReason.STRIKE_DISTANCE)
        return None

    gfs_uncorrected_prob = _gfs_uncorrected_prob(model_dict, correction_f, parsed, weather_client)

    if price_info is None:
        price_info = extract_market_price_info(sc.market)
    if price_info is None:
        logger.warning(
            "Cannot determine market price for %s — skipping",
            parsed.ticker,
        )
        _reject(RejectionReason.NO_PRICE, price_info=None)
        return None

    if price_info.spread is not None and price_info.spread > cfg.max_spread:
        logger.debug(
            "%s: spread %.3f > max_spread %.3f — skipping",
            parsed.ticker,
            price_info.spread,
            cfg.max_spread,
        )
        _reject_scored(RejectionReason.SPREAD)
        return None

    is_threshold_market = parsed.strike_type in (Direction.ABOVE, Direction.BELOW)

    if is_threshold_market and (
        raw_ensemble_prob < _THRESHOLD_TAIL_LO or raw_ensemble_prob > _THRESHOLD_TAIL_HI
    ):
        logger.debug(
            "%s: raw_ensemble_prob %.3f outside [0.05, 0.95] on threshold market — skipping",
            parsed.ticker,
            raw_ensemble_prob,
        )
        _reject_scored(RejectionReason.RAW_PROB_GATE)
        return None

    mid = price_info.midpoint
    if _calibrator_disagreement_blocks(
        parsed, live_calibrator_kind, raw_ensemble_prob, model_prob, mid
    ):
        _reject_scored(RejectionReason.CALIBRATOR_DISAGREEMENT)
        return None

    gap_extra = _model_market_gap_extra(cfg, parsed, scan_today, model_prob, mid)
    if gap_extra is not None:
        _reject_scored(RejectionReason.MODEL_MARKET_GAP_INTERIM, extra=gap_extra)
        return None

    # YES-side minimum-midpoint floor. Threshold (greater/less) markets use the
    # $0.15 floor; between markets use a $0.10 floor (the cheap-longshot bleed,
    # see _BETWEEN_YES_SIDE_MIN_MIDPOINT). Both fire only on the YES side
    # (model_prob > midpoint).
    yes_side_floor: float | None = None
    if is_threshold_market:
        yes_side_floor = _YES_SIDE_MIN_MIDPOINT
    elif parsed.strike_type == Direction.BETWEEN:
        yes_side_floor = _BETWEEN_YES_SIDE_MIN_MIDPOINT
    if (
        yes_side_floor is not None
        and model_prob > price_info.midpoint
        and price_info.midpoint < yes_side_floor
    ):
        logger.debug(
            "%s: YES-side trade at midpoint %.3f < $%.2f floor — skipping",
            parsed.ticker,
            price_info.midpoint,
            yes_side_floor,
        )
        _reject_scored(RejectionReason.YES_FLOOR)
        return None

    ensemble_std = ens_std_local
    ensemble_confidence = _ensemble_confidence(cfg, live_calibrator_kind, ensemble_std)

    exec_price, side_price, side_is_yes = _select_execution_side(model_prob, price_info)
    if side_is_yes and model_prob <= exec_price:
        logger.debug(
            "%s: YES side inverted at exec_price (model=%.3f <= ask=%.3f) — skipping",
            parsed.ticker,
            model_prob,
            exec_price,
        )
        _reject_scored(RejectionReason.YES_INVERTED, exec_price=exec_price)
        return None
    if not side_is_yes and model_prob >= exec_price:
        logger.debug(
            "%s: NO side inverted at exec_price (model=%.3f >= bid=%.3f) — skipping",
            parsed.ticker,
            model_prob,
            exec_price,
        )
        _reject_scored(RejectionReason.NO_INVERTED, exec_price=exec_price)
        return None

    # NBM-bracket-position veto — see ``_evaluate_nbm_veto``. A fail-open fetch
    # failure leaves ``nbm_temperature`` / ``nbm_veto_fired`` as ``None``.
    nbm = _evaluate_nbm_veto(parsed, nbm_cache, side_is_yes)
    if nbm.fetch_failed:
        if not cfg.nbm_fail_open:
            logger.warning(
                "%s: NBM_FETCH_FAILED (fail_open=False) — rejecting",
                parsed.ticker,
            )
            _reject_scored(RejectionReason.NBM_FETCH_FAILED, exec_price=exec_price)
            return None
    elif nbm.evaluated:
        nbm_temperature = nbm.temperature
        nbm_veto_fired = nbm.veto_fired
        if nbm_veto_fired and cfg.nbm_bracket_position_veto_enabled:
            logger.info(
                "%s: NBM_BRACKET_POSITION_VETO — side=%s NBM=%.2f°F "
                "vs bracket [%.1f, %.1f] (model_prob=%.3f, mid=%.3f)",
                parsed.ticker,
                "YES" if side_is_yes else "NO",
                nbm.temperature,
                parsed.floor_strike,
                parsed.cap_strike,
                model_prob,
                price_info.midpoint,
            )
            _reject_scored(
                RejectionReason.NBM_BRACKET_POSITION_VETO,
                exec_price=exec_price,
                extra={
                    "nbm_side_is_yes": side_is_yes,
                    "nbm_inside_bracket": nbm.inside,
                },
            )
            return None

    raw_edge = abs(model_prob - exec_price)
    fee = kalshi_fee(side_price)
    effective_edge = raw_edge * ensemble_confidence - fee

    if effective_edge < cfg.edge_threshold:
        logger.debug(
            "%s: effective_edge %.3f below threshold %.3f — skipping "
            "(raw=%.3f, conf=%.3f, exec=%.3f, mid=%.3f)",
            parsed.ticker,
            effective_edge,
            cfg.edge_threshold,
            raw_edge,
            ensemble_confidence,
            exec_price,
            price_info.midpoint,
        )
        _reject_scored(
            RejectionReason.EDGE_BELOW_THRESHOLD,
            exec_price=exec_price,
            raw_edge=raw_edge,
            effective_edge=effective_edge,
            fee=fee,
        )
        return None

    if cfg.max_edge < 1.0 and effective_edge > cfg.max_edge:
        logger.warning(
            "%s: effective_edge %.3f > max_edge %.3f — likely stale quote, "
            "skipping (model=%.3f, exec=%.3f, mid=%.3f)",
            parsed.ticker,
            effective_edge,
            cfg.max_edge,
            model_prob,
            exec_price,
            price_info.midpoint,
        )
        _reject_scored(
            RejectionReason.EDGE_ABOVE_MAX,
            exec_price=exec_price,
            raw_edge=raw_edge,
            effective_edge=effective_edge,
            fee=fee,
        )
        return None

    sizing = fractional_kelly(
        model_prob=model_prob,
        market_price=exec_price,
        kelly_fraction=cfg.kelly_fraction,
        max_pct=cfg.max_pct,
        bankroll=bankroll,
        fee_per_contract=fee,
        n_existing_same_day_bets=n_existing_same_day_bets,
        correlation_rho=correlation_rho,
    )
    if sizing.side == Side.NONE or sizing.dollars < 1.0:
        _reject_scored(
            RejectionReason.KELLY_ZERO,
            exec_price=exec_price,
            raw_edge=raw_edge,
            effective_edge=effective_edge,
            fee=fee,
        )
        return None

    contracts = dollars_to_contracts(
        sizing.dollars,
        exec_price,
        sizing.side,
    )

    return _MarketCandidate(
        parsed=parsed,
        market=sc.market,
        forecast=primary_forecast,
        model_prob=model_prob,
        correction_f=correction_f,
        price_info=price_info,
        raw_edge=raw_edge,
        ensemble_confidence=ensemble_confidence,
        effective_edge=effective_edge,
        sizing=sizing,
        contracts=contracts,
        exec_price=exec_price,
        per_model_probs=per_model_probs,
        ensemble_std_pooled=ensemble_std,
        spread_inflation_factor=get_spread_inflation_factor(
            parsed.city,
            parsed.daily_variable,
            spread_inflation_overrides,
        ),
        raw_ensemble_prob=raw_ensemble_prob,
        strike_abs_z=strike_abs_z,
        ensemble_median_pooled=(raw_ensemble_medians.get(fkey) if fkey is not None else None),
        shadow_prob=shadow_prob,
        gfs_uncorrected_prob=gfs_uncorrected_prob,
        **emos.audit_fields(),
        isotonic_artifact_hash=isotonic_artifact_hash,
        gfs_bias_correction_f=bias_corrections.get(fkey, 0.0) if fkey is not None else None,
        live_calibrator_kind=live_calibrator_kind,
        nbm_temperature=nbm_temperature,
        nbm_veto_fired=nbm_veto_fired,
    )
