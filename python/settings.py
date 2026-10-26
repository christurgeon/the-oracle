"""
Typed configuration for The Oracle.

Loads settings from config.yaml and .env, with environment variable overrides.
"""

from __future__ import annotations

import logging
import os
import re
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml  # type: ignore[import-untyped]
from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from python.data.kxrain_observer import KxRainObserverConfig
from python.data.weather import CITY_COORDS
from python.utils.log_config import LoggingConfig
from python.utils.ticker_dates import KNOWN_CITY_CODES

logger = logging.getLogger(__name__)

# Calibration kinds supported by python.data.calibration. Keep this in
# sync with ``CalibrationKind`` there — Pydantic's Literal validates
# strict equality, not structural compatibility.
CalibrationTypeLiteral = Literal["none", "isotonic", "platt", "per_type_platt"]

# Canonical cell-label pattern (``<CITY>/<high|low>``). Shared by the
# weather and intraday cell-list validators.
_CELL_PATTERN = re.compile(r"^[A-Z]{2,5}/(high|low)$")


def _coerce_null_to_empty(value: Any) -> Any:
    """Coerce a YAML ``null`` cell list to ``[]``.

    ``null`` is operator shorthand for "no cells" — same intent as ``[]`` or
    omitting the key. Coercing here lets the after-validator see a list rather
    than raising ``list_type``.
    """
    if value is None:
        return []
    return value


def _validate_cell_list(value: list[str], *, field_label: str) -> list[str]:
    """Validate that each entry matches ``<CITY>/<high|low>``; dedupe silently.

    Rejects typos like ``MIA/HIGH``, ``MIA-high``, or just ``MIA``. Dedupes
    silently — duplicates are a benign typo, not a fatal misconfiguration.
    ``field_label`` is the config-path prefix used in the error message.
    """
    seen: set[str] = set()
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not _CELL_PATTERN.match(entry):
            raise ValueError(
                f"{field_label}: entry {entry!r} must match '<CITY>/<high|low>' (e.g. 'MIA/high')."
            )
        if entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
    return out


def _validate_max_bets_per_day(v: Any) -> Any:
    """Validate ``max_bets_per_day_per_event``: ``>= 1`` integer or ``None``."""
    if v is None:
        return None
    if isinstance(v, int) and not isinstance(v, bool) and v < 1:
        raise ValueError(
            "max_bets_per_day_per_event must be >= 1 or null; use null to disable the cap"
        )
    return v


_HHMM_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _validate_hhmm(value: Any, *, field_label: str) -> str:
    """Validate an ``HH:MM`` 24-hour local-standard-time string."""
    if not isinstance(value, str) or not _HHMM_PATTERN.match(value):
        raise ValueError(
            f"{field_label}={value!r} must be an 'HH:MM' 24-hour string (e.g. '10:00')."
        )
    return value


# ---------------------------------------------------------------------------
# Sub-models (pure Pydantic models — no env loading)
# ---------------------------------------------------------------------------


class ExecutionConfig(BaseModel):
    mode: Literal["shadow", "paper", "live"] = "shadow"
    poll_interval_seconds: int = Field(default=60, ge=1, le=3600)
    websocket_reconnect_delay: int = Field(default=5, ge=0, le=300)
    # Idle bound (seconds) on the WS receive loop; on expiry the client
    # reconnects. MUST stay < risk.stale_data_cutoff_seconds (300) so recovery
    # lands before the STALE_DATA gate halts trading.
    websocket_idle_timeout_seconds: int = Field(default=90, ge=10, le=280)
    # cancel unfilled orders after this many seconds; 0 = disabled
    order_ttl_seconds: int = Field(default=120, ge=0, le=86400)
    # Cross to the ask (taker) when book spread is <= this many cents.
    # Default 1: only 1c-spread orders cross (deterministic-taker rule). Set
    # to 2 for "modified Lever A" — 2c spreads also cross to capture
    # extensive-margin fills the bid-resting path TTL-expires on. >=3 not
    # currently supported by empirical evidence.
    cross_to_ask_max_spread_cents: int = Field(default=1, ge=1, le=2)
    # wait before first trade scan after (re)start; 0 = disabled
    startup_delay_seconds: int = Field(default=30, ge=0, le=3600)
    reconcile_interval_ticks: int = Field(default=15, ge=1, le=10000)
    reconcile_bankroll_threshold: float = Field(default=1.0, ge=0.0, le=1000.0)
    reconcile_position_threshold: float = Field(default=0.5, ge=0.0, le=100.0)
    # Phase 1 feature flag for placement-row writes.
    # Default off; operator flips after verifying consumers handle event="placement".
    emit_placement_rows: bool = False
    # Settlement deferral race-handling.
    max_settle_defer_ticks: int = Field(default=20, ge=1, le=200)
    deferral_aggregate_min_count: int = Field(default=3, ge=1, le=100)
    deferral_aggregate_min_dollars: float = Field(default=25.0, ge=0.0, le=10000.0)
    deferral_aggregate_cooldown_seconds: int = Field(default=1800, ge=60, le=86400)


class FillsReconcilerConfig(BaseModel):
    """Trade-log fidelity reconciler."""

    enabled: bool = False
    reconcile_interval_ticks: int = Field(default=5, ge=1, le=100)
    eventual_consistency_window_seconds: int = Field(default=120, ge=1, le=3600)
    startup_safety_window_hours: int = Field(default=24, ge=1, le=720)
    startup_max_window_hours: int = Field(default=168, ge=1, le=2160)
    max_pages_per_tick: int = Field(default=20, ge=1, le=100)
    max_fills_per_tick: int = Field(default=50, ge=1, le=1000)
    page_delay_ms: int = Field(default=250, ge=0, le=10000)
    placement_row_lookback: int = Field(default=10000, ge=100, le=1_000_000)
    alert_threshold_consecutive: int = Field(default=3, ge=1, le=100)
    alert_threshold_critical_delta: int = Field(default=3, ge=1, le=1000)
    critical_alert_cooldown_seconds: int = Field(default=3600, ge=60, le=86400)
    state_file: str = Field(default="state/fills_reconciler.json", min_length=1)

    # Manual-close reconciliation. Default OFF — shakedown phase.
    manual_close_enabled: bool = False
    # Canary apply cap. 0 = unlimited; 1 = first apply per restart only.
    manual_close_apply_limit_per_restart: int = Field(default=1, ge=0, le=1000)
    # Shakedown dedup state file. Created on demand.
    shakedown_seen_oids_file: str = Field(default="state/shakedown_seen_oids.json", min_length=1)

    # Cash-out reconciliation (v1.1). Default OFF — shakedown phase.
    cash_out_apply_enabled: bool = False
    cash_out_apply_limit_per_restart: int = Field(default=1, ge=0, le=1000)
    # Catastrophic-bug detector. $500 is ~7×-20× the largest plausible
    # single-position P&L under current strategies.
    cash_out_max_pnl_apply_dollars: float = Field(default=500.0, ge=10.0, le=100000.0)
    shakedown_seen_oids_cash_out_file: str = Field(
        default="state/shakedown_seen_oids_cash_out.json", min_length=1
    )


class BracketMonitorConfig(BaseModel):
    enabled: bool = False
    directional_boost_threshold: float = Field(default=0.03, ge=0.0, le=0.5)
    directional_boost_factor: float = Field(default=1.05, ge=1.0, le=2.0)
    max_boost: float = Field(default=1.30, ge=1.0, le=3.0)
    weather_maker_free: bool = True

    @model_validator(mode="after")
    def _check_boost_cap(self) -> BracketMonitorConfig:
        if self.directional_boost_factor > self.max_boost:
            raise ValueError(
                f"bracket_monitor.directional_boost_factor={self.directional_boost_factor} "
                f"must be <= max_boost={self.max_boost}"
            )
        return self


class SameDayTradingSettings(BaseModel):
    """Per-city allow-listed same-day weather-edge trading.

    Default off — empty allow-list means no city is checked even if enabled=True.
    """

    enabled: bool = False
    allowed_cities: list[str] = Field(default_factory=list)
    cache_ttl_seconds: int = Field(default=600, ge=0, le=86400)
    near_peak_ttl_seconds: int = Field(default=90, ge=0, le=86400)
    near_peak_window_minutes: int = Field(default=120, ge=0, le=720)
    peak_cutoff_minutes: int = Field(default=60, ge=0, le=720)
    nws_request_timeout_seconds: float = Field(default=4.0, gt=0.0, le=120.0)
    nws_gate_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)

    @field_validator("allowed_cities", mode="after")
    @classmethod
    def _validate_allowed_cities(cls, value: list[str]) -> list[str]:
        """Canonicalize and validate ``allowed_cities`` against ``CITY_COORDS``.

        Downstream consumers (``same_day_gate.py``, ``weather_edge.py``, and
        the ``Settings`` cross-config deconfliction validator) compare
        entries by exact string equality against uppercase city codes. A
        lowercase / whitespace / unknown entry silently misses, defeating
        the guard. We strip whitespace and require an exact match against
        the canonical ``CITY_COORDS`` set — case mismatches are surfaced as
        operator errors rather than silently uppercased so the typo gets
        fixed at its source. Duplicates are silently deduped (mirror of
        ``WeatherStrategyConfig._validate_emos_enabled_cells``).
        """
        valid_codes = sorted(CITY_COORDS.keys())
        seen: set[str] = set()
        out: list[str] = []
        for entry in value:
            stripped = entry.strip() if isinstance(entry, str) else entry
            if not isinstance(stripped, str) or stripped not in CITY_COORDS:
                raise ValueError(
                    f"strategies.weather.same_day_trading.allowed_cities: "
                    f"entry {stripped!r} is not a valid city code. Valid "
                    f"codes (case-sensitive): {valid_codes}."
                )
            if stripped in seen:
                continue
            seen.add(stripped)
            out.append(stripped)
        return out


class NbmBracketPositionVetoConfig(BaseModel):
    """NBM-bracket-position veto config (shadow-only this release).

    Backtest evidence (n=32 vetoed trades): on between-strike candidates,
    abstaining when NBM lands on the opposite side of the bracket from
    the model's directional bet turned joined-window P&L from a loss to a
    gain. n=32 is below the n≥80 trust floor — this config ships
    ``enabled=False`` so the gate logs ``nbm_temperature`` and
    ``nbm_veto_fired`` on every between-strike eval_audit row WITHOUT
    blocking trades. Flip ``enabled=True`` only after the shadow-promotion
    gate clears (n≥80, positive P&L delta, win-rate gap ≥ 15pp).
    """

    # Master switch. ``False`` (the only value shipped this release) computes
    # ``nbm_veto_fired`` on every between-strike row, writes it to the audit
    # stream, and lets the trade proceed regardless. ``True`` adds the actual
    # rejection — defer until the shadow gate clears.
    enabled: bool = False
    # When the NBM fetch fails (timeout, network, parse error): ``True`` skips
    # the gate entirely (no veto, no audit row contribution beyond null
    # nbm_temperature / nbm_veto_fired). ``False`` rejects the candidate with
    # ``NBM_FETCH_FAILED``. Defaults to fail-open so a transient Open-Meteo
    # outage cannot wedge the live strategy.
    fail_open: bool = True
    # Per-fetch ``asyncio.wait_for`` budget, seconds. Matches the NWS gate
    # request-timeout default (``SameDayTradingSettings.nws_request_timeout_seconds``).
    fetch_timeout_seconds: float = Field(default=4.0, gt=0.0, le=60.0)


class WeatherStrategyConfig(BaseModel):
    enabled: bool = True
    # When True, weather_edge keeps evaluating + logging the calibration corpus
    # but places ZERO orders (the orchestrator's apply_weather_shadow_drop hook
    # suppresses its signals before execution). Default False preserves the
    # historical live behaviour; the deployed config sets it True. Single-flag
    # rollback: set False + restart.
    shadow_only: bool = False
    edge_threshold: float = Field(default=0.08, ge=0.0, le=0.5)
    # disabled by default — weather markets legitimately have huge edges
    max_edge: float = Field(default=1.0, ge=0.01, le=1.0)
    cities: list[str] = Field(
        default_factory=lambda: ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA"],
        min_length=1,
    )
    ensemble_source: str = "open-meteo"
    ensemble_models: list[str] = Field(
        default_factory=lambda: ["gfs_seamless"],
        min_length=1,
    )
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {"gfs_seamless": 1.0},
    )
    # ensemble uncertainty discount (k); 0=off
    spread_skill_sensitivity: float = Field(default=0.0, ge=0.0, le=10.0)
    # skip markets with bid-ask spread wider than this
    max_spread: float = Field(default=0.15, ge=0.0, le=1.0)
    orderbook_depth_check: bool = True  # fetch full orderbook to validate fills
    min_book_depth: int = Field(default=5, ge=0, le=10000)
    # price levels per side requested from Kalshi /orderbook
    book_depth_levels: int = Field(default=10, ge=1, le=100)
    # max parallel /orderbook fetches per scan
    orderbook_concurrency: int = Field(default=2, ge=1, le=32)
    # AUTHORITATIVE cap on concurrent outbound Open-Meteo forecast requests,
    # enforced at the WeatherClient level (wraps the ensemble inner-gather,
    # multi-model fan-out, and NBM point fetches). Open-Meteo's free tier
    # rejects bursts with HTTP 429; a 7-city scan x 2 models can otherwise fire
    # ~8 concurrent requests and trip the rate limit (2026-06 watchdog
    # crash-loop). Mirrors orderbook_concurrency. Single-knob rollback: raise
    # it to effectively disable the cap.
    forecast_concurrency: int = Field(default=2, ge=1, le=32)
    # Per-cell ``asyncio.wait_for`` budget on each multi-model ensemble fetch in
    # weather_edge._prefetch_forecasts (fail-soft: a timed-out cell is skipped
    # for the tick). Mirrors the NBM gate's per-fetch wait_for.
    forecast_fetch_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    # Overall ceiling on the whole ensemble prefetch fan-out per tick. MUST stay
    # well under the systemd WatchdogSec (120s) so external-API slowness can
    # never starve the watchdog -> SIGABRT (2026-06 crash-loop). Raise both
    # timeouts to effectively disable bounding.
    forecast_prefetch_timeout_seconds: float = Field(default=45.0, gt=0.0, le=110.0)
    # per-fetch settle delay seconds; 0 = no sleep
    orderbook_settle_delay_seconds: float = Field(default=0.0, ge=0.0, le=10.0)
    # reject Stage-2 if VWAP drifts more than this from best level (dollars); null = disabled
    max_fill_slippage: float | None = Field(default=None, ge=0.0, le=1.0)
    bias_correction_enabled: bool = False  # default off; production config.yaml overrides to true
    bias_shrinkage: float = Field(default=0.3, ge=0.0, le=1.0)
    bias_max_correction_f: float = Field(default=5.0, ge=0.0, le=20.0)
    # Bias-correction table path. ``None`` => the canonical ERA5-fit table
    # (data/gfs_bias/bias_correction.json) for bit-identical rollback. Set to
    # data/nws_bias/bias_correction.json to use the settlement-anchored (NWS CLI)
    # table.
    bias_table_path: str | None = None
    # Production calibrator. ``None`` defaults to identity.
    calibration_type: CalibrationTypeLiteral | None = None
    # Optional non-default path for the isotonic artifact. ``None`` falls back
    # to ``data/calibration/isotonic.json`` (canonical). Set to a non-default
    # path to ship a refit while preserving the canonical file as the rollback
    # target (2026-05-22 isotonic_v2 bounded-exposure pilot pattern).
    isotonic_path: str | None = Field(default=None, min_length=1)
    # Parallel calibrator whose output is logged for A/B but never
    # applied to trading decisions. ``None`` disables.
    shadow_calibrator: CalibrationTypeLiteral | None = None
    excluded_city_vars: list[str] = Field(default_factory=list)
    # Boundary-distance suppressive filter — per-cell threshold in °F.
    # Path 7a; evidence in docs/research/2026-05-28-boundary-aware-phase0-backtest.md.
    # Empty dict (default) disables the filter for all cells.
    # Example: {"MIA/low": 0.5} drops candidates whose predicted
    # forecast-mean-to-nearest-NWS-rounding-boundary distance is < 0.5°F.
    boundary_filter_cells: dict[str, float] = Field(default_factory=dict)
    # Per-(city, var) spread-inflation overrides for the ensemble-blending
    # path. ``None`` (default) uses the in-code factor table from
    # ``weather_evaluation.get_spread_inflation_factor``. Shape:
    # ``{"NYC": {"high": 1.2, "low": 1.0}, ...}``. Declared so the strategy's
    # ``self._config.get("spread_inflation_factors")`` read does not silently
    # default — bracket-window follow-up #5.
    spread_inflation_factors: dict[str, dict[str, float]] | None = Field(default=None)
    max_bets_per_day_per_event: int | None = Field(default=1)
    # strike-proximity gate (greater/less only); 0=off
    min_distance_from_strike_std: float = Field(default=0.0, ge=0.0, le=10.0)
    # Skip trades whose raw ensemble prob sits inside this range. The isotonic
    # plateau region is genuinely non-monotone — trades there rely on a single
    # pooled calibrated prob with no discrimination. Set to ``null`` to disable.
    block_raw_prob_range: list[float] | None = None
    # Calibrated-probability floor block (live + paper modes only; shadow exempt
    # per generate_signals). Reject candidates whose post-calibration probability
    # is below the floor. Set to ``null`` to disable.
    min_calibrated_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    # Reject BETWEEN candidates whose raw ensemble probability is below this floor.
    min_raw_prob_for_between: float = Field(default=0.05, ge=0.0, le=0.5)
    # Interim EMOS-gap safeguard: reject near-term (lead_days <= 2) markets where
    # |model_prob - market_mid| exceeds this threshold. Open-Meteo GFS ingestion lag
    # (3-6h) can leave short-horizon forecasts stale enough that the market has priced
    # in information our ensemble missed. Set to ``null`` to disable.
    # Remove or relax once EMOS scope-B is live >= 30d and Open-Meteo lag < 1h.
    max_model_market_gap_near_term: float | None = None
    # Diagnostic audit streams; see python/analytics/audit_log.py.
    candidate_audit_log: bool = True
    eval_audit_log: bool = True
    scan_funnel_log: bool = True
    # Retain rotated JSONL siblings for this many days; pruned lazily on
    # each UTC-midnight rotation.
    audit_log_retention_days: int = Field(default=14, ge=1, le=365)
    bracket_monitor: BracketMonitorConfig = Field(default_factory=BracketMonitorConfig)
    same_day_trading: SameDayTradingSettings = Field(default_factory=SameDayTradingSettings)
    # NBM-bracket-position veto (shadow-only this release). Computes
    # ``nbm_veto_fired`` on every between-strike candidate; blocks trades only
    # when ``enabled=True``. See ``NbmBracketPositionVetoConfig`` for the
    # promotion gate.
    nbm_bracket_position_veto: NbmBracketPositionVetoConfig = Field(
        default_factory=NbmBracketPositionVetoConfig
    )
    # EMOS scope-B runtime loader. Disabled by default.
    # When False, EMOS load failures degrade silently (WARNING + ensemble fallback);
    # when True, failures escalate to ERROR + Telegram alert.
    emos_shadow_log: bool = False
    emos_coefficients_path: str = Field(default="data/emos/coefficients.json", min_length=1)
    # Per-cell EMOS live cutover allowlist (Stage A).
    # Cells in this list use ``emos_prob_raw`` as the live model
    # probability; all others use the configured isotonic calibrator.
    # Format: each entry must be ``<CITY>/<high|low>``. Empty list =
    # today's behavior (isotonic everywhere).
    emos_enabled_cells: list[str] = Field(default_factory=list)
    # Runtime σ_emos / bracket_width gate (Tier 2 A).
    # Reject EMOS-live candidates whose predictive σ_emos = √(c + d·ens_std²)
    # at the live ens_std exceeds this multiple of the 1°F KXHIGH/KXLOWT
    # bracket width. Mirrors the fit-time gate's
    # ``DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD = 2.0``. Set to a very large
    # value (e.g. 9999) to disable. Upper bound 10000 accommodates the
    # documented rollback value.
    emos_runtime_sigma_threshold: float = Field(default=2.0, ge=0.0, le=10000.0)
    # Same-day Kelly haircut (F4.1). Pairwise outcome correlation ρ applied
    # across already-open trades sharing a ``forecast_date``. The haircut
    # is ``1/√(1 + n_existing · ρ)`` and is applied inside
    # ``fractional_kelly`` BEFORE the per-market cap so the cap stays a hard
    # ceiling. ``None`` disables the haircut entirely (legacy independent
    # Kelly). Default ``0.0`` ships the infrastructure but is a no-op until
    # operator activation — empirical analysis (n=1392 cross-(city,var)
    # pairs, 95% bootstrap CI [-0.05, +0.09]) found ρ statistically
    # indistinguishable from zero on the post-event-cap regime; the knob is
    # in place for future activation if a regime shift makes correlation
    # visible.
    same_day_correlation_rho: float | None = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("edge_threshold", mode="before")
    @classmethod
    def _validate_edge_threshold_scale(cls, value: Any) -> Any:
        """Catch probability-scale confusion (e.g. 7 instead of 0.07)."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0.5 or value < 0.0:
                raise ValueError(
                    f"strategies.weather.edge_threshold = {value}: must be in [0, 0.5]. "
                    f"Edge is a probability-scale quantity. Did you mean {value / 100}?"
                )
        return value

    @model_validator(mode="after")
    def _check_edge_threshold_below_max(self) -> WeatherStrategyConfig:
        if self.edge_threshold >= self.max_edge:
            raise ValueError(
                f"strategies.weather.edge_threshold={self.edge_threshold} must be "
                f"strictly less than max_edge={self.max_edge}"
            )
        return self

    @model_validator(mode="after")
    def _check_ensemble_weights(self) -> WeatherStrategyConfig:
        weight_keys = set(self.ensemble_weights.keys())
        model_keys = set(self.ensemble_models)
        extras = weight_keys - model_keys
        if extras:
            raise ValueError(
                f"strategies.weather.ensemble_weights has keys not in ensemble_models: "
                f"{sorted(extras)}. Declared models: {sorted(model_keys)}"
            )
        # Sanity check on the nominal weights; runtime renormalizes to
        # available models so a small drift is tolerable.
        total = sum(self.ensemble_weights.values())
        if self.ensemble_weights and abs(total - 1.0) > 0.01:
            raise ValueError(
                f"strategies.weather.ensemble_weights must sum to ~1.0; got {total:.4f}"
            )
        return self

    @field_validator("max_bets_per_day_per_event", mode="before")
    @classmethod
    def _validate_max_bets(cls, v: Any) -> Any:
        return _validate_max_bets_per_day(v)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_one_bet_per_event(cls, data: Any) -> Any:
        if isinstance(data, dict) and "one_bet_per_event" in data:
            raise ValueError(
                "Legacy config field 'one_bet_per_event' is removed. "
                "Use 'max_bets_per_day_per_event: int | null' instead "
                "(see docs/operations-runbook.md, 'Disabling the cap' section)."
            )
        return data

    @field_validator("emos_enabled_cells", mode="before")
    @classmethod
    def _coerce_emos_enabled_cells_null(cls, value: Any) -> Any:
        # YAML ``emos_enabled_cells: null`` is operator shorthand for "no EMOS
        # cells" — same intent as ``[]`` or omitting the key.
        return _coerce_null_to_empty(value)

    @field_validator("emos_enabled_cells", mode="after")
    @classmethod
    def _validate_emos_enabled_cells(cls, value: list[str]) -> list[str]:
        """Each entry must match ``<CITY>/<high|low>``.

        Rejects typos like ``MIA/HIGH`` (silent isotonic), ``MIA-high``,
        or just ``MIA``. Dedupes silently — duplicates are a benign
        typo, not a fatal misconfiguration.
        """
        return _validate_cell_list(value, field_label="strategies.weather.emos_enabled_cells")

    @field_validator("boundary_filter_cells", mode="before")
    @classmethod
    def _coerce_boundary_filter_null(cls, value: Any) -> Any:
        # YAML ``boundary_filter_cells: null`` is operator shorthand for "no
        # boundary filter on any cell". Mirrors _coerce_emos_enabled_cells_null.
        if value is None:
            return {}
        return value

    @field_validator("boundary_filter_cells", mode="after")
    @classmethod
    def _validate_boundary_filter_cells(cls, value: dict[str, float]) -> dict[str, float]:
        # Validate cell-label format AND threshold range. Rejects at Settings.load()
        # so a bad config crashes the bot at startup, not mid-scan.
        for cell, threshold in value.items():
            if "/" not in cell:
                raise ValueError(
                    f"strategies.weather.boundary_filter_cells: cell {cell!r} "
                    "must be of the form '<CITY>/<var>' (e.g., 'MIA/low')."
                )
            _city, var = cell.split("/", 1)
            if var not in {"high", "low"}:
                raise ValueError(
                    f"strategies.weather.boundary_filter_cells: cell {cell!r} "
                    "must end in '/high' or '/low'."
                )
            if threshold < 0:
                raise ValueError(
                    f"strategies.weather.boundary_filter_cells[{cell!r}]: "
                    f"threshold must be >= 0 (got {threshold})."
                )
            if threshold > 50.0:
                raise ValueError(
                    f"strategies.weather.boundary_filter_cells[{cell!r}]: "
                    f"threshold {threshold} implausibly large (max 50°F)."
                )
        return value


class IntradayEmosHighStrategyConfig(BaseModel):
    """Intraday EMOS HIGH strategy config.

    Pydantic-enforced bounds protect the v6-audited posture from silent
    operator drift. The strategy module itself lands in milestone M4; M2
    ships the config plumbing with ``enabled = False`` so the loader can
    accept the YAML block without activating the strategy.
    """

    enabled: bool = False
    shadow_only: bool = True
    enabled_cells: list[str] = Field(default_factory=list)
    edge_threshold: float = Field(default=0.07, ge=0.02, le=0.30)
    max_edge: float = Field(default=0.30, ge=0.01, le=1.0)
    # v6 audited value — operator MAY tighten (raise), MUST NOT relax (lower).
    min_distance_from_strike_std: float = Field(default=2.0, ge=2.0, le=10.0)
    # v6 audited cap — operator MAY tighten (lower), MUST NOT exceed.
    max_contracts_per_trade: int = Field(default=50, ge=1, le=50)
    max_trades_per_cell_per_day: int = Field(default=3, ge=1, le=10)
    # Hot-reload pause window. 60s matches the EmosTable build-then-swap envelope.
    emos_hot_reload_pause_seconds: int = Field(default=60, ge=0, le=600)
    # Audit-log toggles + retention. Declared (rather than inherited via the
    # orchestrator's merge with ``weather``) so the operator can override per
    # strategy without affecting ``weather_edge``. Bracket-window follow-up #5.
    eval_audit_log: bool = Field(default=True)
    candidate_audit_log: bool = Field(default=True)
    scan_funnel_log: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=14, ge=1, le=365)

    @field_validator("enabled_cells", mode="before")
    @classmethod
    def _coerce_enabled_cells_null(cls, value: Any) -> Any:
        # YAML ``enabled_cells: null`` is operator shorthand for "no cells" —
        # same intent as ``[]`` or omitting the key. Symmetric with
        # ``WeatherStrategyConfig._coerce_emos_enabled_cells_null``.
        return _coerce_null_to_empty(value)

    @field_validator("enabled_cells", mode="after")
    @classmethod
    def _validate_cells_format(cls, value: list[str]) -> list[str]:
        """Each entry must match ``<CITY>/<high|low>``.

        Mirrors ``WeatherStrategyConfig._validate_emos_enabled_cells``: rejects
        typos like ``MIA/HIGH``, ``MIA-high``, or just ``MIA``. Dedupes
        silently — duplicates are a benign typo, not a fatal misconfig.
        """
        return _validate_cell_list(value, field_label="strategies.intraday_emos_high.enabled_cells")

    @model_validator(mode="after")
    def _check_edge_threshold_below_max(self) -> IntradayEmosHighStrategyConfig:
        if self.edge_threshold >= self.max_edge:
            raise ValueError(
                f"strategies.intraday_emos_high.edge_threshold={self.edge_threshold} "
                f"must be strictly less than max_edge={self.max_edge}"
            )
        return self


class FavoriteBiasStrategyConfig(BaseModel):
    """Model-free favorite-longshot strategy (greater/less brackets).

    Per-strategy sizing knobs are deliberately separate from the global
    ``risk.kelly_fraction`` — this strategy sizes more conservatively than
    weather_edge.
    """

    enabled: bool = False
    band_low: float = Field(default=0.40, gt=0.0, lt=1.0)
    band_high: float = Field(default=0.75, gt=0.0, lt=1.0)
    min_volume_fp: int = Field(default=200, ge=0)
    edge_floor: float = Field(default=0.02, ge=0.0, le=0.5)
    # Reject when yes_ask - yes_bid exceeds this (dollars). None = gate off.
    max_spread: float | None = Field(default=None, gt=0.0, lt=1.0)
    lead: int = Field(default=1, ge=1, le=2)
    strike_types: list[str] = Field(default_factory=lambda: ["greater", "less"], min_length=1)
    cities: list[str] = Field(
        default_factory=lambda: ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "BOS", "PHIL"],
        min_length=1,
    )
    isotonic_path: str = "data/calibration/favorite_bias_isotonic.json"
    # Per-strategy sizing (NOT the global risk.kelly_fraction).
    kelly_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    max_pct: float = Field(default=0.05, gt=0.0, le=0.25)
    max_contracts_per_trade: int = Field(default=25, ge=1, le=200)
    # Bounded book-walk: price the taker limit up to this many cents above the
    # signal-time ask, only through levels whose own edge clears edge_floor.
    # 0 = limit at the top-of-book ask (legacy behavior).
    max_book_walk_cents: int = Field(default=0, ge=0, le=5)
    max_bets_per_day_per_event: int | None = Field(default=1)
    audit_path: str = "state/favorite_bias_audit.jsonl"
    execution_taker: bool = True  # cross to ask (matches backtest fill model)
    # Fill-timing study, phase 1: rest a taker order at the frozen signal-time ask for
    # this many seconds instead of the global execution.order_ttl_seconds (120).
    # None = no change (byte-identical to today). Set >= execution.poll_interval_seconds
    # (60) so the order survives at least one tick.
    resting_ttl_seconds: int | None = Field(default=None, ge=1, le=86400)
    # (city,var) exclusion — drops a persistently-negative cell from the pooled
    # g(mid) book. Format "{CITY}_{var}", CITY a canonical ticker city code and var
    # in {high,low} (mirrors the runtime key f"{city}_{var}"). Validated below;
    # default [] = no-op.
    excluded_city_vars: list[str] = Field(default_factory=list)
    # --- Strike-climatology instrumentation + tail guard (spec 2026-07-03) ---
    # Artifact built offline by scripts/build_fb_climatology.py; loaded fail-open
    # (a missing/corrupt artifact disables only the new features, never trading).
    climatology_path: str = "data/calibration/fb_climatology.json"
    # Tail-strike suppression guard. None = OFF (default). When set, suppress a
    # signal whose |strike z-score vs (city,var,month) climatology| >= this, but
    # ONLY on source=="month" cells with clim_n >= clim_min_n. VARIANCE INSURANCE,
    # NO DEMONSTRATED EV (backtest ROI-neutral-to-slightly-negative). Arm at
    # operator discretion. Reversible (null + restart).
    max_strike_climatology_z: float | None = Field(default=None, ge=0.5, le=6.0)
    # Minimum monthly-cell obs before the guard may fire (thin cells fail open).
    clim_min_n: int = Field(default=15, ge=1)

    @field_validator("max_bets_per_day_per_event", mode="before")
    @classmethod
    def _validate_max_bets(cls, v: Any) -> Any:
        return _validate_max_bets_per_day(v)

    @field_validator("excluded_city_vars")
    @classmethod
    def _validate_excluded_city_vars(cls, v: list[str]) -> list[str]:
        # An entry that doesn't match the runtime key f"{city}_{var}" silently fails
        # to exclude the cell — unacceptable on a live safety lever — so reject typos
        # like "NY_high" (raw suffix, not "NYC") or "sea_high" (wrong case) at load.
        # City is checked against KNOWN_CITY_CODES (the ticker parser's own output);
        # codes contain no "_", so rsplit("_", 1) splits city/var cleanly.
        for cell in v:
            parts = cell.rsplit("_", 1)
            ok = len(parts) == 2 and parts[1] in ("high", "low") and parts[0] in KNOWN_CITY_CODES
            if not ok:
                raise ValueError(
                    f"strategies.favorite_bias.excluded_city_vars entry {cell!r} must be "
                    f'"{{CITY}}_{{var}}" with var in {{high,low}} and CITY in '
                    f"{sorted(KNOWN_CITY_CODES)} (e.g. \"BOS_high\") — an entry that "
                    "does not match the runtime key silently fails to exclude the cell"
                )
        return v

    @model_validator(mode="after")
    def _check_band_ordered(self) -> FavoriteBiasStrategyConfig:
        if self.band_low >= self.band_high:
            raise ValueError(
                f"strategies.favorite_bias.band_low={self.band_low} must be "
                f"< band_high={self.band_high}"
            )
        return self


class IntradayAsosHighStrategyConfig(BaseModel):
    """Intraday ASOS HIGH strategy config.

    Pydantic-enforced bounds protect the shadow-only pilot posture from silent
    operator drift.  The strategy is HIGH-only (``/high`` cells exclusively);
    the validator rejects any non-HIGH entry in ``enabled_cells`` or
    ``informational_cells``, and rejects any overlap between the two lists.

    Notes on defaults:
        ``max_contracts_per_trade`` defaults to 15 (tighter than the EMOS
        strategy's 50 — ASOS is a newer posterior and we start conservatively).
        ``enabled`` defaults to ``False`` so the bare model can be constructed
        in tests without a YAML file.
    """

    enabled: bool = False
    shadow_only: bool = True
    enabled_cells: list[str] = Field(default_factory=list)
    informational_cells: list[str] = Field(default_factory=list)
    climatology_path: str = "data/asos/climatology.json"
    posterior_t_dof: float = 5.0
    boundary_distance_min_f: float = 1.0
    asos_max_staleness_hours: float = 2.0
    sigma_gain_floor_f: float = 0.5
    bad_prelim_max_drift_f: float = 10.0
    bad_prelim_active: bool = False
    edge_threshold: float = Field(default=0.07, ge=0.02, le=0.30)
    min_calibrated_prob: float = 0.110
    # Per-strategy sizing knobs — deliberately separate from the global
    # ``risk.kelly_fraction`` and ``risk.max_single_market_pct`` so the
    # operator can tune ASOS sizing independently without touching the
    # firm-wide risk config.  Mirror ``FavoriteBiasStrategyConfig``.
    kelly_fraction: float = Field(default=0.075, ge=0.0, le=0.5)
    max_single_market_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    max_contracts_per_trade: int = Field(default=15, ge=1, le=50)
    max_trades_per_cell_per_day: int = Field(default=3, ge=1, le=10)
    orderbook_depth_check: bool = True
    # IEM/ASOS observation cache TTL. ASOS/METAR publishes ~hourly, so 600s
    # (10 min) matches the data cadence and keeps IEM request volume low — the
    # prior hardcoded 120s default refetched every ~2 ticks and triggered IEM
    # 429s. Bounded [60, 3600]: too short re-hammers IEM, too long serves stale
    # max-so-far (capped well under the 2h asos_max_staleness_hours guard).
    asos_obs_cache_ttl_seconds: float = Field(default=600.0, ge=60.0, le=3600.0)

    @field_validator("enabled_cells", "informational_cells", mode="before")
    @classmethod
    def _coerce_cell_list_null(cls, value: Any) -> Any:
        # YAML ``enabled_cells: null`` or ``informational_cells: null`` is
        # operator shorthand for "no cells" — same intent as ``[]`` or
        # omitting the key.
        return _coerce_null_to_empty(value)

    @field_validator("enabled_cells", "informational_cells", mode="after")
    @classmethod
    def _validate_cells_format(cls, value: list[str]) -> list[str]:
        """Each entry must match ``<CITY>/<high|low>``.

        Rejects typos like ``MIA/HIGH``, ``MIA-high``, or just ``MIA``.
        Dedupes silently — duplicates are a benign typo, not a fatal misconfig.
        """
        return _validate_cell_list(value, field_label="strategies.intraday_asos_high.cells")

    @model_validator(mode="after")
    def _check_asos_invariants(self) -> IntradayAsosHighStrategyConfig:
        """Enforce three structural invariants.

        1. If ``enabled``, ``enabled_cells`` must be non-empty — an active
           strategy with no cells is a silent misconfiguration.
        2. Every entry in ``enabled_cells`` and ``informational_cells`` must
           end with ``/high`` — this strategy is HIGH-only.
        3. ``enabled_cells`` and ``informational_cells`` must be disjoint — a
           cell in both lists would produce ambiguous routing.
        """
        # Invariant 1: enabled → non-empty enabled_cells.
        if self.enabled and not self.enabled_cells:
            raise ValueError(
                "strategies.intraday_asos_high.enabled_cells must be non-empty "
                "when strategies.intraday_asos_high.enabled=True. "
                "Add at least one '/high' cell or set enabled=False."
            )

        # Invariant 2: HIGH-only — every cell must end with '/high'.
        all_configured = self.enabled_cells + self.informational_cells
        non_high = [c for c in all_configured if not c.endswith("/high")]
        if non_high:
            raise ValueError(
                f"strategies.intraday_asos_high: cell(s) {sorted(non_high)} are "
                f"not HIGH cells. The intraday ASOS strategy is HIGH-only; "
                f"each cell must end with '/high' (e.g. 'MIA/high')."
            )

        # Invariant 3: enabled_cells ∩ informational_cells must be empty.
        overlap = set(self.enabled_cells) & set(self.informational_cells)
        if overlap:
            raise ValueError(
                f"strategies.intraday_asos_high: cell(s) {sorted(overlap)} appear "
                f"in both enabled_cells and informational_cells. Each cell must be "
                f"in at most one category."
            )

        return self


class StrategiesConfig(BaseModel):
    weather: WeatherStrategyConfig = Field(default_factory=WeatherStrategyConfig)
    intraday_emos_high: IntradayEmosHighStrategyConfig = Field(
        default_factory=IntradayEmosHighStrategyConfig
    )
    intraday_asos_high: IntradayAsosHighStrategyConfig = Field(
        default_factory=IntradayAsosHighStrategyConfig
    )
    favorite_bias: FavoriteBiasStrategyConfig = Field(default_factory=FavoriteBiasStrategyConfig)


class RiskConfig(BaseModel):
    max_single_market_pct: float = Field(default=0.05, gt=0.0, le=0.25)
    max_total_exposure_pct: float = Field(default=0.40, gt=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.05, gt=0.0, le=0.5)
    max_drawdown_pct: float = Field(default=0.15, gt=0.0, le=0.5)
    drawdown_size_reduction: float = Field(default=0.50, gt=0.0, lt=1.0)
    kelly_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    max_correlated_positions: int = Field(default=3, ge=1, le=100)
    stale_data_cutoff_seconds: int = Field(default=300, ge=1, le=3600)
    strategy_allocations: dict[str, float] = Field(default_factory=dict)
    """Per-strategy virtual bankroll, as a fraction of total equity.

    Empty (default) means every strategy sizes off full equity (legacy behavior).
    Keys are runtime strategy names (e.g. "weather_edge"); unknown-key and
    enabled-coverage validation live in the orchestrator (it knows the
    registered names — RiskConfig does not). Values must sum to <= 1.0.
    """

    @model_validator(mode="after")
    def _check_cross_field_invariants(self) -> RiskConfig:
        if self.max_daily_loss_pct >= self.max_drawdown_pct:
            raise ValueError(
                f"risk.max_daily_loss_pct={self.max_daily_loss_pct} must be "
                f"strictly less than max_drawdown_pct={self.max_drawdown_pct}; "
                f"the daily halt should trip before the drawdown halt"
            )
        effective = self.kelly_fraction * self.max_single_market_pct
        if not (0.0005 <= effective <= 0.10):
            raise ValueError(
                f"risk.kelly_fraction * risk.max_single_market_pct = "
                f"{effective:.5f} (got kelly={self.kelly_fraction}, "
                f"market_cap={self.max_single_market_pct}): must be in "
                f"[0.0005, 0.10]. Too low means no trade ever sizes above "
                f"noise; too high combines Kelly with a loose cap and risks "
                f"blow-up."
            )
        if self.strategy_allocations:
            for name, pct in self.strategy_allocations.items():
                if not (0.0 <= pct <= 1.0):
                    raise ValueError(
                        f"risk.strategy_allocations[{name!r}]={pct} must be in [0.0, 1.0]."
                    )
            total = sum(self.strategy_allocations.values())
            if total > 1.0:
                raise ValueError(
                    f"risk.strategy_allocations sum {total:.4f} exceeds 1.0; "
                    f"the pods would over-commit total equity."
                )
            if total > self.max_total_exposure_pct:
                logger.warning(
                    "risk.strategy_allocations sum %.4f exceeds "
                    "max_total_exposure_pct %.4f — pods will compete for the "
                    "firm cap (starvation risk). Set the sum <= the firm cap so "
                    "per-pod books bind first.",
                    total,
                    self.max_total_exposure_pct,
                )
        return self


class CircuitBreakerConfig(BaseModel):
    rolling_window_days: int = Field(default=7, ge=1, le=90)
    max_loss_pct: float = Field(default=0.03, gt=0.0, le=0.5)
    min_win_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    min_trades_for_win_rate: int = Field(default=20, ge=1, le=10000)
    data_staleness_multiplier: int = Field(default=2, ge=1, le=20)


class TelegramConfig(BaseModel):
    """Telegram notifier configuration.

    ``silent_bot_alert_hours`` thresholds the aggregate SILENT BOT watchdog over
    the whole tick loop. ``strategy_silent_alert_hours`` thresholds the
    per-strategy liveness detector and is deliberately independent and larger: a
    single strategy can legitimately produce nothing for a full trading day
    (favorite_bias is lead-1, band 0.50-0.75, one bet per event per day), so a
    shared 6-12h threshold would false-positive on ordinary quiet days.
    """

    enabled: bool = False
    poll_interval: float = Field(default=1.0, gt=0.0, le=60.0)
    max_messages_per_minute: int = Field(default=20, ge=1, le=30)
    silent_bot_alert_hours: float = Field(default=6.0, gt=0.0, le=168.0)
    strategy_silent_alert_hours: float = Field(default=30.0, gt=0.0, le=168.0)


class AttributionConfig(BaseModel):
    """P&L attribution engine configuration.

    Attribution is read-only analytics over the shadow settlements file.
    No background loop; Telegram handlers call into it on demand.
    """

    enabled: bool = True
    settlements_path: str = Field(default="data/shadow_settlements.full.json", min_length=1)


class CalibrationMonitorConfig(BaseModel):
    """Live calibration monitor configuration.

    See ``python.analytics.calibration_monitor.CalibrationMonitor``.
    """

    enabled: bool = False
    tick_interval_seconds: float = Field(default=300.0, gt=0.0, le=86400.0)
    warmup_pairs: int = Field(default=40, ge=0, le=10000)
    brier_warn_mult: float = Field(default=1.3, ge=1.0, le=10.0)
    brier_crit_mult: float = Field(default=1.6, ge=1.0, le=10.0)
    bin_deviation_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    # Master switch for the absolute-miscalibration ``bin_deviation`` alert.
    # ``False`` silences it without affecting the relative ``brier_7d_vs_30d``
    # drift alert or the ``city_var_brier`` economic alert. See the dataclass
    # docstring in ``python.analytics.calibration_monitor``.
    bin_deviation_enabled: bool = True
    bin_min_n: int = Field(default=20, ge=1, le=10000)
    city_var_brier_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    city_var_min_n: int = Field(default=20, ge=1, le=10000)
    alert_cooldown_seconds: float = Field(default=4 * 3600.0, ge=0.0, le=7 * 86400.0)
    state_path: str = Field(default="state/calibration_monitor.json", min_length=1)
    snapshots_dir: str = Field(default="data/snapshots", min_length=1)
    prediction_log_path: str = Field(default="state/predictions.jsonl", min_length=1)
    # F8.1: outcome source for prediction-vs-outcome join.
    # ``"kalshi"`` (default) joins ``kalshi_settlements_path`` on
    # ``market_ticker`` — same authoritative settlements the trade
    # dashboards use. ``"nws"`` is the legacy NWS-CLI join (kept available
    # as a fast-path lead indicator; not for kill-decision Brier).
    settlement_source: Literal["kalshi", "nws"] = "kalshi"
    kalshi_settlements_path: str = Field(default="data/shadow_settlements.full.json", min_length=1)
    # 2026-07-04: corpus source for the prediction side of the join.
    # ``"eval_audit"`` (default) reads ``state/eval_audit.jsonl`` — written at
    # Stage-1 for every evaluated market, before depth/Kelly/allocation, so a
    # $0 pod cannot dark the corpus (the 2026-06-16 predictions.jsonl freeze).
    # ``"predictions"`` is the legacy signal-gated stream, kept as a rollback.
    corpus_source: Literal["eval_audit", "predictions"] = "eval_audit"
    eval_audit_path: str = Field(default="state/eval_audit.jsonl", min_length=1)

    @model_validator(mode="after")
    def _check_brier_mult_order(self) -> CalibrationMonitorConfig:
        if self.brier_warn_mult >= self.brier_crit_mult:
            raise ValueError(
                f"calibration_monitor.brier_warn_mult={self.brier_warn_mult} must be "
                f"strictly less than brier_crit_mult={self.brier_crit_mult}"
            )
        return self


class MarketFilterConfig(BaseModel):
    min_volume: int = Field(default=50, ge=0, le=1_000_000)
    # skip markets with fewer open contracts (stale quotes)
    min_open_interest: int = Field(default=5, ge=0, le=1_000_000)
    min_time_to_expiry_seconds: int = Field(default=3600, ge=0, le=30 * 86400)
    min_spread: float = Field(default=0.03, ge=0.0, le=0.5)
    # skip penny markets (midpoint below this)
    min_price: float = Field(default=0.03, gt=0.0, lt=1.0)
    # skip reverse-penny markets (midpoint above this)
    max_price: float = Field(default=0.97, gt=0.0, lt=1.0)
    # cap contracts at this fraction of best-level size
    max_orderbook_take_pct: float = Field(default=0.50, gt=0.0, le=1.0)

    @field_validator("min_price", mode="before")
    @classmethod
    def _validate_min_price_scale(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value >= 1.0 or value <= 0.0:
                raise ValueError(
                    f"market_filter.min_price = {value}: must be in (0, 1). "
                    f"Kalshi prices are probabilities in dollar units "
                    f"(e.g. 0.03 = 3¢). Did you mean {value / 100}?"
                )
        return value

    @field_validator("max_price", mode="before")
    @classmethod
    def _validate_max_price_scale(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value >= 1.0 or value <= 0.0:
                raise ValueError(
                    f"market_filter.max_price = {value}: must be in (0, 1). "
                    f"Kalshi prices are probabilities in dollar units "
                    f"(e.g. 0.97 = 97¢). Did you mean {value / 100}?"
                )
        return value

    @model_validator(mode="after")
    def _check_price_band_width(self) -> MarketFilterConfig:
        if self.min_price + 0.10 > self.max_price:
            raise ValueError(
                f"market_filter.min_price={self.min_price} and max_price="
                f"{self.max_price} must span at least a 0.10 band; the filter "
                f"would admit no markets."
            )
        return self


class KxRainShadowConfig(BaseModel):
    """KXRAIN shadow observer config. Logging only, zero orders.

    Top-level, not under ``strategies``: a ``risk.strategy_allocations`` entry
    for a non-strategy crash-loops the bot at startup.
    """

    enabled: bool = False
    series_ticker: str = "KXRAIN"
    tick_interval_seconds: float = Field(default=600, ge=60, le=3600)
    # Local standard time, never DST.
    entry_window_start_lst: str = "10:00"
    entry_window_end_lst: str = "18:00"
    yes_bid_min: float = Field(default=0.10, ge=0.0, le=1.0)
    yes_bid_max: float = Field(default=0.90, ge=0.0, le=1.0)
    no_trade_lookback_minutes: int = Field(default=30, ge=1, le=240)
    max_entries_per_day: int | None = None
    assumed_max_walk_cents: int = Field(default=1, ge=0, le=10)
    depth_levels: int = Field(default=10, ge=1, le=50)
    depth_capture_mode: Literal["band", "all"] = "band"
    max_requests_per_tick: int = Field(default=60, ge=1, le=200)
    request_pacing_seconds: float = Field(default=1.0, ge=0.0, le=5.0)
    tick_wall_clock_budget_seconds: float = Field(default=180, ge=10, le=600)
    post_entry_follow_minutes: int = Field(default=45, ge=0, le=240)
    # A floor of 0 would silently disable the disk guard.
    min_free_disk_bytes: int = Field(default=2_147_483_648, ge=100_000_000)
    log_dir: str = "state"
    log_retention_days: int = Field(default=240, ge=7, le=730)
    trades_page_limit: int = Field(default=100, ge=1, le=1000)
    settlement_capture_hour_utc: int = Field(default=14, ge=0, le=23)
    settlement_sweep_max_pages: int = Field(default=8, ge=1, le=100)

    @field_validator("entry_window_start_lst")
    @classmethod
    def _validate_entry_window_start(cls, v: str) -> str:
        return _validate_hhmm(v, field_label="kxrain_shadow.entry_window_start_lst")

    @field_validator("entry_window_end_lst")
    @classmethod
    def _validate_entry_window_end(cls, v: str) -> str:
        return _validate_hhmm(v, field_label="kxrain_shadow.entry_window_end_lst")

    @field_validator("max_entries_per_day")
    @classmethod
    def _validate_max_entries_per_day(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 50):
            raise ValueError(f"kxrain_shadow.max_entries_per_day={v} must be null or in [1, 50]")
        return v

    @model_validator(mode="after")
    def _check_price_band_ordered(self) -> KxRainShadowConfig:
        if self.yes_bid_min >= self.yes_bid_max:
            raise ValueError(
                f"kxrain_shadow.yes_bid_min={self.yes_bid_min} must be < "
                f"yes_bid_max={self.yes_bid_max}"
            )
        return self

    def to_observer_config(self, state_dir: Path | str | None = None) -> KxRainObserverConfig:
        """Build the runtime observer config.

        ``state_dir`` overrides ``log_dir`` so the disk guard measures the
        filesystem that holds the risk state.
        """
        return KxRainObserverConfig(
            log_dir=Path(state_dir if state_dir is not None else self.log_dir),
            series_ticker=self.series_ticker,
            tick_interval_seconds=self.tick_interval_seconds,
            entry_window_start_lst=datetime.strptime(self.entry_window_start_lst, "%H:%M").time(),
            entry_window_end_lst=datetime.strptime(self.entry_window_end_lst, "%H:%M").time(),
            yes_bid_min=self.yes_bid_min,
            yes_bid_max=self.yes_bid_max,
            no_trade_lookback_minutes=self.no_trade_lookback_minutes,
            max_entries_per_day=self.max_entries_per_day,
            assumed_max_walk_cents=self.assumed_max_walk_cents,
            depth_levels=self.depth_levels,
            depth_capture_mode=self.depth_capture_mode,
            max_requests_per_tick=self.max_requests_per_tick,
            request_pacing_seconds=self.request_pacing_seconds,
            tick_wall_clock_budget_seconds=self.tick_wall_clock_budget_seconds,
            post_entry_follow_minutes=self.post_entry_follow_minutes,
            min_free_disk_bytes=self.min_free_disk_bytes,
            log_retention_days=self.log_retention_days,
            trades_page_limit=self.trades_page_limit,
        )


# ---------------------------------------------------------------------------
# Root settings — merges config.yaml + environment variables
# ---------------------------------------------------------------------------


def _load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and return as a dict. Returns empty dict on failure."""
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


CONFIG_OVERLAY_ENV = "ORACLE_CONFIG_OVERLAY"

# Subtrees a private deployment overlay may set. Everything else must come from
# the checked-in config.yaml: several scripts read ``strategies.weather.*``
# straight from that file and would silently disagree with the running bot.
CONFIG_OVERLAY_ROOTS: tuple[tuple[str, ...], ...] = (
    ("strategies", "favorite_bias"),
    ("risk",),
    ("circuit_breakers",),
    ("kxrain_shadow",),
    ("fills_reconciler",),
    ("telegram",),
)


def config_overlay_path() -> Path | None:
    """Resolve ``ORACLE_CONFIG_OVERLAY`` from the environment, then ``./.env``.

    The ``.env`` fallback lets every entry point that runs from the deploy
    directory (systemd units, ``deploy/update.sh``, scripts) agree on the
    overlay without exporting the variable. An empty value disables it.
    """
    raw = os.environ.get(CONFIG_OVERLAY_ENV)
    if raw is None:
        env_file = Path(".env")
        if env_file.is_file():
            raw = dotenv_values(env_file).get(CONFIG_OVERLAY_ENV)
    return Path(raw) if raw else None


def _submodel(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        models = [a for a in get_args(annotation) if _submodel(a) is not None]
        if len(models) == 1:
            return _submodel(models[0])
    return None


def _check_overlay_roots(node: dict[str, Any], path: tuple[str, ...] = ()) -> None:
    for key, value in node.items():
        here = (*path, key)
        if here in CONFIG_OVERLAY_ROOTS:
            continue
        if isinstance(value, dict) and any(r[: len(here)] == here for r in CONFIG_OVERLAY_ROOTS):
            _check_overlay_roots(value, here)
            continue
        raise ValueError(
            f"config overlay may not set {'.'.join(here)}; allowed roots: "
            + ", ".join(".".join(r) for r in CONFIG_OVERLAY_ROOTS)
        )


def _merge_overlay(
    base: dict[str, Any],
    overlay: dict[str, Any],
    model: type[BaseModel],
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Merge ``overlay`` into ``base``, recursing only into pydantic sub-models.

    Map- and list-typed fields are replaced whole so an overlay can remove
    entries (e.g. drop a ``risk.strategy_allocations`` key for a parked
    strategy). Unknown keys raise: a typo must never fall back to the base.
    """
    merged = dict(base)
    for key, value in overlay.items():
        here = (*path, key)
        field = model.model_fields.get(key)
        if field is None:
            raise ValueError(f"config overlay sets unknown key {'.'.join(here)}")
        sub = _submodel(field.annotation)
        if sub is not None and isinstance(value, dict):
            base_value = base.get(key)
            merged[key] = _merge_overlay(
                base_value if isinstance(base_value, dict) else {}, value, sub, here
            )
        else:
            merged[key] = value
    return merged


def apply_config_overlay(yaml_data: dict[str, Any], overlay_path: Path) -> dict[str, Any]:
    if not overlay_path.is_file():
        raise FileNotFoundError(f"{CONFIG_OVERLAY_ENV} points at a missing file: {overlay_path}")
    with open(overlay_path) as f:
        overlay = yaml.safe_load(f) or {}
    if not isinstance(overlay, dict):
        raise ValueError(f"config overlay {overlay_path} must be a YAML mapping")
    _check_overlay_roots(overlay)
    logger.info("Applying config overlay %s", overlay_path)
    return _merge_overlay(yaml_data, overlay, Settings)


# Keys that must be set explicitly in config.yaml. Pydantic has a default for
# every field, which keeps bare ``Settings()`` construction working in tests
# but means a typo or missing key in the YAML silently falls back to a stale
# default. For tunable trading parameters that drive signal generation we
# require the key to exist so a misconfigured deploy fails loudly at startup
# rather than trading with an unintended value.
_REQUIRED_CONFIG_KEYS: tuple[tuple[str, ...], ...] = (("strategies", "weather", "edge_threshold"),)


def _assert_required_keys(yaml_data: dict[str, Any]) -> None:
    for path in _REQUIRED_CONFIG_KEYS:
        node: Any = yaml_data
        for key in path:
            if not isinstance(node, dict) or key not in node:
                joined = ".".join(path)
                raise ValueError(
                    f"config.yaml is missing required key: {joined}. "
                    "Pydantic defaults must not silently substitute for "
                    "tunable trading parameters."
                )
            node = node[key]


class Settings(BaseSettings):
    """
    Root settings object for The Oracle.

    Priority (highest wins):
      1. Environment variables (prefixed ORACLE_)
      2. .env file
      3. config overlay (``ORACLE_CONFIG_OVERLAY``)
      4. config.yaml
      5. Defaults defined here
    """

    # -- Kalshi credentials (always from env / .env) --
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = ""
    # Passphrase for an encrypted PEM (C-KALSHI-PEM). Empty = unencrypted PEM
    # (current behavior). Set ORACLE_KALSHI_PRIVATE_KEY_PASSWORD to opt in.
    kalshi_private_key_password: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_demo_base_url: str = "https://external-api.demo.kalshi.co/trade-api/v2"
    kalshi_demo_ws_url: str = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"

    # -- Telegram --
    telegram_bot_token: str = ""
    telegram_owner_chat_id: int = Field(default=0, ge=0)

    # -- Claude Code CLI (for /agent) --
    claude_code_bin: str = ""  # Full path to claude binary (bypasses PATH lookup)

    # -- Healthchecks.io --
    healthcheck_ping_url: str = ""

    # -- Bankroll --
    bankroll: float = Field(default=10_000.0, gt=0.0, lt=1e9)

    # -- Structured sub-configs (loaded from config.yaml) --
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    circuit_breakers: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    market_filter: MarketFilterConfig = Field(default_factory=MarketFilterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    calibration_monitor: CalibrationMonitorConfig = Field(default_factory=CalibrationMonitorConfig)
    fills_reconciler: FillsReconcilerConfig = Field(default_factory=FillsReconcilerConfig)
    kxrain_shadow: KxRainShadowConfig = Field(default_factory=KxRainShadowConfig)

    # -- Flat env overrides --
    execution_mode: Literal["shadow", "paper", "live"] | None = None

    # -- State directory --
    state_dir: str = "state"

    model_config = {
        "env_prefix": "ORACLE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # A blank ``KEY=`` copied from .env.example means "unset", not "".
        "env_ignore_empty": True,
        "extra": "ignore",
    }

    @classmethod
    def load(cls, config_path: Path | None = None) -> Settings:
        """
        Build Settings by merging config.yaml defaults, the optional
        ``ORACLE_CONFIG_OVERLAY`` file, and env overrides.
        """
        yaml_data = _load_yaml_config(config_path)
        _assert_required_keys(yaml_data)
        overlay_path = config_overlay_path()
        if overlay_path is not None:
            yaml_data = apply_config_overlay(yaml_data, overlay_path)
        return cls(**yaml_data)

    @model_validator(mode="after")
    def _propagate_execution_mode(self) -> Settings:
        if self.execution_mode is not None:
            # execution_mode is already validated as Literal["shadow","paper","live"]
            self.execution.mode = self.execution_mode
        return self

    @model_validator(mode="after")
    def _check_ws_idle_below_stale_cutoff(self) -> Settings:
        idle = self.execution.websocket_idle_timeout_seconds
        cutoff = self.risk.stale_data_cutoff_seconds
        if 2 * idle >= cutoff:
            raise ValueError(
                f"execution.websocket_idle_timeout_seconds ({idle}) must satisfy "
                f"2×idle < risk.stale_data_cutoff_seconds ({cutoff}); otherwise the "
                f"WS pipe-health discriminator (PIPE_HEALTH_WINDOW) collapses to "
                f"harm-only detection. Lower idle_timeout or raise stale_data_cutoff."
            )
        return self

    @model_validator(mode="after")
    def _check_edge_above_fee_floor(self) -> Settings:
        # Fee floor: taker fee can eat ~2% of mid; edge_threshold below that
        # is almost certainly a typo or a misunderstanding of the fee schedule.
        if self.strategies.weather.edge_threshold <= 0.02:
            raise ValueError(
                f"strategies.weather.edge_threshold="
                f"{self.strategies.weather.edge_threshold} is below the "
                f"fee floor (~0.02). Kalshi taker fees alone can consume "
                f"this much edge. Raise above 0.02."
            )
        return self

    @model_validator(mode="after")
    def _check_intraday_same_day_deconfliction(self) -> Settings:
        """Reject configs that double-execute on the same (city, var, today) row.

        When ``strategies.intraday_emos_high.enabled == True`` the intraday
        strategy claims same-day rows on its ``enabled_cells``. The production
        same-day gate (``weather.same_day_trading.allowed_cities``) must NOT
        include any of those cities, else ``weather_edge`` would re-enter on
        the same row and produce double execution. Defense-in-depth: the
        rule fires regardless of ``same_day_trading.enabled`` so an operator
        cannot flip the master switch later without re-triggering the
        misconfiguration.
        """
        intraday = self.strategies.intraday_emos_high
        if not intraday.enabled:
            return self
        allowed = set(self.strategies.weather.same_day_trading.allowed_cities or [])
        intraday_cities = {cell.split("/")[0] for cell in intraday.enabled_cells}
        conflict = allowed & intraday_cities
        if conflict:
            raise ValueError(
                f"Config conflict: strategies.intraday_emos_high.enabled=True with "
                f"enabled_cells={sorted(intraday.enabled_cells)}, but "
                f"strategies.weather.same_day_trading.allowed_cities contains "
                f"{sorted(conflict)}. This would cause double-execution on same-day "
                f"rows. Remove {sorted(conflict)} from "
                f"strategies.weather.same_day_trading.allowed_cities, or disable "
                f"strategies.intraday_emos_high."
            )
        return self

    @model_validator(mode="after")
    def _check_kelly_size_executable(self) -> Settings:
        # Ensure that a Kelly-sized trade at the single-market cap can
        # produce at least $1 of notional — otherwise no trade is ever
        # placed on a market (Kalshi min contract is $0.01, but $1 total
        # is the smallest sensible signal).
        notional = (
            self.risk.kelly_fraction
            * self.risk.max_single_market_pct
            * self.risk.max_total_exposure_pct
            * self.bankroll
        )
        if notional < 1.0:
            raise ValueError(
                f"risk.kelly_fraction({self.risk.kelly_fraction}) * "
                f"risk.max_single_market_pct({self.risk.max_single_market_pct}) * "
                f"risk.max_total_exposure_pct({self.risk.max_total_exposure_pct}) * "
                f"bankroll({self.bankroll}) = {notional:.4f} < $1.00; "
                f"no trade would ever clear the sizing floor. Raise one of "
                f"these four values."
            )
        return self

    @property
    def is_shadow(self) -> bool:
        return self.execution.mode == "shadow"

    @property
    def is_paper(self) -> bool:
        return self.execution.mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.execution.mode == "live"

    @property
    def active_base_url(self) -> str:
        """Return demo URL for paper mode, production URL otherwise."""
        if self.is_paper:
            return self.kalshi_demo_base_url
        return self.kalshi_base_url

    @property
    def active_ws_url(self) -> str:
        """Return demo WebSocket URL for paper mode, production URL otherwise.

        Demo and production credentials are not interchangeable, so REST and
        WebSocket must target the same environment.
        """
        if self.is_paper:
            return self.kalshi_demo_ws_url
        return self.kalshi_ws_url

    @property
    def stop_file(self) -> Path:
        return Path(self.state_dir) / "STOP"

    def is_kill_switch_active(self) -> bool:
        """Check if the STOP file exists. Fails closed on filesystem errors."""
        try:
            return self.stop_file.exists()
        except OSError:
            logging.getLogger(__name__).critical(
                "KILL SWITCH FAIL-CLOSED: cannot read %s — treating as active",
                self.stop_file,
                exc_info=True,
            )
            return True

    def activate_kill_switch(self) -> None:
        """Create the STOP file to halt all trading."""
        try:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("auto-activated")
        except OSError:
            logging.getLogger(__name__).critical(
                "Failed to create kill switch file %s", self.stop_file, exc_info=True
            )
