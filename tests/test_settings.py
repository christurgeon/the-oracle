"""Tests for Settings loading, required-key enforcement, and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from python.settings import (
    AttributionConfig,
    BracketMonitorConfig,
    CalibrationMonitorConfig,
    CircuitBreakerConfig,
    ExecutionConfig,
    MarketFilterConfig,
    RiskConfig,
    Settings,
    TelegramConfig,
    WeatherStrategyConfig,
)


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# ── Required-key enforcement (unchanged) ─────────────────────────────────


def test_load_fails_when_edge_threshold_missing(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"strategies": {"weather": {"enabled": True}}})
    with pytest.raises(ValueError, match="strategies.weather.edge_threshold"):
        Settings.load(config_path=config)


def test_load_fails_when_weather_block_missing(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"strategies": {}})
    with pytest.raises(ValueError, match="strategies.weather.edge_threshold"):
        Settings.load(config_path=config)


def test_load_fails_when_config_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ValueError, match="strategies.weather.edge_threshold"):
        Settings.load(config_path=missing)


def test_load_succeeds_with_edge_threshold_present(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path, {"strategies": {"weather": {"enabled": True, "edge_threshold": 0.07}}}
    )
    settings = Settings.load(config_path=config)
    assert settings.strategies.weather.edge_threshold == 0.07


def test_load_succeeds_with_real_config() -> None:
    """The checked-in config.yaml must pass the required-key check and schema."""
    settings = Settings.load()
    assert settings.strategies.weather.edge_threshold is not None


def test_blank_env_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .env copied from .env.example leaves typed fields blank; they must not fail to parse."""
    monkeypatch.setenv("ORACLE_TELEGRAM_OWNER_CHAT_ID", "")
    monkeypatch.setenv("ORACLE_EXECUTION_MODE", "")
    settings = Settings.load()
    assert settings.telegram_owner_chat_id == 0
    assert settings.execution_mode is None


def test_min_calibrated_prob_survives_pydantic_model_dump(tmp_path: Path) -> None:
    """Regression: WeatherStrategyConfig must include min_calibrated_prob so the
    field survives `s.strategies.weather.model_dump()` (the dict the orchestrator
    passes to WeatherEdgeStrategy). When the field is missing from the model,
    Pydantic silently drops it and the floor block is permanently disabled —
    exactly the bug that shipped to live before this test
    was added."""
    config = _write_config(
        tmp_path,
        {"strategies": {"weather": {"edge_threshold": 0.07, "min_calibrated_prob": 0.10}}},
    )
    settings = Settings.load(config_path=config)
    assert settings.strategies.weather.min_calibrated_prob == 0.10
    dumped = settings.strategies.weather.model_dump()
    assert dumped["min_calibrated_prob"] == 0.10


def test_min_calibrated_prob_real_config_value() -> None:
    """The checked-in config.yaml ships min_calibrated_prob: 0.110 (raised 2026-05-22
    from 0.10 under isotonic_v2 to restore FLOOR_DEFEAT_EPSILON safety margin) — pin
    that so a future config edit that drops it (or fat-fingers the value) is caught
    at test time, not at the next deploy."""
    settings = Settings.load()
    assert settings.strategies.weather.min_calibrated_prob == 0.110


def test_min_calibrated_prob_default_is_none() -> None:
    """Bare-construction default is None (filter disabled) for back-compat."""
    cfg = WeatherStrategyConfig()
    assert cfg.min_calibrated_prob is None


def test_min_calibrated_prob_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(min_calibrated_prob=1.5)
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(min_calibrated_prob=-0.1)


# ── FillsReconcilerConfig (spec §9) ──────────────────────────────────────


def test_fills_reconciler_config_defaults() -> None:
    s = Settings(_env_file=None)
    cfg = s.fills_reconciler
    assert cfg.enabled is False
    assert cfg.reconcile_interval_ticks == 5
    assert cfg.eventual_consistency_window_seconds == 120
    assert cfg.startup_safety_window_hours == 24
    assert cfg.startup_max_window_hours == 168
    assert cfg.max_pages_per_tick == 20
    assert cfg.max_fills_per_tick == 50
    assert cfg.page_delay_ms == 250
    assert cfg.placement_row_lookback == 10000
    assert cfg.alert_threshold_consecutive == 3
    assert cfg.alert_threshold_critical_delta == 3
    assert cfg.critical_alert_cooldown_seconds == 3600
    assert cfg.state_file == "state/fills_reconciler.json"


def test_fills_reconciler_config_manual_close_defaults() -> None:
    from python.settings import FillsReconcilerConfig
    cfg = FillsReconcilerConfig()
    assert cfg.manual_close_enabled is False
    assert cfg.manual_close_apply_limit_per_restart == 1
    assert cfg.shakedown_seen_oids_file == "state/shakedown_seen_oids.json"


def test_fills_reconciler_config_manual_close_override() -> None:
    from python.settings import FillsReconcilerConfig
    cfg = FillsReconcilerConfig(
        manual_close_enabled=True,
        manual_close_apply_limit_per_restart=0,
    )
    assert cfg.manual_close_enabled is True
    assert cfg.manual_close_apply_limit_per_restart == 0


def test_fills_reconciler_config_cash_out_defaults():
    from python.settings import FillsReconcilerConfig
    cfg = FillsReconcilerConfig()
    assert cfg.cash_out_apply_enabled is False
    assert cfg.cash_out_apply_limit_per_restart == 1
    assert cfg.cash_out_max_pnl_apply_dollars == 500.0
    assert cfg.shakedown_seen_oids_cash_out_file == "state/shakedown_seen_oids_cash_out.json"


def test_fills_reconciler_config_cash_out_validation():
    from python.settings import FillsReconcilerConfig
    # Cap must be in [10.0, 100000.0].
    with pytest.raises(Exception):
        FillsReconcilerConfig(cash_out_max_pnl_apply_dollars=1.0)
    with pytest.raises(Exception):
        FillsReconcilerConfig(cash_out_max_pnl_apply_dollars=200000.0)


# ── Bare construction with defaults (all sub-models + Settings) ───────────

_BARE_MODELS = [
    Settings,
    RiskConfig,
    WeatherStrategyConfig,
    MarketFilterConfig,
    CalibrationMonitorConfig,
    ExecutionConfig,
    BracketMonitorConfig,
    CircuitBreakerConfig,
    TelegramConfig,
    AttributionConfig,
]


@pytest.mark.parametrize("model", _BARE_MODELS)
def test_bare_construction_with_defaults_succeeds(model) -> None:
    if model is Settings:
        model(_env_file=None)
    else:
        model()


# ── Scale-confusion catch tests ──────────────────────────────────────────


def test_weather_edge_threshold_scale_confusion() -> None:
    with pytest.raises(ValidationError, match="Did you mean"):
        WeatherStrategyConfig(edge_threshold=7)


def test_max_price_scale_confusion() -> None:
    with pytest.raises(ValidationError, match="Did you mean"):
        MarketFilterConfig(max_price=97)


def test_min_price_scale_confusion() -> None:
    with pytest.raises(ValidationError, match="Did you mean"):
        MarketFilterConfig(min_price=5, max_price=0.97)


def test_kelly_fraction_scale_confusion() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(kelly_fraction=15)


def test_max_single_market_pct_scale_confusion() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(max_single_market_pct=5)


# ── Per-field range boundaries ───────────────────────────────────────────
# Each tuple: (model, field, low_valid, below_low_invalid, high_valid, above_high_invalid)
_RANGE_CASES = [
    # ExecutionConfig
    (ExecutionConfig, "poll_interval_seconds", 1, 0, 3600, 3601),
    (ExecutionConfig, "websocket_reconnect_delay", 0, -1, 300, 301),
    (ExecutionConfig, "order_ttl_seconds", 0, -1, 86400, 86401),
    (ExecutionConfig, "startup_delay_seconds", 0, -1, 3600, 3601),
    (ExecutionConfig, "reconcile_interval_ticks", 1, 0, 10000, 10001),
    (ExecutionConfig, "reconcile_bankroll_threshold", 0.0, -0.1, 1000.0, 1000.1),
    (ExecutionConfig, "reconcile_position_threshold", 0.0, -0.1, 100.0, 100.1),
    # BracketMonitorConfig (boost <= max_boost cross-field stays satisfied at defaults)
    (BracketMonitorConfig, "directional_boost_threshold", 0.0, -0.01, 0.5, 0.51),
    # CircuitBreakerConfig
    (CircuitBreakerConfig, "rolling_window_days", 1, 0, 90, 91),
    (CircuitBreakerConfig, "min_win_rate", 0.0, -0.01, 1.0, 1.01),
    (CircuitBreakerConfig, "min_trades_for_win_rate", 1, 0, 10000, 10001),
    (CircuitBreakerConfig, "data_staleness_multiplier", 1, 0, 20, 21),
    # TelegramConfig
    (TelegramConfig, "max_messages_per_minute", 1, 0, 30, 31),
    # MarketFilterConfig
    (MarketFilterConfig, "min_volume", 0, -1, 1_000_000, 1_000_001),
    (MarketFilterConfig, "min_open_interest", 0, -1, 1_000_000, 1_000_001),
    (MarketFilterConfig, "min_time_to_expiry_seconds", 0, -1, 30 * 86400, 30 * 86400 + 1),
    (MarketFilterConfig, "min_spread", 0.0, -0.01, 0.5, 0.51),
    # CalibrationMonitorConfig (picks values that keep warn<crit cross-field satisfied)
    (CalibrationMonitorConfig, "warmup_pairs", 0, -1, 10000, 10001),
    (CalibrationMonitorConfig, "bin_deviation_threshold", 0.0, -0.01, 1.0, 1.01),
    (CalibrationMonitorConfig, "bin_min_n", 1, 0, 10000, 10001),
    (CalibrationMonitorConfig, "city_var_brier_threshold", 0.0, -0.01, 1.0, 1.01),
    (CalibrationMonitorConfig, "city_var_min_n", 1, 0, 10000, 10001),
]


@pytest.mark.parametrize("model,field,low,below,high,above", _RANGE_CASES)
def test_range_boundaries(model, field, low, below, high, above) -> None:
    model(**{field: low})
    model(**{field: high})
    with pytest.raises(ValidationError):
        model(**{field: below})
    with pytest.raises(ValidationError):
        model(**{field: above})


# ── RiskConfig range tests (inline — each needs companion fields to keep
# cross-field invariants satisfied) ──────────────────────────────────────


def test_risk_per_field_ranges() -> None:
    # max_single_market_pct upper bound and zero
    RiskConfig(max_single_market_pct=0.25)
    with pytest.raises(ValidationError):
        RiskConfig(max_single_market_pct=0.26)
    with pytest.raises(ValidationError):
        RiskConfig(max_single_market_pct=0.0)
    # max_total_exposure_pct
    RiskConfig(max_total_exposure_pct=1.0)
    with pytest.raises(ValidationError):
        RiskConfig(max_total_exposure_pct=1.01)
    with pytest.raises(ValidationError):
        RiskConfig(max_total_exposure_pct=0.0)
    # kelly_fraction upper bound — need small max_single_market_pct to keep
    # kelly*market_cap under 0.10.
    RiskConfig(kelly_fraction=1.0, max_single_market_pct=0.01)
    with pytest.raises(ValidationError):
        RiskConfig(kelly_fraction=1.01)
    with pytest.raises(ValidationError):
        RiskConfig(kelly_fraction=0.0)
    # drawdown_size_reduction: open interval
    RiskConfig(drawdown_size_reduction=0.99)
    with pytest.raises(ValidationError):
        RiskConfig(drawdown_size_reduction=1.0)
    with pytest.raises(ValidationError):
        RiskConfig(drawdown_size_reduction=0.0)
    # max_correlated_positions
    RiskConfig(max_correlated_positions=1)
    RiskConfig(max_correlated_positions=100)
    with pytest.raises(ValidationError):
        RiskConfig(max_correlated_positions=0)
    with pytest.raises(ValidationError):
        RiskConfig(max_correlated_positions=101)
    # stale_data_cutoff_seconds
    RiskConfig(stale_data_cutoff_seconds=1)
    RiskConfig(stale_data_cutoff_seconds=3600)
    with pytest.raises(ValidationError):
        RiskConfig(stale_data_cutoff_seconds=0)
    with pytest.raises(ValidationError):
        RiskConfig(stale_data_cutoff_seconds=3601)


def test_weather_edge_threshold_range() -> None:
    WeatherStrategyConfig(edge_threshold=0.0)
    WeatherStrategyConfig(edge_threshold=0.5)
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(edge_threshold=-0.01)
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(edge_threshold=0.51)


def test_weather_cities_and_models_non_empty() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(cities=[])
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(ensemble_models=[])


def test_market_filter_price_bounds() -> None:
    MarketFilterConfig(min_price=0.01, max_price=0.99)
    with pytest.raises(ValidationError):
        MarketFilterConfig(min_price=0.0)
    with pytest.raises(ValidationError):
        MarketFilterConfig(max_price=1.0)


# ── Cross-field invariants ───────────────────────────────────────────────


def test_price_band_too_narrow_rejected() -> None:
    with pytest.raises(ValidationError, match="0.10 band"):
        MarketFilterConfig(min_price=0.95, max_price=0.97)
    # Boundary: exactly 0.10 wide is allowed.
    MarketFilterConfig(min_price=0.01, max_price=0.11)


def test_weather_edge_threshold_below_max_edge() -> None:
    with pytest.raises(ValidationError, match="strictly less than max_edge"):
        WeatherStrategyConfig(edge_threshold=0.10, max_edge=0.10)
    WeatherStrategyConfig(edge_threshold=0.07, max_edge=0.30)  # passes


def test_weather_ensemble_weights_checks() -> None:
    # Extra keys not in models
    with pytest.raises(ValidationError, match="not in ensemble_models"):
        WeatherStrategyConfig(
            ensemble_models=["gfs_seamless"],
            ensemble_weights={"gfs_seamless": 0.5, "nope": 0.5},
        )
    # Weights don't sum to ~1
    with pytest.raises(ValidationError, match="sum to"):
        WeatherStrategyConfig(
            ensemble_models=["gfs_seamless", "ecmwf_ifs025"],
            ensemble_weights={"gfs_seamless": 0.4, "ecmwf_ifs025": 0.3},
        )
    # Subset (omit a model — gets 0 weight) is allowed.
    WeatherStrategyConfig(
        ensemble_models=["gfs_seamless", "ecmwf_ifs025"],
        ensemble_weights={"gfs_seamless": 1.0},
    )


def test_bracket_boost_factor_above_max_boost_rejected() -> None:
    with pytest.raises(ValidationError, match="<= max_boost"):
        BracketMonitorConfig(directional_boost_factor=1.5, max_boost=1.2)


def test_risk_cross_field_invariants() -> None:
    with pytest.raises(ValidationError, match="strictly less than max_drawdown_pct"):
        RiskConfig(max_daily_loss_pct=0.15, max_drawdown_pct=0.15)
    # max_single_market_pct > max_total_exposure_pct is now allowed: the per-market
    # cap formula (max_pct * max_total * bankroll) guarantees budget safety regardless.
    RiskConfig(max_single_market_pct=0.20, max_total_exposure_pct=0.10)  # must not raise


def test_risk_kelly_product_bounds() -> None:
    # Too small: 0.01 * 0.02 = 0.0002 < 0.0005
    with pytest.raises(ValidationError, match=r"\[0.0005, 0.10\]"):
        RiskConfig(kelly_fraction=0.01, max_single_market_pct=0.02)
    # Too large: 0.5 * 0.25 = 0.125 > 0.10
    with pytest.raises(ValidationError, match=r"\[0.0005, 0.10\]"):
        RiskConfig(kelly_fraction=0.5, max_single_market_pct=0.25)


def test_calibration_brier_warn_below_crit() -> None:
    with pytest.raises(ValidationError, match="strictly less than brier_crit_mult"):
        CalibrationMonitorConfig(brier_warn_mult=1.6, brier_crit_mult=1.6)


# ── Settings root cross-field ─────────────────────────────────────────────


def test_settings_rejects_edge_threshold_below_fee_floor(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"strategies": {"weather": {"edge_threshold": 0.02}}})
    with pytest.raises(ValidationError, match="fee floor"):
        Settings.load(config_path=config)
    config = _write_config(tmp_path, {"strategies": {"weather": {"edge_threshold": 0.03}}})
    Settings.load(config_path=config)  # 0.03 passes


def test_settings_rejects_non_executable_kelly_size() -> None:
    # kelly(0.15) * market_cap(0.05) * bankroll(10) = 0.075 < $1
    with pytest.raises(ValidationError, match="no trade would ever clear"):
        Settings(bankroll=10.0, _env_file=None)


def test_settings_execution_mode_literal() -> None:
    for mode in ("shadow", "paper", "live"):
        assert Settings(execution_mode=mode, _env_file=None).execution.mode == mode
    with pytest.raises(ValidationError):
        Settings(execution_mode="invalid", _env_file=None)


def test_bankroll_range() -> None:
    with pytest.raises(ValidationError):
        Settings(bankroll=0.0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(bankroll=1e9, _env_file=None)


def test_attribution_settlements_path_non_empty() -> None:
    with pytest.raises(ValidationError):
        AttributionConfig(settlements_path="")


def test_execution_emit_placement_rows_default_false() -> None:
    """The flag defaults to False so Phase 1 deploy is a no-op until
    the operator explicitly flips it."""
    s = Settings(_env_file=None)
    assert s.execution.emit_placement_rows is False


def test_execution_emit_placement_rows_can_be_enabled() -> None:
    cfg = ExecutionConfig(emit_placement_rows=True)
    assert cfg.emit_placement_rows is True


def test_same_day_trading_settings_default_off() -> None:
    from python.settings import StrategiesConfig

    s = StrategiesConfig()
    assert s.weather.same_day_trading.enabled is False
    assert s.weather.same_day_trading.allowed_cities == []
    # Defaults match the spec.
    assert s.weather.same_day_trading.cache_ttl_seconds == 600
    assert s.weather.same_day_trading.near_peak_ttl_seconds == 90
    assert s.weather.same_day_trading.near_peak_window_minutes == 120
    assert s.weather.same_day_trading.peak_cutoff_minutes == 60
    assert s.weather.same_day_trading.nws_request_timeout_seconds == 4
    assert s.weather.same_day_trading.nws_gate_timeout_seconds == 20


# ── SameDayTradingSettings.allowed_cities field validator ─────────────────
# Mirrors the canonical city set in python/data/weather.py:CITY_COORDS.
# Lowercase / whitespace / unknown entries silently miss downstream
# exact-match comparisons (same_day_gate.py, weather_edge.py, and the
# Settings cross-config deconfliction validator). The field validator
# rejects malformed entries at the source so all consumers benefit.


def test_allowed_cities_valid_uppercase_passes() -> None:
    from python.settings import SameDayTradingSettings

    cfg = SameDayTradingSettings(allowed_cities=["MIA", "LAX"])
    assert cfg.allowed_cities == ["MIA", "LAX"]


def test_allowed_cities_lowercase_raises_with_valid_codes_listed() -> None:
    from python.settings import SameDayTradingSettings

    with pytest.raises(ValidationError) as excinfo:
        SameDayTradingSettings(allowed_cities=["mia"])
    msg = str(excinfo.value)
    # Error must name the offender verbatim …
    assert "'mia'" in msg or '"mia"' in msg
    # … and list at least one valid code so the operator sees the canonical set.
    assert "MIA" in msg
    assert "LAX" in msg


def test_allowed_cities_whitespace_stripped_but_case_still_rejected() -> None:
    from python.settings import SameDayTradingSettings

    # `" mia "` → strip → `"mia"` → still wrong case → reject.
    with pytest.raises(ValidationError) as excinfo:
        SameDayTradingSettings(allowed_cities=[" mia "])
    msg = str(excinfo.value)
    # Stripped form should appear in the error (proves whitespace was stripped).
    assert "'mia'" in msg or '"mia"' in msg


def test_allowed_cities_unknown_city_raises() -> None:
    from python.settings import SameDayTradingSettings

    with pytest.raises(ValidationError) as excinfo:
        SameDayTradingSettings(allowed_cities=["XYZ"])
    msg = str(excinfo.value)
    assert "'XYZ'" in msg or '"XYZ"' in msg


def test_allowed_cities_dedupes_silently() -> None:
    from python.settings import SameDayTradingSettings

    cfg = SameDayTradingSettings(allowed_cities=["MIA", "MIA"])
    assert cfg.allowed_cities == ["MIA"]


def test_allowed_cities_whitespace_stripped_when_valid() -> None:
    from python.settings import SameDayTradingSettings

    # `" MIA "` → strip → `"MIA"` → valid; the stripped value is kept.
    cfg = SameDayTradingSettings(allowed_cities=[" MIA "])
    assert cfg.allowed_cities == ["MIA"]


# ── 2026-05-11 LAX EMOS promotion → 2026-05-18 LAX/high rollback ─────
#
# The original `test_emos_enabled_cells_contains_lax_after_2026_05_11_override`
# tripwire fired exactly as designed when LAX/high was rolled back 2026-05-18
# (live ROI −39.6% on n≈11 trades; σ_emos = 6.8°F incompatible with 1°F
# brackets per the σ-audit). Its documented response to a rollback was
# "remove the override entry + update memory" — both done. Tripwire
# removed 2026-05-19 to clear the stale CI failure. The companion
# artifact-load check below remains.


def test_emos_enabled_cells_null_coerces_to_empty_list() -> None:
    """``emos_enabled_cells: null`` in YAML is operator shorthand for ``[]``.

    Without the ``mode='before'`` coercion the after-validator raises
    ``list_type`` and the bot fails to start — a startup footgun for any
    operator who comments out the cells to disable EMOS.
    """
    cfg = WeatherStrategyConfig(emos_enabled_cells=None)  # type: ignore[arg-type]
    assert cfg.emos_enabled_cells == []


def test_lax_cells_present_in_production_emos_artifact() -> None:
    """LAX/high + LAX/low must be in coefficients.json for the 2026-05-11
    promotion to take effect. If this fails, roll back the config change
    until the next refit fills the cells.

    Paired with the settings assertion above (same override commit).
    """
    from python.data.emos import EmosTable

    coeff_path = Path(__file__).resolve().parents[1] / "data/emos/coefficients.json"
    table = EmosTable.load(coeff_path)
    assert table.is_loaded, table.fallback_reason
    assert table.get(city="LAX", var="high", lead=0) is not None
    assert table.get(city="LAX", var="low", lead=0) is not None


class TestWsIdleBelowStaleCutoff:
    """2*websocket_idle_timeout_seconds < risk.stale_data_cutoff_seconds."""

    def test_invalid_when_double_idle_reaches_cutoff(self, make_settings):
        # 2*160 = 320 >= 300 -> invalid
        with pytest.raises(ValidationError, match="websocket_idle_timeout_seconds"):
            make_settings(execution={"websocket_idle_timeout_seconds": 160})

    def test_default_is_valid(self, make_settings):
        s = make_settings()  # default idle 90; 2*90=180 < 300
        assert (
            s.execution.websocket_idle_timeout_seconds * 2
            < s.risk.stale_data_cutoff_seconds
        )
