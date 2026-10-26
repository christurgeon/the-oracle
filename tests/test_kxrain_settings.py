"""Tests for ``KxRainShadowConfig``."""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from python.data.kxrain_markets import KXRAIN_CITY_STATION
from python.data.kxrain_observer import KxRainObserverConfig
from python.data.kxrain_settlements import SETTLEMENT_PAGE_LIMIT, SETTLEMENT_SWEEP_MAX_PAGES
from python.settings import KxRainShadowConfig, Settings

# Defaults


def test_defaults() -> None:
    cfg = KxRainShadowConfig()
    assert cfg.enabled is False
    assert cfg.series_ticker == "KXRAIN"
    assert cfg.tick_interval_seconds == pytest.approx(600)
    assert cfg.entry_window_start_lst == "10:00"
    assert cfg.entry_window_end_lst == "18:00"
    assert cfg.yes_bid_min == pytest.approx(0.10)
    assert cfg.yes_bid_max == pytest.approx(0.90)
    assert cfg.no_trade_lookback_minutes == 30
    assert cfg.max_entries_per_day is None
    assert cfg.assumed_max_walk_cents == 1
    assert cfg.depth_levels == 10
    assert cfg.depth_capture_mode == "band"
    assert cfg.max_requests_per_tick == 60
    assert cfg.request_pacing_seconds == pytest.approx(1.0)
    assert cfg.tick_wall_clock_budget_seconds == pytest.approx(180)
    assert cfg.post_entry_follow_minutes == 45
    assert cfg.min_free_disk_bytes == 2_147_483_648
    assert cfg.log_dir == "state"
    assert cfg.log_retention_days == 240
    assert cfg.trades_page_limit == 100
    assert cfg.settlement_capture_hour_utc == 14
    assert cfg.settlement_sweep_max_pages == SETTLEMENT_SWEEP_MAX_PAGES


def test_deliberately_omitted_fields_are_not_present() -> None:
    # These streams aren't implemented; a config key would look enabled but do nothing.
    cfg = KxRainShadowConfig()
    dumped = cfg.model_dump()
    assert "iem_cache_ttl_seconds" not in dumped
    assert "gauge_capture_enabled" not in dumped
    assert "pop_capture_enabled" not in dumped


# Bounds: each rejects an out-of-range value


def test_tick_interval_seconds_bounds() -> None:
    KxRainShadowConfig(tick_interval_seconds=60)
    KxRainShadowConfig(tick_interval_seconds=3600)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(tick_interval_seconds=59)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(tick_interval_seconds=3601)


def test_yes_bid_min_bounds() -> None:
    KxRainShadowConfig(yes_bid_min=0.0, yes_bid_max=0.90)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(yes_bid_min=-0.01)


def test_yes_bid_max_bounds() -> None:
    KxRainShadowConfig(yes_bid_max=1.0, yes_bid_min=0.10)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(yes_bid_max=1.01)


def test_no_trade_lookback_minutes_bounds() -> None:
    KxRainShadowConfig(no_trade_lookback_minutes=1)
    KxRainShadowConfig(no_trade_lookback_minutes=240)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(no_trade_lookback_minutes=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(no_trade_lookback_minutes=241)


def test_max_entries_per_day_bounds() -> None:
    KxRainShadowConfig(max_entries_per_day=None)
    KxRainShadowConfig(max_entries_per_day=1)
    KxRainShadowConfig(max_entries_per_day=50)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(max_entries_per_day=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(max_entries_per_day=51)


def test_assumed_max_walk_cents_bounds() -> None:
    KxRainShadowConfig(assumed_max_walk_cents=0)
    KxRainShadowConfig(assumed_max_walk_cents=10)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(assumed_max_walk_cents=-1)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(assumed_max_walk_cents=11)


def test_depth_levels_bounds() -> None:
    KxRainShadowConfig(depth_levels=1)
    KxRainShadowConfig(depth_levels=50)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(depth_levels=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(depth_levels=51)


def test_depth_capture_mode_literal() -> None:
    KxRainShadowConfig(depth_capture_mode="band")
    KxRainShadowConfig(depth_capture_mode="all")
    with pytest.raises(ValidationError):
        KxRainShadowConfig(depth_capture_mode="everything")  # type: ignore[arg-type]


def test_max_requests_per_tick_bounds() -> None:
    KxRainShadowConfig(max_requests_per_tick=1)
    KxRainShadowConfig(max_requests_per_tick=200)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(max_requests_per_tick=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(max_requests_per_tick=201)


def test_request_pacing_seconds_bounds() -> None:
    KxRainShadowConfig(request_pacing_seconds=0.0)
    KxRainShadowConfig(request_pacing_seconds=5.0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(request_pacing_seconds=-0.01)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(request_pacing_seconds=5.01)


def test_tick_wall_clock_budget_seconds_bounds() -> None:
    KxRainShadowConfig(tick_wall_clock_budget_seconds=10)
    KxRainShadowConfig(tick_wall_clock_budget_seconds=600)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(tick_wall_clock_budget_seconds=9)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(tick_wall_clock_budget_seconds=601)


def test_post_entry_follow_minutes_bounds() -> None:
    KxRainShadowConfig(post_entry_follow_minutes=0)
    KxRainShadowConfig(post_entry_follow_minutes=240)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(post_entry_follow_minutes=-1)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(post_entry_follow_minutes=241)


def test_min_free_disk_bytes_bounds() -> None:
    KxRainShadowConfig(min_free_disk_bytes=100_000_000)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(min_free_disk_bytes=99_999_999)


def test_min_free_disk_bytes_zero_is_rejected() -> None:
    # 0 would silently disable the guard, since free_bytes >= 0 is always true.
    with pytest.raises(ValidationError):
        KxRainShadowConfig(min_free_disk_bytes=0)


def test_log_retention_days_bounds() -> None:
    KxRainShadowConfig(log_retention_days=7)
    KxRainShadowConfig(log_retention_days=730)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(log_retention_days=6)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(log_retention_days=731)


def test_trades_page_limit_bounds() -> None:
    KxRainShadowConfig(trades_page_limit=1)
    KxRainShadowConfig(trades_page_limit=1000)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(trades_page_limit=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(trades_page_limit=1001)


def test_settlement_capture_hour_utc_bounds() -> None:
    KxRainShadowConfig(settlement_capture_hour_utc=0)
    KxRainShadowConfig(settlement_capture_hour_utc=23)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(settlement_capture_hour_utc=-1)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(settlement_capture_hour_utc=24)


def test_settlement_sweep_max_pages_bounds() -> None:
    KxRainShadowConfig(settlement_sweep_max_pages=1)
    KxRainShadowConfig(settlement_sweep_max_pages=100)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(settlement_sweep_max_pages=0)
    with pytest.raises(ValidationError):
        KxRainShadowConfig(settlement_sweep_max_pages=101)


# yes_bid_min < yes_bid_max


def test_yes_bid_min_equal_max_rejected() -> None:
    with pytest.raises(ValidationError, match="yes_bid_min"):
        KxRainShadowConfig(yes_bid_min=0.5, yes_bid_max=0.5)


def test_yes_bid_min_greater_than_max_rejected() -> None:
    with pytest.raises(ValidationError, match="yes_bid_min"):
        KxRainShadowConfig(yes_bid_min=0.90, yes_bid_max=0.10)


def test_yes_bid_min_less_than_max_accepted() -> None:
    cfg = KxRainShadowConfig(yes_bid_min=0.10, yes_bid_max=0.90)
    assert cfg.yes_bid_min < cfg.yes_bid_max


# entry_window_* format validation


@pytest.mark.parametrize(
    "bad_value",
    ["10", "10:0", "25:00", "10:60", "10-00", "", "noon", "10:00:00", " 10:00"],
)
def test_bad_entry_window_start_rejected(bad_value: str) -> None:
    with pytest.raises(ValidationError, match="entry_window_start_lst"):
        KxRainShadowConfig(entry_window_start_lst=bad_value)


@pytest.mark.parametrize(
    "bad_value",
    ["18", "18:0", "24:00", "18:60", "18-00", "", "6pm"],
)
def test_bad_entry_window_end_rejected(bad_value: str) -> None:
    with pytest.raises(ValidationError, match="entry_window_end_lst"):
        KxRainShadowConfig(entry_window_end_lst=bad_value)


def test_good_entry_window_accepted() -> None:
    cfg = KxRainShadowConfig(entry_window_start_lst="09:30", entry_window_end_lst="19:45")
    assert cfg.entry_window_start_lst == "09:30"
    assert cfg.entry_window_end_lst == "19:45"


# to_observer_config() round-trip


def test_to_observer_config_returns_observer_config_instance() -> None:
    cfg = KxRainShadowConfig()
    observer_cfg = cfg.to_observer_config()
    assert isinstance(observer_cfg, KxRainObserverConfig)


def test_to_observer_config_round_trips_every_field() -> None:
    cfg = KxRainShadowConfig(
        series_ticker="KXRAIN",
        tick_interval_seconds=900,
        entry_window_start_lst="09:15",
        entry_window_end_lst="17:45",
        yes_bid_min=0.15,
        yes_bid_max=0.85,
        no_trade_lookback_minutes=20,
        max_entries_per_day=5,
        assumed_max_walk_cents=2,
        depth_levels=8,
        depth_capture_mode="all",
        max_requests_per_tick=40,
        request_pacing_seconds=0.5,
        tick_wall_clock_budget_seconds=120,
        post_entry_follow_minutes=30,
        min_free_disk_bytes=500_000_000,
        log_dir="state/kxrain_test",
        log_retention_days=90,
        trades_page_limit=50,
    )
    observer_cfg = cfg.to_observer_config()

    assert observer_cfg.log_dir == Path("state/kxrain_test")
    assert isinstance(observer_cfg.log_dir, Path)
    assert observer_cfg.series_ticker == "KXRAIN"
    assert observer_cfg.tick_interval_seconds == pytest.approx(900)
    assert observer_cfg.entry_window_start_lst == time(9, 15)
    assert isinstance(observer_cfg.entry_window_start_lst, time)
    assert observer_cfg.entry_window_end_lst == time(17, 45)
    assert isinstance(observer_cfg.entry_window_end_lst, time)
    assert observer_cfg.yes_bid_min == pytest.approx(0.15)
    assert observer_cfg.yes_bid_max == pytest.approx(0.85)
    assert observer_cfg.no_trade_lookback_minutes == 20
    assert observer_cfg.max_entries_per_day == 5
    assert observer_cfg.assumed_max_walk_cents == 2
    assert observer_cfg.depth_levels == 8
    assert observer_cfg.depth_capture_mode == "all"
    assert observer_cfg.max_requests_per_tick == 40
    assert observer_cfg.request_pacing_seconds == pytest.approx(0.5)
    assert observer_cfg.tick_wall_clock_budget_seconds == pytest.approx(120)
    assert observer_cfg.post_entry_follow_minutes == 30
    assert observer_cfg.min_free_disk_bytes == 500_000_000
    assert observer_cfg.log_retention_days == 90
    assert observer_cfg.trades_page_limit == 50


def test_to_observer_config_defaults_round_trip() -> None:
    observer_cfg = KxRainShadowConfig().to_observer_config()
    default_observer_cfg = KxRainObserverConfig(log_dir=Path("state"))

    assert observer_cfg.series_ticker == default_observer_cfg.series_ticker
    assert observer_cfg.tick_interval_seconds == default_observer_cfg.tick_interval_seconds
    assert observer_cfg.entry_window_start_lst == default_observer_cfg.entry_window_start_lst
    assert observer_cfg.entry_window_end_lst == default_observer_cfg.entry_window_end_lst
    assert observer_cfg.yes_bid_min == default_observer_cfg.yes_bid_min
    assert observer_cfg.yes_bid_max == default_observer_cfg.yes_bid_max
    assert observer_cfg.min_free_disk_bytes == default_observer_cfg.min_free_disk_bytes
    assert observer_cfg.log_dir == default_observer_cfg.log_dir


# config.yaml parses via the normal Settings load path


def test_shipped_config_yaml_parses_into_kxrain_shadow() -> None:
    settings = Settings.load()
    assert isinstance(settings.kxrain_shadow, KxRainShadowConfig)
    assert settings.kxrain_shadow.series_ticker == "KXRAIN"
    assert settings.kxrain_shadow.log_retention_days == 240


def test_shipped_yaml_sweep_page_cap_matches_the_module_constant() -> None:
    settings = Settings.load()
    assert settings.kxrain_shadow.settlement_sweep_max_pages == SETTLEMENT_SWEEP_MAX_PAGES


def test_the_daily_sweep_reaches_back_at_least_25_days() -> None:
    # Reach is the lapse the corpus survives with no operator action; a sweep outage
    # raises no alert until the listing has already scrolled past the missing days.
    markets_per_day = len(KXRAIN_CITY_STATION)
    reach_days = SETTLEMENT_SWEEP_MAX_PAGES * SETTLEMENT_PAGE_LIMIT / markets_per_day
    assert reach_days >= 25


def test_enabled_defaults_false_in_model() -> None:
    assert KxRainShadowConfig().enabled is False


def test_shipped_yaml_ships_the_observer_off() -> None:
    settings = Settings.load()
    assert settings.kxrain_shadow.enabled is False
    assert settings.kxrain_shadow.tick_interval_seconds in (600, 1800)


# kxrain_shadow must never appear in risk.strategy_allocations


def test_kxrain_shadow_not_in_risk_strategy_allocations() -> None:
    settings = Settings.load()
    assert "kxrain_shadow" not in settings.risk.strategy_allocations


# The shipped YAML is the pre-registration — pin it literally


def test_the_shipped_yaml_pins_the_pre_registered_rule_parameters() -> None:
    # Every value equals a model default, so this is what proves the YAML is read at all.
    import yaml

    raw = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "python" / "config.yaml").read_text()
    )
    assert "kxrain_shadow" in raw, "the block must stay TOP-LEVEL, never under strategies:"
    block = raw["kxrain_shadow"]

    assert block["yes_bid_min"] == 0.10
    assert block["yes_bid_max"] == 0.90
    assert block["entry_window_start_lst"] == "10:00"
    assert block["entry_window_end_lst"] == "18:00"
    assert block["no_trade_lookback_minutes"] == 30
    assert block["max_entries_per_day"] is None
    assert block["series_ticker"] == "KXRAIN"


def test_every_observer_field_is_wired_from_config() -> None:
    # to_observer_config() enumerates fields by hand; pinning the set forces a decision.
    import dataclasses

    from python.data.kxrain_observer import KxRainObserverConfig

    assert {f.name for f in dataclasses.fields(KxRainObserverConfig)} == {
        "log_dir",
        "series_ticker",
        "tick_interval_seconds",
        "entry_window_start_lst",
        "entry_window_end_lst",
        "yes_bid_min",
        "yes_bid_max",
        "no_trade_lookback_minutes",
        "max_entries_per_day",
        "assumed_max_walk_cents",
        "depth_levels",
        "depth_capture_mode",
        "max_requests_per_tick",
        "request_pacing_seconds",
        "tick_wall_clock_budget_seconds",
        "post_entry_follow_minutes",
        "min_free_disk_bytes",
        "log_retention_days",
        "trades_page_limit",
    }
