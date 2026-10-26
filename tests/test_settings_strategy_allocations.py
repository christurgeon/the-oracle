"""Tests for risk.strategy_allocations config validation."""

from __future__ import annotations

import pytest

from python.orchestrator import Orchestrator, validate_strategy_allocations
from python.settings import RiskConfig, Settings


def test_shipped_config_allocations_match_registered_strategies() -> None:
    """The shipped config.yaml must not leave an allocation entry for a parked
    (enabled:false) strategy. Disabled strategies are not registered at runtime,
    and validate_strategy_allocations (run in orchestrator.initialize) rejects
    allocation keys for unregistered strategies — crashing startup.

    Regression for the 2026-06-07 incident: intraday_emos_high was parked
    (enabled:false) but its `intraday_emos_high: 0.0` allocation entry remained,
    so the first restart raised ValueError and the bot crash-looped.
    """
    s = Settings.load()
    st = s.strategies
    # runtime name -> config enabled flag (mirrors orchestrator strategy registration)
    runtime_enabled = {
        "weather_edge": st.weather.enabled,
        "favorite_bias": st.favorite_bias.enabled,
        "intraday_emos_high": st.intraday_emos_high.enabled,
        "intraday_asos_high": st.intraday_asos_high.enabled,
    }
    # Only enabled strategies register; that is the set validate_* sees at startup.
    registered = [(name, True) for name, enabled in runtime_enabled.items() if enabled]
    # Must not raise — this is exactly the startup check in initialize().
    validate_strategy_allocations(s.risk.strategy_allocations, registered)


def test_startup_without_kalshi_credentials_skips_the_allocation_check() -> None:
    """No credentials means no strategy is built, so the shipped allocations must not crash startup."""
    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = Settings.load()
    orch._strategies = []
    orch._kalshi_client = None
    orch._validate_strategy_allocations()

    orch._kalshi_client = object()
    with pytest.raises(ValueError, match="unknown strategy"):
        orch._validate_strategy_allocations()


class TestStrategyAllocationsValidation:
    def test_default_is_empty_dict(self):
        assert RiskConfig().strategy_allocations == {}

    def test_valid_allocations_accepted(self):
        cfg = RiskConfig(strategy_allocations={"weather_edge": 0.15, "favorite_bias": 0.15})
        assert cfg.strategy_allocations["weather_edge"] == 0.15

    def test_zero_allocation_allowed(self):
        cfg = RiskConfig(strategy_allocations={"intraday_emos_high": 0.0})
        assert cfg.strategy_allocations["intraday_emos_high"] == 0.0

    def test_sum_over_one_rejected(self):
        with pytest.raises(ValueError, match="sum"):
            RiskConfig(strategy_allocations={"a": 0.6, "b": 0.6})

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            RiskConfig(strategy_allocations={"a": -0.1})

    def test_value_over_one_rejected(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            RiskConfig(strategy_allocations={"a": 1.5})

    def test_sum_over_firm_cap_warns_not_errors(self, caplog):
        # max_total_exposure_pct default 0.40; sum 0.50 > 0.40 -> warn, not raise
        import logging

        with caplog.at_level(logging.WARNING):
            cfg = RiskConfig(strategy_allocations={"a": 0.25, "b": 0.25})
        assert cfg.strategy_allocations == {"a": 0.25, "b": 0.25}
        assert any("max_total_exposure_pct" in r.message for r in caplog.records)
