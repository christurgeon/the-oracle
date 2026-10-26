"""Tests for per-strategy virtual bankrolls (pod) on the RiskManager."""

from __future__ import annotations

import pytest

from python.risk.manager import RiskManager
from python.settings import RiskConfig


class TestBookFor:
    def test_empty_map_returns_full_equity(self, default_settings):
        rm = RiskManager(default_settings)
        assert rm.book_for("weather_edge") == rm.bankroll

    def test_allocated_strategy_returns_slice(self, make_settings):
        s = make_settings(risk=RiskConfig(strategy_allocations={"weather_edge": 0.15}))
        rm = RiskManager(s)  # bankroll defaults to 10_000
        assert rm.book_for("weather_edge") == pytest.approx(0.15 * 10_000)

    def test_unlisted_strategy_falls_back_to_full_equity(self, make_settings):
        s = make_settings(risk=RiskConfig(strategy_allocations={"weather_edge": 0.15}))
        rm = RiskManager(s)
        assert rm.book_for("favorite_bias") == rm.bankroll

    def test_zero_allocation_returns_zero(self, make_settings):
        s = make_settings(risk=RiskConfig(strategy_allocations={"intraday_emos_high": 0.0}))
        rm = RiskManager(s)
        assert rm.book_for("intraday_emos_high") == 0.0


class TestStrategyExposureAccumulator:
    def test_record_fill_attributes_and_accumulates(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.record_fill("M1", "weather_edge", 50.0)  # repeat fill, delta
        assert risk_manager.strategy_exposure()["weather_edge"] == pytest.approx(150.0)
        assert risk_manager.strategy_for_position("M1") == "weather_edge"

    def test_close_position_decrements_correct_pod(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.record_fill("M2", "favorite_bias", 40.0)
        risk_manager.close_position("M1", pnl=10.0)
        exp = risk_manager.strategy_exposure()
        assert exp.get("weather_edge", 0.0) == pytest.approx(0.0)
        assert exp["favorite_bias"] == pytest.approx(40.0)

    def test_release_exposure_clamps_and_decrements(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.release_exposure("M1", 30.0)
        assert risk_manager.strategy_exposure()["weather_edge"] == pytest.approx(70.0)

    def test_realize_partial_decrements(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.realize_partial("M1", dollars_released=40.0, pnl=5.0)
        assert risk_manager.strategy_exposure()["weather_edge"] == pytest.approx(60.0)

    def test_decrement_missing_market_no_keyerror_unknown_bucket(self, risk_manager):
        risk_manager._positions["MX"] = 25.0  # position with no attribution
        risk_manager.close_position("MX", pnl=0.0)
        assert risk_manager.strategy_exposure().get("unknown", 0.0) >= 0.0

    def test_invariant_sum_equals_total_exposure(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.record_fill("M2", "favorite_bias", 40.0)
        risk_manager.release_exposure("M1", 30.0)
        total = sum(risk_manager.strategy_exposure().values())
        assert total == pytest.approx(risk_manager.total_exposure)

    def test_accumulator_never_negative_on_over_release(self, risk_manager):
        risk_manager.record_fill("M1", "weather_edge", 50.0)
        risk_manager.release_exposure("M1", 80.0)  # over-release
        assert risk_manager.strategy_exposure()["weather_edge"] >= 0.0

    def test_owner_reassignment_preserves_invariant(self, risk_manager):
        # Same market re-filled under a different strategy: ownership moves and
        # the prior exposure must NOT remain stuck in the old pod's bucket.
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        risk_manager.record_fill("M1", "favorite_bias", 50.0)
        exp = risk_manager.strategy_exposure()
        assert exp.get("weather_edge", 0.0) == pytest.approx(0.0)
        assert exp["favorite_bias"] == pytest.approx(150.0)
        assert sum(exp.values()) == pytest.approx(risk_manager.total_exposure)
        risk_manager.close_position("M1", pnl=0.0)
        assert sum(risk_manager.strategy_exposure().values()) == pytest.approx(
            risk_manager.total_exposure
        )


class TestHydrateAttribution:
    def test_hydrate_reconstructs_strategy_exposure(self, risk_manager):
        positions = {"M1": 100.0, "M2": 40.0, "M3": 10.0}
        position_strategy = {"M1": "weather_edge", "M2": "favorite_bias"}  # M3 unattributed
        risk_manager.hydrate_positions(positions, 10_000.0, position_strategy=position_strategy)
        exp = risk_manager.strategy_exposure()
        assert exp["weather_edge"] == pytest.approx(100.0)
        assert exp["favorite_bias"] == pytest.approx(40.0)
        assert exp["unknown"] == pytest.approx(10.0)
        assert sum(exp.values()) == pytest.approx(risk_manager.total_exposure)

    def test_hydrate_without_attribution_buckets_all_unknown(self, risk_manager):
        risk_manager.hydrate_positions({"M1": 100.0}, 10_000.0)
        assert risk_manager.strategy_exposure()["unknown"] == pytest.approx(100.0)


class TestObserveOnlyLogging:
    def test_logs_on_change_only(self, make_settings, caplog):
        import logging

        s = make_settings(risk=RiskConfig(strategy_allocations={"weather_edge": 0.15}))
        rm = RiskManager(s)
        rm.record_fill("M1", "weather_edge", 100.0)
        with caplog.at_level(logging.INFO):
            rm.maybe_log_pod_exposure()
            first = len(caplog.records)
            rm.maybe_log_pod_exposure()  # no change -> no new log
            second = len(caplog.records)
        assert first >= 1
        assert second == first

    def test_logs_again_after_change(self, make_settings, caplog):
        import logging

        s = make_settings(risk=RiskConfig(strategy_allocations={"weather_edge": 0.15}))
        rm = RiskManager(s)
        with caplog.at_level(logging.INFO):
            rm.maybe_log_pod_exposure()
            rm.record_fill("M1", "weather_edge", 100.0)
            before = len(caplog.records)
            rm.maybe_log_pod_exposure()
            after = len(caplog.records)
        assert after > before

    def test_empty_allocations_no_log(self, default_settings, caplog):
        import logging

        rm = RiskManager(default_settings)
        rm.record_fill("M1", "weather_edge", 100.0)
        with caplog.at_level(logging.INFO):
            rm.maybe_log_pod_exposure()
        assert not any("pod_exposure" in r.message for r in caplog.records)

    def test_logs_unknown_bucket(self, make_settings, caplog):
        import logging

        s = make_settings(risk=RiskConfig(strategy_allocations={"weather_edge": 0.15}))
        rm = RiskManager(s)
        rm._positions["MX"] = 25.0
        rm._strategy_exposure["unknown"] = 25.0  # simulate unattributed exposure
        with caplog.at_level(logging.INFO):
            rm.maybe_log_pod_exposure()
        assert any("strategy=unknown" in r.message for r in caplog.records)


class TestAttributionSplitGuard:
    def test_strategy_for_position_is_authoritative(self, risk_manager):
        # _position_strategy holds the real strategy; reconciler may resolve
        # "unknown" via its own order_id path. The manager accessor is the
        # authority used by the WARNING guard.
        risk_manager.record_fill("M1", "weather_edge", 100.0)
        assert risk_manager.strategy_for_position("M1") == "weather_edge"
        assert risk_manager.strategy_for_position("M-absent") == "unknown"


class TestHydrateBoundsAttributionToOpenPositions:
    def test_stale_closed_market_attribution_dropped(self, risk_manager):
        # The reconstructed index spans all historically-traded markets; hydrate
        # must keep attribution ONLY for markets that are currently open.
        positions = {"OPEN1": 100.0}
        position_strategy = {
            "OPEN1": "weather_edge",
            "CLOSED_LONG_AGO": "favorite_bias",  # not in positions
        }
        risk_manager.hydrate_positions(
            positions, 10_000.0, position_strategy=position_strategy
        )
        assert risk_manager.strategy_for_position("OPEN1") == "weather_edge"
        # Stale closed-market entry must not be retained.
        assert risk_manager.strategy_for_position("CLOSED_LONG_AGO") == "unknown"
        assert "favorite_bias" not in risk_manager.strategy_exposure()
        assert sum(risk_manager.strategy_exposure().values()) == pytest.approx(
            risk_manager.total_exposure
        )
