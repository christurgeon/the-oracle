"""Tests for the risk manager (python/risk/manager.py)."""

from __future__ import annotations

import dataclasses
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from python.risk.manager import (
    RiskDecision,
    RiskManager,
    StrategyTracker,
    _correlated_event_count,
)
from python.settings import RiskConfig

pytestmark = pytest.mark.usefixtures("monotonic_past_boot")

# ── helpers ────────────────────────────────────────────────────────────────


def _trade(pnl: float, won: bool, days_ago: int = 0) -> dict:
    return {
        "pnl": pnl,
        "won": won,
        "ts": datetime.now(UTC) - timedelta(days=days_ago),
    }


# ── StrategyTracker ───────────────────────────────────────────────────────


class TestStrategyTracker:
    def test_record_trade_appends(self):
        t = StrategyTracker(name="s")
        t.record_trade(10.0, True)
        assert len(t.trades) == 1

    def test_auto_timestamp(self):
        t = StrategyTracker(name="s")
        t.record_trade(5.0, True)
        assert isinstance(t.trades[0]["ts"], datetime)

    def test_custom_timestamp(self):
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        t = StrategyTracker(name="s")
        t.record_trade(5.0, True, timestamp=ts)
        assert t.trades[0]["ts"] == ts

    def test_rolling_pnl_within_window(self):
        t = StrategyTracker(name="s")
        t.record_trade(10.0, True)
        t.record_trade(-5.0, False)
        assert t.rolling_pnl(7) == pytest.approx(5.0)

    def test_rolling_pnl_outside_window(self):
        t = StrategyTracker(name="s")
        old = datetime.now(UTC) - timedelta(days=10)
        t.record_trade(10.0, True, timestamp=old)
        assert t.rolling_pnl(7) == pytest.approx(0.0)

    def test_rolling_pnl_mixed(self):
        t = StrategyTracker(name="s")
        old = datetime.now(UTC) - timedelta(days=10)
        t.record_trade(10.0, True, timestamp=old)
        t.record_trade(5.0, True)
        assert t.rolling_pnl(7) == pytest.approx(5.0)

    def test_rolling_pnl_empty(self):
        t = StrategyTracker(name="s")
        assert t.rolling_pnl(7) == 0.0

    def test_win_rate_all_wins(self):
        t = StrategyTracker(name="s")
        for _ in range(5):
            t.record_trade(1.0, True)
        rate, count = t.rolling_win_rate(7)
        assert rate == 1.0
        assert count == 5

    def test_win_rate_no_wins(self):
        t = StrategyTracker(name="s")
        for _ in range(5):
            t.record_trade(-1.0, False)
        rate, count = t.rolling_win_rate(7)
        assert rate == 0.0
        assert count == 5

    def test_win_rate_mixed(self):
        t = StrategyTracker(name="s")
        for _ in range(3):
            t.record_trade(1.0, True)
        for _ in range(7):
            t.record_trade(-1.0, False)
        rate, count = t.rolling_win_rate(7)
        assert rate == pytest.approx(0.3)
        assert count == 10

    def test_win_rate_empty(self):
        t = StrategyTracker(name="s")
        rate, count = t.rolling_win_rate(7)
        assert rate == 0.0
        assert count == 0

    def test_win_rate_excludes_old(self):
        t = StrategyTracker(name="s")
        old = datetime.now(UTC) - timedelta(days=10)
        t.record_trade(10.0, True, timestamp=old)
        t.record_trade(-5.0, False)
        rate, count = t.rolling_win_rate(7)
        assert rate == 0.0
        assert count == 1

    def test_prune_removes_old(self):
        t = StrategyTracker(name="s")
        old = datetime.now(UTC) - timedelta(days=20)
        t.record_trade(10.0, True, timestamp=old)
        t.record_trade(5.0, True)
        t.prune(max_age_days=7)
        assert len(t.trades) == 1

    def test_prune_keeps_recent(self):
        t = StrategyTracker(name="s")
        t.record_trade(5.0, True)
        t.record_trade(3.0, True)
        t.prune(max_age_days=7)
        assert len(t.trades) == 2

    def test_data_age_never_updated(self):
        t = StrategyTracker(name="s")
        assert t.data_age() == float("inf")

    def test_data_age_after_update(self):
        t = StrategyTracker(name="s")
        with patch("python.risk.manager.time.monotonic") as mock_mono:
            mock_mono.return_value = 100.0
            t.update_data_freshness()
            mock_mono.return_value = 105.0
            assert t.data_age() == 5.0

    def test_update_freshness_sets_monotonic(self):
        t = StrategyTracker(name="s")
        with patch("python.risk.manager.time.monotonic", return_value=42.0):
            t.update_data_freshness()
        assert t.last_data_ts == 42.0


# ── RiskDecision dataclass ────────────────────────────────────────────────


class TestRiskDecisionDataclass:
    def test_is_frozen(self, make_signal):
        signal = make_signal()
        d = RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.approved = False  # type: ignore[misc]

    def test_default_reasons_empty(self, make_signal):
        signal = make_signal()
        d = RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0)
        assert d.rejection_reasons == []


# ── RiskManager init ──────────────────────────────────────────────────────


class TestRiskManagerInit:
    def test_initial_bankroll(self, default_settings):
        rm = RiskManager(default_settings)
        assert rm.bankroll == default_settings.bankroll

    def test_initial_daily_pnl(self, risk_manager):
        assert risk_manager.daily_pnl == 0.0

    def test_initial_no_positions(self, risk_manager):
        assert risk_manager.positions == {}

    def test_initial_total_exposure(self, risk_manager):
        assert risk_manager.total_exposure == 0.0

    def test_initial_not_halted(self, risk_manager):
        assert risk_manager.is_halted is False

    def test_initial_drawdown(self, risk_manager):
        assert risk_manager.drawdown == 0.0


# ── check_signal: kill switch ─────────────────────────────────────────────


class TestCheckSignalKillSwitch:
    def test_active_rejects(self, risk_manager, make_signal, stop_file_path):
        stop_file_path.touch()
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("KILL_SWITCH" in r for r in decision.rejection_reasons)

    def test_inactive_passes(self, risk_manager, make_signal):
        decision = risk_manager.check_signal(make_signal())
        assert "KILL_SWITCH" not in str(decision.rejection_reasons)

    def test_hard_reject_returns_immediately(self, risk_manager, make_signal, stop_file_path):
        stop_file_path.touch()
        risk_manager._trading_halted = True  # would also fail
        decision = risk_manager.check_signal(make_signal())
        assert len(decision.rejection_reasons) == 1
        assert "KILL_SWITCH" in decision.rejection_reasons[0]

    def test_filesystem_error_fails_closed(self, risk_manager, make_signal):
        """OSError on stop_file.exists() must reject (fail-closed)."""
        with patch.object(
            type(risk_manager._settings.stop_file), "exists", side_effect=OSError("disk error")
        ):
            decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("KILL_SWITCH" in r for r in decision.rejection_reasons)


# ── check_signal: daily halt ──────────────────────────────────────────────


class TestCheckSignalDailyHalt:
    def test_halted_rejects(self, risk_manager, make_signal):
        risk_manager._trading_halted = True
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("DAILY_LOSS_HALT" in r for r in decision.rejection_reasons)

    def test_hard_reject(self, risk_manager, make_signal):
        risk_manager._trading_halted = True
        risk_manager._last_ws_data_time = time.monotonic() - 400  # also stale
        decision = risk_manager.check_signal(make_signal())
        assert len(decision.rejection_reasons) == 1

    def test_not_halted_passes(self, risk_manager, make_signal):
        decision = risk_manager.check_signal(make_signal())
        assert "DAILY_LOSS_HALT" not in str(decision.rejection_reasons)


# ── check_signal: stale data ─────────────────────────────────────────────


class TestStaleDataRejectionRecency:
    def test_none_until_a_stale_rejection(self, risk_manager, make_signal):
        assert risk_manager.stale_data_rejected_within(300) is False

    def test_true_right_after_a_stale_rejection(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400  # force stale
        decision = risk_manager.check_signal(make_signal())
        assert any("STALE_DATA" in r for r in decision.rejection_reasons)
        assert risk_manager.stale_data_rejected_within(300) is True

    def test_decays_after_window_without_fresh_data(
        self, risk_manager, make_signal, monkeypatch
    ):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        risk_manager.check_signal(make_signal())
        # Advance the clock 301s past the rejection — NO fresh frame arrives.
        base = time.monotonic()
        monkeypatch.setattr("python.risk.manager.time.monotonic", lambda: base + 301.0)
        assert risk_manager.stale_data_rejected_within(300) is False

    def test_non_stale_rejection_does_not_stamp(
        self, risk_manager, make_signal, stop_file_path
    ):
        risk_manager.update_data_freshness()  # fresh -> no STALE_DATA reason
        stop_file_path.touch()  # reject via kill switch (tmp stop file) instead
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert risk_manager.stale_data_rejected_within(300) is False


class TestCheckSignalStaleData:
    def test_stale_rejects(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("STALE_DATA" in r for r in decision.rejection_reasons)

    def test_fresh_passes(self, risk_manager, make_signal):
        risk_manager.update_data_freshness()
        decision = risk_manager.check_signal(make_signal())
        assert "STALE_DATA" not in str(decision.rejection_reasons)

    def test_is_soft_reason(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        decision = risk_manager.check_signal(make_signal())
        # Stale + disabled → at least 2 reasons (soft, not short-circuit)
        assert len(decision.rejection_reasons) >= 2


# ── check_signal: circuit breaker ─────────────────────────────────────────


class TestCheckSignalCircuitBreaker:
    def test_disabled_strategy_rejects(self, risk_manager, make_signal):
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("STRATEGY_DISABLED" in r for r in decision.rejection_reasons)

    def test_enabled_passes(self, risk_manager, make_signal):
        decision = risk_manager.check_signal(make_signal())
        assert "STRATEGY_DISABLED" not in str(decision.rejection_reasons)

    def test_auto_creates_tracker(self, risk_manager, make_signal):
        signal = make_signal(strategy_name="brand_new")
        decision = risk_manager.check_signal(signal)
        assert "STRATEGY_DISABLED" not in str(decision.rejection_reasons)


# ── check_signal: market cap ─────────────────────────────────────────────


class TestCheckSignalMarketCap:
    def test_no_position_full_room(self, risk_manager, make_signal):
        # cap = 0.05 * 0.40 * 10000 = 200; signal=100 ≤ 200
        decision = risk_manager.check_signal(make_signal(suggested_dollars=100))
        assert decision.approved
        assert decision.adjusted_dollars == 100.0

    def test_existing_position_reduces(self, risk_manager, make_signal):
        risk_manager._positions["MKT-001"] = 150  # room = 200 - 150 = 50
        decision = risk_manager.check_signal(make_signal(suggested_dollars=200))
        assert decision.approved
        assert decision.adjusted_dollars == 50.0

    def test_at_cap_rejects(self, risk_manager, make_signal):
        risk_manager._positions["MKT-001"] = 201  # just over cap of 200 → room = 0
        decision = risk_manager.check_signal(make_signal(suggested_dollars=100))
        assert not decision.approved
        assert any("MARKET_CAP" in r for r in decision.rejection_reasons)

    def test_over_cap_rejects(self, risk_manager, make_signal):
        risk_manager._positions["MKT-001"] = 300  # exceeds new cap of 200
        decision = risk_manager.check_signal(make_signal(suggested_dollars=100))
        assert not decision.approved
        assert any("MARKET_CAP" in r for r in decision.rejection_reasons)

    def test_standard_cap(self, risk_manager, make_signal):
        # cap = 0.05 * 0.40 * 10000 = 200, signal=400 → capped at 200
        signal = make_signal(strategy_name="weather_edge", suggested_dollars=400)
        decision = risk_manager.check_signal(signal)
        assert decision.approved
        assert decision.adjusted_dollars == 200.0


# ── check_signal: exposure-relative market cap ────────────────────────────


class TestCheckSignalMarketCapExposureScaling:
    def test_no_override_cap_uses_config_exposure(self, risk_manager, make_signal):
        # cap = 0.05 * 0.40 * 10000 = 200; signal 250 is reduced to 200
        decision = risk_manager.check_signal(make_signal(suggested_dollars=250))
        assert decision.approved
        assert decision.adjusted_dollars == 200.0

    def test_exposure_override_scales_cap_down(self, risk_manager, make_signal):
        # override 20%: cap = 0.05 * 0.20 * 10000 = 100
        risk_manager.set_exposure_override(0.20)
        decision = risk_manager.check_signal(make_signal(suggested_dollars=150))
        assert decision.approved
        assert decision.adjusted_dollars == 100.0

    def test_exposure_override_scales_cap_up(self, risk_manager, make_signal):
        # override 80%: cap = 0.05 * 0.80 * 10000 = 400
        risk_manager.set_exposure_override(0.80)
        decision = risk_manager.check_signal(make_signal(suggested_dollars=500))
        assert decision.approved
        assert decision.adjusted_dollars == 400.0

    def test_cap_proportional_to_exposure_fraction(self, risk_manager, make_signal):
        # 50% override: cap = 0.05 * 0.50 * 10000 = 250
        risk_manager.set_exposure_override(0.50)
        d1 = risk_manager.check_signal(make_signal(suggested_dollars=1000))
        assert d1.adjusted_dollars == 250.0
        # 25% override: cap = 0.05 * 0.25 * 10000 = 125
        risk_manager._positions.clear()
        risk_manager.set_exposure_override(0.25)
        d2 = risk_manager.check_signal(make_signal(suggested_dollars=1000))
        assert d2.adjusted_dollars == 125.0
        # Halving exposure fraction halves the per-trade cap
        assert d1.adjusted_dollars == pytest.approx(2 * d2.adjusted_dollars)


# ── check_signal: total exposure ──────────────────────────────────────────


class TestCheckSignalTotalExposure:
    def test_no_exposure_full_room(self, risk_manager, make_signal):
        # per-market cap = 0.05 * 0.40 * 10000 = 200; total cap = 4000
        # signal=100 ≤ 200 (per-market) and ≤ 4000 (total) → passes unchanged
        decision = risk_manager.check_signal(make_signal(suggested_dollars=100))
        assert decision.approved
        assert decision.adjusted_dollars == 100.0

    def test_partial_reduces(self, make_settings, make_signal):
        # max_single_market_pct=0.20; max_total stays 0.40 (default)
        # per-market cap = 0.20 * 0.40 * 10000 = 800; total cap = 4000
        # existing 3000; total room = 1000; per-market room = 800 (new position)
        # signal 1500 → per-market caps to 800 → total room 1000 ≥ 800 → approved at 800
        s = make_settings(risk=RiskConfig(max_single_market_pct=0.20))
        rm = RiskManager(s)
        rm._positions = {"M1": 1000, "M2": 1000, "M3": 1000}
        decision = rm.check_signal(make_signal(suggested_dollars=1500))
        assert decision.approved
        assert decision.adjusted_dollars == 800.0

    def test_at_cap_rejects(self, make_settings, make_signal):
        s = make_settings(risk=RiskConfig(max_single_market_pct=0.20))
        rm = RiskManager(s)
        rm._positions = {"M1": 2000, "M2": 2000}  # total = 4000 = cap
        decision = rm.check_signal(make_signal(suggested_dollars=100))
        assert not decision.approved
        assert any("TOTAL_EXPOSURE_CAP" in r for r in decision.rejection_reasons)

    def test_interacts_with_market_cap(self, risk_manager, make_signal):
        # per-market cap = 0.05 * 0.40 * 10000 = 200; MKT-001 has 150 → room = 50
        # total = 150+1700+1850 = 3700; total room = 300; 50 < 300 → market cap binds
        risk_manager._positions = {"MKT-001": 150, "M2": 1700, "M3": 1850}
        decision = risk_manager.check_signal(make_signal(suggested_dollars=500))
        assert decision.approved
        assert decision.adjusted_dollars == 50.0


# ── check_signal: drawdown ────────────────────────────────────────────────


class TestCheckSignalDrawdown:
    def test_reduces_size(self, risk_manager, make_signal):
        risk_manager._bankroll = 8500
        risk_manager._peak_bankroll = 10_000  # drawdown = 15%
        # max_dollars = 0.05 * 0.40 * 8500 = 170, reduced_cap = 170 * 0.5 = 85
        # signal 400 > 170 → per-market clamps to 170 → drawdown clamps to 85
        decision = risk_manager.check_signal(make_signal(suggested_dollars=400))
        assert decision.adjusted_dollars == 85.0

    def test_below_threshold_no_reduction(self, risk_manager, make_signal):
        risk_manager._bankroll = 9000
        risk_manager._peak_bankroll = 10_000  # drawdown = 10% < 15% threshold
        # max_dollars = 0.05 * 0.40 * 9000 = 180; signal=100 ≤ 180 → no clamping
        decision = risk_manager.check_signal(make_signal(suggested_dollars=100))
        assert decision.adjusted_dollars == 100.0

    def test_not_a_rejection_reason(self, risk_manager, make_signal):
        risk_manager._bankroll = 8500
        risk_manager._peak_bankroll = 10_000
        decision = risk_manager.check_signal(make_signal(suggested_dollars=400))
        assert decision.approved
        assert decision.rejection_reasons == []

    def test_combined_with_cap(self, risk_manager, make_signal):
        risk_manager._bankroll = 8500
        risk_manager._peak_bankroll = 10_000  # drawdown = 15%
        # max_dollars = 0.05 * 0.40 * 8500 = 170; reduced_cap = 170 * 0.5 = 85
        # No existing position. signal 400 → per-market cap 170 → drawdown cap 85
        decision = risk_manager.check_signal(make_signal(suggested_dollars=400))
        assert decision.approved
        assert decision.adjusted_dollars == 85.0


# ── check_signal: combined scenarios ──────────────────────────────────────


class TestCheckSignalCombined:
    def test_stale_and_disabled(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert len(decision.rejection_reasons) == 2

    def test_stale_and_market_cap(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        risk_manager._positions["MKT-001"] = 500
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert any("STALE_DATA" in r for r in decision.rejection_reasons)
        assert any("MARKET_CAP" in r for r in decision.rejection_reasons)

    def test_all_soft_reasons(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        risk_manager._positions = {
            "M1": 1000, "M2": 1000, "M3": 1000, "M4": 500, "MKT-001": 500,
        }
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert len(decision.rejection_reasons) == 4

    def test_clean_approved(self, risk_manager, make_signal):
        decision = risk_manager.check_signal(make_signal())
        assert decision.approved
        assert decision.rejection_reasons == []
        assert decision.adjusted_dollars == 100.0

    def test_reasons_zero_adjusted_dollars(self, risk_manager, make_signal):
        risk_manager._last_ws_data_time = time.monotonic() - 400
        decision = risk_manager.check_signal(make_signal())
        assert not decision.approved
        assert decision.adjusted_dollars == 0.0


# ── record_fill ───────────────────────────────────────────────────────────


class TestRecordFill:
    def test_new_market_position(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        assert risk_manager.positions["M1"] == 100

    def test_add_to_existing(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.record_fill("M1", "s", 50)
        assert risk_manager.positions["M1"] == 150

    def test_pnl_updates_daily(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-20)
        assert risk_manager.daily_pnl == -20

    def test_pnl_updates_bankroll(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-20)
        assert risk_manager.bankroll == 10_000 - 20

    def test_positive_pnl_updates_peak(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=500)
        assert risk_manager._peak_bankroll == 10_500

    def test_negative_pnl_preserves_peak(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-200)
        assert risk_manager._peak_bankroll == 10_000

    def test_halt_triggered_at_limit(self, risk_manager):
        # daily loss limit = 0.05 * 10000 = 500
        risk_manager.record_fill("M1", "s", 100, pnl=-500)
        assert risk_manager.is_halted

    def test_halt_not_triggered_above(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-499.99)
        assert not risk_manager.is_halted

    def test_halt_cumulative(self, risk_manager):
        for i in range(4):
            risk_manager.record_fill(f"M{i}", "s", 100, pnl=-100)
        assert not risk_manager.is_halted
        risk_manager.record_fill("M4", "s", 100, pnl=-100)  # total -500
        assert risk_manager.is_halted

    def test_total_exposure_after_fills(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.record_fill("M2", "s", 200)
        assert risk_manager.total_exposure == 300


# ── close_position ────────────────────────────────────────────────────────


class TestClosePosition:
    def test_removes_position(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.close_position("M1")
        assert "M1" not in risk_manager.positions

    def test_pnl_updates_bankroll(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.close_position("M1", pnl=50)
        assert risk_manager.bankroll == 10_050

    def test_pnl_updates_daily(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.close_position("M1", pnl=-30)
        assert risk_manager.daily_pnl == -30

    def test_positive_pnl_updates_peak(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.close_position("M1", pnl=500)
        assert risk_manager._peak_bankroll == 10_500

    def test_nonexistent_no_error(self, risk_manager):
        risk_manager.close_position("NONEXISTENT")  # no-op, no error

    def test_triggers_halt_on_large_loss(self, risk_manager):
        """Fix #5: settlement losses through close_position must trigger halt."""
        risk_manager.close_position("M1", pnl=-600)
        assert risk_manager.is_halted

    def test_no_halt_on_small_loss(self, risk_manager):
        risk_manager.close_position("M1", pnl=-100)
        assert not risk_manager.is_halted

    def test_exposure_decreases(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.record_fill("M2", "s", 200)
        risk_manager.close_position("M1")
        assert risk_manager.total_exposure == 200


# ── reset_daily ───────────────────────────────────────────────────────────


class TestResetDaily:
    def test_clears_daily_pnl(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-200)
        risk_manager.reset_daily()
        assert risk_manager.daily_pnl == 0.0

    def test_updates_start_bankroll(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-200)
        risk_manager.reset_daily()
        assert risk_manager._daily_start_bankroll == risk_manager.bankroll

    def test_clears_halt(self, risk_manager):
        risk_manager._trading_halted = True
        risk_manager.reset_daily()
        assert not risk_manager.is_halted

    def test_preserves_positions(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100)
        risk_manager.reset_daily()
        assert "M1" in risk_manager.positions

    def test_preserves_bankroll(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=500)
        risk_manager.reset_daily()
        assert risk_manager.bankroll == 10_500


# ── check_circuit_breakers ────────────────────────────────────────────────


class TestCheckCircuitBreakers:
    def test_pnl_trips(self, risk_manager):
        # threshold = -(0.03 * 10000) = -300 → need < -300
        tracker = risk_manager._get_tracker("s")
        tracker.record_trade(-310, False)
        disabled = risk_manager.check_circuit_breakers()
        assert not tracker.enabled
        assert "s" in disabled

    def test_pnl_at_threshold_no_trip(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        tracker.record_trade(-300, False)  # exactly -300, not < -300
        disabled = risk_manager.check_circuit_breakers()
        assert tracker.enabled
        assert disabled == []

    def test_win_rate_trips(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        for _ in range(5):
            tracker.record_trade(1, True)
        for _ in range(15):
            tracker.record_trade(-0.5, False)
        # rolling pnl = 5 - 7.5 = -2.5 (above -300, so P&L passes)
        # win rate = 5/20 = 0.25 < 0.35 → trips
        disabled = risk_manager.check_circuit_breakers()
        assert not tracker.enabled
        assert "s" in disabled

    def test_not_enough_trades_skips(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        for _ in range(5):
            tracker.record_trade(1, True)
        for _ in range(5):
            tracker.record_trade(-1, False)
        # 10 trades < 20 → win rate skipped; P&L = 0 → no trip
        disabled = risk_manager.check_circuit_breakers()
        assert tracker.enabled
        assert disabled == []

    def test_win_rate_above_min(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        for _ in range(8):
            tracker.record_trade(1, True)
        for _ in range(12):
            tracker.record_trade(-0.5, False)
        # P&L = 8 - 6 = 2 (fine); win rate = 8/20 = 0.40 > 0.35 → no trip
        risk_manager.check_circuit_breakers()
        assert tracker.enabled

    def test_disabled_skipped(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        tracker.enabled = False
        tracker.record_trade(-100, False)
        disabled = risk_manager.check_circuit_breakers()
        assert disabled == []

    def test_prune_2x_window(self, risk_manager):
        # window = 7 → prune max_age = 14
        tracker = risk_manager._get_tracker("s")
        old = datetime.now(UTC) - timedelta(days=20)
        tracker.record_trade(10.0, True, timestamp=old)
        tracker.record_trade(5.0, True)
        risk_manager.check_circuit_breakers()
        assert len(tracker.trades) == 1

    def test_multiple_strategies(self, risk_manager):
        t1 = risk_manager._get_tracker("a")
        t2 = risk_manager._get_tracker("b")
        t1.record_trade(-310, False)  # exceeds 0.03 * 10000 = 300 threshold
        t2.record_trade(10, True)
        disabled = risk_manager.check_circuit_breakers()
        assert not t1.enabled
        assert t2.enabled
        assert disabled == ["a"]

    def test_pnl_before_win_rate(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        for _ in range(20):
            tracker.record_trade(-16, False)
        # P&L = -320 < -300 → trips on P&L, continue skips win rate
        risk_manager.check_circuit_breakers()
        assert "Rolling" in tracker.disabled_reason

    def test_returns_disabled_names(self, risk_manager):
        risk_manager._get_tracker("a").record_trade(-310, False)
        risk_manager._get_tracker("b").record_trade(-310, False)
        disabled = risk_manager.check_circuit_breakers()
        assert set(disabled) == {"a", "b"}

    def test_zero_bankroll_no_trip(self, risk_manager):
        risk_manager._bankroll = 0
        tracker = risk_manager._get_tracker("s")
        tracker.record_trade(-100, False)
        disabled = risk_manager.check_circuit_breakers()
        assert tracker.enabled
        assert disabled == []


# ── enable_strategy ───────────────────────────────────────────────────────


class TestEnableStrategy:
    def test_sets_enabled(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        tracker.enabled = False
        risk_manager.enable_strategy("s")
        assert tracker.enabled

    def test_clears_reason(self, risk_manager):
        tracker = risk_manager._get_tracker("s")
        tracker.enabled = False
        tracker.disabled_reason = "bad luck"
        risk_manager.enable_strategy("s")
        assert tracker.disabled_reason == ""

    def test_creates_if_missing(self, risk_manager):
        risk_manager.enable_strategy("brand_new")
        tracker = risk_manager._get_tracker("brand_new")
        assert tracker.enabled


# ── Properties ────────────────────────────────────────────────────────────


class TestProperties:
    def test_bankroll(self, risk_manager):
        assert risk_manager.bankroll == risk_manager._bankroll

    def test_daily_pnl(self, risk_manager):
        assert risk_manager.daily_pnl == risk_manager._daily_pnl

    def test_total_exposure_sums(self, risk_manager):
        risk_manager._positions = {"A": 100, "B": 200, "C": 300}
        assert risk_manager.total_exposure == 600

    def test_total_exposure_empty(self, risk_manager):
        assert risk_manager.total_exposure == 0.0

    def test_positions_copy(self, risk_manager):
        risk_manager._positions["M1"] = 500
        copy = risk_manager.positions
        copy["NEW"] = 999
        assert "NEW" not in risk_manager.positions

    def test_is_halted_default(self, risk_manager):
        assert risk_manager.is_halted is False

    def test_is_halted_after_halt(self, risk_manager):
        risk_manager.record_fill("M1", "s", 100, pnl=-500)
        assert risk_manager.is_halted is True

    def test_drawdown_formula(self, risk_manager):
        risk_manager._bankroll = 9000
        risk_manager._peak_bankroll = 10_000
        assert risk_manager.drawdown == pytest.approx(0.1)

    def test_drawdown_at_peak(self, risk_manager):
        assert risk_manager.drawdown == 0.0

    def test_drawdown_peak_zero(self, risk_manager):
        risk_manager._peak_bankroll = 0
        assert risk_manager.drawdown == 0.0


# ── update_data_freshness ─────────────────────────────────────────────────


class TestUpdateDataFreshness:
    def test_updates_ws_time(self, risk_manager):
        before = risk_manager._last_ws_data_time
        time.sleep(0.01)
        risk_manager.update_data_freshness()
        assert risk_manager._last_ws_data_time > before


# ── get_tracker ───────────────────────────────────────────────────────────


class TestGetTracker:
    def test_creates_new(self, risk_manager):
        tracker = risk_manager.get_tracker("unknown")
        assert tracker.enabled
        assert tracker.name == "unknown"

    def test_returns_existing(self, risk_manager):
        t1 = risk_manager.get_tracker("x")
        t2 = risk_manager.get_tracker("x")
        assert t1 is t2


# ── record_strategy_trade ────────────────────────────────────────────────


class TestRecordStrategyTrade:
    def test_record_strategy_trade_win(self, risk_manager):
        risk_manager.record_strategy_trade("weather_edge", pnl=5.0, won=True)
        tracker = risk_manager.get_tracker("weather_edge")
        assert len(tracker.trades) == 1
        assert tracker.trades[0]["pnl"] == 5.0
        assert tracker.trades[0]["won"] is True

    def test_record_strategy_trade_loss(self, risk_manager):
        risk_manager.record_strategy_trade("weather_edge", pnl=-3.0, won=False)
        tracker = risk_manager.get_tracker("weather_edge")
        assert len(tracker.trades) == 1
        assert tracker.trades[0]["pnl"] == -3.0
        assert tracker.trades[0]["won"] is False

    def test_record_strategy_trade_creates_tracker(self, risk_manager):
        risk_manager.record_strategy_trade("brand_new_strat", pnl=1.0, won=True)
        tracker = risk_manager.get_tracker("brand_new_strat")
        assert tracker.name == "brand_new_strat"
        assert len(tracker.trades) == 1

    def test_circuit_breaker_trips_after_losses(self, risk_manager):
        # threshold = -(0.03 * 10000) = -300 → need < -300 to trip
        risk_manager.record_strategy_trade("bad_strat", pnl=-310.0, won=False)
        disabled = risk_manager.check_circuit_breakers()
        assert "bad_strat" in disabled
        tracker = risk_manager.get_tracker("bad_strat")
        assert not tracker.enabled


# ── exposure override ────────────────────────────────────────────────────


class TestExposureOverride:
    def test_default_is_config_baseline(self, risk_manager):
        assert risk_manager.exposure_override is None
        assert (
            risk_manager.effective_max_total_exposure_pct
            == risk_manager._risk.max_total_exposure_pct
        )

    def test_set_override_updates_effective(self, risk_manager):
        new_effective = risk_manager.set_exposure_override(0.25)
        assert new_effective == 0.25
        assert risk_manager.exposure_override == 0.25
        assert risk_manager.effective_max_total_exposure_pct == 0.25

    def test_override_tightens_check_signal(self, make_settings, make_signal):
        # Baseline 0.95 → room = $9500. Override 0.10 → total cap = $1000.
        # max_single_market_pct=0.20 so per-market cap = 0.20 * 0.10 * 10000 = 200,
        # which is larger than the total room of $100 (= $1000 - $900 existing).
        # So total exposure is the binding constraint → reduced to $100.
        s = make_settings(risk=RiskConfig(max_total_exposure_pct=0.95, max_single_market_pct=0.20))
        rm = RiskManager(s)
        rm.set_exposure_override(0.10)
        rm._positions = {"M1": 900.0}
        # New signal of $200 should be reduced to $100 (room = $1000 - $900).
        decision = rm.check_signal(make_signal(suggested_dollars=200))
        assert decision.approved
        assert decision.adjusted_dollars == 100.0

    def test_override_zero_rejected(self, risk_manager):
        with pytest.raises(ValueError):
            risk_manager.set_exposure_override(0.0)

    def test_override_above_one_rejected(self, risk_manager):
        with pytest.raises(ValueError):
            risk_manager.set_exposure_override(1.01)

    def test_override_negative_rejected(self, risk_manager):
        with pytest.raises(ValueError):
            risk_manager.set_exposure_override(-0.1)

    def test_clear_override_reverts_to_baseline(self, risk_manager):
        baseline = risk_manager._risk.max_total_exposure_pct
        risk_manager.set_exposure_override(0.10)
        assert risk_manager.effective_max_total_exposure_pct == 0.10
        risk_manager.set_exposure_override(None)
        assert risk_manager.exposure_override is None
        assert risk_manager.effective_max_total_exposure_pct == baseline

    def test_persist_and_hydrate_roundtrip(self, risk_manager, tmp_state_dir):
        risk_manager.set_exposure_override(0.30)
        risk_manager.persist_exposure_override(tmp_state_dir)
        # Fresh manager with the same settings should pick up the override.
        fresh = RiskManager(risk_manager._settings)
        assert fresh.exposure_override is None
        fresh.hydrate_exposure_override(tmp_state_dir)
        assert fresh.exposure_override == 0.30
        assert fresh.effective_max_total_exposure_pct == 0.30

    def test_clear_override_removes_file(self, risk_manager, tmp_state_dir):
        from python.risk.persistence import EXPOSURE_OVERRIDE_FILENAME

        risk_manager.set_exposure_override(0.30)
        risk_manager.persist_exposure_override(tmp_state_dir)
        path = tmp_state_dir / EXPOSURE_OVERRIDE_FILENAME
        assert path.exists()

        risk_manager.set_exposure_override(None)
        risk_manager.persist_exposure_override(tmp_state_dir)
        assert not path.exists()

    def test_hydrate_missing_file_no_op(self, risk_manager, tmp_state_dir):
        # No file present — hydration is a no-op.
        risk_manager.hydrate_exposure_override(tmp_state_dir)
        assert risk_manager.exposure_override is None


# ── RiskManager.realize_partial (v1.1 §6.7) ────────────────────────────────


def _make_risk(state_dir, bankroll: float = 500.0) -> RiskManager:
    """Real RiskManager constructed via the Task 0 conftest helper.

    Use this (NOT a SimpleNamespace mock) so the test verifies actual
    state mutations on _bankroll / _daily_pnl / _peak_bankroll / _state_dirty
    AND triggers _check_daily_loss_halt's real logic.

    bankroll defaults to 500.0 to clear the settings-floor invariant
    (kelly 0.15 * single 0.05 * exposure 0.40 * bankroll ≥ 1.0
    → bankroll ≥ 333.3).
    """
    from tests.conftest import make_test_risk_manager
    return make_test_risk_manager(bankroll=bankroll, state_dir=state_dir)


def test_realize_partial_reduces_position_and_books_pnl(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 2.34
    prev_bankroll = risk._bankroll
    prev_daily = risk._daily_pnl

    risk.realize_partial("KXT", dollars_released=2.25, pnl=0.92)

    assert risk._positions["KXT"] == pytest.approx(0.09)
    assert risk._bankroll == pytest.approx(prev_bankroll + 0.92)
    assert risk._daily_pnl == pytest.approx(prev_daily + 0.92)


def test_realize_partial_clamps_at_zero_when_dollars_exceed_current(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 1.00

    risk.realize_partial("KXT", dollars_released=1.50, pnl=0.50)

    # Clamped at 0.0; NOT popped; NOT negative.
    assert risk._positions["KXT"] == 0.0


def test_realize_partial_clamps_at_zero_when_current_is_zero(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    # current == 0 initial state (e.g., post-divergence rehydrate).
    risk._positions["KXT"] = 0.0

    risk.realize_partial("KXT", dollars_released=1.00, pnl=0.50)

    # The earlier draft's condition (current > 0) bypassed the guard;
    # the unconditional max(0, ...) prevents negative exposure.
    assert risk._positions["KXT"] == 0.0


def test_realize_partial_clamps_when_dollars_released_equal_current(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 1.00

    risk.realize_partial("KXT", dollars_released=1.00, pnl=0.50)

    # Equal case: clamps at 0.0 (no pop). Residual contracts will settle later.
    assert risk._positions["KXT"] == 0.0


def test_realize_partial_updates_peak_bankroll(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 2.34
    prev_peak = risk._peak_bankroll

    risk.realize_partial("KXT", dollars_released=2.25, pnl=10.0)

    assert risk._peak_bankroll == max(prev_peak, risk._bankroll)


def test_realize_partial_state_dirty_flag(tmp_state_dir) -> None:
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 2.34
    risk._state_dirty = False

    risk.realize_partial("KXT", dollars_released=2.25, pnl=0.92)

    assert risk._state_dirty is True


def test_realize_partial_triggers_daily_loss_halt_on_large_negative_pnl(tmp_state_dir) -> None:
    """Regression guard: dropping the _check_daily_loss_halt call from
    realize_partial must trip this test. Spec §6.7 mirrors close_position's
    halt-check ordering.
    """
    risk = _make_risk(tmp_state_dir)
    risk._positions["KXT"] = 50.0
    # Halt threshold = max_daily_loss_pct * _daily_start_bankroll (500.0).
    # Set pct=0.06 → halt at -$30 → a -$30 partial loss trips it
    # (the inequality is `_daily_pnl <= -daily_loss_limit`).
    risk._risk.max_daily_loss_pct = 0.06  # type: ignore[attr-defined]

    risk.realize_partial("KXT", dollars_released=20.0, pnl=-30.0)

    assert risk._trading_halted is True


# ── Correlated-positions cap (event-aware) ─────────────────────────────────


class TestCorrelatedEventCount:
    """Pure correlation-count helper: distinct OTHER forecast dates held in
    the same weather series, collapsing within-event brackets to one unit."""

    def test_no_open_positions_returns_zero(self):
        assert _correlated_event_count("KXHIGHAUS-26APR22-B79.5", []) == 0

    def test_within_event_brackets_collapse_to_zero(self):
        # 3 brackets, SAME (series, date) as candidate → mutually exclusive
        # coverage → the candidate's own event never counts.
        open_keys = [
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR22-B81.5",
            "KXHIGHAUS-26APR22-B83.5",
        ]
        assert _correlated_event_count("KXHIGHAUS-26APR22-B85.5", open_keys) == 0

    def test_distinct_dates_same_series_counted(self):
        open_keys = [
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR23-B81.5",
            "KXHIGHAUS-26APR24-B83.5",
        ]
        # Candidate on a 4th date: three OTHER dates held in the series.
        assert _correlated_event_count("KXHIGHAUS-26APR25-B85.5", open_keys) == 3

    def test_same_date_multiple_brackets_counts_once(self):
        # 5 brackets on APR22 + 1 on APR23 → two distinct OTHER dates.
        open_keys = [
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR22-B81.5",
            "KXHIGHAUS-26APR22-B83.5",
            "KXHIGHAUS-26APR22-B85.5",
            "KXHIGHAUS-26APR22-B87.5",
            "KXHIGHAUS-26APR23-B81.5",
        ]
        assert _correlated_event_count("KXHIGHAUS-26APR24-B85.5", open_keys) == 2

    def test_candidate_date_already_held_not_counted(self):
        # Adding a bracket to a date already held does not count that date.
        open_keys = [
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR23-B81.5",
        ]
        assert _correlated_event_count("KXHIGHAUS-26APR22-B85.5", open_keys) == 1

    def test_different_series_not_counted(self):
        # Same city, different variable (HIGH vs LOW) = different series → ignored.
        open_keys = [
            "KXHIGHNY-26APR22-B56.5",
            "KXHIGHNY-26APR23-B58.5",
        ]
        assert _correlated_event_count("KXLOWTNYC-26APR22-B40.5", open_keys) == 0

    def test_different_city_same_var_not_counted(self):
        open_keys = [
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR23-B81.5",
        ]
        assert _correlated_event_count("KXHIGHNY-26APR22-B56.5", open_keys) == 0

    def test_non_weather_key_returns_zero(self):
        assert _correlated_event_count("MKT-001-T", ["MKT-002-T"]) == 0

    def test_short_or_malformed_keys_ignored(self):
        # Candidate with <3 parts → 0; malformed open keys are skipped.
        assert _correlated_event_count("KXHIGHAUS-26APR22", ["KXHIGHAUS"]) == 0
        assert (
            _correlated_event_count(
                "KXHIGHAUS-26APR25-B85.5", ["KXHIGHAUS", "KXHIGHAUS-26APR22-B79.5"]
            )
            == 1
        )

    def test_case_insensitive_kx_prefix(self):
        # position_key.startswith check is on upper(); lowercase kx still applies.
        assert (
            _correlated_event_count(
                "kxhighaus-26apr25-b85.5", ["kxhighaus-26apr22-b79.5"]
            )
            == 1
        )


class TestCheckSignalCorrelatedPositions:
    """Event-aware correlated-positions cap: within-event bracket coverage is
    free; the cap binds on distinct other dates in the same series."""

    @staticmethod
    def _seed(rm, *keys):
        for k in keys:
            rm._positions[k] = 10.0

    def test_within_event_bracket_coverage_not_blocked(self, risk_manager, make_signal):
        # 3 existing brackets in ONE event; a 4th bracket in the same event must
        # NOT be blocked by the correlation cap (old behavior blocked it).
        self._seed(
            risk_manager,
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR22-B81.5",
            "KXHIGHAUS-26APR22-B83.5",
        )
        sig = make_signal(
            market_id="KXHIGHAUS-26APR22-B85.5", market_ticker="KXHIGHAUS-26APR22-B85.5"
        )
        decision = risk_manager.check_signal(sig)
        assert not any("CORRELATED_POSITIONS" in r for r in decision.rejection_reasons)

    def test_fourth_distinct_date_same_series_blocked(self, risk_manager, make_signal):
        # 3 distinct dates already held in the series; a 4th date trips the cap.
        self._seed(
            risk_manager,
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR23-B81.5",
            "KXHIGHAUS-26APR24-B83.5",
        )
        sig = make_signal(
            market_id="KXHIGHAUS-26APR25-B85.5", market_ticker="KXHIGHAUS-26APR25-B85.5"
        )
        decision = risk_manager.check_signal(sig)
        assert any("CORRELATED_POSITIONS" in r for r in decision.rejection_reasons)
        assert not decision.approved

    def test_third_distinct_date_allowed_boundary(self, risk_manager, make_signal):
        # Only 2 distinct other dates held → 3rd date is allowed (count 2 < max 3).
        self._seed(
            risk_manager,
            "KXHIGHAUS-26APR22-B79.5",
            "KXHIGHAUS-26APR23-B81.5",
        )
        sig = make_signal(
            market_id="KXHIGHAUS-26APR24-B85.5", market_ticker="KXHIGHAUS-26APR24-B85.5"
        )
        decision = risk_manager.check_signal(sig)
        assert not any("CORRELATED_POSITIONS" in r for r in decision.rejection_reasons)

    def test_different_series_not_blocked(self, risk_manager, make_signal):
        # A loaded HIGH book must not block a LOW signal for the same city.
        self._seed(
            risk_manager,
            "KXHIGHNY-26APR22-B56.5",
            "KXHIGHNY-26APR23-B58.5",
            "KXHIGHNY-26APR24-B60.5",
            "KXHIGHNY-26APR25-B62.5",
        )
        sig = make_signal(
            market_id="KXLOWTNYC-26APR22-B40.5", market_ticker="KXLOWTNYC-26APR22-B40.5"
        )
        decision = risk_manager.check_signal(sig)
        assert not any("CORRELATED_POSITIONS" in r for r in decision.rejection_reasons)

    def test_non_weather_signal_not_blocked(self, risk_manager, make_signal):
        self._seed(risk_manager, "MKT-001-T", "MKT-002-T", "MKT-003-T", "MKT-004-T")
        sig = make_signal()  # default MKT-001 / MKT-001-T
        decision = risk_manager.check_signal(sig)
        assert not any("CORRELATED_POSITIONS" in r for r in decision.rejection_reasons)
