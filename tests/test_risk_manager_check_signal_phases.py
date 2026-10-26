"""Characterization tests pinning the phase ordering of ``RiskManager.check_signal``.

Complements tests/test_risk_manager.py: those assert that individual reasons
appear; these pin the exact content and order of ``rejection_reasons`` and
``size_adjustments`` across the hard-stop, size-cap, and soft-block phases.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import pytest

from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal

pytestmark = pytest.mark.usefixtures("monotonic_past_boot")

# Default test settings: bankroll 10_000, max_single_market_pct 0.05,
# max_total_exposure_pct 0.40, max_drawdown_pct 0.15,
# drawdown_size_reduction 0.5, max_correlated_positions 3, stale cutoff 300s.
# => market cap $200, total exposure cap $4000.
# Positions of exactly the cap are avoided: 0.05 * 0.40 * 10_000 is 200.00000000000003.
TOTAL_CAP = 4000.0
OVER_MARKET_CAP = 250.0

SignalFactory = Callable[..., TradeSignal]


def _reasons_kinds(reasons: list[str]) -> list[str]:
    return [r.split(":", 1)[0] for r in reasons]


def _go_stale(rm: RiskManager) -> None:
    rm._last_ws_data_time = time.monotonic() - 400


def _enter_drawdown(rm: RiskManager) -> None:
    rm._peak_bankroll = 12_000.0  # 10_000 / 12_000 -> drawdown 16.7% >= 15%


class TestHardStopShortCircuit:
    def test_kill_switch_masks_every_other_check(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        stop_file_path,
    ) -> None:
        stop_file_path.touch()
        risk_manager._positions["MKT-001"] = OVER_MARKET_CAP
        risk_manager._positions["OTHER"] = TOTAL_CAP
        _go_stale(risk_manager)
        _enter_drawdown(risk_manager)
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"

        decision = risk_manager.check_signal(make_signal())

        assert not decision.approved
        assert decision.rejection_reasons == ["KILL_SWITCH: state/STOP file detected"]
        assert decision.size_adjustments == []
        assert decision.adjusted_dollars == 0.0
        assert risk_manager._last_stale_reject_monotonic is None

    def test_daily_halt_masks_every_other_check(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        risk_manager._trading_halted = True
        risk_manager._daily_pnl = -600.0
        risk_manager._positions["MKT-001"] = OVER_MARKET_CAP
        risk_manager._positions["OTHER"] = TOTAL_CAP
        _go_stale(risk_manager)
        _enter_drawdown(risk_manager)

        decision = risk_manager.check_signal(make_signal())

        assert not decision.approved
        assert decision.rejection_reasons == [
            "DAILY_LOSS_HALT: daily P&L -600.00 exceeded 5% of bankroll"
        ]
        assert decision.size_adjustments == []
        assert decision.adjusted_dollars == 0.0
        assert risk_manager._last_stale_reject_monotonic is None

    def test_kill_switch_wins_over_daily_halt(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        stop_file_path,
    ) -> None:
        stop_file_path.touch()
        risk_manager._trading_halted = True
        decision = risk_manager.check_signal(make_signal())
        assert _reasons_kinds(decision.rejection_reasons) == ["KILL_SWITCH"]


class TestSizeCapOrdering:
    def test_market_then_total_then_drawdown_adjustments_in_order(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        # market room 180 (< 300) -> total room 130 (< 180) -> drawdown cap 100 (< 130).
        # The drawdown cap only binds because current_exposure (20) < 200 * 0.5.
        risk_manager._positions["MKT-001"] = 20.0
        risk_manager._positions["OTHER"] = 3850.0
        _enter_drawdown(risk_manager)

        decision = risk_manager.check_signal(make_signal(suggested_dollars=300.0))

        assert decision.approved
        assert decision.rejection_reasons == []
        assert decision.size_adjustments == [
            "MARKET_CAP: reduced $300.00 → $180.00 (existing exposure $20.00)",
            "TOTAL_EXPOSURE: reduced $180.00 → $130.00 (total exposure $3870.00)",
            "DRAWDOWN: reduced $130.00 → $100.00 (drawdown 16.7% >= 15%)",
        ]
        assert decision.adjusted_dollars == 100.0

    def test_drawdown_not_binding_after_market_cap_is_silent(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # market room 50 already below drawdown cap 100.
        risk_manager._positions["MKT-001"] = 150.0
        _enter_drawdown(risk_manager)

        with caplog.at_level(logging.WARNING, logger="python.risk.manager"):
            decision = risk_manager.check_signal(make_signal(suggested_dollars=300.0))

        assert decision.approved
        assert decision.size_adjustments == [
            "MARKET_CAP: reduced $300.00 → $50.00 (existing exposure $150.00)",
        ]
        assert decision.adjusted_dollars == 50.0
        assert [r for r in caplog.records if "Drawdown" in r.getMessage()] == []

    def test_drawdown_binding_logs_warning(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _enter_drawdown(risk_manager)

        with caplog.at_level(logging.WARNING, logger="python.risk.manager"):
            decision = risk_manager.check_signal(make_signal(suggested_dollars=150.0))

        assert decision.size_adjustments == [
            "DRAWDOWN: reduced $150.00 → $100.00 (drawdown 16.7% >= 15%)",
        ]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].getMessage() == (
            "Drawdown 16.7% >= 15.0% — capping position at $100.00 (50% reduction)"
        )

    def test_market_cap_is_drawdown_base(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        _enter_drawdown(risk_manager)
        decision = risk_manager.check_signal(make_signal(suggested_dollars=300.0))
        assert decision.size_adjustments == [
            "MARKET_CAP: reduced $300.00 → $200.00 (existing exposure $0.00)",
            "DRAWDOWN: reduced $200.00 → $100.00 (drawdown 16.7% >= 15%)",
        ]
        assert decision.adjusted_dollars == 100.0


class TestSoftBlockAccumulation:
    @staticmethod
    def _seed_all_blocks(rm: RiskManager) -> str:
        key = "KXHIGHAUS-26APR25-B85.5"
        rm._positions.update(
            {
                key: OVER_MARKET_CAP,
                "KXHIGHAUS-26APR22-B79.5": 10.0,
                "KXHIGHAUS-26APR23-B81.5": 10.0,
                "KXHIGHAUS-26APR24-B83.5": 10.0,
                "OTHER": 3800.0,
            }
        )
        _go_stale(rm)
        tracker = rm._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        return key

    def test_five_reasons_in_exact_order_with_no_adjustments(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        key = self._seed_all_blocks(risk_manager)

        decision = risk_manager.check_signal(make_signal(market_id=key, market_ticker=key))

        assert not decision.approved
        assert _reasons_kinds(decision.rejection_reasons) == [
            "MARKET_CAP",
            "TOTAL_EXPOSURE_CAP",
            "STALE_DATA",
            "STRATEGY_DISABLED",
            "CORRELATED_POSITIONS",
        ]
        assert decision.rejection_reasons[0] == (
            f"MARKET_CAP: already at max exposure for {key} (250.00 / 200.00)"
        )
        assert decision.rejection_reasons[1] == (
            "TOTAL_EXPOSURE_CAP: total exposure 4080.00 at max 4000.00"
        )
        assert decision.rejection_reasons[3] == "STRATEGY_DISABLED: weather_edge — test"
        assert decision.rejection_reasons[4] == (
            "CORRELATED_POSITIONS: 3 other forecast date(s) held in series 'KXHIGHAUS' >= max 3"
        )
        assert decision.size_adjustments == []
        assert decision.adjusted_dollars == 0.0

    def test_rejection_carries_partial_market_cap_adjustment(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        risk_manager._positions["MKT-001"] = 150.0
        _go_stale(risk_manager)

        decision = risk_manager.check_signal(make_signal(suggested_dollars=100.0))

        assert not decision.approved
        assert _reasons_kinds(decision.rejection_reasons) == ["STALE_DATA"]
        assert decision.size_adjustments == [
            "MARKET_CAP: reduced $100.00 → $50.00 (existing exposure $150.00)",
        ]
        assert decision.adjusted_dollars == 0.0

    def test_rejection_still_carries_drawdown_adjustment_and_warning(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _enter_drawdown(risk_manager)
        _go_stale(risk_manager)

        with caplog.at_level(logging.WARNING, logger="python.risk.manager"):
            decision = risk_manager.check_signal(make_signal(suggested_dollars=150.0))

        assert not decision.approved
        assert _reasons_kinds(decision.rejection_reasons) == ["STALE_DATA"]
        assert decision.size_adjustments == [
            "DRAWDOWN: reduced $150.00 → $100.00 (drawdown 16.7% >= 15%)",
        ]
        assert decision.adjusted_dollars == 0.0
        assert any(
            r.levelno == logging.WARNING and "Drawdown" in r.getMessage() for r in caplog.records
        )


class TestStaleRejectStamp:
    def test_aged_data_branch_stamps(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = 5_000.0
        monkeypatch.setattr("python.risk.manager.time.monotonic", lambda: now)
        risk_manager._last_ws_data_time = now - 400.0

        decision = risk_manager.check_signal(make_signal())

        assert decision.rejection_reasons == ["STALE_DATA: data is 400.0s old (cutoff 300s)"]
        assert risk_manager._last_stale_reject_monotonic == now

    def test_never_connected_branch_stamps_in_non_shadow_mode(
        self,
        make_settings: Callable[..., Settings],
        make_signal: SignalFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rm = RiskManager(make_settings(execution_mode="paper"))
        now = 5_000.0
        monkeypatch.setattr("python.risk.manager.time.monotonic", lambda: now)
        rm._startup_monotonic = now - 400.0
        assert rm._last_ws_data_time == 0.0

        decision = rm.check_signal(make_signal())

        assert decision.rejection_reasons == [
            "STALE_DATA: WebSocket never connected (uptime 400s, no data received)"
        ]
        assert rm._last_stale_reject_monotonic == now

    def test_never_connected_within_grace_period_does_not_stamp(
        self, make_settings: Callable[..., Settings], make_signal: SignalFactory
    ) -> None:
        rm = RiskManager(make_settings(execution_mode="paper"))
        decision = rm.check_signal(make_signal())
        assert decision.approved
        assert rm._last_stale_reject_monotonic is None

    def test_shadow_mode_without_ws_data_does_not_stamp(
        self,
        risk_manager: RiskManager,
        make_signal: SignalFactory,
    ) -> None:
        risk_manager._startup_monotonic = time.monotonic() - 10_000.0
        decision = risk_manager.check_signal(make_signal())
        assert decision.approved
        assert risk_manager._last_stale_reject_monotonic is None

    def test_fresh_data_does_not_stamp(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        risk_manager.update_data_freshness()
        decision = risk_manager.check_signal(make_signal())
        assert decision.approved
        assert risk_manager._last_stale_reject_monotonic is None

    def test_non_stale_soft_block_does_not_stamp(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        risk_manager.update_data_freshness()
        tracker = risk_manager._get_tracker("weather_edge")
        tracker.enabled = False
        tracker.disabled_reason = "test"
        decision = risk_manager.check_signal(make_signal())
        assert _reasons_kinds(decision.rejection_reasons) == ["STRATEGY_DISABLED"]
        assert risk_manager._last_stale_reject_monotonic is None

    def test_stamp_set_even_when_other_reasons_present(
        self, risk_manager: RiskManager, make_signal: SignalFactory
    ) -> None:
        _go_stale(risk_manager)
        risk_manager._positions["MKT-001"] = OVER_MARKET_CAP
        decision = risk_manager.check_signal(make_signal())
        assert _reasons_kinds(decision.rejection_reasons) == ["MARKET_CAP", "STALE_DATA"]
        assert risk_manager._last_stale_reject_monotonic is not None
