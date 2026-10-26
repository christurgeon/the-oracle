"""Tests for bankroll synchronization between risk manager and strategies (Fix #8)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.sizing import fractional_kelly
from python.strategies.base import Strategy, TradeSignal

# ---------------------------------------------------------------------------
# Concrete strategy stub for testing the ABC
# ---------------------------------------------------------------------------


class StubStrategy(Strategy):
    """Minimal concrete implementation of the Strategy ABC for testing."""

    def __init__(self, *, bankroll: float = 10_000.0) -> None:
        super().__init__(name="stub", config={"enabled": True}, bankroll=bankroll)

    async def scan(self) -> list[dict[str, Any]]:
        return []

    async def evaluate(self, market: dict[str, Any]) -> float | None:
        return None

    async def generate_signals(self) -> list[TradeSignal]:
        return []


# ---------------------------------------------------------------------------
# Tests: Strategy ABC bankroll
# ---------------------------------------------------------------------------


class TestStrategyBankroll:
    """Verify bankroll property and update method on the Strategy ABC."""

    def test_initial_bankroll(self) -> None:
        s = StubStrategy(bankroll=10_000.0)
        assert s.bankroll == 10_000.0

    def test_default_bankroll(self) -> None:
        s = StubStrategy()
        assert s.bankroll == 10_000.0

    def test_update_bankroll(self) -> None:
        s = StubStrategy(bankroll=10_000.0)
        s.update_bankroll(8_000.0)
        assert s.bankroll == 8_000.0

    def test_update_bankroll_idempotent(self) -> None:
        s = StubStrategy(bankroll=5_000.0)
        s.update_bankroll(5_000.0)
        assert s.bankroll == 5_000.0

    def test_update_bankroll_clamps_negative(self) -> None:
        s = StubStrategy(bankroll=10_000.0)
        s.update_bankroll(-500.0)
        assert s.bankroll == 0.0

    def test_update_bankroll_zero(self) -> None:
        s = StubStrategy(bankroll=10_000.0)
        s.update_bankroll(0.0)
        assert s.bankroll == 0.0

    def test_bankroll_accessible_via_private_attr(self) -> None:
        """Strategies use self._bankroll directly in Kelly calls."""
        s = StubStrategy(bankroll=10_000.0)
        s.update_bankroll(7_500.0)
        assert s._bankroll == 7_500.0  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tests: Kelly sizing reflects bankroll changes
# ---------------------------------------------------------------------------


class TestKellySizingWithBankroll:
    """Verify that updated bankroll flows through to position sizing."""

    def test_kelly_sizing_scales_with_bankroll(self) -> None:
        result_10k = fractional_kelly(
            model_prob=0.65,
            market_price=0.50,
            kelly_fraction=0.15,
            max_pct=0.05,
            bankroll=10_000.0,
        )
        result_8k = fractional_kelly(
            model_prob=0.65,
            market_price=0.50,
            kelly_fraction=0.15,
            max_pct=0.05,
            bankroll=8_000.0,
        )
        assert result_10k.dollars > 0
        assert result_8k.dollars > 0
        assert result_8k.dollars == pytest.approx(result_10k.dollars * 0.8)

    def test_kelly_sizing_zero_bankroll(self) -> None:
        result = fractional_kelly(
            model_prob=0.65,
            market_price=0.50,
            kelly_fraction=0.15,
            max_pct=0.05,
            bankroll=0.0,
        )
        assert result.dollars == 0.0


# ---------------------------------------------------------------------------
# Tests: Orchestrator bankroll sync
# ---------------------------------------------------------------------------


class TestOrchestratorBankrollSync:
    """Verify the orchestrator syncs bankroll to strategies each tick."""

    @pytest.mark.asyncio
    async def test_tick_syncs_bankroll_to_strategies(self) -> None:
        """After _tick(), strategies should have the risk manager's bankroll."""
        from python.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        orchestrator._running = True

        # Mock risk manager with drawdown
        risk_mgr = MagicMock()
        risk_mgr.bankroll = 8_000.0
        risk_mgr.book_for.side_effect = lambda _name: risk_mgr.bankroll
        risk_mgr.is_halted = False
        risk_mgr.daily_pnl = 0.0
        risk_mgr.check_circuit_breakers.return_value = []
        risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
        risk_mgr.state_dirty = False
        orchestrator._risk_manager = risk_mgr

        # Mock engine
        engine = MagicMock()
        engine.settle_expired_positions = AsyncMock(return_value=[])
        engine.check_all_pending_fills = AsyncMock(return_value=([], []))
        engine.execute_batch = AsyncMock(return_value=[])
        orchestrator._engine = engine

        # Create a stub strategy
        strategy = StubStrategy(bankroll=10_000.0)
        strategy.generate_signals = AsyncMock(return_value=[])  # type: ignore[method-assign]
        orchestrator._strategies = [strategy]

        # Mock kill switch + date
        orchestrator._settings = MagicMock()
        orchestrator._settings.is_kill_switch_active.return_value = False
        orchestrator._settings.is_shadow = True

        await orchestrator._tick()

        assert strategy.bankroll == 8_000.0

    @pytest.mark.asyncio
    async def test_bankroll_synced_before_signal_generation(self) -> None:
        """Bankroll should be updated before generate_signals() runs."""
        from python.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        orchestrator._running = True

        risk_mgr = MagicMock()
        risk_mgr.bankroll = 7_500.0
        risk_mgr.book_for.side_effect = lambda _name: risk_mgr.bankroll
        risk_mgr.is_halted = False
        risk_mgr.daily_pnl = 0.0
        risk_mgr.check_circuit_breakers.return_value = []
        risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
        risk_mgr.state_dirty = False
        orchestrator._risk_manager = risk_mgr

        engine = MagicMock()
        engine.settle_expired_positions = AsyncMock(return_value=[])
        engine.check_all_pending_fills = AsyncMock(return_value=([], []))
        engine.execute_batch = AsyncMock(return_value=[])
        orchestrator._engine = engine

        strategy = StubStrategy(bankroll=10_000.0)
        bankroll_at_signal_time: list[float] = []

        async def capture_bankroll() -> list[TradeSignal]:
            bankroll_at_signal_time.append(strategy.bankroll)
            return []

        strategy.generate_signals = capture_bankroll  # type: ignore[method-assign]
        orchestrator._strategies = [strategy]

        orchestrator._settings = MagicMock()
        orchestrator._settings.is_kill_switch_active.return_value = False
        orchestrator._settings.is_shadow = True

        await orchestrator._tick()

        assert bankroll_at_signal_time == [7_500.0]


# ---------------------------------------------------------------------------
# Tests: Orchestrator kill-switch handler — cancel-truncation alert
# ---------------------------------------------------------------------------


class TestOrchestratorKillSwitchTruncation:
    """When the kill switch fires and cancel_all_open returns truncated=True,
    the orchestrator must fire a SECOND CRITICAL Telegram alert so the
    operator knows additional live orders may remain."""

    @pytest.mark.asyncio
    async def test_kill_switch_truncated_fires_second_critical_alert(self) -> None:
        from python.notifications.base import AlertLevel
        from python.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        orchestrator._running = True

        risk_mgr = MagicMock()
        risk_mgr.bankroll = 10_000.0
        orchestrator._risk_manager = risk_mgr

        engine = MagicMock()
        engine.cancel_all_open = AsyncMock(return_value=(["o1", "o2"], True))
        orchestrator._engine = engine

        orchestrator._settings = MagicMock()
        orchestrator._settings.is_kill_switch_active.return_value = True

        notifier = AsyncMock()
        orchestrator._notifier = notifier

        await orchestrator._tick()

        # Two alerts: kill-switch, then truncation.
        assert notifier.send_alert.call_count == 2

        first_call = notifier.send_alert.call_args_list[0]
        assert "KILL SWITCH activated" in first_call.args[0]
        assert first_call.kwargs["level"] is AlertLevel.CRITICAL

        second_call = notifier.send_alert.call_args_list[1]
        assert "ORDER CANCEL TRUNCATED" in second_call.args[0]
        assert "only 2 resting orders cancelled" in second_call.args[0]
        assert second_call.kwargs["level"] is AlertLevel.CRITICAL

        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_kill_switch_clean_cancel_fires_only_one_alert(self) -> None:
        from python.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        orchestrator._running = True

        risk_mgr = MagicMock()
        risk_mgr.bankroll = 10_000.0
        orchestrator._risk_manager = risk_mgr

        engine = MagicMock()
        engine.cancel_all_open = AsyncMock(return_value=(["o1"], False))
        orchestrator._engine = engine

        orchestrator._settings = MagicMock()
        orchestrator._settings.is_kill_switch_active.return_value = True

        notifier = AsyncMock()
        orchestrator._notifier = notifier

        await orchestrator._tick()

        # Only the original kill-switch alert; no truncation alert.
        assert notifier.send_alert.call_count == 1
        assert "KILL SWITCH activated" in notifier.send_alert.call_args.args[0]
        assert orchestrator._running is False
