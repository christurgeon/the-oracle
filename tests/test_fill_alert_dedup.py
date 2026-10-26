"""Telegram fill alerts must fire for later-tick, degraded-recovery, and
cancel-race fills, not only synchronous ``execute_batch`` fills.

All fill announcements route through a single per-tick scan of the engine's
order book (``collect_unalerted_fills``), deduped by an on-record
``last_alerted_fill_count`` counter so a resting partial polled every tick is
announced exactly once per new fill quantity.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.execution.engine import (
    ExecutionEngine,
    OrderRecord,
    is_unalerted_new_fill,
)
from python.risk.manager import RiskDecision
from python.strategies.base import TradeSignal


def _record(
    *,
    status: OrderStatus,
    fill_count: int,
    last_alerted: int = -1,
    ticker: str = "KXHIGHNY-99APR14-B70.5",
) -> OrderRecord:
    signal = TradeSignal(
        market_id=ticker,
        market_ticker=ticker,
        model_probability=0.65,
        market_price=0.50,
        confidence=0.8,
        strategy_name="favorite_bias",
        side=Side.YES,
        suggested_dollars=100.0,
        suggested_contracts=10,
        metadata={},
    )
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=10.0)
    return OrderRecord(
        order_id=f"ORD-{ticker}",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.50,
        order_price=0.50,
        fill_count=fill_count,
        requested_count=10,
        status=status,
        last_alerted_fill_count=last_alerted,
    )


# ---------------------------------------------------------------------------
# Pure predicate: is_unalerted_new_fill
# ---------------------------------------------------------------------------


class TestIsUnalertedNewFill:
    def test_default_field_is_sentinel_minus_one(self) -> None:
        r = _record(status=OrderStatus.FILLED, fill_count=5)
        # Constructed WITHOUT overriding: default must be -1 so a first alert
        # always fires (including the degraded FILLED/0 case).
        r2 = OrderRecord(
            order_id="X",
            signal=r.signal,
            risk_decision=r.risk_decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.5,
            order_price=0.5,
        )
        assert r2.last_alerted_fill_count == -1

    def test_fresh_full_fill_is_unalerted(self) -> None:
        assert is_unalerted_new_fill(_record(status=OrderStatus.FILLED, fill_count=10))

    def test_fresh_partial_is_unalerted(self) -> None:
        assert is_unalerted_new_fill(_record(status=OrderStatus.PARTIAL, fill_count=4))

    def test_shadow_fill_is_unalerted(self) -> None:
        assert is_unalerted_new_fill(_record(status=OrderStatus.SHADOW, fill_count=2))

    def test_degraded_filled_zero_alerts_once(self) -> None:
        """Synchronous immediate-fill recovery can leave FILLED/fill_count=0.
        The -1 sentinel means it announces once (matching pre-fix behaviour)."""
        assert is_unalerted_new_fill(
            _record(status=OrderStatus.FILLED, fill_count=0, last_alerted=-1)
        )

    def test_degraded_filled_zero_not_realerted_after_stamp(self) -> None:
        assert not is_unalerted_new_fill(
            _record(status=OrderStatus.FILLED, fill_count=0, last_alerted=0)
        )

    def test_resting_partial_same_count_not_realerted(self) -> None:
        """A resting PARTIAL re-polled each tick with no new fills must not
        re-announce."""
        assert not is_unalerted_new_fill(
            _record(status=OrderStatus.PARTIAL, fill_count=4, last_alerted=4)
        )

    def test_partial_progress_realerts_on_new_count(self) -> None:
        assert is_unalerted_new_fill(
            _record(status=OrderStatus.PARTIAL, fill_count=7, last_alerted=4)
        )

    def test_expired_with_partial_fill_is_unalerted(self) -> None:
        """TTL cancel-race partial: status EXPIRED but contracts were acquired."""
        assert is_unalerted_new_fill(
            _record(status=OrderStatus.EXPIRED, fill_count=6, last_alerted=-1)
        )

    def test_expired_zero_fill_is_not_alertable(self) -> None:
        assert not is_unalerted_new_fill(
            _record(status=OrderStatus.EXPIRED, fill_count=0, last_alerted=-1)
        )

    def test_cancelled_with_partial_fill_is_unalerted(self) -> None:
        assert is_unalerted_new_fill(
            _record(status=OrderStatus.CANCELLED, fill_count=3, last_alerted=-1)
        )

    def test_expired_partial_not_realerted_after_stamp(self) -> None:
        assert not is_unalerted_new_fill(
            _record(status=OrderStatus.EXPIRED, fill_count=6, last_alerted=6)
        )

    @pytest.mark.parametrize(
        "status",
        [
            OrderStatus.REJECTED,
            OrderStatus.SKIPPED,
            OrderStatus.ERROR,
            OrderStatus.RESTING,
            OrderStatus.PENDING,
            OrderStatus.HYDRATED,
        ],
    )
    def test_non_fill_states_never_alert(self, status: OrderStatus) -> None:
        assert not is_unalerted_new_fill(_record(status=status, fill_count=0))


# ---------------------------------------------------------------------------
# Engine scan: collect_unalerted_fills
# ---------------------------------------------------------------------------


def _make_settings(tmp_path, mode: str = "live"):
    from python.settings import Settings

    s = Settings()
    s.execution.mode = mode
    s.state_dir = str(tmp_path)
    return s


class TestCollectUnalertedFills:
    def _engine(self, tmp_path) -> ExecutionEngine:
        settings = _make_settings(tmp_path, mode="live")
        risk = MagicMock()
        return ExecutionEngine(settings, kalshi_client=MagicMock(), risk_manager=risk)

    def test_returns_only_unalerted_fills(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        filled = _record(status=OrderStatus.FILLED, fill_count=10, ticker="A")
        rejected = _record(status=OrderStatus.REJECTED, fill_count=0, ticker="B")
        already = _record(status=OrderStatus.PARTIAL, fill_count=4, last_alerted=4, ticker="C")
        for r in (filled, rejected, already):
            engine._orders[r.order_id] = r

        out = engine.collect_unalerted_fills()
        assert filled in out
        assert rejected not in out
        assert already not in out

    def test_empty_when_no_orders(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        assert engine.collect_unalerted_fills() == []

    def test_hydrated_shadow_positions_are_not_announced(self, tmp_path) -> None:
        """Restart replays open shadow positions as SHADOW/fill_count=1 stubs;
        they must start already-alerted so the scan does not re-announce a
        days-old position as a fresh fill."""
        from datetime import date

        from python.execution.shadow_broker import ShadowPosition

        settings = _make_settings(tmp_path, mode="shadow")
        risk = MagicMock()
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)
        pos = ShadowPosition(
            position_key="KXHIGHNY-99APR14-B70.5:yes",
            market_ticker="KXHIGHNY-99APR14-B70.5",
            market_id="KXHIGHNY-99APR14-B70.5",
            side=Side.YES,
            dollars=50.0,
            strategy_name="weather_edge",
            close_time="",
            ticker_date=date(2099, 4, 14),
            order_id="shadow-old-1",
        )
        engine.hydrate_shadow_positions([pos])
        assert engine.collect_unalerted_fills() == []


# ---------------------------------------------------------------------------
# Orchestrator helper: _send_fill_alerts (stamp / dedup / failure handling)
# ---------------------------------------------------------------------------


def _make_orch_with_notifier():
    """Orchestrator whose notifier echoes back every record as 'announced'."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator()
    notifier = MagicMock()
    # send_trade_batch_alert returns the records it actually announced.
    notifier.send_trade_batch_alert = AsyncMock(side_effect=lambda recs: list(recs))
    orch._notifier = notifier
    return orch, notifier


class TestSendFillAlerts:
    @pytest.mark.asyncio
    async def test_stamps_last_alerted_on_success(self) -> None:
        orch, notifier = _make_orch_with_notifier()
        rec = _record(status=OrderStatus.FILLED, fill_count=10)
        await orch._send_fill_alerts([rec])
        notifier.send_trade_batch_alert.assert_awaited_once_with([rec])
        assert rec.last_alerted_fill_count == 10

    @pytest.mark.asyncio
    async def test_does_not_stamp_when_send_raises(self) -> None:
        """A transient notifier exception must NOT stamp — so the next tick's
        scan retries the announcement."""
        orch, notifier = _make_orch_with_notifier()
        notifier.send_trade_batch_alert = AsyncMock(side_effect=RuntimeError("boom"))
        rec = _record(status=OrderStatus.FILLED, fill_count=10)
        await orch._send_fill_alerts([rec])  # must not raise
        assert rec.last_alerted_fill_count == -1

    @pytest.mark.asyncio
    async def test_does_not_stamp_when_send_dropped(self) -> None:
        """A rate-limited / failed send announces nothing (returns []), so no
        record is stamped and the next tick's scan retries."""
        orch, notifier = _make_orch_with_notifier()
        notifier.send_trade_batch_alert = AsyncMock(return_value=[])
        rec = _record(status=OrderStatus.FILLED, fill_count=10)
        await orch._send_fill_alerts([rec])
        assert rec.last_alerted_fill_count == -1

    @pytest.mark.asyncio
    async def test_stamps_only_announced_subset(self) -> None:
        """Only the records the notifier reports as announced are stamped."""
        orch, notifier = _make_orch_with_notifier()
        sent = _record(status=OrderStatus.FILLED, fill_count=10, ticker="SENT")
        dropped = _record(status=OrderStatus.FILLED, fill_count=5, ticker="DROP")
        notifier.send_trade_batch_alert = AsyncMock(return_value=[sent])
        await orch._send_fill_alerts([sent, dropped])
        assert sent.last_alerted_fill_count == 10
        assert dropped.last_alerted_fill_count == -1

    @pytest.mark.asyncio
    async def test_no_notifier_is_noop(self) -> None:
        from python.orchestrator import Orchestrator

        orch = Orchestrator()
        orch._notifier = None
        rec = _record(status=OrderStatus.FILLED, fill_count=10)
        await orch._send_fill_alerts([rec])  # must not raise
        assert rec.last_alerted_fill_count == -1

    @pytest.mark.asyncio
    async def test_empty_records_no_send(self) -> None:
        orch, notifier = _make_orch_with_notifier()
        await orch._send_fill_alerts([])
        notifier.send_trade_batch_alert.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tick wiring: _tick scans the engine and announces later-tick fills
# ---------------------------------------------------------------------------


class TestTickAnnouncesLaterTickFills:
    def _make_orch(self):
        from python.orchestrator import Orchestrator

        orch = Orchestrator()
        orch._running = True

        risk_mgr = MagicMock()
        risk_mgr.bankroll = 10_000.0
        risk_mgr.is_halted = False
        risk_mgr.daily_pnl = 0.0
        risk_mgr.check_circuit_breakers.return_value = []
        risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
        risk_mgr.state_dirty = False
        orch._risk_manager = risk_mgr

        engine = MagicMock()
        engine.settle_expired_positions = AsyncMock(return_value=[])
        engine.check_all_pending_fills = AsyncMock(return_value=([], []))
        engine.execute_batch = AsyncMock(return_value=[])
        orch._engine = engine

        notifier = MagicMock()
        notifier.send_trade_batch_alert = AsyncMock(side_effect=lambda recs: list(recs))
        orch._notifier = notifier

        settings = MagicMock()
        settings.is_kill_switch_active.return_value = False
        settings.is_shadow = True
        # Real float: _tick compares it during ET market hours, and a MagicMock
        # raises TypeError there, which _tick swallows.
        settings.telegram.silent_bot_alert_hours = 6.0
        orch._settings = settings

        class _NoSignalStrategy:
            name = "favorite_bias"
            enabled = True
            # Duck-typed stub, not a real Strategy subclass — Task 9's
            # per-strategy liveness check reads configured_to_trade on every
            # strategy every tick, so this stub needs it too.
            configured_to_trade = True
            bankroll = 10_000.0

            def update_bankroll(self, *_a, **_k) -> None:
                pass

            async def generate_signals(self) -> list[Any]:
                return []

        orch._strategies = [_NoSignalStrategy()]
        return orch, engine, notifier

    @pytest.mark.asyncio
    async def test_later_tick_fill_is_announced_and_stamped(self) -> None:
        orch, engine, notifier = self._make_orch()
        rec = _record(status=OrderStatus.FILLED, fill_count=10)
        engine.collect_unalerted_fills = MagicMock(return_value=[rec])

        await orch._tick()

        notifier.send_trade_batch_alert.assert_awaited_once_with([rec])
        assert rec.last_alerted_fill_count == 10

    @pytest.mark.asyncio
    async def test_no_unalerted_fills_no_announcement(self) -> None:
        orch, engine, notifier = self._make_orch()
        engine.collect_unalerted_fills = MagicMock(return_value=[])

        await orch._tick()

        notifier.send_trade_batch_alert.assert_not_awaited()
