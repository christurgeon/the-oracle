"""Tests for WebSocket health check and auto-restart logic (Fix #13)."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python.notifications.base import AlertLevel
from python.orchestrator import Orchestrator, _pipe_health_window

pytestmark = pytest.mark.usefixtures("monotonic_past_boot")


class TestPipeHealthWindow:
    def test_default_is_clamped_to_floor(self):
        # idle=90: base=180, floor=90+120+15=225, ceil=300-15=285 -> 225
        assert _pipe_health_window(90, 300) == 225.0

    def test_never_below_one_slow_reconnect_cycle(self):
        # idle=10: base=20, but one reconnect can take idle+backoff=130 -> floor wins
        assert _pipe_health_window(10, 300) >= 10 + 120

    def test_never_above_stale_cutoff_margin(self):
        # idle=149 (max the validator allows): base=298 -> clamped under ceil
        assert _pipe_health_window(149, 300) <= 300 - 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator() -> Orchestrator:
    """Return a minimal Orchestrator with mocked risk manager and engine."""
    orch = Orchestrator()
    orch._running = True

    risk_mgr = MagicMock()
    risk_mgr.bankroll = 10_000.0
    risk_mgr.book_for.side_effect = lambda _name: risk_mgr.bankroll
    risk_mgr.is_halted = False
    risk_mgr.daily_pnl = 0.0
    risk_mgr.check_circuit_breakers.return_value = []
    risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
    risk_mgr.state_dirty = False
    risk_mgr.stale_data_rejected_within.return_value = False  # default: no harm
    orch._risk_manager = risk_mgr

    engine = MagicMock()
    engine.settle_expired_positions = AsyncMock(return_value=[])
    engine.check_all_pending_fills = AsyncMock(return_value=([], []))
    engine.execute_batch = AsyncMock(return_value=[])
    orch._engine = engine

    orch._strategies = []

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.execution.websocket_idle_timeout_seconds = 90
    settings.risk.stale_data_cutoff_seconds = 300
    orch._settings = settings

    return orch


def _make_done_task(exc: Exception | None = None) -> asyncio.Task:  # type: ignore[type-arg]
    """Return a completed (done) asyncio.Task that optionally stores an exception."""
    task: asyncio.Task = MagicMock(spec=asyncio.Task)
    task.done.return_value = True
    if exc is not None:
        task.exception.return_value = exc
    else:
        task.exception.return_value = None
    return task


def _make_running_task() -> asyncio.Task:  # type: ignore[type-arg]
    """Return a running (not-done) asyncio.Task mock."""
    task: asyncio.Task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    return task


def _noop_create_task(coro, **kwargs):  # type: ignore[no-untyped-def]
    """
    Drop-in replacement for asyncio.create_task used in tests.

    The real create_task schedules a coroutine on the event loop.  When we
    patch it with a plain MagicMock the coroutine object is never awaited,
    which triggers RuntimeWarning. Closing the coroutine before returning
    a mock task suppresses that warning without requiring a running loop.
    """
    coro.close()
    mock_task: asyncio.Task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    return mock_task


# ---------------------------------------------------------------------------
# TestWsHealthCheckNoWebSocket
# ---------------------------------------------------------------------------


class TestWsHealthCheckNoWebSocket:
    """When no WebSocket is configured, health check is a no-op."""

    @pytest.mark.asyncio
    async def test_no_ws_configured_is_noop(self) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = None
        orch._ws_task = None

        await orch._check_ws_health()

        # Nothing should have changed.
        assert orch._ws_task is None
        assert orch._ws_restart_count == 0


# ---------------------------------------------------------------------------
# TestWsHealthCheckHealthyTask
# ---------------------------------------------------------------------------


class TestWsHealthCheckHealthyTask:
    """When the task is alive, the health check resets the failure counter."""

    @staticmethod
    def _fresh_ws() -> MagicMock:
        """A WS mock whose stream is fresh (recent data frame, under cutoff)."""
        ws = MagicMock()
        ws.received_any_message = True
        ws.last_message_age = 0.0
        return ws

    @pytest.mark.asyncio
    async def test_healthy_task_resets_counter(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        orch._kalshi_ws = self._fresh_ws()
        orch._ws_task = _make_running_task()
        orch._ws_restart_count = 3  # pretend some previous failures

        await orch._check_ws_health()

        # Counter should be reset; no new task created.
        assert orch._ws_restart_count == 0
        assert orch._ws_task.done() is False  # still the same mock

    @pytest.mark.asyncio
    async def test_healthy_task_does_not_restart(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        orch._kalshi_ws = self._fresh_ws()
        original_task = _make_running_task()
        orch._ws_task = original_task

        with patch("asyncio.create_task") as mock_create:
            await orch._check_ws_health()

        mock_create.assert_not_called()
        assert orch._ws_task is original_task


# ---------------------------------------------------------------------------
# TestWsHealthCheckDeadTask
# ---------------------------------------------------------------------------


class TestWsHealthCheckDeadTask:
    """When the task has exited, the health check spawns a new one."""

    @pytest.mark.asyncio
    async def test_done_task_increments_counter(self) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_task = _make_done_task()
        orch._ws_restart_count = 0

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            await orch._check_ws_health()

        assert orch._ws_restart_count == 1

    @pytest.mark.asyncio
    async def test_done_task_spawns_new_task(self) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_task = _make_done_task()

        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()

        mock_create.assert_called_once()
        # _ws_task is now the mock task returned by _noop_create_task.
        assert orch._ws_task is not None
        assert orch._ws_task.done() is False

    @pytest.mark.asyncio
    async def test_done_task_with_exception_logs_error(self, caplog) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        exc = RuntimeError("connection reset")
        orch._ws_task = _make_done_task(exc=exc)

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            with caplog.at_level("ERROR", logger="python.orchestrator"):
                await orch._check_ws_health()

        assert any("connection reset" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_done_task_no_exception_logs_warning(self, caplog) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_task = _make_done_task(exc=None)

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            with caplog.at_level("WARNING", logger="python.orchestrator"):
                await orch._check_ws_health()

        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("cleanly" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_none_task_with_ws_restarts(self) -> None:
        """_ws_task=None but _kalshi_ws set means WS was never started; restart it."""
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_task = None

        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()

        mock_create.assert_called_once()
        assert orch._ws_task is not None
        assert orch._ws_restart_count == 1


# ---------------------------------------------------------------------------
# TestWsHealthCheckCriticalThreshold
# ---------------------------------------------------------------------------


class TestWsHealthCheckCriticalThreshold:
    """After _WS_MAX_RESTARTS consecutive failures a CRITICAL message is emitted."""

    @pytest.mark.asyncio
    async def test_critical_logged_after_max_restarts(self, caplog) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        # Simulate that we've already hit the max restart count.
        orch._ws_restart_count = Orchestrator._WS_MAX_RESTARTS
        orch._ws_task = _make_done_task()

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            with caplog.at_level("CRITICAL", logger="python.orchestrator"):
                await orch._check_ws_health()

        critical_msgs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
        assert any("consecutive" in m for m in critical_msgs)

    @pytest.mark.asyncio
    async def test_critical_not_logged_below_max_restarts(self, caplog) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_restart_count = 0
        orch._ws_task = _make_done_task()

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            with caplog.at_level("CRITICAL", logger="python.orchestrator"):
                await orch._check_ws_health()

        critical_msgs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
        assert not critical_msgs


# ---------------------------------------------------------------------------
# TestWsHealthCheckInTick
# ---------------------------------------------------------------------------


class TestWsHealthCheckInTick:
    """Verify _check_ws_health is called from _tick before signal generation."""

    @pytest.mark.asyncio
    async def test_tick_calls_ws_health_check(self) -> None:
        orch = _make_orchestrator()
        orch._kalshi_ws = None  # no WS — health check is a fast no-op

        health_called: list[bool] = []
        original_health = orch._check_ws_health

        async def tracking_health() -> None:
            health_called.append(True)
            await original_health()

        orch._check_ws_health = tracking_health  # type: ignore[method-assign]

        await orch._tick()

        assert health_called == [True], "_check_ws_health was not called during _tick"

    @pytest.mark.asyncio
    async def test_tick_calls_ws_health_before_signal_generation(self) -> None:
        """Health check must run before generate_signals to ensure fresh data."""
        from python.strategies.base import TradeSignal

        orch = _make_orchestrator()
        orch._kalshi_ws = None
        call_order: list[str] = []

        original_health = orch._check_ws_health

        async def tracking_health() -> None:
            call_order.append("health")
            await original_health()

        orch._check_ws_health = tracking_health  # type: ignore[method-assign]

        from tests.test_bankroll_sync import StubStrategy

        strategy = StubStrategy()
        signal_calls: list[str] = []

        async def tracking_signals() -> list[TradeSignal]:
            signal_calls.append("signals")
            call_order.append("signals")
            return []

        strategy.generate_signals = tracking_signals  # type: ignore[method-assign]
        orch._strategies = [strategy]

        await orch._tick()

        health_idx = call_order.index("health")
        signals_idx = call_order.index("signals")
        assert health_idx < signals_idx, "health check must run before signal generation"

    @pytest.mark.asyncio
    async def test_dead_ws_task_restarted_during_tick(self) -> None:
        """A dead _ws_task is replaced by a new one during _tick."""
        orch = _make_orchestrator()
        orch._kalshi_ws = MagicMock()
        orch._ws_task = _make_done_task()

        with patch("asyncio.create_task", side_effect=_noop_create_task):
            await orch._tick()

        assert orch._ws_task is not None
        assert orch._ws_task.done() is False
        assert orch._ws_restart_count == 1


# ---------------------------------------------------------------------------
# Subscribe shape — production regression guard. The orchestrator's
# _ws_listener must NOT subscribe to ``orderbook_delta`` without a
# ``market_tickers`` filter; Kalshi rejects with code=2 'Params required'
# (surfaced by the ACK reader as SubscribeFailed).
# ---------------------------------------------------------------------------


class TestWsListenerSubscribeShape:
    @pytest.mark.asyncio
    async def test_ws_listener_does_not_request_orderbook_delta_without_tickers(
        self,
    ) -> None:
        """Reproduces the production SubscribeFailed: ['orderbook_delta', 'ticker']
        with tickers=None. The orchestrator must not request orderbook_delta
        unless it is going to pass a market_tickers filter (which it currently
        cannot — there's no plumbing).
        """
        orch = _make_orchestrator()

        captured: dict[str, Any] = {}

        async def _fake_listen(
            *, channels: list[str], market_tickers: list[str] | None = None
        ):
            captured["channels"] = list(channels)
            captured["market_tickers"] = market_tickers
            # Async generator returning nothing.
            if False:
                yield  # pragma: no cover
            return

        kalshi_ws = MagicMock()
        kalshi_ws.listen = _fake_listen
        orch._kalshi_ws = kalshi_ws

        await orch._ws_listener()

        assert "channels" in captured, "_ws_listener should invoke listen()"
        channels = captured["channels"]
        market_tickers = captured["market_tickers"]
        # The actual contract: orderbook_delta requires market_tickers.
        if "orderbook_delta" in channels:
            assert market_tickers, (
                "orderbook_delta subscribe MUST include a non-empty "
                "market_tickers filter or Kalshi rejects with "
                "code=2 'Params required'"
            )


# ---------------------------------------------------------------------------
# TestWsHealthCheckStalledButAlive — silent half-open stall (2026-07-11 fix)
# ---------------------------------------------------------------------------


class TestWsHealthCheckStalledButAlive:
    """A live-but-silent listener must be force-restarted and alerted on."""

    @pytest.mark.asyncio
    async def test_stalled_stream_forces_restart_and_alerts(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        ws = MagicMock()
        ws.received_any_message = True
        ws.last_message_age = 999.0  # >> cutoff
        ws.last_subscribe_ok_age = 999.0  # pipe cannot re-subscribe -> pipe-down
        orch._kalshi_ws = ws
        orch._ws_task = _make_running_task()
        orch._ws_restart_count = orch._WS_ALERT_THRESHOLD - 1  # next tick crosses threshold
        orch._ws_last_alerted_count = 0
        orch._ws_last_critical_alert_monotonic = 0.0  # throttle open
        orch._notify_alert = AsyncMock()

        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()

        assert orch._ws_restart_count == orch._WS_ALERT_THRESHOLD
        mock_create.assert_called_once()  # respawned the listener
        orch._notify_alert.assert_awaited_once()
        assert orch._notify_alert.await_args.kwargs["level"] == AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_cold_start_no_message_past_cutoff_is_stale(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        ws = MagicMock()
        ws.received_any_message = False
        ws.last_message_age = 0.0
        orch._kalshi_ws = ws
        orch._startup_monotonic = time.monotonic() - 400.0  # 400s uptime, never a msg
        assert orch._ws_stream_stale_seconds() is not None

    @pytest.mark.asyncio
    async def test_cold_start_within_grace_not_stale(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        ws = MagicMock()
        ws.received_any_message = False
        ws.last_message_age = 0.0
        orch._kalshi_ws = ws
        orch._startup_monotonic = time.monotonic() - 10.0  # only 10s in — still starting up
        assert orch._ws_stream_stale_seconds() is None

    @pytest.mark.asyncio
    async def test_fresh_stream_not_stale_no_restart(self) -> None:
        orch = _make_orchestrator()
        orch._settings.risk.stale_data_cutoff_seconds = 300
        ws = MagicMock()
        ws.received_any_message = True
        ws.last_message_age = 5.0
        orch._kalshi_ws = ws
        orch._ws_task = _make_running_task()
        orch._ws_restart_count = 2

        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()

        assert orch._ws_stream_stale_seconds() is None
        mock_create.assert_not_called()
        assert orch._ws_restart_count == 0  # healthy tick reset the counter

# ---------------------------------------------------------------------------
# TestWsHealthImpactGated — impact-gated alerting (spec 2026-07-23)
# ---------------------------------------------------------------------------


class TestWsHealthImpactGated:
    def _silent_alive_orch(self) -> Orchestrator:
        orch = _make_orchestrator()
        ws = MagicMock()
        ws.received_any_message = True
        ws.last_message_age = 999.0  # data-silent (>> cutoff)
        ws.last_subscribe_ok_age = 5.0  # pipe healthy (just re-subscribed)
        orch._kalshi_ws = ws
        orch._ws_task = _make_running_task()
        orch._ws_restart_count = 0
        orch._ws_last_critical_alert_monotonic = 0.0
        orch._ws_last_silent_info_monotonic = 0.0
        orch._notify_alert = AsyncMock()
        return orch

    @pytest.mark.asyncio
    async def test_silent_but_healthy_does_not_page_or_respawn(self) -> None:
        orch = self._silent_alive_orch()  # pipe healthy, harm=False
        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()
        orch._notify_alert.assert_not_awaited()
        mock_create.assert_not_called()
        assert orch._ws_restart_count == 0

    @pytest.mark.asyncio
    async def test_harm_pages_without_respawn(self) -> None:
        orch = self._silent_alive_orch()
        orch._risk_manager.stale_data_rejected_within.return_value = True
        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()
        orch._notify_alert.assert_awaited_once()
        assert orch._notify_alert.await_args.kwargs["level"] == AlertLevel.CRITICAL
        mock_create.assert_not_called()  # pipe healthy -> inner loop owns reconnect

    @pytest.mark.asyncio
    async def test_pipe_down_pages_and_respawns(self) -> None:
        orch = self._silent_alive_orch()
        orch._kalshi_ws.last_subscribe_ok_age = 999.0  # cannot re-subscribe
        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()
        orch._notify_alert.assert_awaited_once()
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_never_subscribed_is_pipe_down(self) -> None:
        orch = self._silent_alive_orch()
        orch._kalshi_ws.last_subscribe_ok_age = None  # cold start, never subscribed
        with patch("asyncio.create_task", side_effect=_noop_create_task) as mock_create:
            await orch._check_ws_health()  # must NOT raise TypeError
        orch._notify_alert.assert_awaited_once()
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_critical_alert_is_time_throttled(self) -> None:
        orch = self._silent_alive_orch()
        orch._risk_manager.stale_data_rejected_within.return_value = True
        orch._ws_last_critical_alert_monotonic = time.monotonic()  # just alerted
        with patch("asyncio.create_task", side_effect=_noop_create_task):
            await orch._check_ws_health()
        orch._notify_alert.assert_not_awaited()  # inside throttle window

    @pytest.mark.asyncio
    async def test_watchdog_never_raises_into_tick(self) -> None:
        orch = self._silent_alive_orch()
        # Force an error deep in the classification (bad settings type).
        orch._settings.execution.websocket_idle_timeout_seconds = object()
        with patch("asyncio.create_task", side_effect=_noop_create_task):
            await orch._check_ws_health()  # swallowed, no exception propagates
