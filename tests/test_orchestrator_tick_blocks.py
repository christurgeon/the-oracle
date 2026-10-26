"""Characterization tests for Orchestrator._tick blocks: day-roll,
settlement-loss alerts, daily-loss-halt edge-trigger, reconciliation
cadence, circuit-breaker alerts, EMOS reload, per-day-cap commit wiring,
SILENT BOT, and the tick-failure escalation ladder."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ET, ExecutionMode, OrderStatus, Side
from python.execution.engine import OrderRecord, SettlementResult
from python.notifications.base import AlertLevel
from python.orchestrator import Orchestrator
from python.risk.manager import RiskDecision
from python.strategies.base import TradeSignal
from tests.conftest import _stub

# ---------------------------------------------------------------------------
# Day-roll block (orchestrator.py: _reset_daily_with_fanout call site)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_day_roll_fires_fanout_and_resets_alert_flags(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._current_trading_date = orch._et_today() - timedelta(days=1)
    orch._was_halted = True
    orch._silent_alert_sent = True

    reset_calls: list[str] = []
    strategy = _stub("weather_edge")
    strategy.reset_daily_counters = lambda: reset_calls.append("reset")  # type: ignore[attr-defined]
    orch._strategies = [strategy]

    await orch._tick()

    assert reset_calls == ["reset"]
    assert orch._was_halted is False
    assert orch._silent_alert_sent is False
    assert orch._current_trading_date == orch._et_today()


@pytest.mark.asyncio
async def test_no_day_roll_when_same_trading_date(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._current_trading_date = orch._et_today()
    orch._was_halted = True

    reset_calls: list[str] = []
    strategy = _stub("weather_edge")
    strategy.reset_daily_counters = lambda: reset_calls.append("reset")  # type: ignore[attr-defined]
    orch._strategies = [strategy]

    await orch._tick()

    assert reset_calls == []  # the day-roll block itself did not fire


# ---------------------------------------------------------------------------
# Settlement-loss alerts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settlement_alerts_only_on_losing_settlements(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, risk_mgr, engine = tick_orch
    risk_mgr.bankroll = 9_500.0
    engine.settle_expired_positions = AsyncMock(
        return_value=[
            SettlementResult("k1", "MKT-1", "no", -25.0),
            SettlementResult("k2", "MKT-2", "yes", 10.0),
            SettlementResult("k3", "MKT-3", "expired", 0.0),
        ]
    )
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    warnings = [msg for msg, level in notifier.sent if level == AlertLevel.WARNING]  # type: ignore[attr-defined]
    assert len(warnings) == 1
    assert "MKT-1" in warnings[0]


# ---------------------------------------------------------------------------
# Daily-loss-halt edge-triggered alert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_loss_halt_alerts_once_on_transition(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, risk_mgr, _engine = tick_orch
    risk_mgr.is_halted = True
    risk_mgr.daily_pnl = -123.45
    orch._was_halted = False
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    criticals = [msg for msg, level in notifier.sent if level == AlertLevel.CRITICAL]  # type: ignore[attr-defined]
    assert any("DAILY LOSS HALT" in m for m in criticals)
    assert orch._was_halted is True


@pytest.mark.asyncio
async def test_daily_loss_halt_does_not_re_alert_while_still_halted(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, risk_mgr, _engine = tick_orch
    risk_mgr.is_halted = True
    risk_mgr.daily_pnl = -123.45
    orch._was_halted = True  # already alerted previously
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    criticals = [msg for msg, level in notifier.sent if level == AlertLevel.CRITICAL]  # type: ignore[attr-defined]
    assert not any("DAILY LOSS HALT" in m for m in criticals)


# ---------------------------------------------------------------------------
# Scheduler cadence: periodic reconciliation + fills reconciler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_periodic_reconciliation_fires_on_interval_and_stamps_tick(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._settings.execution.mode = "live"  # is_shadow False -> interval applies
    orch._settings.execution.reconcile_interval_ticks = 1
    orch._tick_count = 5
    orch._last_reconcile_tick = 4  # 6 - 4 >= 1 after increment

    called: list[int] = []

    async def fake_reconcile() -> None:
        called.append(1)

    orch._periodic_reconciliation = fake_reconcile  # type: ignore[method-assign]

    await orch._tick()

    assert called == [1]
    assert orch._last_reconcile_tick == orch._tick_count


@pytest.mark.asyncio
async def test_periodic_reconciliation_failure_is_swallowed(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._settings.execution.mode = "live"
    orch._settings.execution.reconcile_interval_ticks = 1
    orch._tick_count = 5
    orch._last_reconcile_tick = 4

    async def raising() -> None:
        raise RuntimeError("boom")

    orch._periodic_reconciliation = raising  # type: ignore[method-assign]

    await orch._tick()  # must not raise / must not trip the failure counter

    assert orch._consecutive_tick_failures == 0


@pytest.mark.asyncio
async def test_fills_reconciler_tick_fires_on_interval_and_stamps_tick(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._settings.fills_reconciler.reconcile_interval_ticks = 1
    orch._tick_count = 5
    orch._last_scheduled_fills_tick = 4

    reconciler = MagicMock()
    reconciler.tick = AsyncMock()
    orch._fills_reconciler = reconciler

    await orch._tick()

    reconciler.tick.assert_awaited_once()
    assert orch._last_scheduled_fills_tick == orch._tick_count


@pytest.mark.asyncio
async def test_fills_reconciler_tick_failure_is_swallowed(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._settings.fills_reconciler.reconcile_interval_ticks = 1
    orch._tick_count = 5
    orch._last_scheduled_fills_tick = 4

    reconciler = MagicMock()
    reconciler.tick = AsyncMock(side_effect=RuntimeError("boom"))
    orch._fills_reconciler = reconciler

    await orch._tick()

    assert orch._consecutive_tick_failures == 0


# ---------------------------------------------------------------------------
# Circuit-breaker alert dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_trip_sends_critical_alert_with_reason(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, risk_mgr, _engine = tick_orch
    risk_mgr.check_circuit_breakers.return_value = ["favorite_bias"]
    risk_mgr.get_tracker.return_value = MagicMock(
        enabled=False, disabled_reason="3 consecutive losses"
    )
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    criticals = [msg for msg, level in notifier.sent if level == AlertLevel.CRITICAL]  # type: ignore[attr-defined]
    assert any(
        "favorite_bias" in m and "3 consecutive losses" in m for m in criticals
    )


# ---------------------------------------------------------------------------
# EMOS hot-reload call site's OWN behavior (swap / no-swap / swallow)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emos_reload_swaps_table_on_new_version(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    old_table = MagicMock(name="old_table")
    new_table = MagicMock(name="new_table")
    old_table.check_and_reload.return_value = new_table
    orch._emos_holder = SimpleNamespace(table=old_table)  # type: ignore[assignment]

    await orch._tick()

    assert orch._emos_holder.table is new_table  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_emos_reload_keeps_prior_table_on_no_change(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    old_table = MagicMock(name="old_table")
    old_table.check_and_reload.return_value = None
    orch._emos_holder = SimpleNamespace(table=old_table)  # type: ignore[assignment]

    await orch._tick()

    assert orch._emos_holder.table is old_table  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_emos_reload_failure_is_swallowed_and_keeps_prior_table(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    old_table = MagicMock(name="old_table")
    old_table.check_and_reload.side_effect = RuntimeError("boom")
    orch._emos_holder = SimpleNamespace(table=old_table)  # type: ignore[assignment]

    await orch._tick()  # must not raise

    assert orch._emos_holder.table is old_table  # type: ignore[union-attr]
    assert orch._consecutive_tick_failures == 0


# ---------------------------------------------------------------------------
# Per-day-cap commit + mark_traded wiring
# ---------------------------------------------------------------------------


def _signal(strategy_name: str = "favorite_bias") -> TradeSignal:
    return TradeSignal(
        market_id="MKT-1",
        market_ticker="MKT-1-T",
        model_probability=0.6,
        market_price=0.5,
        confidence=0.8,
        strategy_name=strategy_name,
        side=Side.YES,
        suggested_dollars=100.0,
        suggested_contracts=2,
        metadata={"city": "MIA", "market_type": "high", "forecast_date": "2026-09-05"},
    )


def _order_record(signal: TradeSignal, status: OrderStatus, order_id: str) -> OrderRecord:
    return OrderRecord(
        order_id=order_id,
        signal=signal,
        risk_decision=RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0),
        mode=ExecutionMode.SHADOW,
        signal_price=0.5,
        order_price=0.5,
        status=status,
    )


@pytest.mark.asyncio
async def test_per_day_cap_commits_only_non_excluded_statuses(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, engine = tick_orch
    strategy = _stub("favorite_bias")
    strategy.generate_signals = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]
    cap = MagicMock()
    strategy._per_day_cap = cap  # type: ignore[attr-defined]
    orch._strategies = [strategy]

    filled_signal = _signal()
    rejected_signal = _signal()
    filled_record = _order_record(filled_signal, OrderStatus.FILLED, "o1")
    rejected_record = _order_record(rejected_signal, OrderStatus.REJECTED, "o2")
    engine.execute_batch = AsyncMock(return_value=[filled_record, rejected_record])

    await orch._tick()

    cap.commit.assert_called_once()
    key_arg, order_id_arg = cap.commit.call_args.args
    assert order_id_arg == "o1"
    assert key_arg == ("MIA", "high", "2026-09-05")


@pytest.mark.asyncio
async def test_per_day_cap_commit_value_error_is_logged_not_raised(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, engine = tick_orch
    strategy = _stub("favorite_bias")
    strategy.generate_signals = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]
    cap = MagicMock()
    cap.commit.side_effect = ValueError("already bound")
    strategy._per_day_cap = cap  # type: ignore[attr-defined]
    orch._strategies = [strategy]

    filled_record = _order_record(_signal(), OrderStatus.FILLED, "o1")
    engine.execute_batch = AsyncMock(return_value=[filled_record])

    await orch._tick()  # must not raise

    assert orch._consecutive_tick_failures == 0


@pytest.mark.asyncio
async def test_mark_traded_called_only_for_executed_statuses(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, engine = tick_orch
    strategy = _stub("favorite_bias")
    strategy.generate_signals = AsyncMock(return_value=[_signal()])  # type: ignore[method-assign]
    marked: list[str] = []
    strategy.mark_traded = lambda ticker: marked.append(ticker)  # type: ignore[attr-defined]
    orch._strategies = [strategy]

    filled_record = _order_record(_signal(), OrderStatus.FILLED, "o1")
    rejected_record = _order_record(_signal(), OrderStatus.REJECTED, "o2")
    engine.execute_batch = AsyncMock(return_value=[filled_record, rejected_record])

    await orch._tick()

    assert marked == ["MKT-1-T"]


# ---------------------------------------------------------------------------
# SILENT BOT alert
# ---------------------------------------------------------------------------


class _FixedDatetime(datetime):
    """Freezes ``datetime.now()`` within orchestrator.py for market-hours math."""

    _fixed: datetime

    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
        return cls._fixed if tz is None else cls._fixed.astimezone(tz)


@pytest.mark.asyncio
async def test_silent_bot_alerts_once_when_eligible_strategies_produce_nothing(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import python.orchestrator as orch_mod

    orch, risk_mgr, _engine = tick_orch
    strategy = _stub("weather_edge")
    strategy.generate_signals = AsyncMock(return_value=[])  # type: ignore[method-assign]
    orch._strategies = [strategy]
    risk_mgr.get_tracker.return_value = MagicMock(enabled=True)

    # Wednesday 12:00 ET — inside market hours (9-18, weekday<5).
    fixed_now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=ET)
    _FixedDatetime._fixed = fixed_now
    monkeypatch.setattr(orch_mod, "datetime", _FixedDatetime)

    orch._last_signal_time = time.monotonic() - (
        orch._settings.telegram.silent_bot_alert_hours + 1
    ) * 3600.0
    orch._silent_alert_sent = False
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    warnings = [msg for msg, level in notifier.sent if level == AlertLevel.WARNING]  # type: ignore[attr-defined]
    assert any("SILENT BOT" in m for m in warnings)
    assert orch._silent_alert_sent is True

    notifier.sent.clear()  # type: ignore[attr-defined]
    await orch._tick()
    warnings2 = [msg for msg, level in notifier.sent if level == AlertLevel.WARNING]  # type: ignore[attr-defined]
    assert not any("SILENT BOT" in m for m in warnings2)


@pytest.mark.asyncio
async def test_silent_bot_never_alerts_with_no_eligible_strategies(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import python.orchestrator as orch_mod

    orch, _risk_mgr, _engine = tick_orch
    orch._strategies = []  # no strategies at all -> eligible == []

    fixed_now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=ET)
    _FixedDatetime._fixed = fixed_now
    monkeypatch.setattr(orch_mod, "datetime", _FixedDatetime)

    orch._last_signal_time = time.monotonic() - 999_999.0
    orch._silent_alert_sent = False
    notifier = orch._notifier
    assert notifier is not None

    await orch._tick()

    warnings = [msg for msg, level in notifier.sent if level == AlertLevel.WARNING]  # type: ignore[attr-defined]
    assert not any("SILENT BOT" in m for m in warnings)


# ---------------------------------------------------------------------------
# Tick-failure escalation ladder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_failure_ladder_alerts_then_kills(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._check_ws_health = AsyncMock(side_effect=RuntimeError("simulated"))  # type: ignore[method-assign]
    notifier = orch._notifier
    assert notifier is not None

    for _ in range(orch._TICK_FAILURE_ALERT_THRESHOLD - 1):
        await orch._tick()
    assert orch._consecutive_tick_failures == orch._TICK_FAILURE_ALERT_THRESHOLD - 1
    criticals_before = [msg for msg, level in notifier.sent if level == AlertLevel.CRITICAL]  # type: ignore[attr-defined]
    assert not any("consecutive tick failures" in m for m in criticals_before)

    await orch._tick()
    assert orch._consecutive_tick_failures == orch._TICK_FAILURE_ALERT_THRESHOLD
    criticals = [msg for msg, level in notifier.sent if level == AlertLevel.CRITICAL]  # type: ignore[attr-defined]
    assert any("consecutive tick failures" in m for m in criticals)
    assert orch._running is True  # not killed yet

    for _ in range(orch._TICK_FAILURE_KILL_THRESHOLD - orch._TICK_FAILURE_ALERT_THRESHOLD):
        await orch._tick()

    assert orch._consecutive_tick_failures == orch._TICK_FAILURE_KILL_THRESHOLD
    assert orch._running is False
    assert orch._settings.is_kill_switch_active() is True


@pytest.mark.asyncio
async def test_tick_failure_logs_traceback(
    tick_orch: tuple[Orchestrator, MagicMock, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    orch, _risk_mgr, _engine = tick_orch
    orch._check_ws_health = AsyncMock(side_effect=RuntimeError("simulated"))  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR, logger="python.orchestrator"):
        await orch._tick()

    tick_failed_records = [r for r in caplog.records if "Tick failed" in r.getMessage()]
    assert tick_failed_records, "expected a 'Tick failed' log record"
    assert tick_failed_records[-1].exc_info is not None
