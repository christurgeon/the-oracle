"""KXRAIN observer orchestrator wiring, and its zero-orders proofs.

Pins the guarantees that keep the observer from placing an order: it's never
registered as a strategy, and it gets no risk allocation entry."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.orchestrator import Orchestrator, validate_strategy_allocations
from python.settings import Settings


class _FakeKalshiClient:
    """A stand-in with the three GETs and nothing that could place an order."""

    async def get_markets(self, **kwargs: Any) -> dict[str, Any]:
        return {"markets": [], "cursor": ""}

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return {}

    async def get_market_trades(self, **kwargs: Any) -> dict[str, Any]:
        return {"trades": [], "cursor": ""}

    def backoff_remaining_seconds(self) -> float:
        return 0.0

    async def close(self) -> None:
        return None


def _bare_orchestrator(tmp_path: Path, *, enabled: bool) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    settings = Settings.load()
    settings.kxrain_shadow = settings.kxrain_shadow.model_copy(
        update={"enabled": enabled, "log_dir": str(tmp_path)}
    )
    settings.state_dir = str(tmp_path)
    orch._settings = settings
    orch._running = True
    orch._engine = None
    orch._kalshi_client = _FakeKalshiClient()
    orch._kalshi_ws = None
    orch._hc_client = None
    orch._calibration_monitor = None
    orch._tripwire_task = None
    orch._watchdog_task = None
    orch._notifier = None
    orch._telegram_task = None
    orch._risk_manager = None
    orch._ws_task = None
    orch._strategies = []
    orch._weather_client = None
    orch._trade_log_pidlock_fd = None
    orch._phase0_depth_monitor = None
    orch._phase0_depth_task = None
    orch._kxrain_observer = None
    orch._kxrain_task = None
    orch._kxrain_sweep = None
    orch._kxrain_sweep_task = None
    return orch


# Construction


def test_disabled_config_constructs_nothing(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=False)
    orch._setup_kxrain_observer()
    assert orch._kxrain_observer is None
    assert orch._kxrain_sweep is None


def test_enabled_config_constructs_both(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()
    assert orch._kxrain_observer is not None
    assert orch._kxrain_sweep is not None


def test_setup_never_aborts_initialize(tmp_path: Path) -> None:
    # An OSError from mkdir/ledger-read must not abort initialize().
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._settings.kxrain_shadow = MagicMock()
    orch._settings.kxrain_shadow.enabled = True
    orch._settings.kxrain_shadow.to_observer_config.side_effect = OSError("read-only fs")

    orch._setup_kxrain_observer()  # must not raise
    assert orch._kxrain_observer is None


def test_no_kalshi_client_constructs_nothing(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._kalshi_client = None
    orch._setup_kxrain_observer()
    assert orch._kxrain_observer is None


def test_the_notifier_is_resolved_lazily_not_captured(tmp_path: Path) -> None:
    # Capturing the notifier here would pin None and silence the liveness alert.
    orch = _bare_orchestrator(tmp_path, enabled=True)
    assert orch._notifier is None
    orch._setup_kxrain_observer()

    later = object()
    orch._notifier = later  # what initialize() does a few lines on
    assert orch._kxrain_observer is not None
    assert orch._kxrain_observer._notifier_getter() is later


# Zero orders, orchestrator side


def test_the_observer_is_never_registered_as_a_strategy(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    before = list(orch._strategies)
    orch._setup_kxrain_observer()

    assert orch._strategies == before
    assert not any("kxrain" in s.name.lower() for s in orch._strategies)


def test_an_allocation_entry_is_a_startup_crash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        validate_strategy_allocations(
            {"kxrain_shadow": 0.1, "favorite_bias": 0.9},
            [("favorite_bias", True)],
        )


def test_the_shipped_config_has_no_kxrain_allocation() -> None:
    allocations = Settings.load().risk.strategy_allocations
    assert not any("kxrain" in name.lower() for name in allocations)


async def test_a_full_observer_tick_places_no_orders(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._engine = MagicMock()
    orch._engine.execute_batch = AsyncMock(return_value=[])
    orch._setup_kxrain_observer()

    assert orch._kxrain_observer is not None
    await orch._kxrain_observer.run_once()
    await orch._kxrain_sweep.run_once()

    orch._engine.execute_batch.assert_not_called()


async def test_the_test_orchestrator_writes_only_under_tmp_path(tmp_path: Path) -> None:
    # The orchestrator hands the observer settings.state_dir, not kxrain_shadow.log_dir.
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()
    assert orch._kxrain_observer is not None
    await orch._kxrain_observer.run_once()
    await orch._kxrain_observer.aclose()
    assert (tmp_path / "kxrain_ticks.jsonl").exists()


# Task lifecycle


async def test_shutdown_cancels_both_tasks_and_closes_the_observer(
    tmp_path: Path,
) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()
    observer = orch._kxrain_observer
    assert observer is not None

    async def _forever() -> None:
        while True:
            await asyncio.sleep(3600)

    orch._kxrain_task = asyncio.create_task(_forever(), name="kxrain-test")
    orch._kxrain_sweep_task = asyncio.create_task(_forever(), name="kxrain-sweep-test")
    await asyncio.sleep(0)

    await orch.shutdown()

    assert orch._kxrain_task.cancelled() or orch._kxrain_task.done()
    assert orch._kxrain_sweep_task.cancelled() or orch._kxrain_sweep_task.done()
    # aclose() shuts the observer's own single-worker executor down.
    assert observer._executor._shutdown is True


async def test_shutdown_is_clean_when_the_observer_was_never_built(
    tmp_path: Path,
) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=False)
    orch._setup_kxrain_observer()
    await orch.shutdown()  # must not raise


async def test_a_dying_task_is_escalated(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)

    async def _boom() -> None:
        raise RuntimeError("the corpus has stopped accruing")

    task = asyncio.create_task(_boom(), name="kxrain-observer")
    task.add_done_callback(orch._on_kxrain_task_done)
    with pytest.raises(RuntimeError):
        await task
    await asyncio.sleep(0)


async def test_cancellation_is_not_escalated(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)

    async def _forever() -> None:
        while True:
            await asyncio.sleep(3600)

    task = asyncio.create_task(_forever(), name="kxrain-observer")
    task.add_done_callback(orch._on_kxrain_task_done)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# The settlement sweep schedule


async def test_the_sweep_runs_once_per_utc_day(tmp_path: Path) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()
    assert orch._kxrain_sweep is not None

    calls = {"n": 0}
    real = orch._kxrain_sweep.run_once

    async def counted() -> Any:
        calls["n"] += 1
        return await real()

    orch._kxrain_sweep.run_once = counted  # type: ignore[method-assign]
    orch._settings.kxrain_shadow = orch._settings.kxrain_shadow.model_copy(
        update={"settlement_capture_hour_utc": 0}
    )
    orch._KXRAIN_SWEEP_POLL_SECONDS = 0.0  # type: ignore[misc]

    task = asyncio.create_task(orch._kxrain_sweep_loop())
    for _ in range(200):
        await asyncio.sleep(0)
        if calls["n"]:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls["n"] == 1


async def test_an_anomalous_sweep_alerts(tmp_path: Path) -> None:
    # The sweep's reach is ~28 days, so silence is only recoverable inside it.
    from python.data.kxrain_settlements import SettlementSweepResult

    orch = _bare_orchestrator(tmp_path, enabled=True)
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()
    orch._notifier = notifier

    await orch._alert_kxrain_sweep(SettlementSweepResult(n_returned=0))
    notifier.send_alert.assert_awaited_once()
    assert "anomaly" in notifier.send_alert.await_args.args[0].lower()


async def test_the_sweep_alert_survives_a_missing_notifier(tmp_path: Path) -> None:
    from python.data.kxrain_settlements import SettlementSweepResult

    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._notifier = None
    await orch._alert_kxrain_sweep(SettlementSweepResult(n_returned=0))


# The observer must never cost the bot money or a filesystem


def test_the_observer_writes_to_the_bots_own_state_dir(tmp_path: Path) -> None:
    # log_dir must follow state_dir, or shutil.disk_usage checks the wrong filesystem.
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._settings.state_dir = str(tmp_path / "elsewhere")
    orch._settings.kxrain_shadow = orch._settings.kxrain_shadow.model_copy(
        update={"log_dir": "state"}
    )
    orch._setup_kxrain_observer()

    assert orch._kxrain_observer is not None
    assert orch._kxrain_observer._config.log_dir == Path(orch._settings.state_dir)


def test_the_sweep_shares_the_observers_private_worker(tmp_path: Path) -> None:
    # Not the default pool: its saturation has crashed the bot before.
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()
    assert orch._kxrain_sweep is not None
    assert orch._kxrain_sweep._executor is orch._kxrain_observer._executor


def test_the_sweep_inherits_the_configured_series_ticker(tmp_path: Path) -> None:
    # A defaulted series would silently split decisions from settlements across two.
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._settings.kxrain_shadow = orch._settings.kxrain_shadow.model_copy(
        update={"series_ticker": "KXRAINV2"}
    )
    orch._setup_kxrain_observer()
    assert orch._kxrain_sweep is not None
    assert orch._kxrain_sweep._series_ticker == "KXRAINV2"


async def test_shutdown_survives_a_task_that_dies_while_cancelling(
    tmp_path: Path,
) -> None:
    orch = _bare_orchestrator(tmp_path, enabled=True)
    orch._setup_kxrain_observer()

    async def _explode_on_cancel() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError("the flush blew up in the cancellation path") from None

    orch._kxrain_task = asyncio.create_task(_explode_on_cancel())
    await asyncio.sleep(0)

    await orch.shutdown()  # must not raise


async def test_a_dead_task_raises_a_critical_alert(tmp_path: Path) -> None:
    # This callback is the only detector: run_once's own liveness check dies with the task.
    from python.notifications.base import AlertLevel

    orch = _bare_orchestrator(tmp_path, enabled=True)
    notifier = MagicMock()
    notifier.send_alert = AsyncMock()
    orch._notifier = notifier

    async def _boom() -> None:
        raise RuntimeError("corpus stopped")

    task = asyncio.create_task(_boom(), name="kxrain-observer")
    task.add_done_callback(orch._on_kxrain_task_done)
    with pytest.raises(RuntimeError):
        await task
    for _ in range(50):
        await asyncio.sleep(0)
        if notifier.send_alert.await_count:
            break

    notifier.send_alert.assert_awaited_once()
    assert notifier.send_alert.await_args.kwargs["level"] is AlertLevel.CRITICAL
