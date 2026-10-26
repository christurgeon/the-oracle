"""C-SIGTERM: shutdown() must cancel resting orders before closing the REST client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_shutdown_cancels_resting_orders_before_rest_close():
    """shutdown() must call engine.cancel_all_open() before kalshi_client.close()."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = "/tmp/oracle-test-state"

    call_order: list[str] = []

    orch._engine = MagicMock()
    async def _cancel_all_open():
        call_order.append("engine.cancel_all_open")
        return (["O-1", "O-2"], False)
    orch._engine.cancel_all_open = _cancel_all_open

    orch._kalshi_client = MagicMock()
    async def _client_close():
        call_order.append("kalshi_client.close")
    orch._kalshi_client.close = _client_close

    # Mock other shutdown-touched attrs to no-op
    orch._hc_client = None
    orch._calibration_monitor = None
    orch._tripwire_task = None
    orch._watchdog_task = None
    orch._notifier = None
    orch._telegram_task = None
    orch._risk_manager = None
    orch._ws_task = None
    orch._kalshi_ws = None
    orch._strategies = []
    orch._weather_client = None
    orch._trade_log_pidlock_fd = None
    orch._phase0_depth_task = None
    orch._kxrain_observer = None
    orch._kxrain_task = None
    orch._kxrain_sweep = None
    orch._kxrain_sweep_task = None

    await orch.shutdown()

    assert "engine.cancel_all_open" in call_order, (
        "engine.cancel_all_open was not called during shutdown"
    )
    cancel_idx = call_order.index("engine.cancel_all_open")
    close_idx = call_order.index("kalshi_client.close")
    assert cancel_idx < close_idx, (
        f"engine.cancel_all_open ({cancel_idx}) must run before "
        f"kalshi_client.close ({close_idx})"
    )


@pytest.mark.asyncio
async def test_shutdown_no_engine_skips_cancel():
    """shutdown() with engine=None must not raise (partial-init shutdown)."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = "/tmp/oracle-test-state"
    orch._engine = None
    orch._kalshi_client = None
    orch._hc_client = None
    orch._calibration_monitor = None
    orch._tripwire_task = None
    orch._watchdog_task = None
    orch._notifier = None
    orch._telegram_task = None
    orch._risk_manager = None
    orch._ws_task = None
    orch._kalshi_ws = None
    orch._strategies = []
    orch._weather_client = None
    orch._trade_log_pidlock_fd = None
    orch._phase0_depth_task = None
    orch._kxrain_observer = None
    orch._kxrain_task = None
    orch._kxrain_sweep = None
    orch._kxrain_sweep_task = None

    await orch.shutdown()  # must not raise


@pytest.mark.asyncio
async def test_shutdown_cancel_exception_does_not_block_shutdown():
    """An exception from cancel_all_open must be logged but not block shutdown."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = "/tmp/oracle-test-state"

    closed = {"value": False}

    orch._engine = MagicMock()
    async def _cancel_all_open():
        raise RuntimeError("simulated kalshi 503 during shutdown")
    orch._engine.cancel_all_open = _cancel_all_open

    orch._kalshi_client = MagicMock()
    async def _client_close():
        closed["value"] = True
    orch._kalshi_client.close = _client_close

    orch._hc_client = None
    orch._calibration_monitor = None
    orch._tripwire_task = None
    orch._watchdog_task = None
    orch._notifier = None
    orch._telegram_task = None
    orch._risk_manager = None
    orch._ws_task = None
    orch._kalshi_ws = None
    orch._strategies = []
    orch._weather_client = None
    orch._trade_log_pidlock_fd = None
    orch._phase0_depth_task = None
    orch._kxrain_observer = None
    orch._kxrain_task = None
    orch._kxrain_sweep = None
    orch._kxrain_sweep_task = None

    await orch.shutdown()
    assert closed["value"] is True, (
        "kalshi_client.close was not called after cancel exception"
    )
