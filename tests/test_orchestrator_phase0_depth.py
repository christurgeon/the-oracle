"""M0 — Phase 0 depth monitor orchestrator integration.

Verifies that the orchestrator spawns ``Phase0DepthMonitor.run`` as a
background task at startup and cancels it cleanly at shutdown. Pure
asyncio-level wiring test — no Kalshi network calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from python.orchestrator import Orchestrator


def _bare_orchestrator() -> Orchestrator:
    """Build a partially-initialised Orchestrator skeleton for shutdown tests."""
    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = "/tmp/oracle-test-state"
    orch._engine = None
    orch._kalshi_client = None
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


@pytest.mark.asyncio
async def test_shutdown_cancels_phase0_depth_task():
    """shutdown() must cancel the Phase 0 depth-monitor task if it's running."""
    orch = _bare_orchestrator()

    async def _forever() -> None:
        while True:
            await asyncio.sleep(3600)

    orch._phase0_depth_task = asyncio.create_task(_forever(), name="phase0-test")
    # Give the loop one tick so the task is actually running.
    await asyncio.sleep(0)
    assert not orch._phase0_depth_task.done()

    await orch.shutdown()

    assert orch._phase0_depth_task.cancelled() or orch._phase0_depth_task.done()


@pytest.mark.asyncio
async def test_shutdown_handles_none_phase0_task_gracefully():
    """shutdown() with no phase0 task must not raise (partial-init path)."""
    orch = _bare_orchestrator()
    assert orch._phase0_depth_task is None
    await orch.shutdown()  # must not raise


def test_setup_phase0_depth_monitor_skips_without_kalshi_client():
    """No Kalshi client -> monitor stays None (offline / missing-creds path)."""
    orch = _bare_orchestrator()
    orch._kalshi_client = None
    orch._settings.state_dir = "/tmp/oracle-test-state"

    orch._setup_phase0_depth_monitor()
    assert orch._phase0_depth_monitor is None


def test_setup_phase0_depth_monitor_constructs_with_kalshi_client(tmp_path):
    """With a Kalshi client present the monitor is constructed under state_dir."""
    from python.data.phase0_depth_monitor import (
        PHASE0_DEPTH_SINK_FILENAME,
        Phase0DepthMonitor,
    )

    orch = _bare_orchestrator()
    orch._kalshi_client = MagicMock()
    orch._settings.state_dir = str(tmp_path)

    orch._setup_phase0_depth_monitor()
    assert isinstance(orch._phase0_depth_monitor, Phase0DepthMonitor)
    assert orch._phase0_depth_monitor.sink_path == tmp_path / PHASE0_DEPTH_SINK_FILENAME
