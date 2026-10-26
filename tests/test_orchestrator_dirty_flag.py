"""C-DIRTY-RACE: failed save_state must re-arm _state_dirty for next-tick retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_save_failure_rearms_dirty_flag(tmp_path):
    """When save_state returns False, mark_dirty must be called."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = str(tmp_path)
    orch._save_lock = MagicMock()
    orch._save_lock.__aenter__ = AsyncMock(return_value=None)
    orch._save_lock.__aexit__ = AsyncMock(return_value=None)
    orch._save_failure_alert_sent = False

    rm = MagicMock()
    rm.consecutive_save_failures = 0
    rm.record_save_failure = MagicMock(return_value=1)
    rm.mark_dirty = MagicMock()
    rm.mark_state_clean = MagicMock()
    rm.clear_save_failures = MagicMock()
    orch._risk_manager = rm

    orch._notify_alert = AsyncMock()

    with patch("python.orchestrator.save_state", new=AsyncMock(return_value=False)):
        await orch._save_under_lock(tmp_path, snapshot={"k": "v"})

    rm.mark_dirty.assert_called_once()
    rm.record_save_failure.assert_called_once()
    rm.mark_state_clean.assert_not_called()


@pytest.mark.asyncio
async def test_save_success_clears_failure_state(tmp_path):
    """When save_state returns True, consecutive_save_failures is cleared."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = str(tmp_path)
    orch._save_lock = MagicMock()
    orch._save_lock.__aenter__ = AsyncMock(return_value=None)
    orch._save_lock.__aexit__ = AsyncMock(return_value=None)
    orch._save_failure_alert_sent = True

    rm = MagicMock()
    rm.consecutive_save_failures = 3
    rm.clear_save_failures = MagicMock()
    rm.mark_dirty = MagicMock()
    rm.mark_state_clean = MagicMock()
    orch._risk_manager = rm

    orch._notify_alert = AsyncMock()

    with patch("python.orchestrator.save_state", new=AsyncMock(return_value=True)):
        await orch._save_under_lock(tmp_path, snapshot={"k": "v"})

    rm.clear_save_failures.assert_called_once()
    rm.mark_dirty.assert_not_called()
    assert orch._save_failure_alert_sent is False


@pytest.mark.asyncio
async def test_save_exception_rearms_dirty_flag(tmp_path):
    """An unexpected exception in _save_under_lock must re-arm dirty."""
    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = str(tmp_path)
    orch._save_lock = MagicMock()
    orch._save_lock.__aenter__ = AsyncMock(return_value=None)
    orch._save_lock.__aexit__ = AsyncMock(return_value=None)
    orch._save_failure_alert_sent = False

    rm = MagicMock()
    rm.consecutive_save_failures = 0
    rm.mark_dirty = MagicMock()
    rm.mark_state_clean = MagicMock()
    rm.clear_save_failures = MagicMock(side_effect=RuntimeError("simulated bug"))
    orch._risk_manager = rm

    orch._notify_alert = AsyncMock()

    with patch("python.orchestrator.save_state", new=AsyncMock(return_value=True)):
        await orch._save_under_lock(tmp_path, snapshot={"k": "v"})

    rm.mark_dirty.assert_called_once()


@pytest.mark.asyncio
async def test_persist_overflow_preserves_dirty_flag(tmp_path):
    """T4: when a save is already in flight, _persist_if_dirty defers and
    leaves _state_dirty=True so the next tick retries (drop-on-overflow).

    Tests the existing in-flight check at orchestrator.py:1198-1205 in
    interaction with the C-DIRTY-RACE re-arm semantics: if the in-flight
    save eventually fails, dirty must still be True for next-tick retry.
    """
    import asyncio as _asyncio

    from python.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = str(tmp_path)
    orch._save_lock = MagicMock()
    orch._save_lock.__aenter__ = AsyncMock(return_value=None)
    orch._save_lock.__aexit__ = AsyncMock(return_value=None)
    orch._save_failure_alert_sent = False

    rm = MagicMock()
    rm.state_dirty = True  # caller wants to persist
    rm.consecutive_save_failures = 0
    rm.record_save_failure = MagicMock(return_value=1)
    rm.mark_dirty = MagicMock()
    rm.mark_state_clean = MagicMock()
    rm.clear_save_failures = MagicMock()
    orch._risk_manager = rm

    orch._notify_alert = AsyncMock()

    # Simulate a still-running save task by creating a fake "not done" task.
    pending_task_event = _asyncio.Event()

    async def _slow_save():
        await pending_task_event.wait()

    orch._save_task = _asyncio.create_task(_slow_save())
    orch._heartbeat_task = None

    # Build the snapshot once so _persist_if_dirty doesn't need a real risk_manager.
    orch._build_state_snapshot = MagicMock(return_value={"k": "v"})

    # Capture the original save_task reference to ensure it's not replaced.
    original_save_task = orch._save_task

    # Call _persist_if_dirty: should NOT spawn a new save (existing one
    # in flight) and should NOT call mark_state_clean (preserving dirty).
    orch._persist_if_dirty()

    # The original save_task should still be the same task (no replacement).
    assert orch._save_task is original_save_task
    rm.mark_state_clean.assert_not_called()
    rm.mark_dirty.assert_not_called()

    # Cleanup: complete the slow save so the test doesn't leak the task.
    pending_task_event.set()
    await orch._save_task
    # _persist_if_dirty also schedules a heartbeat write; await it so the
    # test doesn't leak a pending task into the loop teardown.
    if orch._heartbeat_task is not None:
        await orch._heartbeat_task


def test_risk_manager_mark_dirty():
    """mark_dirty() is the inverse of mark_state_clean()."""
    from python.risk.manager import RiskManager

    rm = RiskManager.__new__(RiskManager)
    rm._state_dirty = False

    rm.mark_dirty()
    assert rm._state_dirty is True

    rm.mark_state_clean()
    assert rm._state_dirty is False

    rm.mark_dirty()
    rm.mark_dirty()
    assert rm._state_dirty is True
