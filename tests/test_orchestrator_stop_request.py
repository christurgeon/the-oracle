"""A stop request must end the poll sleep immediately.

systemd gives the bot ``TimeoutStopSec=60`` and ``poll_interval_seconds`` is
also 60. When the loop only checked ``_running`` after a plain
``asyncio.sleep(poll_interval)``, a SIGTERM that arrived during a tick waited
out the whole sleep and systemd SIGKILLed the process before teardown
(``cancel_all_open`` and the final risk-state save) ran.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from python.orchestrator import Orchestrator


def _orch() -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch._running = True
    orch._stop_event = asyncio.Event()
    return orch


async def test_stop_request_during_tick_skips_poll_sleep() -> None:
    orch = _orch()

    async def tick() -> None:
        orch._request_stop()

    orch._tick = AsyncMock(side_effect=tick)  # type: ignore[method-assign]

    await asyncio.wait_for(orch._tick_loop(poll_interval=60), timeout=2)

    assert orch._tick.await_count == 1


async def test_stop_request_during_poll_sleep_wakes_it() -> None:
    orch = _orch()
    orch._tick = AsyncMock()  # type: ignore[method-assign]

    task = asyncio.create_task(orch._tick_loop(poll_interval=60))
    await asyncio.sleep(0.05)
    assert not task.done()

    orch._request_stop()

    await asyncio.wait_for(task, timeout=2)
    assert orch._tick.await_count == 1


async def test_poll_sleep_still_paces_ticks_without_stop() -> None:
    orch = _orch()
    calls = 0

    async def tick() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            orch._request_stop()

    orch._tick = AsyncMock(side_effect=tick)  # type: ignore[method-assign]

    await asyncio.wait_for(orch._tick_loop(poll_interval=0.05), timeout=2)

    assert calls == 3


def test_request_stop_clears_running_flag() -> None:
    orch = _orch()

    orch._request_stop()

    assert orch._running is False
    assert orch._stop_event.is_set()


@pytest.mark.parametrize("poll_interval", [1, 60])
async def test_loop_exits_without_tick_when_already_stopped(poll_interval: int) -> None:
    orch = _orch()
    orch._tick = AsyncMock()  # type: ignore[method-assign]
    orch._request_stop()

    await asyncio.wait_for(orch._tick_loop(poll_interval=poll_interval), timeout=2)

    orch._tick.assert_not_awaited()
