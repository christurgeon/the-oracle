"""systemd watchdog keepalive coverage.

The real decoupling guarantee is the INDEPENDENT heartbeat task
(``_watchdog_heartbeat_loop``): the main loop runs ticks serially, so during
an Open-Meteo outage a single ensemble fetch can block ~127s inside one tick
(4 × 30s timeout + backoff). With the ping inlined in ``_tick`` (start or end),
the main-loop coroutine is parked inside that tick and can't re-ping until it
returns — tick_duration + poll_interval then exceeds WatchdogSec (120s) →
SIGABRT crash-loop (the 2026-06 incident). Only a separate task refreshes the
deadline mid-tick.

The start-of-tick ping remains as a freshness bonus (and is covered below), but
it is NOT sufficient on its own — that's what the independent-task tests guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import python.orchestrator as orch_mod
from python.orchestrator import Orchestrator


def _minimal_orch(monkeypatch: pytest.MonkeyPatch) -> tuple[Orchestrator, list[str]]:
    pings: list[str] = []
    monkeypatch.setattr(orch_mod, "_sd_notify", lambda state: pings.append(state))

    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.is_kill_switch_active.return_value = False
    orch._risk_manager = MagicMock()
    orch._engine = MagicMock()
    orch._save_task = None
    orch._tick_count = 0
    orch._consecutive_tick_failures = 0
    orch._notify_alert = AsyncMock()
    return orch, pings


@pytest.mark.asyncio
async def test_watchdog_pings_at_tick_start_even_when_phase_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the tick body fails early (e.g. a hung/slow weather phase), the
    start-of-tick WATCHDOG=1 ping must already have been sent."""
    orch, pings = _minimal_orch(monkeypatch)

    # Make the tick body blow up immediately after the start ping. _tick
    # swallows the exception (consecutive-failure path), so _tick returns.
    orch._check_ws_health = AsyncMock(side_effect=RuntimeError("simulated slow/hung phase"))

    await orch._tick()

    assert "WATCHDOG=1" in pings, "watchdog keepalive was not sent at tick start"
    # The early failure means we never reached the end-of-tick ping, proving the
    # start ping is what saved us.
    assert orch._consecutive_tick_failures == 1


@pytest.mark.asyncio
async def test_watchdog_heartbeat_pings_independently_of_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated heartbeat task pings on its own cadence — it does not need
    a tick to complete (or even start). This is the guarantee that survives a
    single tick blocking longer than WatchdogSec."""
    orch, pings = _minimal_orch(monkeypatch)
    orch._running = True
    monkeypatch.setenv("WATCHDOG_USEC", "120000000")  # 120s -> ping every 40s

    # Spin the loop fast and stop it after a few pings so the test is instant.
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) >= 3:
            orch._running = False

    monkeypatch.setattr(orch_mod.asyncio, "sleep", fake_sleep)

    await orch._watchdog_heartbeat_loop()

    assert pings.count("WATCHDOG=1") >= 3, "heartbeat task must ping repeatedly on its own"
    assert sleeps[0] == pytest.approx(40.0), "interval should be WatchdogSec/3"


@pytest.mark.asyncio
async def test_watchdog_heartbeat_noop_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No NOTIFY/watchdog env (local dev) → the loop returns immediately and
    never pings, so it can't busy-spin off-systemd."""
    orch, pings = _minimal_orch(monkeypatch)
    orch._running = True
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)

    await orch._watchdog_heartbeat_loop()

    assert pings == []
