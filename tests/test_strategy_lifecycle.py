"""Strategy ABC lifecycle hooks default to no-ops."""

from __future__ import annotations

import pytest

from python.strategies.base import Strategy, TradeSignal  # noqa: F401


class _Dummy(Strategy):
    async def scan(self) -> list[dict]:
        return []

    async def evaluate(self, market: dict) -> float | None:
        return None

    async def generate_signals(self) -> list[TradeSignal]:
        return []


@pytest.mark.asyncio
async def test_strategy_setup_default_is_noop():
    s = _Dummy(name="dummy", config={})
    # Must not raise — default is a no-op.
    await s.setup()


@pytest.mark.asyncio
async def test_strategy_aclose_default_is_noop():
    s = _Dummy(name="dummy", config={})
    await s.aclose()


class _Recorder(Strategy):
    def __init__(self) -> None:
        super().__init__(name="recorder", config={})
        self.setup_called = 0
        self.aclose_called = 0

    async def scan(self) -> list[dict]:
        return []

    async def evaluate(self, market: dict) -> float | None:
        return None

    async def generate_signals(self) -> list[TradeSignal]:
        return []

    async def setup(self) -> None:
        self.setup_called += 1

    async def aclose(self) -> None:
        self.aclose_called += 1


@pytest.mark.asyncio
async def test_orchestrator_invokes_setup_and_aclose_on_each_strategy():
    """Every registered strategy receives setup() during initialize() and
    aclose() during shutdown()."""
    from python.orchestrator import Orchestrator
    from python.settings import Settings

    settings = Settings()  # default settings; nothing real wired

    orch = Orchestrator(settings)
    rec = _Recorder()
    orch._strategies.append(rec)

    # Drive only the lifecycle loops, not full initialize/shutdown.
    await orch._run_strategy_lifecycle_hook("setup")
    assert rec.setup_called == 1

    await orch._run_strategy_lifecycle_hook("aclose")
    assert rec.aclose_called == 1
