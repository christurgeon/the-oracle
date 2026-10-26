"""Tests for the pure apply_intraday_shadow_drop function and its tick-loop wiring."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import Side
from python.orchestrator import Orchestrator, apply_intraday_shadow_drop
from python.strategies.base import Strategy, TradeSignal


def _sig(strategy: str, ticker: str, edge: float = 0.1, contracts: int = 10) -> TradeSignal:
    return TradeSignal(
        market_id=ticker,
        market_ticker=ticker,
        model_probability=0.65,
        market_price=0.55,
        confidence=0.7,
        strategy_name=strategy,
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=contracts,
        effective_edge=edge,
        metadata={"city": "NYC", "market_type": "high", "forecast_date": "2026-05-30"},
    )


def test_shadow_drop_removes_intraday_signals() -> None:
    we = _sig("weather_edge", "TICKER_WE")
    intraday = _sig("intraday_emos_high", "TICKER_INTRA")
    out = apply_intraday_shadow_drop([we, intraday], shadow_only=True)
    assert we in out
    assert intraday not in out


def test_shadow_drop_passthrough_when_disabled() -> None:
    we = _sig("weather_edge", "TICKER_WE")
    intraday = _sig("intraday_emos_high", "TICKER_INTRA")
    out = apply_intraday_shadow_drop([we, intraday], shadow_only=False)
    assert we in out
    assert intraday in out


def test_shadow_drop_empty_batch() -> None:
    out = apply_intraday_shadow_drop([], shadow_only=True)
    assert out == []


def test_shadow_drop_no_intraday_signals() -> None:
    we1 = _sig("weather_edge", "TICKER_A")
    we2 = _sig("weather_edge", "TICKER_B")
    out = apply_intraday_shadow_drop([we1, we2], shadow_only=True)
    assert out == [we1, we2]


def test_shadow_drop_log_per_signal(caplog: pytest.LogCaptureFixture) -> None:
    intra1 = _sig("intraday_emos_high", "TICKER_X1", edge=0.12, contracts=5)
    intra2 = _sig("intraday_emos_high", "TICKER_X2", edge=0.15, contracts=8)
    we = _sig("weather_edge", "TICKER_WE")

    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        out = apply_intraday_shadow_drop([intra1, intra2, we], shadow_only=True)

    assert intra1 not in out
    assert intra2 not in out
    assert we in out

    drop_records = [r for r in caplog.records if "intraday_shadow_drop" in r.getMessage()]
    assert len(drop_records) == 2
    messages = [r.getMessage() for r in drop_records]
    assert any("TICKER_X1" in m for m in messages)
    assert any("TICKER_X2" in m for m in messages)


# ---------------------------------------------------------------------------
# Tick-loop wiring test: the safety guarantee lives in the wiring, not the
# pure function. A regression that deletes the orchestrator's call site
# would let intraday signals reach execute_batch in live mode. This test
# proves the wiring is present.
# ---------------------------------------------------------------------------


class _StubStrategy(Strategy):
    def __init__(self, *, name: str, signals: list[TradeSignal]) -> None:
        super().__init__(name=name, config={"enabled": True}, bankroll=10_000.0)
        self._signals_to_emit = signals

    async def scan(self) -> list[dict[str, Any]]:
        return []

    async def evaluate(self, market: dict[str, Any]) -> float | None:
        return None

    async def generate_signals(self) -> list[TradeSignal]:
        return list(self._signals_to_emit)


@pytest.mark.asyncio
async def test_tick_drops_intraday_before_execute_batch() -> None:
    """End-to-end: the orchestrator's tick loop must call the drop hook so
    intraday signals do not reach execute_batch when shadow_only=True. This
    catches a regression where someone deletes the call site in _tick.
    """
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
    orch._risk_manager = risk_mgr

    engine = MagicMock()
    engine.settle_expired_positions = AsyncMock(return_value=[])
    engine.check_all_pending_fills = AsyncMock(return_value=([], []))
    engine.execute_batch = AsyncMock(return_value=[])
    orch._engine = engine

    intra = _sig("intraday_emos_high", "KXHIGHMIA-26MAY30-T88", edge=0.12)
    we = _sig("weather_edge", "KXHIGHNYC-26MAY30-B72.5", edge=0.10)
    orch._strategies = [
        _StubStrategy(name="intraday_emos_high", signals=[intra]),
        _StubStrategy(name="weather_edge", signals=[we]),
    ]

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.strategies.weather.shadow_only = False  # weather_edge live (passthrough)
    settings.strategies.intraday_emos_high.shadow_only = True
    orch._settings = settings

    await orch._tick()

    assert engine.execute_batch.await_count == 1
    executed = list(engine.execute_batch.await_args.args[0])
    tickers = [s.market_ticker for s in executed]
    assert "KXHIGHMIA-26MAY30-T88" not in tickers, (
        "regression: intraday signal reached execute_batch despite shadow_only=True — "
        "drop-hook wiring removed?"
    )
    assert "KXHIGHNYC-26MAY30-B72.5" in tickers, (
        "weather_edge signal must pass through the shadow drop hook"
    )
