"""Tests for apply_weather_shadow_drop and its tick-loop wiring.

weather_edge has no demonstrated edge against the efficient morning 1°F market,
so it runs shadow-only (keeps evaluating + logging the calibration corpus, places
zero orders). The drop hook is wired BEFORE the favorite_bias / intraday-asos
deference hooks so those strategies do not defer to a weather_edge that will not
actually trade.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import Side
from python.orchestrator import (
    Orchestrator,
    apply_favorite_bias_deference,
    apply_weather_shadow_drop,
)
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


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------


def test_weather_shadow_drop_removes_weather_edge_signals() -> None:
    we = _sig("weather_edge", "TICKER_WE")
    fav = _sig("favorite_bias", "TICKER_FAV")
    out = apply_weather_shadow_drop([we, fav], shadow_only=True)
    assert we not in out
    assert fav in out


def test_weather_shadow_drop_passthrough_when_disabled() -> None:
    we = _sig("weather_edge", "TICKER_WE")
    fav = _sig("favorite_bias", "TICKER_FAV")
    out = apply_weather_shadow_drop([we, fav], shadow_only=False)
    assert we in out
    assert fav in out


def test_weather_shadow_drop_empty_batch() -> None:
    out = apply_weather_shadow_drop([], shadow_only=True)
    assert out == []


def test_weather_shadow_drop_no_weather_signals() -> None:
    fav1 = _sig("favorite_bias", "TICKER_A")
    fav2 = _sig("intraday_asos_high", "TICKER_B")
    out = apply_weather_shadow_drop([fav1, fav2], shadow_only=True)
    assert out == [fav1, fav2]


def test_weather_shadow_drop_log_per_signal(caplog: pytest.LogCaptureFixture) -> None:
    we1 = _sig("weather_edge", "TICKER_X1", edge=0.12, contracts=5)
    we2 = _sig("weather_edge", "TICKER_X2", edge=0.15, contracts=8)
    fav = _sig("favorite_bias", "TICKER_FAV")

    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        out = apply_weather_shadow_drop([we1, we2, fav], shadow_only=True)

    assert we1 not in out
    assert we2 not in out
    assert fav in out

    drop_records = [r for r in caplog.records if "weather_shadow_drop" in r.getMessage()]
    assert len(drop_records) == 2
    messages = [r.getMessage() for r in drop_records]
    assert any("TICKER_X1" in m for m in messages)
    assert any("TICKER_X2" in m for m in messages)


# ---------------------------------------------------------------------------
# Ordering interaction: dropping weather_edge BEFORE favorite_bias deference
# means favorite_bias no longer defers to a weather_edge that will not trade.
# ---------------------------------------------------------------------------


def test_favorite_bias_survives_when_weather_shadow_dropped_first() -> None:
    """When weather_edge is shadow-dropped first, a favorite_bias signal sharing
    its event key is NOT deferred (nothing real to stack against)."""
    we = _sig("weather_edge", "KXHIGHNYC-26MAY30-B72.5")
    fav = _sig("favorite_bias", "KXHIGHNYC-26MAY30-T75")  # same (city,var,date) event

    after_drop = apply_weather_shadow_drop([we, fav], shadow_only=True)
    after_deference = apply_favorite_bias_deference(after_drop, set())

    assert fav in after_deference
    assert we not in after_deference


def test_favorite_bias_deferred_when_weather_live() -> None:
    """Contrast: with weather_edge live (no shadow drop), the existing deference
    behaviour is preserved — favorite_bias yields to the same-event weather_edge."""
    we = _sig("weather_edge", "KXHIGHNYC-26MAY30-B72.5")
    fav = _sig("favorite_bias", "KXHIGHNYC-26MAY30-T75")

    after_drop = apply_weather_shadow_drop([we, fav], shadow_only=False)
    after_deference = apply_favorite_bias_deference(after_drop, set())

    assert we in after_deference
    assert fav not in after_deference


# ---------------------------------------------------------------------------
# Tick-loop wiring: the safety guarantee lives in the call site. A regression
# deleting it would let weather_edge orders reach execute_batch despite
# shadow_only=True.
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
async def test_tick_drops_weather_edge_before_execute_batch() -> None:
    """End-to-end: with weather.shadow_only=True the tick must drop weather_edge
    before execute_batch, while a same-event favorite_bias signal still trades."""
    orch = Orchestrator()
    orch._running = True
    orch._open_event_keys = MagicMock(return_value=set())

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

    we = _sig("weather_edge", "KXHIGHNYC-26MAY30-B72.5", edge=0.10)
    fav = _sig("favorite_bias", "KXHIGHNYC-26MAY30-T75", edge=0.08)
    orch._strategies = [
        _StubStrategy(name="weather_edge", signals=[we]),
        _StubStrategy(name="favorite_bias", signals=[fav]),
    ]

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.strategies.weather.shadow_only = True
    settings.strategies.intraday_emos_high.shadow_only = True
    settings.strategies.intraday_asos_high.shadow_only = True
    orch._settings = settings

    await orch._tick()

    assert engine.execute_batch.await_count == 1
    executed = list(engine.execute_batch.await_args.args[0])
    tickers = [s.market_ticker for s in executed]
    assert "KXHIGHNYC-26MAY30-B72.5" not in tickers, (
        "regression: weather_edge signal reached execute_batch despite shadow_only=True — "
        "drop-hook wiring removed?"
    )
    assert "KXHIGHNYC-26MAY30-T75" in tickers, (
        "favorite_bias must still trade and must NOT defer to a shadowed weather_edge"
    )


@pytest.mark.asyncio
async def test_tick_refreshes_liveness_when_weather_generated_but_shadowed() -> None:
    """SILENT BOT watchdog must not false-alarm under shadow-only: a
    generated-but-shadow-dropped weather_edge signal still proves the strategy
    layer is alive, so _last_signal_time must refresh even though zero orders
    are placed (otherwise a spurious SILENT BOT alert fires ~12h after deploy)."""
    orch = Orchestrator()
    orch._running = True
    orch._open_event_keys = MagicMock(return_value=set())

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

    orch._last_signal_time = time.monotonic() - 100_000.0  # stale: pretend long silence

    we = _sig("weather_edge", "KXHIGHNYC-26MAY30-B72.5")
    orch._strategies = [_StubStrategy(name="weather_edge", signals=[we])]

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.strategies.weather.shadow_only = True
    settings.strategies.intraday_emos_high.shadow_only = True
    settings.strategies.intraday_asos_high.shadow_only = True
    orch._settings = settings

    await orch._tick()

    # All signals shadow-dropped → no orders placed.
    assert engine.execute_batch.await_count == 0
    # ...but the strategy layer DID generate a signal, so liveness must refresh.
    assert time.monotonic() - orch._last_signal_time < 60.0, (
        "regression: _last_signal_time not refreshed on a generated-but-shadowed "
        "tick — SILENT BOT watchdog would false-alarm under shadow-only"
    )
    assert orch._silent_alert_sent is False
