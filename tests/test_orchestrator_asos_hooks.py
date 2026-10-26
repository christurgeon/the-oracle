"""Tests for the intraday_asos_high orchestrator hooks.

Covers:
  - ``apply_intraday_asos_deference`` (pure function, Task 6.1)
  - ``apply_intraday_asos_shadow_drop`` (pure function, Task 6.1)
  - Task 0.4 parity: open-position key round-trip (_open_event_keys vs signal key)
  - Tick-loop zero-order guarantee: intraday_asos_high signals must NOT reach
    execute_batch when shadow_only=True (Task 6.2 safety-critical wiring test)
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import Side
from python.orchestrator import (
    Orchestrator,
    apply_intraday_asos_deference,
    apply_intraday_asos_shadow_drop,
)
from python.strategies.base import Strategy, TradeSignal
from python.strategies.per_day_event_cap import (
    _event_key_from_signal,
    _parse_event_key_from_ticker,
)

# ---------------------------------------------------------------------------
# Signal factory helpers
# ---------------------------------------------------------------------------


def _sig(
    strategy: str,
    ticker: str,
    city: str = "MIA",
    var: str = "high",
    fdate: str = "2026-06-06",
    edge: float = 0.10,
    contracts: int = 10,
) -> TradeSignal:
    return TradeSignal(
        market_id=ticker,
        market_ticker=ticker,
        model_probability=0.65,
        market_price=0.55,
        confidence=1.0,
        strategy_name=strategy,
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=contracts,
        effective_edge=edge,
        metadata={"city": city, "market_type": var, "forecast_date": fdate},
    )


# ---------------------------------------------------------------------------
# apply_intraday_asos_shadow_drop — pure function tests
# ---------------------------------------------------------------------------


def test_asos_shadow_drop_removes_asos_signals() -> None:
    """shadow_only=True: intraday_asos_high signals are removed; others kept."""
    we = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75")
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-B89")
    out = apply_intraday_asos_shadow_drop([we, asos], shadow_only=True)
    assert we in out
    assert asos not in out


def test_asos_shadow_drop_passthrough_when_disabled() -> None:
    """shadow_only=False: all signals pass through unchanged."""
    we = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75")
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-B89")
    out = apply_intraday_asos_shadow_drop([we, asos], shadow_only=False)
    assert we in out
    assert asos in out


def test_asos_shadow_drop_empty_batch() -> None:
    out = apply_intraday_asos_shadow_drop([], shadow_only=True)
    assert out == []


def test_asos_shadow_drop_no_asos_signals() -> None:
    """No intraday_asos_high signals — batch returned unchanged."""
    we1 = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75")
    we2 = _sig("weather_edge", "KXHIGHCHI-26JUN07-B85")
    out = apply_intraday_asos_shadow_drop([we1, we2], shadow_only=True)
    assert out == [we1, we2]


def test_asos_shadow_drop_multiple_signals(caplog: pytest.LogCaptureFixture) -> None:
    """shadow_only=True: each dropped signal emits a structured log line."""
    asos1 = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", edge=0.12, contracts=5)
    asos2 = _sig("intraday_asos_high", "KXHIGHAUS-26JUN06-T95", edge=0.15, contracts=8)
    we = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75")

    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        out = apply_intraday_asos_shadow_drop([asos1, asos2, we], shadow_only=True)

    assert asos1 not in out
    assert asos2 not in out
    assert we in out

    drop_records = [
        r for r in caplog.records if "intraday_asos_shadow_drop" in r.getMessage()
    ]
    assert len(drop_records) == 2
    messages = [r.getMessage() for r in drop_records]
    assert any("KXHIGHMIA-26JUN06-T89" in m for m in messages)
    assert any("KXHIGHAUS-26JUN06-T95" in m for m in messages)


# ---------------------------------------------------------------------------
# apply_intraday_asos_deference — pure function tests
# ---------------------------------------------------------------------------


def test_asos_dropped_when_weather_edge_signals_same_event() -> None:
    """ASOS signal sharing a (city, var, date) with a weather_edge signal is dropped."""
    we = _sig("weather_edge", "KXHIGHMIA-26JUN06-B89", city="MIA", fdate="2026-06-06")
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", city="MIA", fdate="2026-06-06")
    out = apply_intraday_asos_deference([we, asos], open_position_event_keys=set())
    assert we in out
    assert asos not in out


def test_asos_kept_when_no_overlap() -> None:
    """ASOS signal on a different (city, var, date) is NOT dropped."""
    we = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75", city="NYC", fdate="2026-06-07")
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", city="MIA", fdate="2026-06-06")
    out = apply_intraday_asos_deference([we, asos], open_position_event_keys=set())
    assert we in out
    assert asos in out


def test_asos_dropped_when_open_position_on_event() -> None:
    """ASOS signal sharing event key with an open position is dropped."""
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", city="MIA", fdate="2026-06-06")
    out = apply_intraday_asos_deference(
        [asos],
        open_position_event_keys={("MIA", "high", "2026-06-06")},
    )
    assert asos not in out


def test_asos_not_dropped_when_open_position_different_event() -> None:
    """Open position on a different event does NOT drop the ASOS signal."""
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", city="MIA", fdate="2026-06-06")
    out = apply_intraday_asos_deference(
        [asos],
        open_position_event_keys={("NYC", "high", "2026-06-06")},
    )
    assert asos in out


def test_asos_no_signals_noop() -> None:
    """No intraday_asos_high signals — returns batch unchanged."""
    we1 = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75", city="NYC", var="high", fdate="2026-06-07")
    we2 = _sig("weather_edge", "KXHIGHCHI-26JUN07-B85", city="CHI", var="high", fdate="2026-06-07")
    original = [we1, we2]
    out = apply_intraday_asos_deference(original, open_position_event_keys=set())
    assert out == original


def test_asos_empty_metadata_guard() -> None:
    """Empty-metadata weather_edge signal must NOT cause empty-metadata ASOS signal to be dropped.

    _event_key_from_signal returns ("","","") for absent city/market_type/forecast_date.
    Before the empty-key guard, the empty key would land in `occupied`, causing any
    ASOS signal also with empty metadata to be wrongly deferred.
    """
    we_empty = TradeSignal(
        market_id="X",
        market_ticker="X",
        model_probability=0.6,
        market_price=0.55,
        confidence=1.0,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=10,
        effective_edge=0.1,
        metadata={},
    )
    asos_empty = TradeSignal(
        market_id="Y",
        market_ticker="Y",
        model_probability=0.6,
        market_price=0.55,
        confidence=1.0,
        strategy_name="intraday_asos_high",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=10,
        effective_edge=0.1,
        metadata={},
    )
    out = apply_intraday_asos_deference([we_empty, asos_empty], open_position_event_keys=set())
    # Both must survive: the empty key must NOT be in occupied.
    assert we_empty in out
    assert asos_empty in out


# ---------------------------------------------------------------------------
# Task 0.4 parity test: open-position key round-trip
# Prove that _parse_event_key_from_ticker (used by _open_event_keys) produces
# a key that matches _event_key_from_signal for a realistic intraday_asos_high
# TradeSignal. This is the precondition that makes the deference open-position
# arm meaningful.
# ---------------------------------------------------------------------------


def test_open_position_key_matches_asos_signal_key() -> None:
    """Parity: _parse_event_key_from_ticker(ticker) == _event_key_from_signal(signal).

    Builds a realistic open-position ticker (KXHIGHMIA-26JUN06-B89.5) and a
    matching intraday_asos_high TradeSignal. Asserts that the key derived by
    _parse_event_key_from_ticker (the path _open_event_keys uses) equals the
    key derived by _event_key_from_signal (the path apply_intraday_asos_deference
    uses). Proves the deference open-position arm can actually match and drop.
    """
    # A daily KXHIGH market ticker in the format used by Kalshi.
    ticker = "KXHIGHMIA-26JUN06-B89.5"

    # Key as derived from the open-position ticker (the _open_event_keys path).
    ticker_key = _parse_event_key_from_ticker(ticker)
    assert ticker_key is not None, f"_parse_event_key_from_ticker({ticker!r}) returned None"

    # A TradeSignal from intraday_asos_high with matching metadata.
    # metadata["city"], ["market_type"], and ["forecast_date"] are the three
    # components of the EventKey — must match exactly.
    city, var, fdate = ticker_key
    signal = _sig(
        "intraday_asos_high",
        "KXHIGHMIA-26JUN06-T89",  # intraday-style ticker for the same market
        city=city,
        var=var,
        fdate=fdate,
    )

    signal_key = _event_key_from_signal(signal)

    assert ticker_key == signal_key, (
        f"Key mismatch — ticker_key={ticker_key!r}, signal_key={signal_key!r}. "
        f"The ASOS deference open-position arm cannot match; the contract is broken."
    )

    # Also verify the keys intersect: placing ticker_key in open_position_event_keys
    # causes the signal to be deferred.
    out = apply_intraday_asos_deference([signal], open_position_event_keys={ticker_key})
    assert signal not in out, (
        "ASOS signal was NOT dropped despite its event key being in open_position_event_keys — "
        "deference open-position arm is broken."
    )


# ---------------------------------------------------------------------------
# Tick-loop zero-order test (Task 6.2 safety guarantee)
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
async def test_tick_drops_intraday_asos_before_execute_batch() -> None:
    """End-to-end: the orchestrator's tick loop must call the ASOS shadow-drop hook
    so intraday_asos_high signals do NOT reach execute_batch when shadow_only=True.

    This catches a regression where someone deletes the call site in _tick.
    Mirrors test_tick_drops_intraday_before_execute_batch in test_orchestrator_shadow_drop.py.
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

    # intraday_asos_high signal that must be suppressed.
    asos = _sig("intraday_asos_high", "KXHIGHMIA-26JUN06-T89", edge=0.12)
    # weather_edge signal that must pass through.
    we = _sig("weather_edge", "KXHIGHNYC-26JUN07-B75", city="NYC", fdate="2026-06-07", edge=0.10)
    orch._strategies = [
        _StubStrategy(name="intraday_asos_high", signals=[asos]),
        _StubStrategy(name="weather_edge", signals=[we]),
    ]

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.strategies.weather.shadow_only = False  # weather_edge live (passthrough)
    settings.strategies.intraday_emos_high.shadow_only = False  # don't interfere
    settings.strategies.intraday_asos_high.shadow_only = True   # ASOS in shadow-only
    orch._settings = settings

    await orch._tick()

    assert engine.execute_batch.await_count == 1
    executed = list(engine.execute_batch.await_args.args[0])
    tickers = [s.market_ticker for s in executed]
    assert "KXHIGHMIA-26JUN06-T89" not in tickers, (
        "regression: intraday_asos_high signal reached execute_batch despite "
        "shadow_only=True — ASOS drop-hook wiring removed?"
    )
    assert "KXHIGHNYC-26JUN07-B75" in tickers, (
        "weather_edge signal must pass through the ASOS shadow drop hook"
    )
