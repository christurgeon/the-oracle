"""Tests for the pure apply_favorite_bias_deference function."""

from __future__ import annotations

from python.common import Side
from python.orchestrator import apply_favorite_bias_deference
from python.strategies.base import TradeSignal


def _sig(
    strategy: str, ticker: str, city: str, var: str, fdate: str, edge: float = 0.1
) -> TradeSignal:
    # var maps to metadata["market_type"] — the key per_day_event_cap uses.
    return TradeSignal(
        market_id=ticker,
        market_ticker=ticker,
        model_probability=0.6,
        market_price=0.55,
        confidence=0.6,
        strategy_name=strategy,
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=10,
        effective_edge=edge,
        metadata={"city": city, "market_type": var, "forecast_date": fdate},
    )


def test_fb_dropped_when_weather_edge_signals_same_event() -> None:
    we = _sig("weather_edge", "A", "NYC", "high", "2026-05-20")
    fb = _sig("favorite_bias", "B", "NYC", "high", "2026-05-20")  # same event, diff ticker
    out = apply_favorite_bias_deference([we, fb], open_position_event_keys=set())
    assert we in out and fb not in out


def test_fb_kept_when_no_overlap() -> None:
    we = _sig("weather_edge", "A", "CHI", "low", "2026-05-20")
    fb = _sig("favorite_bias", "B", "NYC", "high", "2026-05-20")
    out = apply_favorite_bias_deference([we, fb], open_position_event_keys=set())
    assert we in out and fb in out


def test_fb_dropped_when_open_weather_edge_position_on_event() -> None:
    fb = _sig("favorite_bias", "B", "NYC", "high", "2026-05-20")
    out = apply_favorite_bias_deference(
        [fb], open_position_event_keys={("NYC", "high", "2026-05-20")}
    )
    assert fb not in out


def test_no_fb_signals_returns_list_unchanged() -> None:
    """When there are no favorite_bias signals, the function is a no-op."""
    we1 = _sig("weather_edge", "A", "NYC", "high", "2026-05-20")
    we2 = _sig("weather_edge", "C", "CHI", "low", "2026-05-21")
    original = [we1, we2]
    out = apply_favorite_bias_deference(original, open_position_event_keys=set())
    assert out == original


def test_empty_metadata_fb_not_dropped_by_empty_metadata_weather_edge() -> None:
    """A weather_edge signal with empty metadata produces key ("","","") which must NOT
    cause a favorite_bias signal with the same empty key to be wrongly deferred.

    _event_key_from_signal returns ("","","") for absent city/market_type/forecast_date.
    Before the guard, the empty key would land in `occupied`, causing any FB signal also
    with empty metadata to be silently dropped — a self-collision on malformed signals.
    """
    we_empty = TradeSignal(
        market_id="X",
        market_ticker="X",
        model_probability=0.6,
        market_price=0.55,
        confidence=0.6,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=10,
        effective_edge=0.1,
        metadata={},  # no city/market_type/forecast_date -> ("","","")
    )
    fb_empty = TradeSignal(
        market_id="Y",
        market_ticker="Y",
        model_probability=0.6,
        market_price=0.55,
        confidence=0.6,
        strategy_name="favorite_bias",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=10,
        effective_edge=0.1,
        metadata={},  # same empty key
    )
    out = apply_favorite_bias_deference([we_empty, fb_empty], open_position_event_keys=set())
    # Both must survive: the empty key must NOT be in occupied.
    assert we_empty in out
    assert fb_empty in out
