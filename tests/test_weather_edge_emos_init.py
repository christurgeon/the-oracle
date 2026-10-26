"""Tests for WeatherEdgeStrategy.__init__ — notifier kwarg + EMOS state seed (PR-2 / C5)."""
from __future__ import annotations

from python.strategies.weather_edge import WeatherEdgeStrategy


def test_init_seeds_emos_state_to_none():
    """Fresh strategy has EMOS state seeded to safe defaults — identity fallback
    until async setup() runs."""
    strat = WeatherEdgeStrategy(
        config={},
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
    )
    assert strat.emos_table is None
    assert strat._emos_log_dedup == {}
    assert strat._forecasts_raw_writer is None
    assert strat._notifier is None  # default


def test_init_accepts_notifier_kwarg():
    """Passing a notifier instance stores it on the strategy for setup()-time
    use (EMOS load ERROR-level alerts)."""
    sentinel = object()
    strat = WeatherEdgeStrategy(
        config={},
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
        notifier=sentinel,  # type: ignore[arg-type]
    )
    assert strat._notifier is sentinel
