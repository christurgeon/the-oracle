"""The ensemble prefetch must be time-bounded so a slow/429-storming Open-Meteo
cannot let a single tick blow past the systemd WatchdogSec (-> SIGABRT).

Two layers, both fail-soft (a cell with no forecast is simply skipped this tick):
  * per-fetch  ``asyncio.wait_for`` (forecast_fetch_timeout_seconds) — one slow
    cell degrades gracefully without starving the others.
  * overall    ``asyncio.wait_for`` (forecast_prefetch_timeout_seconds) — a hard
    ceiling on the whole fan-out, well under WatchdogSec.

Mirrors the per-fetch wait_for already present in ``_prefetch_nbm``.
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from python.common import DailyVariable, Direction
from python.strategies.weather_edge import WeatherEdgeStrategy
from python.strategies.weather_parsing import ParsedMarket, ScanCandidate


def _candidate(city: str, fc_date: date) -> ScanCandidate:
    parsed = ParsedMarket(
        ticker=f"KXHIGH{city}-26JUN06-T70",
        city=city,
        forecast_date=fc_date,
        daily_variable=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
        cap_strike=None,
    )
    return ScanCandidate(
        parsed=parsed,
        market={"event_ticker": f"KXHIGH{city}-26JUN06"},
        price_info=None,
    )


class _HangingWeather:
    """A weather client whose every fetch hangs forever."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_multi_model_forecast(self, **_kwargs: object) -> dict:
        self.calls += 1
        await asyncio.sleep(3600)
        return {}  # pragma: no cover


def _strategy(config: dict) -> WeatherEdgeStrategy:
    strat = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
    strat._config = config  # type: ignore[attr-defined]
    strat._weather = _HangingWeather()  # type: ignore[attr-defined]
    strat._bias_table = None  # type: ignore[attr-defined]
    return strat


@pytest.mark.asyncio
async def test_per_fetch_timeout_fails_soft() -> None:
    """A hanging fetch is abandoned after forecast_fetch_timeout_seconds and the
    cell is omitted from the forecast cache rather than hanging the tick."""
    strat = _strategy(
        {
            "ensemble_models": ["gfs_seamless", "ecmwf_ifs025"],
            "forecast_fetch_timeout_seconds": 0.05,
            "forecast_prefetch_timeout_seconds": 10.0,
        }
    )
    markets = [_candidate("NYC", date(2026, 6, 6))]

    forecast_cache, _bias, _stds, _medians = await asyncio.wait_for(
        strat._prefetch_forecasts(markets), timeout=2.0
    )

    assert forecast_cache == {}  # fail-soft: no forecast for the hung cell


@pytest.mark.asyncio
async def test_overall_prefetch_timeout_bounds_fanout() -> None:
    """Even with a generous per-fetch timeout, the overall fan-out is bounded by
    forecast_prefetch_timeout_seconds so the weather phase can't run away."""
    strat = _strategy(
        {
            "ensemble_models": ["gfs_seamless", "ecmwf_ifs025"],
            "forecast_fetch_timeout_seconds": 100.0,
            "forecast_prefetch_timeout_seconds": 0.05,
        }
    )
    markets = [
        _candidate("NYC", date(2026, 6, 6)),
        _candidate("MIA", date(2026, 6, 6)),
        _candidate("LAX", date(2026, 6, 6)),
    ]

    forecast_cache, _bias, _stds, _medians = await asyncio.wait_for(
        strat._prefetch_forecasts(markets), timeout=2.0
    )

    assert forecast_cache == {}
