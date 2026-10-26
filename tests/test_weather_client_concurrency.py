"""Regression tests for the client-level Open-Meteo concurrency cap.

Root cause of the 2026-06 watchdog crash-loop: the weather strategy burst ~8
concurrent requests at Open-Meteo's free endpoint (4 prefetch keys x 2 models,
plus a second NBM semaphore), Open-Meteo returned HTTP 429, retries amplified
the burst, and the tick blew past WatchdogSec -> SIGABRT.

The authoritative fix is a ``WeatherClient``-level ``asyncio.Semaphore`` that
bounds EVERY outbound Open-Meteo HTTP request regardless of which call path
(ensemble inner-gather, multi-model, NBM) issued it. These tests pin that the
cap holds across call paths.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest

import python.data.weather as weather_mod
from python.common import DailyVariable
from python.data.weather import WeatherClient


class _ConcurrencyTracker:
    """Counts overlapping calls to the patched HTTP layer."""

    def __init__(self, hold_seconds: float = 0.02) -> None:
        self.hold_seconds = hold_seconds
        self.in_flight = 0
        self.max_in_flight = 0

    async def fake_get_json(self, _client: Any, _url: str, **kwargs: Any) -> dict[str, Any]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.hold_seconds)
            params = kwargs.get("params", {})
            daily = str(params.get("daily", DailyVariable.TEMP_MAX))
            date_str = str(params.get("start_date"))
            # Minimal valid ensemble payload: two members for the requested date.
            return {
                "daily": {
                    "time": [date_str],
                    f"{daily}_member01": [72.0],
                    f"{daily}_member02": [73.0],
                }
            }
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
async def test_client_semaphore_caps_concurrent_ensemble_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """8 concurrent ensemble fetches must never exceed forecast_concurrency."""
    tracker = _ConcurrencyTracker()
    monkeypatch.setattr(weather_mod, "get_json", tracker.fake_get_json)

    cap = 2
    fc_date = date(2026, 6, 6)
    cities = ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "PHIL"]

    async with WeatherClient(forecast_concurrency=cap) as client:
        await asyncio.gather(
            *(
                client.get_ensemble_forecast(city, fc_date, DailyVariable.TEMP_MAX)
                for city in cities
            )
        )

    assert tracker.max_in_flight <= cap, (
        f"max in-flight {tracker.max_in_flight} exceeded cap {cap}"
    )
    # Sanity: the burst really was concurrent (otherwise the test proves nothing).
    assert tracker.max_in_flight >= 1


@pytest.mark.asyncio
async def test_client_semaphore_spans_ensemble_and_nbm_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is shared across the ensemble AND NBM request paths."""
    tracker = _ConcurrencyTracker()
    monkeypatch.setattr(weather_mod, "get_json", tracker.fake_get_json)

    # NBM path needs a non-ensemble response shape (single value, not members).
    async def fake_get_json_dispatch(client: Any, url: str, **kwargs: Any) -> dict[str, Any]:
        if "ensemble" in url:
            return await tracker.fake_get_json(client, url, **kwargs)
        # NBM forecast endpoint
        tracker.in_flight += 1
        tracker.max_in_flight = max(tracker.max_in_flight, tracker.in_flight)
        try:
            await asyncio.sleep(tracker.hold_seconds)
            params = kwargs.get("params", {})
            daily = str(params.get("daily", DailyVariable.TEMP_MAX))
            date_str = str(params.get("start_date"))
            return {"daily": {"time": [date_str], daily: [80.0]}}
        finally:
            tracker.in_flight -= 1

    monkeypatch.setattr(weather_mod, "get_json", fake_get_json_dispatch)

    cap = 2
    fc_date = date(2026, 6, 6)

    async with WeatherClient(forecast_concurrency=cap) as client:
        ensemble_calls = [
            client.get_ensemble_forecast(city, fc_date, DailyVariable.TEMP_MAX)
            for city in ["NYC", "CHI", "MIA", "LAX"]
        ]
        nbm_calls = [
            client.get_nbm_point_forecast(city, fc_date, DailyVariable.TEMP_MAX)
            for city in ["NYC", "CHI", "MIA", "LAX"]
        ]
        await asyncio.gather(*ensemble_calls, *nbm_calls)

    assert tracker.max_in_flight <= cap, (
        f"max in-flight {tracker.max_in_flight} exceeded shared cap {cap}"
    )


@pytest.mark.asyncio
async def test_default_forecast_concurrency_is_two() -> None:
    """Default cap mirrors orderbook_concurrency (2)."""
    client = WeatherClient()
    assert client.forecast_concurrency == 2
