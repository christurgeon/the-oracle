"""The weather.forecast_concurrency knob caps client-level Open-Meteo requests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.settings import WeatherStrategyConfig


def test_forecast_concurrency_defaults_to_two() -> None:
    cfg = WeatherStrategyConfig()
    assert cfg.forecast_concurrency == 2


def test_forecast_concurrency_is_configurable() -> None:
    cfg = WeatherStrategyConfig(forecast_concurrency=4)
    assert cfg.forecast_concurrency == 4


def test_forecast_concurrency_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(forecast_concurrency=0)


def test_prefetch_timeout_defaults() -> None:
    cfg = WeatherStrategyConfig()
    assert cfg.forecast_fetch_timeout_seconds == 15.0
    assert cfg.forecast_prefetch_timeout_seconds == 45.0
    # Overall ceiling must sit well under the systemd WatchdogSec (120s).
    assert cfg.forecast_prefetch_timeout_seconds < 120.0


def test_prefetch_timeouts_reject_nonpositive() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(forecast_fetch_timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(forecast_prefetch_timeout_seconds=0.0)
