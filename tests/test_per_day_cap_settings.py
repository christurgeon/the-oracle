"""Settings tests for the per-day event cap."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.settings import WeatherStrategyConfig


def test_max_bets_per_day_per_event_defaults_to_1() -> None:
    cfg = WeatherStrategyConfig()
    assert cfg.max_bets_per_day_per_event == 1


def test_max_bets_per_day_per_event_accepts_null() -> None:
    cfg = WeatherStrategyConfig(max_bets_per_day_per_event=None)
    assert cfg.max_bets_per_day_per_event is None


def test_max_bets_per_day_per_event_accepts_two() -> None:
    cfg = WeatherStrategyConfig(max_bets_per_day_per_event=2)
    assert cfg.max_bets_per_day_per_event == 2


def test_max_bets_per_day_per_event_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(max_bets_per_day_per_event=0)


def test_max_bets_per_day_per_event_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(max_bets_per_day_per_event=-1)


def test_legacy_one_bet_per_event_key_rejected() -> None:
    with pytest.raises(ValidationError):
        WeatherStrategyConfig(one_bet_per_event=True)
