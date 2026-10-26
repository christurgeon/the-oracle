"""Invariant tests for the city/series keyspace.

The series→(city, var) map has a single source of truth
(``ticker_dates.SERIES_PREFIX_TO_CITY_VAR``); these guard the remaining
independent maps that drive data collection + reporting — NWS CLI locations
and forecast coordinates — from drifting out of sync with it.
"""

from __future__ import annotations

from python.analytics.attribution import _infer_city
from python.common import DailyVariable
from python.data.nws import CITY_TO_CLI_LOCATION
from python.data.weather import CITY_COORDS
from python.strategies.weather_edge import WEATHER_SERIES
from python.utils.ticker_dates import SERIES_PREFIX_TO_CITY_VAR
from scripts.fetch_shadow_settlements import get_city as fetch_get_city
from scripts.shadow_dashboard import get_city as dash_get_city

# DailyVariable -> ticker_dates var string
_VAR_STR = {DailyVariable.TEMP_MAX: "high", DailyVariable.TEMP_MIN: "low"}

# The 11 new cities and their expected NWS CLI settlement codes (verified live
# against Kalshi settlement_sources + api.weather.gov CLI products).
_NEW_CITY_CLI = {
    "ATL": "ATL", "DAL": "DFW", "HOU": "HOU", "LV": "LAS", "MIN": "MSP",
    "NOLA": "MSY", "OKC": "OKC", "PHX": "PHX", "SATX": "SAT", "SFO": "SFO",
    "DC": "DCA",
}
_NEW_SERIES = [f"KXHIGHT{c}" for c in _NEW_CITY_CLI] + [
    f"KXLOWT{c}" for c in _NEW_CITY_CLI
]


def test_weather_series_matches_ticker_dates() -> None:
    """WEATHER_SERIES and SERIES_PREFIX_TO_CITY_VAR must cover the same series
    with consistent (city, var)."""
    assert set(WEATHER_SERIES) == set(SERIES_PREFIX_TO_CITY_VAR)
    for series, (city, var) in WEATHER_SERIES.items():
        td_city, td_var = SERIES_PREFIX_TO_CITY_VAR[series]
        assert td_city == city, series
        assert td_var == _VAR_STR[var], series


def test_reporting_consumers_resolve_every_weather_series() -> None:
    """Every WEATHER_SERIES series resolves to a real city through all three
    reporting consumers."""
    for series, (expected_city, _) in WEATHER_SERIES.items():
        row = {"series_ticker": series, "market_ticker": f"{series}-26JUN09-T80"}
        assert fetch_get_city(row) == expected_city, series
        assert dash_get_city(row) == expected_city, series
        assert _infer_city(row) == expected_city, series


def test_reporting_consumers_reject_unknown_series() -> None:
    """A non-weather ticker must not resolve to a city."""
    row = {"market_ticker": "KXINX-26JUN09-T5800"}
    assert fetch_get_city(row) == "???"
    assert dash_get_city(row) == "???"
    assert _infer_city(row) is None


def test_collection_maps_are_consistent() -> None:
    """Every series' city is collectable: present in CITY_TO_CLI_LOCATION and
    CITY_COORDS; CLI set is a subset of COORDS."""
    series_cities = {city for city, _ in WEATHER_SERIES.values()}
    assert series_cities <= set(CITY_TO_CLI_LOCATION)
    assert set(CITY_TO_CLI_LOCATION) <= set(CITY_COORDS)


def test_new_cities_present_with_expected_cli_codes() -> None:
    """The 11 new cities are wired with their verified CLI settlement codes and
    both series."""
    for city, cli in _NEW_CITY_CLI.items():
        assert CITY_TO_CLI_LOCATION.get(city) == cli, city
        assert city in CITY_COORDS, city
    for series in _NEW_SERIES:
        assert series in WEATHER_SERIES, series
