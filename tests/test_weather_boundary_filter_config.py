"""Config validation for weather.boundary_filter_cells."""

import pytest
from pydantic import ValidationError

from python.settings import WeatherStrategyConfig


def test_default_is_empty_dict():
    c = WeatherStrategyConfig()
    assert c.boundary_filter_cells == {}


def test_valid_mia_low_entry_accepted():
    c = WeatherStrategyConfig(boundary_filter_cells={"MIA/low": 0.5})
    assert c.boundary_filter_cells == {"MIA/low": 0.5}


def test_rejects_unknown_cell_format():
    with pytest.raises(ValidationError, match="must be of the form '<CITY>/<var>'"):
        WeatherStrategyConfig(boundary_filter_cells={"MIAlow": 0.5})


def test_rejects_unknown_var():
    with pytest.raises(ValidationError, match="must end in '/high' or '/low'"):
        WeatherStrategyConfig(boundary_filter_cells={"MIA/medium": 0.5})


def test_rejects_negative_threshold():
    with pytest.raises(ValidationError, match="threshold must be >= 0"):
        WeatherStrategyConfig(boundary_filter_cells={"MIA/low": -0.5})


def test_rejects_threshold_too_large():
    with pytest.raises(ValidationError, match="threshold .* implausibly large"):
        WeatherStrategyConfig(boundary_filter_cells={"MIA/low": 100.0})


def test_yaml_null_coerced_to_empty():
    """Matches the existing emos_enabled_cells: null shorthand."""
    c = WeatherStrategyConfig(boundary_filter_cells=None)
    assert c.boundary_filter_cells == {}
