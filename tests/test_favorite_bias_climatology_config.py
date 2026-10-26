from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.settings import FavoriteBiasStrategyConfig


def test_climatology_config_defaults() -> None:
    c = FavoriteBiasStrategyConfig()
    assert c.climatology_path == "data/calibration/fb_climatology.json"
    assert c.max_strike_climatology_z is None  # guard OFF by default
    assert c.clim_min_n == 15


def test_climatology_z_bounds() -> None:
    assert FavoriteBiasStrategyConfig(max_strike_climatology_z=2.0).max_strike_climatology_z == 2.0
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_strike_climatology_z=0.1)  # < ge=0.5
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_strike_climatology_z=7.0)  # > le=6.0


def test_clim_min_n_bound() -> None:
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(clim_min_n=0)  # < ge=1
