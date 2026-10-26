import pytest
from pydantic import ValidationError

from python.settings import FavoriteBiasStrategyConfig, StrategiesConfig


def test_defaults_disabled_and_bounded():
    c = FavoriteBiasStrategyConfig()
    assert c.enabled is False
    assert c.band_low == 0.40 and c.band_high == 0.75
    assert c.kelly_fraction == 0.05
    assert c.max_contracts_per_trade == 25
    assert c.min_volume_fp == 200
    assert c.edge_floor == 0.02
    assert c.lead == 1
    assert c.execution_taker is True


def test_strategies_config_has_favorite_bias():
    s = StrategiesConfig()
    assert s.favorite_bias.enabled is False


def test_band_must_be_ordered():
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(band_low=0.8, band_high=0.4)


def test_max_contracts_bounded():
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_contracts_per_trade=0)


def test_excluded_city_vars_default_empty_and_survives_model_dump():
    """Default [] no-op, AND the field must survive model_dump() — the orchestrator
    passes s.strategies.favorite_bias.model_dump() to the strategy, so a missing
    field would be SILENTLY dropped before reaching the (city,var) gate."""
    c = FavoriteBiasStrategyConfig()
    assert c.excluded_city_vars == []
    d = FavoriteBiasStrategyConfig(excluded_city_vars=["BOS_high"]).model_dump()
    assert d["excluded_city_vars"] == ["BOS_high"]


def test_excluded_city_vars_format_validated():
    """A malformed entry must be rejected (silent no-op on a live safety lever).

    Both halves of the key are checked: the var must be high/low AND the city must
    be a canonical ticker code the parser emits. A city typo like 'NY_high' (the raw
    Kalshi suffix instead of 'NYC') or 'sea_high' (wrong case) would never match the
    runtime key f"{city}_{var}", so it must raise, not pass.
    """
    bad = (
        "BOShigh",  # no separator
        "BOS_High",  # var wrong case
        "BOS_max",  # var not high/low
        "_high",  # empty city
        "NY_high",  # raw Kalshi suffix, not canonical "NYC" — the key slip
        "sea_high",  # city wrong case
        "XXX_high",  # unknown city
        "DEN_low_high",  # multi-underscore garbage
    )
    for b in bad:
        with pytest.raises(ValidationError):
            FavoriteBiasStrategyConfig(excluded_city_vars=[b])
    for ok in ("BOS_high", "NYC_low", "DEN_high"):
        FavoriteBiasStrategyConfig(excluded_city_vars=[ok])  # no raise


def test_max_bets_zero_rejected():
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_bets_per_day_per_event=0)

    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_bets_per_day_per_event=-1)

    # None is the documented disable path — must be accepted
    c = FavoriteBiasStrategyConfig(max_bets_per_day_per_event=None)
    assert c.max_bets_per_day_per_event is None


def test_resting_ttl_seconds_defaults_none() -> None:
    from python.settings import FavoriteBiasStrategyConfig

    cfg = FavoriteBiasStrategyConfig()
    assert cfg.resting_ttl_seconds is None


def test_resting_ttl_seconds_accepts_positive_int() -> None:
    from python.settings import FavoriteBiasStrategyConfig

    cfg = FavoriteBiasStrategyConfig(resting_ttl_seconds=600)
    assert cfg.resting_ttl_seconds == 600


def test_resting_ttl_seconds_rejects_nonpositive() -> None:
    import pytest
    from pydantic import ValidationError

    from python.settings import FavoriteBiasStrategyConfig

    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(resting_ttl_seconds=0)


def test_max_spread_defaults_none_and_survives_model_dump() -> None:
    assert FavoriteBiasStrategyConfig().max_spread is None
    assert FavoriteBiasStrategyConfig(max_spread=0.05).model_dump()["max_spread"] == 0.05


@pytest.mark.parametrize("bad", [0.0, -0.01, 1.0, 1.5])
def test_max_spread_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        FavoriteBiasStrategyConfig(max_spread=bad)
