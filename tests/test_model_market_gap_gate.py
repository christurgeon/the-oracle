"""Tests for the MODEL_MARKET_GAP_INTERIM near-term gap gate (Fix G)
and DEN_high / SEA_high excluded_city_vars additions (Fix E).

TDD: these tests were written BEFORE the production code. They should FAIL
until evaluate_market_pure / _EvalConfig gain the gap gate and the config
gains the two new exclusions.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import et_today
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import WeatherEdgeStrategy

# ---------------------------------------------------------------------------
# Helpers — reuse the pattern from test_weather_signals.py
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def _event_ticker_for(d: date, city: str = "NY") -> str:
    return f"KXHIGH{city}-{d.strftime('%y')}{_MONTH_MAP[d.month]}{d.strftime('%d')}"


def _make_market(
    *,
    forecast_date: date,
    city: str = "NY",
    strike_type: str = "greater",
    floor_strike: float = 55.0,
    cap_strike: float | None = None,
    yes_bid_dollars: float = 0.53,
    yes_ask_dollars: float = 0.59,
    volume: int = 100,
    open_interest: int = 20,
) -> dict[str, Any]:
    et = _event_ticker_for(forecast_date, city)
    tk = f"{et}-T{floor_strike:.0f}"
    market: dict[str, Any] = {
        "id": "",
        "ticker": tk,
        "event_ticker": et,
        "strike_type": strike_type,
        "volume_fp": str(float(volume)),
        "open_interest_fp": str(float(open_interest)),
        "yes_bid_dollars": str(yes_bid_dollars),
        "yes_ask_dollars": str(yes_ask_dollars),
        "yes_bid_size_fp": "50.0",
        "yes_ask_size_fp": "50.0",
        "close_time": "2099-12-31T23:59:59Z",
    }
    if floor_strike is not None:
        market["floor_strike"] = floor_strike
    if cap_strike is not None:
        market["cap_strike"] = cap_strike
    return market


def _make_forecast(city: str, forecast_date: date, temps: list[float]) -> EnsembleForecast:
    return EnsembleForecast(city=city, forecast_date=forecast_date, temperatures=temps)


def _make_strategy(
    *,
    max_model_market_gap_near_term: float | None = 0.45,
    extra_config: dict[str, Any] | None = None,
) -> tuple[WeatherEdgeStrategy, AsyncMock, AsyncMock]:
    """Build a minimal WeatherEdgeStrategy with the gap gate configured."""
    kalshi = AsyncMock()
    kalshi.iter_all_markets.return_value = ([], False)
    weather = AsyncMock()
    weather.compute_model_probability = MagicMock()

    async def _multi_model_side_effect(city, forecast_date, daily_variable, models=None):
        try:
            fc = await weather.get_ensemble_forecast(city, forecast_date, daily_variable)
            return {"gfs_seamless": fc}
        except Exception:
            return {}

    weather.get_multi_model_forecast = AsyncMock(side_effect=_multi_model_side_effect)

    config: dict[str, Any] = {
        "enabled": True,
        "edge_threshold": 0.07,
        "max_edge": 0.90,  # high enough to not interfere with gap tests
        "max_spread": 0.15,
        "spread_skill_sensitivity": 0.0,
        "cities": ["NYC", "DEN", "SEA"],
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        "orderbook_depth_check": False,
        "calibration_type": "none",
        "candidate_audit_log": False,
        "eval_audit_log": False,
        "scan_funnel_log": False,
    }
    if max_model_market_gap_near_term is not None:
        config["max_model_market_gap_near_term"] = max_model_market_gap_near_term
    if extra_config:
        config.update(extra_config)

    strategy = WeatherEdgeStrategy(
        config=config,
        kalshi_client=kalshi,
        weather_client=weather,
        bankroll=10_000.0,
        market_filter={
            "min_volume": 50,
            "min_open_interest": 5,
            "min_price": 0.03,
            "max_price": 0.97,
            "max_orderbook_take_pct": 0.50,
        },
    )
    return strategy, kalshi, weather


# ===========================================================================
# Fix G — MODEL_MARKET_GAP_INTERIM near-term gap gate
# ===========================================================================


class TestModelMarketGapGate:
    """
    The gap gate rejects markets where |model_prob - market_mid| > threshold
    AND lead_days <= 2. Catches stale GFS on short-horizon trades.
    """

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio
        monkeypatch.setattr(_asyncio, "sleep", AsyncMock(return_value=None))

    @pytest.mark.asyncio
    async def test_gate_blocks_large_gap_near_term(self):
        """model_prob=0.08, market_mid=0.56 → gap=0.48 > 0.45, lead=1 → BLOCKED."""
        tomorrow = et_today() + timedelta(days=1)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # market_mid = (0.53 + 0.59) / 2 = 0.56
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.53,
            yes_ask_dollars=0.59,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 8% of 100 members above threshold → model_prob ≈ 0.08
        temps = [82.0] * 8 + [50.0] * 92
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        # Must be blocked by MODEL_MARKET_GAP_INTERIM
        assert len(signals) == 0, (
            f"Expected gap gate to block; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_does_not_block_small_gap_near_term(self):
        """model_prob=0.08, market_mid=0.52 → gap=0.44 < 0.45, lead=1 → PASSES gate."""
        tomorrow = et_today() + timedelta(days=1)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # market_mid = (0.50 + 0.54) / 2 = 0.52
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.50,
            yes_ask_dollars=0.54,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 8% of 100 members above threshold → model_prob ≈ 0.08; gap = 0.44 < 0.45
        temps = [82.0] * 8 + [50.0] * 92
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        # gap = |0.08 - 0.52| = 0.44 < 0.45 → gate should NOT block.
        # With edge_threshold=0.01, NO-side: exec_price = 1 - yes_bid = 0.50,
        # raw_edge = (1 - 0.08) - 0.50 = 0.42 > 0.01 → signal should fire.
        strategy._config["edge_threshold"] = 0.01  # type: ignore[index]
        signals = await strategy.generate_signals()
        assert len(signals) >= 1, (
            f"gap=0.44 < 0.45 should NOT be blocked by gap gate; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_does_not_block_large_gap_far_term(self):
        """Same large gap (0.52) but lead_days=3 → gate DOES NOT apply."""
        three_days = et_today() + timedelta(days=3)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # market_mid = (0.53 + 0.59) / 2 = 0.56; model_prob = 0.08 → gap = 0.48
        market = _make_market(
            forecast_date=three_days,
            yes_bid_dollars=0.53,
            yes_ask_dollars=0.59,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 8% above threshold
        temps = [82.0] * 8 + [50.0] * 92
        forecast = _make_forecast("NYC", three_days, temps)
        weather.get_ensemble_forecast.return_value = forecast

        # With edge_threshold low enough that only the gap gate would block
        strategy._config["edge_threshold"] = 0.01  # type: ignore[index]
        signals = await strategy.generate_signals()

        # For a far-term market (lead=3), gap gate must NOT reject.
        # Signal may or may not be produced (other gates may filter), but it should
        # NOT be zero solely because of the MODEL_MARKET_GAP_INTERIM gate.
        # Specifically: a near-term test with the same conditions must give 0 (blocked
        # by gap) while this far-term test gives a non-zero result OR passes through
        # to other gates. We test this by confirming the far-term scenario is NOT
        # blocked by the gap gate when a matching near-term scenario IS blocked.
        #
        # We have already tested near-term blocking above. So here we just assert
        # the result is truthy (at least 1 signal, since edge would be sufficient
        # and no other filter blocks this specific config).
        # Actually with model=0.08 and exec_price=0.59 (ask, YES direction),
        # raw_edge = 0.59 - 0.08 = 0.51... wait, that's NO side.
        # model < mid=0.56, so we're selling YES (NO side).
        # exec_price for NO = (1 - yes_bid) = 1 - 0.53 = 0.47
        # model_prob=0.08 < mid=0.56 → NO, edge = (1 - model_prob) - exec_NO = 0.92 - 0.47 = 0.45
        # That would be >max_edge=0.90? No, max_edge=0.90 and eff_edge ≈ 0.45 < 0.90 → passes
        # edge_threshold=0.01 → passes. Should produce a signal.
        assert len(signals) >= 1, (
            f"Far-term market should NOT be blocked by gap gate; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_blocks_large_gap_today_plus_two(self):
        """lead_days=2 is still within near-term window → gate applies."""
        two_days = et_today() + timedelta(days=2)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # market_mid = 0.56, model_prob = 0.08, gap = 0.48 > 0.45
        market = _make_market(
            forecast_date=two_days,
            yes_bid_dollars=0.53,
            yes_ask_dollars=0.59,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        temps = [82.0] * 8 + [50.0] * 92
        forecast = _make_forecast("NYC", two_days, temps)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0, (
            f"lead_days=2 should still trigger gap gate; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_blocks_symmetric_gap_model_high_market_low(self):
        """model_prob=0.92, market_mid=0.44 → gap=0.48 > 0.45, lead=1 → BLOCKED."""
        tomorrow = et_today() + timedelta(days=1)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # market_mid = (0.41 + 0.47) / 2 = 0.44
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.41,
            yes_ask_dollars=0.47,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 92% of members above threshold → model_prob ≈ 0.92; gap = |0.92 - 0.44| = 0.48
        temps = [82.0] * 92 + [50.0] * 8
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0, (
            f"High model_prob / low market gap should also trigger gate; got {len(signals)}"
        )

    @pytest.mark.asyncio
    async def test_gate_does_not_block_at_exact_threshold(self):
        """gap == threshold exactly (0.45) should NOT block — gate uses strict >."""
        tomorrow = et_today() + timedelta(days=1)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # 7 of 100 members above strike → model_prob = 0.07
        # market_mid = (0.50 + 0.54) / 2 = 0.52 → gap = |0.07 - 0.52| = 0.45 exactly
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.50,
            yes_ask_dollars=0.54,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        temps = [82.0] * 7 + [50.0] * 93
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        strategy._config["edge_threshold"] = 0.01  # type: ignore[index]
        signals = await strategy.generate_signals()
        # gap == 0.45 is NOT strictly > 0.45 → gate must not block
        # NO-side edge = (1 - 0.07) - (1 - 0.50) = 0.93 - 0.50 = 0.43 > 0.01
        assert len(signals) >= 1, (
            f"gap==threshold (0.45) should NOT be blocked (gate is strict >); got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_blocks_just_above_threshold(self):
        """gap just above 0.45 should block — confirms strict > boundary."""
        tomorrow = et_today() + timedelta(days=1)
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=0.45)

        # 7 of 100 members above strike → model_prob = 0.07
        # market_mid = (0.51 + 0.55) / 2 = 0.53 → gap = |0.07 - 0.53| = 0.46 > 0.45
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.51,
            yes_ask_dollars=0.55,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        temps = [82.0] * 7 + [50.0] * 93
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0, (
            f"gap=0.46 > 0.45 on lead=1 should be blocked by gap gate; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_gate_disabled_when_param_absent(self):
        """When max_model_market_gap_near_term is not set, gate is inactive."""
        tomorrow = et_today() + timedelta(days=1)
        # Build strategy WITHOUT the gap param
        strategy, kalshi, weather = _make_strategy(max_model_market_gap_near_term=None)

        # Same setup as blocking test: gap=0.48 > 0.45, lead=1
        market = _make_market(
            forecast_date=tomorrow,
            yes_bid_dollars=0.53,
            yes_ask_dollars=0.59,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # model_prob=0.08; gap=0.48 → would block if param were active
        temps = [82.0] * 8 + [50.0] * 92
        forecast = _make_forecast("NYC", tomorrow, temps)
        weather.get_ensemble_forecast.return_value = forecast

        strategy._config["edge_threshold"] = 0.01  # type: ignore[index]
        signals = await strategy.generate_signals()

        # Without the gate, the same gap=0.48 / lead=1 combo that blocks in
        # test_gate_blocks_large_gap_near_term must NOT block here.
        # NO-side edge = (1 - 0.08) - (1 - 0.53) = 0.92 - 0.47 = 0.45 > 0.01.
        assert len(signals) >= 1, (
            "Disabled gate (param absent) must not block high-gap near-term trade; "
            f"got {len(signals)} signals"
        )


# ===========================================================================
# Fix E — DEN_high and SEA_high exclusions
# ===========================================================================


class TestExcludedCityVarsDenHighSeaHigh:
    """DEN_high and SEA_high must be in excluded_city_vars in the default config."""

    def test_den_high_excluded_in_default_config(self):
        """Loading the production config must include DEN_high in excluded_city_vars."""
        from python.settings import Settings

        settings = Settings.load()
        excluded = settings.strategies.weather.excluded_city_vars
        assert "DEN_high" in excluded, (
            f"DEN_high should be in excluded_city_vars; got: {excluded}"
        )

    def test_sea_high_excluded_in_default_config(self):
        """Loading the production config must include SEA_high in excluded_city_vars."""
        from python.settings import Settings

        settings = Settings.load()
        excluded = settings.strategies.weather.excluded_city_vars
        assert "SEA_high" in excluded, (
            f"SEA_high should be in excluded_city_vars; got: {excluded}"
        )

    @pytest.mark.asyncio
    async def test_den_high_market_rejected_with_excluded_reason(self):
        """DEN high market is rejected with EXCLUDED_CITY_VAR when DEN_high is in config."""
        tomorrow = et_today() + timedelta(days=1)
        _, kalshi, weather = _make_strategy()  # don't use this strategy

        # Build a strategy with DEN_high in excluded list
        kalshi2 = AsyncMock()
        kalshi2.iter_all_markets.return_value = ([], False)
        weather2 = AsyncMock()
        weather2.compute_model_probability = MagicMock()

        async def _multi(city, forecast_date, daily_variable, models=None):
            try:
                fc = await weather2.get_ensemble_forecast(city, forecast_date, daily_variable)
                return {"gfs_seamless": fc}
            except Exception:
                return {}

        weather2.get_multi_model_forecast = AsyncMock(side_effect=_multi)

        config = {
            "enabled": True,
            "edge_threshold": 0.07,
            "max_edge": 0.90,
            "max_spread": 0.15,
            "spread_skill_sensitivity": 0.0,
            "cities": ["DEN"],
            "ensemble_models": ["gfs_seamless"],
            "ensemble_weights": {"gfs_seamless": 1.0},
            "orderbook_depth_check": False,
            "calibration_type": "none",
            "candidate_audit_log": False,
            "eval_audit_log": False,
            "scan_funnel_log": False,
            "excluded_city_vars": ["DEN_high"],
        }

        strategy = WeatherEdgeStrategy(
            config=config,
            kalshi_client=kalshi2,
            weather_client=weather2,
            bankroll=10_000.0,
            market_filter={
                "min_volume": 50,
                "min_open_interest": 5,
                "min_price": 0.03,
                "max_price": 0.97,
                "max_orderbook_take_pct": 0.50,
            },
        )

        # DEN KXHIGHDEN series ticker maps to DEN / TEMP_MAX (high)
        den_event = f"KXHIGHDEN-{tomorrow.strftime('%y')}{_MONTH_MAP[tomorrow.month]}{tomorrow.strftime('%d')}"
        den_market: dict[str, Any] = {
            "id": "",
            "ticker": f"{den_event}-T65",
            "event_ticker": den_event,
            "strike_type": "greater",
            "floor_strike": 65.0,
            "volume_fp": "100.0",
            "open_interest_fp": "20.0",
            "yes_bid_dollars": "0.45",
            "yes_ask_dollars": "0.55",
            "yes_bid_size_fp": "50.0",
            "yes_ask_size_fp": "50.0",
            "close_time": "2099-12-31T23:59:59Z",
        }
        kalshi2.iter_all_markets.side_effect = [([den_market], False), ([], False)]

        temps = [82.0] * 25 + [50.0] * 6
        forecast = _make_forecast("DEN", tomorrow, temps)
        weather2.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0, (
            f"DEN_high should be excluded; got {len(signals)} signals"
        )

    @pytest.mark.asyncio
    async def test_sea_high_market_rejected_with_excluded_reason(self):
        """SEA high market is rejected with EXCLUDED_CITY_VAR when SEA_high is in config."""
        tomorrow = et_today() + timedelta(days=1)

        kalshi2 = AsyncMock()
        kalshi2.iter_all_markets.return_value = ([], False)
        weather2 = AsyncMock()
        weather2.compute_model_probability = MagicMock()

        async def _multi(city, forecast_date, daily_variable, models=None):
            try:
                fc = await weather2.get_ensemble_forecast(city, forecast_date, daily_variable)
                return {"gfs_seamless": fc}
            except Exception:
                return {}

        weather2.get_multi_model_forecast = AsyncMock(side_effect=_multi)

        config = {
            "enabled": True,
            "edge_threshold": 0.07,
            "max_edge": 0.90,
            "max_spread": 0.15,
            "spread_skill_sensitivity": 0.0,
            "cities": ["SEA"],
            "ensemble_models": ["gfs_seamless"],
            "ensemble_weights": {"gfs_seamless": 1.0},
            "orderbook_depth_check": False,
            "calibration_type": "none",
            "candidate_audit_log": False,
            "eval_audit_log": False,
            "scan_funnel_log": False,
            "excluded_city_vars": ["SEA_high"],
        }

        strategy = WeatherEdgeStrategy(
            config=config,
            kalshi_client=kalshi2,
            weather_client=weather2,
            bankroll=10_000.0,
            market_filter={
                "min_volume": 50,
                "min_open_interest": 5,
                "min_price": 0.03,
                "max_price": 0.97,
                "max_orderbook_take_pct": 0.50,
            },
        )

        # SEA KXHIGHTSEA series ticker maps to SEA / TEMP_MAX (high)
        sea_event = f"KXHIGHTSEA-{tomorrow.strftime('%y')}{_MONTH_MAP[tomorrow.month]}{tomorrow.strftime('%d')}"
        sea_market: dict[str, Any] = {
            "id": "",
            "ticker": f"{sea_event}-T65",
            "event_ticker": sea_event,
            "strike_type": "greater",
            "floor_strike": 65.0,
            "volume_fp": "100.0",
            "open_interest_fp": "20.0",
            "yes_bid_dollars": "0.45",
            "yes_ask_dollars": "0.55",
            "yes_bid_size_fp": "50.0",
            "yes_ask_size_fp": "50.0",
            "close_time": "2099-12-31T23:59:59Z",
        }
        kalshi2.iter_all_markets.side_effect = [([sea_market], False), ([], False)]

        temps = [82.0] * 25 + [50.0] * 6
        forecast = _make_forecast("SEA", tomorrow, temps)
        weather2.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0, (
            f"SEA_high should be excluded; got {len(signals)} signals"
        )
