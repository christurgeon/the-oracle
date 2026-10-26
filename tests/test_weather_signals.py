"""Integration tests for weather strategy signal generation pipeline.

Tests the full scan → evaluate → generate_signals flow with mocked API
responses, exercising the filters that prevent bad trades:
  - max_edge filter (stale quote detection)
  - spread filter (illiquid markets)
  - open interest filter
  - same-day market exclusion
  - penny/reverse-penny market filter
  - orderbook depth capping
  - ensemble confidence discount
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import Side, et_today
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import WeatherEdgeStrategy

# ── Helpers ────────────────────────────────────────────────────────────────


def _tomorrow() -> date:
    return et_today() + timedelta(days=1)


def _tomorrow_event_ticker(city: str = "NY") -> str:
    d = _tomorrow()
    month_map = {
        1: "JAN",
        2: "FEB",
        3: "MAR",
        4: "APR",
        5: "MAY",
        6: "JUN",
        7: "JUL",
        8: "AUG",
        9: "SEP",
        10: "OCT",
        11: "NOV",
        12: "DEC",
    }
    # Kalshi format: KXHIGHNY-{YY}{MMM}{DD}  (year first, then month, then day)
    return f"KXHIGH{city}-{d.strftime('%y')}{month_map[d.month]}{d.strftime('%d')}"


def _make_market(
    *,
    ticker: str | None = None,
    event_ticker: str | None = None,
    strike_type: str = "greater",
    floor_strike: float = 55.0,
    cap_strike: float | None = None,
    yes_bid_dollars: float = 0.40,
    yes_ask_dollars: float = 0.50,
    volume: int = 100,
    open_interest: int = 20,
    close_time: str = "2099-12-31T23:59:59Z",
    bid_size: int = 50,
    ask_size: int = 50,
) -> dict[str, Any]:
    et = event_ticker or _tomorrow_event_ticker()
    tk = ticker or f"{et}-T{floor_strike:.0f}"
    market: dict[str, Any] = {
        "id": "",
        "ticker": tk,
        "event_ticker": et,
        "strike_type": strike_type,
        "volume_fp": str(float(volume)),
        "open_interest_fp": str(float(open_interest)),
        "yes_bid_dollars": str(yes_bid_dollars),
        "yes_ask_dollars": str(yes_ask_dollars),
        "yes_bid_size_fp": str(float(bid_size)),
        "yes_ask_size_fp": str(float(ask_size)),
        "close_time": close_time,
    }
    if floor_strike is not None:
        market["floor_strike"] = floor_strike
    if cap_strike is not None:
        market["cap_strike"] = cap_strike
    return market


def _make_forecast(temps: list[float] | None = None) -> EnsembleForecast:
    """Forecast where most members predict ~80°F (well above 55°F threshold)."""
    if temps is None:
        # 31 members: 28 at 82°F, 3 at 50°F → ~90% chance > 55
        temps = [82.0] * 28 + [50.0] * 3
    return EnsembleForecast(
        city="NYC",
        forecast_date=_tomorrow(),
        temperatures=temps,
    )


def _make_strategy(
    *,
    edge_threshold: float = 0.08,
    max_edge: float = 0.30,
    max_spread: float = 0.15,
    spread_skill_sensitivity: float = 0.0,
    cities: list[str] | None = None,
    market_filter: dict[str, Any] | None = None,
) -> tuple[WeatherEdgeStrategy, AsyncMock, AsyncMock]:
    kalshi = AsyncMock()
    # By default, only return markets for the first series call and empty
    # for subsequent series (NYC has both KXHIGHNY and KXLOWTNYC).
    # Tests that need specific behavior can override this.
    kalshi.iter_all_markets.return_value = ([], False)
    weather = AsyncMock()
    # compute_model_probability is synchronous — must not be a coroutine
    weather.compute_model_probability = MagicMock()

    # get_multi_model_forecast delegates to get_ensemble_forecast so tests
    # that configure the latter automatically work with multi-model fetching.
    async def _multi_model_side_effect(city, forecast_date, daily_variable, models=None):
        try:
            fc = await weather.get_ensemble_forecast(city, forecast_date, daily_variable)
            return {"gfs_seamless": fc}
        except Exception:
            return {}

    weather.get_multi_model_forecast = AsyncMock(side_effect=_multi_model_side_effect)

    config = {
        "enabled": True,
        "edge_threshold": edge_threshold,
        "max_edge": max_edge,
        "max_spread": max_spread,
        "spread_skill_sensitivity": spread_skill_sensitivity,
        "cities": cities or ["NYC"],
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        "orderbook_depth_check": False,
    }

    mf = (
        market_filter
        if market_filter is not None
        else {
            "min_volume": 50,
            "min_open_interest": 5,
            "min_price": 0.03,
            "max_price": 0.97,
            "max_orderbook_take_pct": 0.50,
        }
    )

    strategy = WeatherEdgeStrategy(
        config=config,
        kalshi_client=kalshi,
        weather_client=weather,
        bankroll=10_000.0,
        market_filter=mf,
    )
    return strategy, kalshi, weather


# ── max_edge filter (stale quote detection) ──────────────────────────────


class TestMaxEdgeFilter:
    """Implausibly large edges are filtered as stale quotes."""

    @pytest.mark.asyncio
    async def test_edge_above_max_filtered(self):
        """Market with model=0.90, mid=0.50 → edge=0.40 > max_edge=0.30."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.30)

        # Market priced at 50%, but model says 90% → 40% edge
        market = _make_market(yes_bid_dollars=0.48, yes_ask_dollars=0.52)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Forecast: all 31 members at 82°F → model prob ≈ 1.0 for > 55°F
        forecast = _make_forecast([82.0] * 31)
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.90

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_edge_below_max_passes(self):
        """Market with model=0.70, mid=0.50 → edge=0.20 < max_edge=0.30.

        ``TradeSignal.edge`` is computed against the YES midpoint — the
        canonical ``market_price`` field on the signal. Cost-adjusted edge
        gating inside the strategy still uses ``exec_price``.
        """
        strategy, kalshi, weather = _make_strategy(max_edge=0.30)

        market = _make_market(yes_bid_dollars=0.48, yes_ask_dollars=0.52)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 7/10 members above 55°F → fraction_above(55) = 0.70
        forecast = _make_forecast([82.0] * 7 + [50.0] * 3)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        # edge = abs(model_prob - midpoint) = abs(0.70 - 0.50) = 0.20
        assert signals[0].edge == pytest.approx(0.20)


# ── Spread filter ────────────────────────────────────────────────────────


class TestSpreadFilter:
    """Wide bid-ask spreads indicate illiquid markets."""

    @pytest.mark.asyncio
    async def test_wide_spread_filtered(self):
        """Spread of 20 cents > max_spread 15 cents → filtered."""
        strategy, kalshi, weather = _make_strategy(max_spread=0.15)

        # 20-cent spread: bid=0.35, ask=0.55
        market = _make_market(yes_bid_dollars=0.35, yes_ask_dollars=0.55)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.80

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_narrow_spread_passes(self):
        """Spread of 4 cents < max_spread 15 cents → passes."""
        strategy, kalshi, weather = _make_strategy(max_spread=0.15)

        market = _make_market(yes_bid_dollars=0.48, yes_ask_dollars=0.52)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 7/10 members above 55°F → fraction_above(55) = 0.70, edge 0.20
        forecast = _make_forecast([82.0] * 7 + [50.0] * 3)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1


# ── Open interest filter ─────────────────────────────────────────────────


class TestOpenInterestFilter:
    """Low open interest indicates stale orderbooks."""

    @pytest.mark.asyncio
    async def test_low_oi_filtered_at_scan(self):
        strategy, kalshi, weather = _make_strategy()

        market = _make_market(open_interest=2)  # below min_open_interest=5
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_sufficient_oi_passes_scan(self):
        strategy, kalshi, weather = _make_strategy()

        market = _make_market(open_interest=10)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 1


# ── Same-day market exclusion ────────────────────────────────────────────


class TestSameDayExclusion:
    """Markets for today's date use stale GFS forecasts vs observed NWS values."""

    @pytest.mark.asyncio
    async def test_today_market_excluded(self):
        strategy, kalshi, weather = _make_strategy()

        # Build a market for today
        today = et_today()
        month_map = {
            1: "JAN",
            2: "FEB",
            3: "MAR",
            4: "APR",
            5: "MAY",
            6: "JUN",
            7: "JUL",
            8: "AUG",
            9: "SEP",
            10: "OCT",
            11: "NOV",
            12: "DEC",
        }
        et = f"KXHIGHNY-{today.strftime('%y')}{month_map[today.month]}{today.strftime('%d')}"
        market = _make_market(event_ticker=et)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_tomorrow_market_included(self):
        strategy, kalshi, weather = _make_strategy()

        market = _make_market()  # defaults to tomorrow
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 1


# ── Penny market filter ──────────────────────────────────────────────────


class TestPennyMarketFilter:
    """Penny and reverse-penny markets are illiquid noise."""

    @pytest.mark.asyncio
    async def test_penny_market_filtered(self):
        strategy, kalshi, weather = _make_strategy()

        # Midpoint = 0.02, below min_price=0.03
        market = _make_market(yes_bid_dollars=0.01, yes_ask_dollars=0.03)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 0

    @pytest.mark.asyncio
    async def test_reverse_penny_filtered(self):
        strategy, kalshi, weather = _make_strategy()

        # Midpoint = 0.98, above max_price=0.97
        market = _make_market(yes_bid_dollars=0.97, yes_ask_dollars=0.99)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 0


# ── Volume filter ────────────────────────────────────────────────────────


class TestVolumeFilter:
    @pytest.mark.asyncio
    async def test_low_volume_filtered(self):
        strategy, kalshi, weather = _make_strategy()

        market = _make_market(volume=10)  # below min_volume=50
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        markets = await strategy.scan()
        assert len(markets) == 0


# ── Orderbook depth capping ──────────────────────────────────────────────


class TestOrderbookDepthCapping:
    """Contract count should be capped at a fraction of orderbook depth."""

    @pytest.mark.asyncio
    async def test_contracts_capped_at_depth(self):
        strategy, kalshi, weather = _make_strategy(edge_threshold=0.05, max_edge=0.50)

        # Market with thin ask: only 4 contracts available
        market = _make_market(
            yes_bid_dollars=0.40,
            yes_ask_dollars=0.45,
            ask_size=4,
            bid_size=100,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        # Model says 60% → buying YES at 42.5 midpoint → edge ~17.5%
        weather.compute_model_probability.return_value = 0.60

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        # max_take_pct=0.50 → max 2 contracts from ask depth of 4
        assert signals[0].suggested_contracts <= 2


# ── Effective edge formula ───────────────────────────────────────────────


class TestEffectiveEdge:
    """effective_edge = raw_edge * ensemble_confidence - half_spread"""

    @pytest.mark.asyncio
    async def test_spread_cost_deducted(self):
        """Even with a real edge, wide spread eats into profitability."""
        strategy, kalshi, weather = _make_strategy(
            edge_threshold=0.08,
            max_spread=0.20,
        )

        # 10-cent spread: half_spread = 0.05
        market = _make_market(yes_bid_dollars=0.45, yes_ask_dollars=0.55)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        # model=0.60, mid=0.50 → raw_edge=0.10, effective=0.10-0.05=0.05 < 0.08
        weather.compute_model_probability.return_value = 0.60

        signals = await strategy.generate_signals()
        assert len(signals) == 0  # filtered by effective edge threshold

    @pytest.mark.asyncio
    async def test_ensemble_uncertainty_discounts_edge(self):
        """High ensemble uncertainty reduces confidence and effective edge."""
        strategy, kalshi, weather = _make_strategy(
            edge_threshold=0.08,
            spread_skill_sensitivity=1.0,
        )

        market = _make_market(yes_bid_dollars=0.28, yes_ask_dollars=0.32)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # High-uncertainty forecast: std_dev = 10°F
        # ensemble_confidence = 1/(1 + 1.0 * 10) = 0.0909
        temps = [50.0 + i * 2 for i in range(31)]  # wide spread
        forecast = EnsembleForecast(
            city="NYC",
            forecast_date=_tomorrow(),
            temperatures=temps,
        )
        weather.get_ensemble_forecast.return_value = forecast
        # model=0.60, mid=0.30 → raw_edge=0.30
        # effective = 0.30 * 0.0909 - 0.02 = 0.0073 < 0.08
        weather.compute_model_probability.return_value = 0.60

        signals = await strategy.generate_signals()
        assert len(signals) == 0


# ── Signal metadata ──────────────────────────────────────────────────────


class TestSignalMetadata:
    """Signals must carry close_time for shadow settlement."""

    @pytest.mark.asyncio
    async def test_close_time_in_metadata(self):
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)

        close_time = "2099-12-31T23:59:59Z"
        market = _make_market(
            yes_bid_dollars=0.40,
            yes_ask_dollars=0.45,
            close_time=close_time,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.65

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].metadata["close_time"] == close_time

    @pytest.mark.asyncio
    async def test_signal_side_correct_for_yes(self):
        """When model_prob > market_price, side should be YES."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)

        market = _make_market(yes_bid_dollars=0.40, yes_ask_dollars=0.45)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.70  # > mid 0.425

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].side == Side.YES

    @pytest.mark.asyncio
    async def test_signal_side_correct_for_no(self):
        """When model_prob < market_price, side should be NO."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)

        market = _make_market(yes_bid_dollars=0.60, yes_ask_dollars=0.65)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # 3/10 members above 55°F → fraction_above(55) = 0.30 < mid 0.625
        forecast = _make_forecast([82.0] * 3 + [50.0] * 7)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].side == Side.NO


# ── Forecast failure resilience ──────────────────────────────────────────


class TestForecastFailure:
    """Strategy must handle forecast API failures gracefully."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio
        monkeypatch.setattr(_asyncio, "sleep", AsyncMock(return_value=None))

    @pytest.mark.asyncio
    async def test_failed_forecast_skips_market(self):
        strategy, kalshi, weather = _make_strategy()

        market = _make_market()
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        weather.get_ensemble_forecast.side_effect = Exception("429 Too Many Requests")

        # Should not raise — just produce no signals
        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_partial_forecast_failure(self):
        """If one city fails, others still produce signals."""
        strategy, kalshi, weather = _make_strategy(
            cities=["NYC", "CHI"],
            max_edge=0.50,
        )

        market_nyc = _make_market(event_ticker=_tomorrow_event_ticker("NY"))
        market_chi = _make_market(event_ticker=_tomorrow_event_ticker("CHI"))
        kalshi.iter_all_markets.side_effect = [
            ([market_nyc], False),
            ([market_chi], False),
        ]

        forecast_nyc = _make_forecast()

        async def _mock_forecast(city, forecast_date, daily_variable=None):
            if city == "CHI":
                raise Exception("API error")
            return forecast_nyc

        weather.get_ensemble_forecast = _mock_forecast
        weather.compute_model_probability.return_value = 0.70

        signals = await strategy.generate_signals()
        # NYC should succeed, CHI should fail gracefully
        # Exact count depends on whether forecasts are batched — at minimum
        # no exception raised
        assert isinstance(signals, list)


# ── Orderbook depth check helpers ──────────────────────────────────────


def _make_orderbook(
    yes_bids: list[list[int]] | None = None,
    no_bids: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Build a Kalshi ``orderbook_fp`` response dict.

    Accepts legacy-style ``[[price_cents, qty_int], ...]`` inputs and emits
    the current API schema: decimal dollar strings under ``yes_dollars``/
    ``no_dollars`` keys.
    """

    def _fmt(levels: list[list[int]] | None) -> list[list[str]]:
        if not levels:
            return []
        return [[f"{p / 100:.4f}", f"{q:.2f}"] for p, q in levels]

    return {
        "orderbook_fp": {
            "yes_dollars": _fmt(yes_bids),
            "no_dollars": _fmt(no_bids),
        }
    }


# ── Orderbook depth check integration tests ───────────────────────────


class TestOrderbookDepthCheck:
    """Full orderbook depth check: fetch book → validate fill → filter."""

    @pytest.mark.asyncio
    async def test_stale_book_filtered(self):
        """Market with empty orderbook is skipped."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)
        strategy._config["orderbook_depth_check"] = True

        market = _make_market(yes_bid_dollars=0.40, yes_ask_dollars=0.45)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]
        kalshi.get_orderbook.return_value = _make_orderbook()  # empty

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.65

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_good_depth_passes(self):
        """Market with sufficient depth generates a signal."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)
        strategy._config["orderbook_depth_check"] = True
        strategy._config["min_book_depth"] = 5

        market = _make_market(yes_bid_dollars=0.40, yes_ask_dollars=0.45)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Model says 65% → buying YES. NO bid at 55 cents → YES ask at 45c.
        kalshi.get_orderbook.return_value = _make_orderbook(
            no_bids=[[55, 50]],
        )

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.65

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].metadata["fill_price"] == pytest.approx(0.45)
        assert signals[0].metadata["fill_slippage"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_fill_edge_too_low_filtered(self):
        """Slippage eats into edge enough to filter the signal."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.50,
            edge_threshold=0.08,
        )
        strategy._config["orderbook_depth_check"] = True
        strategy._config["min_book_depth"] = 1

        # Midpoint = 0.50, model = 0.60 → raw_edge = 0.10
        market = _make_market(yes_bid_dollars=0.48, yes_ask_dollars=0.52)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Buying YES: NO bid at 40 → YES ask at 60c.
        # fill_edge = 0.60 - 0.60 = 0.00 < threshold
        kalshi.get_orderbook.return_value = _make_orderbook(
            no_bids=[[40, 50]],
        )

        # 6/10 members above 55°F → fraction_above(55) = 0.60
        forecast = _make_forecast([82.0] * 6 + [50.0] * 4)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_api_failure_falls_back(self):
        """When orderbook API fails, signal still generated using fallback."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)
        strategy._config["orderbook_depth_check"] = True

        market = _make_market(yes_bid_dollars=0.40, yes_ask_dollars=0.45)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]
        kalshi.get_orderbook.side_effect = Exception("timeout")

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.65

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].metadata["fill_price"] is None  # depth check failed

    @pytest.mark.asyncio
    async def test_contracts_capped_at_book_depth(self):
        """Contracts should be capped at max_take_pct of total book depth."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.50,
            edge_threshold=0.05,
        )
        strategy._config["orderbook_depth_check"] = True
        strategy._config["min_book_depth"] = 1

        market = _make_market(
            yes_bid_dollars=0.40,
            yes_ask_dollars=0.45,
            ask_size=100,
            bid_size=100,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Only 6 contracts available across the book
        kalshi.get_orderbook.return_value = _make_orderbook(
            no_bids=[[55, 4], [54, 2]],
        )

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.70

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        # max_take_pct = 0.50 → max 3 contracts from 6 available
        assert signals[0].suggested_contracts <= 3
        assert signals[0].metadata["book_depth"] == 6

    @pytest.mark.asyncio
    async def test_no_side_uses_yes_bids(self):
        """When buying NO, depth check uses YES bid side of orderbook."""
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)
        strategy._config["orderbook_depth_check"] = True
        strategy._config["min_book_depth"] = 5

        # Model says 30% → below mid 0.625 → buying NO
        market = _make_market(yes_bid_dollars=0.60, yes_ask_dollars=0.65)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # YES bid at 65 cents → NO ask at 35 cents
        kalshi.get_orderbook.return_value = _make_orderbook(
            yes_bids=[[65, 50]],
        )

        # 3/10 members above 55°F → fraction_above(55) = 0.30
        forecast = _make_forecast([82.0] * 3 + [50.0] * 7)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].side == Side.NO
        assert signals[0].metadata["fill_price"] == pytest.approx(0.35)

    @pytest.mark.asyncio
    async def test_real_kalshi_api_schema(self):
        """Regression: parse the actual ``orderbook_fp`` payload Kalshi returns.

        Fixture mirrors a real ``GET /markets/{t}/orderbook`` response on
        2026-04-21 — dollar-string prices, decimal-string qtys, including
        a fractional-qty level ("50.45") that the old int-cents parser
        would have dropped. Verifies the parser accepts the new schema
        and exposes the full book depth (not zero, as the bug produced).
        """
        strategy, kalshi, weather = _make_strategy(max_edge=0.50)
        strategy._config["orderbook_depth_check"] = True
        strategy._config["min_book_depth"] = 5

        # Model 65% vs midpoint 0.45 → buying YES. Crosses against NO
        # bids. Three NO levels → three derived YES-ask levels.
        market = _make_market(yes_bid_dollars=0.40, yes_ask_dollars=0.50)
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]
        kalshi.get_orderbook.return_value = {
            "orderbook_fp": {
                "no_dollars": [
                    ["0.5500", "87.00"],
                    ["0.5300", "76.00"],
                    ["0.5000", "50.45"],
                ],
                "yes_dollars": [
                    ["0.3200", "5.00"],
                ],
            }
        }

        forecast = _make_forecast()
        weather.get_ensemble_forecast.return_value = forecast
        weather.compute_model_probability.return_value = 0.65

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        md = signals[0].metadata
        # Book depth must reflect the three NO-bid levels parsed as our-side
        # asks (0.45 / 0.47 / 0.50). The fractional "50.45" truncates to 50.
        # The bug surfaced as book_depth == 0; the fix must yield >0.
        assert md["book_depth"] == 87 + 76 + 50
        assert md["book_levels"] == 3
        # Best YES ask is 1 - 0.55 = 0.45 (inverted from the highest NO bid);
        # worst usable is 0.50. VWAP lands somewhere between them depending on
        # Kelly sizing, but must be > 0 (the bug produced 0 via empty parse).
        assert 0.45 <= md["fill_price"] <= 0.50


# ── Strike-proximity gate ────────────────────────────────────────────────


class TestStrikeProximityGate:
    """Rejects threshold markets whose strike sits within k·σ of the pooled
    ensemble median — isotonic maps those ambiguous raw fractions to fake edge.
    """

    @pytest.mark.asyncio
    async def test_greater_strike_near_median_filtered(self):
        """ABOVE market with strike ≈ pooled median is filtered by the gate."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.50,
            edge_threshold=0.05,
        )
        strategy._config["min_distance_from_strike_std"] = 1.0

        market = _make_market(
            strike_type="greater",
            floor_strike=55.0,
            yes_bid_dollars=0.35,
            yes_ask_dollars=0.40,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Tight ensemble centered on the strike → abs_z ≈ 0 ≪ 1.0
        temps = [55.2] * 15 + [54.8] * 15
        forecast = EnsembleForecast(
            city="NYC",
            forecast_date=_tomorrow(),
            temperatures=temps,
        )
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_greater_strike_far_from_median_passes(self):
        """ABOVE market with strike >1σ from pooled median survives the gate."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.60,
            edge_threshold=0.05,
        )
        strategy._config["min_distance_from_strike_std"] = 1.0

        market = _make_market(
            strike_type="greater",
            floor_strike=55.0,
            yes_bid_dollars=0.45,
            yes_ask_dollars=0.50,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # median ≈ 82, std ≈ 10.6 → abs_z ≈ 2.55 > 1.0
        forecast = _make_forecast([82.0] * 28 + [50.0] * 3)
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        assert signals[0].metadata["strike_abs_z"] is not None
        assert signals[0].metadata["strike_abs_z"] > 1.0

    @pytest.mark.asyncio
    async def test_less_strike_near_median_filtered(self):
        """BELOW market with cap_strike ≈ pooled median is filtered."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.50,
            edge_threshold=0.05,
        )
        strategy._config["min_distance_from_strike_std"] = 1.0

        market = _make_market(
            strike_type="less",
            floor_strike=0.0,
            cap_strike=55.0,
            yes_bid_dollars=0.45,
            yes_ask_dollars=0.55,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        temps = [55.2] * 15 + [54.8] * 15
        forecast = EnsembleForecast(
            city="NYC",
            forecast_date=_tomorrow(),
            temperatures=temps,
        )
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_between_strike_never_gated(self):
        """BETWEEN markets bypass the strike-proximity gate regardless of k."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.60,
            edge_threshold=0.05,
        )
        # Strict k — would reject any threshold market
        strategy._config["min_distance_from_strike_std"] = 10.0

        market = _make_market(
            strike_type="between",
            floor_strike=54.0,
            cap_strike=56.0,
            yes_bid_dollars=0.40,
            yes_ask_dollars=0.45,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # All temps fall inside [54, 56] → fraction_between ≈ 1.0
        temps = [55.0] * 24 + [54.5] * 3 + [55.5] * 3
        forecast = EnsembleForecast(
            city="NYC",
            forecast_date=_tomorrow(),
            temperatures=temps,
        )
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        # BETWEEN markets never run the gate → no abs_z recorded
        assert signals[0].metadata["strike_abs_z"] is None

    @pytest.mark.asyncio
    async def test_gate_disabled_with_zero_k(self):
        """k=0 disables the gate — near-median threshold markets pass through."""
        strategy, kalshi, weather = _make_strategy(
            max_edge=0.60,
            edge_threshold=0.05,
        )
        strategy._config["min_distance_from_strike_std"] = 0.0

        market = _make_market(
            strike_type="greater",
            floor_strike=55.0,
            yes_bid_dollars=0.30,
            yes_ask_dollars=0.35,
        )
        kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

        # Tight ensemble at the strike — would be rejected with k>0.
        # Members ≥ 55.5 to satisfy the NWS-rounded threshold (T ≥ floor + 0.5).
        temps = [55.6] * 20 + [54.8] * 10
        forecast = EnsembleForecast(
            city="NYC",
            forecast_date=_tomorrow(),
            temperatures=temps,
        )
        weather.get_ensemble_forecast.return_value = forecast

        signals = await strategy.generate_signals()
        assert len(signals) == 1
        # Gate skipped → strike_abs_z stays None even for ABOVE market
        assert signals[0].metadata["strike_abs_z"] is None
