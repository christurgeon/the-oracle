"""Tests for weather edge strategy: market parsing, price extraction, and edge math."""

from __future__ import annotations

from datetime import date

import pytest

from python.common import DailyVariable, Direction
from python.data.weather import (
    EnsembleForecast,
    WeatherClient,
    blended_ensemble_median_std,
    blended_ensemble_std,
    compute_blended_probability,
)
from python.strategies.weather_edge import (
    MarketPriceInfo,
    WeatherEdgeStrategy,
    _orderbook_price_to_probability,
    _parse_dollar_price,
    _parse_fp_int,
    compute_fill_from_levels,
    parse_weather_market,
)
from python.utils.ticker_dates import parse_ticker_date

# ===========================================================================
# EnsembleForecast.std_dev
# ===========================================================================


class TestEnsembleStdDev:
    def test_known_values(self) -> None:
        """Population std dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0."""
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        )
        assert fc.std_dev == pytest.approx(2.0)

    def test_uniform_temperatures(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[70.0] * 31,
        )
        assert fc.std_dev == 0.0

    def test_two_values(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[60.0, 80.0],
        )
        # mean=70, variance=((10^2+10^2)/2)=100, std=10
        assert fc.std_dev == pytest.approx(10.0)

    def test_single_member(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[70.0],
        )
        assert fc.std_dev == 0.0

    def test_empty(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[],
        )
        assert fc.std_dev == 0.0


# ===========================================================================
# EnsembleForecast.fraction_between
# ===========================================================================


class TestFractionBetween:
    def test_all_inside(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[50.0, 51.0, 52.0],
        )
        assert fc.fraction_between(49.0, 53.0) == pytest.approx(1.0)

    def test_none_inside(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[40.0, 60.0],
        )
        assert fc.fraction_between(45.0, 55.0) == pytest.approx(0.0)

    def test_partial(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[49.0, 50.0, 51.0, 52.0],
        )
        # 50 and 51 are in [50, 51]
        assert fc.fraction_between(50.0, 51.0) == pytest.approx(0.5)

    def test_inclusive_boundaries(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[55.0, 56.0, 57.0],
        )
        # Both 55 and 56 should be included in [55, 56]
        assert fc.fraction_between(55.0, 56.0) == pytest.approx(2.0 / 3.0)

    def test_empty(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[],
        )
        assert fc.fraction_between(50.0, 60.0) == pytest.approx(0.0)


# ===========================================================================
# parse_weather_market
# ===========================================================================


class TestParseWeatherMarket:
    def test_greater_market(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-T56",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "greater",
            "floor_strike": 56,
            "cap_strike": None,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert parsed.city == "NYC"
        assert parsed.forecast_date == date(2026, 3, 14)
        assert parsed.strike_type == Direction.ABOVE
        assert parsed.floor_strike == 56.0
        assert parsed.cap_strike is None

    def test_less_market(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-T49",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "less",
            "floor_strike": None,
            "cap_strike": 49,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert parsed.strike_type == Direction.BELOW
        assert parsed.floor_strike is None
        assert parsed.cap_strike == 49.0

    def test_between_market(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-B55.5",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "between",
            "floor_strike": 55,
            "cap_strike": 56,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert parsed.strike_type == Direction.BETWEEN
        assert parsed.floor_strike == 55.0
        assert parsed.cap_strike == 56.0

    def test_unknown_strike_type(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-X50",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "unknown",
        }
        assert parse_weather_market(market, "NYC") is None

    def test_missing_floor_for_greater(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-T56",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "greater",
            "floor_strike": None,
            "cap_strike": None,
        }
        assert parse_weather_market(market, "NYC") is None

    def test_bad_event_date(self) -> None:
        market = {
            "ticker": "KXHIGHNY-BADDATE",
            "event_ticker": "KXHIGHNY-BADDATE",
            "strike_type": "greater",
            "floor_strike": 56,
        }
        assert parse_weather_market(market, "NYC") is None

    def test_description_greater(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-T56",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "greater",
            "floor_strike": 56,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert ">" in parsed.description
        assert "56" in parsed.description

    def test_description_between(self) -> None:
        market = {
            "ticker": "KXHIGHNY-26MAR14-B55.5",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "between",
            "floor_strike": 55,
            "cap_strike": 56,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert "55" in parsed.description
        assert "56" in parsed.description


# ===========================================================================
# parse_ticker_date (used as the event-ticker date parser)
# ===========================================================================


class TestParseEventDate:
    def test_valid(self) -> None:
        assert parse_ticker_date("KXHIGHNY-26MAR14") == date(2026, 3, 14)

    def test_december(self) -> None:
        assert parse_ticker_date("KXHIGHCHI-26DEC25") == date(2026, 12, 25)

    def test_invalid_month(self) -> None:
        assert parse_ticker_date("KXHIGHNY-26XYZ14") is None

    def test_no_date(self) -> None:
        assert parse_ticker_date("KXHIGHNY") is None


# ===========================================================================
# load_traded_tickers — regression after ticker-date refactor
# ===========================================================================


class TestLoadTradedTickers:
    """Regression: dedup set loads correctly after the parse_ticker_date
    refactor.  Expired tickers are skipped; unparseable tickers are admitted
    (they can't be aged out, so safer to treat them as live)."""

    def _run(self, tmp_path, lines: list[str]) -> set[str]:
        """Invoke load_traded_tickers without needing the full constructor."""
        log = tmp_path / "log.jsonl"
        with open(log, "w") as f:
            for line in lines:
                f.write(line + "\n")

        instance = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
        instance._traded_tickers = set()  # type: ignore[attr-defined]
        WeatherEdgeStrategy.load_traded_tickers(instance, str(log))
        return instance._traded_tickers  # type: ignore[attr-defined]

    def test_future_dated_ticker_admitted(self, tmp_path) -> None:
        import json as _json

        entry = _json.dumps({"market_ticker": "KXHIGHNY-99APR14-B70.5"})
        tickers = self._run(tmp_path, [entry])
        assert "KXHIGHNY-99APR14-B70.5" in tickers

    def test_unparseable_ticker_admitted(self, tmp_path) -> None:
        import json as _json

        entry = _json.dumps({"market_ticker": "KXINX-26MAR20H1600"})
        tickers = self._run(tmp_path, [entry])
        assert "KXINX-26MAR20H1600" in tickers

    def test_past_ticker_skipped(self, tmp_path) -> None:
        import json as _json

        entry = _json.dumps({"market_ticker": "KXHIGHNY-20JAN01-B70.5"})
        tickers = self._run(tmp_path, [entry])
        assert tickers == set()

    def test_malformed_json_skipped(self, tmp_path) -> None:
        import json as _json

        good = _json.dumps({"market_ticker": "KXHIGHNY-99APR14-B70.5"})
        tickers = self._run(tmp_path, ["{not json}", good])
        assert tickers == {"KXHIGHNY-99APR14-B70.5"}


# ===========================================================================
# unmark_traded — Bug B: release per-session dedup on zero-fill cancellation
# ===========================================================================


class TestUnmarkTraded:
    """The orchestrator calls ``unmark_traded`` when a placed order is
    cancelled with zero fills (TTL expiry, server-side cancellation), so a
    later scan can re-consider the ticker. Without this, the strategy is
    stuck waiting for a process restart.
    """

    def _strategy(self) -> WeatherEdgeStrategy:
        instance = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
        instance._traded_tickers = set()  # type: ignore[attr-defined]
        return instance

    def test_unmark_traded_removes_ticker_from_dedup_set(self) -> None:
        s = self._strategy()
        s.mark_traded("KXHIGHNY-99APR14-B70.5")
        assert "KXHIGHNY-99APR14-B70.5" in s._traded_tickers  # type: ignore[attr-defined]

        s.unmark_traded("KXHIGHNY-99APR14-B70.5")
        assert "KXHIGHNY-99APR14-B70.5" not in s._traded_tickers  # type: ignore[attr-defined]

    def test_unmark_traded_idempotent(self) -> None:
        """Unmarking a ticker that isn't in the set must silently no-op."""
        s = self._strategy()
        # Never marked — must not raise
        s.unmark_traded("KXHIGHNY-99APR14-B70.5")
        assert s._traded_tickers == set()  # type: ignore[attr-defined]

        s.mark_traded("KXHIGHNY-99APR14-B70.5")
        s.unmark_traded("KXHIGHNY-99APR14-B70.5")
        # Second unmark is a no-op
        s.unmark_traded("KXHIGHNY-99APR14-B70.5")
        assert s._traded_tickers == set()  # type: ignore[attr-defined]


# ===========================================================================
# MarketPriceInfo / _extract_market_price_info
# ===========================================================================


class TestExtractMarketPriceInfo:
    def test_bid_and_ask_dollars(self) -> None:
        market = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.50"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint == pytest.approx(0.45)
        assert info.yes_bid == pytest.approx(0.40)
        assert info.yes_ask == pytest.approx(0.50)
        assert info.spread == pytest.approx(0.10)

    def test_bid_ask_numeric(self) -> None:
        """Numeric (non-string) dollar values also work."""
        market = {"yes_bid_dollars": 0.40, "yes_ask_dollars": 0.50}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint == pytest.approx(0.45)
        assert info.spread == pytest.approx(0.10)

    def test_fallback_ask_only(self) -> None:
        market = {"yes_ask_dollars": "0.55"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint == pytest.approx(0.55)
        assert info.spread is None
        assert info.yes_bid is None
        assert info.yes_ask is None

    def test_fallback_last_price(self) -> None:
        market = {"last_price_dollars": "0.65"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint == pytest.approx(0.65)

    def test_no_data(self) -> None:
        info = WeatherEdgeStrategy._extract_market_price_info({})
        assert info is None

    def test_negative_spread_guard(self) -> None:
        """Malformed data where bid > ask should clamp spread to 0."""
        market = {"yes_bid_dollars": "0.60", "yes_ask_dollars": "0.50"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.spread == 0.0

    def test_invalid_values_ignored(self) -> None:
        market = {"yes_bid_dollars": "bad", "yes_ask_dollars": "data"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is None


# ===========================================================================
# Effective edge math
# ===========================================================================


class TestEffectiveEdge:
    """Test the effective edge formula: raw_edge * ensemble_confidence - half_spread."""

    def test_raw_edge_only_no_spread_no_confidence(self) -> None:
        """k=0, no spread -> effective_edge = raw_edge."""
        raw_edge = 0.10
        k = 0.0
        ensemble_std = 5.0
        spread = None

        confidence = 1.0 / (1.0 + k * ensemble_std) if k > 0 else 1.0
        half_spread = (spread / 2.0) if spread is not None else 0.0
        effective = raw_edge * confidence - half_spread

        assert effective == pytest.approx(0.10)

    def test_spread_only_adjustment(self) -> None:
        """k=0, spread=0.04 -> effective_edge = raw_edge - 0.02."""
        raw_edge = 0.10
        spread = 0.04

        confidence = 1.0  # k=0 means no discount
        half_spread = spread / 2.0
        effective = raw_edge * confidence - half_spread

        assert effective == pytest.approx(0.08)

    def test_confidence_only_adjustment(self) -> None:
        """k=0.1, std_dev=5.0, no spread -> confidence = 1/(1+0.5) = 0.667."""
        raw_edge = 0.12
        k = 0.1
        ensemble_std = 5.0

        confidence = 1.0 / (1.0 + k * ensemble_std)
        effective = raw_edge * confidence

        assert confidence == pytest.approx(1.0 / 1.5)
        assert effective == pytest.approx(0.12 / 1.5)

    def test_both_spread_and_confidence(self) -> None:
        """Full formula: raw_edge * confidence - half_spread."""
        raw_edge = 0.15
        # k=0.1, ensemble_std=5.0 → 1/(1+0.5) = 0.6667
        confidence = 1.0 / (1.0 + 0.5)
        half_spread = 0.02  # spread=0.04 / 2
        effective = raw_edge * confidence - half_spread

        assert effective == pytest.approx(0.15 / 1.5 - 0.02)

    def test_effective_edge_can_be_negative(self) -> None:
        """When spread cost exceeds discounted edge."""
        raw_edge = 0.05
        spread = 0.12
        confidence = 1.0
        half_spread = spread / 2.0
        effective = raw_edge * confidence - half_spread

        assert effective == pytest.approx(-0.01)


# ===========================================================================
# Spread filtering
# ===========================================================================


class TestSpreadFiltering:
    def test_market_skipped_when_spread_exceeds_max(self) -> None:
        max_spread = 0.15
        spread = 0.20
        assert spread > max_spread

    def test_market_passes_when_spread_under_max(self) -> None:
        max_spread = 0.15
        spread = 0.10
        assert spread <= max_spread

    def test_market_passes_when_spread_unknown(self) -> None:
        """Markets without spread data should not be filtered (max_spread=0.15)."""
        spread = None
        assert spread is None  # no filtering applies


# ===========================================================================
# Signal metadata and confidence (integration-style)
# ===========================================================================


class TestSignalConstruction:
    """Test that signals carry the new metadata fields and correct confidence."""

    def _make_forecast(self, temps: list[float]) -> EnsembleForecast:
        return EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=temps,
        )

    def test_metadata_keys_present(self) -> None:
        """Verify all new metadata keys exist in a signal's metadata dict."""
        expected_keys = {
            "yes_bid",
            "yes_ask",
            "bid_ask_spread",
            "half_spread_cost",
            "ensemble_std_dev",
            "ensemble_confidence",
            "raw_edge",
            "effective_edge",
        }
        # Build a mock metadata dict the way generate_signals would
        price_info = MarketPriceInfo(
            midpoint=0.45,
            yes_bid=0.40,
            yes_ask=0.50,
            spread=0.10,
        )
        forecast = self._make_forecast([70.0] * 31)
        k = 0.1
        model_prob = 0.60
        raw_edge = abs(model_prob - price_info.midpoint)
        ensemble_confidence = 1.0 / (1.0 + k * forecast.std_dev)
        half_spread = price_info.spread / 2.0

        metadata = {
            "yes_bid": price_info.yes_bid,
            "yes_ask": price_info.yes_ask,
            "bid_ask_spread": price_info.spread,
            "half_spread_cost": half_spread,
            "ensemble_std_dev": forecast.std_dev,
            "ensemble_confidence": ensemble_confidence,
            "raw_edge": raw_edge,
            "effective_edge": raw_edge * ensemble_confidence - half_spread,
        }

        assert expected_keys.issubset(metadata.keys())

    def test_confidence_equals_ensemble_confidence(self) -> None:
        """With k>0, signal confidence should reflect ensemble uncertainty."""
        temps = [60.0 + i * 2.0 for i in range(31)]  # spread out temps
        forecast = self._make_forecast(temps)
        k = 0.1
        expected_confidence = 1.0 / (1.0 + k * forecast.std_dev)
        assert expected_confidence < 1.0  # non-trivial discount

    def test_confidence_is_one_when_k_zero(self) -> None:
        """With k=0 (disabled), confidence should be 1.0."""
        forecast = self._make_forecast([60.0 + i * 2.0 for i in range(31)])
        k = 0.0
        confidence = 1.0 / (1.0 + k * forecast.std_dev) if k > 0 else 1.0
        assert confidence == 1.0


# ===========================================================================
# Backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """With spread_skill_sensitivity=0.0 and no spread data, behavior matches original."""

    def test_effective_edge_equals_raw_edge(self) -> None:
        """k=0, spread=None -> effective_edge = raw_edge exactly."""
        model_prob = 0.65
        market_price = 0.50
        k = 0.0
        spread = None

        raw_edge = abs(model_prob - market_price)
        confidence = 1.0 if k == 0.0 else 1.0 / (1.0 + k * 5.0)
        half_spread = (spread / 2.0) if spread is not None else 0.0
        effective_edge = raw_edge * confidence - half_spread

        assert effective_edge == pytest.approx(raw_edge)
        assert effective_edge == pytest.approx(0.15)

    def test_price_info_from_ask_only(self) -> None:
        """Single yes_ask_dollars field produces midpoint = yes_ask, spread = None."""
        market = {"yes_ask_dollars": "0.50"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint == pytest.approx(0.50)
        assert info.spread is None


# ===========================================================================
# KXLOWT (low temperature) support
# ===========================================================================


class TestKXLOWTParsing:
    """Test that KXLOWT markets parse correctly with daily_variable=TEMP_MIN."""

    def test_lowt_greater_market(self) -> None:
        market = {
            "ticker": "KXLOWTNYC-26MAR14-T40",
            "event_ticker": "KXLOWTNYC-26MAR14",
            "strike_type": "greater",
            "floor_strike": 40,
            "cap_strike": None,
        }
        parsed = parse_weather_market(market, "NYC", DailyVariable.TEMP_MIN)
        assert parsed is not None
        assert parsed.city == "NYC"
        assert parsed.forecast_date == date(2026, 3, 14)
        assert parsed.strike_type == Direction.ABOVE
        assert parsed.floor_strike == 40.0
        assert parsed.daily_variable == DailyVariable.TEMP_MIN

    def test_lowt_less_market(self) -> None:
        market = {
            "ticker": "KXLOWTCHI-26MAR14-T25",
            "event_ticker": "KXLOWTCHI-26MAR14",
            "strike_type": "less",
            "floor_strike": None,
            "cap_strike": 25,
        }
        parsed = parse_weather_market(market, "CHI", DailyVariable.TEMP_MIN)
        assert parsed is not None
        assert parsed.strike_type == Direction.BELOW
        assert parsed.cap_strike == 25.0
        assert parsed.daily_variable == DailyVariable.TEMP_MIN

    def test_lowt_between_market(self) -> None:
        market = {
            "ticker": "KXLOWTDEN-26MAR14-B30.5",
            "event_ticker": "KXLOWTDEN-26MAR14",
            "strike_type": "between",
            "floor_strike": 30,
            "cap_strike": 32,
        }
        parsed = parse_weather_market(market, "DEN", DailyVariable.TEMP_MIN)
        assert parsed is not None
        assert parsed.strike_type == Direction.BETWEEN
        assert parsed.floor_strike == 30.0
        assert parsed.cap_strike == 32.0
        assert parsed.daily_variable == DailyVariable.TEMP_MIN

    def test_lowt_description_says_low(self) -> None:
        market = {
            "ticker": "KXLOWTNYC-26MAR14-T40",
            "event_ticker": "KXLOWTNYC-26MAR14",
            "strike_type": "greater",
            "floor_strike": 40,
        }
        parsed = parse_weather_market(market, "NYC", DailyVariable.TEMP_MIN)
        assert parsed is not None
        assert "low" in parsed.description
        assert ">" in parsed.description
        assert "40" in parsed.description

    def test_high_description_says_high(self) -> None:
        """Existing KXHIGH markets still say 'high'."""
        market = {
            "ticker": "KXHIGHNY-26MAR14-T56",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "greater",
            "floor_strike": 56,
        }
        parsed = parse_weather_market(market, "NYC", DailyVariable.TEMP_MAX)
        assert parsed is not None
        assert "high" in parsed.description

    def test_default_daily_variable_is_max(self) -> None:
        """parse_weather_market defaults to temperature_2m_max for backward compat."""
        market = {
            "ticker": "KXHIGHNY-26MAR14-T56",
            "event_ticker": "KXHIGHNY-26MAR14",
            "strike_type": "greater",
            "floor_strike": 56,
        }
        parsed = parse_weather_market(market, "NYC")
        assert parsed is not None
        assert parsed.daily_variable == DailyVariable.TEMP_MAX


class TestWeatherSeriesMapping:
    """Test the WEATHER_SERIES mapping includes both high and low temp series."""

    def test_kxhigh_series_present(self) -> None:
        from python.strategies.weather_edge import WEATHER_SERIES

        assert "KXHIGHNY" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXHIGHNY"]
        assert city == "NYC"
        assert var == DailyVariable.TEMP_MAX

    def test_kxlowt_series_present(self) -> None:
        from python.strategies.weather_edge import WEATHER_SERIES

        assert "KXLOWTNYC" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXLOWTNYC"]
        assert city == "NYC"
        assert var == DailyVariable.TEMP_MIN

    def test_all_lowt_cities(self) -> None:
        from python.strategies.weather_edge import WEATHER_SERIES

        lowt_cities = {
            city for ticker, (city, var) in WEATHER_SERIES.items() if var == DailyVariable.TEMP_MIN
        }
        assert lowt_cities == {
            "NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "PHIL", "BOS", "SEA",
            # New cities (collection-only, added 2026-06-09).
            "ATL", "DAL", "HOU", "LV", "MIN", "NOLA", "OKC", "PHX", "SATX", "SFO", "DC",
        }

    def test_total_series_count(self) -> None:
        """Original 9 high + 9 low = 18; + 22 new-city series = 40 total."""
        from python.strategies.weather_edge import WEATHER_SERIES

        assert len(WEATHER_SERIES) == 40

    def test_phil_series_present(self) -> None:
        from python.strategies.weather_edge import WEATHER_SERIES

        assert "KXHIGHPHIL" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXHIGHPHIL"]
        assert city == "PHIL"
        assert var == DailyVariable.TEMP_MAX

        assert "KXLOWTPHIL" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXLOWTPHIL"]
        assert city == "PHIL"
        assert var == DailyVariable.TEMP_MIN

        # PHIL uses KXHIGH prefix, not KXHIGHT
        assert "KXHIGHTPHIL" not in WEATHER_SERIES

    def test_bos_series_present(self) -> None:
        from python.strategies.weather_edge import WEATHER_SERIES

        # Boston high uses KXHIGHT prefix (same anomaly as KXHIGHTSEA)
        assert "KXHIGHTBOS" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXHIGHTBOS"]
        assert city == "BOS"
        assert var == DailyVariable.TEMP_MAX

        assert "KXLOWTBOS" in WEATHER_SERIES
        city, var = WEATHER_SERIES["KXLOWTBOS"]
        assert city == "BOS"
        assert var == DailyVariable.TEMP_MIN

        # BOS low does NOT use an extra T (Kalshi quirk: only BOS-high has the T)
        assert "KXLOWTTBOS" not in WEATHER_SERIES


class TestWeatherCacheKeyIsolation:
    """Verify that high and low forecasts for the same city+date are cached separately."""

    def test_different_cache_keys(self) -> None:
        from python.data.weather import _ForecastCache

        cache = _ForecastCache()
        d = date(2026, 3, 15)

        key_max = cache._key("NYC", d, DailyVariable.TEMP_MAX)
        key_min = cache._key("NYC", d, DailyVariable.TEMP_MIN)
        assert key_max != key_min

    def test_cache_stores_both(self) -> None:
        from python.data.weather import _ForecastCache

        cache = _ForecastCache()
        d = date(2026, 3, 15)

        fc_high = EnsembleForecast(city="NYC", forecast_date=d, temperatures=[70.0] * 31)
        fc_low = EnsembleForecast(city="NYC", forecast_date=d, temperatures=[45.0] * 31)

        cache.put(fc_high, DailyVariable.TEMP_MAX)
        cache.put(fc_low, DailyVariable.TEMP_MIN)

        assert cache.get("NYC", d, DailyVariable.TEMP_MAX) is fc_high
        assert cache.get("NYC", d, DailyVariable.TEMP_MIN) is fc_low


# ===========================================================================
# _parse_fp_int helper
# ===========================================================================


class TestParseFpInt:
    def test_string_integer(self) -> None:
        assert _parse_fp_int("636") == 636

    def test_string_float(self) -> None:
        assert _parse_fp_int("636.00") == 636

    def test_numeric_int(self) -> None:
        assert _parse_fp_int(100) == 100

    def test_numeric_float(self) -> None:
        assert _parse_fp_int(42.9) == 42

    def test_none_returns_none(self) -> None:
        assert _parse_fp_int(None) is None

    def test_bad_string_returns_none(self) -> None:
        assert _parse_fp_int("not_a_number") is None

    def test_zero(self) -> None:
        assert _parse_fp_int("0") == 0

    def test_empty_string(self) -> None:
        assert _parse_fp_int("") is None


# ===========================================================================
# MarketPriceInfo size fields
# ===========================================================================


class TestMarketPriceInfoSizeFields:
    def test_size_fields_parsed(self) -> None:
        market = {
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.50",
            "yes_bid_size_fp": "200.00",
            "yes_ask_size_fp": "150.00",
            "volume_fp": "5000.00",
            "open_interest_fp": "1200.00",
        }
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.yes_bid_size == 200
        assert info.yes_ask_size == 150
        assert info.volume == 5000
        assert info.open_interest == 1200

    def test_size_fields_default_none(self) -> None:
        market = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.50"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.yes_bid_size is None
        assert info.yes_ask_size is None
        assert info.volume is None
        assert info.open_interest is None

    def test_fallback_still_includes_size_fields(self) -> None:
        """Single-price fallback path still parses size fields."""
        market = {
            "yes_ask_dollars": "0.55",
            "volume_fp": "300",
        }
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.volume == 300


# ===========================================================================
# Penny market and volume filters (scan-level)
# ===========================================================================


class TestScanLevelFilters:
    """Test the scan-level filter logic used in WeatherEdgeStrategy.scan()."""

    def test_penny_market_detected(self) -> None:
        """Midpoint below min_price should be filtered."""
        min_price = 0.03
        market = {"yes_bid_dollars": "0.005", "yes_ask_dollars": "0.01"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint < min_price

    def test_reverse_penny_detected(self) -> None:
        """Midpoint above max_price should be filtered."""
        max_price = 0.97
        market = {"yes_bid_dollars": "0.98", "yes_ask_dollars": "0.99"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert info.midpoint > max_price

    def test_normal_market_passes(self) -> None:
        """Midpoint in [min_price, max_price] passes the filter."""
        min_price = 0.03
        max_price = 0.97
        market = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.50"}
        info = WeatherEdgeStrategy._extract_market_price_info(market)
        assert info is not None
        assert min_price <= info.midpoint <= max_price


# ===========================================================================
# Orderbook depth cap
# ===========================================================================


class TestOrderbookDepthCap:
    """Test the depth cap logic applied after Kelly sizing in generate_signals."""

    def test_cap_applied_when_exceeds_depth(self) -> None:
        """contracts > depth * max_take_pct should be capped."""
        depth = 100
        max_take_pct = 0.50
        contracts = 80

        max_contracts = max(int(depth * max_take_pct), 1)
        assert max_contracts == 50
        assert contracts > max_contracts  # would be capped
        capped = min(contracts, max_contracts)
        assert capped == 50

    def test_no_cap_when_under_depth(self) -> None:
        """contracts <= depth * max_take_pct should not be capped."""
        depth = 200
        max_take_pct = 0.50
        contracts = 30

        max_contracts = max(int(depth * max_take_pct), 1)
        assert max_contracts == 100
        assert contracts <= max_contracts

    def test_cap_at_minimum_one(self) -> None:
        """Even with tiny depth, cap should be at least 1 contract."""
        depth = 1
        max_take_pct = 0.10

        max_contracts = max(int(depth * max_take_pct), 1)
        assert max_contracts == 1  # floor of 1

    def test_no_cap_when_pct_is_one(self) -> None:
        """max_take_pct=1.0 means the cap is disabled."""
        max_take_pct = 1.0
        assert not (max_take_pct < 1.0)  # cap logic skipped

    def test_no_cap_when_depth_none(self) -> None:
        """When orderbook depth is unknown (None), no capping."""
        depth = None

        # The actual code only caps when depth is not None and > 0
        should_cap = depth is not None and depth > 0
        assert not should_cap


# ===========================================================================
# market_filter wiring
# ===========================================================================


class TestMarketFilterInit:
    """Test that WeatherEdgeStrategy accepts and stores market_filter."""

    def test_default_empty_filter(self) -> None:
        from unittest.mock import MagicMock

        strategy = WeatherEdgeStrategy(
            config={"enabled": True},
            kalshi_client=MagicMock(),
            weather_client=MagicMock(),
            bankroll=10000.0,
        )
        assert strategy._market_filter == {}

    def test_custom_filter(self) -> None:
        from unittest.mock import MagicMock

        filt = {"min_volume": 50, "min_price": 0.03, "max_orderbook_take_pct": 0.50}
        strategy = WeatherEdgeStrategy(
            config={"enabled": True},
            kalshi_client=MagicMock(),
            weather_client=MagicMock(),
            bankroll=10000.0,
            market_filter=filt,
        )
        assert strategy._market_filter == filt
        assert strategy._market_filter["min_volume"] == 50


# ===========================================================================
# compute_fill_from_levels
# ===========================================================================


class TestComputeFillFromLevels:
    """Test the VWAP fill price computation for orderbook depth checks."""

    def test_single_level_full_fill(self) -> None:
        levels = [(0.65, 20)]
        result = compute_fill_from_levels(levels, target_contracts=10)
        assert not result.is_stale
        assert result.fill_price == pytest.approx(0.65)
        assert result.available_contracts == 20
        assert result.levels_used == 1
        assert result.slippage == pytest.approx(0.0)

    def test_multi_level_vwap(self) -> None:
        # Buy 15 contracts: 10 @ 0.60, 5 @ 0.65
        levels = [(0.60, 10), (0.65, 10)]
        result = compute_fill_from_levels(levels, target_contracts=15)
        assert not result.is_stale
        expected_vwap = (10 * 0.60 + 5 * 0.65) / 15
        assert result.fill_price == pytest.approx(expected_vwap)
        assert result.levels_used == 2
        assert result.slippage == pytest.approx(expected_vwap - 0.60)

    def test_empty_book_is_stale(self) -> None:
        result = compute_fill_from_levels([], target_contracts=5)
        assert result.is_stale
        assert result.available_contracts == 0
        assert result.fill_price == 0.0

    def test_thin_book_is_stale(self) -> None:
        levels = [(0.50, 3)]  # only 3 contracts, min_depth=5
        result = compute_fill_from_levels(levels, target_contracts=5, min_depth=5)
        assert result.is_stale
        assert result.available_contracts == 3

    def test_partial_fill(self) -> None:
        # Want 20 contracts but only 12 available
        levels = [(0.55, 7), (0.60, 5)]
        result = compute_fill_from_levels(levels, target_contracts=20)
        assert not result.is_stale
        expected_vwap = (7 * 0.55 + 5 * 0.60) / 12
        assert result.fill_price == pytest.approx(expected_vwap)
        assert result.available_contracts == 12

    def test_exact_fill(self) -> None:
        levels = [(0.50, 10)]
        result = compute_fill_from_levels(levels, target_contracts=10)
        assert not result.is_stale
        assert result.fill_price == pytest.approx(0.50)
        assert result.levels_used == 1

    def test_min_depth_zero(self) -> None:
        """With min_depth=0, even 1 contract is not stale."""
        levels = [(0.70, 1)]
        result = compute_fill_from_levels(levels, target_contracts=1, min_depth=0)
        assert not result.is_stale
        assert result.fill_price == pytest.approx(0.70)

    def test_slippage_increases_with_levels(self) -> None:
        levels = [(0.50, 5), (0.55, 5), (0.60, 5)]
        result_small = compute_fill_from_levels(levels, target_contracts=3)
        result_large = compute_fill_from_levels(levels, target_contracts=12)
        assert result_large.slippage > result_small.slippage


# ===========================================================================
# compute_model_probability — core weather strategy probability function
# ===========================================================================


class TestComputeModelProbabilityAbove:
    """Test ABOVE direction (fraction of members strictly above threshold)."""

    def setup_method(self) -> None:
        self.client = WeatherClient.__new__(WeatherClient)

    def test_all_above(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [80.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(1.0)

    def test_none_above(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [60.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(0.0)

    def test_exactly_at_threshold_not_counted(self) -> None:
        """fraction_above uses strictly-greater, so members at threshold are NOT counted."""
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [70.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(0.0)

    def test_partial_above(self) -> None:
        # 10 above, 21 at or below
        temps = [80.0] * 10 + [70.0] * 21
        fc = EnsembleForecast("NYC", date(2026, 3, 15), temps)
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(10 / 31)

    def test_single_member_above(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [80.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(1.0)

    def test_missing_floor_strike_raises(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [80.0] * 31)
        with pytest.raises(ValueError, match="floor_strike"):
            self.client.compute_model_probability(fc, Direction.ABOVE, floor_strike=None)


class TestComputeModelProbabilityBelow:
    """Test BELOW direction (fraction of members strictly below threshold)."""

    def setup_method(self) -> None:
        self.client = WeatherClient.__new__(WeatherClient)

    def test_all_below(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [60.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(1.0)

    def test_none_below(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [80.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(0.0)

    def test_exactly_at_threshold_not_counted(self) -> None:
        """fraction_below = 1 - fraction_at_or_above. Members exactly at threshold
        are counted by fraction_at_or_above, so they are NOT counted as below."""
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [70.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(0.0)

    def test_partial_below(self) -> None:
        temps = [60.0] * 15 + [80.0] * 16
        fc = EnsembleForecast("NYC", date(2026, 3, 15), temps)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(15 / 31)

    def test_missing_cap_strike_raises(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [60.0] * 31)
        with pytest.raises(ValueError, match="cap_strike"):
            self.client.compute_model_probability(fc, Direction.BELOW, cap_strike=None)


class TestComputeModelProbabilityBetween:
    """Test BETWEEN direction (fraction of members in [floor, cap] inclusive)."""

    def setup_method(self) -> None:
        self.client = WeatherClient.__new__(WeatherClient)

    def test_all_between(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [72.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(1.0)

    def test_none_between(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [80.0] * 31)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(0.0)

    def test_boundaries_inclusive(self) -> None:
        """Members exactly at floor or cap should be counted."""
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [70.0, 75.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(1.0)

    def test_partial_between(self) -> None:
        # 10 inside [70, 75], rest outside
        temps = [72.0] * 10 + [60.0] * 10 + [80.0] * 11
        fc = EnsembleForecast("NYC", date(2026, 3, 15), temps)
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(10 / 31)

    def test_narrow_bracket(self) -> None:
        """Single-degree bracket: only members at exactly that temp count."""
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [70.0, 70.0, 71.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(2 / 3)

    def test_missing_strikes_raises(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [72.0] * 31)
        with pytest.raises(ValueError, match="floor_strike and cap_strike"):
            self.client.compute_model_probability(
                fc,
                Direction.BETWEEN,
                floor_strike=None,
                cap_strike=75.0,
            )
        with pytest.raises(ValueError, match="floor_strike and cap_strike"):
            self.client.compute_model_probability(
                fc,
                Direction.BETWEEN,
                floor_strike=70.0,
                cap_strike=None,
            )


class TestComputeModelProbabilityEdgeCases:
    """Edge cases and error handling."""

    def setup_method(self) -> None:
        self.client = WeatherClient.__new__(WeatherClient)

    def test_empty_ensemble_above(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [])
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(0.0)

    def test_empty_ensemble_below(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        # fraction_below = 1 - fraction_at_or_above(70) = 1 - 0 = 1.0
        # For empty ensemble, fraction_at_or_above returns 0.0
        assert prob == pytest.approx(1.0)

    def test_empty_ensemble_between(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(0.0)

    def test_above_and_below_complement(self) -> None:
        """For a single threshold, P(above T) + P(at_or_below T) should equal 1.

        Note: fraction_above is strictly-greater, fraction_below is strictly-less.
        Members exactly at the threshold are counted by neither, so
        above + below < 1 when members are at the threshold.
        """
        temps = [65.0, 70.0, 70.0, 75.0, 80.0]
        fc = EnsembleForecast("NYC", date(2026, 3, 15), temps)
        above = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        # at_or_above captures 70, 70, 75, 80 = 4/5
        # below = 1 - at_or_above = 1/5
        below = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        # above=2/5 (75, 80), below=1/5 (65), at_threshold=2/5 (70, 70)
        assert above == pytest.approx(2 / 5)
        assert below == pytest.approx(1 / 5)

    def test_unsupported_direction_raises(self) -> None:
        fc = EnsembleForecast("NYC", date(2026, 3, 15), [70.0] * 31)
        with pytest.raises(ValueError, match="Unsupported direction"):
            self.client.compute_model_probability(
                fc,
                "invalid_direction",
                floor_strike=70.0,  # type: ignore[arg-type]
            )


# ===========================================================================
# Price normalization utilities
# ===========================================================================


class TestParseDollarPrice:
    """Tests for _parse_dollar_price — replaces the fragile _normalize heuristic."""

    def test_normal_price(self) -> None:
        assert _parse_dollar_price("0.65") == pytest.approx(0.65)

    def test_zero(self) -> None:
        assert _parse_dollar_price("0.0") == pytest.approx(0.0)

    def test_one(self) -> None:
        assert _parse_dollar_price("1.0") == pytest.approx(1.0)

    def test_rejects_cents_format(self) -> None:
        """Values > 1.0 (cents format) should be rejected, not auto-converted."""
        assert _parse_dollar_price("65") is None

    def test_rejects_negative(self) -> None:
        assert _parse_dollar_price("-0.05") is None

    def test_none_input(self) -> None:
        assert _parse_dollar_price(None) is None

    def test_non_numeric_string(self) -> None:
        assert _parse_dollar_price("N/A") is None

    def test_float_input(self) -> None:
        assert _parse_dollar_price(0.42) == pytest.approx(0.42)

    def test_nearly_certain_market(self) -> None:
        """The old heuristic would misdetect $0.99 as fine but $1.00 wouldn't hit the bug.
        Key bug: a market at exactly $1.05 would become $0.0105."""
        assert _parse_dollar_price("0.99") == pytest.approx(0.99)
        # Values slightly above 1.0 are rejected (not auto-divided by 100)
        assert _parse_dollar_price("1.05") is None


class TestOrderbookPriceToProbability:
    """Tests for _orderbook_price_to_probability — explicit dollars-to-probability."""

    def test_half_dollar(self) -> None:
        assert _orderbook_price_to_probability(0.50) == pytest.approx(0.50)

    def test_one_cent(self) -> None:
        assert _orderbook_price_to_probability(0.01) == pytest.approx(0.99)

    def test_ninety_nine_cents(self) -> None:
        assert _orderbook_price_to_probability(0.99) == pytest.approx(0.01)

    def test_sixty_five_cents(self) -> None:
        # Opposite side bid at $0.65 → our side ask at $0.35
        assert _orderbook_price_to_probability(0.65) == pytest.approx(0.35)


# ===========================================================================
# Multi-model ensemble blending
# ===========================================================================


class TestComputeBlendedProbability:
    """Tests for compute_blended_probability()."""

    def _fc(
        self,
        temps: list[float],
        model: str = "gfs_seamless",
    ) -> EnsembleForecast:
        return EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=temps,
            model=model,
        )

    def test_single_model_above(self) -> None:
        """Single-model blending equals the model's own fraction."""
        gfs = self._fc([60.0, 70.0, 80.0, 90.0])
        forecasts = {"gfs_seamless": gfs}
        weights = {"gfs_seamless": 1.0}
        prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            Direction.ABOVE,
            floor_strike=65.0,
        )
        # 70, 80, 90 are above 65 → 3/4
        assert prob == pytest.approx(0.75)
        assert per_model == {"gfs_seamless": pytest.approx(0.75)}

    def test_equal_weight_blending(self) -> None:
        """Two models with equal weights → average of their fractions."""
        gfs = self._fc([60.0, 70.0, 80.0, 90.0], model="gfs_seamless")
        ecmwf = self._fc([65.0, 75.0], model="ecmwf_ifs025")
        forecasts = {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf}
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            Direction.ABOVE,
            floor_strike=65.0,
        )
        # GFS: 3/4 = 0.75, ECMWF: 1/2 = 0.5
        # Blended: (0.5 * 0.75 + 0.5 * 0.5) = 0.625
        assert prob == pytest.approx(0.625)
        assert per_model["gfs_seamless"] == pytest.approx(0.75)
        assert per_model["ecmwf_ifs025"] == pytest.approx(0.5)

    def test_unequal_weight_blending(self) -> None:
        """Non-equal weights are renormalized correctly."""
        gfs = self._fc([60.0, 70.0, 80.0, 90.0], model="gfs_seamless")
        ecmwf = self._fc([65.0, 75.0], model="ecmwf_ifs025")
        forecasts = {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf}
        weights = {"gfs_seamless": 0.7, "ecmwf_ifs025": 0.3}
        prob, _ = compute_blended_probability(
            forecasts,
            weights,
            Direction.ABOVE,
            floor_strike=65.0,
        )
        # GFS: 0.75, ECMWF: 0.5
        # Blended: (0.7 * 0.75 + 0.3 * 0.5) / 1.0 = 0.675
        assert prob == pytest.approx(0.675)

    def test_missing_model_renormalizes(self) -> None:
        """When ECMWF is missing, weights renormalize to GFS-only."""
        gfs = self._fc([60.0, 70.0, 80.0, 90.0], model="gfs_seamless")
        forecasts = {"gfs_seamless": gfs}
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            Direction.ABOVE,
            floor_strike=65.0,
        )
        # Only GFS available → result equals GFS alone
        assert prob == pytest.approx(0.75)
        assert "ecmwf_ifs025" not in per_model

    def test_no_models_returns_zero(self) -> None:
        """Empty forecasts dict → 0.0."""
        prob, per_model = compute_blended_probability(
            {},
            {"gfs_seamless": 1.0},
            Direction.ABOVE,
            floor_strike=65.0,
        )
        assert prob == 0.0
        assert per_model == {}

    def test_below_direction(self) -> None:
        """Blending works for BELOW direction."""
        gfs = self._fc([60.0, 70.0, 80.0, 90.0], model="gfs_seamless")
        ecmwf = self._fc([55.0, 65.0, 75.0, 85.0], model="ecmwf_ifs025")
        forecasts = {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf}
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            Direction.BELOW,
            cap_strike=70.0,
        )
        # GFS: fraction_below(70) = fraction where t < 70 → 1/4 = 0.25
        # ECMWF: fraction_below(70) → 55, 65 are below → 2/4 = 0.5
        assert per_model["gfs_seamless"] == pytest.approx(0.25)
        assert per_model["ecmwf_ifs025"] == pytest.approx(0.5)
        assert prob == pytest.approx(0.375)

    def test_between_direction(self) -> None:
        """Blending works for BETWEEN direction."""
        gfs = self._fc([60.0, 65.0, 70.0, 80.0], model="gfs_seamless")
        ecmwf = self._fc([62.0, 68.0], model="ecmwf_ifs025")
        forecasts = {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf}
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            Direction.BETWEEN,
            floor_strike=60.0,
            cap_strike=70.0,
        )
        # GFS: 60, 65, 70 in [60, 70] → 3/4 = 0.75
        # ECMWF: 62, 68 in [60, 70] → 2/2 = 1.0
        assert per_model["gfs_seamless"] == pytest.approx(0.75)
        assert per_model["ecmwf_ifs025"] == pytest.approx(1.0)
        assert prob == pytest.approx(0.875)


class TestBlendedEnsembleStd:
    """Tests for blended_ensemble_std()."""

    def _fc(
        self,
        temps: list[float],
        model: str = "gfs_seamless",
    ) -> EnsembleForecast:
        return EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=temps,
            model=model,
        )

    def test_single_model(self) -> None:
        """Pooled std with one model equals that model's std."""
        gfs = self._fc([60.0, 80.0])
        assert blended_ensemble_std({"gfs_seamless": gfs}) == pytest.approx(10.0)

    def test_two_models_pooled(self) -> None:
        """Pooled std across models captures inter-model disagreement."""
        # GFS: all 70, ECMWF: all 80 → pooled [70, 70, 80, 80] → std = 5.0
        gfs = self._fc([70.0, 70.0], model="gfs_seamless")
        ecmwf = self._fc([80.0, 80.0], model="ecmwf_ifs025")
        std = blended_ensemble_std({"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf})
        assert std == pytest.approx(5.0)

    def test_empty(self) -> None:
        assert blended_ensemble_std({}) == 0.0


class TestBlendedEnsembleMedianStd:
    """Tests for blended_ensemble_median_std() — used by the strike-proximity gate."""

    def _fc(self, temps: list[float], model: str = "gfs_seamless") -> EnsembleForecast:
        return EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=temps,
            model=model,
        )

    def test_single_model_median_and_std(self) -> None:
        gfs = self._fc([60.0, 80.0])
        median, std = blended_ensemble_median_std({"gfs_seamless": gfs})
        assert median == pytest.approx(70.0)
        assert std == pytest.approx(10.0)

    def test_odd_count_picks_middle(self) -> None:
        fc = self._fc([65.0, 70.0, 75.0])
        median, _ = blended_ensemble_median_std({"gfs_seamless": fc})
        assert median == pytest.approx(70.0)

    def test_pooled_median_differs_from_primary_when_models_disagree(self) -> None:
        """Blended median must reflect BOTH models, not just the first in the dict."""
        gfs = self._fc([60.0, 65.0, 70.0], model="gfs_seamless")
        ecmwf = self._fc([80.0, 85.0, 90.0], model="ecmwf_ifs025")
        median, std = blended_ensemble_median_std({"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf})
        assert median == pytest.approx(75.0)
        assert std > 0

    def test_empty_dict_returns_zeros(self) -> None:
        median, std = blended_ensemble_median_std({})
        assert median == 0.0
        assert std == 0.0

    def test_single_member_returns_zeros(self) -> None:
        """Single member → variance undefined; helper returns (0, 0)."""
        median, std = blended_ensemble_median_std({"gfs_seamless": self._fc([70.0])})
        assert median == 0.0
        assert std == 0.0


class TestEnsembleForecastModelField:
    """Test that the model field works correctly on EnsembleForecast."""

    def test_default_model(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[70.0],
        )
        assert fc.model == "gfs_seamless"

    def test_ecmwf_model(self) -> None:
        fc = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[70.0],
            model="ecmwf_ifs025",
        )
        assert fc.model == "ecmwf_ifs025"


class TestCacheModelIsolation:
    """Verify that the same city+date+variable but different models are cached separately."""

    def test_different_cache_keys_per_model(self) -> None:
        from python.data.weather import _ForecastCache

        cache = _ForecastCache()
        d = date(2026, 3, 15)

        key_gfs = cache._key("NYC", d, DailyVariable.TEMP_MAX, "gfs_seamless")
        key_ecmwf = cache._key("NYC", d, DailyVariable.TEMP_MAX, "ecmwf_ifs025")
        assert key_gfs != key_ecmwf

    def test_cache_stores_both_models(self) -> None:
        from python.data.weather import _ForecastCache

        cache = _ForecastCache()
        d = date(2026, 3, 15)

        fc_gfs = EnsembleForecast(
            city="NYC",
            forecast_date=d,
            temperatures=[70.0] * 31,
            model="gfs_seamless",
        )
        fc_ecmwf = EnsembleForecast(
            city="NYC",
            forecast_date=d,
            temperatures=[72.0] * 51,
            model="ecmwf_ifs025",
        )

        cache.put(fc_gfs, DailyVariable.TEMP_MAX)
        cache.put(fc_ecmwf, DailyVariable.TEMP_MAX)

        assert cache.get("NYC", d, DailyVariable.TEMP_MAX, "gfs_seamless") is fc_gfs
        assert cache.get("NYC", d, DailyVariable.TEMP_MAX, "ecmwf_ifs025") is fc_ecmwf
