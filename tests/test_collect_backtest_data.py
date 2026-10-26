"""
Tests for scripts/collect_backtest_data.py.

Covers:
  1. Market age filtering by expiration_ts.
  2. Output file structure (markets.json, weather_forecasts.json,
     collection_metadata.json).
  3. Dry-run mode skips all real API calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

# Make sure the project root is on sys.path so we can import the script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.common import DailyVariable  # noqa: E402, I001
from scripts.collect_backtest_data import (  # noqa: E402
    CollectionMetadata,
    MarketRecord,
    WeatherForecastRecord,
    fetch_candlesticks_for_market,
    fetch_markets_for_series,
    fetch_weather_forecasts,
    filter_markets_by_age,
    parse_candlestick,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW_TS = int(datetime.now(UTC).timestamp())
CUTOFF_TS = NOW_TS - 7 * 24 * 3600  # 7 days ago


def _make_raw_market(ticker: str, exp_ts: int, result: str = "yes") -> dict[str, Any]:
    """Build a minimal raw market dict as the Kalshi API would return."""
    return {
        "ticker": ticker,
        "event_ticker": f"KXHIGHNY-{ticker}",
        "series_ticker": "KXHIGHNY",
        "strike_type": "greater",
        "floor_strike": 56,
        "cap_strike": None,
        "result": result,
        "expiration_ts": exp_ts,
    }


# ---------------------------------------------------------------------------
# Test 1: Market age filtering
# ---------------------------------------------------------------------------


class TestFilterMarketsByAge:
    """filter_markets_by_age should keep only markets within the time window."""

    def test_keeps_recent_market(self) -> None:
        recent_ts = NOW_TS - 2 * 24 * 3600  # 2 days ago
        markets = [_make_raw_market("MKT-A", recent_ts)]
        result = filter_markets_by_age(markets, CUTOFF_TS)
        assert len(result) == 1
        assert result[0]["ticker"] == "MKT-A"

    def test_excludes_old_market(self) -> None:
        old_ts = NOW_TS - 30 * 24 * 3600  # 30 days ago (beyond cutoff)
        markets = [_make_raw_market("MKT-OLD", old_ts)]
        result = filter_markets_by_age(markets, CUTOFF_TS)
        assert result == []

    def test_keeps_market_at_exact_cutoff(self) -> None:
        """A market exactly at the cutoff boundary should be kept (inclusive)."""
        markets = [_make_raw_market("MKT-EDGE", CUTOFF_TS)]
        result = filter_markets_by_age(markets, CUTOFF_TS)
        assert len(result) == 1

    def test_skips_market_with_no_expiration(self) -> None:
        market: dict[str, Any] = {"ticker": "NO-EXP", "result": "yes"}
        result = filter_markets_by_age([market], CUTOFF_TS)
        assert result == []

    def test_mixed_markets(self) -> None:
        recent_ts = NOW_TS - 2 * 24 * 3600
        old_ts = NOW_TS - 30 * 24 * 3600
        markets = [
            _make_raw_market("MKT-NEW", recent_ts),
            _make_raw_market("MKT-OLD", old_ts),
        ]
        result = filter_markets_by_age(markets, CUTOFF_TS)
        assert len(result) == 1
        assert result[0]["ticker"] == "MKT-NEW"

    def test_empty_input(self) -> None:
        assert filter_markets_by_age([], CUTOFF_TS) == []


# ---------------------------------------------------------------------------
# Test 2: Output file structure
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    """save_outputs should write three JSON files with the correct structure."""

    def _make_records(self) -> tuple[list[MarketRecord], dict[str, WeatherForecastRecord]]:
        exp_ts = NOW_TS - 2 * 24 * 3600
        exp_date = datetime.fromtimestamp(exp_ts, tz=UTC).date().isoformat()

        market = MarketRecord(
            ticker="KXHIGHNY-26MAR14-T56",
            series_ticker="KXHIGHNY",
            city="NYC",
            daily_variable=str(DailyVariable.TEMP_MAX),
            strike_type="greater",
            floor_strike=56.0,
            cap_strike=None,
            result="yes",
            expiration_ts=exp_ts,
            expiration_date=exp_date,
        )

        forecast = WeatherForecastRecord(
            city="NYC",
            forecast_date=exp_date,
            daily_variable=str(DailyVariable.TEMP_MAX),
            temperatures=[70.0, 72.0, 68.0],
            mean_temperature=70.0,
            std_dev=1.63,
            member_count=3,
            fetched_at=float(NOW_TS),
        )

        forecasts: dict[str, WeatherForecastRecord] = {
            f"NYC:{exp_date}:{DailyVariable.TEMP_MAX}": forecast
        }
        return [market], forecasts

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "backtest" / "sub"
        records, forecasts = self._make_records()
        metadata = CollectionMetadata(
            run_timestamp="2026-03-15T12:00:00+00:00",
            days_covered=7,
            cutoff_ts=CUTOFF_TS,
            market_count=1,
            series_count=1,
            weather_forecast_count=1,
            interval_minutes=60,
            dry_run=False,
            series_tickers=["KXHIGHNY"],
        )
        save_outputs(output_dir, records, forecasts, metadata)
        assert output_dir.exists()

    def test_three_files_created(self, tmp_path: Path) -> None:
        records, forecasts = self._make_records()
        metadata = CollectionMetadata(
            run_timestamp="2026-03-15T12:00:00+00:00",
            days_covered=7,
            cutoff_ts=CUTOFF_TS,
            market_count=1,
            series_count=1,
            weather_forecast_count=1,
            interval_minutes=60,
            dry_run=False,
            series_tickers=["KXHIGHNY"],
        )
        save_outputs(tmp_path, records, forecasts, metadata)

        assert (tmp_path / "markets.json").exists()
        assert (tmp_path / "weather_forecasts.json").exists()
        assert (tmp_path / "collection_metadata.json").exists()

    def test_markets_json_structure(self, tmp_path: Path) -> None:
        records, forecasts = self._make_records()
        metadata = CollectionMetadata(
            run_timestamp="2026-03-15T12:00:00+00:00",
            days_covered=7,
            cutoff_ts=CUTOFF_TS,
            market_count=1,
            series_count=1,
            weather_forecast_count=1,
            interval_minutes=60,
            dry_run=False,
            series_tickers=["KXHIGHNY"],
        )
        save_outputs(tmp_path, records, forecasts, metadata)

        with open(tmp_path / "markets.json") as f:
            markets_data = json.load(f)

        assert isinstance(markets_data, list)
        assert len(markets_data) == 1
        mkt = markets_data[0]
        assert mkt["ticker"] == "KXHIGHNY-26MAR14-T56"
        assert mkt["city"] == "NYC"
        assert mkt["daily_variable"] == str(DailyVariable.TEMP_MAX)
        assert mkt["strike_type"] == "greater"
        assert mkt["floor_strike"] == 56.0
        assert mkt["cap_strike"] is None
        assert mkt["result"] == "yes"
        assert "candlesticks" in mkt
        assert isinstance(mkt["candlesticks"], list)

    def test_weather_forecasts_json_structure(self, tmp_path: Path) -> None:
        records, forecasts = self._make_records()
        metadata = CollectionMetadata(
            run_timestamp="2026-03-15T12:00:00+00:00",
            days_covered=7,
            cutoff_ts=CUTOFF_TS,
            market_count=1,
            series_count=1,
            weather_forecast_count=1,
            interval_minutes=60,
            dry_run=False,
            series_tickers=["KXHIGHNY"],
        )
        save_outputs(tmp_path, records, forecasts, metadata)

        with open(tmp_path / "weather_forecasts.json") as f:
            fc_data = json.load(f)

        assert isinstance(fc_data, dict)
        assert len(fc_data) == 1
        key = next(iter(fc_data))
        assert "NYC:" in key
        fc = fc_data[key]
        assert fc["city"] == "NYC"
        assert isinstance(fc["temperatures"], list)
        assert fc["member_count"] == 3

    def test_metadata_json_structure(self, tmp_path: Path) -> None:
        records, forecasts = self._make_records()
        metadata = CollectionMetadata(
            run_timestamp="2026-03-15T12:00:00+00:00",
            days_covered=7,
            cutoff_ts=CUTOFF_TS,
            market_count=1,
            series_count=1,
            weather_forecast_count=1,
            interval_minutes=60,
            dry_run=False,
            series_tickers=["KXHIGHNY"],
        )
        save_outputs(tmp_path, records, forecasts, metadata)

        with open(tmp_path / "collection_metadata.json") as f:
            meta_data = json.load(f)

        assert meta_data["days_covered"] == 7
        assert meta_data["market_count"] == 1
        assert meta_data["series_count"] == 1
        assert meta_data["weather_forecast_count"] == 1
        assert meta_data["interval_minutes"] == 60
        assert meta_data["dry_run"] is False
        assert "KXHIGHNY" in meta_data["series_tickers"]
        assert "run_timestamp" in meta_data
        assert "cutoff_ts" in meta_data


# ---------------------------------------------------------------------------
# Test 3: Dry-run mode skips real API calls
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run should print intent without making any real API calls."""

    def test_dry_run_does_not_call_kalshi(self, tmp_path: Path) -> None:
        """
        When dry_run=True, run_collection must not instantiate KalshiClient
        or make any network calls.
        """
        from scripts.collect_backtest_data import run_collection

        with patch("scripts.collect_backtest_data.KalshiClient") as mock_kalshi_cls, \
             patch("scripts.collect_backtest_data.WeatherClient") as mock_weather_cls:

            asyncio.run(
                run_collection(
                    days=7,
                    output_dir=tmp_path / "out",
                    interval_minutes=60,
                    dry_run=True,
                    kalshi_api_key="test-key",
                    kalshi_private_key_path="test-key.pem",
                    kalshi_base_url="https://api.example.com",
                )
            )

            mock_kalshi_cls.assert_not_called()
            mock_weather_cls.assert_not_called()

    def test_dry_run_does_not_write_files(self, tmp_path: Path) -> None:
        """Dry-run should not produce any output files."""
        from scripts.collect_backtest_data import run_collection

        output_dir = tmp_path / "dry_run_out"

        with patch("scripts.collect_backtest_data.KalshiClient"), \
             patch("scripts.collect_backtest_data.WeatherClient"):
            asyncio.run(
                run_collection(
                    days=3,
                    output_dir=output_dir,
                    interval_minutes=60,
                    dry_run=True,
                    kalshi_api_key="test-key",
                    kalshi_private_key_path="test.pem",
                    kalshi_base_url="https://api.example.com",
                )
            )

        # Directory should not exist (dry-run creates nothing on disk)
        assert not output_dir.exists()


# ---------------------------------------------------------------------------
# Test 4: fetch_markets_for_series filtering
# ---------------------------------------------------------------------------


class TestFetchMarketsForSeries:
    """fetch_markets_for_series should apply the cutoff filter to API results."""

    def test_filters_old_markets_from_api_response(self) -> None:
        """Only markets within the cutoff window should survive."""
        recent_ts = NOW_TS - 2 * 24 * 3600
        old_ts = NOW_TS - 30 * 24 * 3600
        raw = [
            _make_raw_market("MKT-A", recent_ts),
            _make_raw_market("MKT-B", old_ts),
        ]

        mock_client = AsyncMock()
        mock_client.iter_all_markets = AsyncMock(return_value=(raw, False))

        records = asyncio.run(
            fetch_markets_for_series(
                client=mock_client,
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable=DailyVariable.TEMP_MAX,
                cutoff_ts=CUTOFF_TS,
            )
        )

        assert len(records) == 1
        assert records[0].ticker == "MKT-A"

    def test_returns_market_records(self) -> None:
        """Returned items should be MarketRecord instances with correct fields."""
        recent_ts = NOW_TS - 1 * 24 * 3600
        raw = [_make_raw_market("MKT-NYC", recent_ts)]

        mock_client = AsyncMock()
        mock_client.iter_all_markets = AsyncMock(return_value=(raw, False))

        records = asyncio.run(
            fetch_markets_for_series(
                client=mock_client,
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable=DailyVariable.TEMP_MAX,
                cutoff_ts=CUTOFF_TS,
            )
        )

        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, MarketRecord)
        assert rec.series_ticker == "KXHIGHNY"
        assert rec.city == "NYC"
        assert rec.daily_variable == str(DailyVariable.TEMP_MAX)


# ---------------------------------------------------------------------------
# Test 5: Candlestick parsing
# ---------------------------------------------------------------------------


class TestParseCandlestick:
    """parse_candlestick should correctly extract OHLC fields."""

    def test_basic_candle(self) -> None:
        raw: dict[str, Any] = {
            "end_period_ts": 1741910400,
            "yes_price": {"open": 45, "high": 55, "low": 40, "close": 50},
            "no_price": {"open": 55, "high": 60, "low": 45, "close": 50},
            "volume": 200,
            "open_interest": 500,
        }
        candle = parse_candlestick(raw)
        assert candle.end_period_ts == 1741910400
        assert candle.yes_price_open == 45
        assert candle.yes_price_high == 55
        assert candle.yes_price_low == 40
        assert candle.yes_price_close == 50
        assert candle.no_price_open == 55
        assert candle.volume == 200
        assert candle.open_interest == 500

    def test_missing_price_fields(self) -> None:
        """Missing sub-fields should produce None values."""
        raw: dict[str, Any] = {
            "end_period_ts": 1741910400,
            "yes_price": None,
            "no_price": None,
        }
        candle = parse_candlestick(raw)
        assert candle.yes_price_open is None
        assert candle.no_price_close is None

    def test_non_integer_values(self) -> None:
        """String prices should be coerced to int."""
        raw: dict[str, Any] = {
            "end_period_ts": 1741910400,
            "yes_price": {"open": "42", "close": "51"},
        }
        candle = parse_candlestick(raw)
        assert candle.yes_price_open == 42
        assert candle.yes_price_close == 51


# ---------------------------------------------------------------------------
# Test 6: Candlestick fetch error handling
# ---------------------------------------------------------------------------


class TestFetchCandlesticksErrorHandling:
    """Errors fetching candlesticks should be caught and stored on the record."""

    def test_api_error_sets_error_field(self) -> None:
        from python.data.kalshi_client import KalshiAPIError

        record = MarketRecord(
            ticker="KXHIGHNY-26MAR01-T56",
            series_ticker="KXHIGHNY",
            city="NYC",
            daily_variable=str(DailyVariable.TEMP_MAX),
            strike_type="greater",
            floor_strike=56.0,
            cap_strike=None,
            result="yes",
            expiration_ts=NOW_TS - 2 * 24 * 3600,
            expiration_date="2026-03-01",
        )

        mock_client = AsyncMock()
        mock_client.get_candlesticks = AsyncMock(
            side_effect=KalshiAPIError(404, {"error": "Market too old"})
        )

        updated = asyncio.run(
            fetch_candlesticks_for_market(
                client=mock_client,
                record=record,
                interval_minutes=60,
            )
        )

        assert updated.candlestick_error is not None
        assert "404" in updated.candlestick_error or "Market too old" in updated.candlestick_error
        assert updated.candlesticks == []

    def test_success_populates_candlesticks(self) -> None:
        record = MarketRecord(
            ticker="KXHIGHNY-26MAR14-T56",
            series_ticker="KXHIGHNY",
            city="NYC",
            daily_variable=str(DailyVariable.TEMP_MAX),
            strike_type="greater",
            floor_strike=56.0,
            cap_strike=None,
            result="yes",
            expiration_ts=NOW_TS - 2 * 24 * 3600,
            expiration_date="2026-03-14",
        )

        raw_candles = [
            {
                "end_period_ts": NOW_TS - 24 * 3600,
                "yes_price": {"open": 45, "high": 50, "low": 40, "close": 48},
                "volume": 100,
            }
        ]
        mock_client = AsyncMock()
        mock_client.get_candlesticks = AsyncMock(
            return_value={"candlesticks": raw_candles}
        )

        updated = asyncio.run(
            fetch_candlesticks_for_market(
                client=mock_client,
                record=record,
                interval_minutes=60,
            )
        )

        assert len(updated.candlesticks) == 1
        assert updated.candlestick_error is None
        assert updated.candlesticks[0].yes_price_open == 45


# ---------------------------------------------------------------------------
# Test 7: fetch_weather_forecasts skips failed lookups
# ---------------------------------------------------------------------------


class TestFetchWeatherForecasts:
    """fetch_weather_forecasts should skip cities with API errors gracefully."""

    def test_skips_failed_forecasts(self) -> None:
        exp_ts = NOW_TS - 1 * 24 * 3600
        exp_date = datetime.fromtimestamp(exp_ts, tz=UTC).date().isoformat()

        records = [
            MarketRecord(
                ticker="KXHIGHNY-26MAR14-T56",
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable=str(DailyVariable.TEMP_MAX),
                strike_type="greater",
                floor_strike=56.0,
                cap_strike=None,
                result="yes",
                expiration_ts=exp_ts,
                expiration_date=exp_date,
            )
        ]

        mock_weather = AsyncMock()
        mock_weather.get_ensemble_forecast = AsyncMock(
            side_effect=Exception("API timeout")
        )
        # Simulate async context manager
        mock_weather.__aenter__ = AsyncMock(return_value=mock_weather)
        mock_weather.__aexit__ = AsyncMock(return_value=False)

        forecasts = asyncio.run(fetch_weather_forecasts(mock_weather, records))

        # Should return empty dict (the failed fetch is skipped)
        assert forecasts == {}

    def test_deduplicates_city_date_variable(self) -> None:
        """Multiple markets for same city+date+variable = one forecast call."""
        exp_ts = NOW_TS - 1 * 24 * 3600
        exp_date = datetime.fromtimestamp(exp_ts, tz=UTC).date().isoformat()

        # Two markets for same city/date/variable
        records = [
            MarketRecord(
                ticker=f"KXHIGHNY-T{i}",
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable=str(DailyVariable.TEMP_MAX),
                strike_type="greater",
                floor_strike=float(50 + i),
                cap_strike=None,
                result="yes",
                expiration_ts=exp_ts,
                expiration_date=exp_date,
            )
            for i in range(3)
        ]

        from python.data.weather import EnsembleForecast

        mock_fc = EnsembleForecast(
            city="NYC",
            forecast_date=date.fromisoformat(exp_date),
            temperatures=[70.0] * 10,
        )

        mock_weather = AsyncMock()
        mock_weather.get_ensemble_forecast = AsyncMock(return_value=mock_fc)
        mock_weather.__aenter__ = AsyncMock(return_value=mock_weather)
        mock_weather.__aexit__ = AsyncMock(return_value=False)

        forecasts = asyncio.run(fetch_weather_forecasts(mock_weather, records))

        # Only one unique key, so get_ensemble_forecast called exactly once
        mock_weather.get_ensemble_forecast.assert_called_once()
        assert len(forecasts) == 1
