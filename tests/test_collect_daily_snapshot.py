"""
Tests for scripts/collect_daily_snapshot.py.

Covers:
- Correct snapshot directory structure is created (markets.json, weather_forecasts.json,
  index.jsonl) with valid content.
- Markets are collected for all WEATHER_SERIES tickers; skips series that raise.
- Forecast is fetched for the fixed (city, var, date) grid (decoupled from
  Kalshi market listings — see _fixed_forecast_keys docstring).
- Dry-run skips all API calls.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from python.data.weather import EnsembleForecast, NbmPointForecast
from python.strategies.weather_edge import WEATHER_SERIES

# Script lives in scripts/ — import by module path.
# pyproject.toml roots the project at the repo root, so we can import directly.
from scripts.collect_daily_snapshot import (
    CollectionStats,
    ForecastSnapshot,
    MarketSnapshot,
    NbmSnapshot,
    _fixed_forecast_keys,
    fetch_nbm_forecasts,
    fetch_open_weather_markets,
    fetch_weather_forecasts,
    save_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_market(
    ticker: str = "KXHIGHNY-26MAR14-T56",
    event_ticker: str = "KXHIGHNY-26MAR14",
    strike_type: str = "greater",
    floor_strike: float | None = 56.0,
    cap_strike: float | None = None,
    yes_bid_dollars: str = "0.40",
    yes_ask_dollars: str = "0.50",
    volume_fp: str = "500.00",
    expiration_time: int | None = None,
) -> dict[str, Any]:
    """Build a minimal raw Kalshi market dict for testing."""
    if expiration_time is None:
        # 2026-03-14 noon UTC
        expiration_time = int(datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC).timestamp())
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "yes_bid_dollars": yes_bid_dollars,
        "yes_ask_dollars": yes_ask_dollars,
        "volume_fp": volume_fp,
        "expiration_time": expiration_time,
    }


def _make_ensemble_forecast(
    city: str = "NYC",
    forecast_date: date = date(2026, 3, 14),
    temps: list[float] | None = None,
    model: str = "gfs_seamless",
) -> EnsembleForecast:
    if temps is None:
        temps = [70.0 + i * 0.5 for i in range(31)]
    return EnsembleForecast(city=city, forecast_date=forecast_date, temperatures=temps, model=model)


def _make_market_snapshot(
    city: str,
    daily_variable: str,
    expiration_ts: int,
    ticker: str = "KXHIGHNY-26MAR14-T56",
) -> MarketSnapshot:
    """Build a MarketSnapshot for testing forecast/NBM deduplication."""
    return MarketSnapshot(
        ticker=ticker,
        series_ticker="KXHIGHNY",
        city=city,
        daily_variable=daily_variable,
        strike_type="greater",
        floor_strike=56.0,
        cap_strike=None,
        yes_bid_dollars=0.40,
        yes_ask_dollars=0.50,
        volume_fp=500,
        expiration_ts=expiration_ts,
        snapshot_ts=time.time(),
    )


# ---------------------------------------------------------------------------
# 1. save_snapshot creates correct directory structure
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    """save_snapshot writes markets.json, weather_forecasts.json, and index.jsonl."""

    def _make_market_snap(self, ticker: str = "KXHIGHNY-26MAR14-T56") -> MarketSnapshot:
        return MarketSnapshot(
            ticker=ticker,
            series_ticker="KXHIGHNY",
            city="NYC",
            daily_variable="temperature_2m_max",
            strike_type="greater",
            floor_strike=56.0,
            cap_strike=None,
            yes_bid_dollars=0.40,
            yes_ask_dollars=0.50,
            volume_fp=500,
            expiration_ts=int(datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC).timestamp()),
            snapshot_ts=time.time(),
        )

    def _make_forecast_snap(self) -> ForecastSnapshot:
        return ForecastSnapshot(
            city="NYC",
            forecast_date="2026-03-14",
            daily_variable="temperature_2m_max",
            temperatures=[70.0 + i * 0.5 for i in range(31)],
            mean=77.5,
            std_dev=4.5,
            fetched_at=time.time(),
            model="gfs_seamless",
        )

    def test_creates_date_subdirectory(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[self._make_market_snap()],
            forecasts=[self._make_forecast_snap()],
            settlements=[],
            run_ts=time.time(),
        )
        assert (tmp_path / "2026-03-15").is_dir()

    def test_markets_json_created(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        snap = self._make_market_snap()
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[snap],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
        )
        markets_path = tmp_path / "2026-03-15" / "markets.json"
        assert markets_path.exists()
        data = json.loads(markets_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["ticker"] == snap.ticker
        assert data[0]["city"] == "NYC"
        assert data[0]["series_ticker"] == "KXHIGHNY"

    def test_weather_forecasts_json_created(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        fc = self._make_forecast_snap()
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[],
            forecasts=[fc],
            settlements=[],
            run_ts=time.time(),
        )
        forecasts_path = tmp_path / "2026-03-15" / "weather_forecasts.json"
        assert forecasts_path.exists()
        data = json.loads(forecasts_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["city"] == "NYC"
        assert data[0]["forecast_date"] == "2026-03-14"
        assert len(data[0]["temperatures"]) == 31
        assert data[0]["model"] == "gfs_seamless"

    def test_index_jsonl_appended(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        run_ts = 1_742_000_000.0
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[self._make_market_snap()],
            forecasts=[self._make_forecast_snap()],
            settlements=[],
            run_ts=run_ts,
        )
        index_path = tmp_path / "index.jsonl"
        assert index_path.exists()
        lines = index_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["date"] == "2026-03-15"
        assert entry["market_count"] == 1
        assert entry["forecast_count"] == 1
        assert entry["run_ts"] == run_ts

    def test_index_jsonl_appends_across_runs(self, tmp_path: Path) -> None:
        """Two calls should produce two lines in index.jsonl."""
        for run_date in [date(2026, 3, 15), date(2026, 3, 16)]:
            save_snapshot(
                output_dir=tmp_path,
                run_date=run_date,
                markets=[],
                forecasts=[],
                settlements=[],
                run_ts=time.time(),
            )
        index_path = tmp_path / "index.jsonl"
        lines = [ln for ln in index_path.read_text().strip().split("\n") if ln]
        assert len(lines) == 2

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        """Running twice for the same date overwrites markets.json cleanly."""
        run_date = date(2026, 3, 15)

        # First run — 1 market
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[self._make_market_snap("KXHIGHNY-26MAR14-T56")],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
        )

        # Second run — 2 markets (overwrite)
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[
                self._make_market_snap("KXHIGHNY-26MAR14-T56"),
                self._make_market_snap("KXHIGHNY-26MAR14-T60"),
            ],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
        )

        markets_path = tmp_path / "2026-03-15" / "markets.json"
        data = json.loads(markets_path.read_text())
        # Second write should have 2 entries, not 3
        assert len(data) == 2


# ---------------------------------------------------------------------------
# 2. fetch_open_weather_markets — filters via status="open", iterates all series
# ---------------------------------------------------------------------------


class TestFetchOpenWeatherMarkets:
    """fetch_open_weather_markets calls iter_all_markets with status='open' per series."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio
        monkeypatch.setattr(_asyncio, "sleep", AsyncMock(return_value=None))

    @pytest.mark.asyncio
    async def test_queries_each_series_with_open_status(self) -> None:
        """Every WEATHER_SERIES ticker is queried with status='open'."""
        mock_client = AsyncMock()
        mock_client.iter_all_markets = AsyncMock(return_value=([], False))

        await fetch_open_weather_markets(mock_client, snapshot_ts=time.time())

        # Should have been called once per series
        assert mock_client.iter_all_markets.call_count == len(WEATHER_SERIES)

        # Every call must use status="open"
        for call in mock_client.iter_all_markets.call_args_list:
            assert call.kwargs.get("status") == "open"

    @pytest.mark.asyncio
    async def test_each_series_ticker_used(self) -> None:
        """The series_ticker kwarg matches each key in WEATHER_SERIES."""
        mock_client = AsyncMock()
        mock_client.iter_all_markets = AsyncMock(return_value=([], False))

        await fetch_open_weather_markets(mock_client, snapshot_ts=time.time())

        queried = {
            call.kwargs.get("series_ticker") for call in mock_client.iter_all_markets.call_args_list
        }
        assert queried == set(WEATHER_SERIES.keys())

    @pytest.mark.asyncio
    async def test_returns_market_snapshots(self) -> None:
        """Markets returned by the API become MarketSnapshot objects."""
        raw = [_make_raw_market()]
        mock_client = AsyncMock()
        mock_client.iter_all_markets = AsyncMock(return_value=(raw, False))

        snapshots = await fetch_open_weather_markets(mock_client, snapshot_ts=12345.0)

        # 11 series * 1 market each
        assert len(snapshots) == len(WEATHER_SERIES)
        first = snapshots[0]
        assert isinstance(first, MarketSnapshot)
        assert first.ticker == "KXHIGHNY-26MAR14-T56"
        assert first.snapshot_ts == 12345.0

    @pytest.mark.asyncio
    async def test_series_failure_skipped_gracefully(self) -> None:
        """If one series raises, the others still succeed."""
        call_count = 0

        async def side_effect(
            *, status: str, series_ticker: str
        ) -> tuple[list[dict[str, Any]], bool]:
            nonlocal call_count
            call_count += 1
            if series_ticker == "KXHIGHNY":
                raise RuntimeError("simulated API error")
            return [_make_raw_market()], False

        mock_client = AsyncMock()
        mock_client.iter_all_markets = side_effect

        snapshots = await fetch_open_weather_markets(mock_client, snapshot_ts=time.time())

        # All series were attempted
        assert call_count == len(WEATHER_SERIES)
        # KXHIGHNY contributed 0; the remaining 10 series contributed 1 each
        assert len(snapshots) == len(WEATHER_SERIES) - 1

    @pytest.mark.asyncio
    async def test_snapshot_carries_city_and_variable(self) -> None:
        """City and daily_variable are taken from WEATHER_SERIES, not market dict."""
        raw = [_make_raw_market(ticker="KXHIGHNY-26MAR14-T56")]
        mock_client = AsyncMock()

        async def side_effect(
            *, status: str, series_ticker: str
        ) -> tuple[list[dict[str, Any]], bool]:
            if series_ticker == "KXHIGHNY":
                return raw, False
            return [], False

        mock_client.iter_all_markets = side_effect

        snapshots = await fetch_open_weather_markets(mock_client, snapshot_ts=time.time())
        ny_snaps = [s for s in snapshots if s.series_ticker == "KXHIGHNY"]
        assert len(ny_snaps) == 1
        assert ny_snaps[0].city == "NYC"
        assert ny_snaps[0].daily_variable == "temperature_2m_max"


# ---------------------------------------------------------------------------
# 3a. _fixed_forecast_keys — fixed (city, var, date) grid
# ---------------------------------------------------------------------------


class TestFixedForecastKeys:
    """_fixed_forecast_keys returns a deterministic N×2×3 grid (currently 20 cities)."""

    def test_grid_has_seven_cities_two_vars_three_dates(self) -> None:
        keys = _fixed_forecast_keys(date(2026, 3, 14))
        # 20 cities × 2 variables × 3 dates = 120
        assert len(keys) == 120

    def test_grid_includes_sea_temperature_min(self) -> None:
        """Regression: SEA-low must be collected even when no Kalshi series exists."""
        keys = _fixed_forecast_keys(date(2026, 3, 14))
        assert any(
            city == "SEA" and var == "temperature_2m_min" for (city, _, var) in keys
        )

    def test_grid_uses_three_day_horizon(self) -> None:
        run = date(2026, 3, 14)
        dates = {fd for (_, fd, _) in _fixed_forecast_keys(run)}
        assert dates == {run, date(2026, 3, 15), date(2026, 3, 16)}


# ---------------------------------------------------------------------------
# 3b. fetch_weather_forecasts — fetches the fixed grid
# ---------------------------------------------------------------------------


class TestFetchWeatherForecasts:
    """fetch_weather_forecasts fetches one forecast per (city, var, date) grid key."""

    @pytest.mark.asyncio
    async def test_fetches_full_fixed_grid(self) -> None:
        """One call per grid key (20 cities × 2 vars × 3 dates = 120), independent of markets."""
        run_date = date(2026, 3, 14)

        mock_weather = AsyncMock()
        mock_weather.get_multi_model_forecast = AsyncMock(
            side_effect=lambda city, forecast_date, daily_variable, models: {
                "gfs_seamless": _make_ensemble_forecast(city, forecast_date)
            }
        )

        forecasts = await fetch_weather_forecasts(mock_weather, run_date)

        assert mock_weather.get_multi_model_forecast.call_count == 120
        assert len(forecasts) == 120
        # All 20 cities represented
        cities = {fc.city for fc in forecasts}
        assert len(cities) == 20
        # SEA temperature_2m_min specifically present (regression case)
        assert any(
            fc.city == "SEA" and fc.daily_variable == "temperature_2m_min" for fc in forecasts
        )

    @pytest.mark.asyncio
    async def test_forecast_failure_skipped_gracefully(self) -> None:
        """If the outer fetch raises for a city, those keys are skipped; others succeed."""
        run_date = date(2026, 3, 14)

        async def side_effect(
            city: str, forecast_date: date, daily_variable: Any, models: Any
        ) -> dict[str, EnsembleForecast]:
            if city == "NYC":
                raise RuntimeError("simulated weather API failure")
            return {"gfs_seamless": _make_ensemble_forecast(city, forecast_date)}

        mock_weather = AsyncMock()
        mock_weather.get_multi_model_forecast = side_effect

        forecasts = await fetch_weather_forecasts(mock_weather, run_date)

        # NYC has 2 vars × 3 dates = 6 failed keys; remaining 19 cities × 2 × 3 = 114 succeed
        assert len(forecasts) == 114
        assert all(fc.city != "NYC" for fc in forecasts)

    @pytest.mark.asyncio
    async def test_forecast_snapshot_fields(self) -> None:
        """ForecastSnapshot fields are populated from EnsembleForecast."""
        run_date = date(2026, 3, 14)

        async def side_effect(
            city: str, forecast_date: date, daily_variable: Any, models: Any
        ) -> dict[str, EnsembleForecast]:
            temps = [65.0 + i * 0.5 for i in range(31)]
            return {
                "gfs_seamless": EnsembleForecast(
                    city=city,
                    forecast_date=forecast_date,
                    temperatures=temps,
                    model="gfs_seamless",
                )
            }

        mock_weather = AsyncMock()
        mock_weather.get_multi_model_forecast = side_effect

        forecasts = await fetch_weather_forecasts(mock_weather, run_date)

        # Pick the (NYC, temperature_2m_max, run_date) snapshot for assertions
        snap = next(
            fc
            for fc in forecasts
            if fc.city == "NYC"
            and fc.daily_variable == "temperature_2m_max"
            and fc.forecast_date == "2026-03-14"
        )
        assert len(snap.temperatures) == 31
        assert snap.mean == pytest.approx(sum(snap.temperatures) / len(snap.temperatures))
        assert snap.model == "gfs_seamless"
        assert snap.model_version == "GFSv16"

    @pytest.mark.asyncio
    async def test_two_model_fetch(self) -> None:
        """Two models per grid key → 120 calls, 240 snapshots, both models represented."""
        run_date = date(2026, 3, 14)

        async def side_effect(
            city: str, forecast_date: date, daily_variable: Any, models: Any
        ) -> dict[str, EnsembleForecast]:
            return {
                "gfs_seamless": _make_ensemble_forecast(
                    city, forecast_date, model="gfs_seamless"
                ),
                "ecmwf_ifs025": _make_ensemble_forecast(
                    city,
                    forecast_date,
                    temps=[68.0 + i * 0.4 for i in range(51)],
                    model="ecmwf_ifs025",
                ),
            }

        mock_weather = AsyncMock()
        mock_weather.get_multi_model_forecast = AsyncMock(side_effect=side_effect)

        forecasts = await fetch_weather_forecasts(
            mock_weather, run_date, models=["gfs_seamless", "ecmwf_ifs025"]
        )

        assert mock_weather.get_multi_model_forecast.call_count == 120
        assert len(forecasts) == 240
        models_found = {snap.model for snap in forecasts}
        assert models_found == {"gfs_seamless", "ecmwf_ifs025"}
        # Spot-check model_version propagation
        a_gfs = next(s for s in forecasts if s.model == "gfs_seamless")
        an_ecmwf = next(s for s in forecasts if s.model == "ecmwf_ifs025")
        assert a_gfs.model_version == "GFSv16"
        assert an_ecmwf.model_version == "IFS-49r1"


# ---------------------------------------------------------------------------
# 4. Dry-run skips all API calls
# ---------------------------------------------------------------------------


class TestDryRun:
    """When dry_run=True, main() logs intent but never touches KalshiClient or WeatherClient."""

    @pytest.mark.asyncio
    async def test_dry_run_makes_no_api_calls(self, tmp_path: Path, capsys: Any) -> None:
        """main(dry_run=True) exits without calling any API client."""
        from scripts.collect_daily_snapshot import main

        with (
            patch("scripts.collect_daily_snapshot.KalshiClient") as mock_kalshi_cls,
            patch("scripts.collect_daily_snapshot.WeatherClient") as mock_weather_cls,
            patch("scripts.collect_daily_snapshot.NWSClient") as mock_nws_cls,
        ):
            await main(output_dir=tmp_path, dry_run=True)

            # No client should have been instantiated (context managers not entered)
            mock_kalshi_cls.assert_not_called()
            mock_weather_cls.assert_not_called()
            mock_nws_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_creates_no_files(self, tmp_path: Path) -> None:
        """Dry-run does not write any files to disk."""
        from scripts.collect_daily_snapshot import main

        with (
            patch("scripts.collect_daily_snapshot.KalshiClient"),
            patch("scripts.collect_daily_snapshot.WeatherClient"),
            patch("scripts.collect_daily_snapshot.NWSClient"),
        ):
            await main(output_dir=tmp_path, dry_run=True)

        # Only the tmp_path itself should exist; no subdirectories or files created
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 5. fetch_nbm_forecasts — one NBM point forecast per unique (city, date, variable)
# ---------------------------------------------------------------------------


def _make_nbm_point_forecast(
    city: str = "NYC",
    forecast_date: date = date(2026, 3, 14),
    daily_variable: str = "temperature_2m_max",
    temperature: float = 72.5,
) -> NbmPointForecast:
    return NbmPointForecast(
        city=city,
        forecast_date=forecast_date,
        daily_variable=daily_variable,
        temperature=temperature,
    )


class TestFetchNbmForecasts:
    """fetch_nbm_forecasts fetches NBM forecasts for the fixed grid."""

    @pytest.mark.asyncio
    async def test_fetches_full_fixed_grid(self) -> None:
        """One NBM call per grid key (20 cities × 2 vars × 3 dates = 120)."""
        run_date = date(2026, 3, 14)

        mock_weather = AsyncMock()
        mock_weather.get_nbm_point_forecast = AsyncMock(
            side_effect=lambda city, forecast_date, daily_variable: _make_nbm_point_forecast(
                city, forecast_date, daily_variable=str(daily_variable)
            )
        )

        nbm_snaps = await fetch_nbm_forecasts(mock_weather, run_date)

        assert mock_weather.get_nbm_point_forecast.call_count == 120
        assert len(nbm_snaps) == 120
        cities = {s.city for s in nbm_snaps}
        assert len(cities) == 20
        assert any(
            s.city == "SEA" and s.daily_variable == "temperature_2m_min" for s in nbm_snaps
        )

    @pytest.mark.asyncio
    async def test_failure_skipped_gracefully(self) -> None:
        """If NBM fetch raises for one city, others still succeed."""
        run_date = date(2026, 3, 14)

        async def side_effect(
            city: str, forecast_date: date, daily_variable: Any
        ) -> NbmPointForecast:
            if city == "NYC":
                raise RuntimeError("simulated NBM API failure")
            return _make_nbm_point_forecast(
                city, forecast_date, daily_variable=str(daily_variable)
            )

        mock_weather = AsyncMock()
        mock_weather.get_nbm_point_forecast = side_effect

        nbm_snaps = await fetch_nbm_forecasts(mock_weather, run_date)

        # NYC has 6 keys (2 vars × 3 dates); remaining 19 cities × 6 keys = 114 succeed
        assert len(nbm_snaps) == 114
        assert all(s.city != "NYC" for s in nbm_snaps)

    @pytest.mark.asyncio
    async def test_nbm_snapshot_fields(self) -> None:
        """NbmSnapshot fields are populated from NbmPointForecast."""
        run_date = date(2026, 3, 14)

        async def side_effect(
            city: str, forecast_date: date, daily_variable: Any
        ) -> NbmPointForecast:
            return _make_nbm_point_forecast(
                city, forecast_date, daily_variable=str(daily_variable), temperature=72.5
            )

        mock_weather = AsyncMock()
        mock_weather.get_nbm_point_forecast = side_effect

        nbm_snaps = await fetch_nbm_forecasts(mock_weather, run_date)

        snap = next(
            s
            for s in nbm_snaps
            if s.city == "NYC"
            and s.daily_variable == "temperature_2m_max"
            and s.forecast_date == "2026-03-14"
        )
        assert snap.temperature == pytest.approx(72.5)


# ---------------------------------------------------------------------------
# 6. save_snapshot includes NBM forecasts
# ---------------------------------------------------------------------------


class TestSaveSnapshotNbm:
    """save_snapshot writes nbm_forecasts.json when NBM data is provided."""

    def _make_nbm_snap(self) -> NbmSnapshot:
        return NbmSnapshot(
            city="NYC",
            forecast_date="2026-03-14",
            daily_variable="temperature_2m_max",
            temperature=72.5,
            fetched_at=time.time(),
        )

    def test_nbm_forecasts_json_created(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        nbm_snap = self._make_nbm_snap()
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
            nbm_forecasts=[nbm_snap],
        )
        nbm_path = tmp_path / "2026-03-15" / "nbm_forecasts.json"
        assert nbm_path.exists()
        data = json.loads(nbm_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["city"] == "NYC"
        assert data[0]["temperature"] == pytest.approx(72.5)
        assert data[0]["forecast_date"] == "2026-03-14"

    def test_nbm_omitted_when_empty(self, tmp_path: Path) -> None:
        """nbm_forecasts.json is not created when no NBM data is collected."""
        run_date = date(2026, 3, 15)
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
        )
        nbm_path = tmp_path / "2026-03-15" / "nbm_forecasts.json"
        assert not nbm_path.exists()

    def test_index_entry_includes_nbm_count(self, tmp_path: Path) -> None:
        run_date = date(2026, 3, 15)
        save_snapshot(
            output_dir=tmp_path,
            run_date=run_date,
            markets=[],
            forecasts=[],
            settlements=[],
            run_ts=time.time(),
            nbm_forecasts=[self._make_nbm_snap(), self._make_nbm_snap()],
        )
        index_path = tmp_path / "index.jsonl"
        entry = json.loads(index_path.read_text().strip())
        assert entry["nbm_count"] == 2


# ---------------------------------------------------------------------------
# Task 1: Atomic writes
# ---------------------------------------------------------------------------


def test_save_snapshot_writes_valid_json_and_no_tmp_remnants(tmp_path: Path) -> None:
    save_snapshot(
        output_dir=tmp_path, run_date=date(2026, 6, 9),
        markets=[], forecasts=[], settlements=[], run_ts=0.0, run_id="0200",
    )
    d = tmp_path / "2026-06-09" / "0200"
    assert json.loads((d / "markets.json").read_text()) == []
    assert json.loads((d / "weather_forecasts.json").read_text()) == []
    # No leftover atomic tempfiles
    assert not list(d.glob("*.tmp"))
    assert not list(tmp_path.glob("**/*.tmp"))


def test_save_snapshot_uses_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.collect_daily_snapshot as cds

    calls: list[str] = []
    real = cds._atomic_write_json
    monkeypatch.setattr(cds, "_atomic_write_json",
                        lambda p, d: (calls.append(p.name), real(p, d))[1])
    cds.save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                      markets=[], forecasts=[], settlements=[], run_ts=0.0, run_id="0200")
    assert "markets.json" in calls and "weather_forecasts.json" in calls


# ---------------------------------------------------------------------------
# Task 2: CollectionStats + integrity status
# ---------------------------------------------------------------------------


def _read_index(tmp_path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return rows


def test_status_degraded_when_no_forecasts(tmp_path: Path) -> None:
    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                  markets=[], forecasts=[], settlements=[], run_ts=0.0, run_id="0200")
    row = _read_index(tmp_path)[0]
    assert row["status"] == "degraded"  # forecast_count == 0 (and market_count == 0)


def test_status_ok_on_clean_run(tmp_path: Path) -> None:
    # reuse existing snapshot builders from TestSaveSnapshot
    tc = TestSaveSnapshot()
    m = tc._make_market_snap()
    fc = tc._make_forecast_snap()

    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                  markets=[m], forecasts=[fc], settlements=[], run_ts=time.time(),
                  run_id="0200")
    row = _read_index(tmp_path)[0]
    assert row["status"] == "ok"


def test_status_ok_when_only_nbm_failed(tmp_path: Path) -> None:
    """NBM failures are recorded in stats but do NOT degrade status."""
    tc = TestSaveSnapshot()
    m = tc._make_market_snap()
    fc = tc._make_forecast_snap()
    stats = CollectionStats(nbm_failed=3)

    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                  markets=[m], forecasts=[fc], settlements=[], run_ts=time.time(),
                  run_id="0200", stats=stats)
    row = _read_index(tmp_path)[0]
    assert row["status"] == "ok"
    assert row["stats"]["nbm_failed"] == 3


def test_status_degraded_when_markets_truncated(tmp_path: Path) -> None:
    tc = TestSaveSnapshot()
    m = tc._make_market_snap()
    fc = tc._make_forecast_snap()
    stats = CollectionStats(markets_truncated=1)

    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                  markets=[m], forecasts=[fc], settlements=[], run_ts=time.time(),
                  run_id="0200", stats=stats)
    row = _read_index(tmp_path)[0]
    assert row["status"] == "degraded"


# ---------------------------------------------------------------------------
# Task 3: Index idempotency
# ---------------------------------------------------------------------------


def test_index_upsert_is_idempotent(tmp_path: Path) -> None:
    for _ in range(2):
        save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                      markets=[], forecasts=[], settlements=[], run_ts=1.0, run_id="0200")
    rows = _read_index(tmp_path)
    same = [r for r in rows if r["date"] == "2026-06-09" and r["run_id"] == "0200"]
    assert len(same) == 1  # second run overwrote, not appended


def test_index_keeps_distinct_run_ids(tmp_path: Path) -> None:
    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9), markets=[], forecasts=[],
                  settlements=[], run_ts=1.0, run_id="0200")
    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9), markets=[], forecasts=[],
                  settlements=[], run_ts=1.0, run_id="1400")
    assert len({(r["date"], r["run_id"]) for r in _read_index(tmp_path)}) == 2


def test_index_upsert_preserves_legacy_rows(tmp_path: Path) -> None:
    idx = tmp_path / "index.jsonl"
    idx.write_text(json.dumps({"run_ts": 0.0, "date": "2026-06-08",
                               "market_count": 5, "forecast_count": 5,
                               "settlement_count": 0, "nbm_count": 0, "run_id": "0200"}) + "\n")
    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9), markets=[], forecasts=[],
                  settlements=[], run_ts=1.0, run_id="0200")
    rows = _read_index(tmp_path)
    assert any(r["date"] == "2026-06-08" for r in rows)  # legacy row preserved
    assert any(r["date"] == "2026-06-09" for r in rows)


def test_index_upsert_idempotent_no_run_id(tmp_path: Path) -> None:
    for _ in range(2):
        save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                      markets=[], forecasts=[], settlements=[], run_ts=1.0)
    rows = _read_index(tmp_path)
    same = [r for r in rows if r["date"] == "2026-06-09" and r.get("run_id") is None]
    assert len(same) == 1


def test_index_corrupt_line_tolerated(tmp_path: Path) -> None:
    idx = tmp_path / "index.jsonl"
    good = json.dumps({"run_ts": 0.0, "date": "2026-06-08", "market_count": 5,
                       "forecast_count": 5, "settlement_count": 0, "nbm_count": 0,
                       "run_id": "0200"})
    idx.write_text(good + "\n{bad json\n")
    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9), markets=[], forecasts=[],
                  settlements=[], run_ts=1.0, run_id="0200")
    rows = _read_index(tmp_path)
    assert len(rows) == 2  # good legacy row + new row; corrupt line dropped
    assert any(r["date"] == "2026-06-08" for r in rows)


def test_status_degraded_when_forecasts_failed(tmp_path: Path) -> None:
    tc = TestSaveSnapshot()
    m = tc._make_market_snap()
    fc = tc._make_forecast_snap()
    stats = CollectionStats(forecasts_failed=1)

    save_snapshot(output_dir=tmp_path, run_date=date(2026, 6, 9),
                  markets=[m], forecasts=[fc], settlements=[], run_ts=time.time(),
                  run_id="0200", stats=stats)
    row = _read_index(tmp_path)[0]
    assert row["status"] == "degraded"
