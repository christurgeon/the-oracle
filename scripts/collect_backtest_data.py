"""
Collect historical weather market data from Kalshi for backtesting.

Fetches closed weather markets, their candlestick price history, and the
corresponding GFS ensemble weather forecasts from Open-Meteo, then saves
everything to a local directory for use in backtesting.

Usage::

    uv run python scripts/collect_backtest_data.py [--days N] [--output DIR]
        [--interval MINUTES] [--dry-run]

TODO: Run this script with real credentials to pull the past 7 days of closed
weather markets + candlestick prices + GFS forecasts before building the
backtest engine. Requires ORACLE_KALSHI_API_KEY and
ORACLE_KALSHI_PRIVATE_KEY_PATH to be set in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path so we can import python.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.common import DailyVariable  # noqa: E402
from python.data.kalshi_client import KalshiAPIError, KalshiClient  # noqa: E402
from python.data.weather import WeatherClient  # noqa: E402
from python.strategies.weather_edge import WEATHER_SERIES  # noqa: E402

logger = logging.getLogger(__name__)

# Maximum retries and base delay for Kalshi 429 backoff
_MAX_RETRIES = 4
_BASE_DELAY = 15.0  # seconds; doubles each retry → 15, 30, 60, 120

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CandlestickRecord:
    """A single OHLC candlestick from the Kalshi API."""

    end_period_ts: int
    yes_price_open: int | None = None
    yes_price_high: int | None = None
    yes_price_low: int | None = None
    yes_price_close: int | None = None
    no_price_open: int | None = None
    no_price_high: int | None = None
    no_price_low: int | None = None
    no_price_close: int | None = None
    volume: int | None = None
    open_interest: int | None = None


@dataclass
class MarketRecord:
    """A closed Kalshi weather market with embedded candlestick history."""

    ticker: str
    series_ticker: str
    city: str
    daily_variable: str
    strike_type: str | None
    floor_strike: float | None
    cap_strike: float | None
    result: str | None           # "yes" | "no" | None (unsettled)
    expiration_ts: int
    expiration_date: str         # ISO date derived from expiration_ts
    candlesticks: list[CandlestickRecord] = field(default_factory=list)
    candlestick_error: str | None = None  # set if fetch failed


@dataclass
class WeatherForecastRecord:
    """GFS ensemble forecast stored for backtesting."""

    city: str
    forecast_date: str           # ISO date
    daily_variable: str
    temperatures: list[float]
    mean_temperature: float
    std_dev: float
    member_count: int
    fetched_at: float


@dataclass
class CollectionMetadata:
    """Summary metadata for a collection run."""

    run_timestamp: str
    days_covered: int
    cutoff_ts: int               # markets older than this are excluded
    market_count: int
    series_count: int
    weather_forecast_count: int
    interval_minutes: int
    dry_run: bool
    series_tickers: list[str]


# ---------------------------------------------------------------------------
# Core fetch functions (testable — no global state)
# ---------------------------------------------------------------------------


def _parse_expiration_ts(market: dict[str, Any]) -> int:
    """Extract a unix timestamp from whichever expiration field exists.

    Kalshi API v2 uses ``expiration_time`` (ISO 8601 string).  Older cached
    data may still carry ``expiration_ts`` (int).  Returns 0 if neither is
    usable.
    """
    # Prefer expected_expiration_time (actual settlement) over expiration_time
    # (latest possible, often a week later for weather markets)
    exp_str = market.get("expected_expiration_time") or market.get("expiration_time")
    if exp_str:
        try:
            dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            pass

    # Fallback for cached/older data
    raw = market.get("expiration_ts")
    if raw is not None:
        try:
            return int(raw)
        except (ValueError, TypeError):
            pass
    return 0


def filter_markets_by_age(
    markets: list[dict[str, Any]],
    cutoff_ts: int,
) -> list[dict[str, Any]]:
    """
    Return only markets whose expiration time is >= cutoff_ts.

    Args:
        markets: Raw market dicts from the Kalshi API.
        cutoff_ts: Unix timestamp (seconds) for the earliest allowed expiration.

    Returns:
        Filtered list of market dicts.
    """
    result: list[dict[str, Any]] = []
    for market in markets:
        exp_ts = _parse_expiration_ts(market)
        if exp_ts == 0:
            logger.debug("Skipping market with no expiration time: %s", market.get("ticker"))
            continue
        if exp_ts >= cutoff_ts:
            result.append(market)
    return result


def build_market_record(
    market: dict[str, Any],
    series_ticker: str,
    city: str,
    daily_variable: DailyVariable,
) -> MarketRecord:
    """
    Extract a MarketRecord from a raw Kalshi market dict.

    Args:
        market: Raw market dict from the Kalshi API.
        series_ticker: Series this market belongs to (e.g. "KXHIGHNY").
        city: City code derived from WEATHER_SERIES.
        daily_variable: Open-Meteo variable for this series.

    Returns:
        MarketRecord with all available fields populated.
    """
    exp_ts = _parse_expiration_ts(market)
    exp_date = (
        datetime.fromtimestamp(exp_ts, tz=UTC).date().isoformat()
        if exp_ts
        else ""
    )

    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    # Kalshi returns "" for unsettled markets, normalize to None
    raw_result = market.get("result")
    result_val = raw_result if raw_result else None

    return MarketRecord(
        ticker=market.get("ticker", ""),
        series_ticker=series_ticker,
        city=city,
        daily_variable=str(daily_variable),
        strike_type=market.get("strike_type"),
        floor_strike=float(floor_strike) if floor_strike is not None else None,
        cap_strike=float(cap_strike) if cap_strike is not None else None,
        result=result_val,
        expiration_ts=exp_ts,
        expiration_date=exp_date,
    )


def parse_candlestick(raw: dict[str, Any]) -> CandlestickRecord:
    """Parse a single candlestick dict from the Kalshi API response.

    Handles both the current API format (dollar strings in nested objects
    like ``price.close_dollars``, ``yes_ask.close_dollars``) and the legacy
    format (integer cents in ``yes_price.close``).
    """

    def _dollars_to_cents(val: str | int | float | None) -> int | None:
        """Convert a dollar string like '0.6500' to integer cents (65)."""
        if val is None:
            return None
        try:
            return int(round(float(val) * 100))
        except (ValueError, TypeError):
            return None

    def _int(val: str | int | float | None) -> int | None:
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _int_fp(val: str | int | float | None) -> int | None:
        """Parse an '_fp' field like '1624.00' to int."""
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    # --- Current API format: dollar strings in nested objects ---
    price = raw.get("price", {}) or {}
    yes_ask = raw.get("yes_ask", {}) or {}
    yes_bid = raw.get("yes_bid", {}) or {}

    if price.get("close_dollars") is not None:
        # Use midpoint of yes_bid and yes_ask as the YES price
        bid_close = _dollars_to_cents(yes_bid.get("close_dollars"))
        ask_close = _dollars_to_cents(yes_ask.get("close_dollars"))
        # Fall back to price.close_dollars if bid/ask not available
        price_close = _dollars_to_cents(price.get("close_dollars"))

        # Compute yes_price as midpoint of bid/ask, or use trade price
        if bid_close is not None and ask_close is not None:
            yes_close = (bid_close + ask_close) // 2
        elif price_close is not None:
            yes_close = price_close
        else:
            yes_close = None

        bid_open = _dollars_to_cents(yes_bid.get("open_dollars"))
        ask_open = _dollars_to_cents(yes_ask.get("open_dollars"))
        price_open = _dollars_to_cents(price.get("open_dollars"))
        if bid_open is not None and ask_open is not None:
            yes_open = (bid_open + ask_open) // 2
        elif price_open is not None:
            yes_open = price_open
        else:
            yes_open = None

        return CandlestickRecord(
            end_period_ts=int(raw.get("end_period_ts", 0)),
            yes_price_open=yes_open,
            yes_price_high=_dollars_to_cents(price.get("high_dollars")),
            yes_price_low=_dollars_to_cents(price.get("low_dollars")),
            yes_price_close=yes_close,
            no_price_open=100 - yes_open if yes_open is not None else None,
            no_price_high=None,
            no_price_low=None,
            no_price_close=100 - yes_close if yes_close is not None else None,
            volume=_int_fp(raw.get("volume_fp")),
            open_interest=_int_fp(raw.get("open_interest_fp")),
        )

    # --- Legacy format: integer cents in yes_price/no_price objects ---
    yes = raw.get("yes_price", {}) or {}
    no = raw.get("no_price", {}) or {}

    return CandlestickRecord(
        end_period_ts=int(raw.get("end_period_ts", 0)),
        yes_price_open=_int(yes.get("open")),
        yes_price_high=_int(yes.get("high")),
        yes_price_low=_int(yes.get("low")),
        yes_price_close=_int(yes.get("close")),
        no_price_open=_int(no.get("open")),
        no_price_high=_int(no.get("high")),
        no_price_low=_int(no.get("low")),
        no_price_close=_int(no.get("close")),
        volume=_int(raw.get("volume")),
        open_interest=_int(raw.get("open_interest")),
    )


async def fetch_markets_for_series(
    client: KalshiClient,
    series_ticker: str,
    city: str,
    daily_variable: DailyVariable,
    cutoff_ts: int,
) -> list[MarketRecord]:
    """
    Fetch all closed markets for a single series and filter by age.

    Args:
        client: Authenticated KalshiClient.
        series_ticker: Series to query (e.g. "KXHIGHNY").
        city: City code for the series.
        daily_variable: Open-Meteo variable for the series.
        cutoff_ts: Exclude markets that expired before this timestamp.

    Returns:
        List of MarketRecord objects within the requested time window.
    """
    logger.info("Fetching settled markets for series %s (%s)", series_ticker, city)

    raw_markets, markets_truncated = await client.iter_all_markets(
        status="settled",
        series_ticker=series_ticker,
    )
    if markets_truncated:
        logger.critical(
            "iter_all_markets TRUNCATED for series %s — got %d settled markets "
            "before MAX_PAGES cap; backtest corpus is INCOMPLETE for this series",
            series_ticker, len(raw_markets),
        )
    logger.debug("  %d raw settled markets returned for %s", len(raw_markets), series_ticker)

    recent = filter_markets_by_age(raw_markets, cutoff_ts)
    logger.info("  %d markets within window for %s", len(recent), series_ticker)

    records = [
        build_market_record(m, series_ticker, city, daily_variable)
        for m in recent
    ]
    return records


async def fetch_candlesticks_for_market(
    client: KalshiClient,
    record: MarketRecord,
    interval_minutes: int,
) -> MarketRecord:
    """
    Fetch candlestick history for a market and return an updated record.

    Window: (expiration_ts - 48 hours) to expiration_ts.
    Errors are caught; the record's candlestick_error field is set on failure.

    Args:
        client: Authenticated KalshiClient.
        record: Market record to enrich with candlestick data.
        interval_minutes: Candle period in minutes (1, 60, or 1440).

    Returns:
        Updated MarketRecord with candlesticks (or candlestick_error set).
    """
    if not record.expiration_ts:
        logger.warning("No expiration_ts for %s — skipping candlesticks", record.ticker)
        return record

    start_ts = record.expiration_ts - 48 * 3600
    end_ts = record.expiration_ts

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get_candlesticks(
                series_ticker=record.series_ticker,
                ticker=record.ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=interval_minutes,
            )
            raw_candles: list[dict[str, Any]] = resp.get("candlesticks", [])
            record.candlesticks = [parse_candlestick(c) for c in raw_candles]
            logger.debug(
                "  %s: fetched %d candlesticks", record.ticker, len(record.candlesticks)
            )
            break
        except KalshiAPIError as exc:
            if exc.status == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "429 on candlestick %s — retry %d/%d in %.0fs",
                    record.ticker, attempt + 1, _MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            logger.warning(
                "Candlestick fetch failed for %s (HTTP %s): %s — skipping",
                record.ticker, exc.status, exc,
            )
            record.candlestick_error = str(exc)
            break
        except Exception as exc:
            logger.warning(
                "Unexpected error fetching candlesticks for %s: %s",
                record.ticker, exc,
            )
            record.candlestick_error = str(exc)
            break

    return record


async def fetch_weather_forecasts(
    weather_client: WeatherClient,
    market_records: list[MarketRecord],
) -> dict[str, WeatherForecastRecord]:
    """
    Fetch GFS ensemble forecasts for all unique (city, date, variable) combos.

    Args:
        weather_client: Authenticated WeatherClient.
        market_records: All collected market records.

    Returns:
        Dict keyed by "{city}:{date}:{variable}" → WeatherForecastRecord.
    """
    # Deduplicate (city, date, variable) tuples
    unique_keys: set[tuple[str, str, str]] = set()
    for rec in market_records:
        if rec.expiration_date:
            unique_keys.add((rec.city, rec.expiration_date, rec.daily_variable))

    logger.info(
        "Fetching weather forecasts for %d unique (city, date, variable) combos",
        len(unique_keys),
    )

    forecasts: dict[str, WeatherForecastRecord] = {}

    for city, date_str, variable_str in sorted(unique_keys):
        key = f"{city}:{date_str}:{variable_str}"
        try:
            forecast_date = date.fromisoformat(date_str)
            daily_variable = DailyVariable(variable_str)

            forecast = await weather_client.get_ensemble_forecast(
                city=city,
                forecast_date=forecast_date,
                daily_variable=daily_variable,
            )

            temps = forecast.temperatures
            mean = statistics.mean(temps) if temps else 0.0
            std = statistics.pstdev(temps) if len(temps) > 1 else 0.0

            forecasts[key] = WeatherForecastRecord(
                city=city,
                forecast_date=date_str,
                daily_variable=variable_str,
                temperatures=temps,
                mean_temperature=mean,
                std_dev=std,
                member_count=len(temps),
                fetched_at=forecast.fetched_at,
            )
            logger.debug(
                "  %s: mean=%.1f°F std=%.1f°F members=%d",
                key,
                mean,
                std,
                len(temps),
            )
        except Exception as exc:
            logger.warning("Failed to fetch forecast for %s: %s — skipping", key, exc)

    logger.info("Fetched %d weather forecasts", len(forecasts))
    return forecasts


# ---------------------------------------------------------------------------
# Output serialization helpers
# ---------------------------------------------------------------------------


def _record_to_dict(record: MarketRecord) -> dict[str, Any]:
    """Convert a MarketRecord (and nested CandlestickRecords) to JSON-safe dict."""
    d = asdict(record)
    # candlesticks is already serialized by asdict
    return d


def save_outputs(
    output_dir: Path,
    market_records: list[MarketRecord],
    weather_forecasts: dict[str, WeatherForecastRecord],
    metadata: CollectionMetadata,
) -> None:
    """
    Write markets.json, weather_forecasts.json, and collection_metadata.json.

    Args:
        output_dir: Directory to write files into (created if absent).
        market_records: All collected market records.
        weather_forecasts: Forecast records keyed by "{city}:{date}:{variable}".
        metadata: Run summary metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    markets_path = output_dir / "markets.json"
    forecasts_path = output_dir / "weather_forecasts.json"
    metadata_path = output_dir / "collection_metadata.json"

    markets_data = [_record_to_dict(r) for r in market_records]
    with open(markets_path, "w") as f:
        json.dump(markets_data, f, indent=2)
    logger.info("Wrote %d market records to %s", len(markets_data), markets_path)

    forecasts_data = {k: asdict(v) for k, v in weather_forecasts.items()}
    with open(forecasts_path, "w") as f:
        json.dump(forecasts_data, f, indent=2)
    logger.info("Wrote %d weather forecasts to %s", len(forecasts_data), forecasts_path)

    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    logger.info("Wrote collection metadata to %s", metadata_path)


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------


async def run_collection(
    days: int,
    output_dir: Path,
    interval_minutes: int,
    dry_run: bool,
    kalshi_api_key: str,
    kalshi_private_key_path: str,
    kalshi_base_url: str,
) -> None:
    """
    Main collection pipeline: fetch markets → candlesticks → forecasts → save.

    Args:
        days: Number of past days to collect data for.
        output_dir: Directory to write output files.
        interval_minutes: Candlestick period in minutes.
        dry_run: If True, skip all real API calls and print what would be fetched.
        kalshi_api_key: Kalshi API key.
        kalshi_private_key_path: Path to RSA private key PEM file.
        kalshi_base_url: Kalshi REST API base URL.
    """
    cutoff_ts = int((datetime.now(UTC) - timedelta(days=days)).timestamp())
    run_ts = datetime.now(UTC).isoformat()

    logger.info("=" * 70)
    logger.info("Backtest data collection")
    cutoff_iso = datetime.fromtimestamp(cutoff_ts, tz=UTC).isoformat()
    logger.info("  Days: %d  |  Cutoff: %s  |  Dry-run: %s", days, cutoff_iso, dry_run)
    logger.info("  Output: %s  |  Interval: %d min", output_dir, interval_minutes)
    logger.info("=" * 70)

    series_list: list[tuple[str, str, DailyVariable]] = [
        (series, city, var) for series, (city, var) in WEATHER_SERIES.items()
    ]

    if dry_run:
        logger.info("[DRY RUN] Would fetch %d series:", len(series_list))
        for series, city, var in series_list:
            logger.info("  %s -> %s (%s)", series, city, var)
        logger.info("[DRY RUN] Would save output to: %s", output_dir)
        return

    all_market_records: list[MarketRecord] = []

    async with KalshiClient(
        api_key=kalshi_api_key,
        private_key_path=kalshi_private_key_path,
        base_url=kalshi_base_url,
    ) as kalshi:
        # Step 1: Fetch settled markets for each series
        for i, (series_ticker, city, daily_variable) in enumerate(series_list):
            if i > 0:
                await asyncio.sleep(3.0)  # throttle between series

            for attempt in range(_MAX_RETRIES + 1):
                try:
                    records = await fetch_markets_for_series(
                        client=kalshi,
                        series_ticker=series_ticker,
                        city=city,
                        daily_variable=daily_variable,
                        cutoff_ts=cutoff_ts,
                    )
                    all_market_records.extend(records)
                    break
                except KalshiAPIError as exc:
                    if exc.status == 429 and attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "429 on %s — retry %d/%d in %.0fs",
                            series_ticker, attempt + 1, _MAX_RETRIES, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(
                        "Failed to fetch markets for %s: %s — skipping",
                        series_ticker, exc,
                    )
                    break
                except Exception as exc:
                    logger.error(
                        "Failed to fetch markets for %s: %s — skipping",
                        series_ticker, exc,
                    )
                    break

        logger.info(
            "Collected %d market records across %d series",
            len(all_market_records),
            len(series_list),
        )

        # Step 2: Fetch candlesticks for each market
        logger.info("Fetching candlesticks for %d markets...", len(all_market_records))
        for i, record in enumerate(all_market_records):
            if i > 0:
                await asyncio.sleep(1.5)  # throttle candlestick requests
            if i > 0 and i % 10 == 0:
                logger.info("  Candlesticks: %d/%d", i, len(all_market_records))
            all_market_records[i] = await fetch_candlesticks_for_market(
                client=kalshi,
                record=record,
                interval_minutes=interval_minutes,
            )

    # Step 3: Fetch weather forecasts (no auth required — Open-Meteo is public)
    weather_forecasts: dict[str, WeatherForecastRecord] = {}
    async with WeatherClient() as weather:
        weather_forecasts = await fetch_weather_forecasts(weather, all_market_records)

    # Step 4: Build metadata and save
    unique_series: set[str] = {r.series_ticker for r in all_market_records}
    metadata = CollectionMetadata(
        run_timestamp=run_ts,
        days_covered=days,
        cutoff_ts=cutoff_ts,
        market_count=len(all_market_records),
        series_count=len(unique_series),
        weather_forecast_count=len(weather_forecasts),
        interval_minutes=interval_minutes,
        dry_run=False,
        series_tickers=sorted(unique_series),
    )

    save_outputs(
        output_dir=output_dir,
        market_records=all_market_records,
        weather_forecasts=weather_forecasts,
        metadata=metadata,
    )

    logger.info(
        "Collection complete: %d markets, %d forecasts → %s",
        len(all_market_records),
        len(weather_forecasts),
        output_dir,
    )


# ---------------------------------------------------------------------------
# Credentials loader
# ---------------------------------------------------------------------------


def load_credentials(settings_path: Path | None = None) -> tuple[str, str, str]:
    """
    Load Kalshi credentials from Settings (config.yaml + .env) or env vars.

    Falls back to ORACLE_KALSHI_API_KEY / ORACLE_KALSHI_PRIVATE_KEY_PATH
    environment variables if Settings fails to load.

    Returns:
        (api_key, private_key_path, base_url)

    Raises:
        SystemExit: If credentials are not available.
    """
    try:
        from python.settings import Settings

        settings = Settings.load(config_path=settings_path)
        api_key = settings.kalshi_api_key
        key_path = settings.kalshi_private_key_path
        base_url = settings.kalshi_base_url
    except Exception as exc:
        logger.debug("Settings.load() failed (%s) — falling back to env vars", exc)
        api_key = os.environ.get("ORACLE_KALSHI_API_KEY", "")
        key_path = os.environ.get("ORACLE_KALSHI_PRIVATE_KEY_PATH", "")
        base_url = os.environ.get(
            "ORACLE_KALSHI_BASE_URL",
            "https://api.elections.kalshi.com/trade-api/v2",
        )

    if not api_key:
        logger.error(
            "No Kalshi API key found. Set ORACLE_KALSHI_API_KEY in .env or environment."
        )
        sys.exit(1)
    if not key_path:
        logger.error(
            "No Kalshi private key path found. Set ORACLE_KALSHI_PRIVATE_KEY_PATH."
        )
        sys.exit(1)

    return api_key, key_path, base_url


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical Kalshi weather market data for backtesting."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to collect data for (default: 7).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest",
        help="Output directory (default: data/backtest/).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        choices=[1, 60, 1440],
        help="Candlestick period in minutes — 1, 60, or 1440 (default: 60).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be fetched without making real API calls.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output)

    if args.dry_run:
        api_key = "dry-run-placeholder"
        key_path = "dry-run-placeholder"
        base_url = "https://api.elections.kalshi.com/trade-api/v2"
    else:
        api_key, key_path, base_url = load_credentials()

    asyncio.run(
        run_collection(
            days=args.days,
            output_dir=output_dir,
            interval_minutes=args.interval,
            dry_run=args.dry_run,
            kalshi_api_key=api_key,
            kalshi_private_key_path=key_path,
            kalshi_base_url=base_url,
        )
    )


if __name__ == "__main__":
    main()
