"""
Daily snapshot collector for open weather markets + multi-model ensemble forecasts +
NWS settlements.

Captures the current state of Kalshi weather markets, corresponding multi-model
ensemble forecasts, NBM deterministic point forecasts, and NWS CLI settlement
observations into timestamped JSON files under data/snapshots/. After 7 days
of snapshots the data enables a clean forward-looking backtest with proper
calibration.

Usage::

    # Single daily run (backwards-compatible flat layout)
    uv run python scripts/collect_daily_snapshot.py [--output DIR] [--dry-run]

    # Multi-run per day (one per model run, nested layout)
    uv run python scripts/collect_daily_snapshot.py --run-id 0200
    uv run python scripts/collect_daily_snapshot.py --run-id 0800
    uv run python scripts/collect_daily_snapshot.py --run-id 1400
    uv run python scripts/collect_daily_snapshot.py --run-id 2000

    # Skip NWS settlement fetch (only needed once per day)
    uv run python scripts/collect_daily_snapshot.py --run-id 0200 --skip-nws

When ``--run-id`` is provided, snapshots are saved under a subdirectory::

    data/snapshots/2026-03-27/0200/
        markets.json
        weather_forecasts.json
    data/snapshots/2026-03-27/2000/
        markets.json
        weather_forecasts.json
        settlements.json          ← only on the final run

Without ``--run-id`` (default), the flat layout is preserved for backwards
compatibility::

    data/snapshots/2026-03-15/
        markets.json
        weather_forecasts.json
        settlements.json

Each run appends one line to:

    data/snapshots/index.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path so the script works standalone (e.g. from cron)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.common import DailyVariable  # noqa: E402
from python.data.kalshi_client import KalshiClient  # noqa: E402
from python.data.nws import CITY_TO_CLI_LOCATION, CliObservation, NWSClient  # noqa: E402
from python.data.weather import EnsembleForecast, NbmPointForecast, WeatherClient  # noqa: E402
from python.strategies.weather_edge import WEATHER_SERIES  # noqa: E402
from python.utils.atomic_write import atomic_write_text  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MarketSnapshot:
    """A single Kalshi weather market captured at snapshot time."""

    ticker: str
    series_ticker: str
    city: str
    daily_variable: str
    strike_type: str | None
    floor_strike: float | None
    cap_strike: float | None
    yes_bid_dollars: float | None
    yes_ask_dollars: float | None
    volume_fp: int | None
    expiration_ts: int | None
    snapshot_ts: float  # Unix timestamp of this run


_DEFAULT_MODEL = "gfs_seamless"


@dataclass
class ForecastSnapshot:
    """Ensemble forecast captured at snapshot time."""

    city: str
    forecast_date: str  # ISO date string e.g. "2026-03-15"
    daily_variable: str
    temperatures: list[float]  # N members in Fahrenheit (31 for GFS, 51 for ECMWF)
    mean: float
    std_dev: float
    fetched_at: float  # Unix timestamp from EnsembleForecast.fetched_at
    model: str = _DEFAULT_MODEL  # which NWP model this forecast came from
    model_version: str | None = None  # model cycle/version for post-hoc weighting


@dataclass
class NbmSnapshot:
    """NBM deterministic point forecast captured at snapshot time."""

    city: str
    forecast_date: str  # ISO date string e.g. "2026-03-15"
    daily_variable: str
    temperature: float  # single NBM point forecast in Fahrenheit
    fetched_at: float  # Unix timestamp from NbmPointForecast.fetched_at


@dataclass
class SettlementSnapshot:
    """NWS CLI settlement observation captured at snapshot time."""

    city: str
    cli_location: str
    observation_date: str  # ISO date string
    high_f: int | None
    low_f: int | None
    product_id: str
    issued_at: str
    is_preliminary: bool
    snapshot_ts: float


@dataclass
class CollectionStats:
    """Per-source fetch counters threaded through the collection pipeline."""

    markets_attempted: int = 0
    markets_failed: int = 0          # series whose fetch raised
    markets_truncated: int = 0       # series that hit MAX_PAGES
    forecasts_attempted: int = 0
    forecasts_failed: int = 0        # (city, date, var) keys whose fetch raised
    nbm_attempted: int = 0
    nbm_failed: int = 0
    settlements_attempted: int = 0
    settlements_failed: int = 0


@dataclass
class IndexEntry:
    """One row in index.jsonl — written at the end of each successful run."""

    run_ts: float
    date: str
    market_count: int
    forecast_count: int
    settlement_count: int
    nbm_count: int = 0  # NBM point forecasts collected (0 for older runs)
    run_id: str | None = None  # e.g. "0200", "1400" — None for legacy flat runs
    status: str = "ok"          # "ok" | "degraded"
    stats: dict | None = None   # asdict(CollectionStats), full per-source detail


# ---------------------------------------------------------------------------
# Model version lookup
# ---------------------------------------------------------------------------

# Hardcoded model version map. Update when NWP centers upgrade cycles.
# ECMWF Cycle 49r1 (Jun 2024): improved short-range 2m temperature via 4D-Var
# GFSv16 (Mar 2021): FV3 dynamical core, 127 vertical levels
_MODEL_VERSIONS: dict[str, str] = {
    "gfs_seamless": "GFSv16",
    "gfs_ensemble_025": "GFSv16",
    "gfs_ensemble_05": "GFSv16",
    "ecmwf_ifs025": "IFS-49r1",
    "ecmwf_aifs_025": "AIFS-0.3",
    "gem_global": "GEPSv8",
    "icon_seamless_eps": "ICON-2.6.6",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write *data* as indented JSON to *path* atomically and durably. A crash
    leaves the prior file or the new file, never a truncated one."""
    atomic_write_text(path, json.dumps(data, indent=2), fsync=True)


def _atomic_write_index(path: Path, rows: list[dict]) -> None:
    """Write *rows* as JSONL to *path* atomically and durably."""
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows), fsync=True)


def _parse_expiration_ts(market: dict[str, Any]) -> int | None:
    """Parse expiration timestamp from Kalshi market dict.

    Handles both ISO string (``expected_expiration_time`` / ``expiration_time``)
    and legacy integer unix timestamp (``expiration_ts``).
    """
    for field in ("expected_expiration_time", "expiration_time"):
        raw = market.get(field)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return int(raw)
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            continue
    # Legacy fallback
    raw = market.get("expiration_ts")
    if isinstance(raw, (int, float)):
        return int(raw)
    return None


def _parse_dollar_field(val: object) -> float | None:
    """Normalise a Kalshi dollar field (string or float) to a float in [0,1]."""
    if val is None:
        return None
    try:
        p = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    # Values > 1.0 are in cents
    if p > 1.0:
        p /= 100.0
    return p


def _parse_fp_int(val: object) -> int | None:
    """Parse a Kalshi '_fp' field (e.g. '636.00') to int."""
    if val is None:
        return None
    try:
        return int(float(val))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _market_to_snapshot(
    market: dict[str, Any],
    series_ticker: str,
    city: str,
    daily_variable: str,
    snapshot_ts: float,
) -> MarketSnapshot | None:
    """
    Convert a raw Kalshi market dict into a MarketSnapshot.

    Returns None if the market has no parseable expiration date (safety guard;
    expiration_ts is also captured raw for debugging).
    """
    ticker = market.get("ticker", "")
    if not ticker:
        return None

    # Parse floor/cap strikes
    floor_strike_raw = market.get("floor_strike")
    cap_strike_raw = market.get("cap_strike")
    floor_strike = float(floor_strike_raw) if floor_strike_raw is not None else None
    cap_strike = float(cap_strike_raw) if cap_strike_raw is not None else None

    return MarketSnapshot(
        ticker=ticker,
        series_ticker=series_ticker,
        city=city,
        daily_variable=daily_variable,
        strike_type=market.get("strike_type"),
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        yes_bid_dollars=_parse_dollar_field(market.get("yes_bid_dollars")),
        yes_ask_dollars=_parse_dollar_field(market.get("yes_ask_dollars")),
        volume_fp=_parse_fp_int(market.get("volume_fp")),
        expiration_ts=_parse_expiration_ts(market),
        snapshot_ts=snapshot_ts,
    )


def _forecast_to_snapshot(
    forecast: EnsembleForecast,
    daily_variable: str,
) -> ForecastSnapshot:
    """Convert an EnsembleForecast into a serialisable ForecastSnapshot."""
    return ForecastSnapshot(
        city=forecast.city,
        forecast_date=forecast.forecast_date.isoformat(),
        daily_variable=daily_variable,
        temperatures=list(forecast.temperatures),
        mean=forecast.mean_temperature,
        std_dev=forecast.std_dev,
        fetched_at=forecast.fetched_at,
        model=forecast.model,
        model_version=_MODEL_VERSIONS.get(forecast.model),
    )


def _nbm_to_snapshot(
    nbm: NbmPointForecast,
) -> NbmSnapshot:
    """Convert a NbmPointForecast into a serialisable NbmSnapshot."""
    return NbmSnapshot(
        city=nbm.city,
        forecast_date=nbm.forecast_date.isoformat(),
        daily_variable=nbm.daily_variable,
        temperature=nbm.temperature,
        fetched_at=nbm.fetched_at,
    )


# Forecast horizon for the fixed grid: {0, +1, +2} days from run_date.
# Empirically Kalshi only lists weather markets at lead 1 (+ occasionally 0/2),
# so this is a small superset of the operationally-tradeable window. Must stay
# ≤ NBM's ~11-day cap if anyone ever extends it.
_FORECAST_HORIZON_DAYS = 3


def _fixed_forecast_keys(run_date: date) -> dict[tuple[str, date, str], None]:
    """Fixed (city, forecast_date, daily_variable) grid for forecast collection.

    Decouples forecast targets from Kalshi market listings: every snapshot run
    fetches the full CITY_TO_CLI_LOCATION set (20 as of 2026-06: NYC, CHI, MIA,
    LAX, DEN, AUS, SEA, PHIL, BOS + ATL, DAL, HOU, LV, MIN, NOLA, OKC, PHX,
    SATX, SFO, DC) × 2 variables × 3 forecast dates regardless of which markets are
    open. Guarantees a continuous training corpus for calibration and EMOS even
    when Kalshi happens not to list a (city, var) on a given day (e.g. SEA
    temperature_2m_min during the ECMWF ramp-up).

    The 3 dates are run_date and the next 2 UTC days (lead times {0, 1, 2}).
    SEA temperature_2m_min is included even though it currently has no Kalshi
    series — we keep collecting so the calibration corpus stays whole if a
    series is ever listed.
    """
    cities = list(CITY_TO_CLI_LOCATION.keys())
    variables = [str(DailyVariable.TEMP_MAX), str(DailyVariable.TEMP_MIN)]
    keys: dict[tuple[str, date, str], None] = {}
    for city in cities:
        for offset in range(_FORECAST_HORIZON_DAYS):
            fd = run_date + timedelta(days=offset)
            for var in variables:
                keys[(city, fd, var)] = None
    return keys


def _cli_to_snapshot(
    obs: CliObservation,
    snapshot_ts: float,
) -> SettlementSnapshot:
    """Convert a CliObservation into a serialisable SettlementSnapshot."""
    return SettlementSnapshot(
        city=obs.city,
        cli_location=obs.cli_location,
        observation_date=obs.observation_date.isoformat(),
        high_f=obs.high_f,
        low_f=obs.low_f,
        product_id=obs.product_id,
        issued_at=obs.issued_at,
        is_preliminary=obs.is_preliminary,
        snapshot_ts=snapshot_ts,
    )


# ---------------------------------------------------------------------------
# Core async functions (also imported by tests)
# ---------------------------------------------------------------------------


async def fetch_open_weather_markets(
    kalshi_client: KalshiClient,
    snapshot_ts: float,
    stats: CollectionStats | None = None,
) -> list[MarketSnapshot]:
    """
    Fetch all open weather markets from Kalshi across every series in WEATHER_SERIES.

    Each series is queried individually (Kalshi requires series_ticker filtering).
    Markets without a parseable expiration date are silently skipped.

    Args:
        kalshi_client: Authenticated and connected KalshiClient.
        snapshot_ts: Unix timestamp to embed in every MarketSnapshot.
        stats: Optional CollectionStats to accumulate per-source fetch counters.

    Returns:
        Flat list of MarketSnapshot objects for all open weather markets.
    """
    snapshots: list[MarketSnapshot] = []

    for i, (series_ticker, (city, daily_variable)) in enumerate(WEATHER_SERIES.items()):
        if i > 0:
            # Small stagger to avoid Kalshi 429 rate limits
            await asyncio.sleep(0.5)

        logger.info("Fetching open markets for series %s (city=%s)", series_ticker, city)

        if stats is not None:
            stats.markets_attempted += 1

        try:
            raw_markets, markets_truncated = await kalshi_client.iter_all_markets(
                status="open",
                series_ticker=series_ticker,
            )
        except Exception:
            logger.exception("Failed to fetch markets for series %s — skipping", series_ticker)
            if stats is not None:
                stats.markets_failed += 1
            continue

        if markets_truncated:
            logger.critical(
                "iter_all_markets TRUNCATED for series %s — got %d open markets "
                "before MAX_PAGES cap; daily snapshot is INCOMPLETE for this series",
                series_ticker, len(raw_markets),
            )
            if stats is not None:
                stats.markets_truncated += 1

        logger.info("  → %d open markets returned", len(raw_markets))

        for market in raw_markets:
            snap = _market_to_snapshot(
                market,
                series_ticker=series_ticker,
                city=city,
                daily_variable=str(daily_variable),
                snapshot_ts=snapshot_ts,
            )
            if snap is not None:
                snapshots.append(snap)

    logger.info("Total open weather market snapshots collected: %d", len(snapshots))
    return snapshots


async def fetch_weather_forecasts(
    weather_client: WeatherClient,
    run_date: date,
    models: list[str] | None = None,
    stats: CollectionStats | None = None,
) -> list[ForecastSnapshot]:
    """
    Fetch ensemble forecasts for the fixed (city, forecast_date, daily_variable)
    grid anchored at ``run_date``.

    Forecast targets are intentionally decoupled from Kalshi market listings —
    see ``_fixed_forecast_keys``. This guarantees the calibration/EMOS corpus
    stays whole even when Kalshi delists a (city, var) combination.

    When multiple models are requested, each model produces a separate
    ``ForecastSnapshot`` per (city, date, variable) combination, so that
    calibration and blend-weight analysis can be done offline without re-fetching.

    ``get_multi_model_forecast`` handles per-model failures gracefully: failed
    models are omitted from the returned dict rather than raising, so each key
    in the outer loop is only skipped if *all* models fail unexpectedly.

    Args:
        weather_client: Connected WeatherClient.
        run_date: UTC date of the snapshot run; anchors the forecast horizon.
        models: List of Open-Meteo model names to fetch. Defaults to
            ``["gfs_seamless"]`` (GFS-only, preserving legacy behaviour).
        stats: Optional CollectionStats to accumulate per-source fetch counters.

    Returns:
        List of ForecastSnapshot, one per model per unique (city, date, variable)
        combination.
    """
    if models is None:
        models = [_DEFAULT_MODEL]

    unique_keys = _fixed_forecast_keys(run_date)

    logger.info(
        "Fetching forecasts for %d unique (city, date, variable) combinations × %d model(s): %s",
        len(unique_keys),
        len(models),
        models,
    )

    forecasts: list[ForecastSnapshot] = []

    for city, forecast_date, daily_variable in unique_keys:
        logger.info("  → %s / %s / %s", city, forecast_date, daily_variable)
        if stats is not None:
            stats.forecasts_attempted += 1
        try:
            dv = DailyVariable(daily_variable)
            model_forecasts = await weather_client.get_multi_model_forecast(
                city=city,
                forecast_date=forecast_date,
                daily_variable=dv,
                models=models,
            )
            for ensemble in model_forecasts.values():
                forecasts.append(_forecast_to_snapshot(ensemble, daily_variable))
        except Exception:
            logger.exception(
                "Failed to fetch ensemble forecasts for %s on %s (%s) — skipping",
                city,
                forecast_date,
                daily_variable,
            )
            if stats is not None:
                stats.forecasts_failed += 1

    logger.info("Fetched %d forecast snapshots", len(forecasts))
    return forecasts


async def fetch_nbm_forecasts(
    weather_client: WeatherClient,
    run_date: date,
    stats: CollectionStats | None = None,
) -> list[NbmSnapshot]:
    """
    Fetch NBM deterministic point forecasts for the fixed forecast grid.

    Uses the same fixed grid as fetch_weather_forecasts so NBM forecasts can
    be compared directly against ensemble means and settlements. NBM is no
    longer used in production trading (per memory note 2026-04-23) but
    collection continues for research / EMOS comparison; the grid is small
    enough that the API cost is trivial. NBM's effective horizon is ~11 days,
    which comfortably contains the {0, +1, +2} grid.

    Args:
        weather_client: Connected WeatherClient.
        run_date: UTC date of the snapshot run; anchors the forecast horizon.
        stats: Optional CollectionStats to accumulate per-source fetch counters.

    Returns:
        List of NbmSnapshot, one per unique (city, date, variable) combination.
    """
    unique_keys = _fixed_forecast_keys(run_date)

    logger.info(
        "Fetching NBM point forecasts for %d unique (city, date, variable) combinations",
        len(unique_keys),
    )

    nbm_snapshots: list[NbmSnapshot] = []

    for city, forecast_date, daily_variable in unique_keys:
        logger.info("  → NBM %s / %s / %s", city, forecast_date, daily_variable)
        if stats is not None:
            stats.nbm_attempted += 1
        try:
            dv = DailyVariable(daily_variable)
            nbm = await weather_client.get_nbm_point_forecast(
                city=city,
                forecast_date=forecast_date,
                daily_variable=dv,
            )
            nbm_snapshots.append(_nbm_to_snapshot(nbm))
        except Exception:
            logger.exception(
                "Failed to fetch NBM for %s on %s (%s) — skipping",
                city,
                forecast_date,
                daily_variable,
            )
            if stats is not None:
                stats.nbm_failed += 1

    logger.info("Fetched %d NBM point forecasts", len(nbm_snapshots))
    return nbm_snapshots


async def fetch_nws_settlements(
    nws_client: NWSClient,
    snapshot_ts: float,
    max_days: int = 3,
    stats: CollectionStats | None = None,
) -> list[SettlementSnapshot]:
    """
    Fetch NWS CLI settlement observations for all cities.

    Retrieves the most recent CLI products (up to max_days dates) for each
    city in CITY_TO_CLI_LOCATION. These are the official high/low temperatures
    that Kalshi uses to settle weather markets.

    Args:
        nws_client: Connected NWSClient.
        snapshot_ts: Unix timestamp to embed in each SettlementSnapshot.
        max_days: Max distinct dates to fetch per city (default 3).
        stats: Optional CollectionStats to accumulate per-source fetch counters.

    Returns:
        List of SettlementSnapshot across all cities.
    """
    settlements: list[SettlementSnapshot] = []

    for i, city in enumerate(CITY_TO_CLI_LOCATION):
        if i > 0:
            await asyncio.sleep(0.3)  # Be polite to NWS API

        logger.info("Fetching NWS CLI observations for %s", city)
        if stats is not None:
            stats.settlements_attempted += 1
        try:
            observations = await nws_client.get_recent_observations(
                city=city,
                max_days=max_days,
            )
            for obs in observations:
                settlements.append(_cli_to_snapshot(obs, snapshot_ts))
        except Exception:
            logger.exception("Failed to fetch NWS CLI for %s — skipping", city)
            if stats is not None:
                stats.settlements_failed += 1

    logger.info("Fetched %d NWS settlement observations", len(settlements))
    return settlements


def save_snapshot(
    output_dir: Path,
    run_date: date,
    markets: list[MarketSnapshot],
    forecasts: list[ForecastSnapshot],
    settlements: list[SettlementSnapshot],
    run_ts: float,
    run_id: str | None = None,
    nbm_forecasts: list[NbmSnapshot] | None = None,
    stats: CollectionStats | None = None,
) -> None:
    """
    Persist snapshot data to disk.

    Directory layout (without ``run_id`` — backwards-compatible flat layout)::

        <output_dir>/
            <YYYY-MM-DD>/
                markets.json
                weather_forecasts.json
                nbm_forecasts.json
                settlements.json
            index.jsonl

    Directory layout (with ``run_id`` — multi-run per day)::

        <output_dir>/
            <YYYY-MM-DD>/
                <run_id>/
                    markets.json
                    weather_forecasts.json
                    nbm_forecasts.json     ← NBM point forecasts
                    settlements.json       ← only when settlements are non-empty
            index.jsonl

    The date (or date/run_id) directory is created (or silently overwritten for
    idempotency).

    Args:
        output_dir: Root output directory (e.g. Path("data/snapshots")).
        run_date: The UTC date of this snapshot run.
        markets: Market snapshots to save.
        forecasts: Forecast snapshots to save.
        settlements: NWS CLI settlement observations to save.
        run_ts: Unix timestamp of the run start (for index.jsonl).
        run_id: Optional run identifier (e.g. "0200", "1400"). When provided,
            files are saved under ``<date>/<run_id>/`` instead of ``<date>/``.
        nbm_forecasts: Optional NBM point forecast snapshots to save.
    """
    if nbm_forecasts is None:
        nbm_forecasts = []

    date_str = run_date.isoformat()
    if run_id:
        data_dir = output_dir / date_str / run_id
    else:
        data_dir = output_dir / date_str
    data_dir.mkdir(parents=True, exist_ok=True)

    markets_path = data_dir / "markets.json"
    forecasts_path = data_dir / "weather_forecasts.json"
    index_path = output_dir / "index.jsonl"

    # Serialise market snapshots
    markets_data = [asdict(m) for m in markets]
    _atomic_write_json(markets_path, markets_data)
    logger.info("Saved %d markets → %s", len(markets), markets_path)

    # Serialise forecast snapshots
    forecasts_data = [asdict(fc) for fc in forecasts]
    _atomic_write_json(forecasts_path, forecasts_data)
    logger.info("Saved %d forecasts → %s", len(forecasts), forecasts_path)

    # Serialise NBM point forecasts (only when present)
    if nbm_forecasts:
        nbm_path = data_dir / "nbm_forecasts.json"
        nbm_data = [asdict(n) for n in nbm_forecasts]
        _atomic_write_json(nbm_path, nbm_data)
        logger.info("Saved %d NBM forecasts → %s", len(nbm_forecasts), nbm_path)

    # Serialise settlement observations (only when present)
    if settlements:
        settlements_path = data_dir / "settlements.json"
        settlements_data = [asdict(s) for s in settlements]
        _atomic_write_json(settlements_path, settlements_data)
        logger.info("Saved %d settlements → %s", len(settlements), settlements_path)

    # Compute run integrity status (corpus-critical sources only)
    degraded = (
        len(markets) == 0
        or len(forecasts) == 0
        or (stats is not None and (stats.markets_truncated > 0 or stats.forecasts_failed > 0))
    )
    status = "degraded" if degraded else "ok"
    if degraded:
        logger.warning(
            "snapshot run DEGRADED: %s", asdict(stats) if stats else "(no CollectionStats)"
        )

    # Append index entry
    index_entry = IndexEntry(
        run_ts=run_ts,
        date=date_str,
        market_count=len(markets),
        forecast_count=len(forecasts),
        settlement_count=len(settlements),
        nbm_count=len(nbm_forecasts),
        run_id=run_id,
        status=status,
        stats=asdict(stats) if stats is not None else None,
    )
    key = (index_entry.date, index_entry.run_id)
    rows: list[dict] = []
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Dropping corrupt index.jsonl line during upsert: %r", line[:120])
                continue  # tolerate a partially-written legacy line
            if (r.get("date"), r.get("run_id")) == key:
                continue  # drop the prior row for this slot (upsert)
            rows.append(r)
    rows.append(asdict(index_entry))
    _atomic_write_index(index_path, rows)
    logger.info("Wrote index entry (upsert) → %s", index_path)


# ---------------------------------------------------------------------------
# Credentials helper
# ---------------------------------------------------------------------------


def _get_credentials() -> tuple[str, str]:
    """
    Load Kalshi credentials from environment variables.

    Checks ORACLE_KALSHI_API_KEY / ORACLE_KALSHI_PRIVATE_KEY_PATH first
    (Settings convention), then falls back to Settings.load() which reads
    from config.yaml and .env.

    Returns:
        (api_key, private_key_path) tuple.

    Raises:
        RuntimeError: If credentials are not available.
    """
    # Try direct env vars first (works without pydantic overhead, useful in cron)
    api_key = os.environ.get("ORACLE_KALSHI_API_KEY", "")
    key_path = os.environ.get("ORACLE_KALSHI_PRIVATE_KEY_PATH", "")

    if not api_key or not key_path:
        # Fall back to Settings (reads config.yaml + .env)
        try:
            from python.settings import Settings

            settings = Settings.load()
            api_key = api_key or settings.kalshi_api_key
            key_path = key_path or settings.kalshi_private_key_path
        except Exception:
            logger.debug("Settings.load() failed — relying on env vars only")

    if not api_key:
        raise RuntimeError(
            "No Kalshi API key found. Set ORACLE_KALSHI_API_KEY or add kalshi_api_key to .env"
        )
    if not key_path:
        raise RuntimeError(
            "No Kalshi private key path found. "
            "Set ORACLE_KALSHI_PRIVATE_KEY_PATH or add kalshi_private_key_path to .env"
        )
    return api_key, key_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main(
    output_dir: Path,
    dry_run: bool,
    run_id: str | None = None,
    skip_nws: bool = False,
) -> None:
    """
    Orchestrate the snapshot collection.

    1. Fetch all open weather markets across every WEATHER_SERIES ticker.
    2. Fetch multi-model ensemble forecasts for each unique (city, date, variable).
    3. Fetch NBM deterministic point forecasts for the same keys.
    4. Optionally fetch NWS CLI settlement observations for all cities.
    5. Save output and append to index.jsonl.

    When ``run_id`` is provided, files are saved under
    ``<output_dir>/<date>/<run_id>/`` instead of ``<output_dir>/<date>/``.

    When ``skip_nws`` is True, NWS settlement collection is skipped (useful
    when running multiple times per day — settlements only need to be collected
    once).

    In dry-run mode, prints what would be collected without making any API calls.
    """
    run_ts = time.time()
    run_date = datetime.now(UTC).date()

    if run_id:
        snapshot_label = f"{run_date.isoformat()}/{run_id}"
    else:
        snapshot_label = run_date.isoformat()

    if dry_run:
        logger.info("[DRY-RUN] Would collect snapshots for %d series:", len(WEATHER_SERIES))
        for series_ticker, (city, daily_variable) in WEATHER_SERIES.items():
            logger.info("  %s → city=%s variable=%s", series_ticker, city, daily_variable)
        forecast_grid = _fixed_forecast_keys(run_date)
        logger.info(
            "[DRY-RUN] Would fetch forecasts for %d (city, date, variable) keys "
            "(fixed grid, decoupled from market listings):",
            len(forecast_grid),
        )
        for city, fd, var in forecast_grid:
            logger.info("  %s / %s / %s", city, fd.isoformat(), var)
        if not skip_nws:
            logger.info(
                "[DRY-RUN] Would fetch NWS CLI for %d cities: %s",
                len(CITY_TO_CLI_LOCATION),
                list(CITY_TO_CLI_LOCATION.keys()),
            )
        else:
            logger.info("[DRY-RUN] NWS settlement fetch skipped (--skip-nws)")
        logger.info("[DRY-RUN] Output directory: %s", output_dir / snapshot_label)
        if run_id:
            logger.info("[DRY-RUN] Run ID: %s", run_id)
        logger.info("[DRY-RUN] No API calls made.")
        return

    # Resolve credentials
    api_key, key_path = _get_credentials()

    # Load ensemble_models from settings (falls back to GFS-only if settings unavailable)
    forecast_concurrency = 1
    try:
        from python.settings import Settings

        _settings = Settings.load()
        models = list(_settings.strategies.weather.ensemble_models)
        forecast_concurrency = int(_settings.strategies.weather.forecast_concurrency)
    except Exception:
        logger.warning("Could not load settings — falling back to GFS-only, concurrency=1")
        models = ["gfs_seamless"]

    # Run all fetches with shared clients.
    # NWS settlements are independent of Kalshi/weather, so run concurrently.
    settlements: list[SettlementSnapshot] = []
    nbm_forecasts: list[NbmSnapshot] = []
    collection_stats = CollectionStats()

    async with (
        KalshiClient(api_key=api_key, private_key_path=key_path) as kalshi_client,
        WeatherClient(forecast_concurrency=forecast_concurrency) as weather_client,
    ):
        # Start NWS fetch concurrently if enabled
        settlements_task = None
        if not skip_nws:
            nws_client = NWSClient()
            await nws_client.connect()
            settlements_task = asyncio.create_task(
                fetch_nws_settlements(nws_client, snapshot_ts=run_ts, stats=collection_stats)
            )

        # Forecasts use a fixed (city, var, date) grid anchored at run_date —
        # independent of which Kalshi markets happen to be listed today.
        markets = await fetch_open_weather_markets(
            kalshi_client, snapshot_ts=run_ts, stats=collection_stats
        )

        # Fetch ensemble and NBM point forecasts concurrently (independent).
        forecast_task = asyncio.create_task(
            fetch_weather_forecasts(weather_client, run_date, models=models, stats=collection_stats)
        )
        nbm_task = asyncio.create_task(
            fetch_nbm_forecasts(weather_client, run_date, stats=collection_stats)
        )
        forecasts = await forecast_task
        nbm_forecasts = await nbm_task

        if settlements_task is not None:
            settlements = await settlements_task
            await nws_client.close()  # type: ignore[possibly-undefined]

    # Persist
    save_snapshot(
        output_dir=output_dir,
        run_date=run_date,
        markets=markets,
        forecasts=forecasts,
        settlements=settlements,
        run_ts=run_ts,
        run_id=run_id,
        nbm_forecasts=nbm_forecasts,
        stats=collection_stats,
    )

    logger.info(
        "Snapshot complete: %d markets, %d forecasts, %d NBM, %d settlements → %s/%s",
        len(markets),
        len(forecasts),
        len(nbm_forecasts),
        len(settlements),
        output_dir,
        snapshot_label,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a snapshot of open weather markets + ensemble forecasts.",
    )
    parser.add_argument(
        "--output",
        default="data/snapshots",
        help="Output directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be collected without making API calls",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier for multi-run-per-day collection (e.g. '0200', '1400'). "
            "When set, snapshot files are saved under <date>/<run-id>/ instead of "
            "<date>/. Omit for backwards-compatible flat layout."
        ),
    )
    parser.add_argument(
        "--skip-nws",
        action="store_true",
        help=(
            "Skip NWS settlement collection. Useful when running multiple times "
            "per day — settlements only need to be collected once."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()
    asyncio.run(
        main(
            output_dir=Path(args.output),
            dry_run=args.dry_run,
            run_id=args.run_id,
            skip_nws=args.skip_nws,
        )
    )
