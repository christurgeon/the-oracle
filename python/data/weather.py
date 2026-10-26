"""
Open-Meteo ensemble forecast fetcher for weather strategy.

Fetches ensemble forecasts (GFS and/or ECMWF) from the free Open-Meteo API
and computes the fraction of members above/below a given temperature threshold.
That fraction IS the model probability used by the weather edge strategy.

Supports multi-model blending: fetch GFS (30 members) and ECMWF (50 members)
in parallel, then compute a weighted average of per-model bracket probabilities.

No API key required. Requests Fahrenheit directly from the API to avoid
manual unit conversion.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import httpx

from python.common import NWS_ROUND_HALF_F, DailyVariable, Direction
from python.utils.retry import get_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# City coordinate mapping
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    # Coordinates target NWS settlement stations (NCDC GHCND values).
    # Each city settles on the NWS Climatological Report (Daily) for that station.
    "NYC": (40.77898, -73.96925),  # Central Park, New York (GHCND:USW00094728)
    "CHI": (41.78412, -87.75514),  # Chicago Midway Airport (GHCND:USW00014819)
    "MIA": (25.78805, -80.31694),  # Miami International Airport (GHCND:USW00012839)
    "LAX": (33.93816, -118.38660),  # Los Angeles International Airport (GHCND:USW00023174)
    "DEN": (39.84657, -104.65623),  # Denver International Airport (GHCND:USW00003017)
    "AUS": (30.18311, -97.67989),  # Austin-Bergstrom Airport (WBAN:13904)
    "SEA": (47.44467, -122.31442),  # Seattle-Tacoma Airport (WBAN:24233)
    "PHIL": (39.87290, -75.24100),  # Philadelphia International Airport (GHCND:USW00013739)
    "BOS": (42.36010, -71.00970),  # Boston Logan Airport (GHCND:USW00014739)
    # New cities (collection-only) — coords target the NWS settlement station.
    "ATL": (33.6407, -84.4277),  # Hartsfield-Jackson Atlanta (GHCND:USW00013874)
    "DAL": (32.8998, -97.0403),  # Dallas-Fort Worth Intl (GHCND:USW00003927)
    "HOU": (29.6454, -95.2789),  # Houston William P. Hobby (GHCND:USW00012918)
    "LV": (36.0840, -115.1537),  # Las Vegas Harry Reid Intl (GHCND:USW00023169)
    "MIN": (44.8831, -93.2289),  # Minneapolis-St Paul Intl (GHCND:USW00014922)
    "NOLA": (29.9934, -90.2580),  # New Orleans Louis Armstrong Intl (GHCND:USW00012916)
    "OKC": (35.3889, -97.6008),  # Oklahoma City Will Rogers World (GHCND:USW00013967)
    "PHX": (33.4278, -112.0037),  # Phoenix Sky Harbor Intl (GHCND:USW00023183)
    "SATX": (29.5337, -98.4698),  # San Antonio Intl (GHCND:USW00012921)
    "SFO": (37.6197, -122.3647),  # San Francisco Intl (GHCND:USW00023234)
    "DC": (38.8483, -77.0341),  # Washington Reagan National (GHCND:USW00013743)
}

ENSEMBLE_API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Retry hardening for Open-Meteo's free tier (which rate-limits with HTTP 429).
# Jitter de-synchronizes retry storms across the concurrent fan-out so retries
# don't re-burst in lockstep; the elapsed-time budget bounds total retry wall
# clock per request so a string of ``Retry-After`` headers can never let one
# fetch run away and starve the systemd watchdog. Both are belt-and-braces with
# the per-fetch ``asyncio.wait_for`` in weather_edge._prefetch_* and the
# client-level request semaphore.
_OPEN_METEO_RETRY_JITTER = 0.25
_OPEN_METEO_MAX_RETRY_ELAPSED_SECONDS = 12.0

# ECMWF forecasts extend 16 days; GFS extends 35 days
ECMWF_MAX_FORECAST_DAYS = 16

# Default model weights for blending (equal weight when both available)
DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "gfs_seamless": 0.5,
    "ecmwf_ifs025": 0.5,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleForecast:
    """Ensemble forecast for a single city, date, and model.

    INVARIANT: ``temperatures`` is treated as immutable. No code path may
    ``append``/``__setitem__``/alias-then-mutate this list. Transformations
    (``apply_bias_correction``, ``inflated``, etc.) return new instances via
    ``dataclasses.replace`` with a freshly-constructed list — preserve that
    pattern. The EMOS shadow path (python/data/emos.py) reconstructs raw
    pre-correction members from a stored corrected forecast and relies on
    each list being unique to its instance.
    """

    city: str
    forecast_date: date
    temperatures: list[float]  # one value per ensemble member (Fahrenheit)
    model: str = "gfs_seamless"  # "gfs_seamless" or "ecmwf_ifs025"
    fetched_at: float = field(default_factory=time.time)

    @property
    def member_count(self) -> int:
        return len(self.temperatures)

    def fraction_above(self, threshold: float) -> float:
        """Fraction of ensemble members with temperature strictly above threshold."""
        if not self.temperatures:
            return 0.0
        count = sum(1 for t in self.temperatures if t > threshold)
        return count / len(self.temperatures)

    def fraction_at_or_above(self, threshold: float) -> float:
        """Fraction of ensemble members with temperature >= threshold."""
        if not self.temperatures:
            return 0.0
        count = sum(1 for t in self.temperatures if t >= threshold)
        return count / len(self.temperatures)

    def fraction_below(self, threshold: float) -> float:
        """Fraction of ensemble members with temperature strictly below threshold."""
        return 1.0 - self.fraction_at_or_above(threshold)

    def fraction_between(self, floor: float, cap: float) -> float:
        """Fraction of ensemble members with temperature in [floor, cap] (inclusive)."""
        if not self.temperatures:
            return 0.0
        count = sum(1 for t in self.temperatures if floor <= t <= cap)
        return count / len(self.temperatures)

    @property
    def mean_temperature(self) -> float:
        if not self.temperatures:
            return 0.0
        return sum(self.temperatures) / len(self.temperatures)

    @property
    def min_temperature(self) -> float:
        return min(self.temperatures) if self.temperatures else 0.0

    @property
    def max_temperature(self) -> float:
        return max(self.temperatures) if self.temperatures else 0.0

    @property
    def std_dev(self) -> float:
        """Population standard deviation of ensemble member temperatures."""
        if len(self.temperatures) < 2:
            return 0.0
        mean = self.mean_temperature
        variance = sum((t - mean) ** 2 for t in self.temperatures) / len(self.temperatures)
        return float(variance**0.5)

    def inflated(self, factor: float) -> EnsembleForecast:
        """Return a new forecast with ensemble spread inflated by *factor*.

        Each member temperature is scaled away from the ensemble mean:
        ``inflated_temp = mean + factor * (temp - mean)``

        A factor of 1.0 returns an identical forecast.  A factor of 2.0
        doubles the spread (each member moves twice as far from the mean).
        """
        if factor <= 0.0:
            raise ValueError(f"Spread inflation factor must be positive, got {factor}")
        if factor == 1.0 or not self.temperatures:
            return self
        mean = self.mean_temperature
        inflated_temps = [mean + factor * (t - mean) for t in self.temperatures]
        return replace(self, temperatures=inflated_temps)


@dataclass(frozen=True)
class NbmPointForecast:
    """NBM deterministic point forecast for a single city, date, and variable.

    NBM (National Blend of Models) is NOAA's calibrated consensus of GFS, NAM,
    HRRR, ECMWF, and MOS.  It outputs a single best-estimate temperature, not
    an ensemble.  Useful for validating ensemble mean accuracy and (eventually)
    re-centering ensembles toward the NBM point forecast.
    """

    city: str
    forecast_date: date
    daily_variable: DailyVariable
    temperature: float  # Fahrenheit
    fetched_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Spread inflation config
# ---------------------------------------------------------------------------

# Per-city, per-variable inflation factors derived from debiased spread-skill
# ratios. Correct for ensemble underdispersion (GFS/ECMWF ensembles cluster
# too tightly around the mean). Factors are 1/debiased_spread_skill_ratio;
# debiased RMSE strips mean bias so inflation doesn't double-count it.
GFS_DERIVED_SPREAD_INFLATION: dict[str, dict[str, float]] = {
    "AUS": {DailyVariable.TEMP_MAX: 1.0, DailyVariable.TEMP_MIN: 3.0},
    "CHI": {DailyVariable.TEMP_MAX: 2.0, DailyVariable.TEMP_MIN: 2.5},
    "DEN": {DailyVariable.TEMP_MAX: 3.0, DailyVariable.TEMP_MIN: 2.0},
    "LAX": {DailyVariable.TEMP_MAX: 3.5, DailyVariable.TEMP_MIN: 2.0},
    "MIA": {DailyVariable.TEMP_MAX: 3.0, DailyVariable.TEMP_MIN: 2.5},
    "NYC": {DailyVariable.TEMP_MAX: 1.0, DailyVariable.TEMP_MIN: 2.5},
    "SEA": {DailyVariable.TEMP_MAX: 1.5, DailyVariable.TEMP_MIN: 2.0},
}

GFS_DERIVED_SPREAD_INFLATION_DEFAULT = 2.0


def get_spread_inflation_factor(
    city: str,
    daily_variable: str,
    overrides: dict[str, dict[str, float]] | None = None,
) -> float:
    """Look up the spread inflation factor for a city and variable.

    Checks *overrides* first (from config.yaml), then falls back to the
    calibration-derived defaults, then to the global default (2.0).
    """
    city = city.upper()
    if overrides:
        city_overrides = overrides.get(city, {})
        if daily_variable in city_overrides:
            return city_overrides[daily_variable]
    defaults = GFS_DERIVED_SPREAD_INFLATION.get(city, {})
    return defaults.get(daily_variable, GFS_DERIVED_SPREAD_INFLATION_DEFAULT)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    forecast: EnsembleForecast
    expires_at: float


class _ForecastCache:
    """Simple in-memory cache with TTL (seconds)."""

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _CacheEntry] = {}

    def _key(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
        model: str = "gfs_seamless",
    ) -> str:
        return f"{city}:{forecast_date.isoformat()}:{daily_variable}:{model}"

    def get(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
        model: str = "gfs_seamless",
    ) -> EnsembleForecast | None:
        key = self._key(city, forecast_date, daily_variable, model)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            return None
        return entry.forecast

    def put(
        self,
        forecast: EnsembleForecast,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ) -> None:
        key = self._key(
            forecast.city,
            forecast.forecast_date,
            daily_variable,
            forecast.model,
        )
        self._store[key] = _CacheEntry(
            forecast=forecast,
            expires_at=time.time() + self._ttl,
        )

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# NWS-rounding settlement window
# ---------------------------------------------------------------------------


def _apply_nws_window(
    forecast: EnsembleForecast,
    direction: Direction,
    floor_strike: float | None,
    cap_strike: float | None,
) -> float:
    """Ensemble probability over the NWS integer-rounding settle window."""
    if direction == Direction.ABOVE:
        if floor_strike is None:
            raise ValueError("floor_strike required for ABOVE direction")
        return forecast.fraction_at_or_above(floor_strike + NWS_ROUND_HALF_F)
    if direction == Direction.BELOW:
        if cap_strike is None:
            raise ValueError("cap_strike required for BELOW direction")
        return forecast.fraction_below(cap_strike - NWS_ROUND_HALF_F)
    if direction == Direction.BETWEEN:
        if floor_strike is None or cap_strike is None:
            raise ValueError("floor_strike and cap_strike required for BETWEEN direction")
        return forecast.fraction_between(
            floor_strike - NWS_ROUND_HALF_F,
            cap_strike + NWS_ROUND_HALF_F,
        )
    raise ValueError(f"Unsupported direction: '{direction}'")


# ---------------------------------------------------------------------------
# Open-Meteo GFS ensemble client
# ---------------------------------------------------------------------------


class WeatherClient:
    """
    Async client for fetching GFS ensemble forecasts from Open-Meteo.

    Usage::

        async with WeatherClient() as client:
            forecast = await client.get_ensemble_forecast("NYC", date(2026, 3, 15))
            prob = forecast.fraction_above(70.0)
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 1800.0,
        timeout_seconds: float = 30.0,
        forecast_concurrency: int = 2,
    ) -> None:
        self._cache = _ForecastCache(ttl_seconds=cache_ttl_seconds)
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        # CLIENT-LEVEL cap on concurrent outbound Open-Meteo requests. This is
        # the AUTHORITATIVE limiter — it wraps every outbound HTTP call (ensemble
        # inner-gather + multi-model fan-out + NBM point forecasts) regardless of
        # which call path issued it, and supersedes the per-scan ``Semaphore(4)``
        # sites in weather_edge.py (``_prefetch_forecasts``/``_prefetch_nbm``).
        # Sized from ``weather.forecast_concurrency`` (default 2). Open-Meteo's
        # free tier rejects bursts with HTTP 429; without this cap a 7-city scan
        # x 2 models could fire ~8 concurrent requests and trip the rate limit.
        # Single-knob rollback: set ``forecast_concurrency`` high to effectively
        # disable the cap. ``asyncio.Semaphore`` is loop-agnostic at construction
        # in 3.10+, so building it here (no running loop) is safe.
        self._forecast_concurrency = max(1, int(forecast_concurrency))
        self._request_semaphore = asyncio.Semaphore(self._forecast_concurrency)

    @property
    def forecast_concurrency(self) -> int:
        """Max concurrent outbound Open-Meteo requests (client-level cap)."""
        return self._forecast_concurrency

    async def __aenter__(self) -> WeatherClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def city_coords(city: str) -> tuple[float, float]:
        """
        Look up latitude/longitude for a city code.

        Raises KeyError if the city code is not recognized.
        """
        code = city.upper()
        if code not in CITY_COORDS:
            raise KeyError(
                f"Unknown city code '{city}'. Supported: {', '.join(sorted(CITY_COORDS))}"
            )
        return CITY_COORDS[code]

    async def get_ensemble_forecast(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
        model: str = "gfs_seamless",
    ) -> EnsembleForecast:
        """
        Fetch an ensemble forecast for a daily temperature variable.

        Args:
            city: City code (NYC, CHI, MIA, LAX, DEN, AUS, SEA).
            forecast_date: The date for which to retrieve the forecast.
            daily_variable: Open-Meteo daily variable name. Use
                ``"temperature_2m_max"`` for high-temp markets (KXHIGH) or
                ``"temperature_2m_min"`` for low-temp markets (KXLOWT).
            model: Open-Meteo ensemble model name. ``"gfs_seamless"`` (30
                members) or ``"ecmwf_ifs025"`` (50 members).

        Returns:
            EnsembleForecast with one temperature per ensemble member.

        Raises:
            KeyError: If the city code is unknown.
            WeatherAPIError: If the Open-Meteo API request fails.
        """
        city = city.upper()

        cached = self._cache.get(city, forecast_date, daily_variable, model)
        if cached is not None:
            logger.debug(
                "Cache hit for %s on %s (%s, %s)",
                city,
                forecast_date,
                daily_variable,
                model,
            )
            return cached

        lat, lon = self.city_coords(city)

        date_str = forecast_date.isoformat()

        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lon,
            "daily": daily_variable,
            "temperature_unit": "fahrenheit",
            "start_date": date_str,
            "end_date": date_str,
            "models": model,
        }

        if self._client is None or self._client.is_closed:
            raise RuntimeError("Client not connected — use `async with` or call connect()")

        model_label = model.upper().replace("_", " ")
        logger.info(
            "Fetching %s ensemble for %s (%s) on %s [%s]",
            model_label,
            city,
            f"{lat},{lon}",
            forecast_date,
            daily_variable,
        )

        client = self._client

        async with self._request_semaphore:
            data = await get_json(
                client,
                ENSEMBLE_API_URL,
                params=params,
                error_label="Open-Meteo",
                retryable_statuses={429, 500, 502, 503},
                exc_type=WeatherAPIError,
                jitter=_OPEN_METEO_RETRY_JITTER,
                max_elapsed_seconds=_OPEN_METEO_MAX_RETRY_ELAPSED_SECONDS,
            )
        temperatures = self._parse_ensemble_temperatures(
            data,
            forecast_date,
            daily_variable,
        )

        forecast = EnsembleForecast(
            city=city,
            forecast_date=forecast_date,
            temperatures=temperatures,
            model=model,
        )

        self._cache.put(forecast, daily_variable)

        logger.info(
            "%s ensemble for %s on %s: mean=%.1f°F, min=%.1f°F, max=%.1f°F, members=%d",
            model_label,
            city,
            forecast_date,
            forecast.mean_temperature,
            forecast.min_temperature,
            forecast.max_temperature,
            forecast.member_count,
        )

        return forecast

    async def get_multi_model_forecast(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
        models: list[str] | None = None,
    ) -> dict[str, EnsembleForecast]:
        """Fetch ensemble forecasts from multiple models in parallel.

        Models that fail (e.g., ECMWF beyond its 16-day horizon) are omitted
        from the result rather than raising.

        Args:
            city: City code.
            forecast_date: Target date.
            daily_variable: Temperature variable.
            models: List of Open-Meteo model names. Defaults to ``["gfs_seamless"]``.

        Returns:
            Dict mapping model name to its ``EnsembleForecast``.
            Empty dict if all models fail.
        """
        if models is None:
            models = ["gfs_seamless"]

        async def _fetch_one(model_name: str) -> tuple[str, EnsembleForecast | None]:
            try:
                fc = await self.get_ensemble_forecast(
                    city,
                    forecast_date,
                    daily_variable,
                    model=model_name,
                )
                return model_name, fc
            except Exception:
                logger.warning(
                    "Failed to fetch %s for %s on %s (%s)",
                    model_name,
                    city,
                    forecast_date,
                    daily_variable,
                    exc_info=True,
                )
                return model_name, None

        results = await asyncio.gather(*[_fetch_one(m) for m in models])
        return {name: fc for name, fc in results if fc is not None}

    async def get_nbm_point_forecast(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ) -> NbmPointForecast:
        """Fetch the NBM deterministic point forecast for a city/date/variable.

        Uses the Open-Meteo standard forecast API (not the ensemble API) with
        ``models=ncep_nbm_conus``.  NBM extends ~11 days and updates hourly.

        Args:
            city: City code (NYC, CHI, MIA, LAX, DEN, AUS, SEA).
            forecast_date: The date for which to retrieve the forecast.
            daily_variable: ``"temperature_2m_max"`` or ``"temperature_2m_min"``.

        Returns:
            NbmPointForecast with a single temperature value in Fahrenheit.

        Raises:
            KeyError: If the city code is unknown.
            WeatherAPIError: If the API request fails or returns no data.
        """
        city = city.upper()
        lat, lon = self.city_coords(city)
        date_str = forecast_date.isoformat()

        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lon,
            "daily": daily_variable,
            "temperature_unit": "fahrenheit",
            "start_date": date_str,
            "end_date": date_str,
            "models": "ncep_nbm_conus",
        }

        if self._client is None or self._client.is_closed:
            raise RuntimeError("Client not connected — use `async with` or call connect()")

        logger.info(
            "Fetching NBM point forecast for %s (%s,%s) on %s [%s]",
            city,
            lat,
            lon,
            forecast_date,
            daily_variable,
        )

        client = self._client

        async with self._request_semaphore:
            data = await get_json(
                client,
                FORECAST_API_URL,
                params=params,
                error_label="Open-Meteo NBM",
                retryable_statuses={429, 500, 502, 503},
                exc_type=WeatherAPIError,
                jitter=_OPEN_METEO_RETRY_JITTER,
                max_elapsed_seconds=_OPEN_METEO_MAX_RETRY_ELAPSED_SECONDS,
            )
        daily = data.get("daily")
        if not daily:
            raise WeatherAPIError(f"No 'daily' data in NBM response. Keys: {list(data.keys())}")

        times: list[str] = daily.get("time", [])
        try:
            date_idx = times.index(date_str)
        except ValueError:
            raise WeatherAPIError(f"Date {date_str} not found in NBM response. Available: {times}")

        values = daily.get(str(daily_variable), [])
        if date_idx >= len(values) or values[date_idx] is None:
            raise WeatherAPIError(
                f"No NBM value for {daily_variable} on {date_str}. Daily keys: {list(daily.keys())}"
            )

        temperature = float(values[date_idx])

        logger.info(
            "NBM point forecast for %s on %s: %.1f°F [%s]",
            city,
            forecast_date,
            temperature,
            daily_variable,
        )

        return NbmPointForecast(
            city=city,
            forecast_date=forecast_date,
            daily_variable=daily_variable,
            temperature=temperature,
        )

    @staticmethod
    def _parse_ensemble_temperatures(
        data: dict[str, Any],
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ) -> list[float]:
        """
        Extract ensemble member temperatures from the Open-Meteo response.

        The ensemble endpoint returns daily data with keys like:
            "daily": {
                "time": ["2026-03-15"],
                "temperature_2m_max_member01": [72.3],
                "temperature_2m_max_member02": [71.8],
                ...
            }

        The key prefix matches the requested daily_variable
        (e.g. ``temperature_2m_min`` for low-temp markets).
        """
        daily = data.get("daily")
        if not daily:
            raise WeatherAPIError(
                f"No 'daily' data in Open-Meteo response. Keys: {list(data.keys())}"
            )

        times: list[str] = daily.get("time", [])
        date_str = forecast_date.isoformat()
        try:
            date_idx = times.index(date_str)
        except ValueError:
            raise WeatherAPIError(f"Date {date_str} not found in response. Available: {times}")

        member_prefix = f"{daily_variable}_member"
        temperatures: list[float] = []
        for key, values in daily.items():
            if key.startswith(member_prefix) and isinstance(values, list):
                if date_idx < len(values) and values[date_idx] is not None:
                    temperatures.append(float(values[date_idx]))

        if not temperatures:
            raise WeatherAPIError(
                f"No ensemble member data found for {daily_variable} on {date_str}. "
                f"Daily keys: {list(daily.keys())}"
            )

        logger.debug("Parsed %d ensemble members for %s", len(temperatures), date_str)
        return temperatures

    def compute_model_probability(
        self,
        forecast: EnsembleForecast,
        direction: Direction = Direction.ABOVE,
        floor_strike: float | None = None,
        cap_strike: float | None = None,
    ) -> float:
        """
        Compute the model probability from ensemble forecasts.

        The fraction of ensemble members above, below, or between the
        threshold(s) IS the model probability.

        Args:
            forecast: GFS ensemble forecast data.
            direction: Direction.ABOVE — probability of exceeding floor_strike
                       (NWS-rounded: T ≥ floor + 0.5).
                       Direction.BELOW — probability of being below cap_strike
                       (NWS-rounded: T < cap − 0.5).
                       Direction.BETWEEN — probability of being in [floor, cap]
                       (NWS-rounded: T ∈ [floor − 0.5, cap + 0.5)).
            floor_strike: Lower bound (required for ABOVE and BETWEEN).
            cap_strike: Upper bound (required for BELOW and BETWEEN).

        Returns:
            Model probability (0.0 to 1.0).
        """
        return _apply_nws_window(forecast, direction, floor_strike, cap_strike)


# ---------------------------------------------------------------------------
# Multi-model blending
# ---------------------------------------------------------------------------


def compute_blended_probability(
    forecasts: dict[str, EnsembleForecast],
    weights: dict[str, float],
    direction: Direction,
    floor_strike: float | None = None,
    cap_strike: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute weighted-average probability across multiple ensemble models.

    Renormalizes weights to only the models present in *forecasts* so that
    missing models (e.g., ECMWF beyond 16 days) degrade gracefully.

    Args:
        forecasts: model_name -> EnsembleForecast.
        weights: model_name -> raw weight.
        direction: ABOVE, BELOW, or BETWEEN.
        floor_strike: Lower bound (required for ABOVE and BETWEEN).
        cap_strike: Upper bound (required for BELOW and BETWEEN).

    Returns:
        ``(blended_probability, per_model_probabilities)``
    """
    available = {m: w for m, w in weights.items() if m in forecasts}
    total_weight = sum(available.values())

    if total_weight <= 0:
        return 0.0, {}

    per_model: dict[str, float] = {}
    blended = 0.0

    for model_name, weight in available.items():
        fc = forecasts[model_name]
        prob = _apply_nws_window(fc, direction, floor_strike, cap_strike)
        per_model[model_name] = prob
        blended += (weight / total_weight) * prob

    return blended, per_model


def blended_ensemble_median_std(
    forecasts: dict[str, EnsembleForecast],
) -> tuple[float, float]:
    """Pooled median and population std-dev over all ensemble members.

    Returned together so callers (e.g. the strike-proximity gate) can score
    markets against the same distribution the blended probability was computed on.
    Returns ``(0.0, 0.0)`` if fewer than two members are available.
    """
    all_temps: list[float] = []
    for fc in forecasts.values():
        all_temps.extend(fc.temperatures)
    if len(all_temps) < 2:
        return 0.0, 0.0
    return statistics.median(all_temps), statistics.pstdev(all_temps)


def blended_ensemble_std(
    forecasts: dict[str, EnsembleForecast],
) -> float:
    """Population std-dev of all ensemble members pooled across models.

    Captures both intra-model spread and inter-model disagreement — a better
    uncertainty measure than any single model's std-dev.
    """
    return blended_ensemble_median_std(forecasts)[1]


def blended_forecast_mean(
    forecasts: dict[str, EnsembleForecast],
    weights: dict[str, float],
) -> float:
    """Weight-blended forecast mean (°F) across ensemble models.

    Uses the SAME weight-renormalization as :func:`compute_blended_probability`
    (restrict to models present in *forecasts*, normalize by their weight sum)
    so a consumer dispatching on this mean stays consistent with the blended
    probability computed alongside it (the boundary-distance filter relies on
    this). Falls back to an equal-weight mean over the present models when no
    positive weights are available (degenerate config). Returns ``nan`` when
    *forecasts* is empty so callers can detect the no-data case.
    """
    if not forecasts:
        return float("nan")
    available = {m: w for m, w in weights.items() if m in forecasts}
    total_weight = sum(available.values())
    if total_weight > 0:
        return sum(
            (w / total_weight) * forecasts[m].mean_temperature for m, w in available.items()
        )
    # Degenerate weights (sum<=0): equal-weight mean over present models.
    return sum(fc.mean_temperature for fc in forecasts.values()) / len(forecasts)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WeatherAPIError(Exception):
    """Raised when the Open-Meteo API request fails or returns unexpected data."""
