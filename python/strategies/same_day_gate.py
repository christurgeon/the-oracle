"""Same-day trading gate for the weather-edge strategy.

Returns a :class:`SameDayDecision` for ``(city, forecast_date, daily_variable)``
triples on the trading day itself. Decision order (early-exit):

  1. ``BLOCK_DISABLED`` — feature flag off or city not in allow-list. No HTTP.
  2. ``BLOCK_PEAK_PASSED`` — local clock past the configured cutoff. No HTTP.
  3. Cache hit — sticky positive (BLOCK), TTL-bounded negative (ALLOW or refetch).
  4. Fresh NWS fetch — bounded by ``asyncio.wait_for`` at the call site.

Thread-safety: single-process, single-loop. Cache is an in-memory dict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from python.common import DailyVariable
from python.data.nws import CliObservation, NWSAPIError, NWSClient

logger = logging.getLogger(__name__)


class SameDayDecision(Enum):
    """Terminal decision returned by :meth:`SameDayGate.check`."""

    ALLOW = "allow"
    BLOCK_PEAK_PASSED = "peak_passed"
    BLOCK_NWS_OBSERVATION_EXISTS = "nws_observation_exists"
    BLOCK_NWS_FETCH_FAILED = "nws_fetch_failed"
    BLOCK_DISABLED = "same_day_disabled"


PEAK_LOCAL_TIME: dict[DailyVariable, time] = {
    DailyVariable.TEMP_MAX: time(15, 0),
    DailyVariable.TEMP_MIN: time(6, 0),
}


# IANA timezone per city. Missing keys must raise — silently defaulting to UTC
# would compute the cutoff against the wrong wall clock.
CITY_TIMEZONE: dict[str, ZoneInfo] = {
    "NYC": ZoneInfo("America/New_York"),
    "MIA": ZoneInfo("America/New_York"),
    "CHI": ZoneInfo("America/Chicago"),
    "AUS": ZoneInfo("America/Chicago"),
    "LAX": ZoneInfo("America/Los_Angeles"),
    "SEA": ZoneInfo("America/Los_Angeles"),
    "DEN": ZoneInfo("America/Denver"),
}


@dataclass(frozen=True)
class _CacheEntry:
    timestamp: datetime  # UTC
    has_observation: bool


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _extract_value(obs: CliObservation, var: DailyVariable) -> int | None:
    """Return the relevant field for the requested variable, or ``None``.

    Raises ``ValueError`` on an unsupported ``DailyVariable`` so adding precip /
    wind / snowfall later cannot silently default to ALLOW.
    """
    if var == DailyVariable.TEMP_MAX:
        return obs.high_f
    if var == DailyVariable.TEMP_MIN:
        return obs.low_f
    raise ValueError(f"SameDayGate: unsupported daily_variable {var!r}")


class SameDayGate:
    """Gate that decides whether to allow a same-day weather-market trade."""

    def __init__(
        self,
        nws_client: NWSClient | None,
        *,
        enabled: bool,
        allowed_cities: frozenset[str],
        cache_ttl_seconds: int = 600,
        peak_cutoff_minutes: int = 60,
        near_peak_window_minutes: int = 120,
        near_peak_ttl_seconds: int = 90,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._nws_client = nws_client
        self._enabled = enabled
        self._allowed_cities = allowed_cities
        self._cache_ttl_seconds = int(cache_ttl_seconds)
        self._peak_cutoff_minutes = int(peak_cutoff_minutes)
        self._near_peak_window_minutes = int(near_peak_window_minutes)
        self._near_peak_ttl_seconds = int(near_peak_ttl_seconds)
        self._clock = clock
        self._cache: dict[tuple[str, date, DailyVariable], _CacheEntry] = {}

    @classmethod
    def disabled(cls) -> SameDayGate:
        """Construct a permanently-disabled sentinel (never opens a network client)."""
        return cls(nws_client=None, enabled=False, allowed_cities=frozenset())

    async def check(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable,
    ) -> SameDayDecision:
        """Return the gate decision for one (city, date, daily_variable) triple."""
        if not self._enabled:
            return SameDayDecision.BLOCK_DISABLED
        if city not in self._allowed_cities:
            return SameDayDecision.BLOCK_DISABLED
        # Single clock read per check — keeps test fixtures with deterministic
        # clock iterators predictable, and avoids cross-method skew.
        now = self._clock()
        self._prune_stale(forecast_date)
        if self._is_past_cutoff(city, forecast_date, daily_variable, now=now):
            return SameDayDecision.BLOCK_PEAK_PASSED

        key = (city, forecast_date, daily_variable)
        cached = self._cache.get(key)
        if cached is not None:
            if cached.has_observation:
                return SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS
            if not self._is_negative_expired(cached, key, now):
                return SameDayDecision.ALLOW

        if self._nws_client is None:
            # Sentinel without a client (shouldn't reach here when enabled=True
            # — we'd already have BLOCK_DISABLED'd. Fail closed defensively.)
            return SameDayDecision.BLOCK_NWS_FETCH_FAILED

        try:
            obs = await self._nws_client.get_daily_observation(city, forecast_date)
        except NWSAPIError:
            return SameDayDecision.BLOCK_NWS_FETCH_FAILED

        if obs is None:
            self._cache[key] = _CacheEntry(timestamp=now, has_observation=False)
            return SameDayDecision.ALLOW

        relevant = _extract_value(obs, daily_variable)
        has_obs = relevant is not None
        self._cache[key] = _CacheEntry(timestamp=now, has_observation=has_obs)
        return (
            SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS
            if has_obs
            else SameDayDecision.ALLOW
        )

    def _prune_stale(self, current_forecast_date: date) -> None:
        """Drop cache entries for past forecast dates so the dict stays bounded."""
        stale = [k for k in self._cache if k[1] < current_forecast_date]
        for k in stale:
            del self._cache[k]

    def _is_negative_expired(
        self,
        entry: _CacheEntry,
        key: tuple[str, date, DailyVariable],
        now: datetime,
    ) -> bool:
        """Return True if the negative cache entry is older than the applicable TTL.

        Within ±``near_peak_window_minutes`` of the peak (city-local), use
        ``near_peak_ttl_seconds``; otherwise the default ``cache_ttl_seconds``.
        """
        ttl_s = self._applicable_ttl_seconds(key, now)
        age_s = (now - entry.timestamp).total_seconds()
        return age_s > ttl_s

    def _applicable_ttl_seconds(
        self,
        key: tuple[str, date, DailyVariable],
        now: datetime,
    ) -> int:
        city, forecast_date, daily_variable = key
        peak = PEAK_LOCAL_TIME.get(daily_variable)
        if peak is None:
            return self._cache_ttl_seconds
        tz = CITY_TIMEZONE[city]
        local_now = now.astimezone(tz)
        peak_local = datetime.combine(forecast_date, peak, tzinfo=tz)
        delta_s = abs((local_now - peak_local).total_seconds())
        if delta_s <= self._near_peak_window_minutes * 60:
            return self._near_peak_ttl_seconds
        return self._cache_ttl_seconds

    def _is_past_cutoff(
        self,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return True if the city-local clock is past peak + peak_cutoff_minutes.

        Raises:
            KeyError: when ``city`` is not in CITY_TIMEZONE (we never silently
                default to UTC).
        """
        tz = CITY_TIMEZONE[city]
        local_now = (now if now is not None else self._clock()).astimezone(tz)
        if local_now.date() > forecast_date:
            return True
        peak = PEAK_LOCAL_TIME.get(daily_variable)
        if peak is None:
            # Unsupported variable — leave the decision to the NWS path.
            return False
        cutoff_local = datetime.combine(forecast_date, peak, tzinfo=tz) + timedelta(
            minutes=self._peak_cutoff_minutes
        )
        return local_now > cutoff_local

    async def aclose(self) -> None:
        """Close the underlying NWSClient if one was constructed."""
        if self._nws_client is not None:
            await self._nws_client.close()
