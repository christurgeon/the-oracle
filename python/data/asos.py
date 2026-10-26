"""Runtime IEM/ASOS hourly-observation client (fail-open) and climatology loader.

Fetches observed temperatures for a city's ASOS station via the Iowa
Environmental Mesonet (IEM) ASOS API and returns the observed max-so-far
for a HIGH.  Also provides ``ClimatologyTable``, a pure in-memory loader for
the per-``(city, month, hour_utc)`` gain statistics produced by
``scripts/build_asos_climatology.py``.

CRITICAL CONTRACT: ``max_so_far`` is **fail-open** — it swallows ALL
exceptions and returns ``None`` so an IEM outage degrades the strategy to
"no signal" rather than crashing the orchestrator.

City -> ASOS station map is asserted against ``CITY_TO_CLI_LOCATION`` at
import so a mismatched station (e.g. CHI ORD vs MDW settlement) is caught
early.

IEM endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
Station ids: 4-letter ICAO codes (KNYC, KMDW, KLAX, KMIA, KAUS) per the
prototype script. The 4-letter codes are what IEM expects; we strip the
leading "K" when normalising to our internal 3-letter convention for the
returned rows.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import httpx

from python.data.nws import CITY_TO_CLI_LOCATION
from python.utils.retry import async_retry_request

logger = logging.getLogger(__name__)

# IEM ASOS endpoint (same one used by the research prototype script)
_IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Mapping from internal city code -> IEM 4-letter ICAO station code.
# CHI MUST use MDW (Midway) — Kalshi settles on Midway, not O'Hare.
# These are the ICAO codes IEM actually accepts (matches the prototype's
# CITY_STATIONS dict which used KNYC, KMDW, KLAX, KMIA, KAUS).
ASOS_STATION: dict[str, str] = {
    "NYC": "KNYC",
    "CHI": "KMDW",
    "LAX": "KLAX",
    "MIA": "KMIA",
    "AUS": "KAUS",
}

# City LOCAL-STANDARD-TIME (LST) offsets — MUST match
# ``build_asos_climatology.py`` CITY_STD_OFFSET_HOURS exactly so that the runtime
# M-frame (the climate day used to compute the running extremum) is bit-identical
# to the fit frame used when computing gains during climatology build.
#
# NWS CLI climate days run midnight-to-midnight LOCAL STANDARD TIME and do NOT
# observe DST (spec §0a; memo finding #1). Using a civil ``ZoneInfo`` (which
# shifts with DST) mis-assigns the 00:00–01:00 LDT boundary hour to the wrong
# climate day during DST, contaminating M. We therefore use FIXED standard-time
# offsets year-round:
#   NYC / MIA  EST = UTC−5   (Eastern Standard, no DST)
#   CHI / AUS  CST = UTC−6   (Central Standard, no DST)
#   LAX        PST = UTC−8   (Pacific Standard, no DST)
CITY_STD_OFFSET_HOURS: dict[str, int] = {
    "NYC": -5,
    "CHI": -6,
    "LAX": -8,
    "MIA": -5,
    "AUS": -6,
}

# Fixed-offset tzinfo objects derived from CITY_STD_OFFSET_HOURS. ``timezone``
# with a fixed offset never applies DST, so ``dt.astimezone(tz).date()`` yields
# the LST climate day.
CITY_TZ: dict[str, timezone] = {
    city: timezone(timedelta(hours=off)) for city, off in CITY_STD_OFFSET_HOURS.items()
}

# Station-identity assertion: checks both KEYS (city subset) and VALUES (station
# matches CLI settlement station).  This catches the CHI/KORD vs KMDW trap that
# a pure key-subset check misses.
for _city, _icao in ASOS_STATION.items():
    assert _city in CITY_TO_CLI_LOCATION, (
        f"ASOS_STATION city {_city!r} not in CITY_TO_CLI_LOCATION"
    )
    _short = _icao[1:] if (len(_icao) == 4 and _icao.startswith("K")) else _icao
    _expected = CITY_TO_CLI_LOCATION[_city]
    assert _short == _expected, (
        f"ASOS_STATION[{_city!r}]={_icao!r} -> station {_short!r} does not match "
        f"settlement station {_expected!r} for {_city} "
        f"(Kalshi settles {_city} on {_expected})"
    )

# Shared-station guard: ``max_so_far_batch`` keys its cache by station and frames
# each station's M to the FIRST city using it (see ``city_for_station`` below).
# That is only correct while the station->city map is 1:1. No two cities share a
# station today; this assertion makes a future collision (e.g. a Stage 2 city
# reusing an existing station with a different LST offset) fail loudly at import
# rather than silently mis-framing M to the wrong timezone.
assert len(set(ASOS_STATION.values())) == len(ASOS_STATION), (
    f"ASOS_STATION has a shared station (one station mapped to multiple cities); "
    f"max_so_far_batch assumes one station <-> one city/LST offset: {ASOS_STATION}"
)
del _city, _icao, _short, _expected  # avoid polluting module namespace

# Set of tmpf values that mean "missing" in IEM CSV
_MISSING_TMPF = {"", "M", "T"}


@dataclass(frozen=True)
class AsosObservation:
    M: float  # noqa: N815  — physics name: observed max-so-far (°F)
    age_hours: float
    n_obs: int


def _icao_to_short(icao: str) -> str:
    """Strip leading 'K' from 4-letter ICAO code -> 3-letter short code."""
    if len(icao) == 4 and icao.startswith("K"):
        return icao[1:]
    return icao


def _parse_iem_csv_multi(text: str) -> dict[str, list[tuple[str, datetime, float]]]:
    """Demultiplex an IEM CSV response into per-station row lists.

    The IEM ASOS endpoint returns ALL requested stations' rows in a single CSV
    with a ``station`` column.  This is the single source of truth for parsing
    the IEM CSV shape; both the single-station (:func:`_parse_iem_csv`) and the
    batch (``max_so_far_batch``) paths build on it.

    Each row is keyed by its **short** station code (leading ``"K"`` stripped,
    e.g. ``"KMIA" -> "MIA"``) so callers can look up by the short code returned
    by :func:`_icao_to_short`.

    Skips rows where ``tmpf`` is missing (``"M"``), trace (``"T"``), or blank.
    ``valid`` is expected to already be UTC per the ``tz=Etc/UTC`` request param.

    Args:
        text: Raw CSV text from IEM (one or many stations).

    Returns:
        Mapping ``short_station -> [(short_station, valid_utc, tmpf_float), ...]``.
        Stations absent from the response simply do not appear as keys.
    """
    out: dict[str, list[tuple[str, datetime, float]]] = {}
    sio = StringIO(text)
    reader = csv.reader(sio)
    header = next(reader, None)
    if header is None:
        return out
    h = [c.strip().lower() for c in header]
    try:
        i_station = h.index("station")
        i_valid = h.index("valid")
        i_tmpf = h.index("tmpf")
    except ValueError:
        logger.warning("IEM response has unexpected header: %s", header)
        return out

    for row in reader:
        if len(row) <= max(i_station, i_valid, i_tmpf):
            continue
        short = _icao_to_short(row[i_station].strip().upper())
        tmpf_str = row[i_tmpf].strip()
        if tmpf_str in _MISSING_TMPF:
            continue
        try:
            tmpf = float(tmpf_str)
        except ValueError:
            continue
        valid_str = row[i_valid].strip()
        # IEM returns "YYYY-MM-DD HH:MM" (UTC); some responses include seconds
        valid_dt: datetime | None = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                valid_dt = datetime.strptime(valid_str[:19], fmt).replace(tzinfo=UTC)
                break
            except ValueError:
                continue
        if valid_dt is None:
            continue
        out.setdefault(short, []).append((short, valid_dt, tmpf))
    return out


def _parse_iem_csv(
    text: str,
    station: str,
) -> list[tuple[str, datetime, float]]:
    """Parse a single-station IEM CSV response into ``(short_station, valid_utc, tmpf)`` rows.

    Thin wrapper over :func:`_parse_iem_csv_multi` that selects the requested
    station's rows (matched on the short code).  Preserves the historical
    single-station signature/behavior for existing callers and tests.

    Args:
        text: Raw CSV text from IEM.
        station: The ICAO station code used in the request (e.g. ``"KMIA"``).

    Returns:
        List of ``(short_station, valid_utc, tmpf_float)`` tuples for *station*.
    """
    short = _icao_to_short(station.upper())
    return _parse_iem_csv_multi(text).get(short, [])


def _filter_to_local_day(
    rows: list[tuple[str, datetime, float]],
    city: str,
    local_day: date,
) -> list[tuple[str, datetime, float]]:
    """Return only rows whose UTC timestamp falls on *local_day* in the LST climate frame.

    NWS CLI climate days run midnight-to-midnight LOCAL STANDARD TIME (no DST).
    The climatology builder (``scripts/build_asos_climatology.py``) groups
    observations by the same fixed-offset LST date — ``dt_utc.astimezone(tz).date()``
    where ``tz`` is the city's fixed standard-time offset.  We apply the identical
    transformation at runtime so that ``max_so_far``'s M-frame matches the fit
    frame exactly (train/serve parity).

    For all five cities (NYC/MIA UTC-5, CHI/AUS UTC-6, LAX UTC-8 — fixed,
    no DST) the early UTC hours of a UTC day belong to the PRIOR LST climate
    day.  Without this filter, rows fetched from IEM for a UTC calendar window
    would contaminate ``M`` with the prior day's temperatures.

    Args:
        rows: Parsed IEM rows ``(short_station, valid_utc, tmpf)``.
        city: Internal city code (e.g. ``"LAX"``).
        local_day: The LST climate date to keep.

    Returns:
        Subset of *rows* where ``valid_utc.astimezone(CITY_TZ[city]).date() == local_day``.
        If *city* is not in ``CITY_TZ``, returns *rows* unchanged (fail-open).
    """
    tz = CITY_TZ.get(city)
    if tz is None:
        return rows
    return [r for r in rows if r[1].astimezone(tz).date() == local_day]


class AsosClient:
    """Async IEM/ASOS client with in-process caching.

    All public methods are fail-open — any exception returns ``None`` rather
    than propagating to the caller.
    """

    def __init__(
        self,
        *,
        cache_ttl_s: float = 600.0,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        # Default TTL 600s (10 min) matches ASOS's ~hourly publish cadence and
        # keeps IEM request volume low. NOTE: a cache hit returns the
        # AsosObservation.age_hours frozen at fetch time, so observation age is
        # under-reported by up to the TTL — harmless at 600s against the 2h
        # asos_max_staleness_hours freshness guard. Production overrides this
        # from weather config (strategies.intraday_asos_high.asos_obs_cache_ttl_seconds).
        # Cache keyed by (station_icao, local_day) -> (cached_at_monotonic, result)
        self._cache: dict[tuple[str, date], tuple[float, AsosObservation | None]] = {}
        self._cache_ttl_s = cache_ttl_s
        self._now = now_fn or (lambda: datetime.now(UTC))

    @staticmethod
    def _build_iem_url(stations: list[str], local_day: date) -> str:
        """Build a single IEM ASOS request URL covering *stations* for *local_day*.

        IEM accepts **repeated** ``station=`` parameters and returns all the
        requested stations' rows in one CSV (demuxed by :func:`_parse_iem_csv_multi`).
        It also requires repeated ``report_type`` keys, so the query string is
        constructed explicitly — httpx would encode lists with indexed suffixes
        (``station[0]=...``) which IEM does not accept.

        The window covers ``local_day`` (UTC) through ``local_day+1`` (UTC)
        inclusive — guaranteed to contain the full city-local calendar day for
        any city west of UTC (the widest offset is UTC-8 for LAX on PST).

        Args:
            stations: IEM 4-letter ICAO station codes (e.g. ``["KMIA", "KAUS"]``).
            local_day: The city-local calendar date to cover.

        Returns:
            The fully-formed request URL (single GET).
        """
        import datetime as _dt
        from urllib.parse import urlencode

        day2 = local_day + _dt.timedelta(days=1)

        base_params: dict[str, str | int] = {
            "data": "tmpf",
            "year1": local_day.year,
            "month1": local_day.month,
            "day1": local_day.day,
            "hour1": 0,
            "year2": day2.year,
            "month2": day2.month,
            "day2": day2.day,
            "hour2": 23,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "missing": "M",
            "trace": "T",
            "direct": "no",
        }
        report_types: list[int] = [3, 4]
        repeated = [("station", s) for s in stations] + [
            ("report_type", rt) for rt in report_types
        ]
        qs = urlencode(base_params) + "&" + urlencode(repeated)
        return f"{_IEM_URL}?{qs}"

    async def _fetch_rows_batch(
        self,
        stations: list[str],
        local_day: date,
    ) -> dict[str, list[tuple[str, datetime, float]]]:
        """Fetch rows for MANY stations in ONE IEM request, demuxed by station.

        This is the single underlying HTTP fetch for both the batch
        (``max_so_far_batch``) and single-station (``_fetch_rows`` →
        ``max_so_far``) paths. One GET with repeated ``station=`` params; the
        single CSV response is split by the ``station`` column.

        The raw rows are NOT yet filtered to the local day — callers apply
        ``_filter_to_local_day`` before computing M.

        Wrapped in ``async_retry_request`` for exponential backoff on 429/5xx.
        The retry covers the ONE batched request as a single unit — it never
        fans back out into per-station requests.

        Args:
            stations: IEM 4-letter ICAO station codes (e.g. ``["KMIA", "KAUS"]``).
            local_day: The city-local calendar date to cover.

        Returns:
            ``short_station -> rows`` (see :func:`_parse_iem_csv_multi`).
            Stations with no rows in the response are absent from the mapping.

        Raises:
            Any network/parse exception — callers are responsible for catching.
        """
        url = self._build_iem_url(stations, local_day)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await async_retry_request(
                lambda: client.get(url),
                max_retries=3,
                retry_delays=[1.0, 2.0, 4.0],
                retryable_statuses={429, 500, 502, 503},
                error_label="IEM-ASOS",
            )
        return _parse_iem_csv_multi(response.text)

    async def _fetch_rows(
        self,
        station: str,
        local_day: date,
    ) -> list[tuple[str, datetime, float]]:
        """Return ``[(short_station, valid_utc, tmpf), ...]`` for one station/local_day.

        Single-station convenience over :func:`_fetch_rows_batch` (a 1-element
        batch). Kept as a named seam so existing tests that monkeypatch
        ``_fetch_rows`` continue to work.

        Raises:
            Any network/parse exception — callers are responsible for catching.
            (``max_so_far`` provides the fail-open wrapper.)
        """
        by_station = await self._fetch_rows_batch([station], local_day)
        return by_station.get(_icao_to_short(station.upper()), [])

    def _compute_obs(
        self,
        rows: list[tuple[str, datetime, float]],
        city: str,
        local_day: date,
    ) -> AsosObservation | None:
        """Compute the max-so-far ``AsosObservation`` from raw station rows.

        Applies the LST local-day filter (Stage 0a frame), then computes the
        running maximum and observation age. Returns ``None`` when no usable
        rows remain for the city-local day. Shared by ``max_so_far`` and
        ``max_so_far_batch`` so M-framing/age logic lives in one place.
        """
        rows = _filter_to_local_day(rows, city, local_day)
        temps = [(v, t) for (_s, v, t) in rows if t is not None]
        if not temps:
            return None
        M = max(t for _v, t in temps)  # noqa: N806
        latest = max(v for v, _t in temps)
        age = (self._now() - latest).total_seconds() / 3600.0
        return AsosObservation(
            M=float(M),  # noqa: N803
            age_hours=float(age),
            n_obs=len(temps),
        )

    async def max_so_far(
        self,
        *,
        city: str,
        local_day: date,
    ) -> AsosObservation | None:
        """Return observed max-so-far temperature for *city* on *local_day*.

        Looks up the station from ``ASOS_STATION``, fetches (or returns
        cached) IEM rows, and computes the running maximum plus observation
        age.

        FAIL-OPEN: returns ``None`` on ANY exception — network errors, parse
        failures, missing cities — so an IEM outage degrades to "no signal"
        without crashing the orchestrator.
        """
        try:
            station = ASOS_STATION.get(city)
            if station is None:
                return None

            # Check in-process cache
            import time

            cache_key = (station, local_day)
            now_mono = time.monotonic()
            if cache_key in self._cache:
                cached_at, cached_result = self._cache[cache_key]
                if now_mono - cached_at < self._cache_ttl_s:
                    return cached_result

            rows = await self._fetch_rows(station, local_day)
            result = self._compute_obs(rows, city, local_day)
            self._cache[cache_key] = (now_mono, result)
            return result

        except Exception:
            logger.warning(
                "AsosClient.max_so_far fail-open for %s/%s",
                city,
                local_day,
                exc_info=True,
            )
            return None

    async def max_so_far_batch(
        self,
        *,
        cities: list[str],
        local_day: date,
    ) -> dict[str, float | None]:
        """Fetch max-so-far for MANY cities in ONE IEM request; return per-city M.

        This is the N→1 collapse: the strategy evaluates N cells per tick and,
        instead of issuing N near-simultaneous single-station requests (the
        2026-06-06 IEM 429 storm at 5 stations; guaranteed breakage at any
        expansion), this issues ONE request with repeated ``station=`` params.

        It populates the existing per-``(station, local_day)`` cache for ALL
        fetched stations from the one response, so subsequent per-city
        ``max_so_far`` calls within the TTL window hit cache and make ZERO
        additional fetches. Honors the existing cache TTL and the existing LST
        local-day framing (Stage 0a).

        FAIL-OPEN, per-city: any error → ``None`` for the affected city; a
        station missing from the response → ``None`` for that city while others
        still resolve. Never raises into the caller. If the SINGLE batched
        request itself fails, every city resolves to ``None`` (the cache is left
        untouched so a later retry can re-fetch).

        Args:
            cities: Internal city codes (e.g. ``["MIA", "AUS"]``). Unknown or
                unmapped cities resolve to ``None``.
            local_day: The city-local calendar date (LST frame).

        Returns:
            ``{city: M_or_None}`` for every requested city. ``M`` is the
            observed daily max-so-far in °F (the ``AsosObservation.M`` value);
            the full observation (with age/n_obs) is available via the warmed
            cache through ``max_so_far``.
        """
        import time

        out: dict[str, float | None] = {}
        # Resolve the station for each requested city (dedup stations; two
        # cities never share a station today but the contract is per-station).
        city_to_station: dict[str, str] = {}
        stations_to_fetch: list[str] = []
        seen_stations: set[str] = set()
        for city in cities:
            out.setdefault(city, None)  # default so every city appears
            station = ASOS_STATION.get(city)
            if station is None:
                continue
            city_to_station[city] = station

        now_mono = time.monotonic()
        # Build the fetch list from cache-cold stations only; warm-cache
        # stations are resolved straight from the cache (zero fetch).
        for station in city_to_station.values():
            cache_key = (station, local_day)
            cached = self._cache.get(cache_key)
            if cached is not None and now_mono - cached[0] < self._cache_ttl_s:
                continue
            if station not in seen_stations:
                seen_stations.add(station)
                stations_to_fetch.append(station)

        by_station: dict[str, list[tuple[str, datetime, float]]] = {}
        fetched_ok = False
        if stations_to_fetch:
            try:
                by_station = await self._fetch_rows_batch(stations_to_fetch, local_day)
                fetched_ok = True
            except Exception:
                logger.warning(
                    "AsosClient.max_so_far_batch fetch fail-open for stations=%s day=%s",
                    stations_to_fetch,
                    local_day,
                    exc_info=True,
                )
                fetched_ok = False

        # Populate the cache for every station we attempted to fetch (so a
        # station that returned no rows caches a None result and does not get
        # re-fetched every tick within the TTL). On a failed batch request we
        # leave the cache untouched so a subsequent tick re-attempts the fetch.
        if fetched_ok:
            for station in stations_to_fetch:
                short = _icao_to_short(station.upper())
                # Find any city using this station to apply the correct LST frame.
                # ASSUMPTION (enforced by the import-time shared-station assert
                # above): one station <-> one city, so "first city using this
                # station" is the ONLY city and its LST offset is unambiguous. If a
                # future stage maps two cities to one station, both would be framed
                # to whichever city is found first here -- re-key the cache by
                # (station, local_day, offset) or split per city before then.
                city_for_station = next(
                    (c for c, s in city_to_station.items() if s == station),
                    None,
                )
                if city_for_station is None:
                    continue
                rows = by_station.get(short, [])
                result = self._compute_obs(rows, city_for_station, local_day)
                self._cache[(station, local_day)] = (now_mono, result)

        # Resolve each city from the (now warmed) cache.
        for city, station in city_to_station.items():
            try:
                cached = self._cache.get((station, local_day))
                if cached is None:
                    out[city] = None
                    continue
                _cached_at, obs = cached
                out[city] = obs.M if obs is not None else None
            except Exception:
                logger.warning(
                    "AsosClient.max_so_far_batch resolve fail-open for %s/%s",
                    city,
                    local_day,
                    exc_info=True,
                )
                out[city] = None

        return out


# ---------------------------------------------------------------------------
# ClimatologyTable — pure in-memory loader for the gain-statistics artifact
# ---------------------------------------------------------------------------


class ClimatologyTable:
    """Lookup table for per-``(city, var, month, hour_utc)`` gain statistics.

    Loaded from the JSON artifact produced by
    ``scripts/build_asos_climatology.py`` (schema: ``data/asos/climatology.json``).

    Lookup ladder (mirrors the builder's fallback ladder):
      1. Exact ``(city, var, month, hour)`` cell — ``source`` in
         ``{"primary", "hour_pool"}`` — returns ``(mu_gain, sigma_gain, n)``.
      2. If the exact cell is absent or has ``source == "unavailable"``, scan
         all cells for the given ``(city, var)`` that have
         ``source == "hour_pool"`` and whose hour component matches.
      3. If neither resolves, return ``None`` (caller treats as unavailable).

    Pure: no scipy, no numpy, no I/O beyond the initial ``load()`` call.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        # ``raw`` is the parsed JSON dict (top-level keys: spec_version,
        # fit_window_utc, params, cities, diagnostics).
        self._cities: dict[str, Any] = raw.get("cities", {})

    @classmethod
    def load(cls, path: str | Path) -> ClimatologyTable:
        """Load the climatology artifact from *path*.

        Args:
            path: Filesystem path to the JSON artifact.

        Returns:
            A ``ClimatologyTable`` ready for ``lookup()`` calls.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Climatology artifact not found: {p}")
        raw = json.loads(p.read_text())
        return cls(raw)

    def lookup(
        self,
        city: str,
        month: int,
        hour: int,
        *,
        var: str = "high",
    ) -> tuple[float, float, int, str] | None:
        """Return ``(mu_gain, sigma_gain, n, source)`` for the requested cell, or ``None``.

        Args:
            city: Internal city code, e.g. ``"MIA"``.
            month: Calendar month (1–12).
            hour: UTC hour (0–23).
            var: ``"high"`` or ``"low"`` (default ``"high"``).

        Returns:
            ``(mu_gain, sigma_gain, n, source)`` from the best-matching cell,
            where ``source`` is the cell's ``"primary"`` or ``"hour_pool"``
            label from the artifact. Returns ``None`` when neither the exact
            cell nor any pool cell covers this ``(city, var, hour)``
            combination.
        """
        city_block = self._cities.get(city)
        if city_block is None:
            return None
        var_block = city_block.get(var)
        if var_block is None:
            return None
        cells: dict[str, Any] = var_block.get("cells", {})

        cell_key = f"{month:02d}-{hour:02d}"
        cell = cells.get(cell_key)

        # Step 1: exact cell with usable data
        if cell is not None and cell.get("source") in ("primary", "hour_pool"):
            return (
                float(cell["mu"]),
                float(cell["sigma"]),
                int(cell["n"]),
                str(cell["source"]),
            )

        # Step 2: fall back to any hour_pool cell for this (city, var, hour)
        # The builder stores pool cells under any month key; the hour component
        # of the key (``MM-HH``) must match the requested ``hour``.
        hour_suffix = f"-{hour:02d}"
        for key, c in cells.items():
            if key.endswith(hour_suffix) and c.get("source") == "hour_pool":
                return (
                    float(c["mu"]),
                    float(c["sigma"]),
                    int(c["n"]),
                    str(c["source"]),
                )

        return None
