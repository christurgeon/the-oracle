"""KXRAIN ticker, geography and local-standard-time window primitives.

Hermetic: stdlib only, no I/O. Settlement day is midnight-to-midnight local
STANDARD time, year-round — fixed UTC offsets, never `ZoneInfo`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from python.data.nws import CITY_TO_CLI_LOCATION
from python.utils.ticker_dates import parse_ticker_date

logger = logging.getLogger(__name__)

__all__ = [
    "KXRAIN_SERIES_TICKER",
    "KXRAIN_CITY_STATION",
    "KXRAIN_CITY_STD_OFFSET_HOURS",
    "KXRAIN_CITY_TZ",
    "KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE",
    "KXRAIN_LEGACY_CITY_ALIASES",
    "KXRAIN_PREREGISTERED_CITIES",
    "KxRainMarketKey",
    "parse_kxrain_ticker",
    "lst_day",
    "lst_local_datetime",
    "lst_local_time",
    "in_entry_window",
    "station_from_rules",
    "station_mismatch",
    "validate_station_map",
]

KXRAIN_SERIES_TICKER = "KXRAIN"

# City suffix -> NWS CLI station, read off each market's rules_primary text.
# CHI settles KXRAIN on ORD, not MDW (the temperature series' station).
KXRAIN_CITY_STATION: dict[str, str] = {
    "ATL": "ATL",
    "AUS": "AUS",
    "BOS": "BOS",
    "CHI": "ORD",
    "CLL": "CLL",
    "CMH": "CMH",
    "DAL": "DFW",
    "DC": "DCA",
    "DEN": "DEN",
    "EWR": "EWR",
    "HOU": "HOU",
    "LAX": "LAX",
    "LEX": "LEX",
    "LV": "LAS",
    "MIA": "MIA",
    "MIN": "MSP",
    "MKE": "MKE",
    "NOLA": "MSY",
    "NYC": "NYC",
    "OKC": "OKC",
    "PHIL": "PHL",
    "PHX": "PHX",
    "PIT": "PIT",
    "PVD": "PVD",
    "SATX": "SAT",
    "SEA": "SEA",
    "SFO": "SFO",
    "TTN": "TTN",
}

# Cities whose KXRAIN station deliberately differs from the temperature-series
# CLI location; validate_station_map() flags any other divergence.
KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE: frozenset[str] = frozenset({"CHI"})

# PHX is UTC-7 year-round: Arizona does not observe DST.
KXRAIN_CITY_STD_OFFSET_HOURS: dict[str, int] = {
    "ATL": -5,
    "AUS": -6,
    "BOS": -5,
    "CHI": -6,
    "CLL": -6,
    "CMH": -5,
    "DAL": -6,
    "DC": -5,
    "DEN": -7,
    "EWR": -5,
    "HOU": -6,
    "LAX": -8,
    "LEX": -5,
    "LV": -8,
    "MIA": -5,
    "MIN": -6,
    "MKE": -6,
    "NOLA": -6,
    "NYC": -5,
    "OKC": -6,
    "PHIL": -5,
    "PHX": -7,
    "PIT": -5,
    "PVD": -5,
    "SATX": -6,
    "SEA": -8,
    "SFO": -8,
    "TTN": -5,
}

# The gate universe: cities live when the rule was pre-registered. Cities launched
# later (PIT, 2026-09-23) are logged but never enter the gate.
KXRAIN_PREREGISTERED_CITIES: frozenset[str] = frozenset({
    "ATL", "AUS", "BOS", "CHI", "CLL", "CMH", "DAL", "DC", "DEN", "EWR",
    "HOU", "LAX", "LEX", "LV", "MIA", "MIN", "MKE", "NOLA", "NYC", "OKC",
    "PHIL", "PHX", "PVD", "SATX", "SEA", "SFO", "TTN",
})

KXRAIN_CITY_TZ: dict[str, timezone] = {
    city: timezone(timedelta(hours=off)) for city, off in KXRAIN_CITY_STD_OFFSET_HOURS.items()
}

# Retired suffix -> (canonical city, last settlement date used, inclusive).
# Date-bounded so a recycled suffix can't silently mis-attribute.
KXRAIN_LEGACY_CITY_ALIASES: dict[str, tuple[str, date]] = {
    "PROV": ("PVD", date(2026, 9, 16)),
    "MIL": ("MKE", date(2026, 9, 16)),
    "COL": ("CLL", date(2026, 9, 16)),
}

_VALID_STD_OFFSETS = frozenset({-5, -6, -7, -8})

# Matches both observed spellings ("CLIORD", "KPVD"); prose can change again.
_RULES_STATION_RE = re.compile(r"\bat (?:CLI|K)([A-Z0-9]{3})\b")

# Anchored on both ends so sibling series (e.g. "KXRAINWKND-...") don't match.
_KXRAIN_TICKER_RE = re.compile(
    rf"^{KXRAIN_SERIES_TICKER}-(\d{{2}}[A-Z]{{3}}\d{{2}})-([A-Z]+)$", re.IGNORECASE
)


@dataclass(frozen=True)
class KxRainMarketKey:
    """Everything a KXRAIN ticker encodes. `city` is canonical; `ticker_city_suffix` is verbatim."""

    ticker: str
    city: str
    station: str
    settlement_date: date
    lst_offset_hours: int
    ticker_city_suffix: str
    is_legacy_suffix: bool

    @property
    def corpus_key(self) -> tuple[str, date]:
        """`(station, settlement_date)`, not city/ticker — Kalshi renames the suffix."""
        return (self.station, self.settlement_date)


def parse_kxrain_ticker(ticker: str) -> KxRainMarketKey | None:
    """Parse a KXRAIN ticker into its city/station/date key. Never raises; `None` otherwise."""
    if not ticker:
        return None

    normalised = ticker.strip()
    match = _KXRAIN_TICKER_RE.match(normalised)
    if match is None:
        return None

    # Reuse the shared parser so accepted date formats stay in lockstep.
    settlement_date = parse_ticker_date(normalised)
    if settlement_date is None:
        return None

    suffix = match.group(2).upper()
    city = suffix
    is_legacy = False
    if city not in KXRAIN_CITY_STATION:
        alias = KXRAIN_LEGACY_CITY_ALIASES.get(suffix)
        if alias is not None and settlement_date <= alias[1]:
            city = alias[0]
            is_legacy = True

    station = KXRAIN_CITY_STATION.get(city)
    offset = KXRAIN_CITY_STD_OFFSET_HOURS.get(city)
    if station is None or offset is None:
        return None

    return KxRainMarketKey(
        ticker=normalised,
        city=city,
        station=station,
        settlement_date=settlement_date,
        lst_offset_hours=offset,
        ticker_city_suffix=suffix,
        is_legacy_suffix=is_legacy,
    )


def station_from_rules(rules_primary: str) -> str | None:
    """Extract the settlement station named in a market's rules text, or `None` if unreadable."""
    if not rules_primary:
        return None
    match = _RULES_STATION_RE.search(rules_primary)
    return match.group(1) if match else None


def station_mismatch(key: KxRainMarketKey, rules_primary: str) -> str | None:
    """Reconcile the station map against the contract text; `None` if they agree."""
    stated = station_from_rules(rules_primary)
    if stated is None:
        return (
            f"{key.ticker}: cannot read a station from rules_primary "
            "(wording changed — the settlement-source tripwire is blind)"
        )
    if stated != key.station:
        return (
            f"{key.ticker}: rules_primary says the market settles on {stated!r} but "
            f"KXRAIN_CITY_STATION[{key.city!r}] is {key.station!r}"
        )
    return None


def lst_local_datetime(city: str, now_utc: datetime) -> datetime:
    """Convert a UTC instant to the city's local STANDARD time. `KeyError` for an unknown city."""
    tz = KXRAIN_CITY_TZ.get(city)
    if tz is None:
        raise KeyError(f"Unknown KXRAIN city {city!r}; known: {', '.join(sorted(KXRAIN_CITY_TZ))}")
    if now_utc.tzinfo is None:
        raise ValueError(f"now_utc must be timezone-aware, got naive {now_utc!r}")
    return now_utc.astimezone(tz)


def lst_local_time(city: str, now_utc: datetime) -> time:
    """Wall-clock local standard time at ``now_utc`` — the frame the window uses."""
    return lst_local_datetime(city, now_utc).time()


def lst_day(city: str, now_utc: datetime) -> date:
    """The CLI climate day (local standard midnight to midnight) containing ``now_utc``."""
    return lst_local_datetime(city, now_utc).date()


def in_entry_window(city: str, now_utc: datetime, start: time, end: time) -> bool:
    """True when local standard time is in the half-open range `[start, end)`."""
    local = lst_local_time(city, now_utc)
    return start <= local < end


def validate_station_map() -> list[str]:
    """Check the station/offset maps; return human-readable mismatches, or `[]`. Never raises."""
    problems: list[str] = []

    missing_offsets = sorted(set(KXRAIN_CITY_STATION) - set(KXRAIN_CITY_STD_OFFSET_HOURS))
    if missing_offsets:
        problems.append(f"cities with a station but no LST offset: {missing_offsets}")

    orphan_offsets = sorted(set(KXRAIN_CITY_STD_OFFSET_HOURS) - set(KXRAIN_CITY_STATION))
    if orphan_offsets:
        problems.append(f"cities with an LST offset but no station: {orphan_offsets}")

    bad_offsets = sorted(
        f"{city}={off}"
        for city, off in KXRAIN_CITY_STD_OFFSET_HOURS.items()
        if off not in _VALID_STD_OFFSETS
    )
    if bad_offsets:
        problems.append(f"LST offsets outside {{-5,-6,-7,-8}}: {bad_offsets}")

    seen: dict[str, str] = {}
    for city, station in sorted(KXRAIN_CITY_STATION.items()):
        if station in seen:
            problems.append(
                f"station {station!r} maps to both {seen[station]!r} and {city!r} "
                "(the map must stay 1:1)"
            )
        else:
            seen[station] = city

    for city, station in sorted(KXRAIN_CITY_STATION.items()):
        temp_station = CITY_TO_CLI_LOCATION.get(city)
        if temp_station is None:
            continue
        diverges = station != temp_station
        expected = city in KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE
        if diverges and not expected:
            problems.append(
                f"{city}: KXRAIN station {station!r} != temperature CLI location "
                f"{temp_station!r} and the divergence is not declared"
            )
        elif expected and not diverges:
            problems.append(
                f"{city}: declared to diverge from the temperature CLI location but both "
                f"are {station!r} — Kalshi may have changed the settlement station"
            )

    return problems


def _log_station_map_validation() -> None:
    # Runs at import time (orchestrator.py imports this at module top level),
    # so an uncaught exception here would kill the live bot before startup.
    try:
        problems = validate_station_map()
    except Exception:  # pragma: no cover - defensive; validate_station_map is pure
        logger.exception("KXRAIN station-map validation raised; continuing with the map as-is")
        return
    for problem in problems:
        logger.warning("KXRAIN station map: %s", problem)


_log_station_map_validation()
