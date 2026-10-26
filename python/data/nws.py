"""
NWS Daily Climate Report (CLI) fetcher for settlement verification.

Fetches official high/low temperatures from the NWS API's CLI products —
the exact data source Kalshi uses to settle weather markets. Each CLI is
a fixed-width text report; we parse the TEMPERATURE section for MAXIMUM
and MINIMUM values.

No API key required. Only a User-Agent header is needed.

Usage::

    async with NWSClient() as client:
        obs = await client.get_daily_observation("NYC", date(2026, 3, 17))
        print(obs.high_f, obs.low_f)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from python.utils.retry import get_json

logger = logging.getLogger(__name__)

NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = "(The-Oracle, weather-bot@example.com)"

# ---------------------------------------------------------------------------
# City → NWS CLI location code mapping
# ---------------------------------------------------------------------------

# Maps our internal city codes to the NWS CLI location codes used in the
# /products/types/CLI/locations/{code} endpoint.
CITY_TO_CLI_LOCATION: dict[str, str] = {
    "NYC": "NYC",  # Central Park (WFO: OKX)
    "CHI": "MDW",  # Chicago Midway (WFO: LOT) — Kalshi settles on Midway, not O'Hare
    "MIA": "MIA",  # Miami International (WFO: MFL)
    "LAX": "LAX",  # Los Angeles International (WFO: LOX)
    "DEN": "DEN",  # Denver International (WFO: BOU)
    "AUS": "AUS",  # Austin-Bergstrom (WFO: EWX)
    "SEA": "SEA",  # Seattle-Tacoma (WFO: SEW)
    "PHIL": "PHL",  # Philadelphia International Airport (WFO: PHI)
    "BOS": "BOS",  # Boston Logan Airport (WFO: BOX)
    # New cities (collection-only) — code≠CLI where Kalshi settles on a station
    # whose CLI id differs from our internal code (mirrors CHI→MDW, PHIL→PHL).
    "ATL": "ATL",   # Hartsfield-Jackson Atlanta (WFO: FFC)
    "DAL": "DFW",   # Dallas-Fort Worth Intl (WFO: FWD)
    "HOU": "HOU",   # Houston Wm P. Hobby — Kalshi settles Hobby, NOT IAH (WFO: HGX)
    "LV": "LAS",    # Las Vegas Harry Reid Intl (WFO: VEF)
    "MIN": "MSP",   # Minneapolis-St Paul Intl (WFO: MPX)
    "NOLA": "MSY",  # New Orleans Louis Armstrong Intl (WFO: LIX)
    "OKC": "OKC",   # Oklahoma City Will Rogers World (WFO: OUN)
    "PHX": "PHX",   # Phoenix Sky Harbor Intl (WFO: PSR)
    "SATX": "SAT",  # San Antonio Intl (WFO: EWX)
    "SFO": "SFO",   # San Francisco Intl (WFO: MTR)
    "DC": "DCA",    # Washington Reagan National — NOT IAD/BWI (WFO: LWX)
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliObservation:
    """Parsed NWS CLI observation — official settlement values."""

    city: str  # Internal city code (NYC, CHI, etc.)
    cli_location: str  # NWS CLI location code (NYC, MDW, etc.)
    observation_date: date  # The date these temps were observed
    high_f: int | None  # Official daily high (Fahrenheit), None if missing
    low_f: int | None  # Official daily low (Fahrenheit), None if missing
    product_id: str  # NWS product UUID for traceability
    issued_at: str  # ISO timestamp of when the CLI was issued
    is_preliminary: bool  # True if "VALID TODAY" (afternoon/evening report)


# ---------------------------------------------------------------------------
# CLI text parser
# ---------------------------------------------------------------------------

# Matches: "  MAXIMUM         48   1207 AM  75    1945  50     -2       59"
# Also handles negative temps, "R" suffix (record), "MM" (missing)
_TEMP_PATTERN = re.compile(
    r"^\s*(MAXIMUM|MINIMUM)\s+(-?\d+)R?\s",
    re.MULTILINE,
)

# Matches: "...THE CHICAGO-OHARE CLIMATE SUMMARY FOR MARCH 17 2026..."
_DATE_PATTERN = re.compile(
    r"CLIMATE SUMMARY FOR (\w+)\s+(\d{1,2})\s+(\d{4})",
    re.IGNORECASE,
)

# Matches preliminary headers: "VALID [TODAY] AS OF HHMM <AM|PM> LOCAL TIME".
# Start-of-line anchor + LOCAL TIME suffix avoid false positives on CORRECTION
# remarks lines like "VALID AS OF 1100 AM CORRECTION: ...".
_PRELIMINARY_PATTERN = re.compile(
    r"^VALID\s+(TODAY\s+)?AS\s+OF\s+\d{3,4}\s+(AM|PM)\s+LOCAL\s+TIME",
    re.IGNORECASE | re.MULTILINE,
)

_MONTH_NAMES = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
    "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}


def parse_cli_text(
    text: str,
) -> tuple[date | None, int | None, int | None, bool]:
    """
    Parse a NWS CLI product text for the observation date and temperatures.

    Returns:
        (observation_date, high_f, low_f, is_preliminary) — date and temps
        may be None if not found or marked as missing (MM). is_preliminary
        is True when the CLI text says "VALID TODAY" (afternoon same-day
        report, before the final next-morning report).
    """
    obs_date: date | None = None
    date_match = _DATE_PATTERN.search(text)
    if date_match:
        month_name = date_match.group(1).upper()
        day = int(date_match.group(2))
        year = int(date_match.group(3))
        month = _MONTH_NAMES.get(month_name)
        if month:
            obs_date = date(year, month, day)

    high_f: int | None = None
    low_f: int | None = None

    for match in _TEMP_PATTERN.finditer(text):
        label = match.group(1).upper()
        value = int(match.group(2))
        if label == "MAXIMUM":
            high_f = value
        elif label == "MINIMUM":
            low_f = value

    is_preliminary = bool(_PRELIMINARY_PATTERN.search(text))

    return obs_date, high_f, low_f, is_preliminary


# ---------------------------------------------------------------------------
# NWS API client
# ---------------------------------------------------------------------------


class NWSClient:
    """
    Async client for fetching NWS Daily Climate Reports (CLI products).

    Usage::

        async with NWSClient() as client:
            obs = await client.get_daily_observation("NYC", date(2026, 3, 17))
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> NWSClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "User-Agent": NWS_USER_AGENT,
                    "Accept": "application/geo+json",
                },
            )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _ensure_connected(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            raise RuntimeError("Client not connected — use `async with` or call connect()")
        return self._client

    async def _request_with_retry(self, url: str) -> dict[str, Any]:
        """GET a URL with retry on transient failures (429, 500, 503)."""
        client = self._ensure_connected()
        return await get_json(  # type: ignore[no-any-return]
            client,
            url,
            error_label="NWS API",
            retryable_statuses={429, 500, 503},
            exc_type=NWSAPIError,
        )

    async def _list_cli_products(self, cli_location: str) -> list[dict[str, Any]]:
        """List recent CLI products for a station, newest first."""
        url = f"{NWS_API_BASE}/products/types/CLI/locations/{cli_location}"
        data = await self._request_with_retry(url)
        return data.get("@graph", [])  # type: ignore[no-any-return]

    async def _fetch_product_text(self, product_id: str) -> tuple[str, str]:
        """Fetch a CLI product's text and issuance time by its UUID."""
        url = f"{NWS_API_BASE}/products/{product_id}"
        data = await self._request_with_retry(url)
        return data.get("productText", ""), data.get("issuanceTime", "")

    async def get_daily_observation(
        self,
        city: str,
        target_date: date,
    ) -> CliObservation | None:
        """
        Fetch the NWS CLI observation for a city and date.

        Scans recent CLI products for the station and returns the one whose
        parsed date matches target_date. Prefers the latest (most complete)
        product for that date.

        Args:
            city: Internal city code (NYC, CHI, MIA, LAX, DEN, AUS, SEA).
            target_date: The date to look up.

        Returns:
            CliObservation if found, None if no matching CLI product exists
            (e.g. the date is too old or too new).
        """
        city = city.upper()
        cli_location = CITY_TO_CLI_LOCATION.get(city)
        if cli_location is None:
            raise KeyError(
                f"Unknown city code '{city}'. "
                f"Supported: {', '.join(sorted(CITY_TO_CLI_LOCATION))}"
            )

        logger.info("Fetching CLI products for %s (NWS location: %s)", city, cli_location)

        products = await self._list_cli_products(cli_location)
        if not products:
            logger.warning("No CLI products found for %s", cli_location)
            return None

        # Walk products newest-first, collect every match for the target date,
        # then return the latest-issued **final** (non-preliminary) product if
        # any; otherwise the latest-issued preliminary. NWS issues multiple CLI
        # products per observation date — morning partial, afternoon partial,
        # next-morning final — and later issuances always subsume earlier ones.
        # Early-terminate when the product's issuance date is too old.
        cutoff = target_date - timedelta(days=2)

        candidates: list[CliObservation] = []

        for product in products:
            product_id = product.get("id", "")
            if not product_id:
                continue

            # Early termination: if the product issuance is before the cutoff,
            # all remaining products are older (list is newest-first).
            issuance_raw = product.get("issuanceTime", "")
            if issuance_raw:
                try:
                    issued_dt = datetime.fromisoformat(issuance_raw)
                    if issued_dt.date() < cutoff:
                        break
                except (ValueError, TypeError):
                    pass

            text, issued_at = await self._fetch_product_text(product_id)
            obs_date, high_f, low_f, is_preliminary = parse_cli_text(text)

            if obs_date == target_date:
                candidates.append(
                    CliObservation(
                        city=city,
                        cli_location=cli_location,
                        observation_date=obs_date,
                        high_f=high_f,
                        low_f=low_f,
                        product_id=product_id,
                        issued_at=issued_at,
                        is_preliminary=is_preliminary,
                    )
                )

        if not candidates:
            logger.info("No CLI product found for %s on %s", city, target_date)
            return None

        # Prefer a non-preliminary record; within each class, latest issuance
        # wins. Fall back to preliminary if no final exists yet.
        finals = [c for c in candidates if not c.is_preliminary]
        pool = finals or candidates
        # Sort key: (issued_at, product_id) — deterministic tiebreak on
        # product_id in the rare case two records share an issuance timestamp.
        pool.sort(key=lambda o: (o.issued_at, o.product_id))
        chosen = pool[-1]

        logger.info(
            "CLI for %s on %s: high=%s low=%s (product=%s, preliminary=%s, "
            "issued=%s; selected from %d candidates)",
            city, target_date, chosen.high_f, chosen.low_f,
            chosen.product_id, chosen.is_preliminary, chosen.issued_at,
            len(candidates),
        )
        return chosen

    async def get_recent_observations(
        self,
        city: str,
        max_days: int = 3,
    ) -> list[CliObservation]:
        """
        Fetch CLI observations for the most recent days available.

        Walks through recent CLI products and returns one observation per
        unique date, up to max_days dates.

        Args:
            city: Internal city code.
            max_days: Maximum number of distinct dates to return.

        Returns:
            List of CliObservation, newest first. May be fewer than max_days
            if fewer CLIs are available.
        """
        city = city.upper()
        cli_location = CITY_TO_CLI_LOCATION.get(city)
        if cli_location is None:
            raise KeyError(
                f"Unknown city code '{city}'. "
                f"Supported: {', '.join(sorted(CITY_TO_CLI_LOCATION))}"
            )

        products = await self._list_cli_products(cli_location)
        if not products:
            return []

        observations: list[CliObservation] = []
        seen_dates: set[date] = set()

        for product in products:
            if len(seen_dates) >= max_days:
                break

            product_id = product.get("id", "")
            if not product_id:
                continue

            text, issued_at = await self._fetch_product_text(product_id)
            obs_date, high_f, low_f, is_preliminary = parse_cli_text(text)

            if obs_date is None or obs_date in seen_dates:
                continue

            seen_dates.add(obs_date)
            observations.append(
                CliObservation(
                    city=city,
                    cli_location=cli_location,
                    observation_date=obs_date,
                    high_f=high_f,
                    low_f=low_f,
                    product_id=product_id,
                    issued_at=issued_at,
                    is_preliminary=is_preliminary,
                )
            )

        logger.info(
            "Fetched %d CLI observations for %s (dates: %s)",
            len(observations),
            city,
            [o.observation_date.isoformat() for o in observations],
        )
        return observations


class NWSAPIError(Exception):
    """Raised when the NWS API request fails or returns unexpected data."""
