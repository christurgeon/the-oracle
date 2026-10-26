"""
Tests for python/data/nws.py — NWS CLI parser and client.

Covers:
- CLI text parsing: extracts date, high, low, preliminary flag from real CLI formats.
- Handles edge cases: negative temps, records (R suffix), missing data (MM).
- Preliminary detection across product variants (AUS/DEN morning partial,
  afternoon partial, next-morning final, and remarks-block false positives).
- ``get_daily_observation`` prefers the latest-issued product.
- City-to-CLI-location mapping completeness.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import patch

from python.data.nws import (
    CITY_TO_CLI_LOCATION,
    NWSClient,
    parse_cli_text,
)

# ---------------------------------------------------------------------------
# Sample CLI text fragments (based on real NWS products)
# ---------------------------------------------------------------------------

CLI_CHICAGO_ORD = """
CLIMATE REPORT
NATIONAL WEATHER SERVICE CHICAGO IL
440 PM CDT TUE MAR 17 2026

...THE CHICAGO-OHARE CLIMATE SUMMARY FOR MARCH 17 2026...
VALID TODAY AS OF 0400 PM LOCAL TIME.

CLIMATE NORMAL PERIOD: 1991 TO 2020
CLIMATE RECORD PERIOD: 1871 TO 2026


WEATHER ITEM   OBSERVED TIME   RECORD YEAR NORMAL DEPARTURE LAST
                VALUE   (LST)  VALUE       VALUE  FROM      YEAR
                                                  NORMAL
...................................................................
TEMPERATURE (F)
 TODAY
  MAXIMUM         24   1:21 PM  82    2012  48    -24       53
  MINIMUM         14   7:17 AM  -1    1900  31    -17       26
  AVERAGE         19                        39    -20       40

PRECIPITATION (IN)
  TODAY            0.00          1.42 1965   0.08  -0.08     0.00
"""

CLI_NYC_CENTRAL_PARK = """
CLIMATE REPORT
NATIONAL WEATHER SERVICE NEW YORK NY
432 PM EDT TUE MAR 17 2026

...THE CENTRAL PARK NY CLIMATE SUMMARY FOR MARCH 17 2026...
VALID TODAY AS OF 0400 PM LOCAL TIME.

CLIMATE NORMAL PERIOD: 1991 TO 2020
CLIMATE RECORD PERIOD: 1869 TO 2026


WEATHER ITEM   OBSERVED TIME   RECORD YEAR NORMAL DEPARTURE LAST
                VALUE   (LST)  VALUE       VALUE  FROM      YEAR
                                                  NORMAL
...................................................................
TEMPERATURE (F)
 TODAY
  MAXIMUM         48   1207 AM  75    1945  50     -2       59
  MINIMUM         34    920 AM   9    1916  36     -2       37
  AVERAGE         41                        43     -2       48

PRECIPITATION (IN)
  TODAY            T            0.91  1936   0.12  -0.12     0.00
"""

CLI_NEGATIVE_TEMPS = """
...THE DENVER CO CLIMATE SUMMARY FOR JANUARY 15 2026...
VALID TODAY AS OF 0400 PM LOCAL TIME.

TEMPERATURE (F)
 TODAY
  MAXIMUM         -5   2:00 PM  65    2005  45    -50       30
  MINIMUM        -22   6:30 AM -25    1888  20    -42       10
  AVERAGE        -14                        33    -47       20
"""

CLI_RECORD_SET = """
...THE MIAMI CLIMATE SUMMARY FOR JULY 22 2026...
VALID TODAY AS OF 0400 PM LOCAL TIME.

TEMPERATURE (F)
 TODAY
  MAXIMUM         97R  2:30 PM  97    1942  92      5       91
  MINIMUM         82   7:00 AM  74    1972  77      5       79
  AVERAGE         90                        85      5       85
"""

CLI_MISSING_DATA = """
...THE LOS ANGELES INTL AIRPORT CA CLIMATE SUMMARY FOR MARCH 10 2026...
VALID TODAY AS OF 0400 PM LOCAL TIME.

TEMPERATURE (F)
 TODAY
  MAXIMUM         MM            85    2000  68    MM        70
  MINIMUM         MM            45    1999  52    MM        55
  AVERAGE         MM                        60    MM        63
"""

CLI_FINAL_REPORT = """
...THE CHICAGO-OHARE CLIMATE SUMMARY FOR MARCH 16 2026...
VALID FOR YESTERDAY.

TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         35   2:00 PM  82    2012  48    -13       53
  MINIMUM         20   6:00 AM  -1    1900  31    -11       26
  AVERAGE         28                        39    -11       40
"""

# Real AUS morning partial CLI (VALID AS OF 0700 AM LOCAL TIME — note no TODAY token).
# This is the product that previously leaked through the preliminary filter
# because the regex only matched "VALID TODAY".
CLI_AUS_MORNING_PARTIAL = """
CLIMATE REPORT
NATIONAL WEATHER SERVICE AUSTIN/SAN ANTONIO TX
708 AM CDT SAT APR 18 2026

...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 18 2026...
VALID AS OF 0700 AM LOCAL TIME.

CLIMATE NORMAL PERIOD: 1991 TO 2020
CLIMATE RECORD PERIOD: 1942 TO 2026


WEATHER ITEM   OBSERVED TIME   RECORD YEAR NORMAL DEPARTURE LAST
                VALUE   (LST)  VALUE       VALUE  FROM      YEAR
                                                  NORMAL
...................................................................
TEMPERATURE (F)
 TODAY
  MAXIMUM         74   6:00 AM  95    1989  79    -5        82
  MINIMUM         68   3:15 AM  40    1953  57    11        65
  AVERAGE         71                        68     3        74
"""

# Real DEN afternoon partial CLI — "VALID TODAY AS OF 0500 PM LOCAL TIME".
# This should also be preliminary; regression guard for the classic variant.
CLI_DEN_AFTERNOON_PARTIAL = """
CLIMATE REPORT
NATIONAL WEATHER SERVICE DENVER CO
501 PM MDT SAT APR 18 2026

...THE DENVER CLIMATE SUMMARY FOR APRIL 18 2026...
VALID TODAY AS OF 0500 PM LOCAL TIME.


TEMPERATURE (F)
 TODAY
  MAXIMUM         62   3:15 PM  85    2013  65    -3        70
  MINIMUM         34   5:30 AM   9    1920  38    -4        42
  AVERAGE         48                        52    -4        56
"""

# Real AUS next-morning final — "VALID FOR YESTERDAY", no "VALID AS OF" header.
CLI_AUS_FINAL = """
CLIMATE REPORT
NATIONAL WEATHER SERVICE AUSTIN/SAN ANTONIO TX
708 AM CDT SUN APR 19 2026

...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 19 2026...
VALID FOR YESTERDAY.


TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         74   2:00 PM  95    1989  79    -5        82
  MINIMUM         51   6:00 AM  40    1953  57    -6        65
  AVERAGE         62                        68    -6        74
"""

# Negative-case: a final product with the literal text "VALID AS OF" appearing
# inside a mid-document remarks/narrative block (not at the start of a line
# followed by a time digit). The regex must NOT flag this as preliminary.
CLI_FINAL_WITH_REMARKS = """
...THE MIAMI CLIMATE SUMMARY FOR APRIL 20 2026...
VALID FOR YESTERDAY.


TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         85   2:00 PM  95    2011  82     3        87
  MINIMUM         71   5:45 AM  54    1972  68     3        74
  AVERAGE         78                        75     3        80

...REMARKS...

SEA LEVEL PRESSURE DATA VALID AS OF A PREVIOUS REVIEW WAS CORRECTED
AT 1100 AM. NO OTHER CHANGES WERE MADE.
"""


# Negative-case: a CORRECTION product whose remarks section opens with a line
# starting "VALID AS OF <HHMM> <AM/PM>" at column 0 but lacks the "LOCAL TIME"
# suffix present in all real preliminary headers. Closes the false-positive
# class the reviewer flagged before heal execution.
CLI_FINAL_CORRECTION = """
...THE AUSTIN CLIMATE SUMMARY FOR APRIL 19 2026...
VALID FOR YESTERDAY.


TEMPERATURE (F)
 YESTERDAY
  MAXIMUM         76   3:45 PM  97    2011  82     3        87
  MINIMUM         51   5:30 AM  38    1999  58     3        64

...CORRECTION...

VALID AS OF 1100 AM CORRECTION: MAXIMUM WAS REPORTED AS 74, NOW 76.
NO OTHER CHANGES WERE MADE.
"""


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseCliText:
    """parse_cli_text extracts date, high, low, and preliminary flag from NWS CLI text."""

    def test_chicago_standard(self) -> None:
        obs_date, high, low, is_prelim = parse_cli_text(CLI_CHICAGO_ORD)
        assert obs_date == date(2026, 3, 17)
        assert high == 24
        assert low == 14
        assert is_prelim is True

    def test_nyc_standard(self) -> None:
        obs_date, high, low, is_prelim = parse_cli_text(CLI_NYC_CENTRAL_PARK)
        assert obs_date == date(2026, 3, 17)
        assert high == 48
        assert low == 34
        assert is_prelim is True

    def test_negative_temperatures(self) -> None:
        obs_date, high, low, is_prelim = parse_cli_text(CLI_NEGATIVE_TEMPS)
        assert obs_date == date(2026, 1, 15)
        assert high == -5
        assert low == -22

    def test_record_suffix_stripped(self) -> None:
        """The 'R' suffix (record tied/set) should not affect the parsed value."""
        obs_date, high, low, _ = parse_cli_text(CLI_RECORD_SET)
        assert obs_date == date(2026, 7, 22)
        assert high == 97
        assert low == 82

    def test_missing_data_returns_none(self) -> None:
        """When temp is 'MM' (missing), parser should return None."""
        obs_date, high, low, _ = parse_cli_text(CLI_MISSING_DATA)
        assert obs_date == date(2026, 3, 10)
        assert high is None
        assert low is None

    def test_empty_text_returns_nones(self) -> None:
        obs_date, high, low, is_prelim = parse_cli_text("")
        assert obs_date is None
        assert high is None
        assert low is None
        assert is_prelim is False

    def test_no_temperature_section(self) -> None:
        # Real CLI preliminary headers are always the full form
        # ``VALID TODAY AS OF HHMM <AM|PM> LOCAL TIME`` — never bare
        # ``VALID TODAY``. The updated regex requires the full form.
        text = (
            "...THE CHICAGO-OHARE CLIMATE SUMMARY FOR MARCH 17 2026..."
            "\nVALID TODAY AS OF 0400 PM LOCAL TIME.\nNo temp data here."
        )
        obs_date, high, low, is_prelim = parse_cli_text(text)
        assert obs_date == date(2026, 3, 17)
        assert high is None
        assert low is None
        assert is_prelim is True

    def test_final_report_not_preliminary(self) -> None:
        """A next-morning CLI with 'VALID FOR YESTERDAY' should not be preliminary."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_FINAL_REPORT)
        assert obs_date == date(2026, 3, 16)
        assert high == 35
        assert low == 20
        assert is_prelim is False

    def test_parse_cli_morning_partial_is_preliminary(self) -> None:
        """AUS morning partial: ``VALID AS OF 0700 AM LOCAL TIME`` (no TODAY)
        must be flagged preliminary. This is the bug this patch fixes: the
        old regex only matched 'VALID TODAY', so this leaked through as a
        canonical final settlement and clobbered the next-morning final."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_AUS_MORNING_PARTIAL)
        assert obs_date == date(2026, 4, 18)
        assert high == 74
        assert low == 68
        assert is_prelim is True

    def test_parse_cli_afternoon_partial_is_preliminary(self) -> None:
        """Regression: the classic ``VALID TODAY AS OF 0500 PM LOCAL TIME``
        form must still be flagged preliminary."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_DEN_AFTERNOON_PARTIAL)
        assert obs_date == date(2026, 4, 18)
        assert high == 62
        assert low == 34
        assert is_prelim is True

    def test_parse_cli_final_is_not_preliminary(self) -> None:
        """Next-morning ``VALID FOR YESTERDAY`` final (no ``VALID AS OF``
        header) must not be flagged preliminary."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_AUS_FINAL)
        assert obs_date == date(2026, 4, 19)
        assert high == 74
        assert low == 51
        assert is_prelim is False

    def test_parse_cli_valid_as_of_in_remarks_is_not_preliminary(self) -> None:
        """False-positive guard: 'VALID AS OF' inside a mid-document remarks
        paragraph (not at start of line with a time digit) must NOT be flagged
        preliminary. The ``^...\\s+\\d`` anchor with MULTILINE enforces this."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_FINAL_WITH_REMARKS)
        assert obs_date == date(2026, 4, 20)
        assert high == 85
        assert low == 71
        assert is_prelim is False

    def test_parse_cli_correction_remark_is_not_preliminary(self) -> None:
        """Correction products can open a remarks line with 'VALID AS OF
        <HHMM> <AM/PM> CORRECTION:...' at column 0 without the 'LOCAL TIME'
        suffix. The tightened pattern requires the full header form and must
        NOT flag this as preliminary."""
        obs_date, high, low, is_prelim = parse_cli_text(CLI_FINAL_CORRECTION)
        assert obs_date == date(2026, 4, 19)
        assert high == 76
        assert low == 51
        assert is_prelim is False


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------


class TestGetDailyObservation:
    """``get_daily_observation`` prefers the latest-issued product for a date."""

    async def test_get_daily_observation_prefers_latest_issuance(self) -> None:
        """Two final products exist for the same date with different
        ``issuanceTime`` values. The later-issued one must win."""

        target = date(2026, 4, 18)

        products = [
            # Newest-first ordering, as returned by the NWS list endpoint.
            {"id": "final-revised", "issuanceTime": "2026-04-19T14:00:00+00:00"},
            {"id": "final-original", "issuanceTime": "2026-04-19T07:07:00+00:00"},
        ]

        product_texts: dict[str, tuple[str, str]] = {
            "final-revised": (
                "...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 18 2026...\n"
                "VALID FOR YESTERDAY.\n"
                "TEMPERATURE (F)\n YESTERDAY\n"
                "  MAXIMUM         75   2:00 PM  95    1989  79    -4        82\n"
                "  MINIMUM         52   6:00 AM  40    1953  57    -5        65\n",
                "2026-04-19T14:00:00+00:00",
            ),
            "final-original": (
                "...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 18 2026...\n"
                "VALID FOR YESTERDAY.\n"
                "TEMPERATURE (F)\n YESTERDAY\n"
                "  MAXIMUM         74   2:00 PM  95    1989  79    -5        82\n"
                "  MINIMUM         51   6:00 AM  40    1953  57    -6        65\n",
                "2026-04-19T07:07:00+00:00",
            ),
        }

        async def fake_fetch(pid: str) -> tuple[str, str]:
            return product_texts[pid]

        async def fake_list(_loc: str) -> list[dict[str, Any]]:
            return products

        client = NWSClient()
        with (
            patch.object(client, "_list_cli_products", side_effect=fake_list) as list_mock,
            patch.object(client, "_fetch_product_text", side_effect=fake_fetch) as fetch_mock,
        ):
            obs = await client.get_daily_observation("AUS", target)

        assert list_mock.await_count == 1
        # Both products should have been fetched before selection.
        assert fetch_mock.await_count == 2
        assert obs is not None
        assert obs.observation_date == target
        assert obs.product_id == "final-revised"
        assert obs.high_f == 75
        assert obs.low_f == 52
        assert obs.is_preliminary is False

    async def test_get_daily_observation_falls_back_to_preliminary(self) -> None:
        """If only preliminary products exist for the target date (e.g. the
        final hasn't been issued yet), the latest preliminary is returned."""

        target = date(2026, 4, 18)

        products = [
            {"id": "afternoon", "issuanceTime": "2026-04-18T21:00:00+00:00"},
            {"id": "morning", "issuanceTime": "2026-04-18T12:00:00+00:00"},
        ]

        product_texts: dict[str, tuple[str, str]] = {
            "afternoon": (
                "...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 18 2026...\n"
                "VALID TODAY AS OF 0400 PM LOCAL TIME.\n"
                "TEMPERATURE (F)\n TODAY\n"
                "  MAXIMUM         80   3:00 PM  95    1989  79     1        82\n"
                "  MINIMUM         60   5:00 AM  40    1953  57     3        65\n",
                "2026-04-18T21:00:00+00:00",
            ),
            "morning": (
                "...THE AUSTIN-BERGSTROM CLIMATE SUMMARY FOR APRIL 18 2026...\n"
                "VALID AS OF 0700 AM LOCAL TIME.\n"
                "TEMPERATURE (F)\n TODAY\n"
                "  MAXIMUM         72   6:00 AM  95    1989  79    -7        82\n"
                "  MINIMUM         65   3:00 AM  40    1953  57     8        65\n",
                "2026-04-18T12:00:00+00:00",
            ),
        }

        async def fake_list(_loc: str) -> list[dict[str, Any]]:
            return products

        async def fake_fetch(pid: str) -> tuple[str, str]:
            return product_texts[pid]

        client = NWSClient()
        with (
            patch.object(client, "_list_cli_products", side_effect=fake_list),
            patch.object(client, "_fetch_product_text", side_effect=fake_fetch),
        ):
            obs = await client.get_daily_observation("AUS", target)

        # Should prefer the latest preliminary when no finals exist.
        assert obs is not None
        assert obs.is_preliminary is True
        assert obs.product_id == "afternoon"
        assert obs.high_f == 80


# ---------------------------------------------------------------------------
# Mapping tests
# ---------------------------------------------------------------------------


class TestCityMapping:
    """CITY_TO_CLI_LOCATION covers all supported cities."""

    def test_all_weather_cities_have_cli_mapping(self) -> None:
        expected_cities = {
            "NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "PHIL", "BOS",
            # New cities (collection-only, added 2026-06-09).
            "ATL", "DAL", "HOU", "LV", "MIN", "NOLA", "OKC", "PHX", "SATX", "SFO", "DC",
        }
        assert set(CITY_TO_CLI_LOCATION.keys()) == expected_cities

    def test_known_cli_codes(self) -> None:
        assert CITY_TO_CLI_LOCATION["CHI"] == "MDW"
        assert CITY_TO_CLI_LOCATION["NYC"] == "NYC"
        assert CITY_TO_CLI_LOCATION["DEN"] == "DEN"
        assert CITY_TO_CLI_LOCATION["PHIL"] == "PHL"
        assert CITY_TO_CLI_LOCATION["BOS"] == "BOS"
