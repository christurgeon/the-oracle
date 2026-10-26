"""Tests for the shared ticker-date parser."""

from __future__ import annotations

from datetime import date

from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date


class TestParseTickerDate:
    def test_market_ticker(self) -> None:
        assert parse_ticker_date("KXHIGHNY-26APR14-B85.5") == date(2026, 4, 14)

    def test_event_ticker_fallback(self) -> None:
        assert parse_ticker_date("KXHIGHNY-26MAR14") == date(2026, 3, 14)

    def test_spx_ticker_with_hour_suffix_unparseable(self) -> None:
        # KXINX tickers embed a resolution hour after the date (-26MAR20H1600)
        # and don't match either regex — broker treats these as "date unknown."
        assert parse_ticker_date("KXINX-26MAR20H1600") is None

    def test_invalid_month(self) -> None:
        assert parse_ticker_date("KXHIGHNY-26XYZ14-B85.5") is None

    def test_invalid_calendar_day(self) -> None:
        # Feb 30 — regex matches, date() raises ValueError, we swallow
        assert parse_ticker_date("KX-26FEB30-T0") is None

    def test_garbage(self) -> None:
        assert parse_ticker_date("garbage") is None

    def test_empty(self) -> None:
        assert parse_ticker_date("") is None

    def test_lowercase_month(self) -> None:
        # Regex is case-insensitive per design
        assert parse_ticker_date("KXHIGHNY-26apr14-B85.5") == date(2026, 4, 14)

    # ------------------------------------------------------------------
    # Intraday format (v1.3 §4.3 — KXHIGH<CITY>-<YYMMDD>H<HHMM>[-B<strike>])
    # ------------------------------------------------------------------
    def test_intraday_market_ticker_with_strike(self) -> None:
        assert parse_ticker_date("KXHIGHMIA-26MAY15H1700-B85") == date(2026, 5, 15)

    def test_intraday_event_ticker_no_strike(self) -> None:
        # No "-B<strike>" suffix — bare event-level intraday ticker.
        assert parse_ticker_date("KXHIGHMIA-26MAY15H1700") == date(2026, 5, 15)

    def test_intraday_lax_matches_legacy_date(self) -> None:
        # The same forecast date is recovered whether the ticker carries
        # the intraday H<HHMM> suffix or not — the resolution hour is
        # NOT part of the forecast date.
        assert parse_ticker_date("KXHIGHLAX-26MAY15H1300-B72") == parse_ticker_date(
            "KXHIGHLAX-26MAY15-B72"
        )

    def test_intraday_lowercase_month(self) -> None:
        assert parse_ticker_date("KXHIGHMIA-26may15h1700-B85") == date(2026, 5, 15)


class TestParseTickerCityVar:
    # ------------------------------------------------------------------
    # Legacy daily KXHIGH (high temp)
    # ------------------------------------------------------------------
    def test_legacy_daily_kxhigh_mia(self) -> None:
        assert parse_ticker_city_var("KXHIGHMIA-26MAY15") == ("MIA", "high")

    def test_legacy_daily_kxhigh_mia_with_strike(self) -> None:
        assert parse_ticker_city_var("KXHIGHMIA-26MAY15-B85") == ("MIA", "high")

    def test_legacy_daily_kxhigh_ny_maps_to_nyc(self) -> None:
        # KXHIGHNY uses 'NY' as the suffix; the canonical city code is 'NYC'.
        assert parse_ticker_city_var("KXHIGHNY-26MAR14-B85.5") == ("NYC", "high")

    def test_legacy_daily_kxhigh_lax(self) -> None:
        assert parse_ticker_city_var("KXHIGHLAX-26APR14-B72") == ("LAX", "high")

    def test_legacy_daily_kxhight_sea_quirk(self) -> None:
        # Kalshi series 'KXHIGHTSEA' (extra T) — same mapping as KXHIGH.
        assert parse_ticker_city_var("KXHIGHTSEA-26MAY15-B72") == ("SEA", "high")

    def test_legacy_daily_kxhight_bos_quirk(self) -> None:
        assert parse_ticker_city_var("KXHIGHTBOS-26MAY15-B72") == ("BOS", "high")

    def test_legacy_daily_kxhigh_chi(self) -> None:
        assert parse_ticker_city_var("KXHIGHCHI-26MAY15-B85") == ("CHI", "high")

    def test_legacy_daily_kxhigh_den(self) -> None:
        assert parse_ticker_city_var("KXHIGHDEN-26MAY15-B85") == ("DEN", "high")

    def test_legacy_daily_kxhigh_aus(self) -> None:
        assert parse_ticker_city_var("KXHIGHAUS-26MAY15-B85") == ("AUS", "high")

    def test_legacy_daily_kxhigh_phil(self) -> None:
        assert parse_ticker_city_var("KXHIGHPHIL-26MAY15-B85") == ("PHIL", "high")

    # ------------------------------------------------------------------
    # Legacy daily KXLOWT (low temp)
    # ------------------------------------------------------------------
    def test_legacy_daily_kxlowt_nyc(self) -> None:
        assert parse_ticker_city_var("KXLOWTNYC-26MAY15") == ("NYC", "low")

    def test_legacy_daily_kxlowt_mia_with_strike(self) -> None:
        assert parse_ticker_city_var("KXLOWTMIA-26MAY15-B70") == ("MIA", "low")

    def test_legacy_daily_kxlowt_lax(self) -> None:
        assert parse_ticker_city_var("KXLOWTLAX-26MAY15-B58") == ("LAX", "low")

    def test_legacy_daily_kxlowt_bos(self) -> None:
        assert parse_ticker_city_var("KXLOWTBOS-26MAY15-B58") == ("BOS", "low")

    def test_legacy_daily_kxlowt_chi(self) -> None:
        assert parse_ticker_city_var("KXLOWTCHI-26MAY15-B58") == ("CHI", "low")

    def test_legacy_daily_kxlowt_den(self) -> None:
        assert parse_ticker_city_var("KXLOWTDEN-26MAY15-B58") == ("DEN", "low")

    def test_legacy_daily_kxlowt_aus(self) -> None:
        assert parse_ticker_city_var("KXLOWTAUS-26MAY15-B58") == ("AUS", "low")

    def test_legacy_daily_kxlowt_phil(self) -> None:
        assert parse_ticker_city_var("KXLOWTPHIL-26MAY15-B58") == ("PHIL", "low")

    # ------------------------------------------------------------------
    # Intraday KXHIGH (the primary v1.3 §9.2(c) MINOR #2 path)
    # ------------------------------------------------------------------
    def test_intraday_kxhigh_mia_event_ticker(self) -> None:
        assert parse_ticker_city_var("KXHIGHMIA-26MAY15H1700") == ("MIA", "high")

    def test_intraday_kxhigh_mia_market_ticker(self) -> None:
        assert parse_ticker_city_var("KXHIGHMIA-26MAY15H1700-B85") == ("MIA", "high")

    def test_intraday_kxhigh_lax_market_ticker(self) -> None:
        assert parse_ticker_city_var("KXHIGHLAX-26MAY15H1300-B72") == ("LAX", "high")

    def test_intraday_kxhight_sea_quirk(self) -> None:
        # The KXHIGHT prefix quirk also applies to the intraday format.
        assert parse_ticker_city_var("KXHIGHTSEA-26MAY15H1700-B72") == ("SEA", "high")

    # ------------------------------------------------------------------
    # Negative / edge cases
    # ------------------------------------------------------------------
    def test_empty_string(self) -> None:
        assert parse_ticker_city_var("") is None

    def test_garbage(self) -> None:
        assert parse_ticker_city_var("garbage") is None

    def test_unrelated_series_kxinx(self) -> None:
        # SPX brackets — neither high nor low temp; helper returns None.
        assert parse_ticker_city_var("KXINX-26MAR20H1600") is None

    def test_unknown_city_code(self) -> None:
        # KXHIGH series prefix but city suffix not in the known map.
        assert parse_ticker_city_var("KXHIGHZZZ-26MAY15-B85") is None

    def test_missing_date_suffix(self) -> None:
        # Series prefix recognised, but no date — helper returns None.
        assert parse_ticker_city_var("KXHIGHMIA") is None

    def test_malformed_prefix(self) -> None:
        # Looks like a Kalshi ticker but does not start with KXHIGH/KXLOWT.
        assert parse_ticker_city_var("KXFOOMIA-26MAY15-B85") is None

    # ------------------------------------------------------------------
    # Round-trip: derive (city, var) from a known intraday fixture
    # ------------------------------------------------------------------
    def test_roundtrip_intraday_fixture(self) -> None:
        # The §9.2(c) MINOR #2 ticker-derivation path: given the
        # settlement-row market_ticker, the daily-report helper must
        # recover (city, var) = "MIA/high".
        ticker = "KXHIGHMIA-26MAY14H1600-B85"
        city, var = parse_ticker_city_var(ticker)  # type: ignore[misc]
        assert (city, var) == ("MIA", "high")
        assert f"{city}/{var}" == "MIA/high"

    def test_roundtrip_intraday_lax_fixture(self) -> None:
        ticker = "KXHIGHLAX-26MAY15H1300-B72"
        city, var = parse_ticker_city_var(ticker)  # type: ignore[misc]
        assert f"{city}/{var}" == "LAX/high"

    def test_lowercase_input(self) -> None:
        # Case-insensitive — mirror parse_ticker_date's design.
        assert parse_ticker_city_var("kxhighmia-26may15h1700-b85") == ("MIA", "high")
