"""Unit tests for ``python/data/kxrain_markets.py``.

Station-map invariants are asserted here only — the module itself has no
import-time assertion, since it loads at orchestrator top level."""

from __future__ import annotations

import importlib
import logging
import sys
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data import kxrain_markets  # noqa: E402
from python.data.kxrain_markets import (  # noqa: E402
    KXRAIN_CITY_STATION,
    KXRAIN_CITY_STD_OFFSET_HOURS,
    KXRAIN_CITY_TZ,
    KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE,
    KXRAIN_LEGACY_CITY_ALIASES,
    KXRAIN_PREREGISTERED_CITIES,
    in_entry_window,
    lst_day,
    lst_local_time,
    parse_kxrain_ticker,
    station_from_rules,
    station_mismatch,
    validate_station_map,
)
from python.data.nws import CITY_TO_CLI_LOCATION  # noqa: E402

WINDOW_START = time(10, 0)
WINDOW_END = time(18, 0)


# Station / offset map invariants


def test_station_map_is_clean():
    assert validate_station_map() == []


PREREGISTERED_27 = {
    "ATL", "AUS", "BOS", "CHI", "CLL", "CMH", "DAL", "DC", "DEN", "EWR",
    "HOU", "LAX", "LEX", "LV", "MIA", "MIN", "MKE", "NOLA", "NYC", "OKC",
    "PHIL", "PHX", "PVD", "SATX", "SEA", "SFO", "TTN",
}


def test_map_covers_the_live_city_set():
    assert set(KXRAIN_CITY_STATION) == PREREGISTERED_27 | {"PIT"}
    assert set(KXRAIN_CITY_STD_OFFSET_HOURS) == set(KXRAIN_CITY_STATION)


def test_the_gate_universe_stays_the_27_cities_live_at_enable():
    # PIT launched 2026-09-23, after the rule was pre-registered; it is logged, not gated.
    assert KXRAIN_PREREGISTERED_CITIES == frozenset(PREREGISTERED_27)
    assert KXRAIN_PREREGISTERED_CITIES <= set(KXRAIN_CITY_STATION)


def test_pit_parses_to_its_cli_station_on_eastern_standard_time():
    key = parse_kxrain_ticker("KXRAIN-26SEP24-PIT")
    assert key is not None
    assert (key.city, key.station, key.lst_offset_hours) == ("PIT", "PIT", -5)
    assert key.settlement_date == date(2026, 9, 24)
    assert station_from_rules(
        "If the total precipitation at CLIPIT in Pittsburgh in Sep 24, 2026 is strictly "
        "greater than 0 inches, then the market resolves to Yes."
    ) == "PIT"


def test_chi_settles_on_ord_not_the_temperature_stations_mdw():
    assert KXRAIN_CITY_STATION["CHI"] == "ORD"
    assert CITY_TO_CLI_LOCATION["CHI"] == "MDW"
    assert KXRAIN_CITY_STATION["CHI"] != CITY_TO_CLI_LOCATION["CHI"]
    assert "CHI" in KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE


def test_shared_cities_other_than_chi_match_the_temperature_map():
    shared = set(KXRAIN_CITY_STATION) & set(CITY_TO_CLI_LOCATION)
    assert len(shared) == 20
    for city in sorted(shared - KXRAIN_EXPECTED_TEMP_MAP_DIVERGENCE):
        assert KXRAIN_CITY_STATION[city] == CITY_TO_CLI_LOCATION[city], city


def test_rain_only_cities_have_no_temperature_map_entry():
    for city in ("EWR", "TTN", "CLL", "CMH", "LEX", "MKE", "PVD", "PIT"):
        assert city in KXRAIN_CITY_STATION
        assert city not in CITY_TO_CLI_LOCATION


def test_offsets_are_standard_time_only():
    assert set(KXRAIN_CITY_STD_OFFSET_HOURS.values()) <= {-5, -6, -7, -8}


def test_stations_are_unique():
    stations = list(KXRAIN_CITY_STATION.values())
    assert len(set(stations)) == len(stations)


def test_validate_station_map_reports_a_seeded_mismatch(monkeypatch):
    monkeypatch.setitem(KXRAIN_CITY_STATION, "NYC", "JFK")
    problems = validate_station_map()
    assert any("NYC" in p for p in problems)


def test_validate_station_map_flags_a_duplicate_station(monkeypatch):
    monkeypatch.setitem(KXRAIN_CITY_STATION, "EWR", "NYC")
    assert any("1:1" in p for p in validate_station_map())


def test_module_imports_without_raising_on_a_bad_map(monkeypatch, caplog):
    monkeypatch.setattr(kxrain_markets, "KXRAIN_CITY_STATION", {"NYC": "JFK"})
    with caplog.at_level(logging.WARNING):
        kxrain_markets._log_station_map_validation()
    assert any("KXRAIN station map" in r.message for r in caplog.records)


def test_reimport_is_clean():
    importlib.reload(kxrain_markets)


# Ticker parsing


def test_parse_every_city():
    for city, station in KXRAIN_CITY_STATION.items():
        key = parse_kxrain_ticker(f"KXRAIN-26SEP19-{city}")
        assert key is not None, city
        assert key.city == city
        assert key.station == station
        assert key.settlement_date == date(2026, 9, 19)
        assert key.lst_offset_hours == KXRAIN_CITY_STD_OFFSET_HOURS[city]


def test_parse_is_case_insensitive():
    key = parse_kxrain_ticker("kxrain-26sep19-nyc")
    assert key is not None
    assert key.city == "NYC"
    assert key.station == "NYC"


@pytest.mark.parametrize(
    "ticker",
    [
        "",
        "KXRAIN-26SEP19",  # event ticker, no city segment
        "KXRAIN-26SEP19-ZZZ",  # unknown city
        "KXRAINWKND-26SEP19-NYC",  # sibling series (measured dead)
        "KXHIGHNY-26SEP19-B85.5",  # temperature series
        "KXRAIN-NYC",  # no date segment
        "KXRAIN-26XXX19-NYC",  # bad month
        "KXRAIN-26SEP99-NYC",  # impossible day
        "KXRAIN-26SEP19-NYC-EXTRA",  # unexpected fourth segment
    ],
)
def test_parse_rejects(ticker):
    assert parse_kxrain_ticker(ticker) is None


# LST window and day bucketing


def _utc_for_local(city: str, day: date, hh: int, mm: int = 0) -> datetime:
    off = KXRAIN_CITY_STD_OFFSET_HOURS[city]
    local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone(timedelta(hours=off)))
    return local.astimezone(UTC)


@pytest.mark.parametrize("city", ["NYC", "CHI", "DEN", "LAX"])
@pytest.mark.parametrize(
    ("hh", "mm", "expected"),
    [(9, 59, False), (10, 0, True), (17, 59, True), (18, 0, False), (23, 0, False)],
)
def test_window_edges_per_offset_class(city, hh, mm, expected):
    now = _utc_for_local(city, date(2026, 9, 19), hh, mm)
    assert in_entry_window(city, now, WINDOW_START, WINDOW_END) is expected


def test_lax_window_end_is_next_utc_day():
    now = _utc_for_local("LAX", date(2026, 9, 19), 18, 0)
    assert now == datetime(2026, 9, 20, 2, 0, tzinfo=UTC)
    assert in_entry_window("LAX", now, WINDOW_START, WINDOW_END) is False
    assert in_entry_window("LAX", now - timedelta(minutes=1), WINDOW_START, WINDOW_END) is True


@pytest.mark.parametrize("day", [date(2026, 10, 31), date(2026, 11, 1), date(2026, 11, 2)])
@pytest.mark.parametrize("city", ["NYC", "CHI", "PHX", "LAX"])
def test_window_does_not_shift_across_the_2026_11_01_dst_end(city, day):
    # A ZoneInfo implementation fails this: it would move the window by an hour.
    open_utc = _utc_for_local(city, day, 10, 0)
    assert in_entry_window(city, open_utc, WINDOW_START, WINDOW_END) is True
    assert in_entry_window(city, open_utc - timedelta(minutes=1), WINDOW_START, WINDOW_END) is False

    close_utc = _utc_for_local(city, day, 18, 0)
    assert in_entry_window(city, close_utc, WINDOW_START, WINDOW_END) is False
    assert in_entry_window(city, close_utc - timedelta(minutes=1), WINDOW_START, WINDOW_END) is True

    assert open_utc.hour == (10 - KXRAIN_CITY_STD_OFFSET_HOURS[city]) % 24


def test_dst_end_window_is_identical_to_a_summer_day():
    for city in KXRAIN_CITY_STATION:
        sep = _utc_for_local(city, date(2026, 9, 19), 10)
        nov = _utc_for_local(city, date(2026, 11, 2), 10)
        assert sep.hour == nov.hour, city


def test_lst_day_across_utc_midnight_for_a_utc_minus_8_city():
    # 2026-09-20 02:00Z is still 2026-09-19 18:00 PST.
    assert lst_day("LAX", datetime(2026, 9, 20, 2, 0, tzinfo=UTC)) == date(2026, 9, 19)
    # 2026-09-20 08:00Z is 2026-09-20 00:00 PST — the new climate day.
    assert lst_day("LAX", datetime(2026, 9, 20, 8, 0, tzinfo=UTC)) == date(2026, 9, 20)
    assert lst_day("NYC", datetime(2026, 9, 20, 2, 0, tzinfo=UTC)) == date(2026, 9, 19)


def test_lst_local_time():
    assert lst_local_time("NYC", datetime(2026, 9, 19, 19, 30, tzinfo=UTC)) == time(14, 30)
    assert lst_local_time("PHX", datetime(2026, 9, 19, 19, 30, tzinfo=UTC)) == time(12, 30)


def test_helpers_reject_naive_datetimes():
    with pytest.raises(ValueError):
        lst_day("NYC", datetime(2026, 9, 19, 12, 0))


def test_helpers_reject_unknown_city():
    with pytest.raises(KeyError):
        lst_day("ZZZ", datetime(2026, 9, 19, 12, 0, tzinfo=UTC))


def test_city_tz_is_fixed_offset_never_zoneinfo():
    for city, tz in KXRAIN_CITY_TZ.items():
        assert isinstance(tz, timezone), city
        summer = datetime(2026, 7, 1, 12, tzinfo=UTC)
        winter = datetime(2026, 12, 1, 12, tzinfo=UTC)
        assert tz.utcoffset(summer) == tz.utcoffset(winter), city


# Corpus identity: station-keyed, with date-bounded suffix aliases


def test_corpus_key_is_station_and_date_not_city():
    key = parse_kxrain_ticker("KXRAIN-26SEP19-CHI")
    assert key is not None
    assert key.corpus_key == ("ORD", date(2026, 9, 19))


def test_renamed_city_joins_to_the_same_corpus_key_as_its_successor():
    legacy = parse_kxrain_ticker("KXRAIN-26SEP16-PROV")
    current = parse_kxrain_ticker("KXRAIN-26SEP16-PVD")
    assert legacy is not None and current is not None
    assert legacy.corpus_key == current.corpus_key == ("PVD", date(2026, 9, 16))
    assert legacy.ticker_city_suffix == "PROV"
    assert current.ticker_city_suffix == "PVD"


@pytest.mark.parametrize(("suffix", "city"), [("PROV", "PVD"), ("MIL", "MKE"), ("COL", "CLL")])
def test_legacy_alias_applies_inside_its_bound(suffix, city):
    key = parse_kxrain_ticker(f"KXRAIN-26SEP16-{suffix}")
    assert key is not None
    assert (key.city, key.is_legacy_suffix) == (city, True)


@pytest.mark.parametrize("suffix", ["PROV", "MIL", "COL"])
def test_legacy_alias_does_not_apply_past_its_bound(suffix):
    # Past the bound, a recycled suffix must fail loudly, not mis-attribute.
    assert parse_kxrain_ticker(f"KXRAIN-26SEP17-{suffix}") is None
    assert parse_kxrain_ticker(f"KXRAIN-26DEC01-{suffix}") is None


def test_current_suffixes_are_not_flagged_legacy():
    for city in KXRAIN_CITY_STATION:
        key = parse_kxrain_ticker(f"KXRAIN-26SEP19-{city}")
        assert key is not None and key.is_legacy_suffix is False, city


def test_alias_targets_are_all_real_cities():
    for suffix, (city, bound) in KXRAIN_LEGACY_CITY_ALIASES.items():
        assert city in KXRAIN_CITY_STATION, suffix
        assert suffix not in KXRAIN_CITY_STATION, f"{suffix} is live; it cannot also be an alias"
        assert isinstance(bound, date)


def test_station_from_rules_handles_both_observed_spellings():
    assert station_from_rules("total precipitation at CLIORD in Chicago in Sep 19, 2026") == "ORD"
    assert station_from_rules("total precipitation at KMKE in Milwaukee in Sep 16, 2026") == "MKE"
    assert station_from_rules("no station here") is None
    assert station_from_rules("") is None


def test_station_mismatch_is_silent_when_the_contract_agrees():
    key = parse_kxrain_ticker("KXRAIN-26SEP19-CHI")
    assert key is not None
    assert station_mismatch(key, "precipitation at CLIORD in Chicago in Sep 19, 2026") is None
