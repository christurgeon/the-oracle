"""Guard tests pinning the live API shape of the KXRAIN fixtures.

The fixture values are synthetic. Keep the keys, string typing and ordering
of the live payloads; never derive them from ``data/snapshots/``
(differently shaped)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_markets import (  # noqa: E402
    KXRAIN_CITY_STATION,
    parse_kxrain_ticker,
    station_from_rules,
    station_mismatch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "kxrain"


def _load(name: str):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not captured; run scripts/capture_kxrain_fixture.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def markets():
    return _load("live_markets.json")["markets"]


@pytest.fixture(scope="module")
def orderbook():
    return _load("live_orderbook.json")


@pytest.fixture(scope="module")
def trades():
    return _load("live_trades.json")["trades"]


@pytest.fixture(scope="module")
def settled():
    return _load("live_settled_market.json")


# /markets: live dicts are string-typed, int-striked, and have no `city` key


def test_markets_prices_are_strings_not_floats(markets):
    for m in markets:
        for field in ("yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars"):
            assert isinstance(m[field], str), f"{m['ticker']}.{field} is {type(m[field])}"


def test_markets_have_no_city_key(markets):
    assert all("city" not in m for m in markets)


def test_markets_strikes_are_ints(markets):
    assert all(isinstance(m["floor_strike"], int) for m in markets)


def test_markets_size_and_volume_are_strings(markets):
    for m in markets:
        assert isinstance(m["volume_fp"], str)
        assert isinstance(m["open_interest_fp"], str)
        assert isinstance(m["yes_bid_size_fp"], str)


def test_open_markets_carry_an_empty_expiration_value(markets):
    for m in markets:
        assert isinstance(m["expiration_value"], str)
        assert m["expiration_value"] == ""
        assert m["result"] == ""


def test_every_market_ticker_parses_to_a_known_city(markets):
    for m in markets:
        key = parse_kxrain_ticker(m["ticker"])
        assert key is not None, m["ticker"]
        assert key.city in KXRAIN_CITY_STATION


def test_markets_fit_in_one_page():
    page = _load("live_markets.json")
    assert page.get("cursor", "") == ""


def test_rules_name_the_settlement_station_and_date(markets):
    # A disagreement means Kalshi changed the settlement gauge — a kill trigger.
    for m in markets:
        key = parse_kxrain_ticker(m["ticker"])
        assert key is not None
        assert station_mismatch(key, m["rules_primary"]) is None
        assert key.settlement_date.strftime("%b %-d, %Y") in m["rules_primary"], m["ticker"]


def test_station_mismatch_detects_a_changed_settlement_station():
    key = parse_kxrain_ticker("KXRAIN-26SEP19-CHI")
    assert key is not None and key.station == "ORD"
    assert station_mismatch(key, "...precipitation at CLIMDW in Chicago...") is not None
    assert "blind" in (station_mismatch(key, "no station named here") or "")


# /markets/{ticker}/orderbook


def test_orderbook_levels_are_ascending_with_the_touch_last(orderbook):
    # Ascending: the best bid is levels[-1]; a levels[:10] slice would be the worst ten.
    book = orderbook["orderbook_fp"]
    for side in ("yes_dollars", "no_dollars"):
        prices = [float(p) for p, _ in book.get(side) or []]
        assert prices == sorted(prices), side


def test_orderbook_touch_matches_the_markets_quote(orderbook, markets):
    # Resolved against live_markets.json, not the manifest — a shape test
    # must not depend on capture metadata.
    ticker = _load("capture_manifest.json")["orderbook"]["ticker"]
    quote = next((m for m in markets if m["ticker"] == ticker), None)
    assert quote is not None, f"{ticker} not in live_markets.json"
    book_touch = float(orderbook["orderbook_fp"]["yes_dollars"][-1][0])
    assert book_touch == float(quote["yes_bid_dollars"])


def test_orderbook_shape_is_orderbook_fp_with_string_price_and_size(orderbook):
    assert set(orderbook) == {"orderbook_fp"}
    book = orderbook["orderbook_fp"]
    assert "yes_dollars" in book
    for side in ("yes_dollars", "no_dollars"):
        levels = book.get(side) or []
        for level in levels:
            assert len(level) == 2
            assert isinstance(level[0], str), f"{side} price is {type(level[0])}"
            assert isinstance(level[1], str), f"{side} size is {type(level[1])}"


# /markets/trades


def test_trades_carry_the_fields_the_eligibility_leg_needs(trades):
    assert trades
    for t in trades:
        for field in ("taker_side", "yes_price_dollars", "no_price_dollars", "count_fp",
                      "created_time", "trade_id", "ticker"):
            assert field in t, field


def test_taker_side_vocabulary_is_lowercase_yes_no(trades):
    assert {t["taker_side"] for t in trades} <= {"yes", "no"}


def test_trades_are_newest_first(trades):
    # Truncation must drop the oldest prints so the page covers the recent window.
    times = [t["created_time"] for t in trades]
    assert times == sorted(times, reverse=True)


def test_trade_prices_and_counts_are_strings(trades):
    for t in trades:
        assert isinstance(t["yes_price_dollars"], str)
        assert isinstance(t["no_price_dollars"], str)
        assert isinstance(t["count_fp"], str)


# A settled market


def test_settled_market_expiration_value_is_a_string(settled):
    assert isinstance(settled["expiration_value"], str)
    assert settled["result"] in ("yes", "no")


def test_settled_outcome_derivation_matches_the_stated_rule(settled):
    raw = settled["expiration_value"]
    parsed = 0.0 if raw in ("", "T", None) else float(raw)
    assert (parsed > 0.0) == (settled["result"] == "yes")


def test_settled_last_price_is_a_stale_trade_price_not_an_outcome(settled):
    # Not a 0/1 settlement flag; deriving outcome from it mis-scored real rows.
    price = float(settled["last_price_dollars"])
    assert 0.0 <= price <= 1.0
    assert price not in (0.0, 1.0), "a 0/1 value here would mean the field changed meaning"


def test_settled_close_time_is_standard_midnight_ending_the_settlement_date(settled):
    # Audit-only: no runtime path derives a day from close_time.
    key = parse_kxrain_ticker(settled["ticker"])
    assert key is not None
    assert settled["close_time"].endswith("Z")
    assert isinstance(key.settlement_date, date)


# Known gap, deliberately pinned rather than silently handled


@pytest.mark.parametrize(("ticker", "city"), [
    ("KXRAIN-26SEP16-PROV", "PVD"),  # Providence, relaunched as PVD
    ("KXRAIN-26SEP16-MIL", "MKE"),   # Milwaukee, relaunched as MKE
    ("KXRAIN-26SEP16-COL", "CLL"),   # College Station, relaunched as CLL
])
def test_retired_city_suffixes_resolve_within_their_date_bound(ticker, city):
    key = parse_kxrain_ticker(ticker)
    assert key is not None
    assert key.city == city
    assert key.is_legacy_suffix is True
    assert key.ticker_city_suffix == ticker.rsplit("-", 1)[-1]


def test_rules_text_of_the_legacy_cohort_uses_the_other_station_spelling():
    # Older rows say "at KPVD", newer say "at CLIPVD" for the same station.
    assert station_from_rules("...precipitation at KPVD in Providence in Sep 16, 2026...") == "PVD"
    assert station_from_rules("...precipitation at CLIPVD in Providence...") == "PVD"
