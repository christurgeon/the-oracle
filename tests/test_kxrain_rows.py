"""Unit tests for `python/data/kxrain_rows.py`.

Driven from the payloads in `tests/fixtures/kxrain/`, which keep the live API
shape (string prices, fractional sizes) — hand-written ones would use "5.00" everywhere.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_decision import SKIP_REASONS, evaluate_decision  # noqa: E402
from python.data.kxrain_markets import parse_kxrain_ticker  # noqa: E402
from python.data.kxrain_rows import (  # noqa: E402
    CAPTURE_OUTCOMES,
    FRAME_VERSION,
    MAX_UNPARSEABLE_TICKERS,
    PIPE_BUF,
    SCHEMA_VERSION,
    TICK_OUTCOMES,
    build_book_row,
    build_decision_row,
    build_oversize_stub_row,
    build_settlement_row,
    build_tick_row,
    build_trades_row,
    encode_row,
    quantise_price,
)
from python.data.kxrain_trades import (  # noqa: E402
    MAX_PRINTS,
    summarise_trade_window,
    to_dollar_price,
    to_epoch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "kxrain"
NOW = datetime(2026, 9, 18, 18, 44, 11, tzinfo=UTC)


# Fixtures


def _load(name: str):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} not captured")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def live_markets():
    return _load("live_markets.json")["markets"]


@pytest.fixture(scope="module")
def live_orderbook():
    return _load("live_orderbook.json")


@pytest.fixture(scope="module")
def live_trades():
    return _load("live_trades.json")


@pytest.fixture(scope="module")
def settled_market():
    return _load("live_settled_market.json")


@pytest.fixture(scope="module")
def mia_market(live_markets):
    """The market the order-book fixture was captured against (yes_bid 0.82)."""
    for market in live_markets:
        if market["ticker"] == "KXRAIN-26SEP18-MIA":
            return market
    pytest.skip("KXRAIN-26SEP18-MIA absent from the capture")


def _decision(market, **overrides):
    key = parse_kxrain_ticker(market["ticker"])
    assert key is not None
    yes_bid = to_dollar_price(market["yes_bid_dollars"])
    outcome = evaluate_decision(
        yes_bid=yes_bid,
        in_window=overrides.pop("in_window_eval", True),
        no_trades_30m=overrides.pop("no_trades_30m", 4),
        already_entered=overrides.pop("already_entered_eval", False),
    )
    kwargs: dict = {
        "key": key,
        "market": market,
        "outcome": outcome,
        "scan_id": "1f2e3d",
        "ts": NOW,
        "in_window": True,
        "already_entered": False,
        "virtual_entry": True,
        "pass_index": 0,
        "book_capture": "ok",
        "trades_capture": "ok",
        "follow_capture": False,
        "observer_version": "abc1234",
        "max_entries_per_day": None,
        "assumed_max_walk_cents": 1,
    }
    kwargs.update(overrides)
    return build_decision_row(**kwargs)


# quantise_price


def test_quantise_kills_the_float_tail():
    assert 1 - 0.34 != 0.66
    assert quantise_price(1 - 0.34) == 0.66


def test_quantise_passes_through_none_and_non_finite():
    assert quantise_price(None) is None
    assert quantise_price(float("nan")) is None
    assert quantise_price(float("inf")) is None


def test_quantise_keeps_four_decimals():
    assert quantise_price(0.123456) == 0.1235


# encode_row / PIPE_BUF


def test_encode_row_returns_utf8_json():
    encoded = encode_row({"a": 1, "b": None})
    assert encoded == b'{"a": 1, "b": null}'


def test_encode_row_uses_default_str_like_the_writer():
    encoded = encode_row({"d": date(2026, 9, 19)})
    assert encoded == b'{"d": "2026-09-19"}'


def test_encode_row_rejects_an_oversized_row():
    assert encode_row({"pad": "x" * 3000}) is not None
    assert encode_row({"pad": "x" * 5000}) is None


def test_encode_row_accounts_for_the_newline_the_writer_appends():
    """``ForecastsRawWriter`` writes ``json.dumps(row) + "\\n"`` in one syscall."""
    for pad in range(4000, 4200):
        row = {"p": "x" * pad}
        encoded = encode_row(row)
        written = len(json.dumps(row, default=str).encode()) + 1
        assert (encoded is None) is (written > PIPE_BUF)


def test_encode_row_survives_an_unserialisable_row():
    assert encode_row({"bad": object()}) is not None  # default=str stringifies it


def test_oversize_stub_carries_only_join_keys():
    row = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": "abc:KXRAIN-26SEP19-NYC",
        "scan_id": "abc",
        "ts": "2026-09-19T15:00:00+00:00",
        "ticker": "KXRAIN-26SEP19-NYC",
        "prints": ["..."] * 800,
    }
    stub = build_oversize_stub_row(row)
    assert stub["decision_id"] == "abc:KXRAIN-26SEP19-NYC"
    assert stub["ticker"] == "KXRAIN-26SEP19-NYC"
    assert stub["oversize"] is True
    assert stub["oversize_original_bytes"] > PIPE_BUF
    assert "prints" not in stub
    assert encode_row(stub) is not None


# build_decision_row


def test_decision_row_field_set_is_the_frozen_contract(mia_market):
    row = _decision(mia_market)
    assert set(row) == {
        "schema_version",
        "frame_version",
        "decision_id",
        "scan_id",
        "ts",
        "ticker",
        "city",
        "station",
        "settlement_date",
        "lst_offset_hours",
        "local_time_lst",
        "in_window",
        "yes_bid",
        "yes_ask",
        "no_ask",
        "no_ask_api",
        "spread",
        "yes_bid_size_fp",
        "yes_ask_size_fp",
        "volume_fp",
        "open_interest",
        "last_price_dollars",
        "close_time",
        "status",
        "can_close_early",
        "last_seen_yes_bid",
        "last_seen_ts",
        "in_price_band",
        "n_no_trades_30m",
        "no_volume_30m",
        "last_no_trade_ts",
        "last_no_trade_price",
        "no_trade_recent",
        "effective_lookback_seconds",
        "window_oldest_print_ts",
        "already_entered_market_day",
        "max_entries_per_day",
        "yes_bid_ge_040",
        "eligible",
        "skip_reason",
        "virtual_entry",
        "assumed_entry_price",
        "assumed_max_walk_cents",
        "pass_index",
        "book_capture",
        "trades_capture",
        "follow_capture",
        "rules_station_ok",
        "rules_station",
        "observer_version",
    }


def test_decision_row_has_no_local_time_civil(mia_market):
    """Pure function of ts and city; computed offline instead."""
    assert "local_time_civil" not in _decision(mia_market)


def test_decision_row_join_keys(mia_market):
    row = _decision(mia_market)
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["frame_version"] == FRAME_VERSION
    assert row["decision_id"] == "1f2e3d:KXRAIN-26SEP18-MIA"
    assert row["ts"] == NOW.isoformat()
    assert row["settlement_date"] == "2026-09-18"
    assert row["station"] == "MIA"
    assert row["city"] == "MIA"
    assert row["lst_offset_hours"] == -5


def test_decision_row_local_time_is_lst(mia_market):
    # 18:44 UTC - 5h = 13:44 LST.
    assert _decision(mia_market)["local_time_lst"] == "13:44"


def test_decision_row_prices_from_the_live_payload(mia_market):
    row = _decision(mia_market)
    assert row["yes_bid"] == 0.82
    assert row["yes_ask"] == 0.87
    assert row["no_ask"] == 0.18
    assert row["no_ask_api"] == 0.18
    assert row["spread"] == pytest.approx(0.05)


def test_no_ask_api_confirms_the_derived_identity(live_markets):
    checked = 0
    for market in live_markets:
        row = _decision(market)
        if row["no_ask"] is None or row["no_ask_api"] is None:
            continue
        assert row["no_ask"] == pytest.approx(row["no_ask_api"], abs=1e-9)
        checked += 1
    assert checked == len(live_markets)


def test_decision_row_sizes_are_floats(mia_market):
    row = _decision(mia_market)
    assert isinstance(row["yes_bid_size_fp"], float)
    assert row["yes_bid_size_fp"] == pytest.approx(64.27)
    assert isinstance(row["yes_ask_size_fp"], float)
    assert isinstance(row["volume_fp"], float)
    assert isinstance(row["open_interest"], float)
    assert row["open_interest"] == pytest.approx(15382.64)


def test_fractional_volumes_survive_across_the_whole_page(live_markets):
    # 48 of 54 live markets carry a fractional volume_fp.
    fractional = 0
    for market in live_markets:
        row = _decision(market)
        raw = float(market["volume_fp"])
        assert row["volume_fp"] == pytest.approx(raw)
        if raw % 1:
            fractional += 1
    assert fractional >= 40


def test_decision_row_census_fields(mia_market):
    row = _decision(mia_market)
    assert row["status"] == "active"
    assert row["can_close_early"] is True


def test_decision_row_station_tripwire_passes_on_live_rules(live_markets):
    for market in live_markets:
        row = _decision(market)
        assert row["rules_station_ok"] is True, row["ticker"]
        assert row["rules_station"] == row["station"]


def test_station_tripwire_fires_on_a_changed_rule(mia_market):
    tampered = dict(mia_market)
    tampered["rules_primary"] = "...the total precipitation at CLIXXX in Miami..."
    row = _decision(tampered)
    assert row["rules_station_ok"] is False
    assert row["rules_station"] == "XXX"


def test_station_tripwire_goes_not_ok_when_it_cannot_read_the_rules(mia_market):
    tampered = dict(mia_market)
    tampered["rules_primary"] = "rewritten wording with no station at all"
    row = _decision(tampered)
    assert row["rules_station_ok"] is False
    assert row["rules_station"] is None


def test_decision_row_carries_the_outcome(mia_market):
    row = _decision(mia_market)
    assert row["eligible"] is True
    assert row["skip_reason"] is None
    assert row["in_price_band"] is True
    assert row["yes_bid_ge_040"] is True
    assert row["assumed_entry_price"] == 0.18


def test_eligible_null_is_never_coerced(mia_market):
    row = _decision(mia_market, no_trades_30m=None, trades_capture="error")
    assert row["eligible"] is None
    assert row["skip_reason"] == "trades_unavailable"
    assert row["no_trade_recent"] is None


def test_already_entered_field_is_renamed(mia_market):
    row = _decision(mia_market, already_entered=True, already_entered_eval=True)
    assert row["already_entered_market_day"] is True
    assert "already_entered_today" not in row
    assert row["skip_reason"] == "already_entered"


def test_decision_row_trade_window_fields(mia_market, live_trades):
    window = summarise_trade_window(live_trades)
    row = _decision(
        mia_market,
        trade_window=window,
        window_end_ts=1789757051,
    )
    assert row["n_no_trades_30m"] == window.n_trades_no_side
    assert isinstance(row["no_volume_30m"], float)
    assert row["no_volume_30m"] == pytest.approx(window.no_volume)
    assert row["last_no_trade_ts"] == window.last_no_trade_ts
    assert row["window_oldest_print_ts"] == window.oldest_print_ts
    assert row["effective_lookback_seconds"] is not None
    assert row["effective_lookback_seconds"] > 500.0


def test_decision_row_without_a_trade_window(mia_market):
    row = _decision(mia_market, no_trades_30m=None, trades_capture="skipped_band")
    assert row["n_no_trades_30m"] is None
    assert row["no_volume_30m"] is None
    assert row["window_oldest_print_ts"] is None
    assert row["effective_lookback_seconds"] is None


def test_decision_row_capture_vocabulary_is_closed():
    assert CAPTURE_OUTCOMES == frozenset(
        {
            "ok",
            "error",
            "skipped_band",
            "skipped_not_current_day",
            "skipped_market_absent",
            "budget_abort",
            "wall_clock_abort",
            "backoff_abort",
        }
    )


def test_skip_reason_vocabulary_gained_market_absent():
    assert "market_absent" in SKIP_REASONS


def test_market_absent_row(mia_market):
    key = parse_kxrain_ticker("KXRAIN-26SEP18-MIA")
    assert key is not None
    row = build_decision_row(
        key=key,
        market=None,
        outcome=None,
        scan_id="1f2e3d",
        ts=NOW,
        in_window=True,
        already_entered=False,
        virtual_entry=False,
        pass_index=-1,
        book_capture="skipped_market_absent",
        trades_capture="skipped_market_absent",
        follow_capture=False,
        observer_version="abc1234",
        max_entries_per_day=None,
        assumed_max_walk_cents=1,
        last_seen_yes_bid=0.84,
        last_seen_ts="2026-09-18T18:34:11+00:00",
    )
    assert row["skip_reason"] == "market_absent"
    assert row["eligible"] is False
    assert row["yes_bid"] is None
    assert row["status"] is None
    assert row["last_seen_yes_bid"] == 0.84
    assert row["last_seen_ts"] == "2026-09-18T18:34:11+00:00"
    assert row["rules_station_ok"] is False
    assert row["rules_station"] is None


def test_last_seen_fields_are_null_on_every_non_absent_row(mia_market):
    row = _decision(mia_market)
    assert "last_seen_yes_bid" in row
    assert "last_seen_ts" in row
    assert row["last_seen_yes_bid"] is None
    assert row["last_seen_ts"] is None


def test_every_derived_price_is_quantised(live_markets):
    priced = ("yes_bid", "yes_ask", "no_ask", "no_ask_api", "spread", "assumed_entry_price")
    for market in live_markets:
        row = _decision(market)
        for field in priced:
            value = row[field]
            if value is None:
                continue
            assert value == round(value, 4), (market["ticker"], field, value)


def test_unpriced_market_yields_nulls_not_zeros(mia_market):
    blank = dict(mia_market)
    blank["yes_bid_dollars"] = ""
    blank["yes_ask_dollars"] = None
    row = _decision(blank)
    assert row["yes_bid"] is None
    assert row["no_ask"] is None
    assert row["spread"] is None
    assert row["skip_reason"] == "no_price"


# build_book_row — touch LAST, fractional sizes


def test_book_row_field_set(live_orderbook):
    row = build_book_row(
        decision_id="1f2e3d:KXRAIN-26SEP18-MIA",
        ts=NOW,
        ticker="KXRAIN-26SEP18-MIA",
        orderbook=live_orderbook,
        markets_yes_bid=0.82,
    )
    assert set(row) == {
        "schema_version",
        "frame_version",
        "decision_id",
        "ts",
        "ticker",
        "yes_bid_levels",
        "no_bid_levels",
        "no_ask_levels",
        "no_ask_depth_within_1c_contracts",
        "no_ask_depth_within_1c_dollars",
        "markets_yes_bid",
        "book_top_yes_bid",
        "total_yes_depth_dollars",
        "total_no_depth_dollars",
        "n_yes_levels",
        "n_no_levels",
        "raw_level_parse_failures",
    }


def test_book_touch_is_the_last_level(live_orderbook):
    row = build_book_row(
        decision_id="d",
        ts=NOW,
        ticker="KXRAIN-26SEP18-MIA",
        orderbook=live_orderbook,
        markets_yes_bid=0.82,
    )
    prices = [p for p, _ in row["yes_bid_levels"]]
    assert prices == sorted(prices)
    assert prices[-1] == 0.82
    assert row["book_top_yes_bid"] == 0.82
    assert row["markets_yes_bid"] == 0.82


def test_book_no_bid_ladder_also_ends_at_the_touch(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    prices = [p for p, _ in row["no_bid_levels"]]
    assert prices == sorted(prices)
    assert prices[-1] == 0.12


def test_no_ask_ladder_is_the_mirrored_yes_bid_ladder(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    yes = row["yes_bid_levels"]
    no_ask = row["no_ask_levels"]
    assert len(no_ask) == len(yes)
    for (yp, ys), (np_, ns) in zip(yes, no_ask, strict=True):
        assert np_ == pytest.approx(round(1.0 - yp, 4))
        assert ns == ys
    # Touch stays LAST positionally: the cheapest NO ask is the best one.
    assert no_ask[-1][0] == 0.18
    assert no_ask[-1][0] == min(p for p, _ in no_ask)


def test_book_sizes_are_fractional_floats(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    sizes = [s for _, s in row["yes_bid_levels"]]
    assert all(isinstance(s, float) for s in sizes)
    assert 96.18 in sizes
    assert 0.86 in sizes  # sub-whole-contract level from the live book


def test_sub_one_contract_level_is_not_truncated():
    book = {"orderbook_fp": {"yes_dollars": [["0.5000", "0.19"]], "no_dollars": []}}
    row = build_book_row(decision_id="d", ts=NOW, ticker="t", orderbook=book, markets_yes_bid=0.50)
    assert row["yes_bid_levels"] == [[0.5, 0.19]]
    assert row["n_yes_levels"] == 1


def test_no_ask_depth_within_1c_sums_the_side_the_rule_lifts(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    # NO-ask touch 0.18 (yes_bid 0.82); within 1c => 0.18 and 0.19 (yes 0.81).
    assert row["no_ask_depth_within_1c_contracts"] == pytest.approx(96.18 + 37.00)
    assert row["no_ask_depth_within_1c_dollars"] == pytest.approx(
        96.18 * 0.18 + 37.00 * 0.19, abs=1e-6
    )


def test_book_totals_are_price_times_size(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    expected_yes = sum(p * s for p, s in row["yes_bid_levels"])
    assert row["total_yes_depth_dollars"] == pytest.approx(expected_yes)
    expected_no = sum(p * s for p, s in row["no_bid_levels"])
    assert row["total_no_depth_dollars"] == pytest.approx(expected_no)


def test_book_depth_cap_keeps_the_best_levels():
    levels = [[f"{c / 100:.4f}", "1.00"] for c in range(1, 21)]
    book = {"orderbook_fp": {"yes_dollars": levels, "no_dollars": []}}
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=book, markets_yes_bid=0.20, depth_levels=10
    )
    prices = [p for p, _ in row["yes_bid_levels"]]
    assert len(prices) == 10
    assert prices[-1] == 0.20  # the touch, never the ten worst levels
    assert prices[0] == 0.11
    assert row["n_yes_levels"] == 20


def test_book_row_sorts_rather_than_trusting_the_order():
    book = {
        "orderbook_fp": {
            "yes_dollars": [["0.8400", "1.00"], ["0.1000", "2.00"], ["0.5000", "3.00"]],
            "no_dollars": [],
        }
    }
    row = build_book_row(decision_id="d", ts=NOW, ticker="t", orderbook=book, markets_yes_bid=0.84)
    assert [p for p, _ in row["yes_bid_levels"]] == [0.10, 0.50, 0.84]


def test_book_row_counts_level_parse_failures():
    book = {
        "orderbook_fp": {
            "yes_dollars": [["0.8400", "1.00"], ["bad", "1.00"], ["0.5000"], "junk"],
            "no_dollars": [["0.1000", "nope"]],
        }
    }
    row = build_book_row(decision_id="d", ts=NOW, ticker="t", orderbook=book, markets_yes_bid=0.84)
    assert row["raw_level_parse_failures"] == 4
    assert row["n_yes_levels"] == 1
    assert row["n_no_levels"] == 0


def test_empty_book_still_produces_a_row():
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=None, markets_yes_bid=0.84
    )
    assert row["yes_bid_levels"] == []
    assert row["no_ask_levels"] == []
    assert row["book_top_yes_bid"] is None
    assert row["no_ask_depth_within_1c_contracts"] == 0.0
    assert row["n_yes_levels"] == 0


def test_book_prices_are_quantised(live_orderbook):
    row = build_book_row(
        decision_id="d", ts=NOW, ticker="t", orderbook=live_orderbook, markets_yes_bid=0.82
    )
    for ladder in ("yes_bid_levels", "no_bid_levels", "no_ask_levels"):
        for price, _ in row[ladder]:
            assert price == round(price, 4)


# build_trades_row


def _trades_row(live_trades, **overrides):
    window = summarise_trade_window(live_trades)
    kwargs: dict = {
        "decision_id": "1f2e3d:KXRAIN-26SEP18-MIA",
        "ts": NOW,
        "ticker": "KXRAIN-26SEP18-MIA",
        "window": window,
        "window_start_ts": 1789756251,
        "window_end_ts": 1789758051,
        "n_pages_fetched": 1,
    }
    kwargs.update(overrides)
    return build_trades_row(**kwargs)


def test_trades_row_field_set(live_trades):
    row = _trades_row(live_trades)
    assert set(row) == {
        "schema_version",
        "frame_version",
        "decision_id",
        "ts",
        "ticker",
        "window_start_ts",
        "window_end_ts",
        "n_trades_total",
        "n_trades_no_side",
        "no_volume_30m",
        "yes_volume_30m",
        "vwap_no_30m",
        "prints",
        "prints_truncated",
        "page_truncated",
        "window_oldest_print_ts",
        "effective_lookback_seconds",
        "n_pages_fetched",
        "print_parse_failures",
    }


def test_trades_row_aggregates_match_the_live_page(live_trades):
    row = _trades_row(live_trades)
    raw = live_trades["trades"]
    assert row["n_trades_total"] == len(raw)
    assert row["n_trades_no_side"] == sum(1 for t in raw if t["taker_side"] == "no")
    assert row["no_volume_30m"] == pytest.approx(
        sum(float(t["count_fp"]) for t in raw if t["taker_side"] == "no")
    )
    assert isinstance(row["no_volume_30m"], float)
    assert isinstance(row["yes_volume_30m"], float)


def test_trades_row_prints_are_capped_at_twelve(live_trades):
    assert MAX_PRINTS == 12  # lowered from 20 for PIPE_BUF headroom
    row = _trades_row(live_trades)
    assert len(row["prints"]) == MAX_PRINTS
    assert row["prints_truncated"] is True
    assert set(row["prints"][0]) == {"ts", "taker_side", "yes_price", "no_price", "count"}


def test_trades_row_page_truncation_is_recorded(live_trades):
    row = _trades_row(live_trades)
    assert row["page_truncated"] is True  # the live page carries a cursor
    assert row["n_pages_fetched"] == 1
    assert _trades_row(live_trades, n_pages_fetched=2)["n_pages_fetched"] == 2


def test_trades_row_effective_lookback(live_trades):
    row = _trades_row(live_trades)
    oldest = min(t["created_time"] for t in live_trades["trades"])
    assert row["window_oldest_print_ts"] == oldest == "2026-09-18T18:33:48.951207Z"
    expected = 1789758051 - (to_epoch(oldest) or 0.0)
    assert row["effective_lookback_seconds"] == pytest.approx(expected, abs=0.01)
    # The page reaches back ~9.9 minutes, not the 30 the query asked for.
    newest = max(t["created_time"] for t in live_trades["trades"])
    span = (to_epoch(newest) or 0.0) - (to_epoch(oldest) or 0.0)
    assert span == pytest.approx(592.25, abs=1.0)


def test_trades_row_effective_lookback_is_null_on_an_empty_page():
    window = summarise_trade_window({"trades": []})
    row = build_trades_row(
        decision_id="d",
        ts=NOW,
        ticker="t",
        window=window,
        window_start_ts=0,
        window_end_ts=1800,
        n_pages_fetched=1,
    )
    assert row["window_oldest_print_ts"] is None
    assert row["effective_lookback_seconds"] is None
    assert row["vwap_no_30m"] is None


def test_trades_row_print_prices_are_quantised(live_trades):
    row = _trades_row(live_trades)
    for print_row in row["prints"]:
        assert print_row["yes_price"] == round(print_row["yes_price"], 4)
        assert print_row["no_price"] == round(print_row["no_price"], 4)
        assert isinstance(print_row["count"], float)


def test_trades_row_keeps_sub_contract_prints():
    payload = {
        "trades": [
            {
                "created_time": "2026-09-19T14:00:00Z",
                "taker_side": "no",
                "yes_price_dollars": "0.8100",
                "no_price_dollars": "0.1900",
                "count_fp": "0.19",
            }
        ]
    }
    row = build_trades_row(
        decision_id="d",
        ts=NOW,
        ticker="t",
        window=summarise_trade_window(payload),
        window_start_ts=0,
        window_end_ts=1,
        n_pages_fetched=1,
    )
    assert row["n_trades_no_side"] == 1
    assert row["no_volume_30m"] == pytest.approx(0.19)
    assert row["prints"][0]["count"] == pytest.approx(0.19)


# build_tick_row

TICK_PROCESS_START = datetime(2026, 9, 18, 14, 0, 0, tzinfo=UTC)


def _tick_row(**overrides):
    kwargs: dict = {
        "scan_id": "1f2e3d",
        "ts": NOW,
        "process_start_ts": TICK_PROCESS_START,
        "tick_outcome": "ok",
        "n_cities_in_window": 6,
        "n_markets_returned": 54,
        "n_parsed": 54,
        "n_unparseable": 0,
        "unparseable_tickers": [],
        "markets_truncated": False,
        "markets_cycle_aborted": False,
        "markets_pages_consumed": 1,
        "census_suppressed": False,
        "requests_used": 13,
        "max_requests_per_tick": 60,
        "tick_seed": 42,
        "elapsed_seconds": 1.2345,
        "backoff_remaining_at_start": 0.0,
        "backoff_accessor_present": True,
        "rows_lost": 0,
        "oversize_rows_dropped": 0,
        "n_decision_rows": 54,
        "observer_version": "abc1234",
        "tick_interval_seconds": 600.0,
    }
    kwargs.update(overrides)
    return build_tick_row(**kwargs)


def test_tick_row_field_set_is_the_frozen_contract():
    row = _tick_row()
    assert set(row) == {
        "schema_version",
        "frame_version",
        "scan_id",
        "ts",
        "process_start_ts",
        "tick_outcome",
        "n_cities_in_window",
        "n_markets_returned",
        "n_parsed",
        "n_unparseable",
        "unparseable_tickers",
        "unparseable_tickers_truncated",
        "markets_truncated",
        "markets_cycle_aborted",
        "markets_pages_consumed",
        "census_suppressed",
        "requests_used",
        "max_requests_per_tick",
        "tick_seed",
        "elapsed_seconds",
        "backoff_remaining_at_start",
        "backoff_accessor_present",
        "rows_lost",
        "oversize_rows_dropped",
        "n_decision_rows",
        "observer_version",
        "tick_interval_seconds",
    }


def test_tick_row_records_the_configured_cadence():
    assert _tick_row(tick_interval_seconds=1800)["tick_interval_seconds"] == 1800.0
    assert _tick_row(tick_interval_seconds=600)["tick_interval_seconds"] == 600.0


def test_tick_row_join_keys_and_versions():
    row = _tick_row()
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["frame_version"] == FRAME_VERSION
    assert row["scan_id"] == "1f2e3d"
    assert row["ts"] == NOW.isoformat()
    assert row["process_start_ts"] == TICK_PROCESS_START.isoformat()


def test_tick_row_passes_through_counters():
    row = _tick_row(
        n_cities_in_window=3,
        n_markets_returned=27,
        n_parsed=25,
        n_unparseable=2,
        requests_used=8,
        max_requests_per_tick=60,
        n_decision_rows=25,
    )
    assert row["n_cities_in_window"] == 3
    assert row["n_markets_returned"] == 27
    assert row["n_parsed"] == 25
    assert row["n_unparseable"] == 2
    assert row["requests_used"] == 8
    assert row["max_requests_per_tick"] == 60
    assert row["n_decision_rows"] == 25
    assert isinstance(row["n_cities_in_window"], int)


def test_tick_row_tick_seed_is_int():
    row = _tick_row(tick_seed=987654321)
    assert row["tick_seed"] == 987654321
    assert isinstance(row["tick_seed"], int)


def test_tick_row_caps_unparseable_tickers_at_five():
    tickers = [f"KXRAIN-26SEP19-BAD{i}" for i in range(9)]
    row = _tick_row(n_unparseable=9, unparseable_tickers=tickers)
    assert MAX_UNPARSEABLE_TICKERS == 5
    assert row["unparseable_tickers"] == tickers[:5]
    assert row["unparseable_tickers_truncated"] is True
    assert row["n_unparseable"] == 9


def test_tick_row_unparseable_tickers_not_truncated_under_the_cap():
    row = _tick_row(n_unparseable=2, unparseable_tickers=["A", "B"])
    assert row["unparseable_tickers"] == ["A", "B"]
    assert row["unparseable_tickers_truncated"] is False


def test_tick_row_unparseable_tickers_exactly_at_the_cap_is_not_truncated():
    tickers = [f"T{i}" for i in range(5)]
    row = _tick_row(n_unparseable=5, unparseable_tickers=tickers)
    assert row["unparseable_tickers"] == tickers
    assert row["unparseable_tickers_truncated"] is False


def test_tick_row_never_raises_on_a_non_string_ticker_entry():
    row = _tick_row(n_unparseable=1, unparseable_tickers=[12345])
    assert row["unparseable_tickers"] == ["12345"]


def test_tick_outcome_vocabulary_is_exactly_nine_members():
    assert TICK_OUTCOMES == frozenset(
        {
            "ok",
            "skipped_no_city_in_window",
            "skipped_backoff",
            "markets_fetch_error",
            "budget_abort",
            "wall_clock_abort",
            "backoff_abort",
            "disk_disabled",
            "error",
        }
    )


def test_tick_outcome_vocabulary_is_not_abort_reasons():
    # They overlap only on backoff_abort; a tick skipped before any capture
    # began is skipped_backoff, not a capture-abort reason at all.
    from python.data.kxrain_budget import ABORT_REASONS

    assert "skipped_backoff" in TICK_OUTCOMES
    assert "skipped_backoff" not in ABORT_REASONS
    assert ABORT_REASONS < TICK_OUTCOMES
    assert "skipped_market_absent" not in TICK_OUTCOMES


@pytest.mark.parametrize("outcome", sorted(TICK_OUTCOMES))
def test_every_documented_tick_outcome_is_accepted_silently(caplog, outcome):
    with caplog.at_level(logging.WARNING):
        _tick_row(tick_outcome=outcome)
    assert not any("outside the vocabulary" in r.message for r in caplog.records)


def test_tick_row_warns_but_does_not_raise_on_an_unknown_outcome(caplog):
    with caplog.at_level(logging.WARNING):
        row = _tick_row(tick_outcome="not_a_real_outcome")
    assert row["tick_outcome"] == "not_a_real_outcome"
    assert any("outside the vocabulary" in r.message for r in caplog.records)


def test_tick_row_elapsed_and_backoff_are_quantised_floats():
    row = _tick_row(elapsed_seconds=1.23456789, backoff_remaining_at_start=0.987654321)
    assert row["elapsed_seconds"] == 1.235
    assert row["backoff_remaining_at_start"] == 0.988
    assert isinstance(row["elapsed_seconds"], float)
    assert isinstance(row["backoff_remaining_at_start"], float)


def test_tick_row_markets_truncated_is_bool():
    assert _tick_row(markets_truncated=True)["markets_truncated"] is True
    assert _tick_row(markets_truncated=False)["markets_truncated"] is False


def test_tick_row_is_well_under_pipe_buf_even_at_the_ticker_cap():
    tickers = [f"KXRAIN-26SEP19-{'W' * 24}-{i}" for i in range(20)]
    row = _tick_row(n_unparseable=20, unparseable_tickers=tickers, observer_version="a" * 40)
    encoded = encode_row(row)
    assert encoded is not None
    assert len(encoded) < 1000


# build_settlement_row


def test_settlement_row_from_the_live_settled_fixture(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    assert row["ticker"] == "KXRAIN-26SEP17-PHIL"
    assert row["city"] == "PHIL"
    assert row["station"] == "PHL"
    assert row["settlement_date"] == "2026-09-17"
    assert row["result"] == "yes"
    assert row["expiration_value_raw"] == "0.05"
    assert row["expiration_value_in"] == pytest.approx(0.05)
    assert row["expiration_value_trace"] is False
    assert row["derived_yes"] is True
    assert row["rule_consistent"] is True


def test_settlement_row_reads_the_correct_field_names(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    # settlement_ts/open_interest_fp/finalized are the real API spellings.
    assert row["settlement_ts"] == "2026-09-18T11:32:07.318204Z"
    assert row["open_interest"] == pytest.approx(31204.40)
    assert row["status"] == "finalized"
    assert "settled_ts" not in settled_market
    assert "open_interest" not in settled_market


def test_settlement_row_field_set(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    assert set(row) == {
        "schema_version",
        "frame_version",
        "ts",
        "ticker",
        "city",
        "station",
        "settlement_date",
        "expiration_value_raw",
        "expiration_value_in",
        "expiration_value_trace",
        "result",
        "derived_yes",
        "rule_consistent",
        "last_price_dollars",
        "close_time",
        "settlement_ts",
        "status",
        "rules_secondary_sha256",
        "volume_fp",
        "open_interest",
    }


def test_open_market_from_the_live_fixture_is_refused(live_markets):
    refused = 0
    for market in live_markets:
        assert market["result"] == ""
        assert build_settlement_row(market, ts=NOW) is None
        refused += 1
    assert refused == len(live_markets)


def test_the_phantom_no_settlement_is_structurally_impossible(live_markets):
    open_market = dict(live_markets[0])
    assert open_market["expiration_value"] == ""
    assert open_market["result"] == ""
    # Naive: "" parses to 0.0 -> derived_yes False -> consistent with result != "yes".
    naive_derived_yes = False
    naive_consistent = naive_derived_yes == (open_market["result"] == "yes")
    assert naive_consistent is True  # the tripwire would NOT have fired
    assert build_settlement_row(open_market, ts=NOW) is None


@pytest.mark.parametrize("result", ["", None, "void", 0, "settled", "yes/no"])
def test_only_yes_and_no_are_written(settled_market, result):
    market = dict(settled_market)
    market["result"] = result
    assert build_settlement_row(market, ts=NOW) is None


def test_result_is_normalised_before_the_guard(settled_market):
    # Refusing a real "YES " would discard a settled market-day permanently.
    market = dict(settled_market)
    market["result"] = " YES "
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["result"] == "yes"


def test_no_settlement_is_written(settled_market):
    market = dict(settled_market)
    market["result"] = "no"
    market["expiration_value"] = "0.00"
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["derived_yes"] is False
    assert row["rule_consistent"] is True


def test_rule_inconsistency_is_recorded_not_suppressed(settled_market):
    market = dict(settled_market)
    market["result"] = "no"
    market["expiration_value"] = "0.05"
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["derived_yes"] is True
    assert row["rule_consistent"] is False


def test_trace_expiration_value_parses_to_zero(settled_market):
    market = dict(settled_market)
    market["result"] = "no"
    market["expiration_value"] = "T"
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["expiration_value_raw"] == "T"
    assert row["expiration_value_in"] == 0.0
    assert row["expiration_value_trace"] is True
    assert row["derived_yes"] is False


def test_missing_expiration_value_is_trace(settled_market):
    market = dict(settled_market)
    market["result"] = "no"
    market.pop("expiration_value")
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["expiration_value_raw"] is None
    assert row["expiration_value_in"] == 0.0
    assert row["expiration_value_trace"] is True


def test_last_price_is_stored_raw_and_never_as_an_outcome(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    assert row["last_price_dollars"] == pytest.approx(0.98)
    assert row["derived_yes"] is True  # derived from expiration_value, not price


def test_unparseable_ticker_is_refused(settled_market):
    market = dict(settled_market)
    market["ticker"] = "KXRAIN-26SEP17-ZZZ"
    assert build_settlement_row(market, ts=NOW) is None


def test_legacy_suffix_settles_under_its_canonical_station(settled_market):
    market = dict(settled_market)
    market["ticker"] = "KXRAIN-26SEP16-PROV"
    market["result"] = "no"
    market["expiration_value"] = "0.00"
    row = build_settlement_row(market, ts=NOW)
    assert row is not None
    assert row["city"] == "PVD"
    assert row["station"] == "PVD"


# PIPE_BUF — every row type, including a worst case


def test_every_live_decision_row_is_under_pipe_buf(live_markets, live_trades):
    window = summarise_trade_window(live_trades)
    for market in live_markets:
        row = _decision(market, trade_window=window, window_end_ts=1789758051)
        encoded = encode_row(row)
        assert encoded is not None, market["ticker"]
        assert len(encoded) < 1500


def test_live_book_row_is_well_under_pipe_buf(live_orderbook):
    row = build_book_row(
        decision_id="1f2e3d:KXRAIN-26SEP18-MIA",
        ts=NOW,
        ticker="KXRAIN-26SEP18-MIA",
        orderbook=live_orderbook,
        markets_yes_bid=0.82,
    )
    encoded = encode_row(row)
    assert encoded is not None
    assert len(encoded) < 1400


def test_live_trades_row_is_well_under_pipe_buf(live_trades):
    encoded = encode_row(_trades_row(live_trades))
    assert encoded is not None
    assert len(encoded) < 2000


def test_settlement_row_is_under_pipe_buf(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    assert encode_row(row) is not None


def test_worst_case_trades_row_is_under_pipe_buf():
    ticker = "KXRAIN-26SEP19-" + "W" * 24
    payload = {
        "cursor": "EhIKEAciYuWbMZSOCy4q_Lcnmn8aDAiAiLbVBhDoxvjLAQ",
        "trades": [
            {
                "created_time": f"2026-09-18T18:4{i % 10}:29.694354Z",
                "taker_side": "no" if i % 2 else "yes",
                "yes_price_dollars": "0.3300",
                "no_price_dollars": "0.6700",
                "count_fp": "1796.34",
                "trade_id": "b1b0f8f6-0e2a-4f9e-9a3a-0f7b4c2d1e5a",
            }
            for i in range(40)
        ],
    }
    window = summarise_trade_window(payload)
    row = build_trades_row(
        decision_id=f"1f2e3d4c5b6a:{ticker}",
        ts=NOW,
        ticker=ticker,
        window=window,
        window_start_ts=1789756251,
        window_end_ts=1789758051,
        n_pages_fetched=2,
    )
    encoded = encode_row(row)
    assert encoded is not None
    assert len(row["prints"]) == 12
    assert len(encoded) < 2000, len(encoded)


def test_a_row_past_pipe_buf_is_rejected_and_stubbed(live_trades):
    row = _trades_row(live_trades)
    row["prints"] = row["prints"] * 40
    assert encode_row(row) is None
    stub = build_oversize_stub_row(row)
    assert encode_row(stub) is not None


def test_decision_row_does_not_raise_on_a_naive_timestamp(mia_market):
    row = _decision(mia_market, ts=datetime(2026, 9, 19, 18, 44))
    assert row["local_time_lst"] is None
    assert encode_row(row) is not None


def test_every_stream_carries_schema_and_frame_version(
    mia_market, live_orderbook, live_trades, settled_market
):
    rows = [
        _decision(mia_market),
        build_book_row(
            decision_id="s:t", ts=NOW, ticker=mia_market["ticker"],
            orderbook=live_orderbook, markets_yes_bid=0.82,
        ),
        build_trades_row(
            decision_id="s:t", ts=NOW, ticker=mia_market["ticker"],
            window=summarise_trade_window(live_trades),
            window_start_ts=1789757008, window_end_ts=1789758808,
            n_pages_fetched=1,
        ),
        build_settlement_row(settled_market, ts=NOW),
        _tick_row(),
    ]
    for row in rows:
        assert row is not None
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["frame_version"] == FRAME_VERSION


def test_empty_book_depth_totals_are_float_not_int():
    row = build_book_row(
        decision_id="s:t", ts=NOW, ticker="KXRAIN-26SEP19-NYC",
        orderbook={"orderbook_fp": {"yes_dollars": [], "no_dollars": []}},
        markets_yes_bid=None,
    )
    assert isinstance(row["total_yes_depth_dollars"], float)
    assert isinstance(row["total_no_depth_dollars"], float)


def test_settlement_row_records_the_rules_secondary_hash(settled_market):
    row = build_settlement_row(settled_market, ts=NOW)
    assert row is not None
    assert isinstance(row["rules_secondary_sha256"], str)
    assert len(row["rules_secondary_sha256"]) == 16
