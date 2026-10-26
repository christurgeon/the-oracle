"""Unit tests for ``python/data/kxrain_trades.py``.

Driven partly by the ``tests/fixtures/kxrain/live_trades.json`` page so
the aggregation is pinned against the live payload shape, not hand-written ones."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_trades import (  # noqa: E402
    MAX_PRINTS,
    summarise_trade_window,
)

FIXTURE = Path(__file__).parent / "fixtures" / "kxrain" / "live_trades.json"


def _print(ts, side, yes, no, count):
    return {
        "created_time": ts,
        "taker_side": side,
        "yes_price_dollars": yes,
        "no_price_dollars": no,
        "count_fp": count,
        "trade_id": ts,
    }


# Against the live fixture


@pytest.fixture(scope="module")
def live_payload():
    if not FIXTURE.exists():
        pytest.skip("live_trades.json not captured")
    return json.loads(FIXTURE.read_text())


def test_live_page_parses_with_zero_failures(live_payload):
    window = summarise_trade_window(live_payload)
    assert window.parse_failures == 0
    assert window.n_trades_total == len(live_payload["trades"])


def test_live_page_splits_yes_and_no(live_payload):
    window = summarise_trade_window(live_payload)
    raw = live_payload["trades"]
    assert window.n_trades_no_side == sum(1 for t in raw if t["taker_side"] == "no")
    assert window.n_trades_no_side + sum(
        1 for t in raw if t["taker_side"] == "yes"
    ) == window.n_trades_total


def test_live_page_volumes_match_the_raw_counts(live_payload):
    window = summarise_trade_window(live_payload)
    raw = live_payload["trades"]
    assert window.no_volume == pytest.approx(
        sum(float(t["count_fp"]) for t in raw if t["taker_side"] == "no")
    )
    assert window.yes_volume == pytest.approx(
        sum(float(t["count_fp"]) for t in raw if t["taker_side"] == "yes")
    )


def test_live_page_carries_fractional_contract_counts():
    payload = json.loads(FIXTURE.read_text())
    counts = [float(t["count_fp"]) for t in payload["trades"]]
    assert any(c != int(c) for c in counts), "fixture no longer exercises fractional sizes"
    window = summarise_trade_window(payload)
    assert window.no_volume + window.yes_volume == pytest.approx(sum(counts))
    assert (window.no_volume + window.yes_volume) != int(window.no_volume + window.yes_volume)


def test_sub_one_contract_print_is_kept_not_truncated_to_zero():
    payload = {"trades": [_print("2026-09-19T14:00:00Z", "no", "0.74", "0.26", "0.19")]}
    window = summarise_trade_window(payload)
    assert window.parse_failures == 0
    assert window.has_recent_no_trade is True
    assert window.no_volume == pytest.approx(0.19)


def test_live_page_caps_prints_and_flags_truncation(live_payload):
    window = summarise_trade_window(live_payload)
    assert len(window.prints) <= MAX_PRINTS
    assert window.prints_truncated is (len(live_payload["trades"]) > MAX_PRINTS)


def test_live_page_prints_are_newest_first(live_payload):
    window = summarise_trade_window(live_payload)
    assert [p.ts for p in window.prints] == sorted((p.ts for p in window.prints), reverse=True)


def test_live_page_vwap_is_inside_the_no_price_range(live_payload):
    window = summarise_trade_window(live_payload)
    no_prices = [float(t["no_price_dollars"]) for t in live_payload["trades"]
                 if t["taker_side"] == "no"]
    if not no_prices:
        pytest.skip("no NO prints on the captured page")
    assert window.vwap_no is not None
    assert min(no_prices) <= window.vwap_no <= max(no_prices)


def test_live_page_reports_its_cursor_as_page_truncated(live_payload):
    window = summarise_trade_window(live_payload)
    assert window.page_truncated is bool(live_payload.get("cursor"))


# The eligibility leg


def test_no_print_present_makes_the_window_eligible():
    payload = {"trades": [_print("2026-09-19T14:00:00Z", "no", "0.30", "0.70", "5.00")]}
    window = summarise_trade_window(payload)
    assert window.has_recent_no_trade is True
    assert window.last_no_trade_price == 0.70
    assert window.last_no_trade_ts == "2026-09-19T14:00:00Z"


def test_yes_only_window_is_not_eligible():
    payload = {"trades": [_print("2026-09-19T14:00:00Z", "yes", "0.30", "0.70", "5.00")]}
    window = summarise_trade_window(payload)
    assert window.has_recent_no_trade is False
    assert window.n_trades_no_side == 0
    assert window.last_no_trade_ts is None
    assert window.vwap_no is None


def test_empty_page_is_a_valid_not_eligible_answer():
    window = summarise_trade_window({"trades": []})
    assert window.has_recent_no_trade is False
    assert window.n_trades_total == 0
    assert window.parse_failures == 0


def test_none_payload_does_not_raise():
    window = summarise_trade_window(None)
    assert window.n_trades_total == 0
    assert window.has_recent_no_trade is False


# Ordering — the trap a lexicographic sort walks into


def test_variable_subsecond_precision_sorts_correctly():
    # Lexicographically "Z" > ".", so a raw string sort mis-orders these.
    payload = {"trades": [
        _print("2026-09-19T14:00:25Z", "no", "0.30", "0.70", "1.00"),
        _print("2026-09-19T14:00:25.900000Z", "no", "0.31", "0.69", "1.00"),
    ]}
    window = summarise_trade_window(payload)
    assert window.last_no_trade_ts == "2026-09-19T14:00:25.900000Z"
    assert window.last_no_trade_price == 0.69


def test_out_of_order_page_is_reordered():
    payload = {"trades": [
        _print("2026-09-19T14:00:00Z", "no", "0.30", "0.70", "1.00"),
        _print("2026-09-19T15:00:00Z", "no", "0.31", "0.69", "1.00"),
    ]}
    window = summarise_trade_window(payload)
    assert window.last_no_trade_ts == "2026-09-19T15:00:00Z"
    assert [p.ts for p in window.prints][0] == "2026-09-19T15:00:00Z"


def test_print_cap_keeps_the_most_recent():
    payload = {"trades": [
        _print(f"2026-09-19T14:{m:02d}:00Z", "no", "0.30", "0.70", "1.00") for m in range(30)
    ]}
    window = summarise_trade_window(payload)
    assert len(window.prints) == MAX_PRINTS
    assert window.prints[0].ts == "2026-09-19T14:29:00Z"
    assert window.prints_truncated is True
    # Aggregates still cover the whole page, not just the capped prints.
    assert window.n_trades_no_side == 30
    assert window.no_volume == pytest.approx(30.0)


# Malformed rows: counted, excluded, never fatal


@pytest.mark.parametrize("bad", [
    _print("2026-09-19T14:00:00Z", "NO_SUCH_SIDE", "0.30", "0.70", "5.00"),
    _print("2026-09-19T14:00:00Z", "no", "not-a-price", "0.70", "5.00"),
    _print("2026-09-19T14:00:00Z", "no", "0.30", "0.70", "0.00"),
    _print("2026-09-19T14:00:00Z", "no", "0.30", "0.70", "-1.00"),
    _print("not-a-timestamp", "no", "0.30", "0.70", "5.00"),
    _print("2026-09-19T14:00:00Z", "no", "1.40", "0.70", "5.00"),
    "not-even-a-dict",
])
def test_malformed_rows_are_counted_and_excluded(bad):
    good = _print("2026-09-19T15:00:00Z", "no", "0.30", "0.70", "2.00")
    window = summarise_trade_window({"trades": [bad, good]})
    assert window.parse_failures == 1
    assert window.n_trades_total == 2
    assert window.n_trades_no_side == 1
    assert window.no_volume == pytest.approx(2.0)


def test_uppercase_taker_side_is_accepted():
    payload = {"trades": [_print("2026-09-19T14:00:00Z", "NO", "0.30", "0.70", "5.00")]}
    window = summarise_trade_window(payload)
    assert window.parse_failures == 0
    assert window.has_recent_no_trade is True


def test_vwap_is_size_weighted_not_a_plain_mean():
    payload = {"trades": [
        _print("2026-09-19T14:00:00Z", "no", "0.10", "0.90", "1.00"),
        _print("2026-09-19T14:01:00Z", "no", "0.50", "0.50", "99.00"),
    ]}
    window = summarise_trade_window(payload)
    assert window.vwap_no is not None
    assert window.vwap_no == pytest.approx((0.90 * 1 + 0.50 * 99) / 100)
