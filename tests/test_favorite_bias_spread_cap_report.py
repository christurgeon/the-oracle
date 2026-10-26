from __future__ import annotations

from scripts.favorite_bias_spread_cap_report import pair_events, pnl, summarize, verdict


def _row(ts: str, ticker: str, bid: float, ask: float, decision: str = "buy") -> dict:
    return {
        "ts": ts, "market_ticker": ticker, "city": "NYC", "var": "high",
        "yes_bid": bid, "yes_ask": ask, "decision": decision,
    }


T1 = "KXHIGHNY-26OCT05-T70"
T2 = "KXHIGHNY-26OCT05-B71.5"


def test_wide_then_tight_changes_entry_to_tight_row() -> None:
    rows = [
        _row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.60, "skip_spread"),
        _row("2026-10-04T15:10:00+00:00", T1, 0.56, 0.58, "buy"),
    ]
    [e] = pair_events(rows, 0.05)
    assert e.changed and e.nocap.ask == 0.60 and e.cap is not None and e.cap.ask == 0.58


def test_spread_equal_to_cap_is_not_changed() -> None:
    [e] = pair_events([_row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.55)], 0.05)
    assert not e.changed and e.cap == e.nocap


def test_never_tight_means_no_cap_trade() -> None:
    [e] = pair_events([_row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.60, "skip_spread")], 0.05)
    assert e.changed and e.cap is None
    assert pnl(e.cap, {T1: 1}) == 0.0


def test_skip_and_unparseable_rows_ignored() -> None:
    rows = [
        _row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.60, "skip"),
        _row("2026-10-04T14:10:00+00:00", "GARBAGE", 0.50, 0.51),
    ]
    assert pair_events(rows, 0.05) == []


def test_summary_diff_and_verdict_wait_below_min_n() -> None:
    rows = [
        _row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.60, "skip_spread"),
        _row("2026-10-04T15:10:00+00:00", T2, 0.56, 0.58, "buy"),
    ]
    events = pair_events(rows, 0.05)
    s = summarize(events, {T1: 0, T2: 1})
    assert s["changed"] == 1 and s["changed_later_tight"] == 1
    assert s["diff_per_changed_event"] > 0
    assert verdict(s).startswith("WAIT")


def test_unsettled_events_are_not_scored() -> None:
    events = pair_events([_row("2026-10-04T14:10:00+00:00", T1, 0.50, 0.60)], 0.05)
    assert summarize(events, {})["events"] == 0
