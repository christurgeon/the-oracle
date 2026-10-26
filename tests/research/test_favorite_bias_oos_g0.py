"""Replay-rule tests for scripts/research/favorite_bias_oos_g0.py on synthetic markets."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from scripts.research.favorite_bias_oos_g0 import (
    Bar,
    Entry,
    Event,
    Market,
    Quote,
    RuleConfig,
    Trade,
    build_events,
    grid_quotes,
    hourly_grid,
    parse_event_date,
    q1_verdict,
    q2_signal,
    replay_event,
    taker_fee,
)

FD = date(2025, 6, 10)
LEAD1 = datetime(2025, 6, 9, tzinfo=UTC)
OPEN = LEAD1 + timedelta(hours=14)


def _ts(hour: int) -> int:
    return int((LEAD1 + timedelta(hours=hour)).timestamp())


def _market(suffix: str, strike_type: str | None, result: str = "yes") -> Market:
    return Market(
        ticker=f"KXHIGHNY-25JUN10-{suffix}",
        event_ticker="KXHIGHNY-25JUN10",
        series="KXHIGHNY",
        strike_type=strike_type,
        open_time=OPEN,
        close_time=LEAD1 + timedelta(days=2),
        result=result,
        tier="hist",
    )


GREATER = _market("T80", "greater")
LESS = _market("T73", "less", result="no")
EVENT = Event("KXHIGHNY-25JUN10", "KXHIGHNY", "NYC", "high", FD, (GREATER, LESS))


def _g_plus(delta: float):
    return lambda mid: mid + delta


def _q(hour: int, bid: float, ask: float, vol: float = 1000.0) -> Quote:
    return Quote(_ts(hour), bid, ask, vol)


def test_band_low_edge_is_included():
    e = replay_event(EVENT, {GREATER.ticker: [_q(16, 0.49, 0.51)]}, _g_plus(0.10), RuleConfig())
    assert e is not None and e.mid == pytest.approx(0.50)


def test_band_high_edge_is_excluded():
    e = replay_event(EVENT, {GREATER.ticker: [_q(16, 0.74, 0.76)]}, _g_plus(0.20), RuleConfig())
    assert e is None


def test_edge_floor_inclusive():
    ask = 0.51
    fee = taker_fee(ask)
    at_floor = lambda mid: ask + fee + 0.01  # noqa: E731
    below = lambda mid: ask + fee + 0.009  # noqa: E731
    quotes = {GREATER.ticker: [_q(16, 0.49, ask)]}
    assert replay_event(EVENT, quotes, at_floor, RuleConfig()) is not None
    assert replay_event(EVENT, quotes, below, RuleConfig()) is None


def test_lead1_window_bounds():
    g = _g_plus(0.10)
    start = Quote(_ts(0), 0.55, 0.57, 1000)
    last_bar = Quote(_ts(24), 0.55, 0.57, 1000)
    after = Quote(_ts(25), 0.55, 0.57, 1000)
    assert replay_event(EVENT, {GREATER.ticker: [start]}, g, RuleConfig()) is None
    assert replay_event(EVENT, {GREATER.ticker: [after]}, g, RuleConfig()) is None
    e = replay_event(EVENT, {GREATER.ticker: [last_bar]}, g, RuleConfig())
    assert e is not None and e.t == _ts(24)


def test_hourly_grid_stays_inside_lead1_after_open():
    grid = hourly_grid(EVENT, OPEN)
    assert grid[0] == _ts(15)
    assert grid[-1] == _ts(24)


def test_event_cap_picks_highest_edge_on_the_first_bar():
    quotes = {
        GREATER.ticker: [_q(16, 0.55, 0.57)],
        LESS.ticker: [_q(16, 0.55, 0.56)],  # cheaper ask -> higher edge
    }
    e = replay_event(EVENT, quotes, _g_plus(0.10), RuleConfig())
    assert e is not None and e.market.ticker == LESS.ticker


def test_event_cap_tie_breaks_on_ticker():
    quotes = {GREATER.ticker: [_q(16, 0.55, 0.57)], LESS.ticker: [_q(16, 0.55, 0.57)]}
    e = replay_event(EVENT, quotes, _g_plus(0.10), RuleConfig())
    assert e is not None and e.market.ticker == min(GREATER.ticker, LESS.ticker)


def test_first_qualifying_bar_wins_over_later_higher_edge():
    quotes = {
        GREATER.ticker: [_q(16, 0.55, 0.57)],
        LESS.ticker: [_q(17, 0.55, 0.56)],
    }
    e = replay_event(EVENT, quotes, _g_plus(0.10), RuleConfig())
    assert e is not None and e.market.ticker == GREATER.ticker and e.t == _ts(16)


def test_cumulative_volume_counts_from_open_time():
    bars = [
        Bar(_ts(15), 0.55, 0.57, 30.0),
        Bar(_ts(17), 0.55, 0.57, 30.0),
    ]
    qs = grid_quotes(bars, [_ts(15), _ts(16), _ts(17)])
    assert [q.cum_volume for q in qs] == [30.0, 30.0, 60.0]
    assert qs[1].bid == 0.55  # forward-filled
    e = replay_event(EVENT, {GREATER.ticker: qs}, _g_plus(0.10), RuleConfig())
    assert e is not None and e.t == _ts(17)


def test_quote_before_first_bar_is_missing():
    qs = grid_quotes([Bar(_ts(17), 0.55, 0.57, 100.0)], [_ts(16)])
    assert qs[0].bid is None and qs[0].cum_volume == 0.0


def test_min_volume_schedule():
    rule = RuleConfig(min_volume_fn=lambda t: 200.0 if t < _ts(18) else 50.0)
    quotes = {GREATER.ticker: [_q(16, 0.55, 0.57, 100), _q(18, 0.55, 0.57, 100)]}
    e = replay_event(EVENT, quotes, _g_plus(0.10), rule)
    assert e is not None and e.t == _ts(18)


def test_excluded_city_var():
    rule = RuleConfig(excluded=frozenset({"NYC_high"}))
    assert replay_event(EVENT, {GREATER.ticker: [_q(16, 0.55, 0.57)]}, _g_plus(0.1), rule) is None


def test_non_binary_result_is_skipped():
    scalar = _market("T80", "greater", result="scalar")
    ev = Event(EVENT.event_ticker, "KXHIGHNY", "NYC", "high", FD, (scalar,))
    assert (
        replay_event(ev, {scalar.ticker: [_q(16, 0.55, 0.57)]}, _g_plus(0.1), RuleConfig()) is None
    )


def _between(i: int) -> Market:
    return _market(f"B7{i}.5", "between", result="no")


def test_build_events_keeps_ladder_tails_only():
    ms = [GREATER, LESS, _between(4), _between(6)]
    events, drops = build_events(ms)
    assert len(events) == 1 and not drops
    assert {m.strike_type for m in events[0].tails} == {"greater", "less"}


def test_build_events_drops_single_threshold_and_null_tail_events():
    single = Market(
        "HIGHNY-22JAN01-T54", "HIGHNY-22JAN01", "KXHIGHNY", None, OPEN, OPEN, "yes", "hist"
    )
    null_tail = [
        Market(
            "KXHIGHMIA-25FEB05-T79",
            "KXHIGHMIA-25FEB05",
            "KXHIGHMIA",
            None,
            OPEN,
            OPEN,
            "yes",
            "hist",
        ),
        Market(
            "KXHIGHMIA-25FEB05-T72",
            "KXHIGHMIA-25FEB05",
            "KXHIGHMIA",
            "less",
            OPEN,
            OPEN,
            "no",
            "hist",
        ),
        Market(
            "KXHIGHMIA-25FEB05-B78.5",
            "KXHIGHMIA-25FEB05",
            "KXHIGHMIA",
            "between",
            OPEN,
            OPEN,
            "no",
            "hist",
        ),
        Market(
            "KXHIGHMIA-25FEB05-B76.5",
            "KXHIGHMIA-25FEB05",
            "KXHIGHMIA",
            "between",
            OPEN,
            OPEN,
            "no",
            "hist",
        ),
    ]
    one_between = [GREATER, LESS, _between(4)]
    events, drops = build_events([single, *null_tail])
    assert events == [] and drops == {"not_ladder": 2}
    events, drops = build_events(one_between)
    assert events == [] and drops == {"not_ladder": 1}


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("HIGHNY-22JAN01", date(2022, 1, 1)),
        ("KXHIGHTBOS-26FEB05", date(2026, 2, 5)),
        ("KXLOWTNYC-25DEC13", date(2025, 12, 13)),
        ("KXHIGHNY-BAD", None),
    ],
)
def test_parse_event_date(ticker, expected):
    assert parse_event_date(ticker) == expected


def _pooled(roi: float, lo: float) -> dict[str, float]:
    return {"roi": roi, "lo": lo}


def test_q1_verdict_table():
    assert q1_verdict(99, 80, _pooled(0.2, 0.1), [(2024, 0.1), (2025, 0.1)]) == "INCONCLUSIVE"
    assert q1_verdict(200, 59, _pooled(0.2, 0.1), [(2024, 0.1), (2025, 0.1)]) == "INCONCLUSIVE"
    assert q1_verdict(200, 80, _pooled(-0.01, -0.1), [(2024, 0.1), (2025, 0.1)]) == "FAIL"
    assert q1_verdict(200, 80, _pooled(0.2, 0.1), [(2024, 0.1), (2025, 0.1)]) == "PASS"
    assert q1_verdict(200, 80, _pooled(0.2, 0.1), [(2024, 0.1), (2025, -0.01)]) == "DECAYING"
    falling = [(2023, 0.3), (2024, 0.2), (2025, 0.1)]
    assert q1_verdict(200, 80, _pooled(0.2, 0.1), falling) == "DECAYING"
    assert q1_verdict(200, 80, _pooled(0.2, -0.05), [(2024, 0.1), (2025, 0.1)]) == "INCONCLUSIVE"
    assert q1_verdict(200, 80, _pooled(0.2, 0.1), [(2025, 0.1)]) == "PASS"


def _entry(t_hour: int = 16) -> Entry:
    return Entry(EVENT, GREATER, _ts(t_hour), 0.55, 0.57, 0.56, 0.66, 0.07, 500.0)


def test_q2_fill_definitions_and_hybrid_cross():
    e = _entry()
    at_bid = [Trade(_ts(16) + 300, False, 0.55, 5)]
    through = [Trade(_ts(16) + 300, False, 0.54, 5)]
    taker_buy = [Trade(_ts(16) + 300, True, 0.54, 5)]
    minute = [Bar(_ts(16) + 600, 0.56, 0.58, 0.0)]
    g = _g_plus(0.10)
    s = q2_signal(e, at_bid, minute, 10, g, maker_fee_on=False)
    assert s["fill_opt"] and not s["fill_pes"]
    assert s["h_wag"] == pytest.approx(0.58)  # crossed at the ask observed at T
    s = q2_signal(e, through, minute, 10, g, maker_fee_on=False)
    assert s["fill_pes"] and s["h_pnl"] == pytest.approx(1 - 0.55)
    s = q2_signal(e, taker_buy, minute, 10, g, maker_fee_on=False)
    assert not s["fill_opt"]
    s = q2_signal(e, [], minute, 10, lambda mid: 0.0, maker_fee_on=False)
    assert not s["h_traded"] and s["h_pnl"] == 0.0


def test_q2_last_bar_entry_crosses_immediately():
    e = _entry(t_hour=24)
    s = q2_signal(e, [], [], 60, _g_plus(0.10), maker_fee_on=False)
    assert s["h_traded"] and s["h_wag"] == pytest.approx(e.ask)


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.content = b"{}"
        self.text = "{}"

    def json(self) -> dict:
        return {"ok": 1}


class _FakeHttp:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses

    def get(self, url: str) -> _Resp:
        return _Resp(self.statuses.pop(0))


def _client(tmp_path, monkeypatch, statuses: list[int]):
    import scripts.research.favorite_bias_oos_g0 as mod

    sleeps: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", sleeps.append)
    monkeypatch.setattr(mod, "REQUEST_LOG", tmp_path / "requests.jsonl")
    c = mod.G0Client(raw_dir=tmp_path / "raw")
    c._http = _FakeHttp(statuses)  # type: ignore[assignment]
    return mod, c, sleeps


def test_client_backs_off_then_recovers(tmp_path, monkeypatch):
    mod, c, sleeps = _client(tmp_path, monkeypatch, [429, 429, 429, 429, 200])
    assert c.get("/historical/cutoff") == {"ok": 1}
    assert [s for s in sleeps if s >= 15] == [15.0, 30.0, 60.0, 120.0]
    assert c.get("/historical/cutoff") == {"ok": 1}  # served from cache, no network


def test_client_aborts_when_still_limited_after_max_backoff(tmp_path, monkeypatch):
    mod, c, _ = _client(tmp_path, monkeypatch, [429] * 5)
    with pytest.raises(mod.RateLimitAbortError):
        c.get("/historical/cutoff")


def test_client_rejects_disallowed_paths(tmp_path, monkeypatch):
    _mod, c, _ = _client(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError):
        c.get("/portfolio/orders")
