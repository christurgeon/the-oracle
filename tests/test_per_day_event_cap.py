"""Tests for PerDayEventCap and its hydration helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python.strategies.per_day_event_cap import (
    PerDayEventCap,
    hydrate_per_day_state_from_trades_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_hydration_dedups_placement_and_fill_for_same_order(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    rows = [
        {
            "event": "placement",
            "order_id": "ord-1",
            "mode": "live",
            "strategy": "weather_edge",
            "timestamp": "2026-05-20T14:00:00+00:00",
            "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-21"},
            "market_ticker": "KXHIGHNY-26MAY21-B72.5",
        },
        {
            "event": "fill",
            "order_id": "ord-1",
            "mode": "live",
            "strategy": "weather_edge",
            "timestamp": "2026-05-20T14:00:05+00:00",
            "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-21"},
            "market_ticker": "KXHIGHNY-26MAY21-B72.5",
        },
    ]
    _write_jsonl(path, rows)

    counts, first_commit = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    key = ("NYC", "high", "2026-05-21")
    assert len(counts[key]) == 1  # not 2 — same order_id
    assert first_commit[key]["ticker"] == "KXHIGHNY-26MAY21-B72.5"
    assert first_commit[key]["source"] == "placement"


def test_hydration_ignores_shadow_rows(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "placement",
                "order_id": "ord-1",
                "mode": "shadow",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:00+00:00",
                "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-21"},
                "market_ticker": "KXHIGHNY-26MAY21-B72.5",
            }
        ],
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert counts == {}


def test_hydration_excludes_prior_day_rows(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "fill",
                "order_id": "ord-1",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-19T23:50:00+00:00",
                "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-20"},
                "market_ticker": "KXHIGHNY-26MAY20-B72.5",
            }
        ],
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert counts == {}


def test_hydration_falls_back_to_ticker_for_sparse_metadata(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "fill",
                "order_id": "ord-1",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:05+00:00",
                "market_ticker": "KXHIGHCHI-26MAY21-B58.5",
                "metadata": {"strategy": "weather_edge", "reconciled": True},
            }
        ],
    )
    counts, first_commit = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert {k: len(v) for k, v in counts.items()} == {("CHI", "high", "2026-05-21"): 1}
    assert first_commit[("CHI", "high", "2026-05-21")]["source"] == "fill"


def test_hydration_normalizes_ny_high_to_nyc(tmp_path: Path) -> None:
    """NYC high tickers use the `KXHIGHNY` stem (no C) but the cap stores
    city as 'NYC'. Ticker parser must normalize."""
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "fill",
                "order_id": "ord-1",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:05+00:00",
                "market_ticker": "KXHIGHNY-26MAY21-B72.5",
                "metadata": {"strategy": "weather_edge"},
            }
        ],
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert ("NYC", "high", "2026-05-21") in counts


def test_hydration_handles_nyc_low_stem(tmp_path: Path) -> None:
    """NYC low tickers use `KXLOWTNYC` (with C) — the stem differs from
    high. Verify the asymmetry is handled."""
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "fill",
                "order_id": "ord-1",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:05+00:00",
                "market_ticker": "KXLOWTNYC-26MAY21-B48.5",
                "metadata": {"strategy": "weather_edge"},
            }
        ],
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert ("NYC", "low", "2026-05-21") in counts


def test_hydration_handles_sea_high_stem(tmp_path: Path) -> None:
    """SEA high uses `KXHIGHTSEA` (with T), NOT `KXHIGHSEA`."""
    path = tmp_path / "trades.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "fill",
                "order_id": "ord-1",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:05+00:00",
                "market_ticker": "KXHIGHTSEA-26MAY21-B78.5",
                "metadata": {"strategy": "weather_edge"},
            }
        ],
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert ("SEA", "high", "2026-05-21") in counts


class _FakeWriter:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def write(self, row: dict) -> None:
        self.rows.append(row)


def _make_signal(
    ticker: str,
    *,
    city: str = "NYC",
    var: str = "high",
    forecast_date: str = "2026-05-21",
    effective_edge: float = 0.10,
    confidence: float = 0.5,
    side: str = "NO",
    strike_type: str = "between",
    floor: int = 72,
    cap: int = 73,
) -> Any:
    from dataclasses import dataclass, field

    @dataclass
    class _Sig:
        market_ticker: str
        effective_edge: float
        confidence: float
        metadata: dict = field(default_factory=dict)

    return _Sig(
        market_ticker=ticker,
        effective_edge=effective_edge,
        confidence=confidence,
        metadata={
            "city": city,
            "market_type": var,
            "forecast_date": forecast_date,
            "side": side,
            "strike_type": strike_type,
            "floor_strike": floor,
            "cap_strike": cap,
        },
    )


def test_max1_second_scan_blocks_same_event(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))

    sig1 = _make_signal("KXHIGHNY-26MAY21-B72.5")
    kept, n_blocked = cap.filter_and_select(
        [sig1], scan_id="s1", scan_ts="2026-05-20T14:00:00+00:00"
    )
    assert len(kept) == 1
    assert n_blocked == 0
    # Simulate the orchestrator's post-execute_batch commit binding the
    # placed order_id to the event key — the API replaces the old
    # signal-time pre-emptive register() with order_id-keyed commits.
    cap.commit(("NYC", "high", "2026-05-21"), f"oid-{kept[0].market_ticker}")

    sig2 = _make_signal("KXHIGHNY-26MAY21-B70.5", floor=70, cap=71)
    kept2, n_blocked2 = cap.filter_and_select(
        [sig2], scan_id="s2", scan_ts="2026-05-20T14:15:00+00:00"
    )
    assert kept2 == []
    assert n_blocked2 == 1
    assert len(writer.rows) == 1
    row = writer.rows[0]
    assert row["reason"] == "PER_DAY_EVENT_CAP_HIT"
    assert row["sub_reason"] == "exhausted_prior_to_scan"
    assert row["max_per_day"] == 1
    assert row["committed_before"] == 1
    assert row["event_key"] == {"city": "NYC", "var": "high", "forecast_date": "2026-05-21"}
    assert row["blocked_signal"]["market_ticker"] == "KXHIGHNY-26MAY21-B70.5"


def test_max_null_disables_cap(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=None, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sigs = [_make_signal(f"KXHIGHNY-26MAY21-B{i}", floor=i, cap=i + 1) for i in (70, 72, 74)]
    kept, n_blocked = cap.filter_and_select(sigs, scan_id="s1", scan_ts="t")
    assert len(kept) == 3
    assert n_blocked == 0
    assert writer.rows == []


def test_constructor_rejects_max_zero() -> None:
    writer = _FakeWriter()
    try:
        PerDayEventCap(max_per_day=0, audit_writer=writer)
    except ValueError:
        return
    raise AssertionError("expected ValueError for max_per_day=0")


def test_different_events_not_blocked(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sig_a = _make_signal("KXHIGHNY-26MAY21-B72.5", city="NYC", var="high")
    sig_b = _make_signal("KXLOWTMIA-26MAY21-B72.5", city="MIA", var="low")
    kept, _ = cap.filter_and_select([sig_a, sig_b], scan_id="s1", scan_ts="t")
    assert len(kept) == 2


def test_utc_midnight_reset_clears_state(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sig = _make_signal("KXHIGHNY-26MAY21-B72.5")
    cap.filter_and_select([sig], scan_id="s1", scan_ts="t1")
    cap.commit(("NYC", "high", "2026-05-21"), "oid-day1")
    cap.maybe_reset(utc_today="2026-05-21", hydrate_fn=lambda d: ({}, {}))
    sig2 = _make_signal("KXHIGHNY-26MAY22-B72.5", forecast_date="2026-05-22")
    kept, n_blocked = cap.filter_and_select([sig2], scan_id="s2", scan_ts="t2")
    assert len(kept) == 1
    assert n_blocked == 0


def test_within_scan_topk_for_max2(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=2, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sigs = [
        _make_signal("KXHIGHNY-26MAY21-B70.5", effective_edge=0.05, floor=70, cap=71),
        _make_signal("KXHIGHNY-26MAY21-B72.5", effective_edge=0.20, floor=72, cap=73),
        _make_signal("KXHIGHNY-26MAY21-B74.5", effective_edge=0.10, floor=74, cap=75),
    ]
    kept, n_blocked = cap.filter_and_select(sigs, scan_id="s1", scan_ts="t1")
    kept_tickers = sorted(s.market_ticker for s in kept)
    assert kept_tickers == ["KXHIGHNY-26MAY21-B72.5", "KXHIGHNY-26MAY21-B74.5"]
    assert n_blocked == 1
    assert writer.rows[0]["sub_reason"] == "within_scan_excess"


def test_within_scan_dropouts_populate_backcompat_metadata(tmp_path: Path) -> None:
    """Existing daily report reads event_candidates_count /
    event_dropped_edges / event_dropped_tickers from the top keeper's
    metadata. The new cap must preserve this for within-scan drops."""
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sigs = [
        _make_signal("KXHIGHNY-26MAY21-B70.5", effective_edge=0.05, floor=70, cap=71),
        _make_signal("KXHIGHNY-26MAY21-B72.5", effective_edge=0.20, floor=72, cap=73),
        _make_signal("KXHIGHNY-26MAY21-B74.5", effective_edge=0.10, floor=74, cap=75),
    ]
    kept, _ = cap.filter_and_select(sigs, scan_id="s1", scan_ts="t1")
    assert len(kept) == 1
    top = kept[0]
    assert top.market_ticker == "KXHIGHNY-26MAY21-B72.5"
    assert top.metadata["event_candidates_count"] == 3
    assert sorted(top.metadata["event_dropped_tickers"]) == sorted(
        [
            "KXHIGHNY-26MAY21-B70.5",
            "KXHIGHNY-26MAY21-B74.5",
        ]
    )
    assert sorted(top.metadata["event_dropped_edges"]) == sorted([0.05, 0.10])


def test_max2_across_scans(tmp_path: Path) -> None:
    writer = _FakeWriter()
    cap = PerDayEventCap(max_per_day=2, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))
    sig_a = _make_signal("KXHIGHNY-26MAY21-B70.5", effective_edge=0.05, floor=70, cap=71)
    cap.filter_and_select([sig_a], scan_id="s1", scan_ts="t1")
    cap.commit(("NYC", "high", "2026-05-21"), "oid-a")
    sig_b = _make_signal("KXHIGHNY-26MAY21-B72.5", effective_edge=0.20, floor=72, cap=73)
    sig_c = _make_signal("KXHIGHNY-26MAY21-B74.5", effective_edge=0.10, floor=74, cap=75)
    kept, n_blocked = cap.filter_and_select([sig_b, sig_c], scan_id="s2", scan_ts="t2")
    assert [s.market_ticker for s in kept] == ["KXHIGHNY-26MAY21-B72.5"]
    assert n_blocked == 1
    cap.commit(("NYC", "high", "2026-05-21"), "oid-b")
    sig_d = _make_signal("KXHIGHNY-26MAY21-B76.5", effective_edge=0.30, floor=76, cap=77)
    kept3, n_blocked3 = cap.filter_and_select([sig_d], scan_id="s3", scan_ts="t3")
    assert kept3 == []
    assert n_blocked3 == 1


def test_hydration_tolerates_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "trades.jsonl"
    path.write_text(
        "this is not json\n"
        + json.dumps(
            {
                "event": "fill",
                "order_id": "ord-ok",
                "mode": "live",
                "strategy": "weather_edge",
                "timestamp": "2026-05-20T14:00:05+00:00",
                "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-21"},
                "market_ticker": "KXHIGHNY-26MAY21-B72.5",
            }
        )
        + "\n"
        + "\n"
    )
    counts, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-20",
        trades_path=path,
    )
    assert {k: len(v) for k, v in counts.items()} == {("NYC", "high", "2026-05-21"): 1}


def test_maybe_reset_swallows_hydrate_fn_exception(caplog) -> None:
    import logging

    cap = PerDayEventCap(max_per_day=1, audit_writer=_FakeWriter())

    def _broken_hydrate(date_str):
        raise OSError("trades.jsonl unreadable")

    with caplog.at_level(logging.WARNING):
        cap.maybe_reset(utc_today="2026-05-21", hydrate_fn=_broken_hydrate)

    # Must not raise. State falls back to empty.
    assert cap._committed_order_ids == {}
    assert cap._first_commit == {}
    # Repeated call on same day must not retry hydration (idempotent).
    cap.maybe_reset(utc_today="2026-05-21", hydrate_fn=_broken_hydrate)
    assert "trades.jsonl unreadable" in caplog.text


def test_series_city_matches_canonical_ticker_dates_map() -> None:
    from python.strategies.per_day_event_cap import _SERIES_CITY
    from python.utils.ticker_dates import SERIES_PREFIX_TO_CITY_VAR

    assert set(_SERIES_CITY) == set(SERIES_PREFIX_TO_CITY_VAR), (
        "_SERIES_CITY drifted from canonical map; sync the two"
    )


def test_rejection_reason_and_funnel_exist() -> None:
    from python.analytics.audit_log import (
        RejectionReason,
        ScanFunnel,
        reason_numeric,
    )

    assert RejectionReason.PER_DAY_EVENT_CAP_HIT.value == "per_day_event_cap_hit"
    funnel = ScanFunnel()
    assert funnel.n_per_day_cap_blocked == 0
    funnel.n_per_day_cap_blocked += 3
    assert funnel.n_per_day_cap_blocked == 3
    assert reason_numeric(RejectionReason.PER_DAY_EVENT_CAP_HIT) != 0
