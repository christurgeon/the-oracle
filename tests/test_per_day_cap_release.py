"""Tests for the per-day cap's commit/release API and cancel-aware hydration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from python.strategies.per_day_event_cap import (
    PerDayEventCap,
    hydrate_per_day_state_from_trades_jsonl,
)

# ----- helpers ---------------------------------------------------------------


@dataclass
class _Sig:
    market_ticker: str
    effective_edge: float
    confidence: float
    metadata: dict = field(default_factory=dict)


def _sig(
    ticker: str,
    *,
    city: str = "LAX",
    var: str = "low",
    forecast_date: str = "2026-05-29",
    effective_edge: float = 0.2,
    confidence: float = 1.0,
) -> _Sig:
    return _Sig(
        market_ticker=ticker,
        effective_edge=effective_edge,
        confidence=confidence,
        metadata={
            "city": city,
            "market_type": var,
            "forecast_date": forecast_date,
            "side": "YES",
            "strike_type": "between",
            "floor_strike": 55,
            "cap_strike": 56,
        },
    )


def _write_trades(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _placement_row(
    order_id: str,
    *,
    city: str = "LAX",
    var: str = "low",
    forecast_date: str = "2026-05-29",
    ts: str = "2026-05-28T14:14:15+00:00",
    strategy: str = "weather_edge",
    mode: str = "live",
    ticker: str = "KXLOWTLAX-26MAY29-B55.5",
) -> dict:
    return {
        "event": "placement",
        "order_id": order_id,
        "timestamp": ts,
        "mode": mode,
        "strategy": strategy,
        "market_ticker": ticker,
        "metadata": {"city": city, "market_type": var, "forecast_date": forecast_date},
    }


def _cancel_row(
    order_id: str,
    *,
    city: str = "LAX",
    var: str = "low",
    forecast_date: str = "2026-05-29",
    ts: str = "2026-05-28T14:16:32+00:00",
    strategy: str = "weather_edge",
    mode: str = "live",
    ticker: str = "KXLOWTLAX-26MAY29-B55.5",
) -> dict:
    return {
        "event": "cancel",
        "order_id": order_id,
        "timestamp": ts,
        "mode": mode,
        "strategy": strategy,
        "market_ticker": ticker,
        "metadata": {"city": city, "market_type": var, "forecast_date": forecast_date},
    }


def _fill_row(
    order_id: str,
    *,
    city: str = "LAX",
    var: str = "low",
    forecast_date: str = "2026-05-29",
    ts: str = "2026-05-28T14:14:20+00:00",
    strategy: str = "weather_edge",
    mode: str = "live",
    ticker: str = "KXLOWTLAX-26MAY29-B55.5",
) -> dict:
    return {
        "event": "fill",
        "order_id": order_id,
        "timestamp": ts,
        "mode": mode,
        "strategy": strategy,
        "market_ticker": ticker,
        "metadata": {"city": city, "market_type": var, "forecast_date": forecast_date},
    }


# ----- commit / release unit tests -------------------------------------------


KEY_LAX_LOW_29 = ("LAX", "low", "2026-05-29")
KEY_MIA_HIGH_29 = ("MIA", "high", "2026-05-29")


def test_commit_increases_committed_count() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-1")
    kept, n_blocked = cap.filter_and_select(
        [_sig("KXLOWTLAX-26MAY29-B55.5")], scan_id="s1", scan_ts="t1"
    )
    assert kept == []
    assert n_blocked == 1


def test_commit_same_pair_is_idempotent() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-1")
    cap.commit(KEY_LAX_LOW_29, "oid-1")  # no-op
    kept, n_blocked = cap.filter_and_select(
        [_sig("KXLOWTLAX-26MAY29-B55.5")], scan_id="s1", scan_ts="t1"
    )
    assert n_blocked == 1


def test_commit_same_order_id_different_key_raises() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-1")
    with pytest.raises(ValueError, match="already committed"):
        cap.commit(KEY_MIA_HIGH_29, "oid-1")


def test_release_decrements_committed_count_and_unblocks_retry() -> None:
    """The primary bug-fix test at the cap level."""
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-original")
    released_key = cap.release("oid-original")
    assert released_key == KEY_LAX_LOW_29
    # A fresh signal for the same event must now pass.
    kept, n_blocked = cap.filter_and_select(
        [_sig("KXLOWTLAX-26MAY29-B55.5")], scan_id="s2", scan_ts="t2"
    )
    assert len(kept) == 1
    assert n_blocked == 0


def test_release_unknown_order_id_returns_none() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    assert cap.release("never-committed") is None


def test_release_already_released_returns_none() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-1")
    cap.release("oid-1")
    assert cap.release("oid-1") is None  # idempotent


def test_release_then_commit_same_event_succeeds() -> None:
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-original")
    cap.release("oid-original")
    cap.commit(KEY_LAX_LOW_29, "oid-retry")  # must not raise; slot released
    kept, n_blocked = cap.filter_and_select(
        [_sig("KXLOWTLAX-26MAY29-B55.5")], scan_id="s3", scan_ts="t3"
    )
    assert n_blocked == 1


def test_commit_with_max_none_is_noop() -> None:
    cap = PerDayEventCap(max_per_day=None, audit_writer=None)
    cap.commit(KEY_LAX_LOW_29, "oid-1")  # no-op, no exception
    assert cap.release("oid-1") is None


def test_filter_and_select_audit_committed_before_uses_new_state(
    tmp_path: Path,
) -> None:
    """Regression guard on per_day_event_cap.py:285 — audit row's
    `committed_before` field must reflect post-commit state (not legacy
    _spent_counts).
    """
    from python.strategies.weather_edge import PerDayCapBlockWriter

    audit_path = tmp_path / "blocks.jsonl"
    writer = PerDayCapBlockWriter(audit_path)
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.commit(KEY_LAX_LOW_29, "oid-1")
    # Now a fresh same-event signal should fire the block-audit row.
    cap.filter_and_select(
        [_sig("KXLOWTLAX-26MAY29-B55.5")], scan_id="s1", scan_ts="t1"
    )
    rows = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["committed_before"] == 1


def test_release_callback_safe_across_cap_reconstruction() -> None:
    """Spec §5.8: stale callback into a freshly-reconstructed cap is a no-op."""
    cap1 = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap1.commit(KEY_LAX_LOW_29, "oid-1")
    cap2 = PerDayEventCap(max_per_day=1, audit_writer=None)
    assert cap2.release("oid-1") is None
    assert cap2._committed_order_ids == {}


# ----- hydration tests -------------------------------------------------------


def test_hydration_placement_with_cancel_no_fill_is_released(tmp_path: Path) -> None:
    """The primary bug-fix test at the hydration level."""
    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_placement_row("oid-1"), _cancel_row("oid-1")])
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert KEY_LAX_LOW_29 not in committed


def test_hydration_placement_with_cancel_and_fill_stays_committed(
    tmp_path: Path,
) -> None:
    """Partial-fill case: a cancel row alongside a fill row for the same
    order_id does NOT release the slot (the fill took exposure)."""
    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _placement_row("oid-1"),
            _fill_row("oid-1"),
            _cancel_row("oid-1", ts="2026-05-28T14:16:32+00:00"),
        ],
    )
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert KEY_LAX_LOW_29 in committed
    assert "oid-1" in committed[KEY_LAX_LOW_29]


def test_hydration_orphan_cancel_is_ignored(tmp_path: Path) -> None:
    """Cancel for an order_id with no matching placement: no error, no
    negative count; the key simply doesn't appear in committed.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_cancel_row("orphan-1")])
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert committed == {}


def test_hydration_returns_sets_not_counts(tmp_path: Path) -> None:
    """API contract: hydration returns dict[EventKey, set[str]], not
    dict[EventKey, int]. Validates the new return-type signature.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_placement_row("oid-1")])
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert isinstance(committed[KEY_LAX_LOW_29], set)
    assert committed[KEY_LAX_LOW_29] == {"oid-1"}


def test_hydration_cancel_for_other_strategy_is_ignored(tmp_path: Path) -> None:
    """A cancel row for favorite_bias must not release a weather_edge slot
    (and vice versa). Each cap hydrates with its own relevant_strategy.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _placement_row("oid-we", strategy="weather_edge"),
            _placement_row("oid-fb", strategy="favorite_bias"),
            _cancel_row("oid-fb", strategy="favorite_bias"),
        ],
    )
    we_committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades, relevant_strategy="weather_edge"
    )
    fb_committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades, relevant_strategy="favorite_bias"
    )
    assert KEY_LAX_LOW_29 in we_committed
    assert "oid-we" in we_committed[KEY_LAX_LOW_29]
    assert KEY_LAX_LOW_29 not in fb_committed


def test_hydration_cancel_for_other_mode_is_ignored(tmp_path: Path) -> None:
    """A cancel row with mode='shadow' must not release a 'live' slot."""
    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _placement_row("oid-1", mode="live"),
            _cancel_row("oid-1", mode="shadow"),
        ],
    )
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    # Cancel is filtered out by mode gate → placement remains committed.
    assert KEY_LAX_LOW_29 in committed


def test_hydration_manual_close_does_not_release_committed_slot(
    tmp_path: Path,
) -> None:
    """Spec §5.3: manual_close is forensic-only; cap stays committed via
    the prior placement+fill.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _placement_row("oid-1"),
            _fill_row("oid-1"),
            {
                "event": "manual_close",
                "order_id": "oid-mc",
                "timestamp": "2026-05-28T15:00:00+00:00",
                "mode": "live",
                "strategy": "weather_edge",
                "market_ticker": "KXLOWTLAX-26MAY29-B55.5",
                "metadata": {
                    "city": "LAX",
                    "market_type": "low",
                    "forecast_date": "2026-05-29",
                },
            },
        ],
    )
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert KEY_LAX_LOW_29 in committed
    assert "oid-1" in committed[KEY_LAX_LOW_29]


def test_hydration_cash_out_does_not_release_committed_slot(
    tmp_path: Path,
) -> None:
    """Spec §5.3: cash_out is forensic-only; cap stays committed via
    the prior placement+fill.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _placement_row("oid-1"),
            _fill_row("oid-1"),
            {
                "event": "cash_out",
                "order_id": "oid-co",
                "timestamp": "2026-05-28T15:30:00+00:00",
                "mode": "live",
                "strategy": "weather_edge",
                "market_ticker": "KXLOWTLAX-26MAY29-B55.5",
                "metadata": {
                    "city": "LAX",
                    "market_type": "low",
                    "forecast_date": "2026-05-29",
                },
            },
        ],
    )
    committed, _ = hydrate_per_day_state_from_trades_jsonl(
        utc_today="2026-05-28", trades_path=trades
    )
    assert KEY_LAX_LOW_29 in committed


def test_maybe_reset_rebuilds_reverse_map(tmp_path: Path) -> None:
    """After hydration, release(order_id) must work via the rebuilt
    reverse map.
    """
    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_placement_row("oid-1")])
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)
    cap.maybe_reset(
        utc_today="2026-05-28",
        hydrate_fn=lambda d: hydrate_per_day_state_from_trades_jsonl(
            utc_today=d, trades_path=trades
        ),
    )
    # The reverse map should let us release oid-1 even though we didn't
    # call commit() this session.
    released_key = cap.release("oid-1")
    assert released_key == KEY_LAX_LOW_29


# ----- end-to-end ------------------------------------------------------------


def test_full_cycle_signal_cancel_resignal_in_same_utc_day() -> None:
    """End-to-end regression test for the original bug.

    1. filter_and_select returns a signal for LAX/low/29 → orchestrator commits.
    2. Second tick's filter_and_select blocks the same event (cap is 1).
    3. Engine observes 0-fill terminal status → on_cancel callback releases.
    4. Third tick's filter_and_select allows the same event again.
    """
    cap = PerDayEventCap(max_per_day=1, audit_writer=None)

    # Tick 1: signal generated, accepted, committed by orchestrator.
    sig1 = _sig("KXLOWTLAX-26MAY29-B55.5")
    kept1, n_blocked1 = cap.filter_and_select([sig1], scan_id="s1", scan_ts="t1")
    assert len(kept1) == 1
    cap.commit(KEY_LAX_LOW_29, "oid-1")

    # Tick 2: same event re-signaled while order is still pending → blocked.
    sig2 = _sig("KXLOWTLAX-26MAY29-B55.5")
    kept2, n_blocked2 = cap.filter_and_select([sig2], scan_id="s2", scan_ts="t2")
    assert n_blocked2 == 1

    # Engine fires the on_cancel callback after 120s TTL with 0 fills.
    cap.release("oid-1")

    # Tick 3: same event re-signaled — cap must now allow it.
    sig3 = _sig("KXLOWTLAX-26MAY29-B55.5")
    kept3, n_blocked3 = cap.filter_and_select([sig3], scan_id="s3", scan_ts="t3")
    assert n_blocked3 == 0
    assert len(kept3) == 1
