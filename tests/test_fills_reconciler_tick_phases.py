"""Characterization tests for FillsReconciler._tick_impl.

These pin the tick's observable contract (lock window, call order, status and
cursor matrix, alert ordering, log payloads, failure path) so the method can be
decomposed without behavior change. Every expectation here was recorded from
the pre-refactor implementation.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import python.execution.fills_reconciler as fr
from python.notifications.base import AlertLevel
from tests.conftest import make_minimal_reconciler

_LOGGER_NAME = "python.execution.fills_reconciler"
_NOW = 1_700_000_000
_CURSOR0 = 1_699_999_000
_ADVANCED = _NOW - 120
_SENTINEL_CLEAN_TS = 5


def _buy(oid: str, *, ticker: str = "T", count: int = 5, ts: str = "2026-05-10T15:00:00Z"):
    return {
        "order_id": oid,
        "action": "buy",
        "count_fp": str(count),
        "no_price_dollars": "0.55",
        "yes_price_dollars": "0.45",
        "fee_cost": "0.04",
        "is_taker": True,
        "side": "no",
        "ticker": ticker,
        "market_ticker": ticker,
        "created_time": ts,
    }


def _unknown(oid: str) -> dict[str, Any]:
    fill = _buy(oid)
    fill["action"] = "hold"
    return fill


def _make(
    tmp_path: Path,
    fills: list[dict[str, Any]],
    *,
    truncated: bool = False,
    cycle: bool = False,
    pages: int = 1,
):
    rec = make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    rec._clock = lambda: float(_NOW)
    rec._last_safe_ts_unix = _CURSOR0
    rec._last_clean_completion_ts_unix = _SENTINEL_CLEAN_TS

    async def fetch(*, min_ts: int) -> dict[str, Any]:
        return {
            "fills": fills,
            "fills_count": len(fills),
            "pages_consumed": pages,
            "truncated": truncated,
            "cycle_aborted": cycle,
        }

    rec._fetch_fills_paginated = fetch
    return rec


def _alerts(rec) -> list[tuple[Any, str]]:
    return list(rec._notifier.sent_alerts)


def _tick_record(caplog) -> logging.LogRecord:
    records = [r for r in caplog.records if r.getMessage() == "reconciler_tick"]
    assert len(records) == 1
    return records[0]


def _log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER_NAME)


# ---------------------------------------------------------------------------
# Status / cursor matrix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("truncated", "cycle", "unknown", "delta"),
    list(itertools.product([False, True], [False, True], [False, True], [0, 1, 4])),
)
async def test_status_and_cursor_matrix(tmp_path, caplog, truncated, cycle, unknown, delta):
    fills = [_buy(f"B{i}") for i in range(delta)]
    if unknown:
        fills.append(_unknown("U1"))
    rec = _make(tmp_path, fills, truncated=truncated, cycle=cycle, pages=3)

    tick = await rec.tick()

    clean = delta == 0 and not truncated and not cycle
    if clean:
        status = "clean"
    elif delta > 3 or truncated or cycle:
        status = "critical"
    else:
        status = "warning"
    held = truncated or unknown
    cursor_after = _CURSOR0 if held else _ADVANCED

    assert tick.status == status
    assert tick.delta_count == delta
    assert tick.pages_consumed == 3
    assert tick.truncated_at_max_pages is truncated
    assert tick.cycle_aborted_at_cursor is cycle
    assert tick.last_safe_ts_unix_before == _CURSOR0
    assert tick.last_safe_ts_unix_after == cursor_after
    assert tick.order_ids_synthesised == [f"B{i}" for i in range(delta)]
    assert rec._last_safe_ts_unix == cursor_after
    assert rec._last_clean_completion_ts_unix == (_NOW if clean else _SENTINEL_CLEAN_TS)

    persisted = json.loads(rec._state_path.read_text())
    assert persisted["last_safe_ts_unix"] == cursor_after
    assert persisted["last_clean_completion_ts_unix"] == (_NOW if clean else _SENTINEL_CLEAN_TS)

    extra = _tick_record(caplog)
    assert extra.alert_state == status
    assert extra.last_safe_ts_unix_before == _CURSOR0
    assert extra.last_safe_ts_unix_after == cursor_after
    assert extra.delta_count == delta
    assert extra.fills_fetched == len(fills)


# ---------------------------------------------------------------------------
# Lock window and call order.
# ---------------------------------------------------------------------------


_SPIED = (
    "_fetch_fills_paginated",
    "_flush_pending_startup_alerts",
    "_build_placement_index",
    "_load_tracked_oids",
    "_build_open_positions_by_ticker_side",
    "_safe_resolve_strategy",
    "_build_synthetic_row",
    "_apply_fill_kind",
    "_safe_alert",
    "_append_synthetic_rows",
    "_compute_next_cursor",
    "_alert_on_status",
    "_persist_state",
)


def _spy_all(rec, seq: list[tuple[str, bool]]) -> None:
    import inspect

    for name in _SPIED:
        orig = getattr(rec, name)
        if inspect.iscoroutinefunction(orig):

            def make_async(orig=orig, name=name):
                async def wrapper(*a, **k):
                    seq.append((name, rec._trade_log_lock.locked()))
                    return await orig(*a, **k)

                return wrapper

            setattr(rec, name, make_async())
        else:

            def make_sync(orig=orig, name=name):
                def wrapper(*a, **k):
                    seq.append((name, rec._trade_log_lock.locked()))
                    return orig(*a, **k)

                return wrapper

            setattr(rec, name, make_sync())


@pytest.mark.asyncio
async def test_lock_window_and_call_order_with_unknown(tmp_path):
    rec = _make(tmp_path, [_buy("A"), _unknown("U")])
    seq: list[tuple[str, bool]] = []
    _spy_all(rec, seq)
    await rec.tick()
    assert seq == [
        ("_fetch_fills_paginated", False),
        ("_flush_pending_startup_alerts", True),
        ("_build_placement_index", True),
        ("_load_tracked_oids", True),
        ("_build_open_positions_by_ticker_side", True),
        ("_safe_resolve_strategy", True),
        ("_build_synthetic_row", True),
        ("_apply_fill_kind", True),
        ("_safe_alert", True),
        ("_append_synthetic_rows", True),
        ("_alert_on_status", False),
        ("_safe_alert", False),
        ("_persist_state", False),
    ]


@pytest.mark.asyncio
async def test_lock_window_and_call_order_cursor_advance(tmp_path):
    rec = _make(tmp_path, [_buy("A")])
    seq: list[tuple[str, bool]] = []
    _spy_all(rec, seq)
    await rec.tick()
    assert seq == [
        ("_fetch_fills_paginated", False),
        ("_flush_pending_startup_alerts", True),
        ("_build_placement_index", True),
        ("_load_tracked_oids", True),
        ("_build_open_positions_by_ticker_side", True),
        ("_safe_resolve_strategy", True),
        ("_build_synthetic_row", True),
        ("_append_synthetic_rows", True),
        ("_compute_next_cursor", False),
        ("_alert_on_status", False),
        ("_safe_alert", False),
        ("_persist_state", False),
    ]


@pytest.mark.asyncio
async def test_no_append_and_no_cursor_call_when_held_and_empty(tmp_path):
    rec = _make(tmp_path, [], truncated=True)
    seq: list[tuple[str, bool]] = []
    _spy_all(rec, seq)
    await rec.tick()
    assert [n for n, _ in seq] == [
        "_fetch_fills_paginated",
        "_flush_pending_startup_alerts",
        "_build_placement_index",
        "_load_tracked_oids",
        "_build_open_positions_by_ticker_side",
        "_alert_on_status",
        "_persist_state",
    ]


@pytest.mark.asyncio
async def test_alert_ordering_startup_dispatch_status_backfill(tmp_path):
    rec = _make(tmp_path, [_unknown("U"), _buy("A")], truncated=True)
    rec._pending_startup_alerts.append((AlertLevel.CRITICAL, "startup-corrupt"))
    await rec.tick()
    alerts = _alerts(rec)
    assert [lvl for lvl, _ in alerts] == [
        AlertLevel.CRITICAL,
        AlertLevel.CRITICAL,
        AlertLevel.CRITICAL,
        AlertLevel.INFO,
    ]
    assert alerts[0][1] == "startup-corrupt"
    assert alerts[1][1] == (
        "Reconciler classifier UNKNOWN for T close_oid=U; manual review needed."
    )
    assert alerts[2][1] == (
        "Fills reconciler: pagination truncated at max_pages_per_tick (delta=1); backlog may exist."
    )
    rows = _log_rows(rec._trade_log_path)
    assert alerts[3][1] == fr._format_backfill_fill_alert(rows)
    assert rec._pending_startup_alerts == []


# ---------------------------------------------------------------------------
# Candidate selection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_fills_cap_defers_sorted_tail(tmp_path, caplog):
    fills = [_buy(f"O{i}") for i in (5, 3, 1, 4, 2)]
    rec = _make(tmp_path, fills)
    rec._max_fills_per_tick = 3
    tick = await rec.tick()
    assert tick.order_ids_synthesised == ["O1", "O2", "O3"]
    assert [r["order_id"] for r in _log_rows(rec._trade_log_path)] == ["O1", "O2", "O3"]
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        "Reconciler over max_fills_per_tick (5 > 3); deferring 2 candidates to next tick"
    ]
    extra = _tick_record(caplog)
    assert extra.classifier_decision_counts["buy_entry"] == 3


@pytest.mark.asyncio
async def test_max_fills_cap_not_hit_at_exact_limit(tmp_path, caplog):
    rec = _make(tmp_path, [_buy(f"O{i}") for i in range(3)])
    rec._max_fills_per_tick = 3
    tick = await rec.tick()
    assert tick.delta_count == 3
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


@pytest.mark.asyncio
async def test_candidates_exclude_all_three_tracked_sets(tmp_path, caplog):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in [
                {"event": "fill", "order_id": "T1", "mode": "live"},
                {"event": "manual_close", "order_id": "M1", "mode": "live"},
                {"event": "cash_out", "order_id": "C1", "mode": "live"},
                {"event": "fill", "order_id": "S1", "mode": "shadow"},
            ]
        )
    )
    rec = _make(tmp_path, [_buy(o) for o in ("T1", "M1", "C1", "S1", "N1")])
    tick = await rec.tick()
    assert tick.order_ids_synthesised == ["N1", "S1"]
    extra = _tick_record(caplog)
    assert extra.classifier_decision_counts["buy_entry"] == 2
    assert extra.fills_fetched == 5


@pytest.mark.asyncio
async def test_grouping_skips_missing_order_id_and_uses_first_fill_as_sample(tmp_path, caplog):
    first = _buy("A", count=5, ts="2026-05-10T15:00:05Z")
    second = _buy("A", count=7, ts="2026-05-10T15:00:01Z")
    second["action"] = "hold"
    no_oid = _buy("X")
    del no_oid["order_id"]
    none_oid = _buy("Y")
    none_oid["order_id"] = None
    rec = _make(tmp_path, [first, no_oid, second, none_oid])
    tick = await rec.tick()
    assert tick.order_ids_synthesised == ["A"]
    (row,) = _log_rows(rec._trade_log_path)
    assert row["contracts"] == 12
    assert row["timestamp"] == "2026-05-10T15:00:01Z"
    extra = _tick_record(caplog)
    assert extra.classifier_decision_counts["buy_entry"] == 1
    assert extra.classifier_decision_counts["unknown"] == 0
    assert extra.fills_fetched == 4


# ---------------------------------------------------------------------------
# Dispatch: ordering, tallies, log payload dict order.
# ---------------------------------------------------------------------------


def _manual_close_rec(tmp_path: Path, fills: list[dict[str, Any]]):
    rec = _make(tmp_path, fills)
    rec._trade_log_path.write_text(
        json.dumps(
            {
                "event": "fill",
                "order_id": "seed",
                "position_key": "K",
                "market_ticker": "K",
                "fill_price": 0.40,
                "contracts": 10,
                "fees_from_api": 0.05,
                "mode": "live",
                "metadata": {"strategy": "weather_edge"},
            }
        )
        + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K": 4.0}
    risk.strategy_for_position.return_value = "unknown"
    rec._engine._risk = risk
    rec._manual_close_enabled = True
    return rec, risk


def _sell(oid: str) -> dict[str, Any]:
    return {
        "order_id": oid,
        "action": "sell",
        "side": "yes",
        "yes_price_dollars": 0.65,
        "no_price_dollars": 0.35,
        "count_fp": 10,
        "fee_cost": 0.04,
        "is_taker": True,
        "created_time": "2026-05-17T12:34:56Z",
        "market_ticker": "K",
    }


@pytest.mark.asyncio
async def test_interleaved_dispatch_in_sorted_order(tmp_path, caplog):
    fills = [_unknown("D-unk"), _buy("C-buy"), _sell("B-sell"), _buy("A-buy")]
    rec, risk = _manual_close_rec(tmp_path, fills)

    tick = await rec.tick()

    assert tick.order_ids_synthesised == ["A-buy", "B-sell", "C-buy"]
    assert tick.delta_count == 2
    assert tick.status == "warning"
    assert tick.last_safe_ts_unix_after == _CURSOR0
    assert tick.strategy_resolution_breakdown == {
        "in_memory": 0,
        "placement_scan": 0,
        "unresolved": 2,
    }
    rows = _log_rows(rec._trade_log_path)
    assert [(r["event"], r["order_id"]) for r in rows] == [
        ("fill", "seed"),
        ("manual_close", "B-sell"),
        ("fill", "A-buy"),
        ("fill", "C-buy"),
    ]
    risk.close_position.assert_called_once()
    extra = _tick_record(caplog)
    assert list(extra.classifier_decision_counts.items()) == [
        ("buy_entry", 2),
        ("buy_cash_out", 0),
        ("sell_bot", 0),
        ("sell_bot_anomaly", 0),
        ("sell_manual_candidate", 1),
        ("unknown", 1),
    ]
    assert list(extra.apply_path_outcome_counts.items()) == [
        ("APPLIED", 1),
        ("UNKNOWN", 1),
    ]
    assert extra.order_ids_synthesised == ["A-buy", "B-sell", "C-buy"]
    assert extra.manual_close_realized_pnl_cents_total == 241
    assert extra.manual_close_apply_latency_seconds > 0


@pytest.mark.asyncio
async def test_resolution_breakdown_preseeded_order_then_custom_methods(tmp_path, caplog):
    rec = _make(tmp_path, [_buy("A"), _buy("B"), _buy("C")])
    methods = {"A": "zeta", "B": "in_memory", "C": "alpha"}
    rec._strategy_resolver = lambda oid, fill: ("s", methods[oid])
    tick = await rec.tick()
    assert list(tick.strategy_resolution_breakdown.items()) == [
        ("in_memory", 1),
        ("placement_scan", 0),
        ("unresolved", 0),
        ("zeta", 1),
        ("alpha", 1),
    ]
    extra = _tick_record(caplog)
    assert list(extra.strategy_resolution_breakdown.items()) == list(
        tick.strategy_resolution_breakdown.items()
    )


@pytest.mark.asyncio
async def test_empty_tick_payload_defaults(tmp_path, caplog):
    rec = _make(tmp_path, [])
    rec._manual_close_realized_pnl_cents_this_tick = 99
    rec._manual_close_apply_latency_max_this_tick = 9.0
    rec._co_realized_pnl_cents_this_tick = 99
    rec._co_apply_latency_max_this_tick = 9.0
    rec._co_applies_this_tick = 4
    tick = await rec.tick()
    assert tick.status == "clean"
    assert tick.strategy_resolution_breakdown == {
        "in_memory": 0,
        "placement_scan": 0,
        "unresolved": 0,
    }
    extra = _tick_record(caplog)
    assert list(extra.classifier_decision_counts.keys()) == [
        "buy_entry",
        "buy_cash_out",
        "sell_bot",
        "sell_bot_anomaly",
        "sell_manual_candidate",
        "unknown",
    ]
    assert all(v == 0 for v in extra.classifier_decision_counts.values())
    assert extra.apply_path_outcome_counts == {}
    assert extra.manual_close_realized_pnl_cents_total == 0
    assert extra.manual_close_apply_latency_seconds == 0.0
    assert extra.co_realized_pnl_cents_this_tick == 0
    assert extra.co_apply_latency_seconds is None
    assert extra.co_applies_this_tick == 0
    assert extra.consecutive_failures == 0
    assert extra.pending_partial_sells_count == 0


# ---------------------------------------------------------------------------
# Backfill detail alert.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_alert_follows_status_alert_for_warning(tmp_path):
    rec = _make(tmp_path, [_buy("A")])
    await rec.tick()
    alerts = _alerts(rec)
    assert [lvl for lvl, _ in alerts] == [AlertLevel.WARNING, AlertLevel.INFO]
    assert alerts[0][1] == (
        "Fills reconciler: backfilled 1 missing fill row(s); local trade log was "
        "missing fills present in /portfolio/fills."
    )
    assert alerts[1][1] == fr._format_backfill_fill_alert(_log_rows(rec._trade_log_path))


@pytest.mark.asyncio
async def test_backfill_formatter_failure_is_contained(tmp_path, caplog, monkeypatch):
    def boom(rows):
        raise ValueError("formatter bug")

    monkeypatch.setattr(fr, "_format_backfill_fill_alert", boom)
    rec = _make(tmp_path, [_buy("A")])
    rec._consecutive_failures = 2
    tick = await rec.tick()
    assert tick.status == "warning"
    assert tick.delta_count == 1
    assert tick.exc_class is None
    assert rec._consecutive_failures == 0
    assert rec._state_path.exists()
    assert [lvl for lvl, _ in _alerts(rec)] == [AlertLevel.WARNING]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert [r.getMessage() for r in errors] == [
        "backfill alert formatter raised; skipping detail alert"
    ]
    assert errors[0].exc_info is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["", None])
async def test_backfill_empty_message_sends_nothing(tmp_path, monkeypatch, message):
    monkeypatch.setattr(fr, "_format_backfill_fill_alert", lambda rows: message)
    rec = _make(tmp_path, [_buy("A")])
    await rec.tick()
    assert [lvl for lvl, _ in _alerts(rec)] == [AlertLevel.WARNING]


@pytest.mark.asyncio
async def test_backfill_formatter_not_called_without_notifier(tmp_path, monkeypatch):
    calls: list[Any] = []
    monkeypatch.setattr(fr, "_format_backfill_fill_alert", lambda rows: calls.append(rows))
    rec = _make(tmp_path, [_buy("A")])
    rec._notifier = None
    tick = await rec.tick()
    assert tick.delta_count == 1
    assert calls == []


@pytest.mark.asyncio
async def test_no_backfill_alert_without_synthetic_rows(tmp_path, monkeypatch):
    calls: list[Any] = []
    monkeypatch.setattr(fr, "_format_backfill_fill_alert", lambda rows: calls.append(rows))
    rec = _make(tmp_path, [])
    await rec.tick()
    assert calls == []
    assert _alerts(rec) == []


# ---------------------------------------------------------------------------
# Success-path state handling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_failures_reset_after_persist(tmp_path, caplog):
    rec = _make(tmp_path, [])
    rec._consecutive_failures = 2
    seen: list[int] = []
    orig = rec._persist_state

    async def spy() -> None:
        seen.append(rec._consecutive_failures)
        await orig()

    rec._persist_state = spy
    await rec.tick()
    assert seen == [2]
    assert json.loads(rec._state_path.read_text())["consecutive_failures"] == 2
    assert rec._consecutive_failures == 0
    assert _tick_record(caplog).consecutive_failures == 0


@pytest.mark.asyncio
async def test_status_alert_sees_clean_timestamp_already_set(tmp_path):
    rec = _make(tmp_path, [])
    captured: list[tuple[str, int]] = []
    orig = rec._alert_on_status

    async def spy(**kwargs: Any) -> None:
        captured.append((kwargs["status"], rec._last_clean_completion_ts_unix))
        assert set(kwargs) == {"status", "delta_count", "truncated", "cycle_aborted"}
        await orig(**kwargs)

    rec._alert_on_status = spy
    await rec.tick()
    assert captured == [("clean", _NOW)]


# ---------------------------------------------------------------------------
# Failure path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_path_order_and_payload(tmp_path, caplog):
    rec = _make(tmp_path, [])
    counter = itertools.count()
    rec._clock = lambda: _NOW + next(counter) * 0.5

    async def fetch(*, min_ts: int) -> dict[str, Any]:
        raise RuntimeError("boom")

    rec._fetch_fills_paginated = fetch
    seen: dict[str, Any] = {}
    orig_handle = rec._handle_tick_failure

    async def handle(exc: Exception) -> None:
        seen["failures_at_handle"] = rec._consecutive_failures
        seen["tick_logged_before_handle"] = any(
            r.getMessage() == "reconciler_tick" for r in caplog.records
        )
        await orig_handle(exc)

    persisted: list[int] = []

    async def persist() -> None:
        persisted.append(1)

    rec._handle_tick_failure = handle
    rec._persist_state = persist
    rec._manual_close_realized_pnl_cents_this_tick = 7

    tick = await rec.tick()

    assert seen == {"failures_at_handle": 1, "tick_logged_before_handle": False}
    assert persisted == []
    assert tick.status == "warning"
    assert tick.delta_count == 0
    assert tick.duration_ms == 500
    assert tick.pages_consumed == 0
    assert tick.exc_class == "RuntimeError"
    assert tick.exc_repr == "RuntimeError('boom')"
    assert tick.order_ids_synthesised == []
    assert tick.strategy_resolution_breakdown == {}
    assert tick.last_safe_ts_unix_before == _CURSOR0
    assert tick.last_safe_ts_unix_after == _CURSOR0
    assert rec._consecutive_failures == 1
    assert rec._manual_close_realized_pnl_cents_this_tick == 0
    assert _alerts(rec) == [
        (
            AlertLevel.WARNING,
            "Fills reconciler: 1 consecutive failures — last error: RuntimeError: boom",
        )
    ]
    extra = _tick_record(caplog)
    assert extra.alert_state == "warning"
    assert extra.consecutive_failures == 1
    assert extra.exc_class == "RuntimeError"
    assert extra.exc_repr == "RuntimeError('boom')"
    assert extra.classifier_decision_counts == {}
    assert extra.apply_path_outcome_counts == {}
    assert extra.strategy_resolution_breakdown == {}
    assert extra.order_ids_synthesised == []
    assert extra.pages_consumed == 0
    assert extra.fills_fetched == 0
    assert extra.truncated_at_max_pages is False
    assert extra.cycle_aborted_at_cursor is False
    assert extra.last_safe_ts_unix_before == _CURSOR0
    assert extra.last_safe_ts_unix_after == _CURSOR0
    assert extra.duration_ms == 500


@pytest.mark.asyncio
async def test_append_failure_holds_cursor_and_skips_persist(tmp_path, caplog):
    rec = _make(tmp_path, [_buy("A")])

    async def fail(rows: list[dict[str, Any]]) -> None:
        raise OSError("disk full")

    rec._append_synthetic_rows = fail
    tick = await rec.tick()
    assert tick.status == "warning"
    assert tick.delta_count == 0
    assert tick.exc_class == "OSError"
    assert tick.last_safe_ts_unix_after == _CURSOR0
    assert rec._last_safe_ts_unix == _CURSOR0
    assert rec._last_clean_completion_ts_unix == _SENTINEL_CLEAN_TS
    assert not rec._state_path.exists()
    assert rec._consecutive_failures == 1
    assert _alerts(rec) == [
        (
            AlertLevel.WARNING,
            "Fills reconciler: 1 consecutive failures — last error: OSError: disk full",
        )
    ]
    extra = _tick_record(caplog)
    assert extra.last_safe_ts_unix_after == _CURSOR0
    assert extra.delta_count == 0


@pytest.mark.asyncio
async def test_persist_failure_reports_advanced_cursor_and_keeps_counter(tmp_path, caplog):
    rec = _make(tmp_path, [])
    rec._consecutive_failures = 0

    async def fail() -> None:
        raise OSError("read-only fs")

    rec._persist_state = fail
    tick = await rec.tick()
    assert tick.status == "warning"
    assert tick.exc_class == "OSError"
    assert tick.delta_count == 0
    assert tick.last_safe_ts_unix_before == _CURSOR0
    assert tick.last_safe_ts_unix_after == _ADVANCED
    assert rec._last_safe_ts_unix == _ADVANCED
    assert rec._consecutive_failures == 1
    assert _tick_record(caplog).last_safe_ts_unix_after == _ADVANCED


@pytest.mark.asyncio
async def test_failure_counter_increments_across_ticks_without_notifier(tmp_path):
    rec = _make(tmp_path, [])
    rec._notifier = None

    async def fetch(*, min_ts: int) -> dict[str, Any]:
        raise RuntimeError("boom")

    rec._fetch_fills_paginated = fetch
    for expected in (1, 2, 3):
        tick = await rec.tick()
        assert tick.status == "warning"
        assert rec._consecutive_failures == expected
