"""Tests for the reconciler per-fill backfill alert."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.execution.fills_reconciler import (
    _BACKFILL_ALERT_MAX_LINES,
    FillsReconciler,
    _format_backfill_fill_alert,
)
from python.notifications.base import AlertLevel


def _row(**over):
    base = {
        "side": "NO",
        "market_ticker": "KXHIGHNY-26MAY10-B55",
        "contracts": 5,
        "fill_price": 0.55,
        "metadata": {"strategy": "weather_edge"},
    }
    base.update(over)
    return base


class TestFormatBackfillFillAlert:
    def test_header_carries_reconciled_label_and_count(self):
        msg = _format_backfill_fill_alert([_row(), _row()])
        assert msg.splitlines()[0] == "BACKFILLED FILLS [reconciled] (2)"

    def test_line_format_matches_engine_batch(self):
        msg = _format_backfill_fill_alert([_row()])
        # Mirrors TelegramNotifier.send_trade_batch_alert:
        #   "  {side} {ticker} {n}x @ ${price:.2f} | {strategy}"
        assert "  NO KXHIGHNY-26MAY10-B55 5x @ $0.55 | weather_edge" in msg.splitlines()[1]

    def test_multi_fill_vwap_row_formats_summed_contracts(self):
        msg = _format_backfill_fill_alert([_row(contracts=6, fill_price=0.73)])
        assert "6x @ $0.73" in msg

    def test_missing_keys_do_not_raise(self):
        msg = _format_backfill_fill_alert([{}])
        assert msg.startswith("BACKFILLED FILLS [reconciled] (1)")
        assert "? ? 0x @ $0.00 | unknown" in msg

    def test_present_but_wrong_typed_values_do_not_raise(self):
        # The real raise vector: present-but-None / non-numeric values that
        # .get(default) does NOT catch, flowing into f"${x:.2f}".
        rows = [
            _row(fill_price=None, contracts="x", metadata=None, side=None),
        ]
        msg = _format_backfill_fill_alert(rows)
        assert "0x @ $0.00 | unknown" in msg
        assert "?" in msg  # side coerced to "?"

    def test_non_numeric_string_price_drives_float_guard(self):
        # fill_price="abc" is truthy → survives `or 0.0` → hits float() → ValueError.
        # Exercises the price try/except (fill_price=None short-circuits before it).
        msg = _format_backfill_fill_alert([_row(fill_price="abc")])
        assert "@ $0.00 |" in msg

    def test_infinite_contracts_degrades_not_raises(self):
        # int(float('inf')) raises OverflowError (not ValueError/TypeError) — the
        # inner guard must catch it so the row degrades to 0x rather than dropping
        # the whole message. (Unreachable from _build_synthetic_row, but the
        # formatter's "cannot raise" contract must hold regardless.)
        msg = _format_backfill_fill_alert([_row(contracts=float("inf"))])
        assert "0x @" in msg

    def test_line_cap_appends_and_more(self):
        rows = [_row() for _ in range(_BACKFILL_ALERT_MAX_LINES + 3)]
        msg = _format_backfill_fill_alert(rows)
        lines = msg.splitlines()
        # header + capped rows + "…and K more"
        assert len(lines) == 1 + _BACKFILL_ALERT_MAX_LINES + 1
        assert lines[-1] == "  …and 3 more"
        assert lines[0] == (f"BACKFILLED FILLS [reconciled] ({_BACKFILL_ALERT_MAX_LINES + 3})")

    def test_no_more_line_when_within_cap(self):
        msg = _format_backfill_fill_alert([_row(), _row()])
        assert "more" not in msg


def _make_reconciler(tmp_path, notifier):
    return FillsReconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
        trade_log_lock=asyncio.Lock(),
        kalshi_client=None,
        engine=None,
        notifier=notifier,
        clock=lambda: 1_700_000_000.0,
    )


class _ReturnsOneBuyFill:
    """A /portfolio/fills client returning one un-logged BUY fill."""

    def __init__(self, order_id="OID-MISSED"):
        self._order_id = order_id

    async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
        return {
            "fills": [
                {
                    "order_id": self._order_id,
                    "action": "buy",
                    "count_fp": "5",
                    "no_price_dollars": "0.55",
                    "yes_price_dollars": "0.45",
                    "fee_cost": "0.08",
                    "is_taker": True,
                    "side": "no",
                    "ticker": "KXHIGHNY-26MAY10-B55",
                    "market_ticker": "KXHIGHNY-26MAY10-B55",
                    "created_time": "2026-05-10T15:00:00Z",
                }
            ],
            "cursor": None,
        }


def _alert_calls(notifier):
    """[(message, level), ...] in call order.

    NOTE: production passes `message` positionally from `_safe_alert`
    (the new INFO path) but as a KEYWORD from `_alert_on_status` /
    `_handle_tick_failure` (the WARNING/CRITICAL health paths). Read both
    or `c.args[0]` raises IndexError on the keyword calls.
    """
    out = []
    for c in notifier.send_alert.call_args_list:
        msg = c.args[0] if c.args else c.kwargs.get("message")
        out.append((msg, c.kwargs.get("level")))
    return out


class TestBackfillAlertEmission:
    @pytest.mark.asyncio
    async def test_detail_alert_fires_after_warning(self, tmp_path):
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        r = _make_reconciler(tmp_path, notifier)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"
        r._kalshi_client = _ReturnsOneBuyFill()

        tick = await r.tick()
        assert tick.delta_count == 1

        calls = _alert_calls(notifier)
        levels = [lvl for _, lvl in calls]
        # Health WARNING preserved AND fires before the INFO detail.
        assert AlertLevel.WARNING in levels
        assert AlertLevel.INFO in levels
        assert levels.index(AlertLevel.WARNING) < levels.index(AlertLevel.INFO)
        info_msg = next(m for m, lvl in calls if lvl == AlertLevel.INFO)
        assert "[reconciled]" in info_msg
        assert "KXHIGHNY-26MAY10-B55" in info_msg
        assert "5x" in info_msg

    @pytest.mark.asyncio
    async def test_no_detail_alert_on_clean_tick(self, tmp_path):
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        r = _make_reconciler(tmp_path, notifier)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"

        class _NoFills:
            async def get_fills(self, **kwargs):
                return {"fills": [], "cursor": None}

        r._kalshi_client = _NoFills()
        tick = await r.tick()
        assert tick.delta_count == 0
        assert all(lvl != AlertLevel.INFO for _, lvl in _alert_calls(notifier))

    @pytest.mark.asyncio
    async def test_notifier_send_raises_does_not_break_tick(self, tmp_path):
        notifier = MagicMock()
        notifier.send_alert = AsyncMock(side_effect=RuntimeError("telegram down"))
        r = _make_reconciler(tmp_path, notifier)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"
        r._kalshi_client = _ReturnsOneBuyFill()

        tick = await r.tick()
        # Row still synthesised; tick accounting intact; no failure recorded.
        assert tick.delta_count == 1
        assert tick.exc_class is None
        assert r._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_formatter_raise_does_not_corrupt_tick(self, tmp_path, monkeypatch):
        """HIGH-finding regression: a formatter raise must be caught locally,
        NOT escape into _tick_impl's outer handler (which would skip
        _persist_state, bump _consecutive_failures, fire a spurious failure
        alert, and misreport the tick as delta=0/exc_class set)."""
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        r = _make_reconciler(tmp_path, notifier)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"
        r._kalshi_client = _ReturnsOneBuyFill()

        monkeypatch.setattr(
            "python.execution.fills_reconciler._format_backfill_fill_alert",
            MagicMock(side_effect=RuntimeError("formatter boom")),
        )

        tick = await r.tick()
        # True accounting preserved — the raise did NOT reach the outer handler.
        assert tick.delta_count == 1
        assert tick.exc_class is None
        assert r._consecutive_failures == 0
        # No INFO detail (it was skipped) — but the health WARNING still fired.
        levels = [lvl for _, lvl in _alert_calls(notifier)]
        assert AlertLevel.WARNING in levels
        assert AlertLevel.INFO not in levels

    @pytest.mark.asyncio
    async def test_notifier_none_is_noop(self, tmp_path):
        r = _make_reconciler(tmp_path, None)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"
        r._kalshi_client = _ReturnsOneBuyFill()
        tick = await r.tick()
        assert tick.delta_count == 1  # completes, no crash

    @pytest.mark.asyncio
    async def test_dedup_engine_written_oid_no_detail_alert(self, tmp_path):
        """An oid already in the log (engine wrote it) is in tracked.buys →
        not synthesised → no INFO detail alert."""
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        log_path = tmp_path / "trades.jsonl"
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "event": "fill",
                        "order_id": "OID-MISSED",
                        "mode": "live",
                        "fill_price": 0.55,
                        "contracts": 5,
                        "timestamp": "2026-05-10T15:00:00+00:00",
                    }
                )
                + "\n"
            )
        r = _make_reconciler(tmp_path, notifier)
        r._trade_log_path = log_path
        r._kalshi_client = _ReturnsOneBuyFill()
        tick = await r.tick()
        assert tick.delta_count == 0
        assert all(lvl != AlertLevel.INFO for _, lvl in _alert_calls(notifier))

    @pytest.mark.asyncio
    async def test_truncation_cofire_critical_and_info(self, tmp_path):
        """Truncated tick that also backfills a row emits BOTH the CRITICAL
        (backlog) and exactly one INFO listing the written subset."""
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        r = _make_reconciler(tmp_path, notifier)
        (tmp_path / "trades.jsonl").touch()
        r._trade_log_path = tmp_path / "trades.jsonl"

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [
                    {
                        "order_id": "OID-TRUNC",
                        "action": "buy",
                        "count_fp": "3",
                        "no_price_dollars": "0.50",
                        "yes_price_dollars": "0.50",
                        "fee_cost": "0.04",
                        "is_taker": True,
                        "side": "no",
                        "ticker": "KXHIGHNY-26MAY10-B55",
                        "market_ticker": "KXHIGHNY-26MAY10-B55",
                        "created_time": "2026-05-10T15:00:00Z",
                    }
                ],
                "pages_consumed": r._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "latest_created_time_unix": None,
            }

        r._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        tick = await r.tick()
        assert tick.truncated_at_max_pages is True
        assert tick.status == "critical"
        assert tick.delta_count == 1
        calls = _alert_calls(notifier)
        info_msgs = [m for m, lvl in calls if lvl == AlertLevel.INFO]
        assert len(info_msgs) == 1
        assert "[reconciled]" in info_msgs[0]
        assert any(lvl == AlertLevel.CRITICAL for _, lvl in calls)
