"""Tests for FillsReconciler (spec §11)."""

from __future__ import annotations

import asyncio
import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import OrderStatus, Side
from python.execution.fills_reconciler import (
    STATE_FILE_SCHEMA_VERSION,
    ApplyOutcome,
    BackfillResult,
    ClassifierContext,
    FillKind,
    FillsReconciler,
    ReconcileTick,
    StrategyResolver,
    TrackedOids,
    classify_fill,
)
from python.notifications.base import AlertLevel


class TestApplyOutcomeEnum:
    """ApplyOutcome StrEnum must stay byte-identical to the old string sentinels
    so existing on-disk `apply_path_outcome_counts` keys and `outcome ==` asserts
    keep working."""

    def test_members_equal_their_string_value(self):
        assert ApplyOutcome.APPLIED == "APPLIED"
        assert ApplyOutcome.APPLIED_PARTIAL_CASH_OUT == "APPLIED_PARTIAL_CASH_OUT"
        assert ApplyOutcome.SHAKEDOWN_SUPPRESSED == "SHAKEDOWN_SUPPRESSED"
        # Distinct from FillKind's lowercase values — no collision.
        assert ApplyOutcome.UNKNOWN == "UNKNOWN"
        assert ApplyOutcome.UNKNOWN != FillKind.UNKNOWN  # "UNKNOWN" vs "unknown"

    def test_dict_key_and_json_are_bare_strings(self):
        # Keyed by the enum, looked up by raw string (hash equality), and
        # json-serialized to the bare string — the on-disk contract.
        counts: dict = {}
        counts[ApplyOutcome.APPLIED] = counts.get(ApplyOutcome.APPLIED, 0) + 1
        assert counts["APPLIED"] == 1
        assert json.dumps(counts) == '{"APPLIED": 1}'
        assert json.dumps(ApplyOutcome.APPLIED_FULL_CASH_OUT) == '"APPLIED_FULL_CASH_OUT"'


class TestReconcileTick:
    def test_skipped_classmethod(self):
        tick = ReconcileTick.skipped(reason="reentrant")
        assert tick.status == "skipped"
        assert tick.reason == "reentrant"
        assert tick.delta_count == 0
        assert tick.duration_ms == 0

    def test_clean_tick(self):
        tick = ReconcileTick(status="clean", delta_count=0, duration_ms=42)
        assert tick.status == "clean"
        assert tick.delta_count == 0
        assert tick.duration_ms == 42
        # Default fields populated
        assert tick.pages_consumed == 0
        assert tick.truncated_at_max_pages is False
        assert tick.reason is None
        assert tick.exc_class is None

    def test_critical_tick_with_exception(self):
        tick = ReconcileTick(
            status="critical",
            delta_count=0,
            duration_ms=100,
            exc_class="RuntimeError",
            exc_repr="RuntimeError('boom')",
        )
        assert tick.exc_class == "RuntimeError"
        assert tick.exc_repr == "RuntimeError('boom')"

    def test_frozen(self):
        """ReconcileTick is frozen — assignment raises."""
        tick = ReconcileTick(status="clean", delta_count=0, duration_ms=0)
        with pytest.raises((AttributeError, TypeError)):
            tick.status = "warning"  # type: ignore[misc]


class TestBackfillResult:
    def test_basic_construction(self):
        result = BackfillResult(
            planned_writes=[{"order_id": "abc"}],
            written_count=1,
            skipped_count=0,
            unresolved_count=0,
            aborted=False,
            abort_reason=None,
        )
        assert result.written_count == 1
        assert result.aborted is False
        assert result.abort_reason is None

    def test_frozen(self):
        """BackfillResult is frozen — assignment raises."""
        result = BackfillResult(
            planned_writes=[],
            written_count=0,
            skipped_count=0,
            unresolved_count=0,
            aborted=True,
            abort_reason="dry_run",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.aborted = False  # type: ignore[misc]


class TestStrategyResolverProtocol:
    def test_protocol_signature(self):
        """A class satisfying the (str, dict[str, Any]) -> tuple[str, str]
        signature is a valid StrategyResolver. Note: Protocol is static-only
        (not @runtime_checkable), so this assertion is validated by mypy at
        check-time rather than by pytest at runtime."""

        class Stub:
            def __call__(
                self, order_id: str, kalshi_fill: dict[str, Any]
            ) -> tuple[str, str]:
                return ("weather_edge", "in_memory")

        s: StrategyResolver = Stub()
        assert s("oid-1", {}) == ("weather_edge", "in_memory")


# ---------------------------------------------------------------------------
# S8: FillsReconciler.__init__ + state file persist (Tasks 13–14)
# ---------------------------------------------------------------------------


class _NoopClient:
    """Minimal stub Kalshi client for state-file tests."""

    async def get_fills(self, **kwargs):
        return {"fills": [], "cursor": None}


class _MockNotifier:
    """In-memory notifier capturing alerts for assertions."""

    def __init__(self):
        self.alerts_sent: list[tuple] = []

    async def send_alert(self, *, level, message: str, **kwargs):
        Alert = namedtuple("Alert", ["level", "message"])
        self.alerts_sent.append(Alert(level, message))


class _CycleClient:
    """Stub Kalshi client whose get_fills always returns a non-None cursor
    ("LOOP") with no fills — i.e. a /portfolio/fills cursor cycle. Used to
    exercise the cursor-cycle defense in _fetch_fills_paginated. (The one
    cycle test that needs a fill on the first page defines its own local
    variant closing over ``single_fill``.)"""

    async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
        return {"fills": [], "cursor": "LOOP"}


def _make_minimal_reconciler(tmp_path):
    """Helper for tests that don't need a real Kalshi client or engine."""
    return FillsReconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
        trade_log_lock=asyncio.Lock(),
        kalshi_client=_NoopClient(),
        engine=None,
        notifier=None,
        clock=lambda: 1_700_000_000.0,
    )


class TestStateFileRecovery:
    def test_missing_state_file_falls_back_to_safety_window(self, tmp_path):
        """Spec §8 row 4: missing state file → cursor = now - startup_safety_window."""

        def _clock():
            return 1_700_000_000.0

        reconciler = FillsReconciler(
            state_path=tmp_path / "fills_reconciler.json",
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=None,
            startup_safety_window_seconds=24 * 3600,
            clock=_clock,
        )
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 24 * 3600
        assert reconciler._consecutive_failures == 0
        assert reconciler._pending_startup_alerts == []

    def test_corrupt_state_file_falls_back_and_queues_alert(self, tmp_path):
        """Spec §8 row 5: corrupt state file → fallback + queued WARNING."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text("not valid json {{{")

        notifier = _MockNotifier()
        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=notifier,
            startup_safety_window_seconds=24 * 3600,
            clock=lambda: 1_700_000_000.0,
        )
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 24 * 3600
        # Alert queued; not yet flushed (no event loop in __init__)
        assert len(reconciler._pending_startup_alerts) == 1
        level, message = reconciler._pending_startup_alerts[0]
        assert "corrupt" in message.lower()

    def test_future_schema_version_falls_back(self, tmp_path):
        """Unknown schema_version → same fallback as corruption."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text(
            json.dumps({"schema_version": 999, "last_safe_ts_unix": 100})
        )

        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=None,
            startup_safety_window_seconds=24 * 3600,
            clock=lambda: 1_700_000_000.0,
        )
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 24 * 3600

    def test_long_outage_cursor_uses_persisted(self, tmp_path):
        """Spec review #13: when persisted cursor is older than safety window,
        use the persisted value (i.e., min(persisted, floor)) so long-outage
        recovery reaches further back automatically."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "last_safe_ts_unix": 1_700_000_000 - 7 * 24 * 3600,
                    "last_clean_completion_ts_unix": 0,
                    "consecutive_failures": 0,
                    "last_critical_alert_ts_unix": 0,
                }
            )
        )

        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=None,
            startup_safety_window_seconds=24 * 3600,
            startup_max_window_seconds=30 * 24 * 3600,  # 30d cap, larger than 7d persisted
            clock=lambda: 1_700_000_000.0,
        )
        # Persisted (7d ago) is older than now-24h, so it wins.
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 7 * 24 * 3600

    def test_cursor_capped_at_startup_max_window(self, tmp_path):
        """Spec §5.1: a 30-day-old persisted cursor is capped at
        startup_max_window_seconds to prevent unbounded first-tick scans
        on a misconfigured or long-dormant deploy. The CLI is the escape
        hatch for older windows."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "last_safe_ts_unix": 1_700_000_000 - 30 * 24 * 3600,  # 30d ago
                    "last_clean_completion_ts_unix": 0,
                    "consecutive_failures": 0,
                    "last_critical_alert_ts_unix": 0,
                }
            )
        )

        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=None,
            startup_safety_window_seconds=24 * 3600,
            startup_max_window_seconds=7 * 24 * 3600,  # 7d cap
            clock=lambda: 1_700_000_000.0,
        )
        # Cap pulls cursor forward to 7d-ago, even though persisted is 30d-ago.
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 7 * 24 * 3600

    def test_corrupt_state_with_no_notifier_does_not_queue_alert(self, tmp_path):
        """Defensive: if notifier is None, the corrupt-state recovery path
        must NOT queue alerts (they'd never be flushed → memory leak). The
        guard is `if self._notifier is not None:` in _load_state."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text("not valid json {{{")

        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=None,  # explicit — no notifier
            startup_safety_window_seconds=24 * 3600,
            clock=lambda: 1_700_000_000.0,
        )
        # Cursor reset (recovery happened)
        assert reconciler._last_safe_ts_unix == 1_700_000_000 - 24 * 3600
        # But no alert queued (no notifier to flush to)
        assert reconciler._pending_startup_alerts == []

    @pytest.mark.asyncio
    async def test_corrupt_state_alert_flushes_on_first_tick(self, tmp_path):
        """Spec review N5: alerts queued during sync __init__ get flushed
        on the first tick() call. Without this, corrupt-state warnings
        would never reach the operator."""
        state_path = tmp_path / "fills_reconciler.json"
        state_path.write_text("not valid json {{{")

        notifier = _MockNotifier()
        reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=notifier,
            startup_safety_window_seconds=24 * 3600,
            clock=lambda: 1_700_000_000.0,
        )
        # Pre-tick: alert queued, not yet sent
        assert len(reconciler._pending_startup_alerts) == 1
        assert notifier.alerts_sent == []

        await reconciler.tick()

        # Post-tick: alert flushed and queue empty
        assert reconciler._pending_startup_alerts == []
        assert len(notifier.alerts_sent) == 1
        assert "corrupt" in notifier.alerts_sent[0].message.lower()


class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_persist_state_atomic_write_with_chmod(self, tmp_path):
        """_persist_state writes atomically and chmods to 0o600."""
        state_path = tmp_path / "fills_reconciler.json"
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._last_safe_ts_unix = 1_699_999_880
        reconciler._consecutive_failures = 2
        await reconciler._persist_state()

        loaded = json.loads(state_path.read_text())
        assert loaded["schema_version"] == STATE_FILE_SCHEMA_VERSION
        assert loaded["last_safe_ts_unix"] == 1_699_999_880
        assert loaded["consecutive_failures"] == 2

        # File mode should be 0o600 (matches state/risk_state.json convention)
        mode = state_path.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# S9: tick() guards, cursor advancement, tracked order IDs (Tasks 15–17)
# ---------------------------------------------------------------------------


class TestReentrancyAndDisable:
    @pytest.mark.asyncio
    async def test_reentrant_tick_is_skipped(self, tmp_path):
        """Spec §11 test 13. If _tick_lock is already held (a prior tick is
        in flight), tick() returns ReconcileTick.skipped(reason='reentrant')."""
        reconciler = _make_minimal_reconciler(tmp_path)
        await reconciler._tick_lock.acquire()
        try:
            tick = await reconciler.tick()
            assert tick.status == "skipped"
            assert tick.reason == "reentrant"
        finally:
            reconciler._tick_lock.release()

    @pytest.mark.asyncio
    async def test_disabled_flag_skips_tick(self, tmp_path):
        """Spec §11 test 14. /disable_reconciler sets _disabled=True;
        tick() returns ReconcileTick.skipped(reason='operator_disabled')."""
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._disabled = True
        tick = await reconciler.tick()
        assert tick.status == "skipped"
        assert tick.reason == "operator_disabled"

        # Re-enable: tick proceeds (returns clean from the stub _tick_impl)
        reconciler._disabled = False
        tick = await reconciler.tick()
        assert tick.status == "clean"


class TestCursorAdvancement:
    """Spec §5.1 cursor semantics — the load-bearing regression for #4."""

    def test_cursor_advances_to_now_minus_window(self, tmp_path):
        """The cursor floor is now - eventual_consistency_window_seconds.
        Cursor advances to this floor when it's greater than persisted."""
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._eventual_consistency_window_seconds = 120
        reconciler._last_safe_ts_unix = 1_699_999_000

        new_cursor = reconciler._compute_next_cursor(now=1_700_000_000)
        assert new_cursor == 1_700_000_000 - 120

    def test_cursor_never_regresses(self, tmp_path):
        """If now - window < persisted cursor (e.g., clock skew),
        keep the higher persisted value. Cursor monotonicity invariant."""
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._eventual_consistency_window_seconds = 120
        reconciler._last_safe_ts_unix = 1_700_000_000

        new_cursor = reconciler._compute_next_cursor(now=1_699_999_999)
        assert new_cursor == 1_700_000_000

    def test_late_arriving_fill_is_not_skipped(self, tmp_path):
        """REGRESSION TEST FOR SPEC #4. A fill with created_time = fill_ts is not
        yet queryable at fill_ts+30s; cursor must not advance past fill_ts."""
        fill_ts = 1_700_000_000
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._eventual_consistency_window_seconds = 120
        reconciler._last_safe_ts_unix = fill_ts - 300  # 5 min before fill

        new_cursor = reconciler._compute_next_cursor(now=fill_ts + 30)
        # Cursor advances to (fill_ts+30) - 120 = fill_ts - 90 — well behind the fill.
        # Next tick at fill_ts+130 will query min_ts=fill_ts-90 and pick up the fill.
        assert new_cursor == fill_ts - 90
        assert new_cursor < fill_ts  # the fill is still in scope


class TestTrackedOrderIds:
    def test_load_tracked_filters_event_and_mode(self, tmp_path):
        """Only event=='fill' AND mode=='live' rows count. Placement,
        settlement, shadow rows are excluded."""
        log_path = tmp_path / "trades.jsonl"
        rows = [
            {"event": "fill", "order_id": "F-1", "mode": "live",
             "timestamp": "2026-05-10T15:00:00+00:00"},
            {"event": "placement", "order_id": "P-1", "mode": "live",
             "timestamp": "2026-05-10T15:00:00+00:00"},
            {"event": "settlement", "order_id": "S-1", "mode": "live",
             "timestamp": "2026-05-10T15:00:00+00:00"},
            {"event": "fill", "order_id": "SH-1", "mode": "shadow",
             "timestamp": "2026-05-10T15:00:00+00:00"},
            {"event": "fill", "order_id": "F-2", "mode": "live",
             "timestamp": "2026-05-10T15:00:00+00:00"},
        ]
        with open(log_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path

        tracked = reconciler._load_tracked_live_order_ids()
        assert tracked == {"F-1", "F-2"}

    def test_load_tracked_skips_torn_lines(self, tmp_path):
        """Spec §8 row 3: JSONDecodeError lines are skipped, not fatal."""
        log_path = tmp_path / "trades.jsonl"
        with open(log_path, "w") as f:
            f.write(json.dumps(
                {"event": "fill", "order_id": "F-1", "mode": "live"}
            ) + "\n")
            f.write("not valid json {{{\n")
            f.write(json.dumps(
                {"event": "fill", "order_id": "F-2", "mode": "live"}
            ) + "\n")

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path

        tracked = reconciler._load_tracked_live_order_ids()
        assert tracked == {"F-1", "F-2"}


# ---------------------------------------------------------------------------
# S10: 3-step strategy resolver chain (Task 18)
# ---------------------------------------------------------------------------


class TestStrategyResolution:
    """Spec §11 test 12 / spec §6.4 — the 3-step lookup chain."""

    def test_step1_in_memory_lookup(self, tmp_path):
        """Step (a): order present in engine._orders → strategy from signal."""
        from types import SimpleNamespace

        mock_signal = SimpleNamespace(strategy_name="weather_edge")
        mock_record = SimpleNamespace(signal=mock_signal)

        class _MockEngine:
            _orders = {"OID-1": mock_record}

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._engine = _MockEngine()

        strategy, method = reconciler._default_strategy_resolver(
            "OID-1", {"order_id": "OID-1"}
        )
        assert strategy == "weather_edge"
        assert method == "in_memory"

    def test_step2_placement_row_scan(self, tmp_path):
        """Step (b): order absent from memory; scan placement rows."""
        log_path = tmp_path / "trades.jsonl"
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "event": "placement",
                        "order_id": "OID-2",
                        "mode": "live",
                        "timestamp": "2026-05-10T15:00:00+00:00",
                        "metadata": {"strategy": "ai_news"},
                    }
                )
                + "\n"
            )

        class _EmptyEngine:
            _orders: dict[str, Any] = {}

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._engine = _EmptyEngine()
        reconciler._trade_log_path = log_path
        # Pre-build the placement index (S12 does this once per tick)
        reconciler._build_placement_index()

        strategy, method = reconciler._default_strategy_resolver(
            "OID-2", {"order_id": "OID-2"}
        )
        assert strategy == "ai_news"
        assert method == "placement_scan"

    def test_step3_unresolved_fallback(self, tmp_path):
        """Step (c): neither path finds the order → unknown + unresolved."""

        class _EmptyEngine:
            _orders: dict[str, Any] = {}

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._engine = _EmptyEngine()
        # No placement row exists for OID-3
        reconciler._build_placement_index()

        strategy, method = reconciler._default_strategy_resolver(
            "OID-3", {"order_id": "OID-3"}
        )
        assert strategy == "unknown"
        assert method == "unresolved"

    def test_resolver_exception_returns_unresolved(self, tmp_path):
        """Spec §6.4 N7: if the resolver raises, the wrapper catches and
        returns ('unknown', 'unresolved'). The exception does NOT propagate
        out of tick()."""

        def _broken_resolver(
            order_id: str, kalshi_fill: dict[str, Any]
        ) -> tuple[str, str]:
            raise RuntimeError("resolver bug")

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._strategy_resolver = _broken_resolver

        strategy, method = reconciler._safe_resolve_strategy(
            "OID-X", {"order_id": "OID-X"}
        )
        assert strategy == "unknown"
        assert method == "unresolved"


# ---------------------------------------------------------------------------
# S11: paginated fetch + synthetic-row build (Tasks 19–20)
# ---------------------------------------------------------------------------


class TestPagination:
    """Spec §11 tests 5, 6 — paginated fetch with cap."""

    @pytest.mark.asyncio
    async def test_pagination_aggregates_across_pages(self, tmp_path):
        """Three pages then empty cursor → 30 fills aggregated."""

        calls: list[dict[str, Any]] = []

        class _PaginatingClient:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                calls.append({"min_ts": min_ts, "cursor": cursor})
                if cursor is None:
                    return {
                        "fills": [{"order_id": f"OID-{i}"} for i in range(10)],
                        "cursor": "PAGE2",
                    }
                if cursor == "PAGE2":
                    return {
                        "fills": [{"order_id": f"OID-{i}"} for i in range(10, 20)],
                        "cursor": "PAGE3",
                    }
                if cursor == "PAGE3":
                    return {
                        "fills": [{"order_id": f"OID-{i}"} for i in range(20, 30)],
                        "cursor": None,
                    }
                return {"fills": [], "cursor": None}

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _PaginatingClient()
        reconciler._page_delay_ms = 0  # speed up test

        result = await reconciler._fetch_fills_paginated(min_ts=42)
        assert result["fills_count"] == 30
        assert result["pages_consumed"] == 3
        assert result["truncated"] is False
        # Verify cursor threading: page 1 had cursor=None, page 2 had PAGE2, page 3 had PAGE3
        assert [c["cursor"] for c in calls] == [None, "PAGE2", "PAGE3"]
        # min_ts is forwarded unchanged on every page
        assert all(c["min_ts"] == 42 for c in calls)

    @pytest.mark.asyncio
    async def test_pagination_distinct_cursors_truncate_at_max_pages(self, tmp_path):
        """Mock returns DISTINCT cursors indefinitely — reconciler stops at
        max_pages_per_tick and reports truncation (not cycle abort)."""

        class _InfiniteUniqueClient:
            def __init__(self):
                self._i = 0

            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                self._i += 1
                return {
                    "fills": [{"order_id": f"OID-{self._i}"}],
                    "cursor": f"NEXT-{self._i}",
                }

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _InfiniteUniqueClient()
        reconciler._max_pages_per_tick = 3
        reconciler._page_delay_ms = 0

        result = await reconciler._fetch_fills_paginated(min_ts=0)
        assert result["pages_consumed"] == 3
        assert result["truncated"] is True
        assert result["cycle_aborted"] is False


class TestPaginationCycleDefense:
    """Spec §4.1 + §5.1 — _fetch_fills_paginated aborts when Kalshi returns
    a repeating cursor instead of running to max_pages_per_tick.

    Surfaced by the 2026-05-11 trade-log writer post-mortem.
    """

    @pytest.mark.asyncio
    async def test_immediate_cycle_aborts_after_one_repeat(self, tmp_path):
        """Server returns the same cursor on every call → abort on iter 2."""

        class _ImmediateCycleClient:
            def __init__(self):
                self.calls = 0

            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                self.calls += 1
                return {
                    "fills": [{"order_id": "OID-CYCLE"}],
                    "cursor": "LOOP",
                }

        client = _ImmediateCycleClient()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = client
        reconciler._max_pages_per_tick = 20
        reconciler._page_delay_ms = 0

        result = await reconciler._fetch_fills_paginated(min_ts=0)
        assert result["cycle_aborted"] is True
        assert result["truncated"] is False
        assert result["pages_consumed"] == 2
        assert client.calls == 2  # NOT 20

    @pytest.mark.asyncio
    async def test_a_b_a_cycle_aborts_at_repeat(self, tmp_path):
        """Cursors A → B → A → … must abort on the third call."""

        class _ABACycleClient:
            def __init__(self):
                self.calls = 0
                self._seq = ["A", "B", "A", "B", "A"]

            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                self.calls += 1
                next_cursor = self._seq[self.calls - 1]
                return {
                    "fills": [{"order_id": f"OID-{self.calls}"}],
                    "cursor": next_cursor,
                }

        client = _ABACycleClient()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = client
        reconciler._max_pages_per_tick = 20
        reconciler._page_delay_ms = 0

        result = await reconciler._fetch_fills_paginated(min_ts=0)
        assert result["cycle_aborted"] is True
        assert result["truncated"] is False
        assert result["pages_consumed"] == 3
        assert client.calls == 3

    @pytest.mark.asyncio
    async def test_no_cycle_terminates_normally(self, tmp_path):
        """Distinct cursors PAGE2 → PAGE3 → None must NOT trigger cycle abort."""

        class _NoCycleClient:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                if cursor is None:
                    return {"fills": [{"order_id": "A"}], "cursor": "PAGE2"}
                if cursor == "PAGE2":
                    return {"fills": [{"order_id": "B"}], "cursor": "PAGE3"}
                if cursor == "PAGE3":
                    return {"fills": [{"order_id": "C"}], "cursor": None}
                return {"fills": [], "cursor": None}

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _NoCycleClient()
        reconciler._max_pages_per_tick = 20
        reconciler._page_delay_ms = 0

        result = await reconciler._fetch_fills_paginated(min_ts=0)
        assert result["cycle_aborted"] is False
        assert result["truncated"] is False
        assert result["pages_consumed"] == 3
        assert result["fills_count"] == 3


class TestNoCycleInflation:
    """Spec §5.2 — regression lock for the 2026-05-11 20× inflation incident.

    With the cycle defense in _fetch_fills_paginated, a Kalshi server that
    returns a repeating cursor must NOT inflate the synthetic row's contracts,
    fees, or is_taker_fills.
    """

    @pytest.mark.asyncio
    async def test_cycle_does_not_inflate_synthetic_row(self, tmp_path):
        """One Kalshi fill (count=5, fee=$0.04) repeated under cursor cycle
        produces ONE synthetic row with contracts=5, fees=0.04 — not 20× any
        of those values."""

        single_fill = _make_fill("OID-LOOP")
        # _make_fill defaults: count_fp="5", fee_cost="0.04", is_taker=True,
        # created_time="2026-05-10T15:00:00Z" → unix 1778425200.

        class _CycleClient:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                return {"fills": [single_fill], "cursor": "LOOP"}

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _CycleClient()
        reconciler._page_delay_ms = 0
        # Strategy resolver chain needs a known answer for unresolved order_ids.
        reconciler._strategy_resolver = lambda oid, fill: ("weather_edge", "in_memory")

        # Use backfill(dry_run=True) to exercise _fetch_fills_paginated +
        # _build_synthetic_row end-to-end without writing to disk.
        # Window brackets the fill's parsed timestamp (1778425200).
        result = await reconciler.backfill(
            min_ts=1778400000, max_ts=1778500000, dry_run=True
        )

        # Exactly one planned write (cycle defense discarded the duplicate
        # page → only the first call's fills survived to _build_synthetic_row).
        assert len(result.planned_writes) == 1
        row = result.planned_writes[0]

        # No inflation. count_fp="5" → contracts=5 (NOT 10, NOT 100).
        assert row["contracts"] == 5
        # fee_cost="0.04" → fees_from_api=0.04 (NOT 0.08, NOT 0.80).
        assert row["fees_from_api"] == 0.04
        # Sum-of-contracts semantic: one taker fill of 5 contracts → is_taker_fills=5
        # (== contracts here). The cycle-inflation lock still holds: a duplicated
        # page would double this to 10, which is the value this test guards against.
        assert row["is_taker_fills"] == 5


class TestCycleAbortTelemetry:
    """Spec §5.3 — ReconcileTick exposes cycle_aborted_at_cursor field,
    propagated from _fetch_fills_paginated through _tick_impl."""

    @pytest.mark.asyncio
    async def test_tick_reports_cycle_aborted(self, tmp_path):
        """Cycle in /portfolio/fills → ReconcileTick.cycle_aborted_at_cursor=True."""

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _CycleClient()
        reconciler._page_delay_ms = 0

        tick = await reconciler.tick()
        assert tick.cycle_aborted_at_cursor is True
        assert tick.truncated_at_max_pages is False

    @pytest.mark.asyncio
    async def test_normal_tick_reports_no_cycle(self, tmp_path):
        """Clean tick → ReconcileTick.cycle_aborted_at_cursor=False."""

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _ReturnsFills([])
        reconciler._page_delay_ms = 0

        tick = await reconciler.tick()
        assert tick.cycle_aborted_at_cursor is False
        assert tick.status == "clean"

    @pytest.mark.asyncio
    async def test_cycle_abort_status_is_critical(self, tmp_path):
        """Spec §4.5 — cycle abort escalates status to critical."""

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _CycleClient()
        reconciler._page_delay_ms = 0

        tick = await reconciler.tick()
        assert tick.status == "critical"
        assert tick.cycle_aborted_at_cursor is True

    @pytest.mark.asyncio
    async def test_cycle_abort_alert_message_distinct_from_truncation(self, tmp_path):
        """Spec §4.5 — Telegram CRITICAL alert message names the cycle, not truncation."""

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._kalshi_client = _CycleClient()
        reconciler._page_delay_ms = 0

        await reconciler.tick()

        assert len(notifier.alerts_sent) == 1
        alert = notifier.alerts_sent[0]
        # Distinct copy from the truncation alert — assert the cycle-specific
        # phrase rather than absence of "truncated" so future copy edits that
        # legitimately mention truncation in passing don't break the test.
        assert "cursor cycle" in alert.message.lower()


class TestSyntheticRowBuild:
    """Spec §11 tests 2, 3 — single-fill and multi-fill (VWAP) rows."""

    def test_single_fill_synthesises_correctly(self, tmp_path):
        reconciler = _make_minimal_reconciler(tmp_path)
        kalshi_fills = [
            {
                "order_id": "OID-A",
                "ticker": "KXHIGHNY-26MAY10-B70.5",
                "market_ticker": "KXHIGHNY-26MAY10-B70.5",
                "side": "no",
                "count_fp": "12",
                "no_price_dollars": "0.65",
                "yes_price_dollars": "0.35",
                "fee_cost": "0.18",
                "is_taker": True,
                "created_time": "2026-05-10T15:00:00Z",
            }
        ]
        row = reconciler._build_synthetic_row(
            order_id="OID-A",
            kalshi_fills=kalshi_fills,
            strategy="weather_edge",
            resolution_method="in_memory",
        )
        assert row["event"] == "fill"
        assert row["order_id"] == "OID-A"
        assert row["mode"] == "live"
        assert row["fill_price"] == 0.65
        assert row["contracts"] == 12
        assert row["side"] == "NO"
        assert row["market_ticker"] == "KXHIGHNY-26MAY10-B70.5"
        assert row["metadata"]["strategy"] == "weather_edge"
        assert row["metadata"]["reconciled"] is True
        assert row["metadata"]["reconcile_source"] == "fills_reconciler"
        assert row["metadata"]["strategy_resolution"] == "in_memory"
        assert row["fees_from_api"] == 0.18

    def test_multi_fill_vwap_aggregation(self, tmp_path):
        """Spec §11 test 3 — three partial fills aggregate by VWAP.

        Fixture deliberately puts the earliest timestamp at index 1 (NOT the
        first fill) so the assertion `row["timestamp"] == earliest` actually
        validates that the loop's min-comparison fires — not just that the
        sample-0 fallback happens to be earliest.
        """
        reconciler = _make_minimal_reconciler(tmp_path)
        kalshi_fills = [
            {
                "order_id": "OID-B",
                "count_fp": "1",
                "no_price_dollars": "0.72",
                "yes_price_dollars": "0.28",
                "fee_cost": "0.02",
                "is_taker": True,
                "side": "no",
                "ticker": "TICKER",
                "market_ticker": "TICKER",
                "created_time": "2026-05-10T15:00:01Z",  # NOT earliest
            },
            {
                "order_id": "OID-B",
                "count_fp": "4",
                "no_price_dollars": "0.73",
                "yes_price_dollars": "0.27",
                "fee_cost": "0.06",
                "is_taker": True,
                "side": "no",
                "ticker": "TICKER",
                "market_ticker": "TICKER",
                "created_time": "2026-05-10T15:00:00Z",  # earliest, at index 1
            },
            {
                "order_id": "OID-B",
                "count_fp": "1",
                "no_price_dollars": "0.74",
                "yes_price_dollars": "0.26",
                "fee_cost": "0.02",
                "is_taker": False,
                "side": "no",
                "ticker": "TICKER",
                "market_ticker": "TICKER",
                "created_time": "2026-05-10T15:00:02Z",
            },
        ]
        row = reconciler._build_synthetic_row(
            order_id="OID-B",
            kalshi_fills=kalshi_fills,
            strategy="weather_edge",
            resolution_method="placement_scan",
        )
        # VWAP: (1*0.72 + 4*0.73 + 1*0.74) / 6 = 4.38/6 = 0.73
        assert row["contracts"] == 6
        assert row["fill_price"] == pytest.approx(0.73, rel=1e-3)
        assert row["fees_from_api"] == pytest.approx(0.10, rel=1e-3)
        # Sum-of-contracts: taker fills are 1 + 4 contracts → 5; maker fill is 1.
        assert row["is_taker_fills"] == 5
        assert row["is_maker_fills"] == 1
        # Earliest timestamp wins — pulled from index 1, not index 0.
        assert row["timestamp"] == "2026-05-10T15:00:00Z"


# ---------------------------------------------------------------------------
# S12: full _tick_impl with locked critical section (Task 21)
# ---------------------------------------------------------------------------


class TestTickFlow:
    """Spec §11 tests 1, 2, 4 — happy path, single missed, idempotency."""

    @pytest.mark.asyncio
    async def test_happy_path_no_missed_rows(self, tmp_path):
        """Engine has written the fill row; reconciler sees order_id in
        tracked set; emits no synthetic row. Spec §11 test 1."""
        log_path = tmp_path / "trades.jsonl"
        with open(log_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "event": "fill",
                        "order_id": "OID-A",
                        "mode": "live",
                        "fill_price": 0.65,
                        "contracts": 12,
                        "timestamp": "2026-05-10T15:00:00+00:00",
                    }
                )
                + "\n"
            )

        class _ReturnsOneFill:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                # Same order_id is already in the log
                return {
                    "fills": [
                        {
                            "order_id": "OID-A",
                            "count_fp": "12",
                            "no_price_dollars": "0.65",
                            "yes_price_dollars": "0.35",
                            "fee_cost": "0.18",
                            "is_taker": True,
                            "side": "no",
                            "ticker": "T",
                            "market_ticker": "T",
                            "created_time": "2026-05-10T15:00:00Z",
                        }
                    ],
                    "cursor": None,
                }

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _ReturnsOneFill()

        tick = await reconciler.tick()
        assert tick.status == "clean"
        assert tick.delta_count == 0

        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        # Log unchanged — only the original engine-written row
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_single_missed_row_synthesised(self, tmp_path):
        """Kalshi has a fill not in the local log → synthetic row written
        with metadata.reconciled=True. Spec §11 test 2."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()  # empty log

        class _ReturnsOneFill:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                return {
                    "fills": [
                        {
                            "order_id": "OID-MISSED",
                            "action": "buy",
                            "count_fp": "5",
                            "no_price_dollars": "0.55",
                            "yes_price_dollars": "0.45",
                            "fee_cost": "0.08",
                            "is_taker": True,
                            "side": "no",
                            "ticker": "TICKER",
                            "market_ticker": "TICKER",
                            "created_time": "2026-05-10T15:00:00Z",
                        }
                    ],
                    "cursor": None,
                }

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _ReturnsOneFill()

        tick = await reconciler.tick()
        # delta=1; alert_threshold_critical_delta default is 3; below threshold = warning
        assert tick.status == "warning"
        assert tick.delta_count == 1
        assert "OID-MISSED" in tick.order_ids_synthesised

        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["order_id"] == "OID-MISSED"
        assert rows[0]["event"] == "fill"
        assert rows[0]["metadata"]["reconciled"] is True

    @pytest.mark.asyncio
    async def test_idempotency_second_tick_no_op(self, tmp_path):
        """Run tick twice with the same fixture; second emits zero rows.
        Spec §11 test 4 — the dedup invariant."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()

        class _ReturnsOneFill:
            async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
                return {
                    "fills": [
                        {
                            "order_id": "OID-ONCE",
                            "action": "buy",
                            "count_fp": "3",
                            "no_price_dollars": "0.50",
                            "yes_price_dollars": "0.50",
                            "fee_cost": "0.04",
                            "is_taker": True,
                            "side": "no",
                            "ticker": "T",
                            "market_ticker": "T",
                            "created_time": "2026-05-10T15:00:00Z",
                        }
                    ],
                    "cursor": None,
                }

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _ReturnsOneFill()

        tick1 = await reconciler.tick()
        assert tick1.delta_count == 1

        tick2 = await reconciler.tick()
        assert tick2.delta_count == 0
        assert tick2.status == "clean"

    @pytest.mark.asyncio
    async def test_cursor_held_on_page_truncation(self, tmp_path):
        """When _fetch_fills_paginated reports truncated=True, the cursor
        must NOT advance — otherwise the abandoned pages are dropped
        forever. Code-quality regression: SF-1 in S12 review."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        cursor_before = reconciler._last_safe_ts_unix

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],  # delta=0 so we don't conflate with synth-row write
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "latest_created_time_unix": None,
            }

        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]

        tick = await reconciler.tick()

        assert tick.truncated_at_max_pages is True
        assert tick.status == "critical"
        # Cursor held — next tick will re-fetch the same window
        assert reconciler._last_safe_ts_unix == cursor_before
        assert tick.last_safe_ts_unix_after == cursor_before


# ---------------------------------------------------------------------------
# S13: Failure handling — API-degraded vs code-bug + cooldown (Task 22)
# ---------------------------------------------------------------------------


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_kalshi_api_error_stays_warning(self, tmp_path):
        """Spec §11 test 7. KalshiAPIError class — never escalates to CRITICAL."""
        from python.data.kalshi_client import KalshiAPIError

        class _FailingClient:
            async def get_fills(self, **kwargs):
                raise KalshiAPIError(429, {"message": "rate limited"})

        notifier = _MockNotifier()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _FailingClient()
        reconciler._notifier = notifier
        reconciler._alert_threshold_consecutive = 3

        # Five consecutive failures
        for _ in range(5):
            tick = await reconciler.tick()
            assert tick.status == "warning"

        # All alerts are WARNING; none are CRITICAL
        levels = [a.level.name for a in notifier.alerts_sent]
        assert "CRITICAL" not in levels
        assert "WARNING" in levels

    @pytest.mark.asyncio
    async def test_code_bug_escalates_to_critical_with_cooldown(self, tmp_path):
        """Spec §11 test 8. Generic exception escalates to CRITICAL on
        consecutive_failures >= 3; cooldown suppresses subsequent CRITICAL."""

        class _BuggyClient:
            async def get_fills(self, **kwargs):
                raise RuntimeError("code bug")

        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _BuggyClient()
        reconciler._notifier = notifier
        reconciler._alert_threshold_consecutive = 3
        reconciler._critical_alert_cooldown_seconds = 3600
        reconciler._clock = lambda: clock_value[0]

        # 3 consecutive failures → CRITICAL on the 3rd
        for _ in range(3):
            await reconciler.tick()

        critical_alerts = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(critical_alerts) == 1

        # Failures 4, 5 within cooldown — no new CRITICAL
        for _ in range(2):
            await reconciler.tick()
        critical_alerts = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(critical_alerts) == 1, "cooldown should suppress repeats"

        # Advance time past cooldown
        clock_value[0] += 3601
        await reconciler.tick()
        critical_alerts = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(critical_alerts) == 2, "after cooldown a new CRITICAL is allowed"


class TestStructuredLogging:
    """Spec §10.1 — every tick emits a stable structured log contract."""

    @pytest.mark.asyncio
    async def test_tick_emits_structured_log_with_required_fields(
        self, tmp_path, caplog
    ):
        """Spec §10.1 — every tick emits a structured log with the contract fields."""
        import logging

        reconciler = _make_minimal_reconciler(tmp_path)
        with caplog.at_level(logging.INFO):
            await reconciler.tick()

        tick_records = [r for r in caplog.records if r.message == "reconciler_tick"]
        assert len(tick_records) == 1
        rec = tick_records[0]
        assert hasattr(rec, "schema_version")
        assert rec.schema_version == 1
        assert hasattr(rec, "duration_ms")
        assert hasattr(rec, "delta_count")
        assert hasattr(rec, "fills_fetched")
        assert hasattr(rec, "pages_consumed")
        assert hasattr(rec, "alert_state")

    @pytest.mark.asyncio
    async def test_tick_emits_structured_log_on_failure(self, tmp_path, caplog):
        """Spec §10.1 — failure path also emits the reconciler_tick log
        with exc_class and alert_state populated."""
        import logging

        class _BuggyClient:
            async def get_fills(self, **kwargs):
                raise RuntimeError("boom")

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._kalshi_client = _BuggyClient()

        with caplog.at_level(logging.INFO):
            await reconciler.tick()

        tick_records = [r for r in caplog.records if r.message == "reconciler_tick"]
        assert len(tick_records) == 1
        rec = tick_records[0]
        assert rec.exc_class == "RuntimeError"
        assert rec.alert_state == "warning"

    @pytest.mark.asyncio
    async def test_tick_log_includes_cycle_aborted_at_cursor_key(self, tmp_path, caplog):
        """Spec §5.4 — reconciler_tick log row carries cycle_aborted_at_cursor."""

        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._kalshi_client = _CycleClient()
        reconciler._page_delay_ms = 0

        import logging
        with caplog.at_level(logging.INFO, logger="python.execution.fills_reconciler"):
            await reconciler.tick()

        tick_records = [
            r for r in caplog.records if r.message == "reconciler_tick"
        ]
        assert len(tick_records) >= 1
        # The extra payload appears as attributes on the LogRecord.
        assert getattr(tick_records[-1], "cycle_aborted_at_cursor", None) is True


class TestRaceRegression:
    """Spec §11 test 9 — the 404 race regression."""

    @pytest.mark.parametrize("fixture_date", ["2026-05-04", "2026-05-05"])
    @pytest.mark.asyncio
    async def test_404_race_recovered_by_reconciler(self, tmp_path, fixture_date):
        """For each historical missed order:
        1. Engine 'placed' the order (placement row exists).
        2. Engine's check_fills got 404 — no fill row written.
        3. Reconciler runs against /portfolio/fills which DOES return the fill.
        4. Synthetic fill row appears in the log with metadata.reconciled=True.
        """
        fixture_path = (
            Path(__file__).parent / "fixtures" / "fills_reconciler"
            / f"missed_orders_{fixture_date}.json"
        )
        kalshi_fills_for_missed = json.loads(fixture_path.read_text())
        # Multi-fill orders share an order_id — seed exactly one placement row
        # per unique OID so the placement-scan resolver has unambiguous input.
        unique_oid_to_fill: dict[str, dict[str, Any]] = {}
        for fill in kalshi_fills_for_missed:
            unique_oid_to_fill.setdefault(fill["order_id"], fill)

        log_path = tmp_path / "trades.jsonl"
        # Pre-existing placement rows for each (carries strategy attribution)
        with open(log_path, "w") as f:
            for fill in unique_oid_to_fill.values():
                placement_row = {
                    "event": "placement",
                    "order_id": fill["order_id"],
                    "market_ticker": fill["market_ticker"],
                    "side": fill["side"].upper(),
                    "mode": "live",
                    "timestamp": fill["created_time"],
                    "metadata": {"strategy": "weather_edge"},
                }
                f.write(json.dumps(placement_row) + "\n")

        # Mock client returns the missed fills
        class _MockClient:
            async def get_fills(self, **kwargs):
                return {"fills": kalshi_fills_for_missed, "cursor": None}

        reconciler = FillsReconciler(
            state_path=tmp_path / "fills_reconciler.json",
            trade_log_path=log_path,
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_MockClient(),
            engine=None,  # no in-memory orders — must resolve via placement scan
            notifier=_MockNotifier(),
            clock=lambda: 1_700_000_000.0,
        )

        # Pre-condition: zero fill rows for the missed orders
        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        fill_oids_before = {r["order_id"] for r in rows if r.get("event") == "fill"}
        for fill in kalshi_fills_for_missed:
            assert fill["order_id"] not in fill_oids_before

        # Run the reconciler
        tick = await reconciler.tick()
        assert tick.delta_count == len(unique_oid_to_fill)

        # Post-condition: every missed order_id now has exactly one fill row
        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        for fill in kalshi_fills_for_missed:
            matching = [r for r in rows
                        if r.get("event") == "fill" and r["order_id"] == fill["order_id"]]
            assert len(matching) == 1, f"expected 1 fill row for {fill['order_id']}"
            assert matching[0]["metadata"]["reconciled"] is True
            assert matching[0]["metadata"]["strategy"] == "weather_edge"
            assert matching[0]["metadata"]["strategy_resolution"] == "placement_scan"


# ---------------------------------------------------------------------------
# S16: FillsReconciler.backfill() operator-driven path (Task 28)
# ---------------------------------------------------------------------------


class TestBackfill:
    @pytest.mark.asyncio
    async def test_backfill_dry_run_returns_planned_writes(self, tmp_path):
        """Dry-run does not mutate state but returns the plan."""
        reconciler = _make_minimal_reconciler(tmp_path)
        # Mock client returns 2 fills not in the log
        class _Client:
            async def get_fills(self, **kwargs):
                return {"fills": [
                    {"order_id": "OID-1", "count_fp": "5",
                     "no_price_dollars": "0.55", "fee_cost": "0.08",
                     "is_taker": True, "side": "no",
                     "ticker": "T", "market_ticker": "T",
                     "created_time": "2026-05-04T15:00:00Z"},
                    {"order_id": "OID-2", "count_fp": "10",
                     "no_price_dollars": "0.60", "fee_cost": "0.15",
                     "is_taker": True, "side": "no",
                     "ticker": "U", "market_ticker": "U",
                     "created_time": "2026-05-04T15:30:00Z"},
                ], "cursor": None}
        reconciler._kalshi_client = _Client()

        result = await reconciler.backfill(
            min_ts=1_777_000_000,
            max_ts=1_778_000_000,
            dry_run=True,
        )
        assert len(result.planned_writes) == 2
        assert result.written_count == 0  # dry-run

    @pytest.mark.asyncio
    async def test_backfill_allowlist_rejects_unexpected(self, tmp_path):
        """If --allowlist is set and the planned write set has extras,
        backfill aborts unless --allow-extras."""
        reconciler = _make_minimal_reconciler(tmp_path)
        class _Client:
            async def get_fills(self, **kwargs):
                return {"fills": [
                    {"order_id": "OID-1", "count_fp": "5",
                     "no_price_dollars": "0.55", "fee_cost": "0.08",
                     "is_taker": True, "side": "no",
                     "ticker": "T", "market_ticker": "T",
                     "created_time": "2026-05-04T15:00:00Z"},
                    {"order_id": "OID-2", "count_fp": "10",
                     "no_price_dollars": "0.60", "fee_cost": "0.15",
                     "is_taker": True, "side": "no",
                     "ticker": "U", "market_ticker": "U",
                     "created_time": "2026-05-04T15:30:00Z"},
                ], "cursor": None}
        reconciler._kalshi_client = _Client()

        result = await reconciler.backfill(
            min_ts=1_777_000_000,
            max_ts=1_778_000_000,
            allowlist={"OID-1"},  # OID-2 is unexpected
            dry_run=True,
        )
        assert result.aborted is True
        assert "OID-2" in result.abort_reason

        # With allow_extras, the abort is overridden
        result_allowed = await reconciler.backfill(
            min_ts=1_777_000_000,
            max_ts=1_778_000_000,
            allowlist={"OID-1"},
            allow_extras=True,
            dry_run=True,
        )
        # Filter applies: only OID-1 is written
        assert result_allowed.aborted is False
        assert len(result_allowed.planned_writes) == 1
        assert result_allowed.planned_writes[0]["order_id"] == "OID-1"

    @pytest.mark.asyncio
    async def test_backfill_expect_n_mismatch_aborts(self, tmp_path):
        """If --expect-n is set and N differs, backfill aborts."""
        reconciler = _make_minimal_reconciler(tmp_path)
        class _Client:
            async def get_fills(self, **kwargs):
                # Returns 2 fills
                return {"fills": [
                    {"order_id": "OID-1", "count_fp": "5",
                     "no_price_dollars": "0.55", "fee_cost": "0.08",
                     "is_taker": True, "side": "no",
                     "ticker": "T", "market_ticker": "T",
                     "created_time": "2026-05-04T15:00:00Z"},
                    {"order_id": "OID-2", "count_fp": "10",
                     "no_price_dollars": "0.60", "fee_cost": "0.15",
                     "is_taker": True, "side": "no",
                     "ticker": "U", "market_ticker": "U",
                     "created_time": "2026-05-04T15:30:00Z"},
                ], "cursor": None}
        reconciler._kalshi_client = _Client()

        result = await reconciler.backfill(
            min_ts=1_777_000_000,
            max_ts=1_778_000_000,
            expect_n=3,  # mock returns 2, mismatch
            dry_run=True,
        )
        assert result.aborted is True
        assert "expected 3" in result.abort_reason
        assert "got 2" in result.abort_reason

        # Matching expect_n succeeds
        result_match = await reconciler.backfill(
            min_ts=1_777_000_000,
            max_ts=1_778_000_000,
            expect_n=2,
            dry_run=True,
        )
        assert result_match.aborted is False
        assert len(result_match.planned_writes) == 2


# ---------------------------------------------------------------------------
# Delta-driven success-path alerts — runbook "Telegram alerts" contract.
# Closes the reconciler-telegram-alert-gap: success path with delta>0 was silent.
# ---------------------------------------------------------------------------


def _make_fill(order_id: str, price: float = 0.55) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "action": "buy",  # default: legacy bot fills are always buys
        "count_fp": "5",
        "no_price_dollars": str(price),
        "yes_price_dollars": str(round(1 - price, 2)),
        "fee_cost": "0.04",
        "is_taker": True,
        "side": "no",
        "ticker": "T",
        "market_ticker": "T",
        "created_time": "2026-05-10T15:00:00Z",
    }


class _ReturnsFills:
    """Stub Kalshi client returning a fixed fills list each call."""

    def __init__(self, fills: list[dict[str, Any]]):
        self._fills = fills

    async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
        return {"fills": self._fills, "cursor": None}


class _StreamsUniqueFill:
    """Stub Kalshi client returning a fresh OID-N fill on each call,
    guaranteeing every tick sees a missing order_id."""

    def __init__(self):
        self._i = 0

    async def get_fills(self, *, min_ts=None, cursor=None, **kwargs):
        f = _make_fill(f"OID-{self._i}")
        self._i += 1
        return {"fills": [f], "cursor": None}


class TestDeltaDrivenAlerts:
    """Success-path alert ladder.

    Bug discovered 2026-05-07: reconciler rescued 4 missed fills in production,
    no Telegram alert fired because the success path computed `status` but
    never invoked the notifier.
    """

    @pytest.mark.asyncio
    async def test_clean_tick_emits_no_alert(self, tmp_path):
        """delta=0, not truncated → no alert."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._kalshi_client = _ReturnsFills([])

        tick = await reconciler.tick()
        assert tick.status == "clean"
        assert notifier.alerts_sent == []

    @pytest.mark.asyncio
    async def test_warning_status_fires_warning_alert(self, tmp_path):
        """delta ∈ [1, alert_threshold_critical_delta], not truncated →
        WARNING alert."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._kalshi_client = _ReturnsFills([_make_fill("OID-1")])

        tick = await reconciler.tick()
        assert tick.status == "warning"
        assert tick.delta_count == 1
        warnings = [a for a in notifier.alerts_sent if a.level.name == "WARNING"]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_warning_fires_every_tick_without_cooldown(self, tmp_path):
        """WARNING is not cooldown-suppressed. Three warning ticks back-to-back
        → three WARNING alerts."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._kalshi_client = _StreamsUniqueFill()

        for _ in range(3):
            await reconciler.tick()
            clock_value[0] += 10  # well within any cooldown

        warnings = [a for a in notifier.alerts_sent if a.level.name == "WARNING"]
        assert len(warnings) == 3, "WARNING has no cooldown"

    @pytest.mark.asyncio
    async def test_critical_on_truncation_fires_critical(self, tmp_path):
        """Pagination truncated → CRITICAL."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]

        tick = await reconciler.tick()
        assert tick.status == "critical"
        assert tick.truncated_at_max_pages is True
        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 1
        assert "truncat" in criticals[0].message.lower()

    @pytest.mark.asyncio
    async def test_critical_on_delta_over_threshold_fires_critical(self, tmp_path):
        """delta > alert_threshold_critical_delta → CRITICAL."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        # default alert_threshold_critical_delta=3 → 4 fills triggers critical
        fills = [_make_fill(f"OID-{i}") for i in range(4)]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._kalshi_client = _ReturnsFills(fills)

        tick = await reconciler.tick()
        assert tick.status == "critical"
        assert tick.delta_count == 4
        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 1

    @pytest.mark.asyncio
    async def test_critical_within_cooldown_downgrades_to_warning(self, tmp_path):
        """Within cooldown a CRITICAL-eligible event downgrades to WARNING
        (not suppressed entirely)."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._critical_alert_cooldown_seconds = 3600

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]

        # First truncation tick → CRITICAL
        await reconciler.tick()
        clock_value[0] += 10
        # Second truncation tick within cooldown → downgraded to WARNING
        await reconciler.tick()

        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        warnings = [a for a in notifier.alerts_sent if a.level.name == "WARNING"]
        assert len(criticals) == 1
        assert len(warnings) == 1, (
            "second truncation within cooldown should downgrade to WARNING"
        )

    @pytest.mark.asyncio
    async def test_critical_after_cooldown_fires_again(self, tmp_path):
        """Past cooldown a fresh CRITICAL is allowed."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._critical_alert_cooldown_seconds = 3600

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]

        await reconciler.tick()
        clock_value[0] += 3601  # past cooldown
        await reconciler.tick()

        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 2

    @pytest.mark.asyncio
    async def test_clean_tick_clears_all_critical_cooldowns(self, tmp_path):
        """A clean tick clears ALL three per-mode cooldowns so the next
        critical-class event after recovery fires CRITICAL again. This path
        exercises the truncation sub-mode specifically; the clean reset must
        cover it (not just the delta sub-mode)."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._critical_alert_cooldown_seconds = 3600

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        # Tick 1: truncation → CRITICAL
        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        await reconciler.tick()
        clock_value[0] += 10

        # Tick 2: clean (no fills, not truncated) — restore default fetch path
        del reconciler._fetch_fills_paginated  # remove the monkeypatch
        reconciler._kalshi_client = _ReturnsFills([])
        await reconciler.tick()
        clock_value[0] += 10

        # Tick 3: truncation again, well within original cooldown — should
        # fire a fresh CRITICAL because the clean tick cleared the cooldown.
        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        await reconciler.tick()

        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 2, (
            "clean tick should reset the truncation-path cooldown"
        )

    @pytest.mark.asyncio
    async def test_truncation_critical_does_not_silence_cycle_critical(self, tmp_path):
        """Regression for the shared-cooldown bug: a truncation CRITICAL must NOT
        downgrade a cursor-cycle CRITICAL that fires within the cooldown window.
        The two are structurally different failure modes (load/backlog vs Kalshi
        API contract break) and keep independent per-mode cooldown timestamps."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._critical_alert_cooldown_seconds = 3600

        mode = {"cycle": False}

        async def _fetch(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": not mode["cycle"],
                "cycle_aborted": mode["cycle"],
                "fills_count": 0,
            }

        reconciler._fetch_fills_paginated = _fetch  # type: ignore[method-assign]

        # Tick 1: truncation → CRITICAL (sets the truncation cooldown only).
        await reconciler.tick()
        clock_value[0] += 60  # well within the 3600s cooldown

        # Tick 2: cursor cycle → must ALSO be CRITICAL; its own cooldown is fresh.
        mode["cycle"] = True
        await reconciler.tick()

        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 2, (
            "cycle CRITICAL must not be silenced by a recent truncation CRITICAL"
        )
        assert "cursor cycle" in criticals[1].message.lower()

    @pytest.mark.asyncio
    async def test_failure_cooldown_does_not_block_delta_critical(self, tmp_path):
        """Failure-path CRITICAL recently fired → delta-path CRITICAL must
        still fire (separate cooldown timestamps)."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        notifier = _MockNotifier()
        clock_value = [1_700_000_000.0]
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = notifier
        reconciler._clock = lambda: clock_value[0]
        reconciler._critical_alert_cooldown_seconds = 3600
        reconciler._alert_threshold_consecutive = 1  # any failure → CRITICAL fast

        # Failure tick → fires failure-path CRITICAL
        class _Boom:
            async def get_fills(self, **kwargs):
                raise RuntimeError("boom")

        reconciler._kalshi_client = _Boom()
        await reconciler.tick()
        clock_value[0] += 1

        # Switch to a delta-path-CRITICAL situation (truncation)
        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": reconciler._max_pages_per_tick,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        reconciler._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        await reconciler.tick()

        criticals = [a for a in notifier.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals) == 2, (
            "delta-path CRITICAL has its own cooldown timestamp; failure-path "
            "cooldown must not block it"
        )

    @pytest.mark.asyncio
    async def test_no_notifier_does_not_crash(self, tmp_path):
        """Defensive: notifier=None must be a no-op for delta alerts."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = None
        reconciler._kalshi_client = _ReturnsFills([_make_fill("OID-1")])

        tick = await reconciler.tick()
        assert tick.status == "warning"
        assert tick.delta_count == 1

    @pytest.mark.asyncio
    async def test_notifier_exception_does_not_break_tick(self, tmp_path):
        """tick() never-raises contract: notifier.send_alert raising in the
        delta-alert path must not break the tick (synthetic row already on disk)."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()

        class _BoomNotifier:
            async def send_alert(self, **kwargs):
                raise RuntimeError("notifier down")

        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._trade_log_path = log_path
        reconciler._notifier = _BoomNotifier()
        reconciler._kalshi_client = _ReturnsFills([_make_fill("OID-1")])

        tick = await reconciler.tick()
        assert tick.status == "warning"
        assert tick.delta_count == 1

    @pytest.mark.asyncio
    async def test_delta_critical_timestamp_persists_across_restart(self, tmp_path):
        """Cooldown survives process restart via fills_reconciler.json."""
        log_path = tmp_path / "trades.jsonl"
        log_path.touch()
        state_path = tmp_path / "fills_reconciler.json"

        async def _fetch_truncated(*, min_ts: int):
            return {
                "fills": [],
                "pages_consumed": 20,
                "truncated": True,
                "cycle_aborted": False,
                "fills_count": 0,
            }

        # First reconciler — truncation fires CRITICAL, timestamp persisted
        notifier1 = _MockNotifier()
        reconciler1 = FillsReconciler(
            state_path=state_path,
            trade_log_path=log_path,
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=notifier1,
            clock=lambda: 1_700_000_000.0,
        )
        reconciler1._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        await reconciler1.tick()
        criticals1 = [a for a in notifier1.alerts_sent if a.level.name == "CRITICAL"]
        assert len(criticals1) == 1

        # State file should now carry the timestamp on the TRUNCATION key (this
        # test fires the truncation sub-mode). Per-mode split: the delta key stays
        # 0 because the delta sub-mode never fired.
        loaded = json.loads(state_path.read_text())
        assert loaded["last_truncation_critical_alert_ts_unix"] == 1_700_000_000
        assert loaded["last_delta_critical_alert_ts_unix"] == 0

        # Second reconciler — restart, immediately tick within cooldown
        notifier2 = _MockNotifier()
        reconciler2 = FillsReconciler(
            state_path=state_path,
            trade_log_path=log_path,
            trade_log_lock=asyncio.Lock(),
            kalshi_client=_NoopClient(),
            engine=None,
            notifier=notifier2,
            clock=lambda: 1_700_000_000.0 + 100,
        )
        reconciler2._fetch_fills_paginated = _fetch_truncated  # type: ignore[method-assign]
        await reconciler2.tick()

        criticals2 = [a for a in notifier2.alerts_sent if a.level.name == "CRITICAL"]
        warnings2 = [a for a in notifier2.alerts_sent if a.level.name == "WARNING"]
        assert len(criticals2) == 0, "cooldown should survive restart"
        assert len(warnings2) == 1, "downgraded to WARNING during cooldown"


class TestPlacementIndexShape:
    def test_placement_index_records_action_alongside_strategy(self, tmp_path) -> None:
        """_build_placement_index must produce {strategy, action} dicts so classify_fill
        can distinguish bot-placed sells (future) from manual sells (today).
        """
        from python.execution.fills_reconciler import FillsReconciler
        log = tmp_path / "trades.jsonl"
        rows = [
            {
                "event": "placement",
                "order_id": "p1",
                "mode": "live",
                "action": "buy",
                "metadata": {"strategy": "weather_edge"},
            },
            {
                "event": "placement",
                "order_id": "p2",
                "mode": "live",
                "metadata": {"strategy": "legacy_no_action"},
            },
        ]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        rec = FillsReconciler.__new__(FillsReconciler)
        rec._trade_log_path = log
        rec._placement_row_lookback = 1000
        rec._placement_index = {}
        rec._build_placement_index()
        assert rec._placement_index["p1"] == {"strategy": "weather_edge", "action": "buy"}
        # Legacy rows missing 'action' default to "buy" (bot has only ever bought)
        assert rec._placement_index["p2"] == {"strategy": "legacy_no_action", "action": "buy"}


def test_count_fp_is_plain_integer_count() -> None:
    """Regression guard: Kalshi count_fp is a plain integer contract count,
    NOT a fixed-point scaled value. Mishandling this corrupts every P&L
    calculation (see plan-review BLOCKER #1 / #5)."""
    from python.execution.fills_reconciler import _contracts_from_count_fp
    assert _contracts_from_count_fp(10) == 10
    assert _contracts_from_count_fp("5") == 5
    assert _contracts_from_count_fp(0) == 0
    assert _contracts_from_count_fp(None) == 0


class TestFillKindClassifier:
    def test_classifier_buy_returns_buy_entry(self) -> None:
        fill = {"order_id": "abc", "action": "buy"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.BUY_ENTRY

    def test_classifier_buy_with_known_oid_still_buy_entry(self) -> None:
        # Dedup is the caller's job, not the classifier's
        fill = {"order_id": "abc", "action": "buy"}
        ctx = ClassifierContext(
            tracked_buy_oids={"abc"},
            placement_index={},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.BUY_ENTRY

    def test_classifier_missing_action_returns_unknown(self) -> None:
        fill = {"order_id": "abc"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={},
            open_positions_by_ticker_side={},
        )
        assert classify_fill(fill, ctx) == FillKind.UNKNOWN

    def test_classifier_invalid_action_returns_unknown(self) -> None:
        fill = {"order_id": "abc", "action": "transfer"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={},
            open_positions_by_ticker_side={},
        )
        assert classify_fill(fill, ctx) == FillKind.UNKNOWN

    def test_classifier_sell_no_match_returns_sell_manual_candidate(self) -> None:
        fill = {"order_id": "newoid", "action": "sell"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.SELL_MANUAL_CANDIDATE

    def test_classifier_sell_matching_tracked_buy_returns_anomaly(self) -> None:
        # SELL_BOT_ANOMALY fires when a sell hits an oid known as a buy
        fill = {"order_id": "buy_oid", "action": "sell"}
        ctx = ClassifierContext(
            tracked_buy_oids={"buy_oid"},
            placement_index={},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.SELL_BOT_ANOMALY

    def test_classifier_sell_with_placement_action_sell_returns_sell_bot(self) -> None:
        # Cold-restart bot-sell recovery path (future feature)
        fill = {"order_id": "X", "action": "sell"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={"X": {"action": "sell", "strategy": "future_strategy"}},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.SELL_BOT

    def test_classifier_sell_with_placement_action_buy_returns_sell_manual_candidate(self) -> None:
        # CRITICAL — placement-as-buy is NOT proof of bot-owned sell.
        # Getting this backwards routes manual sells into the SELL_BOT path.
        fill = {"order_id": "X", "action": "sell"}
        ctx = ClassifierContext(
            tracked_buy_oids=set(),
            placement_index={"X": {"action": "buy", "strategy": "weather_edge"}},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.SELL_MANUAL_CANDIDATE

    def test_classifier_dispatch_order_anomaly_beats_placement(self) -> None:
        # Pin dispatch order: anomaly check runs BEFORE placement-action check.
        # If a sell's oid is in BOTH tracked_buy_oids AND a sell-placement
        # (impossible today, but pin the invariant), anomaly wins (fail-safe).
        fill = {"order_id": "X", "action": "sell"}
        ctx = ClassifierContext(
            tracked_buy_oids={"X"},
            placement_index={"X": {"action": "sell", "strategy": "future"}},
            open_positions_by_ticker_side={},
        )
        result = classify_fill(fill, ctx)
        assert result == FillKind.SELL_BOT_ANOMALY


class TestLoadTrackedOids:
    def test_single_pass_returns_buy_and_manual_close_sets(self, tmp_path) -> None:
        # Verify the combined scan replaces the previous two-scan pattern
        log = tmp_path / "trades.jsonl"
        log.write_text(
            '{"event": "fill", "order_id": "buy1", "mode": "live"}\n'
            '{"event": "placement", "order_id": "p1", "mode": "live"}\n'
            '{"event": "manual_close", "order_id": "mc1", "mode": "live"}\n'
            '{"event": "fill", "order_id": "buy2", "mode": "shadow"}\n'  # shadow excluded
            '{"event": "settlement", "order_id": "s1", "mode": "live"}\n'
        )
        rec = FillsReconciler.__new__(FillsReconciler)
        rec._trade_log_path = log
        tracked = rec._load_tracked_oids()
        assert tracked.buys == {"buy1"}
        assert tracked.manual_closes == {"mc1"}


def test_load_tracked_oids_returns_dataclass_with_three_sets(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"event": "fill", "order_id": "FILL-1", "mode": "live"}),
                json.dumps({"event": "manual_close", "order_id": "MC-1", "mode": "live"}),
                json.dumps({"event": "cash_out", "order_id": "CO-1", "mode": "live"}),
                json.dumps({"event": "fill", "order_id": "FILL-2", "mode": "shadow"}),  # filtered
            ]
        )
    )
    rec = FillsReconciler.__new__(FillsReconciler)
    rec._trade_log_path = log
    tracked = rec._load_tracked_oids()
    assert isinstance(tracked, TrackedOids)
    assert tracked.buys == {"FILL-1"}
    assert tracked.manual_closes == {"MC-1"}
    assert tracked.cash_outs == {"CO-1"}


def test_load_tracked_oids_empty_log() -> None:
    rec = FillsReconciler.__new__(FillsReconciler)
    rec._trade_log_path = Path("/nonexistent")
    tracked = rec._load_tracked_oids()
    assert tracked.buys == set()
    assert tracked.manual_closes == set()
    assert tracked.cash_outs == set()


@pytest.mark.asyncio
async def test_sell_manual_full_path_writes_row_closes_position_stamps_record(tmp_path) -> None:
    """End-to-end happy path for _apply_manual_close."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0  # unlimited for this test

    # Seed cost basis via one entry FILL row
    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill",
            "order_id": "buy_oid",
            "position_key": "K-YES",
            "market_ticker": "K",
            "side": "YES",
            "fill_price": 0.40,
            "contracts": 10,
            "fees_from_api": 0.05,
            "mode": "live",
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )

    # OrderRecord stub for stamping
    entry_record = MagicMock()
    entry_record.signal.position_key = "K-YES"
    entry_record.status = OrderStatus.FILLED
    engine._orders["buy_oid"] = entry_record

    # RiskManager mock — expose `positions` (the property the apply path uses)
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    risk.close_position = MagicMock()
    engine._risk = risk

    # Simulate one sell fill from Kalshi. count_fp is plain count (NOT fp).
    sell_fills = [
        {
            "order_id": "sell_oid",
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
    ]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=sell_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "APPLIED"

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2
    mc = rows[1]
    assert mc["event"] == "manual_close"
    assert mc["order_id"] == "sell_oid"
    assert mc["position_key"] == "K-YES"
    assert mc["close_price"] == 0.65
    assert mc["contracts_closed"] == 10
    assert mc["close_proceeds"] == pytest.approx(6.50)
    assert mc["cost_basis_dollars"] == pytest.approx(4.00)
    assert mc["entry_fees"] == pytest.approx(0.05)
    assert mc["exit_fees"] == pytest.approx(0.04)
    # realized_pnl = 6.50 - 4.00 - 0.05 - 0.04 = 2.41
    assert mc["realized_pnl"] == pytest.approx(2.41)
    assert mc["metadata"]["classification"] == "SELL_MANUAL"
    assert mc["metadata"]["close_type"] == "full"
    assert "applied_at" in mc
    assert mc["metadata"]["strategy"] == "weather_edge"

    risk.close_position.assert_called_once_with("K-YES", pnl=pytest.approx(2.41))
    assert entry_record.status == OrderStatus.EXTERNALLY_CLOSED

    # Verify WARNING-level alert was queued (level is keyword-only)
    rec._notifier.send_alert.assert_called()
    last = rec._notifier.send_alert.call_args_list[-1]
    assert last.kwargs.get("level") == AlertLevel.WARNING
    msg = last.args[0] if last.args else last.kwargs["message"]
    assert "Manual close reconciled" in msg


@pytest.mark.asyncio
async def test_orphan_close_oids_persisted_across_restart(tmp_path) -> None:
    """spec §8.3 — orphan close oids persist via state/fills_reconciler.json so
    subsequent ticks (and restarts) don't re-CRITICAL-alert."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0
    rec._trade_log_path.write_text("")  # empty log → cost basis returns 0 contracts

    # Risk has a phantom position (no entry FILL row); triggers ZERO_COST_BASIS
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    sell_fills = [{
        "order_id": "orphan_sell", "action": "sell", "side": "yes",
        "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
        "count_fp": 10, "fee_cost": 0.04,
        "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
        "market_ticker": "K",
    }]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="orphan_sell",
            kalshi_fills=sell_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "ZERO_COST_BASIS"
    assert "orphan_sell" in rec._orphan_close_oids

    # Persist + reload simulating restart
    await rec._persist_state()
    rec2, _ = make_reconciler_with_engine(tmp_path)
    assert "orphan_sell" in rec2._orphan_close_oids


@pytest.mark.asyncio
async def test_dispatcher_sell_bot_anomaly_alerts_and_skips_mutation(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk
    async with rec._trade_log_lock:
        outcome = await rec._apply_fill_kind(
            kind=FillKind.SELL_BOT_ANOMALY,
            order_id="X",
            kalshi_fills=[{"action": "sell", "market_ticker": "K"}],
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
            tracked_cash_out_oids=set(),
        )
    assert outcome == "SELL_BOT_ANOMALY"
    risk.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_unknown_alerts_and_skips_mutation(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    async with rec._trade_log_lock:
        outcome = await rec._apply_fill_kind(
            kind=FillKind.UNKNOWN,
            order_id="X",
            kalshi_fills=[{"market_ticker": "K"}],
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
            tracked_cash_out_oids=set(),
        )
    assert outcome == "UNKNOWN"


@pytest.mark.asyncio
async def test_dispatcher_sell_bot_raises_today(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, _ = make_reconciler_with_engine(tmp_path)
    async with rec._trade_log_lock:
        with pytest.raises(NotImplementedError, match="SELL_BOT"):
            await rec._apply_fill_kind(
                kind=FillKind.SELL_BOT,
                order_id="X",
                kalshi_fills=[{"action": "sell"}],
                position_key="K-YES",
                market_ticker="K",
                tracked_manual_close_oids=set(),
                tracked_cash_out_oids=set(),
            )


@pytest.mark.asyncio
async def test_partial_sell_held_until_complete(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    # Entry of 10 contracts
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    # Sell only 6 of 10 contracts
    partial_fills = [
        {"order_id": "sell_oid", "action": "sell", "side": "yes",
         "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
         "count_fp": 6, "fee_cost": 0.03,
         "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
         "market_ticker": "K"}
    ]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=partial_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "PARTIAL_HELD"
    assert "sell_oid" in rec._pending_partial_sells
    # No row should be written
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1  # original entry only
    risk.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_flag_off_shakedown_alerts_without_mutating(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = False  # shakedown
    rec._shakedown_seen_oids_path = tmp_path / "shakedown_seen.json"

    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        outcome = await rec._handle_shakedown(
            order_id="sell_oid",
            kalshi_fills=sell_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "SHAKEDOWN_ALERTED"
    risk.close_position.assert_not_called()
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1  # no new row
    assert "sell_oid" in rec._shakedown_seen_oids


@pytest.mark.asyncio
async def test_shakedown_emits_info_log_before_alert(tmp_path, caplog) -> None:
    """Observability guard: INFO log mirrors the shadow alert so a silent
    Telegram drop (rate-limit, notifier unavailable) is still detectable in
    journald. Regression coverage for the 2026-05-18 incident where the
    [SHADOW] alert never reached Telegram and the only evidence was an oid
    in shakedown_seen_oids.json.
    """
    import logging

    from tests.conftest import make_reconciler_with_engine

    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = False
    rec._shakedown_seen_oids_path = tmp_path / "shakedown_seen.json"
    rec._notifier = None  # simulate the silent-drop case explicitly
    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    with caplog.at_level(logging.INFO, logger="python.execution.fills_reconciler"):
        async with rec._trade_log_lock:
            outcome = await rec._handle_shakedown(
                order_id="sell_oid",
                kalshi_fills=sell_fills,
                position_key="K-YES",
                market_ticker="K",
                tracked_manual_close_oids=set(),
            )
    assert outcome == "SHAKEDOWN_ALERTED"
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("shadow_manual_close_detected" in m for m in info_messages), (
        f"INFO log not emitted; got: {info_messages}"
    )
    assert any("oid=sell_oid" in m for m in info_messages)
    assert any("ticker=K" in m for m in info_messages)


@pytest.mark.asyncio
async def test_shakedown_dedup_persists_across_restart(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = False
    rec._shakedown_seen_oids_path = tmp_path / "shakedown_seen.json"
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        await rec._handle_shakedown(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    rec._notifier.send_alert.assert_called_once()

    # Simulate restart by constructing a fresh reconciler with same shakedown path
    rec2, engine2 = make_reconciler_with_engine(tmp_path)
    rec2._manual_close_enabled = False
    rec2._shakedown_seen_oids_path = tmp_path / "shakedown_seen.json"
    rec2._load_shakedown_seen_oids()
    assert "sell_oid" in rec2._shakedown_seen_oids
    # Re-firing same oid → suppress
    risk2 = MagicMock()
    risk2.positions = {"K-YES": 4.0}
    engine2._risk = risk2
    async with rec2._trade_log_lock:
        outcome = await rec2._handle_shakedown(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "SHAKEDOWN_SUPPRESSED"
    rec2._notifier.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_canary_apply_limit_first_succeeds_second_held(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 1  # canary

    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy1",
            "position_key": "K1-YES", "market_ticker": "K1",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n" +
        json.dumps({
            "event": "fill", "order_id": "buy2",
            "position_key": "K2-YES", "market_ticker": "K2",
            "fill_price": 0.30, "contracts": 5, "fees_from_api": 0.02,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K1-YES": 4.0, "K2-YES": 1.5}
    engine._risk = risk
    engine._orders = {}

    sell1 = [{"order_id": "s1", "action": "sell", "side": "yes",
              "yes_price_dollars": 0.5, "no_price_dollars": 0.5,
              "count_fp": 10, "fee_cost": 0.03,
              "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
              "market_ticker": "K1"}]
    sell2 = [{"order_id": "s2", "action": "sell", "side": "yes",
              "yes_price_dollars": 0.5, "no_price_dollars": 0.5,
              "count_fp": 5, "fee_cost": 0.02,
              "is_taker": True, "created_time": "2026-05-17T12:01:00Z",
              "market_ticker": "K2"}]

    async with rec._trade_log_lock:
        out1 = await rec._apply_fill_kind(
            kind=FillKind.SELL_MANUAL_CANDIDATE,
            order_id="s1", kalshi_fills=sell1,
            position_key="K1-YES", market_ticker="K1",
            tracked_manual_close_oids=set(),
            tracked_cash_out_oids=set(),
        )
        out2 = await rec._apply_fill_kind(
            kind=FillKind.SELL_MANUAL_CANDIDATE,
            order_id="s2", kalshi_fills=sell2,
            position_key="K2-YES", market_ticker="K2",
            tracked_manual_close_oids=set(),
            tracked_cash_out_oids=set(),
        )
    assert out1 == "APPLIED"
    assert out2 == "CANARY_LIMIT_REACHED"


@pytest.mark.asyncio
async def test_multi_leg_position_stamps_all_records(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy1",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 5, "fees_from_api": 0.025,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n" +
        json.dumps({
            "event": "fill", "order_id": "buy2",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.42, "contracts": 5, "fees_from_api": 0.026,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    rec1 = MagicMock(); rec1.signal.position_key = "K-YES"; rec1.status = OrderStatus.FILLED
    rec2 = MagicMock(); rec2.signal.position_key = "K-YES"; rec2.status = OrderStatus.FILLED
    engine._orders = {"buy1": rec1, "buy2": rec2}
    risk = MagicMock(); risk.positions = {"K-YES": 4.1}
    engine._risk = risk

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.55, "no_price_dollars": 0.45,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        await rec._apply_manual_close(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert rec1.status == OrderStatus.EXTERNALLY_CLOSED
    assert rec2.status == OrderStatus.EXTERNALLY_CLOSED


@pytest.mark.asyncio
async def test_strategy_fallback_to_cost_basis_last_strategy(tmp_path) -> None:
    """When the in-memory oid lookup AND placement scan both miss,
    fall back to cost_basis.last_strategy (4th step in resolver chain)."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    # Entry FILL written BEFORE placement-row feature shipped — no placement row.
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "legacy_buy",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    # Bot has no in-memory record for the legacy buy after restart
    engine._orders = {}
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        await rec._apply_manual_close(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    mc = rows[-1]
    assert mc["metadata"]["strategy"] == "weather_edge"
    assert mc["metadata"]["strategy_resolution"] == "cost_basis_fallback"


@pytest.mark.asyncio
async def test_manual_close_idempotent_on_replay(tmp_path) -> None:
    """[CRITICAL] — replaying the same sell fill must not double-apply."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n" +
        # Pre-existing manual_close row simulating prior tick
        json.dumps({
            "event": "manual_close", "order_id": "sell_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "mode": "live", "realized_pnl": 2.41,
        }) + "\n"
    )
    risk = MagicMock(); risk.positions = {}  # already closed
    engine._risk = risk
    engine._orders = {}

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids={"sell_oid"},
        )
    assert outcome == "ALREADY_CLOSED_DEDUP"
    risk.close_position.assert_not_called()
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 2  # original entry + pre-existing manual_close only


@pytest.mark.asyncio
async def test_zero_cost_basis_orphan_alerts_critical_no_mutation(tmp_path) -> None:
    """Sell exists but no entry FILL row — wrong P&L is worse than missing P&L."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    rec._trade_log_path.write_text("")  # empty log → no cost basis
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}  # risk thinks position exists
    engine._risk = risk

    sell_fills = [{"order_id": "sell_oid", "action": "sell", "side": "yes",
                   "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                   "count_fp": 10, "fee_cost": 0.04,
                   "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                   "market_ticker": "K"}]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid", kalshi_fills=sell_fills,
            position_key="K-YES", market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "ZERO_COST_BASIS"
    risk.close_position.assert_not_called()
    # CRITICAL alert emitted (level is keyword-only)
    from python.notifications.base import AlertLevel
    call_args = rec._notifier.send_alert.call_args_list
    assert any(c.kwargs.get("level") == AlertLevel.CRITICAL for c in call_args)


@pytest.mark.asyncio
async def test_mixed_buy_entry_and_sell_manual_in_one_tick(tmp_path) -> None:
    """Tick that contains both a missed-bot-entry fill AND a manual close."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    # Seed: K-YES already has entry FILL (cost basis); new buy on K2 hasn't been
    # logged yet (missed bot fill).
    rec._trade_log_path.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_K",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )

    # The "K-YES" position must be resolvable from market_ticker "K".
    # Seed an OrderRecord whose signal.market_ticker="K" and signal.position_key="K-YES"
    # so _derive_position_key_for_ticker can find it.
    entry_rec = MagicMock()
    entry_rec.signal.market_ticker = "K"
    entry_rec.signal.position_key = "K-YES"
    entry_rec.status = OrderStatus.FILLED
    engine._orders["buy_K"] = entry_rec

    # Mock Kalshi /portfolio/fills returning one buy (K2) + one sell (K)
    rec._kalshi_client.get_fills = AsyncMock(return_value={
        "fills": [
            {"order_id": "new_buy_K2", "action": "buy", "side": "no",
             "no_price_dollars": 0.30, "yes_price_dollars": 0.70,
             "count_fp": 5, "fee_cost": 0.02,
             "is_taker": False, "created_time": "2026-05-17T11:00:00Z",
             "market_ticker": "K2"},
            {"order_id": "manual_sell", "action": "sell", "side": "yes",
             "yes_price_dollars": 0.55, "no_price_dollars": 0.45,
             "count_fp": 10, "fee_cost": 0.04,
             "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
             "market_ticker": "K"},
        ], "cursor": None,
    })

    await rec.tick()
    rows = [json.loads(line) for line in rec._trade_log_path.read_text().splitlines()]
    events = [r["event"] for r in rows]
    # Original entry + new buy_fill row (from BUY_ENTRY path) + manual_close row
    assert "fill" in events
    assert "manual_close" in events
    # 3 rows total: original entry FILL, new BUY_ENTRY synthetic FILL, new manual_close
    assert len(rows) == 3


def test_build_tick_log_extra_includes_manual_close_fields(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine
    rec, _ = make_reconciler_with_engine(tmp_path)
    rec._manual_close_realized_pnl_cents_this_tick = 241
    rec._manual_close_apply_latency_max_this_tick = 0.5
    rec._shakedown_seen_oids = {"oid1"}
    extra = rec._build_tick_log_extra(
        started_at_ts=0.0,
        ended_at_ts=1.0,
        duration_ms=1000,
        pages_consumed=1,
        truncated=False,
        cycle_aborted=False,
        fills_fetched=2,
        delta_count=1,
        order_ids_synthesised=["x"],
        resolution_breakdown={"in_memory": 1},
        alert_state="clean",
        cursor_before=0,
        cursor_after=1,
        classifier_decision_counts={"buy_entry": 1, "sell_manual_candidate": 1},
        apply_path_outcome_counts={"APPLIED": 1},
        pending_partial_sells_count=0,
    )
    assert extra["classifier_decision_counts"]["buy_entry"] == 1
    assert extra["classifier_decision_counts"]["sell_manual_candidate"] == 1
    assert extra["apply_path_outcome_counts"]["APPLIED"] == 1
    assert extra["manual_close_realized_pnl_cents_total"] == 241
    assert extra["manual_close_apply_latency_seconds"] == 0.5
    assert extra["pending_partial_sells_count"] == 0
    assert extra["shakedown_suppressed_count"] == 1
    assert extra["manual_close_applied"] == rec._manual_close_enabled


@pytest.mark.asyncio
async def test_shared_lock_serializes_engine_append_and_manual_close_apply(tmp_path) -> None:
    """[BLOCKER] (spec §9.4.6) — the engine cannot append a fill row between
    the reconciler's tracked-oid snapshot and its synthetic-row write. Pinning
    this invariant guards the apply path against future refactors that
    accidentally release the lock mid-mutation.
    """
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    # Seed entry FILL so cost basis exists
    log = rec._trade_log_path
    log.write_text(json.dumps({
        "event": "fill", "order_id": "buy", "position_key": "K-YES",
        "market_ticker": "K", "fill_price": 0.40, "contracts": 10,
        "fees_from_api": 0.05, "mode": "live",
        "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    real_lock = rec._trade_log_lock
    apply_done = asyncio.Event()
    engine_grabbed_after_apply = asyncio.Event()

    async def reconciler_apply():
        async with real_lock:
            await rec._apply_manual_close(
                order_id="sell", kalshi_fills=[{
                    "order_id": "sell", "action": "sell", "side": "yes",
                    "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
                    "count_fp": 10, "fee_cost": 0.04,
                    "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
                    "market_ticker": "K",
                }],
                position_key="K-YES", market_ticker="K",
                tracked_manual_close_oids=set(),
            )
            apply_done.set()
            # Hold the lock briefly so the competing writer must wait
            await asyncio.sleep(0.02)

    async def competing_engine_writer():
        # Try to grab the lock — should block until reconciler_apply releases.
        await real_lock.acquire()
        try:
            # Apply MUST have finished before we got the lock
            assert apply_done.is_set()
            engine_grabbed_after_apply.set()
        finally:
            real_lock.release()

    await asyncio.gather(reconciler_apply(), competing_engine_writer())
    assert apply_done.is_set()
    assert engine_grabbed_after_apply.is_set()


def test_buy_entry_synthetic_row_schema_stable() -> None:
    """[BLOCKER] (spec §9.5) — synthetic BUY_ENTRY row schema is the
    backward-compat contract with downstream trade-log consumers. Any
    change to _build_synthetic_row that drifts from this expected shape
    is a schema-breaking change.

    Lighter-weight than a byte-for-byte golden fixture: pins the field
    names and types directly. If the implementation drifts, this fails
    with a clear assertion.
    """
    from python.execution.fills_reconciler import FillsReconciler

    rec = FillsReconciler.__new__(FillsReconciler)
    row = rec._build_synthetic_row(
        order_id="bot_oid_42",
        kalshi_fills=[{
            "order_id": "bot_oid_42",
            "action": "buy",
            "side": "yes",
            "yes_price_dollars": 0.45,
            "no_price_dollars": 0.55,
            "count_fp": 7,
            "fee_cost": 0.03,
            "is_taker": True,
            "created_time": "2026-05-17T10:00:00Z",
            "market_ticker": "KXHIGH-NYC-T80",
        }],
        strategy="weather_edge",
        resolution_method="in_memory",
    )

    # Required field set — this is the schema contract
    required_keys = {
        "event", "order_id", "market_ticker", "side", "mode", "timestamp",
        "fill_price", "contracts", "dollars",
        "is_taker_fills", "is_maker_fills", "fees_from_api",
        "metadata",
    }
    assert required_keys.issubset(row.keys()), (
        f"BUY_ENTRY row schema drift; missing: {required_keys - row.keys()}"
    )
    assert row["event"] == "fill"
    assert row["order_id"] == "bot_oid_42"
    assert row["side"] == "YES"  # uppercase per existing convention
    assert row["mode"] == "live"
    assert row["contracts"] == 7
    assert row["metadata"]["strategy"] == "weather_edge"
    assert row["metadata"]["reconciled"] is True
    assert row["metadata"]["reconcile_source"] == "fills_reconciler"
    assert row["metadata"]["strategy_resolution"] == "in_memory"


@pytest.mark.asyncio
async def test_apply_manual_close_calls_record_strategy_trade(tmp_path) -> None:
    """Finding #2 — circuit-breaker accounting must include manual-close P&L."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    log.write_text(json.dumps({
        "event": "fill", "order_id": "buy",
        "position_key": "K-YES", "market_ticker": "K",
        "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
        "mode": "live", "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    sell_fills = [{
        "order_id": "sell_oid", "action": "sell", "side": "yes",
        "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
        "count_fp": 10, "fee_cost": 0.04,
        "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
        "market_ticker": "K",
    }]
    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=sell_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome == "APPLIED"
    # Circuit-breaker accounting: record_strategy_trade called with positive P&L
    risk.record_strategy_trade.assert_called_once()
    call = risk.record_strategy_trade.call_args
    # Accept either positional or keyword args
    args = call.args
    kwargs = call.kwargs
    # Strategy resolved from cost_basis.last_strategy = "weather_edge"
    assert "weather_edge" in (args + tuple(kwargs.values()))
    # won=True because realized_pnl=2.41 > 0
    won_kw = kwargs.get("won")
    if won_kw is None and len(args) >= 3:
        won_kw = args[2]
    assert won_kw is True


@pytest.mark.asyncio
async def test_pending_partial_sells_cleared_when_partial_completes(tmp_path) -> None:
    """Finding #3 — _pending_partial_sells must not leak across ticks."""
    from tests.conftest import make_reconciler_with_engine
    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    log = rec._trade_log_path
    log.write_text(json.dumps({
        "event": "fill", "order_id": "buy",
        "position_key": "K-YES", "market_ticker": "K",
        "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
        "mode": "live", "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    risk = MagicMock(); risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    # First tick: only 6 of 10 sold → PARTIAL_HELD
    partial_fills = [{
        "order_id": "sell_oid", "action": "sell", "side": "yes",
        "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
        "count_fp": 6, "fee_cost": 0.03,
        "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
        "market_ticker": "K",
    }]
    async with rec._trade_log_lock:
        outcome1 = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=partial_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome1 == "PARTIAL_HELD"
    assert "sell_oid" in rec._pending_partial_sells

    # Second tick: full 10/10 arrives → APPLIED → pending entry cleared
    complete_fills = [{
        "order_id": "sell_oid", "action": "sell", "side": "yes",
        "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
        "count_fp": 10, "fee_cost": 0.04,
        "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
        "market_ticker": "K",
    }]
    async with rec._trade_log_lock:
        outcome2 = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=complete_fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )
    assert outcome2 == "APPLIED"
    # Key MUST be removed from the pending set
    assert "sell_oid" not in rec._pending_partial_sells


def test_fill_kind_buy_cash_out_exists() -> None:
    assert FillKind.BUY_CASH_OUT.value == "buy_cash_out"


def test_classifier_context_frozen() -> None:
    ctx = ClassifierContext(
        tracked_buy_oids=set(),
        placement_index={},
        open_positions_by_ticker_side={},
    )
    with pytest.raises(Exception):
        ctx.tracked_buy_oids = {"x"}  # type: ignore[misc]


def test_tracked_oids_three_fields() -> None:
    tr = TrackedOids(buys={"a"}, manual_closes={"b"}, cash_outs={"c"})
    assert tr.buys == {"a"}
    assert tr.manual_closes == {"b"}
    assert tr.cash_outs == {"c"}


# NOTE: ShakedownContext test removed — dataclass dropped from v1.1 scope
# per round-2 plan review MINOR-8 (the spec promised kind-parameterized
# helper extraction, but v1.1 ships two parallel methods instead).


def _ctx(
    *,
    placement: dict | None = None,
    positions: dict | None = None,
    tracked_buys: set | None = None,
) -> ClassifierContext:
    return ClassifierContext(
        tracked_buy_oids=tracked_buys or set(),
        placement_index=placement or {},
        open_positions_by_ticker_side=positions or {},
    )


def test_classify_buy_cash_out_when_opposite_side_present() -> None:
    fill = {"action": "buy", "order_id": "X", "market_ticker": "KXT", "side": "yes"}
    ctx = _ctx(positions={("KXT", "NO"): 2.34})
    assert classify_fill(fill, ctx) == FillKind.BUY_CASH_OUT


def test_classify_buy_entry_when_placement_row_present_even_if_opposite_position() -> None:
    fill = {"action": "buy", "order_id": "BOT-1", "market_ticker": "KXT", "side": "yes"}
    ctx = _ctx(
        placement={"BOT-1": {"strategy": "weather_edge", "action": "buy"}},
        positions={("KXT", "NO"): 2.34},
    )
    assert classify_fill(fill, ctx) == FillKind.BUY_ENTRY


def test_classify_buy_entry_fallthrough_when_no_match() -> None:
    fill = {"action": "buy", "order_id": "X", "market_ticker": "KXT", "side": "yes"}
    ctx = _ctx()
    assert classify_fill(fill, ctx) == FillKind.BUY_ENTRY


def test_classify_buy_cash_out_yes_held_no_buy() -> None:
    fill = {"action": "buy", "order_id": "X", "market_ticker": "KXT", "side": "no"}
    ctx = _ctx(positions={("KXT", "YES"): 2.34})
    assert classify_fill(fill, ctx) == FillKind.BUY_CASH_OUT


def test_classify_buy_cash_out_no_held_yes_buy() -> None:
    fill = {"action": "buy", "order_id": "X", "market_ticker": "KXT", "side": "yes"}
    ctx = _ctx(positions={("KXT", "NO"): 2.34})
    assert classify_fill(fill, ctx) == FillKind.BUY_CASH_OUT


def test_classify_pure_function_deterministic() -> None:
    fill = {"action": "buy", "order_id": "X", "market_ticker": "KXT", "side": "yes"}
    ctx = _ctx(positions={("KXT", "NO"): 1.0})
    assert classify_fill(fill, ctx) == classify_fill(fill, ctx)


def test_classify_unknown_on_missing_action() -> None:
    ctx = _ctx()
    assert classify_fill({"order_id": "X"}, ctx) == FillKind.UNKNOWN


def test_classify_sell_branches_preserved() -> None:
    """v1 sell branches must continue to work via the new signature."""
    fill_anomaly = {"action": "sell", "order_id": "BUY-1"}
    ctx = _ctx(tracked_buys={"BUY-1"})
    assert classify_fill(fill_anomaly, ctx) == FillKind.SELL_BOT_ANOMALY

    fill_bot_sell = {"action": "sell", "order_id": "SELL-1"}
    ctx = _ctx(placement={"SELL-1": {"strategy": "x", "action": "sell"}})
    assert classify_fill(fill_bot_sell, ctx) == FillKind.SELL_BOT

    fill_manual = {"action": "sell", "order_id": "USER-1"}
    ctx = _ctx()
    assert classify_fill(fill_manual, ctx) == FillKind.SELL_MANUAL_CANDIDATE


# ---------------------------------------------------------------------------
# Task 7: _build_open_positions_by_ticker_side helper (v1.1 §6.1)
# ---------------------------------------------------------------------------


def test_build_open_positions_by_ticker_side_single_record(tmp_path) -> None:
    # Use a test-harness reconciler with a fake engine._orders dict.
    reconciler = _make_minimal_reconciler(tmp_path)
    fake_signal = SimpleNamespace(position_key="KXT", market_ticker="KXT", side=Side.YES)
    fake_record = SimpleNamespace(
        signal=fake_signal, status=OrderStatus.FILLED,
    )
    reconciler._engine = SimpleNamespace(
        _orders={"O1": fake_record},
        _risk=SimpleNamespace(_positions={"KXT": 2.34}),
    )

    result = reconciler._build_open_positions_by_ticker_side()
    assert result == {("KXT", "YES"): 2.34}


def test_build_open_positions_skips_terminal_status(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    sig = SimpleNamespace(position_key="KXT", market_ticker="KXT", side=Side.YES)
    rec_settled = SimpleNamespace(signal=sig, status=OrderStatus.SETTLED)
    reconciler._engine = SimpleNamespace(
        _orders={"O1": rec_settled},
        _risk=SimpleNamespace(_positions={"KXT": 2.34}),
    )

    result = reconciler._build_open_positions_by_ticker_side()
    # No live record for that position → omitted.
    assert result == {}


def test_build_open_positions_empty_risk_returns_empty(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    reconciler._engine = SimpleNamespace(
        _orders={},
        _risk=SimpleNamespace(_positions={}),
    )
    assert reconciler._build_open_positions_by_ticker_side() == {}


# ---------------------------------------------------------------------------
# Task 9: _decrement_fill_counts_for_position helper (v1.1 §6.8)
# ---------------------------------------------------------------------------


def _make_record(
    *,
    position_key: str,
    fill_count: int,
    fees: float = 0.0,
    status=OrderStatus.FILLED,
):
    sig = SimpleNamespace(position_key=position_key)
    return SimpleNamespace(
        signal=sig,
        status=status,
        fill_count=fill_count,
        fees_from_api=fees,
    )


def test_decrement_fill_counts_single_record(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    rec = _make_record(position_key="KXT", fill_count=26, fees=0.10)
    reconciler._engine = SimpleNamespace(_orders={"O1": rec})

    reconciler._decrement_fill_counts_for_position(
        position_key="KXT", contracts_offset=25, cb_total_contracts=26
    )

    assert rec.fill_count == 1
    # Fees scale proportionally: 0.10 * (1/26) ≈ 0.0038, rounded to 4dp.
    assert rec.fees_from_api == pytest.approx(0.0038, abs=1e-4)


def test_decrement_fill_counts_multi_record_proportional(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    rec1 = _make_record(position_key="KXT", fill_count=20)
    rec2 = _make_record(position_key="KXT", fill_count=10)
    reconciler._engine = SimpleNamespace(_orders={"A": rec1, "B": rec2})

    # Cash-out 15 contracts across two records (20 + 10 = 30 total).
    # Proportional: 15 * 20/30 = 10, 15 * 10/30 = 5. Drift = 0.
    reconciler._decrement_fill_counts_for_position(
        position_key="KXT", contracts_offset=15, cb_total_contracts=30
    )

    assert rec1.fill_count == 10
    assert rec2.fill_count == 5


def test_decrement_fill_counts_skips_terminal_status(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    rec_settled = _make_record(position_key="KXT", fill_count=26, status=OrderStatus.SETTLED)
    reconciler._engine = SimpleNamespace(_orders={"O1": rec_settled})

    reconciler._decrement_fill_counts_for_position(
        position_key="KXT", contracts_offset=10, cb_total_contracts=26
    )

    # Settled records are skipped — fill_count unchanged.
    assert rec_settled.fill_count == 26


def test_decrement_fill_counts_no_matches_logs_critical(tmp_path, caplog) -> None:
    """No live records found for position_key → CRITICAL log; safe no-op."""
    reconciler = _make_minimal_reconciler(tmp_path)
    reconciler._engine = SimpleNamespace(_orders={})

    import logging
    caplog.set_level(logging.CRITICAL)
    reconciler._decrement_fill_counts_for_position(
        position_key="KXT", contracts_offset=10, cb_total_contracts=26
    )

    assert any("no matching live OrderRecords" in r.message for r in caplog.records)


def test_decrement_fill_counts_precondition_assert_on_full(tmp_path) -> None:
    """Precondition violation: contracts_offset == cb_total → AssertionError."""
    reconciler = _make_minimal_reconciler(tmp_path)
    rec = _make_record(position_key="KXT", fill_count=26)
    reconciler._engine = SimpleNamespace(_orders={"O1": rec})

    with pytest.raises(AssertionError):
        reconciler._decrement_fill_counts_for_position(
            position_key="KXT", contracts_offset=26, cb_total_contracts=26
        )


def test_decrement_fill_counts_clamps_below_zero(tmp_path, caplog) -> None:
    """Divergence: sum(matches) < contracts_offset → zero out + CRITICAL."""
    reconciler = _make_minimal_reconciler(tmp_path)
    rec = _make_record(position_key="KXT", fill_count=5)  # only 5 live; basis says 26
    reconciler._engine = SimpleNamespace(_orders={"O1": rec})

    import logging
    caplog.set_level(logging.CRITICAL)
    reconciler._decrement_fill_counts_for_position(
        position_key="KXT", contracts_offset=10, cb_total_contracts=26
    )

    assert rec.fill_count == 0
    assert any("sum(fill_count)" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_persist_state_includes_cash_out_dedup_sets(tmp_path: Path) -> None:
    """Spec §6.6 — five cash-out dedup sets persist to state file."""
    reconciler = _make_minimal_reconciler(tmp_path)
    reconciler._co_race_lost_oids = {"R1"}
    reconciler._co_zero_basis_oids = {"Z1"}
    reconciler._co_invalid_partial_oids = {"I1"}
    reconciler._co_pnl_cap_oids = {"P1"}
    reconciler._co_reverted_oids = {"V1"}

    await reconciler._persist_state()

    payload = json.loads((tmp_path / "fills_reconciler.json").read_text())
    assert payload["co_race_lost_oids"] == ["R1"]
    assert payload["co_zero_basis_oids"] == ["Z1"]
    assert payload["co_invalid_partial_oids"] == ["I1"]
    assert payload["co_pnl_cap_oids"] == ["P1"]
    assert payload["co_reverted_oids"] == ["V1"]


def test_load_state_legacy_file_empty_co_sets(tmp_path: Path) -> None:
    """A pre-v1.1 state file (missing co_* keys) loads with empty sets."""
    state_path = tmp_path / "fills_reconciler.json"
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "last_safe_ts_unix": 1000,
        "last_clean_completion_ts_unix": 0,
        "consecutive_failures": 0,
        "last_critical_alert_ts_unix": 0,
        "last_delta_critical_alert_ts_unix": 0,
        # No co_* keys — legacy v1 format.
    }))
    reconciler = _make_minimal_reconciler(tmp_path)
    # _load_state runs in __init__; new sets default to empty.
    assert reconciler._co_race_lost_oids == set()
    assert reconciler._co_zero_basis_oids == set()
    assert reconciler._co_invalid_partial_oids == set()
    assert reconciler._co_pnl_cap_oids == set()
    assert reconciler._co_reverted_oids == set()
    # The legacy file also predates the per-mode cooldown split (truncation/cycle
    # keys absent) → both default to 0 with no schema bump.
    assert reconciler._last_truncation_critical_alert_ts_unix == 0
    assert reconciler._last_cycle_critical_alert_ts_unix == 0


class TestFetchFillsPaginatedDelegation:
    """_fetch_fills_paginated now delegates to the shared paginate_detailed free
    function. Lock the empty-string terminal fix (divergence #2) and the
    PaginationResult -> dict mapping layer."""

    @pytest.mark.asyncio
    async def test_empty_string_cursor_is_terminal(self, tmp_path):
        """An empty-string cursor terminates pagination (no extra fetch). The old
        reconciler used `if cursor is None` and would have fetched once more."""
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._page_delay_ms = 0
        calls = {"n": 0}

        class _Client:
            async def get_fills(self, *, min_ts, cursor=None, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    return {"fills": [{"order_id": "A"}], "cursor": ""}
                raise AssertionError("must not fetch past an empty-string cursor")

        reconciler._kalshi_client = _Client()
        result = await reconciler._fetch_fills_paginated(min_ts=0)

        assert calls["n"] == 1
        assert result["fills"] == [{"order_id": "A"}]
        assert result["fills_count"] == 1
        assert result["pages_consumed"] == 1
        assert result["truncated"] is False
        assert result["cycle_aborted"] is False

    @pytest.mark.asyncio
    async def test_maps_pagination_result_fields_to_dict(self, tmp_path):
        """Two clean pages → the dict mirrors PaginationResult field-by-field."""
        reconciler = _make_minimal_reconciler(tmp_path)
        reconciler._max_pages_per_tick = 5
        reconciler._page_delay_ms = 0
        pages = [
            {"fills": [{"order_id": "A"}], "cursor": "c1"},
            {"fills": [{"order_id": "B"}], "cursor": None},
        ]

        class _Client:
            def __init__(self):
                self.i = 0

            async def get_fills(self, *, min_ts, cursor=None, **kw):
                page = pages[self.i]
                self.i += 1
                return page

        reconciler._kalshi_client = _Client()
        result = await reconciler._fetch_fills_paginated(min_ts=0)

        assert result["fills"] == [{"order_id": "A"}, {"order_id": "B"}]
        assert result["fills_count"] == 2
        assert result["pages_consumed"] == 2
        assert result["truncated"] is False
        assert result["cycle_aborted"] is False


def _multi_sell_fills() -> list[dict]:
    """Three sell fills: uneven counts, earliest ts at index 1, mixed is_taker.

    VWAP = (3*0.60 + 5*0.66 + 2*0.70) / 10 = 6.50 / 10 = 0.65
    fees = 0.01 + 0.02 + 0.01 = 0.04
    taker contracts = 3 + 2 = 5; maker contracts = 5.
    """
    return [
        {
            "order_id": "sell_oid", "action": "sell", "side": "yes",
            "yes_price_dollars": 0.60, "no_price_dollars": 0.40,
            "count_fp": 3, "fee_cost": 0.01, "is_taker": True,
            "created_time": "2026-05-17T12:00:01Z", "market_ticker": "K",
        },
        {
            "order_id": "sell_oid", "action": "sell", "side": "yes",
            "yes_price_dollars": 0.66, "no_price_dollars": 0.34,
            "count_fp": 5, "fee_cost": 0.02, "is_taker": False,
            "created_time": "2026-05-17T12:00:00Z", "market_ticker": "K",
        },
        {
            "order_id": "sell_oid", "action": "sell", "side": "yes",
            "yes_price_dollars": 0.70, "no_price_dollars": 0.30,
            "count_fp": 2, "fee_cost": 0.01, "is_taker": True,
            "created_time": "2026-05-17T12:00:02Z", "market_ticker": "K",
        },
    ]


def test_synthetic_row_raises_on_none_fee_cost(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    fills = _multi_sell_fills()
    fills[1]["fee_cost"] = None

    with pytest.raises(TypeError):
        reconciler._build_synthetic_row(
            order_id="sell_oid", kalshi_fills=fills,
            strategy="weather_edge", resolution_method="in_memory",
        )


def test_synthetic_row_raises_on_none_side(tmp_path) -> None:
    reconciler = _make_minimal_reconciler(tmp_path)
    fills = _multi_sell_fills()
    fills[0]["side"] = None

    with pytest.raises(AttributeError):
        reconciler._build_synthetic_row(
            order_id="sell_oid", kalshi_fills=fills,
            strategy="weather_edge", resolution_method="in_memory",
        )


@pytest.mark.asyncio
async def test_manual_close_multi_fill_aggregation_is_pinned(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine

    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0

    rec._trade_log_path.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    async with rec._trade_log_lock:
        outcome = await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=_multi_sell_fills(),
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )

    assert outcome == "APPLIED"
    mc = [json.loads(x) for x in rec._trade_log_path.read_text().splitlines()][-1]
    assert mc["contracts_closed"] == 10
    assert mc["close_price"] == pytest.approx(0.65)
    assert mc["close_proceeds"] == pytest.approx(6.50)
    assert mc["exit_fees"] == pytest.approx(0.04)
    assert mc["fees_from_api"] == pytest.approx(0.04)
    assert mc["is_taker_fills"] == 5
    assert mc["is_maker_fills"] == 5
    assert mc["timestamp"] == "2026-05-17T12:00:00Z"
    # 6.50 - 4.00 (basis) - 0.05 (entry fees) - 0.04 (exit fees)
    assert mc["realized_pnl"] == pytest.approx(2.41)


@pytest.mark.asyncio
async def test_manual_close_counts_absent_is_taker_toward_neither(tmp_path) -> None:
    from tests.conftest import make_reconciler_with_engine

    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0
    rec._trade_log_path.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    fills = _multi_sell_fills()
    fills[2].pop("is_taker")

    async with rec._trade_log_lock:
        await rec._apply_manual_close(
            order_id="sell_oid",
            kalshi_fills=fills,
            position_key="K-YES",
            market_ticker="K",
            tracked_manual_close_oids=set(),
        )

    mc = [json.loads(x) for x in rec._trade_log_path.read_text().splitlines()][-1]
    assert mc["is_taker_fills"] == 3
    assert mc["is_maker_fills"] == 5
    assert mc["contracts_closed"] == 10


@pytest.mark.asyncio
async def test_manual_close_full_sell_raises_on_none_fee_cost(tmp_path) -> None:
    """Full (non-partial) sell with a malformed fee must still raise today —
    this is the exact case an earlier draft of this plan would have silenced."""
    from tests.conftest import make_reconciler_with_engine

    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0
    rec._trade_log_path.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    fills = _multi_sell_fills()
    fills[1]["fee_cost"] = None

    with pytest.raises(TypeError):
        async with rec._trade_log_lock:
            await rec._apply_manual_close(
                order_id="sell_oid",
                kalshi_fills=fills,
                position_key="K-YES",
                market_ticker="K",
                tracked_manual_close_oids=set(),
            )
    assert rec._trade_log_path.read_text().count("\n") == 1  # only the seed row


@pytest.mark.asyncio
async def test_shakedown_multi_fill_pnl_is_pinned(tmp_path, caplog) -> None:
    import logging

    from tests.conftest import make_reconciler_with_engine

    rec, engine = make_reconciler_with_engine(tmp_path)
    rec._manual_close_enabled = False
    rec._shakedown_seen_oids_path = tmp_path / "shakedown_seen.json"
    rec._notifier = None
    rec._trade_log_path.write_text(
        json.dumps({
            "event": "fill", "order_id": "buy_oid",
            "position_key": "K-YES", "market_ticker": "K",
            "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
            "mode": "live", "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    risk = MagicMock()
    risk.positions = {"K-YES": 4.0}
    engine._risk = risk

    with caplog.at_level(logging.INFO, logger="python.execution.fills_reconciler"):
        async with rec._trade_log_lock:
            outcome = await rec._handle_shakedown(
                order_id="sell_oid",
                kalshi_fills=_multi_sell_fills(),
                position_key="K-YES",
                market_ticker="K",
                tracked_manual_close_oids=set(),
            )

    assert outcome == "SHAKEDOWN_ALERTED"
    msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    # proceeds 6.50, basis 4.00, entry fees 0.05, exit fees 0.04 → pnl 2.41
    assert any("proceeds_dollars=6.50" in m for m in msgs), msgs
    assert any("pnl_dollars=2.41" in m for m in msgs), msgs
    assert any("fees_dollars=0.09" in m for m in msgs), msgs
