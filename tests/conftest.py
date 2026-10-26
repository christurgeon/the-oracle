"""Shared fixtures for The Oracle test suite."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import Side
from python.execution.fills_reconciler import FillsReconciler
from python.notifications.base import AlertLevel
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal

if TYPE_CHECKING:
    from python.orchestrator import Orchestrator

# Tests assert the checked-in config.yaml, never a deployment's private overlay.
# Set (not unset) so the ``./.env`` fallback cannot re-enable it, and at import
# so subprocess-spawning tests inherit it.
os.environ["ORACLE_CONFIG_OVERLAY"] = ""
# Placeholder so ``--pull-vps`` code paths under test never target a real host.
os.environ["ORACLE_VPS_HOST"] = "op@vps.invalid"


@pytest.fixture
def monotonic_past_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shift ``time.monotonic`` a day forward.

    It counts from host boot, so on a CI runner that booted minutes ago
    ``time.monotonic() - 400`` is negative (RiskManager then reads
    ``_last_ws_data_time`` as "no data yet") and a 0.0 "never alerted"
    throttle stamp is still inside its cutoff.
    """
    real = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real() + 86_400.0)

# ---------------------------------------------------------------------------
# Helper (not a fixture — import where needed)
# ---------------------------------------------------------------------------


def _trade(pnl: float, won: bool, days_ago: int = 0) -> dict[str, Any]:
    """Build a trade dict for populating StrategyTracker.trades."""
    return {
        "pnl": pnl,
        "won": won,
        "ts": datetime.now(UTC) - timedelta(days=days_ago),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _sandbox_default_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default ``Settings().state_dir`` at a per-test tmp dir.

    The default is ``"state"`` relative to cwd, and the production checkout
    (/opt/oracle) is also the bot's cwd: a test that builds a strategy without
    explicit settings once appended fixture rows to the live
    ``state/per_day_cap_blocks.jsonl``.
    """
    monkeypatch.setenv("ORACLE_STATE_DIR", str(tmp_path / "default_state"))


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """State directory with no STOP file."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def make_settings(tmp_state_dir: Path) -> Callable[..., Settings]:
    """Factory that returns a Settings object with test defaults.

    Note: ``_env_file=None`` is a v1-fidelity convention — pydantic-settings
    v2 binds the env-file source via ``model_config`` and may not honor the
    init-kwarg. Tests that require full env isolation should also use
    ``monkeypatch.delenv`` on each ORACLE_* var. Future test isolation
    hardening tracked separately. (MINOR-4 round-2.)
    """

    def _factory(**overrides: Any) -> Settings:
        return Settings(state_dir=str(tmp_state_dir), _env_file=None, **overrides)

    return _factory


@pytest.fixture
def default_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def make_signal() -> Callable[..., TradeSignal]:
    """Factory that returns a TradeSignal with sensible defaults."""

    def _factory(**overrides: Any) -> TradeSignal:
        defaults: dict[str, Any] = {
            "market_id": "MKT-001",
            "market_ticker": "MKT-001-T",
            "model_probability": 0.60,
            "market_price": 0.50,
            "confidence": 0.8,
            "strategy_name": "weather_edge",
            "side": Side.YES,
            "suggested_dollars": 100.0,
            "suggested_contracts": 2,
        }
        defaults.update(overrides)
        return TradeSignal(**defaults)

    return _factory


@pytest.fixture
def risk_manager(default_settings: Settings) -> RiskManager:
    """Fresh RiskManager with default settings."""
    return RiskManager(default_settings)


@pytest.fixture
def stop_file_path(tmp_state_dir: Path) -> Path:
    """Path to the STOP file (does not create it)."""
    return tmp_state_dir / "STOP"


# ---------------------------------------------------------------------------
# Alert-relevance fixtures (Task 4)
#
# Shared by tests/test_alert_relevance.py and reused by Tasks 5, 7, 8, 9 — do
# not duplicate these in per-module fixtures. `orch` is a bare
# Orchestrator(settings=...) construction (the proven-safe pattern from
# tests/test_orchestrator.py — no async I/O happens at __init__ time) with a
# stub notifier pre-wired onto `_notifier`. Later tasks may freely set
# additional attributes on the yielded orchestrator (e.g. `orch._acked`,
# `orch._last_strategy_signal`) — Orchestrator has no __slots__.
# ---------------------------------------------------------------------------


class _StubNotifier:
    """Records every send_alert call as (message, level). Never raises.

    Mirrors the full python.notifications.base.Notifier.send_alert signature
    (including failure_bucket, notifications/base.py:41-45) — orchestrator.py
    calls it with failure_bucket= directly (e.g. :890), so a later task
    exercising that path through the `orch` fixture must not hit a TypeError.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, AlertLevel]] = []

    async def send_alert(
        self,
        message: str,
        *,
        level: AlertLevel = AlertLevel.INFO,
        failure_bucket: str | None = None,
    ) -> None:
        self.sent.append((message, level))


@pytest.fixture
def notifier() -> _StubNotifier:
    return _StubNotifier()


@pytest.fixture
def orch(default_settings: Settings, notifier: _StubNotifier) -> Orchestrator:
    """A plain Orchestrator with a stub notifier wired onto `_notifier`.

    Depends on the `notifier` fixture (rather than constructing its own) so a
    test that takes both `orch` and `notifier` parameters observes the same
    instance the orchestrator dispatches through.
    """
    from python.orchestrator import Orchestrator

    o = Orchestrator(settings=default_settings)
    o._notifier = notifier  # type: ignore[assignment]
    return o


@pytest.fixture
def tick_orch(orch: Orchestrator) -> tuple[Orchestrator, MagicMock, MagicMock]:
    """`orch` wired with mocked `_risk_manager`/`_engine` in a neutral,
    all-no-op shape that individual tests override piecemeal."""
    risk_mgr = MagicMock()
    risk_mgr.bankroll = 10_000.0
    risk_mgr.book_for.side_effect = lambda _name: risk_mgr.bankroll
    risk_mgr.is_halted = False
    risk_mgr.daily_pnl = 0.0
    risk_mgr.check_circuit_breakers.return_value = []
    risk_mgr.get_tracker.return_value = MagicMock(enabled=True, disabled_reason=None)
    risk_mgr.state_dirty = False
    orch._risk_manager = risk_mgr

    engine = MagicMock()
    engine.settle_expired_positions = AsyncMock(return_value=[])
    engine.check_all_pending_fills = AsyncMock(return_value=([], []))
    engine.execute_batch = AsyncMock(return_value=[])
    engine.collect_unalerted_fills = MagicMock(return_value=[])
    engine.prune_completed_orders = MagicMock()
    orch._engine = engine

    orch._running = True
    return orch, risk_mgr, engine


class _StubStrategy(Strategy):
    """Minimal concrete Strategy — the ABC's abstract methods are unused here.

    Same shape as tests/test_strategy_base.py::_Stub, duplicated (not
    imported) so conftest.py has no dependency on a sibling test module's
    private class.
    """

    async def scan(self) -> list[dict[str, Any]]:
        return []

    async def evaluate(self, market: dict[str, Any]) -> float | None:
        return None

    async def generate_signals(self) -> list[TradeSignal]:
        return []


def _stub(name: str, *, enabled: bool = True, shadow_only: bool = False) -> Strategy:
    """Build a minimal Strategy for tests that exercise configured_to_trade.

    Plain helper, not a fixture — import it directly (`from tests.conftest
    import _stub`) where a test needs a strategy instance rather than a
    fixture-injected one.
    """
    return _StubStrategy(name, {"enabled": enabled, "shadow_only": shadow_only})


# ---------------------------------------------------------------------------
# Manual-close reconciler helpers (Tasks 10-16, 20)
# ---------------------------------------------------------------------------


def make_reconciler_with_engine(tmp_path: Path) -> tuple[FillsReconciler, MagicMock]:
    """Plain helper (NOT a fixture) — construct a minimal FillsReconciler + engine
    pair for manual-close apply tests. Used by Tasks 10-16, 20.

    Returns (rec, engine). Tests call this directly with their own tmp_path so
    they can also use tmp_path for other things (e.g., shakedown_seen file paths).

    Caller-tunable attributes (defaults are flag-OFF, unlimited apply):
      - rec._manual_close_enabled
      - rec._manual_close_apply_limit
      - rec._shakedown_seen_oids_path
      - engine._orders
      - engine._risk
    """
    state = tmp_path / "fills_reconciler.json"
    log = tmp_path / "trades.jsonl"
    kalshi = AsyncMock()
    engine = MagicMock()
    engine._orders = {}
    notifier = AsyncMock()
    rec = FillsReconciler(
        state_path=state,
        trade_log_path=log,
        trade_log_lock=asyncio.Lock(),
        kalshi_client=kalshi,
        engine=engine,
        notifier=notifier,
    )
    return rec, engine


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Helper for seeding state/trades.jsonl in tests."""
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Cash-out v1.1 shared factories (Task 0 pre-flight)
# ---------------------------------------------------------------------------
# Lifted from v1's per-test-module helpers (Task 0). Downstream test modules
# (tests/test_cash_out_reconciler.py,
# tests/test_close_reconciliation_e2e.py) import these instead of duplicating
# construction boilerplate.
#
# The notifier protocol is python.notifications.base.Notifier:
#   send_alert(message: str, *, level: AlertLevel = INFO)
# v1's _MockNotifier in test_fills_reconciler.py uses keyword-only
# message=...; level=... — we replicate that here so cash-out tests use the
# same call shape.


def make_minimal_reconciler(
    *,
    state_path: Path,
    trade_log_path: Path,
) -> FillsReconciler:
    """Construct a FillsReconciler with minimal mock engine/risk/notifier.

    Use for unit tests that exercise the reconciler in isolation. The
    returned reconciler has:
      - engine._orders == {} (caller can populate)
      - engine._risk == SimpleNamespace(_positions={}, positions={})
      - notifier.sent_alerts captures (level, message) tuples

    Differs from v1's _make_minimal_reconciler (test_fills_reconciler.py:132)
    in that this variant wires a SimpleNamespace engine + capturing notifier
    by default, since cash-out unit tests typically need to assert on
    sent_alerts and engine state. v1 tests that pass engine=None / notifier=None
    continue using the local _make_minimal_reconciler helper.

    Both ``state_path`` and ``trade_log_path`` are required, keyword-only, so
    callers must pass tmp_path-scoped paths and never share /tmp defaults
    between tests (hermeticity rule, MINOR-10 round-2 fix).
    """

    engine = SimpleNamespace(
        _orders={},
        _risk=SimpleNamespace(_positions={}, positions={}),
    )

    class _CapturingNotifier:
        """Test-only notifier. Matches the production Notifier.send_alert
        signature (``message: str, *, level=...``) so cash-out tests that
        pass message positionally still work; v1 tests that call
        ``send_alert(level=..., message=...)`` keyword-style also work.
        """

        def __init__(self) -> None:
            self.sent_alerts: list[tuple[Any, str]] = []

        async def send_alert(
            self,
            message: str = "",
            *,
            level: Any = None,
            **_: Any,
        ) -> None:
            self.sent_alerts.append((level, message))

    notifier = _CapturingNotifier()

    class _NoopFillsClient:
        async def get_fills(self, **_: Any) -> dict[str, Any]:
            return {"fills": [], "cursor": None}

    # The Notifier Protocol is structurally satisfied by _CapturingNotifier
    # for the methods the reconciler actually calls (send_alert). We cast
    # via Any to skip the structural-subtype check on init-time wiring —
    # see MINOR-7 round-2 commentary.
    reconciler = FillsReconciler(
        state_path=state_path,
        trade_log_path=trade_log_path,
        trade_log_lock=asyncio.Lock(),
        kalshi_client=_NoopFillsClient(),
        engine=engine,
        notifier=notifier,  # type: ignore[arg-type]
    )
    return reconciler


def make_test_risk_manager(
    bankroll: float = 10_000.0,
    *,
    state_dir: Path,
) -> RiskManager:
    """Construct a real RiskManager instance for tests.

    Use this (NOT a SimpleNamespace mock) when a test must verify
    bankroll / daily_pnl / peak_bankroll mutations from
    realize_partial / close_position / record_strategy_trade.

    ``state_dir`` is required, keyword-only (MINOR-11 round-2 fix): there
    is no /tmp default — callers MUST pass a tmp_path-scoped directory so
    STOP-file probes and risk-state persistence are sandboxed per-test.

    Pattern mirrors v1's tests/test_risk_manager.py (via the ``make_settings``
    fixture): ``Settings(state_dir=..., _env_file=None)`` so test runs do
    NOT pick up the user's local .env. Note: ``_env_file=None`` is a v1
    convention — under pydantic-settings v2 the env-file source is bound
    via ``model_config`` and the init-kwarg isn't strictly honored, so it
    silently falls back to whatever ``model_config.env_file`` resolves to.
    For test isolation that matters, use ``monkeypatch.delenv`` on each
    ``ORACLE_*`` env var. Tracked separately; we keep v1 fidelity here.
    """
    settings = Settings(
        state_dir=str(state_dir),
        bankroll=bankroll,
        execution_mode="live",
        _env_file=None,
    )
    return RiskManager(settings)


class E2EHarness:
    """End-to-end harness wiring real engine + risk + reconciler.

    Used for full-lifecycle e2e tests (Tasks 18, 25, 26, 32). The harness
    exposes seed_fill / set_position / queue_cash_out_fill /
    queue_settlement / read_trade_log_event_count helpers so the test bodies
    stay short and readable.

    **Task 0 status:**
    The simple JSON-row / state-mutation helpers (``seed_fill``,
    ``set_position``, ``read_trade_log_event_count``, ``read_trade_log_rows``,
    ``write_placement_row``, ``set_cash_out_apply_enabled``) are fully
    implemented in this pre-flight task.

    The integration-glue methods (``__init__`` engine+reconciler wiring,
    ``queue_cash_out_fill``, ``queue_settlement``, ``add_fake_hydrated_record``,
    ``queue_kalshi_fill``, ``register_order_record``,
    ``persist_risk_state_to_disk``) are stubbed with ``NotImplementedError``
    pointing back to the implementation plan. They will be filled in by the
    Task 18/25/26/32 dispatches that introduce the e2e test files needing
    them; that's also when the cash-out-apply settings flags they depend on
    exist on the FillsReconciler.

    Calling a stubbed method raises immediately with a clear pointer to the
    plan section — there are no silent NoOps.
    """

    _STUB_PLAN_REF = (
        "this stub is intentionally unimplemented in the test harness"
    )

    # Type declarations for attributes set by ``_init_simple_helpers_only``
    # (lets mypy see them on the simple-helpers path).
    tmp_path: Path
    starting_bankroll: float
    trade_log: Path
    risk: RiskManager
    reconciler: FillsReconciler | None
    engine: Any
    _starting_daily_pnl: float
    # Full-wiring path additions (Task 18). Populated by
    # ``_init_full_wiring``; remain ``None`` / empty on the simple-helpers
    # path. Type-declared here so mypy sees them on both paths.
    _queued_cash_out_fills: list[dict[str, Any]]
    _queued_settlements: dict[str, dict[str, Any]]
    _kalshi_client: Any

    def __init__(self, tmp_path: Path, *, full_wiring: bool = False) -> None:
        """Construct an E2EHarness.

        By default (``full_wiring=False``) only the simple JSON-row /
        state-mutation helpers are wired — see ``_init_simple_helpers_only``.
        Pass ``full_wiring=True`` to opt into the real engine + reconciler
        integration path used by Tasks 18/25/26/32 e2e tests.

        Simple path attributes (always set):
          - ``tmp_path``, ``starting_bankroll``, ``trade_log``, ``risk``,
            ``_starting_daily_pnl``

        Full-wiring path additionally sets:
          - ``engine`` — real ``ExecutionEngine`` in ``live`` mode with a
            mocked Kalshi client
          - ``reconciler`` — real ``FillsReconciler`` sharing the engine's
            trade-log lock
          - ``_kalshi_client`` — ``AsyncMock``; ``get_fills`` / ``get_market``
            are dynamically populated by ``queue_cash_out_fill`` and
            ``queue_settlement``
          - ``_queued_cash_out_fills`` / ``_queued_settlements`` — queues
            drained by the mocked client on the next call
        """
        self._init_simple_helpers_only(tmp_path)
        self._queued_cash_out_fills = []
        self._queued_settlements = {}
        self._kalshi_client = None
        if full_wiring:
            self._init_full_wiring(tmp_path)

    def _init_full_wiring(self, tmp_path: Path) -> None:
        """Wire real ``ExecutionEngine`` + ``FillsReconciler`` with a mocked
        Kalshi client. Used by Task 18+ e2e tests that need ``_settle_live``.

        Implementation notes (kept terse — see harness docstring for the
        attribute contract):
          - The same ``RiskManager`` instance built by
            ``_init_simple_helpers_only`` is reused so ``set_position``
            mutations remain visible to the engine + reconciler.
          - The mocked Kalshi client serves queues populated by
            ``queue_cash_out_fill`` / ``queue_settlement``. Each ``get_fills``
            call drains the queue then returns ``{"fills": [], "cursor": None}``.
          - The reconciler is constructed with ``startup_safety_window_seconds=0``
            so any test-created fill (regardless of ``created_time``) is
            visible to the first tick.
        """
        from python.execution.engine import ExecutionEngine
        from python.execution.fills_reconciler import FillsReconciler
        from python.settings import Settings

        # Mock Kalshi client — async methods return from queues.
        kalshi = AsyncMock()
        self._kalshi_client = kalshi

        async def _get_fills(**_kwargs: Any) -> dict[str, Any]:
            # Drain entire queue in one page (test scenarios are small).
            fills = list(self._queued_cash_out_fills)
            self._queued_cash_out_fills.clear()
            return {"fills": fills, "cursor": None}

        async def _get_market(ticker: str) -> dict[str, Any]:
            queued = self._queued_settlements.get(ticker)
            if queued is None:
                # Default: unsettled (so _settle_live skips this position).
                return {"market": {"status": "open", "result": ""}}
            return {"market": queued}

        kalshi.get_fills = AsyncMock(side_effect=_get_fills)
        kalshi.get_market = AsyncMock(side_effect=_get_market)

        # Engine — real ExecutionEngine in live mode.
        engine_settings = Settings(
            state_dir=str(tmp_path),
            bankroll=self.starting_bankroll,
            execution_mode="live",
            _env_file=None,
        )
        trade_log_lock = asyncio.Lock()
        self.engine = ExecutionEngine(
            engine_settings,
            kalshi_client=kalshi,
            risk_manager=self.risk,
            trade_log_lock=trade_log_lock,
        )
        # Engine constructs its own trade-log path; point harness at it so
        # read_trade_log_event_count / seed_fill stay consistent.
        self.trade_log = self.engine._trade_log_path
        # Re-seed any rows that may already have been written via seed_fill
        # before full-wiring init — we intentionally clobber here so the
        # engine's path is authoritative. Tests that need pre-existing rows
        # should call seed_fill AFTER constructing the full-wiring harness.

        # Reconciler — real FillsReconciler sharing the engine's lock.
        notifier = AsyncMock()
        self.reconciler = FillsReconciler(
            state_path=tmp_path / "fills_reconciler.json",
            trade_log_path=self.trade_log,
            trade_log_lock=trade_log_lock,
            kalshi_client=kalshi,
            engine=self.engine,
            notifier=notifier,
            startup_safety_window_seconds=0,
            page_delay_ms=0,
        )

    def _init_simple_helpers_only(self, tmp_path: Path) -> None:
        """Wire the minimal attribute surface the simple helpers need.

        Sets:
          - ``tmp_path`` — sandbox dir for any helper-written files
          - ``starting_bankroll`` (100.0)
          - ``trade_log`` — touched-empty ``tmp_path/trades.jsonl``
          - ``risk`` — real ``RiskManager`` (so ``set_position`` mutates
            real state)
          - ``reconciler = None`` — Task 0 leaves reconciler wiring to the
            e2e dispatches; ``set_cash_out_apply_enabled`` guards on this
          - ``engine = None`` — same rationale
          - ``_starting_daily_pnl`` — captured from RiskManager at construct
            time (0.0 right after init)
        """
        self.tmp_path = tmp_path
        # 10_000 matches the production Settings default — anything smaller
        # fails the Settings sizing-floor validator
        # (kelly * single * exposure * bankroll >= 1.0).
        self.starting_bankroll = 10_000.0
        self.trade_log = tmp_path / "trades.jsonl"
        self.trade_log.touch()
        self.risk = make_test_risk_manager(
            bankroll=self.starting_bankroll,
            state_dir=tmp_path,
        )
        self.reconciler = None
        self.engine = None
        self._starting_daily_pnl = self.risk._daily_pnl

    # ------------------------------------------------------------------
    # Fully implemented helpers — safe to use from any test today.
    # ------------------------------------------------------------------

    def seed_fill(
        self,
        *,
        market_ticker: str,
        side: str,
        price: float,
        contracts: int,
        fees: float = 0.0,
        strategy: str = "weather_edge",
    ) -> None:
        """Inject a v1 fill row into the trade log."""
        row = {
            "event": "fill",
            "order_id": f"BOT-{market_ticker}",
            "market_ticker": market_ticker,
            "position_key": market_ticker,
            "side": side.upper(),
            "mode": "live",
            "fill_price": price,
            "contracts": contracts,
            "fees_from_api": fees,
            "metadata": {"strategy": strategy},
        }
        with self.trade_log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def set_position(self, position_key: str, *, dollars: float) -> None:
        """Seed a position in the harness's RiskManager."""
        self.risk._positions[position_key] = dollars

    def set_cash_out_apply_enabled(self, enabled: bool) -> None:
        """Toggle the cash-out apply flag on the wired reconciler.

        The attribute name (_co_apply_enabled) is the v1.1 spec naming;
        Task 1+ will add the attribute to FillsReconciler.__init__.

        On the simple-helpers path (``self.reconciler is None``) this is
        integration-glue territory and raises ``NotImplementedError`` with
        a plan pointer rather than silently NoOp'ing.
        """
        if self.reconciler is None:
            raise NotImplementedError(
                f"E2EHarness.set_cash_out_apply_enabled requires the engine+"
                f"reconciler wiring path; {self._STUB_PLAN_REF}. The simple-"
                f"helpers constructor leaves self.reconciler=None — "
                f"reconciler integration lands in Tasks 18/25/26/32."
            )
        # ``_co_apply_enabled`` is the v1.1-spec attribute name; Task 1+ adds
        # it to FillsReconciler.__init__. Forward-referenced here so the
        # downstream tests (post-Task-1) work without churn.
        self.reconciler._co_apply_enabled = enabled  # type: ignore[attr-defined]

    def read_trade_log_event_count(self, event: str) -> int:
        """Count rows in the trade log whose 'event' field matches."""
        count = 0
        with self.trade_log.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == event:
                    count += 1
        return count

    def read_trade_log_rows(self) -> list[dict[str, Any]]:
        """Return all parsed rows from the trade log (skips torn lines)."""
        rows: list[dict[str, Any]] = []
        with self.trade_log.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def write_placement_row(
        self,
        *,
        market_ticker: str,
        order_id: str,
        strategy: str,
        action: str = "buy",
    ) -> None:
        """Inject a placement row so classify_fill's placement_index match fires."""
        row = {
            "event": "placement",
            "order_id": order_id,
            "market_ticker": market_ticker,
            "action": action,
            "mode": "live",
            "metadata": {"strategy": strategy},
        }
        with self.trade_log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    # Stubbed integration-glue helpers — Task 0 defers to e2e dispatches.
    # ------------------------------------------------------------------

    def queue_cash_out_fill(
        self,
        *,
        market_ticker: str,
        hedge_side: str,
        price: float,
        contracts: int,
        order_id: str,
        fee_cost: float = 0.0,
    ) -> None:
        """Queue a cash-out hedge fill returned by the mocked kalshi_client on next tick.

        The hedge_side is the side the cash-out hedge was filled on
        (e.g., "no" hedges out a held YES position). The Kalshi /portfolio/fills
        record carries ``yes_price_dollars`` / ``no_price_dollars`` — we
        populate only the relevant key so ``_apply_cash_out`` Step 5 picks
        it up correctly.
        """
        if self._kalshi_client is None:
            raise NotImplementedError(
                f"E2EHarness.queue_cash_out_fill requires full-wiring path; "
                f"construct with E2EHarness(tmp_path, full_wiring=True). "
                f"{self._STUB_PLAN_REF}."
            )
        hs = hedge_side.lower()
        price_key = "yes_price_dollars" if hs == "yes" else "no_price_dollars"
        self._queued_cash_out_fills.append(
            {
                "order_id": order_id,
                "action": "buy",  # cash-out hedge is a BUY of the opposite side
                "side": hs,
                price_key: price,
                "count_fp": contracts,
                "fee_cost": fee_cost,
                "is_taker": True,
                "created_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "market_ticker": market_ticker,
            }
        )

    def queue_settlement(self, *, market_ticker: str, result: str) -> None:
        """Queue a settlement on the engine's pending settlements.

        The mocked kalshi_client.get_market(ticker) returns a payload
        shaped like ``{"market": {"status": "settled", "result": <result>,
        "ticker": <ticker>}}`` for any subsequent ``_settle_live`` call.
        """
        if self._kalshi_client is None:
            raise NotImplementedError(
                f"E2EHarness.queue_settlement requires full-wiring path; "
                f"construct with E2EHarness(tmp_path, full_wiring=True). "
                f"{self._STUB_PLAN_REF}."
            )
        self._queued_settlements[market_ticker] = {
            "status": "settled",
            "result": result,
            "ticker": market_ticker,
        }

    def add_fake_hydrated_record(
        self,
        *,
        market_ticker: str,
        position_key: str,
        side: str,
        fill_price: float = 0.09,
        contracts: int = 26,
        fees: float = 0.10,
    ) -> None:
        """Inject a FILLED OrderRecord so _settle_live's ticker_map sees the position.

        Despite the name (kept for plan-template compatibility), we register
        a status=FILLED record with a real fill_price/fill_count so
        _settle_orders has data to settle on — that's the race precondition
        the Task-18 guard protects against. A hydrated stub (fill_price=None)
        would also exercise the guard but with weaker assertions because
        _settle_orders skips its P&L branch.

        ``fill_price`` / ``contracts`` / ``fees`` parametrize the record so
        multi-position tests (e.g., the Task-18 race scenario) can produce
        distinguishable settlement P&Ls per market.
        """
        if self._kalshi_client is None:
            raise NotImplementedError(
                f"E2EHarness.add_fake_hydrated_record requires full-wiring "
                f"path; construct with E2EHarness(tmp_path, full_wiring=True). "
                f"{self._STUB_PLAN_REF}."
            )
        from python.common import ExecutionMode, OrderStatus, Side
        from python.execution.engine import OrderRecord
        from python.risk.manager import RiskDecision
        from python.strategies.base import TradeSignal

        side_enum = Side.YES if side.upper() == "YES" else Side.NO
        signal = TradeSignal(
            market_id=position_key,
            market_ticker=market_ticker,
            model_probability=0.0,
            market_price=0.0,
            confidence=0.0,
            strategy_name="weather_edge",
            side=side_enum,
            suggested_dollars=0.0,
            suggested_contracts=0,
            metadata={},
        )
        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=0.0)
        order_id = f"BOT-{position_key}"
        record = OrderRecord(
            order_id=order_id,
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.0,
            order_price=0.0,
            fill_price=fill_price,
            fill_count=contracts,
            requested_count=contracts,
            status=OrderStatus.FILLED,
            fees_from_api=fees,
            fills_payload_received=True,
        )
        self.engine._orders[order_id] = record

    def queue_kalshi_fill(
        self,
        *,
        market_ticker: str,
        side: str,
        price: float,
        contracts: int,
        order_id: str,
        fee_cost: float = 0.0,
    ) -> None:
        """Queue a generic Kalshi /portfolio/fills response for next tick.

        Distinct from queue_cash_out_fill — this is a bot-placed buy fill that
        the reconciler should classify as BUY_ENTRY via placement-row match.
        The "action" key is "buy" and the side matches the entry side (NOT the
        opposite, as queue_cash_out_fill does).
        """
        if self._kalshi_client is None:
            raise NotImplementedError(
                f"E2EHarness.queue_kalshi_fill requires full-wiring path; "
                f"construct with E2EHarness(tmp_path, full_wiring=True). "
                f"{self._STUB_PLAN_REF}."
            )
        sd = side.lower()
        price_key = "yes_price_dollars" if sd == "yes" else "no_price_dollars"
        self._queued_cash_out_fills.append(
            {
                "order_id": order_id,
                "action": "buy",
                "side": sd,
                price_key: price,
                "count_fp": contracts,
                "fee_cost": fee_cost,
                "is_taker": True,
                "created_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "market_ticker": market_ticker,
            }
        )

    def register_order_record(
        self,
        *,
        order_id: str,
        market_ticker: str,
        position_key: str,
        side: str,
        fill_count: int,
        fill_price: float,
        fees: float = 0.0,
        status: Any = None,
    ) -> Any:
        """Inject an OrderRecord into engine._orders so settlement / decrement
        paths find it. Returns the record so tests can assert on mutated state
        (fill_count decrement, status stamping).

        Used by Task 26 e2e to seed a FILLED record the reconciler's partial
        cash-out path can decrement (and _settle_live can subsequently read
        the decremented fill_count off of).
        """
        if self._kalshi_client is None:
            raise NotImplementedError(
                f"E2EHarness.register_order_record requires full-wiring path; "
                f"construct with E2EHarness(tmp_path, full_wiring=True). "
                f"{self._STUB_PLAN_REF}."
            )
        from python.common import ExecutionMode, OrderStatus, Side
        from python.execution.engine import OrderRecord
        from python.risk.manager import RiskDecision
        from python.strategies.base import TradeSignal

        side_str = side.upper()
        side_enum = Side.YES if side_str == "YES" else Side.NO
        signal = TradeSignal(
            market_id=position_key,
            market_ticker=market_ticker,
            model_probability=0.0,
            market_price=0.0,
            confidence=0.0,
            strategy_name="weather_edge",
            side=side_enum,
            suggested_dollars=0.0,
            suggested_contracts=0,
            metadata={},
        )
        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=0.0)
        if status is None:
            status = OrderStatus.FILLED
        record = OrderRecord(
            order_id=order_id,
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.0,
            order_price=0.0,
            fill_price=fill_price,
            fill_count=fill_count,
            requested_count=fill_count,
            status=status,
            fees_from_api=fees,
            fills_payload_received=True,
        )
        self.engine._orders[order_id] = record
        return record

    def persist_risk_state_to_disk(self) -> Path:
        """Serialize the in-memory RiskManager to state/risk_state.json + ensure
        state/fills_reconciler.json exists, then return the state dir path.

        Mirrors the orchestrator's shutdown serialize_risk_state path so the
        offline ``scripts/revert_cash_out.py`` CLI (which reads risk_state.json,
        fills_reconciler.json, and trades.jsonl from ``--state-dir``) can act
        on a snapshot of the harness's in-memory state.

        The script's ``_revert`` performs an up-front existence check on all
        three files (exit codes 11/12/13). On the full-wiring path the
        reconciler may not have persisted ``fills_reconciler.json`` yet — we
        write a minimal empty-set file in that case so the script's load step
        succeeds.
        """
        from python.risk.persistence import serialize_risk_state

        state_dir = self.tmp_path
        # Build the serialized payload the way the orchestrator does
        # (``_build_state_snapshot`` in python/orchestrator.py).
        payload = serialize_risk_state(
            **self.risk.export_state(),
            current_trading_date=None,
            execution_mode="live",
        )
        risk_state_path = state_dir / "risk_state.json"
        risk_state_path.write_text(json.dumps(payload, indent=2))

        # Ensure fills_reconciler.json exists so the script's load step
        # (exit code 12 on missing) doesn't error. If the wired reconciler
        # already persisted state via ``_persist_state``, leave it alone.
        fills_state_path = state_dir / "fills_reconciler.json"
        if not fills_state_path.exists():
            fills_state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "last_safe_ts_unix": 0,
                        "co_reverted_oids": [],
                    }
                )
            )
        return state_dir

    @property
    def starting_daily_pnl(self) -> float:
        """The harness records daily_pnl at construction time for tests
        that assert deltas after restart.
        """
        return self._starting_daily_pnl


@pytest.fixture
def make_e2e_harness(tmp_path: Path) -> Callable[..., E2EHarness]:
    """Pytest fixture providing the e2e harness factory.

    The factory returns a fully-constructed ``E2EHarness``. By default it
    opts into the full-wiring path (real engine + reconciler + risk + mocked
    Kalshi client) so the e2e tests in Tasks 18/25/26/32 can call helpers
    like ``queue_cash_out_fill`` / ``queue_settlement`` / ``register_order_record``
    without extra boilerplate. Callers that only need the simple JSON-row
    helpers can pass ``full_wiring=False``.
    """

    def factory(
        path: Path | None = None, *, full_wiring: bool = True
    ) -> E2EHarness:
        return E2EHarness(path or tmp_path, full_wiring=full_wiring)

    return factory
