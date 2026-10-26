"""Tests for orchestrator-level behaviour that doesn't require a full async stack."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ET
from python.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

    from python.orchestrator import Orchestrator
    from python.risk.manager import RiskManager
    from tests.conftest import _StubNotifier


def _make_orch_settings(tmp_path, *, mode: str, fr_enabled: bool) -> Settings:
    """Build a minimal Settings for orchestrator wiring tests.

    Avoids loading config.yaml — Pydantic defaults are sufficient for the
    construction-logic invariants tested here. ``edge_threshold`` must
    exceed the fee floor (0.02) per Settings._check_edge_above_fee_floor.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state_dir),
        execution={"mode": mode},
        fills_reconciler={"enabled": fr_enabled},
        strategies={"weather": {"edge_threshold": 0.07}},
        _env_file=None,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock is POSIX-only")
def test_pidlock_acquired_on_startup(tmp_path):
    """Orchestrator startup acquires state/trades.jsonl.pidlock; second
    instance trying to acquire the same lockfile gets BlockingIOError.

    The kernel releases the lock when the holding FD is closed (or the
    process exits), so a stale lockfile on disk does not block a new
    orchestrator. fcntl.flock kernel semantics are the source of truth.
    """
    from python.orchestrator import _acquire_trade_log_pidlock

    pidlock_path = tmp_path / "trades.jsonl.pidlock"

    # First acquisition succeeds and returns an open FD.
    # POSIX postcondition: a valid FD is >= 0 (FDs 0/1/2 are stdin/stdout/stderr,
    # but os.open never returns those in a normal test context — the >= 0 form
    # is the conventional invariant and avoids a spurious failure if some
    # earlier test closed fd 0 and the OS reused it).
    fd1 = _acquire_trade_log_pidlock(pidlock_path)
    try:
        assert fd1 >= 0

        # Second acquisition blocks
        with pytest.raises(BlockingIOError):
            _acquire_trade_log_pidlock(pidlock_path)
    finally:
        os.close(fd1)

    # After the first FD is closed, kernel releases the lock; a new
    # acquisition succeeds
    fd2 = _acquire_trade_log_pidlock(pidlock_path)
    try:
        assert fd2 >= 0
    finally:
        os.close(fd2)


# ─────────────────────────────────────────────────────────────────────────
# Task 27 — FillsReconciler wiring (spec §5.3)
#
# These tests exercise the construction-logic helper directly, because the
# full Orchestrator.initialize() does heavy async I/O (Kalshi connect,
# weather connect) that's out of scope for a pure wiring assertion. The
# helper is the actual decision point: "should we build a reconciler?"
# ─────────────────────────────────────────────────────────────────────────


class _FakeKalshiClient:
    """Minimal stand-in for KalshiClient — _setup_fills_reconciler only
    needs `is not None`, so a sentinel object is sufficient."""


def _make_orch_with_engine(tmp_path, *, mode: str, fr_enabled: bool):
    """Construct an Orchestrator and seed the fields the wiring helper
    reads (engine, kalshi_client). Returns the orchestrator."""
    from python.orchestrator import Orchestrator

    settings = _make_orch_settings(tmp_path, mode=mode, fr_enabled=fr_enabled)
    orch = Orchestrator(settings=settings)

    # Seed the dependencies the helper expects — both fields are populated
    # by initialize() in production. ExecutionEngine accepts None for the
    # optional kalshi/risk/contract_watcher kwargs (see engine.py:215+).
    from python.execution.engine import ExecutionEngine
    from python.risk.manager import RiskManager

    orch._kalshi_client = _FakeKalshiClient()  # type: ignore[assignment]
    orch._risk_manager = RiskManager(settings)
    orch._engine = ExecutionEngine(
        settings=settings,
        kalshi_client=None,
        risk_manager=orch._risk_manager,
        contract_watcher=None,
        trade_log_lock=orch._trade_log_lock,
    )
    return orch


def test_orchestrator_constructs_fills_reconciler_when_enabled(tmp_path):
    """When fills_reconciler.enabled and execution.mode in {live, paper},
    the orchestrator instantiates FillsReconciler."""
    orch = _make_orch_with_engine(tmp_path, mode="paper", fr_enabled=True)
    orch._setup_fills_reconciler()
    assert orch._fills_reconciler is not None


def test_orchestrator_skips_reconciler_when_disabled(tmp_path):
    orch = _make_orch_with_engine(tmp_path, mode="paper", fr_enabled=False)
    orch._setup_fills_reconciler()
    assert orch._fills_reconciler is None


def test_orchestrator_skips_reconciler_in_shadow_mode(tmp_path):
    """Shadow mode never has real Kalshi fills — reconciler is unnecessary
    and would query an empty endpoint forever."""
    orch = _make_orch_with_engine(tmp_path, mode="shadow", fr_enabled=True)
    orch._setup_fills_reconciler()
    assert orch._fills_reconciler is None


def test_orchestrator_engine_and_reconciler_share_trade_log_lock(tmp_path):
    """Spec §5.3 dedup invariant: engine and reconciler MUST share the same
    asyncio.Lock instance, otherwise the locked critical section in
    _tick_impl can race against engine writes."""
    orch = _make_orch_with_engine(tmp_path, mode="paper", fr_enabled=True)
    orch._setup_fills_reconciler()
    assert orch._fills_reconciler is not None
    # Identity check — same object, not just equal.
    assert orch._engine._trade_log_lock is orch._trade_log_lock
    assert orch._fills_reconciler._trade_log_lock is orch._trade_log_lock


def test_et_today_returns_a_date():
    from datetime import date

    from python.orchestrator import Orchestrator

    result = Orchestrator._et_today()
    assert isinstance(result, date)


# ─────────────────────────────────────────────────────────────────────────
# Task 2 — _restore_risk_state refactor
#
# These tests verify that the startup path routes through reset_daily()
# for both mode-switch and day-boundary cases, rather than mutating the
# snapshot directly.
# ─────────────────────────────────────────────────────────────────────────


def _make_restore_orch(tmp_path, *, execution_mode: str = "live"):
    from python.orchestrator import Orchestrator
    orch = Orchestrator.__new__(Orchestrator)
    orch._settings = MagicMock()
    orch._settings.state_dir = str(tmp_path)
    orch._settings.execution.mode = execution_mode
    rm = MagicMock()
    rm.bankroll = 199.99   # deliberately not a round number for assertion clarity
    rm.hydrate_from_state = MagicMock()
    rm.reset_daily = MagicMock()
    orch._risk_manager = rm
    orch._current_trading_date = None
    # _reset_daily_with_fanout iterates self._strategies — seed empty list so
    # the helper doesn't AttributeError in tests that don't register strategies.
    orch._strategies = []
    return orch, rm


def _write_state_file(
    tmp_path,
    *,
    date_str: str | None = None,
    execution_mode: str = "live",
    daily_pnl: float = -10.0,
    peak_bankroll: float = 210.0,
) -> None:
    from python.risk.persistence import STATE_FILENAME, serialize_risk_state
    data = serialize_risk_state(
        peak_bankroll=peak_bankroll,
        daily_pnl=daily_pnl,
        daily_start_bankroll=205.0,
        trading_halted=True,
        strategy_trackers={},
        current_trading_date=date_str,
        execution_mode=execution_mode,
    )
    (tmp_path / STATE_FILENAME).write_text(json.dumps(data))


def test_restore_risk_state_new_day_calls_reset_daily(tmp_path):
    """When stored date differs from today, reset_daily() must be called."""
    orch, rm = _make_restore_orch(tmp_path)
    _write_state_file(tmp_path, date_str="2000-01-01")
    today = date(2026, 5, 9)
    orch._restore_risk_state(_today=today)
    rm.reset_daily.assert_called_once()


def test_restore_risk_state_same_day_no_reset(tmp_path):
    """When stored date matches today, reset_daily() must NOT be called."""
    orch, rm = _make_restore_orch(tmp_path)
    today = date(2026, 5, 9)
    _write_state_file(tmp_path, date_str=today.isoformat())
    orch._restore_risk_state(_today=today)
    rm.reset_daily.assert_not_called()


def test_restore_risk_state_mode_switch_calls_reset_daily(tmp_path):
    """Mode switch (shadow→live) must call reset_daily() even on the same day."""
    orch, rm = _make_restore_orch(tmp_path, execution_mode="live")
    today = date(2026, 5, 9)
    _write_state_file(tmp_path, date_str=today.isoformat(), execution_mode="shadow")
    orch._restore_risk_state(_today=today)
    rm.reset_daily.assert_called_once()


def test_restore_risk_state_mode_switch_resets_peak_via_hydrate(tmp_path):
    """Mode switch must overwrite snapshot.peak_bankroll with rm.bankroll before hydration.

    Uses a stored peak (210.0) that differs from rm.bankroll (199.99) to avoid
    a trivially-passing assertion.
    """
    orch, rm = _make_restore_orch(tmp_path, execution_mode="live")
    today = date(2026, 5, 9)
    _write_state_file(
        tmp_path,
        date_str=today.isoformat(),
        execution_mode="shadow",
        peak_bankroll=210.0,
    )
    orch._restore_risk_state(_today=today)
    call_args = rm.hydrate_from_state.call_args
    snapshot_arg = call_args[0][0]
    assert snapshot_arg.peak_bankroll == rm.bankroll   # 199.99, not 210.0
    assert snapshot_arg.peak_bankroll != 210.0


def test_restore_risk_state_none_current_date_falls_back_to_today(tmp_path):
    """When snapshot.current_trading_date is None, _current_trading_date must equal _today."""
    orch, rm = _make_restore_orch(tmp_path)
    today = date(2026, 5, 9)
    _write_state_file(tmp_path, date_str=None)
    orch._restore_risk_state(_today=today)
    assert orch._current_trading_date == today
    rm.reset_daily.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Task 20 — v1.1 cash-out config wiring (spec §6.5 / §6.6)
# ─────────────────────────────────────────────────────────────────────────


def test_orchestrator_wires_cash_out_config(tmp_path):
    """The orchestrator must populate _co_apply_enabled, _co_apply_limit,
    _co_max_pnl_apply_dollars, and _shakedown_seen_oids_cash_out_path on
    the reconciler from FillsReconcilerConfig.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    settings = Settings(
        state_dir=str(state_dir),
        execution={"mode": "live"},
        fills_reconciler={
            "enabled": True,
            "cash_out_apply_enabled": True,
            "cash_out_apply_limit_per_restart": 5,
            "cash_out_max_pnl_apply_dollars": 250.0,
            "shakedown_seen_oids_cash_out_file": "state/shakedown_seen_oids_cash_out.json",
        },
        strategies={"weather": {"edge_threshold": 0.07}},
        _env_file=None,
    )

    from python.execution.engine import ExecutionEngine
    from python.orchestrator import Orchestrator
    from python.risk.manager import RiskManager

    orch = Orchestrator(settings=settings)
    orch._kalshi_client = _FakeKalshiClient()  # type: ignore[assignment]
    orch._risk_manager = RiskManager(settings)
    orch._engine = ExecutionEngine(
        settings=settings,
        kalshi_client=None,
        risk_manager=orch._risk_manager,
        contract_watcher=None,
        trade_log_lock=orch._trade_log_lock,
    )

    orch._setup_fills_reconciler()

    reconciler = orch._fills_reconciler
    assert reconciler is not None
    assert reconciler._co_apply_enabled is True
    assert reconciler._co_apply_limit == 5
    assert reconciler._co_max_pnl_apply_dollars == 250.0
    assert reconciler._shakedown_seen_oids_cash_out_path == (
        state_dir / "shakedown_seen_oids_cash_out.json"
    )


# ─────────────────────────────────────────────────────────────────────────
# Task 5 — tripwire dispatch bookkeeping
# ─────────────────────────────────────────────────────────────────────────


async def test_suppressed_alert_is_not_recorded_as_delivered(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """A suppressed plateau alert must not be persisted as seen.

    _tripwire_alerted_codes.add ran BEFORE dispatch and persist_seen_code
    ran after it unconditionally, so a suppressed alert was written to disk,
    re-seeded at startup (orchestrator.py:1130) and NEVER fired again — silently
    disabling the detector that exists because of the 2026-04-22 CHI sign-flip
    loss.
    """
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    alert = DriftAlert(
        code="calibrator_plateau:2026-07-04:0.400-0.800",
        severity=AlertLevel.WARNING,
        message="flat segment",
        detail={},
        owner="weather_edge",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert alert.code not in orch._tripwire_alerted_codes
    assert not (tmp_path / "alert_acks.json").exists()


async def test_dispatched_plateau_alert_is_persisted_as_seen(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """Cross-restart dedup must survive. Deleting the persist outright would make
    every plateau alert re-fire on every restart — the nagging this spec removes.
    """
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=False)]
    alert = DriftAlert(
        code="calibrator_plateau:2026-07-04:0.400-0.800",
        severity=AlertLevel.WARNING,
        message="flat segment",
        detail={},
        owner="weather_edge",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert alert.code in orch._tripwire_alerted_codes
    assert (tmp_path / "alert_acks.json").exists()


# ─────────────────────────────────────────────────────────────────────────
# Task 8 — populating _ackable
# ─────────────────────────────────────────────────────────────────────────


async def test_dispatch_records_ackable_code_with_owner_and_severity(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """An alert that declares an ack_key must become /ack-able, recording the
    (owner, severity) /ack needs to persist a verdict."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    alert = DriftAlert(
        code="calibrator_artifact_stale:favorite_bias:2026-07-04",
        severity=AlertLevel.WARNING,
        message="stale artifact",
        detail={},
        owner="favorite_bias",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert orch._ackable[alert.code] == ("favorite_bias", "WARNING")


async def test_dispatch_does_not_record_a_non_ackable_alert(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """No ack_key means no invalidation context — /ack must refuse it, so it
    must never appear in _ackable."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel

    alert = DriftAlert(
        code="audit_log_stale:eval_audit.jsonl",
        severity=AlertLevel.WARNING,
        message="writer stalled",
        detail={},
        owner=None,
        ack_key=None,
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert alert.code not in orch._ackable


async def test_dispatch_records_ackable_even_when_alert_is_suppressed(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """A code must be /ack-able even when this sweep's copy was suppressed
    (owner not live) — the operator can still see the code in logs/history
    and should be able to act on it before the owner ever goes live."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    alert = DriftAlert(
        code="calibrator_plateau:2026-07-04:0.400-0.800",
        severity=AlertLevel.WARNING,
        message="flat segment",
        detail={},
        owner="weather_edge",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert orch._ackable[alert.code] == ("weather_edge", "WARNING")
    # Suppressed, so it must NOT have reached the dispatched-dedup set.
    assert alert.code not in orch._tripwire_alerted_codes


async def test_dispatch_records_ackable_for_already_dispatched_code(
    orch: Orchestrator, tmp_path: Path
) -> None:
    """A code already in _tripwire_alerted_codes (hourly re-sweep of the same
    finding) still updates _ackable — the early `continue` must not skip it."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    code = "calibrator_artifact_stale:favorite_bias:2026-07-04"
    orch._tripwire_alerted_codes.add(code)
    alert = DriftAlert(
        code=code,
        severity=AlertLevel.CRITICAL,
        message="still stale",
        detail={},
        owner="favorite_bias",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert orch._ackable[code] == ("favorite_bias", "CRITICAL")


async def test_ackable_alert_message_carries_ack_hint(
    orch: Orchestrator, notifier: _StubNotifier, tmp_path: Path
) -> None:
    """Spec §3.2: an ack-able alert's Telegram message must say so — the
    operator has no way to discover /ack for a specific code otherwise."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    alert = DriftAlert(
        code="calibrator_artifact_stale:favorite_bias:2026-07-04",
        severity=AlertLevel.WARNING,
        message="stale artifact",
        detail={},
        owner="favorite_bias",
        ack_key="2026-07-04",
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert len(notifier.sent) == 1
    sent_message = notifier.sent[0][0]
    assert "reply /ack calibrator_artifact_stale:favorite_bias:2026-07-04" in sent_message


async def test_non_ackable_alert_message_has_no_ack_hint(
    orch: Orchestrator, notifier: _StubNotifier, tmp_path: Path
) -> None:
    """An alert with no `ack_key` (no invalidating context) must not claim
    /ack is available — acking it would be a permanent mute."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel

    alert = DriftAlert(
        code="audit_log_stale:eval_audit.jsonl",
        severity=AlertLevel.WARNING,
        message="writer stalled",
        detail={},
        owner=None,
        ack_key=None,
    )
    await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
    assert len(notifier.sent) == 1
    sent_message = notifier.sent[0][0]
    assert "reply /ack" not in sent_message


async def test_suppressed_alert_logs_warning_once_not_every_sweep(
    orch: Orchestrator, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A code suppressed every sweep (owner not live here) never enters
    `_tripwire_alerted_codes`, so the log-line dedup must use a SEPARATE set
    (`_tripwire_logged_codes`) — otherwise the `logger.warning("Audit
    tripwire: ...")` line re-fires on every hourly sweep forever, diluting
    the WARNING channel real risk alerts live on."""
    from python.analytics.calibration_monitor import DriftAlert
    from python.notifications.base import AlertLevel
    from tests.conftest import _stub

    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    alert = DriftAlert(
        code="bias_table_stale",
        severity=AlertLevel.WARNING,
        message="bias table is stale",
        detail={},
        owner="weather_edge",
        ack_key=None,
    )
    with caplog.at_level("WARNING"):
        await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
        await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)
        await orch._dispatch_tripwire_alerts([alert], state_dir=tmp_path)

    # Still never dispatched (owner not live) — confirms this is exercising
    # the suppressed path, not the already-dispatched one.
    assert alert.code not in orch._tripwire_alerted_codes
    warning_lines = [
        rec for rec in caplog.records if rec.getMessage().startswith("Audit tripwire:")
    ]
    assert len(warning_lines) == 1


# ---------------------------------------------------------------------------
# `_alert_is_relevant` is blind to circuit-breaker state.
#
# Replaces tests/test_strategy_base.py::test_ignores_circuit_breaker_state,
# which was byte-identical to test_absent_shadow_only_trades: `Strategy` has
# no breaker field, so asserting on `Strategy.configured_to_trade` alone
# could never fail regardless of circuit-breaker state. The property that
# actually matters lives at the orchestrator boundary — a breaker trip must
# NOT feed into `_alert_is_relevant`, because a breaker trips exactly when
# something is going wrong and the alert must keep firing.
# ---------------------------------------------------------------------------


def test_alert_is_relevant_ignores_a_tripped_circuit_breaker(
    orch: Orchestrator, risk_manager: RiskManager
) -> None:
    """A strategy whose RiskManager tracker is disabled (circuit-broken) is
    still `configured_to_trade` from the orchestrator's point of view — the
    breaker lives on the RiskManager tracker, never on `Strategy.enabled`
    (orchestrator.py:1846-1850) — so `_alert_is_relevant` must still return
    True for it."""
    from tests.conftest import _stub

    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    orch._risk_manager = risk_manager
    tracker = risk_manager.get_tracker("favorite_bias")
    tracker.enabled = False
    tracker.disabled_reason = "circuit breaker tripped: daily loss limit"

    assert orch._alert_is_relevant("favorite_bias") is True


# ---------------------------------------------------------------------------
# Per-strategy liveness detector (Task 9)
# ---------------------------------------------------------------------------


async def test_per_strategy_silence_alerts_for_a_live_strategy(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """SILENT BOT cannot catch this: _last_signal_time is refreshed by ANY
    strategy and pre-shadow-drop (orchestrator.py:1889-1896), so weather_edge's
    zero-order shadow signals keep it quiet while favorite_bias is dead."""
    from tests.conftest import _stub

    orch._strategies = [
        _stub("weather_edge", enabled=True, shadow_only=True),
        _stub("favorite_bias", enabled=True, shadow_only=False),
    ]
    orch._settings.telegram.strategy_silent_alert_hours = 6.0
    orch._last_strategy_signal = {"favorite_bias": time.monotonic() - 10 * 3600}
    await orch._check_strategy_liveness(in_market_hours=True)
    assert any("favorite_bias" in m for m, _level in notifier.sent)


async def test_shadow_only_strategy_never_alerts(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    from tests.conftest import _stub

    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    orch._settings.telegram.strategy_silent_alert_hours = 6.0
    orch._last_strategy_signal = {"weather_edge": time.monotonic() - 10 * 3600}
    await orch._check_strategy_liveness(in_market_hours=True)
    assert notifier.sent == []


async def test_check_strategy_liveness_never_raises(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """A duck-typed / misbehaving strategy must not crash the check.

    Fix-round-1 item 1: _check_strategy_liveness sits directly in the _tick
    critical path (before _persist_if_dirty, the healthchecks ping, and the
    _consecutive_tick_failures reset), so an unhandled exception here has a
    structural path to the kill switch via _TICK_FAILURE_KILL_THRESHOLD. It
    must be never-raises, matching _audit_tripwire_loop and _notify_alert.
    """

    class _BoomStrategy:
        name = "favorite_bias"

        @property
        def configured_to_trade(self) -> bool:
            raise RuntimeError("simulated attribute failure")

    orch._strategies = [_BoomStrategy()]  # type: ignore[list-item]
    orch._settings.telegram.strategy_silent_alert_hours = 6.0
    orch._last_strategy_signal = {"favorite_bias": time.monotonic() - 10 * 3600}
    await orch._check_strategy_liveness(in_market_hours=True)  # must not raise
    assert notifier.sent == []


async def test_tick_runs_liveness_check_when_another_strategy_produced_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the HOISTING requirement.

    _check_strategy_liveness must run every tick, including when the
    `if all_signals:` branch is taken because another strategy produced
    signals — that is precisely the case the aggregate SILENT BOT watchdog
    cannot see. Drives the real ``_tick()`` (not a direct call to
    ``_check_strategy_liveness``), so a future re-nest of the call back
    inside the no-signals `else` branch would make this test fail.
    """
    import python.orchestrator as orch_mod
    from python.common import Side
    from python.orchestrator import Orchestrator
    from python.strategies.base import TradeSignal
    from tests.conftest import _StubNotifier

    # Pin "now" to a weekday inside market hours (Tuesday noon ET) so
    # in_market_hours is deterministic regardless of when the suite runs.
    fixed_now = datetime(2026, 5, 5, 12, 0, tzinfo=ET)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return fixed_now

    monkeypatch.setattr(orch_mod, "datetime", _FixedDatetime)

    orch = Orchestrator()
    orch._running = True

    risk_mgr = MagicMock()
    risk_mgr.bankroll = 10_000.0
    risk_mgr.is_halted = False
    risk_mgr.daily_pnl = 0.0
    risk_mgr.check_circuit_breakers.return_value = []
    risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
    risk_mgr.state_dirty = False
    orch._risk_manager = risk_mgr

    engine = MagicMock()
    engine.settle_expired_positions = AsyncMock(return_value=[])
    engine.check_all_pending_fills = AsyncMock(return_value=([], []))
    engine.execute_batch = AsyncMock(return_value=[])
    engine.collect_unalerted_fills = MagicMock(return_value=[])
    orch._engine = engine

    notifier = _StubNotifier()
    orch._notifier = notifier  # type: ignore[assignment]

    settings = MagicMock()
    settings.is_kill_switch_active.return_value = False
    settings.is_shadow = True
    settings.telegram.silent_bot_alert_hours = 6.0
    settings.telegram.strategy_silent_alert_hours = 6.0
    orch._settings = settings

    signal = TradeSignal(
        market_id="MKT-001",
        market_ticker="MKT-001-T",
        model_probability=0.60,
        market_price=0.50,
        confidence=0.8,
        # Deliberately NOT one of the four names any shadow-drop/deference hook
        # special-cases (weather_edge / favorite_bias / intraday_emos_high /
        # intraday_asos_high) — this signal must survive every hook untouched
        # so `if all_signals:` is unambiguously the branch _tick() takes.
        strategy_name="other_strategy",
        side=Side.YES,
        suggested_dollars=100.0,
        suggested_contracts=2,
    )

    class _SignalProducingStrategy:
        name = "other_strategy"
        enabled = True
        configured_to_trade = True

        def update_bankroll(self, *_a: object, **_k: object) -> None:
            pass

        async def generate_signals(self) -> list[TradeSignal]:
            return [signal]

    class _SilentLiveStrategy:
        name = "favorite_bias"
        enabled = True
        configured_to_trade = True

        def update_bankroll(self, *_a: object, **_k: object) -> None:
            pass

        async def generate_signals(self) -> list[TradeSignal]:
            return []

    orch._strategies = [_SignalProducingStrategy(), _SilentLiveStrategy()]  # type: ignore[list-item]
    orch._last_strategy_signal = {"favorite_bias": time.monotonic() - 10 * 3600}

    await orch._tick()

    assert any("favorite_bias" in m for m, _level in notifier.sent), (
        "per-strategy liveness alert did not fire on a tick where ANOTHER "
        "strategy produced signals — the check may have been re-nested "
        "inside the no-signals `else` branch"
    )


# ---------------------------------------------------------------------------
# Per-strategy liveness threshold is independent of the aggregate SILENT BOT
# watchdog (follow-up to Task 9)
# ---------------------------------------------------------------------------


async def test_strategy_liveness_uses_its_own_threshold_not_the_silent_bot_one(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """A silence longer than the aggregate SILENT BOT threshold but shorter than
    the per-strategy one must NOT alert.

    favorite_bias trades lead-1, band 0.50-0.75, greater/less only, one bet per
    (city, var, forecast_date) per day; the 2026-08-11 band change cut flow ~40%
    (first post-change scan: 2 in-band markets of 204). A full day with zero
    signals is an ordinary outcome, so the two thresholds must be tunable apart.
    """
    from tests.conftest import _stub

    orch._settings.telegram.silent_bot_alert_hours = 6.0
    orch._settings.telegram.strategy_silent_alert_hours = 30.0
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    orch._last_strategy_signal = {"favorite_bias": time.monotonic() - 10 * 3600}

    await orch._check_strategy_liveness(in_market_hours=True)

    assert notifier.sent == []


async def test_strategy_liveness_alerts_past_its_own_threshold(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """Past the per-strategy threshold it still fires, quoting elapsed hours."""
    from tests.conftest import _stub

    orch._settings.telegram.silent_bot_alert_hours = 6.0
    orch._settings.telegram.strategy_silent_alert_hours = 30.0
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    orch._last_strategy_signal = {"favorite_bias": time.monotonic() - 31 * 3600}

    await orch._check_strategy_liveness(in_market_hours=True)

    assert any("favorite_bias" in m and "31.0h" in m for m, _level in notifier.sent)


def test_strategy_silent_alert_hours_default_is_over_a_full_trading_day() -> None:
    """The default must exceed 24h: a whole quiet weekday is ordinary for
    favorite_bias, so anything <= 24h false-positives by construction."""
    from python.settings import TelegramConfig

    assert TelegramConfig().strategy_silent_alert_hours > 24.0


def test_calibration_monitor_paths_follow_state_dir(tmp_path):
    """Configured paths keep only their basename, placed under ``state_dir``."""
    from python.orchestrator import Orchestrator

    state_dir = tmp_path / "state"
    settings = Settings(
        state_dir=str(state_dir),
        calibration_monitor={
            "enabled": True,
            "state_path": "elsewhere/cm.json",
            "prediction_log_path": "elsewhere/preds.jsonl",
            "eval_audit_path": "elsewhere/eval.jsonl",
        },
        strategies={"weather": {"edge_threshold": 0.07}},
        _env_file=None,
    )
    orch = Orchestrator(settings=settings)
    orch._build_analytics_services()

    assert orch._prediction_log is not None
    assert orch._prediction_log._path == state_dir / "preds.jsonl"
    assert orch._calibration_monitor is not None
    assert orch._calibration_monitor._state_path == state_dir / "cm.json"
    assert orch._calibration_monitor._corpus_stream_path == state_dir / "eval.jsonl"
