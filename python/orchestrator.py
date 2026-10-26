"""
Main orchestrator for The Oracle.

Ties together: config -> Kalshi client -> strategies -> risk manager ->
execution engine in an async event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import signal
import socket
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx

from python.analytics.attribution import AttributionEngine
from python.analytics.audit_tripwires import (
    PLATEAU_ALERT_CODE_PREFIX,
    TripwireConfig,
    check_calibrator_plateau_coverage,
    load_acks,
    persist_seen_code,
)
from python.analytics.audit_tripwires import (
    run_all as run_all_tripwires,
)
from python.analytics.calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
    DriftAlert,
)
from python.analytics.prediction_log import EvalAuditRecordSource, PredictionLog
from python.analytics.strategy_attribution import build_market_strategy_index
from python.common import ET, OrderStatus, Side
from python.data.bias_correction import BiasTable
from python.data.contract_watcher import ContractWatcher
from python.data.emos import EmosTable, EmosTableHolder
from python.data.kalshi_client import KalshiClient, KalshiWebSocket
from python.data.kalshi_schemas import SCHEMA_BALANCE, SCHEMA_POSITION_ROW, KalshiSchema
from python.data.kxrain_observer import KxRainObserver
from python.data.kxrain_settlements import KxRainSettlementSweep, SettlementSweepResult
from python.data.phase0_depth_monitor import Phase0DepthMonitor
from python.data.weather import WeatherClient
from python.execution.engine import ExecutionEngine, LivePosition, OrderRecord
from python.execution.fills_reconciler import FillsReconciler
from python.execution.shadow_broker import ShadowBroker
from python.notifications.base import AlertLevel
from python.notifications.formatters import (
    format_calibration_drift_alert,
    format_settlement_alert,
)
from python.risk.manager import (
    PERSISTENCE_BROKEN_ALERT_PREFIX,
    SAVE_FAILURE_ALERT_THRESHOLD,
    STRATEGY_ERROR_ALERT_THRESHOLD,
    RiskManager,
)
from python.risk.persistence import (
    STATE_FILENAME,
    load_state,
    save_state,
    serialize_risk_state,
    write_heartbeat,
)
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal
from python.strategies.intraday_asos_high import IntradayAsosHighStrategy
from python.strategies.intraday_emos_high import IntradayEmosHighStrategy
from python.strategies.per_day_event_cap import (
    _event_key_from_signal,  # (city, market_type, forecast_date) — deference hook
)
from python.strategies.weather_edge import WeatherEdgeStrategy
from python.utils.jsonl import iter_jsonl
from python.utils.log_config import bind_contextvars, setup_logging, unbind_contextvars

try:
    from python.notifications.telegram_bot import TelegramNotifier

    _HAS_TELEGRAM = True
except ImportError:
    TelegramNotifier = None  # type: ignore[assignment,misc]
    _HAS_TELEGRAM = False

logger = logging.getLogger(__name__)


# WS pipe-health window (seconds) — clamped so a single slow reconnect can't
# false-fire pipe-down, and so it always fires before the data-silence cutoff.
_RECONNECT_MAX_BACKOFF_S: float = 120.0  # matches kalshi_client listen() max_delay
_PIPE_HEALTH_MARGIN_S: float = 15.0

# An ack recorded at one severity never silences a more severe firing of the same code.
_SEV_RANK: dict[str, int] = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}


def _pipe_health_window(idle_timeout_s: float, stale_cutoff_s: float) -> float:
    """Clamped window past which a non-re-subscribing pipe is 'down'.

    Floor: one worst-case reconnect cycle (idle + max backoff + margin).
    Ceiling: stale_cutoff - margin (so pipe-down can fire before data-silence).
    """
    base = 2.0 * idle_timeout_s
    low = idle_timeout_s + _RECONNECT_MAX_BACKOFF_S + _PIPE_HEALTH_MARGIN_S
    high = stale_cutoff_s - _PIPE_HEALTH_MARGIN_S
    return max(low, min(base, high))


def validate_strategy_allocations(
    allocations: dict[str, float],
    strategies: list[tuple[str, bool]],
) -> None:
    """Fail loud on a misconfigured ``risk.strategy_allocations`` map.

    ``strategies`` is a list of ``(runtime_name, enabled)`` pairs. Money-sizing
    knob → fail closed: an unknown key (typo) must NOT silently fall back to
    global bankroll, and an enabled strategy with no entry must NOT silently
    size off full equity while its peers are booked.

    No-op when ``allocations`` is empty (feature off).
    """
    if not allocations:
        return
    known = {name for name, _ in strategies}
    unknown = set(allocations) - known
    if unknown:
        raise ValueError(
            f"risk.strategy_allocations contains unknown strategy "
            f"name(s) {sorted(unknown)}; known: {sorted(known)}. "
            f"Config-side keys (e.g. 'weather') are NOT runtime names "
            f"(e.g. 'weather_edge')."
        )
    missing = [name for name, enabled in strategies if enabled and name not in allocations]
    if missing:
        raise ValueError(
            f"risk.strategy_allocations is non-empty but enabled strategies "
            f"{sorted(missing)} are missing an allocation entry. Add an entry "
            f"(use 0.0 to park a strategy)."
        )


def configured_strategy_flags(settings: Settings) -> list[tuple[str, bool]]:
    """Runtime ``(name, enabled)`` pairs for the strategies startup will register.

    Mirrors the ``enabled`` gates in ``Orchestrator.setup`` so a config can be
    checked with ``validate_strategy_allocations`` before a restart, without a
    Kalshi client.
    """
    s = settings.strategies
    flags = (
        ("weather_edge", s.weather.enabled),
        ("intraday_emos_high", s.intraday_emos_high.enabled),
        ("intraday_asos_high", s.intraday_asos_high.enabled),
        ("favorite_bias", s.favorite_bias.enabled),
    )
    return [(name, True) for name, enabled in flags if enabled]


def build_position_strategy_index(trade_log_path: Path) -> dict[str, str]:
    """Reconstruct ``market_id -> strategy`` from the durable trade log.

    Returns an empty dict if the log is absent (fresh deploy). The
    reconstruction is the only post-restart source of pod attribution for
    strategy-anonymous Kalshi positions.
    """
    if not trade_log_path.exists():
        return {}
    return build_market_strategy_index(iter_jsonl(trade_log_path))


_sd_sock: socket.socket | None = None
_sd_addr: str | None = None  # None = not yet resolved; "" = unavailable


def _sd_notify(state: str) -> None:
    """Send a notification to systemd via the sd_notify protocol.

    Pure stdlib implementation using AF_UNIX. Caches the socket across calls.
    No-op when NOTIFY_SOCKET is not set (local dev, non-systemd environments).
    """
    global _sd_sock, _sd_addr  # noqa: PLW0603
    if _sd_addr is None:
        raw = os.environ.get("NOTIFY_SOCKET", "")
        if not raw:
            _sd_addr = ""
            return
        # Abstract socket (starts with @) → null-byte prefix
        _sd_addr = ("\0" + raw[1:]) if raw.startswith("@") else raw
        _sd_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    if not _sd_addr:
        return
    try:
        assert _sd_sock is not None
        _sd_sock.sendto(state.encode(), _sd_addr)
    except Exception:
        logger.warning("sd_notify(%s) failed", state, exc_info=True)


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    """Cancel an asyncio task and wait for it to finish. No-op if None or already done."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# OrderStatuses that represent "no real placement happened" — these records
# carry no Kalshi order_id and consume no per-day-cap slot. Used by the
# orchestrator's post-execute_batch commit loop to skip them. Partition
# contract: this set + the `_EXECUTED_STATUSES` set on the mark_traded loop
# together cover every OrderStatus the engine emits from
# execute_batch — if a new status is added, it MUST be placed deliberately
# in one or the other; an uncovered status would either leak a cap slot
# (if missing from EXCLUDED) or get incorrectly committed.
_PER_DAY_CAP_EXCLUDED_STATUSES = frozenset(
    {
        OrderStatus.REJECTED,
        OrderStatus.SKIPPED,
        OrderStatus.ZERO_SIZE,
        OrderStatus.ERROR,
    }
)

# OrderStatuses that represent a real placement that should be marked as
# "traded" in the originating strategy's mark_traded loop. This is the
# executed half of the partition described above; see the
# `_PER_DAY_CAP_EXCLUDED_STATUSES` contract.
_EXECUTED_STATUSES = frozenset(
    {
        OrderStatus.SHADOW,
        OrderStatus.FILLED,
        OrderStatus.PARTIAL,
        OrderStatus.PENDING,
        OrderStatus.RESTING,
    }
)


def _acquire_trade_log_pidlock(pidlock_path: Path) -> int:
    """Acquire the sidecar pidlock at state/trades.jsonl.pidlock.

    Uses fcntl.flock(LOCK_EX | LOCK_NB). The kernel releases the lock
    on process exit (graceful or crash), so a stale lockfile on disk
    does not block a new process. The lockfile content is unused for
    liveness detection — flock's kernel semantics are the source of truth.

    Returns the open file descriptor on success. Caller must keep the
    FD open for the duration of the orchestrator process; closing it
    or letting it be garbage-collected releases the lock.

    Raises BlockingIOError if another process holds the lock.
    """
    pidlock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pidlock_path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise
    # Optional: write the PID for human-readable debugging (NOT used for liveness)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def apply_favorite_bias_deference(
    all_signals: list[TradeSignal],
    open_position_event_keys: set[tuple[str, str, str]],
) -> list[TradeSignal]:
    """Drop favorite_bias signals that share a (city, var, forecast_date) event
    with (a) a weather_edge same-tick signal OR (b) any open position.
    Pure; safe to call on the flattened signal batch.

    Uses per_day_event_cap._event_key_from_signal for keying parity with the
    per-day cap (var component is ``market_type``). open_position_event_keys is
    supplied by the caller (derived from open-position tickers — RiskManager
    positions are market_id->dollars with no metadata).
    """
    _empty_key: tuple[str, str, str] = ("", "", "")
    occupied = {
        k
        for s in all_signals
        if s.strategy_name == "weather_edge"
        for k in (_event_key_from_signal(s),)
        if k != _empty_key
    } | {k for k in open_position_event_keys if k != _empty_key}
    kept = []
    for s in all_signals:
        if s.strategy_name == "favorite_bias" and _event_key_from_signal(s) in occupied:
            continue
        kept.append(s)
    return kept


def apply_intraday_shadow_drop(
    all_signals: list[TradeSignal],
    shadow_only: bool,
) -> list[TradeSignal]:
    """Drop intraday_emos_high signals when the strategy is in shadow-only mode.

    Pure; safe to call on the flattened signal batch. Logs a structured line per
    drop so "intraday_shadow_drop" greps identify all suppressed signals.
    """
    if not shadow_only:
        return all_signals
    kept = []
    for s in all_signals:
        if s.strategy_name == "intraday_emos_high":
            logger.info(
                "intraday_shadow_drop ticker=%s contracts=%d edge=%.4f",
                s.market_ticker,
                s.suggested_contracts,
                s.effective_edge,
            )
            continue
        kept.append(s)
    return kept


def apply_intraday_asos_deference(
    all_signals: list[TradeSignal],
    open_position_event_keys: set[tuple[str, str, str]],
) -> list[TradeSignal]:
    """Drop intraday_asos_high signals that share a (city, var, forecast_date) event
    with (a) a weather_edge same-tick signal OR (b) any open position.
    Pure; safe to call on the flattened signal batch. Mirrors
    ``apply_favorite_bias_deference``.

    Uses per_day_event_cap._event_key_from_signal for keying parity with the
    per-day cap (var component is ``market_type``). open_position_event_keys is
    supplied by the caller (derived from open-position tickers — RiskManager
    positions are market_id->dollars with no metadata).
    """
    _empty_key: tuple[str, str, str] = ("", "", "")
    occupied = {
        k
        for s in all_signals
        if s.strategy_name == "weather_edge"
        for k in (_event_key_from_signal(s),)
        if k != _empty_key
    } | {k for k in open_position_event_keys if k != _empty_key}
    kept = []
    for s in all_signals:
        if s.strategy_name == "intraday_asos_high" and _event_key_from_signal(s) in occupied:
            continue
        kept.append(s)
    return kept


def apply_intraday_asos_shadow_drop(
    all_signals: list[TradeSignal],
    shadow_only: bool,
) -> list[TradeSignal]:
    """Drop intraday_asos_high signals when the strategy is in shadow-only mode.

    Pure; safe to call on the flattened signal batch. Logs a structured line per
    drop so "intraday_asos_shadow_drop" greps identify all suppressed signals.
    """
    if not shadow_only:
        return all_signals
    kept = []
    for s in all_signals:
        if s.strategy_name == "intraday_asos_high":
            logger.info(
                "intraday_asos_shadow_drop ticker=%s contracts=%d edge=%.4f",
                s.market_ticker,
                s.suggested_contracts,
                s.effective_edge,
            )
            continue
        kept.append(s)
    return kept


def apply_weather_shadow_drop(
    all_signals: list[TradeSignal],
    shadow_only: bool,
) -> list[TradeSignal]:
    """Drop weather_edge signals when the strategy is in shadow-only mode.

    Pure; safe to call on the flattened signal batch. Logs a structured line per
    drop so "weather_shadow_drop" greps identify all suppressed signals.

    Shadow-only keeps weather_edge evaluating + logging the calibration corpus
    while placing zero orders. Wired BEFORE the favorite_bias / intraday-asos
    deference hooks so they do not defer to a weather_edge that will not trade.
    """
    if not shadow_only:
        return all_signals
    kept = []
    for s in all_signals:
        if s.strategy_name == "weather_edge":
            logger.info(
                "weather_shadow_drop ticker=%s contracts=%d edge=%.4f",
                s.market_ticker,
                s.suggested_contracts,
                s.effective_edge,
            )
            continue
        kept.append(s)
    return kept


class Orchestrator:
    """
    Central event loop that coordinates all subsystems.

    Lifecycle:
        1. Load config / settings
        2. Initialize Kalshi client, risk manager, execution engine
        3. Register strategies
        4. Main loop: poll markets -> run strategies -> risk check -> execute
        5. Graceful shutdown on SIGINT/SIGTERM
    """

    _WS_MAX_RESTARTS = 5
    _WS_ALERT_THRESHOLD: int = 3
    _WS_SILENT_INFO_LOG_INTERVAL_S: float = 900.0  # throttle for silent-but-healthy INFO log
    _TICK_FAILURE_ALERT_THRESHOLD = 10
    _TICK_FAILURE_KILL_THRESHOLD = 20

    @staticmethod
    def _et_today() -> date:
        return datetime.now(ET).date()

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.load()
        self._strategies: list[Strategy] = []
        self._kalshi_client: KalshiClient | None = None
        self._kalshi_ws: KalshiWebSocket | None = None
        self._weather_client: WeatherClient | None = None
        self._risk_manager: RiskManager | None = None
        self._engine: ExecutionEngine | None = None
        self._running = False
        self._stop_event = asyncio.Event()
        self._ws_task: asyncio.Task[None] | None = None
        self._ws_restart_count: int = 0
        self._current_trading_date: date | None = None
        self._last_broadcast_bankroll: float = float("nan")  # force first broadcast
        self._save_task: asyncio.Task[None] | None = None
        self._save_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._consecutive_tick_failures: int = 0
        self._notifier: TelegramNotifier | None = None
        self._telegram_task: asyncio.Task[None] | None = None
        self._tripwire_task: asyncio.Task[None] | None = None
        # Per-code dedup so the hourly tripwire sweep doesn't spam Telegram
        # with the same alert every hour. Both calibrator codes embed
        # fit_date, so a genuine refit produces a new code and re-alerts
        # naturally — even if the refit was botched and the artifact is
        # still stale.
        self._tripwire_alerted_codes: set[str] = set()
        # Codes that have already produced the `logger.warning("Audit tripwire: ...")`
        # line, added UNCONDITIONALLY (unlike `_tripwire_alerted_codes` above, which
        # only gains a code once it's actually dispatched). Without this separate
        # set, a suppressed code (owner not live, or acked) never enters
        # `_tripwire_alerted_codes` and so re-logs that WARNING on every hourly
        # sweep forever — diluting the WARNING channel real risk alerts live on.
        self._tripwire_logged_codes: set[str] = set()
        # (code, owner) -> count of alerts suppressed as not-live. Surfaced by
        # /acks so suppression is never silent.
        self._alert_suppressions: Counter[tuple[str, str]] = Counter()
        # Human-recorded verdicts, keyed by alert code. Written ONLY by /ack.
        self._acked: dict[str, dict[str, Any]] = {}
        # Codes seen this process whose alert declared an ack_key, with the owner
        # and severity to record if the operator acks them: code -> (owner, severity).
        self._ackable: dict[str, tuple[str | None, str]] = {}
        self._hc_client: httpx.AsyncClient | None = None

        self._contract_watcher: ContractWatcher | None = None

        self._was_halted: bool = False
        self._last_signal_time: float = time.monotonic()
        self._startup_monotonic: float = time.monotonic()
        self._silent_alert_sent: bool = False
        # Per-strategy liveness: unlike _last_signal_time above (which
        # any strategy's signals refresh, pre-shadow-drop), this tracks each
        # strategy individually so a dead order-placing strategy cannot hide
        # behind a shadow-only strategy that is still evaluating fine.
        self._last_strategy_signal: dict[str, float] = {}
        self._strategy_silent_alerted: set[str] = set()
        # Edge-triggered: a sustained outage fires the CRITICAL alert
        # once and re-arms only after a successful save.
        self._save_failure_alert_sent: bool = False
        self._ws_last_alerted_count: int = 0
        # Time-based throttles (monotonic); 0.0 lets the first alert/log fire immediately.
        self._ws_last_critical_alert_monotonic: float = 0.0
        self._ws_last_silent_info_monotonic: float = 0.0
        self._tick_count: int = 0
        self._last_reconcile_tick: int = 0

        # Shared lock owned by orchestrator and passed to BOTH the
        # ExecutionEngine and the FillsReconciler so the dedup invariant
        # (one fill row per order_id) holds across writers.
        self._trade_log_lock: asyncio.Lock = asyncio.Lock()
        self._fills_reconciler: FillsReconciler | None = None
        self._last_scheduled_fills_tick: int = 0

        # Sidecar pidlock FD — held open for the lifetime of the process.
        # Set to None until initialize() acquires the lock.
        self._trade_log_pidlock_fd: int | None = None

        # Analytics services: instantiated in initialize().
        self._prediction_log: PredictionLog | None = None
        self._attribution_engine: AttributionEngine | None = None
        self._calibration_monitor: CalibrationMonitor | None = None

        # Phase 0 passive depth monitor for the intraday EMOS HIGH strategy.
        # Auto-enabled whenever a Kalshi client is connected; runs
        # independently of any strategy's ``enabled`` flag.
        self._phase0_depth_monitor: Phase0DepthMonitor | None = None
        self._phase0_depth_task: asyncio.Task[None] | None = None

        # KXRAIN observer: logging only, never a registered strategy, so it can
        # never receive a risk.strategy_allocations entry.
        self._kxrain_observer: KxRainObserver | None = None
        self._kxrain_task: asyncio.Task[None] | None = None
        self._kxrain_sweep: KxRainSettlementSweep | None = None
        self._kxrain_sweep_task: asyncio.Task[None] | None = None

        # Shared EmosTable holder. The
        # orchestrator owns hot-reload: it constructs the holder once before
        # any strategy is registered, injects the same instance into every
        # EMOS-consuming strategy, and rebinds ``holder.table`` once per
        # tick before invoking ``generate_signals``. Strategies MUST NOT
        # call ``EmosTable.check_and_reload`` themselves — they read
        # ``self._emos_holder.table`` on every access so any rebind is
        # visible. ``None`` when no weather strategy is enabled (the
        # holder is constructed lazily inside the weather-strategy block).
        self._emos_holder: EmosTableHolder | None = None

    def register_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to the orchestrator."""
        self._strategies.append(strategy)
        # Seed liveness now: a strategy that has never produced a signal must be
        # measured from startup, not skipped for want of a key.
        self._last_strategy_signal[strategy.name] = time.monotonic()
        logger.info("Registered strategy: %s (enabled=%s)", strategy.name, strategy.enabled)

    async def _check_strategy_liveness(self, *, in_market_hours: bool) -> None:
        """Alert when a strategy that actually trades has gone silent.

        The aggregate SILENT BOT watchdog cannot cover this: it refreshes
        ``_last_signal_time`` from ANY strategy's signals, deliberately
        pre-shadow-drop, so a shadow-only strategy (e.g. weather_edge) keeps
        it quiet while the only order-placing strategy is dead. Uses
        ``configured_to_trade``, so shadow-only strategies are excluded and a
        future strategy is covered automatically on registration.

        Thresholded on ``telegram.strategy_silent_alert_hours``, NOT on the
        aggregate watchdog's ``silent_bot_alert_hours``: one strategy can
        legitimately be quiet for a whole trading day (favorite_bias is lead-1,
        band 0.50-0.75, one bet per event per day), so the two need to be tuned
        independently or this alert false-positives itself into being ignored.

        Never raises — this runs every tick, ahead of ``_persist_if_dirty``,
        the healthchecks ping, and the ``_consecutive_tick_failures`` reset
        (see ``_tick``); an unhandled exception here would abort the rest of
        the tick and, sustained, has a structural path to the kill switch via
        ``_TICK_FAILURE_KILL_THRESHOLD``. A monitoring bug must not halt
        trading. Mirrors ``_audit_tripwire_loop``'s "never raises" contract.
        """
        if not in_market_hours:
            return
        try:
            threshold = self._settings.telegram.strategy_silent_alert_hours
            now = time.monotonic()
            for s in self._strategies:
                if not s.configured_to_trade:
                    self._strategy_silent_alerted.discard(s.name)
                    continue
                last = self._last_strategy_signal.get(s.name)
                if last is None:
                    continue
                hours = (now - last) / 3600.0
                if hours >= threshold and s.name not in self._strategy_silent_alerted:
                    await self._notify_alert(
                        f"STRATEGY SILENT: {s.name} has produced no signals for "
                        f"{hours:.1f}h while configured to trade. Check the g(mid) "
                        f"artifact, market availability, and the band/lead filters.",
                        level=AlertLevel.WARNING,
                    )
                    self._strategy_silent_alerted.add(s.name)
                elif hours < threshold:
                    self._strategy_silent_alerted.discard(s.name)
        except Exception:
            logger.exception("Per-strategy liveness check failed")

    def _validate_strategy_allocations(self) -> None:
        # Without Kalshi credentials no strategy is built, so every key would read as a typo.
        if self._kalshi_client is None:
            return
        validate_strategy_allocations(
            self._settings.risk.strategy_allocations,
            [(s.name, s.enabled) for s in self._strategies],
        )

    def enable_strategy(self, name: str) -> str:
        """Re-enable a strategy on both the Strategy object and risk tracker.

        Returns a human-readable status message.
        """
        assert self._risk_manager is not None
        found = None
        for s in self._strategies:
            if s.name == name:
                found = s
                break
        if found is None:
            known = [s.name for s in self._strategies]
            return f"Unknown strategy '{name}'. Available: {', '.join(known)}"

        found.enabled = True
        self._risk_manager.enable_strategy(name)
        return f"Strategy '{name}' re-enabled."

    def _activate_kill_switch(self) -> None:
        """Create STOP file and halt immediately. Used by Telegram /kill."""
        stop_file = self._settings.stop_file
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.touch()
        self._request_stop()
        logger.critical("Kill switch activated via Telegram")

    def _deactivate_kill_switch(self) -> None:
        """Remove STOP file. Bot must be restarted to resume."""
        try:
            self._settings.stop_file.unlink(missing_ok=True)
            logger.info("Kill switch deactivated via Telegram")
        except OSError:
            logger.exception("Failed to remove STOP file")

    async def initialize(self) -> None:
        """
        Initialize all components.

        Call this before ``run()``.  In shadow mode, the Kalshi client
        is still created (to fetch market data) but orders are never sent.
        """
        s = self._settings

        # Defence-in-depth against accidental dual-orchestrator runs.
        # The FD is stored on self so the kernel keeps the lock until process exit.
        state_dir = Path(s.state_dir)
        pidlock_path = state_dir / "trades.jsonl.pidlock"
        try:
            self._trade_log_pidlock_fd = _acquire_trade_log_pidlock(pidlock_path)
        except BlockingIOError:
            logger.critical(
                "Another orchestrator already holds %s; refusing to start.",
                pidlock_path,
            )
            raise SystemExit(1)

        self._contract_watcher = ContractWatcher()

        if s.kalshi_api_key and s.kalshi_private_key_path:
            # Empty passphrase → None (unencrypted PEM); set only when the
            # operator opts into an encrypted PEM.
            pem_password = s.kalshi_private_key_password.encode() or None
            self._kalshi_client = KalshiClient(
                api_key=s.kalshi_api_key,
                private_key_path=s.kalshi_private_key_path,
                base_url=s.active_base_url,
                contract_watcher=self._contract_watcher,
                private_key_password=pem_password,
            )
            await self._kalshi_client.connect()
            logger.info("Kalshi client connected (%s)", s.active_base_url)

            ws_url = s.active_ws_url
            self._kalshi_ws = KalshiWebSocket(
                api_key=s.kalshi_api_key,
                private_key_path=s.kalshi_private_key_path,
                ws_url=ws_url,
                reconnect_delay=s.execution.websocket_reconnect_delay,
                idle_timeout=s.execution.websocket_idle_timeout_seconds,
                private_key_password=pem_password,
            )
        else:
            logger.warning(
                "No Kalshi credentials configured — running without market data or strategies. "
                "Set ORACLE_KALSHI_API_KEY and ORACLE_KALSHI_PRIVATE_KEY_PATH in .env."
            )

        # Weather data client (free API, always available). The client-level
        # forecast-concurrency cap is the authoritative limiter on outbound
        # Open-Meteo requests (see WeatherClient + weather.forecast_concurrency).
        self._weather_client = WeatherClient(
            forecast_concurrency=s.strategies.weather.forecast_concurrency,
        )
        await self._weather_client.connect()
        logger.info("Weather client connected")

        self._risk_manager = RiskManager(self._settings)

        # Execution engine — pass the shared trade_log_lock so the engine and
        # the FillsReconciler use the same Lock instance.
        self._engine = ExecutionEngine(
            settings=self._settings,
            kalshi_client=self._kalshi_client,
            risk_manager=self._risk_manager,
            contract_watcher=self._contract_watcher,
            trade_log_lock=self._trade_log_lock,
            notify_warning=self._notify_warning_telegram,
            notify_critical=self._notify_critical_telegram,
        )

        await self._reconcile_from_kalshi()
        self._restore_risk_state()

        # Must be built before strategies register.
        self._build_analytics_services()

        # Built after engine, before the run loop.
        self._setup_fills_reconciler()

        # Independent of any strategy's enabled flag — depth probing must
        # accumulate while the intraday strategy ships disabled.
        self._setup_phase0_depth_monitor()

        self._setup_kxrain_observer()

        # Constructed BEFORE strategy registration so strategies that accept a
        # ``notifier`` kwarg (e.g. WeatherEdgeStrategy → EMOS load ERROR alert)
        # see a real notifier from their first ``setup()`` call. The notifier
        # holds a reference to ``self._strategies``, so strategies registered
        # below remain visible to ``/status``, ``/positions``, etc.
        if (
            _HAS_TELEGRAM
            and s.telegram.enabled
            and s.telegram_bot_token
            and s.telegram_owner_chat_id
        ):
            self._notifier = TelegramNotifier(
                settings=s,
                risk_manager=self._risk_manager,
                execution_engine=self._engine,
                strategies=self._strategies,
                kill_callback=self._activate_kill_switch,
                unkill_callback=self._deactivate_kill_switch,
                enable_strategy_callback=self.enable_strategy,
                attribution_engine=self._attribution_engine,
                calibration_monitor=self._calibration_monitor,
                fills_reconciler=self._fills_reconciler,
                kalshi_client=self._kalshi_client,
                orchestrator=self,
            )
            logger.info("Telegram notifier configured")
        elif s.telegram.enabled and not _HAS_TELEGRAM:
            logger.warning(
                "Telegram enabled in config but python-telegram-bot not installed "
                "(install with: uv sync --extra telegram)"
            )

        if self._kalshi_client is not None and s.strategies.weather.enabled:
            # Bias correction table (optional, disabled by default). Shared
            # contract with the intraday strategy — both consume the same
            # read-only BiasTable artifact when bias correction is enabled
            # on the weather config.
            bias_table: BiasTable | None = self._build_bias_table_for_strategy()

            # Construct the shared EmosTable holder once, BEFORE the strategy.
            # The strategy reads ``self._emos_holder.table`` on every access;
            # the orchestrator rebinds it via the per-tick hot-reload call
            # (see ``_tick`` below).
            self._emos_holder = await self._build_emos_holder()

            weather_strategy = WeatherEdgeStrategy(
                config=s.strategies.weather.model_dump(),
                kalshi_client=self._kalshi_client,
                weather_client=self._weather_client,
                bankroll=s.bankroll,
                market_filter=s.market_filter.model_dump(),
                settings=s,
                bias_table=bias_table,
                prediction_log=self._prediction_log,
                notifier=self._notifier,
                emos_holder=self._emos_holder,
                # Same-day Kelly haircut. Bound method so each scan reads
                # the live ``_positions`` dict — count is recomputed every
                # candidate, not cached. ``RiskManager.count_same_day_bets``
                # returns 0 for unparseable tickers (non-weather series), so
                # plumbing through this callback can never throttle non-weather
                # strategies that share the same RiskManager.
                same_day_count_provider=self._risk_manager.count_same_day_bets,
            )

            # Restore per-ticker dedup state from shadow log on restart
            shadow_log = Path(s.state_dir) / "trades.jsonl"
            if shadow_log.exists():
                weather_strategy.load_traded_tickers(str(shadow_log))

            self.register_strategy(weather_strategy)

        # Intraday EMOS HIGH strategy. Constructed when the config flag is
        # on AND a Kalshi client is available; gated behind a loaded
        # EmosTable holder (silent identity-fallback would be unsafe per v6
        # evidence). Independent of weather_edge's enabled flag: when
        # weather is disabled but intraday is on, the holder is built here.
        # When BOTH are enabled, the weather block builds the holder first
        # and intraday shares the SAME EmosTableHolder reference so the
        # per-tick hot-reload (see _tick) propagates to both strategies in
        # lockstep.
        if self._kalshi_client is not None and s.strategies.intraday_emos_high.enabled:
            # Ensure the shared EMOS holder exists. The weather block above
            # built it when weather is enabled; otherwise we build it now.
            if self._emos_holder is None:
                self._emos_holder = await self._build_emos_holder()

            intraday_strategy = await self._build_intraday_emos_high_strategy(
                bias_table=self._build_bias_table_for_strategy(),
            )
            if intraday_strategy is not None:
                self.register_strategy(intraday_strategy)
                # Rehydrate per-cell daily cap + per-ticker dedup from the
                # trade log so a process restart mid-ET-day doesn't reset the
                # daily cap counter to 0. Mirrors the weather_strategy
                # hydration above.
                intraday_trade_log = Path(s.state_dir) / "trades.jsonl"
                if intraday_trade_log.exists():
                    intraday_strategy.load_state_from_log(intraday_trade_log)

        # Intraday ASOS HIGH strategy. Constructed when the config flag is on AND
        # a Kalshi client is available. Independent of weather_edge and
        # intraday_emos_high — no shared EMOS holder needed. Gracefully skipped
        # (with WARNING) when the climatology artifact is missing.
        if self._kalshi_client is not None and s.strategies.intraday_asos_high.enabled:
            asos_strategy = await self._build_intraday_asos_high_strategy()
            if asos_strategy is not None:
                self.register_strategy(asos_strategy)
                # Rehydrate per-cell daily cap + per-ticker dedup from the
                # trade log so a process restart mid-ET-day doesn't reset the
                # daily cap counter to 0. Mirrors the intraday_emos_high hydration.
                asos_trade_log = Path(s.state_dir) / "trades.jsonl"
                if asos_trade_log.exists():
                    asos_strategy.load_state_from_log(asos_trade_log)

        # Favorite-bias strategy (model-free; ships disabled by default).
        # Gated on the enabled flag so that with enabled=false (the default)
        # nothing is constructed and live behavior is completely unchanged.
        if self._kalshi_client is not None and s.strategies.favorite_bias.enabled:
            from python.strategies.favorite_bias import FavoriteBiasStrategy

            fb_config = s.strategies.favorite_bias.model_dump()
            fb_config["audit_path"] = str(
                Path(s.state_dir) / Path(s.strategies.favorite_bias.audit_path).name
            )
            fb_strategy = FavoriteBiasStrategy(
                config=fb_config,
                kalshi_client=self._kalshi_client,
                bankroll=s.bankroll,
                trades_log_path=str(Path(s.state_dir) / "trades.jsonl"),
            )
            self.register_strategy(fb_strategy)

        # Fail loud on typos and coverage gaps before any orders are placed.
        self._validate_strategy_allocations()

        Path(s.state_dir).mkdir(parents=True, exist_ok=True)

        # Run any per-strategy async setup (e.g. weather_edge upgrades the
        # SameDayGate sentinel to a live NWS-backed gate when configured on,
        # AND triggers the EMOS-coefficients hot-reload — which routes ERROR
        # alerts via the notifier wired in above).
        await self._run_strategy_lifecycle_hook("setup")

        # Wire the engine's on_cancel callback to every strategy's per-day
        # cap so 0-fill cancellations release the (city,var,date) slot
        # within the same UTC day. Both weather_edge and favorite_bias own
        # their own cap with its own keyspace; fan-out is safe because
        # cap.release(order_id) is idempotent and returns None for unknown
        # order_ids (so calling it on the "wrong" cap is a no-op).
        from python.strategies.per_day_event_cap import PerDayEventCap

        caps_for_cancel: list[PerDayEventCap] = []
        for strat in self._strategies:
            cap = getattr(strat, "_per_day_cap", None)
            if isinstance(cap, PerDayEventCap):
                caps_for_cancel.append(cap)
        if caps_for_cancel:

            def _fanout_release(order_id: str) -> None:
                for cap in caps_for_cancel:
                    cap.release(order_id)

            self._engine.set_on_cancel_callback(_fanout_release)

        if self._contract_watcher is not None:
            self._contract_watcher.alert_fn = self._notify_alert

        if s.healthcheck_ping_url:
            self._hc_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
            logger.info("Healthchecks.io ping configured")

        logger.info(
            "Orchestrator initialized: mode=%s, strategies=%d, bankroll=%.2f",
            s.execution.mode,
            len(self._strategies),
            s.bankroll,
        )

    async def _build_emos_holder(self) -> EmosTableHolder:
        """Load, smoke-test, and publish the orchestrator-owned EMOS holder.

        - ``EmosTable.load`` -> ``smoke_test`` on the freshly loaded artifact.
        - On smoke failure: downgrade to a disabled sentinel via
          ``EmosTable._disabled(path, reason)`` so the strategy's
          identity-fallback path in ``evaluate_market_pure`` still functions
          for non-EMOS cells.
        - When ``emos_shadow_log: true`` AND the table is not loaded (load
          failure OR smoke failure), dispatch an ERROR-level Telegram alert
          with ``failure_bucket="emos_load:_global"`` for dedup.

        Returns the holder regardless of load outcome — a disabled
        EmosTable is still wrapped so downstream consumers receive a
        non-None holder.
        """
        s = self._settings
        emos_coeffs_path = s.strategies.weather.emos_coefficients_path
        emos_table = EmosTable.load(emos_coeffs_path)
        if emos_table.is_loaded:
            try:
                emos_table.smoke_test()
            except Exception as exc:
                logger.warning("EMOS load/smoke failed: %s - identity fallback", exc)
                emos_table = EmosTable._disabled(
                    Path(emos_coeffs_path),
                    f"smoke-error:{type(exc).__name__}:{exc}",
                )

        if emos_table.is_loaded:
            logger.info(
                "EmosTable loaded by orchestrator: artifact_hash=%s n_cells=%d",
                emos_table.artifact_hash,
                len(emos_table.cells),
            )
        else:
            logger.info(
                "EmosTable load disabled at orchestrator: reason=%s — "
                "strategy will run with identity fallback for EMOS cells",
                emos_table.fallback_reason,
            )
            # ``EmosTable.load`` never raises FileNotFoundError — it returns
            # ``_disabled`` with reason ``file-missing``, which still routes
            # through the alert branch below when ``emos_shadow_log`` is on.
            if s.strategies.weather.emos_shadow_log and self._notifier is not None:
                try:
                    await self._notifier.send_alert(
                        f"EMOS load failed: {emos_table.fallback_reason}",
                        level=AlertLevel.ERROR,
                        failure_bucket="emos_load:_global",
                    )
                except Exception:
                    logger.exception("EMOS load alert dispatch failed")

        return EmosTableHolder(table=emos_table)

    def _build_bias_table_for_strategy(self) -> BiasTable | None:
        """Construct the shared bias correction table per ``weather`` config.

        Both ``WeatherEdgeStrategy`` and ``IntradayEmosHighStrategy`` receive
        the same ``BiasTable`` instance (read-only artifact). Failure to load
        is non-fatal: callers receive ``None`` and the downstream strategy
        disables bias correction for that run.
        """
        s = self._settings
        if not s.strategies.weather.bias_correction_enabled:
            return None
        try:
            table_path = s.strategies.weather.bias_table_path
            table = BiasTable(Path(table_path) if table_path else None)
            if not table.loaded:
                logger.warning("Bias correction enabled but table failed to load")
                return None
            logger.info(
                "GFS bias correction enabled (shrinkage=%.1f, table=%s)",
                s.strategies.weather.bias_shrinkage,
                table_path or "data/gfs_bias/bias_correction.json (ERA5 default)",
            )
            # Guard: a bias-table entry for an EMOS-owned cell double-corrects the
            # mean (EMOS already absorbs it via a + b*ens_mean). The NWS table is
            # built to omit these, but a stale regeneration after promoting a new
            # EMOS cell would reintroduce overlap — surface it loudly.
            for cell in s.strategies.weather.emos_enabled_cells:
                city, _, var = cell.partition("/")
                if var and table.has_cell(city, var):
                    logger.warning(
                        "Bias table contains EMOS-owned cell %s — double-correction "
                        "risk (EMOS already corrects its mean). Regenerate the table "
                        "with --exclude-cells %s.",
                        cell,
                        cell,
                    )
            return table
        except Exception:
            logger.warning(
                "Failed to initialize BiasTable — running without correction",
                exc_info=True,
            )
            return None

    async def _build_intraday_emos_high_strategy(
        self,
        *,
        bias_table: BiasTable | None,
    ) -> IntradayEmosHighStrategy | None:
        """Construct the intraday EMOS HIGH strategy when preconditions hold.

        Returns ``None`` (with a WARNING log) when ``self._emos_holder`` is
        unavailable. The intraday strategy refuses to start with an identity
        calibrator (v6 evidence does not extend to identity); constructing
        it with ``emos_holder=None`` would violate the typed contract.

        Note: ``setup()`` failure is caught by ``_run_strategy_lifecycle_hook``;
        the operator-visible refuse-to-trade behavior comes from
        ``_setup_completed`` staying False, not from an automatic Telegram
        CRITICAL. No CRITICAL is wired here.
        """
        s = self._settings
        assert self._kalshi_client is not None  # caller guarantees
        assert self._weather_client is not None  # established in initialize()

        if self._emos_holder is None:
            logger.warning(
                "intraday_emos_high enabled but EmosTableHolder is None — "
                "skipping construction. Strategy refuses to run with identity "
                "calibrator; check earlier EMOS-load alerts for the root cause."
            )
            return None

        # Merge parent weather config so shared knobs (bias_shrinkage,
        # ensemble_*, min_book_depth, plateau filter, etc.) inherit rather
        # than silently default. Intraday-declared keys override weather
        # for the same name (e.g. intraday's audited edge_threshold=0.07).
        merged_config = {
            **s.strategies.weather.model_dump(),
            **s.strategies.intraday_emos_high.model_dump(),
        }
        intraday_strategy = IntradayEmosHighStrategy(
            config=merged_config,
            kalshi_client=self._kalshi_client,
            weather_client=self._weather_client,
            bankroll=s.bankroll,
            market_filter=s.market_filter.model_dump(),
            settings=s,
            emos_holder=self._emos_holder,
            bias_table=bias_table,
            prediction_log=self._prediction_log,
            notifier=self._notifier,
        )
        logger.info(
            "intraday_emos_high constructed: enabled_cells=%s",
            sorted(s.strategies.intraday_emos_high.enabled_cells),
        )
        return intraday_strategy

    async def _build_intraday_asos_high_strategy(
        self,
    ) -> IntradayAsosHighStrategy | None:
        """Construct the intraday ASOS HIGH strategy when preconditions hold.

        Returns ``None`` (with a WARNING log) when the climatology artifact
        cannot be loaded. The orchestrator still starts — the strategy simply
        will not generate signals. Note: ``setup()`` failure is caught by
        ``_run_strategy_lifecycle_hook``; the operator-visible refuse-to-trade
        behavior comes from ``_setup_completed`` staying False.
        """
        from python.data.asos import AsosClient, ClimatologyTable

        s = self._settings
        assert self._kalshi_client is not None  # caller guarantees

        climatology_path = s.strategies.intraday_asos_high.climatology_path
        try:
            climatology = ClimatologyTable.load(climatology_path)
        except Exception:
            logger.warning(
                "intraday_asos_high: failed to load climatology artifact from %s "
                "— skipping construction. Run 'scripts/build_asos_climatology.py' "
                "to rebuild the artifact.",
                climatology_path,
                exc_info=True,
            )
            return None

        asos_client = AsosClient(
            cache_ttl_s=s.strategies.intraday_asos_high.asos_obs_cache_ttl_seconds
        )

        asos_strategy = IntradayAsosHighStrategy(
            config=s.strategies.intraday_asos_high.model_dump(),
            kalshi_client=self._kalshi_client,
            bankroll=s.bankroll,
            market_filter=s.market_filter.model_dump(),
            settings=s,
            asos_client=asos_client,
            climatology=climatology,
            prediction_log=self._prediction_log,
            notifier=self._notifier,
        )
        logger.info(
            "intraday_asos_high constructed: enabled_cells=%s informational_cells=%s",
            sorted(s.strategies.intraday_asos_high.enabled_cells),
            sorted(s.strategies.intraday_asos_high.informational_cells),
        )
        return asos_strategy

    def _reset_daily_with_fanout(self) -> None:
        """Reset the risk manager's daily counters and fan out to strategies.

        The intraday strategy maintains a per-cell daily trade counter that
        MUST be cleared at every daily-reset boundary. Defense-in-depth
        lives inside the strategy (self-clean on et_today() rollover) but
        the explicit fan-out call is mandatory. Dispatch is defensive: any
        strategy exposing a callable ``reset_daily_counters`` attribute is
        invoked, so future strategies with the same hook get the same call.

        Called from every site that previously called ``risk.reset_daily()``
        directly: the runtime tick boundary (``_tick`` day-rollover) and the
        startup ``_restore_risk_state`` path (mode-switch + new-day branches).
        """
        assert self._risk_manager is not None
        self._risk_manager.reset_daily()
        for strategy in self._strategies:
            hook = getattr(strategy, "reset_daily_counters", None)
            if hook is None or not callable(hook):
                continue
            try:
                hook()
            except Exception:
                logger.exception(
                    "Strategy %s reset_daily_counters() failed",
                    strategy.name,
                )

    def _build_analytics_services(self) -> None:
        """Instantiate prediction log, attribution engine, calibration monitor."""
        s = self._settings
        state_dir = Path(s.state_dir)
        cm = s.calibration_monitor

        # Prediction log is cheap to construct — just wires a file path.
        # Strategies that don't want it can accept None; weather does write to it.
        try:
            self._prediction_log = PredictionLog(state_dir / Path(cm.prediction_log_path).name)
        except Exception:
            logger.exception("Failed to initialize PredictionLog")
            self._prediction_log = None

        if s.attribution.enabled:
            self._attribution_engine = AttributionEngine(s.attribution.settlements_path)
            logger.info(
                "Attribution engine configured (settlements=%s)",
                s.attribution.settlements_path,
            )

        # Load-time check on the live isotonic artifact — flags pathological
        # plateaus (see audit_tripwires.check_calibrator_plateau_coverage).
        # No-op for non-isotonic calibrators. Reads the configured artifact
        # (``weather.isotonic_path``) so the bounds reflect the calibrator
        # the strategy actually loaded, not the canonical fallback file.
        try:
            iso_path = self._isotonic_artifact_path()
            plateau_alerts = check_calibrator_plateau_coverage(
                iso_path.parent,
                isotonic_filename=iso_path.name,
            )
        except Exception:
            logger.exception("Calibrator plateau tripwire check failed")
            plateau_alerts = []
        for alert in plateau_alerts:
            logger.warning(
                "Calibrator plateau tripwire: %s (code=%s)",
                alert.message,
                alert.code,
            )
        # Seed the hourly-loop dedup set and the human-ack map from the
        # persisted alert_acks.json (v2; a v1-only tripwire_plateau_seen.json
        # migrates into `seen`, never `acked` — see load_acks). Codes in
        # `seen` have already reached Telegram on a prior run; seeding them
        # here suppresses restart noise. New codes (different fit_date or
        # bounds) won't be in the file and will reach Telegram once via the
        # first loop tick, then get written to the file to suppress future
        # restarts.
        seen, acked = load_acks(Path(self._settings.state_dir))
        self._tripwire_alerted_codes |= seen
        self._acked = acked

        if s.calibration_monitor.enabled and self._prediction_log is not None:
            cfg = CalibrationMonitorConfig(
                **s.calibration_monitor.model_dump(
                    exclude={
                        "state_path",
                        "snapshots_dir",
                        "prediction_log_path",
                        # kalshi_settlements_path is a path, not a tunable —
                        # pass it through the constructor kwarg, not the config.
                        "kalshi_settlements_path",
                        # Corpus wiring is resolved here, not a monitor-dataclass tunable.
                        "corpus_source",
                        "eval_audit_path",
                    }
                )
            )
            # Corpus source: ``eval_audit`` (default) reads the Stage-1 eval
            # stream — written for every evaluated market before depth/Kelly/
            # allocation — so a $0 pod can't dark the corpus (2026-06-16
            # incident). ``predictions`` is the legacy signal-gated stream,
            # kept as a config rollback. The strategies keep WRITING
            # predictions.jsonl either way; only the monitor's read side moves.
            if s.calibration_monitor.corpus_source == "eval_audit":
                corpus_source = EvalAuditRecordSource(state_dir / Path(cm.eval_audit_path).name)
                corpus_stream_path: Path | None = corpus_source.active_path
                monitor_log: Any = corpus_source
            else:
                monitor_log = self._prediction_log
                corpus_stream_path = None  # ambiguous silence — dark-check off
            self._calibration_monitor = CalibrationMonitor(
                prediction_log=monitor_log,
                snapshots_dir=Path(s.calibration_monitor.snapshots_dir),
                state_path=state_dir / Path(cm.state_path).name,
                config=cfg,
                alert_callback=self._on_calibration_drift,
                isotonic_fit_date=self._read_isotonic_fit_date(),
                kalshi_settlements_path=Path(s.calibration_monitor.kalshi_settlements_path),
                corpus_stream_path=corpus_stream_path,
            )
            logger.info(
                "Calibration monitor configured (tick=%.0fs, warmup=%d, corpus=%s)",
                cfg.tick_interval_seconds,
                cfg.warmup_pairs,
                s.calibration_monitor.corpus_source,
            )

    def _setup_fills_reconciler(self) -> None:
        """Construct the FillsReconciler when enabled and execution mode
        is live or paper.

        Shadow mode is excluded: there are no real Kalshi fills to
        reconcile, so the reconciler would fetch an empty endpoint
        forever. The kalshi client must be present — without it there's
        no way to query /portfolio/fills.

        Idempotent: safe to call multiple times; later calls overwrite
        the previous reconciler. The default is to build at most one
        per orchestrator lifetime.
        """
        fr_cfg = self._settings.fills_reconciler
        exec_mode = self._settings.execution.mode

        if not fr_cfg.enabled:
            self._fills_reconciler = None
            return
        if exec_mode not in {"live", "paper"}:
            self._fills_reconciler = None
            return
        if self._kalshi_client is None or self._engine is None:
            logger.warning("FillsReconciler enabled but kalshi_client/engine missing; skipping")
            self._fills_reconciler = None
            return

        state_dir = Path(self._settings.state_dir)
        # Honour the configured state_file's *basename* under our state_dir
        # so tests that pass a tmp_path state_dir still write to a sandboxed
        # location, while production deploys keep the documented filename.
        state_path = state_dir / Path(fr_cfg.state_file).name
        trade_log_path = state_dir / "trades.jsonl"

        self._fills_reconciler = FillsReconciler(
            state_path=state_path,
            trade_log_path=trade_log_path,
            trade_log_lock=self._trade_log_lock,
            kalshi_client=self._kalshi_client,
            engine=self._engine,
            notifier=self._notifier,
            startup_safety_window_seconds=fr_cfg.startup_safety_window_hours * 3600,
            startup_max_window_seconds=fr_cfg.startup_max_window_hours * 3600,
            eventual_consistency_window_seconds=fr_cfg.eventual_consistency_window_seconds,
            max_pages_per_tick=fr_cfg.max_pages_per_tick,
            max_fills_per_tick=fr_cfg.max_fills_per_tick,
            page_delay_ms=fr_cfg.page_delay_ms,
            placement_row_lookback=fr_cfg.placement_row_lookback,
            alert_threshold_consecutive=fr_cfg.alert_threshold_consecutive,
            alert_threshold_critical_delta=fr_cfg.alert_threshold_critical_delta,
            critical_alert_cooldown_seconds=fr_cfg.critical_alert_cooldown_seconds,
        )

        # Wire manual-close config. Prefix the shakedown dedup file with
        # state_dir using the same basename-under-state_dir pattern as
        # state_file above so tests pointing state_dir at a tmp_path stay
        # sandboxed.
        shakedown_path = state_dir / Path(fr_cfg.shakedown_seen_oids_file).name
        shakedown_path.parent.mkdir(parents=True, exist_ok=True)
        self._fills_reconciler._manual_close_enabled = fr_cfg.manual_close_enabled
        self._fills_reconciler._manual_close_apply_limit = (
            fr_cfg.manual_close_apply_limit_per_restart
        )
        self._fills_reconciler._shakedown_seen_oids_path = shakedown_path
        self._fills_reconciler._load_shakedown_seen_oids()

        co_shakedown_path = state_dir / Path(fr_cfg.shakedown_seen_oids_cash_out_file).name
        co_shakedown_path.parent.mkdir(parents=True, exist_ok=True)
        self._fills_reconciler._co_apply_enabled = fr_cfg.cash_out_apply_enabled
        self._fills_reconciler._co_apply_limit = fr_cfg.cash_out_apply_limit_per_restart
        self._fills_reconciler._co_max_pnl_apply_dollars = fr_cfg.cash_out_max_pnl_apply_dollars
        self._fills_reconciler._shakedown_seen_oids_cash_out_path = co_shakedown_path
        self._fills_reconciler._load_shakedown_seen_oids_cash_out()

        logger.info(
            "FillsReconciler enabled (cadence=%d ticks, mode=%s)",
            fr_cfg.reconcile_interval_ticks,
            exec_mode,
        )

    def _setup_phase0_depth_monitor(self) -> None:
        """Construct the Phase 0 depth monitor when a Kalshi client is up.

        The monitor is independent of any strategy's ``enabled`` flag — it
        must accumulate depth samples for the full 2-week Phase 0 window
        even while the intraday strategy ships ``enabled: false``.

        Skips construction when there is no Kalshi client (no creds /
        offline test runs); the run-loop launcher tolerates a ``None``
        monitor.
        """
        if self._kalshi_client is None:
            self._phase0_depth_monitor = None
            return
        state_dir = Path(self._settings.state_dir)
        self._phase0_depth_monitor = Phase0DepthMonitor(
            kalshi_client=self._kalshi_client,
            state_dir=state_dir,
        )
        logger.info(
            "Phase0DepthMonitor configured (sink=%s)",
            self._phase0_depth_monitor.sink_path,
        )

    def _on_kxrain_task_done(self, task: asyncio.Task[None]) -> None:
        """Alert CRITICAL when a KXRAIN task exits for any reason other than cancel."""
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        logger.error(
            "KXRAIN task %s exited (%r) — the forward corpus has stopped accruing",
            task.get_name(),
            exc,
        )
        # The only detector for a dead task: the observer's liveness check runs
        # inside the task it would need to report on.
        try:
            asyncio.get_running_loop().create_task(
                self._alert_kxrain_task_death(task.get_name(), exc)
            )
        except RuntimeError:
            # No running loop (interpreter shutdown) — the ERROR above stands.
            logger.debug("KXRAIN death alert could not be scheduled", exc_info=True)

    async def _alert_kxrain_task_death(self, name: str, exc: BaseException | None) -> None:
        notifier = self._notifier
        if notifier is None:
            return
        try:
            await notifier.send_alert(
                f"KXRAIN task {name} died: {exc!r}. The forward corpus has stopped "
                "accruing and it cannot be backfilled. Restart the bot.",
                level=AlertLevel.CRITICAL,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("KXRAIN death alert delivery failed", exc_info=True)

    def _setup_kxrain_observer(self) -> None:
        """Construct the KXRAIN observer and settlement sweep. Never raises.

        ``notifier_getter`` is lazy because this runs before ``self._notifier`` is set.
        """
        cfg = getattr(self._settings, "kxrain_shadow", None)
        if cfg is None or not cfg.enabled:
            self._kxrain_observer = None
            self._kxrain_sweep = None
            return
        if self._kalshi_client is None:
            # No tick rows will be written, so this log line is the only record.
            logger.warning(
                "KXRAIN observer is enabled but there is no Kalshi client; "
                "the forward corpus will NOT accrue this run"
            )
            self._kxrain_observer = None
            self._kxrain_sweep = None
            return
        try:
            observer_config = cfg.to_observer_config(state_dir=self._settings.state_dir)
            self._kxrain_observer = KxRainObserver(
                self._kalshi_client,
                observer_config,
                notifier_getter=lambda: self._notifier,
            )
            self._kxrain_sweep = KxRainSettlementSweep(
                self._kalshi_client,
                log_dir=observer_config.log_dir,
                max_pages=cfg.settlement_sweep_max_pages,
                retention_days=cfg.log_retention_days,
                # Explicit so a series rename cannot split decisions and
                # settlements across two series.
                series_ticker=cfg.series_ticker,
                keep_dates=cfg.log_retention_days,
                # Not the default pool: its saturation has crashed the bot.
                executor=self._kxrain_observer._executor,
            )
        except Exception:
            logger.exception(
                "KXRAIN observer setup failed; continuing without it. Trading is "
                "unaffected — this feature places no orders."
            )
            self._kxrain_observer = None
            self._kxrain_sweep = None
            return
        logger.info(
            "KxRainObserver configured (log_dir=%s, version=%s, sweep_hour_utc=%d)",
            observer_config.log_dir,
            self._kxrain_observer.observer_version,
            cfg.settlement_capture_hour_utc,
        )

    async def _kxrain_sweep_loop(self) -> None:
        """Run the settlement sweep once per UTC day. Never raises.

        Alerts on ``result.anomaly``: the sweep reaches back only ~28 days.
        """
        cfg = self._settings.kxrain_shadow
        last_swept: date | None = None
        # Every restart is due; wait one poll so a crash loop does not burst
        # the shared Kalshi client at startup.
        await asyncio.sleep(self._KXRAIN_SWEEP_POLL_SECONDS)
        try:
            while True:
                if not self._running:
                    # Returning would trip the dead-task alert on a routine halt.
                    await asyncio.sleep(self._KXRAIN_SWEEP_POLL_SECONDS)
                    continue
                try:
                    now = datetime.now(UTC)
                    due = now.hour >= cfg.settlement_capture_hour_utc
                    if self._kxrain_sweep is not None and due and last_swept != now.date():
                        # Mark after success so a raise retries today.
                        result = await self._kxrain_sweep.run_once()
                        last_swept = now.date()
                        logger.info("KXRAIN settlement sweep: %s", result.to_dict())
                        if result.anomaly:
                            await self._alert_kxrain_sweep(result)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("KXRAIN settlement sweep tick failed; continuing")
                await asyncio.sleep(self._KXRAIN_SWEEP_POLL_SECONDS)
        except asyncio.CancelledError:
            raise

    async def _alert_kxrain_sweep(self, result: SettlementSweepResult) -> None:
        """WARNING when a sweep result looks like a dead sweep."""
        notifier = self._notifier
        if notifier is None:
            return
        try:
            await notifier.send_alert(
                f"KXRAIN settlement sweep anomaly: {result.to_dict()}. The corpus's "
                "only disaster-recovery path may be dead and its reach is ~28 days.",
                level=AlertLevel.WARNING,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("KXRAIN sweep alert delivery failed", exc_info=True)

    def _isotonic_artifact_path(self) -> Path:
        """Resolve the live isotonic artifact path.

        Mirrors ``weather_edge``'s loader: honors
        ``strategies.weather.isotonic_path`` when set and falls back to the
        canonical ``data/calibration/isotonic.json`` otherwise. Every tripwire
        and lineage read MUST go through this so they observe the same
        artifact the strategy is actually serving — a prior plateau-tripwire
        bug came from reading the wrong file.
        """
        configured = getattr(self._settings.strategies.weather, "isotonic_path", None)
        if configured:
            return Path(configured)
        return Path("data/calibration/isotonic.json")

    def _read_isotonic_fit_date(self) -> str | None:
        """Pull lineage from the isotonic calibration file if present."""
        try:
            data = json.loads(self._isotonic_artifact_path().read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        fit_date = data.get("fit_date") or data.get("fitted_at")
        return str(fit_date) if fit_date else None

    async def _on_calibration_drift(self, alert: DriftAlert) -> None:
        await self._notify_alert(
            format_calibration_drift_alert(alert),
            level=alert.severity,
            owner=alert.owner,
            code=alert.code,
        )

    async def _reconcile_from_kalshi(self) -> None:
        """Reconcile positions + equity on startup.

        Live/paper: query the Kalshi API.  Shadow: replay
        ``state/trades.jsonl`` via :class:`ShadowBroker` so the
        ``TOTAL_EXPOSURE_CAP`` stays binding across restarts.

        Bankroll semantics: ``RiskManager._bankroll`` is **total equity**
        (cash + open position notional), not bare Kalshi cash.  Sizing,
        drawdown, and the periodic drift check all operate against this
        equity figure, so a position-heavy session does not look like a
        phantom drawdown the moment cash drops.
        """
        assert self._risk_manager is not None

        if self._settings.is_shadow:
            await self._reconcile_shadow()
            return

        if self._kalshi_client is None:
            return

        # Balance — ``cash`` stays None on failure so we can preserve the
        # last-known equity rather than reset to a stale config value mid-
        # session (a fresh-init RiskManager simply keeps Settings.bankroll).
        cash: float | None = None
        try:
            balance_resp = await self._kalshi_client.get_balance()
            self._check_contract(balance_resp, SCHEMA_BALANCE, context="balance_startup")
            balance_cents = balance_resp.get("balance", 0)
            cash = balance_cents / 100.0

            if cash > self._settings.bankroll * 100:
                logger.warning(
                    "Kalshi balance $%.2f is >100x config bankroll $%.2f "
                    "— possible cents/dollars confusion",
                    cash,
                    self._settings.bankroll,
                )
        except Exception:
            logger.warning(
                "Failed to fetch Kalshi balance — preserving last-known "
                "risk-manager state (bankroll=$%.2f)",
                self._risk_manager.bankroll,
                exc_info=True,
            )
            return

        # Positions (Kalshi v2 schema — see :meth:`_parse_kalshi_decimal`)
        positions: dict[str, float] = {}
        _live_position_rows: list[tuple[str, float, float]] = []
        market_positions = await self._fetch_unsettled_positions(stage="startup hydration")
        if market_positions is not None:
            for pos in market_positions:
                self._check_contract(pos, SCHEMA_POSITION_ROW, context="position_startup")
                ticker = pos.get("ticker", "")
                # position_fp is signed (positive=YES, negative=NO) and may be
                # fractional; we only need the sign and zero-detection here.
                quantity = self._parse_kalshi_decimal(pos.get("position_fp", "0"))
                # market_exposure_dollars is the canonical positive cost basis
                # regardless of side, so no per-side derivation is needed.
                dollars = self._parse_kalshi_decimal(pos.get("market_exposure_dollars", "0"))

                if not ticker or quantity == 0:
                    continue

                positions[ticker] = dollars
                _live_position_rows.append((ticker, quantity, dollars))
                logger.info(
                    "Reconciled position: %s = %.2f contracts, $%.2f exposure",
                    ticker,
                    quantity,
                    dollars,
                )

        position_notional = sum(positions.values())
        equity = cash + position_notional

        logger.info(
            "Kalshi reconcile: cash=$%.2f, positions=$%.2f (n=%d), equity=$%.2f",
            cash,
            position_notional,
            len(positions),
            equity,
        )

        position_strategy = build_position_strategy_index(
            Path(self._settings.state_dir) / "trades.jsonl"
        )
        self._risk_manager.hydrate_positions(positions, equity, position_strategy=position_strategy)

        live_positions = [
            LivePosition(
                position_key=ticker,
                market_ticker=ticker,
                side=Side.YES if quantity > 0 else Side.NO,
                dollars=dollars,
            )
            for ticker, quantity, dollars in _live_position_rows
        ]
        if live_positions and self._engine is not None:
            n = self._engine.hydrate_live_positions(live_positions)
            logger.info("Live reconcile: hydrated %d engine stub(s)", n)
            upgraded = self._engine.restore_fills_from_log()
            logger.info("Live reconcile: upgraded %d stub(s) to FILLED from trade log", upgraded)

    async def _reconcile_shadow(self) -> None:
        """Rebuild shadow-mode positions from the trade log.

        Hydrates the risk manager AND the execution engine so the normal
        ``_settle_shadow`` expiry sweep can release positions opened in a
        previous process.
        """
        assert self._risk_manager is not None
        assert self._engine is not None

        broker = ShadowBroker(
            trade_log_path=Path(self._settings.state_dir) / "trades.jsonl",
            configured_bankroll=self._settings.bankroll,
        )
        positions = broker.replay()
        exposure_by_key = {p.position_key: p.dollars for p in positions}

        if positions:
            logger.info(
                "Shadow reconcile: hydrated %d position(s), total exposure $%.2f",
                len(positions),
                sum(exposure_by_key.values()),
            )
        else:
            logger.info("Shadow reconcile: no open positions in trade log")

        position_strategy = build_position_strategy_index(
            Path(self._settings.state_dir) / "trades.jsonl"
        )
        self._risk_manager.hydrate_positions(
            exposure_by_key, broker.bankroll, position_strategy=position_strategy
        )
        self._engine.hydrate_shadow_positions(positions)

    def _restore_risk_state(self, _today: date | None = None) -> None:
        """Load persisted risk state from JSON file.

        ``_today`` is injectable for testing; production callers omit it and
        get the current ET date via ``_et_today()``.
        """
        assert self._risk_manager is not None
        state_dir = Path(self._settings.state_dir)

        # Exposure override is a separate file from the main risk_state.json —
        # load it independently so a corrupt main file doesn't also wipe the
        # operator's last-known cap.
        self._risk_manager.hydrate_exposure_override(state_dir)

        snapshot = load_state(state_dir)

        if snapshot is None:
            logger.info("No persisted risk state found — starting fresh")
            return

        today = _today or self._et_today()
        today_str = today.isoformat()

        # Mode-switch guard: clear daily counters; also reset peak to current
        # bankroll since the previous mode's peak is not comparable.
        mode_switched = snapshot.execution_mode != self._settings.execution.mode
        if mode_switched:
            logger.warning(
                "Execution mode changed (%s → %s) — skipping daily P&L and peak restore",
                snapshot.execution_mode,
                self._settings.execution.mode,
            )
            snapshot.peak_bankroll = self._risk_manager.bankroll

        self._risk_manager.hydrate_from_state(snapshot)

        # Daily-counter reset — runs for both mode-switch and day-boundary.
        # reset_daily() is also what the tick loop calls at runtime, so both
        # paths now share the same mechanism.
        # Note: _was_halted and _silent_alert_sent are intentionally NOT cleared
        # here — they are set to False in __init__ so a fresh process always
        # starts with clean defaults; the tick-loop path clears them only when
        # the date rolls while the process is already running.
        if mode_switched:
            self._reset_daily_with_fanout()
        elif snapshot.current_trading_date and snapshot.current_trading_date != today_str:
            logger.info(
                "New trading day (was %s, now %s) — resetting daily counters",
                snapshot.current_trading_date,
                today_str,
            )
            self._reset_daily_with_fanout()

        self._current_trading_date = (
            date.fromisoformat(snapshot.current_trading_date)
            if snapshot.current_trading_date
            else today
        )

    async def run(self) -> None:
        """
        Main event loop.

        Runs until cancelled or kill switch is activated.
        """
        if self._risk_manager is None or self._engine is None:
            raise RuntimeError("Call initialize() before run()")

        self._running = True
        self._install_signal_handlers()

        # Independent systemd watchdog heartbeat. MUST run on its own task: the
        # main loop runs ticks serially (``await self._tick()``), so a single
        # slow tick (Open-Meteo outage → one ensemble fetch burns ~127s across
        # awaited retries) parks the main-loop coroutine inside the tick and a
        # ping inlined in the tick can't re-fire until the tick returns. Only a
        # separate task keeps the deadline fresh mid-tick. Started before the
        # startup_delay so the deadline stays fresh during the initial wait and
        # the first (potentially slow) tick. See _watchdog_heartbeat_loop.
        self._watchdog_task = asyncio.create_task(
            self._watchdog_heartbeat_loop(), name="watchdog-heartbeat"
        )

        if self._kalshi_ws is not None:
            self._ws_task = asyncio.create_task(self._ws_listener(), name="kalshi-ws")

        if self._notifier is not None:
            self._telegram_task = asyncio.create_task(self._start_telegram(), name="telegram-bot")

        if self._calibration_monitor is not None:
            try:
                await self._calibration_monitor.start()
            except Exception:
                logger.warning("Calibration monitor failed to start", exc_info=True)

        self._tripwire_task = asyncio.create_task(
            self._audit_tripwire_loop(), name="audit-tripwires"
        )

        # Independent of strategy enabled flags; cancelled in shutdown().
        if self._phase0_depth_monitor is not None:
            self._phase0_depth_task = asyncio.create_task(
                self._phase0_depth_monitor.run(), name="phase0-depth-monitor"
            )

        # Exactly one task: run_once is not re-entrant.
        if self._kxrain_observer is not None:
            self._kxrain_task = asyncio.create_task(
                self._kxrain_observer.run(), name="kxrain-observer"
            )
            self._kxrain_task.add_done_callback(self._on_kxrain_task_done)
        if self._kxrain_sweep is not None:
            self._kxrain_sweep_task = asyncio.create_task(
                self._kxrain_sweep_loop(), name="kxrain-settlement-sweep"
            )
            self._kxrain_sweep_task.add_done_callback(self._on_kxrain_task_done)

        poll_interval = self._settings.execution.poll_interval_seconds
        startup_delay = self._settings.execution.startup_delay_seconds
        self._log_kill_switch_state_at_startup()
        logger.info("Starting main loop (poll every %ds)", poll_interval)

        if startup_delay > 0:
            logger.info("Startup delay: waiting %ds before first trade scan", startup_delay)
            await self._wait_or_stop(startup_delay)

        try:
            await self._tick_loop(poll_interval)
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        finally:
            await self.shutdown()

    def _log_kill_switch_state_at_startup(self) -> None:
        """Record the STOP-file state once per process start.

        Clearing a kill means deleting ``state/STOP`` and restarting, which
        logs no "cleared" line, so ``scripts/pull_post_mortem_data.py`` uses
        this line to close the kill interval. Keep the message text stable.
        """
        try:
            active = self._settings.is_kill_switch_active()
        except Exception:
            logger.warning("Kill switch state unknown at startup", exc_info=True)
            return
        if active:
            logger.warning("Kill switch ACTIVE at startup")
        else:
            logger.info("Kill switch inactive at startup")

    def _request_stop(self) -> None:
        """Ask the main loop to stop, waking it if it is between ticks."""
        self._running = False
        self._stop_event.set()

    async def _tick_loop(self, poll_interval: float) -> None:
        """Run ticks until stopped.

        The inter-tick wait must end the moment a stop is requested. The poll
        interval equals systemd's ``TimeoutStopSec`` (60s), so a plain sleep
        lets SIGKILL land before ``shutdown()`` runs.
        """
        while self._running:
            await self._tick()
            await self._wait_or_stop(poll_interval)

    async def _wait_or_stop(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)

    async def _watchdog_heartbeat_loop(self) -> None:
        """Independent systemd watchdog heartbeat.

        Pings ``WATCHDOG=1`` on a fixed cadence derived from ``WATCHDOG_USEC``
        so liveness reflects "the event loop can still schedule a coroutine",
        NOT "the most recent tick finished its work".

        Why a dedicated task and not an inline ping: the main loop runs ticks
        serially (``while self._running: await self._tick(); await sleep()``).
        During an Open-Meteo outage a single ensemble fetch can block ~127s
        (4 × 30s timeout + backoff) inside one tick — the main-loop coroutine
        is parked inside ``_tick`` the whole time, so a ping at tick start or
        end can't fire again until the tick returns. tick_duration +
        poll_interval then exceeds WatchdogSec (120s) → systemd SIGABRT crash
        loop (the 2026-06 incident). Only a separate task can refresh the
        deadline while a tick is mid-flight.

        The watchdog stays MEANINGFUL: a genuinely wedged event loop (a
        sync/CPU-bound block, not an ``await``) starves this coroutine too, so
        systemd still kills a truly hung process — exactly what a watchdog is
        for. ``_sd_notify`` swallows its own errors, so the body can't raise;
        ``CancelledError`` propagates for clean shutdown.
        """
        raw = os.environ.get("WATCHDOG_USEC")
        if not raw:
            return  # watchdog inactive (local dev / non-systemd); _sd_notify no-ops anyway
        try:
            interval = max(5.0, (int(raw) / 1_000_000) / 3.0)
        except ValueError:
            interval = 30.0
        logger.info("Watchdog heartbeat loop active (ping every %.0fs)", interval)
        while self._running:
            _sd_notify("WATCHDOG=1")
            await asyncio.sleep(interval)

    def _open_event_keys(self) -> set[tuple[str, str, str]]:
        """Return the set of (city, var, forecast_date) event keys for all
        currently-open positions.  Used by the favorite_bias deference hook so
        that new favorite_bias signals don't stack on existing open positions.

        RiskManager._positions is dict[market_id -> dollars] with no strategy
        metadata (verified), so we derive event keys from the ticker directly
        via the per_day_event_cap parser. Unparseable tickers (non-weather
        market formats) are silently skipped; this is safe — they could
        never collide with weather-market event keys anyway.
        """
        from python.strategies.per_day_event_cap import _parse_event_key_from_ticker

        keys: set[tuple[str, str, str]] = set()
        if self._risk_manager is None:
            return keys
        for market_id in self._risk_manager.open_position_keys():
            k = _parse_event_key_from_ticker(market_id)
            if k is not None:
                keys.add(k)
        return keys

    async def _send_fill_alerts(self, records: list[OrderRecord]) -> None:
        """Announce newly-filled orders and stamp their dedup counter.

        ``records`` is the engine's unalerted-fill set
        (``ExecutionEngine.collect_unalerted_fills``). Only the records the
        notifier actually announced are stamped, so a rate-limit drop, a
        send-failure, or a transient exception leaves the record unstamped and
        the next tick's scan retries the announcement.
        """
        if self._notifier is None or not records:
            return
        try:
            announced = await self._notifier.send_trade_batch_alert(records)
        except Exception:
            logger.warning("Trade batch alert failed", exc_info=True)
            return
        for r in announced:
            r.last_alerted_fill_count = r.fill_count

    def _reap_save_task(self) -> None:
        """Clear a completed save task slot so the next tick can spawn a new
        one. Failure handling lives in ``_save_under_lock``; touching
        ``.exception()`` here only prevents an "exception was never
        retrieved" asyncio log."""
        if self._save_task is not None and self._save_task.done():
            if not self._save_task.cancelled():
                exc = self._save_task.exception()
                if exc is not None:
                    logger.error(
                        "Risk state save task raised unexpectedly: %s",
                        exc,
                        exc_info=exc,
                    )
            self._save_task = None

    async def _halt_for_kill_switch(self) -> None:
        """Cancel all open orders and alert. Fails closed on filesystem
        errors (see ``Settings.is_kill_switch_active``)."""
        assert self._engine is not None
        logger.critical("KILL SWITCH — halting all trading and cancelling orders")
        cancelled, cancel_truncated = await self._engine.cancel_all_open()
        await self._notify_alert(
            "KILL SWITCH activated — halting trading", level=AlertLevel.CRITICAL
        )
        if cancel_truncated:
            await self._notify_alert(
                f"Kill switch fired but ORDER CANCEL TRUNCATED — "
                f"only {len(cancelled)} resting orders cancelled before "
                f"MAX_PAGES cap; additional live orders may remain. "
                f"Manual intervention required.",
                level=AlertLevel.CRITICAL,
            )
        self._request_stop()

    def _roll_trading_day(self) -> None:
        """Reset daily P&L counters at midnight ET. ``_reset_daily_with_fanout``
        calls ``risk_manager.reset_daily`` AND fans out to every strategy
        exposing a ``reset_daily_counters`` hook (intraday's per-cell daily
        trade-count cap)."""
        today = self._et_today()
        if self._current_trading_date is not None and today != self._current_trading_date:
            self._reset_daily_with_fanout()
            self._was_halted = False
            self._silent_alert_sent = False
        self._current_trading_date = today

    async def _settle_and_alert(self) -> None:
        """Settle expired positions to free exposure for new trades. Alerts
        only on losing settlements — winning/zero-P&L settlements are
        routine and already visible via /status + the daily report (pnl ==
        0 is shadow-mode sparse results)."""
        assert self._engine is not None
        assert self._risk_manager is not None
        settlements = await self._engine.settle_expired_positions()
        for s in settlements:
            if s.pnl < 0.0:
                await self._notify_alert(
                    format_settlement_alert(
                        s.market_ticker, s.result, s.pnl, self._risk_manager.bankroll
                    ),
                    level=AlertLevel.WARNING,
                )

    async def _refresh_pending_fills(self) -> None:
        """Check fills on pending orders (live/paper; no-op for shadow).
        ``zero_fill_cancelled`` carries TTL-expired and server-cancelled
        records with ``fill_count == 0`` — release per-strategy ticker
        dedup so a follow-up scan can re-attempt the same market."""
        assert self._engine is not None
        _filled_records, zero_fill_cancelled = await self._engine.check_all_pending_fills()
        for record in zero_fill_cancelled:
            if record.fill_count != 0:
                continue  # defensive — engine already filters
            strat_name = record.signal.strategy_name
            for strategy in self._strategies:
                if strategy.name == strat_name and hasattr(strategy, "unmark_traded"):
                    strategy.unmark_traded(record.signal.market_ticker)

    async def _alert_daily_loss_halt(self) -> None:
        """Edge-triggered: alert once on the halt transition."""
        assert self._risk_manager is not None
        if self._risk_manager.is_halted and not self._was_halted:
            await self._notify_alert(
                f"DAILY LOSS HALT: P&L ${self._risk_manager.daily_pnl:+.2f} "
                f"breached {self._settings.risk.max_daily_loss_pct:.0%} limit. "
                f"Trading suspended until next day.",
                level=AlertLevel.CRITICAL,
            )
        self._was_halted = self._risk_manager.is_halted

    async def _run_scheduled_reconciliation(self) -> None:
        """Periodic position reconciliation on its own cadence, disabled
        entirely in shadow mode."""
        if not self._settings.is_shadow:
            interval = self._settings.execution.reconcile_interval_ticks
        else:
            interval = 0
        if interval > 0 and self._tick_count - self._last_reconcile_tick >= interval:
            self._last_reconcile_tick = self._tick_count
            try:
                await self._periodic_reconciliation()
            except Exception:
                logger.warning("Periodic reconciliation failed", exc_info=True)

    async def _run_scheduled_fills_reconciler(self) -> None:
        """Fills reconciler scheduling. Outer try/except is belt-and-braces —
        tick() already swallows its own exceptions and never raises by
        contract — but a regression there must not crash the orchestrator
        main loop."""
        fr_interval = self._settings.fills_reconciler.reconcile_interval_ticks
        if (
            self._fills_reconciler is not None
            and fr_interval > 0
            and self._tick_count - self._last_scheduled_fills_tick >= fr_interval
        ):
            self._last_scheduled_fills_tick = self._tick_count
            try:
                await self._fills_reconciler.tick()
            except Exception:
                logger.exception(
                    "FillsReconciler.tick() raised — never-raises "
                    "contract violated; treat as a bug, not an API hiccup"
                )

    def _broadcast_bankroll(self) -> None:
        """Sync per-strategy book (pct × equity) to each strategy when equity
        changes. ``book_for()`` returns full equity for unallocated
        strategies and the empty-map default, so this is identical to
        legacy when ``risk.strategy_allocations`` is empty. Allocations are
        static within a process, so book_s changes iff equity changes — the
        single-float change-detector stays valid."""
        assert self._risk_manager is not None
        current_bankroll = self._risk_manager.bankroll
        if current_bankroll != self._last_broadcast_bankroll:
            for strategy in self._strategies:
                strategy.update_bankroll(self._risk_manager.book_for(strategy.name))
            self._last_broadcast_bankroll = current_bankroll

    async def _alert_circuit_breakers(self) -> None:
        assert self._risk_manager is not None
        disabled = self._risk_manager.check_circuit_breakers()
        if disabled:
            logger.warning("Circuit breakers tripped for: %s", disabled)
            for name in disabled:
                tracker = self._risk_manager.get_tracker(name)
                await self._notify_alert(
                    f"Circuit breaker tripped: {name} — {tracker.disabled_reason}",
                    level=AlertLevel.CRITICAL,
                )

    def _reload_emos_table(self) -> None:
        """EmosTable hot-reload. Owned by the orchestrator: rebinds the
        shared holder once per tick BEFORE any strategy's
        ``generate_signals`` runs. ``EmosTable.check_and_reload`` returns
        the new table on swap, or None on no-change / failure; the ``or
        holder.table`` guard preserves the prior reference in the no-swap
        path so consumers never observe ``None``. Every EMOS-consuming
        strategy holds the SAME holder reference, so the rebind is visible
        to all of them. Fires AFTER the kill-switch and circuit-breaker
        checks (matching weather_edge's prior per-scan reload site)."""
        if self._emos_holder is not None:
            try:
                self._emos_holder.table = (
                    self._emos_holder.table.check_and_reload() or self._emos_holder.table
                )
            except Exception:
                logger.exception("EMOS hot-reload check failed — keeping prior table")

    def _eligible_strategies(self) -> list[Strategy]:
        """Enabled strategies whose circuit-breaker tracker is also enabled."""
        assert self._risk_manager is not None
        eligible: list[Strategy] = []
        for strategy in self._strategies:
            if not strategy.enabled:
                continue
            tracker = self._risk_manager.get_tracker(strategy.name)
            if not tracker.enabled:
                continue
            eligible.append(strategy)
        return eligible

    async def _run_strategy(self, strategy: Strategy) -> list[TradeSignal]:
        bind_contextvars(strategy=strategy.name)
        assert self._risk_manager is not None  # set by initialize()
        try:
            signals = await strategy.generate_signals()
            if signals:
                logger.info("Strategy %s generated %d signal(s)", strategy.name, len(signals))
                self._last_strategy_signal[strategy.name] = time.monotonic()
            # Reset error counter on successful scan (no-op + no
            # save when already 0).
            self._risk_manager.clear_strategy_errors(strategy.name)
            return signals
        except Exception:
            logger.exception("Strategy %s failed during signal generation", strategy.name)
            # Track errors separately — do NOT record as a losing trade,
            # because transient API/network failures would poison the
            # circuit breaker win-rate metric and disable the strategy.
            consecutive = self._risk_manager.record_strategy_error(strategy.name)
            if consecutive >= STRATEGY_ERROR_ALERT_THRESHOLD:
                await self._notify_alert(
                    f"Strategy {strategy.name}: {consecutive} "
                    f"consecutive errors — check logs",
                    level=AlertLevel.CRITICAL,
                )
            else:
                await self._notify_alert(
                    f"Strategy {strategy.name} failed during signal generation",
                    level=AlertLevel.WARNING,
                )
            return []
        finally:
            unbind_contextvars("strategy")

    async def _collect_signals(self, eligible: list[Strategy]) -> list[TradeSignal]:
        results = await asyncio.gather(*(self._run_strategy(s) for s in eligible))
        return [sig for batch in results for sig in batch]

    def _drop_intraday_shadow_signals(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """Drop intraday_emos_high signals when the strategy is not yet
        cleared for live execution. Fail-CLOSED on error — this feature
        exists to prevent live execution of an unvalidated strategy."""
        shadow_only = self._settings.strategies.intraday_emos_high.shadow_only
        try:
            before = len(signals)
            signals = apply_intraday_shadow_drop(signals, shadow_only=shadow_only)
            if len(signals) != before:
                logger.info(
                    "intraday_shadow_drop suppressed %d signal(s) (shadow_only=True)",
                    before - len(signals),
                )
        except Exception:
            if shadow_only:
                signals = [s for s in signals if s.strategy_name != "intraday_emos_high"]
            logger.warning(
                "intraday_shadow_drop hook failed; fail-closed filter applied",
                exc_info=True,
            )
        return signals

    def _defer_intraday_asos(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """intraday_asos_high yields to weather_edge on shared (city, var,
        forecast_date) events AND to any open position. Log-and-pass-through
        on error so a hook failure never silently drops a whole tick's
        signals."""
        try:
            asos_open_keys = self._open_event_keys()
            before = len(signals)
            signals = apply_intraday_asos_deference(signals, asos_open_keys)
            if len(signals) != before:
                logger.info(
                    "intraday_asos deference dropped %d signal(s)",
                    before - len(signals),
                )
        except Exception:
            logger.warning(
                "intraday_asos deference hook failed; passing signals through",
                exc_info=True,
            )
        return signals

    def _drop_intraday_asos_shadow_signals(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """Drop intraday_asos_high signals when the strategy is not yet
        cleared for live execution. Fail-CLOSED on error — this feature
        exists to prevent live execution of an unvalidated strategy."""
        shadow_only = self._settings.strategies.intraday_asos_high.shadow_only
        try:
            before = len(signals)
            signals = apply_intraday_asos_shadow_drop(signals, shadow_only=shadow_only)
            if len(signals) != before:
                logger.info(
                    "intraday_asos_shadow_drop suppressed %d signal(s) (shadow_only=True)",
                    before - len(signals),
                )
        except Exception:
            if shadow_only:
                signals = [s for s in signals if s.strategy_name != "intraday_asos_high"]
            logger.warning(
                "intraday_asos_shadow_drop hook failed; fail-closed filter applied",
                exc_info=True,
            )
        return signals

    def _drop_weather_shadow_signals(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """Drop weather_edge signals when the strategy runs shadow-only (no
        live orders — it keeps evaluating + logging the calibration
        corpus). Runs AFTER generate_signals (predictions/eval-audit
        already logged at signal-generation time), so the corpus is
        preserved. Fail-CLOSED on error (filter via list comp that cannot
        raise)."""
        weather_shadow_only = self._settings.strategies.weather.shadow_only
        try:
            before = len(signals)
            signals = apply_weather_shadow_drop(signals, shadow_only=weather_shadow_only)
            if len(signals) != before:
                logger.info(
                    "weather_shadow_drop suppressed %d signal(s) (shadow_only=True)",
                    before - len(signals),
                )
        except Exception:
            if weather_shadow_only:
                signals = [s for s in signals if s.strategy_name != "weather_edge"]
            logger.warning(
                "weather_shadow_drop hook failed; fail-closed filter applied",
                exc_info=True,
            )
        return signals

    def _defer_favorite_bias(self, signals: list[TradeSignal]) -> list[TradeSignal]:
        """favorite_bias yields to weather_edge on shared (city, var,
        forecast_date) events AND to any open position. Log-and-pass-through
        on error so a hook failure never silently drops a whole tick's
        signals."""
        try:
            open_keys = self._open_event_keys()
            before = len(signals)
            signals = apply_favorite_bias_deference(signals, open_keys)
            if len(signals) != before:
                logger.info(
                    "favorite_bias deference dropped %d signal(s)",
                    before - len(signals),
                )
        except Exception:
            logger.warning(
                "favorite_bias deference hook failed; passing signals through",
                exc_info=True,
            )
        return signals

    def _commit_per_day_cap(self, records: list[OrderRecord]) -> None:
        """For each successfully placed order from a cap-using strategy,
        bind (event_key, order_id) so a later cancel can release. REJECTED
        / SKIPPED / ZERO_SIZE / ERROR records have no real placement and
        consume no slot."""
        for strategy in self._strategies:
            cap = getattr(strategy, "_per_day_cap", None)
            if cap is None:
                continue
            for record in records:
                if record.signal.strategy_name != strategy.name:
                    continue
                if record.status in _PER_DAY_CAP_EXCLUDED_STATUSES:
                    continue
                key = _event_key_from_signal(record.signal)
                if not all(key):
                    logger.warning(
                        "per_day_cap: order %s has incomplete signal metadata "
                        "(city/market_type/forecast_date); not committing",
                        record.order_id,
                    )
                    continue
                try:
                    cap.commit(key, record.order_id)
                except ValueError:
                    logger.exception(
                        "per_day_cap: failed to commit %s for %s — "
                        "already bound elsewhere?",
                        record.order_id,
                        key,
                    )

    def _mark_signals_traded(self, records: list[OrderRecord]) -> None:
        for record in records:
            if record.status not in _EXECUTED_STATUSES:
                continue
            strat_name = record.signal.strategy_name
            for strategy in self._strategies:
                if strategy.name == strat_name and hasattr(strategy, "mark_traded"):
                    strategy.mark_traded(record.signal.market_ticker)

    async def _execute_signals(self, signals: list[TradeSignal]) -> None:
        """Sort by effective edge (largest first) for priority execution.
        effective_edge accounts for spread costs and ensemble confidence,
        unlike raw edge which can prioritise wide-spread markets."""
        assert self._engine is not None
        signals.sort(key=lambda s: s.effective_edge, reverse=True)

        records = await self._engine.execute_batch(signals)

        self._commit_per_day_cap(records)
        self._mark_signals_traded(records)

        # Fill announcements are NOT sent here — they go through the
        # single end-of-tick scan in _tick, which also covers later-tick,
        # degraded-recovery, cancel-race, and no-signal-tick fills that
        # this synchronous ``records``-only path never announced.

        approved = sum(
            1
            for r in records
            if r.status not in (OrderStatus.REJECTED, OrderStatus.SKIPPED, OrderStatus.ERROR)
        )
        rejected = sum(1 for r in records if r.status == OrderStatus.REJECTED)
        skipped = sum(1 for r in records if r.status == OrderStatus.SKIPPED)
        logger.info(
            "Tick complete: %d signals, %d approved, %d rejected, %d skipped",
            len(signals),
            approved,
            rejected,
            skipped,
        )

    async def _maybe_alert_silent_bot(
        self, *, eligible: list[Strategy], in_market_hours: bool
    ) -> None:
        logger.debug("No signals this tick")
        if not eligible:  # only alert if strategies were actually running
            return
        silence_hours = (time.monotonic() - self._last_signal_time) / 3600.0
        threshold = self._settings.telegram.silent_bot_alert_hours
        if in_market_hours and silence_hours >= threshold and not self._silent_alert_sent:
            await self._notify_alert(
                f"SILENT BOT: No signals for {silence_hours:.1f}h. "
                f"Check market availability and strategy health.",
                level=AlertLevel.WARNING,
            )
            self._silent_alert_sent = True

    async def _tick(self) -> None:
        """
        Single iteration of the main loop:
          1. Check kill switch
          2. Run each enabled strategy
          3. Filter signals through risk manager
          4. Execute approved signals
        """
        assert self._risk_manager is not None
        assert self._engine is not None

        self._reap_save_task()

        if self._settings.is_kill_switch_active():
            await self._halt_for_kill_switch()
            return

        try:
            # Refresh the systemd watchdog deadline at the START of the tick,
            # decoupled from tick duration. Without this, a slow weather-fetch
            # phase (Open-Meteo 429 storms + retries) could burn past
            # WatchdogSec before the end-of-tick keepalive is reached ->
            # SIGABRT crash-loop. The bounded prefetch timeouts keep the
            # watchdog MEANINGFUL: a genuinely wedged event loop still can't
            # re-enter _tick to ping, so it still dies.
            _sd_notify("WATCHDOG=1")
            self._tick_count += 1
            await self._check_ws_health()

            self._roll_trading_day()

            await self._settle_and_alert()

            # Prune completed order records to bound memory
            self._engine.prune_completed_orders()

            await self._refresh_pending_fills()

            await self._alert_daily_loss_halt()

            await self._run_scheduled_reconciliation()

            await self._run_scheduled_fills_reconciler()

            self._broadcast_bankroll()

            await self._alert_circuit_breakers()

            # Observe-only pod-exposure logging (on-change, no block).
            self._risk_manager.maybe_log_pod_exposure()

            self._reload_emos_table()

            eligible = self._eligible_strategies()
            all_signals = await self._collect_signals(eligible)

            # Liveness watchdog: the strategy layer producing ANY candidate signal
            # proves the bot is alive — even when those signals are subsequently
            # shadow-dropped (weather_edge / intraday shadow-only place zero orders).
            # Refresh liveness PRE-drop so intentional shadow suppression does not
            # trip the SILENT BOT alert (which only cares about strategy health).
            if all_signals:
                self._last_signal_time = time.monotonic()
                self._silent_alert_sent = False

            # Weather shadow gate runs BEFORE the deference hooks so they do
            # not defer to a weather_edge that will not actually trade.
            all_signals = self._drop_weather_shadow_signals(all_signals)
            all_signals = self._defer_favorite_bias(all_signals)
            all_signals = self._drop_intraday_shadow_signals(all_signals)
            all_signals = self._defer_intraday_asos(all_signals)
            all_signals = self._drop_intraday_asos_shadow_signals(all_signals)

            # Hoisted out of the no-signals branch below: the per-strategy check
            # must run even when ANOTHER strategy produced signals — that is
            # exactly the case the aggregate SILENT BOT watchdog cannot see.
            now_et = datetime.now(ET)
            in_market_hours = 9 <= now_et.hour < 18 and now_et.weekday() < 5

            # (Liveness already refreshed pre-drop above.)
            if all_signals:
                await self._execute_signals(all_signals)
            else:
                await self._maybe_alert_silent_bot(
                    eligible=eligible, in_market_hours=in_market_hours
                )

            await self._check_strategy_liveness(in_market_hours=in_market_hours)

            # Sole feed for fill notifications: a per-tick scan of the engine's
            # order book, deduped per new fill quantity. Runs on every tick
            # (including quiet ones) so a later-tick fill is never silent.
            if self._notifier is not None:
                await self._send_fill_alerts(self._engine.collect_unalerted_fills())

            self._persist_if_dirty()
            _sd_notify("WATCHDOG=1")
            await self._ping_healthcheck()
            self._consecutive_tick_failures = 0
        except asyncio.CancelledError:
            raise  # Must propagate for clean shutdown
        except Exception:
            await self._handle_tick_exception()

    async def _handle_tick_exception(self) -> None:
        self._consecutive_tick_failures += 1
        logger.exception(
            "Tick failed (%d consecutive) — will retry next interval",
            self._consecutive_tick_failures,
        )
        if self._consecutive_tick_failures >= self._TICK_FAILURE_ALERT_THRESHOLD:
            await self._notify_alert(
                f"CRITICAL: {self._consecutive_tick_failures} consecutive tick failures "
                f"— check logs immediately",
                level=AlertLevel.CRITICAL,
            )
        if self._consecutive_tick_failures >= self._TICK_FAILURE_KILL_THRESHOLD:
            logger.critical("20 consecutive tick failures — activating kill switch")
            self._settings.activate_kill_switch()
            self._request_stop()

    def _build_state_snapshot(self) -> dict[str, Any]:
        """Build the full serializable risk state dict."""
        assert self._risk_manager is not None
        return serialize_risk_state(
            **self._risk_manager.export_state(),
            current_trading_date=(
                self._current_trading_date.isoformat() if self._current_trading_date else None
            ),
            execution_mode=self._settings.execution.mode,
        )

    async def _save_under_lock(self, state_dir: Path, snapshot: dict[str, Any]) -> None:
        """Persist a snapshot under ``self._save_lock``.

        The lock guarantees that even if a future refactor lets two save
        tasks coexist, only one writes at a time — so ``_atomic_write`` is
        never racing two threads against the same file.

        Tracks consecutive failures via ``RiskManager.record_save_failure``
        / ``clear_save_failures`` and fires a CRITICAL alert when the
        threshold is crossed. Edge-triggered: re-arms on the next success
        so a sustained outage doesn't spam Telegram every tick. On save
        failure or post-lock exception, re-arms the dirty flag so the next
        tick retries.
        """
        assert self._risk_manager is not None
        try:
            async with self._save_lock:
                ok = await save_state(state_dir, snapshot)
            # Counter mutations and alerting happen OUTSIDE the lock — both safe
            # because _persist_if_dirty serializes _save_under_lock invocations on
            # the orchestrator's single event loop, and alerting under the lock
            # would block the next save.
            if ok:
                if self._risk_manager.consecutive_save_failures > 0:
                    logger.info(
                        "Risk state save recovered after %d failure(s)",
                        self._risk_manager.consecutive_save_failures,
                    )
                self._risk_manager.clear_save_failures()
                # Re-arm on every successful save (not just N→0 transitions) — a
                # successful save means persistence is healthy, so any pending
                # alert state is stale.
                self._save_failure_alert_sent = False
                return
            # Failure path: re-arm dirty so next tick retries against the
            # latest in-memory state.
            self._risk_manager.mark_dirty()
            consecutive = self._risk_manager.record_save_failure()
            if consecutive >= SAVE_FAILURE_ALERT_THRESHOLD and not self._save_failure_alert_sent:
                await self._notify_alert(
                    f"{PERSISTENCE_BROKEN_ALERT_PREFIX}: {consecutive} consecutive "
                    f"risk-state save failures. Check disk space and permissions "
                    f"on {state_dir / STATE_FILENAME}.",
                    level=AlertLevel.CRITICAL,
                )
                self._save_failure_alert_sent = True
        except Exception:
            logger.exception("Unexpected exception in _save_under_lock")
            try:
                self._risk_manager.mark_dirty()
            except Exception:
                logger.exception("Failed to re-arm state_dirty after _save_under_lock exception")

    def _persist_if_dirty(self) -> None:
        """Write risk state (if changed) and heartbeat.  Fire-and-forget."""
        assert self._risk_manager is not None
        state_dir = Path(self._settings.state_dir)

        if self._risk_manager.state_dirty:
            # Drop-on-overflow: if a previous save is still in flight, defer
            # to the next tick. The dirty flag stays set so we don't lose
            # the change, and we avoid building a queue of stale snapshots.
            # The lock inside ``_save_under_lock`` is belt-and-suspenders.
            if self._save_task is None or self._save_task.done():
                self._save_task = asyncio.create_task(
                    self._save_under_lock(state_dir, self._build_state_snapshot()),
                    name="persist-risk-state",
                )
                self._risk_manager.mark_state_clean()
            else:
                logger.debug("Risk state save still in flight — deferring to next tick")

        # Hold the reference: CPython only weakly tracks tasks via
        # ``_all_tasks``, and the documented contract is to keep the result
        # of ``create_task`` alive until completion. Overwriting on each tick
        # is safe — ``write_heartbeat`` is short-lived and self-handles errors.
        self._heartbeat_task = asyncio.create_task(write_heartbeat(state_dir), name="heartbeat")

    async def _ping_healthcheck(self) -> None:
        """Ping healthchecks.io dead-man's switch. Never raises."""
        if self._hc_client is None:
            return
        try:
            await self._hc_client.get(self._settings.healthcheck_ping_url)
        except Exception:
            logger.debug("Healthcheck ping failed", exc_info=True)

    # Hourly cadence: cheap (one stat + one tail-line read per stream) but
    # high enough that a sustained audit-stream blackout still lights up
    # Telegram within a single trading session.
    _TRIPWIRE_INTERVAL_SECONDS = 3600.0
    # Hard-coded for now — configurable via TripwireConfig if operators ever
    # want to tune. The default (45 days) matches the production refit cadence
    # target in ``DEFAULT_STALE_CALIBRATOR_DAYS``.
    _TRIPWIRE_STALE_CALIBRATOR_DAYS = 45

    # How often the daily sweep checks whether its hour has arrived.
    _KXRAIN_SWEEP_POLL_SECONDS = 600.0

    def _tripwire_calibrator_paths(self) -> dict[str, Path]:
        """Return the {label: path} dict for ``run_all_tripwires``.

        Filters to existing files so missing artifacts (legitimate "calibrator
        disabled" state) never produce a spurious read warning. The labels
        flow through into ``DriftAlert.code`` so downstream dedup keys stay
        readable. The isotonic entry honors ``weather.isotonic_path`` so the
        staleness check reads the artifact the strategy is actually serving.

        The EMOS entry is included only when ``weather.emos_enabled_cells`` is
        non-empty — i.e. at least one cell actually serves EMOS live. With it
        empty (all-isotonic), ``data/emos/coefficients.json`` is a dormant
        future-re-add option, not a live calibrator, so a staleness alert on it
        false-alarms on a file no trade depends on. Same misdirected-tripwire
        class as ``_tripwire_bias_table_path`` / ``_isotonic_artifact_path``:
        monitor the artifact the live config actually loads, never one keyed off
        mere existence on disk.

        The ``favorite_bias`` entry covers ``strategies.favorite_bias.isotonic_path``
        whenever that strategy is enabled. It is the g(mid) map behind the only
        order-placing strategy, so it is the one calibrator here actually backing
        money — it went unmonitored while the ``isotonic`` entry above covered an
        artifact serving shadow-only ``weather_edge``.
        """
        candidates: dict[str, Path] = {
            "isotonic": self._isotonic_artifact_path(),
        }
        if self._settings.strategies.weather.emos_enabled_cells:
            candidates["emos"] = Path("data/emos/coefficients.json")
        favorite_bias = self._settings.strategies.favorite_bias
        if favorite_bias.enabled and favorite_bias.isotonic_path:
            candidates["favorite_bias"] = Path(favorite_bias.isotonic_path)
        return {label: p for label, p in candidates.items() if p.exists()}

    def _tripwire_bias_table_path(self) -> Path | None:
        """Resolve the *live* GFS bias table for the freshness tripwire.

        Mirrors ``_load_bias_table`` / ``BiasTable.__init__``: monitors the
        artifact the strategy actually loads — the configured
        ``weather.bias_table_path`` (currently the NWS-anchored table) or the
        ERA5 canonical default when the config is ``null``. Returns ``None``
        when bias correction is disabled (nothing is loaded, so there is
        nothing to monitor), which ``run_all_tripwires`` treats as "skip".

        MUST be used instead of hard-coding the ERA5 path: otherwise the
        tripwire watches the rollback fallback unconditionally — false-alarming
        on a non-deployed file while staying blind to the live NWS table aging
        out (the 2026-06-21 misdirected-tripwire fix; same class as the
        2026-05-22 stale-bounds plateau bug ``_isotonic_artifact_path`` solves).
        """
        w = self._settings.strategies.weather
        if not w.bias_correction_enabled:
            return None
        return (
            Path(w.bias_table_path)
            if w.bias_table_path
            else Path("data/gfs_bias/bias_correction.json")
        )

    async def _audit_tripwire_loop(self) -> None:
        """Hourly sweep of audit-log + calibrator staleness tripwires.

        Routes any returned ``DriftAlert``s through ``self._notifier`` at
        the alert's own severity. Per-code dedup prevents the same alert
        from firing every hour — a legitimate refit changes the artifact
        path's content and produces a new alert code, so genuine drift
        re-alerts naturally.

        Never raises — a tripwire bug must not crash the event loop.
        """
        state_dir = Path(self._settings.state_dir)
        config = TripwireConfig(
            stale_calibrator_days=self._TRIPWIRE_STALE_CALIBRATOR_DAYS,
        )
        try:
            while self._running:
                try:
                    iso_path = self._isotonic_artifact_path()
                    alerts = run_all_tripwires(
                        state_dir=state_dir,
                        calibrator_paths=self._tripwire_calibrator_paths(),
                        config=config,
                        calibration_dir=iso_path.parent,
                        isotonic_filename=iso_path.name,
                        bias_table_path=self._tripwire_bias_table_path(),
                        # The resolver returns a path only when correction is
                        # enabled, so a path here always means "expected to
                        # exist" — a missing live table is a CRITICAL config
                        # error, not a silent no-op.
                        bias_table_missing_is_error=True,
                    )
                except Exception:
                    logger.exception("Audit tripwire sweep failed")
                    alerts = []

                await self._dispatch_tripwire_alerts(alerts, state_dir=state_dir)

                try:
                    await asyncio.sleep(self._TRIPWIRE_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            pass

    async def _dispatch_tripwire_alerts(self, alerts: list[DriftAlert], *, state_dir: Path) -> None:
        """Dispatch tripwire alerts, recording only the ones actually delivered.

        The dedup add is conditional on dispatch: recording a SUPPRESSED alert
        as delivered silences it for the process, and (before this fix) the
        unconditional ``persist_seen_code`` call silenced it forever, because
        ``_tripwire_alerted_codes`` is re-seeded from that file at startup.
        Acknowledgement is now written only on actual dispatch.

        The ``logger.warning("Audit tripwire: ...")`` line below is deduped
        separately, via ``_tripwire_logged_codes`` (added unconditionally,
        regardless of dispatch outcome). If it instead shared
        ``_tripwire_alerted_codes``, a code that is suppressed every sweep
        (owner not live, or acked) would never enter that set and would
        re-log the WARNING on every hourly sweep forever — diluting the
        WARNING channel real risk alerts live on. This matches ``main``'s
        pre-suppression behaviour of logging each code once per process.
        """
        for alert in alerts:
            # Recorded for every alert this sweep saw — even one that's about
            # to be skipped as already-dispatched or suppressed below — so
            # /ack works on any code the operator can see in the logs, not
            # only ones that reached Telegram this tick.
            if alert.ack_key is not None:
                self._ackable[alert.code] = (alert.owner, alert.severity.name)
            if alert.code not in self._tripwire_logged_codes:
                logger.warning(
                    "Audit tripwire: %s (code=%s, severity=%s)",
                    alert.message,
                    alert.code,
                    alert.severity.name,
                )
                self._tripwire_logged_codes.add(alert.code)
            if alert.code in self._tripwire_alerted_codes:
                continue
            message = f"[{alert.code}] {alert.message}"
            if alert.ack_key is not None:
                # An ack-able alert must say so in the message itself — the
                # operator otherwise has no way
                # to know /ack is an option for this specific code without
                # already knowing the feature exists.
                message += f"\nreply /ack {alert.code} to silence until this artifact changes"
            dispatched = await self._notify_alert(
                message,
                level=alert.severity,
                owner=alert.owner,
                code=alert.code,
            )
            if dispatched:
                self._tripwire_alerted_codes.add(alert.code)
                if alert.code.startswith(PLATEAU_ALERT_CODE_PREFIX):
                    persist_seen_code(state_dir, alert.code)

    async def _start_telegram(self) -> None:
        """Start the Telegram notifier, handling errors gracefully."""
        if self._notifier is None:
            return
        try:
            await self._notifier.start()
            await self._notifier.send_alert("The Oracle started.")
        except Exception:
            logger.warning(
                "Telegram bot failed to start — continuing without notifications",
                exc_info=True,
            )
            self._notifier = None

    def _check_contract(self, data: dict[str, Any], schema: KalshiSchema, *, context: str) -> None:
        if self._contract_watcher is not None:
            self._contract_watcher.check(data, schema, context=context)

    def _alert_is_relevant(self, owner: str | None) -> bool:
        """True when an alert owned by *owner* should be delivered.

        Fails OPEN: an owner matching no registered strategy is dispatched with
        a warning. A typo or an unregistered name must be noisy, never silent —
        fail-closed here would rebuild the 'silently blind' failure this exists
        to remove.
        """
        if owner is None:
            return True
        for s in self._strategies:
            if s.name == owner:
                return s.configured_to_trade
        logger.warning("alert owner %r matches no registered strategy — firing anyway", owner)
        return True

    def _owner_is_live(self, owner: str | None) -> bool:
        """True when *owner* names a registered strategy that is configured to trade."""
        if owner is None:
            return False
        return any(s.name == owner and s.configured_to_trade for s in self._strategies)

    def _ack_suppresses(self, code: str, level: AlertLevel) -> bool:
        """True when a human verdict silences this alert at this severity.

        Liveness is evaluated HERE, at dispatch, not pruned at startup — the
        relevance rule re-arms automatically when a strategy goes live, and an
        ack recorded while its owner was NOT live must re-arm with it,
        including via ``/enable``, which flips ``Strategy.enabled`` at
        runtime (``enable_strategy``) and would defeat any startup-only prune.

        This is a TRANSITION test, not a point-in-time one:

            acked not-live -> now not-live : suppress (if severity qualifies)
            acked not-live -> now live     : ignore   (judgment made in a
                                              no-money context; must not
                                              silently carry over once real
                                              money is on the line)
            acked live     -> now live     : suppress (if severity qualifies)
                                              — the common case, e.g. acking
                                              a stale-calibrator alert for a
                                              strategy that is live right now
            acked live     -> now not-live : suppress (if severity qualifies)
                                              — strictly safer; the alert
                                              matters less with the owner off

        Only the "acked not-live, now live" cell drops the ack outright. Every
        other cell falls through to the severity-scoped comparison below, so
        an ack recorded at WARNING still never suppresses a later CRITICAL of
        the same code — both age checks escalate WARNING->CRITICAL at 2x
        threshold and the code carries no severity of its own.

        A legacy ack with no recorded ``owner_live_at_ack`` (pre-dating this
        field) defaults to ``False`` — treated as acked while NOT live. That
        is the safe default: it re-arms (rather than silently keeps
        suppressing) the moment its owner is live, instead of assuming a
        stale record still applies to a strategy now trading real money.

        An unparseable recorded severity (missing, or a value not in
        ``_SEV_RANK``) is treated as NO ack at all, i.e. fails OPEN — the
        opposite of defaulting it to ``CRITICAL``, which would silence
        anything up to and including a real CRITICAL. This matters because
        ``AlertLevel`` is a ``StrEnum`` whose VALUES are lowercase
        (``"warning"``/``"critical"``) while its NAMES are uppercase; a
        hand-edited or malformed ack entry (e.g. ``severity: "warning"``)
        must not silently mute real alerts. Likewise an unrecognized
        *incoming* ``level.name`` is ranked at 3 (most severe), not 0 — the
        least suppressible outcome — rather than defaulting to something an
        ordinary ack would cover.
        """
        entry = self._acked.get(code)
        if entry is None:
            return False
        owner_live_at_ack = bool(entry.get("owner_live_at_ack", False))
        owner_live_now = self._owner_is_live(entry.get("owner"))
        if not owner_live_at_ack and owner_live_now:
            return False
        acked_rank = _SEV_RANK.get(str(entry.get("severity", "")))
        if acked_rank is None:
            logger.warning(
                "ack entry for %s has unusable severity %r — ignoring",
                code,
                entry.get("severity"),
            )
            return False
        return _SEV_RANK.get(level.name, 3) <= acked_rank

    async def _notify_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        *,
        owner: str | None = None,
        code: str | None = None,
    ) -> bool:
        """Send a Telegram alert if the notifier is active. Never raises.

        Returns False ONLY when the alert was suppressed as not-relevant.
        A missing notifier (Telegram disabled, or the bot failed to start —
        ``_start_telegram`` sets ``self._notifier = None`` on any startup
        error) is not suppression: it returns True so callers still record
        the alert as handled. Callers MUST use the return value to decide
        whether to record the alert as delivered — recording a suppressed
        alert as delivered silences it permanently.

        ``owner`` and ``code`` are keyword-only: ``ContractWatcher`` calls this
        positionally as ``alert_fn(message, AlertLevel.WARNING)``, so a third
        positional parameter would let a future reorder bind a level into one
        of them.

        ``code`` is passed explicitly rather than parsed back out of the
        formatted ``message``. Recovering it with a string split would make an
        un-bracketed message key a suppression counter — or an acknowledgement —
        on arbitrary prose.
        """
        try:
            if not self._alert_is_relevant(owner):
                self._alert_suppressions[(code or message[:60], owner or "")] += 1
                logger.info("alert suppressed (owner=%s not live): %s", owner, code or "-")
                return False
            if code is not None and self._ack_suppresses(code, level):
                self._alert_suppressions[(code, owner or "")] += 1
                logger.info("alert suppressed (acknowledged): %s", code)
                return False
        except Exception:
            # Fail open — relevance is an optimisation, delivery is the contract.
            logger.warning("alert relevance check failed; dispatching", exc_info=True)
        if self._notifier is not None:
            try:
                await self._notifier.send_alert(message, level=level)
            except Exception:
                logger.debug("Failed to send Telegram alert", exc_info=True)
        return True

    async def _notify_warning_telegram(self, msg: str) -> None:
        """Thin wrapper: forwards *msg* as a WARNING-level Telegram alert."""
        await self._notify_alert(msg, level=AlertLevel.WARNING)

    async def _notify_critical_telegram(self, msg: str) -> None:
        """Thin wrapper: forwards *msg* as a CRITICAL-level Telegram alert."""
        await self._notify_alert(msg, level=AlertLevel.CRITICAL)

    @staticmethod
    def _parse_kalshi_decimal(value: Any, default: float = 0.0) -> float:
        """Parse a Kalshi v2 numeric field that ships as a JSON string.

        Fields like ``position_fp`` and ``market_exposure_dollars`` are
        emitted as decimal-formatted strings (``"-16.00"``, ``"11.000000"``).
        Falls back to *default* when the value is missing or unparseable so
        the caller's zero/empty check filters the row out.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def _fetch_unsettled_positions(self, *, stage: str) -> list[dict[str, Any]] | None:
        """Fetch all unsettled positions with timeout + truncation alerts.

        Returns ``None`` on timeout or fetch failure (caller decides whether
        to proceed with empty positions or skip the tick). Truncation is
        surfaced via Telegram CRITICAL but the partial result is still
        returned so callers can use what they got.
        """
        assert self._kalshi_client is not None
        try:
            positions, truncated = await asyncio.wait_for(
                self._kalshi_client.iter_all_positions(settlement_status="unsettled"),
                timeout=30.0,
            )
        except TimeoutError:
            logger.warning("iter_all_positions timed out after 30s during %s", stage)
            await self._notify_alert(
                f"RECONCILIATION TIMEOUT: /portfolio/positions did not return "
                f"within 30s during {stage}.",
                level=AlertLevel.CRITICAL,
            )
            return None
        except Exception:
            logger.warning("Failed to fetch Kalshi positions during %s", stage, exc_info=True)
            return None

        if truncated:
            logger.critical(
                "Kalshi positions response truncated at MAX_PAGES during %s — "
                "got %d positions before cap; result is INCOMPLETE",
                stage,
                len(positions),
            )
            await self._notify_alert(
                f"RECONCILIATION TRUNCATED: /portfolio/positions hit MAX_PAGES "
                f"during {stage} ({len(positions)} positions loaded before the "
                f"cap; result is INCOMPLETE).",
                level=AlertLevel.CRITICAL,
            )
        return positions

    async def _periodic_reconciliation(self) -> None:
        """Compare local risk state against Kalshi API and alert on drift.

        Equity (cash + open position notional) is the unit of comparison;
        ``risk_manager.bankroll`` carries the same equity semantics, so a
        live position-laden session no longer flags a phantom drift the
        moment Kalshi cash drops below the local figure.
        """
        assert self._risk_manager is not None
        assert self._kalshi_client is not None

        divergences: list[str] = []

        try:
            balance_resp = await self._kalshi_client.get_balance()
            self._check_contract(balance_resp, SCHEMA_BALANCE, context="balance_reconcile")
            api_cash = balance_resp.get("balance", 0) / 100.0
        except Exception:
            logger.warning("Periodic reconciliation: failed to fetch balance", exc_info=True)
            return

        market_positions = await self._fetch_unsettled_positions(stage="periodic check")
        if market_positions is None:
            return

        api_positions: dict[str, float] = {}
        for pos in market_positions:
            ticker = pos.get("ticker", "")
            quantity = self._parse_kalshi_decimal(pos.get("position_fp", "0"))
            dollars = self._parse_kalshi_decimal(pos.get("market_exposure_dollars", "0"))
            if not ticker or quantity == 0:
                continue
            api_positions[ticker] = dollars

        # Repair orphans before the drift loop so it stays silent on
        # now-corrected keys. hydrate_live_positions registers stubs in
        # engine._orders only — risk_manager.positions is unchanged, so the
        # divergence loop below would still alert on these keys without the
        # orphan_tickers skip guard.
        local_positions = self._risk_manager.positions
        orphan_rows: list[LivePosition] = []
        for pos in market_positions:
            ticker = pos.get("ticker", "")
            if not ticker or ticker in local_positions:
                continue
            quantity = self._parse_kalshi_decimal(pos.get("position_fp", "0"))
            dollars = self._parse_kalshi_decimal(pos.get("market_exposure_dollars", "0"))
            if quantity == 0 or dollars <= 0:
                continue
            orphan_rows.append(
                LivePosition(
                    position_key=ticker,
                    market_ticker=ticker,
                    side=Side.YES if quantity > 0 else Side.NO,
                    dollars=dollars,
                )
            )

        orphan_tickers: set[str] = set()
        if orphan_rows and self._engine is not None:
            n = self._engine.hydrate_live_positions(orphan_rows)
            orphan_tickers = {row.position_key for row in orphan_rows}
            logger.warning("PERIODIC HYDRATE: repaired %d orphan position(s)", n)
            self._engine.restore_fills_from_log()

        api_position_notional = sum(api_positions.values())
        api_equity = api_cash + api_position_notional
        local_equity = self._risk_manager.bankroll
        equity_drift = abs(api_equity - local_equity)
        threshold_bankroll = self._settings.execution.reconcile_bankroll_threshold
        if equity_drift > threshold_bankroll:
            divergences.append(
                f"Equity drift: API=${api_equity:,.2f} (cash=${api_cash:,.2f} + "
                f"positions=${api_position_notional:,.2f}) vs "
                f"local=${local_equity:,.2f} (delta=${equity_drift:,.2f})"
            )

        all_keys = set(api_positions.keys()) | set(local_positions.keys())
        threshold_pos = self._settings.execution.reconcile_position_threshold

        for key in all_keys:
            if key in orphan_tickers:
                # Already alerted via PERIODIC HYDRATE above.
                continue
            api_val = api_positions.get(key, 0.0)
            local_val = local_positions.get(key, 0.0)
            drift = abs(api_val - local_val)
            if drift > threshold_pos:
                divergences.append(
                    f"Position {key}: API=${api_val:.2f} vs local=${local_val:.2f} "
                    f"(delta=${drift:.2f})"
                )

        if divergences:
            # Drift auto-corrects below, so there is no operator action — the
            # per-divergence WARNING logs are the audit trail; no push alert.
            for d in divergences:
                logger.warning("Reconciliation divergence: %s", d)
            # Re-sync risk manager to API state so the same drift does not
            # fire again next cycle. Mirrors _reconcile_from_kalshi() which
            # always calls hydrate_positions() at startup.
            position_strategy = build_position_strategy_index(
                Path(self._settings.state_dir) / "trades.jsonl"
            )
            self._risk_manager.hydrate_positions(
                api_positions, api_equity, position_strategy=position_strategy
            )
        else:
            logger.info(
                "Reconciliation OK: equity=$%.2f (cash=$%.2f + positions=$%.2f), %d positions",
                api_equity,
                api_cash,
                api_position_notional,
                len(api_positions),
            )

    def _ws_stream_stale_seconds(self) -> float | None:
        """Seconds the WS data stream has been silent past the STALE_DATA cutoff,
        or None if fresh.

        Pong-immune: driven by last data-frame receipt, not the heartbeat. Also
        covers the cold-start case (connected/subscribed but no data frame ever
        received) via process uptime, which ``last_message_age``'s 0.0 sentinel
        cannot express.
        """
        ws = self._kalshi_ws
        if ws is None:
            return None
        cutoff = self._settings.risk.stale_data_cutoff_seconds
        if ws.received_any_message:
            age = ws.last_message_age
            return age if age >= cutoff else None
        uptime = time.monotonic() - self._startup_monotonic
        return uptime if uptime >= cutoff else None

    async def _check_ws_health(self) -> None:
        """
        Verify the WebSocket background task is alive; restart it if not.

        Called at the start of every ``_tick``.  Tracks consecutive restart
        attempts and logs CRITICAL when the count exceeds ``_WS_MAX_RESTARTS``
        so an operator can investigate a persistent connection failure.
        """
        if self._kalshi_ws is None:
            # No WebSocket configured — nothing to monitor.
            return

        task = self._ws_task
        if task is not None and not task.done():
            # Task is alive, but the stream can be silently data-quiet — either a
            # half-open stall (TCP pong-alive, no data frames) OR a legitimately
            # quiet market (nothing to tick). Classify into harm / pipe-down /
            # silent-but-healthy and page only on the first two. Wrapped
            # exception-safe: this runs inside _tick, and an unhandled raise here
            # feeds _consecutive_tick_failures -> kill switch at 20. Degrade-
            # never-raise.
            try:
                stale = self._ws_stream_stale_seconds()
                if stale is None:
                    # Fresh — healthy. Reset counters.
                    self._ws_restart_count = 0
                    self._ws_last_alerted_count = 0
                    return

                cutoff = self._settings.risk.stale_data_cutoff_seconds
                harm = self._risk_manager is not None and (
                    self._risk_manager.stale_data_rejected_within(cutoff)
                )
                sub_age = self._kalshi_ws.last_subscribe_ok_age
                window = _pipe_health_window(
                    self._settings.execution.websocket_idle_timeout_seconds, cutoff
                )
                # None == never subscribed (cold-start July-14 case) == pipe-down.
                pipe_down = sub_age is None or sub_age >= window

                if harm or pipe_down:
                    # DEGRADED (July-14 class): page + (respawn iff pipe is down).
                    self._ws_restart_count += 1
                    if pipe_down:
                        sub_txt = "never" if sub_age is None else f"{sub_age:.0f}s ago"
                        reason = f"pipe not re-subscribing (last ok {sub_txt})"
                    else:
                        reason = "live signal(s) rejected as STALE_DATA"
                    logger.warning(
                        "WebSocket data-silent %.0fs — %s — forcing reconnect (restart #%d)",
                        stale,
                        reason,
                        self._ws_restart_count,
                    )
                    now = time.monotonic()
                    if now - self._ws_last_critical_alert_monotonic >= cutoff:
                        await self._notify_alert(
                            f"WebSocket data-silent {stale:.0f}s — {reason}. "
                            f"Forcing reconnect (#{self._ws_restart_count}).",
                            level=AlertLevel.CRITICAL,
                        )
                        self._ws_last_critical_alert_monotonic = now
                    if pipe_down:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await task
                        self._ws_task = asyncio.create_task(self._ws_listener(), name="kalshi-ws")
                    return

                # SILENT-BUT-HEALTHY: pipe fine, no harm (e.g. quiet market).
                # Inner listen() loop owns reconnects — do NOT cancel/respawn or
                # page. Fact-based INFO log, throttled. Reset alert counters.
                self._ws_restart_count = 0
                self._ws_last_alerted_count = 0
                now = time.monotonic()
                if now - self._ws_last_silent_info_monotonic >= self._WS_SILENT_INFO_LOG_INTERVAL_S:
                    logger.info(
                        "WebSocket data-silent %.0fs; pipe healthy (last resubscribe "
                        "%.0fs ago), 0 STALE_DATA rejections in last %ds",
                        stale,
                        sub_age,
                        cutoff,
                    )
                    self._ws_last_silent_info_monotonic = now
                return
            except Exception:
                logger.exception("WebSocket health check errored — skipping this tick (no restart)")
                return

        # Task is missing or has exited — log the reason and restart.
        if task is not None:
            exc = None
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                pass  # Task was cancelled deliberately (e.g. during shutdown).

            if exc is not None:
                logger.error(
                    "WebSocket listener exited with exception (restart #%d): %s",
                    self._ws_restart_count + 1,
                    exc,
                    exc_info=exc,
                )
            else:
                logger.warning(
                    "WebSocket listener exited cleanly (restart #%d) — restarting",
                    self._ws_restart_count + 1,
                )

        self._ws_restart_count += 1

        # A single transient disconnect self-heals on the next tick (still
        # logged above) — only push once failures persist past the threshold.
        if (
            self._ws_restart_count >= self._WS_ALERT_THRESHOLD
            and self._ws_restart_count > self._ws_last_alerted_count
        ):
            await self._notify_alert(
                f"WebSocket failed {self._ws_restart_count} consecutive times. "
                f"Real-time data may be stale.",
                level=AlertLevel.CRITICAL,
            )
            self._ws_last_alerted_count = self._ws_restart_count

        if self._ws_restart_count > self._WS_MAX_RESTARTS:
            logger.critical(
                "WebSocket listener has failed %d consecutive times — "
                "real-time data unavailable; stale data checks will eventually halt trading",
                self._ws_restart_count,
            )

        logger.info("Restarting WebSocket listener (attempt %d)", self._ws_restart_count)
        self._ws_task = asyncio.create_task(self._ws_listener(), name="kalshi-ws")

    async def _ws_listener(self) -> None:
        """Background task that consumes WebSocket messages.

        We subscribe ONLY to the ``ticker`` channel because Kalshi requires
        a non-empty ``market_tickers`` filter for ``orderbook_delta`` (rejects
        a global subscribe with ``code=2 'Params required'``) and we have no
        plumbing to discover the right ticker set at subscribe time. The
        ``ticker`` channel accepts a global subscribe and is sufficient for
        the only thing this listener actually drives: bumping
        ``RiskManager.update_data_freshness()`` so the live-mode STALE_DATA
        gate stays satisfied. Strategies do not consume WS data today —
        they re-fetch via REST in their scan loops.

        If a future strategy needs orderbook_delta streams, refactor the
        subscribe path to take a ticker list (sourced from
        ``iter_all_markets(status="open")`` at startup, refreshed
        periodically) and add it to the ``listen()`` call below.
        """
        if self._kalshi_ws is None or self._risk_manager is None:
            return

        try:
            async for message in self._kalshi_ws.listen(channels=["ticker"]):
                self._risk_manager.update_data_freshness()

                msg_type = message.get("type", "")
                logger.debug("WS message: %s", msg_type)

                # Strategies and other consumers can register callbacks on
                # the KalshiWebSocket instance directly.

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("WebSocket listener error")

    async def shutdown(self) -> None:
        """Gracefully shut down all components."""
        _sd_notify("STOPPING=1")
        logger.info("Shutting down orchestrator...")
        self._running = False

        if self._hc_client is not None:
            await self._hc_client.aclose()

        # Stop calibration monitor before Telegram (its alerts target the notifier)
        if self._calibration_monitor is not None:
            try:
                await self._calibration_monitor.stop()
            except Exception:
                logger.warning("Calibration monitor stop failed", exc_info=True)

        # Stop the independent watchdog heartbeat. systemd's STOPPING=1 (sent
        # at the top of shutdown) already tells systemd a clean stop is in
        # progress, so dropping the keepalive here won't trip the watchdog.
        await _cancel_task(self._watchdog_task)

        # Stop the audit-tripwire periodic sweep before tearing down notifier.
        await _cancel_task(self._tripwire_task)

        # Stop the Phase 0 depth monitor (no Telegram alerts to drain).
        await _cancel_task(self._phase0_depth_task)

        # Before notifier teardown: the observer can alert while cancelling.
        try:
            await _cancel_task(self._kxrain_sweep_task)
            await _cancel_task(self._kxrain_task)
        except Exception:
            # Must never block save_state and cancel_all_open below.
            logger.exception("KXRAIN task cancellation failed during shutdown")
        if self._kxrain_observer is not None:
            try:
                await self._kxrain_observer.aclose()
            except Exception:
                logger.exception("KXRAIN observer aclose failed during shutdown")

        # Stop Telegram bot first (while other services are still up)
        if self._notifier is not None:
            try:
                await self._notifier.stop()
            except Exception:
                logger.warning("Telegram bot stop failed", exc_info=True)
        await _cancel_task(self._telegram_task)

        if self._risk_manager is not None:
            ok = await save_state(Path(self._settings.state_dir), self._build_state_snapshot())
            if ok:
                logger.info("Final risk state persisted")
            else:
                # Don't try to fire a Telegram alert here — _notifier may
                # already be torn down by this point in shutdown(). The
                # error is logged inside save_state.
                logger.error("Final risk state persistence FAILED — check logs")

        await _cancel_task(self._ws_task)

        if self._kalshi_ws is not None:
            await self._kalshi_ws.close()

        # Cancel any resting orders before tearing down the Kalshi REST
        # client so `systemctl stop` / `systemctl restart` doesn't leave
        # orders to fill during the deploy window. WS is already closed at
        # this point, so no concurrent fill notifications can race the
        # cancel loop. cancel_all_open has its own internal timeout; no
        # outer wait_for needed.
        if self._engine is not None and self._kalshi_client is not None:
            try:
                cancelled, truncated = await self._engine.cancel_all_open()
                logger.info(
                    "Shutdown cancelled %d resting orders (truncated=%s)",
                    len(cancelled),
                    truncated,
                )
                if truncated:
                    logger.error(
                        "Shutdown order-cancel TRUNCATED — additional resting "
                        "orders may remain; manually verify at "
                        "https://kalshi.com/portfolio/orders"
                    )
            except Exception:
                logger.exception(
                    "Shutdown order-cancel failed; resting orders will expire "
                    "at server-side TTL (~120s)"
                )

        if self._kalshi_client is not None:
            await self._kalshi_client.close()

        # Release per-strategy resources (NWS clients, etc.) before the
        # shared weather/HTTP clients are torn down.
        await self._run_strategy_lifecycle_hook("aclose")

        if self._weather_client is not None:
            await self._weather_client.close()

        # Release the sidecar pidlock so in-process teardown is deterministic
        # (kernel releases on process exit anyway, but this matters for tests
        # that construct multiple Orchestrators in one process).
        if self._trade_log_pidlock_fd is not None:
            try:
                os.close(self._trade_log_pidlock_fd)
            except OSError:
                logger.warning("Failed to close trade-log pidlock FD", exc_info=True)
            self._trade_log_pidlock_fd = None

        logger.info("Orchestrator shut down")

    async def _run_strategy_lifecycle_hook(self, hook: str) -> None:
        """Await the named lifecycle hook on every registered strategy.

        ``hook`` is ``"setup"`` or ``"aclose"``. Errors are logged and
        skipped — a strategy that fails to set up or tear down must not
        take the orchestrator with it. For WeatherEdgeStrategy a failed
        setup leaves the disabled SameDayGate sentinel in place, which
        never blocks anything that wasn't already blocked.
        """
        for strategy in self._strategies:
            try:
                await getattr(strategy, hook)()
            except Exception:
                logger.exception("Strategy %s %s() failed", strategy.name, hook)

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            logger.info("Received shutdown signal")
            self._request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)


async def main() -> None:
    """Launch the orchestrator."""
    settings = Settings.load()
    setup_logging(settings.logging)

    watchdog_usec = os.environ.get("WATCHDOG_USEC")
    if watchdog_usec:
        logger.info(
            "systemd watchdog active: WATCHDOG_USEC=%s (%.0fs)",
            watchdog_usec,
            int(watchdog_usec) / 1_000_000,
        )

    orchestrator = Orchestrator(settings)
    await orchestrator.initialize()
    _sd_notify("READY=1")
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
