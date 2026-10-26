"""Telegram bot integration for The Oracle.

Provides two-way communication:
  - Command handlers: owner sends /status, /kill, etc.
  - Push alerts: bot sends trade fills, risk events, errors.

Security:
  - Owner-only: every handler checks chat_id == owner_chat_id.
  - Messages from unknown chat IDs are silently ignored.
  - No secrets or internal IDs appear in messages.
  - Long polling (outbound HTTPS only) — no inbound ports needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from python.analytics.audit_tripwires import persist_ack_code
from python.notifications.base import AlertLevel
from python.notifications.formatters import (
    format_acks,
    format_attribution,
    format_calibration_health,
    format_emos_writer_counters,
    format_help,
    format_positions,
    format_risk,
    format_risk_alert,
    format_signals,
    format_status,
    format_trade_alert,
    format_trades,
)
from python.utils.log_config import get_log_buffer

if TYPE_CHECKING:
    from python.analytics.attribution import AttributionEngine
    from python.analytics.calibration_monitor import CalibrationMonitor
    from python.data.kalshi_client import KalshiClient
    from python.execution.engine import ExecutionEngine, OrderRecord
    from python.execution.fills_reconciler import FillsReconciler
    from python.orchestrator import Orchestrator
    from python.risk.manager import RiskManager
    from python.settings import Settings
    from python.strategies.base import Strategy

logger = logging.getLogger(__name__)

# Telegram enforces 4096-char limit per message.
_MAX_MESSAGE_LEN = 4096

# /loglevel: only debug and info are accepted — warning/error would silence
# risk manager breach alerts and leave the /agent ring buffer near-empty.
_LOG_LEVEL_ALLOWED = {"debug": logging.DEBUG, "info": logging.INFO}
_LOG_DURATION_RE = re.compile(r"^([1-9]|[12]\d|30)m$")
_LOG_REVERT_STOP_TIMEOUT_S = 2.0

_DIAGNOSE_SYSTEM_PROMPT = """\
You are diagnosing The Oracle, the Kalshi prediction-market trading bot that runs from \
this directory. CLAUDE.md describes its architecture; CLAUDE.local.md, when present, \
describes this deployment's current live state. python/config.yaml holds the thresholds \
and limits (edge thresholds, market filters, risk caps, circuit breakers, which strategies \
are enabled), and the file named by ORACLE_CONFIG_OVERLAY in .env, when set, overrides \
parts of it; `.venv/bin/python scripts/check_live_config.py --dump` prints the merged \
result. These values change often, so read them for any number you cite instead of \
assuming one.

Below are the operator's question, a snapshot of the running bot's state (positions, \
P&L, strategy status, recent signals) and its most recent log lines. Files under state/ \
hold longer history if the snapshot is not enough.

The usual reasons for no signals or trades: every market filtered out (volume, open \
interest, spread, book depth, max edge) or below the edge threshold (the model agrees \
with the market), off-hours or between settlement windows, a \
risk limit reached (exposure cap, daily loss halt, drawdown reduction), a tripped circuit \
breaker, or the kill switch (state/STOP). The usual operational failures: Kalshi 429s (an \
undocumented sliding window trips after ~100 requests, then a 5-10 minute cooldown), \
WebSocket disconnects (auto-reconnect, logged as WARNING) and transient Open-Meteo timeouts.

Your reply is relayed verbatim to the operator's Telegram chat as plain text (Markdown is \
not rendered) and is usually read on a phone. Lead with the most likely root cause and the \
evidence for it, then what to do next."""


# C-ENV-INHERIT: explicit allowlist of env vars passed to the /agent
# subprocess. Claude Code runs with --dangerously-skip-permissions and 100
# turns; inheriting os.environ would let the subprocess read trading secrets.
# OAuth still works because HOME=/opt/oracle resolves Claude Code's
# default credentials path (/opt/oracle/.claude/.credentials.json).
_DIAGNOSE_ENV_ALLOWLIST: tuple[str, ...] = (
    # POSIX baseline — required for any subshell / file resolution.
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    # Locale — required for terminal output, json decoding paths.
    "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
    # Claude Code config — read by the CLI to find user config dir.
    "CLAUDE_CONFIG_DIR",
    # Optional OAuth token override. If the operator sets this in .env,
    # they ACCEPT it as a leaking secret to the agent subprocess; this
    # is the only inheritable secret in the post-fix allowlist. Typical
    # operator path is file-based via HOME, so this entry is benign-when-unset.
    "CLAUDE_CODE_OAUTH_TOKEN",
)

# /agent diagnose subprocess configuration. Kept as module constants so the
# CLI invocation and the user-facing timeout message stay in sync.
_DIAGNOSE_MODEL = "claude-opus-4-7"
_DIAGNOSE_MAX_TURNS = 100
_DIAGNOSE_TIMEOUT_S = 900.0


def _loglevel_usage(reason: str | None) -> str:
    header = f"Error: {reason}\n" if reason else ""
    return (
        f"{header}Usage:\n"
        "  /loglevel                - show current level\n"
        "  /loglevel debug 10m      - debug for N minutes (1m..30m)\n"
        "  /loglevel info           - revert to baseline"
    )


def _owner_only(
    handler: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator that silently ignores messages from non-owner chat IDs."""

    @wraps(handler)
    async def wrapper(
        self: TelegramNotifier, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.effective_chat is None or update.effective_chat.id != self._chat_id:
            logger.warning(
                "Unauthorized Telegram access from chat_id=%s",
                update.effective_chat.id if update.effective_chat else "unknown",
            )
            return
        await handler(self, update, context)

    return wrapper


def _reason_from_args(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Join /kill /unkill command args into an optional reason string.

    Defensive: only a real list/tuple of args becomes a reason. Test/MagicMock
    contexts (where ``context.args`` is a Mock, not a list) yield None rather than
    raising ``" ".join(Mock)`` — the audit write must never break the kill path.
    """
    args = getattr(context, "args", None)
    if isinstance(args, (list, tuple)) and args:
        return " ".join(str(a) for a in args)
    return None


class TelegramNotifier:
    """Telegram notification backend for The Oracle.

    Lifecycle:
      1. Orchestrator creates this after all subsystems are initialized.
      2. ``start()`` launches long-polling in the background.
      3. ``stop()`` sends a shutdown message and stops polling.

    The notifier holds references to subsystem objects for read-only
    queries.  Write operations (kill switch, enable strategy) go through
    callbacks to avoid circular dependencies.
    """

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
        strategies: list[Strategy],
        *,
        kill_callback: Callable[[], Any],
        unkill_callback: Callable[[], Any],
        enable_strategy_callback: Callable[[str], str],
        attribution_engine: AttributionEngine | None = None,
        calibration_monitor: CalibrationMonitor | None = None,
        fills_reconciler: FillsReconciler | None = None,
        kalshi_client: KalshiClient | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        self._settings = settings
        self._risk = risk_manager
        self._engine = execution_engine
        self._strategies = strategies

        # Callbacks for write operations (avoid circular import of Orchestrator)
        self._kill_callback = kill_callback
        self._unkill_callback = unkill_callback
        self._enable_strategy_callback = enable_strategy_callback

        self._attribution = attribution_engine
        self._monitor = calibration_monitor
        self._fills_reconciler = fills_reconciler
        # Kalshi client is optional — passed only in live/paper mode by the
        # orchestrator. ``_cmd_status`` uses it (LIVE only) to compute a
        # mark-to-market portfolio snapshot alongside cost-basis bankroll.
        self._kalshi_client = kalshi_client
        # ``_cmd_ack``/``_cmd_acks`` read ``_acked``/``_ackable``/
        # ``_alert_suppressions`` and call ``_owner_is_live`` on this directly
        # — see python/orchestrator.py's ``_ack_suppresses`` for the rule.
        self._orchestrator = orchestrator

        self._chat_id: int = settings.telegram_owner_chat_id
        self._application: Application | None = None  # type: ignore[type-arg]

        # Rate limiting
        self._rate_limit = settings.telegram.max_messages_per_minute
        self._message_timestamps: list[float] = []

        # ERROR-level routing state (per-bucket dedup + 6/hour global cap).
        # See `_error_should_send` and `send_alert` for semantics.
        self._error_dedup: dict[tuple[str, int], None] = {}
        self._error_send_count_this_hour: int = 0
        self._error_suppressed_count_this_hour: int = 0
        self._error_count_hour: int | None = None
        self._error_pending_summary: int = 0

        # /loglevel state. Baselines captured here and re-synced in start() so
        # revert always restores the configured level, not the last-observed one.
        # NB: coupled with settings.logging — a future hot-reload PR must refresh these.
        self._original_log_level: int = logging.INFO
        self._original_module_levels: dict[str, int] = {}
        self._log_revert_task: asyncio.Task[None] | None = None
        self._log_revert_at: datetime | None = None
        self._capture_log_baseline()

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Build the Application and start long-polling."""
        token = self._settings.telegram_bot_token
        if not token:
            logger.error("Telegram bot token is empty — cannot start")
            return

        self._application = (
            Application.builder()
            .token(token)
            .concurrent_updates(False)  # single-threaded, saves RAM on CX22
            .build()
        )

        # Re-sync log-level baselines now that setup_logging() has run.
        self._capture_log_baseline()

        handlers = [
            ("status", self._cmd_status),
            ("positions", self._cmd_positions),
            ("trades", self._cmd_trades),
            ("signals", self._cmd_signals),
            ("risk", self._cmd_risk),
            ("attribution", self._cmd_attribution),
            ("calibhealth", self._cmd_calibration_health),
            ("ack", self._cmd_ack),
            ("acks", self._cmd_acks),
            ("kill", self._cmd_kill),
            ("unkill", self._cmd_unkill),
            ("disable_reconciler", self._cmd_disable_reconciler),
            ("enable_reconciler", self._cmd_enable_reconciler),
            ("enable", self._cmd_enable),
            ("setexposure", self._cmd_set_exposure),
            ("resetexposure", self._cmd_reset_exposure),
            ("loglevel", self._cmd_loglevel),
            ("agent", self._cmd_diagnose),
            ("help", self._cmd_help),
            ("start", self._cmd_help),  # Telegram shows /start on first open
        ]
        for name, callback in handlers:
            self._application.add_handler(CommandHandler(name, callback))

        await self._application.initialize()
        await self._application.start()
        if self._application.updater is not None:
            await self._application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message"],
            )
        logger.info("Telegram bot started (long-polling)")

    async def stop(self) -> None:
        """Stop polling and shut down the Application."""
        if self._application is None:
            return

        # Revert first so a hung sleep can't block shutdown.
        try:
            await self._cancel_log_revert_task()
            self._restore_log_baseline(reason="shutdown")
        except Exception:
            logger.warning("log-level revert cleanup failed", exc_info=True)

        try:
            await self.send_alert("The Oracle shutting down.", level=AlertLevel.WARNING)
        except Exception:
            pass  # best-effort farewell

        try:
            if self._application.updater is not None:
                await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
        except Exception:
            logger.warning("Error during Telegram bot shutdown", exc_info=True)

        logger.info("Telegram bot stopped")

    # -- Push alerts ---------------------------------------------------------

    def _utc_hour(self) -> int:
        """Return a monotonically-increasing integer keyed on the current UTC hour.

        Test seam — overridden in fixtures to freeze the clock.
        """
        now = datetime.now(UTC)
        return now.year * 1_000_000 + now.month * 10_000 + now.day * 100 + now.hour

    def _error_should_send(self, failure_bucket: str | None) -> bool:
        """Return True if this ERROR alert should reach Telegram.

        Implements:
          - Per-(bucket, utc_hour) dedup — repeats in the same hour are dropped.
          - Global cap of 6 ERROR sends per UTC hour. Past the cap, increments
            ``_error_suppressed_count_this_hour`` and returns False.
          - Hour-rollover: when the UTC hour ticks over, if anything was
            suppressed last hour, sets ``_error_pending_summary`` so the
            caller (``send_alert``) can drain it on the same call. Counters
            reset; dedup keys older than the current hour are pruned.
        """
        hour = self._utc_hour()
        if self._error_count_hour is None:
            self._error_count_hour = hour

        # Hour-rollover: emit summary if we suppressed last hour, then reset.
        if hour != self._error_count_hour:
            if self._error_suppressed_count_this_hour > 0:
                self._error_pending_summary = self._error_suppressed_count_this_hour
            self._error_send_count_this_hour = 0
            self._error_suppressed_count_this_hour = 0
            self._error_count_hour = hour
            # Prune dedup keys whose hour is older than current.
            self._error_dedup = {
                k: v for k, v in self._error_dedup.items() if k[1] >= hour
            }

        bucket = failure_bucket or "_global"
        key = (bucket, hour)

        if key in self._error_dedup:
            return False

        if self._error_send_count_this_hour >= 6:
            self._error_suppressed_count_this_hour += 1
            return False

        self._error_dedup[key] = None
        self._error_send_count_this_hour += 1
        return True

    async def send_alert(
        self,
        message: str,
        *,
        level: AlertLevel = AlertLevel.INFO,
        failure_bucket: str | None = None,
    ) -> None:
        """Send a text message to the owner.

        Routing:
          - CRITICAL: always sent, bypasses rate limit.
          - ERROR: per-(failure_bucket, utc_hour) dedup with a 6/hour global
            cap. Overflow increments a suppressed counter; on hour rollover,
            a `[ERROR-SUMMARY]` message fires alongside the next ERROR.
          - WARNING / INFO: subject to the standard token-bucket rate limit.
        """
        if level == AlertLevel.CRITICAL:
            await self._safe_send(f"[CRITICAL] {message}")
        elif level == AlertLevel.ERROR:
            if self._error_should_send(failure_bucket):
                # Drain pending summary AFTER should_send (which may have just
                # set it on hour rollover). Summary precedes the new alert.
                # ``getattr`` keeps test fixtures that bypass __init__ working.
                pending = getattr(self, "_error_pending_summary", 0)
                if pending:
                    self._error_pending_summary = 0
                    await self._safe_send(
                        f"[ERROR-SUMMARY] {pending} EMOS ERROR alerts "
                        f"suppressed in the prior hour"
                    )
                await self._safe_send(f"[ERROR] {message}")
        elif level == AlertLevel.WARNING:
            if self._rate_limit_ok():
                await self._safe_send(f"[WARNING] {message}")
            else:
                logger.warning(
                    "telegram_rate_limit_dropped level=WARNING bucket=%s msg=%r",
                    failure_bucket or "_global", message[:200],
                )
        else:
            if self._rate_limit_ok():
                await self._safe_send(message)
            else:
                logger.info(
                    "telegram_rate_limit_dropped level=INFO bucket=%s msg=%r",
                    failure_bucket or "_global", message[:200],
                )

    async def send_trade_alert(self, order_record: OrderRecord) -> bool:
        """Push notification for a trade execution.

        Returns True iff the message was actually sent (passed the rate limit
        and reached the Telegram API without error). Callers that dedup on a
        confirmed send (fill alerts) must not treat a rate-limit / send-failure
        drop as success.
        """
        if not self._rate_limit_ok():
            return False
        text = format_trade_alert(order_record)
        return await self._safe_send(text)

    async def send_risk_alert(self, event: str, details: str) -> None:
        """Push notification for a risk event. Always sent (critical)."""
        text = format_risk_alert(event, details)
        await self._safe_send(f"[CRITICAL] {text}")

    async def send_trade_batch_alert(self, records: list[OrderRecord]) -> list[OrderRecord]:
        """Send a single summary for multiple trades in one tick.

        Returns the records actually announced (empty if none passed the fill
        filter or the send was dropped by the rate limiter / API error). Callers
        that dedup fill alerts stamp only the returned records, so a dropped
        alert is retried rather than silently marked seen.

        A fill is any FILLED/SHADOW/PARTIAL record, plus an EXPIRED/CANCELLED
        record with fill_count > 0 (a partial position acquired before the
        remainder cancelled — e.g. the TTL cancel-race window). REJECTED/SKIPPED/
        ERROR and zero-fill terminals are dropped.
        """
        from python.common import OrderStatus

        fills = [
            r
            for r in records
            if r.status in (OrderStatus.FILLED, OrderStatus.SHADOW, OrderStatus.PARTIAL)
            or (r.status in (OrderStatus.EXPIRED, OrderStatus.CANCELLED) and r.fill_count > 0)
        ]
        if not fills:
            return []
        if len(fills) == 1:
            return fills if await self.send_trade_alert(fills[0]) else []

        if not self._rate_limit_ok():
            return []
        lines = [f"TRADES EXECUTED ({len(fills)})"]
        for r in fills:
            sig = r.signal
            price = r.fill_price if r.fill_price is not None else r.order_price
            lines.append(
                f"  {sig.side.value} {sig.market_ticker} "
                f"{r.fill_count}x @ ${price:.2f} | {sig.strategy_name}"
            )
        return fills if await self._safe_send("\n".join(lines)) else []

    # -- Command handlers (owner-only) ---------------------------------------

    @_owner_only
    async def _cmd_diagnose(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """AI-assisted diagnosis via Claude Code CLI subprocess."""
        claude_bin = self._settings.claude_code_bin or shutil.which("claude")
        if claude_bin is None:
            await self._reply(update, "claude CLI not found on PATH. Set ORACLE_CLAUDE_CODE_BIN.")
            return


        query = (
            " ".join(context.args)
            if context.args
            else (
                "What is the bot doing right now? Are there any issues? "
                "Why might it not be generating signals or trades?"
            )
        )
        await self._reply(update, f"Analyzing: {query}")

        # Build context from live objects + in-memory log buffer
        sections = [
            format_status(self._risk, self._engine, self._settings, self._strategies),
            format_signals(self._engine, limit=20),
            format_risk(self._risk, self._settings),
            format_positions(self._risk, self._engine),
        ]

        # EMOS side-stream writer counters — visible only when the writer
        # has been attached (PR-2 task C7). ``format_emos_writer_counters``
        # uses ``getattr`` so this is safe before C5/C7 land.
        writer_lines: list[str] = []
        for strat in self._strategies:
            line = format_emos_writer_counters(strat)
            if line is not None:
                writer_lines.append(f"  {strat.name}: {line}")
        if writer_lines:
            sections.append("EMOS WRITERS\n" + "\n".join(writer_lines))

        structured = "\n\n".join(sections)

        log_lines = get_log_buffer().get_lines(150)
        log_section = "\n".join(log_lines) if log_lines else "(no log lines captured)"

        prompt = (
            f"{_DIAGNOSE_SYSTEM_PROMPT}\n\n"
            f"User query: {query}\n\n"
            f"=== BOT STATE ===\n{structured}\n\n"
            f"=== RECENT LOGS (last {len(log_lines)} lines) ===\n{log_section}"
        )

        # C-ENV-INHERIT: deny ORACLE_* secrets to the /agent subprocess.
        env = {k: v for k, v in os.environ.items() if k in _DIAGNOSE_ENV_ALLOWLIST}

        # Timeout prevents a hung subprocess from blocking all command processing
        # (concurrent_updates=False means one handler at a time).
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                claude_bin,
                "-p", prompt,
                "--output-format", "text",
                "--model", _DIAGNOSE_MODEL,
                "--max-turns", str(_DIAGNOSE_MAX_TURNS),
                "--dangerously-skip-permissions",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd="/opt/oracle",
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_DIAGNOSE_TIMEOUT_S
            )
            if proc.returncode == 0:
                analysis = stdout.decode().strip() or "(empty response)"
            else:
                err_msg = stderr.decode().strip() or f"exit code {proc.returncode}"
                logger.warning("Diagnose: claude exited with %d: %s", proc.returncode, err_msg)
                analysis = f"Claude Code failed: {err_msg}"
        except TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("Diagnose: killed process did not exit within 5s")
            analysis = (
                f"Claude Code timed out ({int(_DIAGNOSE_TIMEOUT_S // 60)} min). "
                "Try again or check auth."
            )
        except Exception as exc:
            logger.warning("Diagnose: failed to run claude CLI", exc_info=True)
            analysis = f"Failed to run claude CLI: {type(exc).__name__}: {exc}"

        await self._reply(update, analysis)

    @_owner_only
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from python.analytics.mtm import compute_portfolio_mtm
        from python.common import ExecutionMode

        mtm = None
        mtm_attempted = False
        # Gate to LIVE only — paper hits sandbox (positions ≠ tracked state);
        # shadow uses ShadowBroker (no Kalshi calls).
        if (
            self._engine.mode == ExecutionMode.LIVE
            and self._kalshi_client is not None
        ):
            mtm_attempted = True
            try:
                # 30s timeout; longer than typical Kalshi response, shorter
                # than the worst-case 429 backoff (120s). On rate-limit
                # cooldown the operator gets "MTM: unavailable" within 30s
                # rather than a 2-minute hang.
                mtm = await asyncio.wait_for(
                    compute_portfolio_mtm(self._kalshi_client),
                    timeout=30.0,
                )
            except Exception:
                logger.warning(
                    "compute_portfolio_mtm failed or timed out", exc_info=True
                )
                mtm = None

        text = format_status(
            self._risk,
            self._engine,
            self._settings,
            self._strategies,
            mtm=mtm,
            mtm_attempted=mtm_attempted,
        )
        await self._reply(update, text)

    @_owner_only
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = format_positions(self._risk, self._engine)
        await self._reply(update, text)

    @_owner_only
    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = format_trades(self._engine)
        await self._reply(update, text)

    @_owner_only
    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = format_signals(self._engine)
        await self._reply(update, text)

    @_owner_only
    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = format_risk(self._risk, self._settings)
        await self._reply(update, text)

    @_owner_only
    async def _cmd_attribution(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if self._attribution is None:
            await self._reply(update, "Attribution engine not configured.")
            return

        from datetime import UTC, datetime, timedelta

        arg = (context.args[0].lower() if context.args else "7d")
        now = datetime.now(UTC)
        if arg == "all":
            since = None
            label = "all"
        elif arg == "30d":
            since = now - timedelta(days=30)
            label = "30d"
        else:
            since = now - timedelta(days=7)
            label = "7d"

        try:
            report = await asyncio.to_thread(self._attribution.report, since=since)
        except Exception:
            logger.warning("attribution: report failed", exc_info=True)
            await self._reply(update, "Attribution failed — check logs.")
            return

        text = format_attribution(
            window_label=label,
            summary=report["overall"],
            by_side=report["by_side"],
            by_city=report["by_city"],
        )
        await self._reply(update, text)

    async def _require_monitor(self, update: Update) -> bool:
        """Guard: reply with the shared "not configured" message and return
        ``False`` when the calibration monitor is unavailable, else ``True``.
        """
        if self._monitor is None:
            await self._reply(update, "Calibration monitor not configured.")
            return False
        return True

    @_owner_only
    async def _cmd_calibration_health(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._require_monitor(update):
            return
        assert self._monitor is not None
        await self._reply(
            update,
            format_calibration_health(
                self._monitor.latest_snapshot, self._monitor.health
            ),
        )

    @_owner_only
    async def _cmd_ack(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Owner-only: record 'reviewed, no action' for one alert code.

        Only codes whose alert declared an ``ack_key`` are acceptable — a code
        with no invalidating context would become a permanent mute (see
        ``Orchestrator._ackable``). There is deliberately no 'seen this
        process' guard: the tripwire sweep runs hourly, and an unplanned
        restart between an alert's delivery and the operator typing /ack
        would otherwise reject a legitimate acknowledgement with no
        explanation. Membership in ``_ackable`` plus this confirmation echo
        is the whole guard.

        Captures whether ``owner`` is live RIGHT NOW and persists it as
        ``owner_live_at_ack`` — this is a transition test, not a point-in-
        time one (see ``persist_ack_code`` / ``Orchestrator._ack_suppresses``):
        an ack taken while the owner was not live is dropped the moment that
        owner goes live, because the judgment was made in a no-money context.
        """
        if self._orchestrator is None:
            await self._reply(update, "Orchestrator not wired — cannot record acknowledgements.")
            return
        args = getattr(context, "args", None)
        if not isinstance(args, (list, tuple)) or not args:
            await self._reply(update, "Usage: /ack <alert-code>")
            return
        code = str(args[0])
        entry = self._orchestrator._ackable.get(code)
        if entry is None:
            await self._reply(
                update,
                f"'{code}' is not acknowledgeable. Only artifact-state alerts that carry an "
                "invalidating context (a fit date or artifact path) can be acked; liveness and "
                "missing-artifact alerts cannot — acking those would be a permanent mute. "
                "See /acks for codes seen so far.",
            )
            return
        owner, severity = entry
        owner_live_at_ack = self._orchestrator._owner_is_live(owner)
        persist_ack_code(
            Path(self._settings.state_dir),
            code,
            owner=owner,
            severity=severity,
            owner_live_at_ack=owner_live_at_ack,
        )
        self._orchestrator._acked[code] = {
            "ts": datetime.now(UTC).isoformat(),
            "owner": owner,
            "severity": severity,
            "owner_live_at_ack": owner_live_at_ack,
        }
        if owner_live_at_ack:
            rearm = "the artifact changes or the severity escalates"
        else:
            owner_label = owner or "its owner"
            rearm = f"the artifact changes, the severity escalates, or {owner_label} goes live"
        await self._reply(
            update,
            f"Acknowledged {code} at {severity}. Silent until {rearm}.",
        )

    @_owner_only
    async def _cmd_acks(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Owner-only: list acknowledgements and alert-suppression counts."""
        if self._orchestrator is None:
            await self._reply(update, "Orchestrator not wired — nothing to show.")
            return
        await self._reply(
            update,
            format_acks(
                acked=self._orchestrator._acked,
                owner_is_live=self._orchestrator._owner_is_live,
                suppressions=self._orchestrator._alert_suppressions,
                ackable=self._orchestrator._ackable,
            ),
        )

    @_owner_only
    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Fire the kill FIRST — the safety action must not depend on the audit
        # write (which is best-effort and fully swallowed on error).
        self._kill_callback()
        assert update.effective_chat is not None  # @_owner_only guarantees non-None
        await self._write_audit_event(
            event="kill",
            chat_id=update.effective_chat.id,
            reason=_reason_from_args(context),
        )
        await self._reply(update, "KILL SWITCH ACTIVATED. Trading halted immediately.")

    @_owner_only
    async def _cmd_unkill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self._unkill_callback()
        assert update.effective_chat is not None  # @_owner_only guarantees non-None
        await self._write_audit_event(
            event="unkill",
            chat_id=update.effective_chat.id,
            reason=_reason_from_args(context),
        )
        await self._reply(
            update,
            "Kill switch deactivated. Bot must be restarted to resume trading.",
        )

    async def _write_audit_event(
        self, *, event: str, chat_id: int, reason: str | None
    ) -> None:
        """Append one structured row to ``state/audit_events.jsonl`` for
        kill/unkill forensics (the "when" the Telegram history doesn't durably keep).

        Best-effort: the row is written off the event loop via ``asyncio.to_thread``
        and any failure is logged and swallowed, so a disk error can NEVER block the
        kill path. Uses a dedicated audit file — never ``trades.jsonl`` — to avoid any
        coupling with the FillsReconciler dedup/lock invariants.
        """

        def _write() -> None:
            audit_dir = Path(self._settings.state_dir)
            audit_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "event": event,
                "chat_id": chat_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "reason": reason,
            }
            with open(audit_dir / "audit_events.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")

        try:
            await asyncio.to_thread(_write)
        except Exception:
            logger.warning(
                "Failed to write %s audit event (chat_id=%s)",
                event,
                chat_id,
                exc_info=True,
            )

    @_owner_only
    async def _cmd_disable_reconciler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Owner-only: disable the FillsReconciler in-process.

        Process-local kill switch — survives only until the orchestrator
        restarts. For a persistent disable, set ``fills_reconciler.enabled:
        false`` in ``python/config.yaml`` and redeploy. Spec §13.
        """
        if self._fills_reconciler is None:
            await self._reply(
                update,
                "Fills reconciler is not configured (fills_reconciler.enabled is false).",
            )
            return
        self._fills_reconciler.disable()
        await self._reply(
            update,
            "Fills reconciler disabled (process-local — survives until restart).",
        )

    @_owner_only
    async def _cmd_enable_reconciler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Owner-only: re-enable the FillsReconciler after a /disable_reconciler."""
        if self._fills_reconciler is None:
            await self._reply(update, "Fills reconciler is not configured.")
            return
        self._fills_reconciler.enable()
        await self._reply(update, "Fills reconciler re-enabled.")

    @_owner_only
    async def _cmd_enable(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            names = [s.name for s in self._strategies]
            await self._reply(update, f"Usage: /enable <strategy>\nAvailable: {', '.join(names)}")
            return

        name = context.args[0]
        result = self._enable_strategy_callback(name)
        await self._reply(update, result)

    @_owner_only
    async def _cmd_set_exposure(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set the runtime max_total_exposure_pct cap.

        Accepts either a percent (``1..100``, e.g. ``/setexposure 75``) or a
        fraction (``0 < x <= 1``, e.g. ``/setexposure 0.75``).  Values <= 0 or
        > 100 are rejected; use ``/kill`` for zero exposure.

        Owner-only, swallows errors so a Telegram failure never crashes the
        orchestrator. Persists to ``state/exposure_override.json``.
        """
        baseline = self._settings.risk.max_total_exposure_pct
        if not context.args:
            current = self._risk.effective_max_total_exposure_pct
            override = self._risk.exposure_override
            tag = " (override)" if override is not None else ""
            await self._reply(
                update,
                f"Current exposure cap: {current * 100:.1f}%{tag} "
                f"(baseline {baseline * 100:.1f}%)\n"
                f"Usage: /setexposure <1-100 pct | 0-1 frac>",
            )
            return

        raw = context.args[0]
        try:
            value = float(raw)
        except ValueError:
            await self._reply(
                update, f"Bad value {raw!r}. Use e.g. /setexposure 80 or /setexposure 0.80"
            )
            return

        # Interpret >1 as percent, <=1 as fraction. Accepts 100 as 1.0.
        if value > 1.0:
            fraction = value / 100.0
        else:
            fraction = value

        if not (0.0 < fraction <= 1.0):
            await self._reply(
                update,
                "Exposure must be in (0, 100]% or (0, 1]. Use /kill for zero.",
            )
            return

        try:
            prior = self._risk.effective_max_total_exposure_pct
            new_effective = self._risk.set_exposure_override(fraction)
        except ValueError as exc:
            await self._reply(update, f"Rejected: {exc}")
            return

        assert update.effective_chat is not None  # @_owner_only guarantees non-None
        chat_id = update.effective_chat.id
        logger.warning(
            "exposure cap changed via telegram chat=%s: %.4f -> %.4f (baseline=%.4f)",
            chat_id,
            prior,
            new_effective,
            baseline,
        )

        # Persistence: survives restarts so a tightening sticks.
        try:
            self._risk.persist_exposure_override(Path(self._settings.state_dir))
        except Exception:
            logger.warning("failed to persist exposure override", exc_info=True)

        await self._reply(
            update,
            f"Exposure cap set to {new_effective * 100:.1f}% "
            f"(was {prior * 100:.1f}%, baseline {baseline * 100:.1f}%).",
        )

    @_owner_only
    async def _cmd_reset_exposure(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Clear any runtime exposure override, reverting to the config baseline."""
        baseline = self._settings.risk.max_total_exposure_pct
        if self._risk.exposure_override is None:
            await self._reply(
                update,
                f"No override active. Exposure cap is already at baseline "
                f"{baseline * 100:.1f}%.",
            )
            return

        prior = self._risk.effective_max_total_exposure_pct
        try:
            self._risk.set_exposure_override(None)
        except ValueError as exc:  # unreachable for None but be defensive
            await self._reply(update, f"Rejected: {exc}")
            return

        assert update.effective_chat is not None  # @_owner_only guarantees non-None
        chat_id = update.effective_chat.id
        logger.warning(
            "exposure cap reset via telegram chat=%s: %.4f -> %.4f (baseline)",
            chat_id,
            prior,
            baseline,
        )

        try:
            self._risk.persist_exposure_override(Path(self._settings.state_dir))
        except Exception:
            logger.warning("failed to clear persisted exposure override", exc_info=True)

        await self._reply(
            update,
            f"Exposure cap reverted to baseline {baseline * 100:.1f}% "
            f"(was {prior * 100:.1f}%).",
        )

    @_owner_only
    async def _cmd_loglevel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Temporarily change root log level.

        Usage:
          /loglevel                - show current level + pending revert
          /loglevel debug 10m      - set DEBUG for 10 minutes (1..30m)
          /loglevel info           - revert immediately to the configured baseline
        """
        args = [a.lower() for a in (context.args or [])]

        if not args:
            await self._reply(update, self._format_loglevel_status())
            return

        level_name = args[0]
        if level_name not in _LOG_LEVEL_ALLOWED:
            await self._reply(update, _loglevel_usage("unknown level"))
            return

        duration_seconds: int | None = None
        if len(args) == 1:
            if level_name == "debug":
                await self._reply(update, _loglevel_usage("debug requires a duration (1m..30m)"))
                return
        elif len(args) == 2:
            if level_name != "debug":
                await self._reply(update, _loglevel_usage("duration only valid for debug"))
                return
            match = _LOG_DURATION_RE.match(args[1])
            if match is None:
                await self._reply(update, _loglevel_usage("bad duration (use 1m..30m)"))
                return
            duration_seconds = int(match.group(1)) * 60
        else:
            await self._reply(update, _loglevel_usage("too many arguments"))
            return

        new_level = _LOG_LEVEL_ALLOWED[level_name]
        assert update.effective_chat is not None  # @_owner_only guarantees non-None
        chat_id = update.effective_chat.id

        await self._cancel_log_revert_task()

        if new_level == self._original_log_level and duration_seconds is None:
            self._restore_log_baseline(reason=f"manual revert via telegram chat={chat_id}")
            await self._reply(update, f"Log level reverted to {logging.getLevelName(new_level)}.")
            return

        self._apply_log_override(new_level, chat_id=chat_id)

        if duration_seconds is None:
            await self._reply(update, f"Log level set to {logging.getLevelName(new_level)}.")
            return

        revert_at = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=duration_seconds)
        self._log_revert_at = revert_at
        self._log_revert_task = asyncio.create_task(
            self._schedule_log_revert(duration_seconds, chat_id=chat_id)
        )
        await self._reply(
            update,
            f"Log level set to {logging.getLevelName(new_level)} for {duration_seconds // 60}m "
            f"(reverts at {revert_at.strftime('%H:%M:%SZ')}).",
        )

    @_owner_only
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, format_help())

    # -- Log-level helpers ---------------------------------------------------

    def _capture_log_baseline(self) -> None:
        self._original_log_level = logging.getLogger().level or logging.INFO
        self._original_module_levels = {
            name: logging.getLogger(name).level
            for name in self._settings.logging.module_levels
        }

    def _apply_log_override(self, new_level: int, *, chat_id: int) -> None:
        """Set root to ``new_level`` and clear module_levels pins so DEBUG propagates to them."""
        prior = logging.getLogger().level
        logging.getLogger().setLevel(new_level)
        for name in self._original_module_levels:
            logging.getLogger(name).setLevel(logging.NOTSET)
        logger.warning(
            "log level changed to %s via telegram chat=%s (prior=%s)",
            logging.getLevelName(new_level),
            chat_id,
            logging.getLevelName(prior),
        )

    def _restore_log_baseline(self, *, reason: str) -> None:
        """Restore root + module_levels to the values captured at init/start."""
        prior = logging.getLogger().level
        logging.getLogger().setLevel(self._original_log_level)
        for name, level in self._original_module_levels.items():
            logging.getLogger(name).setLevel(level)
        self._log_revert_at = None
        if prior != self._original_log_level:
            logger.warning(
                "log level reverted to %s (%s, prior=%s)",
                logging.getLevelName(self._original_log_level),
                reason,
                logging.getLevelName(prior),
            )

    async def _schedule_log_revert(self, delay_seconds: int, *, chat_id: int) -> None:
        """Sleep then restore baseline. Cancellation exits cleanly."""
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            return
        try:
            self._restore_log_baseline(reason=f"auto revert after {delay_seconds // 60}m")
        except Exception:
            logger.warning("auto log-level revert failed", exc_info=True)
        finally:
            self._log_revert_task = None

    async def _cancel_log_revert_task(self) -> None:
        """Cancel and await the pending revert task, bounded by a short timeout."""
        task = self._log_revert_task
        if task is None or task.done():
            self._log_revert_task = None
            return
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=_LOG_REVERT_STOP_TIMEOUT_S)
        except (asyncio.CancelledError, TimeoutError):
            pass
        except Exception:
            logger.warning("log revert task raised during cancel", exc_info=True)
        finally:
            self._log_revert_task = None
            self._log_revert_at = None

    def _format_loglevel_status(self) -> str:
        current = logging.getLevelName(logging.getLogger().level or logging.INFO)
        baseline = logging.getLevelName(self._original_log_level)
        lines = [
            f"LOG LEVEL: {current} (baseline {baseline})",
        ]
        if self._log_revert_at is not None:
            lines.append(f"  Reverts at {self._log_revert_at.strftime('%H:%M:%SZ')}")
        lines.append(_loglevel_usage(None))
        return "\n".join(lines)

    # -- Internal helpers ----------------------------------------------------

    async def _reply(self, update: Update, text: str) -> None:
        """Reply to a command, splitting into chunks if it exceeds Telegram's limit."""
        if update.message is None:
            return
        for i in range(0, len(text), _MAX_MESSAGE_LEN):
            try:
                await update.message.reply_text(text[i : i + _MAX_MESSAGE_LEN])
            except Exception:
                logger.warning("Failed to reply to Telegram command", exc_info=True)
                break

    async def _safe_send(self, text: str) -> bool:
        """Send a message to the owner, swallowing all errors.

        Returns True iff the message reached the Telegram API without raising.
        """
        if self._application is None:
            return False
        try:
            await self._application.bot.send_message(
                chat_id=self._chat_id,
                text=text[:_MAX_MESSAGE_LEN],
            )
            self._message_timestamps.append(time.monotonic())
            return True
        except Exception:
            logger.warning("Failed to send Telegram message", exc_info=True)
            return False

    def _rate_limit_ok(self) -> bool:
        """Check if we're within the per-minute rate limit."""
        now = time.monotonic()
        cutoff = now - 60.0
        self._message_timestamps = [t for t in self._message_timestamps if t > cutoff]
        return len(self._message_timestamps) < self._rate_limit
