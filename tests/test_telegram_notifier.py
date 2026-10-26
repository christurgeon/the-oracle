"""Tests for the Telegram notifier.

These tests mock python-telegram-bot internals to verify auth, commands,
rate limiting, and error resilience without hitting the Telegram API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

telegram = pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from python.common import ExecutionMode, OrderStatus, Side  # noqa: E402
from python.execution.engine import OrderRecord  # noqa: E402
from python.notifications.base import AlertLevel  # noqa: E402
from python.notifications.telegram_bot import TelegramNotifier  # noqa: E402
from python.risk.manager import RiskDecision, StrategyTracker  # noqa: E402
from python.settings import Settings  # noqa: E402
from python.strategies.base import TradeSignal  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        state_dir=str(tmp_path / "state"),
        telegram_bot_token="test-token-123",
        telegram_owner_chat_id=42,
        _env_file=None,
        **overrides,
    )


def _make_signal(**overrides: Any) -> TradeSignal:
    defaults: dict[str, Any] = {
        "market_id": "MKT-001",
        "market_ticker": "KXHIGHNYC-26MAR22-T58",
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


def _make_record(status: OrderStatus = OrderStatus.SHADOW) -> OrderRecord:
    signal = _make_signal()
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0)
    return OrderRecord(
        order_id="shadow-1",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.SHADOW,
        signal_price=0.50,
        order_price=0.50,
        fill_price=0.50,
        fill_count=2,
        requested_count=2,
        status=status,
    )


@pytest.fixture
def notifier(tmp_path: Path) -> TelegramNotifier:
    settings = _make_settings(tmp_path)
    risk = MagicMock()
    risk.bankroll = 10_000.0
    risk.peak_bankroll = 10_000.0
    risk.daily_pnl = 0.0
    risk.total_exposure = 0.0
    risk.positions = {}
    risk.is_halted = False
    risk.drawdown = 0.0
    risk.effective_max_total_exposure_pct = 0.95
    risk.exposure_override = None
    risk.get_tracker.return_value = StrategyTracker(name="weather_edge")

    engine = MagicMock()
    type(engine).mode = PropertyMock(return_value=ExecutionMode.SHADOW)
    engine.order_history = []

    strategy = MagicMock()
    strategy.name = "weather_edge"
    strategy.enabled = True

    return TelegramNotifier(
        settings=settings,
        risk_manager=risk,
        execution_engine=engine,
        strategies=[strategy],
        kill_callback=MagicMock(),
        unkill_callback=MagicMock(),
        enable_strategy_callback=MagicMock(return_value="Strategy 'weather_edge' re-enabled."),
    )


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestOwnerAuth:
    async def test_owner_allowed(self, notifier: TelegramNotifier):
        """Messages from owner chat_id should be processed."""
        update = MagicMock()
        update.effective_chat.id = 42  # matches owner
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_help(update, context)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "THE ORACLE COMMANDS" in text

    async def test_non_owner_rejected(self, notifier: TelegramNotifier):
        """Messages from non-owner chat_id should be silently ignored."""
        update = MagicMock()
        update.effective_chat.id = 999  # does not match owner
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_status(update, context)

        update.message.reply_text.assert_not_called()

    async def test_no_chat_rejected(self, notifier: TelegramNotifier):
        """Messages with no effective_chat should be silently ignored."""
        update = MagicMock()
        update.effective_chat = None
        context = MagicMock()

        # Should not raise
        await notifier._cmd_status(update, context)


# ---------------------------------------------------------------------------
# Command tests
# ---------------------------------------------------------------------------


class TestCommands:
    async def test_kill_calls_callback(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_kill(update, context)

        notifier._kill_callback.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "KILL SWITCH ACTIVATED" in text

    async def test_unkill_calls_callback(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_unkill(update, context)

        notifier._unkill_callback.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "restarted" in text.lower()

    async def test_kill_writes_audit_event_with_reason(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["market", "looks", "wrong"]

        await notifier._cmd_kill(update, context)

        audit_path = Path(notifier._settings.state_dir) / "audit_events.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["event"] == "kill"
        assert rows[0]["chat_id"] == 42
        assert rows[0]["reason"] == "market looks wrong"
        assert rows[0]["timestamp"]  # non-empty UTC ISO timestamp

    async def test_unkill_writes_audit_event_no_reason(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []  # no reason supplied

        await notifier._cmd_unkill(update, context)

        audit_path = Path(notifier._settings.state_dir) / "audit_events.jsonl"
        rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        assert rows[-1]["event"] == "unkill"
        assert rows[-1]["reason"] is None

    async def test_kill_audit_write_failure_does_not_block_kill(
        self, notifier: TelegramNotifier
    ):
        """A disk error in the best-effort audit write must NOT prevent the kill
        from firing or the reply from being sent, and must not raise."""
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []

        with patch(
            "python.notifications.telegram_bot.asyncio.to_thread",
            side_effect=RuntimeError("disk full"),
        ):
            await notifier._cmd_kill(update, context)  # must not raise

        notifier._kill_callback.assert_called_once()
        update.message.reply_text.assert_called_once()  # reply still sent

    async def test_enable_no_args(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []

        await notifier._cmd_enable(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "Usage" in text

    async def test_enable_with_arg(self, notifier: TelegramNotifier):
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["weather_edge"]

        await notifier._cmd_enable(update, context)

        notifier._enable_strategy_callback.assert_called_once_with("weather_edge")

    async def test_disable_reconciler_toggles_disabled_flag(
        self, notifier: TelegramNotifier
    ):
        """/disable_reconciler calls reconciler.disable()."""
        reconciler = MagicMock()
        notifier._fills_reconciler = reconciler

        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_disable_reconciler(update, context)

        reconciler.disable.assert_called_once_with()
        reconciler.enable.assert_not_called()
        text = update.message.reply_text.call_args[0][0]
        assert "disabled" in text.lower()

    async def test_enable_reconciler_clears_disabled_flag(
        self, notifier: TelegramNotifier
    ):
        """/enable_reconciler calls reconciler.enable()."""
        reconciler = MagicMock()
        notifier._fills_reconciler = reconciler

        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_enable_reconciler(update, context)

        reconciler.enable.assert_called_once_with()
        reconciler.disable.assert_not_called()
        text = update.message.reply_text.call_args[0][0]
        assert "re-enabled" in text.lower() or "enabled" in text.lower()

    async def test_disable_reconciler_when_not_configured(
        self, notifier: TelegramNotifier
    ):
        """/disable_reconciler with no reconciler attached replies politely."""
        notifier._fills_reconciler = None

        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        # Must not raise AttributeError
        await notifier._cmd_disable_reconciler(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()

    async def test_enable_reconciler_when_not_configured(
        self, notifier: TelegramNotifier
    ):
        """/enable_reconciler with no reconciler attached replies politely."""
        notifier._fills_reconciler = None

        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_enable_reconciler(update, context)

        text = update.message.reply_text.call_args[0][0]
        assert "not configured" in text.lower()

    async def test_disable_reconciler_non_owner_rejected(
        self, notifier: TelegramNotifier
    ):
        """Non-owner /disable_reconciler is silently ignored — no flag mutation."""
        reconciler = MagicMock()
        notifier._fills_reconciler = reconciler

        update = MagicMock()
        update.effective_chat.id = 999  # not owner
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        await notifier._cmd_disable_reconciler(update, context)

        reconciler.disable.assert_not_called()
        update.message.reply_text.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    async def test_rate_limit_drops_messages(self, notifier: TelegramNotifier):
        """Messages beyond max_messages_per_minute should be dropped."""
        # Fill the rate limit window
        now = time.monotonic()
        notifier._message_timestamps = [now - i for i in range(20)]

        assert not notifier._rate_limit_ok()

    async def test_rate_limit_allows_under_limit(self, notifier: TelegramNotifier):
        """Messages under the limit should be allowed."""
        notifier._message_timestamps = []
        assert notifier._rate_limit_ok()

    async def test_critical_bypasses_rate_limit(self, notifier: TelegramNotifier):
        """Critical alerts should always send."""
        # Fill rate limit
        now = time.monotonic()
        notifier._message_timestamps = [now - i for i in range(25)]

        # Mock the application
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        await notifier.send_alert("CRITICAL TEST", level=AlertLevel.CRITICAL)

        notifier._application.bot.send_message.assert_called_once()

    async def test_warning_rate_limit_drop_logs_warning(
        self, notifier: TelegramNotifier, caplog
    ):
        """A WARNING-level send dropped by the rate limiter must emit a
        WARNING log so the silent drop is detectable in journald.

        Regression coverage for the 2026-05-18 [SHADOW] cash-out incident
        where rate-limit suppression left no trail anywhere.
        """
        import logging as stdlogging

        now = time.monotonic()
        notifier._message_timestamps = [now - i for i in range(20)]
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        with caplog.at_level(
            stdlogging.WARNING, logger="python.notifications.telegram_bot"
        ):
            await notifier.send_alert("test-shadow", level=AlertLevel.WARNING)

        notifier._application.bot.send_message.assert_not_called()
        warns = [r for r in caplog.records if r.levelno == stdlogging.WARNING]
        assert any("telegram_rate_limit_dropped" in r.message for r in warns), (
            f"rate-limit-drop WARNING log not emitted; got: "
            f"{[r.message for r in warns]}"
        )

    async def test_info_rate_limit_drop_logs_info(
        self, notifier: TelegramNotifier, caplog
    ):
        """INFO-level drops log at INFO so they're visible without spamming
        WARNING priority. Same observability principle as the WARNING case.
        """
        import logging as stdlogging

        now = time.monotonic()
        notifier._message_timestamps = [now - i for i in range(20)]
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        with caplog.at_level(
            stdlogging.INFO, logger="python.notifications.telegram_bot"
        ):
            await notifier.send_alert("test-info", level=AlertLevel.INFO)

        notifier._application.bot.send_message.assert_not_called()
        infos = [r for r in caplog.records if r.levelno == stdlogging.INFO]
        assert any("telegram_rate_limit_dropped" in r.message for r in infos), (
            f"rate-limit-drop INFO log not emitted; got: "
            f"{[r.message for r in infos]}"
        )


# ---------------------------------------------------------------------------
# Push alert tests
# ---------------------------------------------------------------------------


class TestPushAlerts:
    async def test_send_trade_alert(self, notifier: TelegramNotifier):
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()
        record = _make_record()

        await notifier.send_trade_alert(record)

        notifier._application.bot.send_message.assert_called_once()
        text = notifier._application.bot.send_message.call_args.kwargs["text"]
        assert "TRADE EXECUTED" in text

    async def test_send_trade_batch_single(self, notifier: TelegramNotifier):
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()
        record = _make_record()

        await notifier.send_trade_batch_alert([record])

        text = notifier._application.bot.send_message.call_args.kwargs["text"]
        assert "TRADE EXECUTED" in text  # single = full format

    async def test_send_trade_batch_multiple(self, notifier: TelegramNotifier):
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        await notifier.send_trade_batch_alert([_make_record(), _make_record()])

        text = notifier._application.bot.send_message.call_args.kwargs["text"]
        assert "TRADES EXECUTED (2)" in text

    async def test_send_trade_batch_filters_rejected(self, notifier: TelegramNotifier):
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        await notifier.send_trade_batch_alert([_make_record(OrderStatus.REJECTED)])

        notifier._application.bot.send_message.assert_not_called()

    async def test_send_trade_batch_announces_expired_partial(
        self, notifier: TelegramNotifier
    ):
        """A partial position acquired before the remainder cancelled (status
        EXPIRED/CANCELLED, fill_count > 0) is a real fill and must be announced."""
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        announced = await notifier.send_trade_batch_alert([_make_record(OrderStatus.EXPIRED)])

        notifier._application.bot.send_message.assert_called_once()
        assert len(announced) == 1

    async def test_send_trade_batch_drops_zero_fill_expired(self, notifier: TelegramNotifier):
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()

        record = _make_record(OrderStatus.EXPIRED)
        record.fill_count = 0
        announced = await notifier.send_trade_batch_alert([record])

        notifier._application.bot.send_message.assert_not_called()
        assert announced == []

    async def test_send_trade_batch_returns_empty_when_rate_limited(
        self, notifier: TelegramNotifier
    ):
        """A rate-limited send must report nothing announced so callers retry."""
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()
        notifier._rate_limit_ok = MagicMock(return_value=False)

        announced = await notifier.send_trade_batch_alert(
            [_make_record(), _make_record()]
        )

        notifier._application.bot.send_message.assert_not_called()
        assert announced == []


# ---------------------------------------------------------------------------
# Error resilience tests
# ---------------------------------------------------------------------------


class TestLogLevel:
    """Tests for /loglevel temporary runtime log level override."""

    @staticmethod
    def _update(text_ref: list[str]) -> MagicMock:
        update = MagicMock()
        update.effective_chat.id = 42
        update.message.reply_text = AsyncMock(
            side_effect=lambda msg: text_ref.append(msg)
        )
        return update

    @pytest.fixture(autouse=True)
    def _reset_root_logger(self):
        """Snapshot and restore root level so tests can't leak into each other."""
        root = logging.getLogger()
        prior = root.level
        yield
        root.setLevel(prior or logging.WARNING)

    async def test_no_args_shows_current_level(self, notifier: TelegramNotifier):
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = []

        await notifier._cmd_loglevel(update, context)

        assert replies, "expected a reply"
        assert "LOG LEVEL" in replies[0]
        assert "Usage" in replies[0]

    async def test_debug_with_duration_sets_level_and_schedules_revert(
        self, notifier: TelegramNotifier
    ):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug", "5m"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.DEBUG
        assert notifier._log_revert_task is not None
        assert not notifier._log_revert_task.done()
        assert "DEBUG" in replies[0]
        assert "5m" in replies[0]

        await notifier._cancel_log_revert_task()

    async def test_info_reverts_immediately(self, notifier: TelegramNotifier):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.DEBUG)
        # Simulate a prior pending revert.
        prior_task = asyncio.create_task(asyncio.sleep(3600))
        notifier._log_revert_task = prior_task
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["info"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.INFO
        assert prior_task.cancelled() or prior_task.done()
        assert notifier._log_revert_task is None
        assert "reverted" in replies[0].lower()

    async def test_rejects_debug_without_duration(self, notifier: TelegramNotifier):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.INFO
        assert "Usage" in replies[0]
        assert notifier._log_revert_task is None

    async def test_rejects_bad_duration(self, notifier: TelegramNotifier):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        for bad in ["0m", "31m", "10s", "10", "10min", ""]:
            replies: list[str] = []
            update = self._update(replies)
            context = MagicMock()
            context.args = ["debug", bad]
            await notifier._cmd_loglevel(update, context)
            assert logging.getLogger().level == logging.INFO, f"level changed for {bad!r}"
            assert "Usage" in replies[0], f"expected usage reply for {bad!r}"

    async def test_rejects_unknown_level(self, notifier: TelegramNotifier):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["trace", "5m"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.INFO
        assert "Usage" in replies[0]

    async def test_rejects_warning_level(self, notifier: TelegramNotifier):
        """warning/error are not allowed — they'd silence risk alerts."""
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["warning"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.INFO
        assert "Usage" in replies[0]

    async def test_second_call_cancels_prior_revert(self, notifier: TelegramNotifier):
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug", "5m"]

        await notifier._cmd_loglevel(update, context)
        first_task = notifier._log_revert_task
        assert first_task is not None

        await notifier._cmd_loglevel(update, context)
        second_task = notifier._log_revert_task

        assert second_task is not None
        assert second_task is not first_task
        assert first_task.cancelled() or first_task.done()

        await notifier._cancel_log_revert_task()

    async def test_module_levels_cleared_during_debug_restored_on_revert(
        self, notifier: TelegramNotifier
    ):
        """module_levels pins must yield to DEBUG and be restored afterwards."""
        notifier._original_log_level = logging.INFO
        notifier._original_module_levels = {"noisy.module": logging.WARNING}
        logging.getLogger("noisy.module").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug", "1m"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger("noisy.module").level == logging.NOTSET

        # Manual revert restores the pin.
        context.args = ["info"]
        update2 = self._update(replies)
        await notifier._cmd_loglevel(update2, context)

        assert logging.getLogger("noisy.module").level == logging.WARNING

    async def test_auto_revert_restores_baseline(
        self, notifier: TelegramNotifier, monkeypatch: pytest.MonkeyPatch
    ):
        """When the sleep expires naturally, baseline is restored."""
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)

        async def _instant_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug", "1m"]

        await notifier._cmd_loglevel(update, context)

        # Await the revert task to completion.
        task = notifier._log_revert_task
        assert task is not None
        await task

        assert logging.getLogger().level == logging.INFO
        assert notifier._log_revert_task is None
        assert notifier._log_revert_at is None

    async def test_stop_cancels_revert_and_restores(self, notifier: TelegramNotifier):
        """stop() must cancel the revert task and restore the baseline level."""
        notifier._original_log_level = logging.INFO
        logging.getLogger().setLevel(logging.INFO)
        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["debug", "30m"]

        await notifier._cmd_loglevel(update, context)
        task = notifier._log_revert_task
        assert task is not None

        # Stub out Application pieces that stop() touches.
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock()
        notifier._application.updater = MagicMock()
        notifier._application.updater.stop = AsyncMock()
        notifier._application.stop = AsyncMock()
        notifier._application.shutdown = AsyncMock()

        await notifier.stop()

        assert task.cancelled() or task.done()
        assert logging.getLogger().level == logging.INFO

    async def test_non_owner_rejected(self, notifier: TelegramNotifier):
        logging.getLogger().setLevel(logging.INFO)
        update = MagicMock()
        update.effective_chat.id = 999
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["debug", "5m"]

        await notifier._cmd_loglevel(update, context)

        assert logging.getLogger().level == logging.INFO
        update.message.reply_text.assert_not_called()

    async def test_handlers_inherit_root_level(self, notifier: TelegramNotifier):
        """Guardrail: if a future change sets an explicit handler level, /loglevel
        would stop propagating DEBUG to that handler. This asserts handlers rely
        on root level (NOTSET = 0)."""
        from python.utils.log_config import LoggingConfig, get_log_buffer, setup_logging

        setup_logging(LoggingConfig())
        root = logging.getLogger()
        for handler in root.handlers:
            assert handler.level == logging.NOTSET, (
                f"handler {handler} has explicit level {handler.level} — "
                "/loglevel will not propagate DEBUG through it"
            )
        assert get_log_buffer().level == logging.NOTSET


class TestSetExposure:
    """Tests for /setexposure and /resetexposure handlers."""

    @staticmethod
    def _update(text_ref: list[str], chat_id: int = 42) -> MagicMock:
        update = MagicMock()
        update.effective_chat.id = chat_id
        update.message.reply_text = AsyncMock(
            side_effect=lambda msg: text_ref.append(msg)
        )
        return update

    async def test_percent_value_applied(self, notifier: TelegramNotifier):
        notifier._risk.exposure_override = None
        notifier._risk.set_exposure_override = MagicMock(return_value=0.75)
        notifier._risk.persist_exposure_override = MagicMock()
        type(notifier._risk).effective_max_total_exposure_pct = PropertyMock(
            return_value=0.95
        )

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["75"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_called_once_with(0.75)
        notifier._risk.persist_exposure_override.assert_called_once()
        assert replies
        assert "75" in replies[0]

    async def test_fraction_value_applied(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock(return_value=0.4)
        notifier._risk.persist_exposure_override = MagicMock()
        type(notifier._risk).effective_max_total_exposure_pct = PropertyMock(
            return_value=0.95
        )

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["0.4"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_called_once_with(0.4)

    async def test_zero_rejected(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock()
        notifier._risk.persist_exposure_override = MagicMock()

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["0"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_not_called()
        notifier._risk.persist_exposure_override.assert_not_called()
        assert "/kill" in replies[0]

    async def test_above_100_rejected(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock()
        notifier._risk.persist_exposure_override = MagicMock()

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["150"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_not_called()
        assert "must be in" in replies[0] or "(0, 100" in replies[0]

    async def test_bad_value_rejected(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock()
        notifier._risk.persist_exposure_override = MagicMock()

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["banana"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_not_called()
        assert "Bad value" in replies[0]

    async def test_no_args_shows_current(self, notifier: TelegramNotifier):
        type(notifier._risk).effective_max_total_exposure_pct = PropertyMock(
            return_value=0.95
        )
        notifier._risk.exposure_override = None

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = []

        await notifier._cmd_set_exposure(update, context)

        assert "Current exposure cap" in replies[0]
        assert "Usage" in replies[0]

    async def test_non_owner_rejected(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock()
        update = MagicMock()
        update.effective_chat.id = 999  # not owner
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["75"]

        await notifier._cmd_set_exposure(update, context)

        notifier._risk.set_exposure_override.assert_not_called()
        update.message.reply_text.assert_not_called()

    async def test_persist_failure_swallowed(self, notifier: TelegramNotifier):
        notifier._risk.set_exposure_override = MagicMock(return_value=0.5)
        notifier._risk.persist_exposure_override = MagicMock(
            side_effect=Exception("disk full")
        )
        type(notifier._risk).effective_max_total_exposure_pct = PropertyMock(
            return_value=0.95
        )

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = ["50"]

        # Must not raise — persist failure is logged and swallowed.
        await notifier._cmd_set_exposure(update, context)

        # Reply still sent because the in-memory set succeeded.
        assert replies

    async def test_reset_reverts_override(self, notifier: TelegramNotifier):
        notifier._risk.exposure_override = 0.25
        notifier._risk.set_exposure_override = MagicMock(return_value=0.95)
        notifier._risk.persist_exposure_override = MagicMock()
        type(notifier._risk).effective_max_total_exposure_pct = PropertyMock(
            return_value=0.25
        )

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = []

        await notifier._cmd_reset_exposure(update, context)

        notifier._risk.set_exposure_override.assert_called_once_with(None)
        notifier._risk.persist_exposure_override.assert_called_once()
        assert "baseline" in replies[0]

    async def test_reset_when_no_override_active(self, notifier: TelegramNotifier):
        notifier._risk.exposure_override = None
        notifier._risk.set_exposure_override = MagicMock()
        notifier._risk.persist_exposure_override = MagicMock()

        replies: list[str] = []
        update = self._update(replies)
        context = MagicMock()
        context.args = []

        await notifier._cmd_reset_exposure(update, context)

        notifier._risk.set_exposure_override.assert_not_called()
        notifier._risk.persist_exposure_override.assert_not_called()
        assert "already" in replies[0].lower() or "baseline" in replies[0].lower()


class TestResilience:
    async def test_safe_send_swallows_errors(self, notifier: TelegramNotifier):
        """_safe_send should never raise, even if Telegram API fails."""
        notifier._application = MagicMock()
        notifier._application.bot.send_message = AsyncMock(side_effect=Exception("network error"))

        # Should not raise
        await notifier._safe_send("test message")

    async def test_reply_swallows_errors(self, notifier: TelegramNotifier):
        """_reply should not raise if reply_text fails."""
        update = MagicMock()
        update.message.reply_text = AsyncMock(side_effect=Exception("timeout"))

        await notifier._reply(update, "test")

    async def test_send_alert_without_application(self, notifier: TelegramNotifier):
        """send_alert should be a no-op if application is None."""
        notifier._application = None
        # Should not raise
        await notifier.send_alert("test")
