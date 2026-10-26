"""Tests for the /ack and /acks Telegram commands.

/ack is the ONLY writer to the human-verdict ``acked`` namespace (see
python/analytics/audit_tripwires.py::persist_ack_code). These tests cover:
  - rejecting a code with no invalidation context (not in ``_ackable``)
  - persisting a verdict, including ``owner_live_at_ack`` for both a live
    and a not-live owner (the feature's hinge — see Task 8 brief)
  - /acks listing acknowledgements + suppression counters, and labeling an
    ack INACTIVE once its (not-live-at-ack) owner goes live

The ``notifier`` fixture wires a REAL ``Orchestrator`` (bare construction —
no async I/O at __init__ time, same pattern as tests/conftest.py's ``orch``
fixture) as ``notifier._orchestrator``, because /acks must observe live
liveness through ``Orchestrator._owner_is_live`` reading ``self._strategies``
fresh on each call. Tests mutate ``notifier._orchestrator._strategies``
in place — a REBIND, not a mutation, but safe here because the assertion
always goes through ``orchestrator._owner_is_live`` (a bound method that
reads the orchestrator's own current attribute), never through
``notifier._strategies`` (which IS captured by reference at construction
and would go stale on a rebind).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Optional dependency — see tests/test_telegram_bot_alerts.py for the same guard.
pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from python.analytics.audit_tripwires import load_acks  # noqa: E402
from python.notifications.telegram_bot import TelegramNotifier  # noqa: E402
from python.orchestrator import Orchestrator  # noqa: E402
from python.settings import Settings  # noqa: E402
from tests.conftest import _stub  # noqa: E402

_OWNER_CHAT_ID = 555


def _make_settings(tmp_path: Path) -> Settings:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state_dir),
        telegram_bot_token="test-token",
        telegram_owner_chat_id=_OWNER_CHAT_ID,
        _env_file=None,
    )


def _update(text: str) -> MagicMock:
    """A MagicMock Update from the owner chat. ``text`` is cosmetic (unused by
    the handlers, which read args from context) but documents the command."""
    update = MagicMock()
    update.effective_chat.id = _OWNER_CHAT_ID
    update.message.reply_text = AsyncMock()
    return update


def _ctx(args: list[str]) -> MagicMock:
    context = MagicMock()
    context.args = args
    return context


@pytest.fixture
def notifier(tmp_path: Path) -> TelegramNotifier:
    """A real TelegramNotifier wired to a real (bare) Orchestrator.

    ``_reply`` is swapped for a recorder so tests can assert on
    ``notifier.replies`` instead of mocking python-telegram-bot's
    ``Update.message.reply_text`` plumbing.
    """
    settings = _make_settings(tmp_path)
    orchestrator = Orchestrator(settings=settings)
    n = TelegramNotifier(
        settings=settings,
        risk_manager=MagicMock(),
        execution_engine=MagicMock(),
        strategies=[],
        kill_callback=MagicMock(),
        unkill_callback=MagicMock(),
        enable_strategy_callback=MagicMock(),
        orchestrator=orchestrator,
    )
    n.replies = []  # type: ignore[attr-defined]

    async def _reply(update: Any, text: str) -> None:
        n.replies.append(text)  # type: ignore[attr-defined]

    n._reply = _reply  # type: ignore[method-assign]
    return n


# ---------------------------------------------------------------------------
# /ack
# ---------------------------------------------------------------------------


async def test_ack_rejects_unackable_code(notifier: TelegramNotifier) -> None:
    notifier._orchestrator._ackable = {}  # type: ignore[union-attr]
    await notifier._cmd_ack(
        _update("/ack audit_log_stale:eval_audit.jsonl"),
        _ctx(["audit_log_stale:eval_audit.jsonl"]),
    )
    assert "not acknowledgeable" in notifier.replies[-1]  # type: ignore[attr-defined]


async def test_ack_with_no_args_shows_usage(notifier: TelegramNotifier) -> None:
    await notifier._cmd_ack(_update("/ack"), _ctx([]))
    assert "Usage" in notifier.replies[-1]  # type: ignore[attr-defined]


async def test_ack_records_a_verdict(notifier: TelegramNotifier) -> None:
    code = "calibrator_artifact_stale:favorite_bias:2026-07-04"
    notifier._orchestrator._ackable = {code: ("favorite_bias", "WARNING")}  # type: ignore[union-attr]
    await notifier._cmd_ack(_update(f"/ack {code}"), _ctx([code]))

    _seen, acked = load_acks(Path(notifier._settings.state_dir))
    assert code in acked
    assert acked[code]["owner"] == "favorite_bias"
    assert acked[code]["severity"] == "WARNING"

    # In-memory copy must agree with the persisted one on every field
    # _ack_suppresses reads — /ack is a second writer on the same event
    # loop, and each call stamps its own ``ts`` independently, so ``ts``
    # itself may differ by microseconds without being a bug.
    in_memory = notifier._orchestrator._acked[code]  # type: ignore[union-attr]
    assert in_memory["owner"] == acked[code]["owner"]
    assert in_memory["severity"] == acked[code]["severity"]
    assert in_memory["owner_live_at_ack"] == acked[code]["owner_live_at_ack"]

    reply = notifier.replies[-1]  # type: ignore[attr-defined]
    assert code in reply
    assert "WARNING" in reply


async def test_ack_confirmation_names_severity_and_rearm_trigger(
    notifier: TelegramNotifier,
) -> None:
    """The confirmation echo must name what re-arms the ack (spec requirement)."""
    code = "bias_table_stale:data/nws_bias/bias_correction.json:2026-06-01"
    notifier._orchestrator._strategies = [  # type: ignore[union-attr]
        _stub("weather_edge", enabled=False)
    ]
    notifier._orchestrator._ackable = {code: ("weather_edge", "CRITICAL")}  # type: ignore[union-attr]
    await notifier._cmd_ack(_update(f"/ack {code}"), _ctx([code]))
    reply = notifier.replies[-1]  # type: ignore[attr-defined]
    assert "CRITICAL" in reply
    assert "weather_edge" in reply
    assert "goes live" in reply


# ---------------------------------------------------------------------------
# owner_live_at_ack — the feature's hinge (mandatory per Task 8 instructions)
# ---------------------------------------------------------------------------


async def test_ack_records_owner_live_at_ack_true_for_a_live_owner(
    notifier: TelegramNotifier,
) -> None:
    notifier._orchestrator._strategies = [  # type: ignore[union-attr]
        _stub("favorite_bias", enabled=True, shadow_only=False)
    ]
    code = "calibrator_artifact_stale:favorite_bias:2026-07-04"
    notifier._orchestrator._ackable = {code: ("favorite_bias", "WARNING")}  # type: ignore[union-attr]

    await notifier._cmd_ack(_update(f"/ack {code}"), _ctx([code]))

    _seen, acked = load_acks(Path(notifier._settings.state_dir))
    assert acked[code]["owner_live_at_ack"] is True
    assert notifier._orchestrator._acked[code]["owner_live_at_ack"] is True  # type: ignore[union-attr]


async def test_ack_records_owner_live_at_ack_false_for_a_not_live_owner(
    notifier: TelegramNotifier,
) -> None:
    notifier._orchestrator._strategies = [  # type: ignore[union-attr]
        _stub("favorite_bias", enabled=False, shadow_only=False)
    ]
    code = "calibrator_artifact_stale:favorite_bias:2026-07-04"
    notifier._orchestrator._ackable = {code: ("favorite_bias", "WARNING")}  # type: ignore[union-attr]

    await notifier._cmd_ack(_update(f"/ack {code}"), _ctx([code]))

    _seen, acked = load_acks(Path(notifier._settings.state_dir))
    assert acked[code]["owner_live_at_ack"] is False
    assert notifier._orchestrator._acked[code]["owner_live_at_ack"] is False  # type: ignore[union-attr]


async def test_ack_owner_none_records_owner_live_at_ack_false(
    notifier: TelegramNotifier,
) -> None:
    """An infrastructure alert (owner=None) is never 'live' — _owner_is_live(None)
    is False by construction, so the ack must record False, not raise."""
    code = "audit_log_stale:eval_audit.jsonl:2026-08-01"
    notifier._orchestrator._ackable = {code: (None, "WARNING")}  # type: ignore[union-attr]

    await notifier._cmd_ack(_update(f"/ack {code}"), _ctx([code]))

    _seen, acked = load_acks(Path(notifier._settings.state_dir))
    assert acked[code]["owner_live_at_ack"] is False
    assert acked[code]["owner"] is None


# ---------------------------------------------------------------------------
# /acks
# ---------------------------------------------------------------------------


async def test_acks_lists_acked_and_suppressions(notifier: TelegramNotifier) -> None:
    from collections import Counter

    notifier._orchestrator._acked = {  # type: ignore[union-attr]
        "c1": {"ts": "t", "owner": "fb", "severity": "WARNING", "owner_live_at_ack": True}
    }
    notifier._orchestrator._alert_suppressions = Counter(  # type: ignore[union-attr]
        {("bias_table_stale:x", "weather_edge"): 6}
    )
    await notifier._cmd_acks(_update("/acks"), _ctx([]))
    reply = notifier.replies[-1]  # type: ignore[attr-defined]
    assert "c1" in reply
    assert "bias_table_stale:x" in reply
    assert "6" in reply


async def test_acks_with_nothing_recorded(notifier: TelegramNotifier) -> None:
    await notifier._cmd_acks(_update("/acks"), _ctx([]))
    reply = notifier.replies[-1]  # type: ignore[attr-defined]
    assert "none recorded" in reply.lower()


async def test_acks_labels_an_ack_whose_owner_is_live(notifier: TelegramNotifier) -> None:
    """An ack is dropped once its owner trades; /acks must say so rather than
    listing it as if it were still in force."""
    notifier._orchestrator._acked = {  # type: ignore[union-attr]
        "c1": {"ts": "t", "owner": "fb", "severity": "WARNING"}
    }
    notifier._orchestrator._strategies = [  # type: ignore[union-attr]
        _stub("fb", enabled=True, shadow_only=False)
    ]
    await notifier._cmd_acks(_update("/acks"), _ctx([]))
    assert "inactive" in notifier.replies[-1].lower()  # type: ignore[attr-defined]


async def test_acks_does_not_label_an_ack_taken_while_owner_was_live(
    notifier: TelegramNotifier,
) -> None:
    """The common case: acking a stale-calibrator alert for a strategy that is
    live right now must NOT show as inactive."""
    notifier._orchestrator._acked = {  # type: ignore[union-attr]
        "c1": {"ts": "t", "owner": "fb", "severity": "WARNING", "owner_live_at_ack": True}
    }
    notifier._orchestrator._strategies = [  # type: ignore[union-attr]
        _stub("fb", enabled=True, shadow_only=False)
    ]
    await notifier._cmd_acks(_update("/acks"), _ctx([]))
    assert "inactive" not in notifier.replies[-1].lower()  # type: ignore[attr-defined]


async def test_acks_shows_ackable_codes_seen_this_process(notifier: TelegramNotifier) -> None:
    """The /ack rejection message says 'See /acks for codes seen so far' —
    /acks must actually list them (spec §3.2), or that pointer leads nowhere."""
    notifier._orchestrator._ackable = {  # type: ignore[union-attr]
        "calibrator_artifact_stale:favorite_bias:2026-07-04": ("favorite_bias", "WARNING")
    }
    await notifier._cmd_acks(_update("/acks"), _ctx([]))
    reply = notifier.replies[-1]  # type: ignore[attr-defined]
    assert "calibrator_artifact_stale:favorite_bias:2026-07-04" in reply
    assert "ACKNOWLEDGEABLE" in reply


async def test_acks_not_owner_only_is_ignored(notifier: TelegramNotifier) -> None:
    update = _update("/acks")
    update.effective_chat.id = 999  # not the owner
    await notifier._cmd_acks(update, _ctx([]))
    assert notifier.replies == []  # type: ignore[attr-defined]
