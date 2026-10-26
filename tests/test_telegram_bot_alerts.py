"""Tests for Telegram bot alert level routing + ordering helpers.

Owns regression coverage for `python.notifications.base.level_priority`
and the alert-routing/dedup behavior of `TelegramNotifier.send_alert`
(see EMOS scope-B PR-1, Tasks B1/B2).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

# Optional dependency — the telegram bot package is in the `[telegram]` extra
# (`uv sync --extra telegram`). Skip this whole module when it's not available
# rather than blocking the full pytest collection.
pytest.importorskip("telegram")

from python.notifications.base import AlertLevel
from python.notifications.telegram_bot import TelegramNotifier


def test_level_priority_ranks_all_four_levels() -> None:
    from python.notifications.base import AlertLevel, level_priority

    assert level_priority(AlertLevel.INFO) == 0
    assert level_priority(AlertLevel.WARNING) == 1
    assert level_priority(AlertLevel.ERROR) == 2
    assert level_priority(AlertLevel.CRITICAL) == 3


@pytest.fixture
def fake_notifier():
    """A TelegramNotifier with _safe_send mocked + clock controllable."""
    n = TelegramNotifier.__new__(TelegramNotifier)  # bypass __init__
    n._safe_send = AsyncMock()
    n._error_dedup = {}
    n._error_send_count_this_hour = 0
    n._error_suppressed_count_this_hour = 0
    n._error_count_hour = None
    # Provide a no-op stub for the existing rate-limit token bucket
    n._rate_limit_ok = lambda: True
    n._utc_hour = lambda: 100  # frozen hour for tests
    return n


@pytest.mark.asyncio
async def test_error_level_emits_error_prefix(fake_notifier):
    await fake_notifier.send_alert(
        "disk full", level=AlertLevel.ERROR, failure_bucket="emos_disk_full"
    )
    fake_notifier._safe_send.assert_called_once_with("[ERROR] disk full")


@pytest.mark.asyncio
async def test_two_error_sends_same_bucket_same_hour_dedup_to_one(fake_notifier):
    await fake_notifier.send_alert(
        "first", level=AlertLevel.ERROR, failure_bucket="emos_load:AUS/high"
    )
    await fake_notifier.send_alert(
        "second", level=AlertLevel.ERROR, failure_bucket="emos_load:AUS/high"
    )
    assert fake_notifier._safe_send.await_count == 1


@pytest.mark.asyncio
async def test_distinct_buckets_each_send_within_global_cap(fake_notifier):
    for i in range(6):
        await fake_notifier.send_alert(
            f"err{i}", level=AlertLevel.ERROR, failure_bucket=f"emos_load:CELL{i}"
        )
    assert fake_notifier._safe_send.await_count == 6


@pytest.mark.asyncio
async def test_global_cap_suppresses_overflow_and_summarizes_at_next_hour(fake_notifier):
    for i in range(8):
        await fake_notifier.send_alert(
            f"err{i}", level=AlertLevel.ERROR, failure_bucket=f"emos_load:CELL{i}"
        )
    # 6 sent, 2 suppressed
    assert fake_notifier._safe_send.await_count == 6
    assert fake_notifier._error_suppressed_count_this_hour == 2

    # Roll the clock to the next hour and fire one more ERROR
    fake_notifier._utc_hour = lambda: 101
    await fake_notifier.send_alert(
        "next-hour", level=AlertLevel.ERROR, failure_bucket="emos_load:NEW"
    )
    # First should be the summary, second the new alert
    calls = [c.args[0] for c in fake_notifier._safe_send.await_args_list]
    assert any("[ERROR-SUMMARY] 2 EMOS ERROR alerts suppressed" in c for c in calls)
    assert any("[ERROR] next-hour" in c for c in calls)
