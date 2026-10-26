"""Tests for python/common.py shared enums."""

from __future__ import annotations

from python.common import OrderStatus


def test_externally_closed_status_exists() -> None:
    assert OrderStatus.EXTERNALLY_CLOSED.value == "externally_closed"


def test_externally_closed_is_stringy() -> None:
    # StrEnum invariant — useful for trade-log row serialization
    assert isinstance(OrderStatus.EXTERNALLY_CLOSED, str)
