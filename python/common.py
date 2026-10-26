"""
Shared enums used across The Oracle codebase.

Centralises string constants that appear in multiple modules so that
typos are caught at import time rather than at runtime.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

# Canonical US Eastern timezone used for day-boundary detection,
# daily P&L resets, and market date comparisons.
ET = ZoneInfo("America/New_York")

# NWS reports daily high/low as integers (round-half-up). A bracket ticker
# [floor, cap] settles YES whenever the true temp falls in [floor − 0.5, cap + 0.5).
NWS_ROUND_HALF_F = 0.5


def et_today() -> date:
    """Current calendar date in ET."""
    return datetime.now(ET).date()


class Side(StrEnum):
    YES = "YES"
    NO = "NO"
    NONE = "NONE"


class Direction(StrEnum):
    ABOVE = "above"
    BELOW = "below"
    BETWEEN = "between"


class OrderStatus(StrEnum):
    PENDING = "pending"
    RESTING = "resting"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    SHADOW = "shadow"
    HYDRATED = "hydrated"
    SETTLED = "settled"
    EXTERNALLY_CLOSED = "externally_closed"  # closed via manual sell in Kalshi UI
    REJECTED = "rejected"
    SKIPPED = "skipped"
    ERROR = "error"
    ZERO_SIZE = "zero-size"


class DailyVariable(StrEnum):
    """Open-Meteo daily variable used for ensemble forecasts."""

    TEMP_MAX = "temperature_2m_max"
    TEMP_MIN = "temperature_2m_min"


class ExecutionMode(StrEnum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"
