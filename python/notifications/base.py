"""Abstract notifier protocol and shared types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol


class AlertLevel(StrEnum):
    """Priority level for push alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# StrEnum has no ordinal comparison — never write `level >= AlertLevel.WARNING`.
# Use level_priority() for ordering decisions.
_LEVEL_PRIORITY: dict[AlertLevel, int] = {
    AlertLevel.INFO: 0,
    AlertLevel.WARNING: 1,
    AlertLevel.ERROR: 2,
    AlertLevel.CRITICAL: 3,
}


def level_priority(level: AlertLevel) -> int:
    return _LEVEL_PRIORITY[level]


class Notifier(Protocol):
    """Interface that any notification backend must satisfy."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send_alert(
        self,
        message: str,
        *,
        level: AlertLevel = AlertLevel.INFO,
        failure_bucket: str | None = None,
    ) -> None: ...

    async def send_trade_alert(self, order_record: Any) -> bool: ...

    async def send_risk_alert(self, event: str, details: str) -> None: ...
