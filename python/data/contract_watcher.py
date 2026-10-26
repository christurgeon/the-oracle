"""Deduplicated Telegram alerting for Kalshi API contract violations."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from python.notifications.base import AlertLevel

if TYPE_CHECKING:
    from python.data.kalshi_schemas import KalshiSchema

logger = logging.getLogger(__name__)

_PREFIX = "[CONTRACT]"


@dataclass
class ContractWatcher:
    """Validates Kalshi API response shapes and fires deduplicated Telegram alerts.

    alert_fn accepts both sync and async callables so tests can use plain
    lambdas without a running event loop. Production sets an async callable
    and receives fire-and-forget dispatch via asyncio.create_task().
    """

    alert_fn: Callable[[str, AlertLevel], Any] | None = None
    dedup_ttl: float = 3600.0  # seconds before the same violation re-alerts

    _seen: dict[str, float] = field(default_factory=dict, repr=False, init=False)

    def check(
        self,
        data: dict[str, Any],
        schema: KalshiSchema,
        *,
        context: str,
    ) -> None:
        """Validate *data* against *schema*; fire a WARNING alert on violations.

        Extra fields are ignored. Violations from one call are batched into a
        single alert message. The alert is suppressed if the same violation code
        fired within *dedup_ttl* seconds.
        """
        violated_fields: list[str] = []
        violations: list[str] = []

        for fname in schema.required:
            if fname not in data:
                violated_fields.append(fname)
                violations.append(f"missing required: {fname}")

        for fname, expected_types in schema.type_checks.items():
            value = data.get(fname)
            if value is not None and not isinstance(value, expected_types):
                violated_fields.append(fname)
                actual = type(value).__name__
                expected = "/".join(t.__name__ for t in expected_types)
                violations.append(f"type mismatch: {fname} expected {expected} got {actual}")

        if not violations:
            return

        code = f"{context}:{','.join(sorted(violated_fields))}"
        if self._is_suppressed(code):
            return

        violations_text = "\n".join(f"  {v}" for v in violations)
        payload_sample = repr(data)[:200]
        message = (
            f"{_PREFIX} Kalshi API drift detected — {context}\n"
            f"{violations_text}\n"
            f"Payload sample: {payload_sample}"
        )
        logger.warning("%s", message)
        self._fire(message)

    def warn(self, code: str, message: str) -> None:
        """Fire a deduplicated WARNING alert for a non-schema anomaly.

        Alert text: "[CONTRACT] {message} (code={code})"
        """
        if self._is_suppressed(code):
            return

        full = f"{_PREFIX} {message} (code={code})"
        logger.warning("%s", full)
        self._fire(full)

    def _is_suppressed(self, code: str) -> bool:
        now = time.monotonic()
        if code in self._seen and now - self._seen[code] < self.dedup_ttl:
            return True
        self._seen[code] = now
        return False

    def _fire(self, message: str) -> None:
        if self.alert_fn is None:
            return
        if asyncio.iscoroutinefunction(self.alert_fn):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.alert_fn(message, AlertLevel.WARNING))
            except RuntimeError:
                # No running loop (startup race window). Alert already logged.
                logger.debug("%s could not schedule Telegram task (no running loop)", _PREFIX)
        else:
            self.alert_fn(message, AlertLevel.WARNING)
