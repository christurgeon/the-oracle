"""Per-tick request ledger for the KXRAIN shadow observer. Hermetic — no client, no I/O.

Reserves both legs of a market together; the book leg alone gets a bounded
overdraft since it can't be recomputed later."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "ABORT_REASONS",
    "DEFAULT_OVERDRAFT_REQUESTS",
    "REQUESTS_PER_MARKET",
    "RequestBudget",
]

# Trades + book, issued together so an aborted tick can't split a pair.
REQUESTS_PER_MARKET = 2

# Headroom for the book leg of a market that just became eligible.
DEFAULT_OVERDRAFT_REQUESTS = 5

# Distinct from TICK_OUTCOMES in kxrain_rows.py.
ABORT_REASONS: frozenset[str] = frozenset(
    {"budget_abort", "wall_clock_abort", "backoff_abort"}
)


class RequestBudget:
    """Mutable per-tick request ledger. Never sleeps, never does I/O.

    ``abort_reason`` is first-wins: once set, it is never relabeled."""

    def __init__(
        self,
        *,
        max_requests: int,
        wall_clock_budget_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        overdraft_requests: int = DEFAULT_OVERDRAFT_REQUESTS,
    ) -> None:
        self._max_requests = max(0, int(max_requests))
        self._wall_clock_budget = max(0.0, float(wall_clock_budget_seconds))
        self._clock = clock
        self._overdraft = max(0, int(overdraft_requests))
        self._started_at = clock()
        self._requests_used = 0
        self._abort_reason: str | None = None

    @property
    def requests_used(self) -> int:
        return self._requests_used

    @property
    def max_requests(self) -> int:
        return self._max_requests

    @property
    def remaining(self) -> int:
        """Slots left under the cap, ignoring the overdraft."""
        return max(0, self._max_requests - self._requests_used)

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._started_at

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    @property
    def aborted(self) -> bool:
        return self._abort_reason is not None

    def can_spend(self, n: int, *, overdraft: bool = False) -> bool:
        """Would ``n`` more fit? Does not record an abort."""
        limit = self._max_requests + (self._overdraft if overdraft else 0)
        return self._requests_used + max(0, int(n)) <= limit

    def spend(self, n: int) -> None:
        self._requests_used += max(0, int(n))

    def wall_clock_exhausted(self) -> bool:
        """True once past the wall-clock budget; also records the abort."""
        if self.elapsed_seconds > self._wall_clock_budget:
            self._mark("wall_clock_abort")
            return True
        return False

    def reserve_market_pair(self) -> bool:
        """Reserve both legs for one market, or neither, recording the abort on failure."""
        if self._abort_reason is not None:
            return False
        if self.wall_clock_exhausted():
            return False
        if not self.can_spend(REQUESTS_PER_MARKET):
            self._mark("budget_abort")
            return False
        self.spend(REQUESTS_PER_MARKET)
        return True

    def reserve_book_overdraft(self) -> bool:
        """Reserve one book-only request past the cap, up to the overdraft ceiling.

        Reachable after a budget abort (the book can't be reconstructed
        later); refused after a wall-clock or backoff abort."""
        if self._abort_reason == "wall_clock_abort" or self.wall_clock_exhausted():
            return False
        if self._abort_reason == "backoff_abort":
            return False
        if not self.can_spend(1, overdraft=True):
            self._mark("budget_abort")
            return False
        self.spend(1)
        return True

    def mark_backoff_abort(self) -> None:
        self._mark("backoff_abort")

    def _mark(self, reason: str) -> None:
        if self._abort_reason is None:
            self._abort_reason = reason
            logger.debug(
                "kxrain budget aborted (%s) after %d/%d requests, %.1fs",
                reason,
                self._requests_used,
                self._max_requests,
                self.elapsed_seconds,
            )
