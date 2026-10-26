"""Unit tests for ``python/data/kxrain_budget.py``.

The clock is injected, so nothing here sleeps."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_budget import (  # noqa: E402
    DEFAULT_OVERDRAFT_REQUESTS,
    REQUESTS_PER_MARKET,
    RequestBudget,
)


class FakeClock:
    """Monotonic-shaped clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _budget(cap: int = 60, wall: float = 180.0, clock: FakeClock | None = None) -> RequestBudget:
    return RequestBudget(
        max_requests=cap,
        wall_clock_budget_seconds=wall,
        clock=clock or FakeClock(),
    )


# Basic accounting


def test_fresh_budget_has_no_abort_reason():
    budget = _budget()
    assert budget.abort_reason is None
    assert budget.requests_used == 0
    assert budget.can_spend(1) is True


def test_spend_accumulates():
    budget = _budget(cap=5)
    budget.spend(2)
    budget.spend(1)
    assert budget.requests_used == 3
    assert budget.remaining == 2


def test_can_spend_respects_the_cap():
    budget = _budget(cap=3)
    budget.spend(3)
    assert budget.can_spend(1) is False
    assert budget.can_spend(0) is True


def test_spend_rejects_a_negative_count():
    budget = _budget()
    budget.spend(-5)
    assert budget.requests_used == 0


# Per-market reservation: never split a pair


def test_pair_reservation_needs_room_for_both_legs():
    budget = _budget(cap=5)
    budget.spend(4)
    # One slot left: must fail both legs together, not split the pair.
    assert budget.remaining == 1
    assert budget.reserve_market_pair() is False
    assert budget.requests_used == 4
    assert budget.abort_reason == "budget_abort"


def test_pair_reservation_succeeds_with_exactly_two_slots():
    budget = _budget(cap=4)
    budget.spend(2)
    assert budget.reserve_market_pair() is True
    assert budget.requests_used == 4
    assert budget.abort_reason is None


def test_pair_reservation_consumes_exactly_two_requests():
    budget = _budget(cap=60)
    assert REQUESTS_PER_MARKET == 2
    budget.reserve_market_pair()
    assert budget.requests_used == REQUESTS_PER_MARKET


def test_a_cap_that_binds_mid_pass_stops_every_later_market():
    budget = _budget(cap=4)
    assert budget.reserve_market_pair() is True
    assert budget.reserve_market_pair() is True
    assert budget.reserve_market_pair() is False
    assert budget.reserve_market_pair() is False
    assert budget.requests_used == 4


# Overdraft: only for the book of a market that just became eligible


def test_overdraft_book_is_allowed_past_the_cap():
    budget = _budget(cap=4)
    budget.spend(4)
    assert budget.can_spend(1) is False
    assert budget.reserve_book_overdraft() is True
    assert budget.requests_used == 5
    assert budget.abort_reason is None


def test_overdraft_is_bounded():
    budget = _budget(cap=4)
    budget.spend(4)
    assert DEFAULT_OVERDRAFT_REQUESTS == 5
    for _ in range(DEFAULT_OVERDRAFT_REQUESTS):
        assert budget.reserve_book_overdraft() is True
    assert budget.requests_used == 4 + DEFAULT_OVERDRAFT_REQUESTS
    assert budget.reserve_book_overdraft() is False
    assert budget.abort_reason == "budget_abort"


def test_overdraft_does_not_widen_the_pair_reservation():
    budget = _budget(cap=4)
    budget.spend(4)
    # Overdraft applies only to the book leg, not a whole new market.
    assert budget.reserve_market_pair() is False
    assert budget.requests_used == 4


def test_can_spend_with_overdraft_flag():
    budget = _budget(cap=4)
    budget.spend(4)
    assert budget.can_spend(1) is False
    assert budget.can_spend(1, overdraft=True) is True
    assert budget.can_spend(6, overdraft=True) is False


# Wall clock


def test_wall_clock_exhausted_only_after_the_budget_elapses():
    clock = FakeClock()
    budget = _budget(cap=60, wall=180.0, clock=clock)
    assert budget.wall_clock_exhausted() is False
    clock.advance(179.0)
    assert budget.wall_clock_exhausted() is False
    clock.advance(2.0)
    assert budget.wall_clock_exhausted() is True
    assert budget.abort_reason == "wall_clock_abort"


def test_wall_clock_beats_the_budget_when_both_would_bind():
    clock = FakeClock()
    budget = _budget(cap=2, wall=10.0, clock=clock)
    budget.spend(2)
    clock.advance(11.0)
    assert budget.reserve_market_pair() is False
    assert budget.abort_reason == "wall_clock_abort"


def test_elapsed_seconds_tracks_the_injected_clock():
    clock = FakeClock()
    budget = _budget(clock=clock)
    clock.advance(12.5)
    assert budget.elapsed_seconds == 12.5


# Abort reasons


def test_backoff_abort_is_explicit():
    budget = _budget()
    budget.mark_backoff_abort()
    assert budget.abort_reason == "backoff_abort"
    assert budget.aborted is True


def test_first_abort_reason_wins():
    budget = _budget(cap=1)
    assert budget.reserve_market_pair() is False
    assert budget.abort_reason == "budget_abort"
    budget.mark_backoff_abort()
    assert budget.abort_reason == "budget_abort"


def test_abort_reasons_are_the_documented_vocabulary():
    from python.data.kxrain_budget import ABORT_REASONS

    assert ABORT_REASONS == frozenset(
        {"budget_abort", "wall_clock_abort", "backoff_abort"}
    )


def test_aborted_budget_refuses_further_reservations():
    budget = _budget(cap=60)
    budget.mark_backoff_abort()
    assert budget.reserve_market_pair() is False
    assert budget.reserve_book_overdraft() is False
    assert budget.requests_used == 0


def test_book_overdraft_is_reachable_after_the_pair_reservation_hits_the_cap():
    # A sticky abort must not block the overdraft, or it can never fire.
    budget = RequestBudget(max_requests=5, wall_clock_budget_seconds=180, clock=lambda: 0.0)
    budget.spend(4)
    assert budget.reserve_market_pair() is False
    assert budget.reserve_book_overdraft() is True
    assert budget.abort_reason == "budget_abort"


def test_book_overdraft_still_refuses_after_a_wall_clock_abort():
    now = [0.0]
    budget = RequestBudget(max_requests=50, wall_clock_budget_seconds=10, clock=lambda: now[0])
    now[0] = 99.0
    assert budget.reserve_market_pair() is False
    assert budget.reserve_book_overdraft() is False
