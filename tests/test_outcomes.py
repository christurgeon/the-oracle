"""Tests for python.analytics.outcomes — canonical threshold outcome rules."""

from __future__ import annotations

from python.analytics.outcomes import derive_outcome_for_side, derive_outcome_yes


class TestDeriveOutcomeYes:
    def test_greater_strict_on_strike(self) -> None:
        # Kalshi rule: t > floor (STRICT). Verified 2026-05-19 via Kalshi
        # API rules_primary on KXHIGHNY-26MAY16-T83: "greater than 83°".
        # Prior test asserted t==floor is True, encoding a transient
        # `>=` over-correction from the 2026-05-19 outcome-bug-fix.
        assert derive_outcome_yes("greater", 80.0, None, 80.0) is False
        assert derive_outcome_yes("greater", 80.0, None, 80.01) is True
        assert derive_outcome_yes("greater", 80.0, None, 79.99) is False

    def test_less_exclusive_on_strike(self) -> None:
        # Mirrors scripts/fit_isotonic.py: YES if actual < cap
        assert derive_outcome_yes("less", None, 80.0, 79.99) is True
        assert derive_outcome_yes("less", None, 80.0, 80.0) is False
        assert derive_outcome_yes("less", None, 80.0, 80.01) is False

    def test_between_right_inclusive(self) -> None:
        # Kalshi rule: floor <= t <= cap (CLOSED, right-inclusive).
        # Updated 2026-05-19 — the prior right-exclusive assertion on
        # t == cap was guarding the bug fixed in outcomes.py. See
        # tests/test_outcomes_kalshi_semantics.py for the empirical
        # 42/42 truth-audit citation.
        assert derive_outcome_yes("between", 70.0, 80.0, 70.0) is True
        assert derive_outcome_yes("between", 70.0, 80.0, 79.99) is True
        assert derive_outcome_yes("between", 70.0, 80.0, 80.0) is True
        assert derive_outcome_yes("between", 70.0, 80.0, 69.99) is False
        assert derive_outcome_yes("between", 70.0, 80.0, 80.01) is False

    def test_missing_strikes_returns_none(self) -> None:
        assert derive_outcome_yes("greater", None, None, 80.0) is None
        assert derive_outcome_yes("less", None, None, 80.0) is None
        assert derive_outcome_yes("between", 70.0, None, 75.0) is None
        assert derive_outcome_yes("between", None, 80.0, 75.0) is None

    def test_unknown_strike_type(self) -> None:
        assert derive_outcome_yes("equals", 80.0, 80.0, 80.0) is None


class TestDeriveOutcomeForSide:
    def test_yes_side_mirrors_yes_outcome(self) -> None:
        assert derive_outcome_for_side("YES", "greater", 80.0, None, 81.0) is True
        assert derive_outcome_for_side("YES", "greater", 80.0, None, 79.0) is False

    def test_no_side_flips(self) -> None:
        assert derive_outcome_for_side("NO", "greater", 80.0, None, 81.0) is False
        assert derive_outcome_for_side("NO", "greater", 80.0, None, 79.0) is True

    def test_invalid_side(self) -> None:
        assert derive_outcome_for_side("MAYBE", "greater", 80.0, None, 81.0) is None

    def test_missing_strikes_returns_none(self) -> None:
        assert derive_outcome_for_side("YES", "greater", None, None, 81.0) is None
