"""Kalshi-truth tests for canonical outcome derivation.

These tests pin ``python.analytics.outcomes.derive_outcome_yes`` to the
empirically-verified Kalshi settlement convention for weather threshold
markets:

* ``between`` — ``floor <= t <= cap`` (CLOSED, right-inclusive). Verified
  on n=42/42 settled markets where ``actual_temp == cap_strike`` resolved
  YES on Kalshi.
* ``greater`` — ``t > floor`` (STRICT). Verified 2026-05-19 via Kalshi
  API rules_primary on KXHIGHNY-26MAY16-T83: "is greater than 83°, then
  the market resolves to Yes". Matches production
  ``EnsembleForecast.fraction_above`` (also strict ``>``). The brackets
  partition cleanly without overlap or gap at integer NWS observations:
  e.g., B66.5([66,67] closed) ∪ T67(greater, ≥68) covers all integers
  ≥66 with no double-coverage of 67.
* ``less`` — ``t < cap`` (strict). Matches production
  ``fraction_below = 1 - fraction_at_or_above`` exactly. Verified on
  n=5/5 settled markets.
"""

from __future__ import annotations

from python.analytics.outcomes import derive_outcome_yes


class TestBetweenBoundarySemantics:
    """Kalshi rule: ``between`` is CLOSED — ``floor <= t <= cap``."""

    def test_between_includes_right_boundary(self) -> None:
        # Kalshi rule: floor <= t <= cap → YES on t == cap.
        assert derive_outcome_yes("between", 70.0, 71.0, 71.0) is True

    def test_between_includes_left_boundary(self) -> None:
        # Kalshi rule: floor <= t <= cap → YES on t == floor.
        assert derive_outcome_yes("between", 70.0, 71.0, 70.0) is True

    def test_between_interior_yes(self) -> None:
        assert derive_outcome_yes("between", 70.0, 80.0, 75.0) is True

    def test_between_excludes_below_floor(self) -> None:
        assert derive_outcome_yes("between", 70.0, 71.0, 69.999) is False

    def test_between_excludes_above_cap(self) -> None:
        assert derive_outcome_yes("between", 70.0, 71.0, 71.001) is False

    def test_between_excludes_far_below(self) -> None:
        assert derive_outcome_yes("between", 70.0, 71.0, 50.0) is False

    def test_between_excludes_far_above(self) -> None:
        assert derive_outcome_yes("between", 70.0, 71.0, 90.0) is False


class TestGreaterBoundarySemantics:
    """Kalshi rule: ``greater`` is STRICT — ``t > floor``.

    Verified via Kalshi API rules_primary text on 2026-05-19:
    KXHIGHNY-26MAY16-T83 → "is greater than 83°, then the market
    resolves to Yes" (strict). Bracket partition shows the same:
    B66.5(between, {66,67}) ∪ T67(greater, ≥68) covers ≥66 without
    double-covering 67.
    """

    def test_greater_excludes_strike(self) -> None:
        # Kalshi rule: t > floor (STRICT) → NO on t == floor.
        # Verified on KXHIGHNY-26MAY16-T83 rules_primary 2026-05-19.
        assert derive_outcome_yes("greater", 80.0, None, 80.0) is False

    def test_greater_just_above_strike(self) -> None:
        assert derive_outcome_yes("greater", 80.0, None, 80.01) is True

    def test_greater_above_strike(self) -> None:
        assert derive_outcome_yes("greater", 80.0, None, 81.0) is True

    def test_greater_below_strike(self) -> None:
        assert derive_outcome_yes("greater", 80.0, None, 79.99) is False


class TestLessBoundarySemantics:
    """Kalshi rule: ``less`` is strict — ``t < cap``."""

    def test_less_below_strike(self) -> None:
        assert derive_outcome_yes("less", None, 80.0, 79.99) is True

    def test_less_at_strike_is_no(self) -> None:
        # Kalshi rule: t < cap (strict) → NO when t == cap.
        assert derive_outcome_yes("less", None, 80.0, 80.0) is False

    def test_less_above_strike(self) -> None:
        assert derive_outcome_yes("less", None, 80.0, 80.01) is False


class TestCanonicalBugExamples:
    """The three settled YES markets that the buggy right-exclusive code
    reported as NO (from the F8.1 calibration-monitor audit, 2026-05-19)."""

    def test_lax_high_may17_b70_5_settles_yes(self) -> None:
        # KXHIGHLAX-26MAY17-B70.5: bracket [70, 71], actual_temp=71.0,
        # Kalshi settled YES. Right-inclusive between rule.
        assert derive_outcome_yes("between", 70.0, 71.0, 71.0) is True

    def test_aus_low_may16_b68_5_settles_yes(self) -> None:
        # KXLOWTAUS-26MAY16-B68.5: bracket [68, 69], actual_temp=69.0,
        # Kalshi settled YES. Right-inclusive between rule.
        assert derive_outcome_yes("between", 68.0, 69.0, 69.0) is True

    def test_ny_high_may15_b65_5_settles_yes(self) -> None:
        # KXHIGHNY-26MAY15-B65.5: bracket [65, 66], actual_temp=66.0,
        # Kalshi settled YES. Right-inclusive between rule.
        assert derive_outcome_yes("between", 65.0, 66.0, 66.0) is True


class TestMissingStrikesStillReturnNone:
    def test_between_missing_floor(self) -> None:
        assert derive_outcome_yes("between", None, 80.0, 75.0) is None

    def test_between_missing_cap(self) -> None:
        assert derive_outcome_yes("between", 70.0, None, 75.0) is None

    def test_greater_missing_floor(self) -> None:
        assert derive_outcome_yes("greater", None, None, 80.0) is None

    def test_less_missing_cap(self) -> None:
        assert derive_outcome_yes("less", None, None, 80.0) is None

    def test_unknown_strike_type(self) -> None:
        assert derive_outcome_yes("equals", 80.0, 80.0, 80.0) is None
