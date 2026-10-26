"""Tests for the F4.1 same-day Kelly correlation haircut.

These tests pin three invariants:

  1. ``same_day_correlation_haircut(n, ρ)`` is monotone-decreasing in both
     arguments and equals 1.0 at the boundaries (n=0 or ρ≤0).
  2. The haircut value matches ``1/√(1 + n·ρ)`` exactly at a fiducial point.
  3. ``fractional_kelly`` applies the haircut iff both ``n_existing_same_day_bets``
     > 0 AND ``correlation_rho`` is a positive finite number. Otherwise the
     dollar size matches the legacy independent-Kelly result bit-for-bit.
"""

from __future__ import annotations

import math

import pytest

from python.analytics.sizing import (
    SizingResult,
    fractional_kelly,
    same_day_correlation_haircut,
)
from python.common import Side

# ── same_day_correlation_haircut ────────────────────────────────────────────


class TestHaircutFunction:
    def test_haircut_one_when_no_existing_bets(self) -> None:
        """n=0 → 1.0 regardless of rho. Boundary condition for the very
        first bet of the day."""
        for rho in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            assert same_day_correlation_haircut(0, rho) == 1.0

    def test_haircut_one_when_rho_zero_or_negative(self) -> None:
        """ρ≤0 → 1.0. Negative ρ would by symmetry justify increasing size,
        but we conservatively decline to do so."""
        for n in (1, 3, 10, 100):
            assert same_day_correlation_haircut(n, 0.0) == 1.0
            assert same_day_correlation_haircut(n, -0.1) == 1.0
            assert same_day_correlation_haircut(n, -1.0) == 1.0

    def test_haircut_decreases_with_n(self) -> None:
        """Monotone-decreasing in n when ρ>0."""
        rho = 0.2
        previous = same_day_correlation_haircut(1, rho)
        for n in range(2, 15):
            current = same_day_correlation_haircut(n, rho)
            assert current < previous, (
                f"haircut({n}, {rho}) = {current} should be < haircut({n - 1}, {rho}) = {previous}"
            )
            previous = current

    def test_haircut_decreases_with_rho(self) -> None:
        """Monotone-decreasing in ρ when n>0."""
        n = 5
        previous = same_day_correlation_haircut(n, 0.05)
        for rho in (0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.0):
            current = same_day_correlation_haircut(n, rho)
            assert current < previous, (
                f"haircut({n}, {rho}) = {current} should be < haircut({n}, prev_rho)"
            )
            previous = current

    def test_haircut_matches_formula(self) -> None:
        """At n=6, ρ=0.2: 1/sqrt(1 + 6·0.2) = 1/sqrt(2.2) ≈ 0.6741998624…
        Compared to the n=0 baseline of 1.0, that is ≈32.6% reduction."""
        expected = 1.0 / math.sqrt(2.2)
        observed = same_day_correlation_haircut(6, 0.2)
        assert observed == pytest.approx(expected, abs=1e-12)
        assert observed == pytest.approx(0.674199862463, abs=1e-9)
        # And n=0 baseline:
        assert same_day_correlation_haircut(0, 0.2) == 1.0

    def test_haircut_nonfinite_rho_is_noop(self) -> None:
        """NaN / inf rho → 1.0 (defensive against config corruption)."""
        assert same_day_correlation_haircut(5, float("nan")) == 1.0
        assert same_day_correlation_haircut(5, float("inf")) == 1.0
        assert same_day_correlation_haircut(5, -math.inf) == 1.0

    def test_haircut_negative_n_is_noop(self) -> None:
        """Defensive: caller bug shouldn't crash sizing."""
        assert same_day_correlation_haircut(-1, 0.2) == 1.0
        assert same_day_correlation_haircut(-100, 0.2) == 1.0


# ── fractional_kelly haircut plumbing ──────────────────────────────────────


def _kelly(**kwargs) -> SizingResult:
    """Sane default fixture for fractional_kelly. model_prob > market_price
    so we land on the YES side and the Kelly fraction is non-zero. bankroll
    chosen so we don't hit the per-market cap."""
    defaults = dict(
        model_prob=0.60,
        market_price=0.50,
        kelly_fraction=0.15,
        max_pct=1.0,  # disable per-market cap so haircut is the only multiplier
        bankroll=10_000.0,
        fee_per_contract=0.0,
    )
    defaults.update(kwargs)
    return fractional_kelly(**defaults)


class TestFractionalKellyHaircutPlumbing:
    def test_kelly_no_haircut_when_rho_none(self) -> None:
        """rho=None → legacy independent-Kelly behavior; haircut path
        entirely skipped."""
        baseline = _kelly()
        with_n = _kelly(n_existing_same_day_bets=10, correlation_rho=None)
        assert baseline.dollars == with_n.dollars
        assert baseline.side == with_n.side
        assert baseline.kelly_raw == with_n.kelly_raw

    def test_kelly_no_haircut_when_n_zero(self) -> None:
        """First bet of the day: haircut=1.0, dollar size unchanged."""
        baseline = _kelly()
        with_rho_only = _kelly(n_existing_same_day_bets=0, correlation_rho=0.2)
        assert baseline.dollars == with_rho_only.dollars

    def test_kelly_no_haircut_when_rho_zero(self) -> None:
        """rho=0 (default config) → no haircut even with many open bets."""
        baseline = _kelly()
        rho_zero = _kelly(n_existing_same_day_bets=10, correlation_rho=0.0)
        assert baseline.dollars == rho_zero.dollars

    def test_kelly_dollar_reduced_by_haircut_at_high_n(self) -> None:
        """At n=10, rho=0.2: haircut = 1/sqrt(1 + 10·0.2) = 1/sqrt(3).
        Returned dollars = baseline · 1/sqrt(3)."""
        baseline = _kelly()
        haircut_applied = _kelly(n_existing_same_day_bets=10, correlation_rho=0.2)
        ratio = haircut_applied.dollars / baseline.dollars
        expected_ratio = 1.0 / math.sqrt(3.0)
        # Tolerance loose enough to absorb the round(..., 2) at the end of
        # fractional_kelly, but tight enough to fail if the haircut path
        # is bypassed.
        assert ratio == pytest.approx(expected_ratio, abs=1e-3)

    def test_kelly_dollar_reduced_by_haircut_at_low_n(self) -> None:
        """At n=1, rho=0.2: haircut = 1/sqrt(1.2) ≈ 0.9128."""
        baseline = _kelly()
        haircut_applied = _kelly(n_existing_same_day_bets=1, correlation_rho=0.2)
        ratio = haircut_applied.dollars / baseline.dollars
        expected_ratio = 1.0 / math.sqrt(1.2)
        assert ratio == pytest.approx(expected_ratio, abs=1e-3)

    def test_haircut_does_not_flip_side(self) -> None:
        """Side is fixed by sign(model_prob - market_price); haircut only
        scales the dollar size, so side is invariant."""
        baseline = _kelly(model_prob=0.40, market_price=0.50)  # NO side
        haircut_applied = _kelly(
            model_prob=0.40,
            market_price=0.50,
            n_existing_same_day_bets=10,
            correlation_rho=0.2,
        )
        assert baseline.side == Side.NO
        assert haircut_applied.side == Side.NO

    def test_haircut_applied_before_max_pct_cap(self) -> None:
        """The cap should still bind on a large Kelly that the haircut
        couldn't shrink below the cap. Verifies the haircut is applied
        BEFORE the cap, not as a separate multiplicative reduction of the
        already-capped value (which would double-reduce)."""
        # Huge edge → big Kelly that would exceed max_pct
        # max_pct=0.01 of $10k = $100 cap
        baseline = fractional_kelly(
            model_prob=0.99,
            market_price=0.05,
            kelly_fraction=0.15,
            max_pct=0.01,
            bankroll=10_000.0,
        )
        haircut_applied = fractional_kelly(
            model_prob=0.99,
            market_price=0.05,
            kelly_fraction=0.15,
            max_pct=0.01,
            bankroll=10_000.0,
            n_existing_same_day_bets=10,
            correlation_rho=0.2,
        )
        # Both should be capped at $100 because the pre-cap kelly_raw is
        # huge and a 1/sqrt(3) haircut isn't enough to bring it below $100.
        assert baseline.dollars == pytest.approx(100.0)
        assert haircut_applied.dollars == pytest.approx(100.0)
        assert baseline.capped is True
        assert haircut_applied.capped is True

    def test_haircut_shrinks_below_cap_when_kelly_modest(self) -> None:
        """When the un-capped Kelly position is in the same neighborhood as
        the cap, the haircut brings the final size below the cap. Confirms
        the multiplicative composition with the cap is correct."""
        # max_pct=0.05 of $10k = $500 cap; baseline Kelly here lands at
        # $300 (< cap), so haircut should freely shrink it.
        baseline = fractional_kelly(
            model_prob=0.60,
            market_price=0.50,
            kelly_fraction=0.15,
            max_pct=0.05,
            bankroll=10_000.0,
        )
        haircut_applied = fractional_kelly(
            model_prob=0.60,
            market_price=0.50,
            kelly_fraction=0.15,
            max_pct=0.05,
            bankroll=10_000.0,
            n_existing_same_day_bets=10,
            correlation_rho=0.2,
        )
        assert baseline.dollars == pytest.approx(300.0)
        assert baseline.capped is False
        assert haircut_applied.dollars < baseline.dollars
        assert haircut_applied.capped is False
        # Check ratio
        assert haircut_applied.dollars / baseline.dollars == pytest.approx(
            1.0 / math.sqrt(3.0), abs=1e-3
        )
