"""Tests for Kelly criterion position sizing (python/analytics/sizing.py)."""

from __future__ import annotations

import dataclasses

import pytest

from python.analytics.sizing import SizingResult, dollars_to_contracts, fractional_kelly
from python.common import Side

# ── SizingResult dataclass ─────────────────────────────────────────────────


class TestSizingResultDataclass:
    def test_is_frozen(self):
        r = SizingResult(
            side=Side.YES, dollars=100.0, kelly_raw=0.1,
            kelly_fraction_used=0.15, capped=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.dollars = 999.0  # type: ignore[misc]

    def test_equality(self):
        a = SizingResult(
            side=Side.YES, dollars=50.0, kelly_raw=0.1,
            kelly_fraction_used=0.15, capped=False,
        )
        b = SizingResult(
            side=Side.YES, dollars=50.0, kelly_raw=0.1,
            kelly_fraction_used=0.15, capped=False,
        )
        assert a == b

    def test_fields_accessible(self):
        r = SizingResult(
            side=Side.NO, dollars=200.0, kelly_raw=0.3,
            kelly_fraction_used=0.25, capped=True,
        )
        assert isinstance(r.side, Side)
        assert isinstance(r.dollars, float)
        assert isinstance(r.kelly_raw, float)
        assert isinstance(r.kelly_fraction_used, float)
        assert isinstance(r.capped, bool)


# ── Edge detection ─────────────────────────────────────────────────────────


class TestFractionalKellyEdgeDetection:
    def test_edge_below_threshold(self):
        # edge = |0.500 - 0.5005| = 0.0005 < 0.001
        r = fractional_kelly(0.500, 0.5005)
        assert r.side == Side.NONE
        assert r.dollars == 0.0

    def test_edge_just_under_threshold(self):
        # edge = |0.500 - 0.5009| = 0.0009 < 0.001
        r = fractional_kelly(0.500, 0.5009)
        assert r.side == Side.NONE

    def test_edge_exactly_at_threshold(self):
        # edge = |0.501 - 0.500| = 0.001 — NOT < 0.001
        r = fractional_kelly(0.501, 0.500)
        assert r.side != Side.NONE

    def test_equal_probabilities(self):
        r = fractional_kelly(0.50, 0.50)
        assert r.side == Side.NONE
        assert r.dollars == 0.0


# ── Side selection ─────────────────────────────────────────────────────────


class TestFractionalKellySideSelection:
    def test_model_higher_selects_yes(self):
        r = fractional_kelly(0.70, 0.50)
        assert r.side == Side.YES

    def test_model_lower_selects_no(self):
        r = fractional_kelly(0.30, 0.50)
        assert r.side == Side.NO

    def test_yes_uses_market_price(self):
        # YES: price=0.50, win=0.70, odds=1.0, kelly_raw=0.40
        # dollars = 0.40 * 0.15 * 10000 = 600, capped to 500
        r = fractional_kelly(0.70, 0.50, kelly_fraction=0.15, max_pct=0.05, bankroll=10_000)
        assert r.side == Side.YES
        assert r.kelly_raw == pytest.approx(0.4)
        assert r.dollars == 500.0
        assert r.capped

    def test_no_uses_complement(self):
        # NO: price=1-0.50=0.50, win=1-0.30=0.70, same math as YES mirror
        r = fractional_kelly(0.30, 0.50, kelly_fraction=0.15, max_pct=0.05, bankroll=10_000)
        assert r.side == Side.NO
        assert r.kelly_raw == pytest.approx(0.4)
        assert r.dollars == 500.0
        assert r.capped


# ── Price boundaries ───────────────────────────────────────────────────────


class TestFractionalKellyPriceBoundaries:
    def test_market_price_zero_yes(self):
        # model > market → YES, price=0.0 → guard returns NONE
        r = fractional_kelly(0.50, 0.0)
        assert r.side == Side.NONE

    def test_market_price_one_no(self):
        # model < market → NO, price=1-1=0 → guard returns NONE
        r = fractional_kelly(0.50, 1.0)
        assert r.side == Side.NONE

    def test_model_zero_price_one(self):
        # model < market → NO, price=1-1=0 → guard returns NONE
        r = fractional_kelly(0.0, 1.0)
        assert r.side == Side.NONE

    def test_model_zero_price_near_zero(self):
        # model < market → NO, price=0.999, win_prob=1.0
        r = fractional_kelly(0.0, 0.001)
        assert r.side == Side.NO
        assert r.dollars > 0

    def test_price_near_one_yes(self):
        # model > market → YES, price=0.99, small odds
        r = fractional_kelly(0.999, 0.99)
        assert r.side == Side.YES
        assert r.dollars > 0

    def test_price_near_zero_no(self):
        # model < market → NO, price=1-0.01=0.99
        r = fractional_kelly(0.001, 0.01)
        assert r.side == Side.NO
        assert r.dollars > 0


# ── Kelly math ─────────────────────────────────────────────────────────────


class TestFractionalKellyMath:
    def test_hand_computed_uncapped(self):
        # YES: price=0.50, win=0.55, odds=1.0
        # kelly_raw = (0.55 - 0.45)/1.0 = 0.10
        # position = 0.10 * 0.15 * 10000 = 150, max=1000 → not capped
        r = fractional_kelly(0.55, 0.50, kelly_fraction=0.15, max_pct=0.10, bankroll=10_000)
        assert r.kelly_raw == pytest.approx(0.1)
        assert r.dollars == pytest.approx(150.0)
        assert not r.capped

    def test_hand_computed_capped(self):
        # YES: price=0.50, win=0.80, odds=1.0
        # kelly_raw = (0.80 - 0.20)/1.0 = 0.60
        # position = 0.60 * 0.50 * 10000 = 3000, max=500 → capped
        r = fractional_kelly(0.80, 0.50, kelly_fraction=0.50, max_pct=0.05, bankroll=10_000)
        assert r.kelly_raw == pytest.approx(0.6)
        assert r.dollars == 500.0
        assert r.capped

    def test_hand_computed_no_side(self):
        # NO: price=0.40, win=0.80, odds=1.5
        # kelly_raw = (0.80*1.5 - 0.20)/1.5 = 1.0/1.5 = 0.666667
        # position = 0.666667 * 0.15 * 10000 = 1000, max=500 → capped
        r = fractional_kelly(0.20, 0.60, kelly_fraction=0.15, bankroll=10_000)
        assert r.side == Side.NO
        assert r.kelly_raw == pytest.approx(0.666667, abs=1e-5)
        assert r.dollars == 500.0
        assert r.capped

    @pytest.mark.parametrize(
        "prob,price",
        [
            (0.5, 0.5),
            (0.9, 0.1),
            (0.1, 0.9),
            (0.51, 0.50),
            (0.99, 0.01),
            (0.01, 0.99),
            (0.0, 0.5),
            (1.0, 0.5),
            (0.5, 0.0),
            (0.5, 1.0),
        ],
    )
    def test_kelly_raw_non_negative(self, prob: float, price: float):
        r = fractional_kelly(prob, price)
        assert r.kelly_raw >= 0

    def test_dollars_rounded_two_decimals(self):
        r = fractional_kelly(0.55, 0.50, bankroll=3333.33)
        assert r.dollars == round(r.dollars, 2)

    def test_kelly_raw_rounded_six_decimals(self):
        # odds = 0.70/0.30 = 2.333…  →  irrational kelly_raw
        r = fractional_kelly(0.55, 0.30)
        assert r.kelly_raw == round(r.kelly_raw, 6)

    def test_symmetry_yes_no(self):
        yes = fractional_kelly(0.70, 0.50)
        no = fractional_kelly(0.30, 0.50)
        assert yes.dollars == no.dollars
        assert yes.kelly_raw == no.kelly_raw


# ── Scaling behaviour ──────────────────────────────────────────────────────


class TestFractionalKellyScaling:
    def test_kelly_fraction_linear(self):
        r1 = fractional_kelly(0.60, 0.50, kelly_fraction=0.10, max_pct=1.0)
        r2 = fractional_kelly(0.60, 0.50, kelly_fraction=0.20, max_pct=1.0)
        assert r2.dollars == pytest.approx(r1.dollars * 2)

    def test_bankroll_linear(self):
        r1 = fractional_kelly(0.60, 0.50, bankroll=5_000, max_pct=1.0)
        r2 = fractional_kelly(0.60, 0.50, bankroll=10_000, max_pct=1.0)
        assert r2.dollars == pytest.approx(r1.dollars * 2)

    def test_cap_at_exact_value(self):
        r = fractional_kelly(0.90, 0.50, max_pct=0.05, bankroll=10_000)
        assert r.capped
        assert r.dollars == 0.05 * 10_000

    def test_zero_bankroll(self):
        r = fractional_kelly(0.70, 0.50, bankroll=0)
        assert r.dollars == 0.0

    def test_large_bankroll(self):
        r = fractional_kelly(0.70, 0.50, bankroll=1_000_000)
        assert r.capped
        assert r.dollars == 0.05 * 1_000_000

    def test_kelly_fraction_zero(self):
        r = fractional_kelly(0.70, 0.50, kelly_fraction=0)
        assert r.dollars == 0.0

    def test_max_pct_one(self):
        r = fractional_kelly(0.70, 0.50, max_pct=1.0)
        assert not r.capped


# ── dollars_to_contracts ───────────────────────────────────────────────────


class TestDollarsToContracts:
    def test_yes_basic(self):
        assert dollars_to_contracts(100, 0.50, Side.YES) == 200

    def test_no_basic(self):
        assert dollars_to_contracts(100, 0.50, Side.NO) == 200

    def test_no_complement(self):
        # complement = 1 - 0.80 = 0.20 → 100/0.20 = 500
        assert dollars_to_contracts(100, 0.80, Side.NO) == 500

    def test_truncation(self):
        # 100 / 0.30 = 333.33… → int truncates to 333
        assert dollars_to_contracts(100, 0.30, Side.YES) == 333

    def test_minimum_one(self):
        # 0.01 / 0.50 = 0.02 → int=0 → max(1,0) = 1
        assert dollars_to_contracts(0.01, 0.50, Side.YES) == 1

    def test_zero_dollars(self):
        assert dollars_to_contracts(0, 0.50, Side.YES) == 0

    def test_negative_dollars(self):
        assert dollars_to_contracts(-10, 0.50, Side.YES) == 0

    def test_price_zero_yes(self):
        # price=0 → guard returns 0
        assert dollars_to_contracts(100, 0.0, Side.YES) == 0

    def test_price_one_no(self):
        # complement = 0 → guard returns 0
        assert dollars_to_contracts(100, 1.0, Side.NO) == 0

    def test_price_zero_no(self):
        # complement = 1.0 → 100/1.0 = 100
        assert dollars_to_contracts(100, 0.0, Side.NO) == 100

    def test_none_side(self):
        # Side.NONE → always returns 0 (no valid side to price)
        assert dollars_to_contracts(100, 0.50, Side.NONE) == 0
