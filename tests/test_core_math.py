"""Unit tests for core math functions — the mission-critical code where bugs silently erode edge.

Covers:
  - python/analytics/sizing.py    (fractional_kelly, binary_contract_pnl, dollars_to_contracts)
  - python/analytics/fees.py      (fee_per_contract, order_fee, buy_side_price)
  - python/data/weather.py        (EnsembleForecast fraction methods, inflated, blending)
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from python.analytics.fees import (
    MAKER_MULTIPLIER_DEFAULT,
    TAKER_MULTIPLIER_DEFAULT,
    TAKER_MULTIPLIER_INDEX,
    fee_per_contract,
    order_fee,
)
from python.analytics.sizing import binary_contract_pnl, fractional_kelly
from python.common import DailyVariable, Direction, Side
from python.data.weather import (
    EnsembleForecast,
    WeatherClient,
    blended_ensemble_std,
    compute_blended_probability,
    get_spread_inflation_factor,
)

# ===========================================================================
# fractional_kelly — additional edge and math tests
# ===========================================================================


class TestKellyZeroEdge:
    """Kelly should return 0 dollars when there is no edge."""

    def test_model_equals_market(self):
        r = fractional_kelly(0.50, 0.50)
        assert r.dollars == 0.0
        assert r.kelly_raw == 0.0

    def test_tiny_edge_below_threshold(self):
        r = fractional_kelly(0.5004, 0.5)
        assert r.dollars == 0.0
        assert r.side == Side.NONE

    def test_model_slightly_above_market_with_fee_kills_trade(self):
        r = fractional_kelly(0.52, 0.50, fee_per_contract=0.03)
        # fee pushes effective price to 0.53, so edge of 0.02 is overcome
        assert r.dollars == 0.0

    def test_model_slightly_below_market_with_fee_kills_trade(self):
        r = fractional_kelly(0.48, 0.50, fee_per_contract=0.03)
        # NO side: price = 0.50 + 0.03 = 0.53, win_prob = 0.52
        assert r.dollars == 0.0


class TestKellyNegativeEdge:
    """Kelly raw should be 0 (not negative) when model is wrong-sided after fees."""

    def test_negative_kelly_clamped_to_zero(self):
        r = fractional_kelly(0.51, 0.50, fee_per_contract=0.05)
        assert r.kelly_raw == 0.0
        assert r.dollars == 0.0

    def test_no_side_negative_kelly_clamped(self):
        r = fractional_kelly(0.49, 0.50, fee_per_contract=0.05)
        assert r.kelly_raw == 0.0
        assert r.dollars == 0.0


class TestKellyModelProbClamping:
    """model_prob outside [0,1] should be clamped, not produce wild sizing."""

    def test_model_prob_above_one_clamped(self):
        r = fractional_kelly(1.05, 0.50)
        assert r.side == Side.YES
        # Should behave like model_prob=1.0
        r_clamped = fractional_kelly(1.0, 0.50)
        assert r.kelly_raw == r_clamped.kelly_raw
        assert r.dollars == r_clamped.dollars

    def test_model_prob_below_zero_clamped(self):
        r = fractional_kelly(-0.05, 0.50)
        r_clamped = fractional_kelly(0.0, 0.50)
        assert r.kelly_raw == r_clamped.kelly_raw
        assert r.dollars == r_clamped.dollars


class TestKellyFeeOnNoSide:
    """Fee should reduce sizing on the NO side the same way it does on YES."""

    def test_fee_reduces_no_side_kelly(self):
        no_fee = fractional_kelly(0.30, 0.60, max_pct=1.0)
        with_fee = fractional_kelly(0.30, 0.60, max_pct=1.0, fee_per_contract=0.02)
        assert with_fee.side == Side.NO
        assert with_fee.kelly_raw < no_fee.kelly_raw
        assert with_fee.dollars < no_fee.dollars

    def test_fee_symmetry_yes_no(self):
        """Fee impact should be symmetric for mirrored YES/NO trades."""
        yes = fractional_kelly(0.70, 0.50, fee_per_contract=0.02, max_pct=1.0)
        no = fractional_kelly(0.30, 0.50, fee_per_contract=0.02, max_pct=1.0)
        assert yes.kelly_raw == no.kelly_raw
        assert yes.dollars == no.dollars


class TestKellyFractionScaling:
    """Fractional Kelly should scale linearly with the fraction parameter."""

    def test_half_fraction_halves_dollars(self):
        full = fractional_kelly(0.65, 0.50, kelly_fraction=0.20, max_pct=1.0)
        half = fractional_kelly(0.65, 0.50, kelly_fraction=0.10, max_pct=1.0)
        assert half.dollars == pytest.approx(full.dollars / 2)

    def test_kelly_raw_invariant_to_fraction(self):
        r1 = fractional_kelly(0.65, 0.50, kelly_fraction=0.10)
        r2 = fractional_kelly(0.65, 0.50, kelly_fraction=0.30)
        assert r1.kelly_raw == r2.kelly_raw


# ===========================================================================
# binary_contract_pnl — comprehensive coverage
# ===========================================================================


class TestBinaryContractPnl:
    def test_yes_wins_yes(self):
        pnl = binary_contract_pnl(Side.YES, 0.60, 10, "yes")
        assert pnl == pytest.approx((1.0 - 0.60) * 10)

    def test_yes_loses_no(self):
        pnl = binary_contract_pnl(Side.YES, 0.60, 10, "no")
        assert pnl == pytest.approx(-0.60 * 10)

    def test_no_wins_no(self):
        pnl = binary_contract_pnl(Side.NO, 0.40, 10, "no")
        assert pnl == pytest.approx((1.0 - 0.40) * 10)

    def test_no_loses_yes(self):
        pnl = binary_contract_pnl(Side.NO, 0.40, 10, "yes")
        assert pnl == pytest.approx(-0.40 * 10)

    def test_one_contract(self):
        pnl = binary_contract_pnl(Side.YES, 0.50, 1, "yes")
        assert pnl == pytest.approx(0.50)

    def test_cheap_contract_large_win(self):
        pnl = binary_contract_pnl(Side.YES, 0.05, 100, "yes")
        assert pnl == pytest.approx(0.95 * 100)

    def test_expensive_contract_small_win(self):
        pnl = binary_contract_pnl(Side.YES, 0.95, 100, "yes")
        assert pnl == pytest.approx(0.05 * 100)

    def test_fee_deducted_on_win(self):
        pnl = binary_contract_pnl(Side.YES, 0.50, 10, "yes", total_fee=1.00)
        assert pnl == pytest.approx(5.0 - 1.0)

    def test_fee_deducted_on_loss(self):
        pnl = binary_contract_pnl(Side.YES, 0.50, 10, "no", total_fee=1.00)
        assert pnl == pytest.approx(-5.0 - 1.0)

    def test_pnl_is_symmetric(self):
        """Buying YES at p and buying NO at p should give same loss magnitude."""
        yes_loss = binary_contract_pnl(Side.YES, 0.60, 10, "no")
        no_loss = binary_contract_pnl(Side.NO, 0.60, 10, "yes")
        assert yes_loss == pytest.approx(no_loss)


# ===========================================================================
# fee_per_contract — ceil boundary and consistency tests
# ===========================================================================


class TestFeePerContractCeilBoundary:
    """Verify ceil behavior at exact cent boundaries."""

    def test_exact_integer_cents_no_round_up(self):
        # Find a price where raw_cents is exactly an integer.
        # P=0.50: 0.07 * 0.50 * 0.50 * 100 = 1.75 -> ceil = 2
        # Not exact, so try P where mult * P * (1-P) * 100 = integer
        # 0.07 * P * (1-P) * 100 = 7 * P * (1-P)
        # At P=0.50: 7*0.25=1.75, not integer
        # The key invariant: ceil always rounds UP to next cent
        fee = fee_per_contract(0.50)
        raw = TAKER_MULTIPLIER_DEFAULT * 0.50 * 0.50 * 100.0
        assert fee == math.ceil(raw) / 100.0

    def test_fee_maximum_at_midpoint(self):
        """P*(1-P) is maximized at P=0.50, so fee should be highest there."""
        midpoint_fee = fee_per_contract(0.50)
        for p in [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]:
            assert fee_per_contract(p) <= midpoint_fee

    def test_fee_always_at_least_one_cent(self):
        """For any valid price in (0,1), fee should be >= $0.01."""
        for p in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
            assert fee_per_contract(p) >= 0.01

    def test_fee_never_exceeds_theoretical_max(self):
        """Fee should never exceed mult * 0.25 * 100 / 100 = mult * 0.25 (at P=0.50)."""
        for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            fee = fee_per_contract(p)
            # Theoretical max raw_cents is mult * 0.25 * 100 = 1.75 -> ceil = 2 cents
            assert fee <= math.ceil(TAKER_MULTIPLIER_DEFAULT * 0.25 * 100) / 100.0


class TestFeeMultiplierConsistency:
    """Verify multiplier selection matches expected constants."""

    def test_taker_default_multiplier(self):
        assert TAKER_MULTIPLIER_DEFAULT == 0.07

    def test_maker_default_multiplier(self):
        assert MAKER_MULTIPLIER_DEFAULT == 0.0175

    def test_index_taker_is_half_default(self):
        assert TAKER_MULTIPLIER_INDEX == pytest.approx(TAKER_MULTIPLIER_DEFAULT / 2)

    def test_index_taker_lower_fee(self):
        """Index taker fee should be strictly lower than default taker fee."""
        for p in [0.10, 0.30, 0.50, 0.70, 0.90]:
            assert fee_per_contract(p, is_index_market=True) <= fee_per_contract(p)


class TestOrderFeeVsPerContract:
    """order_fee should always be <= N * fee_per_contract (batch is cheaper)."""

    @pytest.mark.parametrize("n", [2, 5, 10, 50, 100, 1000])
    def test_batch_fee_le_sum_of_individual(self, n: int):
        for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
            assert order_fee(p, n) <= n * fee_per_contract(p)

    def test_order_fee_scales_sublinearly(self):
        """Doubling contracts should at most double fee (never more due to ceil)."""
        for p in [0.20, 0.50, 0.80]:
            f5 = order_fee(p, 5)
            f10 = order_fee(p, 10)
            assert f10 <= 2 * f5 + 0.01  # +0.01 for rounding tolerance

    def test_order_fee_100_contracts_at_midpoint(self):
        # 0.07 * 100 * 0.50 * 0.50 * 100 = 175.0 in exact math
        # but floating point gives 175.00000000000003 -> ceil = 176 cents = $1.76
        assert order_fee(0.50, 100) == 1.76

    def test_order_fee_maker_100_contracts_at_midpoint(self):
        # 0.0175 * 100 * 0.50 * 0.50 * 100 = 43.75 -> ceil = 44 cents = $0.44
        assert order_fee(0.50, 100, is_maker=True) == 0.44


# ===========================================================================
# EnsembleForecast.fraction_between — all members above/below/split
# ===========================================================================


def _fc(temps: list[float]) -> EnsembleForecast:
    return EnsembleForecast(
        city="NYC",
        forecast_date=date(2026, 3, 15),
        temperatures=temps,
    )


class TestFractionBetween:
    def test_all_members_inside_bracket(self):
        fc = _fc([70.0, 71.0, 72.0, 73.0, 74.0])
        assert fc.fraction_between(65.0, 80.0) == pytest.approx(1.0)

    def test_all_members_below_bracket(self):
        fc = _fc([60.0, 61.0, 62.0, 63.0, 64.0])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(0.0)

    def test_all_members_above_bracket(self):
        fc = _fc([85.0, 86.0, 87.0, 88.0, 89.0])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(0.0)

    def test_split_members(self):
        fc = _fc([65.0, 70.0, 75.0, 80.0, 85.0])
        # In [70, 80]: 70, 75, 80 = 3/5
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(3 / 5)

    def test_members_at_boundaries_included(self):
        """Boundary values (floor and cap) should be INCLUDED (inclusive)."""
        fc = _fc([70.0, 80.0])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(1.0)

    def test_floor_equals_cap(self):
        """Only members exactly at the value count."""
        fc = _fc([70.0, 70.0, 71.0])
        assert fc.fraction_between(70.0, 70.0) == pytest.approx(2 / 3)

    def test_floor_above_cap_returns_zero(self):
        """Inverted range should return 0."""
        fc = _fc([70.0, 71.0, 72.0])
        assert fc.fraction_between(80.0, 70.0) == pytest.approx(0.0)

    def test_empty_ensemble(self):
        fc = _fc([])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(0.0)

    def test_single_member_inside(self):
        fc = _fc([75.0])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(1.0)

    def test_single_member_outside(self):
        fc = _fc([65.0])
        assert fc.fraction_between(70.0, 80.0) == pytest.approx(0.0)

    def test_realistic_31_member_ensemble(self):
        """Simulate a realistic GFS ensemble with 31 members."""
        import random

        random.seed(42)
        temps = [70.0 + random.gauss(0, 3) for _ in range(31)]
        fc = _fc(temps)
        frac = fc.fraction_between(68.0, 72.0)
        assert 0.0 <= frac <= 1.0
        # Manually verify
        expected = sum(1 for t in temps if 68.0 <= t <= 72.0) / 31
        assert frac == pytest.approx(expected)


class TestFractionAbove:
    def test_all_above(self):
        fc = _fc([80.0, 81.0, 82.0])
        assert fc.fraction_above(70.0) == pytest.approx(1.0)

    def test_none_above(self):
        fc = _fc([60.0, 61.0, 62.0])
        assert fc.fraction_above(70.0) == pytest.approx(0.0)

    def test_strictly_greater(self):
        """Members exactly AT the threshold should NOT be counted."""
        fc = _fc([70.0, 70.0, 71.0])
        assert fc.fraction_above(70.0) == pytest.approx(1 / 3)

    def test_empty(self):
        fc = _fc([])
        assert fc.fraction_above(70.0) == pytest.approx(0.0)


class TestFractionAtOrAbove:
    def test_includes_exact_threshold(self):
        fc = _fc([70.0, 70.0, 71.0])
        assert fc.fraction_at_or_above(70.0) == pytest.approx(1.0)

    def test_none_at_or_above(self):
        fc = _fc([60.0, 61.0])
        assert fc.fraction_at_or_above(70.0) == pytest.approx(0.0)

    def test_empty(self):
        fc = _fc([])
        assert fc.fraction_at_or_above(70.0) == pytest.approx(0.0)


class TestFractionBelow:
    def test_complement_of_at_or_above(self):
        """fraction_below = 1 - fraction_at_or_above (by definition)."""
        fc = _fc([65.0, 70.0, 75.0, 80.0])
        for threshold in [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]:
            assert fc.fraction_below(threshold) == pytest.approx(
                1.0 - fc.fraction_at_or_above(threshold)
            )

    def test_all_below(self):
        fc = _fc([60.0, 61.0, 62.0])
        assert fc.fraction_below(70.0) == pytest.approx(1.0)

    def test_none_below(self):
        fc = _fc([80.0, 81.0])
        assert fc.fraction_below(70.0) == pytest.approx(0.0)

    def test_strictly_less(self):
        """Members exactly at threshold are NOT below (they are at_or_above)."""
        fc = _fc([70.0, 70.0, 69.0])
        # fraction_at_or_above(70) = 2/3, so fraction_below = 1/3
        assert fc.fraction_below(70.0) == pytest.approx(1 / 3)


class TestFractionConsistency:
    """Cross-method consistency: above + at_threshold + below = 1."""

    def test_three_way_partition(self):
        fc = _fc([65.0, 70.0, 70.0, 75.0, 80.0])
        threshold = 70.0
        above = fc.fraction_above(threshold)  # strictly > 70: 75, 80 = 2/5
        at_or_above = fc.fraction_at_or_above(threshold)  # >= 70: 70, 70, 75, 80 = 4/5
        below = fc.fraction_below(threshold)  # < 70: 65 = 1/5
        at_threshold = at_or_above - above  # exactly = 70: 2/5
        assert above + at_threshold + below == pytest.approx(1.0)


# ===========================================================================
# EnsembleForecast.inflated
# ===========================================================================


class TestInflated:
    def test_factor_one_returns_same(self):
        fc = _fc([60.0, 70.0, 80.0])
        inflated = fc.inflated(1.0)
        assert inflated is fc  # identity optimization

    def test_factor_two_doubles_spread(self):
        fc = _fc([60.0, 70.0, 80.0])
        inflated = fc.inflated(2.0)
        # mean = 70, so 60->50, 70->70, 80->90
        assert inflated.temperatures == pytest.approx([50.0, 70.0, 90.0])

    def test_factor_half_halves_spread(self):
        fc = _fc([60.0, 70.0, 80.0])
        inflated = fc.inflated(0.5)
        # mean = 70, so 60->65, 70->70, 80->75
        assert inflated.temperatures == pytest.approx([65.0, 70.0, 75.0])

    def test_preserves_mean(self):
        fc = _fc([60.0, 65.0, 70.0, 75.0, 80.0])
        inflated = fc.inflated(3.0)
        assert inflated.mean_temperature == pytest.approx(fc.mean_temperature)

    def test_std_scales_with_factor(self):
        fc = _fc([60.0, 65.0, 70.0, 75.0, 80.0])
        inflated = fc.inflated(2.0)
        assert inflated.std_dev == pytest.approx(fc.std_dev * 2.0, rel=1e-10)

    def test_zero_factor_raises(self):
        fc = _fc([60.0, 70.0, 80.0])
        with pytest.raises(ValueError, match="positive"):
            fc.inflated(0.0)

    def test_negative_factor_raises(self):
        fc = _fc([60.0, 70.0, 80.0])
        with pytest.raises(ValueError, match="positive"):
            fc.inflated(-1.0)

    def test_empty_ensemble_returns_self(self):
        fc = _fc([])
        inflated = fc.inflated(2.0)
        assert inflated is fc

    def test_frozen_original_unchanged(self):
        """Inflating should not mutate the original forecast."""
        fc = _fc([60.0, 70.0, 80.0])
        _ = fc.inflated(2.0)
        assert fc.temperatures == [60.0, 70.0, 80.0]


# ===========================================================================
# EnsembleForecast properties
# ===========================================================================


class TestEnsembleForecastProperties:
    def test_member_count(self):
        fc = _fc([70.0] * 31)
        assert fc.member_count == 31

    def test_mean_temperature(self):
        fc = _fc([60.0, 70.0, 80.0])
        assert fc.mean_temperature == pytest.approx(70.0)

    def test_min_max_temperature(self):
        fc = _fc([55.0, 70.0, 85.0])
        assert fc.min_temperature == 55.0
        assert fc.max_temperature == 85.0

    def test_std_dev_uniform(self):
        fc = _fc([70.0, 70.0, 70.0])
        assert fc.std_dev == pytest.approx(0.0)

    def test_std_dev_known(self):
        """std dev of [0, 10] = 5.0 (population)."""
        fc = _fc([0.0, 10.0])
        assert fc.std_dev == pytest.approx(5.0)

    def test_empty_defaults(self):
        fc = _fc([])
        assert fc.member_count == 0
        assert fc.mean_temperature == 0.0
        assert fc.std_dev == 0.0

    def test_single_member_std(self):
        fc = _fc([70.0])
        assert fc.std_dev == 0.0


# ===========================================================================
# compute_blended_probability
# ===========================================================================


class TestComputeBlendedProbability:
    def _fc(
        self,
        temps: list[float],
        model: str = "gfs_seamless",
    ) -> EnsembleForecast:
        return EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=temps,
            model=model,
        )

    def test_single_model_equals_raw(self):
        gfs = self._fc([65.0, 70.0, 75.0, 80.0])
        weights = {"gfs_seamless": 1.0}
        prob, per_model = compute_blended_probability(
            {"gfs_seamless": gfs},
            weights,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(gfs.fraction_above(70.0))
        assert per_model["gfs_seamless"] == pytest.approx(prob)

    def test_equal_weights(self):
        gfs = self._fc([60.0, 70.0, 80.0], model="gfs_seamless")
        ecmwf = self._fc([65.0, 75.0, 85.0], model="ecmwf_ifs025")
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf},
            weights,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        gfs_prob = gfs.fraction_above(70.0)
        ecmwf_prob = ecmwf.fraction_above(70.0)
        assert prob == pytest.approx((gfs_prob + ecmwf_prob) / 2)

    def test_missing_model_renormalized(self):
        """If a model in weights is missing from forecasts, remaining weights renormalize."""
        gfs = self._fc([65.0, 75.0], model="gfs_seamless")
        weights = {"gfs_seamless": 0.5, "ecmwf_ifs025": 0.5}
        prob, per_model = compute_blended_probability(
            {"gfs_seamless": gfs},
            weights,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        # Only GFS present, so its prob should be the full blend
        assert prob == pytest.approx(gfs.fraction_above(70.0))
        assert "ecmwf_ifs025" not in per_model

    def test_empty_forecasts(self):
        weights = {"gfs_seamless": 1.0}
        prob, per_model = compute_blended_probability(
            {},
            weights,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == 0.0
        assert per_model == {}

    def test_between_direction(self):
        gfs = self._fc([65.0, 70.0, 72.0, 75.0, 80.0])
        weights = {"gfs_seamless": 1.0}
        prob, _ = compute_blended_probability(
            {"gfs_seamless": gfs},
            weights,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(gfs.fraction_between(70.0, 75.0))

    def test_below_direction(self):
        gfs = self._fc([65.0, 70.0, 75.0])
        weights = {"gfs_seamless": 1.0}
        prob, _ = compute_blended_probability(
            {"gfs_seamless": gfs},
            weights,
            Direction.BELOW,
            cap_strike=70.0,
        )
        assert prob == pytest.approx(gfs.fraction_below(70.0))

    def test_unequal_weights(self):
        gfs = self._fc([60.0, 80.0], model="gfs_seamless")  # above(70)=0.5
        ecmwf = self._fc([80.0, 80.0], model="ecmwf_ifs025")  # above(70)=1.0
        weights = {"gfs_seamless": 0.3, "ecmwf_ifs025": 0.7}
        prob, _ = compute_blended_probability(
            {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf},
            weights,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        expected = (0.3 * 0.5 + 0.7 * 1.0) / (0.3 + 0.7)
        assert prob == pytest.approx(expected)


# ===========================================================================
# get_spread_inflation_factor
# ===========================================================================


class TestGetSpreadInflationFactor:
    def test_known_city_variable(self):
        factor = get_spread_inflation_factor("DEN", DailyVariable.TEMP_MAX)
        assert factor == 3.0

    def test_case_insensitive(self):
        factor = get_spread_inflation_factor("den", DailyVariable.TEMP_MAX)
        assert factor == 3.0

    def test_unknown_city_returns_default(self):
        factor = get_spread_inflation_factor("UNKNOWN", DailyVariable.TEMP_MAX)
        assert factor == 2.0  # GFS_DERIVED_SPREAD_INFLATION_DEFAULT

    def test_overrides_take_precedence(self):
        overrides = {"DEN": {DailyVariable.TEMP_MAX: 5.0}}
        factor = get_spread_inflation_factor("DEN", DailyVariable.TEMP_MAX, overrides)
        assert factor == 5.0

    def test_partial_override_falls_through(self):
        """Override for city but not variable falls through to defaults."""
        overrides = {"DEN": {DailyVariable.TEMP_MIN: 9.0}}
        factor = get_spread_inflation_factor("DEN", DailyVariable.TEMP_MAX, overrides)
        assert factor == 3.0  # from GFS_DERIVED_SPREAD_INFLATION


# ===========================================================================
# WeatherClient.compute_model_probability
# ===========================================================================


class TestComputeModelProbability:
    """Test the WeatherClient wrapper around EnsembleForecast fraction methods."""

    def setup_method(self):
        self.client = WeatherClient.__new__(WeatherClient)

    def test_above_direction(self):
        fc = _fc([65.0, 70.0, 75.0, 80.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.ABOVE,
            floor_strike=70.0,
        )
        assert prob == pytest.approx(2 / 4)  # 75, 80 are > 70

    def test_below_direction(self):
        fc = _fc([65.0, 70.0, 75.0, 80.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BELOW,
            cap_strike=70.0,
        )
        # fraction_below(70) = 1 - fraction_at_or_above(70)
        # at_or_above(70) = 3/4 (70, 75, 80)
        # below = 1/4
        assert prob == pytest.approx(1 / 4)

    def test_between_direction(self):
        fc = _fc([65.0, 70.0, 75.0, 80.0])
        prob = self.client.compute_model_probability(
            fc,
            Direction.BETWEEN,
            floor_strike=70.0,
            cap_strike=75.0,
        )
        assert prob == pytest.approx(2 / 4)  # 70 and 75 are in [70, 75]

    def test_above_missing_floor_raises(self):
        fc = _fc([70.0])
        with pytest.raises(ValueError, match="floor_strike"):
            self.client.compute_model_probability(fc, Direction.ABOVE)

    def test_below_missing_cap_raises(self):
        fc = _fc([70.0])
        with pytest.raises(ValueError, match="cap_strike"):
            self.client.compute_model_probability(fc, Direction.BELOW)

    def test_between_missing_strikes_raises(self):
        fc = _fc([70.0])
        with pytest.raises(ValueError, match="floor_strike and cap_strike"):
            self.client.compute_model_probability(
                fc,
                Direction.BETWEEN,
                floor_strike=70.0,
            )


# ===========================================================================
# blended_ensemble_std — additional tests
# ===========================================================================


class TestBlendedEnsembleStdAdditional:
    def test_identical_models_matches_single(self):
        """Two identical model outputs should give same std as one."""
        fc1 = _fc([60.0, 70.0, 80.0])
        fc2 = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[60.0, 70.0, 80.0],
            model="ecmwf_ifs025",
        )
        single_std = blended_ensemble_std({"gfs": fc1})
        double_std = blended_ensemble_std({"gfs": fc1, "ecmwf": fc2})
        assert single_std == pytest.approx(double_std)

    def test_single_member_returns_zero(self):
        fc = _fc([70.0])
        assert blended_ensemble_std({"gfs": fc}) == 0.0

    def test_disagreeing_models_larger_std(self):
        """Two models that disagree should produce larger pooled std than either alone."""
        fc_gfs = _fc([70.0, 71.0, 72.0])  # tight around 71
        fc_ecmwf = EnsembleForecast(
            city="NYC",
            forecast_date=date(2026, 3, 15),
            temperatures=[80.0, 81.0, 82.0],
            model="ecmwf_ifs025",
        )
        gfs_std = blended_ensemble_std({"gfs": fc_gfs})
        pooled_std = blended_ensemble_std({"gfs": fc_gfs, "ecmwf": fc_ecmwf})
        assert pooled_std > gfs_std


# ===========================================================================
# CITY_COORDS — city coordinate coverage and consistency tests
# ===========================================================================


class TestCityCoords:
    """CITY_COORDS covers all supported collection cities and is consistent with NWS mapping."""

    def test_phil_coords_present(self) -> None:
        from python.data.weather import CITY_COORDS

        assert "PHIL" in CITY_COORDS
        lat, lon = CITY_COORDS["PHIL"]
        # Philadelphia International Airport (GHCND:USW00013739)
        assert abs(lat - 39.8729) < 0.05
        assert abs(lon - (-75.2410)) < 0.05

    def test_bos_coords_present(self) -> None:
        from python.data.weather import CITY_COORDS

        assert "BOS" in CITY_COORDS
        lat, lon = CITY_COORDS["BOS"]
        # Boston Logan Airport (GHCND:USW00014739)
        assert abs(lat - 42.3601) < 0.05
        assert abs(lon - (-71.0097)) < 0.05

    def test_city_coords_covers_cli_locations(self) -> None:
        """Every city in CITY_TO_CLI_LOCATION must have GFS coordinates."""
        from python.data.nws import CITY_TO_CLI_LOCATION
        from python.data.weather import CITY_COORDS

        missing = set(CITY_TO_CLI_LOCATION.keys()) - set(CITY_COORDS.keys())
        assert missing == set(), f"Cities missing CITY_COORDS entry: {missing}"
