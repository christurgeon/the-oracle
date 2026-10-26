"""Tests for Kalshi fee calculations (python/analytics/fees.py)."""

from __future__ import annotations

import pytest

from python.analytics.fees import (
    MAKER_FEE_FREE_SERIES,
    buy_side_price,
    charges_maker_fee,
    fee_per_contract,
    order_fee,
)
from python.utils.ticker_dates import SERIES_PREFIX_TO_CITY_VAR


class TestBuySidePrice:
    def test_yes_side(self):
        assert buy_side_price(0.70, 0.50) == 0.50

    def test_no_side(self):
        assert buy_side_price(0.30, 0.50) == 0.50

    def test_yes_asymmetric(self):
        assert buy_side_price(0.80, 0.60) == 0.60

    def test_no_asymmetric(self):
        assert buy_side_price(0.20, 0.60) == 0.40

    def test_equal_returns_no(self):
        # model == market → not > market, so returns 1 - market (NO side)
        assert buy_side_price(0.50, 0.50) == 0.50


class TestFeePerContract:
    """Test conservative per-contract fee (ceil applied per-contract)."""

    def test_midpoint_taker(self):
        # P=0.50: 0.07 * 0.50 * 0.50 * 100 = 1.75 → ceil = 2 cents
        assert fee_per_contract(0.50) == 0.02

    def test_midpoint_maker(self):
        # P=0.50: 0.0175 * 0.50 * 0.50 * 100 = 0.4375 → ceil = 1 cent
        assert fee_per_contract(0.50, is_maker=True) == 0.01

    def test_extreme_low_price(self):
        # P=0.05: 0.07 * 0.05 * 0.95 * 100 = 0.3325 → ceil = 1 cent
        assert fee_per_contract(0.05) == 0.01

    def test_extreme_high_price(self):
        # P=0.95: 0.07 * 0.95 * 0.05 * 100 = 0.3325 → ceil = 1 cent
        assert fee_per_contract(0.95) == 0.01

    def test_symmetry(self):
        # P*(1-P) is symmetric: fee at 0.20 == fee at 0.80
        assert fee_per_contract(0.20) == fee_per_contract(0.80)
        assert fee_per_contract(0.10) == fee_per_contract(0.90)
        assert fee_per_contract(0.30) == fee_per_contract(0.70)

    def test_price_zero(self):
        assert fee_per_contract(0.0) == 0.0

    def test_price_one(self):
        assert fee_per_contract(1.0) == 0.0

    def test_negative_price(self):
        assert fee_per_contract(-0.1) == 0.0

    def test_index_market_taker(self):
        # P=0.50: 0.035 * 0.50 * 0.50 * 100 = 0.875 → ceil = 1 cent
        assert fee_per_contract(0.50, is_index_market=True) == 0.01

    def test_index_market_maker_same(self):
        # Maker multiplier is the same regardless of index
        assert fee_per_contract(0.50, is_maker=True, is_index_market=True) == 0.01

    def test_at_20_cents(self):
        # P=0.20: 0.07 * 0.20 * 0.80 * 100 = 1.12 → ceil = 2 cents
        assert fee_per_contract(0.20) == 0.02

    def test_at_10_cents(self):
        # P=0.10: 0.07 * 0.10 * 0.90 * 100 = 0.63 → ceil = 1 cent
        assert fee_per_contract(0.10) == 0.01

    def test_maker_at_20_cents(self):
        # P=0.20: 0.0175 * 0.20 * 0.80 * 100 = 0.28 → ceil = 1 cent
        assert fee_per_contract(0.20, is_maker=True) == 0.01

    def test_always_non_negative(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert fee_per_contract(p) >= 0.0
            assert fee_per_contract(p, is_maker=True) >= 0.0

    def test_taker_gte_maker(self):
        for p in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
            assert fee_per_contract(p) >= fee_per_contract(p, is_maker=True)


class TestOrderFee:
    """Test exact order-level fee (ceil applied to total)."""

    def test_single_contract_matches_per_contract(self):
        # For 1 contract, order_fee == fee_per_contract
        for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
            assert order_fee(p, 1) == fee_per_contract(p)

    def test_multi_contract_lte_per_contract_sum(self):
        # order_fee(p, n) <= n * fee_per_contract(p) due to ceil difference
        for p in [0.10, 0.25, 0.50]:
            for n in [5, 10, 25, 100]:
                assert order_fee(p, n) <= n * fee_per_contract(p)

    def test_10_contracts_at_10_cents(self):
        # 0.07 * 10 * 0.10 * 0.90 * 100 = 6.3 → ceil = 7 cents
        assert order_fee(0.10, 10) == 0.07

    def test_zero_contracts(self):
        assert order_fee(0.50, 0) == 0.0

    def test_negative_contracts(self):
        assert order_fee(0.50, -5) == 0.0

    def test_price_boundary(self):
        assert order_fee(0.0, 10) == 0.0
        assert order_fee(1.0, 10) == 0.0

    def test_maker_multi_contract(self):
        # P=0.50, 10 contracts: 0.0175 * 10 * 0.25 * 100 = 4.375 → ceil = 5 cents
        assert order_fee(0.50, 10, is_maker=True) == 0.05


class TestSeriesAwareMakerFee:
    """Weather series are ``fee_type: quadratic`` (taker-only), verified via GET /series."""

    def test_weather_maker_fill_is_free(self):
        assert fee_per_contract(0.50, is_maker=True, ticker="KXHIGHNY") == 0.0
        assert order_fee(0.50, 10, is_maker=True, ticker="KXLOWTNYC-26OCT02-T55") == 0.0
        assert order_fee(0.30, 7, is_maker=True, ticker="KXRAIN") == 0.0

    def test_weather_taker_fee_unchanged_by_ticker(self):
        for p in [0.05, 0.20, 0.50, 0.70, 0.95]:
            assert fee_per_contract(p, ticker="KXHIGHMIA-26OCT02-B90.5") == fee_per_contract(p)
            assert order_fee(p, 59, ticker="KXLOWTLAX") == order_fee(p, 59)

    def test_unknown_or_missing_ticker_keeps_maker_fee(self):
        assert fee_per_contract(0.50, is_maker=True) == 0.01
        assert fee_per_contract(0.50, is_maker=True, ticker=None) == 0.01
        assert fee_per_contract(0.50, is_maker=True, ticker="") == 0.01
        assert order_fee(0.50, 10, is_maker=True, ticker="KXNBASERIES-26") == 0.05

    def test_prefix_lookalikes_are_not_maker_free(self):
        assert charges_maker_fee("KXRAINNYCM-26OCT")
        assert charges_maker_fee("KXHIGHFOO-26OCT02")
        assert not charges_maker_fee("KXRAIN-26OCT02-NYC")

    def test_covers_every_traded_temperature_series(self):
        assert set(SERIES_PREFIX_TO_CITY_VAR) | {"KXRAIN"} == MAKER_FEE_FREE_SERIES


class TestKellyWithFees:
    """Test that fractional_kelly correctly accounts for fees."""

    def test_fee_reduces_kelly_size(self):
        from python.analytics.sizing import fractional_kelly

        no_fee = fractional_kelly(0.60, 0.50, max_pct=1.0)
        with_fee = fractional_kelly(0.60, 0.50, max_pct=1.0, fee_per_contract=0.02)
        assert with_fee.dollars < no_fee.dollars
        assert with_fee.kelly_raw < no_fee.kelly_raw

    def test_zero_fee_unchanged(self):
        from python.analytics.sizing import fractional_kelly

        no_fee = fractional_kelly(0.60, 0.50, max_pct=1.0)
        zero_fee = fractional_kelly(0.60, 0.50, max_pct=1.0, fee_per_contract=0.0)
        assert no_fee.dollars == zero_fee.dollars

    def test_fee_can_kill_trade(self):
        # Tiny edge with large fee → no trade
        from python.analytics.sizing import fractional_kelly

        # edge = 0.501 - 0.50 = 0.001; fee = 0.02 makes effective cost 0.52
        # odds = 0.48/0.52 ≈ 0.923; kelly = (0.501*0.923 - 0.499)/0.923 < 0
        result = fractional_kelly(0.501, 0.50, fee_per_contract=0.02)
        assert result.dollars == 0.0


class TestBinaryContractPnlWithFees:
    """Test that binary_contract_pnl deducts fees."""

    def test_win_with_fee(self):
        from python.analytics.sizing import binary_contract_pnl
        from python.common import Side

        # Win: (1 - 0.50) * 10 - 0.20 = 4.80
        pnl = binary_contract_pnl(Side.YES, 0.50, 10, "yes", total_fee=0.20)
        assert pnl == pytest.approx(4.80)

    def test_loss_with_fee(self):
        from python.analytics.sizing import binary_contract_pnl
        from python.common import Side

        # Loss: -(0.50 * 10) - 0.20 = -5.20
        pnl = binary_contract_pnl(Side.YES, 0.50, 10, "no", total_fee=0.20)
        assert pnl == pytest.approx(-5.20)

    def test_zero_fee_unchanged(self):
        from python.analytics.sizing import binary_contract_pnl
        from python.common import Side

        pnl_no_fee = binary_contract_pnl(Side.YES, 0.50, 10, "yes")
        pnl_zero = binary_contract_pnl(Side.YES, 0.50, 10, "yes", total_fee=0.0)
        assert pnl_no_fee == pnl_zero
