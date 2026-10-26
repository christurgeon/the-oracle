"""Tests for the deterministic limit-price helper.

Goal: eliminate banker's-rounding parity from the order-pricing path.  The
helper must:
  - Return an integer cents value.
  - Stay strictly inside the spread when spread >= 2c.
  - Cross to the ask (= taker) on 1c spreads, deterministically.
  - Handle locked/crossed books by returning the ask.
"""

from __future__ import annotations

from python.execution.engine import compute_limit_cents


class TestOneCentSpread:
    """1c spread: deterministic taker (cross to ask)."""

    def test_yes_1c_low(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.325, side_bid_cents=32, side_ask_cents=33
            )
            == 33
        )

    def test_yes_1c_mid(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.455, side_bid_cents=45, side_ask_cents=46
            )
            == 46
        )

    def test_no_1c_high(self):
        # NO-side prices: bid=67, ask=68 (NO-denominated). Rule still says cross.
        assert (
            compute_limit_cents(
                order_price_dollars=0.675, side_bid_cents=67, side_ask_cents=68
            )
            == 68
        )


class TestTwoCentSpread:
    """2c spread: post inside, biased toward bid (maker)."""

    def test_yes_2c_clamps_to_only_inside_cent(self):
        # bid 30, ask 32 -> only inside cent is 31.
        assert (
            compute_limit_cents(
                order_price_dollars=0.31, side_bid_cents=30, side_ask_cents=32
            )
            == 31
        )

    def test_yes_2c_high_input_clamps_to_inside(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.319, side_bid_cents=30, side_ask_cents=32
            )
            == 31
        )


class TestWideSpread:
    """>=3c spread: respect intended midpoint, biased toward bid (truncates)."""

    def test_5c_spread_truncates_toward_bid(self):
        # bid 30, ask 35, mid = 32.5 -> 32 (toward bid).
        assert (
            compute_limit_cents(
                order_price_dollars=0.325, side_bid_cents=30, side_ask_cents=35
            )
            == 32
        )

    def test_10c_spread_floors_correctly(self):
        # bid 20, ask 30, requested 24.7 -> 24.
        assert (
            compute_limit_cents(
                order_price_dollars=0.247, side_bid_cents=20, side_ask_cents=30
            )
            == 24
        )


class TestCrossToAskMaxSpreadCents:
    """``cross_to_ask_max_spread_cents`` flag: widen the cross-to-ask band.

    Default is 1 (only 1c spreads cross). Production may set to 2 to also cross
    2c spreads (modified Lever A) when the maker resting-at-bid path has low
    fill rates and adverse selection is not present.
    """

    def test_2c_default_threshold_1_still_posts_inside(self):
        # Default behavior unchanged: 2c spread posts at inside cent.
        assert (
            compute_limit_cents(
                order_price_dollars=0.31, side_bid_cents=30, side_ask_cents=32
            )
            == 31
        )

    def test_2c_threshold_2_crosses_to_ask(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.31,
                side_bid_cents=30,
                side_ask_cents=32,
                cross_to_ask_max_spread_cents=2,
            )
            == 32
        )

    def test_3c_threshold_2_still_posts_inside(self):
        # Threshold of 2 must NOT pull >=3c spreads into the taker path.
        assert (
            compute_limit_cents(
                order_price_dollars=0.325,
                side_bid_cents=30,
                side_ask_cents=35,
                cross_to_ask_max_spread_cents=2,
            )
            == 32
        )

    def test_1c_always_crosses_regardless_of_threshold(self):
        for t in (1, 2):
            assert (
                compute_limit_cents(
                    order_price_dollars=0.325,
                    side_bid_cents=32,
                    side_ask_cents=33,
                    cross_to_ask_max_spread_cents=t,
                )
                == 33
            )


class TestEdgeCases:
    """Locked/crossed/zero-spread books."""

    def test_locked_book_returns_ask(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.50, side_bid_cents=50, side_ask_cents=50
            )
            == 50
        )

    def test_crossed_book_returns_ask(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.50, side_bid_cents=51, side_ask_cents=50
            )
            == 50
        )

    def test_order_price_above_ask_clamps_to_inside_high(self):
        # Strategy hands us a price above the ask on a wide spread -> clamp to ask-1.
        assert (
            compute_limit_cents(
                order_price_dollars=0.99, side_bid_cents=20, side_ask_cents=30
            )
            == 29
        )

    def test_order_price_below_bid_clamps_to_inside_low(self):
        assert (
            compute_limit_cents(
                order_price_dollars=0.01, side_bid_cents=20, side_ask_cents=30
            )
            == 21
        )
