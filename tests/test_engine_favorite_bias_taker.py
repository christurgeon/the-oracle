from __future__ import annotations

from python.execution.engine import compute_limit_cents


def test_taker_crosses_to_ask_on_wide_spread():
    # 6c spread; with a high cross threshold the limit must be the ask
    cents = compute_limit_cents(
        order_price_dollars=0.55,
        side_bid_cents=53,
        side_ask_cents=59,
        cross_to_ask_max_spread_cents=100,
    )
    assert cents == 59  # ask


def test_maker_default_does_not_cross_wide_spread():
    # 6c spread with threshold=1 -> rests inside the spread (maker), not at ask
    # spread=6 > 1, so compute_limit_cents falls to maker path:
    #   raw=55.0, inside_low=54, inside_high=58, capped=55 -> returns 55
    cents = compute_limit_cents(
        order_price_dollars=0.55,
        side_bid_cents=53,
        side_ask_cents=59,
        cross_to_ask_max_spread_cents=1,
    )
    assert cents == 55  # maker price: inside spread, biased toward bid
    assert cents != 59  # NOT at the ask


def test_metadata_flag_drives_threshold():
    # documents the caller contract: execution_taker -> 100c threshold -> ask
    md = {"execution_taker": True}
    thr = 100 if md.get("execution_taker") else 1
    assert (
        compute_limit_cents(
            order_price_dollars=0.55,
            side_bid_cents=53,
            side_ask_cents=59,
            cross_to_ask_max_spread_cents=thr,
        )
        == 59
    )
