"""
Kalshi fee calculations for prediction market contracts.

Fee formula: ceil(multiplier × C × P × (1 − P) × 100) / 100
where P = contract price in [0, 1], C = number of contracts, and
multiplier depends on maker/taker status and the series' fee type.

Taker fills pay 0.07 (0.035 on index series). Maker fills pay 0.0175 only
on series whose ``fee_type`` is ``quadratic_with_maker_fees``; series with
``fee_type: quadratic`` charge makers nothing. ``MAKER_FEE_FREE_SERIES``
lists the series verified as ``quadratic`` via ``GET /series``; any other
ticker (or none) is charged the maker fee, so an unverified series is never
priced more aggressively than before.

``order_fee`` ceils the whole order to the cent. Since ~June 2026 this
account's fills report fees at $0.0001 precision (Kalshi rounds per order to
the member's balance precision — docs: getting_started/fee_rounding), so
``order_fee`` overstates the billed fee by < 1¢ per order. Prefer the
Kalshi-reported ``fees_from_api`` for P&L whenever it is present. For
conservative per-contract estimates used in edge filtering, use
``fee_per_contract()``, which rounds up on a single-contract basis.

IMPORTANT: Kalshi can change fee multipliers or a series' fee type at any
time. Multipliers are from the Feb 2026 fee schedule; the maker-free series
set was verified 2026-10-01. Monitor https://kalshi.com/fee-schedule.
"""

from __future__ import annotations

import math

# Fee multipliers by market category.
# Index markets (KXINX, KXNDX) have halved taker fees.
TAKER_MULTIPLIER_DEFAULT = 0.07
MAKER_MULTIPLIER_DEFAULT = 0.0175
TAKER_MULTIPLIER_INDEX = 0.035

MAKER_FEE_FREE_SERIES: frozenset[str] = frozenset(
    {
        "KXHIGHAUS",
        "KXHIGHCHI",
        "KXHIGHDEN",
        "KXHIGHLAX",
        "KXHIGHMIA",
        "KXHIGHNY",
        "KXHIGHPHIL",
        "KXHIGHTATL",
        "KXHIGHTBOS",
        "KXHIGHTDAL",
        "KXHIGHTDC",
        "KXHIGHTHOU",
        "KXHIGHTLV",
        "KXHIGHTMIN",
        "KXHIGHTNOLA",
        "KXHIGHTOKC",
        "KXHIGHTPHX",
        "KXHIGHTSATX",
        "KXHIGHTSEA",
        "KXHIGHTSFO",
        "KXLOWTATL",
        "KXLOWTAUS",
        "KXLOWTBOS",
        "KXLOWTCHI",
        "KXLOWTDAL",
        "KXLOWTDC",
        "KXLOWTDEN",
        "KXLOWTHOU",
        "KXLOWTLAX",
        "KXLOWTLV",
        "KXLOWTMIA",
        "KXLOWTMIN",
        "KXLOWTNOLA",
        "KXLOWTNYC",
        "KXLOWTOKC",
        "KXLOWTPHIL",
        "KXLOWTPHX",
        "KXLOWTSATX",
        "KXLOWTSEA",
        "KXLOWTSFO",
        "KXRAIN",
    }
)


def charges_maker_fee(ticker: str | None) -> bool:
    """Whether a maker fill on ``ticker`` (series or market ticker) pays a fee.

    False only for series in ``MAKER_FEE_FREE_SERIES``; unknown or missing
    tickers return True (conservative).
    """
    if not ticker:
        return True
    return ticker.split("-", 1)[0] not in MAKER_FEE_FREE_SERIES


def _multiplier(is_maker: bool, is_index_market: bool, ticker: str | None) -> float:
    if is_maker:
        return MAKER_MULTIPLIER_DEFAULT if charges_maker_fee(ticker) else 0.0
    return TAKER_MULTIPLIER_INDEX if is_index_market else TAKER_MULTIPLIER_DEFAULT


def buy_side_price(model_prob: float, market_price: float) -> float:
    """Side-denominated buy price for the side the model favors.

    Returns ``market_price`` when buying YES (model > market),
    or ``1 - market_price`` when buying NO (model < market).
    """
    if model_prob > market_price:
        return market_price
    return 1.0 - market_price


def fee_per_contract(
    price: float,
    *,
    is_maker: bool = False,
    is_index_market: bool = False,
    ticker: str | None = None,
) -> float:
    """Conservative per-contract fee estimate (ceil applied per-contract).

    Use this in edge calculations and Kelly sizing where we want to be
    conservative.  The actual order fee (``order_fee``) is lower because
    ceil is applied to the total.

    Args:
        price: Side-denominated contract price in [0, 1].
               (YES price if buying YES, 1−YES price if buying NO.)
        is_maker: True for resting limit orders.
        is_index_market: True for KXINX/KXNDX (halved taker fee).
        ticker: Series or market ticker; selects the maker fee type.

    Returns:
        Fee per contract in dollars (e.g. 0.01 = 1 cent).
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    mult = _multiplier(is_maker, is_index_market, ticker)
    raw_cents = mult * price * (1.0 - price) * 100.0
    return math.ceil(raw_cents) / 100.0


def order_fee(
    price: float,
    contracts: int,
    *,
    is_maker: bool = False,
    is_index_market: bool = False,
    ticker: str | None = None,
) -> float:
    """Kalshi fee for a full order (ceil to the cent applied to the total).

    Use this for backtest accounting and as the P&L fallback when
    ``fees_from_api`` is absent.

    Args:
        price: Side-denominated contract price in [0, 1].
        contracts: Number of contracts in the order.
        is_maker: True for resting limit orders.
        is_index_market: True for KXINX/KXNDX.
        ticker: Series or market ticker; selects the maker fee type.

    Returns:
        Total fee in dollars for the order.
    """
    if price <= 0.0 or price >= 1.0 or contracts <= 0:
        return 0.0
    mult = _multiplier(is_maker, is_index_market, ticker)
    raw_cents = mult * contracts * price * (1.0 - price) * 100.0
    return math.ceil(raw_cents) / 100.0
