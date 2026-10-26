"""
Parsing primitives for the weather-edge strategy.

Pure types and parsers — no I/O, no audit emission, no business logic.
Imported by ``weather_audit``, ``weather_evaluation``, and the
``WeatherEdgeStrategy`` class. All public symbols are re-exported by
``python/strategies/weather_edge.py`` for back-compat with tests and the
backtest engine; do not import from this module in new code outside the
weather strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from python.common import DailyVariable, Direction
from python.utils.ticker_dates import parse_ticker_date

logger = logging.getLogger(__name__)


# Map ``ParsedMarket.strike_type`` (Direction enum) to the canonical string
# labels used throughout the calibration pipeline and settlement data.
_DIRECTION_TO_STRIKE_LABEL: dict[Direction, str] = {
    Direction.ABOVE: "greater",
    Direction.BELOW: "less",
    Direction.BETWEEN: "between",
}


# --- Dataclasses ------------------------------------------------------------


@dataclass(frozen=True)
class ParsedMarket:
    """Parsed components from a Kalshi weather market, using API fields."""

    ticker: str
    city: str
    forecast_date: date
    strike_type: Direction
    floor_strike: float | None  # lower bound (ABOVE: threshold, BETWEEN: floor)
    cap_strike: float | None  # upper bound (BELOW: threshold, BETWEEN: cap)
    daily_variable: DailyVariable = DailyVariable.TEMP_MAX

    @property
    def temp_label(self) -> str:
        return "low" if self.daily_variable == DailyVariable.TEMP_MIN else "high"

    @property
    def description(self) -> str:
        label = self.temp_label
        if self.strike_type == Direction.ABOVE:
            return f"{self.city} {label} > {self.floor_strike:.0f}°F on {self.forecast_date}"
        elif self.strike_type == Direction.BELOW:
            return f"{self.city} {label} < {self.cap_strike:.0f}°F on {self.forecast_date}"
        else:
            return (
                f"{self.city} {label} {self.floor_strike:.0f}-{self.cap_strike:.0f}°F "
                f"on {self.forecast_date}"
            )


@dataclass(frozen=True)
class MarketPriceInfo:
    """Extracted and normalized price data from a Kalshi market."""

    midpoint: float
    yes_bid: float | None
    yes_ask: float | None
    spread: float | None  # yes_ask - yes_bid; None if either missing
    yes_bid_size: int | None = None  # contracts at best YES bid
    yes_ask_size: int | None = None  # contracts at best YES ask
    volume: int | None = None  # total volume traded
    open_interest: int | None = None  # open interest


@dataclass(frozen=True)
class ScanCandidate:
    """A scanned market with parsed and price data attached.

    Replaces the previous pattern of mutating raw API response dicts
    with ``market["_parsed"]`` and ``market["_price_info"]``.
    """

    market: dict[str, Any]  # raw Kalshi API market dict
    parsed: ParsedMarket
    price_info: MarketPriceInfo | None


@dataclass(frozen=True)
class DepthCheck:
    """Result of checking orderbook depth for a target fill size."""

    fill_price: float  # VWAP fill price (0-1 scale)
    available_contracts: int  # total contracts available on our fill side
    levels_used: int  # number of price levels consumed for fill
    total_levels: int  # total levels in the book on our fill side
    is_stale: bool  # True if book is empty or too thin to trade
    slippage: float  # fill_price - best_level_price (0-1 scale)


@dataclass(frozen=True)
class OpponentSideView:
    """Summary of the opposite (exit) side of the orderbook."""

    available: int
    levels: int
    best_price: float | None  # best bid in 0-1 space, None if no bids


# --- Orderbook math ---------------------------------------------------------


def compute_fill_from_levels(
    levels: list[tuple[float, int]],
    target_contracts: int,
    min_depth: int = 5,
) -> DepthCheck:
    """
    Walk orderbook ask levels to compute VWAP fill price.

    Args:
        levels: Ask-side levels as [(price_0_to_1, qty), ...], best (lowest) first.
        target_contracts: Number of contracts we want to fill.
        min_depth: Minimum total available contracts to consider the book liquid.

    Returns:
        DepthCheck with fill cost analysis. is_stale=True if book is too thin.
    """
    if not levels:
        return DepthCheck(
            fill_price=0.0,
            available_contracts=0,
            levels_used=0,
            total_levels=0,
            is_stale=True,
            slippage=0.0,
        )

    total_available = sum(qty for _, qty in levels)

    if total_available < min_depth:
        return DepthCheck(
            fill_price=0.0,
            available_contracts=total_available,
            levels_used=0,
            total_levels=len(levels),
            is_stale=True,
            slippage=0.0,
        )

    filled = 0
    total_cost = 0.0
    levels_used = 0

    for price, qty in levels:
        if filled >= target_contracts:
            break
        take = min(qty, target_contracts - filled)
        total_cost += take * price
        filled += take
        levels_used += 1

    if filled == 0:
        return DepthCheck(
            fill_price=0.0,
            available_contracts=total_available,
            levels_used=0,
            total_levels=len(levels),
            is_stale=True,
            slippage=0.0,
        )

    vwap = total_cost / filled
    best_price = levels[0][0]

    return DepthCheck(
        fill_price=vwap,
        available_contracts=total_available,
        levels_used=levels_used,
        total_levels=len(levels),
        is_stale=False,
        slippage=vwap - best_price,
    )


# --- Kalshi market parsing --------------------------------------------------


def parse_weather_market(
    market: dict[str, Any],
    city: str,
    daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
) -> ParsedMarket | None:
    """
    Parse a Kalshi weather market dict into structured data.

    Uses API-provided fields (strike_type, floor_strike, cap_strike) instead
    of regex-parsing the ticker string, which is more reliable and handles
    all market types (greater, less, between).

    Args:
        market: Raw market dict from the Kalshi API.
        city: City code (from WEATHER_SERIES mapping).
        daily_variable: Open-Meteo variable for forecast fetching.

    Returns:
        ParsedMarket or None if the market can't be parsed.
    """
    ticker = market.get("ticker", "")
    event_ticker = market.get("event_ticker", "")
    strike_type_str = market.get("strike_type", "")

    if strike_type_str == "greater":
        direction = Direction.ABOVE
    elif strike_type_str == "less":
        direction = Direction.BELOW
    elif strike_type_str == "between":
        direction = Direction.BETWEEN
    else:
        logger.debug("Skipping %s — unknown strike_type: %s", ticker, strike_type_str)
        return None

    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    if floor_strike is not None:
        floor_strike = float(floor_strike)
    if cap_strike is not None:
        cap_strike = float(cap_strike)

    if direction == Direction.ABOVE and floor_strike is None:
        logger.debug("Skipping %s — greater market missing floor_strike", ticker)
        return None
    if direction == Direction.BELOW and cap_strike is None:
        logger.debug("Skipping %s — less market missing cap_strike", ticker)
        return None
    if direction == Direction.BETWEEN and (floor_strike is None or cap_strike is None):
        logger.debug("Skipping %s — between market missing floor/cap_strike", ticker)
        return None

    forecast_date = parse_ticker_date(event_ticker)
    if forecast_date is None:
        logger.debug("Skipping %s — cannot parse date from event_ticker: %s", ticker, event_ticker)
        return None

    return ParsedMarket(
        ticker=ticker,
        city=city,
        forecast_date=forecast_date,
        strike_type=direction,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        daily_variable=daily_variable,
    )


# --- Field-level parsing primitives -----------------------------------------


def _parse_fp_int(val: object) -> int | None:
    """Parse a Kalshi '_fp' field (e.g., '636.00') to int. Returns None on failure."""
    if val is None:
        return None
    try:
        return int(float(val))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def _parse_dollar_price(val: object) -> float | None:
    """Parse a Kalshi ``*_dollars`` field to a probability in [0, 1].

    Kalshi API v2 dollar-denominated fields (``yes_bid_dollars``,
    ``yes_ask_dollars``, ``last_price_dollars``) are always strings in
    the ``"0.65"`` format — i.e. already in 0-1 range.  Values outside
    [0, 1] are rejected rather than guessed at.
    """
    if val is None:
        return None
    try:
        p = float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if p < 0.0 or p > 1.0:
        logger.warning("Dollar price %.4f outside [0, 1] — rejecting", p)
        return None
    return p


def _orderbook_price_to_probability(opp_price_dollars: float) -> float:
    """Convert an opposite-side orderbook bid (in dollars) to our-side ask probability.

    The Kalshi orderbook API (``orderbook_fp``) returns prices as decimal
    dollar strings in the range "0.01"-"0.99". An opposite-side bid at
    ``P`` dollars means we can buy our side at ``1 - P``.
    """
    return 1.0 - opp_price_dollars


def _parse_orderbook_level(level: object) -> tuple[float, int] | None:
    """Parse one ``orderbook_fp`` level entry to ``(price_probability, qty)``.

    Why debug-log on the skip path: silent level-parse failure is exactly
    how the 2026-04-21 schema flip (integer cents → decimal-dollar strings)
    evaded detection for a full day of shadow trading. A one-liner keeps
    drift visible without flooding prod logs on the happy path.
    """
    if not isinstance(level, list) or len(level) < 2:
        logger.debug("orderbook level ignored — unexpected shape: %r", level)
        return None
    price = _parse_dollar_price(level[0])
    qty = _parse_fp_int(level[1])
    if price is None or qty is None or qty <= 0:
        logger.debug("orderbook level ignored — unparseable: %r", level)
        return None
    return price, qty


# --- Market price extraction (lifted from WeatherEdgeStrategy staticmethod) -


def extract_market_price_info(market: dict[str, Any]) -> MarketPriceInfo | None:
    """
    Extract bid/ask/midpoint price info from a Kalshi market dict.

    Kalshi API v2 returns dollar-denominated string fields:
      yes_bid_dollars, yes_ask_dollars, last_price_dollars.
    These are always in [0, 1] range (e.g. "0.65" = 65 cents).
    """
    bid = _parse_dollar_price(market.get("yes_bid_dollars"))
    ask = _parse_dollar_price(market.get("yes_ask_dollars"))

    bid_size = _parse_fp_int(market.get("yes_bid_size_fp"))
    ask_size = _parse_fp_int(market.get("yes_ask_size_fp"))
    volume = _parse_fp_int(market.get("volume_fp"))
    oi = _parse_fp_int(market.get("open_interest_fp"))

    if bid is not None and ask is not None:
        spread = max(ask - bid, 0.0)
        midpoint = (bid + ask) / 2.0
        return MarketPriceInfo(
            midpoint=midpoint,
            yes_bid=bid,
            yes_ask=ask,
            spread=spread,
            yes_bid_size=bid_size,
            yes_ask_size=ask_size,
            volume=volume,
            open_interest=oi,
        )

    for field_name in ("yes_ask_dollars", "last_price_dollars"):
        price = _parse_dollar_price(market.get(field_name))
        if price is not None:
            return MarketPriceInfo(
                midpoint=price,
                yes_bid=None,
                yes_ask=None,
                spread=None,
                yes_bid_size=bid_size,
                yes_ask_size=ask_size,
                volume=volume,
                open_interest=oi,
            )

    return None
