"""Bracket sum analytics for MECE prediction market bracket sets.

Computes aggregate metrics for mutually exclusive, collectively exhaustive
bracket markets (e.g., Kalshi weather temperature brackets). When the sum
of YES bids across all brackets exceeds $1.00, the market is collectively
overpricing YES — signalling either arb opportunities or directional
confirmation for existing weather strategy signals.

Zero additional API cost: uses ScanCandidate data already fetched by the
weather strategy scan phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from python.analytics.fees import order_fee
from python.common import Side

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketDetail:
    """Per-bracket price data within a MECE set."""

    ticker: str
    yes_bid: float | None
    yes_ask: float | None
    midpoint: float
    spread: float | None
    bid_depth: int | None
    ask_depth: int | None
    mid_deviation: float  # midpoint - (1.0 / n_brackets)


@dataclass(frozen=True)
class BracketSumMetrics:
    """Aggregate metrics for a complete MECE bracket set."""

    event_ticker: str  # e.g. "KXHIGHTSEA-26MAR27"
    city: str  # e.g. "SEA"
    forecast_date: str  # ISO format "2026-03-27"
    n_brackets: int  # should be 6 for weather
    ask_sum: float  # sum of yes_ask across all brackets
    bid_sum: float  # sum of yes_bid across all brackets
    mid_sum: float  # sum of midpoints across all brackets
    ask_excess: float  # ask_sum - 1.00 (positive = overpriced YES)
    bid_excess: float  # bid_sum - 1.00 (positive = arb opportunity)
    spread_sum: float  # sum of individual bid-ask spreads
    n_missing_bids: int  # brackets with no bid (None or 0)
    n_missing_asks: int  # brackets with no ask (None or 0)
    min_bid_depth: int  # minimum yes_bid_size across all brackets
    min_ask_depth: int  # minimum yes_ask_size across all brackets
    net_profit_by_count: dict[int, float] = field(default_factory=dict)
    bracket_details: list[BracketDetail] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BracketSumAnalyzer
# ---------------------------------------------------------------------------


class BracketSumAnalyzer:
    """Groups ScanCandidates by event and computes BracketSumMetrics.

    Designed to be called from WeatherEdgeStrategy.generate_signals()
    using data already fetched by scan(). Zero additional API calls.
    """

    EVAL_COUNTS = (1, 5, 10, 20)

    @staticmethod
    def group_by_event(candidates: list[Any]) -> dict[str, list[Any]]:
        """Group ScanCandidates by event_ticker."""
        groups: dict[str, list[Any]] = {}
        for sc in candidates:
            market = sc.market if hasattr(sc, "market") else sc
            event_ticker = market.get("event_ticker", "")
            if event_ticker:
                groups.setdefault(event_ticker, []).append(sc)
        return groups

    @classmethod
    def compute_metrics(
        cls,
        event_ticker: str,
        candidates: list[Any],
        *,
        maker_fee: bool = True,
        weather_maker_free: bool = False,
    ) -> BracketSumMetrics | None:
        """Compute bracket sum metrics for a single event's bracket set.

        Args:
            event_ticker: The event ticker string.
            candidates: List of ScanCandidate objects for this event.
            maker_fee: If True, compute profit assuming maker (limit) orders.
            weather_maker_free: If True, assume $0 maker fees on weather
                markets (per Kalshi fee schedule — most non-sports markets).

        Returns:
            None if the bracket set is incomplete (too many missing prices).
        """
        if not candidates:
            return None

        n = len(candidates)
        ask_sum = bid_sum = mid_sum = spread_sum = 0.0
        n_missing_bids = n_missing_asks = 0
        min_bid_depth = float("inf")
        min_ask_depth = float("inf")
        details: list[BracketDetail] = []

        for sc in candidates:
            if hasattr(sc, "price_info") and sc.price_info is not None:
                yes_bid = sc.price_info.yes_bid
                yes_ask = sc.price_info.yes_ask
                bd = sc.price_info.yes_bid_size or 0
                ad = sc.price_info.yes_ask_size or 0
                ticker = sc.parsed.ticker if hasattr(sc, "parsed") else ""
            else:
                market = sc.market if hasattr(sc, "market") else sc
                yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid")
                yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask")
                bd = market.get("yes_bid_size", 0) or 0
                ad = market.get("yes_ask_size", 0) or 0
                ticker = market.get("ticker", "")

            if yes_bid is not None and yes_bid > 0:
                bid_sum += yes_bid
                min_bid_depth = min(min_bid_depth, bd)
            else:
                n_missing_bids += 1
                yes_bid = None
                bd = 0

            if yes_ask is not None and yes_ask > 0:
                ask_sum += yes_ask
                min_ask_depth = min(min_ask_depth, ad)
            else:
                n_missing_asks += 1
                yes_ask = None
                ad = 0

            if yes_bid is not None and yes_ask is not None:
                mid = (yes_bid + yes_ask) / 2.0
            elif yes_bid is not None:
                mid = yes_bid
            elif yes_ask is not None:
                mid = yes_ask
            else:
                mid = 0.0
            mid_sum += mid
            sp = (yes_ask - yes_bid) if (yes_bid and yes_ask) else None
            if sp is not None:
                spread_sum += sp

            details.append(
                BracketDetail(
                    ticker=ticker,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    midpoint=mid,
                    spread=sp,
                    bid_depth=bd if yes_bid else None,
                    ask_depth=ad if yes_ask else None,
                    mid_deviation=mid - (1.0 / n) if n > 0 else 0.0,
                )
            )

        if n_missing_bids > n // 2 or n_missing_asks > n // 2:
            return None

        if min_bid_depth == float("inf"):
            min_bid_depth = 0
        if min_ask_depth == float("inf"):
            min_ask_depth = 0

        net_profit: dict[int, float] = {}
        bid_prices = [d.yes_bid for d in details if d.yes_bid is not None]
        for c in cls.EVAL_COUNTS:
            if len(bid_prices) == n and bid_sum > 1.0:
                gross = (bid_sum - 1.0) * c
                if weather_maker_free:
                    total_fee = 0.0
                else:
                    total_fee = sum(
                        order_fee(p, c, is_maker=maker_fee) for p in bid_prices
                    )
                net_profit[c] = round((gross - total_fee) / c, 4)
            else:
                net_profit[c] = 0.0

        city = ""
        fdate = ""
        if hasattr(candidates[0], "parsed"):
            city = candidates[0].parsed.city
            fdate = candidates[0].parsed.forecast_date.isoformat()

        return BracketSumMetrics(
            event_ticker=event_ticker,
            city=city,
            forecast_date=fdate,
            n_brackets=n,
            ask_sum=round(ask_sum, 4),
            bid_sum=round(bid_sum, 4),
            mid_sum=round(mid_sum, 4),
            ask_excess=round(ask_sum - 1.0, 4),
            bid_excess=round(bid_sum - 1.0, 4),
            spread_sum=round(spread_sum, 4),
            n_missing_bids=n_missing_bids,
            n_missing_asks=n_missing_asks,
            min_bid_depth=int(min_bid_depth),
            min_ask_depth=int(min_ask_depth),
            net_profit_by_count=net_profit,
            bracket_details=details,
        )


# ---------------------------------------------------------------------------
# DirectionalSignalEnhancer
# ---------------------------------------------------------------------------


class DirectionalSignalEnhancer:
    """Adjusts directional weather signal confidence based on bracket sum.

    When the bracket sum deviates from $1.00 in a direction that confirms
    the weather model's view, we increase the effective edge.

    Mechanism:
    - bid_sum > 1.00 + threshold -> YES prices are collectively too high.
      If weather model says "sell YES" (buy NO) on a bracket, boost edge.
    - ask_sum < 1.00 - threshold -> YES prices are collectively too low.
      If weather model says "buy YES" on a bracket, boost edge.
    """

    def __init__(
        self,
        *,
        bid_threshold: float = 0.03,
        ask_threshold: float = 0.03,
        boost_factor: float = 1.15,
        max_boost: float = 1.30,
    ) -> None:
        self._bid_threshold = bid_threshold
        self._ask_threshold = ask_threshold
        self._boost_factor = boost_factor
        self._max_boost = max_boost

    def compute_boost(
        self,
        side: Side,
        metrics: BracketSumMetrics | None,
    ) -> float:
        """Compute the edge boost factor for a directional signal.

        Args:
            side: The side the weather model wants to buy.
            metrics: BracketSumMetrics for the event this market belongs to.

        Returns:
            Multiplicative factor >= 1.0 to apply to effective_edge.
        """
        if metrics is None:
            return 1.0

        boost = 1.0

        if side == Side.NO and metrics.bid_excess > self._bid_threshold:
            boost = self._boost_factor
            logger.debug(
                "%s: bracket bid_excess %.3f confirms NO signal, boost=%.2f",
                metrics.event_ticker,
                metrics.bid_excess,
                boost,
            )
        elif side == Side.YES and metrics.ask_excess < -self._ask_threshold:
            boost = self._boost_factor
            logger.debug(
                "%s: bracket ask_excess %.3f confirms YES signal, boost=%.2f",
                metrics.event_ticker,
                metrics.ask_excess,
                boost,
            )

        return min(boost, self._max_boost)
