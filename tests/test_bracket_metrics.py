"""Tests for bracket sum analytics (python/analytics/bracket_metrics.py)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from python.analytics.bracket_metrics import (
    BracketSumAnalyzer,
    BracketSumMetrics,
    DirectionalSignalEnhancer,
)
from python.analytics.fees import order_fee
from python.common import Side

# ---------------------------------------------------------------------------
# Helpers: minimal stand-ins for ScanCandidate / ParsedMarket / MarketPriceInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeParsed:
    ticker: str = ""
    city: str = "SEA"
    forecast_date: date = date(2026, 3, 27)


@dataclass(frozen=True)
class _FakePriceInfo:
    yes_bid: float | None = None
    yes_ask: float | None = None
    midpoint: float = 0.0
    spread: float | None = None
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None


@dataclass
class _FakeCandidate:
    market: dict
    parsed: _FakeParsed
    price_info: _FakePriceInfo | None


def _make_candidate(
    ticker: str,
    event_ticker: str,
    yes_bid: float | None,
    yes_ask: float | None,
    bid_size: int = 50,
    ask_size: int = 50,
    city: str = "SEA",
    fdate: date = date(2026, 3, 27),
) -> _FakeCandidate:
    mid = ((yes_bid or 0) + (yes_ask or 0)) / 2.0
    spread = (yes_ask - yes_bid) if (yes_bid and yes_ask) else None
    return _FakeCandidate(
        market={"ticker": ticker, "event_ticker": event_ticker},
        parsed=_FakeParsed(ticker=ticker, city=city, forecast_date=fdate),
        price_info=_FakePriceInfo(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            midpoint=mid,
            spread=spread,
            yes_bid_size=bid_size,
            yes_ask_size=ask_size,
        ),
    )


# ---------------------------------------------------------------------------
# Seattle Mar 24 example from feasibility study (bid_sum=$1.07)
# ---------------------------------------------------------------------------

SEATTLE_MAR24_EVENT = "KXHIGHTSEA-26MAR24"
SEATTLE_MAR24 = [
    _make_candidate("KXHIGHTSEA-26MAR24-T50", SEATTLE_MAR24_EVENT, 0.02, 0.04, 45, 30),
    _make_candidate("KXHIGHTSEA-26MAR24-B50.5", SEATTLE_MAR24_EVENT, 0.07, 0.08, 40, 50),
    _make_candidate("KXHIGHTSEA-26MAR24-B52.5", SEATTLE_MAR24_EVENT, 0.35, 0.41, 100, 80),
    _make_candidate("KXHIGHTSEA-26MAR24-B54.5", SEATTLE_MAR24_EVENT, 0.54, 0.55, 200, 150),
    _make_candidate("KXHIGHTSEA-26MAR24-B56.5", SEATTLE_MAR24_EVENT, 0.08, 0.10, 60, 40),
    _make_candidate("KXHIGHTSEA-26MAR24-T57", SEATTLE_MAR24_EVENT, 0.01, 0.07, 30, 20),
]


# ---------------------------------------------------------------------------
# BracketSumAnalyzer tests
# ---------------------------------------------------------------------------


class TestBracketSumAnalyzer:
    def test_group_by_event(self):
        groups = BracketSumAnalyzer.group_by_event(SEATTLE_MAR24)
        assert SEATTLE_MAR24_EVENT in groups
        assert len(groups[SEATTLE_MAR24_EVENT]) == 6

    def test_group_by_event_multiple_events(self):
        extra = _make_candidate("KXHIGHNY-26MAR24-T50", "KXHIGHNY-26MAR24", 0.10, 0.12)
        groups = BracketSumAnalyzer.group_by_event(SEATTLE_MAR24 + [extra])
        assert len(groups) == 2

    def test_complete_bracket_set(self):
        metrics = BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)
        assert metrics is not None
        assert metrics.n_brackets == 6
        assert metrics.city == "SEA"
        assert metrics.event_ticker == SEATTLE_MAR24_EVENT

    def test_bid_sum(self):
        metrics = BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)
        assert metrics is not None
        # 0.02 + 0.07 + 0.35 + 0.54 + 0.08 + 0.01 = 1.07
        assert abs(metrics.bid_sum - 1.07) < 0.001

    def test_ask_sum(self):
        metrics = BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)
        assert metrics is not None
        # 0.04 + 0.08 + 0.41 + 0.55 + 0.10 + 0.07 = 1.25
        assert abs(metrics.ask_sum - 1.25) < 0.001

    def test_bid_excess_positive(self):
        metrics = BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)
        assert metrics is not None
        assert abs(metrics.bid_excess - 0.07) < 0.001

    def test_incomplete_set_returns_none(self):
        """More than half the brackets missing bids → return None."""
        candidates = [
            _make_candidate("T1", "EVT", None, 0.10),
            _make_candidate("T2", "EVT", None, 0.20),
            _make_candidate("T3", "EVT", None, 0.30),
            _make_candidate("T4", "EVT", 0.40, 0.50),
        ]
        metrics = BracketSumAnalyzer.compute_metrics("EVT", candidates)
        assert metrics is None

    def test_empty_candidates(self):
        assert BracketSumAnalyzer.compute_metrics("EVT", []) is None

    def test_min_bid_depth(self):
        metrics = BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)
        assert metrics is not None
        # Smallest bid_size is 30 (T57 bracket)
        assert metrics.min_bid_depth == 30

    def test_missing_bids_counted(self):
        candidates = [
            _make_candidate("T1", "EVT", None, 0.10),
            _make_candidate("T2", "EVT", 0.20, 0.25),
            _make_candidate("T3", "EVT", 0.30, 0.35),
            _make_candidate("T4", "EVT", 0.40, 0.45),
        ]
        metrics = BracketSumAnalyzer.compute_metrics("EVT", candidates)
        assert metrics is not None
        assert metrics.n_missing_bids == 1


class TestBracketArbFees:
    """Test the fee-adjusted net profit calculations."""

    def test_seattle_profitable_at_c10_with_maker_fees(self):
        """At C=10, the arb should be profitable even with 1.75% maker fees."""
        metrics = BracketSumAnalyzer.compute_metrics(
            SEATTLE_MAR24_EVENT,
            SEATTLE_MAR24,
            weather_maker_free=False,
        )
        assert metrics is not None
        net10 = metrics.net_profit_by_count.get(10, 0)
        # Design doc says $0.055/contract at C=10 maker
        assert net10 > 0.04, f"Expected positive net@10, got {net10}"

    def test_seattle_more_profitable_with_zero_maker_fees(self):
        """With $0 maker fees, net profit should be the full bid_excess."""
        metrics_free = BracketSumAnalyzer.compute_metrics(
            SEATTLE_MAR24_EVENT,
            SEATTLE_MAR24,
            weather_maker_free=True,
        )
        metrics_paid = BracketSumAnalyzer.compute_metrics(
            SEATTLE_MAR24_EVENT,
            SEATTLE_MAR24,
            weather_maker_free=False,
        )
        assert metrics_free is not None and metrics_paid is not None
        net10_free = metrics_free.net_profit_by_count.get(10, 0)
        net10_paid = metrics_paid.net_profit_by_count.get(10, 0)
        assert net10_free > net10_paid
        # With $0 fees, net/contract == bid_excess == $0.07
        assert abs(net10_free - 0.07) < 0.001

    def test_seattle_marginal_at_c1_with_maker_fees(self):
        """At C=1 with maker fees, the arb is marginal due to ceil() floor."""
        metrics = BracketSumAnalyzer.compute_metrics(
            SEATTLE_MAR24_EVENT,
            SEATTLE_MAR24,
            weather_maker_free=False,
        )
        assert metrics is not None
        net1 = metrics.net_profit_by_count.get(1, 0)
        # $0.07 gross - $0.06 fee = $0.01 (marginal)
        assert net1 <= 0.02

    def test_no_profit_when_bid_sum_below_1(self):
        """When bid_sum < $1.00, all net profits should be zero."""
        candidates = [
            _make_candidate("T1", "EVT", 0.10, 0.15),
            _make_candidate("T2", "EVT", 0.20, 0.25),
            _make_candidate("T3", "EVT", 0.30, 0.35),
            _make_candidate("T4", "EVT", 0.15, 0.20),
        ]
        metrics = BracketSumAnalyzer.compute_metrics("EVT", candidates)
        assert metrics is not None
        assert metrics.bid_sum < 1.0
        for c, net in metrics.net_profit_by_count.items():
            assert net == 0.0

    def test_6_bracket_maker_c10_per_contract_fee_much_less_than_c1(self):
        """Per-contract maker fee at C=10 is much less than at C=1 due to ceil()."""
        prices = [0.02, 0.07, 0.35, 0.54, 0.08, 0.01]
        total_c10 = sum(order_fee(p, 10, is_maker=True) for p in prices)
        total_c1 = sum(order_fee(p, 1, is_maker=True) for p in prices)
        per_contract_c10 = total_c10 / 10
        per_contract_c1 = total_c1 / 1
        # C=10: $0.15 total = $0.015/contract; C=1: $0.06 total = $0.06/contract
        assert per_contract_c10 < per_contract_c1 * 0.5, (
            f"C=10 per-contract ${per_contract_c10} should be < half of "
            f"C=1 per-contract ${per_contract_c1}"
        )

    def test_fee_per_contract_vs_order_fee_divergence(self):
        """Demonstrate the critical ceil() difference at C=10."""
        from python.analytics.fees import fee_per_contract

        per_contract_total = 10 * fee_per_contract(0.10, is_maker=True)
        batch = order_fee(0.10, 10, is_maker=True)
        assert batch < per_contract_total


# ---------------------------------------------------------------------------
# DirectionalSignalEnhancer tests
# ---------------------------------------------------------------------------


class TestDirectionalSignalEnhancer:
    @pytest.fixture()
    def enhancer(self):
        return DirectionalSignalEnhancer(
            bid_threshold=0.03,
            ask_threshold=0.03,
            boost_factor=1.15,
            max_boost=1.30,
        )

    @pytest.fixture()
    def seattle_metrics(self):
        return BracketSumAnalyzer.compute_metrics(SEATTLE_MAR24_EVENT, SEATTLE_MAR24)

    def test_no_boost_below_threshold(self, enhancer):
        """No boost when bid_excess is below threshold."""
        metrics = BracketSumMetrics(
            event_ticker="EVT",
            city="SEA",
            forecast_date="2026-03-27",
            n_brackets=6,
            ask_sum=1.05,
            bid_sum=1.02,  # excess = 0.02, below 0.03 threshold
            mid_sum=1.035,
            ask_excess=0.05,
            bid_excess=0.02,
            spread_sum=0.06,
            n_missing_bids=0,
            n_missing_asks=0,
            min_bid_depth=50,
            min_ask_depth=50,
        )
        assert enhancer.compute_boost(Side.NO, metrics) == 1.0

    def test_boost_for_no_side_with_bid_excess(self, enhancer, seattle_metrics):
        """Boost NO signals when bid_excess > threshold (market overprices YES)."""
        assert seattle_metrics is not None
        assert seattle_metrics.bid_excess > 0.03
        boost = enhancer.compute_boost(Side.NO, seattle_metrics)
        assert boost == 1.15

    def test_no_boost_for_yes_when_bid_excess(self, enhancer, seattle_metrics):
        """Do NOT boost YES signals when bid_excess is high (contradictory)."""
        assert seattle_metrics is not None
        boost = enhancer.compute_boost(Side.YES, seattle_metrics)
        assert boost == 1.0

    def test_boost_for_yes_when_ask_deficit(self, enhancer):
        """Boost YES signals when ask_sum is below $1.00 (market underprices YES)."""
        metrics = BracketSumMetrics(
            event_ticker="EVT",
            city="SEA",
            forecast_date="2026-03-27",
            n_brackets=6,
            ask_sum=0.95,
            bid_sum=0.88,
            mid_sum=0.915,
            ask_excess=-0.05,  # below -0.03 threshold
            bid_excess=-0.12,
            spread_sum=0.06,
            n_missing_bids=0,
            n_missing_asks=0,
            min_bid_depth=50,
            min_ask_depth=50,
        )
        boost = enhancer.compute_boost(Side.YES, metrics)
        assert boost == 1.15

    def test_max_boost_cap(self):
        """Boost should never exceed max_boost even with aggressive factor."""
        enhancer = DirectionalSignalEnhancer(
            bid_threshold=0.01,
            boost_factor=1.50,  # aggressive
            max_boost=1.30,
        )
        metrics = BracketSumMetrics(
            event_ticker="EVT",
            city="SEA",
            forecast_date="2026-03-27",
            n_brackets=6,
            ask_sum=1.15,
            bid_sum=1.10,
            mid_sum=1.125,
            ask_excess=0.15,
            bid_excess=0.10,
            spread_sum=0.06,
            n_missing_bids=0,
            n_missing_asks=0,
            min_bid_depth=50,
            min_ask_depth=50,
        )
        boost = enhancer.compute_boost(Side.NO, metrics)
        assert boost == 1.30  # capped

    def test_none_metrics_returns_1(self, enhancer):
        """None metrics should return no boost."""
        assert enhancer.compute_boost(Side.NO, None) == 1.0
        assert enhancer.compute_boost(Side.YES, None) == 1.0
