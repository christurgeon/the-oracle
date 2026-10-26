"""Tests for python.analytics.mtm — portfolio mark-to-market computation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from python.analytics.mtm import (
    PortfolioMtm,
    _parse_optional_decimal,
    _quote_prices,
    compute_portfolio_mtm,
    position_mtm_contribution,
)
from python.common import Side


class TestParseOptionalDecimal:
    def test_int_value(self):
        assert _parse_optional_decimal(12345) == 12345.0

    def test_float_value(self):
        assert _parse_optional_decimal(12.5) == 12.5

    def test_decimal_string(self):
        assert _parse_optional_decimal("123.45") == 123.45

    def test_none_returns_none(self):
        assert _parse_optional_decimal(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_optional_decimal("") is None

    def test_garbage_string_returns_none(self):
        assert _parse_optional_decimal("n/a") is None


class TestPositionMtmContribution:
    def test_yes_position(self):
        side, qty, px, contrib = position_mtm_contribution(10.0, 0.42, 0.45)
        assert side == Side.YES
        assert qty == 10.0
        assert px == pytest.approx(0.42)
        assert contrib == pytest.approx(4.20)

    def test_no_position_priced_at_one_minus_ask(self):
        side, qty, px, contrib = position_mtm_contribution(-10.0, 0.42, 0.45)
        assert side == Side.NO
        assert qty == 10.0
        assert px == pytest.approx(0.55)
        assert contrib == pytest.approx(5.50)

    def test_flat_position(self):
        side, qty, px, contrib = position_mtm_contribution(0.0, 0.5, 0.5)
        assert side == Side.NONE
        assert qty == 0.0
        assert px == 0.0
        assert contrib == 0.0

    def test_fractional_yes(self):
        side, qty, _px, contrib = position_mtm_contribution(2.5, 0.30, 0.32)
        assert side == Side.YES
        assert qty == 2.5
        assert contrib == pytest.approx(0.75)

    def test_missing_bid_yes_contributes_zero(self):
        side, _, _, contrib = position_mtm_contribution(10.0, None, 0.45)
        assert side == Side.YES
        assert contrib == 0.0

    def test_missing_ask_no_contributes_zero(self):
        side, _, _, contrib = position_mtm_contribution(-10.0, 0.42, None)
        assert side == Side.NO
        assert contrib == 0.0

    def test_out_of_range_clamps_to_unit_interval(self):
        _, _, _, contrib_high = position_mtm_contribution(10.0, 1.05, 0.5)
        assert contrib_high == pytest.approx(10.0)
        _, _, _, contrib_neg = position_mtm_contribution(10.0, -0.1, 0.5)
        assert contrib_neg == 0.0

    def test_zero_bid_yields_zero_contribution(self):
        side, _, _, contrib = position_mtm_contribution(10.0, 0.0, 0.05)
        assert side == Side.YES
        assert contrib == 0.0


class TestQuotePrices:
    def test_active_status_uses_live_quote(self):
        market = {
            "status": "active",
            "yes_bid_dollars": "0.84",
            "yes_ask_dollars": "0.94",
            "last_price_dollars": "0.93",
        }
        assert _quote_prices(market) == (0.84, 0.94)

    def test_missing_status_defaults_to_live_quote(self):
        market = {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"}
        assert _quote_prices(market) == (0.40, 0.42)

    def test_closed_status_falls_back_to_last_price(self):
        market = {
            "status": "closed",
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "1.0000",
            "last_price_dollars": "0.9900",
        }
        assert _quote_prices(market) == pytest.approx((0.99, 0.99))

    def test_closed_status_missing_last_price_returns_none(self):
        market = {"status": "closed", "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}
        assert _quote_prices(market) == (None, None)


class TestComputePortfolioMtm:
    @staticmethod
    def _make_client(
        *,
        balance_resp=None,
        positions_resp=None,
        market_responses=None,
        balance_exc=None,
        positions_exc=None,
        market_exc_by_ticker=None,
    ):
        client = AsyncMock()
        if balance_exc is not None:
            client.get_balance.side_effect = balance_exc
        else:
            client.get_balance.return_value = balance_resp or {"balance": 100_000}
        if positions_exc is not None:
            client.iter_all_positions.side_effect = positions_exc
        else:
            client.iter_all_positions.return_value = positions_resp or ([], False)

        market_responses = market_responses or {}
        market_exc_by_ticker = market_exc_by_ticker or {}

        async def get_market(ticker):
            if ticker in market_exc_by_ticker:
                raise market_exc_by_ticker[ticker]
            return market_responses[ticker]

        client.get_market = AsyncMock(side_effect=get_market)
        return client

    async def test_mixed_portfolio_priced(self):
        client = self._make_client(
            balance_resp={"balance": 500_000},  # $5,000.00
            positions_resp=(
                [
                    {"ticker": "A", "position_fp": "10.00"},
                    {"ticker": "B", "position_fp": "20.00"},
                    {"ticker": "C", "position_fp": "-15.00"},
                ],
                False,
            ),
            market_responses={
                "A": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"},
                "B": {"yes_bid_dollars": "0.50", "yes_ask_dollars": "0.55"},
                "C": {"yes_bid_dollars": "0.30", "yes_ask_dollars": "0.32"},
            },
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert isinstance(mtm, PortfolioMtm)
        assert not mtm.degraded
        assert mtm.cash == pytest.approx(5000.0)
        # A (YES 10 @ 0.40) = 4.00
        # B (YES 20 @ 0.50) = 10.00
        # C (NO 15 @ (1-0.32)=0.68) = 10.20
        assert mtm.position_mtm == pytest.approx(24.20)
        assert mtm.equity_mtm == pytest.approx(5024.20)
        assert mtm.n_positions_fetched == 3
        assert len(mtm.breakdown) == 3

    async def test_one_market_fails_degrades_gracefully(self):
        client = self._make_client(
            balance_resp={"balance": 100_000},
            positions_resp=(
                [
                    {"ticker": "A", "position_fp": "10.00"},
                    {"ticker": "B", "position_fp": "10.00"},
                ],
                False,
            ),
            market_responses={
                "A": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"},
            },
            market_exc_by_ticker={"B": RuntimeError("market gone")},
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert mtm.degraded
        assert mtm.n_positions_fetched == 2
        assert mtm.position_mtm == pytest.approx(4.0)

    async def test_balance_raises_returns_none(self):
        client = self._make_client(balance_exc=RuntimeError("auth failure"))
        mtm = await compute_portfolio_mtm(client)
        assert mtm is None

    async def test_session_closed_returns_none(self):
        # cosmetic check during graceful shutdown — aiohttp session closure
        client = self._make_client(balance_exc=RuntimeError("Session is closed"))
        mtm = await compute_portfolio_mtm(client)
        assert mtm is None

    async def test_positions_raises_returns_none(self):
        client = self._make_client(positions_exc=RuntimeError("network blip"))
        mtm = await compute_portfolio_mtm(client)
        assert mtm is None

    async def test_positions_truncated_yields_degraded(self):
        client = self._make_client(
            balance_resp={"balance": 100_000},
            positions_resp=(
                [{"ticker": "A", "position_fp": "10.00"}],
                True,  # truncated at MAX_PAGES
            ),
            market_responses={"A": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"}},
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert mtm.degraded
        assert mtm.position_mtm == pytest.approx(4.0)

    async def test_zero_quantity_filtered_before_fetch(self):
        client = self._make_client(
            balance_resp={"balance": 50_000},
            positions_resp=(
                [
                    {"ticker": "A", "position_fp": "0.00"},
                    {"ticker": "B", "position_fp": "10.00"},
                ],
                False,
            ),
            market_responses={"B": {"yes_bid_dollars": "0.30", "yes_ask_dollars": "0.35"}},
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        # only B should have been fetched — A skipped pre-fetch
        client.get_market.assert_awaited_once_with("B")
        assert mtm.n_positions_fetched == 1
        assert mtm.position_mtm == pytest.approx(3.0)

    async def test_balance_int_cents_converted_to_dollars(self):
        # Kalshi v2 balance is cents (int) — guard against silent cents/dollars confusion
        client = self._make_client(
            balance_resp={"balance": 12345},
            positions_resp=([], False),
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert mtm.cash == pytest.approx(123.45)

    async def test_market_response_wrapped_in_market_key(self):
        # Kalshi v2 /markets/{ticker} wraps the response: {"market": {...}}.
        # Regression guard for the production bug where every position
        # priced at $0 because the helper read bids off the wrapper.
        client = self._make_client(
            balance_resp={"balance": 100_000},
            positions_resp=(
                [{"ticker": "A", "position_fp": "10.00"}],
                False,
            ),
            market_responses={
                "A": {"market": {"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42"}},
            },
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert not mtm.degraded
        assert mtm.position_mtm == pytest.approx(4.0)

    async def test_closed_market_priced_at_last_price(self):
        client = self._make_client(
            balance_resp={"balance": 47_839},
            positions_resp=(
                [
                    {"ticker": "LIVE", "position_fp": "45.79"},
                    {"ticker": "CLOSED", "position_fp": "8.00"},
                ],
                False,
            ),
            market_responses={
                "LIVE": {
                    "status": "active",
                    "yes_bid_dollars": "0.84",
                    "yes_ask_dollars": "0.94",
                },
                "CLOSED": {
                    "status": "closed",
                    "yes_bid_dollars": "0.0000",
                    "yes_ask_dollars": "1.0000",
                    "last_price_dollars": "0.9900",
                },
            },
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert not mtm.degraded
        # LIVE (YES 45.79 @ 0.84) = 38.4636
        # CLOSED (YES 8 @ last_price 0.99) = 7.92, not 0
        assert mtm.position_mtm == pytest.approx(38.4636 + 7.92)
        assert mtm.equity_mtm == pytest.approx(478.39 + 38.4636 + 7.92)

    async def test_closed_market_no_side_uses_last_price(self):
        client = self._make_client(
            balance_resp={"balance": 100_000},
            positions_resp=(
                [{"ticker": "CLOSED", "position_fp": "-10.00"}],
                False,
            ),
            market_responses={
                "CLOSED": {
                    "status": "closed",
                    "yes_bid_dollars": "0.0000",
                    "yes_ask_dollars": "1.0000",
                    "last_price_dollars": "0.9900",
                },
            },
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        # NO 10 @ (1 - last_price 0.99) = 0.10
        assert mtm.position_mtm == pytest.approx(0.10)

    async def test_empty_portfolio(self):
        client = self._make_client(
            balance_resp={"balance": 10_000},
            positions_resp=([], False),
        )
        mtm = await compute_portfolio_mtm(client)
        assert mtm is not None
        assert not mtm.degraded
        assert mtm.cash == pytest.approx(100.0)
        assert mtm.position_mtm == 0.0
        assert mtm.equity_mtm == pytest.approx(100.0)
        assert mtm.n_positions_fetched == 0
        assert mtm.breakdown == ()
