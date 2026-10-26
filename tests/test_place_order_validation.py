"""Tests for place_order input validation (Fix #7)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from python.data.kalshi_client import KalshiClient, OrderValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> KalshiClient:
    """Create a KalshiClient with mocked internals for validation tests."""
    with patch("python.data.kalshi_client._load_private_key", return_value=MagicMock()):
        client = KalshiClient(
            api_key="test-key",
            private_key_path="/dev/null",
        )
    client._session = MagicMock(spec=aiohttp.ClientSession)
    client._session.closed = False
    client._auth_headers = MagicMock(return_value={"Authorization": "test"})
    client._rate_limiter = MagicMock()
    client._rate_limiter.acquire = AsyncMock()
    return client


def _valid_yes_limit() -> dict:
    """Baseline valid YES limit order kwargs."""
    return {
        "ticker": "KXHIGHNY-26MAR20-T50",
        "side": "yes",
        "action": "buy",
        "order_type": "limit",
        "count": 5,
        "yes_price": 50,
        "no_price": None,
        "expiration_ts": None,
        "sell_position_floor": None,
    }


def _valid_no_limit() -> dict:
    """Baseline valid NO limit order kwargs."""
    return {
        "ticker": "KXHIGHNY-26MAR20-T50",
        "side": "no",
        "action": "buy",
        "order_type": "limit",
        "count": 3,
        "yes_price": None,
        "no_price": 40,
        "expiration_ts": None,
        "sell_position_floor": None,
    }


# ---------------------------------------------------------------------------
# Tests: _validate_order — valid cases
# ---------------------------------------------------------------------------


class TestOrderValidationValid:
    """Verify that valid order parameters pass validation."""

    def test_valid_yes_limit_order(self) -> None:
        client = _make_client()
        client._validate_order(**_valid_yes_limit())

    def test_valid_no_limit_order(self) -> None:
        client = _make_client()
        client._validate_order(**_valid_no_limit())

    def test_valid_market_order_no_price(self) -> None:
        client = _make_client()
        client._validate_order(
            ticker="KXHIGHNY-26MAR20-T50",
            side="yes",
            action="buy",
            order_type="market",
            count=1,
            yes_price=None,
            no_price=None,
            expiration_ts=None,
            sell_position_floor=None,
        )

    def test_valid_sell_action(self) -> None:
        client = _make_client()
        params = _valid_yes_limit()
        params["action"] = "sell"
        client._validate_order(**params)

    @pytest.mark.parametrize("price", [1, 50, 99])
    def test_boundary_yes_prices(self, price: int) -> None:
        client = _make_client()
        client._validate_order(
            ticker="T", side="yes", action="buy", order_type="limit",
            count=1, yes_price=price, no_price=None,
            expiration_ts=None, sell_position_floor=None,
        )

    @pytest.mark.parametrize("price", [1, 50, 99])
    def test_boundary_no_prices(self, price: int) -> None:
        client = _make_client()
        client._validate_order(
            ticker="T", side="no", action="buy", order_type="limit",
            count=1, yes_price=None, no_price=price,
            expiration_ts=None, sell_position_floor=None,
        )

    def test_sell_position_floor_zero_is_valid(self) -> None:
        client = _make_client()
        params = _valid_yes_limit()
        params["sell_position_floor"] = 0
        client._validate_order(**params)

    def test_expiration_ts_positive(self) -> None:
        client = _make_client()
        params = _valid_yes_limit()
        params["expiration_ts"] = 1700000000
        client._validate_order(**params)


# ---------------------------------------------------------------------------
# Tests: _validate_order — invalid cases
# ---------------------------------------------------------------------------


class TestOrderValidationInvalid:
    """Verify that invalid order parameters raise OrderValidationError."""

    # -- Ticker ---------------------------------------------------------------

    def test_empty_ticker(self) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="ticker"):
            client._validate_order(
                ticker="", side="yes", action="buy", order_type="limit",
                count=1, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Side -----------------------------------------------------------------

    @pytest.mark.parametrize("side", ["YES", "NO", "maybe", "none", ""])
    def test_invalid_side(self, side: str) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="side"):
            client._validate_order(
                ticker="T", side=side, action="buy", order_type="limit",
                count=1, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Action ---------------------------------------------------------------

    @pytest.mark.parametrize("action", ["BUY", "cancel", ""])
    def test_invalid_action(self, action: str) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="action"):
            client._validate_order(
                ticker="T", side="yes", action=action, order_type="limit",
                count=1, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Order type -----------------------------------------------------------

    @pytest.mark.parametrize("order_type", ["stop", "LIMIT", ""])
    def test_invalid_order_type(self, order_type: str) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="order_type"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type=order_type,
                count=1, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Count ----------------------------------------------------------------

    @pytest.mark.parametrize("count", [0, -1, -5])
    def test_count_non_positive(self, count: int) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="count"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=count, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    def test_count_float_rejected(self) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="count"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=1.5,  # type: ignore[arg-type]
                yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Price ----------------------------------------------------------------

    @pytest.mark.parametrize("price", [0, 100, -1, 500])
    def test_yes_price_out_of_range(self, price: int) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="yes_price"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=1, yes_price=price, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    @pytest.mark.parametrize("price", [0, 100, -10, 500])
    def test_no_price_out_of_range(self, price: int) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="no_price"):
            client._validate_order(
                ticker="T", side="no", action="buy", order_type="limit",
                count=1, yes_price=None, no_price=price,
                expiration_ts=None, sell_position_floor=None,
            )

    def test_limit_yes_missing_price(self) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="yes_price is required"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=1, yes_price=None, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    def test_limit_no_missing_price(self) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="no_price is required"):
            client._validate_order(
                ticker="T", side="no", action="buy", order_type="limit",
                count=1, yes_price=None, no_price=None,
                expiration_ts=None, sell_position_floor=None,
            )

    # -- Optional params ------------------------------------------------------

    @pytest.mark.parametrize("ts", [0, -1])
    def test_expiration_ts_invalid(self, ts: int) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="expiration_ts"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=1, yes_price=50, no_price=None,
                expiration_ts=ts, sell_position_floor=None,
            )

    def test_sell_position_floor_negative(self) -> None:
        client = _make_client()
        with pytest.raises(OrderValidationError, match="sell_position_floor"):
            client._validate_order(
                ticker="T", side="yes", action="buy", order_type="limit",
                count=1, yes_price=50, no_price=None,
                expiration_ts=None, sell_position_floor=-1,
            )


# ---------------------------------------------------------------------------
# Tests: Integration with place_order
# ---------------------------------------------------------------------------


class TestPlaceOrderValidation:
    """Verify validation is called before the API request."""

    @pytest.mark.asyncio
    async def test_validation_blocks_api_call(self) -> None:
        """Invalid params should raise before _request is ever called."""
        client = _make_client()
        client._request = AsyncMock()

        with pytest.raises(OrderValidationError):
            await client.place_order(ticker="T", side="maybe", count=1)

        client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_order_reaches_api(self) -> None:
        """Valid params should make it through to _request."""
        client = _make_client()
        client._request = AsyncMock(return_value={"order": {"order_id": "abc"}})

        result = await client.place_order(
            ticker="T", side="yes", order_type="limit",
            count=1, yes_price=50,
        )

        client._request.assert_awaited_once()
        assert result["order"]["order_id"] == "abc"
