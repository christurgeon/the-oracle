"""Tests for the Kalshi V2 create-order migration.

Kalshi deprecated the legacy ``POST /portfolio/orders`` create endpoint
(returns HTTP 410 ``deprecated_v1_order_endpoint`` since ~2026-06-19). The V2
endpoint is ``POST /portfolio/events/orders`` with a single-book ``bid``/``ask``
side and fixed-point *dollar* prices (YES-denominated) instead of separate
``yes_price``/``no_price`` cents.

``build_v2_order_body`` maps our legacy call signature (side=yes/no,
action=buy/sell, cents prices) onto the V2 request body.

DOCS-VERIFIED: only the (side=yes, action=buy) → "bid" mapping is confirmed
against the published example request. The NO / sell mappings are derived from
economic equivalence (buy NO @ p ≡ sell YES @ 1-p) and MUST be confirmed against
the demo API before any NO-placing or sell strategy goes live. The only live
order-placing strategy today (favorite_bias) is YES-buy-only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from python.data.kalshi_client import (
    _V2_ORDERS_PATH,
    KalshiClient,
    build_v2_order_body,
)

# ---------------------------------------------------------------------------
# build_v2_order_body — pure mapping
# ---------------------------------------------------------------------------


class TestBuildV2OrderBody:
    def test_yes_buy_limit_matches_docs_example(self) -> None:
        """(side=yes, action=buy) → side="bid", YES price in fixed-point dollars.

        Mirrors the verbatim example request from the create-order-v2 docs.
        """
        body = build_v2_order_body(
            ticker="HIGHNY-24JAN01-T60",
            side="yes",
            action="buy",
            count=10,
            yes_price=56,
            no_price=None,
            expiration_ts=None,
        )
        assert body == {
            "ticker": "HIGHNY-24JAN01-T60",
            "side": "bid",
            "count": "10.00",
            "price": "0.5600",
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }

    def test_no_buy_limit_maps_to_ask_at_complement_price(self) -> None:
        """Buy NO @ 40¢ ≡ sell YES @ 60¢ → side="ask", price="0.6000"."""
        body = build_v2_order_body(
            ticker="T",
            side="no",
            action="buy",
            count=3,
            yes_price=None,
            no_price=40,
            expiration_ts=None,
        )
        assert body["side"] == "ask"
        assert body["price"] == "0.6000"
        assert body["count"] == "3.00"

    def test_yes_sell_maps_to_ask(self) -> None:
        body = build_v2_order_body(
            ticker="T", side="yes", action="sell", count=1,
            yes_price=50, no_price=None, expiration_ts=None,
        )
        assert body["side"] == "ask"
        assert body["price"] == "0.5000"

    def test_no_sell_maps_to_bid(self) -> None:
        body = build_v2_order_body(
            ticker="T", side="no", action="sell", count=1,
            yes_price=None, no_price=30, expiration_ts=None,
        )
        assert body["side"] == "bid"
        assert body["price"] == "0.7000"

    def test_expiration_ts_becomes_expiration_time_with_gtc(self) -> None:
        body = build_v2_order_body(
            ticker="T", side="yes", action="buy", count=1,
            yes_price=50, no_price=None, expiration_ts=1700000000,
        )
        assert body["time_in_force"] == "good_till_canceled"
        assert body["expiration_time"] == 1700000000

    def test_no_expiration_omits_expiration_time(self) -> None:
        body = build_v2_order_body(
            ticker="T", side="yes", action="buy", count=1,
            yes_price=50, no_price=None, expiration_ts=None,
        )
        assert "expiration_time" not in body

    def test_price_boundaries_format_to_four_decimals(self) -> None:
        lo = build_v2_order_body(
            ticker="T", side="yes", action="buy", count=1,
            yes_price=1, no_price=None, expiration_ts=None,
        )
        hi = build_v2_order_body(
            ticker="T", side="yes", action="buy", count=1,
            yes_price=99, no_price=None, expiration_ts=None,
        )
        assert lo["price"] == "0.0100"
        assert hi["price"] == "0.9900"


# ---------------------------------------------------------------------------
# place_order — posts the V2 body to the V2 path
# ---------------------------------------------------------------------------


def _make_client() -> KalshiClient:
    with patch("python.data.kalshi_client._load_private_key", return_value=MagicMock()):
        client = KalshiClient(api_key="test-key", private_key_path="/dev/null")
    client._session = MagicMock(spec=aiohttp.ClientSession)
    client._session.closed = False
    client._auth_headers = MagicMock(return_value={"Authorization": "test"})
    client._rate_limiter = MagicMock()
    client._rate_limiter.acquire = AsyncMock()
    return client


class TestPlaceOrderUsesV2Endpoint:
    @pytest.mark.asyncio
    async def test_posts_v2_path_and_body(self) -> None:
        client = _make_client()
        client._request = AsyncMock(return_value={"order_id": "abc", "fill_count": "0.00"})

        await client.place_order(
            ticker="HIGHNY-24JAN01-T60",
            side="yes",
            action="buy",
            order_type="limit",
            count=10,
            yes_price=56,
            expiration_ts=1700000000,
        )

        client._request.assert_awaited_once()
        args, kwargs = client._request.call_args
        assert args[0] == "POST"
        assert args[1] == _V2_ORDERS_PATH == "/portfolio/events/orders"
        assert kwargs["json_body"]["side"] == "bid"
        assert kwargs["json_body"]["price"] == "0.5600"
        assert kwargs["json_body"]["count"] == "10.00"
        assert kwargs["json_body"]["expiration_time"] == 1700000000

    @pytest.mark.asyncio
    async def test_legacy_v1_path_is_not_used(self) -> None:
        client = _make_client()
        client._request = AsyncMock(return_value={"order_id": "abc"})

        await client.place_order(
            ticker="T", side="yes", order_type="limit", count=1, yes_price=50,
        )

        path = client._request.call_args.args[1]
        assert path != "/portfolio/orders", "must not use deprecated V1 create endpoint"
