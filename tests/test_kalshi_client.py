"""Tests for KalshiClient — candlesticks, iter_all_markets/positions/orders."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from python.data.kalshi_client import (
    MAX_PAGES,
    KalshiAPIError,
    KalshiClient,
    KalshiWebSocket,
    SubscribeFailed,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client(tmp_path: object) -> KalshiClient:
    """
    Return a KalshiClient whose _load_private_key call is patched out so no
    real PEM file is needed.
    """
    with patch("python.data.kalshi_client._load_private_key", return_value=object()):
        client = KalshiClient(
            api_key="test-key",
            private_key_path="/dev/null",
        )
    return client


# ---------------------------------------------------------------------------
# _load_private_key — encrypted + unencrypted PEM support (C-KALSHI-PEM)
# ---------------------------------------------------------------------------


class TestLoadPrivateKey:
    @staticmethod
    def _write_pem(tmp_path: Any, password: bytes | None) -> str:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        enc = (
            serialization.BestAvailableEncryption(password)
            if password is not None
            else serialization.NoEncryption()
        )
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=enc,
        )
        path = tmp_path / "kalshi.pem"
        path.write_bytes(pem)
        return str(path)

    def test_loads_unencrypted_pem_with_no_password(self, tmp_path: Any) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa

        from python.data.kalshi_client import _load_private_key

        path = self._write_pem(tmp_path, password=None)
        key = _load_private_key(path)
        assert isinstance(key, rsa.RSAPrivateKey)

    def test_loads_encrypted_pem_with_password(self, tmp_path: Any) -> None:
        from cryptography.hazmat.primitives.asymmetric import rsa

        from python.data.kalshi_client import _load_private_key

        path = self._write_pem(tmp_path, password=b"s3cret")
        key = _load_private_key(path, password=b"s3cret")
        assert isinstance(key, rsa.RSAPrivateKey)

    def test_encrypted_pem_without_password_raises(self, tmp_path: Any) -> None:
        from python.data.kalshi_client import _load_private_key

        path = self._write_pem(tmp_path, password=b"s3cret")
        with pytest.raises((TypeError, ValueError)):
            _load_private_key(path, password=None)


# ---------------------------------------------------------------------------
# get_candlesticks
# ---------------------------------------------------------------------------


SAMPLE_CANDLESTICKS_RESPONSE: dict[str, Any] = {
    "candlesticks": [
        {
            "end_period_ts": 1741914000,
            "yes_price": {"open": 42, "high": 55, "low": 40, "close": 50},
            "no_price": {"open": 58, "high": 60, "low": 45, "close": 50},
            "volume": 250,
            "open_interest": 1000,
        },
        {
            "end_period_ts": 1741917600,
            "yes_price": {"open": 50, "high": 60, "low": 48, "close": 57},
            "no_price": {"open": 50, "high": 52, "low": 40, "close": 43},
            "volume": 180,
            "open_interest": 1050,
        },
    ]
}


class TestGetCandlesticks:
    async def test_returns_mocked_response(self, mock_client: KalshiClient) -> None:
        """get_candlesticks should return whatever _request returns."""
        mock_client._request = AsyncMock(return_value=SAMPLE_CANDLESTICKS_RESPONSE)

        result = await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
            period_interval=60,
        )

        assert result == SAMPLE_CANDLESTICKS_RESPONSE
        assert len(result["candlesticks"]) == 2

    async def test_calls_correct_path_and_params(self, mock_client: KalshiClient) -> None:
        """get_candlesticks must pass the right URL path and query params to _request."""
        mock_client._request = AsyncMock(return_value=SAMPLE_CANDLESTICKS_RESPONSE)

        await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
            period_interval=60,
        )

        mock_client._request.assert_called_once_with(
            "GET",
            "/series/KXHIGHNY/markets/KXHIGHNY-26MAR14-T56/candlesticks",
            params={
                "start_ts": 1741910400,
                "end_ts": 1741996800,
                "period_interval": 60,
            },
        )

    async def test_default_period_interval_is_60(self, mock_client: KalshiClient) -> None:
        """When period_interval is omitted it should default to 60 minutes."""
        mock_client._request = AsyncMock(return_value={"candlesticks": []})

        await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
        )

        _args, kwargs = mock_client._request.call_args
        assert kwargs["params"]["period_interval"] == 60

    async def test_custom_period_interval_forwarded(self, mock_client: KalshiClient) -> None:
        """A caller-supplied period_interval (e.g. 1440 for daily) is forwarded as-is."""
        mock_client._request = AsyncMock(return_value={"candlesticks": []})

        await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
            period_interval=1440,
        )

        _args, kwargs = mock_client._request.call_args
        assert kwargs["params"]["period_interval"] == 1440

    async def test_empty_candlesticks_list(self, mock_client: KalshiClient) -> None:
        """An empty candlestick list is a valid response (no candles in range)."""
        mock_client._request = AsyncMock(return_value={"candlesticks": []})

        result = await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
        )

        assert result == {"candlesticks": []}

    async def test_candle_fields_present(self, mock_client: KalshiClient) -> None:
        """Each returned candle should expose the expected OHLC + volume fields."""
        mock_client._request = AsyncMock(return_value=SAMPLE_CANDLESTICKS_RESPONSE)

        result = await mock_client.get_candlesticks(
            series_ticker="KXHIGHNY",
            ticker="KXHIGHNY-26MAR14-T56",
            start_ts=1741910400,
            end_ts=1741996800,
        )

        candle = result["candlesticks"][0]
        assert "end_period_ts" in candle
        assert "yes_price" in candle
        assert "no_price" in candle
        assert "volume" in candle
        assert "open_interest" in candle

        yes = candle["yes_price"]
        for field in ("open", "high", "low", "close"):
            assert field in yes


# ---------------------------------------------------------------------------
# REST auth signature path prefix
# ---------------------------------------------------------------------------


class TestRestAuthSignaturePath:
    """Kalshi signs the FULL path including /trade-api/v2; signing only the
    endpoint suffix returns 401 INCORRECT_API_KEY_SIGNATURE on every
    authenticated endpoint. Regression for live-pilot 401 incident."""

    def test_auth_headers_signs_full_path_with_api_prefix(self) -> None:
        with (
            patch("python.data.kalshi_client._load_private_key", return_value=object()),
            patch("python.data.kalshi_client._sign_request", return_value="SIG") as mock_sign,
        ):
            client = KalshiClient(
                api_key="test-key",
                private_key_path="/dev/null",
                base_url="https://api.elections.kalshi.com/trade-api/v2",
            )
            client._auth_headers("POST", "/portfolio/orders")

        # _sign_request(private_key, ts_ms, method, path)
        args = mock_sign.call_args.args
        assert args[2] == "POST"
        assert args[3] == "/trade-api/v2/portfolio/orders"

    def test_auth_headers_handles_base_url_with_trailing_slash(self) -> None:
        with (
            patch("python.data.kalshi_client._load_private_key", return_value=object()),
            patch("python.data.kalshi_client._sign_request", return_value="SIG") as mock_sign,
        ):
            client = KalshiClient(
                api_key="test-key",
                private_key_path="/dev/null",
                base_url="https://api.elections.kalshi.com/trade-api/v2/",
            )
            client._auth_headers("GET", "/portfolio/balance")

        assert mock_sign.call_args.args[3] == "/trade-api/v2/portfolio/balance"


# ---------------------------------------------------------------------------
# iter_all_markets — pagination safety guards
# ---------------------------------------------------------------------------


def _make_page_response(markets: list[dict[str, Any]], cursor: str | None) -> dict[str, Any]:
    """Helper to build a fake get_markets response."""
    resp: dict[str, Any] = {"markets": markets}
    if cursor is not None:
        resp["cursor"] = cursor
    return resp


class TestIterAllMarketsPaginationGuards:
    """Tests for the MAX_PAGES and cursor-cycle safety guards in iter_all_markets."""

    async def test_normal_pagination_collects_all_markets(
        self, mock_client: KalshiClient
    ) -> None:
        """Normal multi-page pagination should collect markets from all pages."""
        page1 = _make_page_response([{"ticker": "A"}], cursor="cursor1")
        page2 = _make_page_response([{"ticker": "B"}], cursor="cursor2")
        page3 = _make_page_response([{"ticker": "C"}], cursor=None)

        mock_client.get_markets = AsyncMock(side_effect=[page1, page2, page3])

        markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert markets == [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}]
        assert mock_client.get_markets.call_count == 3
        assert truncated is False

    async def test_single_page_no_cursor_returns_immediately(
        self, mock_client: KalshiClient
    ) -> None:
        """When the first page has no cursor, only one request is made."""
        mock_client.get_markets = AsyncMock(
            return_value=_make_page_response([{"ticker": "X"}], cursor=None)
        )

        markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert markets == [{"ticker": "X"}]
        assert mock_client.get_markets.call_count == 1
        assert truncated is False

    async def test_max_pages_guard_breaks_loop_and_flags_truncated(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the API keeps returning fresh cursors, the loop stops at MAX_PAGES."""
        # Each call returns a unique cursor (counter-based), so the loop would never
        # terminate naturally without the MAX_PAGES guard.
        counter = {"n": 0}

        async def _infinite_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_page_response(
                [{"ticker": f"t{counter['n']}"}], cursor=f"cursor-{counter['n']}"
            )

        mock_client.get_markets = AsyncMock(side_effect=_infinite_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert mock_client.get_markets.call_count == MAX_PAGES
        assert len(markets) == MAX_PAGES
        assert truncated is True
        assert any("MAX_PAGES" in msg for msg in caplog.messages)
        assert any("TRUNCATED" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor on pages 2 and 3 (A->A) breaks the loop without flagging
        truncated. The cycling page-3 rows are DISCARDED (cycle check fires before
        extend) so the repeating page can't inflate the result."""
        stuck_cursor = "stuck-cursor"
        page1 = _make_page_response([{"ticker": "A"}], cursor="first-cursor")
        page2 = _make_page_response([{"ticker": "B"}], cursor=stuck_cursor)
        page3 = _make_page_response([{"ticker": "C"}], cursor=stuck_cursor)

        mock_client.get_markets = AsyncMock(side_effect=[page1, page2, page3])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert mock_client.get_markets.call_count == 3
        # Page-3 ("C") is discarded — the cycling page does not contribute rows.
        assert markets == [{"ticker": "A"}, {"ticker": "B"}]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_fires_on_second_page(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor returned by page 1 and page 2 trips the cycle guard."""
        stuck_cursor = "repeated-immediately"
        page1 = _make_page_response([{"ticker": "M1"}], cursor=stuck_cursor)
        page2 = _make_page_response([{"ticker": "M2"}], cursor=stuck_cursor)

        mock_client.get_markets = AsyncMock(side_effect=[page1, page2])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert mock_client.get_markets.call_count == 2
        # Page-2 ("M2") is the cycling page — discarded before extend.
        assert markets == [{"ticker": "M1"}]
        assert truncated is False

    async def test_alternating_cursor_cycle_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A->B->A cycle (the case the old prev_cursor guard missed) breaks loop."""
        page1 = _make_page_response([{"ticker": "A"}], cursor="cursor-A")
        page2 = _make_page_response([{"ticker": "B"}], cursor="cursor-B")
        # Page 3 returns cursor-A again — A->B->A cycle.
        page3 = _make_page_response([{"ticker": "C"}], cursor="cursor-A")
        # Page 4 should NEVER be requested.
        page4 = _make_page_response([{"ticker": "D"}], cursor=None)

        mock_client.get_markets = AsyncMock(side_effect=[page1, page2, page3, page4])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            markets, truncated = await mock_client.iter_all_markets(page_delay=0.0)

        assert mock_client.get_markets.call_count == 3
        # Page-3 ("C") returns the already-seen cursor-A → discarded before extend.
        assert markets == [{"ticker": "A"}, {"ticker": "B"}]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_max_pages_warning_includes_filter_params(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The MAX_PAGES warning log should include the status and series_ticker params."""
        counter = {"n": 0}

        async def _unique_cursor_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_page_response([], cursor=f"cursor-{counter['n']}")

        mock_client.get_markets = AsyncMock(side_effect=_unique_cursor_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await mock_client.iter_all_markets(
                status="open", series_ticker="KXHIGH", page_delay=0.0
            )

        warning = next(m for m in caplog.messages if "MAX_PAGES" in m)
        assert "open" in warning
        assert "KXHIGH" in warning

    async def test_passes_filter_params_to_get_markets(
        self, mock_client: KalshiClient
    ) -> None:
        """Each get_markets call should receive the expected status= and series_ticker="""
        page1 = _make_page_response([{"ticker": "A"}], cursor="cursor1")
        page2 = _make_page_response([{"ticker": "B"}], cursor=None)
        mock_client.get_markets = AsyncMock(side_effect=[page1, page2])

        await mock_client.iter_all_markets(
            status="open", series_ticker="KXHIGH", page_delay=0.0
        )

        for call in mock_client.get_markets.call_args_list:
            assert call.kwargs["status"] == "open"
            assert call.kwargs["series_ticker"] == "KXHIGH"


# ---------------------------------------------------------------------------
# iter_all_positions — pagination safety guards
# ---------------------------------------------------------------------------


def _make_positions_page_response(
    positions: list[dict[str, Any]], cursor: str | None
) -> dict[str, Any]:
    """Helper to build a fake get_positions response."""
    resp: dict[str, Any] = {"market_positions": positions}
    if cursor is not None:
        resp["cursor"] = cursor
    return resp


class TestIterAllPositionsPaginationGuards:
    """Tests for the MAX_PAGES and cursor-cycle safety guards in iter_all_positions."""

    async def test_normal_pagination_collects_all_positions(
        self, mock_client: KalshiClient
    ) -> None:
        """Normal multi-page pagination collects positions from all pages."""
        page1 = _make_positions_page_response([{"market_ticker": "A"}], cursor="cursor1")
        page2 = _make_positions_page_response([{"market_ticker": "B"}], cursor="cursor2")
        page3 = _make_positions_page_response([{"market_ticker": "C"}], cursor=None)

        mock_client.get_positions = AsyncMock(side_effect=[page1, page2, page3])

        positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        assert positions == [
            {"market_ticker": "A"},
            {"market_ticker": "B"},
            {"market_ticker": "C"},
        ]
        assert mock_client.get_positions.call_count == 3
        assert truncated is False

    async def test_single_page_no_cursor_returns_immediately(
        self, mock_client: KalshiClient
    ) -> None:
        """When the first page has no cursor, only one request is made."""
        mock_client.get_positions = AsyncMock(
            return_value=_make_positions_page_response(
                [{"market_ticker": "X"}], cursor=None
            )
        )

        positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        assert positions == [{"market_ticker": "X"}]
        assert mock_client.get_positions.call_count == 1
        assert truncated is False

    async def test_max_pages_guard_breaks_loop_and_flags_truncated(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the API keeps returning fresh cursors, the loop stops at MAX_PAGES."""
        counter = {"n": 0}

        async def _infinite_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_positions_page_response(
                [{"market_ticker": f"t{counter['n']}"}],
                cursor=f"cursor-{counter['n']}",
            )

        mock_client.get_positions = AsyncMock(side_effect=_infinite_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        assert mock_client.get_positions.call_count == MAX_PAGES
        assert len(positions) == MAX_PAGES
        assert truncated is True
        assert any("MAX_PAGES" in msg for msg in caplog.messages)
        assert any("TRUNCATED" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor on pages 2 and 3 (A->A) breaks the loop without flagging truncated."""
        stuck_cursor = "stuck-cursor"
        page1 = _make_positions_page_response(
            [{"market_ticker": "A"}], cursor="first-cursor"
        )
        page2 = _make_positions_page_response(
            [{"market_ticker": "B"}], cursor=stuck_cursor
        )
        page3 = _make_positions_page_response(
            [{"market_ticker": "C"}], cursor=stuck_cursor
        )

        mock_client.get_positions = AsyncMock(side_effect=[page1, page2, page3])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        assert mock_client.get_positions.call_count == 3
        # Page-3 ("C") is the cycling page — discarded before extend.
        assert positions == [
            {"market_ticker": "A"},
            {"market_ticker": "B"},
        ]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_fires_on_second_page(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor returned by page 1 and page 2 trips the cycle guard."""
        stuck_cursor = "repeated-immediately"
        page1 = _make_positions_page_response(
            [{"market_ticker": "M1"}], cursor=stuck_cursor
        )
        page2 = _make_positions_page_response(
            [{"market_ticker": "M2"}], cursor=stuck_cursor
        )

        mock_client.get_positions = AsyncMock(side_effect=[page1, page2])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        assert mock_client.get_positions.call_count == 2
        # Page-2 ("M2") is the cycling page — discarded before extend.
        assert positions == [{"market_ticker": "M1"}]
        assert truncated is False

    async def test_alternating_cursor_cycle_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A->B->A cycle (the case the old prev_cursor guard missed) breaks loop."""
        page1 = _make_positions_page_response([{"market_ticker": "A"}], cursor="cursor-A")
        page2 = _make_positions_page_response([{"market_ticker": "B"}], cursor="cursor-B")
        # Page 3 returns cursor-A again — A->B->A cycle.
        page3 = _make_positions_page_response([{"market_ticker": "C"}], cursor="cursor-A")
        # Page 4 should NEVER be requested.
        page4 = _make_positions_page_response([{"market_ticker": "D"}], cursor=None)

        mock_client.get_positions = AsyncMock(side_effect=[page1, page2, page3, page4])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            positions, truncated = await mock_client.iter_all_positions(page_delay=0.0)

        # The loop broke at page 3 (cycle detected on cursor-A) WITHOUT extending
        # page-3 rows, and before requesting page 4.
        assert mock_client.get_positions.call_count == 3
        assert positions == [
            {"market_ticker": "A"},
            {"market_ticker": "B"},
        ]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_max_pages_warning_includes_filter_params(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The MAX_PAGES warning log should include the ticker and settlement_status params."""
        counter = {"n": 0}

        async def _unique_cursor_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_positions_page_response([], cursor=f"cursor-{counter['n']}")

        mock_client.get_positions = AsyncMock(side_effect=_unique_cursor_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await mock_client.iter_all_positions(
                ticker="KXHIGH-T56",
                settlement_status="unsettled",
                page_delay=0.0,
            )

        warning = next(m for m in caplog.messages if "MAX_PAGES" in m)
        assert "KXHIGH-T56" in warning
        assert "unsettled" in warning

    async def test_passes_filter_params_to_get_positions(
        self, mock_client: KalshiClient
    ) -> None:
        """Each get_positions call should receive the expected ticker= and settlement_status="""
        page1 = _make_positions_page_response(
            [{"market_ticker": "A"}], cursor="cursor1"
        )
        page2 = _make_positions_page_response(
            [{"market_ticker": "B"}], cursor=None
        )
        mock_client.get_positions = AsyncMock(side_effect=[page1, page2])

        await mock_client.iter_all_positions(
            ticker="KXHIGH-T60",
            settlement_status="unsettled",
            page_delay=0.0,
        )

        for call in mock_client.get_positions.call_args_list:
            assert call.kwargs["ticker"] == "KXHIGH-T60"
            assert call.kwargs["settlement_status"] == "unsettled"


# ---------------------------------------------------------------------------
# iter_all_orders — pagination safety guards
# ---------------------------------------------------------------------------


def _make_orders_page_response(
    orders: list[dict[str, Any]], cursor: str | None
) -> dict[str, Any]:
    """Helper to build a fake get_orders response."""
    resp: dict[str, Any] = {"orders": orders}
    if cursor is not None:
        resp["cursor"] = cursor
    return resp


class TestIterAllOrdersPaginationGuards:
    """Tests for the MAX_PAGES and cursor-cycle safety guards in iter_all_orders."""

    async def test_normal_pagination_collects_all_orders(
        self, mock_client: KalshiClient
    ) -> None:
        """Normal multi-page pagination collects orders from all pages."""
        page1 = _make_orders_page_response([{"order_id": "o1"}], cursor="cursor1")
        page2 = _make_orders_page_response([{"order_id": "o2"}], cursor="cursor2")
        page3 = _make_orders_page_response([{"order_id": "o3"}], cursor=None)

        mock_client.get_orders = AsyncMock(side_effect=[page1, page2, page3])

        orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert orders == [
            {"order_id": "o1"},
            {"order_id": "o2"},
            {"order_id": "o3"},
        ]
        assert mock_client.get_orders.call_count == 3
        assert truncated is False

    async def test_single_page_no_cursor_returns_immediately(
        self, mock_client: KalshiClient
    ) -> None:
        """When the first page has no cursor, only one request is made."""
        mock_client.get_orders = AsyncMock(
            return_value=_make_orders_page_response([{"order_id": "x"}], cursor=None)
        )

        orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert orders == [{"order_id": "x"}]
        assert mock_client.get_orders.call_count == 1
        assert truncated is False

    async def test_max_pages_guard_breaks_loop_and_flags_truncated(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the API keeps returning fresh cursors, the loop stops at MAX_PAGES."""
        counter = {"n": 0}

        async def _infinite_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_orders_page_response(
                [{"order_id": f"o{counter['n']}"}],
                cursor=f"cursor-{counter['n']}",
            )

        mock_client.get_orders = AsyncMock(side_effect=_infinite_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert mock_client.get_orders.call_count == MAX_PAGES
        assert len(orders) == MAX_PAGES
        assert truncated is True
        assert any("MAX_PAGES" in msg for msg in caplog.messages)
        assert any("TRUNCATED" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor on pages 2 and 3 (A->A) breaks the loop without flagging truncated."""
        stuck_cursor = "stuck-cursor"
        page1 = _make_orders_page_response([{"order_id": "o1"}], cursor="first-cursor")
        page2 = _make_orders_page_response([{"order_id": "o2"}], cursor=stuck_cursor)
        page3 = _make_orders_page_response([{"order_id": "o3"}], cursor=stuck_cursor)

        mock_client.get_orders = AsyncMock(side_effect=[page1, page2, page3])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert mock_client.get_orders.call_count == 3
        # Page-3 ("o3") is the cycling page — discarded before extend.
        assert orders == [
            {"order_id": "o1"},
            {"order_id": "o2"},
        ]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_immediate_repeat_cursor_fires_on_second_page(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same cursor returned by page 1 and page 2 trips the cycle guard."""
        stuck_cursor = "repeated-immediately"
        page1 = _make_orders_page_response([{"order_id": "o1"}], cursor=stuck_cursor)
        page2 = _make_orders_page_response([{"order_id": "o2"}], cursor=stuck_cursor)

        mock_client.get_orders = AsyncMock(side_effect=[page1, page2])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert mock_client.get_orders.call_count == 2
        # Page-2 ("o2") is the cycling page — discarded before extend.
        assert orders == [{"order_id": "o1"}]
        assert truncated is False

    async def test_alternating_cursor_cycle_breaks_loop_truncated_false(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A->B->A cycle (the case the old prev_cursor guard missed) breaks loop."""
        page1 = _make_orders_page_response([{"order_id": "o1"}], cursor="cursor-A")
        page2 = _make_orders_page_response([{"order_id": "o2"}], cursor="cursor-B")
        page3 = _make_orders_page_response([{"order_id": "o3"}], cursor="cursor-A")
        page4 = _make_orders_page_response([{"order_id": "o4"}], cursor=None)

        mock_client.get_orders = AsyncMock(side_effect=[page1, page2, page3, page4])

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            orders, truncated = await mock_client.iter_all_orders(page_delay=0.0)

        assert mock_client.get_orders.call_count == 3
        # Page-3 ("o3") returns the already-seen cursor-A → discarded before extend.
        assert orders == [
            {"order_id": "o1"},
            {"order_id": "o2"},
        ]
        assert truncated is False
        assert any("cursor cycle detected" in msg for msg in caplog.messages)

    async def test_max_pages_warning_includes_filter_params(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The MAX_PAGES warning log should include the ticker and status params."""
        counter = {"n": 0}

        async def _unique_cursor_page(**_kw: Any) -> dict[str, Any]:
            counter["n"] += 1
            return _make_orders_page_response([], cursor=f"cursor-{counter['n']}")

        mock_client.get_orders = AsyncMock(side_effect=_unique_cursor_page)

        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await mock_client.iter_all_orders(
                ticker="KXHIGH-T60",
                status="resting",
                page_delay=0.0,
            )

        warning = next(m for m in caplog.messages if "MAX_PAGES" in m)
        assert "KXHIGH-T60" in warning
        assert "resting" in warning

    async def test_passes_filter_params_to_get_orders(
        self, mock_client: KalshiClient
    ) -> None:
        """Each get_orders call should receive the expected ticker= and status="""
        page1 = _make_orders_page_response([{"order_id": "o1"}], cursor="cursor1")
        page2 = _make_orders_page_response([{"order_id": "o2"}], cursor=None)
        mock_client.get_orders = AsyncMock(side_effect=[page1, page2])

        await mock_client.iter_all_orders(
            ticker="KXHIGH-T60",
            status="resting",
            page_delay=0.0,
        )

        for call in mock_client.get_orders.call_args_list:
            assert call.kwargs["ticker"] == "KXHIGH-T60"
            assert call.kwargs["status"] == "resting"


# ---------------------------------------------------------------------------
# WebSocket subscribe ACK behaviour (A3)
# ---------------------------------------------------------------------------


_CLOSE_TYPES = (
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSING,
    aiohttp.WSMsgType.CLOSED,
)


class _FakeWS:
    """Scriptable stand-in for aiohttp.ClientWebSocketResponse.

    ``frames`` is a list of ``aiohttp.WSMessage``-like objects.
    ``receive()`` pops from the front; if the list is empty it sleeps
    until the timeout fires (simulating a stuck connection — tests that
    exercise this path use ``subscribe_timeout=0.05`` to time out
    promptly).

    ``__anext__`` matches real aiohttp: it raises ``StopAsyncIteration``
    only when it sees a ``CLOSE``/``CLOSING``/``CLOSED`` frame. For tests
    that need multiple subscribe attempts in sequence, patch ``_connect``
    with ``side_effect=[_FakeWS(...), _FakeWS(...), ...]`` rather than
    relying on a single fake to "loop".
    """

    def __init__(self, frames: list[Any], pong_reset: bool = False) -> None:
        self._frames = list(frames)
        self._pong_reset = pong_reset
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive(self, timeout: float | None = None) -> Any:
        if not self._frames:
            if self._pong_reset:
                # Model aiohttp consuming heartbeat pongs internally: the per-read
                # timeout keeps resetting on each pong, so receive() never returns a
                # data frame and never raises TimeoutError, even when given a
                # ``timeout``. Only asyncio.wait_for (whole-coroutine cancellation)
                # can bound this — which is exactly what listen() relies on.
                while True:
                    await asyncio.sleep(0.005)
            if timeout is not None:
                await asyncio.sleep(timeout)
                raise TimeoutError
            await asyncio.sleep(3600)  # block forever
        return self._frames.pop(0)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> Any:
        msg = await self.receive()
        if msg.type in _CLOSE_TYPES:
            raise StopAsyncIteration
        return msg


def _text_msg(payload: dict[str, Any]) -> Any:
    """Build a fake aiohttp WSMessage of type TEXT."""
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(payload)
    return msg


def _close_msg() -> Any:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.CLOSED
    msg.data = None
    return msg


def _binary_msg(data: bytes = b"\x00") -> Any:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.BINARY
    msg.data = data
    return msg


def _make_ws(
    subscribe_timeout: float = 5.0,
    pending_frames_cap: int | None = None,
) -> KalshiWebSocket:
    """Construct a KalshiWebSocket with the private-key loader patched out."""
    with patch(
        "python.data.kalshi_client._load_private_key", return_value=object()
    ):
        return KalshiWebSocket(
            api_key="test-key",
            private_key_path="/dev/null",
            subscribe_timeout=subscribe_timeout,
            pending_frames_cap=pending_frames_cap,
        )


class TestSubscribeLiveness:
    """``last_subscribe_ok_age`` pipe-health signal for the WS-health watchdog."""

    def test_none_before_any_subscribe(self) -> None:
        ws = _make_ws()
        assert ws.last_subscribe_ok_age is None

    def test_age_reflects_time_since_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = _make_ws()
        base = 1000.0
        monkeypatch.setattr("python.data.kalshi_client.time.monotonic", lambda: base)
        ws._last_subscribe_ok_monotonic = base
        monkeypatch.setattr(
            "python.data.kalshi_client.time.monotonic", lambda: base + 4.0
        )
        assert ws.last_subscribe_ok_age == pytest.approx(4.0, abs=0.01)


class TestSubscribeAck:
    """Subscribe ACK / failure paths in ``KalshiWebSocket._subscribe``."""

    @pytest.mark.asyncio
    async def test_subscribe_succeeds_on_documented_ack_literal(self) -> None:
        """Documented Kalshi ACK shape is accepted."""
        ws = _make_ws()
        fake = _FakeWS([_text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 42}})])
        await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert fake.sent == [
            {"id": 1, "cmd": "subscribe", "params": {"channels": ["ticker"]}}
        ]
        assert len(ws._pending_frames) == 0

    @pytest.mark.asyncio
    async def test_subscribe_succeeds_failopen_on_unknown_ack_type(self) -> None:
        """An unknown but non-error type echoing our id is accepted as ACK."""
        ws = _make_ws()
        fake = _FakeWS([_text_msg({"type": "subscriptions", "id": 1})])
        await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_subscribe_succeeds_failopen_on_minimal_ack(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A minimal ACK (no msg field) is accepted; sid logs as None."""
        ws = _make_ws()
        fake = _FakeWS([_text_msg({"type": "ack", "id": 1})])
        with caplog.at_level(logging.INFO, logger="python.data.kalshi_client"):
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert any("sid=None" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_subscribe_succeeds_failopen_on_idless_ack(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """M1 fix: an id-less non-error frame is accepted as ACK with WARNING."""
        ws = _make_ws()
        fake = _FakeWS([_text_msg({"type": "subscribed", "msg": {"sid": 7}})])
        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert any(
            "id-less ACK" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_subscribe_raises_on_error_frame(self) -> None:
        """An error frame echoing our id raises SubscribeFailed with code."""
        ws = _make_ws()
        fake = _FakeWS(
            [
                _text_msg(
                    {
                        "type": "error",
                        "id": 1,
                        "msg": {"code": 6, "msg": "invalid channel"},
                    }
                )
            ]
        )
        with pytest.raises(SubscribeFailed) as exc_info:
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert exc_info.value.code == 6
        assert "invalid channel" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subscribe_ignores_error_frame_for_other_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An error frame for a different id is logged WARNING and we keep waiting."""
        ws = _make_ws()
        fake = _FakeWS(
            [
                _text_msg({"type": "error", "id": 999, "msg": {"code": 1, "msg": "x"}}),
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}}),
            ]
        )
        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert any(
            "error frame with id=999" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_subscribe_raises_on_timeout(self) -> None:
        """No frames → SubscribeFailed within ~timeout window."""
        ws = _make_ws(subscribe_timeout=0.05)
        fake = _FakeWS([])
        with pytest.raises(SubscribeFailed) as exc_info:
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_subscribe_raises_on_close_during_handshake(self) -> None:
        """A CLOSED frame mid-handshake raises SubscribeFailed."""
        ws = _make_ws()
        fake = _FakeWS([_close_msg()])
        with pytest.raises(SubscribeFailed) as exc_info:
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert "connection closed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subscribe_buffers_pre_ack_unmatched_id_frame(self) -> None:
        """A frame with a non-matching id is buffered for the listener."""
        ws = _make_ws()
        fake = _FakeWS(
            [
                _text_msg({"type": "ticker", "id": 999, "ticker": "X"}),
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}}),
            ]
        )
        await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert len(ws._pending_frames) == 1
        assert ws._pending_frames[0]["ticker"] == "X"

    @pytest.mark.asyncio
    async def test_subscribe_raises_on_buffer_overflow(self) -> None:
        """M2 fix: cap is configurable; >cap unmatched-id frames overflow."""
        ws = _make_ws(pending_frames_cap=4)
        # 5 unmatched-id frames followed by an ACK that should never be reached.
        frames = [
            _text_msg({"type": "ticker", "id": 999, "n": i}) for i in range(5)
        ] + [_text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}})]
        fake = _FakeWS(frames)
        with pytest.raises(SubscribeFailed) as exc_info:
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert "buffer overflow" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subscribe_ignores_non_text_frames_until_ack(self) -> None:
        """Binary frames are ignored; the next TEXT ACK succeeds."""
        ws = _make_ws()
        fake = _FakeWS(
            [
                _binary_msg(),
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}}),
            ]
        )
        await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_subscribe_ignores_non_json_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A TEXT frame with invalid JSON is logged WARNING and skipped."""
        ws = _make_ws()
        bad = MagicMock()
        bad.type = aiohttp.WSMsgType.TEXT
        bad.data = "not-json{"
        fake = _FakeWS(
            [bad, _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}})]
        )
        with caplog.at_level(logging.WARNING, logger="python.data.kalshi_client"):
            await ws._subscribe(fake, channels=["ticker"])  # type: ignore[arg-type]
        assert any("Non-JSON frame" in rec.message for rec in caplog.records)


class TestListenIntegration:
    """``listen()`` integration: pending-frame drain + SubscribeFailed propagation."""

    @pytest.mark.asyncio
    async def test_listen_propagates_subscribe_failed(self) -> None:
        """An error-frame subscribe rejection escapes ``listen()``."""
        ws = _make_ws()
        fake = _FakeWS(
            [
                _text_msg(
                    {
                        "type": "error",
                        "id": 1,
                        "msg": {"code": 6, "msg": "invalid channel"},
                    }
                )
            ]
        )
        with patch.object(ws, "_connect", AsyncMock(return_value=fake)):
            gen = ws.listen(channels=["ticker"])
            with pytest.raises(SubscribeFailed):
                async for _ in gen:
                    pytest.fail("listener should not yield before raising")

    @pytest.mark.asyncio
    async def test_listen_drains_pending_frames_before_yield_loop(self) -> None:
        """Frames buffered during _subscribe are yielded before fresh frames."""
        ws = _make_ws()
        # snapshot frame (unmatched id, buffered) → ACK → ticker frame → close
        fake = _FakeWS(
            [
                _text_msg({"type": "snapshot", "id": 999, "marker": "buffered"}),
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}}),
                _text_msg({"type": "ticker", "marker": "fresh"}),
                _close_msg(),
            ]
        )
        with patch.object(ws, "_connect", AsyncMock(return_value=fake)):
            ws._running = True  # set up loop
            received: list[dict[str, Any]] = []
            gen = ws.listen(channels=["ticker"])

            async def _drain() -> None:
                async for msg in gen:
                    received.append(msg)
                    if len(received) >= 2:
                        ws._running = False
                        await gen.aclose()
                        return

            await _drain()
            assert received[0]["marker"] == "buffered"
            assert received[1]["marker"] == "fresh"

    @pytest.mark.asyncio
    async def test_listen_logs_clean_close_on_remote_stopasynciteration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A clean StopAsyncIteration from aiohttp logs an INFO line."""
        ws = _make_ws()
        # First connection: ACK, then CLOSE → triggers StopAsyncIteration.
        fake1 = _FakeWS(
            [
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 1}}),
                _close_msg(),
            ]
        )
        # Second connection: ACK, then a normal ticker, then CLOSE.
        fake2 = _FakeWS(
            [
                _text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 2}}),
                _text_msg({"type": "ticker", "marker": "fresh"}),
                _close_msg(),
            ]
        )
        with (
            patch.object(
                ws, "_connect", AsyncMock(side_effect=[fake1, fake2])
            ),
            patch(
                "python.data.kalshi_client._compute_backoff", return_value=0.0
            ),
            patch(
                "python.data.kalshi_client.asyncio.sleep",
                AsyncMock(return_value=None),
            ),
            caplog.at_level(logging.INFO, logger="python.data.kalshi_client"),
        ):
            received: list[dict[str, Any]] = []
            gen = ws.listen(channels=["ticker"])

            async def _drain() -> None:
                async for msg in gen:
                    received.append(msg)
                    if msg.get("marker") == "fresh":
                        ws._running = False
                        await gen.aclose()
                        return

            await _drain()

        assert any(
            "WebSocket closed by remote" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Subscribe params shape — Kalshi requires market_tickers for orderbook_delta
# (regression test for production SubscribeFailed: code=2 'Params required',
# surfaced once the ACK reader was added so error frames are no longer
# silently swallowed by the listen loop).
# ---------------------------------------------------------------------------


class TestSubscribeParamsShape:
    """The orderbook_delta channel REQUIRES ``market_tickers`` per Kalshi.

    Without ``market_tickers`` Kalshi rejects the subscribe with
    ``{"type":"error","msg":{"code":2,"msg":"Params required"}}``. Pre-A3
    this rejection was silently consumed by the listen loop because the
    error frame was just yielded into a no-op orchestrator handler; A3's
    ACK reader correctly surfaces it as ``SubscribeFailed``, exposing a
    latent bug that has been live since Phase 2.

    The fix is to fail at the subscribe-params construction step with a
    clear ValueError so any future caller that re-introduces the bad
    shape gets caught at the unit-test level rather than in production.
    """

    @pytest.mark.asyncio
    async def test_subscribe_rejects_orderbook_delta_without_market_tickers(
        self,
    ) -> None:
        """orderbook_delta + tickers=None must fail before the send."""
        ws = _make_ws()
        fake = _FakeWS([])  # would block forever if we even got to receive()
        with pytest.raises(ValueError, match="market_tickers"):
            await ws._subscribe(  # type: ignore[arg-type]
                fake, channels=["orderbook_delta"], market_tickers=None
            )
        assert fake.sent == []  # nothing was sent on the wire

    @pytest.mark.asyncio
    async def test_subscribe_rejects_orderbook_delta_with_empty_market_tickers(
        self,
    ) -> None:
        """Empty list is the same as None for Kalshi's purposes."""
        ws = _make_ws()
        fake = _FakeWS([])
        with pytest.raises(ValueError, match="market_tickers"):
            await ws._subscribe(  # type: ignore[arg-type]
                fake, channels=["orderbook_delta"], market_tickers=[]
            )
        assert fake.sent == []

    @pytest.mark.asyncio
    async def test_subscribe_allows_orderbook_delta_with_market_tickers(
        self,
    ) -> None:
        """orderbook_delta + a non-empty ticker list is the documented happy path."""
        ws = _make_ws()
        fake = _FakeWS(
            [_text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 7}})]
        )
        await ws._subscribe(  # type: ignore[arg-type]
            fake,
            channels=["orderbook_delta"],
            market_tickers=["KXHIGHNY-26APR27-T70"],
        )
        assert fake.sent == [
            {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": ["KXHIGHNY-26APR27-T70"],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_subscribe_allows_ticker_channel_without_market_tickers(
        self,
    ) -> None:
        """The ``ticker`` channel accepts a global subscribe (no market_tickers)."""
        ws = _make_ws()
        fake = _FakeWS(
            [_text_msg({"type": "subscribed", "id": 1, "msg": {"sid": 7}})]
        )
        await ws._subscribe(  # type: ignore[arg-type]
            fake, channels=["ticker"], market_tickers=None
        )
        assert fake.sent == [
            {"id": 1, "cmd": "subscribe", "params": {"channels": ["ticker"]}}
        ]

    @pytest.mark.asyncio
    async def test_subscribe_rejects_mixed_channels_with_orderbook_delta_and_no_tickers(
        self,
    ) -> None:
        """The exact production bug: ['orderbook_delta', 'ticker'] + tickers=None.

        Pre-fix, this was the orchestrator's call shape and Kalshi
        rejected with ``code=2 msg='Params required'`` because
        orderbook_delta requires market_tickers even when other channels
        in the same subscribe don't.
        """
        ws = _make_ws()
        fake = _FakeWS([])
        with pytest.raises(ValueError, match="market_tickers"):
            await ws._subscribe(  # type: ignore[arg-type]
                fake,
                channels=["orderbook_delta", "ticker"],
                market_tickers=None,
            )
        assert fake.sent == []


# ---------------------------------------------------------------------------
# AsyncResolver / DNS off the thread pool
# ---------------------------------------------------------------------------


class TestBuildConnector:
    """``_build_connector`` keeps DNS off the default thread pool.

    The 2-CPU VPS has only ~6 default-executor threads. aiohttp's
    ThreadedResolver runs ``socket.getaddrinfo`` there, competing with
    ``asyncio.to_thread`` work (settlements walk, Phase0/heartbeat writes);
    under load the pool saturates, DNS queues behind it, and the event loop
    stalls long enough to miss the systemd watchdog ping → SIGABRT.
    AsyncResolver (aiodns/c-ares) resolves fully async with no thread pool.
    """

    def test_uses_async_resolver(self) -> None:
        from python.data.kalshi_client import _build_connector

        with (
            patch("python.data.kalshi_client.aiohttp.AsyncResolver") as mock_resolver,
            patch("python.data.kalshi_client.aiohttp.TCPConnector") as mock_conn,
        ):
            result = _build_connector()

        mock_resolver.assert_called_once_with()
        mock_conn.assert_called_once_with(
            resolver=mock_resolver.return_value,
            keepalive_timeout=75.0,
        )
        assert result is mock_conn.return_value

    def test_falls_back_when_aiodns_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from python.data.kalshi_client import _build_connector

        with (
            patch(
                "python.data.kalshi_client.aiohttp.AsyncResolver",
                side_effect=RuntimeError("aiodns library is not installed"),
            ),
            patch("python.data.kalshi_client.aiohttp.TCPConnector") as mock_conn,
            caplog.at_level(logging.WARNING),
        ):
            result = _build_connector()

        # Falls back to the default resolver but keeps the keepalive tuning.
        mock_conn.assert_called_once_with(keepalive_timeout=75.0)
        assert result is mock_conn.return_value
        assert "aiodns" in caplog.text


class TestSessionsUseConnector:
    """Both Kalshi sessions are built with the AsyncResolver connector."""

    @pytest.mark.asyncio
    async def test_rest_connect_passes_connector(
        self, mock_client: KalshiClient
    ) -> None:
        sentinel = object()
        with (
            patch(
                "python.data.kalshi_client._build_connector", return_value=sentinel
            ) as mock_build,
            patch("python.data.kalshi_client.aiohttp.ClientSession") as mock_session,
        ):
            mock_session.return_value.closed = False
            await mock_client.connect()

        mock_build.assert_called_once_with()
        _, kwargs = mock_session.call_args
        assert kwargs["connector"] is sentinel

    @pytest.mark.asyncio
    async def test_ws_connect_passes_connector(self) -> None:
        ws = _make_ws()
        sentinel = object()
        fake_session = MagicMock()
        fake_session.closed = False
        fake_session.ws_connect = AsyncMock(return_value=MagicMock())
        with (
            patch(
                "python.data.kalshi_client._build_connector", return_value=sentinel
            ) as mock_build,
            patch(
                "python.data.kalshi_client.aiohttp.ClientSession",
                return_value=fake_session,
            ) as mock_session,
            patch("python.data.kalshi_client._sign_request", return_value="sig"),
        ):
            await ws._connect()

        mock_build.assert_called_once_with()
        _, kwargs = mock_session.call_args
        assert kwargs["connector"] is sentinel


# ---------------------------------------------------------------------------
# ContractWatcher integration
# ---------------------------------------------------------------------------


def test_kalshi_client_accepts_contract_watcher() -> None:
    """KalshiClient should accept and store a ContractWatcher."""
    from python.data.contract_watcher import ContractWatcher

    watcher = ContractWatcher()
    with patch("python.data.kalshi_client._load_private_key", return_value=MagicMock()):
        client = KalshiClient(
            api_key="test-key",
            private_key_path="/nonexistent",
            contract_watcher=watcher,
        )
    assert client._contract_watcher is watcher


# ---------------------------------------------------------------------------
# _request — anticipated (caller-handled) error statuses
# ---------------------------------------------------------------------------


class TestRequestExpectedErrorStatuses:
    """``_request`` must log caller-anticipated error statuses (e.g. the
    post-placement ``get_order`` 404 query-exchange read-after-write race) at
    DEBUG, not ERROR — while still raising ``KalshiAPIError`` so the caller can
    react. This keeps per-order ERROR noise out of the journal; the 2026-06
    V1-deprecation outage hid in exactly that kind of noise.
    """

    @staticmethod
    def _session_returning(status: int, body: dict) -> MagicMock:
        """A session whose ``.request(...)`` async-CM yields one response."""
        resp = MagicMock()
        resp.status = status
        resp.headers = {}
        resp.text = AsyncMock(return_value=json.dumps(body))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.closed = False
        session.request = MagicMock(return_value=cm)
        return session

    @pytest.mark.asyncio
    async def test_expected_404_logged_at_debug_not_error(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_client._session = self._session_returning(404, {"error": {"code": "not_found"}})
        mock_client._auth_headers = MagicMock(return_value={})

        with caplog.at_level(logging.DEBUG, logger="python.data.kalshi_client"):
            with pytest.raises(KalshiAPIError) as excinfo:
                await mock_client._request(
                    "GET", "/portfolio/orders/abc", expected_error_statuses={404}
                )

        assert excinfo.value.status == 404  # still raises so the caller reacts
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, f"expected no ERROR logs, got {[r.getMessage() for r in errors]}"
        assert any(
            r.levelno == logging.DEBUG and "404" in r.getMessage() for r in caplog.records
        ), "the anticipated 404 should still be visible at DEBUG"

    @pytest.mark.asyncio
    async def test_unexpected_404_still_logged_at_error(
        self, mock_client: KalshiClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_client._session = self._session_returning(404, {"error": {"code": "not_found"}})
        mock_client._auth_headers = MagicMock(return_value={})

        with caplog.at_level(logging.DEBUG, logger="python.data.kalshi_client"):
            with pytest.raises(KalshiAPIError):
                await mock_client._request("GET", "/portfolio/orders/abc")

        assert [r for r in caplog.records if r.levelno >= logging.ERROR], (
            "a 404 the caller did not anticipate must still log at ERROR"
        )

    @pytest.mark.asyncio
    async def test_get_order_forwards_expected_error_statuses(
        self, mock_client: KalshiClient
    ) -> None:
        mock_client._request = AsyncMock(return_value={"order": {}})
        await mock_client.get_order("abc", expected_error_statuses={404})
        mock_client._request.assert_called_once_with(
            "GET", "/portfolio/orders/abc", expected_error_statuses={404}
        )


class TestListenIdleTimeout:
    """Idle-bounded receive loop (pong-immune) — the 2026-07-11 silent-stall fix."""

    @pytest.mark.asyncio
    async def test_naive_receive_timeout_cannot_bound_pong_reset(self) -> None:
        """Regression guard: a bare receive(timeout=) never fires under a
        pong-resetting stream; only the outer asyncio.wait_for does. This is
        precisely why listen() wraps receive() in wait_for."""
        fake = _FakeWS([], pong_reset=True)
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            # Inner receive got timeout=0.05 but does NOT complete within 0.3s;
            # the OUTER wait_for is what fires.
            await asyncio.wait_for(fake.receive(timeout=0.05), timeout=0.3)

    @pytest.mark.asyncio
    async def test_listen_reconnects_on_idle_stream(self) -> None:
        ws = _make_ws()
        ws.idle_timeout = 0.05
        ws.reconnect_delay = 0.0  # no backoff sleep between the stall and reconnect
        ack = _text_msg({"type": "subscribed", "id": 1})
        # 1st connection: ACK then a silent (pong-only) stream → must reconnect.
        # 2nd connection: ACK, one TEXT data frame, then CLOSE.
        stalled = _FakeWS([ack], pong_reset=True)
        healthy = _FakeWS([ack, _text_msg({"seq": 7}), _close_msg()])
        connect = AsyncMock(side_effect=[stalled, healthy])
        with patch.object(ws, "_connect", connect):
            got: list[Any] = []

            async def run() -> None:
                async for msg in ws.listen(channels=["ticker"]):
                    got.append(msg)
                    ws._running = False
                    break

            await asyncio.wait_for(run(), timeout=2.0)
            assert connect.call_count == 2  # idle broke the 1st, reconnected to the 2nd
        assert got == [{"seq": 7}]

    @pytest.mark.asyncio
    async def test_received_any_message_flips_on_first_frame(self) -> None:
        ws = _make_ws()
        assert ws.received_any_message is False
        ws._last_message_time = 123.0
        assert ws.received_any_message is True
