"""Test get_fills accepts min_ts/max_ts kwargs."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from python.data.kalshi_client import KalshiClient


@pytest.mark.asyncio
async def test_get_fills_passes_time_window_params() -> None:
    client = KalshiClient.__new__(KalshiClient)  # bypass __init__
    client._request = AsyncMock(return_value={"fills": []})

    await client.get_fills(min_ts=1700000000, max_ts=1700086400)

    client._request.assert_awaited_once()
    call_kwargs = client._request.await_args.kwargs
    params = call_kwargs["params"]
    assert params["min_ts"] == 1700000000
    assert params["max_ts"] == 1700086400


@pytest.mark.asyncio
async def test_get_fills_omits_time_window_when_not_provided() -> None:
    client = KalshiClient.__new__(KalshiClient)
    client._request = AsyncMock(return_value={"fills": []})

    await client.get_fills(ticker="KXHIGH-26MAY15H1600")

    params = client._request.await_args.kwargs["params"]
    assert "min_ts" not in params
    assert "max_ts" not in params
    assert params["ticker"] == "KXHIGH-26MAY15H1600"
