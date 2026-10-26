from __future__ import annotations

import pytest

from python.settings import Settings


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_non_paper_modes_use_production_urls(mode: str) -> None:
    s = Settings(execution_mode=mode)
    assert s.active_base_url == s.kalshi_base_url
    assert s.active_ws_url == s.kalshi_ws_url


def test_paper_mode_uses_demo_urls_for_rest_and_websocket() -> None:
    s = Settings(execution_mode="paper")
    assert s.active_base_url == "https://external-api.demo.kalshi.co/trade-api/v2"
    assert s.active_ws_url == "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
