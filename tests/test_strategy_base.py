from __future__ import annotations

from typing import Any

from python.strategies.base import Strategy


class _Stub(Strategy):
    """Minimal concrete Strategy — the ABC's abstract methods are unused here."""

    async def scan(self) -> list[dict[str, Any]]:
        return []

    async def evaluate(self, market: dict[str, Any]) -> float | None:
        return None

    async def generate_signals(self) -> list:
        return []


def test_enabled_and_not_shadow_only_trades() -> None:
    assert _Stub("s", {"enabled": True, "shadow_only": False}).configured_to_trade is True


def test_disabled_does_not_trade() -> None:
    assert _Stub("s", {"enabled": False, "shadow_only": False}).configured_to_trade is False


def test_shadow_only_does_not_trade() -> None:
    assert _Stub("s", {"enabled": True, "shadow_only": True}).configured_to_trade is False


def test_absent_shadow_only_trades() -> None:
    """The favorite_bias shape: its settings model has no shadow_only field."""
    assert _Stub("s", {"enabled": True}).configured_to_trade is True


def test_tracks_the_enabled_setter() -> None:
    s = _Stub("s", {"enabled": True})
    s.enabled = False
    assert s.configured_to_trade is False


def test_merged_config_uses_own_shadow_only() -> None:
    """intraday_emos_high merges weather's config; its own key must win.

    orchestrator.py:975-978 builds {**weather.model_dump(), **intraday.model_dump()}.
    If intraday ever loses the field, weather's value leaks in silently.
    """
    merged = {**{"enabled": True, "shadow_only": False}, **{"shadow_only": True}}
    assert _Stub("intraday_emos_high", merged).configured_to_trade is False
