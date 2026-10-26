"""Tests for python/analytics/strategy_attribution.py.

The resolver exists to make `strategy` reliable on trade-log fill rows for
reporting. Engine-path placement AND fill rows carry `strategy` at top level;
reconciler-synthesized fill rows carry it under `metadata.strategy` (often
"unknown" after a restart). The resolver joins by order_id, preferring a
top-level value from any row, so a reconciler fill whose order_id has an engine
placement row recovers the true strategy.
"""

from __future__ import annotations

from python.analytics.strategy_attribution import (
    build_strategy_index,
    resolve_strategy_by_order_id,
)


def test_top_level_strategy_indexed() -> None:
    """An engine fill row's top-level strategy is indexed by order_id."""
    rows = [{"event": "fill", "order_id": "o1", "strategy": "weather_edge"}]
    assert build_strategy_index(rows) == {"o1": "weather_edge"}


def test_placement_row_supplies_strategy_for_fill_only_oid() -> None:
    """A fill row lacking top-level strategy gets it from its placement row."""
    rows = [
        {"event": "placement", "order_id": "o1", "strategy": "favorite_bias"},
        {
            "event": "fill",
            "order_id": "o1",
            "metadata": {"strategy": "unknown", "reconcile_source": "fills_reconciler"},
        },
    ]
    idx = build_strategy_index(rows)
    assert idx["o1"] == "favorite_bias"


def test_metadata_strategy_used_only_absent_top_level() -> None:
    """metadata.strategy is used only when no top-level value exists for the oid."""
    rows = [{"event": "fill", "order_id": "o2", "metadata": {"strategy": "weather_edge"}}]
    assert build_strategy_index(rows) == {"o2": "weather_edge"}


def test_top_level_wins_over_metadata_conflict() -> None:
    """When a fill carries BOTH a top-level strategy and a conflicting
    metadata.strategy for the same oid, the top-level value wins (N1)."""
    rows = [
        {
            "event": "fill",
            "order_id": "o3",
            "strategy": "favorite_bias",
            "metadata": {"strategy": "weather_edge"},
        }
    ]
    assert build_strategy_index(rows)["o3"] == "favorite_bias"


def test_top_level_wins_regardless_of_row_order() -> None:
    """A metadata-only row seen first must not block a later top-level value (N1)."""
    rows_a = [
        {"event": "fill", "order_id": "o4", "metadata": {"strategy": "unknown"}},
        {"event": "placement", "order_id": "o4", "strategy": "favorite_bias"},
    ]
    rows_b = list(reversed(rows_a))
    assert build_strategy_index(rows_a)["o4"] == "favorite_bias"
    assert build_strategy_index(rows_b)["o4"] == "favorite_bias"


def test_empty_and_none_strategies_absent_from_index() -> None:
    """Legacy rows with None/empty strategy and no metadata are not indexed."""
    rows = [
        {"event": "fill", "order_id": "o5", "strategy": None},
        {"event": "fill", "order_id": "o6", "strategy": ""},
        {"event": "fill", "order_id": "o7", "metadata": {"strategy": ""}},
    ]
    assert build_strategy_index(rows) == {}


def test_resolve_returns_fallback_for_unknown_oid() -> None:
    idx = {"o1": "weather_edge"}
    assert resolve_strategy_by_order_id("missing", idx) == "unknown"
    assert resolve_strategy_by_order_id("missing", idx, fallback="favorite_bias") == "favorite_bias"


def test_resolve_returns_indexed_value() -> None:
    idx = {"o1": "weather_edge"}
    assert resolve_strategy_by_order_id("o1", idx, fallback="x") == "weather_edge"


from python.analytics.strategy_attribution import build_market_strategy_index


class TestBuildMarketStrategyIndex:
    def test_engine_fill_row_top_level_strategy(self):
        rows = [
            {"event": "fill", "market_id": "M1", "strategy": "weather_edge", "timestamp": "t1"},
        ]
        assert build_market_strategy_index(rows) == {"M1": "weather_edge"}

    def test_reconciler_synthetic_row_metadata_strategy(self):
        rows = [
            {
                "event": "fill",
                "market_ticker": "M2",
                "timestamp": "",
                "metadata": {"strategy": "favorite_bias"},
            },
        ]
        assert build_market_strategy_index(rows) == {"M2": "favorite_bias"}

    def test_placement_beats_fill_for_same_market(self):
        rows = [
            {"event": "fill", "market_id": "M3", "strategy": "unknown", "timestamp": "t2"},
            {"event": "placement", "market_id": "M3", "strategy": "weather_edge", "timestamp": "t1"},
        ]
        assert build_market_strategy_index(rows) == {"M3": "weather_edge"}

    def test_more_recent_fill_wins_when_no_placement(self):
        rows = [
            {"event": "fill", "market_id": "M4", "strategy": "favorite_bias", "timestamp": "t1"},
            {"event": "fill", "market_id": "M4", "strategy": "weather_edge", "timestamp": "t2"},
        ]
        assert build_market_strategy_index(rows) == {"M4": "weather_edge"}

    def test_top_level_beats_metadata_at_equal_recency(self):
        rows = [
            {"event": "fill", "market_id": "M5", "timestamp": "t1", "metadata": {"strategy": "a"}},
            {"event": "fill", "market_id": "M5", "strategy": "b", "timestamp": "t1"},
        ]
        assert build_market_strategy_index(rows) == {"M5": "b"}

    def test_market_key_prefers_market_id_then_position_key_then_ticker(self):
        rows = [{"event": "fill", "position_key": "PK", "strategy": "weather_edge", "timestamp": "t"}]
        assert build_market_strategy_index(rows) == {"PK": "weather_edge"}

    def test_non_fill_non_placement_rows_ignored(self):
        rows = [{"event": "settlement", "market_id": "M6", "strategy": "weather_edge"}]
        assert build_market_strategy_index(rows) == {}

    def test_row_with_no_resolvable_strategy_skipped(self):
        rows = [{"event": "fill", "market_id": "M7", "timestamp": "t"}]
        assert build_market_strategy_index(rows) == {}
