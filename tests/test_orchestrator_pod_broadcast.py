"""Tests for per-strategy book_s broadcast and startup allocation validation."""

from __future__ import annotations

import pytest

from python.risk.manager import RiskManager
from python.settings import RiskConfig


def _broadcast_books(rm: RiskManager, strategy_names: list[str]) -> dict[str, float]:
    """Mirror of the orchestrator broadcast computation under test:
    each strategy receives rm.book_for(name)."""
    return {name: rm.book_for(name) for name in strategy_names}


class TestBroadcastBooks:
    def test_empty_map_broadcasts_full_equity_to_all(self, default_settings):
        rm = RiskManager(default_settings)
        books = _broadcast_books(rm, ["weather_edge", "favorite_bias"])
        assert books == {"weather_edge": rm.bankroll, "favorite_bias": rm.bankroll}

    def test_allocated_broadcasts_slices(self, make_settings):
        s = make_settings(
            risk=RiskConfig(
                strategy_allocations={
                    "weather_edge": 0.15,
                    "favorite_bias": 0.15,
                    "intraday_emos_high": 0.0,
                }
            )
        )
        rm = RiskManager(s)
        books = _broadcast_books(rm, ["weather_edge", "favorite_bias", "intraday_emos_high"])
        assert books["weather_edge"] == pytest.approx(1500.0)
        assert books["favorite_bias"] == pytest.approx(1500.0)
        assert books["intraday_emos_high"] == 0.0


class _FakeStrategy:
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled


def _validate(allocations: dict[str, float], strategies: list[_FakeStrategy]) -> None:
    """Standalone mirror of the orchestrator validation, so we can unit-test
    the rule without booting a full orchestrator."""
    from python.orchestrator import validate_strategy_allocations

    validate_strategy_allocations(allocations, [(s.name, s.enabled) for s in strategies])


class TestAllocationValidation:
    def test_empty_map_always_valid(self):
        _validate({}, [_FakeStrategy("weather_edge")])  # no raise

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            _validate({"weather": 0.15}, [_FakeStrategy("weather_edge")])

    def test_missing_enabled_coverage_rejected(self):
        with pytest.raises(ValueError, match="missing an allocation"):
            _validate(
                {"weather_edge": 0.30},
                [_FakeStrategy("weather_edge"), _FakeStrategy("favorite_bias", enabled=True)],
            )

    def test_disabled_strategy_need_not_be_covered(self):
        _validate(
            {"weather_edge": 0.30},
            [_FakeStrategy("weather_edge"), _FakeStrategy("favorite_bias", enabled=False)],
        )  # no raise

    def test_zero_alloc_counts_as_covered(self):
        _validate(
            {"weather_edge": 0.15, "favorite_bias": 0.15, "intraday_emos_high": 0.0},
            [
                _FakeStrategy("weather_edge"),
                _FakeStrategy("favorite_bias"),
                _FakeStrategy("intraday_emos_high"),
            ],
        )  # no raise


class TestHydrationIndexFromTradeLog:
    def test_builds_market_strategy_index_from_trades_jsonl(self, tmp_path):
        import json

        from python.orchestrator import build_position_strategy_index

        log = tmp_path / "trades.jsonl"
        rows = [
            {"event": "placement", "market_id": "M1", "strategy": "weather_edge", "timestamp": "t1"},
            {"event": "fill", "market_id": "M1", "strategy": "weather_edge", "timestamp": "t2"},
            {
                "event": "fill",
                "market_ticker": "M2",
                "timestamp": "",
                "metadata": {"strategy": "favorite_bias"},
            },
        ]
        with log.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        idx = build_position_strategy_index(log)
        assert idx == {"M1": "weather_edge", "M2": "favorite_bias"}

    def test_missing_log_returns_empty(self, tmp_path):
        from python.orchestrator import build_position_strategy_index

        assert build_position_strategy_index(tmp_path / "nope.jsonl") == {}


class TestDriftResyncPreservesAttribution:
    def test_resync_with_rebuilt_index_restores_pods(self, risk_manager):
        # First hydrate (startup) with attribution.
        risk_manager.hydrate_positions(
            {"M1": 100.0}, 10_000.0, position_strategy={"M1": "weather_edge"}
        )
        assert risk_manager.strategy_exposure()["weather_edge"] == 100.0
        # Drift re-sync MUST pass the rebuilt index again, else attribution is
        # wiped to "unknown". Simulate the correct (wired) re-sync:
        risk_manager.hydrate_positions(
            {"M1": 120.0}, 10_050.0, position_strategy={"M1": "weather_edge"}
        )
        exp = risk_manager.strategy_exposure()
        assert exp["weather_edge"] == 120.0
        assert "unknown" not in exp or exp["unknown"] == 0.0
