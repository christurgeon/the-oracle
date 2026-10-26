"""Tests for scripts/shadow_dashboard.py per-strategy reporting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.shadow_dashboard import build_report, compute_strategy_summary


def _row(*, strategy, won: bool, order_price: float = 0.50, contracts: int = 10) -> dict:
    pnl = (1.0 - order_price) * contracts if won else -(order_price * contracts)
    return {
        "settled": True,
        "strategy": strategy,
        "won": won,
        "order_price": order_price,
        "contracts": contracts,
        "pnl": round(pnl, 2),
        "market_ticker": "KXHIGHNY-26MAY27-T84",
    }


def test_compute_strategy_summary_buckets_by_strategy() -> None:
    results = [
        _row(strategy="weather_edge", won=True),
        _row(strategy="weather_edge", won=False),
        _row(strategy="favorite_bias", won=True),
    ]
    summary = compute_strategy_summary(results)
    assert set(summary) == {"weather_edge", "favorite_bias"}
    assert summary["weather_edge"]["trades"] == 2
    assert summary["weather_edge"]["wins"] == 1
    assert summary["weather_edge"]["win_rate_pct"] == 50.0
    assert summary["favorite_bias"]["trades"] == 1
    assert summary["favorite_bias"]["win_rate_pct"] == 100.0


def test_compute_strategy_summary_missing_strategy_is_unknown() -> None:
    results = [_row(strategy=None, won=True)]
    summary = compute_strategy_summary(results)
    assert set(summary) == {"unknown"}


def test_compute_strategy_summary_ignores_pending() -> None:
    pending = {**_row(strategy="favorite_bias", won=True), "settled": False}
    summary = compute_strategy_summary([pending])
    assert summary == {}


def test_build_report_includes_by_strategy() -> None:
    strat = compute_strategy_summary([_row(strategy="favorite_bias", won=True)])
    report = build_report({}, [], {}, [], by_strategy=strat)
    assert "by_strategy" in report
    assert report["by_strategy"]["favorite_bias"]["trades"] == 1
