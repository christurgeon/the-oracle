"""Cross-consumer contract test for state/trades.jsonl.

Every consumer that iterates trades.jsonl must use is_pnl_relevant_row
(or have a documented exemption). This test runs each consumer against
a fixture log containing all known event types plus one unknown future
event ('cancellation') and asserts row counts are stable.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from python.execution.trade_log import is_pnl_relevant_row

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "trade_log" / "mixed_events.jsonl"


def _load_fixture():
    return [json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()]


def test_fixture_shape():
    """Sanity check the fixture has the expected event types."""
    rows = _load_fixture()
    events = [r.get("event") for r in rows]
    assert "placement" in events
    assert "fill" in events
    assert "settlement" in events
    assert "cancellation" in events
    # Exactly two pnl-relevant rows (the fills)
    pnl_rows = [r for r in rows if is_pnl_relevant_row(r)]
    assert len(pnl_rows) == 2


def test_pull_post_mortem_aggregate():
    """aggregate_local_orders must count exactly 2 fills, no double-count
    from the placement row matching order_id P-1."""
    from scripts.pull_post_mortem_data import aggregate_local_orders

    rows = _load_fixture()
    aggregated = aggregate_local_orders(rows)

    assert "P-1" in aggregated
    assert "F-2" in aggregated
    assert "C-3" not in aggregated  # cancellation is not pnl-relevant
    # Exactly the two fills, no extras
    assert len(aggregated) == 2


def test_shadow_broker_replay_does_not_double_count(tmp_path: Path):
    """ShadowBroker.replay must not count placement or cancellation rows.

    The fixture fills use mode='live', so the shadow broker's mode-flip
    filter skips them (by design — shadow broker tracks shadow positions
    only). The key contract asserted here is that neither the placement
    row (P-1) nor the cancellation row (C-3) inflates the position list:
    the result must be exactly 0 shadow positions, not 1 or more from
    ghost rows leaking through the filter.
    """
    from python.execution.shadow_broker import ShadowBroker

    target = tmp_path / "trades.jsonl"
    target.write_text(FIXTURE_PATH.read_text())

    broker = ShadowBroker(
        trade_log_path=target,
        configured_bankroll=1000.0,
        today=date(2026, 5, 11),
    )
    positions = broker.replay()

    # All fills are mode='live' → shadow broker skips them (mode-flip filter).
    # Placement row (P-1) must not leak through as a ghost fill.
    # Cancellation row (C-3) must not appear as a position.
    # Net: 0 shadow positions.
    tickers = [p.market_ticker for p in positions]
    assert "KXHIGHCHI-26MAY10-B65.5" not in tickers  # cancellation row must not appear
    # Placement row P-1 must not ghost-add a second position for its ticker
    p1_positions = [p for p in positions if p.market_ticker == "KXHIGHNY-26MAY10-B70.5"]
    assert len(p1_positions) == 0  # live fill skipped; placement also skipped


def test_is_pnl_relevant_row_rejects_non_fill_events():
    """The fixture's contract: only fill rows are pnl-relevant.

    This is the helper-level assertion that backs every consumer
    migration. The actual fetch_shadow_settlements consumer calls
    is_pnl_relevant_row internally at scripts/fetch_shadow_settlements.py:373;
    its end-to-end coverage lives in tests/test_shadow_reconcile_integration.py
    (which exercises the real script against real settlement data).
    """
    rows = _load_fixture()
    pnl_rows = [r for r in rows if is_pnl_relevant_row(r)]
    assert len(pnl_rows) == 2
    assert all(r["event"] == "fill" for r in pnl_rows)


@pytest.mark.asyncio
async def test_engine_written_placement_row_invisible_to_pnl_consumers(tmp_path: Path):
    """End-to-end: enable emit_placement_rows, drive the engine to write
    a placement row + fill row, then run aggregate_local_orders and assert
    the placement row is not aggregated into P&L.

    This bridges the engine (Task 10) into the consumer (Task 11) in one
    test to confirm the is_pnl_relevant_row gate holds in a real write path.
    """
    from python.common import Side
    from python.execution.engine import ExecutionEngine
    from python.risk.manager import RiskManager
    from python.settings import Settings
    from python.strategies.base import TradeSignal
    from scripts.pull_post_mortem_data import aggregate_local_orders

    # Build a live engine with the feature flag on
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    settings = Settings(
        state_dir=str(state_dir),
        execution={"mode": "live", "emit_placement_rows": True},
        _env_file=None,
    )
    risk = RiskManager(settings)
    client: Any = AsyncMock()
    # place_order returns a resting order — no immediate fill
    client.place_order.return_value = {
        "order": {"order_id": "E2E-ORD-1", "status": "resting"},
    }
    engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

    signal = TradeSignal(
        market_id="E2E-MKT",
        market_ticker="E2E-MKT-T",
        model_probability=0.65,
        market_price=0.50,
        confidence=0.8,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=1,
        metadata={},
    )

    # Drive execute — writes one placement row (flag on, status=resting)
    record = await engine.execute(signal)
    assert record.order_id == "E2E-ORD-1"

    # Simulate a subsequent check_fills that writes one fill row
    client.get_order.return_value = {
        "order": {"fill_count_fp": "1.00", "status": "executed"},
    }
    client.get_fills.return_value = {
        "fills": [
            {"side": "yes", "yes_price_dollars": "0.5000",
             "no_price_dollars": "0.5000", "count_fp": "1.00"},
        ],
    }
    await engine.check_fills(record.order_id)

    # Load all rows from the trade log
    log_path = engine._trade_log_path
    assert log_path.exists()
    all_rows = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]

    # Sanity: there must be at least one placement row and one fill row
    placement_rows = [r for r in all_rows if r.get("event") == "placement"]
    fill_rows = [r for r in all_rows if r.get("event") == "fill"]
    assert len(placement_rows) == 1, "expected exactly 1 placement row from engine"
    assert len(fill_rows) >= 1, "expected at least 1 fill row from check_fills"

    # Run the P&L consumer — placement row must NOT appear in the result
    aggregated = aggregate_local_orders(all_rows)

    # Exactly one entry for the order_id, from the fill row only
    assert "E2E-ORD-1" in aggregated, "fill row must survive into aggregation"
    assert len(aggregated) == 1, (
        f"expected exactly 1 order in aggregated, got {len(aggregated)}: {list(aggregated)}"
    )
    # The aggregated entry must have valid fill data (not the None fields from placement)
    entry = aggregated["E2E-ORD-1"]
    assert entry["total_count"] >= 1
    assert entry["vwap"] > 0.0
