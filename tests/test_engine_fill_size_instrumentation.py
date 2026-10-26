"""Fill-size instrumentation: intended_contracts / placed_contracts on rows.

Lets a report compute filled/intended (overall), filled/placed (book/TTL), and
placed/intended (pre-place shrink) — the size shortfall a 98% *market-level*
fill rate hides.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.execution.engine import ExecutionEngine, OrderRecord
from python.risk.manager import RiskManager


def _make_settings(tmp_path: Path) -> Any:
    from tests.test_execution_engine import _make_settings as _outer

    return _outer(tmp_path, mode="live")


def _make_record() -> OrderRecord:
    from python.strategies.base import TradeSignal

    signal = TradeSignal(
        market_id="m-bos-high",
        market_ticker="KXHIGHTBOS-26JUL03-T99",
        model_probability=0.61,
        market_price=0.485,
        confidence=0.61,
        strategy_name="favorite_bias",
        side=Side.YES,
        suggested_dollars=36.0,
        suggested_contracts=70,  # intended (Kelly target)
        effective_edge=0.07,
        metadata={"city": "BOS", "var": "high", "forecast_date": "2026-07-03"},
    )
    decision = MagicMock(approved=True, adjusted_dollars=20.0, rejection_reasons=[])
    return OrderRecord(
        order_id="order-bos-1",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.485,
        order_price=0.52,
        fill_count=25,  # filled (partial)
        requested_count=40,  # placed (after risk/dollar cap)
        status=OrderStatus.PARTIAL,
    )


def _make_engine(tmp_path: Path) -> ExecutionEngine:
    settings = _make_settings(tmp_path)
    settings.execution.emit_placement_rows = True
    risk = RiskManager(settings)
    return ExecutionEngine(settings, kalshi_client=AsyncMock(), risk_manager=risk)


def _read_log(engine: ExecutionEngine) -> list[dict[str, Any]]:
    p = engine._trade_log_path
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


@pytest.mark.asyncio
async def test_placement_row_carries_intended_and_placed(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    record = _make_record()
    engine._orders[record.order_id] = record
    await engine._emit_placement_row(record, timestamp=datetime.now(UTC).isoformat())
    placement = next(r for r in _read_log(engine) if r["event"] == "placement")
    assert placement["intended_contracts"] == 70
    assert placement["placed_contracts"] == 40


def test_fill_row_carries_all_three_counts(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    record = _make_record()
    entry = engine._build_log_entry(
        record, timestamp=datetime.now(UTC).isoformat(), contracts=25, dollars=13.0
    )
    # intended 70 (Kelly) -> placed 40 (risk/dollar cap) -> filled 25 (book/TTL)
    assert entry["intended_contracts"] == 70
    assert entry["placed_contracts"] == 40
    assert entry["contracts"] == 25  # filled (already logged)


@pytest.mark.asyncio
async def test_placement_and_fill_share_the_new_base_fields(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    record = _make_record()
    engine._orders[record.order_id] = record
    await engine._emit_placement_row(record, timestamp=datetime.now(UTC).isoformat())
    fill = engine._build_log_entry(
        record, timestamp=datetime.now(UTC).isoformat(), contracts=25, dollars=13.0
    )
    placement = next(r for r in _read_log(engine) if r["event"] == "placement")
    for f in ("intended_contracts", "placed_contracts"):
        assert placement[f] == fill[f]
