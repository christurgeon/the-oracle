from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from python.execution.engine import ExecutionEngine, OrderRecord


def test_effective_ttl_prefers_override() -> None:
    assert ExecutionEngine._effective_ttl_seconds(600, 120) == 600


def test_effective_ttl_falls_back_to_global_when_none() -> None:
    assert ExecutionEngine._effective_ttl_seconds(None, 120) == 120


def test_order_record_defaults_no_override() -> None:
    # New field must default to None so every existing construction path is unchanged.
    rec = OrderRecord.__dataclass_fields__["ttl_seconds_override"]
    assert rec.default is None


def _pending_record(make_signal, *, age_s: int, ttl_override: int | None) -> OrderRecord:
    from python.common import ExecutionMode, OrderStatus
    from python.risk.manager import RiskDecision

    sig = make_signal(market_ticker="KXLOWTLAX-26JUL01-T60", exec_price=0.76)
    rec = OrderRecord(
        order_id="ord-1",
        signal=sig,
        risk_decision=RiskDecision(approved=True, signal=sig, adjusted_dollars=0.0),
        mode=ExecutionMode.LIVE,
        signal_price=0.76,
        order_price=0.76,
        requested_count=38,
        status=OrderStatus.PENDING,
        ttl_seconds_override=ttl_override,
    )
    rec.timestamp = datetime.now(UTC) - timedelta(seconds=age_s)
    return rec


def _live_engine(make_settings, risk_manager):
    settings = make_settings(execution={"mode": "live"})  # order_ttl_seconds defaults to 120
    client = AsyncMock()
    # Post-cancel re-poll returns a terminal, zero-fill order (V2 field name fill_count_fp).
    client.get_order = AsyncMock(
        return_value={"order": {"order_id": "ord-1", "status": "canceled", "fill_count_fp": "0.00"}}
    )
    client.get_fills = AsyncMock(return_value={"fills": []})
    engine = ExecutionEngine(settings, client, risk_manager)
    return engine, client


@pytest.mark.asyncio
async def test_override_keeps_order_alive_past_global_ttl(
    make_settings, risk_manager, make_signal
) -> None:
    # global TTL 120; order aged 200s with a 600s override must NOT be cancelled.
    engine, client = _live_engine(make_settings, risk_manager)
    rec = _pending_record(make_signal, age_s=200, ttl_override=600)
    engine._orders[rec.order_id] = rec
    expired = await engine._expire_stale_orders()
    assert expired == []
    client.cancel_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_order_expires_at_global_ttl(
    make_settings, risk_manager, make_signal
) -> None:
    # No override, aged 200s > global 120 -> cancelled (and the re-poll path runs).
    engine, client = _live_engine(make_settings, risk_manager)
    rec = _pending_record(make_signal, age_s=200, ttl_override=None)
    engine._orders[rec.order_id] = rec
    expired = await engine._expire_stale_orders()
    client.cancel_order.assert_awaited_once_with("ord-1")
    assert rec.order_id in {r.order_id for r in expired}
