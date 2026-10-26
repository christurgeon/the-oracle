"""Tests for ExecutionEngine.hydrate_live_positions.

Mirrors the orphan-stub flow that the orchestrator runs at startup and during
periodic reconciliation when a Kalshi-side position has no local OrderRecord.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.execution.engine import ExecutionEngine, LivePosition, OrderRecord
from python.risk.manager import RiskDecision, RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_settings(tmp_path, mode: str = "live", **execution_kw: Any) -> Settings:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state_dir),
        execution={"mode": mode, **execution_kw},
        _env_file=None,
    )


def _make_signal(
    *,
    market_id: str = "MKT-001",
    market_ticker: str = "MKT-001-T",
    market_price: float = 0.50,
    side: Side = Side.YES,
    suggested_dollars: float = 100.0,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TradeSignal:
    defaults: dict[str, Any] = {
        "market_id": market_id,
        "market_ticker": market_ticker,
        "model_probability": 0.65,
        "market_price": market_price,
        "confidence": 0.8,
        "strategy_name": "weather_edge",
        "side": side,
        "suggested_dollars": suggested_dollars,
        "suggested_contracts": 2,
        "metadata": metadata or {},
    }
    defaults.update(kwargs)
    return TradeSignal(**defaults)


def _make_engine(tmp_path, mode: str = "live") -> tuple[ExecutionEngine, RiskManager]:
    settings = _make_settings(tmp_path, mode=mode)
    risk = RiskManager(settings)
    client = AsyncMock() if mode != "shadow" else None
    engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)
    return engine, risk


# ── 1. Shadow mode is no-op ────────────────────────────────────────────────


def test_shadow_mode_is_no_op(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path, mode="shadow")
    pos = LivePosition(
        position_key="MKT-A",
        market_ticker="MKT-A-T",
        side=Side.YES,
        dollars=50.0,
    )
    n = engine.hydrate_live_positions([pos])
    assert n == 0
    assert engine._orders == {}


# ── 2. Empty input returns zero ────────────────────────────────────────────


def test_empty_input_returns_zero(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path)
    assert engine.hydrate_live_positions([]) == 0
    assert engine._orders == {}


# ── 3. Single orphan creates a HYDRATED stub ───────────────────────────────


def test_single_orphan_creates_hydrated_stub(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path)
    pos = LivePosition(
        position_key="MKT-A",
        market_ticker="MKT-A-T",
        side=Side.NO,
        dollars=50.0,
    )
    n = engine.hydrate_live_positions([pos])
    assert n == 1
    assert len(engine._orders) == 1

    [(_, record)] = engine._orders.items()
    assert record.status == OrderStatus.HYDRATED
    assert record.fill_price is None
    assert record.fill_count == 0
    assert record.last_recorded_dollars == pytest.approx(50.0)
    assert record.signal.side == Side.NO
    assert record.signal.strategy_name == "live_hydrated"
    assert record.signal.position_key == "MKT-A"
    assert record.signal.market_ticker == "MKT-A-T"
    assert record.signal.metadata == {}


# ── 4. Idempotent on second call ───────────────────────────────────────────


def test_idempotent_double_hydrate(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path)
    pos = LivePosition(
        position_key="MKT-A",
        market_ticker="MKT-A-T",
        side=Side.YES,
        dollars=42.0,
    )
    first = engine.hydrate_live_positions([pos])
    before_count = len(engine._orders)
    second = engine.hydrate_live_positions([pos])

    assert first == 1
    assert second == 0
    assert len(engine._orders) == before_count


# ── 5. Existing record blocks duplicate ────────────────────────────────────


def test_existing_record_blocks_duplicate_hydrate(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path)

    signal = _make_signal(market_id="T1", market_ticker="T1")
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=50.0)
    record = OrderRecord(
        order_id="prior-1",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.50,
        order_price=0.50,
        fill_price=0.50,
        fill_count=10,
        requested_count=10,
        status=OrderStatus.FILLED,
    )
    engine._orders["prior-1"] = record

    pos = LivePosition(
        position_key="T1",
        market_ticker="T1",
        side=Side.YES,
        dollars=50.0,
    )
    n = engine.hydrate_live_positions([pos])
    assert n == 0
    # Only the pre-existing FILLED record remains; no stub added.
    assert list(engine._orders) == ["prior-1"]


# ── 6. Pruning preserves HYDRATED stubs ────────────────────────────────────


def test_pruning_preserves_hydrated_stub(tmp_path) -> None:
    engine, _risk = _make_engine(tmp_path)

    # Add the HYDRATED stub first (oldest timestamp).
    pos = LivePosition(
        position_key="MKT-H",
        market_ticker="MKT-H-T",
        side=Side.YES,
        dollars=25.0,
    )
    engine.hydrate_live_positions([pos])

    # Now flood with SETTLED records that all have *newer* timestamps.
    base_ts = datetime.now(UTC)
    cap = engine._MAX_RETAINED_ORDERS
    for i in range(cap + 1):
        sig = _make_signal(market_id=f"S{i}", market_ticker=f"S{i}-T")
        decision = RiskDecision(approved=True, signal=sig, adjusted_dollars=10.0)
        rec = OrderRecord(
            order_id=f"settled-{i}",
            signal=sig,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.5,
            order_price=0.5,
            fill_price=0.5,
            fill_count=1,
            requested_count=1,
            status=OrderStatus.SETTLED,
            timestamp=base_ts + timedelta(seconds=i + 1),
        )
        engine._orders[rec.order_id] = rec

    pruned = engine.prune_completed_orders()
    assert pruned > 0
    assert len(engine._orders) <= cap
    # The HYDRATED stub must still be there (not in _PRUNABLE_STATUSES).
    statuses = {r.status for r in engine._orders.values()}
    assert OrderStatus.HYDRATED in statuses


# ── 7. Settlement releases hydrated position ───────────────────────────────


@pytest.mark.asyncio
async def test_settlement_releases_hydrated_position(tmp_path) -> None:
    engine, risk = _make_engine(tmp_path)
    ticker = "MKT-S"

    pos = LivePosition(
        position_key=ticker,
        market_ticker=ticker,
        side=Side.YES,
        dollars=50.0,
    )
    engine.hydrate_live_positions([pos])
    # Mirror what the risk manager would have at this point: hydrate_positions
    # was already called by the orchestrator's _reconcile_from_kalshi.
    risk.hydrate_positions({ticker: 50.0}, bankroll=5000.0)
    assert risk.total_exposure == pytest.approx(50.0)

    # Mock Kalshi market API to report settlement.
    engine._client.get_market = AsyncMock(
        return_value={"market": {"status": "settled", "result": "yes"}},
    )

    settled = await engine.settle_expired_positions()

    assert len(settled) == 1
    assert settled[0].market_ticker == ticker
    assert settled[0].position_key == ticker
    # Stub flipped to SETTLED.
    stub = next(iter(engine._orders.values()))
    assert stub.status == OrderStatus.SETTLED
    # Exposure released.
    assert risk.total_exposure == pytest.approx(0.0)


# ── 8. Settlement skips P&L for HYDRATED stub (no fake P&L) ────────────────


@pytest.mark.asyncio
async def test_settle_skips_pnl_for_hydrated_stub(tmp_path) -> None:
    engine, risk = _make_engine(tmp_path)
    ticker = "MKT-PNL"

    pos = LivePosition(
        position_key=ticker,
        market_ticker=ticker,
        side=Side.YES,
        dollars=50.0,
    )
    engine.hydrate_live_positions([pos])
    risk.hydrate_positions({ticker: 50.0}, bankroll=5000.0)

    engine._client.get_market = AsyncMock(
        return_value={"market": {"status": "settled", "result": "yes"}},
    )

    await engine.settle_expired_positions()

    # No fake P&L recorded against the synthetic "live_hydrated" tracker.
    assert "live_hydrated" not in risk._strategy_trackers


# ── 9. Cancellation-race reconciliation in _live_execute ───────────────────


@pytest.mark.asyncio
async def test_cancellation_race_reconciliation_in_live_execute(tmp_path) -> None:
    engine, risk = _make_engine(tmp_path)
    ticker = "MKT-RACE"

    # Step 1: hydration creates a HYDRATED stub for $50 on the position_key.
    pos = LivePosition(
        position_key=ticker,
        market_ticker=ticker,
        side=Side.YES,
        dollars=50.0,
    )
    engine.hydrate_live_positions([pos])
    # Mirror orchestrator startup: risk manager already has the $50 exposure.
    risk.hydrate_positions({ticker: 50.0}, bankroll=5000.0)
    assert risk.positions[ticker] == pytest.approx(50.0)

    # Step 2: a fresh signal goes through execute -> _live_execute.
    # Mock the Kalshi client to accept the order.
    engine._client.place_order = AsyncMock(
        return_value={
            "order": {
                "order_id": "real-1",
                "status": "pending",
            }
        },
    )

    signal = _make_signal(
        market_id=ticker,
        market_ticker=ticker,
        side=Side.YES,
        market_price=0.40,
        suggested_dollars=20.0,
    )
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=20.0)
    record = await engine._live_execute(
        signal=signal,
        decision=decision,
        order_price=0.40,
        contract_count=50,
    )

    # Stub must be flipped to CANCELLED.
    stub_records = [r for r in engine._orders.values() if r.order_id != record.order_id]
    assert len(stub_records) == 1
    assert stub_records[0].status == OrderStatus.CANCELLED

    # Risk manager exposure for this key must equal ONLY the new tentative
    # exposure (50 contracts @ $0.40 = $20.0), not $50 + $20 = $70.
    expected_tentative = 50 * 0.40
    assert risk.positions[ticker] == pytest.approx(expected_tentative)


# ── 10. Periodic reconciliation: orphan repaired without divergence alert ──


class _FakeNotifier:
    """Captures every alert message + level so we can assert dedup."""

    def __init__(self) -> None:
        self.alerts: list[tuple[str, Any]] = []

    async def send_alert(self, message: str, level: Any = None) -> None:
        self.alerts.append((message, level))


@pytest.mark.asyncio
async def test_periodic_reconciliation_repairs_orphan_no_duplicate_alert(
    tmp_path, caplog,
) -> None:
    """Orphan tickers go through PERIODIC HYDRATE only — no per-position drift alert."""
    import logging

    from python.orchestrator import Orchestrator

    settings = _make_settings(tmp_path, mode="live")
    settings.kalshi_api_key = ""  # skip real client setup paths
    settings.kalshi_private_key_path = ""

    orch = Orchestrator(settings)
    risk = RiskManager(settings)
    risk.hydrate_positions({}, bankroll=5000.0)
    # No local positions — the orphan ticker is purely API-side.

    engine, _ = _make_engine(tmp_path, mode="live")
    # Re-attach the same risk manager so the assertions match.
    engine._risk = risk

    orch._risk_manager = risk
    orch._engine = engine

    # Mock Kalshi client on the orchestrator.
    fake_client = AsyncMock()
    fake_client.get_balance = AsyncMock(return_value={"balance": 500_000})  # $5000
    fake_client.iter_all_positions = AsyncMock(
        return_value=(
            [
                {
                    "ticker": "ORPHAN-T",
                    "position_fp": "10.0",
                    "market_exposure_dollars": "30.00",
                }
            ],
            False,
        )
    )
    orch._kalshi_client = fake_client

    notifier = _FakeNotifier()
    orch._notifier = MagicMock()
    orch._notifier.send_alert = notifier.send_alert

    with caplog.at_level(logging.WARNING, logger="python.orchestrator"):
        await orch._periodic_reconciliation()

    # The orphan stub was registered.
    stub_keys = {r.signal.position_key for r in engine._orders.values()}
    assert "ORPHAN-T" in stub_keys

    # Hydration log fired exactly once.
    hydrate_msgs = [
        r for r in caplog.records
        if r.name == "python.orchestrator"
        and r.levelno == logging.WARNING
        and "PERIODIC HYDRATE" in r.getMessage()
    ]
    assert len(hydrate_msgs) == 1

    # Per-position divergence alert did NOT fire for ORPHAN-T.
    drift_msgs = [
        r for r in caplog.records
        if r.name == "python.orchestrator"
        and "Reconciliation divergence" in r.getMessage()
        and "ORPHAN-T" in r.getMessage()
    ]
    assert drift_msgs == []
    # And no Telegram drift body was sent containing the orphan ticker.
    assert not any("ORPHAN-T" in msg for msg, _ in notifier.alerts)
