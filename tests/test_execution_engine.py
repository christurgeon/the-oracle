"""Tests for the execution engine (python/execution/engine.py).

Focuses on the code paths that control whether money moves correctly:
  - Shadow mode records fills in risk manager (position dedup)
  - Position key fallback (market_id vs market_ticker)
  - Settlement releases exposure in all modes
  - P&L computation on settlement
  - Risk rejection flow
  - Batch execution respects risk limits across signals
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.data.kalshi_client import KalshiAPIError
from python.execution.engine import (
    KILL_SWITCH_TIMEOUT_PREFIX,
    KILL_SWITCH_TRUNCATED_PREFIX,
    ExecutionEngine,
    OrderRecord,
    _parse_order_status,
)
from python.execution.fills_reconciler import FillsReconciler
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_settings(tmp_path, mode: str = "shadow", **execution_kw: Any) -> Settings:
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
    model_probability: float = 0.65,
    market_price: float = 0.50,
    side: Side = Side.YES,
    suggested_dollars: float = 100.0,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TradeSignal:
    defaults: dict[str, Any] = {
        "market_id": market_id,
        "market_ticker": market_ticker,
        "model_probability": model_probability,
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


# ── Shadow execute records fills ──────────────────────────────────────────


class TestShadowRecordsFills:
    """Shadow mode must call record_fill so the risk manager tracks positions."""

    @pytest.mark.asyncio
    async def test_shadow_updates_risk_positions(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        record = await engine.execute(signal)

        assert record.status == OrderStatus.SHADOW
        position_key = signal.market_id or signal.market_ticker
        assert position_key in risk.positions
        assert risk.positions[position_key] > 0

    @pytest.mark.asyncio
    async def test_shadow_second_signal_same_market_rejected(self, tmp_path):
        """After filling a market to the cap, a second signal gets no fill."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # First signal — should be approved and fill at the per-market cap
        # Per-market cap = max_single_market_pct * max_total_exposure_pct * bankroll
        #                = 5% * 40% * $10k = $200
        sig1 = _make_signal(suggested_dollars=500.0)
        r1 = await engine.execute(sig1)
        assert r1.status == OrderStatus.SHADOW

        # Second signal — same market, already at cap; no meaningful fill
        # (REJECTED means risk gate blocked it; ZERO_SIZE means FP residual
        # slipped through the gate but rounded to 0 contracts — both are no-fills)
        sig2 = _make_signal(suggested_dollars=100.0)
        r2 = await engine.execute(sig2)
        assert r2.status in (OrderStatus.REJECTED, OrderStatus.ZERO_SIZE)

    @pytest.mark.asyncio
    async def test_shadow_total_exposure_cap(self, tmp_path):
        """Shadow trades accumulate toward the 40% total exposure cap."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Per-market cap = 5% * 40% * $10k = $200; total cap = 40% * $10k = $4000.
        # Need 20 trades × $200 = $4000 to fill the total exposure budget.
        for i in range(20):
            sig = _make_signal(
                market_id=f"MKT-{i:03d}",
                market_ticker=f"MKT-{i:03d}-T",
                suggested_dollars=500.0,
            )
            await engine.execute(sig)

        # Next trade should be rejected — total exposure cap hit
        sig_over = _make_signal(
            market_id="MKT-OVER",
            market_ticker="MKT-OVER-T",
            suggested_dollars=100.0,
        )
        record = await engine.execute(sig_over)
        assert record.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_shadow_writes_log_file(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        await engine.execute(signal)

        log_path = tmp_path / "state" / "trades.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["market_ticker"] == "MKT-001-T"


# ── Position key fallback ─────────────────────────────────────────────────


class TestPositionKeyFallback:
    """market_id is often empty for weather markets; must fall back to ticker."""

    @pytest.mark.asyncio
    async def test_empty_market_id_uses_ticker(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(market_id="", market_ticker="KXHIGHNYC-26MAR16-T55")
        await engine.execute(signal)

        assert "KXHIGHNYC-26MAR16-T55" in risk.positions
        assert "" not in risk.positions

    @pytest.mark.asyncio
    async def test_nonempty_market_id_preferred(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(market_id="abc-123", market_ticker="KXHIGHNYC-26MAR16-T55")
        await engine.execute(signal)

        assert "abc-123" in risk.positions

    @pytest.mark.asyncio
    async def test_empty_id_dedup_works(self, tmp_path):
        """Two signals with empty market_id but same ticker must dedup."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig1 = _make_signal(market_id="", market_ticker="KXHIGH-T55", suggested_dollars=500.0)
        sig2 = _make_signal(market_id="", market_ticker="KXHIGH-T55", suggested_dollars=100.0)

        r1 = await engine.execute(sig1)
        r2 = await engine.execute(sig2)

        assert r1.status == OrderStatus.SHADOW
        # REJECTED means risk gate blocked it; ZERO_SIZE means FP residual
        # slipped the gate but rounded to 0 contracts — both are no-fills.
        assert r2.status in (OrderStatus.REJECTED, OrderStatus.ZERO_SIZE)


# ── Shadow settlement ────────────────────────────────────────────────────


class TestShadowSettlement:
    """Shadow settlement must release exposure when close_time has passed."""

    @pytest.mark.asyncio
    async def test_expired_market_settled(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        signal = _make_signal(metadata={"close_time": past})
        await engine.execute(signal)

        assert risk.total_exposure > 0
        settled = await engine.settle_expired_positions()
        assert len(settled) == 1
        assert risk.total_exposure == 0

    @pytest.mark.asyncio
    async def test_future_market_not_settled(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        future = (datetime.now(UTC) + timedelta(hours=24)).isoformat()
        signal = _make_signal(metadata={"close_time": future})
        await engine.execute(signal)

        settled = await engine.settle_expired_positions()
        assert len(settled) == 0
        assert risk.total_exposure > 0

    @pytest.mark.asyncio
    async def test_settlement_frees_capacity_for_new_trades(self, tmp_path):
        """After settlement, the same market can be re-entered."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        sig1 = _make_signal(suggested_dollars=500.0, metadata={"close_time": past})
        await engine.execute(sig1)
        assert risk.total_exposure > 0

        # Settle
        await engine.settle_expired_positions()
        assert risk.total_exposure == 0

        # New signal for same market should now be approved
        sig2 = _make_signal(
            market_id="MKT-002",
            market_ticker="MKT-002-T",
            suggested_dollars=100.0,
            metadata={"close_time": (datetime.now(UTC) + timedelta(hours=24)).isoformat()},
        )
        r2 = await engine.execute(sig2)
        assert r2.status == OrderStatus.SHADOW

    @pytest.mark.asyncio
    async def test_already_settled_not_double_counted(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        signal = _make_signal(metadata={"close_time": past})
        await engine.execute(signal)

        settled1 = await engine.settle_expired_positions()
        settled2 = await engine.settle_expired_positions()
        assert len(settled1) == 1
        assert len(settled2) == 0  # already marked SETTLED

    @pytest.mark.asyncio
    async def test_missing_close_time_not_settled(self, tmp_path):
        """Signals without close_time metadata are never auto-settled."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(metadata={})
        await engine.execute(signal)

        settled = await engine.settle_expired_positions()
        assert len(settled) == 0
        assert risk.total_exposure > 0

    @pytest.mark.asyncio
    async def test_invalid_close_time_skipped(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(metadata={"close_time": "not-a-date"})
        await engine.execute(signal)

        settled = await engine.settle_expired_positions()
        assert len(settled) == 0

    @pytest.mark.asyncio
    async def test_z_suffix_close_time_parsed(self, tmp_path):
        """Kalshi uses Z suffix for UTC timestamps."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = "2020-01-01T00:00:00Z"
        signal = _make_signal(metadata={"close_time": past})
        await engine.execute(signal)

        settled = await engine.settle_expired_positions()
        assert len(settled) == 1

    @pytest.mark.asyncio
    async def test_settlement_row_written_to_trade_log(self, tmp_path):
        """Shadow settlement must emit a settlement row so ShadowBroker can
        drop the position on next restart."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        signal = _make_signal(metadata={"close_time": past})
        await engine.execute(signal)
        await engine.settle_expired_positions()

        log_path = tmp_path / "state" / "trades.jsonl"
        lines = log_path.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines]

        fills = [e for e in entries if e.get("event") == "fill"]
        settlements = [e for e in entries if e.get("event") == "settlement"]
        assert len(fills) == 1
        assert len(settlements) == 1
        s = settlements[0]
        assert s["market_ticker"] == signal.market_ticker
        assert s["position_key"] == signal.position_key
        assert s["result"] == "expired"
        assert s["side"] == signal.side.value

    @pytest.mark.asyncio
    async def test_settlement_write_failure_skips_close(self, tmp_path):
        """If the settlement row cannot be written, the risk manager must
        NOT mutate — otherwise log and in-memory state diverge and
        ShadowBroker's replay would resurrect a closed position."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        signal = _make_signal(metadata={"close_time": past})
        await engine.execute(signal)
        assert risk.total_exposure > 0
        pre = risk.total_exposure

        # Patch the strict append so the settlement write fails.
        async def broken(entries):
            raise OSError("disk full")

        with patch.object(engine, "_append_trade_log_strict", side_effect=broken):
            settled = await engine.settle_expired_positions()

        assert settled == []  # no settlements reported
        assert risk.total_exposure == pre  # risk manager untouched


# ── Live settlement ──────────────────────────────────────────────────────


class TestLiveSettlement:
    """Live mode settlement polls the Kalshi API and computes P&L."""

    @pytest.mark.asyncio
    async def test_settled_market_releases_exposure(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        # Manually place a position (simulating a filled order with fills payload)
        signal = _make_signal(side=Side.YES)
        decision = risk.check_signal(signal)
        record = OrderRecord(
            order_id="order-1",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=2,
            requested_count=2,
            status=OrderStatus.FILLED,
            fills_payload_received=True,
        )
        engine._orders["order-1"] = record
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        # Mock Kalshi API: market settled with YES result
        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"},
        }

        settled = await engine.settle_expired_positions()
        assert any(s.position_key == "MKT-001" for s in settled)
        assert risk.total_exposure == 0
        assert record.status == OrderStatus.SETTLED

    @pytest.mark.asyncio
    async def test_open_market_not_settled(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        decision = risk.check_signal(signal)
        record = OrderRecord(
            order_id="order-1",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=2,
            requested_count=2,
            status=OrderStatus.FILLED,
        )
        engine._orders["order-1"] = record
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        client.get_market.return_value = {
            "market": {"status": "open", "result": ""},
        }

        settled = await engine.settle_expired_positions()
        assert len(settled) == 0
        assert risk.total_exposure == 100.0

    @pytest.mark.asyncio
    async def test_no_client_returns_empty(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        settled = await engine.settle_expired_positions()
        assert len(settled) == 0

    @pytest.mark.asyncio
    async def test_api_error_does_not_crash(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        decision = risk.check_signal(signal)
        record = OrderRecord(
            order_id="order-1",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=2,
            requested_count=2,
            status=OrderStatus.FILLED,
        )
        engine._orders["order-1"] = record
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        client.get_market.side_effect = Exception("API timeout")

        # Should not raise
        settled = await engine.settle_expired_positions()
        assert len(settled) == 0
        assert risk.total_exposure == 100.0  # unchanged


# ── P&L computation ──────────────────────────────────────────────────────


class TestSettlementPnL:
    """P&L math must be correct — wrong sign means tracking wrong bankroll."""

    def _make_engine(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        return ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

    def _add_record(
        self, engine, *, market_id="MKT-001", side=Side.YES,
        fill_price=0.50, fill_count=10,
    ):
        from python.analytics.fees import order_fee as _order_fee
        signal = _make_signal(market_id=market_id, side=side)
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        # Set fills_payload_received=True so _settle_orders takes the authoritative
        # path.  fees_from_api matches the taker fee formula the old code used, so
        # existing P&L assertions remain valid after the 2→3-tuple refactor.
        fees_from_api = _order_fee(fill_price, fill_count, is_maker=False)
        record = OrderRecord(
            order_id=f"order-{len(engine._orders)}",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=fill_price,
            order_price=fill_price,
            fill_price=fill_price,
            fill_count=fill_count,
            requested_count=fill_count,
            status=OrderStatus.FILLED,
            fills_payload_received=True,
            fees_from_api=fees_from_api,
            is_taker_fills=fill_count,
        )
        engine._orders[record.order_id] = record

    def test_yes_wins(self, tmp_path):
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.YES, fill_price=0.40, fill_count=10)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        # (1.0 - 0.4) * 10 - fee; fee = ceil(0.07*10*0.4*0.6*100)/100 = 0.17
        assert pnl == pytest.approx(5.83)

    def test_yes_loses(self, tmp_path):
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.YES, fill_price=0.40, fill_count=10)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "no")
        # -0.4 * 10 - fee = -4.0 - 0.17 = -4.17
        assert pnl == pytest.approx(-4.17)

    def test_no_wins(self, tmp_path):
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.NO, fill_price=0.60, fill_count=5)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "no")
        # (1.0 - 0.6) * 5 - fee; fee = ceil(0.07*5*0.6*0.4*100)/100 = 0.09
        assert pnl == pytest.approx(1.91)

    def test_no_loses(self, tmp_path):
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.NO, fill_price=0.60, fill_count=5)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        # -0.6 * 5 - fee = -3.0 - 0.09 = -3.09
        assert pnl == pytest.approx(-3.09)

    def test_no_fills_zero_pnl(self, tmp_path):
        engine = self._make_engine(tmp_path)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        assert pnl == 0.0

    def test_multiple_fills_summed(self, tmp_path):
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.YES, fill_price=0.40, fill_count=10)
        self._add_record(engine, side=Side.YES, fill_price=0.50, fill_count=5)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        # (1-0.4)*10 - fee1 + (1-0.5)*5 - fee2
        # fee1 = ceil(0.07*10*0.4*0.6*100)/100 = 0.17
        # fee2 = ceil(0.07*5*0.5*0.5*100)/100 = 0.09
        # = 6.0 - 0.17 + 2.5 - 0.09 = 8.24
        assert pnl == pytest.approx(8.24)

    def test_only_matching_market(self, tmp_path):
        """P&L computation only considers records for the given market."""
        engine = self._make_engine(tmp_path)
        self._add_record(engine, market_id="MKT-001", side=Side.YES,
                         fill_price=0.40, fill_count=10)
        self._add_record(engine, market_id="MKT-OTHER", side=Side.YES,
                         fill_price=0.80, fill_count=20)
        pnl, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        # only MKT-001: 6.0 - fee(0.40, 10) = 6.0 - 0.17 = 5.83
        assert pnl == pytest.approx(5.83)

    def test_double_settle_not_double_counted(self, tmp_path):
        """Calling _settle_orders twice for the same position returns P&L only once.

        Regression: periodic hydrate re-adds already-closed positions to the risk
        manager, causing _settle_live to call _settle_orders a second time for the
        same position. Records already marked SETTLED must be skipped.
        """
        engine = self._make_engine(tmp_path)
        self._add_record(engine, side=Side.NO, fill_price=0.69, fill_count=14)
        pnl1, _, _settled, _per_rec = engine._settle_orders("MKT-001", "yes")
        pnl2, _, _settled2, _per_rec2 = engine._settle_orders("MKT-001", "yes")
        assert pnl1 != 0.0
        assert pnl2 == 0.0  # second call must produce no P&L


# ── Risk rejection ───────────────────────────────────────────────────────


class TestRiskRejection:
    """Signals rejected by risk must not update positions or write logs."""

    @pytest.mark.asyncio
    async def test_rejected_signal_no_position(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Trigger kill switch
        (tmp_path / "state" / "STOP").touch()

        signal = _make_signal()
        record = await engine.execute(signal)
        assert record.status == OrderStatus.REJECTED
        assert risk.total_exposure == 0

    @pytest.mark.asyncio
    async def test_zero_contracts_skipped(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # dollars_to_contracts floors: need price high enough that
        # floor(dollars / price) = 0
        signal = _make_signal(suggested_dollars=0.001, market_price=0.50)
        record = await engine.execute(signal)
        assert record.status in (OrderStatus.ZERO_SIZE, OrderStatus.SKIPPED, OrderStatus.REJECTED)


# ── Batch execution ──────────────────────────────────────────────────────


class TestBatchExecution:
    """Batch execution must apply risk limits cumulatively across signals."""

    @pytest.mark.asyncio
    async def test_batch_respects_total_exposure(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Per-market cap = 5% * 40% * $10k = $200; total cap = 40% * $10k = $4000.
        # 20 signals × $200 = $4000 exhausts the budget; 21st is blocked.
        signals = [
            _make_signal(
                market_id=f"MKT-{i:03d}",
                market_ticker=f"MKT-{i:03d}-T",
                suggested_dollars=500.0,
            )
            for i in range(21)
        ]

        records = await engine.execute_batch(signals)
        approved = [r for r in records if r.status == OrderStatus.SHADOW]
        blocked = [
            r
            for r in records
            if r.status in (OrderStatus.REJECTED, OrderStatus.ZERO_SIZE)
        ]

        # First 20 fill at $200 each = $4000 total; last 1 blocked
        assert len(approved) == 20
        assert len(blocked) == 1

    @pytest.mark.asyncio
    async def test_batch_deduplicates_same_market(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Two signals for the same market
        signals = [
            _make_signal(suggested_dollars=500.0),
            _make_signal(suggested_dollars=100.0),
        ]

        records = await engine.execute_batch(signals)
        assert records[0].status == OrderStatus.SHADOW
        assert records[1].status == OrderStatus.SKIPPED  # dedup catches before risk

    @pytest.mark.asyncio
    async def test_two_signal_batch_respects_total_exposure_cap(self, tmp_path):
        # Future-refactor tripwire: if execute_batch is ever switched from
        # sequential to asyncio.gather without adding atomicity around the
        # risk manager's check_signal -> record_fill window, signals will
        # read zero exposure, pass the check, and all record_fill —
        # breaching the cap. This test fails in that scenario because all
        # records would land as SHADOW with the cap blown.
        #
        # Per-market cap = max_single_market_pct * max_total_exposure_pct * bankroll
        #                = 5% * 5% * $10k = $25 per market.
        # Total cap = max_total_exposure_pct * bankroll = 5% * $10k = $500.
        # 20 signals × $25 = $500 exhausts the budget; 21st is TOTAL_EXPOSURE_CAP.
        settings = Settings(
            state_dir=str(tmp_path / "state"),
            execution={"mode": "shadow"},
            risk={"max_total_exposure_pct": 0.05, "max_single_market_pct": 0.05},
            _env_file=None,
        )
        (tmp_path / "state").mkdir(exist_ok=True)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # 21 signals on distinct markets; first 20 fill $25 each = $500 total.
        # 21st has no room: TOTAL_EXPOSURE_CAP (or ZERO_SIZE from FP residual).
        signals = [
            _make_signal(
                market_id=f"MKT-{i:03d}",
                market_ticker=f"MKT-{i:03d}-T",
                suggested_dollars=500.0,
            )
            for i in range(21)
        ]

        records = await engine.execute_batch(signals)
        approved = [r for r in records if r.status == OrderStatus.SHADOW]
        blocked = [
            r
            for r in records
            if r.status in (OrderStatus.REJECTED, OrderStatus.ZERO_SIZE)
        ]

        assert len(approved) == 20, f"expected 20 fills, got {len(approved)}"
        assert len(blocked) >= 1
        # Verify at least one TOTAL_EXPOSURE_CAP rejection (not just a size trim)
        total_cap_blocked = [
            r
            for r in blocked
            if any(
                "TOTAL_EXPOSURE_CAP" in reason
                for reason in r.risk_decision.rejection_reasons
            )
        ]
        assert len(total_cap_blocked) >= 1


# ── Live mode routing ────────────────────────────────────────────────────


class TestLiveModeRouting:
    """Live mode must call the Kalshi API, not just log."""

    @pytest.mark.asyncio
    async def test_live_calls_place_order(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-123", "status": "pending"},
        }
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        record = await engine.execute(signal)

        client.place_order.assert_called_once()
        assert record.order_id == "ORD-123"

    @pytest.mark.asyncio
    async def test_live_api_error_returns_error_status(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.side_effect = Exception("Connection refused")
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        record = await engine.execute(signal)
        assert record.status == OrderStatus.ERROR

    @pytest.mark.asyncio
    async def test_live_no_client_raises(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        with pytest.raises(RuntimeError, match="without a Kalshi client"):
            await engine.execute(signal)

    @pytest.mark.asyncio
    async def test_shadow_never_calls_api(self, tmp_path):
        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        await engine.execute(signal)

        client.place_order.assert_not_called()


# ── NO-side price conversion ─────────────────────────────────────────────


class TestNoSidePriceConversion:
    """NO-side orders must convert price: order_price = 1 - market_price."""

    @pytest.mark.asyncio
    async def test_shadow_no_side_fill_price(self, tmp_path):
        """Shadow NO fill_price must be 1 - market_price (side-denominated)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(side=Side.NO, market_price=0.60)
        record = await engine.execute(signal)

        assert record.order_price == pytest.approx(0.40)
        assert record.fill_price == pytest.approx(0.40)

    @pytest.mark.asyncio
    async def test_shadow_yes_side_fill_price(self, tmp_path):
        """Shadow YES fill_price must equal market_price (unchanged)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(side=Side.YES, market_price=0.60)
        record = await engine.execute(signal)

        assert record.order_price == pytest.approx(0.60)
        assert record.fill_price == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_live_no_side_sends_no_price(self, tmp_path):
        """Live NO order must send no_price (not yes_price) in cents."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-NO-1", "status": "pending"},
        }
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal(side=Side.NO, market_price=0.60)
        await engine.execute(signal)

        call_kwargs = client.place_order.call_args.kwargs
        assert call_kwargs["no_price"] == 40
        assert call_kwargs.get("yes_price") is None
        assert call_kwargs["side"] == "no"

    @pytest.mark.asyncio
    async def test_live_yes_side_sends_yes_price(self, tmp_path):
        """Live YES order must send yes_price in cents."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-YES-1", "status": "pending"},
        }
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal(side=Side.YES, market_price=0.60)
        await engine.execute(signal)

        call_kwargs = client.place_order.call_args.kwargs
        assert call_kwargs["yes_price"] == 60
        assert call_kwargs.get("no_price") is None
        assert call_kwargs["side"] == "yes"

    @pytest.mark.asyncio
    async def test_no_side_penny_market(self, tmp_path):
        """Near-certain YES (0.95) → NO price should be 5 cents."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-NO-2", "status": "pending"},
        }
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal(side=Side.NO, market_price=0.95)
        await engine.execute(signal)

        call_kwargs = client.place_order.call_args.kwargs
        assert call_kwargs["no_price"] == 5

    @pytest.mark.asyncio
    async def test_shadow_no_side_pnl_correct(self, tmp_path):
        """Full round-trip: shadow NO trade → settlement → correct P&L."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Buy NO at market_price=0.40 (YES), so NO cost = 0.60
        signal = _make_signal(
            side=Side.NO,
            market_price=0.40,
            model_probability=0.25,
        )
        record = await engine.execute(signal)
        assert record.fill_price == pytest.approx(0.60)

        # Compute P&L — NO wins (includes exchange fee).
        # Simulate the fills payload arriving (as check_fills would do in live mode)
        # so _settle_orders takes the authoritative path instead of deferring.
        from python.analytics.fees import order_fee as kalshi_order_fee

        fee = kalshi_order_fee(0.60, record.fill_count)
        record.fills_payload_received = True
        record.fees_from_api = fee

        position_key = signal.position_key
        pnl, _, _settled, _per_rec = engine._settle_orders(position_key, "no")
        # Bought NO at 0.60, NO wins: profit = (1.0 - 0.60) * contracts - fee
        expected = (1.0 - 0.60) * record.fill_count - fee
        assert pnl == pytest.approx(expected)


# ── check_fills and check_all_pending_fills ───────────────────────────────


def _make_live_engine(tmp_path, client=None, trade_log_lock=None, **execution_kw: Any):
    """Create a live-mode engine with a mock client."""
    settings = _make_settings(tmp_path, mode="live", **execution_kw)
    risk = RiskManager(settings)
    if client is None:
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-1", "status": "resting"},
        }
        client.cancel_order.return_value = {}
    engine = ExecutionEngine(
        settings, kalshi_client=client, risk_manager=risk, trade_log_lock=trade_log_lock
    )
    return engine, risk, client


class TestCheckFills:
    """check_fills must update risk manager with fill data."""

    @pytest.mark.asyncio
    async def test_check_fills_records_exposure(self, tmp_path):
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()  # default side=Side.YES
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                {"side": "yes", "yes_price_dollars": "0.5000",
                 "no_price_dollars": "0.5000", "count_fp": "5.00"},
            ],
        }

        result = await engine.check_fills(record.order_id)
        assert result.fill_count == 5
        assert result.fill_price == pytest.approx(0.50)
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] > 0

    @pytest.mark.asyncio
    async def test_check_fills_no_double_counting(self, tmp_path):
        """Calling check_fills twice must not double exposure."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                {"side": "yes", "yes_price_dollars": "0.5000",
                 "no_price_dollars": "0.5000", "count_fp": "5.00"},
            ],
        }

        await engine.check_fills(record.order_id)
        position_key = signal.market_id or signal.market_ticker
        exposure_after_first = risk.positions[position_key]

        await engine.check_fills(record.order_id)
        exposure_after_second = risk.positions[position_key]

        assert exposure_after_first == exposure_after_second

    @pytest.mark.asyncio
    async def test_check_fills_partial_then_more(self, tmp_path):
        """Partial fill then more fills must record delta only."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # First check: 3 contracts filled at 50c
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "resting"},
        }
        client.get_fills.return_value = {
            "fills": [
                {"side": "yes", "yes_price_dollars": "0.5000",
                 "no_price_dollars": "0.5000", "count_fp": "3.00"},
            ],
        }
        await engine.check_fills(record.order_id)
        position_key = signal.market_id or signal.market_ticker
        exposure_1 = risk.positions[position_key]
        assert exposure_1 == pytest.approx(3 * 0.50)

        # Second check: now 7 contracts filled at 50c average
        client.get_order.return_value = {
            "order": {"fill_count_fp": "7.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                {"side": "yes", "yes_price_dollars": "0.5000",
                 "no_price_dollars": "0.5000", "count_fp": "7.00"},
            ],
        }
        await engine.check_fills(record.order_id)
        exposure_2 = risk.positions[position_key]
        assert exposure_2 == pytest.approx(7 * 0.50)

    @pytest.mark.asyncio
    async def test_check_fills_shadow_noop(self, tmp_path):
        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)
        result = await engine.check_fills("any-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_fills_unfilled_order(self, tmp_path):
        """A resting (unfilled) order keeps its tentative exposure in last_recorded_dollars.

        Before fix #19 this was 0.0 because no tentative exposure was recorded.
        After the fix, last_recorded_dollars holds the tentative amount set at
        order submission so the risk manager sees exposure immediately.
        """
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # Tentative exposure must already be recorded before check_fills.
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0

        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }
        result = await engine.check_fills(record.order_id)
        assert result.fill_count == 0
        # last_recorded_dollars holds the tentative reservation (still resting)
        assert result.last_recorded_dollars > 0

    @pytest.mark.asyncio
    async def test_check_fills_api_error_survives(self, tmp_path):
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.side_effect = Exception("API down")
        result = await engine.check_fills(record.order_id)
        assert result is not None  # returns record, doesn't crash

    @pytest.mark.asyncio
    async def test_check_fills_404_logged_quietly(self, tmp_path, caplog):
        """A post-placement get_order 404 (query-exchange read-after-write race)
        is expected and recovered by the caller (_recover_after_immediate_404 +
        the FillsReconciler) — check_fills must log it quietly, not as an ERROR
        traceback. The 2026-06 V1-deprecation outage went 2 days unnoticed
        precisely because real failures hid in per-order ERROR noise.
        """
        import logging

        from python.data.kalshi_client import KalshiAPIError

        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.side_effect = KalshiAPIError(
            404, {"error": {"code": "not_found", "message": "not found"}}
        )

        with caplog.at_level(logging.DEBUG, logger="python.execution.engine"):
            result = await engine.check_fills(record.order_id)

        assert result is not None
        assert result.fill_count == 0  # left for fills-based recovery
        # check_fills must ask the client to log the anticipated 404 quietly too.
        client.get_order.assert_called_once_with(
            record.order_id, expected_error_statuses={404}
        )
        engine_errors = [
            r
            for r in caplog.records
            if r.name == "python.execution.engine" and r.levelno >= logging.ERROR
        ]
        assert not engine_errors, (
            f"expected no ERROR logs for the recovered 404, got "
            f"{[r.getMessage() for r in engine_errors]}"
        )

    @pytest.mark.asyncio
    async def test_check_fills_non_404_error_logged_loud(self, tmp_path, caplog):
        """A non-404 API error is unexpected — check_fills must still log it at
        ERROR so a genuine fill-check failure is never silently swallowed.
        """
        import logging

        from python.data.kalshi_client import KalshiAPIError

        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.side_effect = KalshiAPIError(500, {"error": "server boom"})

        with caplog.at_level(logging.DEBUG, logger="python.execution.engine"):
            result = await engine.check_fills(record.order_id)

        assert result is not None
        engine_errors = [
            r
            for r in caplog.records
            if r.name == "python.execution.engine" and r.levelno >= logging.ERROR
        ]
        assert engine_errors, "a non-404 API error must be logged at ERROR"

    @pytest.mark.asyncio
    async def test_check_fills_404_from_get_fills_stays_loud(self, tmp_path, caplog):
        """The 404-quieting is scoped to get_order only. A 404 from a LATER call
        (get_fills, which runs once the order reports a fill) is a genuine
        anomaly — it must still log at ERROR, not be swallowed by the get_order
        propagation-lag path.
        """
        import logging

        from python.data.kalshi_client import KalshiAPIError

        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.side_effect = KalshiAPIError(
            404, {"error": {"code": "not_found"}}
        )

        with caplog.at_level(logging.DEBUG, logger="python.execution.engine"):
            result = await engine.check_fills(record.order_id)

        assert result is not None
        engine_errors = [
            r
            for r in caplog.records
            if r.name == "python.execution.engine" and r.levelno >= logging.ERROR
        ]
        assert engine_errors, "a 404 from get_fills (not get_order) must stay loud"


class TestCheckFillsCapturesIsTaker:
    """check_fills must aggregate is_taker counts and API-reported fees per fill.

    Kalshi's /portfolio/fills returns one record per fill with `is_taker: bool`
    and a fee field (cents).  We aggregate those onto OrderRecord so downstream
    accounting and edge-math calibration use ground truth instead of guessing
    the maker/taker split from the limit price.
    """

    @pytest.mark.asyncio
    async def test_captures_taker_count_and_fee(self, tmp_path):
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()  # default side=Side.YES
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "50.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                # 30 contracts as taker @ 45c YES, $0.42 fee
                {"side": "yes", "yes_price_dollars": "0.4500",
                 "no_price_dollars": "0.5500", "count_fp": "30.00",
                 "is_taker": True, "fee_cost": "0.420000"},
                # 20 contracts as maker @ 45c YES, $0.07 fee
                {"side": "yes", "yes_price_dollars": "0.4500",
                 "no_price_dollars": "0.5500", "count_fp": "20.00",
                 "is_taker": False, "fee_cost": "0.070000"},
            ],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        assert result.is_taker_fills == 30
        assert result.is_maker_fills == 20
        assert result.fees_from_api == pytest.approx(0.49)

    @pytest.mark.asyncio
    async def test_trade_log_contains_is_taker_breakdown(self, tmp_path):
        """check_fills must write is_taker_fills/is_maker_fills/fees_from_api to log."""
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "10.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                {"side": "yes", "yes_price_dollars": "0.5000",
                 "no_price_dollars": "0.5000", "count_fp": "10.00",
                 "is_taker": False, "fee_cost": "0.090000"},
            ],
        }

        await engine.check_fills(record.order_id)

        log_path = engine._trade_log_path
        assert log_path.exists()
        entries = [
            json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
        ]
        # Last entry is the fill row from check_fills.
        fill_entries = [e for e in entries if e.get("event") == "fill"]
        assert fill_entries, "expected a fill row in the trade log"
        last = fill_entries[-1]
        assert last["is_taker_fills"] == 0
        assert last["is_maker_fills"] == 10
        assert last["fees_from_api"] == pytest.approx(0.09)

    @pytest.mark.asyncio
    async def test_unattributed_fills_dont_break(self, tmp_path):
        """Older fills missing is_taker should still update price/count."""
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "10.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000",
                       "no_price_dollars": "0.5000", "count_fp": "10.00"}],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        assert result.fill_count == 10
        assert result.is_taker_fills == 0
        assert result.is_maker_fills == 0
        assert result.fees_from_api == 0.0


class TestFillsPayloadReceived:
    """B1/T18: fills_payload_received must be set based on loop completion,
    not on len(fills) or is_taker classification.

    Semantics:
      - Reset to False at the top of the try-block (every tick).
      - Set to True ONLY after the for-loop over fills completes cleanly.
      - is_taker=None must NOT block the True transition.
    """

    @pytest.mark.asyncio
    async def test_check_fills_sets_fills_payload_received_with_is_taker_none(self, tmp_path):
        """B1/T18: is_taker=None on a fill MUST NOT block fills_payload_received=True.

        Verifies the gate is loop-completion, not taker/maker classification.
        """
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()  # default side=Side.YES
        record = await engine.execute(signal)

        # 5 contracts total; Kalshi returns counts as decimal strings (e.g. "5.00")
        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        # Two fills: first has is_taker=None (unattributed), second is taker.
        # fee_cost is a dollar string matching the pattern in existing tests.
        # 0.04 + 0.21 = $0.25 total fees.
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "2.00",
                    "is_taker": None,
                    "fee_cost": "0.040000",
                },
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "3.00",
                    "is_taker": True,
                    "fee_cost": "0.210000",
                },
            ],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        # Primary assertion: loop completed → flag must be True
        assert result.fills_payload_received is True
        # fee_cost values: 0.04 + 0.21 = $0.25 total fees
        assert result.fees_from_api == pytest.approx(0.25)
        # Only the is_taker=True fill (3 contracts) counts as taker
        assert result.is_taker_fills == 3
        # is_taker=None did NOT increment maker count
        assert result.is_maker_fills == 0

    @pytest.mark.asyncio
    async def test_fills_payload_received_set_on_short_payload(self, tmp_path):
        """B2/T18b: get_fills returns 1 fill but fill_count is 100 →
        receive flag still True (pagination-safe; gate is loop-completion, not count-match)."""
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()  # default side=Side.YES
        record = await engine.execute(signal)

        # Mock get_order to return fill_count_fp=100_000_000_000 (100 contracts).
        client.get_order.return_value = {
            "order": {"fill_count_fp": "100000000000.00", "status": "executed"},
        }
        # Mock get_fills to return ONLY ONE fill with 1 contract.
        # The loop will complete after one iteration, setting the flag to True.
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "1.00",
                    "is_taker": True,
                    "fee_cost": "7.000000",
                },
            ],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        # Primary assertion: loop completed despite short payload → flag must be True
        assert result.fills_payload_received is True
        # fee_cost is $7.00
        assert result.fees_from_api == pytest.approx(7.00)
        # is_taker=True fill (1 contract)
        assert result.is_taker_fills == 1

    @pytest.mark.asyncio
    async def test_fills_payload_received_reset_on_get_fills_exception(self, tmp_path):
        """B3/T18c: Tick 1 sets True; tick 2 get_fills raises; receive flag must be False after.
        Verifies the reset-at-top-of-try-block semantics."""
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()  # default side=Side.YES
        record = await engine.execute(signal)

        # Tick 1 — clean: get_order returns fill_count_fp=5_000_000_000;
        # get_fills returns one fill that matches.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "5.00",
                    "is_taker": True,
                    "fee_cost": "35.000000",
                },
            ],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        # First tick sets the flag to True
        assert result.fills_payload_received is True

        # Tick 2 — get_fills raises RuntimeError
        # Note: check_fills swallows exceptions internally (engine.py:611-613)
        # DO NOT use pytest.raises. The flag should be reset on the next call
        # before the exception occurs.
        client.get_fills.side_effect = RuntimeError("kalshi 429")

        # Call check_fills again; it will internally swallow the exception
        result2 = await engine.check_fills(record.order_id)

        # After the exception, the flag must be False (reset at top of try-block)
        assert result2 is not None
        assert result2.fills_payload_received is False


class TestCheckAllPendingFills:
    """Batch fill check iterates pending orders only.

    Returns ``(updated_with_fills, zero_fill_cancellations)``. The
    cancellations list lets the orchestrator release per-strategy
    ticker dedup state (Bug B).
    """

    @pytest.mark.asyncio
    async def test_processes_pending_orders(self, tmp_path):
        engine, risk, client = _make_live_engine(tmp_path)

        # Place two orders with distinct order IDs
        client.place_order.side_effect = [
            {"order": {"order_id": "ORD-A", "status": "resting"}},
            {"order": {"order_id": "ORD-B", "status": "resting"}},
        ]
        s1 = _make_signal(market_id="A", market_ticker="A-T")
        s2 = _make_signal(market_id="B", market_ticker="B-T")
        await engine.execute(s1)
        await engine.execute(s2)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.6000", "no_price_dollars": "0.4000", "count_fp": "3.00"}],
        }

        updated, cancelled = await engine.check_all_pending_fills()
        assert len(updated) == 2
        assert cancelled == []
        assert "A" in risk.positions or "A-T" in risk.positions
        assert "B" in risk.positions or "B-T" in risk.positions

    @pytest.mark.asyncio
    async def test_skips_filled_orders(self, tmp_path):
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # Manually mark as filled
        record.status = OrderStatus.FILLED

        updated, cancelled = await engine.check_all_pending_fills()
        assert len(updated) == 0
        assert cancelled == []
        client.get_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_shadow(self, tmp_path):
        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)
        result = await engine.check_all_pending_fills()
        assert result == ([], [])

    @pytest.mark.asyncio
    async def test_noop_no_pending(self, tmp_path):
        engine, risk, client = _make_live_engine(tmp_path)
        updated, cancelled = await engine.check_all_pending_fills()
        assert updated == []
        assert cancelled == []
        client.get_order.assert_not_called()


class TestZeroFillCancellationReturn:
    """Bug B: ``check_all_pending_fills`` must surface zero-fill
    cancellations (TTL expiry + server-cancellation) so the orchestrator
    can release per-strategy ticker dedup state. Partial-fill cancellations
    are NOT surfaced (the strategy already traded the ticker)."""

    @pytest.mark.asyncio
    async def test_returns_ttl_zero_fill_cancellations(self, tmp_path):
        """A TTL-expired order with zero fills must appear in the
        cancellations list returned to the orchestrator."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        signal = _make_signal(market_ticker="KXHIGHNY-99APR14-B70.5")
        record = await engine.execute(signal)
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }

        updated, cancelled = await engine.check_all_pending_fills()
        assert updated == []
        assert len(cancelled) == 1
        assert cancelled[0].order_id == record.order_id
        assert cancelled[0].fill_count == 0
        assert cancelled[0].status == OrderStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_excludes_partial_fill_ttl_cancellations(self, tmp_path):
        """A partial-fill order then TTL-expired must NOT be surfaced as a
        zero-fill cancellation (the strategy successfully traded part of it)."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        signal = _make_signal(
            market_ticker="KXHIGHNY-99APR14-B70.5",
            market_price=0.50,
            suggested_dollars=25.0,
        )
        record = await engine.execute(signal)

        # Tick once with a partial fill
        client.get_order.return_value = {
            "order": {"fill_count_fp": "20.00", "status": "resting"},
        }
        client.get_fills.return_value = {"fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "20.00"}]}
        await engine.check_all_pending_fills()
        assert record.status == OrderStatus.PARTIAL

        # Backdate past TTL and tick again
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)
        updated, cancelled = await engine.check_all_pending_fills()
        # PARTIAL → has fills → goes into updated; TTL expiry sees fill_count > 0 so
        # it does NOT count as a zero-fill cancellation.
        assert all(r.order_id != record.order_id for r in cancelled)

    @pytest.mark.asyncio
    async def test_returns_server_cancellation(self, tmp_path):
        """When Kalshi reports status="canceled" with zero fills, the record
        must appear in the cancellations list (server-side TTL or operator action)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_ticker="KXHIGHNY-99APR14-B70.5")
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "canceled"},
        }

        updated, cancelled = await engine.check_all_pending_fills()
        assert updated == []
        assert len(cancelled) == 1
        assert cancelled[0].order_id == record.order_id
        assert cancelled[0].status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_excludes_server_cancellation_with_partial_fill(self, tmp_path):
        """A server-side cancellation with non-zero fills is NOT a zero-fill
        cancellation (strategy already traded; dedup must persist)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(
            market_ticker="KXHIGHNY-99APR14-B70.5",
            market_price=0.50,
            suggested_dollars=25.0,
        )
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "20.00", "status": "canceled"},
        }
        client.get_fills.return_value = {"fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "20.00"}]}

        updated, cancelled = await engine.check_all_pending_fills()
        # fill_count > 0 → goes into updated, never into cancelled
        assert all(r.order_id != record.order_id for r in cancelled)
        assert any(r.order_id == record.order_id for r in updated)

    @pytest.mark.asyncio
    async def test_excludes_filled_orders(self, tmp_path):
        """A fully filled order must not appear in the cancellations list."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "executed"},
        }
        client.get_fills.return_value = {"fills": [{"side": "yes", "yes_price_dollars": "0.6000", "no_price_dollars": "0.4000", "count_fp": "3.00"}]}

        updated, cancelled = await engine.check_all_pending_fills()
        assert any(r.order_id == record.order_id for r in updated)
        assert cancelled == []


# ── Tentative exposure (#19) ─────────────────────────────────────────────


class TestTentativeExposure:
    """Fix #19: live orders record tentative exposure on submission so the risk
    manager blocks duplicate signals before fill confirmation arrives.

    The tentative amount (contracts × limit price) is conservative: it
    represents the worst-case cost assuming full fill at the submitted price.
    check_fills() reconciles up or down once the Kalshi API confirms the
    actual fill.
    """

    @pytest.mark.asyncio
    async def test_live_records_exposure_immediately_after_order(self, tmp_path):
        """Exposure must be in the risk manager right after place_order returns."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()

        assert risk.total_exposure == 0.0
        await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0.0

    @pytest.mark.asyncio
    async def test_live_second_signal_same_market_rejected_before_fill(self, tmp_path):
        """A second signal for the same market must be rejected on tentative exposure,
        even if check_fills() has not been called yet.
        """
        engine, risk, client = _make_live_engine(tmp_path)

        # Per-market cap = 5% * 40% * $10k = $200; use $500 to saturate the cap.
        client.place_order.side_effect = [
            {"order": {"order_id": "ORD-1", "status": "resting"}},
            {"order": {"order_id": "ORD-2", "status": "resting"}},
        ]
        sig1 = _make_signal(suggested_dollars=500.0)
        sig2 = _make_signal(suggested_dollars=100.0)

        r1 = await engine.execute(sig1)
        r2 = await engine.execute(sig2)

        assert r1.status == OrderStatus.RESTING
        # REJECTED means risk gate blocked it; ZERO_SIZE means FP residual
        # slipped the gate but rounded to 0 contracts — both are no-fills.
        assert r2.status in (OrderStatus.REJECTED, OrderStatus.ZERO_SIZE)

    @pytest.mark.asyncio
    async def test_tentative_exposure_correct_amount(self, tmp_path):
        """Tentative exposure equals contracts × order_price (worst-case cost)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)

        await engine.execute(signal)

        # contracts = floor(50.0 / 0.50) = 100; tentative = 100 * 0.50 = 50.0
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_check_fills_adjusts_down_when_fill_cheaper(self, tmp_path):
        """When actual fill price is lower than tentative, release_exposure reduces
        exposure to the confirmed cost.
        """
        engine, risk, client = _make_live_engine(tmp_path)
        # order_price = 0.50 for YES side
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        record = await engine.execute(signal)

        # Tentative: 100 contracts × $0.50 = $50.00
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] == pytest.approx(50.0)

        # Actual fill: 100 contracts but at $0.45 (better than limit) → $45.00
        client.get_order.return_value = {
            "order": {"fill_count_fp": "100.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.4500", "no_price_dollars": "0.5500", "count_fp": "100.00"}],  # 45 cents
        }
        await engine.check_fills(record.order_id)

        # Exposure must now reflect actual cost, not tentative.
        assert risk.positions[position_key] == pytest.approx(45.0)

    @pytest.mark.asyncio
    async def test_check_fills_adjusts_up_on_slippage(self, tmp_path):
        """When actual fill price is higher than tentative (price improvement reversed),
        record_fill adds the extra cost to exposure.
        """
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        record = await engine.execute(signal)

        # Tentative: 100 × $0.50 = $50.00
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] == pytest.approx(50.0)

        # Actual fill: 100 contracts at $0.52 (slippage) → $52.00
        client.get_order.return_value = {
            "order": {"fill_count_fp": "100.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5200", "no_price_dollars": "0.4800", "count_fp": "100.00"}],  # 52 cents
        }
        await engine.check_fills(record.order_id)

        assert risk.positions[position_key] == pytest.approx(52.0)

    @pytest.mark.asyncio
    async def test_cancelled_order_releases_tentative_exposure(self, tmp_path):
        """A cancelled (zero-fill) order must release all tentative exposure."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(suggested_dollars=500.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0.0

        # Simulate Kalshi reporting the order as cancelled with no fill.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "canceled"},
        }
        await engine.check_fills(record.order_id)

        assert risk.positions.get(position_key, 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_cancelled_after_partial_fill_releases_unfilled_portion(self, tmp_path):
        """A partially-filled then cancelled order retains only the filled cost."""
        engine, risk, client = _make_live_engine(tmp_path)
        # 50 contracts × $0.50 = $25.00 tentative
        signal = _make_signal(market_price=0.50, suggested_dollars=25.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        # Tentative: 50 × $0.50 = $25.00
        assert risk.positions[position_key] == pytest.approx(25.0)

        # Only 20 contracts actually filled before the order was cancelled.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "20.00", "status": "canceled"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "20.00"}],
        }
        await engine.check_fills(record.order_id)

        # Exposure should reflect only the 20 filled contracts × $0.50 = $10.00.
        assert risk.positions[position_key] == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_no_tentative_exposure_on_error(self, tmp_path):
        """A failed place_order call must not leave any tentative exposure."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.side_effect = Exception("Connection refused")
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        record = await engine.execute(signal)

        assert record.status == OrderStatus.ERROR
        assert risk.total_exposure == 0.0

    @pytest.mark.asyncio
    async def test_release_exposure_on_risk_manager_clamps_at_zero(self, tmp_path):
        """release_exposure never drives a position negative."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        risk.record_fill("MKT-A", "weather_edge", 10.0)
        risk.release_exposure("MKT-A", 50.0)  # release more than present

        # Position should be removed (not negative).
        assert "MKT-A" not in risk.positions
        assert risk.total_exposure == 0.0

    @pytest.mark.asyncio
    async def test_release_exposure_partial_reduction(self, tmp_path):
        """release_exposure reduces exposure by the specified amount."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        risk.record_fill("MKT-B", "weather_edge", 100.0)
        risk.release_exposure("MKT-B", 30.0)

        assert risk.positions["MKT-B"] == pytest.approx(70.0)


# ── VWAP-adjusted contract count (#17) ───────────────────────────────────


class TestVwapAdjustedContracts:
    """Fix #17: engine must use the depth-adjusted VWAP fill price from signal
    metadata when computing contract count, not the raw midpoint.

    Without this fix the engine calls:
        dollars_to_contracts(dollars, signal.market_price, signal.side)
    which uses the original YES midpoint and can produce more contracts than
    the orderbook can absorb.  With the fix it uses ``metadata["fill_price"]``
    (the VWAP from the weather strategy's depth check) and also caps by
    ``signal.suggested_contracts``.
    """

    @pytest.mark.asyncio
    async def test_yes_vwap_price_used_for_contract_count(self, tmp_path):
        """When fill_price metadata is present for a YES signal, contracts are
        computed at the VWAP price, not the raw midpoint."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # midpoint = 0.40, VWAP fill_price = 0.55 (slippage from walking book)
        # At midpoint: floor(100 / 0.40) = 250 contracts
        # At VWAP:     floor(100 / 0.55) = 181 contracts
        signal = _make_signal(
            side=Side.YES,
            market_price=0.40,
            model_probability=0.70,
            suggested_dollars=100.0,
            suggested_contracts=181,  # what the strategy computed at VWAP
            metadata={"fill_price": 0.55},
        )
        record = await engine.execute(signal)

        # Contract count must not exceed strategy's VWAP-based figure
        assert record.requested_count <= 181

    @pytest.mark.asyncio
    async def test_no_vwap_price_used_for_contract_count(self, tmp_path):
        """For a NO signal, fill_price is cost-per-NO-contract.  The engine
        must convert back to YES-space before calling dollars_to_contracts."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # YES midpoint = 0.70, so NO cost-per-contract at midpoint = 0.30
        # VWAP for NO = 0.38 (slippage)
        # At NO midpoint (0.30): floor(100 / 0.30) = 333 contracts
        # At NO VWAP   (0.38): floor(100 / 0.38) = 263 contracts
        signal = _make_signal(
            side=Side.NO,
            market_price=0.70,
            model_probability=0.25,
            suggested_dollars=100.0,
            suggested_contracts=263,
            metadata={"fill_price": 0.38},
        )
        record = await engine.execute(signal)

        assert record.requested_count <= 263

    @pytest.mark.asyncio
    async def test_no_fill_price_metadata_uses_midpoint(self, tmp_path):
        """When fill_price is absent from metadata the engine falls back to the
        original signal.market_price (preserving pre-fix behaviour)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # $100 at 0.50 midpoint -> floor(100 / 0.50) = 200 contracts
        signal = _make_signal(
            side=Side.YES,
            market_price=0.50,
            model_probability=0.70,
            suggested_dollars=100.0,
            suggested_contracts=200,
            metadata={},  # no fill_price key
        )
        record = await engine.execute(signal)

        # Should compute 200 from midpoint, then cap at suggested_contracts=200
        assert record.requested_count == 200

    @pytest.mark.asyncio
    async def test_fill_price_none_falls_back_to_midpoint(self, tmp_path):
        """fill_price=None (depth check skipped) must use midpoint as fallback."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(
            side=Side.YES,
            market_price=0.50,
            model_probability=0.70,
            suggested_dollars=100.0,
            suggested_contracts=200,
            metadata={"fill_price": None},
        )
        record = await engine.execute(signal)

        assert record.requested_count == 200

    @pytest.mark.asyncio
    async def test_fill_price_zero_falls_back_to_midpoint(self, tmp_path):
        """fill_price=0.0 (stale/empty book) must use midpoint as fallback."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal(
            side=Side.YES,
            market_price=0.50,
            model_probability=0.70,
            suggested_dollars=100.0,
            suggested_contracts=200,
            metadata={"fill_price": 0.0},
        )
        record = await engine.execute(signal)

        assert record.requested_count == 200

    @pytest.mark.asyncio
    async def test_suggested_contracts_caps_vwap_count(self, tmp_path):
        """suggested_contracts acts as a depth cap: if dollars-at-VWAP gives
        more contracts than the book can absorb, it is capped down."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # VWAP fill_price = 0.10 -> very cheap, dollars / price = many contracts
        # Strategy's depth check capped it to 50
        signal = _make_signal(
            side=Side.YES,
            market_price=0.10,
            model_probability=0.40,
            suggested_dollars=100.0,
            suggested_contracts=50,  # book depth cap applied by strategy
            metadata={"fill_price": 0.10},
        )
        record = await engine.execute(signal)

        # Without cap: floor(100 / 0.10) = 1000; with cap: 50
        assert record.requested_count == 50

    @pytest.mark.asyncio
    async def test_vwap_count_lower_than_suggested_not_raised(self, tmp_path):
        """If dollars-at-VWAP gives fewer contracts than suggested_contracts,
        the lower count is used (we never inflate the size)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # fill_price = 0.90 -> expensive; $50 / 0.90 = 55 contracts
        # suggested_contracts = 500 (higher)
        signal = _make_signal(
            side=Side.YES,
            market_price=0.85,
            model_probability=0.99,
            suggested_dollars=50.0,
            suggested_contracts=500,
            metadata={"fill_price": 0.90},
        )
        record = await engine.execute(signal)

        # dollars_to_contracts(50, effective=0.90, YES) = floor(50/0.90) = 55
        # min(55, 500) = 55
        assert record.requested_count == 55

    @pytest.mark.asyncio
    async def test_vwap_shadow_log_uses_correct_contract_count(self, tmp_path):
        """Shadow log must record the VWAP-adjusted contract count, not the
        raw midpoint-based count."""
        import json as _json

        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # $100 at midpoint 0.40 would give 250 contracts.
        # $100 at VWAP 0.55 gives 181; capped by suggested_contracts=181.
        signal = _make_signal(
            market_ticker="KXHIGH-TEST",
            side=Side.YES,
            market_price=0.40,
            model_probability=0.70,
            suggested_dollars=100.0,
            suggested_contracts=181,
            metadata={"fill_price": 0.55},
        )
        await engine.execute(signal)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [_json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["contracts"] <= 181


# ── Unique order IDs (#18) ────────────────────────────────────────────────


class TestUniqueOrderIds:
    """Multiple rejections / errors / zero-size orders must each get a unique
    order_id so they do not overwrite each other in self._orders.

    Fix #18: 'rejected', 'error', and 'zero-size' were fixed strings; second
    occurrence would silently clobber the first in the orders dict.
    """

    @pytest.mark.asyncio
    async def test_multiple_rejections_have_unique_ids(self, tmp_path):
        """Two risk-rejected signals must get distinct order_ids."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Trigger kill switch so every signal is rejected
        (tmp_path / "state" / "STOP").touch()

        sig1 = _make_signal(market_id="MKT-A", market_ticker="MKT-A-T")
        sig2 = _make_signal(market_id="MKT-B", market_ticker="MKT-B-T")

        r1 = await engine.execute(sig1)
        r2 = await engine.execute(sig2)

        assert r1.status == OrderStatus.REJECTED
        assert r2.status == OrderStatus.REJECTED
        assert r1.order_id != r2.order_id
        assert r1.order_id.startswith("rejected-")
        assert r2.order_id.startswith("rejected-")

    @pytest.mark.asyncio
    async def test_multiple_rejections_all_stored(self, tmp_path):
        """Three rejections must all appear in order_history (no overwrites)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        (tmp_path / "state" / "STOP").touch()

        for i in range(3):
            sig = _make_signal(market_id=f"MKT-{i}", market_ticker=f"MKT-{i}-T")
            await engine.execute(sig)

        rejected = [r for r in engine.order_history if r.status == OrderStatus.REJECTED]
        assert len(rejected) == 3
        ids = {r.order_id for r in rejected}
        assert len(ids) == 3  # all unique

    @pytest.mark.asyncio
    async def test_multiple_errors_have_unique_ids(self, tmp_path):
        """Two API errors must produce distinct error-N order_ids."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.side_effect = Exception("API down")
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        sig1 = _make_signal(market_id="E1", market_ticker="E1-T")
        sig2 = _make_signal(market_id="E2", market_ticker="E2-T")

        r1 = await engine.execute(sig1)
        r2 = await engine.execute(sig2)

        assert r1.status == OrderStatus.ERROR
        assert r2.status == OrderStatus.ERROR
        assert r1.order_id != r2.order_id
        assert r1.order_id.startswith("error-")
        assert r2.order_id.startswith("error-")

    @pytest.mark.asyncio
    async def test_multiple_errors_all_stored(self, tmp_path):
        """Three API errors must all be retained in order_history."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.side_effect = Exception("API down")
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        for i in range(3):
            sig = _make_signal(market_id=f"ERR-{i}", market_ticker=f"ERR-{i}-T")
            await engine.execute(sig)

        errors = [r for r in engine.order_history if r.status == OrderStatus.ERROR]
        assert len(errors) == 3
        ids = {r.order_id for r in errors}
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_zero_size_orders_have_unique_ids(self, tmp_path):
        """Two zero-size signals must get distinct zero-size-N order_ids."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Use a tiny dollar amount so floor(dollars / price) == 0 contracts
        sig1 = _make_signal(
            market_id="Z1", market_ticker="Z1-T", suggested_dollars=0.001,
        )
        sig2 = _make_signal(
            market_id="Z2", market_ticker="Z2-T", suggested_dollars=0.001,
        )

        r1 = await engine.execute(sig1)
        r2 = await engine.execute(sig2)

        # Both should be ZERO_SIZE (zero contracts)
        assert r1.status == OrderStatus.ZERO_SIZE
        assert r2.status == OrderStatus.ZERO_SIZE
        assert r1.order_id != r2.order_id
        assert r1.order_id.startswith("zero-size-")
        assert r2.order_id.startswith("zero-size-")

    @pytest.mark.asyncio
    async def test_zero_size_orders_all_stored(self, tmp_path):
        """Three zero-size signals must all appear in order_history."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        for i in range(3):
            sig = _make_signal(
                market_id=f"Z{i}", market_ticker=f"Z{i}-T", suggested_dollars=0.001,
            )
            await engine.execute(sig)

        skipped = [r for r in engine.order_history if r.status == OrderStatus.ZERO_SIZE]
        assert len(skipped) == 3
        ids = {r.order_id for r in skipped}
        assert len(ids) == 3

    @pytest.mark.asyncio
    async def test_counters_are_monotonically_increasing(self, tmp_path):
        """Rejected counter increments produce 'rejected-1', 'rejected-2', etc."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        (tmp_path / "state" / "STOP").touch()

        records = []
        for i in range(4):
            sig = _make_signal(market_id=f"M{i}", market_ticker=f"M{i}-T")
            records.append(await engine.execute(sig))

        ids = [r.order_id for r in records]
        assert ids == ["rejected-1", "rejected-2", "rejected-3", "rejected-4"]


# ── Signal dedup in execute_batch (#10) ──────────────────────────────────


class TestSignalDedup:
    """Fix #10: execute_batch deduplicates signals per (market_ticker, side).

    Only the first signal per key executes; subsequent duplicates get
    OrderStatus.SKIPPED with a DEDUP rejection reason.  The dedup set resets
    between batch calls (per-tick scope).
    """

    @pytest.mark.asyncio
    async def test_dedup_same_ticker_same_side_skips_second(self, tmp_path):
        """Two signals with the same market_ticker and side — second is SKIPPED."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig1 = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )
        sig2 = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=50.0,
        )

        records = await engine.execute_batch([sig1, sig2])

        assert records[0].status == OrderStatus.SHADOW
        assert records[1].status == OrderStatus.SKIPPED
        assert any("DEDUP" in r for r in records[1].risk_decision.rejection_reasons)
        assert records[1].order_id.startswith("dedup-")

    @pytest.mark.asyncio
    async def test_dedup_same_ticker_different_side_both_execute(self, tmp_path):
        """YES and NO on the same market both execute (different sides)."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig_yes = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )
        sig_no = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.NO,
            market_price=0.60,
            model_probability=0.35,
            suggested_dollars=100.0,
        )

        records = await engine.execute_batch([sig_yes, sig_no])

        assert records[0].status == OrderStatus.SHADOW
        assert records[1].status == OrderStatus.SHADOW

    @pytest.mark.asyncio
    async def test_dedup_different_tickers_both_execute(self, tmp_path):
        """Different market tickers both execute normally."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig1 = _make_signal(
            market_id="MKT-A",
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )
        sig2 = _make_signal(
            market_id="MKT-B",
            market_ticker="KXHIGH-T60",
            side=Side.YES,
            suggested_dollars=100.0,
        )

        records = await engine.execute_batch([sig1, sig2])

        assert records[0].status == OrderStatus.SHADOW
        assert records[1].status == OrderStatus.SHADOW


# ── Settlement records trade in circuit breaker tracker ──────────────────


class TestSettlementRecordsTrade:
    """_settle_live must call record_strategy_trade so circuit breakers work."""

    @pytest.mark.asyncio
    async def test_settle_live_records_trade_in_tracker(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        # Create a filled order record (with fills payload to avoid deferral)
        signal = _make_signal(side=Side.YES)
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="order-1",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.40,
            fill_count=10,
            requested_count=10,
            status=OrderStatus.FILLED,
            fills_payload_received=True,
            fees_from_api=0.17,  # ceil(0.07 * 10 * 0.4 * 0.6 * 100)/100 = 0.17
            is_taker_fills=10,
        )
        engine._orders["order-1"] = record
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        # Mock Kalshi API: market settled with YES result
        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"},
        }

        await engine.settle_expired_positions()

        # Verify the circuit breaker tracker received the trade
        tracker = risk.get_tracker("weather_edge")
        assert len(tracker.trades) == 1
        # P&L = (1.0 - 0.4) * 10 - fee; fee = 0.17 (taker, set in fees_from_api above)
        assert tracker.trades[0]["pnl"] == pytest.approx(5.83)
        assert tracker.trades[0]["won"] is True

    @pytest.mark.asyncio
    async def test_settle_live_no_fill_skips_record(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        # Create an order record with no fill
        signal = _make_signal(side=Side.YES)
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="order-1",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=None,
            fill_count=0,
            requested_count=10,
            status=OrderStatus.FILLED,
        )
        engine._orders["order-1"] = record
        risk.record_fill("MKT-001", "weather_edge", 100.0)

        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"},
        }

        await engine.settle_expired_positions()

        # No filled records → no strategy trade recorded
        tracker = risk.get_tracker("weather_edge")
        assert len(tracker.trades) == 0

    @pytest.mark.asyncio
    async def test_dedup_skipped_not_in_risk(self, tmp_path):
        """Deduplicated (SKIPPED) signal must NOT call record_fill."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig1 = _make_signal(
            market_id="MKT-D",
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )
        sig2 = _make_signal(
            market_id="MKT-D",
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )

        # Execute just the first signal to get baseline exposure
        await engine.execute(sig1)
        position_key = sig1.market_id or sig1.market_ticker
        baseline_exposure = risk.positions.get(position_key, 0.0)
        assert baseline_exposure > 0

        # Now execute both in a batch via a fresh engine to verify dedup
        # doesn't add extra exposure
        settings2 = _make_settings(tmp_path)
        risk2 = RiskManager(settings2)
        engine2 = ExecutionEngine(settings2, kalshi_client=None, risk_manager=risk2)

        await engine2.execute_batch([sig1, sig2])
        exposure_after = risk2.positions.get(position_key, 0.0)

        # Exposure should equal a single signal's fill, not double
        assert exposure_after == pytest.approx(baseline_exposure)

    @pytest.mark.asyncio
    async def test_dedup_highest_edge_wins(self, tmp_path):
        """When sorted by edge descending, the highest-edge signal wins."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig_low_edge = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            model_probability=0.55,
            market_price=0.50,
            suggested_dollars=50.0,
            strategy_name="strategy_a",
        )
        sig_high_edge = _make_signal(
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            model_probability=0.70,
            market_price=0.50,
            suggested_dollars=100.0,
            strategy_name="strategy_b",
        )

        # Sort by edge descending (as the orchestrator does)
        signals = sorted([sig_low_edge, sig_high_edge], key=lambda s: s.edge, reverse=True)

        records = await engine.execute_batch(signals)

        # The high-edge signal should execute, low-edge should be skipped
        executed = [r for r in records if r.status == OrderStatus.SHADOW]
        skipped = [r for r in records if r.status == OrderStatus.SKIPPED]

        assert len(executed) == 1
        assert len(skipped) == 1
        assert executed[0].signal.strategy_name == "strategy_b"
        assert skipped[0].signal.strategy_name == "strategy_a"

    @pytest.mark.asyncio
    async def test_dedup_resets_between_batches(self, tmp_path):
        """Dedup resets per batch — same signal can execute in separate batches."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        sig = _make_signal(
            market_id="MKT-RESET",
            market_ticker="KXHIGH-T55",
            side=Side.YES,
            suggested_dollars=100.0,
        )

        records1 = await engine.execute_batch([sig])
        assert records1[0].status == OrderStatus.SHADOW

        # Second batch with the same signal — dedup has reset, but risk
        # manager may reject if market cap is hit.  Use a fresh market to
        # isolate the dedup-reset assertion.
        sig2 = _make_signal(
            market_id="MKT-RESET2",
            market_ticker="KXHIGH-T60",
            side=Side.YES,
            suggested_dollars=100.0,
        )

        # First call should have been approved; now call the same ticker
        # in a second batch to prove the seen set was recreated.
        sig3 = _make_signal(
            market_id="MKT-RESET3",
            market_ticker="KXHIGH-T60",
            side=Side.YES,
            suggested_dollars=100.0,
        )

        # Batch 2: both sig2 and sig3 have the same ticker, so sig3 is deduped
        records2 = await engine.execute_batch([sig2, sig3])
        assert records2[0].status == OrderStatus.SHADOW  # sig2 executes (dedup reset)
        assert records2[1].status == OrderStatus.SKIPPED  # sig3 deduped within batch 2


# ===========================================================================
# cancel_all_open — kill switch emergency path
# ===========================================================================


class TestCancelAllOpenShadowMode:
    """In shadow mode, cancel_all_open is a no-op."""

    @pytest.mark.asyncio
    async def test_shadow_returns_empty(self, tmp_path):
        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        cancelled, truncated = await engine.cancel_all_open()
        assert cancelled == []
        assert truncated is False

    @pytest.mark.asyncio
    async def test_no_client_returns_empty(self, tmp_path):
        """Even in live mode, no client means nothing to cancel."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        cancelled, truncated = await engine.cancel_all_open()
        assert cancelled == []
        assert truncated is False


class TestCancelAllOpenLiveMode:
    """Live mode fetches resting orders and cancels each."""

    @pytest.mark.asyncio
    async def test_cancels_all_resting_orders(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(return_value=(
            [
                {"order_id": "order-1"},
                {"order_id": "order-2"},
                {"order_id": "order-3"},
            ],
            False,
        ))
        mock_client.cancel_order = AsyncMock()

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)
        cancelled, truncated = await engine.cancel_all_open()

        assert cancelled == ["order-1", "order-2", "order-3"]
        assert truncated is False
        mock_client.iter_all_orders.assert_called_once_with(status="resting")
        assert mock_client.cancel_order.call_count == 3

    @pytest.mark.asyncio
    async def test_no_resting_orders(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(return_value=([], False))

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)
        cancelled, truncated = await engine.cancel_all_open()

        assert cancelled == []
        assert truncated is False
        mock_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_cancel_failure(self, tmp_path):
        """If one cancel fails, the others should still proceed."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(return_value=(
            [
                {"order_id": "order-1"},
                {"order_id": "order-2"},
                {"order_id": "order-3"},
            ],
            False,
        ))
        # Second cancel raises, others succeed
        mock_client.cancel_order = AsyncMock(
            side_effect=[None, RuntimeError("API error"), None]
        )

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)
        cancelled, truncated = await engine.cancel_all_open()

        # order-1 and order-3 succeed, order-2 fails silently
        assert cancelled == ["order-1", "order-3"]
        assert truncated is False
        assert mock_client.cancel_order.call_count == 3

    @pytest.mark.asyncio
    async def test_iter_all_orders_failure(self, tmp_path):
        """If fetching open orders fails, return empty (don't crash)."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(
            side_effect=RuntimeError("Connection refused")
        )

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)
        cancelled, truncated = await engine.cancel_all_open()

        assert cancelled == []
        assert truncated is False
        mock_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_order_id_key(self, tmp_path):
        """Orders without order_id should still attempt cancel with empty string."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(return_value=(
            [{"status": "resting"}],  # no order_id key
            False,
        ))
        mock_client.cancel_order = AsyncMock()

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)
        cancelled, truncated = await engine.cancel_all_open()

        # Falls back to "" for order_id
        assert cancelled == [""]
        assert truncated is False
        mock_client.cancel_order.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_truncated_resting_orders_logs_critical_after_loop(
        self, tmp_path, caplog
    ):
        """When iter_all_orders returns truncated=True, log CRITICAL after the loop."""
        import logging as _logging

        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        mock_client.iter_all_orders = AsyncMock(return_value=(
            [{"order_id": "o1"}],
            True,
        ))
        mock_client.cancel_order = AsyncMock()

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)

        # Capture INFO so we can also see the "Cancelled order" log lines for
        # the ordering check below.
        with caplog.at_level(_logging.INFO):
            cancelled, truncated = await engine.cancel_all_open()

        # NIT #5: assert list contents, not just call count
        assert cancelled == ["o1"]
        assert truncated is True
        mock_client.cancel_order.assert_awaited_once_with("o1")

        # NIT #7: CRITICAL log fires AFTER the loop with the actual cancelled count.
        critical_records = [r for r in caplog.records if r.levelno == _logging.CRITICAL]
        truncated_records = [
            r for r in critical_records if KILL_SWITCH_TRUNCATED_PREFIX in r.message
        ]
        assert len(truncated_records) == 1
        assert "cancelled 1" in truncated_records[0].message

        # The CRITICAL log should appear after the cancel_order call in
        # caplog.records ordering — guards against future refactors that
        # re-order the log/loop.
        cancel_log_indices = [
            i for i, r in enumerate(caplog.records) if "Cancelled order" in r.message
        ]
        truncated_log_idx = next(
            i
            for i, r in enumerate(caplog.records)
            if KILL_SWITCH_TRUNCATED_PREFIX in r.message
        )
        assert cancel_log_indices, "Expected at least one 'Cancelled order' log"
        assert max(cancel_log_indices) < truncated_log_idx

    @pytest.mark.asyncio
    async def test_iter_all_orders_timeout_logs_critical_and_returns_truncated(
        self, tmp_path, caplog
    ):
        """If iter_all_orders raises TimeoutError, log KILL_SWITCH_TIMEOUT + return truncated."""
        import logging as _logging

        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)

        mock_client = AsyncMock()
        # Simulate the wait_for(...) raising TimeoutError
        mock_client.iter_all_orders = AsyncMock(side_effect=TimeoutError("simulated"))
        mock_client.cancel_order = AsyncMock()

        engine = ExecutionEngine(settings, kalshi_client=mock_client, risk_manager=risk)

        with caplog.at_level(_logging.CRITICAL):
            cancelled, truncated = await engine.cancel_all_open()

        assert cancelled == []
        assert truncated is True
        mock_client.cancel_order.assert_not_called()
        assert any(KILL_SWITCH_TIMEOUT_PREFIX in r.message for r in caplog.records)


# ===========================================================================
# _compute_fill_price — VWAP from fill records (refactoring item #8)
# ===========================================================================


class TestComputeFillPrice:
    """Unit tests for the extracted _compute_fill_price static method."""

    def test_single_fill(self):
        fills = [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "10.00"}]
        result = ExecutionEngine._compute_fill_price(fills)
        # 50 * 10 / 10 / 100 = 0.50
        assert result == pytest.approx(0.50)

    def test_multiple_fills_vwap(self):
        fills = [
            {"side": "yes", "yes_price_dollars": "0.4000", "no_price_dollars": "0.6000", "count_fp": "10.00"},
            {"side": "yes", "yes_price_dollars": "0.6000", "no_price_dollars": "0.4000", "count_fp": "10.00"},
        ]
        result = ExecutionEngine._compute_fill_price(fills)
        # (40*10 + 60*10) / (10+10) / 100 = 1000/20/100 = 0.50
        assert result == pytest.approx(0.50)

    def test_weighted_vwap(self):
        fills = [
            {"side": "yes", "yes_price_dollars": "0.4000", "no_price_dollars": "0.6000", "count_fp": "30.00"},  # weight: 30
            {"side": "yes", "yes_price_dollars": "0.6000", "no_price_dollars": "0.4000", "count_fp": "10.00"},  # weight: 10
        ]
        result = ExecutionEngine._compute_fill_price(fills)
        # (40*30 + 60*10) / (30+10) / 100 = 1800/40/100 = 0.45
        assert result == pytest.approx(0.45)

    def test_empty_fills_returns_none(self):
        result = ExecutionEngine._compute_fill_price([])
        assert result is None

    def test_zero_count_fills_returns_none(self):
        fills = [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "0.00"}]
        result = ExecutionEngine._compute_fill_price(fills)
        assert result is None

    def test_missing_side_field_skips_fill(self):
        # No `side` field: cannot determine which price to read; fill is skipped.
        fills = [{"count_fp": "10.00", "yes_price_dollars": "0.5000"}]
        result = ExecutionEngine._compute_fill_price(fills)
        assert result is None

    def test_missing_count_key_returns_none(self):
        # Missing count_fp parses to 0; total_count = 0 → None.
        fills = [{"side": "yes", "yes_price_dollars": "0.5000"}]
        result = ExecutionEngine._compute_fill_price(fills)
        assert result is None

    def test_no_side_fill_returns_no_price(self):
        fills = [{"side": "no", "no_price_dollars": "0.7400",
                  "yes_price_dollars": "0.2600", "count_fp": "5.00"}]
        result = ExecutionEngine._compute_fill_price(fills)
        # NO side: returns 0.74 (the actual NO contract cost)
        assert result == pytest.approx(0.74)

    def test_single_penny_fill(self):
        fills = [{"side": "yes", "yes_price_dollars": "0.0100", "no_price_dollars": "0.9900", "count_fp": "100.00"}]
        result = ExecutionEngine._compute_fill_price(fills)
        # 1 * 100 / 100 / 100 = 0.01
        assert result == pytest.approx(0.01)

    def test_single_99_cent_fill(self):
        fills = [{"side": "yes", "yes_price_dollars": "0.9900", "no_price_dollars": "0.0100", "count_fp": "5.00"}]
        result = ExecutionEngine._compute_fill_price(fills)
        # 99 * 5 / 5 / 100 = 0.99
        assert result == pytest.approx(0.99)


class TestKalshiProbeFixture:
    """Regression tests pinned to the Kalshi /portfolio/fills response shape.

    The payloads are synthetic but keep the field names and string-typed
    values of the live API. If Kalshi renames a field, update them to match.
    """

    REAL_NO_FILL = {
        "action": "buy",
        "count_fp": "5.00",
        "created_time": "2026-04-28T03:46:54.749284Z",
        "fee_cost": "0.070000",
        "fill_id": "862765d8-acb7-5ada-b2b3-6cc78482dbfd",
        "is_taker": True,
        "market_ticker": "KXLOWTCHI-26APR28-B52.5",
        "no_price_dollars": "0.7400",
        "order_id": "20525539-d1da-5f57-b333-e75b1b8b1dcf",
        "side": "no",
        "subaccount_number": 0,
        "ticker": "KXLOWTCHI-26APR28-B52.5",
        "trade_id": "862765d8-acb7-5ada-b2b3-6cc78482dbfd",
        "ts": 1777348014,
        "yes_price_dollars": "0.2600",
    }
    REAL_NO_FILL_MAKER = {
        "action": "buy",
        "count_fp": "2.00",
        "created_time": "2026-04-28T03:46:54.200085Z",
        "fee_cost": "0.000000",
        "fill_id": "aa439df5-da61-58ee-9502-fd89c14c0ce6",
        "is_taker": False,
        "market_ticker": "KXLOWTAUS-26APR28-B74.5",
        "no_price_dollars": "0.5800",
        "order_id": "82a8e6ae-f9bf-5e64-b7a4-63b6faddec0c",
        "side": "no",
        "subaccount_number": 0,
        "ticker": "KXLOWTAUS-26APR28-B74.5",
        "trade_id": "aa439df5-da61-58ee-9502-fd89c14c0ce6",
        "ts": 1777348014,
        "yes_price_dollars": "0.4200",
    }
    REAL_ORDER = {
        "action": "buy",
        "client_order_id": "",
        "created_time": "2026-04-28T03:57:20.126765Z",
        "expiration_time": "2026-04-28T03:59:20Z",
        "fill_count_fp": "0.00",
        "initial_count_fp": "82.00",
        "last_update_time": "2026-04-28T03:59:20.00008Z",
        "maker_fees_dollars": "0.000000",
        "maker_fill_cost_dollars": "0.000000",
        "no_price_dollars": "0.9800",
        "order_id": "41336c66-5aaf-5415-8c26-5afbf717555b",
        "remaining_count_fp": "0.00",
        "side": "yes",
        "status": "canceled",
        "subaccount_number": 0,
        "taker_fees_dollars": "0.000000",
        "taker_fill_cost_dollars": "0.000000",
        "ticker": "KXHIGHNY-26APR28-B61.5",
        "type": "limit",
        "user_id": "fb514f3e-2d2a-553e-9e0b-2767316b1e11",
        "yes_price_dollars": "0.0200",
    }

    def test_real_no_fill_price_is_no_side_dollars(self):
        result = ExecutionEngine._compute_fill_price([self.REAL_NO_FILL])
        # NO contract bought at $0.74 — side-denominated, no flip.
        assert result == pytest.approx(0.74)

    def test_real_two_no_fills_vwap(self):
        # 5 contracts @ $0.74 + 2 contracts @ $0.58 → (5*0.74+2*0.58)/7
        result = ExecutionEngine._compute_fill_price(
            [self.REAL_NO_FILL, self.REAL_NO_FILL_MAKER]
        )
        expected = (5 * 0.74 + 2 * 0.58) / 7
        assert result == pytest.approx(expected, abs=1e-6)

    def test_real_order_fill_count_parses(self):
        # `fill_count_fp` is a string-decimal that must parse via _parse_kalshi_fp.
        assert ExecutionEngine._parse_kalshi_fp(
            self.REAL_ORDER["fill_count_fp"]
        ) == pytest.approx(0.0)
        # And a non-zero value:
        assert ExecutionEngine._parse_kalshi_fp("82.00") == pytest.approx(82.0)

    def test_real_fee_cost_parses_as_dollars(self):
        # `fee_cost` is "0.070000" string-dollars, NOT cents.
        assert ExecutionEngine._parse_kalshi_fp(
            self.REAL_NO_FILL["fee_cost"]
        ) == pytest.approx(0.07)


# ===========================================================================
# _reconcile_exposure — delta logic for risk manager (refactoring item #8)
# ===========================================================================


class TestReconcileExposure:
    """Unit tests for the extracted _reconcile_exposure method."""

    def _make_engine_and_record(self, tmp_path, *, last_recorded_dollars: float = 50.0):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        client.place_order.return_value = {
            "order": {"order_id": "ORD-1", "status": "resting"},
        }
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="order-recon",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=100,
            requested_count=100,
            last_recorded_dollars=last_recorded_dollars,
            status=OrderStatus.RESTING,
        )
        engine._orders[record.order_id] = record
        # Seed the risk manager with the tentative exposure
        risk.record_fill(signal.position_key, signal.strategy_name, last_recorded_dollars)
        return engine, risk, record

    def test_positive_delta_adds_exposure(self, tmp_path):
        """Actual fill cost > tentative → record_fill with the gap."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key
        assert risk.positions[position_key] == pytest.approx(50.0)

        # Actual cost: $60 (slippage)
        engine._reconcile_exposure(record, total_dollars=60.0)

        assert risk.positions[position_key] == pytest.approx(60.0)
        assert record.last_recorded_dollars == pytest.approx(60.0)

    def test_negative_delta_releases_exposure(self, tmp_path):
        """Actual fill cost < tentative → release_exposure with the overage."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key

        # Actual cost: $40 (better fill)
        engine._reconcile_exposure(record, total_dollars=40.0)

        assert risk.positions[position_key] == pytest.approx(40.0)
        assert record.last_recorded_dollars == pytest.approx(40.0)

    def test_zero_delta_no_change(self, tmp_path):
        """Exact match → no exposure change, no risk manager calls."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key

        engine._reconcile_exposure(record, total_dollars=50.0)

        assert risk.positions[position_key] == pytest.approx(50.0)
        assert record.last_recorded_dollars == pytest.approx(50.0)

    def test_idempotent_on_repeated_calls(self, tmp_path):
        """Calling twice with same total_dollars must not double-count."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key

        engine._reconcile_exposure(record, total_dollars=45.0)
        engine._reconcile_exposure(record, total_dollars=45.0)

        assert risk.positions[position_key] == pytest.approx(45.0)

    def test_increasing_fills_over_time(self, tmp_path):
        """Simulate partial fill growing: 30 → 45 → 60 dollars."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key

        # First fill: actual cost $30 (partial, less than tentative)
        engine._reconcile_exposure(record, total_dollars=30.0)
        assert risk.positions[position_key] == pytest.approx(30.0)

        # Second fill: more contracts → total $45
        engine._reconcile_exposure(record, total_dollars=45.0)
        assert risk.positions[position_key] == pytest.approx(45.0)

        # Third fill: full fill → total $60
        engine._reconcile_exposure(record, total_dollars=60.0)
        assert risk.positions[position_key] == pytest.approx(60.0)


# ===========================================================================
# _handle_cancellation — release tentative for cancelled orders (item #8)
# ===========================================================================


class TestHandleCancellation:
    """Unit tests for the extracted _handle_cancellation method."""

    def _make_engine_and_record(
        self,
        tmp_path,
        *,
        fill_count: int = 0,
        last_recorded_dollars: float = 50.0,
    ):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="order-cancel",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_count=fill_count,
            requested_count=100,
            last_recorded_dollars=last_recorded_dollars,
            status=OrderStatus.CANCELLED,
        )
        engine._orders[record.order_id] = record
        if last_recorded_dollars > 0:
            risk.record_fill(
                signal.position_key, signal.strategy_name, last_recorded_dollars,
            )
        return engine, risk, record

    @pytest.mark.asyncio
    async def test_zero_fill_releases_all_exposure(self, tmp_path):
        """Cancelled with zero fills → release entire tentative reservation."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, fill_count=0, last_recorded_dollars=50.0,
        )
        position_key = record.signal.position_key
        assert risk.positions[position_key] == pytest.approx(50.0)

        await engine._handle_cancellation(record, "order-cancel")

        assert risk.positions.get(position_key, 0.0) == pytest.approx(0.0)
        assert record.last_recorded_dollars == 0.0

    @pytest.mark.asyncio
    async def test_partial_fill_does_not_release(self, tmp_path):
        """Cancelled after partial fill → _handle_cancellation is a no-op
        (reconciliation was already done by _reconcile_exposure)."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, fill_count=20, last_recorded_dollars=10.0,
        )
        position_key = record.signal.position_key
        exposure_before = risk.positions[position_key]

        await engine._handle_cancellation(record, "order-cancel")

        # No change — partial fill case is handled by _reconcile_exposure
        assert risk.positions[position_key] == pytest.approx(exposure_before)
        assert record.last_recorded_dollars == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_zero_recorded_dollars_is_noop(self, tmp_path):
        """If last_recorded_dollars is 0, nothing to release."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, fill_count=0, last_recorded_dollars=0.0,
        )

        # Should not raise or change anything
        await engine._handle_cancellation(record, "order-cancel")
        assert record.last_recorded_dollars == 0.0

    @pytest.mark.asyncio
    async def test_cancellation_idempotent(self, tmp_path):
        """Calling _handle_cancellation twice on same record is safe."""
        engine, risk, record = self._make_engine_and_record(
            tmp_path, fill_count=0, last_recorded_dollars=50.0,
        )

        await engine._handle_cancellation(record, "order-cancel")
        assert record.last_recorded_dollars == 0.0

        # Second call — last_recorded_dollars is 0, so it's a no-op
        await engine._handle_cancellation(record, "order-cancel")
        assert record.last_recorded_dollars == 0.0


# ── Order TTL ─────────────────────────────────────────────────────────────


class TestOrderTTL:
    """Order TTL: cancel unfilled orders after a configurable timeout."""

    @pytest.mark.asyncio
    async def test_expiration_ts_passed_to_place_order(self, tmp_path):
        """When TTL > 0, expiration_ts is passed to place_order."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=120)
        signal = _make_signal()

        with patch("python.execution.engine.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            mock_time.time.return_value = 1700000000.0
            await engine.execute(signal)

        _, kwargs = client.place_order.call_args
        assert kwargs["expiration_ts"] == 1700000120

    @pytest.mark.asyncio
    async def test_expiration_ts_not_passed_when_ttl_disabled(self, tmp_path):
        """When TTL = 0, expiration_ts is None."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=0)
        signal = _make_signal()

        await engine.execute(signal)

        _, kwargs = client.place_order.call_args
        assert kwargs["expiration_ts"] is None

    @pytest.mark.asyncio
    async def test_stale_order_cancelled_after_ttl(self, tmp_path):
        """A zero-fill order past TTL is cancelled and exposure released."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        signal = _make_signal(suggested_dollars=500.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0.0

        # Backdate the order past TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # check_fills sees the order as still resting
        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }

        await engine.check_all_pending_fills()

        assert record.status == OrderStatus.EXPIRED
        assert risk.positions.get(position_key, 0.0) == 0.0
        client.cancel_order.assert_called_once_with("ORD-1")

    @pytest.mark.asyncio
    async def test_partial_fill_ttl_expiry_releases_unfilled_portion(self, tmp_path):
        """A partially-filled order past TTL releases only the unfilled exposure."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        # 50 contracts × $0.50 = $25.00 tentative
        signal = _make_signal(market_price=0.50, suggested_dollars=25.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] == pytest.approx(25.0)

        # Simulate partial fill of 20/50 contracts via check_fills
        client.get_order.return_value = {
            "order": {"fill_count_fp": "20.00", "status": "resting"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "20.00"}],
        }
        await engine.check_all_pending_fills()

        # After check_fills reconciliation: exposure = 20 × $0.50 = $10.00
        assert risk.positions[position_key] == pytest.approx(10.0)
        assert record.status == OrderStatus.PARTIAL

        # Now backdate past TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # check_fills sees the same partial state; then TTL kicks in
        await engine.check_all_pending_fills()

        assert record.status == OrderStatus.EXPIRED
        client.cancel_order.assert_called()

        # The post-cancel check_fills re-poll reconciles exposure to the actual
        # fill: 20 × $0.50 = $10.00 held for the filled portion; the 30 unfilled
        # contracts' exposure was released by that reconciliation. (The old path
        # blindly released (50-20) × $0.50 = $15 on top of the already-reconciled
        # $10, clamping to $0 and under-counting the real held position — that
        # over-release is fixed by routing the release through check_fills.)
        assert risk.positions[position_key] == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_full_fill_in_cancel_race_window_kept_not_expired(self, tmp_path):
        """TOCTOU: a fill landing between the pre-cancel status read and the
        cancel taking effect must be absorbed — the order is kept FILLED with
        full exposure held, NOT mis-marked EXPIRED with exposure released."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        # 50 contracts × $0.50 = $25.00 tentative
        signal = _make_signal(market_price=0.50, suggested_dollars=25.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        assert risk.positions[position_key] == pytest.approx(25.0)

        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # First check_fills (pre-expire loop) sees the order still resting with
        # zero fills; the SECOND check_fills (post-cancel race-absorbing re-poll)
        # sees it fully executed — the fill landed in the cancel window.
        client.get_order.side_effect = [
            {"order": {"fill_count_fp": "0.00", "status": "resting"}},
            {"order": {"fill_count_fp": "50.00", "status": "executed"}},
        ]
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "50.00",
                }
            ],
        }

        updated, cancelled = await engine.check_all_pending_fills()

        # Kept as a real position — NOT expired.
        assert record.status == OrderStatus.FILLED
        # Full 50 × $0.50 = $25 held; exposure NOT released.
        assert risk.positions[position_key] == pytest.approx(25.0)
        # The race-filled order is not surfaced as a zero-fill cancellation.
        assert all(r.order_id != record.order_id for r in cancelled)
        client.cancel_order.assert_called_once_with(record.order_id)

    @pytest.mark.asyncio
    async def test_ttl_disabled_no_expiry(self, tmp_path):
        """When TTL = 0, orders are never expired."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=0)
        signal = _make_signal()
        record = await engine.execute(signal)

        # Backdate far past any reasonable TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=10000)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }

        await engine.check_all_pending_fills()

        assert record.status == OrderStatus.RESTING
        client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_shadow_mode_unaffected_by_ttl(self, tmp_path):
        """Shadow mode never attempts to expire orders (returns early)."""
        settings = Settings(
            state_dir=str(tmp_path / "state"),
            execution={"mode": "shadow", "order_ttl_seconds": 60},
            _env_file=None,
        )
        (tmp_path / "state").mkdir(exist_ok=True)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        record = await engine.execute(signal)

        # Backdate past TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # check_all_pending_fills returns early in shadow mode
        result = await engine.check_all_pending_fills()
        assert result == ([], [])
        assert record.status == OrderStatus.SHADOW

    @pytest.mark.asyncio
    async def test_cancel_api_failure_does_not_change_status(self, tmp_path):
        """If cancel_order raises, the order status stays unchanged."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        signal = _make_signal(suggested_dollars=500.0)
        record = await engine.execute(signal)

        position_key = signal.market_id or signal.market_ticker
        exposure_before = risk.positions.get(position_key, 0.0)

        # Backdate past TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # check_fills sees it as still resting
        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }
        # cancel_order fails
        client.cancel_order.side_effect = Exception("Connection refused")

        await engine.check_all_pending_fills()

        # Status unchanged — still resting, not expired
        assert record.status == OrderStatus.RESTING
        assert risk.positions.get(position_key, 0.0) == pytest.approx(exposure_before)

    @pytest.mark.asyncio
    async def test_filled_order_not_affected_by_ttl(self, tmp_path):
        """An already-filled order is never subjected to TTL expiry."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)
        # $1 at $0.50 = 2 contracts requested, so 2 filled = fully filled
        signal = _make_signal(suggested_dollars=1.0, market_price=0.50)
        record = await engine.execute(signal)

        # Simulate full fill (2/2 contracts)
        client.get_order.return_value = {
            "order": {"fill_count_fp": "2.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "2.00"}],
        }
        await engine.check_fills(record.order_id)
        assert record.status == OrderStatus.FILLED

        # Backdate past TTL
        record.timestamp = datetime.now(UTC) - timedelta(seconds=90)

        # check_all_pending_fills should skip filled orders entirely
        client.cancel_order.reset_mock()
        await engine.check_all_pending_fills()

        assert record.status == OrderStatus.FILLED
        client.cancel_order.assert_not_called()


# ── Trade log for live fills ────────────────────────────────────────────


class TestLiveTradeLog:
    """Live/paper mode must write confirmed fills to the trade log (trades.jsonl)."""

    @pytest.mark.asyncio
    async def test_live_fill_writes_to_trade_log(self, tmp_path):
        """A confirmed live fill must produce a JSONL entry with fill_price and mode."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # Simulate full fill at 50 cents
        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "5.00"}],
        }
        await engine.check_fills(record.order_id)

        log_path = tmp_path / "state" / "trades.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["order_id"] == record.order_id
        assert entry["market_ticker"] == "MKT-001-T"
        assert entry["mode"] == "live"
        assert entry["fill_price"] == pytest.approx(0.50)
        assert entry["contracts"] == 5
        assert entry["dollars"] == pytest.approx(5 * 0.50)

    @pytest.mark.asyncio
    async def test_live_partial_fill_logs_each_increment(self, tmp_path):
        """Partial fills must log each new batch of contracts separately."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # First check: 3 contracts filled
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "resting"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "3.00"}],
        }
        await engine.check_fills(record.order_id)

        # Second check: now 7 contracts total
        client.get_order.return_value = {
            "order": {"fill_count_fp": "7.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "7.00"}],
        }
        await engine.check_fills(record.order_id)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 2
        assert entries[0]["contracts"] == 3
        assert entries[1]["contracts"] == 4  # delta: 7 - 3

    @pytest.mark.asyncio
    async def test_live_no_double_log_on_repeated_check(self, tmp_path):
        """Calling check_fills twice with unchanged fill count must not duplicate logs."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "5.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "5.00"}],
        }
        await engine.check_fills(record.order_id)
        await engine.check_fills(record.order_id)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1  # only one write, not two

    @pytest.mark.asyncio
    async def test_unfilled_order_no_log(self, tmp_path):
        """A resting order with zero fills must not produce a trade log entry."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "resting"},
        }
        await engine.check_fills("ORD-1")

        log_path = tmp_path / "state" / "trades.jsonl"
        assert not log_path.exists() or log_path.read_text().strip() == ""

    @pytest.mark.asyncio
    async def test_shadow_log_includes_mode_field(self, tmp_path):
        """Shadow mode log entries must include mode='shadow'."""
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        await engine.execute(signal)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["mode"] == "shadow"
        # Shadow entries should NOT have fill_price (they use order_price)
        assert "fill_price" not in entries[0]

    @pytest.mark.asyncio
    async def test_live_log_schema_matches_shadow(self, tmp_path):
        """Live fill log entries must have the same first 13 fields as shadow entries."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        client.get_order.return_value = {
            "order": {"fill_count_fp": "2.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "2.00"}],
        }
        await engine.check_fills(record.order_id)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        entry = entries[0]

        # All shadow-compatible fields must be present
        required_fields = [
            "order_id", "timestamp", "market_id", "market_ticker",
            "strategy", "side", "model_probability", "market_price",
            "edge", "confidence", "dollars", "contracts", "order_price",
        ]
        for field_name in required_fields:
            assert field_name in entry, f"Missing field: {field_name}"

        # Live-specific fields
        assert "fill_price" in entry
        assert "mode" in entry
        assert entry["mode"] == "live"

    @pytest.mark.asyncio
    async def test_no_side_live_fill_price_logged_correctly(self, tmp_path):
        """NO-side fills must log the side-denominated fill_price (NO cost)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(side=Side.NO, market_price=0.60)
        record = await engine.execute(signal)

        # Real Kalshi fills carry side="no" with no_price_dollars = NO cost
        # ($0.60) and yes_price_dollars = YES-equivalent ($0.40). The engine
        # reads no_price_dollars directly — no flip required.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "executed"},
        }
        client.get_fills.return_value = {
            "fills": [{"side": "no", "yes_price_dollars": "0.4000",
                       "no_price_dollars": "0.6000", "count_fp": "3.00"}],
        }
        await engine.check_fills(record.order_id)

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        # fill_price is the NO contract cost ($0.60), side-denominated.
        assert entries[0]["fill_price"] == pytest.approx(0.60)
        assert entries[0]["side"] == "NO"


class TestLiveExecuteImmediateFill:
    """Bug A: when Kalshi returns ``status="executed"`` (and/or non-zero
    ``filled_count``) on the placement response itself — i.e. the order
    fills synchronously as a taker — the placement path must drive the
    same post-fill bookkeeping the polling loop runs.

    Without the fix, the OrderRecord stays at FILLED with no fill_price,
    no exposure reconciliation, and no trade-log row. The polling loop
    then skips it (``FILLED not in _PENDING_STATUSES``), so the leak is
    permanent until restart.
    """

    @pytest.mark.asyncio
    async def test_immediate_full_fill_sets_fill_price_and_count(self, tmp_path):
        """status=executed at placement → fill_price + fill_count are populated."""
        engine, risk, client = _make_live_engine(tmp_path)
        # 100 contracts × $0.50 = $50 tentative
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "100.00"}],
        }

        record = await engine.execute(signal)

        assert record.status == OrderStatus.FILLED
        assert record.fill_count == 100
        assert record.fill_price == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_immediate_full_fill_writes_trade_log_row(self, tmp_path):
        """The placement-time fill path must write a single fill-row to trades.jsonl."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "100.00"}],
        }

        await engine.execute(signal)

        log_path = tmp_path / "state" / "trades.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["order_id"] == "ORD-IMMED"
        assert entry["mode"] == "live"
        assert entry["fill_price"] == pytest.approx(0.50)
        assert entry["contracts"] == 100

    @pytest.mark.asyncio
    async def test_immediate_full_fill_reconciles_exposure_on_slippage(self, tmp_path):
        """Slippage between order_price ($0.50) and actual fill ($0.60) must
        record additional exposure via _reconcile_exposure (record_fill delta)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.6000", "no_price_dollars": "0.4000", "count_fp": "100.00"}],  # filled at 60c, 10c slippage
        }

        record = await engine.execute(signal)
        position_key = signal.market_id or signal.market_ticker

        # Confirmed exposure should be 100 × $0.60 = $60.00
        assert risk.positions[position_key] == pytest.approx(60.0)
        assert record.last_recorded_dollars == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_immediate_full_fill_idempotent_with_polling_tick(self, tmp_path):
        """A subsequent check_all_pending_fills tick must not double-count exposure
        or write a second log row — the FILLED status keeps it out of polling.
        """
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-IMMED",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "100.00"}],
        }

        await engine.execute(signal)
        position_key = signal.market_id or signal.market_ticker
        exposure_after_immediate = risk.positions[position_key]

        # Now tick the polling loop — FILLED should not be re-checked.
        client.get_order.reset_mock()
        await engine.check_all_pending_fills()

        assert risk.positions[position_key] == pytest.approx(exposure_after_immediate)
        client.get_order.assert_not_called()

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_immediate_partial_fill_writes_partial_row_and_keeps_pending(
        self, tmp_path
    ):
        """A placement that returns filled_count < requested_count and status="executed"
        is rare but possible (taker exhausts available liquidity). The engine should:
          - log the partial fill row
          - leave the order in PARTIAL status (in _PENDING_STATUSES) so polling continues
        """
        engine, risk, client = _make_live_engine(tmp_path)
        # Want 100, only get 30 immediately
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-PART",
                "status": "executed",
                "fill_count_fp": "30.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-PART",
                "status": "executed",
                "fill_count_fp": "30.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "30.00"}],
        }

        record = await engine.execute(signal)

        # check_fills sets PARTIAL when 0 < fill_count < requested_count
        assert record.status == OrderStatus.PARTIAL
        assert record.fill_count == 30
        # PARTIAL is in _PENDING_STATUSES — polling will continue
        assert OrderStatus.PARTIAL in engine._PENDING_STATUSES

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["contracts"] == 30

    @pytest.mark.asyncio
    async def test_resting_order_does_not_call_get_fills(self, tmp_path):
        """A normal resting placement must not trigger the immediate-fill path."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-REST",
                "status": "resting",
                "fill_count_fp": "0.00",
            },
        }

        await engine.execute(signal)

        # Resting → no get_order/get_fills call
        client.get_order.assert_not_called()
        client.get_fills.assert_not_called()

    @pytest.mark.asyncio
    async def test_executed_status_with_zero_filled_count_still_triggers_check(
        self, tmp_path
    ):
        """Defensive: if status="executed" but filled_count=0 (Kalshi protocol weirdness),
        still drive the check — let check_fills disambiguate via get_order."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        # status=executed parses to FILLED via the alias
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-WEIRD",
                "status": "executed",
                "fill_count_fp": "0.00",
            },
        }
        # check_fills re-fetches from Kalshi and gets the canonical truth
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-WEIRD",
                "status": "executed",
                "fill_count_fp": "5.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{"side": "yes", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000", "count_fp": "5.00"}],
        }

        await engine.execute(signal)

        client.get_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_immediate_fill_check_fills_exception_does_not_lose_record(
        self, tmp_path
    ):
        """If check_fills blows up during the immediate-fill path, the record
        is still stored and execute() still returns it (exposure stays tentative)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-BOOM",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        # Simulate an unexpected exception during check_fills
        client.get_order.side_effect = RuntimeError("synthetic catastrophic failure")

        record = await engine.execute(signal)

        # Record exists in engine history
        assert record.order_id == "ORD-BOOM"
        assert engine._orders["ORD-BOOM"] is record
        # Tentative exposure stays in place (didn't get reconciled, but didn't get lost)
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0.0

    @pytest.mark.asyncio
    async def test_immediate_fill_with_no_fill_rows_logs_warning_no_crash(
        self, tmp_path
    ):
        """If get_fills returns no fill rows despite filled_count > 0 (race),
        check_fills' _compute_fill_price returns None and we don't crash;
        record exists, no log row, exposure stays tentative."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-RACE",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-RACE",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        # Empty fills despite filled_count=100 (transient API inconsistency)
        client.get_fills.return_value = {"fills": []}

        record = await engine.execute(signal)

        # Record exists, fill_count populated from get_order
        assert record.order_id == "ORD-RACE"
        assert record.fill_count == 100
        # fill_price stayed None because _compute_fill_price returned None
        assert record.fill_price is None
        # No log row written
        log_path = tmp_path / "state" / "trades.jsonl"
        assert not log_path.exists() or log_path.read_text().strip() == ""


class TestImmediateFill404Recovery:
    """When the immediate-fill ``check_fills`` 404s on
    ``GET /portfolio/orders/<oid>`` due to Kalshi's query-exchange lag (the
    order has already filled and is no longer in the open-orders index),
    the engine falls back to ``GET /portfolio/fills?order_id=<oid>`` to
    recover fill metadata.

    Without this recovery, the OrderRecord stays at ``status=FILLED,
    fill_count=0, fill_price=None`` — and the regular polling loop won't
    re-check it because FILLED is not in ``_PENDING_STATUSES``. The
    consequences (observed in production 2026-05-13 for KXHIGHAUS-26MAY13-B88.5,
    KXHIGHCHI-26MAY14-B64.5, KXLOWTMIA-26MAY14-B76.5):
      1. Silent settlement-row skip in ``trades.jsonl`` (engine.py:1155-1157
         filters records with no fill data — so the ``event=settlement`` row
         never lands).
      2. ``_settle_orders`` credits $0 P&L to bankroll and records the trade
         as a loss in strategy-stats (engine.py:1293-1299, 1215).
      3. ``/positions`` shows the position as ``WARNING: untracked exposure``.

    ``/portfolio/fills`` uses a different Kalshi index that does not suffer
    the same query-exchange lag, so a direct lookup recovers the data.
    """

    @pytest.mark.asyncio
    async def test_404_on_get_order_recovered_via_get_fills(self, tmp_path, caplog):
        """get_order raises the graceful 404 (KalshiAPIError status=404) but
        get_fills returns fills → record.fill_count, fill_price, fees_from_api
        populated. The 404 must be handled by check_fills' inner graceful
        branch (quiet DEBUG return, engine.py:969), NOT the loud outer
        ``except Exception`` (logger.exception at ERROR) — so no ERROR log is
        emitted for the expected read-after-write race."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        with caplog.at_level(logging.ERROR, logger="python.execution.engine"):
            record = await engine.execute(signal)

        assert record.status == OrderStatus.FILLED
        assert record.fill_count == 100
        assert record.fill_price == pytest.approx(0.50)
        assert record.fills_payload_received is True
        assert record.is_taker_fills == 100
        assert record.fees_from_api == pytest.approx(0.35)
        # Quiet-logging contract: the graceful 404 must not surface as an ERROR
        # (a RuntimeError stand-in would fall through to the loud outer handler).
        assert not [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "check fills" in r.getMessage().lower()
        ]

    @pytest.mark.asyncio
    async def test_404_partial_immediate_fill_sets_partial_status(self, tmp_path):
        """Production 2026-06-30: a taker that only PARTIALLY fills (1 of 38)
        comes back ``status="resting"`` (not ``"executed"``), then ``get_order``
        404s. ``check_fills``' partial-status transition (engine.py:986) never
        runs, so the record must get ``status=PARTIAL`` set inside the recovery
        path. Otherwise it keeps its placement status (RESTING), which every
        downstream status consumer — the Telegram batch-alert filter
        (``status in {FILLED, SHADOW, PARTIAL}``), settlement — reads as
        "not filled", and no fill alert is ever sent.
        """
        from python.data.kalshi_client import KalshiAPIError

        engine, risk, client = _make_live_engine(tmp_path)
        # Want 100, only 30 fill immediately; the rest rests.
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404-PART",
                "status": "resting",
                "fill_count_fp": "30.00",
            },
        }
        # Faithful to the production race: get_order raises the graceful 404
        # (KalshiAPIError status=404), which check_fills swallows quietly,
        # leaving fill_count==0 → recovery fires.
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "30.00",
                "is_taker": True,
                "fee_cost": "0.10",
            }],
        }

        record = await engine.execute(signal)

        assert record.fill_count == 30
        assert record.requested_count > record.fill_count  # genuinely partial
        # The bug: stays RESTING because the get_order 404 skipped the
        # check_fills partial-status block; the fix sets PARTIAL in recovery.
        assert record.status == OrderStatus.PARTIAL
        # PARTIAL is a pending status → the poller keeps watching the resting
        # remainder (no orphaned exposure).
        assert OrderStatus.PARTIAL in engine._PENDING_STATUSES

    @pytest.mark.asyncio
    async def test_404_full_immediate_fill_nonexecuted_placement_sets_filled(self, tmp_path):
        """Companion to the partial case: a FULL immediate fill can also come
        back with a non-"executed" placement status (e.g. "pending") and then
        404 on get_order. Recovery must set FILLED — otherwise the record stays
        PENDING and the Telegram filter drops it, the same silent-alert bug.
        (The sibling full-fill test uses status="executed", which is FILLED at
        placement, so it never exercises this recovery branch.)
        """
        from python.data.kalshi_client import KalshiAPIError

        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)  # wants 100
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404-FULL",
                "status": "pending",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        record = await engine.execute(signal)

        assert record.fill_count == 100
        assert record.fill_count >= record.requested_count  # genuinely full
        # Was PENDING before the fix (placement status) → silently dropped.
        assert record.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_404_recovery_writes_fill_log_row(self, tmp_path):
        """Recovery produces exactly one fill-row in trades.jsonl with the
        recovered fill data — same row a successful check_fills would write."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        await engine.execute(signal)

        log_path = tmp_path / "state" / "trades.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        # placement row + one fill row from recovery
        fill_rows = [e for e in entries if e.get("event") == "fill"]
        assert len(fill_rows) == 1
        assert fill_rows[0]["order_id"] == "ORD-404"
        assert fill_rows[0]["contracts"] == 100
        assert fill_rows[0]["fill_price"] == pytest.approx(0.50)
        assert fill_rows[0]["mode"] == "live"

    @pytest.mark.asyncio
    async def test_404_recovery_reconciles_exposure_on_slippage(self, tmp_path):
        """Recovery must run _reconcile_exposure: actual fill ($0.60) over
        tentative ($0.50) means risk.positions reflects $60, not $50."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.6000",
                "no_price_dollars": "0.4000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.42",
            }],
        }

        record = await engine.execute(signal)
        position_key = signal.market_id or signal.market_ticker

        assert risk.positions[position_key] == pytest.approx(60.0)
        assert record.last_recorded_dollars == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_404_recovery_idempotent_with_next_polling_tick(self, tmp_path):
        """After recovery the record is FILLED → not in _PENDING_STATUSES →
        check_all_pending_fills must skip it (no double-count, no double-log)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        await engine.execute(signal)
        position_key = signal.market_id or signal.market_ticker
        exposure_after_recovery = risk.positions[position_key]

        # Reset call tracking and clear the side_effect so a stray call would
        # succeed (not raise) — this proves polling truly skips, not just
        # silently fails.
        client.get_order.reset_mock(side_effect=True)
        client.get_fills.reset_mock()
        await engine.check_all_pending_fills()

        assert risk.positions[position_key] == pytest.approx(exposure_after_recovery)
        client.get_order.assert_not_called()
        client.get_fills.assert_not_called()

        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        fill_rows = [e for e in entries if e.get("event") == "fill"]
        assert len(fill_rows) == 1

    @pytest.mark.asyncio
    async def test_404_recovery_when_get_fills_also_fails(self, tmp_path):
        """If both check_fills and the recovery's get_fills fail, no crash;
        record persists with tentative exposure (FillsReconciler will backfill
        the log row on its next paginated sweep)."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-404",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.side_effect = RuntimeError("network unreachable")

        record = await engine.execute(signal)

        assert record.order_id == "ORD-404"
        assert engine._orders["ORD-404"] is record
        # Tentative exposure stays in place — better than zero, will be
        # reconciled by the next periodic Kalshi sync.
        position_key = signal.market_id or signal.market_ticker
        assert risk.positions.get(position_key, 0.0) > 0.0
        # fill data still absent
        assert record.fill_count == 0
        assert record.fill_price is None

    @pytest.mark.asyncio
    async def test_404_recovery_skipped_when_check_fills_succeeded(self, tmp_path):
        """When check_fills succeeds normally, the recovery path must not
        fire — no second get_fills call beyond the one inside check_fills."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-OK",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.return_value = {
            "order": {
                "order_id": "ORD-OK",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        await engine.execute(signal)

        # get_fills called exactly once — by check_fills, NOT by recovery
        assert client.get_fills.call_count == 1

    @pytest.mark.asyncio
    async def test_404_recovery_skipped_for_resting_order(self, tmp_path):
        """A resting placement (not ``already_filled``) skips the entire
        immediate-fill block including recovery — no get_order / get_fills call."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-REST",
                "status": "resting",
                "fill_count_fp": "0.00",
            },
        }

        await engine.execute(signal)

        client.get_order.assert_not_called()
        client.get_fills.assert_not_called()

    @pytest.mark.asyncio
    async def test_404_recovery_skips_log_append_when_reconciler_wrote_first(
        self, tmp_path
    ):
        """Race defense: the FillsReconciler can win the trade-log lock during
        recovery's ``get_fills`` await, write a synthetic fill row for the
        same order_id, and release.  When recovery resumes, it must detect
        the existing row and suppress its own append — otherwise two
        ``event=="fill"`` rows for the same order_id end up in
        ``state/trades.jsonl``, violating the dedup invariant
        (``engine.py:215-220``, ``fills_reconciler.py:319-322``) and
        double-counting in downstream P&L aggregators that read the log
        naïvely.

        In-memory state (``fill_count``, ``fill_price``, ``fees_from_api``,
        ``_reconcile_exposure``) still gets applied so the engine view
        agrees with disk + Kalshi.
        """
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-RACE",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {
            "fills": [{
                "side": "yes",
                "yes_price_dollars": "0.5000",
                "no_price_dollars": "0.5000",
                "count_fp": "100.00",
                "is_taker": True,
                "fee_cost": "0.35",
            }],
        }

        # Pre-seed trades.jsonl with a synthetic fill row for ORD-RACE —
        # simulating the reconciler having written first during recovery's
        # get_fills await.
        log_path = tmp_path / "state" / "trades.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "event": "fill",
            "order_id": "ORD-RACE",
            "timestamp": "2026-05-13T14:00:00Z",
            "market_id": signal.market_id,
            "market_ticker": signal.market_ticker,
            "position_key": signal.market_id or signal.market_ticker,
            "strategy": signal.strategy_name,
            "side": signal.side.value,
            "mode": "live",
            "contracts": 100,
            "dollars": 50.0,
            "fill_price": 0.50,
            "metadata": {"reconciled": True, "reconcile_source": "fills_reconciler"},
        }) + "\n")

        record = await engine.execute(signal)

        # In-memory state must be applied (the bug we're fixing exists
        # because the reconciler doesn't touch engine._orders).
        assert record.fill_count == 100
        assert record.fill_price == pytest.approx(0.50)
        assert record.fills_payload_received is True

        # Exactly ONE fill row for ORD-RACE on disk — the pre-seeded one.
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        fill_rows = [
            e for e in entries
            if e.get("event") == "fill" and e.get("order_id") == "ORD-RACE"
        ]
        assert len(fill_rows) == 1, (
            f"Expected exactly 1 fill row for ORD-RACE; got {len(fill_rows)}: {fill_rows}"
        )

    @pytest.mark.asyncio
    async def test_404_recovery_empty_fills_response_noop(self, tmp_path):
        """If get_fills returns {"fills": []} (race window — order shows
        executed but fills not yet visible), recovery is a no-op: no log
        row, no exposure change, no crash. FillsReconciler will catch it."""
        engine, risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        client.place_order.return_value = {
            "order": {
                "order_id": "ORD-EMPTY",
                "status": "executed",
                "fill_count_fp": "100.00",
            },
        }
        client.get_order.side_effect = KalshiAPIError(404, {"error": {"code": "not_found"}})
        client.get_fills.return_value = {"fills": []}

        record = await engine.execute(signal)
        position_key = signal.market_id or signal.market_ticker

        assert record.fill_count == 0
        assert record.fill_price is None
        # Tentative exposure preserved
        assert risk.positions.get(position_key, 0.0) > 0.0
        # No fill row in log (placement row may or may not exist depending on emit gate)
        log_path = tmp_path / "state" / "trades.jsonl"
        if log_path.exists():
            entries = [json.loads(line) for line in log_path.read_text().splitlines()]
            fill_rows = [e for e in entries if e.get("event") == "fill"]
            assert fill_rows == []


class TestTradeLogFsync:
    """A4: strict trade-log writes must fsync; fire-and-forget writes must not.

    The strict path is the source of truth for shadow-position replay on
    restart. If a settlement-sweep write is buffered in the OS cache when
    the box loses power, it can silently diverge from the atomically-
    written risk_state.json. The fire-and-forget path serves the hot
    fill loop and tolerates loss of the most-recent record.
    """

    @pytest.mark.asyncio
    async def test_strict_path_fsyncs(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        with patch("python.execution.engine.os.fsync") as fsync_mock:
            await engine._append_trade_log_strict(
                [{"event": "test", "value": 1}]
            )

        assert fsync_mock.call_count == 1
        log_path = tmp_path / "state" / "trades.jsonl"
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert entries == [{"event": "test", "value": 1}]

    @pytest.mark.asyncio
    async def test_fire_and_forget_path_does_not_fsync(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        with patch("python.execution.engine.os.fsync") as fsync_mock:
            await engine._append_trade_log({"event": "test", "value": 1})

        assert fsync_mock.call_count == 0
        log_path = tmp_path / "state" / "trades.jsonl"
        assert log_path.read_text().strip() != ""

    @pytest.mark.asyncio
    async def test_strict_path_empty_entries_skips_write(self, tmp_path):
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        with patch("python.execution.engine.os.fsync") as fsync_mock:
            await engine._append_trade_log_strict([])

        assert fsync_mock.call_count == 0
        log_path = tmp_path / "state" / "trades.jsonl"
        assert not log_path.exists() or log_path.read_text() == ""

    @pytest.mark.asyncio
    async def test_fire_and_forget_path_holds_lock_during_write(self, tmp_path):
        """The shared trade_log_lock must be held during _append_trade_log's
        call to _write_log_lines, so the upcoming Phase-2 reconciler cannot
        interleave writes with the engine's hot-fill path. (Spec §7 BLOCKER #3.)"""
        lock = asyncio.Lock()
        engine, _, _ = _make_live_engine(tmp_path, trade_log_lock=lock)

        assert engine._trade_log_lock is lock

        saw_locked_during_write: list[bool] = []
        original_to_thread = asyncio.to_thread

        async def spy_to_thread(func, *args, **kwargs):
            saw_locked_during_write.append(lock.locked())
            return await original_to_thread(func, *args, **kwargs)

        with patch("python.execution.engine.asyncio.to_thread", spy_to_thread):
            await engine._append_trade_log({
                "event": "fill",
                "order_id": "test",
                "timestamp": "2026-05-10T15:00:00+00:00",
            })

        assert saw_locked_during_write, "asyncio.to_thread should have been called"
        assert all(saw_locked_during_write), (
            "lock must be held during every _write_log_lines invocation"
        )

    @pytest.mark.asyncio
    async def test_strict_path_holds_lock_during_write(self, tmp_path):
        """The strict writer is the source of truth for shadow-position replay
        on restart; its lock acquisition is therefore higher-stakes than the
        fire-and-forget path. Verify the same lock semantics apply."""
        lock = asyncio.Lock()
        engine, _, _ = _make_live_engine(tmp_path, trade_log_lock=lock)

        saw_locked_during_write: list[bool] = []
        original_to_thread = asyncio.to_thread

        async def spy_to_thread(func, *args, **kwargs):
            saw_locked_during_write.append(lock.locked())
            return await original_to_thread(func, *args, **kwargs)

        with patch("python.execution.engine.asyncio.to_thread", spy_to_thread):
            await engine._append_trade_log_strict([
                {"event": "settlement", "order_id": "test", "pnl": 1.23}
            ])

        assert saw_locked_during_write, "asyncio.to_thread should have been called"
        assert all(saw_locked_during_write), (
            "lock must be held during _append_trade_log_strict's _write_log_lines call"
        )


class TestOrchestratorZeroFillUnmark:
    """Bug B integration: the orchestrator's tick must call ``unmark_traded``
    on the originating strategy when the engine surfaces a zero-fill
    cancellation. Partial fills must NOT trigger unmark; strategies that
    don't expose ``unmark_traded`` must be skipped silently.
    """

    def _make_orch(self):
        """Minimal orchestrator with the bare attributes _tick reads."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator()
        orch._running = True

        risk_mgr = MagicMock()
        risk_mgr.bankroll = 10_000.0
        risk_mgr.is_halted = False
        risk_mgr.daily_pnl = 0.0
        risk_mgr.check_circuit_breakers.return_value = []
        risk_mgr.get_tracker.return_value = MagicMock(enabled=True)
        risk_mgr.state_dirty = False
        orch._risk_manager = risk_mgr

        engine = MagicMock()
        engine.settle_expired_positions = AsyncMock(return_value=[])
        engine.execute_batch = AsyncMock(return_value=[])
        orch._engine = engine

        settings = MagicMock()
        settings.is_kill_switch_active.return_value = False
        settings.is_shadow = True
        orch._settings = settings

        return orch, engine

    def _make_record_for_signal(
        self, *, ticker: str, strategy_name: str, fill_count: int, status: OrderStatus
    ) -> OrderRecord:
        signal = _make_signal(market_ticker=ticker, strategy_name=strategy_name)
        from python.risk.manager import RiskDecision

        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=10.0)
        return OrderRecord(
            order_id=f"ORD-{ticker}",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=signal.market_price,
            order_price=signal.market_price,
            fill_count=fill_count,
            requested_count=10,
            status=status,
        )

    @pytest.mark.asyncio
    async def test_zero_fill_cancellation_unmarks_ticker_for_retry(self):
        """A TTL-expired record returned by the engine triggers strategy.unmark_traded."""
        orch, engine = self._make_orch()

        record = self._make_record_for_signal(
            ticker="KXHIGHNY-99APR14-B70.5",
            strategy_name="weather_edge",
            fill_count=0,
            status=OrderStatus.EXPIRED,
        )
        engine.check_all_pending_fills = AsyncMock(return_value=([], [record]))

        unmark_calls: list[str] = []

        class FakeStrategy:
            name = "weather_edge"
            enabled = True
            bankroll = 10_000.0

            def update_bankroll(self, *_args, **_kw) -> None:
                pass

            async def generate_signals(self) -> list:
                return []

            def unmark_traded(self, ticker: str) -> None:
                unmark_calls.append(ticker)

        orch._strategies = [FakeStrategy()]

        await orch._tick()

        assert unmark_calls == ["KXHIGHNY-99APR14-B70.5"]

    @pytest.mark.asyncio
    async def test_partial_fill_cancellation_does_not_unmark(self):
        """A partial-fill record (engine routes to ``updated``, not ``cancelled``)
        must not produce an unmark call. We assert the contract: when the engine
        passes such a record only via ``updated``, the orchestrator does not unmark.
        """
        orch, engine = self._make_orch()

        record = self._make_record_for_signal(
            ticker="KXHIGHNY-99APR14-B70.5",
            strategy_name="weather_edge",
            fill_count=20,
            status=OrderStatus.EXPIRED,  # partial-filled then TTL'd
        )
        engine.check_all_pending_fills = AsyncMock(return_value=([record], []))

        unmark_calls: list[str] = []

        class FakeStrategy:
            name = "weather_edge"
            enabled = True
            bankroll = 10_000.0

            def update_bankroll(self, *_args, **_kw) -> None:
                pass

            async def generate_signals(self) -> list:
                return []

            def unmark_traded(self, ticker: str) -> None:  # pragma: no cover
                unmark_calls.append(ticker)

        orch._strategies = [FakeStrategy()]

        await orch._tick()

        assert unmark_calls == []

    @pytest.mark.asyncio
    async def test_strategy_without_unmark_traded_is_skipped(self):
        """Strategies that don't define ``unmark_traded`` (e.g. an AI-signal
        strategy) are silently skipped via hasattr — no AttributeError."""
        orch, engine = self._make_orch()

        record = self._make_record_for_signal(
            ticker="ELECTION-26-DEM",
            strategy_name="ai_news",
            fill_count=0,
            status=OrderStatus.EXPIRED,
        )
        engine.check_all_pending_fills = AsyncMock(return_value=([], [record]))

        class FakeAiStrategy:
            name = "ai_news"
            enabled = True
            bankroll = 10_000.0

            def update_bankroll(self, *_args, **_kw) -> None:
                pass

            async def generate_signals(self) -> list:
                return []

            # No unmark_traded method

        orch._strategies = [FakeAiStrategy()]

        # Must not raise.
        await orch._tick()


class TestLiveExecuteUsesDeterministicPricing:
    """_live_execute must use compute_limit_cents, not banker's-rounding.

    The banker's-rounding bug made 1c-spread orders land deterministically at
    the bid (no fill) or ask (taker) based on cents parity.  These tests pin
    the new rule: 1c spreads cross to the ask; wider spreads post inside
    biased toward the bid.
    """

    @pytest.mark.asyncio
    async def test_yes_1c_spread_crosses_to_ask(self, tmp_path):
        engine, _risk, client = _make_live_engine(tmp_path)
        captured: dict[str, Any] = {}

        async def fake_place_order(**kwargs):
            captured.update(kwargs)
            return {"order": {"order_id": "ord-pricing-1", "status": "resting"}}

        client.place_order = AsyncMock(side_effect=fake_place_order)

        signal = _make_signal(
            side=Side.YES,
            market_price=0.325,
            metadata={"yes_bid": 0.32, "yes_ask": 0.33, "fill_price": 0.33},
        )
        await engine.execute(signal)

        # 1c spread -> cross to YES ask = 33c.
        assert captured["yes_price"] == 33
        assert captured["no_price"] is None

    @pytest.mark.asyncio
    async def test_no_1c_spread_crosses_to_no_ask(self, tmp_path):
        engine, _risk, client = _make_live_engine(tmp_path)
        captured: dict[str, Any] = {}

        async def fake_place_order(**kwargs):
            captured.update(kwargs)
            return {"order": {"order_id": "ord-pricing-2", "status": "resting"}}

        client.place_order = AsyncMock(side_effect=fake_place_order)

        signal = _make_signal(
            side=Side.NO,
            market_price=0.325,  # YES midpoint; engine flips to NO via 1 - mid.
            metadata={"yes_bid": 0.32, "yes_ask": 0.33, "fill_price": 0.68},
        )
        await engine.execute(signal)

        # NO bid 67c, NO ask 68c -> 1c spread -> cross to NO ask = 68c.
        assert captured["no_price"] == 68
        assert captured["yes_price"] is None

    @pytest.mark.asyncio
    async def test_5c_spread_posts_inside_toward_bid(self, tmp_path):
        engine, _risk, client = _make_live_engine(tmp_path)
        captured: dict[str, Any] = {}

        async def fake_place_order(**kwargs):
            captured.update(kwargs)
            return {"order": {"order_id": "ord-pricing-3", "status": "resting"}}

        client.place_order = AsyncMock(side_effect=fake_place_order)

        signal = _make_signal(
            side=Side.YES,
            market_price=0.325,  # bid 30, ask 35 -> mid 32.5.
            metadata={"yes_bid": 0.30, "yes_ask": 0.35, "fill_price": 0.35},
        )
        await engine.execute(signal)

        # 32.5 truncates toward bid -> 32c.
        assert captured["yes_price"] == 32

    @pytest.mark.asyncio
    async def test_missing_book_falls_back_to_truncation(self, tmp_path):
        """No book metadata -> fall back to deterministic truncation (no banker's)."""
        engine, _risk, client = _make_live_engine(tmp_path)
        captured: dict[str, Any] = {}

        async def fake_place_order(**kwargs):
            captured.update(kwargs)
            return {"order": {"order_id": "ord-pricing-4", "status": "resting"}}

        client.place_order = AsyncMock(side_effect=fake_place_order)

        signal = _make_signal(side=Side.YES, market_price=0.325, metadata={})
        await engine.execute(signal)

        # Fallback: int(0.325 * 100) = 32.
        assert captured["yes_price"] == 32


class TestKalshiStatusAliases:
    """Kalshi returns 'canceled' (American) and 'executed' — must map to our enum."""

    def test_canceled_alias_maps_to_cancelled(self):
        assert _parse_order_status("canceled") == OrderStatus.CANCELLED

    def test_executed_alias_maps_to_filled(self):
        assert _parse_order_status("executed") == OrderStatus.FILLED

    def test_known_status_still_works(self):
        assert _parse_order_status("resting") == OrderStatus.RESTING
        assert _parse_order_status("filled") == OrderStatus.FILLED

    def test_unknown_status_falls_back(self):
        assert (
            _parse_order_status("bogus", fallback=OrderStatus.RESTING)
            == OrderStatus.RESTING
        )


class TestPlacementRow:
    """Phase 1: placement-event row written immediately on successful place_order
    gated on execution.emit_placement_rows. Spec §5.2."""

    @pytest.mark.asyncio
    async def test_disabled_when_flag_off(self, tmp_path):
        """Default config (flag off) → no placement row written even on
        successful live placement."""
        # engine with flag off (default)
        engine, _risk, _client = _make_live_engine(tmp_path)
        signal = _make_signal()
        await engine.execute(signal)

        log_path = engine._trade_log_path
        rows = []
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
        placement_rows = [r for r in rows if r.get("event") == "placement"]
        assert placement_rows == []

    @pytest.mark.asyncio
    async def test_writes_when_flag_on(self, tmp_path):
        """Flag on, live mode → exactly one placement row before any fill."""
        engine, _risk, _client = _make_live_engine(tmp_path, emit_placement_rows=True)
        signal = _make_signal()
        await engine.execute(signal)

        log_path = engine._trade_log_path
        assert log_path.exists(), "trade log must exist after live execute"
        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        placement_rows = [r for r in rows if r.get("event") == "placement"]
        assert len(placement_rows) == 1, f"expected 1 placement row, got {len(placement_rows)}"

        row = placement_rows[0]
        assert row["order_id"] == "ORD-1"
        assert row["market_ticker"] == signal.market_ticker
        assert row["side"] == signal.side
        assert row["mode"] == "live"
        assert row["strategy"] == signal.strategy_name
        # Fill-time fields must be explicitly null on a placement row
        assert row["fill_price"] is None
        assert row["contracts"] is None
        assert row["dollars"] is None

    @pytest.mark.asyncio
    async def test_shadow_mode_never_writes(self, tmp_path):
        """Even with the flag on, shadow mode does not write placement rows
        (the inline _shadow_execute path doesn't go through _live_execute)."""
        settings = _make_settings(tmp_path, mode="shadow", emit_placement_rows=True)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        signal = _make_signal()
        await engine.execute(signal)

        log_path = engine._trade_log_path
        rows = []
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
        placement_rows = [r for r in rows if r.get("event") == "placement"]
        assert placement_rows == []

    @pytest.mark.asyncio
    async def test_emit_placement_row_method_guards_against_non_live_mode(
        self, tmp_path
    ):
        """The in-method `if self._mode != ExecutionMode.LIVE: return` guard is
        defence-in-depth — exercised here directly so it's not coupled to the
        upstream _live_execute call site (which already gates by mode itself)."""
        # Flag on, build a live engine, then patch self._mode to SHADOW
        # before calling _emit_placement_row directly. The guard inside the
        # method must early-return and write nothing.
        engine, _, signal = _make_live_engine(tmp_path, emit_placement_rows=True)
        # Construct a minimal OrderRecord-like object via the existing fixtures.
        # Easiest path: drive a real placement first to populate _orders,
        # capture the record, then re-drive with mode patched.
        engine._mode = ExecutionMode.SHADOW

        # Build a synthetic OrderRecord. We import locally to avoid a
        # top-level import surface.
        from python.execution.engine import OrderRecord
        from python.risk.manager import RiskDecision
        record = OrderRecord(
            order_id="GUARD-1",
            signal=signal,
            risk_decision=RiskDecision(approved=True, signal=signal, adjusted_dollars=10.0),
            mode=ExecutionMode.SHADOW,
            signal_price=0.5,
            order_price=0.5,
            requested_count=1,
            status=OrderStatus.RESTING,
        )
        await engine._emit_placement_row(
            record, timestamp="2026-05-10T15:00:00+00:00"
        )

        log_path = engine._trade_log_path
        if log_path.exists():
            rows = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            placement_rows = [r for r in rows if r.get("event") == "placement"]
            assert placement_rows == [], (
                "the in-method mode guard should suppress the write"
            )


# ── OrderRecord deferral field defaults ──────────────────────────────────────


def test_order_record_settlement_deferral_fields_default_and_mutable():
    """OrderRecord must expose the three settlement-deferral fields with safe defaults and mutable state."""
    from python.risk.manager import RiskDecision

    signal = _make_signal()
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=10.0)
    record = OrderRecord(
        order_id="TEST-001",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.SHADOW,
        signal_price=signal.market_price,
        order_price=signal.market_price,
    )
    assert record.settle_defer_count == 0
    assert record.settle_deferred is False
    assert record.fills_payload_received is False

    # Verify that the fields are mutable (catch any accidental frozen=True flip).
    record.settle_defer_count += 1
    record.settle_deferred = True
    record.fills_payload_received = True
    assert record.settle_defer_count == 1
    assert record.settle_deferred is True
    assert record.fills_payload_received is True


def test_execution_config_deferral_knobs():
    """ExecutionConfig exposes deferral + aggregate-alert knobs with documented defaults."""
    from python.settings import ExecutionConfig

    cfg = ExecutionConfig()
    assert cfg.max_settle_defer_ticks == 20
    assert cfg.deferral_aggregate_min_count == 3
    assert cfg.deferral_aggregate_min_dollars == 25.0
    assert cfg.deferral_aggregate_cooldown_seconds == 1800


# ── C1: _settle_orders deferral decision tree ────────────────────────────


class TestSettleOrdersDeferral:
    """T1–T5: _settle_orders uses fees_from_api (authoritative) or defers/falls back."""

    def _make_engine(self, tmp_path, **execution_kw):
        settings = _make_settings(tmp_path, mode="live", **execution_kw)
        risk = RiskManager(settings)
        client = AsyncMock()
        return ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

    def _add_record(
        self,
        engine,
        *,
        market_id: str = "KXHIGHNY-X",
        side: Side = Side.YES,
        fill_price: float = 0.50,
        fill_count: int = 10,
        fees_from_api: float = 0.10,
        fills_payload_received: bool = True,
        status: OrderStatus = OrderStatus.FILLED,
        is_taker_fills: int = 10,
        is_maker_fills: int = 0,
    ) -> OrderRecord:
        signal = _make_signal(market_id=market_id, side=side)
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id=f"order-{len(engine._orders)}",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=fill_price,
            order_price=fill_price,
            fill_price=fill_price,
            fill_count=fill_count,
            requested_count=fill_count,
            status=status,
            is_taker_fills=is_taker_fills,
            is_maker_fills=is_maker_fills,
            fees_from_api=fees_from_api,
            fills_payload_received=fills_payload_received,
        )
        engine._orders[record.order_id] = record
        return record

    # T1 — authoritative path: fees_from_api used when payload received
    def test_settle_orders_uses_fees_from_api_when_payload_received(self, tmp_path):
        """T1: fills_payload_received=True → use fees_from_api, all_settled=True."""
        engine = self._make_engine(tmp_path)
        record = self._add_record(
            engine,
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.10,
            fills_payload_received=True,
        )

        total_pnl, strategy_pnls, all_settled, per_record_pnl = (
            engine._settle_orders("KXHIGHNY-X", "yes")
        )

        # YES wins at 0.50: (1.0 - 0.50) * 10 = 5.00 gross, minus 0.10 fee = 4.90
        assert all_settled is True
        assert total_pnl == pytest.approx(4.90)
        assert record.status == OrderStatus.SETTLED
        assert record.settle_deferred is False

    # T2 — race-state defer
    def test_settle_orders_defers_when_payload_not_received(self, tmp_path):
        """T2: fills_payload_received=False → defer, increment counter,
        keep status (NOT SETTLED), all_settled=False."""
        engine = self._make_engine(tmp_path)
        record = self._add_record(
            engine,
            market_id="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=5,
            fees_from_api=0.0,
            fills_payload_received=False,
            status=OrderStatus.FILLED,
        )

        total_pnl, _, all_settled, per_record_pnl = (
            engine._settle_orders("KXHIGHNY-X", "yes")
        )

        assert all_settled is False
        assert total_pnl == pytest.approx(0.0)
        assert record.status == OrderStatus.FILLED  # NOT SETTLED
        assert record.settle_deferred is True
        assert record.settle_defer_count == 1

    # T3 — exhaustion fallback (default multiplier, non-index)
    def test_settle_orders_falls_back_after_exhaustion_default(self, tmp_path, caplog):
        """T3: 21st deferral → fallback to taker recompute (non-index), ERROR logged,
        mark SETTLED."""
        import logging

        engine = self._make_engine(tmp_path)
        record = self._add_record(
            engine,
            market_id="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.0,
            fills_payload_received=False,
            status=OrderStatus.FILLED,
        )
        # Pre-set counter to 20 so the next call is the 21st deferral → exhaustion
        record.settle_defer_count = 20

        with caplog.at_level(logging.ERROR):
            total_pnl, _, all_settled, per_record_pnl = (
                engine._settle_orders("KXHIGHNY-X", "yes")
            )

        # Exhaustion fallback DID settle the record
        assert all_settled is True
        assert record.status == OrderStatus.SETTLED
        assert record.settle_deferred is False
        assert any("SETTLE FEE FALLBACK" in r.message for r in caplog.records)

        # Non-index taker fee: ceil(0.07 * 10 * 0.5 * 0.5 * 100) / 100
        #   = ceil(17.5) / 100 = 18 / 100 = 0.18
        # P&L = 10 * (1 - 0.50) - 0.18 = 5.0 - 0.18 = 4.82
        assert total_pnl == pytest.approx(4.82)
        assert record.order_id in engine._fee_fallback_records_this_tick

    # T4 — exhaustion fallback (KXINX index multiplier)
    def test_settle_orders_falls_back_with_index_multiplier(self, tmp_path, caplog):
        """T4: same as T3 but KXINX ticker → is_index_market=True (multiplier 0.035)."""
        import logging

        engine = self._make_engine(tmp_path)
        record = self._add_record(
            engine,
            market_id="KXINXSPY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.0,
            fills_payload_received=False,
            status=OrderStatus.FILLED,
        )
        record.settle_defer_count = 20

        with caplog.at_level(logging.ERROR):
            total_pnl, _, all_settled, per_record_pnl = (
                engine._settle_orders("KXINXSPY-X", "yes")
            )

        assert all_settled is True
        assert record.status == OrderStatus.SETTLED
        assert record.settle_deferred is False
        assert any("SETTLE FEE FALLBACK" in r.message for r in caplog.records)

        # Index taker fee: ceil(0.035 * 10 * 0.5 * 0.5 * 100) / 100
        #   = ceil(8.75) / 100 = 9 / 100 = 0.09
        # P&L = 10 * (1 - 0.50) - 0.09 = 5.0 - 0.09 = 4.91
        assert total_pnl == pytest.approx(4.91)

    # T4b — exhaustion fallback (KXNDX index multiplier)
    def test_settle_orders_falls_back_with_index_multiplier_kxndx(self, tmp_path, caplog):
        """T4b: KXNDX prefix also routes to is_index_market=True."""
        import logging

        engine = self._make_engine(tmp_path)
        record = self._add_record(
            engine,
            market_id="KXNDXSPY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.0,
            fills_payload_received=False,
            status=OrderStatus.FILLED,
        )
        record.settle_defer_count = 20

        with caplog.at_level(logging.ERROR):
            total_pnl, _, all_settled, per_record_pnl = (
                engine._settle_orders("KXNDXSPY-X", "yes")
            )

        assert all_settled is True
        assert record.status == OrderStatus.SETTLED
        assert record.settle_deferred is False

        # Index taker fee same as T4: 0.09 → P&L = 4.91
        assert total_pnl == pytest.approx(4.91)

    # T5 — multi-record position holds when any record is deferred
    def test_settle_orders_holds_position_when_any_record_deferred(self, tmp_path):
        """T5: 1 record consistent + 1 race-state for same position → all_settled=False."""
        engine = self._make_engine(tmp_path)

        # Consistent record: fills payload received, fees authoritative
        consistent = self._add_record(
            engine,
            market_id="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=5,
            fees_from_api=0.05,
            fills_payload_received=True,
            status=OrderStatus.FILLED,
        )
        consistent.order_id = "o5a"
        engine._orders["o5a"] = consistent
        # Remove the auto-keyed entry to avoid duplicate position records
        keys_to_remove = [k for k, v in engine._orders.items() if v is consistent and k != "o5a"]
        for k in keys_to_remove:
            del engine._orders[k]

        # Racing record: fills payload NOT received
        racing = self._add_record(
            engine,
            market_id="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=3,
            fees_from_api=0.0,
            fills_payload_received=False,
            status=OrderStatus.FILLED,
        )
        racing.order_id = "o5b"
        engine._orders["o5b"] = racing
        keys_to_remove = [k for k, v in engine._orders.items() if v is racing and k != "o5b"]
        for k in keys_to_remove:
            del engine._orders[k]

        total_pnl, _, all_settled, per_record_pnl = (
            engine._settle_orders("KXHIGHNY-X", "yes")
        )

        # Position is NOT fully settled because racing record deferred
        assert all_settled is False
        # Consistent record settled this tick
        assert consistent.status == OrderStatus.SETTLED
        # Racing record still in flight
        assert racing.status == OrderStatus.FILLED
        assert racing.settle_deferred is True


# ── D1: prune_completed_orders excludes settle_deferred records ─────────────


class TestPruneCompletedOrdersSkipsDeferred:
    """T-extra: settle_deferred=True excludes record from prune regardless of status."""

    def _make_engine_with_records(self, tmp_path):
        """Return (engine, deferred_record) with _MAX_RETAINED_ORDERS+5 prunable records."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        # One deferred record that must survive pruning.
        signal_d = _make_signal(market_id="DEFERRED-MKT", market_ticker="DEFERRED-T")
        decision_d = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        deferred = OrderRecord(
            order_id="deferred-order",
            signal=signal_d,
            risk_decision=decision_d,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            status=OrderStatus.SETTLED,
        )
        deferred.settle_deferred = True
        engine._orders[deferred.order_id] = deferred

        # Pad with _MAX_RETAINED_ORDERS + 5 prunable records (SETTLED, no deferral).
        n_pad = engine._MAX_RETAINED_ORDERS + 5
        for i in range(n_pad):
            signal_p = _make_signal(
                market_id=f"PAD-MKT-{i}",
                market_ticker=f"PAD-T-{i}",
            )
            decision_p = MagicMock(approved=True, adjusted_dollars=10.0, rejection_reasons=[])
            rec = OrderRecord(
                order_id=f"pad-{i}",
                signal=signal_p,
                risk_decision=decision_p,
                mode=ExecutionMode.LIVE,
                signal_price=0.50,
                order_price=0.50,
                status=OrderStatus.SETTLED,
            )
            rec.settle_deferred = False
            engine._orders[rec.order_id] = rec

        return engine, deferred

    def test_prune_skips_deferred_record(self, tmp_path):
        """settle_deferred=True on a SETTLED record keeps it alive through prune."""
        engine, deferred = self._make_engine_with_records(tmp_path)
        count_before = len(engine._orders)

        pruned = engine.prune_completed_orders()

        # At least some pruning happened.
        assert pruned > 0
        # The deferred record must still be present.
        assert deferred.order_id in engine._orders
        # Total records reduced but deferred survives.
        assert len(engine._orders) == count_before - pruned


# ── D2: _handle_cancellation resets settle_deferred ─────────────────────────


class TestHandleCancellationResetsSettleDeferred:
    """T-extra-2: deferred → cancelled → reset → prunable."""

    def _make_engine_and_deferred_record(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        signal = _make_signal()
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="deferred-cancel",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_count=0,
            requested_count=100,
            last_recorded_dollars=10.0,
            status=OrderStatus.CANCELLED,
        )
        record.settle_deferred = True
        record.settle_defer_count = 5
        # Pre-load exposure so _handle_cancellation can release it.
        risk.record_fill(signal.position_key, signal.strategy_name, 10.0)
        engine._orders[record.order_id] = record
        return engine, risk, record

    @pytest.mark.asyncio
    async def test_handle_cancellation_resets_settle_deferred(self, tmp_path):
        """_handle_cancellation must clear settle_deferred so cancelled records become prunable."""
        engine, risk, record = self._make_engine_and_deferred_record(tmp_path)

        await engine._handle_cancellation(record, "deferred-cancel")

        assert record.settle_deferred is False

    @pytest.mark.asyncio
    async def test_handle_cancellation_deferred_then_prunable(self, tmp_path):
        """After _handle_cancellation clears settle_deferred, CANCELLED record is prunable."""
        engine, risk, record = self._make_engine_and_deferred_record(tmp_path)

        await engine._handle_cancellation(record, "deferred-cancel")
        # Simulate caller setting terminal status (engine does this separately).
        record.status = OrderStatus.CANCELLED

        # Pad to exceed _MAX_RETAINED_ORDERS so prune actually fires.
        n_pad = engine._MAX_RETAINED_ORDERS + 5
        for i in range(n_pad):
            sig = _make_signal(market_id=f"PAD-{i}", market_ticker=f"PAD-T-{i}")
            dec = MagicMock(approved=True, adjusted_dollars=10.0, rejection_reasons=[])
            rec = OrderRecord(
                order_id=f"pad2-{i}",
                signal=sig,
                risk_decision=dec,
                mode=ExecutionMode.LIVE,
                signal_price=0.50,
                order_price=0.50,
                status=OrderStatus.SETTLED,
            )
            rec.settle_deferred = False
            engine._orders[rec.order_id] = rec

        # Make the deferred-cancel record the oldest so it gets pruned first.
        from datetime import timedelta

        record.timestamp = record.timestamp - timedelta(days=1)

        pruned = engine.prune_completed_orders()
        assert pruned > 0
        # The formerly-deferred record is now prunable and should be gone.
        assert record.order_id not in engine._orders


# ── D3: _expire_stale_orders partial-fill branch resets settle_deferred ──────


class TestExpireStaleOrdersPartialFillResetsSettleDeferred:
    """T-extra-3: deferred + partial-fill expired → settle_deferred reset.

    The zero-fill branch calls _handle_cancellation which (after D2) already
    clears settle_deferred.  The partial-fill branch bypasses _handle_cancellation
    and must reset the flag independently.
    """

    @pytest.mark.asyncio
    async def test_expire_stale_orders_partial_fill_resets_settle_deferred(self, tmp_path):
        """Partial-fill expiry (fill_count>0) must clear settle_deferred=True."""
        engine, risk, client = _make_live_engine(tmp_path, order_ttl_seconds=60)

        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        # Execute to create the order record.
        record = await engine.execute(signal)

        # Simulate a partial fill of 3 out of 10 contracts via check_fills.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "resting"},
        }
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "3.00",
                    "is_taker": True,
                    "fee_cost": "0.030000",
                },
            ],
        }
        await engine.check_fills(record.order_id)

        assert record.fill_count == 3
        assert record.status == OrderStatus.PARTIAL

        # Mark as deferred (as if _settle_orders set it on a previous tick).
        record.settle_deferred = True
        record.settle_defer_count = 2

        # Backdate past TTL so _expire_stale_orders picks it up.
        record.timestamp = datetime.now(UTC) - timedelta(seconds=120)

        await engine._expire_stale_orders()

        assert record.status == OrderStatus.EXPIRED
        # Core assertion: partial-fill branch must have cleared the flag.
        assert record.settle_deferred is False


# ── D4: check_fills resets settle_deferred on terminal-status transitions ────


class TestCheckFillsResetsSettleDeferredOnTerminalStatus:
    """T-extra-4: deferred record observed as CANCELLED via check_fills → reset.

    Key gap: _handle_cancellation has an early return for fill_count != 0, so a
    record with partial fills that Kalshi later cancels will NOT have settle_deferred
    cleared by _handle_cancellation.  check_fills must do it explicitly after the
    status parse whenever the new status is in _TERMINAL_NO_FILL_STATUSES.
    """

    @pytest.mark.asyncio
    async def test_check_fills_resets_settle_deferred_on_cancel_zero_fill(self, tmp_path):
        """Zero-fill CANCELLED observed by check_fills → settle_deferred cleared."""
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal()
        record = await engine.execute(signal)

        # Mark as deferred.
        record.settle_deferred = True
        record.settle_defer_count = 5

        # Kalshi returns status=canceled with zero fills.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "0.00", "status": "canceled"},
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        assert result.status == OrderStatus.CANCELLED
        assert result.settle_deferred is False

    @pytest.mark.asyncio
    async def test_check_fills_resets_settle_deferred_on_cancel_with_partial_fills(self, tmp_path):
        """Partial-fill record observed as CANCELLED by check_fills → settle_deferred cleared.

        This is the critical gap: _handle_cancellation has fill_count != 0 early-return
        so it would NOT reset settle_deferred.  The explicit reset in check_fills
        (after _parse_order_status) is the only clearing mechanism for this path.
        """
        engine, _risk, client = _make_live_engine(tmp_path)
        signal = _make_signal(market_price=0.50, suggested_dollars=50.0)
        record = await engine.execute(signal)

        # First tick: partial fill establishes fill_count and fill_price.
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "resting"},
        }
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "3.00",
                    "is_taker": True,
                    "fee_cost": "0.030000",
                },
            ],
        }
        await engine.check_fills(record.order_id)
        assert record.fill_count == 3
        assert record.fill_price is not None

        # Mark as deferred (simulate settlement race).
        record.settle_deferred = True
        record.settle_defer_count = 3

        # Second tick: Kalshi cancels the order (with fills still present).
        client.get_order.return_value = {
            "order": {"fill_count_fp": "3.00", "status": "canceled"},
        }
        client.get_fills.return_value = {
            "fills": [
                {
                    "side": "yes",
                    "yes_price_dollars": "0.5000",
                    "no_price_dollars": "0.5000",
                    "count_fp": "3.00",
                    "is_taker": True,
                    "fee_cost": "0.030000",
                },
            ],
        }

        result = await engine.check_fills(record.order_id)

        assert result is not None
        assert result.status == OrderStatus.CANCELLED
        # Core assertion: fill_count != 0 means _handle_cancellation's early-return
        # fires, so settle_deferred MUST be cleared by check_fills itself.
        assert result.settle_deferred is False


# ── Aggregate-deferral alert (spec §5.5) ─────────────────────────────────


class TestAggregateDeferralAlert:
    """T17/T17b/T17c — aggregate deferral WARNING fires on edge, gates on dollars."""

    def _make_deferred_engine(self, tmp_path, *, dollars_each: float = 10.0):
        """Create a LIVE engine pre-loaded with 3 deferred records ≥3 ticks.

        Each record has a distinct position_key and a corresponding entry in
        risk._positions so _settle_live does not early-return.  client.get_market
        returns 'open' so the position loop completes without settling — allowing
        the post-loop alert block to run.
        """
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

        # Wire a callback collector.
        sent: list[str] = []

        async def collect(m: str) -> None:
            sent.append(m)

        engine._notify_warning = collect

        for i in range(3):
            pos_key = f"P-{i}"
            ticker = f"MKT-{i:03d}-T"
            signal = _make_signal(
                market_id=pos_key,
                market_ticker=ticker,
                side=Side.YES,
                suggested_dollars=dollars_each,
            )
            decision = risk.check_signal(signal)
            record = OrderRecord(
                order_id=f"order-{i}",
                signal=signal,
                risk_decision=decision,
                mode=ExecutionMode.LIVE,
                signal_price=0.50,
                order_price=0.50,
                fill_price=0.50,
                fill_count=10,
                requested_count=10,
                status=OrderStatus.FILLED,
                last_recorded_dollars=dollars_each,
                fills_payload_received=False,  # keeps them deferred
                settle_deferred=True,
                settle_defer_count=3,
            )
            engine._orders[f"order-{i}"] = record
            # Register position so _settle_live doesn't early-return.
            risk.record_fill(pos_key, "weather_edge", dollars_each)

        # Market is still open → position loop continues without settling.
        client.get_market.return_value = {"market": {"status": "open", "result": ""}}

        return engine, sent

    @pytest.mark.asyncio
    async def test_aggregate_alert_fires_once_on_transition(self, tmp_path):
        """T17: 3 records deferred ≥3 ticks, ≥$25 total → exactly one WARNING
        across two consecutive _settle_live ticks (edge-triggered)."""
        engine, sent = self._make_deferred_engine(tmp_path, dollars_each=10.0)
        # sum = $30 ≥ $25 default threshold

        # Tick 1 — rising edge, should fire.
        await engine._settle_live()
        await asyncio.sleep(0)  # let create_task'd coroutine run

        # Tick 2 — state still active, cooldown NOT elapsed → must NOT re-fire.
        await engine._settle_live()
        await asyncio.sleep(0)

        assert len(sent) == 1, f"Expected exactly 1 alert, got {len(sent)}: {sent}"
        assert "3+ positions deferred" in sent[0]
        assert "$30.00" in sent[0]

    @pytest.mark.asyncio
    async def test_aggregate_alert_refires_after_cooldown(self, tmp_path):
        """T17b: state still active after cooldown period → second alert fires."""
        engine, sent = self._make_deferred_engine(tmp_path, dollars_each=10.0)

        # Tick 1 — rising edge fires.
        await engine._settle_live()
        await asyncio.sleep(0)
        assert len(sent) == 1

        # Simulate 1801 seconds elapsed since the last fire.
        engine._deferral_alert_last_fired = time.time() - 1801

        # Tick 2 — cooldown elapsed, state still active → re-fires.
        await engine._settle_live()
        await asyncio.sleep(0)

        assert len(sent) == 2, f"Expected 2 alerts after cooldown, got {len(sent)}: {sent}"

    @pytest.mark.asyncio
    async def test_aggregate_alert_dollar_gate(self, tmp_path):
        """T17c: 3 deferred ≥3 ticks but only $5 each (sum=$15 < $25) → no alert."""
        engine, sent = self._make_deferred_engine(tmp_path, dollars_each=5.0)
        # sum = $15 < $25 default threshold

        await engine._settle_live()
        await asyncio.sleep(0)

        assert sent == [], f"Expected no alert below dollar threshold, got {sent}"


class TestSettleLiveFiresCriticalOnFallback:
    """E2: SETTLE FEE FALLBACK triggers Telegram CRITICAL via _notify_critical."""

    @pytest.mark.asyncio
    async def test_settle_live_fires_critical_on_fee_fallback(self, tmp_path):
        """T-E2: engine with notify_critical fires CRITICAL when a record exhausts
        deferral and falls back to recomputed fee during _settle_live."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()

        # Collect CRITICAL messages.
        critical_msgs: list[str] = []

        async def collect_critical(m: str) -> None:
            critical_msgs.append(m)

        engine = ExecutionEngine(
            settings,
            kalshi_client=client,
            risk_manager=risk,
            notify_critical=collect_critical,
        )

        pos_key = "KXHIGHNY-X"
        ticker = "KXHIGHNY-26MAY15-T25"
        signal = _make_signal(market_id=pos_key, market_ticker=ticker, side=Side.YES)
        decision = risk.check_signal(signal)

        # Build a record that has exhausted its deferral budget (defer_count=20,
        # fills_payload_received=False) — the next _settle_orders call will
        # hit the fallback branch and append to _fee_fallback_records_this_tick.
        record = OrderRecord(
            order_id="order-fallback",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=10,
            requested_count=10,
            status=OrderStatus.FILLED,
            fills_payload_received=False,
            settle_defer_count=20,  # already at max; next call exhausts → fallback
        )
        engine._orders["order-fallback"] = record

        # Register a matching position so _settle_live doesn't early-return.
        risk.record_fill(pos_key, "weather_edge", 5.00)

        # Market is settled → triggers _settle_orders.
        client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}

        await engine._settle_live()
        # Let create_task'd coroutine run.
        await asyncio.sleep(0)

        assert len(critical_msgs) == 1, (
            f"Expected exactly 1 CRITICAL message, got {len(critical_msgs)}: {critical_msgs}"
        )
        assert "SETTLE FEE FALLBACK" in critical_msgs[0]
        assert pos_key in critical_msgs[0]

    @pytest.mark.asyncio
    async def test_settle_live_no_critical_when_authoritative(self, tmp_path):
        """T-E2b: no fallback → no CRITICAL fire (authoritative path takes fees_from_api)."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()

        critical_msgs: list[str] = []

        async def collect_critical(m: str) -> None:
            critical_msgs.append(m)

        engine = ExecutionEngine(
            settings,
            kalshi_client=client,
            risk_manager=risk,
            notify_critical=collect_critical,
        )

        pos_key = "KXHIGHNY-X"
        ticker = "KXHIGHNY-26MAY15-T25"
        signal = _make_signal(market_id=pos_key, market_ticker=ticker, side=Side.YES)
        decision = risk.check_signal(signal)

        # Authoritative record: fills_payload_received=True → no fallback.
        record = OrderRecord(
            order_id="order-auth",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.50,
            order_price=0.50,
            fill_price=0.50,
            fill_count=10,
            requested_count=10,
            status=OrderStatus.FILLED,
            fills_payload_received=True,
            fees_from_api=0.10,
        )
        engine._orders["order-auth"] = record
        risk.record_fill(pos_key, "weather_edge", 5.00)

        client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}

        await engine._settle_live()
        await asyncio.sleep(0)

        assert critical_msgs == [], (
            f"Expected no CRITICAL on authoritative settle, got: {critical_msgs}"
        )


# ── F1: settled_order_ids.json dedup-token helpers ──────────────────────────


class TestSettledIdsHelpers:
    """F1 round-trip tests for _persist_settled_ids / _load_settled_ids."""

    @pytest.mark.asyncio
    async def test_settled_ids_persist_and_load_round_trip(self, tmp_path):
        """F1 round-trip: persist ids, load returns them."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        await engine._persist_settled_ids({"a", "b", "c"})
        loaded = await engine._load_settled_ids()

        assert loaded == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_settled_ids_load_handles_missing_file(self, tmp_path):
        """First-run: file doesn't exist → empty set."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Do NOT pre-create the file.
        loaded = await engine._load_settled_ids()

        assert loaded == set()

    @pytest.mark.asyncio
    async def test_load_settled_ids_recovers_from_corrupt_json(self, tmp_path):
        """F2/T11b: corrupt JSON triggers rebuild from trades.jsonl."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        state_dir = tmp_path / "state"
        # Pre-create settled_order_ids.json with garbage content.
        (state_dir / "settled_order_ids.json").write_text("not json")
        # Pre-create trades.jsonl with a single settlement row.
        (state_dir / "trades.jsonl").write_text(
            json.dumps({"event": "settlement", "order_id": "X"}) + "\n"
        )

        loaded = await engine._load_settled_ids()

        assert loaded == {"X"}
        # The persisted form must now be valid JSON containing ["X"].
        persisted = json.loads((state_dir / "settled_order_ids.json").read_text())
        assert persisted == ["X"]

    @pytest.mark.asyncio
    async def test_recovery_counts_malformed_lines(self, tmp_path, caplog):
        """F3/T11e: rebuild from trades.jsonl with mixed valid+malformed lines logs counts."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        state_dir = tmp_path / "state"
        # Pre-create corrupt settled_order_ids.json to trigger recovery.
        (state_dir / "settled_order_ids.json").write_text("not json at all")
        # Pre-create trades.jsonl with mixed valid, non-settlement, and malformed lines.
        lines = [
            json.dumps({"event": "settlement", "order_id": "A"}),
            "this is not json",
            json.dumps({"event": "fill", "order_id": "F"}),
            "{half-line",
            json.dumps({"event": "settlement", "order_id": "B"}),
        ]
        (state_dir / "trades.jsonl").write_text("\n".join(lines) + "\n")

        import logging

        with caplog.at_level(logging.WARNING):
            loaded = await engine._load_settled_ids()

        assert loaded == {"A", "B"}
        # Assert a warning was logged about the 2 malformed lines.
        messages = [r.message for r in caplog.records]
        assert any(
            "skipped 2 malformed" in m for m in messages
        ), f"Expected 'skipped 2 malformed' in log records; got: {messages}"

    @pytest.mark.asyncio
    async def test_persist_settled_ids_preserves_original_on_write_failure(
        self, tmp_path, monkeypatch
    ):
        """F4/T11d: simulate OSError mid-write; original file content intact."""
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # First, persist a known-good state.
        await engine._persist_settled_ids({"original"})
        original_content = (tmp_path / "state" / "settled_order_ids.json").read_text()

        # Monkey-patch tempfile.NamedTemporaryFile to raise OSError on __enter__.
        def _raising_ntf(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", _raising_ntf)

        # Attempt the failing persist — OSError must propagate.
        with pytest.raises(OSError):
            await engine._persist_settled_ids({"new"})

        # The original file must be completely untouched.
        assert (tmp_path / "state" / "settled_order_ids.json").read_text() == original_content


# ── G1: _settle_live writes settlement rows + persists dedup ids ────────────


class TestSettleLiveWritesSettlementRows:
    """T9/T10: _settle_live writes event=settlement rows to trades.jsonl
    and persists order_ids to settled_order_ids.json (C-LIVE-SETTLE-LOG).
    """

    def _make_engine(self, tmp_path):
        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        client = AsyncMock()
        return ExecutionEngine(settings, kalshi_client=client, risk_manager=risk), risk, client

    def _add_record(
        self,
        engine,
        risk,
        *,
        order_id: str = "o1",
        pos_key: str = "KXHIGHNY-X",
        ticker: str = "KXHIGHNY-26MAY15-T25",
        side: Side = Side.YES,
        fill_price: float = 0.50,
        fill_count: int = 10,
        fees_from_api: float = 0.10,
        fills_payload_received: bool = True,
        dollars: float = 5.00,
    ) -> OrderRecord:
        signal = _make_signal(
            market_id=pos_key,
            market_ticker=ticker,
            side=side,
        )
        decision = MagicMock(approved=True, adjusted_dollars=dollars, rejection_reasons=[])
        record = OrderRecord(
            order_id=order_id,
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=fill_price,
            order_price=fill_price,
            fill_price=fill_price,
            fill_count=fill_count,
            requested_count=fill_count,
            status=OrderStatus.FILLED,
            fills_payload_received=fills_payload_received,
            fees_from_api=fees_from_api,
            is_taker_fills=fill_count,
            last_recorded_dollars=dollars,
        )
        engine._orders[order_id] = record
        risk.record_fill(pos_key, "weather_edge", dollars)
        return record

    @pytest.mark.asyncio
    async def test_settle_live_writes_settlement_row_per_record(self, tmp_path):
        """T9: live settlement writes one event=settlement row per OrderRecord."""
        engine, risk, client = self._make_engine(tmp_path)
        state_dir = tmp_path / "state"

        self._add_record(
            engine,
            risk,
            order_id="o1",
            pos_key="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.10,
            fills_payload_received=True,
            dollars=5.00,
        )

        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"}
        }

        await engine._settle_live()

        # Read trades.jsonl and find settlement rows.
        trades_path = state_dir / "trades.jsonl"
        assert trades_path.exists(), "trades.jsonl must exist after settlement"
        rows = [json.loads(line) for line in trades_path.read_text().splitlines() if line.strip()]
        settlement_rows = [r for r in rows if r.get("event") == "settlement"]

        assert len(settlement_rows) == 1, (
            f"Expected 1 settlement row, got {len(settlement_rows)}: {settlement_rows}"
        )
        row = settlement_rows[0]
        assert row["order_id"] == "o1"
        # YES wins at 0.50: (1.0 - 0.50) * 10 - 0.10 = 4.90
        assert row["pnl"] == pytest.approx(4.90)
        assert row["result"] == "yes"
        # Settlement rows must NOT carry fees_from_api (spec §5.3 m9-dropped).
        assert "fees_from_api" not in row

        # Dedup-token file must be created and contain "o1".
        ids_path = state_dir / "settled_order_ids.json"
        assert ids_path.exists(), "settled_order_ids.json must exist after settlement"
        persisted = json.loads(ids_path.read_text())
        assert "o1" in persisted

    @pytest.mark.asyncio
    async def test_settle_live_skips_close_position_on_log_write_failure(
        self, tmp_path, monkeypatch
    ):
        """T10: strict-write fails → close_position NOT called → state stays consistent."""
        engine, risk, client = self._make_engine(tmp_path)
        state_dir = tmp_path / "state"

        self._add_record(
            engine,
            risk,
            order_id="o1",
            pos_key="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.10,
            fills_payload_received=True,
            dollars=5.00,
        )

        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"}
        }

        # Track close_position calls.
        close_calls: list[str] = []
        original_close = risk.close_position

        def _tracking_close(position_key, **kwargs):
            close_calls.append(position_key)
            return original_close(position_key, **kwargs)

        monkeypatch.setattr(risk, "close_position", _tracking_close)

        # Make _append_trade_log_strict raise OSError.
        async def _raising_write(entries):
            raise OSError("disk full")

        monkeypatch.setattr(engine, "_append_trade_log_strict", _raising_write)

        await engine._settle_live()

        # close_position must NOT have been called (log+state invariant).
        assert close_calls == [], (
            f"close_position should not be called on write failure; got: {close_calls}"
        )
        # settled_order_ids.json must NOT exist (no persist either).
        assert not (state_dir / "settled_order_ids.json").exists(), (
            "settled_order_ids.json must not be created when write fails"
        )

    @pytest.mark.asyncio
    async def test_settle_live_crash_after_write_recovery(self, tmp_path, monkeypatch):
        """T11 (B2'): pre-existing settlement row + pre-existing dedup id →
        close_position fires once, no duplicate row written."""
        engine, risk, client = self._make_engine(tmp_path)
        state_dir = tmp_path / "state"

        # Add a FILLED record so _settle_orders has something to process.
        self._add_record(
            engine,
            risk,
            order_id="o1",
            pos_key="KXHIGHNY-X",
            side=Side.YES,
            fill_price=0.50,
            fill_count=10,
            fees_from_api=0.10,
            fills_payload_received=True,
            dollars=5.00,
        )

        # Simulate: prior process wrote the settlement row + persisted the dedup
        # id successfully, then crashed before (or during) close_position.
        (state_dir / "trades.jsonl").write_text(
            json.dumps({
                "event": "settlement",
                "order_id": "o1",
                "result": "yes",
                "pnl": 4.90,
                "market_ticker": "KXHIGHNY-26MAY15-T25",
                "side": "YES",
                "mode": "live",
                "timestamp": "2026-05-08T12:00:00+00:00",
                "market_id": "KXHIGHNY-X",
                "position_key": "KXHIGHNY-X",
                "strategy": "weather_edge",
            }) + "\n"
        )
        (state_dir / "settled_order_ids.json").write_text(json.dumps(["o1"]))

        # Track close_position calls.
        close_calls: list[str] = []
        original_close = risk.close_position

        def _tracking_close(position_key, **kwargs):
            close_calls.append(position_key)
            return original_close(position_key, **kwargs)

        monkeypatch.setattr(risk, "close_position", _tracking_close)

        client.get_market.return_value = {
            "market": {"status": "settled", "result": "yes"}
        }

        await engine._settle_live()

        # close_position must fire exactly once (crash-after-write recovery).
        assert len(close_calls) == 1, (
            f"Expected close_position to fire once; got {len(close_calls)}: {close_calls}"
        )

        # No duplicate settlement row — only the original pre-populated one.
        rows = [
            json.loads(line)
            for line in (state_dir / "trades.jsonl").read_text().splitlines()
            if line.strip()
        ]
        settlement_rows = [r for r in rows if r.get("event") == "settlement"]
        assert len(settlement_rows) == 1, (
            f"Expected 1 settlement row (no duplicate); got {len(settlement_rows)}: {settlement_rows}"
        )

        # In-memory record is now SETTLED (marked by _settle_orders).
        record = engine._orders["o1"]
        assert record.status == OrderStatus.SETTLED
        assert record.settle_deferred is False


# ── H1/H2: tentative_dollars uses submitted limit, not signal mid ────────────


class TestLiveExecuteTentativeDollars:
    """T7/T8: tentative_dollars must use the cents-rounded limit price submitted
    to Kalshi (price_cents/100), not the raw signal mid (order_price) (C-EXPOSURE-MID).
    """

    @pytest.mark.asyncio
    async def test_live_execute_tentative_dollars_uses_limit_not_mid(self, tmp_path):
        """T7: 1c-spread YES → compute_limit_cents crosses to ask=50c →
        tentative_dollars = 100 * (50/100) = 50.0, not 100 * 0.495 = 49.5."""
        from python.risk.manager import RiskDecision

        engine, risk, client = _make_live_engine(tmp_path)

        # Track record_fill calls so we can inspect the dollars argument.
        record_fills: list[dict[str, Any]] = []
        original_record_fill = risk.record_fill

        def _tracking_record_fill(market_id, strategy_name, dollars, **kwargs):
            record_fills.append({"market_id": market_id, "dollars": dollars})
            return original_record_fill(market_id, strategy_name, dollars, **kwargs)

        risk.record_fill = _tracking_record_fill

        # YES bid=0.49, ask=0.50 → 1c spread → compute_limit_cents crosses to ask=50c.
        # Signal mid = 0.495. Pre-fix: tentative = 100 * 0.495 = 49.5.
        # Post-fix: tentative = 100 * (50/100) = 50.0.
        signal = _make_signal(
            side=Side.YES,
            market_price=0.495,
            metadata={"yes_bid": 0.49, "yes_ask": 0.50},
        )
        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=49.5)

        client.place_order = AsyncMock(return_value={
            "order": {"order_id": "live-t7", "status": "resting", "fill_count_fp": 0},
        })

        await engine._live_execute(
            signal=signal,
            decision=decision,
            order_price=0.495,
            contract_count=100,
        )

        # First record_fill call is the tentative exposure registration.
        assert record_fills, "record_fill must have been called"
        tentative = record_fills[0]["dollars"]
        assert tentative == pytest.approx(50.0), (
            f"Expected tentative_dollars=50.0 (limit=50c); got {tentative:.4f} "
            f"(pre-fix would be 49.5 = 100 * 0.495)"
        )

    @pytest.mark.asyncio
    async def test_live_execute_tentative_dollars_no_book_fallback(self, tmp_path):
        """T8: no book metadata → truncate fallback: int(0.515*100)=51c →
        tentative_dollars = 100 * (51/100) = 51.0, not 100 * 0.515 = 51.5."""
        from python.risk.manager import RiskDecision

        engine, risk, client = _make_live_engine(tmp_path)

        record_fills: list[dict[str, Any]] = []
        original_record_fill = risk.record_fill

        def _tracking_record_fill(market_id, strategy_name, dollars, **kwargs):
            record_fills.append({"market_id": market_id, "dollars": dollars})
            return original_record_fill(market_id, strategy_name, dollars, **kwargs)

        risk.record_fill = _tracking_record_fill

        # No yes_bid/yes_ask in metadata → fallback: price_cents = int(0.515 * 100) = 51.
        # Pre-fix: tentative = 100 * 0.515 = 51.5.
        # Post-fix: tentative = 100 * (51/100) = 51.0.
        signal = _make_signal(
            side=Side.YES,
            market_price=0.515,
            metadata={},  # no book metadata
        )
        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=51.5)

        client.place_order = AsyncMock(return_value={
            "order": {"order_id": "live-t8", "status": "resting", "fill_count_fp": 0},
        })

        await engine._live_execute(
            signal=signal,
            decision=decision,
            order_price=0.515,
            contract_count=100,
        )

        assert record_fills, "record_fill must have been called"
        tentative = record_fills[0]["dollars"]
        assert tentative == pytest.approx(51.0), (
            f"Expected tentative_dollars=51.0 (limit=51c); got {tentative:.4f} "
            f"(pre-fix would be 51.5 = 100 * 0.515)"
        )


# ── J1/T19: Silent-money cluster end-to-end integration ─────────────────────


class TestSilentMoneyClusterIntegration:
    """T19 (m5''/m4'''): End-to-end integration test that exercises all four
    silent-money-cluster fixes composing correctly on a single live trade.

    Scenario:
      - $5.10 NO trade on KXINXSPY-26MAY08-B5500 (index market)
      - 1¢-spread book: yes_bid=0.49, yes_ask=0.50
      - 3 fills (one with is_taker=None), total fees_from_api=$0.07
      - Market resolves YES → NO loses

    The four fixes asserted:
      C-EXPOSURE-MID:  tentative_dollars = 10 * (51/100) = 5.10
      C-FEE-LIVE:      settlement uses record.fees_from_api = 0.07 (not recomputed)
      C-LIVE-SETTLE-LOG: event=settlement row written + order_id in settled_order_ids.json
      C-FETCH-PNL:     compute_pnl(fees_from_api=0.07) returns fee=0.07 exactly
    """

    @pytest.mark.asyncio
    async def test_silent_money_cluster_end_to_end(self, tmp_path):
        """T19: live $5 NO trade on KXINX-... composes all four fixes correctly.

        Uses tmp_path to redirect both state/trades.jsonl and
        state/settled_order_ids.json — real on-disk round-trip, no stubs at the
        trade-log boundary (spec §7 m4''').
        """
        import json as _json

        from scripts.fetch_shadow_settlements import compute_pnl

        engine, risk, client = _make_live_engine(tmp_path)

        # ── Phase 1: _live_execute places order via engine.execute() ──────────
        # Signal: NO-side on a KXINX index market with 1¢-spread book.
        # YES mid = 0.50; NO mid = 0.50 (symmetric book).
        # yes_bid=0.49, yes_ask=0.50 → NO bid=0.50, NO ask=0.51 → 1¢ spread.
        # compute_limit_cents crosses to ask → price_cents=51¢.
        # tentative_dollars = 10 * (51/100) = 5.10  (C-EXPOSURE-MID).
        signal = _make_signal(
            side=Side.NO,
            market_id="KXINXSPY-26MAY08-B5500",
            market_ticker="KXINXSPY-26MAY08-B5500",
            market_price=0.50,       # YES mid; NO order_price = 1 - 0.50 = 0.50
            model_probability=0.35,  # model thinks YES prob is 35%, NO is 65%
            suggested_dollars=5.10,
            suggested_contracts=10,
            metadata={"yes_bid": 0.49, "yes_ask": 0.50},
        )

        client.place_order = AsyncMock(return_value={
            "order": {"order_id": "live-1", "status": "resting", "fill_count_fp": 0},
        })

        record = await engine.execute(signal)
        order_id = record.order_id
        assert order_id == "live-1"

        # C-EXPOSURE-MID: tentative reservation must use the submitted limit (51¢),
        # not the signal mid (50¢).  10 * 0.51 = 5.10.
        assert record.last_recorded_dollars == pytest.approx(5.10), (
            f"C-EXPOSURE-MID: expected tentative_dollars=5.10 (10 × 51¢); "
            f"got {record.last_recorded_dollars:.4f}"
        )

        position_key = signal.position_key
        assert risk.positions.get(position_key, 0.0) == pytest.approx(5.10), (
            "Risk manager must hold exactly 5.10 as tentative exposure"
        )

        # ── Phase 2: check_fills with 3 fills, one is_taker=None ─────────────
        # Spec §7 T19: 3 fills, one with is_taker=None.
        # fill counts: 4 + 3 + 3 = 10 total.
        # fees: $0.04 + $0.03 + $0.00 = $0.07 total.
        # Fill side is "no"; no_price_dollars = 0.51 (the limit we submitted).
        # is_taker=None must NOT block fills_payload_received=True (B1/T18).
        client.get_order = AsyncMock(return_value={
            "order": {"fill_count_fp": "10.00", "status": "executed"},
        })
        client.get_fills = AsyncMock(return_value={
            "fills": [
                {
                    "side": "no",
                    "yes_price_dollars": "0.4900",
                    "no_price_dollars": "0.5100",
                    "count_fp": "4.00",
                    "is_taker": True,
                    "fee_cost": "0.040000",
                },
                {
                    "side": "no",
                    "yes_price_dollars": "0.4900",
                    "no_price_dollars": "0.5100",
                    "count_fp": "3.00",
                    "is_taker": None,           # unclassified — must not block flag
                    "fee_cost": "0.030000",
                },
                {
                    "side": "no",
                    "yes_price_dollars": "0.4900",
                    "no_price_dollars": "0.5100",
                    "count_fp": "3.00",
                    "is_taker": False,          # maker — $0 fee
                    "fee_cost": "0.000000",
                },
            ],
        })

        result = await engine.check_fills(order_id)

        # Loop completed → flag must be True regardless of is_taker=None (C-FEE-LIVE gate).
        assert result.fills_payload_received is True, (
            "C-FEE-LIVE: fills_payload_received must be True after successful get_fills loop"
        )
        assert result.fees_from_api == pytest.approx(0.07), (
            f"C-FEE-LIVE: expected fees_from_api=0.07; got {result.fees_from_api:.4f}"
        )
        assert result.is_taker_fills == 4, (
            f"is_taker=True fill has 4 contracts; got {result.is_taker_fills}"
        )
        assert result.is_maker_fills == 3, (
            f"is_taker=False fill has 3 contracts; got {result.is_maker_fills}"
        )
        # is_taker=None fill (3 contracts) contributes to fee total but not to
        # taker/maker breakdown — so is_taker + is_maker = 7 < 10 total contracts.

        # ── Phase 3: _settle_live settles the position ───────────────────────
        # Market resolves YES → NO loses.
        # _settle_orders must use fees_from_api=0.07 (authoritative, C-FEE-LIVE).
        # _settle_live must write event=settlement row (C-LIVE-SETTLE-LOG).
        # settled_order_ids.json must contain "live-1" (C-LIVE-SETTLE-LOG dedup).
        client.get_market = AsyncMock(return_value={
            "market": {"status": "settled", "result": "yes"},
        })

        await engine._settle_live()

        state_dir = tmp_path / "state"

        # C-LIVE-SETTLE-LOG: settlement row must appear in trades.jsonl.
        trades_path = state_dir / "trades.jsonl"
        assert trades_path.exists(), "trades.jsonl must exist after settlement"
        all_rows = [
            _json.loads(line)
            for line in trades_path.read_text().splitlines()
            if line.strip()
        ]
        settlement_rows = [r for r in all_rows if r.get("event") == "settlement"]
        assert len(settlement_rows) == 1, (
            f"C-LIVE-SETTLE-LOG: expected 1 settlement row; got {len(settlement_rows)}"
        )
        s_row = settlement_rows[0]
        assert s_row["order_id"] == "live-1", "Settlement row must carry the correct order_id"
        assert s_row["result"] == "yes", "Settlement row must carry the market result"

        # C-LIVE-SETTLE-LOG: dedup-token file must contain the order_id.
        ids_path = state_dir / "settled_order_ids.json"
        assert ids_path.exists(), "settled_order_ids.json must exist after settlement"
        persisted_ids = _json.loads(ids_path.read_text())
        assert "live-1" in persisted_ids, (
            "C-LIVE-SETTLE-LOG: 'live-1' must be persisted in settled_order_ids.json"
        )

        # C-FEE-LIVE: settlement P&L must use authoritative fee (0.07), not
        # the conservative taker-recompute.  For a NO loss: pnl = -(0.51 * 10) - 0.07.
        expected_pnl = -(0.51 * 10) - 0.07
        assert s_row["pnl"] == pytest.approx(expected_pnl), (
            f"C-FEE-LIVE: expected pnl={expected_pnl:.4f}; got {s_row['pnl']:.4f}. "
            f"If fee was recomputed (taker, non-index), pnl would differ."
        )

        # ── Phase 4: round-trip through compute_pnl (C-FETCH-PNL) ───────────
        # Find the fill row written by check_fills.  It carries fees_from_api=0.07
        # because that's what _build_log_entry uses.
        fill_rows = [r for r in all_rows if r.get("event") == "fill"]
        assert fill_rows, "A fill row must have been written by check_fills"
        fill_row = fill_rows[-1]  # last fill row = the most-recent check_fills write

        raw_fees = fill_row.get("fees_from_api")
        assert raw_fees == pytest.approx(0.07), (
            f"Fill row must carry fees_from_api=0.07; got {raw_fees!r}"
        )

        is_index = fill_row["market_ticker"].startswith(("KXINX", "KXNDX"))
        assert is_index is True, "KXINXSPY-... must be detected as an index market"

        pnl, fee, won = compute_pnl(
            side=fill_row["side"],
            order_price=fill_row["fill_price"],
            contracts=fill_row["contracts"],
            market_result="yes",
            fees_from_api=raw_fees,
            is_index_market=is_index,
        )

        # C-FETCH-PNL: fee must equal the API-reported 0.07, not any recompute.
        assert fee == pytest.approx(0.07), (
            f"C-FETCH-PNL: compute_pnl must use fees_from_api=0.07; got fee={fee:.4f}"
        )

        # NO bet, market resolved YES → NO loses.
        assert won is False, "NO bet on a YES-resolved market must be a loss"

        # Full P&L round-trip: -(fill_price * contracts) - fee.
        # fill_price = 0.51 (NO-side VWAP), contracts = 10, fee = 0.07.
        assert pnl == pytest.approx(-(fill_row["fill_price"] * fill_row["contracts"]) - fee), (
            f"C-FETCH-PNL: expected pnl=-(fill_price*contracts)-fee; got pnl={pnl:.4f}"
        )


# ── EXTERNALLY_CLOSED status audit (manual-close spec §7.3.1) ─────────────


def test_externally_closed_in_prunable_statuses() -> None:
    """Pure invariant check — no engine instance needed.

    EXTERNALLY_CLOSED is a terminal status (the position was closed via manual
    Kalshi UI). It MUST be in _PRUNABLE_STATUSES so the engine's order-history
    pruner treats it as garbage-collectable just like SETTLED/CANCELLED/etc.
    """
    assert OrderStatus.EXTERNALLY_CLOSED in ExecutionEngine._PRUNABLE_STATUSES


@pytest.mark.asyncio
async def test_settle_live_skips_externally_closed_records(tmp_path) -> None:
    """Live correctness bug guard (manual-close spec §7.3.1).

    Without the filter extension at engine.py:1273-1278, an EXTERNALLY_CLOSED
    record sharing a position_key with a live position would pollute ticker_map
    (last-write-wins) and the settle path could try to settle a position the
    risk manager already closed via the manual-close pipeline.

    Guard: when the only OrderRecord for a position_key is EXTERNALLY_CLOSED,
    _settle_live must not call get_market for that ticker. We assert by
    wiring get_market to an AsyncMock with an AssertionError side_effect, so
    any invocation fails loudly.
    """
    settings = _make_settings(tmp_path, mode="live")
    risk = RiskManager(settings)
    client = AsyncMock()
    engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

    # Simulate a live position whose only OrderRecord is EXTERNALLY_CLOSED.
    # (In production this happens when /apply_manual_close stamps the record
    # but a stale position_key entry briefly remains in risk._positions, or
    # there's a multi-leg position whose last fill record was stamped.)
    signal = _make_signal(side=Side.YES)
    decision = risk.check_signal(signal)
    record = OrderRecord(
        order_id="order-ext-closed",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.40,
        order_price=0.40,
        fill_price=0.40,
        fill_count=10,
        requested_count=10,
        status=OrderStatus.EXTERNALLY_CLOSED,
    )
    engine._orders["order-ext-closed"] = record
    # Force a live position keyed on the same position_key the record carries.
    risk.record_fill(signal.position_key, "weather_edge", 4.0)

    client.get_market = AsyncMock(
        side_effect=AssertionError(
            "get_market called for EXTERNALLY_CLOSED record — settle path "
            "must skip records the manual-close pipeline already closed"
        )
    )

    result = await engine.settle_expired_positions()
    assert result == []
    client.get_market.assert_not_called()


def test_externally_closed_not_in_pending_statuses() -> None:
    """spec §7.3.1 — EXTERNALLY_CLOSED is terminal; check_all_pending_fills
    and _expire_stale_orders MUST skip it."""
    from python.common import OrderStatus
    from python.execution.engine import ExecutionEngine
    assert OrderStatus.EXTERNALLY_CLOSED not in ExecutionEngine._PENDING_STATUSES


def test_externally_closed_not_in_terminal_no_fill_statuses() -> None:
    """spec §7.3.1 — EXTERNALLY_CLOSED had fills (the manual sell), just not
    on the buy-side bot order; semantically distinct from CANCELLED/EXPIRED."""
    from python.common import OrderStatus
    from python.execution.engine import ExecutionEngine
    assert OrderStatus.EXTERNALLY_CLOSED not in ExecutionEngine._TERMINAL_NO_FILL_STATUSES


class TestIsTakerFillsWriterParity:
    """is_taker_fills / is_maker_fills must mean the same thing (SUM OF CONTRACTS)
    whether the row was written by the engine path or a reconciler synthetic-row
    builder. Regression lock for the engine(+=count) vs reconciler(+=1) semantic
    collision documented in trade_log.py's canonical-field-semantics block."""

    @pytest.mark.asyncio
    async def test_engine_and_reconciler_agree_on_taker_contract_sum(self, tmp_path):
        # Same logical (is_taker, contracts) split for both writers:
        # two taker fills (5 + 15), one maker fill (4), and one UNCLASSIFIED
        # is_taker=None fill (7) → taker=20, maker=4; the None fill counts toward
        # NEITHER (both writers must agree on None handling, not just the sum).
        settings = _make_settings(tmp_path)
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        engine_fills = [
            {"side": "yes", "yes_price_dollars": "0.50", "no_price_dollars": "0.50",
             "count_fp": "5.00", "is_taker": True, "fee_cost": "0.020000"},
            {"side": "yes", "yes_price_dollars": "0.50", "no_price_dollars": "0.50",
             "count_fp": "15.00", "is_taker": True, "fee_cost": "0.050000"},
            {"side": "yes", "yes_price_dollars": "0.50", "no_price_dollars": "0.50",
             "count_fp": "4.00", "is_taker": False, "fee_cost": "0.000000"},
            {"side": "yes", "yes_price_dollars": "0.50", "no_price_dollars": "0.50",
             "count_fp": "7.00", "is_taker": None, "fee_cost": "0.030000"},
        ]
        decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
        record = OrderRecord(
            order_id="oid-parity", signal=_make_signal(),
            risk_decision=decision, mode=ExecutionMode.LIVE,
            signal_price=0.50, order_price=0.50,
        )
        await engine._apply_fills_response(record, engine_fills)

        # Reconciler synthetic-row builder over the same logical split.
        reconciler = FillsReconciler(
            state_path=tmp_path / "fills_reconciler.json",
            trade_log_path=tmp_path / "trades.jsonl",
            trade_log_lock=asyncio.Lock(),
            kalshi_client=MagicMock(),
            engine=None,
            notifier=None,
        )
        recon_fills = [
            {"order_id": "oid-parity", "market_ticker": "KXHIGHNY-X", "side": "yes",
             "yes_price_dollars": "0.50", "no_price_dollars": "0.50", "count_fp": "5",
             "is_taker": True, "fee_cost": "0.02", "created_time": "2026-05-10T15:00:00Z"},
            {"order_id": "oid-parity", "market_ticker": "KXHIGHNY-X", "side": "yes",
             "yes_price_dollars": "0.50", "no_price_dollars": "0.50", "count_fp": "15",
             "is_taker": True, "fee_cost": "0.05", "created_time": "2026-05-10T15:00:01Z"},
            {"order_id": "oid-parity", "market_ticker": "KXHIGHNY-X", "side": "yes",
             "yes_price_dollars": "0.50", "no_price_dollars": "0.50", "count_fp": "4",
             "is_taker": False, "fee_cost": "0.00", "created_time": "2026-05-10T15:00:02Z"},
            {"order_id": "oid-parity", "market_ticker": "KXHIGHNY-X", "side": "yes",
             "yes_price_dollars": "0.50", "no_price_dollars": "0.50", "count_fp": "7",
             "is_taker": None, "fee_cost": "0.03", "created_time": "2026-05-10T15:00:03Z"},
        ]
        row = reconciler._build_synthetic_row(
            order_id="oid-parity", kalshi_fills=recon_fills,
            strategy="weather_edge", resolution_method="in_memory",
        )

        assert record.is_taker_fills == 20
        assert record.is_maker_fills == 4  # None fill (7) counts toward neither
        # The two writers must agree — this is the cross-writer contract, including
        # that an unclassified is_taker=None fill is excluded from both counts.
        assert row["is_taker_fills"] == record.is_taker_fills
        assert row["is_maker_fills"] == record.is_maker_fills
