from __future__ import annotations

from datetime import UTC, datetime, timedelta

from python.common import ExecutionMode, OrderStatus
from python.execution.engine import ExecutionEngine, OrderRecord
from python.risk.manager import RiskDecision


def _record(make_signal, *, exec_price: float, placed_ago_s: int) -> OrderRecord:
    sig = make_signal(market_ticker="KXLOWTLAX-26JUL01-T60", exec_price=exec_price)
    rec = OrderRecord(
        order_id="ord-1",
        signal=sig,
        risk_decision=RiskDecision(approved=True, signal=sig, adjusted_dollars=0.0),
        mode=ExecutionMode.LIVE,
        signal_price=exec_price,
        order_price=exec_price,
        requested_count=38,
        status=OrderStatus.PENDING,
    )
    rec.timestamp = datetime.now(UTC) - timedelta(seconds=placed_ago_s)
    return rec


def test_fill_row_tags_immediate_fill(default_settings, risk_manager, make_signal) -> None:
    engine = ExecutionEngine(default_settings, None, risk_manager)
    rec = _record(make_signal, exec_price=0.76, placed_ago_s=0)
    ts = datetime.now(UTC).isoformat()
    entry = engine._build_log_entry(rec, timestamp=ts, contracts=1, dollars=0.76, fill_price=0.76)
    assert entry["seconds_since_placement"] is not None
    assert entry["seconds_since_placement"] < 5.0
    assert entry["fill_price_minus_signal_ask"] == 0.0


def test_fill_row_tags_patient_fill_and_price_delta(
    default_settings, risk_manager, make_signal
) -> None:
    engine = ExecutionEngine(default_settings, None, risk_manager)
    rec = _record(make_signal, exec_price=0.76, placed_ago_s=180)
    ts = datetime.now(UTC).isoformat()
    entry = engine._build_log_entry(rec, timestamp=ts, contracts=3, dollars=2.22, fill_price=0.74)
    assert entry["seconds_since_placement"] >= 175.0
    # Filled 2c cheaper than the frozen signal-time ask.
    assert entry["fill_price_minus_signal_ask"] == -0.02


def test_fill_row_price_delta_none_when_exec_price_missing(
    default_settings, risk_manager, make_signal
) -> None:
    engine = ExecutionEngine(default_settings, None, risk_manager)
    rec = _record(make_signal, exec_price=float("nan"), placed_ago_s=0)
    ts = datetime.now(UTC).isoformat()
    entry = engine._build_log_entry(rec, timestamp=ts, contracts=1, dollars=0.76, fill_price=0.76)
    assert entry["fill_price_minus_signal_ask"] is None
