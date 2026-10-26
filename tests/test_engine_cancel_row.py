"""Tests for the engine's cancel-row write + on_cancel callback path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.execution.engine import ExecutionEngine, OrderRecord
from python.execution.trade_log import (
    TRADE_LOG_EVENT_CANCEL,
    TRADE_LOG_EVENT_KNOWN,
    TradeLogEvent,
    is_known_event,
    is_pnl_relevant_row,
)
from python.risk.manager import RiskManager

# ===========================================================================
# Step A — trade_log schema
# ===========================================================================


def test_trade_log_event_cancel_is_known() -> None:
    assert TradeLogEvent.CANCEL == "cancel"
    assert TradeLogEvent.CANCEL.value == "cancel"
    assert TRADE_LOG_EVENT_CANCEL == "cancel"
    assert TRADE_LOG_EVENT_CANCEL in TRADE_LOG_EVENT_KNOWN
    assert is_known_event({"event": "cancel"}) is True


def test_cancel_row_is_not_pnl_relevant() -> None:
    assert is_pnl_relevant_row({"event": "cancel"}) is False
    # Sanity: a default fill row is still pnl-relevant.
    assert is_pnl_relevant_row({"event": "fill"}) is True


# ===========================================================================
# Step C — engine cancel-row write + on_cancel callback
# ===========================================================================


def _make_settings(tmp_path: Path, *, mode: str = "live") -> Any:
    """Minimal Settings stub — reuses the helper pattern from test_execution_engine."""
    from tests.test_execution_engine import _make_settings as _outer

    return _outer(tmp_path, mode=mode)


def _make_signal_lax_low() -> Any:
    """Fake LAX/low/2026-05-29 signal mirroring the live-incident shape."""
    from python.strategies.base import TradeSignal

    return TradeSignal(
        market_id="m-lax-low",
        market_ticker="KXLOWTLAX-26MAY29-B55.5",
        model_probability=0.35,
        market_price=0.14,
        confidence=1.0,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=10.0,
        suggested_contracts=70,
        effective_edge=0.20,
        metadata={
            "city": "LAX",
            "market_type": "low",
            "forecast_date": "2026-05-29",
            "strike_type": "between",
            "floor_strike": 55,
            "cap_strike": 56,
        },
    )


def _make_engine_with_record(
    tmp_path: Path,
    *,
    fill_count: int = 0,
    last_recorded_dollars: float = 50.0,
    emit_placement_rows: bool = True,
) -> tuple[ExecutionEngine, RiskManager, OrderRecord]:
    settings = _make_settings(tmp_path, mode="live")
    # Force placement-row emission on so the cancel-row gate is satisfied.
    settings.execution.emit_placement_rows = emit_placement_rows
    risk = RiskManager(settings)
    client = AsyncMock()
    engine = ExecutionEngine(settings, kalshi_client=client, risk_manager=risk)

    signal = _make_signal_lax_low()
    decision = MagicMock(approved=True, adjusted_dollars=100.0, rejection_reasons=[])
    record = OrderRecord(
        order_id="order-lax-low-1",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.14,
        order_price=0.13,
        fill_count=fill_count,
        requested_count=100,
        last_recorded_dollars=last_recorded_dollars,
        status=OrderStatus.CANCELLED,
    )
    engine._orders[record.order_id] = record
    if last_recorded_dollars > 0:
        risk.record_fill(
            signal.position_key, signal.strategy_name, last_recorded_dollars
        )
    return engine, risk, record


def _read_trade_log(engine: ExecutionEngine) -> list[dict[str, Any]]:
    path = engine._trade_log_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_handle_cancellation_zero_fills_writes_cancel_row(tmp_path: Path) -> None:
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    await engine._handle_cancellation(record, record.order_id)
    rows = _read_trade_log(engine)
    cancels = [r for r in rows if r.get("event") == "cancel"]
    assert len(cancels) == 1
    c = cancels[0]
    assert c["order_id"] == record.order_id
    assert c["market_ticker"] == "KXLOWTLAX-26MAY29-B55.5"
    assert c["metadata"]["city"] == "LAX"
    assert c["metadata"]["market_type"] == "low"
    assert c["metadata"]["forecast_date"] == "2026-05-29"
    assert c["strategy"] == "weather_edge"
    assert c["mode"] == "live"
    assert record.cancel_row_written is True


@pytest.mark.asyncio
async def test_handle_cancellation_partial_fill_does_not_write_cancel_row(
    tmp_path: Path,
) -> None:
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=3)
    await engine._handle_cancellation(record, record.order_id)
    rows = _read_trade_log(engine)
    assert not any(r.get("event") == "cancel" for r in rows)
    # Flag should remain unset since the row was not written.
    assert record.cancel_row_written is False


@pytest.mark.asyncio
async def test_handle_cancellation_invokes_on_cancel_callback(tmp_path: Path) -> None:
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    called_with: list[str] = []
    engine.set_on_cancel_callback(lambda oid: called_with.append(oid))
    await engine._handle_cancellation(record, record.order_id)
    assert called_with == [record.order_id]


@pytest.mark.asyncio
async def test_handle_cancellation_no_callback_is_safe(tmp_path: Path) -> None:
    """Default-None callback path — used by shadow/backtest."""
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    # No set_on_cancel_callback call — _on_cancel_callback stays None.
    await engine._handle_cancellation(record, record.order_id)  # must not raise


@pytest.mark.asyncio
async def test_handle_cancellation_callback_fires_even_if_row_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSError on row write must NOT block the in-process callback.

    The cap state must still update so same-day retry works; durable state
    is missing but UTC-midnight rollover heals it.
    """
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    called_with: list[str] = []
    engine.set_on_cancel_callback(lambda oid: called_with.append(oid))

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated disk-full")

    monkeypatch.setattr(engine, "_emit_cancel_row", _raise)
    await engine._handle_cancellation(record, record.order_id)
    assert called_with == [record.order_id]


@pytest.mark.asyncio
async def test_handle_cancellation_callback_exception_is_swallowed(
    tmp_path: Path,
) -> None:
    """A throwing callback must not propagate out of _handle_cancellation."""
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)

    def _boom(_oid: str) -> None:
        raise RuntimeError("callback exploded")

    engine.set_on_cancel_callback(_boom)
    # Must not raise.
    await engine._handle_cancellation(record, record.order_id)
    # Row write still happened (callback fires AFTER the row write).
    rows = _read_trade_log(engine)
    assert any(r.get("event") == "cancel" for r in rows)


@pytest.mark.asyncio
async def test_cancel_row_written_flag_blocks_repeat(tmp_path: Path) -> None:
    """cancel_row_written prevents double-write on a second invocation
    when the early-return guard isn't tripped.

    The first call sets last_recorded_dollars=0 (early-return on
    subsequent calls). We manually reset it to exercise the
    cancel_row_written guard specifically.
    """
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    await engine._handle_cancellation(record, record.order_id)
    # Restore last_recorded_dollars so the early-return at line 770 doesn't
    # short-circuit and we actually exercise cancel_row_written.
    record.last_recorded_dollars = 5.0
    await engine._handle_cancellation(record, record.order_id)
    rows = _read_trade_log(engine)
    cancels = [r for r in rows if r.get("event") == "cancel"]
    assert len(cancels) == 1, "cancel_row_written guard should prevent duplicate"


@pytest.mark.asyncio
async def test_cancel_row_schema_matches_placement_schema_base_fields(
    tmp_path: Path,
) -> None:
    """Cancel rows must carry the same _base_log_fields as placement rows."""
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    from datetime import UTC, datetime

    ts = datetime.now(UTC).isoformat()
    await engine._emit_placement_row(record, timestamp=ts)
    await engine._handle_cancellation(record, record.order_id)
    rows = _read_trade_log(engine)
    placement = next(r for r in rows if r["event"] == "placement")
    cancel = next(r for r in rows if r["event"] == "cancel")
    base_fields = {"order_id", "market_id", "market_ticker", "position_key",
                   "strategy", "side", "mode"}
    for f in base_fields:
        assert placement[f] == cancel[f], (
            f"base field {f!r}: placement={placement[f]!r} cancel={cancel[f]!r}"
        )


@pytest.mark.asyncio
async def test_cancel_row_not_written_when_emit_placement_rows_disabled(
    tmp_path: Path,
) -> None:
    """The cancel row gate mirrors the placement-row gate."""
    engine, _risk, record = _make_engine_with_record(
        tmp_path, fill_count=0, emit_placement_rows=False
    )
    await engine._handle_cancellation(record, record.order_id)
    rows = _read_trade_log(engine)
    assert not any(r.get("event") == "cancel" for r in rows)


@pytest.mark.asyncio
async def test_handle_cancellation_zero_exposure_still_fires_callback_and_row(
    tmp_path: Path,
) -> None:
    """Regression: the cap-release path used to be nested inside the
    `last_recorded_dollars <= 0` early-return, so a 0-fill cancel for an
    order whose exposure was already cleared (e.g. by a reconciler race
    or a second _handle_cancellation invocation) silently leaked the cap
    slot for the rest of the UTC day. The fix moves the cap release
    above the early-return; this test pins that ordering.
    """
    engine, _risk, record = _make_engine_with_record(
        tmp_path, fill_count=0, last_recorded_dollars=0.0
    )
    called_with: list[str] = []
    engine.set_on_cancel_callback(lambda oid: called_with.append(oid))
    await engine._handle_cancellation(record, record.order_id)

    # Cap callback fired even though last_recorded_dollars was already 0.
    assert called_with == [record.order_id]
    # Durable cancel row was written.
    rows = _read_trade_log(engine)
    cancels = [r for r in rows if r.get("event") == "cancel"]
    assert len(cancels) == 1
    assert record.cancel_row_written is True


@pytest.mark.asyncio
async def test_handle_cancellation_re_raises_cancelled_error(tmp_path: Path) -> None:
    """asyncio.CancelledError must propagate so SIGTERM unwinds cleanly.

    The `except Exception` around _emit_cancel_row would silently swallow
    a CancelledError because CancelledError inherits from BaseException
    in Python 3.8+, but the code now also has `except asyncio.CancelledError: raise`
    to make the intent explicit and survive a future refactor that uses
    `except BaseException`.
    """
    import asyncio

    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)

    async def _cancelled(*_args: Any, **_kwargs: Any) -> None:
        raise asyncio.CancelledError()

    engine._emit_cancel_row = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        await engine._handle_cancellation(record, record.order_id)


@pytest.mark.asyncio
async def test_kill_switch_cancel_all_orders_does_not_write_cancel_rows(
    tmp_path: Path,
) -> None:
    """Spec §5.4: cancel_all_orders (kill-switch) does NOT route through
    _handle_cancellation, so it must not produce cancel rows or fire the
    on_cancel callback.
    """
    engine, _risk, record = _make_engine_with_record(tmp_path, fill_count=0)
    record.status = OrderStatus.RESTING  # cancel_all_orders targets resting
    called_with: list[str] = []
    engine.set_on_cancel_callback(lambda oid: called_with.append(oid))

    # Mock the Kalshi client to return our resting order, then accept cancel.
    engine._client.iter_all_orders = AsyncMock(  # type: ignore[union-attr]
        return_value=([{"order_id": record.order_id}], False)
    )
    engine._client.cancel_order = AsyncMock(return_value=None)  # type: ignore[union-attr]

    await engine.cancel_all_open()

    rows = _read_trade_log(engine)
    assert [r for r in rows if r.get("event") == "cancel"] == []
    assert called_with == [], "kill-switch flatten must not fire on_cancel callback"
