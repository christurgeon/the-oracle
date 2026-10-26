"""Characterization tests for FillsReconciler._apply_cash_out / _apply_manual_close.

These pin the observable contract (sticky sets, alert dedup, event ordering,
row key order, latency accumulators) so the apply paths can be decomposed
without behavior change. Every expectation here was recorded from the
pre-refactor implementation.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import python.execution.fills_reconciler as fr
from python.common import OrderStatus
from python.notifications.base import AlertLevel
from tests.conftest import make_minimal_reconciler

_LOGGER_NAME = "python.execution.fills_reconciler"
_FIXED_NOW = datetime(2026, 5, 17, 17, 20, 0, tzinfo=UTC)
_FILL_TS = "2026-05-17T17:18:00Z"
_APPLIED_AT = "2026-05-17T17:20:00+00:00"


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return _FIXED_NOW


class _LogCapture(logging.Handler):
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        super().__init__(logging.DEBUG)
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(("log", record.levelname, record.getMessage()))


@pytest.fixture
def events():
    ev: list[tuple[Any, ...]] = []
    handler = _LogCapture(ev)
    lg = logging.getLogger(_LOGGER_NAME)
    old_level = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    yield ev
    lg.removeHandler(handler)
    lg.setLevel(old_level)


@pytest.fixture
def fixed_now(monkeypatch):
    monkeypatch.setattr(fr, "datetime", _FixedDatetime)


class _FakeRisk:
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        positions: dict[str, float],
        authoritative: str = "unknown",
    ) -> None:
        self._events = events
        self._positions = positions
        self._authoritative = authoritative

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def strategy_for_position(self, position_key: str) -> str:
        return self._authoritative

    def close_position(self, position_key: str, *, pnl: float) -> None:
        self._events.append(("risk.close_position", position_key, pnl))

    def realize_partial(self, position_key: str, *, dollars_released: float, pnl: float) -> None:
        self._events.append(("risk.realize_partial", position_key, dollars_released, pnl))

    def record_strategy_trade(self, strategy: str, pnl: float, *, won: bool) -> None:
        self._events.append(("risk.record_strategy_trade", strategy, pnl, won))


class _Rec:
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        name: str,
        position_key: str | None,
        status: OrderStatus = OrderStatus.FILLED,
        fill_count: int = 26,
    ) -> None:
        self._events = events
        self._name = name
        self.signal = None if position_key is None else SimpleNamespace(position_key=position_key)
        self._status = status
        self.fill_count = fill_count
        self.fees_from_api = 0.10

    @property
    def status(self) -> OrderStatus:
        return self._status

    @status.setter
    def status(self, value: OrderStatus) -> None:
        self._events.append(("stamp", self._name, value))
        self._status = value


def _fill_row(
    *,
    position_key: str = "KXT",
    contracts: int = 26,
    price: float = 0.09,
    fees: float = 0.10,
    strategy: str | None = "weather_edge",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event": "fill",
        "position_key": position_key,
        "market_ticker": "KXT",
        "fill_price": price,
        "contracts": contracts,
        "fees_from_api": fees,
        "mode": "live",
    }
    if strategy is not None:
        row["metadata"] = {"strategy": strategy}
    return row


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _build(
    tmp_path: Path,
    events: list[tuple[Any, ...]],
    *,
    rows: list[dict[str, Any]] | None = None,
    positions: dict[str, float] | None = None,
    authoritative: str = "unknown",
):
    log = tmp_path / "trades.jsonl"
    if rows is not None:
        _write_log(log, rows)
    rec = make_minimal_reconciler(state_path=tmp_path / "fills_reconciler.json", trade_log_path=log)
    risk = _FakeRisk(events, {"KXT": 2.34} if positions is None else positions, authoritative)
    rec._engine._risk = risk

    orig_append = rec._append_synthetic_rows
    orig_alert = rec._safe_alert
    orig_resolve = rec._safe_resolve_strategy

    async def append(rows_: list[dict[str, Any]]) -> None:
        events.append(("append", [dict(r) for r in rows_]))
        await orig_append(rows_)

    async def alert(level: AlertLevel, message: str) -> None:
        events.append(("alert", level, message))
        await orig_alert(level, message)

    def resolve(order_id: str, fill: dict[str, Any]) -> tuple[str, str]:
        events.append(("resolve", order_id))
        return orig_resolve(order_id, fill)

    rec._append_synthetic_rows = append
    rec._safe_alert = alert
    rec._safe_resolve_strategy = resolve
    events.clear()
    return rec, risk, log


def _co_kwargs(
    oid: str = "OID-1",
    *,
    count: int = 26,
    price: float = 0.86,
    fees: float = 0.10,
    ts: str | None = _FILL_TS,
    side: str | None = "no",
) -> dict[str, Any]:
    fill: dict[str, Any] = {
        "count_fp": count,
        "no_price_dollars": price,
        "fee_cost": fees,
        "is_taker": True,
        "market_ticker": "KXT",
    }
    if side is not None:
        fill["side"] = side
    if ts is not None:
        fill["created_time"] = ts
    return {
        "order_id": oid,
        "kalshi_fills": [fill],
        "position_key": "KXT",
        "market_ticker": "KXT",
        "tracked_cash_out_oids": set(),
    }


def _kinds(events: list[tuple[Any, ...]], kind: str) -> list[tuple[Any, ...]]:
    return [e for e in events if e[0] == kind]


def _logs(events: list[tuple[Any, ...]], level: str | None = None) -> list[str]:
    return [e[2] for e in events if e[0] == "log" and (level is None or e[1] == level)]


def _alerts(events: list[tuple[Any, ...]]) -> list[tuple[AlertLevel, str]]:
    return [(e[1], e[2]) for e in events if e[0] == "alert"]


def _assert_row(row: dict[str, Any], expected: dict[str, Any]) -> None:
    assert list(row.keys()) == list(expected.keys())
    for key, want in expected.items():
        if isinstance(want, dict):
            _assert_row(row[key], want)
        elif isinstance(want, float):
            assert row[key] == pytest.approx(want, abs=1e-9), key
        else:
            assert row[key] == want, key


# ---------------------------------------------------------------------------
# Reject paths: sticky sets and alert dedup, one oid applied on two ticks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_fills_logs_error_and_returns(tmp_path, events):
    rec, risk, _ = _build(tmp_path, events)
    kw = _co_kwargs()
    kw["kalshi_fills"] = []
    outcome = await rec._apply_cash_out(**kw)
    assert outcome == "EMPTY_FILLS_CASH_OUT"
    assert events == [("log", "ERROR", "_apply_cash_out: empty kalshi_fills for oid=OID-1")]


@pytest.mark.asyncio
async def test_dedup_returns_silently(tmp_path, events):
    rec, _, _ = _build(tmp_path, events)
    kw = _co_kwargs()
    kw["tracked_cash_out_oids"] = {"OID-1"}
    assert await rec._apply_cash_out(**kw) == "ALREADY_CLOSED_DEDUP_CASH_OUT"
    assert events == []


@pytest.mark.asyncio
async def test_race_lost_sets_sticky_and_warns_every_time(tmp_path, events):
    rec, _, log = _build(tmp_path, events, positions={})
    for _ in range(2):
        assert await rec._apply_cash_out(**_co_kwargs()) == "RACE_LOST_TO_SETTLEMENT"
    assert rec._co_race_lost_oids == {"OID-1"}
    assert (
        _logs(events, "WARNING")
        == [
            "_apply_cash_out: RACE_LOST oid=OID-1 position_key=KXT market_ticker=KXT "
            "hedge_side=NO — position closed between classify and apply"
        ]
        * 2
    )
    assert (
        _alerts(events)
        == [
            (
                AlertLevel.WARNING,
                "Cash out fill KXT oid=OID-1 for hedge side NO: position closed between "
                "classify and apply — likely settlement race. No state mutation.",
            )
        ]
        * 2
    )
    assert [e[0] for e in events] == ["log", "alert", "log", "alert"]
    assert not log.exists()


@pytest.mark.asyncio
async def test_race_lost_missing_side_uses_unknown_label(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, positions={})
    assert await rec._apply_cash_out(**_co_kwargs(side=None)) == "RACE_LOST_TO_SETTLEMENT"
    assert "hedge_side=UNKNOWN" in _logs(events)[0]
    assert "for hedge side UNKNOWN:" in _alerts(events)[0][1]


@pytest.mark.asyncio
async def test_race_lost_when_engine_risk_is_none(tmp_path, events):
    rec, _, _ = _build(tmp_path, events)
    rec._engine._risk = None
    assert await rec._apply_cash_out(**_co_kwargs()) == "RACE_LOST_TO_SETTLEMENT"
    assert rec._co_race_lost_oids == {"OID-1"}


@pytest.mark.asyncio
async def test_zero_basis_adds_once_and_logs_alerts_only_first_time(tmp_path, events):
    rec, _, _ = _build(tmp_path, events)
    for _ in range(2):
        assert await rec._apply_cash_out(**_co_kwargs()) == "ZERO_COST_BASIS_CASH_OUT"
    assert rec._co_zero_basis_oids == {"OID-1"}
    assert events == [
        (
            "log",
            "ERROR",
            "_apply_cash_out: ZERO_COST_BASIS oid=OID-1 position_key=KXT "
            "market_ticker=KXT — position exists but no entry FILL rows",
        ),
        (
            "alert",
            AlertLevel.CRITICAL,
            "Cash out for KXT oid=OID-1: position exists but no entry FILL rows; "
            "operator must reconcile.",
        ),
    ]


@pytest.mark.asyncio
async def test_overflow_adds_and_alerts_critical_every_time(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(contracts=10)])
    for _ in range(2):
        assert await rec._apply_cash_out(**_co_kwargs()) == "INVALID_PARTIAL_CASH_OUT"
    assert rec._co_invalid_partial_oids == {"OID-1"}
    assert (
        _logs(events, "ERROR")
        == [
            "_apply_cash_out: INVALID_PARTIAL (overflow) oid=OID-1 position_key=KXT "
            "market_ticker=KXT contracts_offset=26 cb_total_contracts=10"
        ]
        * 2
    )
    assert (
        _alerts(events)
        == [
            (
                AlertLevel.CRITICAL,
                "Cash out contracts (26) exceeded local cost-basis contracts (10) for "
                "KXT oid=OID-1; not applied.",
            )
        ]
        * 2
    )
    assert [e[0] for e in events] == ["log", "alert", "log", "alert"]


@pytest.mark.asyncio
async def test_basis_allocation_logs_every_time_alerts_only_first(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row()])
    for _ in range(2):
        outcome = await rec._apply_cash_out(**_co_kwargs(count=0))
        assert outcome == "INVALID_PARTIAL_CASH_OUT"
    assert rec._co_invalid_partial_oids == {"OID-1"}
    assert (
        _logs(events, "ERROR")
        == [
            "_apply_cash_out: INVALID_PARTIAL (basis allocation) oid=OID-1 "
            "position_key=KXT market_ticker=KXT contracts_offset=0 cb_total_contracts=26"
        ]
        * 2
    )
    assert _alerts(events) == [
        (
            AlertLevel.CRITICAL,
            "Cash out contracts (0) failed basis allocation for KXT oid=OID-1; not applied.",
        )
    ]
    assert [e[0] for e in events] == ["log", "alert", "log"]


@pytest.mark.asyncio
async def test_basis_allocation_after_overflow_does_not_alert(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row()])
    rec._co_invalid_partial_oids.add("OID-1")
    outcome = await rec._apply_cash_out(**_co_kwargs(count=0))
    assert outcome == "INVALID_PARTIAL_CASH_OUT"
    assert len(_logs(events, "ERROR")) == 1
    assert _alerts(events) == []


@pytest.mark.asyncio
async def test_pnl_cap_logs_every_time_alerts_only_first(tmp_path, events):
    rec, _, _ = _build(
        tmp_path,
        events,
        rows=[_fill_row(price=0.05, contracts=1000, fees=0.50)],
        positions={"KXT": 50.0},
    )
    rec._co_max_pnl_apply_dollars = 10.0
    for _ in range(2):
        outcome = await rec._apply_cash_out(**_co_kwargs(count=1000, price=0.99, fees=0.0))
        assert outcome == "PNL_CAP_EXCEEDED_CASH_OUT"
    assert rec._co_pnl_cap_oids == {"OID-1"}
    assert (
        _logs(events, "ERROR")
        == [
            "_apply_cash_out: PNL_CAP_EXCEEDED oid=OID-1 position_key=KXT "
            "market_ticker=KXT realized_pnl=-40.5000 cap=10.0000"
        ]
        * 2
    )
    assert _alerts(events) == [
        (
            AlertLevel.CRITICAL,
            "Cash out P&L exceeded cap: $-40.50 > $10.00 for KXT oid=OID-1; "
            "NOT applied. Investigate.",
        )
    ]
    assert [e[0] for e in events] == ["log", "alert", "log"]
    assert _kinds(events, "resolve") == []


@pytest.mark.asyncio
async def test_pnl_cap_not_hit_at_exact_cap(tmp_path, events):
    rec, _, _ = _build(
        tmp_path,
        events,
        rows=[_fill_row(price=0.05, contracts=1000, fees=0.50)],
        positions={"KXT": 50.0},
    )
    rec._co_max_pnl_apply_dollars = 40.5
    outcome = await rec._apply_cash_out(**_co_kwargs(count=1000, price=0.99, fees=0.0))
    assert outcome == "APPLIED_FULL_CASH_OUT"
    assert rec._co_pnl_cap_oids == set()


@pytest.mark.asyncio
async def test_guard_priority_race_then_zero_basis_then_overflow(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, positions={})
    assert await rec._apply_cash_out(**_co_kwargs()) == "RACE_LOST_TO_SETTLEMENT"
    assert rec._co_zero_basis_oids == set()

    rec2, _, _ = _build(tmp_path, [])
    assert await rec2._apply_cash_out(**_co_kwargs()) == "ZERO_COST_BASIS_CASH_OUT"
    assert rec2._co_invalid_partial_oids == set()


@pytest.mark.asyncio
async def test_rejects_never_mutate_or_resolve(tmp_path, events):
    rec, _, log = _build(tmp_path, events, rows=[_fill_row(contracts=10)])
    before = log.read_text()
    await rec._apply_cash_out(**_co_kwargs())
    assert log.read_text() == before
    assert _kinds(events, "append") == []
    assert _kinds(events, "resolve") == []
    assert [e for e in events if e[0].startswith("risk.")] == []
    assert rec._co_applies_this_tick == 0
    assert rec._co_realized_pnl_cents_this_tick == 0


# ---------------------------------------------------------------------------
# Cash-out commit phase.
# ---------------------------------------------------------------------------


_FULL_ROW = {
    "event": "cash_out",
    "order_id": "OID-1",
    "market_ticker": "KXT",
    "position_key": "KXT",
    "side": "YES",
    "mode": "live",
    "timestamp": _FILL_TS,
    "applied_at": _APPLIED_AT,
    "closed_side": "YES",
    "hedge_side": "NO",
    "close_type": "full",
    "contracts_offset": 26,
    "contracts_residual": 0,
    "hedge_price": 0.86,
    "hedge_proceeds": 22.36,
    "hedge_fees": 0.10,
    "cost_basis_offset_dollars": 2.34,
    "cost_basis_offset_fees": 0.10,
    "cost_basis_residual_dollars": 0.0,
    "cost_basis_residual_fees": 0.0,
    "collateral_return_estimate": 26.0,
    "realized_pnl": 1.10,
    "is_taker_fills": 26,
    "is_maker_fills": 0,
    "metadata": {
        "strategy": "weather_edge",
        "reconciled": True,
        "reconcile_source": "fills_reconciler",
        "classification": "BUY_CASH_OUT",
        "strategy_resolution": "in_memory",
    },
}


def _resolver(strategy: str, method: str):
    return lambda order_id, fill: (strategy, method)


@pytest.mark.asyncio
async def test_full_cash_out_transcript(tmp_path, events, fixed_now):
    rec, _, log = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    live = _Rec(events, "live", "KXT")
    done = _Rec(events, "done", "KXT", status=OrderStatus.SETTLED)
    other = _Rec(events, "other", "OTHER")
    nosig = _Rec(events, "nosig", None)
    rec._engine._orders = {"a": live, "b": done, "c": other, "d": nosig}

    outcome = await rec._apply_cash_out(**_co_kwargs())

    assert outcome == "APPLIED_FULL_CASH_OUT"
    assert [e[0] for e in events] == [
        "resolve",
        "append",
        "risk.close_position",
        "risk.record_strategy_trade",
        "stamp",
        "log",
        "alert",
    ]
    _assert_row(events[1][1][0], _FULL_ROW)
    assert events[2][1] == "KXT"
    assert events[2][2] == pytest.approx(1.10)
    assert events[3][1] == "weather_edge"
    assert events[3][2] == pytest.approx(1.10)
    assert events[3][3] is True
    assert events[4] == ("stamp", "live", OrderStatus.EXTERNALLY_CLOSED)
    assert done.status == OrderStatus.SETTLED
    assert other.status == OrderStatus.FILLED
    assert events[5] == (
        "log",
        "INFO",
        "_apply_cash_out: applied full for oid=OID-1 pos=KXT contracts=26 pnl=1.1000",
    )
    assert events[6] == (
        "alert",
        AlertLevel.WARNING,
        "🔄 Cash out reconciled: KXT pos=KXT [full] 26c offset @ 0.8600 "
        "realized_pnl=$+1.10. Exposure freed.",
    )
    persisted = [json.loads(line) for line in log.read_text().splitlines()]
    assert persisted[-1]["event"] == "cash_out"
    assert list(persisted[-1].keys()) == list(_FULL_ROW.keys())
    assert rec._co_realized_pnl_cents_this_tick == 110
    assert rec._co_applies_this_tick == 1
    assert rec._co_apply_latency_max_this_tick == 120.0
    assert rec._co_race_lost_oids == set()
    assert rec._co_zero_basis_oids == set()
    assert rec._co_invalid_partial_oids == set()
    assert rec._co_pnl_cap_oids == set()


@pytest.mark.asyncio
async def test_partial_cash_out_transcript(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "placement_scan")
    live = _Rec(events, "live", "KXT")
    rec._engine._orders = {"a": live}

    outcome = await rec._apply_cash_out(**_co_kwargs(count=25, fees=0.235))

    assert outcome == "APPLIED_PARTIAL_CASH_OUT"
    assert [e[0] for e in events] == [
        "resolve",
        "append",
        "risk.realize_partial",
        "risk.record_strategy_trade",
        "log",
        "alert",
    ]
    row = events[1][1][0]
    assert row["close_type"] == "partial"
    assert row["contracts_offset"] == 25
    assert row["contracts_residual"] == 1
    assert row["metadata"]["strategy_resolution"] == "placement_scan"
    assert row["cost_basis_offset_dollars"] == pytest.approx(2.25)
    assert events[2][1] == "KXT"
    assert events[2][2] == pytest.approx(2.25)
    assert live.fill_count == 1
    assert live.status == OrderStatus.FILLED
    assert events[4][2].startswith(
        "_apply_cash_out: applied partial for oid=OID-1 pos=KXT contracts=25 pnl="
    )
    assert events[5][1] == AlertLevel.WARNING
    assert events[5][2] == (
        "🔄 Cash out reconciled: KXT pos=KXT [partial: 25 offset, 1 contracts "
        "still held] @ 0.8600 realized_pnl=$+0.92. Residual exposure $0.09."
    )
    assert rec._co_applies_this_tick == 1


@pytest.mark.asyncio
async def test_pod_attribution_split_warning_precedes_risk_mutation(tmp_path, events, fixed_now):
    rec, _, _ = _build(
        tmp_path, events, rows=[_fill_row(price=0.09)], authoritative="favorite_bias"
    )
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    rec._engine._orders = {}
    await rec._apply_cash_out(**_co_kwargs())
    kinds = [e[0] for e in events]
    assert kinds[:4] == ["resolve", "append", "log", "risk.close_position"]
    assert events[2][1] == "WARNING"
    assert events[2][2] == (
        "Pod attribution split for KXT: reconciler resolved strategy='weather_edge' "
        "(for circuit-breaker P&L) but risk manager attributes exposure to "
        "'favorite_bias'. Exposure decrements the risk manager's pod; circuit-breaker "
        "P&L credits the reconciler's. Investigate if recurring."
    )


@pytest.mark.asyncio
async def test_no_pod_split_warning_when_authoritative_matches(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)], authoritative="weather_edge")
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    await rec._apply_cash_out(**_co_kwargs())
    assert not [m for m in _logs(events, "WARNING") if m.startswith("Pod attribution")]


@pytest.mark.asyncio
async def test_cash_out_cost_basis_fallback_is_silent_and_primary(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy="fav")])
    await rec._apply_cash_out(**_co_kwargs())
    row = events[[e[0] for e in events].index("append")][1][0]
    assert row["metadata"]["strategy"] == "fav"
    assert row["metadata"]["strategy_resolution"] == "cost_basis_fallback"
    assert not [m for m in _logs(events) if "cost_basis_fallback" in m]
    assert _kinds(events, "risk.record_strategy_trade")[0][1] == "fav"


@pytest.mark.asyncio
async def test_cash_out_unresolved_without_last_strategy(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy=None)])
    await rec._apply_cash_out(**_co_kwargs())
    row = _kinds(events, "append")[0][1][0]
    assert row["metadata"]["strategy"] == "unknown"
    assert row["metadata"]["strategy_resolution"] == "unresolved"
    assert _kinds(events, "risk.record_strategy_trade")[0][1] == "unknown"


@pytest.mark.asyncio
async def test_cash_out_resolver_exception_falls_back(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy="fav")])

    def boom(order_id: str, fill: dict[str, Any]) -> tuple[str, str]:
        raise RuntimeError("resolver down")

    rec._strategy_resolver = boom
    await rec._apply_cash_out(**_co_kwargs())
    assert _logs(events, "WARNING")[0] == (
        "Strategy resolver raised for order_id=OID-1 (RuntimeError); treating as unresolved"
    )
    row = _kinds(events, "append")[0][1][0]
    assert row["metadata"]["strategy"] == "fav"
    assert row["metadata"]["strategy_resolution"] == "cost_basis_fallback"


@pytest.mark.asyncio
async def test_resolver_runs_once_after_guards_before_append(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    await rec._apply_cash_out(**_co_kwargs())
    kinds = [e[0] for e in events]
    assert kinds.count("resolve") == 1
    assert kinds.index("resolve") < kinds.index("append")
    assert events[kinds.index("resolve")] == ("resolve", "OID-1")


@pytest.mark.asyncio
async def test_row_append_failure_leaves_risk_untouched(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._engine._orders = {"a": _Rec(events, "live", "KXT")}

    async def fail(rows_: list[dict[str, Any]]) -> None:
        events.append(("append-failed",))
        raise OSError("disk full")

    rec._append_synthetic_rows = fail
    with pytest.raises(OSError):
        await rec._apply_cash_out(**_co_kwargs())
    assert [e for e in events if e[0].startswith("risk.") or e[0] in ("stamp", "alert")] == []
    assert rec._co_applies_this_tick == 0
    assert rec._co_realized_pnl_cents_this_tick == 0


@pytest.mark.asyncio
async def test_cash_out_latency_keeps_larger_existing_max(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    rec._co_apply_latency_max_this_tick = 500.0
    await rec._apply_cash_out(**_co_kwargs())
    assert rec._co_apply_latency_max_this_tick == 500.0


@pytest.mark.asyncio
async def test_cash_out_empty_timestamp_counts_apply_without_latency(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    outcome = await rec._apply_cash_out(**_co_kwargs(ts=None))
    assert outcome == "APPLIED_FULL_CASH_OUT"
    assert rec._co_applies_this_tick == 1
    assert rec._co_realized_pnl_cents_this_tick == 110
    assert rec._co_apply_latency_max_this_tick == 0.0
    row = _kinds(events, "append")[0][1][0]
    assert row["timestamp"] == ""


@pytest.mark.asyncio
async def test_cash_out_unparseable_timestamp_swallowed(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    outcome = await rec._apply_cash_out(**_co_kwargs(ts="not-a-timestamp"))
    assert outcome == "APPLIED_FULL_CASH_OUT"
    assert rec._co_applies_this_tick == 1
    assert rec._co_apply_latency_max_this_tick == 0.0


@pytest.mark.asyncio
async def test_cash_out_naive_timestamp_raises_after_commit(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    with pytest.raises(TypeError):
        await rec._apply_cash_out(**_co_kwargs(ts="2026-05-17T17:18:00"))
    assert len(_kinds(events, "append")) == 1
    assert len(_kinds(events, "risk.close_position")) == 1
    assert rec._co_applies_this_tick == 1
    assert rec._co_realized_pnl_cents_this_tick == 110


@pytest.mark.asyncio
async def test_cash_out_latency_accumulates_max_across_applies(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, contracts=52)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    await rec._apply_cash_out(**_co_kwargs("OID-1", ts="2026-05-17T17:10:00Z"))
    assert rec._co_apply_latency_max_this_tick == 600.0
    await rec._apply_cash_out(**_co_kwargs("OID-2", ts=_FILL_TS))
    assert rec._co_apply_latency_max_this_tick == 600.0
    assert rec._co_applies_this_tick == 2
    assert [r["close_type"] for r in (e[1][0] for e in _kinds(events, "append"))] == [
        "partial",
        "full",
    ]


# ---------------------------------------------------------------------------
# Manual-close shared-helper pins (cost_basis_fallback warning, latency).
# ---------------------------------------------------------------------------


def _mc_kwargs(
    oid: str = "SELL-1", *, count: int = 26, ts: str | None = _FILL_TS
) -> dict[str, Any]:
    fill: dict[str, Any] = {
        "order_id": oid,
        "action": "sell",
        "side": "yes",
        "yes_price_dollars": 0.65,
        "no_price_dollars": 0.35,
        "count_fp": count,
        "fee_cost": 0.04,
        "is_taker": True,
        "market_ticker": "KXT",
    }
    if ts is not None:
        fill["created_time"] = ts
    return {
        "order_id": oid,
        "kalshi_fills": [fill],
        "position_key": "KXT",
        "market_ticker": "KXT",
        "tracked_manual_close_oids": set(),
    }


_MC_ROW = {
    "event": "manual_close",
    "order_id": "SELL-1",
    "market_ticker": "KXT",
    "position_key": "KXT",
    "side": "YES",
    "mode": "live",
    "timestamp": _FILL_TS,
    "applied_at": _APPLIED_AT,
    "close_price": 0.65,
    "contracts_closed": 26,
    "close_proceeds": 16.9,
    "cost_basis_dollars": 2.34,
    "cost_basis_contracts": 26,
    "entry_fees": 0.10,
    "exit_fees": 0.04,
    "realized_pnl": 14.42,
    "is_taker_fills": 26,
    "is_maker_fills": 0,
    "fees_from_api": 0.04,
    "metadata": {
        "strategy": "weather_edge",
        "reconciled": True,
        "reconcile_source": "fills_reconciler",
        "classification": "SELL_MANUAL",
        "close_type": "full",
        "strategy_resolution": "in_memory",
    },
}


@pytest.mark.asyncio
async def test_manual_close_transcript(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    live = _Rec(events, "live", "KXT")
    done = _Rec(events, "done", "KXT", status=OrderStatus.SETTLED)
    other = _Rec(events, "other", "OTHER")
    nosig = _Rec(events, "nosig", None)
    rec._engine._orders = {"a": live, "b": done, "c": other, "d": nosig}

    outcome = await rec._apply_manual_close(**_mc_kwargs())

    assert outcome == "APPLIED"
    assert [e[0] for e in events] == [
        "resolve",
        "append",
        "risk.close_position",
        "risk.record_strategy_trade",
        "stamp",
        "stamp",
        "alert",
    ]
    _assert_row(events[1][1][0], _MC_ROW)
    assert events[4] == ("stamp", "live", OrderStatus.EXTERNALLY_CLOSED)
    assert events[5] == ("stamp", "done", OrderStatus.EXTERNALLY_CLOSED)
    assert other.status == OrderStatus.FILLED
    assert events[6] == (
        "alert",
        AlertLevel.WARNING,
        "🔄 Manual close reconciled: KXT pos=KXT 26c @ 65¢ realized_pnl=$+14.42. Exposure freed.",
    )
    assert rec._manual_close_realized_pnl_cents_this_tick == 1442
    assert rec._manual_close_apply_latency_max_this_tick == 120.0


@pytest.mark.asyncio
async def test_manual_close_cost_basis_fallback_warns_before_append(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy="fav")])
    outcome = await rec._apply_manual_close(**_mc_kwargs())
    assert outcome == "APPLIED"
    kinds = [e[0] for e in events]
    assert kinds[:3] == ["resolve", "log", "append"]
    assert events[1] == (
        "log",
        "WARNING",
        "manual_close: strategy attributed via cost_basis_fallback oid=SELL-1 "
        "position_key=KXT strategy=fav — verify if multi-leg",
    )
    row = events[2][1][0]
    assert row["metadata"]["strategy"] == "fav"
    assert row["metadata"]["strategy_resolution"] == "cost_basis_fallback"
    assert _kinds(events, "risk.record_strategy_trade")[0][1] == "fav"


@pytest.mark.asyncio
async def test_manual_close_resolved_strategy_does_not_warn(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy="fav")])
    rec._strategy_resolver = _resolver("weather_edge", "placement_scan")
    await rec._apply_manual_close(**_mc_kwargs())
    assert not [m for m in _logs(events) if "cost_basis_fallback" in m]
    row = _kinds(events, "append")[0][1][0]
    assert row["metadata"]["strategy"] == "weather_edge"
    assert row["metadata"]["strategy_resolution"] == "placement_scan"


@pytest.mark.asyncio
async def test_manual_close_unresolved_without_last_strategy_does_not_warn(
    tmp_path, events, fixed_now
):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09, strategy=None)])
    await rec._apply_manual_close(**_mc_kwargs())
    assert not [m for m in _logs(events) if "cost_basis_fallback" in m]
    row = _kinds(events, "append")[0][1][0]
    assert row["metadata"]["strategy"] == "unknown"
    assert row["metadata"]["strategy_resolution"] == "unresolved"


@pytest.mark.asyncio
async def test_manual_close_pod_attribution_split_warning(tmp_path, events, fixed_now):
    rec, _, _ = _build(
        tmp_path, events, rows=[_fill_row(price=0.09)], authoritative="favorite_bias"
    )
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    await rec._apply_manual_close(**_mc_kwargs())
    kinds = [e[0] for e in events]
    assert kinds[:4] == ["resolve", "append", "log", "risk.close_position"]
    assert events[2][1] == "WARNING"
    assert events[2][2].startswith(
        "Pod attribution split for KXT: reconciler resolved strategy='weather_edge'"
    )
    assert "attributes exposure to 'favorite_bias'." in events[2][2]


@pytest.mark.asyncio
async def test_manual_close_latency_paths(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    await rec._apply_manual_close(**_mc_kwargs("S1", ts="2026-05-17T17:10:00Z"))
    assert rec._manual_close_apply_latency_max_this_tick == 600.0
    await rec._apply_manual_close(**_mc_kwargs("S2", ts=_FILL_TS))
    assert rec._manual_close_apply_latency_max_this_tick == 600.0
    await rec._apply_manual_close(**_mc_kwargs("S3", ts=None))
    assert rec._manual_close_apply_latency_max_this_tick == 600.0
    await rec._apply_manual_close(**_mc_kwargs("S4", ts="garbage"))
    assert rec._manual_close_apply_latency_max_this_tick == 600.0
    assert rec._manual_close_realized_pnl_cents_this_tick == 4 * 1442


@pytest.mark.asyncio
async def test_manual_close_naive_timestamp_raises_after_commit(tmp_path, events, fixed_now):
    rec, _, _ = _build(tmp_path, events, rows=[_fill_row(price=0.09)])
    rec._strategy_resolver = _resolver("weather_edge", "in_memory")
    with pytest.raises(TypeError):
        await rec._apply_manual_close(**_mc_kwargs(ts="2026-05-17T17:18:00"))
    assert len(_kinds(events, "append")) == 1
    assert len(_kinds(events, "risk.close_position")) == 1
    assert rec._manual_close_realized_pnl_cents_this_tick == 1442


@pytest.mark.asyncio
async def test_manual_close_orphan_when_position_missing(tmp_path, events):
    rec, _, _ = _build(tmp_path, events, positions={})
    assert await rec._apply_manual_close(**_mc_kwargs()) == "ORPHAN"
    assert events == [
        (
            "log",
            "WARNING",
            "manual_close: ORPHAN sell oid=SELL-1 position_key=KXT — no position to close",
        ),
        (
            "alert",
            AlertLevel.WARNING,
            "Sell fill SELL-1 for KXT but no local position (KXT) to close — likely "
            "already-closed or external.",
        ),
    ]


@pytest.mark.asyncio
async def test_manual_close_zero_basis_dedups_alert(tmp_path, events):
    rec, _, _ = _build(tmp_path, events)
    for _ in range(2):
        assert await rec._apply_manual_close(**_mc_kwargs()) == "ZERO_COST_BASIS"
    assert rec._orphan_close_oids == {"SELL-1"}
    assert [e[0] for e in events] == ["log", "alert"]
    assert events[1][1] == AlertLevel.CRITICAL
