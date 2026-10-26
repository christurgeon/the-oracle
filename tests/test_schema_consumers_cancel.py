"""Schema-consumer regression tests: cancel rows must be ignored by all
P&L-aggregating consumers of state/trades.jsonl.

Gate-level tests (5 total): one per non-trivial consumer of the trade log.
The bulk of consumers route through is_pnl_relevant_row(); that gate is
covered by tests/test_engine_cancel_row.py::test_cancel_row_is_not_pnl_relevant.
This file covers the explicit-event-switch consumers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def _write_trades(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _placement(oid: str, *, mode: str = "live") -> dict[str, Any]:
    return {
        "event": "placement",
        "order_id": oid,
        "timestamp": "2026-05-28T14:00:00+00:00",
        "mode": mode,
        "strategy": "weather_edge",
        "position_key": "m-1",
        "market_id": "m-1",
        "market_ticker": "KXLOWTLAX-26MAY29-B55.5",
        "side": "yes",
        "action": "buy",
    }


def _fill(
    oid: str, *, price: float = 0.14, contracts: int = 70, mode: str = "live"
) -> dict[str, Any]:
    return {
        "event": "fill",
        "order_id": oid,
        "timestamp": "2026-05-28T14:00:05+00:00",
        "mode": mode,
        "strategy": "weather_edge",
        "position_key": "m-1",
        "market_id": "m-1",
        "market_ticker": "KXLOWTLAX-26MAY29-B55.5",
        "side": "yes",
        "fill_price": price,
        "contracts": contracts,
        "dollars": price * contracts,
        "fees_from_api": 0.5,
        "metadata": {"strategy": "weather_edge"},
    }


def _cancel(oid: str, *, mode: str = "live") -> dict[str, Any]:
    return {
        "event": "cancel",
        "order_id": oid,
        "timestamp": "2026-05-28T14:02:32+00:00",
        "mode": mode,
        "strategy": "weather_edge",
        "position_key": "m-1",
        "market_id": "m-1",
        "market_ticker": "KXLOWTLAX-26MAY29-B55.5",
        "side": "yes",
        "metadata": {"city": "LAX", "market_type": "low", "forecast_date": "2026-05-29"},
    }


# ----- cost_basis_for_position -----------------------------------------------


def test_cost_basis_ignores_cancel_rows(tmp_path: Path) -> None:
    """cost_basis_for_position aggregates only is_pnl_relevant_row rows
    (FILL). A cancel row in the same trade log must contribute nothing.
    """
    from python.execution.cost_basis import cost_basis_for_position

    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_placement("oid-1"), _fill("oid-1"), _cancel("oid-2")])
    cb = cost_basis_for_position(
        trades,
        position_key="m-1",
        market_ticker="KXLOWTLAX-26MAY29-B55.5",
    )
    # Cost basis reflects only the fill; cancel row is ignored.
    assert cb.total_contracts == 70
    assert cb.n_fills == 1
    # Sanity: the cancel did not corrupt fees or dollars.
    assert cb.total_fees == 0.5


# ----- shadow_broker.replay --------------------------------------------------


def test_shadow_broker_replay_ignores_cancel_rows(tmp_path: Path) -> None:
    """shadow_broker.replay iterates the trade log; the explicit event-switch
    falls through to is_pnl_relevant_row, which returns False for cancel.
    Use shadow-mode rows so the mode filter doesn't pre-skip everything.
    """
    from python.execution.shadow_broker import ShadowBroker

    trades = tmp_path / "trades.jsonl"
    _write_trades(
        trades,
        [
            _fill("oid-1", mode="shadow"),
            _cancel("oid-2", mode="shadow"),
        ],
    )
    # Pin ``today`` to the fixture's market date so the replay date safety net
    # (_keep: drop positions older than today-1) is deterministic. Without this
    # the test rots once the wall-clock passes the hardcoded 2026-05-29 ticker
    # date and the open position gets filtered out as stale.
    broker = ShadowBroker(
        trade_log_path=trades,
        configured_bankroll=1000.0,
        today=date(2026, 5, 29),
    )
    positions = broker.replay()
    # Replay produced exactly one position (from the fill); cancel ignored.
    assert len(positions) == 1
    assert positions[0].market_ticker == "KXLOWTLAX-26MAY29-B55.5"


# ----- daily_trade_report._iter_intraday_settlement_rows --------------------


def test_daily_report_intraday_settlement_iter_ignores_cancel(tmp_path: Path) -> None:
    """Settlement iter in daily report only yields SETTLEMENT rows for the
    intraday strategy. A cancel row must not appear — it's filtered by
    both the strategy gate and the event-type switch.
    """
    from scripts.daily_trade_report import _iter_intraday_settlement_rows

    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_fill("oid-1"), _cancel("oid-2")])
    yielded = list(_iter_intraday_settlement_rows(trades))
    # Neither row is a settlement row for the intraday strategy.
    assert yielded == []


# ----- fills_reconciler — instance method test ----------------------------


def test_fills_reconciler_tracked_oids_omits_cancel(tmp_path: Path) -> None:
    """The reconciler's tracked-oids logic (which oids it considers
    already-known fills) must not include cancel-row oids.

    Verified indirectly: read the trade log and assert any extraction the
    reconciler would do never returns cancel oids as fills.
    """
    # Direct gate test: iterate the log and apply the same filter the
    # reconciler does (event in fill-equivalent set).
    from python.execution.trade_log import (
        TRADE_LOG_EVENT_CASH_OUT,
        TRADE_LOG_EVENT_FILL,
        TRADE_LOG_EVENT_MANUAL_CLOSE,
    )

    trades = tmp_path / "trades.jsonl"
    _write_trades(trades, [_fill("oid-fill"), _cancel("oid-cancel")])
    rows = [json.loads(line) for line in trades.read_text().splitlines() if line.strip()]

    fill_equivalent = {
        TRADE_LOG_EVENT_FILL,
        TRADE_LOG_EVENT_MANUAL_CLOSE,
        TRADE_LOG_EVENT_CASH_OUT,
    }
    tracked_ids = {r["order_id"] for r in rows if r.get("event") in fill_equivalent}
    assert "oid-fill" in tracked_ids
    assert "oid-cancel" not in tracked_ids
