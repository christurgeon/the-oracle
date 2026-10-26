"""Tests for the v1.1 cash-out apply path. Spec §6.2."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

# Task 0 conftest.py provides these factories (no per-module imports needed
# since conftest.py is auto-discovered by pytest):
#   - make_minimal_reconciler (module-level helper, importable for non-fixture use)
#   - make_test_risk_manager (real RiskManager, NOT a SimpleNamespace mock)
from tests.conftest import (
    make_minimal_reconciler as _make_minimal_reconciler,
)
from tests.conftest import (
    make_test_risk_manager,
)


@pytest.mark.asyncio
async def test_apply_cash_out_dedup_returns_already_closed(tmp_path):
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[{"side": "no", "count_fp": 26, "no_price_dollars": 0.86}],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids={"OID-1"},  # already in trade log
    )
    assert outcome == "ALREADY_CLOSED_DEDUP_CASH_OUT"


@pytest.mark.asyncio
async def test_apply_cash_out_race_lost_when_position_missing(tmp_path):
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    # M1 fix: _apply_cash_out now uses risk.positions (public property),
    # mirroring _apply_manual_close. SimpleNamespace exposes `positions`
    # so the membership test (position_key not in risk.positions) works.
    reconciler._engine._risk = SimpleNamespace(positions={})  # vanished
    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 26,
                "no_price_dollars": 0.86,
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )
    assert outcome == "RACE_LOST_TO_SETTLEMENT"
    assert "OID-1" in reconciler._co_race_lost_oids


@pytest.mark.asyncio
async def test_apply_cash_out_zero_basis_when_no_fill_rows(tmp_path):
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 1.00})
    # Empty trade log → cost basis is 0.
    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 26,
                "no_price_dollars": 0.86,
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )
    assert outcome == "ZERO_COST_BASIS_CASH_OUT"
    assert "OID-1" in reconciler._co_zero_basis_oids


@pytest.mark.asyncio
async def test_apply_cash_out_contracts_overflow(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "fill",
                "position_key": "KXT",
                "market_ticker": "KXT",
                "fill_price": 0.09,
                "contracts": 10,
                "fees_from_api": 0.05,
                "metadata": {"strategy": "weather_edge"},
            }
        )
        + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 0.90})

    # contracts_offset (26) > cb.total_contracts (10).
    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 26,
                "no_price_dollars": 0.86,
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )
    assert outcome == "INVALID_PARTIAL_CASH_OUT"
    assert "OID-1" in reconciler._co_invalid_partial_oids


@pytest.mark.asyncio
async def test_apply_cash_out_pnl_cap_exceeded(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "fill",
                "position_key": "KXT",
                "market_ticker": "KXT",
                "fill_price": 0.05,
                "contracts": 1000,
                "fees_from_api": 0.50,
                "metadata": {"strategy": "weather_edge"},
            }
        )
        + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    # cb: total_dollars=$50, total_fees=$0.50; hedge_proceeds=$990, hedge_fees=$0
    # realized_pnl = 1000*1.00 - 50 - 0.50 - 990 - 0 = -$40.50
    # Use a $10 cap so |realized_pnl| > cap triggers.
    reconciler._co_max_pnl_apply_dollars = 10.0
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 50.0})

    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 1000,
                "no_price_dollars": 0.99,
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )
    assert outcome == "PNL_CAP_EXCEEDED_CASH_OUT"
    assert "OID-1" in reconciler._co_pnl_cap_oids


# ---------------------------------------------------------------------------
# Task 13: apply-body tests (full / partial / second-cash-out-on-residual).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_cash_out_full_close_path(tmp_path):
    """Full close: real RiskManager so bankroll/daily_pnl mutations are
    verifiable; OrderRecord stamping + record_strategy_trade args verified.
    """
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "fill",
                "position_key": "KXT",
                "market_ticker": "KXT",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            }
        )
        + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )

    # Real RiskManager — NOT a SimpleNamespace. Round-2 plan-review CRIT-5
    # requires verifiable bankroll/daily_pnl mutations. bankroll=10_000.0
    # because the Settings sizing-floor validator rejects smaller values
    # (kelly * single * exposure * bankroll >= 1.0).
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    risk._positions["KXT"] = 2.34
    prev_bankroll = risk._bankroll
    prev_daily = risk._daily_pnl
    reconciler._engine._risk = risk

    # Track record_strategy_trade args explicitly (spec §8 invariant 4).
    recorded_trades = []
    original_rst = risk.record_strategy_trade

    def capturing_rst(strategy, pnl, *, won):
        recorded_trades.append({"strategy": strategy, "pnl": pnl, "won": won})
        return original_rst(strategy, pnl, won=won)

    risk.record_strategy_trade = capturing_rst

    # Add a matching OrderRecord so EXTERNALLY_CLOSED stamping is verifiable.
    from python.common import OrderStatus

    rec = SimpleNamespace(
        signal=SimpleNamespace(position_key="KXT"),
        status=OrderStatus.FILLED,
        fill_count=26,
        fees_from_api=0.10,
    )
    reconciler._engine._orders = {"O1": rec}

    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 26,
                "no_price_dollars": 0.86,
                "fee_cost": 0.10,
                "is_taker": True,
                "created_time": "2026-05-17T17:18:00Z",
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )

    assert outcome == "APPLIED_FULL_CASH_OUT"

    # Row appended (FIRST in the apply order — durable dedup token).
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    cash_out_rows = [r for r in rows if r.get("event") == "cash_out"]
    assert len(cash_out_rows) == 1
    assert cash_out_rows[0]["close_type"] == "full"
    assert cash_out_rows[0]["contracts_offset"] == 26
    assert cash_out_rows[0]["contracts_residual"] == 0

    # Bankroll mutated correctly (BLOCK-1 / CRIT-5 round-1 fix verification).
    expected_pnl = 26 * 1.00 - 2.34 - 0.10 - (26 * 0.86) - 0.10  # ≈ +1.14
    assert risk._bankroll == pytest.approx(prev_bankroll + expected_pnl, abs=0.01)
    assert risk._daily_pnl == pytest.approx(prev_daily + expected_pnl, abs=0.01)

    # Position popped (full close).
    assert "KXT" not in risk._positions

    # record_strategy_trade called with offset-leg P&L + won flag (spec §8 #4).
    assert len(recorded_trades) == 1
    assert recorded_trades[0]["strategy"] == "weather_edge"
    assert recorded_trades[0]["pnl"] == pytest.approx(expected_pnl, abs=0.01)
    assert recorded_trades[0]["won"] is True

    # OrderRecord stamped EXTERNALLY_CLOSED (full close only).
    assert rec.status == OrderStatus.EXTERNALLY_CLOSED


@pytest.mark.asyncio
async def test_apply_cash_out_partial_close_path(tmp_path):
    """Partial close: real RiskManager + bankroll/daily_pnl/peak_bankroll
    triple assertion (spec §11 CRIT-5 — the test that catches BLOCK-1).
    """
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "fill",
                "position_key": "KXT",
                "market_ticker": "KXT",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            }
        )
        + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )

    # Real RiskManager so realize_partial actually mutates bankroll/daily_pnl.
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    risk._positions["KXT"] = 2.34
    prev_bankroll = risk._bankroll
    prev_daily = risk._daily_pnl
    prev_peak = risk._peak_bankroll
    reconciler._engine._risk = risk

    # Track record_strategy_trade args (spec §8 invariant 5 — accepted bias).
    recorded_trades = []
    original_rst = risk.record_strategy_trade

    def capturing_rst(strategy, pnl, *, won):
        recorded_trades.append({"strategy": strategy, "pnl": pnl, "won": won})
        return original_rst(strategy, pnl, won=won)

    risk.record_strategy_trade = capturing_rst

    from python.common import OrderStatus

    rec = SimpleNamespace(
        signal=SimpleNamespace(position_key="KXT"),
        status=OrderStatus.FILLED,
        fill_count=26,
        fees_from_api=0.10,
    )
    reconciler._engine._orders = {"O1": rec}

    outcome = await reconciler._apply_cash_out(
        order_id="OID-1",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 25,
                "no_price_dollars": 0.86,
                "fee_cost": 0.235,
                "is_taker": True,
                "created_time": "2026-05-17T17:18:00Z",
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )

    assert outcome == "APPLIED_PARTIAL_CASH_OUT"

    # Row reflects partial (close_type + residual fields).
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    cash_out_rows = [r for r in rows if r.get("event") == "cash_out"]
    assert len(cash_out_rows) == 1
    row = cash_out_rows[0]
    assert row["close_type"] == "partial"
    assert row["contracts_offset"] == 25
    assert row["contracts_residual"] == 1
    assert row["cost_basis_offset_dollars"] == pytest.approx(2.25)
    assert row["cost_basis_residual_dollars"] == pytest.approx(0.09, abs=1e-3)

    # fill_count decremented proportionally (round-1 BLOCK-1 fix).
    assert rec.fill_count == 1
    # OrderRecord NOT stamped (partial leaves records intact for residual settlement).
    assert rec.status == OrderStatus.FILLED

    # CRIT-5 TRIPLE ASSERTION — round-2 plan-review BLOCKER.
    # Compute expected offset-leg realized P&L.
    expected_pnl = 25 * 1.00 - 2.25 - 0.096 - (25 * 0.86) - 0.235  # ≈ +0.92
    assert risk._bankroll == pytest.approx(prev_bankroll + expected_pnl, abs=0.01), (
        "BLOCK-1 regression: realize_partial did not flow P&L to bankroll"
    )
    assert risk._daily_pnl == pytest.approx(prev_daily + expected_pnl, abs=0.01), (
        "BLOCK-1 regression: realize_partial did not flow P&L to _daily_pnl"
    )
    assert risk._peak_bankroll == max(prev_peak, risk._bankroll), (
        "BLOCK-1 regression: _peak_bankroll not updated"
    )

    # Position retained at residual basis (NOT popped).
    assert risk._positions["KXT"] == pytest.approx(0.09, abs=0.01)

    # record_strategy_trade called with offset-leg P&L (spec §8 #5 accepted bias).
    assert len(recorded_trades) == 1
    assert recorded_trades[0]["pnl"] == pytest.approx(expected_pnl, abs=0.01)

    # Telegram WARNING text includes the residual breakdown (spec §11 partial
    # test #9 — round-2 plan-review MINOR-7).
    alerts = reconciler._notifier.sent_alerts  # type: ignore[attr-defined]
    partial_alerts = [
        msg
        for level, msg in alerts
        if "partial: 25 offset" in msg and "1 contracts still held" in msg
    ]
    assert len(partial_alerts) == 1, (
        "partial cash-out WARNING must name offset + residual counts explicitly"
    )


@pytest.mark.asyncio
async def test_apply_cash_out_second_cash_out_on_residual(tmp_path):
    """Spec §9 #5: after a partial cash-out, a SECOND cash-out on the
    residual gets the correct effective basis from cost_basis_for_position's
    cash_out subtraction pass (round-3 R3-2 verification).
    """
    log = tmp_path / "trades.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "fill",
                        "position_key": "KXT",
                        "market_ticker": "KXT",
                        "fill_price": 0.09,
                        "contracts": 26,
                        "fees_from_api": 0.10,
                        "metadata": {"strategy": "weather_edge"},
                    }
                ),
                # Prior partial cash-out (25-of-26) — subtracts from effective basis.
                json.dumps(
                    {
                        "event": "cash_out",
                        "position_key": "KXT",
                        "market_ticker": "KXT",
                        "close_type": "partial",
                        "contracts_offset": 25,
                        "cost_basis_offset_dollars": 2.25,
                        "cost_basis_offset_fees": 0.096,
                    }
                ),
            ]
        )
        + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    # _positions reflects residual after the first partial (set externally
    # to simulate state-after-first-cash-out).
    risk._positions["KXT"] = 0.09
    reconciler._engine._risk = risk

    from python.common import OrderStatus

    rec = SimpleNamespace(
        signal=SimpleNamespace(position_key="KXT"),
        status=OrderStatus.FILLED,
        fill_count=1,
        fees_from_api=0.004,
    )
    reconciler._engine._orders = {"O1": rec}

    # Second cash-out for the 1-contract residual.
    outcome = await reconciler._apply_cash_out(
        order_id="OID-2",
        kalshi_fills=[
            {
                "side": "no",
                "count_fp": 1,
                "no_price_dollars": 0.86,
                "fee_cost": 0.01,
                "is_taker": True,
                "created_time": "2026-05-17T17:19:00Z",
                "market_ticker": "KXT",
            }
        ],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids={"OID-1"},  # first cash-out already in trade log
    )

    # Effective basis from cost_basis_for_position = $2.34 - $2.25 = $0.09;
    # 26 - 25 = 1 contract. Second cash-out at 1-of-1 → FULL CLOSE.
    assert outcome == "APPLIED_FULL_CASH_OUT", (
        "second cash-out on residual must route to FULL via effective basis"
    )
    # Position popped.
    assert "KXT" not in risk._positions


# ---------------------------------------------------------------------------
# Task 14: _handle_shakedown_cash_out shadow path (spec §6.2 / §6.9 / §10.2).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shakedown_cash_out_emits_shadow_alert_no_mutation(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(json.dumps({
        "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
        "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
        "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    reconciler._co_apply_enabled = False  # shakedown mode
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 2.34})
    seen_alerts = []
    reconciler._notifier = SimpleNamespace(
        send_alert=lambda msg, level=None: seen_alerts.append((level, msg))
    )

    outcome = await reconciler._handle_shakedown_cash_out(
        order_id="OID-1",
        kalshi_fills=[{
            "side": "no", "count_fp": 25, "no_price_dollars": 0.86,
            "fee_cost": 0.235, "is_taker": True,
            "market_ticker": "KXT",
        }],
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )

    assert outcome == "SHAKEDOWN_ALERTED"
    assert "OID-1" in reconciler._shakedown_seen_oids_cash_out
    assert any("[SHADOW] Cash out detected" in m for _, m in seen_alerts)


# ---------------------------------------------------------------------------
# Task 15: BUY_CASH_OUT routed through _apply_fill_kind dispatcher.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_fill_kind_buy_cash_out_routes_to_shakedown_when_disabled(tmp_path):
    from python.execution.fills_reconciler import FillKind

    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    reconciler._co_apply_enabled = False
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 2.34})
    outcome = await reconciler._apply_fill_kind(
        kind=FillKind.BUY_CASH_OUT,
        order_id="OID-1",
        kalshi_fills=[{
            "side": "no", "count_fp": 25,
            "no_price_dollars": 0.86, "market_ticker": "KXT",
        }],
        position_key="KXT", market_ticker="KXT",
        tracked_manual_close_oids=set(), tracked_cash_out_oids=set(),
    )
    assert outcome in ("SHAKEDOWN_ALERTED", "SHAKEDOWN_SUPPRESSED")


@pytest.mark.asyncio
async def test_apply_fill_kind_buy_cash_out_canary_limit_blocks_apply(tmp_path):
    from python.execution.fills_reconciler import FillKind

    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    reconciler._co_apply_enabled = True
    reconciler._co_apply_limit = 1
    reconciler._co_applies_this_restart = 1  # already at limit
    outcome = await reconciler._apply_fill_kind(
        kind=FillKind.BUY_CASH_OUT,
        order_id="OID-1",
        kalshi_fills=[{
            "side": "no", "count_fp": 25,
            "no_price_dollars": 0.86, "market_ticker": "KXT",
        }],
        position_key="KXT", market_ticker="KXT",
        tracked_manual_close_oids=set(), tracked_cash_out_oids=set(),
    )
    assert outcome == "CANARY_LIMIT_REACHED_CASH_OUT"


@pytest.mark.asyncio
async def test_tick_log_extra_includes_cash_out_fields(tmp_path):
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    reconciler._co_realized_pnl_cents_this_tick = 92
    reconciler._co_apply_latency_max_this_tick = 1.234
    reconciler._co_applies_this_tick = 1

    extra = reconciler._build_tick_log_extra(
        started_at_ts=0.0,
        ended_at_ts=1.0,
        duration_ms=1000,
        pages_consumed=1,
        truncated=False,
        cycle_aborted=False,
        fills_fetched=0,
        delta_count=0,
        order_ids_synthesised=[],
        resolution_breakdown={},
        alert_state="clean",
        cursor_before=0,
        cursor_after=0,
        classifier_decision_counts={},
        apply_path_outcome_counts={},
        pending_partial_sells_count=0,
    )
    assert extra["co_realized_pnl_cents_this_tick"] == 92
    assert extra["co_apply_latency_seconds"] == 1.234
    assert extra["co_applies_this_tick"] == 1


@pytest.mark.asyncio
async def test_tick_log_extra_latency_none_when_no_applies(tmp_path):
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    reconciler._co_realized_pnl_cents_this_tick = 0
    reconciler._co_apply_latency_max_this_tick = 0.0
    reconciler._co_applies_this_tick = 0

    extra = reconciler._build_tick_log_extra(
        started_at_ts=0.0,
        ended_at_ts=1.0,
        duration_ms=1000,
        pages_consumed=1,
        truncated=False,
        cycle_aborted=False,
        fills_fetched=0,
        delta_count=0,
        order_ids_synthesised=[],
        resolution_breakdown={},
        alert_state="clean",
        cursor_before=0,
        cursor_after=0,
        classifier_decision_counts={},
        apply_path_outcome_counts={},
        pending_partial_sells_count=0,
    )
    assert extra["co_apply_latency_seconds"] is None


def _multi_hedge_fills() -> list[dict]:
    """Three hedge (opposite-side buy) fills offsetting a 26-contract YES position.

    VWAP = (6*0.84 + 15*0.86 + 5*0.88) / 26 = 22.34 / 26
    fees = 0.03 + 0.05 + 0.02 = 0.10
    taker contracts = 6 + 5 = 11; maker contracts = 15.
    """
    return [
        {
            "side": "no", "count_fp": 6, "no_price_dollars": 0.84,
            "fee_cost": 0.03, "is_taker": True,
            "created_time": "2026-05-17T17:18:01Z", "market_ticker": "KXT",
        },
        {
            "side": "no", "count_fp": 15, "no_price_dollars": 0.86,
            "fee_cost": 0.05, "is_taker": False,
            "created_time": "2026-05-17T17:18:00Z", "market_ticker": "KXT",
        },
        {
            "side": "no", "count_fp": 5, "no_price_dollars": 0.88,
            "fee_cost": 0.02, "is_taker": True,
            "created_time": "2026-05-17T17:18:02Z", "market_ticker": "KXT",
        },
    ]


@pytest.mark.asyncio
async def test_apply_cash_out_multi_fill_aggregation_is_pinned(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    risk._positions["KXT"] = 2.34
    reconciler._engine._risk = risk
    reconciler._engine._orders = {}

    outcome = await reconciler._apply_cash_out(
        order_id="OID-M",
        kalshi_fills=_multi_hedge_fills(),
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )

    assert outcome == "APPLIED_FULL_CASH_OUT"
    row = [
        json.loads(x) for x in log.read_text().splitlines()
        if json.loads(x).get("event") == "cash_out"
    ][0]
    assert row["contracts_offset"] == 26
    assert row["contracts_residual"] == 0
    assert row["close_type"] == "full"
    assert row["hedge_proceeds"] == pytest.approx(22.34)
    assert row["hedge_price"] == pytest.approx(22.34 / 26, abs=1e-4)
    assert row["hedge_fees"] == pytest.approx(0.10)
    assert row["is_taker_fills"] == 11
    assert row["is_maker_fills"] == 15
    assert row["timestamp"] == "2026-05-17T17:18:00Z"
    assert row["hedge_side"] == "NO"
    assert row["closed_side"] == "YES"
    # 26 * 1.00 - 2.34 (basis) - 0.10 (entry fees) - 22.34 - 0.10
    assert row["realized_pnl"] == pytest.approx(1.12, abs=0.005)


@pytest.mark.asyncio
async def test_apply_cash_out_raises_on_none_side(tmp_path):
    """An explicit None side now raises, matching manual_close/synthetic_row/
    shakedown — the (x or "").upper() leniency this site had is removed."""
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    risk._positions["KXT"] = 2.34
    reconciler._engine._risk = risk
    reconciler._engine._orders = {}

    fills = _multi_hedge_fills()
    fills[0]["side"] = None

    with pytest.raises(AttributeError):
        await reconciler._apply_cash_out(
            order_id="OID-N",
            kalshi_fills=fills,
            position_key="KXT",
            market_ticker="KXT",
            tracked_cash_out_oids=set(),
        )
    assert log.read_text().count("\n") == 1  # only the seed row


@pytest.mark.asyncio
async def test_apply_cash_out_raises_on_none_fee_cost(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    risk = make_test_risk_manager(bankroll=10_000.0, state_dir=tmp_path)
    risk._positions["KXT"] = 2.34
    reconciler._engine._risk = risk
    reconciler._engine._orders = {}

    fills = _multi_hedge_fills()
    fills[1]["fee_cost"] = None

    with pytest.raises(TypeError):
        await reconciler._apply_cash_out(
            order_id="OID-O",
            kalshi_fills=fills,
            position_key="KXT",
            market_ticker="KXT",
            tracked_cash_out_oids=set(),
        )
    assert log.read_text().count("\n") == 1  # only the seed row — no cash_out row written


@pytest.mark.asyncio
async def test_shakedown_cash_out_multi_fill_pnl_is_pinned(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    reconciler._co_apply_enabled = False
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 2.34})
    seen_alerts = []
    reconciler._notifier = SimpleNamespace(
        send_alert=lambda msg, level=None: seen_alerts.append((level, msg))
    )

    outcome = await reconciler._handle_shakedown_cash_out(
        order_id="OID-M",
        kalshi_fills=_multi_hedge_fills(),
        position_key="KXT",
        market_ticker="KXT",
        tracked_cash_out_oids=set(),
    )

    assert outcome == "SHAKEDOWN_ALERTED"
    msg = [m for _, m in seen_alerts if "[SHADOW] Cash out detected" in m][0]
    assert "hedge_proceeds=$22.34" in msg
    # basis_share = 2.34 (full offset), fees = 0.10 basis share + 0.10 hedge
    assert "basis_share=$2.34" in msg
    assert "fees=$0.20" in msg
    assert "computed_pnl=$+1.12" in msg


@pytest.mark.asyncio
async def test_shakedown_cash_out_raises_on_none_side(tmp_path):
    log = tmp_path / "trades.jsonl"
    log.write_text(
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "fill_price": 0.09, "contracts": 26, "fees_from_api": 0.10,
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )
    reconciler = _make_minimal_reconciler(
        state_path=tmp_path / "fills_reconciler.json",
        trade_log_path=log,
    )
    reconciler._co_apply_enabled = False
    reconciler._engine._risk = SimpleNamespace(positions={"KXT": 2.34})
    reconciler._notifier = SimpleNamespace(send_alert=lambda msg, level=None: None)

    fills = _multi_hedge_fills()
    fills[0]["side"] = None

    with pytest.raises(AttributeError):
        await reconciler._handle_shakedown_cash_out(
            order_id="OID-P",
            kalshi_fills=fills,
            position_key="KXT",
            market_ticker="KXT",
            tracked_cash_out_oids=set(),
        )
