"""End-to-end manual-close lifecycle test (spec §9.6).

Scenario: bot has an entry FILL on disk, user closes via Kalshi UI,
the next reconciler tick detects the sell and applies the close.
Verify the trade-log gets a manual_close row, exposure is freed on
the risk manager, the entry OrderRecord is stamped EXTERNALLY_CLOSED,
and a WARNING-level Telegram alert is queued.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import OrderStatus
from python.execution.cost_basis import cost_basis_for_position


@pytest.mark.asyncio
async def test_full_lifecycle_place_manual_close_reopen(tmp_path: Path) -> None:
    from python.execution.fills_reconciler import FillsReconciler

    log = tmp_path / "trades.jsonl"
    state = tmp_path / "fills_reconciler.json"
    shakedown = tmp_path / "shakedown_seen.json"

    # Seed an entry FILL row (bot already placed and the fill was logged earlier)
    log.write_text(
        json.dumps({
            "event": "fill", "order_id": "bot_buy",
            "position_key": "K", "market_ticker": "K",
            "side": "YES", "fill_price": 0.40, "contracts": 10,
            "fees_from_api": 0.05, "mode": "live",
            "metadata": {"strategy": "weather_edge"},
        }) + "\n"
    )

    kalshi = AsyncMock()
    # Manual sell appears on /portfolio/fills
    kalshi.get_fills = AsyncMock(return_value={
        "fills": [{
            "order_id": "manual_sell", "action": "sell", "side": "yes",
            "yes_price_dollars": 0.65, "no_price_dollars": 0.35,
            "count_fp": 10, "fee_cost": 0.04,
            "is_taker": True, "created_time": "2026-05-17T12:00:00Z",
            "market_ticker": "K",
        }],
        "cursor": None,
    })

    # Engine + risk
    engine = MagicMock()
    entry_record = MagicMock()
    entry_record.signal.position_key = "K"
    entry_record.signal.market_ticker = "K"
    entry_record.status = OrderStatus.FILLED
    engine._orders = {"bot_buy": entry_record}
    risk = MagicMock()
    risk.positions = {"K": 4.0}
    engine._risk = risk

    notifier = AsyncMock()

    rec = FillsReconciler(
        state_path=state,
        trade_log_path=log,
        trade_log_lock=asyncio.Lock(),
        kalshi_client=kalshi,
        engine=engine,
        notifier=notifier,
    )
    rec._manual_close_enabled = True
    rec._manual_close_apply_limit = 0
    rec._shakedown_seen_oids_path = shakedown

    await rec.tick()

    # 1. Trade-log has new manual_close row
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["fill", "manual_close"]
    mc = rows[1]
    assert mc["realized_pnl"] == pytest.approx(2.41)
    assert mc["metadata"]["strategy"] == "weather_edge"

    # 2. Risk manager released exposure + booked P&L
    risk.close_position.assert_called_once_with("K", pnl=pytest.approx(2.41))

    # 3. Entry OrderRecord stamped EXTERNALLY_CLOSED
    assert entry_record.status == OrderStatus.EXTERNALLY_CLOSED

    # 4. Telegram WARNING queued (level is keyword-only per Notifier protocol)
    notifier.send_alert.assert_called()
    from python.notifications.base import AlertLevel
    last = notifier.send_alert.call_args_list[-1]
    assert last.kwargs.get("level") == AlertLevel.WARNING
    msg = last.args[0] if last.args else last.kwargs["message"]
    assert "Manual close reconciled" in msg


# ---------------------------------------------------------------------------
# v1.1 §13 A: _settle_live defensive guard against cash-out race (Task 18).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settle_live_skips_when_position_already_popped_by_cashout(
    tmp_path: Path,
) -> None:
    """The backward race: reconciler popped position via cash-out apply
    mid-_settle_live. Without the guard, _settle_live would write a phantom
    settlement row AND double-book P&L. With the guard at the TOP of the
    per-position iteration, it skips entry build + row write + _settle_orders
    + close_position together.

    Scenario: two open positions (KXY, KXT). _settle_live builds its
    ticker_map snapshot, then iterates. The first iteration's
    ``await self._client.get_market(...)`` yields to a (simulated) concurrent
    reconciler tick whose _apply_cash_out pops KXT from risk._positions. The
    second iteration reaches KXT in ticker_map but the position is gone from
    risk.positions — that's the exact branch the guard protects.

    We simulate the concurrent reconciler tick deterministically by attaching
    a side_effect to the mocked ``get_market`` that pops KXT on the first
    call. The harness's full-wiring path provides a real ExecutionEngine
    in live mode against a mocked Kalshi client. A live-system test with
    actual asyncio interleaving between the reconciler task and the engine's
    settle loop is deferred to manual verification.
    """
    from python.common import OrderStatus
    from tests.conftest import E2EHarness

    harness = E2EHarness(tmp_path, full_wiring=True)
    # 1. Bot enters YES on KXT (the cash-out target) and a SECOND independent
    # YES on KXY (the iteration-1 settlement that yields to the reconciler).
    # Two positions are required for the multi-position interleave the guard
    # is designed to catch: iter-1's get_market await yields to a concurrent
    # reconciler tick which pops KXT (scheduled for iter-2). Iter-2's loop
    # body then runs with KXT already gone from risk.positions — that is the
    # exact branch the guard protects.
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    harness.seed_fill(
        market_ticker="KXY", side="yes", price=0.10, contracts=10, fees=0.05
    )
    harness.set_position("KXY", dollars=1.00)

    # 2. OrderRecords for both — KXY uses the harness helper, KXT must be
    # injected manually with a distinct order_id so they coexist in
    # engine._orders without overwriting.
    harness.add_fake_hydrated_record(
        market_ticker="KXY",
        position_key="KXY",
        side="YES",
        fill_price=0.10,
        contracts=10,
        fees=0.05,
    )
    # KXT record — handcrafted to coexist with KXY (different order_id key).
    from python.common import ExecutionMode, Side
    from python.execution.engine import OrderRecord
    from python.risk.manager import RiskDecision
    from python.strategies.base import TradeSignal

    kxt_signal = TradeSignal(
        market_id="KXT",
        market_ticker="KXT",
        model_probability=0.0,
        market_price=0.0,
        confidence=0.0,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=0.0,
        suggested_contracts=0,
        metadata={},
    )
    kxt_decision = RiskDecision(approved=True, signal=kxt_signal, adjusted_dollars=0.0)
    kxt_record = OrderRecord(
        order_id="BOT-KXT",
        signal=kxt_signal,
        risk_decision=kxt_decision,
        mode=ExecutionMode.LIVE,
        signal_price=0.0,
        order_price=0.0,
        fill_price=0.09,
        fill_count=26,
        requested_count=26,
        status=OrderStatus.FILLED,
        fees_from_api=0.10,
        fills_payload_received=True,
    )
    harness.engine._orders["BOT-KXT"] = kxt_record

    # 3. Capture pre-_settle_live bankroll (no cash-out apply yet —
    # the pop happens DURING _settle_live via the mid-await side_effect).
    bankroll_before_settle = harness.risk.bankroll

    # 4. Queue settlement for BOTH markets (both look settled to Kalshi).
    harness.queue_settlement(market_ticker="KXY", result="yes")
    harness.queue_settlement(market_ticker="KXT", result="yes")

    # 5. Rig get_market so iter-1 (whichever ticker runs first) pops KXT —
    # simulating a concurrent reconciler tick popping KXT mid-await. By the
    # time iter-2 reaches the guard, KXT is gone from risk.positions even
    # though it's in ticker_map.
    original_get_market = harness._kalshi_client.get_market.side_effect
    kxt_popped = {"done": False}

    async def _get_market_pops_kxt_on_first_call(ticker: str):
        # Simulate the concurrent reconciler tick by popping KXT during the
        # first get_market await — the analog of the backward race the spec
        # describes (v1.1 §13 A).
        if not kxt_popped["done"]:
            harness.risk._positions.pop("KXT", None)
            kxt_popped["done"] = True
        return await original_get_market(ticker)

    from unittest.mock import AsyncMock as _AsyncMock

    harness._kalshi_client.get_market = _AsyncMock(
        side_effect=_get_market_pops_kxt_on_first_call
    )

    # 6. Run _settle_live with the guard.
    rows_before = harness.read_trade_log_event_count("settlement")
    await harness.engine._settle_live()

    # 7. Assertions: guard skipped KXT entirely (no row, no double-book).
    rows_after = harness.read_trade_log_event_count("settlement")
    # KXY settled normally; KXT must NOT have a settlement row.
    new_rows = [
        json.loads(line)
        for line in harness.trade_log.read_text().splitlines()
        if line.strip()
    ]
    settlement_rows = [r for r in new_rows if r.get("event") == "settlement"]
    kxt_settlement_rows = [r for r in settlement_rows if r.get("market_ticker") == "KXT"]
    assert kxt_settlement_rows == [], (
        f"phantom settlement row written for KXT despite guard: "
        f"{kxt_settlement_rows!r}"
    )
    # Exactly one settlement row (the legitimate KXY one).
    assert rows_after == rows_before + 1, (
        f"expected exactly one KXY settlement row, got delta="
        f"{rows_after - rows_before}"
    )

    # Bankroll: KXY's legitimate settlement adds its P&L, but KXT must NOT.
    # KXY YES @ $0.10 × 10 contracts settled YES → payout 10 × $1.00 − 10 ×
    # $0.10 − fees(0.05) = 10 − 1 − 0.05 = +$8.95 (approx; the engine uses
    # binary_contract_pnl with the actual fee record).
    kxy_pnl = harness.risk.bankroll - bankroll_before_settle
    assert kxy_pnl == pytest.approx(8.95, abs=0.05), (
        f"KXY settlement P&L deviates from expected ~+$8.95: got {kxy_pnl:+.4f}. "
        f"A double-book on KXT would push this much higher."
    )


# ---------------------------------------------------------------------------
# v1.1 §11: full-lifecycle cash-out e2e (Task 25).
# Tests the full end-to-end path with real engine + risk + reconciler wired:
# entry FILL on disk → user cashes out via Kalshi UI → next tick applies.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_lifecycle_cash_out_e2e(make_e2e_harness):
    """Bot enters YES → user cashes out via Cash Out UI → reconciler applies.

    End-to-end with real engine + risk + reconciler wired (see Task 0 harness).
    """
    harness = make_e2e_harness()

    # 1. Seed bot's entry: YES @ $0.09 × 26 contracts, $0.10 fees.
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    # Register an OrderRecord so the classifier's open_positions_by_ticker_side
    # snapshot identifies (KXT, YES) as an open position — a BUY of NO against
    # this position is the BUY_CASH_OUT discriminator (spec §6 classifier
    # step 2).
    harness.add_fake_hydrated_record(
        market_ticker="KXT",
        position_key="KXT",
        side="YES",
        fill_price=0.09,
        contracts=26,
        fees=0.10,
    )
    prev_bankroll = harness.risk._bankroll

    # 2. Set cash-out apply enabled (post-canary state for the test).
    harness.set_cash_out_apply_enabled(True)
    harness.reconciler._co_apply_limit = 0  # unlimited for the test

    # 3. Queue Kalshi's cash-out hedge fill: NO @ $0.86 × 26.
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
        fee_cost=0.10,
    )

    # 4. Run one reconciler tick.
    tick_result = await harness.reconciler.tick()

    # 5. Verify classifier saw BUY_CASH_OUT and apply succeeded.
    assert tick_result.status in ("clean", "warning")
    log_rows = harness.read_trade_log_rows()
    cash_out_rows = [r for r in log_rows if r.get("event") == "cash_out"]
    assert len(cash_out_rows) == 1
    assert cash_out_rows[0]["close_type"] == "full"
    assert cash_out_rows[0]["contracts_offset"] == 26

    # 6. Verify risk state mutations.
    assert "KXT" not in harness.risk._positions, "full close must pop the position"
    # realized_pnl = 26 * 1.00 - 2.34 - 0.10 - (26 * 0.86) - 0.10 = 0.90
    expected_pnl = 26 * 1.00 - 2.34 - 0.10 - (26 * 0.86) - 0.10
    assert harness.risk._bankroll == pytest.approx(
        prev_bankroll + expected_pnl, abs=0.01
    )

    # 7. Verify OrderRecord stamped EXTERNALLY_CLOSED.
    for rec in harness.engine._orders.values():
        if getattr(rec, "signal", None) and rec.signal.position_key == "KXT":
            assert rec.status == OrderStatus.EXTERNALLY_CLOSED


@pytest.mark.asyncio
async def test_re_entry_opposite_side_after_full_cash_out(make_e2e_harness):
    """Spec §9 #4: After full cash-out, fresh strategy signal on opposite side
    within safety window → cash-out dedups via tracked_cash_out_oids; the
    re-entry classifies as BUY_ENTRY via placement-row match (NOT BUY_CASH_OUT
    despite the opposite-side history).
    """
    harness = make_e2e_harness()

    # Step 1: bot enters YES; user cashes out (full).
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    harness.add_fake_hydrated_record(
        market_ticker="KXT",
        position_key="KXT",
        side="YES",
        fill_price=0.09,
        contracts=26,
        fees=0.10,
    )
    harness.set_cash_out_apply_enabled(True)
    harness.reconciler._co_apply_limit = 0
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
    )
    await harness.reconciler.tick()
    assert "KXT" not in harness.risk._positions  # cash-out popped it

    # Step 2: bot's strategy generates an OPPOSITE-side entry (NO @ $0.20 × 10).
    # Simulate by writing a placement row first, then queuing the fill.
    harness.write_placement_row(
        market_ticker="KXT", order_id="BOT-2", strategy="weather_edge", action="buy"
    )
    # Also re-queue CO-1 to exercise the dedup gate (tracked_cash_out_oids
    # should exclude it from the candidate set so it does NOT re-apply).
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
    )
    harness.queue_kalshi_fill(
        market_ticker="KXT", side="no", price=0.20, contracts=10, order_id="BOT-2"
    )

    # Step 3: next reconciler tick re-scans safety window — sees BOTH the
    # cash-out hedge AND the new entry.
    await harness.reconciler.tick()

    # Assertion: BUY_CASH_OUT dedup'd; BOT-2 classified as BUY_ENTRY via
    # placement-row match (placement_index priority wins).
    log_rows = harness.read_trade_log_rows()
    bot_2_fills = [
        r
        for r in log_rows
        if r.get("order_id") == "BOT-2" and r.get("event") == "fill"
    ]
    assert len(bot_2_fills) == 1, "re-entry must produce ONE fill row (BUY_ENTRY)"
    # Cash-out remains a single row (dedup'd, NOT re-applied on tick 2).
    cash_out_rows = [r for r in log_rows if r.get("event") == "cash_out"]
    assert len(cash_out_rows) == 1, (
        "CO-1 must dedup via tracked_cash_out_oids on tick 2; "
        f"got {len(cash_out_rows)} cash_out rows"
    )


@pytest.mark.asyncio
async def test_crash_mid_full_close_apply_recovery(make_e2e_harness):
    """Spec §9 #14: simulate crash between _append_synthetic_rows and
    close_position (full-close path) — next tick dedups via
    tracked_cash_out_oids (row was written; state mutation didn't happen).
    Conservative-wrong: bot's local exposure overstates relative to Kalshi
    until next hydrate sweep. Verified no double-apply.

    Note (round-2 review MINOR-5): this test exercises the FULL-CLOSE crash
    path. The partial-close crash path (between row write and realize_partial)
    has the same invariant — row on disk, state didn't mutate, dedup catches
    re-attempt. A separate test for the partial path would be near-duplicate
    coverage; skipped for v1.1 (covered by the row-first-ordering invariant
    documented in spec §8 #9).
    """
    harness = make_e2e_harness()
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    harness.add_fake_hydrated_record(
        market_ticker="KXT",
        position_key="KXT",
        side="YES",
        fill_price=0.09,
        contracts=26,
        fees=0.10,
    )
    harness.set_cash_out_apply_enabled(True)
    harness.reconciler._co_apply_limit = 0
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
    )

    # Monkey-patch close_position to raise AFTER the row was written. The
    # cash-out path order is: append_synthetic_rows → risk.close_position →
    # record_strategy_trade → OrderRecord stamping (spec §6 Step 11). So a
    # raise from close_position simulates the "row on disk but state
    # mutation failed" crash. We use RuntimeError because the reconciler
    # tick path swallows exceptions inside the per-oid apply call only
    # narrowly (see _apply_fill_kind); CancelledError specifically can
    # propagate out of asyncio.Lock contexts in unexpected ways.
    original_close = harness.risk.close_position

    def crashing_close(*args, **kwargs):
        raise RuntimeError("simulated crash mid-apply")

    harness.risk.close_position = crashing_close

    # First tick: row gets written, then crash.
    try:
        await harness.reconciler.tick()
    except RuntimeError:
        pass

    # Verify the cash_out row IS on disk (durable dedup token).
    assert harness.read_trade_log_event_count("cash_out") == 1, (
        "crash-mid-apply must still leave the cash_out row on disk "
        "(durable dedup token written before state mutation)"
    )

    # Restore close_position for the recovery tick.
    harness.risk.close_position = original_close

    # Re-queue the same fill — the recovery tick should re-fetch it from the
    # safety window and the dedup gate should catch it (tracked_cash_out_oids
    # now contains CO-1 from the on-disk row).
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
    )

    # Second tick: the cash_out row is on disk → tracked_cash_out_oids
    # includes CO-1 → dedup catches the same hedge fill → no double-apply.
    await harness.reconciler.tick()
    # No new cash_out row (dedup).
    assert harness.read_trade_log_event_count("cash_out") == 1, (
        "crash recovery must not re-apply the cash-out"
    )


# ---------------------------------------------------------------------------
# v1.1 §11: partial-lifecycle cash-out e2e (Task 26).
# THE round-1 BLOCK-1 catch: verifies OrderRecord.fill_count is decremented
# so _settle_live doesn't double-book residual P&L on the full original
# contract count at settlement.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_lifecycle_cash_out_e2e(make_e2e_harness):
    """LAX 25-of-26 case: partial cash-out → residual settles correctly.

    This is THE round-1 BLOCK-1 catch: without the OrderRecord.fill_count
    decrement, _settle_orders would book P&L on the full 26 contracts at
    settlement, double-counting ~$22 on a $2.34 position.
    """
    harness = make_e2e_harness()

    # 1. Bot enters YES @ $0.09 × 26 (cost basis $2.34, $0.10 fees).
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    starting_bankroll = harness.risk._bankroll

    # 2. Register the OrderRecord so _decrement_fill_counts has a match AND
    # the classifier's open_positions_by_ticker_side snapshot identifies
    # (KXT, YES) as an open position (so BUY of NO classifies as cash-out).
    rec = harness.register_order_record(
        order_id="BOT-1",
        market_ticker="KXT",
        position_key="KXT",
        side="YES",
        fill_count=26,
        fill_price=0.09,
        fees=0.10,
        status=OrderStatus.FILLED,
    )

    # 3. User cashes out 25 of 26 via NO @ $0.86.
    harness.set_cash_out_apply_enabled(True)
    harness.reconciler._co_apply_limit = 0
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=25,
        order_id="CO-1",
        fee_cost=0.235,
    )
    await harness.reconciler.tick()

    # 4. APPLIED_PARTIAL_CASH_OUT verification.
    # offset_dollars = round(2.34 * 25/26, 4) = 2.25
    # offset_fees   = round(0.10 * 25/26, 4) = 0.0962
    # realized_pnl  = 25 - 2.25 - 0.0962 - (25 * 0.86) - 0.235 = +0.9188
    expected_offset_pnl = 25 * 1.00 - 2.25 - 0.0962 - (25 * 0.86) - 0.235  # ≈ +0.9188
    assert harness.risk._bankroll == pytest.approx(
        starting_bankroll + expected_offset_pnl, abs=0.01
    )
    assert harness.risk._daily_pnl == pytest.approx(expected_offset_pnl, abs=0.01)
    # Residual exposure remains.
    # partial_basis returns offset_dollars=2.25 → residual = 2.34 - 2.25 = 0.09
    assert harness.risk._positions["KXT"] == pytest.approx(0.09, abs=0.01)
    # fill_count decremented: 26 → 1.
    assert rec.fill_count == 1, (
        "BLOCK-1 regression: OrderRecord.fill_count not decremented"
    )
    # OrderRecord NOT stamped EXTERNALLY_CLOSED on partial.
    assert rec.status == OrderStatus.FILLED
    # Fees on the residual leg also scaled proportionally (sanity-check the
    # downstream settlement P&L assertion).
    # round(0.10 * (1/26), 4) = round(0.003846..., 4) = 0.0038
    assert rec.fees_from_api == pytest.approx(0.0038, abs=1e-4)

    # 5. Now simulate market settling YES. _settle_live's existing path
    # finds rec via the ticker_map, reads its (decremented) fill_count == 1,
    # and books residual P&L on 1 contract only — NOT 26.
    harness.queue_settlement(market_ticker="KXT", result="yes")
    bankroll_before_settle = harness.risk._bankroll
    await harness.engine._settle_live()

    # 6. THE CRITICAL ASSERTION (spec §11 partial-test #8): residual P&L is
    # booked on the DECREMENTED count. For 1 contract YES winning at
    # settlement: payout = (1.00 - 0.09) * 1 = $0.91; fees scaled to
    # 0.0038 → rec_pnl = 0.91 - 0.0038 = +0.9062.
    expected_residual_pnl = (1.00 - 0.09) * 1 - 0.0038  # ≈ +0.9062
    assert harness.risk._bankroll == pytest.approx(
        bankroll_before_settle + expected_residual_pnl, abs=0.02
    ), (
        "BLOCK-1 catch: _settle_live booked P&L on wrong fill_count "
        "(should be 1, not 26)"
    )

    # 7. Total realized = offset_pnl + residual_pnl ≈ $1.83 — matches the
    # LAX research doc's expected ledger.
    total_realized = harness.risk._bankroll - starting_bankroll
    assert total_realized == pytest.approx(
        expected_offset_pnl + expected_residual_pnl, abs=0.02
    )


# ---------------------------------------------------------------------------
# v1.1 §11: full reverted-cash-out lifecycle e2e (Task 32).
# Bot enters → reconciler applies cash-out → operator runs
# scripts/revert_cash_out.py --revert (subprocess) → simulated restart reads
# cost_basis off disk and the cash_out + cash_out_reverted offsets cancel,
# restoring the ORIGINAL basis. daily_pnl on disk reflects the revert.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reverted_cash_out_lifecycle(make_e2e_harness):
    """Full lifecycle: bot enters → cash-out apply → operator runs
    revert_cash_out.py --revert → simulate restart → cost_basis returns
    the ORIGINAL basis (cash_out + cash_out_reverted cancel) and the
    on-disk daily_pnl reflects the revert.
    """
    harness = make_e2e_harness()

    # 1. Bot enters YES @ $0.09 × 26.
    harness.seed_fill(
        market_ticker="KXT", side="yes", price=0.09, contracts=26, fees=0.10
    )
    harness.set_position("KXT", dollars=2.34)
    harness.add_fake_hydrated_record(
        market_ticker="KXT",
        position_key="KXT",
        side="YES",
        fill_price=0.09,
        contracts=26,
        fees=0.10,
    )
    starting_bankroll = harness.risk._bankroll
    starting_daily_pnl = harness.risk._daily_pnl

    # 2. Full cash-out applies.
    harness.set_cash_out_apply_enabled(True)
    harness.reconciler._co_apply_limit = 0
    harness.queue_cash_out_fill(
        market_ticker="KXT",
        hedge_side="no",
        price=0.86,
        contracts=26,
        order_id="CO-1",
        fee_cost=0.10,
    )
    await harness.reconciler.tick()
    # Cash-out applied; bankroll moved.
    assert harness.risk._bankroll != starting_bankroll
    cash_out_pnl = harness.risk._bankroll - starting_bankroll
    # Confirm the daily_pnl reflects the apply before we revert.
    assert harness.risk._daily_pnl == pytest.approx(
        starting_daily_pnl + cash_out_pnl, abs=0.01
    )

    # 3. Operator decides apply was wrong; runs revert_cash_out.py --revert.
    # Serialize the in-memory risk state to disk before invoking the script
    # (offline-only contract — the orchestrator is "stopped").
    state_dir = harness.persist_risk_state_to_disk()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/revert_cash_out.py",
            "--revert",
            "--order-id",
            "CO-1",
            "--yes-i-am-sure",
            "--state-dir",
            str(state_dir),
            "--reason",
            "test revert",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"revert failed (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )

    # 4. Verify trade log has both cash_out AND cash_out_reverted rows.
    trade_log = state_dir / "trades.jsonl"
    rows = [
        json.loads(line)
        for line in trade_log.read_text().splitlines()
        if line.strip()
    ]
    assert any(r.get("event") == "cash_out" for r in rows)
    assert any(r.get("event") == "cash_out_reverted" for r in rows)

    # 5. Simulate restart: read cost_basis from disk; the revert restores the
    # effective basis to original (cash_out subtracts, cash_out_reverted adds
    # back — net zero offset).
    cb = cost_basis_for_position(
        trade_log, position_key="KXT", market_ticker="KXT"
    )
    # Original: 26 contracts × $0.09 = $2.34, $0.10 fees.
    assert cb.total_dollars == pytest.approx(2.34, abs=0.01)
    assert cb.total_contracts == 26
    assert cb.total_fees == pytest.approx(0.10, abs=0.01)

    # 6. daily_pnl on disk reflects the revert (decreased by cash_out_pnl).
    new_risk_state = json.loads((state_dir / "risk_state.json").read_text())
    assert new_risk_state["daily_pnl"] == pytest.approx(
        starting_daily_pnl, abs=0.01
    )
    # Equivalently: post-revert daily_pnl == pre-revert daily_pnl − cash_out_pnl
    # (i.e., the cash-out's contribution was undone).
    pre_revert_daily_pnl = starting_daily_pnl + cash_out_pnl
    assert new_risk_state["daily_pnl"] == pytest.approx(
        pre_revert_daily_pnl - cash_out_pnl, abs=0.01
    )
