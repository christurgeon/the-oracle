"""Tests for scripts/fetch_shadow_settlements.compute_pnl."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from python.analytics.strategy_attribution import build_strategy_index
from scripts.fetch_shadow_settlements import (
    _entry_price,
    apply_metadata_enrichment,
    apply_strategy_enrichment,
    build_result_row,
    compute_fee_charge_overrides,
    compute_pnl,
    load_settlement_store,
    merge_settlements,
    print_analysis,
)


def _settled(ticker: str, result: str = "yes") -> dict:
    return {"ticker": ticker, "status": "finalized", "result": result}


def _unsettled(ticker: str) -> dict:
    return {"ticker": ticker, "status": "active", "result": None}


class TestMergeSettlements:
    """Kalshi 404s markets older than ~70 days, so a settled outcome must
    survive a refresh that can no longer see it."""

    def test_ticker_absent_from_fresh_keeps_cached_outcome(self) -> None:
        cached = {"OLD": _settled("OLD", "no")}
        assert merge_settlements(cached, {})["OLD"]["result"] == "no"

    def test_settled_cache_wins_over_unsettled_fetch(self) -> None:
        cached = {"OLD": _settled("OLD", "yes")}
        merged = merge_settlements(cached, {"OLD": _unsettled("OLD")})
        assert merged["OLD"]["result"] == "yes"

    def test_fresh_settlement_replaces_unsettled_cache(self) -> None:
        cached = {"T": _unsettled("T")}
        merged = merge_settlements(cached, {"T": _settled("T", "no")})
        assert merged["T"]["result"] == "no"

    def test_fresh_data_refreshes_an_open_market(self) -> None:
        cached = {"T": {"ticker": "T", "status": "active", "result": None, "volume": 1}}
        merged = merge_settlements(cached, {"T": {"ticker": "T", "status": "closed", "volume": 9}})
        assert merged["T"]["status"] == "closed"
        assert merged["T"]["volume"] == 9

    def test_new_tickers_are_added(self) -> None:
        merged = merge_settlements({"A": _settled("A")}, {"B": _settled("B")})
        assert set(merged) == {"A", "B"}

    def test_inputs_are_not_mutated(self) -> None:
        cached = {"A": _settled("A")}
        fresh = {"B": _settled("B")}
        merge_settlements(cached, fresh)
        assert set(cached) == {"A"}
        assert set(fresh) == {"B"}


class TestLoadSettlementStore:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_settlement_store(tmp_path / "nope.json") == {}

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "store.json"
        path.write_text("{not json")
        assert load_settlement_store(path) == {}

    def test_non_object_payload_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "store.json"
        path.write_text("[1, 2, 3]")
        assert load_settlement_store(path) == {}

    def test_non_dict_entries_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "store.json"
        path.write_text('{"A": {"result": "yes"}, "B": "garbage"}')
        assert load_settlement_store(path) == {"A": {"result": "yes"}}


def test_metadata_enrichment_backfills_city_var_forecast_from_ticker() -> None:
    """A reconciler synthetic row with a thin metadata dict gets city/var/
    forecast_date re-derived from the Kalshi ticker."""
    results = [
        {"market_ticker": "KXLOWTLAX-26MAY30-B56.5", "metadata": {"strategy": "weather_edge"}},
    ]
    apply_metadata_enrichment(results)
    meta = results[0]["metadata"]
    assert meta["city"] == "LAX"
    assert meta["market_type"] == "low"
    assert meta["forecast_date"] == "2026-05-30"
    assert meta["strategy"] == "weather_edge"  # pre-existing field preserved


def test_metadata_enrichment_never_overwrites_existing_attribution() -> None:
    """Real signal rows keep their own metadata even if the ticker would differ."""
    results = [
        {
            "market_ticker": "KXHIGHNY-26MAY31-B74.5",
            "metadata": {"city": "NYC", "market_type": "high", "forecast_date": "2026-05-31"},
        },
    ]
    apply_metadata_enrichment(results)
    meta = results[0]["metadata"]
    assert (meta["city"], meta["market_type"], meta["forecast_date"]) == (
        "NYC",
        "high",
        "2026-05-31",
    )


def test_metadata_enrichment_tolerates_missing_or_garbage_metadata() -> None:
    """No metadata dict, or an unparseable ticker, must not raise."""
    results = [
        {"market_ticker": "KXLOWTMIA-26MAY31-B76.5"},  # metadata key absent
        {"market_ticker": "GARBAGE", "metadata": {}},  # unknown series
    ]
    apply_metadata_enrichment(results)
    assert results[0]["metadata"]["city"] == "MIA"
    assert "city" not in results[1]["metadata"]  # garbage ticker fabricates nothing


def test_enrichment_recovers_strategy_from_placement_row() -> None:
    """A reconciler fill (metadata.strategy=unknown) whose order_id has a
    favorite_bias placement row is re-tagged favorite_bias in the result rows."""
    all_rows = [
        {"event": "placement", "order_id": "o1", "strategy": "favorite_bias"},
        {
            "event": "fill",
            "order_id": "o1",
            "metadata": {"strategy": "unknown", "reconcile_source": "fills_reconciler"},
        },
        {"event": "fill", "order_id": "o2", "strategy": "weather_edge"},
        {"event": "fill", "order_id": "o3", "strategy": None},  # legacy untagged
    ]
    index = build_strategy_index(all_rows)
    results = [
        {"order_id": "o1", "settled": True},  # reconciler fill, own strategy missing
        {"order_id": "o2", "strategy": "weather_edge", "settled": True},
        {"order_id": "o3", "strategy": None, "settled": True},
    ]
    apply_strategy_enrichment(results, index)
    assert results[0]["strategy"] == "favorite_bias"  # recovered from placement
    assert results[1]["strategy"] == "weather_edge"  # unchanged
    assert results[2]["strategy"] == "unknown"  # legacy stays unknown


class TestComputeFeeChargeOverrides:
    """engine.py stamps the order's CUMULATIVE fee on every partial-fill log
    row, so compute_pnl (called once per row) double-counts it. Zero the
    charge on every row but the LAST-BY-TIMESTAMP one — matching
    aggregate_local_orders' existing convention (pull_post_mortem_data.py) —
    without touching the rows themselves."""

    def test_two_fills_same_cumulative_value_charges_the_last_row_only(self) -> None:
        rows = [
            {"order_id": "o1", "timestamp": "2026-01-01T00:00:00+00:00", "fees_from_api": 0.2775},
            {"order_id": "o1", "timestamp": "2026-01-01T00:00:05+00:00", "fees_from_api": 0.2775},
        ]
        overrides = compute_fee_charge_overrides(rows)
        assert overrides == {0: 0.0}

    def test_a_later_downward_fee_correction_still_wins_over_an_earlier_larger_value(self) -> None:
        """fills_reconciler.py can legitimately DECREASE fees_from_api on a partial
        cash-out. last-by-timestamp must win even though it's numerically smaller
        than an earlier row — max() would get this backwards."""
        rows = [
            {"order_id": "o1", "timestamp": "2026-01-01T00:00:00+00:00", "fees_from_api": 0.25},
            {"order_id": "o1", "timestamp": "2026-01-01T00:05:00+00:00", "fees_from_api": 0.20},
        ]
        overrides = compute_fee_charge_overrides(rows)
        assert overrides == {0: 0.0}  # row 1 (later, smaller) keeps its charge

    def test_single_fill_order_has_no_override(self) -> None:
        rows = [{"order_id": "o1", "timestamp": "t", "fees_from_api": 0.15}]
        assert compute_fee_charge_overrides(rows) == {}

    def test_different_orders_are_independent(self) -> None:
        rows = [
            {"order_id": "o1", "timestamp": "t0", "fees_from_api": 0.10},
            {"order_id": "o1", "timestamp": "t1", "fees_from_api": 0.20},
            {"order_id": "o2", "timestamp": "t0", "fees_from_api": 0.05},
        ]
        assert compute_fee_charge_overrides(rows) == {0: 0.0}

    def test_legacy_rows_without_order_id_or_fee_produce_no_override(self) -> None:
        rows = [
            {"order_id": None, "timestamp": "t", "fees_from_api": None},
            {"timestamp": "t", "fees_from_api": None},
        ]
        assert compute_fee_charge_overrides(rows) == {}

    def test_input_rows_are_never_mutated(self) -> None:
        rows = [
            {"order_id": "o1", "timestamp": "t0", "fees_from_api": 0.10},
            {"order_id": "o1", "timestamp": "t1", "fees_from_api": 0.20},
        ]
        original = [dict(r) for r in rows]
        compute_fee_charge_overrides(rows)
        assert rows == original


def test_compute_pnl_prefers_fees_from_api_when_present():
    """T12: trade row with fees_from_api=0.10 → fee=0.10, not recomputed."""
    pnl, fee, won = compute_pnl(
        side="YES",
        order_price=0.50,
        contracts=10,
        market_result="yes",
        fees_from_api=0.10,
    )
    assert fee == pytest.approx(0.10)
    assert won is True
    assert pnl == pytest.approx(10 * (1.0 - 0.50) - 0.10)


def test_compute_pnl_falls_back_to_recompute_when_field_absent():
    """T13: legacy row (fees_from_api=None) → recomputes via order_fee default (taker)."""
    pnl, fee, won = compute_pnl(
        side="YES",
        order_price=0.50,
        contracts=10,
        market_result="yes",
        fees_from_api=None,
    )
    # Default taker, non-index: ceil(0.07 * 10 * 0.5 * 0.5 * 100) / 100 = 0.18
    assert fee == pytest.approx(0.18)


def test_compute_pnl_authoritative_zero_fee():
    """T14 (M4 corrected): fees_from_api=0.0 → fee=0.0 (NOT recompute) — authoritative zero.

    Shadow trades pay zero fees; the gate is `is not None`, not `> 0.0`.
    Earlier draft used `> 0.0` which would have wrongly recomputed shadow trades.
    """
    pnl, fee, won = compute_pnl(
        side="YES",
        order_price=0.50,
        contracts=10,
        market_result="yes",
        fees_from_api=0.0,
    )
    assert fee == 0.0


def test_compute_pnl_index_market_fallback():
    """T14b: legacy row + is_index_market=True → recompute uses index multiplier."""
    pnl, fee, won = compute_pnl(
        side="YES",
        order_price=0.50,
        contracts=10,
        market_result="yes",
        fees_from_api=None,
        is_index_market=True,
    )
    # Index taker: ceil(0.035 * 10 * 0.5 * 0.5 * 100) / 100 = ceil(8.75)/100 = 0.09
    assert fee == pytest.approx(0.09)


def test_compute_pnl_no_side_loss():
    """Sanity: NO bet loses when market resolves YES; fee deducted."""
    pnl, fee, won = compute_pnl(
        side="NO",
        order_price=0.40,  # NO side price (1 - yes_price)
        contracts=10,
        market_result="yes",
        fees_from_api=0.05,
    )
    assert won is False
    assert pnl == pytest.approx(-(0.40 * 10) - 0.05)


def test_settled_row_carries_settlement_source_from_the_store_entry() -> None:
    """A settlement entry with 'source': 'nws_cli_backfill' (written by
    backfill_settlements_from_snapshots.py) must be visible on the resulting
    .full.json row, not silently collapsed to indistinguishable-from-Kalshi.

    Exercises the real production code path (build_result_row), not a
    hand-reconstructed row dict.
    """
    settlement = {"result": "yes", "strike_type": "greater", "source": "nws_cli_backfill"}
    trade = {"market_ticker": "T1", "side": "YES", "order_price": 0.5, "contracts": 5,
              "fees_from_api": 0.0}
    result_row = build_result_row(trade, settlement)
    assert result_row["settlement_source"] == "nws_cli_backfill"
    assert result_row["settled"] is True
    assert result_row["won"] is True


def test_settled_row_defaults_settlement_source_to_kalshi_when_absent() -> None:
    """No 'source' key on the settlement entry means a real Kalshi fetch."""
    settlement = {"result": "yes", "strike_type": "greater"}
    trade = {"market_ticker": "T1", "side": "YES", "order_price": 0.5, "contracts": 5,
              "fees_from_api": 0.0}
    result_row = build_result_row(trade, settlement)
    assert result_row["settlement_source"] == "kalshi"


def test_unsettled_row_has_no_settlement_source() -> None:
    trade = {"market_ticker": "T1", "side": "YES", "order_price": 0.5, "contracts": 5}
    result_row = build_result_row(trade, {"result": None})
    assert result_row["settled"] is False
    assert "settlement_source" not in result_row


def test_build_result_row_applies_a_fees_override_of_zero() -> None:
    """0.0 is a valid override (see compute_fee_charge_overrides) and must not
    be treated as falsy / missing."""
    settlement = {"result": "yes"}
    trade = {"market_ticker": "T1", "side": "YES", "order_price": 0.5, "contracts": 5,
              "fees_from_api": 0.28}
    result_row = build_result_row(trade, settlement, fees_override=0.0)
    assert result_row["fee"] == 0.0
    assert result_row["fees_from_api"] == 0.28  # persisted raw field untouched


# ---------------------------------------------------------------------------
# _entry_price helper — covers reconciler-backfilled rows (shipped 2026-05-07)
# ---------------------------------------------------------------------------


def test_entry_price_engine_emitted_row_uses_order_price():
    """Engine fills carry both fill_price and order_price; helper prefers
    fill_price when present (matches pre-existing line-487 behaviour)."""
    row = {
        "order_id": "abc",
        "market_ticker": "KXHIGHNY-26MAY10-T70",
        "order_price": 0.50,
        "fill_price": 0.49,
        "contracts": 10,
    }
    assert _entry_price(row) == pytest.approx(0.49)


def test_entry_price_engine_row_without_fill_price_falls_back():
    """Legacy / shadow rows have order_price but no fill_price."""
    row = {
        "order_id": "shadow-1",
        "market_ticker": "KXHIGHNY-26MAY10-T70",
        "order_price": 0.55,
        "contracts": 10,
    }
    assert _entry_price(row) == pytest.approx(0.55)


def test_entry_price_reconciler_row_uses_fill_price():
    """FillsReconciler-backfilled row: no order_price, only fill_price.

    Pre-fix this would crash with KeyError('order_price') downstream.
    """
    row = {
        "event": "fill",
        "order_id": "711ffe94-2ab8-5081-a446-018b29333d3e",
        "market_ticker": "KXLOWTMIA-26MAY08-B73.5",
        "side": "NO",
        "mode": "live",
        "fill_price": 0.61,
        "contracts": 400,
        "fees_from_api": 6.8,
        "metadata": {
            "strategy": "unknown",
            "reconciled": True,
            "reconcile_source": "fills_reconciler",
        },
    }
    assert _entry_price(row) == pytest.approx(0.61)


def test_entry_price_raises_when_neither_field_present():
    row = {"order_id": "broken", "market_ticker": "KXHIGHNY-26MAY10-T70"}
    with pytest.raises(KeyError, match="missing both fill_price and order_price"):
        _entry_price(row)


def test_entry_price_treats_none_as_absent():
    """Reconciler rows have signal-side fields explicitly set to None;
    a None ``order_price`` should not satisfy the resolver."""
    row = {
        "order_id": "x",
        "market_ticker": "KXHIGHNY-26MAY10-T70",
        "order_price": None,
        "fill_price": 0.42,
    }
    assert _entry_price(row) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# print_analysis — regression for the 2026-05-09 KeyError crash
# ---------------------------------------------------------------------------


def _settled_reconciler_row() -> dict:
    """Mirror a real reconciler-backfilled row from data/vps_trades.jsonl,
    augmented with the post-settlement fields print_analysis expects.

    The shape matches what the main loop in fetch_shadow_settlements builds
    after the fix: order_price is normalised from fill_price."""
    return {
        "event": "fill",
        "order_id": "711ffe94-2ab8-5081-a446-018b29333d3e",
        "market_ticker": "KXLOWTMIA-26MAY08-B73.5",
        "side": "NO",
        "mode": "live",
        "timestamp": "2026-05-07T14:42:23.706298Z",
        "fill_price": 0.61,
        "contracts": 400,
        "dollars": 244.0,
        "fees_from_api": 6.8,
        # Reconciler rows have these signal-side fields set to None.
        "model_probability": None,
        "market_price": None,
        "edge": None,
        "confidence": None,
        "exec_price": None,
        # Fields normalised / added by the main loop post-settlement:
        "order_price": 0.61,
        "market_result": "no",
        "strike_type": "between",
        "floor_strike": 73.0,
        "cap_strike": 74.0,
        "pnl": 156.00,
        "fee": 6.8,
        "won": True,
        "settled": True,
        "metadata": {
            "strategy": "unknown",
            "reconciled": True,
            "reconcile_source": "fills_reconciler",
        },
    }


def _settled_engine_row() -> dict:
    return {
        "order_id": "shadow-1",
        "market_ticker": "KXHIGHNY-26APR22-B57.5",
        "side": "NO",
        "mode": "shadow",
        "timestamp": "2026-04-21T14:16:29.820596Z",
        "order_price": 0.675,
        "fill_price": 0.68,
        "contracts": 367,
        "model_probability": 0.13,
        "market_price": 0.325,
        "edge": 0.19,
        "confidence": 0.6,
        "market_result": "no",
        "strike_type": "between",
        "pnl": 109.05,
        "fee": 5.0,
        "won": True,
        "settled": True,
    }


def test_print_analysis_runs_on_reconciler_only_rows():
    """Regression: prior to the fix, print_analysis crashed on
    KeyError('order_price') when iterating reconciler-backfilled rows.

    Pin the post-fix invariant: the function runs to completion and the
    expected sections are present in the output.
    """
    rows = [_settled_reconciler_row()]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_analysis(rows, rows)
    out = buf.getvalue()
    assert "SHADOW TRADING P&L ANALYSIS" in out
    assert "OVERALL" in out
    assert "BY CITY" in out
    assert "TOP 10 WINNERS" in out
    # The TOP-N section should not crash on None model_probability/market_price.
    assert "model=  n/a" in out
    assert "mkt=  n/a" in out


def test_print_analysis_mixed_engine_and_reconciler_rows():
    """The wagered-sum aggregation must include reconciler rows."""
    engine = _settled_engine_row()
    recon = _settled_reconciler_row()
    rows = [engine, recon]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_analysis(rows, rows)
    out = buf.getvalue()
    expected_wagered = (
        engine["order_price"] * engine["contracts"]
        + recon["order_price"] * recon["contracts"]
    )
    # Format matches f"${total_wagered:>10,.2f}" — search for the dollar value.
    assert f"{expected_wagered:,.2f}" in out
