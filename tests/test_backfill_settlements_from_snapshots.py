"""Tests for scripts/backfill_settlements_from_snapshots.py."""

from __future__ import annotations

from python.analytics.strategy_attribution import build_strategy_index
from scripts.backfill_settlements_from_snapshots import (
    backfill_missing,
    checked_trades_by_month,
    find_disagreements,
    reconstruct_outcome,
    reconstructable_trades_by_month,
)
from scripts.fetch_shadow_settlements import apply_strategy_enrichment


def _trade(
    ticker: str, *, city: str | None = "NYC", var: str | None = "high",
    forecast_date: str | None = "2026-05-26", strike_type: str | None = "greater",
    floor_strike: float | None = 84.0, cap_strike: float | None = None,
    strategy: str = "favorite_bias",
) -> dict:
    meta = {}
    if city is not None:
        meta["city"] = city
    if var is not None:
        meta["market_type"] = var
    if forecast_date is not None:
        meta["forecast_date"] = forecast_date
    if strike_type is not None:
        meta["strike_type"] = strike_type
    meta["floor_strike"] = floor_strike
    meta["cap_strike"] = cap_strike
    return {"market_ticker": ticker, "metadata": meta, "strategy": strategy}


def test_reconstruct_outcome_derives_yes_from_snapshot_greater() -> None:
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    trade = _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "yes"
    assert entry["strike_type"] == "greater"
    assert entry["floor_strike"] == 84.0
    assert entry["source"] == "nws_cli_backfill"


def test_reconstruct_outcome_derives_no_from_snapshot_greater() -> None:
    settlements = {("NYC", "2026-05-26"): {"high_f": 82.0}}
    trade = _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "no"


def test_reconstruct_outcome_less_is_strict() -> None:
    settlements = {("NYC", "2026-05-26"): {"low_f": 60.0}}
    trade = _trade("KXLOWTNYC-26MAY26-T60", city="NYC", var="low",
                    strike_type="less", floor_strike=None, cap_strike=60.0)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "no"  # 60.0 is not < 60.0 (strict)


def test_reconstruct_outcome_between_is_closed_inclusive_on_the_cap() -> None:
    """Regression guard for the historical between-bracket bug (42/42 mislabeled):
    temp == cap_strike must resolve YES (CLOSED, right-inclusive)."""
    settlements = {("CHI", "2026-05-26"): {"high_f": 70.0}}
    trade = _trade("KXHIGHCHI-26MAY26-B70", city="CHI", var="high",
                    strike_type="between", floor_strike=65.0, cap_strike=70.0)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "yes"


def test_reconstruct_outcome_returns_none_when_no_snapshot_match() -> None:
    trade = _trade("KXHIGHNY-26MAY26-T84")
    entry = reconstruct_outcome(trade, settlements={})
    assert entry is None


def test_reconstruct_outcome_returns_none_when_strike_type_missing() -> None:
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    trade = _trade("KXHIGHNY-26MAY26-T84", strike_type=None)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is None


def test_reconstruct_outcome_falls_back_to_ticker_parsing_when_metadata_missing_city_and_date() -> (
    None
):
    # No city/forecast_date in metadata; must be recovered from the ticker itself.
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    trade = _trade("KXHIGHNY-26MAY26-T84", city=None, forecast_date=None)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "yes"


def test_reconstruct_outcome_uses_low_f_for_low_variable() -> None:
    settlements = {("CHI", "2026-05-26"): {"low_f": 60.0, "high_f": 90.0}}
    trade = _trade("KXLOWTCHI-26MAY26-T62", city="CHI", var="low", forecast_date="2026-05-26",
                   strike_type="greater", floor_strike=62.0)
    entry = reconstruct_outcome(trade, settlements)
    assert entry is not None
    assert entry["result"] == "no"  # 60 is not > 62


def test_backfill_missing_never_overwrites_an_already_settled_ticker() -> None:
    cached = {"T1": {"ticker": "T1", "result": "yes", "status": "finalized"}}
    settlements = {("NYC", "2026-05-26"): {"high_f": 10.0}}  # would derive "no" if consulted
    raw_trades = [_trade("T1")]
    merged, n_recon, n_skip = backfill_missing(cached, raw_trades, settlements)
    assert merged["T1"]["result"] == "yes"
    assert n_recon == 0


def test_backfill_missing_reconstructs_a_missing_ticker() -> None:
    cached: dict = {}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    raw_trades = [_trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)]
    merged, n_recon, n_skip = backfill_missing(cached, raw_trades, settlements)
    assert merged["KXHIGHNY-26MAY26-T84"]["result"] == "yes"
    assert n_recon == 1
    assert n_skip == 0


def test_backfill_missing_counts_unreconstructable_tickers_as_skipped() -> None:
    cached: dict = {}
    raw_trades = [_trade("KXHIGHNY-26MAY26-T84")]
    merged, n_recon, n_skip = backfill_missing(cached, raw_trades, settlements={})
    assert n_recon == 0
    assert n_skip == 1
    assert "KXHIGHNY-26MAY26-T84" not in merged


def test_backfill_missing_dedupes_multiple_fill_rows_for_the_same_ticker() -> None:
    cached: dict = {}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    raw_trades = [
        _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0),
        _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0),  # second fill row, same ticker
    ]
    merged, n_recon, n_skip = backfill_missing(cached, raw_trades, settlements)
    assert n_recon == 1  # counted once, not once per fill row


def test_backfill_missing_can_be_scoped_to_one_strategy() -> None:
    cached: dict = {}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    raw_trades = [
        _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0, strategy="favorite_bias"),
        _trade("KXHIGHNY-26MAY27-T84", floor_strike=84.0, strategy="weather_edge"),
    ]
    merged, n_recon, n_skip = backfill_missing(
        cached, raw_trades, settlements, strategies={"favorite_bias"}
    )
    assert n_recon == 1
    assert "KXHIGHNY-26MAY26-T84" in merged
    assert "KXHIGHNY-26MAY27-T84" not in merged


def test_backfill_missing_is_idempotent() -> None:
    cached: dict = {}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    raw_trades = [_trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)]
    once, _, _ = backfill_missing(cached, raw_trades, settlements)
    twice, n_recon2, _ = backfill_missing(once, raw_trades, settlements)
    assert once == twice
    assert n_recon2 == 0  # already settled on the second pass, nothing new to reconstruct


def test_find_disagreements_empty_when_derivation_agrees() -> None:
    cached = {"KXHIGHNY-26MAY26-T84": {"ticker": "KXHIGHNY-26MAY26-T84", "result": "yes"}}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    raw_trades = [_trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)]
    mismatches, n_checked = find_disagreements(cached, raw_trades, settlements)
    assert mismatches == []
    assert n_checked == 1


def test_find_disagreements_flags_a_mismatch() -> None:
    cached = {"KXHIGHNY-26MAY26-T84": {"ticker": "KXHIGHNY-26MAY26-T84", "result": "no"}}
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}  # derives "yes"
    raw_trades = [_trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0)]
    mismatches, n_checked = find_disagreements(cached, raw_trades, settlements)
    assert mismatches == [("KXHIGHNY-26MAY26-T84", "no", "yes")]
    assert n_checked == 1


def test_find_disagreements_skips_tickers_it_cannot_derive() -> None:
    cached = {"T1": {"ticker": "T1", "result": "yes"}}
    raw_trades = [_trade("T1")]
    mismatches, n_checked = find_disagreements(cached, raw_trades, settlements={})
    assert mismatches == []
    assert n_checked == 0


def test_month_stratification_buckets_checked_and_reconstructable_separately() -> None:
    """Mixed already-settled (checked) and missing (reconstructable) tickers
    across different months must land in independent, correctly-labeled
    month buckets — the coverage gap the agreement-check print exists to
    surface (see Finding 1: checked skews Jul-Sep, reconstructable skews
    May-Jun on the live corpus)."""
    settlements = {
        ("NYC", "2026-07-10"): {"high_f": 86.0},
        ("NYC", "2026-08-05"): {"high_f": 86.0},
        ("NYC", "2026-05-20"): {"high_f": 86.0},
        ("NYC", "2026-06-15"): {"high_f": 86.0},
        ("NYC", "2026-06-16"): {"high_f": 86.0},
    }
    cached = {
        "T-JUL": {"ticker": "T-JUL", "result": "yes", "status": "finalized"},
        "T-AUG": {"ticker": "T-AUG", "result": "yes", "status": "finalized"},
    }
    raw_trades = [
        _trade("T-JUL", forecast_date="2026-07-10"),
        _trade("T-AUG", forecast_date="2026-08-05"),
        _trade("T-MAY", forecast_date="2026-05-20"),
        _trade("T-JUN-1", forecast_date="2026-06-15"),
        _trade("T-JUN-2", forecast_date="2026-06-16"),
    ]

    assert checked_trades_by_month(cached, raw_trades, settlements) == {
        "2026-07": 1,
        "2026-08": 1,
    }
    assert reconstructable_trades_by_month(cached, raw_trades, settlements) == {
        "2026-05": 1,
        "2026-06": 2,
    }


def test_backfill_missing_scoping_requires_strategy_index_enrichment_for_reconciler_rows() -> (
    None
):
    """A reconciler-backfilled fill row's own top-level ``strategy`` can be
    missing/blank; the true strategy is only recoverable via the order_id
    join that ``build_strategy_index`` + ``apply_strategy_enrichment`` perform
    (the fix ``main()`` applies before scoping). Without that enrichment
    step, ``--strategy`` scoping would silently drop such a row."""
    settlements = {("NYC", "2026-05-26"): {"high_f": 86.0}}
    placement_row = {"event": "placement", "order_id": "O1", "strategy": "favorite_bias"}
    fill_row = _trade("KXHIGHNY-26MAY26-T84", floor_strike=84.0, strategy="")
    fill_row["order_id"] = "O1"

    # Unenriched: the row's own blank top-level strategy fails the scope filter.
    merged, n_recon, _ = backfill_missing(
        {}, [fill_row], settlements, strategies={"favorite_bias"}
    )
    assert n_recon == 0
    assert "KXHIGHNY-26MAY26-T84" not in merged

    # After enrichment (the fix): the row resolves to favorite_bias via order_id.
    strategy_index = build_strategy_index([placement_row, fill_row])
    apply_strategy_enrichment([fill_row], strategy_index)
    assert fill_row["strategy"] == "favorite_bias"

    merged2, n_recon2, _ = backfill_missing(
        {}, [fill_row], settlements, strategies={"favorite_bias"}
    )
    assert n_recon2 == 1
    assert "KXHIGHNY-26MAY26-T84" in merged2
