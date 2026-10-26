"""Unit tests for the shared per-fill reduction in fills_reconciler.

Five apply paths used to hand-roll this loop with drift between copies.
These tests own the contract; the per-call-site characterization tests own
the behavior at each caller.
"""

from __future__ import annotations

import pytest

from python.execution.fills_reconciler import FillAggregate, _aggregate_fills


def _fill(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "side": "yes",
        "count_fp": 1,
        "yes_price_dollars": 0.50,
        "no_price_dollars": 0.50,
        "fee_cost": 0.0,
        "is_taker": True,
        "created_time": "2026-05-17T12:00:00Z",
    }
    base.update(kw)
    return base


def test_empty_list_returns_the_zero_aggregate() -> None:
    agg = _aggregate_fills([])

    assert agg == FillAggregate(
        contracts=0, dollars=0.0, fees=0.0,
        taker_contracts=0, maker_contracts=0, earliest_ts="", side="",
    )
    assert agg.vwap == 0.0


def test_single_yes_fill_reads_the_yes_price_column() -> None:
    agg = _aggregate_fills([_fill(count_fp=4, yes_price_dollars=0.65, fee_cost=0.04)])

    assert agg.contracts == 4
    assert agg.dollars == 2.60
    assert agg.fees == 0.04
    assert agg.vwap == 0.65
    assert agg.side == "YES"


def test_single_no_fill_reads_the_no_price_column() -> None:
    agg = _aggregate_fills(
        [_fill(side="no", count_fp=12, no_price_dollars=0.65, yes_price_dollars=0.35)]
    )

    assert agg.contracts == 12
    assert agg.vwap == 0.65
    assert agg.side == "NO"


def test_multi_fill_vwap_sums_dollars_and_fees() -> None:
    agg = _aggregate_fills([
        _fill(count_fp=3, yes_price_dollars=0.60, fee_cost=0.01),
        _fill(count_fp=5, yes_price_dollars=0.66, fee_cost=0.02),
        _fill(count_fp=2, yes_price_dollars=0.70, fee_cost=0.01),
    ])

    assert agg.contracts == 10
    assert agg.dollars == 6.50
    assert agg.vwap == 0.65
    assert round(agg.fees, 4) == 0.04


def test_taker_and_maker_are_sums_of_contracts_and_none_counts_to_neither() -> None:
    agg = _aggregate_fills([
        _fill(count_fp=20, is_taker=True),
        _fill(count_fp=4, is_taker=False),
        _fill(count_fp=7, is_taker=None),
    ])

    assert agg.taker_contracts == 20
    assert agg.maker_contracts == 4
    assert agg.contracts == 31


def test_earliest_ts_is_the_min_not_the_first() -> None:
    agg = _aggregate_fills([
        _fill(created_time="2026-05-17T12:00:01Z"),
        _fill(created_time="2026-05-17T12:00:00Z"),
        _fill(created_time="2026-05-17T12:00:02Z"),
    ])

    assert agg.earliest_ts == "2026-05-17T12:00:00Z"


def test_earliest_ts_ignores_empty_and_missing_timestamps() -> None:
    agg = _aggregate_fills([
        _fill(created_time=""),
        _fill(created_time="2026-05-17T12:00:05Z"),
    ])
    assert agg.earliest_ts == "2026-05-17T12:00:05Z"

    no_ts = _fill()
    no_ts.pop("created_time")
    assert _aggregate_fills([no_ts]).earliest_ts == ""


def test_side_comes_from_the_first_fill_only() -> None:
    agg = _aggregate_fills([_fill(side="no"), _fill(side="yes")])

    assert agg.side == "NO"


def test_absent_side_key_falls_back_to_the_no_price_column() -> None:
    """Unreachable for a real Kalshi row (side is required); this pins the
    fallback direction now that an explicit None side raises instead."""
    fill = _fill(count_fp=2, yes_price_dollars=0.90, no_price_dollars=0.10)
    fill.pop("side")

    agg = _aggregate_fills([fill])

    assert agg.dollars == 0.20
    assert agg.side == ""


def test_none_price_does_not_raise() -> None:
    agg = _aggregate_fills([_fill(yes_price_dollars=None, fee_cost=0.0)])

    assert agg.dollars == 0.0


def test_absent_count_fp_defaults_to_zero() -> None:
    """`.get("count_fp", 0)` only substitutes the default for an ABSENT key —
    an explicit None (test_none_count_fp_raises below) still raises."""
    fill = _fill(fee_cost=0.0)
    fill.pop("count_fp")

    agg = _aggregate_fills([fill])

    assert agg.contracts == 0


def test_none_count_fp_raises() -> None:
    """count_fp is the most P&L-load-bearing field on the row; this plan
    unifies every site toward site A's strict form (Design delta 1), never
    the lenient _contracts_from_count_fp form the other four sites used to
    share. Line 1421 in _apply_manual_close is the one deliberate exception —
    it keeps calling _contracts_from_count_fp directly (Finding 8)."""
    with pytest.raises(TypeError):
        _aggregate_fills([_fill(count_fp=None, fee_cost=0.0)])


def test_junk_count_fp_raises() -> None:
    with pytest.raises(ValueError):
        _aggregate_fills([_fill(count_fp="not-a-number", fee_cost=0.0)])


def test_junk_price_raises() -> None:
    """`x or 0` only substitutes for a falsy value (None/0/""); a non-numeric
    string is truthy and passes through to float(), which raises — unchanged
    from every one of the five sites today."""
    with pytest.raises(ValueError):
        _aggregate_fills([_fill(yes_price_dollars="junk", fee_cost=0.0)])


def test_none_side_raises() -> None:
    """side is a required Kalshi enum; an explicit None already raises at
    3 of 5 call sites today. Unifying to raise (not silently defaulting)
    is this plan's one behavior-hardening delta — see Design."""
    with pytest.raises(AttributeError):
        _aggregate_fills([_fill(side=None)])


def test_none_fee_cost_raises() -> None:
    """fee_cost already raises on an explicit None at all 5 sites today —
    there is no drift here and this plan does not widen it."""
    with pytest.raises(TypeError):
        _aggregate_fills([_fill(fee_cost=None)])


def test_junk_fee_cost_raises() -> None:
    with pytest.raises(ValueError):
        _aggregate_fills([_fill(fee_cost="junk")])


def test_numeric_strings_are_accepted_for_count_and_price() -> None:
    agg = _aggregate_fills(
        [_fill(side="no", count_fp="12", no_price_dollars="0.65", fee_cost=0.18)]
    )

    assert agg.contracts == 12
    assert agg.vwap == 0.65
    assert agg.fees == 0.18


def test_aggregate_is_frozen() -> None:
    import dataclasses

    agg = _aggregate_fills([_fill()])
    with pytest.raises(dataclasses.FrozenInstanceError):
        agg.contracts = 99  # type: ignore[misc]
