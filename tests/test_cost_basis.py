"""Tests for python/execution/cost_basis.py — manual-close cost-basis helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.execution.cost_basis import (
    CostBasis,
    PartialBasis,
    cost_basis_for_position,
    partial_basis,
)


def _write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_single_fill_cost_basis(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {
                "event": "fill",
                "order_id": "abc",
                "position_key": "KXHIGH-MIA-26MAY16-T80-YES",
                "market_ticker": "KXHIGH-MIA-26MAY16-T80",
                "side": "YES",
                "fill_price": 0.40,
                "contracts": 10,
                "fees_from_api": 0.05,
                "mode": "live",
                "metadata": {"strategy": "weather_edge"},
            }
        ],
    )
    cb = cost_basis_for_position(
        log,
        position_key="KXHIGH-MIA-26MAY16-T80-YES",
        market_ticker="KXHIGH-MIA-26MAY16-T80",
    )
    assert cb == CostBasis(
        total_dollars=4.0,
        total_contracts=10,
        total_fees=0.05,
        last_strategy="weather_edge",
        n_fills=1,
    )


def test_multi_fill_aggregation(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {
                "event": "fill",
                "order_id": "a",
                "position_key": "K-YES",
                "market_ticker": "K",
                "fill_price": 0.40,
                "contracts": 10,
                "fees_from_api": 0.05,
                "metadata": {"strategy": "weather_edge"},
            },
            {
                "event": "fill",
                "order_id": "b",
                "position_key": "K-YES",
                "market_ticker": "K",
                "fill_price": 0.50,
                "contracts": 5,
                "fees_from_api": 0.02,
                "metadata": {"strategy": "weather_edge"},
            },
        ],
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.total_dollars == pytest.approx(4.0 + 2.5)
    assert cb.total_contracts == 15
    assert cb.total_fees == pytest.approx(0.07)
    assert cb.n_fills == 2


def test_empty_ticker_returns_zero_basis(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    log.write_text("")
    cb = cost_basis_for_position(log, position_key="X-YES", market_ticker="X")
    assert cb == CostBasis(0.0, 0, 0.0, None, 0)


def test_missing_log_file_returns_zero_basis(tmp_path: Path) -> None:
    cb = cost_basis_for_position(
        tmp_path / "missing.jsonl", position_key="X-YES", market_ticker="X"
    )
    assert cb == CostBasis(0.0, 0, 0.0, None, 0)


def test_ignores_placement_settlement_and_manual_close_rows(tmp_path: Path) -> None:
    # CRITICAL — including manual_close double-counts as a buy
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {"event": "placement", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.99, "contracts": 999, "fees_from_api": 99.0},
            {"event": "settlement", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.99, "contracts": 999, "fees_from_api": 99.0},
            {"event": "manual_close", "position_key": "K-YES", "market_ticker": "K",
             "close_price": 0.99, "contracts_closed": 999, "exit_fees": 99.0},
            {"event": "fill", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05,
             "metadata": {"strategy": "weather_edge"}},
        ],
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.total_contracts == 10
    assert cb.total_dollars == pytest.approx(4.0)
    assert cb.total_fees == pytest.approx(0.05)
    assert cb.n_fills == 1


def test_fees_excluded_from_dollars_field(tmp_path: Path) -> None:
    # CRITICAL — collapsing fees into total_dollars causes P&L double-counting
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {"event": "fill", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.40, "contracts": 10, "fees_from_api": 1.23},
        ],
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.total_dollars == pytest.approx(4.0)
    assert cb.total_fees == pytest.approx(1.23)


def test_last_strategy_returned_from_most_recent_fill(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {"event": "fill", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.4, "contracts": 5, "fees_from_api": 0.0,
             "metadata": {"strategy": "weather_edge"}},
            {"event": "fill", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.5, "contracts": 5, "fees_from_api": 0.0,
             "metadata": {"strategy": "weather_edge_v2"}},
        ],
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.last_strategy == "weather_edge_v2"


def test_torn_jsonl_line_skipped(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    log.write_text(
        '{"event": "fill", "position_key": "K-YES", "market_ticker": "K", '
        '"fill_price": 0.4, "contracts": 5, "fees_from_api": 0.0, '
        '"metadata": {"strategy": "weather_edge"}}\n'
        '{"event": "fill", "position_key": "K-YES", '  # torn line — no closing brace
        '\n'
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.n_fills == 1
    assert cb.total_contracts == 5


def test_distinct_position_keys_for_same_ticker_not_aggregated(tmp_path: Path) -> None:
    # Multi-leg footgun guard: same ticker, two position_keys (YES + future NO leg)
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {"event": "fill", "position_key": "K-YES", "market_ticker": "K",
             "fill_price": 0.4, "contracts": 10, "fees_from_api": 0.05,
             "metadata": {"strategy": "weather_edge"}},
            {"event": "fill", "position_key": "K-NO", "market_ticker": "K",
             "fill_price": 0.7, "contracts": 20, "fees_from_api": 0.10,
             "metadata": {"strategy": "future_strategy"}},
        ],
    )
    cb_yes = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb_yes.total_contracts == 10
    assert cb_yes.total_dollars == pytest.approx(4.0)
    cb_no = cost_basis_for_position(log, position_key="K-NO", market_ticker="K")
    assert cb_no.total_contracts == 20
    assert cb_no.total_dollars == pytest.approx(14.0)


def test_legacy_row_without_position_key_matches_on_market_ticker(tmp_path: Path) -> None:
    # Fallback path for rows predating position_key schema
    log = tmp_path / "trades.jsonl"
    _write_log(
        log,
        [
            {"event": "fill", "market_ticker": "K",  # no position_key field
             "fill_price": 0.3, "contracts": 4, "fees_from_api": 0.01,
             "metadata": {"strategy": "weather_edge"}},
        ],
    )
    cb = cost_basis_for_position(log, position_key="K-YES", market_ticker="K")
    assert cb.n_fills == 1
    assert cb.total_contracts == 4


def _cb(dollars: float, contracts: int, fees: float = 0.0) -> CostBasis:
    return CostBasis(
        total_dollars=dollars,
        total_contracts=contracts,
        total_fees=fees,
        last_strategy="weather_edge",
        n_fills=1,
    )


def test_partial_basis_full_returns_total_and_zero_residual() -> None:
    pb = partial_basis(_cb(2.34, 26, 0.10), 26)
    assert pb == PartialBasis(2.34, 0.10, 0.0, 0.0)


def test_partial_basis_half() -> None:
    pb = partial_basis(_cb(2.00, 26, 0.10), 13)
    # offset = 2.00 * 13/26 = 1.00; residual = 2.00 - 1.00 = 1.00.
    assert pb.offset_dollars == 1.00
    assert pb.residual_dollars == 1.00


def test_partial_basis_lax_case() -> None:
    # LAX 25-of-26 from the research doc.
    pb = partial_basis(_cb(2.34, 26, 0.10), 25)
    # offset = round(2.34 * 25/26, 4) = round(2.25, 4) = 2.25.
    # residual = 2.34 - 2.25 = 0.09 (unrounded subtraction).
    assert pb.offset_dollars == 2.25
    assert pb.residual_dollars == pytest.approx(0.09, abs=1e-10)


def test_partial_basis_subtraction_trick_no_drift() -> None:
    # n=1, M=3, total=$1.00 — would drift under double-rounding;
    # the unrounded-subtraction discipline guarantees offset + residual
    # equals total to the cent.
    pb = partial_basis(_cb(1.00, 3), 1)
    assert pb.offset_dollars + pb.residual_dollars == pytest.approx(1.00)


def test_partial_basis_raises_on_zero_or_negative() -> None:
    with pytest.raises(ValueError):
        partial_basis(_cb(1.00, 3), 0)
    with pytest.raises(ValueError):
        partial_basis(_cb(1.00, 3), -1)


def test_partial_basis_raises_on_overflow() -> None:
    with pytest.raises(ValueError):
        partial_basis(_cb(1.00, 3), 4)


def test_partial_basis_sub_cent_total() -> None:
    # Sub-cent fixtures: offset rounds to 0; residual carries the whole basis.
    pb = partial_basis(_cb(0.0001, 3), 1)
    assert pb.offset_dollars == 0.0
    assert pb.residual_dollars == pytest.approx(0.0001, abs=1e-10)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_cost_basis_no_cash_out_rows_back_compat(tmp_path: Path) -> None:
    """No cash_out rows → identical behavior to v1."""
    log = tmp_path / "trades.jsonl"
    _write_rows(
        log,
        [
            {
                "event": "fill",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            }
        ],
    )
    cb = cost_basis_for_position(log, position_key="KX-TEST", market_ticker="KX-TEST")
    assert cb.total_dollars == pytest.approx(2.34)
    assert cb.total_contracts == 26
    assert cb.total_fees == pytest.approx(0.10)


def test_cost_basis_subtracts_cash_out_row(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    _write_rows(
        log,
        [
            {
                "event": "fill",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            },
            {
                "event": "cash_out",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "close_type": "partial",
                "contracts_offset": 25,
                "cost_basis_offset_dollars": 2.25,
                "cost_basis_offset_fees": 0.096,
            },
        ],
    )
    cb = cost_basis_for_position(log, position_key="KX-TEST", market_ticker="KX-TEST")
    # Effective: 2.34 - 2.25 = 0.09 dollars; 26 - 25 = 1 contract;
    # 0.10 - 0.096 = 0.004 fees.
    assert cb.total_dollars == pytest.approx(0.09)
    assert cb.total_contracts == 1
    assert cb.total_fees == pytest.approx(0.004)


def test_cost_basis_subtracts_cash_out_force_applied(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    _write_rows(
        log,
        [
            {
                "event": "fill",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            },
            {
                "event": "cash_out_force_applied",  # operator-forced, same accounting
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "close_type": "full",
                "contracts_offset": 26,
                "cost_basis_offset_dollars": 2.34,
                "cost_basis_offset_fees": 0.10,
            },
        ],
    )
    cb = cost_basis_for_position(log, position_key="KX-TEST", market_ticker="KX-TEST")
    assert cb.total_dollars == pytest.approx(0.0)
    assert cb.total_contracts == 0
    assert cb.total_fees == pytest.approx(0.0)


def test_cost_basis_reverted_adds_back(tmp_path: Path) -> None:
    """cash_out_reverted re-adds the offset that cash_out subtracted."""
    log = tmp_path / "trades.jsonl"
    _write_rows(
        log,
        [
            {
                "event": "fill",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "fill_price": 0.09,
                "contracts": 26,
                "fees_from_api": 0.10,
                "metadata": {"strategy": "weather_edge"},
            },
            {
                "event": "cash_out",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "close_type": "partial",
                "contracts_offset": 25,
                "cost_basis_offset_dollars": 2.25,
                "cost_basis_offset_fees": 0.096,
            },
            {
                "event": "cash_out_reverted",
                "position_key": "KX-TEST",
                "market_ticker": "KX-TEST",
                "close_type": "partial",
                "contracts_offset": 25,
                "cost_basis_offset_dollars": 2.25,
                "cost_basis_offset_fees": 0.096,
            },
        ],
    )
    cb = cost_basis_for_position(log, position_key="KX-TEST", market_ticker="KX-TEST")
    # cash_out + cash_out_reverted cancel: back to original.
    assert cb.total_dollars == pytest.approx(2.34)
    assert cb.total_contracts == 26
    assert cb.total_fees == pytest.approx(0.10)
