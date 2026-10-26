"""Tests for python.analytics.attribution."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from python.analytics.attribution import (
    AttributionEngine,
    attribute_many,
    attribute_trade,
    by_city,
    by_side,
    summarize,
)
from python.common import Side


def _record(**overrides: Any) -> dict[str, Any]:
    """Build a settled shadow-trade record mirroring shadow_settlements.full.json."""
    rec: dict[str, Any] = {
        "order_id": "shadow-001",
        "timestamp": datetime(2026, 4, 10, 14, 0, tzinfo=UTC).isoformat(),
        "strategy": "weather_edge",
        "market_ticker": "KXHIGHNY-26APR10-T80",
        "side": "YES",
        "contracts": 100,
        "model_probability": 0.70,
        "order_price": 0.55,
        "fill_price": 0.55,
        "exec_price": 0.55,
        "mid_at_signal": 0.50,
        "pnl": 44.0,  # (1 - 0.55)*100 - fee(=1.0) == 44.0
        "fee": 1.0,
        "won": True,
        "settled": True,
        "metadata": {
            "city": "NYC",
            "market_type": "high",
            "ensemble_std_dev": 2.3,
            "isotonic_calibrated": True,
        },
    }
    rec.update(overrides)
    return rec


class TestAttributeTrade:
    def test_full_mode_decomposition_reconciles(self) -> None:
        rec = _record()
        a = attribute_trade(rec)
        assert a is not None
        assert a.decomposition_mode == "full"
        # Identity: model_error + predicted_alpha - slippage - fees == net_pnl
        total = a.model_error + a.predicted_alpha - a.slippage_cost - a.fees
        assert math.isclose(total, a.net_pnl, abs_tol=1e-6)
        assert math.isclose(a.net_pnl, rec["pnl"], abs_tol=1e-2)

    def test_degraded_mode_when_mid_missing(self) -> None:
        rec = _record(mid_at_signal=None)
        a = attribute_trade(rec)
        assert a is not None
        assert a.decomposition_mode == "degraded"
        assert math.isnan(a.predicted_alpha)
        assert math.isnan(a.slippage_cost)
        # Degraded identity still holds via predicted_alpha_at_exec
        total = a.model_error + a.predicted_alpha_at_exec - a.fees
        assert math.isclose(total, a.net_pnl, abs_tol=1e-6)

    def test_no_side_flips_probability(self) -> None:
        # NO side with model_probability_yes=0.3 means side_prob=0.7
        rec = _record(side="NO", model_probability=0.3, won=True, order_price=0.55)
        a = attribute_trade(rec)
        assert a is not None
        assert a.side == Side.NO
        assert math.isclose(a.side_prob, 0.70, abs_tol=1e-9)

    def test_unsettled_returns_none(self) -> None:
        rec = _record(settled=False)
        assert attribute_trade(rec) is None

    def test_missing_core_fields_returns_none(self) -> None:
        rec = _record()
        rec.pop("contracts")
        assert attribute_trade(rec) is None

    def test_city_inferred_from_ticker_when_metadata_missing(self) -> None:
        rec = _record(metadata={})
        a = attribute_trade(rec)
        assert a is not None
        assert a.city == "NYC"
        assert a.daily_variable == "high"


class TestRecordedOnlyMode:
    """Reconciler-backfilled synthetic rows lack model_probability/exec_price but
    carry authoritative recorded P&L — they must be attributed, not dropped."""

    def test_missing_model_probability_is_recorded_only_not_dropped(self) -> None:
        # Mirrors order 8db990c0 from the 2026-05-30 post-mortem: a settled
        # reconciler row with model_probability=None, exec_price=None.
        rec = _record(
            model_probability=None,
            exec_price=None,
            fill_price=None,
            order_price=None,
            mid_at_signal=None,
            dollars=3.6,
            contracts=20,
            pnl=-3.81,
            won=False,
        )
        a = attribute_trade(rec)
        assert a is not None  # NOT dropped
        assert a.decomposition_mode == "recorded_only"
        # net_pnl pinned to recorded → zero reconciliation drift.
        assert math.isclose(a.net_pnl, -3.81, abs_tol=1e-9)
        assert math.isclose(a.recorded_pnl, -3.81, abs_tol=1e-9)
        assert a.reconciliation_diff == 0.0
        assert a.side_outcome == 0
        # Decomposition fields are unavailable (NaN), not fabricated zeros.
        assert math.isnan(a.side_prob)
        assert math.isnan(a.model_error)
        assert math.isnan(a.predicted_alpha_at_exec)
        # side_exec recovered from dollars/contracts so wagered is still defined.
        assert math.isclose(a.side_exec, 0.18, abs_tol=1e-9)

    def test_summarize_is_nan_safe_with_recorded_only_rows(self) -> None:
        attrs = attribute_many(
            [
                _record(order_id="full", pnl=44.0, won=True),  # full mode
                _record(
                    order_id="rec",
                    model_probability=None,
                    exec_price=None,
                    fill_price=None,
                    order_price=None,
                    dollars=3.6,
                    contracts=20,
                    pnl=-3.81,
                    won=False,
                ),
            ]
        )
        assert len(attrs) == 2  # recorded-only row included
        s = summarize(attrs)
        assert s.n == 2
        assert s.wins == 1 and s.losses == 1
        # total_recorded_pnl is exact and NaN-free despite the recorded-only row.
        assert math.isclose(s.total_recorded_pnl, 44.0 - 3.81, abs_tol=1e-2)
        # Decomposition subtotals stay finite (recorded-only NaNs excluded).
        assert math.isfinite(s.total_model_error)
        assert math.isfinite(s.total_predicted_alpha_at_exec)
        assert math.isfinite(s.total_wagered)
        assert s.n_full_mode == 1
        assert s.n_degraded_mode == 1  # recorded-only counts as non-full

    def test_summary_tracks_recorded_only_subtotal(self) -> None:
        attrs = attribute_many(
            [
                _record(order_id="full", pnl=44.0, won=True),
                _record(
                    order_id="rec",
                    model_probability=None,
                    exec_price=None,
                    fill_price=None,
                    order_price=None,
                    dollars=3.6,
                    contracts=20,
                    pnl=-3.81,
                    won=False,
                ),
            ]
        )
        s = summarize(attrs)
        assert s.n_recorded_only == 1
        assert math.isclose(s.total_recorded_only_pnl, -3.81, abs_tol=1e-9)
        # The recorded headline == decomposable recorded P&L + recorded-only P&L,
        # so a report can close the decomposition block against the headline.
        decomposable_recorded = s.total_recorded_pnl - s.total_recorded_only_pnl
        assert math.isclose(decomposable_recorded, 44.0, abs_tol=1e-9)

    def test_recorded_roi_uses_realized_pnl(self) -> None:
        # exec_price diverges from the recorded fill → decomposition != recorded.
        rec = _record(
            order_id="diverge",
            model_probability=0.20,
            exec_price=0.03,
            fill_price=0.03,
            order_price=0.03,
            mid_at_signal=0.03,
            contracts=203,
            pnl=-2.03,  # actual fill was 0.01: 203*0.01
            won=False,
        )
        s = summarize(attribute_many([rec]))
        # Decomposed total diverges; recorded matches the persisted pnl exactly.
        assert math.isclose(s.total_recorded_pnl, -2.03, abs_tol=1e-9)
        assert abs(s.total_reconciliation_diff) > 0.01
        # roi_recorded_pct is computed off recorded P&L, not the decomposition.
        expected_roi = 100.0 * (-2.03) / s.total_wagered
        assert math.isclose(s.roi_recorded_pct, expected_roi, abs_tol=1e-6)


class TestSummarize:
    def test_empty_summary_has_zero_totals(self) -> None:
        s = summarize([])
        assert s.n == 0
        assert s.total_net_pnl == 0.0
        assert s.roi_pct == 0.0
        assert s.win_rate_pct == 0.0

    def test_multi_trade_rollup(self) -> None:
        attrs = attribute_many(
            [
                _record(order_id="a", pnl=44.0, won=True),
                _record(
                    order_id="b",
                    pnl=-56.0,  # (0 - 0.55)*100 - 1.0 = -56.0
                    won=False,
                ),
            ]
        )
        assert len(attrs) == 2
        s = summarize(attrs)
        assert s.n == 2
        assert s.wins == 1
        assert s.losses == 1
        assert math.isclose(s.total_net_pnl, -12.0, abs_tol=1e-2)


class TestGrouping:
    def test_by_side_and_by_city(self) -> None:
        attrs = attribute_many(
            [
                _record(order_id="a", side="YES"),
                _record(
                    order_id="b",
                    side="NO",
                    market_ticker="KXHIGHCHI-26APR10-T80",
                    metadata={"city": "CHI", "market_type": "high"},
                    model_probability=0.3,  # side_prob=0.7
                    order_price=0.55,
                    fill_price=0.55,
                    exec_price=0.55,
                    mid_at_signal=0.50,
                    pnl=44.0,
                    won=True,
                ),
            ]
        )
        bs = by_side(attrs)
        assert Side.YES in bs and Side.NO in bs
        bc = by_city(attrs)
        assert "NYC" in bc and "CHI" in bc


class TestEngine:
    def test_reads_and_filters_by_since(self, tmp_path: Path) -> None:
        settlements = [
            _record(
                order_id="old",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            ),
            _record(
                order_id="new",
                timestamp=datetime.now(UTC).isoformat(),
            ),
        ]
        p = tmp_path / "settlements.json"
        p.write_text(json.dumps(settlements))

        engine = AttributionEngine(str(p))
        all_attrs = engine.attributions()
        assert len(all_attrs) == 2

        recent = engine.attributions(since=datetime.now(UTC) - timedelta(days=1))
        assert len(recent) == 1
        assert recent[0].order_id == "new"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        engine = AttributionEngine(str(tmp_path / "does_not_exist.json"))
        assert engine.attributions() == []


class TestFillPriceIsAuthoritative:
    """``exec_price`` is the SIGNAL ASK, not the price paid.

    Regression for the 2026-09-01 finding: ``attribute_trade`` preferred
    ``exec_price`` over ``fill_price``, so any order that filled away from the
    signal-time ask decomposed against a price never paid. Measured on the live
    corpus, 409 of 707 fill rows (58%) disagreed by >0.5c, and every one of the
    ~40 "P&L reconciliation off" warnings per report run came from this.

    ``dollars/contracts`` and ``fill_price`` agreed on 707/707 rows where both
    were present, and ``exec_price`` was never the only price available.
    """

    def test_fill_price_beats_exec_price(self) -> None:
        # Real row: KXLOWTPHIL-26AUG19-T70, order 793df686. Limit placed at the
        # 0.59 ask, filled at 0.21. Settlement P&L independently confirms the
        # $1.47 cost basis: 7*(1-0.21) - 0.0813 == 5.4487.
        rec = _record(
            contracts=7,
            exec_price=0.59,
            fill_price=0.21,
            dollars=1.47,
            order_price=0.535,
            pnl=5.4487,
            fee=0.0813,
        )
        a = attribute_trade(rec)
        assert a is not None
        assert math.isclose(a.side_exec, 0.21, abs_tol=1e-9)

    def test_decomposition_reconciles_on_price_improved_fill(self) -> None:
        """The reconciliation gap this bug produced must be gone."""
        rec = _record(
            contracts=7,
            exec_price=0.59,
            fill_price=0.21,
            dollars=1.47,
            order_price=0.535,
            pnl=5.4487,
            fee=0.0813,
        )
        a = attribute_trade(rec)
        assert a is not None
        assert math.isclose(a.reconciliation_diff, 0.0, abs_tol=1e-2)

    def test_dollars_over_contracts_when_fill_price_absent(self) -> None:
        """Pre-2026-04-27 rows carry no ``fill_price`` but do carry dollars."""
        rec = _record(contracts=100, exec_price=0.55, fill_price=None, dollars=18.0)
        a = attribute_trade(rec)
        assert a is not None
        assert math.isclose(a.side_exec, 0.18, abs_tol=1e-9)

    def test_exec_price_alone_does_not_supply_a_price(self) -> None:
        """exec_price is wrong by construction; degrade honestly instead."""
        rec = _record(
            contracts=100, exec_price=0.55, fill_price=None, dollars=None, order_price=None
        )
        a = attribute_trade(rec)
        assert a is not None
        assert a.decomposition_mode != "full"
