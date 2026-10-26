"""Tests for scripts/intraday_shadow_report.py."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from python.analytics.prediction_log import PredictionRecord
from python.common import Side
from scripts.intraday_shadow_report import (
    brier_score,
    build_report,
    evaluate_outcome,
    lookup_actual_temp,
    reliability_table,
    secondary_pnl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rec(
    *,
    market_ticker: str = "KXHIGHNY-26MAY20-T84",
    city: str = "NYC",
    daily_variable: str = "high",
    strike_type: str = "greater",
    floor_strike: float | None = 84.0,
    cap_strike: float | None = None,
    expiration_date: date | None = date(2026, 5, 20),
    calibrated_prob: float = 0.65,
    raw_ensemble_prob: float | None = 0.60,
    emos_prob_raw: float | None = 0.62,
    exec_price: float | None = 0.60,
    side: Side = Side.YES,
    strategy: str = "intraday_emos_high",
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
        strategy=strategy,
        market_ticker=market_ticker,
        side=side,
        city=city,
        daily_variable=daily_variable,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        expiration_date=expiration_date,
        raw_ensemble_prob=raw_ensemble_prob,
        calibrated_prob=calibrated_prob,
        market_price=0.58,
        exec_price=exec_price,
        confidence=0.8,
        emos_prob_raw=emos_prob_raw,
    )


def _make_settlements(entries: list[dict]) -> dict[tuple[str, str], dict]:
    """Build a settlements dict keyed by (city, observation_date_iso)."""
    result = {}
    for e in entries:
        result[(e["city"], e["observation_date"])] = e
    return result


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score_perfect() -> None:
    pairs = [(1.0, 1), (0.0, 0)]
    assert brier_score(pairs) == pytest.approx(0.0)


def test_brier_score_worst() -> None:
    pairs = [(1.0, 0), (0.0, 1)]
    assert brier_score(pairs) == pytest.approx(1.0)


def test_brier_score_uniform() -> None:
    # All predictions 0.5, all outcomes 1 → BS = 0.25
    pairs = [(0.5, 1)] * 4
    assert brier_score(pairs) == pytest.approx(0.25)


def test_brier_score_empty() -> None:
    assert brier_score([]) is None


def test_brier_score_mixed() -> None:
    pairs = [(0.8, 1), (0.2, 0), (0.6, 1), (0.4, 0)]
    # (0.04 + 0.04 + 0.16 + 0.16) / 4 = 0.10
    assert brier_score(pairs) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Reliability table (calibration bucketing)
# ---------------------------------------------------------------------------


def test_reliability_table_single_bucket() -> None:
    # All predictions in [0.6, 0.7), outcome mix: 3 hits, 1 miss
    pairs = [(0.65, 1), (0.63, 1), (0.67, 0), (0.64, 1)]
    table = reliability_table(pairs, n_bins=10)
    assert len(table) == 1
    row = table[0]
    assert row["bin"] == "60-70%"
    assert row["n"] == 4
    assert row["observed"] == pytest.approx(0.75)
    assert row["predicted"] == pytest.approx((0.65 + 0.63 + 0.67 + 0.64) / 4, rel=1e-4)
    # gap = predicted - observed
    expected_gap = row["predicted"] - 0.75
    assert row["gap"] == pytest.approx(expected_gap, abs=1e-4)


def test_reliability_table_multiple_buckets() -> None:
    pairs = [
        (0.15, 0),  # 10-20% bucket
        (0.15, 0),
        (0.75, 1),  # 70-80% bucket
        (0.75, 1),
        (0.75, 0),
    ]
    table = reliability_table(pairs, n_bins=10)
    assert len(table) == 2
    low_row = next(r for r in table if r["bin"] == "10-20%")
    high_row = next(r for r in table if r["bin"] == "70-80%")
    assert low_row["n"] == 2
    assert low_row["observed"] == pytest.approx(0.0)
    assert high_row["n"] == 3
    assert high_row["observed"] == pytest.approx(2 / 3, abs=1e-4)


def test_reliability_table_empty() -> None:
    assert reliability_table([]) == []


def test_reliability_table_top_bin_includes_one() -> None:
    # prob=1.0 should land in the last bin (90-100%)
    pairs = [(1.0, 1), (0.95, 1)]
    table = reliability_table(pairs, n_bins=10)
    assert len(table) == 1
    assert table[0]["bin"] == "90-100%"
    assert table[0]["n"] == 2


# ---------------------------------------------------------------------------
# Settlement lookup / join
# ---------------------------------------------------------------------------


def test_lookup_actual_temp_found() -> None:
    settlements = _make_settlements([
        {"city": "NYC", "observation_date": "2026-05-20", "high_f": 87, "low_f": 65},
    ])
    assert lookup_actual_temp(settlements, "NYC", "high", "2026-05-20") == 87
    assert lookup_actual_temp(settlements, "NYC", "low", "2026-05-20") == 65


def test_lookup_actual_temp_not_found() -> None:
    settlements = _make_settlements([
        {"city": "NYC", "observation_date": "2026-05-20", "high_f": 87, "low_f": 65},
    ])
    assert lookup_actual_temp(settlements, "CHI", "high", "2026-05-20") is None
    assert lookup_actual_temp(settlements, "NYC", "high", "2026-05-21") is None


def test_lookup_actual_temp_unknown_variable() -> None:
    settlements = _make_settlements([
        {"city": "NYC", "observation_date": "2026-05-20", "high_f": 87, "low_f": 65},
    ])
    assert lookup_actual_temp(settlements, "NYC", "precipitation", "2026-05-20") is None


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------


def test_evaluate_outcome_greater_hit() -> None:
    rec = _make_rec(strike_type="greater", floor_strike=84.0)
    assert evaluate_outcome(rec, 85.0) == 1


def test_evaluate_outcome_greater_miss() -> None:
    rec = _make_rec(strike_type="greater", floor_strike=84.0)
    assert evaluate_outcome(rec, 84.0) == 0  # strict >


def test_evaluate_outcome_less_hit() -> None:
    rec = _make_rec(strike_type="less", floor_strike=None, cap_strike=80.0)
    assert evaluate_outcome(rec, 79.0) == 1


def test_evaluate_outcome_less_miss() -> None:
    rec = _make_rec(strike_type="less", floor_strike=None, cap_strike=80.0)
    assert evaluate_outcome(rec, 80.0) == 0


def test_evaluate_outcome_between_hit_at_boundary() -> None:
    rec = _make_rec(strike_type="between", floor_strike=80.0, cap_strike=84.0)
    # CLOSED interval: floor <= t <= cap
    assert evaluate_outcome(rec, 80.0) == 1
    assert evaluate_outcome(rec, 84.0) == 1
    assert evaluate_outcome(rec, 82.0) == 1


def test_evaluate_outcome_between_miss() -> None:
    rec = _make_rec(strike_type="between", floor_strike=80.0, cap_strike=84.0)
    assert evaluate_outcome(rec, 79.0) == 0
    assert evaluate_outcome(rec, 85.0) == 0


def test_evaluate_outcome_none_for_bad_params() -> None:
    rec = _make_rec(strike_type="greater", floor_strike=None)
    assert evaluate_outcome(rec, 85.0) is None


# ---------------------------------------------------------------------------
# Secondary P&L
# ---------------------------------------------------------------------------


def test_secondary_pnl_yes_win() -> None:
    rec = _make_rec(exec_price=0.60, side=Side.YES)
    outcomes = {rec.market_ticker: 1}
    result = secondary_pnl([rec], outcomes)
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["net_pnl"] == pytest.approx(0.40)  # 1 - 0.60
    assert result["wagered"] == pytest.approx(0.60)


def test_secondary_pnl_yes_loss() -> None:
    rec = _make_rec(exec_price=0.60, side=Side.YES)
    outcomes = {rec.market_ticker: 0}
    result = secondary_pnl([rec], outcomes)
    assert result["wins"] == 0
    assert result["losses"] == 1
    assert result["net_pnl"] == pytest.approx(-0.60)


def test_secondary_pnl_no_win() -> None:
    # Side=NO, outcome_yes=0 → our NO side won
    rec = _make_rec(exec_price=0.35, side=Side.NO)
    outcomes = {rec.market_ticker: 0}
    result = secondary_pnl([rec], outcomes)
    assert result["wins"] == 1
    assert result["net_pnl"] == pytest.approx(0.65)  # 1 - 0.35


def test_secondary_pnl_empty() -> None:
    result = secondary_pnl([], {})
    assert result["n"] == 0
    assert result["win_rate_pct"] is None
    assert result["roi_pct"] is None
    assert result["label"] == "FILL-OPTIMISTIC, NON-DECISION-GRADE"


def test_secondary_pnl_skips_none_exec_price() -> None:
    rec = _make_rec(exec_price=None)
    outcomes = {rec.market_ticker: 1}
    result = secondary_pnl([rec], outcomes)
    assert result["n"] == 0


def test_secondary_pnl_roi() -> None:
    rec_win = _make_rec(
        market_ticker="KXHIGHNY-26MAY20-T84", exec_price=0.60, side=Side.YES
    )
    rec_loss = _make_rec(
        market_ticker="KXHIGHNY-26MAY20-T85", exec_price=0.40, side=Side.YES
    )
    outcomes = {rec_win.market_ticker: 1, rec_loss.market_ticker: 0}
    result = secondary_pnl([rec_win, rec_loss], outcomes)
    assert result["wins"] == 1
    assert result["losses"] == 1
    expected_pnl = (1 - 0.60) + (-0.40)  # 0.40 - 0.40 = 0.0
    assert result["net_pnl"] == pytest.approx(expected_pnl, abs=1e-6)
    assert result["wagered"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: build_report with synthetic data
# ---------------------------------------------------------------------------


def _write_predictions_jsonl(path: Path, recs: list[PredictionRecord]) -> None:
    """Serialize PredictionRecords to a predictions.jsonl file."""
    import json as _json

    from python.analytics.prediction_log import _serialize

    lines = []
    for rec in recs:
        lines.append(_json.dumps(_serialize(rec), default=str, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n")


def _write_settlements_json(snap_dir: Path, entries: list[dict]) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "settlements.json").write_text(json.dumps(entries))


def test_build_report_settled_and_unsettled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pred_path = tmp / "predictions.jsonl"
        snaps_dir = tmp / "snapshots"

        settled_rec = _make_rec(
            market_ticker="KXHIGHNY-26MAY20-T84",
            city="NYC",
            daily_variable="high",
            expiration_date=date(2026, 5, 20),
            strike_type="greater",
            floor_strike=84.0,
            calibrated_prob=0.70,
        )
        unsettled_rec = _make_rec(
            market_ticker="KXHIGHNY-26MAY25-T84",
            city="NYC",
            daily_variable="high",
            expiration_date=date(2026, 5, 25),
            strike_type="greater",
            floor_strike=84.0,
            calibrated_prob=0.60,
        )
        other_strategy_rec = _make_rec(
            market_ticker="KXHIGHNY-26MAY20-T80",
            city="NYC",
            daily_variable="high",
            expiration_date=date(2026, 5, 20),
            strategy="weather_edge",  # should be filtered out
        )

        _write_predictions_jsonl(pred_path, [settled_rec, unsettled_rec, other_strategy_rec])

        # Only provide settlement for 2026-05-20 (NYC, high=85 > 84 → outcome=1)
        day_dir = snaps_dir / "2026-05-20"
        _write_settlements_json(
            day_dir,
            [
                {
                    "city": "NYC",
                    "observation_date": "2026-05-20",
                    "high_f": 85,
                    "low_f": 60,
                    "is_preliminary": False,
                    "issued_at": "2026-05-21T06:00:00+00:00",
                    "product_id": "abc123",
                }
            ],
        )

        report = build_report(pred_path, snaps_dir)

        assert report["summary"]["n_total"] == 2  # other_strategy filtered
        assert report["summary"]["n_settled"] == 1
        assert report["summary"]["n_unsettled"] == 1
        assert report["summary"]["brier_score_calibrated"] is not None
        # (0.70 - 1)^2 = 0.09
        assert report["summary"]["brier_score_calibrated"] == pytest.approx(0.09, abs=1e-6)

        # Reliability table should have one bin populated (0.70 → "70-80%")
        assert len(report["reliability"]) == 1
        rel_row = report["reliability"][0]
        assert rel_row["bin"] == "70-80%"
        assert rel_row["observed"] == pytest.approx(1.0)

        # by_bracket should have one key: "greater"
        assert "greater" in report["by_bracket"]
        assert report["by_bracket"]["greater"]["n"] == 1

        # by_city should have NYC
        assert "NYC" in report["by_city"]
        assert report["by_city"]["NYC"]["n"] == 1

        # secondary_pnl: settled_rec has exec_price=0.60, side=YES, outcome=1
        pnl = report["secondary_pnl"]
        assert pnl["n"] == 1
        assert pnl["wins"] == 1
        assert pnl["net_pnl"] == pytest.approx(0.40, abs=1e-6)


def test_build_report_empty_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pred_path = tmp / "predictions.jsonl"
        pred_path.write_text("")
        snaps_dir = tmp / "snapshots"
        snaps_dir.mkdir()

        report = build_report(pred_path, snaps_dir)
        assert report["summary"]["n_total"] == 0
        assert report["summary"]["n_settled"] == 0
        assert report["summary"]["brier_score_calibrated"] is None
        assert report["reliability"] == []
        assert report["secondary_pnl"]["n"] == 0
