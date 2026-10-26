"""Tests for scripts/salvage_frame_v1_predictions.py (Stage 0a parent salvage).

Hermetic: builds a tiny predictions.jsonl + a synthetic ASOS CSV in tmp_path
with a boundary-hour observation that lands on different climate days under the
civil vs LST frames, and asserts the script flags the row frame_affected and
reports the low-p NO-bin contamination counts.

Run with: uv run pytest tests/test_salvage_frame_v1.py -v
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from python.analytics.prediction_log import PredictionRecord, _serialize
from python.common import Side
from scripts.salvage_frame_v1_predictions import salvage, summarize


def _asos_csv(path: Path, station: str, rows: list[tuple[str, float]]) -> None:
    lines = ["station,valid,tmpf"]
    lines += [f"{station},{ts},{t:.2f}" for ts, t in rows]
    path.write_text("\n".join(lines) + "\n")


def _pred(
    *,
    ticker: str,
    city: str,
    expiration: date,
    ts: datetime,
    asos_m: float,
    posterior: float,
    side: Side = Side.NO,
    frame_version: int = 1,
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=ts,
        strategy="intraday_asos_high",
        market_ticker=ticker,
        side=side,
        city=city,
        daily_variable="high",
        strike_type="between",
        floor_strike=88.0,
        cap_strike=90.0,
        expiration_date=expiration,
        raw_ensemble_prob=None,
        calibrated_prob=posterior,
        market_price=0.5,
        exec_price=0.9,
        confidence=1.0,
        posterior_p_raw=posterior,
        asos_M=asos_m,
        sigma_gain=0.6,
        guard_outcome="pass",
        frame_version=frame_version,
    )


def _write_preds(path: Path, recs: list[PredictionRecord]) -> None:
    lines = [json.dumps(_serialize(r), default=str, separators=(",", ":")) for r in recs]
    path.write_text("\n".join(lines) + "\n")


def test_boundary_hour_row_flagged_frame_affected(tmp_path: Path) -> None:
    """A NYC row whose M is held by the DST boundary hour is frame_affected.

    2026-07-15 04:51 UTC = 00:51 EDT (civil → climate day 07-15) but 23:51 EST
    (LST → climate day 07-14). If that boundary ob is the warmest, the civil
    frame includes it in 07-15's M while the LST frame does not → M differs.
    """
    asos_dir = tmp_path / "asos"
    asos_dir.mkdir()
    _asos_csv(
        asos_dir / "knyc_asos.csv",
        "NYC",
        [
            ("2026-07-15 04:51", 95.0),  # 00:51 EDT (civil 07-15) / 23:51 EST (LST 07-14)
            ("2026-07-15 14:51", 70.0),  # 10:51 EDT/09:51 EST → 07-15 in both frames
        ],
    )
    # Evaluate at 15:00 UTC on 07-15, climate day = 07-15.
    rec = _pred(
        ticker="KXHIGHNY-26JUL15-B89.5",
        city="NYC",
        expiration=date(2026, 7, 15),
        ts=datetime(2026, 7, 15, 15, 0, tzinfo=UTC),
        asos_m=95.0,  # logged under the (buggy) civil frame
        posterior=0.05,  # low-p NO bin
    )
    preds = tmp_path / "predictions.jsonl"
    _write_preds(preds, [rec])

    report = salvage(preds, asos_dir)
    summary = summarize(report)

    assert report["n_frame_v1_rows"] == 1
    row = report["rows"][0]
    assert row["recompute_status"] == "ok"
    # civil-frame M includes the 95° boundary ob; LST-frame M is 70°.
    assert row["recomputed_civil_M"] == 95.0
    assert row["recomputed_lst_M"] == 70.0
    assert row["frame_affected"] is True

    assert summary["n_affected_rows"] == 1
    assert summary["n_settlement_events"] == 1
    assert summary["n_events_with_any_affected_row"] == 1
    assert summary["n_low_p_no_bin_events"] == 1
    assert summary["n_low_p_no_bin_events_affected"] == 1


def test_unaffected_row_not_flagged(tmp_path: Path) -> None:
    """A row whose M is held by a daytime ob (same in both frames) is unaffected."""
    asos_dir = tmp_path / "asos"
    asos_dir.mkdir()
    _asos_csv(
        asos_dir / "kmia_asos.csv",
        "MIA",
        [
            ("2026-07-15 18:53", 89.0),  # afternoon — same climate day in both frames
            ("2026-07-15 19:53", 88.0),
        ],
    )
    rec = _pred(
        ticker="KXHIGHMIA-26JUL15-B89.5",
        city="MIA",
        expiration=date(2026, 7, 15),
        ts=datetime(2026, 7, 15, 20, 0, tzinfo=UTC),
        asos_m=89.0,
        posterior=0.30,  # NOT low-p
    )
    preds = tmp_path / "predictions.jsonl"
    _write_preds(preds, [rec])

    report = salvage(preds, asos_dir)
    summary = summarize(report)

    row = report["rows"][0]
    assert row["recomputed_civil_M"] == 89.0
    assert row["recomputed_lst_M"] == 89.0
    assert row["frame_affected"] is False
    assert summary["n_affected_rows"] == 0
    assert summary["n_events_with_any_affected_row"] == 0
    assert summary["n_low_p_no_bin_events"] == 0


def test_frame_v2_rows_excluded_from_salvage(tmp_path: Path) -> None:
    """frame_version>=2 rows are out of salvage scope (already LST)."""
    asos_dir = tmp_path / "asos"
    asos_dir.mkdir()
    _asos_csv(asos_dir / "kmia_asos.csv", "MIA", [("2026-07-15 18:53", 89.0)])
    rec = _pred(
        ticker="KXHIGHMIA-26JUL15-B89.5",
        city="MIA",
        expiration=date(2026, 7, 15),
        ts=datetime(2026, 7, 15, 20, 0, tzinfo=UTC),
        asos_m=89.0,
        posterior=0.05,
        frame_version=2,
    )
    preds = tmp_path / "predictions.jsonl"
    _write_preds(preds, [rec])
    report = salvage(preds, asos_dir)
    assert report["n_frame_v1_rows"] == 0


def test_missing_csv_marks_no_coverage(tmp_path: Path) -> None:
    """A row for a city with no CSV is reported unadjudicable, not affected."""
    asos_dir = tmp_path / "asos"
    asos_dir.mkdir()  # empty — no CSVs
    rec = _pred(
        ticker="KXHIGHAUS-26JUL15-B99.5",
        city="AUS",
        expiration=date(2026, 7, 15),
        ts=datetime(2026, 7, 15, 22, 0, tzinfo=UTC),
        asos_m=99.0,
        posterior=0.05,
    )
    preds = tmp_path / "predictions.jsonl"
    _write_preds(preds, [rec])
    report = salvage(preds, asos_dir)
    summary = summarize(report)
    assert report["rows"][0]["recompute_status"] == "no_csv_coverage"
    assert summary["n_no_csv_coverage_rows"] == 1
    assert summary["n_affected_rows"] == 0
