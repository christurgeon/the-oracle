from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from python.utils.jsonl import ForecastsRawWriter


def _row(scan_id: str = "abc") -> dict:
    return {
        "scan_id": scan_id,
        "city": "AUS",
        "var": "high",
        "forecast_date": "2026-05-05",
        "lead": 0,
        "gfs_members_raw": [70.0] * 30,
        "gfs_ens_mean_raw_ddof1": 70.0,
        "gfs_ens_std_raw_ddof1": 1.0,
        "ecmwf_members_raw": [70.0] * 50,
        "ecmwf_ens_mean_raw_ddof1": 70.0,
        "ecmwf_ens_std_raw_ddof1": 1.0,
        "gfs_bias_correction_f": 0.0,
    }


def test_write_batch_appends_one_line_per_row(tmp_path: Path) -> None:
    p = tmp_path / "forecasts_raw.jsonl"
    w = ForecastsRawWriter(p)
    w.write_batch([_row("a"), _row("b")])
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scan_id"] == "a"
    assert json.loads(lines[1])["scan_id"] == "b"


def test_each_row_payload_under_pipe_buf(tmp_path: Path) -> None:
    p = tmp_path / "forecasts_raw.jsonl"
    ForecastsRawWriter(p)
    row = _row("x")
    payload = (json.dumps(row) + "\n").encode("utf-8")
    assert len(payload) < 4096, "row exceeds PIPE_BUF — atomic-append guarantee at risk"


def test_rotation_on_utc_day_boundary(tmp_path: Path) -> None:
    p = tmp_path / "forecasts_raw.jsonl"
    times = iter(
        [
            datetime(2026, 5, 4, 23, 59, tzinfo=UTC),
            datetime(2026, 5, 5, 0, 1, tzinfo=UTC),
        ]
    )
    w = ForecastsRawWriter(p, clock=lambda: next(times))
    w.write_batch([_row("day1")])
    w.write_batch([_row("day2")])
    rotated = p.with_name("forecasts_raw-2026-05-04.jsonl")
    assert rotated.exists()
    assert p.exists()
    assert json.loads(p.read_text().splitlines()[0])["scan_id"] == "day2"


def test_retention_prunes_old_siblings(tmp_path: Path) -> None:
    p = tmp_path / "forecasts_raw.jsonl"
    # Create 16 rotated siblings spanning 16 days (2026-04-20 .. 2026-05-05).
    base = date(2026, 4, 20)
    for i in range(16):
        d = base + timedelta(days=i)
        sib = p.with_name(f"forecasts_raw-{d.isoformat()}.jsonl")
        sib.touch()

    times = iter([datetime(2026, 5, 6, 12, 0, tzinfo=UTC)])
    w = ForecastsRawWriter(p, retention_days=14, clock=lambda: next(times))
    w.write_batch([_row("today")])

    # Files older than today - 14 days should be deleted (siblings dated < 2026-04-22).
    siblings = sorted(s.stem for s in p.parent.glob("forecasts_raw-*.jsonl"))
    assert "forecasts_raw-2026-04-20" not in siblings
    assert "forecasts_raw-2026-04-21" not in siblings
    assert "forecasts_raw-2026-04-22" in siblings
