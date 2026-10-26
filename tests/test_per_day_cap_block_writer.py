"""Tests for PerDayCapBlockWriter (audit stream for per-day-cap blocks)."""

from __future__ import annotations

import json
import os as _os
from datetime import UTC, datetime
from pathlib import Path

from python.utils.jsonl import PerDayCapBlockWriter


def test_writes_one_row_per_record(tmp_path: Path) -> None:
    path = tmp_path / "blocks.jsonl"
    writer = PerDayCapBlockWriter(path)
    writer.write({"scan_id": "scan-1", "event_key": {"city": "NYC"}})
    writer.write({"scan_id": "scan-2", "event_key": {"city": "MIA"}})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scan_id"] == "scan-1"
    assert json.loads(lines[1])["scan_id"] == "scan-2"


def test_write_swallows_oserror_and_records_timestamp(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blocks.jsonl"
    writer = PerDayCapBlockWriter(path)

    def _raise_oserror(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "open", _raise_oserror)
    # Must not raise
    writer.write({"x": 1})
    assert writer._last_error_at is not None


def test_init_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "subdir" / "blocks.jsonl"
    PerDayCapBlockWriter(nested)
    assert nested.parent.exists()


def test_prunes_after_retention_window(tmp_path: Path) -> None:
    path = tmp_path / "blocks.jsonl"
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    w0 = PerDayCapBlockWriter(path, retention_days=14, clock=lambda: t0)
    w0.write({"x": 1})
    assert path.exists()
    old_ts = t0.timestamp() - (30 * 86400)
    _os.utime(path, (old_ts, old_ts))
    w1 = PerDayCapBlockWriter(path, retention_days=14, clock=lambda: t0)
    w1.write({"x": 2})
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["x"] == 2
