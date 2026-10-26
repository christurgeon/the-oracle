"""
Tests for scripts/backfill_forecast_model_field.py.

Covers:
- Idempotency: records that already have a model field are not modified.
- Dry-run: no file writes are performed when dry_run=True.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.backfill_forecast_model_field import _process_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_forecasts(path: Path, records: list[dict]) -> None:
    """Write a weather_forecasts.json file with the given records."""
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _read_forecasts(path: Path) -> list[dict]:
    """Read and parse a weather_forecasts.json file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_idempotent_when_already_patched(tmp_path: Path) -> None:
    """Records that already have model/model_version set must not be modified."""
    forecast_file = tmp_path / "weather_forecasts.json"
    original_records = [
        {"city": "NYC", "model": "gfs_seamless", "model_version": "GFSv16", "value": 72.0},
        {"city": "CHI", "model": "ecmwf_ifs025", "model_version": "IFS-49r1", "value": 65.0},
    ]
    _write_forecasts(forecast_file, original_records)
    original_mtime = forecast_file.stat().st_mtime

    records_total, records_patched = _process_file(forecast_file, dry_run=False)

    assert records_patched == 0, "No records should be patched when all already have model"
    assert records_total == len(original_records)

    # File content must be unchanged
    after_records = _read_forecasts(forecast_file)
    assert after_records == original_records


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """When dry_run=True, a file with records needing patching must not be modified."""
    forecast_file = tmp_path / "weather_forecasts.json"
    original_records = [
        {"city": "NYC", "value": 72.0},  # missing model field — would normally be patched
        {"city": "CHI", "value": 65.0},  # missing model field — would normally be patched
    ]
    original_text = json.dumps(original_records, indent=2) + "\n"
    forecast_file.write_text(original_text, encoding="utf-8")

    records_total, records_patched = _process_file(forecast_file, dry_run=True)

    # Should report that records need patching
    assert records_total == len(original_records)
    assert records_patched == len(original_records)

    # File must be unchanged
    assert forecast_file.read_text(encoding="utf-8") == original_text
