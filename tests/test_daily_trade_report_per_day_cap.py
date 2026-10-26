"""Tests for per-day cap block count surfacing in the daily trade report."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.daily_trade_report import build_json_output, load_cap_blocks_by_cell


def _write_blocks(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_cap_blocks_counts_by_forecast_date(tmp_path: Path) -> None:
    blocks_path = tmp_path / "per_day_cap_blocks.jsonl"

    def _ek(city: str, var: str, fdate: str) -> dict:
        return {"event_key": {"city": city, "var": var, "forecast_date": fdate}}

    rows = [
        {"scan_id": "s1", **_ek("MIA", "high", "2026-05-21")},
        {"scan_id": "s2", **_ek("MIA", "high", "2026-05-21")},
        {"scan_id": "s3", **_ek("MIA", "high", "2026-05-22")},
        {"scan_id": "s4", **_ek("NYC", "low", "2026-05-21")},
    ]
    _write_blocks(blocks_path, rows)

    result = load_cap_blocks_by_cell(blocks_path, report_date="2026-05-21")

    assert result[("MIA", "high")] == 2
    assert result[("NYC", "low")] == 1
    assert ("MIA", "high") in result
    assert sum(result.values()) == 3


def test_load_cap_blocks_excludes_other_forecast_dates(tmp_path: Path) -> None:
    blocks_path = tmp_path / "per_day_cap_blocks.jsonl"
    rows = [
        {
            "scan_id": "s1",
            "event_key": {"city": "CHI", "var": "high", "forecast_date": "2026-05-22"},
        },
    ]
    _write_blocks(blocks_path, rows)

    result = load_cap_blocks_by_cell(blocks_path, report_date="2026-05-21")
    assert result == {}


def test_load_cap_blocks_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    result = load_cap_blocks_by_cell(missing, report_date="2026-05-21")
    assert result == {}


def test_load_cap_blocks_tolerates_corrupt_lines(tmp_path: Path) -> None:
    blocks_path = tmp_path / "per_day_cap_blocks.jsonl"
    good_row = {
        "scan_id": "s1",
        "event_key": {"city": "LAX", "var": "low", "forecast_date": "2026-05-21"},
    }
    blocks_path.write_text("not valid json\n" + json.dumps(good_row) + "\n\n")
    result = load_cap_blocks_by_cell(blocks_path, report_date="2026-05-21")
    assert result[("LAX", "low")] == 1


def test_json_output_omits_per_cell_when_include_cap_blocks_unset(tmp_path: Path) -> None:
    """build_json_output with cap_blocks_by_cell=None must produce an empty per_cell mapping."""
    # Simulate the main() call where --include-cap-blocks is not set:
    # cap_blocks_by_cell=cap_blocks if (cap_blocks and args.include_cap_blocks) else None
    cap_blocks = {("MIA", "high"): 2}  # non-empty file exists
    include_cap_blocks = False  # flag not set
    cap_blocks_arg = cap_blocks if (cap_blocks and include_cap_blocks) else None

    output = build_json_output(
        trade_date="2026-05-21",
        settled_days=[],
        pending=[],
        calibration={},
        cap_blocks_by_cell=cap_blocks_arg,
    )
    # per_cell must be empty (not populated with cap block counts)
    assert output["per_cell"] == {}


def test_json_output_includes_per_cell_when_include_cap_blocks_set(tmp_path: Path) -> None:
    """build_json_output with cap_blocks_by_cell set must include per_cell data."""
    cap_blocks = {("MIA", "high"): 2}
    include_cap_blocks = True
    cap_blocks_arg = cap_blocks if (cap_blocks and include_cap_blocks) else None

    output = build_json_output(
        trade_date="2026-05-21",
        settled_days=[],
        pending=[],
        calibration={},
        cap_blocks_by_cell=cap_blocks_arg,
    )
    assert "MIA/high" in output["per_cell"]
    assert output["per_cell"]["MIA/high"]["per_day_cap_blocks"] == 2


def test_load_cap_blocks_ignores_rows_missing_event_key(tmp_path: Path) -> None:
    blocks_path = tmp_path / "per_day_cap_blocks.jsonl"
    rows = [
        {"scan_id": "s1"},
        {
            "scan_id": "s2",
            "event_key": {"city": "MIA", "var": "high", "forecast_date": "2026-05-21"},
        },
    ]
    _write_blocks(blocks_path, rows)
    result = load_cap_blocks_by_cell(blocks_path, report_date="2026-05-21")
    assert result[("MIA", "high")] == 1
    assert len(result) == 1
