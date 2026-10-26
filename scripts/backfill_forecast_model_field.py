"""
Backfill ``model`` and ``model_version`` fields in existing weather_forecasts.json files.

All snapshots collected before multi-model support was enabled (2026-04-23) contain
GFS-only forecasts with no explicit ``model`` field.  Leaving that field absent is an
implicit convention that causes silent bugs downstream — calibration code that groups
by model will silently merge GFS and "unknown" rows, and blend-weight analysis cannot
distinguish sources.

This script makes the convention explicit: any record without a ``model`` field (or
with ``model`` set to None / empty string) is assumed to be GFS and patched with::

    model        = "gfs_seamless"
    model_version = "GFSv16"

The script is safe to re-run (idempotent): records that already have an explicit
``model`` field are not modified.

Usage::

    uv run python scripts/backfill_forecast_model_field.py [--dry-run] [--snapshots-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults applied to records that lack an explicit model field
# ---------------------------------------------------------------------------

_BACKFILL_MODEL = "gfs_seamless"
_BACKFILL_MODEL_VERSION = "GFSv16"


def _needs_backfill(record: dict[str, Any]) -> bool:
    """Return True if the record is missing a non-empty ``model`` field."""
    model = record.get("model")
    return model is None or model == ""


def _patch_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a new record dict with ``model`` and ``model_version`` set."""
    patched = dict(record)
    patched["model"] = _BACKFILL_MODEL
    patched["model_version"] = _BACKFILL_MODEL_VERSION
    return patched


def _process_file(
    path: Path,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Process a single weather_forecasts.json file.

    Returns:
        (records_total, records_patched) — counts for reporting.
    """
    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
    except Exception as exc:
        logger.warning("Skipping %s — could not read/parse: %s", path, exc)
        return 0, 0

    if not isinstance(data, list):
        logger.warning("Skipping %s — top-level JSON is not a list", path)
        return 0, 0

    records_total = len(data)
    to_patch = [i for i, rec in enumerate(data) if isinstance(rec, dict) and _needs_backfill(rec)]

    if not to_patch:
        logger.debug(
            "  %s — all %d records already have model field, skipping", path, records_total
        )
        return records_total, 0

    records_patched = len(to_patch)

    if dry_run:
        logger.info(
            "  [DRY-RUN] %s — would patch %d/%d records",
            path,
            records_patched,
            records_total,
        )
        return records_total, records_patched

    # Patch in-place (build new list to avoid mutating during iteration)
    patched_data = [
        _patch_record(rec) if isinstance(rec, dict) and _needs_backfill(rec) else rec
        for rec in data
    ]

    # Atomic write: write to a temp file in the same directory, then rename
    dir_path = path.parent
    original_mode = path.stat().st_mode
    try:
        fd, tmp_path_str = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            os.chmod(tmp_path_str, original_mode)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(patched_data, f, indent=2)
                f.write("\n")  # trailing newline for unix hygiene
        except Exception:
            os.unlink(tmp_path_str)
            raise
        os.replace(tmp_path_str, path)
    except Exception as exc:
        logger.error("Failed to write patched file %s: %s", path, exc)
        return records_total, 0

    logger.info(
        "  %s — patched %d/%d records",
        path,
        records_patched,
        records_total,
    )
    return records_total, records_patched


def run(
    snapshots_dir: Path,
    *,
    dry_run: bool,
) -> None:
    """Walk the snapshots directory and backfill all weather_forecasts.json files."""
    if not snapshots_dir.exists():
        logger.error("Snapshots directory does not exist: %s", snapshots_dir)
        sys.exit(1)

    forecast_files = sorted(snapshots_dir.rglob("weather_forecasts.json"))

    if not forecast_files:
        logger.info("No weather_forecasts.json files found under %s", snapshots_dir)
        return

    logger.info(
        "Found %d weather_forecasts.json file(s) under %s%s",
        len(forecast_files),
        snapshots_dir,
        " [DRY-RUN]" if dry_run else "",
    )

    total_files_updated = 0
    total_records_seen = 0
    total_records_patched = 0

    for path in forecast_files:
        records_total, records_patched = _process_file(path, dry_run=dry_run)
        total_records_seen += records_total
        if records_patched > 0:
            total_files_updated += 1
            total_records_patched += records_patched

    # Summary
    action = "Would update" if dry_run else "Updated"
    print(
        f"\n{'[DRY-RUN] ' if dry_run else ''}Backfill complete:\n"
        f"  Files examined : {len(forecast_files)}\n"
        f"  {action:<16}: {total_files_updated} file(s)\n"
        f"  Records seen   : {total_records_seen}\n"
        f"  Records patched: {total_records_patched}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill model/model_version fields in existing weather_forecasts.json snapshots. "
            "Records without an explicit model field are assumed to be GFS (gfs_seamless / GFSv16)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing any files.",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="data/snapshots",
        help="Root directory containing snapshot subdirectories (default: data/snapshots).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    args = _parse_args()
    run(
        snapshots_dir=Path(args.snapshots_dir),
        dry_run=args.dry_run,
    )
