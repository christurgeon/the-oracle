"""Tests for ``scripts/heal_settlements.py``.

Covers the three safety invariants the script must enforce:

1. Dry-run makes no writes.
2. Execute mode picks the latest-issued record, drops the stale partial-day
   morning CLI, and leaves a backup.
3. A concurrent lock holder causes the script to abort with a non-zero exit.
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.heal_settlements import heal  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_settlements(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def _build_corrupt_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal snapshot corpus that replicates the AUS bug.

    Returns ``(snapshots_dir, backup_dir)``.

    Two settlement files:

      - 2026-04-18/2000/settlements.json — contains the corrupt AUS morning
        partial (high=74, low=68) that was misclassified as final.
      - 2026-04-19/2000/settlements.json — contains the real next-morning
        final (high=74, low=51).

    A clean NYC record exists in both files with identical values so the
    "no_change" action is exercised.
    """
    snaps = tmp_path / "snapshots"
    backup = tmp_path / "archive"
    snaps.mkdir()
    backup.mkdir()

    corrupt_aus = {
        "city": "AUS",
        "cli_location": "AUS",
        "observation_date": "2026-04-18",
        "high_f": 74,
        "low_f": 68,  # wrong — morning partial
        "product_id": "morning-partial-aus",
        "issued_at": "2026-04-18T12:41:00+00:00",
        "is_preliminary": False,  # the bug: should have been True
    }
    final_aus = {
        "city": "AUS",
        "cli_location": "AUS",
        "observation_date": "2026-04-18",
        "high_f": 74,
        "low_f": 51,  # correct
        "product_id": "final-aus",
        "issued_at": "2026-04-19T07:07:00+00:00",
        "is_preliminary": False,
    }
    clean_nyc = {
        "city": "NYC",
        "cli_location": "NYC",
        "observation_date": "2026-04-18",
        "high_f": 70,
        "low_f": 55,
        "product_id": "final-nyc",
        "issued_at": "2026-04-19T06:30:00+00:00",
        "is_preliminary": False,
    }

    _write_settlements(
        snaps / "2026-04-18" / "2000" / "settlements.json",
        [corrupt_aus, clean_nyc],
    )
    _write_settlements(
        snaps / "2026-04-19" / "2000" / "settlements.json",
        [final_aus, clean_nyc],
    )

    return snaps, backup


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_heal_dry_run_makes_no_writes(tmp_path: Path) -> None:
    snaps, backup = _build_corrupt_corpus(tmp_path)
    target_file = snaps / "2026-04-18" / "2000" / "settlements.json"
    mtime_before = target_file.stat().st_mtime
    before_contents = target_file.read_text()

    archive_contents_before = sorted(backup.rglob("*"))

    summary = heal(
        snapshots_dir=snaps,
        backup_dir=backup,
        dry_run=True,
        skip_today=False,
    )

    mtime_after = target_file.stat().st_mtime
    after_contents = target_file.read_text()

    assert mtime_before == mtime_after
    assert before_contents == after_contents
    # No backup directory should have been created.
    assert sorted(backup.rglob("*")) == archive_contents_before
    # Summary should still report what *would* change.
    assert summary.records_dropped == 1  # the corrupt AUS record
    # The 2026-04-19 file still carries the clean NYC duplicate, idempotent.
    assert summary.records_kept_unchanged >= 1
    # CSV report is expected in the snapshots dir.
    reports = list(snaps.glob("heal_report_*.csv"))
    assert len(reports) == 1


def test_heal_execute_keeps_latest_issued(tmp_path: Path) -> None:
    snaps, backup = _build_corrupt_corpus(tmp_path)
    corrupt_file = snaps / "2026-04-18" / "2000" / "settlements.json"
    final_file = snaps / "2026-04-19" / "2000" / "settlements.json"

    summary = heal(
        snapshots_dir=snaps,
        backup_dir=backup,
        dry_run=False,
        skip_today=False,
    )

    # The corrupt AUS record must be gone from the earlier file.
    corrupt_records = json.loads(corrupt_file.read_text())
    cities_in_corrupt = {r["city"] for r in corrupt_records}
    assert "AUS" not in cities_in_corrupt  # corrupt was dropped
    # The NYC record is an idempotent duplicate; it should be dropped from
    # this file too because the winner for NYC lives in 2026-04-19.
    # (At worst, the record remains only if identical-valued duplicate is
    # retained — but per spec only winners remain.)
    assert all(r["city"] != "NYC" for r in corrupt_records)

    # The later file must retain the correct final AUS record.
    final_records = json.loads(final_file.read_text())
    aus_records = [r for r in final_records if r["city"] == "AUS"]
    assert len(aus_records) == 1
    assert aus_records[0]["low_f"] == 51
    assert aus_records[0]["high_f"] == 74

    # A backup must exist at the expected path.
    backup_dirs = list(backup.glob("settlements_backup_*"))
    assert len(backup_dirs) == 1
    backup_root = backup_dirs[0]
    backup_copy = backup_root / "2026-04-18" / "2000" / "settlements.json"
    assert backup_copy.exists(), f"expected backup at {backup_copy}"
    # Backup contents must match the pre-heal state.
    backup_records = json.loads(backup_copy.read_text())
    assert any(r["city"] == "AUS" and r["low_f"] == 68 for r in backup_records)

    # Summary sanity checks.
    assert summary.records_dropped == 1
    assert summary.files_modified >= 1


def _hold_lock(lock_path: Path, ready_flag: Path, stop_flag: Path) -> None:
    """Child-process entry point that grabs the flock and holds it."""
    with open(lock_path, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        ready_flag.write_text("ready")
        # Hold the lock until the parent removes the stop_flag file.
        while stop_flag.exists():
            time.sleep(0.05)


def test_heal_refuses_without_lock(tmp_path: Path) -> None:
    snaps, backup = _build_corrupt_corpus(tmp_path)
    lock_path = snaps / ".heal.lock"
    ready_flag = tmp_path / "holder-ready"
    stop_flag = tmp_path / "keep-holding"
    stop_flag.write_text("yes")

    # Hold the flock in a separate process so fcntl.LOCK_EX actually blocks
    # the test's attempt (same-process flock is re-entrant on Linux).
    proc = multiprocessing.Process(target=_hold_lock, args=(lock_path, ready_flag, stop_flag))
    proc.start()
    try:
        # Wait for the child to acquire the lock.
        for _ in range(200):
            if ready_flag.exists():
                break
            time.sleep(0.05)
        else:
            proc.terminate()
            pytest.fail("lock-holder subprocess never signalled ready")

        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            heal(
                snapshots_dir=snaps,
                backup_dir=backup,
                dry_run=True,
                skip_today=False,
            )
    finally:
        # Signal the child to release and clean up.
        stop_flag.unlink(missing_ok=True)
        proc.join(timeout=5)
        if proc.is_alive():  # pragma: no cover - defensive
            proc.terminate()
            proc.join()


# ---------------------------------------------------------------------------
# CLI smoke test — exercises the argparse + main() return codes.
# ---------------------------------------------------------------------------


def test_cli_dry_run_exits_zero(tmp_path: Path) -> None:
    snaps, backup = _build_corrupt_corpus(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "heal_settlements.py"),
            "--dry-run",
            "--snapshots-dir",
            str(snaps),
            "--backup-dir",
            str(backup),
            "--no-skip-today",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
