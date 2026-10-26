"""Spec §11 test 19. CLI smoke test for scripts/reconcile_fills.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests/fixtures/fills_reconciler/missed_orders_2026-05-04.json"


def test_cli_dry_run_smoke(tmp_path):
    """Dry-run prints planned writes; exits 0; no state written."""
    log_path = tmp_path / "trades.jsonl"
    log_path.touch()

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.reconcile_fills",
            "--start", "2026-05-04T00:00:00Z",
            "--end", "2026-05-06T00:00:00Z",
            "--trade-log", str(log_path),
            "--dry-run",
            "--mock-fixture", str(FIXTURE),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "PLANNED:" in result.stdout
    assert (
        "2 planned writes" in result.stdout
        or "planned_writes: 2" in result.stdout
    )


def test_cli_apply_requires_no_dry_run_flag():
    """--dry-run and --apply are mutually exclusive."""
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.reconcile_fills",
            "--start", "2026-05-04T00:00:00Z",
            "--end", "2026-05-06T00:00:00Z",
            "--dry-run", "--apply",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert (
        "mutually exclusive" in result.stderr.lower()
        or "not allowed" in result.stderr.lower()
    )
