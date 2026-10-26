"""Heal settlement snapshots corrupted by the AUS/DEN morning-partial CLI bug.

Background
==========

``python/data/nws.py`` previously detected only ``"VALID TODAY"`` as a
preliminary-report marker. AUS and DEN stations issue a morning partial-day
CLI at ~07:00 AM local (``VALID AS OF 0700 AM LOCAL TIME`` — note: no
"TODAY" token) capturing the partial-day max/min observed so far. Because
the regex missed ``"VALID AS OF"``, that morning partial was flagged
``is_preliminary=False`` and written as canonical settlement, wiping the
next-morning ``VALID FOR YESTERDAY`` final when downstream loaders merged
on ``is_preliminary``.

What this does
==============

Scans every ``settlements.json`` under ``--snapshots-dir``, groups records
across all files by ``(city, observation_date)``, and for each group picks
the record with the greatest ``(issued_at, product_id)`` tuple as the
canonical "winner." In ``--execute`` mode, rewrites each file so that only
records that are winners remain (idempotent duplicates that agree with the
winner are removed and reported as ``no_change``; non-winner records that
disagree with the winner are reported as ``drop``).

Orphan groups — where the only available record is itself a corrupt
morning-partial with no clean alternative in the corpus — are dropped
entirely. Expected orphans: AUS/DEN for dates where only the morning
partial was ever snapshot'd.

Safety
======

- Default is ``--dry-run``; ``--execute`` required for writes.
- Acquires ``flock`` on ``{snapshots_dir}/.heal.lock`` before scanning.
- Backs up every file to ``{backup_dir}/settlements_backup_<utc_iso>/...``
  before rewriting; fully-atomic per-file tempfile + ``os.replace``.
- Today's directory is skipped (in-progress cron).

Usage
=====

    uv run python scripts/heal_settlements.py --dry-run
    uv run python scripts/heal_settlements.py --execute \\
        --snapshots-dir data/snapshots --backup-dir data/archive
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.utils.atomic_write import atomic_write_text  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_SNAPSHOTS_DIR = "data/snapshots"
DEFAULT_BACKUP_DIR = "data/archive"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RecordRef:
    """A single settlement record tagged with its source file."""

    file_path: Path
    record_index: int
    record: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.record.get("city", "")), str(self.record.get("observation_date", "")))

    @property
    def sort_key(self) -> tuple[str, str]:
        return (
            str(self.record.get("issued_at", "") or ""),
            str(self.record.get("product_id", "") or ""),
        )

    @property
    def is_same_day_partial(self) -> bool:
        """True when the record was issued on the same day as the observation,
        i.e. it is provably a same-day partial CLI (morning or afternoon).

        NWS finals for observation date D are issued pre-dawn on D+1, so
        ``issued_at.date() > observation_date``. Any record where
        ``issued_at.date() <= observation_date`` is a partial by definition,
        regardless of what the ``is_preliminary`` flag says — this is how we
        identify corpus records mis-tagged before the regex fix.
        """
        obs_raw = self.record.get("observation_date")
        issued_raw = self.record.get("issued_at")
        if not obs_raw or not issued_raw:
            return False
        try:
            obs = datetime.strptime(str(obs_raw), "%Y-%m-%d").date()
            issued_dt = datetime.fromisoformat(str(issued_raw))
        except (ValueError, TypeError):
            return False
        return issued_dt.date() <= obs


@dataclass
class _HealSummary:
    files_scanned: int = 0
    records_examined: int = 0
    records_dropped: int = 0
    records_kept_unchanged: int = 0
    files_modified: int = 0
    groups_total: int = 0
    # Groups where every candidate was issued same-day as the observation —
    # no clean next-morning final exists; all records drop. Expected for AUS/DEN
    # on dates where only the morning partial was snapshot'd pre-parser-fix.
    all_partial_groups: int = 0


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _load_settlement_files(snapshots_dir: Path, *, skip_today: bool) -> list[Path]:
    """Return every ``settlements.json`` under ``snapshots_dir``, skipping
    today's directory (still being written by cron) when requested."""
    today_iso = datetime.now(UTC).date().isoformat()
    out: list[Path] = []
    for p in sorted(snapshots_dir.rglob("settlements.json")):
        # The date component is the first path part under snapshots_dir.
        try:
            relative = p.relative_to(snapshots_dir)
        except ValueError:
            continue
        parts = relative.parts
        if skip_today and parts and parts[0] == today_iso:
            logger.info("heal: skipping today's dir: %s", p)
            continue
        out.append(p)
    return out


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("heal: could not read %s: %s", path, exc)
        return []
    if not isinstance(data, list):
        logger.warning("heal: %s is not a JSON list, skipping", path)
        return []
    return data


def _atomic_write_json(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, json.dumps(records, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Core heal logic
# ---------------------------------------------------------------------------


def _compare_values(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if two records have the same ``high_f`` and ``low_f``."""
    return a.get("high_f") == b.get("high_f") and a.get("low_f") == b.get("low_f")


def _plan_heal(
    files: list[Path],
) -> tuple[
    dict[Path, list[tuple[int, dict[str, Any], str]]],  # file -> [(idx, rec, action)]
    _HealSummary,
    list[dict[str, Any]],  # csv rows (ordered)
]:
    """Produce the heal plan without touching disk.

    Returns:
        - per_file_actions: maps each file to the list of
          ``(record_index, record, action)`` tuples it contains.
          ``action`` is one of ``keep`` / ``drop`` / ``no_change``.
        - summary: aggregate counters.
        - csv_rows: flat list of dicts for the CSV report.
    """
    per_file: dict[Path, list[_RecordRef]] = defaultdict(list)
    groups: dict[tuple[str, str], list[_RecordRef]] = defaultdict(list)

    for f in files:
        records = _read_records(f)
        for i, rec in enumerate(records):
            city = rec.get("city")
            obs = rec.get("observation_date")
            if not city or not obs:
                # Malformed record: nothing useful to do. Leave as-is in
                # the file (treated as no-change). We can't group it.
                continue
            ref = _RecordRef(file_path=f, record_index=i, record=rec)
            per_file[f].append(ref)
            groups[ref.key].append(ref)

    summary = _HealSummary()
    summary.files_scanned = len(files)
    summary.groups_total = len(groups)
    summary.records_examined = sum(len(v) for v in per_file.values())

    # Select a winner per group: max (issued_at, product_id), then deterministic
    # fallback on (file_path, record_index) so a tie on both fields resolves
    # consistently regardless of dict iteration order. A group whose ONLY
    # candidate(s) are all same-day partials has no clean alternative in the
    # corpus — every record drops. This is the orphan-corrupt case: expected
    # for AUS/DEN on observation dates where only the morning partial was ever
    # snapshot'd before the parser fix (e.g. 2026-04-22).
    winners: dict[tuple[str, str], _RecordRef | None] = {}
    for key, refs in groups.items():
        if all(r.is_same_day_partial for r in refs):
            # Every record in this group was issued same-day as the observation
            # — all partials, no clean final exists. Drop them all.
            winners[key] = None
            summary.all_partial_groups += 1
            continue
        # Otherwise pick the latest non-same-day-partial we have; if none
        # exists but partials do, fall back to the latest record overall.
        # (This handles the case where a pure partial-only group leaks into
        # a group that also has a legitimate next-morning final.)
        finals = [r for r in refs if not r.is_same_day_partial]
        pool = finals if finals else refs
        pool_sorted = sorted(
            pool,
            key=lambda r: (r.sort_key, str(r.file_path), r.record_index),
        )
        winners[key] = pool_sorted[-1]

    # Walk each file and label each record.
    per_file_actions: dict[Path, list[tuple[int, dict[str, Any], str]]] = {}
    csv_rows: list[dict[str, Any]] = []
    for f, refs in per_file.items():
        per_file_actions[f] = []
        file_modified = False
        for ref in refs:
            winner = winners[ref.key]
            if winner is None:
                # Orphan-corrupt group: drop the record.
                action = "drop"
                summary.records_dropped += 1
                file_modified = True
            elif (winner.file_path, winner.record_index) == (ref.file_path, ref.record_index):
                action = "keep"
            elif _compare_values(ref.record, winner.record):
                # Identical-valued duplicate — removed from file but nothing
                # materially changes. Report as no_change.
                action = "no_change"
                summary.records_kept_unchanged += 1
                file_modified = True
            else:
                action = "drop"
                summary.records_dropped += 1
                file_modified = True

            per_file_actions[f].append((ref.record_index, ref.record, action))
            csv_rows.append(
                {
                    "file_path": str(ref.file_path),
                    "city": ref.record.get("city", ""),
                    "observation_date": ref.record.get("observation_date", ""),
                    "issued_at": ref.record.get("issued_at", ""),
                    "is_preliminary": ref.record.get("is_preliminary", ""),
                    "high_f": ref.record.get("high_f", ""),
                    "low_f": ref.record.get("low_f", ""),
                    "action": action,
                }
            )

        if file_modified:
            summary.files_modified += 1

    return per_file_actions, summary, csv_rows


def _write_csv_report(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path",
        "city",
        "observation_date",
        "issued_at",
        "is_preliminary",
        "high_f",
        "low_f",
        "action",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _backup_files(
    files: list[Path],
    snapshots_dir: Path,
    backup_root: Path,
    run_ts: str,
) -> Path:
    """Copy every file into ``backup_root/settlements_backup_<ts>/`` preserving
    its path relative to ``snapshots_dir``. Returns the backup root directory.
    """
    target = backup_root / f"settlements_backup_{run_ts}"
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            rel = f.relative_to(snapshots_dir)
        except ValueError:
            # Outside snapshots_dir — fall back to the absolute path turned
            # into a relative one by stripping the leading slash.
            rel = Path(str(f).lstrip("/"))
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    return target


def _apply_heal(
    per_file_actions: dict[Path, list[tuple[int, dict[str, Any], str]]],
) -> None:
    """Rewrite each file keeping only records labeled ``keep``."""
    for f, actions in per_file_actions.items():
        if all(a == "keep" for _, _, a in actions):
            continue
        # actions is already accumulated in record_index order (see _plan_heal),
        # so the keep filter preserves original file ordering.
        kept = [rec for _, rec, action in actions if action == "keep"]
        _atomic_write_json(f, kept)


# ---------------------------------------------------------------------------
# Lock handling
# ---------------------------------------------------------------------------


class _Lock:
    """Non-blocking flock context manager."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any = None

    def __enter__(self) -> _Lock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"Could not acquire lock at {self._path}: another heal run "
                f"or snapshot writer may be active. Aborting."
            ) from exc
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _per_city_drop_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    drops: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["action"] == "drop":
            drops[str(row["city"])] += 1
    return dict(drops)


def heal(
    snapshots_dir: Path,
    backup_dir: Path,
    *,
    dry_run: bool,
    skip_today: bool = True,
) -> _HealSummary:
    """Run the heal workflow. Returns the aggregate summary."""
    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = snapshots_dir / f"heal_report_{run_ts}.csv"

    lock_path = snapshots_dir / ".heal.lock"
    with _Lock(lock_path):
        files = _load_settlement_files(snapshots_dir, skip_today=skip_today)
        per_file_actions, summary, csv_rows = _plan_heal(files)
        _write_csv_report(csv_rows, report_path)

        drops_by_city = _per_city_drop_counts(csv_rows)

        logger.info("heal: report written to %s", report_path)
        logger.info(
            "heal: files_scanned=%d records_examined=%d records_dropped=%d "
            "records_kept_unchanged=%d files_modified=%d groups=%d all_partial_groups=%d",
            summary.files_scanned,
            summary.records_examined,
            summary.records_dropped,
            summary.records_kept_unchanged,
            summary.files_modified,
            summary.groups_total,
            summary.all_partial_groups,
        )
        if drops_by_city:
            logger.info("heal: drops by city: %s", drops_by_city)

        if dry_run:
            logger.info("heal: dry-run — no files mutated")
            return summary

        # Execute mode: backup then rewrite.
        modified_files = [
            f for f, actions in per_file_actions.items() if any(a != "keep" for _, _, a in actions)
        ]
        if not modified_files:
            logger.info("heal: no files require modification")
            return summary

        backup_root = _backup_files(modified_files, snapshots_dir, backup_dir, run_ts)
        logger.info("heal: backup written under %s", backup_root)

        _apply_heal(per_file_actions)
        logger.info("heal: rewrote %d file(s)", len(modified_files))

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Heal NWS settlement snapshots corrupted by the AUS/DEN morning-"
            "partial bug. Default is dry-run; pass --execute to mutate."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Scan and produce the report without modifying any files (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually rewrite files after backing them up",
    )
    parser.add_argument(
        "--snapshots-dir",
        default=DEFAULT_SNAPSHOTS_DIR,
        help=f"Snapshots directory (default: {DEFAULT_SNAPSHOTS_DIR})",
    )
    parser.add_argument(
        "--backup-dir",
        default=DEFAULT_BACKUP_DIR,
        help=f"Backup root directory (default: {DEFAULT_BACKUP_DIR})",
    )
    parser.add_argument(
        "--no-skip-today",
        action="store_true",
        default=False,
        help="Do not skip today's snapshot directory (use only for tests)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = _parse_args()
    # When --execute is passed we override the default --dry-run=True.
    dry_run = not args.execute

    snapshots_dir = Path(args.snapshots_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    if not snapshots_dir.exists():
        logger.error("snapshots dir does not exist: %s", snapshots_dir)
        return 2

    try:
        summary = heal(
            snapshots_dir=snapshots_dir,
            backup_dir=backup_dir,
            dry_run=dry_run,
            skip_today=not args.no_skip_today,
        )
    except RuntimeError as exc:
        logger.error("heal aborted: %s", exc)
        return 3

    mode_label = "DRY-RUN" if dry_run else "EXECUTE"
    print(
        f"\n[{mode_label}] files_scanned={summary.files_scanned} "
        f"records_examined={summary.records_examined} "
        f"records_dropped={summary.records_dropped} "
        f"records_kept_unchanged={summary.records_kept_unchanged} "
        f"files_modified={summary.files_modified}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
