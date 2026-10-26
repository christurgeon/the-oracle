"""Backfill KXRAIN settlement rows after a lapse longer than the daily sweep's ~28-day reach.

Same sweep, page cap raised; idempotent against ``state/kxrain_settled_index.json``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kalshi_client import KalshiClient  # noqa: E402
from python.data.kxrain_settlements import (  # noqa: E402
    DEFAULT_KEEP_SETTLEMENT_DATES,
    KXRAIN_SETTLEMENTS_FILENAME,
    SETTLED_INDEX_FILENAME,
    SETTLEMENT_SWEEP_MAX_PAGES,
    KxRainSettlementSweep,
    SettledIndex,
    SettlementSweepResult,
)
from python.settings import Settings  # noqa: E402

# Wider than the sweep's default: a deep walk is a real burst against Kalshi's rate limit.
DEEP_WALK_PAGE_DELAY_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="state",
        help="Directory holding kxrain_settlements.jsonl (default: state)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help=(
            "Pages of the settled listing to walk, 100 market-days each "
            f"(default: 10; the daily sweep uses {SETTLEMENT_SWEEP_MAX_PAGES})"
        ),
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=DEEP_WALK_PAGE_DELAY_SECONDS,
        help=f"Seconds between pages (default: {DEEP_WALK_PAGE_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the index already covers and exit without calling Kalshi",
    )
    parser.add_argument("--json", dest="json_path", help="Write the result as JSON to PATH")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="Add rows already on disk but missing from the index to the index, then exit "
        "without calling Kalshi",
    )
    return parser.parse_args()


def _rows_on_disk(log_dir: Path) -> dict[tuple[str, str], str]:
    """``(station, settlement_date) -> row ts`` across all rotated files, first row wins."""
    stem = Path(KXRAIN_SETTLEMENTS_FILENAME).stem
    found: dict[tuple[str, str], str] = {}
    for path in sorted(log_dir.glob(f"{stem}*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            station, day = row.get("station"), row.get("settlement_date")
            if station and day:
                found.setdefault((str(station), str(day)), str(row.get("ts") or ""))
    return found


def corpus_keys(log_dir: Path) -> set[tuple[str, str]]:
    """Keys on disk across all rotated files, not just the current one."""
    return set(_rows_on_disk(log_dir))


def reconcile_index(log_dir: Path) -> int:
    """Mark every on-disk row missing from the index; returns how many were added.

    Repairs a write that landed but was never marked, so the next sweep does not
    write it again.
    """
    index = SettledIndex.load(log_dir / SETTLED_INDEX_FILENAME)
    added = 0
    for (station, day), ts in sorted(_rows_on_disk(log_dir).items()):
        try:
            key = (station, date.fromisoformat(day))
        except ValueError:
            continue
        if not index.contains(key):
            index.mark(key, ts=ts)
            added += 1
    if added and not index.save_atomic():
        raise OSError(f"could not save {log_dir / SETTLED_INDEX_FILENAME}")
    return added


def report_coverage(log_dir: Path) -> None:
    """Reports the index and the on-disk stream — losing either looks different."""
    index = SettledIndex.load(log_dir / SETTLED_INDEX_FILENAME)
    on_disk = corpus_keys(log_dir)
    dates = sorted({key[1] for key in index.keys()}, reverse=True)
    print(f"Settled index: {log_dir / SETTLED_INDEX_FILENAME}")
    print(f"  market-days in index: {len(index)}")
    print(f"  market-days on disk:  {len(on_disk)}")

    indexed = {(st, d.isoformat()) for st, d in index.keys()}
    unindexed = on_disk - indexed
    missing = indexed - on_disk
    if unindexed:
        print(
            f"  !! {len(unindexed)} row(s) on disk are NOT in the index — a sweep "
            "would write them again. Restore the index before backfilling."
        )
    if missing:
        print(f"  !! {len(missing)} index entr(ies) have no row on disk — outcomes lost.")

    if not dates:
        print("  settlement dates:     none — nothing has been swept yet")
        return
    print(f"  settlement dates:     {len(dates)} ({dates[-1]} .. {dates[0]})")
    per_date: dict[str, int] = {}
    for _station, day in index.keys():
        per_date[day.isoformat()] = per_date.get(day.isoformat(), 0) + 1
    # Newest/oldest dates are thin by construction (still settling, or the
    # page cut lands mid-day) — excluded rather than reported as damage.
    widest = max(per_date.values())
    interior = sorted(per_date)[1:-1]
    thin = [d for d in sorted(interior, reverse=True) if per_date[d] < widest]
    if thin:
        print(
            f"  dates below the widest observed day ({widest}): "
            f"{len(thin)} (newest {thin[0]})"
        )


def print_result(result: SettlementSweepResult) -> None:
    print("\nSweep result:")
    print(f"  pages consumed:        {result.pages_consumed}")
    print(f"  market-days returned:  {result.n_returned}")
    print(f"  rows written:          {result.n_written} ({result.n_persisted} reached disk)")
    print(f"  already recorded:      {result.n_already_present}")
    print(f"  refused (not settled): {result.n_refused_unsettled}")
    if result.n_refused_not_finalized:
        print(f"  refused (not final):   {result.n_refused_not_finalized}")
    if result.n_refused_oversize:
        print(f"  refused (oversize):    {result.n_refused_oversize} — investigate")
    if result.n_persisted < result.n_written:
        print(
            f"  !! ONLY {result.n_persisted} OF {result.n_written} ROWS REACHED DISK — "
            "check free space; the rest stay retryable."
        )
    if not result.index_saved:
        print("  !! THE SETTLED INDEX COULD NOT BE SAVED — the next run will duplicate.")
    if result.n_unparseable_ticker:
        print(
            f"  UNPARSEABLE TICKERS:   {result.n_unparseable_ticker} "
            f"{result.unparseable_tickers} — the universe changed shape"
        )
    if result.n_rule_inconsistent:
        print(
            f"  RULE-INCONSISTENT:     {result.n_rule_inconsistent} — "
            "result disagrees with expiration_value; this is a CRITICAL gate line"
        )
    if result.backoff_aborted:
        print(
            "  BACKOFF: Kalshi's shared 429 gate went active mid-walk, so the sweep "
            "stopped rather than deepen a cooldown the live bot is queued behind. "
            "Re-run in a few minutes."
        )
    if result.cycle_aborted:
        print(
            "  CURSOR CYCLE: Kalshi returned an already-seen cursor; the walk was cut "
            "short and the repeating page was discarded. Re-run."
        )
    if result.truncated:
        print(
            "  TRUNCATED: the listing did not end within the page cap. Re-run with a "
            "higher --max-pages if you are recovering a long lapse."
        )
    if result.error:
        print(f"  ERROR: {result.error}")



def exit_code(result: SettlementSweepResult) -> int:
    """Non-zero on anything that lost, or may have lost, data — not just ``result.error``."""
    if result.anomaly or result.n_unparseable_ticker or result.n_rule_inconsistent:
        return 1
    return 0


async def run(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    if args.reconcile:
        print(f"Reconciled: {reconcile_index(log_dir)} market-day(s) added to the index.")
        report_coverage(log_dir)
        return 0
    report_coverage(log_dir)
    if args.dry_run:
        print("\n--dry-run: no Kalshi calls made.")
        return 0

    settings = Settings.load()
    print(
        f"\nWalking up to {args.max_pages} page(s) of the settled listing "
        f"({args.max_pages * 100} market-days, ~{args.max_pages * 100 // 27} days)..."
    )
    async with KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
    ) as client:
        sweep = KxRainSettlementSweep(
            client,
            log_dir=log_dir,
            max_pages=args.max_pages,
            page_delay=args.page_delay,
            # Must not be shorter than the daily sweep's window, or it prunes this walk's marks.
            keep_dates=max(args.max_pages * 4, DEFAULT_KEEP_SETTLEMENT_DATES),
        )
        result = await sweep.run_once()

    print_result(result)

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nJSON written to {args.json_path}")

    print()
    report_coverage(log_dir)
    return exit_code(result)


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
