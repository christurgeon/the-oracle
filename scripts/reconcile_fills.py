"""Operator CLI: backfill missing rows in state/trades.jsonl from Kalshi.

Spec §12.1. Defaults to dry-run. Wraps FillsReconciler.backfill().

Usage:
    uv run python -m scripts.reconcile_fills \\
        --start 2026-05-04T00:00:00Z \\
        --end 2026-05-06T00:00:00Z \\
        --trade-log data/vps_trades.jsonl \\
        --allowlist OID1,OID2,...,OID8 \\
        --expect-n 8 \\
        --dry-run

For real backfill add --apply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


def _parse_iso_to_unix(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reconcile_fills", description=__doc__)
    p.add_argument("--start", required=True, help="ISO8601 start timestamp (inclusive).")
    p.add_argument("--end", required=True, help="ISO8601 end timestamp (inclusive).")
    p.add_argument("--trade-log", default="state/trades.jsonl", help="Path to trades.jsonl")
    p.add_argument("--allowlist", default=None,
                   help="Comma-separated order_ids; planned writes outside this set abort.")
    p.add_argument("--allow-extras", action="store_true",
                   help="Override the allowlist abort.")
    p.add_argument("--expect-n", type=int, default=None,
                   help="Abort if planned writes != N.")
    p.add_argument("--max-window-days", type=int, default=30,
                   help="Sanity cap on (end - start).")
    p.add_argument("--force-strategy", default=None,
                   help="Strategy name applied uniformly to unresolved rows. "
                        "Requires interactive confirmation when paired with --apply.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--dry-run", action="store_true", default=True,
                     help="Default: print planned writes, do not mutate.")
    grp.add_argument("--apply", action="store_true",
                     help="Write the planned synthetic rows to trade-log.")
    # Test-only — hidden from --help so operators don't accidentally use it.
    p.add_argument("--mock-fixture", default=None, help=argparse.SUPPRESS)
    return p


async def _run(args: argparse.Namespace) -> int:
    start_unix = _parse_iso_to_unix(args.start)
    end_unix = _parse_iso_to_unix(args.end)
    if (end_unix - start_unix) > args.max_window_days * 86400:
        print(f"ERROR: window exceeds --max-window-days={args.max_window_days}", file=sys.stderr)
        return 2

    allowlist = None
    if args.allowlist:
        allowlist = set(args.allowlist.split(","))

    # Confirmation prompt for --force-strategy + --apply
    if args.force_strategy and args.apply:
        try:
            confirm = input(
                f'Type "yes-i-am-sure" to confirm overriding strategy '
                f'attribution with --force-strategy={args.force_strategy}: '
            )
        except EOFError:
            confirm = ""
        if confirm.strip() != "yes-i-am-sure":
            print("Aborted: confirmation failed.", file=sys.stderr)
            return 3

    # Construct a minimal FillsReconciler for backfill use
    from python.execution.fills_reconciler import FillsReconciler

    is_live_client = False
    if args.mock_fixture:
        # Test path: load fixture in place of a real Kalshi client
        fixture = json.loads(Path(args.mock_fixture).read_text())

        class _MockClient:
            async def get_fills(self, **_kwargs):
                return {"fills": fixture, "cursor": None}

        client: object = _MockClient()
    else:
        from python.data.kalshi_client import KalshiClient
        from python.settings import Settings
        settings = Settings()
        client = KalshiClient(
            api_key=settings.kalshi_api_key,
            private_key_path=settings.kalshi_private_key_path,
            base_url=settings.active_base_url,
            contract_watcher=None,
        )
        await client.connect()
        is_live_client = True

    try:
        reconciler = FillsReconciler(
            state_path=Path("state/fills_reconciler.json"),
            trade_log_path=Path(args.trade_log),
            trade_log_lock=asyncio.Lock(),
            kalshi_client=client,
            engine=None,  # CLI: no in-memory orders
            notifier=None,
        )

        result = await reconciler.backfill(
            min_ts=start_unix,
            max_ts=end_unix,
            allowlist=allowlist,
            expect_n=args.expect_n,
            allow_extras=args.allow_extras,
            force_strategy=args.force_strategy,
            dry_run=not args.apply,
        )

        for row in result.planned_writes:
            print(
                f"PLANNED: order_id={row['order_id']} "
                f"ticker={row['market_ticker']} "
                f"side={row['side']} contracts={row['contracts']} "
                f"fill_price=${row['fill_price']:.2f} "
                f"created_time={row.get('timestamp', '')} "
                f"strategy_resolved_via={row['metadata']['strategy_resolution']}"
            )
        print()
        print("SUMMARY:")
        print(f"  planned_writes: {len(result.planned_writes)}")
        if allowlist:
            print(f"  expected_order_ids (--allowlist): {sorted(allowlist)}")
        print(f"  unresolved_count: {result.unresolved_count}")
        print(f"  written_count: {result.written_count}")
        if result.aborted:
            print(f"  decision: ABORT - {result.abort_reason}")
            return 4
        print(
            f"  decision: {'WRITTEN' if args.apply else 'DRY-RUN (no writes)'}"
        )
        return 0
    finally:
        # Live client holds an HTTP session; close it so the CLI exits cleanly.
        if is_live_client:
            await client.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
