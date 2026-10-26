"""Backfill settlement outcomes lost before the settlement store became append-only.

``data/shadow_settlements.json`` only became append-only on 2026-09-06
(``scripts/fetch_shadow_settlements.py``): before that, a
plain overwrite silently dropped any ticker Kalshi 404s (markets older than
~70 days), turning cumulative cohorts into a moving window. Outcomes lost
before the fix landed cannot be recovered from Kalshi — but as
``fetch_shadow_settlements.py``'s own module docstring notes, they CAN be
rebuilt from ``data/snapshots/<date>/settlements.json`` (NWS-CLI finals)
plus the strike metadata already stored on each trade row.

This reuses the exact derivation ``scripts/favorite_bias_report.py``'s
``load_buy_rows`` already applies for the calibration section — same
(city, forecast_date) settlement lookup, same ``derive_outcome_yes`` call.
Before writing anything, it re-derives every ticker ALREADY settled in the
store and aborts if the derivation disagrees with a single one of them —
the only forward check available that the recovery logic is trustworthy.

Additive, idempotent, and scoped: a ticker already settled in the store (a
real Kalshi result) is NEVER overwritten by a reconstructed one; pass
``--strategy`` (repeatable) to restrict WHICH trades' tickers get
reconstructed (recommended — an unscoped run reconstructs outcomes for
every strategy present in the trade log, which may be a much bigger and
separately-reviewable change than intended).

Defaults to a dry-run preview; pass ``--write`` to persist. A write is
preceded by a timestamped backup of the existing store and performed
atomically (temp file + os.replace).

Usage:
    uv run python scripts/backfill_settlements_from_snapshots.py --strategy favorite_bias
    uv run python scripts/backfill_settlements_from_snapshots.py --strategy favorite_bias --write

After writing, regenerate shadow_settlements.full.json with the normal
refresh (the backfilled entries persist through it — merge_settlements only
overwrites tickers Kalshi's fresh fetch can still see):

    uv run python scripts/favorite_bias_report.py --refresh

NOTE: this changes the corpus consumed by scripts/daily_trade_report.py's
automated Telegram output (oracle-daily-report.service). Run that report
manually once to review the change before the next scheduled tick.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.analytics.strategy_attribution import build_strategy_index  # noqa: E402
from python.execution.trade_log import is_pnl_relevant_row  # noqa: E402
from python.utils.atomic_write import atomic_write_text  # noqa: E402
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date  # noqa: E402
from scripts._calibration_common import load_snapshot_data  # noqa: E402
from scripts.fetch_shadow_settlements import (  # noqa: E402
    apply_strategy_enrichment,
    is_settled,
    load_settlement_store,
)

DEFAULT_TRADES = _PROJECT_ROOT / "data" / "vps_trades.jsonl"
DEFAULT_STORE = _PROJECT_ROOT / "data" / "shadow_settlements.json"
DEFAULT_SNAPSHOTS = _PROJECT_ROOT / "data" / "snapshots"


def _resolve_forecast_date(trade: dict[str, Any]) -> str | None:
    """Resolve a trade's forecast date (ISO ``YYYY-MM-DD``), same precedence
    ``reconstruct_outcome`` uses: metadata first, ticker parse only when
    metadata is missing. Shared so agreement checking and reconstruction
    always agree on which month a ticker belongs to.
    """
    meta = trade.get("metadata") or {}
    forecast_date = meta.get("forecast_date")
    if forecast_date is not None:
        return forecast_date
    fd = parse_ticker_date(trade.get("market_ticker") or "")
    if fd is None:
        return None
    return fd.isoformat()


def _month_of(trade: dict[str, Any]) -> str | None:
    """``YYYY-MM`` for a trade's resolved forecast date, or ``None`` if unresolvable."""
    forecast_date = _resolve_forecast_date(trade)
    if forecast_date is None or len(forecast_date) < 7:
        return None
    return forecast_date[:7]


def _month_counts(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Bucket a list of trades by resolved forecast-date month (``YYYY-MM``).

    A trade whose month can't be resolved is silently excluded from the
    buckets (its ticker also has no snapshot-derivable outcome, so it
    wouldn't be reconstructed either).
    """
    counts: dict[str, int] = {}
    for trade in trades:
        month = _month_of(trade)
        if month is None:
            continue
        counts[month] = counts.get(month, 0) + 1
    return counts


def _format_month_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return "; ".join(f"{month}: {n}" for month, n in sorted(counts.items()))


def reconstruct_outcome(
    trade: dict[str, Any],
    settlements: dict[tuple[str, str], dict],
) -> dict[str, Any] | None:
    """Rebuild a settlement-store entry for one trade from NWS-CLI snapshot data.

    Uses the same two primitives ``load_buy_rows`` (scripts/favorite_bias_report.py)
    already applies for the calibration section — a (city, forecast_date)
    settlement lookup and ``derive_outcome_yes`` — so a backfilled outcome
    cannot diverge from what the live calibration path already trusts. Prefers
    the trade's own ``metadata`` (city/var/forecast_date); when any of those
    three is absent, falls back to parsing them from the ticker instead
    (unlike ``load_buy_rows``, which always parses the ticker for the date).
    On the live corpus these two sources never disagreed (0 mismatches across
    767 fill rows as of 2026-09-12), so the fallback is a convenience for rows
    with thin ``metadata``, not an observed divergence.

    Returns ``None`` when the trade lacks enough strike/city/date metadata, or
    no matching NWS-CLI settlement exists for that (city, date) — callers
    should leave such tickers unsettled rather than guess.
    """
    ticker = trade.get("market_ticker") or ""
    meta = trade.get("metadata") or {}

    city = meta.get("city")
    var = meta.get("market_type")
    if city is None or var is None:
        parsed = parse_ticker_city_var(ticker)
        if parsed is None:
            return None
        city = city or parsed[0]
        var = var or parsed[1]

    forecast_date = _resolve_forecast_date(trade)
    if forecast_date is None:
        return None

    strike_type = meta.get("strike_type")
    if strike_type is None:
        return None
    floor_strike = meta.get("floor_strike")
    cap_strike = meta.get("cap_strike")

    st = settlements.get((city, forecast_date))
    if st is None:
        return None
    temp = st.get("high_f") if var == "high" else st.get("low_f")
    if temp is None:
        return None

    yes = derive_outcome_yes(strike_type, floor_strike, cap_strike, temp)
    if yes is None:
        return None

    return {
        "ticker": ticker,
        "status": "finalized",
        "result": "yes" if yes else "no",
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "series_ticker": None,
        "event_ticker": None,
        "volume": None,
        "open_interest": None,
        "source": "nws_cli_backfill",
    }


def _scoped_trades(raw_trades: list[dict], strategies: set[str] | None) -> list[dict]:
    if strategies is None:
        return raw_trades
    return [t for t in raw_trades if (t.get("strategy") or "") in strategies]


def backfill_missing(
    cached: dict[str, dict],
    raw_trades: list[dict],
    settlements: dict[tuple[str, str], dict],
    *,
    strategies: set[str] | None = None,
) -> tuple[dict[str, dict], int, int]:
    """Add reconstructed outcomes for tickers the store has no settled result for.

    NEVER overwrites a ticker that is already settled in ``cached`` — Kalshi's
    own record always wins over a derived one. Additive and idempotent:
    running this twice on the same inputs produces the same store.

    ``strategies``, when given, restricts reconstruction to trades whose
    top-level ``strategy`` field is in the set — an unscoped run (``None``,
    the default) will attempt to reconstruct EVERY strategy present in
    ``raw_trades``, which can be a much larger change than intended.

    Returns ``(merged_store, n_reconstructed, n_skipped)``.
    """
    merged = dict(cached)
    seen: set[str] = set()
    n_reconstructed = 0
    n_skipped = 0
    for trade in _scoped_trades(raw_trades, strategies):
        ticker = trade.get("market_ticker") or ""
        if not ticker or ticker in seen:
            continue
        if is_settled(merged.get(ticker, {})):
            continue
        seen.add(ticker)
        entry = reconstruct_outcome(trade, settlements)
        if entry is None:
            n_skipped += 1
            continue
        merged[ticker] = entry
        n_reconstructed += 1
    return merged, n_reconstructed, n_skipped


def _checkable_settled_trades(
    cached: dict[str, dict],
    raw_trades: list[dict],
    settlements: dict[tuple[str, str], dict],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """One ``(trade, cached_entry, derived_entry)`` per unique already-settled
    ticker the derivation can reproduce. Shared base for ``find_disagreements``
    (mismatch detection) and ``checked_trades_by_month`` (coverage reporting)
    so the two can never disagree about what "checked" means.
    """
    seen: set[str] = set()
    out: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for trade in raw_trades:
        ticker = trade.get("market_ticker") or ""
        if not ticker or ticker in seen:
            continue
        cached_entry = cached.get(ticker, {})
        if not is_settled(cached_entry):
            continue
        seen.add(ticker)
        derived = reconstruct_outcome(trade, settlements)
        if derived is None:
            continue
        out.append((trade, cached_entry, derived))
    return out


def checked_trades_by_month(
    cached: dict[str, dict],
    raw_trades: list[dict],
    settlements: dict[tuple[str, str], dict],
) -> dict[str, int]:
    """Month-bucketed (``YYYY-MM``, by resolved forecast date) counts of the
    tickers ``find_disagreements`` actually re-derives.

    The agreement check's "N checked, 0 disagreements" headline is only as
    strong as its coverage of the population being reconstructed. On the live
    corpus the checkable population skews heavily to Jul-Sep (the store's
    settled months), while the reconstruction population is May-Jun — this
    lets an operator see that gap before trusting the headline. Compare
    against ``reconstructable_trades_by_month``.
    """
    return _month_counts(
        [trade for trade, _, _ in _checkable_settled_trades(cached, raw_trades, settlements)]
    )


def reconstructable_trades_by_month(
    cached: dict[str, dict],
    raw_trades: list[dict],
    settlements: dict[tuple[str, str], dict],
    *,
    strategies: set[str] | None = None,
) -> dict[str, int]:
    """Month-bucketed (``YYYY-MM``) counts of tickers ``backfill_missing``
    would reconstruct — the population the agreement check's coverage
    (``checked_trades_by_month``) should be judged against.
    """
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for trade in _scoped_trades(raw_trades, strategies):
        ticker = trade.get("market_ticker") or ""
        if not ticker or ticker in seen:
            continue
        if is_settled(cached.get(ticker, {})):
            continue
        seen.add(ticker)
        if reconstruct_outcome(trade, settlements) is None:
            continue
        month = _month_of(trade)
        if month is None:
            continue
        counts[month] = counts.get(month, 0) + 1
    return counts


def find_disagreements(
    cached: dict[str, dict],
    raw_trades: list[dict],
    settlements: dict[tuple[str, str], dict],
) -> tuple[list[tuple[str, str, str]], int]:
    """Re-derive the outcome for every ALREADY-settled ticker and report any
    mismatch against the real Kalshi result.

    This is the only forward check available that the derivation is
    trustworthy for tickers we CANNOT independently verify (the ones that
    are actually missing). Run across ALL strategies regardless of the
    ``--strategy`` scope — agreement on a weather_edge ticker is just as much
    evidence the derivation works as agreement on a favorite_bias one.

    Its coverage of the population actually being reconstructed can be thin
    — see ``checked_trades_by_month`` / ``reconstructable_trades_by_month``,
    which ``main()`` prints alongside this function's result.

    Returns ``(mismatches, n_checked)`` where each mismatch is
    ``(ticker, kalshi_result, derived_result)``; empty ``mismatches`` means
    the derivation agrees everywhere it COULD be checked.
    """
    checked = _checkable_settled_trades(cached, raw_trades, settlements)
    mismatches = [
        (trade.get("market_ticker") or "", cached_entry.get("result", ""), derived["result"])
        for trade, cached_entry, derived in checked
        if derived["result"] != cached_entry.get("result")
    ]
    return mismatches, len(checked)


def _load_raw_trades(trades_path: Path) -> tuple[list[dict], list[dict]]:
    """Read a JSONL trade log, returning ``(raw_trades, all_rows)``.

    ``raw_trades`` holds only ``is_pnl_relevant_row`` fills (what
    ``backfill_missing``/``find_disagreements`` operate on). ``all_rows`` holds
    every parsed entry, placement rows included — the join source
    ``build_strategy_index`` needs to resolve a reliable ``strategy`` for
    reconciler-backfilled fills, whose own top-level ``strategy`` field can be
    missing or unreliable (mirrors ``fetch_shadow_settlements.py``'s ``main()``).

    Tolerates a truncated last line — trades_path is often a fresh `scp` pull
    of a file the live bot is actively appending to (see load_buy_rows in
    favorite_bias_report.py for the same guard on the same race).
    """
    raw_trades: list[dict] = []
    all_rows: list[dict] = []
    n_bad_lines = 0
    for line in trades_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            n_bad_lines += 1
            continue
        all_rows.append(entry)
        if is_pnl_relevant_row(entry):
            raw_trades.append(entry)
    if n_bad_lines:
        print(f"  (skipped {n_bad_lines} unparseable line(s) — likely a mid-write truncation)")
    return raw_trades, all_rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Backfill lost pre-append-only settlement outcomes from NWS-CLI snapshots"
    )
    p.add_argument("--trades", default=str(DEFAULT_TRADES))
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    p.add_argument(
        "--strategy", action="append", default=None,
        help="Restrict reconstruction to this strategy (repeatable). "
             "Omitting this reconstructs EVERY strategy in the trade log.",
    )
    p.add_argument(
        "--write", action="store_true", help="Persist the merged store (default: preview only)"
    )
    args = p.parse_args(argv)

    trades_path = Path(args.trades)
    store_path = Path(args.store)
    snapshots_dir = Path(args.snapshots)
    strategies = set(args.strategy) if args.strategy else None

    if not trades_path.exists():
        print(f"Error: {trades_path} not found")
        sys.exit(1)
    if strategies is None:
        print(
            "Warning: no --strategy filter given — this will attempt to reconstruct "
            "outcomes for EVERY strategy present in the trade log, not just one."
        )

    raw_trades, all_rows = _load_raw_trades(trades_path)
    strategy_index = build_strategy_index(all_rows)
    apply_strategy_enrichment(raw_trades, strategy_index)
    cached = load_settlement_store(store_path)
    _forecasts, settlements, _markets = load_snapshot_data(snapshots_dir)

    mismatches, n_checked = find_disagreements(cached, raw_trades, settlements)
    checked_months = _format_month_counts(
        checked_trades_by_month(cached, raw_trades, settlements)
    )
    checked_suffix = f" ({checked_months})" if checked_months else ""
    print(
        f"Agreement check: {n_checked} checked{checked_suffix}; "
        f"{len(mismatches)} disagreement(s)."
    )
    if mismatches:
        print(
            "ABORT: the derivation disagrees with a known Kalshi result — "
            "refusing to write anything."
        )
        for ticker, kalshi_result, derived_result in mismatches[:20]:
            print(f"  {ticker}: kalshi={kalshi_result} derived={derived_result}")
        sys.exit(1)

    recon_months = _format_month_counts(
        reconstructable_trades_by_month(cached, raw_trades, settlements, strategies=strategies)
    )
    recon_suffix = f" ({recon_months})" if recon_months else ""
    merged, n_reconstructed, n_skipped = backfill_missing(
        cached, raw_trades, settlements, strategies=strategies
    )
    print(
        f"Reconstructing {n_reconstructed} outcomes{recon_suffix}; "
        f"{n_skipped} tickers had no snapshot match."
    )

    if not args.write:
        print("Preview only (pass --write to persist).")
        return

    if store_path.exists():
        backup_path = store_path.with_name(
            store_path.name + f".bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        backup_path.write_text(store_path.read_text())
        print(f"Backed up existing store to {backup_path}")

    atomic_write_text(store_path, json.dumps(merged, indent=2))
    print(f"Wrote {len(merged)} entries to {store_path}")
    print("Next: uv run python scripts/favorite_bias_report.py --refresh  (regenerates .full.json)")


if __name__ == "__main__":
    main()
