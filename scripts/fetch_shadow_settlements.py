"""
Fetch Kalshi settlement results for all shadow trades.

Queries the Kalshi API for each unique market ticker in the shadow trade log,
gets the settlement result (yes/no) and strike metadata, then re-runs the
full P&L analysis with complete data.

The settlement store is append-only: Kalshi 404s markets older than roughly 70
days, so a plain overwrite silently re-emits every older trade as
``settled: false`` and turns cumulative cohorts into a moving window (per-cell
``n`` can go *down* between runs). ``merge_settlements`` keeps any cached
settled outcome that the API can no longer serve. Outcomes lost before this
became append-only are not recoverable from Kalshi; they would have to be
rebuilt from ``data/snapshots/<date>/settlements.json`` (NWS finals plus the
strike already on each row).

Usage:
    uv run python scripts/fetch_shadow_settlements.py [--trades PATH]
    uv run python scripts/fetch_shadow_settlements.py --pull-vps
    uv run python scripts/fetch_shadow_settlements.py --trades data/vps_trades.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.fees import order_fee  # noqa: E402
from python.analytics.strategy_attribution import (  # noqa: E402
    build_strategy_index,
    resolve_strategy_by_order_id,
)
from python.data.kalshi_client import KalshiClient  # noqa: E402
from python.execution.trade_log import is_pnl_relevant_row  # noqa: E402
from python.settings import Settings  # noqa: E402
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date  # noqa: E402
from scripts._vps import VPS_TRADES_PATH, pull_audit_files, require_vps_host  # noqa: E402

DEFAULT_AUDIT_DIR = _PROJECT_ROOT / "data" / "vps_audit"


# ---------------------------------------------------------------------------
# Settlement store
# ---------------------------------------------------------------------------


def is_settled(entry: dict) -> bool:
    """True when a settlement entry carries a terminal Kalshi outcome."""
    return entry.get("result") in ("yes", "no")


def load_settlement_store(path: Path) -> dict[str, dict]:
    """Read the existing settlement store, or ``{}`` when absent/unreadable.

    A corrupt or hand-edited store must not abort a refresh — the fetch below
    rebuilds every ticker the API still serves.
    """
    if not path.exists():
        return {}
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read existing settlement store {path}: {e}")
        return {}
    if not isinstance(cached, dict):
        print(f"Warning: settlement store {path} is not an object — ignoring")
        return {}
    return {k: v for k, v in cached.items() if isinstance(v, dict)}


def merge_settlements(cached: dict[str, dict], fresh: dict[str, dict]) -> dict[str, dict]:
    """Merge a fetch into the cached store, never losing a settled outcome.

    A Kalshi settlement is terminal, so a cached ``result`` of yes/no always
    wins over a fresh lookup that has lost it (404 → absent from ``fresh``, or
    a response without a result). Everything else takes the fresh value.
    """
    merged = dict(cached)
    for ticker, entry in fresh.items():
        if is_settled(merged.get(ticker, {})) and not is_settled(entry):
            continue
        merged[ticker] = entry
    return merged


# ---------------------------------------------------------------------------
# Row-shape helpers
# ---------------------------------------------------------------------------


def _entry_price(row: dict) -> float:
    """Resolve the per-contract entry price for a trade-log row.

    Engine-emitted fills carry ``order_price`` (and ``exec_price``).
    FillsReconciler-backfilled rows (``metadata.reconcile_source ==
    "fills_reconciler"``, shipped 2026-05-07) carry only ``fill_price`` —
    they have no synthetic ``order_price``.

    Prefer ``fill_price`` when present (it reflects actual Kalshi fill
    economics for live/paper trades), then fall back to ``order_price``.
    Raise ``KeyError`` with a clear message if the row has neither — that
    would indicate a genuine schema regression worth surfacing rather than
    silently zeroing out.
    """
    if "fill_price" in row and row["fill_price"] is not None:
        return float(row["fill_price"])
    if "order_price" in row and row["order_price"] is not None:
        return float(row["order_price"])
    raise KeyError(
        f"trade row missing both fill_price and order_price: "
        f"order_id={row.get('order_id')!r} ticker={row.get('market_ticker')!r}"
    )


def apply_strategy_enrichment(results: list[dict], strategy_index: dict[str, str]) -> None:
    """Stamp a reliable ``strategy`` on each result row, in place.

    Uses the FULL-log ``strategy_index`` (placement + fill rows) to recover the
    true strategy for reconciler-backfilled fills whose own ``strategy`` is
    missing. Rows absent from the index keep their own top-level strategy, or
    fall back to ``"unknown"`` — legacy untagged rows are never coerced. Makes
    ``shadow_settlements.full.json`` strategy-reliable for all downstream
    consumers.
    """
    for r in results:
        oid = r.get("order_id") or ""
        own = r.get("strategy")
        fallback = own if isinstance(own, str) and own else "unknown"
        r["strategy"] = resolve_strategy_by_order_id(oid, strategy_index, fallback=fallback)


def apply_metadata_enrichment(results: list[dict]) -> None:
    """Backfill ``metadata.city`` / ``market_type`` / ``forecast_date`` from the
    ticker, in place, only where missing.

    Reconciler-backfilled (synthetic) fills carry a thin ``metadata`` dict that
    lacks the signal-time city/var/forecast_date, which collapses them into the
    daily report's ``"/"`` cell and a blank-city settlement row. These three
    fields are deterministically recoverable from the Kalshi ticker, so we
    re-derive them here. Existing metadata values are never overwritten — a
    real signal row keeps its own attribution.
    """
    for r in results:
        ticker = r.get("market_ticker") or ""
        meta = r.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            r["metadata"] = meta

        if not meta.get("city") or not meta.get("market_type"):
            city_var = parse_ticker_city_var(ticker)
            if city_var is not None:
                city, var = city_var
                meta.setdefault("city", city)
                # build_settled_report reads var from metadata.market_type.
                meta.setdefault("market_type", var)

        if not meta.get("forecast_date"):
            fc = parse_ticker_date(ticker)
            if fc is not None:
                meta.setdefault("forecast_date", fc.isoformat())


# ---------------------------------------------------------------------------
# P&L calculation
# ---------------------------------------------------------------------------


def compute_pnl(
    side: str,
    order_price: float,
    contracts: int,
    market_result: str,
    *,
    fees_from_api: float | None = None,
    is_index_market: bool = False,
) -> tuple[float, float, bool]:
    """Compute P&L. Returns (pnl, fee, won).

    Prefers Kalshi-reported ``fees_from_api`` whenever the field is present
    (including authoritative-zero, e.g. shadow trades or maker rebates).
    Falls back to ``order_fee`` recompute only for legacy pre-2026-04-27
    rows where the field is absent (None).
    """
    if fees_from_api is not None:
        # Field present — authoritative even when value is 0.0.
        # Shadow trades pay zero fees; live trades report exact fees from
        # /portfolio/fills. A zero here is correct, not a sentinel for "missing".
        fee = fees_from_api
    else:
        # Legacy pre-2026-04-27 rows: best-effort recompute.
        fee = order_fee(order_price, contracts, is_index_market=is_index_market)
    won = (
        (side == "YES" and market_result == "yes")
        or (side == "NO" and market_result == "no")
    )
    if won:
        pnl = (1.0 - order_price) * contracts - fee
    else:
        pnl = -(order_price * contracts) - fee
    return pnl, fee, won


def compute_fee_charge_overrides(raw_trades: list[dict]) -> dict[int, float]:
    """Zero the duplicated fee-charge on partial-fill rows of the same order.

    ``OrderRecord.fees_from_api`` (engine.py:700) is the CUMULATIVE fee for
    the whole order at each fill event, but every fill-row log entry for
    that order is stamped with that same cumulative value
    (engine.py:2196-2224). A multi-fill order's true total fee is therefore
    counted more than once by ``compute_pnl``, which charges
    ``fees_from_api`` once per row.

    Fix: for orders with more than one fee-bearing row, the charge is kept
    only on the LAST-BY-TIMESTAMP row (ties broken by original list order via
    a stable sort) — matching the repo's existing convention for this exact
    field in ``scripts/pull_post_mortem_data.py``'s ``aggregate_local_orders``
    (``fee_total = last_row.fees_from_api``). NOT the row with the numerically
    largest value: ``fees_from_api`` is not guaranteed monotonic —
    ``python/execution/fills_reconciler.py`` can legitimately DECREASE it on a
    partial cash-out — so "last" and "max" can disagree, and "last" is the one
    that reflects the order's true final state.

    Returns ``{row_index: 0.0}`` for every row that must NOT be charged; a row
    absent from the returned dict keeps its own ``fees_from_api``. Does not
    mutate ``raw_trades`` — callers must apply the override only when
    computing P&L, leaving the row's own ``fees_from_api`` field (and
    everything else persisted about the row) exactly as the engine wrote it.
    """
    by_order: dict[str, list[int]] = {}
    for i, r in enumerate(raw_trades):
        oid = r.get("order_id")
        if oid and r.get("fees_from_api") is not None:
            by_order.setdefault(oid, []).append(i)

    overrides: dict[int, float] = {}
    for idxs in by_order.values():
        if len(idxs) < 2:
            continue
        keep = sorted(idxs, key=lambda i: raw_trades[i].get("timestamp", ""))[-1]
        for i in idxs:
            if i != keep:
                overrides[i] = 0.0
    return overrides


def build_result_row(
    trade: dict, settlement: dict, *, fees_override: float | None = None
) -> dict:
    """Build one ``.full.json`` result row for a trade against its settlement.

    Extracted from ``main()``'s per-trade P&L loop so tests can exercise the
    real row-construction logic (including ``settlement_source``) instead of
    hand-reconstructing the row shape. ``fees_override``, when not ``None``,
    is used instead of ``trade["fees_from_api"]`` — see
    ``compute_fee_charge_overrides`` (0.0 is a valid override, so this checks
    ``is not None`` rather than truthiness).
    """
    market_result = settlement.get("result")

    if market_result not in ("yes", "no"):
        return {
            **trade,
            "market_result": None,
            "strike_type": settlement.get("strike_type"),
            "pnl": None,
            "fee": None,
            "won": None,
            "settled": False,
        }

    # Prefer actual fill_price (live/paper + reconciler-backfilled rows)
    # over order_price (shadow / engine-emitted legacy rows). Reconciler
    # rows (metadata.reconcile_source == "fills_reconciler", shipped
    # 2026-05-07) carry only fill_price.
    price = _entry_price(trade)

    fees = fees_override if fees_override is not None else trade.get("fees_from_api")
    is_index = trade["market_ticker"].startswith(("KXINX", "KXNDX"))
    pnl, fee, won = compute_pnl(
        side=trade["side"],
        order_price=price,
        contracts=trade["contracts"],
        market_result=market_result,
        fees_from_api=fees,
        is_index_market=is_index,
    )

    # Normalise order_price on the result dict so downstream consumers
    # (print_analysis, shadow_dashboard reading the .full.json) can rely
    # on the field being present even for reconciler-backfilled rows.
    return {
        **trade,
        "order_price": price,
        "market_result": market_result,
        "strike_type": settlement.get("strike_type"),
        "floor_strike": settlement.get("floor_strike"),
        "cap_strike": settlement.get("cap_strike"),
        "pnl": round(pnl, 2),
        "fee": round(fee, 2),
        "won": won,
        "settled": True,
        # "kalshi" for a real settlement result; "nws_cli_backfill" for one
        # reconstructed by backfill_settlements_from_snapshots.py — see
        # that script's reconstruct_outcome, which stamps "source" onto
        # the settlement-store entry this reads.
        "settlement_source": settlement.get("source", "kalshi"),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def get_city(r: dict) -> str:
    """Extract city code from a result dict, or "???" if unresolvable."""
    city_var = parse_ticker_city_var(r.get("market_ticker") or "")
    return city_var[0] if city_var is not None else "???"


def print_analysis(settled_results: list[dict], all_results: list[dict]) -> None:
    """Print comprehensive P&L analysis for settled shadow trades."""
    unsettled_results = [r for r in all_results if not r["settled"]]

    print("\n" + "=" * 80)
    print("SHADOW TRADING P&L ANALYSIS (Kalshi Settlement Data)")
    print("=" * 80)

    print(f"\nTotal trades: {len(all_results)}")
    print(f"Settled: {len(settled_results)}")
    print(f"Pending: {len(unsettled_results)}")

    if not settled_results:
        print("\nNo settled trades to analyze.")
        return

    # --- Overall ---
    total_pnl = sum(r["pnl"] for r in settled_results)
    total_fees = sum(r["fee"] for r in settled_results)
    total_wagered = sum(r["order_price"] * r["contracts"] for r in settled_results)
    wins = [r for r in settled_results if r["won"]]
    losses = [r for r in settled_results if not r["won"]]
    win_rate = len(wins) / len(settled_results) * 100

    print("\n--- OVERALL ---")
    print(f"Net P&L:        ${total_pnl:>10,.2f}")
    print(f"Total fees:     ${total_fees:>10,.2f}")
    print(f"Total wagered:  ${total_wagered:>10,.2f}")
    if total_wagered:
        print(f"ROI:            {total_pnl / total_wagered * 100:>9.1f}%")
    print(f"Win rate:       {win_rate:>9.1f}% ({len(wins)}/{len(settled_results)})")
    if wins:
        print(f"Avg win:        ${sum(r['pnl'] for r in wins) / len(wins):>10,.2f}")
    if losses:
        print(f"Avg loss:       ${sum(r['pnl'] for r in losses) / len(losses):>10,.2f}")

    # --- Exclude first day ---
    first_day = min(r["timestamp"][:10] for r in settled_results)
    first_day_results = [r for r in settled_results if r["timestamp"][:10] == first_day]
    post_first = [r for r in settled_results if r["timestamp"][:10] != first_day]

    if post_first:
        fd_pnl = sum(r["pnl"] for r in first_day_results)
        pf_pnl = sum(r["pnl"] for r in post_first)
        pf_wins = sum(1 for r in post_first if r["won"])
        pf_wagered = sum(r["order_price"] * r["contracts"] for r in post_first)
        print("\n--- FIRST DAY vs REST ---")
        print(f"Day 1 ({first_day}): {len(first_day_results)} trades, P&L ${fd_pnl:,.2f}")
        print(
            f"Rest:               {len(post_first)} trades, P&L ${pf_pnl:,.2f}, "
            f"WR {pf_wins}/{len(post_first)} ({pf_wins / len(post_first) * 100:.1f}%)"
        )
        if pf_wagered:
            print(f"Rest ROI:           {pf_pnl / pf_wagered * 100:.1f}%")

    # --- By city ---
    print("\n--- BY CITY ---")
    cities: dict[str, list[dict]] = defaultdict(list)
    for r in settled_results:
        cities[get_city(r)].append(r)

    hdr = (
        f"{'City':<6} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'P&L':>10} {'Wagered':>10} {'ROI%':>7}"
    )
    print(hdr)
    for city in sorted(cities.keys()):
        cr = cities[city]
        cpnl = sum(r["pnl"] for r in cr)
        cwag = sum(r["order_price"] * r["contracts"] for r in cr)
        cwins = sum(1 for r in cr if r["won"])
        cwr = cwins / len(cr) * 100
        roi = cpnl / cwag * 100 if cwag else 0
        print(
            f"{city:<6} {len(cr):>6} {cwins:>5} {cwr:>5.1f}%"
            f" ${cpnl:>9,.2f} ${cwag:>9,.2f} {roi:>6.1f}%"
        )

    # --- By city (excluding day 1) ---
    if post_first:
        print("\n--- BY CITY (excl. day 1) ---")
        cities_pf: dict[str, list[dict]] = defaultdict(list)
        for r in post_first:
            cities_pf[get_city(r)].append(r)

        print(hdr)
        for city in sorted(cities_pf.keys()):
            cr = cities_pf[city]
            cpnl = sum(r["pnl"] for r in cr)
            cwag = sum(r["order_price"] * r["contracts"] for r in cr)
            cwins = sum(1 for r in cr if r["won"])
            cwr = cwins / len(cr) * 100
            roi = cpnl / cwag * 100 if cwag else 0
            print(
                f"{city:<6} {len(cr):>6} {cwins:>5} {cwr:>5.1f}%"
                f" ${cpnl:>9,.2f} ${cwag:>9,.2f} {roi:>6.1f}%"
            )

    # --- By market date ---
    print("\n--- BY MARKET DATE ---")
    dates: dict[str, list[dict]] = defaultdict(list)
    for r in settled_results:
        # Extract date from ticker: KXHIGHNY-26MAR30-T62 -> 26MAR30
        parts = r["market_ticker"].split("-")
        if len(parts) >= 2:
            dates[parts[1]].append(r)

    print(f"{'Date':<10} {'Trades':>6} {'Wins':>5} {'P&L':>10}")
    for d in sorted(dates.keys()):
        dr = dates[d]
        dpnl = sum(r["pnl"] for r in dr)
        dwins = sum(1 for r in dr if r["won"])
        print(f"{d:<10} {len(dr):>6} {dwins:>5} ${dpnl:>9,.2f}")

    # --- By side ---
    print("\n--- BY SIDE ---")
    for side_label in ("YES", "NO"):
        sr = [r for r in settled_results if r["side"] == side_label]
        if sr:
            spnl = sum(r["pnl"] for r in sr)
            swins = sum(1 for r in sr if r["won"])
            swr = swins / len(sr) * 100
            print(f"{side_label}: {len(sr)} trades, {swins} wins ({swr:.1f}%), P&L ${spnl:,.2f}")

    # --- By strike type (from Kalshi API) ---
    print("\n--- BY STRIKE TYPE ---")
    for st in ("greater", "less", "between"):
        str_results = [r for r in settled_results if r.get("strike_type") == st]
        if str_results:
            stpnl = sum(r["pnl"] for r in str_results)
            stwins = sum(1 for r in str_results if r["won"])
            stwr = stwins / len(str_results) * 100
            print(
                f"{st:>8}: {len(str_results)} trades, "
                f"{stwins} wins ({stwr:.1f}%), P&L ${stpnl:,.2f}"
            )

    # --- Edge calibration ---
    print("\n--- EDGE CALIBRATION ---")
    edge_buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
    edge_hdr = f"{'Edge Range':<12} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'Avg Edge':>9}"
    print(edge_hdr)
    for lo, hi in edge_buckets:
        bucket = [r for r in settled_results if lo <= (r.get("edge") or 0) < hi]
        if bucket:
            bpnl = sum(r["pnl"] for r in bucket)
            bwins = sum(1 for r in bucket if r["won"])
            bwr = bwins / len(bucket) * 100
            avg_edge = sum(r.get("edge") or 0 for r in bucket) / len(bucket)
            print(
                f"{lo:.0%}-{hi:.0%}      {len(bucket):>6}"
                f" {bwr:>5.1f}% ${bpnl:>9,.2f} {avg_edge:>8.1%}"
            )

    # --- Edge calibration (excl. day 1) ---
    if post_first:
        print("\n--- EDGE CALIBRATION (excl. day 1) ---")
        print(edge_hdr)
        for lo, hi in edge_buckets:
            bucket = [r for r in post_first if lo <= (r.get("edge") or 0) < hi]
            if bucket:
                bpnl = sum(r["pnl"] for r in bucket)
                bwins = sum(1 for r in bucket if r["won"])
                bwr = bwins / len(bucket) * 100
                avg_edge = sum(r.get("edge") or 0 for r in bucket) / len(bucket)
                print(
                    f"{lo:.0%}-{hi:.0%}      {len(bucket):>6}"
                    f" {bwr:>5.1f}% ${bpnl:>9,.2f} {avg_edge:>8.1%}"
                )

    # --- Top winners/losers ---
    # Reconciler-backfilled rows (shipped 2026-05-07) lack signal-side
    # metadata (model_probability, market_price, edge are all None), so
    # format defensively rather than crash on None values.
    def _fmt_optional_pct(value: object) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "  n/a"

    sorted_by_pnl = sorted(settled_results, key=lambda r: r["pnl"], reverse=True)
    print("\n--- TOP 10 WINNERS ---")
    for r in sorted_by_pnl[:10]:
        print(
            f"  {r['market_ticker']} {r['side']} @ {r['order_price']:.3f} "
            f"({r['contracts']}c) -> {r['market_result'].upper()} "
            f"P&L=${r['pnl']:,.2f}  edge={(r.get('edge') or 0):.1%}  "
            f"model={_fmt_optional_pct(r.get('model_probability'))} "
            f"mkt={_fmt_optional_pct(r.get('market_price'))}"
        )

    print("\n--- TOP 10 LOSERS ---")
    for r in sorted_by_pnl[-10:]:
        print(
            f"  {r['market_ticker']} {r['side']} @ {r['order_price']:.3f} "
            f"({r['contracts']}c) -> {r['market_result'].upper()} "
            f"P&L=${r['pnl']:,.2f}  edge={(r.get('edge') or 0):.1%}  "
            f"model={_fmt_optional_pct(r.get('model_probability'))} "
            f"mkt={_fmt_optional_pct(r.get('market_price'))}"
        )

    # --- Cumulative P&L ---
    print("\n--- CUMULATIVE P&L (chronological) ---")
    # Sort by timestamp
    chrono = sorted(settled_results, key=lambda r: r["timestamp"])
    cum = 0.0
    for i, r in enumerate(chrono, 1):
        cum += r["pnl"]
        marker = "+" if r["pnl"] > 0 else "-"
        if i <= 15 or i > len(chrono) - 5 or i % 10 == 0:
            d = r["timestamp"][:10]
            print(
                f"  {i:>3}. [{d}] {marker}${abs(r['pnl']):>8,.2f}"
                f"  cum=${cum:>10,.2f}  {r['market_ticker']}"
            )
        elif i == 16:
            print("  ...")


# ---------------------------------------------------------------------------
# VPS data pull
# ---------------------------------------------------------------------------


def pull_trades_from_vps(local_path: Path, audit_dir: Path | None = None) -> None:
    """SCP shadow trades + optional diagnostic audit streams from VPS."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", f"{require_vps_host()}:{VPS_TRADES_PATH}", str(local_path)]
    print(f"Pulling trades from VPS: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error pulling from VPS: {result.stderr.strip()}")
        sys.exit(1)
    print(f"Saved to {local_path}")

    if audit_dir is not None:
        pull_audit_files(audit_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Kalshi settlements for shadow trades")
    parser.add_argument(
        "--trades",
        default=str(_PROJECT_ROOT / "data" / "vps_trades.jsonl"),
        help="Path to trades.jsonl (default: data/vps_trades.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "data" / "shadow_settlements.json"),
        help="Output path for settlement data (default: data/shadow_settlements.json)",
    )
    parser.add_argument(
        "--pull-vps",
        action="store_true",
        help="SCP trades.jsonl + diagnostic audit streams from VPS before processing",
    )
    parser.add_argument(
        "--audit-dir",
        default=str(DEFAULT_AUDIT_DIR),
        help=(
            "Local directory for the 3 diagnostic audit JSONL files fetched "
            "with --pull-vps (default: data/vps_audit/)"
        ),
    )
    parser.add_argument(
        "--skip-audit-pull",
        action="store_true",
        help="Skip pulling eval_audit/candidate_audit/scan_funnel even when --pull-vps is set",
    )
    args = parser.parse_args()

    trades_path = Path(args.trades)

    # Pull from VPS if requested
    if args.pull_vps:
        audit_dir: Path | None = None if args.skip_audit_pull else Path(args.audit_dir)
        pull_trades_from_vps(trades_path, audit_dir=audit_dir)

    # Load shadow trades
    if not trades_path.exists():
        print(f"Error: {trades_path} not found")
        print("  Hint: use --pull-vps to fetch from VPS, or --trades to specify a local path")
        sys.exit(1)

    raw_trades: list[dict] = []
    all_rows: list[dict] = []
    with open(trades_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Keep ALL rows for strategy attribution: placement rows are the
            # durable top-level-strategy join source for reconciler-backfilled
            # fills (see build_strategy_index).
            all_rows.append(entry)
            # P&L is computed from fill rows only. Skip non-fill rows (e.g.
            # settlement markers, placement rows). Legacy rows with no "event"
            # field are treated as fills.
            if not is_pnl_relevant_row(entry):
                continue
            raw_trades.append(entry)

    print(f"Loaded {len(raw_trades)} shadow trades")
    strategy_index = build_strategy_index(all_rows)

    # Get unique tickers
    tickers = sorted(set(t["market_ticker"] for t in raw_trades))
    print(f"Unique market tickers: {len(tickers)}")

    # Load settings for Kalshi auth
    settings = Settings.load()
    if not settings.kalshi_api_key or not settings.kalshi_private_key_path:
        print("Error: ORACLE_KALSHI_API_KEY and ORACLE_KALSHI_PRIVATE_KEY_PATH required in .env")
        sys.exit(1)

    output_path = Path(args.output)
    cached_settlements = load_settlement_store(output_path)
    if cached_settlements:
        print(f"Loaded {len(cached_settlements)} cached settlements from {output_path}")

    # Fetch settlement results from Kalshi
    print("\nFetching settlement data from Kalshi API...")
    settlements: dict[str, dict] = {}
    errors: list[str] = []

    async with KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
    ) as client:
        for i, ticker in enumerate(tickers, 1):
            try:
                resp = await client.get_market(ticker=ticker)
                market = resp.get("market", resp)  # handle nested or flat response
                settlements[ticker] = {
                    "ticker": ticker,
                    "status": market.get("status"),
                    "result": market.get("result"),
                    "strike_type": market.get("strike_type"),
                    "floor_strike": market.get("floor_strike"),
                    "cap_strike": market.get("cap_strike"),
                    "series_ticker": market.get("series_ticker"),
                    "event_ticker": market.get("event_ticker"),
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest"),
                }
                status = market.get("status", "?")
                result = market.get("result", "n/a")
                print(f"  [{i}/{len(tickers)}] {ticker}: status={status} result={result}")
            except Exception as e:
                errors.append(f"{ticker}: {e}")
                print(f"  [{i}/{len(tickers)}] {ticker}: ERROR -- {e}")

            # Rate limit: ~2 requests/sec to be safe
            if i % 5 == 0:
                await asyncio.sleep(1.0)

    # Save raw settlement data
    fetched_settled = {t for t, v in settlements.items() if is_settled(v)}
    settlements = merge_settlements(cached_settlements, settlements)
    output_path.write_text(json.dumps(settlements, indent=2))
    print(f"\nSettlement data saved to {output_path}")
    if errors:
        print(f"  Errors: {len(errors)}")

    # Count settlements over this run's tickers only — the store also holds
    # aged-out markets from earlier runs that the API no longer serves.
    in_run = {t: settlements.get(t, {}) for t in tickers}
    settled = {k: v for k, v in in_run.items() if is_settled(v)}
    unsettled = {k: v for k, v in in_run.items() if not is_settled(v)}
    preserved = sum(1 for t in settled if t not in fetched_settled)
    print(f"  Settled: {len(settled)}")
    if preserved:
        print(f"    (of which {preserved} preserved from cache — not settled by this fetch)")
    print(f"  Not yet settled: {len(unsettled)}")
    if unsettled:
        for t, v in sorted(unsettled.items()):
            print(f"    {t}: status={v.get('status')}")

    # ---------------------------------------------------------------------------
    # Compute P&L for all trades using Kalshi settlement data
    # ---------------------------------------------------------------------------

    fee_overrides = compute_fee_charge_overrides(raw_trades)

    results: list[dict] = []
    for idx, trade in enumerate(raw_trades):
        settlement = settlements.get(trade["market_ticker"], {})
        # idx in fee_overrides means this row's cumulative fees_from_api was
        # already charged on a LATER row of the same order — see
        # compute_fee_charge_overrides. The row's own persisted
        # fees_from_api field (spread via **trade in build_result_row) is
        # left untouched.
        results.append(
            build_result_row(trade, settlement, fees_override=fee_overrides.get(idx))
        )

    # Stamp strategy on every row (recovers reconciler-backfilled fills).
    apply_strategy_enrichment(results, strategy_index)
    # Backfill city/var/forecast_date from the ticker for thin synthetic rows.
    apply_metadata_enrichment(results)

    # Split settled vs unsettled and print analysis
    settled_results = [r for r in results if r["settled"]]

    # Save full results BEFORE rendering analysis. print_analysis is rendering,
    # not data production: a crash there must not silently freeze the
    # artifact. Downstream consumers (notably scripts/daily_trade_report.py)
    # read this file as their source of truth for settlement state, so a
    # stale .full.json is operationally invisible until the next post-mortem.
    full_output = Path(args.output).with_suffix(".full.json")
    full_output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to {full_output}")

    print_analysis(settled_results, results)


if __name__ == "__main__":
    asyncio.run(main())
