"""
Shadow trading daily P&L dashboard.

Fast, local dashboard that reads pre-computed settlement data and displays
daily performance metrics. Optionally pulls fresh data from VPS and/or
refreshes settlements from Kalshi API.

Usage:
    # Quick view (reads cached data, instant)
    uv run python scripts/shadow_dashboard.py

    # Pull latest trades from VPS first
    uv run python scripts/shadow_dashboard.py --pull-vps

    # Full refresh: pull VPS trades + fetch settlements from Kalshi
    uv run python scripts/shadow_dashboard.py --pull-vps --refresh

    # Export structured JSON report
    uv run python scripts/shadow_dashboard.py --json data/dashboard_report.json

    # Combine: refresh everything and export JSON
    uv run python scripts/shadow_dashboard.py --pull-vps --refresh --json data/dashboard_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.utils.ticker_dates import parse_ticker_city_var  # noqa: E402
from scripts._vps import VPS_TRADES_PATH, pull_audit_files, require_vps_host  # noqa: E402

DEFAULT_TRADES = _PROJECT_ROOT / "data" / "vps_trades.jsonl"
DEFAULT_FULL = _PROJECT_ROOT / "data" / "shadow_settlements.full.json"
DEFAULT_AUDIT_DIR = _PROJECT_ROOT / "data" / "vps_audit"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def pull_from_vps(local_path: Path, audit_dir: Path | None = None) -> None:
    """SCP shadow trades from VPS + optionally the diagnostic audit JSONLs."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", f"{require_vps_host()}:{VPS_TRADES_PATH}", str(local_path)]
    print("Pulling trades from VPS...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        sys.exit(1)
    n = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    print(f"  -> {n} trades saved to {local_path}")

    if audit_dir is not None:
        pull_audit_files(audit_dir)


def refresh_settlements(trades_path: Path) -> None:
    """Run fetch_shadow_settlements.py to get fresh Kalshi data."""
    script = _PROJECT_ROOT / "scripts" / "fetch_shadow_settlements.py"
    cmd = ["uv", "run", "python", str(script), "--trades", str(trades_path)]
    print("Refreshing settlement data from Kalshi API...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Warning: settlement refresh failed")
        sys.exit(1)


def load_full_results(path: Path) -> list[dict]:
    """Load the pre-computed full results JSON."""
    if not path.exists():
        print(f"Error: {path} not found")
        print("  Run with --pull-vps --refresh to fetch data, or:")
        print("  uv run python scripts/fetch_shadow_settlements.py --pull-vps")
        sys.exit(1)
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def get_city(r: dict) -> str:
    """Extract city code from a result dict, or "???" if unresolvable."""
    city_var = parse_ticker_city_var(r.get("market_ticker") or "")
    return city_var[0] if city_var is not None else "???"


def get_var_type(r: dict) -> str:
    """Extract high/low from ticker."""
    ticker = r["market_ticker"]
    series = ticker.split("-")[0]
    if "HIGH" in series:
        return "high"
    if "LOWT" in series:
        return "low"
    return "?"


def trade_date(r: dict) -> str:
    """Extract the trade date (YYYY-MM-DD) from timestamp."""
    return r["timestamp"][:10]


def compute_daily(results: list[dict]) -> list[dict]:
    """Group settled results by trade date and compute daily stats."""
    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["settled"]:
            by_date[trade_date(r)].append(r)

    days = []
    cumulative_pnl = 0.0
    for date in sorted(by_date.keys()):
        dr = by_date[date]
        pnl = sum(r["pnl"] for r in dr)
        fees = sum(r["fee"] for r in dr)
        wagered = sum(r["order_price"] * r["contracts"] for r in dr)
        wins = sum(1 for r in dr if r["won"])
        losses = len(dr) - wins
        cumulative_pnl += pnl

        # By city breakdown
        city_stats: dict[str, dict] = {}
        by_city: dict[str, list[dict]] = defaultdict(list)
        for r in dr:
            by_city[get_city(r)].append(r)
        for city in sorted(by_city.keys()):
            cr = by_city[city]
            city_stats[city] = {
                "trades": len(cr),
                "wins": sum(1 for r in cr if r["won"]),
                "pnl": round(sum(r["pnl"] for r in cr), 2),
            }

        # By side breakdown
        side_stats: dict[str, dict] = {}
        for side in ("YES", "NO"):
            sr = [r for r in dr if r["side"] == side]
            if sr:
                side_stats[side] = {
                    "trades": len(sr),
                    "wins": sum(1 for r in sr if r["won"]),
                    "pnl": round(sum(r["pnl"] for r in sr), 2),
                }

        days.append(
            {
                "date": date,
                "trades": len(dr),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(wins / len(dr) * 100, 1) if dr else 0,
                "pnl": round(pnl, 2),
                "fees": round(fees, 2),
                "wagered": round(wagered, 2),
                "roi_pct": round(pnl / wagered * 100, 1) if wagered else 0,
                "cumulative_pnl": round(cumulative_pnl, 2),
                "avg_edge": round(sum(r.get("edge") or 0 for r in dr) / len(dr), 3) if dr else 0,
                "by_city": city_stats,
                "by_side": side_stats,
            }
        )

    return days


def compute_summary(results: list[dict], daily: list[dict]) -> dict:
    """Compute overall summary stats."""
    settled = [r for r in results if r["settled"]]
    pending = [r for r in results if not r["settled"]]

    if not settled:
        return {
            "total_trades": len(results),
            "settled": 0,
            "pending": len(pending),
        }

    total_pnl = sum(r["pnl"] for r in settled)
    total_fees = sum(r["fee"] for r in settled)
    total_wagered = sum(r["order_price"] * r["contracts"] for r in settled)
    wins = [r for r in settled if r["won"]]
    losses = [r for r in settled if not r["won"]]

    # Streaks
    chrono = sorted(settled, key=lambda r: r["timestamp"])
    current_streak_type = None
    current_streak_count = 0
    max_win_streak = 0
    max_loss_streak = 0
    ws = ls = 0
    for r in chrono:
        if r["won"]:
            ws += 1
            ls = 0
            max_win_streak = max(max_win_streak, ws)
        else:
            ls += 1
            ws = 0
            max_loss_streak = max(max_loss_streak, ls)
    # Current streak from the end
    for r in reversed(chrono):
        if current_streak_type is None:
            current_streak_type = "win" if r["won"] else "loss"
            current_streak_count = 1
        elif (r["won"] and current_streak_type == "win") or (
            not r["won"] and current_streak_type == "loss"
        ):
            current_streak_count += 1
        else:
            break

    # Best/worst days
    best_day = max(daily, key=lambda d: d["pnl"]) if daily else None
    worst_day = min(daily, key=lambda d: d["pnl"]) if daily else None

    # Profitable days
    profitable_days = sum(1 for d in daily if d["pnl"] > 0)

    return {
        "total_trades": len(results),
        "settled": len(settled),
        "pending": len(pending),
        "net_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round(total_pnl / total_wagered * 100, 1) if total_wagered else 0,
        "win_rate_pct": round(len(wins) / len(settled) * 100, 1),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": round(sum(r["pnl"] for r in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(r["pnl"] for r in losses) / len(losses), 2) if losses else 0,
        "avg_edge": round(sum(r.get("edge") or 0 for r in settled) / len(settled), 3),
        "trading_days": len(daily),
        "profitable_days": profitable_days,
        "losing_days": len(daily) - profitable_days,
        "best_day": {"date": best_day["date"], "pnl": best_day["pnl"]} if best_day else None,
        "worst_day": {"date": worst_day["date"], "pnl": worst_day["pnl"]} if worst_day else None,
        "current_streak": {
            "type": current_streak_type or "none",
            "count": current_streak_count,
        },
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def compute_city_summary(results: list[dict]) -> dict:
    """Overall by-city breakdown."""
    settled = [r for r in results if r["settled"]]
    by_city: dict[str, list[dict]] = defaultdict(list)
    for r in settled:
        by_city[get_city(r)].append(r)

    summary = {}
    for city in sorted(by_city.keys()):
        cr = by_city[city]
        pnl = sum(r["pnl"] for r in cr)
        wagered = sum(r["order_price"] * r["contracts"] for r in cr)
        wins = sum(1 for r in cr if r["won"])
        summary[city] = {
            "trades": len(cr),
            "wins": wins,
            "losses": len(cr) - wins,
            "win_rate_pct": round(wins / len(cr) * 100, 1),
            "pnl": round(pnl, 2),
            "wagered": round(wagered, 2),
            "roi_pct": round(pnl / wagered * 100, 1) if wagered else 0,
        }
    return summary


def compute_strategy_summary(results: list[dict]) -> dict:
    """Per-strategy settled breakdown (mirrors compute_city_summary).

    Uses the file's pre-computed ``pnl``/``won``/``order_price`` fields (raw
    order-price view). For the favorite_bias kill-criteria verdict use
    scripts/favorite_bias_report.py (attribution-based source of truth).
    """
    settled = [r for r in results if r["settled"]]
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in settled:
        by_strategy[r.get("strategy") or "unknown"].append(r)

    summary = {}
    for strategy in sorted(by_strategy.keys()):
        sr = by_strategy[strategy]
        pnl = sum(r["pnl"] for r in sr)
        wagered = sum(r["order_price"] * r["contracts"] for r in sr)
        wins = sum(1 for r in sr if r["won"])
        summary[strategy] = {
            "trades": len(sr),
            "wins": wins,
            "losses": len(sr) - wins,
            "win_rate_pct": round(wins / len(sr) * 100, 1),
            "pnl": round(pnl, 2),
            "wagered": round(wagered, 2),
            "roi_pct": round(pnl / wagered * 100, 1) if wagered else 0,
        }
    return summary


def compute_edge_calibration(results: list[dict]) -> list[dict]:
    """Edge bucket analysis."""
    settled = [r for r in results if r["settled"]]
    buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    out = []
    for lo, hi in buckets:
        b = [r for r in settled if lo <= (r.get("edge") or 0) < hi]
        if b:
            pnl = sum(r["pnl"] for r in b)
            wins = sum(1 for r in b if r["won"])
            out.append(
                {
                    "range": f"{lo:.0%}-{hi:.0%}",
                    "trades": len(b),
                    "wins": wins,
                    "win_rate_pct": round(wins / len(b) * 100, 1),
                    "pnl": round(pnl, 2),
                    "avg_edge": round(sum(r.get("edge") or 0 for r in b) / len(b), 3),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def print_dashboard(
    summary: dict,
    daily: list[dict],
    city_summary: dict,
    edge_cal: list[dict],
    strategy_summary: dict | None = None,
) -> None:
    """Print a compact terminal dashboard."""
    print()
    print("=" * 88)
    print("  SHADOW TRADING DASHBOARD")
    print("=" * 88)

    # Summary
    s = summary
    print(
        f"\n  Total: {s['settled']} settled / {s['pending']} pending / "
        f"{s['total_trades']} total trades across {s.get('trading_days', 0)} days"
    )
    if s["settled"] == 0:
        print("\n  No settled trades yet — nothing to score.\n")
        return
    print()
    print(
        f"  Net P&L:       ${s['net_pnl']:>10,.2f}     Win rate: {s['win_rate_pct']}% "
        f"({s['wins']}W / {s['losses']}L)"
    )
    print(f"  Total wagered: ${s['total_wagered']:>10,.2f}     ROI:      {s['roi_pct']}%")
    print(f"  Total fees:    ${s['total_fees']:>10,.2f}     Avg edge: {s['avg_edge']:.1%}")
    print(f"  Avg win:       ${s['avg_win']:>10,.2f}     Avg loss: ${s['avg_loss']:>,.2f}")
    print()
    if s.get("best_day"):
        print(
            f"  Best day:  {s['best_day']['date']}  ${s['best_day']['pnl']:>+,.2f}    "
            f"Profitable days: {s['profitable_days']}/{s['trading_days']}"
        )
    if s.get("worst_day"):
        print(
            f"  Worst day: {s['worst_day']['date']}  ${s['worst_day']['pnl']:>+,.2f}    "
            f"Streak: {s['current_streak']['count']} {s['current_streak']['type']}"
            f"  (max W:{s['max_win_streak']} L:{s['max_loss_streak']})"
        )

    # Daily table
    print()
    print("-" * 88)
    print(
        f"  {'Date':<12} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} "
        f"{'P&L':>10} {'Fees':>7} {'Wagered':>10} {'ROI%':>7} {'Cum P&L':>11}"
    )
    print("-" * 88)
    for d in daily:
        pnl_str = f"${d['pnl']:>+,.2f}"
        cum_str = f"${d['cumulative_pnl']:>+,.2f}"
        print(
            f"  {d['date']:<12} {d['trades']:>6} {d['wins']:>4} {d['losses']:>4} "
            f"{d['win_rate_pct']:>5.1f}% {pnl_str:>10} ${d['fees']:>6,.2f} "
            f"${d['wagered']:>9,.2f} {d['roi_pct']:>6.1f}% {cum_str:>11}"
        )
    print("-" * 88)

    # City summary
    print()
    print(
        f"  {'City':<6} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} "
        f"{'P&L':>10} {'Wagered':>10} {'ROI%':>7}"
    )
    print(f"  {'-' * 6} {'-' * 6} {'-' * 4} {'-' * 4} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 7}")
    for city, cs in city_summary.items():
        pnl_str = f"${cs['pnl']:>+,.2f}"
        print(
            f"  {city:<6} {cs['trades']:>6} {cs['wins']:>4} {cs['losses']:>4} "
            f"{cs['win_rate_pct']:>5.1f}% {pnl_str:>10} ${cs['wagered']:>9,.2f} "
            f"{cs['roi_pct']:>6.1f}%"
        )

    # Strategy summary
    if strategy_summary:
        print()
        print(
            f"  {'Strategy':<16} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} "
            f"{'P&L':>10} {'Wagered':>10} {'ROI%':>7}"
        )
        print(
            f"  {'-' * 16} {'-' * 6} {'-' * 4} {'-' * 4} {'-' * 6} "
            f"{'-' * 10} {'-' * 10} {'-' * 7}"
        )
        for strategy, ss in strategy_summary.items():
            pnl_str = f"${ss['pnl']:>+,.2f}"
            print(
                f"  {strategy:<16} {ss['trades']:>6} {ss['wins']:>4} {ss['losses']:>4} "
                f"{ss['win_rate_pct']:>5.1f}% {pnl_str:>10} ${ss['wagered']:>9,.2f} "
                f"{ss['roi_pct']:>6.1f}%"
            )
        print(
            "  (raw order-price view — for the favorite_bias pilot kill-criteria "
            "verdict, use favorite_bias_report.py)"
        )

    # Edge calibration
    if edge_cal:
        print()
        print(f"  {'Edge':>8} {'Trades':>6} {'WR%':>6} {'P&L':>10} {'Avg Edge':>9}")
        print(f"  {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 9}")
        for b in edge_cal:
            pnl_str = f"${b['pnl']:>+,.2f}"
            print(
                f"  {b['range']:>8} {b['trades']:>6} {b['win_rate_pct']:>5.1f}% "
                f"{pnl_str:>10} {b['avg_edge']:>8.1%}"
            )

    print()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def build_report(
    summary: dict,
    daily: list[dict],
    city_summary: dict,
    edge_cal: list[dict],
    by_strategy: dict | None = None,
) -> dict:
    """Build the full JSON report."""
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "summary": summary,
        "daily": daily,
        "by_city": city_summary,
        "by_strategy": by_strategy,
        "edge_calibration": edge_cal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow trading daily P&L dashboard",
    )
    parser.add_argument(
        "--trades",
        default=str(DEFAULT_TRADES),
        help="Path to trades.jsonl (default: data/vps_trades.jsonl)",
    )
    parser.add_argument(
        "--full",
        default=str(DEFAULT_FULL),
        help="Path to shadow_settlements.full.json (default: data/shadow_settlements.full.json)",
    )
    parser.add_argument(
        "--pull-vps",
        action="store_true",
        help="SCP latest shadow trades + audit streams from VPS before analysis",
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch settlement data from Kalshi API (requires --pull-vps or local trades)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Export structured JSON report to file",
    )
    parser.add_argument(
        "--mode",
        choices=("shadow", "live", "paper", "all"),
        default="all",
        help=(
            "Restrict the dashboard to rows whose execution mode matches. "
            "Default 'all' preserves pre-rename behavior; 'shadow' filters out "
            "the live pilot rows, etc. Settlement file is unchanged."
        ),
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help=(
            "Restrict the dashboard to a single strategy (e.g. weather_edge, "
            "favorite_bias). Applied AFTER --mode. NOTE: missing strategy maps to "
            "'unknown' (NOT 'shadow' as --mode does), so --strategy weather_edge "
            "excludes legacy untagged rows. The by-strategy table here is the raw "
            "order-price P&L view; for the favorite_bias pilot kill-criteria "
            "verdict use favorite_bias_report.py."
        ),
    )
    args = parser.parse_args()

    trades_path = Path(args.trades)

    # Step 1: optionally pull from VPS
    if args.pull_vps:
        audit_dir: Path | None = None if args.skip_audit_pull else Path(args.audit_dir)
        pull_from_vps(trades_path, audit_dir=audit_dir)

    # Step 2: optionally refresh settlements
    if args.refresh:
        refresh_settlements(trades_path)

    # Step 3: load pre-computed results
    full_path = Path(args.full)
    results = load_full_results(full_path)
    print(f"Loaded {len(results)} trades from {full_path.name}")

    # Step 3b: optionally slice by execution mode (rows missing the field
    # predate the live pilot — treat as shadow).
    if args.mode != "all":
        before = len(results)
        results = [
            r for r in results if (r.get("mode") or "shadow") == args.mode
        ]
        print(
            f"!! --mode {args.mode}: showing {len(results)} of {before} rows "
            f"(filtered {before - len(results)} other-mode rows)"
        )

    # Step 3c: optionally slice by strategy (applied AFTER --mode).
    if args.strategy is not None:
        before = len(results)
        results = [r for r in results if (r.get("strategy") or "unknown") == args.strategy]
        print(
            f"!! --strategy {args.strategy}: showing {len(results)} of {before} rows "
            f"(filtered {before - len(results)} other-strategy rows)"
        )

    # Step 4: compute dashboard data
    daily = compute_daily(results)
    summary = compute_summary(results, daily)
    city_summary = compute_city_summary(results)
    strategy_summary = compute_strategy_summary(results)
    edge_cal = compute_edge_calibration(results)

    # Step 5: display
    print_dashboard(summary, daily, city_summary, edge_cal, strategy_summary)

    # Step 6: optionally export JSON
    if args.json:
        report = build_report(summary, daily, city_summary, edge_cal, by_strategy=strategy_summary)
        report["mode"] = args.mode
        report["strategy"] = args.strategy
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON report saved to {out_path}")


if __name__ == "__main__":
    main()
