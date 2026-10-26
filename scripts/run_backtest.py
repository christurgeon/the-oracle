"""
CLI runner for The Oracle backtesting engine.

Loads collected historical data, replays the weather strategy, and prints
a plain-text performance report.

Usage::

    uv run python scripts/run_backtest.py [--data-dir data/backtest]
        [--edge-threshold 0.08] [--max-edge 0.30] [--bankroll 10000]
        [--sweep PARAM=val1,val2,val3]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.backtest import (  # noqa: E402, I001
    BacktestConfig,
    BacktestResult,
    WalkForwardConfig,
    WalkForwardResult,
    load_forecasts,
    load_markets,
    load_snapshot_forecasts,
    replay_markets,
    walk_forward_replay,
)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(result: BacktestResult, label: str = "") -> str:
    """Build a plain-text backtest report."""
    lines: list[str] = []

    header = "=== BACKTEST REPORT ==="
    if label:
        header += f"  ({label})"
    lines.append(header)

    # --- Period ---
    if result.trades:
        dates = sorted({t.ticker for t in result.trades})
        lines.append(f"Markets: {len(dates)} unique tickers")
    lines.append("")

    # --- Summary ---
    lines.append("--- SUMMARY ---")
    lines.append(f"Markets analyzed:    {result.total_markets}")
    lines.append(f"Trades executed:     {result.num_trades}")
    lines.append(f"Win rate:            {result.win_rate:.1%}")
    lines.append(f"Total P&L:           ${result.total_pnl:+.2f}")
    lines.append(f"Avg P&L/trade:       ${result.avg_pnl_per_trade:+.2f}")
    lines.append(f"ROI:                 {result.roi:.2%}")
    lines.append(f"Sharpe ratio:        {result.sharpe_ratio:.2f}")
    lines.append(f"Max drawdown:        ${result.max_drawdown:.2f}")
    lines.append(f"Bankroll:            ${result.config.bankroll:,.0f}")
    lines.append("")

    # --- Skip reasons ---
    lines.append("--- SKIP REASONS ---")
    lines.append(f"No result (unsettled):       {result.skip_no_result}")
    lines.append(f"No forecast:                 {result.skip_no_forecast}")
    lines.append(f"No candles:                  {result.skip_no_candles}")
    lines.append(f"Candle error:                {result.skip_candle_error}")
    lines.append(f"No entry price:              {result.skip_no_entry}")
    lines.append(f"Unmappable market:           {result.skip_unmappable_market}")
    lines.append(f"Excluded city/var:           {result.skip_excluded_city_var}")
    lines.append(f"Spread too wide:             {result.skip_spread}")
    lines.append(f"Raw prob gate:               {result.skip_raw_prob_gate}")
    lines.append(f"Strike distance:             {result.skip_strike_distance}")
    lines.append(f"Calibrator disagreement:     {result.skip_calibrator_disagreement}")
    lines.append(f"Plateau block:               {result.skip_plateau_block}")
    lines.append(f"YES floor:                   {result.skip_yes_floor}")
    lines.append(f"YES inverted:                {result.skip_yes_inverted}")
    lines.append(f"NO inverted:                 {result.skip_no_inverted}")
    lines.append(f"Below threshold:             {result.skip_below_threshold}")
    lines.append(f"Above max_edge:              {result.skip_above_max_edge}")
    lines.append(f"Kelly zero:                  {result.skip_kelly_zero}")
    lines.append(f"Zero size (legacy alias):    {result.skip_zero_size}")
    lines.append(f"No price info:               {result.skip_no_price_info}")
    lines.append(f"Other:                       {result.skip_other}")
    lines.append("")

    # --- Calibration ---
    buckets = result.calibration_buckets(n_bins=10)
    non_empty = [b for b in buckets if b.count > 0]
    if non_empty:
        lines.append("--- CALIBRATION ---")
        lines.append(f"{'Model Prob':>12}  {'Count':>6}  {'Actual YES%':>11}  {'Delta':>7}")
        lines.append("-" * 44)
        for b in non_empty:
            lines.append(
                f"{b.range_low:.1f}-{b.range_high:.1f}       "
                f"{b.count:>6}  "
                f"{b.actual_yes_rate:>10.1%}  "
                f"{b.delta:>+6.1%}"
            )
        mae = result.calibration_mae(buckets=buckets)
        lines.append(f"\nMean Absolute Error: {mae:.3f}")
        lines.append("")

    # --- Edge distribution ---
    edge_b = result.edge_buckets()
    if edge_b:
        lines.append("--- EDGE DISTRIBUTION ---")
        lines.append(
            f"{'Edge Range':>12}  {'Trades':>7}  {'Win%':>6}  "
            f"{'Avg P&L':>9}  {'Total P&L':>10}"
        )
        lines.append("-" * 52)
        for eb in edge_b:
            lines.append(
                f"{eb.range_label:>12}  "
                f"{eb.count:>7}  "
                f"{eb.win_pct:>5.0%}  "
                f"${eb.avg_pnl:>+8.2f}  "
                f"${eb.total_pnl:>+9.2f}"
            )
        lines.append("")

    # --- Max-edge filter analysis ---
    if result.config.max_edge < 1.0:
        lines.append("--- MAX_EDGE FILTER ANALYSIS ---")
        lines.append(
            f"Current max_edge: {result.config.max_edge:.2f}  "
            f"(excluded {result.skip_above_max_edge} markets)"
        )
        lines.append(
            "Re-run with --max-edge 1.0 to see full edge distribution"
        )
        lines.append("")

    # --- Individual trades ---
    if result.trades:
        lines.append("--- INDIVIDUAL TRADES ---")
        lines.append(
            f"{'Ticker':<25} {'City':>4} {'Side':>4} {'Model':>6} "
            f"{'Mkt':>5} {'Edge':>6} {'Contracts':>9} {'P&L':>9} {'Won':>4}"
        )
        lines.append("-" * 82)
        for t in result.trades:
            lines.append(
                f"{t.ticker:<25} {t.city:>4} {t.side:>4} {t.model_prob:>5.1%} "
                f"{t.market_price:>5.2f} {t.effective_edge:>5.1%} "
                f"{t.contracts:>9} ${t.pnl:>+8.2f} "
                f"{'Y' if t.won else 'N':>4}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward report
# ---------------------------------------------------------------------------


def format_walk_forward_report(wf_result: WalkForwardResult) -> str:
    """Build a plain-text walk-forward validation report."""
    lines: list[str] = []
    lines.append("=== WALK-FORWARD VALIDATION REPORT ===")
    lines.append(
        f"Config: min_train_days={wf_result.config.min_train_days}  "
        f"test_window_days={wf_result.config.test_window_days}  "
        f"step_days={wf_result.config.step_days}  "
        f"expanding={'yes' if wf_result.config.expanding_window else 'no (rolling)'}"
    )
    lines.append(f"Total unique settled dates: {wf_result.total_unique_dates}")
    lines.append("")

    if wf_result.skipped_reason:
        lines.append(f"SKIPPED: {wf_result.skipped_reason}")
        lines.append(
            f"Need at least {wf_result.config.min_train_days + wf_result.config.test_window_days}"
            f" settled dates; found {wf_result.total_unique_dates}."
        )
        return "\n".join(lines)

    n_folds = len(wf_result.folds)
    lines.append(f"Folds: {n_folds}")
    lines.append("")

    # Per-fold table
    lines.append("--- PER-FOLD RESULTS ---")
    header = (
        f"{'Fold':>4}  {'Train':>21}  {'Test':>21}  "
        f"{'Trades':>6}  {'Win%':>5}  {'P&L':>9}  {'Sharpe':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for f in wf_result.folds:
        train_range = f"{f.train_start} – {f.train_end}"
        test_range = f"{f.test_start} – {f.test_end}"
        lines.append(
            f"{f.fold_index:>4}  {train_range:>21}  {test_range:>21}  "
            f"{f.result.num_trades:>6}  {f.result.win_rate:>4.0%}  "
            f"${f.result.total_pnl:>+8.2f}  {f.result.sharpe_ratio:>7.2f}"
        )
    lines.append("")

    # Aggregate stats
    lines.append("--- AGGREGATE ---")
    lines.append(
        f"Mean P&L per fold:    ${wf_result.mean_pnl:+.2f} "
        f"± ${wf_result.stdev_pnl:.2f}"
    )
    lines.append(
        f"Mean Sharpe:          {wf_result.mean_sharpe:.2f} "
        f"± {wf_result.stdev_sharpe:.2f}"
    )
    lines.append(f"Total trades:         {wf_result.total_trades}")
    lines.append(f"Mean win rate:        {wf_result.mean_win_rate:.1%}")
    lines.append(f"Instability score:    {wf_result.instability_score:.2f}")
    lines.append(f"t-statistic:          {wf_result.t_statistic:.3f}")
    if n_folds >= 2:
        lines.append(f"p-value (one-sided):  {wf_result.p_value_one_sided:.4f}")
    lines.append(f"Significant (p<0.05): {'YES' if wf_result.significant else 'NO'}")
    lines.append("")

    if wf_result.instability_score > 2.0:
        lines.append("Results are noisy — increase corpus or relax filters.")
    if not wf_result.significant and n_folds >= 5:
        lines.append("Strategy not proven profitable on out-of-sample data.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweep support
# ---------------------------------------------------------------------------


def parse_sweep(sweep_str: str) -> tuple[str, list[str]]:
    """Parse 'PARAM=val1,val2,val3' into (param_name, [values])."""
    if "=" not in sweep_str:
        raise ValueError(f"Sweep format must be PARAM=val1,val2,val3, got: {sweep_str}")
    param, values_str = sweep_str.split("=", 1)
    values = [v.strip() for v in values_str.split(",")]
    return param.strip(), values


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run The Oracle weather strategy backtest."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/backtest",
        help="Directory containing markets.json and weather_forecasts.json.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.08,
        help="Minimum effective edge to trade (default: 0.08).",
    )
    parser.add_argument(
        "--max-edge",
        type=float,
        default=0.30,
        help="Maximum effective edge (skip likely stale quotes, default: 0.30).",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=10_000.0,
        help="Total bankroll for Kelly sizing (default: 10000).",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.15,
        help="Fractional Kelly multiplier (default: 0.15).",
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=24.0,
        help="Target hours before expiry for entry price (default: 24).",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Sweep a parameter: PARAM=val1,val2,val3 (e.g., edge_threshold=0.05,0.08,0.12).",
    )
    parser.add_argument(
        "--data-source",
        choices=["legacy", "snapshot"],
        default="legacy",
        help="Forecast data source: 'legacy' reads data/backtest/, 'snapshot' reads corpus.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default="data/snapshots",
        help="Root directory for snapshot corpus (default: data/snapshots).",
    )
    parser.add_argument(
        "--forecast-lookup-fallback",
        choices=["none", "earliest"],
        default="none",
        help=(
            "Fallback when no snapshot run predates the entry candle: "
            "'none' skips the market, 'earliest' uses the first available run."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation instead of a full replay.",
    )
    parser.add_argument(
        "--wf-min-train-days",
        type=int,
        default=5,
        help="Minimum number of training days per fold (default: 5).",
    )
    parser.add_argument(
        "--wf-test-window-days",
        type=int,
        default=1,
        help="Number of test days per fold (default: 1).",
    )
    parser.add_argument(
        "--wf-step-days",
        type=int,
        default=1,
        help="Step size in days between folds (default: 1).",
    )
    parser.add_argument(
        "--wf-rolling",
        action="store_true",
        help="Use rolling window instead of expanding window for walk-forward.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    markets_path = data_dir / "markets.json"

    if not markets_path.exists():
        print(f"Error: {markets_path} not found. Run collect_backtest_data.py first.")
        sys.exit(1)

    markets = load_markets(markets_path)

    data_source = args.data_source
    snapshot_index = None
    forecasts = None

    if data_source == "snapshot":
        snapshot_root = Path(args.snapshot_root)
        if not snapshot_root.exists():
            print(f"Error: snapshot root {snapshot_root} not found.")
            sys.exit(1)
        print(f"Loading snapshot corpus from {snapshot_root} ...")
        snapshot_index = load_snapshot_forecasts(snapshot_root)
        print(f"Loaded {len(snapshot_index.by_key)} unique forecast keys\n")
    else:
        forecasts_path = data_dir / "weather_forecasts.json"
        if not forecasts_path.exists():
            print(f"Error: {forecasts_path} not found. Run collect_backtest_data.py first.")
            sys.exit(1)
        forecasts = load_forecasts(forecasts_path)
        print(f"Loaded {len(markets)} markets and {len(forecasts)} forecasts from {data_dir}\n")

    base_config = BacktestConfig(
        edge_threshold=args.edge_threshold,
        max_edge=args.max_edge,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        target_hours_before_expiry=args.target_hours,
        data_source=data_source,
        forecast_lookup_fallback=args.forecast_lookup_fallback,
    )

    if args.walk_forward:
        wf_config = WalkForwardConfig(
            min_train_days=args.wf_min_train_days,
            test_window_days=args.wf_test_window_days,
            step_days=args.wf_step_days,
            expanding_window=not args.wf_rolling,
        )
        wf_result = walk_forward_replay(
            markets, forecasts, base_config, wf_config,
            snapshot_index=snapshot_index,
        )
        print(format_walk_forward_report(wf_result))
        return

    if args.sweep:
        param, values = parse_sweep(args.sweep)
        print(f"Sweeping {param} over: {values}\n")
        print("=" * 60)

        for val_str in values:
            val = float(val_str)
            sweep_config = dataclasses.replace(base_config, **{param: val})
            result = replay_markets(
                markets, forecasts, sweep_config, snapshot_index=snapshot_index,
            )
            print(f"\n--- {param}={val_str} ---")
            print(
                f"Trades: {result.num_trades}  "
                f"Win: {result.win_rate:.0%}  "
                f"P&L: ${result.total_pnl:+.2f}  "
                f"ROI: {result.roi:.2%}  "
                f"Sharpe: {result.sharpe_ratio:.2f}  "
                f"MaxDD: ${result.max_drawdown:.2f}"
            )
        print("\n" + "=" * 60)
    else:
        result = replay_markets(
            markets, forecasts, base_config, snapshot_index=snapshot_index,
        )
        print(format_report(result))


if __name__ == "__main__":
    main()
