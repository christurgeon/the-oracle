"""
Backtesting engine for The Oracle weather strategy.

Replays historical weather markets against GFS ensemble forecasts to measure
strategy performance, calibration, and risk metrics. Reuses the exact same
functions (Kelly sizing, model probability, P&L math) as the production
pipeline so results are representative.

Usage::

    from python.analytics.backtest import (
        BacktestConfig, load_markets, load_forecasts, replay_markets,
    )

    markets = load_markets(Path("data/backtest/markets.json"))
    forecasts = load_forecasts(Path("data/backtest/weather_forecasts.json"))
    result = replay_markets(markets, forecasts, BacktestConfig())
    print(result.total_pnl, result.win_rate, result.sharpe_ratio)
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from python.analytics.audit_log import RejectionReason, ScanFunnel, funnel_attr_for
from python.analytics.fees import (
    order_fee as kalshi_order_fee,
)
from python.analytics.sizing import binary_contract_pnl
from python.common import DailyVariable, Direction, Side
from python.data.bias_correction import BiasTable
from python.data.calibration import IsotonicCalibrator
from python.data.weather import EnsembleForecast, WeatherClient
from python.strategies.weather_evaluation import (
    _EvalConfig,
    _MarketCandidate,
    evaluate_market_pure,
    postprocess_forecasts,
)
from python.strategies.weather_parsing import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
)
from scripts.collect_backtest_data import (
    CandlestickRecord,
    MarketRecord,
    WeatherForecastRecord,
)

logger = logging.getLogger(__name__)

# Maps RejectionReason string values to the BacktestResult attribute they increment.
_REJECT_REASON_TO_ATTR: dict[str, str] = {
    str(RejectionReason.EXCLUDED_CITY_VAR):      "skip_excluded_city_var",
    str(RejectionReason.SPREAD):                  "skip_spread",
    str(RejectionReason.RAW_PROB_GATE):           "skip_raw_prob_gate",
    str(RejectionReason.STRIKE_DISTANCE):         "skip_strike_distance",
    str(RejectionReason.CALIBRATOR_DISAGREEMENT): "skip_calibrator_disagreement",
    str(RejectionReason.PLATEAU_BLOCK):           "skip_plateau_block",
    str(RejectionReason.YES_FLOOR):               "skip_yes_floor",
    str(RejectionReason.YES_INVERTED):            "skip_yes_inverted",
    str(RejectionReason.NO_INVERTED):             "skip_no_inverted",
    str(RejectionReason.EDGE_BELOW_THRESHOLD):    "skip_below_threshold",
    str(RejectionReason.EDGE_ABOVE_MAX):          "skip_above_max_edge",
    str(RejectionReason.KELLY_ZERO):              "skip_kelly_zero",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """All tunable parameters for a backtest run.

    Mirrors the production ``config.yaml`` ``strategies.weather`` block so the
    backtest runs the exact same filters as the live strategy.
    """

    edge_threshold: float = 0.07
    max_edge: float = 0.30
    spread_skill_sensitivity: float = 0.15
    max_spread: float = 0.15
    kelly_fraction: float = 0.15
    max_single_market_pct: float = 0.05
    max_total_exposure_pct: float = 0.80   # backtest default (independent of live config.yaml)
    bankroll: float = 10_000.0
    target_hours_before_expiry: float = 24.0
    min_hours_before_expiry: float = 12.0
    max_hours_before_expiry: float = 48.0
    # Fees are always included in the shared evaluation path; retained for
    # report output (``total_fee`` can be added back for pre-fee analysis).
    include_fees: bool = True
    # Pure-function gates — defaults match live config.yaml.
    min_raw_prob_for_between: float = 0.05
    min_distance_from_strike_std: float = 2.0
    excluded_city_vars: tuple[str, ...] = ("DEN_low", "LAX_low", "LAX_high")
    block_raw_prob_range: tuple[float, float] | None = (0.533, 0.967)
    # Boundary-distance suppressive filter (Path 7a). Maps "<CITY>/<var>" → °F
    # threshold; empty (default) disables. Lets the backtest/replay exercise
    # the same filter the live bot runs — needed for Path 7b's Phase A.5 replay
    # and to re-validate Path 7a on the GEFS corpus.
    boundary_filter_cells: dict[str, float] = field(default_factory=dict)
    # Snapshot corpus settings
    data_source: Literal["legacy", "snapshot"] = "legacy"
    snapshot_entry_source: Literal["legacy_candles", "snapshot_quotes"] = "legacy_candles"
    snapshot_root: Path | None = None
    forecast_lookup_fallback: Literal["none", "earliest"] = "none"


# ---------------------------------------------------------------------------
# Snapshot index structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotForecastIndex:
    by_key: dict[
        tuple[str, date, DailyVariable],
        list[tuple[float, str | None, dict[str, EnsembleForecast]]],
    ]

    def runs_for(
        self, city: str, forecast_date: date, daily_variable: DailyVariable
    ) -> list[tuple[float, str | None, dict[str, EnsembleForecast]]]:
        return self.by_key.get((city, forecast_date, daily_variable), [])


# ---------------------------------------------------------------------------
# Walk-forward validation structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Parameters for walk-forward validation.

    Does NOT refit the calibrator per fold — strategy config is held fixed.
    """

    min_train_days: int = 5
    test_window_days: int = 1
    step_days: int = 1
    expanding_window: bool = True

    def __post_init__(self) -> None:
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")
        if self.test_window_days < 1:
            raise ValueError("test_window_days must be >= 1")
        if self.step_days < 1:
            raise ValueError("step_days must be >= 1")


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    config: WalkForwardConfig
    backtest_config: BacktestConfig
    total_unique_dates: int
    skipped_reason: str | None = None

    @property
    def fold_pnls(self) -> list[float]:
        return [f.result.total_pnl for f in self.folds]

    @property
    def fold_trade_counts(self) -> list[int]:
        return [f.result.num_trades for f in self.folds]

    @property
    def fold_win_rates(self) -> list[float]:
        return [f.result.win_rate for f in self.folds]

    @property
    def fold_sharpes(self) -> list[float]:
        return [f.result.sharpe_ratio for f in self.folds]

    @property
    def mean_pnl(self) -> float:
        return statistics.mean(self.fold_pnls) if self.folds else 0.0

    @property
    def stdev_pnl(self) -> float:
        return statistics.stdev(self.fold_pnls) if len(self.folds) >= 2 else 0.0

    @property
    def mean_win_rate(self) -> float:
        total = sum(self.fold_trade_counts)
        if total == 0:
            return 0.0
        return sum(f.result.win_rate * f.result.num_trades for f in self.folds) / total

    @property
    def mean_sharpe(self) -> float:
        return statistics.mean(self.fold_sharpes) if self.folds else 0.0

    @property
    def stdev_sharpe(self) -> float:
        return statistics.stdev(self.fold_sharpes) if len(self.folds) >= 2 else 0.0

    @property
    def total_trades(self) -> int:
        return sum(self.fold_trade_counts)

    @property
    def instability_score(self) -> float:
        if not self.folds:
            return 0.0
        mean = self.mean_pnl
        if abs(mean) < 1e-9:
            return float("inf")
        return self.stdev_pnl / abs(mean)

    @property
    def t_statistic(self) -> float:
        n = len(self.folds)
        if n < 2 or self.stdev_pnl == 0:
            return 0.0
        return self.mean_pnl / (self.stdev_pnl / math.sqrt(n))

    @property
    def p_value_one_sided(self) -> float:
        """One-sided p-value for H0: mean_pnl <= 0 via Student's t with df=n-1.

        Uses the regularized incomplete beta for the t-CDF (no scipy needed).
        """
        n = len(self.folds)
        if n < 2:
            return 0.5
        t = self.t_statistic
        if t <= 0:
            return 0.5
        df = n - 1
        # P(T > t) for t > 0: use regularized incomplete beta
        # P(T > t) = 0.5 * I_x(df/2, 1/2) where x = df/(df + t^2)
        x = df / (df + t * t)
        return 0.5 * _regularized_incomplete_beta(x, df / 2, 0.5)

    @property
    def significant(self) -> bool:
        return len(self.folds) >= 5 and self.mean_pnl > 0 and self.p_value_one_sided < 0.05


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) via Lentz continued fraction.

    Used for Student's t-distribution survival function (see
    ``WalkForwardResult.p_value_one_sided``). Accurate for x in [0,1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    # for better convergence of the CF
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    # Log of the beta function normalizer
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # Lentz's continued fraction
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - log_beta) / a
    # Modified Lentz's method (cf=C numerator accumulator, dv=D denominator accumulator)
    tiny = 1e-30
    f = tiny
    cf = f
    dv = 0.0
    for m in range(200):
        for sign in (0, 1):
            if sign == 0:
                if m == 0:
                    num = 1.0
                else:
                    num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            dv = 1.0 + num * dv
            if abs(dv) < tiny:
                dv = tiny
            dv = 1.0 / dv
            cf = 1.0 + num / cf
            if abs(cf) < tiny:
                cf = tiny
            delta = cf * dv
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                return front * f
    return front * f  # return best estimate if max iterations reached


# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------


def _safe_iso_to_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _compute_fold_windows(
    sorted_dates: list[date],
    config: WalkForwardConfig,
) -> list[tuple[date, date, date, date]]:
    """Return (train_start, train_end, test_start, test_end) tuples, inclusive bounds."""
    n = len(sorted_dates)
    needed = config.min_train_days + config.test_window_days
    if n < needed:
        return []
    folds: list[tuple[date, date, date, date]] = []
    fold_idx = 0
    while True:
        test_start_idx = config.min_train_days + fold_idx * config.step_days
        test_end_idx = test_start_idx + config.test_window_days - 1
        if test_end_idx >= n:
            break
        if config.expanding_window:
            train_start_idx = 0
        else:
            train_start_idx = test_start_idx - config.min_train_days
        train_end_idx = test_start_idx - 1
        folds.append((
            sorted_dates[train_start_idx],
            sorted_dates[train_end_idx],
            sorted_dates[test_start_idx],
            sorted_dates[test_end_idx],
        ))
        fold_idx += 1
    return folds


def _filter_by_dates(
    markets: list[MarketRecord],
    forecasts: dict[str, WeatherForecastRecord] | None,
    date_set: frozenset[date],
) -> tuple[list[MarketRecord], dict[str, WeatherForecastRecord] | None]:
    iso_set = {d.isoformat() for d in date_set}
    filtered_markets = [
        m for m in markets if _safe_iso_to_date(m.expiration_date) in date_set
    ]
    if forecasts is None:
        return filtered_markets, None
    filtered_forecasts = {
        k: v for k, v in forecasts.items()
        if len(parts := k.split(":")) >= 2 and parts[1] in iso_set
    }
    return filtered_markets, filtered_forecasts


# ---------------------------------------------------------------------------
# Walk-forward replay
# ---------------------------------------------------------------------------


def walk_forward_replay(
    markets: list[MarketRecord],
    forecasts: dict[str, WeatherForecastRecord] | None = None,
    backtest_config: BacktestConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
    *,
    snapshot_index: SnapshotForecastIndex | None = None,
    calibrator: IsotonicCalibrator | None = None,
    bias_table: BiasTable | None = None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
    ensemble_weights: dict[str, float] | None = None,
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
) -> WalkForwardResult:
    """Walk-forward validation: splits corpus by expiration_date, runs replay_markets on each
    test fold, and aggregates fold statistics with a t-test for significance.

    The calibrator and strategy config are held fixed across folds (no per-fold refit).
    """
    if backtest_config is None:
        backtest_config = BacktestConfig()
    if wf_config is None:
        wf_config = WalkForwardConfig()

    unique_dates_raw = {
        _safe_iso_to_date(m.expiration_date)
        for m in markets
        if m.result is not None and _safe_iso_to_date(m.expiration_date) is not None
    }
    sorted_dates: list[date] = sorted(d for d in unique_dates_raw if d is not None)

    fold_windows = _compute_fold_windows(sorted_dates, wf_config)
    if not fold_windows:
        logger.warning(
            "walk_forward_replay: insufficient data for validation "
            "(%d settled dates, need %d)",
            len(sorted_dates),
            wf_config.min_train_days + wf_config.test_window_days,
        )
        return WalkForwardResult(
            folds=[],
            config=wf_config,
            backtest_config=backtest_config,
            total_unique_dates=len(sorted_dates),
            skipped_reason="insufficient_data",
        )

    folds: list[FoldResult] = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(fold_windows):
        test_date_range = test_end - test_start
        test_dates: frozenset[date] = frozenset(
            test_start + timedelta(days=i) for i in range(test_date_range.days + 1)
        )
        test_markets, test_forecasts = _filter_by_dates(markets, forecasts, test_dates)

        fold_result = replay_markets(
            test_markets,
            test_forecasts,
            backtest_config,
            calibrator=calibrator,
            bias_table=bias_table,
            bias_shrinkage=bias_shrinkage,
            bias_max_correction_f=bias_max_correction_f,
            ensemble_weights=ensemble_weights,
            spread_inflation_overrides=spread_inflation_overrides,
            snapshot_index=snapshot_index,
        )
        folds.append(FoldResult(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            result=fold_result,
        ))

    return WalkForwardResult(
        folds=folds,
        config=wf_config,
        backtest_config=backtest_config,
        total_unique_dates=len(sorted_dates),
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedEntry:
    """Market state at the simulated entry time."""

    yes_price: float  # 0-1 scale (midpoint / candle close)
    spread: float  # always 0 for candlestick data (no bid/ask)
    candle_ts: int  # end_period_ts of the selected candle
    hours_before_expiry: float
    yes_bid: float | None = None  # best bid in 0-1 scale (if available)
    yes_ask: float | None = None  # best ask in 0-1 scale (if available)


@dataclass(frozen=True)
class BacktestTrade:
    """One simulated trade in the backtest."""

    ticker: str
    city: str
    daily_variable: str
    strike_type: str
    floor_strike: float | None
    cap_strike: float | None
    model_prob: float
    market_price: float
    side: Side
    raw_edge: float
    effective_edge: float
    ensemble_confidence: float
    ensemble_std: float
    kelly_raw: float
    kelly_dollars: float
    kelly_capped: bool
    contracts: int
    buy_price: float  # actual cost per contract (YES price or 1-YES price)
    result: str  # "yes" or "no"
    pnl: float
    won: bool
    total_fee: float = 0.0
    entry_date: date | None = None


@dataclass(frozen=True)
class CalibrationBucket:
    """One bin in calibration analysis."""

    range_low: float
    range_high: float
    count: int
    actual_yes_rate: float
    avg_model_prob: float

    @property
    def delta(self) -> float:
        return abs(self.actual_yes_rate - self.avg_model_prob)


@dataclass(frozen=True)
class EdgeBucket:
    """One bin in edge distribution analysis."""

    range_low: float
    range_high: float
    count: int
    win_pct: float
    avg_pnl: float
    total_pnl: float

    @property
    def range_label(self) -> str:
        return f"{self.range_low:.2f}-{self.range_high:.2f}"


@dataclass
class BacktestResult:
    """Full backtest output with trades and skip diagnostics."""

    trades: list[BacktestTrade] = field(default_factory=list)
    config: BacktestConfig = field(default_factory=BacktestConfig)
    # Skip counters
    skip_no_result: int = 0
    skip_no_candles: int = 0
    skip_candle_error: int = 0
    skip_no_forecast: int = 0
    skip_no_entry: int = 0
    skip_below_threshold: int = 0
    skip_above_max_edge: int = 0
    skip_zero_size: int = 0
    skip_excluded_city_var: int = 0
    skip_spread: int = 0
    skip_raw_prob_gate: int = 0
    skip_strike_distance: int = 0
    skip_calibrator_disagreement: int = 0
    skip_plateau_block: int = 0
    skip_yes_floor: int = 0
    skip_yes_inverted: int = 0
    skip_no_inverted: int = 0
    skip_kelly_zero: int = 0
    skip_unmappable_market: int = 0
    skip_no_price_info: int = 0
    skip_no_pre_entry_run: int = 0
    skip_other: int = 0
    total_markets: int = 0

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.won) / len(self.trades)

    @property
    def avg_pnl_per_trade(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio using daily-aggregated P&L.

        Falls back to per-trade computation when any trade lacks entry_date
        (legacy fixture path), preserving backward compatibility with tests
        that construct BacktestTrade without entry_date.
        """
        if len(self.trades) < 2:
            return 0.0
        if any(t.entry_date is None for t in self.trades):
            pnls = [t.pnl for t in self.trades]
            std = statistics.stdev(pnls)
            if std == 0:
                return 0.0
            return (statistics.mean(pnls) / std) * math.sqrt(252)
        daily: dict[date, float] = {}
        for t in self.trades:
            assert t.entry_date is not None
            daily[t.entry_date] = daily.get(t.entry_date, 0.0) + t.pnl
        daily_pnls = list(daily.values())
        if len(daily_pnls) < 2:
            return 0.0
        std = statistics.stdev(daily_pnls)
        if std == 0:
            return 0.0
        return (statistics.mean(daily_pnls) / std) * math.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline in cumulative P&L."""
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def roi(self) -> float:
        if self.config.bankroll <= 0:
            return 0.0
        return self.total_pnl / self.config.bankroll

    def calibration_buckets(self, n_bins: int = 10) -> list[CalibrationBucket]:
        """Bin model probabilities into deciles and compare to actual YES rate."""
        if not self.trades:
            return []

        bin_width = 1.0 / n_bins
        buckets: list[CalibrationBucket] = []

        for i in range(n_bins):
            low = i * bin_width
            high = (i + 1) * bin_width

            in_bin = [
                t for t in self.trades
                if low <= t.model_prob < high or (i == n_bins - 1 and t.model_prob == high)
            ]
            if not in_bin:
                buckets.append(CalibrationBucket(
                    range_low=low, range_high=high,
                    count=0, actual_yes_rate=0.0, avg_model_prob=0.0,
                ))
                continue

            actual_yes = sum(1 for t in in_bin if t.result == "yes") / len(in_bin)
            avg_prob = statistics.mean(t.model_prob for t in in_bin)
            buckets.append(CalibrationBucket(
                range_low=low, range_high=high,
                count=len(in_bin), actual_yes_rate=actual_yes, avg_model_prob=avg_prob,
            ))

        return buckets

    def calibration_mae(
        self,
        n_bins: int = 10,
        buckets: list[CalibrationBucket] | None = None,
    ) -> float:
        """Mean absolute error across non-empty calibration buckets."""
        if buckets is None:
            buckets = self.calibration_buckets(n_bins)
        non_empty = [b for b in buckets if b.count > 0]
        if not non_empty:
            return 0.0
        return statistics.mean(b.delta for b in non_empty)

    def edge_buckets(self) -> list[EdgeBucket]:
        """Group trades by effective edge range and compute per-bucket stats."""
        if not self.trades:
            return []

        ranges = [
            (0.00, 0.10), (0.10, 0.15), (0.15, 0.20),
            (0.20, 0.25), (0.25, 0.30), (0.30, 1.00),
        ]
        result: list[EdgeBucket] = []
        for low, high in ranges:
            in_range = [
                t for t in self.trades
                if low <= t.effective_edge < high
            ]
            if not in_range:
                continue
            wins = sum(1 for t in in_range if t.won)
            total_pnl = sum(t.pnl for t in in_range)
            result.append(EdgeBucket(
                range_low=low,
                range_high=high,
                count=len(in_range),
                win_pct=wins / len(in_range),
                avg_pnl=total_pnl / len(in_range),
                total_pnl=total_pnl,
            ))
        return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_markets(path: Path) -> list[MarketRecord]:
    """Deserialize markets.json into MarketRecord objects."""
    with open(path) as f:
        data = json.load(f)

    records = []
    for d in data:
        candles = [
            CandlestickRecord(**c) for c in d.pop("candlesticks", [])
        ]
        record = MarketRecord(**d, candlesticks=candles)
        records.append(record)
    return records


def load_forecasts(path: Path) -> dict[str, WeatherForecastRecord]:
    """Deserialize weather_forecasts.json into WeatherForecastRecord objects."""
    with open(path) as f:
        data = json.load(f)

    return {key: WeatherForecastRecord(**val) for key, val in data.items()}


def forecast_to_ensemble(record: WeatherForecastRecord) -> EnsembleForecast:
    """Bridge a stored WeatherForecastRecord to the production EnsembleForecast."""
    return EnsembleForecast(
        city=record.city,
        forecast_date=date.fromisoformat(record.forecast_date),
        temperatures=record.temperatures,
        fetched_at=record.fetched_at,
    )


# ---------------------------------------------------------------------------
# Entry price simulation
# ---------------------------------------------------------------------------


def simulate_entry_price(
    market: MarketRecord,
    target_hours: float = 24.0,
    min_hours: float = 12.0,
    max_hours: float = 48.0,
) -> SimulatedEntry | None:
    """
    Find the candlestick closest to target_hours before expiration.

    Returns None if the market has no usable candles or none fall within
    the [min_hours, max_hours] window before expiration.
    """
    if market.candlestick_error or not market.candlesticks:
        return None

    target_ts = market.expiration_ts - target_hours * 3600
    min_ts = market.expiration_ts - max_hours * 3600
    max_ts = market.expiration_ts - min_hours * 3600

    eligible = [
        c for c in market.candlesticks
        if min_ts <= c.end_period_ts <= max_ts
    ]
    if not eligible:
        return None

    best = min(eligible, key=lambda c: abs(c.end_period_ts - target_ts))

    if best.yes_price_close is None:
        return None

    yes_price = best.yes_price_close / 100.0  # cents → dollars
    hours_before = (market.expiration_ts - best.end_period_ts) / 3600.0

    return SimulatedEntry(
        yes_price=yes_price,
        spread=0.0,  # candles don't have bid/ask
        candle_ts=best.end_period_ts,
        hours_before_expiry=hours_before,
    )


# ---------------------------------------------------------------------------
# Direction mapping
# ---------------------------------------------------------------------------


_STRIKE_TYPE_TO_DIRECTION: dict[str, Direction] = {
    "greater": Direction.ABOVE,
    "less": Direction.BELOW,
    "between": Direction.BETWEEN,
}


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _build_scan_candidate(
    market: MarketRecord,
    entry: SimulatedEntry,
) -> ScanCandidate | None:
    """Build a ``ScanCandidate`` from a stored ``MarketRecord`` + ``SimulatedEntry``.

    The live strategy constructs ScanCandidate objects from raw Kalshi API
    dicts; here we reconstitute the same shape from on-disk backtest data so
    the shared ``evaluate_market_pure`` can score the market.
    """
    direction = _STRIKE_TYPE_TO_DIRECTION.get(market.strike_type or "")
    if direction is None:
        return None

    try:
        daily_var = DailyVariable(market.daily_variable)
    except ValueError:
        return None

    try:
        forecast_date = date.fromisoformat(market.expiration_date)
    except ValueError:
        return None

    parsed = ParsedMarket(
        ticker=market.ticker,
        city=market.city,
        forecast_date=forecast_date,
        strike_type=direction,
        floor_strike=market.floor_strike,
        cap_strike=market.cap_strike,
        daily_variable=daily_var,
    )

    if entry.yes_bid is not None and entry.yes_ask is not None:
        spread: float | None = max(entry.yes_ask - entry.yes_bid, 0.0)
    else:
        spread = None

    price_info = MarketPriceInfo(
        midpoint=entry.yes_price,
        yes_bid=entry.yes_bid,
        yes_ask=entry.yes_ask,
        spread=spread,
        yes_bid_size=None,
        yes_ask_size=None,
        volume=None,
        open_interest=None,
    )

    # The pure path expects a Kalshi-shaped dict for metadata fallback. Only
    # ``ticker`` and ``event_ticker`` are strictly required by downstream
    # signal builders; fill the rest with what we know.
    raw_market: dict[str, object] = {
        "ticker": market.ticker,
        "series_ticker": market.series_ticker,
        "event_ticker": market.ticker.rsplit("-", 1)[0],
        "strike_type": market.strike_type,
        "floor_strike": market.floor_strike,
        "cap_strike": market.cap_strike,
    }

    return ScanCandidate(market=raw_market, parsed=parsed, price_info=price_info)


def _build_forecast_cache(
    forecasts: dict[str, WeatherForecastRecord],
    needed_keys: set[tuple[str, date, DailyVariable]],
) -> dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]:
    """Convert the flat backtest forecast store into the tuple-keyed model dict
    expected by ``postprocess_forecasts`` / ``evaluate_market_pure``.

    The backtest currently stores only one model per (city, date, variable)
    keyed as ``"<city>:<iso_date>:<variable>"``. We wrap each into the
    ``{model_name: EnsembleForecast}`` shape using ``"gfs_seamless"`` as the
    model label so bias correction and per-model metadata paths fire.
    """
    cache: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] = {}
    for city, forecast_date, daily_variable in needed_keys:
        flat_key = f"{city}:{forecast_date.isoformat()}:{daily_variable}"
        record = forecasts.get(flat_key)
        if record is None:
            continue
        cache[(city, forecast_date, daily_variable)] = {
            "gfs_seamless": forecast_to_ensemble(record),
        }
    return cache


_DATE_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_snapshot_forecasts(snapshot_root: Path) -> SnapshotForecastIndex:
    """Build a look-ahead-free forecast index from the daily snapshot corpus.

    Walks ``snapshot_root`` for date-named directories (YYYY-MM-DD). Within
    each date dir it looks for either a flat ``weather_forecasts.json`` (legacy
    single-run layout) or run-id subdirectories each containing their own
    ``weather_forecasts.json`` (multi-run-per-day layout).

    Each record's ``run_ts`` is the minimum ``fetched_at`` across all records in
    that run dir — a conservative "available since" timestamp that avoids
    look-ahead bias. Records are sorted by run_ts ascending so
    ``select_run_for_entry`` can binary-scan for the latest run before entry.
    """
    by_key: dict[
        tuple[str, date, DailyVariable],
        list[tuple[float, str | None, dict[str, EnsembleForecast]]],
    ] = {}

    for date_dir in sorted(snapshot_root.iterdir()):
        if not date_dir.is_dir():
            continue
        if not _DATE_DIR_PATTERN.match(date_dir.name):
            continue
        try:
            date.fromisoformat(date_dir.name)
        except ValueError:
            continue

        run_dirs: list[tuple[str | None, Path]] = []

        flat_file = date_dir / "weather_forecasts.json"
        if flat_file.is_file():
            run_dirs.append((None, date_dir))
        else:
            for sub in sorted(date_dir.iterdir()):
                if sub.is_dir() and (sub / "weather_forecasts.json").is_file():
                    run_dirs.append((sub.name, sub))

        for run_id, run_dir in run_dirs:
            forecasts_path = run_dir / "weather_forecasts.json"
            try:
                with open(forecasts_path) as fh:
                    records: list[dict[str, Any]] = json.load(fh)
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping malformed snapshot file: %s", forecasts_path)
                continue

            if not isinstance(records, list) or not records:
                continue

            fetched_ats: list[float] = []
            per_key_models: dict[
                tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
            ] = {}

            for rec in records:
                try:
                    city = rec["city"]
                    fd = date.fromisoformat(rec["forecast_date"])
                    dv = DailyVariable(rec["daily_variable"])
                    temps: list[float] = rec["temperatures"]
                    if not temps:
                        continue
                    model: str = rec.get("model", "gfs_seamless")
                    fetched_at: float = float(rec["fetched_at"])
                    fetched_ats.append(fetched_at)
                except (KeyError, ValueError, TypeError):
                    continue

                fkey = (city, fd, dv)
                if fkey not in per_key_models:
                    per_key_models[fkey] = {}
                per_key_models[fkey][model] = EnsembleForecast(
                    city=city,
                    forecast_date=fd,
                    temperatures=temps,
                    model=model,
                    fetched_at=fetched_at,
                )

            if not fetched_ats:
                continue
            run_ts = min(fetched_ats)

            for fkey, model_dict in per_key_models.items():
                if fkey not in by_key:
                    by_key[fkey] = []
                by_key[fkey].append((run_ts, run_id, model_dict))

    for runs_list in by_key.values():
        runs_list.sort(key=lambda r: r[0])

    logger.info(
        "Loaded snapshot forecast index: %d unique (city, date, variable) keys",
        len(by_key),
    )
    return SnapshotForecastIndex(by_key=by_key)


def select_run_for_entry(
    runs: list[tuple[float, str | None, dict[str, EnsembleForecast]]],
    entry_ts: float,
    *,
    fallback: str = "none",
) -> tuple[float, str | None, dict[str, EnsembleForecast]] | None:
    """Return the latest snapshot run whose run_ts <= entry_ts.

    When no run predates the entry candle:
    - ``fallback="earliest"``: return the first (earliest) run anyway.
    - ``fallback="none"``: return None (skip the market).
    """
    if not runs:
        return None

    chosen: tuple[float, str | None, dict[str, EnsembleForecast]] | None = None
    for run in runs:
        if run[0] <= entry_ts:
            chosen = run
        else:
            break

    if chosen is not None:
        return chosen
    if fallback == "earliest":
        return runs[0]
    return None


def _eval_config_from(config: BacktestConfig) -> _EvalConfig:
    return _EvalConfig(
        max_spread=config.max_spread,
        spread_skill_sensitivity=config.spread_skill_sensitivity,
        edge_threshold=config.edge_threshold,
        max_edge=config.max_edge,
        kelly_fraction=config.kelly_fraction,
        max_pct=config.max_single_market_pct * config.max_total_exposure_pct,
        max_take_pct=1.0,  # unused in pure path (orderbook handling is async)
        depth_check_enabled=False,
        min_raw_prob_for_between=config.min_raw_prob_for_between,
        min_distance_from_strike_std=config.min_distance_from_strike_std,
        block_raw_prob_range=config.block_raw_prob_range,
        boundary_filter_cells=dict(config.boundary_filter_cells),
    )


def _resolve_candidate(
    candidate: _MarketCandidate,
    market: MarketRecord,
    entry: SimulatedEntry,
    config: BacktestConfig,
) -> BacktestTrade:
    """Turn a passing ``_MarketCandidate`` into a settled ``BacktestTrade``."""
    sizing = candidate.sizing
    buy_price = (
        candidate.exec_price if sizing.side == Side.YES else (1.0 - candidate.exec_price)
    )
    total_fee = (
        kalshi_order_fee(buy_price, candidate.contracts) if config.include_fees else 0.0
    )
    assert market.result is not None  # gated by replay_markets
    pnl = binary_contract_pnl(
        sizing.side, buy_price, candidate.contracts, market.result, total_fee=total_fee,
    )
    return BacktestTrade(
        ticker=market.ticker,
        city=market.city,
        daily_variable=market.daily_variable,
        strike_type=market.strike_type or "",
        floor_strike=market.floor_strike,
        cap_strike=market.cap_strike,
        model_prob=candidate.model_prob,
        market_price=entry.yes_price,
        side=sizing.side,
        raw_edge=candidate.raw_edge,
        effective_edge=candidate.effective_edge,
        ensemble_confidence=candidate.ensemble_confidence,
        ensemble_std=candidate.ensemble_std_pooled or 0.0,
        kelly_raw=sizing.kelly_raw,
        kelly_dollars=sizing.dollars,
        kelly_capped=sizing.capped,
        contracts=candidate.contracts,
        buy_price=buy_price,
        result=market.result,
        pnl=pnl,
        won=pnl > 0,
        total_fee=total_fee,
        entry_date=datetime.fromtimestamp(entry.candle_ts, tz=UTC).date(),
    )


def replay_markets(
    markets: list[MarketRecord],
    forecasts: dict[str, WeatherForecastRecord] | None = None,
    config: BacktestConfig | None = None,
    *,
    calibrator: IsotonicCalibrator | None = None,
    bias_table: BiasTable | None = None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
    ensemble_weights: dict[str, float] | None = None,
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
    snapshot_index: SnapshotForecastIndex | None = None,
) -> BacktestResult:
    """Replay historical markets against stored forecasts.

    Uses the production ``evaluate_market_pure`` + ``postprocess_forecasts``
    pipeline so every filter the live strategy applies (excluded_city_vars,
    isotonic calibration, strike-proximity gate, min_raw_prob_for_between,
    tail-probability clip, $0.15 YES floor, inverted-side guards) also runs
    in the backtest. Pass ``calibrator`` / ``bias_table`` to match live config.

    When ``config.data_source == "snapshot"`` pass ``snapshot_index`` (built
    with ``load_snapshot_forecasts``). The engine then resolves the latest
    snapshot run available BEFORE each market's entry candle timestamp, which
    eliminates look-ahead bias. Set ``config.forecast_lookup_fallback`` to
    ``"earliest"`` to use the first available run for markets whose entry candle
    predates all snapshot runs (useful for the oldest markets in the corpus).
    """
    if config is None:
        config = BacktestConfig()

    if config.data_source == "snapshot" and snapshot_index is None:
        raise ValueError("snapshot mode requires snapshot_index=load_snapshot_forecasts(...)")
    if config.data_source == "legacy" and forecasts is None:
        raise ValueError("legacy mode requires forecasts=load_forecasts(...)")

    if config.snapshot_entry_source == "snapshot_quotes":
        raise NotImplementedError("snapshot_quotes mode not yet implemented")

    if ensemble_weights is None:
        # Matches live config.yaml ensemble_weights. compute_blended_probability
        # normalizes to available models, so legacy GFS-only snapshot corpora
        # silently collapse to gfs_seamless=1.0 without a KeyError.
        ensemble_weights = {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45}

    result = BacktestResult(config=config)
    result.total_markets = len(markets)

    weather_client = WeatherClient()
    eval_cfg = _eval_config_from(config)

    # --- Pre-build scan candidates and collect forecast keys we need ---
    prepared: list[tuple[MarketRecord, SimulatedEntry, ScanCandidate]] = []
    needed_keys: set[tuple[str, date, DailyVariable]] = set()
    for market in markets:
        if market.result is None:
            result.skip_no_result += 1
            continue
        if market.candlestick_error:
            result.skip_candle_error += 1
            continue
        if not market.candlesticks:
            result.skip_no_candles += 1
            continue

        entry = simulate_entry_price(
            market,
            target_hours=config.target_hours_before_expiry,
            min_hours=config.min_hours_before_expiry,
            max_hours=config.max_hours_before_expiry,
        )
        if entry is None:
            result.skip_no_entry += 1
            continue

        sc = _build_scan_candidate(market, entry)
        if sc is None:
            result.skip_unmappable_market += 1
            continue

        prepared.append((market, entry, sc))
        needed_keys.add(
            (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)
        )

    excluded = list(config.excluded_city_vars)

    if config.data_source == "legacy":
        assert forecasts is not None  # validated above
        forecast_cache = _build_forecast_cache(forecasts, needed_keys)
        forecast_cache, bias_corrections, raw_stds, raw_medians = postprocess_forecasts(
            forecast_cache=forecast_cache,
            bias_table=bias_table,
            bias_shrinkage=bias_shrinkage,
            bias_max_correction_f=bias_max_correction_f,
            spread_inflation_overrides=spread_inflation_overrides,
        )

        for market, entry, sc in prepared:
            fkey = (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)
            if fkey not in forecast_cache:
                result.skip_no_forecast += 1
                continue

            funnel = ScanFunnel()
            candidate = evaluate_market_pure(
                sc,
                forecast_cache,
                bias_corrections,
                raw_stds,
                raw_medians,
                eval_cfg,
                calibrator=calibrator,
                weights=ensemble_weights,
                weather_client=weather_client,
                bankroll=config.bankroll,
                excluded_city_vars=excluded,
                spread_inflation_overrides=spread_inflation_overrides,
                funnel=funnel,
            )
            if candidate is None:
                mapped = False
                for reason_str, attr in _REJECT_REASON_TO_ATTR.items():
                    funnel_attr = funnel_attr_for(reason_str)
                    if funnel_attr and getattr(funnel, funnel_attr, 0) > 0:
                        setattr(result, attr, getattr(result, attr) + 1)
                        if attr == "skip_kelly_zero":
                            result.skip_zero_size += 1
                        mapped = True
                        break
                if not mapped:
                    result.skip_no_price_info += 1
                continue

            result.trades.append(_resolve_candidate(candidate, market, entry, config))

    else:
        assert snapshot_index is not None  # validated above

        for market, entry, sc in prepared:
            fkey = (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)

            runs = snapshot_index.runs_for(*fkey)
            if not runs:
                result.skip_no_forecast += 1
                continue

            chosen = select_run_for_entry(
                runs,
                float(entry.candle_ts),
                fallback=config.forecast_lookup_fallback,
            )
            if chosen is None:
                result.skip_no_pre_entry_run += 1
                continue

            _, _, model_dict = chosen
            single_forecast_cache: dict[
                tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
            ] = {fkey: dict(model_dict)}

            single_processed, bias_corr, raw_stds_s, raw_medians_s = postprocess_forecasts(
                forecast_cache=single_forecast_cache,
                bias_table=bias_table,
                bias_shrinkage=bias_shrinkage,
                bias_max_correction_f=bias_max_correction_f,
                spread_inflation_overrides=spread_inflation_overrides,
            )

            funnel = ScanFunnel()
            candidate = evaluate_market_pure(
                sc,
                single_processed,
                bias_corr,
                raw_stds_s,
                raw_medians_s,
                eval_cfg,
                calibrator=calibrator,
                weights=ensemble_weights,
                weather_client=weather_client,
                bankroll=config.bankroll,
                excluded_city_vars=excluded,
                spread_inflation_overrides=spread_inflation_overrides,
                funnel=funnel,
            )
            if candidate is None:
                mapped = False
                for reason_str, attr in _REJECT_REASON_TO_ATTR.items():
                    funnel_attr = funnel_attr_for(reason_str)
                    if funnel_attr and getattr(funnel, funnel_attr, 0) > 0:
                        setattr(result, attr, getattr(result, attr) + 1)
                        if attr == "skip_kelly_zero":
                            result.skip_zero_size += 1
                        mapped = True
                        break
                if not mapped:
                    result.skip_no_price_info += 1
                continue

            result.trades.append(_resolve_candidate(candidate, market, entry, config))

    logger.info(
        "Backtest complete: %d trades from %d markets (skipped: %d no_result, "
        "%d no_forecast, %d no_pre_entry_run, %d no_candles, %d no_entry, "
        "%d below_threshold/filter)",
        result.num_trades,
        result.total_markets,
        result.skip_no_result,
        result.skip_no_forecast,
        result.skip_no_pre_entry_run,
        result.skip_no_candles + result.skip_candle_error,
        result.skip_no_entry,
        result.skip_below_threshold,
    )

    return result
