"""Tests for the backtesting engine."""

from __future__ import annotations

import dataclasses
import json
import math
import statistics as st
import tempfile
from datetime import date
from pathlib import Path

from python.analytics.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    SnapshotForecastIndex,
    WalkForwardConfig,
    _compute_fold_windows,
    forecast_to_ensemble,
    load_forecasts,
    load_markets,
    load_snapshot_forecasts,
    replay_markets,
    select_run_for_entry,
    simulate_entry_price,
    walk_forward_replay,
)
from python.common import DailyVariable, Side
from python.data.weather import EnsembleForecast
from scripts.collect_backtest_data import (
    CandlestickRecord,
    MarketRecord,
    WeatherForecastRecord,
)

FIXTURES = Path(__file__).parent / "fixtures" / "backtest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market(
    ticker: str = "KXHIGHNY-T60",
    city: str = "NYC",
    strike_type: str = "greater",
    floor_strike: float | None = 60.0,
    cap_strike: float | None = None,
    result: str | None = "yes",
    expiration_ts: int = 1_741_651_200,
    candles: list[CandlestickRecord] | None = None,
    candlestick_error: str | None = None,
) -> MarketRecord:
    if candles is None:
        # Default: one candle at T-24h with yes_price_close=65 (0.65)
        candles = [
            CandlestickRecord(
                end_period_ts=expiration_ts - 24 * 3600,
                yes_price_close=65,
            )
        ]
    return MarketRecord(
        ticker=ticker,
        series_ticker="KXHIGHNY",
        city=city,
        daily_variable="temperature_2m_max",
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        result=result,
        expiration_ts=expiration_ts,
        expiration_date="2025-03-11",
        candlesticks=candles,
        candlestick_error=candlestick_error,
    )


def _forecast(
    city: str = "NYC",
    temperatures: list[float] | None = None,
) -> WeatherForecastRecord:
    if temperatures is None:
        # 25 of 31 members above 60 → model_prob ≈ 0.806
        temperatures = [62.0 + i * 0.5 for i in range(25)] + [58.0 - i * 0.5 for i in range(6)]
    return WeatherForecastRecord(
        city=city,
        forecast_date="2025-03-11",
        daily_variable="temperature_2m_max",
        temperatures=temperatures,
        mean_temperature=sum(temperatures) / len(temperatures),
        std_dev=1.0,
        member_count=len(temperatures),
        fetched_at=1_741_564_000.0,
    )


def _make_trade(
    side: Side = Side.YES,
    market_price: float = 0.65,
    result: str = "yes",
    contracts: int = 2,
    model_prob: float = 0.80,
) -> BacktestTrade:
    """Build a BacktestTrade with the correct P&L."""
    buy_price = market_price if side == Side.YES else (1.0 - market_price)
    won = (
        (side == Side.YES and result == "yes")
        or (side == Side.NO and result == "no")
    )
    pnl = (1.0 - buy_price) * contracts if won else -buy_price * contracts
    return BacktestTrade(
        ticker="TEST",
        city="NYC",
        daily_variable="temperature_2m_max",
        strike_type="greater",
        floor_strike=60.0,
        cap_strike=None,
        model_prob=model_prob,
        market_price=market_price,
        side=side,
        raw_edge=abs(model_prob - market_price),
        effective_edge=abs(model_prob - market_price),
        ensemble_confidence=1.0,
        ensemble_std=1.0,
        kelly_raw=0.05,
        kelly_dollars=50.0,
        kelly_capped=False,
        contracts=contracts,
        buy_price=buy_price,
        result=result,
        pnl=pnl,
        won=won,
    )


# ===========================================================================
# Entry price simulation
# ===========================================================================


class TestSimulateEntryPrice:
    """Tests for simulate_entry_price."""

    def test_finds_candle_at_target(self):
        """Selects the candle closest to T-24h."""
        exp_ts = 1_741_651_200
        market = _market(
            candles=[
                CandlestickRecord(end_period_ts=exp_ts - 24 * 3600, yes_price_close=65),
                CandlestickRecord(end_period_ts=exp_ts - 36 * 3600, yes_price_close=60),
            ],
        )
        entry = simulate_entry_price(market, target_hours=24.0)
        assert entry is not None
        assert entry.yes_price == 0.65
        assert entry.candle_ts == exp_ts - 24 * 3600
        assert abs(entry.hours_before_expiry - 24.0) < 0.01

    def test_falls_back_to_nearest(self):
        """If no candle at exactly T-24h, picks the closest within window."""
        exp_ts = 1_741_651_200
        market = _market(
            candles=[
                CandlestickRecord(end_period_ts=exp_ts - 20 * 3600, yes_price_close=70),
                CandlestickRecord(end_period_ts=exp_ts - 30 * 3600, yes_price_close=55),
            ],
        )
        entry = simulate_entry_price(market, target_hours=24.0)
        assert entry is not None
        # T-20h is closer to target T-24h than T-30h
        assert entry.yes_price == 0.70

    def test_skips_empty_candles(self):
        """Returns None when market has no candlesticks."""
        market = _market(candles=[])
        assert simulate_entry_price(market) is None

    def test_skips_error_market(self):
        """Returns None when candlestick_error is set."""
        market = _market(candlestick_error="HTTP 404")
        assert simulate_entry_price(market) is None

    def test_respects_time_window(self):
        """Candles outside [min_hours, max_hours] are excluded."""
        exp_ts = 1_741_651_200
        market = _market(
            candles=[
                # At T-6h (outside min_hours=12)
                CandlestickRecord(end_period_ts=exp_ts - 6 * 3600, yes_price_close=80),
                # At T-60h (outside max_hours=48)
                CandlestickRecord(end_period_ts=exp_ts - 60 * 3600, yes_price_close=40),
            ],
        )
        entry = simulate_entry_price(market, target_hours=24.0, min_hours=12.0, max_hours=48.0)
        assert entry is None


# ===========================================================================
# P&L computation
# ===========================================================================


class TestPnlComputation:
    """Tests for binary contract P&L math."""

    def test_yes_buy_yes_result(self):
        """Buy YES at 0.65, result YES → profit (1-0.65)*2 = $0.70."""
        trade = _make_trade(side=Side.YES, market_price=0.65, result="yes", contracts=2)
        assert trade.won is True
        assert abs(trade.pnl - 0.70) < 0.001

    def test_yes_buy_no_result(self):
        """Buy YES at 0.65, result NO → loss -0.65*2 = -$1.30."""
        trade = _make_trade(side=Side.YES, market_price=0.65, result="no", contracts=2)
        assert trade.won is False
        assert abs(trade.pnl - (-1.30)) < 0.001

    def test_no_buy_no_result(self):
        """Buy NO at (1-0.55)=0.45, result NO → profit (1-0.45)*3 = $1.65."""
        trade = _make_trade(side=Side.NO, market_price=0.55, result="no", contracts=3)
        assert trade.won is True
        assert abs(trade.pnl - 1.65) < 0.001

    def test_no_buy_yes_result(self):
        """Buy NO at (1-0.55)=0.45, result YES → loss -0.45*3 = -$1.35."""
        trade = _make_trade(side=Side.NO, market_price=0.55, result="yes", contracts=3)
        assert trade.won is False
        assert abs(trade.pnl - (-1.35)) < 0.001


# ===========================================================================
# Replay logic
# ===========================================================================


class TestReplayMarkets:
    """Tests for the replay_markets function."""

    def test_basic_trade_generated(self):
        """A market with forecast and sufficient edge produces a trade."""
        markets = [_market()]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast()}
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(
                edge_threshold=0.05,
                min_distance_from_strike_std=0.0,
                block_raw_prob_range=None,
            ),
        )

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.ticker == "KXHIGHNY-T60"
        assert trade.city == "NYC"
        assert trade.side in (Side.YES, Side.NO)
        assert trade.contracts >= 1
        assert trade.won is True  # model says high prob > 60, result is "yes"

    def test_skip_on_missing_forecast(self):
        """Markets with no matching forecast are skipped."""
        markets = [_market()]
        result = replay_markets(markets, {}, BacktestConfig())

        assert result.num_trades == 0
        assert result.skip_no_forecast == 1

    def test_skip_on_unsettled(self):
        """Markets with result=None are skipped."""
        markets = [_market(result=None)]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast()}
        result = replay_markets(markets, forecasts, BacktestConfig())

        assert result.num_trades == 0
        assert result.skip_no_result == 1

    def test_edge_threshold_filtering(self):
        """Markets below edge_threshold are skipped via some filter gate."""
        # 15 members at 59°F, 1 at 60°F, 15 at 61°F → fraction_above(60) ≈ 0.484
        # Market price 0.50. This near-50/50 distribution trips the strike-distance
        # filter (std is small, strike is near the mean) before the edge threshold
        # check; confirmed by granular diagnostics. The test asserts the market is
        # rejected and no trade is recorded.
        near_split_temps = [59.0] * 15 + [60.0] + [61.0] * 15
        exp_ts = 1_741_651_200
        markets = [_market(
            candles=[CandlestickRecord(end_period_ts=exp_ts - 24 * 3600, yes_price_close=50)],
        )]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast(temperatures=near_split_temps)}
        result = replay_markets(markets, forecasts, BacktestConfig(edge_threshold=0.08))

        assert result.num_trades == 0
        assert result.skip_strike_distance == 1
        assert result.skip_below_threshold == 0

    def test_max_edge_filtering(self):
        """Markets above max_edge are skipped and counted in skip_above_max_edge."""
        # 3 low + 28 high members → raw_prob ≈ 0.903, effective_edge ≈ 0.30.
        # max_edge=0.10 forces rejection. Plateau block is disabled so the
        # market reaches the edge check rather than being caught upstream.
        mixed = [55.0] * 3 + [75.0] * 28
        exp_ts = 1_741_651_200
        markets = [_market(
            candles=[CandlestickRecord(end_period_ts=exp_ts - 24 * 3600, yes_price_close=30)],
        )]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast(temperatures=mixed)}
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(max_edge=0.10, block_raw_prob_range=None),
        )

        assert result.num_trades == 0
        assert result.skip_above_max_edge == 1
        assert result.skip_below_threshold == 0

    def test_skip_excluded_city_var_counted(self):
        """skip_excluded_city_var increments when the city/var is excluded."""
        markets = [_market(city="NYC", strike_type="greater", floor_strike=60.0)]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast()}
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(excluded_city_vars=("NYC_high",)),
        )
        assert result.num_trades == 0
        assert result.skip_excluded_city_var >= 1

    def test_skip_above_max_edge_not_in_below_threshold(self):
        """EDGE_ABOVE_MAX rejection goes to skip_above_max_edge, not skip_below_threshold."""
        mixed = [55.0] * 3 + [75.0] * 28
        exp_ts = 1_741_651_200
        markets = [_market(
            candles=[CandlestickRecord(end_period_ts=exp_ts - 24 * 3600, yes_price_close=30)],
        )]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast(temperatures=mixed)}
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(max_edge=0.10, block_raw_prob_range=None),
        )

        assert result.num_trades == 0
        assert result.skip_above_max_edge == 1
        assert result.skip_below_threshold == 0

    def test_between_market(self):
        """Between-type markets are handled correctly."""
        # NYC between 40-50 (1.0x spread inflation so raw_prob is preserved):
        # 1 member @ 45 inside bracket, rest @ 55 → raw_prob = 1/31 ≈ 0.032
        # (above min_raw_prob_for_between=0.02). Market price 0.30 →
        # model_prob < midpoint → NO side with edge ≈ 0.25.
        temps = [45.0] + [55.0] * 30
        exp_ts = 1_741_651_200
        markets = [_market(
            ticker="KXHIGHNY-B40-50",
            city="NYC",
            strike_type="between",
            floor_strike=40.0,
            cap_strike=50.0,
            result="no",
            candles=[CandlestickRecord(end_period_ts=exp_ts - 24 * 3600, yes_price_close=30)],
        )]
        forecasts = {"NYC:2025-03-11:temperature_2m_max": _forecast(
            city="NYC", temperatures=temps,
        )}
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(max_edge=1.0, min_raw_prob_for_between=0.02),
        )

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.side == Side.NO
        assert trade.won is True


# ===========================================================================
# Calibration
# ===========================================================================


class TestCalibration:
    """Tests for calibration bucket computation."""

    def test_bucket_counts(self):
        """Trades are placed in the correct calibration buckets."""
        result = BacktestResult(trades=[
            _make_trade(model_prob=0.25, result="no"),
            _make_trade(model_prob=0.25, result="yes"),
            _make_trade(model_prob=0.75, result="yes"),
            _make_trade(model_prob=0.75, result="yes"),
            _make_trade(model_prob=0.75, result="no"),
        ])
        buckets = result.calibration_buckets(n_bins=10)

        # 0.2-0.3 bucket should have 2 trades
        bucket_2 = buckets[2]  # index 2 = [0.2, 0.3)
        assert bucket_2.count == 2
        assert abs(bucket_2.actual_yes_rate - 0.5) < 0.01

        # 0.7-0.8 bucket should have 3 trades
        bucket_7 = buckets[7]  # index 7 = [0.7, 0.8)
        assert bucket_7.count == 3
        assert abs(bucket_7.actual_yes_rate - 2.0 / 3.0) < 0.01

    def test_mae_computation(self):
        """MAE is the mean of |actual - model| across non-empty buckets."""
        result = BacktestResult(trades=[
            _make_trade(model_prob=0.25, result="yes"),   # actual=1.0, delta=0.75
            _make_trade(model_prob=0.75, result="yes"),   # actual=1.0, delta=0.25
        ])
        mae = result.calibration_mae(n_bins=10)
        # Two non-empty buckets with deltas 0.75 and 0.25 → MAE = 0.50
        assert abs(mae - 0.50) < 0.01


# ===========================================================================
# Summary statistics
# ===========================================================================


class TestSummaryStats:
    """Tests for BacktestResult computed properties."""

    def test_sharpe_with_known_trades(self):
        """Sharpe ratio computed correctly for known P&L sequence."""
        import statistics as st

        trades = [
            _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1),
            _make_trade(side=Side.YES, market_price=0.50, result="no", contracts=1),
            _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1),
        ]
        pnls = [t.pnl for t in trades]
        expected_sharpe = (st.mean(pnls) / st.stdev(pnls)) * math.sqrt(252)

        result = BacktestResult(trades=trades)
        assert abs(result.sharpe_ratio - expected_sharpe) < 0.001

    def test_max_drawdown_sequence(self):
        """Max drawdown is peak-to-trough in cumulative P&L."""
        # P&L: +10, +5, -8, -12, +20 → cumulative: 10, 15, 7, -5, 15
        # Peak at 15, trough at -5 → drawdown = 20
        base = _make_trade(
            side=Side.YES, market_price=0.50, result="yes", contracts=1,
        )
        fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
        trades_seq = [
            BacktestTrade(**{**fields, "pnl": 10.0, "won": True}),
            BacktestTrade(**{**fields, "pnl": 5.0, "won": True}),
            BacktestTrade(**{**fields, "pnl": -8.0, "won": False}),
            BacktestTrade(**{**fields, "pnl": -12.0, "won": False}),
            BacktestTrade(**{**fields, "pnl": 20.0, "won": True}),
        ]
        result = BacktestResult(trades=trades_seq)
        assert abs(result.max_drawdown - 20.0) < 0.001

    def test_empty_trades_no_crash(self):
        """All stats return sensible defaults on empty trade list."""
        result = BacktestResult()
        assert result.total_pnl == 0.0
        assert result.win_rate == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.roi == 0.0
        assert result.calibration_buckets() == []
        assert result.edge_buckets() == []

    def test_sharpe_aggregates_per_day(self):
        """Same-day trades aggregate: equal daily P&L → stdev=0 → sharpe=0.0."""
        base = _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1)
        day1 = date(2025, 3, 11)
        day2 = date(2025, 3, 12)
        t1 = dataclasses.replace(base, pnl=0.60, entry_date=day1)
        t2 = dataclasses.replace(base, pnl=0.40, entry_date=day1)
        t3 = dataclasses.replace(base, pnl=1.00, entry_date=day2)
        result = BacktestResult(trades=[t1, t2, t3])
        assert result.sharpe_ratio == 0.0

    def test_sharpe_single_day_returns_zero(self):
        """Two trades on the same day → only 1 daily observation → sharpe=0.0."""
        base = _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1)
        day1 = date(2025, 3, 11)
        t1 = dataclasses.replace(base, pnl=0.50, entry_date=day1)
        t2 = dataclasses.replace(base, pnl=0.50, entry_date=day1)
        result = BacktestResult(trades=[t1, t2])
        assert result.sharpe_ratio == 0.0

    def test_sharpe_legacy_fallback_when_entry_date_missing(self):
        """Trades built without entry_date fall back to per-trade Sharpe computation."""
        trades = [
            _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1),
            _make_trade(side=Side.YES, market_price=0.50, result="no", contracts=1),
            _make_trade(side=Side.YES, market_price=0.50, result="yes", contracts=1),
        ]
        pnls = [t.pnl for t in trades]
        expected = (st.mean(pnls) / st.stdev(pnls)) * math.sqrt(252)
        result = BacktestResult(trades=trades)
        assert abs(result.sharpe_ratio - expected) < 0.001


# ===========================================================================
# Data loading
# ===========================================================================


class TestDataLoading:
    """Tests for load_markets and load_forecasts."""

    def test_load_markets(self):
        """Fixture markets.json deserializes correctly."""
        markets = load_markets(FIXTURES / "markets.json")
        assert len(markets) == 5
        assert markets[0].ticker == "KXHIGHNY-26MAR10-T60"
        assert markets[0].city == "NYC"
        assert markets[0].result == "yes"
        assert len(markets[0].candlesticks) == 2
        assert markets[0].candlesticks[0].yes_price_close == 65

    def test_load_forecasts(self):
        """Fixture weather_forecasts.json deserializes correctly."""
        forecasts = load_forecasts(FIXTURES / "weather_forecasts.json")
        assert len(forecasts) == 4
        nyc = forecasts["NYC:2025-03-11:temperature_2m_max"]
        assert nyc.city == "NYC"
        assert nyc.member_count == 31
        assert len(nyc.temperatures) == 31


# ===========================================================================
# forecast_to_ensemble bridge
# ===========================================================================


class TestForecastToEnsemble:
    """Test the forecast → EnsembleForecast bridge."""

    def test_converts_correctly(self):
        record = _forecast()
        ensemble = forecast_to_ensemble(record)
        assert ensemble.city == "NYC"
        assert len(ensemble.temperatures) == 31
        assert ensemble.member_count == 31


# ===========================================================================
# Integration
# ===========================================================================


class TestIntegration:
    """End-to-end test with fixture data."""

    def test_full_pipeline(self):
        """Load fixtures, replay, and verify basic sanity."""
        markets = load_markets(FIXTURES / "markets.json")
        forecasts = load_forecasts(FIXTURES / "weather_forecasts.json")
        result = replay_markets(
            markets, forecasts,
            BacktestConfig(
                max_edge=1.0,
                min_distance_from_strike_std=0.0,
                min_raw_prob_for_between=0.02,
                block_raw_prob_range=None,
            ),
        )

        # Market 5 has result=None → skipped
        assert result.skip_no_result == 1

        # Should have at least some trades
        assert result.num_trades >= 1

        # Every trade should have a valid side and non-zero contracts
        for trade in result.trades:
            assert trade.side in (Side.YES, Side.NO)
            assert trade.contracts >= 1
            assert trade.result in ("yes", "no")

        # Total markets should match input
        assert result.total_markets == 5

        # Win rate should be between 0 and 1
        assert 0.0 <= result.win_rate <= 1.0


# ===========================================================================
# Snapshot mode
# ===========================================================================

_EXP_TS = 1_741_651_200  # reuse same expiration_ts as _market helper
_ENTRY_TS = _EXP_TS - 24 * 3600  # candle at T-24h


def _make_snapshot_index(
    run_ts: float,
    model_dict: dict[str, EnsembleForecast],
    city: str = "NYC",
    forecast_date_str: str = "2025-03-11",
    daily_variable: str = "temperature_2m_max",
) -> SnapshotForecastIndex:
    """Build a minimal SnapshotForecastIndex with one run for one key."""
    fd = date.fromisoformat(forecast_date_str)
    dv = DailyVariable(daily_variable)
    fkey = (city, fd, dv)
    return SnapshotForecastIndex(by_key={fkey: [(run_ts, None, model_dict)]})


def _ensemble(temperatures: list[float] | None = None) -> EnsembleForecast:
    if temperatures is None:
        temperatures = [62.0 + i * 0.5 for i in range(25)] + [58.0 - i * 0.5 for i in range(6)]
    return EnsembleForecast(
        city="NYC",
        forecast_date=date(2025, 3, 11),
        temperatures=temperatures,
        model="gfs_seamless",
        fetched_at=1_741_564_000.0,
    )


class TestSnapshotMode:
    """Tests for snapshot corpus mode with look-ahead-free run selection."""

    def test_replay_snapshot_mode_blocks_lookahead(self):
        """Run with run_ts AFTER entry_ts should be blocked (fallback=none)."""
        run_ts_after_entry = float(_ENTRY_TS) + 3600.0  # 1h after entry candle
        idx = _make_snapshot_index(
            run_ts=run_ts_after_entry, model_dict={"gfs_seamless": _ensemble()},
        )

        markets = [_market()]
        result = replay_markets(
            markets,
            config=BacktestConfig(
                data_source="snapshot",
                forecast_lookup_fallback="none",
                edge_threshold=0.05,
                min_distance_from_strike_std=0.0,
                block_raw_prob_range=None,
            ),
            snapshot_index=idx,
        )

        assert result.skip_no_pre_entry_run == 1
        assert result.num_trades == 0

    def test_replay_snapshot_mode_fallback_earliest(self):
        """Run with run_ts AFTER entry_ts but fallback=earliest should use it."""
        run_ts_after_entry = float(_ENTRY_TS) + 3600.0
        idx = _make_snapshot_index(
            run_ts=run_ts_after_entry, model_dict={"gfs_seamless": _ensemble()},
        )

        markets = [_market()]
        result = replay_markets(
            markets,
            config=BacktestConfig(
                data_source="snapshot",
                forecast_lookup_fallback="earliest",
                edge_threshold=0.05,
                min_distance_from_strike_std=0.0,
                block_raw_prob_range=None,
            ),
            snapshot_index=idx,
        )

        assert result.skip_no_pre_entry_run == 0

    def test_replay_snapshot_mode_no_forecasts(self):
        """Snapshot index with no data for the market key should skip_no_forecast."""
        idx = SnapshotForecastIndex(by_key={})  # empty index

        markets = [_market()]
        result = replay_markets(
            markets,
            config=BacktestConfig(
                data_source="snapshot",
                edge_threshold=0.05,
            ),
            snapshot_index=idx,
        )

        assert result.skip_no_forecast == 1
        assert result.num_trades == 0

    def test_load_snapshot_forecasts_multi_run(self):
        """Two run dirs each with a weather_forecasts.json produce 2 entries per key."""
        temps = [62.0] * 31
        rec = {
            "city": "NYC",
            "forecast_date": "2025-03-11",
            "daily_variable": "temperature_2m_max",
            "temperatures": temps,
            "mean": 62.0,
            "std_dev": 0.0,
            "fetched_at": 1_741_500_000.0,
            "model": "gfs_seamless",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "snapshots"
            date_dir = root / "2025-03-10"

            run1 = date_dir / "0200"
            run1.mkdir(parents=True)
            rec1 = {**rec, "fetched_at": 1_741_500_000.0}
            with open(run1 / "weather_forecasts.json", "w") as fh:
                json.dump([rec1], fh)

            run2 = date_dir / "1400"
            run2.mkdir(parents=True)
            rec2 = {**rec, "fetched_at": 1_741_543_200.0}
            with open(run2 / "weather_forecasts.json", "w") as fh:
                json.dump([rec2], fh)

            idx = load_snapshot_forecasts(root)

        fkey = (
            "NYC",
            date(2025, 3, 11),
            DailyVariable("temperature_2m_max"),
        )
        assert fkey in idx.by_key
        runs = idx.by_key[fkey]
        assert len(runs) == 2
        assert runs[0][0] < runs[1][0], "runs should be sorted by run_ts ascending"

    def test_select_run_for_entry_picks_latest_before(self):
        """select_run_for_entry returns the latest run still <= entry_ts."""
        ensemble = _ensemble()
        runs = [
            (1000.0, None, {"gfs_seamless": ensemble}),
            (2000.0, None, {"gfs_seamless": ensemble}),
            (3000.0, None, {"gfs_seamless": ensemble}),
        ]
        chosen = select_run_for_entry(runs, entry_ts=2500.0)
        assert chosen is not None
        assert chosen[0] == 2000.0

    def test_select_run_for_entry_none_when_all_after(self):
        """Returns None when all runs are after entry_ts (fallback=none)."""
        ensemble = _ensemble()
        runs = [(5000.0, None, {"gfs_seamless": ensemble})]
        assert select_run_for_entry(runs, entry_ts=1000.0, fallback="none") is None

    def test_select_run_for_entry_empty_returns_none(self):
        """Empty run list always returns None."""
        assert select_run_for_entry([], entry_ts=9999.0) is None

    def test_snapshot_mode_requires_index(self):
        """replay_markets raises ValueError when snapshot_index is missing."""
        import pytest

        markets = [_market()]
        with pytest.raises(ValueError, match="snapshot mode requires snapshot_index"):
            replay_markets(
                markets,
                config=BacktestConfig(data_source="snapshot"),
            )

    def test_legacy_mode_requires_forecasts(self):
        """replay_markets raises ValueError when forecasts is missing in legacy mode."""
        import pytest

        markets = [_market()]
        with pytest.raises(ValueError, match="legacy mode requires forecasts"):
            replay_markets(markets, config=BacktestConfig(data_source="legacy"))


# ===========================================================================
# Walk-forward validation
# ===========================================================================


class TestWalkForward:
    """Tests for walk-forward validation helpers and the main entry point."""

    def test_compute_fold_windows_7_dates(self):
        """7 dates with default config (min_train=5, test=1, step=1) → 2 folds."""
        dates = [date(2025, 3, i) for i in range(1, 8)]  # Mar 1-7
        config = WalkForwardConfig(min_train_days=5, test_window_days=1, step_days=1)
        folds = _compute_fold_windows(dates, config)

        assert len(folds) == 2

        # Fold 0: train Mar 1-5, test Mar 6
        train_start0, train_end0, test_start0, test_end0 = folds[0]
        assert train_start0 == date(2025, 3, 1)
        assert train_end0 == date(2025, 3, 5)
        assert test_start0 == date(2025, 3, 6)
        assert test_end0 == date(2025, 3, 6)

        # Fold 1: train Mar 1-6 (expanding), test Mar 7
        train_start1, train_end1, test_start1, test_end1 = folds[1]
        assert train_start1 == date(2025, 3, 1)  # expanding: always starts at 0
        assert train_end1 == date(2025, 3, 6)
        assert test_start1 == date(2025, 3, 7)
        assert test_end1 == date(2025, 3, 7)

    def test_compute_fold_windows_insufficient_data(self):
        """3 dates with min_train_days=5 → empty list (not enough dates)."""
        dates = [date(2025, 3, i) for i in range(1, 4)]
        config = WalkForwardConfig(min_train_days=5, test_window_days=1, step_days=1)
        folds = _compute_fold_windows(dates, config)
        assert folds == []

    def test_walk_forward_replay_returns_correct_fold_count(self):
        """7 settled markets on different dates → 2 folds with default WF config."""
        base_ts = 1_741_651_200  # 2025-03-11 UTC
        one_day = 86400

        # Build 7 markets across 7 consecutive dates (Mar 11-17)
        markets = []
        forecasts: dict[str, object] = {}
        for i in range(7):
            exp_ts = base_ts + i * one_day
            exp_date = date(2025, 3, 11 + i)
            exp_date_str = exp_date.isoformat()
            ticker = f"KXHIGHNY-T{60 + i}"
            candle_ts = exp_ts - 24 * 3600
            markets.append(MarketRecord(
                ticker=ticker,
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=60.0,
                cap_strike=None,
                result="yes",
                expiration_ts=exp_ts,
                expiration_date=exp_date_str,
                candlesticks=[CandlestickRecord(end_period_ts=candle_ts, yes_price_close=65)],
                candlestick_error=None,
            ))
            fkey = f"NYC:{exp_date_str}:temperature_2m_max"
            forecasts[fkey] = WeatherForecastRecord(
                city="NYC",
                forecast_date=exp_date_str,
                daily_variable="temperature_2m_max",
                temperatures=[62.0 + j * 0.5 for j in range(25)] + [58.0 - j * 0.5 for j in range(6)],
                mean_temperature=62.0,
                std_dev=1.0,
                member_count=31,
                fetched_at=float(candle_ts - 3600),
            )

        wf_result = walk_forward_replay(
            markets,
            forecasts,  # type: ignore[arg-type]
            BacktestConfig(
                edge_threshold=0.05,
                min_distance_from_strike_std=0.0,
                block_raw_prob_range=None,
            ),
            WalkForwardConfig(min_train_days=5, test_window_days=1, step_days=1),
        )

        assert wf_result.skipped_reason is None
        assert len(wf_result.folds) == 2
        assert wf_result.total_unique_dates == 7

    def test_walk_forward_insufficient_data_returns_skipped(self):
        """Fewer settled dates than min_train_days + test_window_days → skipped_reason set."""
        base_ts = 1_741_651_200
        one_day = 86400

        # Only 3 markets (3 dates) but min_train=5 needs 6 total
        markets = []
        forecasts: dict[str, object] = {}
        for i in range(3):
            exp_ts = base_ts + i * one_day
            exp_date = date(2025, 3, 11 + i)
            exp_date_str = exp_date.isoformat()
            ticker = f"KXHIGHNY-T{60 + i}"
            candle_ts = exp_ts - 24 * 3600
            markets.append(MarketRecord(
                ticker=ticker,
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=60.0,
                cap_strike=None,
                result="yes",
                expiration_ts=exp_ts,
                expiration_date=exp_date_str,
                candlesticks=[CandlestickRecord(end_period_ts=candle_ts, yes_price_close=65)],
                candlestick_error=None,
            ))
            fkey = f"NYC:{exp_date_str}:temperature_2m_max"
            forecasts[fkey] = WeatherForecastRecord(
                city="NYC",
                forecast_date=exp_date_str,
                daily_variable="temperature_2m_max",
                temperatures=[62.0] * 31,
                mean_temperature=62.0,
                std_dev=0.0,
                member_count=31,
                fetched_at=float(candle_ts - 3600),
            )

        wf_result = walk_forward_replay(
            markets,
            forecasts,  # type: ignore[arg-type]
            BacktestConfig(),
            WalkForwardConfig(min_train_days=5, test_window_days=1, step_days=1),
        )

        assert wf_result.skipped_reason == "insufficient_data"
        assert len(wf_result.folds) == 0
        assert wf_result.total_unique_dates == 3

    def test_walk_forward_replay_snapshot_mode_returns_correct_fold_count(self):
        """Snapshot-mode walk-forward: 7 dates → 2 folds, forecasts flow via index.

        Mirrors test_walk_forward_replay_returns_correct_fold_count but exercises
        ``data_source="snapshot"``: forecasts arrive through a 7-key
        SnapshotForecastIndex (one run per date, fetched before the entry candle so
        run selection is look-ahead-free) instead of the legacy flat dict. The
        snapshot index is threaded directly into each fold's replay_markets call
        (backtest.py walk_forward_replay), bypassing _filter_by_dates; per-market
        look-ahead is the only guard. Guards against a vacuous pass with
        num_trades >= 1 (forecasts reached the model) AND skip_no_forecast == 0
        (no market silently missed its index key).
        """
        base_ts = 1_741_651_200  # 2025-03-11 UTC
        one_day = 86400
        dv = DailyVariable("temperature_2m_max")

        markets: list[MarketRecord] = []
        by_key: dict = {}
        for i in range(7):
            exp_ts = base_ts + i * one_day
            exp_date = date(2025, 3, 11 + i)
            exp_date_str = exp_date.isoformat()
            ticker = f"KXHIGHNY-T{60 + i}"
            candle_ts = exp_ts - 24 * 3600
            markets.append(MarketRecord(
                ticker=ticker,
                series_ticker="KXHIGHNY",
                city="NYC",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=60.0,
                cap_strike=None,
                result="yes",
                expiration_ts=exp_ts,
                expiration_date=exp_date_str,  # per-date: _filter_by_dates routes folds on this
                candlesticks=[CandlestickRecord(end_period_ts=candle_ts, yes_price_close=65)],
                candlestick_error=None,
            ))
            # One run per date, fetched 1h before the entry candle so
            # select_run_for_entry (run_ts <= entry_ts) picks it.
            run_ts = float(candle_ts - 3600)
            temps = [62.0 + j * 0.5 for j in range(25)] + [58.0 - j * 0.5 for j in range(6)]
            ensemble = EnsembleForecast(
                city="NYC",
                forecast_date=exp_date,
                temperatures=temps,
                model="gfs_seamless",
                fetched_at=run_ts,
            )
            by_key[("NYC", exp_date, dv)] = [(run_ts, None, {"gfs_seamless": ensemble})]

        idx = SnapshotForecastIndex(by_key=by_key)

        wf_result = walk_forward_replay(
            markets,
            None,  # snapshot mode: forecasts come from the index, not a flat dict
            BacktestConfig(
                data_source="snapshot",
                edge_threshold=0.05,
                min_distance_from_strike_std=0.0,
                block_raw_prob_range=None,
            ),
            WalkForwardConfig(min_train_days=5, test_window_days=1, step_days=1),
            snapshot_index=idx,
        )

        assert wf_result.skipped_reason is None
        assert len(wf_result.folds) == 2
        assert wf_result.total_unique_dates == 7
        # Anti-vacuous-pass: prove the snapshot index was actually consulted.
        assert sum(f.result.num_trades for f in wf_result.folds) >= 1
        assert sum(f.result.skip_no_forecast for f in wf_result.folds) == 0
