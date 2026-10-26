from __future__ import annotations

import math

import numpy as np

from python.analytics.edge_audit import (
    ReplayInput,
    SettledTrade,
    day_clustered_bootstrap_ci,
    edge_attribution,
    model_vs_market_scores,
    outcome_yes,
    replay_one,
    roi,
    stationary_block_bootstrap_ci,
    trades_from_rows,
)
from python.analytics.edge_audit import (
    edge_monotonicity as run_t1,
)


def _row(**kw):
    base = dict(
        market_ticker="KXHIGHNY-26APR22-B57.5", timestamp="2026-04-22T14:00:00Z",
        model_probability=0.133, market_price=0.325, mid_at_signal=0.325,
        edge=0.19, won=True, pnl=113.63, dollars=117.44, side="NO",
        strike_type="between", contracts=367, exec_price=0.32, fee=5.64,
        settled=True, strategy="weather_edge",
    )
    base.update(kw)
    return base


def test_outcome_yes_no_side_win_means_yes_did_not_happen():
    # Bet NO and won => YES did not occur => outcome_yes == 0
    assert outcome_yes(side="NO", won=True) == 0
    assert outcome_yes(side="NO", won=False) == 1
    assert outcome_yes(side="YES", won=True) == 1
    assert outcome_yes(side="YES", won=False) == 0


def test_trades_from_rows_filters_to_settled_weather_edge():
    rows = [
        _row(),
        _row(settled=False),                  # dropped: unsettled
        _row(strategy="favorite_bias"),       # kept but tagged (T4 needs it)
        _row(pnl=None, dollars=None),         # dropped: missing pnl
    ]
    trades = trades_from_rows(rows)
    assert len(trades) == 2                    # settled rows with pnl present
    t = trades[0]
    assert isinstance(t, SettledTrade)
    assert t.outcome_yes == 0
    assert t.trading_day == "2026-04-22"
    assert math.isclose(t.wagered, 117.44)
    assert t.strategy == "weather_edge"
    assert trades[1].strategy == "favorite_bias"


def test_trades_from_rows_skips_row_with_no_market_price():
    # Row with neither mid_at_signal nor market_price must be silently skipped.
    row_no_price = _row()
    del row_no_price["mid_at_signal"]
    del row_no_price["market_price"]
    rows = [_row(), row_no_price]
    trades = trades_from_rows(rows)
    assert len(trades) == 1


def test_roi_is_sum_pnl_over_sum_wagered():
    assert math.isclose(roi([10.0, -5.0], [100.0, 100.0]), 5.0 / 200.0)
    assert roi([], []) == 0.0


def test_day_clustered_ci_is_deterministic_and_brackets_point():
    # 30 days, each clearly positive ROI => CI lower bound should be > 0
    days = [f"2026-04-{d:02d}" for d in range(1, 31)]
    pnl = [5.0] * 30
    wag = [100.0] * 30
    lo, hi = day_clustered_bootstrap_ci(days, pnl, wag, b=2000, seed=42)
    lo2, hi2 = day_clustered_bootstrap_ci(days, pnl, wag, b=2000, seed=42)
    assert (lo, hi) == (lo2, hi2)              # deterministic given seed
    assert lo > 0.0 and hi >= lo               # clearly-positive corpus


def test_stationary_block_ci_runs_and_is_seeded():
    series = list(np.linspace(-0.2, 0.2, 40))
    lo, hi = stationary_block_bootstrap_ci(series, mean_block=7, b=2000, seed=7)
    lo2, hi2 = stationary_block_bootstrap_ci(series, mean_block=7, b=2000, seed=7)
    assert (lo, hi) == (lo2, hi2)
    assert lo <= hi


def _mk_trade(day, edge, pnl, wagered=100.0, oy=0, strat="weather_edge", st="between"):
    return SettledTrade(
        ticker="T", trading_day=day, model_p_yes=0.1, market_p_yes=0.3,
        claimed_edge=edge, outcome_yes=oy, pnl=pnl, wagered=wagered,
        side="NO", strike_type=st, strategy=strat,
    )


def test_t1_detects_no_monotonicity_when_high_edge_loses():
    # high claimed-edge cohort LOSES => not monotone, top bucket CI_lo not > 0
    trades = []
    for i in range(40):
        day = f"2026-04-{(i % 20) + 1:02d}"
        trades.append(_mk_trade(day, edge=0.05, pnl=+3.0))    # low edge, wins
        trades.append(_mk_trade(day, edge=0.30, pnl=-4.0))    # high edge, loses
    res = run_t1(trades, n_bins=2, seed=42)
    assert res["monotone_increasing"] is False
    assert res["top_bucket_ci_lo"] <= 0.0
    assert res["passes"] is False


def test_t2_rewards_model_when_it_is_closer_to_truth():
    # Construct trades where model_p_yes is always closer to outcome than market.
    trades = []
    for i in range(60):
        day = f"2026-04-{(i % 20) + 1:02d}"
        t = _mk_trade(day, edge=0.1, pnl=0.0, oy=0)
        # outcome_yes=0; model says 0.10 (close), market says 0.40 (far)
        t = t.__class__(**{**t.__dict__, "model_p_yes": 0.10, "market_p_yes": 0.40})
        trades.append(t)
    res = model_vs_market_scores(trades, seed=42)
    assert res["model_brier"] < res["market_brier"]
    assert res["model_logscore"] < res["market_logscore"]
    # model beats market AND the difference CI excludes 0 (lower brier => negative diff)
    assert res["brier_diff_ci_hi"] < 0.0
    assert res["model_beats_market"] is True


def test_t3_replay_skips_when_edge_below_threshold():
    # calibrated prob == market price => zero edge => no trade
    ri = ReplayInput(trading_day="2026-04-01", calibrated_p_yes=0.30,
                     market_p_yes=0.30, outcome_yes=0)
    r = replay_one(ri, edge_threshold=0.07, kelly_fraction=0.075, bankroll=1000.0)
    assert r["traded"] is False
    assert r["pnl"] == 0.0 and r["wagered"] == 0.0


def test_t3_replay_buys_no_and_wins_when_outcome_no():
    # calibrated P(YES)=0.10, market YES price 0.40 => NO edge ~0.30 => trade NO
    # outcome_yes=0 => NO wins => positive pnl, fee-deducted
    # Derivation: NO price=0.60, prob_win=0.90, edge_frac=(0.90-0.60)/(1-0.60)=0.75
    #   dollars=0.75*0.075*1000=56.25, contracts=int(56.25/0.60)=93
    #   fee=ceil(0.07*0.60*0.40*100)/100=0.02
    #   pnl=93*(1-0.60)-93*0.02=37.20-1.86=35.34, wagered=93*0.60=55.80
    ri = ReplayInput(trading_day="2026-04-01", calibrated_p_yes=0.10,
                     market_p_yes=0.40, outcome_yes=0)
    r = replay_one(ri, edge_threshold=0.07, kelly_fraction=0.075, bankroll=1000.0)
    assert r["traded"] is True
    assert r["side"] == "NO"
    assert r["contracts"] == 93
    assert math.isclose(r["pnl"], 35.34)
    assert math.isclose(r["wagered"], 55.80)


def test_t3_replay_yes_side_wins_pins_pnl():
    # calibrated P(YES)=0.70, market YES price 0.40 => YES edge => trade YES
    # outcome_yes=1 => YES wins => positive pnl, fee-deducted
    # Derivation: YES price=0.40, prob_win=0.70, edge_frac=(0.70-0.40)/(1-0.40)=0.5
    #   dollars=0.5*0.075*1000=37.5, contracts=int(37.5/0.40)=93
    #   fee=ceil(0.07*0.40*0.60*100)/100=0.02
    #   pnl=93*(1-0.40)-93*0.02=55.80-1.86=53.94, wagered=93*0.40=37.20
    ri = ReplayInput(trading_day="2026-04-01", calibrated_p_yes=0.70,
                     market_p_yes=0.40, outcome_yes=1)
    r = replay_one(ri, edge_threshold=0.07, kelly_fraction=0.075, bankroll=1000.0)
    assert r["traded"] is True
    assert r["side"] == "YES"
    assert r["contracts"] == 93
    assert math.isclose(r["pnl"], 53.94)
    assert math.isclose(r["wagered"], 37.20)


def test_t3_replay_buys_no_and_loses_when_outcome_yes():
    ri = ReplayInput(trading_day="2026-04-01", calibrated_p_yes=0.10,
                     market_p_yes=0.40, outcome_yes=1)
    r = replay_one(ri, edge_threshold=0.07, kelly_fraction=0.075, bankroll=1000.0)
    assert r["traded"] is True and r["side"] == "NO"
    assert r["pnl"] < 0.0                        # NO lost


def test_t4_splits_between_no_from_greater_less_favorites():
    # All weather_edge; favorite_bias rows are excluded from T4 scope.
    trades = [
        _mk_trade("2026-04-01", 0.1, pnl=+5.0, st="between"),
        _mk_trade("2026-04-01", 0.1, pnl=-2.0, st="greater"),
        _mk_trade("2026-04-02", 0.1, pnl=+1.0, st="less"),
        # favorite_bias greater/less row must NOT appear in either bucket.
        _mk_trade("2026-04-02", 0.1, pnl=+99.0, st="greater", strat="favorite_bias"),
    ]
    res = edge_attribution(trades)
    assert res["between"]["n"] == 1
    assert math.isclose(res["between"]["pnl"], 5.0)
    assert res["greater_less"]["n"] == 2
    assert math.isclose(res["greater_less"]["pnl"], -1.0)


def test_trades_from_rows_prefers_effective_edge():
    # Row with metadata.effective_edge should use it over top-level edge.
    row_with_meta = _row(edge=0.192, metadata={"effective_edge": 0.094})
    row_no_meta = _row(edge=0.192)
    trades = trades_from_rows([row_with_meta, row_no_meta])
    assert len(trades) == 2
    assert math.isclose(trades[0].claimed_edge, 0.094)   # prefers metadata.effective_edge
    assert math.isclose(trades[1].claimed_edge, 0.192)   # falls back to top-level edge
