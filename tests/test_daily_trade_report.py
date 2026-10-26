"""Tests for daily_trade_report pure functions."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from scripts.daily_trade_report import (
    _RISK_ORDER,
    INTRADAY_BOOTSTRAP_B,
    INTRADAY_BOOTSTRAP_SEED,
    INTRADAY_STRATEGY_NAME,
    INTRADAY_TRIGGER_CI_STRADDLE_MIN_N,
    INTRADAY_TRIGGER_CONSEC_LOSS_MAX,
    INTRADAY_TRIGGER_PNL_FLOOR,
    INTRADAY_TRIGGER_WIN_RATE_FLOOR,
    INTRADAY_TRIGGER_WIN_RATE_MIN_N,
    LAX_PERMANENT_DISABLE_RUN_THRESHOLD,
    PHASE0_DEFAULT_LOOKBACK_DAYS,
    DailyPnL,
    PendingRisk,
    SettledTrade,
    _bootstrap_ci_lo,
    _p_loss_from_emos,
    _read_intraday_scan_funnel_cap_total,
    aggregate_per_cell,
    apply_spread_inflation,
    assess_risk,
    build_intraday_json,
    build_pending_scorecard,
    build_settled_report,
    compute_bracket_fraction,
    compute_calibration_gap,
    compute_intraday_consec_loss_runs,
    filter_records,
    find_latest_snapshot_run,
    format_report,
    load_forecast_data,
    load_records,
    summarize_intraday_strategy,
    summarize_phase0_depth,
)

# ---------------------------------------------------------------------------
# compute_bracket_fraction
# ---------------------------------------------------------------------------


def test_bracket_fraction_inclusive_both_bounds():
    temps = [69.9, 70.0, 70.5, 71.0, 71.1]
    assert compute_bracket_fraction(temps, 70.0, 71.0) == pytest.approx(3 / 5)


def test_bracket_fraction_empty():
    assert compute_bracket_fraction([], 70.0, 71.0) == 0.0


def test_bracket_fraction_none_in_range():
    assert compute_bracket_fraction([68.0, 69.0, 72.0], 70.0, 71.0) == 0.0


def test_bracket_fraction_all_in_range():
    assert compute_bracket_fraction([70.0, 70.5, 71.0], 70.0, 71.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_spread_inflation
# ---------------------------------------------------------------------------


def test_spread_inflation_factor_one_is_identity():
    temps = [69.0, 70.0, 71.0]
    assert apply_spread_inflation(temps, 1.0) == pytest.approx(temps)


def test_spread_inflation_factor_two_doubles_deviation():
    temps = [69.0, 70.0, 71.0]
    assert apply_spread_inflation(temps, 2.0) == pytest.approx([68.0, 70.0, 72.0])


def test_spread_inflation_factor_three():
    assert apply_spread_inflation([69.0, 70.0, 71.0], 3.0) == pytest.approx([67.0, 70.0, 73.0])


def test_spread_inflation_preserves_mean():
    temps = [68.0, 70.0, 72.0]
    inflated = apply_spread_inflation(temps, 2.5)
    original_mean = sum(temps) / len(temps)
    inflated_mean = sum(inflated) / len(inflated)
    assert inflated_mean == pytest.approx(original_mean)


def test_spread_inflation_empty():
    assert apply_spread_inflation([], 3.0) == []


# ---------------------------------------------------------------------------
# assess_risk
# ---------------------------------------------------------------------------


def test_assess_risk_safe_win_both_zero():
    assert assess_risk(0.0, 0.0, 70.0, 70.0) == "SAFE WIN"


def test_assess_risk_split_model_overrides_fractions():
    assert assess_risk(0.03, 0.10, 51.0, 47.0) == "SPLIT MODEL"


def test_assess_risk_danger_both_high():
    assert assess_risk(0.30, 0.20, 70.5, 70.1) == "DANGER"


def test_assess_risk_high_ecmwf():
    # ECMWF alone ≥ 30% → HIGH; means only 2°F apart so SPLIT MODEL doesn't fire
    assert assess_risk(0.05, 0.36, 51.0, 49.0) == "HIGH"


def test_assess_risk_medium_high():
    assert assess_risk(0.16, 0.16, 70.5, 70.0) == "Medium-High"


def test_assess_risk_medium():
    assert assess_risk(0.08, 0.12, 70.5, 70.0) == "Medium"


def test_assess_risk_low():
    assert assess_risk(0.02, 0.06, 73.0, 73.5) == "Low"


# ---------------------------------------------------------------------------
# find_latest_snapshot_run
# ---------------------------------------------------------------------------


def test_find_latest_snapshot_run_returns_highest_run(tmp_path):
    date_dir = tmp_path / "2026-05-03"
    for run in ("0500", "0900", "1400", "2000"):
        (date_dir / run).mkdir(parents=True)
    result = find_latest_snapshot_run(tmp_path, date(2026, 5, 3))
    assert result is not None
    assert result.name == "2000"


def test_find_latest_snapshot_run_missing_date(tmp_path):
    assert find_latest_snapshot_run(tmp_path, date(2026, 5, 3)) is None


def test_find_latest_snapshot_run_ignores_non_numeric_dirs(tmp_path):
    date_dir = tmp_path / "2026-05-03"
    (date_dir / "1400").mkdir(parents=True)
    (date_dir / "README").mkdir(parents=True)
    result = find_latest_snapshot_run(tmp_path, date(2026, 5, 3))
    assert result is not None
    assert result.name == "1400"


# ---------------------------------------------------------------------------
# load_forecast_data
# ---------------------------------------------------------------------------


def _write_fc_file(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "weather_forecasts.json"
    p.write_text(json.dumps(entries))
    return tmp_path


def test_load_forecast_data_separates_gfs_ecmwf(tmp_path):
    entries = [
        {
            "city": "NYC", "forecast_date": "2026-05-04",
            "daily_variable": "temperature_2m_max",
            "temperatures": [70.0, 71.0, 72.0], "mean": 71.0,
            "model": "gfs_seamless",
        },
        {
            "city": "NYC", "forecast_date": "2026-05-04",
            "daily_variable": "temperature_2m_max",
            "temperatures": [69.0, 70.0], "mean": 69.5,
            "model": "ecmwf_ifs025",
        },
    ]
    run_dir = _write_fc_file(tmp_path, entries)
    result = load_forecast_data(run_dir, "2026-05-04", {"NYC"})
    assert ("NYC", "high") in result
    fc = result[("NYC", "high")]
    assert fc.gfs_temps == [70.0, 71.0, 72.0]
    assert fc.ecmwf_temps == [69.0, 70.0]
    assert fc.gfs_mean == pytest.approx(71.0)
    assert fc.ecmwf_mean == pytest.approx(69.5)


def test_load_forecast_data_maps_min_to_low(tmp_path):
    entries = [{
        "city": "CHI", "forecast_date": "2026-05-04",
        "daily_variable": "temperature_2m_min",
        "temperatures": [48.0, 50.0], "mean": 49.0,
        "model": "gfs_seamless",
    }]
    run_dir = _write_fc_file(tmp_path, entries)
    result = load_forecast_data(run_dir, "2026-05-04", {"CHI"})
    assert ("CHI", "low") in result


def test_load_forecast_data_skips_wrong_date(tmp_path):
    entries = [{
        "city": "NYC", "forecast_date": "2026-05-05",
        "daily_variable": "temperature_2m_max",
        "temperatures": [70.0], "mean": 70.0,
        "model": "gfs_seamless",
    }]
    run_dir = _write_fc_file(tmp_path, entries)
    result = load_forecast_data(run_dir, "2026-05-04", {"NYC"})
    assert len(result) == 0


def test_load_forecast_data_skips_excluded_cities(tmp_path):
    entries = [{
        "city": "LAX", "forecast_date": "2026-05-04",
        "daily_variable": "temperature_2m_max",
        "temperatures": [75.0], "mean": 75.0,
        "model": "gfs_seamless",
    }]
    run_dir = _write_fc_file(tmp_path, entries)
    result = load_forecast_data(run_dir, "2026-05-04", {"NYC"})
    assert len(result) == 0


def test_load_forecast_data_missing_file(tmp_path):
    result = load_forecast_data(tmp_path, "2026-05-04", {"NYC"})
    assert result == {}


# ---------------------------------------------------------------------------
# load_records / filter_records
# ---------------------------------------------------------------------------


def test_load_records_parses_list(tmp_path):
    data = [{"market_ticker": "X", "mode": "live", "timestamp": "2026-05-01T00:00:00+00:00"}]
    p = tmp_path / "settlements.json"
    p.write_text(json.dumps(data))
    assert load_records(p) == data


def test_filter_records_by_date():
    records = [
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "live"},
        {"timestamp": "2026-05-02T00:00:00+00:00", "mode": "live"},
    ]
    result = filter_records(records, trade_date="2026-05-01", mode="live")
    assert len(result) == 1
    assert result[0]["timestamp"].startswith("2026-05-01")


def test_filter_records_by_mode():
    records = [
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "live"},
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "shadow"},
    ]
    result = filter_records(records, trade_date=None, mode="live")
    assert len(result) == 1
    assert result[0]["mode"] == "live"


def test_filter_records_mode_all():
    records = [
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "live"},
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "shadow"},
    ]
    result = filter_records(records, trade_date=None, mode="all")
    assert len(result) == 2


def test_filter_records_no_date_filter():
    records = [
        {"timestamp": "2026-05-01T00:00:00+00:00", "mode": "live"},
        {"timestamp": "2026-05-02T00:00:00+00:00", "mode": "live"},
    ]
    result = filter_records(records, trade_date=None, mode="live")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# build_settled_report
# ---------------------------------------------------------------------------


def _make_record(
    date_str: str,
    side: str,
    market_result: str | None,
    pnl: float | None,
    city: str = "NYC",
    floor: float = 60.0,
    cap: float = 61.0,
    model_p: float = 0.088,
    fill: float = 0.70,
    contracts: int = 10,
) -> dict:
    dollars = fill * contracts
    return {
        "timestamp": f"{date_str}T14:00:00+00:00",
        "market_ticker": f"KXHIGH{city}-B{floor + 0.5}",
        "side": side,
        "mode": "live",
        "strategy": "weather_edge",
        "model_probability": model_p,
        "fill_price": fill,
        "exec_price": round(1.0 - fill, 2),
        "contracts": contracts,
        "dollars": dollars,
        "market_result": market_result,
        "pnl": pnl,
        "floor_strike": floor,
        "cap_strike": cap,
        "metadata": {
            "city": city,
            "market_type": "high",
            "floor_strike": floor,
            "cap_strike": cap,
            "forecast_date": "2026-05-02",
            "raw_ensemble_prob": 0.10,
        },
    }


def test_build_settled_report_win():
    records = [_make_record("2026-05-01", "NO", "no", 2.80)]
    days = build_settled_report(records)
    assert len(days) == 1
    assert len(days[0].wins) == 1
    assert len(days[0].losses) == 0
    assert days[0].wins[0].pnl == pytest.approx(2.80)


def test_build_settled_report_loss():
    records = [_make_record("2026-05-01", "NO", "yes", -9.67)]
    days = build_settled_report(records)
    assert len(days[0].losses) == 1
    assert days[0].losses[0].pnl == pytest.approx(-9.67)


def test_build_settled_report_pending_excluded():
    records = [_make_record("2026-05-01", "NO", None, None)]
    days = build_settled_report(records)
    assert len(days) == 0


def test_build_settled_report_grouped_by_date():
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0),
        _make_record("2026-05-01", "NO", "yes", -9.0),
        _make_record("2026-05-02", "NO", "no", 3.0),
    ]
    days = build_settled_report(records)
    assert len(days) == 2
    may1 = next(d for d in days if d.trade_date == "2026-05-01")
    assert len(may1.wins) == 1
    assert len(may1.losses) == 1


def test_build_settled_report_yes_side_win():
    records = [_make_record("2026-05-01", "YES", "yes", 5.0)]
    days = build_settled_report(records)
    assert len(days[0].wins) == 1


def test_build_settled_report_total_pnl():
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0),
        _make_record("2026-05-01", "NO", "yes", -9.0),
    ]
    days = build_settled_report(records)
    assert days[0].total_pnl == pytest.approx(-7.0)


def test_build_settled_report_total_wagered_uses_dollars():
    # dollars field = 7.0 per record; total_wagered should sum dollars (not fill_no * contracts)
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0, fill=0.70, contracts=10),
        _make_record("2026-05-01", "NO", "yes", -9.0, fill=0.70, contracts=10),
    ]
    days = build_settled_report(records)
    assert days[0].total_wagered == pytest.approx(14.0)


def test_build_settled_report_shadow_row_no_fill_price():
    # Shadow rows have fill_price=None; cost basis from dollars/contracts
    r = _make_record("2026-05-01", "NO", "no", 2.0, fill=0.70, contracts=10)
    r["fill_price"] = None  # simulate shadow row
    days = build_settled_report([r])
    assert days[0].wins[0].fill_no == pytest.approx(0.70)  # recovered from dollars/contracts


def test_build_settled_report_floor_cap_from_metadata_when_top_level_missing():
    # Simulate a record where top-level floor_strike/cap_strike are absent (pending-style)
    r = _make_record("2026-05-01", "NO", "no", 2.0)
    del r["floor_strike"]
    del r["cap_strike"]
    days = build_settled_report([r])
    assert days[0].wins[0].floor_strike == pytest.approx(60.0)
    assert days[0].wins[0].cap_strike == pytest.approx(61.0)


# ---------------------------------------------------------------------------
# aggregate_per_cell
# ---------------------------------------------------------------------------


def test_aggregate_per_cell_counts_wins_and_losses_per_cell():
    """settled_per_cell aggregates win + loss counts per (city, var).

    Used by post_flip_monitor.sh T3 for true per-cell loss-rate display.
    """
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0, city="NYC"),
        _make_record("2026-05-01", "NO", "no", 3.0, city="NYC"),
        _make_record("2026-05-01", "NO", "yes", -9.0, city="NYC"),
        _make_record("2026-05-02", "NO", "yes", -7.0, city="MIA"),
        _make_record("2026-05-02", "NO", "no", 4.0, city="LAX"),
    ]
    settled = build_settled_report(records)
    stats = aggregate_per_cell(settled)
    assert stats == {
        "NYC/high": {"n_wins": 2, "n_losses": 1},
        "MIA/high": {"n_wins": 0, "n_losses": 1},
        "LAX/high": {"n_wins": 1, "n_losses": 0},
    }


def test_aggregate_per_cell_empty():
    """No settled days yields an empty dict."""
    assert aggregate_per_cell([]) == {}


# ---------------------------------------------------------------------------
# compute_calibration_gap
# ---------------------------------------------------------------------------


def test_compute_calibration_gap_basic():
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0, model_p=0.09),
        _make_record("2026-05-01", "NO", "no", 2.0, model_p=0.09),
        _make_record("2026-05-01", "NO", "yes", -9.0, model_p=0.09),
    ]
    gap = compute_calibration_gap(records)
    assert gap["n_settled"] == 3
    assert gap["n_wins"] == 2
    assert gap["n_losses"] == 1
    assert gap["win_rate"] == pytest.approx(2 / 3)
    assert gap["actual_loss_rate"] == pytest.approx(1 / 3)
    # For NO trades, loss prob = model_p (p(YES))
    assert gap["avg_model_loss_p"] == pytest.approx(0.09)
    assert gap["calibration_ratio"] == pytest.approx((1 / 3) / 0.09)


def test_compute_calibration_gap_yes_side_loss_prob():
    # YES bet: model_p=0.60 means p(YES)=0.60 → model loss prob = 1-0.60 = 0.40
    records = [_make_record("2026-05-01", "YES", "no", -5.0, model_p=0.60)]
    gap = compute_calibration_gap(records)
    assert gap["n_losses"] == 1
    assert gap["avg_model_loss_p"] == pytest.approx(0.40)


def test_compute_calibration_gap_excludes_pending():
    records = [
        _make_record("2026-05-01", "NO", "no", 2.0),
        _make_record("2026-05-01", "NO", None, None),
    ]
    gap = compute_calibration_gap(records)
    assert gap["n_settled"] == 1


def test_compute_calibration_gap_empty():
    assert compute_calibration_gap([]) == {}


def test_compute_calibration_gap_excludes_unattributed_rows():
    # Reconciler-backfilled rows have model_probability=None and must NOT be
    # coerced to 0 when averaging — that biases avg_model_loss_p downward and
    # inflates the calibration ratio. They should be excluded entirely from
    # the calibration stat and surfaced via n_unattributed.
    records = [
        _make_record("2026-05-07", "NO", "no", 2.0, model_p=0.09),
        _make_record("2026-05-07", "NO", "yes", -9.0, model_p=0.09),
        _make_record("2026-05-07", "NO", "no", 7.0, model_p=None),  # reconciler row
        _make_record("2026-05-07", "NO", "no", 4.0, model_p=None),  # reconciler row
    ]
    gap = compute_calibration_gap(records)
    assert gap["n_settled"] == 2  # only model-attributed rows count
    assert gap["n_unattributed"] == 2
    assert gap["avg_model_loss_p"] == pytest.approx(0.09)  # not diluted toward 0
    assert gap["actual_loss_rate"] == pytest.approx(0.5)  # 1 of 2 attributed
    assert gap["calibration_ratio"] == pytest.approx(0.5 / 0.09)


def test_compute_calibration_gap_only_unattributed_rows():
    # Edge case: all settled rows are reconciler rows. We can compute the
    # actual outcome (W/L tally is meaningful) but not the calibration stat.
    # Return zero/None in numeric fields and surface n_unattributed.
    records = [
        _make_record("2026-05-07", "NO", "no", 4.0, model_p=None),
        _make_record("2026-05-07", "NO", "yes", -8.0, model_p=None),
    ]
    gap = compute_calibration_gap(records)
    assert gap["n_settled"] == 0
    assert gap["n_unattributed"] == 2
    assert gap["calibration_ratio"] is None
    assert gap["avg_model_loss_p"] is None


# ---------------------------------------------------------------------------
# build_pending_scorecard
# ---------------------------------------------------------------------------


def _make_pending(
    city: str,
    var: str,
    floor: float,
    cap: float,
    forecast_date: str = "2026-05-04",
    trade_date: str = "2026-05-03",
    model_p: float = 0.088,
    fill: float = 0.70,
    contracts: int = 10,
) -> dict:
    dollars = fill * contracts
    return {
        "timestamp": f"{trade_date}T14:00:00+00:00",
        "market_ticker": f"KXHIGH{city}-26MAY04-B{floor + 0.5}",
        "side": "NO",
        "mode": "live",
        "model_probability": model_p,
        "fill_price": fill,
        "exec_price": round(1.0 - fill, 2),
        "contracts": contracts,
        "dollars": dollars,
        "market_result": None,
        "pnl": None,
        "floor_strike": floor,
        "cap_strike": cap,
        "metadata": {
            "city": city,
            "market_type": var,
            "floor_strike": floor,
            "cap_strike": cap,
            "forecast_date": forecast_date,
        },
    }


def _make_snapshot_dir(
    tmp_path: Path,
    snapshot_date: str,
    forecast_date: str,
    city: str,
    var: str,
    gfs_temps: list[float],
    ecmwf_temps: list[float],
) -> Path:
    daily_var = "temperature_2m_max" if var == "high" else "temperature_2m_min"
    entries = [
        {
            "city": city, "forecast_date": forecast_date,
            "daily_variable": daily_var,
            "temperatures": gfs_temps,
            "mean": sum(gfs_temps) / len(gfs_temps),
            "model": "gfs_seamless",
        },
        {
            "city": city, "forecast_date": forecast_date,
            "daily_variable": daily_var,
            "temperatures": ecmwf_temps,
            "mean": sum(ecmwf_temps) / len(ecmwf_temps),
            "model": "ecmwf_ifs025",
        },
    ]
    run_dir = tmp_path / snapshot_date / "1400"
    run_dir.mkdir(parents=True)
    (run_dir / "weather_forecasts.json").write_text(json.dumps(entries))
    return tmp_path


def test_build_pending_scorecard_safe_win(tmp_path):
    gfs = [69.0, 70.0, 71.0] * 10
    ecmwf = [70.0, 71.0] * 25
    snaps = _make_snapshot_dir(tmp_path, "2026-05-03", "2026-05-04", "NYC", "high", gfs, ecmwf)
    pending = [_make_pending("NYC", "high", 78.0, 79.0)]
    result = build_pending_scorecard(pending, snaps)
    assert len(result) == 1
    assert result[0].risk_level == "SAFE WIN"
    assert result[0].gfs_raw_frac == pytest.approx(0.0)
    assert result[0].ecmwf_raw_frac == pytest.approx(0.0)


def test_build_pending_scorecard_applies_inflation(tmp_path):
    # DEN high inflation factor = 3.0
    gfs = [70.0, 71.0, 72.0] * 10
    ecmwf = [70.0, 71.0, 72.0] * 10 + [71.0] * 20
    snaps = _make_snapshot_dir(tmp_path, "2026-05-03", "2026-05-04", "DEN", "high", gfs, ecmwf)
    pending = [_make_pending("DEN", "high", 70.0, 71.0)]
    result = build_pending_scorecard(pending, snaps)
    assert len(result) == 1
    r = result[0]
    assert r.inflation_factor == pytest.approx(3.0)
    assert r.gfs_raw_frac > r.gfs_inflated_frac


def test_build_pending_scorecard_split_model(tmp_path):
    gfs = [50.5, 51.0, 51.5] * 10
    ecmwf = [46.5, 47.0, 47.5] * 10 + [47.0] * 20
    snaps = _make_snapshot_dir(tmp_path, "2026-05-03", "2026-05-04", "NYC", "low", gfs, ecmwf)
    pending = [_make_pending("NYC", "low", 47.0, 48.0)]
    result = build_pending_scorecard(pending, snaps)
    assert result[0].risk_level == "SPLIT MODEL"


def test_build_pending_scorecard_no_snapshot_returns_empty(tmp_path):
    pending = [_make_pending("NYC", "high", 72.0, 73.0)]
    result = build_pending_scorecard(pending, tmp_path)
    assert result == []


def test_build_pending_scorecard_report_date_takes_priority(tmp_path):
    # Snapshot exists for "today" (report_date) but NOT for trade_date — should find it
    gfs = [65.0] * 30
    ecmwf = [65.0] * 50
    # snapshot under "today" date, not trade date
    snaps = _make_snapshot_dir(tmp_path, "2026-05-04", "2026-05-05", "NYC", "high", gfs, ecmwf)
    pending = [_make_pending("NYC", "high", 78.0, 79.0,
                             forecast_date="2026-05-05", trade_date="2026-05-04")]
    # Without report_date, trade_date IS 2026-05-04, so it would find the snapshot anyway.
    # Test that report_date=2026-05-04 finds the snapshot (same day as trade here).
    result = build_pending_scorecard(pending, snaps, report_date=date(2026, 5, 4))
    assert len(result) == 1  # found snapshot via report_date


def test_build_pending_scorecard_sorted_danger_first(tmp_path):
    safe_gfs = [65.0] * 30
    safe_ecmwf = [65.0] * 50
    danger_gfs = [70.5] * 30
    danger_ecmwf = [70.5] * 50

    entries = []
    for temps, model in [(safe_gfs, "gfs_seamless"), (safe_ecmwf, "ecmwf_ifs025")]:
        entries.append({
            "city": "AUS", "forecast_date": "2026-05-04",
            "daily_variable": "temperature_2m_max",
            "temperatures": temps,
            "mean": sum(temps) / len(temps),
            "model": model,
        })
    for temps, model in [(danger_gfs, "gfs_seamless"), (danger_ecmwf, "ecmwf_ifs025")]:
        entries.append({
            "city": "DEN", "forecast_date": "2026-05-04",
            "daily_variable": "temperature_2m_max",
            "temperatures": temps,
            "mean": sum(temps) / len(temps),
            "model": model,
        })
    run_dir = tmp_path / "2026-05-03" / "1400"
    run_dir.mkdir(parents=True)
    (run_dir / "weather_forecasts.json").write_text(json.dumps(entries))

    pending = [
        _make_pending("AUS", "high", 84.0, 85.0),
        _make_pending("DEN", "high", 70.0, 71.0),
    ]
    result = build_pending_scorecard(pending, tmp_path)
    assert len(result) == 2
    assert result[0].risk_level == "DANGER"
    assert result[1].risk_level == "SAFE WIN"


def test_build_pending_scorecard_dollars_not_exec_price(tmp_path):
    # Simulate a shadow row: fill_price=None, exec_price=YES side price (wrong if used as cost)
    gfs = [65.0] * 30
    ecmwf = [65.0] * 50
    snaps = _make_snapshot_dir(tmp_path, "2026-05-03", "2026-05-04", "NYC", "high", gfs, ecmwf)
    r = _make_pending("NYC", "high", 78.0, 79.0, fill=0.70, contracts=10)
    r["fill_price"] = None  # shadow row
    # dollars = 7.0, contracts = 10 → fill_no should be 0.70, not exec_price (0.30)
    result = build_pending_scorecard([r], snaps)
    assert len(result) == 1
    assert result[0].fill_no == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def _make_settled_trade(
    city="NYC", var="high", floor=60.0, cap=61.0,
    pnl=2.5, is_win=True, model_p=0.09, fill=0.70, contracts=10,
) -> SettledTrade:
    return SettledTrade(
        ticker=f"KXHIGH{city}-B{floor + 0.5}",
        city=city, var=var,
        floor_strike=floor, cap_strike=cap,
        forecast_date="2026-05-02",
        side="NO",
        model_p=model_p,
        fill_no=fill,
        contracts=contracts,
        dollars=fill * contracts,
        pnl=pnl,
        is_win=is_win,
        raw_ensemble_prob=0.10,
    )


def _make_pending_risk(city="DEN", var="high", floor=70.0, cap=71.0, risk="DANGER") -> PendingRisk:
    return PendingRisk(
        ticker=f"KXHIGH{city}-B{floor + 0.5}",
        city=city, var=var,
        floor_strike=floor, cap_strike=cap,
        forecast_date="2026-05-04",
        model_p=0.088,
        fill_no=0.63,
        contracts=5,
        dollars=3.15,
        gfs_raw_frac=0.367,
        ecmwf_raw_frac=0.22,
        gfs_inflated_frac=0.067,
        ecmwf_inflated_frac=0.08,
        gfs_mean=70.8,
        ecmwf_mean=70.1,
        inflation_factor=3.0,
        risk_level=risk,
    )


def test_format_report_contains_pnl_header():
    days = [DailyPnL("2026-05-01", wins=[_make_settled_trade(pnl=2.5)], losses=[])]
    out = format_report(days, [], {}, "2026-05-01")
    assert "SETTLED P&L" in out
    assert "2026-05-01" in out
    assert "2.50" in out


def test_format_report_loss_line_shows_city_and_bracket():
    loss = _make_settled_trade(city="CHI", floor=55.0, cap=56.0, pnl=-9.67, is_win=False)
    days = [DailyPnL("2026-05-01", wins=[], losses=[loss])]
    out = format_report(days, [], {}, "2026-05-01")
    assert "LOSS" in out
    assert "CHI" in out
    assert "55" in out


def test_format_report_pending_scorecard_section():
    pending = [_make_pending_risk("DEN", "high", 70.0, 71.0, "DANGER")]
    out = format_report([], pending, {}, "2026-05-03")
    assert "PENDING" in out
    assert "DEN" in out
    assert "DANGER" in out
    assert "3.0x" in out


def test_format_report_calibration_section():
    cal = {
        "n_settled": 51, "n_wins": 36, "n_losses": 15,
        "win_rate": 0.706, "actual_loss_rate": 0.294,
        "avg_model_loss_p": 0.094, "calibration_ratio": 3.13,
    }
    out = format_report([], [], cal, "2026-05-03")
    assert "CALIBRATION" in out
    assert "70" in out
    assert "3.1" in out


def test_format_report_empty_sections_graceful():
    out = format_report([], [], {}, "2026-05-03")
    assert isinstance(out, str)
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Integration tests (skipped when data files absent)
# ---------------------------------------------------------------------------

SETTLEMENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "shadow_settlements.full.json"
SNAPSHOTS_PATH = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@pytest.mark.skipif(not SETTLEMENTS_PATH.exists(), reason="settlements file not present")
def test_integration_settled_report_runs():
    records = load_records(SETTLEMENTS_PATH)
    live = filter_records(records, trade_date=None, mode="live")
    assert len(live) > 0
    days = build_settled_report(live)
    assert len(days) > 0
    for day in days:
        assert isinstance(day.total_pnl, float)
        assert isinstance(day.total_wagered, float)


@pytest.mark.skipif(not SETTLEMENTS_PATH.exists(), reason="settlements file not present")
def test_integration_calibration_gap_runs():
    records = load_records(SETTLEMENTS_PATH)
    gap = compute_calibration_gap(records)
    if gap:
        assert 0 <= gap["win_rate"] <= 1
        assert 0 <= gap["avg_model_loss_p"] <= 1


@pytest.mark.skipif(
    not SETTLEMENTS_PATH.exists() or not SNAPSHOTS_PATH.exists(),
    reason="data files not present",
)
def test_integration_pending_scorecard_runs():
    records = load_records(SETTLEMENTS_PATH)
    live = filter_records(records, trade_date=None, mode="live")
    pending = build_pending_scorecard(live, SNAPSHOTS_PATH)
    assert isinstance(pending, list)
    for p in pending:
        assert p.risk_level in _RISK_ORDER
        assert 0.0 <= p.gfs_raw_frac <= 1.0
        assert 0.0 <= p.ecmwf_raw_frac <= 1.0


@pytest.mark.skipif(not SETTLEMENTS_PATH.exists(), reason="settlements file not present")
def test_integration_format_report_produces_string():
    records = load_records(SETTLEMENTS_PATH)
    live = filter_records(records, trade_date=None, mode="live")
    days = build_settled_report(live)
    pending = build_pending_scorecard(live, SNAPSHOTS_PATH)
    cal = compute_calibration_gap(records)
    out = format_report(days, pending, cal, None)
    assert isinstance(out, str)
    assert "SETTLED P&L" in out
    assert "CALIBRATION" in out


# ---------------------------------------------------------------------------
# C-DIAG-CALIB-RATIO: calibration_ratio honors --mode
# ---------------------------------------------------------------------------


class TestCalibrationGapModeFilter:
    """C-DIAG-CALIB-RATIO: calibration_ratio should be mode-only filtered.

    The mode filter must NOT also restrict by date — calibration_ratio
    needs rolling N to be statistically informative.
    """

    @pytest.fixture
    def mixed_mode_records(self) -> list[dict]:
        """Records spanning multiple dates AND mixing live/shadow modes."""
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        last_week = (date.today() - timedelta(days=7)).isoformat()
        return [
            {
                "trade_date": last_week, "mode": "live", "side": "NO", "strategy": "weather_edge",
                "model_probability": 0.10, "market_result": "no",
            },
            {
                "trade_date": last_week, "mode": "live", "side": "NO", "strategy": "weather_edge",
                "model_probability": 0.20, "market_result": "yes",
            },
            {
                "trade_date": yesterday, "mode": "shadow", "side": "NO", "strategy": "weather_edge",
                "model_probability": 0.05, "market_result": "yes",
            },
            {
                "trade_date": today, "mode": "live", "side": "NO", "strategy": "weather_edge",
                "model_probability": 0.15, "market_result": "no",
            },
        ]

    def test_compute_calibration_gap_mode_live_includes_all_dates(
        self, mixed_mode_records
    ):
        """Live mode filter must keep records from ALL dates (rolling N)."""
        live_records = [r for r in mixed_mode_records if r["mode"] == "live"]
        result = compute_calibration_gap(live_records)
        assert result["n_settled"] == 3

    def test_compute_calibration_gap_mode_all(self, mixed_mode_records):
        """`--mode all` includes every settled record."""
        result = compute_calibration_gap(mixed_mode_records)
        assert result["n_settled"] == 4

    def test_compute_calibration_gap_no_settled_records(self):
        """Empty input returns empty dict (existing behavior preserved)."""
        result = compute_calibration_gap([])
        assert result == {}

    def test_format_report_header_reflects_mode(self):
        """A1d: header text shows mode when not 'all', shows 'all modes' when 'all'."""
        calibration_live = {
            "mode": "live",
            "n_settled": 3,
            "n_wins": 2,
            "n_losses": 1,
            "win_rate": 0.667,
            "actual_loss_rate": 0.333,
            "avg_model_loss_p": 0.20,
            "calibration_ratio": 1.67,
        }
        out_live = format_report([], [], calibration_live, "2026-05-09")
        assert "=== CALIBRATION GAP (all settled dates, mode=live) ===" in out_live, (
            "Expected mode-specific header literal not found in:\n" + out_live
        )

        calibration_all = {**calibration_live, "mode": "all"}
        out_all = format_report([], [], calibration_all, "2026-05-09")
        assert "=== CALIBRATION GAP (all settled dates, all modes) ===" in out_all, (
            "Expected all-modes header literal not found in:\n" + out_all
        )

        calibration_shadow = {**calibration_live, "mode": "shadow"}
        out_shadow = format_report([], [], calibration_shadow, "2026-05-09")
        assert "=== CALIBRATION GAP (all settled dates, mode=shadow) ===" in out_shadow

    def test_format_report_no_calibration_uses_legacy_header(self):
        """Empty calibration dict still uses the '(no settled data)' fallback."""
        out = format_report([], [], {}, "2026-05-09")
        assert "=== CALIBRATION GAP ===" in out
        assert "(no settled data)" in out


# ---------------------------------------------------------------------------
# Phase 0 depth-probing summary (M0 — intraday EMOS HIGH spec §6.4 / §12.1)
# ---------------------------------------------------------------------------


def _phase0_row(
    *,
    cell: str = "MIA/high",
    ticker: str = "KXHIGHMIA-26MAY15H1700-B85",
    series: str = "KXHIGHMIA",
    yes_bid_size: int = 30,
    yes_ask_size: int = 25,
    empty_book: bool = False,
    ts: datetime | None = None,
    sample_kind: str = "phase0_intraday_depth",
) -> dict:
    """Build a minimal Phase 0 row matching the monitor's emitted schema."""
    t = ts or datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    return {
        "sample_kind": sample_kind,
        "ts": t.isoformat(),
        "ts_unix": t.timestamp(),
        "ticker": ticker,
        "series_ticker": series,
        "cell": cell,
        "city": cell.split("/")[0],
        "yes_bid": 0.42,
        "yes_bid_size": yes_bid_size,
        "yes_bid_l2_size": None,
        "yes_ask": 0.45,
        "yes_ask_size": yes_ask_size,
        "no_bid": 0.55,
        "no_bid_size": yes_ask_size,
        "no_bid_l2_size": None,
        "mid": 0.435,
        "spread": 0.03,
        "empty_book": empty_book,
        "yes_levels_count": None,
        "no_levels_count": None,
    }


def _write_phase0_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _all_cells_empty(cells: dict) -> bool:
    """True iff every cell entry is the n=0 placeholder."""
    return all(c["n_samples"] == 0 for c in cells.values())


def test_summarize_phase0_depth_missing_file(tmp_path: Path):
    """A non-existent sink file must yield an empty-but-well-formed summary.

    The summary pre-populates the eligible cells (MIA/high, LAX/high) with
    n=0 placeholders so the operator sees the section is wired up — distinct
    from "this strategy doesn't exist".
    """
    result = summarize_phase0_depth(tmp_path / "nope.jsonl")
    assert result["n_samples"] == 0
    assert result["lookback_days"] == PHASE0_DEFAULT_LOOKBACK_DAYS
    assert set(result["cells"].keys()) == {"MIA/high", "LAX/high"}
    assert _all_cells_empty(result["cells"])
    assert result["cells"]["MIA/high"]["yes_bid_size_p50"] is None


def test_summarize_phase0_depth_empty_sink(tmp_path: Path):
    """An empty file behaves identically to a missing file."""
    sink = tmp_path / "phase0.jsonl"
    sink.touch()
    result = summarize_phase0_depth(sink)
    assert result["n_samples"] == 0
    assert _all_cells_empty(result["cells"])


def test_summarize_phase0_depth_populated_aggregates_per_cell(tmp_path: Path):
    sink = tmp_path / "phase0.jsonl"
    now = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    rows = [
        _phase0_row(cell="MIA/high", yes_bid_size=10, yes_ask_size=10, ts=now),
        _phase0_row(cell="MIA/high", yes_bid_size=20, yes_ask_size=20, ts=now),
        _phase0_row(cell="MIA/high", yes_bid_size=30, yes_ask_size=30, ts=now),
        _phase0_row(cell="LAX/high", yes_bid_size=5, yes_ask_size=5, ts=now),
        _phase0_row(cell="LAX/high", yes_bid_size=15, yes_ask_size=15, ts=now),
    ]
    _write_phase0_jsonl(sink, rows)

    result = summarize_phase0_depth(sink, now_utc=now)
    assert result["n_samples"] == 5

    mia = result["cells"]["MIA/high"]
    assert mia["n_samples"] == 3
    assert mia["empty_book_frac"] == pytest.approx(0.0)
    # P50 of [10, 20, 30] interpolates to 20.0.
    assert mia["yes_bid_size_p50"] == pytest.approx(20.0)
    assert mia["yes_ask_size_p50"] == pytest.approx(20.0)
    # P10 / P90 interpolation: with n=3 the index keys are at 0.2 and 1.8.
    assert mia["yes_bid_size_p10"] == pytest.approx(12.0)
    assert mia["yes_bid_size_p90"] == pytest.approx(28.0)
    assert mia["nbbo_min_size_p50"] == pytest.approx(20.0)

    lax = result["cells"]["LAX/high"]
    assert lax["n_samples"] == 2
    assert lax["yes_bid_size_p50"] == pytest.approx(10.0)


def test_summarize_phase0_depth_drops_rows_outside_lookback(tmp_path: Path):
    sink = tmp_path / "phase0.jsonl"
    now = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    stale = datetime(2026, 4, 1, 18, 0, 0, tzinfo=UTC)  # >14 days old
    rows = [
        _phase0_row(cell="MIA/high", yes_bid_size=200, yes_ask_size=200, ts=stale),
        _phase0_row(cell="MIA/high", yes_bid_size=20, yes_ask_size=20, ts=now),
    ]
    _write_phase0_jsonl(sink, rows)

    result = summarize_phase0_depth(sink, lookback_days=14, now_utc=now)
    assert result["n_samples"] == 1
    assert result["cells"]["MIA/high"]["yes_bid_size_p50"] == pytest.approx(20.0)


def test_summarize_phase0_depth_empty_book_frac(tmp_path: Path):
    sink = tmp_path / "phase0.jsonl"
    now = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    rows = [
        _phase0_row(cell="MIA/high", empty_book=True, ts=now),
        _phase0_row(cell="MIA/high", empty_book=True, ts=now),
        _phase0_row(cell="MIA/high", yes_bid_size=10, yes_ask_size=10, ts=now),
        _phase0_row(cell="MIA/high", yes_bid_size=12, yes_ask_size=12, ts=now),
    ]
    _write_phase0_jsonl(sink, rows)

    result = summarize_phase0_depth(sink, now_utc=now)
    mia = result["cells"]["MIA/high"]
    assert mia["n_samples"] == 4
    assert mia["empty_book_frac"] == pytest.approx(0.5)
    # Bid/ask percentiles ignore empty-book rows (no size to record).
    assert mia["yes_bid_size_p50"] == pytest.approx(11.0)


def test_summarize_phase0_depth_ignores_unknown_sample_kind(tmp_path: Path):
    """Rows tagged with a different ``sample_kind`` must be skipped."""
    sink = tmp_path / "phase0.jsonl"
    now = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    rows = [
        _phase0_row(cell="MIA/high", sample_kind="other_monitor"),
        _phase0_row(cell="MIA/high", yes_bid_size=20, yes_ask_size=20, ts=now),
    ]
    _write_phase0_jsonl(sink, rows)
    result = summarize_phase0_depth(sink, now_utc=now)
    assert result["n_samples"] == 1


def test_format_report_phase0_section_empty_renders_placeholder(tmp_path: Path):
    """Empty sink → report still has a section with a (no samples...) line."""
    empty_summary = summarize_phase0_depth(tmp_path / "nope.jsonl")
    out = format_report([], [], {}, "2026-05-15", phase0_summary=empty_summary)
    assert "=== PHASE 0 DEPTH PROBING ===" in out
    assert "(no Phase 0 samples" in out


def test_format_report_phase0_section_populated_renders_table(tmp_path: Path):
    sink = tmp_path / "phase0.jsonl"
    now = datetime(2026, 5, 15, 18, 0, 0, tzinfo=UTC)
    rows = [
        _phase0_row(cell="MIA/high", yes_bid_size=10, yes_ask_size=10, ts=now),
        _phase0_row(cell="MIA/high", yes_bid_size=20, yes_ask_size=20, ts=now),
        _phase0_row(cell="LAX/high", yes_bid_size=5, yes_ask_size=5, ts=now),
    ]
    _write_phase0_jsonl(sink, rows)
    summary = summarize_phase0_depth(sink, now_utc=now)
    out = format_report([], [], {}, "2026-05-15", phase0_summary=summary)

    assert "=== PHASE 0 DEPTH PROBING ===" in out
    assert "MIA/high" in out
    assert "LAX/high" in out
    # Table column header present.
    assert "bidP50" in out
    assert "askP50" in out


def test_format_report_no_phase0_arg_omits_section():
    """Backward compat: callers that don't pass phase0_summary skip the section."""
    out = format_report([], [], {}, "2026-05-15")
    assert "PHASE 0 DEPTH PROBING" not in out


# ---------------------------------------------------------------------------
# Phase 0 by-hour bucketing (spec §6.4(b) / §11.4)
# ---------------------------------------------------------------------------


def _et_to_utc(year: int, month: int, day: int, et_hour: int) -> datetime:
    """Build a UTC datetime that lands on ``et_hour`` (ET wall-clock).

    Used by the by-hour tests so the ts_unix → ET hour round-trip in
    ``summarize_phase0_depth`` is exercised end-to-end.
    """
    from zoneinfo import ZoneInfo

    et_dt = datetime(year, month, day, et_hour, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    return et_dt.astimezone(UTC)


def test_summarize_phase0_depth_buckets_by_et_hour(tmp_path: Path):
    """Spec §6.4(b): per-cell by_hour aggregation across multiple hours."""
    sink = tmp_path / "phase0.jsonl"
    # Spread rows across three in-window ET hours (10:00, 14:00, 18:00).
    rows = [
        # 10:00 ET — two rows
        _phase0_row(
            cell="MIA/high", yes_bid_size=10, yes_ask_size=12,
            ts=_et_to_utc(2026, 5, 15, 10),
        ),
        _phase0_row(
            cell="MIA/high", yes_bid_size=20, yes_ask_size=22,
            ts=_et_to_utc(2026, 5, 15, 10),
        ),
        # 14:00 ET — one row
        _phase0_row(
            cell="MIA/high", yes_bid_size=40, yes_ask_size=44,
            ts=_et_to_utc(2026, 5, 15, 14),
        ),
        # 18:00 ET — one row
        _phase0_row(
            cell="MIA/high", yes_bid_size=60, yes_ask_size=66,
            ts=_et_to_utc(2026, 5, 15, 18),
        ),
    ]
    _write_phase0_jsonl(sink, rows)

    # Use an ET-aware "now" cutoff so all rows fall inside the lookback.
    now = _et_to_utc(2026, 5, 15, 20)
    result = summarize_phase0_depth(sink, now_utc=now)
    mia = result["cells"]["MIA/high"]
    by_hour = mia["by_hour"]

    # Exactly the three hours we wrote should be present, with the
    # zero-sample-omission convention applied to every other hour.
    assert set(by_hour.keys()) == {10, 14, 18}

    # 10:00 bucket has two samples → P50 = mean of [10, 20] = 15.0.
    assert by_hour[10]["n"] == 2
    assert by_hour[10]["yes_bid_size_p50"] == pytest.approx(15.0)
    assert by_hour[10]["yes_ask_size_p50"] == pytest.approx(17.0)
    assert by_hour[10]["empty_book_frac"] == pytest.approx(0.0)

    # Single-sample buckets degenerate to the sole value.
    assert by_hour[14]["n"] == 1
    assert by_hour[14]["yes_bid_size_p50"] == pytest.approx(40.0)
    assert by_hour[18]["n"] == 1
    assert by_hour[18]["yes_ask_size_p50"] == pytest.approx(66.0)

    # Top-level lookback-wide aggregates remain populated.
    assert mia["n_samples"] == 4
    assert mia["yes_bid_size_p50"] is not None


def test_summarize_phase0_depth_excludes_hours_outside_window(tmp_path: Path):
    """Hours outside [08, 20] ET must NOT appear in by_hour, but still count in n_samples."""
    sink = tmp_path / "phase0.jsonl"
    rows = [
        # 03:00 ET — outside window
        _phase0_row(
            cell="MIA/high", yes_bid_size=999, yes_ask_size=999,
            ts=_et_to_utc(2026, 5, 15, 3),
        ),
        # 23:00 ET — outside window
        _phase0_row(
            cell="MIA/high", yes_bid_size=999, yes_ask_size=999,
            ts=_et_to_utc(2026, 5, 15, 23),
        ),
        # 08:00 ET — inside (boundary)
        _phase0_row(
            cell="MIA/high", yes_bid_size=10, yes_ask_size=10,
            ts=_et_to_utc(2026, 5, 15, 8),
        ),
        # 20:00 ET — inside (boundary)
        _phase0_row(
            cell="MIA/high", yes_bid_size=20, yes_ask_size=20,
            ts=_et_to_utc(2026, 5, 15, 20),
        ),
    ]
    _write_phase0_jsonl(sink, rows)

    # Cutoff well inside lookback for every row.
    now = _et_to_utc(2026, 5, 16, 12)
    result = summarize_phase0_depth(sink, now_utc=now)
    mia = result["cells"]["MIA/high"]

    # All four rows count toward the lookback-wide total.
    assert mia["n_samples"] == 4

    # Only the in-window hours (08, 20) appear as by_hour keys.
    assert set(mia["by_hour"].keys()) == {8, 20}
    assert mia["by_hour"][8]["yes_bid_size_p50"] == pytest.approx(10.0)
    assert mia["by_hour"][20]["yes_bid_size_p50"] == pytest.approx(20.0)

    # The 999-valued out-of-window rows still flow into the top-level
    # P90 (single-pass aggregation by design).
    assert mia["yes_bid_size_p90"] is not None
    assert mia["yes_bid_size_p90"] >= 20.0


def test_format_report_phase0_renders_p10_and_by_hour(tmp_path: Path):
    """Fix #2: P10 columns appear in the headline table; by_hour line renders."""
    sink = tmp_path / "phase0.jsonl"
    rows = [
        _phase0_row(
            cell="MIA/high", yes_bid_size=10, yes_ask_size=12,
            ts=_et_to_utc(2026, 5, 15, 10),
        ),
        _phase0_row(
            cell="MIA/high", yes_bid_size=30, yes_ask_size=33,
            ts=_et_to_utc(2026, 5, 15, 14),
        ),
        _phase0_row(
            cell="LAX/high", yes_bid_size=5, yes_ask_size=6,
            ts=_et_to_utc(2026, 5, 15, 12),
        ),
    ]
    _write_phase0_jsonl(sink, rows)
    now = _et_to_utc(2026, 5, 15, 20)
    summary = summarize_phase0_depth(sink, now_utc=now)
    out = format_report([], [], {}, "2026-05-15", phase0_summary=summary)

    # Headline table now exposes P10 columns alongside the existing
    # P25/P50/P90 columns (P25 retained — spec only names P10/P50/P90
    # by hand but column budget permits both).
    assert "bidP10" in out
    assert "askP10" in out
    assert "bidP25" in out  # P25 retained in the text rendering
    # By-hour sub-table header is present.
    assert "By ET hour" in out
    # Per-cell by-hour rows include the hour labels of in-window samples.
    assert "10h:" in out
    assert "12h:" in out
    assert "14h:" in out


# ---------------------------------------------------------------------------
# Fix #3 — --pull-vps must also scp the Phase 0 depth sink.
# ---------------------------------------------------------------------------


def test_pull_vps_pulls_phase0_sink_alongside_trades(monkeypatch, tmp_path):
    """``daily_trade_report.py --pull-vps`` must scp BOTH trades.jsonl AND
    the Phase 0 depth sink. Without the second pull, the Phase 0 summary
    on the operator's laptop renders empty throughout the 2-week measurement
    window — see Fix #3 / spec §6.4 audit note.
    """
    import subprocess as _real_subprocess
    import sys

    import scripts.daily_trade_report as dtr
    from python.data.phase0_depth_monitor import PHASE0_DEPTH_SINK_FILENAME
    from scripts._vps import VPS_HOST, VPS_STATE_DIR, VPS_TRADES_PATH

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    # Stub all the heavy-lifting bits the main() body would otherwise hit.
    monkeypatch.setattr(dtr.subprocess, "run", _fake_run)
    monkeypatch.setattr(dtr, "load_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "filter_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_settled_report", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_pending_scorecard", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "compute_calibration_gap", lambda *_a, **_k: {})
    monkeypatch.setattr(
        dtr,
        "summarize_phase0_depth",
        lambda *_a, **_k: {"lookback_days": 14, "n_samples": 0, "cells": {}},
    )
    monkeypatch.setattr(dtr, "format_report", lambda *_a, **_k: "")

    settlements_stub = tmp_path / "settlements.json"
    settlements_stub.write_text("[]")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daily_trade_report.py",
            "--pull-vps",
            "--settlements",
            str(settlements_stub),
        ],
    )

    dtr.main()

    # Capture the scp commands the script issued. We don't care about ssh
    # or other subprocess calls — only the scp ones in the --pull-vps block.
    scp_targets = [c[1] for c in calls if c and c[0] == "scp"]
    assert f"{VPS_HOST}:{VPS_TRADES_PATH}" in scp_targets, (
        f"trades.jsonl scp missing from --pull-vps; got {scp_targets!r}"
    )
    assert (
        f"{VPS_HOST}:{VPS_STATE_DIR}/{PHASE0_DEPTH_SINK_FILENAME}"
        in scp_targets
    ), (
        f"phase0 sink scp missing from --pull-vps; got {scp_targets!r}"
    )

    # Hush ruff: parameter intentionally unused; pinning the import keeps
    # the future maintainer aware that the real subprocess module is the
    # one being monkeypatched (not a re-exported alias).
    _ = _real_subprocess


# ---------------------------------------------------------------------------
# --refresh must forward the trades path to fetch_shadow_settlements.py.
# Cron on the VPS runs --refresh without --pull-vps; live trades live at
# state/trades.jsonl, not data/vps_trades.jsonl. Without forwarding, the
# subprocess defaults to data/vps_trades.jsonl and exits 1.
# ---------------------------------------------------------------------------


def _stub_refresh_main(monkeypatch, tmp_path):
    """Common monkeypatching for --refresh argv-capture tests."""
    import sys

    import scripts.daily_trade_report as dtr

    calls: list[list[str]] = []

    def _fake_run(cmd, *_a, **_k):
        calls.append(list(cmd))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(dtr.subprocess, "run", _fake_run)
    monkeypatch.setattr(dtr, "load_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "filter_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_settled_report", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_pending_scorecard", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "compute_calibration_gap", lambda *_a, **_k: {})
    monkeypatch.setattr(
        dtr,
        "summarize_phase0_depth",
        lambda *_a, **_k: {"lookback_days": 14, "n_samples": 0, "cells": {}},
    )
    monkeypatch.setattr(dtr, "format_report", lambda *_a, **_k: "")

    settlements_stub = tmp_path / "settlements.json"
    settlements_stub.write_text("[]")
    return calls, settlements_stub, dtr, sys


def test_refresh_without_pull_vps_uses_trade_log_path(monkeypatch, tmp_path):
    """VPS-direct cron path: ``--refresh`` alone must point
    fetch_shadow_settlements at state/trades.jsonl (the live trade log),
    not the operator-local data/vps_trades.jsonl artifact. Real-world
    failure 2026-05-16: cron exited 1 with 'vps_trades.jsonl not found'.
    """
    calls, settlements_stub, _dtr, sys_mod = _stub_refresh_main(monkeypatch, tmp_path)

    fake_log = tmp_path / "fake_state" / "trades.jsonl"
    fake_log.parent.mkdir()
    fake_log.write_text("")

    monkeypatch.setattr(
        sys_mod,
        "argv",
        [
            "daily_trade_report.py",
            "--refresh",
            "--settlements",
            str(settlements_stub),
            "--trade-log",
            str(fake_log),
        ],
    )

    import scripts.daily_trade_report as dtr

    dtr.main()

    refresh_calls = [
        c for c in calls if c and c[0] != "scp" and "fetch_shadow_settlements.py" in " ".join(c)
    ]
    assert len(refresh_calls) == 1, f"expected 1 refresh subprocess, got {refresh_calls!r}"
    cmd = refresh_calls[0]
    assert "--trades" in cmd, f"--trades flag missing: {cmd!r}"
    assert cmd[cmd.index("--trades") + 1] == str(fake_log), (
        f"--trades arg should be the --trade-log path, got {cmd!r}"
    )


def test_refresh_with_pull_vps_uses_local_vps_artifact(monkeypatch, tmp_path):
    """Operator-local path: ``--pull-vps --refresh`` must point
    fetch_shadow_settlements at data/vps_trades.jsonl — the SCP landing
    spot from the --pull-vps block — not at state/trades.jsonl.
    """
    calls, settlements_stub, _dtr, sys_mod = _stub_refresh_main(monkeypatch, tmp_path)

    monkeypatch.setattr(
        sys_mod,
        "argv",
        [
            "daily_trade_report.py",
            "--pull-vps",
            "--refresh",
            "--settlements",
            str(settlements_stub),
        ],
    )

    import scripts.daily_trade_report as dtr

    dtr.main()

    refresh_calls = [
        c for c in calls if c and c[0] != "scp" and "fetch_shadow_settlements.py" in " ".join(c)
    ]
    assert len(refresh_calls) == 1, f"expected 1 refresh subprocess, got {refresh_calls!r}"
    cmd = refresh_calls[0]
    assert "--trades" in cmd, f"--trades flag missing: {cmd!r}"
    forwarded = cmd[cmd.index("--trades") + 1]
    assert forwarded.endswith("data/vps_trades.jsonl"), (
        f"--trades arg should be data/vps_trades.jsonl under --pull-vps, got {forwarded!r}"
    )


def test_pull_vps_redirects_intraday_trade_log_to_local_vps_artifact(monkeypatch, tmp_path):
    """``--pull-vps`` must redirect the Intraday EMOS HIGH section's
    trade-log read to the SCP'd ``data/vps_trades.jsonl``, not the
    ``--trade-log`` default (``state/trades.jsonl``). On the operator
    laptop the default path doesn't exist; without the redirect the
    intraday section silently renders "no intraday trades" even after
    a successful --pull-vps. Mirrors the existing --refresh redirect.
    """
    import sys

    import scripts.daily_trade_report as dtr

    captured: dict[str, object] = {}

    def _capture_intraday(trade_log_path, *_a, **_k):
        captured["trade_log"] = trade_log_path
        return {"cells": {}, "lookback_days": 7}

    def _fake_run(cmd, *_a, **_k):
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(dtr.subprocess, "run", _fake_run)
    monkeypatch.setattr(dtr, "load_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "filter_records", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_settled_report", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "build_pending_scorecard", lambda *_a, **_k: [])
    monkeypatch.setattr(dtr, "compute_calibration_gap", lambda *_a, **_k: {})
    monkeypatch.setattr(
        dtr,
        "summarize_phase0_depth",
        lambda *_a, **_k: {"lookback_days": 14, "n_samples": 0, "cells": {}},
    )
    monkeypatch.setattr(dtr, "summarize_intraday_strategy", _capture_intraday)
    monkeypatch.setattr(dtr, "format_report", lambda *_a, **_k: "")

    settlements_stub = tmp_path / "settlements.json"
    settlements_stub.write_text("[]")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daily_trade_report.py",
            "--pull-vps",
            "--settlements",
            str(settlements_stub),
        ],
    )

    dtr.main()

    trade_log = captured.get("trade_log")
    assert trade_log is not None, "summarize_intraday_strategy was not called"
    assert str(trade_log).endswith("data/vps_trades.jsonl"), (
        f"intraday trade-log should redirect to data/vps_trades.jsonl under "
        f"--pull-vps, got {trade_log!r}"
    )


# ---------------------------------------------------------------------------
# Intraday EMOS HIGH strategy section (M6 — spec §9.2(c), §11.4, §8.4(b))
#
# The intraday section is the daily-report consumer of v1.3 path-(b):
# consec-loss runs are computed out-of-band by scanning state/trades.jsonl
# filtered by strategy_name == "intraday_emos_high". The §8.4(b) LAX
# permanent-disable trigger reads ``max_run_len_7d`` from the helper.
# ---------------------------------------------------------------------------


def _intraday_fill_row(
    *,
    order_id: str,
    market_ticker: str,
    timestamp: str,
    side: str = "NO",
    contracts: int = 5,
    dollars: float = 5.0,
    fill_price: float = 0.20,
    strategy: str = "intraday_emos_high",
    mode: str = "shadow",
    metadata: dict | None = None,
) -> dict:
    """Build a fill row matching engine._build_log_entry's schema."""
    md = {
        "strategy_name": strategy,
        "cell": _ticker_to_cell(market_ticker),
        "city": _ticker_to_cell(market_ticker).split("/")[0]
        if _ticker_to_cell(market_ticker) else "",
        "market_type": "high",
        "emos_prob_raw": 0.18,
    }
    if metadata is not None:
        md.update(metadata)
    return {
        "event": "fill",
        "order_id": order_id,
        "timestamp": timestamp,
        "market_id": market_ticker,
        "market_ticker": market_ticker,
        "position_key": market_ticker,
        "strategy": strategy,
        "side": side,
        "mode": mode,
        "model_probability": 0.18,
        "market_price": 0.20,
        "mid_at_signal": 0.20,
        "exec_price": 0.20,
        "edge": 0.05,
        "confidence": 0.8,
        "dollars": dollars,
        "contracts": contracts,
        "order_price": 0.20,
        "fill_price": fill_price,
        "is_taker_fills": True,
        "is_maker_fills": False,
        "fees_from_api": 0.05,
        "metadata": md,
    }


def _intraday_settlement_row(
    *,
    order_id: str,
    market_ticker: str,
    timestamp: str,
    side: str = "NO",
    result: str = "no",  # "no" = NO-side win; "yes" = NO-side loss
    pnl: float = 0.0,
    strategy: str = "intraday_emos_high",
    mode: str = "shadow",
) -> dict:
    """Build a settlement row matching engine._build_settlement_entry's schema.

    NOTE: settlement rows do NOT carry ``metadata`` (spec v1.3 MINOR #2,
    engine.py:1080-1093 verified). Cell is recovered by the helper via
    ``parse_ticker_city_var(market_ticker)``.
    """
    return {
        "event": "settlement",
        "order_id": order_id,
        "timestamp": timestamp,
        "market_id": market_ticker,
        "market_ticker": market_ticker,
        "position_key": market_ticker,
        "strategy": strategy,
        "side": side,
        "mode": mode,
        "result": result,
        "pnl": pnl,
    }


def _ticker_to_cell(ticker: str) -> str:
    """Local mirror used only for fill-row metadata seeding in fixtures."""
    if ticker.startswith("KXHIGHMIA-"):
        return "MIA/high"
    if ticker.startswith("KXHIGHLAX-"):
        return "LAX/high"
    if ticker.startswith("KXHIGHNY-"):
        return "NYC/high"
    return ""


def _write_trades_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_compute_intraday_consec_loss_runs_empty_corpus(tmp_path: Path):
    """Empty / missing file path returns empty per-cell dict, no crash."""
    missing = tmp_path / "trades.jsonl"
    result = compute_intraday_consec_loss_runs(missing)
    assert result == {}

    empty = tmp_path / "empty.jsonl"
    empty.touch()
    assert compute_intraday_consec_loss_runs(empty) == {}


def test_compute_intraday_consec_loss_runs_no_losses(tmp_path: Path):
    """All-wins: current_run_len=0, max_run_len_7d=0."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY12H1700-B85",
            timestamp="2026-05-12T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="O2",
            market_ticker="KXHIGHMIA-26MAY13H1700-B86",
            timestamp="2026-05-13T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    assert mia["current_run_len"] == 0
    assert mia["max_run_len_7d"] == 0
    assert mia["last_run_end_ts"] is None


def test_compute_intraday_consec_loss_runs_current_run_in_progress(tmp_path: Path):
    """A trailing run of N losses must surface as current_run_len=N."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="W1",
            market_ticker="KXHIGHMIA-26MAY10H1700-B85",
            timestamp="2026-05-10T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY11H1700-B85",
            timestamp="2026-05-11T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="L2",
            market_ticker="KXHIGHMIA-26MAY12H1700-B86",
            timestamp="2026-05-12T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="L3",
            market_ticker="KXHIGHMIA-26MAY13H1700-B87",
            timestamp="2026-05-13T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    assert mia["current_run_len"] == 3
    assert mia["max_run_len_7d"] == 3
    assert mia["last_run_end_ts"] == "2026-05-13T17:30:00+00:00"


def test_compute_intraday_consec_loss_runs_run_broken_by_win(tmp_path: Path):
    """A run ending in a win: max_run_len_7d retains run, current_run_len=0."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY10H1700-B85",
            timestamp="2026-05-10T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="L2",
            market_ticker="KXHIGHMIA-26MAY11H1700-B85",
            timestamp="2026-05-11T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="L3",
            market_ticker="KXHIGHMIA-26MAY12H1700-B85",
            timestamp="2026-05-12T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="W1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    assert mia["current_run_len"] == 0
    assert mia["max_run_len_7d"] == 3
    assert mia["last_run_end_ts"] == "2026-05-12T17:30:00+00:00"


def test_compute_intraday_consec_loss_runs_window_drops_old_runs(tmp_path: Path):
    """Losses outside the 7-day window must NOT contribute to max_run_len_7d."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        # Old run of 5 losses, all >7d before now (stale).
        _intraday_settlement_row(
            order_id="STALE1",
            market_ticker="KXHIGHMIA-26APR20H1700-B85",
            timestamp="2026-04-20T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="STALE2",
            market_ticker="KXHIGHMIA-26APR21H1700-B85",
            timestamp="2026-04-21T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="STALE3",
            market_ticker="KXHIGHMIA-26APR22H1700-B85",
            timestamp="2026-04-22T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="STALE4",
            market_ticker="KXHIGHMIA-26APR23H1700-B85",
            timestamp="2026-04-23T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="STALE5",
            market_ticker="KXHIGHMIA-26APR24H1700-B85",
            timestamp="2026-04-24T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        # Recent run of 2 losses (within 7d).
        _intraday_settlement_row(
            order_id="FRESH1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="FRESH2",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    # max_run_len_7d: only count runs whose losses fall inside the 7-day
    # window. The stale 5-run is outside; the fresh 2-run is inside;
    # the helper does NOT carry stale losses across the window boundary.
    assert mia["max_run_len_7d"] == 2
    # current_run_len is the WINDOW-AGNOSTIC trailing tail — 5 stale +
    # 2 fresh = 7 contiguous losses, no win to break them. Spec §11.4
    # frames current_run_len without a window qualifier so the operator
    # sees the actual streak; the §8.4(b) trigger reads max_run_len_7d.
    assert mia["current_run_len"] == 7


def test_compute_intraday_consec_loss_runs_filters_by_strategy(tmp_path: Path):
    """Rows from other strategies must be skipped, even if ticker is intraday."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="OTHER1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
            strategy="weather_edge",
        ),
        _intraday_settlement_row(
            order_id="INTRA1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="no",
            pnl=4.0,
            strategy="intraday_emos_high",
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    # weather_edge loss must not appear in MIA/high run state.
    mia = result.get("MIA/high", {})
    assert mia.get("current_run_len", 0) == 0
    assert mia.get("max_run_len_7d", 0) == 0


def test_compute_intraday_consec_loss_runs_ignores_non_settlement_events(tmp_path: Path):
    """Placement / fill / unknown rows must NOT count as outcomes."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        # A placement row for the to-be-settled position.
        {
            "event": "placement",
            "order_id": "P1",
            "timestamp": "2026-05-13T17:00:00+00:00",
            "market_ticker": "KXHIGHMIA-26MAY13H1700-B85",
            "strategy": "intraday_emos_high",
            "side": "NO",
        },
        # A fill row.
        _intraday_fill_row(
            order_id="F1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:15:00+00:00",
        ),
        # An unknown future event type.
        {
            "event": "cancellation",
            "order_id": "C1",
            "timestamp": "2026-05-13T17:20:00+00:00",
            "market_ticker": "KXHIGHMIA-26MAY13H1700-B85",
            "strategy": "intraday_emos_high",
            "side": "NO",
        },
        # The single real settlement.
        _intraday_settlement_row(
            order_id="S1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    # Exactly one loss event counted.
    assert mia["current_run_len"] == 1
    assert mia["max_run_len_7d"] == 1


def test_compute_intraday_consec_loss_runs_cell_derived_via_ticker(tmp_path: Path):
    """v1.3 MINOR #2: settlement rows have no metadata['cell'].

    The helper MUST derive ``cell`` from ``market_ticker`` via M3's
    ``parse_ticker_city_var``. This test would fail if the helper tried
    to read ``row["metadata"]["cell"]`` because settlement rows carry no
    metadata at all (engine.py:1080-1093 verified).
    """
    sink = tmp_path / "trades.jsonl"
    rows = [
        # MIA → MIA/high
        _intraday_settlement_row(
            order_id="L_MIA",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        # LAX → LAX/high
        _intraday_settlement_row(
            order_id="L_LAX",
            market_ticker="KXHIGHLAX-26MAY14H1700-B80",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    # Defensive assertion: settlement rows in the fixture must NOT carry metadata.
    for r in rows:
        assert "metadata" not in r, (
            "Test fixture invariant: settlement rows do NOT carry metadata "
            "per spec v1.3 MINOR #2 (engine.py:1080-1093)"
        )

    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    # Cells must be derived to "<city>/<var>" form (not bare "MIA").
    assert "MIA/high" in result
    assert "LAX/high" in result
    assert "MIA" not in result
    assert "LAX" not in result


def test_compute_intraday_consec_loss_runs_skips_unparseable_ticker(tmp_path: Path):
    """A settlement row with a non-intraday ticker drops out (defensive)."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="JUNK",
            market_ticker="GARBAGE-NOT-A-TICKER",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    # No crash; no cell entry produced.
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    assert result == {}


def test_compute_intraday_consec_loss_runs_chronological_order(tmp_path: Path):
    """Rows written out-of-order on disk must be walked chronologically."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        # On-disk order: latest first, then earliest, then middle.
        _intraday_settlement_row(
            order_id="C",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="A",
            market_ticker="KXHIGHMIA-26MAY12H1700-B85",
            timestamp="2026-05-12T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="B",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    result = compute_intraday_consec_loss_runs(sink, now_utc=now)
    mia = result["MIA/high"]
    # Chronological: W (5/12), L (5/13), L (5/14) → run=2, max=2.
    assert mia["current_run_len"] == 2
    assert mia["max_run_len_7d"] == 2
    assert mia["last_run_end_ts"] == "2026-05-14T17:30:00+00:00"


# ---------------------------------------------------------------------------
# summarize_intraday_strategy + JSON envelope
# ---------------------------------------------------------------------------


def test_summarize_intraday_strategy_empty_returns_empty_cells(tmp_path: Path):
    """No file → empty cells dict, no crash."""
    summary = summarize_intraday_strategy(tmp_path / "nope.jsonl")
    assert summary["cells"] == {}
    assert summary["lookback_days"] == 7


def test_summarize_intraday_strategy_per_cell_breakdown(tmp_path: Path):
    """One win + one loss on MIA/high → n=2, wins=1, losses=1, win_rate=0.5."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="W1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    mia = summary["cells"]["MIA/high"]
    assert mia["n_trades"] == 2
    assert mia["wins"] == 1
    assert mia["losses"] == 1
    assert mia["win_rate"] == pytest.approx(0.5)
    assert mia["pnl"] == pytest.approx(-1.0)
    assert mia["current_run_len"] == 1
    assert mia["max_run_len_7d"] == 1


def test_summarize_intraday_strategy_skips_non_intraday_strategies(tmp_path: Path):
    """weather_edge rows must not appear in the intraday section."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="OTHER",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
            strategy="weather_edge",
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    assert summary["cells"] == {}


def test_build_intraday_json_envelope_shape(tmp_path: Path):
    """JSON envelope exposes the cells map with documented keys."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHLAX-26MAY14H1700-B80",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    env = build_intraday_json(summary)
    assert "cells" in env
    lax = env["cells"]["LAX/high"]
    expected = {
        "n_trades",
        "wins",
        "losses",
        "win_rate",
        "pnl",
        "current_run_len",
        "max_run_len_7d",
        "last_run_end_ts",
    }
    assert expected.issubset(lax.keys())


# ---------------------------------------------------------------------------
# format_report: intraday section + §8.4(b) LAX permanent-disable ALERT
# ---------------------------------------------------------------------------


def test_format_report_intraday_section_renders_per_cell(tmp_path: Path):
    """Per-cell table appears with n/W/L counts."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="W1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    out = format_report(
        [], [], {}, "2026-05-15", intraday_summary=summary
    )
    assert "=== INTRADAY EMOS HIGH ===" in out
    assert "MIA/high" in out


def test_format_report_intraday_section_omitted_when_no_trades():
    """No intraday trades + no summary arg → section omitted entirely."""
    out = format_report([], [], {}, "2026-05-15")
    assert "INTRADAY EMOS HIGH" not in out


def test_format_report_intraday_section_empty_summary_renders_placeholder():
    """Empty summary (cells={}) → section header present, placeholder body."""
    out = format_report(
        [], [], {}, "2026-05-15",
        intraday_summary={"lookback_days": 7, "cells": {}},
    )
    assert "=== INTRADAY EMOS HIGH ===" in out
    assert "(no intraday trades" in out


def test_format_report_lax_permanent_disable_alert_fires(tmp_path: Path):
    """v1.3 §8.4(b): LAX/high with max_run_len_7d > 5 → ALERT row.

    The trigger threshold is ``LAX_PERMANENT_DISABLE_RUN_THRESHOLD`` (5).
    A run of 6 in any 7-day rolling window must surface in the report.
    """
    sink = tmp_path / "trades.jsonl"
    base = datetime(2026, 5, 8, 17, 30, 0, tzinfo=UTC)
    rows: list[dict] = []
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        rows.append(
            _intraday_settlement_row(
                order_id=f"L{i}",
                market_ticker=f"KXHIGHLAX-26MAY{8 + i:02d}H1700-B80",
                timestamp=ts,
                result="yes",
                pnl=-5.0,
            )
        )
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    out = format_report(
        [], [], {}, "2026-05-14", intraday_summary=summary
    )
    assert "ALERT" in out
    assert "LAX/high permanent-disable trigger armed" in out
    assert summary["cells"]["LAX/high"]["max_run_len_7d"] >= 6
    # Sanity: threshold constant is what the trigger reads.
    assert LAX_PERMANENT_DISABLE_RUN_THRESHOLD == 5
    # HIGH-3 (2026-05-15): the alert action MUST point at the real config
    # field (``enabled_cells``), not the nonexistent ``disabled_cells``.
    assert "enabled_cells" in out
    assert "disabled_cells" not in out


def test_format_report_mia_high_consec_loss_does_not_alert(tmp_path: Path):
    """§8.4 LAX-only asymmetry: MIA/high run of 6 must NOT trigger ALERT."""
    sink = tmp_path / "trades.jsonl"
    base = datetime(2026, 5, 8, 17, 30, 0, tzinfo=UTC)
    rows: list[dict] = []
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        rows.append(
            _intraday_settlement_row(
                order_id=f"L{i}",
                market_ticker=f"KXHIGHMIA-26MAY{8 + i:02d}H1700-B85",
                timestamp=ts,
                result="yes",
                pnl=-5.0,
            )
        )
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    out = format_report(
        [], [], {}, "2026-05-14", intraday_summary=summary
    )
    # The §8.4(b) clause is LAX-only — MIA must not trip the permanent-disable
    # ALERT row even at the same run length.
    assert "permanent-disable trigger armed" not in out


def test_intraday_helper_uses_central_strategy_constant():
    """Locks the strategy_name constant to the canonical value."""
    assert INTRADAY_STRATEGY_NAME == "intraday_emos_high"


def test_build_json_output_includes_intraday_top_level_key(tmp_path: Path):
    """Parallel to phase0_depth_probing: JSON envelope has intraday_emos_high key."""
    from scripts.daily_trade_report import build_json_output

    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    output = build_json_output(
        "2026-05-15", [], [], {}, intraday_summary=summary
    )
    assert "intraday_emos_high" in output
    assert "cells" in output["intraday_emos_high"]
    assert "MIA/high" in output["intraday_emos_high"]["cells"]


# ---------------------------------------------------------------------------
# M6 Fix #1 — pnl_ask / pnl_mid split (settlement-schema gap documented).
# ---------------------------------------------------------------------------


def test_summarize_intraday_strategy_emits_pnl_ask_and_pnl_mid(tmp_path: Path):
    """Per-cell summary must surface both ``pnl_ask`` and ``pnl_mid`` keys.

    ``pnl_mid`` is None because the settlement row's schema persists
    only one P&L number (engine.py:_build_settlement_entry, 1080-1093).
    The split is required by spec §11.4; the helper documents the gap.
    """
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
        _intraday_settlement_row(
            order_id="W1",
            market_ticker="KXHIGHMIA-26MAY14H1800-B85",
            timestamp="2026-05-14T18:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    mia = summary["cells"]["MIA/high"]
    assert mia["pnl_ask"] == pytest.approx(-1.0)
    # Settlement schema gap: explicit None, NOT 0.0 (preserves the
    # "not measured" semantic per the dataclass docstring).
    assert mia["pnl_mid"] is None
    # Legacy ``pnl`` key retained for backward compat.
    assert mia["pnl"] == pytest.approx(-1.0)


def test_build_intraday_json_envelope_includes_pnl_ask_and_pnl_mid(tmp_path: Path):
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHLAX-26MAY14H1700-B80",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-7.5,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    env = build_intraday_json(summary)
    lax = env["cells"]["LAX/high"]
    assert "pnl_ask" in lax
    assert "pnl_mid" in lax
    assert lax["pnl_ask"] == pytest.approx(-7.5)
    assert lax["pnl_mid"] is None


def test_format_report_renders_pnl_ask_and_pnl_mid_columns(tmp_path: Path):
    """Text report must include both pnl_ask + pnl_mid column headers."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    out = format_report([], [], {}, "2026-05-15", intraday_summary=summary)
    assert "pnl_ask" in out
    assert "pnl_mid" in out


# ---------------------------------------------------------------------------
# M6 Fix #2 — rolling 7-day ask CI lo (bootstrap B=2000 seed=20260514).
# ---------------------------------------------------------------------------


def test_bootstrap_ci_lo_empty_returns_none():
    """Absence is not zero — None is the canonical "no data" signal."""
    assert _bootstrap_ci_lo([], n_resamples=10, seed=1) is None


def test_bootstrap_ci_lo_deterministic_with_fixed_seed():
    """Same input + same seed must produce identical output (bit-identical)."""
    values = [3.0, -2.0, 5.0, -4.0, 1.0]
    a = _bootstrap_ci_lo(values, n_resamples=200, seed=42)
    b = _bootstrap_ci_lo(values, n_resamples=200, seed=42)
    assert a is not None and b is not None
    assert a == b


def test_bootstrap_ci_lo_uses_v6_canonical_seed():
    """The seed constant must match v6 audit
    (scripts/audit_same_day_gate_2026-05-14_v5_walkforward.py:81-82)."""
    assert INTRADAY_BOOTSTRAP_B == 2000
    assert INTRADAY_BOOTSTRAP_SEED == 20260514


def test_summarize_intraday_strategy_emits_ask_ci_lo_7d(tmp_path: Path):
    """Per-cell summary must surface ``ask_ci_lo_7d`` (float | None)."""
    sink = tmp_path / "trades.jsonl"
    # Mix of wins + losses so bootstrap distribution has variance.
    rows = []
    for i in range(8):
        pnl = 4.0 if i % 2 == 0 else -5.0
        result = "no" if pnl > 0 else "yes"
        rows.append(
            _intraday_settlement_row(
                order_id=f"O{i}",
                market_ticker=f"KXHIGHMIA-26MAY{8 + i:02d}H1700-B85",
                timestamp=f"2026-05-{8 + i:02d}T17:30:00+00:00",
                result=result,
                pnl=pnl,
            )
        )
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now, lookback_days=10)
    mia = summary["cells"]["MIA/high"]
    assert "ask_ci_lo_7d" in mia
    assert isinstance(mia["ask_ci_lo_7d"], (int, float))


def test_summarize_intraday_strategy_ask_ci_lo_none_for_empty_cell(tmp_path: Path):
    """Cell with no in-window settlements: CI lo should not appear (cell absent)."""
    sink = tmp_path / "trades.jsonl"
    # All rows outside the lookback window.
    rows = [
        _intraday_settlement_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26APR01H1700-B85",
            timestamp="2026-04-01T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now, lookback_days=7)
    assert summary["cells"] == {}


# ---------------------------------------------------------------------------
# M6 Fix #3 — per-cell calibration ratio (fill-row metadata join by order_id).
# ---------------------------------------------------------------------------


def test_p_loss_from_emos_no_side_yes_side_conventions():
    """§6.3 sign convention: emos_prob_raw == p(YES) == p(bracket fires).

    NO side loses on bracket-fire (p_loss = emos);
    YES side loses on no-fire   (p_loss = 1 - emos).
    """
    assert _p_loss_from_emos("NO", 0.30) == pytest.approx(0.30)
    assert _p_loss_from_emos("YES", 0.30) == pytest.approx(0.70)
    assert _p_loss_from_emos("NO", None) is None
    assert _p_loss_from_emos(None, 0.30) is None
    assert _p_loss_from_emos("INVALID", 0.30) is None


def test_calibration_ratio_uses_fill_row_emos_join(tmp_path: Path):
    """Settlement rows don't carry emos_prob_raw; the helper must join via
    order_id to the matching fill row to compute calibration ratio.
    """
    sink = tmp_path / "trades.jsonl"
    rows = [
        # Two fill rows (carry emos_prob_raw in metadata).
        _intraday_fill_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:00:00+00:00",
            metadata={"emos_prob_raw": 0.20},  # p(YES) → p(loss | NO) = 0.20
        ),
        _intraday_fill_row(
            order_id="O2",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:00:00+00:00",
            metadata={"emos_prob_raw": 0.40},  # p(YES) → p(loss | NO) = 0.40
        ),
        # Two matching settlements: 1 NO win + 1 NO loss → actual_loss = 0.5
        _intraday_settlement_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY13H1700-B85",
            timestamp="2026-05-13T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
        _intraday_settlement_row(
            order_id="O2",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    mia = summary["cells"]["MIA/high"]
    # avg p(loss) = (0.20 + 0.40) / 2 = 0.30
    assert mia["avg_model_p_loss"] == pytest.approx(0.30)
    # actual loss rate = 1 / 2 = 0.50
    assert mia["actual_loss_rate"] == pytest.approx(0.50)
    # ratio = 0.30 / 0.50 = 0.60 (model underestimates losses → overconfident)
    assert mia["calibration_ratio"] == pytest.approx(0.60)


def test_calibration_ratio_none_when_no_fill_metadata(tmp_path: Path):
    """If fill rows are absent (settlement-only log), ratio reads as None
    (avg_model_p_loss is None — no model attribution available)."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    mia = summary["cells"]["MIA/high"]
    assert mia["avg_model_p_loss"] is None
    assert mia["calibration_ratio"] is None


def test_calibration_ratio_none_when_no_losses(tmp_path: Path):
    """All-wins cell: actual_loss_rate = 0 → ratio undefined → None."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_fill_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:00:00+00:00",
            metadata={"emos_prob_raw": 0.10},
        ),
        _intraday_settlement_row(
            order_id="O1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="no",
            pnl=4.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    mia = summary["cells"]["MIA/high"]
    assert mia["actual_loss_rate"] == 0.0
    assert mia["calibration_ratio"] is None


# ---------------------------------------------------------------------------
# M6 Fix #4 — §8.5 trigger flags armed today.
# ---------------------------------------------------------------------------


def test_summarize_intraday_strategy_emits_triggers_per_cell(tmp_path: Path):
    """``triggers[cell]`` is a list of dicts with the documented schema."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    triggers = summary["triggers"]["MIA/high"]
    names = {t["name"] for t in triggers}
    # All five §8.5 triggers must be present (mu_shift deferred, others armed).
    assert {
        "mu_shift_gt_5F",
        "pnl_7d_below_floor",
        "win_rate_7d_below_floor",
        "ci_straddles_zero_at_n30",
        "consec_loss_gt_threshold",
    }.issubset(names)


def test_trigger_mu_shift_is_deferred_todo():
    """μ-shift trigger is honestly deferred (TODO) — NOT silently skipped."""
    # Build a tiny inline summary by hand so we don't need a fixture.
    from scripts.daily_trade_report import (
        IntradayCellStats,
        _evaluate_intraday_triggers,
    )

    st = IntradayCellStats(cell="MIA/high", n_trades=5)
    triggers = _evaluate_intraday_triggers("MIA/high", st, ask_ci_lo_7d=None)
    mu = next(t for t in triggers if t["name"] == "mu_shift_gt_5F")
    assert mu["deferred"] is True
    assert mu["fired"] is False
    assert "TODO" in (mu.get("note") or "")
    assert "§8.5(2)(a)" in (mu.get("note") or "")


def test_trigger_pnl_floor_fires_when_below(tmp_path: Path):
    """Single loss of -$250 in the window must trigger §8.5(3) (< -$200)."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-250.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    triggers = summary["triggers"]["MIA/high"]
    pnl_trig = next(t for t in triggers if t["name"] == "pnl_7d_below_floor")
    assert pnl_trig["fired"] is True
    assert pnl_trig["threshold"] == INTRADAY_TRIGGER_PNL_FLOOR


def test_trigger_win_rate_gated_on_min_n(tmp_path: Path):
    """Win-rate trigger is gated on n >= 20 per spec §8.5(4)."""
    # Tiny sample of all-losses → still NOT armed (n < 20).
    sink = tmp_path / "trades.jsonl"
    rows = []
    for i in range(5):
        rows.append(
            _intraday_settlement_row(
                order_id=f"L{i}",
                market_ticker=f"KXHIGHMIA-26MAY{10 + i:02d}H1700-B85",
                timestamp=f"2026-05-{10 + i:02d}T17:30:00+00:00",
                result="yes",
                pnl=-5.0,
            )
        )
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    triggers = summary["triggers"]["MIA/high"]
    wr = next(t for t in triggers if t["name"] == "win_rate_7d_below_floor")
    # n=5 < 20 → not armed even though win_rate is 0% (well below 35%).
    assert wr["fired"] is False
    assert INTRADAY_TRIGGER_WIN_RATE_MIN_N == 20
    assert INTRADAY_TRIGGER_WIN_RATE_FLOOR == 0.35


def test_trigger_consec_loss_fires_above_threshold(tmp_path: Path):
    """Consec-loss > 5 must arm the §8.5(consec) trigger."""
    sink = tmp_path / "trades.jsonl"
    rows = []
    base = datetime(2026, 5, 8, 17, 30, 0, tzinfo=UTC)
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        rows.append(
            _intraday_settlement_row(
                order_id=f"L{i}",
                market_ticker=f"KXHIGHMIA-26MAY{8 + i:02d}H1700-B85",
                timestamp=ts,
                result="yes",
                pnl=-5.0,
            )
        )
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    triggers = summary["triggers"]["MIA/high"]
    consec = next(t for t in triggers if t["name"] == "consec_loss_gt_threshold")
    assert consec["current"] == 6
    assert consec["threshold"] == INTRADAY_TRIGGER_CONSEC_LOSS_MAX
    assert consec["fired"] is True


def test_format_report_renders_triggers_section(tmp_path: Path):
    """Text report must include the §8.5 trigger block."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHMIA-26MAY14H1700-B85",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    out = format_report([], [], {}, "2026-05-15", intraday_summary=summary)
    assert "§8.5 triggers armed today" in out
    assert "mu_shift_gt_5F" in out  # deferred row visible


def test_build_intraday_json_envelope_includes_triggers(tmp_path: Path):
    """JSON envelope must surface triggers per cell for ops automation."""
    sink = tmp_path / "trades.jsonl"
    rows = [
        _intraday_settlement_row(
            order_id="L1",
            market_ticker="KXHIGHLAX-26MAY14H1700-B80",
            timestamp="2026-05-14T17:30:00+00:00",
            result="yes",
            pnl=-5.0,
        ),
    ]
    _write_trades_jsonl(sink, rows)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    env = build_intraday_json(summary)
    assert "triggers" in env
    assert "LAX/high" in env["triggers"]
    assert any(
        t["name"] == "consec_loss_gt_threshold"
        for t in env["triggers"]["LAX/high"]
    )


# ---------------------------------------------------------------------------
# M6 Fix #5 — §9.2(b) trade-count-cap utilisation from scan_funnel.jsonl.
# ---------------------------------------------------------------------------


def test_read_scan_funnel_cap_total_missing_file_returns_zero(tmp_path: Path):
    """Missing scan_funnel.jsonl → 0 (no crash). Early-shadow days."""
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    total = _read_intraday_scan_funnel_cap_total(
        tmp_path / "scan_funnel.jsonl", lookback_days=7, now_utc=now
    )
    assert total == 0


def test_read_scan_funnel_cap_total_sums_intraday_rows(tmp_path: Path):
    """Sum ``n_after_trades_per_cell_per_day_cap`` over intraday rows in window."""
    funnel = tmp_path / "scan_funnel.jsonl"
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    rows = [
        # Inside window, intraday: counted.
        {
            "strategy": "intraday_emos_high",
            "scan_id": "A",
            "scan_ts": (now - timedelta(hours=1)).timestamp(),
            "n_after_trades_per_cell_per_day_cap": 2,
        },
        # Inside window, intraday: counted.
        {
            "strategy": "intraday_emos_high",
            "scan_id": "B",
            "scan_ts": (now - timedelta(hours=2)).timestamp(),
            "n_after_trades_per_cell_per_day_cap": 1,
        },
        # Wrong strategy: skipped.
        {
            "strategy": "weather_edge",
            "scan_id": "C",
            "scan_ts": (now - timedelta(hours=1)).timestamp(),
            "n_after_trades_per_cell_per_day_cap": 5,
        },
        # Outside window: skipped.
        {
            "strategy": "intraday_emos_high",
            "scan_id": "D",
            "scan_ts": (now - timedelta(days=14)).timestamp(),
            "n_after_trades_per_cell_per_day_cap": 7,
        },
    ]
    funnel.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    total = _read_intraday_scan_funnel_cap_total(
        funnel, lookback_days=7, now_utc=now
    )
    assert total == 3  # 2 + 1


def test_summarize_intraday_strategy_surfaces_cap_utilisation(tmp_path: Path):
    """Top-level ``trades_per_cell_cap_utilisation_7d`` reflects funnel total."""
    sink = tmp_path / "trades.jsonl"
    funnel = tmp_path / "scan_funnel.jsonl"
    _write_trades_jsonl(
        sink,
        [
            _intraday_settlement_row(
                order_id="L1",
                market_ticker="KXHIGHMIA-26MAY14H1700-B85",
                timestamp="2026-05-14T17:30:00+00:00",
                result="yes",
                pnl=-5.0,
            ),
        ],
    )
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    funnel_rows = [
        {
            "strategy": "intraday_emos_high",
            "scan_id": "A",
            "scan_ts": (now - timedelta(hours=1)).timestamp(),
            "n_after_trades_per_cell_per_day_cap": 4,
        },
    ]
    funnel.write_text("\n".join(json.dumps(r) for r in funnel_rows) + "\n")
    summary = summarize_intraday_strategy(
        sink, now_utc=now, scan_funnel_path=funnel
    )
    assert summary["trades_per_cell_cap_utilisation_7d"] == 4


def test_format_report_renders_cap_utilisation_alert_when_nonzero(tmp_path: Path):
    """Cap-utilisation > 0 raises a `[!!]` ALERT line citing event-key drift."""
    sink = tmp_path / "trades.jsonl"
    funnel = tmp_path / "scan_funnel.jsonl"
    _write_trades_jsonl(
        sink,
        [
            _intraday_settlement_row(
                order_id="L1",
                market_ticker="KXHIGHMIA-26MAY14H1700-B85",
                timestamp="2026-05-14T17:30:00+00:00",
                result="yes",
                pnl=-5.0,
            ),
        ],
    )
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    funnel.write_text(
        json.dumps(
            {
                "strategy": "intraday_emos_high",
                "scan_id": "A",
                "scan_ts": (now - timedelta(hours=1)).timestamp(),
                "n_after_trades_per_cell_per_day_cap": 2,
            }
        )
        + "\n"
    )
    summary = summarize_intraday_strategy(
        sink, now_utc=now, scan_funnel_path=funnel
    )
    out = format_report([], [], {}, "2026-05-15", intraday_summary=summary)
    assert "[!!]" in out
    assert "event-key drift" in out


def test_format_report_no_cap_utilisation_no_alert(tmp_path: Path):
    """Cap-utilisation == 0 → no ALERT line."""
    sink = tmp_path / "trades.jsonl"
    _write_trades_jsonl(
        sink,
        [
            _intraday_settlement_row(
                order_id="L1",
                market_ticker="KXHIGHMIA-26MAY14H1700-B85",
                timestamp="2026-05-14T17:30:00+00:00",
                result="yes",
                pnl=-5.0,
            ),
        ],
    )
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    # No scan_funnel.jsonl file at all.
    summary = summarize_intraday_strategy(
        sink, now_utc=now, scan_funnel_path=tmp_path / "absent.jsonl"
    )
    out = format_report([], [], {}, "2026-05-15", intraday_summary=summary)
    # Surface line exists but no alert (count: 0).
    assert "§9.2(b) trades-per-cell-cap utilisation" in out
    assert "[!!]" not in out


def test_build_intraday_json_envelope_includes_cap_utilisation(tmp_path: Path):
    sink = tmp_path / "trades.jsonl"
    _write_trades_jsonl(
        sink,
        [
            _intraday_settlement_row(
                order_id="L1",
                market_ticker="KXHIGHLAX-26MAY14H1700-B80",
                timestamp="2026-05-14T17:30:00+00:00",
                result="yes",
                pnl=-5.0,
            ),
        ],
    )
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    summary = summarize_intraday_strategy(sink, now_utc=now)
    env = build_intraday_json(summary)
    assert "trades_per_cell_cap_utilisation_7d" in env
    assert isinstance(env["trades_per_cell_cap_utilisation_7d"], int)


# ---------------------------------------------------------------------------
# Constants — guard against unintended drift.
# ---------------------------------------------------------------------------


def test_intraday_trigger_constants_match_spec():
    """Spec §8.5 threshold pins. Drift here = silent posture relaxation."""
    assert INTRADAY_TRIGGER_PNL_FLOOR == -200.0
    assert INTRADAY_TRIGGER_WIN_RATE_FLOOR == 0.35
    assert INTRADAY_TRIGGER_WIN_RATE_MIN_N == 20
    assert INTRADAY_TRIGGER_CONSEC_LOSS_MAX == 5
    assert INTRADAY_TRIGGER_CI_STRADDLE_MIN_N == 30


# ---------------------------------------------------------------------------
# Strategy first-class reporting (de-conflation + per-strategy P&L)
# ---------------------------------------------------------------------------


def _attr_row(*, strategy: str, won: bool, city: str = "NYC", order_price: float = 0.50) -> dict:
    """Settlement-file-shaped row that attribute_trade accepts."""
    contracts = 10
    pnl = (1.0 - order_price) * contracts if won else -(order_price * contracts)
    return {
        "timestamp": "2026-05-26T14:00:00+00:00",
        "market_ticker": f"KXHIGH{city}-26MAY27-T84",
        "strategy": strategy,
        "mode": "live",
        "side": "YES",
        "settled": True,
        "won": won,
        "contracts": contracts,
        "model_probability": 0.55,
        "order_price": order_price,
        "exec_price": order_price,
        "pnl": round(pnl, 2),
        "fee": 0.0,
        "market_result": "yes" if won else "no",
        "metadata": {"city": city, "market_type": "high", "forecast_date": "2026-05-27"},
    }


def test_filter_by_strategy_allowlist_keeps_only_members():
    from scripts.daily_trade_report import WEATHER_STRATEGY_NAMES, filter_by_strategy

    records = [
        {"strategy": "weather_edge"},
        {"strategy": "favorite_bias"},
        {"strategy": None},  # legacy untagged -> "unknown"
    ]
    weather = filter_by_strategy(records, WEATHER_STRATEGY_NAMES)
    assert weather == [{"strategy": "weather_edge"}]
    # None allowlist keeps everything
    assert filter_by_strategy(records, None) == records
    # single-strategy scope; legacy maps to "unknown"
    assert filter_by_strategy(records, frozenset({"unknown"})) == [{"strategy": None}]


def test_compute_calibration_gap_excludes_non_weather():
    from scripts.daily_trade_report import WEATHER_STRATEGY_NAMES, compute_calibration_gap

    records = [
        _attr_row(strategy="weather_edge", won=True),
        _attr_row(strategy="weather_edge", won=False),
        # favorite_bias rows carry model_probability=g(mid) but must NOT pollute
        # weather calibration:
        _attr_row(strategy="favorite_bias", won=False),
        _attr_row(strategy="favorite_bias", won=False),
    ]
    cal = compute_calibration_gap(records, strategy_allowlist=WEATHER_STRATEGY_NAMES)
    assert cal["n_settled"] == 2  # only the two weather_edge rows
    assert cal["n_strategy_filtered"] == 2  # the two favorite_bias rows dropped


def test_b1_weather_scope_excludes_favorite_bias_from_per_cell():
    """A favorite_bias loss in NYC/high must not appear in settled_per_cell —
    that feed drives post_flip_monitor.sh T3 for the weather_edge pilot."""
    from scripts.daily_trade_report import WEATHER_STRATEGY_NAMES, filter_by_strategy

    records = [
        _attr_row(strategy="weather_edge", won=False, city="NYC"),
        _attr_row(strategy="favorite_bias", won=False, city="NYC"),
    ]
    weather_filtered = filter_by_strategy(records, WEATHER_STRATEGY_NAMES)
    per_cell = aggregate_per_cell(build_settled_report(weather_filtered))
    # Only the single weather_edge loss is counted for NYC/high.
    assert per_cell["NYC/high"]["n_losses"] == 1


def test_build_strategy_pnl_summary_per_strategy_excludes_intraday():
    from scripts.daily_trade_report import build_strategy_pnl_summary

    records = [
        _attr_row(strategy="weather_edge", won=True),
        _attr_row(strategy="weather_edge", won=False),
        _attr_row(strategy="favorite_bias", won=True),
        _attr_row(strategy=INTRADAY_STRATEGY_NAME, won=True),  # excluded
    ]
    summary = build_strategy_pnl_summary(records)
    assert set(summary) == {"weather_edge", "favorite_bias"}
    assert summary["weather_edge"]["n"] == 2
    assert summary["weather_edge"]["wins"] == 1
    assert summary["favorite_bias"]["n"] == 1
    assert summary["favorite_bias"]["win_rate_pct"] == pytest.approx(100.0)


def test_build_json_output_includes_by_strategy():
    from scripts.daily_trade_report import build_json_output, build_strategy_pnl_summary

    records = [_attr_row(strategy="favorite_bias", won=True)]
    summary = build_strategy_pnl_summary(records)
    out = build_json_output("2026-05-26", [], [], {}, strategy_pnl=summary)
    assert "by_strategy" in out
    assert out["by_strategy"]["favorite_bias"]["n"] == 1


def test_format_report_renders_strategy_section():
    from scripts.daily_trade_report import build_strategy_pnl_summary, format_report

    records = [_attr_row(strategy="favorite_bias", won=True)]
    summary = build_strategy_pnl_summary(records)
    text = format_report([], [], {}, "2026-05-26", strategy_pnl=summary)
    assert "SETTLED P&L BY STRATEGY" in text
    assert "favorite_bias" in text
