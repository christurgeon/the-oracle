"""Tests for scripts/intraday_asos_shadow_report.py.

Uses synthetic fixtures: a temp predictions.jsonl with known
``intraday_asos_high`` rows (various guard_outcome values, sigma_gain bands, and
posterior_p_raw values placed into specific bins), plus a fixture settlements
dict.

Tests verify (the three audit-defect fixes + supporting behavior):
  - CRITICAL-1: per-settlement dedup collapses many tick-rows of the SAME open
    market to ONE row (the sharpest posterior, tie-broken latest ts)
  - CRITICAL-2: σ_gain stratification keeps wide-σ rows OUT of the headline
    cohort and in a separate diagnostic table
  - CRITICAL-3: Wilson-CI kill criteria fire PASS / FAIL / INCONCLUSIVE /
    INSUFFICIENT-N at known counts (point-threshold cannot)
  - MAJOR-2: informational LAX cell excluded from the §9 headline cohort;
    per-cell verdicts produced
  - MINOR: parse-loss reconciliation counts malformed rows
  - per-bin reliability table has correct n/pred/real for known fixtures
  - settlement join + prelim/final + zero-corpus graceful exit are preserved
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from python.analytics.prediction_log import PredictionRecord, _serialize
from python.common import Side
from scripts._asos_reliability_bins import (
    ASOS_BINS,
    _bin_label,
    bin_index,
    compute_reliability_bins,
)
from scripts.intraday_asos_shadow_report import (
    SHARP_SIGMA_GAIN_MAX,
    _eval_kill_criteria_raw,
    build_report,
    dedup_to_settlement_events,
    evaluate_outcome,
    is_sharp,
    load_informational_cells,
    lookup_actual_temp,
    render_report,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(
    *,
    market_ticker: str = "KXHIGHMI-26JUN10-T88",
    city: str = "MIA",
    daily_variable: str = "high",
    strike_type: str = "greater",
    floor_strike: float | None = 88.0,
    cap_strike: float | None = None,
    expiration_date: date | None = date(2026, 6, 10),
    posterior_p_raw: float | None = 0.08,
    calibrated_prob: float = 0.08,
    guard_outcome: str | None = "pass",
    strategy: str = "intraday_asos_high",
    side: Side = Side.NO,
    exec_price: float | None = 0.90,
    sigma_gain: float | None = 0.9,
    timestamp: datetime | None = None,
    frame_version: int = 2,
) -> PredictionRecord:
    # Default frame_version=2 (LST): existing tests exercise the BINDING cohort.
    # The frame cut (Stage 0a) routes frame_version<2 into the separate
    # frame_v1_diagnostic block, so headline-cohort fixtures must be frame-v2.
    return PredictionRecord(
        timestamp=timestamp or datetime(2026, 6, 9, 20, 0, 0, tzinfo=UTC),
        strategy=strategy,
        frame_version=frame_version,
        market_ticker=market_ticker,
        side=side,
        city=city,
        daily_variable=daily_variable,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        expiration_date=expiration_date,
        raw_ensemble_prob=None,
        calibrated_prob=calibrated_prob,
        market_price=0.85,
        exec_price=exec_price,
        confidence=0.7,
        posterior_p_raw=posterior_p_raw,
        guard_outcome=guard_outcome,
        post_mean_f=87.0,
        boundary_distance_f=2.5,
        asos_M=86.5,
        mu_gain=0.5,
        sigma_gain=sigma_gain,
        climo_n=300,
        climo_source="asos_climo_v1",
        prelim_drift_f=0.3,
    )


def _write_preds(path: Path, recs: list[PredictionRecord]) -> None:
    lines = [json.dumps(_serialize(r), default=str, separators=(",", ":")) for r in recs]
    path.write_text("\n".join(lines) + "\n")


def _settlement_entry(
    city: str,
    obs_date: str,
    high_f: float,
    *,
    is_preliminary: bool = False,
) -> dict:
    return {
        "city": city,
        "observation_date": obs_date,
        "high_f": high_f,
        "low_f": 70.0,
        "is_preliminary": is_preliminary,
        "issued_at": "2026-06-11T06:00:00+00:00",
        "product_id": "MIA_final",
    }


def _write_snaps(snaps_dir: Path, entries_by_date: dict[str, list[dict]]) -> None:
    for date_str, entries in entries_by_date.items():
        day_dir = snaps_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "settlements.json").write_text(json.dumps(entries))


def _merge_settlements(*dicts: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Merge per-date settlement lists (concat lists for shared dates).

    A plain ``{**a, **b}`` would overwrite shared date keys, silently dropping
    one city's settlements — this concatenates instead.
    """
    out: dict[str, list[dict]] = {}
    for d in dicts:
        for date_str, entries in d.items():
            out.setdefault(date_str, []).extend(entries)
    return out


def _make_report(
    recs: list[PredictionRecord],
    settlements_by_date: dict[str, list[dict]],
    *,
    informational_cells: list[str] | None = None,
) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pred_path = tmp / "predictions.jsonl"
        snaps_dir = tmp / "snapshots"
        _write_preds(pred_path, recs)
        _write_snaps(snaps_dir, settlements_by_date)
        return build_report(pred_path, snaps_dir, informational_cells=informational_cells)


def _low_p_event_recs(
    n: int,
    n_hits: int,
    *,
    city: str = "MIA",
    floor: float = 88.0,
    sigma_gain: float = 0.9,
    p: float = 0.07,
) -> tuple[list[PredictionRecord], dict[str, list[dict]]]:
    """Build ``n`` distinct settlement events (low-p NO bin); ``n_hits`` YES."""
    recs: list[PredictionRecord] = []
    settlements: dict[str, list[dict]] = {}
    cc = {"MIA": "MI", "AUS": "AU", "NYC": "NY", "LAX": "LA"}[city]
    for i in range(n):
        day = f"2026-06-{10 + i:02d}"
        ticker = f"KXHIGH{cc}-26JUN{10 + i:02d}-T{int(floor)}"
        recs.append(
            _rec(
                market_ticker=ticker,
                city=city,
                expiration_date=date(2026, 6, 10 + i),
                posterior_p_raw=p,
                floor_strike=floor,
                sigma_gain=sigma_gain,
                guard_outcome="pass",
            )
        )
        actual = floor + 1.0 if i < n_hits else floor - 1.0
        settlements[day] = [_settlement_entry(city, day, actual)]
    return recs, settlements


# ---------------------------------------------------------------------------
# _asos_reliability_bins helpers (preserved)
# ---------------------------------------------------------------------------


class TestBinIndex:
    def test_first_bin(self) -> None:
        assert bin_index(0.0) == 0
        assert bin_index(0.04) == 0

    def test_second_bin(self) -> None:
        assert bin_index(0.05) == 1
        assert bin_index(0.09) == 1

    def test_last_bin(self) -> None:
        assert bin_index(0.95) == len(ASOS_BINS) - 2
        assert bin_index(1.0) == len(ASOS_BINS) - 2

    def test_out_of_range(self) -> None:
        assert bin_index(-0.01) is None

    def test_various_bins(self) -> None:
        assert bin_index(0.15) == 2
        assert bin_index(0.8) == 6


class TestBinLabel:
    def test_normal_bin(self) -> None:
        assert _bin_label(0.05, 0.10) == "[0.05,0.10)"

    def test_sentinel_upper(self) -> None:
        assert _bin_label(0.95, 1.001) == "[0.95,1.00)"


class TestComputeReliabilityBins:
    def test_empty_input(self) -> None:
        assert compute_reliability_bins([], []) == []

    def test_single_bin(self) -> None:
        probs = [0.06, 0.07, 0.08, 0.09]
        outcomes = [0, 0, 1, 0]
        rows = compute_reliability_bins(probs, outcomes)
        assert len(rows) == 1
        row = rows[0]
        assert row["bin"] == "[0.05,0.10)"
        assert row["n"] == 4
        assert row["pred"] == pytest.approx((0.06 + 0.07 + 0.08 + 0.09) / 4)
        assert row["real"] == pytest.approx(0.25)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_reliability_bins([0.5, 0.6], [1])


# ---------------------------------------------------------------------------
# Wilson CI (CRITICAL-3) — known values
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_n_is_maximally_uninformative(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_known_value_half(self) -> None:
        # p=0.5, n=10: classic Wilson interval ≈ (0.237, 0.763).
        lo, hi = wilson_ci(5, 10)
        assert lo == pytest.approx(0.2366, abs=1e-3)
        assert hi == pytest.approx(0.7634, abs=1e-3)

    def test_known_value_all_success(self) -> None:
        # 10/10: Wilson lower ≈ 0.7224, upper clamps to 1.0.
        lo, hi = wilson_ci(10, 10)
        assert lo == pytest.approx(0.7224, abs=1e-3)
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_known_value_zero_success(self) -> None:
        # 0/20: lower 0.0, upper ≈ 0.1611.
        lo, hi = wilson_ci(0, 20)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(0.1611, abs=1e-3)

    def test_low_rate_at_n20(self) -> None:
        # 2/20 (realized 0.10): Wilson ≈ (0.0279, 0.3010).
        lo, hi = wilson_ci(2, 20)
        assert lo == pytest.approx(0.0279, abs=2e-3)
        assert hi == pytest.approx(0.3010, abs=2e-3)

    def test_invalid_successes(self) -> None:
        with pytest.raises(ValueError):
            wilson_ci(5, 3)

    def test_monotone_narrowing_with_n(self) -> None:
        # Same rate, larger n → narrower interval.
        lo_small, hi_small = wilson_ci(2, 20)
        lo_big, hi_big = wilson_ci(20, 200)
        assert (hi_big - lo_big) < (hi_small - lo_small)


# ---------------------------------------------------------------------------
# Dedup (CRITICAL-1) — pure function
# ---------------------------------------------------------------------------


class TestDedup:
    def test_single_event_many_ticks_collapses_to_one(self) -> None:
        ts0 = datetime(2026, 6, 10, 15, 0, tzinfo=UTC)
        recs = [
            _rec(
                market_ticker="KXHIGHMI-26JUN10-T88",
                expiration_date=date(2026, 6, 10),
                sigma_gain=2.0,  # wide morning tick
                timestamp=ts0,
            ),
            _rec(
                market_ticker="KXHIGHMI-26JUN10-T88",
                expiration_date=date(2026, 6, 10),
                sigma_gain=0.6,  # SHARPEST post-peak tick → winner
                timestamp=ts0.replace(hour=20),
            ),
            _rec(
                market_ticker="KXHIGHMI-26JUN10-T88",
                expiration_date=date(2026, 6, 10),
                sigma_gain=1.0,
                timestamp=ts0.replace(hour=22),
            ),
        ]
        out = dedup_to_settlement_events(recs)
        assert len(out) == 1
        assert out[0].sigma_gain == pytest.approx(0.6)

    def test_tie_break_latest_ts(self) -> None:
        ts0 = datetime(2026, 6, 10, 18, 0, tzinfo=UTC)
        early = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            sigma_gain=0.7,
            timestamp=ts0,
            posterior_p_raw=0.05,
        )
        late = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            sigma_gain=0.7,  # same σ → tie
            timestamp=ts0.replace(hour=21),
            posterior_p_raw=0.09,
        )
        out = dedup_to_settlement_events([early, late])
        assert len(out) == 1
        assert out[0].posterior_p_raw == pytest.approx(0.09)  # latest ts wins

    def test_distinct_tickers_not_merged(self) -> None:
        recs = [
            _rec(market_ticker="KXHIGHMI-26JUN10-T88", expiration_date=date(2026, 6, 10)),
            _rec(market_ticker="KXHIGHMI-26JUN10-T89", expiration_date=date(2026, 6, 10)),
        ]
        out = dedup_to_settlement_events(recs)
        assert len(out) == 2

    def test_none_sigma_loses_to_real_sigma(self) -> None:
        ts0 = datetime(2026, 6, 10, 18, 0, tzinfo=UTC)
        no_sigma = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            sigma_gain=None,
            timestamp=ts0.replace(hour=23),  # later, but no σ
            posterior_p_raw=0.5,
        )
        sharp = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            sigma_gain=0.8,
            timestamp=ts0,
            posterior_p_raw=0.06,
        )
        out = dedup_to_settlement_events([no_sigma, sharp])
        assert len(out) == 1
        assert out[0].sigma_gain == pytest.approx(0.8)


class TestIsSharp:
    def test_sharp_at_cutoff(self) -> None:
        assert is_sharp(_rec(sigma_gain=SHARP_SIGMA_GAIN_MAX)) is True

    def test_wide_above_cutoff(self) -> None:
        assert is_sharp(_rec(sigma_gain=SHARP_SIGMA_GAIN_MAX + 0.5)) is False

    def test_none_not_sharp(self) -> None:
        assert is_sharp(_rec(sigma_gain=None)) is False


# ---------------------------------------------------------------------------
# Settlement lookup (preserved)
# ---------------------------------------------------------------------------


class TestLookupActualTemp:
    def _settlements(self) -> dict:
        return {
            ("MIA", "2026-06-10"): {
                "city": "MIA",
                "observation_date": "2026-06-10",
                "high_f": 91.0,
                "low_f": 78.0,
            }
        }

    def test_found_high(self) -> None:
        assert lookup_actual_temp(self._settlements(), "MIA", "high", "2026-06-10") == 91.0

    def test_found_max_alias(self) -> None:
        assert (
            lookup_actual_temp(self._settlements(), "MIA", "temperature_2m_max", "2026-06-10")
            == 91.0
        )

    def test_not_found_city(self) -> None:
        assert lookup_actual_temp(self._settlements(), "NYC", "high", "2026-06-10") is None

    def test_not_found_date(self) -> None:
        assert lookup_actual_temp(self._settlements(), "MIA", "high", "2026-06-11") is None

    def test_unknown_variable(self) -> None:
        assert lookup_actual_temp(self._settlements(), "MIA", "precip", "2026-06-10") is None


# ---------------------------------------------------------------------------
# Outcome evaluation (preserved)
# ---------------------------------------------------------------------------


class TestEvaluateOutcome:
    def test_greater_hit(self) -> None:
        r = _rec(strike_type="greater", floor_strike=88.0)
        assert evaluate_outcome(r, 89.0) == 1

    def test_greater_miss_at_boundary(self) -> None:
        r = _rec(strike_type="greater", floor_strike=88.0)
        assert evaluate_outcome(r, 88.0) == 0

    def test_less_hit(self) -> None:
        r = _rec(strike_type="less", floor_strike=None, cap_strike=88.0)
        assert evaluate_outcome(r, 87.0) == 1

    def test_between_closed_interval(self) -> None:
        r = _rec(strike_type="between", floor_strike=85.0, cap_strike=89.0)
        assert evaluate_outcome(r, 85.0) == 1
        assert evaluate_outcome(r, 89.0) == 1
        assert evaluate_outcome(r, 90.0) == 0

    def test_none_for_missing_floor(self) -> None:
        r = _rec(strike_type="greater", floor_strike=None)
        assert evaluate_outcome(r, 90.0) is None


# ---------------------------------------------------------------------------
# §9 kill-criteria (_eval_kill_criteria_raw) — Wilson-CI gated (CRITICAL-3)
# ---------------------------------------------------------------------------


class TestEvalKillCriteriaRaw:
    def test_low_p_insufficient_n(self) -> None:
        probs = [0.05] * 5
        outcomes = [1, 1, 1, 1, 1]
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        assert result["low_p_no_bin"]["verdict"] == "INSUFFICIENT-N"

    def test_low_p_fail_confidently_miscalibrated(self) -> None:
        # n=20, pred=0.07, realized=0.50 (10/20). pred+tol=0.12.
        # Wilson lower of 10/20 ≈ 0.299 > 0.12 → FAIL.
        n = 20
        probs = [0.07] * n
        outcomes = [1] * 10 + [0] * 10
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        low = result["low_p_no_bin"]
        assert low["verdict"] == "FAIL"
        assert low["wilson_lo"] > 0.12

    def test_low_p_inconclusive_when_ci_straddles(self) -> None:
        # n=20, pred=0.07, realized=0.20 (4/20). pred+tol=0.12.
        # Wilson CI of 4/20 ≈ (0.081, 0.436): lower < 0.12 < upper → INCONCLUSIVE.
        n = 20
        probs = [0.07] * n
        outcomes = [1] * 4 + [0] * 16
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        low = result["low_p_no_bin"]
        assert low["verdict"] == "INCONCLUSIVE"

    def test_low_p_pass_confidently_fine(self) -> None:
        # n=100, pred=0.07, realized=0.05 (5/100). pred+tol=0.12.
        # Wilson upper of 5/100 ≈ 0.112 ≤ 0.12 → PASS.
        n = 100
        probs = [0.07] * n
        outcomes = [1] * 5 + [0] * 95
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        low = result["low_p_no_bin"]
        assert low["verdict"] == "PASS"
        assert low["wilson_hi"] <= 0.12

    def test_high_p_insufficient_n(self) -> None:
        result = _eval_kill_criteria_raw([], [], [0.97] * 5, [1] * 5, [], [])
        assert result["high_p_bin"]["verdict"] == "INSUFFICIENT-N"

    def test_high_p_fail_confidently_below_floor(self) -> None:
        # n=20, realized=0.50 (10/20). Wilson upper ≈ 0.701 < 0.85 → FAIL.
        probs = [0.97] * 20
        outcomes = [1] * 10 + [0] * 10
        result = _eval_kill_criteria_raw([], [], probs, outcomes, probs, outcomes)
        assert result["high_p_bin"]["verdict"] == "FAIL"

    def test_high_p_pass_confidently_above_floor(self) -> None:
        # n=100, realized=0.95 (95/100). Wilson lower ≈ 0.887 ≥ 0.85 → PASS.
        probs = [0.97] * 100
        outcomes = [1] * 95 + [0] * 5
        result = _eval_kill_criteria_raw([], [], probs, outcomes, probs, outcomes)
        assert result["high_p_bin"]["verdict"] == "PASS"

    def test_high_p_inconclusive(self) -> None:
        # n=20, realized=0.90 (18/20). Wilson ≈ (0.699, 0.972): straddles 0.85.
        probs = [0.97] * 20
        outcomes = [1] * 18 + [0] * 2
        result = _eval_kill_criteria_raw([], [], probs, outcomes, probs, outcomes)
        assert result["high_p_bin"]["verdict"] == "INCONCLUSIVE"

    def test_half_width_reported(self) -> None:
        probs = [0.07] * 20
        outcomes = [1] * 2 + [0] * 18
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        assert result["low_p_no_bin"]["wilson_half_width"] is not None
        assert result["low_p_no_bin"]["wilson_half_width"] > 0

    def test_calibration_in_large_secondary_label(self) -> None:
        probs = [0.5] * 10
        outcomes = [1] * 5 + [0] * 5
        result = _eval_kill_criteria_raw(probs, outcomes, [], [], probs, outcomes)
        cal = result["calibration_in_large"]
        assert "secondary" in cal["note"].lower()
        assert cal["abs_gap"] is not None

    def test_empty_all_lists(self) -> None:
        result = _eval_kill_criteria_raw([], [], [], [], [], [])
        assert result["low_p_no_bin"]["verdict"] == "INSUFFICIENT-N"
        assert result["high_p_bin"]["verdict"] == "INSUFFICIENT-N"
        assert result["calibration_in_large"]["n"] == 0
        assert result["calibration_in_large"]["abs_gap"] is None


# ---------------------------------------------------------------------------
# Config: informational cells (MAJOR-2)
# ---------------------------------------------------------------------------


class TestLoadInformationalCells:
    def test_reads_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "config.yaml"
            cfg.write_text(
                "strategies:\n"
                "  intraday_asos_high:\n"
                "    informational_cells: [LAX/high, FOO/high]\n"
            )
            assert load_informational_cells(cfg) == ["LAX/high", "FOO/high"]

    def test_missing_file_falls_back(self) -> None:
        assert load_informational_cells(Path("/nonexistent/config.yaml")) == ["LAX/high"]

    def test_missing_key_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "config.yaml"
            cfg.write_text("strategies:\n  other: {}\n")
            assert load_informational_cells(cfg) == ["LAX/high"]


# ---------------------------------------------------------------------------
# Integration: build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    # --- per-bin table correctness ---

    def test_bin_table_correct_n_pred_real(self) -> None:
        recs, settlements = _low_p_event_recs(4, n_hits=1)
        report = _make_report(recs, settlements)
        bins = report["reliability_sharp_enabled"]
        assert len(bins) == 1
        row = bins[0]
        assert row["bin"] == "[0.05,0.10)"
        assert row["n"] == 4
        assert row["pred"] == pytest.approx(0.07)
        assert row["real"] == pytest.approx(0.25)

    # --- CRITICAL-1: dedup in the full pipeline ---

    def test_dedup_many_ticks_counted_once(self) -> None:
        """20 tick-rows of the SAME market collapse to a single settlement event."""
        ts0 = datetime(2026, 6, 10, 13, 0, tzinfo=UTC)
        recs = [
            _rec(
                market_ticker="KXHIGHMI-26JUN10-T88",
                expiration_date=date(2026, 6, 10),
                sigma_gain=0.9,
                posterior_p_raw=0.07,
                timestamp=ts0.replace(hour=13 + (i % 10)),
            )
            for i in range(20)
        ]
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        report = _make_report(recs, settlements)

        assert report["parse_reconciliation"]["intraday_asos_high_records"] == 20
        assert report["parse_reconciliation"]["deduped_settlement_events"] == 1
        assert report["summary"]["n_settlement_events"] == 1
        assert report["summary"]["n_settled_trade_eligible"] == 1
        # The single counted row is the deduped event, n=1 in the bin.
        assert sum(r["n"] for r in report["reliability_sharp_enabled"]) == 1

    # --- CRITICAL-2: σ-band stratification ---

    def test_wide_sigma_excluded_from_headline(self) -> None:
        """Wide-σ rows land in the diagnostic table, not the headline cohort."""
        sharp, sharp_settle = _low_p_event_recs(3, n_hits=0, sigma_gain=0.8)
        wide, wide_settle = _low_p_event_recs(3, n_hits=3, city="AUS", floor=95.0, sigma_gain=3.0)
        settlements = _merge_settlements(sharp_settle, wide_settle)
        report = _make_report(sharp + wide, settlements)

        assert report["summary"]["n_settled_sharp_pass"] == 3
        assert report["summary"]["n_settled_wide_pass"] == 3
        # Headline cohort = sharp only.
        assert sum(r["n"] for r in report["reliability_sharp_enabled"]) == 3
        assert sum(r["n"] for r in report["reliability_wide_diagnostic"]) == 3

    def test_wide_rows_do_not_dilute_headline_verdict(self) -> None:
        """Adding many wide-σ YES rows must NOT change the sharp-band verdict."""
        # Sharp cohort: 20 low-p NO bets that all LOSE (realized 0.50, FAIL).
        sharp, sharp_settle = _low_p_event_recs(20, n_hits=10, sigma_gain=0.8)
        rep_sharp_only = _make_report(sharp, sharp_settle)
        v_sharp_only = rep_sharp_only["kill_criteria"]["low_p_no_bin"]["verdict"]

        # Add wide-σ rows that realize cleanly — would mask the FAIL if pooled.
        wide, wide_settle = _low_p_event_recs(20, n_hits=0, city="AUS", floor=95.0, sigma_gain=4.0)
        rep_pooled = _make_report(sharp + wide, _merge_settlements(sharp_settle, wide_settle))
        v_pooled = rep_pooled["kill_criteria"]["low_p_no_bin"]["verdict"]

        assert v_sharp_only == v_pooled == "FAIL"

    # --- CRITICAL-3: Wilson verdict through full pipeline ---

    def test_low_p_bin_fail_through_pipeline(self) -> None:
        recs, settlements = _low_p_event_recs(20, n_hits=10)  # realized 0.50
        report = _make_report(recs, settlements)
        low = report["kill_criteria"]["low_p_no_bin"]
        assert low["verdict"] == "FAIL"
        assert low["mean_real"] == pytest.approx(0.50)

    def test_low_p_bin_insufficient_n(self) -> None:
        recs, settlements = _low_p_event_recs(5, n_hits=5)
        report = _make_report(recs, settlements)
        assert report["kill_criteria"]["low_p_no_bin"]["verdict"] == "INSUFFICIENT-N"

    # --- guard-dropped rows excluded from trade-eligible ---

    def test_guard_dropped_excluded(self) -> None:
        pass_rec = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            expiration_date=date(2026, 6, 10),
            guard_outcome="pass",
        )
        dropped = _rec(
            market_ticker="KXHIGHMI-26JUN11-T88",
            expiration_date=date(2026, 6, 11),
            guard_outcome="boundary",
        )
        settlements = {
            "2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)],
            "2026-06-11": [_settlement_entry("MIA", "2026-06-11", 87.0)],
        }
        report = _make_report([pass_rec, dropped], settlements)
        assert report["summary"]["n_settled_trade_eligible"] == 1
        assert report["summary"]["n_settled_all_guard"] == 2

    # --- pending predictions excluded ---

    def test_pending_excluded(self) -> None:
        settled = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            expiration_date=date(2026, 6, 10),
            guard_outcome="pass",
        )
        pending = _rec(
            market_ticker="KXHIGHMI-26JUN30-T88",
            expiration_date=date(2026, 6, 30),
            guard_outcome="pass",
        )
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        report = _make_report([settled, pending], settlements)
        assert report["summary"]["n_settled_trade_eligible"] == 1
        assert report["summary"]["n_pending"] == 1

    # --- other strategies filtered out ---

    def test_other_strategy_rows_excluded(self) -> None:
        asos = _rec(market_ticker="KXHIGHMI-26JUN10-T88", expiration_date=date(2026, 6, 10))
        other = _rec(
            market_ticker="KXHIGHMI-26JUN10-T89",
            expiration_date=date(2026, 6, 10),
            strategy="weather_edge",
        )
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        report = _make_report([asos, other], settlements)
        assert report["summary"]["n_total_strategy_rows"] == 1

    # --- MAJOR-2: informational LAX excluded from headline ---

    def test_lax_excluded_from_headline_present_in_diagnostic(self) -> None:
        mia, mia_settle = _low_p_event_recs(20, n_hits=10, city="MIA")  # FAIL
        lax, lax_settle = _low_p_event_recs(20, n_hits=0, city="LAX", floor=88.0)  # clean
        report = _make_report(mia + lax, _merge_settlements(mia_settle, lax_settle))

        # Headline excludes LAX → still FAIL (driven by MIA).
        assert report["kill_criteria"]["low_p_no_bin"]["verdict"] == "FAIL"
        assert report["summary"]["n_settled_sharp_enabled"] == 20
        assert report["summary"]["n_settled_sharp_informational"] == 20
        # LAX present as its own diagnostic table.
        assert sum(r["n"] for r in report["reliability_informational"]) == 20
        # LAX present in per-cell, flagged informational.
        assert report["by_cell"]["LAX/high"]["informational"] is True
        assert report["by_cell"]["MIA/high"]["informational"] is False

    def test_informational_override_arg(self) -> None:
        mia, mia_settle = _low_p_event_recs(3, n_hits=0, city="MIA")
        report = _make_report(mia, mia_settle, informational_cells=["MIA/high"])
        assert report["by_cell"]["MIA/high"]["informational"] is True
        assert report["summary"]["n_settled_sharp_enabled"] == 0
        assert report["summary"]["n_settled_sharp_informational"] == 3

    # --- per-cell verdicts ---

    def test_per_cell_verdicts_present(self) -> None:
        mia, mia_settle = _low_p_event_recs(20, n_hits=10, city="MIA")  # FAIL
        aus, aus_settle = _low_p_event_recs(5, n_hits=0, city="AUS", floor=95.0)
        report = _make_report(mia + aus, _merge_settlements(mia_settle, aus_settle))
        by_cell = report["by_cell"]
        assert "MIA/high" in by_cell
        assert "AUS/high" in by_cell
        assert by_cell["MIA/high"]["kill_criteria"]["low_p_no_bin"]["verdict"] == "FAIL"
        assert by_cell["AUS/high"]["kill_criteria"]["low_p_no_bin"]["verdict"] == "INSUFFICIENT-N"

    # --- MINOR: parse-loss reconciliation ---

    def test_parse_loss_counted(self) -> None:
        good = _rec(market_ticker="KXHIGHMI-26JUN10-T88", expiration_date=date(2026, 6, 10))
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "predictions.jsonl"
            snaps_dir = tmp / "snapshots"
            good_line = json.dumps(_serialize(good), default=str, separators=(",", ":"))
            # one good, one non-JSON garbage, one JSON-but-missing-required-key
            pred_path.write_text(
                good_line + "\n" + "{not json]\n" + '{"strategy":"intraday_asos_high"}\n'
            )
            _write_snaps(snaps_dir, settlements)
            report = build_report(pred_path, snaps_dir)

        pr = report["parse_reconciliation"]
        assert pr["total_raw_lines"] == 3
        assert pr["parsed_records"] == 1
        assert pr["rows_dropped"] == 2
        assert pr["intraday_asos_high_records"] == 1

    # --- zero corpus ---

    def test_zero_corpus_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "predictions.jsonl"
            pred_path.write_text("")
            snaps_dir = tmp / "snapshots"
            snaps_dir.mkdir()
            report = build_report(pred_path, snaps_dir)

        assert report["summary"]["n_total_strategy_rows"] == 0
        assert report["summary"]["n_settlement_events"] == 0
        assert report["summary"]["n_settled_trade_eligible"] == 0
        assert report["reliability_sharp_enabled"] == []
        kc = report["kill_criteria"]
        assert kc["low_p_no_bin"]["verdict"] == "INSUFFICIENT-N"
        assert kc["high_p_bin"]["verdict"] == "INSUFFICIENT-N"

    def test_zero_corpus_render_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "predictions.jsonl"
            pred_path.write_text("")
            snaps_dir = tmp / "snapshots"
            snaps_dir.mkdir()
            report = build_report(pred_path, snaps_dir)
        rendered = render_report(report)
        # Stage 0a: the headline cohort is frame-v2; the zero-corpus message now
        # references the binding frame explicitly.
        assert "0 settled frame-v2 (LST) predictions" in rendered

    def test_nonempty_render_no_crash(self) -> None:
        recs, settlements = _low_p_event_recs(20, n_hits=10)
        report = _make_report(recs, settlements)
        rendered = render_report(report)
        assert "§9 KILL CRITERIA" in rendered
        assert "PARSE RECONCILIATION" in rendered
        assert "PER CELL" in rendered

    # --- None posterior_p_raw skipped in bins but still a settled event ---

    def test_none_posterior_excluded_from_bins(self) -> None:
        rec = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            expiration_date=date(2026, 6, 10),
            posterior_p_raw=None,
            guard_outcome="pass",
        )
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        report = _make_report([rec], settlements)
        assert report["summary"]["n_settled_trade_eligible"] == 1
        assert report["reliability_sharp_enabled"] == []

    # --- JSON export ---

    def test_json_export(self) -> None:
        recs, settlements = _low_p_event_recs(2, n_hits=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pred_path = tmp / "predictions.jsonl"
            snaps_dir = tmp / "snapshots"
            out_json = tmp / "report.json"
            _write_preds(pred_path, recs)
            _write_snaps(snaps_dir, settlements)
            report = build_report(pred_path, snaps_dir)
            out_json.write_text(json.dumps(report, indent=2))
            loaded = json.loads(out_json.read_text())

        for key in (
            "summary",
            "reliability_sharp_enabled",
            "reliability_wide_diagnostic",
            "reliability_informational",
            "kill_criteria",
            "by_cell",
            "parse_reconciliation",
            "config",
            "generated_at",
            "frame_partition",
            "frame_v1_diagnostic",
        ):
            assert key in loaded


class TestFrameCut:
    """Stage 0a — frame_version partition: frames are never pooled."""

    def test_frame_v1_rows_routed_to_diagnostic_not_headline(self) -> None:
        """A legacy frame-v1 (civil) row must NOT enter the binding headline cohort."""
        rec = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            expiration_date=date(2026, 6, 10),
            posterior_p_raw=0.07,
            guard_outcome="pass",
            frame_version=1,  # legacy civil frame
        )
        settlements = {"2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)]}
        report = _make_report([rec], settlements)

        # Headline (frame-v2) cohort sees nothing.
        assert report["summary"]["n_settled_trade_eligible"] == 0
        assert report["frame_partition"]["frame_v2_rows"] == 0
        assert report["frame_partition"]["frame_v1_rows"] == 1
        # The diagnostic cohort carries it.
        assert report["frame_v1_diagnostic"]["summary"]["n_settled_trade_eligible"] == 1

    def test_frames_not_pooled(self) -> None:
        """v1 and v2 rows for distinct events land in their respective cohorts."""
        v2 = _rec(
            market_ticker="KXHIGHMI-26JUN10-T88",
            expiration_date=date(2026, 6, 10),
            posterior_p_raw=0.07,
            frame_version=2,
        )
        v1 = _rec(
            market_ticker="KXHIGHMI-26JUN11-T88",
            expiration_date=date(2026, 6, 11),
            posterior_p_raw=0.07,
            frame_version=1,
        )
        settlements = {
            "2026-06-10": [_settlement_entry("MIA", "2026-06-10", 87.0)],
            "2026-06-11": [_settlement_entry("MIA", "2026-06-11", 87.0)],
        }
        report = _make_report([v2, v1], settlements)
        assert report["frame_partition"]["frame_v2_rows"] == 1
        assert report["frame_partition"]["frame_v1_rows"] == 1
        assert report["summary"]["n_settled_trade_eligible"] == 1
        assert report["frame_v1_diagnostic"]["summary"]["n_settled_trade_eligible"] == 1

    def test_render_labels_frame_v1_as_diagnostic(self) -> None:
        recs, settlements = _low_p_event_recs(2, n_hits=0)
        # Force them to the legacy frame.
        recs = [_rec_to_v1(r) for r in recs]
        report = _make_report(recs, settlements)
        rendered = render_report(report)
        assert "FRAME-v1 DIAGNOSTIC" in rendered
        assert "NOT DECISION-GRADE" in rendered


def _rec_to_v1(rec: PredictionRecord):
    """Return a copy of *rec* with frame_version=1 (helper for frame-cut tests)."""
    import dataclasses

    return dataclasses.replace(rec, frame_version=1)
