"""Per-cell EMOS live-calibrator dispatcher tests (Stage A cutover, 2026-05-11).

The dispatcher lives inside ``evaluate_market_pure``: when a market's
``city/var`` cell is in ``emos_enabled_cells``, ``model_prob`` is set
to ``emos_prob_raw`` (the EMOS path); otherwise the isotonic calibrator
runs unchanged. Audit rows record ``live_calibrator_kind`` so the
monthly per-cell re-evaluation can disambiguate.

Fixture-design contract (H3 fix): every test that exercises the
dispatcher resolution MUST land a non-None ``_MarketCandidate`` so
the candidate-level assertions actually run. The fixtures here use
temps that straddle the strike (raw probability in (0.05, 0.95)) and
mid-range market prices so the gauntlet of upstream gates
(EXCLUDED_CITY_VAR, RAW_PROB_GATE, PLATEAU_BLOCK, STRIKE_DISTANCE,
SPREAD, YES_FLOOR, NO_FLOOR, EDGE_BELOW_THRESHOLD, KELLY_ZERO,
CAL_FLOOR_BLOCK, CALIBRATOR_DISAGREEMENT) does not preempt the
dispatcher. Tests deliberately exercising one of those gates (H1, H4)
assert on the rejection row, not on a candidate.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from python.analytics.audit_log import AuditSinks
from python.common import DailyVariable, Direction
from python.data.emos import EmosTable
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    _EvalConfig,
    evaluate_market_pure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_eval_cfg(**overrides: Any) -> _EvalConfig:
    base: dict[str, Any] = dict(
        max_spread=0.15,
        spread_skill_sensitivity=0.0,
        edge_threshold=0.001,
        max_edge=0.99,
        kelly_fraction=0.25,
        max_pct=0.20,
        max_take_pct=0.50,
        depth_check_enabled=False,
        min_raw_prob_for_between=0.0,
        min_distance_from_strike_std=0.0,
        max_fill_slippage=None,
        block_raw_prob_range=None,
        max_model_market_gap_near_term=None,
    )
    base.update(overrides)
    return _EvalConfig(**base)


def _make_parsed(
    *,
    city: str,
    var: DailyVariable,
    strike_type: Direction = Direction.ABOVE,
    floor_strike: float | None = 70.0,
    cap_strike: float | None = None,
) -> ParsedMarket:
    fd = datetime.now(UTC).date() + timedelta(days=1)
    return ParsedMarket(
        ticker=f"KX-{city}-{fd.isoformat()}-T{floor_strike or 0:.0f}",
        city=city,
        forecast_date=fd,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        daily_variable=var,
    )


def _make_scan_candidate(
    parsed: ParsedMarket,
    *,
    yes_bid: float = 0.40,
    yes_ask: float = 0.50,
) -> ScanCandidate:
    """Mid-range pricing by default (mid=0.45). Override for floor / disagreement tests."""
    pi = MarketPriceInfo(
        midpoint=(yes_bid + yes_ask) / 2.0,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        spread=yes_ask - yes_bid,
        yes_bid_size=200,
        yes_ask_size=200,
        volume=100,
        open_interest=50,
    )
    market: dict[str, Any] = {
        "ticker": parsed.ticker,
        "event_ticker": "EVT",
        "strike_type": parsed.strike_type.value,
        "floor_strike": parsed.floor_strike,
        "cap_strike": parsed.cap_strike,
        "yes_bid_dollars": f"{yes_bid:.2f}",
        "yes_ask_dollars": f"{yes_ask:.2f}",
        "volume_fp": "100.0",
        "open_interest_fp": "50.0",
        "close_time": "2099-12-31T23:59:59Z",
    }
    return ScanCandidate(market=market, parsed=parsed, price_info=pi)


def _make_forecast(city: str, fc_date: date, temps: list[float]) -> EnsembleForecast:
    return EnsembleForecast(city=city, forecast_date=fc_date, temperatures=temps)


def _make_emos_table_with_cells(
    tmp_path: Path,
    cells: list[str],
    *,
    coeffs: dict[str, Any] | None = None,
) -> EmosTable:
    """Build a hermetic EMOS table with the given cells.

    Default coefficients ``(a=10, b=0.95, c=1, d=0.2)`` produce a meaningful
    upward μ-shift so EMOS visibly diverges from raw fraction-based isotonic.
    Pass ``coeffs`` to override (e.g., to drive the EMOS output below the
    ``min_calibrated_prob`` floor for H4 coverage).
    """
    default = {
        "a": 10.0,
        "b": 0.95,
        "c": 1.0,
        "d": 0.2,
        "n_train": 50,
        "leads_pooled": [0, 1, 2],
        "low_n": False,
        "ci_warnings": [],
        "ens_std_p99": 6.5,
    }
    cell_dict: dict[str, Any] = {}
    for cell in cells:
        cell_dict[cell] = dict(default, **(coeffs or {}))
    payload = {
        "fit_date": "2026-05-11",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": cell_dict,
    }
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)
    assert table.is_loaded
    return table


class _FixedCalibrator:
    """Stub calibrator returning a known constant."""

    kind = "isotonic"
    is_loaded = True
    fallback_reason: str | None = None
    artifact_hash = "iso-test"

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        self.calls += 1
        return self.value


class _CapturingEvalLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit_eval(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


def _make_sinks_with_capture() -> tuple[AuditSinks, _CapturingEvalLog]:
    cap = _CapturingEvalLog()
    sinks = AuditSinks(eval_log=cap, candidate_log=None, funnel_log=None)  # type: ignore[arg-type]
    return sinks, cap


def _build_eval_kwargs(parsed: ParsedMarket, temps: list[float]) -> dict[str, Any]:
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 2.0}
    raw_meds = {fkey: sum(temps) / len(temps)}
    return dict(
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_ensemble_stds=raw_stds,
        raw_ensemble_medians=raw_meds,
        raw_member_sink=raw_member_sink,
        weights={"gfs_seamless": 1.0},
        weather_client=MagicMock(compute_model_probability=MagicMock(return_value=0.5)),
        bankroll=10_000.0,
    )


# Temperature set that produces raw_above = 2/5 = 0.4 at strike=85, well within
# the [0.05, 0.95] RAW_PROB_GATE window. ens_mean=85.0 (matches strike), so the
# strike-proximity gate is permissive even on default config (k=0).
_STRADDLE_85_TEMPS = [83.0, 84.0, 85.0, 86.0, 87.0]


# ---------------------------------------------------------------------------
# Tests — happy-path dispatcher behavior (H3 fixture fixes)
# ---------------------------------------------------------------------------


def test_emos_enabled_cell_uses_emos_raw(tmp_path: Path) -> None:
    """A market in an emos_enabled_cells entry sees model_prob == emos_prob_raw.

    H3 sanity check: ``candidate is not None`` so the dispatcher branch ran.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    # Wide spread to keep the disagreement gate happy: EMOS produces ~0.65 for
    # MIA/high at this strike (a=10 pushes μ up), and we want the mid to be on
    # the same side as both raw and EMOS predictions — so mid <~ 0.4.
    # mid=0.25; spread keeps fee << edge so EDGE_BELOW_THRESHOLD does not fire.
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.55)  # deliberately diverges from EMOS

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    # H3 sanity: dispatcher branch was exercised, candidate produced.
    assert candidate is not None, "fixture must produce a candidate, else H3 regresses"
    # EMOS overrode the isotonic value.
    assert candidate.emos_prob_raw is not None
    assert candidate.model_prob == candidate.emos_prob_raw
    assert candidate.model_prob != 0.55
    assert candidate.live_calibrator_kind == "emos_raw"
    # evaluate_market_pure emits rows only on rejections; happy-path PASSED
    # rows are emitted from generate_signals (covered by H2 end-to-end test).
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_emos_disabled_cell_uses_isotonic(tmp_path: Path) -> None:
    """A market NOT in emos_enabled_cells continues to use isotonic.

    H3 sanity check: ``candidate is not None``.
    """
    parsed = _make_parsed(
        city="AUS",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    # mid=0.25; spread keeps fee << edge so EDGE_BELOW_THRESHOLD does not fire.
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)

    table = _make_emos_table_with_cells(tmp_path, ["AUS/high", "MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    # iso=0.55 lands on the same side of mid (0.35) as raw=0.4, no disagreement.
    iso = _FixedCalibrator(0.55)

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},  # AUS NOT in list
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.model_prob == 0.55, "AUS/high not in allowlist → iso path"
    assert candidate.live_calibrator_kind == "isotonic"
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_listed_cell_with_no_emos_table_falls_back_to_isotonic(tmp_path: Path) -> None:
    """Cell is in the allowlist but emos_table is None → isotonic fallback.

    H3 sanity check: ``candidate is not None``.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    # mid=0.25; spread keeps fee << edge so EDGE_BELOW_THRESHOLD does not fire.
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)

    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.42)  # same side as raw=0.4 vs mid=0.35

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=None,  # NO EMOS table
        raw_member_sink=None,
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},  # listed but unavailable
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.model_prob == 0.42, "fell back to iso"
    assert candidate.live_calibrator_kind == "isotonic_fallback"
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_empty_allowlist_uses_isotonic_unchanged(tmp_path: Path) -> None:
    """Empty emos_enabled_cells → today's behavior, EMOS stays shadow-only.

    H3 sanity check: ``candidate is not None``.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    # mid=0.25; spread keeps fee << edge so EDGE_BELOW_THRESHOLD does not fire.
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.55)  # same side as raw=0.4 vs mid=0.35

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells=None,  # absent / empty
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.model_prob == 0.55, "iso path"
    # EMOS still computed shadow-side (table loaded, in-horizon).
    assert candidate.emos_prob_raw is not None
    assert candidate.live_calibrator_kind == "isotonic"
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_listed_cell_with_failed_emos_computation_falls_back(tmp_path: Path) -> None:
    """If emos_prob_raw failed to compute (lead out of range), use iso.

    This test deliberately uses a far-future forecast_date that trips the
    ``lead_out_of_range`` warning in the EMOS block, so ``emos_prob_raw``
    stays None. Most upstream gates are unaffected by lead, but the same-day
    gate is not exercised here — the test asserts on the audit row directly
    so it works whether or not a candidate ultimately passes.
    """
    far_future = datetime.now(UTC).date() + timedelta(days=10)
    parsed = ParsedMarket(
        ticker="KX-MIA-FAR",
        city="MIA",
        forecast_date=far_future,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
        cap_strike=None,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    fc = _make_forecast(parsed.city, parsed.forecast_date, _STRADDLE_85_TEMPS)
    forecast_cache = {fkey: {"gfs_seamless": fc}}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, _STRADDLE_85_TEMPS)}
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 2.0}
    raw_meds = {fkey: 85.0}

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.42)  # above mid=0.25, raw_above=0.4 — same side

    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=iso,
        weights={"gfs_seamless": 1.0},
        weather_client=MagicMock(compute_model_probability=MagicMock(return_value=0.5)),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    # emos_prob_raw is None (lead > 2), so dispatcher falls back.
    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.emos_prob_raw is None
    assert candidate.live_calibrator_kind == "isotonic_fallback"
    assert candidate.model_prob == 0.42
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_low_temp_var_uses_low_cell_label(tmp_path: Path) -> None:
    """NYC/low entry fires on TEMP_MIN markets.

    H3 sanity check: ``candidate is not None``.
    """
    parsed = _make_parsed(
        city="NYC",
        var=DailyVariable.TEMP_MIN,
        strike_type=Direction.ABOVE,
        floor_strike=52.0,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.30, yes_ask=0.40)
    # Straddle 52: raw_above = 2/5 = 0.4
    temps = [50.0, 51.0, 52.0, 53.0, 54.0]
    base_kwargs = _build_eval_kwargs(parsed, temps)

    table = _make_emos_table_with_cells(tmp_path, ["NYC/low", "NYC/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.55)

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"NYC/low"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.model_prob == candidate.emos_prob_raw
    assert candidate.live_calibrator_kind == "emos_raw"
    assert cap.rows == [], "happy path should not emit a rejection row"


def test_high_temp_var_not_promoted_when_only_low_listed(tmp_path: Path) -> None:
    """NYC/high market is NOT promoted when only NYC/low is in the allowlist."""
    parsed = _make_parsed(
        city="NYC",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=72.0,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.30, yes_ask=0.40)
    temps = [70.0, 71.0, 72.0, 73.0, 74.0]
    base_kwargs = _build_eval_kwargs(parsed, temps)

    table = _make_emos_table_with_cells(tmp_path, ["NYC/low", "NYC/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.55)

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"NYC/low"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.model_prob == 0.55, "iso path (cell not promoted)"
    assert candidate.live_calibrator_kind == "isotonic"
    assert cap.rows == [], "happy path should not emit a rejection row"


# ---------------------------------------------------------------------------
# H1 — CALIBRATOR_DISAGREEMENT gate is skipped on emos_raw rows
# ---------------------------------------------------------------------------


def test_calibrator_disagreement_skipped_for_emos_raw_cell(tmp_path: Path) -> None:
    """EMOS-promoted cell with raw vs EMOS on opposite sides of mid still passes.

    Reproduces the empirical finding from the code review:
    - MIA/high above strike=85, ens_mean=82°F (all members below strike) →
      raw_above ≈ 0.0, EMOS_above ≈ 0.65 (a=10 pushes μ up to ~87.9).
    - Market mid = 0.30 lies between raw=0.0 and EMOS=0.65.
    - Without the H1 fix, ``CALIBRATOR_DISAGREEMENT`` fires and blocks the
      trade — exactly the trades the +58.6% BSS-on-above gain depends on.
    - With H1 fix, the gate is skipped because ``live_calibrator_kind ==
      "emos_raw"``; the candidate passes downstream.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    # All members below strike → raw_above = 0.0 → would trip RAW_PROB_GATE.
    # To preserve the empirical setup but still reach the disagreement gate,
    # we use temps that produce raw_above ≈ 0.2 — clearly on one side of mid
    # while EMOS lands on the other.
    temps = [80.0, 81.0, 82.0, 83.0, 85.5]  # 1/5 > 85 → raw_above = 0.2
    base_kwargs = _build_eval_kwargs(parsed, temps)

    # Market mid = 0.45 (between raw=0.2 and EMOS≈0.65).
    sc = _make_scan_candidate(parsed, yes_bid=0.40, yes_ask=0.50)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    # iso=0.20 lands on the same side of mid as raw — so isotonic path WOULD
    # pass this gate; we're proving EMOS-specific behavior independent of iso.
    iso = _FixedCalibrator(0.20)

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    # Candidate must exist — H1 fix lets it through the disagreement gate.
    assert candidate is not None, (
        "EMOS-promoted candidate must not be blocked by CALIBRATOR_DISAGREEMENT "
        "when raw and EMOS land on opposite sides of mid"
    )
    assert candidate.live_calibrator_kind == "emos_raw"
    assert candidate.model_prob == candidate.emos_prob_raw
    # No CALIBRATOR_DISAGREEMENT row was emitted (otherwise candidate would be None).
    disagreement_rows = [r for r in cap.rows if r.get("reason") == "CALIBRATOR_DISAGREEMENT"]
    assert not disagreement_rows, (
        f"Disagreement gate should not fire on emos_raw row, got: {disagreement_rows}"
    )


def test_calibrator_disagreement_still_fires_on_isotonic_path(tmp_path: Path) -> None:
    """Non-EMOS-promoted cells: disagreement gate must still fire.

    The H1 fix is per-cell-aware — isotonic cells retain the original
    plateau-pathology protection that motivated the gate (KXLOWTCHI-T53 loss
    postmortem).
    """
    # AUS/high is NOT in any allowlist. raw=0.2 (one member above 85),
    # iso=0.70, mid=0.45 → raw and iso on opposite sides → gate fires.
    parsed = _make_parsed(
        city="AUS",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    temps = [80.0, 81.0, 82.0, 83.0, 85.5]  # raw_above = 0.2
    base_kwargs = _build_eval_kwargs(parsed, temps)

    sc = _make_scan_candidate(parsed, yes_bid=0.40, yes_ask=0.50)
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.70)  # opposite side of mid=0.45 from raw=0.2

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=None,
        raw_member_sink=None,
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},  # AUS not in allowlist
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is None, "disagreement gate must block this trade"
    disagreement_rows = [r for r in cap.rows if r.get("reason") == "CALIBRATOR_DISAGREEMENT"]
    assert len(disagreement_rows) == 1, "exactly one disagreement-block row expected"
    row = disagreement_rows[0]
    assert row.get("live_calibrator_kind") == "isotonic"


def test_calibrator_disagreement_fires_on_isotonic_fallback(tmp_path: Path) -> None:
    """When EMOS fails to compute on a listed cell, the gate still applies.

    Critical asymmetry: ``live_calibrator_kind == "isotonic_fallback"`` means
    the isotonic plateau is back in play. The gate must fire normally.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    temps = [80.0, 81.0, 82.0, 83.0, 85.5]
    base_kwargs = _build_eval_kwargs(parsed, temps)
    sc = _make_scan_candidate(parsed, yes_bid=0.40, yes_ask=0.50)
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.70)  # opposite side of mid=0.45 from raw=0.2

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=None,  # forces isotonic_fallback
        raw_member_sink=None,
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is None, "isotonic_fallback must respect disagreement gate"
    disagreement_rows = [r for r in cap.rows if r.get("reason") == "CALIBRATOR_DISAGREEMENT"]
    assert len(disagreement_rows) == 1
    assert disagreement_rows[0].get("live_calibrator_kind") == "isotonic_fallback"


# ---------------------------------------------------------------------------
# H4 — min_calibrated_prob floor applies to EMOS path
# ---------------------------------------------------------------------------


def test_emos_below_floor_blocked(tmp_path: Path) -> None:
    """EMOS-promoted cell where emos_prob_raw < min_calibrated_prob is blocked.

    Spec §"Goals 5" requires the floor to apply to EMOS output too. The
    floor's asymmetric loss economics (`cal_floor_block_2026-05-07.md`) are
    calibrator-agnostic — the bot's win rate cannot overcome small-win /
    big-loss trades at the bottom of the calibrated-prob distribution
    regardless of which calibrator produced the prediction.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=100.0,  # very high — almost no members above
    )
    # All members well below strike → raw_above = 0.0.
    # EMOS: a=10 pushes μ up to 87.9, σ ≈ sqrt(c + d*σ²) ≈ sqrt(1.8) ≈ 1.34
    # → P(T > 100) ≈ Φ((87.9-100)/1.34) ≈ very small (~10⁻¹⁸). Well below 0.10.
    temps = [80.0, 81.0, 82.0, 83.0, 84.0]
    base_kwargs = _build_eval_kwargs(parsed, temps)

    sc = _make_scan_candidate(parsed, yes_bid=0.05, yes_ask=0.10)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    # Floor=0.10 is the production default; assert the EMOS path respects it.
    cfg = _make_eval_cfg(edge_threshold=0.001, min_calibrated_prob=0.10)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.50)  # iso would pass the floor, isolates EMOS

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is None, "EMOS-path candidate below floor must be blocked"
    floor_rows = [r for r in cap.rows if r.get("reason") == "CAL_FLOOR_BLOCK"]
    assert len(floor_rows) == 1, "exactly one CAL_FLOOR_BLOCK row expected"
    row = floor_rows[0]
    # The blocked row records the EMOS dispatch (so monthly re-eval can filter).
    assert row.get("live_calibrator_kind") == "emos_raw"
    # The calibrated_prob recorded is the EMOS output, not the isotonic value.
    assert row["calibrated_prob"] < 0.10
    assert row["emos_prob_raw"] is not None
    assert row["calibrated_prob"] == row["emos_prob_raw"]


# ---------------------------------------------------------------------------
# H2 — live_calibrator_kind threaded through PASSED + candidate audit rows
# ---------------------------------------------------------------------------


def test_eval_audit_row_passed_carries_live_calibrator_kind() -> None:
    """The _eval_audit_row builder accepts and emits live_calibrator_kind.

    Mirrors the PASSED-row emission contract in
    ``WeatherEdgeStrategy.generate_signals`` (weather_edge.py:~1004). Prior
    implementation set ``live_calibrator_kind`` only on REJECTED rows via
    the ``_reject`` closure; the PASSED row must also carry the field so
    the monthly per-cell re-eval can filter on it.
    """
    from python.analytics.audit_log import RejectionReason
    from python.strategies.weather_audit import _eval_audit_row

    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)

    row = _eval_audit_row(
        scan_id="scan-1",
        sc=sc,
        price_info=sc.price_info,
        reason=RejectionReason.PASSED,
        raw_ens_prob=0.4,
        calibrated_prob=0.65,
        emos_prob_raw=0.65,
        live_calibrator_kind="emos_raw",
    )
    assert row["reason"] == "PASSED"
    assert row["live_calibrator_kind"] == "emos_raw"
    assert row["calibrated_prob"] == 0.65
    assert row["emos_prob_raw"] == 0.65


def test_candidate_audit_row_carries_live_calibrator_kind(tmp_path: Path) -> None:
    """``_build_candidate_audit_row`` emits ``live_calibrator_kind``.

    The candidate_audit stream is the deepest-signal forensic record; the
    monthly re-eval joins on it. Verifying via a direct call rather than
    end-to-end ``generate_signals`` keeps the test deterministic and
    focused on the H2 plumbing fix.
    """
    from python.analytics.audit_log import RejectionReason
    from python.analytics.sizing import SizingResult
    from python.common import Side
    from python.strategies.weather_edge import WeatherEdgeStrategy

    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.30)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, _cap = _make_sinks_with_capture()

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=_FixedCalibrator(0.55),
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )
    assert candidate is not None
    assert candidate.live_calibrator_kind == "emos_raw"

    # Build a minimal WeatherEdgeStrategy to access _build_candidate_audit_row.
    # The method only reads ``candidate.*`` and ``self._config``; no I/O.
    strat = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
    strat._config = {"min_book_depth": 5}

    sizing = SizingResult(
        side=Side.YES,
        dollars=10.0,
        kelly_raw=0.05,
        kelly_fraction_used=0.15,
        capped=False,
    )
    row = strat._build_candidate_audit_row(
        candidate=candidate,
        cfg=cfg,
        sizing=sizing,
        contracts=10,
        depth_capped=False,
        depth_result=None,
        opp_view=None,
        bracket_metric=None,
        fill_edge=None,
        fill_fee=None,
        reason=RejectionReason.PASSED,
        scan_id="scan-1",
        scan_ts=None,
    )
    assert row["reason"] == "PASSED"
    assert row["live_calibrator_kind"] == "emos_raw", (
        "_build_candidate_audit_row must emit live_calibrator_kind on the candidate row"
    )


def test_emos_above_floor_passes(tmp_path: Path) -> None:
    """EMOS-promoted cell where emos_prob_raw >= floor proceeds downstream.

    Complement of ``test_emos_below_floor_blocked``: same floor, EMOS output
    above it, candidate emerges from the gate.
    """
    parsed = _make_parsed(
        city="MIA",
        var=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
    )
    base_kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS)
    sc = _make_scan_candidate(parsed, yes_bid=0.30, yes_ask=0.40)

    table = _make_emos_table_with_cells(tmp_path, ["MIA/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001, min_calibrated_prob=0.10)
    sinks, cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.50)

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert candidate is not None
    assert candidate.live_calibrator_kind == "emos_raw"
    assert candidate.emos_prob_raw is not None
    assert candidate.emos_prob_raw >= 0.10
    # No CAL_FLOOR_BLOCK row fired.
    floor_rows = [r for r in cap.rows if r.get("reason") == "CAL_FLOOR_BLOCK"]
    assert not floor_rows


# ---------------------------------------------------------------------------
# E2E — a PASSED eval_audit row from an EMOS cell carries live_calibrator_kind
# ---------------------------------------------------------------------------


def test_generate_signals_passed_row_carries_live_calibrator_kind(tmp_path: Path) -> None:
    """End-to-end through the dispatcher: an EMOS-promoted cell produces a
    candidate with ``live_calibrator_kind == "emos_raw"``, and the PASSED
    eval_audit row built for it (via the same ``_eval_audit_row`` builder
    ``WeatherEdgeStrategy.generate_signals`` uses for PASSED rows) carries that
    field through to the audit stream.

    Mirrors the EMOS-raw assertion in
    ``tests/test_intraday_emos_high_shadow_e2e.py``
    (``md["live_calibrator_kind"] == "emos_raw"``). Uses the NYC/low fixture
    EMOS cell (mirroring the proven-safe ``test_low_temp_var_uses_low_cell_label``
    fixture); the dispatcher branch is cell-agnostic, and the only *live* EMOS
    cell per ``config.yaml`` ``emos_enabled_cells`` is LAX/low.
    """
    from python.analytics.audit_log import RejectionReason
    from python.strategies.weather_audit import _eval_audit_row

    parsed = _make_parsed(
        city="NYC",
        var=DailyVariable.TEMP_MIN,
        strike_type=Direction.ABOVE,
        floor_strike=52.0,
    )
    # Straddle 52 → raw_above = 2/5 = 0.4; mid-range pricing keeps the gate
    # gauntlet from preempting the dispatcher.
    temps = [50.0, 51.0, 52.0, 53.0, 54.0]
    sc = _make_scan_candidate(parsed, yes_bid=0.30, yes_ask=0.40)
    base_kwargs = _build_eval_kwargs(parsed, temps)

    table = _make_emos_table_with_cells(tmp_path, ["NYC/low", "NYC/high"])
    cfg = _make_eval_cfg(edge_threshold=0.001)
    sinks, _cap = _make_sinks_with_capture()
    iso = _FixedCalibrator(0.55)  # deliberately diverges from EMOS

    candidate = evaluate_market_pure(
        sc,
        base_kwargs["forecast_cache"],
        base_kwargs["bias_corrections"],
        base_kwargs["raw_ensemble_stds"],
        base_kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=iso,
        weights=base_kwargs["weights"],
        weather_client=base_kwargs["weather_client"],
        bankroll=base_kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=base_kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"NYC/low"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    # The dispatcher took the EMOS branch for the promoted cell.
    assert candidate is not None, "fixture must produce a candidate"
    assert candidate.emos_prob_raw is not None
    assert candidate.model_prob == candidate.emos_prob_raw
    assert candidate.live_calibrator_kind == "emos_raw"

    # Build the PASSED eval_audit row exactly as generate_signals does and
    # confirm live_calibrator_kind survives onto the emitted row.
    row = _eval_audit_row(
        scan_id="scan-e2e",
        sc=sc,
        price_info=sc.price_info,
        reason=RejectionReason.PASSED,
        raw_ens_prob=candidate.raw_ensemble_prob,
        calibrated_prob=candidate.model_prob,
        emos_prob_raw=candidate.emos_prob_raw,
        live_calibrator_kind=candidate.live_calibrator_kind,
    )
    assert row["reason"] == "PASSED"
    assert row["live_calibrator_kind"] == "emos_raw"
    assert row["calibrated_prob"] == candidate.emos_prob_raw
