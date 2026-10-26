"""Runtime σ_emos / bracket_width gate (Tier 2 A, 2026-05-19).

Catches the LAX/high-style failure mode at evaluation time. When the live
EMOS calibrator predicts σ_emos exceeding `emos_runtime_sigma_threshold`
multiples of the 1°F bracket width, the trade is rejected with the new
``EMOS_SIGMA_INCOMPATIBLE`` reason.

The fit-time σ-gate (``python.analytics.emos_fit.bootstrap_cell``) catches
new bad fits at the next refit. The runtime gate is the only guard for
currently-loaded coefficients (NYC/high σ=7.71, CHI/low σ=6.49, MIA/low
σ=3.56).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from python.analytics.audit_log import AuditSinks, RejectionReason, ScanFunnel
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
# Fixtures — modeled on tests/test_calibration_dispatcher.py
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
        emos_runtime_sigma_threshold=2.0,
    )
    base.update(overrides)
    return _EvalConfig(**base)


def _make_parsed(
    *,
    city: str = "MIA",
    var: DailyVariable = DailyVariable.TEMP_MAX,
    strike_type: Direction = Direction.ABOVE,
    floor_strike: float | None = 85.0,
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
    yes_bid: float = 0.20,
    yes_ask: float = 0.30,
) -> ScanCandidate:
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
    cells: dict[str, dict[str, Any]],
) -> EmosTable:
    """Build a hermetic EMOS table with per-cell coefficient overrides.

    ``cells`` is a dict keyed by ``"<CITY>/<high|low>"`` whose values may
    override ``(a, b, c, d, ens_std_p99, ...)`` per cell.
    """
    default: dict[str, Any] = {
        "a": 0.0,
        "b": 1.0,
        "c": 1.0,
        "d": 0.2,
        "n_train": 50,
        "leads_pooled": [0, 1, 2],
        "low_n": False,
        "ci_warnings": [],
        "ens_std_p99": 6.5,
    }
    cell_dict: dict[str, Any] = {}
    for cell_key, overrides in cells.items():
        cell_dict[cell_key] = dict(default, **overrides)
    payload = {
        "fit_date": "2026-05-19",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": cell_dict,
    }
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)
    assert table.is_loaded, f"fixture must load; got {table.fallback_reason}"
    return table


class _CapturingEvalLog:
    """In-memory drop-in for AuditLogger.emit_eval."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit_eval(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


def _make_sinks_with_capture() -> tuple[AuditSinks, _CapturingEvalLog]:
    cap = _CapturingEvalLog()
    sinks = AuditSinks(eval_log=cap, candidate_log=None, funnel_log=None)  # type: ignore[arg-type]
    return sinks, cap


def _build_eval_kwargs(
    parsed: ParsedMarket,
    temps: list[float],
    *,
    ens_std_override: float | None = None,
) -> dict[str, Any]:
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    bias_corrections = {fkey: 0.0}
    # The σ-gate looks at raw_ensemble_stds (ens_std_local) at runtime; allow
    # tests to override the value so we can drive the ratio without
    # synthesizing a specific member list.
    raw_stds = {
        fkey: ens_std_override if ens_std_override is not None else (max(temps) - min(temps)) / 4.0
    }
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


# Temps straddle strike=85 → raw above ≈ 0.4 (between RAW_PROB_GATE bounds).
_STRADDLE_85_TEMPS = [83.0, 84.0, 85.0, 86.0, 87.0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emos_sigma_gate_rejects_at_threshold_breach(tmp_path: Path) -> None:
    """LAX/high-like fit (c=0.77, d=7.19) on a market with realistic ens_std
    produces σ_emos ≈ 5.76°F at ens_std=2.0; runtime_ratio ≈ 5.76 > 2.0.

    Expect ``evaluate_market_pure`` returns None and the audit row carries
    ``EMOS_SIGMA_INCOMPATIBLE``.
    """
    parsed = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0)

    table = _make_emos_table_with_cells(
        tmp_path,
        cells={"LAX/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19}},
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"LAX/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    assert result is None, "σ-gate must reject the candidate"
    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE in reasons, (
        f"expected EMOS_SIGMA_INCOMPATIBLE; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 1, (
        "funnel must increment the new sigma-incompatible counter"
    )
    # Sanity on extras payload.
    sig_row = next(
        r for r in cap.rows if r.get("reason") == RejectionReason.EMOS_SIGMA_INCOMPATIBLE
    )
    assert "runtime_ratio" in sig_row, sig_row
    assert sig_row["runtime_ratio"] > 2.0
    assert sig_row["threshold"] == 2.0


def test_emos_sigma_gate_admits_within_threshold(tmp_path: Path) -> None:
    """Same LAX/high-like cell, but ens_std=0.0 → σ_emos = √0.77 ≈ 0.88 < 2.0.

    Expect the trade proceeds — no EMOS_SIGMA_INCOMPATIBLE rejection.
    """
    parsed = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=0.0)

    table = _make_emos_table_with_cells(
        tmp_path,
        cells={"LAX/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19}},
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"LAX/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons, (
        f"σ-gate must NOT fire when ratio <= threshold; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 0
    # Either the candidate passes (returns _MarketCandidate) or it's
    # rejected by a downstream gate — both are acceptable; the gate
    # under test simply must not be the cause.
    if result is not None:
        assert result.live_calibrator_kind == "emos_raw"


def test_emos_sigma_gate_skipped_for_isotonic_cells(tmp_path: Path) -> None:
    """Non-EMOS-listed cell (CHI/high) with the same LAX/high-like coefficients
    must NOT invoke the runtime σ-gate. The dispatcher routes it through
    isotonic (or identity); the gate only fires when live_calibrator_kind ==
    'emos_raw'.
    """
    parsed = _make_parsed(city="CHI", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0)

    # Even though CHI/high carries explosive (c, d), the cell is NOT in
    # emos_enabled_cells so the dispatcher leaves model_prob as isotonic.
    table = _make_emos_table_with_cells(
        tmp_path,
        cells={"CHI/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19}},
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},  # NOT CHI/high
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons, (
        f"σ-gate must NOT fire on isotonic-routed cells; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 0


def test_emos_sigma_gate_respects_config_threshold(tmp_path: Path) -> None:
    """Raising emos_runtime_sigma_threshold to 10.0 must allow the previously
    rejected (LAX/high, σ_emos≈5.76 at ens_std=2.0) candidate through.
    """
    parsed = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0)

    table = _make_emos_table_with_cells(
        tmp_path,
        cells={"LAX/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19}},
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=10.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"LAX/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons, (
        f"σ-gate must respect raised threshold; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 0


def test_emos_sigma_gate_uses_per_cell_coefficients(tmp_path: Path) -> None:
    """Gate must look up coefficients for the cell being evaluated, not a
    hardcoded reference cell. Build a table with two distinct cells:

      - LAX/high: small (c, d) → σ_emos ≈ 1°F at ens_std=1.0  → admits
      - MIA/high: explosive (c, d) → σ_emos > 5°F            → rejects

    Run two evaluations, one per cell, and confirm the gate decisions differ.
    """
    table = _make_emos_table_with_cells(
        tmp_path,
        cells={
            "LAX/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.5},
            "MIA/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19},
        },
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    enabled = {"LAX/high", "MIA/high"}

    # --- LAX/high evaluation: σ_emos = √(0.5 + 0.5·1) = 1.0 → ratio 1.0 < 2 ---
    parsed_lax = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    sc_lax = _make_scan_candidate(parsed_lax)
    kwargs_lax = _build_eval_kwargs(parsed_lax, _STRADDLE_85_TEMPS, ens_std_override=1.0)
    sinks_lax, cap_lax = _make_sinks_with_capture()
    funnel_lax = ScanFunnel()
    evaluate_market_pure(
        sc_lax,
        kwargs_lax["forecast_cache"],
        kwargs_lax["bias_corrections"],
        kwargs_lax["raw_ensemble_stds"],
        kwargs_lax["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs_lax["weights"],
        weather_client=kwargs_lax["weather_client"],
        bankroll=kwargs_lax["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs_lax["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells=enabled,
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks_lax,
        funnel=funnel_lax,
    )
    reasons_lax = [r.get("reason") for r in cap_lax.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons_lax, (
        f"LAX/high (well-behaved) must not be σ-gated; got {reasons_lax}"
    )
    assert funnel_lax.n_emos_sigma_incompatible == 0

    # --- MIA/high evaluation: σ_emos ≈ √(0.77 + 7.19·4) ≈ √29.53 ≈ 5.43 ---
    parsed_mia = _make_parsed(city="MIA", var=DailyVariable.TEMP_MAX)
    sc_mia = _make_scan_candidate(parsed_mia)
    kwargs_mia = _build_eval_kwargs(parsed_mia, _STRADDLE_85_TEMPS, ens_std_override=2.0)
    sinks_mia, cap_mia = _make_sinks_with_capture()
    funnel_mia = ScanFunnel()
    evaluate_market_pure(
        sc_mia,
        kwargs_mia["forecast_cache"],
        kwargs_mia["bias_corrections"],
        kwargs_mia["raw_ensemble_stds"],
        kwargs_mia["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs_mia["weights"],
        weather_client=kwargs_mia["weather_client"],
        bankroll=kwargs_mia["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs_mia["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells=enabled,
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks_mia,
        funnel=funnel_mia,
    )
    reasons_mia = [r.get("reason") for r in cap_mia.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE in reasons_mia, (
        f"MIA/high (explosive) must be σ-gated; got {reasons_mia}"
    )
    assert funnel_mia.n_emos_sigma_incompatible == 1


def test_emos_sigma_gate_boundary_strict_inequality(tmp_path: Path) -> None:
    """Pin the strict-inequality semantics: ratio == threshold ADMITS.

    With c=4.0, d=0.0, ens_std=0.0 → σ_emos = √4 = 2.0 → ratio = 2.0.
    Threshold = 2.0. The gate uses `>`, so the row must NOT be rejected.
    Regression guard against a future edit flipping to `>=`.
    """
    parsed = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=0.0)
    table = _make_emos_table_with_cells(
        tmp_path, cells={"LAX/high": {"a": 0.0, "b": 1.0, "c": 4.0, "d": 0.0}}
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"LAX/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )
    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons, (
        f"ratio == threshold (2.0) must ADMIT; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 0


def test_emos_sigma_gate_skipped_for_isotonic_fallback(tmp_path: Path) -> None:
    """Cell is in emos_enabled_cells but EMOS didn't compute (lead out of range
    → emos_prob_raw stays None → live_calibrator_kind = "isotonic_fallback").

    The σ-gate must NOT fire on isotonic_fallback rows because the isotonic
    plateau is back in play and EMOS's σ shape doesn't apply to the
    actually-used probability.
    """
    parsed = _make_parsed(city="LAX", var=DailyVariable.TEMP_MAX)
    # Push forecast 10 days out so emos_lead > 2 → emos_warnings = [lead_out_of_range]
    # → emos_prob_raw stays None → live_calibrator_kind = "isotonic_fallback".
    parsed = ParsedMarket(
        ticker=parsed.ticker,
        city=parsed.city,
        forecast_date=datetime.now(UTC).date() + timedelta(days=10),
        strike_type=parsed.strike_type,
        floor_strike=parsed.floor_strike,
        cap_strike=parsed.cap_strike,
        daily_variable=parsed.daily_variable,
    )
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0)

    # Coefficients are LAX/high-explosive — the gate WOULD fire if it reached
    # this branch incorrectly.
    table = _make_emos_table_with_cells(
        tmp_path, cells={"LAX/high": {"a": 0.0, "b": 1.0, "c": 0.77, "d": 7.19}}
    )
    cfg = _make_eval_cfg(emos_runtime_sigma_threshold=2.0)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = evaluate_market_pure(
        sc,
        kwargs["forecast_cache"],
        kwargs["bias_corrections"],
        kwargs["raw_ensemble_stds"],
        kwargs["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs["weights"],
        weather_client=kwargs["weather_client"],
        bankroll=kwargs["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"LAX/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.EMOS_SIGMA_INCOMPATIBLE not in reasons, (
        f"σ-gate must NOT fire on isotonic_fallback rows; got {reasons}"
    )
    assert funnel.n_emos_sigma_incompatible == 0
    # Confirm the fallback path actually fired (sanity on the test premise).
    # Either the candidate passed (live_calibrator_kind on the returned
    # candidate) or a downstream gate emitted an audit row carrying the
    # kind. Both confirm the dispatcher routed through the fallback branch.
    kinds: set[str] = {r.get("live_calibrator_kind") for r in cap.rows}
    if result is not None:
        kinds.add(result.live_calibrator_kind)
    assert "isotonic_fallback" in kinds, (
        f"test premise broken — expected isotonic_fallback path, got {kinds}"
    )
