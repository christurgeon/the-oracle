"""Tests for the EMOS shadow block in evaluate_market_pure (PR-2 Wave 2 / C2-C4)."""
from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from python.analytics.audit_log import AuditSinks, RejectionReason
from python.common import DailyVariable, Direction
from python.data.emos import EmosTable
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    _eval_audit_row,
    _EvalConfig,
    evaluate_market_pure,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_eval_cfg(**overrides: Any) -> _EvalConfig:
    base: dict[str, Any] = dict(
        max_spread=0.15,
        spread_skill_sensitivity=0.0,
        edge_threshold=0.05,
        max_edge=0.95,
        kelly_fraction=0.25,
        max_pct=0.20,
        max_take_pct=0.50,
        depth_check_enabled=False,
        min_raw_prob_for_between=0.02,
        min_distance_from_strike_std=0.0,
        max_fill_slippage=None,
        block_raw_prob_range=None,
        max_model_market_gap_near_term=None,
    )
    base.update(overrides)
    return _EvalConfig(**base)


def _make_parsed(
    *,
    city: str = "AUS",
    forecast_date: date | None = None,
    strike_type: Direction = Direction.ABOVE,
    floor_strike: float | None = 70.0,
    cap_strike: float | None = None,
    daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ticker: str | None = None,
) -> ParsedMarket:
    fd = forecast_date or (datetime.now(UTC).date() + timedelta(days=1))
    return ParsedMarket(
        ticker=ticker or f"KX-{city}-{fd.isoformat()}-T{floor_strike or 0:.0f}",
        city=city,
        forecast_date=fd,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        daily_variable=daily_variable,
    )


def _make_price_info(
    yes_bid: float = 0.40,
    yes_ask: float = 0.50,
    yes_bid_size: int = 200,
    yes_ask_size: int = 200,
    volume: int = 100,
    open_interest: int = 50,
) -> MarketPriceInfo:
    return MarketPriceInfo(
        midpoint=(yes_bid + yes_ask) / 2.0,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        spread=yes_ask - yes_bid,
        yes_bid_size=yes_bid_size,
        yes_ask_size=yes_ask_size,
        volume=volume,
        open_interest=open_interest,
    )


def _make_scan_candidate(
    parsed: ParsedMarket | None = None,
    price_info: MarketPriceInfo | None = None,
) -> ScanCandidate:
    p = parsed or _make_parsed()
    pi = price_info or _make_price_info()
    market: dict[str, Any] = {
        "ticker": p.ticker,
        "event_ticker": "EVT",
        "strike_type": p.strike_type.value,
        "floor_strike": p.floor_strike,
        "cap_strike": p.cap_strike,
        "yes_bid_dollars": str(pi.yes_bid),
        "yes_ask_dollars": str(pi.yes_ask),
        "volume_fp": "100.0",
        "open_interest_fp": "50.0",
        "close_time": "2099-12-31T23:59:59Z",
    }
    return ScanCandidate(market=market, parsed=p, price_info=pi)


def _make_forecast(
    city: str, fc_date: date, temps: list[float]
) -> EnsembleForecast:
    return EnsembleForecast(city=city, forecast_date=fc_date, temperatures=temps)


def _make_loaded_emos_table(tmp_path: Path) -> EmosTable:
    payload = {
        "fit_date": "2026-05-04",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": 0.0,
                "b": 1.0,
                "c": 1.0,
                "d": 0.0,
                "n_train": 50,
                "leads_pooled": [0, 1, 2],
                "low_n": False,
                "ci_warnings": [],
                "ens_std_p99": 6.5,
            },
        },
    }
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)
    assert table.is_loaded, f"fixture must be loaded; got {table.fallback_reason}"
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


def _make_weather_client() -> MagicMock:
    wc = MagicMock()
    wc.compute_model_probability = MagicMock(return_value=0.5)
    return wc


# ---------------------------------------------------------------------------
# C2/C3 — backtest path safety: emos_table=None default
# ---------------------------------------------------------------------------


def test_emos_table_none_yields_none_fields_and_table_not_loaded_reason():
    """When emos_table is omitted (backtest path), all EMOS fields stay None
    and emos_disabled_reason='table-not-loaded'."""
    parsed = _make_parsed(forecast_date=datetime.now(UTC).date() + timedelta(days=1))
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, [80.0] * 60 + [50.0] * 40)}}
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 5.0}
    raw_meds = {fkey: 70.0}

    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        _make_eval_cfg(edge_threshold=0.001, min_raw_prob_for_between=0.0),
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
    )

    # We expect either a candidate (passing all gates) or None (rejected on
    # one of the gates downstream of EMOS). Either way, EMOS state must be
    # untouched in the backtest path.
    if candidate is not None:
        assert candidate.emos_prob_raw is None
        assert candidate.emos_prob_corrected is None
        assert candidate.emos_warnings == []
        assert candidate.emos_disabled_reason == "table-not-loaded"
        assert candidate.emos_coeffs_artifact_hash == "identity-fallback"


def test_emos_block_fires_with_loaded_table_and_populates_candidate(tmp_path: Path):
    """Loaded EmosTable + raw_member_sink populated -> candidate carries
    emos_prob_raw, emos_prob_corrected, mu/sigma, artifact_hash."""
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [68.0, 70.0, 72.0, 74.0, 76.0]
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}
    bias_corrections = {fkey: 1.0}  # +1°F shift for the corrected variant
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}

    table = _make_loaded_emos_table(tmp_path)
    cfg = _make_eval_cfg(
        edge_threshold=0.001,
        min_raw_prob_for_between=0.0,
        max_edge=0.99,
    )

    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
    )

    if candidate is None:
        # If gates downstream rejected this market, we still need to be sure
        # the captured eval_audit row carried EMOS values. Re-run with a sink.
        sinks, cap = _make_sinks_with_capture()
        evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_stds,
            raw_meds,
            cfg,
            calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=_make_weather_client(),
            bankroll=10_000.0,
            emos_table=table,
            raw_member_sink=raw_member_sink,
            emos_log_dedup={},
            isotonic_artifact_hash="iso-test-hash",
            sinks=sinks,
        )
        # Pull whichever row went through the EMOS block
        assert cap.rows, "expected at least one rejection row"
        row = cap.rows[-1]
        assert row["emos_prob_raw"] is not None
        assert row["emos_coeffs_artifact_hash"] == table.artifact_hash
        return

    # Happy path
    assert candidate.emos_prob_raw is not None
    assert 0.0 <= candidate.emos_prob_raw <= 1.0
    assert candidate.emos_prob_corrected is not None
    assert candidate.emos_mu_raw is not None
    assert candidate.emos_sigma_raw is not None
    assert candidate.emos_mu_corr is not None
    assert candidate.emos_sigma_corr is not None
    # Coefficients are a=0, b=1 → mu_raw == ens_mean
    expected_mean = sum(raw_temps) / len(raw_temps)
    assert abs(candidate.emos_mu_raw - expected_mean) < 1e-9
    # Corrected variant adds +1°F shift to every member
    expected_corr_mean = expected_mean + 1.0
    assert abs(candidate.emos_mu_corr - expected_corr_mean) < 1e-9
    # Artifact hash is the loaded SHA-256, NOT the identity sentinel
    assert candidate.emos_coeffs_artifact_hash == table.artifact_hash
    assert candidate.isotonic_artifact_hash == "iso-test-hash"
    assert candidate.gfs_bias_correction_f == 1.0
    assert candidate.emos_disabled_reason is None


# ---------------------------------------------------------------------------
# C3 load-bearing: EMOS block fires BEFORE plateau-block gate
# ---------------------------------------------------------------------------


def test_emos_block_fires_BEFORE_plateau_block(tmp_path: Path):
    """A row that gets plateau-blocked must still emit eval_audit with
    emos_prob_raw populated and emos_coeffs_artifact_hash set to the
    loaded table's hash. Load-bearing for scope-C re-validation."""
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    # 70% above-strike: raw_ensemble_prob = 0.70, inside plateau range
    raw_temps = [80.0] * 7 + [50.0] * 3
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 4.0}
    raw_meds = {fkey: 75.0}
    raw_member_sink = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}

    table = _make_loaded_emos_table(tmp_path)
    cfg = _make_eval_cfg(block_raw_prob_range=(0.5, 0.9))

    sinks, cap = _make_sinks_with_capture()
    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    # Plateau-blocked → no candidate
    assert candidate is None
    # eval_audit row was emitted
    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert row["reason"] == RejectionReason.PLATEAU_BLOCK.value
    # EMOS columns are populated despite the rejection
    assert row["emos_prob_raw"] is not None
    assert row["emos_coeffs_artifact_hash"] == table.artifact_hash
    assert row["emos_lead"] == 1
    assert row["isotonic_artifact_hash"] == "iso-test-hash"


# ---------------------------------------------------------------------------
# Lead computation: UTC, NOT ET
# ---------------------------------------------------------------------------


def test_lead_uses_utc_not_et(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Lead is computed against datetime.now(UTC).date(), matching corpus
    convention. ET drift would silently mis-bucket runtime lead-0 rows
    into corpus lead-1 cells."""
    # Pin "today" by patching datetime in the weather_evaluation module
    # (evaluate_market_pure moved out of weather_edge.py in the Task-3 refactor).
    import python.strategies.weather_evaluation as we

    fixed_utc = datetime(2026, 5, 5, 6, 0, 0, tzinfo=UTC)  # 06:00 UTC = 02:00 ET

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_utc if tz is not None else fixed_utc.replace(tzinfo=None)

    monkeypatch.setattr(we, "datetime", _FixedDatetime)

    # forecast_date = 2026-05-06: lead is 1 in UTC
    fc_date = date(2026, 5, 6)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [68.0, 70.0, 72.0, 74.0, 76.0]
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}

    table = _make_loaded_emos_table(tmp_path)
    sinks, cap = _make_sinks_with_capture()

    evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        _make_eval_cfg(
            block_raw_prob_range=(0.0, 1.0),  # force a rejection so we capture the row
        ),
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        sinks=sinks,
    )

    assert len(cap.rows) == 1
    # UTC-based lead = (2026-05-06) - (2026-05-05) = 1 day
    assert cap.rows[0]["emos_lead"] == 1


# ---------------------------------------------------------------------------
# Cell key uses parsed.temp_label (high/low), NOT DailyVariable.name
# ---------------------------------------------------------------------------


def test_var_uses_temp_label_not_dailyvariable_name(tmp_path: Path):
    """The EMOS cell-key lookup uses 'high'/'low' (corpus + eval_audit
    convention), not 'TEMP_MAX'/'TEMP_MIN'. Tested by verifying a TEMP_MAX
    market resolves via the AUS/high cell."""
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
        daily_variable=DailyVariable.TEMP_MAX,  # → temp_label "high"
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [68.0, 70.0, 72.0, 74.0, 76.0]
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}}

    table = _make_loaded_emos_table(tmp_path)  # only AUS/high cell present
    sinks, cap = _make_sinks_with_capture()

    evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        _make_eval_cfg(block_raw_prob_range=(0.0, 1.0)),  # force capture
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        sinks=sinks,
    )

    assert len(cap.rows) == 1
    row = cap.rows[0]
    # Cell resolved → emos_prob_raw populated. If it had used "TEMP_MAX"
    # instead of "high", EmosTable.get would have raised KeyError and
    # emos_warnings would carry "var_shape_mismatch", with prob None.
    assert row["emos_prob_raw"] is not None
    assert "var_shape_mismatch" not in (row["emos_warnings"] or [])


# ---------------------------------------------------------------------------
# Closure pattern: rejection-row carries EMOS columns even when rejection
# fires BEFORE the EMOS block runs (e.g. EXCLUDED_CITY_VAR).
# ---------------------------------------------------------------------------


def test_eval_audit_row_carries_emos_columns_under_rejection(tmp_path: Path):
    """EXCLUDED_CITY_VAR fires before the EMOS block. The rejection row
    must STILL carry emos_coeffs_artifact_hash (from the up-front local)
    so the closure-pattern locals stay consistent across early gates."""
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, [70.0] * 5)}}
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 70.0}

    table = _make_loaded_emos_table(tmp_path)
    sinks, cap = _make_sinks_with_capture()

    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        _make_eval_cfg(),
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        excluded_city_vars=["AUS_high"],
        emos_table=table,
        raw_member_sink={},
        emos_log_dedup={},
        isotonic_artifact_hash="iso-hash-2",
        sinks=sinks,
    )

    assert candidate is None
    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert row["reason"] == RejectionReason.EXCLUDED_CITY_VAR.value
    # CRITICAL: closure-injected EMOS values present even on early reject
    assert row["emos_coeffs_artifact_hash"] == table.artifact_hash
    assert row["isotonic_artifact_hash"] == "iso-hash-2"
    # The EMOS block did NOT run yet (excluded fires first), so the
    # populated-by-block fields stay null. emos_disabled_reason is None
    # (since the table was loaded — disabled_reason reflects "is the
    # loader present", not "did the block fire").
    assert row["emos_prob_raw"] is None
    assert row["emos_disabled_reason"] is None  # table is loaded
    # gfs_bias_correction_f is None when the early reject fires
    # before fkey is set (the closure guards on `fkey is not None`).
    assert row["gfs_bias_correction_f"] is None


# ---------------------------------------------------------------------------
# Critical spec invariant: emos_coeffs_artifact_hash is "identity-fallback"
# (a STRING SENTINEL — never null) when EMOS is off.
# ---------------------------------------------------------------------------


def test_emos_coeffs_artifact_hash_is_identity_fallback_string_when_emos_off():
    """Pre-PR-2 spec invariant #5: emos_coeffs_artifact_hash is the literal
    string 'identity-fallback' (NEVER null) when EMOS is off, so analyzers
    can partition rows on the column value without null-handling."""
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(forecast_date=fc_date)
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, [70.0] * 5)}}

    sinks, cap = _make_sinks_with_capture()

    evaluate_market_pure(
        sc,
        forecast_cache,
        {fkey: 0.0},
        {fkey: 3.0},
        {fkey: 70.0},
        _make_eval_cfg(),
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        excluded_city_vars=["AUS_high"],
        # emos_table omitted (None) → identity fallback
        sinks=sinks,
    )

    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert row["emos_coeffs_artifact_hash"] == "identity-fallback"
    assert row["emos_disabled_reason"] == "table-not-loaded"


# ---------------------------------------------------------------------------
# C4 — _eval_audit_row direct test
# ---------------------------------------------------------------------------


def test_eval_audit_row_carries_emos_columns():
    """_eval_audit_row accepts and writes EMOS kwargs into the returned row."""
    parsed = _make_parsed()
    sc = _make_scan_candidate(parsed=parsed)
    row = _eval_audit_row(
        scan_id="abc",
        sc=sc,
        price_info=sc.price_info,
        reason=RejectionReason.PLATEAU_BLOCK,
        emos_prob_raw=0.42,
        emos_prob_corrected=0.45,
        emos_warnings=["low_n"],
        emos_lead=1,
        emos_coeffs_artifact_hash="abc123",
        isotonic_artifact_hash="iso-h",
        gfs_bias_correction_f=-0.5,
    )
    assert row["emos_prob_raw"] == 0.42
    assert row["emos_prob_corrected"] == 0.45
    assert row["emos_warnings"] == ["low_n"]
    assert row["emos_lead"] == 1
    assert row["emos_coeffs_artifact_hash"] == "abc123"
    assert row["isotonic_artifact_hash"] == "iso-h"
    assert row["gfs_bias_correction_f"] == -0.5
    # Defaulted fields stay None (or empty for warnings).
    assert row["emos_mu_raw"] is None
    assert row["emos_sigma_raw"] is None


def test_eval_audit_row_emos_warnings_default_empty_list():
    """When emos_warnings is omitted, the row carries [] (not None) so
    downstream pandas / jq filters never see null in this column."""
    parsed = _make_parsed()
    sc = _make_scan_candidate(parsed=parsed)
    row = _eval_audit_row(
        scan_id="abc",
        sc=sc,
        price_info=sc.price_info,
        reason=RejectionReason.PLATEAU_BLOCK,
    )
    assert row["emos_warnings"] == []


# ---------------------------------------------------------------------------
# C12b — guard removal regression: candidate.raw_ensemble_prob is populated
# even when the calibrator is identity (None). Required for EMOS analyzer
# baseline `Brier(raw_ensemble_prob, outcome)` over identity-calibrated rows.
# ---------------------------------------------------------------------------


class TestC12bGuardRemoval:
    """Pre-PR-2 spec invariant #4: raw_ensemble_prob must always be the
    blended fraction on the _MarketCandidate, never None just because the
    calibrator is identity. The analyzer needs this field to compute the
    Brier baseline over rows where a Track 1 emergency-unwind disabled
    isotonic during the EMOS shadow window."""

    def test_candidate_raw_ensemble_prob_populated_under_identity_calibrator(
        self,
    ) -> None:
        """When calibrator=None, _MarketCandidate.raw_ensemble_prob equals
        the blended fraction (NOT None)."""
        # Build a forecast where the model probability resolves to 0.42:
        # 42 of 100 ensemble members above the strike.
        fc_date = datetime.now(UTC).date() + timedelta(days=1)
        parsed = _make_parsed(
            city="AUS",
            forecast_date=fc_date,
            strike_type=Direction.ABOVE,
            floor_strike=70.0,
        )
        sc = _make_scan_candidate(parsed=parsed)
        fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
        # 42 above-70, 58 below-70 → raw_ensemble_prob == 0.42.
        raw_temps = [80.0] * 42 + [60.0] * 58
        forecast_cache = {
            fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
        }
        bias_corrections = {fkey: 0.0}
        raw_stds = {fkey: 5.0}
        raw_meds = {fkey: 70.0}

        # Make compute_model_probability return the expected blended fraction
        # so we can pin the assertion regardless of internal blending logic.
        wc = MagicMock()
        wc.compute_model_probability = MagicMock(return_value=0.42)

        candidate = evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_stds,
            raw_meds,
            _make_eval_cfg(
                edge_threshold=0.001,
                min_raw_prob_for_between=0.0,
                max_edge=0.99,
            ),
            calibrator=None,  # identity-calibrator path
            weights={"gfs_seamless": 1.0},
            weather_client=wc,
            bankroll=10_000.0,
            # emos_table omitted → identity fallback (parallel to a Track 1
            # emergency-unwind during the EMOS shadow window).
        )

        # Either the candidate passed all gates or it was rejected downstream.
        # In either case the candidate (when present) must carry the blended
        # fraction in raw_ensemble_prob — never None.
        if candidate is not None:
            assert candidate.raw_ensemble_prob is not None, (
                "Guard drop regression: raw_ensemble_prob must be populated "
                "even when calibrator is None."
            )
            assert candidate.raw_ensemble_prob == pytest.approx(0.42, abs=1e-9)


# ---------------------------------------------------------------------------
# C13 — _build_signal copies EMOS shadow fields into TradeSignal.metadata.
# Without this, the analyzer cannot join state/trades.jsonl ↔ signal metadata
# to compute counterfactual realized PnL (CRD).
# ---------------------------------------------------------------------------


class TestC13BuildSignalMetadata:
    """The winning candidate's TradeSignal.metadata must carry every EMOS
    shadow field the analyzer needs."""

    @staticmethod
    def _make_candidate(
        *,
        emos_prob_raw: float | None = 0.42,
        emos_prob_corrected: float | None = 0.45,
        emos_mu_raw: float | None = 70.5,
        emos_sigma_raw: float | None = 4.25,
        emos_warnings: list[str] | None = None,
        emos_coeffs_artifact_hash: str | None = "abc123def456",
        emos_coeffs_loaded_at: str | None = None,
    ) -> Any:
        """Build a minimal _MarketCandidate sufficient for _build_signal."""
        # Imported lazily — _MarketCandidate is module-private but stable.
        from python.strategies.weather_edge import _MarketCandidate

        fc_date = date(2026, 5, 6)
        parsed = _make_parsed(
            city="AUS",
            forecast_date=fc_date,
            strike_type=Direction.ABOVE,
            floor_strike=70.0,
        )
        price_info = _make_price_info(yes_bid=0.40, yes_ask=0.50)
        forecast = _make_forecast(parsed.city, fc_date, [70.0, 72.0, 74.0])
        market: dict[str, Any] = {
            "ticker": parsed.ticker,
            "event_ticker": "EVT",
            "close_time": "2099-12-31T23:59:59Z",
        }
        from python.analytics.sizing import SizingResult
        from python.common import Side

        sizing = SizingResult(
            side=Side.YES,
            dollars=100.0,
            kelly_raw=0.10,
            kelly_fraction_used=0.25,
            capped=False,
        )
        return _MarketCandidate(
            parsed=parsed,
            market=market,
            forecast=forecast,
            model_prob=0.42,
            correction_f=0.0,
            price_info=price_info,
            raw_edge=0.10,
            ensemble_confidence=0.90,
            effective_edge=0.08,
            sizing=sizing,
            contracts=200,
            exec_price=0.50,
            per_model_probs={"gfs_seamless": 0.42},
            ensemble_std_pooled=4.25,
            spread_inflation_factor=1.0,
            raw_ensemble_prob=0.42,
            emos_prob_raw=emos_prob_raw,
            emos_prob_corrected=emos_prob_corrected,
            emos_mu_raw=emos_mu_raw,
            emos_sigma_raw=emos_sigma_raw,
            emos_warnings=list(emos_warnings) if emos_warnings else [],
            emos_coeffs_artifact_hash=emos_coeffs_artifact_hash,
            emos_coeffs_loaded_at=emos_coeffs_loaded_at,
        ), sizing

    @staticmethod
    def _invoke_build_signal(
        candidate: Any,
        sizing: Any,
        *,
        emos_table: Any = None,
    ) -> Any:
        """Call _build_signal with a strategy-shaped namespace.

        ``_build_signal`` reads only ``self._name``, ``self._calibrator``,
        and ``self.emos_table`` — a SimpleNamespace with those attributes
        is sufficient.
        """
        from types import SimpleNamespace

        from python.strategies.weather_edge import (
            DepthCheck,
            WeatherEdgeStrategy,
        )

        strat = SimpleNamespace(
            _name="weather-edge-test",
            _calibrator=None,
            emos_table=emos_table,
        )
        depth = DepthCheck(
            fill_price=0.50,
            available_contracts=500,
            levels_used=1,
            total_levels=1,
            is_stale=False,
            slippage=0.0,
        )
        return WeatherEdgeStrategy._build_signal(
            strat,
            candidate,
            effective_edge=0.08,
            sizing=sizing,
            contracts=200,
            depth_capped=False,
            depth_result=depth,
        )

    def test_build_signal_metadata_carries_emos_fields(self) -> None:
        """A winning candidate's TradeSignal.metadata includes EMOS scalars,
        warnings copy, and the candidate's artifact hash."""
        candidate, sizing = self._make_candidate(
            emos_prob_raw=0.42,
            emos_prob_corrected=0.45,
            emos_mu_raw=70.5,
            emos_sigma_raw=4.25,
            emos_warnings=["low_n"],
            emos_coeffs_artifact_hash="abc123def456",
        )

        signal = self._invoke_build_signal(candidate, sizing, emos_table=None)

        meta = signal.metadata
        assert meta["emos_prob_raw"] == pytest.approx(0.42)
        assert meta["emos_prob_corrected"] == pytest.approx(0.45)
        assert meta["emos_mu_raw"] == pytest.approx(70.5)
        assert meta["emos_sigma_raw"] == pytest.approx(4.25)
        assert meta["emos_warnings"] == ["low_n"]
        assert meta["emos_coeffs_artifact_hash"] == "abc123def456"

    def test_build_signal_emos_warnings_is_defensive_copy(self) -> None:
        """Mutating the candidate's emos_warnings list after _build_signal
        must NOT mutate the signal's metadata copy."""
        candidate, sizing = self._make_candidate(emos_warnings=["w1"])
        signal = self._invoke_build_signal(candidate, sizing, emos_table=None)

        # Mutate the candidate's list.
        candidate.emos_warnings.append("w2")

        # Signal metadata still carries only the original entry.
        assert signal.metadata["emos_warnings"] == ["w1"]

    def test_build_signal_emos_coeffs_loaded_at_none_when_table_unset(
        self,
    ) -> None:
        """emos_coeffs_loaded_at is None when self.emos_table is None
        (e.g. backtest path or pre-setup)."""
        candidate, sizing = self._make_candidate()
        signal = self._invoke_build_signal(candidate, sizing, emos_table=None)
        assert signal.metadata["emos_coeffs_loaded_at"] is None

    def test_build_signal_emos_coeffs_loaded_at_read_from_candidate_not_self_table(
        self, tmp_path: Path
    ) -> None:
        """R2-H9 invariant: _build_signal reads emos_coeffs_loaded_at from
        candidate.emos_coeffs_loaded_at (pinned at evaluation time alongside
        artifact_hash), NOT from self.emos_table.artifact_loaded_at. A
        mid-scan hot reload would otherwise produce a signal where the hash
        came from the pre-swap table and loaded_at from the post-swap table.

        Construct a candidate with a known pinned loaded_at, and a
        DIFFERENT self.emos_table. Assert the signal's metadata carries
        the candidate's value, NOT the strategy's current table value.
        """
        pinned_loaded_at = "2026-05-06T12:00:00+00:00"
        candidate, sizing = self._make_candidate(
            emos_coeffs_loaded_at=pinned_loaded_at,
        )
        # Strategy's "current" table — different timestamp on purpose.
        post_swap_table = _make_loaded_emos_table(tmp_path)
        assert post_swap_table.is_loaded
        assert post_swap_table.artifact_loaded_at is not None
        assert post_swap_table.artifact_loaded_at != pinned_loaded_at

        signal = self._invoke_build_signal(
            candidate, sizing, emos_table=post_swap_table
        )
        # Reads from the candidate, NOT from the strategy's current table.
        assert signal.metadata["emos_coeffs_loaded_at"] == pinned_loaded_at
        assert (
            signal.metadata["emos_coeffs_loaded_at"]
            != post_swap_table.artifact_loaded_at
        )

    def test_build_signal_emos_warnings_empty_when_candidate_warnings_empty(
        self,
    ) -> None:
        """Candidate's empty emos_warnings → metadata's value is the empty
        list (NOT None) so analyzers can rely on a uniform list type."""
        candidate, sizing = self._make_candidate(emos_warnings=[])
        signal = self._invoke_build_signal(candidate, sizing, emos_table=None)
        assert signal.metadata["emos_warnings"] == []


# ---------------------------------------------------------------------------
# C11 — backtest-path safety: ``python.analytics.backtest:1195`` invokes
# ``postprocess_forecasts`` WITHOUT the ``raw_member_sink`` kwarg, and the
# subsequent ``evaluate_market_pure`` call omits ``emos_table`` /
# ``raw_member_sink`` / ``emos_log_dedup``. PR-2 added these as keyword-only
# parameters with safe defaults; this test pins the call pattern so a future
# change to either signature surfaces as a unit-level regression rather than
# breaking the live backtest entry point silently.
# ---------------------------------------------------------------------------


class TestC11BacktestPathSafety:
    """The legacy backtest path predates PR-2 and must keep working untouched
    after the EMOS shadow extension. Both call sites use the v1 (pre-EMOS)
    kwargs; new EMOS kwargs MUST default safely (identity-fallback)."""

    def test_postprocess_forecasts_backtest_call_pattern_safe(self) -> None:
        """Replays the exact ``python/analytics/backtest.py:1195`` call shape
        — ``raw_member_sink`` is omitted. Asserts the function returns the
        legacy 4-tuple, no exception, no behavior change."""
        from python.strategies.weather_edge import postprocess_forecasts

        fc_date = date(2026, 5, 6)
        city = "AUS"
        var = DailyVariable.TEMP_MAX
        fkey = (city, fc_date, var)
        forecast_cache: dict[
            tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
        ] = {
            fkey: {
                "gfs_seamless": _make_forecast(
                    city, fc_date, [68.0, 70.0, 72.0, 74.0, 76.0]
                )
            }
        }

        # Backtest call shape: positional cache + bias_table=None,
        # bias_shrinkage=0.3, bias_max_correction_f=5.0, no
        # spread_inflation_overrides, NO raw_member_sink kwarg.
        result = postprocess_forecasts(
            forecast_cache=forecast_cache,
            bias_table=None,
            bias_shrinkage=0.3,
            bias_max_correction_f=5.0,
            spread_inflation_overrides=None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

        new_cache, bias_corrections, raw_stds, raw_medians = result
        # Cache contract: same fkeys, gfs_seamless preserved.
        assert set(new_cache.keys()) == {fkey}
        assert "gfs_seamless" in new_cache[fkey]
        # bias_table=None ⇒ no entries (function only populates when loaded).
        assert bias_corrections == {}
        # raw_stds / raw_medians are populated for every fkey, regardless
        # of whether raw_member_sink was passed.
        assert fkey in raw_stds
        assert fkey in raw_medians
        assert raw_stds[fkey] > 0.0
        assert raw_medians[fkey] == pytest.approx(72.0, abs=1e-9)

    def test_evaluate_market_pure_backtest_call_pattern_safe(self) -> None:
        """Replays the exact ``python/analytics/backtest.py:1210`` call shape
        — no ``emos_table`` / ``raw_member_sink`` / ``emos_log_dedup`` /
        ``isotonic_artifact_hash`` kwargs. Confirms the v1 backtest call
        signature still works after PR-2 added those as keyword-only
        parameters with safe defaults."""
        fc_date = datetime.now(UTC).date() + timedelta(days=1)
        parsed = _make_parsed(
            city="AUS",
            forecast_date=fc_date,
            strike_type=Direction.ABOVE,
            floor_strike=70.0,
        )
        sc = _make_scan_candidate(parsed=parsed)
        fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
        forecast_cache = {
            fkey: {
                "gfs_seamless": _make_forecast(
                    parsed.city, fc_date, [80.0] * 60 + [50.0] * 40
                )
            }
        }
        bias_corrections = {fkey: 0.0}
        raw_stds = {fkey: 5.0}
        raw_meds = {fkey: 70.0}

        # Backtest v1 call: no EMOS kwargs at all.
        candidate = evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_stds,
            raw_meds,
            _make_eval_cfg(edge_threshold=0.001, min_raw_prob_for_between=0.0),
            calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=_make_weather_client(),
            bankroll=10_000.0,
            excluded_city_vars=[],
        )

        # Backtest path tolerates rejection. The contract is "no exception".
        # When a candidate IS produced, EMOS state must be the identity
        # fallback (sentinel hash, table-not-loaded reason, no warnings).
        if candidate is not None:
            assert candidate.emos_prob_raw is None
            assert candidate.emos_prob_corrected is None
            assert candidate.emos_warnings == []
            assert candidate.emos_disabled_reason == "table-not-loaded"
            assert candidate.emos_coeffs_artifact_hash == "identity-fallback"


def test_emos_shadow_reads_gfs_from_nested_sink(tmp_path: Path):
    """Regression gate for the EMOS read-site sink-shape change (spec §9 R1).

    Feeds evaluate_market_pure a nested sink containing both GFS and
    ECMWF with deliberately different means. Asserts:
      1. emos_prob_raw is not None (EMOS extracted GFS members)
      2. emos_mu_raw ≈ a + b * mean(gfs_members) — confirms EMOS read
         GFS-specific data, NOT the ECMWF data.

    A wrong-key regression (e.g. reading ECMWF by mistake) would make
    emos_mu_raw match the ECMWF mean, which this test catches.
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    gfs_members = [68.0, 70.0, 72.0, 74.0, 76.0]   # mean 72.0
    ecmwf_members = [88.0, 90.0, 92.0, 94.0, 96.0]  # mean 92.0 — deliberately different
    forecast_cache = {
        fkey: {
            "gfs_seamless": _make_forecast(parsed.city, fc_date, gfs_members),
            "ecmwf_ifs025": _make_forecast(parsed.city, fc_date, ecmwf_members),
        }
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    # Sink: nested per the new spec (spec §5.2).
    raw_member_sink = {
        fkey: {
            "gfs_seamless": _make_forecast(parsed.city, fc_date, gfs_members),
            "ecmwf_ifs025": _make_forecast(parsed.city, fc_date, ecmwf_members),
        }
    }

    table = _make_loaded_emos_table(tmp_path)  # a=0, b=1 ⇒ mu_raw == ens_mean
    cfg = _make_eval_cfg(
        edge_threshold=0.001,
        min_raw_prob_for_between=0.0,
        max_edge=0.99,
    )

    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        # GFS-only blend so the test does not depend on weight resolution.
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
    )

    # If gates downstream rejected, re-run with a sink to capture the row.
    if candidate is None:
        sinks, cap = _make_sinks_with_capture()
        evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_stds,
            raw_meds,
            cfg,
            calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=_make_weather_client(),
            bankroll=10_000.0,
            emos_table=table,
            raw_member_sink=raw_member_sink,
            emos_log_dedup={},
            isotonic_artifact_hash="iso-test-hash",
            sinks=sinks,
        )
        assert cap.rows, "expected at least one rejection row"
        row = cap.rows[-1]
        assert row["emos_prob_raw"] is not None
        assert abs(row["emos_mu_raw"] - 72.0) < 1e-9  # GFS mean, NOT ECMWF mean (92.0)
        return

    assert candidate.emos_prob_raw is not None
    assert abs(candidate.emos_mu_raw - 72.0) < 1e-9  # mu_raw == GFS mean
    # Defensive: if the implementation accidentally reads ECMWF, mu would be 92.0.
    assert abs(candidate.emos_mu_raw - 92.0) > 1.0


# ---------------------------------------------------------------------------
# NaN / Inf filter contract on the EMOS shadow path
# ---------------------------------------------------------------------------


def _run_eval_with_capture(
    *,
    sc: ScanCandidate,
    forecast_cache: dict,
    bias_corrections: dict,
    raw_stds: dict,
    raw_meds: dict,
    raw_member_sink: dict,
    table: EmosTable,
    cfg: _EvalConfig | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """Helper: run evaluate_market_pure once; if no candidate, re-run with a
    capturing sinks to get the eval_audit row. Returns (candidate, row_or_None)
    where exactly one of the two is non-None on the EMOS branch.
    """
    cfg = cfg or _make_eval_cfg(
        edge_threshold=0.001,
        min_raw_prob_for_between=0.0,
        max_edge=0.99,
    )
    candidate = evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
    )
    if candidate is not None:
        return candidate, None
    sinks, cap = _make_sinks_with_capture()
    evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )
    assert cap.rows, "expected at least one rejection row"
    return None, cap.rows[-1]


def _emos_field(candidate: Any, row: dict[str, Any] | None, field_name: str) -> Any:
    """Pull an EMOS field from whichever container (candidate or audit row)
    came back from _run_eval_with_capture."""
    if candidate is not None:
        return getattr(candidate, field_name)
    assert row is not None
    return row[field_name]


def test_evaluate_market_pure_filters_nan_from_emos_raw(tmp_path: Path):
    """NaN members in the GFS sink must be filtered before EMOS reduces.

    Raw-variant fields (emos_prob_raw, emos_mu_raw, emos_sigma_raw,
    emos_cdf_at_floor_raw, emos_cdf_at_cap_raw) must all be finite.
    Identity fixture (a=0, b=1) ⇒ emos_mu_raw == mean(filtered) == 72.0.

    Uses a BETWEEN market so both cdf_at_floor and cdf_at_cap populate
    (the "greater" / "less" types return one of them as None by design).
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.BETWEEN,
        floor_strike=68.0,
        cap_strike=76.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [70.0, math.nan, 72.0, math.nan, 74.0]  # finite mean = 72.0
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)

    candidate, row = _run_eval_with_capture(
        sc=sc,
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_stds=raw_stds,
        raw_meds=raw_meds,
        raw_member_sink=raw_member_sink,
        table=table,
    )

    for fname in (
        "emos_prob_raw",
        "emos_mu_raw",
        "emos_sigma_raw",
        "emos_cdf_at_floor_raw",
        "emos_cdf_at_cap_raw",
    ):
        val = _emos_field(candidate, row, fname)
        assert val is not None, f"{fname} should be populated (got None)"
        assert math.isfinite(val), f"{fname} must be finite; got {val!r}"

    mu_raw = _emos_field(candidate, row, "emos_mu_raw")
    # Identity fixture: μ_raw = a + b·mean = 0 + 1·72.0 = 72.0
    assert abs(mu_raw - 72.0) < 1e-9


def test_evaluate_market_pure_filters_nan_from_emos_corrected(tmp_path: Path):
    """NaN members must also be filtered for the bias-corrected variant.

    With identity fixture (a=0, b=1) and bias_corrections[fkey] = 1.0,
    emos_mu_corr == mean(filtered) + shift == 73.0.

    Uses a BETWEEN market so both cdf_at_floor_corr and cdf_at_cap_corr
    populate.
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.BETWEEN,
        floor_strike=68.0,
        cap_strike=76.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [70.0, math.nan, 72.0, math.nan, 74.0]  # finite mean = 72.0
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    bias_corrections = {fkey: 1.0}  # +1°F shift on the corrected variant
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)

    candidate, row = _run_eval_with_capture(
        sc=sc,
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_stds=raw_stds,
        raw_meds=raw_meds,
        raw_member_sink=raw_member_sink,
        table=table,
    )

    for fname in (
        "emos_prob_corrected",
        "emos_mu_corr",
        "emos_sigma_corr",
        "emos_cdf_at_floor_corr",
        "emos_cdf_at_cap_corr",
    ):
        val = _emos_field(candidate, row, fname)
        assert val is not None, f"{fname} should be populated (got None)"
        assert math.isfinite(val), f"{fname} must be finite; got {val!r}"

    mu_corr = _emos_field(candidate, row, "emos_mu_corr")
    # Identity fixture: μ_corr = a + b·(mean(filtered) + shift) = 72.0 + 1.0 = 73.0
    assert abs(mu_corr - 73.0) < 1e-9


def test_evaluate_market_pure_skips_emos_when_all_members_nan(tmp_path: Path):
    """If every GFS member is NaN, _extract returns None and BOTH variants
    skip via the ``is None`` short-circuit (NOT via the ``len >= 2`` guard).

    All 10 EMOS shadow fields must remain None. The ens_std_extrapolation
    tripwire must NOT fire (no std to compare against the p99).
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [math.nan, math.nan, math.nan]  # all non-finite ⇒ _extract returns None
    # forecast_cache still uses an in-band value so other gates don't fire
    forecast_cache = {
        fkey: {
            "gfs_seamless": _make_forecast(
                parsed.city, fc_date, [68.0, 70.0, 72.0, 74.0, 76.0]
            )
        }
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    # Only the EMOS sink carries the all-NaN payload (the filter is on the
    # sink-side reduction path).
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)

    candidate, row = _run_eval_with_capture(
        sc=sc,
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_stds=raw_stds,
        raw_meds=raw_meds,
        raw_member_sink=raw_member_sink,
        table=table,
    )

    for fname in (
        "emos_prob_raw",
        "emos_prob_corrected",
        "emos_mu_raw",
        "emos_sigma_raw",
        "emos_mu_corr",
        "emos_sigma_corr",
        "emos_cdf_at_floor_raw",
        "emos_cdf_at_cap_raw",
        "emos_cdf_at_floor_corr",
        "emos_cdf_at_cap_corr",
    ):
        val = _emos_field(candidate, row, fname)
        assert val is None, f"{fname} should be None when all members are NaN; got {val!r}"

    warnings = _emos_field(candidate, row, "emos_warnings") or []
    assert "ens_std_extrapolation" not in warnings, (
        "tripwire must not fire when no finite std was computed"
    )


def test_evaluate_market_pure_filters_inf_from_emos(tmp_path: Path):
    """Both +inf and -inf members must be filtered out by math.isfinite.

    Members: [68, 70, 72, 74, 76, +inf, -inf] → filtered to 5 finite floats
    (mean = 72.0). All raw + corrected EMOS scalars must be finite.

    Uses a BETWEEN market so both cdf_at_floor and cdf_at_cap populate.
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.BETWEEN,
        floor_strike=68.0,
        cap_strike=76.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [68.0, 70.0, 72.0, 74.0, 76.0, math.inf, -math.inf]
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)

    candidate, row = _run_eval_with_capture(
        sc=sc,
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_stds=raw_stds,
        raw_meds=raw_meds,
        raw_member_sink=raw_member_sink,
        table=table,
    )

    for fname in (
        "emos_prob_raw",
        "emos_mu_raw",
        "emos_sigma_raw",
        "emos_cdf_at_floor_raw",
        "emos_cdf_at_cap_raw",
        "emos_prob_corrected",
        "emos_mu_corr",
        "emos_sigma_corr",
        "emos_cdf_at_floor_corr",
        "emos_cdf_at_cap_corr",
    ):
        val = _emos_field(candidate, row, fname)
        assert val is not None, f"{fname} should be populated (got None)"
        assert math.isfinite(val), f"{fname} must be finite; got {val!r}"


def test_evaluate_market_pure_emos_lt_2_finite_skips(tmp_path: Path):
    """If filtering leaves <2 finite members, both variants skip via the
    ``len(finite) >= 2`` guard.

    Pin fixture preconditions (lead=1, city=AUS, var=high, identity coeffs
    with empty ci_warnings and low_n=False) so emos_warnings stays exactly
    [] — proving the skip happened on the len-guard path, not via a
    coefficients-side warning.
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)  # lead = 1
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
        daily_variable=DailyVariable.TEMP_MAX,  # → "high"
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [70.0, math.nan]  # filtered → [70.0] (len 1)
    forecast_cache = {
        fkey: {
            "gfs_seamless": _make_forecast(
                parsed.city, fc_date, [68.0, 70.0, 72.0, 74.0, 76.0]
            )
        }
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)

    # Force a rejection so we always get a row to inspect.
    cfg = _make_eval_cfg(block_raw_prob_range=(0.0, 1.0))
    sinks, cap = _make_sinks_with_capture()
    evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert row["emos_prob_raw"] is None
    assert row["emos_prob_corrected"] is None
    assert row["emos_warnings"] == []


def test_evaluate_market_pure_ens_std_extrapolation_fires_post_filter(tmp_path: Path):
    """Tripwire unblinding: with a NaN-laced member list, the std computed on
    the *filtered* list must still be compared against ens_std_p99.

    Members: [60, 70, 80, 90, 100, NaN] → filtered to [60, 70, 80, 90, 100].
    Sample std (ddof=1) ≈ 15.81 ≫ p99=6.5 (set by _make_loaded_emos_table).
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    raw_temps = [60.0, 70.0, 80.0, 90.0, 100.0, math.nan]
    forecast_cache = {
        fkey: {
            "gfs_seamless": _make_forecast(
                parsed.city, fc_date, [68.0, 70.0, 72.0, 74.0, 76.0]
            )
        }
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 72.0}
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, fc_date, raw_temps)}
    }
    table = _make_loaded_emos_table(tmp_path)
    # Defend against fixture drift — p99 MUST be 6.5 for this test to be
    # meaningful (post-filter std ≈ 15.81 > 6.5).
    assert table.training_ens_std_p99("AUS", "high") == 6.5

    cfg = _make_eval_cfg(block_raw_prob_range=(0.0, 1.0))  # force capture
    sinks, cap = _make_sinks_with_capture()
    evaluate_market_pure(
        sc,
        forecast_cache,
        bias_corrections,
        raw_stds,
        raw_meds,
        cfg,
        calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=_make_weather_client(),
        bankroll=10_000.0,
        emos_table=table,
        raw_member_sink=raw_member_sink,
        emos_log_dedup={},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
    )

    assert len(cap.rows) == 1
    row = cap.rows[0]
    assert "ens_std_extrapolation" in (row["emos_warnings"] or []), (
        "tripwire must fire on the filtered list — NaN propagation would silently "
        "blind it (NaN > p99 is always False)"
    )


def test_emos_shadow_filters_nan_gfs_only_when_ecmwf_clean(tmp_path: Path):
    """Mirror of test_emos_shadow_reads_gfs_from_nested_sink with NaN in
    GFS and clean ECMWF.

    GFS: [68, NaN, 72] → filtered mean = 70.0
    ECMWF: [88, 90, 92] (mean 90.0; the comment ties to >1.0 distance)

    With identity coeffs (a=0, b=1), emos_mu_raw must equal the filtered
    GFS mean (70.0), NOT the ECMWF mean. Defends the GFS-only contract
    under NaN conditions.
    """
    fc_date = datetime.now(UTC).date() + timedelta(days=1)
    parsed = _make_parsed(
        city="AUS",
        forecast_date=fc_date,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
    )
    sc = _make_scan_candidate(parsed=parsed)
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    gfs_members = [68.0, math.nan, 72.0]      # finite mean = 70.0
    ecmwf_members = [88.0, 90.0, 92.0]        # mean = 90.0 — deliberately different
    forecast_cache = {
        fkey: {
            "gfs_seamless": _make_forecast(parsed.city, fc_date, gfs_members),
            "ecmwf_ifs025": _make_forecast(parsed.city, fc_date, ecmwf_members),
        }
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: 3.0}
    raw_meds = {fkey: 70.0}
    raw_member_sink = {
        fkey: {
            "gfs_seamless": _make_forecast(parsed.city, fc_date, gfs_members),
            "ecmwf_ifs025": _make_forecast(parsed.city, fc_date, ecmwf_members),
        }
    }
    table = _make_loaded_emos_table(tmp_path)  # a=0, b=1

    candidate, row = _run_eval_with_capture(
        sc=sc,
        forecast_cache=forecast_cache,
        bias_corrections=bias_corrections,
        raw_stds=raw_stds,
        raw_meds=raw_meds,
        raw_member_sink=raw_member_sink,
        table=table,
    )

    mu_raw = _emos_field(candidate, row, "emos_mu_raw")
    assert mu_raw is not None
    assert math.isfinite(mu_raw)
    # mu_raw must be the filtered GFS mean (70.0), NOT the ECMWF mean.
    assert abs(mu_raw - 70.0) < 1e-9
    # Defensive: if NaN had propagated, this distance would have been NaN
    # (comparisons against NaN are False, which would fail the < 1e-9
    # check above already); this guards against accidentally reading ECMWF.
    assert abs(mu_raw - 90.0) > 1.0
