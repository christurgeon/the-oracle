"""Golden characterization of ``evaluate_market_pure``.

Each scenario drives the gate chain to one target outcome and records every
observable side effect: eval_audit rows (exact serialized strings, key order
preserved), ``ScanFunnel`` counters, the returned candidate, and the log
records of the ``python.strategies`` loggers. The expected output lives in
``tests/fixtures/evaluate_market_pure_golden.json``.

Regenerate with ``UPDATE_EVALUATE_GOLDEN=1`` only when the gate chain is meant
to change. A refactor must leave the fixture untouched.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import python.strategies.weather_evaluation as we
from python.analytics.audit_log import AuditSinks, ScanFunnel
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

FIXTURE = Path(__file__).parent / "fixtures" / "evaluate_market_pure_golden.json"
UPDATE = os.environ.get("UPDATE_EVALUATE_GOLDEN") == "1"

TODAY = date(2026, 5, 5)
FORECAST_DATE = date(2026, 5, 6)
FROZEN_NOW = datetime(2026, 5, 5, 6, 0, 0, tzinfo=UTC)
LOADED_AT = "2026-05-04T00:00:00+00:00"

BASE_TEMPS = [67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0]
INSIDE_TEMPS = [70.5, 71.0, 71.5, 70.2, 71.8, 70.9, 71.2, 71.1, 70.7, 71.4]
OUTSIDE_TEMPS = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 60.5, 61.5, 62.5, 63.5]


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return FROZEN_NOW if tz is not None else FROZEN_NOW.replace(tzinfo=None)


class _MapCal:
    kind = "isotonic"

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        return 0.1 + 0.8 * raw_prob


class _ConstCal:
    kind = "isotonic"

    def __init__(self, value: float) -> None:
        self.value = value

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        return self.value


class _RaisingCal:
    kind = "isotonic"

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        raise RuntimeError("shadow boom")


def _calibrator(key: str | None) -> Any:
    return {
        None: None,
        "map": _MapCal(),
        "disagree": _ConstCal(0.2),
        "low": _ConstCal(0.05),
        "raising": _RaisingCal(),
    }[key]


class _CapturingEvalLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit_eval(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append([record.name, record.levelname, record.getMessage()])


def _base_cell(**overrides: Any) -> dict[str, Any]:
    cell: dict[str, Any] = {
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
    cell.update(overrides)
    return cell


def _write_table(tmp_path: Path) -> EmosTable:
    payload = {
        "fit_date": "2026-05-04",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": {
            "NYC/high": _base_cell(),
            "BOS/high": _base_cell(a=-6.0),
            "LAX/high": _base_cell(c=0.77, d=7.19),
            "NYC/low": _base_cell(
                low_n=True, ci_warnings=["unstable_d", "d_at_zero"], ens_std_p99=0.5
            ),
            "MIA/low": _base_cell(),
            "CHI/high": {"skipped": True, "reason": "low_n"},
        },
    }
    path = tmp_path / "coefficients.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    table = EmosTable.load(path)
    assert table.is_loaded
    return dataclasses.replace(table, artifact_loaded_at=LOADED_AT)


@pytest.fixture(scope="module")
def base_table(tmp_path_factory: pytest.TempPathFactory) -> EmosTable:
    return _write_table(tmp_path_factory.mktemp("emos"))


def _resolve_table(key: str | None, base: EmosTable) -> EmosTable | None:
    if key is None:
        return None
    if key == "ok":
        return base
    if key == "not_loaded":
        return EmosTable._disabled(None, "file-missing")
    if key == "empty_hash":
        return dataclasses.replace(base, artifact_hash="")
    if key == "none_hash":
        return dataclasses.replace(base, artifact_hash=None)
    table = copy.copy(base)
    if key == "p99_raises":

        def _boom(city: str, var: str) -> float | None:
            raise RuntimeError("p99 boom")

        table.training_ens_std_p99 = _boom  # type: ignore[method-assign]
    elif key == "shape_mismatch":

        def _get_keyerror(*, city: str, var: str, lead: int) -> Any:
            raise KeyError("var shape")

        table.get = _get_keyerror  # type: ignore[method-assign]
    elif key == "stateful":
        real_get = base.get
        calls = {"n": 0}

        def _get_once(*, city: str, var: str, lead: int) -> Any:
            calls["n"] += 1
            if calls["n"] > 1:
                raise KeyError("second call")
            return real_get(city=city, var=var, lead=lead)

        table.get = _get_once  # type: ignore[method-assign]
    else:
        raise AssertionError(key)
    return table


def _cfg(**overrides: Any) -> _EvalConfig:
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


def _case(expect: str, **overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = dict(
        expect=expect,
        log=None,
        city="NYC",
        var=DailyVariable.TEMP_MAX,
        st=Direction.ABOVE,
        floor=70.0,
        cap=None,
        temps=BASE_TEMPS,
        ecmwf=None,
        ens_std=2.9,
        ens_median=71.5,
        bias=0.0,
        bid=0.30,
        ask=0.40,
        price_mode="given",
        cfg={},
        cal="map",
        shadow=None,
        emos=None,
        cells=None,
        member_sink=True,
        nbm="none",
        excluded=None,
        bankroll=10_000.0,
        fd=FORECAST_DATE,
        scan_id="scan-golden",
        dedup=True,
        weights={"gfs_seamless": 1.0},
        spread_overrides=None,
        emos_prob_raises=False,
        sinks=True,
        forecast_gap=False,
        iso_hash="iso-hash",
        n_existing=0,
        rho=None,
    )
    case.update(overrides)
    return case


def _between(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(st=Direction.BETWEEN, floor=70.0, cap=72.0, temps=INSIDE_TEMPS)
    base.update(overrides)
    return base


def _boundary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        city="MIA",
        var=DailyVariable.TEMP_MIN,
        st=Direction.BETWEEN,
        floor=70.0,
        cap=71.0,
        cfg=dict(boundary_filter_cells={"MIA/low": 0.5}),
    )
    base.update(overrides)
    return base


def _scenarios() -> dict[str, dict[str, Any]]:
    s: dict[str, dict[str, Any]] = {}

    def add(name: str, expect: str, **kw: Any) -> None:
        assert name not in s
        s[name] = _case(expect, **kw)

    add("pass_isotonic", "PASS")
    add("pass_isotonic_conf_discount", "PASS", cfg=dict(spread_skill_sensitivity=0.5))
    add("pass_identity_calibrator", "PASS", cal=None)
    add("pass_no_side", "PASS", bid=0.80, ask=0.90, temps=[60.0] * 5 + [71.0] * 5)
    add(
        "pass_shadow_calibrator_and_ecmwf",
        "PASS",
        shadow="map",
        ecmwf=[66.0, 69.0, 71.0, 72.0, 73.0, 75.0],
        weights={"gfs_seamless": 0.6, "ecmwf_ifs025": 0.4},
        spread_overrides={"NYC": {"high": 1.25}},
    )
    add("shadow_calibrator_raises", "PASS", shadow="raising", log="shadow calibrator raised")
    add("shadow_calibrator_raises_then_rejects", "SPREAD", shadow="raising", bid=0.20, ask=0.50)
    add(
        "shadow_prob_on_rejection_row",
        "CAL_FLOOR_BLOCK",
        shadow="map",
        cfg=dict(min_calibrated_prob=0.9),
    )
    add("forecast_gap", "FORECAST_GAP", forecast_gap=True)
    add("no_sinks_cal_floor", "CAL_FLOOR_BLOCK", cfg=dict(min_calibrated_prob=0.9), sinks=False)

    add("excluded_city_var", "EXCLUDED_CITY_VAR", excluded=["NYC_high"])
    add(
        "excluded_city_var_low",
        "EXCLUDED_CITY_VAR",
        excluded=["NYC_low"],
        var=DailyVariable.TEMP_MIN,
    )
    add("excluded_city_var_emos_loaded", "EXCLUDED_CITY_VAR", excluded=["NYC_high"], emos="ok")
    add(
        "excluded_city_var_emos_not_loaded",
        "EXCLUDED_CITY_VAR",
        excluded=["NYC_high"],
        emos="not_loaded",
    )
    add(
        "excluded_city_var_emos_none",
        "EXCLUDED_CITY_VAR",
        excluded=["NYC_high"],
        emos=None,
        iso_hash=None,
    )
    add(
        "excluded_city_var_empty_hash",
        "EXCLUDED_CITY_VAR",
        excluded=["NYC_high"],
        emos="empty_hash",
    )

    add("cal_floor_block", "CAL_FLOOR_BLOCK", cfg=dict(min_calibrated_prob=0.9))
    add("plateau_block", "PLATEAU_BLOCK", cfg=dict(block_raw_prob_range=(0.5, 0.7)))
    add(
        "boundary_filter_block",
        "BOUNDARY_FILTER_BLOCK",
        **_boundary(temps=[68.4, 69.4, 70.4]),
    )
    add(
        "boundary_downstream_edge_below",
        "EDGE_BELOW_THRESHOLD",
        **_boundary(
            temps=[73.0, 73.0, 73.0],
            bid=0.04,
            ask=0.06,
            cal=None,
            cfg=dict(boundary_filter_cells={"MIA/low": 0.5}, min_raw_prob_for_between=0.0),
        ),
    )
    add(
        "boundary_downstream_model_market_gap",
        "MODEL_MARKET_GAP_INTERIM",
        **_boundary(
            temps=[73.0, 73.0, 73.0],
            bid=0.45,
            ask=0.55,
            cfg=dict(
                boundary_filter_cells={"MIA/low": 0.5},
                min_raw_prob_for_between=0.0,
                max_model_market_gap_near_term=0.3,
            ),
        ),
    )
    add(
        "boundary_cell_not_between_passthrough",
        "PASS",
        city="MIA",
        var=DailyVariable.TEMP_MIN,
        cfg=dict(boundary_filter_cells={"MIA/low": 0.5}),
    )
    add(
        "boundary_nan_geometry_failopen",
        "EDGE_BELOW_THRESHOLD",
        **_boundary(
            floor=None,
            cap=71.0,
            temps=[73.0, 73.0, 73.0],
            bid=0.04,
            ask=0.06,
            cal=None,
            weights={"ecmwf_ifs025": 1.0},
            cfg=dict(boundary_filter_cells={"MIA/low": 0.5}, min_raw_prob_for_between=0.0),
        ),
    )
    add(
        "raw_prob_gate_between",
        "RAW_PROB_GATE",
        **_between(cfg=dict(min_raw_prob_for_between=0.9), temps=BASE_TEMPS),
    )
    add("raw_prob_gate_threshold_tail", "RAW_PROB_GATE", temps=[72.0] * 10)
    add("raw_prob_gate_threshold_low_tail", "RAW_PROB_GATE", temps=[60.0] * 10)

    add(
        "strike_distance_missing_strike",
        "STRIKE_DISTANCE",
        floor=None,
        weights={"ecmwf_ifs025": 1.0},
        cal=None,
        cfg=dict(min_distance_from_strike_std=1.0),
        log="missing strike",
    )
    add(
        "strike_distance_degenerate_std",
        "STRIKE_DISTANCE",
        ens_std=0.0,
        cfg=dict(min_distance_from_strike_std=1.0),
        log="degenerate ensemble",
    )
    add(
        "strike_distance_below_k",
        "STRIKE_DISTANCE",
        ens_median=70.0,
        cfg=dict(min_distance_from_strike_std=1.0),
        log="skipping",
    )
    add(
        "strike_distance_below_direction_below_k",
        "STRIKE_DISTANCE",
        st=Direction.BELOW,
        floor=None,
        cap=72.0,
        ens_median=72.0,
        cfg=dict(min_distance_from_strike_std=1.0),
    )

    add("no_price", "NO_PRICE", price_mode="none_unpriced")
    add("spread", "SPREAD", bid=0.20, ask=0.50)
    add("calibrator_disagreement_isotonic", "CALIBRATOR_DISAGREEMENT", cal="disagree")
    add(
        "model_market_gap_interim",
        "MODEL_MARKET_GAP_INTERIM",
        cfg=dict(max_model_market_gap_near_term=0.1),
    )
    add(
        "model_market_gap_not_near_term_pass",
        "PASS",
        fd=date(2026, 5, 9),
        cfg=dict(max_model_market_gap_near_term=0.1),
    )
    add("yes_floor_threshold", "YES_FLOOR", bid=0.05, ask=0.15)
    add(
        "yes_floor_between",
        "YES_FLOOR",
        **_between(
            bid=0.06,
            ask=0.10,
            cal=None,
            temps=[70.0, 71.0, 72.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
        ),
    )
    add("yes_inverted", "YES_INVERTED", bid=0.40, ask=0.70, cfg=dict(max_spread=0.5))
    add("no_inverted", "NO_INVERTED", bid=0.55, ask=0.70)
    add("edge_below_threshold", "EDGE_BELOW_THRESHOLD", bid=0.40, ask=0.52)
    add("edge_above_max", "EDGE_ABOVE_MAX", bid=0.20, ask=0.30, cfg=dict(max_edge=0.2))
    add("kelly_zero", "KELLY_ZERO", bankroll=5.0)

    add(
        "price_none_priced_market_cal_floor",
        "CAL_FLOOR_BLOCK",
        price_mode="none_priced",
        cfg=dict(min_calibrated_prob=0.9),
    )
    add("price_none_priced_market_spread", "SPREAD", price_mode="none_priced", bid=0.20, ask=0.50)
    add(
        "price_none_priced_market_edge_below",
        "EDGE_BELOW_THRESHOLD",
        price_mode="none_priced",
        bid=0.40,
        ask=0.52,
    )
    add("price_none_priced_market_pass", "PASS", price_mode="none_priced")

    strike_cfg = dict(min_distance_from_strike_std=1.0)
    add("strike_z_spread_row_has_z", "SPREAD", ens_median=75.0, bid=0.20, ask=0.50, cfg=strike_cfg)
    add(
        "strike_z_edge_below_row_has_z",
        "EDGE_BELOW_THRESHOLD",
        ens_median=75.0,
        bid=0.40,
        ask=0.52,
        cfg=strike_cfg,
    )
    add(
        "strike_z_cal_floor_row_z_none",
        "CAL_FLOOR_BLOCK",
        ens_median=75.0,
        cfg=dict(min_distance_from_strike_std=1.0, min_calibrated_prob=0.9),
    )
    add("strike_z_pass_candidate_has_z", "PASS", ens_median=75.0, cfg=strike_cfg)
    add(
        "strike_z_plateau_row_z_none",
        "PLATEAU_BLOCK",
        ens_median=75.0,
        cfg=dict(min_distance_from_strike_std=1.0, block_raw_prob_range=(0.5, 0.7)),
    )

    add("bias_correction_gfs_uncorrected", "PASS", bias=1.5)
    add("bias_correction_below_epsilon", "PASS", bias=0.005)
    add("bias_correction_edge_below_row", "EDGE_BELOW_THRESHOLD", bias=1.5, bid=0.40, ask=0.52)

    nbm_key = "nbm_between"
    add(f"{nbm_key}_agree_pass", "PASS", **_between(nbm=71.0, cal=None))
    add(
        f"{nbm_key}_veto_enabled_yes_rejects",
        "NBM_BRACKET_POSITION_VETO",
        **_between(nbm=75.0, cal=None, cfg=dict(nbm_bracket_position_veto_enabled=True)),
    )
    add(
        f"{nbm_key}_veto_enabled_no_rejects",
        "NBM_BRACKET_POSITION_VETO",
        **_between(
            temps=OUTSIDE_TEMPS,
            nbm=71.0,
            cal=None,
            cfg=dict(nbm_bracket_position_veto_enabled=True, min_raw_prob_for_between=0.0),
        ),
    )
    add(f"{nbm_key}_veto_disabled_pass", "PASS", **_between(nbm=75.0, cal=None))
    add(
        f"{nbm_key}_veto_disabled_edge_below",
        "EDGE_BELOW_THRESHOLD",
        **_between(nbm=75.0, cal=None, cfg=dict(edge_threshold=0.9)),
    )
    add(
        f"{nbm_key}_veto_disabled_edge_above",
        "EDGE_ABOVE_MAX",
        **_between(nbm=75.0, cal=None, bid=0.10, ask=0.20, cfg=dict(max_edge=0.3)),
    )
    add(
        f"{nbm_key}_veto_disabled_kelly_zero",
        "KELLY_ZERO",
        **_between(nbm=75.0, cal=None, bankroll=2.0),
    )
    add(
        f"{nbm_key}_fetch_failed_closed",
        "NBM_FETCH_FAILED",
        **_between(nbm="fail", cal=None, cfg=dict(nbm_fail_open=False)),
    )
    add(f"{nbm_key}_fetch_failed_open_pass", "PASS", **_between(nbm="fail", cal=None))
    add(
        f"{nbm_key}_fetch_failed_open_edge_below",
        "EDGE_BELOW_THRESHOLD",
        **_between(nbm="fail", cal=None, cfg=dict(edge_threshold=0.9)),
    )
    add(f"{nbm_key}_key_absent_noop_pass", "PASS", **_between(nbm="absent", cal=None))
    add("nbm_threshold_market_ignored", "PASS", nbm=75.0)
    add(
        f"{nbm_key}_veto_pre_yes_inverted_has_no_nbm",
        "YES_INVERTED",
        **_between(nbm=75.0, bid=0.40, ask=0.95, cfg=dict(max_spread=0.9)),
    )

    emos_cfg = dict(spread_skill_sensitivity=0.5)
    add("emos_shadow_only_pass", "PASS", emos="ok", cells=set())
    add("emos_raw_live_pass_conf_one", "PASS", emos="ok", cells={"NYC/high"}, cfg=emos_cfg)
    add(
        "emos_raw_live_disagreement_skipped",
        "PASS",
        emos="ok",
        cells={"BOS/high"},
        city="BOS",
        bid=0.40,
        ask=0.50,
        cfg=emos_cfg,
        log="CALIBRATOR_DISAGREEMENT skipped",
    )
    add(
        "emos_sigma_incompatible_ratio",
        "EMOS_SIGMA_INCOMPATIBLE",
        emos="ok",
        cells={"LAX/high"},
        city="LAX",
        ens_std=2.0,
        log="EMOS_SIGMA_INCOMPATIBLE",
    )
    add(
        "emos_sigma_incompatible_missing_coeffs",
        "EMOS_SIGMA_INCOMPATIBLE",
        emos="stateful",
        cells={"NYC/high"},
    )
    add(
        "emos_lead_out_of_range_fallback",
        "PASS",
        emos="ok",
        cells={"NYC/high"},
        fd=date(2026, 5, 9),
    )
    add("emos_lead_negative_fallback", "PASS", emos="ok", cells={"NYC/high"}, fd=date(2026, 5, 4))
    add("emos_var_shape_mismatch", "PASS", emos="shape_mismatch", cells={"NYC/high"})
    add("emos_cell_skipped_or_missing", "PASS", emos="ok", cells={"CHI/high"}, city="CHI")
    add("emos_cell_absent_from_table", "PASS", emos="ok", cells={"SEA/high"}, city="SEA")
    add("emos_extrapolation_ci_warnings_low_n", "PASS", emos="ok", var=DailyVariable.TEMP_MIN)
    add("emos_table_none_cell_listed_fallback", "PASS", emos=None, cells={"NYC/high"})
    add("emos_table_none_cell_listed_identity", "PASS", emos=None, cells={"NYC/high"}, cal=None)
    add("emos_table_not_loaded", "PASS", emos="not_loaded", cells={"NYC/high"})
    add("emos_table_empty_artifact_hash", "PASS", emos="empty_hash")
    add("emos_table_none_artifact_hash", "PASS", emos="none_hash")
    add("emos_no_member_sink", "PASS", emos="ok", cells={"NYC/high"}, member_sink=False)
    add("emos_raw_variant_raises_corr_ok", "PASS", emos="p99_raises", log="emos_raw_failed")
    add("emos_both_variants_raise_dedup", "PASS", emos="ok", emos_prob_raises=True, bias=1.0)
    add(
        "emos_both_variants_raise_scan_id_none",
        "PASS",
        emos="ok",
        emos_prob_raises=True,
        scan_id=None,
    )
    add("emos_both_variants_raise_no_dedup", "PASS", emos="ok", emos_prob_raises=True, dedup=False)
    add(
        "emos_rejection_rows_carry_emos_fields",
        "PLATEAU_BLOCK",
        emos="ok",
        cfg=dict(block_raw_prob_range=(0.5, 0.7)),
    )
    add(
        "emos_rejection_cal_floor_with_emos_live",
        "CAL_FLOOR_BLOCK",
        emos="ok",
        cells={"NYC/high"},
        cfg=dict(min_calibrated_prob=0.9),
    )
    add(
        "emos_rejection_edge_below_with_emos_live",
        "EDGE_BELOW_THRESHOLD",
        emos="ok",
        cells={"NYC/high"},
        bid=0.62,
        ask=0.67,
        cfg=dict(edge_threshold=0.2),
    )
    add(
        "emos_rejection_nbm_veto_with_emos",
        "NBM_BRACKET_POSITION_VETO",
        **_between(
            emos="ok",
            nbm=75.0,
            cells={"NYC/high"},
            cfg=dict(nbm_bracket_position_veto_enabled=True),
        ),
    )
    add("emos_raw_live_kelly_zero", "KELLY_ZERO", emos="ok", cells={"NYC/high"}, bankroll=5.0)
    add(
        "emos_isotonic_fallback_disagreement_gate_active",
        "CALIBRATOR_DISAGREEMENT",
        emos="ok",
        cells={"NYC/high"},
        fd=date(2026, 5, 9),
        cal="disagree",
    )

    return s


SCENARIOS = _scenarios()


def _serialize_row(row: dict[str, Any]) -> str:
    row = dict(row)
    row.pop("ts", None)
    row.pop("hours_to_settlement", None)
    return json.dumps(row, default=str)


def _run(case: dict[str, Any], base: EmosTable, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    c = case
    fd: date = c["fd"]
    floor, cap = c["floor"], c["cap"]
    st: Direction = c["st"]
    var: DailyVariable = c["var"]
    ticker = f"KX-{c['city']}-{fd.isoformat()}-{st.value}-{floor}-{cap}"
    parsed = ParsedMarket(
        ticker=ticker,
        city=c["city"],
        forecast_date=fd,
        strike_type=st,
        floor_strike=floor,
        cap_strike=cap,
        daily_variable=var,
    )
    market: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": "EVT-GOLDEN",
        "strike_type": st.value,
        "floor_strike": floor,
        "cap_strike": cap,
        "volume_fp": "100.0",
        "open_interest_fp": "50.0",
    }
    bid, ask = c["bid"], c["ask"]
    price_info: MarketPriceInfo | None = MarketPriceInfo(
        midpoint=(bid + ask) / 2.0,
        yes_bid=bid,
        yes_ask=ask,
        spread=ask - bid,
        yes_bid_size=200,
        yes_ask_size=200,
        volume=100,
        open_interest=50,
    )
    if c["price_mode"] in ("given", "none_priced"):
        market["yes_bid_dollars"] = f"{bid:.2f}"
        market["yes_ask_dollars"] = f"{ask:.2f}"
    if c["price_mode"] != "given":
        price_info = None
    sc = ScanCandidate(market=market, parsed=parsed, price_info=price_info)

    def forecast(temps: list[float], model: str) -> EnsembleForecast:
        return EnsembleForecast(
            city=c["city"], forecast_date=fd, temperatures=list(temps), model=model, fetched_at=0.0
        )

    fkey = (c["city"], fd, var)
    models = {"gfs_seamless": forecast(c["temps"], "gfs_seamless")}
    if c["ecmwf"] is not None:
        models["ecmwf_ifs025"] = forecast(c["ecmwf"], "ecmwf_ifs025")
    forecast_cache = {} if c["forecast_gap"] else {fkey: models}
    member_sink = (
        {fkey: {"gfs_seamless": forecast(c["temps"], "gfs_seamless")}} if c["member_sink"] else None
    )

    nbm = c["nbm"]
    nbm_cache: dict[Any, float | None] | None
    if nbm == "none":
        nbm_cache = None
    elif nbm == "absent":
        nbm_cache = {}
    elif nbm == "fail":
        nbm_cache = {fkey: None}
    else:
        nbm_cache = {fkey: float(nbm)}

    if c["emos_prob_raises"]:

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("emos boom")

        monkeypatch.setattr(we, "emos_probability", _boom)

    weather_client = MagicMock()
    weather_client.compute_model_probability = MagicMock(return_value=0.5)

    cap_log = _CapturingEvalLog()
    sinks = (
        AuditSinks(eval_log=cap_log, candidate_log=None, funnel_log=None)  # type: ignore[arg-type]
        if c["sinks"]
        else None
    )
    funnel = ScanFunnel()
    dedup: dict[Any, Any] | None = {} if c["dedup"] else None
    handler = _ListHandler()
    logger = logging.getLogger("python.strategies")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        candidate = evaluate_market_pure(
            sc,
            forecast_cache,
            {fkey: c["bias"]},
            {fkey: c["ens_std"]},
            {fkey: c["ens_median"]},
            _cfg(**c["cfg"]),
            calibrator=_calibrator(c["cal"]),
            shadow_calibrator=_calibrator(c["shadow"]),
            weights=c["weights"],
            weather_client=weather_client,
            bankroll=c["bankroll"],
            excluded_city_vars=c["excluded"],
            spread_inflation_overrides=c["spread_overrides"],
            emos_table=_resolve_table(c["emos"], base),
            raw_member_sink=member_sink,
            emos_log_dedup=dedup,
            emos_enabled_cells=c["cells"],
            isotonic_artifact_hash=c["iso_hash"],
            scan_id=c["scan_id"],
            sinks=sinks,
            funnel=funnel,
            scan_today=TODAY,
            n_existing_same_day_bets=c["n_existing"],
            correlation_rho=c["rho"],
            nbm_cache=nbm_cache,
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    return {
        "rows": [_serialize_row(r) for r in cap_log.rows],
        "funnel": json.dumps(dataclasses.asdict(funnel)),
        "candidate": (
            json.dumps(dataclasses.asdict(candidate), default=str)
            if candidate is not None
            else None
        ),
        "logs": handler.records,
        "dedup_tier1_keys": (
            [list(k) for k in dedup if len(k) == 3] if dedup is not None else None
        ),
    }


def _check_target(name: str, case: dict[str, Any], record: dict[str, Any]) -> None:
    expect = case["expect"]
    rows = [json.loads(r) for r in record["rows"]]
    funnel = json.loads(record["funnel"])
    if expect == "PASS":
        assert record["candidate"] is not None, f"{name}: expected a candidate"
        assert rows == [], f"{name}: pass path must not emit rows"
        return
    assert record["candidate"] is None, f"{name}: expected rejection"
    if expect == "FORECAST_GAP":
        assert rows == []
        assert funnel["n_forecast_gap"] == 1
        return
    if case["sinks"]:
        assert [r["reason"] for r in rows] == [expect], (
            f"{name}: drifted to {[r['reason'] for r in rows]}"
        )
    else:
        assert rows == []
    if case["log"] is not None:
        assert any(case["log"] in m for _, _, m in record["logs"]), (
            f"{name}: missing log {case['log']!r}"
        )


@pytest.fixture(autouse=True)
def _frozen_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(we, "datetime", _FrozenDatetime)


@pytest.mark.skipif(UPDATE, reason="regenerating golden fixture")
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_golden_scenario(name: str, base_table: EmosTable, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = json.loads(FIXTURE.read_text())
    case = SCENARIOS[name]
    record = _run(case, base_table, monkeypatch)
    _check_target(name, case, record)
    assert name in golden, f"{name} missing from golden fixture"
    expected = golden[name]
    assert record["rows"] == expected["rows"]
    assert record["funnel"] == expected["funnel"]
    assert record["candidate"] == expected["candidate"]
    assert record["logs"] == expected["logs"]
    assert record["dedup_tier1_keys"] == expected["dedup_tier1_keys"]


@pytest.mark.skipif(UPDATE, reason="regenerating golden fixture")
def test_golden_fixture_has_no_stale_scenarios() -> None:
    golden = json.loads(FIXTURE.read_text())
    assert set(golden) == set(SCENARIOS)


@pytest.mark.skipif(not UPDATE, reason="set UPDATE_EVALUATE_GOLDEN=1 to regenerate")
def test_regenerate_golden(base_table: EmosTable, monkeypatch: pytest.MonkeyPatch) -> None:
    out: dict[str, Any] = {}
    failures: list[str] = []
    for name, case in SCENARIOS.items():
        with monkeypatch.context() as mp:
            mp.setattr(we, "datetime", _FrozenDatetime)
            record = _run(case, base_table, mp)
        try:
            _check_target(name, case, record)
        except AssertionError as exc:
            failures.append(str(exc).splitlines()[0])
        out[name] = record
    assert not failures, "\n".join(failures)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
