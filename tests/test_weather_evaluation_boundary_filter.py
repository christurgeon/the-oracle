"""Integration: weather.boundary_filter_cells suppresses grazer-zone candidates.

Reuses the stable fixture helpers from tests/test_weather_evaluation_nbm_veto.py
(``_make_eval_cfg``, ``_make_scan_candidate``, ``_invoke``) which construct a
minimal ScanCandidate + _EvalConfig + forecast cache for the production
evaluator. The audit-row reason field is ``"reason"`` and holds the
RejectionReason enum; ``extra={...}`` is merged top-level into the row.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from python.analytics.audit_log import RejectionReason, ScanFunnel
from python.common import DailyVariable, Direction
from python.strategies.weather_parsing import ParsedMarket
from tests.test_weather_evaluation_nbm_veto import (
    _invoke,
    _make_eval_cfg,
    _make_scan_candidate,
    _make_sinks_with_capture,
)


def _mia_low_between(floor: float = 70.0, cap: float = 71.0) -> ParsedMarket:
    return ParsedMarket(
        ticker=f"KXLOWTMIA-26MAY30-B{floor:.0f}-{cap:.0f}",
        city="MIA",
        forecast_date=date(2026, 5, 31),
        strike_type=Direction.BETWEEN,
        floor_strike=floor,
        cap_strike=cap,
        daily_variable=DailyVariable.TEMP_MIN,  # → parsed.temp_label == "low"
    )


def _run(parsed, temps, *, boundary_filter_cells, min_calibrated_prob=0.0, **cfg_overrides):
    cfg = replace(
        _make_eval_cfg(
            min_calibrated_prob=min_calibrated_prob,
            block_raw_prob_range=None,
            **cfg_overrides,
        ),
        boundary_filter_cells=boundary_filter_cells,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.30, yes_ask=0.35)
    sinks, capturing = _make_sinks_with_capture()
    funnel = ScanFunnel()
    result = _invoke(sc, parsed, temps, cfg, nbm_cache=None, sinks=sinks, funnel=funnel)
    boundary_rows = [
        r for r in capturing.rows if r.get("reason") == RejectionReason.BOUNDARY_FILTER_BLOCK
    ]
    return result, capturing, funnel, boundary_rows


def test_boundary_filter_disabled_does_not_block():
    """Empty config → no BOUNDARY_FILTER_BLOCK rejections even at d_boundary=0."""
    parsed = _mia_low_between()
    _result, _cap, funnel, boundary_rows = _run(parsed, [70.4] * 6, boundary_filter_cells={})
    assert boundary_rows == []
    assert funnel.n_boundary_filter_block == 0


def test_boundary_filter_enabled_for_mia_low_blocks_grazer():
    """MIA/low + threshold=0.5 → grazer-zone candidate (d_boundary=0) is dropped."""
    parsed = _mia_low_between()
    result, _cap, funnel, boundary_rows = _run(
        parsed, [70.4] * 6, boundary_filter_cells={"MIA/low": 0.5}
    )
    assert result is None
    assert len(boundary_rows) == 1
    assert funnel.n_boundary_filter_block == 1
    row = boundary_rows[0]
    # Self-contained replay state must be present (top-level via extra merge).
    assert row["d_boundary_f"] == pytest.approx(0.0)
    assert row["boundary_filter_threshold_f"] == pytest.approx(0.5)
    assert row["forecast_mean_f"] == pytest.approx(70.4)
    assert row["cell_label"] == "MIA/low"
    assert row["floor_strike"] == 70.0
    assert row["cap_strike"] == 71.0
    assert row["strike_type"] == "between"


def test_boundary_filter_does_not_apply_to_different_cell():
    """Filter on MIA/low does NOT touch NYC/high (different city + temp_label)."""
    parsed = ParsedMarket(
        ticker="KXHIGHNY-26MAY30-B70-71",
        city="NYC",
        forecast_date=date(2026, 5, 31),
        strike_type=Direction.BETWEEN,
        floor_strike=70.0,
        cap_strike=71.0,
        daily_variable=DailyVariable.TEMP_MAX,  # → parsed.temp_label == "high"
    )
    _result, _cap, funnel, boundary_rows = _run(
        parsed, [70.4] * 6, boundary_filter_cells={"MIA/low": 0.5}
    )
    assert boundary_rows == []
    assert funnel.n_boundary_filter_block == 0


def test_boundary_filter_distant_mean_passes():
    """Mean ~5°F below bracket → d_boundary = 4.5°F > threshold → survives filter."""
    parsed = _mia_low_between()
    _result, _cap, funnel, boundary_rows = _run(
        parsed, [65.0] * 6, boundary_filter_cells={"MIA/low": 0.5}
    )
    assert boundary_rows == []
    assert funnel.n_boundary_filter_block == 0


def test_greater_market_not_filtered_even_when_cell_enabled():
    """#6 between-only guard: a greater (threshold) market in an enabled cell is
    NOT subject to the filter, even with the mean on the rounding boundary."""
    parsed = ParsedMarket(
        ticker="KXLOWTMIA-26MAY30-T70",
        city="MIA",
        forecast_date=date(2026, 5, 31),
        strike_type=Direction.ABOVE,  # "greater"
        floor_strike=70.0,
        cap_strike=None,
        daily_variable=DailyVariable.TEMP_MIN,  # → MIA/low
    )
    # mean ~70.5 → boundary at 70.5 → d_boundary would be ~0 IF it were applied.
    _result, _cap, funnel, boundary_rows = _run(
        parsed, [70.5] * 6, boundary_filter_cells={"MIA/low": 0.5}
    )
    assert boundary_rows == []
    assert funnel.n_boundary_filter_block == 0


def test_audit_fields_surface_on_non_block_row():
    """#5 observability: a candidate that passes the filter (distant mean) but is
    rejected downstream still carries d_boundary_f + forecast_mean_f."""
    parsed = _mia_low_between()
    # mean 65 → d_boundary 4.5 > 0.5 (passes filter). High edge_threshold forces
    # a downstream EDGE rejection so a _reject row is emitted.
    _result, capturing, _funnel, boundary_rows = _run(
        parsed,
        [65.0] * 6,
        boundary_filter_cells={"MIA/low": 0.5},
        edge_threshold=0.99,
    )
    assert boundary_rows == []  # not blocked by the boundary filter
    carrying = [r for r in capturing.rows if r.get("d_boundary_f") is not None]
    assert carrying, "expected at least one downstream audit row to carry d_boundary_f"
    row = carrying[0]
    assert row["d_boundary_f"] == pytest.approx(4.5)
    assert row["forecast_mean_f"] == pytest.approx(65.0)


def test_boundary_filter_runs_after_min_calibrated_prob():
    """Documented contract: rows blocked by min_calibrated_prob never reach the
    boundary filter. Regression guard — reordering the gates fails this loud.

    Temps straddle the bracket so model_prob is moderate (~0.33) while the mean
    (70.5) sits in the grazer zone (d_boundary=0). A high floor blocks first.
    """
    parsed = _mia_low_between()
    result, capturing, funnel, boundary_rows = _run(
        parsed,
        [60.0, 65.0, 70.0, 71.0, 76.0, 81.0],  # mean 70.5, ~2/6 in-bracket
        boundary_filter_cells={"MIA/low": 0.5},
        min_calibrated_prob=0.99,
    )
    assert result is None
    cal_blocks = [r for r in capturing.rows if r.get("reason") == RejectionReason.CAL_FLOOR_BLOCK]
    assert len(cal_blocks) == 1
    assert boundary_rows == []
    assert funnel.n_boundary_filter_block == 0
