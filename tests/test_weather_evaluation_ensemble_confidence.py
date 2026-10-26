"""Skip ``ensemble_confidence`` discount for EMOS cells (Tier 2 B, 2026-05-19).

EMOS already prices ensemble spread via its predictive ``σ = √(c + d · ens_std²)``.
Multiplying ``raw_edge`` by ``1 / (1 + k · ens_std)`` on top double-counts spread
uncertainty. This module verifies the conditional at both call sites:

  1. Stage 1 pre-filter (``python/strategies/weather_evaluation.py``): the
     per-cell EMOS dispatch sets ``live_calibrator_kind == "emos_raw"`` and the
     ``ensemble_confidence`` factor must collapse to 1.0.

  2. Stage 2 depth-check recompute (``python/strategies/weather_edge.py``): the
     VWAP ``fill_edge`` calculation also must skip the discount when the
     candidate carries ``live_calibrator_kind == "emos_raw"``.

The condition is on the dispatcher decision (``"emos_raw"``), NOT on cell
membership in ``emos_enabled_cells``. ``isotonic_fallback`` rows (cell listed
but EMOS didn't compute) still receive the standard discount because they're
on the isotonic path.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.audit_log import AuditSinks, ScanFunnel
from python.analytics.sizing import SizingResult
from python.common import DailyVariable, Direction, Side
from python.data.emos import EmosTable
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    WeatherEdgeStrategy,
    _EvalConfig,
    _MarketCandidate,
    evaluate_market_pure,
)

# ---------------------------------------------------------------------------
# Stage 1 fixtures — modeled on tests/test_weather_evaluation_emos_sigma_gate.py
# ---------------------------------------------------------------------------


def _make_eval_cfg(**overrides: Any) -> _EvalConfig:
    base: dict[str, Any] = dict(
        max_spread=0.50,  # generous — we want the candidate to survive Stage 1
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
        emos_runtime_sigma_threshold=9999.0,  # disable the σ-gate for these tests
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
    """Hermetic EMOS table with per-cell coefficient overrides."""
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
    model_prob: float = 0.5,
) -> dict[str, Any]:
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    bias_corrections = {fkey: 0.0}
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
        weather_client=MagicMock(compute_model_probability=MagicMock(return_value=model_prob)),
        bankroll=10_000.0,
    )


# Temps straddle strike=85 → ensemble mean ≈ 85, ens_std ≈ 1.58 (sample std).
_STRADDLE_85_TEMPS = [83.0, 84.0, 85.0, 86.0, 87.0]


# ---------------------------------------------------------------------------
# Stage 1 tests
# ---------------------------------------------------------------------------


def test_ensemble_confidence_one_for_emos_cells(tmp_path: Path) -> None:
    """EMOS-live city/var with ``spread_skill_sensitivity > 0``: the returned
    ``_MarketCandidate.ensemble_confidence`` must be 1.0 because EMOS already
    prices spread via (c, d).
    """
    parsed = _make_parsed(city="MIA", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    # Force a meaningful raw_edge: weather_client returns 0.65, market mid=0.25.
    # → YES side, raw_edge = |0.65 - 0.30| = 0.35 (exec_price = yes_ask).
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0, model_prob=0.65)
    table = _make_emos_table_with_cells(
        tmp_path,
        # Well-behaved coefficients so σ-gate (even if armed) won't fire,
        # and ``emos_probability`` returns a finite swap value.
        cells={"MIA/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3}},
    )
    cfg = _make_eval_cfg(spread_skill_sensitivity=0.15)
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
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    assert result is not None, (
        f"EMOS candidate should pass; rejection rows: {[r.get('reason') for r in cap.rows]}"
    )
    assert result.live_calibrator_kind == "emos_raw"
    assert result.ensemble_confidence == 1.0, (
        f"EMOS cell must collapse ensemble_confidence to 1.0 "
        f"(got {result.ensemble_confidence!r}); EMOS already prices spread "
        f"via (c, d)."
    )


def test_ensemble_confidence_discounted_for_isotonic_cells(tmp_path: Path) -> None:
    """Non-EMOS-listed cell with ``spread_skill_sensitivity > 0`` and
    ``ensemble_std > 0`` must apply the legacy ``1 / (1 + k · σ)`` discount
    (i.e., ``ensemble_confidence < 1.0``).
    """
    parsed = _make_parsed(city="CHI", var=DailyVariable.TEMP_MAX)
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0, model_prob=0.65)
    # CHI/high coefficients present in the table but CHI/high is NOT in
    # emos_enabled_cells → dispatcher leaves model_prob as isotonic/identity.
    table = _make_emos_table_with_cells(
        tmp_path,
        cells={"CHI/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3}},
    )
    cfg = _make_eval_cfg(spread_skill_sensitivity=0.15)
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
        emos_enabled_cells={"MIA/high"},  # NOT CHI/high
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=funnel,
    )

    assert result is not None, (
        f"Isotonic candidate should pass; rejection rows: {[r.get('reason') for r in cap.rows]}"
    )
    assert result.live_calibrator_kind != "emos_raw"
    # k=0.15, σ=2.0 → expected confidence = 1 / (1 + 0.15 * 2.0) = 1/1.3 ≈ 0.7692
    assert result.ensemble_confidence < 1.0, (
        f"Non-EMOS cell with σ>0 must still discount (got {result.ensemble_confidence!r})."
    )
    assert result.ensemble_confidence == pytest.approx(1.0 / (1.0 + 0.15 * 2.0))


def test_ensemble_confidence_one_when_k_zero(tmp_path: Path) -> None:
    """With ``spread_skill_sensitivity = 0`` both calibrators short-circuit to
    1.0 (legacy behavior). The EMOS conditional must not affect this.
    """
    # --- EMOS path ---
    parsed_emos = _make_parsed(city="MIA", var=DailyVariable.TEMP_MAX)
    sc_emos = _make_scan_candidate(parsed_emos)
    kwargs_emos = _build_eval_kwargs(
        parsed_emos, _STRADDLE_85_TEMPS, ens_std_override=2.0, model_prob=0.65
    )
    table = _make_emos_table_with_cells(
        tmp_path,
        cells={
            "MIA/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3},
            "CHI/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3},
        },
    )
    cfg = _make_eval_cfg(spread_skill_sensitivity=0.0)
    sinks_e, cap_e = _make_sinks_with_capture()
    result_emos = evaluate_market_pure(
        sc_emos,
        kwargs_emos["forecast_cache"],
        kwargs_emos["bias_corrections"],
        kwargs_emos["raw_ensemble_stds"],
        kwargs_emos["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs_emos["weights"],
        weather_client=kwargs_emos["weather_client"],
        bankroll=kwargs_emos["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs_emos["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks_e,
        funnel=ScanFunnel(),
    )
    assert result_emos is not None, (
        f"EMOS path; rejections: {[r.get('reason') for r in cap_e.rows]}"
    )
    assert result_emos.live_calibrator_kind == "emos_raw"
    assert result_emos.ensemble_confidence == 1.0

    # --- Isotonic path ---
    parsed_iso = _make_parsed(city="CHI", var=DailyVariable.TEMP_MAX)
    sc_iso = _make_scan_candidate(parsed_iso)
    kwargs_iso = _build_eval_kwargs(
        parsed_iso, _STRADDLE_85_TEMPS, ens_std_override=2.0, model_prob=0.65
    )
    sinks_i, cap_i = _make_sinks_with_capture()
    result_iso = evaluate_market_pure(
        sc_iso,
        kwargs_iso["forecast_cache"],
        kwargs_iso["bias_corrections"],
        kwargs_iso["raw_ensemble_stds"],
        kwargs_iso["raw_ensemble_medians"],
        cfg,
        calibrator=None,
        weights=kwargs_iso["weights"],
        weather_client=kwargs_iso["weather_client"],
        bankroll=kwargs_iso["bankroll"],
        emos_table=table,
        raw_member_sink=kwargs_iso["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells={"MIA/high"},
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks_i,
        funnel=ScanFunnel(),
    )
    assert result_iso is not None, (
        f"Isotonic path; rejections: {[r.get('reason') for r in cap_i.rows]}"
    )
    assert result_iso.live_calibrator_kind != "emos_raw"
    assert result_iso.ensemble_confidence == 1.0


# ---------------------------------------------------------------------------
# Stage 2 (depth-check VWAP recompute) test
# ---------------------------------------------------------------------------


def _build_stage2_candidate(
    *,
    live_calibrator_kind: str,
    ensemble_confidence: float,
    model_prob: float = 0.65,
    yes_bid: float = 0.20,
    yes_ask: float = 0.30,
    side: Side = Side.YES,
    contracts: int = 50,
) -> _MarketCandidate:
    """Build a ``_MarketCandidate`` suitable for direct
    ``_apply_depth_adjustment`` exercise."""
    today = datetime.now(UTC).date() + timedelta(days=1)
    parsed = ParsedMarket(
        ticker=f"KXHIGHMIA-99MAY15H1600-A{int(yes_bid * 100):02d}",
        city="MIA",
        forecast_date=today,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
        cap_strike=None,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    price_info = MarketPriceInfo(
        midpoint=(yes_bid + yes_ask) / 2,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        spread=yes_ask - yes_bid,
        yes_bid_size=500,
        yes_ask_size=500,
        volume=1000,
        open_interest=500,
    )
    forecast = EnsembleForecast(city="MIA", forecast_date=today, temperatures=[85.0] * 31)
    sizing = SizingResult(
        side=side,
        dollars=300.0,
        kelly_raw=0.05,
        kelly_fraction_used=0.05,
        capped=False,
    )
    market: dict[str, Any] = {
        "ticker": parsed.ticker,
        "event_ticker": "KXHIGHMIA-99MAY15H1600",
        "strike_type": "above",
        "floor_strike": 85.0,
        "cap_strike": None,
        "yes_bid_dollars": f"{yes_bid:.2f}",
        "yes_ask_dollars": f"{yes_ask:.2f}",
    }
    exec_price = yes_ask if side == Side.YES else 1.0 - yes_bid
    return _MarketCandidate(
        parsed=parsed,
        market=market,
        forecast=forecast,
        model_prob=model_prob,
        correction_f=0.0,
        price_info=price_info,
        raw_edge=abs(model_prob - price_info.midpoint),
        ensemble_confidence=ensemble_confidence,
        effective_edge=0.10,
        sizing=sizing,
        contracts=contracts,
        exec_price=exec_price,
        per_model_probs={"gfs_seamless": model_prob},
        ensemble_std_pooled=1.5,
        spread_inflation_factor=1.0,
        raw_ensemble_prob=model_prob,
        strike_abs_z=3.0,
        live_calibrator_kind=live_calibrator_kind,
    )


def _make_weather_edge_strategy(tmp_path: Path, *, kalshi_client: Any) -> WeatherEdgeStrategy:
    """Construct a minimal ``WeatherEdgeStrategy`` for ``_apply_depth_adjustment``
    coverage. Stubs out Settings so the test stays hermetic.
    """
    strat = WeatherEdgeStrategy(
        config={
            "orderbook_depth_check": True,
            "min_book_depth": 5,
            "book_depth_levels": 10,
            "max_take_pct": 1.0,
        },
        kalshi_client=kalshi_client,
        weather_client=MagicMock(),
        bankroll=10_000.0,
    )
    strat._settings = SimpleNamespace(  # type: ignore[assignment]
        state_dir=str(tmp_path),
        risk=SimpleNamespace(kelly_fraction=0.25, max_single_market_pct=0.20),
        execution=SimpleNamespace(mode="shadow"),
    )
    return strat


@pytest.mark.asyncio
async def test_stage_2_depth_check_skips_confidence_for_emos(tmp_path: Path) -> None:
    """Stage 2 depth-check VWAP recompute on an EMOS row must use multiplier
    1.0 (i.e., ``fill_edge = (model_prob - fill_price) - fill_fee``), NOT the
    discounted version (``... * candidate.ensemble_confidence - fill_fee``).

    Verifies the change at ``python/strategies/weather_edge.py:1373, 1377``.
    """
    # YES bid at $0.50 with 500 contracts → for a YES buy, our fill walks
    # the no_dollars side; we put a NO bid at $0.70 to give us a YES ask
    # at $0.30 with deep liquidity. That gives a deterministic fill_price.
    kalshi = MagicMock()
    kalshi.get_orderbook = AsyncMock(
        return_value={
            "orderbook_fp": {
                # NO bid at $0.70 with 500 contracts ⇒ YES ask at $0.30
                "no_dollars": [["0.70", "500"]],
                "yes_dollars": [["0.20", "500"]],
            }
        }
    )
    strat = _make_weather_edge_strategy(tmp_path, kalshi_client=kalshi)

    model_prob = 0.65
    # Stage 2 fill_price (YES ask in our space) = 0.30.
    # Plain fill_edge (no confidence factor) = (0.65 - 0.30) - fee.
    # Discounted fill_edge = ((0.65 - 0.30) * 0.5) - fee = 0.175 - fee.
    discount_factor = 0.5  # explicit, large enough to make the diff visible
    cand_emos = _build_stage2_candidate(
        live_calibrator_kind="emos_raw",
        ensemble_confidence=discount_factor,
        model_prob=model_prob,
        side=Side.YES,
    )
    cfg = _EvalConfig(
        max_spread=0.50,
        spread_skill_sensitivity=0.15,
        edge_threshold=0.001,
        max_edge=0.99,
        kelly_fraction=0.25,
        max_pct=0.20,
        max_take_pct=1.0,
        depth_check_enabled=True,
        min_raw_prob_for_between=0.0,
        min_distance_from_strike_std=0.0,
        max_fill_slippage=None,
        block_raw_prob_range=None,
        max_model_market_gap_near_term=None,
        emos_runtime_sigma_threshold=9999.0,
    )

    result = await strat._apply_depth_adjustment(cand_emos, cfg)
    assert result is not None, "EMOS depth-check should not reject"
    effective_edge, _sizing, _contracts, _depth_capped, depth_result = result
    assert depth_result is not None
    fill_price = depth_result.fill_price
    # Recompute the fee the same way the production path does.
    from python.analytics.fees import fee_per_contract as kalshi_fee

    fill_fee = kalshi_fee(fill_price)
    expected_emos_edge = (model_prob - fill_price) - fill_fee
    discounted_edge = (model_prob - fill_price) * discount_factor - fill_fee
    assert effective_edge == pytest.approx(expected_emos_edge), (
        f"EMOS fill_edge must skip ensemble_confidence; got "
        f"{effective_edge!r}, expected {expected_emos_edge!r} "
        f"(discounted would be {discounted_edge!r})"
    )

    # --- And the isotonic case keeps the discount, for parity proof. ---
    cand_iso = _build_stage2_candidate(
        live_calibrator_kind="isotonic",
        ensemble_confidence=discount_factor,
        model_prob=model_prob,
        side=Side.YES,
    )
    kalshi_iso = MagicMock()
    kalshi_iso.get_orderbook = AsyncMock(
        return_value={
            "orderbook_fp": {
                "no_dollars": [["0.70", "500"]],
                "yes_dollars": [["0.20", "500"]],
            }
        }
    )
    strat_iso = _make_weather_edge_strategy(tmp_path, kalshi_client=kalshi_iso)
    result_iso = await strat_iso._apply_depth_adjustment(cand_iso, cfg)
    assert result_iso is not None
    iso_edge, _s, _c, _dc, _dr = result_iso
    assert iso_edge == pytest.approx(discounted_edge), (
        f"Isotonic fill_edge must preserve confidence discount; got {iso_edge!r}"
    )


def test_ensemble_confidence_discounted_for_isotonic_fallback(tmp_path: Path) -> None:
    """``isotonic_fallback`` rows (cell listed in ``emos_enabled_cells`` but
    EMOS didn't compute this scan — e.g., lead > 2) MUST still receive the
    standard ``1 / (1 + k · σ)`` discount because they're back on the
    isotonic path. Per spec invariant: the conditional keys on
    ``live_calibrator_kind == "emos_raw"``, NOT on cell membership.
    """
    # Forecast date 10d out → lead > 2 → emos_prob_raw stays None →
    # live_calibrator_kind is set to "isotonic_fallback".
    fd_far = datetime.now(UTC).date() + timedelta(days=10)
    parsed = ParsedMarket(
        ticker=f"KXHIGHMIA-{fd_far.isoformat()}-T85",
        city="MIA",
        forecast_date=fd_far,
        strike_type=Direction.ABOVE,
        floor_strike=85.0,
        cap_strike=None,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    sc = _make_scan_candidate(parsed)
    kwargs = _build_eval_kwargs(
        parsed, _STRADDLE_85_TEMPS, ens_std_override=2.0, model_prob=0.65
    )
    table = _make_emos_table_with_cells(
        tmp_path, cells={"MIA/high": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3}}
    )
    cfg = _make_eval_cfg(spread_skill_sensitivity=0.15)
    sinks, cap = _make_sinks_with_capture()

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
        emos_enabled_cells={"MIA/high"},  # cell IS listed
        isotonic_artifact_hash="iso-test-hash",
        sinks=sinks,
        funnel=ScanFunnel(),
    )
    assert result is not None, (
        f"isotonic_fallback row should pass; rejections: "
        f"{[r.get('reason') for r in cap.rows]}"
    )
    assert result.live_calibrator_kind == "isotonic_fallback", (
        f"test premise broken — expected isotonic_fallback, got "
        f"{result.live_calibrator_kind}"
    )
    # k=0.15, σ=2.0 → 1 / 1.3. Same discount as a pure isotonic cell.
    assert result.ensemble_confidence == pytest.approx(1.0 / (1.0 + 0.15 * 2.0)), (
        f"isotonic_fallback must keep the discount; got "
        f"{result.ensemble_confidence!r}"
    )


@pytest.mark.asyncio
async def test_stage_2_depth_check_skips_confidence_for_emos_no_side(
    tmp_path: Path,
) -> None:
    """NO-side parity for the Stage 2 EMOS-skip fix. Mirrors the YES-side
    test above but exercises the ``else`` branch at
    ``python/strategies/weather_edge.py:1455``.
    """
    kalshi = MagicMock()
    # NO-side fill: walker hits the YES-bid side (NO buyer pays 1 - yes_bid).
    # yes_dollars at $0.30 ⇒ NO ask ≈ $0.30 fill price for a NO buyer.
    kalshi.get_orderbook = AsyncMock(
        return_value={
            "orderbook_fp": {
                "yes_dollars": [["0.50", "500"]],  # NO ask = 0.50
                "no_dollars": [["0.40", "500"]],
            }
        }
    )
    strat = _make_weather_edge_strategy(tmp_path, kalshi_client=kalshi)

    # Clean NO edge: (1 - 0.15) - 0.50 = 0.35 before fees — won't trigger
    # the post-depth Kelly side-flip guard at weather_edge.py:1515.
    model_prob = 0.15
    discount_factor = 0.5
    cand = _build_stage2_candidate(
        live_calibrator_kind="emos_raw",
        ensemble_confidence=discount_factor,
        model_prob=model_prob,
        yes_bid=0.40,
        yes_ask=0.55,
        side=Side.NO,
    )
    cfg = _EvalConfig(
        max_spread=0.50,
        spread_skill_sensitivity=0.15,
        edge_threshold=-0.99,  # accept any edge; we're checking the formula
        max_edge=0.99,
        kelly_fraction=0.25,
        max_pct=0.20,
        max_take_pct=1.0,
        depth_check_enabled=True,
        min_raw_prob_for_between=0.0,
        min_distance_from_strike_std=0.0,
        max_fill_slippage=None,
        block_raw_prob_range=None,
        max_model_market_gap_near_term=None,
        emos_runtime_sigma_threshold=9999.0,
    )

    result = await strat._apply_depth_adjustment(cand, cfg)
    assert result is not None, "EMOS NO-side depth-check should not reject"
    effective_edge, _sizing, _contracts, _depth_capped, depth_result = result
    assert depth_result is not None
    from python.analytics.fees import fee_per_contract as kalshi_fee

    fill_price = depth_result.fill_price
    fill_fee = kalshi_fee(fill_price)
    expected_emos_edge = ((1.0 - model_prob) - fill_price) - fill_fee
    discounted_edge = ((1.0 - model_prob) - fill_price) * discount_factor - fill_fee
    assert effective_edge == pytest.approx(expected_emos_edge), (
        f"EMOS NO-side fill_edge must skip ensemble_confidence; got "
        f"{effective_edge!r}, expected {expected_emos_edge!r} "
        f"(discounted would be {discounted_edge!r})"
    )
