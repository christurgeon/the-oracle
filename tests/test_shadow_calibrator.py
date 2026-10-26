"""Integration tests for dual-calibrator (live + shadow) logging.

The shadow calibrator runs in parallel with the live one so its output
can be A/B-compared via the ``eval_audit.jsonl`` stream — but it must
*never* influence trading decisions. These tests pin both properties:

1. When a shadow calibrator is configured, every emitted audit row
   carries a ``shadow_prob`` field (or ``None`` for pre-forecast rejects).
2. The live ``model_prob`` reaching ``_MarketCandidate`` is identical
   with or without a shadow calibrator.
3. A shadow calibrator that raises is swallowed — live path continues,
   row shows ``shadow_prob: None``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from python.analytics.audit_log import AuditLogger, AuditSinks, ScanFunnel
from python.data.calibration import IdentityCalibrator
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    DailyVariable,
    Direction,
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    _EvalConfig,
    evaluate_market_pure,
)


def _tomorrow() -> date:
    from datetime import timedelta

    return date.today() + timedelta(days=1)


def _make_scan_candidate(
    *,
    city: str = "NYC",
    strike_type: Direction = Direction.ABOVE,
    floor_strike: float = 55.0,
    cap_strike: float | None = None,
    yes_bid: float = 0.30,
    yes_ask: float = 0.40,
    volume: int = 200,
    open_interest: int = 50,
) -> ScanCandidate:
    parsed = ParsedMarket(
        ticker="KXHIGHNY-99APR01-T55",
        city=city,
        forecast_date=_tomorrow(),
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    price_info = MarketPriceInfo(
        midpoint=(yes_bid + yes_ask) / 2,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        spread=yes_ask - yes_bid,
        yes_bid_size=100,
        yes_ask_size=100,
        volume=volume,
        open_interest=open_interest,
    )
    market = {
        "ticker": parsed.ticker,
        "event_ticker": "KXHIGHNY-99APR01",
        "strike_type": "greater",
        "floor_strike": floor_strike,
        "yes_bid_dollars": str(yes_bid),
        "yes_ask_dollars": str(yes_ask),
        "volume_fp": str(volume),
        "open_interest_fp": str(open_interest),
        "close_time": "2099-12-31T23:59:59Z",
    }
    return ScanCandidate(market=market, parsed=parsed, price_info=price_info)


def _make_forecast_cache(
    city: str = "NYC",
    temps: list[float] | None = None,
) -> dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]:
    if temps is None:
        # 31 members, all well above 55 → raw prob ~1.0.
        temps = [82.0] * 31
    fc = EnsembleForecast(city=city, forecast_date=_tomorrow(), temperatures=temps)
    return {(city, _tomorrow(), DailyVariable.TEMP_MAX): {"gfs_seamless": fc}}


def _make_cfg() -> _EvalConfig:
    return _EvalConfig(
        max_spread=0.20,
        spread_skill_sensitivity=0.0,
        edge_threshold=0.01,  # permissive so evaluate reaches the pass path
        max_edge=0.99,
        kelly_fraction=0.25,
        max_pct=0.02,
        max_take_pct=0.10,
        depth_check_enabled=False,
    )


def _make_sinks(tmp_path: Path) -> AuditSinks:
    return AuditSinks(
        eval_log=AuditLogger(tmp_path / "eval.jsonl"),
        candidate_log=AuditLogger(tmp_path / "cand.jsonl"),
        funnel_log=AuditLogger(tmp_path / "funnel.jsonl"),
    )


def _read_eval(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "eval.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class _FixedCalibrator:
    """Stub calibrator that returns a fixed value regardless of input."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[float, str | None]] = []
        self.is_loaded = True
        self.fallback_reason: str | None = None

    def calibrate(self, raw: float, *, strike_type: str | None = None) -> float:
        self.calls.append((raw, strike_type))
        return self.value


class _RaisingCalibrator:
    is_loaded = True
    fallback_reason: str | None = None

    def calibrate(self, raw: float, *, strike_type: str | None = None) -> float:
        raise RuntimeError("shadow boom")


class TestShadowProbEmitted:
    def test_audit_row_carries_shadow_prob(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)
        live = _FixedCalibrator(0.70)
        shadow = _FixedCalibrator(0.45)

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=live,
            shadow_calibrator=shadow,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-1",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        # Evaluation reached the post-calibration path — the shadow
        # calibrator was called.
        assert len(shadow.calls) == 1
        raw_in, strike_in = shadow.calls[0]
        assert strike_in == "greater", "shadow sees the canonical strike label"

        rows = _read_eval(tmp_path)
        # Pass path does not emit an eval row (candidate audit handles it).
        # So assert either a reject row with shadow_prob or a pass with
        # _MarketCandidate.shadow_prob set.
        if result is not None:
            assert result.shadow_prob == pytest.approx(0.45)
        else:
            assert rows, "expected at least one audit row"
            assert any(r.get("shadow_prob") == pytest.approx(0.45) for r in rows)

    def test_shadow_prob_is_none_when_no_shadow_calibrator(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.70),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-2",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        if result is not None:
            assert result.shadow_prob is None
        rows = _read_eval(tmp_path)
        for row in rows:
            assert row.get("shadow_prob") is None


class TestShadowCalibratorIsolation:
    """The shadow calibrator must not influence the live model probability."""

    def test_live_model_prob_independent_of_shadow(self, tmp_path: Path) -> None:
        cache = _make_forecast_cache()
        cfg = _make_cfg()
        live = _FixedCalibrator(0.70)
        shadow_wild = _FixedCalibrator(0.01)  # would massively change decision

        result_with_shadow = evaluate_market_pure(
            _make_scan_candidate(),
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=cfg,
            calibrator=live,
            shadow_calibrator=shadow_wild,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
        )
        result_without = evaluate_market_pure(
            _make_scan_candidate(),
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=cfg,
            calibrator=live,
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
        )

        # Both paths hit the same live-calibrated model_prob; the shadow
        # calibrator's wildly different output must not bleed through.
        if result_with_shadow is None or result_without is None:
            assert result_with_shadow is None and result_without is None
        else:
            assert result_with_shadow.model_prob == pytest.approx(result_without.model_prob)
            assert result_with_shadow.effective_edge == pytest.approx(result_without.effective_edge)
            assert result_with_shadow.contracts == result_without.contracts


class TestShadowCalibratorFailureMode:
    def test_exception_swallowed_shadow_prob_none(self, tmp_path: Path) -> None:
        """Shadow calibrator raising must not break the live path."""
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.70),
            shadow_calibrator=_RaisingCalibrator(),
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-raise",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        # The live pipeline still produced a decision (or cleanly rejected)
        # — the shadow exception did not propagate.
        if result is not None:
            assert result.shadow_prob is None
        rows = _read_eval(tmp_path)
        for row in rows:
            assert row.get("shadow_prob") is None


class TestIdentityShadow:
    """A shadow calibrator that echoes its input is a valid stub."""

    def test_identity_shadow_matches_raw(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)
        live = _FixedCalibrator(0.70)

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=live,
            shadow_calibrator=IdentityCalibrator(),
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-ident",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        # With 31 members at 82°F and a 55°F floor, raw = 1.0.
        if result is not None:
            assert result.shadow_prob == pytest.approx(1.0)
        rows = _read_eval(tmp_path)
        for row in rows:
            sp = row.get("shadow_prob")
            if sp is not None:
                assert sp == pytest.approx(1.0)


class TestCalibratorDisagreementGate:
    """R1 sign-flip abort: raw and calibrated on opposite sides of mid must reject."""

    def test_sign_flip_rejects_with_calibrator_disagreement(self, tmp_path: Path) -> None:
        # 25/31 members above 55°F → raw ≈ 0.806 (inside [0.05, 0.95] gate).
        # Mid = (0.30 + 0.40) / 2 = 0.35.
        # Calibrator returns 0.10 → cal < mid < raw → sign-flip.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache(temps=[82.0] * 25 + [50.0] * 6)
        sinks = _make_sinks(tmp_path)

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.10),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-flip",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        assert result is None, "sign-flip must not produce a passing candidate"
        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CALIBRATOR_DISAGREEMENT" in reasons, (
            f"expected CALIBRATOR_DISAGREEMENT in audit reasons, got {reasons}"
        )

    def test_same_side_does_not_trip_r1(self, tmp_path: Path) -> None:
        # Raw ≈ 1.0, calibrator returns 0.70 → both above mid 0.35 → no flip.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.70),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-same-side",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CALIBRATOR_DISAGREEMENT" not in reasons, (
            f"same-side raw/cal must not trip R1, got {reasons}"
        )

    def test_cal_at_midpoint_is_not_a_flip(self, tmp_path: Path) -> None:
        # Mid = 0.35, calibrator returns exactly 0.35 → tie, not a flip.
        # Raw ≈ 1.0. R1 must not abort.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.35),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-tie",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CALIBRATOR_DISAGREEMENT" not in reasons, (
            f"tie (cal == mid) must not trip R1, got {reasons}"
        )

    def test_funnel_counter_increments(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache(temps=[82.0] * 25 + [50.0] * 6)
        sinks = _make_sinks(tmp_path)
        funnel = ScanFunnel()

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),
            calibrator=_FixedCalibrator(0.10),  # sign-flip
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-funnel",
            sinks=sinks,
            funnel=funnel,
        )

        assert funnel.n_calibrator_disagreement == 1


class TestPlateauBlockGate:
    """Strategy-layer filter: skip trades whose raw ensemble prob lands in
    the configured plateau range regardless of calibrator output."""

    def _cfg_with_plateau(self) -> _EvalConfig:
        from dataclasses import replace

        return replace(_make_cfg(), block_raw_prob_range=(0.43, 0.87))

    def test_plateau_raw_rejects_with_plateau_block(self, tmp_path: Path) -> None:
        # 20/31 members above 55°F → raw ≈ 0.645, inside [0.43, 0.87].
        sc = _make_scan_candidate()
        cache = _make_forecast_cache(temps=[82.0] * 20 + [50.0] * 11)
        sinks = _make_sinks(tmp_path)
        funnel = ScanFunnel()

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=self._cfg_with_plateau(),
            calibrator=_FixedCalibrator(0.20),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-plateau",
            sinks=sinks,
            funnel=funnel,
        )

        assert result is None
        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "PLATEAU_BLOCK" in reasons, (
            f"expected PLATEAU_BLOCK in audit reasons, got {reasons}"
        )
        assert funnel.n_plateau_block == 1

    def test_raw_outside_plateau_not_blocked(self, tmp_path: Path) -> None:
        # 30/31 members above 55°F → raw ≈ 0.968, above the 0.87 ceiling.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache(temps=[82.0] * 30 + [50.0] * 1)
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=self._cfg_with_plateau(),
            calibrator=_FixedCalibrator(0.90),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-above-plateau",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "PLATEAU_BLOCK" not in reasons

    def test_none_range_disables_filter(self, tmp_path: Path) -> None:
        # Raw in plateau band but filter disabled → must not reject with PLATEAU_BLOCK.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache(temps=[82.0] * 20 + [50.0] * 11)
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(),  # block_raw_prob_range defaults to None
            calibrator=_FixedCalibrator(0.20),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-disabled",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "PLATEAU_BLOCK" not in reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
