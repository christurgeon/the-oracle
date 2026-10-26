"""Tests for the calibrated-probability floor block (``min_calibrated_prob``).

The block rejects candidates whose post-calibration ``model_prob`` falls
below the configured floor. Live + paper modes apply the floor; shadow
mode is exempt (the bin remains observable for validation).

See memory ``cal_floor_block_2026-05-07.md`` for the post-mortem.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.audit_log import AuditLogger, AuditSinks, ScanFunnel
from python.common import DailyVariable, Direction, ExecutionMode
from python.data.weather import EnsembleForecast
from python.settings import Settings
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    WeatherEdgeStrategy,
    _EvalConfig,
    _parse_min_calibrated_prob,
    evaluate_market_pure,
)

# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_shadow_calibrator.py).
# ---------------------------------------------------------------------------


def _tomorrow() -> date:
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


def _make_cfg(**overrides: Any) -> _EvalConfig:
    base = dict(
        max_spread=0.20,
        spread_skill_sensitivity=0.0,
        edge_threshold=0.01,  # permissive
        max_edge=0.99,
        kelly_fraction=0.25,
        max_pct=0.02,
        max_take_pct=0.10,
        depth_check_enabled=False,
    )
    base.update(overrides)
    return _EvalConfig(**base)


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


# ---------------------------------------------------------------------------
# Pure-function (evaluate_market_pure) tests.
# ---------------------------------------------------------------------------


class TestFloorBlockPureFunction:
    """Pure-function-level: floor compare against post-calibration prob."""

    def test_floor_block_rejects_below_floor(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)
        funnel = ScanFunnel()

        result = evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(min_calibrated_prob=0.10),
            calibrator=_FixedCalibrator(0.05),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-floor-below",
            sinks=sinks,
            funnel=funnel,
        )

        assert result is None
        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CAL_FLOOR_BLOCK" in reasons, (
            f"expected CAL_FLOOR_BLOCK in audit reasons, got {reasons}"
        )

    def test_floor_block_passes_at_floor(self, tmp_path: Path) -> None:
        # cal_prob == floor → not rejected for floor reason (>= passes).
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(min_calibrated_prob=0.10),
            calibrator=_FixedCalibrator(0.10),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-floor-at",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CAL_FLOOR_BLOCK" not in reasons, (
            f"floor at exactly {0.10} must NOT trigger CAL_FLOOR_BLOCK; got {reasons}"
        )

    def test_floor_block_passes_above_floor(self, tmp_path: Path) -> None:
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(min_calibrated_prob=0.10),
            calibrator=_FixedCalibrator(0.15),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-floor-above",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CAL_FLOOR_BLOCK" not in reasons

    def test_floor_block_disabled_when_none(self, tmp_path: Path) -> None:
        # cal_prob below typical floor but min_calibrated_prob=None → no floor reject.
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(min_calibrated_prob=None),
            calibrator=_FixedCalibrator(0.05),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-floor-disabled",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        reasons = [r.get("reason") for r in rows]
        assert "CAL_FLOOR_BLOCK" not in reasons

    def test_floor_block_audit_row_carries_calibrated_prob(self, tmp_path: Path) -> None:
        """Verify the eval_audit row records the calibrated prob that
        triggered the rejection (downstream dashboards depend on it)."""
        sc = _make_scan_candidate()
        cache = _make_forecast_cache()
        sinks = _make_sinks(tmp_path)

        evaluate_market_pure(
            sc,
            cache,
            bias_corrections={},
            raw_ensemble_stds={},
            raw_ensemble_medians={},
            cfg=_make_cfg(min_calibrated_prob=0.10),
            calibrator=_FixedCalibrator(0.05),
            shadow_calibrator=None,
            weights={"gfs_seamless": 1.0},
            weather_client=MagicMock(),
            bankroll=1_000.0,
            scan_id="scan-floor-audit",
            sinks=sinks,
            funnel=ScanFunnel(),
        )

        rows = _read_eval(tmp_path)
        floor_rows = [r for r in rows if r.get("reason") == "CAL_FLOOR_BLOCK"]
        assert len(floor_rows) == 1
        assert floor_rows[0].get("calibrated_prob") == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Parser tests.
# ---------------------------------------------------------------------------


class TestParseMinCalibratedProb:
    def test_parse_min_calibrated_prob_none(self) -> None:
        assert _parse_min_calibrated_prob(None) is None

    def test_parse_min_calibrated_prob_valid_float(self) -> None:
        assert _parse_min_calibrated_prob(0.10) == pytest.approx(0.10)

    def test_parse_min_calibrated_prob_zero_ok(self) -> None:
        assert _parse_min_calibrated_prob(0.0) == pytest.approx(0.0)

    def test_parse_min_calibrated_prob_one_ok(self) -> None:
        assert _parse_min_calibrated_prob(1.0) == pytest.approx(1.0)

    def test_parse_min_calibrated_prob_too_high_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_min_calibrated_prob(1.5)

    def test_parse_min_calibrated_prob_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_min_calibrated_prob(-0.1)


# ---------------------------------------------------------------------------
# Strategy-level (mode-aware) tests — verify generate_signals wires the
# floor only in live + paper modes; shadow gets None.
# ---------------------------------------------------------------------------


def _bare_strategy_config(min_cal: float | None = 0.10) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "edge_threshold": 0.07,
        "max_edge": 0.30,
        "cities": ["NYC"],
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        "calibration_type": "none",
        "candidate_audit_log": False,
        "eval_audit_log": False,
        "scan_funnel_log": False,
        "orderbook_depth_check": False,
    }
    if min_cal is not None:
        cfg["min_calibrated_prob"] = min_cal
    return cfg


def _make_settings(mode: ExecutionMode) -> Settings:
    """Build a Settings with the requested execution mode.

    ``Settings()`` defaults to shadow; flipping the nested
    ``execution.mode`` reaches the property used by ``generate_signals``.
    """
    s = Settings()
    s.execution.mode = mode.value  # type: ignore[assignment]
    return s


@pytest.mark.asyncio
async def test_strategy_live_mode_applies_floor(tmp_path: Path) -> None:
    """LIVE mode → cfg.min_calibrated_prob == config value."""
    captured: dict[str, _EvalConfig] = {}

    strat = WeatherEdgeStrategy(
        config=_bare_strategy_config(min_cal=0.10),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
        settings=_make_settings(ExecutionMode.LIVE),
    )

    # Patch out the heavy parts of generate_signals: scan returns empty so
    # we exit before any I/O, but cfg is constructed first.
    async def fake_scan(*a: Any, **kw: Any) -> list[Any]:
        return []

    async def fake_prefetch(*a: Any, **kw: Any) -> tuple[dict, dict, dict, dict]:
        return ({}, {}, {}, {})

    # Capture cfg via _evaluate_market.
    def fake_eval(sc: Any, *a: Any, **kw: Any) -> Any:
        captured["cfg"] = a[4] if len(a) > 4 else kw.get("cfg")
        return None

    from unittest.mock import patch

    with (
        patch.object(strat, "scan", side_effect=fake_scan),
        patch.object(strat, "_prefetch_forecasts", side_effect=fake_prefetch),
        patch.object(strat, "_evaluate_market", side_effect=fake_eval),
    ):
        # Inject a fake market so _evaluate_market gets called once.
        fake_market = ScanCandidate(
            market={"ticker": "X", "event_ticker": "EVT"},
            parsed=ParsedMarket(
                ticker="X",
                city="NYC",
                forecast_date=_tomorrow(),
                strike_type=Direction.ABOVE,
                floor_strike=55.0,
                cap_strike=None,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            price_info=MarketPriceInfo(
                midpoint=0.5,
                yes_bid=0.45,
                yes_ask=0.55,
                spread=0.10,
                yes_bid_size=10,
                yes_ask_size=10,
                volume=10,
                open_interest=10,
            ),
        )

        async def fake_scan_one(*a: Any, **kw: Any) -> list[ScanCandidate]:
            return [fake_market]

        with patch.object(strat, "scan", side_effect=fake_scan_one):
            await strat.generate_signals()

    cfg = captured.get("cfg")
    assert cfg is not None, "expected _evaluate_market to be called and cfg captured"
    assert cfg.min_calibrated_prob == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_strategy_paper_mode_applies_floor(tmp_path: Path) -> None:
    captured: dict[str, _EvalConfig] = {}

    strat = WeatherEdgeStrategy(
        config=_bare_strategy_config(min_cal=0.10),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
        settings=_make_settings(ExecutionMode.PAPER),
    )

    async def fake_prefetch(*a: Any, **kw: Any) -> tuple[dict, dict, dict, dict]:
        return ({}, {}, {}, {})

    def fake_eval(sc: Any, *a: Any, **kw: Any) -> Any:
        captured["cfg"] = a[4] if len(a) > 4 else kw.get("cfg")
        return None

    fake_market = ScanCandidate(
        market={"ticker": "X", "event_ticker": "EVT"},
        parsed=ParsedMarket(
            ticker="X",
            city="NYC",
            forecast_date=_tomorrow(),
            strike_type=Direction.ABOVE,
            floor_strike=55.0,
            cap_strike=None,
            daily_variable=DailyVariable.TEMP_MAX,
        ),
        price_info=MarketPriceInfo(
            midpoint=0.5,
            yes_bid=0.45,
            yes_ask=0.55,
            spread=0.10,
            yes_bid_size=10,
            yes_ask_size=10,
            volume=10,
            open_interest=10,
        ),
    )

    async def fake_scan_one(*a: Any, **kw: Any) -> list[ScanCandidate]:
        return [fake_market]

    from unittest.mock import patch

    with (
        patch.object(strat, "scan", side_effect=fake_scan_one),
        patch.object(strat, "_prefetch_forecasts", side_effect=fake_prefetch),
        patch.object(strat, "_evaluate_market", side_effect=fake_eval),
    ):
        await strat.generate_signals()

    cfg = captured.get("cfg")
    assert cfg is not None
    assert cfg.min_calibrated_prob == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_strategy_shadow_mode_skips_floor(tmp_path: Path) -> None:
    """SHADOW mode → cfg.min_calibrated_prob is None even if config sets it."""
    captured: dict[str, _EvalConfig] = {}

    strat = WeatherEdgeStrategy(
        config=_bare_strategy_config(min_cal=0.10),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
        settings=_make_settings(ExecutionMode.SHADOW),
    )

    async def fake_prefetch(*a: Any, **kw: Any) -> tuple[dict, dict, dict, dict]:
        return ({}, {}, {}, {})

    def fake_eval(sc: Any, *a: Any, **kw: Any) -> Any:
        captured["cfg"] = a[4] if len(a) > 4 else kw.get("cfg")
        return None

    fake_market = ScanCandidate(
        market={"ticker": "X", "event_ticker": "EVT"},
        parsed=ParsedMarket(
            ticker="X",
            city="NYC",
            forecast_date=_tomorrow(),
            strike_type=Direction.ABOVE,
            floor_strike=55.0,
            cap_strike=None,
            daily_variable=DailyVariable.TEMP_MAX,
        ),
        price_info=MarketPriceInfo(
            midpoint=0.5,
            yes_bid=0.45,
            yes_ask=0.55,
            spread=0.10,
            yes_bid_size=10,
            yes_ask_size=10,
            volume=10,
            open_interest=10,
        ),
    )

    async def fake_scan_one(*a: Any, **kw: Any) -> list[ScanCandidate]:
        return [fake_market]

    from unittest.mock import patch

    with (
        patch.object(strat, "scan", side_effect=fake_scan_one),
        patch.object(strat, "_prefetch_forecasts", side_effect=fake_prefetch),
        patch.object(strat, "_evaluate_market", side_effect=fake_eval),
    ):
        await strat.generate_signals()

    cfg = captured.get("cfg")
    assert cfg is not None, "expected cfg captured even in shadow mode"
    assert cfg.min_calibrated_prob is None, (
        "shadow mode must zero the floor so the bin remains observable"
    )
