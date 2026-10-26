"""Tests for the between-YES minimum-midpoint floor.

weather_edge already blocks YES-side *threshold* (greater/less) trades whose
midpoint is below ``_YES_SIDE_MIN_MIDPOINT`` ($0.15) via ``RejectionReason.
YES_FLOOR``. That gate explicitly excluded ``between`` markets, leaving
between-YES longshots below $0.10 unfiltered.

The settled record for that slice (market price < $0.10, all between-YES) was
33 trades, 6.1% win rate, -58.7% ROI -- the model systematically claims ~20%
on 1°F brackets the market prices at ~1.5¢ and loses ~94% of the time. This
guard extends the existing YES-side floor to between markets at a $0.10
threshold (``_BETWEEN_YES_SIDE_MIN_MIDPOINT``).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from python.analytics.audit_log import AuditLogger, AuditSinks, ScanFunnel
from python.common import DailyVariable, Direction
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    _EvalConfig,
    evaluate_market_pure,
)


def _tomorrow() -> date:
    return date.today() + timedelta(days=1)


def _make_between_candidate(
    *,
    city: str = "MIA",
    floor_strike: float = 83.0,
    cap_strike: float = 84.0,
    yes_bid: float,
    yes_ask: float,
    volume: int = 200,
    open_interest: int = 50,
) -> ScanCandidate:
    parsed = ParsedMarket(
        ticker="KXHIGHMIA-99JUN06-B83.5",
        city=city,
        forecast_date=_tomorrow(),
        strike_type=Direction.BETWEEN,
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
        "event_ticker": "KXHIGHMIA-99JUN06",
        "strike_type": "between",
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "yes_bid_dollars": str(yes_bid),
        "yes_ask_dollars": str(yes_ask),
        "volume_fp": str(volume),
        "open_interest_fp": str(open_interest),
        "close_time": "2099-12-31T23:59:59Z",
    }
    return ScanCandidate(market=market, parsed=parsed, price_info=price_info)


def _make_forecast_cache(
    city: str = "MIA",
) -> dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]:
    # ~8/31 members land inside the 83-84 bracket → raw between-prob clears
    # min_raw_prob_for_between (0.02) without hitting any plateau.
    temps = [83.4] * 8 + [80.0] * 12 + [88.0] * 11
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
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[float, str | None]] = []
        self.is_loaded = True
        self.fallback_reason: str | None = None

    def calibrate(self, raw: float, *, strike_type: str | None = None) -> float:
        self.calls.append((raw, strike_type))
        return self.value


def _evaluate(sc: ScanCandidate, tmp_path: Path, cfg: _EvalConfig) -> Any:
    return evaluate_market_pure(
        sc,
        _make_forecast_cache(),
        bias_corrections={},
        raw_ensemble_stds={},
        raw_ensemble_medians={},
        cfg=cfg,
        calibrator=_FixedCalibrator(0.20),
        shadow_calibrator=None,
        weights={"gfs_seamless": 1.0},
        weather_client=MagicMock(),
        bankroll=1_000.0,
        scan_id="scan-between-yes-floor",
        sinks=_make_sinks(tmp_path),
        funnel=ScanFunnel(),
    )


class TestBetweenYesFloor:
    def test_between_yes_below_10c_blocked(self, tmp_path: Path) -> None:
        # midpoint 0.02 (yes 0.01/0.03), model_prob 0.20 → YES side, < $0.10.
        sc = _make_between_candidate(yes_bid=0.01, yes_ask=0.03)
        result = _evaluate(sc, tmp_path, _make_cfg())
        assert result is None
        reasons = [r.get("reason") for r in _read_eval(tmp_path)]
        assert "YES_FLOOR" in reasons, f"expected YES_FLOOR, got {reasons}"

    def test_between_yes_at_10c_not_blocked(self, tmp_path: Path) -> None:
        # midpoint exactly 0.10 (yes 0.09/0.11) → floor uses strict `<`, passes.
        sc = _make_between_candidate(yes_bid=0.09, yes_ask=0.11)
        _evaluate(sc, tmp_path, _make_cfg())
        reasons = [r.get("reason") for r in _read_eval(tmp_path)]
        assert "YES_FLOOR" not in reasons, (
            f"midpoint==0.10 must NOT trigger YES_FLOOR; got {reasons}"
        )
