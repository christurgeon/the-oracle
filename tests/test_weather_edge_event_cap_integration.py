"""Integration test for the per-day event cap inside generate_signals.

Verifies funnel arithmetic (n_signals_fired stays pre-cap; n_per_day_cap_blocked
captures the drops) and audit-stream ordering (eval_audit PASSED rows
fire per-candidate, before the cap suppresses losers).
"""

from __future__ import annotations

import contextlib
import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from python.analytics.audit_log import (
    EVAL_SCHEMA_VERSION,
    FUNNEL_SCHEMA_VERSION,
    AuditLogger,
    AuditSinks,
    ScanFunnel,
)
from python.analytics.sizing import SizingResult
from python.common import DailyVariable, Direction, Side
from python.data.weather import EnsembleForecast
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    WeatherEdgeStrategy,
    _MarketCandidate,
)


def _bare_config(*, max_bets_per_day_per_event: int | None = 1) -> dict[str, Any]:
    """Minimal config for a WeatherEdgeStrategy."""
    return {
        "enabled": True,
        "edge_threshold": 0.07,
        "max_edge": 0.30,
        "cities": ["NYC"],
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        "calibration_type": "none",
        "candidate_audit_log": False,
        "eval_audit_log": True,
        "scan_funnel_log": True,
        "max_bets_per_day_per_event": max_bets_per_day_per_event,
        "orderbook_depth_check": False,
    }


def _make_market_dict(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "event_ticker": "EVT-NYC-26MAY04",
        "yes_bid_dollars": "0.45",
        "yes_ask_dollars": "0.55",
        "close_time": "2026-05-04T20:00:00Z",
    }


def _make_parsed(ticker: str) -> ParsedMarket:
    return ParsedMarket(
        ticker=ticker,
        city="NYC",
        forecast_date=date(2026, 5, 4),
        strike_type=Direction.BETWEEN,
        floor_strike=70.0,
        cap_strike=71.0,
        daily_variable=DailyVariable.TEMP_MAX,
    )


def _make_candidate(ticker: str, *, effective_edge: float) -> _MarketCandidate:
    parsed = _make_parsed(ticker)
    market = _make_market_dict(ticker)
    price_info = MarketPriceInfo(
        midpoint=0.50,
        yes_bid=0.45,
        yes_ask=0.55,
        spread=0.10,
        yes_bid_size=20,
        yes_ask_size=20,
        volume=100,
        open_interest=100,
    )
    sizing = SizingResult(
        side=Side.NO,
        dollars=10.0,
        kelly_raw=0.05,
        kelly_fraction_used=0.15,
        capped=False,
    )
    forecast = EnsembleForecast(
        city="NYC",
        forecast_date=date(2026, 5, 4),
        temperatures=[70.5] * 31,
        model="gfs_seamless",
    )
    return _MarketCandidate(
        parsed=parsed,
        market=market,
        forecast=forecast,
        model_prob=0.60,
        correction_f=0.0,
        price_info=price_info,
        raw_edge=0.10,
        ensemble_confidence=0.80,
        effective_edge=effective_edge,
        sizing=sizing,
        contracts=10,
        exec_price=0.55,
        per_model_probs={"gfs_seamless": 0.50},
        ensemble_std_pooled=2.0,
        spread_inflation_factor=1.0,
        raw_ensemble_prob=0.60,
        gfs_uncorrected_prob=None,
        strike_abs_z=None,
        shadow_prob=None,
    )


def _patch_audit_sinks(strategy: WeatherEdgeStrategy, tmp_path: Path) -> None:
    eval_log = AuditLogger(tmp_path / "eval_audit.jsonl", schema_version=EVAL_SCHEMA_VERSION)
    funnel_log = AuditLogger(tmp_path / "scan_funnel.jsonl", schema_version=FUNNEL_SCHEMA_VERSION)
    strategy._audit_sinks = AuditSinks(eval_log=eval_log, candidate_log=None, funnel_log=funnel_log)


def _build_pipeline_patches(
    strategy: WeatherEdgeStrategy,
    candidates_by_ticker: dict[str, _MarketCandidate],
) -> tuple[Any, ...]:
    """Build the patch context managers that fake out the pipeline."""
    scan_candidates = [
        ScanCandidate(
            market=c.market,
            parsed=c.parsed,
            price_info=c.price_info,
        )
        for c in candidates_by_ticker.values()
    ]

    async def fake_scan(
        *,
        scan_id: str | None = None,
        funnel: ScanFunnel | None = None,
        scan_today: Any = None,
    ) -> list[ScanCandidate]:
        return scan_candidates

    async def fake_prefetch(
        _markets: Any,
        *,
        raw_member_sink: dict | None = None,
    ) -> tuple[dict, dict, dict, dict]:
        return ({}, {}, {}, {})

    def fake_evaluate(
        sc: ScanCandidate, *args: Any, funnel: ScanFunnel | None = None, **kwargs: Any
    ) -> _MarketCandidate:
        return candidates_by_ticker[sc.parsed.ticker]

    async def fake_depth(
        candidate: _MarketCandidate,
        cfg: Any,
        *,
        scan_id: str | None = None,
        scan_ts: float | None = None,
        funnel: ScanFunnel | None = None,
        bracket_metric: Any | None = None,
        n_existing_same_day_bets: int = 0,
        correlation_rho: float | None = None,
    ) -> tuple[float, SizingResult, int, bool, None]:
        return (
            candidate.effective_edge,
            candidate.sizing,
            candidate.contracts,
            False,
            None,
        )

    return (
        patch.object(strategy, "scan", side_effect=fake_scan),
        patch.object(strategy, "_prefetch_forecasts", side_effect=fake_prefetch),
        patch.object(strategy, "_evaluate_market", side_effect=fake_evaluate),
        patch.object(strategy, "_apply_depth_adjustment", side_effect=fake_depth),
        patch.object(strategy, "_record_prediction"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_bets, expected_kept_tickers, expected_n_blocked",
    [
        # Cap on (max=1): only HI (highest effective_edge) survives within-scan.
        # eval_audit PASSED rows still fire per-candidate (pre-cap).
        (1, ["HI"], 2),
        # Cap off (null): all three signals returned, no drops counted.
        (None, ["LO", "MID", "HI"], 0),
    ],
)
async def test_cap_arithmetic_holds_end_to_end(
    tmp_path: Path,
    max_bets: int | None,
    expected_kept_tickers: list[str],
    expected_n_blocked: int,
) -> None:
    """funnel.n_signals_fired - funnel.n_per_day_cap_blocked == len(returned signals)
    in both modes; with the cap on, PASSED audit rows still fire pre-cap."""
    strategy = WeatherEdgeStrategy(
        config=_bare_config(max_bets_per_day_per_event=max_bets),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    _patch_audit_sinks(strategy, tmp_path)

    candidates_by_ticker = {
        "LO": _make_candidate("LO", effective_edge=0.10),
        "MID": _make_candidate("MID", effective_edge=0.20),
        "HI": _make_candidate("HI", effective_edge=0.30),
    }

    with contextlib.ExitStack() as stack:
        for p in _build_pipeline_patches(strategy, candidates_by_ticker):
            stack.enter_context(p)
        signals = await strategy.generate_signals()

    assert sorted(s.market_ticker for s in signals) == sorted(expected_kept_tickers)

    eval_rows = [
        json.loads(line) for line in (tmp_path / "eval_audit.jsonl").read_text().splitlines()
    ]
    passed_rows = [r for r in eval_rows if r.get("reason") == "PASSED"]
    assert len(passed_rows) == 3

    funnel_rows = [
        json.loads(line) for line in (tmp_path / "scan_funnel.jsonl").read_text().splitlines()
    ]
    assert len(funnel_rows) == 1
    f = funnel_rows[0]
    assert f["n_signals_fired"] == 3
    assert f["n_per_day_cap_blocked"] == expected_n_blocked
    assert f["n_signals_fired"] - f["n_per_day_cap_blocked"] == len(signals)


@pytest.mark.asyncio
async def test_default_state_paths_follow_settings_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture rows must never land in ``<cwd>/state`` — in production cwd is /opt/oracle."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    state_dir = tmp_path / "configured_state"
    monkeypatch.setenv("ORACLE_STATE_DIR", str(state_dir))

    strategy = WeatherEdgeStrategy(
        config=_bare_config(max_bets_per_day_per_event=1),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    candidates_by_ticker = {
        "LO": _make_candidate("LO", effective_edge=0.10),
        "MID": _make_candidate("MID", effective_edge=0.20),
        "HI": _make_candidate("HI", effective_edge=0.30),
    }

    with contextlib.ExitStack() as stack:
        for p in _build_pipeline_patches(strategy, candidates_by_ticker):
            stack.enter_context(p)
        await strategy.generate_signals()

    assert not (cwd / "state").exists()
    assert strategy._trades_log_path == state_dir / "trades.jsonl"
    blocks = (state_dir / "per_day_cap_blocks.jsonl").read_text().splitlines()
    assert sorted(json.loads(b)["blocked_signal"]["market_ticker"] for b in blocks) == [
        "LO",
        "MID",
    ]
