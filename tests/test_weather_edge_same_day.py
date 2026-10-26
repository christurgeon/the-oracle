"""Strategy-side wiring tests for SameDayGate / WeatherEdgeStrategy."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from python.common import DailyVariable
from python.strategies.same_day_gate import SameDayDecision, SameDayGate
from python.strategies.weather_edge import WeatherEdgeStrategy


def _bare_config(*, same_day_enabled: bool = False, allowed: list[str] | None = None) -> dict:
    return {
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
        "same_day_trading": {
            "enabled": same_day_enabled,
            "allowed_cities": allowed or [],
            "cache_ttl_seconds": 600,
            "near_peak_ttl_seconds": 90,
            "near_peak_window_minutes": 120,
            "peak_cutoff_minutes": 60,
            "nws_request_timeout_seconds": 4,
            "nws_gate_timeout_seconds": 20,
        },
    }


@pytest.mark.asyncio
async def test_weather_edge_init_uses_disabled_sentinel():
    """Without setup(), the gate is the disabled sentinel — never blocks anything new."""
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    assert isinstance(s._same_day_gate, SameDayGate)
    decision = await s._same_day_gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_DISABLED


@pytest.mark.asyncio
async def test_setup_keeps_sentinel_when_feature_off():
    """setup() must not construct an NWSClient unless feature is configured on."""
    s = WeatherEdgeStrategy(
        config=_bare_config(same_day_enabled=False, allowed=[]),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    sentinel = s._same_day_gate
    await s.setup()
    assert s._same_day_gate is sentinel
    # Closing must be a no-op even though setup ran.
    await s.aclose()


@pytest.mark.asyncio
async def test_setup_keeps_sentinel_when_allowlist_empty():
    """enabled=True but allowed_cities=[] → keep the sentinel (zero behavior change)."""
    s = WeatherEdgeStrategy(
        config=_bare_config(same_day_enabled=True, allowed=[]),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    sentinel = s._same_day_gate
    await s.setup()
    assert s._same_day_gate is sentinel


# ---------------------------------------------------------------------------
# Scan-loop integration tests
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from datetime import timedelta  # noqa: E402

from python.analytics.audit_log import ScanFunnel  # noqa: E402
from python.common import et_today  # noqa: E402


def _make_market(*, ticker: str, event_ticker: str, strike_type: str = "greater") -> dict:
    """Stub Kalshi market dict shaped like the production parser expects."""
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "strike_type": strike_type,
        "floor_strike": 80.0 if strike_type != "less" else None,
        "cap_strike": 80.0 if strike_type == "less" else None,
        "yes_bid_dollars": "0.45",
        "yes_ask_dollars": "0.55",
        "volume_fp": "1000.0",
        "open_interest_fp": "100.0",
    }


def _ticker_for_date(d, strike: int = 80) -> tuple[str, str]:
    """Return (event_ticker, market_ticker) for a NYC-high market on date d."""
    event = f"KXHIGHNY-{d.strftime('%y%b%d').upper()}"
    return event, f"{event}-T{strike}"


def _series_aware_iter_all_markets(*, market_for_high: dict | None = None,
                                   market_for_low: dict | None = None,
                                   truncated: bool = False):
    """Return an AsyncMock side_effect that responds based on series_ticker.

    Returns a ``(markets, truncated)`` tuple to match the post-refactor
    iter_all_markets contract.
    """

    async def _impl(*, series_ticker: str, **_kw):
        if series_ticker == "KXHIGHNY":
            return ([market_for_high] if market_for_high else []), truncated
        if series_ticker == "KXLOWTNYC":
            return ([market_for_low] if market_for_low else []), truncated
        return [], truncated

    return _impl


@pytest.mark.asyncio
async def test_past_date_market_uses_past_date_not_same_day_blocked():
    """forecast_date < scan_today → emits PAST_DATE, never SAME_DAY_BLOCKED."""
    yesterday = et_today() - timedelta(days=1)
    event, ticker = _ticker_for_date(yesterday)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(market_for_high=market),
    )
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )
    funnel = ScanFunnel()
    candidates = await s.scan(funnel=funnel, scan_id="t1")
    assert candidates == []
    assert funnel.n_past_date == 1
    assert funnel.n_same_day_blocked == 0


@pytest.mark.asyncio
async def test_same_day_disabled_emits_same_day_blocked_with_subreason():
    """forecast_date == scan_today + sentinel gate → SAME_DAY_BLOCKED / same_day_disabled."""
    today = et_today()
    event, ticker = _ticker_for_date(today)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(market_for_high=market),
    )
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )
    funnel = ScanFunnel()
    candidates = await s.scan(funnel=funnel, scan_id="t2")
    assert candidates == []
    assert funnel.n_same_day_blocked == 1
    assert funnel.n_past_date == 0


@pytest.mark.asyncio
async def test_same_day_with_live_gate_block_observation_exists():
    """Live gate returning BLOCK_NWS_OBSERVATION_EXISTS → SAME_DAY_BLOCKED with sub-reason."""
    today = et_today()
    event, ticker = _ticker_for_date(today)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(market_for_high=market),
    )
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )
    stub_gate = AsyncMock()
    stub_gate.check = AsyncMock(return_value=SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS)
    s._same_day_gate = stub_gate

    funnel = ScanFunnel()
    candidates = await s.scan(funnel=funnel, scan_id="t3")
    assert candidates == []
    assert funnel.n_same_day_blocked == 1
    stub_gate.check.assert_awaited_once()


@pytest.mark.asyncio
async def test_same_day_gate_timeout_treated_as_fetch_failed():
    """Hanging gate.check() → SAME_DAY_BLOCKED / nws_fetch_failed via wait_for."""
    today = et_today()
    event, ticker = _ticker_for_date(today)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(market_for_high=market),
    )
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )

    async def _hang(*a, **kw):
        await asyncio.sleep(60)

    stub_gate = AsyncMock()
    stub_gate.check = _hang
    s._same_day_gate = stub_gate
    # Tiny budget for the test.
    s._config["same_day_trading"]["nws_gate_timeout_seconds"] = 0.05

    funnel = ScanFunnel()
    candidates = await s.scan(funnel=funnel, scan_id="t4")
    assert candidates == []
    assert funnel.n_same_day_blocked == 1


@pytest.mark.asyncio
async def test_same_day_signal_metadata_carries_is_same_day_market_true():
    """Signals built from a same-day market carry is_same_day_market=True."""
    from unittest.mock import MagicMock

    today = et_today()

    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )

    parsed = MagicMock()
    parsed.city = "NYC"
    parsed.forecast_date = today
    parsed.daily_variable = DailyVariable.TEMP_MAX
    parsed.temp_label = "high"
    parsed.strike_type = MagicMock()
    parsed.strike_type.value = "greater"
    parsed.floor_strike = 80.0
    parsed.cap_strike = None
    parsed.description = "stub"

    candidate = MagicMock()
    candidate.market = {"ticker": "KXHIGHNY-stub", "close_time": ""}
    candidate.parsed = parsed
    candidate.price_info = MagicMock(
        midpoint=0.5, yes_bid=0.45, yes_ask=0.55, spread=0.1,
        volume=100, open_interest=10, yes_bid_size=5, yes_ask_size=5,
    )
    candidate.exec_price = 0.55
    candidate.model_prob = 0.7
    candidate.ensemble_confidence = 0.8
    candidate.raw_edge = 0.2
    candidate.forecast = MagicMock(std_dev=2.0, model="gfs_seamless")
    candidate.ensemble_std_pooled = 2.0
    candidate.per_model_probs = None
    candidate.correction_f = 0.0
    candidate.gfs_uncorrected_prob = None
    candidate.spread_inflation_factor = 1.0
    candidate.raw_ensemble_prob = 0.7
    candidate.strike_abs_z = 1.0

    sizing = MagicMock()
    sizing.side = "YES"
    sizing.dollars = 50.0
    sizing.kelly_raw = 0.05
    sizing.capped = False

    signal = s._build_signal(
        candidate, effective_edge=0.15, sizing=sizing,
        contracts=10, depth_capped=False, depth_result=None,
    )
    assert signal.metadata.get("is_same_day_market") is True


@pytest.mark.asyncio
async def test_funnel_count_matches_audit_rows_with_subreason_breakdown(tmp_path):
    """For one scan: funnel.n_same_day_blocked equals the number of eval_audit
    rows with reason=SAME_DAY_BLOCKED, and the sub-reason histogram sums to
    the same total."""
    import json
    from collections import Counter

    from python.analytics.audit_log import (
        EVAL_SCHEMA_VERSION,
        AuditLogger,
        AuditSinks,
    )

    today = et_today()
    markets = []
    for strike in (80, 81, 82):
        event, ticker = _ticker_for_date(today, strike=strike)
        markets.append(_make_market(ticker=ticker, event_ticker=event))

    kalshi = AsyncMock()

    async def _iter(*, series_ticker, **_kw):
        return (markets if series_ticker == "KXHIGHNY" else []), False

    kalshi.iter_all_markets = AsyncMock(side_effect=_iter)
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )
    eval_path = tmp_path / "eval.jsonl"
    s._audit_sinks = AuditSinks(
        eval_log=AuditLogger(eval_path, schema_version=EVAL_SCHEMA_VERSION),
        candidate_log=None,
        funnel_log=None,
    )

    decisions = iter([
        SameDayDecision.BLOCK_PEAK_PASSED,
        SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS,
        SameDayDecision.BLOCK_NWS_FETCH_FAILED,
    ])
    stub_gate = AsyncMock()
    stub_gate.check = AsyncMock(side_effect=lambda *a, **k: next(decisions))
    s._same_day_gate = stub_gate

    funnel = ScanFunnel()
    await s.scan(funnel=funnel, scan_id="consistency")
    assert funnel.n_same_day_blocked == 3

    rows = [
        json.loads(line)
        for line in eval_path.read_text().splitlines()
        if line
    ]
    sd_rows = [r for r in rows if r["reason"] == "SAME_DAY_BLOCKED"]
    assert len(sd_rows) == 3
    counts = Counter(r["same_day_decision"] for r in sd_rows)
    assert counts == {
        "peak_passed": 1,
        "nws_observation_exists": 1,
        "nws_fetch_failed": 1,
    }


@pytest.mark.asyncio
async def test_iter_all_markets_truncated_emits_error_alert_with_bucket():
    """If iter_all_markets returns truncated=True, weather_edge must emit an
    ERROR-level alert with a per-series failure_bucket so TelegramNotifier's
    per-(bucket, hour) dedup engages — weather scan runs every 60s, so
    CRITICAL would bypass dedup and spam the operator's phone."""
    from python.notifications.base import AlertLevel

    today = et_today()
    event, ticker = _ticker_for_date(today)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(
            market_for_high=market, truncated=True
        ),
    )

    notifier = AsyncMock()
    notifier.send_alert = AsyncMock()
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
        notifier=notifier,
    )
    funnel = ScanFunnel()
    await s.scan(funnel=funnel, scan_id="t-trunc")

    notifier.send_alert.assert_awaited()
    call = notifier.send_alert.await_args_list[0]
    assert call.kwargs.get("level") == AlertLevel.ERROR
    # failure_bucket must be series-scoped so dedup is per-series, not global.
    assert call.kwargs.get("failure_bucket") == "iter_all_markets_truncated:KXHIGHNY"
    msg = call.args[0] if call.args else call.kwargs.get("message", "")
    assert "TRUNCATED" in msg or "truncated" in msg
    assert "KXHIGHNY" in msg


@pytest.mark.asyncio
async def test_same_day_gate_allow_passes_through():
    """gate.check() == ALLOW → market reaches the candidate list."""
    today = et_today()
    event, ticker = _ticker_for_date(today)
    market = _make_market(ticker=ticker, event_ticker=event)

    kalshi = AsyncMock()
    kalshi.iter_all_markets = AsyncMock(
        side_effect=_series_aware_iter_all_markets(market_for_high=market),
    )
    s = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=kalshi,
        weather_client=AsyncMock(),
    )
    stub_gate = AsyncMock()
    stub_gate.check = AsyncMock(return_value=SameDayDecision.ALLOW)
    s._same_day_gate = stub_gate

    funnel = ScanFunnel()
    candidates = await s.scan(funnel=funnel, scan_id="t5")
    assert len(candidates) == 1
    assert funnel.n_same_day_blocked == 0
