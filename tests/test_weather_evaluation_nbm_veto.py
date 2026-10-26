"""NBM-bracket-position veto (shadow-only, 2026-05-20).

Backtest evidence (``docs/research/2026-05-20-weather-strategy-research.md``
§5, n=32): for between-strike trades, abstaining when NBM lands on the
OPPOSITE side of the bracket from the model's directional bet turned
joined-window P&L from a loss to a gain.  This release ships
``enabled=False`` so the gate computes ``nbm_veto_fired`` on every
between-strike row WITHOUT blocking trades — the shadow corpus
accumulates here, an n≥80 promotion gate gates any live activation.

Five scenarios under test:
1. NBM agrees with model bracket position → no veto.
2. NBM disagrees, YES side → ``nbm_veto_fired=True``; rejection only
   when ``enabled=True``.
3. NBM disagrees, NO side → ``nbm_veto_fired=True``; rejection only
   when ``enabled=True``.
4. NBM fetch fails with ``fail_open=True`` → no veto, no audit-row
   contribution beyond ``None``.
5. NBM fetch fails with ``fail_open=False`` → ``NBM_FETCH_FAILED``.

The fetch is patched directly via ``monkeypatch.setattr`` on
``WeatherClient.get_nbm_point_forecast`` — same pattern the rest of the
test suite uses for ``get_multi_model_forecast``.  No respx / httpx_mock.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.audit_log import AuditSinks, RejectionReason, ScanFunnel
from python.common import DailyVariable, Direction
from python.data.weather import EnsembleForecast, NbmPointForecast, WeatherClient
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    _EvalConfig,
    evaluate_market_pure,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_eval_cfg(**overrides: Any) -> _EvalConfig:
    base: dict[str, Any] = dict(
        max_spread=0.50,
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
        # NBM-veto defaults: shipped config — gate logs, does NOT reject.
        nbm_bracket_position_veto_enabled=False,
        nbm_fail_open=True,
    )
    base.update(overrides)
    return _EvalConfig(**base)


def _make_between_parsed(
    *,
    city: str = "NYC",
    floor_strike: float = 70.0,
    cap_strike: float = 75.0,
    daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
) -> ParsedMarket:
    fd = datetime.now(UTC).date() + timedelta(days=1)
    var_tag = "H" if daily_variable == DailyVariable.TEMP_MAX else "L"
    return ParsedMarket(
        ticker=(f"KX-{city}-{fd.isoformat()}-B{floor_strike:.0f}-{cap_strike:.0f}-{var_tag}"),
        city=city,
        forecast_date=fd,
        strike_type=Direction.BETWEEN,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        daily_variable=daily_variable,
    )


def _make_scan_candidate(
    parsed: ParsedMarket,
    *,
    yes_bid: float,
    yes_ask: float,
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


class _CapturingEvalLog:
    """In-memory drop-in for ``AuditLogger.emit_eval``."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def emit_eval(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


def _make_sinks_with_capture() -> tuple[AuditSinks, _CapturingEvalLog]:
    cap = _CapturingEvalLog()
    sinks = AuditSinks(eval_log=cap, candidate_log=None, funnel_log=None)  # type: ignore[arg-type]
    return sinks, cap


def _build_eval_kwargs(parsed: ParsedMarket, temps: list[float]) -> dict[str, Any]:
    """Build the forecast caches + std/median maps needed by
    ``evaluate_market_pure``.

    Returns positional helpers; downstream tests choose YES vs NO by
    tuning the market price relative to the resulting ``model_prob``.
    """
    fkey = (parsed.city, parsed.forecast_date, parsed.daily_variable)
    forecast_cache = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    raw_member_sink = {
        fkey: {"gfs_seamless": _make_forecast(parsed.city, parsed.forecast_date, temps)}
    }
    bias_corrections = {fkey: 0.0}
    raw_stds = {fkey: (max(temps) - min(temps)) / 4.0}
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


# Temperatures inside [70, 75] → blended probability for BETWEEN ~ 1.0
# (all 5 ensemble members land in-bracket).  We use these for YES tests:
# model_prob ~ 1.0 means model_prob > midpoint ⇒ side_is_yes = True.
_TEMPS_INSIDE = [71.0, 72.0, 73.0, 74.0, 74.5]

# Temperatures outside [70, 75] → blended probability for BETWEEN ~ 0.0
# ⇒ model_prob < midpoint ⇒ side_is_yes = False (NO).
_TEMPS_OUTSIDE = [60.0, 61.0, 62.0, 63.0, 64.0]


def _invoke(
    sc: ScanCandidate,
    parsed: ParsedMarket,
    temps: list[float],
    cfg: _EvalConfig,
    *,
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] | None,
    sinks: AuditSinks,
    funnel: ScanFunnel,
) -> Any:
    kwargs = _build_eval_kwargs(parsed, temps)
    return evaluate_market_pure(
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
        emos_table=None,
        raw_member_sink=kwargs["raw_member_sink"],
        emos_log_dedup={},
        emos_enabled_cells=set(),
        isotonic_artifact_hash=None,
        sinks=sinks,
        funnel=funnel,
        nbm_cache=nbm_cache,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [False, True])
def test_nbm_agrees_yes_inside(enabled: bool) -> None:
    """Scenario 1a: YES side + NBM inside bracket → no veto.

    Ensemble in-bracket ⇒ model_prob ~1 ⇒ YES side.  NBM inside [70,75]
    AGREES with the model's directional bet.  The gate runs but does not
    fire — the candidate passes regardless of ``enabled``.  Audit row
    records ``nbm_veto_fired=False``.
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    # Midpoint ~0.30 makes model_prob (≈1.0) > midpoint ⇒ side_is_yes
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)
    nbm_cache = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): 72.5  # inside bracket
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=enabled)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_INSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    # Veto did NOT fire — trade should proceed (returns a _MarketCandidate)
    assert result is not None, "agreement should not veto"
    assert result.nbm_temperature == pytest.approx(72.5)
    assert result.nbm_veto_fired is False
    assert funnel.n_nbm_bracket_position_veto == 0
    assert funnel.n_nbm_fetch_failed == 0
    # No rejection rows on the eval stream from the veto gate
    veto_rows = [
        r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
    ]
    assert veto_rows == []


@pytest.mark.parametrize("enabled", [False, True])
def test_nbm_agrees_no_outside(enabled: bool) -> None:
    """Scenario 1b: NO side + NBM outside bracket → no veto.

    Ensemble outside bracket ⇒ model_prob ~0 ⇒ NO side.  NBM outside
    [70,75] AGREES with the model's directional bet.
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    # Midpoint ~0.70 makes model_prob (≈0.0) < midpoint ⇒ side_is_yes=False
    sc = _make_scan_candidate(parsed, yes_bid=0.60, yes_ask=0.80)
    nbm_cache = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): 65.0  # outside bracket
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=enabled)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_OUTSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    assert result is not None, "agreement should not veto"
    assert result.nbm_temperature == pytest.approx(65.0)
    assert result.nbm_veto_fired is False
    assert funnel.n_nbm_bracket_position_veto == 0


@pytest.mark.parametrize("enabled,expect_rejected", [(False, False), (True, True)])
def test_nbm_disagrees_yes_outside(enabled: bool, expect_rejected: bool) -> None:
    """Scenario 2: YES + NBM outside bracket → ``nbm_veto_fired=True``.

    Model bets YES (ensemble inside [70,75], so model_prob ~ 1), but NBM
    lands at 65°F — outside the bracket.  Veto fires.
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)
    nbm_cache = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): 65.0  # OUTSIDE — disagree
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=enabled)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_INSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    if expect_rejected:
        assert result is None, "enabled=True must reject the candidate"
        assert funnel.n_nbm_bracket_position_veto == 1
        veto_rows = [
            r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
        ]
        assert len(veto_rows) == 1
        row = veto_rows[0]
        assert row["nbm_temperature"] == pytest.approx(65.0)
        assert row["nbm_veto_fired"] is True
        # Side metadata included in the rejection ``extra`` payload
        assert row.get("nbm_side_is_yes") is True
        assert row.get("nbm_inside_bracket") is False
    else:
        # Shipped path: gate runs, candidate proceeds, audit records the
        # would-have-fired outcome on the PASSED-bound _MarketCandidate.
        assert result is not None, "enabled=False must let the trade proceed"
        assert result.nbm_temperature == pytest.approx(65.0)
        assert result.nbm_veto_fired is True
        assert funnel.n_nbm_bracket_position_veto == 0
        veto_rows = [
            r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
        ]
        assert veto_rows == []


@pytest.mark.parametrize("enabled,expect_rejected", [(False, False), (True, True)])
def test_nbm_disagrees_no_inside(enabled: bool, expect_rejected: bool) -> None:
    """Scenario 3: NO + NBM inside bracket → ``nbm_veto_fired=True``.

    Model bets NO (ensemble outside [70,75], so model_prob ~ 0), but NBM
    lands at 72.5°F — inside the bracket.  Veto fires.
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    sc = _make_scan_candidate(parsed, yes_bid=0.60, yes_ask=0.80)
    nbm_cache = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): 72.5  # INSIDE — disagree
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=enabled)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_OUTSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    if expect_rejected:
        assert result is None, "enabled=True must reject the candidate"
        assert funnel.n_nbm_bracket_position_veto == 1
        veto_rows = [
            r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
        ]
        assert len(veto_rows) == 1
        row = veto_rows[0]
        assert row["nbm_temperature"] == pytest.approx(72.5)
        assert row["nbm_veto_fired"] is True
        assert row.get("nbm_side_is_yes") is False
        assert row.get("nbm_inside_bracket") is True
    else:
        assert result is not None
        assert result.nbm_temperature == pytest.approx(72.5)
        assert result.nbm_veto_fired is True
        assert funnel.n_nbm_bracket_position_veto == 0


def test_nbm_fetch_failure_fail_open() -> None:
    """Scenario 4: NBM fetch failed + ``fail_open=True`` → no veto, no rejection.

    Failure is surfaced by the caller as a present cache entry with value
    ``None``.  Under fail-open the gate skips silently — the candidate
    proceeds and the audit row carries ``nbm_temperature=None`` /
    ``nbm_veto_fired=None`` (gate not evaluated).
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): None,  # fetch failed
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=True, nbm_fail_open=True)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_INSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    assert result is not None, "fail_open=True must let the trade proceed"
    assert result.nbm_temperature is None
    assert result.nbm_veto_fired is None
    assert funnel.n_nbm_bracket_position_veto == 0
    assert funnel.n_nbm_fetch_failed == 0
    # No fetch-failure rejection row
    assert all(r.get("reason") != RejectionReason.NBM_FETCH_FAILED for r in cap.rows)


def test_nbm_fetch_failure_fail_closed() -> None:
    """Scenario 5: NBM fetch failed + ``fail_open=False`` → ``NBM_FETCH_FAILED``.

    The fail-closed config is not shipped in production, but the gate
    must honor it when an operator opts in.
    """
    parsed = _make_between_parsed(floor_strike=70.0, cap_strike=75.0)
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] = {
        (parsed.city, parsed.forecast_date, parsed.daily_variable): None,
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=False, nbm_fail_open=False)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_INSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    assert result is None, "fail_open=False must reject the candidate"
    assert funnel.n_nbm_fetch_failed == 1
    reasons = [r.get("reason") for r in cap.rows]
    assert RejectionReason.NBM_FETCH_FAILED in reasons


# ---------------------------------------------------------------------------
# Bonus: assert non-between markets bypass the gate entirely
# ---------------------------------------------------------------------------


def test_non_between_market_bypasses_gate() -> None:
    """The veto rule applies only to BETWEEN markets.

    Greater-strike (``Direction.ABOVE``) candidates must not be evaluated
    by the gate even if the cache happens to have an entry — the
    ``_MarketCandidate`` they emit must carry ``nbm_temperature=None`` /
    ``nbm_veto_fired=None``.
    """
    fd = datetime.now(UTC).date() + timedelta(days=1)
    parsed = ParsedMarket(
        ticker="KX-NYC-T80",
        city="NYC",
        forecast_date=fd,
        strike_type=Direction.ABOVE,
        floor_strike=80.0,
        cap_strike=None,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)
    # Cache entry is present but irrelevant — the gate must skip greater
    # strikes regardless.
    nbm_cache = {(parsed.city, parsed.forecast_date, parsed.daily_variable): 65.0}
    cfg = _make_eval_cfg(
        nbm_bracket_position_veto_enabled=True,
        # Disable the strike-distance gate so the greater candidate
        # actually reaches the veto / candidate construction site under
        # this synthetic temperature distribution.
        min_distance_from_strike_std=0.0,
    )
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    # Temperatures around 85°F so model_prob (P(T > 80)) is large but the
    # row passes the existing $0.15 YES-floor / inverted-side checks.
    temps = [82.0, 84.0, 85.0, 86.0, 88.0]
    result = _invoke(sc, parsed, temps, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel)

    # Either result is None (rejected by an unrelated gate) or it's a
    # _MarketCandidate — but in either case the NBM veto must not fire.
    assert funnel.n_nbm_bracket_position_veto == 0
    veto_rows = [
        r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
    ]
    assert veto_rows == []
    if result is not None:
        assert result.nbm_temperature is None
        assert result.nbm_veto_fired is None


# ---------------------------------------------------------------------------
# Sanity test: the prefetch helper actually calls
# WeatherClient.get_nbm_point_forecast via direct function patching, with
# no respx / httpx_mock dependency.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_uses_weather_client_get_nbm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_prefetch_nbm`` must call ``WeatherClient.get_nbm_point_forecast``
    once per unique (city, forecast_date) across the scan's between markets,
    and surface fetch failures as cache value ``None``.
    """
    from python.strategies.weather_edge import WeatherEdgeStrategy

    fd = datetime.now(UTC).date() + timedelta(days=1)
    # Three between markets: two share (NYC, fd); one is (CHI, fd) ⇒
    # dedup to two unique keys ⇒ two fetches.
    markets: list[ScanCandidate] = []
    for floor, cap in [(70.0, 75.0), (75.0, 80.0)]:
        markets.append(
            _make_scan_candidate(
                _make_between_parsed(city="NYC", floor_strike=floor, cap_strike=cap),
                yes_bid=0.20,
                yes_ask=0.40,
            )
        )
    markets.append(
        _make_scan_candidate(
            _make_between_parsed(city="CHI", floor_strike=70.0, cap_strike=75.0),
            yes_bid=0.20,
            yes_ask=0.40,
        )
    )
    # Plus a non-between market — must NOT participate in the fetch.
    markets.append(
        _make_scan_candidate(
            ParsedMarket(
                ticker="KX-NYC-T80",
                city="NYC",
                forecast_date=fd,
                strike_type=Direction.ABOVE,
                floor_strike=80.0,
                cap_strike=None,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            yes_bid=0.20,
            yes_ask=0.40,
        )
    )

    # Fetch mock — succeeds for NYC, raises for CHI to verify the failure
    # path lands as ``None`` in the cache.
    async def _mock_get_nbm(
        self: WeatherClient,
        *,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ) -> NbmPointForecast:
        if city == "NYC":
            return NbmPointForecast(
                city="NYC",
                forecast_date=forecast_date,
                daily_variable=daily_variable,
                temperature=72.5,
            )
        raise RuntimeError(f"simulated NBM failure for {city}")

    monkeypatch.setattr(WeatherClient, "get_nbm_point_forecast", _mock_get_nbm)

    # Build a minimal strategy stub — we only need ``_prefetch_nbm``, so
    # bypass __init__ and assign the one attribute it touches.
    strat = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
    strat._weather = WeatherClient.__new__(WeatherClient)  # type: ignore[attr-defined]

    cache = await strat._prefetch_nbm(markets, timeout_seconds=4.0)

    # Two unique between (city, fd, var) triples ⇒ exactly two cache entries.
    assert set(cache.keys()) == {
        ("NYC", fd, DailyVariable.TEMP_MAX),
        ("CHI", fd, DailyVariable.TEMP_MAX),
    }
    assert cache[("NYC", fd, DailyVariable.TEMP_MAX)] == pytest.approx(72.5)
    # CHI fetch raised ⇒ cache value None.
    assert cache[("CHI", fd, DailyVariable.TEMP_MAX)] is None


@pytest.mark.asyncio
async def test_prefetch_returns_empty_when_no_between_markets() -> None:
    """When the scan has zero BETWEEN markets the helper must short-circuit
    to an empty cache without invoking the fetch (regression guard against
    accidentally hammering Open-Meteo on greater/less-only scans).
    """
    from python.strategies.weather_edge import WeatherEdgeStrategy

    fd = datetime.now(UTC).date() + timedelta(days=1)
    markets = [
        _make_scan_candidate(
            ParsedMarket(
                ticker="KX-NYC-T80",
                city="NYC",
                forecast_date=fd,
                strike_type=Direction.ABOVE,
                floor_strike=80.0,
                cap_strike=None,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            yes_bid=0.20,
            yes_ask=0.40,
        )
    ]
    strat = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
    # If the helper short-circuits we never need the client; assign an
    # AsyncMock to surface any unintended invocation.
    weather_mock = MagicMock(spec=WeatherClient)
    weather_mock.get_nbm_point_forecast = AsyncMock()
    strat._weather = weather_mock  # type: ignore[attr-defined]

    cache = await strat._prefetch_nbm(markets, timeout_seconds=4.0)
    assert cache == {}
    weather_mock.get_nbm_point_forecast.assert_not_called()


# ---------------------------------------------------------------------------
# Regression: daily_variable must thread through the prefetch + eval keys
# (KXHIGH and KXLOWT settle against different observations, so the NBM
# forecast must be fetched per variable).
# ---------------------------------------------------------------------------


def test_nbm_veto_uses_low_variable_for_kxlowt_between() -> None:
    """A KXLOWT (low-temp) BETWEEN bracket must read the NBM TEMP_MIN
    forecast — NOT the TEMP_MAX forecast.

    Before the fix the prefetch hardcoded ``DailyVariable.TEMP_MAX`` and the
    consumer's cache key omitted ``daily_variable``, so a low-temp bracket
    would silently compare itself against the NBM HIGH forecast.

    Setup: NBM HIGH=85°F (way outside the [70,75] bracket) and NBM LOW=72.5°F
    (inside the bracket). Model bets YES (ensemble lands inside the bracket).
    The veto fires iff NBM lands outside the bracket — so reading the high
    forecast would (incorrectly) veto, while reading the correct low forecast
    must agree and let the trade proceed. The audit row must carry the LOW
    NBM value (72.5°F).
    """
    parsed = _make_between_parsed(
        floor_strike=70.0,
        cap_strike=75.0,
        daily_variable=DailyVariable.TEMP_MIN,
    )
    sc = _make_scan_candidate(parsed, yes_bid=0.20, yes_ask=0.40)

    # DIFFERENT NBM values for TEMP_MAX vs TEMP_MIN at the same (city, date).
    # If the consumer's cache key omits daily_variable it would either
    # collide with the TEMP_MAX entry (and read the wrong forecast) or
    # KeyError out — either way the test will catch the regression.
    nbm_cache: dict[tuple[str, date, DailyVariable], float | None] = {
        # Wrong-variable entry — must be IGNORED by the gate.
        (parsed.city, parsed.forecast_date, DailyVariable.TEMP_MAX): 85.0,
        # Correct-variable entry — must be USED by the gate.
        (parsed.city, parsed.forecast_date, DailyVariable.TEMP_MIN): 72.5,
    }
    cfg = _make_eval_cfg(nbm_bracket_position_veto_enabled=True)
    sinks, cap = _make_sinks_with_capture()
    funnel = ScanFunnel()

    result = _invoke(
        sc, parsed, _TEMPS_INSIDE, cfg, nbm_cache=nbm_cache, sinks=sinks, funnel=funnel
    )

    # The gate must read the LOW NBM (72.5°F, inside) — so it agrees with
    # the YES bet and lets the trade proceed.  If the consumer ever reads
    # the HIGH NBM (85°F, outside) it would (wrongly) veto.
    assert result is not None, (
        "veto must use TEMP_MIN forecast (72.5°F, inside bracket) — "
        "reading TEMP_MAX (85°F, outside) would incorrectly fire the gate"
    )
    assert result.nbm_temperature == pytest.approx(72.5)
    assert result.nbm_veto_fired is False
    assert funnel.n_nbm_bracket_position_veto == 0
    veto_rows = [
        r for r in cap.rows if r.get("reason") == RejectionReason.NBM_BRACKET_POSITION_VETO
    ]
    assert veto_rows == []


@pytest.mark.asyncio
async def test_prefetch_keys_separate_high_and_low_for_same_city_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the same (city, forecast_date) has BOTH a TEMP_MAX and a
    TEMP_MIN between market, the prefetch helper must keep them as separate
    cache entries (one fetch each) and must NOT conflate them.
    """
    from python.strategies.weather_edge import WeatherEdgeStrategy

    fd = datetime.now(UTC).date() + timedelta(days=1)
    markets: list[ScanCandidate] = [
        _make_scan_candidate(
            _make_between_parsed(
                city="NYC",
                floor_strike=78.0,
                cap_strike=82.0,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            yes_bid=0.20,
            yes_ask=0.40,
        ),
        _make_scan_candidate(
            _make_between_parsed(
                city="NYC",
                floor_strike=58.0,
                cap_strike=62.0,
                daily_variable=DailyVariable.TEMP_MIN,
            ),
            yes_bid=0.20,
            yes_ask=0.40,
        ),
    ]

    async def _mock_get_nbm(
        self: WeatherClient,
        *,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable = DailyVariable.TEMP_MAX,
    ) -> NbmPointForecast:
        # Distinct temperatures per variable so we can verify they don't
        # overwrite each other in the cache.
        temp = 80.5 if daily_variable == DailyVariable.TEMP_MAX else 60.5
        return NbmPointForecast(
            city=city,
            forecast_date=forecast_date,
            daily_variable=daily_variable,
            temperature=temp,
        )

    monkeypatch.setattr(WeatherClient, "get_nbm_point_forecast", _mock_get_nbm)

    strat = WeatherEdgeStrategy.__new__(WeatherEdgeStrategy)
    strat._weather = WeatherClient.__new__(WeatherClient)  # type: ignore[attr-defined]

    cache = await strat._prefetch_nbm(markets, timeout_seconds=4.0)

    assert set(cache.keys()) == {
        ("NYC", fd, DailyVariable.TEMP_MAX),
        ("NYC", fd, DailyVariable.TEMP_MIN),
    }
    assert cache[("NYC", fd, DailyVariable.TEMP_MAX)] == pytest.approx(80.5)
    assert cache[("NYC", fd, DailyVariable.TEMP_MIN)] == pytest.approx(60.5)
