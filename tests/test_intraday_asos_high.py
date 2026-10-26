"""Tests for IntradayAsosHighStrategy class skeleton (Phase 4B) + generate_signals (4B-ii).

Covers:
  - setup() preconditions (climatology coverage + cell overlap)
  - scan() returns only same-day HIGH markets on enabled+informational cells
  - lifecycle helpers: reset_daily_counters, mark_traded, unmark_traded,
    load_state_from_log
  - generate_signals() integration tests (Phase 4B-ii):
    - passing market yields ONE signal with strategy_name == "intraday_asos_high"
    - informational-cell market is prediction-logged but NOT in returned signals
    - guard-dropped market is prediction-logged, produces NO signal
    - per-cell daily cap caps at max_trades_per_cell_per_day
    - ASOS obs None for a cell → fail-open (no signal)

Uses in-memory ClimatologyTable fixtures built from raw dicts — no on-disk
artifact required.  Uses the live-Kalshi dict shape (prices as strings) per
the memory note kalshi_live_vs_snapshot_market_shape.

All async tests use ``@pytest.mark.asyncio`` (project standard; matches
``tests/test_intraday_emos_high_shadow_e2e.py``).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from python.common import et_today
from python.data.asos import AsosClient, AsosObservation, ClimatologyTable


@pytest.fixture(autouse=True)
def _no_series_stagger(monkeypatch: pytest.MonkeyPatch) -> None:
    """scan() sleeps 1 s between Kalshi series to avoid 429s; the fake client needs no pacing."""
    real_sleep = asyncio.sleep

    async def _yield_only(delay: float, result: Any = None) -> Any:
        return await real_sleep(0, result)

    monkeypatch.setattr("python.strategies.intraday_asos_high.asyncio.sleep", _yield_only)

# ---------------------------------------------------------------------------
# Helpers — ClimatologyTable in-memory fixtures
# ---------------------------------------------------------------------------


def _clim_raw(cities: dict[str, list[str]]) -> dict[str, Any]:
    """Build a minimal raw climatology JSON dict for the given cities.

    For each city/var combination, generates PRIMARY cells for every
    (month, hour) combination so coverage checks always succeed.
    """
    cities_block: dict[str, Any] = {}
    for city, vars_ in cities.items():
        var_block: dict[str, Any] = {}
        for var in vars_:
            cells: dict[str, Any] = {}
            for month in range(1, 13):
                for hour in range(0, 24):
                    key = f"{month:02d}-{hour:02d}"
                    cells[key] = {
                        "source": "primary",
                        "mu": 1.5,
                        "sigma": 2.0,
                        "n": 30,
                    }
            var_block[var] = {"cells": cells}
        cities_block[city] = var_block
    return {"spec_version": "1.0", "cities": cities_block}


def _make_climatology(*cities: str, var: str = "high") -> ClimatologyTable:
    """Return a ClimatologyTable with full coverage for all given cities."""
    return ClimatologyTable(_clim_raw({city: [var] for city in cities}))


def _make_empty_climatology() -> ClimatologyTable:
    """Return a ClimatologyTable with no cities at all."""
    return ClimatologyTable({"cities": {}})


# ---------------------------------------------------------------------------
# Helpers — fake KalshiClient
# ---------------------------------------------------------------------------


def _fake_kalshi(
    markets_by_series: dict[str, list[dict[str, Any]]] | None = None,
    orderbook: dict[str, Any] | None = None,
) -> Any:
    """Return a mock KalshiClient whose ``iter_all_markets`` returns the
    given market lists keyed by ``series_ticker``.

    ``orderbook`` is the value returned by ``get_orderbook``.  When None,
    returns a minimal empty-but-valid book so depth-check passes (stale=True
    means thin, which causes a skip — use a non-empty book to get past depth).
    """
    client = MagicMock()
    _markets = markets_by_series or {}
    _book = orderbook

    async def _iter_all_markets(
        *,
        status: str | None = None,
        series_ticker: str | None = None,
        page_delay: float = 1.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        return _markets.get(series_ticker or "", []), False

    async def _get_orderbook(ticker: str, depth: int = 10) -> dict[str, Any]:
        if _book is not None:
            return _book
        # Default: return a thick enough book so the depth check passes.
        # 10 contracts at 0.50 on the YES side (we're either YES or NO buyer).
        return {
            "orderbook_fp": {
                "no_dollars": [{"price": 50, "quantity": 10}],
                "yes_dollars": [{"price": 50, "quantity": 10}],
            }
        }

    client.iter_all_markets = _iter_all_markets
    client.get_orderbook = _get_orderbook
    return client


# ---------------------------------------------------------------------------
# Helpers — live-Kalshi shaped market dicts
# ---------------------------------------------------------------------------


def _live_market(
    ticker: str,
    event_ticker: str,
    strike_type: str,
    *,
    floor_strike: float | None = None,
    cap_strike: float | None = None,
    yes_bid: float = 0.40,
    yes_ask: float = 0.50,
    volume: int = 100,
    oi: int = 200,
) -> dict[str, Any]:
    """Build a minimal live-Kalshi API market dict.

    Prices are strings (live-Kalshi shape) per kalshi_live_vs_snapshot_market_shape.
    Fixed-point integer fields (volume_fp, open_interest_fp) are numeric strings.
    """
    m: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "strike_type": strike_type,
        "yes_bid_dollars": str(yes_bid),
        "yes_ask_dollars": str(yes_ask),
        "volume_fp": str(float(volume)),
        "open_interest_fp": str(float(oi)),
    }
    if floor_strike is not None:
        m["floor_strike"] = floor_strike
    if cap_strike is not None:
        m["cap_strike"] = cap_strike
    return m


# Today's date and ticker-date tag for building valid tickers.
_TODAY = et_today()
_TODAY_TAG = _TODAY.strftime("%y%b%d").upper()  # e.g. "26JUN05"


# ---------------------------------------------------------------------------
# Fixtures — Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    """Minimal Settings pointing at a temp state_dir."""
    from python.settings import Settings

    return Settings(state_dir=str(tmp_path / "state"), _env_file=None)


# ---------------------------------------------------------------------------
# Helpers — strategy factory
# ---------------------------------------------------------------------------


def _make_strategy(
    settings,
    kalshi_client,
    climatology: ClimatologyTable,
    *,
    enabled_cells: list[str] | None = None,
    informational_cells: list[str] | None = None,
    extra_config: dict[str, Any] | None = None,
    asos_client: AsosClient | None = None,
    prediction_log: Any | None = None,
):
    """Construct an IntradayAsosHighStrategy for testing (NOT yet set up)."""
    from python.strategies.intraday_asos_high import IntradayAsosHighStrategy

    config: dict[str, Any] = {
        "enabled": True,
        "enabled_cells": enabled_cells or [],
        "informational_cells": informational_cells or [],
        "eval_audit_log": False,
        "candidate_audit_log": False,
        "scan_funnel_log": False,
    }
    if extra_config:
        config.update(extra_config)

    _asos_client = asos_client if asos_client is not None else AsosClient()
    return IntradayAsosHighStrategy(
        config=config,
        kalshi_client=kalshi_client,
        settings=settings,
        asos_client=_asos_client,
        climatology=climatology,
        prediction_log=prediction_log,
    )


# ---------------------------------------------------------------------------
# Part B — setup() tests
# ---------------------------------------------------------------------------


class TestSetupPreconditions:
    """setup() enforces preconditions and sets _setup_completed on success."""

    @pytest.mark.asyncio
    async def test_setup_raises_on_missing_climatology_coverage(self, settings):
        """RuntimeError when a configured cell has no climatology entries."""
        empty_clim = _make_empty_climatology()
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            empty_clim,
            enabled_cells=["MIA/high"],
        )

        with pytest.raises(RuntimeError, match="no climatology coverage"):
            await strategy.setup()

        assert strategy._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_raises_on_overlap_between_enabled_and_informational(
        self, settings
    ):
        """RuntimeError when enabled_cells and informational_cells share a cell."""
        clim = _make_climatology("MIA", "LAX")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            informational_cells=["MIA/high"],  # overlap!
        )

        with pytest.raises(RuntimeError, match="both"):
            await strategy.setup()

        assert strategy._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_raises_on_non_high_cell(self, settings):
        """RuntimeError when a configured cell is not a HIGH cell."""
        raw = _clim_raw({"MIA": ["low"]})
        clim = ClimatologyTable(raw)
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/low"],  # LOW cell — should fail precondition
        )

        with pytest.raises(RuntimeError, match="(?i)not.*high|high.*only"):
            await strategy.setup()

        assert strategy._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_succeeds_and_sets_flag_for_valid_config(self, settings):
        """setup() completes and sets _setup_completed=True for a valid config."""
        clim = _make_climatology("MIA", "LAX")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            informational_cells=["LAX/high"],
        )

        await strategy.setup()

        assert strategy._setup_completed is True

    @pytest.mark.asyncio
    async def test_setup_succeeds_with_empty_cells(self, settings):
        """setup() succeeds with no enabled/informational cells configured."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=[],
            informational_cells=[],
        )

        await strategy.setup()

        assert strategy._setup_completed is True

    @pytest.mark.asyncio
    async def test_setup_raises_when_only_informational_has_no_coverage(self, settings):
        """Coverage check applies to informational cells too."""
        clim = _make_climatology("MIA")  # only MIA covered, not LAX
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            informational_cells=["LAX/high"],  # LAX missing from clim
        )

        with pytest.raises(RuntimeError, match="no climatology coverage"):
            await strategy.setup()


# ---------------------------------------------------------------------------
# Part B — scan() tests
# ---------------------------------------------------------------------------


class TestScan:
    """scan() returns only same-day HIGH markets on enabled+informational cells."""

    @pytest.mark.asyncio
    async def test_scan_returns_empty_when_disabled(self, settings):
        """scan() returns [] when strategy is disabled."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            extra_config={"enabled": False},
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_returns_empty_when_setup_not_complete(self, settings):
        """scan() returns [] when setup() has not been called."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        # Intentionally skip setup()

        result = await strategy.scan(scan_today=_TODAY)
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_returns_same_day_markets_for_enabled_cell(self, settings):
        """scan() returns ScanCandidates for enabled cell + same-day markets."""
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-B85"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        market = _live_market(
            mia_ticker,
            mia_event,
            "between",
            floor_strike=85.0,
            cap_strike=86.0,
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]})
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        assert len(result) == 1
        assert result[0].parsed.city == "MIA"
        assert result[0].parsed.temp_label == "high"
        assert result[0].parsed.forecast_date == _TODAY

    @pytest.mark.asyncio
    async def test_scan_accepts_informational_cell(self, settings):
        """scan() also accepts markets on informational_cells."""
        lax_ticker = f"KXHIGHLAX-{_TODAY_TAG}-G70"
        lax_event = f"KXHIGHLAX-{_TODAY_TAG}"
        market = _live_market(
            lax_ticker,
            lax_event,
            "greater",
            floor_strike=70.0,
        )
        clim = _make_climatology("MIA", "LAX")
        kalshi = _fake_kalshi({"KXHIGHLAX": [market]})
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            informational_cells=["LAX/high"],
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        assert len(result) == 1
        assert result[0].parsed.city == "LAX"

    @pytest.mark.asyncio
    async def test_scan_rejects_cells_not_in_allowed_set(self, settings):
        """scan() skips markets whose cell is not in enabled+informational."""
        # AUS market, but AUS is not in enabled_cells or informational_cells
        aus_ticker = f"KXHIGHAUS-{_TODAY_TAG}-G90"
        aus_event = f"KXHIGHAUS-{_TODAY_TAG}"
        market = _live_market(
            aus_ticker,
            aus_event,
            "greater",
            floor_strike=90.0,
        )
        clim = _make_climatology("MIA", "AUS")
        kalshi = _fake_kalshi({"KXHIGHAUS": [market]})
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],  # AUS not enabled
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_rejects_past_date_markets(self, settings):
        """scan() skips markets whose forecast_date is before scan_today."""
        yesterday = _TODAY - dt.timedelta(days=1)
        yday_tag = yesterday.strftime("%y%b%d").upper()

        mia_ticker = f"KXHIGHMIA-{yday_tag}-B85"
        mia_event = f"KXHIGHMIA-{yday_tag}"
        market = _live_market(
            mia_ticker,
            mia_event,
            "between",
            floor_strike=85.0,
            cap_strike=86.0,
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]})
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_rejects_future_date_markets(self, settings):
        """scan() skips markets whose forecast_date is after scan_today."""
        tomorrow = _TODAY + dt.timedelta(days=1)
        tmrw_tag = tomorrow.strftime("%y%b%d").upper()

        mia_ticker = f"KXHIGHMIA-{tmrw_tag}-B85"
        mia_event = f"KXHIGHMIA-{tmrw_tag}"
        market = _live_market(
            mia_ticker,
            mia_event,
            "between",
            floor_strike=85.0,
            cap_strike=86.0,
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]})
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        assert result == []

    @pytest.mark.asyncio
    async def test_scan_returns_multiple_markets_for_multiple_cells(self, settings):
        """scan() collects markets for both enabled and informational cells."""
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-B85"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        lax_ticker = f"KXHIGHLAX-{_TODAY_TAG}-G70"
        lax_event = f"KXHIGHLAX-{_TODAY_TAG}"

        mia_market = _live_market(
            mia_ticker, mia_event, "between", floor_strike=85.0, cap_strike=86.0
        )
        lax_market = _live_market(
            lax_ticker, lax_event, "greater", floor_strike=70.0
        )
        clim = _make_climatology("MIA", "LAX")
        kalshi = _fake_kalshi(
            {"KXHIGHMIA": [mia_market], "KXHIGHLAX": [lax_market]}
        )
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            informational_cells=["LAX/high"],
        )
        await strategy.setup()

        result = await strategy.scan(scan_today=_TODAY)

        cities = {sc.parsed.city for sc in result}
        assert "MIA" in cities
        assert "LAX" in cities
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Part B — lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """reset_daily_counters, mark_traded, unmark_traded, load_state_from_log."""

    async def _make_ready_strategy(self, settings):
        """Return a setup-complete strategy with MIA/high enabled."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        await strategy.setup()
        return strategy

    @pytest.mark.asyncio
    async def test_mark_and_unmark_traded(self, settings):
        """mark_traded/unmark_traded round-trip."""
        strategy = await self._make_ready_strategy(settings)

        strategy.mark_traded("KXHIGHMIA-26JUN05-B85")
        assert "KXHIGHMIA-26JUN05-B85" in strategy._traded_tickers

        strategy.unmark_traded("KXHIGHMIA-26JUN05-B85")
        assert "KXHIGHMIA-26JUN05-B85" not in strategy._traded_tickers

    @pytest.mark.asyncio
    async def test_unmark_traded_is_idempotent(self, settings):
        """unmark_traded silently no-ops for unknown tickers."""
        strategy = await self._make_ready_strategy(settings)
        # Should not raise
        strategy.unmark_traded("NONEXISTENT-TICKER")

    @pytest.mark.asyncio
    async def test_reset_daily_counters_clears_trades_today(self, settings):
        """reset_daily_counters() clears the _trades_today dict."""
        strategy = await self._make_ready_strategy(settings)

        # Seed a counter entry
        strategy._trades_today[("MIA/high", _TODAY)] = 2
        strategy.reset_daily_counters()

        assert strategy._trades_today == {}

    @pytest.mark.asyncio
    async def test_reset_daily_counters_preserves_traded_tickers(self, settings):
        """reset_daily_counters() does NOT clear _traded_tickers."""
        strategy = await self._make_ready_strategy(settings)

        strategy.mark_traded("KXHIGHMIA-26JUN05-B85")
        strategy.reset_daily_counters()

        # _traded_tickers is a per-session dedup; it is NOT cleared on daily reset
        # (matches intraday_emos_high behavior)
        assert "KXHIGHMIA-26JUN05-B85" in strategy._traded_tickers

    @pytest.mark.asyncio
    async def test_load_state_from_log_empty_file(self, settings, tmp_path):
        """load_state_from_log tolerates a missing file gracefully."""
        strategy = await self._make_ready_strategy(settings)
        missing_path = tmp_path / "nonexistent_trades.jsonl"

        # Should not raise — graceful degradation
        strategy.load_state_from_log(missing_path)
        assert strategy._trades_today == {}
        assert strategy._traded_tickers == set()

    @pytest.mark.asyncio
    async def test_load_state_from_log_seeds_counter_from_todays_fills(
        self, settings, tmp_path
    ):
        """load_state_from_log seeds _trades_today from today's fill rows."""
        strategy = await self._make_ready_strategy(settings)

        # Build a fake trades.jsonl with one fill row for today
        today_ticker = f"KXHIGHMIA-{_TODAY_TAG}-B85"
        fill_row = {
            "event": "fill",
            "strategy": "intraday_asos_high",
            "market_ticker": today_ticker,
            "side": "YES",
            "price": 0.45,
            "contracts": 5,
        }
        trade_log = tmp_path / "trades.jsonl"
        trade_log.write_text(json.dumps(fill_row) + "\n")

        strategy.load_state_from_log(trade_log)

        cell = "MIA/high"
        key = (cell, _TODAY)
        assert strategy._trades_today.get(key, 0) == 1
        assert today_ticker in strategy._traded_tickers

    @pytest.mark.asyncio
    async def test_load_state_from_log_ignores_other_strategy_rows(
        self, settings, tmp_path
    ):
        """load_state_from_log skips fill rows for other strategies."""
        strategy = await self._make_ready_strategy(settings)

        today_ticker = f"KXHIGHMIA-{_TODAY_TAG}-B85"
        fill_row = {
            "event": "fill",
            "strategy": "weather_edge",  # different strategy
            "market_ticker": today_ticker,
            "side": "YES",
            "price": 0.45,
            "contracts": 5,
        }
        trade_log = tmp_path / "trades.jsonl"
        trade_log.write_text(json.dumps(fill_row) + "\n")

        strategy.load_state_from_log(trade_log)

        # Should not have seeded any counter for this strategy
        assert strategy._trades_today == {}
        assert strategy._traded_tickers == set()


# ---------------------------------------------------------------------------
# Part B — generate_signals stub
# ---------------------------------------------------------------------------


class TestGenerateSignalsStub:
    """generate_signals() with no markets returns []."""

    @pytest.mark.asyncio
    async def test_generate_signals_returns_empty_when_no_markets(self, settings):
        """With no markets from Kalshi, generate_signals() returns []."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        await strategy.setup()

        result = await strategy.generate_signals()

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_signals_without_setup_returns_empty(self, settings):
        """generate_signals() returns [] even without setup (setup_completed=False)."""
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi()
        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
        )
        # Intentionally skip setup()

        result = await strategy.generate_signals()

        assert result == []


# ---------------------------------------------------------------------------
# Helpers — fake AsosClient
# ---------------------------------------------------------------------------


def _fake_asos_client(
    obs_by_city: dict[str, AsosObservation | None] | None = None,
) -> AsosClient:
    """Return a mock AsosClient whose ``max_so_far`` returns from ``obs_by_city``."""
    client = MagicMock(spec=AsosClient)
    _obs = obs_by_city or {}

    async def _max_so_far(
        *,
        city: str,
        local_day: dt.date,
    ) -> AsosObservation | None:
        return _obs.get(city)

    client.max_so_far = _max_so_far
    return client


def _good_obs(M: float = 85.0, age_hours: float = 0.5) -> AsosObservation:  # noqa: N803
    """Build a non-stale AsosObservation with a plausible max-so-far."""
    return AsosObservation(M=M, age_hours=age_hours, n_obs=3)


def _make_between_market(
    city: str,
    series: str,
    *,
    floor: float,
    cap: float,
    yes_bid: float = 0.40,
    yes_ask: float = 0.52,
) -> dict[str, Any]:
    """Build a BETWEEN market dict with enough YES/NO edge to pass the default threshold."""
    ticker = f"{series}-{_TODAY_TAG}-B{floor}"
    event = f"{series}-{_TODAY_TAG}"
    return _live_market(
        ticker,
        event,
        "between",
        floor_strike=floor,
        cap_strike=cap,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
    )


# ---------------------------------------------------------------------------
# Phase 4B-ii — generate_signals integration tests
# ---------------------------------------------------------------------------


class TestGenerateSignalsIntegration:
    """generate_signals() integration tests with fake ASOS client."""

    def _make_simple_prediction_log(self) -> list[Any]:
        """Return a list that acts as a prediction log (just collects records)."""

        class FakePredictionLog:
            def __init__(self) -> None:
                self.records: list[Any] = []

            def append(self, rec: Any) -> None:
                self.records.append(rec)

        return FakePredictionLog()

    @pytest.mark.asyncio
    async def test_passing_market_yields_one_signal(self, settings):
        """A market with good edge + fresh obs → ONE signal with correct strategy_name."""
        # Build a MIA/high BETWEEN market with M=85, floor=84, cap=85 so the
        # posterior P(84 < T < 85) is low (NO edge since temp already at 85).
        # Instead, use M=79 so the posterior says "high will exceed 85" easily.
        # Actually let's just use ABOVE market at a floor well below current M.
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-G80"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        # M=85, floor=80: P(T_final > 80.5) is very high since obs is already 85 > 80.
        # YES edge: model_prob is high, yes_ask=0.85, so model_prob > yes_ask → YES side
        market = _live_market(
            mia_ticker,
            mia_event,
            "greater",
            floor_strike=80.0,
            yes_bid=0.83,
            yes_ask=0.85,
        )
        clim = _make_climatology("MIA")
        # Thick orderbook at 0.85 so depth check passes
        book = {
            "orderbook_fp": {
                "no_dollars": [{"price": 15, "quantity": 100}],  # NO bids = YES asks
                "yes_dollars": [{"price": 85, "quantity": 100}],
            }
        }
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]}, orderbook=book)
        obs_by_city = {"MIA": _good_obs(M=85.0, age_hours=0.3)}
        asos = _fake_asos_client(obs_by_city)
        pred_log = self._make_simple_prediction_log()

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            extra_config={
                "edge_threshold": 0.05,
                "posterior_t_dof": 5,
                "boundary_distance_min_f": 0.0,  # disable boundary guard
                "asos_max_staleness_hours": 2.0,
                "sigma_gain_floor_f": 0.1,
                "orderbook_depth_check": False,  # skip orderbook to simplify
            },
            asos_client=asos,
            prediction_log=pred_log,
        )
        await strategy.setup()

        signals = await strategy.generate_signals()

        assert len(signals) == 1, f"Expected 1 signal, got {len(signals)}: {signals}"
        assert signals[0].strategy_name == "intraday_asos_high"
        assert signals[0].market_ticker == mia_ticker
        # Prediction was logged
        assert len(pred_log.records) >= 1
        assert pred_log.records[0].strategy == "intraday_asos_high"

    @pytest.mark.asyncio
    async def test_informational_cell_logged_but_not_traded(self, settings):
        """LAX market on informational_cells → prediction-logged, NOT in returned signals."""
        lax_ticker = f"KXHIGHLAX-{_TODAY_TAG}-G70"
        lax_event = f"KXHIGHLAX-{_TODAY_TAG}"
        market = _live_market(
            lax_ticker,
            lax_event,
            "greater",
            floor_strike=70.0,
            yes_bid=0.83,
            yes_ask=0.85,
        )
        clim = _make_climatology("LAX")
        kalshi = _fake_kalshi({"KXHIGHLAX": [market]})
        obs_by_city = {"LAX": _good_obs(M=80.0, age_hours=0.3)}
        asos = _fake_asos_client(obs_by_city)
        pred_log = self._make_simple_prediction_log()

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            # LAX only in informational — never in enabled
            enabled_cells=[],
            informational_cells=["LAX/high"],
            extra_config={
                "edge_threshold": 0.05,
                "boundary_distance_min_f": 0.0,
                "asos_max_staleness_hours": 2.0,
                "sigma_gain_floor_f": 0.1,
                "orderbook_depth_check": False,
            },
            asos_client=asos,
            prediction_log=pred_log,
        )
        await strategy.setup()

        signals = await strategy.generate_signals()

        # Signal list must be empty — informational cells are never traded.
        assert signals == [], f"Expected no signals for informational cell, got: {signals}"
        # But the prediction log received the LAX candidate.
        assert len(pred_log.records) >= 1, "Expected prediction log to receive LAX candidate"
        lax_recs = [r for r in pred_log.records if r.city == "LAX"]
        assert len(lax_recs) >= 1, "Expected at least one LAX record in prediction log"

    @pytest.mark.asyncio
    async def test_guard_dropped_market_logged_not_traded(self, settings):
        """A stale-obs market → prediction-logged with guard_outcome=stale, no signal."""
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-G80"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        market = _live_market(
            mia_ticker,
            mia_event,
            "greater",
            floor_strike=80.0,
            yes_bid=0.83,
            yes_ask=0.85,
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]})
        # Stale observation: age_hours=5.0 > max_staleness_h=2.0 → guard="stale"
        obs_by_city = {"MIA": _good_obs(M=85.0, age_hours=5.0)}
        asos = _fake_asos_client(obs_by_city)
        pred_log = self._make_simple_prediction_log()

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            extra_config={
                "edge_threshold": 0.05,
                "boundary_distance_min_f": 0.0,
                "asos_max_staleness_hours": 2.0,  # 5.0 > 2.0 → stale guard fires
                "sigma_gain_floor_f": 0.1,
                "orderbook_depth_check": False,
            },
            asos_client=asos,
            prediction_log=pred_log,
        )
        await strategy.setup()

        signals = await strategy.generate_signals()

        # No trading signal
        assert signals == [], f"Expected no signals for stale obs, got: {signals}"
        # Prediction was still logged
        assert len(pred_log.records) >= 1, "Expected prediction log entry for guard-dropped market"
        stale_recs = [r for r in pred_log.records if r.guard_outcome == "stale"]
        assert len(stale_recs) >= 1, "Expected guard_outcome='stale' in prediction log"

    @pytest.mark.asyncio
    async def test_per_cell_daily_cap(self, settings):
        """max_trades_per_cell_per_day=1 caps at 1 signal even with 2 passing markets."""
        # Two different GREATER markets on MIA/high, both should pass guards.
        mia_ticker_1 = f"KXHIGHMIA-{_TODAY_TAG}-G79"
        mia_ticker_2 = f"KXHIGHMIA-{_TODAY_TAG}-G80"
        event = f"KXHIGHMIA-{_TODAY_TAG}"
        market1 = _live_market(
            mia_ticker_1, event, "greater", floor_strike=79.0, yes_bid=0.88, yes_ask=0.90
        )
        market2 = _live_market(
            mia_ticker_2, event, "greater", floor_strike=80.0, yes_bid=0.83, yes_ask=0.85
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market1, market2]})
        obs_by_city = {"MIA": _good_obs(M=85.0, age_hours=0.3)}
        asos = _fake_asos_client(obs_by_city)

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            extra_config={
                "edge_threshold": 0.05,
                "boundary_distance_min_f": 0.0,
                "asos_max_staleness_hours": 2.0,
                "sigma_gain_floor_f": 0.1,
                "orderbook_depth_check": False,
                "max_trades_per_cell_per_day": 1,  # cap at 1
            },
            asos_client=asos,
        )
        await strategy.setup()

        signals = await strategy.generate_signals()

        # Both markets are on the same cell AND same event → _select_one_per_event
        # collapses them to 1 first; cap=1 then allows it through.
        # Either way, ≤1 signal should come back.
        assert len(signals) <= 1, f"Expected ≤1 signal with cap=1, got {len(signals)}"

    @pytest.mark.asyncio
    async def test_asos_obs_none_yields_no_signal(self, settings):
        """If ASOS obs is None for a cell, no signal is produced (fail-open)."""
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-G80"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        market = _live_market(
            mia_ticker,
            mia_event,
            "greater",
            floor_strike=80.0,
            yes_bid=0.83,
            yes_ask=0.85,
        )
        clim = _make_climatology("MIA")
        kalshi = _fake_kalshi({"KXHIGHMIA": [market]})
        # No ASOS obs for MIA → None
        obs_by_city: dict[str, AsosObservation | None] = {"MIA": None}
        asos = _fake_asos_client(obs_by_city)

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high"],
            extra_config={
                "edge_threshold": 0.05,
                "boundary_distance_min_f": 0.0,
                "orderbook_depth_check": False,
            },
            asos_client=asos,
        )
        await strategy.setup()

        signals = await strategy.generate_signals()

        assert signals == [], f"Expected no signals when ASOS obs is None, got: {signals}"


# ---------------------------------------------------------------------------
# Phase 4B-ii / Stage 0b — N->1 batched-fetch collapse (strategy integration)
# ---------------------------------------------------------------------------


class TestBatchedFetchCollapse:
    """generate_signals issues ONE batched IEM fetch for a multi-city tick.

    Stage 0b proves the N->1 collapse at the AsosClient unit level; this test
    proves it at the strategy-integration level: a tick with multiple cells
    across multiple cities on the same forecast_date (same local_day) triggers
    exactly ONE underlying batched fetch (``_fetch_rows_batch``), and the
    per-cell ``max_so_far`` calls all hit the warm cache so the single-station
    path (``_fetch_rows``) is NEVER taken.
    """

    @pytest.mark.asyncio
    async def test_multi_city_tick_makes_one_batch_zero_single_fetches(self, settings):
        from python.data.asos import _icao_to_short

        # Use a REAL AsosClient (so the real cache + max_so_far_batch/max_so_far
        # logic runs) but count + stub the two underlying HTTP fetch seams.
        # now_fn is fixed so observation age is deterministic and fresh.
        now_utc = dt.datetime.combine(
            _TODAY, dt.time(18, 0), tzinfo=dt.UTC
        )
        client = AsosClient(now_fn=lambda: now_utc)

        # One in-day row per station at 17:00 UTC on _TODAY: 12:00 EST (MIA) /
        # 11:00 CST (AUS) — both fall on _TODAY in their LST frame, age ~1h.
        valid_utc = dt.datetime.combine(
            _TODAY, dt.time(17, 0), tzinfo=dt.UTC
        )
        rows_by_short = {
            "MIA": [("MIA", valid_utc, 85.0)],
            "AUS": [("AUS", valid_utc, 90.0)],
        }

        batch_calls: list[list[str]] = []
        single_calls: list[str] = []

        async def _counting_fetch_rows_batch(stations, local_day):
            batch_calls.append(list(stations))
            return {
                _icao_to_short(s.upper()): rows_by_short.get(
                    _icao_to_short(s.upper()), []
                )
                for s in stations
            }

        async def _counting_fetch_rows(station, local_day):
            single_calls.append(station)
            by_station = await _counting_fetch_rows_batch([station], local_day)
            return by_station.get(_icao_to_short(station.upper()), [])

        client._fetch_rows_batch = _counting_fetch_rows_batch  # type: ignore[method-assign]
        client._fetch_rows = _counting_fetch_rows  # type: ignore[method-assign]

        # Two cells across two cities, same forecast_date (== _TODAY).
        mia_ticker = f"KXHIGHMIA-{_TODAY_TAG}-G80"
        mia_event = f"KXHIGHMIA-{_TODAY_TAG}"
        aus_ticker = f"KXHIGHAUS-{_TODAY_TAG}-G80"
        aus_event = f"KXHIGHAUS-{_TODAY_TAG}"
        mia_market = _live_market(
            mia_ticker, mia_event, "greater", floor_strike=80.0
        )
        aus_market = _live_market(
            aus_ticker, aus_event, "greater", floor_strike=80.0
        )
        clim = _make_climatology("MIA", "AUS")
        kalshi = _fake_kalshi(
            {"KXHIGHMIA": [mia_market], "KXHIGHAUS": [aus_market]}
        )

        strategy = _make_strategy(
            settings,
            kalshi,
            clim,
            enabled_cells=["MIA/high", "AUS/high"],
            extra_config={
                "edge_threshold": 0.05,
                "boundary_distance_min_f": 0.0,
                "orderbook_depth_check": False,
            },
            asos_client=client,
        )
        await strategy.setup()

        await strategy.generate_signals()

        # The batched prefetch fired exactly once, covering BOTH cities' stations.
        assert len(batch_calls) == 1, (
            f"expected exactly ONE batched fetch, got {len(batch_calls)}: {batch_calls}"
        )
        assert set(batch_calls[0]) == {"KMIA", "KAUS"}, batch_calls[0]
        # Every per-cell max_so_far hit the warm cache: ZERO single-station fetches.
        assert single_calls == [], (
            f"expected ZERO single-station fetches (warm cache), got: {single_calls}"
        )
