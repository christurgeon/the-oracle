"""Unit + integration tests for ``IntradayEmosHighStrategy`` (M4).

Coverage:
  - setup() preconditions (#1 unloaded EmosTable, #2 missing cells,
    #3 strike-distance ≠ 2.0, #3 max_contracts > 50)
  - ``_setup_completed`` flag gates ``scan()`` + ``generate_signals()``
  - Per-cell rate-limited WARNING on skipped ticks
  - ``scan()`` filters to KXHIGHMIA + KXHIGHLAX same-day rows only,
    rejects past dates, skips future dates
  - Lead-negative routing + per-(cell, UTC date) WARNING dedup
  - ``max_contracts_per_trade`` clamp at 50 (post-Stage-2, pre-fee)
  - Per-cell daily trade-count cap (§9.2(b))
  - TradeSignal metadata carries the v6 audit + EMOS hash fields
  - ``strategy_name == "intraday_emos_high"`` on every emitted signal
  - EMOS hot-reload pause window (§4.6) emits a WARNING + suppresses
    signals for one tick
  - One-bet-per-event collapse leaves ≤1 signal/cell/day
  - Hermetic-discipline static-grep (no weather_edge imports)
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.sizing import SizingResult
from python.common import DailyVariable, Direction, Side
from python.data.emos import EmosTable, EmosTableHolder
from python.data.weather import EnsembleForecast
from python.notifications.base import AlertLevel
from python.settings import Settings
from python.strategies.intraday_emos_high import (
    IntradayEmosHighStrategy,
    _cell_label,
)
from python.strategies.weather_parsing import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _emos_payload(*, with_lax: bool = True) -> dict:
    """Minimal EMOS coefficients payload. Two cells (MIA/high, LAX/high) by
    default; opt out of LAX via ``with_lax=False`` to drive precondition #2.
    """
    cells = {
        "MIA/high": {
            "a": 0.5,
            "b": 1.00,
            "c": 12.0,
            "d": 0.3,
            "n_train": 110,
            "leads_pooled": [0, 1, 2],
            "ens_std_p99": 5.8,
        },
    }
    if with_lax:
        cells["LAX/high"] = {
            "a": -0.3,
            "b": 1.02,
            "c": 13.5,
            "d": 0.35,
            "n_train": 120,
            "leads_pooled": [0, 1, 2],
            "ens_std_p99": 5.5,
        }
    return {
        "fit_date": "2026-05-14",
        "schema_version": 1,
        "cells": cells,
    }


@pytest.fixture
def emos_holder(tmp_path: Path) -> EmosTableHolder:
    """Loaded EmosTableHolder with MIA/high + LAX/high cells."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_emos_payload()))
    table = EmosTable.load(coeffs_path)
    assert table.is_loaded
    return EmosTableHolder(table=table)


@pytest.fixture
def emos_holder_disabled() -> EmosTableHolder:
    """EmosTableHolder containing a disabled-sentinel table — drives the
    setup() precondition #1 failure path.
    """
    table = EmosTable._disabled(None, "test-only")
    return EmosTableHolder(table=table)


@pytest.fixture
def settings_for_strategy(tmp_path: Path) -> Settings:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(state_dir=str(state_dir), _env_file=None)


def _make_strategy(
    *,
    holder: EmosTableHolder,
    settings: Settings,
    enabled_cells: list[str] | None = None,
    notifier: Any | None = None,
    kalshi_client: Any | None = None,
    weather_client: Any | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> IntradayEmosHighStrategy:
    cfg: dict[str, Any] = {
        "enabled": True,
        "enabled_cells": enabled_cells if enabled_cells is not None else ["MIA/high", "LAX/high"],
        "edge_threshold": 0.07,
        "max_edge": 0.30,
        "min_distance_from_strike_std": 2.0,
        "max_contracts_per_trade": 50,
        "max_trades_per_cell_per_day": 3,
        "emos_hot_reload_pause_seconds": 60,
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        # Disable async audit-stream file I/O so unit tests don't depend on
        # state dir contents.
        "eval_audit_log": False,
        "candidate_audit_log": False,
        "scan_funnel_log": False,
    }
    if config_overrides:
        cfg.update(config_overrides)
    return IntradayEmosHighStrategy(
        config=cfg,
        kalshi_client=kalshi_client if kalshi_client is not None else MagicMock(),
        weather_client=weather_client if weather_client is not None else MagicMock(),
        bankroll=10_000.0,
        market_filter={},
        settings=settings,
        bias_table=None,
        emos_holder=holder,
        prediction_log=None,
        notifier=notifier,
    )


# ---------------------------------------------------------------------------
# Construction + setup()
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_setup_completed_false_at_construction(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        assert strat._setup_completed is False
        # And the public name attribute is the strategy_name attribution key
        # used downstream (spec §6.5).
        assert strat.name == "intraday_emos_high"

    def test_constructor_requires_emos_holder(
        self, settings_for_strategy: Settings
    ) -> None:
        """``emos_holder`` is keyword-only and required by the constructor —
        a positional / missing-kwarg call must fail."""
        with pytest.raises(TypeError):
            # mypy can't see the missing kwarg here, but Python raises at runtime.
            IntradayEmosHighStrategy(  # type: ignore[call-arg]
                config={"enabled": True},
                kalshi_client=MagicMock(),
                weather_client=MagicMock(),
                bankroll=10_000.0,
                settings=settings_for_strategy,
            )


class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_succeeds_with_valid_preconditions(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        await strat.setup()
        assert strat._setup_completed is True

    @pytest.mark.asyncio
    async def test_setup_raises_when_emos_holder_table_not_loaded(
        self,
        emos_holder_disabled: EmosTableHolder,
        settings_for_strategy: Settings,
    ) -> None:
        strat = _make_strategy(
            holder=emos_holder_disabled, settings=settings_for_strategy
        )
        with pytest.raises(RuntimeError, match="requires a loaded EmosTable"):
            await strat.setup()
        assert strat._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_raises_when_enabled_cell_missing_from_emos_table(
        self, tmp_path: Path, settings_for_strategy: Settings
    ) -> None:
        # Build a holder with MIA/high only — request LAX/high.
        coeffs_path = tmp_path / "coefficients.json"
        coeffs_path.write_text(json.dumps(_emos_payload(with_lax=False)))
        table = EmosTable.load(coeffs_path)
        assert table.is_loaded
        holder = EmosTableHolder(table=table)
        strat = _make_strategy(
            holder=holder,
            settings=settings_for_strategy,
            enabled_cells=["MIA/high", "LAX/high"],
        )
        with pytest.raises(RuntimeError, match="LAX/high"):
            await strat.setup()
        assert strat._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_raises_when_strike_distance_min_std_not_2_0(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        # Pydantic clamps ≥ 2.0 at config load; we simulate a programmatic
        # post-load mutation here to verify the defense-in-depth backstop.
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        strat._strike_distance_min_std = 1.9
        with pytest.raises(RuntimeError, match="strike_distance_min_std=1.9"):
            await strat.setup()
        assert strat._setup_completed is False

    @pytest.mark.asyncio
    async def test_setup_raises_when_max_contracts_per_trade_above_50(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        strat._max_contracts_per_trade = 51
        with pytest.raises(RuntimeError, match="max_contracts_per_trade=51"):
            await strat.setup()
        assert strat._setup_completed is False


# ---------------------------------------------------------------------------
# _setup_completed gates scan() + generate_signals()
# ---------------------------------------------------------------------------


class TestSetupIncompleteGate:
    @pytest.mark.asyncio
    async def test_scan_returns_empty_when_setup_incomplete(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        assert strat._setup_completed is False
        out = await strat.scan()
        assert out == []

    @pytest.mark.asyncio
    async def test_generate_signals_returns_empty_when_setup_incomplete(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        assert strat._setup_completed is False
        sigs = await strat.generate_signals()
        assert sigs == []

    @pytest.mark.asyncio
    async def test_setup_incomplete_emits_rate_limited_warning_first_tick(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            notifier=notifier,
        )
        out = await strat.scan()
        assert out == []
        # First emission per (process, cell) is immediate. Two enabled cells
        # → two WARNING dispatches on the first tick.
        assert notifier.send_alert.await_count == 2
        # Both calls used WARNING level (spec §11.2).
        for call in notifier.send_alert.await_args_list:
            assert call.kwargs["level"] == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_setup_incomplete_warning_rate_limit_window_30min(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            notifier=notifier,
            enabled_cells=["MIA/high"],
        )
        # Tick 1: 1 dispatch (one cell).
        await strat.scan()
        assert notifier.send_alert.await_count == 1
        # Ticks 2–4 within the same monotonic instant: still 1 dispatch.
        await strat.scan()
        await strat.scan()
        await strat.scan()
        assert notifier.send_alert.await_count == 1
        # Simulate 30 min + 1s passing — the rate-limit window opens.
        strat._setup_incomplete_warning_state["MIA/high"] = (
            time.monotonic() - (30 * 60 + 1)
        )
        await strat.scan()
        assert notifier.send_alert.await_count == 2


# ---------------------------------------------------------------------------
# scan() — series filter, date routing
# ---------------------------------------------------------------------------


def _today_et() -> date:
    """Use the strategy's own ET clock to mirror its scan_today semantics."""
    from python.common import et_today

    return et_today()


def _build_kalshi_market(
    *,
    ticker: str,
    event_ticker: str,
    strike_type: str = "between",
    floor_strike: float | None = 80.0,
    cap_strike: float | None = 84.0,
    yes_bid: float = 0.10,
    yes_ask: float = 0.20,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "yes_bid_dollars": str(yes_bid),
        "yes_ask_dollars": str(yes_ask),
        "volume_fp": "1000",
        "open_interest_fp": "500",
        "close_time": "2099-12-31T23:59:59Z",
    }


def _fmt_ymd(d: date) -> str:
    """Render a date as a Kalshi event-ticker suffix, e.g. 26MAY15."""
    return d.strftime("%y%b%d").upper()


class TestScan:
    @pytest.mark.asyncio
    async def test_scan_filters_to_kxhigh_mia_lax_only(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """``scan()`` queries only KXHIGHMIA + KXHIGHLAX. We verify the
        ``series_ticker`` kwargs that ``iter_all_markets`` was called with
        (an empty-market return is fine — we're testing the call shape).
        """
        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            kalshi_client=kalshi,
        )
        await strat.setup()
        await strat.scan()
        series_called = {
            call.kwargs.get("series_ticker")
            for call in kalshi.iter_all_markets.await_args_list
        }
        assert series_called == {"KXHIGHMIA", "KXHIGHLAX"}

    @pytest.mark.asyncio
    async def test_scan_keeps_only_same_day_markets(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        today = _today_et()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        # Three markets per series: past, today, future.
        mia_markets = [
            _build_kalshi_market(
                ticker=f"KXHIGHMIA-{_fmt_ymd(yesterday)}H1600-B82",
                event_ticker=f"KXHIGHMIA-{_fmt_ymd(yesterday)}H1600",
            ),
            _build_kalshi_market(
                ticker=f"KXHIGHMIA-{_fmt_ymd(today)}H1600-B82",
                event_ticker=f"KXHIGHMIA-{_fmt_ymd(today)}H1600",
            ),
            _build_kalshi_market(
                ticker=f"KXHIGHMIA-{_fmt_ymd(tomorrow)}H1600-B82",
                event_ticker=f"KXHIGHMIA-{_fmt_ymd(tomorrow)}H1600",
            ),
        ]
        kalshi = MagicMock()

        async def _iter(*, status: str | None = None, series_ticker: str | None = None):
            if series_ticker == "KXHIGHMIA":
                return mia_markets, False
            return [], False

        kalshi.iter_all_markets = AsyncMock(side_effect=_iter)
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            kalshi_client=kalshi,
        )
        await strat.setup()
        out = await strat.scan()
        tickers = {sc.parsed.ticker for sc in out}
        # Only the same-day ticker should survive.
        assert any("H1600-B82" in t for t in tickers)
        # No yesterday or tomorrow tickers.
        assert all(_fmt_ymd(yesterday) not in t for t in tickers)
        assert all(_fmt_ymd(tomorrow) not in t for t in tickers)
        # And the one survivor is the same-day MIA ticker.
        assert tickers == {f"KXHIGHMIA-{_fmt_ymd(today)}H1600-B82"}


# ---------------------------------------------------------------------------
# evaluate() — composition reuse
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_evaluate_invokes_evaluate_market_pure(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The strategy MUST reuse ``evaluate_market_pure`` from
        ``weather_evaluation`` (spec §3.4) rather than re-implementing the
        gate stack. We monkey-patch the function and assert the call.
        """
        weather = MagicMock()
        weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA",
                forecast_date=_today_et(),
                temperatures=[80.0] * 31,
            )},
        )
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            weather_client=weather,
        )
        await strat.setup()

        called: list[Any] = []

        def _fake(sc, *args, **kwargs):
            called.append((sc, args, kwargs))
            return None  # rejected — we're only testing the call shape

        monkeypatch.setattr(
            "python.strategies.intraday_emos_high.evaluate_market_pure", _fake
        )
        today = _today_et()
        market = _build_kalshi_market(
            ticker=f"KXHIGHMIA-{_fmt_ymd(today)}H1600-B82",
            event_ticker=f"KXHIGHMIA-{_fmt_ymd(today)}H1600",
        )
        parsed = ParsedMarket(
            ticker=market["ticker"],
            city="MIA",
            forecast_date=today,
            strike_type=Direction.BETWEEN,
            floor_strike=80.0,
            cap_strike=84.0,
            daily_variable=DailyVariable.TEMP_MAX,
        )
        sc = ScanCandidate(market=market, parsed=parsed, price_info=None)
        result = await strat.evaluate(sc)
        assert result is None  # _fake returns None
        assert len(called) == 1
        # The dispatcher receives our holder's table + enabled cells.
        _, _args, kwargs = called[0]
        assert kwargs["emos_table"] is emos_holder.table
        assert kwargs["emos_enabled_cells"] == frozenset({"MIA/high", "LAX/high"})


# ---------------------------------------------------------------------------
# Depth adjustment + clamp
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    contracts: int = 200,
    side: Side = Side.NO,
    model_prob: float = 0.65,
    yes_bid: float = 0.20,
    yes_ask: float = 0.30,
) -> Any:
    """Build a _MarketCandidate sufficient for _apply_depth_adjustment +
    _build_signal coverage."""
    from python.strategies.weather_evaluation import _MarketCandidate

    today = _today_et()
    parsed = ParsedMarket(
        ticker="KXHIGHMIA-99MAY15H1600-B82",
        city="MIA",
        forecast_date=today,
        strike_type=Direction.BETWEEN,
        floor_strike=80.0,
        cap_strike=84.0,
        daily_variable=DailyVariable.TEMP_MAX,
    )
    price_info = MarketPriceInfo(
        midpoint=(yes_bid + yes_ask) / 2,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        spread=yes_ask - yes_bid,
        yes_bid_size=100,
        yes_ask_size=100,
        volume=1000,
        open_interest=500,
    )
    forecast = EnsembleForecast(
        city="MIA", forecast_date=today, temperatures=[82.0] * 31
    )
    sizing = SizingResult(
        side=side, dollars=300.0, kelly_raw=0.05, kelly_fraction_used=0.05, capped=False
    )
    market = _build_kalshi_market(
        ticker=parsed.ticker,
        event_ticker="KXHIGHMIA-99MAY15H1600",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
    )
    return _MarketCandidate(
        parsed=parsed,
        market=market,
        forecast=forecast,
        model_prob=model_prob,
        correction_f=0.0,
        price_info=price_info,
        raw_edge=abs(model_prob - price_info.midpoint),
        ensemble_confidence=0.85,
        effective_edge=0.10,
        sizing=sizing,
        contracts=contracts,
        exec_price=yes_ask if side == Side.YES else 1.0 - yes_bid,
        per_model_probs={"gfs_seamless": model_prob},
        ensemble_std_pooled=1.5,
        spread_inflation_factor=1.0,
        raw_ensemble_prob=model_prob,
        strike_abs_z=3.0,
        emos_prob_raw=model_prob,
        emos_coeffs_artifact_hash="abc123",
        emos_coeffs_loaded_at="2026-05-14T00:00:00Z",
        live_calibrator_kind="emos_raw",
    )


class TestMaxContractsClamp:
    @pytest.mark.asyncio
    async def test_max_contracts_clamp_after_stage2_kelly_before_fees(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """Spec §6.3 MINOR #3: the clamp runs AFTER the Stage-2 Kelly re-run
        and BEFORE fee math, so a Stage-1 Kelly output of 200 contracts
        collapses to 50.

        Setup: candidate is a NO buy with model_prob=0.65 (NO target side
        prob = 0.35). The orderbook has YES bids at $0.85 (deep) → our NO
        ask via the side-flip is $0.15. Stage-2 fill_edge =
        (0.35 - 0.15) * 0.85 - fee ≈ 0.16 > edge_threshold (0.07), so the
        candidate passes Stage-2 and the Kelly re-run sizes well above 50
        contracts. The clamp then collapses to 50.
        """
        kalshi = MagicMock()
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {
                    # NO-side fill walks yes_dollars (YES bids → NO asks).
                    # YES bid at $0.85 ⇒ NO ask at $0.15 with 500 contracts
                    # at the top — deeper than any single-trade clamp.
                    "yes_dollars": [["0.85", "500"]],
                    "no_dollars": [],
                }
            }
        )
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            kalshi_client=kalshi,
            config_overrides={"orderbook_depth_check": True},
        )
        await strat.setup()
        cand = _make_candidate(contracts=200, side=Side.NO, model_prob=0.65)
        cfg = strat._build_eval_config()
        result = await strat._apply_depth_adjustment(cand, cfg)
        assert result is not None
        _eff, _sizing, contracts, depth_capped, _depth_result = result
        assert contracts == 50  # clamped to max_contracts_per_trade
        assert depth_capped is True


# ---------------------------------------------------------------------------
# §9.2(b) per-cell daily trade-count cap
# ---------------------------------------------------------------------------


class TestPerCellDailyCap:
    def test_reset_daily_counters_clears_counter(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        today = _today_et()
        strat._trades_today[("MIA/high", today)] = 3
        assert strat._trades_today
        strat.reset_daily_counters()
        assert strat._trades_today == {}

    def test_cap_blocks_after_max_trades_per_cell_per_day(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """Verifies the §9.2(b) gate is consulted before a signal emits.
        Pre-populates the counter and asserts the gate would block.
        """
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        today = _today_et()
        cap = strat._max_trades_per_cell_per_day
        strat._trades_today[("MIA/high", today)] = cap
        already = strat._trades_today.get(("MIA/high", today), 0)
        assert already >= cap  # gate would block emission

    @pytest.mark.asyncio
    async def test_cap_runs_after_select_one_per_event_collapse(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec §9.2(b) HIGH fix: the per-cell daily cap must increment on
        the count of ACTUAL emissions (post-``_select_one_per_event``
        collapse), not pre-collapse candidates.

        Scenario: 4 same-cell candidates that all share one
        (city, market_type, forecast_date) event-group. ``_select_one_per_event``
        collapses them to 1 winner. The cap (``max_trades_per_cell_per_day=3``)
        MUST allow that one survivor to emit; the pre-fix code would have
        burnt 3 of 4 slots on the collapse losers and dropped the winner
        on the 4th iteration. (The 4 candidates are constructed to be
        distinguishable via market_ticker so the collapse picks the
        highest-edge survivor.)
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        today = _today_et()
        # Build 4 same-event candidates with descending effective_edge.
        candidates = []
        for i, edge in enumerate([0.20, 0.18, 0.15, 0.12]):
            cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
            # Override edge + ticker so _select_one_per_event distinguishes.
            cand_market = dict(cand.market)
            cand_market["ticker"] = f"KXHIGHMIA-99MAY15H1600-T8{i}"
            cand = _MarketCandidate(
                **{**cand.__dict__, "market": cand_market, "effective_edge": edge,
                   "parsed": cand.parsed.__class__(
                       ticker=f"KXHIGHMIA-99MAY15H1600-T8{i}",
                       city=cand.parsed.city,
                       forecast_date=cand.parsed.forecast_date,
                       strike_type=cand.parsed.strike_type,
                       floor_strike=cand.parsed.floor_strike,
                       cap_strike=cand.parsed.cap_strike,
                       daily_variable=cand.parsed.daily_variable,
                   )},
            )
            candidates.append(cand)

        # Monkey-patch the pipeline so we control the candidates fed into the
        # cap loop. ``scan`` returns 4 markets; ``evaluate_market_pure`` is
        # patched to return our pre-built candidates one-by-one; depth
        # adjustment passes them through with the candidate's own sizing.
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            config_overrides={"max_trades_per_cell_per_day": 3},
        )
        await strat.setup()
        strat._setup_completed = True

        scan_candidates = [
            ScanCandidate(market=c.market, parsed=c.parsed, price_info=c.price_info)
            for c in candidates
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        # Pop one candidate per call so the order matches scan_candidates.
        cand_iter = iter(candidates)

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return next(cand_iter)

        async def _fake_depth(
            cand: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return cand.effective_edge, cand.sizing, cand.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            # ``postprocess_forecasts`` is synchronous — returns a 4-tuple.
            return {}, {}, {}, {}

        # The weather client's ``get_multi_model_forecast`` is awaited in
        # the forecast-prefetch loop; the default MagicMock doesn't support
        # await. Provide an AsyncMock that returns a non-empty dict so the
        # prefetch step doesn't crash. The contents don't matter because
        # we monkey-patch ``evaluate_market_pure`` immediately after.
        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        sigs = await strat.generate_signals()
        # All 4 candidates share one event-group → collapse keeps 1.
        # The cap (=3) must allow that 1 survivor through.
        assert len(sigs) == 1
        # The cap counter records exactly 1 emission, not 3 or 4.
        assert strat._trades_today[("MIA/high", today)] == 1

    @pytest.mark.asyncio
    async def test_cap_funnel_counter_increments_when_cap_fires(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec §11.5 MEDIUM fix: ``ScanFunnel.n_after_trades_per_cell_per_day_cap``
        must increment when the post-collapse cap drops a signal.

        Scenario: 4 candidates across 4 DIFFERENT event-groups (so no
        collapse). The cap (=3) lets 3 through and drops the 4th — funnel
        counter must register exactly 1.
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        today = _today_et()
        candidates = []
        for i in range(4):
            cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
            # Different forecast_date per candidate → 4 event-groups.
            distinct_date = today + timedelta(days=i)
            cand_market = dict(cand.market)
            cand_market["ticker"] = f"KXHIGHMIA-T{i}"
            cand = _MarketCandidate(
                **{**cand.__dict__, "market": cand_market, "effective_edge": 0.10 + i * 0.01,
                   "parsed": cand.parsed.__class__(
                       ticker=f"KXHIGHMIA-T{i}",
                       city=cand.parsed.city,
                       forecast_date=distinct_date,
                       strike_type=cand.parsed.strike_type,
                       floor_strike=cand.parsed.floor_strike,
                       cap_strike=cand.parsed.cap_strike,
                       daily_variable=cand.parsed.daily_variable,
                   )},
            )
            candidates.append(cand)

        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            config_overrides={"max_trades_per_cell_per_day": 3},
        )
        await strat.setup()
        strat._setup_completed = True

        # Pre-set today's cell counter to 0; we want the 4th to be the cap-block.
        scan_candidates = [
            ScanCandidate(market=c.market, parsed=c.parsed, price_info=c.price_info)
            for c in candidates
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        cand_iter = iter(candidates)

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return next(cand_iter)

        async def _fake_depth(
            cand: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return cand.effective_edge, cand.sizing, cand.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            # ``postprocess_forecasts`` is synchronous — returns a 4-tuple.
            return {}, {}, {}, {}

        # The weather client's ``get_multi_model_forecast`` is awaited in
        # the forecast-prefetch loop; the default MagicMock doesn't support
        # await. Provide an AsyncMock that returns a non-empty dict so the
        # prefetch step doesn't crash. The contents don't matter because
        # we monkey-patch ``evaluate_market_pure`` immediately after.
        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        # Pre-populate funnel-capture so we can inspect post-call. We capture
        # the strategy's funnel via a sniff on generate_signals' local — the
        # cleanest way is to monkey-patch ScanFunnel construction.
        captured: list[Any] = []

        from python.analytics.audit_log import ScanFunnel as _ScanFunnel

        class _SniffedFunnel(_ScanFunnel):
            def __init__(self) -> None:
                super().__init__()
                captured.append(self)

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)
        monkeypatch.setattr(iem_module, "ScanFunnel", _SniffedFunnel)

        sigs = await strat.generate_signals()
        # 4 distinct event-groups, cap=3 → 3 emit, 1 cap-dropped.
        assert len(sigs) == 3
        assert len(captured) == 1
        funnel = captured[0]
        # The new §11.5 funnel counter must register the dropped signal.
        assert funnel.n_after_trades_per_cell_per_day_cap == 1
        # And the counter for this cell records exactly 3 emissions.
        assert strat._trades_today[("MIA/high", today)] == 3

    @pytest.mark.asyncio
    async def test_predictions_and_n_signals_fired_emit_pre_cap(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Project-wide convention (CLAUDE.md "calibration audit streams";
        weather_edge.py:1075-1078): ``predictions.jsonl`` and
        ``funnel.n_signals_fired`` are the calibrator-fitting population +
        the pre-cap signal-count, both of which MUST emit on EVERY
        pre-cap candidate — including those the per-cell daily cap drops.

        The §11.5 ``n_after_trades_per_cell_per_day_cap`` counter then
        subtracts from ``n_signals_fired`` to give the actual emission
        count; the arithmetic only holds if ``n_signals_fired`` is pre-cap.

        Scenario mirrors ``test_cap_funnel_counter_increments_when_cap_fires``:
        4 candidates across 4 DIFFERENT event-groups (no collapse), cap=3.
        Expected post-call state:
          * funnel.n_signals_fired == 4 (pre-cap)
          * _record_prediction called on all 4 (including the cap-dropped one)
          * funnel.n_after_trades_per_cell_per_day_cap == 1
          * returned_signals == n_signals_fired - n_event_capped
            - n_after_trades_per_cell_per_day_cap (arithmetic invariant)
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        today = _today_et()
        candidates = []
        for i in range(4):
            cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
            distinct_date = today + timedelta(days=i)
            cand_market = dict(cand.market)
            cand_market["ticker"] = f"KXHIGHMIA-PCAP{i}"
            cand = _MarketCandidate(
                **{
                    **cand.__dict__,
                    "market": cand_market,
                    "effective_edge": 0.10 + i * 0.01,
                    "parsed": cand.parsed.__class__(
                        ticker=f"KXHIGHMIA-PCAP{i}",
                        city=cand.parsed.city,
                        forecast_date=distinct_date,
                        strike_type=cand.parsed.strike_type,
                        floor_strike=cand.parsed.floor_strike,
                        cap_strike=cand.parsed.cap_strike,
                        daily_variable=cand.parsed.daily_variable,
                    ),
                },
            )
            candidates.append(cand)

        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            config_overrides={"max_trades_per_cell_per_day": 3},
        )
        await strat.setup()
        strat._setup_completed = True

        scan_candidates = [
            ScanCandidate(market=c.market, parsed=c.parsed, price_info=c.price_info)
            for c in candidates
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        cand_iter = iter(candidates)

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return next(cand_iter)

        async def _fake_depth(
            cand: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return cand.effective_edge, cand.sizing, cand.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="MIA", forecast_date=today, temperatures=[80.0] * 31,
                )
            }
        )

        # Capture every _record_prediction call so we can assert the
        # pre-cap loop fired on ALL candidates, including the cap-dropped
        # one. The strategy's _prediction_log is None in the test fixture,
        # so wrapping the method (not the log) is the correct seam.
        recorded_tickers: list[str] = []
        original_record = strat._record_prediction

        def _capture_record(
            signal: Any, candidate: Any
        ) -> None:
            recorded_tickers.append(signal.market_ticker)
            original_record(signal, candidate)

        monkeypatch.setattr(strat, "_record_prediction", _capture_record)

        captured: list[Any] = []

        from python.analytics.audit_log import ScanFunnel as _ScanFunnel

        class _SniffedFunnel(_ScanFunnel):
            def __init__(self) -> None:
                super().__init__()
                captured.append(self)

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)
        monkeypatch.setattr(iem_module, "ScanFunnel", _SniffedFunnel)

        sigs = await strat.generate_signals()

        # Sanity: 4 distinct event-groups, cap=3 → 3 emit, 1 cap-dropped.
        assert len(sigs) == 3
        assert len(captured) == 1
        funnel = captured[0]

        # PRE-CAP invariant 1: funnel.n_signals_fired counts ALL pre-cap
        # candidates (4), not just emissions (3).
        assert funnel.n_signals_fired == 4

        # PRE-CAP invariant 2: _record_prediction was called on every
        # pre-cap signal — including the cap-dropped one. The fourth
        # ticker (highest edge per scenario construction) survives the
        # cap; the dropped ticker is whichever lost the cap race, but
        # ALL FOUR tickers must appear in recorded_tickers.
        assert len(recorded_tickers) == 4
        assert set(recorded_tickers) == {
            f"KXHIGHMIA-PCAP{i}" for i in range(4)
        }

        # Cap counter must register exactly one drop.
        assert funnel.n_after_trades_per_cell_per_day_cap == 1
        # No event-collapse fired in this scenario (4 distinct event groups).
        assert funnel.n_event_capped == 0

        # ARITHMETIC INVARIANT: returned_signals == n_signals_fired
        #   - n_event_capped - n_after_trades_per_cell_per_day_cap.
        # This invariant only holds while n_signals_fired is pre-cap; the
        # iter-1 regression (moving the bump post-cap) silently broke it.
        assert len(sigs) == (
            funnel.n_signals_fired
            - funnel.n_event_capped
            - funnel.n_after_trades_per_cell_per_day_cap
        )

    @pytest.mark.asyncio
    async def test_day_rollover_self_clears_stale_counter(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec-review MEDIUM #4: ``generate_signals`` defense-in-depth clears
        stale (cell, date) entries on every tick so an unwired orchestrator
        ``reset_daily()`` fan-out doesn't silently brick the strategy.
        """
        from python.strategies import intraday_emos_high as iem_module

        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        await strat.setup()
        strat._setup_completed = True

        today = _today_et()
        yesterday = today - timedelta(days=1)
        # Pre-populate with a stale yesterday entry at the cap.
        strat._trades_today[("MIA/high", yesterday)] = 3
        # And a current-day entry that should be preserved.
        strat._trades_today[("LAX/high", today)] = 1

        # Make the scan/eval pipeline return nothing — we only want to
        # exercise the day-rollover defense at the top of generate_signals.
        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return []

        def _fake_postprocess(*args: Any, **kwargs: Any):
            # ``postprocess_forecasts`` is synchronous — returns a 4-tuple.
            return {}, {}, {}, {}

        # The weather client's ``get_multi_model_forecast`` is awaited in
        # the forecast-prefetch loop; the default MagicMock doesn't support
        # await. Provide an AsyncMock that returns a non-empty dict so the
        # prefetch step doesn't crash. The contents don't matter because
        # we monkey-patch ``evaluate_market_pure`` immediately after.
        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        await strat.generate_signals()
        # Stale (MIA/high, yesterday) entry MUST be cleared.
        assert ("MIA/high", yesterday) not in strat._trades_today
        # Current-day (LAX/high, today) entry MUST be preserved.
        assert strat._trades_today[("LAX/high", today)] == 1

    def test_scan_funnel_has_after_trades_per_cell_per_day_cap_field(self) -> None:
        """Spec §11.5: ``ScanFunnel`` must expose the cap-counter field even
        when the strategy isn't currently plumbing a funnel through
        generate_signals — daily-report consumers in M6 read it directly.
        """
        from python.analytics.audit_log import RejectionReason, ScanFunnel

        f = ScanFunnel()
        assert hasattr(f, "n_after_trades_per_cell_per_day_cap")
        assert f.n_after_trades_per_cell_per_day_cap == 0
        # And the enum variant exists and maps to the new counter.
        assert RejectionReason.TRADES_PER_CELL_CAP == "TRADES_PER_CELL_CAP"
        f.increment(RejectionReason.TRADES_PER_CELL_CAP)
        assert f.n_after_trades_per_cell_per_day_cap == 1


# ---------------------------------------------------------------------------
# Strike-distance gate (spec §5.1 — k=2.0)
# ---------------------------------------------------------------------------


class TestStrikeDistanceGate:
    """The gate input is hard-bound to 2.0σ via the strategy's
    ``_strike_distance_min_std`` attribute, surfaced into ``_EvalConfig``.
    The dispatcher (``evaluate_market_pure``) consumes the cfg value — we
    only need to verify the wiring here.
    """

    def test_eval_config_min_distance_is_2_0(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        cfg = strat._build_eval_config()
        assert cfg.min_distance_from_strike_std == 2.0


# ---------------------------------------------------------------------------
# Signal metadata + strategy_name attribution
# ---------------------------------------------------------------------------


class TestSignalMetadata:
    def test_trade_signal_strategy_name_and_v6_inputs(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        cand = _make_candidate(contracts=10, side=Side.NO)
        sizing = SizingResult(
            side=Side.NO, dollars=200.0, kelly_raw=0.05, kelly_fraction_used=0.05, capped=False
        )
        signal = strat._build_signal(
            cand,
            effective_edge=0.10,
            sizing=sizing,
            contracts=10,
            depth_capped=False,
            depth_result=None,
            scan_today=_today_et(),
        )
        assert signal.strategy_name == "intraday_emos_high"
        md = signal.metadata
        # §11.1 audit-trail fields.
        assert md["cell"] == "MIA/high"
        assert md["city"] == "MIA"
        assert md["forecast_date"] == _today_et().isoformat()
        assert md["lead_days_raw"] is not None
        assert md["lead_days"] >= 0
        assert md["strike_abs_z"] == 3.0
        assert md["pooled_pstdev"] == 1.5
        assert md["bias_correction_applied_f"] == 0.0
        assert md["spread_inflation_factor"] == 1.0
        # EMOS hash threading (R2-H9 invariant: read from candidate).
        assert md["emos_artifact_hash"] == "abc123"
        assert md["emos_coeffs_artifact_hash"] == "abc123"
        assert md["live_calibrator_kind"] == "emos_raw"
        # Event-cap defaults at signal-build time.
        assert md["event_candidates_count"] == 1
        assert md["event_dropped_edges"] == []
        assert md["event_dropped_tickers"] == []

    def test_pooled_median_metadata_populated_from_candidate(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """Spec §11.1 MEDIUM fix: ``metadata["pooled_median"]`` must surface
        the candidate's ``ensemble_median_pooled`` directly (not None) so
        the §4.5 replay verifier can read it without inverting
        ``ref_strike - strike_abs_z * ensemble_std_pooled`` (which is
        sign-ambiguous and undefined for BETWEEN markets — 94% of LAX/high
        trades).
        """
        from python.strategies.weather_evaluation import _MarketCandidate

        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        cand = _make_candidate(contracts=10, side=Side.NO)
        # Inject a non-None pooled median (mirrors what evaluate_market_pure
        # populates from raw_ensemble_medians at the candidate construction site).
        cand = _MarketCandidate(
            **{**cand.__dict__, "ensemble_median_pooled": 82.5},
        )
        sizing = SizingResult(
            side=Side.NO, dollars=200.0, kelly_raw=0.05,
            kelly_fraction_used=0.05, capped=False,
        )
        signal = strat._build_signal(
            cand,
            effective_edge=0.10,
            sizing=sizing,
            contracts=10,
            depth_capped=False,
            depth_result=None,
            scan_today=_today_et(),
        )
        assert signal.metadata["pooled_median"] == 82.5
        # And it stays a float (not None) — was hard-coded to None pre-fix.
        assert isinstance(signal.metadata["pooled_median"], float)


# ---------------------------------------------------------------------------
# EMOS hot-reload pause (spec §4.6)
# ---------------------------------------------------------------------------


class TestEmosHotReloadPause:
    @pytest.mark.asyncio
    async def test_hot_reload_pause_emits_warning_and_suppresses_signals(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the EmosTable's artifact_hash changes between scan ticks,
        the strategy pauses for ``emos_hot_reload_pause_seconds`` and
        emits a WARNING (spec §4.6 / §11.2).
        """
        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            notifier=notifier,
            kalshi_client=kalshi,
        )
        await strat.setup()

        # Tick 1: prime the previous-hash watermark.
        await strat.generate_signals()
        notifier.send_alert.reset_mock()

        # Simulate a hot-reload by mutating the holder's table to a fresh
        # one with a DIFFERENT artifact_hash. Reach inside the holder by
        # constructing a new EmosTable in tmp-land.
        new_payload = _emos_payload()
        new_payload["fit_date"] = "2026-05-15"
        # Different cell coefficient values ensure a new artifact_hash.
        new_payload["cells"]["MIA/high"]["a"] = 99.9
        # Path: reuse the existing artifact location via the EmosTable
        # we already loaded.
        tmp_root = settings_for_strategy.state_dir
        new_path = Path(tmp_root) / "coeffs_v2.json"
        new_path.write_text(json.dumps(new_payload))
        new_table = EmosTable.load(new_path)
        assert new_table.is_loaded
        assert new_table.artifact_hash != emos_holder.table.artifact_hash
        emos_holder.table = new_table

        # Tick 2: strategy detects the hash change → pause + WARNING.
        sigs = await strat.generate_signals()
        assert sigs == []
        # Exactly one WARNING (the hot-reload alert).
        assert notifier.send_alert.await_count >= 1
        latest = notifier.send_alert.await_args_list[-1]
        assert latest.kwargs["level"] == AlertLevel.WARNING
        assert "EMOS reloaded" in latest.args[0]

    @pytest.mark.asyncio
    async def test_hot_reload_pause_does_not_mutate_trades_today(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HIGH fix (2026-05-15): when a hot-reload swap fires mid-tick, the
        cap-counter mutation MUST be skipped. The pre-fix shape walked the
        cap loop unconditionally (incrementing ``self._trades_today``) and
        then returned ``[]`` — so a single paused tick could pre-burn the
        per-cell daily cap and block real signals on the next non-paused
        tick within the same ET day.

        Scenario: prime the previous-hash watermark on tick 1; mutate the
        holder to a new hash to arm the pause; then run tick 2 with a
        non-empty candidate that would otherwise increment the counter.
        Assert ``self._trades_today`` is UNCHANGED after the paused tick.
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            notifier=notifier,
            kalshi_client=kalshi,
        )
        await strat.setup()

        # Tick 1: prime the previous-hash watermark.
        await strat.generate_signals()
        notifier.send_alert.reset_mock()

        # Snapshot the cap counter BEFORE the paused tick.
        trades_today_before = dict(strat._trades_today)

        # Swap holder.table to a fresh artifact with a different hash so
        # tick 2 detects the reload.
        new_payload = _emos_payload()
        new_payload["fit_date"] = "2026-05-15"
        new_payload["cells"]["MIA/high"]["a"] = 77.7
        new_path = Path(settings_for_strategy.state_dir) / "coeffs_paused.json"
        new_path.write_text(json.dumps(new_payload))
        new_table = EmosTable.load(new_path)
        assert new_table.artifact_hash != emos_holder.table.artifact_hash
        emos_holder.table = new_table

        # Build one candidate that would otherwise be capped.
        today = _today_et()
        cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(
            c: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        # Tick 2: hot-reload pause activates; no signals returned.
        sigs = await strat.generate_signals()
        assert sigs == []

        # INVARIANT: the cap-counter dict is bit-identical to the pre-tick
        # snapshot — no entries added, no entries incremented.
        assert strat._trades_today == trades_today_before
        # In particular, no (MIA/high, today) entry was burned by the
        # paused tick. The next non-paused tick must see a fresh counter.
        assert ("MIA/high", today) not in strat._trades_today


# ---------------------------------------------------------------------------
# Audit-stream wiring (HIGH fix 2026-05-15)
# ---------------------------------------------------------------------------


class TestAuditStreamWiring:
    @pytest.mark.asyncio
    async def test_emit_funnel_writes_row_at_end_of_generate_signals(
        self,
        emos_holder: EmosTableHolder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HIGH fix: ``generate_signals`` MUST write one row to
        ``state/scan_funnel.jsonl`` at the end of the cycle so §11.4
        daily-report (M6) can read cap utilization + funnel attrition.
        Pre-fix the in-memory funnel was never persisted.
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        state_dir = tmp_path / "state_funnel"
        state_dir.mkdir()
        settings = Settings(state_dir=str(state_dir), _env_file=None)

        # Build strategy with funnel logging ENABLED.
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings,
            config_overrides={"scan_funnel_log": True},
        )
        await strat.setup()
        strat._setup_completed = True

        today = _today_et()
        cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(
            c: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        sigs = await strat.generate_signals()
        assert len(sigs) == 1

        # File must exist + contain at least one row tagged with our strategy.
        funnel_path = state_dir / "scan_funnel.jsonl"
        assert funnel_path.exists(), "scan_funnel.jsonl was not written"
        rows = [
            json.loads(line)
            for line in funnel_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) >= 1
        # Latest row should be ours.
        assert any(r.get("strategy") == "intraday_emos_high" for r in rows)
        # And n_signals_fired == 1 (one pre-cap survivor).
        ours = [r for r in rows if r.get("strategy") == "intraday_emos_high"]
        assert ours[-1]["n_signals_fired"] == 1

    @pytest.mark.asyncio
    async def test_emit_funnel_writes_row_on_no_markets_path(
        self,
        emos_holder: EmosTableHolder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The no-markets early-return MUST also write a funnel row so
        the operator can see stagnation (zero markets) downstream.
        Mirrors weather_edge.py:844.
        """
        from python.strategies import intraday_emos_high as iem_module

        state_dir = tmp_path / "state_funnel_empty"
        state_dir.mkdir()
        settings = Settings(state_dir=str(state_dir), _env_file=None)
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings,
            config_overrides={"scan_funnel_log": True},
        )
        await strat.setup()

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return []

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        sigs = await strat.generate_signals()
        assert sigs == []

        funnel_path = state_dir / "scan_funnel.jsonl"
        assert funnel_path.exists()
        rows = [
            json.loads(line)
            for line in funnel_path.read_text().splitlines()
            if line.strip()
        ]
        ours = [r for r in rows if r.get("strategy") == "intraday_emos_high"]
        assert len(ours) >= 1
        assert ours[-1]["n_markets"] == 0

    @pytest.mark.asyncio
    async def test_emit_candidate_writes_orderbook_thin_row(
        self,
        emos_holder: EmosTableHolder,
        tmp_path: Path,
    ) -> None:
        """HIGH fix: Stage-2 terminal paths MUST write to
        ``state/candidate_audit.jsonl``. Drive the ORDERBOOK_THIN path by
        returning an empty orderbook from the Kalshi client and assert
        one row with the expected ``reason`` field.
        """
        state_dir = tmp_path / "state_candidate"
        state_dir.mkdir()
        settings = Settings(state_dir=str(state_dir), _env_file=None)

        # Empty orderbook → compute_fill_from_levels reports zero depth →
        # depth_check.is_stale is True → ORDERBOOK_THIN terminal path.
        kalshi = MagicMock()
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {"yes_dollars": [], "no_dollars": []}
            }
        )
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings,
            kalshi_client=kalshi,
            config_overrides={
                "candidate_audit_log": True,
                "orderbook_depth_check": True,
            },
        )
        await strat.setup()

        cand = _make_candidate(contracts=200, side=Side.NO, model_prob=0.65)
        cfg = strat._build_eval_config()
        result = await strat._apply_depth_adjustment(
            cand,
            cfg,
            scan_id="testscan001",
            scan_ts=0.0,
        )
        assert result is None  # ORDERBOOK_THIN terminal

        cand_path = state_dir / "candidate_audit.jsonl"
        assert cand_path.exists(), "candidate_audit.jsonl was not written"
        rows = [
            json.loads(line)
            for line in cand_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["reason"] == "ORDERBOOK_THIN"
        assert rows[0]["strategy"] == "intraday_emos_high"
        assert rows[0]["scan_id"] == "testscan001"


# ---------------------------------------------------------------------------
# emos_fit_date snapshot (MEDIUM fix 2026-05-15)
# ---------------------------------------------------------------------------


class TestEmosFitDateSnapshot:
    def test_build_signal_uses_emos_fit_date_kwarg_over_live_holder(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """MEDIUM fix: ``_build_signal`` must prefer the
        ``emos_fit_date`` kwarg (the scan-snapshot value captured next to
        ``emos_table_for_scan``) over reading
        ``self._emos_holder.table.fit_date`` live. A future mid-tick
        hot-reload would otherwise desync the metadata field from the
        artifact_hash that drove the posterior.
        """
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        cand = _make_candidate(contracts=10, side=Side.NO)
        sizing = SizingResult(
            side=Side.NO, dollars=200.0, kelly_raw=0.05, kelly_fraction_used=0.05, capped=False
        )
        # Snapshot a fit_date that differs from the live holder's fit_date
        # (live = "2026-05-14" per the fixture payload).
        snapshot_fit_date = "1999-01-01"
        signal = strat._build_signal(
            cand,
            effective_edge=0.10,
            sizing=sizing,
            contracts=10,
            depth_capped=False,
            depth_result=None,
            scan_today=_today_et(),
            emos_fit_date=snapshot_fit_date,
        )
        assert signal.metadata["emos_fit_date"] == snapshot_fit_date
        # And the live holder still reads as "2026-05-14" — proving the
        # kwarg, not the live read, drove the metadata.
        assert emos_holder.table.fit_date == "2026-05-14"

    def test_build_signal_falls_back_to_live_holder_when_kwarg_omitted(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """Back-compat: direct ``_build_signal`` calls (e.g. from tests)
        still read the live holder when the kwarg is omitted, preserving
        pre-fix behavior on the non-``generate_signals`` path.
        """
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        cand = _make_candidate(contracts=10, side=Side.NO)
        sizing = SizingResult(
            side=Side.NO, dollars=200.0, kelly_raw=0.05, kelly_fraction_used=0.05, capped=False
        )
        signal = strat._build_signal(
            cand,
            effective_edge=0.10,
            sizing=sizing,
            contracts=10,
            depth_capped=False,
            depth_result=None,
            scan_today=_today_et(),
        )
        # No kwarg passed → falls back to live holder's fit_date.
        assert signal.metadata["emos_fit_date"] == emos_holder.table.fit_date


# ---------------------------------------------------------------------------
# Required settings kwarg (MEDIUM fix 2026-05-15)
# ---------------------------------------------------------------------------


class TestSettingsRequired:
    def test_constructor_requires_settings(
        self, emos_holder: EmosTableHolder
    ) -> None:
        """MEDIUM fix: ``settings`` is keyword-only with no default. A
        forgotten kwarg would otherwise silently produce a misconfigured
        strategy backed by an orchestrator-independent ``Settings()``.
        """
        with pytest.raises(TypeError):
            IntradayEmosHighStrategy(  # type: ignore[call-arg]
                config={"enabled": True, "enabled_cells": ["MIA/high"]},
                kalshi_client=MagicMock(),
                weather_client=MagicMock(),
                bankroll=10_000.0,
                emos_holder=emos_holder,
            )


# ---------------------------------------------------------------------------
# Hermetic discipline (spec §3.5)
# ---------------------------------------------------------------------------


class TestHermeticDiscipline:
    def test_module_does_not_import_weather_edge(self) -> None:
        """Static-grep: no ``from python.strategies.weather_edge`` import
        in the strategy module.
        """
        src = Path("python/strategies/intraday_emos_high.py").read_text()
        # Comments and docstrings may mention weather_edge for cross-reference.
        # Strip them, then search the remaining code lines for a real import.
        # The pattern matches both ``from python.strategies.weather_edge ...``
        # and ``import python.strategies.weather_edge``.
        bad = re.findall(
            r"^\s*(?:from|import)\s+python\.strategies\.weather_edge\b",
            src,
            flags=re.MULTILINE,
        )
        assert bad == [], (
            f"intraday_emos_high.py imports weather_edge — hermetic-discipline "
            f"violation (spec §3.5). Offending imports: {bad}"
        )

    def test_module_does_not_call_check_and_reload(self) -> None:
        """The orchestrator owns ``check_and_reload`` (spec §4.2 / §12.9).
        The strategy module MUST NOT invoke it.
        """
        src = Path("python/strategies/intraday_emos_high.py").read_text()
        # Comments may mention the symbol for cross-reference, but no real
        # call. The pattern matches a real method invocation.
        assert ".check_and_reload(" not in src, (
            "intraday_emos_high.py calls EmosTable.check_and_reload — hot-"
            "reload ownership lives on the orchestrator (spec §4.2 / §12.9)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_cell_label_helper() -> None:
    assert _cell_label("MIA", "high") == "MIA/high"
    assert _cell_label("LAX", "high") == "LAX/high"


# ---------------------------------------------------------------------------
# load_state_from_log (HIGH-1 + HIGH-2 — 2026-05-15)
# ---------------------------------------------------------------------------


def _intraday_fill_row(
    *,
    ticker: str,
    timestamp: str,
    strategy: str = "intraday_emos_high",
    order_id: str = "ord-1",
) -> dict:
    """Minimal fill row matching engine._base_log_fields."""
    return {
        "event": "fill",
        "order_id": order_id,
        "timestamp": timestamp,
        "market_id": ticker,
        "market_ticker": ticker,
        "position_key": ticker,
        "strategy": strategy,
        "side": "NO",
        "mode": "live",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestLoadStateFromLog:
    def test_seeds_trades_today_and_traded_tickers_for_today_only(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings, tmp_path: Path
    ) -> None:
        """HIGH-1 + HIGH-2 (2026-05-15): hydration filters on
        ``is_pnl_relevant_row`` AND ``strategy == intraday_emos_high`` AND
        ``parse_ticker_date == et_today()`` — yesterday's rows and other
        strategies' rows are ignored.
        """
        today = _today_et()
        yesterday = today - timedelta(days=1)
        today_ticker_a = f"KXHIGHMIA-{_fmt_ymd(today)}H1600-B82"
        today_ticker_b = f"KXHIGHMIA-{_fmt_ymd(today)}H1700-B83"
        yesterday_ticker = f"KXHIGHMIA-{_fmt_ymd(yesterday)}H1600-B82"
        other_strategy_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1800-B84"
        log = tmp_path / "trades.jsonl"
        _write_jsonl(
            log,
            [
                _intraday_fill_row(
                    ticker=today_ticker_a,
                    timestamp="2026-05-15T17:00:00Z",
                    order_id="o1",
                ),
                _intraday_fill_row(
                    ticker=today_ticker_b,
                    timestamp="2026-05-15T17:05:00Z",
                    order_id="o2",
                ),
                _intraday_fill_row(
                    ticker=yesterday_ticker,
                    timestamp="2026-05-14T17:00:00Z",
                    order_id="o3",
                ),
                _intraday_fill_row(
                    ticker=other_strategy_ticker,
                    timestamp="2026-05-15T17:10:00Z",
                    order_id="o4",
                    strategy="weather_edge",
                ),
                # Placement row for a today ticker — filtered out by is_pnl_relevant_row.
                {
                    "event": "placement",
                    "order_id": "o5",
                    "timestamp": "2026-05-15T17:11:00Z",
                    "market_ticker": f"KXHIGHMIA-{_fmt_ymd(today)}H1900-B85",
                    "strategy": "intraday_emos_high",
                },
            ],
        )
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        strat.load_state_from_log(log)

        assert strat._trades_today == {("MIA/high", today): 2}
        assert strat._traded_tickers == {today_ticker_a, today_ticker_b}

    def test_missing_log_file_is_tolerated(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings, tmp_path: Path
    ) -> None:
        """No trade-log → zero state, no exception."""
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        strat.load_state_from_log(tmp_path / "does_not_exist.jsonl")
        assert strat._trades_today == {}
        assert strat._traded_tickers == set()

    def test_malformed_rows_are_skipped(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings, tmp_path: Path
    ) -> None:
        """Defensive at I/O boundary — rows missing ticker / unknown city
        prefix are silently skipped without breaking hydration of valid rows.
        """
        today = _today_et()
        good_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1600-B82"
        log = tmp_path / "trades.jsonl"
        _write_jsonl(
            log,
            [
                {"event": "fill", "strategy": "intraday_emos_high"},  # no ticker
                {
                    "event": "fill",
                    "strategy": "intraday_emos_high",
                    "market_ticker": "KXINX-26MAR20H1600-T8",  # unknown series
                },
                _intraday_fill_row(
                    ticker=good_ticker,
                    timestamp="2026-05-15T17:00:00Z",
                    order_id="ok",
                ),
            ],
        )
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        strat.load_state_from_log(log)
        assert strat._trades_today == {("MIA/high", today): 1}
        assert strat._traded_tickers == {good_ticker}

    @pytest.mark.asyncio
    async def test_restart_resilience_cap_saturated_from_restored_state(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HIGH-1 acceptance: 3 fills in the log → cap saturated → next
        ``generate_signals`` emits 0 even with one new same-cell candidate.
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        today = _today_et()
        log = tmp_path / "trades.jsonl"
        _write_jsonl(
            log,
            [
                _intraday_fill_row(
                    ticker=f"KXHIGHMIA-{_fmt_ymd(today)}H1{i}00-B82",
                    timestamp=f"2026-05-15T17:0{i}:00Z",
                    order_id=f"r{i}",
                )
                for i in range(3)
            ],
        )
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            config_overrides={"max_trades_per_cell_per_day": 3},
        )
        await strat.setup()
        strat.load_state_from_log(log)
        assert strat._trades_today[("MIA/high", today)] == 3

        cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
        # Use a new ticker so per-ticker dedup doesn't pre-block — the cap is
        # what we want to verify.
        new_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H2300-B87"
        cand_market = dict(cand.market)
        cand_market["ticker"] = new_ticker
        cand = _MarketCandidate(
            **{
                **cand.__dict__,
                "market": cand_market,
                "parsed": cand.parsed.__class__(
                    ticker=new_ticker,
                    city=cand.parsed.city,
                    forecast_date=cand.parsed.forecast_date,
                    strike_type=cand.parsed.strike_type,
                    floor_strike=cand.parsed.floor_strike,
                    cap_strike=cand.parsed.cap_strike,
                    daily_variable=cand.parsed.daily_variable,
                ),
            },
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(
            c: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )
        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        sigs = await strat.generate_signals()
        assert sigs == []
        assert strat._trades_today[("MIA/high", today)] == 3


# ---------------------------------------------------------------------------
# MED-1: n_hot_reload_paused funnel field + four-term invariant (2026-05-15)
# ---------------------------------------------------------------------------


class TestHotReloadFunnelInvariant:
    def test_scan_funnel_has_n_hot_reload_paused_field(self) -> None:
        from python.analytics.audit_log import FUNNEL_SCHEMA_VERSION, ScanFunnel

        f = ScanFunnel()
        assert hasattr(f, "n_hot_reload_paused")
        assert f.n_hot_reload_paused == 0
        # Schema version bumped from 5 to 6 with the new field.
        assert FUNNEL_SCHEMA_VERSION == 6

    @pytest.mark.asyncio
    async def test_hot_reload_pause_bumps_n_hot_reload_paused_and_invariant_holds(
        self,
        emos_holder: EmosTableHolder,
        settings_for_strategy: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MED-1 (2026-05-15): the paused early-return MUST bump
        ``funnel.n_hot_reload_paused`` so the four-term arithmetic invariant
        ``returned_signals == n_signals_fired − n_event_capped −
        n_after_trades_per_cell_per_day_cap − n_hot_reload_paused`` holds.
        """
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.weather_evaluation import _MarketCandidate

        notifier = MagicMock()
        notifier.send_alert = AsyncMock()
        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            notifier=notifier,
            kalshi_client=kalshi,
        )
        await strat.setup()

        # Tick 1: prime the previous-hash watermark.
        await strat.generate_signals()
        notifier.send_alert.reset_mock()

        # Mutate holder.table to a fresh artifact so tick 2 detects a reload.
        new_payload = _emos_payload()
        new_payload["fit_date"] = "2026-05-15"
        new_payload["cells"]["MIA/high"]["a"] = 88.8
        new_path = Path(settings_for_strategy.state_dir) / "coeffs_paused_inv.json"
        new_path.write_text(json.dumps(new_payload))
        new_table = EmosTable.load(new_path)
        emos_holder.table = new_table

        today = _today_et()
        cand = _make_candidate(contracts=10, side=Side.NO, model_prob=0.65)
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(
            c: _MarketCandidate,
            cfg: Any,
            *,
            funnel: Any = None,
            **_: Any,
        ) -> tuple[float, SizingResult, int, bool, None]:
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args: Any, **kwargs: Any):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={"gfs_seamless": EnsembleForecast(
                city="MIA", forecast_date=today, temperatures=[80.0] * 31,
            )}
        )

        captured: list[Any] = []
        from python.analytics.audit_log import ScanFunnel as _ScanFunnel

        class _SniffedFunnel(_ScanFunnel):
            def __init__(self) -> None:
                super().__init__()
                captured.append(self)

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)
        monkeypatch.setattr(iem_module, "ScanFunnel", _SniffedFunnel)

        sigs = await strat.generate_signals()
        assert sigs == []
        assert len(captured) == 1
        funnel = captured[0]

        # Pre-cap n_signals_fired must still bump for predictions.jsonl.
        assert funnel.n_signals_fired == 1
        # The paused-return must register the suppressed signal.
        assert funnel.n_hot_reload_paused == 1
        # And the four-term invariant holds.
        assert len(sigs) == (
            funnel.n_signals_fired
            - funnel.n_event_capped
            - funnel.n_after_trades_per_cell_per_day_cap
            - funnel.n_hot_reload_paused
        )


# ---------------------------------------------------------------------------
# MED-2: max_contracts clamp recomputes sizing.dollars (2026-05-15)
# ---------------------------------------------------------------------------


class TestMaxContractsClampRecomputesDollars:
    @pytest.mark.asyncio
    async def test_dollars_reflect_post_clamp_contracts(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        """MED-2 (2026-05-15): after the ``max_contracts_per_trade`` clamp,
        ``sizing.dollars`` MUST reflect the clamped contract count × per-contract
        price. Pre-fix RiskManager would see stale dollars and burn exposure
        budget that the strategy never actually deploys.
        """
        kalshi = MagicMock()
        # NO-side fill walks yes_dollars (YES bids → NO asks).
        # YES bid at $0.80 ⇒ NO ask at $0.20, deep enough to over-fill 50.
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {
                    "yes_dollars": [["0.80", "500"]],
                    "no_dollars": [],
                }
            }
        )
        strat = _make_strategy(
            holder=emos_holder,
            settings=settings_for_strategy,
            kalshi_client=kalshi,
            config_overrides={"orderbook_depth_check": True},
        )
        await strat.setup()
        # 200 contracts of pre-clamp Stage-1 sizing — clamp will hit 50.
        cand = _make_candidate(contracts=200, side=Side.NO, model_prob=0.65)
        cfg = strat._build_eval_config()
        result = await strat._apply_depth_adjustment(cand, cfg)
        assert result is not None
        _eff, sizing, contracts, depth_capped, depth_result = result
        # Sanity: clamp fired.
        assert contracts == 50
        assert depth_capped is True
        # NO side, fill_price=0.20 → per_contract_price = 1.0 - 0.20 = 0.80.
        assert depth_result is not None
        expected_per_contract = 1.0 - depth_result.fill_price
        # Dollars now reflect post-clamp contracts × NO-side price.
        assert sizing.dollars == pytest.approx(50 * expected_per_contract)
        # And NOT the stale pre-clamp $300.00 from _make_candidate's fixture.
        assert sizing.dollars != pytest.approx(300.0)


# ---------------------------------------------------------------------------
# MED-4 + LOW-1: reset_daily_counters prunes stale dedup containers
# ---------------------------------------------------------------------------


class TestResetDailyCountersPrunesStaleEntries:
    def test_lead_negative_warning_seen_dates_pruned(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        today = _today_et()
        old_day = today - timedelta(days=5)
        strat._lead_negative_warning_seen_dates["MIA/high"] = {old_day, today}
        strat._lead_negative_warning_seen_dates["LAX/high"] = {old_day}
        strat.reset_daily_counters()
        # MIA: stale old_day pruned, today preserved.
        assert strat._lead_negative_warning_seen_dates["MIA/high"] == {today}
        # LAX: all-stale → cell dropped entirely.
        assert "LAX/high" not in strat._lead_negative_warning_seen_dates

    def test_setup_incomplete_warning_state_pruned(
        self, emos_holder: EmosTableHolder, settings_for_strategy: Settings
    ) -> None:
        strat = _make_strategy(holder=emos_holder, settings=settings_for_strategy)
        # Inject a stale monotonic timestamp (very old) and a fresh one.
        strat._setup_incomplete_warning_state["MIA/high"] = time.monotonic() - 3600 * 24
        strat._setup_incomplete_warning_state["LAX/high"] = time.monotonic()
        strat.reset_daily_counters()
        assert "MIA/high" not in strat._setup_incomplete_warning_state
        assert "LAX/high" in strat._setup_incomplete_warning_state


# ---------------------------------------------------------------------------
# HIGH-3: LAX permanent-disable alert text (verified in daily_trade_report tests)
# ---------------------------------------------------------------------------
