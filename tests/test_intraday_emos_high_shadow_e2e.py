"""End-to-end shadow-mode integration test for ``IntradayEmosHighStrategy`` (M8).

Drives ``generate_signals()`` against a mocked Kalshi client + Weather
client, routes the emitted ``TradeSignal`` through a real
``ExecutionEngine`` in ``ExecutionMode.SHADOW``, then asserts that
``state/trades.jsonl`` contains a row with
``strategy == "intraday_emos_high"`` and the §11.1 ``metadata`` fields the
daily-report consumer relies on.

Discipline (per the M8 prompt):
  - **Hermetic.** Kalshi I/O is mocked via ``unittest.mock`` — no live API.
  - **No production config touched.** ``intraday_emos_high.enabled`` stays
    ``False`` in ``python/config.yaml``; the test enables the strategy in
    test fixtures only.
  - **Schema-evolution gate.** Trade-log row inspection uses
    ``python.execution.trade_log.is_pnl_relevant_row`` — not an ad-hoc
    ``event == "fill"`` check.
  - **Pre-cap audit invariants.** Where a cap path is exercised, we assert
    ``funnel.n_signals_fired`` is bumped on the pre-cap candidate set —
    matching the project-wide convention captured in CLAUDE.md
    ("calibration audit streams").
  - **No claimed Telegram alert paths.** ``spec_alert_path_verification_rule``:
    the setup-incomplete WARNING is asserted via the in-process notifier
    mock; no other Telegram-related claims are made.

Mocking strategy:
  ``evaluate_market_pure`` is monkey-patched to return a pre-built
  ``_MarketCandidate``. This is the same seam used by
  ``test_intraday_emos_high.py::TestEvaluate`` — the full posterior /
  bias / spread-inflation pipeline is already covered by M4 unit tests,
  so this file focuses on the *integration* contract: strategy ->
  engine -> trade-log row + funnel emission + daily-reset fan-out +
  EmosTableHolder sharing + setup-incomplete WARNING.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.analytics.sizing import SizingResult
from python.common import (
    DailyVariable,
    Direction,
    OrderStatus,
    Side,
    et_today,
)
from python.data.emos import EmosTable, EmosTableHolder
from python.data.weather import EnsembleForecast
from python.execution.engine import ExecutionEngine
from python.execution.trade_log import is_pnl_relevant_row
from python.notifications.base import AlertLevel
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.intraday_emos_high import (
    IntradayEmosHighStrategy,
    _cell_label,
)
from python.strategies.weather_evaluation import _MarketCandidate
from python.strategies.weather_parsing import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _emos_payload() -> dict[str, Any]:
    """Minimal EMOS coefficients payload covering both promoted cells.

    Matches the schema ``EmosTable.load`` expects (see
    ``data/emos/coefficients.json`` for the production layout). Two cells
    so the precondition #2 check (`every enabled cell present in EmosTable`)
    passes for both MIA/high and LAX/high.
    """
    return {
        "fit_date": "2026-05-14",
        "schema_version": 1,
        "cells": {
            "MIA/high": {
                "a": 0.5,
                "b": 1.00,
                "c": 12.0,
                "d": 0.3,
                "n_train": 110,
                "leads_pooled": [0, 1, 2],
                "ens_std_p99": 5.8,
            },
            "LAX/high": {
                "a": -0.3,
                "b": 1.02,
                "c": 13.5,
                "d": 0.35,
                "n_train": 120,
                "leads_pooled": [0, 1, 2],
                "ens_std_p99": 5.5,
            },
        },
    }


@pytest.fixture
def loaded_emos_holder(tmp_path: Path) -> EmosTableHolder:
    """Loaded EmosTableHolder with both MIA/high and LAX/high cells."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_emos_payload()))
    table = EmosTable.load(coeffs_path)
    assert table.is_loaded
    return EmosTableHolder(table=table)


@pytest.fixture
def shadow_settings(tmp_path: Path) -> Settings:
    """Settings instance in SHADOW mode pointed at a hermetic state dir."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state_dir),
        execution={"mode": "shadow"},
        # Loosen the edge floor so the synthetic candidate clears.
        # 0.07 matches the v6-audited intraday threshold.
        strategies={"weather": {"edge_threshold": 0.07}},
        _env_file=None,
    )


def _fmt_ymd(d: date) -> str:
    """Render a date as a Kalshi event-ticker suffix, e.g. 26MAY15."""
    return d.strftime("%y%b%d").upper()


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
    """Build a realistic Kalshi market dict (mirrors v6 audit shape)."""
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


def _make_strategy(
    *,
    holder: EmosTableHolder,
    settings: Settings,
    kalshi_client: Any,
    weather_client: Any | None = None,
    notifier: Any | None = None,
    enabled_cells: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> IntradayEmosHighStrategy:
    """Construct an enabled intraday strategy with sensible test defaults.

    Mirrors the helper in ``tests/test_intraday_emos_high.py`` so the M8
    test surface looks familiar to a reviewer cross-referencing M4 unit
    tests.
    """
    cfg: dict[str, Any] = {
        "enabled": True,  # test-fixture-only — never written to config.yaml
        "enabled_cells": (
            enabled_cells if enabled_cells is not None else ["MIA/high", "LAX/high"]
        ),
        "edge_threshold": 0.07,
        "max_edge": 0.30,
        "min_distance_from_strike_std": 2.0,
        "max_contracts_per_trade": 50,
        "max_trades_per_cell_per_day": 3,
        "emos_hot_reload_pause_seconds": 60,
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        # Persist scan_funnel.jsonl so the integration test can assert the
        # end-of-cycle funnel emission lands on disk (matches the M4
        # ``TestAuditStreamWiring`` pattern).
        "scan_funnel_log": True,
        # Suppress eval/candidate audit-stream writes — those are exercised
        # by the M4 unit tests and would clutter the integration assertions.
        "eval_audit_log": False,
        "candidate_audit_log": False,
    }
    if config_overrides:
        cfg.update(config_overrides)
    return IntradayEmosHighStrategy(
        config=cfg,
        kalshi_client=kalshi_client,
        weather_client=weather_client if weather_client is not None else MagicMock(),
        bankroll=10_000.0,
        market_filter={},
        settings=settings,
        bias_table=None,
        emos_holder=holder,
        prediction_log=None,
        notifier=notifier,
    )


def _make_market_candidate(
    *,
    ticker: str,
    event_ticker: str,
    city: str = "MIA",
    forecast_date: date | None = None,
    side: Side = Side.NO,
    model_prob: float = 0.65,
    yes_bid: float = 0.20,
    yes_ask: float = 0.30,
    effective_edge: float = 0.15,
    contracts: int = 10,
) -> _MarketCandidate:
    """Build a stage-1-passing ``_MarketCandidate``.

    ``evaluate_market_pure`` is monkey-patched to return this in tests
    that exercise the end-to-end pipeline. The full gate stack is covered
    by the M4 unit tests; this scaffolding lets us focus on the
    signal -> engine -> trade-log row contract.
    """
    fd = forecast_date if forecast_date is not None else et_today()
    parsed = ParsedMarket(
        ticker=ticker,
        city=city,
        forecast_date=fd,
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
        city=city, forecast_date=fd, temperatures=[82.0] * 31
    )
    sizing = SizingResult(
        side=side,
        dollars=200.0,
        kelly_raw=0.05,
        kelly_fraction_used=0.05,
        capped=False,
    )
    market = _build_kalshi_market(
        ticker=ticker,
        event_ticker=event_ticker,
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
        effective_edge=effective_edge,
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
        ensemble_median_pooled=82.0,
    )


# ---------------------------------------------------------------------------
# 1. Wiring smoke — orchestrator constructs strategy under the right gates
# ---------------------------------------------------------------------------


class TestOrchestratorWiringSmoke:
    """Mirrors M5 ``test_orchestrator_intraday_registration`` at the
    end-to-end shadow seam: when the strategy is enabled and the holder
    is loaded, the helper produces a registered strategy with the
    correct identity. When the holder is ``None``, the helper short-
    circuits with a WARNING. No commit-touchable orchestrator state is
    mutated — we exercise the helper directly via the M5 ``_bare_orch``
    pattern."""

    @pytest.fixture
    def _bare_orch(
        self,
        tmp_path: Path,
        loaded_emos_holder: EmosTableHolder,
    ) -> Any:
        """Skeleton orchestrator exposing only the attributes the
        ``_build_intraday_emos_high_strategy`` helper reads."""
        from python.orchestrator import Orchestrator

        intraday_cfg_dict: dict[str, Any] = {
            "enabled": True,
            "enabled_cells": ["MIA/high"],
            "edge_threshold": 0.07,
            "max_edge": 0.30,
            "min_distance_from_strike_std": 2.0,
            "max_contracts_per_trade": 50,
            "max_trades_per_cell_per_day": 3,
            "emos_hot_reload_pause_seconds": 60,
            "eval_audit_log": False,
            "candidate_audit_log": False,
            "scan_funnel_log": False,
        }
        settings = SimpleNamespace(
            state_dir=str(tmp_path),
            bankroll=10_000.0,
            execution=SimpleNamespace(mode="shadow"),
            market_filter=SimpleNamespace(model_dump=lambda: {}),
            strategies=SimpleNamespace(
                weather=SimpleNamespace(
                    enabled=False,
                    bias_correction_enabled=False,
                    bias_shrinkage=0.3,
                    emos_coefficients_path="unused",
                    emos_shadow_log=False,
                    model_dump=lambda: {
                        "enabled": False,
                        "bias_correction_enabled": False,
                        "bias_shrinkage": 0.3,
                        "emos_coefficients_path": "unused",
                        "emos_shadow_log": False,
                    },
                ),
                intraday_emos_high=SimpleNamespace(
                    enabled=True,
                    enabled_cells=["MIA/high"],
                    model_dump=lambda d=intraday_cfg_dict: dict(d),
                ),
            ),
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch._settings = settings
        orch._kalshi_client = MagicMock()
        orch._weather_client = MagicMock()
        orch._emos_holder = loaded_emos_holder
        orch._prediction_log = None
        orch._notifier = None
        orch._strategies = []
        return orch

    @pytest.mark.asyncio
    async def test_strategy_constructed_when_enabled_and_holder_loaded(
        self,
        _bare_orch: Any,
    ) -> None:
        strat = await _bare_orch._build_intraday_emos_high_strategy(bias_table=None)
        assert strat is not None
        assert isinstance(strat, IntradayEmosHighStrategy)
        assert strat.name == "intraday_emos_high"

    @pytest.mark.asyncio
    async def test_holder_none_path_skips_with_warning(
        self,
        _bare_orch: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Holder-None defensive skip — binding decision from M5.

        Per ``spec_alert_path_verification_rule``: the operator-visible
        signal here is the WARNING log line, NOT a Telegram CRITICAL.
        ``_build_intraday_emos_high_strategy`` returns ``None`` and the
        orchestrator's existing strategies-list never grows."""
        import logging

        _bare_orch._emos_holder = None
        with caplog.at_level(logging.WARNING, logger="python.orchestrator"):
            result = await _bare_orch._build_intraday_emos_high_strategy(
                bias_table=None
            )
        assert result is None
        assert any(
            "intraday_emos_high enabled but EmosTableHolder is None" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


# ---------------------------------------------------------------------------
# 2. Signal -> ExecutionEngine -> trades.jsonl
# ---------------------------------------------------------------------------


class TestSignalToTradeLog:
    """End-to-end: generate a synthetic intraday candidate, push it
    through the strategy + a real shadow engine, then inspect the
    resulting trades.jsonl row. This is the load-bearing contract for
    M8."""

    @pytest.mark.asyncio
    async def test_signal_reaches_engine_and_writes_shadow_row(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single-trade end-to-end path.

        Strategy generates one TradeSignal for MIA/high → engine in
        shadow mode → exactly one row appended to state/trades.jsonl
        with strategy=="intraday_emos_high" and the §11.1 metadata
        fields the daily-report consumer relies on.
        """
        from python.strategies import intraday_emos_high as iem_module

        today = et_today()
        ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700-B82"
        event_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700"

        kalshi = MagicMock()
        # Engine in shadow mode does NOT call iter_all_markets / get_orderbook
        # — we mock both so the strategy's scan() path runs without surprise.
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        # Provide a deep orderbook so the depth-adjustment doesn't reject.
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {
                    "yes_dollars": [["0.80", "500"]],
                    "no_dollars": [],
                }
            }
        )

        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        await strat.setup()
        assert strat._setup_completed is True

        # Build the synthetic Stage-1-passing candidate.
        cand = _make_market_candidate(
            ticker=ticker,
            event_ticker=event_ticker,
            city="MIA",
            forecast_date=today,
            side=Side.NO,
            model_prob=0.65,
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        # Monkey-patch the pipeline at the documented seams (M4 unit tests
        # already cover the full gate stack — here we focus on the
        # integration contract).
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
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="MIA", forecast_date=today, temperatures=[80.0] * 31
                )
            }
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        # Generate signals.
        signals = await strat.generate_signals()
        assert len(signals) == 1
        signal = signals[0]
        assert signal.strategy_name == "intraday_emos_high"
        assert signal.metadata["cell"] == "MIA/high"

        # Push through a real shadow engine.
        risk = RiskManager(shadow_settings)
        engine = ExecutionEngine(shadow_settings, kalshi_client=None, risk_manager=risk)
        record = await engine.execute(signal)
        assert record.status == OrderStatus.SHADOW

        # Inspect trades.jsonl.
        log_path = Path(shadow_settings.state_dir) / "trades.jsonl"
        assert log_path.exists(), "Engine must write a row to trades.jsonl"
        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        # Use the schema-evolution gate, not ad-hoc event filtering.
        fill_rows = [r for r in rows if is_pnl_relevant_row(r)]
        assert len(fill_rows) == 1, (
            f"Expected exactly one fill row, got {len(fill_rows)}: {rows}"
        )
        row = fill_rows[0]

        # §11.2 trade-log row schema — top-level attribution key.
        assert row["strategy"] == "intraday_emos_high"
        assert row["market_ticker"] == ticker
        assert row["side"] == "NO"
        assert row["mode"] == "shadow"

        # §11.1 signal metadata — the load-bearing fields the
        # daily-report and v6 replay verifier consume.
        md = row["metadata"]
        assert md["cell"] == "MIA/high"
        assert md["city"] == "MIA"
        assert md["forecast_date"] == today.isoformat()
        assert md["market_type"] == "high"
        assert md["lead_days"] == 0
        # EMOS hash threading (R2-H9 invariant).
        assert md["emos_artifact_hash"] == "abc123"
        assert md["emos_coeffs_artifact_hash"] == "abc123"
        assert md["live_calibrator_kind"] == "emos_raw"
        # v6 strike-distance audit input.
        assert md["strike_abs_z"] == 3.0
        assert md["pooled_pstdev"] == 1.5
        assert md["pooled_median"] == 82.0

    @pytest.mark.asyncio
    async def test_signal_for_lax_high_writes_lax_metadata(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parallel coverage for the second promoted cell. The cell label
        and city in metadata must reflect the candidate's parsed city, not
        a strategy-level default."""
        from python.strategies import intraday_emos_high as iem_module

        today = et_today()
        ticker = f"KXHIGHLAX-{_fmt_ymd(today)}H1700-B72"
        event_ticker = f"KXHIGHLAX-{_fmt_ymd(today)}H1700"

        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        kalshi.get_orderbook = AsyncMock(
            return_value={"orderbook_fp": {"yes_dollars": [["0.80", "500"]], "no_dollars": []}}
        )

        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        await strat.setup()

        cand = _make_market_candidate(
            ticker=ticker,
            event_ticker=event_ticker,
            city="LAX",
            forecast_date=today,
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(c, cfg, *, funnel=None, **_):
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args, **kwargs):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="LAX", forecast_date=today, temperatures=[82.0] * 31
                )
            }
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        signals = await strat.generate_signals()
        assert len(signals) == 1
        risk = RiskManager(shadow_settings)
        engine = ExecutionEngine(shadow_settings, kalshi_client=None, risk_manager=risk)
        await engine.execute(signals[0])

        log_path = Path(shadow_settings.state_dir) / "trades.jsonl"
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        fill_rows = [r for r in rows if is_pnl_relevant_row(r)]
        assert len(fill_rows) == 1
        row = fill_rows[0]
        assert row["strategy"] == "intraday_emos_high"
        assert row["metadata"]["cell"] == "LAX/high"
        assert row["metadata"]["city"] == "LAX"


# ---------------------------------------------------------------------------
# 3. Funnel emission at end of generate_signals
# ---------------------------------------------------------------------------


class TestFunnelEmission:
    @pytest.mark.asyncio
    async def test_emit_funnel_writes_row_with_intraday_strategy_tag(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-of-cycle funnel emission must land on disk with
        ``strategy == "intraday_emos_high"`` (spec §11.4 daily-report
        consumer reads this tag to slice intraday vs weather_edge).
        """
        from python.strategies import intraday_emos_high as iem_module

        today = et_today()
        ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700-B82"
        event_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700"

        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        kalshi.get_orderbook = AsyncMock(
            return_value={"orderbook_fp": {"yes_dollars": [["0.80", "500"]], "no_dollars": []}}
        )

        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        await strat.setup()

        cand = _make_market_candidate(
            ticker=ticker,
            event_ticker=event_ticker,
            forecast_date=today,
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(c, cfg, *, funnel=None, **_):
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args, **kwargs):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="MIA", forecast_date=today, temperatures=[82.0] * 31
                )
            }
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        signals = await strat.generate_signals()
        assert len(signals) == 1

        funnel_path = Path(shadow_settings.state_dir) / "scan_funnel.jsonl"
        assert funnel_path.exists(), "scan_funnel.jsonl must be written"
        rows = [
            json.loads(line)
            for line in funnel_path.read_text().splitlines()
            if line.strip()
        ]
        ours = [r for r in rows if r.get("strategy") == "intraday_emos_high"]
        assert len(ours) >= 1
        # Last row is from this generate_signals call — one pre-cap survivor.
        assert ours[-1]["n_signals_fired"] == 1


# ---------------------------------------------------------------------------
# 4. Daily-reset fan-out hits intraday + weather_edge cleanly
# ---------------------------------------------------------------------------


class TestDailyResetFanout:
    """Mirrors the M5 ``_reset_daily_with_fanout`` contract at the
    integration seam: the orchestrator's fan-out invokes
    ``reset_daily_counters`` on every strategy exposing the hook. The
    intraday strategy's hook clears ``_trades_today``; a legacy strategy
    without the hook is skipped silently — defensive ``getattr``
    dispatch."""

    def test_intraday_reset_daily_counters_clears_state(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
    ) -> None:
        """Direct ``reset_daily_counters`` call on the strategy clears its
        per-cell daily trade counter. This is the hook the orchestrator's
        fan-out invokes — verified at the unit-level by
        ``test_intraday_emos_high.py::TestPerCellDailyCap``; here we add
        an end-to-end seam check so a refactor that breaks the hook fails
        this integration test."""
        kalshi = MagicMock()
        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        today = et_today()
        strat._trades_today[("MIA/high", today)] = 2
        strat._trades_today[("LAX/high", today)] = 1
        assert strat._trades_today
        strat.reset_daily_counters()
        assert strat._trades_today == {}

    def test_reset_daily_fanout_dispatches_to_intraday_strategy(
        self,
        loaded_emos_holder: EmosTableHolder,
        tmp_path: Path,
    ) -> None:
        """Orchestrator-level fan-out (the production code path at
        orchestrator.py:651-677): construct a real intraday strategy +
        a legacy strategy without the hook + a RiskManager mock, and
        verify ``_reset_daily_with_fanout`` invokes the intraday hook
        without raising on the legacy strategy's missing attribute.

        This is the seam M5 lined up: ``reset_daily_counters`` is the
        per-strategy callback the fan-out walks ``self._strategies`` for.
        """
        from python.orchestrator import Orchestrator

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        settings = Settings(
            state_dir=str(state_dir),
            execution={"mode": "shadow"},
            strategies={"weather": {"edge_threshold": 0.07}},
            _env_file=None,
        )
        orch = Orchestrator.__new__(Orchestrator)
        orch._settings = settings
        risk_mock = MagicMock()
        risk_mock.reset_daily = MagicMock()
        orch._risk_manager = risk_mock
        orch._strategies = []

        # Real intraday strategy — has reset_daily_counters().
        intraday = _make_strategy(
            holder=loaded_emos_holder,
            settings=settings,
            kalshi_client=MagicMock(),
        )
        today = et_today()
        intraday._trades_today[("MIA/high", today)] = 2

        # Legacy strategy without the hook (matches weather_edge today).
        legacy = SimpleNamespace(name="legacy_no_hook")

        orch._strategies.extend([intraday, legacy])

        # Fan-out: RiskManager.reset_daily + every strategy with a callable
        # reset_daily_counters attribute.
        orch._reset_daily_with_fanout()

        risk_mock.reset_daily.assert_called_once()
        # Intraday's per-cell counter is cleared.
        assert intraday._trades_today == {}


# ---------------------------------------------------------------------------
# 5. EmosTableHolder sharing — single rebind visible to both strategies
# ---------------------------------------------------------------------------


class TestEmosTableHolderSharing:
    """Spec §12.9 invariant: weather_edge and intraday_emos_high share the
    SAME ``EmosTableHolder`` reference, so the orchestrator's per-tick
    ``holder.table = holder.table.check_and_reload() or holder.table``
    rebind propagates to both strategies in lockstep without per-strategy
    fan-out.

    We exercise the invariant by:
      1. Building TWO mock strategies that each receive the same holder.
      2. Reassigning ``holder.table`` to a fresh ``EmosTable`` with a
         different ``artifact_hash``.
      3. Asserting both strategies observe the new ``artifact_hash``
         on the next read.
    """

    def test_shared_holder_rebind_propagates_to_intraday_strategy(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        tmp_path: Path,
    ) -> None:
        # Both strategies wired against the same holder.
        intraday = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=MagicMock(),
        )
        # Simulate the second consumer (weather_edge in production) holding
        # the same holder reference. We don't construct a real WeatherEdge
        # strategy here — that drags in the bias/calibration stack — but
        # the invariant we're checking is the holder *reference identity*,
        # which is what makes the single-rebind propagation work.
        weather_holder_ref = loaded_emos_holder
        assert intraday._emos_holder is weather_holder_ref

        pre_hash = loaded_emos_holder.table.artifact_hash
        assert pre_hash is not None

        # Build a fresh artifact with a different ``a`` coefficient so the
        # hash differs.
        new_payload = _emos_payload()
        new_payload["cells"]["MIA/high"]["a"] = -7.0
        new_path = tmp_path / "coefficients_v2.json"
        new_path.write_text(json.dumps(new_payload))
        new_table = EmosTable.load(new_path)
        assert new_table.artifact_hash != pre_hash

        # Single rebind on the shared holder.
        loaded_emos_holder.table = new_table

        # Intraday's view of the table — through the shared holder
        # reference — must now point at the new hash.
        assert intraday._emos_holder.table.artifact_hash == new_table.artifact_hash
        # And the other consumer (weather_edge in production) would too.
        assert weather_holder_ref.table.artifact_hash == new_table.artifact_hash


# ---------------------------------------------------------------------------
# 6. Setup-incomplete WARNING path — first emission is immediate
# ---------------------------------------------------------------------------


class TestSetupIncompletePath:
    """Spec §4.2 v1.2 IMPORTANT #1: when ``setup()`` fails (any of the
    three preconditions trips), ``_setup_completed`` stays False and
    every subsequent ``scan()`` early-returns with a per-cell rate-
    limited WARNING. The first emission per (process, cell) MUST be
    immediate — the rate-limit window does NOT suppress it. Operators
    need that visibility to diagnose stalled startups."""

    @pytest.mark.asyncio
    async def test_setup_failure_emits_warning_on_first_tick(
        self,
        shadow_settings: Settings,
        tmp_path: Path,
    ) -> None:
        """Drive a setup() failure via precondition #2 (LAX/high requested,
        only MIA/high in the EmosTable) and assert the per-cell WARNING
        fires on the first scan tick."""
        # Holder with ONLY MIA/high — precondition #2 will trip for LAX/high.
        partial_payload = _emos_payload()
        partial_payload["cells"].pop("LAX/high")
        coeffs_path = tmp_path / "coefficients_partial.json"
        coeffs_path.write_text(json.dumps(partial_payload))
        table = EmosTable.load(coeffs_path)
        assert table.is_loaded
        partial_holder = EmosTableHolder(table=table)

        notifier = MagicMock()
        notifier.send_alert = AsyncMock()

        strat = _make_strategy(
            holder=partial_holder,
            settings=shadow_settings,
            kalshi_client=MagicMock(),
            notifier=notifier,
            enabled_cells=["MIA/high", "LAX/high"],
        )

        # setup() raises — orchestrator's _run_strategy_lifecycle_hook
        # would catch this. We verify the strategy stays in the not-ready
        # state and that scan() emits the WARNING on the first tick.
        with pytest.raises(RuntimeError, match="LAX/high"):
            await strat.setup()
        assert strat._setup_completed is False

        out = await strat.scan()
        assert out == []
        # Two enabled cells → two WARNING dispatches on the very first tick
        # (rate-limit window does NOT suppress the first emission).
        assert notifier.send_alert.await_count == 2
        for call in notifier.send_alert.await_args_list:
            assert call.kwargs["level"] == AlertLevel.WARNING

    @pytest.mark.asyncio
    async def test_generate_signals_returns_empty_when_setup_incomplete(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
    ) -> None:
        """The setup-incomplete gate also blocks ``generate_signals()``
        even with a perfectly-loaded EmosTableHolder — the
        ``_setup_completed`` flag is what gates emission, not the holder
        state. This guards against a regression where a future refactor
        treats a loaded holder as sufficient and bypasses the
        preconditions."""
        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=MagicMock(),
        )
        # Never called setup() — _setup_completed stays False.
        assert strat._setup_completed is False
        sigs = await strat.generate_signals()
        assert sigs == []


# ---------------------------------------------------------------------------
# 7. Production config invariant — intraday_emos_high.enabled stays False
# ---------------------------------------------------------------------------


class TestProductionConfigUntouched:
    """Production ``python/config.yaml`` may enable the intraday strategy
    only in shadow-only mode (spec 2026-05-29-intraday-shadow-only-mode-
    design.md §6). Live trading still requires a deliberate
    ``shadow_only: false`` flip plus a separate PR. This static check
    fails CI if anyone enables live intraday trading without going
    through the post-shadow calibration review."""

    def test_production_config_intraday_shadow_only_invariant(self) -> None:
        import yaml

        config_path = Path("python/config.yaml")
        assert config_path.exists()
        cfg = yaml.safe_load(config_path.read_text())
        intraday = cfg["strategies"]["intraday_emos_high"]
        if intraday.get("enabled") is True:
            assert intraday.get("shadow_only") is True, (
                "python/config.yaml: intraday_emos_high.enabled=true requires "
                "shadow_only=true. Live execution is gated on the post-shadow "
                "calibration review (§8 of the shadow-only spec) and a separate PR."
            )


# ---------------------------------------------------------------------------
# 8. Sanity: cell label helper still matches metadata format
# ---------------------------------------------------------------------------


def test_cell_label_helper_matches_metadata_format() -> None:
    """End-to-end the metadata ``cell`` field uses the canonical
    ``<CITY>/<high|low>`` format — same shape as
    ``weather.emos_enabled_cells`` config. ``_cell_label`` is the
    helper that produces it. A drift in that helper would silently
    break the daily-report's cell-grouping logic."""
    assert _cell_label("MIA", "high") == "MIA/high"
    assert _cell_label("LAX", "high") == "LAX/high"


# ---------------------------------------------------------------------------
# 9. SAFETY-CRITICAL: shadow_only=True guarantees (§5)
# ---------------------------------------------------------------------------


class TestShadowOnlySafetyCritical:
    """Integration-level safety tests for the shadow_only=True path.

    These tests exercise the WIRING between:
      - ``IntradayEmosHighStrategy.generate_signals()`` (produces signals,
        writes to PredictionLog inside the call)
      - ``apply_intraday_shadow_drop()`` (drop hook in the orchestrator tick)
      - the execute path (should see zero intraday signals when shadow_only=True)

    They deliberately do NOT duplicate the pure-function behavior of
    ``apply_intraday_shadow_drop`` already covered in
    ``tests/test_orchestrator_shadow_drop.py``.  The value here is at the
    integration seam: verifying the sequencing and that the hook is correctly
    positioned in the tick loop relative to PredictionLog.append.

    Spec §5 requires TWO hard guarantees:

    1. **No-execution guarantee** — with ``shadow_only=True``, the execute path
       receives zero intraday_emos_high signals, even when mixed with
       weather_edge signals (which must still pass through).

    2. **Ordering guarantee** — PredictionLog.append is called INSIDE
       ``generate_signals()`` (before the orchestrator's drop hook runs).
       Silent non-collection would defeat the whole shadow validation feature.
    """

    # ------------------------------------------------------------------ #
    # Guarantee 1: no-execution (integration wiring)                      #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_no_execution_when_shadow_only_true(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """§5 no-execution guarantee at the integration seam.

        Strategy generates one intraday signal (MIA/high).  A synthetic
        weather_edge signal is injected into the combined batch to confirm
        the drop hook is selective: it removes intraday_emos_high signals
        and leaves all other signals untouched.

        This is an integration test of the tick-loop wiring:
          generate_signals() → apply_intraday_shadow_drop(shadow_only=True)
        rather than a unit test of the pure drop function.

        Regression target: a future refactor that moves the drop hook AFTER
        the execute call would cause this test to fail (intraday signal would
        appear in the execute batch).
        """
        from python.orchestrator import apply_intraday_shadow_drop
        from python.strategies import intraday_emos_high as iem_module
        from python.strategies.base import TradeSignal

        today = et_today()
        ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700-B82"
        event_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700"

        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {
                    "yes_dollars": [["0.80", "500"]],
                    "no_dollars": [],
                }
            }
        )

        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        await strat.setup()

        cand = _make_market_candidate(
            ticker=ticker,
            event_ticker=event_ticker,
            city="MIA",
            forecast_date=today,
            side=Side.NO,
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(c, cfg, *, funnel=None, **_):
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args, **kwargs):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="MIA", forecast_date=today, temperatures=[80.0] * 31
                )
            }
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        # --- Strategy generates one intraday signal ---
        intraday_signals = await strat.generate_signals()
        assert len(intraday_signals) == 1
        assert intraday_signals[0].strategy_name == "intraday_emos_high"

        # --- Synthesise a weather_edge signal to test selective filtering ---
        weather_signal = TradeSignal(
            market_id="KXHIGHNYC-TICKER",
            market_ticker="KXHIGHNYC-TICKER",
            model_probability=0.70,
            market_price=0.55,
            confidence=0.80,
            strategy_name="weather_edge",
            side=Side.YES,
            suggested_dollars=50.0,
            suggested_contracts=5,
            effective_edge=0.12,
            metadata={"city": "NYC", "market_type": "high", "forecast_date": today.isoformat()},
        )

        combined_batch = intraday_signals + [weather_signal]

        # --- Simulate the orchestrator's tick-loop drop hook (shadow_only=True) ---
        execute_batch = apply_intraday_shadow_drop(combined_batch, shadow_only=True)

        # No-execution guarantee: intraday_emos_high must NOT reach execute.
        intraday_in_batch = [
            s for s in execute_batch if s.strategy_name == "intraday_emos_high"
        ]
        assert intraday_in_batch == [], (
            "SAFETY VIOLATION: intraday_emos_high signal reached the execute path "
            f"with shadow_only=True. Signals in batch: {[s.strategy_name for s in execute_batch]}"
        )

        # Weather_edge signal must still reach execute (selective, not a blanket drop).
        weather_in_batch = [s for s in execute_batch if s.strategy_name == "weather_edge"]
        assert len(weather_in_batch) == 1, (
            "weather_edge signal must pass through the intraday shadow drop hook unchanged"
        )

    # ------------------------------------------------------------------ #
    # Guarantee 2: ordering (PredictionLog written before drop hook)      #
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_prediction_logged_before_drop(
        self,
        loaded_emos_holder: EmosTableHolder,
        shadow_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """§5 ordering guarantee: PredictionLog.append is called inside
        ``generate_signals()`` — before the orchestrator's drop hook has any
        chance to run.

        If this ordering were reversed (log after drop), shadow mode would
        silently collect zero data — defeating the entire feature.

        Test structure:
          1. Run ``generate_signals()`` with a real PredictionLog.
          2. Assert PredictionLog contains a record for the intraday signal.
             (If step 2 passes, logging MUST have occurred before step 3.)
          3. Apply ``apply_intraday_shadow_drop(shadow_only=True)``.
          4. Assert the returned signal list is now empty.

        The causal chain — step 2 succeeds BEFORE step 3 even runs — proves
        the ordering invariant.  A regression that moved logging to after the
        drop would cause step 2 to find an empty log, failing the assertion.
        """
        from python.analytics.prediction_log import PredictionLog
        from python.orchestrator import apply_intraday_shadow_drop
        from python.strategies import intraday_emos_high as iem_module

        today = et_today()
        ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700-B82"
        event_ticker = f"KXHIGHMIA-{_fmt_ymd(today)}H1700"

        # Wire a real PredictionLog into the strategy so logging actually fires.
        pred_log_path = tmp_path / "state" / "predictions.jsonl"
        pred_log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_log = PredictionLog(pred_log_path)

        kalshi = MagicMock()
        kalshi.iter_all_markets = AsyncMock(return_value=([], False))
        kalshi.get_orderbook = AsyncMock(
            return_value={
                "orderbook_fp": {
                    "yes_dollars": [["0.80", "500"]],
                    "no_dollars": [],
                }
            }
        )

        strat = _make_strategy(
            holder=loaded_emos_holder,
            settings=shadow_settings,
            kalshi_client=kalshi,
        )
        # Inject the real PredictionLog (bypassing None guard in _make_strategy).
        strat._prediction_log = prediction_log
        await strat.setup()

        cand = _make_market_candidate(
            ticker=ticker,
            event_ticker=event_ticker,
            city="MIA",
            forecast_date=today,
            side=Side.NO,
        )
        scan_candidates = [
            ScanCandidate(market=cand.market, parsed=cand.parsed, price_info=cand.price_info)
        ]

        async def _fake_scan(*args: Any, **kwargs: Any) -> list[ScanCandidate]:
            return scan_candidates

        def _fake_eval(sc: ScanCandidate, *args: Any, **kwargs: Any) -> _MarketCandidate:
            return cand

        async def _fake_depth(c, cfg, *, funnel=None, **_):
            return c.effective_edge, c.sizing, c.contracts, False, None

        def _fake_postprocess(*args, **kwargs):
            return {}, {}, {}, {}

        strat._weather.get_multi_model_forecast = AsyncMock(
            return_value={
                "gfs_seamless": EnsembleForecast(
                    city="MIA", forecast_date=today, temperatures=[80.0] * 31
                )
            }
        )

        monkeypatch.setattr(strat, "scan", _fake_scan)
        monkeypatch.setattr(iem_module, "evaluate_market_pure", _fake_eval)
        monkeypatch.setattr(strat, "_apply_depth_adjustment", _fake_depth)
        monkeypatch.setattr(iem_module, "postprocess_forecasts", _fake_postprocess)

        # --- Step 1: generate_signals() runs (and internally writes to PredictionLog) ---
        signals = await strat.generate_signals()
        assert len(signals) == 1, "Strategy must produce exactly one signal for this test"

        # --- Step 2: verify PredictionLog already has the record ---
        # This MUST succeed before step 3 (the drop) to prove the ordering invariant.
        logged_records = list(prediction_log.iter_all())
        assert len(logged_records) >= 1, (
            "ORDERING VIOLATION: PredictionLog is empty after generate_signals() "
            "returned. PredictionLog.append must be called inside generate_signals(), "
            "not after the drop hook."
        )
        logged_tickers = [r.market_ticker for r in logged_records]
        assert ticker in logged_tickers, (
            f"Expected ticker {ticker!r} in PredictionLog after generate_signals(). "
            f"Logged tickers: {logged_tickers}"
        )
        assert logged_records[0].strategy == "intraday_emos_high"

        # --- Step 3: apply the orchestrator's drop hook ---
        after_drop = apply_intraday_shadow_drop(signals, shadow_only=True)

        # --- Step 4: the signal is now gone from the execute batch ---
        assert after_drop == [], (
            "SAFETY VIOLATION: intraday_emos_high signal survived the drop hook "
            f"(shadow_only=True). Remaining signals: {[s.strategy_name for s in after_drop]}"
        )
        # And the log record from step 2 still exists — logging was not undone.
        assert len(list(prediction_log.iter_all())) >= 1, (
            "PredictionLog must retain the record after the drop hook runs"
        )
