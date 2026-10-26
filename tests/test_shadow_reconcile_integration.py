"""Integration: shadow-mode exposure survives a simulated restart.

This is the regression test for the bug that motivated ShadowBroker — a
restart in shadow mode zeroed ``RiskManager._positions`` and let the bot
re-trade markets whose exposure had been blocked for hours.  We simulate
the restart cycle by tearing down the engine/risk pair and rebuilding it
with the same state directory, then running the reconcile path that the
orchestrator now uses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from python.common import OrderStatus, Side
from python.execution.engine import ExecutionEngine
from python.execution.shadow_broker import ShadowBroker
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal


def _make_settings(tmp_path: Path) -> Settings:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state_dir),
        execution={"mode": "shadow"},
        _env_file=None,
    )


_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLAX", "KXHIGHAUS"]


def _make_signal(i: int) -> TradeSignal:
    """Build a signal for a distinct city so the correlated-positions cap
    doesn't reject before the test hits the exposure condition."""
    future = (datetime.now(UTC) + timedelta(days=7)).date()
    ymd = future.strftime("%y%b%d").upper()
    series = _SERIES[i % len(_SERIES)]
    ticker = f"{series}-{ymd}-B{70 + i}.5"
    close_time = (datetime.now(UTC) + timedelta(days=6)).isoformat()
    return TradeSignal(
        market_id=ticker,
        market_ticker=ticker,
        model_probability=0.65,
        market_price=0.50,
        confidence=0.8,
        strategy_name="weather_edge",
        side=Side.YES,
        suggested_dollars=200.0,
        suggested_contracts=400,
        metadata={"close_time": close_time},
    )


def _reconcile_shadow(settings: Settings, engine: ExecutionEngine, risk: RiskManager) -> None:
    """Mimic Orchestrator._reconcile_shadow without needing the full class."""
    broker = ShadowBroker(
        trade_log_path=Path(settings.state_dir) / "trades.jsonl",
        configured_bankroll=settings.bankroll,
    )
    positions = broker.replay()
    risk.hydrate_positions({p.position_key: p.dollars for p in positions}, broker.bankroll)
    engine.hydrate_shadow_positions(positions)


# ---------------------------------------------------------------------------
# Positive: with ShadowBroker, exposure survives restart and cap stays binding
# ---------------------------------------------------------------------------


class TestExposurePersistsAcrossRestart:
    @pytest.mark.asyncio
    async def test_total_exposure_preserved(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)

        # ── Run 1 ──────────────────────────────────────────────────────────
        risk1 = RiskManager(settings)
        engine1 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk1)

        for i in range(5):
            rec = await engine1.execute(_make_signal(i))
            assert rec.status == OrderStatus.SHADOW, f"signal {i} was rejected: {rec.status}"
        pre_restart_exposure = risk1.total_exposure
        assert pre_restart_exposure > 0

        # ── Simulated restart: drop in-memory state, rebuild against same log ─
        risk2 = RiskManager(settings)
        engine2 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk2)
        # Before reconcile: exposure is zero — the bug.
        assert risk2.total_exposure == 0
        _reconcile_shadow(settings, engine2, risk2)
        assert risk2.total_exposure == pytest.approx(pre_restart_exposure)

    @pytest.mark.asyncio
    async def test_total_approved_never_exceeds_cap_across_restarts(
        self, tmp_path: Path
    ) -> None:
        """The core invariant: across any sequence of restarts, the sum of
        approved exposure never breaches ``TOTAL_EXPOSURE_CAP``.  Without
        ShadowBroker the cap resets to 0 on each restart and the invariant
        is trivially violated."""
        settings = _make_settings(tmp_path)

        risk1 = RiskManager(settings)
        engine1 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk1)
        # Saturate near the market cap on 5 different cities (~$250 each).
        signals = [_make_signal(i) for i in range(5)]
        for s in signals:
            await engine1.execute(s)
        run1_exposure = risk1.total_exposure
        cap = risk1.effective_max_total_exposure_pct * settings.bankroll

        # Simulated restart
        risk2 = RiskManager(settings)
        engine2 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk2)
        _reconcile_shadow(settings, engine2, risk2)

        run2_approved = 0.0
        for s in signals:
            rec = await engine2.execute(s)
            if rec.status == OrderStatus.SHADOW:
                run2_approved += rec.risk_decision.adjusted_dollars

        assert run1_exposure + run2_approved <= cap + 1e-6, (
            f"cap breached: run1=${run1_exposure:.2f} + run2=${run2_approved:.2f} "
            f"> cap=${cap:.2f}"
        )


# ---------------------------------------------------------------------------
# Reverse assertion: the test actually exercises the bug
# ---------------------------------------------------------------------------


class TestRegressionGuard:
    """If ShadowBroker hydration is skipped, exposure does NOT survive —
    which was the original bug.  Keeping this test proves future refactors
    can't silently regress the fix."""

    @pytest.mark.asyncio
    async def test_without_reconcile_exposure_is_lost(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)

        risk1 = RiskManager(settings)
        engine1 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk1)
        for i in range(3):
            await engine1.execute(_make_signal(i))
        assert risk1.total_exposure > 0

        # Simulated restart WITHOUT reconcile — this is what the code did
        # before ShadowBroker.
        risk2 = RiskManager(settings)
        _engine2 = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk2)
        assert risk2.total_exposure == 0  # the bug in isolation
