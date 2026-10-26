"""Silent-failure alerting for order placement.

A 2-day outage (2026-06-19, Kalshi V1 create-order deprecation) produced 5,296
failed placements with no alert: the process stayed alive (liveness monitors
happy) and there were no fills, hence no losses (P&L circuit breakers happy).
These detectors watch the intent->execution gap directly:

  #1 fast/specific: a non-transient (4xx) placement rejection fires immediately.
  #2 durable net:   N consecutive live ticks where orders were approved but none
                    placed fires regardless of the underlying cause.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from python.common import ExecutionMode, OrderStatus, Side
from python.data.kalshi_client import KalshiAPIError
from python.execution.engine import (
    _PLACEMENT_DIVERGENCE_TICKS,
    ExecutionEngine,
    OrderRecord,
    _batch_attempt_counts,
    _is_non_transient_placement_error,
)
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal


def _settings(tmp_path, mode: str = "live") -> Settings:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return Settings(state_dir=str(state), execution={"mode": mode}, _env_file=None)


def _signal(**kw: Any) -> TradeSignal:
    d: dict[str, Any] = {
        "market_id": "MKT-001",
        "market_ticker": "MKT-001-T",
        "model_probability": 0.65,
        "market_price": 0.50,
        "confidence": 0.8,
        "strategy_name": "favorite_bias",
        "side": Side.YES,
        "suggested_dollars": 100.0,
        "suggested_contracts": 2,
        "metadata": {},
    }
    d.update(kw)
    return TradeSignal(**d)


def _rec(risk: RiskManager, status: OrderStatus) -> OrderRecord:
    sig = _signal()
    return OrderRecord(
        order_id="o",
        signal=sig,
        risk_decision=risk.check_signal(sig),
        mode=ExecutionMode.LIVE,
        signal_price=0.5,
        order_price=0.5,
        status=status,
    )


def _engine(tmp_path, notify=None):
    s = _settings(tmp_path)
    risk = RiskManager(s)
    eng = ExecutionEngine(
        s, kalshi_client=AsyncMock(), risk_manager=risk, notify_critical=notify
    )
    return eng, risk


# ---------------------------------------------------------------------------
# Pure: classify a placement exception
# ---------------------------------------------------------------------------


class TestNonTransientClassifier:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
    def test_4xx_is_non_transient(self, status: int) -> None:
        assert _is_non_transient_placement_error(KalshiAPIError(status, {})) is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_429_and_5xx_are_transient(self, status: int) -> None:
        # 429/5xx are already retried by KalshiClient._request; not worth alerting.
        assert _is_non_transient_placement_error(KalshiAPIError(status, {})) is False

    def test_non_api_exception_is_not_flagged(self) -> None:
        assert _is_non_transient_placement_error(ValueError("boom")) is False


# ---------------------------------------------------------------------------
# Pure: count attempted vs successfully-placed in a batch
# ---------------------------------------------------------------------------


class TestBatchAttemptCounts:
    def test_all_errored(self, tmp_path) -> None:
        risk = RiskManager(_settings(tmp_path))
        recs = [_rec(risk, OrderStatus.ERROR), _rec(risk, OrderStatus.ERROR)]
        assert _batch_attempt_counts(recs) == (2, 0)

    def test_mixed_success_and_error(self, tmp_path) -> None:
        risk = RiskManager(_settings(tmp_path))
        recs = [_rec(risk, OrderStatus.PENDING), _rec(risk, OrderStatus.ERROR)]
        assert _batch_attempt_counts(recs) == (2, 1)

    def test_rejected_and_skipped_are_not_attempts(self, tmp_path) -> None:
        # Risk-rejected / dedup-skipped means we never intended to place: no divergence.
        risk = RiskManager(_settings(tmp_path))
        recs = [_rec(risk, OrderStatus.REJECTED), _rec(risk, OrderStatus.SKIPPED)]
        assert _batch_attempt_counts(recs) == (0, 0)

    def test_filled_counts_as_placed_ok(self, tmp_path) -> None:
        risk = RiskManager(_settings(tmp_path))
        assert _batch_attempt_counts([_rec(risk, OrderStatus.FILLED)]) == (1, 1)


# ---------------------------------------------------------------------------
# Behavioral: divergence watchdog (#2)
# ---------------------------------------------------------------------------


class TestDivergenceWatchdog:
    @pytest.mark.asyncio
    async def test_fires_only_after_threshold_consecutive_failed_ticks(self, tmp_path) -> None:
        notify = AsyncMock()
        eng, risk = _engine(tmp_path, notify)
        all_error = [_rec(risk, OrderStatus.ERROR)]
        for _ in range(_PLACEMENT_DIVERGENCE_TICKS - 1):
            await eng._check_placement_divergence(all_error)
        assert notify.await_count == 0  # not yet
        await eng._check_placement_divergence(all_error)
        assert notify.await_count == 1
        msg = notify.await_args.args[0].lower()
        assert "approved" in msg or "placed" in msg

    @pytest.mark.asyncio
    async def test_a_successful_placement_resets_the_counter(self, tmp_path) -> None:
        notify = AsyncMock()
        eng, risk = _engine(tmp_path, notify)
        all_error = [_rec(risk, OrderStatus.ERROR)]
        success = [_rec(risk, OrderStatus.PENDING)]
        for _ in range(_PLACEMENT_DIVERGENCE_TICKS - 1):
            await eng._check_placement_divergence(all_error)
        await eng._check_placement_divergence(success)  # reset
        await eng._check_placement_divergence(all_error)
        assert notify.await_count == 0  # counter restarted, threshold not reached

    @pytest.mark.asyncio
    async def test_empty_or_no_intent_ticks_do_not_count(self, tmp_path) -> None:
        notify = AsyncMock()
        eng, risk = _engine(tmp_path, notify)
        no_intent = [_rec(risk, OrderStatus.REJECTED)]
        for _ in range(_PLACEMENT_DIVERGENCE_TICKS + 2):
            await eng._check_placement_divergence(no_intent)
            await eng._check_placement_divergence([])
        assert notify.await_count == 0


# ---------------------------------------------------------------------------
# Behavioral: cooldown shared by both detectors
# ---------------------------------------------------------------------------


class TestAlertCooldown:
    @pytest.mark.asyncio
    async def test_repeated_failures_within_cooldown_fire_once(self, tmp_path) -> None:
        notify = AsyncMock()
        eng, risk = _engine(tmp_path, notify)
        all_error = [_rec(risk, OrderStatus.ERROR)]
        for _ in range(_PLACEMENT_DIVERGENCE_TICKS + 10):
            await eng._check_placement_divergence(all_error)
        assert notify.await_count == 1

    @pytest.mark.asyncio
    async def test_no_notifier_is_safe(self, tmp_path) -> None:
        eng, risk = _engine(tmp_path, notify=None)
        all_error = [_rec(risk, OrderStatus.ERROR)]
        for _ in range(_PLACEMENT_DIVERGENCE_TICKS + 1):
            await eng._check_placement_divergence(all_error)  # must not raise
