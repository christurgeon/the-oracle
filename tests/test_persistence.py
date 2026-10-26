"""Tests for risk state persistence (python/risk/persistence.py) and
RiskManager hydration methods (python/risk/manager.py)."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from python.common import ET
from python.notifications.base import AlertLevel
from python.risk.manager import (
    PERSISTENCE_BROKEN_ALERT_PREFIX,
    SAVE_FAILURE_ALERT_THRESHOLD,
    RiskManager,
)
from python.risk.persistence import (
    EXPOSURE_OVERRIDE_FILENAME,
    EXPOSURE_OVERRIDE_SCHEMA_VERSION,
    HEARTBEAT_FILENAME,
    STATE_FILENAME,
    STATE_SCHEMA_VERSION,
    RiskStateSnapshot,
    deserialize_risk_state,
    load_exposure_override,
    load_state,
    save_exposure_override,
    save_state,
    serialize_risk_state,
    write_heartbeat,
)
from python.settings import Settings

# ── helpers ────────────────────────────────────────────────────────────────


def _make_tracker_export(
    *,
    enabled: bool = True,
    disabled_reason: str = "",
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a strategy tracker dict as returned by RiskManager.export_state."""
    return {
        "enabled": enabled,
        "disabled_reason": disabled_reason,
        "trades": trades or [],
    }


def _sample_state_data(**overrides: Any) -> dict[str, Any]:
    """Build a valid serialized state dict."""
    defaults: dict[str, Any] = {
        "peak_bankroll": 10000.0,
        "daily_pnl": -25.0,
        "daily_start_bankroll": 10000.0,
        "trading_halted": False,
        "strategy_trackers": {
            "weather_edge": _make_tracker_export(
                trades=[
                    {"pnl": 5.0, "won": True, "ts": datetime(2026, 3, 20, 12, tzinfo=UTC)},
                    {"pnl": -2.0, "won": False, "ts": datetime(2026, 3, 21, 14, tzinfo=UTC)},
                ]
            ),
        },
        "current_trading_date": "2026-03-22",
        "execution_mode": "shadow",
    }
    defaults.update(overrides)
    return defaults


# ══════════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════════


class TestSerializeDeserializeRoundTrip:
    def test_basic_round_trip(self):
        data = _sample_state_data()
        serialized = serialize_risk_state(**data)
        snapshot = deserialize_risk_state(serialized)

        assert snapshot is not None
        assert snapshot.peak_bankroll == 10000.0
        assert snapshot.daily_pnl == -25.0
        assert snapshot.daily_start_bankroll == 10000.0
        assert snapshot.trading_halted is False
        assert snapshot.current_trading_date == "2026-03-22"
        assert snapshot.execution_mode == "shadow"
        assert snapshot.schema_version == STATE_SCHEMA_VERSION

    def test_trade_timestamps_round_trip(self):
        data = _sample_state_data()
        serialized = serialize_risk_state(**data)
        snapshot = deserialize_risk_state(serialized)

        assert snapshot is not None
        trades = snapshot.strategy_trackers["weather_edge"]["trades"]
        assert len(trades) == 2
        assert isinstance(trades[0]["ts"], datetime)
        assert trades[0]["ts"].year == 2026

    def test_empty_trackers_round_trip(self):
        data = _sample_state_data(strategy_trackers={})
        serialized = serialize_risk_state(**data)
        snapshot = deserialize_risk_state(serialized)

        assert snapshot is not None
        assert snapshot.strategy_trackers == {}

    def test_multiple_strategies(self):
        data = _sample_state_data(
            strategy_trackers={
                "weather_edge": _make_tracker_export(
                    trades=[{"pnl": 5.0, "won": True, "ts": datetime(2026, 3, 20, tzinfo=UTC)}]
                ),
                "ai_news": _make_tracker_export(
                    enabled=False,
                    disabled_reason="Rolling 7d P&L too low",
                    trades=[{"pnl": -3.0, "won": False, "ts": datetime(2026, 3, 21, tzinfo=UTC)}],
                ),
            }
        )
        serialized = serialize_risk_state(**data)
        snapshot = deserialize_risk_state(serialized)

        assert snapshot is not None
        assert len(snapshot.strategy_trackers) == 2
        assert snapshot.strategy_trackers["ai_news"]["enabled"] is False
        assert "too low" in snapshot.strategy_trackers["ai_news"]["disabled_reason"]

    def test_serialized_output_is_json_safe(self):
        data = _sample_state_data()
        serialized = serialize_risk_state(**data)
        # Should not raise
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == STATE_SCHEMA_VERSION


class TestDeserializeEdgeCases:
    def test_wrong_schema_version_returns_none(self):
        data = {"schema_version": 999, "peak_bankroll": 1000.0}
        assert deserialize_risk_state(data) is None

    def test_missing_schema_version_returns_none(self):
        data = {"peak_bankroll": 1000.0}
        assert deserialize_risk_state(data) is None

    def test_corrupt_data_returns_none(self):
        assert deserialize_risk_state({"schema_version": 1}) is None

    def test_halted_state_preserved(self):
        data = _sample_state_data(trading_halted=True)
        serialized = serialize_risk_state(**data)
        snapshot = deserialize_risk_state(serialized)

        assert snapshot is not None
        assert snapshot.trading_halted is True


# ══════════════════════════════════════════════════════════════════════════
# File I/O
# ══════════════════════════════════════════════════════════════════════════


class TestSaveAndLoad:
    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(self, tmp_path: Path):
        data = serialize_risk_state(**_sample_state_data())
        await save_state(tmp_path, data)

        snapshot = load_state(tmp_path)
        assert snapshot is not None
        assert snapshot.peak_bankroll == 10000.0

    @pytest.mark.asyncio
    async def test_no_temp_file_remains(self, tmp_path: Path):
        data = serialize_risk_state(**_sample_state_data())
        await save_state(tmp_path, data)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_load_missing_file_returns_none(self, tmp_path: Path):
        assert load_state(tmp_path) is None

    def test_load_corrupt_json_returns_none(self, tmp_path: Path):
        (tmp_path / STATE_FILENAME).write_text("not valid json{{{")
        assert load_state(tmp_path) is None

    def test_load_wrong_schema_version_returns_none(self, tmp_path: Path):
        bad = {"schema_version": 999}
        (tmp_path / STATE_FILENAME).write_text(json.dumps(bad))
        assert load_state(tmp_path) is None

    @pytest.mark.asyncio
    async def test_save_creates_parent_directory(self, tmp_path: Path):
        nested = tmp_path / "sub" / "dir"
        data = serialize_risk_state(**_sample_state_data())
        await save_state(nested, data)

        assert (nested / STATE_FILENAME).exists()


# ══════════════════════════════════════════════════════════════════════════
# Heartbeat
# ══════════════════════════════════════════════════════════════════════════


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_creates_heartbeat_file(self, tmp_path: Path):
        await write_heartbeat(tmp_path)
        hb = tmp_path / HEARTBEAT_FILENAME
        assert hb.exists()
        ts = float(hb.read_text())
        assert time.time() - ts < 5.0

    @pytest.mark.asyncio
    async def test_updates_timestamp(self, tmp_path: Path):
        await write_heartbeat(tmp_path)
        ts1 = float((tmp_path / HEARTBEAT_FILENAME).read_text())

        await write_heartbeat(tmp_path)
        ts2 = float((tmp_path / HEARTBEAT_FILENAME).read_text())

        assert ts2 >= ts1


# ══════════════════════════════════════════════════════════════════════════
# RiskManager Hydration
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fresh_rm(tmp_path: Path) -> RiskManager:
    """RiskManager with default 10k bankroll."""
    s = Settings(state_dir=str(tmp_path / "state"), bankroll=10000.0, _env_file=None)
    return RiskManager(s)


class TestHydratePositions:
    def test_sets_positions_and_bankroll(self, fresh_rm: RiskManager):
        positions = {"MKT-A": 100.0, "MKT-B": 200.0}
        fresh_rm.hydrate_positions(positions, bankroll=9500.0)

        assert fresh_rm.bankroll == 9500.0
        assert fresh_rm.positions == {"MKT-A": 100.0, "MKT-B": 200.0}
        assert fresh_rm.total_exposure == 300.0

    def test_does_not_modify_daily_pnl(self, fresh_rm: RiskManager):
        fresh_rm.hydrate_positions({"MKT-A": 50.0}, bankroll=9000.0)
        assert fresh_rm.daily_pnl == 0.0

    def test_does_not_set_dirty_flag(self, fresh_rm: RiskManager):
        fresh_rm.hydrate_positions({"MKT-A": 50.0}, bankroll=9000.0)
        assert fresh_rm.state_dirty is False

    def test_empty_positions(self, fresh_rm: RiskManager):
        fresh_rm.hydrate_positions({}, bankroll=10000.0)
        assert fresh_rm.positions == {}
        assert fresh_rm.total_exposure == 0.0


class TestHydrateFromState:
    def _make_snapshot(self, **overrides: Any) -> RiskStateSnapshot:
        defaults: dict[str, Any] = {
            "peak_bankroll": 11000.0,
            "daily_pnl": -50.0,
            "daily_start_bankroll": 10500.0,
            "trading_halted": False,
            "current_trading_date": "2026-03-22",
            "execution_mode": "shadow",
            "strategy_trackers": {},
        }
        defaults.update(overrides)
        return RiskStateSnapshot(**defaults)

    def test_restores_peak_bankroll(self, fresh_rm: RiskManager):
        snapshot = self._make_snapshot(peak_bankroll=12000.0)
        fresh_rm.hydrate_from_state(snapshot)
        # max(12000, 10000) = 12000
        assert fresh_rm.peak_bankroll == 12000.0

    def test_peak_uses_max_of_stored_and_current(self, fresh_rm: RiskManager):
        # Simulate case where Kalshi balance grew while bot was down
        fresh_rm.hydrate_positions({}, bankroll=15000.0)
        snapshot = self._make_snapshot(peak_bankroll=12000.0)
        fresh_rm.hydrate_from_state(snapshot)
        # max(12000, 15000) = 15000
        assert fresh_rm.peak_bankroll == 15000.0

    def test_restores_daily_pnl(self, fresh_rm: RiskManager):
        snapshot = self._make_snapshot(daily_pnl=-75.0)
        fresh_rm.hydrate_from_state(snapshot)
        assert fresh_rm.daily_pnl == -75.0

    def test_restores_halt_flag(self, fresh_rm: RiskManager):
        snapshot = self._make_snapshot(trading_halted=True)
        fresh_rm.hydrate_from_state(snapshot)
        assert fresh_rm.is_halted is True

    def test_restores_strategy_trackers(self, fresh_rm: RiskManager):
        snapshot = self._make_snapshot(
            strategy_trackers={
                "weather_edge": {
                    "enabled": False,
                    "disabled_reason": "Rolling P&L too low",
                    "trades": [
                        {"pnl": -5.0, "won": False, "ts": datetime(2026, 3, 21, tzinfo=UTC)},
                    ],
                }
            }
        )
        fresh_rm.hydrate_from_state(snapshot)
        tracker = fresh_rm.get_tracker("weather_edge")
        assert tracker.enabled is False
        assert "too low" in tracker.disabled_reason
        assert len(tracker.trades) == 1

    def test_does_not_set_dirty_flag(self, fresh_rm: RiskManager):
        snapshot = self._make_snapshot()
        fresh_rm.hydrate_from_state(snapshot)
        assert fresh_rm.state_dirty is False


# ══════════════════════════════════════════════════════════════════════════
# Dirty Flag
# ══════════════════════════════════════════════════════════════════════════


class TestDirtyFlag:
    def test_clean_initially(self, fresh_rm: RiskManager):
        assert fresh_rm.state_dirty is False

    def test_record_fill_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.record_fill("MKT-A", "weather_edge", 100.0)
        assert fresh_rm.state_dirty is True

    def test_release_exposure_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.record_fill("MKT-A", "weather_edge", 100.0)
        fresh_rm.mark_state_clean()
        fresh_rm.release_exposure("MKT-A", 50.0)
        assert fresh_rm.state_dirty is True

    def test_close_position_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.record_fill("MKT-A", "weather_edge", 100.0)
        fresh_rm.mark_state_clean()
        fresh_rm.close_position("MKT-A", pnl=5.0)
        assert fresh_rm.state_dirty is True

    def test_reset_daily_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.reset_daily()
        assert fresh_rm.state_dirty is True

    def test_record_strategy_trade_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.record_strategy_trade("weather_edge", pnl=5.0, won=True)
        assert fresh_rm.state_dirty is True

    def test_enable_strategy_sets_dirty(self, fresh_rm: RiskManager):
        fresh_rm.enable_strategy("weather_edge")
        assert fresh_rm.state_dirty is True

    def test_mark_state_clean_clears_flag(self, fresh_rm: RiskManager):
        fresh_rm.record_fill("MKT-A", "weather_edge", 100.0)
        assert fresh_rm.state_dirty is True
        fresh_rm.mark_state_clean()
        assert fresh_rm.state_dirty is False


# ══════════════════════════════════════════════════════════════════════════
# Export State
# ══════════════════════════════════════════════════════════════════════════


class TestExportState:
    def test_export_contains_all_fields(self, fresh_rm: RiskManager):
        fresh_rm.record_fill("MKT-A", "weather_edge", 100.0, pnl=-5.0)
        export = fresh_rm.export_state()

        assert "peak_bankroll" in export
        assert "daily_pnl" in export
        assert "daily_start_bankroll" in export
        assert "trading_halted" in export
        assert "strategy_trackers" in export

    def test_export_returns_fresh_dicts(self, fresh_rm: RiskManager):
        """Exported dict must not share references with internal state."""
        fresh_rm.record_strategy_trade("s", pnl=1.0, won=True)
        export = fresh_rm.export_state()

        # Mutate the exported trades list — should not affect internal state
        export["strategy_trackers"]["s"]["trades"].clear()
        assert len(fresh_rm.get_tracker("s").trades) == 1


# ══════════════════════════════════════════════════════════════════════════
# Reconciliation (Orchestrator + mock Kalshi API)
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Settings in paper mode with Kalshi credentials for reconciliation tests."""
    from python.settings import ExecutionConfig

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return Settings(
        state_dir=str(state_dir),
        bankroll=10000.0,
        kalshi_api_key="test-key",
        kalshi_private_key_path="/dev/null",
        execution=ExecutionConfig(mode="paper"),
        _env_file=None,
    )


class TestReconcileFromKalshi:
    @pytest.mark.asyncio
    async def test_sets_positions_from_api(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 950000}  # $9500 in cents
        mock_client.iter_all_positions.return_value = (
            [
                # v2: 10 YES contracts; market_exposure_dollars is the canonical
                # cost-basis figure regardless of side (string in API).
                {
                    "ticker": "KXHIGHNY-T56",
                    "position_fp": "10",
                    "market_exposure_dollars": "4.00",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        # Bankroll is total equity (cash + open position notional), not bare cash.
        assert orch._risk_manager.bankroll == pytest.approx(9504.0)
        assert orch._risk_manager.positions == {"KXHIGHNY-T56": pytest.approx(4.0)}

    @pytest.mark.asyncio
    async def test_no_side_position(self, mock_settings: Settings):
        """Negative position_fp = NO side; exposure_dollars is positive."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1000000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-NO",
                    "position_fp": "-5",
                    "market_exposure_dollars": "1.50",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions["MKT-NO"] == pytest.approx(1.50)
        # cash $10,000 + position $1.50 = equity $10,001.50
        assert orch._risk_manager.bankroll == pytest.approx(10001.50)

    @pytest.mark.asyncio
    async def test_yes_position_uses_exposure_dollars_directly(
        self, mock_settings: Settings
    ):
        """v2 schema: dollars come straight from market_exposure_dollars,
        not derived from position * avg_price."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-YES",
                    "position_fp": "16",
                    "market_exposure_dollars": "11.000000",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {"MKT-YES": pytest.approx(11.0)}

    @pytest.mark.asyncio
    async def test_no_position_uses_exposure_dollars_directly(
        self, mock_settings: Settings
    ):
        """NO-side magnitude lives in market_exposure_dollars (already positive)."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-NO-LIVE",
                    "position_fp": "-16.00",
                    "market_exposure_dollars": "11.000000",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {"MKT-NO-LIVE": pytest.approx(11.0)}

    @pytest.mark.asyncio
    async def test_fractional_position_fp_is_kept(self, mock_settings: Settings):
        """Kalshi v2 emits fractional position_fp values (e.g. -6.02) after
        partial fills/conversions; they must not be rounded to zero."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-FRAC",
                    "position_fp": "-6.02",
                    "market_exposure_dollars": "3.25",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {"MKT-FRAC": pytest.approx(3.25)}

    @pytest.mark.asyncio
    async def test_zero_quantity_position_skipped(self, mock_settings: Settings):
        """position_fp=0 (fully closed) must not show up as a tracked position."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-ZERO",
                    "position_fp": "0",
                    "market_exposure_dollars": "0.000000",
                },
                {
                    "ticker": "MKT-LIVE",
                    "position_fp": "5",
                    "market_exposure_dollars": "2.50",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {"MKT-LIVE": pytest.approx(2.50)}

    @pytest.mark.asyncio
    async def test_missing_ticker_skipped(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {"ticker": "", "position_fp": "5", "market_exposure_dollars": "1.00"},
                {"position_fp": "5", "market_exposure_dollars": "1.00"},  # no ticker
                {
                    "ticker": "MKT-OK",
                    "position_fp": "5",
                    "market_exposure_dollars": "1.00",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {"MKT-OK": pytest.approx(1.0)}

    @pytest.mark.asyncio
    async def test_garbage_position_fp_is_treated_as_zero(
        self, mock_settings: Settings
    ):
        """Unparseable position_fp -> treated as 0 -> position skipped."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-BAD",
                    "position_fp": "not-a-number",
                    "market_exposure_dollars": "1.00",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.positions == {}

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_config(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.side_effect = Exception("API down")
        mock_client.iter_all_positions.side_effect = Exception("API down")
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.bankroll == 10000.0
        assert orch._risk_manager.positions == {}

    @pytest.mark.asyncio
    async def test_shadow_mode_uses_shadow_broker_not_kalshi(self, tmp_path: Path):
        """Shadow reconcile must not hit the Kalshi API — it routes through
        ShadowBroker against the local trade log instead."""
        from python.execution.engine import ExecutionEngine
        from python.orchestrator import Orchestrator

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        shadow_settings = Settings(
            state_dir=str(state_dir),
            bankroll=10000.0,
            _env_file=None,
        )

        orch = Orchestrator(shadow_settings)
        orch._risk_manager = RiskManager(shadow_settings)
        mock_client = AsyncMock()
        orch._kalshi_client = mock_client
        # Shadow reconcile needs the engine wired so it can inject stub records.
        orch._engine = ExecutionEngine(
            shadow_settings, kalshi_client=None, risk_manager=orch._risk_manager
        )

        await orch._reconcile_from_kalshi()

        # Kalshi API never touched in shadow mode
        mock_client.get_balance.assert_not_called()
        mock_client.iter_all_positions.assert_not_called()
        # With no trades.jsonl on disk, positions are empty but the
        # risk manager's bankroll was re-hydrated from config.
        assert orch._risk_manager.positions == {}
        assert orch._risk_manager.bankroll == 10000.0

    @pytest.mark.asyncio
    async def test_truncated_response_emits_critical_alert(
        self, mock_settings: Settings
    ):
        """When iter_all_positions returns truncated=True, fire CRITICAL Telegram alert."""
        from python.notifications.base import AlertLevel
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1000000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-1",
                    "position_fp": "1",
                    "market_exposure_dollars": "0.50",
                }
            ],
            True,
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        await orch._reconcile_from_kalshi()

        # _notify_alert calls notifier.send_alert internally
        truncation_calls = [
            c
            for c in mock_notifier.send_alert.call_args_list
            if "RECONCILIATION TRUNCATED" in c.args[0]
        ]
        assert len(truncation_calls) == 1
        assert truncation_calls[0].kwargs["level"] is AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_truncated_response_logs_critical(
        self, mock_settings: Settings, caplog
    ):
        """When iter_all_positions returns truncated=True, log at CRITICAL level."""
        import logging as _logging

        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1000000}
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-1",
                    "position_fp": "1",
                    "market_exposure_dollars": "0.50",
                }
            ],
            True,
        )
        orch._kalshi_client = mock_client

        with caplog.at_level(_logging.CRITICAL):
            await orch._reconcile_from_kalshi()

        assert any(
            "truncated at MAX_PAGES" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_iter_all_positions_timeout_alerts_and_returns_empty(
        self, mock_settings: Settings
    ):
        """When iter_all_positions times out, alert CRITICAL and leave positions empty."""
        from python.notifications.base import AlertLevel
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1000000}
        mock_client.iter_all_positions.side_effect = TimeoutError(
            "simulated wait_for timeout"
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        await orch._reconcile_from_kalshi()

        # Positions should remain empty after the timeout fallthrough.
        assert orch._risk_manager.positions == {}

        timeout_calls = [
            c
            for c in mock_notifier.send_alert.call_args_list
            if "RECONCILIATION TIMEOUT" in c.args[0]
        ]
        assert len(timeout_calls) == 1
        assert timeout_calls[0].kwargs["level"] is AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_periodic_reconciliation_timeout_alerts_and_returns_early(
        self, mock_settings: Settings
    ):
        """When iter_all_positions times out in periodic reconciliation, alert + early return."""
        from python.notifications.base import AlertLevel
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1000000}
        mock_client.iter_all_positions.side_effect = TimeoutError(
            "simulated wait_for timeout"
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        await orch._periodic_reconciliation()

        timeout_calls = [
            c
            for c in mock_notifier.send_alert.call_args_list
            if "RECONCILIATION TIMEOUT" in c.args[0]
            and "periodic check" in c.args[0]
        ]
        assert len(timeout_calls) == 1
        assert timeout_calls[0].kwargs["level"] is AlertLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_hydrates_equity_when_positions_present(
        self, mock_settings: Settings
    ):
        """Bankroll = cash + sum(open position notionals)."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 950000}  # $9500 cash
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-A",
                    "position_fp": "10",
                    "market_exposure_dollars": "4.00",
                },
                {
                    "ticker": "MKT-B",
                    "position_fp": "-5",
                    "market_exposure_dollars": "1.50",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        # equity = 9500 cash + 4.00 + 1.50 = 9505.50
        assert orch._risk_manager.bankroll == pytest.approx(9505.50)
        assert orch._risk_manager.total_exposure == pytest.approx(5.50)

    @pytest.mark.asyncio
    async def test_hydrates_cash_when_no_positions(self, mock_settings: Settings):
        """With no open positions, equity == cash."""
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 1234500}  # $12,345 cash
        mock_client.iter_all_positions.return_value = ([], False)
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        assert orch._risk_manager.bankroll == pytest.approx(12345.00)
        assert orch._risk_manager.positions == {}

    @pytest.mark.asyncio
    async def test_balance_fetch_failure_preserves_last_known_state(
        self, mock_settings: Settings
    ):
        """When the balance API call fails, do not clobber in-memory state.

        Production: if /portfolio/balance returns 5xx mid-session, the
        orchestrator should keep its last-known equity / positions rather
        than reset to whatever's in config (which is stale once any fills
        have happened).
        """
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        # Pre-hydrate to simulate a healthy prior reconciliation cycle.
        orch._risk_manager.hydrate_positions(
            {"MKT-PRIOR": 50.0}, bankroll=9_750.0
        )

        mock_client = AsyncMock()
        mock_client.get_balance.side_effect = Exception("API down")
        orch._kalshi_client = mock_client

        await orch._reconcile_from_kalshi()

        # State must be untouched — neither positions nor bankroll changed.
        assert orch._risk_manager.bankroll == pytest.approx(9_750.0)
        assert orch._risk_manager.positions == {"MKT-PRIOR": pytest.approx(50.0)}
        # And iter_all_positions must NOT have been called — we bailed early.
        mock_client.iter_all_positions.assert_not_called()


def _drift_log_messages(caplog) -> list[str]:
    """Reconciliation-divergence WARNING log messages captured by ``caplog``."""
    return [
        r.getMessage()
        for r in caplog.records
        if "Reconciliation divergence" in r.getMessage()
    ]


def _drift_push_calls(mock_notifier) -> list:
    """Any RECONCILIATION DRIFT push alerts (should always be empty — the drift
    push was removed in favour of the WARNING log audit trail)."""
    return [
        c
        for c in mock_notifier.send_alert.call_args_list
        if c.args and "RECONCILIATION DRIFT" in c.args[0]
    ]


class TestPeriodicReconciliationEquity:
    """Drift checks treat ``risk_manager.bankroll`` as total equity, not cash.

    Regression class for the false-positive 'Bankroll drift' alert that
    fired any time positions were open against a cash-only local
    bankroll.
    """

    @pytest.mark.asyncio
    async def test_no_alert_when_cash_plus_positions_equals_local_bankroll(
        self, mock_settings: Settings, caplog
    ):
        """False-positive sentinel: API cash $120.00 + position
        notional $50 must equal local equity $170.00 with no drift detected."""
        import logging as _logging

        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._risk_manager.hydrate_positions(
            {"KXHIGHDEN-26APR28-B54.5": 11.0, "MKT-OTHER": 39.0},
            bankroll=170.0,  # cash 120.00 + positions 50.0
        )

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 12000}  # $120.00
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "KXHIGHDEN-26APR28-B54.5",
                    "position_fp": "-16.00",
                    "market_exposure_dollars": "11.000000",
                },
                {
                    "ticker": "MKT-OTHER",
                    "position_fp": "20",
                    "market_exposure_dollars": "39.000000",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        with caplog.at_level(_logging.WARNING):
            await orch._periodic_reconciliation()

        # No divergence detected → nothing logged, nothing pushed.
        assert not _drift_log_messages(caplog)
        assert _drift_push_calls(mock_notifier) == []

    @pytest.mark.asyncio
    async def test_drift_is_logged_when_api_equity_diverges_from_local(
        self, mock_settings: Settings, caplog
    ):
        """Equity drift is detected + logged (the audit trail) but NOT pushed —
        the drift auto-syncs below, so there is no operator action."""
        import logging as _logging

        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        # Local equity $200.00; API will report $250.00 worth of equity.
        orch._risk_manager.hydrate_positions(
            {"MKT-LOCAL": 50.0}, bankroll=200.0
        )

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20000}  # $200 cash
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-LOCAL",
                    "position_fp": "20",
                    "market_exposure_dollars": "50.00",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        with caplog.at_level(_logging.WARNING):
            await orch._periodic_reconciliation()

        # api_equity = 200 + 50 = 250, local = 200, drift = 50
        messages = _drift_log_messages(caplog)
        assert len(messages) == 1
        assert "$250.00" in messages[0]
        assert "$200.00" in messages[0]
        # Drift auto-syncs — no push alert.
        assert _drift_push_calls(mock_notifier) == []

    @pytest.mark.asyncio
    async def test_threshold_respected_no_log(
        self, mock_settings: Settings, caplog
    ):
        """A small drift below the configured threshold must not log or push."""
        import logging as _logging

        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        # Threshold default is $1.00; pick a 50¢ delta.
        orch._risk_manager.hydrate_positions({}, bankroll=200.0)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 20050}  # $200.50
        mock_client.iter_all_positions.return_value = ([], False)
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        with caplog.at_level(_logging.WARNING):
            await orch._periodic_reconciliation()

        assert not _drift_log_messages(caplog)
        assert _drift_push_calls(mock_notifier) == []

    @pytest.mark.asyncio
    async def test_drift_log_shows_cash_and_positions_breakdown(
        self, mock_settings: Settings, caplog
    ):
        """The divergence log spells out cash and position contributions."""
        import logging as _logging

        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._risk_manager.hydrate_positions({}, bankroll=100.0)

        mock_client = AsyncMock()
        mock_client.get_balance.return_value = {"balance": 15000}  # $150 cash
        mock_client.iter_all_positions.return_value = (
            [
                {
                    "ticker": "MKT-NEW",
                    "position_fp": "10",
                    "market_exposure_dollars": "25.00",
                },
            ],
            False,
        )
        orch._kalshi_client = mock_client

        mock_notifier = AsyncMock()
        orch._notifier = mock_notifier

        with caplog.at_level(_logging.WARNING):
            await orch._periodic_reconciliation()

        # The equity-drift line spells out the cash + position breakdown.
        messages = _drift_log_messages(caplog)
        assert any(
            "cash=$150.00" in m and "positions=$25.00" in m for m in messages
        )
        assert _drift_push_calls(mock_notifier) == []


class TestRestoreRiskState:
    def test_no_file_starts_fresh(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._restore_risk_state()

        # Should be default state
        assert orch._risk_manager.daily_pnl == 0.0

    @pytest.mark.asyncio
    async def test_restores_from_file(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        # Save state matching current mode (paper)
        data = serialize_risk_state(**_sample_state_data(
            peak_bankroll=12000.0,
            daily_pnl=-30.0,
            execution_mode="paper",
            current_trading_date=datetime.now(ET).date().isoformat(),
        ))
        await save_state(Path(mock_settings.state_dir), data)

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._restore_risk_state()

        assert orch._risk_manager.peak_bankroll == 12000.0
        assert orch._risk_manager.daily_pnl == -30.0

    @pytest.mark.asyncio
    async def test_day_boundary_resets_daily_counters(self, mock_settings: Settings):
        from python.orchestrator import Orchestrator

        # Save state from yesterday in matching mode
        yesterday = (datetime.now(ET) - timedelta(days=1)).date().isoformat()
        data = serialize_risk_state(**_sample_state_data(
            peak_bankroll=12000.0,
            daily_pnl=-100.0,
            trading_halted=True,
            execution_mode="paper",
            current_trading_date=yesterday,
        ))
        await save_state(Path(mock_settings.state_dir), data)

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._restore_risk_state()

        # Daily counters should be reset
        assert orch._risk_manager.daily_pnl == 0.0
        assert orch._risk_manager.is_halted is False
        # But peak should be preserved
        assert orch._risk_manager.peak_bankroll == 12000.0

    @pytest.mark.asyncio
    async def test_mode_switch_skips_financial_state(self, mock_settings: Settings):
        """Switching from shadow to paper should not load shadow P&L."""
        from python.orchestrator import Orchestrator

        # State file was saved in shadow mode
        data = serialize_risk_state(**_sample_state_data(
            peak_bankroll=15000.0,
            daily_pnl=-500.0,
            execution_mode="shadow",
            current_trading_date=datetime.now(ET).date().isoformat(),
        ))
        await save_state(Path(mock_settings.state_dir), data)

        # mock_settings is paper mode — different from saved "shadow"
        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._restore_risk_state()

        # Financial state should NOT be loaded from shadow
        assert orch._risk_manager.daily_pnl == 0.0
        assert orch._risk_manager.peak_bankroll == mock_settings.bankroll


# ── exposure override persistence ─────────────────────────────────────────


class TestExposureOverridePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        save_exposure_override(tmp_path, 0.25)
        assert load_exposure_override(tmp_path) == 0.25

    def test_save_none_removes_file(self, tmp_path: Path):
        save_exposure_override(tmp_path, 0.25)
        path = tmp_path / EXPOSURE_OVERRIDE_FILENAME
        assert path.exists()

        save_exposure_override(tmp_path, None)
        assert not path.exists()

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_exposure_override(tmp_path) is None

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        path = tmp_path / EXPOSURE_OVERRIDE_FILENAME
        path.write_text("{not valid json")
        assert load_exposure_override(tmp_path) is None

    def test_schema_mismatch_returns_none(self, tmp_path: Path):
        path = tmp_path / EXPOSURE_OVERRIDE_FILENAME
        path.write_text(json.dumps({
            "schema_version": EXPOSURE_OVERRIDE_SCHEMA_VERSION + 99,
            "max_total_exposure_pct": 0.25,
        }))
        assert load_exposure_override(tmp_path) is None

    def test_out_of_range_returns_none(self, tmp_path: Path):
        path = tmp_path / EXPOSURE_OVERRIDE_FILENAME
        path.write_text(json.dumps({
            "schema_version": EXPOSURE_OVERRIDE_SCHEMA_VERSION,
            "max_total_exposure_pct": 1.5,
        }))
        assert load_exposure_override(tmp_path) is None

        path.write_text(json.dumps({
            "schema_version": EXPOSURE_OVERRIDE_SCHEMA_VERSION,
            "max_total_exposure_pct": 0.0,
        }))
        assert load_exposure_override(tmp_path) is None

    def test_save_is_atomic_no_partial_file(self, tmp_path: Path):
        """Basic sanity — after save the file is valid JSON."""
        save_exposure_override(tmp_path, 0.5)
        path = tmp_path / EXPOSURE_OVERRIDE_FILENAME
        data = json.loads(path.read_text())
        assert data["schema_version"] == EXPOSURE_OVERRIDE_SCHEMA_VERSION
        assert data["max_total_exposure_pct"] == 0.5
        assert "saved_at" in data


class TestPersistIfDirty:
    """B1: ``Orchestrator._persist_if_dirty`` must not race two saves.

    The hot path used to overwrite ``self._save_task`` if the previous
    save was still running, leaving two ``asyncio.to_thread`` workers
    racing against the same ``risk_state.json``. ``os.replace`` is
    atomic but the *content* is whichever writer finishes second — so
    a slow save followed by a fast one regressed persisted state.

    The fix: skip when the previous task is still in flight (defer to
    the next tick; the dirty flag is preserved), with an inner
    ``asyncio.Lock`` as belt-and-suspenders against future refactors.
    """

    @staticmethod
    def _make_orch(mock_settings: Settings):
        from python.orchestrator import Orchestrator

        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        return orch

    @pytest.mark.asyncio
    async def test_skips_when_save_in_flight(self, mock_settings: Settings):
        """Second tick must not create a second concurrent save task."""
        import asyncio
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        orch._risk_manager._state_dirty = True

        slow_started = asyncio.Event()
        slow_release = asyncio.Event()

        async def slow_save(_state_dir, _snapshot):
            slow_started.set()
            await slow_release.wait()
            return True

        with patch("python.orchestrator.save_state", side_effect=slow_save):
            orch._persist_if_dirty()
            first_task = orch._save_task
            assert first_task is not None
            await slow_started.wait()
            # Re-dirty so the second tick has reason to save.
            orch._risk_manager._state_dirty = True

            orch._persist_if_dirty()
            # No new task created — the in-flight one is preserved.
            assert orch._save_task is first_task
            # Dirty flag preserved so the change isn't lost.
            assert orch._risk_manager.state_dirty is True

            slow_release.set()
            await first_task

    @pytest.mark.asyncio
    async def test_creates_new_task_after_previous_completes(
        self, mock_settings: Settings
    ):
        """Once the previous save is done, the next dirty tick saves again."""
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        orch._risk_manager._state_dirty = True

        save_calls: list[Any] = []

        async def fast_save(_state_dir, snapshot):
            save_calls.append(snapshot)
            return True

        with patch("python.orchestrator.save_state", side_effect=fast_save):
            orch._persist_if_dirty()
            first_task = orch._save_task
            assert first_task is not None
            await first_task

            orch._risk_manager._state_dirty = True
            orch._persist_if_dirty()
            second_task = orch._save_task
            assert second_task is not None
            assert second_task is not first_task
            await second_task

        assert len(save_calls) == 2

    @pytest.mark.asyncio
    async def test_no_save_when_clean(self, mock_settings: Settings):
        """Without a dirty flag, no save task is created."""
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        # Risk manager starts clean.
        assert orch._risk_manager.state_dirty is False

        with patch("python.orchestrator.save_state") as save_mock:
            orch._persist_if_dirty()

        save_mock.assert_not_called()
        assert orch._save_task is None


# ══════════════════════════════════════════════════════════════════════════
# B8: consecutive_errors persistence
# ══════════════════════════════════════════════════════════════════════════


class TestStrategyErrorPersistence:
    """B8: consecutive_errors must round-trip and survive restarts."""

    def test_round_trip_default_zero(self):
        # Tracker with no consecutive_errors field defaults to 0
        data = _sample_state_data()
        snapshot = deserialize_risk_state(serialize_risk_state(**data))
        assert snapshot is not None
        for tracker in snapshot.strategy_trackers.values():
            assert tracker["consecutive_errors"] == 0

    def test_round_trip_nonzero_value(self):
        data = _sample_state_data(strategy_trackers={
            "weather_edge": {
                "enabled": True,
                "disabled_reason": "",
                "consecutive_errors": 7,
                "trades": [],
            }
        })
        snapshot = deserialize_risk_state(serialize_risk_state(**data))
        assert snapshot is not None
        assert snapshot.strategy_trackers["weather_edge"]["consecutive_errors"] == 7

    def test_old_state_file_without_field_loads_as_zero(self):
        # A serialized blob that predates the field — manually constructed.
        blob = {
            "schema_version": STATE_SCHEMA_VERSION,
            "saved_at": "2026-04-26T00:00:00+00:00",
            "execution_mode": "shadow",
            "peak_bankroll": 10000.0,
            "daily_pnl": 0.0,
            "daily_start_bankroll": 10000.0,
            "trading_halted": False,
            "current_trading_date": "2026-04-26",
            "strategy_trackers": {
                "weather_edge": {
                    "enabled": True,
                    "disabled_reason": "",
                    "trades": [],
                    # NO consecutive_errors key
                }
            },
        }
        snapshot = deserialize_risk_state(blob)
        assert snapshot is not None
        assert snapshot.strategy_trackers["weather_edge"]["consecutive_errors"] == 0

    def test_record_strategy_error_increments_and_dirties(self, fresh_rm: RiskManager):
        new = fresh_rm.record_strategy_error("weather_edge")
        assert new == 1
        assert fresh_rm.get_tracker("weather_edge").consecutive_errors == 1
        assert fresh_rm.state_dirty is True
        assert fresh_rm.record_strategy_error("weather_edge") == 2

    def test_clear_strategy_errors_resets(self, fresh_rm: RiskManager):
        fresh_rm.record_strategy_error("weather_edge")
        fresh_rm.record_strategy_error("weather_edge")
        fresh_rm.mark_state_clean()
        fresh_rm.clear_strategy_errors("weather_edge")
        assert fresh_rm.get_tracker("weather_edge").consecutive_errors == 0
        assert fresh_rm.state_dirty is True

    def test_clear_strategy_errors_when_zero_is_noop(self, fresh_rm: RiskManager):
        # Already zero — must not set the dirty flag (avoids forcing a
        # save every tick on the success path).
        assert fresh_rm.get_tracker("weather_edge").consecutive_errors == 0
        fresh_rm.clear_strategy_errors("weather_edge")
        assert fresh_rm.state_dirty is False

    def test_export_state_includes_consecutive_errors(self, fresh_rm: RiskManager):
        fresh_rm.record_strategy_error("weather_edge")
        fresh_rm.record_strategy_error("weather_edge")
        export = fresh_rm.export_state()
        assert export["strategy_trackers"]["weather_edge"]["consecutive_errors"] == 2

    def test_hydrate_from_state_restores_counter(self, fresh_rm: RiskManager):
        snapshot = RiskStateSnapshot(
            peak_bankroll=10000.0,
            daily_pnl=0.0,
            daily_start_bankroll=10000.0,
            trading_halted=False,
            current_trading_date="2026-04-26",
            execution_mode="shadow",
            strategy_trackers={
                "weather_edge": {
                    "enabled": True,
                    "disabled_reason": "",
                    "consecutive_errors": 9,
                    "trades": [],
                }
            },
        )
        fresh_rm.hydrate_from_state(snapshot)
        assert fresh_rm.get_tracker("weather_edge").consecutive_errors == 9

    @pytest.mark.asyncio
    async def test_full_save_load_round_trip_via_disk(
        self, tmp_path: Path, fresh_rm: RiskManager
    ):
        """End-to-end: increment counter, save, reload, counter survives."""
        fresh_rm.record_strategy_error("weather_edge")
        fresh_rm.record_strategy_error("weather_edge")
        fresh_rm.record_strategy_error("weather_edge")

        snapshot_dict = serialize_risk_state(
            **fresh_rm.export_state(),
            current_trading_date="2026-04-26",
            execution_mode="shadow",
        )
        ok = await save_state(tmp_path, snapshot_dict)
        assert ok is True

        reloaded = load_state(tmp_path)
        assert reloaded is not None
        assert reloaded.strategy_trackers["weather_edge"]["consecutive_errors"] == 3


# ══════════════════════════════════════════════════════════════════════════
# B9: save_state return contract + failure counter + alerting
# ══════════════════════════════════════════════════════════════════════════


class TestSaveStateReturnValue:
    """save_state must return True/False so the orchestrator can branch."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path):
        data = serialize_risk_state(**_sample_state_data())
        result = await save_state(tmp_path, data)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_io_error(self, tmp_path: Path):
        # Serializing an unserializable object forces json.dumps to raise
        # before _atomic_write is even called — exercises the except path.
        bad_data = {"unserializable": object()}
        result = await save_state(tmp_path, bad_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_atomic_write_error(
        self, tmp_path: Path, monkeypatch
    ):
        from python.risk import persistence as p

        def boom(path, content):  # noqa: ARG001
            raise OSError("disk full")

        monkeypatch.setattr(p, "_atomic_write", boom)
        result = await save_state(tmp_path, serialize_risk_state(**_sample_state_data()))
        assert result is False


class TestSaveFailureCounter:
    def test_initial_zero(self, fresh_rm: RiskManager):
        assert fresh_rm.consecutive_save_failures == 0

    def test_record_increments(self, fresh_rm: RiskManager):
        assert fresh_rm.record_save_failure() == 1
        assert fresh_rm.record_save_failure() == 2
        assert fresh_rm.consecutive_save_failures == 2

    def test_clear_resets_to_zero(self, fresh_rm: RiskManager):
        fresh_rm.record_save_failure()
        fresh_rm.record_save_failure()
        fresh_rm.clear_save_failures()
        assert fresh_rm.consecutive_save_failures == 0

    def test_failure_counter_does_not_set_dirty_flag(self, fresh_rm: RiskManager):
        # The save-failure counter is in-memory only — persisting it into
        # the same file the save is failing on would be incoherent.
        fresh_rm.record_save_failure()
        assert fresh_rm.state_dirty is False
        fresh_rm.clear_save_failures()
        assert fresh_rm.state_dirty is False


class TestSaveFailureAlerting:
    """B9: ``_save_under_lock`` must fire CRITICAL after N failures and re-arm on success."""

    @staticmethod
    def _make_orch(mock_settings):
        from python.orchestrator import Orchestrator
        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._notifier = AsyncMock()
        return orch

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, mock_settings: Settings):
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        with patch("python.orchestrator.save_state", AsyncMock(return_value=False)):
            for _ in range(SAVE_FAILURE_ALERT_THRESHOLD - 1):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})

        critical_calls = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
        ]
        assert critical_calls == []
        assert (
            orch._risk_manager.consecutive_save_failures
            == SAVE_FAILURE_ALERT_THRESHOLD - 1
        )

    @pytest.mark.asyncio
    async def test_alert_fires_at_threshold(self, mock_settings: Settings):
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        with patch("python.orchestrator.save_state", AsyncMock(return_value=False)):
            for _ in range(SAVE_FAILURE_ALERT_THRESHOLD):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})

        critical = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
            and PERSISTENCE_BROKEN_ALERT_PREFIX in c.args[0]
        ]
        assert len(critical) == 1
        assert orch._save_failure_alert_sent is True

    @pytest.mark.asyncio
    async def test_alert_does_not_repeat_above_threshold(self, mock_settings: Settings):
        """Once fired, no re-fire on each subsequent failure."""
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        with patch("python.orchestrator.save_state", AsyncMock(return_value=False)):
            for _ in range(SAVE_FAILURE_ALERT_THRESHOLD + 5):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})

        critical = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
            and PERSISTENCE_BROKEN_ALERT_PREFIX in c.args[0]
        ]
        assert len(critical) == 1

    @pytest.mark.asyncio
    async def test_success_resets_counter_and_rearms(self, mock_settings: Settings):
        """A successful save clears the counter and re-arms the alert."""
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        # Fail until alert fires
        with patch("python.orchestrator.save_state", AsyncMock(return_value=False)):
            for _ in range(SAVE_FAILURE_ALERT_THRESHOLD):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})
        assert orch._save_failure_alert_sent is True

        # One successful save clears state
        with patch("python.orchestrator.save_state", AsyncMock(return_value=True)):
            await orch._save_under_lock(Path(mock_settings.state_dir), {})
        assert orch._risk_manager.consecutive_save_failures == 0
        assert orch._save_failure_alert_sent is False

        # New burst of failures should fire the alert again
        with patch("python.orchestrator.save_state", AsyncMock(return_value=False)):
            for _ in range(SAVE_FAILURE_ALERT_THRESHOLD):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})

        critical = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
            and PERSISTENCE_BROKEN_ALERT_PREFIX in c.args[0]
        ]
        assert len(critical) == 2

    @pytest.mark.asyncio
    async def test_transient_failure_does_not_fire(self, mock_settings: Settings):
        """1 fail, then success, then fail, etc — counter resets each
        success, never trips."""
        from unittest.mock import patch

        orch = self._make_orch(mock_settings)
        responses = [False, True, True, False, True, False, True, False, True]
        idx = {"i": 0}

        async def flaky(_state_dir, _data):
            i = idx["i"]
            idx["i"] += 1
            return responses[i]

        with patch("python.orchestrator.save_state", side_effect=flaky):
            for _ in range(len(responses)):
                await orch._save_under_lock(Path(mock_settings.state_dir), {})

        critical = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
        ]
        assert critical == []
        assert orch._risk_manager.consecutive_save_failures == 0  # ended on success


class TestRunStrategyConsecutiveErrors:
    """B8: _run_strategy uses the public RiskManager API and dirties state."""

    @staticmethod
    def _make_orch(mock_settings):
        from python.orchestrator import Orchestrator
        orch = Orchestrator(mock_settings)
        orch._risk_manager = RiskManager(mock_settings)
        orch._notifier = AsyncMock()
        return orch

    @pytest.mark.asyncio
    async def test_success_clears_counter_via_public_api(
        self, mock_settings: Settings
    ):
        orch = self._make_orch(mock_settings)
        # Pre-load a tracker with errors, simulating a prior failure
        # streak that recovered this tick.
        orch._risk_manager.record_strategy_error("weather_edge")
        orch._risk_manager.record_strategy_error("weather_edge")
        orch._risk_manager.mark_state_clean()

        # Cleanest path: invoke the public API directly and assert on the
        # observable state. The "via tick" path is covered by
        # test_critical_alert_at_ten_consecutive below.
        orch._risk_manager.clear_strategy_errors("weather_edge")
        assert orch._risk_manager.get_tracker("weather_edge").consecutive_errors == 0
        assert orch._risk_manager.state_dirty is True  # transitioned 2 -> 0

    @pytest.mark.asyncio
    async def test_failure_increments_via_public_api(self, mock_settings: Settings):
        orch = self._make_orch(mock_settings)
        new = orch._risk_manager.record_strategy_error("weather_edge")
        assert new == 1
        assert orch._risk_manager.state_dirty is True
        new = orch._risk_manager.record_strategy_error("weather_edge")
        assert new == 2

    @pytest.mark.asyncio
    async def test_critical_alert_at_ten_consecutive(
        self, mock_settings: Settings, monkeypatch
    ):
        """Regression for the audit-cited 10-error CRITICAL threshold.

        Drives the actual _run_strategy via _tick with a single failing
        strategy.
        """
        import asyncio

        from python.execution.engine import ExecutionEngine
        from python.risk.manager import STRATEGY_ERROR_ALERT_THRESHOLD
        from python.strategies.base import Strategy

        # Use shadow-mode settings so periodic reconciliation is naturally
        # skipped (the paper-mode default would invoke it). Cleaner than
        # mutating execution.reconcile_interval_ticks = 0 — that field is
        # `Field(ge=1)`; shadow mode is the natural branch.
        mock_settings.execution.mode = "shadow"

        orch = self._make_orch(mock_settings)
        orch._engine = ExecutionEngine(
            mock_settings, kalshi_client=None, risk_manager=orch._risk_manager
        )

        strategy = AsyncMock(spec=Strategy)
        strategy.name = "weather_edge"
        strategy.enabled = True
        strategy.generate_signals = AsyncMock(side_effect=RuntimeError("boom"))
        strategy.update_bankroll = lambda *_a, **_kw: None
        orch._strategies = [strategy]

        # Avoid tick-side effects we don't care about
        monkeypatch.setattr(orch, "_check_ws_health", AsyncMock())
        monkeypatch.setattr(orch, "_ping_healthcheck", AsyncMock())
        monkeypatch.setattr(
            orch._engine, "settle_expired_positions", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            orch._engine,
            "check_all_pending_fills",
            AsyncMock(return_value=([], [])),
        )
        # Kill switch checks `STOP` file in state_dir — never created in tmp_path,
        # so the natural branch returns False without needing a monkeypatch
        # (Pydantic v2 doesn't allow setattr on method names of Settings).
        # _persist_if_dirty would try to save — patch save_state to a no-op
        monkeypatch.setattr(
            "python.orchestrator.save_state", AsyncMock(return_value=True)
        )

        # Drive enough ticks to hit the alert threshold; each fails the
        # strategy and increments the counter.
        for _ in range(STRATEGY_ERROR_ALERT_THRESHOLD):
            await orch._tick()

        # Drain in-flight save + heartbeat tasks so we don't leak
        # `Task was destroyed but it is pending!` warnings. Each _tick spawns
        # both via _persist_if_dirty.
        for _ in range(2):
            await asyncio.sleep(0)
        if orch._save_task is not None and not orch._save_task.done():
            await orch._save_task
        if orch._heartbeat_task is not None and not orch._heartbeat_task.done():
            await orch._heartbeat_task

        tracker = orch._risk_manager.get_tracker("weather_edge")
        assert tracker.consecutive_errors == STRATEGY_ERROR_ALERT_THRESHOLD

        critical = [
            c for c in orch._notifier.send_alert.call_args_list
            if c.kwargs.get("level") is AlertLevel.CRITICAL
            and f"{STRATEGY_ERROR_ALERT_THRESHOLD} consecutive errors" in c.args[0]
        ]
        assert len(critical) >= 1
