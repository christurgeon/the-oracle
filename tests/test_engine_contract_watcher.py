"""Tests that ExecutionEngine calls ContractWatcher at the right parse sites."""
from __future__ import annotations

from unittest.mock import MagicMock

from python.data.contract_watcher import ContractWatcher
from python.execution.engine import ExecutionEngine, _parse_order_status


def _make_engine(watcher: ContractWatcher) -> ExecutionEngine:
    settings = MagicMock()
    settings.execution.mode = "shadow"
    settings.state_dir = "/tmp/oracle_test"
    risk = MagicMock()
    return ExecutionEngine(
        settings=settings,
        kalshi_client=None,
        risk_manager=risk,
        contract_watcher=watcher,
    )


def test_engine_accepts_contract_watcher() -> None:
    watcher = ContractWatcher()
    engine = _make_engine(watcher)
    assert engine._contract_watcher is watcher


def test_parse_order_status_unknown_calls_watcher_warn() -> None:
    fired: list[str] = []
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    result = _parse_order_status("totally_new_status", contract_watcher=watcher)

    assert result.value in ("pending",)  # falls back to PENDING default
    assert len(fired) == 1
    assert "totally_new_status" in fired[0]
    assert "[CONTRACT]" in fired[0]


def test_parse_order_status_known_does_not_warn() -> None:
    fired: list[str] = []
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    _parse_order_status("resting", contract_watcher=watcher)

    assert fired == []


def test_parse_order_status_alias_does_not_warn() -> None:
    fired: list[str] = []
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    result = _parse_order_status("canceled", contract_watcher=watcher)

    assert fired == []
    from python.common import OrderStatus
    assert result == OrderStatus.CANCELLED
