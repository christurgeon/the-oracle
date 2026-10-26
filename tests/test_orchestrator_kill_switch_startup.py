"""The startup kill-switch line lets a post-mortem close a kill that a restart cleared."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from python.orchestrator import Orchestrator


def _orch(*, active: bool | None = None, error: Exception | None = None) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    settings = MagicMock()
    if error is not None:
        settings.is_kill_switch_active.side_effect = error
    else:
        settings.is_kill_switch_active.return_value = active
    orch._settings = settings
    return orch


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "python.orchestrator"]


def test_logs_inactive_when_no_stop_file(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        _orch(active=False)._log_kill_switch_state_at_startup()

    assert _messages(caplog) == ["Kill switch inactive at startup"]


def test_logs_active_when_stop_file_present(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        _orch(active=True)._log_kill_switch_state_at_startup()

    assert _messages(caplog) == ["Kill switch ACTIVE at startup"]
    assert caplog.records[-1].levelno == logging.WARNING


def test_check_failure_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="python.orchestrator"):
        _orch(error=RuntimeError("boom"))._log_kill_switch_state_at_startup()

    assert "Kill switch state unknown at startup" in _messages(caplog)
