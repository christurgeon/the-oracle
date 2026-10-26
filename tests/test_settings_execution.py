from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.settings import ExecutionConfig


def test_idle_timeout_defaults_to_90() -> None:
    assert ExecutionConfig().websocket_idle_timeout_seconds == 90


def test_idle_timeout_loads_from_value() -> None:
    assert ExecutionConfig(websocket_idle_timeout_seconds=120).websocket_idle_timeout_seconds == 120


@pytest.mark.parametrize("bad", [9, 281, 0, -1])
def test_idle_timeout_rejects_out_of_bounds(bad: int) -> None:
    with pytest.raises(ValidationError):
        ExecutionConfig(websocket_idle_timeout_seconds=bad)
