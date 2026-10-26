"""Tripwire: the live weather isotonic artifact must not defeat ``min_calibrated_prob``.

Reads ``python/config.yaml`` directly (not ``Settings``) so ``ORACLE_*`` env
overrides cannot make the result machine-dependent. If ``values[0]`` of the
configured artifact rises above ``min_calibrated_prob - epsilon``, the floor
stops suppressing long-shot trades (2026-05-19 incident). Raise the floor or
refit; do not loosen this test. Covers only ``strategies.weather`` (the parked
``intraday_asos_high`` floor is out of scope).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.validate_plateau_bounds import floor_coupling_check

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ISOTONIC_PATH = "data/calibration/isotonic.json"


def _live_weather_config() -> dict[str, Any]:
    cfg = yaml.safe_load((_REPO_ROOT / "python" / "config.yaml").read_text())
    return cfg["strategies"]["weather"]


def test_live_isotonic_floor_does_not_defeat_min_calibrated_prob() -> None:
    weather = _live_weather_config()
    floor = weather.get("min_calibrated_prob")
    if floor is None:
        pytest.skip("strategies.weather.min_calibrated_prob is null")

    artifact_path = _REPO_ROOT / (weather.get("isotonic_path") or _DEFAULT_ISOTONIC_PATH)
    values_zero = float(json.loads(artifact_path.read_text())["values"][0])

    result = floor_coupling_check(values_zero=values_zero, min_calibrated_prob=floor)

    assert result["status"] == "ok", (
        f"min_calibrated_prob={floor} values[0]={values_zero} ({artifact_path.name}): "
        f"{result['message']}\n"
        f"Run `uv run python scripts/validate_plateau_bounds.py --isotonic {artifact_path}`; "
        "see docs/calibration-runbook.md 'Weather isotonic refit gates'. "
        "Raise the floor or reject the refit; do not loosen this test."
    )
