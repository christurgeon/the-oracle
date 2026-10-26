"""Resolver for the calibrator-artifact staleness tripwire.

``Orchestrator._tripwire_calibrator_paths`` returns the ``{label: path}`` dict
that ``run_all`` ages-checks. The EMOS coefficients artifact must be monitored
only when EMOS is actually live (``weather.emos_enabled_cells`` non-empty). With
the bot all-isotonic (empty list, the state since 2026-06-05),
``data/emos/coefficients.json`` is a dormant future-re-add option — staleness
alerts on it false-alarm on a file no trade depends on. Same misdirected-
tripwire class as ``_tripwire_bias_table_path`` (the 2026-06-21 fix): monitor
the artifact the live config loads, never one keyed off mere existence on disk.

The resolver only reads ``self._settings.strategies.weather.emos_enabled_cells``
and calls ``self._isotonic_artifact_path``; it is exercised here with a duck-
typed ``self`` so the test stays a fast pure-logic check (no full
``Orchestrator`` / client construction).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from python.orchestrator import Orchestrator

# Committed repo fixtures the live resolver references by relative path. The
# ``.exists()`` filter in the resolver depends on these being present (pytest
# runs from repo root).
_ISOTONIC = Path("data/calibration/isotonic_v2.json")
_EMOS = Path("data/emos/coefficients.json")
_FB_MAP = Path("data/calibration/favorite_bias_isotonic.json")


def _fake_self(
    *,
    emos_enabled_cells: list[str],
    favorite_bias_enabled: bool = False,
    favorite_bias_path: str = str(_FB_MAP),
) -> SimpleNamespace:
    weather = SimpleNamespace(emos_enabled_cells=emos_enabled_cells)
    favorite_bias = SimpleNamespace(
        enabled=favorite_bias_enabled,
        isotonic_path=favorite_bias_path,
    )
    strategies = SimpleNamespace(weather=weather, favorite_bias=favorite_bias)
    settings = SimpleNamespace(strategies=strategies)
    return SimpleNamespace(
        _settings=settings,
        _isotonic_artifact_path=lambda: _ISOTONIC,
    )


def test_emos_excluded_when_no_cells_enabled() -> None:
    """All-isotonic (empty list) → the dormant EMOS artifact is not monitored."""
    fake = _fake_self(emos_enabled_cells=[])
    paths = Orchestrator._tripwire_calibrator_paths(fake)
    assert "emos" not in paths
    assert paths == {"isotonic": _ISOTONIC}


def test_emos_included_when_a_cell_is_enabled() -> None:
    """At least one EMOS-served cell → monitor the coefficients for staleness."""
    assert _EMOS.exists(), "committed EMOS coefficients fixture is required"
    fake = _fake_self(emos_enabled_cells=["MIA/high"])
    paths = Orchestrator._tripwire_calibrator_paths(fake)
    assert paths.get("emos") == _EMOS


def test_missing_isotonic_file_is_filtered_out() -> None:
    """The ``.exists()`` filter still drops a non-existent live artifact."""
    fake = _fake_self(emos_enabled_cells=[])
    fake._isotonic_artifact_path = lambda: Path("data/calibration/does_not_exist.json")
    paths = Orchestrator._tripwire_calibrator_paths(fake)
    assert paths == {}


def test_favorite_bias_map_is_monitored_when_the_strategy_is_enabled() -> None:
    """Regression: the g(mid) map behind the only order-placing strategy had NO
    staleness coverage, while the ``isotonic`` entry covered an artifact serving
    shadow-only ``weather_edge`` and backing no money."""
    assert _FB_MAP.exists(), "committed favorite_bias map fixture is required"
    fake = _fake_self(emos_enabled_cells=[], favorite_bias_enabled=True)
    paths = Orchestrator._tripwire_calibrator_paths(fake)
    assert paths.get("favorite_bias") == _FB_MAP


def test_favorite_bias_map_not_monitored_when_the_strategy_is_disabled() -> None:
    """Mirrors the EMOS rule: do not age-check a calibrator no trade depends on."""
    fake = _fake_self(emos_enabled_cells=[], favorite_bias_enabled=False)
    assert "favorite_bias" not in Orchestrator._tripwire_calibrator_paths(fake)


def test_favorite_bias_honors_the_configured_path() -> None:
    """The tripwire must read the artifact the live config loads, not a default."""
    fake = _fake_self(
        emos_enabled_cells=[],
        favorite_bias_enabled=True,
        favorite_bias_path="data/calibration/does_not_exist.json",
    )
    paths = Orchestrator._tripwire_calibrator_paths(fake)
    assert "favorite_bias" not in paths  # resolved, then dropped by .exists()
