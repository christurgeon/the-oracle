"""Weather_edge regression tests for the EmosTableHolder pattern (M1).

Load-bearing assertions:
  1. A real ``WeatherEdgeStrategy`` reads the EmosTable via the shared holder
     and sees the orchestrator's rebind immediately. A second consumer
     sharing the same holder reference sees the SAME post-rebind table —
     this is the exact bug class v1.3 MINOR #3 warned about (rebinding
     without a holder leaves siblings on stale references).
  2. ``python/strategies/weather_edge.py`` no longer contains the substring
     ``check_and_reload``. Hot-reload ownership now lives on the
     orchestrator side (spec §4.2); the strategy MUST NOT trigger it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from python.data.emos import EmosTable, EmosTableHolder
from python.strategies.weather_edge import WeatherEdgeStrategy


def _payload(a_value: float = -4.0) -> dict:
    return {
        "fit_date": "2026-05-04", "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": a_value, "b": 1.05, "c": 15.0, "d": 0.4,
                "n_train": 127, "leads_pooled": [0, 1, 2], "ens_std_p99": 6.5,
            },
            "MIA/high": {
                "a": a_value + 1.0, "b": 1.00, "c": 12.0, "d": 0.3,
                "n_train": 110, "leads_pooled": [0, 1, 2], "ens_std_p99": 5.8,
            },
        },
    }


def test_weather_edge_and_sibling_consumer_share_holder_observe_orchestrator_rebind(
    tmp_path: Path,
) -> None:
    """Construct a real ``WeatherEdgeStrategy`` with an injected
    ``EmosTableHolder``. Construct a synthetic sibling consumer (representing
    the future intraday_emos_high strategy) holding the SAME holder reference.

    Simulate an orchestrator tick: mutate the coefficients artifact on disk,
    then perform the orchestrator's rebind:
        ``holder.table = holder.table.check_and_reload() or holder.table``.

    Both consumers' read of ``self._emos_holder.table.get(...).a`` must
    return the NEW coefficient value, NOT the value pinned at construction
    time. This is the bug class spec §12.9 mandates the holder to eliminate:
    if either consumer held a direct ``EmosTable`` reference instead of the
    holder, the orchestrator's rebind would leave that consumer stale.
    """
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_payload(-4.0)))
    initial = EmosTable.load(coeffs_path)
    assert initial.is_loaded
    holder = EmosTableHolder(table=initial)
    pre_hash = holder.table.artifact_hash

    # Real WeatherEdgeStrategy with the orchestrator's holder injected.
    weather = WeatherEdgeStrategy(
        config={"emos_coefficients_path": str(coeffs_path)},
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
        emos_holder=holder,
    )

    # Synthetic sibling: minimal stand-in for the future intraday strategy.
    # It holds the SAME holder reference and reads coefficients via the
    # holder, matching the production access pattern.
    class _SiblingConsumer:
        def __init__(self, h: EmosTableHolder) -> None:
            self._emos_holder = h

        def current_a(self, city: str, var: str) -> float | None:
            coeffs = self._emos_holder.table.get(city=city, var=var, lead=0)
            return None if coeffs is None else coeffs.a

    sibling = _SiblingConsumer(holder)

    # Pre-rebind: both consumers see the initial coefficients (a=-4.0 for AUS,
    # a=-3.0 for MIA).
    aus_coeffs = weather.emos_table.get(city="AUS", var="high", lead=0)  # type: ignore[union-attr]
    assert aus_coeffs is not None
    assert aus_coeffs.a == -4.0
    assert sibling.current_a("AUS", "high") == -4.0
    assert sibling.current_a("MIA", "high") == -3.0

    # --- Orchestrator-owned rebind ---
    # Mutate the artifact on disk, then perform the EXACT rebind the
    # orchestrator runs once per tick (spec §4.2 / §12.9). Any consumer that
    # held a direct ``EmosTable`` reference instead of the holder would be
    # stale after this line.
    time.sleep(0.01)  # ensure mtime moves
    coeffs_path.write_text(json.dumps(_payload(-9.0)))
    holder.table = holder.table.check_and_reload() or holder.table

    # Post-rebind: BOTH consumers see the new coefficients (a=-9.0 for AUS,
    # a=-8.0 for MIA) via the shared holder. This is the load-bearing
    # assertion — failure here means weather_edge silently runs on the
    # pre-swap table, which is exactly the bug v1.3 MINOR #3 warned about.
    new_aus = weather.emos_table.get(city="AUS", var="high", lead=0)  # type: ignore[union-attr]
    assert new_aus is not None
    assert new_aus.a == -9.0
    assert sibling.current_a("AUS", "high") == -9.0
    assert sibling.current_a("MIA", "high") == -8.0

    # And the orchestrator's holder + every consumer's holder reference
    # point at the SAME EmosTable instance — there is exactly one current
    # table at any moment.
    assert weather._emos_holder is holder
    assert sibling._emos_holder is holder
    assert weather.emos_table is sibling._emos_holder.table
    assert holder.table.artifact_hash != pre_hash


def test_weather_edge_module_does_not_call_check_and_reload() -> None:
    """Static-grep enforcement (plan §M1 completion criteria): the
    ``weather_edge.py`` module MUST NOT reference ``check_and_reload`` —
    spec §12.9 mandates that hot-reload ownership lives exclusively on the
    orchestrator after M1. Any future regression that reintroduces a
    per-strategy reload (e.g. a stale rebase) will fail this assertion.
    """
    src = Path("python/strategies/weather_edge.py").read_text()
    assert "check_and_reload" not in src, (
        "weather_edge.py must not reference check_and_reload — hot-reload "
        "ownership lives on the orchestrator (spec §4.2 / §12.9)."
    )
