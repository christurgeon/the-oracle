"""EmosTableHolder unit + regression tests (M1).

The holder is a tiny mutable container so the orchestrator's per-tick
``check_and_reload`` rebind is visible to every consumer (weather_edge +
intraday_emos_high) that holds a reference to the holder. ``EmosTable.check_and_reload``
returns a NEW instance — rebinding any single consumer's reference alone
leaves the other consumers on a stale table (v1.3 MINOR #3).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

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
        },
    }


def test_holder_field_is_mutable(tmp_path: Path) -> None:
    """Assigning a new EmosTable to ``holder.table`` reflects in subsequent reads."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload(-4.0)))
    table1 = EmosTable.load(p)
    holder = EmosTableHolder(table=table1)
    assert holder.table is table1

    # Build a second (independent) table and swap it in.
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps(_payload(-9.0)))
    table2 = EmosTable.load(p2)
    holder.table = table2
    assert holder.table is table2
    assert holder.table is not table1
    coeffs = holder.table.get(city="AUS", var="high", lead=0)
    assert coeffs is not None
    assert coeffs.a == -9.0


def test_strategies_share_holder_observe_swap(tmp_path: Path) -> None:
    """Two consumers share the same holder. After an orchestrator-style
    ``holder.table = holder.table.check_and_reload() or holder.table`` rebind,
    BOTH consumers see the NEW coefficients via ``self._emos_holder.table``.

    This is the exact bug class spec §12.9 / v1.3 MINOR #3 mandates the holder
    to eliminate: under the previous design where each consumer held a direct
    ``EmosTable`` reference, the orchestrator rebinding one reference would
    leave the others stale.
    """
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload(-4.0)))
    initial = EmosTable.load(p)
    assert initial.is_loaded
    holder = EmosTableHolder(table=initial)

    # Two synthetic consumers — each keeps a reference to the SAME holder.
    # In production these are WeatherEdgeStrategy and IntradayEmosHighStrategy;
    # the holder pattern is calibrator-agnostic so the test uses simple objects.
    class _Consumer:
        def __init__(self, h: EmosTableHolder) -> None:
            self._emos_holder = h

        def current_a(self) -> float | None:
            coeffs = self._emos_holder.table.get(city="AUS", var="high", lead=0)
            return None if coeffs is None else coeffs.a

    weather_like = _Consumer(holder)
    intraday_like = _Consumer(holder)

    # Pre-swap: both consumers see a=-4.0.
    assert weather_like.current_a() == -4.0
    assert intraday_like.current_a() == -4.0
    pre_hash = holder.table.artifact_hash

    # Simulate an orchestrator tick: mutate the artifact on disk, then perform
    # the exact rebind the orchestrator owns:
    #     holder.table = holder.table.check_and_reload() or holder.table
    time.sleep(0.01)  # ensure mtime moves
    p.write_text(json.dumps(_payload(-9.0)))
    reloaded = holder.table.check_and_reload()
    assert reloaded is not None, "swap precondition: reload must succeed"
    holder.table = reloaded

    # Post-swap: BOTH consumers see the new a=-9.0 through their shared holder
    # reference. This is the load-bearing assertion — if this fails, the
    # orchestrator-owned reload would silently leave one strategy stale.
    assert weather_like.current_a() == -9.0
    assert intraday_like.current_a() == -9.0
    assert holder.table.artifact_hash != pre_hash
    # And both consumers point at the SAME EmosTable instance.
    assert weather_like._emos_holder.table is intraday_like._emos_holder.table
    assert weather_like._emos_holder.table is reloaded


# ---------------------------------------------------------------------------
# Setter foot-gun fix (audit MEDIUM): the strategy-level emos_table setter
# must REQUIRE an existing holder and MUST refuse a None payload. This
# preserves the cross-consumer aliasing invariant — under M4/M5 weather and
# intraday share the holder, and a setter that silently creates a fresh
# holder would diverge them.
# ---------------------------------------------------------------------------


def _bare_strategy() -> WeatherEdgeStrategy:
    return WeatherEdgeStrategy(
        config={},
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
    )


def test_emos_table_setter_raises_when_no_holder() -> None:
    """No injected holder, no holder constructed by setup() — the setter
    must refuse rather than silently create a divergent fresh holder."""
    strat = _bare_strategy()
    assert strat._emos_holder is None
    with pytest.raises(RuntimeError, match="requires an injected EmosTableHolder"):
        strat.emos_table = EmosTable._disabled(None, "test-only")


def test_emos_table_setter_raises_when_value_is_none(tmp_path: Path) -> None:
    """Setting ``emos_table = None`` is forbidden — callers that need a no-op
    table must pass a disabled-sentinel instead, so the cross-consumer alias
    remains a single non-None value."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload(-4.0)))
    strat = _bare_strategy()
    strat._emos_holder = EmosTableHolder(table=EmosTable.load(p))
    with pytest.raises(RuntimeError, match="cannot be set to None"):
        strat.emos_table = None


def test_emos_table_setter_mutates_existing_holder(tmp_path: Path) -> None:
    """When a holder is present, the setter mutates ``holder.table`` (not the
    strategy's holder field). Any other consumer holding the SAME holder
    reference sees the new value."""
    p1 = tmp_path / "c1.json"
    p1.write_text(json.dumps(_payload(-4.0)))
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps(_payload(-9.0)))

    holder = EmosTableHolder(table=EmosTable.load(p1))
    strat = _bare_strategy()
    strat._emos_holder = holder

    # A second alias of the SAME holder — represents intraday_emos_high.
    aliased_holder = strat._emos_holder
    assert aliased_holder is holder

    new_table = EmosTable.load(p2)
    strat.emos_table = new_table

    # Strategy reads back the new table.
    assert strat.emos_table is new_table
    # The aliased holder sees the swap too (cross-consumer propagation).
    assert aliased_holder.table is new_table
    # The strategy's holder field was NOT swapped — the same holder is mutated.
    assert strat._emos_holder is holder
