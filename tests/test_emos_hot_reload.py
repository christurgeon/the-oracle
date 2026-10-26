from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.data.emos import EmosTable, EmosTableHolder
from python.strategies.weather_edge import WeatherEdgeStrategy


def _payload(a_value: float = -4.0):
    return {
        "fit_date": "2026-05-04", "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": a_value, "b": 1.05, "c": 15.0, "d": 0.4,
                "n_train": 127, "leads_pooled": [0, 1, 2], "ens_std_p99": 6.5,
            },
        },
    }


def test_check_and_reload_no_change_returns_none(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload()))
    table = EmosTable.load(p)
    assert table.check_and_reload() is None


def test_check_and_reload_content_change_returns_new_instance(tmp_path: Path):
    """Returns a new EmosTable; the prior instance is left intact so any
    snapshot reference held by callers remains valid."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload(-4.0)))
    table = EmosTable.load(p)
    pre_hash = table.artifact_hash
    pre_coeffs = table.get(city="AUS", var="high", lead=0)
    assert pre_coeffs is not None
    pre_a = pre_coeffs.a

    # Mutate the file
    time.sleep(0.01)
    p.write_text(json.dumps(_payload(-5.0)))

    new_table = table.check_and_reload()
    assert new_table is not None
    assert new_table is not table  # NEW instance, not in-place mutation
    assert new_table.artifact_hash != pre_hash
    new_coeffs = new_table.get(city="AUS", var="high", lead=0)
    assert new_coeffs is not None
    assert new_coeffs.a == -5.0
    # Prior table is unchanged - load-bearing for the local-snapshot pattern
    assert table.artifact_hash == pre_hash
    prior_coeffs = table.get(city="AUS", var="high", lead=0)
    assert prior_coeffs is not None
    assert prior_coeffs.a == pre_a


def test_check_and_reload_corrupt_new_artifact_returns_none(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload(-4.0)))
    table = EmosTable.load(p)
    pre_hash = table.artifact_hash
    pre_coeffs = table.get(city="AUS", var="high", lead=0)
    assert pre_coeffs is not None
    pre_a = pre_coeffs.a

    # Corrupt
    time.sleep(0.01)
    p.write_text("{not valid json")

    result = table.check_and_reload()
    assert result is None  # failure path returns None, never raises

    # Prior table preserved
    assert table.artifact_hash == pre_hash
    prior_coeffs = table.get(city="AUS", var="high", lead=0)
    assert prior_coeffs is not None
    assert prior_coeffs.a == pre_a
    assert table.is_loaded


# ---------------------------------------------------------------------------
# generate_signals integration — EMOS_HALT + local-snapshot pattern (C8)
# ---------------------------------------------------------------------------


def _build_strategy(tmp_path: Path) -> WeatherEdgeStrategy:
    """Construct a WeatherEdgeStrategy with state_dir pointing at tmp_path.
    Caller installs scan / prefetch stubs as needed."""
    strat = WeatherEdgeStrategy(
        config={},
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
    )
    # Settings stub: provides ``state_dir`` (used by EMOS_HALT path), a
    # ``risk`` namespace (used by ``_EvalConfig``), and an ``execution``
    # namespace (used by the calibrated-probability floor mode-aware gate).
    # Other Settings fields are not exercised when scan / prefetch / evaluate
    # are stubbed.
    strat._settings = SimpleNamespace(  # type: ignore[assignment]
        state_dir=str(tmp_path),
        risk=SimpleNamespace(kelly_fraction=0.25, max_single_market_pct=0.20),
        execution=SimpleNamespace(mode="shadow"),
    )
    return strat


@pytest.mark.asyncio
async def test_emos_halt_skips_shadow(tmp_path: Path):
    """When ``state/EMOS_HALT`` exists, generate_signals captures
    ``emos_table_for_scan=None`` locally, so the side-stream
    ``forecasts_raw.jsonl`` write does NOT fire even when ``self.emos_table``
    is loaded.
    """
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_payload()))
    strat = _build_strategy(tmp_path)
    # Pretend setup() ran — wire a loaded EmosTable + writer. The setter
    # now requires an existing holder, so construct one explicitly first.
    strat._emos_holder = EmosTableHolder(table=EmosTable.load(coeffs_path))
    assert strat.emos_table.is_loaded

    # Track what raw_member_sink was passed into _prefetch_forecasts.
    captured: dict[str, object] = {"raw_member_sink_passed": "UNSET"}

    async def fake_prefetch(markets, *, raw_member_sink=None):
        captured["raw_member_sink_passed"] = raw_member_sink
        return ({}, {}, {}, {})

    strat._prefetch_forecasts = fake_prefetch  # type: ignore[assignment]

    # Stub scan so we go past the markets-empty guard and reach the prefetch
    # / side-stream-write block.
    fake_market = MagicMock()
    fake_market.parsed.ticker = "FAKE-TICKER"
    fake_market.parsed.city = "AUS"
    fake_market.parsed.forecast_date = None
    fake_market.parsed.daily_variable = None
    # Empty event_ticker => BracketSumAnalyzer.group_by_event skips it,
    # avoiding the MagicMock numeric-comparison crash in compute_metrics.
    fake_market.market = {"event_ticker": ""}
    strat.scan = AsyncMock(return_value=[fake_market])  # type: ignore[assignment]

    # Replace forecasts_raw_writer with a recording fake.
    write_calls: list[list[dict]] = []

    class _FakeWriter:
        def write_batch(self, rows: list[dict]) -> None:
            write_calls.append(rows)

    strat._forecasts_raw_writer = _FakeWriter()  # type: ignore[assignment]

    # Touch EMOS_HALT BEFORE the scan.
    halt_path = tmp_path / "EMOS_HALT"
    halt_path.touch()

    # The bracket_metrics import + per-market loop will fail because the
    # market dict is a MagicMock. Catch only that — the EMOS_HALT logic runs
    # before, and the side-stream write happens before, the bracket loop.
    try:
        await strat.generate_signals()
    except Exception:
        pass

    # Under EMOS_HALT, _prefetch_forecasts must have been called with sink=None
    assert captured["raw_member_sink_passed"] is None
    # And no forecasts_raw rows were ever written
    assert write_calls == []


@pytest.mark.asyncio
async def test_two_scans_with_orchestrator_reload_have_disjoint_artifact_hashes(
    tmp_path: Path,
):
    """The local-snapshot pattern: ``emos_table_for_scan`` is captured once at
    the top of ``generate_signals``. A scan-1 hash is pinned for that scan,
    and an orchestrator-owned reload + holder rebind between scans surfaces a
    different hash on scan 2 — never on scan 1.

    Under M1 (spec §4.2 / §12.9) the orchestrator owns the per-tick
    hot-reload — the strategy no longer reloads inside ``generate_signals``.
    This test simulates the orchestrator's tick-site rebind explicitly to
    exercise the holder-propagation invariant.
    """
    from python.data.emos import EmosTableHolder

    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_payload(-4.0)))
    strat = _build_strategy(tmp_path)
    initial = EmosTable.load(coeffs_path)
    holder = EmosTableHolder(table=initial)
    strat._emos_holder = holder
    pre_hash = strat.emos_table.artifact_hash
    assert pre_hash is not None

    # Capture which EmosTable instance the local-snapshot pattern handed
    # to _evaluate_market for each scan invocation.
    snapshots: list[EmosTable | None] = []

    async def fake_prefetch(markets, *, raw_member_sink=None):
        return ({}, {}, {}, {})

    def fake_evaluate_market(*args, **kwargs):
        snapshots.append(kwargs.get("emos_table"))
        return None  # reject so the loop body short-circuits

    strat._prefetch_forecasts = fake_prefetch  # type: ignore[assignment]
    strat._evaluate_market = fake_evaluate_market  # type: ignore[assignment]

    fake_market = MagicMock()
    fake_market.parsed.ticker = "FAKE-TICKER"
    fake_market.parsed.city = "AUS"
    fake_market.parsed.forecast_date = None
    fake_market.parsed.daily_variable = None
    # Empty event_ticker => BracketSumAnalyzer.group_by_event skips it,
    # avoiding the MagicMock numeric-comparison crash in compute_metrics.
    fake_market.market = {"event_ticker": ""}
    strat.scan = AsyncMock(return_value=[fake_market])  # type: ignore[assignment]

    # Side-stream writer is a no-op for this test.
    strat._forecasts_raw_writer = None

    # --- Scan 1 ---
    await strat.generate_signals()
    assert len(snapshots) == 1
    scan1_snapshot = snapshots[0]
    assert scan1_snapshot is not None
    assert scan1_snapshot.artifact_hash == pre_hash

    # --- Rewrite coeffs.json on disk between scans ---
    time.sleep(0.01)  # ensure mtime moves
    coeffs_path.write_text(json.dumps(_payload(-5.0)))

    # --- Orchestrator-owned reload between scans (spec §4.2). ---
    reloaded = holder.table.check_and_reload()
    assert reloaded is not None, "swap precondition: reload must succeed"
    holder.table = reloaded

    # --- Scan 2 ---
    try:
        await strat.generate_signals()
    except Exception:
        pass

    assert len(snapshots) == 2
    scan2_snapshot = snapshots[1]
    assert scan2_snapshot is not None
    # Scan 2's snapshot reflects the rewritten artifact
    assert scan2_snapshot.artifact_hash != pre_hash
    # Scan 1's snapshot is the pre-rewrite instance — still loaded, still
    # pointing at the original hash. This is the invariant the
    # local-snapshot pattern rests on.
    assert scan1_snapshot.artifact_hash == pre_hash
    # And the two snapshots are NOT the same instance
    assert scan1_snapshot is not scan2_snapshot


# ---------------------------------------------------------------------------
# C10 — mocked-mid-loop reload: the local-snapshot pattern (R2-H9) must hold
# even when self.emos_table is swapped under the per-market loop mid-scan.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mocked_mid_loop_reload_does_not_change_within_scan_hash(
    tmp_path: Path,
):
    """``generate_signals`` captures ``emos_table_for_scan = self.emos_table``
    once at the top. Even if something mid-loop (e.g. an over-eager call to
    ``check_and_reload`` plus a disk rewrite) swaps ``self.emos_table`` to a
    different instance, the in-flight scan's per-market evaluations must
    continue to receive the ORIGINAL snapshot — proven here by asserting
    every ``_evaluate_market`` call within the single scan saw the same
    pre-swap artifact_hash, while ``self.emos_table`` is observably swapped
    after the scan finishes.
    """
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_payload(-4.0)))
    strat = _build_strategy(tmp_path)
    # The setter requires an existing holder; build one before assignment.
    strat._emos_holder = EmosTableHolder(table=EmosTable.load(coeffs_path))
    pre_hash = strat.emos_table.artifact_hash
    assert pre_hash is not None

    # Two markets so the per-market loop runs >1 iteration. The mid-loop
    # rewrite + swap fires after the FIRST market is processed.
    def _fake_market(ticker: str) -> MagicMock:
        m = MagicMock()
        m.parsed.ticker = ticker
        m.parsed.city = "AUS"
        m.parsed.forecast_date = None
        m.parsed.daily_variable = None
        # Empty event_ticker => BracketSumAnalyzer.group_by_event skips it,
        # avoiding the MagicMock numeric-comparison crash in compute_metrics.
        m.market = {"event_ticker": ""}
        return m

    fake_markets = [_fake_market("FAKE-1"), _fake_market("FAKE-2")]
    strat.scan = AsyncMock(return_value=fake_markets)  # type: ignore[assignment]

    async def fake_prefetch(markets, *, raw_member_sink=None):
        return ({}, {}, {}, {})

    strat._prefetch_forecasts = fake_prefetch  # type: ignore[assignment]
    strat._forecasts_raw_writer = None

    # Wrap _evaluate_market: record the emos_table kwarg's hash on every
    # call. After the first call, simulate an "over-eager" mid-loop reload:
    # rewrite coeffs.json on disk, then forcibly invoke
    # ``self.emos_table.check_and_reload()`` and assign the result onto
    # ``strat.emos_table``. The local snapshot held by the scan's closure
    # MUST be unaffected.
    seen_hashes: list[str | None] = []
    triggered_swap = {"done": False}

    def fake_evaluate_market(*args, **kwargs):
        emos_table = kwargs.get("emos_table")
        seen_hashes.append(emos_table.artifact_hash if emos_table is not None else None)
        if not triggered_swap["done"]:
            # Rewrite on disk to ensure check_and_reload sees a different signature.
            time.sleep(0.01)
            coeffs_path.write_text(json.dumps(_payload(-9.0)))
            new_table = strat.emos_table.check_and_reload()  # type: ignore[union-attr]
            assert new_table is not None, "swap precondition: reload must succeed"
            strat.emos_table = new_table  # mutate self under the loop
            triggered_swap["done"] = True
        return None  # reject so the rest of the loop short-circuits

    strat._evaluate_market = fake_evaluate_market  # type: ignore[assignment]

    await strat.generate_signals()

    # Sanity: both markets evaluated.
    assert len(seen_hashes) == 2
    # Both calls saw the SAME (pre-swap) artifact hash, even though the
    # swap fired between them. This is the R2-H9 invariant.
    assert seen_hashes[0] == pre_hash
    assert seen_hashes[1] == pre_hash
    # Sanity: the swap actually happened — ``self.emos_table`` now points at
    # a DIFFERENT instance with a different hash.
    assert triggered_swap["done"]
    assert strat.emos_table is not None
    assert strat.emos_table.artifact_hash != pre_hash
