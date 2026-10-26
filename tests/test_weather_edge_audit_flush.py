"""Regression tests for audit writes leaving the event loop.

docs/tech-debt.md § "Audit log writes block the event loop during scans":
generate_signals ran ~1k os.open/os.write/os.close triples inline, stalling
WebSocket reads. The assertion here is a thread identity, not a duration.
"""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from python.analytics.audit_log import (
    AuditLogger,
    AuditSinks,
    BufferedAuditLogger,
    ScanFunnel,
    build_sinks,
)
from python.strategies.weather_edge import WeatherEdgeStrategy
from tests.test_weather_edge_event_cap_integration import (
    _bare_config,
    _build_pipeline_patches,
    _make_candidate,
)
from tests.test_weather_signals import _make_market, _make_strategy


def _buffered_sinks(strategy: WeatherEdgeStrategy, tmp_path: Path) -> AuditSinks:
    """Repoint the strategy's buffered sinks at tmp_path."""
    sinks = build_sinks(
        tmp_path,
        eval_enabled=True,
        candidate_enabled=False,
        funnel_enabled=True,
        buffered=True,
    )
    strategy._audit_sinks = sinks
    return sinks


def test_constructor_builds_buffered_sinks() -> None:
    strategy = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )

    assert isinstance(strategy._audit_sinks.eval_log, BufferedAuditLogger)
    assert isinstance(strategy._audit_sinks.funnel_log, BufferedAuditLogger)


@pytest.mark.asyncio
async def test_generate_signals_never_writes_audit_rows_on_the_loop_thread(
    tmp_path: Path,
) -> None:
    strategy = WeatherEdgeStrategy(
        config=_bare_config(max_bets_per_day_per_event=None),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    _buffered_sinks(strategy, tmp_path)

    candidates_by_ticker = {
        "LO": _make_candidate("LO", effective_edge=0.10),
        "MID": _make_candidate("MID", effective_edge=0.20),
        "HI": _make_candidate("HI", effective_edge=0.30),
    }

    loop_thread = threading.get_ident()
    append_threads: list[int] = []
    real_append = AuditLogger.append

    def spy_append(self: AuditLogger, row: dict[str, Any]) -> None:
        append_threads.append(threading.get_ident())
        real_append(self, row)

    with contextlib.ExitStack() as stack:
        for p in _build_pipeline_patches(strategy, candidates_by_ticker):
            stack.enter_context(p)
        stack.enter_context(patch.object(AuditLogger, "append", spy_append))
        signals = await strategy.generate_signals()

    assert len(signals) == 3
    assert append_threads, "no audit rows were written at all — test is not exercising the path"
    assert loop_thread not in append_threads


@pytest.mark.asyncio
async def test_generate_signals_leaves_no_rows_buffered_on_return(tmp_path: Path) -> None:
    strategy = WeatherEdgeStrategy(
        config=_bare_config(max_bets_per_day_per_event=None),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    sinks = _buffered_sinks(strategy, tmp_path)

    candidates_by_ticker = {
        "LO": _make_candidate("LO", effective_edge=0.10),
        "HI": _make_candidate("HI", effective_edge=0.30),
    }

    with contextlib.ExitStack() as stack:
        for p in _build_pipeline_patches(strategy, candidates_by_ticker):
            stack.enter_context(p)
        await strategy.generate_signals()

    assert isinstance(sinks.eval_log, BufferedAuditLogger)
    assert isinstance(sinks.funnel_log, BufferedAuditLogger)
    assert sinks.eval_log.buffered_count() == 0
    assert sinks.funnel_log.buffered_count() == 0

    eval_rows = [
        json.loads(line) for line in (tmp_path / "eval_audit.jsonl").read_text().splitlines()
    ]
    assert sorted(r["ticker"] for r in eval_rows if r.get("reason") == "PASSED") == ["HI", "LO"]
    funnel_rows = (tmp_path / "scan_funnel.jsonl").read_text().splitlines()
    assert len(funnel_rows) == 1


@pytest.mark.asyncio
async def test_aclose_flushes_rows_parked_by_an_interrupted_scan(tmp_path: Path) -> None:
    strategy = WeatherEdgeStrategy(
        config=_bare_config(),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
    )
    sinks = _buffered_sinks(strategy, tmp_path)
    assert isinstance(sinks.eval_log, BufferedAuditLogger)

    sinks.eval_log.emit_eval({"reason": "PASSED", "ticker": "ORPHAN"})
    assert not (tmp_path / "eval_audit.jsonl").exists()

    await strategy.aclose()

    row = json.loads((tmp_path / "eval_audit.jsonl").read_text().strip())
    assert row["ticker"] == "ORPHAN"


@pytest.mark.asyncio
async def test_scan_never_writes_audit_rows_on_the_loop_thread(tmp_path: Path) -> None:
    """scan()'s own PAST_DATE/VOLUME/OPEN_INTEREST/... reject path
    (`_emit_scan_reject`) is a second blocking region, independent of
    generate_signals' Stage-1 loop covered above — pin its flush too.
    Reuses `_make_strategy`/`_make_market` from test_weather_signals,
    same fixture as `test_low_oi_filtered_at_scan` there."""
    strategy, kalshi, _weather = _make_strategy()
    _buffered_sinks(strategy, tmp_path)

    market = _make_market(open_interest=2)  # below min_open_interest=5 → OPEN_INTEREST
    kalshi.iter_all_markets.side_effect = [([market], False), ([], False)]

    loop_thread = threading.get_ident()
    append_threads: list[int] = []
    real_append = AuditLogger.append

    def spy_append(self: AuditLogger, row: dict[str, Any]) -> None:
        append_threads.append(threading.get_ident())
        real_append(self, row)

    with patch.object(AuditLogger, "append", spy_append):
        markets = await strategy.scan(scan_id="s1", funnel=ScanFunnel())

    assert markets == []
    assert append_threads, "no audit rows were written — test is not exercising the reject path"
    assert loop_thread not in append_threads

    rows = [json.loads(line) for line in (tmp_path / "eval_audit.jsonl").read_text().splitlines()]
    assert any(r.get("reason") == "OPEN_INTEREST" for r in rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
