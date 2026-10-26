"""Tests for the off-event-loop buffered audit writer.

The point of the class under test is *where* the write syscall runs, so the
central assertion is a thread identity, not a timing measurement.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path
from unittest import mock

import pytest

from python.analytics.audit_log import (
    EVAL_SCHEMA_VERSION,
    AuditLogger,
    AuditSinks,
    BufferedAuditLogger,
    build_sinks,
    flush_sinks,
)


def _fixed_clock(d: date):
    return lambda: d


def test_emit_writes_nothing_until_flush(tmp_path: Path) -> None:
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))

    log.emit_eval({"reason": "PASSED", "ticker": "T1"})

    assert log.buffered_count() == 1
    assert not (tmp_path / "eval.jsonl").exists()


@pytest.mark.asyncio
async def test_flush_writes_every_row_in_emit_order(tmp_path: Path) -> None:
    log = BufferedAuditLogger(
        tmp_path / "eval.jsonl",
        schema_version=3,
        clock=_fixed_clock(date(2026, 9, 4)),
    )
    for i in range(5):
        log.emit_eval({"reason": "PASSED", "ticker": f"T{i}"})

    await log.flush_async()

    lines = (tmp_path / "eval.jsonl").read_text().splitlines()
    assert [json.loads(x)["ticker"] for x in lines] == [f"T{i}" for i in range(5)]
    assert json.loads(lines[0])["schema_version"] == 3
    assert json.loads(lines[0])["reason_numeric"] == 99
    assert log.buffered_count() == 0


@pytest.mark.asyncio
async def test_writes_happen_on_a_worker_thread_not_the_event_loop(tmp_path: Path) -> None:
    """The whole point of the class: zero write syscalls on the calling coroutine."""
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))
    loop_thread = threading.get_ident()
    write_threads: list[int] = []
    real_write = os.write

    def spy_write(fd: int, data: bytes) -> int:
        write_threads.append(threading.get_ident())
        return real_write(fd, data)

    with mock.patch("python.analytics.audit_log.os.write", spy_write):
        for i in range(50):
            log.emit_eval({"reason": "PASSED", "ticker": f"T{i}"})
        assert write_threads == []
        await log.flush_async()

    assert len(write_threads) == 50
    assert loop_thread not in write_threads


@pytest.mark.asyncio
async def test_flush_on_empty_buffer_is_a_noop(tmp_path: Path) -> None:
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))

    await log.flush_async()

    assert not (tmp_path / "eval.jsonl").exists()


@pytest.mark.asyncio
async def test_flush_rotates_across_a_utc_day_boundary(tmp_path: Path) -> None:
    current = {"d": date(2026, 9, 4)}
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=lambda: current["d"])

    log.emit_eval({"reason": "PASSED", "ticker": "day1"})
    await log.flush_async()
    current["d"] = date(2026, 9, 5)
    log.emit_eval({"reason": "PASSED", "ticker": "day2"})
    await log.flush_async()

    rotated = tmp_path / "eval.jsonl.2026-09-04"
    assert json.loads(rotated.read_text().strip())["ticker"] == "day1"
    assert json.loads((tmp_path / "eval.jsonl").read_text().strip())["ticker"] == "day2"


@pytest.mark.asyncio
async def test_overflow_falls_back_to_an_inline_write(tmp_path: Path) -> None:
    """A caller that forgets to flush degrades to today's behavior, never to
    unbounded memory growth. The fallback row lands out of emit order — that
    is the accepted cost of never dropping a row."""
    log = BufferedAuditLogger(
        tmp_path / "eval.jsonl",
        clock=_fixed_clock(date(2026, 9, 4)),
        max_buffered_rows=2,
    )
    log.emit_eval({"reason": "PASSED", "ticker": "T0"})
    log.emit_eval({"reason": "PASSED", "ticker": "T1"})
    assert not (tmp_path / "eval.jsonl").exists()

    log.emit_eval({"reason": "PASSED", "ticker": "T2"})

    lines = (tmp_path / "eval.jsonl").read_text().splitlines()
    assert [json.loads(x)["ticker"] for x in lines] == ["T2"]
    assert log.buffered_count() == 2

    await log.flush_async()

    lines = (tmp_path / "eval.jsonl").read_text().splitlines()
    assert [json.loads(x)["ticker"] for x in lines] == ["T2", "T0", "T1"]


@pytest.mark.asyncio
async def test_emit_timestamp_is_stamped_at_emit_not_at_flush(tmp_path: Path) -> None:
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))

    log.emit_eval({"reason": "PASSED", "ticker": "T0", "ts": "2020-01-01T00:00:00+00:00"})
    await log.flush_async()

    row = json.loads((tmp_path / "eval.jsonl").read_text().strip())
    assert row["ts"] == "2020-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_row_is_serialized_at_flush_not_at_emit(tmp_path: Path) -> None:
    """Pins the documented contract: a row parked by reference reflects
    mutations made before the *next* flush, because serialization happens
    at drain time. Callers must not rely on post-emit mutation, but this
    test makes the actual (currently relied-upon-nowhere) behavior explicit
    rather than an undocumented accident."""
    log = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))
    row = {"reason": "PASSED", "ticker": "T0"}

    log.emit_eval(row)
    row["ticker"] = "MUTATED"
    await log.flush_async()

    written = json.loads((tmp_path / "eval.jsonl").read_text().strip())
    assert written["ticker"] == "MUTATED"


def test_build_sinks_defaults_to_the_plain_logger(tmp_path: Path) -> None:
    sinks = build_sinks(
        tmp_path,
        eval_enabled=True,
        candidate_enabled=True,
        funnel_enabled=True,
    )

    assert type(sinks.eval_log) is AuditLogger
    assert type(sinks.candidate_log) is AuditLogger
    assert type(sinks.funnel_log) is AuditLogger


def test_build_sinks_buffered_selects_the_buffered_logger(tmp_path: Path) -> None:
    sinks = build_sinks(
        tmp_path,
        eval_enabled=True,
        candidate_enabled=False,
        funnel_enabled=True,
        retention_days=35,
        buffered=True,
    )

    assert isinstance(sinks.eval_log, BufferedAuditLogger)
    assert sinks.candidate_log is None
    assert isinstance(sinks.funnel_log, BufferedAuditLogger)
    assert sinks.eval_log.schema_version == EVAL_SCHEMA_VERSION
    assert sinks.eval_log.retention_days == 35


@pytest.mark.asyncio
async def test_flush_sinks_drains_buffered_streams_and_ignores_plain_ones(
    tmp_path: Path,
) -> None:
    buffered = BufferedAuditLogger(tmp_path / "eval.jsonl", clock=_fixed_clock(date(2026, 9, 4)))
    plain = AuditLogger(tmp_path / "cand.jsonl", clock=_fixed_clock(date(2026, 9, 4)))
    sinks = AuditSinks(eval_log=buffered, candidate_log=plain, funnel_log=None)

    buffered.emit_eval({"reason": "PASSED", "ticker": "B"})
    plain.emit_candidate({"reason": "PASSED", "ticker": "P"})
    assert json.loads((tmp_path / "cand.jsonl").read_text().strip())["ticker"] == "P"
    assert not (tmp_path / "eval.jsonl").exists()

    await flush_sinks(sinks)

    assert json.loads((tmp_path / "eval.jsonl").read_text().strip())["ticker"] == "B"


@pytest.mark.asyncio
async def test_flush_sinks_tolerates_an_all_none_bundle() -> None:
    await flush_sinks(AuditSinks())


def test_plain_logger_still_writes_synchronously_for_offline_callers(tmp_path: Path) -> None:
    """The backtest, scripts and existing tests have no event loop and rely on
    `append` being durable the moment it returns. build_sinks keeps that by
    default; only an explicit buffered=True changes it."""
    sinks = build_sinks(
        tmp_path,
        eval_enabled=True,
        candidate_enabled=True,
        funnel_enabled=True,
    )
    assert sinks.eval_log is not None

    sinks.eval_log.emit_eval({"reason": "PASSED", "ticker": "SYNC"})

    row = json.loads((tmp_path / "eval_audit.jsonl").read_text().strip())
    assert row["ticker"] == "SYNC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
