"""Tests for the daily UTC rotation + retention policy in audit_log.

The lazy-rotation contract: rotation only fires when the *first* write
after the UTC-date boundary lands. An idle process never rotates (and
never prunes). These tests inject a clock so the boundary can be
simulated without waiting a day.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from python.analytics.audit_log import (
    CANDIDATE_SCHEMA_VERSION,
    DEFAULT_RETENTION_DAYS,
    EVAL_SCHEMA_VERSION,
    FUNNEL_SCHEMA_VERSION,
    AuditLogger,
    build_sinks,
)


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestLazyRotation:
    def test_first_write_seeds_boundary_without_rotating(self, tmp_path: Path) -> None:
        """Day-seed on first append; no existing file, so nothing to rotate."""
        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: date(2026, 4, 20))
        log.append({"x": 1})
        # Only the active file exists — no rotated sibling yet.
        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert siblings == ["eval.jsonl"]

    def test_rotates_on_date_advance(self, tmp_path: Path) -> None:
        current = {"d": date(2026, 4, 20)}
        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: current["d"])
        log.append({"row": "day-one"})

        # Advance one day → next append triggers rotation.
        current["d"] = date(2026, 4, 21)
        log.append({"row": "day-two"})

        rotated = tmp_path / "eval.jsonl.2026-04-20"
        assert rotated.exists(), "expected rotated sibling for 2026-04-20"
        assert (tmp_path / "eval.jsonl").exists()

        # Day-one row lives in the rotated file, day-two in the active one.
        assert [r["row"] for r in _read_lines(rotated)] == ["day-one"]
        assert [r["row"] for r in _read_lines(tmp_path / "eval.jsonl")] == ["day-two"]

    def test_multiple_writes_same_day_do_not_rotate(self, tmp_path: Path) -> None:
        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: date(2026, 4, 20))
        for i in range(5):
            log.append({"i": i})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["eval.jsonl"]
        assert len(_read_lines(tmp_path / "eval.jsonl")) == 5

    def test_backwards_clock_does_not_rotate(self, tmp_path: Path) -> None:
        """Clock skew that goes *backwards* must not trigger a rotation.

        Guards against NTP jitter flipping the date and causing us to
        rename today's file under yesterday's date — which would then
        collide with the real rotation at midnight.
        """
        current = {"d": date(2026, 4, 21)}
        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: current["d"])
        log.append({"row": "1"})
        current["d"] = date(2026, 4, 20)  # clock went backwards
        log.append({"row": "2"})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["eval.jsonl"]


class TestRotationCollision:
    def test_rotation_collision_appends_numeric_suffix(self, tmp_path: Path) -> None:
        """If ``{path}.{date}`` already exists, fall back to ``.{date}.1``."""
        active = tmp_path / "eval.jsonl"
        collision = tmp_path / "eval.jsonl.2026-04-20"
        collision.write_text('{"prior":true}\n')

        current = {"d": date(2026, 4, 20)}
        log = AuditLogger(active, clock=lambda: current["d"])
        log.append({"row": "day-one"})
        current["d"] = date(2026, 4, 21)
        log.append({"row": "day-two"})

        # Collision target preserved, new rotated file gets ``.1``.
        assert collision.exists()
        assert _read_lines(collision) == [{"prior": True}]
        fallback = tmp_path / "eval.jsonl.2026-04-20.1"
        assert fallback.exists()
        assert [r["row"] for r in _read_lines(fallback)] == ["day-one"]


class TestRetentionPruning:
    def test_prunes_files_older_than_retention_window(self, tmp_path: Path) -> None:
        # Seed three rotated siblings: one fresh, one at the edge, one stale.
        (tmp_path / "eval.jsonl.2026-04-19").write_text("fresh\n")
        (tmp_path / "eval.jsonl.2026-04-06").write_text("edge\n")  # 14 days old
        (tmp_path / "eval.jsonl.2026-04-05").write_text("stale\n")  # 15 days old
        (tmp_path / "eval.jsonl.unrelated-artifact").write_text("keep\n")

        current = {"d": date(2026, 4, 19)}
        log = AuditLogger(
            tmp_path / "eval.jsonl",
            retention_days=14,
            clock=lambda: current["d"],
        )
        log.append({"row": "seed"})  # seeds the boundary
        current["d"] = date(2026, 4, 20)
        log.append({"row": "rotate"})  # triggers rotation + prune

        # cutoff = 2026-04-20 - 14 days = 2026-04-06 → strictly older drops.
        assert (tmp_path / "eval.jsonl.2026-04-19").exists()
        assert (tmp_path / "eval.jsonl.2026-04-06").exists()
        assert not (tmp_path / "eval.jsonl.2026-04-05").exists()
        # Files without a parseable date suffix are untouched.
        assert (tmp_path / "eval.jsonl.unrelated-artifact").exists()

    def test_prunes_collision_suffixed_files(self, tmp_path: Path) -> None:
        (tmp_path / "eval.jsonl.2026-04-19.1").write_text("fresh\n")
        (tmp_path / "eval.jsonl.2026-04-05.1").write_text("stale\n")

        current = {"d": date(2026, 4, 19)}
        log = AuditLogger(
            tmp_path / "eval.jsonl",
            retention_days=14,
            clock=lambda: current["d"],
        )
        log.append({"row": "seed"})
        current["d"] = date(2026, 4, 20)
        log.append({"row": "rotate"})

        assert (tmp_path / "eval.jsonl.2026-04-19.1").exists()
        assert not (tmp_path / "eval.jsonl.2026-04-05.1").exists()

    def test_non_matching_prefix_untouched(self, tmp_path: Path) -> None:
        """A sibling with a different base name must not be pruned."""
        (tmp_path / "other.jsonl.2024-01-01").write_text("x\n")
        current = {"d": date(2026, 4, 19)}
        log = AuditLogger(
            tmp_path / "eval.jsonl",
            retention_days=7,
            clock=lambda: current["d"],
        )
        log.append({"row": "1"})
        current["d"] = date(2026, 4, 20)
        log.append({"row": "2"})
        assert (tmp_path / "other.jsonl.2024-01-01").exists()


class TestRotateNow:
    def test_manual_rotate(self, tmp_path: Path) -> None:
        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: date(2026, 4, 20))
        log.append({"row": "x"})
        log.rotate_now(out_date=date(2026, 4, 20))
        assert (tmp_path / "eval.jsonl.2026-04-20").exists()
        assert not (tmp_path / "eval.jsonl").exists()


class TestPerStreamSchemaVersion:
    def test_build_sinks_stamps_distinct_schema_versions(self, tmp_path: Path) -> None:
        sinks = build_sinks(
            tmp_path,
            eval_enabled=True,
            candidate_enabled=True,
            funnel_enabled=True,
        )
        assert sinks.eval_log is not None
        assert sinks.candidate_log is not None
        assert sinks.funnel_log is not None
        assert sinks.eval_log.schema_version == EVAL_SCHEMA_VERSION
        assert sinks.candidate_log.schema_version == CANDIDATE_SCHEMA_VERSION
        assert sinks.funnel_log.schema_version == FUNNEL_SCHEMA_VERSION

    def test_append_stamps_per_stream_version(self, tmp_path: Path) -> None:
        log = AuditLogger(tmp_path / "eval.jsonl", schema_version=7)
        log.append({"row": "x"})
        [written] = _read_lines(tmp_path / "eval.jsonl")
        assert written["schema_version"] == 7

    def test_append_preserves_caller_schema_override(self, tmp_path: Path) -> None:
        """Callers may override schema_version (e.g., legacy migration writer)."""
        log = AuditLogger(tmp_path / "eval.jsonl", schema_version=1)
        log.append({"row": "x", "schema_version": 99})
        [written] = _read_lines(tmp_path / "eval.jsonl")
        assert written["schema_version"] == 99


class TestBuildSinks:
    def test_retention_days_threaded_through(self, tmp_path: Path) -> None:
        sinks = build_sinks(
            tmp_path,
            eval_enabled=True,
            candidate_enabled=True,
            funnel_enabled=True,
            retention_days=30,
        )
        for lg in (sinks.eval_log, sinks.candidate_log, sinks.funnel_log):
            assert lg is not None
            assert lg.retention_days == 30

    def test_default_retention_matches_constant(self, tmp_path: Path) -> None:
        sinks = build_sinks(
            tmp_path,
            eval_enabled=True,
            candidate_enabled=False,
            funnel_enabled=False,
        )
        assert sinks.eval_log is not None
        assert sinks.eval_log.retention_days == DEFAULT_RETENTION_DAYS

    def test_disabled_streams_return_none(self, tmp_path: Path) -> None:
        sinks = build_sinks(
            tmp_path,
            eval_enabled=False,
            candidate_enabled=True,
            funnel_enabled=False,
        )
        assert sinks.eval_log is None
        assert sinks.candidate_log is not None
        assert sinks.funnel_log is None


class TestInputValidation:
    def test_retention_days_coerced_to_int(self, tmp_path: Path) -> None:
        log = AuditLogger(tmp_path / "eval.jsonl", retention_days=7)
        assert log.retention_days == 7
        assert isinstance(log.retention_days, int)

    def test_typeerror_on_non_json_row_is_swallowed(self, tmp_path: Path) -> None:
        """JSON ``TypeError`` from a non-serializable value is dropped, not raised."""

        class NoStr:
            def __str__(self) -> str:
                raise TypeError("unserializable")

        log = AuditLogger(tmp_path / "eval.jsonl", clock=lambda: date(2026, 4, 20))
        log.append({"row": NoStr()})  # must not raise
        if (tmp_path / "eval.jsonl").exists():
            assert _read_lines(tmp_path / "eval.jsonl") == []


def test_same_day_blocked_reason_has_numeric_code_6() -> None:
    from python.analytics.audit_log import RejectionReason, reason_numeric

    assert reason_numeric(RejectionReason.SAME_DAY_BLOCKED) == 6


def test_scan_funnel_increments_same_day_blocked() -> None:
    from python.analytics.audit_log import RejectionReason, ScanFunnel

    f = ScanFunnel()
    f.increment(RejectionReason.SAME_DAY_BLOCKED)
    assert f.n_same_day_blocked == 1


def test_emit_funnel_includes_same_day_field(tmp_path: Path) -> None:
    from python.analytics.audit_log import ScanFunnel

    log = AuditLogger(tmp_path / "scan_funnel.jsonl", schema_version=FUNNEL_SCHEMA_VERSION)
    f = ScanFunnel(n_same_day_blocked=3)
    log.emit_funnel(f, scan_id="s1")
    line = (tmp_path / "scan_funnel.jsonl").read_text().splitlines()[0]
    row = json.loads(line)
    assert row["n_same_day_blocked"] == 3
    assert row["schema_version"] == FUNNEL_SCHEMA_VERSION


def test_funnel_schema_version_bumped_to_3() -> None:
    # Bumped to 6 alongside ``n_hot_reload_paused`` (intraday EMOS MED-1
    # 2026-05-15 fix). The test name is historical — it pins the *current*
    # FUNNEL_SCHEMA_VERSION, not the version-3 milestone.
    assert FUNNEL_SCHEMA_VERSION == 6


def test_event_capped_reason_has_numeric_code_24() -> None:
    from python.analytics.audit_log import RejectionReason, reason_numeric

    assert reason_numeric(RejectionReason.EVENT_CAPPED) == 24


def test_scan_funnel_increments_event_capped() -> None:
    from python.analytics.audit_log import RejectionReason, ScanFunnel

    f = ScanFunnel()
    f.increment(RejectionReason.EVENT_CAPPED)
    f.increment(RejectionReason.EVENT_CAPPED)
    assert f.n_event_capped == 2


def test_emit_funnel_includes_event_capped_field(tmp_path: Path) -> None:
    from python.analytics.audit_log import ScanFunnel

    log = AuditLogger(tmp_path / "scan_funnel.jsonl", schema_version=FUNNEL_SCHEMA_VERSION)
    f = ScanFunnel(n_event_capped=4, n_signals_fired=7)
    log.emit_funnel(f, scan_id="s1")
    line = (tmp_path / "scan_funnel.jsonl").read_text().splitlines()[0]
    row = json.loads(line)
    assert row["n_event_capped"] == 4
    assert row["n_signals_fired"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
