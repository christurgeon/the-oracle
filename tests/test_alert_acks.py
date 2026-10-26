"""Tests for the alert-acknowledgement store (Task 6).

Two namespaces, two meanings, never conflated:
  ``seen``  — machine dedup. Codes already DISPATCHED to Telegram.
  ``acked`` — human verdict. Recorded ONLY by the /ack Telegram command
              (Task 8) via ``persist_ack_code``.

The v1 ``tripwire_plateau_seen.json`` file recorded codes dispatched to
Telegram — no human ever reviewed any of them. Migrating v1 into ``acked``
would relabel dispatch records as review records and make /acks lie to the
operator, so migration must land every v1 code in ``seen`` and none in
``acked``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from python.analytics.audit_tripwires import (
    load_acks,
    persist_ack_code,
    persist_seen_code,
)
from python.notifications.base import AlertLevel
from tests.conftest import _stub

if TYPE_CHECKING:
    from python.orchestrator import Orchestrator
    from tests.conftest import _StubNotifier


class TestAckStore:
    def test_v1_plateau_file_migrates_into_seen_not_acked(self, tmp_path: Path) -> None:
        """v1 recorded codes DISPATCHED to Telegram — no human reviewed any.

        Migrating them into `acked` would relabel dispatch records as review
        records and make /acks lie.
        """
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            json.dumps({"_schema_version": 1, "codes": ["calibrator_plateau:a:0.1-0.2"]})
        )
        seen, acked = load_acks(tmp_path)
        assert "calibrator_plateau:a:0.1-0.2" in seen
        assert acked == {}

    def test_persist_and_load_roundtrip(self, tmp_path: Path) -> None:
        persist_ack_code(
            tmp_path,
            "calibrator_artifact_stale:favorite_bias:2026-07-04",
            owner="favorite_bias",
            severity="WARNING",
            owner_live_at_ack=True,
        )
        _seen, acked = load_acks(tmp_path)
        entry = acked["calibrator_artifact_stale:favorite_bias:2026-07-04"]
        assert entry["owner"] == "favorite_bias"
        assert entry["severity"] == "WARNING"
        assert entry["owner_live_at_ack"] is True
        assert "ts" in entry

    def test_corrupt_file_degrades_to_empty(self, tmp_path: Path) -> None:
        (tmp_path / "alert_acks.json").write_text("{not json")
        assert load_acks(tmp_path) == (set(), {})

    def test_missing_file_degrades_to_empty(self, tmp_path: Path) -> None:
        assert load_acks(tmp_path) == (set(), {})

    # -- boot-sequence coverage --------------------------------------------
    # The brief's four tests above cover "no files at all" and "only a v1
    # file" (plus corruption). The remaining two required boot sequences —
    # "after the first v2 write" and "a v1 file still present alongside a v2
    # file" — are exercised below.

    def test_no_files_at_all_degrades_to_empty(self, tmp_path: Path) -> None:
        """Boot sequence: no v1 file, no v2 file."""
        assert load_acks(tmp_path) == (set(), {})

    def test_after_first_v2_write_round_trips_through_seen(self, tmp_path: Path) -> None:
        """Boot sequence: after the first v2 write (persist_seen_code)."""
        persist_seen_code(tmp_path, "calibrator_plateau:2026-05-01:0.649-0.967")
        seen, acked = load_acks(tmp_path)
        assert seen == {"calibrator_plateau:2026-05-01:0.649-0.967"}
        assert acked == {}

    def test_v1_file_present_alongside_v2_file_prefers_v2(self, tmp_path: Path) -> None:
        """Boot sequence: a v1 file still present alongside a v2 file.

        Once alert_acks.json (v2) exists it is authoritative and read
        directly — the v1 file is never consulted again, so a v1 code that
        was never migrated into v2's ``seen`` must NOT reappear, and must
        never surface in ``acked`` regardless.
        """
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            json.dumps({"_schema_version": 1, "codes": ["calibrator_plateau:old:0.1-0.2"]})
        )
        # Pre-existing v2 file, written independently of the v1 file above —
        # simulates a v2 store that predates this v1 code being written, or
        # was migrated at a prior v1->v2 transition and has since moved on.
        (tmp_path / "alert_acks.json").write_text(
            json.dumps(
                {
                    "_schema_version": 2,
                    "seen": ["calibrator_plateau:new:0.3-0.4"],
                    "acked": {},
                }
            )
        )
        seen, acked = load_acks(tmp_path)
        # v2 is authoritative once it exists: the v1-only code is not pulled in.
        assert "calibrator_plateau:old:0.1-0.2" not in seen
        assert "calibrator_plateau:new:0.3-0.4" in seen
        assert "calibrator_plateau:old:0.1-0.2" not in acked

    def test_persist_seen_code_preserves_existing_acked_entries(self, tmp_path: Path) -> None:
        """A seen-write must never clobber a prior human ack."""
        persist_ack_code(
            tmp_path,
            "calibrator_artifact_stale:favorite_bias:2026-07-04",
            owner="favorite_bias",
            severity="WARNING",
            owner_live_at_ack=True,
        )
        persist_seen_code(tmp_path, "calibrator_plateau:2026-05-01:0.649-0.967")
        seen, acked = load_acks(tmp_path)
        assert "calibrator_plateau:2026-05-01:0.649-0.967" in seen
        assert "calibrator_artifact_stale:favorite_bias:2026-07-04" in acked

    def test_persist_ack_code_preserves_existing_seen_entries(self, tmp_path: Path) -> None:
        """An ack-write must never clobber the machine-dedup set."""
        persist_seen_code(tmp_path, "calibrator_plateau:2026-05-01:0.649-0.967")
        persist_ack_code(
            tmp_path,
            "calibrator_artifact_stale:favorite_bias:2026-07-04",
            owner="favorite_bias",
            severity="WARNING",
            owner_live_at_ack=True,
        )
        seen, acked = load_acks(tmp_path)
        assert "calibrator_plateau:2026-05-01:0.649-0.967" in seen
        assert "calibrator_artifact_stale:favorite_bias:2026-07-04" in acked

    def test_unknown_schema_version_degrades_to_empty(self, tmp_path: Path) -> None:
        (tmp_path / "alert_acks.json").write_text(
            json.dumps({"_schema_version": 99, "seen": ["x"], "acked": {"y": {}}})
        )
        assert load_acks(tmp_path) == (set(), {})

    def test_non_dict_json_degrades_to_empty(self, tmp_path: Path) -> None:
        """A syntactically valid but structurally wrong file (e.g. a bare
        list) must never raise — this is the "unreadable" case the never-
        raises contract explicitly promises to cover."""
        (tmp_path / "alert_acks.json").write_text("[1, 2, 3]")
        assert load_acks(tmp_path) == (set(), {})

    def test_reacking_a_code_overwrites_prior_entry(self, tmp_path: Path) -> None:
        persist_ack_code(
            tmp_path, "c1", owner="favorite_bias", severity="WARNING", owner_live_at_ack=True
        )
        persist_ack_code(
            tmp_path, "c1", owner="favorite_bias", severity="CRITICAL", owner_live_at_ack=True
        )
        _seen, acked = load_acks(tmp_path)
        assert acked["c1"]["severity"] == "CRITICAL"

    def test_no_tmp_file_left_behind_on_success(self, tmp_path: Path) -> None:
        persist_ack_code(tmp_path, "c1", owner=None, severity="WARNING", owner_live_at_ack=False)
        assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Task 7: the dispatch-time acknowledgement rule (`Orchestrator._ack_suppresses`,
# wired into `_notify_alert`). `orch` and `_stub` come from tests/conftest.py.
#
# ROUND 1 FIX: the first version of `_ack_suppresses` returned False (never
# suppress) unconditionally whenever the ack's owner was CURRENTLY live. That
# broke the feature's primary use case: favorite_bias is the only live,
# order-placing strategy in this system, so acking
# `calibrator_artifact_stale:favorite_bias:*` — 9 of 15 alert firings in 60
# days, and the whole reason this feature was requested — did nothing.
#
# The correct semantics are TRANSITION-based, keyed on whether the owner was
# live AT ACK TIME (`owner_live_at_ack`, persisted by `persist_ack_code`)
# versus live NOW (`_owner_is_live`, evaluated at dispatch):
#
#     acked not-live -> now not-live : SUPPRESS (if severity qualifies)
#     acked not-live -> now live     : IGNORE   (judgment made with no money
#                                       on the line; must not silently carry
#                                       over once the owner starts trading)
#     acked live     -> now live     : SUPPRESS (if severity qualifies) —
#                                       the favorite_bias case; the primary
#                                       use case this feature exists for
#     acked live     -> now not-live : SUPPRESS (if severity qualifies) —
#                                       strictly safer; the alert matters
#                                       less with the owner off
#
# Only the "not-live -> live" cell drops the ack. The tests below name each
# cell explicitly (`test_transition_*`) plus keep the severity-scoping,
# invalidation, and runtime-`/enable` tests, now run against a LIVE owner
# (favorite_bias) so they exercise the primary use case rather than dodging
# it via `owner=None`.
# ---------------------------------------------------------------------------


async def test_acked_code_is_suppressed(orch: Orchestrator) -> None:
    """The primary use case: favorite_bias is live, acked while live, and
    the SAME code fires again while still live — must suppress."""
    orch._acked = {
        "c1": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": True,
        }
    }
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is False
    )


async def test_transition_not_live_to_not_live_suppresses(orch: Orchestrator) -> None:
    """Row 1: acked while not-live, still not-live now -> suppress.

    Round 2 fix: a REGISTERED-but-shadow-only owner never reaches this test's
    logic at all — `_alert_is_relevant` short-circuits `_notify_alert` to
    False on its own (`configured_to_trade is False`), before
    `_ack_suppresses` ever runs, so the original version of this test passed
    vacuously (confirmed by monkeypatching `_ack_suppresses` to always
    return False and observing no change — see round-2 report addendum).
    Fixed by using an owner that matches NO registered strategy
    (`orch._strategies = []`): `_alert_is_relevant` then fails OPEN (True,
    dispatches past the relevance gate) while `_owner_is_live` still returns
    False (no match), so `owner_live_now` is False and this genuinely
    exercises the not-live -> not-live transition cell.
    """
    orch._acked = {
        "c1": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": False,
        }
    }
    orch._strategies = []
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is False
    )


async def test_transition_not_live_to_live_ignores(orch: Orchestrator) -> None:
    """Row 2: acked while not-live, now live -> ignore (dispatch resumes).

    Relevance re-arms on un-shadowing; the ack must re-arm with it.
    """
    orch._acked = {
        "c1": {
            "owner": "weather_edge",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": False,
        }
    }
    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="weather_edge", code="c1"
        )
        is True
    )


async def test_transition_live_to_live_suppresses(orch: Orchestrator) -> None:
    """Row 3: acked while live, still live now -> suppress.

    Same shape as `test_acked_code_is_suppressed`; kept as its own named
    transition-row test per the four-row requirement, with a different code
    so the two tests can't accidentally validate each other.
    """
    orch._acked = {
        "c2": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": True,
        }
    }
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[c2] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c2"
        )
        is False
    )


async def test_transition_live_to_not_live_suppresses(orch: Orchestrator) -> None:
    """Row 4: acked while live, now not-live -> suppress (strictly safer).

    E.g. favorite_bias was live when acked, then got shadow-flagged or
    disabled — the alert still matters less, so the ack still counts.

    Round 2 fix: same vacuity as row 1 — a registered shadow-only owner
    triggers the relevance short-circuit before `_ack_suppresses` runs, so
    this test previously passed regardless of the transition logic. Fixed
    the same way: `orch._strategies = []` so `favorite_bias` matches no
    registered strategy, `_alert_is_relevant` fails open (dispatches past
    the relevance gate), and `_owner_is_live` returns False now — leaving
    `owner_live_at_ack=True` as the only thing distinguishing this from row
    1 in the transition table.
    """
    orch._acked = {
        "c1": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": True,
        }
    }
    orch._strategies = []
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is False
    )


async def test_new_fit_date_produces_a_new_code_and_fires(orch: Orchestrator) -> None:
    """Invalidation is free: the code embeds the fit_date.

    Run against the live->live row (favorite_bias, `owner_live_at_ack=True`)
    so this proves invalidation on the primary use case, not just a case the
    transition rule would have let through anyway.
    """
    orch._acked = {
        "calibrator_artifact_stale:favorite_bias:2026-07-04": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": True,
        }
    }
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[calibrator_artifact_stale:favorite_bias:2026-09-15] m",
            level=AlertLevel.WARNING,
            owner="favorite_bias",
            code="calibrator_artifact_stale:favorite_bias:2026-09-15",
        )
        is True
    )


async def test_acked_warning_does_not_suppress_a_later_critical(orch: Orchestrator) -> None:
    """Age checks escalate WARNING->CRITICAL at 2x threshold and the code
    carries no severity. Acking at 45 days must not silence the 90-day
    CRITICAL, whose purpose is to punch through inattention on a WORSENED
    condition. Run against the live->live row (favorite_bias) — the
    transition test alone would let this through either way, so this proves
    the severity gate is a second, independent check on the primary path."""
    orch._acked = {
        "c1": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": True,
        }
    }
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.CRITICAL, owner="favorite_bias", code="c1"
        )
        is True
    )


async def test_ack_is_ignored_after_runtime_enable(orch: Orchestrator) -> None:
    """/enable flips Strategy.enabled at runtime (orchestrator.py enable_strategy).

    Dynamic version of row 2: ack while the strategy is DISABLED (not live,
    so `owner_live_at_ack=False` — what an operator acking it in that state
    would actually have recorded), then enable it, then assert the SAME call
    now dispatches — no restart, no prune in between. A startup-only prune
    would miss this entirely.
    """
    s = _stub("favorite_bias", enabled=False, shadow_only=False)
    orch._strategies = [s]
    orch._acked = {
        "c1": {
            "owner": "favorite_bias",
            "severity": "WARNING",
            "ts": "t",
            "owner_live_at_ack": False,
        }
    }
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is False
    )
    s.enabled = True  # what /enable does
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is True
    )


async def test_legacy_ack_missing_owner_live_at_ack_rearms_when_owner_becomes_live(
    orch: Orchestrator,
) -> None:
    """A pre-round-1 ack entry has no `owner_live_at_ack` key at all. The safe
    default treats it as acked-while-NOT-live, so it re-arms (row 2) rather
    than silently keeping a stale ack live once real money is on the line."""
    orch._acked = {"c1": {"owner": "favorite_bias", "severity": "WARNING", "ts": "t"}}
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert (
        await orch._notify_alert(
            "[c1] m", level=AlertLevel.WARNING, owner="favorite_bias", code="c1"
        )
        is True
    )


async def test_missing_severity_fails_open_dispatches_critical(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """An ack entry with no `severity` key must not suppress anything —
    including a CRITICAL. The old default (`entry.get("severity", "CRITICAL")`)
    treated a missing key as the highest severity, muting everything; the fix
    treats an unparseable severity as no ack at all (fail OPEN)."""
    orch._acked = {"c1": {"owner": None, "ts": "t"}}
    assert await orch._notify_alert("[c1] m", level=AlertLevel.CRITICAL, code="c1") is True
    assert len(notifier.sent) == 1


async def test_lowercase_severity_fails_open_dispatches_critical(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """A lowercase `severity` value (e.g. hand-edited, or `alert.severity`
    instead of `alert.severity.name`) is not a key in `_SEV_RANK` (which is
    all-uppercase) and must not suppress a CRITICAL — the same fail-open
    requirement as a missing severity."""
    orch._acked = {"c1": {"owner": None, "severity": "warning", "ts": "t"}}
    assert await orch._notify_alert("[c1] m", level=AlertLevel.CRITICAL, code="c1") is True
    assert len(notifier.sent) == 1


async def test_owner_none_ack_suppresses_at_or_below_acked_severity(orch: Orchestrator) -> None:
    """Round 2 item 2: restore coverage for the owner=None (infrastructure)
    path through `_notify_alert` into `_ack_suppresses`. This path was the
    only one exercised by the pre-round-1 tests and lost its dedicated test
    when those were rewritten to target the favorite_bias primary case.
    `_owner_is_live(None)` is always False, so an owner=None ack always
    reaches the severity comparison — suppressing at the acked severity
    (WARNING here) and at any lower severity (INFO), never escalating past
    it."""
    orch._acked = {"c1": {"owner": None, "severity": "WARNING", "ts": "t"}}
    assert await orch._notify_alert("[c1] m", level=AlertLevel.WARNING, code="c1") is False
    assert await orch._notify_alert("[c1] m", level=AlertLevel.INFO, code="c1") is False
