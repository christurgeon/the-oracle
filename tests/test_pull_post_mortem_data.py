"""Tests for scripts/pull_post_mortem_data.py."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts._vps import SshFailure
from scripts.pull_post_mortem_data import (
    DateValidationError,
    SchemaVersionWarning,
    _async_main,
    aggregate_kalshi_fills,
    aggregate_local_orders,
    capture_operational_status,
    check_schema_version,
    filter_signal_logs,
    output_dir_for,
    parse_date,
    reconcile,
    validate_date,
    validate_fill_created_time,
)

_UNIT_DESC = "The Oracle — Kalshi Prediction Market Trading Bot"


def test_parse_date_iso() -> None:
    assert parse_date("2026-05-05") == date(2026, 5, 5)


def test_parse_date_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_date("not-a-date")


@pytest.mark.parametrize("offset_days", [0, 1, 30])
def test_validate_date_rejects_today_and_future(offset_days: int) -> None:
    today = datetime.now(UTC).date()
    target = today + timedelta(days=offset_days)
    with pytest.raises(DateValidationError):
        validate_date(target)


@pytest.mark.parametrize("offset_days", [1, 7, 365])
def test_validate_date_accepts_past(offset_days: int) -> None:
    today = datetime.now(UTC).date()
    target = today - timedelta(days=offset_days)
    validate_date(target)  # should not raise


def test_output_dir_for() -> None:
    repo_root = Path("/repo")
    expected = repo_root / "data" / "post_mortems" / "2026-05-05"
    assert output_dir_for(repo_root, date(2026, 5, 5)) == expected


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_filter_signal_logs_keeps_only_target_date(tmp_path: Path) -> None:
    target = date(2026, 5, 5)
    src = tmp_path / "src"
    dst = tmp_path / "dst"

    rows = [
        {"ts": "2026-05-04T23:59:00Z", "_schema_version": 1, "marker": "a"},
        {"ts": "2026-05-05T00:00:01Z", "_schema_version": 1, "marker": "b"},
        {"ts": "2026-05-05T23:59:59Z", "_schema_version": 1, "marker": "c"},
        {"ts": "2026-05-06T00:00:01Z", "_schema_version": 1, "marker": "d"},
    ]
    _write_jsonl(src / "eval_audit.jsonl", rows)

    out_path = dst / "eval_audit.jsonl"
    filter_signal_logs([src / "eval_audit.jsonl"], out_path, target)

    written = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [r["marker"] for r in written] == ["b", "c"]


def test_filter_signal_logs_dedupes_active_and_rotated(tmp_path: Path) -> None:
    target = date(2026, 5, 5)

    # Active file (post-rotation morning) has only today's row
    active = tmp_path / "active" / "eval_audit.jsonl"
    _write_jsonl(active, [
        {"ts": "2026-05-06T00:00:01Z", "_schema_version": 1, "marker": "today"},
    ])
    # Rotated sibling has yesterday's data
    rotated = tmp_path / "active" / "eval_audit.jsonl.2026-05-05"
    yesterday_rows = [
        {"ts": "2026-05-05T10:00:00Z", "_schema_version": 1, "marker": "y1"},
        {"ts": "2026-05-05T11:00:00Z", "_schema_version": 1, "marker": "y2"},
    ]
    _write_jsonl(rotated, yesterday_rows + [
        # Edge case: a row that exists in both files (happens if rotation
        # races a write — same row appears in both)
        {"ts": "2026-05-05T11:00:00Z", "_schema_version": 1, "marker": "y2"},
    ])

    out = tmp_path / "out" / "eval_audit.jsonl"
    filter_signal_logs([active, rotated], out, target)

    written = [json.loads(line) for line in out.read_text().splitlines()]
    markers = [r["marker"] for r in written]
    # Yesterday rows present, deduped (y2 once), today row excluded
    assert "y1" in markers
    assert markers.count("y2") == 1
    assert "today" not in markers


def test_filter_signal_logs_handles_missing_input(tmp_path: Path) -> None:
    target = date(2026, 5, 5)
    out = tmp_path / "out.jsonl"
    # Both inputs missing; helper writes empty output without raising.
    filter_signal_logs([tmp_path / "missing.jsonl", tmp_path / "also_missing.jsonl"], out, target)
    assert out.exists()
    assert out.read_text() == ""


def test_filter_signal_logs_skips_non_string_ts(tmp_path: Path) -> None:
    """A row with a non-string ts (e.g., epoch int) must be skipped, not crash."""
    target = date(2026, 5, 5)
    src = tmp_path / "src.jsonl"
    _write_jsonl(src, [
        {"ts": 1746403200, "_schema_version": 1, "marker": "epoch_int"},
        {"ts": "2026-05-05T10:00:00Z", "_schema_version": 1, "marker": "valid"},
    ])
    out = tmp_path / "out.jsonl"
    filter_signal_logs([src], out, target)
    written = [json.loads(line) for line in out.read_text().splitlines()]
    markers = [r["marker"] for r in written]
    assert markers == ["valid"]


def test_check_schema_version_clean(tmp_path: Path) -> None:
    src = tmp_path / "eval_audit.jsonl"
    # Production-shape row: production audit_log.py writes 'schema_version' (no underscore)
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "schema_version": 1}])
    warning = check_schema_version(src, "eval_audit")
    assert warning is None


def test_check_schema_version_mismatch_returns_warning(tmp_path: Path) -> None:
    src = tmp_path / "eval_audit.jsonl"
    # Production-shape row with mismatched version
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "schema_version": 99}])
    warning = check_schema_version(src, "eval_audit")
    assert isinstance(warning, SchemaVersionWarning)
    assert warning.stream == "eval_audit"
    assert warning.expected == 1
    assert warning.actual == 99


def test_check_schema_version_funnel_tracks_production_constant(tmp_path: Path) -> None:
    """The consumer's expected scan_funnel version must equal the production writer's
    FUNNEL_SCHEMA_VERSION — hardcoding it drifted (V3 vs prod V6) and warned every run."""
    from python.analytics.audit_log import FUNNEL_SCHEMA_VERSION

    src = tmp_path / "scan_funnel.jsonl"
    # A row at the live production version must NOT warn.
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "schema_version": FUNNEL_SCHEMA_VERSION}])
    assert check_schema_version(src, "scan_funnel") is None


def test_check_schema_version_funnel_warns_on_real_mismatch(tmp_path: Path) -> None:
    from python.analytics.audit_log import FUNNEL_SCHEMA_VERSION

    src = tmp_path / "scan_funnel.jsonl"
    stale = FUNNEL_SCHEMA_VERSION - 1
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "schema_version": stale}])
    warning = check_schema_version(src, "scan_funnel")
    assert isinstance(warning, SchemaVersionWarning)
    assert warning.actual == stale
    assert warning.expected == FUNNEL_SCHEMA_VERSION


def test_check_schema_version_reads_production_key_not_underscored(tmp_path: Path) -> None:
    """Regression: production audit_log.py writes 'schema_version', not '_schema_version'."""
    src = tmp_path / "eval_audit.jsonl"
    # Production-shape row (no leading underscore on schema_version)
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "schema_version": 99}])
    warning = check_schema_version(src, "eval_audit")
    assert isinstance(warning, SchemaVersionWarning)
    assert warning.actual == 99
    assert warning.expected == 1


def test_check_schema_version_ignores_underscored_key(tmp_path: Path) -> None:
    """Underscored '_schema_version' is NOT the production key; should be ignored."""
    src = tmp_path / "eval_audit.jsonl"
    _write_jsonl(src, [{"ts": "2026-05-05T00:00:00Z", "_schema_version": 99}])
    # No real 'schema_version' field → no mismatch reported
    warning = check_schema_version(src, "eval_audit")
    assert warning is None


def test_check_schema_version_empty_file(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("")
    # Empty isn't a schema mismatch (could just be no rows for the date)
    assert check_schema_version(src, "eval_audit") is None


def test_operational_status_nominal(tmp_path: Path) -> None:
    target = date(2026, 5, 5)

    # journalctl shows oracle ran continuously, no kill-switch logs,
    # no deploys
    journal_unit = (
        f"May 05 00:00:01 host systemd[1]: Started oracle-bot.service - {_UNIT_DESC}.\n"
        "May 05 00:00:05 host oracle[123]: orchestrator started\n"
        "May 05 23:59:59 host oracle[123]: tick complete\n"
    )
    journal_kill = ""  # no kill-switch lines
    git_log = ""  # no deploys

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        is_unit_query = "-u oracle-bot.service" in cmd and "-g 'kill switch'" not in cmd
        if is_unit_query and "git log" not in cmd:
            return journal_unit
        if "-g 'kill switch'" in cmd:
            return journal_kill
        if "git log" in cmd:
            return git_log
        raise AssertionError(f"unexpected ssh: {cmd}")

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    assert status["summary"] == "nominal"
    assert status["unit_down_intervals"] == []
    assert status["kill_switch_intervals"] == []
    assert status["deploys"] == []
    assert status["ssh_failure"] is None


def test_operational_status_kill_switch(tmp_path: Path) -> None:
    target = date(2026, 5, 5)

    journal_kill = (
        "May 05 14:00:00 host oracle-bot[123]: Kill switch activated via Telegram\n"
        "May 05 16:00:00 host oracle-bot[123]: Kill switch deactivated via Telegram\n"
    )

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "-g 'kill switch'" in cmd:
            return journal_kill
        return ""  # nominal unit, no deploys

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    assert status["summary"] == "kill-switch"
    assert len(status["kill_switch_intervals"]) == 1


def test_operational_status_mixed(tmp_path: Path) -> None:
    target = date(2026, 5, 5)

    journal_kill = "May 05 14:00:00 host oracle-bot[123]: Kill switch activated via Telegram\n"
    git_log = "abc1234 deploy: hotfix\n"

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "-g 'kill switch'" in cmd:
            return journal_kill
        if "git log" in cmd:
            return git_log
        return ""

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    assert status["summary"] == "mixed"


def test_operational_status_ssh_failure_is_unknown(tmp_path: Path) -> None:
    target = date(2026, 5, 5)

    with patch(
        "scripts.pull_post_mortem_data.ssh_command",
        side_effect=SshFailure("ssh: connect refused"),
    ):
        status = capture_operational_status(target)

    assert status["summary"] == "unknown"
    assert status["ssh_failure"] == "ssh: connect refused"


def test_operational_status_no_heartbeat_field(tmp_path: Path) -> None:
    """Heartbeat is unfillable from local inspection (file is overwritten in
    place every tick); spec drops it explicitly."""
    target = date(2026, 5, 5)
    with patch("scripts.pull_post_mortem_data.ssh_command", return_value=""):
        status = capture_operational_status(target)

    assert "heartbeat" not in status
    assert "heartbeat_present" not in status


def test_operational_status_ignores_unrelated_failed_lines(tmp_path: Path) -> None:
    """Kernel/other-service 'Failed' lines must not flip summary to unit-down."""
    target = date(2026, 5, 5)

    journal_unit = (
        "May 05 00:00:01 host kernel: Failed to find file in /var/log\n"
        "May 05 00:00:02 host other.service: Failed with result 'exit-code'\n"
        "May 05 12:00:00 host oracle[123]: tick complete\n"
    )

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "-g 'kill switch'" in cmd:
            return ""
        if "git log" in cmd:
            return ""
        return journal_unit

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    # No oracle-bot.service Started/Stopped/Failed lines, so no unit-down intervals
    assert status["summary"] == "nominal"
    assert status["unit_down_intervals"] == []


def _journal_status(journal_unit: str) -> dict[str, object]:
    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "-g 'kill switch'" in cmd or "git log" in cmd:
            return ""
        return journal_unit

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        return capture_operational_status(date(2026, 5, 5))


def test_operational_status_queries_bot_unit() -> None:
    captured: list[str] = []

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        captured.append(cmd)
        return ""

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        capture_operational_status(date(2026, 5, 5))

    journal_cmds = [c for c in captured if c.startswith("journalctl")]
    assert len(journal_cmds) == 2
    assert all("-u oracle-bot.service " in c for c in journal_cmds)


def test_operational_status_detects_restart() -> None:
    status = _journal_status(
        f"May 05 16:23:33 host systemd[1]: Stopping oracle-bot.service - {_UNIT_DESC}...\n"
        "May 05 16:23:37 host systemd[1]: oracle-bot.service: Deactivated successfully.\n"
        f"May 05 16:23:37 host systemd[1]: Stopped oracle-bot.service - {_UNIT_DESC}.\n"
        f"May 05 16:23:37 host systemd[1]: Starting oracle-bot.service - {_UNIT_DESC}...\n"
        f"May 05 16:23:39 host systemd[1]: Started oracle-bot.service - {_UNIT_DESC}.\n"
    )

    assert status["summary"] == "unit-down"
    assert status["unit_down_intervals"] == [
        {"start": "May 05 16:23:37", "end": "May 05 16:23:39", "reason": "stop+start"}
    ]


def test_operational_status_detects_failed_stop() -> None:
    status = _journal_status(
        f"May 05 10:00:00 host systemd[1]: Stopping oracle-bot.service - {_UNIT_DESC}...\n"
        "May 05 10:01:30 host systemd[1]: oracle-bot.service: "
        "State 'stop-sigterm' timed out. Killing.\n"
        "May 05 10:01:30 host systemd[1]: oracle-bot.service: "
        "Main process exited, code=killed, status=9/KILL\n"
        "May 05 10:01:30 host systemd[1]: oracle-bot.service: Failed with result 'timeout'.\n"
    )

    assert status["unit_down_intervals"] == [
        {"start": "May 05 10:01:30", "end": "still-down-at-eod", "reason": "stopped-no-restart"}
    ]


def test_operational_status_ignores_failed_to_kill_notice() -> None:
    status = _journal_status(
        "May 05 10:01:30 host systemd[1]: oracle-bot.service: Failed to kill control group "
        "/system.slice/oracle-bot.service, ignoring: Invalid argument\n"
    )

    assert status["summary"] == "nominal"
    assert status["unit_down_intervals"] == []


def _bot_line(hms: str, level: str, msg: str) -> str:
    return f"May 05 {hms} host oracle-bot[123]: 2026-05-05T{hms}.000000Z [{level:<9}] {msg}\n"


def _kill_status(kill_log: str) -> dict[str, Any]:
    def fake_ssh(cmd: str, **kwargs: object) -> str:
        return kill_log if "-g 'kill switch'" in cmd else ""

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        return capture_operational_status(date(2026, 5, 5))


_STOP_FILE_ON = "KILL SWITCH ACTIVATED: state/STOP exists"
_STOP_FILE_OFF = "KILL SWITCH cleared: state/STOP no longer exists"
_TELEGRAM_ON = "Kill switch activated via Telegram"
_TELEGRAM_OFF = "Kill switch deactivated via Telegram"
_TICK_HALT = "KILL SWITCH — halting all trading and cancelling orders"
_TICK_FAILURE_ON = "20 consecutive tick failures — activating kill switch"
_STARTUP_ACTIVE = "Kill switch ACTIVE at startup"
_STARTUP_INACTIVE = "Kill switch inactive at startup"


@pytest.mark.parametrize(
    ("on_msg", "off_msg"),
    [
        (_STOP_FILE_ON, _STOP_FILE_OFF),
        (_TELEGRAM_ON, _TELEGRAM_OFF),
        (_TICK_HALT, _STARTUP_INACTIVE),
        (_STARTUP_ACTIVE, _STARTUP_INACTIVE),
    ],
    ids=["stop-file", "telegram", "restart-cleared", "restart-carried-over"],
)
def test_kill_switch_activation_then_clear(on_msg: str, off_msg: str) -> None:
    status = _kill_status(
        _bot_line("14:00:00", "critical", on_msg) + _bot_line("14:05:00", "info", off_msg)
    )

    assert status["summary"] == "kill-switch"
    assert status["kill_switch_intervals"] == [
        {"start": "May 05 14:00:00", "end": "May 05 14:05:00"}
    ]


@pytest.mark.parametrize(
    "on_msg",
    [_STOP_FILE_ON, _TELEGRAM_ON, _TICK_HALT, _TICK_FAILURE_ON, _STARTUP_ACTIVE],
    ids=["stop-file", "telegram", "tick-halt", "tick-failures", "startup"],
)
def test_kill_switch_activation_without_clear_is_still_active(on_msg: str) -> None:
    status = _kill_status(_bot_line("14:00:00", "critical", on_msg))

    assert status["kill_switch_intervals"] == [
        {"start": "May 05 14:00:00", "end": "still-active-at-eod"}
    ]


@pytest.mark.parametrize(
    "off_msg",
    [_STOP_FILE_OFF, _TELEGRAM_OFF, _STARTUP_INACTIVE],
    ids=["stop-file", "telegram", "startup"],
)
def test_kill_switch_clear_without_activation_is_ignored(off_msg: str) -> None:
    status = _kill_status(_bot_line("14:05:00", "info", off_msg))

    assert status["summary"] == "nominal"
    assert status["kill_switch_intervals"] == []


def test_kill_switch_repeat_activation_keeps_first_start() -> None:
    status = _kill_status(
        _bot_line("14:00:00", "critical", _STOP_FILE_ON)
        + _bot_line("14:01:00", "critical", _TICK_HALT)
        + _bot_line("14:05:00", "info", _STOP_FILE_OFF)
    )

    assert status["kill_switch_intervals"] == [
        {"start": "May 05 14:00:00", "end": "May 05 14:05:00"}
    ]


def test_kill_switch_still_active_after_restart_keeps_first_start() -> None:
    status = _kill_status(
        _bot_line("14:00:00", "critical", _TELEGRAM_ON)
        + _bot_line("14:10:00", "warning", _STARTUP_ACTIVE)
        + _bot_line("14:10:01", "critical", _TICK_HALT)
        + _bot_line("14:30:00", "info", _STARTUP_INACTIVE)
    )

    assert status["kill_switch_intervals"] == [
        {"start": "May 05 14:00:00", "end": "May 05 14:30:00"}
    ]


def test_kill_switch_ignores_failed_to_create_stop_file() -> None:
    status = _kill_status(
        _bot_line("14:00:00", "critical", "Failed to create kill switch file state/STOP")
    )

    assert status["kill_switch_intervals"] == []


def test_validate_fill_created_time_clean() -> None:
    fill = {"order_id": "o1", "created_time": "2026-05-05T10:00:00Z"}
    cleaned, warning = validate_fill_created_time(fill)
    assert warning is None
    assert cleaned == fill


def test_validate_fill_created_time_missing() -> None:
    fill = {"order_id": "o1"}  # no created_time
    cleaned, warning = validate_fill_created_time(fill)
    assert warning == "fill o1: missing created_time"
    assert cleaned["_no_created_time"] is True


def test_validate_fill_created_time_non_string() -> None:
    fill = {"order_id": "o1", "created_time": 1700000000}  # int instead of str
    cleaned, warning = validate_fill_created_time(fill)
    assert warning is not None and "non-string" in warning
    assert cleaned["_no_created_time"] is True


def test_aggregate_single_row_order() -> None:
    rows = [{
        "order_id": "ord1",
        "timestamp": "2026-05-05T10:00:00Z",
        "fill_price": 0.42,
        "contracts": 30,
        "fees_from_api": 0.15,
    }]
    agg = aggregate_local_orders(rows)
    assert agg["ord1"]["vwap"] == pytest.approx(0.42)
    assert agg["ord1"]["total_count"] == 30
    assert agg["ord1"]["fee_total"] == pytest.approx(0.15)


def test_aggregate_two_row_order_uses_last_row_cumulative_vwap() -> None:
    """Critical test: trade log fill_price is CUMULATIVE VWAP, not per-poll.
    Helper must NOT weight-average across rows."""
    rows = [
        {
            "order_id": "ord1",
            "timestamp": "2026-05-05T10:00:00Z",
            "fill_price": 0.42,  # cum VWAP after 10 of 30
            "contracts": 10,  # delta
            "fees_from_api": 0.05,  # cumulative
        },
        {
            "order_id": "ord1",
            "timestamp": "2026-05-05T10:00:30Z",
            "fill_price": 0.45,  # cum VWAP after all 30
            "contracts": 20,  # delta
            "fees_from_api": 0.15,  # cumulative
        },
    ]
    agg = aggregate_local_orders(rows)
    # MUST be 0.45 (last-row cumulative), NOT (0.42*10 + 0.45*20)/30 = 0.44
    assert agg["ord1"]["vwap"] == pytest.approx(0.45)
    assert agg["ord1"]["total_count"] == 30  # delta sum
    assert agg["ord1"]["fee_total"] == pytest.approx(0.15)  # last-row cumulative


def test_aggregate_handles_unsorted_rows() -> None:
    """Rows may arrive out of order; we must sort by timestamp before picking last."""
    rows = [
        {"order_id": "ord1", "timestamp": "2026-05-05T10:00:30Z",
         "fill_price": 0.45, "contracts": 20, "fees_from_api": 0.15},
        {"order_id": "ord1", "timestamp": "2026-05-05T10:00:00Z",
         "fill_price": 0.42, "contracts": 10, "fees_from_api": 0.05},
    ]
    agg = aggregate_local_orders(rows)
    assert agg["ord1"]["vwap"] == pytest.approx(0.45)


def test_aggregate_marks_shadow_orders() -> None:
    """Shadow order_ids start with 'shadow-'; the aggregator should flag them."""
    rows = [
        {"order_id": "shadow-1", "timestamp": "2026-05-05T10:00:00Z",
         "fill_price": 0.50, "contracts": 5, "fees_from_api": 0.0},
        {"order_id": "ord-real", "timestamp": "2026-05-05T11:00:00Z",
         "fill_price": 0.40, "contracts": 10, "fees_from_api": 0.05},
    ]
    agg = aggregate_local_orders(rows)
    assert agg["shadow-1"]["is_shadow"] is True
    assert agg["ord-real"]["is_shadow"] is False


def test_aggregate_local_orders_skips_placement_rows() -> None:
    """A placement row + matching fill row must aggregate to one
    contract count, not two. Without is_pnl_relevant_row, if the
    placement row arrives with a later timestamp it becomes the 'last'
    row, corrupting vwap (overwritten with 0.0) and fee_total."""
    from scripts.pull_post_mortem_data import aggregate_local_orders

    rows = [
        {
            "event": "fill",
            "order_id": "OID-A",
            "market_ticker": "KXHIGHNY-26MAY10-B70.5",
            "timestamp": "2026-05-10T15:00:00+00:00",
            "side": "NO",
            "mode": "live",
            "fill_price": 0.65,
            "contracts": 12,
            "fees_from_api": 0.18,
        },
        {
            # Placement row arrives AFTER the fill (later timestamp).
            # Without the filter, it becomes the sorted-last row, so
            # vwap = 0.0 and fee_total = 0.0 — corrupted output.
            "event": "placement",
            "order_id": "OID-A",
            "market_ticker": "KXHIGHNY-26MAY10-B70.5",
            "timestamp": "2026-05-10T15:00:01+00:00",
            "side": "NO",
            "mode": "live",
            "metadata": {"strategy": "weather_edge"},
            # No fill_price / fees_from_api — placement-row schema
        },
    ]

    aggregated = aggregate_local_orders(rows)

    assert "OID-A" in aggregated
    # Must reflect the fill row only — vwap and fee_total from the fill, not the placement row
    assert aggregated["OID-A"]["total_count"] == 12
    assert aggregated["OID-A"]["vwap"] == pytest.approx(0.65)
    assert aggregated["OID-A"]["fee_total"] == pytest.approx(0.18)


def test_aggregate_kalshi_single_yes_fill() -> None:
    fills = [{
        "order_id": "ord1",
        "side": "yes",
        "yes_price_dollars": "0.42",
        "count_fp": 10,  # 10 contracts (plain count, not micros)
        "fee_cost": "0.05",
    }]
    agg = aggregate_kalshi_fills(fills)
    assert agg["ord1"]["vwap"] == pytest.approx(0.42)
    assert agg["ord1"]["total_count"] == 10
    assert agg["ord1"]["fee_total"] == pytest.approx(0.05)


def test_aggregate_kalshi_partial_fills_compute_vwap() -> None:
    fills = [
        {"order_id": "ord1", "side": "yes", "yes_price_dollars": 0.42,
         "count_fp": 10, "fee_cost": 0.05},
        {"order_id": "ord1", "side": "yes", "yes_price_dollars": 0.45,
         "count_fp": 20, "fee_cost": 0.10},
    ]
    agg = aggregate_kalshi_fills(fills)
    expected_vwap = (0.42 * 10 + 0.45 * 20) / 30
    assert agg["ord1"]["vwap"] == pytest.approx(expected_vwap)
    assert agg["ord1"]["total_count"] == 30
    assert agg["ord1"]["fee_total"] == pytest.approx(0.15)


def test_aggregate_kalshi_uses_no_price_for_no_side() -> None:
    fills = [{
        "order_id": "ord1",
        "side": "no",
        "no_price_dollars": "0.55",
        "count_fp": 10,
        "fee_cost": "0.05",
    }]
    agg = aggregate_kalshi_fills(fills)
    assert agg["ord1"]["vwap"] == pytest.approx(0.55)


def test_parse_kalshi_count_returns_plain_count() -> None:
    """Regression: Kalshi count_fp is a plain count, not micros."""
    from scripts.pull_post_mortem_data import _parse_kalshi_count
    assert _parse_kalshi_count("1.00") == 1
    assert _parse_kalshi_count("12.00") == 12
    assert _parse_kalshi_count(10) == 10
    assert _parse_kalshi_count(10.5) == 10  # rounds half to even (banker's), but close enough


def test_operational_status_no_entries_is_nominal(tmp_path: Path) -> None:
    """journalctl returns exit-1 with 'No entries' when there are no logs;
    capture_operational_status must treat this as nominal (no events), NOT unknown."""
    target = date(2026, 5, 5)

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "journalctl" in cmd:
            raise SshFailure("ssh returncode=1: -- No entries --")
        if "git log" in cmd:
            return ""
        return ""

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    assert status["summary"] == "nominal"
    assert status["ssh_failure"] is None


def test_operational_status_real_ssh_failure_is_unknown(tmp_path: Path) -> None:
    """Transport-level SSH failure (e.g., connection refused) IS unknown."""
    target = date(2026, 5, 5)

    with patch(
        "scripts.pull_post_mortem_data.ssh_command",
        side_effect=SshFailure("ssh: connect to host ... refused"),
    ):
        status = capture_operational_status(target)

    assert status["summary"] == "unknown"
    assert "refused" in status["ssh_failure"]


def _local(oid: str, vwap: float, count: int, fee: float, *, shadow: bool = False):
    return {oid: {"vwap": vwap, "total_count": count, "fee_total": fee, "is_shadow": shadow}}


def _kalshi(oid: str, vwap: float, count: int, fee: float):
    return {oid: {"vwap": vwap, "total_count": count, "fee_total": fee}}


def test_reconcile_clean() -> None:
    out = reconcile(_local("o1", 0.45, 30, 0.15), _kalshi("o1", 0.45, 30, 0.15))
    assert out["clean"] is True
    assert out["deltas"] == []


def test_reconcile_within_tolerance() -> None:
    out = reconcile(
        _local("o1", 0.450, 30, 0.150),
        _kalshi("o1", 0.4504, 30, 0.158),  # both within tolerance
    )
    assert out["clean"] is True


def test_reconcile_ghost_trade_flagged() -> None:
    out = reconcile(_local("o1", 0.45, 30, 0.15), _kalshi("o2", 0.40, 10, 0.05))
    assert out["clean"] is False
    kinds = [d["kind"] for d in out["deltas"]]
    assert "ghost_trade" in kinds
    assert "missed_log" in kinds


def test_reconcile_shadow_only_silently_skipped() -> None:
    out = reconcile(_local("shadow-1", 0.45, 30, 0.15, shadow=True), _kalshi("o1", 0.45, 30, 0.15))
    # Only Kalshi-only "missed_log" delta; shadow-* ghost is suppressed
    kinds = [d["kind"] for d in out["deltas"]]
    assert "ghost_trade" not in kinds
    assert "missed_log" in kinds


def test_reconcile_vwap_mismatch_flagged() -> None:
    out = reconcile(_local("o1", 0.450, 30, 0.15), _kalshi("o1", 0.500, 30, 0.15))
    assert out["clean"] is False
    delta = out["deltas"][0]
    assert delta["kind"] == "vwap_mismatch"
    assert delta["order_id"] == "o1"


def test_reconcile_fee_mismatch_flagged() -> None:
    out = reconcile(_local("o1", 0.45, 30, 0.10), _kalshi("o1", 0.45, 30, 0.50))
    assert out["clean"] is False
    delta = out["deltas"][0]
    assert delta["kind"] == "fee_mismatch"


def test_reconcile_count_mismatch_flagged() -> None:
    out = reconcile(_local("o1", 0.45, 25, 0.15), _kalshi("o1", 0.45, 30, 0.15))
    assert out["clean"] is False
    assert out["deltas"][0]["kind"] == "count_mismatch"


def test_no_pull_flag_with_missing_cache_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    """--no-pull mode must exit non-zero when required cache is missing."""
    import subprocess

    monkeypatch.setenv("ORACLE_POSTMORTEM_REPO_ROOT", str(tmp_path))

    target = (datetime.now(UTC).date() - timedelta(days=2)).isoformat()
    r = subprocess.run(
        ["uv", "run", "python", "scripts/pull_post_mortem_data.py",
         "--date", target, "--no-pull"],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={**__import__("os").environ, "ORACLE_POSTMORTEM_REPO_ROOT": str(tmp_path)},
    )
    assert r.returncode != 0
    assert "missing" in (r.stderr + r.stdout).lower()


def test_bundle_manifest_records_warnings(tmp_path: Path) -> None:
    from scripts.pull_post_mortem_data import write_bundle_manifest

    out_dir = tmp_path / "data" / "post_mortems" / "2026-05-05"
    out_dir.mkdir(parents=True)
    (out_dir / "fills.json").write_text("[]")

    write_bundle_manifest(
        out_dir,
        warnings=["VPS unreachable; cached signals from 2026-05-03"],
        schema_warnings=[],
    )
    bundle = json.loads((out_dir / "bundle.json").read_text())
    assert bundle["warnings"] == ["VPS unreachable; cached signals from 2026-05-03"]
    assert "fills.json" in bundle["files"]
    assert isinstance(bundle["files"]["fills.json"]["mtime"], (int, float))


def test_bundle_manifest_includes_pagination_warning(tmp_path: Path) -> None:
    """Manual verification test: pagination logic with max_pages cap."""
    from scripts.pull_post_mortem_data import write_bundle_manifest

    out_dir = tmp_path / "data" / "post_mortems" / "2026-05-05"
    out_dir.mkdir(parents=True)

    # Simulate what _async_main would do if pagination hit the cap
    warnings = [
        "Kalshi fills pagination hit MAX_PAGES=10 cap; results may be truncated"
    ]
    write_bundle_manifest(out_dir, warnings=warnings, schema_warnings=[])

    bundle = json.loads((out_dir / "bundle.json").read_text())
    assert "MAX_PAGES" in bundle["warnings"][0]


def test_bundle_manifest_includes_cached_operational_status_warning(tmp_path: Path) -> None:
    """Manual verification test: no-pull mode cache warning."""
    from scripts.pull_post_mortem_data import write_bundle_manifest

    out_dir = tmp_path / "data" / "post_mortems" / "2026-05-05"
    out_dir.mkdir(parents=True)

    warnings = [
        "operational_status loaded from cache (no-pull mode)"
    ]
    write_bundle_manifest(out_dir, warnings=warnings, schema_warnings=[])

    bundle = json.loads((out_dir / "bundle.json").read_text())
    assert any("operational_status loaded from cache" in w for w in bundle["warnings"])


# ---------------------------------------------------------------------------
# Fix 2: _required_cache_paths checks SOURCE paths not OUTPUT paths
# ---------------------------------------------------------------------------

def test_no_pull_missing_vps_audit_source_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    """--no-pull must fail when vps_audit/eval_audit.jsonl (the source) is absent.

    Previously the gate checked the derived output in data/post_mortems/<D>/signals/
    which could pass even if the source was deleted, causing silently zeroed signals.
    """
    import subprocess

    # Create the old output path (data/post_mortems/.../signals/eval_audit.jsonl)
    # but NOT the source (data/vps_audit/eval_audit.jsonl)
    out_signals = tmp_path / "data" / "post_mortems" / "2026-05-04" / "signals"
    out_signals.mkdir(parents=True)
    (out_signals / "eval_audit.jsonl").write_text("")
    # vps_trades.jsonl present but vps_audit dir is absent
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "vps_trades.jsonl").write_text("")

    r = subprocess.run(
        ["uv", "run", "python", "scripts/pull_post_mortem_data.py",
         "--date", "2026-05-04", "--no-pull"],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={**__import__("os").environ, "ORACLE_POSTMORTEM_REPO_ROOT": str(tmp_path)},
    )
    assert r.returncode != 0, "Expected non-zero exit when vps_audit source is missing"
    assert "missing" in (r.stderr + r.stdout).lower()


# ---------------------------------------------------------------------------
# Fix 3: _parse_unit_down emits "still-down-at-eod" for unrecovered events
# ---------------------------------------------------------------------------

def test_operational_status_unrecovered_unit_down(tmp_path: Path) -> None:
    """A unit-down event with no recovery before EOD must still be recorded."""
    target = date(2026, 5, 5)

    journal_unit = (
        f"May 05 00:00:01 host systemd[1]: Started oracle-bot.service - {_UNIT_DESC}.\n"
        f"May 05 11:00:00 host systemd[1]: Stopped oracle-bot.service - {_UNIT_DESC}.\n"
        # No 'Started' afterwards
    )

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        if "-g 'kill switch'" in cmd:
            return ""
        if "git log" in cmd:
            return ""
        return journal_unit

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        status = capture_operational_status(target)

    assert len(status["unit_down_intervals"]) == 1
    assert status["unit_down_intervals"][0]["end"] == "still-down-at-eod"
    assert status["unit_down_intervals"][0]["reason"] == "stopped-no-restart"
    assert status["summary"] == "unit-down"


# ---------------------------------------------------------------------------
# Fix 4: Missing vps_trades.jsonl produces skipped reconciliation (not false deltas)
# ---------------------------------------------------------------------------

def test_reconcile_skipped_when_trades_missing(tmp_path: Path, monkeypatch) -> None:
    """If vps_trades.jsonl is missing, reconciliation.json should report 'skipped:true',
    not flag every Kalshi fill as missed_log."""
    import asyncio

    monkeypatch.setenv("ORACLE_POSTMORTEM_REPO_ROOT", str(tmp_path))

    # Pre-populate the vps_audit source so --no-pull passes
    (tmp_path / "data" / "vps_audit").mkdir(parents=True)
    (tmp_path / "data" / "vps_audit" / "eval_audit.jsonl").write_text("")
    # vps_trades.jsonl intentionally absent

    out_dir = tmp_path / "data" / "post_mortems" / "2026-05-04"
    (out_dir / "signals").mkdir(parents=True)
    (out_dir / "signals" / "eval_audit.jsonl").write_text("")
    (out_dir / "fills.json").write_text(json.dumps([
        {"order_id": "kalshi-1", "side": "yes", "yes_price_dollars": 0.45,
         "count_fp": 10, "fee_cost": 0.05, "created_time": "2026-05-04T10:00:00Z"},
    ]))

    target = date(2026, 5, 4)
    asyncio.run(_async_main(target, no_pull=True))

    recon = json.loads((out_dir / "reconciliation.json").read_text())
    assert recon["skipped"] is True
    assert recon["clean"] is None
    assert recon["deltas"] == []


# ---------------------------------------------------------------------------
# Fix 5: _no_created_time fills are included in reconciliation
# ---------------------------------------------------------------------------

def test_no_created_time_fills_included_in_reconciliation(tmp_path: Path) -> None:
    """A fill flagged _no_created_time should still participate in reconciliation
    so we don't produce false-positive ghost_trade flags."""
    fills = [
        {"order_id": "ord1", "side": "yes", "yes_price_dollars": 0.45,
         "count_fp": 10, "fee_cost": 0.05, "_no_created_time": True},
    ]
    agg = aggregate_kalshi_fills(fills)
    assert "ord1" in agg
    assert agg["ord1"]["total_count"] == 10


def test_operational_status_git_log_uses_iso_datetime_boundaries(tmp_path: Path) -> None:
    """Regression: git log boundary was naked YYYY-MM-DD, which git treats as
    inclusive of the entire end day. Force ISO datetime form to make
    --until exclusive at midnight UTC."""
    target = date(2026, 5, 4)
    captured: list[str] = []

    def fake_ssh(cmd: str, **kwargs: object) -> str:
        captured.append(cmd)
        if "git log" in cmd:
            return ""  # no deploys
        return ""  # nominal journalctl

    with patch("scripts.pull_post_mortem_data.ssh_command", side_effect=fake_ssh):
        capture_operational_status(target)

    git_cmds = [c for c in captured if "git log" in c]
    assert len(git_cmds) == 1
    cmd = git_cmds[0]
    # Boundaries must be ISO datetime form, not naked dates
    assert "--since='2026-05-04T00:00:00Z'" in cmd
    assert "--until='2026-05-05T00:00:00Z'" in cmd
