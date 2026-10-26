"""Tests for scripts/_vps.py helpers."""
from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts._vps import (
    VPS_AUDIT_FILES,
    SshFailure,
    _rotated_name,
    pull_rotated_audit_files,
    require_vps_host,
    ssh_command,
)


def test_require_vps_host_exits_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts._vps.VPS_HOST", "")
    with pytest.raises(SystemExit, match="ORACLE_VPS_HOST"):
        require_vps_host()


def test_pull_rotated_audit_files_runs_scp_for_each_stream(tmp_path: Path) -> None:
    target_date = date(2026, 5, 5)

    with patch("scripts._vps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        warnings = pull_rotated_audit_files(target_date, tmp_path)

    # On all-success path, return value should be empty
    assert warnings == []

    # 6 streams x 2 files (active + rotated) = 12 scp calls. candidate_audit.jsonl
    # is pulled for analytics even though it is excluded from
    # audit_tripwires.STREAM_NAMES (see _vps.py comment for rationale).
    # forecasts_raw.jsonl and phase0_depth_probes.jsonl use a
    # dash-before-date rotation convention (ForecastsRawWriter); all other
    # streams (incl. favorite_bias_audit.jsonl) use dot-suffix.
    # See python/utils/jsonl.py:146-148.
    assert mock_run.call_count == 2 * len(VPS_AUDIT_FILES)

    invoked_paths = [call.args[0][1] for call in mock_run.call_args_list]
    assert "op@vps.invalid:/opt/oracle/state/eval_audit.jsonl" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/eval_audit.jsonl.2026-05-05" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/candidate_audit.jsonl" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/candidate_audit.jsonl.2026-05-05" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/scan_funnel.jsonl" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/scan_funnel.jsonl.2026-05-05" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/forecasts_raw.jsonl" in invoked_paths
    assert "op@vps.invalid:/opt/oracle/state/forecasts_raw-2026-05-05.jsonl" in invoked_paths
    # And NOT the dotted form, which would silently fail on every monthly refit
    assert "op@vps.invalid:/opt/oracle/state/forecasts_raw.jsonl.2026-05-05" not in invoked_paths
    # Phase 0 intraday depth-probe sink — same writer, same rotation pattern.
    assert "op@vps.invalid:/opt/oracle/state/phase0_depth_probes.jsonl" in invoked_paths
    assert (
        "op@vps.invalid:/opt/oracle/state/phase0_depth_probes-2026-05-05.jsonl"
        in invoked_paths
    )
    assert (
        "op@vps.invalid:/opt/oracle/state/phase0_depth_probes.jsonl.2026-05-05"
        not in invoked_paths
    )
    # favorite_bias_audit.jsonl — no rotation (single growing file, O_APPEND),
    # so dot-suffix rotated sibling is attempted and harmlessly skipped.
    assert "op@vps.invalid:/opt/oracle/state/favorite_bias_audit.jsonl" in invoked_paths
    assert (
        "op@vps.invalid:/opt/oracle/state/favorite_bias_audit.jsonl.2026-05-05"
        in invoked_paths
    )


def test_pull_rotated_audit_files_records_missing_rotated_in_warnings(tmp_path: Path) -> None:
    target_date = date(2026, 5, 5)
    rotated_markers = (
        ".jsonl.2026-05-05",          # eval_audit/candidate_audit/scan_funnel rotation
        "forecasts_raw-2026-05-05",   # forecasts_raw rotation
        "phase0_depth_probes-2026-05-05",  # Phase 0 sink rotation
    )

    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stderr = "scp: No such file or directory"
        class OK:
            returncode = 0
            stderr = ""
        # Active files succeed, rotated siblings fail (simulates pruned data)
        return R() if any(m in cmd[1] for m in rotated_markers) else OK()

    with patch("scripts._vps.subprocess.run", side_effect=fake_run):
        warnings = pull_rotated_audit_files(target_date, tmp_path)

    # Six rotated siblings missing (eval, candidate, scan_funnel,
    # forecasts_raw, phase0_depth_probes, favorite_bias_audit).
    assert len(warnings) == 6
    assert any("eval_audit.jsonl.2026-05-05" in w for w in warnings)
    assert any("candidate_audit.jsonl.2026-05-05" in w for w in warnings)
    assert any("scan_funnel.jsonl.2026-05-05" in w for w in warnings)
    assert any("forecasts_raw-2026-05-05.jsonl" in w for w in warnings)
    assert any("phase0_depth_probes-2026-05-05.jsonl" in w for w in warnings)
    assert any("favorite_bias_audit.jsonl.2026-05-05" in w for w in warnings)


class TestRotatedName:
    """The naming convention lives in ``_rotated_name``; lock it down so a
    future stream addition can't silently fall through to the wrong format.
    See python/utils/jsonl.py:ForecastsRawWriter for the forecasts_raw writer
    contract."""

    def test_dot_suffix_for_audit_logs(self):
        assert _rotated_name("eval_audit.jsonl", "2026-05-05") == "eval_audit.jsonl.2026-05-05"
        assert _rotated_name("candidate_audit.jsonl", "2026-05-05") == "candidate_audit.jsonl.2026-05-05"
        assert _rotated_name("scan_funnel.jsonl", "2026-05-05") == "scan_funnel.jsonl.2026-05-05"

    def test_dash_before_date_for_forecasts_raw(self):
        # ForecastsRawWriter rotates as forecasts_raw-YYYY-MM-DD.jsonl, NOT
        # forecasts_raw.jsonl.YYYY-MM-DD. Mismatch silently drops all rotated
        # siblings; downstream validators see only the active file.
        assert _rotated_name("forecasts_raw.jsonl", "2026-05-05") == "forecasts_raw-2026-05-05.jsonl"

    def test_dash_before_date_for_phase0_depth_probes(self):
        # Phase 0 sink uses the same ForecastsRawWriter rotation pattern.
        # Without this generalised branch, rotated phase0 siblings would
        # silently fall through to the dotted-suffix form and never get
        # pulled from the VPS.
        assert (
            _rotated_name("phase0_depth_probes.jsonl", "2026-05-05")
            == "phase0_depth_probes-2026-05-05.jsonl"
        )


def test_ssh_command_returns_stdout_on_success() -> None:
    with patch("scripts._vps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "hello\n"
        mock_run.return_value.stderr = ""
        result = ssh_command("echo hello")

    assert result == "hello\n"
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "ssh"
    assert cmd[1] == "op@vps.invalid"
    assert cmd[-1] == "echo hello"


def test_ssh_command_raises_on_nonzero_exit() -> None:
    with patch("scripts._vps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 255
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "ssh: connect to host ... refused"
        with pytest.raises(SshFailure) as exc:
            ssh_command("journalctl -u oracle")

    assert "refused" in str(exc.value)


def test_ssh_command_raises_on_subprocess_timeout() -> None:
    with patch(
        "scripts._vps.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30),
    ):
        with pytest.raises(SshFailure) as exc:
            ssh_command("journalctl -u oracle")

    assert "timeout" in str(exc.value).lower()
