"""Tests for audit-log + calibrator staleness tripwires.

These checks guard two silent-failure modes:
  1. The audit log file exists but the writer is stuck — the bot
     *thinks* it's observing, but nothing has hit disk in hours.
  2. The calibration artifact is too old — the fit drifted out from
     under the production markets.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.audit_tripwires import (
    PLATEAU_ALERT_CODE_PREFIX,
    STREAM_NAMES,
    TripwireConfig,
    _read_last_jsonl_record,
    check_active_file_staleness,
    check_bias_table_freshness,
    check_calibrator_artifact_staleness,
    check_calibrator_plateau_coverage,
    check_rotation_regression,
    load_plateau_seen_codes,
    persist_seen_code,
    run_all,
)
from python.notifications.base import AlertLevel


def _touch(path: Path, *, mtime: datetime) -> None:
    """Create ``path`` and set its mtime to the given UTC datetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))


class TestActiveFileStaleness:
    def test_fresh_file_is_silent(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(minutes=5))
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        assert alerts == []

    def test_stale_file_warns(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(hours=4))
        alerts = check_active_file_staleness(
            tmp_path,
            # 2h threshold; 4h age → above threshold but below 3× (= 6h) → WARNING.
            stale_seconds=2 * 3600.0,
            now=now,
        )
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.WARNING
        assert alerts[0].code == "audit_log_stale:eval_audit.jsonl"
        assert "4.0h" in alerts[0].message or "4.1h" in alerts[0].message

    def test_very_stale_escalates_to_critical(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(hours=12))
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,  # threshold 1h; 12h ≥ 3× threshold
            now=now,
        )
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.CRITICAL

    def test_missing_file_is_silent(self, tmp_path: Path) -> None:
        """No file == disabled stream; never alert."""
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=datetime(2026, 4, 20, 12, 0, tzinfo=UTC),
        )
        assert alerts == []

    def test_excluded_stream_does_not_alert_when_stale(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(minutes=5))
        _touch(tmp_path / "candidate_audit.jsonl", mtime=now - timedelta(hours=5))
        _touch(tmp_path / "scan_funnel.jsonl", mtime=now - timedelta(hours=5))
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        # candidate_audit.jsonl is excluded from STREAM_NAMES (activity-dependent,
        # not health-dependent); only scan_funnel.jsonl should alert here.
        assert codes == {
            "audit_log_stale:scan_funnel.jsonl",
        }


class TestReadLastJsonlRecord:
    """Helper used by the forecasts_raw staleness suppression path."""

    def test_returns_last_record(self, tmp_path: Path) -> None:
        p = tmp_path / "log.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
        assert _read_last_jsonl_record(p) == {"a": 3}

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert _read_last_jsonl_record(tmp_path / "missing.jsonl") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "log.jsonl"
        p.write_text("")
        assert _read_last_jsonl_record(p) is None

    def test_skips_trailing_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "log.jsonl"
        p.write_text('{"a": 1}\n\n\n')
        assert _read_last_jsonl_record(p) == {"a": 1}

    def test_skips_unparseable_trailing_line(self, tmp_path: Path) -> None:
        """A torn final line (process killed mid-write) must not mask the
        last *good* record. iter_jsonl already tolerates this; the
        suppression check needs the same forgiveness."""
        p = tmp_path / "log.jsonl"
        p.write_text('{"a": 1}\nincomplete{\n')
        assert _read_last_jsonl_record(p) == {"a": 1}

    def test_handles_torn_first_line_in_tail_window(self, tmp_path: Path) -> None:
        """Bytes before the tail boundary may begin mid-line. The helper
        must not parse that partial line as the record."""
        p = tmp_path / "log.jsonl"
        lines = [json.dumps({"i": i}) for i in range(100)]
        p.write_text("\n".join(lines) + "\n")
        # tail_bytes=50 lands inside a line — the first 'line' in the chunk
        # will be a fragment that fails to parse.
        assert _read_last_jsonl_record(p, tail_bytes=50) == {"i": 99}


class TestForecastsRawStalenessSuppression:
    """When ``scan_funnel.jsonl``'s most-recent row reports ``n_markets=0``,
    the ``forecasts_raw.jsonl`` writer is correctly silent (no tradeable
    universe → ``generate_signals`` short-circuits before the writer).
    Suppress the staleness alert in that window; alert otherwise."""

    @staticmethod
    def _write_funnel(path: Path, *, n_markets: int, mtime: datetime) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"scan_id": "abc", "n_markets": n_markets, "ts": mtime.isoformat()})
        path.write_text(payload + "\n")
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))

    def test_suppressed_when_universe_empty(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 16, 13, 0, tzinfo=UTC)
        _touch(tmp_path / "forecasts_raw.jsonl", mtime=now - timedelta(hours=9))
        self._write_funnel(
            tmp_path / "scan_funnel.jsonl", n_markets=0, mtime=now - timedelta(minutes=1)
        )
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        assert "audit_log_stale:forecasts_raw.jsonl" not in codes

    def test_alerts_when_universe_nonempty(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 16, 13, 0, tzinfo=UTC)
        _touch(tmp_path / "forecasts_raw.jsonl", mtime=now - timedelta(hours=9))
        self._write_funnel(
            tmp_path / "scan_funnel.jsonl", n_markets=5, mtime=now - timedelta(minutes=1)
        )
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        assert "audit_log_stale:forecasts_raw.jsonl" in codes

    def test_alerts_when_funnel_missing(self, tmp_path: Path) -> None:
        """No funnel data → assume nothing; alert (don't silently hide a
        real broken-writer signal behind a missing-file edge case)."""
        now = datetime(2026, 5, 16, 13, 0, tzinfo=UTC)
        _touch(tmp_path / "forecasts_raw.jsonl", mtime=now - timedelta(hours=9))
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        assert "audit_log_stale:forecasts_raw.jsonl" in codes

    def test_alerts_when_funnel_unparseable(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 16, 13, 0, tzinfo=UTC)
        _touch(tmp_path / "forecasts_raw.jsonl", mtime=now - timedelta(hours=9))
        (tmp_path / "scan_funnel.jsonl").write_text("this is not json\n")
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        assert "audit_log_stale:forecasts_raw.jsonl" in codes

    def test_eval_audit_staleness_not_suppressed_by_empty_universe(self, tmp_path: Path) -> None:
        """Suppression is scoped to forecasts_raw only — eval_audit still
        emits scan-reject rows even when n_markets=0, so its staleness
        always means real failure."""
        now = datetime(2026, 5, 16, 13, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(hours=9))
        self._write_funnel(
            tmp_path / "scan_funnel.jsonl", n_markets=0, mtime=now - timedelta(minutes=1)
        )
        alerts = check_active_file_staleness(
            tmp_path,
            stale_seconds=3600.0,
            now=now,
        )
        codes = {a.code for a in alerts}
        assert "audit_log_stale:eval_audit.jsonl" in codes


class TestRotationRegression:
    def test_recent_active_no_alert(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(hours=6))
        assert check_rotation_regression(tmp_path, now=now) == []

    def test_rotated_sibling_suppresses_alert(self, tmp_path: Path) -> None:
        """If a rotated sibling exists, rotation is working; no alert."""
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(days=3))
        _touch(tmp_path / "eval_audit.jsonl.2026-04-18", mtime=now - timedelta(days=2))
        assert check_rotation_regression(tmp_path, now=now) == []

    def test_old_active_no_rotation_warns(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(days=3))
        alerts = check_rotation_regression(tmp_path, now=now)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.WARNING
        assert "Daily rotation" in alerts[0].message

    def test_ignores_unparseable_siblings(self, tmp_path: Path) -> None:
        """A sibling without a parseable date suffix must not suppress the alert."""
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(days=3))
        _touch(tmp_path / "eval_audit.jsonl.bak", mtime=now - timedelta(days=1))
        alerts = check_rotation_regression(tmp_path, now=now)
        assert len(alerts) == 1


class TestCalibratorArtifactStaleness:
    def test_fresh_artifact_silent(self, tmp_path: Path) -> None:
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"fit_date": "2026-04-15"}))
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []

    def test_old_artifact_warns(self, tmp_path: Path) -> None:
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"fit_date": "2026-02-15"}))  # 64 days old
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.WARNING
        assert "64 days" in alerts[0].message

    def test_very_old_escalates_to_critical(self, tmp_path: Path) -> None:
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"fit_date": "2025-12-01"}))  # ~140 days old
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.CRITICAL

    def test_missing_artifact_silent(self, tmp_path: Path) -> None:
        alerts = check_calibrator_artifact_staleness(
            tmp_path / "no_such.json",
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []

    def test_malformed_artifact_silent(self, tmp_path: Path) -> None:
        """Bad JSON must not raise — a parse warning is logged and we move on."""
        path = tmp_path / "platt.json"
        path.write_text("{not-json")
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []

    def test_missing_fit_date_silent(self, tmp_path: Path) -> None:
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"schema_version": 2}))
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []

    def test_invalid_fit_date_silent(self, tmp_path: Path) -> None:
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"fit_date": "not-a-date"}))
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []

    def test_nested_metadata_fit_date_is_read(self, tmp_path: Path) -> None:
        """Regression: the favorite_bias g(mid) map nests ``fit_date`` under
        ``metadata``. Reading only the top level makes this tripwire a silent
        no-op on the calibrator behind the only order-placing strategy."""
        path = tmp_path / "favorite_bias_isotonic.json"
        payload = {
            "breakpoints": [0.4],
            "values": [0.5],
            "metadata": {"fit_date": "2026-02-15"},
        }
        path.write_text(json.dumps(payload))
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
            artifact_label="favorite_bias",
        )
        assert len(alerts) == 1
        assert alerts[0].detail["fit_date"] == "2026-02-15"

    def test_top_level_fit_date_wins_over_metadata(self, tmp_path: Path) -> None:
        """Top-level stays authoritative; the nested lookup is only a fallback."""
        path = tmp_path / "iso.json"
        payload = {"fit_date": "2026-04-15", "metadata": {"fit_date": "2020-01-01"}}
        path.write_text(json.dumps(payload))
        alerts = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
        )
        assert alerts == []  # fresh top-level date; the stale nested one is ignored

    def test_dedup_code_embeds_fit_date(self, tmp_path: Path) -> None:
        """A botched refit (artifact still stale, but fit_date changed) must
        produce a different DriftAlert code so per-code dedup re-alerts.

        Mirrors the plateau-code shape ``calibrator_plateau:<fit_date>:...``
        documented at orchestrator.py around the ``_tripwire_alerted_codes``
        declaration. Without this, the artifact-stale alert stays suppressed
        for the rest of the orchestrator's lifetime after a botched refit.
        """
        path = tmp_path / "platt.json"
        path.write_text(json.dumps({"fit_date": "2026-02-15"}))
        first = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
            artifact_label="platt",
        )
        path.write_text(json.dumps({"fit_date": "2026-02-20"}))
        second = check_calibrator_artifact_staleness(
            path,
            stale_days=45,
            today=date(2026, 4, 20),
            artifact_label="platt",
        )
        assert len(first) == 1 and len(second) == 1
        assert first[0].code != second[0].code
        assert "2026-02-15" in first[0].code
        assert "2026-02-20" in second[0].code
        assert first[0].code.startswith("calibrator_artifact_stale:platt:")


class TestCalibratorPlateauCoverage:
    def _write_isotonic(
        self,
        tmp_path: Path,
        *,
        breakpoints: list[float],
        values: list[float],
        fit_date: str = "2026-04-16",
        extras: dict | None = None,
    ) -> Path:
        payload: dict = {
            "schema_version": 2,
            "fit_date": fit_date,
            "breakpoints": breakpoints,
            "values": values,
        }
        if extras:
            payload.update(extras)
        path = tmp_path / "isotonic.json"
        path.write_text(json.dumps(payload))
        return path

    def test_fires_on_production_like_plateau(self, tmp_path: Path) -> None:
        """Width=0.434, center_deviation=0.449 — both well above defaults."""
        self._write_isotonic(
            tmp_path,
            breakpoints=[0.0, 0.433, 0.867, 1.0],
            values=[0.091, 0.201, 0.201, 0.475],
        )
        alerts = check_calibrator_plateau_coverage(tmp_path)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == AlertLevel.WARNING
        assert alert.code.startswith("calibrator_plateau:")
        assert "0.433" in alert.message
        assert "0.867" in alert.message
        assert "0.201" in alert.message

    def test_silent_on_monotonic_fit(self, tmp_path: Path) -> None:
        """Strictly-increasing values → no plateaus at all."""
        self._write_isotonic(
            tmp_path,
            breakpoints=[0.0, 0.5, 1.0],
            values=[0.1, 0.5, 0.9],
        )
        assert check_calibrator_plateau_coverage(tmp_path) == []

    def test_silent_on_non_isotonic_artifact(self, tmp_path: Path) -> None:
        """A Platt-style artifact has no 'breakpoints' key → silent no-op."""
        path = tmp_path / "isotonic.json"
        path.write_text(
            json.dumps({"schema_version": 2, "fit_date": "2026-04-16", "a": -1.0, "b": 2.0})
        )
        assert check_calibrator_plateau_coverage(tmp_path) == []

    def test_silent_when_file_missing(self, tmp_path: Path) -> None:
        """No isotonic.json in calibration_dir → empty list."""
        assert check_calibrator_plateau_coverage(tmp_path) == []

    def test_silent_on_malformed_json(self, tmp_path: Path) -> None:
        """Parse errors must not raise; they're logged and skipped."""
        (tmp_path / "isotonic.json").write_text("{not valid json")
        assert check_calibrator_plateau_coverage(tmp_path) == []

    def test_custom_threshold_fires_on_narrow_plateau(self, tmp_path: Path) -> None:
        """A 0.1-wide, deep plateau is silent at defaults but fires when
        min_width is lowered below its width."""
        self._write_isotonic(
            tmp_path,
            breakpoints=[0.0, 0.45, 0.55, 1.0],
            values=[0.1, 0.2, 0.2, 0.9],
        )
        # Default min_width=0.25 → plateau (width 0.1) below threshold.
        assert check_calibrator_plateau_coverage(tmp_path) == []
        # Lowered min_width=0.05 → plateau fires (deviation=|0.2-0.5|=0.3).
        alerts = check_calibrator_plateau_coverage(tmp_path, min_width=0.05)
        assert len(alerts) == 1
        assert alerts[0].detail["width"] == pytest.approx(0.1)

    def test_centered_plateau_does_not_fire(self, tmp_path: Path) -> None:
        """Plateau value sits at the segment's midpoint → center_deviation=0 → silent."""
        self._write_isotonic(
            tmp_path,
            breakpoints=[0.0, 0.2, 0.8, 1.0],
            values=[0.1, 0.5, 0.5, 0.9],
        )
        alerts = check_calibrator_plateau_coverage(tmp_path)
        assert alerts == []

    def test_alert_payload_fields(self, tmp_path: Path) -> None:
        """Every documented detail field is populated with the right type/value."""
        self._write_isotonic(
            tmp_path,
            breakpoints=[0.0, 0.3, 0.8, 1.0],
            values=[0.05, 0.2, 0.2, 0.95],
            fit_date="2026-04-16",
        )
        alerts = check_calibrator_plateau_coverage(tmp_path)
        assert len(alerts) == 1
        d = alerts[0].detail
        assert d["fit_date"] == "2026-04-16"
        assert d["plateau_low"] == pytest.approx(0.3)
        assert d["plateau_high"] == pytest.approx(0.8)
        assert d["plateau_value"] == pytest.approx(0.2)
        assert d["width"] == pytest.approx(0.5)
        # center = 0.55, deviation = |0.2 - 0.55| = 0.35
        assert d["center_deviation"] == pytest.approx(0.35)
        # Stable dedup key embeds fit_date + plateau bounds.
        assert alerts[0].code == "calibrator_plateau:2026-04-16:0.300-0.800"

    def test_honors_isotonic_filename_override(self, tmp_path: Path) -> None:
        """When ``isotonic_filename`` points at a non-default artifact, the
        check reads bounds from THAT file and ignores any stale canonical
        ``isotonic.json`` sitting alongside it.

        Regression for the 2026-05-22 stale-bounds tripwire bug: the
        previous version hardcoded ``isotonic.json`` and reported plateau
        geometry from the rollback file even after a refit cutover to
        ``isotonic_v2.json``.
        """
        # Canonical file has a wide dangerous plateau (the stale rollback).
        stale = tmp_path / "isotonic.json"
        stale.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "fit_date": "2026-05-12",
                    "breakpoints": [0.0, 0.649, 0.967, 1.0],
                    "values": [0.05, 0.322, 0.322, 0.99],
                }
            )
        )
        # Live (refit) artifact has a milder, centered plateau — silent at defaults.
        live = tmp_path / "isotonic_v2.json"
        live.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "fit_date": "2026-05-20",
                    "breakpoints": [0.0, 0.2, 0.8, 1.0],
                    "values": [0.05, 0.5, 0.5, 0.99],
                }
            )
        )
        # Reading the live artifact: no alerts (centered plateau).
        assert (
            check_calibrator_plateau_coverage(tmp_path, isotonic_filename="isotonic_v2.json") == []
        )
        # Default still reads the canonical file, proving the override is what
        # selected the live artifact (and demonstrates the bug surface).
        stale_alerts = check_calibrator_plateau_coverage(tmp_path)
        assert len(stale_alerts) == 1
        assert "0.649" in stale_alerts[0].message


class TestRunAll:
    def test_integrates_checks_and_flattens_alerts(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        # Stale active file + old calibrator artifact.
        _touch(tmp_path / "eval_audit.jsonl", mtime=now - timedelta(hours=5))
        platt = tmp_path / "platt_per_strike_type.json"
        platt.write_text(json.dumps({"fit_date": "2026-02-01"}))  # ~78 days old

        alerts = run_all(
            state_dir=tmp_path,
            calibrator_paths={"per_type_platt": platt},
            config=TripwireConfig(
                stale_active_seconds=3600.0,
                stale_calibrator_days=45,
            ),
            now=now,
        )
        codes = sorted(a.code for a in alerts)
        assert "audit_log_stale:eval_audit.jsonl" in codes
        assert any(c.startswith("calibrator_artifact_stale:") for c in codes)

    def test_healthy_state_emits_nothing(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)
        for s in STREAM_NAMES:
            _touch(tmp_path / s, mtime=now - timedelta(minutes=5))
        platt = tmp_path / "platt.json"
        platt.write_text(json.dumps({"fit_date": "2026-04-15"}))

        alerts = run_all(
            state_dir=tmp_path,
            calibrator_paths={"per_type_platt": platt},
            config=TripwireConfig(
                stale_active_seconds=3600.0,
                stale_calibrator_days=45,
            ),
            now=now,
        )
        assert alerts == []


class TestLoadPlateauSeenCodes:
    def test_missing_file_returns_empty_set(self, tmp_path: Path) -> None:
        assert load_plateau_seen_codes(tmp_path) == set()

    def test_corrupt_file_returns_empty_set(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text("not json{{{")
        assert load_plateau_seen_codes(tmp_path) == set()

    def test_empty_codes_list_returns_empty_set(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            '{"_schema_version": 1, "codes": [], "updated_at": "2026-05-09T00:00:00+00:00"}'
        )
        assert load_plateau_seen_codes(tmp_path) == set()

    def test_populated_file_returns_correct_set(self, tmp_path: Path) -> None:
        codes = [
            "calibrator_plateau:2026-05-01:0.649-0.967",
            "calibrator_plateau:2026-04-01:0.433-0.867",
        ]
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            json.dumps(
                {"_schema_version": 1, "codes": codes, "updated_at": "2026-05-09T00:00:00+00:00"}
            )
        )
        assert load_plateau_seen_codes(tmp_path) == set(codes)

    def test_missing_codes_key_returns_empty_set(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            '{"_schema_version": 1, "updated_at": "2026-05-09T00:00:00+00:00"}'
        )
        assert load_plateau_seen_codes(tmp_path) == set()

    def test_non_string_codes_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            '{"_schema_version": 1, "codes": ["valid-code", 42, null, "another-valid"], "updated_at": "x"}'
        )
        assert load_plateau_seen_codes(tmp_path) == {"valid-code", "another-valid"}

    def test_unknown_schema_version_returns_empty_set(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            '{"_schema_version": 99, "codes": ["calibrator_plateau:x:y"], "updated_at": "x"}'
        )
        assert load_plateau_seen_codes(tmp_path) == set()

    def test_extra_top_level_keys_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "tripwire_plateau_seen.json").write_text(
            '{"_schema_version": 1, "codes": ["calibrator_plateau:x:y"], '
            '"updated_at": "x", "unknown_future_key": true}'
        )
        assert load_plateau_seen_codes(tmp_path) == {"calibrator_plateau:x:y"}


class TestPersistSeenCode:
    """``persist_seen_code`` now writes the v2 ``alert_acks.json`` store
    (Task 6). It was renamed from ``persist_plateau_seen_code``, which wrote
    the v1 ``tripwire_plateau_seen.json`` file directly — see
    ``tests/test_alert_acks.py`` for the full migration/ack-store coverage.
    """

    def test_creates_file_on_first_call(self, tmp_path: Path) -> None:
        persist_seen_code(tmp_path, "calibrator_plateau:2026-05-01:0.649-0.967")
        raw = json.loads((tmp_path / "alert_acks.json").read_text())
        assert "calibrator_plateau:2026-05-01:0.649-0.967" in raw["seen"]
        assert raw["_schema_version"] == 2
        assert raw["acked"] == {}

    def test_idempotent_same_code(self, tmp_path: Path) -> None:
        code = "calibrator_plateau:2026-05-01:0.649-0.967"
        persist_seen_code(tmp_path, code)
        persist_seen_code(tmp_path, code)
        raw = json.loads((tmp_path / "alert_acks.json").read_text())
        assert raw["seen"].count(code) == 1

    def test_accumulates_distinct_codes(self, tmp_path: Path) -> None:
        code_a = "calibrator_plateau:2026-05-01:0.649-0.967"
        code_b = "calibrator_plateau:2026-04-01:0.433-0.867"
        persist_seen_code(tmp_path, code_a)
        persist_seen_code(tmp_path, code_b)
        raw = json.loads((tmp_path / "alert_acks.json").read_text())
        assert set(raw["seen"]) == {code_a, code_b}

    def test_tolerates_corrupt_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "alert_acks.json").write_text("not json")
        code = "calibrator_plateau:2026-05-01:0.649-0.967"
        persist_seen_code(tmp_path, code)
        raw = json.loads((tmp_path / "alert_acks.json").read_text())
        assert code in raw["seen"]

    def test_no_tmp_file_left_behind_on_success(self, tmp_path: Path) -> None:
        persist_seen_code(tmp_path, "calibrator_plateau:2026-05-01:0.649-0.967")
        assert list(tmp_path.glob("*.tmp")) == []


def test_plateau_alert_code_prefix_matches_producer() -> None:
    """PLATEAU_ALERT_CODE_PREFIX must match the prefix used in check_calibrator_plateau_coverage."""
    assert PLATEAU_ALERT_CODE_PREFIX == "calibrator_plateau:"


def test_stream_names_includes_forecasts_raw() -> None:
    """C12: ``forecasts_raw.jsonl`` must be tripwired alongside the audit streams.

    Catches a silently-broken ``ForecastsRawWriter`` — the EMOS shadow
    pipeline degrades to no-op without this guard.
    """
    from python.analytics.audit_tripwires import STREAM_NAMES

    assert "forecasts_raw.jsonl" in STREAM_NAMES


class TestBiasTableFreshness:
    def test_fresh_table_silent(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        path = tmp_path / "bias_correction.json"
        _touch(path, mtime=now - timedelta(days=5))
        alerts = check_bias_table_freshness(path, stale_days=30, now=now)
        assert alerts == []

    def test_stale_table_warns(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        path = tmp_path / "bias_correction.json"
        # 45d old → above 30d threshold, below 2× (60d) → WARNING.
        _touch(path, mtime=now - timedelta(days=45))
        alerts = check_bias_table_freshness(path, stale_days=30, now=now)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.WARNING
        assert "bias_table_stale:" in alerts[0].code
        assert "scripts/analyze_gfs_bias.py" in alerts[0].message
        assert alerts[0].detail["threshold_days"] == 30
        assert 44.0 <= alerts[0].detail["age_days"] <= 46.0

    def test_stale_nws_table_points_at_nws_pipeline(self, tmp_path: Path) -> None:
        """The live NWS-anchored table is refreshed by a different pipeline
        than the ERA5 fallback — the alert must name the right scripts so a
        stale-table alert on the deployed artifact is actionable, not
        misdirecting the operator at the ERA5 collect/analyze path.
        """
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        nws_dir = tmp_path / "nws_bias"
        nws_dir.mkdir()
        path = nws_dir / "bias_correction.json"
        _touch(path, mtime=now - timedelta(days=45))
        alerts = check_bias_table_freshness(path, stale_days=30, now=now)
        assert len(alerts) == 1
        assert "scripts/fit_nws_bias.py" in alerts[0].message
        assert "scripts/build_nws_bias_corpus.py" in alerts[0].message
        # Must NOT point at the ERA5 pipeline for the deployed NWS table.
        assert "scripts/analyze_gfs_bias.py" not in alerts[0].message

    def test_very_stale_escalates_to_critical(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        path = tmp_path / "bias_correction.json"
        # 65d old → ≥ 2× 30d threshold → CRITICAL. Mirrors the
        # production state of the 2026-03-21 artifact when this
        # tripwire was authored (61 days old on 2026-05-21).
        _touch(path, mtime=now - timedelta(days=65))
        alerts = check_bias_table_freshness(path, stale_days=30, now=now)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.CRITICAL

    def test_missing_table_silent_by_default(self, tmp_path: Path) -> None:
        """Default contract: a missing artifact is silent (the freshness
        primitive treats absence as "nothing to age-check"). Callers that
        expect the file to exist opt into the stricter check via
        ``missing_is_error`` — see ``test_missing_table_alerts_when_required``.
        """
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        alerts = check_bias_table_freshness(
            tmp_path / "no_such.json",
            stale_days=30,
            now=now,
        )
        assert alerts == []

    def test_missing_table_alerts_when_required(self, tmp_path: Path) -> None:
        """When the caller asserts the table must exist (bias correction
        enabled + a configured path), a missing file is a CRITICAL failure —
        corrections are silently disabled at load with no other tripwire.
        """
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        missing = tmp_path / "nws_bias" / "bias_correction.json"
        alerts = check_bias_table_freshness(
            missing,
            stale_days=30,
            now=now,
            missing_is_error=True,
        )
        assert len(alerts) == 1
        assert alerts[0].severity == AlertLevel.CRITICAL
        assert alerts[0].code.startswith("bias_table_missing:")
        assert "missing" in alerts[0].message.lower()
        assert alerts[0].detail["path"] == str(missing)

    def test_dedup_code_embeds_mtime_date(self, tmp_path: Path) -> None:
        """A botched refresh (file touched but still old) must
        produce a different DriftAlert code so the orchestrator's
        per-code dedup re-alerts.
        """
        now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
        path = tmp_path / "bias_correction.json"
        _touch(path, mtime=now - timedelta(days=65))
        first = check_bias_table_freshness(path, stale_days=30, now=now)
        _touch(path, mtime=now - timedelta(days=60))
        second = check_bias_table_freshness(path, stale_days=30, now=now)
        assert len(first) == 1 and len(second) == 1
        assert first[0].code != second[0].code
        assert first[0].code.startswith("bias_table_stale:")
        assert second[0].code.startswith("bias_table_stale:")


def test_run_all_emits_bias_table_staleness_warning(tmp_path: Path) -> None:
    """run_all must surface a stale bias_correction.json when given a path."""
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    bias = tmp_path / "bias_correction.json"
    _touch(bias, mtime=now - timedelta(days=45))

    alerts = run_all(
        state_dir=tmp_path,
        calibrator_paths={},
        config=TripwireConfig(stale_bias_table_days=30),
        now=now,
        bias_table_path=bias,
    )

    assert any(
        a.code.startswith("bias_table_stale:") and a.severity == AlertLevel.WARNING for a in alerts
    ), f"expected bias-table WARNING in {[a.code for a in alerts]}"


def test_run_all_emits_bias_table_missing_critical(tmp_path: Path) -> None:
    """run_all forwards ``bias_table_missing_is_error`` so the orchestrator's
    "correction enabled + configured path" state surfaces a CRITICAL alert
    when the live table is absent (rather than failing silent).
    """
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    missing = tmp_path / "nws_bias" / "bias_correction.json"

    alerts = run_all(
        state_dir=tmp_path,
        calibrator_paths={},
        config=TripwireConfig(stale_bias_table_days=30),
        now=now,
        bias_table_path=missing,
        bias_table_missing_is_error=True,
    )

    assert any(
        a.code.startswith("bias_table_missing:") and a.severity == AlertLevel.CRITICAL
        for a in alerts
    ), f"expected bias-table CRITICAL in {[a.code for a in alerts]}"


def test_run_all_missing_bias_table_silent_without_flag(tmp_path: Path) -> None:
    """Without the flag a missing table stays silent — preserves the default
    primitive contract for any caller that does not require existence.
    """
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    missing = tmp_path / "nws_bias" / "bias_correction.json"
    alerts = run_all(
        state_dir=tmp_path,
        calibrator_paths={},
        config=TripwireConfig(stale_bias_table_days=30),
        now=now,
        bias_table_path=missing,
    )
    assert not any(a.code.startswith("bias_table_missing:") for a in alerts)


def test_run_all_skips_bias_table_when_path_omitted(tmp_path: Path) -> None:
    """Default (no bias_table_path) must not invent a check — matches the
    existing ``calibration_dir`` omission semantics.
    """
    now = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    alerts = run_all(
        state_dir=tmp_path,
        calibrator_paths={},
        config=TripwireConfig(),
        now=now,
    )
    assert not any(a.code.startswith("bias_table_stale:") for a in alerts)


def test_run_all_emits_emos_artifact_staleness_warning(tmp_path: Path) -> None:
    """Stale EMOS coefficients > stale_calibrator_days produce a WARNING DriftAlert.

    ``check_calibrator_artifact_staleness`` keys off the artifact's
    ``fit_date`` field, not mtime — matches the format scope-A's
    ``scripts/fit_emos.py`` writes.
    """
    coeffs = tmp_path / "coefficients.json"
    # 50 days before 2026-05-05 (today) → just past the 45-day threshold,
    # but well below the 2× CRITICAL escalation. fit_date format mirrors
    # ``scripts/fit_emos.py:152`` (date.today().isoformat()).
    coeffs.write_text(json.dumps({"fit_date": "2026-03-16", "schema_version": 1}))

    now = datetime(2026, 5, 5, 12, 0, tzinfo=UTC)
    alerts = run_all(
        state_dir=tmp_path,
        calibrator_paths={"emos": coeffs},
        config=TripwireConfig(stale_calibrator_days=45),
        now=now,
    )

    assert any(
        a.severity == AlertLevel.WARNING and "emos" in (a.code + a.message).lower() for a in alerts
    ), f"expected EMOS staleness WARNING in {[a.code for a in alerts]}"


class TestAckKeyCodeFormats:
    """Codes that key an acknowledgement must carry their invalidating context.

    See spec 2026-09-01-alert-relevance-design.md §3.2.
    """

    def test_plateau_resolves_nested_metadata_fit_date(self, tmp_path: Path) -> None:
        """The plateau check had its own inline read with an 'unknown' default.

        _read_fit_date already had a nested-metadata fallback; this site
        did not, so a favorite_bias-shaped artifact produced
        'calibrator_plateau:unknown:...' — byte-identical across refits.
        """
        # Off-center (value 0.72 vs segment center 0.50, deviation=0.22 >=
        # min_center_deviation 0.15) so the plateau actually fires — a
        # centered plateau (value == center) is silent by design (see
        # ``test_centered_plateau_does_not_fire`` above) and would fail this
        # test for the wrong reason regardless of the fit_date fix.
        artifact = tmp_path / "iso.json"
        artifact.write_text(
            json.dumps(
                {
                    "breakpoints": [0.0, 0.30, 0.70, 1.0],
                    "values": [0.0, 0.72, 0.72, 1.0],
                    "metadata": {"fit_date": "2026-07-04"},
                }
            )
        )
        alerts = check_calibrator_plateau_coverage(
            calibration_dir=tmp_path, isotonic_filename="iso.json"
        )
        assert alerts, "expected a plateau alert for a wide flat segment"
        assert "2026-07-04" in alerts[0].code
        assert "unknown" not in alerts[0].code

    def test_bias_table_stale_code_carries_the_path(self, tmp_path: Path) -> None:
        """bias_table_path is a live rollback lever (NWS <-> ERA5).

        Keyed on mtime alone, an ack could silence a DIFFERENT artifact whose
        mtime date happens to coincide.
        """
        table = tmp_path / "bias_correction.json"
        table.write_text("{}")
        old = datetime.now(UTC) - timedelta(days=400)
        os.utime(table, (old.timestamp(), old.timestamp()))
        alerts = check_bias_table_freshness(
            bias_table_path=table, stale_days=45, now=datetime.now(UTC)
        )
        assert alerts
        assert str(table) in alerts[0].code


class TestAlertOwnership:
    """Owner and ack_key are producer-set. Spec §3.1.

    ack_key is None by default so a NEW alert code is un-ackable until its
    author declares what invalidates it (fail-noisy).
    """

    def test_stream_alerts_are_infrastructure(self, tmp_path: Path) -> None:
        """A dead writer is actionable regardless of which strategy owns it."""
        stream = tmp_path / "eval_audit.jsonl"
        stream.write_text('{"x":1}\n')
        old = datetime.now(UTC) - timedelta(hours=12)
        os.utime(stream, (old.timestamp(), old.timestamp()))
        alerts = check_active_file_staleness(
            state_dir=tmp_path,
            stale_seconds=3600,
            streams=("eval_audit.jsonl",),
            now=datetime.now(UTC),
        )
        assert alerts
        assert alerts[0].owner is None
        assert alerts[0].ack_key is None

    def test_bias_table_missing_is_infrastructure_and_unackable(self, tmp_path: Path) -> None:
        """Must be fixed, not acknowledged — it catches silent load failure."""
        alerts = check_bias_table_freshness(
            bias_table_path=tmp_path / "gone.json",
            stale_days=45,
            missing_is_error=True,
            now=datetime.now(UTC),
        )
        assert alerts
        assert alerts[0].owner is None
        assert alerts[0].ack_key is None

    def test_bias_table_stale_is_weather_edge_and_ackable(self, tmp_path: Path) -> None:
        table = tmp_path / "bias_correction.json"
        table.write_text("{}")
        old = datetime.now(UTC) - timedelta(days=400)
        os.utime(table, (old.timestamp(), old.timestamp()))
        alerts = check_bias_table_freshness(
            bias_table_path=table, stale_days=45, now=datetime.now(UTC)
        )
        assert alerts[0].owner == "weather_edge"
        assert alerts[0].ack_key is not None

    def test_plateau_ack_key_none_when_fit_date_unresolvable(self, tmp_path: Path) -> None:
        """The ack_key guard is the only thing stopping a permanent-mute ack.

        A plateau artifact with no top-level ``fit_date`` and no
        ``metadata.fit_date`` resolves to the literal code segment "unknown"
        (see ``check_calibrator_plateau_coverage``). Since "unknown" is not
        unique per artifact, acking a code containing it would silently mute
        every future fit_date-less refit whose plateau bounds round the same
        way — including the detector that exists because of the 2026-04-22
        loss. ``ack_key`` must be ``None`` in this case, not the string
        "unknown".
        """
        payload = {
            "schema_version": 2,
            # No top-level fit_date, no metadata block at all.
            "breakpoints": [0.0, 0.30, 0.70, 1.0],
            "values": [0.0, 0.72, 0.72, 1.0],
        }
        path = tmp_path / "isotonic.json"
        path.write_text(json.dumps(payload))
        alerts = check_calibrator_plateau_coverage(tmp_path)
        assert alerts, "expected a plateau alert for a wide off-center segment"
        assert "unknown" in alerts[0].code
        assert alerts[0].ack_key is None

    def test_calibrator_artifact_stale_owner_by_label(self, tmp_path: Path) -> None:
        """owner branches on artifact_label: favorite_bias vs everything else."""
        fb_path = tmp_path / "favorite_bias_isotonic.json"
        fb_path.write_text(json.dumps({"fit_date": "2026-02-15"}))
        fb_alerts = check_calibrator_artifact_staleness(
            fb_path,
            stale_days=45,
            today=date(2026, 4, 20),
            artifact_label="favorite_bias",
        )
        assert len(fb_alerts) == 1
        assert fb_alerts[0].owner == "favorite_bias"

        iso_path = tmp_path / "isotonic.json"
        iso_path.write_text(json.dumps({"fit_date": "2026-02-15"}))
        iso_alerts = check_calibrator_artifact_staleness(
            iso_path,
            stale_days=45,
            today=date(2026, 4, 20),
            artifact_label="isotonic",
        )
        assert len(iso_alerts) == 1
        assert iso_alerts[0].owner == "weather_edge"
