"""Tripwires for the audit-log write path and calibration artifacts.

Five independent checks, each returning zero or more ``DriftAlert``
records for the calibration-monitor gate to dispatch:

1. **Active-file staleness** — If the active ``{stream}.jsonl`` hasn't
   been modified in longer than ``stale_active_seconds``, the writer
   has silently stopped. This catches: disk full, permission flip,
   orphaned file handle after a failed rotation, retention daemon
   deleting the file while the bot is live.

2. **No-rotation regression** — If *all* rotated siblings
   (``{stream}.jsonl.YYYY-MM-DD``) are missing, and the active file
   has modification times spanning more than one UTC day, rotation is
   broken. This catches: a regression to ``_maybe_rotate`` that
   silently no-ops, or a clock that gets stuck.

3. **Calibrator artifact stale** — If the Platt artifact's
   ``fit_date`` is more than ``stale_calibrator_days`` old, the
   calibrator is drifting off what production markets look like.

4. **Calibrator plateau coverage** — Flags isotonic fits with wide
   flat segments whose constant value sits far from the segment's
   center of mass. These geometries sign-flip trades against
   mid-priced markets (see the 2026-04-22 CHI loss).

5. **Bias-table freshness** — If the *live* bias table (the configured
   ``weather.bias_table_path`` — currently the NWS-anchored
   ``data/nws_bias/bias_correction.json`` — or the ERA5 default when
   that is ``null``) has an mtime exceeding ``stale_bias_table_days``,
   the baseline is too old to catch active regime shifts; the SNR gate
   will silently refuse corrections even while live cells bleed. The
   caller resolves which artifact is deployed and passes it in — the
   check must never hard-code the ERA5 fallback, or it false-alarms on
   a non-deployed file while staying blind to the live table aging out
   (the 2026-06-21 misdirected-tripwire fix; same class as the
   2026-05-22 stale-bounds plateau-tripwire bug).

These are deliberately cheap (one ``stat`` + one tail-line read per
stream) so the monitor can run them on every tick. All three take the
"unknown" path (no alert) when data is missing — these check for
regressions on an established baseline, not first-deploy state.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from python.analytics.audit_log import parse_rotated_suffix
from python.analytics.calibration_monitor import DriftAlert
from python.notifications.base import AlertLevel
from python.utils.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

# Sensible production defaults — tuned for our 5-min bot scan cadence.
DEFAULT_STALE_ACTIVE_SECONDS = 3 * 3600.0  # 3h with zero writes → alert
DEFAULT_STALE_CALIBRATOR_DAYS = 45  # Platt refit cadence target: monthly
DEFAULT_STALE_BIAS_TABLE_DAYS = 30  # GFS bias baseline refresh cadence

# Stream basenames emitted by ``audit_log.build_sinks`` plus the EMOS shadow
# stream (``forecasts_raw.jsonl``) written by ``ForecastsRawWriter`` — staleness
# of the latter signals a silently-broken EMOS observation pipeline.
#
# ``candidate_audit.jsonl`` is intentionally excluded: it is only written when
# markets reach Stage-2 (the orderbook depth check), so it goes stale during
# overnight and low-activity windows even when the bot is healthy. Liveness of
# the scanning pipeline is already covered by ``scan_funnel.jsonl`` (every scan)
# and ``eval_audit.jsonl`` (every Stage-1 rejection/pass).
STREAM_NAMES: tuple[str, ...] = (
    "eval_audit.jsonl",
    "scan_funnel.jsonl",
    "forecasts_raw.jsonl",
)


@dataclass(frozen=True)
class TripwireConfig:
    """Per-check thresholds.

    ``stale_active_seconds``, ``stale_calibrator_days``, and
    ``stale_bias_table_days`` are the only knobs worth exposing; the
    rotation check is parameterless (spans more than one UTC date →
    alert).
    """

    stale_active_seconds: float = DEFAULT_STALE_ACTIVE_SECONDS
    stale_calibrator_days: int = DEFAULT_STALE_CALIBRATOR_DAYS
    stale_bias_table_days: int = DEFAULT_STALE_BIAS_TABLE_DAYS


def _active_age_seconds(path: Path, now: datetime) -> float | None:
    """Seconds since ``path`` was last modified. ``None`` if missing."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        logger.exception("tripwire: stat failed for %s", path)
        return None
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    return (now - mtime).total_seconds()


def _read_last_jsonl_record(path: Path, *, tail_bytes: int = 65536) -> dict[str, Any] | None:
    """Return the last parseable JSON object in a JSONL file, or ``None``.

    Reads only the trailing ``tail_bytes`` so the check stays cheap even on
    large rotated logs (scan_funnel can grow to many MB across a UTC day).
    Tolerant of a torn final line — the per-record writers append-on-write
    so the last line on disk may be partial during a write — and of a torn
    *first* line inside the tail window, since the chunk almost always
    begins mid-line.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(0, size - tail_bytes))
            chunk = f.read()
    except OSError:
        return None
    for raw in reversed(chunk.splitlines()):
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _streams_with_rotated_siblings(
    state_dir: Path,
    streams: tuple[str, ...],
) -> set[str]:
    """Return the subset of ``streams`` that have at least one rotated sibling.

    One directory walk serves all streams — callers with three streams
    pay one ``iterdir``, not three.
    """
    try:
        entries = list(state_dir.iterdir())
    except OSError:
        return set()
    found: set[str] = set()
    remaining = set(streams)
    for entry in entries:
        if not remaining:
            break
        if not entry.is_file():
            continue
        if parse_rotated_suffix(entry.name) is None:
            continue
        for stream in list(remaining):
            if entry.name.startswith(f"{stream}."):
                found.add(stream)
                remaining.discard(stream)
                break
    return found


def check_active_file_staleness(
    state_dir: Path,
    *,
    stale_seconds: float,
    now: datetime,
    streams: tuple[str, ...] = STREAM_NAMES,
) -> list[DriftAlert]:
    """Alert once per stream whose active file hasn't been written recently.

    Only considers streams whose file *exists* — a missing file is the
    "bot hasn't started yet / stream disabled" path, which should not
    alert. Severity is ``WARNING`` up to 3× ``stale_seconds`` and
    ``CRITICAL`` beyond, so long blackouts escalate past any cooldown.
    """
    out: list[DriftAlert] = []
    for stream in streams:
        path = state_dir / stream
        age = _active_age_seconds(path, now)
        if age is None:
            continue
        if age < stale_seconds:
            continue

        # ``forecasts_raw.jsonl`` is only written when ``scan()`` returned
        # at least one tradeable market (``generate_signals`` short-circuits
        # at ``markets == []`` before the writer runs — see
        # ``weather_edge.py:generate_signals``). During the post-ET-midnight window
        # where every market is past-dated, same-day-blocked, or
        # below the liquidity gates, the writer is correctly silent.
        # Suppress this stream's alert when the most recent scan_funnel row
        # confirms ``n_markets == 0``. Other streams emit on every scan
        # (scan-reject rows fire from inside scan() itself) so their
        # staleness always indicates real failure.
        if stream == "forecasts_raw.jsonl":
            latest_funnel = _read_last_jsonl_record(state_dir / "scan_funnel.jsonl")
            if latest_funnel is not None and latest_funnel.get("n_markets") == 0:
                logger.info(
                    "tripwire: suppressing %s staleness alert (age=%.1fh) — "
                    "most recent scan_funnel reports n_markets=0",
                    stream,
                    age / 3600,
                )
                continue

        severity = AlertLevel.CRITICAL if age >= 3 * stale_seconds else AlertLevel.WARNING
        out.append(
            DriftAlert(
                code=f"audit_log_stale:{stream}",
                severity=severity,
                message=(
                    f"Audit log {stream} has not been written in "
                    f"{age / 3600:.1f}h (threshold {stale_seconds / 3600:.1f}h). "
                    "The writer may be stuck or the log was deleted under the bot."
                ),
                detail={
                    "stream": stream,
                    "path": str(path),
                    "age_seconds": age,
                    "threshold_seconds": stale_seconds,
                },
                owner=None,
                ack_key=None,
            )
        )
    return out


def check_rotation_regression(
    state_dir: Path,
    *,
    now: datetime,
    streams: tuple[str, ...] = STREAM_NAMES,
) -> list[DriftAlert]:
    """Alert when an active file is older than 1 day with no rotated siblings.

    Rotation fires on the first write after UTC midnight. If the
    active file's mtime is more than one full day old and *no*
    ``{stream}.YYYY-MM-DD`` sibling exists, rotation silently stopped.

    One-alert-per-stream; the monitor's gate handles cooldowns.
    """
    out: list[DriftAlert] = []
    one_day = timedelta(days=1).total_seconds()
    rotated = _streams_with_rotated_siblings(state_dir, streams)
    for stream in streams:
        path = state_dir / stream
        age = _active_age_seconds(path, now)
        if age is None or age < one_day:
            continue
        if stream in rotated:
            continue
        out.append(
            DriftAlert(
                code=f"audit_log_no_rotation:{stream}",
                severity=AlertLevel.WARNING,
                message=(
                    f"Audit log {stream} spans >1 day with no rotated sibling. "
                    "Daily rotation appears broken."
                ),
                detail={
                    "stream": stream,
                    "path": str(path),
                    "age_seconds": age,
                },
                owner=None,
                ack_key=None,
            )
        )
    return out


def _read_fit_date(path: Path) -> date | None:
    """Return the ``fit_date`` field from a calibration artifact, or ``None``.

    Two artifact shapes are in use and both must be readable: the weather
    isotonic/Platt artifacts carry ``fit_date`` at the top level, while the
    favorite_bias g(mid) map nests it under ``metadata``. Reading only the top
    level silently returns None for the nested shape — which does not fail
    loudly, it just makes the staleness tripwire a no-op on that artifact.
    """
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("tripwire: unreadable calibration artifact %s", path, exc_info=True)
        return None
    fit_raw = raw.get("fit_date")
    if not fit_raw:
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            fit_raw = metadata.get("fit_date")
    if not fit_raw:
        return None
    try:
        return date.fromisoformat(str(fit_raw))
    except ValueError:
        return None


def check_calibrator_artifact_staleness(
    artifact_path: Path,
    *,
    stale_days: int,
    today: date,
    artifact_label: str = "calibrator",
) -> list[DriftAlert]:
    """Alert when the calibration artifact's ``fit_date`` is too old.

    Silent on missing artifacts — those represent a valid "calibration
    disabled" state that the orchestrator flags elsewhere. Escalates
    to ``CRITICAL`` at 2× the threshold.
    """
    fit_date = _read_fit_date(artifact_path)
    if fit_date is None:
        return []
    age_days = (today - fit_date).days
    if age_days < stale_days:
        return []
    severity = AlertLevel.CRITICAL if age_days >= 2 * stale_days else AlertLevel.WARNING
    return [
        DriftAlert(
            code=f"calibrator_artifact_stale:{artifact_label}:{fit_date.isoformat()}",
            severity=severity,
            message=(
                f"Calibration artifact {artifact_path.name} was fit "
                f"{age_days} days ago ({fit_date.isoformat()}), "
                f"threshold {stale_days} days. Consider refitting."
            ),
            detail={
                "artifact": artifact_label,
                "path": str(artifact_path),
                "fit_date": fit_date.isoformat(),
                "age_days": age_days,
                "threshold_days": stale_days,
            },
            owner=("favorite_bias" if artifact_label == "favorite_bias" else "weather_edge"),
            ack_key=fit_date.isoformat(),
        )
    ]


def check_bias_table_freshness(
    bias_table_path: Path,
    *,
    stale_days: int,
    now: datetime,
    missing_is_error: bool = False,
) -> list[DriftAlert]:
    """Alert when the GFS bias-correction table is too old to track regime drift.

    Keys off file mtime (the artifact has no fit_date field). Embeds
    the mtime date in the dedup code so a botched refresh (file
    touched but still old) re-alerts.

    ``missing_is_error`` selects the missing-file contract. Default
    ``False`` stays silent (the freshness primitive treats absence as
    "nothing to age-check"). When the caller asserts the table must exist
    — bias correction enabled *and* a configured path resolved — pass
    ``True`` so a missing file raises a CRITICAL ``bias_table_missing``
    alert. Without this a misconfigured/absent live table silently
    disables corrections at load with no tripwire (the 2026-06-21
    follow-up to the misdirected-tripwire fix).
    """
    age = _active_age_seconds(bias_table_path, now)
    if age is None:
        if not missing_is_error:
            return []
        return [
            DriftAlert(
                code=f"bias_table_missing:{bias_table_path}",
                severity=AlertLevel.CRITICAL,
                message=(
                    f"Bias correction is ENABLED but the configured table "
                    f"{bias_table_path} is missing — corrections are silently "
                    f"disabled at load. Restore the artifact, or set "
                    f"weather.bias_table_path: null to fall back to the ERA5 "
                    f"table (data/gfs_bias/bias_correction.json)."
                ),
                detail={"path": str(bias_table_path), "reason": "missing"},
                owner=None,
                ack_key=None,
            )
        ]
    age_days = age / 86400.0
    if age_days < stale_days:
        return []
    try:
        mtime_iso = (
            datetime.fromtimestamp(bias_table_path.stat().st_mtime, tz=UTC).date().isoformat()
        )
    except OSError:
        return []
    severity = AlertLevel.CRITICAL if age_days >= 2 * stale_days else AlertLevel.WARNING
    # The refresh pipeline depends on which table is deployed: the NWS
    # settlement-anchored table and the ERA5 canonical fallback are built by
    # different scripts. Key off the parent dir so the alert points the
    # operator at the right one.
    if bias_table_path.parent.name == "nws_bias":
        refresh_hint = (
            "refresh via scripts/build_nws_bias_corpus.py + "
            "scripts/fit_nws_bias.py (OOS-validate with "
            "scripts/validate_nws_bias.py before deploy)."
        )
    else:
        refresh_hint = "refresh via scripts/collect_gfs_bias_data.py + scripts/analyze_gfs_bias.py."
    return [
        DriftAlert(
            # The path is part of the key: ``weather.bias_table_path`` is a live
            # rollback lever (NWS-anchored <-> ERA5), so an mtime-only code could
            # let an acknowledgement silence a DIFFERENT artifact.
            code=f"bias_table_stale:{bias_table_path}:{mtime_iso}",
            severity=severity,
            message=(
                f"GFS bias-correction table {bias_table_path.name} was last "
                f"modified {age_days:.1f} days ago ({mtime_iso}), "
                f"threshold {stale_days} days. The SNR gate silently refuses "
                f"corrections when the baseline misses recent regime shifts — "
                f"{refresh_hint}"
            ),
            detail={
                "path": str(bias_table_path),
                "age_days": age_days,
                "mtime_date": mtime_iso,
                "threshold_days": stale_days,
            },
            owner="weather_edge",
            ack_key=f"{bias_table_path}:{mtime_iso}",
        )
    ]


# Plateau thresholds — tuned against a production fit whose [0.433, 0.867]→0.201
# segment sign-flipped a losing trade on KXLOWTCHI-T53. ``min_width=0.25`` catches
# any plateau covering ≥25% of raw-prob space (the pathological segment was 0.434
# wide). ``min_center_deviation=0.15`` is ~1/3 of that fit's 0.45 gap between
# plateau value (0.201) and center (0.65) — low enough to flag milder variants,
# high enough to stay silent on smoothly-monotone fits where value ≈ center.
DEFAULT_PLATEAU_MIN_WIDTH = 0.25
DEFAULT_PLATEAU_MIN_CENTER_DEVIATION = 0.15


def check_calibrator_plateau_coverage(
    calibration_dir: Path,
    *,
    min_width: float = DEFAULT_PLATEAU_MIN_WIDTH,
    min_center_deviation: float = DEFAULT_PLATEAU_MIN_CENTER_DEVIATION,
    isotonic_filename: str = "isotonic.json",
) -> list[DriftAlert]:
    """Detect dangerous plateaus in the live isotonic calibrator.

    A "dangerous plateau" is a flat segment [L, U] with constant value v
    where:
      - U - L >= min_width (covers a meaningful swath of raw-prob space), AND
      - |v - (L + U) / 2| >= min_center_deviation (systematically
        disagrees with the plateau's center of mass, enabling sign-flips
        against mid-priced markets).

    Returns a DriftAlert (severity=WARNING, code prefix="calibrator_plateau")
    per dangerous plateau found. Non-isotonic calibrators (Platt, EMOS,
    etc.) have no breakpoints in their metadata — this check is a silent
    no-op for them.

    ``isotonic_filename`` defaults to the canonical ``isotonic.json`` but
    must be overridden by the caller when ``weather.isotonic_path`` points
    at a non-default artifact (e.g. ``isotonic_v2.json`` during a
    bounded-exposure refit pilot). Otherwise this check reads the stale
    canonical file and reports plateau bounds from a calibrator the bot
    isn't actually serving — see 2026-05-22 isotonic_v2 cutover.
    """
    path = calibration_dir / isotonic_filename
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        logger.warning("tripwire: unreadable isotonic artifact %s", path, exc_info=True)
        return []

    if not isinstance(raw, dict) or "breakpoints" not in raw:
        return []

    try:
        breakpoints = [float(x) for x in raw["breakpoints"]]
        values = [float(x) for x in raw["values"]]
    except (KeyError, TypeError, ValueError):
        logger.warning("tripwire: malformed isotonic breakpoints in %s", path, exc_info=True)
        return []

    if len(breakpoints) != len(values) or len(breakpoints) < 2:
        return []

    # Use the shared reader so a nested-``metadata.fit_date`` artifact (the
    # favorite_bias shape) resolves. A bare ``raw.get("fit_date")`` yielded the
    # literal "unknown", which made the code byte-identical across refits and
    # therefore unsafe as an acknowledgement key. IsotonicCalibrator carries
    # the same nested-schema fix.
    fit_date_obj = _read_fit_date(calibration_dir / isotonic_filename)
    fit_date = fit_date_obj.isoformat() if fit_date_obj is not None else "unknown"

    alerts: list[DriftAlert] = []
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        if j > i:
            low = breakpoints[i]
            high = breakpoints[j]
            value = values[i]
            width = high - low
            center = (low + high) / 2.0
            deviation = abs(value - center)
            if width >= min_width and deviation >= min_center_deviation:
                alerts.append(
                    DriftAlert(
                        code=f"calibrator_plateau:{fit_date}:{low:.3f}-{high:.3f}",
                        severity=AlertLevel.WARNING,
                        message=(
                            f"Isotonic plateau [{low:.3f}, {high:.3f}] maps to value "
                            f"{value:.3f} (width={width:.3f}, "
                            f"center_deviation={deviation:.3f}) — "
                            f"many raw probabilities collapse to one calibrated value "
                            f"and will sign-flip against markets priced near "
                            f"({low:.2f} + {high:.2f}) / 2"
                        ),
                        detail={
                            "fit_date": fit_date,
                            "plateau_low": low,
                            "plateau_high": high,
                            "plateau_value": value,
                            "width": width,
                            "center_deviation": deviation,
                        },
                        owner="weather_edge",
                        ack_key=(fit_date if fit_date != "unknown" else None),
                    )
                )
        i = j + 1
    return alerts


def run_all(
    *,
    state_dir: Path,
    calibrator_paths: dict[str, Path],
    config: TripwireConfig,
    now: datetime | None = None,
    calibration_dir: Path | None = None,
    bias_table_path: Path | None = None,
    bias_table_missing_is_error: bool = False,
    isotonic_filename: str = "isotonic.json",
) -> list[DriftAlert]:
    """Evaluate every tripwire and return the flat list of alerts.

    ``calibrator_paths`` maps a human label (e.g. ``"per_type_platt"``)
    to the artifact's JSON path. Missing entries are silent; the
    orchestrator is responsible for deciding which calibrators it
    wants tripwired.

    ``calibration_dir`` is the directory holding the isotonic artifact
    for the plateau-coverage check. ``None`` skips that check.
    ``isotonic_filename`` selects the artifact within that directory —
    callers serving a non-default isotonic path (see
    ``weather.isotonic_path``) must pass the matching filename so the
    plateau check sees the live calibrator's geometry, not the canonical
    file's stale bounds.

    ``bias_table_path`` is the GFS bias-correction artifact for the
    mtime-based freshness check. ``None`` skips that check.
    ``bias_table_missing_is_error`` is forwarded to
    ``check_bias_table_freshness``: pass ``True`` when the caller has
    already established the table must exist (correction enabled + a
    resolved path) so a missing file raises CRITICAL instead of staying
    silent.
    """
    now = now or datetime.now(UTC)
    today = now.date()
    alerts: list[DriftAlert] = []
    alerts.extend(
        check_active_file_staleness(
            state_dir,
            stale_seconds=config.stale_active_seconds,
            now=now,
        )
    )
    alerts.extend(check_rotation_regression(state_dir, now=now))
    for label, path in calibrator_paths.items():
        alerts.extend(
            check_calibrator_artifact_staleness(
                path,
                stale_days=config.stale_calibrator_days,
                today=today,
                artifact_label=label,
            )
        )
    if calibration_dir is not None:
        alerts.extend(
            check_calibrator_plateau_coverage(
                calibration_dir,
                isotonic_filename=isotonic_filename,
            )
        )
    if bias_table_path is not None:
        alerts.extend(
            check_bias_table_freshness(
                bias_table_path,
                stale_days=config.stale_bias_table_days,
                now=now,
                missing_is_error=bias_table_missing_is_error,
            )
        )
    return alerts


# Prefix shared by every plateau alert code. Defined here (the producer) so
# consumers can import and reference it rather than duplicating the string.
PLATEAU_ALERT_CODE_PREFIX = "calibrator_plateau:"

# State file that persists which plateau codes have already been dispatched to
# Telegram, stored in the same state_dir as trades.jsonl.
_PLATEAU_SEEN_FILENAME = "tripwire_plateau_seen.json"
_PLATEAU_SEEN_SCHEMA_VERSION = 1


def load_plateau_seen_codes(state_dir: Path) -> set[str]:
    """Return the set of plateau alert codes previously dispatched to Telegram.

    Returns an empty set if the file is missing, unreadable, corrupt, or uses
    an unknown schema version — the caller treats unknown codes as new and will
    re-persist after dispatch.
    """
    path = state_dir / _PLATEAU_SEEN_FILENAME
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return set()
    except (OSError, json.JSONDecodeError):
        logger.warning("tripwire: unreadable plateau-seen file %s", path, exc_info=True)
        return set()
    if raw.get("_schema_version") != _PLATEAU_SEEN_SCHEMA_VERSION:
        logger.warning(
            "tripwire: plateau-seen file %s has unknown schema version %s, ignoring",
            path,
            raw.get("_schema_version"),
        )
        return set()
    codes = raw.get("codes")
    if not isinstance(codes, list):
        return set()
    return {c for c in codes if isinstance(c, str)}


def _atomic_write(path: Path, payload: str) -> None:
    """Atomically write ``payload`` to ``path``; log and swallow any ``OSError``.

    Shared by every state-file writer in this module.
    """
    try:
        atomic_write_text(path, payload)
    except OSError:
        logger.warning("tripwire: failed to atomically write %s", path, exc_info=True)


# State file holding BOTH machine-dedup and human-acknowledgement records.
# Two namespaces, two meanings — never conflate them:
#   ``seen``  — codes already DISPATCHED to Telegram. Written automatically
#               by ``persist_seen_code`` after every successful dispatch.
#   ``acked`` — codes a HUMAN has reviewed and dismissed via the /ack
#               Telegram command. Written ONLY by ``persist_ack_code``.
_ACKS_FILENAME = "alert_acks.json"
_ACKS_SCHEMA_VERSION = 2


def load_acks(state_dir: Path) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return ``(seen, acked)`` from the alert-acknowledgement file.

    Two namespaces with two meanings. ``seen`` is machine dedup — codes
    already dispatched. ``acked`` is a HUMAN verdict recorded via /ack,
    carrying the owner and the severity it was acknowledged at.

    A v1 ``tripwire_plateau_seen.json`` migrates into ``seen`` and NEVER
    into ``acked``: every code in it got there via an unconditional persist
    after dispatch, so no human reviewed any of them. Once a v2
    ``alert_acks.json`` exists it is authoritative and the v1 file is
    ignored (it is never deleted, so a rollback can still read it).

    Never raises — a missing, unreadable, corrupt, or unknown-schema file
    degrades to empty, which re-fires everything rather than silencing
    anything.
    """
    path = state_dir / _ACKS_FILENAME
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return (load_plateau_seen_codes(state_dir), {})
    except (OSError, json.JSONDecodeError):
        logger.warning("tripwire: unreadable ack file %s", path, exc_info=True)
        return (set(), {})
    if not isinstance(raw, dict):
        logger.warning("tripwire: ack file %s is not a JSON object, ignoring", path)
        return (set(), {})
    if raw.get("_schema_version") != _ACKS_SCHEMA_VERSION:
        logger.warning("tripwire: ack file %s has unknown schema, ignoring", path)
        return (set(), {})
    seen = {c for c in raw.get("seen", []) if isinstance(c, str)}
    acked_raw = raw.get("acked")
    acked = acked_raw if isinstance(acked_raw, dict) else {}
    return (seen, acked)


def persist_seen_code(state_dir: Path, code: str) -> None:
    """Atomically add ``code`` to the ``seen`` (machine-dedup) set.

    Idempotent — adding a code that already exists is a no-op. Preserves
    the existing ``acked`` map untouched.

    Multiple callers are permitted: the tripwire sweep (this function) and
    the Telegram /ack handler (``persist_ack_code``) both write
    ``alert_acks.json`` on the same asyncio event loop. Safety depends on
    both staying SYNCHRONOUS — do not add ``await`` between the read and
    the write in either function. A synchronous read-modify-write cannot be
    interleaved by another task on that loop, which is the only reason two
    writers are safe here.
    """
    seen, acked = load_acks(state_dir)
    if code in seen:
        return
    updated_seen = sorted(seen | {code})
    payload = json.dumps(
        {
            "_schema_version": _ACKS_SCHEMA_VERSION,
            "seen": updated_seen,
            "acked": acked,
        },
        separators=(",", ":"),
    )
    _atomic_write(state_dir / _ACKS_FILENAME, payload)


def persist_ack_code(
    state_dir: Path,
    code: str,
    *,
    owner: str | None,
    severity: str,
    owner_live_at_ack: bool,
) -> None:
    """Atomically record a HUMAN acknowledgement of ``code``.

    Preserves the existing ``seen`` set untouched. Overwrites any prior
    acknowledgement of the same code (re-acking updates ``ts``/``owner``/
    ``severity``/``owner_live_at_ack``).

    ``owner_live_at_ack`` records whether ``owner`` was configured to trade
    at the moment of acknowledgement. ``Orchestrator._ack_suppresses`` uses
    it for a TRANSITION test, not a point-in-time one: an ack taken while
    the owner was live still suppresses while the owner stays live (this is
    the common case — acking a stale-calibrator alert for a strategy that is
    live *right now*, e.g. favorite_bias); only an ack taken while the owner
    was NOT live is dropped once the owner transitions to live, because that
    judgment was made in a no-money context and must not silently carry over
    once real money is on the line.

    MUST STAY SYNCHRONOUS. The Telegram /ack handler and the tripwire loop
    are two writers on the same event loop; a synchronous read-modify-write
    cannot be interleaved by another task, which is what makes that safe.
    Adding an ``await`` between the read and the write reintroduces a
    lost-update race.
    """
    seen, acked = load_acks(state_dir)
    acked = dict(acked)
    acked[code] = {
        "ts": datetime.now(UTC).isoformat(),
        "owner": owner,
        "severity": severity,
        "owner_live_at_ack": owner_live_at_ack,
    }
    payload = json.dumps(
        {
            "_schema_version": _ACKS_SCHEMA_VERSION,
            "seen": sorted(seen),
            "acked": acked,
        },
        separators=(",", ":"),
    )
    _atomic_write(state_dir / _ACKS_FILENAME, payload)


__all__ = [
    "DEFAULT_PLATEAU_MIN_CENTER_DEVIATION",
    "DEFAULT_PLATEAU_MIN_WIDTH",
    "DEFAULT_STALE_ACTIVE_SECONDS",
    "DEFAULT_STALE_BIAS_TABLE_DAYS",
    "DEFAULT_STALE_CALIBRATOR_DAYS",
    "PLATEAU_ALERT_CODE_PREFIX",
    "STREAM_NAMES",
    "TripwireConfig",
    "check_active_file_staleness",
    "check_bias_table_freshness",
    "check_calibrator_artifact_staleness",
    "check_calibrator_plateau_coverage",
    "check_rotation_regression",
    "load_acks",
    "load_plateau_seen_codes",
    "persist_ack_code",
    "persist_seen_code",
    "run_all",
]
