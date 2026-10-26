"""VPS pull helpers shared by shadow-dashboard + settlement scripts.

Defines the remote host, state directory, the set of diagnostic audit
files that may live there, and a best-effort pull routine that skips
missing remote files without aborting.
"""

from __future__ import annotations

import os
import subprocess
from datetime import date as _date
from pathlib import Path

from dotenv import dotenv_values

_DOTENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


# ``user@host`` for ssh/scp. Kept out of the repo: set it in the environment or .env.
VPS_HOST = _env("ORACLE_VPS_HOST", "")
VPS_STATE_DIR = _env("ORACLE_VPS_STATE_DIR", "/opt/oracle/state")
VPS_TRADES_PATH = f"{VPS_STATE_DIR}/trades.jsonl"


def require_vps_host() -> str:
    if not VPS_HOST:
        raise SystemExit(
            "VPS access needs ORACLE_VPS_HOST (e.g. user@host) in the environment or .env"
        )
    return VPS_HOST

# Diagnostic audit streams pulled from the VPS for analytics and post-mortems.
# Missing files are non-fatal: any given file may not exist yet if the bot has
# not been redeployed with the instrumented build.
#
# NOTE: ``candidate_audit.jsonl`` is intentionally included here (for offline
# analytics such as bracket_validation.py and pull_post_mortem_data.py) even
# though it is excluded from ``audit_tripwires.STREAM_NAMES`` (the health-
# monitoring set). The two lists serve different purposes: VPS_AUDIT_FILES is
# "what to pull for analysis"; STREAM_NAMES is "what to check for liveness."
# ``candidate_audit.jsonl`` is activity-gated (only written when markets clear
# Stage-2) and therefore not a reliable health indicator.
VPS_AUDIT_FILES: tuple[str, ...] = (
    "eval_audit.jsonl",
    "candidate_audit.jsonl",
    "scan_funnel.jsonl",
    "forecasts_raw.jsonl",
    # Phase 0 intraday-EMOS-HIGH passive depth probe sink — written by
    # ``python/data/phase0_depth_monitor.py`` via ``ForecastsRawWriter``;
    # daily UTC midnight rotation produces siblings of the form
    # ``phase0_depth_probes-YYYY-MM-DD.jsonl``. Pulled here so the
    # daily-report Phase 0 summary renders on the operator's laptop and
    # not just on the VPS.
    "phase0_depth_probes.jsonl",
    # favorite_bias decision audit — O_APPEND, no rotation (NOT in
    # ``_FORECASTS_RAW_WRITER_ROTATIONS``). Consumed by
    # ``scripts/favorite_bias_report.py`` for the realized-vs-g tripwire.
    "favorite_bias_audit.jsonl",
    # KXRAIN forward shadow corpus — five ForecastsRawWriter streams, all
    # rotating with the dash convention. The corpus cannot be rebuilt if lost.
    "kxrain_decisions.jsonl",
    "kxrain_book.jsonl",
    "kxrain_trades.jsonl",
    "kxrain_ticks.jsonl",
    "kxrain_settlements.jsonl",
)

# KXRAIN's two durable JSON artifacts. Not JSONL, never rotated, and not in
# VPS_AUDIT_FILES since ``_rotated_name`` has no convention for them — they
# are state, not corpus, tracking consumed virtual entries and recorded
# settlements so an offline replay agrees with the VPS.
KXRAIN_STATE_ARTIFACTS: tuple[str, ...] = (
    "kxrain_ledger.json",
    "kxrain_settled_index.json",
)

# Filenames that rotate via ``python.utils.jsonl.ForecastsRawWriter``'s
# ``<stem>-<YYYY-MM-DD><suffix>`` convention (dash before date, full
# extension preserved). Anything else falls back to the legacy
# ``<name>.<YYYY-MM-DD>`` (dot suffix) convention used by the audit-log
# writer. Listed in code (not regex-detected) so a future stream addition
# fails loudly if it forgets to register its rotation pattern.
_FORECASTS_RAW_WRITER_ROTATIONS: frozenset[str] = frozenset({
    "forecasts_raw.jsonl",
    "phase0_depth_probes.jsonl",
    "kxrain_decisions.jsonl",
    "kxrain_book.jsonl",
    "kxrain_trades.jsonl",
    "kxrain_ticks.jsonl",
    "kxrain_settlements.jsonl",
})


def _rotated_name(fname: str, iso: str) -> str:
    """Return the rotated-sibling filename for a daily-rotated state stream.

    Two conventions coexist on disk:
      * ``eval_audit.jsonl``, ``candidate_audit.jsonl``, ``scan_funnel.jsonl``
        rotate as ``{name}.YYYY-MM-DD`` (dot suffix).
      * Files in ``_FORECASTS_RAW_WRITER_ROTATIONS``
        (``forecasts_raw.jsonl``, ``phase0_depth_probes.jsonl``) rotate as
        ``<stem>-YYYY-MM-DD.jsonl`` (dash before date, ``.jsonl`` suffix
        preserved). See ``python/utils/jsonl.py:ForecastsRawWriter`` for
        the writer contract.

    Locking this down in code rather than at call sites prevents a future
    stream addition from silently dropping rotated siblings.
    """
    if fname in _FORECASTS_RAW_WRITER_ROTATIONS:
        stem, _, suffix = fname.rpartition(".")
        return f"{stem}-{iso}.{suffix}"
    return f"{fname}.{iso}"


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def pull_rotated_audit_files(target_date: _date, dest_dir: Path) -> list[str]:
    """SCP both active and rotated-sibling audit files for ``target_date``.

    Audit logs rotate at first UTC-midnight write (audit_log.py):
    yesterday's data lives at ``state/{stream}.jsonl.{YYYY-MM-DD}``, not
    in the active file. We pull both: the active file (in case rotation
    hasn't happened yet because the bot is idle) and the rotated sibling
    (the canonical source for a past date).

    Returns a list of warning strings for any SCPs that failed (typically
    "rotated sibling pruned beyond retention window" for dates older than
    ``DEFAULT_RETENTION_DAYS=14``).
    """
    host = require_vps_host()
    dest_dir.mkdir(parents=True, exist_ok=True)
    iso = target_date.isoformat()
    warnings: list[str] = []
    print(f"Pulling audit streams for {iso} from VPS (active + rotated)...")
    for fname in VPS_AUDIT_FILES:
        for remote_name in (fname, _rotated_name(fname, iso)):
            remote = f"{host}:{VPS_STATE_DIR}/{remote_name}"
            dest = dest_dir / remote_name
            r = subprocess.run(
                ["scp", remote, str(dest)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                warnings.append(f"{remote_name}: {r.stderr.strip()}")
                print(f"  !! skipped {remote_name}: {r.stderr.strip()}")
                continue
            print(f"  -> {_count_lines(dest)} rows saved to {dest}")
    return warnings


def pull_audit_files(dest_dir: Path) -> None:
    """SCP the three audit JSONLs into ``dest_dir``, printing progress.

    Per-file scp (not batched) because batched scp aborts the whole
    transfer if any one source is missing — and on an unmodernised VPS
    these files legitimately may not exist yet.
    """
    host = require_vps_host()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("Pulling audit streams from VPS...")
    for fname in VPS_AUDIT_FILES:
        remote = f"{host}:{VPS_STATE_DIR}/{fname}"
        dest = dest_dir / fname
        r = subprocess.run(
            ["scp", remote, str(dest)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  !! skipped {fname}: {r.stderr.strip()}")
            continue
        print(f"  -> {_count_lines(dest)} rows saved to {dest}")


SSH_COMMAND_TIMEOUT_SECONDS = 30


class SshFailure(RuntimeError):  # noqa: N818
    """Raised when ``ssh_command`` fails (non-zero exit, timeout, or transport error)."""


def ssh_command(remote_cmd: str, *, timeout: int = SSH_COMMAND_TIMEOUT_SECONDS) -> str:
    """Run ``remote_cmd`` on the VPS via ssh, returning stdout.

    Raises ``SshFailure`` on non-zero exit, timeout, or transport error.
    Caller is responsible for wrapping in best-effort try/except where
    a failure should not abort the larger job (e.g., operational-status
    capture).
    """
    try:
        r = subprocess.run(
            ["ssh", require_vps_host(), remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise SshFailure(f"ssh timeout after {timeout}s: {remote_cmd!r}") from e
    except OSError as e:
        raise SshFailure(f"ssh transport error: {e}") from e
    if r.returncode != 0:
        raise SshFailure(
            f"ssh returncode={r.returncode}: {r.stderr.strip() or r.stdout.strip()}"
        )
    return r.stdout
