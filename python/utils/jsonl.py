"""Tolerant JSONL reader used by shadow-log consumers.

Callers parse logs written by the execution engine across many process
lifetimes, so the reader must survive unclean shutdowns (truncated last
line, partial JSON) and rotations (missing files) without raising.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["iter_jsonl", "ForecastsRawWriter", "PerDayCapBlockWriter"]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file.

    Blank lines, malformed JSON, and a missing file are all non-fatal —
    they emit a debug log and are skipped.  Non-object JSON values
    (arrays, scalars) are skipped too; callers always want dicts.
    """
    try:
        handle = open(path)
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("iter_jsonl: failed to open %s", path, exc_info=True)
        return

    with handle as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("iter_jsonl: skipping malformed line in %s", path)
                continue
            if isinstance(entry, dict):
                yield entry


class ForecastsRawWriter:
    """Append-only JSONL writer for ``state/forecasts_raw.jsonl``.

    Mirrors :class:`python.analytics.audit_log.AuditLogger`: lazy daily
    rotation on UTC midnight (per the ``clock`` parameter default; intentionally
    distinct from the ET midnight boundary used by ``RiskManager.reset_daily()``),
    retention pruning, and per-record atomic ``os.write`` appends.

    Pattern: ``os.open(O_WRONLY|O_APPEND|O_CREAT) + os.write`` per record.
    ``O_APPEND`` makes ``write(2)`` atomic with respect to the file's size
    up to ``PIPE_BUF`` (~4 KB on Linux). Rows must serialize to <= 4096
    bytes — see ``tests/test_forecasts_raw_writer.py::
    test_each_row_payload_under_pipe_buf``.

    Cancel-safe by construction: each ``os.write`` is one syscall, so a
    ``CancelledError`` between rows produces zero or more complete lines,
    never a torn line.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        retention_days: int = 14,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_days = int(retention_days)
        self._clock = clock
        self._lock = threading.Lock()
        # Seed the rotation boundary from the existing file's mtime so a
        # restart that crosses a UTC day boundary still rotates correctly
        # on the first new write.
        self._current_date: date | None = (
            date.fromtimestamp(self._path.stat().st_mtime) if self._path.exists() else None
        )
        self.success_count: int = 0
        self.last_error_at: float | None = None

    def write_batch(self, rows: list[dict[str, Any]]) -> None:
        """Append a batch of rows. Synchronous — call from ``asyncio.to_thread``.

        Single ``os.open`` per batch (cheaper than re-open per row); each
        ``os.write`` is independently atomic up to ``PIPE_BUF``. Errors are
        logged and swallowed: ``self.last_error_at`` is set so a monitor can
        detect persistent failures, but a side-stream write must never crash
        the trading loop.
        """
        if not rows:
            return
        now = self._clock()
        with self._lock:
            self._maybe_rotate(now.date())
            try:
                fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            except OSError as exc:
                self.last_error_at = time.time()
                logger.error("forecasts_raw open failed: %s", exc)
                return
            try:
                for row in rows:
                    payload = (json.dumps(row, default=str) + "\n").encode("utf-8")
                    os.write(fd, payload)
                    self.success_count += 1
            except OSError as exc:
                self.last_error_at = time.time()
                logger.error("forecasts_raw write failed: %s", exc)
            finally:
                os.close(fd)

    def _maybe_rotate(self, today: date) -> None:
        """Rotate the live file on the first write past UTC midnight.

        Renames the live file to ``<stem>-<YYYY-MM-DD><suffix>`` and prunes
        rotated siblings whose ISO date suffix is strictly older than
        ``today - retention_days``. The retention sweep also runs when the
        boundary is being seeded for the first time so that stale siblings
        left behind by a long process gap are cleaned up on the next write.
        """
        if self._current_date is None:
            # First write: seed the boundary and run a retention sweep so any
            # stale siblings left from a prior run get pruned.
            self._current_date = today
            self._prune_old(today)
            return

        if today <= self._current_date:
            return

        rotated = self._path.with_name(
            f"{self._path.stem}-{self._current_date.isoformat()}{self._path.suffix}"
        )
        try:
            if self._path.exists():
                os.replace(self._path, rotated)
        except OSError:
            logger.exception("forecasts_raw: rotation failed for %s", self._path)
        self._current_date = today
        self._prune_old(today)

    def _prune_old(self, today: date) -> None:
        """Delete rotated siblings older than ``today - retention_days``."""
        cutoff = date.fromordinal(today.toordinal() - self._retention_days)
        prefix = f"{self._path.stem}-"
        for sibling in self._path.parent.glob(f"{self._path.stem}-*{self._path.suffix}"):
            try:
                date_str = sibling.stem[len(prefix):]
                d = date.fromisoformat(date_str)
            except ValueError:
                # Unrecognized suffix — leave it alone.
                continue
            if d < cutoff:
                try:
                    sibling.unlink()
                except OSError:
                    logger.exception(
                        "forecasts_raw: failed to prune %s during retention sweep",
                        sibling,
                    )


class PerDayCapBlockWriter:
    """Append-only JSONL writer for ``state/per_day_cap_blocks.jsonl``.

    Mirrors :class:`ForecastsRawWriter`: lazy daily UTC rotation (only
    consequential for retention pruning — the file path itself does not
    rotate), retention pruning via ``retention_days``, per-record atomic
    ``os.write`` append (PIPE_BUF guarantee for rows <= 4096 bytes).

    Rows record blocked weather-edge candidates whose ``(city, var,
    forecast_date)`` event key has already hit the per-day budget.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        retention_days: int = 14,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._clock = clock
        self._lock = threading.Lock()
        self._last_prune_date: date | None = None
        self._last_error_at: float | None = None

    def write(self, row: dict[str, Any]) -> None:
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            self._maybe_prune()
            try:
                fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
            except OSError:
                self._last_error_at = time.time()
                logger.warning(
                    "PerDayCapBlockWriter: write failed for %s", self._path, exc_info=True
                )

    def _maybe_prune(self) -> None:
        today = self._clock().date()
        if self._last_prune_date == today:
            return
        self._last_prune_date = today
        if not self._path.exists():
            return
        cutoff_ts = time.time() - (self._retention_days * 86400)
        try:
            if self._path.stat().st_mtime < cutoff_ts:
                self._path.unlink(missing_ok=True)
        except OSError:
            logger.warning("PerDayCapBlockWriter: prune failed for %s", self._path, exc_info=True)
