"""Atomic file writes via a same-directory temp file + ``os.replace``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, fsync: bool = False) -> None:
    """Write *content* to *path* atomically; readers see the old or new file, never a partial one.

    Creates the parent directory if needed. The temp file lives in the target
    directory so ``os.replace`` stays on one filesystem. Removes the temp file
    and re-raises on any failure, including ``KeyboardInterrupt``.

    ``fsync=True`` also makes the write durable across power loss: it fsyncs the
    temp file before the rename and the parent directory after it. The directory
    fsync is best-effort because some filesystems (e.g. NFS) reject it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode())
        if fsync:
            os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp, path)
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
