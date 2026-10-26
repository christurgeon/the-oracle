"""Tests for python/utils/atomic_write.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from python.utils.atomic_write import atomic_write_text


def test_writes_content_and_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    atomic_write_text(target, '{"a": "é"}')
    assert target.read_text(encoding="utf-8") == '{"a": "é"}'
    assert list(target.parent.glob("*.tmp")) == []


def test_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_failed_replace_keeps_original_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("old")

    def _boom(src: str, dst: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def _record_fsyncs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def _spy(fd: int) -> None:
        calls.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    return calls


def test_no_fsync_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_fsyncs(monkeypatch)
    atomic_write_text(tmp_path / "state.json", "x")
    assert calls == []


def test_fsync_syncs_file_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_fsyncs(monkeypatch)
    atomic_write_text(tmp_path / "state.json", "x", fsync=True)
    assert len(calls) == 2
    assert (tmp_path / "state.json").read_text() == "x"


def test_directory_fsync_failure_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.fsync
    calls: list[int] = []

    def _fail_second(fd: int) -> None:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("EINVAL")
        real(fd)

    monkeypatch.setattr(os, "fsync", _fail_second)
    atomic_write_text(tmp_path / "state.json", "x", fsync=True)
    assert (tmp_path / "state.json").read_text() == "x"


def test_file_fsync_failure_keeps_original_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("old")

    def _boom(fd: int) -> None:
        raise OSError("EIO")

    monkeypatch.setattr(os, "fsync", _boom)
    with pytest.raises(OSError, match="EIO"):
        atomic_write_text(target, "new", fsync=True)
    assert target.read_text() == "old"
    assert list(tmp_path.glob("*.tmp")) == []
