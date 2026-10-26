"""End-to-end smoke test for scripts/fit_emos.py against the committed corpus."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "emos" / "corpus.parquet"


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus.parquet not present")
def test_fit_emos_smoke(tmp_path: Path):
    """Run the fitter on the real corpus and check pinned outputs.

    As of 2026-05-11 the corpus has 14 cells, all fitted (SEA/low crossed
    n>=30 with the corpus rebuild that landed alongside the Stage A cutover).
    SEA/low is still flagged ``low_n=True`` and may carry CI warnings; it is
    not eligible for the EMOS allowlist until the per-cell promotion gate
    passes against isotonic.
    """
    out_dir = tmp_path / "emos_out"
    out_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fit_emos",
            "--corpus",
            str(CORPUS),
            "--output-dir",
            str(out_dir),
            "--bootstrap-iters",
            "10",  # keep test fast
            "--seed",
            "20260503",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    # Stdout report exists.
    assert "EMOS fit" in result.stdout

    coef_path = out_dir / "coefficients.json"
    diag_path = out_dir / "coefficients.diagnostics.json"
    assert coef_path.exists()
    assert diag_path.exists()

    coef = json.loads(coef_path.read_text())
    assert coef["schema_version"] == 1
    assert coef["lead_handling"] == "pooled"
    assert "fit_date" in coef and "training_window_days" in coef

    cells = coef["cells"]
    fitted = [k for k, v in cells.items() if not v.get("skipped")]
    skipped = [k for k, v in cells.items() if v.get("skipped")]
    assert len(fitted) == 14, f"expected 14 fitted cells, got {fitted}"
    assert skipped == [], f"expected no skipped cells, got {skipped}"

    # Each fitted cell has all coefficient keys.
    for key, body in cells.items():
        if body.get("skipped"):
            assert "reason" in body
            assert "below floor" in body["reason"]
            continue
        for field in ("a", "b", "c", "d", "n_train", "leads_pooled", "low_n", "ci_warnings"):
            assert field in body, f"{key} missing {field}"
        # b is enforced into [0.3, 1.5]
        assert 0.3 <= body["b"] <= 1.5
        assert body["c"] >= 1e-6
        assert body["d"] >= 0.0


def test_coefficients_write_is_atomic(tmp_path: Path) -> None:
    """If a reader observes the file mid-write, it sees either the prior
    valid JSON or the new valid JSON — never a truncated/partial body."""
    from scripts.fit_emos import _atomic_write

    target = tmp_path / "coefficients.json"
    target.write_text(json.dumps({"prior": True, "schema_version": 1}))

    new_payload = {"new": True, "schema_version": 1, "cells": {f"X/{i}": {} for i in range(5000)}}
    new_text = json.dumps(new_payload, indent=2)

    errors: list[str] = []

    def reader() -> None:
        for _ in range(200):
            try:
                json.loads(target.read_text())
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                return

    t = threading.Thread(target=reader)
    t.start()
    _atomic_write(target, new_text)
    t.join(timeout=5)

    assert not errors, f"Reader observed torn JSON: {errors[0]}"
    assert json.loads(target.read_text()) == new_payload


def test_every_fitted_cell_has_finite_ens_std_p99(tmp_path: Path) -> None:
    """Every cell in coefficients.json carries a finite ens_std_p99 field.

    The runtime EMOS loader's `EmosTable.training_ens_std_p99(...)` consumes
    this per-cell value to flag regime-change extrapolation at runtime. The
    runtime cannot recompute it (corpus.parquet is outside the runtime
    hermeticity allowlist), so the fitter must persist it during fitting.
    """
    coeffs = json.loads((REPO / "data" / "emos" / "coefficients.json").read_text())
    cells = coeffs["cells"]
    assert cells, "coefficients.json must have at least one fitted cell"
    for cell_key, cell in cells.items():
        assert "ens_std_p99" in cell, f"{cell_key} missing ens_std_p99"
        v = cell["ens_std_p99"]
        assert isinstance(v, (int, float)) and 0.0 < v < 100.0, (
            f"{cell_key}: ens_std_p99 must be finite + plausible (0-100°F), got {v!r}"
        )


def test_archive_written_after_atomic_write(tmp_path: Path, monkeypatch):
    """After a fit, data/emos/coefficients_history/<sha256>.json exists."""
    import hashlib

    from scripts.fit_emos import _archive_coefficients, _atomic_write

    target = tmp_path / "coefficients.json"
    payload = json.dumps({"schema_version": 1, "cells": {}}, indent=2)
    _atomic_write(target, payload)

    history_dir = tmp_path / "coefficients_history"
    _archive_coefficients(target, history_dir, retention=180)

    expected_hash = hashlib.sha256(payload.encode()).hexdigest()
    assert (history_dir / f"{expected_hash}.json").exists()


def test_archive_retention_keeps_only_last_n(tmp_path: Path):
    """Archive sweep deletes files past the N-th newest by mtime."""
    import os
    import time

    from scripts.fit_emos import _archive_coefficients

    history_dir = tmp_path / "coefficients_history"
    history_dir.mkdir()
    # Create 200 fake archives with increasing mtime (file 0 oldest, file 199 newest).
    for i in range(200):
        f = history_dir / f"{i:064x}.json"
        f.write_text("{}")
        ts = time.time() - (200 - i)
        os.utime(f, (ts, ts))

    target = tmp_path / "coefficients.json"
    target.write_text("{}")
    _archive_coefficients(target, history_dir, retention=180)

    remaining = list(history_dir.glob("*.json"))
    # 180 by retention + at most 1 just-written copy (overlap if hash collision with existing)
    assert 180 <= len(remaining) <= 181
