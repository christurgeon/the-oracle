"""EmosTable loader + smoke + atomicity + hermeticity tests."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from python.data.emos import EmosTable


def _valid_payload() -> dict:
    return {
        "fit_date": "2026-05-04",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": -4.0, "b": 1.05, "c": 15.0, "d": 0.4,
                "n_train": 127, "leads_pooled": [0, 1, 2],
                "low_n": False, "ci_warnings": [], "ens_std_p99": 6.5,
            },
        },
    }


def test_load_valid_artifact(tmp_path: Path):
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(_valid_payload()))
    table = EmosTable.load(p)
    assert table.is_loaded
    assert table.fallback_reason is None
    assert table.artifact_hash  # SHA-256 set
    assert table.artifact_loaded_at  # UTC ISO


def test_load_missing_file_returns_disabled_table(tmp_path: Path):
    table = EmosTable.load(tmp_path / "nope.json")
    assert not table.is_loaded
    assert table.fallback_reason == "file-missing"


def test_load_corrupt_json_returns_disabled_table(tmp_path: Path):
    p = tmp_path / "coefficients.json"
    p.write_text("{not valid json")
    table = EmosTable.load(p)
    assert not table.is_loaded
    assert table.fallback_reason and table.fallback_reason.startswith("parse-error")


def test_load_bounds_violation_returns_disabled_table(tmp_path: Path):
    payload = _valid_payload()
    payload["cells"]["AUS/high"]["b"] = 2.0  # out of [0.3, 1.5]
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)
    assert not table.is_loaded
    assert table.fallback_reason and table.fallback_reason.startswith("bounds-error")


def test_smoke_test_passes_on_valid_payload(tmp_path: Path):
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(_valid_payload()))
    table = EmosTable.load(p)
    assert table.smoke_test() is True


def test_smoke_test_logs_info_on_large_intercept(tmp_path: Path, caplog):
    payload = _valid_payload()
    payload["cells"]["AUS/high"]["a"] = 60.0  # > 50 INFO threshold
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)

    caplog.set_level(logging.INFO, logger="python.data.emos")
    table.smoke_test()
    # Soft sanity check fired (does NOT cause smoke to fail)
    assert any(
        "AUS/high" in r.message and "intercept" in r.message.lower()
        for r in caplog.records
    )


def test_artifact_signature_uses_mtime_size_inode(tmp_path: Path):
    p = tmp_path / "coefficients.json"
    p.write_text(json.dumps(_valid_payload()))
    table = EmosTable.load(p)
    sig = table.artifact_signature
    assert sig is not None
    assert len(sig) == 3  # (st_mtime_ns, st_size, st_ino)


def test_emos_module_imports_only_allowed_packages():
    """Hermeticity invariant: python.data.emos imports only stdlib, numpy,
    scipy, and python.data.emos itself."""
    import ast
    from pathlib import Path

    src = Path("python/data/emos.py").read_text()
    tree = ast.parse(src)

    allowed_top_level = {
        "hashlib", "json", "logging", "math", "os", "pathlib",
        "datetime", "dataclasses", "typing", "collections",
        "numpy", "scipy",
        "__future__",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in allowed_top_level, f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            # python.data.emos may not import from anywhere else under python.*
            assert top in allowed_top_level or module == "python.data.emos", (
                f"Disallowed `from {module} import ...` in python/data/emos.py"
            )
