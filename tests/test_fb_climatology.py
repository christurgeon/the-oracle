from __future__ import annotations

import json
import math

from python.data.fb_climatology import (
    ClimEntry,
    FbClimatologyTable,
    clim_yes_prob,
    strike_z,
)

# ---- pure math ----


def test_strike_z_basic() -> None:
    assert strike_z(99.0, 85.0, 4.0) == (99.0 - 85.0) / 4.0


def test_strike_z_none_on_bad_sigma_or_missing() -> None:
    assert strike_z(99.0, 85.0, 0.0) is None
    assert strike_z(99.0, 85.0, -1.0) is None
    assert strike_z(None, 85.0, 4.0) is None
    assert strike_z(99.0, None, 4.0) is None


def test_clim_yes_prob_less_and_greater() -> None:
    assert math.isclose(clim_yes_prob(85.0, "less", 85.0, 4.0), 0.5, abs_tol=1e-9)
    assert math.isclose(clim_yes_prob(85.0, "greater", 85.0, 4.0), 0.5, abs_tol=1e-9)
    assert clim_yes_prob(99.0, "less", 85.0, 4.0) > 0.99
    assert clim_yes_prob(99.0, "greater", 85.0, 4.0) < 0.01


def test_clim_yes_prob_none_paths() -> None:
    assert clim_yes_prob(99.0, "between", 85.0, 4.0) is None
    assert clim_yes_prob(99.0, "less", 85.0, 0.0) is None
    assert clim_yes_prob(None, "less", 85.0, 4.0) is None


def test_clim_entry_is_frozen() -> None:
    e = ClimEntry(mu=85.0, sigma=4.0, n=30, source="month")
    assert (e.mu, e.sigma, e.n, e.source) == (85.0, 4.0, 30, "month")


# ---- loader (fail-open) + lookup fallback ----


def _write(tmp_path, payload) -> str:
    p = tmp_path / "clim.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_loader_missing_file_is_disabled_not_raising(tmp_path) -> None:
    t = FbClimatologyTable.load(tmp_path / "does_not_exist.json")
    assert t.is_loaded is False
    assert t.fallback_reason == "file-missing"
    assert t.lookup("NYC", "high", 7) is None


def test_loader_none_path_disabled() -> None:
    assert FbClimatologyTable.load(None).is_loaded is False


def test_loader_corrupt_json_disabled(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    t = FbClimatologyTable.load(p)
    assert t.is_loaded is False and t.fallback_reason.startswith("parse-error")


def test_loader_schema_error_disabled(tmp_path) -> None:
    t = FbClimatologyTable.load(_write(tmp_path, {"cells": [1, 2, 3]}))
    assert t.is_loaded is False and t.fallback_reason.startswith("schema-error")


def test_lookup_month_then_pool_fallback(tmp_path) -> None:
    path = _write(
        tmp_path,
        {
            "cells": {
                "NYC_high_7": {"mu": 86.0, "sigma": 4.0, "n": 29},
                "NYC_high_pool": {"mu": 71.0, "sigma": 15.0, "n": 210},
            }
        },
    )
    t = FbClimatologyTable.load(path)
    assert t.is_loaded is True
    m = t.lookup("NYC", "high", 7)
    assert m is not None and m.source == "month" and m.mu == 86.0
    pool = t.lookup("NYC", "high", 12)  # month 12 absent -> pool
    assert pool is not None and pool.source == "pool" and pool.mu == 71.0
    assert t.lookup("DEN", "low", 1) is None
