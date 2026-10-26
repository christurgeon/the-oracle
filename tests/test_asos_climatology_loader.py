"""Tests for ClimatologyTable loader (Task 3.2).

Hermetic: builds all fixtures in-test using a small in-memory dict written
to a temp JSON — does NOT depend on the real data/asos/climatology.json
artifact so the loader tests stay fast and environment-independent.

Run with: uv run pytest tests/test_asos_climatology_loader.py -v
"""
from __future__ import annotations

import json

import pytest

from python.data.asos import ClimatologyTable

# ---------------------------------------------------------------------------
# Minimal fixture factory
# ---------------------------------------------------------------------------

def _write_fixture(tmp_path, cities: dict) -> str:
    """Dump a minimal climatology.json fixture to tmp_path and return its path."""
    payload = {
        "spec_version": "v2.1",
        "fit_window_utc": {
            "start": "2024-05-01T00:00:00+00:00",
            "end_exclusive": "2026-04-28T00:00:00+00:00",
        },
        "params": {
            "min_n_per_cell": 30,
            "min_n_pooled": 60,
            "min_sigma_f": 0.5,
        },
        "cities": cities,
        "diagnostics": {},
    }
    path = tmp_path / "fixture_climatology.json"
    path.write_text(json.dumps(payload))
    return str(path)


# ---------------------------------------------------------------------------
# Fixture dict used across tests
# ---------------------------------------------------------------------------

# city=MIA, var=high
# - month=6, hour=14 -> primary cell (present, n=40)
# - month=6, hour=15 -> absent in month=6 but pool (hour=15 pooled) covers it
# - month=6, hour=16 -> unavailable (no month cell, no pool)
# - month=7, hour=14 -> primary cell (present in a different month)

_MIA_HIGH_CELLS = {
    # Present month-specific cell
    "06-14": {"mu": 5.5, "sigma": 3.2, "n": 40, "source": "primary"},
    # A different month — so lookup(MIA, 7, 14) resolves here, not pool
    "07-14": {"mu": 4.0, "sigma": 2.8, "n": 35, "source": "primary"},
    # Hour 15 has NO month=6 primary cell, but the pool covers it
    "01-15": {"mu": 6.0, "sigma": 2.5, "n": 31, "source": "primary"},  # diff month
    # Pool-sourced cell for hour 15 (month=1 for sanity, not month=6)
    "06-15": {"mu": 7.0, "sigma": 2.1, "n": 210, "source": "hour_pool"},
    # Hour 16 month 6 is unavailable
    "06-16": {"source": "unavailable"},
    # Hour 16 has no pool entry at all (we'll rely on "06-16" being unavailable
    # and no pool fallback in our fixture)
}

# We'll construct a fixture where:
#   - (MIA, high, month=6, hour=14)  -> primary, n=40  [test: exact lookup]
#   - (MIA, high, month=6, hour=15)  -> absent as primary but hour_pool cell is "06-15"
#     wait — the pool logic in loader uses (city,hour) not (city,month,hour).
#     The loader must fall back to ANY cell in (city, var) with source=hour_pool
#     that matches the hour. The research builder writes "06-15" (month=6) for a
#     pool entry — meaning month=6, hour=15 has source=hour_pool.
#     BUT the test case for "cell absent for a month but present in pool" requires
#     a DIFFERENT month to be absent while the pool resolves it.
#     Correct interpretation: if lookup(city, month, hour) finds a cell with
#     source=unavailable (or no key at all), the loader should try any
#     hour_pool cell for that (city, hour).
#     We'll model "absent month" as: month=9, hour=15 has no entry at all.
#     Pool is found in "06-15" (hour_pool for hour 15).
#   - (MIA, high, month=9, hour=99) -> None (fully absent)

_CITIES_FIXTURE = {
    "MIA": {
        "high": {
            "cells": {
                "06-14": {"mu": 5.5, "sigma": 3.2, "n": 40, "source": "primary"},
                "07-14": {"mu": 4.0, "sigma": 2.8, "n": 35, "source": "primary"},
                # month=6, hour=15 has a pool cell
                "06-15": {"mu": 7.0, "sigma": 2.1, "n": 210, "source": "hour_pool"},
                # month=6, hour=16 is unavailable
                "06-16": {"source": "unavailable"},
                # There is NO entry for (month=9, hour=15) — triggers pool fallback
                # Pool for hour=15 is "06-15" (hour_pool source)
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_returns_climatology_table(tmp_path):
    """ClimatologyTable.load() returns a ClimatologyTable instance."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    assert isinstance(table, ClimatologyTable)


def test_present_cell_returns_mu_sigma_n(tmp_path):
    """A cell present in the artifact with source='primary' returns its (mu, sigma, n, source)."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    result = table.lookup("MIA", month=6, hour=14)
    assert result is not None
    mu, sigma, n, source = result
    assert mu == pytest.approx(5.5)
    assert sigma == pytest.approx(3.2)
    assert n == 40
    assert source == "primary"


def test_present_cell_different_month(tmp_path):
    """A cell in a different month resolves correctly from its own slot."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    result = table.lookup("MIA", month=7, hour=14)
    assert result is not None
    mu, sigma, n, source = result
    assert mu == pytest.approx(4.0)
    assert n == 35
    assert source == "primary"


def test_absent_month_falls_back_to_hour_pool(tmp_path):
    """A (city, month, hour) absent as primary/exact but with a pool cell for that hour
    returns the pooled value.

    month=9, hour=15 has no entry -> falls back to the hour_pool cell "06-15"
    which has source='hour_pool' and represents the (city, hour=15) pool.
    """
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    result = table.lookup("MIA", month=9, hour=15)
    assert result is not None
    mu, sigma, n, source = result
    assert mu == pytest.approx(7.0)
    assert sigma == pytest.approx(2.1)
    assert n == 210
    assert source == "hour_pool"


def test_unavailable_cell_falls_back_to_hour_pool_if_present(tmp_path):
    """A cell with source='unavailable' for the exact (month, hour) key should
    also fall back to the pool if one exists for that hour.

    month=6, hour=15 has source='hour_pool' directly — this also resolves.
    (month=6, hour=16) is 'unavailable' and has no pool for hour=16 -> None.
    """
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    # The "06-16" slot is unavailable and no pool covers hour=16
    result = table.lookup("MIA", month=6, hour=16)
    assert result is None


def test_fully_absent_city_month_hour_returns_none(tmp_path):
    """A (city, month, hour) absent from the artifact with no pool coverage -> None."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    # hour=99 doesn't exist at all
    result = table.lookup("MIA", month=6, hour=99)
    assert result is None


def test_unknown_city_returns_none(tmp_path):
    """A city not in the artifact returns None."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    result = table.lookup("DEN", month=6, hour=14)
    assert result is None


def test_hour_pool_cell_direct_lookup(tmp_path):
    """An exact (month, hour) key that IS a pool cell returns the pool's (mu, sigma, n, source)."""
    path = _write_fixture(tmp_path, _CITIES_FIXTURE)
    table = ClimatologyTable.load(path)
    # (month=6, hour=15) has source='hour_pool' — direct lookup returns pool values
    result = table.lookup("MIA", month=6, hour=15)
    assert result is not None
    mu, sigma, n, source = result
    assert mu == pytest.approx(7.0)
    assert n == 210
    assert source == "hour_pool"


def test_load_missing_file_raises(tmp_path):
    """ClimatologyTable.load() raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        ClimatologyTable.load(str(tmp_path / "does_not_exist.json"))
