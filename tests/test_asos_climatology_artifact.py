"""Artifact-level smoke tests for data/asos/climatology.json (Task 3.1).

Validates the real production artifact produced by
``scripts/build_asos_climatology.py``.  These tests load the on-disk file and
assert structural / value invariants.  They are NOT hermetic (they touch the
repo artifact), but they are cheap and run in < 1 second.

Run with: uv run pytest tests/test_asos_climatology_artifact.py -v
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Artifact location
# ---------------------------------------------------------------------------

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "asos" / "climatology.json"
)

# The four intraday-strategy trading cities (MIA, AUS, CHI, NYC).
# LAX is included in the artifact for completeness (excluded from live trading
# by excluded_city_vars) but the strategy specifically needs these four.
TRADING_CITIES = {"MIA", "AUS", "CHI", "NYC"}

# Parameter constants (must match the builder)
MIN_SIGMA_F = 0.5
MIN_N_PER_CELL = 30
MIN_N_POOLED = 60


# ---------------------------------------------------------------------------
# Load fixture (module-scope to avoid repeated JSON parse in each test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), (
        f"Artifact not found: {ARTIFACT_PATH}. "
        "Run: uv run python scripts/build_asos_climatology.py"
    )
    return json.loads(ARTIFACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Top-level structure
# ---------------------------------------------------------------------------


def test_artifact_has_required_top_level_keys(artifact):
    """Artifact must contain spec_version, fit_window_utc, params, cities."""
    assert "spec_version" in artifact
    assert "fit_window_utc" in artifact
    assert "params" in artifact
    assert "cities" in artifact


def test_artifact_spec_version(artifact):
    assert artifact["spec_version"] == "v2.1"


def test_artifact_fit_window_present(artifact):
    fw = artifact["fit_window_utc"]
    assert "start" in fw
    assert "end_exclusive" in fw


def test_artifact_params(artifact):
    p = artifact["params"]
    assert p["min_n_per_cell"] == MIN_N_PER_CELL
    assert p["min_n_pooled"] == MIN_N_POOLED
    assert p["min_sigma_f"] == MIN_SIGMA_F


# ---------------------------------------------------------------------------
# City coverage
# ---------------------------------------------------------------------------


def test_all_five_cities_present(artifact):
    """All five source cities (AUS, CHI, LAX, MIA, NYC) must be in the artifact."""
    assert set(artifact["cities"].keys()) == {"AUS", "CHI", "LAX", "MIA", "NYC"}


def test_trading_cities_have_high_and_low(artifact):
    for city in TRADING_CITIES:
        assert "high" in artifact["cities"][city], f"{city} missing 'high'"
        assert "low" in artifact["cities"][city], f"{city} missing 'low'"


# ---------------------------------------------------------------------------
# Per-cell value invariants
# ---------------------------------------------------------------------------


def _iter_filled_cells(artifact: dict):
    """Yield (city, var, key, cell_dict) for every non-unavailable cell."""
    for city, city_block in artifact["cities"].items():
        for var, var_block in city_block.items():
            for key, cell in var_block.get("cells", {}).items():
                if cell.get("source") != "unavailable":
                    yield city, var, key, cell


def test_all_filled_cells_have_required_fields(artifact):
    """Every non-unavailable cell must have mu, sigma, n, source."""
    for city, var, key, cell in _iter_filled_cells(artifact):
        assert "mu" in cell, f"{city}/{var}/{key} missing mu"
        assert "sigma" in cell, f"{city}/{var}/{key} missing sigma"
        assert "n" in cell, f"{city}/{var}/{key} missing n"
        assert "source" in cell, f"{city}/{var}/{key} missing source"


def test_all_filled_cells_sigma_above_min(artifact):
    """Every non-unavailable cell must have sigma >= MIN_SIGMA_F (0.5 °F)."""
    for city, var, key, cell in _iter_filled_cells(artifact):
        assert cell["sigma"] >= MIN_SIGMA_F, (
            f"{city}/{var}/{key}: sigma={cell['sigma']:.4f} < {MIN_SIGMA_F}"
        )


def test_primary_cells_n_at_least_min_n(artifact):
    """Primary cells must have n >= MIN_N_PER_CELL (30)."""
    for city, var, key, cell in _iter_filled_cells(artifact):
        if cell.get("source") == "primary":
            assert cell["n"] >= MIN_N_PER_CELL, (
                f"{city}/{var}/{key}: primary cell n={cell['n']} < {MIN_N_PER_CELL}"
            )


def test_pool_cells_n_at_least_min_n_pooled(artifact):
    """Hour-pool cells must have n >= MIN_N_POOLED (60)."""
    for city, var, key, cell in _iter_filled_cells(artifact):
        if cell.get("source") == "hour_pool":
            assert cell["n"] >= MIN_N_POOLED, (
                f"{city}/{var}/{key}: pool cell n={cell['n']} < {MIN_N_POOLED}"
            )


def test_high_cells_mu_nonnegative(artifact):
    """All non-unavailable cells for var='high' must have mu >= 0
    (remaining gain to daily high is non-negative by construction).
    """
    for city, var, key, cell in _iter_filled_cells(artifact):
        if var == "high":
            assert cell["mu"] >= 0, (
                f"{city}/{var}/{key}: mu={cell['mu']:.4f} < 0"
            )


def test_source_values_valid(artifact):
    """Every cell must have source in {primary, hour_pool, unavailable}."""
    valid_sources = {"primary", "hour_pool", "unavailable"}
    for city, city_block in artifact["cities"].items():
        for var, var_block in city_block.items():
            for key, cell in var_block.get("cells", {}).items():
                src = cell.get("source")
                assert src in valid_sources, (
                    f"{city}/{var}/{key}: unexpected source={src!r}"
                )


def test_cell_keys_format(artifact):
    """All cell keys must match MM-HH format (01–12, 00–23)."""
    import re

    pattern = re.compile(r"^(0[1-9]|1[0-2])-(0\d|1\d|2[0-3])$")
    for city, city_block in artifact["cities"].items():
        for var, var_block in city_block.items():
            for key in var_block.get("cells", {}):
                assert pattern.match(key), (
                    f"{city}/{var}: malformed cell key {key!r}"
                )


def test_all_288_cells_present_per_city_var(artifact):
    """Each (city, var) block must have exactly 288 cells (12 months × 24 hours)."""
    for city, city_block in artifact["cities"].items():
        for var, var_block in city_block.items():
            n = len(var_block.get("cells", {}))
            assert n == 288, (
                f"{city}/{var}: expected 288 cells, got {n}"
            )


# ---------------------------------------------------------------------------
# Current-month coverage for trading cities
# ---------------------------------------------------------------------------


def test_trading_cities_high_covered_current_month(artifact):
    """All four trading cities have at least one non-unavailable 'high' cell
    in the current calendar month.

    The fit window ends 2026-04-28, so month=5 (May) and month=6 (June) are
    outside the primary fit window.  Pool cells drawn from earlier months
    still cover these months if the (city, hour) pool passes the n≥60 gate.
    If pool coverage is genuinely absent for the current month, this test
    fails — that is a real signal that the corpus is stale and needs a refit.
    """
    current_month = datetime.date.today().month
    month_prefix = f"{current_month:02d}-"

    for city in TRADING_CITIES:
        var_block = artifact["cities"][city]["high"]
        covered = any(
            key.startswith(month_prefix) and cell.get("source") != "unavailable"
            for key, cell in var_block["cells"].items()
        )
        assert covered, (
            f"{city}/high has NO covered cells for month={current_month} "
            f"(all cells unavailable or missing). "
            f"The artifact may need a refit with fresher ASOS data."
        )


def test_trading_cities_have_at_least_one_covered_month(artifact):
    """Fallback: each trading city has at least one non-unavailable 'high' cell
    across any month (artifact is not degenerate).
    """
    for city in TRADING_CITIES:
        var_block = artifact["cities"][city]["high"]
        covered_any = any(
            cell.get("source") != "unavailable"
            for cell in var_block["cells"].values()
        )
        assert covered_any, f"{city}/high has NO covered cells at all"
