"""Every cell in ``strategies.weather.emos_enabled_cells`` must be defensible.

A promoted cell whose fit diagnostics carry an overridable warning (``unstable_b``,
``unstable_d``, ``d_at_zero``) needs an active (not rolled back) entry in
``data/emos/overrides.json``. ``incompatible_sigma`` is never overridable. This test
makes ``overrides.json`` enforced instead of descriptive.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "python" / "config.yaml"
OVERRIDES_PATH = ROOT / "data" / "emos" / "overrides.json"
DIAGNOSTICS_PATH = ROOT / "data" / "emos" / "coefficients.diagnostics.json"

OVERRIDABLE_WARNINGS = frozenset({"unstable_b", "unstable_d", "d_at_zero"})
NON_OVERRIDABLE_WARNINGS = frozenset({"incompatible_sigma"})
REQUIRED_OVERRIDE_KEYS = frozenset(
    {
        "cell",
        "warning_overridden",
        "warning_definition",
        "fit_artifact",
        "cv_oos_brier",
        "justification",
        "rollback_trigger",
        "re_evaluate_at",
    }
)


def find_violations(
    enabled_cells: Iterable[str],
    diagnostics: Mapping[str, Any],
    overrides: Iterable[Mapping[str, Any]],
) -> list[str]:
    active = {
        (entry.get("cell"), entry.get("warning_overridden"))
        for entry in overrides
        if entry.get("rolled_back") is not True
    }
    violations: list[str] = []
    for cell in enabled_cells:
        cell_diag = diagnostics.get(cell)
        if cell_diag is None:
            violations.append(f"{cell}: no entry in coefficients.diagnostics.json")
            continue
        if cell_diag.get("skipped"):
            violations.append(f"{cell}: skipped in coefficients.diagnostics.json (no fit)")
            continue
        warnings = set(cell_diag.get("ci_warnings") or [])
        for warning in sorted(warnings & NON_OVERRIDABLE_WARNINGS):
            violations.append(f"{cell}: {warning} is not overridable")
        for warning in sorted(warnings & OVERRIDABLE_WARNINGS):
            if (cell, warning) not in active:
                violations.append(f"{cell}: {warning} has no active override in overrides.json")
    return violations


@pytest.fixture(scope="module")
def overrides_doc() -> dict[str, Any]:
    return json.loads(OVERRIDES_PATH.read_text())


@pytest.fixture(scope="module")
def diagnostics_cells() -> dict[str, Any]:
    return json.loads(DIAGNOSTICS_PATH.read_text())["cells"]


@pytest.fixture(scope="module")
def enabled_cells() -> list[str]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return list(config["strategies"]["weather"].get("emos_enabled_cells") or [])


def test_overrides_file_shape(overrides_doc: dict[str, Any]) -> None:
    assert "schema_version" in overrides_doc, "overrides.json must carry schema_version"
    assert isinstance(overrides_doc["overrides"], list), "overrides.json 'overrides' must be a list"
    for entry in overrides_doc["overrides"]:
        label = f"{entry.get('cell')}/{entry.get('warning_overridden')}"
        missing = REQUIRED_OVERRIDE_KEYS - entry.keys()
        assert not missing, f"{label}: missing keys {sorted(missing)}"
        assert entry["warning_overridden"] in OVERRIDABLE_WARNINGS, (
            f"{label}: warning_overridden must be one of {sorted(OVERRIDABLE_WARNINGS)}"
        )
        if entry.get("rolled_back") is True:
            assert entry.get("rollback_date"), f"{label}: rolled_back without rollback_date"


def test_enabled_cells_have_overrides_for_their_warnings(
    enabled_cells: list[str],
    diagnostics_cells: dict[str, Any],
    overrides_doc: dict[str, Any],
) -> None:
    violations = find_violations(enabled_cells, diagnostics_cells, overrides_doc["overrides"])
    assert not violations, "\n".join(
        [
            *violations,
            "Fix: add an active entry to data/emos/overrides.json with keys "
            f"{sorted(REQUIRED_OVERRIDE_KEYS)}, remove the cell from "
            "strategies.weather.emos_enabled_cells, or refit. "
            "incompatible_sigma cannot be overridden.",
        ]
    )


def _override(cell: str, warning: str, **extra: Any) -> dict[str, Any]:
    return {"cell": cell, "warning_overridden": warning, **extra}


def test_guard_flags_missing_override() -> None:
    diagnostics = {"LAX/low": {"ci_warnings": ["unstable_d"]}}
    assert find_violations(["LAX/low"], diagnostics, [])


def test_guard_ignores_rolled_back_override() -> None:
    diagnostics = {"LAX/low": {"ci_warnings": ["unstable_d"]}}
    overrides = [_override("LAX/low", "unstable_d", rolled_back=True, rollback_date="2026-05-18")]
    assert find_violations(["LAX/low"], diagnostics, overrides)


def test_guard_rejects_incompatible_sigma_even_with_override() -> None:
    diagnostics = {"LAX/high": {"ci_warnings": ["incompatible_sigma"]}}
    overrides = [_override("LAX/high", "incompatible_sigma")]
    assert find_violations(["LAX/high"], diagnostics, overrides)


def test_guard_accepts_matching_active_override() -> None:
    diagnostics = {"LAX/low": {"ci_warnings": ["unstable_d"]}}
    overrides = [_override("LAX/low", "unstable_d")]
    assert find_violations(["LAX/low"], diagnostics, overrides) == []


def test_guard_requires_override_for_each_warning() -> None:
    diagnostics = {"SEA/low": {"ci_warnings": ["unstable_b", "d_at_zero"]}}
    overrides = [_override("SEA/low", "unstable_b")]
    assert len(find_violations(["SEA/low"], diagnostics, overrides)) == 1


def test_guard_does_not_match_override_of_another_cell() -> None:
    diagnostics = {"LAX/low": {"ci_warnings": ["unstable_d"]}}
    overrides = [_override("LAX/high", "unstable_d")]
    assert find_violations(["LAX/low"], diagnostics, overrides)


def test_guard_flags_missing_diagnostics() -> None:
    assert find_violations(["NYC/high"], {}, [])


def test_guard_accepts_clean_cell_without_override() -> None:
    diagnostics = {"MIA/high": {"ci_warnings": []}}
    assert find_violations(["MIA/high"], diagnostics, []) == []


def test_guard_flags_skipped_cell() -> None:
    diagnostics = {"MIA/high": {"skipped": True, "n_train": 12}}
    assert find_violations(["MIA/high"], diagnostics, []) == [
        "MIA/high: skipped in coefficients.diagnostics.json (no fit)"
    ]
