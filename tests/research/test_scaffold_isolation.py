"""M7 — static-grep hermetic-isolation guard.

Hermetic discipline (spec §2.2 / A12 / B4): the `python.research.*`
subtree is research-only and MUST NOT be imported by any production
module (orchestrator, strategies, risk, execution, notifications,
dashboard, utils, data, analytics, ...). If a production module ever
imports the scaffold, the research substrate would leak into the live
trading loop, and the "research code can be broken without breaking
prod" invariant collapses.

This test enforces that invariant by static substring scan over
`python/**/*.py` minus an allowlist. It is fresh-interpreter-
independent (no `sys.modules` introspection), so a sibling pytest
session that legitimately imports `python.research.scaffold` cannot
contaminate the result. The v1.0 `sys.modules` approach was rejected
for exactly that reason (spec A12); v1.2 B4 broadened the v1.1 glob
from four explicit paths to the whole `python/` tree because the
orchestrator transitively touches `python/notifications/`,
`python/utils/`, `python/data/`, etc., and an accidental import from
any of those modules would slip through a four-path check.

`if TYPE_CHECKING: import python.research` is intentionally NOT
exempt: a future refactor that moves a typing-only import to runtime
would silently break the invariant. Better to flag it now.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"

_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "from python.research",
    "import python.research",
    "python.research.scaffold",
)

_ALLOWLIST_FILES: frozenset[Path] = frozenset({_PYTHON_ROOT / "__init__.py"})
_ALLOWLIST_DIRS: tuple[Path, ...] = (_PYTHON_ROOT / "research",)


def _is_allowlisted(path: Path) -> bool:
    if path in _ALLOWLIST_FILES:
        return True
    for allowed_dir in _ALLOWLIST_DIRS:
        try:
            path.relative_to(allowed_dir)
        except ValueError:
            continue
        return True
    return False


def test_research_scaffold_isolation_static_grep() -> None:
    """No production module imports from `python.research.*`.

    See module docstring for WHY this matters.
    """
    assert _PYTHON_ROOT.is_dir(), f"expected python/ tree at {_PYTHON_ROOT}"

    violations: list[str] = []
    for source_path in sorted(_PYTHON_ROOT.rglob("*.py")):
        if _is_allowlisted(source_path):
            continue
        text = source_path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern in line:
                    violations.append(f"{source_path}:{lineno}: {pattern!r} in {line.strip()!r}")

    if violations:
        joined = "\n  ".join(violations)
        pytest.fail(
            "ResearchScaffoldIsolationError:\n  "
            f"{joined}\n\n"
            "The python/research/ subtree is research-only and MUST NOT be imported by\n"
            "production modules (orchestrator, strategies, risk, execution, notifications,\n"
            "dashboard, utils, data, analytics). See spec §2.2 / A12 / B4.\n\n"
            "Either:\n"
            "1. Remove the python.research.* import from the offending production module, OR\n"
            "2. If the import is intentional, add the file path to the allowlist in\n"
            "   tests/research/test_scaffold_isolation.py with a comment explaining why."
        )
