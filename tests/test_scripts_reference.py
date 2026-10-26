from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "scripts-reference.md"
ROW_PATH = re.compile(r"^\|\s*`(scripts/[\w/]+\.(?:py|sh))`", re.MULTILINE)
UNDOCUMENTED = {"scripts/__init__.py", "scripts/__main__.py"}


def _scripts_on_disk() -> set[str]:
    found = {
        p.relative_to(ROOT).as_posix()
        for pattern in ("scripts/*.py", "scripts/*.sh", "scripts/research/*.py")
        for p in ROOT.glob(pattern)
    }
    return found - UNDOCUMENTED


def _scripts_in_doc() -> set[str]:
    return set(ROW_PATH.findall(DOC.read_text()))


def test_every_script_is_documented() -> None:
    missing = sorted(_scripts_on_disk() - _scripts_in_doc())
    assert not missing, f"add to docs/scripts-reference.md: {missing}"


def test_documented_scripts_exist() -> None:
    stale = sorted(_scripts_in_doc() - _scripts_on_disk())
    assert not stale, f"remove from docs/scripts-reference.md: {stale}"
