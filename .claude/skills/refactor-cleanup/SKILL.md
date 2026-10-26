---
name: refactor-cleanup
description: Use when the user asks to clean up, tidy, or refactor existing code under a given path without changing behavior, and wants the result as a PR.
argument-hint: <path> [--no-pr]
disable-model-invocation: true
---

# Refactor Cleanup

Behavior-preserving cleanup of everything under the path in `$ARGUMENTS`, delivered as one reviewable PR.
Not a feature change. If a change could alter observable behavior, don't make it — list it in the PR instead.

## 1. Scope and baseline (before any edit)

1. Confirm the path exists and is git-tracked; list the files in scope (skip generated/vendored).
2. Read the repo conventions: CLAUDE.md files covering the path, `pyproject.toml` (ruff/mypy), CI config.
3. Branch from the up-to-date default branch: `refactor/cleanup-<path-slug>`.
4. Record the baseline for the target: `ruff check`, `ruff format --check`, `mypy`, and the tests that
   cover it (find them via imports of the target modules). Pre-existing failures are reported, not fixed.
5. Note which files have no test coverage — only mechanical changes there (imports, comments, obvious
   dead code).

## 2. Invariants — never change

- Public names and signatures used elsewhere (grep for callers before renaming).
- Exception types and deliberate broad `except` / fail-open / never-raises paths.
- Log message text and levels.
- Order of side effects and `await` points.
- Serialized field names (JSON/JSONL keys, config keys).
- Test files, except import paths. A failing test means the refactor is wrong.

## 3. Changes to make

- Remove dead code, unused imports, redundant branches. Flatten nesting where clearly equivalent.
- Rename unclear private names. Add type hints where they fix or clarify, not everywhere.
- Formatter: run it only on files that were format-clean at baseline, so the diff stays reviewable.
- Comments: keep a comment only if it explains why, a constraint, an ordering or concurrency
  requirement, an external workaround, or an incident (keep incident dates and names; do not cite
  commits or PRs from before the public release). Delete comments that restate the code. Test: would it help someone a year from now with no
  memory of the change?
- Docstrings: keep for public APIs that carry context; delete ones that repeat the name.

## 4. Verify

Re-run the baseline commands. Result must be: no new failures, tests unchanged and passing.
Then run `/code-review` on the diff and fix confirmed findings. Check `git status` for stray files.

## 5. Deliver

Commit per change type (`refactor(<area>): ...`). If the diff exceeds ~400 lines, split into several
PRs by file.

Unless `--no-pr` is given: push, then `gh pr create` with title
`refactor(<area>): clean up code and comments` and this body:

    ## Summary
    Main changes, grouped by type.

    ## Not changed
    Areas left alone and why (untested, risky, behavior-bearing).

    ## Validation
    Baseline vs final results; anything not run and why.

Report: PR URL, summary, validation results, follow-ups.
