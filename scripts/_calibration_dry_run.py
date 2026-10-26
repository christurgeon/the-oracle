"""Dry-run input-contract preflight — checks corpus coverage without fitting.

The only permitted side-effect is writing the data-availability calendar when
``--regenerate-calendar`` is set; diffs vs. the previous calendar are logged
so first-seen-date drift is visible.

Usage::

    uv run python scripts/_calibration_dry_run.py [--snapshots DIR]
        [--calendar PATH] [--ensemble-weights JSON] [--regenerate-calendar]

Exit codes: 0 on a clean tree, 1 on any abort condition.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path so the script works standalone.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._calibration_common import load_snapshot_data  # noqa: E402
from scripts._calibration_input_contract import (  # noqa: E402
    MIN_TRIPLES_FOR_FIT,
    CalibratorInputContractError,
    DataAvailabilityCalendar,
    InputContractReport,
    assert_snapshot_input_contract,
    build_data_availability_calendar,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def _resolve_ensemble_weights(cli_arg: str | None) -> dict[str, float]:
    """Resolve ensemble weights: CLI override > config.yaml."""
    if cli_arg:
        parsed = json.loads(cli_arg)
        if not isinstance(parsed, dict):
            raise CalibratorInputContractError(
                "--ensemble-weights must decode to a JSON object "
                "(model_name -> weight); got "
                f"{type(parsed).__name__}"
            )
        return {str(k): float(v) for k, v in parsed.items()}

    import yaml as _yaml

    cfg_path = _PROJECT_ROOT / "python" / "config.yaml"
    cfg = _yaml.safe_load(cfg_path.read_text())
    weights = cfg.get("strategies", {}).get("weather", {}).get("ensemble_weights")
    if not weights:
        raise CalibratorInputContractError(
            "ensemble_weights not configured: expected key path "
            "'strategies.weather.ensemble_weights' in "
            f"{cfg_path}. Pass --ensemble-weights '{{...}}' to "
            "override for ablation runs."
        )
    return {str(k): float(v) for k, v in weights.items()}


def _compute_calendar_diff(
    old_path: Path, new_payload: dict[str, Any]
) -> dict[str, list[tuple[str, str | None, str | None]]]:
    """Return structured diff of first-seen-date changes between existing and regenerated calendar.

    Keys: ``moved_later``, ``removed``, ``moved_earlier``, ``added``.
    Each value is a list of ``(model_name, old_date_or_None, new_date_or_None)``.
    Returns all-empty dict when there is nothing to compare (no previous file, or unparseable).
    """
    empty: dict[str, list[tuple[str, str | None, str | None]]] = {
        "moved_later": [],
        "removed": [],
        "moved_earlier": [],
        "added": [],
    }
    if not old_path.exists():
        return empty
    try:
        old_raw = json.loads(old_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "calendar diff: could not parse existing %s (%s); overwriting without diff",
            old_path,
            exc,
        )
        return empty

    old_models = old_raw.get("models", {}) or {}
    new_models = new_payload.get("models", {}) or {}

    def _date(entry: object) -> str | None:
        return entry.get("first_seen_date") if isinstance(entry, dict) else None

    result: dict[str, list[tuple[str, str | None, str | None]]] = {
        "moved_later": [],
        "removed": [],
        "moved_earlier": [],
        "added": [],
    }
    for name in sorted(set(old_models) | set(new_models)):
        old_d = _date(old_models.get(name))
        new_d = _date(new_models.get(name))
        if old_d == new_d:
            continue
        if old_d is None:
            result["added"].append((name, None, new_d))
        elif new_d is None:
            result["removed"].append((name, old_d, None))
        elif new_d > old_d:
            result["moved_later"].append((name, old_d, new_d))
        else:
            result["moved_earlier"].append((name, old_d, new_d))
    return result


def _log_calendar_diff(
    diff: dict[str, list[tuple[str, str | None, str | None]]], old_path: Path
) -> None:
    """Emit structured diff as log lines."""
    # All-empty means either no-previous-file or no-changes; old_path.exists()
    # distinguishes the two so the log message is accurate.
    all_empty = not any(diff.values())
    if all_empty:
        if not old_path.exists():
            logger.info("calendar diff: no previous calendar at %s; writing fresh", old_path)
        else:
            logger.info("calendar diff: no first-seen-date changes vs %s", old_path)
        return
    for name, old_d, new_d in diff.get("added", []):
        logger.warning("calendar diff: model %r added (first-seen %s)", name, new_d)
    for name, old_d, new_d in diff.get("removed", []):
        logger.warning("calendar diff: model %r removed (was first-seen %s)", name, old_d)
    for name, old_d, new_d in [*diff.get("moved_later", []), *diff.get("moved_earlier", [])]:
        logger.warning("calendar diff: model %r first-seen moved %s -> %s", name, old_d, new_d)


def _calendar_regressions(diff: dict[str, list[tuple[str, str | None, str | None]]]) -> list[str]:
    """Return human-readable lines for regression entries (moved_later + removed).

    Empty list means no regression.
    """
    lines: list[str] = []
    for name, old_d, new_d in diff.get("moved_later", []):
        lines.append(f"{name}: first-seen moved {old_d} -> {new_d} (LATER)")
    for name, old_d, _new_d in diff.get("removed", []):
        lines.append(f"{name}: removed (was first-seen {old_d})")
    return lines


def _apply_abort_policy(report: InputContractReport) -> list[str]:
    """Return abort reasons. Empty list = clean tree.

    Partial and empty triples always abort here — ``--allow-partial-model-coverage``
    and ``--allow-empty-cells`` are not exposed; this wrapper is a clean-tree gate.
    """
    reasons: list[str] = []
    if report.n_triples_total == 0:
        reasons.append(
            "n_triples_total == 0 (snapshots dir is empty or markets/settlements did not join)"
        )
        return reasons
    if report.n_triples_exempt == report.n_triples_total:
        reasons.append(
            "every triple is 'exempt' (calendar says no model is published "
            "on any of these dates); corpus is unusable for "
            "production-mirrored fitting"
        )
    if report.n_triples_full + report.n_triples_single_model_era < MIN_TRIPLES_FOR_FIT:
        reasons.append(
            "only "
            f"{report.n_triples_full + report.n_triples_single_model_era} "
            "triples in 'full' + 'single_model_era' buckets (need >= 100); "
            "sample-size sanity gate"
        )
    if report.n_triples_partial > 0:
        reasons.append(
            f"{report.n_triples_partial} post-cutover triples have only a "
            "subset of the expected ensemble models present"
        )
    if report.n_triples_empty > 0:
        reasons.append(
            f"{report.n_triples_empty} triples have NO model rows present "
            "despite having a market and settlement"
        )
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run input-contract preflight. Checks corpus coverage without fitting."
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Snapshot data directory (default: data/snapshots).",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=_PROJECT_ROOT / "data" / "calibration" / "data_availability_calendar.json",
        help="Path to the data-availability calendar JSON (default: "
        "data/calibration/data_availability_calendar.json).",
    )
    parser.add_argument(
        "--regenerate-calendar",
        action="store_true",
        help="Rebuild the calendar from --snapshots and write it to "
        "--calendar before running the gate. Logs a diff vs. the previously "
        "committed file if it existed.",
    )
    parser.add_argument(
        "--allow-calendar-regression",
        action="store_true",
        default=False,
        help="Permit --regenerate-calendar to overwrite the existing calendar even when an "
        "existing model's first-seen date moved LATER or a model disappeared. Default behavior "
        "aborts on regression to protect against running on a partial snapshot tree.",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="JSON dict of model_name -> weight. Defaults to "
        "config.yaml strategies.weather.ensemble_weights.",
    )
    args = parser.parse_args()

    # --- Optional calendar regeneration -------------------------------------
    if args.regenerate_calendar:
        logger.info("regenerating calendar from %s", args.snapshots)
        new_payload = build_data_availability_calendar(args.snapshots)
        diff = _compute_calendar_diff(args.calendar, new_payload)
        _log_calendar_diff(diff, args.calendar)
        regressions = _calendar_regressions(diff)
        if regressions and not args.allow_calendar_regression:
            print("\nABORT — calendar regeneration would regress committed first-seen dates:")
            for line in regressions:
                print(f"  - {line}")
            print(
                "\nThis usually means --snapshots points at a partial tree. "
                "If the regression is intentional, re-run with "
                "--allow-calendar-regression.\n"
            )
            return 1
        args.calendar.parent.mkdir(parents=True, exist_ok=True)
        args.calendar.write_text(json.dumps(new_payload, indent=2, sort_keys=True) + "\n")
        logger.info("wrote regenerated calendar to %s", args.calendar)

    # --- Load corpus + calendar ---------------------------------------------
    logger.info("loading snapshot data from %s", args.snapshots)
    forecasts, settlements, markets = load_snapshot_data(args.snapshots)
    logger.info(
        "loaded %d forecasts, %d settlements, %d market entries",
        len(forecasts),
        len(settlements),
        len(markets),
    )

    if not args.calendar.exists():
        logger.error(
            "calendar not found at %s; pass --regenerate-calendar to build it",
            args.calendar,
        )
        return 1
    try:
        calendar = DataAvailabilityCalendar.from_path(args.calendar)
    except CalibratorInputContractError as exc:
        print(f"\nABORT: {exc}\n")
        return 1

    # --- Ensemble weights ---------------------------------------------------
    try:
        ensemble_weights = _resolve_ensemble_weights(args.ensemble_weights)
    except CalibratorInputContractError as exc:
        logger.error("%s", exc)
        return 1

    # --- Run the gate -------------------------------------------------------
    try:
        report = assert_snapshot_input_contract(
            forecasts,
            settlements,
            markets,
            ensemble_weights=ensemble_weights,
            calendar=calendar,
        )
    except CalibratorInputContractError as exc:
        # Hard contract failure — e.g., ensemble_weights references a model
        # not in the calendar. The report itself was never produced.
        print(f"\nABORT: {exc}\n")
        return 1

    print(report.format_for_console())

    reasons = _apply_abort_policy(report)
    if reasons:
        print("")
        print("ABORT — input-contract conditions failed:")
        for r in reasons:
            print(f"  - {r}")
        print("")
        return 1
    print("\nOK — no abort conditions triggered.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
