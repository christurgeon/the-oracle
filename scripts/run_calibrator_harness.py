#!/usr/bin/env python
"""Per-cell calibrator evaluation harness CLI.

Exit codes
----------
0: success
1: corpus failure (FileNotFoundError, OSError)
2: replay-parity assertion failed (FR-16 module-load assert)
3: write failure (currently folded into 1 by run_harness exception types;
   I-2-bis test accepts any nonzero)
4: schema validation failure (ValueError with "schema" or "invariant" in
   message)

Notes
-----
The plan referenced ``python.utils.log_config.configure_logging`` for a
structured-logging hook. That entry point does not exist (the real symbol is
:func:`python.utils.log_config.setup_logging`, which expects a
``LoggingConfig`` object). For a one-shot CLI a plain
:func:`logging.basicConfig` is sufficient, so we wire that directly rather
than carry the aspirational hook + try/except fallback.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Allow both `python -m scripts.run_calibrator_harness` and the more natural
# `python scripts/run_calibrator_harness.py`. Without this, the latter form
# raises ModuleNotFoundError: No module named 'python' because the repo root
# isn't on sys.path. Insert at index 0 so our copy wins over any installed
# package shadow.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/calibration/per_cell_winners.json"),
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        default=Path("data/gfs_bias/bias_correction.json"),
    )
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help=(
            "FM-2 bypass: skip bias-correction load even if artifact present. "
            "Output metadata records bias_correction_applied=false."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("python/config.yaml"))
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help='JSON dict, e.g., {"gfs_seamless": 1.0} (currently informational; '
        "weather config takes precedence)",
    )
    parser.add_argument(
        "--cell-allowlist",
        type=str,
        default=None,
        help="Comma-separated cells, e.g., 'MIA/high,NYC/low'",
    )
    parser.add_argument(
        "--write-diagnostics",
        type=Path,
        default=None,
        help="Optional: write per-fold raw scores JSON for debugging.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        from python.analytics.calibrator_harness import (
            SchemaValidationError,
            load_feature_rows,
            run_harness,
        )
    except AssertionError as exc:
        # FR-16 module-load assertion failed.
        sys.stderr.write(f"REPLAY-PARITY ASSERTION FAILED: {exc}\n")
        return 2

    cell_allowlist = (
        [c.strip() for c in args.cell_allowlist.split(",")]
        if args.cell_allowlist
        else None
    )

    if args.smoke_test:
        try:
            import yaml

            config = yaml.safe_load(args.config.read_text())
            weather = config.get("strategies", {}).get("weather", {})
            ensemble_weights = weather.get("ensemble_weights") or {"gfs_seamless": 1.0}
            rows = load_feature_rows(
                snapshots_dir=args.snapshots_dir,
                bias_table_path=args.bias_table if args.bias_table.exists() else None,
                ensemble_weights=ensemble_weights,
            )
            counts = Counter((r.city, r.var) for r in rows)
            sys.stdout.write("Per-cell row counts:\n")
            for (city, var), n in sorted(counts.items()):
                short = "high" if var.endswith("max") else "low"
                sys.stdout.write(f"  {city}/{short}: {n}\n")
            return 0
        except (FileNotFoundError, OSError) as exc:
            sys.stderr.write(f"CORPUS LOAD FAILURE: {exc}\n")
            return 1

    # --no-bias-correction explicit bypass (FM-2 promise).
    bias_table_path_arg: Path | None = None
    if not args.no_bias_correction and args.bias_table.exists():
        bias_table_path_arg = args.bias_table

    try:
        run_harness(
            snapshots_dir=args.snapshots_dir,
            config_path=args.config,
            bias_table_path=bias_table_path_arg,
            cell_allowlist=cell_allowlist,
            write_output_to=args.output,
            write_diagnostics_to=args.write_diagnostics,
        )
    except (FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"CORPUS LOAD FAILURE: {exc}\n")
        return 1
    except AssertionError as exc:
        sys.stderr.write(f"REPLAY-PARITY ASSERTION FAILED: {exc}\n")
        return 2
    except SchemaValidationError as exc:
        # Isinstance dispatch (was substring match on the message; FM-8 exit 4).
        sys.stderr.write(f"SCHEMA VALIDATION FAILED: {exc}\n")
        return 4
    except ValueError as exc:
        sys.stderr.write(f"VALIDATION ERROR: {exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
