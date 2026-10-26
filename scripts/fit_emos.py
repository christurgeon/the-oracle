"""Offline EMOS fitter (scope A).

Reads data/emos/corpus.parquet, fits per-(city, var) Gaussian EMOS coefficients
pooled across leads (decision A1), and writes coefficients.json plus
coefficients.diagnostics.json. Hermetic — no production runtime imports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from python.analytics.emos_fit import (
    BOOTSTRAP_RESAMPLE_DATE_CLUSTER,
    BOOTSTRAP_RESAMPLE_ROW,
    MIN_CLUSTER_DATES,
    fit_one_cell,
)
from python.utils.atomic_write import atomic_write_text as _atomic_write

FITTER_VERSION = "0.1.0"
CV_PROTOCOL = "5-fold time-series, expanding"

logger = logging.getLogger(__name__)


def _archive_coefficients(coeffs_path: Path, history_dir: Path, *, retention: int = 180) -> None:
    """Copy the just-written coefficients.json to history_dir/<sha256>.json
    and prune any files past the retention-th newest by mtime.

    Skips the copy if a file with that hash already exists (same content
    already archived). Uses shutil.copy2 to preserve mtime so the
    retention-by-mtime sweep ordering is consistent across copies.
    """
    history_dir.mkdir(parents=True, exist_ok=True)
    raw = coeffs_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    dest = history_dir / f"{sha}.json"
    if not dest.exists():
        shutil.copy2(coeffs_path, dest)

    siblings = sorted(history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for stale in siblings[retention:]:
        try:
            stale.unlink()
        except OSError:
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline EMOS fitter (scope A).")
    p.add_argument("--corpus", default="data/emos/corpus.parquet")
    p.add_argument("--output-dir", default="data/emos/")
    p.add_argument(
        "--window-days", type=int, default=None,
        help="Use only the most recent N settled forecast_dates. None means all.",
    )
    p.add_argument("--bootstrap-iters", type=int, default=400)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260503)
    p.add_argument(
        "--cluster-by-date", action="store_true",
        help=(
            "Resample whole forecast_dates instead of rows (row CIs are too narrow). "
            "Required before any EMOS re-promotion; see docs/calibration-runbook.md. "
            f"Cells with fewer than {MIN_CLUSTER_DATES} dates are skipped."
        ),
    )
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--low-n-flag-threshold", type=int, default=60)
    return p.parse_args()


def _resolve_fit_date(df: pd.DataFrame) -> dt.date:
    """Return ``max(df.forecast_date)`` or fall back to today.

    F1.5 (2026-05-19 adversarial review): the previous behavior stamped
    artifacts with ``dt.date.today()``. A backfill rebuild against older
    snapshots therefore looked freshly fit, silently buying one extra day
    of grace from any tripwire keyed on artifact age. Stamping
    ``max(corpus.forecast_date)`` ties the artifact's age to the data it
    saw, not when the script happened to run.
    """
    if "forecast_date" not in df.columns or len(df) == 0:
        logger.warning(
            "fit_date fallback: corpus has no forecast_date column or is empty; "
            "stamping today's date."
        )
        return dt.date.today()
    parsed = pd.to_datetime(df["forecast_date"], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        logger.warning(
            "fit_date fallback: all forecast_date values are null/unparseable; "
            "stamping today's date."
        )
        return dt.date.today()
    max_ts = parsed.max()
    if hasattr(max_ts, "date"):
        return cast(dt.date, max_ts.date())
    return cast(dt.date, max_ts)


def _load_corpus(path: Path, window_days: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["settlement_status"] != "pending"].reset_index(drop=True)
    if window_days is not None and window_days > 0:
        cutoff = pd.to_datetime(df["forecast_date"]).max() - pd.Timedelta(days=window_days - 1)
        df = df[pd.to_datetime(df["forecast_date"]) >= cutoff].reset_index(drop=True)
    return df


def _print_report(
    cells_coef: dict[str, dict[str, object]],
    cells_diag: dict[str, dict[str, object]],
    bootstrap_resample: str,
) -> None:
    print(f"\nEMOS fit (version {FITTER_VERSION})")
    print(f"bootstrap resample: {bootstrap_resample}")
    print("=" * 100)
    header = (
        f"{'cell':<10} {'n':>5} {'a':>8} {'b':>6} {'c':>8} {'d':>8} "
        f"{'CRPS_A1':>9} {'CRPS_A2':>9} {'Br_emos':>9} {'Br_raw':>9}  flags"
    )
    print(header)
    print("-" * len(header))
    n_fit = n_skip = n_low = n_warn = 0
    for key, body in sorted(cells_coef.items()):
        if body.get("skipped"):
            n_skip += 1
            print(f"{key:<10}  SKIP   {body.get('reason', '')}")
            continue
        diag = cells_diag.get(key, {})
        flags: list[str] = []
        if body.get("low_n"):
            flags.append("low_n")
            n_low += 1
        if body.get("ci_warnings"):
            flags.extend(cast(list[str], body["ci_warnings"]))
            n_warn += 1
        n_fit += 1
        print(
            f"{key:<10} {body['n_train']:>5} "
            f"{body['a']:>8.3f} {body['b']:>6.3f} {body['c']:>8.3f} {body['d']:>8.3f} "
            f"{diag.get('oos_crps_a1', float('nan')):>9.4f} "
            f"{diag.get('oos_crps_a2', float('nan')):>9.4f} "
            f"{diag.get('oos_brier_emos', float('nan')):>9.4f} "
            f"{diag.get('oos_brier_raw', float('nan')):>9.4f}  "
            f"{','.join(flags) if flags else '-'}"
        )
    a2_winners = [
        k for k, body in cells_coef.items()
        if not body.get("skipped")
        and (cells_diag.get(k, {}).get("oos_crps_a1") is not None)
        and (cells_diag.get(k, {}).get("oos_crps_a2") is not None)
        and (
            cast(float, cells_diag[k]["oos_crps_a1"])
            - cast(float, cells_diag[k]["oos_crps_a2"])
        ) / max(cast(float, cells_diag[k]["oos_crps_a1"]), 1e-9) > 0.05
    ]
    print()
    print(f"fitted: {n_fit}    skipped: {n_skip}    low_n: {n_low}    ci_warned: {n_warn}")
    print(f"A2-beats-A1 by >5% (candidates for future per-cell switch): {a2_winners or 'none'}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_corpus(Path(args.corpus), args.window_days)

    unique_dates = pd.to_datetime(df["forecast_date"]).nunique() if len(df) else 0
    fit_date = _resolve_fit_date(df).isoformat()

    bootstrap_resample = (
        BOOTSTRAP_RESAMPLE_DATE_CLUSTER if args.cluster_by_date else BOOTSTRAP_RESAMPLE_ROW
    )
    if args.cluster_by_date:
        logger.warning(
            "unstable_b/unstable_d width thresholds were tuned on row-level CIs; expect more "
            "unstable_* warnings under --cluster-by-date until they are re-derived "
            "(docs/calibration-runbook.md)."
        )

    cells_coef: dict[str, dict[str, object]] = {}
    cells_diag: dict[str, dict[str, object]] = {}

    for (city, var), cell_df in df.groupby(["city", "var"]):
        key = f"{city}/{var}"
        coef, diag = fit_one_cell(
            cell_df.reset_index(drop=True),
            seed=args.seed,
            bootstrap_iters=args.bootstrap_iters,
            cluster_by_date=args.cluster_by_date,
            cell=key,
        )
        if args.cluster_by_date and coef.get("skipped"):
            logger.warning("skipping %s: %s", key, coef["reason"])
        # Persist the 99th percentile of training ens_std for the runtime
        # extrapolation guard (EmosTable.training_ens_std_p99). Computed on the
        # same rows fit_one_cell pools across (all leads for this city/var).
        # The corpus's ens_std column is recomputed with ddof=1 in
        # scripts/build_emos_corpus.py, matching the fitter's expectations.
        ens_std_col = cell_df["ens_std"].to_numpy()
        if ens_std_col.size > 0 and np.all(np.isfinite(ens_std_col)):
            coef["ens_std_p99"] = float(np.percentile(ens_std_col, 99))
        cells_coef[key] = coef
        cells_diag[key] = diag

    coefficients = {
        "fit_date": fit_date,
        "training_window_days": int(unique_dates),
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": cells_coef,
    }
    diagnostics = {
        "fit_date": fit_date,
        "fitter_version": FITTER_VERSION,
        "cv_protocol": CV_PROTOCOL,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_resample": bootstrap_resample,
        "random_seed": args.seed,
        "schema_version": 1,
        "cells": cells_diag,
    }

    _atomic_write(out_dir / "coefficients.json", json.dumps(coefficients, indent=2, default=float))
    _archive_coefficients(
        out_dir / "coefficients.json",
        out_dir / "coefficients_history",
        retention=180,
    )
    _atomic_write(
        out_dir / "coefficients.diagnostics.json",
        json.dumps(diagnostics, indent=2, default=float),
    )
    _print_report(cells_coef, cells_diag, bootstrap_resample)


if __name__ == "__main__":
    main()
