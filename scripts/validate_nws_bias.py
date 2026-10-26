"""
Out-of-sample validation of the settlement-anchored (NWS) bias correction.

Leak-safe walk-forward: fit the bias table on a TRAIN window, then apply the
*production* correction code (apply_bias_correction with production shrinkage /
SNR gate / °F cap) to a held-out TEST window. Compares three regimes per cell:

  none  — raw GFS ensemble mean, no correction (baseline)
  era5  — the incumbent ERA5-fit table (data/gfs_bias/bias_correction.json)
  nws   — the NWS-CLI table fit on TRAIN only (this proposal)

Metrics on the TEST set: mean signed error, MAE, RMSE, and ±1°F hit-rate of the
rounded mean (a proxy for landing in the correct 1°F settlement bracket).

A cell is PROMOTED only if nws beats BOTH none and era5 on test MAE — this is
the per-cell ship gate and directly answers the "recentering just moves us
toward the market" objection empirically (a cell that doesn't improve OOS is
not shipped).

Usage:
    uv run python scripts/validate_nws_bias.py
    uv run python scripts/validate_nws_bias.py --test-start 2026-05-01
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from math import sqrt
from pathlib import Path

from python.data.bias_correction import BiasTable, apply_bias_correction
from python.data.weather import EnsembleForecast

IN_DIR = Path("data/nws_bias")
FORECASTS_CSV = IN_DIR / "forecasts.csv"
ACTUALS_CSV = IN_DIR / "nws_actuals.csv"
ERA5_TABLE = Path("data/gfs_bias/bias_correction.json")
SHRINKAGE = 0.3
CAP_F = 5.0

from datetime import date  # noqa: E402


def _pd(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def _fit_table(rows: list[dict]) -> dict:
    """Build a bias_correction.json-shaped dict from train rows (GFS)."""
    errs_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    errs_month: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        errs_overall[(r["city"], r["variable"])].append(r["err"])
        errs_month[(r["city"], r["variable"], r["month"])].append(r["err"])

    def stat(e: list[float]) -> dict | None:
        n = len(e)
        if n == 0:
            return None
        mean = sum(e) / n
        std = sqrt(sum((x - mean) ** 2 for x in e) / n)
        return {
            "count": n,
            "mean_bias_f": round(mean, 2),
            "mae_f": round(sum(abs(x) for x in e) / n, 2),
            "rmse_f": round(sqrt(sum(x * x for x in e) / n), 2),
            "std_bias_f": round(std, 2),
        }

    table: dict = {}
    for city, var in errs_overall:
        ov = stat(errs_overall[(city, var)])
        if ov is None:
            continue
        bm = {}
        for m in range(1, 13):
            s = stat(errs_month.get((city, var, m), []))
            if s:
                bm[str(m)] = s
        table.setdefault(city, {})[var] = {"overall": ov, "by_month": bm}
    return table


def _correction_for(table: BiasTable, city: str, var: str, fdate: date, raw_mean: float) -> float:
    """Run the production correction; return the °F shift applied to the mean."""
    fc = EnsembleForecast(city=city, forecast_date=fdate, temperatures=[raw_mean])
    _, corr = apply_bias_correction(fc, table, var, shrinkage=SHRINKAGE, max_correction_f=CAP_F)
    return corr


def _metrics(errs: list[float]) -> tuple[float, float, float, float]:
    n = len(errs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    mean = sum(errs) / n
    mae = sum(abs(e) for e in errs) / n
    rmse = sqrt(sum(e * e for e in errs) / n)
    hit = sum(1 for e in errs if abs(e) <= 1.0) / n  # within ±1°F
    return (mean, mae, rmse, hit)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2026-05-01", help="first TEST date (YYYY-MM-DD)")
    args = ap.parse_args()
    cutoff = _pd(args.test_start)

    actuals: dict[tuple[str, str, str], float] = {}
    for r in csv.DictReader(open(ACTUALS_CSV)):
        actuals[(r["city"], r["date"], r["variable"])] = float(r["actual"])

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for r in csv.DictReader(open(FORECASTS_CSV)):
        if r["model"] != "gfs_seamless":
            continue
        a = actuals.get((r["city"], r["date"], r["variable"]))
        if a is None:
            continue
        d = _pd(r["date"])
        rec = {
            "city": r["city"],
            "var": r["variable"],
            "variable": r["variable"],
            "date": d,
            "month": d.month,
            "mean": float(r["forecast_mean"]),
            "actual": a,
            "err": float(r["forecast_mean"]) - a,
        }
        (test_rows if d >= cutoff else train_rows).append(rec)

    if not train_rows or not test_rows:
        print("ERROR: empty train or test split — adjust --test-start.")
        return 1

    # Fit NWS table on TRAIN only, write to a temp file, load via production BiasTable.
    nws_train = _fit_table(train_rows)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(nws_train, tf)
        tmp_path = Path(tf.name)
    nws_tbl = BiasTable(tmp_path)
    era5_tbl = BiasTable(ERA5_TABLE)

    # Apply each regime to the TEST set.
    per_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"none": [], "era5": [], "nws": []}
    )
    for r in test_rows:
        cell = (r["city"], r["var"])
        per_cell[cell]["none"].append(r["err"])
        c_era5 = _correction_for(era5_tbl, r["city"], r["var"], r["date"], r["mean"])
        c_nws = _correction_for(nws_tbl, r["city"], r["var"], r["date"], r["mean"])
        per_cell[cell]["era5"].append((r["mean"] + c_era5) - r["actual"])
        per_cell[cell]["nws"].append((r["mean"] + c_nws) - r["actual"])

    print(f"Train n={len(train_rows)}  Test n={len(test_rows)}  (test >= {cutoff})\n")
    hdr = (
        f"{'cell':<10} {'n':>3} | {'MAE none':>8} {'era5':>6} {'nws':>6} "
        f"| {'hit none':>8} {'nws':>6} | verdict"
    )
    print(hdr)
    print("-" * len(hdr))

    pooled = {"none": [], "era5": [], "nws": []}
    promoted = []
    for cell in sorted(per_cell):
        d = per_cell[cell]
        n = len(d["none"])
        _, mae_n, _, hit_n = _metrics(d["none"])
        _, mae_e, _, _ = _metrics(d["era5"])
        _, mae_w, _, hit_w = _metrics(d["nws"])
        for k in pooled:
            pooled[k].extend(d[k])
        improves = mae_w < mae_n - 1e-9 and mae_w <= mae_e + 1e-9
        verdict = "SHIP" if improves else ("hurts" if mae_w > mae_n + 1e-9 else "neutral")
        if improves:
            promoted.append(f"{cell[0]}/{cell[1]}")
        print(
            f"{cell[0] + '/' + cell[1]:<10} {n:>3} | {mae_n:>8.2f} {mae_e:>6.2f} {mae_w:>6.2f} "
            f"| {hit_n:>7.0%} {hit_w:>5.0%} | {verdict}"
        )

    print("-" * len(hdr))
    _, p_n, r_n, h_n = _metrics(pooled["none"])
    _, p_e, _, _ = _metrics(pooled["era5"])
    _, p_w, r_w, h_w = _metrics(pooled["nws"])
    print(
        f"{'POOLED':<10} {len(pooled['none']):>3} | {p_n:>8.2f} {p_e:>6.2f} {p_w:>6.2f} "
        f"| {h_n:>7.0%} {h_w:>5.0%} |"
    )
    print(f"\nPooled RMSE: none={r_n:.2f}  nws={r_w:.2f}   (Δ {r_w - r_n:+.2f}°F)")
    print(
        f"Pooled MAE : none={p_n:.2f}  era5={p_e:.2f}  nws={p_w:.2f}   "
        f"(Δ vs none {p_w - p_n:+.2f}°F)"
    )
    print(f"\nPROMOTE (OOS beats none AND era5): {', '.join(promoted) if promoted else 'NONE'}")
    tmp_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
