"""
Fit a settlement-anchored (NWS CLI) GFS bias-correction table.

Consumes the corpus produced by ``build_nws_bias_corpus.py`` and emits
``data/nws_bias/bias_correction.json`` in the EXACT schema of the ERA5-fit
``data/gfs_bias/bias_correction.json`` so the production ``BiasTable`` /
``apply_bias_correction`` runtime loads it with zero code changes.

Differences from the ERA5 fitter (analyze_gfs_bias.py):
  - Truth is NWS CLI settled temps (the source Kalshi settles on), not ERA5.
  - GFS-only (the runtime corrects only gfs_seamless).
  - EMOS-live cells are OMITTED from the table: EMOS's affine ``a + b*ens_mean``
    term already absorbs a constant mean offset, so a temperature-scale shift
    there would double-correct. Omitting them => get_bias() returns None =>
    no correction => EMOS owns the mean for its cells. Pass --exclude-cells to
    override (default tracks config emos_enabled_cells).

The SNR gate, shrinkage, and °F cap live in the runtime (bias_correction.py)
and are unchanged: cells whose NWS bias is not statistically systematic
(|t| < 2) are gated to zero automatically.

Usage:
    uv run python scripts/fit_nws_bias.py
    uv run python scripts/fit_nws_bias.py --exclude-cells LAX/low NYC/high
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

IN_DIR = Path("data/nws_bias")
FORECASTS_CSV = IN_DIR / "forecasts.csv"
ACTUALS_CSV = IN_DIR / "nws_actuals.csv"
OUT_PATH = IN_DIR / "bias_correction.json"

CITIES = ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "PHIL", "BOS"]
VARS = ["high", "low"]
CONFIG_PATH = Path("python/config.yaml")
# Fallback if config can't be read (mirrors strategies.weather.emos_enabled_cells).
_FALLBACK_EXCLUDE = ["LAX/low"]


def _emos_cells_from_config() -> list[str]:
    """Read strategies.weather.emos_enabled_cells from config.yaml.

    Single source of truth so the bias table never double-corrects an EMOS
    cell (EMOS owns its mean via a + b*ens_mean). Falls back to the known
    default if config is unreadable.
    """
    try:
        import yaml

        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        cells = cfg["strategies"]["weather"].get("emos_enabled_cells") or []
        return [str(c) for c in cells]
    except Exception:  # noqa: BLE001 - offline tool, fail soft to known default
        return list(_FALLBACK_EXCLUDE)


def _stats(errors: list[float]) -> dict | None:
    n = len(errors)
    if n == 0:
        return None
    mean = sum(errors) / n
    mae = sum(abs(e) for e in errors) / n
    rmse = sqrt(sum(e * e for e in errors) / n)
    std = sqrt(sum((e - mean) ** 2 for e in errors) / n)
    return {
        "count": n,
        "mean_bias_f": round(mean, 2),
        "mae_f": round(mae, 2),
        "rmse_f": round(rmse, 2),
        "std_bias_f": round(std, 2),
    }


def _tstat(stat: dict | None) -> float:
    """Two-sided t = mean_bias / (std / sqrt(n)). 0 when undefined."""
    if not stat or stat["count"] <= 1 or stat["std_bias_f"] <= 0:
        return 0.0
    se = stat["std_bias_f"] / sqrt(stat["count"])
    return stat["mean_bias_f"] / se if se > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exclude-cells",
        nargs="*",
        default=_emos_cells_from_config(),
        help="cells (City/var) to omit so EMOS owns their mean "
        "(default: config strategies.weather.emos_enabled_cells)",
    )
    ap.add_argument(
        "--min-month-count",
        type=int,
        default=15,
        help=(
            "drop by_month entries below this count so sparse months fall through "
            "to the well-powered 'overall' entry via the runtime fallback chain "
            "(monthly->overall->None). Prevents a low-n month being SNR-gated to "
            "zero and silently losing the correction (default 15)."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="output path for the fitted table (default: the live production table)",
    )
    args = ap.parse_args()
    exclude = {c.strip() for c in args.exclude_cells}

    if not FORECASTS_CSV.exists() or not ACTUALS_CSV.exists():
        print("ERROR: run build_nws_bias_corpus.py first.")
        return 1

    actuals: dict[tuple[str, str, str], float] = {}
    for r in csv.DictReader(open(ACTUALS_CSV)):
        actuals[(r["city"], r["date"], r["variable"])] = float(r["actual"])

    errors_overall: dict[tuple[str, str], list[float]] = defaultdict(list)
    errors_month: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    matched = unmatched = 0
    for r in csv.DictReader(open(FORECASTS_CSV)):
        if r["model"] != "gfs_seamless":
            continue
        key = (r["city"], r["date"], r["variable"])
        actual = actuals.get(key)
        if actual is None:
            unmatched += 1
            continue
        matched += 1
        err = float(r["forecast_mean"]) - actual  # +ve = forecast warm
        month = int(r["date"].split("-")[1])
        errors_overall[(r["city"], r["variable"])].append(err)
        errors_month[(r["city"], r["variable"], month)].append(err)

    table: dict = {}
    report: list[tuple[str, dict, float, bool]] = []
    for city in CITIES:
        city_block: dict = {}
        for var in VARS:
            cell = f"{city}/{var}"
            if cell in exclude:
                continue
            overall = _stats(errors_overall.get((city, var), []))
            if overall is None:
                continue
            by_month: dict[str, dict] = {}
            for month in range(1, 13):
                ms = _stats(errors_month.get((city, var, month), []))
                if ms is not None and ms["count"] >= args.min_month_count:
                    by_month[str(month)] = ms
            city_block[var] = {"overall": overall, "by_month": by_month}
            t = _tstat(overall)
            report.append((cell, overall, t, abs(t) >= 2.0))
        if city_block:
            table[city] = city_block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))

    print(f"Matched GFS pairs: {matched} (unmatched {unmatched})")
    print(f"Excluded (EMOS-owned) cells: {sorted(exclude)}")
    print(f"Wrote {args.out}\n")
    print(f"{'cell':<10} {'n':>4} {'bias°F':>8} {'mae':>6} {'rmse':>6} {'t':>7} systematic")
    print("-" * 56)
    for cell, st, t, sysmatic in sorted(report, key=lambda x: -abs(x[2])):
        flag = "YES" if sysmatic else "."
        print(
            f"{cell:<10} {st['count']:>4} {st['mean_bias_f']:>+8.2f} "
            f"{st['mae_f']:>6.2f} {st['rmse_f']:>6.2f} {t:>+7.1f}  {flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
