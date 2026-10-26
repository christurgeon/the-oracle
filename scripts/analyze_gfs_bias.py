"""
Analyze GFS deterministic forecast bias against ERA5 reanalysis actuals.

Reads the CSVs produced by collect_gfs_bias_data.py and outputs:
  - Per-city, per-variable, per-month bias statistics
  - Overall summary statistics
  - Bias correction lookup table (JSON) for use by the weather strategy

Usage:
    uv run python scripts/analyze_gfs_bias.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

DATA_DIR = Path("data/gfs_bias")
FORECAST_CSV = DATA_DIR / "gfs_forecasts.csv"
ACTUALS_CSV = DATA_DIR / "era5_actuals.csv"


@dataclass
class BiasStats:
    """Bias statistics for a group of forecast-actual pairs."""

    count: int
    mean_bias: float  # forecast - actual (positive = GFS runs warm)
    mae: float  # mean absolute error
    rmse: float  # root mean squared error
    std_bias: float  # standard deviation of bias

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_bias_f": round(self.mean_bias, 2),
            "mae_f": round(self.mae, 2),
            "rmse_f": round(self.rmse, 2),
            "std_bias_f": round(self.std_bias, 2),
        }


def compute_bias_stats(errors: list[float]) -> BiasStats:
    n = len(errors)
    if n == 0:
        return BiasStats(0, 0.0, 0.0, 0.0, 0.0)

    mean = sum(errors) / n
    mae = sum(abs(e) for e in errors) / n
    mse = sum(e ** 2 for e in errors) / n
    rmse = sqrt(mse)
    variance = sum((e - mean) ** 2 for e in errors) / n
    std = sqrt(variance)

    return BiasStats(count=n, mean_bias=mean, mae=mae, rmse=rmse, std_bias=std)


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not FORECAST_CSV.exists() or not ACTUALS_CSV.exists():
        print("ERROR: Run collect_gfs_bias_data.py first to generate the CSVs.")
        sys.exit(1)

    forecasts_raw = load_csv(FORECAST_CSV)
    actuals_raw = load_csv(ACTUALS_CSV)

    print(f"Loaded {len(forecasts_raw)} forecast rows, {len(actuals_raw)} actual rows\n")

    # Index actuals by (city, date, variable)
    actuals_idx: dict[tuple[str, str, str], float] = {}
    for row in actuals_raw:
        key = (row["city"], row["date"], row["variable"])
        actuals_idx[key] = float(row["actual_f"])

    # Join forecast to actual, compute errors
    # error = forecast - actual (positive = GFS too warm)
    errors_by_city_var: dict[tuple[str, str], list[float]] = defaultdict(list)
    errors_by_city_var_month: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    errors_by_city_var_season: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    all_errors: list[float] = []
    matched = 0
    unmatched = 0

    for row in forecasts_raw:
        key = (row["city"], row["date"], row["variable"])
        actual = actuals_idx.get(key)
        if actual is None:
            unmatched += 1
            continue

        matched += 1
        forecast = float(row["forecast_f"])
        error = forecast - actual

        month = int(row["date"].split("-")[1])
        season = {12: "DJF", 1: "DJF", 2: "DJF",
                  3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA",
                  9: "SON", 10: "SON", 11: "SON"}[month]

        errors_by_city_var[(row["city"], row["variable"])].append(error)
        errors_by_city_var_month[(row["city"], row["variable"], month)].append(error)
        errors_by_city_var_season[(row["city"], row["variable"], season)].append(error)
        all_errors.append(error)

    print(f"Matched pairs: {matched}, Unmatched forecasts: {unmatched}\n")

    # ---------------------------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------------------------
    overall = compute_bias_stats(all_errors)
    print("=" * 70)
    print("OVERALL GFS BIAS (all cities, all variables)")
    print("=" * 70)
    print(f"  N = {overall.count}")
    print(f"  Mean bias:  {overall.mean_bias:+.2f}°F  (positive = GFS runs warm)")
    print(f"  MAE:        {overall.mae:.2f}°F")
    print(f"  RMSE:       {overall.rmse:.2f}°F")
    print(f"  Std(bias):  {overall.std_bias:.2f}°F")
    print()

    # ---------------------------------------------------------------------------
    # Per-city, per-variable summary
    # ---------------------------------------------------------------------------
    var_labels = {
        "temperature_2m_max": "HIGH",
        "temperature_2m_min": "LOW",
    }

    print("=" * 70)
    print("PER-CITY, PER-VARIABLE BIAS")
    print("=" * 70)
    print(f"{'City':<5} {'Var':<5} {'N':>5} {'Bias':>8} {'MAE':>7} {'RMSE':>7} {'Std':>7}")
    print("-" * 50)

    for city in ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA"]:
        for var in ["temperature_2m_max", "temperature_2m_min"]:
            errors = errors_by_city_var.get((city, var), [])
            stats = compute_bias_stats(errors)
            label = var_labels[var]
            print(
                f"{city:<5} {label:<5} {stats.count:>5} "
                f"{stats.mean_bias:>+7.2f} {stats.mae:>7.2f} "
                f"{stats.rmse:>7.2f} {stats.std_bias:>7.2f}"
            )
        print()

    # ---------------------------------------------------------------------------
    # Per-city, per-variable, per-season
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("SEASONAL BIAS (DJF=winter, MAM=spring, JJA=summer, SON=fall)")
    print("=" * 70)

    for city in ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA"]:
        for var in ["temperature_2m_max", "temperature_2m_min"]:
            label = var_labels[var]
            print(f"\n  {city} {label}:")
            print(f"    {'Season':<8} {'N':>5} {'Bias':>8} {'MAE':>7} {'RMSE':>7}")
            print(f"    {'-'*35}")
            for season in ["DJF", "MAM", "JJA", "SON"]:
                errors = errors_by_city_var_season.get((city, var, season), [])
                if not errors:
                    continue
                stats = compute_bias_stats(errors)
                print(
                    f"    {season:<8} {stats.count:>5} "
                    f"{stats.mean_bias:>+7.2f} {stats.mae:>7.2f} {stats.rmse:>7.2f}"
                )

    # ---------------------------------------------------------------------------
    # Monthly bias table (for fine-grained correction)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MONTHLY MEAN BIAS (°F) — rows=city+var, cols=month")
    print("=" * 70)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = f"{'City':<5} {'Var':<5} " + " ".join(f"{m:>6}" for m in month_names)
    print(header)
    print("-" * len(header))

    for city in ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA"]:
        for var in ["temperature_2m_max", "temperature_2m_min"]:
            label = var_labels[var]
            biases = []
            for month in range(1, 13):
                errors = errors_by_city_var_month.get((city, var, month), [])
                if errors:
                    bias = sum(errors) / len(errors)
                    biases.append(f"{bias:>+6.1f}")
                else:
                    biases.append(f"{'N/A':>6}")
            print(f"{city:<5} {label:<5} " + " ".join(biases))
        print()

    # ---------------------------------------------------------------------------
    # Export bias correction table as JSON
    # ---------------------------------------------------------------------------
    bias_table: dict = {}
    for city in ["NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA"]:
        bias_table[city] = {}
        for var in ["temperature_2m_max", "temperature_2m_min"]:
            label = var_labels[var].lower()
            bias_table[city][label] = {
                "overall": compute_bias_stats(
                    errors_by_city_var.get((city, var), [])
                ).to_dict(),
                "by_month": {},
            }
            for month in range(1, 13):
                errors = errors_by_city_var_month.get((city, var, month), [])
                if errors:
                    stats = compute_bias_stats(errors)
                    bias_table[city][label]["by_month"][str(month)] = stats.to_dict()

    output_path = DATA_DIR / "bias_correction.json"
    with open(output_path, "w") as f:
        json.dump(bias_table, f, indent=2)
    print(f"\nBias correction table written to {output_path}")

    # ---------------------------------------------------------------------------
    # Actionable summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ACTIONABLE FINDINGS")
    print("=" * 70)

    # Find worst biases
    worst = []
    for (city, var), errors in errors_by_city_var.items():
        stats = compute_bias_stats(errors)
        worst.append((city, var_labels[var], stats))
    worst.sort(key=lambda x: abs(x[2].mean_bias), reverse=True)

    print("\nLargest systematic biases (by mean error):")
    for city, label, stats in worst[:5]:
        direction = "warm" if stats.mean_bias > 0 else "cool"
        print(
            f"  {city} {label}: GFS runs {abs(stats.mean_bias):.1f}°F too {direction} "
            f"(MAE={stats.mae:.1f}°F, N={stats.count})"
        )

    print("\nLargest forecast uncertainty (by RMSE):")
    worst_rmse = sorted(worst, key=lambda x: x[2].rmse, reverse=True)
    for city, label, stats in worst_rmse[:5]:
        print(f"  {city} {label}: RMSE={stats.rmse:.1f}°F (std={stats.std_bias:.1f}°F)")

    # Impact assessment
    print("\nIMPACT ON TRADING:")
    print("  - Bias correction shifts the ensemble mean before computing bracket fractions")
    print("  - A 2°F bias on a 5°F-wide bracket can shift probability by ~40%")
    print("  - Cities with MAE > 3°F need aggressive correction or wider edge thresholds")
    print("  - Seasonal variation means corrections should use monthly tables, not annual")


if __name__ == "__main__":
    main()
