"""
Collect 2 years of GFS deterministic forecasts and ERA5 reanalysis observations
from Open-Meteo for all 7 weather cities.

Produces two CSVs in data/gfs_bias/:
  - gfs_forecasts.csv  (what GFS predicted)
  - era5_actuals.csv   (what actually happened per ERA5 reanalysis)

These are joined by (city, date, variable) to compute bias curves.

Usage:
    uv run python scripts/collect_gfs_bias_data.py
    uv run python scripts/collect_gfs_bias_data.py --years 1  # shorter range
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CITIES: dict[str, tuple[float, float]] = {
    "NYC": (40.77898, -73.96925),
    "CHI": (41.78412, -87.75514),
    "MIA": (25.78805, -80.31694),
    "LAX": (33.93816, -118.38660),
    "DEN": (39.84657, -104.65623),
    "AUS": (30.18311, -97.67989),
    "SEA": (47.44467, -122.31442),
}

VARIABLES = ["temperature_2m_max", "temperature_2m_min"]

HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data/gfs_bias")

# Open-Meteo free tier: be polite
REQUEST_DELAY = 1.0  # seconds between requests


def fetch_historical_forecasts(
    city: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    client: httpx.Client,
) -> list[dict]:
    """Fetch GFS deterministic forecasts for a city over a date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(VARIABLES),
        "temperature_unit": "fahrenheit",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "models": "gfs_seamless",
    }

    resp = client.get(HIST_FORECAST_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    times = daily.get("time", [])
    rows = []
    for i, dt_str in enumerate(times):
        for var in VARIABLES:
            vals = daily.get(var, [])
            val = vals[i] if i < len(vals) else None
            if val is not None:
                rows.append({
                    "city": city,
                    "date": dt_str,
                    "variable": var,
                    "forecast_f": val,
                })
    return rows


def fetch_era5_actuals(
    city: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    client: httpx.Client,
) -> list[dict]:
    """Fetch ERA5 reanalysis observations for a city over a date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(VARIABLES),
        "temperature_unit": "fahrenheit",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    resp = client.get(ARCHIVE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    times = daily.get("time", [])
    rows = []
    for i, dt_str in enumerate(times):
        for var in VARIABLES:
            vals = daily.get(var, [])
            val = vals[i] if i < len(vals) else None
            if val is not None:
                rows.append({
                    "city": city,
                    "date": dt_str,
                    "variable": var,
                    "actual_f": val,
                })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GFS bias data from Open-Meteo")
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Number of years of history to fetch (default: 2)",
    )
    args = parser.parse_args()

    # Date range: from N years ago to 5 days ago (ERA5 may lag a few days)
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=int(args.years * 365))

    print(f"Collecting GFS bias data: {start_date} to {end_date}")
    print(f"Cities: {list(CITIES.keys())}")
    print(f"Variables: {VARIABLES}")
    print()

    all_forecasts: list[dict] = []
    all_actuals: list[dict] = []

    with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
        for city, (lat, lon) in CITIES.items():
            print(f"[{city}] Fetching GFS forecasts...")
            try:
                forecasts = fetch_historical_forecasts(
                    city, lat, lon, start_date, end_date, client,
                )
                all_forecasts.extend(forecasts)
                print(f"  {len(forecasts)} forecast rows")
            except httpx.HTTPStatusError as e:
                print(f"  ERROR: {e.response.status_code} — {e.response.text[:200]}")
                sys.exit(1)

            time.sleep(REQUEST_DELAY)

            print(f"[{city}] Fetching ERA5 actuals...")
            try:
                actuals = fetch_era5_actuals(
                    city, lat, lon, start_date, end_date, client,
                )
                all_actuals.extend(actuals)
                print(f"  {len(actuals)} actual rows")
            except httpx.HTTPStatusError as e:
                print(f"  ERROR: {e.response.status_code} — {e.response.text[:200]}")
                sys.exit(1)

            time.sleep(REQUEST_DELAY)
            print()

    # Write CSVs
    write_csv(
        OUTPUT_DIR / "gfs_forecasts.csv",
        all_forecasts,
        ["city", "date", "variable", "forecast_f"],
    )
    write_csv(
        OUTPUT_DIR / "era5_actuals.csv",
        all_actuals,
        ["city", "date", "variable", "actual_f"],
    )

    print(f"\nDone. Files in {OUTPUT_DIR}/")
    print(f"  Forecast rows: {len(all_forecasts)}")
    print(f"  Actual rows:   {len(all_actuals)}")


if __name__ == "__main__":
    main()
