"""
Build a settlement-anchored bias corpus: GFS/ECMWF ensemble means paired with
the *NWS CLI* settled temperatures Kalshi actually settles on (NOT ERA5).

This is the NWS-CLI analogue of ``collect_gfs_bias_data.py`` (which pairs GFS
against ERA5 reanalysis). It reads the existing shadow snapshot corpus rather
than calling any external API, so it is hermetic and reproducible.

Pairing rule (matches the production trading horizon):
  - lead-1: for observation_date D, use the snapshot COLLECTED on D-1, the
    latest run of that day, the forecast record with forecast_date == D.
  - truth: the FINAL (``is_preliminary == False``) NWS CLI settlement for
    observation_date D, taken from whichever snapshot recorded it. Preliminary
    settlements are ignored when a final exists.

Outputs (mirroring data/gfs_bias/*.csv column shape, plus a ``model`` column):
  - data/nws_bias/forecasts.csv   columns: city,date,variable,model,forecast_mean
  - data/nws_bias/nws_actuals.csv columns: city,date,variable,actual

``variable`` is the canonical bias key: "high" (temperature_2m_max) /
"low" (temperature_2m_min), matching bias_correction.py's _VARIABLE_TO_BIAS_KEY.

Usage:
    uv run python scripts/build_nws_bias_corpus.py
    uv run python scripts/build_nws_bias_corpus.py --lead 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

SNAPSHOTS_DIR = Path("data/snapshots")
OUT_DIR = Path("data/nws_bias")

# Open-Meteo daily_variable string -> canonical bias key used by the runtime.
_VAR_TO_KEY = {
    "temperature_2m_max": "high",
    "temperature_2m_min": "low",
}


def _parse_date(s: str) -> date | None:
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except (ValueError, AttributeError):
        return None


def _iter_snapshot_days() -> list[date]:
    days = []
    for child in sorted(SNAPSHOTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        dt = _parse_date(child.name)
        if dt is not None:
            days.append(dt)
    return days


def _latest_run_forecasts(day: date) -> Path | None:
    """Return the weather_forecasts.json for the latest run of *day*, or None."""
    day_dir = SNAPSHOTS_DIR / day.isoformat()
    if not day_dir.is_dir():
        return None
    best: tuple[str, Path] | None = None
    for run_dir in day_dir.iterdir():
        if not run_dir.is_dir():
            continue
        fc = run_dir / "weather_forecasts.json"
        if fc.exists():
            # Run dirs are zero-padded HHMM strings; lexical max == latest.
            if best is None or run_dir.name > best[0]:
                best = (run_dir.name, fc)
    return best[1] if best else None


def _collect_final_settlements() -> dict[tuple[str, date, str], float]:
    """Map (city, observation_date, var_key) -> settled temp (°F), final-preferred.

    Scans every settlements.json in the corpus. A final reading
    (is_preliminary == False) always overrides a preliminary one for the
    same key; preliminary is used only as a last resort.
    """
    final: dict[tuple[str, date, str], float] = {}
    prelim: dict[tuple[str, date, str], float] = {}
    for path in SNAPSHOTS_DIR.glob("*/*/settlements.json"):
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            city = row.get("city")
            obs = _parse_date(row.get("observation_date", ""))
            if city is None or obs is None:
                continue
            is_prelim = bool(row.get("is_preliminary", False))
            for var_key, field in (("high", "high_f"), ("low", "low_f")):
                val = row.get(field)
                if val is None:
                    continue
                key = (city, obs, var_key)
                sink = prelim if is_prelim else final
                sink[key] = float(val)
    # Final wins; fall back to preliminary only where no final exists.
    merged = dict(prelim)
    merged.update(final)
    return merged


def build(lead: int) -> tuple[list[dict], dict[tuple[str, date, str], float]]:
    """Return (forecast_rows, actuals) joined on (city, forecast_date, var)."""
    actuals = _collect_final_settlements()
    snapshot_days = set(_iter_snapshot_days())

    # forecast_rows: one per (city, forecast_date, var_key, model) that has a
    # matching final settlement. Keyed so a later snapshot for the same target
    # at the same lead overwrites (deterministic: we want lead-`lead` exactly).
    rows: dict[tuple[str, date, str, str], float] = {}

    for city, obs_date, var_key in actuals:
        collect_day = obs_date - timedelta(days=lead)
        if collect_day not in snapshot_days:
            continue
        fc_path = _latest_run_forecasts(collect_day)
        if fc_path is None:
            continue
        try:
            records = json.loads(fc_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for rec in records:
            if rec.get("city") != city:
                continue
            if rec.get("forecast_date") != obs_date.isoformat():
                continue
            rkey = _VAR_TO_KEY.get(rec.get("daily_variable", ""))
            if rkey != var_key:
                continue
            model = rec.get("model")
            mean = rec.get("mean")
            if model is None or mean is None:
                continue
            rows[(city, obs_date, var_key, model)] = float(mean)

    forecast_rows = [
        {"city": c, "date": d.isoformat(), "variable": v, "model": m, "forecast_mean": fm}
        for (c, d, v, m), fm in sorted(rows.items())
    ]
    return forecast_rows, actuals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", type=int, default=1, help="forecast lead in days (default 1)")
    args = ap.parse_args()

    forecast_rows, actuals = build(args.lead)
    if not forecast_rows:
        print("No paired forecast/settlement rows found — check snapshot corpus.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fc_path = OUT_DIR / "forecasts.csv"
    with open(fc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["city", "date", "variable", "model", "forecast_mean"])
        w.writeheader()
        w.writerows(forecast_rows)

    # Emit actuals only for the dates that appear in the forecast join.
    used_keys = {(r["city"], r["date"], r["variable"]) for r in forecast_rows}
    act_path = OUT_DIR / "nws_actuals.csv"
    with open(act_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "date", "variable", "actual"])
        for (city, obs, var_key), val in sorted(actuals.items()):
            if (city, obs.isoformat(), var_key) in used_keys:
                w.writerow([city, obs.isoformat(), var_key, val])

    # Quick coverage summary.
    by_model: dict[str, int] = defaultdict(int)
    by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for r in forecast_rows:
        if r["model"] == "gfs_seamless":
            by_cell[(r["city"], r["variable"])] += 1
        by_model[r["model"]] += 1
    print(f"Wrote {fc_path} ({len(forecast_rows)} model-rows) and {act_path}")
    print(f"  lead={args.lead}  models={dict(by_model)}")
    print("  GFS pairs per cell:")
    for (city, var), n in sorted(by_cell.items()):
        print(f"    {city}/{var}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
