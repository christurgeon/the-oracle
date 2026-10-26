"""Build (city,var,month)->(mu,sigma) daily-high/low climatology for favorite_bias.

Aggregates NWS settlements from BOTH snapshot layouts:
  data/snapshots/<date>/settlements.json        (legacy flat)
  data/snapshots/<date>/<HHMM>/settlements.json (current 4x-daily nested)
dedup by (city, observation_date) preferring final over preliminary. Writes a
month cell per (city,var,month) with n >= min_month_n, plus an all-months
(city,var)_pool cell. Atomic write.

Look-ahead: pass --as-of YYYY-MM-DD to include only observations strictly
before that date (for leave-future-out backtest replay). The live-deployed
artifact uses all settlements to date, which IS ex-ante for live rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.utils.atomic_write import atomic_write_text  # noqa: E402

_DEFAULT_OUT = "data/calibration/fb_climatology.json"


def _load_settlements(as_of: str | None) -> list[dict]:
    files = sorted(
        set(glob.glob("data/snapshots/*/settlements.json"))
        | set(glob.glob("data/snapshots/*/*/settlements.json"))
    )
    rows: list[dict] = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        for r in data if isinstance(data, list) else []:
            od = r.get("observation_date")
            if od is None or r.get("high_f") is None:
                continue
            if as_of is not None and od >= as_of:
                continue
            rows.append(r)
    return rows


def aggregate_cells(rows: list[dict], *, min_month_n: int = 5) -> dict[str, dict]:
    # dedup (city, obs_date) preferring final (is_preliminary is False)
    seen: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r.get("city"), r.get("observation_date"))
        if None in k:
            continue
        prev = seen.get(k)
        if (
            prev is not None
            and prev.get("is_preliminary") is False
            and r.get("is_preliminary") is not False
        ):
            continue
        seen[k] = r

    # bucket temps by (city,var,month) and (city,var) pool
    month_buckets: dict[str, list[float]] = {}
    pool_buckets: dict[str, list[float]] = {}
    for (city, od), r in seen.items():
        month = int(od[5:7])
        for var, field in (("high", "high_f"), ("low", "low_f")):
            val = r.get(field)
            if val is None:
                continue
            month_buckets.setdefault(f"{city}_{var}_{month}", []).append(float(val))
            pool_buckets.setdefault(f"{city}_{var}_pool", []).append(float(val))

    def _stats(vals: list[float]) -> dict | None:
        if len(vals) < 2:
            return None
        return {
            "mu": round(statistics.fmean(vals), 3),
            "sigma": round(statistics.stdev(vals), 3),
            "n": len(vals),
        }

    cells: dict[str, dict] = {}
    for key, vals in month_buckets.items():
        if len(vals) < min_month_n:
            continue
        s = _stats(vals)
        if s is not None and s["sigma"] > 0:
            cells[key] = s
    for key, vals in pool_buckets.items():
        s = _stats(vals)
        if s is not None and s["sigma"] > 0:
            cells[key] = s
    return cells


def build_artifact(rows: list[dict], *, min_month_n: int = 5) -> dict:
    return {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "source": "nws_snapshots",
        "min_month_n": min_month_n,
        "cells": aggregate_cells(rows, min_month_n=min_month_n),
    }


def _write_atomic(path: str, artifact: dict) -> None:
    atomic_write_text(Path(path), json.dumps(artifact, indent=1, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--as-of", default=None, help="include only obs strictly before YYYY-MM-DD")
    ap.add_argument("--min-month-n", type=int, default=5)
    args = ap.parse_args()
    rows = _load_settlements(args.as_of)
    art = build_artifact(rows, min_month_n=args.min_month_n)
    _write_atomic(args.out, art)
    print(f"wrote {args.out}: {len(art['cells'])} cells from {len(rows)} obs")


if __name__ == "__main__":
    main()
