"""New-city same-day spread census — reproducible evidence + compression watch.

Measures, per snapshot (date, slot, city), the quoted spread distribution of
SAME-DAY temperature markets (ticker obs-date == snapshot dir date) with an
active mid (mid ∈ [0.05, 0.95]) — the evidence base and spread-compression
watch for the new-city intraday lock-in research.

Definition is FIXED here; do not re-slice ad hoc. Reports per-city avg,
median, and max spread plus pooled new-vs-mature comparison, per day and
over the trailing window.

Usage: uv run python scripts/research_2026_06_09_newcity_spread_census.py [--days N]
Output: data/research/newcity_spread_census.json + console table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
SNAP_DIR = REPO / "data" / "snapshots"
OUT_PATH = REPO / "data" / "research" / "newcity_spread_census.json"

MATURE = {"NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "PHIL", "BOS"}
SLOTS = ("0200", "0800", "1400", "2000")
MID_LO, MID_HI = 0.05, 0.95


def day_ticker_token(snap_day: str) -> str:
    """'2026-06-09' -> '26JUN09' (Kalshi ticker date segment)."""
    y, m, d = snap_day.split("-")
    mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"][int(m) - 1]
    return f"{y[2:]}{mon}{d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()

    days = sorted(p.name for p in SNAP_DIR.iterdir() if p.is_dir())[-args.days:]
    per_day: dict[str, dict[str, list[float]]] = {}
    for snap_day in days:
        token = day_ticker_token(snap_day)
        spreads: dict[str, list[float]] = defaultdict(list)
        for slot in SLOTS:
            mpath = SNAP_DIR / snap_day / slot / "markets.json"
            if not mpath.exists():
                continue
            try:
                markets = json.loads(mpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for m in markets:
                city = m.get("city")
                bid, ask = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
                if (
                    not city
                    or bid is None
                    or ask is None
                    or token not in m.get("ticker", "")
                ):
                    continue
                if MID_LO <= (bid + ask) / 2 <= MID_HI:
                    spreads[city].append(round(ask - bid, 4))
        if spreads:
            per_day[snap_day] = dict(spreads)

    def stats(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "avg_c": round(100 * sum(vals) / len(vals), 1),
            "median_c": round(100 * median(vals), 1),
            "max_c": round(100 * max(vals), 1),
        } if vals else {"n": 0}

    report = {"definition": {
        "same_day_only": True, "mid_range": [MID_LO, MID_HI], "slots": SLOTS,
    }, "per_day": {}}
    print(f"{'day':<12}{'cohort':<8}{'n':>5}{'avg':>7}{'med':>7}{'max':>7}")
    for snap_day, spreads in per_day.items():
        new = [s for c, v in spreads.items() if c not in MATURE for s in v]
        mat = [s for c, v in spreads.items() if c in MATURE for s in v]
        report["per_day"][snap_day] = {
            "new_pooled": stats(new),
            "mature_pooled": stats(mat),
            "by_city": {c: stats(v) for c, v in sorted(spreads.items())},
        }
        for label, vals in (("new", new), ("mature", mat)):
            st = stats(vals)
            if st["n"]:
                print(f"{snap_day:<12}{label:<8}{st['n']:>5}{st['avg_c']:>6.1f}c"
                      f"{st['median_c']:>6.1f}c{st['max_c']:>6.1f}c")

    # trailing pooled view (most recent 7 days with new-city data)
    recent = [d for d in per_day if report["per_day"][d]["new_pooled"]["n"]][-7:]
    pooled_new = [s for d in recent for c, v in per_day[d].items()
                  if c not in MATURE for s in v]
    pooled_mat = [s for d in recent for c, v in per_day[d].items()
                  if c in MATURE for s in v]
    report["trailing_7d"] = {
        "days": recent,
        "new_pooled": stats(pooled_new),
        "mature_pooled": stats(pooled_mat),
        "by_city": {},
    }
    bycity: dict[str, list[float]] = defaultdict(list)
    for d in recent:
        for c, v in per_day[d].items():
            bycity[c].extend(v)
    report["trailing_7d"]["by_city"] = {c: stats(v) for c, v in sorted(bycity.items())}
    print("\ntrailing window by city (new cities):")
    for c, st in report["trailing_7d"]["by_city"].items():
        if c not in MATURE:
            print(f"  {c}: {st}")
    print(f"\ntrailing pooled: new={report['trailing_7d']['new_pooled']}"
          f" mature={report['trailing_7d']['mature_pooled']}")

    OUT_PATH.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
