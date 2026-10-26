"""Stage 0a parent-corpus salvage — quantify frame-v1 contamination.

The parent strategy ``intraday_asos_high`` accrued its forward shadow corpus
under the LEGACY CIVIL (DST-aware) climate-day frame (``frame_version`` 1 or
absent). The LST frame fix (§0a) corrects future rows, but the historical rows
are not wholesale discarded: the frame error is ROW-IDENTIFIABLE. For most rows
the contaminated 00:00–01:00 LDT boundary hour does NOT hold the running
extremum ``M`` at evaluation time, so ``M`` is unchanged between frames and the
row is poolable; only rows whose ``M`` actually differs are quarantined.

This script, for each frame-v1 prediction row in a ``predictions.jsonl``:

  1. recomputes ``M`` under BOTH frames from the local ASOS CSVs
     (``data/research/asos/``), as the running max over observations on the
     row's climate day (``expiration_date``) up to the row's evaluation
     ``timestamp``;
  2. compares the recomputed LST-frame ``M`` with the logged ``asos_M``
     (and with the recomputed civil-frame ``M``) to flag ``frame_affected``;
  3. writes a per-row JSON report and prints a console summary: total rows,
     deduped settlement events, how many events have ANY affected row, and how
     many of those fall in the low-p (posterior < 0.10) NO bin — i.e. how much
     of the parent's reliability-gate accrual is actually contaminated.

The operator runs this against the VPS ``predictions.jsonl`` later; it must work
on any local predictions file. CSV coverage gaps (a row outside the local CSV
window) are reported as ``recompute_status="no_csv_coverage"`` and excluded from
the affected counts (they cannot be adjudicated locally).

Run:
    uv run python scripts/salvage_frame_v1_predictions.py \
        --predictions state/predictions.jsonl
    uv run python scripts/salvage_frame_v1_predictions.py \
        --predictions /path/to/vps_predictions.jsonl --out data/research/frame_v1_salvage.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.prediction_log import PredictionRecord, _deserialize  # noqa: E402
from python.data.asos import ASOS_STATION, CITY_STD_OFFSET_HOURS  # noqa: E402
from scripts.intraday_asos_shadow_report import (  # noqa: E402
    LOW_P_THRESHOLD,
    STRATEGY_NAME,
)

DEFAULT_PREDICTIONS = _PROJECT_ROOT / "state" / "predictions.jsonl"
DEFAULT_ASOS_DIR = _PROJECT_ROOT / "data" / "research" / "asos"
DEFAULT_OUT = _PROJECT_ROOT / "data" / "research" / "frame_v1_salvage.json"

# Fixed-offset tzinfos. The CIVIL (DST-aware) frame is reconstructed via ZoneInfo
# below; the LST frame uses the canonical fixed offsets from the production map.
_CITY_LST_TZ: dict[str, timezone] = {
    city: timezone(timedelta(hours=off)) for city, off in CITY_STD_OFFSET_HOURS.items()
}

# M comparison tolerance (°F). Logged M and recomputed M can differ by a hair
# from float round-trips; only a real boundary-hour reassignment changes M by a
# full degree-ish, so anything above this is a genuine frame effect.
_M_EPS = 0.01


def _civil_tz(city: str) -> ZoneInfo:
    """Return the DST-aware civil ZoneInfo the buggy frame used, by city."""
    # NYC/MIA Eastern, CHI/AUS Central, LAX Pacific (mirrors the pre-fix map).
    return {
        "NYC": ZoneInfo("America/New_York"),
        "MIA": ZoneInfo("America/New_York"),
        "CHI": ZoneInfo("America/Chicago"),
        "AUS": ZoneInfo("America/Chicago"),
        "LAX": ZoneInfo("America/Los_Angeles"),
    }[city]


def load_city_csv(city: str, asos_dir: Path) -> list[tuple[datetime, float]]:
    """Return ``[(utc_dt, tmpf)]`` for *city* from ``k<city>_asos.csv``.

    Returns ``[]`` if the CSV is absent (the row is then unadjudicable locally).
    """
    csv_path = asos_dir / f"k{city.lower()}_asos.csv"
    if not csv_path.exists():
        return []
    rows: list[tuple[datetime, float]] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3 or row[2] in ("M", "", "T"):
                continue
            try:
                tmpf = float(row[2])
                dt = datetime.strptime(row[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except ValueError:
                continue
            rows.append((dt, tmpf))
    rows.sort(key=lambda r: r[0])
    return rows


def _recompute_m(
    rows: list[tuple[datetime, float]],
    climate_day: date,
    eval_ts: datetime,
    tz,
) -> float | None:
    """Running max tmpf over *climate_day* (in *tz*'s frame) up to *eval_ts* (UTC).

    Returns ``None`` if no qualifying observations exist.
    """
    qualifying = [
        t for (dt, t) in rows if dt <= eval_ts and dt.astimezone(tz).date() == climate_day
    ]
    return max(qualifying) if qualifying else None


def _csv_covers(rows: list[tuple[datetime, float]], climate_day: date) -> bool:
    """True iff the CSV window plausibly contains *climate_day* observations."""
    if not rows:
        return False
    first = rows[0][0].date()
    last = rows[-1][0].date()
    # Allow a one-day margin on each side (UTC vs LST boundary).
    return first <= climate_day <= last


def salvage(
    predictions_path: Path,
    asos_dir: Path,
) -> dict[str, Any]:
    """Build the per-row salvage report dict (pure aside from file reads)."""
    # Parse frame-v1 intraday_asos_high rows.
    v1_recs: list[PredictionRecord] = []
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = _deserialize(obj)
                if rec is None or rec.strategy != STRATEGY_NAME:
                    continue
                if rec.frame_version >= 2:
                    continue  # already LST — not salvage scope
                v1_recs.append(rec)

    # Cache CSVs per city.
    csv_cache: dict[str, list[tuple[datetime, float]]] = {}

    def _rows_for(city: str) -> list[tuple[datetime, float]]:
        if city not in csv_cache:
            csv_cache[city] = load_city_csv(city, asos_dir)
        return csv_cache[city]

    per_row: list[dict[str, Any]] = []
    for rec in v1_recs:
        city = rec.city
        climate_day = rec.expiration_date
        entry: dict[str, Any] = {
            "timestamp": rec.timestamp.astimezone(UTC).isoformat(),
            "market_ticker": rec.market_ticker,
            "city": city,
            "expiration_date": climate_day.isoformat() if climate_day else None,
            "logged_asos_M": rec.asos_M,
            "posterior_p_raw": rec.posterior_p_raw,
            "side": str(rec.side.value) if rec.side else None,
            "frame_affected": False,
            "recompute_status": "ok",
            "recomputed_lst_M": None,
            "recomputed_civil_M": None,
        }
        if city is None or climate_day is None or city not in ASOS_STATION:
            entry["recompute_status"] = "no_city_or_date"
            per_row.append(entry)
            continue
        rows = _rows_for(city)
        if not _csv_covers(rows, climate_day):
            entry["recompute_status"] = "no_csv_coverage"
            per_row.append(entry)
            continue

        lst_m = _recompute_m(rows, climate_day, rec.timestamp, _CITY_LST_TZ[city])
        civil_m = _recompute_m(rows, climate_day, rec.timestamp, _civil_tz(city))
        entry["recomputed_lst_M"] = lst_m
        entry["recomputed_civil_M"] = civil_m

        if lst_m is None and civil_m is None:
            entry["recompute_status"] = "no_obs_at_eval_ts"
            per_row.append(entry)
            continue

        # frame_affected ⇔ the two frames disagree on M (the boundary hour moved
        # and held/changed the running extremum at eval time).
        if lst_m is None or civil_m is None:
            entry["frame_affected"] = True
        else:
            entry["frame_affected"] = abs(lst_m - civil_m) > _M_EPS
        per_row.append(entry)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "predictions_path": str(predictions_path),
        "asos_dir": str(asos_dir),
        "n_frame_v1_rows": len(v1_recs),
        "rows": per_row,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Compute the console-summary counts from the per-row report."""
    rows = report["rows"]
    n_rows = len(rows)
    n_adjudicable = sum(1 for r in rows if r["recompute_status"] == "ok")
    n_no_coverage = sum(1 for r in rows if r["recompute_status"] == "no_csv_coverage")
    n_affected_rows = sum(1 for r in rows if r["frame_affected"])

    # Re-derive settlement events via the report's dedup, then ask which events
    # have ANY affected row, and how many of THOSE are low-p NO-bin events.
    # We dedup on (city, expiration_date, ticker) — the same key the report uses.
    affected_by_event: dict[tuple[str, str, str], bool] = {}
    lowp_no_by_event: dict[tuple[str, str, str], bool] = {}
    for r in rows:
        key = (
            r["city"] or "unknown",
            r["expiration_date"] or "unknown",
            r["market_ticker"],
        )
        affected_by_event[key] = affected_by_event.get(key, False) or bool(r["frame_affected"])
        # The shadow report's binding low-p NO bin is defined purely by
        # ``posterior_p_raw < LOW_P_THRESHOLD`` (a low YES-posterior is the NO
        # bet); it does not additionally filter on side. Match that definition.
        p = r["posterior_p_raw"]
        is_low_p_no = p is not None and p < LOW_P_THRESHOLD
        lowp_no_by_event[key] = lowp_no_by_event.get(key, False) or is_low_p_no

    n_events = len(affected_by_event)
    n_events_affected = sum(1 for v in affected_by_event.values() if v)
    n_lowp_no_events = sum(1 for v in lowp_no_by_event.values() if v)
    n_lowp_no_events_affected = sum(
        1 for k in affected_by_event if affected_by_event[k] and lowp_no_by_event[k]
    )

    return {
        "n_frame_v1_rows": n_rows,
        "n_adjudicable_rows": n_adjudicable,
        "n_no_csv_coverage_rows": n_no_coverage,
        "n_affected_rows": n_affected_rows,
        "n_settlement_events": n_events,
        "n_events_with_any_affected_row": n_events_affected,
        "n_low_p_no_bin_events": n_lowp_no_events,
        "n_low_p_no_bin_events_affected": n_lowp_no_events_affected,
    }


def render_summary(report: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "=== FRAME-v1 PREDICTION SALVAGE (Stage 0a) ===",
        f"predictions: {report['predictions_path']}",
        f"frame-v1 intraday_asos_high rows:      {summary['n_frame_v1_rows']}",
        f"  adjudicable (CSV covers row):        {summary['n_adjudicable_rows']}",
        f"  no CSV coverage (unadjudicable):     {summary['n_no_csv_coverage_rows']}",
        f"frame-AFFECTED rows (M differs):       {summary['n_affected_rows']}",
        "",
        f"deduped settlement events:             {summary['n_settlement_events']}",
        f"events with ANY frame-affected row:    {summary['n_events_with_any_affected_row']}",
        f"low-p (<{LOW_P_THRESHOLD}) NO-bin events:          "
        f"{summary['n_low_p_no_bin_events']}",
        f"  └─ of those, frame-affected:         {summary['n_low_p_no_bin_events_affected']}",
        "",
        "Interpretation: 'events with ANY affected row' is the parent corpus that "
        "must be quarantined;",
        "the low-p NO-bin affected count is the binding-gate accrual actually lost "
        "to the frame error.",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantify frame-v1 (civil-frame) contamination of the ASOS corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
        help=f"Path to predictions.jsonl (default: {DEFAULT_PREDICTIONS})",
    )
    p.add_argument(
        "--asos-dir",
        default=str(DEFAULT_ASOS_DIR),
        help=f"Directory of k<city>_asos.csv files (default: {DEFAULT_ASOS_DIR})",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Per-row JSON report output path (default: {DEFAULT_OUT})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = salvage(Path(args.predictions), Path(args.asos_dir))
    summary = report["summary"] = summarize(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(render_summary(report, summary))
    print(f"\nPer-row report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
