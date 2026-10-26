"""
Build the EMOS training corpus from daily snapshots.

Walks ``data/snapshots/<date>/<run_id>/weather_forecasts.json`` and
``settlements.json``, dedupes forecasts to the latest run per snapshot
day, merges settlements via ``prefer_latest_settlement``, and emits one
parquet row per ``(snapshot_date, run_id, city, forecast_date, var,
model)`` that has a settled observation.

Usage::

    uv run python scripts/build_emos_corpus.py [--snapshots DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import date as _date
from pathlib import Path

# Add project root so the script works standalone (matches fit_isotonic.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from scripts._calibration_common import (  # noqa: E402
    _CHI_SNAPSHOT_FIX_DATE,
    _VARIABLE_TO_SETTLEMENT_FIELD,
    prefer_latest_settlement,
)

logger = logging.getLogger(__name__)

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_VARIABLE_TO_VAR_SHORT = {
    "temperature_2m_max": "high",
    "temperature_2m_min": "low",
}


def _load_forecasts_per_run(
    run_dir: Path,
    *,
    min_members: int,
    allowed_models: frozenset[str] | set[str],
) -> list[dict]:
    """Read ``run_dir/weather_forecasts.json``, filter to eligible rows.

    Filters:
        * ``daily_variable`` must be one of ``temperature_2m_max`` /
          ``temperature_2m_min``
        * ``model`` must be in ``allowed_models``
        * ``len(temperatures) >= min_members``

    Returns the (filtered) raw row dicts. Returns ``[]`` if the file
    does not exist.
    """
    fc_file = run_dir / "weather_forecasts.json"
    if not fc_file.exists():
        return []

    try:
        raw = json.loads(fc_file.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Skipping malformed forecast JSON %s: %s", fc_file, exc)
        return []
    out: list[dict] = []
    for row in raw:
        if row.get("daily_variable") not in _VARIABLE_TO_VAR_SHORT:
            continue
        if row.get("model") not in allowed_models:
            continue
        members = row.get("temperatures") or []
        if len(members) < min_members:
            continue
        out.append(row)
    return out


def _iter_snapshot_runs(snapshots_dir: Path) -> Iterator[tuple[_date, str, Path]]:
    """Yield (snapshot_date, run_id, run_dir) tuples in deterministic order.

    For flat layouts (pre-cutover, where ``weather_forecasts.json`` lives
    directly under the date dir), ``run_id`` is the literal string ``"flat"``
    and ``run_dir`` is the date dir itself.

    Non-directory entries (``index.jsonl``, ``heal_report_*.csv``) and
    directory names that don't match ``YYYY-MM-DD`` are skipped silently.
    """
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"Snapshots dir not found: {snapshots_dir}")

    for day_dir in sorted(snapshots_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        if not _DATE_DIR_RE.match(day_dir.name):
            continue
        snapshot_date = _date.fromisoformat(day_dir.name)

        # Prefer nested run subdirs if present. Some cutover days
        # (e.g. 2026-03-26) have BOTH a flat weather_forecasts.json
        # AND a numbered run subdir from a later automated run; the
        # nested run is the more up-to-date forecast and must win.
        run_subdirs = [p for p in sorted(day_dir.iterdir()) if p.is_dir()]
        if run_subdirs:
            for run_dir in run_subdirs:
                yield snapshot_date, run_dir.name, run_dir
            continue

        # No nested runs — try flat layout.
        if (day_dir / "weather_forecasts.json").exists():
            yield snapshot_date, "flat", day_dir


def _run_id_sort_key(run_id: str) -> tuple[int, str]:
    """Sort key: 'flat' is a sentinel that always loses to numeric run_ids.

    Returns (0, "") for 'flat', (1, run_id) for everything else, so a max()
    over this key picks any numeric run_id over 'flat'.
    """
    if run_id == "flat":
        return (0, "")
    return (1, run_id)


def _select_latest_run_per_day(
    rows: list[tuple[str, str, dict]],
) -> Iterator[tuple[str, str, dict]]:
    """Within each (snapshot_date, city, forecast_date, variable, model),
    yield only the row from the highest-precedence run_id.

    Input: list of (snapshot_date_str, run_id, forecast_row) tuples — i.e.
    the flattened output of walking ``_iter_snapshot_runs`` x
    ``_load_forecasts_per_run``.

    Selection key per group is ``_run_id_sort_key(run_id)``; ties keep the
    last seen (deterministic via input ordering).
    """
    best: dict[tuple[str, str, str, str, str], tuple[str, str, dict]] = {}
    for snap, run_id, fc in rows:
        key = (
            snap,
            fc["city"],
            fc["forecast_date"],
            fc["daily_variable"],
            fc["model"],
        )
        prev = best.get(key)
        if prev is None or _run_id_sort_key(run_id) >= _run_id_sort_key(prev[1]):
            best[key] = (snap, run_id, fc)
    yield from best.values()


def _load_all_settlements(snapshots_dir: Path) -> dict[tuple[str, str], dict]:
    """Walk all snapshots, merge settlements via prefer_latest_settlement.

    Unlike ``scripts._calibration_common.load_snapshot_data``, we **do
    not** drop preliminaries here. The row builder tags them via the
    ``settlement_status`` column so the fitter can filter at training
    time without a corpus rebuild.

    Returns ``{(city, observation_date): record}``.
    """
    out: dict[tuple[str, str], dict] = {}
    for _snapshot_date, _run_id, run_dir in _iter_snapshot_runs(snapshots_dir):
        st_file = run_dir / "settlements.json"
        if not st_file.exists():
            continue
        try:
            raw = json.loads(st_file.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed settlement JSON %s: %s", st_file, exc)
            continue
        for s in raw:
            key = (s["city"], s["observation_date"])
            existing = out.get(key)
            if existing is None or prefer_latest_settlement(existing, s):
                out[key] = s
    return out


def _build_rows(
    forecasts: list[tuple[str, str, dict]],
    settlements: dict[tuple[str, str], dict],
) -> Iterator[dict]:
    """Join forecasts → settlements, derive output rows.

    Drops:
        * Negative-lead rows (forecast_date < snapshot_date)
        * CHI rows where forecast_date < 2026-03-20 (legacy coord-fix exclusion)
        * Rows with no matching settlement
        * Rows where the settlement field for the variable is null

    Per the spec, ``ens_mean`` / ``ens_std`` are recomputed from
    ``temperatures`` (ddof=1) — the JSON ``mean`` / ``std_dev`` fields
    are ignored.
    """
    for snap_str, run_id, fc in forecasts:
        snap_d = _date.fromisoformat(snap_str)
        fc_d = _date.fromisoformat(fc["forecast_date"])
        lead = (fc_d - snap_d).days

        city = fc["city"]
        if lead < 0:
            logger.warning(
                "Negative lead %d for %s %s in snapshot %s — skipping (corrupt snapshot?)",
                lead,
                city,
                fc["forecast_date"],
                snap_str,
            )
            continue
        if city == "CHI" and fc_d < _CHI_SNAPSHOT_FIX_DATE:
            continue

        settle = settlements.get((city, fc["forecast_date"]))
        if settle is None:
            continue

        daily_var = fc["daily_variable"]
        settle_field = _VARIABLE_TO_SETTLEMENT_FIELD.get(daily_var)
        if settle_field is None:
            continue
        obs = settle.get(settle_field)
        if obs is None:
            continue

        members = fc["temperatures"]
        ens_mean = sum(members) / len(members)
        ens_std = statistics.stdev(members)  # ddof=1; len(members) >= 25

        # Tag as final only if the field is explicitly False. Missing
        # or None (legacy snapshots without the field) defaults to
        # preliminary — `bool(None) is False` would otherwise wrongly
        # tag null values as final.
        is_prelim = settle.get("is_preliminary") is not False

        yield {
            "snapshot_date": snap_str,
            "run_id": run_id,
            "run_ts": float(fc.get("fetched_at", 0.0)),
            "forecast_date": fc["forecast_date"],
            "lead": lead,
            "city": city,
            "var": _VARIABLE_TO_VAR_SHORT[daily_var],
            "daily_variable": daily_var,
            "model": fc["model"],
            "model_version": fc.get("model_version") or "",
            "n_members": len(members),
            "ens_mean": ens_mean,
            "ens_std": ens_std,
            "obs": float(obs),
            "settlement_status": "preliminary" if is_prelim else "final",
            "doy": fc_d.timetuple().tm_yday,
        }


_OUTPUT_COLUMNS = [
    "snapshot_date",
    "run_id",
    "run_ts",
    "forecast_date",
    "lead",
    "city",
    "var",
    "daily_variable",
    "model",
    "model_version",
    "n_members",
    "ens_mean",
    "ens_std",
    "obs",
    "settlement_status",
    "doy",
]


def _collect_forecasts(
    snapshots_dir: Path,
    *,
    min_members: int,
    allowed_models: frozenset[str],
    start_date: _date | None,
    end_date: _date | None,
) -> list[tuple[str, str, dict]]:
    """Walk all runs, apply date window, return (snap_str, run_id, fc) tuples.

    Date-window filter is applied here on ``snapshot_date`` so we don't
    waste IO on out-of-window runs.
    """
    out: list[tuple[str, str, dict]] = []
    for snap_d, run_id, run_dir in _iter_snapshot_runs(snapshots_dir):
        if start_date and snap_d < start_date:
            continue
        if end_date and snap_d > end_date:
            continue
        rows = _load_forecasts_per_run(
            run_dir, min_members=min_members, allowed_models=allowed_models
        )
        for fc in rows:
            out.append((snap_d.isoformat(), run_id, fc))
    return out


def _summarize(df: pd.DataFrame) -> str:
    """Build the human summary string for stdout."""
    if df.empty:
        return "EMOS corpus built: 0 rows"

    lines = [f"EMOS corpus built: {len(df)} rows"]
    lines.append(
        "  models:   " + ", ".join(f"{m}={c}" for m, c in sorted(Counter(df["model"]).items()))
    )
    lines.append(
        "  vars:     " + ", ".join(f"{v}={c}" for v, c in sorted(Counter(df["var"]).items()))
    )
    lines.append(
        "  cities:   " + ", ".join(f"{c}={n}" for c, n in sorted(Counter(df["city"]).items()))
    )
    lines.append(
        "  leads:    " + ", ".join(f"{lead}={n}" for lead, n in sorted(Counter(df["lead"]).items()))
    )
    lines.append(
        "  status:   "
        + ", ".join(f"{s}={n}" for s, n in sorted(Counter(df["settlement_status"]).items()))
    )
    lines.append(f"  date span: {df['snapshot_date'].min()} → {df['snapshot_date'].max()}")
    return "\n".join(lines)


def main() -> None:
    # basicConfig lives here, not at module scope, so importing this module
    # (e.g. from a future fit_emos.py or a test) doesn't pre-configure the
    # caller's root logger as an import side effect.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build the EMOS training corpus")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Snapshot data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "emos" / "corpus.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--start-date", type=_date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=_date.fromisoformat, default=None)
    parser.add_argument(
        "--include-ecmwf",
        action="store_true",
        help="Also emit ECMWF rows (default: GFS only)",
    )
    parser.add_argument("--min-members", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.snapshots.is_dir():
        logger.error("Snapshots directory not found: %s", args.snapshots)
        sys.exit(1)

    allowed_models = frozenset({"gfs_seamless"})
    if args.include_ecmwf:
        allowed_models = frozenset({"gfs_seamless", "ecmwf_ifs025"})

    logger.info("Walking snapshots in %s", args.snapshots)
    raw_forecasts = _collect_forecasts(
        args.snapshots,
        min_members=args.min_members,
        allowed_models=allowed_models,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info("Loaded %d forecast rows pre-dedup", len(raw_forecasts))

    deduped = list(_select_latest_run_per_day(raw_forecasts))
    logger.info("After latest-run dedup: %d forecast rows", len(deduped))

    settlements = _load_all_settlements(args.snapshots)
    logger.info("Loaded %d merged settlement records", len(settlements))

    rows = list(_build_rows(deduped, settlements))
    logger.info("Built %d output rows", len(rows))

    if not rows:
        logger.error("No usable rows after filtering — corpus would be empty")
        sys.exit(1)

    if set(rows[0].keys()) != set(_OUTPUT_COLUMNS):
        extra = set(rows[0].keys()) - set(_OUTPUT_COLUMNS)
        missing = set(_OUTPUT_COLUMNS) - set(rows[0].keys())
        logger.error(
            "Schema drift: _OUTPUT_COLUMNS does not match _build_rows keys. extra=%s missing=%s",
            extra,
            missing,
        )
        sys.exit(1)

    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info("Wrote %s", args.output)

    print()
    print(_summarize(df))
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
