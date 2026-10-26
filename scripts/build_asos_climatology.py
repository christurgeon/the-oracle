"""Build ASOS climatology lookup table — production artifact.

Promoted from an unpublished research script
(``research_2026_05_21_build_asos_climatology.py``, spec v2.1).

Key:   (city, var, month_local_1_12, hour_utc_0_23)
Value: (mu_gain, sigma_gain, n)

Implements §4 fit gates:
  - per-cell: n >= MIN_N_PER_CELL (30)
  - pooled fallback: n_pool >= MIN_N_POOLED (60)
  - sanity: mu_gain >= 0 for HIGH (remaining gain must be non-negative);
            sigma_gain >= 0.5 °F

Fit window: 2024-05-01 UTC → 2026-04-28 UTC (exclusive)
  Eval / live period (2026-04-28 → present) is held out of the fit so the
  artifact is not trained on the evaluation window.  If the source CSVs are
  later extended past 2026-04-28, the fit window is NOT extended automatically
  — edit FIT_END here and re-run to produce a new artifact.

Station-identity guard (HARD ABORT):
  For each city, every station code found in the source CSV ``station`` column
  must equal the expected NWS settlement station from
  ``python.data.nws.CITY_TO_CLI_LOCATION[city]``.  A mismatch aborts the build
  with a non-zero exit code and a clear error message.

  This guard is load-bearing for the CHI/MDW case: Kalshi settles CHI weather
  markets on Chicago Midway (MDW), not O'Hare (ORD).  If the source CSV were
  accidentally built from ORD readings the artifact would be silently wrong.

Output: data/asos/climatology.json

Usage:
    uv run python scripts/build_asos_climatology.py
    uv run python scripts/build_asos_climatology.py --out data/asos/climatology.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to sys.path so the script works standalone (same pattern
# as scripts/build_emos_corpus.py and scripts/fit_isotonic.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import the settlement station map from the production module so the guard
# stays in sync with the live client automatically.
# ---------------------------------------------------------------------------
from python.data.nws import CITY_TO_CLI_LOCATION  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASOS_DIR = REPO / "data" / "research" / "asos"
DEFAULT_OUTPUT = REPO / "data" / "asos" / "climatology.json"

# City LOCAL-STANDARD-TIME (LST) offsets used to assign each observation to its
# NWS CLI climate day. NWS climate days run midnight-to-midnight LOCAL STANDARD
# TIME (no DST) — see spec §0a / memo finding #1. A civil ``ZoneInfo`` (DST-aware)
# mis-assigns the 00:00–01:00 LDT boundary hour to the wrong climate day during
# DST, contaminating the fitted gains. We therefore group by FIXED standard-time
# offsets year-round.
#
# MUST match ``python/data/asos.py`` CITY_STD_OFFSET_HOURS exactly so the fit
# frame is bit-identical to the runtime M-frame (train/serve parity). A parity
# test asserts this in tests/.
CITY_STD_OFFSET_HOURS: dict[str, int] = {
    "NYC": -5,  # EST
    "CHI": -6,  # CST
    "LAX": -8,  # PST
    "MIA": -5,  # EST
    "AUS": -6,  # CST
}

# Fixed-offset tzinfo objects derived from CITY_STD_OFFSET_HOURS (never DST).
CITY_TZ: dict[str, timezone] = {
    city: timezone(timedelta(hours=off)) for city, off in CITY_STD_OFFSET_HOURS.items()
}

# Fit window: inclusive start, exclusive end.
# Eval window 2026-04-28 → present is held out.
FIT_START = datetime(2024, 5, 1, tzinfo=UTC)
FIT_END = datetime(2026, 4, 28, tzinfo=UTC)  # exclusive

MIN_N_PER_CELL = 30
MIN_N_POOLED = 60
MIN_SIGMA_F = 0.5

VARS = ("high", "low")

log = logging.getLogger("build_asos_climatology")


# ---------------------------------------------------------------------------
# Station-identity guard
# ---------------------------------------------------------------------------


# Map from internal city code to the expected 3-letter station code that
# appears in the source ASOS CSV ``station`` column.  This matches the NWS
# settlement station: CITY_TO_CLI_LOCATION[city].
#
# For cities where the ASOS research CSV uses a 3-letter code that differs
# from IEM's ICAO 4-letter representation, we derive it from CITY_TO_CLI_LOCATION
# (which already carries the correct bare-3-letter settlement codes).
def _expected_csv_station(city: str) -> str:
    """Return the expected ``station`` column value for *city*'s source CSV.

    Derived from ``CITY_TO_CLI_LOCATION`` so it stays in sync with the
    production settlement source automatically.

    Raises:
        KeyError: If *city* is not in ``CITY_TO_CLI_LOCATION`` (should not
        happen for the five cities in ``CITY_TZ``).
    """
    return CITY_TO_CLI_LOCATION[city]


def check_csv_stations(city: str, csv_path: Path) -> None:
    """HARD ABORT guard: verify the CSV ``station`` column only contains the
    expected settlement station for *city*.

    Scans all non-header rows and collects every distinct station code seen.
    If any code differs from ``CITY_TO_CLI_LOCATION[city]``, aborts immediately
    with a descriptive error message and ``sys.exit(1)``.

    The CHI/MDW case is the canonical example: if the CSV were accidentally
    built from ORD (O'Hare) rather than MDW (Midway), the artifact would be
    silently wrong because Kalshi settles CHI markets on Midway.

    Args:
        city: Internal city code (e.g. ``"CHI"``).
        csv_path: Path to the source ASOS CSV.
    """
    expected = _expected_csv_station(city)
    seen: set[str] = set()
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            station_val = row[0].strip()
            if station_val:
                seen.add(station_val)

    # Filter out obvious non-data rows (e.g., blank station values already skipped)
    unexpected = seen - {expected}
    if unexpected:
        log.error(
            "STATION MISMATCH for city=%s: expected station=%r but CSV %s "
            "contains station(s) %r. Aborting build — verify source data.",
            city,
            expected,
            csv_path.name,
            sorted(unexpected),
        )
        sys.exit(
            f"ERROR: Station mismatch for {city}: expected {expected!r}, "
            f"got {sorted(unexpected)} in {csv_path}. "
            f"Kalshi settles {city} on station {expected}. "
            f"Re-download the ASOS CSV from the correct station before re-running."
        )
    if not seen:
        log.error(
            "No station codes found in %s for city=%s. Aborting.",
            csv_path.name,
            city,
        )
        sys.exit(f"ERROR: No station codes found in {csv_path} for city={city}.")

    log.info("  station check OK for %s: station=%r (matches %s)", city, expected, expected)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_asos_rows(city: str, csv_path: Path) -> list[tuple[datetime, float]]:
    """Return ``[(utc_dt, tmpf)]`` for *city*, parsed from *csv_path*.

    Skips rows where ``tmpf`` is ``"M"`` (missing), ``"T"`` (trace), or
    blank.  Invalid date formats or non-numeric temperatures are also skipped
    silently.
    """
    rows: list[tuple[datetime, float]] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header: station, valid, tmpf
        for row in reader:
            if len(row) < 3:
                continue
            _station, valid, tmpf_s = row[0], row[1], row[2]
            if tmpf_s in ("M", "", "T"):
                continue
            try:
                tmpf = float(tmpf_s)
            except ValueError:
                continue
            try:
                dt = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except ValueError:
                continue
            rows.append((dt, tmpf))
    return rows


def _local_date_for_utc(dt_utc: datetime, tz: timezone) -> date:
    """Return the LST climate date for *dt_utc* under the fixed-offset *tz*."""
    return dt_utc.astimezone(tz).date()


def compute_gains(
    rows: list[tuple[datetime, float]],
    tz: timezone,
    var: str,
) -> list[tuple[int, int, float]]:
    """Return ``[(month, hour_utc, gain)]`` for each observation in the fit window.

    For each LST climate day in the fit window, ``gain = day_extremum - extremum_so_far``.

    - ``var='high'``: extremum is ``max``; gain >= 0 (remaining gain to daily high).
    - ``var='low'``: extremum is ``min``; gain <= 0 (remaining drop to daily low,
      stored as signed value — same parameterization as the spec).

    Local days with fewer than 6 observations are skipped (insufficient data to
    bound the day's extremum).

    Fit window is applied here (``FIT_START <= dt < FIT_END``).
    """
    filtered = [(dt, t) for dt, t in rows if FIT_START <= dt < FIT_END]

    by_local: dict[date, list[tuple[datetime, float]]] = defaultdict(list)
    for dt, t in filtered:
        ld = _local_date_for_utc(dt, tz)
        by_local[ld].append((dt, t))

    extremum_fn = max if var == "high" else min

    results: list[tuple[int, int, float]] = []
    for ld, obs in by_local.items():
        if len(obs) < 6:
            continue
        obs.sort(key=lambda x: x[0])
        day_extremum = extremum_fn(t for _, t in obs)

        by_utc_hour: dict[int, float] = {}
        running = obs[0][1]
        for dt, t in obs:
            running = extremum_fn(running, t)
            by_utc_hour[dt.hour] = running

        for h, so_far in by_utc_hour.items():
            gain = day_extremum - so_far
            results.append((ld.month, h, gain))
    return results


def aggregate_to_cells(
    gains: list[tuple[int, int, float]],
) -> dict[tuple[int, int], dict]:
    """Aggregate gain samples into per-``(month, hour)`` summary statistics."""
    by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for m, h, g in gains:
        by_cell[(m, h)].append(g)

    out: dict[tuple[int, int], dict] = {}
    for (m, h), gs in by_cell.items():
        arr = np.array(gs, dtype=float)
        out[(m, h)] = {
            "mu": float(arr.mean()),
            "sigma": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return out


def build_hour_pool(gains: list[tuple[int, int, float]]) -> dict[int, dict]:
    """Pool gain samples across all months for each UTC hour.

    Used as the fallback when ``(month, hour)`` cell has insufficient data.
    """
    by_hour: dict[int, list[float]] = defaultdict(list)
    for _m, h, g in gains:
        by_hour[h].append(g)
    out: dict[int, dict] = {}
    for h, gs in by_hour.items():
        arr = np.array(gs, dtype=float)
        out[h] = {
            "mu": float(arr.mean()),
            "sigma": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return out


def apply_sanity_gates(mu: float, sigma: float, var: str) -> tuple[bool, str]:
    """Return ``(passes, reason_if_not)`` for the §4 sanity gates.

    Gates:
    - ``var='high'`` requires ``mu >= 0`` (remaining gain to high must be >= 0).
    - ``var='low'`` requires ``mu <= 0`` (remaining drop to low must be <= 0).
    - ``sigma >= MIN_SIGMA_F`` (0.5 °F) for both vars.
    """
    if var == "high" and mu < 0:
        return False, "mu_negative_for_high"
    if var == "low" and mu > 0:
        return False, "mu_positive_for_low"
    if sigma < MIN_SIGMA_F:
        return False, f"sigma_below_min ({sigma:.3f} < {MIN_SIGMA_F})"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_climatology() -> dict:
    """Build the full climatology artifact dict."""
    table: dict = {
        "spec_version": "v2.1",
        "fit_window_utc": {
            "start": FIT_START.isoformat(),
            "end_exclusive": FIT_END.isoformat(),
        },
        "params": {
            "min_n_per_cell": MIN_N_PER_CELL,
            "min_n_pooled": MIN_N_POOLED,
            "min_sigma_f": MIN_SIGMA_F,
        },
        "cities": {},
        "diagnostics": {
            "per_city_fit_days": {},
            "per_city_dropped_cells": {},
        },
    }

    total_primary = total_pool = total_unavailable = 0
    total_cells = 0

    for city in sorted(CITY_TZ.keys()):
        tz = CITY_TZ[city]
        csv_path = ASOS_DIR / f"k{city.lower()}_asos.csv"

        # ----------------------------------------------------------------
        # STATION-IDENTITY GUARD (HARD ABORT on mismatch)
        # ----------------------------------------------------------------
        check_csv_stations(city, csv_path)

        log.info("loading ASOS for %s (tz=%s)", city, tz)
        rows = load_asos_rows(city, csv_path)
        log.info("  %d raw rows", len(rows))

        city_block: dict = {}
        city_dropped: dict[str, list] = {}

        for var in VARS:
            gains = compute_gains(rows, tz, var)
            primary = aggregate_to_cells(gains)
            hour_pool = build_hour_pool(gains)

            cells: dict[str, dict] = {}
            dropped: list[tuple[int, int, str]] = []
            used_primary = used_pool = unavailable = 0

            for month in range(1, 13):
                for hour in range(0, 24):
                    key = f"{month:02d}-{hour:02d}"
                    p = primary.get((month, hour))
                    if p and p["n"] >= MIN_N_PER_CELL:
                        ok, why = apply_sanity_gates(p["mu"], p["sigma"], var)
                        if ok:
                            cells[key] = {**p, "source": "primary"}
                            used_primary += 1
                            continue
                        dropped.append((month, hour, f"sanity:{why}"))
                    pool = hour_pool.get(hour)
                    if pool and pool["n"] >= MIN_N_POOLED:
                        ok, why = apply_sanity_gates(pool["mu"], pool["sigma"], var)
                        if ok:
                            cells[key] = {**pool, "source": "hour_pool"}
                            used_pool += 1
                            continue
                        dropped.append((month, hour, f"pool_sanity:{why}"))
                    cells[key] = {"source": "unavailable"}
                    unavailable += 1

            city_block[var] = {
                "cells": cells,
                "summary": {
                    "primary": used_primary,
                    "hour_pool": used_pool,
                    "unavailable": unavailable,
                },
            }
            city_dropped[var] = dropped

            n_cells = used_primary + used_pool + unavailable
            total_cells += n_cells
            total_primary += used_primary
            total_pool += used_pool
            total_unavailable += unavailable

            pool_rate = (
                used_pool / (used_primary + used_pool) * 100
                if (used_primary + used_pool) > 0
                else 0.0
            )
            log.info(
                "  %s/%s: primary=%d  pool=%d (%.0f%% of filled)  unavailable=%d  (of %d cells)",
                city,
                var,
                used_primary,
                used_pool,
                pool_rate,
                unavailable,
                n_cells,
            )

        table["cities"][city] = city_block
        table["diagnostics"]["per_city_dropped_cells"][city] = city_dropped

    # Print fit coverage summary
    filled = total_primary + total_pool
    overall_coverage = filled / total_cells * 100 if total_cells > 0 else 0.0
    overall_pool_rate = filled and total_pool / filled * 100 or 0.0
    print(
        f"\nFit coverage summary:\n"
        f"  Total cells:     {total_cells}\n"
        f"  Primary filled:  {total_primary} ({total_primary / total_cells * 100:.1f}%)\n"
        f"  Pool fallback:   {total_pool}    ({overall_pool_rate:.1f}% of filled)\n"
        f"  Unavailable:     {total_unavailable} ({total_unavailable / total_cells * 100:.1f}%)\n"
        f"  Overall coverage: {overall_coverage:.1f}%"
    )

    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ASOS climatology artifact.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    output: Path = args.out

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("building ASOS climatology -> %s", output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = build_climatology()
    output.write_text(json.dumps(table, indent=2, default=str))
    log.info("wrote %s (%.1f KB)", output, output.stat().st_size / 1024)

    # Sanity print: NYC/high, July (peak heat), hour 20 UTC (4 PM EDT)
    nyc_high_july_20 = table["cities"]["NYC"]["high"]["cells"].get("07-20")
    log.info("Sanity NYC/high July hour 20 UTC: %s", nyc_high_july_20)

    return 0


if __name__ == "__main__":
    sys.exit(main())
