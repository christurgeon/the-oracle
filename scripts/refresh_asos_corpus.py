"""Incremental refresh of the ASOS hourly research corpus (all 16 cities).

Keeps `data/research/asos/k{city}_asos.csv` current by reading each file's last
`valid` timestamp and pulling only the new IEM rows up to *now* (append +
dedup by `(station, valid)`). This is the recurring "insurance" tool that keeps
the new-city intraday lock-in forward corpus fresh so the weekly centering / V1
/ spread-census runs have same-day `M`. See
`docs/research/2026-06-09-newcity-stage1a-findings.md`. The lock-in program was
closed 2026-09-30, so weekly runs are no longer needed.

Note: IEM is a multi-year archive, so a missed week is not lost — any gap can be
re-pulled later (or via the from-scratch new-city backfill, an unpublished
research script). This tool is for
convenience + interim checks, not data preservation.

Supersedes, for ONGOING refresh, the 5-city `research_2026_05_21_refresh_asos.py`
(unpublished) and the from-scratch new-city backfill. The CSVs are
gitignored data; this SCRIPT is the committed artifact.

Usage:
  uv run python scripts/refresh_asos_corpus.py                  # all 16, through now
  uv run python scripts/refresh_asos_corpus.py --only ATL,PHX   # subset
  uv run python scripts/refresh_asos_corpus.py --end 2026-07-22 # explicit exclusive end
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

REPO = Path(__file__).resolve().parent.parent
ASOS_DIR = REPO / "data" / "research" / "asos"

# city -> (IEM ICAO, short code as written in the CSV `station` column).
# Short code == CITY_TO_CLI_LOCATION[city] (the NWS settlement station).
# CSV file stem is uniformly "k" + city.lower().
CITY_STATIONS: dict[str, tuple[str, str]] = {
    # mature (5)
    "NYC": ("KNYC", "NYC"), "CHI": ("KMDW", "MDW"), "LAX": ("KLAX", "LAX"),
    "MIA": ("KMIA", "MIA"), "AUS": ("KAUS", "AUS"),
    # new (11)
    "ATL": ("KATL", "ATL"), "DAL": ("KDFW", "DFW"), "DC": ("KDCA", "DCA"),
    "HOU": ("KHOU", "HOU"), "LV": ("KLAS", "LAS"), "MIN": ("KMSP", "MSP"),
    "NOLA": ("KMSY", "MSY"), "OKC": ("KOKC", "OKC"), "PHX": ("KPHX", "PHX"),
    "SATX": ("KSAT", "SAT"), "SFO": ("KSFO", "SFO"),
}

log = logging.getLogger("refresh_asos_corpus")


def last_timestamp(csv_path: Path) -> datetime | None:
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    last = datetime.strptime(row[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                except ValueError:
                    continue
    return last


def fetch_iem(icao: str, start: datetime, end_exclusive: date) -> str:
    params = {
        "station": icao, "data": "tmpf",
        "year1": start.year, "month1": start.month, "day1": start.day, "hour1": start.hour,
        "year2": end_exclusive.year, "month2": end_exclusive.month,
        "day2": end_exclusive.day, "hour2": 0,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
        "missing": "M", "trace": "T", "direct": "no", "report_type": [3, 4],
    }
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    url = base + urlencode(params, doseq=True)
    log.info("  GET %s", url)
    with urlopen(url, timeout=60) as resp:  # fixed IEM host
        return resp.read().decode("utf-8")


def parse_iem_rows(text: str, icao: str, rename_to: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    reader = csv.reader(StringIO(text))
    header = next(reader, None)
    if header is None:
        return out
    h = [c.strip().lower() for c in header]
    try:
        i_station, i_valid, i_tmpf = h.index("station"), h.index("valid"), h.index("tmpf")
    except ValueError:
        log.warning("  unexpected header: %s", header)
        return out
    accept = {icao.upper(), rename_to.upper()}
    for row in reader:
        if len(row) <= max(i_station, i_valid, i_tmpf):
            continue
        if row[i_station].strip().upper() not in accept:
            continue
        tmpf = row[i_tmpf].strip()
        if tmpf in ("", "M", "T"):
            continue
        valid = row[i_valid].strip()[:16]
        try:
            datetime.strptime(valid, "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        out.append((rename_to, valid, tmpf))
    return out


def existing_keys(csv_path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not csv_path.exists():
        return keys
    with csv_path.open() as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                keys.add((row[0].strip(), row[1].strip()))
    return keys


def append_csv(csv_path: Path, rows: list[tuple[str, str, str]], keys: set[tuple[str, str]]) -> int:
    written = 0
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        for station, valid, tmpf in rows:
            if (station, valid) in keys:
                continue
            w.writerow([station, valid, tmpf])
            keys.add((station, valid))
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Incremental ASOS corpus refresh (16 cities).")
    parser.add_argument(
        "--only", default=None, help="Comma-separated city subset (e.g. ATL,PHX)."
    )
    parser.add_argument(
        "--end", default=None,
        help="Exclusive end date YYYY-MM-DD (default: tomorrow UTC).",
    )
    parser.add_argument(
        "--pause", type=float, default=3.0, help="Seconds between stations (IEM politeness)."
    )
    args = parser.parse_args(argv)

    end_excl = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end else (datetime.now(UTC).date() + timedelta(days=1))
    )
    cities = list(CITY_STATIONS)
    if args.only:
        want = {c.strip().upper() for c in args.only.split(",")}
        cities = [c for c in cities if c in want]
        if not cities:
            log.error("--only matched no known cities: %s", sorted(want))
            return 1

    total = 0
    missing: list[str] = []
    for i, city in enumerate(cities):
        icao, code = CITY_STATIONS[city]
        csv_path = ASOS_DIR / f"k{city.lower()}_asos.csv"
        last = last_timestamp(csv_path)
        if last is None:
            log.warning("%s: no CSV (%s) — run backfill first; skipping", city, csv_path.name)
            missing.append(city)
            continue
        fetch_start = last + timedelta(hours=1)
        if fetch_start.date() >= end_excl:
            log.info("%s already current (last %s)", city, last)
            continue
        log.info("%s: fetching %s from %s to %s (excl)", city, icao,
                 fetch_start.isoformat(), end_excl)
        try:
            text = fetch_iem(icao, fetch_start, end_excl)
        except Exception:
            log.exception("%s: IEM fetch failed — skipping (re-pull later)", city)
            continue
        rows = parse_iem_rows(text, icao, code)
        added = append_csv(csv_path, rows, existing_keys(csv_path))
        log.info("  appended %d new rows (deduped from %d)", added, len(rows))
        total += added
        if args.pause > 0 and i < len(cities) - 1:
            time.sleep(args.pause)

    log.info("DONE — appended %d rows across %d cities (end<%s)",
             total, len(cities) - len(missing), end_excl)
    if missing:
        log.warning("missing CSVs (run backfill): %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
