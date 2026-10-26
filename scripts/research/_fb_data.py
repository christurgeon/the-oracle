"""Shared data layer for favorite_bias improvement research.

Loads the committed snapshot corpus into per-(ticker, snapshot-slot) candidate
rows, joined to FINAL settlement and the production YES-outcome helper. All
levers in scripts/research/fb_improvements.py build on these rows.

Hard rules baked in here (per the project's burned-history guardrails):
  * Outcome via python.analytics.outcomes.derive_outcome_yes (single source).
  * Fees via python.analytics.fees (taker order_fee), fills AT THE ASK.
  * Lead is computed from the snapshot's UTC calendar day vs forecast_date —
    NEVER from time-to-expiry features (the retracted look-ahead trap).
  * Only FINAL settlements survive load_snapshot_data (preliminaries dropped).

Run nothing here directly; imported by the experiment script.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.utils.ticker_dates import (  # noqa: E402
    parse_ticker_city_var,
    parse_ticker_date,
)
from scripts._calibration_common import load_snapshot_data  # noqa: E402

SNAP = Path("data/snapshots")

# Snapshot slots (UTC hour) the daily workflow collects.
SLOTS = (2, 8, 14, 20)

# The cities the live g(mid) corpus is frozen to (scripts/fit_favorite_bias_map.py).
CORPUS_CITIES = frozenset({"NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "BOS", "PHIL"})


@dataclass(frozen=True)
class Cand:
    """One trade candidate: a (ticker, snapshot-slot) observation joined to truth."""

    ticker: str
    city: str
    var: str  # "high" | "low"
    stype: str  # "greater" | "less" | "between"
    fdate: date
    lead: int  # forecast_date - snapshot_calendar_day, in days
    slot: int  # UTC hour bucket of the snapshot (2/8/14/20)
    snap_ts: float
    ask: float
    bid: float
    mid: float
    vol: int
    yes: int  # realized YES outcome (0/1), from FINAL settlement


def _slot_of(ts: float) -> int:
    """Map a snapshot_ts to its nearest collection slot (2/8/14/20 UTC)."""
    h = datetime.fromtimestamp(ts, tz=UTC).hour + datetime.fromtimestamp(ts, tz=UTC).minute / 60.0
    return min(SLOTS, key=lambda s: abs(h - s))


def load_candidates(
    *,
    leads: tuple[int, ...] = (0, 1, 2),
    restrict_cities: frozenset[str] | None = CORPUS_CITIES,
) -> list[Cand]:
    """Build one candidate per (ticker, snapshot slot, snapshot calendar day).

    For each weather market dict in the corpus we emit a row for every distinct
    snapshot slot whose lead is in ``leads``. Within a (ticker, calendar-day,
    slot) we keep the latest snapshot_ts (closest to the scan). Joined to FINAL
    settlement; rows with no settlement or undefined outcome are dropped.
    """
    _f, settle, markets = load_snapshot_data(SNAP)

    # Group by (ticker, snap_calendar_day, slot) -> keep latest ts.
    best: dict[tuple[str, str, int], dict] = {}
    for m in markets:
        t = m.get("ticker", "")
        if not (t.startswith("KXHIGH") or t.startswith("KXLOWT")):
            continue
        if m.get("strike_type") not in ("greater", "less", "between"):
            continue
        ts = m.get("snapshot_ts")
        fd = parse_ticker_date(t)
        if ts is None or fd is None:
            continue
        snap_day = datetime.fromtimestamp(ts, tz=UTC).date()
        lead = (fd - snap_day).days
        if lead not in leads:
            continue
        slot = _slot_of(ts)
        key = (t, snap_day.isoformat(), slot)
        if key not in best or ts > best[key]["snapshot_ts"]:
            best[key] = m

    out: list[Cand] = []
    for (t, _snap_day, slot), m in best.items():
        fd = parse_ticker_date(t)
        cv = parse_ticker_city_var(t)
        if fd is None or cv is None:
            continue
        city, var = cv
        if restrict_cities is not None and city not in restrict_cities:
            continue
        ts = m["snapshot_ts"]
        snap_day = datetime.fromtimestamp(ts, tz=UTC).date()
        lead = (fd - snap_day).days
        st = settle.get((city, fd.isoformat()))
        if st is None:
            continue
        temp = st.get("high_f") if var == "high" else st.get("low_f")
        if temp is None:
            continue
        yo = derive_outcome_yes(
            m.get("strike_type"), m.get("floor_strike"), m.get("cap_strike"), temp
        )
        if yo is None:
            continue
        ya = m.get("yes_ask_dollars")
        yb = m.get("yes_bid_dollars")
        if ya is None or yb is None or not (0.0 < yb < ya < 1.0):
            continue
        out.append(
            Cand(
                ticker=t,
                city=city,
                var=var,
                stype=m.get("strike_type"),
                fdate=fd,
                lead=lead,
                slot=slot,
                snap_ts=float(ts),
                ask=float(ya),
                bid=float(yb),
                mid=(float(ya) + float(yb)) / 2.0,
                vol=int(float(m.get("volume_fp", 0) or 0)),
                yes=1 if yo else 0,
            )
        )
    out.sort(key=lambda c: (c.fdate, c.ticker, c.slot))
    return out


def latest_slot_per_ticker(cands: list[Cand], lead: int = 1) -> list[Cand]:
    """Pick, per ticker, the latest-slot lead-``lead`` candidate (the live rule:
    favorite_bias scans repeatedly through the day; the last scan that sees the
    market is the operative one). Mirrors the production lead-1 'eve-before'."""
    best: dict[str, Cand] = {}
    for c in cands:
        if c.lead != lead:
            continue
        if c.ticker not in best or c.snap_ts > best[c.ticker].snap_ts:
            best[c.ticker] = c
    return sorted(best.values(), key=lambda c: (c.fdate, c.ticker))
