"""favorite_bias out-of-sample test (G0) and maker-entry study on historical Kalshi data.

Pre-registration: docs/research/2026-10-01-favorite-bias-oos-g0-prereg.md (rules are frozen there).

Offline research. Unauthenticated GETs only, on /historical/*, /markets*, /series/* and /events*.
Every raw response is cached as gzip JSON under data/research/g0/raw/ keyed by a hash of the URL,
so a rerun makes zero network calls.

    fetch --stage catalog|hourly|minute [--pilot]   pull one stage (later stages read earlier ones)
    analyze [--pilot] [--audit PATH]                replay + Q1 / capacity / Q2 / live-replay check

Methodology rules carried over from scripts/research/_fb_data.py: lead from the UTC calendar day
(never time to expiry), only FINAL settlements (Kalshi ``result`` on finalized markets), g(mid) via
IsotonicCalibrator, fees via python.analytics.fees.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.analytics.edge_audit import day_clustered_bootstrap_ci  # noqa: E402
from python.analytics.fees import fee_per_contract  # noqa: E402
from python.data.calibration import IsotonicCalibrator  # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"
G0_DIR = Path("data/research/g0")
RAW_DIR = G0_DIR / "raw"
REQUEST_LOG = G0_DIR / "requests.jsonl"
GMAP_PATH = Path("data/calibration/favorite_bias_isotonic.json")
_ALLOWED_PATHS = ("/historical/", "/markets", "/series/", "/events")

SERIES_CITY_VAR: dict[str, tuple[str, str]] = {
    "KXHIGHNY": ("NYC", "high"),
    "KXHIGHCHI": ("CHI", "high"),
    "KXHIGHMIA": ("MIA", "high"),
    "KXHIGHLAX": ("LAX", "high"),
    "KXHIGHDEN": ("DEN", "high"),
    "KXHIGHAUS": ("AUS", "high"),
    "KXHIGHTBOS": ("BOS", "high"),
    "KXHIGHPHIL": ("PHIL", "high"),
    "KXLOWTNYC": ("NYC", "low"),
    "KXLOWTCHI": ("CHI", "low"),
    "KXLOWTMIA": ("MIA", "low"),
    "KXLOWTLAX": ("LAX", "low"),
    "KXLOWTDEN": ("DEN", "low"),
    "KXLOWTAUS": ("AUS", "low"),
    "KXLOWTBOS": ("BOS", "low"),
    "KXLOWTPHIL": ("PHIL", "low"),
}

OOS_END = date(2026, 3, 18)  # g(mid) corpus starts here; OOS is strictly before
IS_END = date(2026, 7, 31)
LIVE_CHECK = (date(2026, 8, 14), date(2026, 9, 25))
PILOT_MONTHS = ((2022, 12), (2023, 6), (2024, 6), (2025, 6), (2026, 6), (2026, 9))
Q2_HORIZONS_MIN: tuple[int | None, ...] = (10, 60, 240, None)  # None = 23:55 UTC on lead-1
BOOT_B = 4000

_MONTHS = {
    m: i
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
    )
}


# --------------------------------------------------------------------------- HTTP


class RateLimitAbortError(RuntimeError):
    pass


class G0Client:
    """Single-threaded cached GET client. Spacing ≥ 1 s on the VPS, ≥ 0.5 s elsewhere; 429 backoff
    15→120 s; aborts when a request is still rate-limited after a full 120 s backoff."""

    def __init__(
        self, *, min_interval: float | None = None, raw_dir: Path = RAW_DIR, offline: bool = False
    ) -> None:
        floor = 1.0 if Path("/opt/oracle").is_dir() else 0.5
        self.min_interval = max(min_interval or floor, floor)
        self.raw_dir = raw_dir
        self.offline = offline
        self._last = 0.0
        self._429s: deque[float] = deque()
        self._http: httpx.Client | None = None
        self.n_network = 0
        self.n_cached = 0
        self.n_miss_offline = 0
        self.bytes_network = 0

    @staticmethod
    def url(path: str, params: dict[str, Any] | None = None) -> str:
        q = {k: v for k, v in (params or {}).items() if v is not None}
        return BASE + path + (("?" + urlencode(sorted(q.items()))) if q else "")

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.raw_dir / h[:2] / f"{h}.json.gz"

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parsed JSON. A 4xx returns ``{"_status": code}``; offline miss ``{"_miss": 1}``."""
        if not path.startswith(_ALLOWED_PATHS):
            raise ValueError(f"path not allowed: {path}")
        url = self.url(path, params)
        cp = self._cache_path(url)
        if cp.exists():
            self.n_cached += 1
            with gzip.open(cp, "rt") as f:
                return json.load(f)["body"]
        if self.offline:
            self.n_miss_offline += 1
            return {"_miss": 1}
        body = self._fetch(url)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump({"url": url, "fetched_at": time.time(), "body": body}, f)
        tmp.replace(cp)
        return body

    def _fetch(self, url: str) -> dict[str, Any]:
        if self._http is None:
            self._http = httpx.Client(timeout=30.0, headers={"Accept": "application/json"})
        backoff = 15.0
        waited_max = False
        while True:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            t0 = time.monotonic()
            try:
                r = self._http.get(url)
            except httpx.HTTPError as e:
                self._log(url, -1, 0, time.monotonic() - t0, str(e))
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            self.n_network += 1
            self.bytes_network += len(r.content)
            self._log(url, r.status_code, len(r.content), time.monotonic() - t0)
            if r.status_code == 429:
                now = time.time()
                self._429s.append(now)
                while self._429s and now - self._429s[0] > 600:
                    self._429s.popleft()
                print(
                    f"429 #{len(self._429s)} in 10 min; backing off {backoff:.0f}s: {url}",
                    file=sys.stderr,
                    flush=True,
                )
                if waited_max:
                    raise RateLimitAbortError(f"still 429 after a 120 s backoff: {url}")
                waited_max = backoff >= 120.0
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            if r.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            if r.status_code >= 400:
                return {"_status": r.status_code, "_text": r.text[:500]}
            return r.json()

    @staticmethod
    def _log(url: str, status: int, nbytes: int, secs: float, err: str = "") -> None:
        REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with REQUEST_LOG.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "url": url,
                        "status": status,
                        "bytes": nbytes,
                        "secs": round(secs, 3),
                        "err": err,
                    }
                )
                + "\n"
            )


# --------------------------------------------------------------------------- data model


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_event_date(event_ticker: str) -> date | None:
    """forecast_date from ``{SERIES}-{YY}{MON}{DD}`` (old ``HIGHNY-`` and ``KX`` formats alike)."""
    tail = event_ticker.rsplit("-", 1)[-1].upper()
    if len(tail) != 7 or not (tail[:2].isdigit() and tail[5:].isdigit()):
        return None
    mon = _MONTHS.get(tail[2:5])
    if mon is None:
        return None
    try:
        return date(2000 + int(tail[:2]), mon, int(tail[5:]))
    except ValueError:
        return None


@dataclass(frozen=True)
class Market:
    ticker: str
    event_ticker: str
    series: str
    strike_type: str | None
    open_time: datetime
    close_time: datetime
    result: str
    tier: str  # "hist" | "live"


@dataclass(frozen=True)
class Event:
    event_ticker: str
    series: str
    city: str
    var: str
    forecast_date: date
    tails: tuple[Market, ...]  # greater + less only

    @property
    def lead1_start(self) -> datetime:
        return datetime.combine(self.forecast_date - timedelta(days=1), datetime.min.time(), UTC)

    @property
    def lead1_end(self) -> datetime:
        return self.lead1_start + timedelta(days=1)


def is_ladder(strike_types: Iterable[str | None]) -> bool:
    st = list(strike_types)
    return st.count("greater") == 1 and st.count("less") == 1 and st.count("between") >= 2


def _norm_market(m: dict[str, Any], series: str, cutoff: datetime) -> Market:
    ct = _ts(m["close_time"])
    return Market(
        ticker=m["ticker"],
        event_ticker=m["event_ticker"],
        series=series,
        strike_type=m.get("strike_type"),
        open_time=_ts(m["open_time"]),
        close_time=ct,
        result=m.get("result") or "",
        tier="hist" if ct < cutoff else "live",
    )


def build_events(markets: Iterable[Market]) -> tuple[list[Event], dict[str, int]]:
    """Group into ladder events (greater/less legs kept). Returns events and drop counts."""
    by_ev: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        by_ev[m.event_ticker].append(m)
    out: list[Event] = []
    drops: dict[str, int] = defaultdict(int)
    for ev, ms in by_ev.items():
        fd = parse_event_date(ev)
        if fd is None:
            drops["unparsed_date"] += 1
            continue
        if not is_ladder(m.strike_type for m in ms):
            drops["not_ladder"] += 1
            continue
        city, var = SERIES_CITY_VAR[ms[0].series]
        tails = tuple(
            sorted((m for m in ms if m.strike_type in ("greater", "less")), key=lambda m: m.ticker)
        )
        out.append(Event(ev, ms[0].series, city, var, fd, tails))
    out.sort(key=lambda e: (e.forecast_date, e.event_ticker))
    return out, dict(drops)


# --------------------------------------------------------------------------- bars


@dataclass(frozen=True)
class Bar:
    end_ts: int
    bid: float | None
    ask: float | None
    volume: float


def _close(c: dict[str, Any], key: str) -> float | None:
    v = c.get(key) or {}
    x = v.get("close_dollars", v.get("close"))
    return float(x) if x is not None else None


def parse_bars(candles: list[dict[str, Any]]) -> list[Bar]:
    """Normalize historical (``close``/``volume``) and live (``close_dollars``/``volume_fp``)."""
    out = []
    for c in candles:
        vol = c.get("volume_fp", c.get("volume")) or 0
        out.append(
            Bar(int(c["end_period_ts"]), _close(c, "yes_bid"), _close(c, "yes_ask"), float(vol))
        )
    out.sort(key=lambda b: b.end_ts)
    return out


@dataclass(frozen=True)
class Quote:
    t: int
    bid: float | None
    ask: float | None
    cum_volume: float


def grid_quotes(bars: list[Bar], times: list[int]) -> list[Quote]:
    """Forward-filled quote and cumulative volume at each grid time (bars ending at or before)."""
    ends = [b.end_ts for b in bars]
    cum = np.cumsum([b.volume for b in bars]) if bars else np.array([])
    out = []
    for t in times:
        i = bisect.bisect_right(ends, t) - 1
        if i < 0:
            out.append(Quote(t, None, None, 0.0))
        else:
            out.append(Quote(t, bars[i].bid, bars[i].ask, float(cum[i])))
    return out


def hourly_grid(ev: Event, open_time: datetime) -> list[int]:
    """Hourly bar-close times inside the lead-1 window, after the market opened."""
    start = max(ev.lead1_start, open_time)
    first = int(start.timestamp()) // 3600 * 3600 + 3600
    return list(range(first, int(ev.lead1_end.timestamp()) + 1, 3600))


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class RuleConfig:
    band_low: float = 0.50
    band_high: float = 0.75
    min_volume: float = 50.0
    edge_floor: float = 0.01
    gmap_path: Path = GMAP_PATH
    excluded: frozenset[str] = frozenset({"SEA_high"})
    # Optional per-scan-time min_volume override (Step 4: 200 before the 50-gate deploy).
    min_volume_fn: Callable[[int], float] | None = field(default=None, compare=False)

    def min_volume_at(self, t: int) -> float:
        return self.min_volume_fn(t) if self.min_volume_fn else self.min_volume


def taker_fee(price: float) -> float:
    return fee_per_contract(price, is_maker=False)


def qualify(
    q: Quote, g: Callable[[float], float], rule: RuleConfig
) -> tuple[float, float, float] | None:
    """(mid, g, edge) if the quote passes every live gate, else None."""
    if q.bid is None or q.ask is None or not (0.0 < q.bid < q.ask < 1.0):
        return None
    mid = (q.bid + q.ask) / 2.0
    if not (rule.band_low <= mid < rule.band_high):
        return None
    if q.cum_volume < rule.min_volume_at(q.t):
        return None
    gm = g(mid)
    edge = gm - q.ask - taker_fee(q.ask)
    if edge < rule.edge_floor:
        return None
    return mid, gm, edge


@dataclass(frozen=True)
class Entry:
    event: Event
    market: Market
    t: int
    bid: float
    ask: float
    mid: float
    g: float
    edge: float
    cum_volume: float

    @property
    def yes(self) -> int:
        return 1 if self.market.result == "yes" else 0

    @property
    def day(self) -> str:
        return self.event.forecast_date.isoformat()


def replay_event(
    ev: Event, quotes: dict[str, list[Quote]], g: Callable[[float], float], rule: RuleConfig
) -> Entry | None:
    """First qualifying grid time per event; ties within it go to the highest edge (then g, ticker).

    ``quotes`` maps market ticker -> quotes on that market's grid. Grid times are UTC and already
    restricted to the lead-1 window by the caller.
    """
    if f"{ev.city}_{ev.var}" in rule.excluded:
        return None
    lo, hi = int(ev.lead1_start.timestamp()), int(ev.lead1_end.timestamp())
    by_t: dict[int, list[Entry]] = defaultdict(list)
    for m in ev.tails:
        if m.strike_type not in ("greater", "less") or m.result not in ("yes", "no"):
            continue
        for q in quotes.get(m.ticker, []):
            if not (lo < q.t <= hi):
                continue
            r = qualify(q, g, rule)
            if r is None:
                continue
            assert q.bid is not None and q.ask is not None
            by_t[q.t].append(Entry(ev, m, q.t, q.bid, q.ask, r[0], r[1], r[2], q.cum_volume))
    if not by_t:
        return None
    cands = by_t[min(by_t)]
    return sorted(cands, key=lambda e: (-e.edge, -e.g, e.market.ticker))[0]


# --------------------------------------------------------------------------- catalog / fetch


def read_cutoff(c: G0Client) -> datetime:
    r = c.get("/historical/cutoff")
    return _ts(r["market_settled_ts"])


def _paginate(c: G0Client, path: str, params: dict[str, Any], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur = None
    while True:
        r = c.get(path, {**params, "cursor": cur})
        rows = r.get(key) or []
        out.extend(rows)
        cur = r.get("cursor")
        if not cur or not rows:
            return out


def load_catalog(c: G0Client, cutoff: datetime) -> list[Market]:
    seen: dict[str, Market] = {}
    for s in SERIES_CITY_VAR:
        hist = _paginate(c, "/historical/markets", {"series_ticker": s, "limit": 1000}, "markets")
        live = _paginate(
            c, "/markets", {"series_ticker": s, "status": "settled", "limit": 1000}, "markets"
        )
        for raw, tier in ((hist, "hist"), (live, "live")):
            for m in raw:
                nm = _norm_market(m, s, cutoff)
                if nm.tier != tier:
                    continue  # route strictly by close_time vs cutoff
                seen[nm.ticker] = nm
    return list(seen.values())


def in_scope(ev: Event, pilot: bool) -> bool:
    fd = ev.forecast_date
    if pilot:
        return ev.series == "KXHIGHNY" and (fd.year, fd.month) in PILOT_MONTHS
    return fd <= IS_END or LIVE_CHECK[0] <= fd <= LIVE_CHECK[1]


def hourly_candles(c: G0Client, m: Market, ev: Event) -> list[Bar] | None:
    if m.open_time >= ev.lead1_end:
        return []
    p = {
        "start_ts": int(m.open_time.timestamp()),
        "end_ts": int(ev.lead1_end.timestamp()),
        "period_interval": 60,
    }
    if m.tier == "hist":
        r = c.get(f"/historical/markets/{m.ticker}/candlesticks", p)
    else:
        r = c.get(f"/series/{m.series}/markets/{m.ticker}/candlesticks", p)
    if "_miss" in r or "_status" in r:
        return None
    return parse_bars(r.get("candlesticks") or [])


def minute_candles(c: G0Client, m: Market, ev: Event) -> list[Bar] | None:
    p = {
        "start_ts": int(ev.lead1_start.timestamp()),
        "end_ts": int(ev.lead1_end.timestamp()),
        "period_interval": 1,
    }
    if m.tier == "hist":
        r = c.get(f"/historical/markets/{m.ticker}/candlesticks", p)
    else:
        r = c.get(f"/series/{m.series}/markets/{m.ticker}/candlesticks", p)
    if "_miss" in r or "_status" in r:
        return None
    return parse_bars(r.get("candlesticks") or [])


@dataclass(frozen=True)
class Trade:
    t: float
    taker_yes: bool
    yes_price: float
    count: float


def tape(c: G0Client, m: Market, ev: Event) -> list[Trade] | None:
    lo, hi = int(ev.lead1_start.timestamp()), int(ev.lead1_end.timestamp())
    path = "/historical/trades" if m.tier == "hist" else "/markets/trades"
    out: list[Trade] = []
    cur = None
    while True:
        r = c.get(
            path, {"ticker": m.ticker, "limit": 1000, "min_ts": lo, "max_ts": hi, "cursor": cur}
        )
        if "_miss" in r or "_status" in r:
            return None
        rows = r.get("trades") or []
        for t in rows:
            tt = _ts(t["created_time"]).timestamp()
            if not (lo <= tt < hi):
                continue
            out.append(
                Trade(
                    tt,
                    t.get("taker_outcome_side") == "yes",
                    float(t["yes_price_dollars"]),
                    float(t.get("count_fp") or 0),
                )
            )
        cur = r.get("cursor")
        if not cur or not rows:
            break
    out.sort(key=lambda x: x.t)
    return out


def load_universe(c: G0Client, pilot: bool) -> tuple[datetime, list[Event], dict[str, int]]:
    cutoff = read_cutoff(c)
    events, drops = build_events(load_catalog(c, cutoff))
    return cutoff, [e for e in events if in_scope(e, pilot)], drops


def hourly_quotes(c: G0Client, ev: Event) -> dict[str, list[Quote]] | None:
    out: dict[str, list[Quote]] = {}
    for m in ev.tails:
        bars = hourly_candles(c, m, ev)
        if bars is None:
            return None
        out[m.ticker] = grid_quotes(bars, hourly_grid(ev, m.open_time))
    return out


def replay_all(
    c: G0Client, events: list[Event], rule: RuleConfig, g: Callable[[float], float]
) -> tuple[list[Entry], int]:
    entries: list[Entry] = []
    missing = 0
    for ev in events:
        qs = hourly_quotes(c, ev)
        if qs is None:
            missing += 1
            continue
        e = replay_event(ev, qs, g, rule)
        if e is not None:
            entries.append(e)
    return entries, missing


def cmd_fetch(args: argparse.Namespace) -> None:
    c = G0Client(min_interval=args.min_interval)
    cutoff, events, drops = load_universe(c, args.pilot)
    print(f"cutoff {cutoff.isoformat()} events in scope {len(events)} drops {drops}", flush=True)
    t0 = time.time()
    if args.stage in ("hourly", "minute"):
        for i, ev in enumerate(events):
            for m in ev.tails:
                hourly_candles(c, m, ev)
            if i % 200 == 0:
                print(
                    f"hourly {i}/{len(events)} net={c.n_network} bytes={c.bytes_network} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    if args.stage == "minute":
        g = IsotonicCalibrator(GMAP_PATH).calibrate
        entries, _ = replay_all(c, events, RuleConfig(), g)
        # Step 4 entries (min_volume 200 earlier in the window) and band-0.55 entries are subsets
        # or same-ticker variants of these; fetch minute data for the union of entered tickers.
        extra, _ = replay_all(
            c,
            [e for e in events if e.forecast_date >= LIVE_CHECK[0]],
            RuleConfig(min_volume=200.0),
            g,
        )
        todo = {(e.market.ticker): e for e in entries + extra}
        print(f"minute stage: {len(todo)} entered markets", flush=True)
        for i, e in enumerate(todo.values()):
            minute_candles(c, e.market, e.event)
            tape(c, e.market, e.event)
            if i % 100 == 0:
                print(
                    f"minute {i}/{len(todo)} net={c.n_network} bytes={c.bytes_network} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    print(
        f"done: network={c.n_network} cached={c.n_cached} bytes={c.bytes_network} "
        f"secs={time.time() - t0:.0f}",
        flush=True,
    )


# --------------------------------------------------------------------------- stats


def fee_variant(price: float, variant: str) -> float:
    if variant == "conservative":
        return taker_fee(price)
    if variant == "realistic":
        return 0.07 * price * (1.0 - price)
    if variant == "prefee":
        return 0.0
    raise ValueError(variant)


def roi_ci(days: list[str], pnl: list[float], wag: list[float]) -> dict[str, Any]:
    w = sum(wag)
    roi = sum(pnl) / w if w else 0.0
    lo, hi = day_clustered_bootstrap_ci(days, pnl, wag, b=BOOT_B) if days else (0.0, 0.0)
    return {"roi": roi, "lo": lo, "hi": hi, "n": len(pnl), "days": len(set(days))}


def taker_stats(entries: list[Entry], variant: str) -> dict[str, Any]:
    days = [e.day for e in entries]
    pnl = [e.yes - e.ask - fee_variant(e.ask, variant) for e in entries]
    return roi_ci(days, pnl, [e.ask for e in entries])


def paired_gap_ci(
    days: list[str], x: list[float], mask: list[bool], seed: int = 42
) -> tuple[float, float, float]:
    """Day-clustered bootstrap of mean(x | mask) − mean(x) (adverse-selection gap)."""
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(days):
        by_day[d].append(i)
    keys = list(by_day)
    xa, ma = np.asarray(x, float), np.asarray(mask, bool)
    point = float(xa[ma].mean() - xa.mean()) if ma.any() else float("nan")
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(BOOT_B):
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in by_day[keys[k]]]
        sub_m = ma[idx]
        if sub_m.any():
            s.append(xa[idx][sub_m].mean() - xa[idx].mean())
    if not s:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def year_coverage(events: list[Event]) -> dict[int, int]:
    months: dict[int, set[int]] = defaultdict(set)
    for e in events:
        months[e.forecast_date.year].add(e.forecast_date.month)
    return {y: len(ms) for y, ms in months.items()}


def q1_verdict(
    n: int, n_days: int, pooled: dict[str, Any], full_years: list[tuple[int, float]]
) -> str:
    """Pre-registered Q1 table. ``full_years`` = [(year, conservative point ROI)] ascending."""
    if n < 100 or n_days < 60:
        return "INCONCLUSIVE"
    if pooled["roi"] <= 0:
        return "FAIL"
    if pooled["lo"] > 0:
        rois = [r for _y, r in full_years]
        falling = any(rois[i] > rois[i + 1] > rois[i + 2] for i in range(len(rois) - 2))
        if rois and (rois[-1] <= 0 or falling):
            return "DECAYING"
        if len(rois) >= 2 and rois[-1] > 0 and rois[-2] > 0:
            return "PASS"
        if len(rois) == 1 and rois[0] > 0:
            return "PASS"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------- Q2 / capacity


def quote_at(bars: list[Bar], t: float) -> tuple[float | None, float | None]:
    i = bisect.bisect_right([b.end_ts for b in bars], t) - 1
    return (bars[i].bid, bars[i].ask) if i >= 0 else (None, None)


def q2_end(ev: Event) -> int:
    return int((ev.lead1_end - timedelta(minutes=5)).timestamp())


def q2_signal(
    e: Entry,
    trades: list[Trade],
    minute: list[Bar],
    horizon: int | None,
    g: Callable[[float], float],
    maker_fee_on: bool,
) -> dict[str, Any]:
    """One signal under rest-at-bid for T, then hybrid cross (pessimistic fill definition)."""
    end = q2_end(e.event)
    t_stop = end if horizon is None else min(e.t + horizon * 60, end)
    window = [tr for tr in trades if e.t < tr.t <= t_stop and not tr.taker_yes]
    fill_opt = any(tr.yes_price <= e.bid for tr in window)
    fill_pes = any(tr.yes_price < e.bid for tr in window)
    maker_fee = fee_per_contract(e.bid, is_maker=True) if maker_fee_on else 0.0
    out: dict[str, Any] = {
        "fill_opt": fill_opt,
        "fill_pes": fill_pes,
        "yes": e.yes,
        "day": e.day,
        "year": e.event.forecast_date.year,
    }
    if fill_pes:
        out.update(h_pnl=e.yes - e.bid - maker_fee, h_wag=e.bid, h_traded=True)
        return out
    if t_stop <= e.t:
        bid_t, ask_t = e.bid, e.ask
    else:
        bid_t, ask_t = quote_at(minute, t_stop)
    if bid_t is None or ask_t is None or not (0.0 < bid_t < ask_t < 1.0):
        out.update(h_pnl=0.0, h_wag=0.0, h_traded=False)
        return out
    mid_t = (bid_t + ask_t) / 2.0
    if g(mid_t) - ask_t - taker_fee(ask_t) >= 0.01:
        out.update(h_pnl=e.yes - ask_t - taker_fee(ask_t), h_wag=ask_t, h_traded=True)
    else:
        out.update(h_pnl=0.0, h_wag=0.0, h_traded=False)
    return out


def capacity(e: Entry, trades: list[Trade], g: Callable[[float], float]) -> tuple[float, float]:
    """Contracts and dollars that takers bought YES at prices still clearing the edge floor."""
    gm = g(e.mid)
    end = e.event.lead1_end.timestamp()
    n = d = 0.0
    for tr in trades:
        if (
            tr.taker_yes
            and e.t <= tr.t < end
            and gm - tr.yes_price - taker_fee(tr.yes_price) >= 0.01
        ):
            n += tr.count
            d += tr.count * tr.yes_price
    return n, d


def minute_entry(
    e: Entry, minute: list[Bar], g: Callable[[float], float], rule: RuleConfig
) -> Entry | None:
    """First qualifying 1-minute grid point on the entered market (sensitivity only)."""
    ev = e.event
    lo = int(max(ev.lead1_start, e.market.open_time).timestamp()) // 60 * 60 + 60
    times = list(range(lo, int(ev.lead1_end.timestamp()) + 1, 60))
    for q in grid_quotes(minute, times):
        r = qualify(q, g, rule)
        if r is not None:
            assert q.bid is not None and q.ask is not None
            return Entry(ev, e.market, q.t, q.bid, q.ask, r[0], r[1], r[2], q.cum_volume)
    return None


def _pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


# --------------------------------------------------------------------------- Step 4


def load_audit(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("decision") == "buy":
                rows.append(r)
    return rows


def live_check(
    c: G0Client, events: list[Event], audit_path: Path, g: Callable[[float], float], map_hash: str
) -> dict[str, Any]:
    lo, hi = LIVE_CHECK
    rows = [
        r
        for r in load_audit(audit_path)
        if (fd := parse_event_date(r["market_ticker"].rsplit("-", 1)[0])) and lo <= fd <= hi
    ]
    bad_map = [r for r in rows if r.get("map_version") != map_hash]
    bad_mid = [r for r in rows if float(r["mid"]) < 0.50]
    bad_sea = [r for r in rows if r.get("city") == "SEA" and r.get("var") == "high"]
    assert not bad_map, f"map_version mismatch in {len(bad_map)} buy rows"
    assert not bad_mid, f"{len(bad_mid)} buy rows with mid < 0.50"
    assert not bad_sea, f"{len(bad_sea)} SEA_high buy rows"
    all_buys = load_audit(audit_path)
    low_vol = sorted(r["ts"] for r in all_buys if float(r.get("volume_fp") or 0) < 200)
    switch = _ts(low_vol[0]) if low_vol else None
    assert switch is not None, "no buy row with volume_fp < 200 — cannot locate the 50-gate deploy"
    pre_rows = [r for r in rows if _ts(r["ts"]) < switch and float(r["volume_fp"]) < 200]
    assert not pre_rows, "buy rows with volume_fp < 200 before the detected switch"
    sw = int(switch.timestamp())
    commit_ts = datetime(2026, 9, 6, 18, 17, tzinfo=UTC)
    rule = RuleConfig(min_volume_fn=lambda t: 200.0 if t < sw else 50.0)

    earliest: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        earliest.setdefault(r["market_ticker"], r)
    live_ev: dict[str, dict[str, Any]] = {}
    for r in sorted(earliest.values(), key=lambda r: (r["ts"], -float(r["edge"]))):
        live_ev.setdefault(r["market_ticker"].rsplit("-", 1)[0], r)

    win = [e for e in events if lo <= e.forecast_date <= hi]
    excl = {
        e.event_ticker for e in win if e.lead1_start <= commit_ts < e.lead1_end
    }  # lead-1 scans straddle 18:17–18:58Z
    rep: dict[str, Entry] = {}
    missing = 0
    for ev in win:
        qs = hourly_quotes(c, ev)
        if qs is None:
            missing += 1
            continue
        e = replay_event(ev, qs, g, rule)
        if e is not None:
            rep[ev.event_ticker] = e
    keys = {e.event_ticker for e in win} - excl
    both = [k for k in keys if k in rep and k in live_ev]
    live_only = sorted(k for k in keys if k in live_ev and k not in rep)
    rep_only = sorted(k for k in keys if k in rep and k not in live_ev)
    neither = [k for k in keys if k not in rep and k not in live_ev]
    same_ticker = sum(1 for k in both if rep[k].market.ticker == live_ev[k]["market_ticker"])
    dt_min = [(rep[k].t - _ts(live_ev[k]["ts"]).timestamp()) / 60 for k in both]
    mid_diff = [
        rep[k].mid - float(live_ev[k]["mid"])
        for k in both
        if rep[k].market.ticker == live_ev[k]["market_ticker"]
    ]
    agree = (len(both) + len(neither)) / len(keys) if keys else 0.0
    entered_any = len(both) + len(live_only) + len(rep_only)
    return {
        "switch_ts": switch.isoformat(),
        "n_events": len(keys),
        "excluded_events": len(excl),
        "missing_candles": missing,
        "both": len(both),
        "live_only": len(live_only),
        "replay_only": len(rep_only),
        "neither": len(neither),
        "agreement_all_events": agree,
        "agreement_entered_events": len(both) / entered_any if entered_any else 0.0,
        "same_ticker_rate": same_ticker / len(both) if both else 0.0,
        "entry_dt_min": {
            "median": _pct(dt_min, 50),
            "p10": _pct(dt_min, 10),
            "p90": _pct(dt_min, 90),
        },
        "mid_replay_minus_live": {
            "median": _pct(mid_diff, 50),
            "p10": _pct(mid_diff, 10),
            "p90": _pct(mid_diff, 90),
            "mean_abs": float(np.mean(np.abs(mid_diff))) if mid_diff else 0.0,
        },
        "live_only_events": live_only[:40],
        "replay_only_events": rep_only[:40],
        "live_only_detail": [
            {
                "ev": k,
                "ts": live_ev[k]["ts"],
                "mid": live_ev[k]["mid"],
                "vol": live_ev[k]["volume_fp"],
                "ticker": live_ev[k]["market_ticker"],
            }
            for k in live_only[:15]
        ],
        "replay_only_detail": [
            {
                "ev": k,
                "t": datetime.fromtimestamp(rep[k].t, UTC).isoformat(),
                "mid": rep[k].mid,
                "vol": rep[k].cum_volume,
                "ticker": rep[k].market.ticker,
            }
            for k in rep_only[:15]
        ],
    }


# --------------------------------------------------------------------------- analyze


def _entry_row(e: Entry) -> dict[str, Any]:
    return {
        "ticker": e.market.ticker,
        "t": datetime.fromtimestamp(e.t, UTC).isoformat(),
        "bid": e.bid,
        "ask": e.ask,
        "mid": round(e.mid, 4),
        "g": round(e.g, 4),
        "edge": round(e.edge, 4),
        "cum_vol": e.cum_volume,
        "result": e.market.result,
    }


def year_table(entries: list[Entry], coverage: dict[int, int]) -> list[dict[str, Any]]:
    out = []
    for y in sorted({e.event.forecast_date.year for e in entries}):
        es = [e for e in entries if e.event.forecast_date.year == y]
        out.append(
            {
                "year": y,
                "full": coverage.get(y, 0) >= 9,
                "months": coverage.get(y, 0),
                "n": len(es),
                "days": len({e.day for e in es}),
                "win": float(np.mean([e.yes for e in es])),
                "mean_g": float(np.mean([e.g for e in es])),
                "mean_ask": float(np.mean([e.ask for e in es])),
                **{v: taker_stats(es, v) for v in ("conservative", "realistic", "prefee")},
            }
        )
    return out


def cmd_analyze(args: argparse.Namespace) -> None:
    c = G0Client(offline=True)
    cal = IsotonicCalibrator(GMAP_PATH)
    assert cal.is_loaded, cal.fallback_reason
    g = cal.calibrate
    cutoff, events, drops = load_universe(c, args.pilot)
    oos_ev = [e for e in events if e.forecast_date < OOS_END]
    is_ev = [e for e in events if OOS_END <= e.forecast_date <= IS_END]
    rule = RuleConfig()
    out: dict[str, Any] = {
        "cutoff": cutoff.isoformat(),
        "drops": drops,
        "n_events": len(events),
        "map_hash": cal.artifact_hash,
    }
    scalar = sum(1 for e in events for m in e.tails if m.result not in ("yes", "no"))
    out["scalar_or_other_tails"] = scalar

    oos, miss_oos = replay_all(c, oos_ev, rule, g)
    ins, miss_is = replay_all(c, is_ev, rule, g)
    out["missing_hourly"] = {"oos": miss_oos, "is": miss_is}
    cov = year_coverage(oos_ev)
    out["oos_coverage_months"] = cov
    out["oos_years"] = year_table(oos, cov)
    out["is_years"] = year_table(ins, year_coverage(is_ev))
    pooled = {v: taker_stats(oos, v) for v in ("conservative", "realistic", "prefee")}
    out["oos_pooled"] = pooled
    out["is_pooled"] = {v: taker_stats(ins, v) for v in ("conservative", "realistic", "prefee")}
    full = [(r["year"], r["conservative"]["roi"]) for r in out["oos_years"] if r["full"]]
    out["q1_verdict"] = q1_verdict(
        len(oos), len({e.day for e in oos}), pooled["conservative"], full
    )
    out["q1_verdict_by_variant"] = {
        v: q1_verdict(
            len(oos),
            len({e.day for e in oos}),
            pooled[v],
            [(r["year"], r[v]["roi"]) for r in out["oos_years"] if r["full"]],
        )
        for v in ("realistic", "prefee")
    }
    b55, _ = replay_all(c, oos_ev, RuleConfig(band_low=0.55), g)
    out["sens_band055"] = {
        "pooled": taker_stats(b55, "conservative"),
        "years": [
            {"year": r["year"], "n": r["n"], **r["conservative"]} for r in year_table(b55, cov)
        ],
    }
    out["by_cell_oos"] = {
        cv: taker_stats([e for e in oos if f"{e.event.city}_{e.event.var}" == cv], "conservative")
        for cv in sorted({f"{e.event.city}_{e.event.var}" for e in oos})
    }
    out["sample_rows"] = [_entry_row(e) for e in (oos + ins)[:: max(1, len(oos + ins) // 10)][:10]]

    # Minute data: 1-minute-entry sensitivity, capacity, Q2.
    minute: dict[str, list[Bar]] = {}
    trades: dict[str, list[Trade]] = {}
    for e in oos + ins:
        mb = minute_candles(c, e.market, e.event)
        tp = tape(c, e.market, e.event)
        if mb is not None and tp is not None:
            minute[e.market.ticker] = mb
            trades[e.market.ticker] = tp
    out["minute_coverage"] = {
        "oos": sum(e.market.ticker in minute for e in oos),
        "oos_n": len(oos),
        "is": sum(e.market.ticker in minute for e in ins),
        "is_n": len(ins),
    }
    sub = [e for e in oos if e.market.ticker in minute]
    me = [x for e in sub if (x := minute_entry(e, minute[e.market.ticker], g, rule)) is not None]
    out["sens_minute_entry"] = {
        "hourly_same_subset": taker_stats(sub, "conservative"),
        "minute": taker_stats(me, "conservative"),
        "n_hourly": len(sub),
        "n_minute": len(me),
    }

    cap_by_day: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for e in sub:
        n, d = capacity(e, trades[e.market.ticker], g)
        cap_by_day[e.day][0] += n
        cap_by_day[e.day][1] += d
    cap_years: dict[str, Any] = {}
    for y in sorted({int(d[:4]) for d in cap_by_day} | {0}):
        vals = [v for d, v in cap_by_day.items() if y == 0 or int(d[:4]) == y]
        if not vals:
            continue
        cs, ds = [v[0] for v in vals], [v[1] for v in vals]
        cap_years["all" if y == 0 else str(y)] = {
            "days": len(vals),
            "contracts_p25": _pct(cs, 25),
            "contracts_median": _pct(cs, 50),
            "contracts_p75": _pct(cs, 75),
            "dollars_p25": _pct(ds, 25),
            "dollars_median": _pct(ds, 50),
            "dollars_p75": _pct(ds, 75),
        }
    out["capacity_oos"] = cap_years

    q2: dict[str, Any] = {}
    for maker_fee_on in (False, True):
        for h in Q2_HORIZONS_MIN:
            sig = [
                q2_signal(e, trades[e.market.ticker], minute[e.market.ticker], h, g, maker_fee_on)
                | {"taker_pnl": e.yes - e.ask - taker_fee(e.ask), "ask": e.ask, "bid": e.bid}
                for e in sub
            ]
            key = f"{'end' if h is None else h}{'_makerfee' if maker_fee_on else ''}"
            q2[key] = q2_summary(sig)
    out["q2"] = q2
    full_years = [r["year"] for r in out["oos_years"] if r["full"]]
    for v in q2.values():
        yrs = {r["year"]: r for r in v["years"]}
        last = full_years[-1] if full_years else None
        v["most_recent_full_year"] = last
        v["most_recent_full_year_diff"] = (
            yrs[last]["diff_per_signal"]["roi"] if last in yrs else None
        )
    out["q2_promising"] = {
        k: (v["pooled"]["diff_per_signal"]["lo"] > 0 and (v["most_recent_full_year_diff"] or 0) > 0)
        for k, v in q2.items()
        if not k.endswith("_makerfee")
    }

    if args.audit:
        live_ev = [e for e in events if LIVE_CHECK[0] <= e.forecast_date <= LIVE_CHECK[1]]
        out["live_check"] = live_check(c, live_ev, Path(args.audit), g, cal.artifact_hash or "")
    out["cache"] = {"cached_reads": c.n_cached, "offline_misses": c.n_miss_offline}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "q1_verdict",
                    "q1_verdict_by_variant",
                    "oos_pooled",
                    "q2_promising",
                    "minute_coverage",
                    "missing_hourly",
                    "cache",
                )
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {dest}")


def q2_summary(sig: list[dict[str, Any]]) -> dict[str, Any]:
    def block(s: list[dict[str, Any]]) -> dict[str, Any]:
        if not s:
            return {"n": 0}
        days = [x["day"] for x in s]
        res: dict[str, Any] = {"n": len(s)}
        for kind in ("opt", "pes"):
            f = [x[f"fill_{kind}"] for x in s]
            filled = [x for x in s if x[f"fill_{kind}"]]
            unfilled = [x for x in s if not x[f"fill_{kind}"]]
            gap = paired_gap_ci(days, [x["yes"] for x in s], f)
            res[kind] = {
                "fill_rate": float(np.mean(f)),
                "filled": _wr_roi(filled, maker=True),
                "unfilled": _wr_roi(unfilled, maker=False),
                "adverse_gap": {"point": gap[0], "lo": gap[1], "hi": gap[2]},
            }
        traded = [x for x in s if x["h_traded"]]
        res["hybrid"] = {
            "n_traded": len(traded),
            "per_signal": roi_ci(days, [x["h_pnl"] for x in s], [1.0] * len(s)),
            "per_dollar": roi_ci(
                [x["day"] for x in traded],
                [x["h_pnl"] for x in traded],
                [x["h_wag"] for x in traded],
            )
            if traded
            else None,
        }
        res["taker"] = {
            "per_signal": roi_ci(days, [x["taker_pnl"] for x in s], [1.0] * len(s)),
            "per_dollar": roi_ci(days, [x["taker_pnl"] for x in s], [x["ask"] for x in s]),
        }
        res["diff_per_signal"] = roi_ci(
            days, [x["h_pnl"] - x["taker_pnl"] for x in s], [1.0] * len(s)
        )
        return res

    years = sorted({x["year"] for x in sig})
    return {
        "pooled": block(sig),
        "years": [{"year": y, **block([x for x in sig if x["year"] == y])} for y in years],
    }


def _wr_roi(s: list[dict[str, Any]], *, maker: bool) -> dict[str, Any]:
    if not s:
        return {"n": 0}
    price = "bid" if maker else "ask"
    pnl = [x["yes"] - x[price] - (0.0 if maker else taker_fee(x[price])) for x in s]
    return {
        "n": len(s),
        "win": float(np.mean([x["yes"] for x in s])),
        **roi_ci([x["day"] for x in s], pnl, [x[price] for x in s]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--stage", choices=("catalog", "hourly", "minute"), required=True)
    f.add_argument("--pilot", action="store_true", help="NYC high, one month per year")
    f.add_argument(
        "--min-interval",
        type=float,
        default=None,
        help="seconds between requests (default 1.0 on the VPS, 0.5 elsewhere; never lower)",
    )
    a = sub.add_parser("analyze")
    a.add_argument("--pilot", action="store_true")
    a.add_argument("--audit", help="favorite_bias_audit.jsonl for the Step 4 live-replay check")
    a.add_argument("--out", default=str(G0_DIR / "results.json"))
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        cmd_fetch(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
