"""favorite_bias spread-cap check: did ``max_spread`` help or hurt?

For each event (city, var, forecast_date) with an in-band favorite_bias candidate, compare two
hypothetical entries, one contract each, at the audit-row ask with the taker fee:

* **no cap:** the first ``buy`` or ``skip_spread`` audit row for the event;
* **cap:** the first such row whose spread is at or under the cap (none = no trade, P&L 0).

The per-event difference (cap − no cap) is pooled with a day-clustered CI. Only events where the
cap changed the entry (first row wider than the cap) can differ. The audit logs every scan whether
or not the gate is on, so this works with ``max_spread`` off. Events before
``--forward-from`` are the retrospective check; events after it are the forward check that decides.

Outcomes come from the public ``GET /markets/{ticker}`` endpoint (``/historical/markets/{ticker}``
for markets past the historical cutoff), cached in
``data/audit/fb_market_results.json``.

    PYTHONPATH=. uv run python scripts/favorite_bias_spread_cap_report.py \
        --forward-from 2026-10-05T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.edge_audit import day_clustered_bootstrap_ci  # noqa: E402
from python.analytics.fees import fee_per_contract  # noqa: E402
from python.utils.ticker_dates import parse_ticker_date  # noqa: E402

AUDIT_PATH = Path("/opt/oracle/state/favorite_bias_audit.jsonl")
RESULTS_CACHE = _PROJECT_ROOT / "data" / "audit" / "fb_market_results.json"
API = "https://api.elections.kalshi.com/trade-api/v2"
EPS = 1e-9
MIN_N = 40


@dataclass(frozen=True)
class Candidate:
    ts: datetime
    ticker: str
    bid: float
    ask: float


@dataclass(frozen=True)
class EventOutcome:
    event: str
    day: str
    changed: bool
    nocap: Candidate
    cap: Candidate | None


def event_key(row: dict[str, Any]) -> str | None:
    fd = parse_ticker_date(row["market_ticker"])
    if fd is None:
        return None
    return f"{row['city']}_{row['var']}_{fd.isoformat()}"


def pair_events(rows: list[dict[str, Any]], cap: float) -> list[EventOutcome]:
    by_ev: dict[str, list[Candidate]] = {}
    days: dict[str, str] = {}
    for r in rows:
        if r.get("decision") not in ("buy", "skip_spread"):
            continue
        if r.get("yes_bid") is None or r.get("yes_ask") is None:
            continue
        k = event_key(r)
        if k is None:
            continue
        by_ev.setdefault(k, []).append(
            Candidate(
                datetime.fromisoformat(r["ts"]), r["market_ticker"],
                float(r["yes_bid"]), float(r["yes_ask"]),
            )
        )
        days[k] = k.rsplit("_", 1)[1]
    out = []
    for k, cs in by_ev.items():
        cs.sort(key=lambda c: c.ts)
        tight = next((c for c in cs if c.ask - c.bid <= cap + EPS), None)
        out.append(EventOutcome(k, days[k], cs[0].ask - cs[0].bid > cap + EPS, cs[0], tight))
    return out


def pnl(c: Candidate | None, yes: dict[str, int]) -> float:
    if c is None:
        return 0.0
    return yes[c.ticker] - c.ask - fee_per_contract(c.ask, is_maker=False)


def load_results(tickers: set[str], *, offline: bool) -> dict[str, int]:
    cache: dict[str, str] = json.loads(RESULTS_CACHE.read_text()) if RESULTS_CACHE.exists() else {}
    todo = sorted(t for t in tickers if cache.get(t) not in ("yes", "no"))
    if todo and not offline:
        with httpx.Client(timeout=20.0) as http:
            for t in todo:
                for path in (f"/markets/{t}", f"/historical/markets/{t}"):
                    r = http.get(API + path)
                    time.sleep(0.5)
                    if r.status_code == 200:
                        cache[t] = (r.json().get("market") or {}).get("result") or ""
                        break
        RESULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return {t: (1 if v == "yes" else 0) for t, v in cache.items() if v in ("yes", "no")}


def summarize(events: list[EventOutcome], yes: dict[str, int]) -> dict[str, Any]:
    scored = [
        e for e in events
        if e.nocap.ticker in yes and (e.cap is None or e.cap.ticker in yes)
    ]
    changed = [e for e in scored if e.changed]
    diffs = [pnl(e.cap, yes) - pnl(e.nocap, yes) for e in changed]
    out: dict[str, Any] = {
        "events": len(scored),
        "changed": len(changed),
        "changed_never_tight": sum(e.cap is None for e in changed),
        "changed_later_tight": sum(e.cap is not None for e in changed),
    }
    if changed:
        lo, hi = day_clustered_bootstrap_ci(
            [e.day for e in changed], diffs, [1.0] * len(diffs), b=4000
        )
        out.update(
            diff_per_changed_event=sum(diffs) / len(diffs), ci=(lo, hi),
            diff_total_per_contract=sum(diffs),
            nocap_pnl_changed=sum(pnl(e.nocap, yes) for e in changed) / len(changed),
        )
    return out


def verdict(s: dict[str, Any]) -> str:
    if s["changed"] < MIN_N:
        return f"WAIT (n={s['changed']} < {MIN_N} changed events)"
    if s["diff_per_changed_event"] > 0:
        return "CAP HELPS: consider max_spread: 0.05"
    return "CAP HURTS OR NEUTRAL: keep max_spread: null"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, default=AUDIT_PATH)
    ap.add_argument("--cap", type=float, default=0.05)
    ap.add_argument("--forward-from", required=True, help="UTC ISO start of the forward window")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    start = datetime.fromisoformat(a.forward_from.replace("Z", "+00:00"))
    rows = [json.loads(line) for line in a.audit.read_text().splitlines() if line.strip()]
    events = pair_events(rows, a.cap)
    today = datetime.now(UTC).date().isoformat()
    events = [e for e in events if e.day < today]
    yes = load_results({e.nocap.ticker for e in events} | {e.cap.ticker for e in events if e.cap},
                       offline=a.offline)
    for label, sel in (
        ("retrospective (before --forward-from)", [e for e in events if e.nocap.ts < start]),
        ("forward (decides)", [e for e in events if e.nocap.ts >= start]),
    ):
        s = summarize(sel, yes)
        print(f"== {label}")
        print("   " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in s.items()}, default=str))
        if "forward" in label:
            print(f"   verdict: {verdict(s)}")


if __name__ == "__main__":
    main()
