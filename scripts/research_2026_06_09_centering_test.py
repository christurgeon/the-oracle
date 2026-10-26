"""Stage 1b centering test — the binding gate for the new-city intraday lane.

Question (spec §Stage-1 1b): do new-city same-day HIGH quote MIDPOINTS deviate
from an observation-anchored posterior systematically and by more than the
executable-cost bar (half-spread + taker fee)? Width alone is NOT a green light;
only a *centering* miss is harvestable by a taker.

THE TRAP this guards against (Opus finding 2): "market disagrees with our
posterior" is only harvestable edge if OUR posterior is the better-calibrated
one. Our draft new-city climatology is untested, so a raw new-city deviation
could be our model being wrong rather than the market being wrong. Control:
run the IDENTICAL computation on the 5 MATURE cities using the PRODUCTION
climatology — those books are known-efficient (the 2026-06-09 falsification),
so their mean(posterior − mid) is OUR MODEL'S OWN BIAS against truth. A
new-city signal is only credible NET of that mature baseline.

This is a SCREENING gate, not a profit measurement:
  - well-centered (new-city deviation ≈ mature baseline, within cost) → NULL,
    stop cheaply; no taker edge can exist.
  - mis-centered beyond baseline + cost → NECESSARY condition met; whether the
    disagreement is real edge is adjudicated by the forward V1 P&L + V2 per-bin
    reliability (settled outcomes), not by this script.

Parity with the live strategy (`intraday_asos_high`): same `ClimatologyTable`
loader + `.lookup(city, month, hour_utc, var)`, same `bracket_probability`
(truncated Student-t, nu=5), same LST-frame M (Stage 0a), same NWS rounding.

Usage: uv run python scripts/research_2026_06_09_centering_test.py
Output: data/research/2026-06-09-centering-test.json + console report.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from python.analytics.fees import fee_per_contract  # noqa: E402
from python.common import Direction  # noqa: E402
from python.data.asos import ClimatologyTable  # noqa: E402
from python.data.asos_posterior import bracket_probability  # noqa: E402

SNAP_DIR = REPO / "data" / "snapshots"
ASOS_DIR = REPO / "data" / "research" / "asos"
DRAFT_CLIM = REPO / "data" / "research" / "newcity_climatology.json"
PROD_CLIM = REPO / "data" / "asos" / "climatology.json"
OUT_PATH = REPO / "data" / "research" / "2026-06-09-centering-test.json"

NU = 5.0
SLOTS = ("0200", "0800", "1400", "2000")
MIN_OBS = 3
POST_PEAK_LST_HOUR = 13  # Stage-1a finding: METAR ~1°F bias contaminates pre-peak
MID_LO, MID_HI = 0.05, 0.95

# LST standard offsets (no DST) — must match build_newcity_climatology + production.
NEW_OFFSET = {
    "ATL": -5, "DC": -5, "DAL": -6, "HOU": -6, "MIN": -6, "NOLA": -6,
    "OKC": -6, "SATX": -6, "PHX": -7, "LV": -8, "SFO": -8,
}
MATURE_OFFSET = {"NYC": -5, "MIA": -5, "CHI": -6, "AUS": -6, "LAX": -8}
# city -> ASOS CSV file stem (k{stem}_asos.csv); station column = short code.
NEW_FILE = {
    "ATL": "katl", "DAL": "kdal", "DC": "kdc", "HOU": "khou", "LV": "klv",
    "MIN": "kmin", "NOLA": "knola", "OKC": "kokc", "PHX": "kphx",
    "SATX": "ksatx", "SFO": "ksfo",
}
MATURE_FILE = {"NYC": "knyc", "CHI": "kchi", "LAX": "klax", "MIA": "kmia", "AUS": "kaus"}
_STRIKE_TO_DIR = {
    "greater": Direction.ABOVE, "less": Direction.BELOW, "between": Direction.BETWEEN,
}


def load_asos_lst(stem: str, offset_h: int) -> list[tuple[float, int, float]]:
    """Return sorted [(utc_ts, lst_date_ordinal, tmpf)] for one city CSV."""
    path = ASOS_DIR / f"{stem}_asos.csv"
    if not path.exists():
        return []
    off = timedelta(hours=offset_h)
    rows: list[tuple[float, int, float]] = []
    with path.open() as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            try:
                tu = datetime.strptime(row[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
                rows.append(
                    (tu.timestamp(), (tu + off).date().toordinal(), float(row[2]))
                )
            except ValueError:
                continue
    rows.sort()
    return rows


def m_so_far(rows: list[tuple[float, int, float]], target_ord: int, run_ts: float) -> float | None:
    vals = [v for ts, dord, v in rows if ts <= run_ts and dord == target_ord]
    return max(vals) if len(vals) >= MIN_OBS else None


def load_run_ts() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with (SNAP_DIR / "index.jsonl").open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") in SLOTS:
                out[(rec["date"], rec["run_id"])] = float(rec["run_ts"])
    return out


def parse_target_date(ticker: str):
    try:
        return datetime.strptime(ticker.split("-")[1], "%y%b%d").date()
    except (IndexError, ValueError):
        return None


def evaluate_cohort(
    cities: dict[str, int],
    files: dict[str, str],
    clim: ClimatologyTable,
    run_ts_map: dict[tuple[str, str], float],
) -> list[dict]:
    """One (posterior, mid, executable-edge) record per same-day HIGH market eval."""
    asos = {c: load_asos_lst(files[c], off) for c, off in cities.items()}
    out: list[dict] = []
    for snap_day in sorted(p.name for p in SNAP_DIR.iterdir() if p.is_dir()):
        for slot in SLOTS:
            mpath = SNAP_DIR / snap_day / slot / "markets.json"
            rts = run_ts_map.get((snap_day, slot))
            if not mpath.exists() or rts is None:
                continue
            try:
                markets = json.loads(mpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            run_dt = datetime.fromtimestamp(rts, tz=UTC)
            for m in markets:
                city = m.get("city")
                if city not in cities or m.get("daily_variable") != "temperature_2m_max":
                    continue
                target = parse_target_date(m.get("ticker", ""))
                if target is None:
                    continue
                lst_now = run_dt + timedelta(hours=cities[city])
                if lst_now.date() != target:  # same-LST-day, pre-resolution only
                    continue
                bid, ask = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
                st = _STRIKE_TO_DIR.get(m.get("strike_type"))
                if bid is None or ask is None or st is None:
                    continue
                mid = (bid + ask) / 2
                if not (MID_LO <= mid <= MID_HI):
                    continue
                M = m_so_far(asos[city], target.toordinal(), rts)  # noqa: N806
                if M is None:
                    continue
                look = clim.lookup(city, target.month, run_dt.hour, var="high")
                if look is None:
                    continue
                mu_gain, sigma_gain, _n, _src = look
                try:
                    post = bracket_probability(
                        st, floor=m.get("floor_strike"), cap=m.get("cap_strike"),
                        M=M, mu_gain=mu_gain, sigma_gain=sigma_gain, nu=NU,
                    )
                except ValueError:
                    continue
                fee_y, fee_n = fee_per_contract(ask), fee_per_contract(1 - bid)
                out.append({
                    "snap_day": snap_day, "slot": slot, "city": city,
                    "lst_hour": lst_now.hour, "ticker": m["ticker"],
                    "mid": round(mid, 4), "half_spread": round((ask - bid) / 2, 4),
                    "posterior": round(post, 4),
                    "deviation": round(post - mid, 4),  # +: we say YES more likely than market
                    "edge_yes": round(post - ask - fee_y, 4),  # taker YES at ask
                    "edge_no": round(bid - post - fee_n, 4),    # taker NO at 1-bid
                    "sigma_gain": round(sigma_gain, 3),
                })
    return out


def summarize(records: list[dict], label: str) -> dict:
    if not records:
        return {"label": label, "n": 0}
    devs = [r["deviation"] for r in records]
    best_edge = [max(r["edge_yes"], r["edge_no"]) for r in records]
    harvestable = [r for r in records if max(r["edge_yes"], r["edge_no"]) > 0]
    return {
        "label": label,
        "n": len(records),
        "n_days": len({r["snap_day"] for r in records}),
        "n_tickers": len({r["ticker"] for r in records}),
        "mean_deviation": round(sum(devs) / len(devs), 4),
        "median_deviation": round(median(devs), 4),
        "mean_abs_deviation": round(sum(abs(d) for d in devs) / len(devs), 4),
        "median_half_spread": round(median(r["half_spread"] for r in records), 4),
        "frac_executable_edge_gt0": round(len(harvestable) / len(records), 4),
        "median_best_edge": round(median(best_edge), 4),
    }


def main() -> None:
    run_ts_map = load_run_ts()
    draft = ClimatologyTable.load(DRAFT_CLIM)
    prod = ClimatologyTable.load(PROD_CLIM)

    new_recs = evaluate_cohort(NEW_OFFSET, NEW_FILE, draft, run_ts_map)
    mat_recs = evaluate_cohort(MATURE_OFFSET, MATURE_FILE, prod, run_ts_map)

    def postpeak(recs):
        return [r for r in recs if r["lst_hour"] >= POST_PEAK_LST_HOUR]

    report: dict = {
        "generated": "2026-06-09",
        "method": "posterior(clim) vs market mid; LST frame; nu=5; post-peak LST>=13 binding",
        "post_peak_lst_hour": POST_PEAK_LST_HOUR,
        "mature_control_baseline": {
            "all": summarize(mat_recs, "mature_all"),
            "post_peak": summarize(postpeak(mat_recs), "mature_post_peak"),
        },
        "newcity": {
            "all": summarize(new_recs, "new_all"),
            "post_peak": summarize(postpeak(new_recs), "new_post_peak"),
            "post_peak_by_city": {
                c: summarize([r for r in postpeak(new_recs) if r["city"] == c], c)
                for c in sorted(NEW_OFFSET)
            },
        },
    }

    # Verdict. The discriminating quantity is new-city deviation NET of the
    # mature baseline — but the mature control also exposes a CONFOUND: on
    # known-efficient books our posterior still shows a large mean-abs-deviation
    # and ~all markets flag "executable edge" (the disagreement is OUR model's
    # miscalibration, not market mispricing). So the mature mean-abs-deviation is
    # the NOISE FLOOR: a net signal is only credible if it both clears the cost
    # bar AND is large relative to that floor, on enough INDEPENDENT days.
    mat_pp = report["mature_control_baseline"]["post_peak"]
    mat_base = mat_pp.get("mean_deviation", 0.0)
    noise_floor = mat_pp.get("mean_abs_deviation", 0.0)  # model error vs efficient market
    np_ = report["newcity"]["post_peak"]
    cost_bar = np_.get("median_half_spread", 0.0)
    n_pp = np_.get("n", 0)
    n_days = np_.get("n_days", 0)
    net_dev = round(np_.get("mean_deviation", 0.0) - mat_base, 4) if n_pp else None
    min_days, min_n = 14, 60  # forward-accrual floor before a real screen
    if n_pp < min_n or n_days < min_days:
        decision = (
            f"UNDERPOWERED — only {n_pp} post-peak evals across {n_days} day(s) "
            f"(need >= {min_n} over >= {min_days} days). No screen possible yet."
        )
    elif net_dev is not None and (abs(net_dev) > cost_bar and abs(net_dev) > 0.5 * noise_floor):
        decision = (
            "NECESSARY-CONDITION-MET — new-city deviation departs from the mature "
            "baseline beyond cost and noise floor. Proceed to V1 forward P&L "
            "(settled outcomes), which is the actual edge adjudicator."
        )
    else:
        decision = (
            "NULL-SCREEN — new-city centering indistinguishable from the mature "
            "baseline within cost/noise. No taker edge; stop."
        )
    report["verdict"] = {
        "decision": decision,
        "mature_baseline_mean_deviation": mat_base,
        "mature_noise_floor_mean_abs_deviation": noise_floor,
        "mature_frac_executable_edge_gt0": mat_pp.get("frac_executable_edge_gt0"),
        "newcity_post_peak_mean_deviation": np_.get("mean_deviation"),
        "newcity_post_peak_n": n_pp,
        "newcity_post_peak_days": n_days,
        "net_deviation_vs_baseline": net_dev,
        "cost_bar_median_half_spread": cost_bar,
        "confound_note": (
            "The mature control shows ~all known-efficient-book markets flag "
            "'executable edge' against our posterior — apparent edge is dominated "
            "by model miscalibration, NOT mispricing. Centering is therefore a "
            "WEAK screen: only the new-vs-mature NET difference, accrued over many "
            "independent days, is informative, and even a clean net signal is "
            "merely NECESSARY — V1 forward settled P&L is the binding adjudicator."
        ),
    }
    report["newcity_records"] = new_recs

    OUT_PATH.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT_PATH}\n")
    print("MATURE control (production clim, known-efficient books):")
    print(" ", report["mature_control_baseline"]["post_peak"])
    print("\nNEW-CITY (draft clim):")
    print("  all:      ", report["newcity"]["all"])
    print("  post-peak:", report["newcity"]["post_peak"])
    print("\nNEW-CITY post-peak by city:")
    for c, s in report["newcity"]["post_peak_by_city"].items():
        print(f"  {c}: {s}")
    print("\nVERDICT:")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
