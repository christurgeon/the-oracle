"""Intraday EMOS HIGH shadow calibration report.

Reads intraday prediction log entries from ``state/predictions.jsonl`` and
joins them to NWS-settled temperatures from ``data/snapshots/*/settlements.json``
to produce a calibration report for the intraday_emos_high model.

Headline metric is CALIBRATION (reliability table), not P&L.
Secondary P&L is labeled NON-DECISION-GRADE and FILL-OPTIMISTIC.

Run (cached):  PYTHONPATH=. uv run python scripts/intraday_shadow_report.py
Run (fresh):   uv run python scripts/intraday_shadow_report.py --pull-vps
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.analytics.prediction_log import PredictionLog, PredictionRecord  # noqa: E402
from scripts._calibration_common import prefer_latest_settlement  # noqa: E402
from scripts._vps import VPS_STATE_DIR, pull_audit_files, require_vps_host  # noqa: E402

STRATEGY_NAME = "intraday_emos_high"

DEFAULT_PREDICTIONS = _PROJECT_ROOT / "state" / "predictions.jsonl"
DEFAULT_SNAPSHOTS = _PROJECT_ROOT / "data" / "snapshots"

_VARIABLE_TO_SETTLEMENT_FIELD = {
    "high": "high_f",
    "temperature_2m_max": "high_f",
    "low": "low_f",
    "temperature_2m_min": "low_f",
}

VPS_PREDICTIONS_PATH = f"{VPS_STATE_DIR}/predictions.jsonl"


# ---------------------------------------------------------------------------
# Settlement join
# ---------------------------------------------------------------------------


def load_settlements(snapshots_dir: Path) -> dict[tuple[str, str], dict]:
    """Return final-only settlements keyed by ``(city, observation_date_iso)``.

    Scans all snapshot subdirectories (flat ``<date>/`` and nested
    ``<date>/<run_id>/``) for ``settlements.json`` files. Does NOT require
    ``markets.json`` to be present — unlike ``load_snapshot_data``, which
    only enters a day directory when markets exist. The intraday report
    only needs settlements, so we scan every directory unconditionally.
    Merges by ``prefer_latest_settlement``; preliminary winners are dropped
    after the merge.
    """
    settlements: dict[tuple[str, str], dict] = {}
    if not snapshots_dir.is_dir():
        return settlements

    def _merge_settlements_file(path: Path) -> None:
        try:
            entries = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for s in entries:
            city = s.get("city")
            obs_date = s.get("observation_date")
            if not city or not obs_date:
                continue
            key = (city, obs_date)
            existing = settlements.get(key)
            if existing is None or prefer_latest_settlement(existing, s):
                settlements[key] = s

    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        st_file = day_dir / "settlements.json"
        if st_file.exists():
            _merge_settlements_file(st_file)
        else:
            for run_dir in sorted(day_dir.iterdir()):
                if run_dir.is_dir():
                    run_st = run_dir / "settlements.json"
                    if run_st.exists():
                        _merge_settlements_file(run_st)

    return {k: v for k, v in settlements.items() if not v.get("is_preliminary")}


def lookup_actual_temp(
    settlements: dict[tuple[str, str], dict],
    city: str,
    daily_variable: str,
    observation_date_iso: str,
) -> float | None:
    """Return the NWS-settled temperature for the given (city, variable, date).

    Returns None when no final settlement exists in the snapshot corpus.
    """
    st = settlements.get((city, observation_date_iso))
    if st is None:
        return None
    settle_field = _VARIABLE_TO_SETTLEMENT_FIELD.get(daily_variable)
    if settle_field is None:
        return None
    return st.get(settle_field)


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------


def evaluate_outcome(rec: PredictionRecord, actual_temp: float) -> int | None:
    """Return 1 if the prediction bracket was hit, 0 if not, None if indeterminate."""
    yes = derive_outcome_yes(
        rec.strike_type,
        rec.floor_strike,
        rec.cap_strike,
        actual_temp,
    )
    if yes is None:
        return None
    # If the signal favors YES, outcome==1 means the bracket was hit.
    # If the signal favors NO, the bracket outcome is still reported in YES-space
    # for calibration (calibrated_prob is always YES-space per PredictionRecord contract).
    return 1 if yes else 0


# ---------------------------------------------------------------------------
# Reliability table
# ---------------------------------------------------------------------------


def reliability_table(
    pairs: list[tuple[float, int]],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute a reliability (calibration) table from (predicted_prob, outcome) pairs.

    Each bin reports: bin label, n, mean predicted probability, empirical hit
    rate, and calibration gap (predicted - observed).
    Empty bins are omitted.
    """
    table: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [
            j
            for j, (p, _) in enumerate(pairs)
            if (p >= lo and p < hi) or (i == n_bins - 1 and p == 1.0)
        ]
        n = len(idx)
        if n == 0:
            continue
        pred = sum(pairs[j][0] for j in idx) / n
        obs = sum(pairs[j][1] for j in idx) / n
        table.append(
            {
                "bin": f"{int(lo * 100)}-{int(hi * 100)}%",
                "n": n,
                "predicted": round(pred, 4),
                "observed": round(obs, 4),
                "gap": round(pred - obs, 4),
            }
        )
    return table


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Mean squared error of probability forecasts. Returns None for empty input."""
    n = len(pairs)
    if n == 0:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / n


# ---------------------------------------------------------------------------
# Secondary (fill-optimistic) P&L
# ---------------------------------------------------------------------------


def secondary_pnl(
    records: list[PredictionRecord],
    outcomes: dict[str, int],
) -> dict[str, Any]:
    """Compute fill-optimistic P&L for settled predictions.

    Labeled NON-DECISION-GRADE / FILL-OPTIMISTIC:
      YES side: P&L = (1 - exec_price) * 1 if win, = -exec_price * 1 if loss.
      NO  side: P&L = (1 - exec_price) * 1 if win, = -exec_price * 1 if loss.

    Contracts are treated as 1 unit (no position sizing) since shadow mode
    does not produce fills. exec_price is the side-denominated fill price.
    """
    total_pnl = 0.0
    wagered = 0.0
    wins = 0
    losses = 0
    rows = []

    for rec in records:
        key = rec.market_ticker
        outcome_yes = outcomes.get(key)
        if outcome_yes is None:
            continue
        exec_p = rec.exec_price
        if exec_p is None:
            continue

        side_str = str(rec.side.value) if hasattr(rec.side, "value") else str(rec.side)
        won = (outcome_yes == 1 and side_str == "YES") or (
            outcome_yes == 0 and side_str == "NO"
        )
        pnl = (1.0 - exec_p) if won else -exec_p
        wagered += exec_p
        total_pnl += pnl
        if won:
            wins += 1
        else:
            losses += 1
        rows.append(
            {
                "ticker": rec.market_ticker,
                "side": side_str,
                "exec_price": round(exec_p, 4),
                "outcome_yes": outcome_yes,
                "won": won,
                "pnl": round(pnl, 4),
            }
        )

    n = wins + losses
    roi_pct = round((total_pnl / wagered) * 100, 1) if wagered > 0 else None
    win_rate_pct = round((wins / n) * 100, 1) if n > 0 else None
    return {
        "label": "FILL-OPTIMISTIC, NON-DECISION-GRADE",
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "roi_pct": roi_pct,
        "net_pnl": round(total_pnl, 4),
        "wagered": round(wagered, 4),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------


def build_report(
    predictions_path: Path,
    snapshots_dir: Path,
) -> dict[str, Any]:
    """Build the full calibration report dict. Pure — no I/O side effects."""
    log = PredictionLog(predictions_path)
    all_recs = [r for r in log.iter_all() if r.strategy == STRATEGY_NAME]

    settlements = load_settlements(snapshots_dir)

    settled_recs: list[PredictionRecord] = []
    unsettled_recs: list[PredictionRecord] = []
    outcome_by_ticker: dict[str, int] = {}
    actual_temp_by_ticker: dict[str, float] = {}

    for rec in all_recs:
        exp_iso = rec.expiration_date.isoformat() if rec.expiration_date else None
        if exp_iso is None or rec.city is None or rec.daily_variable is None:
            unsettled_recs.append(rec)
            continue
        actual = lookup_actual_temp(settlements, rec.city, rec.daily_variable, exp_iso)
        if actual is None:
            unsettled_recs.append(rec)
            continue
        outcome = evaluate_outcome(rec, actual)
        if outcome is None:
            unsettled_recs.append(rec)
            continue
        settled_recs.append(rec)
        outcome_by_ticker[rec.market_ticker] = outcome
        actual_temp_by_ticker[rec.market_ticker] = actual

    # Build calibrated_prob vs outcome pairs (YES-space calibration).
    cal_pairs: list[tuple[float, int]] = [
        (rec.calibrated_prob, outcome_by_ticker[rec.market_ticker])
        for rec in settled_recs
    ]
    raw_pairs: list[tuple[float, int]] = []
    for rec in settled_recs:
        raw = rec.raw_ensemble_prob if rec.raw_ensemble_prob is not None else rec.emos_prob_raw
        if raw is not None:
            raw_pairs.append((raw, outcome_by_ticker[rec.market_ticker]))

    emos_pairs: list[tuple[float, int]] = [
        (rec.emos_prob_raw, outcome_by_ticker[rec.market_ticker])
        for rec in settled_recs
        if rec.emos_prob_raw is not None
    ]

    # Per-bracket split.
    bracket_buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for rec in settled_recs:
        st = rec.strike_type or "unknown"
        bracket_buckets[st].append((rec.calibrated_prob, outcome_by_ticker[rec.market_ticker]))

    by_bracket = {}
    for st, pairs in bracket_buckets.items():
        bs = brier_score(pairs)
        n = len(pairs)
        hit_rate = sum(y for _, y in pairs) / n if n > 0 else None
        by_bracket[st] = {
            "n": n,
            "brier_score": round(bs, 6) if bs is not None else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        }

    # Per-city split.
    city_buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for rec in settled_recs:
        city = rec.city or "unknown"
        city_buckets[city].append((rec.calibrated_prob, outcome_by_ticker[rec.market_ticker]))

    by_city = {}
    for city, pairs in city_buckets.items():
        bs = brier_score(pairs)
        n = len(pairs)
        hit_rate = sum(y for _, y in pairs) / n if n > 0 else None
        by_city[city] = {
            "n": n,
            "brier_score": round(bs, 6) if bs is not None else None,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        }

    bs_cal = brier_score(cal_pairs)
    bs_raw = brier_score(raw_pairs) if raw_pairs else None
    bs_emos = brier_score(emos_pairs) if emos_pairs else None

    summary = {
        "strategy": STRATEGY_NAME,
        "n_total": len(all_recs),
        "n_settled": len(settled_recs),
        "n_unsettled": len(unsettled_recs),
        "brier_score_calibrated": round(bs_cal, 6) if bs_cal is not None else None,
        "brier_score_raw": round(bs_raw, 6) if bs_raw is not None else None,
        "brier_score_emos_raw": round(bs_emos, 6) if bs_emos is not None else None,
        "n_raw_pairs": len(raw_pairs),
        "n_emos_pairs": len(emos_pairs),
    }

    reliability = reliability_table(cal_pairs)
    sec_pnl = secondary_pnl(settled_recs, outcome_by_ticker)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "reliability": reliability,
        "by_bracket": by_bracket,
        "by_city": by_city,
        "secondary_pnl": sec_pnl,
    }


# ---------------------------------------------------------------------------
# VPS pull helpers
# ---------------------------------------------------------------------------


def pull_predictions_from_vps(local_path: Path) -> None:
    """SCP the predictions.jsonl from the VPS."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{require_vps_host()}:{VPS_PREDICTIONS_PATH}"
    print("Pulling predictions.jsonl from VPS...")
    result = subprocess.run(
        ["scp", remote, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        sys.exit(1)
    n = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    print(f"  -> {n} prediction rows saved to {local_path}")


# ---------------------------------------------------------------------------
# Console renderer
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1f}%" if v is not None else "n/a"


def _fmt_float(v: float | None, prec: int = 4) -> str:
    return f"{v:.{prec}f}" if v is not None else "n/a"


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    s = report["summary"]
    lines.append("=== INTRADAY EMOS HIGH SHADOW CALIBRATION REPORT ===")
    lines.append(f"generated: {report['generated_at']}")
    lines.append("")

    lines.append("--- SUMMARY ---")
    lines.append(f"strategy:        {s['strategy']}")
    lines.append(f"n_total:         {s['n_total']}")
    lines.append(f"n_settled:       {s['n_settled']}")
    lines.append(f"n_unsettled:     {s['n_unsettled']}")
    lines.append(
        f"brier_calibrated: {_fmt_float(s['brier_score_calibrated'])}  "
        f"(lower = better; identity baseline for reference)"
    )
    lines.append(
        f"brier_raw:        {_fmt_float(s['brier_score_raw'])}  "
        f"(n={s['n_raw_pairs']})"
    )
    lines.append(
        f"brier_emos_raw:   {_fmt_float(s['brier_score_emos_raw'])}  "
        f"(n={s['n_emos_pairs']})"
    )
    lines.append("")

    lines.append("--- RELIABILITY TABLE (calibrated_prob vs outcome) ---")
    rel = report["reliability"]
    if not rel:
        lines.append("  (no settled data)")
    else:
        header = f"  {'bin':>9}  {'n':>5}  {'predicted':>10}  {'observed':>10}  {'gap':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for row in rel:
            gap_str = f"{row['gap']:+.4f}"
            lines.append(
                f"  {row['bin']:>9}  {row['n']:>5}  "
                f"{row['predicted']:>10.4f}  {row['observed']:>10.4f}  {gap_str:>8}"
            )
    lines.append("")

    lines.append("--- BY BRACKET TYPE ---")
    for st, v in sorted(report["by_bracket"].items()):
        bs_str = _fmt_float(v["brier_score"])
        hr_str = _fmt_float(v["hit_rate"])
        lines.append(f"  {st:<10}  n={v['n']:<5}  brier={bs_str}  hit_rate={hr_str}")
    if not report["by_bracket"]:
        lines.append("  (no settled data)")
    lines.append("")

    lines.append("--- BY CITY ---")
    for city, v in sorted(report["by_city"].items()):
        bs_str = _fmt_float(v["brier_score"])
        hr_str = _fmt_float(v["hit_rate"])
        lines.append(f"  {city:<6}  n={v['n']:<5}  brier={bs_str}  hit_rate={hr_str}")
    if not report["by_city"]:
        lines.append("  (no settled data)")
    lines.append("")

    pnl = report["secondary_pnl"]
    lines.append(f"--- SECONDARY P&L [{pnl['label']}] ---")
    wr_str = f"{pnl['win_rate_pct']:.1f}%" if pnl["win_rate_pct"] is not None else "n/a"
    roi_str = _fmt_pct(pnl["roi_pct"])
    lines.append(
        f"  n={pnl['n']}  W={pnl['wins']}  L={pnl['losses']}  "
        f"win_rate={wr_str}  roi={roi_str}  "
        f"net_pnl=${pnl['net_pnl']:+.2f}  wagered=${pnl['wagered']:.2f}"
    )
    lines.append("  NOTE: contracts=1 unit (no Kelly sizing), exec_price=fill-optimistic")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Intraday EMOS HIGH shadow calibration report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Headline metric is CALIBRATION (reliability table), not P&L.\n"
            "Secondary P&L section is fill-optimistic and non-decision-grade."
        ),
    )
    p.add_argument(
        "--pull-vps",
        action="store_true",
        help="SCP latest predictions.jsonl from VPS before reporting",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Pull audit files from VPS (mirrors favorite_bias_report.py convention)",
    )
    p.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
        help=f"Path to predictions.jsonl (default: {DEFAULT_PREDICTIONS})",
    )
    p.add_argument(
        "--snapshots",
        default=str(DEFAULT_SNAPSHOTS),
        help=f"Snapshots dir for NWS settlements (default: {DEFAULT_SNAPSHOTS})",
    )
    p.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write structured JSON report to this path",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    predictions_path = Path(args.predictions)

    if args.pull_vps:
        pull_predictions_from_vps(predictions_path)

    if args.refresh:
        audit_dir = _PROJECT_ROOT / "data" / "vps_audit"
        pull_audit_files(audit_dir)

    report = build_report(predictions_path, Path(args.snapshots))
    print(render_report(report))

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON report written to {out_path}")


if __name__ == "__main__":
    main()
