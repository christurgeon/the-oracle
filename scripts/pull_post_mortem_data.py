"""Pull post-mortem data for a past date.

Pulls signal logs (active + rotated siblings) from VPS, captures
operational status via SSH, queries Kalshi /portfolio/fills, reconciles
local trade log against broker reality, and writes a manifest.

Usage:
    uv run python scripts/pull_post_mortem_data.py --date YYYY-MM-DD [--no-pull]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.analytics.audit_log import FUNNEL_SCHEMA_VERSION  # noqa: E402
from python.execution.trade_log import is_pnl_relevant_row  # noqa: E402
from scripts._vps import SshFailure, ssh_command  # noqa: E402, I001

# Import the funnel schema version from the production writer so this consumer
# tracks it automatically — hardcoding drifted (V3 vs prod V6) and produced a
# spurious "stale interpretations" warning on every post-mortem pull.
EXPECTED_SCHEMA_VERSIONS: dict[str, int] = {
    "eval_audit": 1,
    "candidate_audit": 1,
    "scan_funnel": FUNNEL_SCHEMA_VERSION,
}


@dataclass(frozen=True)
class SchemaVersionWarning:
    stream: str
    expected: int
    actual: int

    def message(self) -> str:
        return (
            f"Audit schema version mismatch: {self.stream} got "
            f"V{self.actual}, expected V{self.expected} - interpretations "
            f"may be stale"
        )


def check_schema_version(src: Path, stream: str) -> SchemaVersionWarning | None:
    """Return a warning if the first row's schema_version doesn't match expectation."""
    if not src.exists():
        return None
    expected = EXPECTED_SCHEMA_VERSIONS.get(stream)
    if expected is None:
        return None
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        actual = obj.get("schema_version")  # production key (no leading underscore)
        if actual is not None and actual != expected:
            return SchemaVersionWarning(stream=stream, expected=expected, actual=actual)
        return None  # first valid row had matching (or missing) version
    return None  # empty file


def validate_fill_created_time(fill: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Ensure ``created_time`` exists and is a string. Otherwise mark and warn."""
    ct = fill.get("created_time")
    order_id = fill.get("order_id", "<unknown>")
    if ct is None:
        out = dict(fill)
        out["_no_created_time"] = True
        return out, f"fill {order_id}: missing created_time"
    if not isinstance(ct, str):
        out = dict(fill)
        out["_no_created_time"] = True
        return out, f"fill {order_id}: non-string created_time ({type(ct).__name__})"
    return fill, None


class DateValidationError(ValueError):
    """Raised when --date is today or future (CD2)."""


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def validate_date(target: date) -> None:
    """Raise DateValidationError if ``target`` is today (UTC) or future."""
    today_utc = datetime.now(UTC).date()
    if target >= today_utc:
        earliest = today_utc - timedelta(days=1)
        raise DateValidationError(
            f"Refusing to analyze {target}: settlement data unavailable for "
            f"today/future. Earliest analyzable date: {earliest.isoformat()}."
        )


def output_dir_for(repo_root: Path, target: date) -> Path:
    return repo_root / "data" / "post_mortems" / target.isoformat()


def filter_signal_logs(sources: list[Path], dest: Path, target: date) -> None:
    """Concatenate ``sources``, dedupe by row identity, filter by ``ts[:10]``.

    Writes filtered output to ``dest``. Missing source files are silently
    skipped (caller surfaces missing-file warnings via the bundle).
    """
    target_iso = target.isoformat()
    seen_lines: set[str] = set()
    out_rows: list[str] = []
    for src in sources:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("ts")
            if not isinstance(ts, str) or ts[:10] != target_iso:
                continue
            if line in seen_lines:
                continue
            seen_lines.add(line)
            out_rows.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out_rows) + ("\n" if out_rows else ""))


JOURNALCTL_TIMEOUT_SECONDS = 120  # journalctl over a UTC day can be slow on loaded VPS
BOT_UNIT = "oracle-bot.service"


def _journalctl_safe(cmd: str) -> str:
    """Wrap ssh_command for journalctl queries that return exit 1 on no matches.

    Returns the journalctl output, or empty string if exit-1 means "no entries".
    Re-raises on real transport failures (connection refused, permission denied, etc.).
    """
    try:
        return ssh_command(cmd, timeout=JOURNALCTL_TIMEOUT_SECONDS)
    except SshFailure as e:
        msg = str(e).lower()
        # journalctl prints "-- No entries --" to stderr when nothing matches
        if "no entries" in msg:
            return ""
        # Real transport failures: re-raise so the caller can surface "unknown"
        raise


def capture_operational_status(target: date) -> dict[str, Any]:
    """Capture VPS operational state for ``target`` via SSH'd journalctl + git log.

    Returns a dict with ``summary`` and detail arrays. On SSH failure,
    returns ``summary="unknown"`` with ``ssh_failure`` set; never raises.
    """
    next_day = (target + timedelta(days=1)).isoformat()
    iso = target.isoformat()

    try:
        unit_log = _journalctl_safe(
            f"journalctl -u {BOT_UNIT} --since '{iso}T00:00:00Z' "
            f"--until '{next_day}T00:00:00Z' --no-pager",
        )
        kill_log = _journalctl_safe(
            f"journalctl -u {BOT_UNIT} --since '{iso}T00:00:00Z' "
            f"--until '{next_day}T00:00:00Z' --no-pager -g 'kill switch'",
        )
        # git log doesn't have the "no entries" semantics — empty stdout on success is fine.
        # Use ISO datetime form to match the journalctl pattern and avoid git treating
        # naked YYYY-MM-DD boundaries as inclusive of the entire end day.
        git_log = ssh_command(
            f"cd /opt/oracle && git log "
            f"--since='{iso}T00:00:00Z' --until='{next_day}T00:00:00Z' "
            f"--pretty=format:'%H %s'",
            timeout=JOURNALCTL_TIMEOUT_SECONDS,
        )
    except SshFailure as e:
        return {
            "summary": "unknown",
            "unit_down_intervals": [],
            "kill_switch_intervals": [],
            "deploys": [],
            "ssh_failure": str(e),
        }

    unit_down_intervals = _parse_unit_down(unit_log)
    kill_switch_intervals = _parse_kill_switch(kill_log)
    deploys = _parse_deploys(git_log)

    has_unit_down = bool(unit_down_intervals)
    has_kill = bool(kill_switch_intervals)
    has_deploy = bool(deploys)
    flags = sum([has_unit_down, has_kill, has_deploy])

    if flags >= 2:
        summary = "mixed"
    elif has_unit_down:
        summary = "unit-down"
    elif has_kill:
        summary = "kill-switch"
    elif has_deploy:
        summary = "deploy"
    else:
        summary = "nominal"

    return {
        "summary": summary,
        "unit_down_intervals": unit_down_intervals,
        "kill_switch_intervals": kill_switch_intervals,
        "deploys": deploys,
        "ssh_failure": None,
    }


def _parse_unit_down(log: str) -> list[dict[str, str]]:
    """Best-effort extraction of unit-stop/start intervals from journalctl.

    Matches only systemd Stopped/Started/"Failed with result" lines for the
    bot unit, not other 'Failed' content (kernel, other services, or
    systemd's "Failed to kill control group" notices for the bot unit).

    If a unit-down event has no recovery before EOD, emits a
    "still-down-at-eod" record mirroring _parse_kill_switch behaviour.
    """
    intervals: list[dict[str, str]] = []
    cur_stop: str | None = None
    for line in log.splitlines():
        is_unit_failure = f"{BOT_UNIT}: Failed with result" in line
        if f"Stopped {BOT_UNIT}" in line or is_unit_failure:
            cur_stop = line[:15]  # journalctl date prefix
        elif f"Started {BOT_UNIT}" in line and cur_stop is not None:
            intervals.append({"start": cur_stop, "end": line[:15], "reason": "stop+start"})
            cur_stop = None
    if cur_stop is not None:
        intervals.append(
            {"start": cur_stop, "end": "still-down-at-eod", "reason": "stopped-no-restart"}
        )
    return intervals


_KILL_SWITCH_ON_PHRASES = (
    "kill switch activated",
    "kill switch — halting",
    "activating kill switch",
    "kill switch active at startup",
)
_KILL_SWITCH_OFF_PHRASES = (
    "kill switch cleared",
    "kill switch deactivated",
    "kill switch inactive at startup",
)


def _parse_kill_switch(log: str) -> list[dict[str, str]]:
    """Extract kill-switch activation ... clear intervals from journalctl.

    Phrases match the bot's own log messages (risk manager STOP-file
    transitions, Telegram /kill and /unkill, the tick-loop halt, the
    tick-failure auto-kill, and the orchestrator's startup state line). One
    activation can log several of them, so a repeat activation keeps the
    first start. A restart with ``state/STOP`` removed logs no "cleared" line;
    the startup "inactive" line closes the interval instead.
    """
    intervals: list[dict[str, str]] = []
    cur_start: str | None = None
    for line in log.splitlines():
        lowered = line.lower()
        if any(p in lowered for p in _KILL_SWITCH_OFF_PHRASES):
            if cur_start is not None:
                intervals.append({"start": cur_start, "end": line[:15]})
                cur_start = None
        elif any(p in lowered for p in _KILL_SWITCH_ON_PHRASES):
            if cur_start is None:
                cur_start = line[:15]
    if cur_start is not None:
        intervals.append({"start": cur_start, "end": "still-active-at-eod"})
    return intervals


def _parse_deploys(log: str) -> list[dict[str, str]]:
    """Parse `git log --pretty=format:'%H %s'` output into deploy records."""
    deploys: list[dict[str, str]] = []
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        deploys.append({"sha": sha, "subject": subject})
    return deploys


def _parse_kalshi_count(count_fp: Any) -> int:
    """Convert Kalshi count_fp to integer contracts.

    count_fp is a plain contract count (e.g. "1.00", "12.00"), NOT in units
    of 10^9 micros. Production engine.py:556 treats it the same way.
    """
    return round(float(count_fp))


def _parse_kalshi_price(value: Any) -> float:
    return float(value)


def aggregate_kalshi_fills(fills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate Kalshi fills by order_id.

    Each fill is atomic and non-cumulative (true sum across fills):
    - vwap        = sum(price * count) / sum(count)  (count-weighted)
    - total_count = sum(count)
    - fee_total   = sum(fee_cost)

    Side flag picks which price field to read.
    """
    by_order: dict[str, list[dict[str, Any]]] = {}
    for f in fills:
        oid = f.get("order_id")
        if not oid:
            continue
        by_order.setdefault(oid, []).append(f)

    agg: dict[str, dict[str, Any]] = {}
    for oid, group in by_order.items():
        weighted_price_sum = 0.0
        count_sum = 0
        fee_sum = 0.0
        for f in group:
            count = _parse_kalshi_count(f.get("count_fp", 0))
            side = f.get("side", "yes").lower()
            price_field = "yes_price_dollars" if side == "yes" else "no_price_dollars"
            price = _parse_kalshi_price(f.get(price_field, 0.0))
            weighted_price_sum += price * count
            count_sum += count
            fee_sum += _parse_kalshi_price(f.get("fee_cost", 0.0))
        agg[oid] = {
            "vwap": (weighted_price_sum / count_sum) if count_sum else 0.0,
            "total_count": count_sum,
            "fee_total": fee_sum,
        }
    return agg


FILL_PRICE_TOLERANCE = 0.005  # one tick on Kalshi's cents grid
FEE_TOLERANCE = 0.01  # 2x worst-case rounding


def reconcile(
    local: dict[str, dict[str, Any]],
    kalshi: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare aggregated local orders vs Kalshi fills.

    - shadow-* in local: silent skip
    - local-only non-shadow: ghost_trade
    - kalshi-only: missed_log
    - both present: compare vwap (tolerance $0.005), count (exact), fee (tolerance $0.01)
    """
    deltas: list[dict[str, Any]] = []

    all_ids = set(local.keys()) | set(kalshi.keys())
    for oid in sorted(all_ids):
        in_local = oid in local
        in_kalshi = oid in kalshi

        if in_local and not in_kalshi:
            if local[oid].get("is_shadow"):
                continue
            deltas.append({"kind": "ghost_trade", "order_id": oid, "local": local[oid]})
            continue

        if in_kalshi and not in_local:
            deltas.append({"kind": "missed_log", "order_id": oid, "kalshi": kalshi[oid]})
            continue

        loc, kal = local[oid], kalshi[oid]
        if abs(loc["vwap"] - kal["vwap"]) > FILL_PRICE_TOLERANCE:
            deltas.append({
                "kind": "vwap_mismatch",
                "order_id": oid,
                "local_vwap": loc["vwap"],
                "kalshi_vwap": kal["vwap"],
                "diff": loc["vwap"] - kal["vwap"],
            })
        if loc["total_count"] != kal["total_count"]:
            deltas.append({
                "kind": "count_mismatch",
                "order_id": oid,
                "local_count": loc["total_count"],
                "kalshi_count": kal["total_count"],
            })
        if abs(loc["fee_total"] - kal["fee_total"]) > FEE_TOLERANCE:
            deltas.append({
                "kind": "fee_mismatch",
                "order_id": oid,
                "local_fee": loc["fee_total"],
                "kalshi_fee": kal["fee_total"],
                "diff": loc["fee_total"] - kal["fee_total"],
            })

    n_local_non_shadow = sum(1 for v in local.values() if not v.get("is_shadow"))
    return {
        "clean": len(deltas) == 0,
        "deltas": deltas,
        "summary": {
            "local_orders_total": len(local),
            "local_non_shadow": n_local_non_shadow,
            "kalshi_orders_total": len(kalshi),
            "delta_count": len(deltas),
            "tolerances": {"fill_price": FILL_PRICE_TOLERANCE, "fee": FEE_TOLERANCE},
        },
    }


def aggregate_local_orders(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate trade-log rows by order_id.

    The trade log writes one row per polling cycle with cumulative semantics:
    - ``fill_price`` is the cumulative VWAP across all fills so far for
      this order (see ``ExecutionEngine._compute_fill_price``).
    - ``contracts`` is the *delta* new fills since the last poll.
    - ``fees_from_api`` is the cumulative fee total per Kalshi.

    Aggregation rule:
    - vwap        = last_row.fill_price (already cumulative; NEVER re-weight)
    - total_count = sum(contracts) across rows  (delta sum)
    - fee_total   = last_row.fees_from_api    (cumulative; NEVER sum)
    """
    by_order: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if not is_pnl_relevant_row(r):
            continue
        oid = r.get("order_id")
        if not oid:
            continue
        by_order.setdefault(oid, []).append(r)

    agg: dict[str, dict[str, Any]] = {}
    for oid, group in by_order.items():
        sorted_group = sorted(group, key=lambda r: r.get("timestamp", ""))
        last = sorted_group[-1]
        agg[oid] = {
            "vwap": float(last.get("fill_price", 0.0)),
            "total_count": sum(int(r.get("contracts", 0)) for r in sorted_group),
            "fee_total": float(last.get("fees_from_api", 0.0)),
            "is_shadow": str(oid).startswith("shadow-"),
        }
    return agg


def write_bundle_manifest(
    out_dir: Path,
    *,
    warnings: list[str],
    schema_warnings: list[SchemaVersionWarning],
) -> None:
    files = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "bundle.json":
            rel = str(p.relative_to(out_dir))
            files[rel] = {"size": p.stat().st_size, "mtime": p.stat().st_mtime}
    bundle = {
        "generated_at": datetime.now(UTC).isoformat(),
        "warnings": warnings,
        "schema_warnings": [w.message() for w in schema_warnings],
        "files": files,
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))


def _required_cache_paths(repo_root: Path, target: date) -> list[Path]:
    """Sources the helper relies on in --no-pull mode.

    Checks the VPS-pulled source files, not the derived output files written
    to data/post_mortems/<D>/signals/. If the sources are missing, the helper
    would silently zero out the cached signals even if the output exists.

    Note: data/vps_trades.jsonl is intentionally excluded — its absence is a
    handled graceful-skip (reconciliation writes {"skipped": true}). Gating
    on it here would prevent legitimate no-pull runs on dates before live
    trading began.
    """
    audit_dir = repo_root / "data" / "vps_audit"
    return [
        # At least one of (active, rotated) for eval_audit must exist;
        # check active here as proxy.
        audit_dir / "eval_audit.jsonl",
    ]


async def _async_main(target: date, no_pull: bool) -> None:
    repo_root = Path(os.environ.get("ORACLE_POSTMORTEM_REPO_ROOT", _REPO_ROOT))
    out = output_dir_for(repo_root, target)
    out.mkdir(parents=True, exist_ok=True)
    signals = out / "signals"
    signals.mkdir(exist_ok=True)

    warnings: list[str] = []
    schema_warnings: list[SchemaVersionWarning] = []

    if no_pull:
        missing = [p for p in _required_cache_paths(repo_root, target) if not p.exists()]
        if missing:
            print(
                f"ERROR (--no-pull): required cache missing: {[str(m) for m in missing]}",
                file=sys.stderr,
            )
            sys.exit(3)

    # 1) signal logs (active + rotated, dedupe + filter)
    audit_pull_dir = repo_root / "data" / "vps_audit"
    if not no_pull:
        from scripts._vps import pull_rotated_audit_files
        pull_warnings = pull_rotated_audit_files(target, audit_pull_dir)
        warnings.extend(pull_warnings)

    for stream in ("eval_audit", "candidate_audit", "scan_funnel"):
        sources = [
            audit_pull_dir / f"{stream}.jsonl",
            audit_pull_dir / f"{stream}.jsonl.{target.isoformat()}",
        ]
        out_path = signals / f"{stream}.jsonl"
        filter_signal_logs(sources, out_path, target)
        if out_path.exists() and out_path.stat().st_size > 0:
            sw = check_schema_version(out_path, stream)
            if sw is not None:
                schema_warnings.append(sw)
        elif not any(s.exists() for s in sources):
            warnings.append(f"{stream}: rotated file pruned beyond retention window")

    # 2) operational status
    if not no_pull:
        status = capture_operational_status(target)
    else:
        status_path = out / "operational_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            warnings.append(
                "operational_status loaded from cache (no-pull mode)"
            )
        else:
            status = {
                "summary": "unknown",
                "ssh_failure": "no-pull mode",
                "unit_down_intervals": [],
                "kill_switch_intervals": [],
                "deploys": [],
            }
    (out / "operational_status.json").write_text(json.dumps(status, indent=2))

    # 3) Kalshi fills (Option 4a — confirmed by Task 2 probe)
    fills_out: list[dict[str, Any]] = []
    if not no_pull:
        from python.data.kalshi_client import KalshiClient
        from python.settings import Settings
        settings = Settings.load()
        if settings.kalshi_api_key and settings.kalshi_private_key_path:
            try:
                async with KalshiClient(
                    api_key=settings.kalshi_api_key,
                    private_key_path=settings.kalshi_private_key_path,
                    base_url=settings.kalshi_base_url,
                ) as client:
                    d0 = datetime.combine(target, datetime.min.time(), tzinfo=UTC)
                    d1 = d0 + timedelta(days=1)
                    raw_fills: list[dict[str, Any]] = []
                    cursor: str | None = None
                    max_pages = 10  # safety cap; 1000 fills/day >> pilot scale
                    for _ in range(max_pages):
                        kwargs: dict[str, Any] = {
                            "min_ts": int(d0.timestamp()),
                            "max_ts": int(d1.timestamp()),
                            "limit": 100,
                        }
                        if cursor:
                            kwargs["cursor"] = cursor
                        resp = await client.get_fills(**kwargs)
                        page = resp.get("fills", [])
                        raw_fills.extend(page)
                        cursor = resp.get("cursor") or None
                        if not cursor:
                            break
                    else:
                        warnings.append(
                            "Kalshi fills pagination hit MAX_PAGES=10 cap; "
                            "results may be truncated"
                        )
            except Exception as e:  # noqa: BLE001 — best-effort
                warnings.append(f"Kalshi fills retrieval failed: {e}")
                raw_fills = []
        else:
            warnings.append("Kalshi credentials missing; skipping fills retrieval")
            raw_fills = []

        cleaned: list[dict[str, Any]] = []
        for f in raw_fills:
            c, w = validate_fill_created_time(f)
            cleaned.append(c)
            if w:
                warnings.append(w)
        fills_out = cleaned
    else:
        fills_path = out / "fills.json"
        if fills_path.exists():
            fills_out = json.loads(fills_path.read_text())
    (out / "fills.json").write_text(json.dumps(fills_out, indent=2))

    # 4) reconciliation
    # Reconcile against ALL fills regardless of created_time (the per-fill warning
    # already informs the operator that the timestamp is suspect; excluding these
    # fills entirely creates worse false-positives in reconciliation).
    eligible_fills = fills_out
    n_no_ct = sum(1 for f in fills_out if f.get("_no_created_time"))
    if n_no_ct:
        warnings.append(
            f"{n_no_ct} fill(s) had missing/non-string created_time but are "
            f"INCLUDED in reconciliation (see per-fill warnings above)"
        )

    trades_path = repo_root / "data" / "vps_trades.jsonl"
    if not trades_path.exists():
        warnings.append("data/vps_trades.jsonl missing; reconciliation skipped")
        recon: dict[str, Any] = {
            "clean": None,
            "skipped": True,
            "reason": "data/vps_trades.jsonl missing",
            "deltas": [],
            "summary": {
                "local_orders_total": 0,
                "local_non_shadow": 0,
                "kalshi_orders_total": len(eligible_fills),
                "delta_count": 0,
                "tolerances": {"fill_price": FILL_PRICE_TOLERANCE, "fee": FEE_TOLERANCE},
            },
        }
    else:
        local_rows: list[dict[str, Any]] = []
        target_iso = target.isoformat()
        for line in trades_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("timestamp", "")
            if isinstance(ts, str) and ts[:10] == target_iso:
                local_rows.append(obj)
        local_agg = aggregate_local_orders(local_rows)
        kalshi_agg = aggregate_kalshi_fills(eligible_fills)
        recon = reconcile(local_agg, kalshi_agg)
    (out / "reconciliation.json").write_text(json.dumps(recon, indent=2))

    # 5) manifest
    write_bundle_manifest(out, warnings=warnings, schema_warnings=schema_warnings)
    print(f"Wrote bundle to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=parse_date, help="ISO date (YYYY-MM-DD)")
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip VPS SCP and Kalshi API calls; use cached data only.",
    )
    args = parser.parse_args()

    try:
        validate_date(args.date)
    except DateValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    asyncio.run(_async_main(args.date, args.no_pull))


if __name__ == "__main__":
    main()
