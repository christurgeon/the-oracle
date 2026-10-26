#!/usr/bin/env python3
"""revert_cash_out.py — operator escape-hatch for cash-out reconciliation.

OFFLINE-ONLY. Requires `sudo systemctl stop oracle-bot.service` first.

Modes:
  --reset-dedup --order-id <oid>     Remove oid from all five _co_*_oids sets.
  --revert --order-id <oid>          Revert an applied cash-out (writes
                                     cash_out_reverted row; adjusts daily_pnl
                                     + strategy_trackers).
  --apply --order-id <oid>           Hand-apply a fill the safety knob blocked
                                     (writes cash_out_force_applied row).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

# Allow `python.*` imports when invoked as `python scripts/revert_cash_out.py`
# (the test harness does this) by ensuring the repo root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.utils.atomic_write import atomic_write_text  # noqa: E402


def _abort_if_bot_running() -> None:
    """pgrep for 'python -m python.orchestrator'; abort if found."""
    result = subprocess.run(
        ["pgrep", "-f", "python -m python.orchestrator"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(
            "ERROR: oracle orchestrator is running. Stop it first:\n"
            "  sudo systemctl stop oracle-bot.service\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cash-out revert / reset CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reset-dedup", action="store_true")
    group.add_argument("--revert", action="store_true")
    group.add_argument("--apply", action="store_true")
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--yes-i-am-sure", action="store_true", required=False)
    parser.add_argument("--reason", default="")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()

    if not args.yes_i_am_sure:
        print(
            "ERROR: this script mutates risk state. Re-run with --yes-i-am-sure "
            "after confirming the operator workflow.",
            file=sys.stderr,
        )
        return 3

    _abort_if_bot_running()

    state_dir = Path(args.state_dir)
    if not state_dir.exists():
        print(f"ERROR: state dir {state_dir} not found", file=sys.stderr)
        return 4

    if args.reset_dedup:
        return _reset_dedup(state_dir, args.order_id)
    if args.revert:
        return _revert(state_dir, args.order_id, args.reason)
    if args.apply:
        return _apply(state_dir, args.order_id, args.reason)
    return 1


def _load_state_or_exit(path: Path, label: str, exit_code: int) -> tuple[dict | None, int]:
    """Read + parse a JSON state file. Returns (data, 0) on success; (None, exit_code)
    on missing/unreadable/un-parseable file (after printing a clear error to stderr).

    For HIGH #5: validate ALL state files up-front before any mutation so we never
    leave the on-disk state half-written (e.g. a trade-log row appended without
    the matching risk-state adjustment).
    """
    if not path.exists():
        print(f"ERROR: {label} not found at {path}", file=sys.stderr)
        return None, exit_code
    try:
        text = path.read_text()
    except OSError as e:
        print(f"ERROR: {label} at {path} could not be read: {e}", file=sys.stderr)
        return None, exit_code
    try:
        return json.loads(text), 0
    except json.JSONDecodeError as e:
        print(
            f"ERROR: {label} at {path} is not valid JSON: {e}",
            file=sys.stderr,
        )
        return None, exit_code


def _find_row(
    trade_log: Path,
    *,
    predicate: Callable[[dict], bool],
    last_match: bool = False,
) -> dict | None:
    """Walk trades.jsonl; return first or last matching row.

    Tolerant of torn/blank lines. Returns None if no match.
    """
    if not trade_log.exists():
        return None
    match: dict | None = None
    with trade_log.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if predicate(row):
                if last_match:
                    match = row
                else:
                    return row
    return match


def _find_cash_out_row(trade_log: Path, oid: str) -> dict | None:
    """Latest-wins cash_out row matching order_id==oid (defensive against dups)."""
    return _find_row(
        trade_log,
        predicate=lambda r: r.get("event") == "cash_out" and r.get("order_id") == oid,
        last_match=True,
    )


def _find_existing_revert(trade_log: Path, oid: str) -> dict | None:
    """Existing ``cash_out_reverted`` row whose ``reverts_order_id`` matches ``oid``.
    Used by ``_revert`` to refuse a double-revert (CRITICAL #3).
    """
    return _find_row(
        trade_log,
        predicate=lambda r: r.get("event") == "cash_out_reverted"
                            and r.get("reverts_order_id") == oid,
    )


def _find_existing_force_applied(trade_log: Path, oid: str) -> dict | None:
    """Existing ``cash_out_force_applied`` row whose
    ``metadata.force_applied_for_oid`` matches ``oid``. Used by ``_apply``
    to refuse a double-apply (CRITICAL #3).
    """
    return _find_row(
        trade_log,
        predicate=lambda r: r.get("event") == "cash_out_force_applied"
                            and (r.get("metadata") or {}).get("force_applied_for_oid") == oid,
    )


def _reset_dedup(state_dir: Path, oid: str) -> int:
    """Remove oid from all five _co_*_oids sets in state/fills_reconciler.json.

    Returns 0 on success.

    The five sets correspond to distinct guard failure modes:
      - co_race_lost_oids        (RACE_LOST_TO_SETTLEMENT)
      - co_zero_basis_oids       (ZERO_COST_BASIS_CASH_OUT)
      - co_invalid_partial_oids  (INVALID_PARTIAL_CASH_OUT)
      - co_pnl_cap_oids          (PNL_CAP_EXCEEDED_CASH_OUT)
      - co_reverted_oids         (set by --revert mode)

    All five are cleared because the operator may not know which guard
    fired; safer to reset all and let the next reconciler tick re-classify.
    """
    fills_state_path = state_dir / "fills_reconciler.json"
    data, rc = _load_state_or_exit(fills_state_path, "fills_reconciler.json", 5)
    if data is None:
        return rc
    co_keys = [
        "co_race_lost_oids",
        "co_zero_basis_oids",
        "co_invalid_partial_oids",
        "co_pnl_cap_oids",
        "co_reverted_oids",
    ]
    removed_from = []
    for key in co_keys:
        oids = list(data.get(key, []))
        if oid in oids:
            oids.remove(oid)
            data[key] = oids
            removed_from.append(key)

    atomic_write_text(fills_state_path, json.dumps(data, indent=2))

    if removed_from:
        print(
            f"Reset dedup for oid={oid}: removed from {', '.join(removed_from)}",
            file=sys.stderr,
        )
    else:
        print(
            f"WARN: oid={oid} not present in any of the five _co_*_oids sets",
            file=sys.stderr,
        )
    return 0


def _revert(state_dir: Path, oid: str, reason: str) -> int:
    """Revert an applied cash-out.

    1. Find the cash_out row with order_id==oid in state/trades.jsonl.
    2. Append a cash_out_reverted row.
    3. Adjust state/risk_state.json: subtract realized_pnl from daily_pnl;
       remove matching entry from strategy_trackers["<name>"]["trades"].
    4. Add oid to _co_reverted_oids in state/fills_reconciler.json.
    5. Best-effort Telegram POST.
    """
    import uuid
    from datetime import UTC, datetime

    trade_log = state_dir / "trades.jsonl"
    risk_state_path = state_dir / "risk_state.json"
    fills_state_path = state_dir / "fills_reconciler.json"

    # HIGH #5: validate ALL state files BEFORE any mutation. We're about to
    # append a row to trades.jsonl; if risk_state.json is then unreadable we'd
    # leave half-written state. Pre-validate up front, distinct exit codes.
    if not trade_log.exists():
        print(f"ERROR: trades.jsonl not found at {trade_log}", file=sys.stderr)
        return 13
    risk_state, rc_risk = _load_state_or_exit(risk_state_path, "risk_state.json", 11)
    if risk_state is None:
        return rc_risk
    fills_state, rc_fills = _load_state_or_exit(
        fills_state_path, "fills_reconciler.json", 12
    )
    if fills_state is None:
        return rc_fills

    cash_out_row = _find_cash_out_row(trade_log, oid)
    if cash_out_row is None:
        print(f"ERROR: no cash_out row found for oid={oid}", file=sys.stderr)
        return 5

    # CRITICAL #3: refuse double-revert. If a cash_out_reverted row already
    # references this oid, re-running would append a SECOND revert row AND
    # double-subtract realized_pnl from daily_pnl. Bail with a distinct exit.
    existing_revert = _find_existing_revert(trade_log, oid)
    if existing_revert is not None:
        print(
            f"ERROR: oid={oid} has already been reverted "
            f"(existing revert order_id={existing_revert.get('order_id')}). "
            f"Refusing to double-revert.",
            file=sys.stderr,
        )
        return 9

    realized_pnl = float(cash_out_row["realized_pnl"])
    strategy = cash_out_row.get("metadata", {}).get("strategy", "unknown")

    now_iso = datetime.now(UTC).isoformat()
    reverted_row = {
        "event": "cash_out_reverted",
        "order_id": str(uuid.uuid4()),
        "market_ticker": cash_out_row["market_ticker"],
        "position_key": cash_out_row["position_key"],
        "mode": "live",
        "timestamp": now_iso,
        "applied_at": now_iso,
        "reverts_order_id": oid,
        "closed_side": cash_out_row["closed_side"],
        "hedge_side": cash_out_row["hedge_side"],
        "close_type": cash_out_row["close_type"],
        "contracts_offset": cash_out_row["contracts_offset"],
        "cost_basis_offset_dollars": cash_out_row["cost_basis_offset_dollars"],
        "cost_basis_offset_fees": cash_out_row["cost_basis_offset_fees"],
        "cost_basis_residual_dollars": cash_out_row.get("cost_basis_residual_dollars", 0.0),
        "cost_basis_residual_fees": cash_out_row.get("cost_basis_residual_fees", 0.0),
        "realized_pnl": -realized_pnl,
        "metadata": {
            "operator": os.environ.get("USER", "unknown"),
            "reason": reason,
            "reverts": oid,
        },
    }
    with trade_log.open("a") as f:
        f.write(json.dumps(reverted_row) + "\n")

    # Adjust risk state (file already loaded above for pre-validation).
    risk_state["daily_pnl"] = float(risk_state.get("daily_pnl", 0.0)) - realized_pnl
    trackers = risk_state.get("strategy_trackers", {})
    if strategy in trackers:
        # MEDIUM #7: setdefault so the live list reference is mutated; .get
        # would return a fresh list when the key was absent, making the pop()
        # a no-op against the on-disk state.
        trades = trackers[strategy].setdefault("trades", [])
        won = realized_pnl > 0
        for i in range(len(trades) - 1, -1, -1):
            t = trades[i]
            if abs(float(t.get("pnl", 0)) - realized_pnl) < 1e-6 and bool(t.get("won")) == won:
                trades.pop(i)
                break
    atomic_write_text(risk_state_path, json.dumps(risk_state, indent=2))

    # Add to _co_reverted_oids (fills_state already loaded above).
    reverted = set(fills_state.get("co_reverted_oids", []))
    reverted.add(oid)
    fills_state["co_reverted_oids"] = sorted(reverted)
    atomic_write_text(fills_state_path, json.dumps(fills_state, indent=2))

    print(f"Reverted cash-out oid={oid}: realized_pnl=${realized_pnl:+.2f} undone")
    _telegram_post_best_effort(
        f"Operator revert cash-out: {cash_out_row['market_ticker']} "
        f"pos={cash_out_row['position_key']} oid={oid} reason={reason}. "
        f"Bot must be restarted."
    )
    return 0


def _telegram_post_best_effort(message: str) -> None:
    """Best-effort Telegram POST. Failure does NOT abort the script's
    operator-confirmed mutation; operator can grep state/trades.jsonl
    for the audit row to verify the action.
    """
    try:
        token = os.environ.get("ORACLE_TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("ORACLE_TELEGRAM_OWNER_CHAT_ID")
        if not token or not chat_id:
            print(
                "WARN: Telegram credentials not configured; skipping alert",
                file=sys.stderr,
            )
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        httpx.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        print(
            f"WARN: Telegram POST failed: {e}; mutation completed regardless",
            file=sys.stderr,
        )


def _apply(state_dir: Path, oid: str, reason: str) -> int:
    """Hand-apply a cash-out that the safety knob blocked.

    1. Read the fill from /portfolio/fills (REST call, NOT the running orchestrator's
       in-memory state - the bot is stopped).
    2. Read state/trades.jsonl to find cost basis via cost_basis_for_position.
    3. Compute realized_pnl using _apply_cash_out's formula.
    4. Append a cash_out_force_applied row to trades.jsonl.
    5. Mutate state/risk_state.json: add pnl to daily_pnl and append an entry to
       strategy_trackers[strategy]["trades"].  We deliberately do NOT touch
       ``positions``, ``bankroll``, or ``peak_bankroll`` — those are not in the
       persisted state file and self-heal from Kalshi on next hydrate
       (spec §15, line 1056).
    6. Remove oid from co_pnl_cap_oids in fills_reconciler.json (if present).
    7. Best-effort Telegram POST.
    """
    import uuid
    from datetime import UTC, datetime

    from python.execution.cost_basis import cost_basis_for_position, partial_basis

    trade_log = state_dir / "trades.jsonl"
    risk_state_path = state_dir / "risk_state.json"
    fills_state_path = state_dir / "fills_reconciler.json"

    # HIGH #5: validate ALL state files BEFORE any mutation so we never leave
    # a partial-write footprint (e.g. trade-log append succeeds, risk-state
    # mutation then fails on unreadable file).
    if not trade_log.exists():
        print(f"ERROR: trades.jsonl not found at {trade_log}", file=sys.stderr)
        return 13
    risk_state, rc_risk = _load_state_or_exit(risk_state_path, "risk_state.json", 11)
    if risk_state is None:
        return rc_risk
    fills_state, rc_fills = _load_state_or_exit(
        fills_state_path, "fills_reconciler.json", 12
    )
    if fills_state is None:
        return rc_fills

    # CRITICAL #3: refuse double-apply. If a cash_out_force_applied row whose
    # metadata.force_applied_for_oid matches this oid already exists, re-running
    # would append a SECOND row and double-credit P&L. Bail with a distinct exit.
    existing_force = _find_existing_force_applied(trade_log, oid)
    if existing_force is not None:
        print(
            f"ERROR: oid={oid} has already been force-applied "
            f"(existing force-apply order_id={existing_force.get('order_id')}). "
            f"Refusing to double-apply.",
            file=sys.stderr,
        )
        return 10

    fill = _fetch_fill_for_oid(state_dir, oid)
    if fill is None:
        print(f"ERROR: no fill found for oid={oid} on /portfolio/fills",
              file=sys.stderr)
        return 6

    market_ticker = fill["market_ticker"]
    position_key = market_ticker
    hedge_side = (fill.get("side", "") or "").upper()
    closed_side = "NO" if hedge_side == "YES" else "YES"

    cb = cost_basis_for_position(
        trade_log, position_key=position_key, market_ticker=market_ticker,
    )
    if cb.total_contracts == 0:
        print(f"ERROR: cost basis is zero for {market_ticker}", file=sys.stderr)
        return 7

    contracts_offset = int(float(fill.get("count_fp", 0)))
    if contracts_offset > cb.total_contracts:
        print(
            f"ERROR: contracts_offset {contracts_offset} > "
            f"cb.total_contracts {cb.total_contracts}",
            file=sys.stderr,
        )
        return 8
    pb = partial_basis(cb, contracts_offset)
    price_key = "no_price_dollars" if hedge_side == "NO" else "yes_price_dollars"
    hedge_price = float(fill.get(price_key) or 0)
    hedge_proceeds = contracts_offset * hedge_price
    hedge_fees = float(fill.get("fee_cost", 0))
    realized_pnl = (
        contracts_offset * 1.00
        - pb.offset_dollars - pb.offset_fees
        - hedge_proceeds - hedge_fees
    )
    close_type = "full" if contracts_offset == cb.total_contracts else "partial"

    applied_at = datetime.now(UTC).isoformat()
    cap_value = _read_cap_value(state_dir)
    row = {
        "event": "cash_out_force_applied",
        "order_id": str(uuid.uuid4()),
        "market_ticker": market_ticker,
        "position_key": position_key,
        "side": closed_side,
        "mode": "live",
        "timestamp": fill.get("created_time", ""),
        "applied_at": applied_at,
        "closed_side": closed_side,
        "hedge_side": hedge_side,
        "close_type": close_type,
        "contracts_offset": contracts_offset,
        "contracts_residual": cb.total_contracts - contracts_offset,
        "hedge_price": round(hedge_price, 4),
        "hedge_proceeds": round(hedge_proceeds, 4),
        "hedge_fees": round(hedge_fees, 4),
        "cost_basis_offset_dollars": pb.offset_dollars,
        "cost_basis_offset_fees": pb.offset_fees,
        "cost_basis_residual_dollars": pb.residual_dollars,
        "cost_basis_residual_fees": pb.residual_fees,
        "realized_pnl": round(realized_pnl, 4),
        "metadata": {
            "operator": os.environ.get("USER", "unknown"),
            "reason": reason,
            "bypassed_cap_dollars": cap_value,
            "force_applied_for_oid": oid,
        },
    }
    with trade_log.open("a") as f:
        f.write(json.dumps(row) + "\n")

    # Mutate risk state.  CRITICAL #2: do NOT touch ``positions``, ``bankroll``,
    # or ``peak_bankroll`` — ``serialize_risk_state`` (python/risk/persistence.py
    # §97-107) does not persist those keys; they hydrate from Kalshi on next
    # restart.  Spec §15 (line 1056) explicitly forbids mutating them.  Mutating
    # them either KeyErrors on real production schema (no ``positions`` key) or
    # silently corrupts ``peak_bankroll = max(real_peak, 0 + realized_pnl)``.
    # Only ``daily_pnl`` and ``strategy_trackers`` are persisted side-channels.
    risk_state["daily_pnl"] = float(risk_state.get("daily_pnl", 0.0)) + realized_pnl
    strategy = cb.last_strategy or "unknown"
    trackers = risk_state.setdefault("strategy_trackers", {})
    # MEDIUM #7: setdefault ensures a live reference to the trades list.
    trades_list = trackers.setdefault(strategy, {"trades": []}).setdefault("trades", [])
    trades_list.append({
        "pnl": realized_pnl,
        "won": realized_pnl > 0,
        # CRITICAL #1: ts MUST be an ISO-format string.
        # ``deserialize_risk_state`` (python/risk/persistence.py:134) calls
        # ``datetime.fromisoformat(t["ts"])`` which raises TypeError on a float.
        # The blanket ``except Exception`` (line 156) swallows the error and
        # discards ALL risk state on next restart — a catastrophic silent fail.
        "ts": datetime.now(UTC).isoformat(),
    })
    atomic_write_text(risk_state_path, json.dumps(risk_state, indent=2))

    # Remove from co_pnl_cap_oids (fills_state already loaded above).
    cap_oids = set(fills_state.get("co_pnl_cap_oids", []))
    cap_oids.discard(oid)
    fills_state["co_pnl_cap_oids"] = sorted(cap_oids)
    atomic_write_text(fills_state_path, json.dumps(fills_state, indent=2))

    _telegram_post_best_effort(
        f"Operator force-applied cash-out: {market_ticker} pos={position_key} "
        f"oid={oid} bypassed_cap=${cap_value} realized_pnl=${realized_pnl:+.2f} "
        f"reason={reason}. Bot must be restarted."
    )

    print(f"Force-applied cash-out oid={oid}: realized_pnl=${realized_pnl:+.2f}")
    return 0


def _fetch_fill_for_oid(state_dir: Path, oid: str) -> dict | None:
    """Fetch the fill matching oid from Kalshi /portfolio/fills.

    This is a test seam - production calls the Kalshi REST API; tests
    monkeypatch this function to inject fake fills.

    TODO: wire a real KalshiClient.iter_fills() call here for production
    day-1 of v1.1. For now, operators use the hand-edit fallback documented
    in docs/operations-runbook.md when --apply is required.
    """
    print(
        "ERROR: _fetch_fill_for_oid not implemented for production use. "
        "Use the hand-edit fallback documented in docs/operations-runbook.md, "
        "or monkeypatch this seam in tests.",
        file=sys.stderr,
    )
    return None


def _read_cap_value(state_dir: Path) -> float:
    """Read the cash_out_max_pnl_apply_dollars value from the persisted state.

    Defaults to 500.0 (the config default) if not present.
    """
    fills_state_path = state_dir / "fills_reconciler.json"
    if not fills_state_path.exists():
        return 500.0
    try:
        data = json.loads(fills_state_path.read_text())
        return float(data.get("cash_out_max_pnl_apply_dollars", 500.0))
    except (json.JSONDecodeError, OSError, ValueError):
        return 500.0


if __name__ == "__main__":
    sys.exit(main())
