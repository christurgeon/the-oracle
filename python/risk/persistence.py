"""Risk state persistence for crash recovery.

Provides atomic JSON file persistence for risk manager state that cannot
be reconstructed from the Kalshi API (peak bankroll, daily PnL, circuit
breaker history, trading halt flag).  Also writes a heartbeat file for
external uptime monitoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from python.utils.atomic_write import atomic_write_text as _atomic_write

logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1
STATE_FILENAME = "risk_state.json"
HEARTBEAT_FILENAME = "heartbeat"
EXPOSURE_OVERRIDE_FILENAME = "exposure_override.json"
EXPOSURE_OVERRIDE_SCHEMA_VERSION = 1


@dataclass
class RiskStateSnapshot:
    """Deserialized risk state from disk.

    Mutable: ``peak_bankroll`` is overwritten on mode-switch before hydration
    so the previous mode's peak is not compared against the new mode's bankroll.
    Daily counters (``daily_pnl``, ``daily_start_bankroll``, ``trading_halted``)
    are reset after hydration via ``RiskManager.reset_daily()``, not by mutating
    this object.
    """

    peak_bankroll: float
    daily_pnl: float
    daily_start_bankroll: float
    trading_halted: bool
    current_trading_date: str | None
    execution_mode: str
    strategy_trackers: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: int = STATE_SCHEMA_VERSION
    saved_at: str = ""


def serialize_risk_state(
    *,
    peak_bankroll: float,
    daily_pnl: float,
    daily_start_bankroll: float,
    trading_halted: bool,
    strategy_trackers: dict[str, dict[str, Any]],
    current_trading_date: str | None,
    execution_mode: str,
) -> dict[str, Any]:
    """Convert risk manager export into a JSON-safe dict.

    Timestamps in strategy tracker trades are converted from ``datetime``
    to ISO 8601 strings.  ``last_data_ts`` (monotonic clock) is excluded.
    """
    safe_trackers: dict[str, dict[str, Any]] = {}
    for name, tracker in strategy_trackers.items():
        safe_trades: list[dict[str, Any]] = []
        for t in tracker.get("trades", []):
            ts = t["ts"]
            safe_trades.append({
                "pnl": t["pnl"],
                "won": t["won"],
                "ts": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            })
        safe_trackers[name] = {
            "enabled": tracker.get("enabled", True),
            "disabled_reason": tracker.get("disabled_reason", ""),
            # Persisted so a long-burning failure across restarts can still
            # trip the CRITICAL alert at STRATEGY_ERROR_ALERT_THRESHOLD.
            "consecutive_errors": int(tracker.get("consecutive_errors", 0)),
            "trades": safe_trades,
        }

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "execution_mode": execution_mode,
        "peak_bankroll": peak_bankroll,
        "daily_pnl": daily_pnl,
        "daily_start_bankroll": daily_start_bankroll,
        "trading_halted": trading_halted,
        "current_trading_date": current_trading_date,
        "strategy_trackers": safe_trackers,
    }


def deserialize_risk_state(data: dict[str, Any]) -> RiskStateSnapshot | None:
    """Parse a JSON dict into a ``RiskStateSnapshot``.

    Returns ``None`` on schema version mismatch or any parse error.
    """
    try:
        version = data.get("schema_version")
        if version != STATE_SCHEMA_VERSION:
            logger.warning(
                "Risk state schema version mismatch: expected %d, got %s — starting fresh",
                STATE_SCHEMA_VERSION,
                version,
            )
            return None

        raw_trackers = data.get("strategy_trackers", {})
        trackers: dict[str, dict[str, Any]] = {}
        for name, tracker_data in raw_trackers.items():
            parsed_trades: list[dict[str, Any]] = []
            for t in tracker_data.get("trades", []):
                parsed_trades.append({
                    "pnl": float(t["pnl"]),
                    "won": bool(t["won"]),
                    "ts": datetime.fromisoformat(t["ts"]),
                })
            trackers[name] = {
                "enabled": bool(tracker_data.get("enabled", True)),
                "disabled_reason": str(tracker_data.get("disabled_reason", "")),
                # Defaults to 0 for state files written before this field
                # existed — bidirectionally compatible without a schema bump.
                "consecutive_errors": int(tracker_data.get("consecutive_errors", 0)),
                "trades": parsed_trades,
            }

        return RiskStateSnapshot(
            peak_bankroll=float(data["peak_bankroll"]),
            daily_pnl=float(data["daily_pnl"]),
            daily_start_bankroll=float(data["daily_start_bankroll"]),
            trading_halted=bool(data["trading_halted"]),
            current_trading_date=data.get("current_trading_date"),
            execution_mode=str(data.get("execution_mode", "shadow")),
            strategy_trackers=trackers,
            schema_version=version,
            saved_at=str(data.get("saved_at", "")),
        )
    except Exception:
        logger.warning("Failed to parse risk state file — starting fresh", exc_info=True)
        return None


async def save_state(state_dir: Path, data: dict[str, Any]) -> bool:
    """Persist risk state to ``state/risk_state.json`` atomically.

    Runs file I/O in a thread.  Logs errors but never raises — callers
    can fire-and-forget. Returns ``True`` on success, ``False`` on any
    error so the orchestrator can update its consecutive-failure
    counter and decide whether to alert (B9).
    """
    try:
        path = state_dir / STATE_FILENAME
        content = json.dumps(data, indent=2)
        await asyncio.to_thread(_atomic_write, path, content)
        return True
    except Exception:
        logger.error("Failed to persist risk state", exc_info=True)
        return False


def load_state(state_dir: Path) -> RiskStateSnapshot | None:
    """Load risk state from ``state/risk_state.json``.

    Synchronous — called once at startup.  Returns ``None`` if the file
    is missing, corrupt, or has a schema version mismatch.
    """
    path = state_dir / STATE_FILENAME
    try:
        data = json.loads(path.read_text())
        return deserialize_risk_state(data)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Failed to read risk state file %s — starting fresh", path, exc_info=True)
        return None


async def write_heartbeat(state_dir: Path) -> None:
    """Write current timestamp to ``state/heartbeat`` for monitoring.

    Uses a plain write (not atomic) — partial reads of a float timestamp
    are harmless for monitoring purposes.  Runs the synchronous I/O in a
    thread to avoid blocking the event loop.
    """
    try:
        path = state_dir / HEARTBEAT_FILENAME
        await asyncio.to_thread(path.write_text, str(time.time()))
    except Exception:
        logger.error("Failed to write heartbeat", exc_info=True)


def save_exposure_override(state_dir: Path, fraction: float | None) -> None:
    """Persist (or clear) the runtime exposure override.

    Synchronous; called from the Telegram command handler which is already
    off the trading hot path.  The write is atomic.
    ``fraction=None`` removes the file (revert to config baseline).
    """
    path = state_dir / EXPOSURE_OVERRIDE_FILENAME
    if fraction is None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove exposure override file", exc_info=True)
        return
    payload = {
        "schema_version": EXPOSURE_OVERRIDE_SCHEMA_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "max_total_exposure_pct": float(fraction),
    }
    try:
        _atomic_write(path, json.dumps(payload, indent=2))
    except Exception:
        logger.error("Failed to persist exposure override", exc_info=True)


def load_exposure_override(state_dir: Path) -> float | None:
    """Load the persisted exposure override fraction if any.

    Returns ``None`` when the file is missing, unreadable, schema mismatched,
    or out of bounds.  Out-of-bounds values are logged and ignored so a
    corrupted file cannot deploy an unsafe cap.
    """
    path = state_dir / EXPOSURE_OVERRIDE_FILENAME
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning(
            "Failed to read exposure override file %s — ignoring", path, exc_info=True
        )
        return None

    if not isinstance(data, dict):
        logger.warning("exposure override file has wrong shape — ignoring")
        return None
    if data.get("schema_version") != EXPOSURE_OVERRIDE_SCHEMA_VERSION:
        logger.warning(
            "exposure override schema mismatch (got %s) — ignoring",
            data.get("schema_version"),
        )
        return None
    try:
        fraction = float(data["max_total_exposure_pct"])
    except (KeyError, TypeError, ValueError):
        logger.warning("exposure override missing/invalid value — ignoring")
        return None
    if not (0.0 < fraction <= 1.0):
        logger.warning(
            "exposure override %.4f out of (0.0, 1.0] — ignoring", fraction
        )
        return None
    return fraction
