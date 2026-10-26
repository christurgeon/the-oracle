"""Per-day event cap for weather-market strategies (weather_edge and favorite_bias).

Caps weather signals to at most ``max_bets_per_day_per_event`` per
``(city, var, forecast_date)`` per UTC day, persisting across scans via
``state/trades.jsonl`` hydration.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from python.utils.jsonl import iter_jsonl
from python.utils.ticker_dates import SERIES_PREFIX_TO_CITY_VAR as _SERIES_CITY

logger = logging.getLogger(__name__)

EventKey = tuple[str, str, str]  # (city, var, forecast_date)
_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
_DATE_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})")


def _parse_event_key_from_ticker(ticker: str) -> EventKey | None:
    if not ticker:
        return None
    parts = ticker.split("-")
    if len(parts) < 3 or parts[0] not in _SERIES_CITY:
        return None
    m = _DATE_RE.match(parts[1])
    if not m:
        return None
    yy, mon, dd = m.groups()
    if mon not in _MONTHS:
        return None
    city, var = _SERIES_CITY[parts[0]]
    return (city, var, f"20{yy}-{_MONTHS[mon]:02d}-{int(dd):02d}")


def _event_key_from_row(row: dict[str, Any]) -> EventKey | None:
    meta = row.get("metadata") or {}
    city = meta.get("city")
    var = meta.get("market_type")
    fdate = meta.get("forecast_date")
    if city and var and fdate:
        return (str(city), str(var), str(fdate))
    return _parse_event_key_from_ticker(str(row.get("market_ticker") or ""))


def hydrate_per_day_state_from_trades_jsonl(
    *,
    utc_today: str,
    trades_path: Path,
    relevant_modes: frozenset[str] = frozenset({"live", "paper"}),
    relevant_strategy: str = "weather_edge",
) -> tuple[dict[EventKey, set[str]], dict[EventKey, dict[str, Any]]]:
    """Return (committed_order_ids, first_commit) for events committed today.

    Cap state is computed as ``(placement ∪ fill) − cancelled-no-fill`` per
    event key — a placement that was cancelled with zero fills releases the
    slot; a partial-fill cancel (cancel row present alongside a fill row for
    the same order_id) does NOT release.

    ``first_commit[key]`` carries the earliest placement/fill/cancel row's
    ticker and timestamp for audit ``prior_commitment`` enrichment.
    """
    from python.execution.trade_log import is_known_event

    placements_by_key: dict[EventKey, set[str]] = {}
    fills_by_key: dict[EventKey, set[str]] = {}
    cancels_by_key: dict[EventKey, set[str]] = {}
    first_commit: dict[EventKey, dict[str, Any]] = {}

    relevant_event_types = {"placement", "fill", "cancel"}

    for row in iter_jsonl(trades_path):
        if not is_known_event(row):
            continue
        event = row.get("event")
        if event not in relevant_event_types:
            continue
        if row.get("mode") not in relevant_modes:
            continue
        if row.get("strategy") != relevant_strategy:
            continue
        ts = str(row.get("timestamp") or "")
        if not ts.startswith(utc_today):
            continue
        order_id = row.get("order_id")
        if not isinstance(order_id, str):
            continue
        key = _event_key_from_row(row)
        if key is None:
            logger.warning("per_day_cap hydration: could not derive event key for row %s", order_id)
            continue
        if event == "placement":
            placements_by_key.setdefault(key, set()).add(order_id)
        elif event == "fill":
            fills_by_key.setdefault(key, set()).add(order_id)
        elif event == "cancel":
            cancels_by_key.setdefault(key, set()).add(order_id)
            # Cancels must not seed `first_commit`: block-audit consumers read it
            # as the order that held the slot. They are subtracted below instead.
            continue
        first_commit.setdefault(
            key,
            {
                "source": str(event),
                "ticker": row.get("market_ticker"),
                "ts": ts,
                "order_id": order_id,
            },
        )

    committed: dict[EventKey, set[str]] = {}
    for key in placements_by_key.keys() | fills_by_key.keys():
        placement_ids = placements_by_key.get(key, set())
        fill_ids = fills_by_key.get(key, set())
        cancel_ids = cancels_by_key.get(key, set())
        cancelled_no_fill = cancel_ids - fill_ids
        live = (placement_ids | fill_ids) - cancelled_no_fill
        if live:
            committed[key] = live
        else:
            first_commit.pop(key, None)

    return committed, first_commit


def _event_key_from_signal(sig: Any) -> EventKey:
    meta = sig.metadata or {}
    return (
        str(meta.get("city", "")),
        str(meta.get("market_type", "")),
        str(meta.get("forecast_date", "")),
    )


def _block_audit_row(
    sig: Any,
    key: EventKey,
    scan_id: str,
    scan_ts: str,
    blocked_ts: str,
    *,
    committed_before: int,
    max_per_day: int,
    sub_reason: str,
    prior_commitment: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = sig.metadata or {}
    return {
        "schema_version": 1,
        "scan_id": scan_id,
        "scan_ts": scan_ts,
        "blocked_ts": blocked_ts,
        "event_key": {"city": key[0], "var": key[1], "forecast_date": key[2]},
        "max_per_day": max_per_day,
        "committed_before": committed_before,
        "sub_reason": sub_reason,
        "reason": "PER_DAY_EVENT_CAP_HIT",
        "blocked_signal": {
            "market_ticker": sig.market_ticker,
            "side": meta.get("side"),
            "strike_type": meta.get("strike_type"),
            "floor_strike": meta.get("floor_strike"),
            "cap_strike": meta.get("cap_strike"),
            "model_probability": meta.get("model_probability"),
            "market_price": meta.get("market_price"),
            "exec_price": meta.get("exec_price"),
            "effective_edge": getattr(sig, "effective_edge", None),
            "raw_edge": meta.get("raw_edge"),
            "ensemble_confidence": meta.get("ensemble_confidence"),
            "ensemble_std_dev": meta.get("ensemble_std_dev"),
            "ensemble_std_pooled": meta.get("ensemble_std_pooled"),
            "live_calibrator_kind": meta.get("live_calibrator_kind"),
            "raw_ensemble_prob": meta.get("raw_ensemble_prob"),
            "isotonic_calibrated": meta.get("isotonic_calibrated"),
            "emos_prob_raw": meta.get("emos_prob_raw"),
            "suggested_contracts": getattr(sig, "suggested_contracts", None),
            "suggested_dollars": getattr(sig, "suggested_dollars", None),
            "kelly_raw": meta.get("kelly_raw"),
            "kelly_capped": meta.get("kelly_capped"),
        },
        "prior_commitment": prior_commitment,
    }


class PerDayEventCap:
    """Cap weather signals to ``max_per_day`` bets per
    ``(city, var, forecast_date)`` per UTC day.
    """

    def __init__(
        self,
        *,
        max_per_day: int | None,
        audit_writer: Any,
        now_fn: Callable[[], str] | None = None,
    ) -> None:
        if max_per_day is not None and max_per_day < 1:
            raise ValueError("max_per_day must be >= 1 or None")
        self._max = max_per_day
        self._audit_writer = audit_writer
        self._committed_order_ids: dict[EventKey, set[str]] = {}
        self._order_id_to_key: dict[str, EventKey] = {}
        self._first_commit: dict[EventKey, dict[str, Any]] = {}
        self._last_reset_utc_date: str | None = None
        if now_fn is None:
            from datetime import UTC, datetime

            self._now_fn: Callable[[], str] = lambda: datetime.now(UTC).isoformat()
        else:
            self._now_fn = now_fn

    def maybe_reset(
        self,
        utc_today: str,
        hydrate_fn: Callable[
            [str], tuple[dict[EventKey, set[str]], dict[EventKey, dict[str, Any]]]
        ],
    ) -> None:
        if utc_today == self._last_reset_utc_date:
            return
        if self._max is None:
            self._committed_order_ids, self._first_commit = {}, {}
        else:
            try:
                self._committed_order_ids, self._first_commit = hydrate_fn(utc_today)
            except Exception:
                logger.warning(
                    "PerDayEventCap.maybe_reset: hydration failed for %s; "
                    "falling back to empty state (fail-safe: no-cap)",
                    utc_today,
                    exc_info=True,
                )
                self._committed_order_ids, self._first_commit = {}, {}
        self._order_id_to_key = {
            oid: key for key, oids in self._committed_order_ids.items() for oid in oids
        }
        self._last_reset_utc_date = utc_today

    def filter_and_select(
        self,
        signals: list[Any],
        scan_id: str,
        scan_ts: str,
    ) -> tuple[list[Any], int]:
        if self._max is None or not signals:
            return signals, 0

        groups: dict[EventKey, list[Any]] = {}
        for sig in signals:
            groups.setdefault(_event_key_from_signal(sig), []).append(sig)

        kept: list[Any] = []
        blocked: list[tuple[Any, EventKey, str]] = []
        for key, group in groups.items():
            committed = len(self._committed_order_ids.get(key, ()))
            remaining = max(0, self._max - committed)
            if remaining == 0:
                for sig in group:
                    blocked.append((sig, key, "exhausted_prior_to_scan"))
                continue
            ranked = sorted(
                group,
                key=lambda s: (-s.effective_edge, -s.confidence, s.market_ticker),
            )
            keepers = ranked[:remaining]
            droppers = ranked[remaining:]
            kept.extend(keepers)
            for sig in droppers:
                blocked.append((sig, key, "within_scan_excess"))
            if droppers:
                top = keepers[0]
                top.metadata["event_candidates_count"] = len(group)
                top.metadata["event_dropped_edges"] = [s.effective_edge for s in droppers]
                top.metadata["event_dropped_tickers"] = [s.market_ticker for s in droppers]

        if blocked and self._audit_writer is not None:
            blocked_ts = self._now_fn()
            for sig, key, sub_reason in blocked:
                self._audit_writer.write(
                    _block_audit_row(
                        sig,
                        key,
                        scan_id,
                        scan_ts,
                        blocked_ts,
                        committed_before=len(self._committed_order_ids.get(key, ())),
                        max_per_day=self._max,
                        sub_reason=sub_reason,
                        prior_commitment=self._first_commit.get(key),
                    )
                )
        return kept, len(blocked)

    def commit(self, key: EventKey, order_id: str) -> None:
        """Mark ``order_id`` as committed for ``key``.

        Idempotent on the same ``(key, order_id)`` pair. Re-committing the
        same ``order_id`` under a different key raises ``ValueError`` — an
        order_id is uniquely bound to one event for the duration of its
        UTC day. Called by the orchestrator post-``execute_batch`` for each
        successfully placed order whose strategy owns a ``_per_day_cap``
        (today: weather_edge AND favorite_bias; the orchestrator fans the
        commit out per-strategy via a ``strategy_name`` match).
        """
        if self._max is None:
            return
        existing_key = self._order_id_to_key.get(order_id)
        if existing_key is not None:
            if existing_key == key:
                return
            raise ValueError(
                f"order_id {order_id} already committed to {existing_key}; "
                f"refusing rebind to {key}"
            )
        self._committed_order_ids.setdefault(key, set()).add(order_id)
        self._order_id_to_key[order_id] = key
        self._first_commit.setdefault(
            key,
            {
                "source": "commit",
                "ticker": None,
                "ts": None,
                "order_id": order_id,
            },
        )

    def release(self, order_id: str) -> EventKey | None:
        """Release the cap slot for a previously committed ``order_id``.

        Returns the event key that was released, or ``None`` if the
        ``order_id`` is unknown (already released, never committed, or
        committed before this code shipped). Idempotent. Called by the
        engine's on_cancel callback when ``fill_count == 0`` at terminal
        status (CANCELLED/EXPIRED).
        """
        if self._max is None:
            return None
        key = self._order_id_to_key.pop(order_id, None)
        if key is None:
            return None
        bucket = self._committed_order_ids.get(key)
        if bucket is not None:
            bucket.discard(order_id)
            if not bucket:
                del self._committed_order_ids[key]
                # A stale anchor would make PER_DAY_EVENT_CAP_HIT rows carry the
                # cancelled order's `prior_commitment`.
                self._first_commit.pop(key, None)
        return key
