"""Strategy attribution resolver for trade-log rows.

Engine-path fills carry ``strategy`` at the top level. Reconciler-synthesized
fills carry it only under ``metadata.strategy``, and when the placement scan
fails after a restart that field is ``"unknown"`` (``fills_reconciler.
_build_placement_index`` reads ``metadata.strategy`` and cannot match engine
placement rows).

``build_strategy_index`` joins all rows by ``order_id`` and prefers any
top-level ``strategy`` over ``metadata.strategy``, recovering the true strategy
for reconciler fills whose order_id has a matching engine placement row. The
reconciler itself is deliberately left untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _top_level_strategy(row: dict[str, Any]) -> str | None:
    val = row.get("strategy")
    if isinstance(val, str) and val:
        return val
    return None


def _metadata_strategy(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        val = meta.get("strategy")
        if isinstance(val, str) and val:
            return val
    return None


def build_strategy_index(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Build an ``order_id -> strategy`` map from placement + fill rows.

    Priority per order_id:
      1. top-level ``strategy`` on any row (placement or engine fill) — wins;
      2. ``metadata.strategy`` (reconciler synthetics);
      3. absent from index — caller uses its fallback.

    Row order does not matter: a top-level value always overrides a
    metadata-only value previously recorded for the same order_id.
    """
    index: dict[str, str] = {}
    metadata_only: dict[str, str] = {}

    for row in rows:
        oid = row.get("order_id")
        if not isinstance(oid, str) or not oid:
            continue

        top = _top_level_strategy(row)
        if top is not None:
            index[oid] = top
            continue

        if oid not in index:
            meta = _metadata_strategy(row)
            if meta is not None:
                metadata_only.setdefault(oid, meta)

    # Metadata-only resolutions for order_ids that never had a top-level value.
    for oid, strategy in metadata_only.items():
        index.setdefault(oid, strategy)

    return index


def resolve_strategy_by_order_id(
    order_id: str,
    index: dict[str, str],
    *,
    fallback: str = "unknown",
) -> str:
    """Look up ``order_id`` in ``index``, returning ``fallback`` when absent."""
    return index.get(order_id, fallback)


def _market_key(row: dict[str, Any]) -> str | None:
    """Resolve a position/market key from any trade-log row shape.

    Engine rows carry ``market_id``/``position_key``; reconciler-synthetic
    fills carry only ``market_ticker``. All three are equal by design
    (market_id == position_key == ticker for every live strategy).
    """
    for field in ("market_id", "position_key", "market_ticker"):
        val = row.get(field)
        if isinstance(val, str) and val:
            return val
    return None


def build_market_strategy_index(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Build a ``market_id -> strategy`` map from fill + placement rows.

    Used at hydrate to re-attribute strategy-anonymous Kalshi positions to the
    pod that opened them. Resolution is a deterministic total order per market:

      1. a ``placement`` row beats a ``fill`` row;
      2. then the more recent ``timestamp`` wins (``""`` treated as oldest);
      3. then top-level ``strategy`` beats ``metadata.strategy``.

    Rows that are not fills/placements, or that resolve to neither a market key
    nor a strategy, are skipped (their markets land in the caller's "unknown"
    bucket).
    """
    # Per market: (is_placement, timestamp, is_top_level) ranking tuple.
    best_rank: dict[str, tuple[int, str, int]] = {}
    index: dict[str, str] = {}

    for row in rows:
        event = row.get("event")
        if event not in ("fill", "placement"):
            continue
        key = _market_key(row)
        if key is None:
            continue
        top = _top_level_strategy(row)
        strategy = top if top is not None else _metadata_strategy(row)
        if strategy is None:
            continue

        ts = row.get("timestamp")
        rank = (
            1 if event == "placement" else 0,
            ts if isinstance(ts, str) else "",
            1 if top is not None else 0,
        )
        if key not in best_rank or rank > best_rank[key]:
            best_rank[key] = rank
            index[key] = strategy

    return index
