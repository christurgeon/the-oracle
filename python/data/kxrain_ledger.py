"""Durable virtual-entry ledger for the KXRAIN shadow observer.

Nothing here raises. Keyed on `corpus_key` = `(station, settlement_date)`,
never the ticker (Kalshi renames it) or a JSONL replay (rotation is boundary-unsafe).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KEEP_DATES",
    "atomic_write_json",
    "LEDGER_FILENAME",
    "LEDGER_SCHEMA_VERSION",
    "CorpusKey",
    "KxRainLedger",
    "LedgerEntry",
    "decode_corpus_key",
    "encode_corpus_key",
]

LEDGER_SCHEMA_VERSION = 1
LEDGER_FILENAME = "kxrain_ledger.json"

# Three settlement dates covers yesterday, today and tomorrow's early-opening
# market-day with margin, which is all the "already entered?" question needs.
DEFAULT_KEEP_DATES = 3

CorpusKey = tuple[str, date]

_KEY_SEPARATOR = "|"


def encode_corpus_key(key: CorpusKey) -> str:
    """``("MSY", date(2026, 9, 19))`` -> ``"MSY|2026-09-19"``."""
    station, settlement_date = key
    return f"{station}{_KEY_SEPARATOR}{settlement_date.isoformat()}"


def decode_corpus_key(raw: str) -> CorpusKey | None:
    """Inverse of :func:`encode_corpus_key`. Returns ``None`` on anything else."""
    if not raw or raw.count(_KEY_SEPARATOR) != 1:
        return None
    station, _, date_str = raw.partition(_KEY_SEPARATOR)
    if not station:
        return None
    try:
        return (station, date.fromisoformat(date_str))
    except ValueError:
        return None


@dataclass(frozen=True)
class LedgerEntry:
    """The audit record of one consumed virtual entry."""

    decision_id: str
    ts: str
    assumed_entry_price: float | None
    scan_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "ts": self.ts,
            "assumed_entry_price": self.assumed_entry_price,
            "scan_id": self.scan_id,
        }

    @classmethod
    def from_dict(cls, data: Any) -> LedgerEntry | None:
        """Parse one on-disk entry, or ``None`` if it is not one."""
        if not isinstance(data, dict):
            return None
        price = data.get("assumed_entry_price")
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                return None
        return cls(
            decision_id=str(data.get("decision_id", "")),
            ts=str(data.get("ts", "")),
            assumed_entry_price=price,
            scan_id=str(data.get("scan_id", "")),
        )


class KxRainLedger:
    """Which `(station, settlement_date)` market-days have consumed an entry.

    Deliberately synchronous, blocking I/O — the async caller must offload
    (`asyncio.to_thread`) or it stalls the event loop.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._entries: dict[CorpusKey, LedgerEntry] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> KxRainLedger:
        """Read the ledger from disk. Fail-open — never raises; a bad file just starts empty."""
        ledger = cls(path)
        ledger._entries = _read_entries(ledger._path)
        return ledger

    # -- reads -------------------------------------------------------------

    def already_entered(self, key: CorpusKey) -> bool:
        return key in self._entries

    def entry(self, key: CorpusKey) -> LedgerEntry | None:
        return self._entries.get(key)

    def keys(self) -> list[CorpusKey]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_schema_version": LEDGER_SCHEMA_VERSION,
            "entries": {
                encode_corpus_key(key): entry.to_dict()
                for key, entry in sorted(self._entries.items())
            },
        }

    # -- writes ------------------------------------------------------------

    def record_entry(
        self,
        key: CorpusKey,
        *,
        decision_id: str,
        ts: str,
        assumed_entry_price: float | None,
        scan_id: str,
        prune_after: bool = True,
    ) -> bool:
        """Consume the virtual entry for one market-day and persist immediately.

        Persists here rather than at end-of-tick to close the mid-tick-crash
        hole. Returns `True` only on a genuinely new, successfully saved entry;
        a duplicate key or a failed write both return `False` and keep the
        original in-memory entry.
        """
        existing = self._entries.get(key)
        if existing is not None:
            logger.warning(
                "kxrain ledger: %s already entered (%s); keeping the first entry, "
                "ignoring %s",
                encode_corpus_key(key),
                existing.decision_id,
                decision_id,
            )
            return False

        self._entries[key] = LedgerEntry(
            decision_id=decision_id,
            ts=ts,
            assumed_entry_price=assumed_entry_price,
            scan_id=scan_id,
        )
        if prune_after:
            self.prune(protect=key)
        if key not in self._entries:  # pragma: no cover - prune(protect=) guarantees it
            logger.error(
                "kxrain ledger: %s was evicted by the same call that recorded it; "
                "refusing to report success",
                encode_corpus_key(key),
            )
            return False
        return self.save_atomic()

    def prune(
        self, keep_dates: int = DEFAULT_KEEP_DATES, *, protect: CorpusKey | None = None
    ) -> int:
        """Keep the `keep_dates` most recent settlement dates; drop the rest.

        Returns the count dropped. Ranked over dates the ledger holds, never a
        clock. `protect` is never evicted.
        """
        keep = max(0, int(keep_dates))
        dates = sorted({key[1] for key in self._entries}, reverse=True)
        if len(dates) <= keep:
            return 0
        kept = set(dates[:keep])
        if protect is not None:
            kept.add(protect[1])
        before = len(self._entries)
        self._entries = {key: e for key, e in self._entries.items() if key[1] in kept}
        return before - len(self._entries)

    def save_atomic(self, path: Path | str | None = None) -> bool:
        """Write the ledger via tempfile + `os.replace`. Never raises (swallows `OSError`)."""
        target = Path(path) if path is not None else self._path
        return atomic_write_json(target, self.to_dict())


def atomic_write_json(target: Path, payload: Any) -> bool:
    """Write `payload` as JSON via tempfile + `os.replace`. Never raises (swallows `OSError`)."""
    content = json.dumps(payload, indent=2, sort_keys=True)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    except OSError:
        logger.error("kxrain: cannot prepare %s for writing", target, exc_info=True)
        return False

    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp, target)
        return True
    except OSError:
        logger.error("kxrain: atomic write to %s failed", target, exc_info=True)
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _read_entries(path: Path) -> dict[CorpusKey, LedgerEntry]:
    try:
        if not path.is_file():
            logger.info("kxrain ledger: no file at %s — starting empty", path)
            return {}
        raw = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("kxrain ledger: cannot read %s — starting empty", path, exc_info=True)
        return {}

    if not raw.strip():
        logger.warning("kxrain ledger: %s is empty — starting empty", path)
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("kxrain ledger: %s is not valid JSON — starting empty", path)
        return {}

    if not isinstance(data, dict):
        logger.warning("kxrain ledger: %s is not an object — starting empty", path)
        return {}

    version = data.get("_schema_version")
    if version != LEDGER_SCHEMA_VERSION:
        logger.warning(
            "kxrain ledger: %s has schema version %s, expected %d — starting empty",
            path,
            version,
            LEDGER_SCHEMA_VERSION,
        )
        return {}

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, dict):
        logger.warning("kxrain ledger: %s has no entries object — starting empty", path)
        return {}

    entries: dict[CorpusKey, LedgerEntry] = {}
    rejected = 0
    for raw_key, raw_entry in raw_entries.items():
        key = decode_corpus_key(str(raw_key))
        entry = LedgerEntry.from_dict(raw_entry)
        if key is None or entry is None:
            rejected += 1
            continue
        entries[key] = entry

    if rejected:
        # Partial loss, not total: a single bad row must not discard the
        # market-days that are correctly recorded alongside it.
        logger.warning(
            "kxrain ledger: dropped %d unreadable entr%s from %s",
            rejected,
            "y" if rejected == 1 else "ies",
            path,
        )
    return entries
