"""Settlement sweep for the KXRAIN forward shadow corpus.

The corpus's only disaster-recovery path (~28 days of reach, see
`SETTLEMENT_SWEEP_MAX_PAGES`). Nothing here raises; idempotent on `corpus_key`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from python.data.kalshi_client import paginate_detailed
from python.data.kxrain_ledger import CorpusKey, decode_corpus_key, encode_corpus_key
from python.data.kxrain_ledger import atomic_write_json as _atomic_write_json
from python.data.kxrain_markets import KXRAIN_SERIES_TICKER, parse_kxrain_ticker
from python.data.kxrain_rows import build_settlement_row, encode_row
from python.utils.jsonl import ForecastsRawWriter

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_KEEP_SETTLEMENT_DATES",
    "KXRAIN_SETTLEMENTS_FILENAME",
    "MAX_UNPARSEABLE_TICKERS",
    "SETTLED_INDEX_FILENAME",
    "SETTLED_INDEX_SCHEMA_VERSION",
    "SETTLEMENT_PAGE_LIMIT",
    "SETTLEMENT_SWEEP_MAX_PAGES",
    "KxRainSettlementSweep",
    "SettledIndex",
    "SettlementSweepResult",
    "SettlementsDataClient",
]

KXRAIN_SETTLEMENTS_FILENAME = "kxrain_settlements.jsonl"
SETTLED_INDEX_FILENAME = "kxrain_settled_index.json"
SETTLED_INDEX_SCHEMA_VERSION = 1

# At 100/page and 28 market-days/day, 8 pages is roughly 28 days of reach. The cap is
# fixed so the daily request count stays bounded against the shared 429 window.
SETTLEMENT_SWEEP_MAX_PAGES = 8
SETTLEMENT_PAGE_LIMIT = 100

# Must cover as long as the corpus is retained, or two callers (the daily
# sweep and a backfill) can prune out each other's marks.
DEFAULT_KEEP_SETTLEMENT_DATES = 240

MAX_UNPARSEABLE_TICKERS = 5


class _BackoffActiveError(Exception):
    """The shared 429 gate went active mid-walk; stop rather than deepen it."""

    def __init__(self, remaining: float) -> None:
        super().__init__(f"Kalshi backoff active for {remaining:.1f}s")
        self.remaining = remaining


class SettlementsDataClient(Protocol):
    """The one read-only method this module may call."""

    async def get_markets(
        self,
        *,
        status: str | None = ...,
        series_ticker: str | None = ...,
        event_ticker: str | None = ...,
        cursor: str | None = ...,
        limit: int = ...,
    ) -> dict[str, Any]: ...


@dataclass
class SettlementSweepResult:
    """What one sweep did. Every refusal is a count, never a silent drop."""

    n_returned: int = 0
    n_written: int = 0
    n_persisted: int = 0
    n_already_present: int = 0
    n_refused_unsettled: int = 0
    n_refused_not_finalized: int = 0
    n_refused_oversize: int = 0
    n_unparseable_ticker: int = 0
    n_rule_inconsistent: int = 0
    unparseable_tickers: list[str] = field(default_factory=list)
    pages_consumed: int = 0
    truncated: bool = False
    cycle_aborted: bool = False
    backoff_aborted: bool = False
    error: str | None = None
    index_saved: bool = True

    @property
    def anomaly(self) -> bool:
        """True when this sweep looks healthy but silently is not.

        E.g. an empty listing, or rows built but not persisted.
        """
        return (
            self.error is not None
            or self.n_returned == 0
            or self.n_persisted < self.n_written
            or self.cycle_aborted
            or self.backoff_aborted
            or not self.index_saved
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_returned": self.n_returned,
            "n_written": self.n_written,
            "n_persisted": self.n_persisted,
            "n_already_present": self.n_already_present,
            "n_refused_unsettled": self.n_refused_unsettled,
            "n_refused_not_finalized": self.n_refused_not_finalized,
            "n_refused_oversize": self.n_refused_oversize,
            "n_unparseable_ticker": self.n_unparseable_ticker,
            "n_rule_inconsistent": self.n_rule_inconsistent,
            "unparseable_tickers": self.unparseable_tickers,
            "pages_consumed": self.pages_consumed,
            "truncated": self.truncated,
            "cycle_aborted": self.cycle_aborted,
            "backoff_aborted": self.backoff_aborted,
            "index_saved": self.index_saved,
            "anomaly": self.anomaly,
            "error": self.error,
        }


class SettledIndex:
    """Which `(station, settlement_date)` market-days already have a row.

    A durable artifact, not a replay of the rotated JSONL (rotation is
    boundary-unsafe). Synchronous, like `KxRainLedger` — an async caller offloads.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._entries: dict[CorpusKey, str] = {}

    @classmethod
    def load(cls, path: Path | str) -> SettledIndex:
        """Read the index. Fail-open: a missing or corrupt file starts empty and warns."""
        index = cls(path)
        target = Path(path)
        if not target.is_file():
            return index
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("kxrain settled index: %s is unreadable; starting empty", target)
            return index
        if not isinstance(payload, dict):
            logger.warning("kxrain settled index: %s is not an object; starting empty", target)
            return index
        version = payload.get("_schema_version")
        if version != SETTLED_INDEX_SCHEMA_VERSION:
            logger.warning(
                "kxrain settled index: schema %r != %d; starting empty",
                version,
                SETTLED_INDEX_SCHEMA_VERSION,
            )
            return index
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            return index
        for raw_key, ts in raw_entries.items():
            key = decode_corpus_key(str(raw_key))
            if key is None:
                logger.warning("kxrain settled index: dropping unparseable key %r", raw_key)
                continue
            index._entries[key] = str(ts)
        return index

    def contains(self, key: CorpusKey) -> bool:
        return key in self._entries

    def mark(self, key: CorpusKey, *, ts: str) -> None:
        self._entries[key] = ts

    def keys(self) -> list[CorpusKey]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "_schema_version": SETTLED_INDEX_SCHEMA_VERSION,
            "entries": {encode_corpus_key(k): v for k, v in sorted(self._entries.items())},
        }

    def prune(self, keep_dates: int = DEFAULT_KEEP_SETTLEMENT_DATES) -> int:
        """Keep the `keep_dates` most recent settlement dates.

        Ranked over its own dates, never a clock.
        """
        keep = max(0, int(keep_dates))
        dates = sorted({key[1] for key in self._entries}, reverse=True)
        if len(dates) <= keep:
            return 0
        kept = set(dates[:keep])
        before = len(self._entries)
        self._entries = {k: v for k, v in self._entries.items() if k[1] in kept}
        return before - len(self._entries)

    def merge_from_disk(self) -> int:
        """Re-read the file and add any keys written since this copy loaded.

        Two callers share this index.
        """
        fresh = SettledIndex.load(self._path)
        added = 0
        for key, ts in fresh._entries.items():
            if key not in self._entries:
                self._entries[key] = ts
                added += 1
        return added

    def save_atomic(
        self, path: Path | str | None = None, *, keep_dates: int | None = None
    ) -> bool:
        """Merge, then prune, then tempfile + `os.replace`.

        Pruning before the merge would be undone by it.
        """
        target = Path(path) if path is not None else self._path
        if target == self._path:
            self.merge_from_disk()
        if keep_dates is not None:
            self.prune(keep_dates)
        return _atomic_write_json(target, self.to_dict())


class KxRainSettlementSweep:
    """Fetch settled KXRAIN markets and append the ones not yet recorded."""

    def __init__(
        self,
        client: SettlementsDataClient,
        *,
        log_dir: Path | str,
        clock: Callable[[], datetime] | None = None,
        max_pages: int = SETTLEMENT_SWEEP_MAX_PAGES,
        page_delay: float = 1.0,
        series_ticker: str = KXRAIN_SERIES_TICKER,
        retention_days: int = 240,
        keep_dates: int = DEFAULT_KEEP_SETTLEMENT_DATES,
        executor: Executor | None = None,
    ) -> None:
        self._client = client
        # A private worker keeps this off the default pool, whose saturation can stall DNS.
        self._executor = executor
        self._dir = Path(log_dir)
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._max_pages = max(1, int(max_pages))
        self._page_delay = max(0.0, float(page_delay))
        self._series_ticker = series_ticker
        self._keep_dates = keep_dates
        self.writer = ForecastsRawWriter(
            self._dir / KXRAIN_SETTLEMENTS_FILENAME,
            retention_days=retention_days,
            clock=self._clock,
        )
        self._index_path = self._dir / SETTLED_INDEX_FILENAME

    async def _offload(self, fn: Any, *args: Any) -> Any:
        if self._executor is None:
            return await asyncio.to_thread(fn, *args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    def _backoff_remaining(self) -> float:
        # The shared 429 gate — read only through the client's public accessor.
        accessor = getattr(self._client, "backoff_remaining_seconds", None)
        if not callable(accessor):
            return 0.0
        try:
            return max(0.0, float(accessor()))
        except Exception:
            return 0.0

    async def run_once(self) -> SettlementSweepResult:
        """One sweep. Never raises; the error lands on the result."""
        result = SettlementSweepResult()
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        index = await self._offload(SettledIndex.load, self._index_path)
        markets = await self._fetch_settled(result)

        rows, marked = self._build_rows(markets, index, now, result)
        if rows:
            await self._offload(self._persist, rows, marked, index, now, result)
        if result.anomaly:
            logger.error(
                "kxrain settlement sweep: ANOMALOUS result %s — the corpus's only "
                "disaster-recovery path may be dead, and its reach is ~28 days",
                result.to_dict(),
            )
        return result

    async def _fetch_settled(self, result: SettlementSweepResult) -> list[dict[str, Any]]:
        # paginate_detailed discards every page if one raises, so seen_pages
        # accumulates independently for the error path; the happy path uses
        # page.rows, which correctly discards a cycling page's duplicates.
        seen_pages: list[dict[str, Any]] = []

        async def fetch(cursor: str | None) -> dict[str, Any]:
            # The account-wide sliding window is shared even across processes;
            # walking through a live 429 deepens the cooldown live orders queue behind.
            remaining = self._backoff_remaining()
            if remaining > 0.0:
                result.backoff_aborted = True
                raise _BackoffActiveError(remaining)
            result.pages_consumed += 1
            payload = await self._client.get_markets(
                # The query is "settled"; the market's own status is "finalized".
                status="settled",
                series_ticker=self._series_ticker,
                cursor=cursor,
                limit=SETTLEMENT_PAGE_LIMIT,
            )
            if isinstance(payload, dict):
                rows = payload.get("markets")
                if isinstance(rows, list):
                    seen_pages.extend(rows)
            return payload

        try:
            page = await paginate_detailed(
                method_name="kxrain_settlements.sweep",
                list_field="markets",
                fetch=fetch,
                page_delay=self._page_delay,
                filter_repr=f"series_ticker={self._series_ticker!r} status='settled'",
                max_pages=self._max_pages,
            )
        except _BackoffActiveError as exc:
            logger.warning(
                "kxrain settlement sweep: the shared Kalshi backoff went active "
                "(%.1fs) after %d page(s); stopping the walk and keeping the %d "
                "market(s) already returned",
                exc.remaining,
                result.pages_consumed,
                len(seen_pages),
            )
            result.n_returned = len(seen_pages)
            return seen_pages
        except Exception as exc:
            logger.warning(
                "kxrain settlement sweep: the fetch failed after %d page(s); "
                "keeping the %d market(s) already returned",
                result.pages_consumed,
                len(seen_pages),
                exc_info=True,
            )
            result.error = f"{type(exc).__name__}: {exc}"
            result.n_returned = len(seen_pages)
            return seen_pages

        result.truncated = bool(page.truncated)
        result.cycle_aborted = bool(page.cycle_aborted)
        result.n_returned = len(page.rows)
        return page.rows

    def _build_rows(
        self,
        markets: list[dict[str, Any]],
        index: SettledIndex,
        now: datetime,
        result: SettlementSweepResult,
    ) -> tuple[list[dict[str, Any]], list[CorpusKey | None]]:
        rows: list[dict[str, Any]] = []
        marked: list[CorpusKey | None] = []
        seen: set[CorpusKey] = set()
        for market in markets:
            if not isinstance(market, dict):
                result.n_unparseable_ticker += 1
                continue
            ticker = str(market.get("ticker") or "")
            key = parse_kxrain_ticker(ticker)
            if key is None:
                # A suffix outside the known city map means the universe changed shape.
                result.n_unparseable_ticker += 1
                if len(result.unparseable_tickers) < MAX_UNPARSEABLE_TICKERS:
                    result.unparseable_tickers.append(ticker)
                continue
            corpus_key = key.corpus_key
            if index.contains(corpus_key) or corpus_key in seen:
                result.n_already_present += 1
                continue
            row = build_settlement_row(market, ts=now)
            if row is None:
                # An open market carries result "" — writing it would
                # fabricate a NO win that passes the consistency tripwire.
                result.n_refused_unsettled += 1
                continue
            if not _is_finalized(market):
                # A market can carry a result before finalization — same fabricated-NO risk.
                result.n_refused_not_finalized += 1
                continue
            if encode_row(row) is None:
                # Not stubbed: the oversize stub drops station/settlement_date/result,
                # so marking it would retire the market-day with its outcome lost.
                logger.error(
                    "kxrain settlement: row for %s exceeds PIPE_BUF; refusing and "
                    "leaving the market-day for the next sweep",
                    ticker,
                )
                result.n_refused_oversize += 1
                continue
            inconsistent = row.get("rule_consistent") is False
            if inconsistent:
                result.n_rule_inconsistent += 1
                logger.error(
                    "kxrain settlement: %s reports result=%r with expiration_value=%r",
                    row.get("ticker"),
                    row.get("result"),
                    row.get("expiration_value_raw"),
                )
            seen.add(corpus_key)
            rows.append(row)
            marked.append(None if inconsistent else corpus_key)  # inconsistent stays retryable
        result.n_written = len(rows)
        return rows, marked

    def _persist(
        self,
        rows: list[dict[str, Any]],
        marked: list[CorpusKey | None],
        index: SettledIndex,
        now: datetime,
        result: SettlementSweepResult,
    ) -> None:
        """Write, prove it landed via the file-size delta, then mark — never any other order.

        `write_batch` can leave a line torn mid-JSON while still reporting success.
        """
        path = Path(self.writer._path) if hasattr(self.writer, "_path") else None
        widths = [len(encode_row(row) or b"") + 1 for row in rows]
        before = _file_identity(path)
        try:
            self.writer.write_batch(rows)
        except Exception:
            logger.exception("kxrain settlement sweep: write_batch failed")
        after = _file_identity(path)
        # The writer rotates on the first write past UTC midnight, which is always this
        # write; a replaced file started empty, so its whole size is this batch.
        baseline = before[1] if before[0] == after[0] else 0
        grew = max(0, after[1] - baseline)

        persisted = 0
        total = 0
        for width in widths:
            total += width
            if total > grew:
                break
            persisted += 1
        result.n_persisted = persisted

        if persisted < len(rows):
            logger.error(
                "kxrain settlement sweep: only %d of %d rows reached disk "
                "(%d bytes of %d); the rest stay retryable",
                persisted,
                len(rows),
                grew,
                sum(widths),
            )
        if persisted <= 0:
            return

        stamp = now.isoformat()
        for key in marked[:persisted]:
            if key is not None:
                index.mark(key, ts=stamp)
        result.index_saved = index.save_atomic(keep_dates=self._keep_dates)


def sweep_dates(index: SettledIndex) -> list[date]:
    """The settlement dates the index currently covers, newest first."""
    return sorted({key[1] for key in index.keys()}, reverse=True)


# "finalized" is what Kalshi actually returns; "settled" (the query value) and
# "determined" are accepted as plausible spellings.
_FINAL_STATUSES = frozenset({"finalized", "settled", "determined"})


def _is_finalized(market: Mapping[str, Any]) -> bool:
    if market.get("settlement_ts"):
        return True
    status = market.get("status")
    return isinstance(status, str) and status.strip().lower() in _FINAL_STATUSES


def _file_identity(path: Path | None) -> tuple[int | None, int]:
    """``(inode, size)``; ``(None, 0)`` when the file is absent or unreadable."""
    if path is None:
        return None, 0
    try:
        st = path.stat()
    except OSError:
        return None, 0
    return st.st_ino, st.st_size
