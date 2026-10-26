"""Phase 0 passive depth-probing monitor for the intraday EMOS HIGH strategy.

Background coroutine that samples the NBBO orderbook on same-day
KXHIGHMIA / KXHIGHLAX brackets every ~5 minutes and writes one JSONL row
per (ticker, scan_tick) pair to ``state/phase0_depth_probes.jsonl``.

Decoupled from any trading strategy on purpose: the spec (v1.3 §6.4 /
§12.1) requires a 2-week passive measurement window BEFORE the intraday
strategy can ship to shadow, so the depth data must accumulate even
while ``strategies.intraday_emos_high.enabled`` is False. The monitor
auto-enables whenever at least one same-day eligible ticker is open;
when no eligible tickers exist (off-hours, weekends), the loop logs a
debug line and skips the API calls entirely.

L2 capture (1-2 ticks deep per spec §6.4(a)) is ACTIVE as of
2026-05-17. After the TOB pass, ``run_once`` issues one
``get_orderbook(ticker, depth=2)`` per same-day market and populates
``yes_bid_l2_size`` / ``yes_ask_l2_size`` / ``yes_levels_count`` /
``no_levels_count`` (plus the defense-in-depth
``yes_bid_l2_price`` / ``yes_ask_l2_price`` and derived ``l2_present``).
The L2 pass is bounded by a 60s wall-clock budget (``monotonic`` is
checked BEFORE each ``get_orderbook`` await) and skipped entirely when
``len(rows) > PHASE0_L2_MAX_MARKETS_PER_TICK`` (evaluated once at
top-of-pass so a single tick writes all-L2 or all-TOB rows, never
mixed). Four counters surface failure modes: ``l2_skipped_ticks`` /
``l2_wall_clock_aborts_ticks`` / ``l2_fetch_errors_ticks`` /
``l2_monotonicity_violations_ticks``. Internal monotonicity (L2 ≤ L1
within one orderbook payload) is asserted at runtime as a
sort-regression guard. Cross-call invariants are NOT asserted because
``iter_all_markets`` and ``get_orderbook`` are independent API calls
that can legitimately diverge (snapshot skew + the OTM trash-bin
pattern where ``iter_all_markets`` reports yes_bid=0.0 while the
orderbook has small YES bids at 0.01+). If Kalshi flips semantics in a
way that the internal monotonicity catches, the outer ``run_once``
try-catch logs, bumps the monotonicity counter, and the row writes
with L2 fields = ``None``. The prober deliberately does NOT import
parsers from ``python.strategies`` — the hermetic discipline keeps the
prober's import surface stable across strategy refactors.

Spec references:
 - v1.3 §6.4 NBBO_SIZE_CAP-vs-reality reconciliation (Phase 0 calibrated)
 - v1.3 §12.1 Phase 0 depth-probing — exact metric definitions
 - implementation plan §M0 (file is `state/phase0_depth_probes.jsonl`
   per implementer choice; plan suggested `depth_samples_intraday.jsonl`
   but the implementer-chosen name is locked here for forward consumers).
 - L2 reinstatement plan (intraday EMOS HIGH, Phase 0 depth probing)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from python.common import et_today
from python.utils.jsonl import ForecastsRawWriter
from python.utils.ticker_dates import parse_ticker_date

if TYPE_CHECKING:
    from python.data.kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

__all__ = [
    "Phase0DepthMonitor",
    "PHASE0_DEPTH_SINK_FILENAME",
    "PHASE0_SAMPLE_INTERVAL_SECONDS",
    "PHASE0_ELIGIBLE_SERIES",
    "PHASE0_SAMPLE_KIND",
    "PHASE0_L2_MAX_MARKETS_PER_TICK",
    "PHASE0_L2_WALL_CLOCK_BUDGET_SECONDS",
    "PHASE0_L2_PACING_SLEEP_SECONDS",
]

# Hard-coded per binding decision #1: 5-min sample interval, no config knob.
# Per-tick read budget = ``len(PHASE0_ELIGIBLE_SERIES)`` paginated
# ``iter_all_markets`` calls (2 today, one per series) PLUS one
# ``get_orderbook(ticker, depth=2)`` per same-day market (when the L2
# pass fires, bounded by ``PHASE0_L2_MAX_MARKETS_PER_TICK`` and a 60s
# wall-clock budget — see the L2 capture section in ``run_once``).
PHASE0_SAMPLE_INTERVAL_SECONDS: float = 300.0

# Filename for the JSONL sink. The implementation plan suggested
# ``depth_samples_intraday.jsonl``; ``phase0_depth_probes.jsonl`` is
# chosen here because the name is more discoverable from the spec
# vocabulary ("Phase 0 depth probing") and survives the rename when
# Phase 1 starts.
PHASE0_DEPTH_SINK_FILENAME: str = "phase0_depth_probes.jsonl"

# Sentinel that goes on every row so downstream consumers (daily report,
# future passive monitors) can multiplex JSONL streams without ambiguity.
PHASE0_SAMPLE_KIND: str = "phase0_intraday_depth"

# Eligible series tickers map to the (cell_label, city) pair recorded
# on every emitted row. Locked to the spec's MIA/high + LAX/high carve-out;
# adding cells requires a fresh §8.2 audit at the new size.
PHASE0_ELIGIBLE_SERIES: dict[str, tuple[str, str]] = {
    "KXHIGHMIA": ("MIA/high", "MIA"),
    "KXHIGHLAX": ("LAX/high", "LAX"),
}

# L2 capture per-tick guard: if the same-day market count exceeds this,
# the L2 pass is skipped entirely for the tick (no mixed-state JSONL).
# 2.5x today's empirical per-tick count of 12; documents the upper bound
# at which the operator must reassess /orderbook share of Kalshi's
# 100/10s rate-limit window.
PHASE0_L2_MAX_MARKETS_PER_TICK: int = 30

# L2 capture wall-clock guard: per-tick budget for the L2 pass. Kalshi
# outage-induced retry-backoff is the realistic operational threat —
# the 60s budget aborts the pass cleanly so the tick still writes TOB
# rows on time. Evaluated BEFORE each ``get_orderbook`` await.
PHASE0_L2_WALL_CLOCK_BUDGET_SECONDS: float = 60.0

# Inter-call pacing for the L2 pass; matches the convention
# ``iter_all_markets`` already uses (``page_delay``); smooths the burst
# across the ``RateLimiter(100, 10.0)`` window the prober shares with
# the live weather strategy.
PHASE0_L2_PACING_SLEEP_SECONDS: float = 0.05


def _parse_fp_int(val: Any) -> int | None:
    """Parse a Kalshi ``*_size_fp`` field (e.g., ``"636.00"``) to ``int``.

    Mirrors :func:`python.strategies.weather_parsing._parse_fp_int` (kept
    local so the monitor never imports the strategy package).
    """
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _parse_dollar_price(val: Any) -> float | None:
    """Parse a Kalshi ``*_dollars`` field (e.g., ``"0.42"``) to ``float`` in [0, 1].

    Mirrors :func:`python.strategies.weather_parsing._parse_dollar_price` —
    out-of-range values log WARNING before returning None. This is the
    safety signal for a cents↔dollars encoding flip on the orderbook side
    (the only schema-drift class that survives the local sort + envelope
    checks; the previous cross-call assertion attempted to catch it but
    was structurally unsound — see module docstring).
    """
    if val is None:
        return None
    try:
        p = float(val)
    except (TypeError, ValueError):
        return None
    if p < 0.0 or p > 1.0:
        logger.warning("Dollar price %.4f outside [0, 1] — rejecting", p)
        return None
    return p


def _parse_orderbook_level(level: object) -> tuple[float, int] | None:
    """Parse one ``orderbook_fp`` level entry to ``(price_probability, qty)``.

    Re-implemented locally (Option A in the L2 plan) — mirrors
    :func:`python.strategies.weather_parsing._parse_orderbook_level`
    exactly. The duplication is intentional: the prober's hermetic
    discipline (see module docstring) requires zero imports from
    ``python.strategies``. If the Kalshi level shape ever changes, both
    copies need updating — this is enforced by Test #1 (schema-shape
    regression) and Test #5 (defense-in-depth assertions).
    """
    if not isinstance(level, list) or len(level) < 2:
        logger.debug("orderbook level ignored — unexpected shape: %r", level)
        return None
    price = _parse_dollar_price(level[0])
    qty = _parse_fp_int(level[1])
    if price is None or qty is None or qty <= 0:
        logger.debug("orderbook level ignored — unparseable: %r", level)
        return None
    return price, qty


def _orderbook_price_to_probability(opp_price_dollars: float) -> float:
    """Convert an opposite-side orderbook bid (in dollars) to our-side ask probability.

    Mirrors :func:`python.strategies.weather_parsing._orderbook_price_to_probability`
    — local copy per the hermetic-prober discipline.
    """
    return 1.0 - opp_price_dollars


def build_depth_sample(
    market: dict[str, Any],
    series_ticker: str,
    *,
    now_utc: datetime,
    orderbook: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a single Phase-0 JSONL row from a Kalshi ``/markets`` row dict.

    Reads NBBO depth from the per-market dict's top-of-book size fields
    (``yes_bid_size_fp`` / ``yes_ask_size_fp``) — the same fields
    :mod:`python.strategies.weather_parsing` already uses for production
    sizing. When an ``orderbook`` dict is supplied (the response from
    ``KalshiClient.get_orderbook(ticker, depth=2)``), the L2 fields
    (``yes_bid_l2_size`` / ``yes_bid_l2_price`` / ``yes_ask_l2_size`` /
    ``yes_ask_l2_price`` / ``yes_levels_count`` / ``no_levels_count``)
    are populated from the ``orderbook_fp`` envelope; otherwise they
    are left as ``None``. A derived ``l2_present`` bool is set ``True``
    iff at least one of ``yes_bid_l2_size`` / ``yes_ask_l2_size`` is
    non-None.

    Returns ``None`` only when ``series_ticker`` isn't in
    ``PHASE0_ELIGIBLE_SERIES``. A fully empty book is captured as a row
    with ``empty_book=True`` so downstream P50 stats can include the
    empty-book frequency without a separate counter (spec §6.4(c)).

    Defense-in-depth invariant enforced via ``assert``: internal
    monotonicity (L2 ≤ L1 within one orderbook payload) is checked as a
    sort-regression guard. Cross-call invariants (e.g., L2 vs NBBO from
    ``iter_all_markets``) are NOT asserted because the two API calls are
    independent snapshots and can legitimately diverge (snapshot skew +
    the OTM trash-bin pattern where ``iter_all_markets`` reports
    ``yes_bid=0.0`` while the orderbook has small YES bids at 0.01+).
    Caller is expected to invoke this inside a ``try`` so that
    AssertionError demotes to a WARNING log and an L2-None row instead
    of crashing the prober.

    # NOTE: assert here is intentional; Oracle does not run with python -O.
    """
    cell_label, city = PHASE0_ELIGIBLE_SERIES.get(series_ticker, ("", ""))
    if not cell_label:
        return None

    ticker = market.get("ticker") or ""

    yes_bid = _parse_dollar_price(market.get("yes_bid_dollars"))
    yes_ask = _parse_dollar_price(market.get("yes_ask_dollars"))
    yes_bid_size_int = _parse_fp_int(market.get("yes_bid_size_fp"))
    yes_ask_size_int = _parse_fp_int(market.get("yes_ask_size_fp"))

    # NO-side prices/sizes aren't exposed on ``/markets``; downstream
    # consumers should read ``yes_*`` directly. Keep ``no_*`` keys for
    # forward-compat with the V0 schema (set to ``None``).
    yes_bid_size: int = yes_bid_size_int if yes_bid_size_int is not None else 0
    yes_ask_size: int = yes_ask_size_int if yes_ask_size_int is not None else 0

    if yes_bid is not None and yes_ask is not None:
        mid: float | None = (yes_bid + yes_ask) / 2.0
        spread: float | None = max(yes_ask - yes_bid, 0.0)
    else:
        mid = None
        spread = None

    empty_book = yes_bid_size == 0 and yes_ask_size == 0

    # --- L2 parsing (orderbook_fp envelope) -------------------------------
    yes_bid_l2_size: int | None = None
    yes_bid_l2_price: float | None = None
    yes_ask_l2_size: int | None = None
    yes_ask_l2_price: float | None = None
    yes_levels_count: int | None = None
    no_levels_count: int | None = None

    if orderbook is not None:
        # Production parsers all read ``data["orderbook_fp"]["yes_dollars"]``
        # — NOT ``data["yes_dollars"]`` directly. See module docstring.
        envelope: dict[str, Any] = orderbook.get("orderbook_fp") or {}
        yes_dollars_raw = envelope.get("yes_dollars") or []
        no_dollars_raw = envelope.get("no_dollars") or []

        parsed_yes: list[tuple[float, int]] = []
        for lvl in yes_dollars_raw:
            parsed = _parse_orderbook_level(lvl)
            if parsed is not None:
                parsed_yes.append(parsed)
        parsed_yes.sort(key=lambda pq: pq[0], reverse=True)  # best (highest) YES bid first

        parsed_no: list[tuple[float, int]] = []
        for lvl in no_dollars_raw:
            parsed = _parse_orderbook_level(lvl)
            if parsed is not None:
                parsed_no.append(parsed)
        parsed_no.sort(key=lambda pq: pq[0], reverse=True)  # best (highest) NO bid first

        yes_levels_count = len(parsed_yes)
        no_levels_count = len(parsed_no)

        if len(parsed_yes) >= 2:
            yes_bid_l2_price = parsed_yes[1][0]
            yes_bid_l2_size = parsed_yes[1][1]
        if len(parsed_no) >= 2:
            # Second-highest NO bid → second-lowest derived YES ask after inversion.
            yes_ask_l2_price = _orderbook_price_to_probability(parsed_no[1][0])
            yes_ask_l2_size = parsed_no[1][1]

        # Defense-in-depth: orderbook payload is internally monotone after our
        # sort. This assertion catches a future regression where the sort line
        # is accidentally removed — it cannot fire against well-formed data
        # from our own sort. It does NOT compare against ``yes_bid`` from
        # ``iter_all_markets``, which is a SEPARATE API snapshot and can
        # legitimately diverge by tens of cents (snapshot skew, plus the OTM
        # trash-bin pattern where ``iter_all_markets`` reports yes_bid=0.0
        # while the orderbook has small YES bids).
        # NOTE: assert here is intentional; Oracle does not run with python -O.
        if len(parsed_yes) >= 2:
            assert parsed_yes[1][0] <= parsed_yes[0][0], (
                f"YES L2 monotonicity violation: "
                f"L1={parsed_yes[0][0]} L2={parsed_yes[1][0]} ticker={ticker}"
            )
        if len(parsed_no) >= 2:
            assert parsed_no[1][0] <= parsed_no[0][0], (
                f"NO L2 monotonicity violation: "
                f"L1={parsed_no[0][0]} L2={parsed_no[1][0]} ticker={ticker}"
            )

    l2_present = yes_bid_l2_size is not None or yes_ask_l2_size is not None

    return {
        "sample_kind": PHASE0_SAMPLE_KIND,
        "ts": now_utc.isoformat(),
        "ts_unix": now_utc.timestamp(),
        "ticker": ticker,
        "series_ticker": series_ticker,
        "cell": cell_label,
        "city": city,
        # YES side (the "buy YES" perspective the strategy will trade)
        "yes_bid": yes_bid,
        "yes_bid_size": yes_bid_size,
        "yes_bid_l2_size": yes_bid_l2_size,
        "yes_bid_l2_price": yes_bid_l2_price,
        "yes_ask": yes_ask,
        "yes_ask_size": yes_ask_size,
        "yes_ask_l2_size": yes_ask_l2_size,
        "yes_ask_l2_price": yes_ask_l2_price,
        # NO side not exposed by /markets; forward-compat fields stay
        # in the schema (set to None).
        "no_bid": None,
        "no_bid_size": None,
        "no_bid_l2_size": None,
        # Derived
        "mid": mid,
        "spread": spread,
        "empty_book": empty_book,
        # Ladder counts: populated from ``orderbook_fp`` when L2 capture fires.
        "yes_levels_count": yes_levels_count,
        "no_levels_count": no_levels_count,
        # Derived bool — True iff at least one L2 size field is non-None.
        "l2_present": l2_present,
    }


class Phase0DepthMonitor:
    """Background coroutine that samples NBBO depth for Phase 0.

    The monitor is fail-soft: a Kalshi API error on any single ticker is
    logged and skipped, and the loop continues with the next tick. The
    monitor never raises into the orchestrator's main loop.

    Lifecycle:
      1. ``__init__`` — construct with a Kalshi client + state dir; the
         sink writer is created lazily on first sample.
      2. ``run`` — entry point for the background task; loops until
         ``CancelledError``.
      3. ``run_once`` — one iteration; exposed for unit testing the
         sample-loop body without the asyncio sleep.
    """

    def __init__(
        self,
        kalshi_client: KalshiClient,
        state_dir: Path | str,
        *,
        interval_seconds: float = PHASE0_SAMPLE_INTERVAL_SECONDS,
        retention_days: int = 30,
        clock: Any = None,
    ) -> None:
        self._kalshi = kalshi_client
        self._state_dir = Path(state_dir)
        self._interval = float(interval_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        sink_path = self._state_dir / PHASE0_DEPTH_SINK_FILENAME
        # Retention >= 14 days (spec §6.4 needs a 2-week measurement
        # window before re-anchoring); pick 30 to give head-room for a
        # delayed daily-report pull. The ForecastsRawWriter rotates on
        # UTC midnight + prunes older siblings.
        self._writer = ForecastsRawWriter(
            sink_path,
            retention_days=retention_days,
            clock=self._clock,
        )
        # Cached lookup over PHASE0_ELIGIBLE_SERIES keys for the inner loop.
        self._series_tickers: list[str] = sorted(PHASE0_ELIGIBLE_SERIES.keys())
        # Cumulative count of ticks that hit iter_all_markets' MAX_PAGES
        # cap (truncated=True). Surfaced in the daily report via the
        # ``markets_truncated_ticks`` accessor — operators need a single
        # number to know "Phase 0 missed markets this window".
        self._markets_truncated_ticks: int = 0
        # L2-pass differentiated counters (see module docstring + plan).
        self._l2_skipped_ticks: int = 0
        self._l2_wall_clock_aborts_ticks: int = 0
        self._l2_fetch_errors_ticks: int = 0
        # Bumped when ``build_depth_sample``'s internal monotonicity
        # AssertionError fires (sort-regression guard). Separate from
        # ``_l2_fetch_errors_ticks`` because the cause is structural
        # (our code), not transport (Kalshi).
        self._l2_monotonicity_violations_ticks: int = 0

    async def run(self) -> None:
        """Loop body — sleep-then-sample; cancellation-safe."""
        logger.info(
            "Phase0DepthMonitor starting (interval=%.0fs, sink=%s)",
            self._interval,
            self._state_dir / PHASE0_DEPTH_SINK_FILENAME,
        )
        try:
            while True:
                try:
                    await self.run_once()
                except Exception:
                    # Per-tick guard: we never want a bad orderbook
                    # response or a transient network error to stop
                    # depth probing for the rest of the session.
                    logger.exception("Phase0DepthMonitor tick failed; continuing")
                try:
                    await asyncio.sleep(self._interval)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            logger.info("Phase0DepthMonitor cancelled")
            raise

    async def run_once(self) -> int:
        """Sample every eligible same-day market once. Returns rows written.

        Per-tick read budget:
          - ``len(PHASE0_ELIGIBLE_SERIES)`` paginated ``iter_all_markets``
            calls (2 today) for the TOB pass.
          - Up to one ``get_orderbook(ticker, depth=2)`` per same-day
            market for the L2 pass, bounded by:
              * ``PHASE0_L2_MAX_MARKETS_PER_TICK`` (skip entire pass at
                top-of-tick if exceeded).
              * ``PHASE0_L2_WALL_CLOCK_BUDGET_SECONDS`` (abort mid-pass
                if exceeded; remaining rows write with L2 = None).
            Pacing: ``await asyncio.sleep(PHASE0_L2_PACING_SLEEP_SECONDS)``
            between consecutive ``get_orderbook`` calls.

        Exceptions are caught and counted, never propagated. The L2
        pass either fires for ALL eligible rows or NONE (guard checks at
        top-of-pass); inside the pass, per-call errors leave individual
        rows L2-None without aborting the rest. CancelledError still
        propagates (test #10).
        """
        today = et_today()
        market_rows: list[tuple[dict[str, Any], str]] = []
        now_utc = self._clock()
        n_eligible_markets = 0

        for series_ticker in self._series_tickers:
            try:
                # Production parity — see weather_edge.py for the same
                # (rows, truncated) return-shape contract. iter_all_markets
                # already throttles via its ``page_delay`` to avoid Kalshi's
                # undocumented sliding-window rate limit.
                raw_markets, markets_truncated = await self._kalshi.iter_all_markets(
                    status="open",
                    series_ticker=series_ticker,
                )
            except Exception:
                logger.warning(
                    "Phase0DepthMonitor: iter_all_markets(%s) failed",
                    series_ticker,
                    exc_info=True,
                )
                continue

            if markets_truncated:
                self._markets_truncated_ticks += 1
                logger.warning(
                    "Phase 0 depth monitor truncated market list for series %s — "
                    "pagination cap hit; some markets not sampled",
                    series_ticker,
                )

            for market in raw_markets:
                ticker = market.get("ticker") or ""
                if not ticker:
                    continue
                parsed = parse_ticker_date(ticker)
                if parsed != today:
                    continue
                n_eligible_markets += 1
                market_rows.append((market, series_ticker))

        if not market_rows:
            if n_eligible_markets == 0:
                logger.debug("Phase0DepthMonitor: no eligible same-day markets this tick")
            return 0

        # --- L2 pass: fetch /orderbook for each eligible same-day market.
        # Decision is made ONCE at top-of-pass so a single tick writes
        # all-L2 or all-TOB rows, never mixed.
        l2_payloads: dict[str, dict[str, Any] | None] = {}

        if len(market_rows) > PHASE0_L2_MAX_MARKETS_PER_TICK:
            self._l2_skipped_ticks += 1
            logger.warning(
                "Phase0DepthMonitor: L2 pass skipped — "
                "%d markets exceeds PHASE0_L2_MAX_MARKETS_PER_TICK=%d",
                len(market_rows),
                PHASE0_L2_MAX_MARKETS_PER_TICK,
            )
            for market, _ in market_rows:
                ticker = market.get("ticker") or ""
                if ticker:
                    l2_payloads[ticker] = None
        else:
            pass_start = time.monotonic()
            wall_clock_aborted = False
            for i, (market, _) in enumerate(market_rows):
                ticker = market.get("ticker") or ""
                if not ticker:
                    continue

                # Wall-clock guard is checked BEFORE each get_orderbook
                # await so the await itself cannot blow the budget.
                if time.monotonic() - pass_start >= PHASE0_L2_WALL_CLOCK_BUDGET_SECONDS:
                    if not wall_clock_aborted:
                        self._l2_wall_clock_aborts_ticks += 1
                        wall_clock_aborted = True
                        logger.warning(
                            "Phase0DepthMonitor: L2 pass wall-clock budget "
                            "exceeded after %d/%d markets; remaining rows "
                            "will write with L2=None",
                            i,
                            len(market_rows),
                        )
                    l2_payloads[ticker] = None
                    continue

                # Inter-call pacing — smooths the burst across the
                # shared RateLimiter(100, 10.0) window.
                if i > 0:
                    await asyncio.sleep(PHASE0_L2_PACING_SLEEP_SECONDS)

                try:
                    payload = await self._kalshi.get_orderbook(ticker, depth=2)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._l2_fetch_errors_ticks += 1
                    logger.warning(
                        "Phase0DepthMonitor: get_orderbook(%s) failed",
                        ticker,
                        exc_info=True,
                    )
                    l2_payloads[ticker] = None
                    continue

                l2_payloads[ticker] = payload

        # Build rows now that the (entire) L2 decision is settled.
        rows: list[dict[str, Any]] = []
        for market, series_ticker in market_rows:
            ticker = market.get("ticker") or ""
            row_payload: dict[str, Any] | None = l2_payloads.get(ticker)
            try:
                row = build_depth_sample(
                    market,
                    series_ticker,
                    now_utc=now_utc,
                    orderbook=row_payload,
                )
            except AssertionError:
                # Defense-in-depth internal monotonicity fired — fall
                # back to L2-None for this row only (the rest of the
                # tick still writes with whatever L2 it has). This
                # surfaces a real sort regression in our own parser, NOT
                # cross-call snapshot skew (the original cross-call
                # assertion was structurally unsound — see module
                # docstring).
                self._l2_monotonicity_violations_ticks += 1
                logger.warning(
                    "Phase0DepthMonitor: L2 monotonicity violated for %s; writing row with L2=None",
                    ticker,
                    exc_info=True,
                )
                row = build_depth_sample(market, series_ticker, now_utc=now_utc, orderbook=None)
            if row is not None:
                rows.append(row)

        if not rows:
            return 0

        try:
            # ForecastsRawWriter.write_batch is synchronous; offload so
            # the asyncio loop isn't blocked by fsync latency.
            await asyncio.to_thread(self._writer.write_batch, rows)
        except Exception:
            logger.exception("Phase0DepthMonitor: write_batch failed")
            return 0
        return len(rows)

    @property
    def sink_path(self) -> Path:
        return self._state_dir / PHASE0_DEPTH_SINK_FILENAME

    @property
    def markets_truncated_ticks(self) -> int:
        """Cumulative count of ticks where iter_all_markets hit MAX_PAGES.

        Surfaced into the daily report so operators see at a glance that
        Phase 0 missed markets during the lookback window.
        """
        return self._markets_truncated_ticks

    @property
    def l2_skipped_ticks(self) -> int:
        """Cumulative count of ticks where the market-count guard skipped L2."""
        return self._l2_skipped_ticks

    @property
    def l2_wall_clock_aborts_ticks(self) -> int:
        """Cumulative count of ticks where the L2 pass aborted on the wall-clock budget."""
        return self._l2_wall_clock_aborts_ticks

    @property
    def l2_fetch_errors_ticks(self) -> int:
        """Cumulative count of individual get_orderbook fetch errors across all ticks."""
        return self._l2_fetch_errors_ticks

    @property
    def l2_monotonicity_violations_ticks(self) -> int:
        """Cumulative count of rows where the internal L2 ≤ L1 sort-regression
        guard fired. Should stay at 0 in steady state; a non-zero value
        signals an actual code bug (e.g., the sort was accidentally removed),
        NOT cross-call snapshot skew."""
        return self._l2_monotonicity_violations_ticks
