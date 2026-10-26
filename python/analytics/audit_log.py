"""Append-only JSONL audit streams for diagnostic instrumentation.

Three streams answer "why no trades?" with data:

* ``eval_audit.jsonl`` — one row per market evaluated, tagged with the
  terminal rejection reason (or ``PASSED``).
* ``candidate_audit.jsonl`` — one row per Stage-1 candidate that reached
  the Stage-2 depth check, with full pricing / sizing / orderbook /
  bracket context.
* ``scan_funnel.jsonl`` — one row per scan cycle summarising the per-gate
  survivor counts.

Each stream carries its own ``schema_version`` constant so a bump to
one stream does not force a version bump on the other two. Rows are
rotated daily at UTC midnight (lazy — rotation fires on the first
write after the boundary) into ``{stream}.jsonl.{YYYY-MM-DD}`` files,
with files older than ``retention_days`` deleted on each rotation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-stream schema versions.  Bump *only* the one whose row shape
# changed; readers check their expected stream's version and can
# upgrade independently.  Kept as distinct constants so "one bump
# affects all three" can't happen by accident.
EVAL_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
FUNNEL_SCHEMA_VERSION = 6

# Default retention for rotated JSONL files (days).  Two weeks is long
# enough for the investigations we run (72h rolling Brier analysis,
# week-long incident drill-down) and short enough to keep per-stream
# disk use bounded even if rows average ~1 KB and we emit 10k/day.
DEFAULT_RETENTION_DAYS = 14

# Fail-safe ceiling for BufferedAuditLogger: past this, append() falls back
# to inline writes rather than growing the buffer unbounded.
DEFAULT_MAX_BUFFERED_ROWS = 20000

# Filename pattern for rotated files: {base}.{YYYY-MM-DD}.  Parsed on
# retention scans to decide whether a sibling file is stale.
_ROTATED_SUFFIX_RE = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})(?:\.\d+)?$")


class RejectionReason(StrEnum):
    """Terminal decisions recorded by the weather evaluation pipeline.

    Ordering is the funnel order: scan-level filters first, then
    per-market evaluation, then Stage-2 depth/edge adjustment.
    ``PASSED`` means the signal fired.
    """

    PAST_DATE = "PAST_DATE"
    EXCLUDED_CITY_VAR = "EXCLUDED_CITY_VAR"
    VOLUME = "VOLUME"
    OPEN_INTEREST = "OPEN_INTEREST"
    PRICE_GATE = "PRICE_GATE"
    SAME_DAY_BLOCKED = "SAME_DAY_BLOCKED"

    SPREAD = "SPREAD"
    RAW_PROB_GATE = "RAW_PROB_GATE"
    STRIKE_DISTANCE = "STRIKE_DISTANCE"
    CALIBRATOR_DISAGREEMENT = "CALIBRATOR_DISAGREEMENT"
    PLATEAU_BLOCK = "PLATEAU_BLOCK"
    CAL_FLOOR_BLOCK = "CAL_FLOOR_BLOCK"
    YES_FLOOR = "YES_FLOOR"
    YES_INVERTED = "YES_INVERTED"
    NO_INVERTED = "NO_INVERTED"
    EDGE_BELOW_THRESHOLD = "EDGE_BELOW_THRESHOLD"
    EDGE_ABOVE_MAX = "EDGE_ABOVE_MAX"
    KELLY_ZERO = "KELLY_ZERO"

    ORDERBOOK_THIN = "ORDERBOOK_THIN"
    FILL_EDGE_NEGATIVE = "FILL_EDGE_NEGATIVE"
    FILL_SLIPPAGE = "FILL_SLIPPAGE"
    EVENT_CAPPED = "EVENT_CAPPED"

    # Per-cell daily trade-count cap. Fires AFTER ``_select_one_per_event``
    # collapse so the counter measures actual emissions, not pre-collapse
    # candidates.
    TRADES_PER_CELL_CAP = "TRADES_PER_CELL_CAP"

    # Runtime σ_emos / bracket_width incompatibility gate (Tier 2 A).
    # Fires when a live EMOS cell's predictive σ at the current ens_std exceeds
    # ``weather.emos_runtime_sigma_threshold`` × bracket_width (1.0°F for
    # KXHIGH/KXLOWT). Catches the LAX/high failure mode (σ_emos≈5.76°F vs 1°F
    # bracket → live ROI=−39.6%) at evaluation time for already-loaded
    # coefficients. The fit-time sibling lives in ``emos_fit.bootstrap_cell``.
    # Only fires when ``live_calibrator_kind == "emos_raw"``; isotonic cells
    # and isotonic_fallback paths are unaffected.
    EMOS_SIGMA_INCOMPATIBLE = "EMOS_SIGMA_INCOMPATIBLE"

    # Interim EMOS-gap safeguard: |model_prob - market_mid| exceeds threshold on
    # a near-term market (lead_days <= 2). Indicates stale GFS ensemble disagrees
    # wildly with the market — likely market has priced in recent observations.
    # Remove once EMOS scope-B is live >= 30d. See config: max_model_market_gap_near_term.
    MODEL_MARKET_GAP_INTERIM = "MODEL_MARKET_GAP_INTERIM"

    # NBM-bracket-position veto (shadow-only). Fires on between-strike
    # candidates where NBM's deterministic point forecast lands on the OPPOSITE
    # side of the bracket from the model's directional bet (YES + NBM outside, or
    # NO + NBM inside). Backtest evidence (n=32): the vetoed cohort had 46.9%
    # win-rate vs 68.4% for the rest; dropping it turned joined-window P&L from
    # a loss to a gain. Shipped ``enabled=False`` so the gate logs
    # ``nbm_veto_fired`` on every between-strike row WITHOUT rejecting —
    # waiting on n≥80 shadow promotion gate. See config:
    # ``weather.nbm_bracket_position_veto``.
    NBM_BRACKET_POSITION_VETO = "NBM_BRACKET_POSITION_VETO"
    # Companion reason for NBM fetch failure when ``fail_open=False`` (default
    # is ``fail_open=True``; this reason will not fire in shipped config).
    NBM_FETCH_FAILED = "NBM_FETCH_FAILED"

    FORECAST_MISSING = "FORECAST_MISSING"  # data-ingestion gap; no forecast for (city, date, var)
    NO_PRICE = "NO_PRICE"  # market has no usable yes_bid/yes_ask/midpoint
    KELLY_SIDE_FLIP = "KELLY_SIDE_FLIP"  # Kelly re-run after depth-check flipped trade side

    PER_DAY_EVENT_CAP_HIT = "per_day_event_cap_hit"

    # Boundary-distance suppressive filter (Path 7a; evidence in
    # docs/research/2026-05-28-boundary-aware-phase0-backtest.md). Fires when the
    # blended forecast mean is closer than the cell's configured threshold to
    # the nearest NWS-rounding boundary. Only active for cells listed in
    # ``weather.boundary_filter_cells``; stays at zero in the default config.
    BOUNDARY_FILTER_BLOCK = "BOUNDARY_FILTER_BLOCK"

    # ASOS-nowcast physical plausibility guards (intraday_asos_high).
    # These fire when the ASOS observation or posterior fails a physical
    # plausibility check before trading.
    # Candidate's posterior mean sits within the configured boundary-distance
    # threshold of the nearest NWS-rounding boundary (grazer zone).
    ASOS_BOUNDARY_GUARD = "ASOS_BOUNDARY_GUARD"
    # The ASOS observation is older than the configured max age (hours).
    ASOS_STALE_OBS = "ASOS_STALE_OBS"
    # The ASOS observation leaves zero headroom toward the strike before the
    # posterior becomes uninformative (e.g. observed temp already past strike).
    ASOS_ZERO_HEADROOM = "ASOS_ZERO_HEADROOM"
    # The preliminary NWS report drifts from the ASOS observation by more
    # than the configured threshold — indicates unstable preliminary reading.
    ASOS_BAD_PRELIM = "ASOS_BAD_PRELIM"
    # Climatology data unavailable for this cell/date combination — required
    # for posterior computation; reject rather than fall back to a flat prior.
    ASOS_CLIMO_UNAVAILABLE = "ASOS_CLIMO_UNAVAILABLE"

    PASSED = "PASSED"


# Stable numeric codes for space-efficient dashboards.
_REASON_NUMERIC: dict[str, int] = {
    RejectionReason.PAST_DATE: 1,
    RejectionReason.EXCLUDED_CITY_VAR: 2,
    RejectionReason.VOLUME: 3,
    RejectionReason.OPEN_INTEREST: 4,
    RejectionReason.PRICE_GATE: 5,
    RejectionReason.SAME_DAY_BLOCKED: 6,
    RejectionReason.SPREAD: 10,
    RejectionReason.RAW_PROB_GATE: 11,
    RejectionReason.STRIKE_DISTANCE: 12,
    RejectionReason.CALIBRATOR_DISAGREEMENT: 19,
    RejectionReason.PLATEAU_BLOCK: 23,
    RejectionReason.CAL_FLOOR_BLOCK: 26,
    RejectionReason.YES_FLOOR: 13,
    RejectionReason.YES_INVERTED: 14,
    RejectionReason.NO_INVERTED: 15,
    RejectionReason.EDGE_BELOW_THRESHOLD: 16,
    RejectionReason.EDGE_ABOVE_MAX: 17,
    RejectionReason.KELLY_ZERO: 18,
    RejectionReason.ORDERBOOK_THIN: 20,
    RejectionReason.FILL_EDGE_NEGATIVE: 21,
    RejectionReason.FILL_SLIPPAGE: 22,
    RejectionReason.EVENT_CAPPED: 24,
    RejectionReason.MODEL_MARKET_GAP_INTERIM: 25,
    RejectionReason.FORECAST_MISSING: 27,
    RejectionReason.NO_PRICE: 28,
    RejectionReason.KELLY_SIDE_FLIP: 29,
    RejectionReason.TRADES_PER_CELL_CAP: 30,
    RejectionReason.EMOS_SIGMA_INCOMPATIBLE: 31,
    RejectionReason.NBM_BRACKET_POSITION_VETO: 32,
    RejectionReason.NBM_FETCH_FAILED: 33,
    RejectionReason.PER_DAY_EVENT_CAP_HIT: 34,
    RejectionReason.BOUNDARY_FILTER_BLOCK: 35,
    RejectionReason.ASOS_BOUNDARY_GUARD: 36,
    RejectionReason.ASOS_STALE_OBS: 37,
    RejectionReason.ASOS_ZERO_HEADROOM: 38,
    RejectionReason.ASOS_BAD_PRELIM: 39,
    RejectionReason.ASOS_CLIMO_UNAVAILABLE: 40,
    RejectionReason.PASSED: 99,
}


def reason_numeric(reason: str | RejectionReason) -> int:
    """Return the stable numeric code for a reason string."""
    return _REASON_NUMERIC.get(str(reason), 0)


# Rejection → ScanFunnel attribute. Module-level so incrementing a
# counter doesn't rebuild the mapping on every call.
_REASON_TO_FUNNEL_ATTR: dict[str, str] = {
    RejectionReason.PAST_DATE: "n_past_date",
    RejectionReason.VOLUME: "n_volume",
    RejectionReason.OPEN_INTEREST: "n_oi",
    RejectionReason.PRICE_GATE: "n_price",
    RejectionReason.SAME_DAY_BLOCKED: "n_same_day_blocked",
    RejectionReason.EXCLUDED_CITY_VAR: "n_excluded_city_var",
    RejectionReason.SPREAD: "n_spread_reject",
    RejectionReason.RAW_PROB_GATE: "n_raw_prob_reject",
    RejectionReason.STRIKE_DISTANCE: "n_strike_distance_reject",
    RejectionReason.CALIBRATOR_DISAGREEMENT: "n_calibrator_disagreement",
    RejectionReason.PLATEAU_BLOCK: "n_plateau_block",
    RejectionReason.CAL_FLOOR_BLOCK: "n_cal_floor_block",
    RejectionReason.YES_FLOOR: "n_yes_floor",
    RejectionReason.YES_INVERTED: "n_yes_inverted",
    RejectionReason.NO_INVERTED: "n_no_inverted",
    RejectionReason.EDGE_BELOW_THRESHOLD: "n_edge_below",
    RejectionReason.EDGE_ABOVE_MAX: "n_edge_above_max",
    RejectionReason.KELLY_ZERO: "n_kelly_zero",
    RejectionReason.ORDERBOOK_THIN: "n_orderbook_thin",
    RejectionReason.FILL_EDGE_NEGATIVE: "n_fill_edge_low",
    RejectionReason.FILL_SLIPPAGE: "n_fill_slippage",
    RejectionReason.EVENT_CAPPED: "n_event_capped",
    RejectionReason.MODEL_MARKET_GAP_INTERIM: "n_model_market_gap",
    RejectionReason.FORECAST_MISSING: "n_forecast_gap",
    RejectionReason.NO_PRICE: "n_no_price",
    RejectionReason.KELLY_SIDE_FLIP: "n_kelly_side_flip",
    RejectionReason.TRADES_PER_CELL_CAP: "n_after_trades_per_cell_per_day_cap",
    RejectionReason.EMOS_SIGMA_INCOMPATIBLE: "n_emos_sigma_incompatible",
    RejectionReason.NBM_BRACKET_POSITION_VETO: "n_nbm_bracket_position_veto",
    RejectionReason.NBM_FETCH_FAILED: "n_nbm_fetch_failed",
    RejectionReason.PER_DAY_EVENT_CAP_HIT: "n_per_day_cap_blocked",
    RejectionReason.BOUNDARY_FILTER_BLOCK: "n_boundary_filter_block",
    RejectionReason.ASOS_BOUNDARY_GUARD: "n_asos_boundary_guard",
    RejectionReason.ASOS_STALE_OBS: "n_asos_stale_obs",
    RejectionReason.ASOS_ZERO_HEADROOM: "n_asos_zero_headroom",
    RejectionReason.ASOS_BAD_PRELIM: "n_asos_bad_prelim",
    RejectionReason.ASOS_CLIMO_UNAVAILABLE: "n_asos_climo_unavailable",
    RejectionReason.PASSED: "n_signals_fired",
}


def funnel_attr_for(reason: str | RejectionReason) -> str | None:
    """Return the ScanFunnel attribute that ``reason`` increments, or None."""
    return _REASON_TO_FUNNEL_ATTR.get(str(reason))


@dataclass
class ScanFunnel:
    """Per-scan-cycle counter used to emit ``scan_funnel.jsonl``."""

    n_raw: int = 0
    n_past_date: int = 0
    n_volume: int = 0
    n_oi: int = 0
    n_price: int = 0
    n_same_day_blocked: int = 0
    n_excluded_city_var: int = 0
    n_eval_enter: int = 0
    n_spread_reject: int = 0
    n_raw_prob_reject: int = 0
    n_strike_distance_reject: int = 0
    n_calibrator_disagreement: int = 0
    n_plateau_block: int = 0
    n_cal_floor_block: int = 0
    n_yes_floor: int = 0
    n_yes_inverted: int = 0
    n_no_inverted: int = 0
    n_edge_below: int = 0
    n_edge_above_max: int = 0
    n_kelly_zero: int = 0
    n_stage1_pass: int = 0
    n_orderbook_thin: int = 0
    n_fill_edge_low: int = 0
    n_fill_slippage: int = 0
    n_fill_kelly_zero: int = 0
    n_event_capped: int = 0
    n_per_day_cap_blocked: int = 0
    n_model_market_gap: int = 0
    n_forecast_gap: int = 0
    n_no_price: int = 0
    n_kelly_side_flip: int = 0
    # Count of candidates dropped by the per-cell daily trade-count cap,
    # measured AFTER ``_select_one_per_event`` collapse so the counter
    # aligns with daily-report telemetry.
    n_after_trades_per_cell_per_day_cap: int = 0
    # Count of EMOS-live candidates rejected by the runtime σ_emos /
    # bracket_width gate. Sibling of the fit-time ``incompatible_sigma``
    # warning emitted by ``emos_fit.bootstrap_cell``.
    n_emos_sigma_incompatible: int = 0
    # NBM-bracket-position veto (shadow-only). Counts vetoed candidates when
    # ``weather.nbm_bracket_position_veto.enabled = True``. In the shipped
    # config (``enabled = False``) the gate runs but does not reject, so
    # this counter stays at zero in production.
    n_nbm_bracket_position_veto: int = 0
    # NBM fetch failures rejected by the gate when ``fail_open = False``.
    # In the shipped config (``fail_open = True``) this counter stays at zero.
    n_nbm_fetch_failed: int = 0
    # Boundary-distance suppressive filter (Path 7a). Counts candidates
    # dropped because the blended forecast mean is closer than the cell's
    # configured threshold to the nearest NWS-rounding boundary. Only
    # increments for cells in ``weather.boundary_filter_cells``; stays at
    # zero in the default (empty) config.
    n_boundary_filter_block: int = 0
    # ASOS-nowcast physical plausibility guard counters (intraday_asos_high).
    # Each counter maps to the corresponding ASOS_* RejectionReason.
    n_asos_boundary_guard: int = 0
    n_asos_stale_obs: int = 0
    n_asos_zero_headroom: int = 0
    n_asos_bad_prelim: int = 0
    n_asos_climo_unavailable: int = 0
    # Informational-cell drop seam: candidates dropped because the cell is
    # configured as informational-only (no live trading, data-collection only).
    # Does not map to a RejectionReason (not a rejection — these candidates
    # never entered the evaluation pipeline).
    n_informational_cell_drop: int = 0
    # Count of collapsed signals suppressed by the hot-reload pause
    # early-return. Needed so the four-term arithmetic invariant
    # ``returned_signals == n_signals_fired − n_event_capped −
    # n_after_trades_per_cell_per_day_cap − n_hot_reload_paused`` holds on
    # every tick.
    n_hot_reload_paused: int = 0
    n_signals_fired: int = 0

    def increment(self, reason: str | RejectionReason) -> None:
        """Increment the counter that maps to ``reason``. Unknown reasons are ignored."""
        attr = _REASON_TO_FUNNEL_ATTR.get(str(reason))
        if attr is not None:
            setattr(self, attr, getattr(self, attr) + 1)


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def parse_rotated_suffix(name: str) -> date | None:
    """Return the date of a rotated file suffix, or ``None`` if absent/invalid.

    Exported so the tripwire module can share one definition of the
    rotation filename contract.
    """
    m = _ROTATED_SUFFIX_RE.search(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


class AuditLogger:
    """Append-only JSONL writer with per-stream schema and daily rotation.

    Concurrency model: single-writer process. ``self._lock`` guards
    against interleaving from threaded callers sharing one instance;
    cross-process interleaving is the caller's responsibility.

    Rotation: lazy. On each ``append`` the logger checks the current
    UTC date; if it has advanced past the stored boundary, the active
    file is renamed to ``{path}.{YYYY-MM-DD}`` (the *out-going* date),
    a fresh file is opened, and files older than ``retention_days`` are
    pruned. The lazy policy means an idle process at midnight UTC
    defers rotation until the next write — which is fine for our
    write-on-every-tick cadence.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: int = 1,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        clock: Any = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._schema_version = int(schema_version)
        self._retention_days = int(retention_days)
        # Test hook: ``clock()`` returns the current ``date``.
        self._clock = clock or _today_utc
        # Seeded at first write so an idle startup doesn't trigger an
        # immediate (and meaningless) rotation of an empty file.
        self._current_date: date | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def retention_days(self) -> int:
        return self._retention_days

    # -- Rotation / retention ------------------------------------------------

    def _rotate_to(self, out_date: date) -> None:
        """Atomically move the active file to ``{path}.{YYYY-MM-DD}``.

        Called under ``self._lock``. Safe if the active file does not
        exist (first write of the day): the rename is skipped.
        Collisions (rare — would require clock skew) fall back to a
        numeric suffix so we never silently overwrite prior-day data.
        """
        if not self._path.exists():
            return
        target = self._path.with_name(f"{self._path.name}.{out_date.isoformat()}")
        if target.exists():
            for i in range(1, 100):
                candidate = target.with_name(f"{target.name}.{i}")
                if not candidate.exists():
                    target = candidate
                    break
            else:
                logger.error(
                    "audit_log: could not find a free rotation suffix for %s",
                    self._path,
                )
                return
        try:
            os.rename(self._path, target)
            logger.info("audit_log: rotated %s -> %s", self._path, target)
        except OSError:
            logger.exception("audit_log: rotation failed for %s", self._path)

    def _prune_old(self, cutoff: date) -> None:
        """Delete rotated siblings with date suffix strictly older than ``cutoff``.

        ``cutoff`` is inclusive — files dated on or after the cutoff
        are kept. Any sibling whose suffix doesn't parse cleanly is
        left alone (might be a human artifact or an unrelated file).
        """
        parent = self._path.parent
        prefix = f"{self._path.name}."
        try:
            entries = list(parent.iterdir())
        except OSError:
            logger.exception("audit_log: retention scan failed for %s", parent)
            return
        for sibling in entries:
            if not sibling.is_file():
                continue
            if not sibling.name.startswith(prefix):
                continue
            d = parse_rotated_suffix(sibling.name)
            if d is None:
                continue
            if d < cutoff:
                try:
                    sibling.unlink()
                    logger.info(
                        "audit_log: retention pruned %s (%d days old)",
                        sibling,
                        (cutoff - d).days,
                    )
                except OSError:
                    logger.exception(
                        "audit_log: failed to prune %s during retention scan",
                        sibling,
                    )

    def _maybe_rotate(self, now: date) -> None:
        """Rotate the active file if the UTC date has advanced.

        Called under ``self._lock``. First call just seeds the
        boundary; subsequent calls compare against the stored date.
        Retention pruning runs after every successful rotation (cheap,
        bounded by the number of sibling files in the state dir).
        """
        if self._current_date is None:
            self._current_date = now
            return
        if now <= self._current_date:
            return
        self._rotate_to(self._current_date)
        cutoff = date.fromordinal(now.toordinal() - self._retention_days)
        self._prune_old(cutoff)
        self._current_date = now

    def rotate_now(self, out_date: date | None = None) -> None:
        """Force an immediate rotation (useful from tests / maintenance)."""
        with self._lock:
            target_date = out_date or self._current_date or self._clock()
            self._rotate_to(target_date)
            cutoff = date.fromordinal(self._clock().toordinal() - self._retention_days)
            self._prune_old(cutoff)
            self._current_date = self._clock()

    # -- Write path ---------------------------------------------------------

    def append(self, row: dict[str, Any]) -> None:
        """Serialize ``row`` and append as a single JSONL line."""
        row.setdefault("schema_version", self._schema_version)
        row.setdefault("ts", datetime.now(tz=UTC).isoformat())
        try:
            line = json.dumps(row, default=str, separators=(",", ":")) + "\n"
        except (TypeError, ValueError):
            logger.exception("audit_log: serialization failed for row keys=%s", list(row.keys()))
            return
        payload = line.encode("utf-8")
        now = self._clock()
        with self._lock:
            self._maybe_rotate(now)
            try:
                # O_APPEND is atomic up to PIPE_BUF (~4 KB); rows are ~500 B-1 KB.
                fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
            except OSError:
                logger.exception("audit_log: write failed for %s", self._path)

    def emit_eval(self, row: dict[str, Any]) -> None:
        """Write one row to ``eval_audit.jsonl``. Reason code is stamped if absent."""
        reason = row.get("reason")
        if reason is not None:
            row.setdefault("reason_numeric", reason_numeric(reason))
        self.append(row)

    def emit_candidate(self, row: dict[str, Any]) -> None:
        """Write one row to ``candidate_audit.jsonl``."""
        reason = row.get("reason")
        if reason is not None:
            row.setdefault("reason_numeric", reason_numeric(reason))
        self.append(row)

    def emit_funnel(
        self,
        funnel: ScanFunnel,
        scan_id: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write one row to ``scan_funnel.jsonl`` summarising a scan cycle."""
        row: dict[str, Any] = {
            "scan_id": scan_id,
            "n_raw": funnel.n_raw,
            "n_past_date": funnel.n_past_date,
            "n_volume": funnel.n_volume,
            "n_oi": funnel.n_oi,
            "n_price": funnel.n_price,
            "n_same_day_blocked": funnel.n_same_day_blocked,
            "n_excluded_city_var": funnel.n_excluded_city_var,
            "n_eval_enter": funnel.n_eval_enter,
            "n_spread_reject": funnel.n_spread_reject,
            "n_raw_prob_reject": funnel.n_raw_prob_reject,
            "n_strike_distance_reject": funnel.n_strike_distance_reject,
            "n_calibrator_disagreement": funnel.n_calibrator_disagreement,
            "n_plateau_block": funnel.n_plateau_block,
            "n_cal_floor_block": funnel.n_cal_floor_block,
            "n_yes_floor": funnel.n_yes_floor,
            "n_yes_inverted": funnel.n_yes_inverted,
            "n_no_inverted": funnel.n_no_inverted,
            "n_edge_below": funnel.n_edge_below,
            "n_edge_above_max": funnel.n_edge_above_max,
            "n_kelly_zero": funnel.n_kelly_zero,
            "n_stage1_pass": funnel.n_stage1_pass,
            "n_orderbook_thin": funnel.n_orderbook_thin,
            "n_fill_edge_low": funnel.n_fill_edge_low,
            "n_fill_slippage": funnel.n_fill_slippage,
            "n_fill_kelly_zero": funnel.n_fill_kelly_zero,
            "n_event_capped": funnel.n_event_capped,
            "n_per_day_cap_blocked": funnel.n_per_day_cap_blocked,
            "n_model_market_gap": funnel.n_model_market_gap,
            "n_forecast_gap": funnel.n_forecast_gap,
            "n_no_price": funnel.n_no_price,
            "n_kelly_side_flip": funnel.n_kelly_side_flip,
            "n_after_trades_per_cell_per_day_cap": (funnel.n_after_trades_per_cell_per_day_cap),
            "n_emos_sigma_incompatible": funnel.n_emos_sigma_incompatible,
            "n_nbm_bracket_position_veto": funnel.n_nbm_bracket_position_veto,
            "n_nbm_fetch_failed": funnel.n_nbm_fetch_failed,
            "n_boundary_filter_block": funnel.n_boundary_filter_block,
            "n_asos_boundary_guard": funnel.n_asos_boundary_guard,
            "n_asos_stale_obs": funnel.n_asos_stale_obs,
            "n_asos_zero_headroom": funnel.n_asos_zero_headroom,
            "n_asos_bad_prelim": funnel.n_asos_bad_prelim,
            "n_asos_climo_unavailable": funnel.n_asos_climo_unavailable,
            "n_informational_cell_drop": funnel.n_informational_cell_drop,
            "n_hot_reload_paused": funnel.n_hot_reload_paused,
            "n_signals_fired": funnel.n_signals_fired,
        }
        if extra:
            row.update(extra)
        self.append(row)


class BufferedAuditLogger(AuditLogger):
    """``AuditLogger`` that parks rows in memory; durable only after ``flush_async``.

    Rows are held by reference and serialized at drain time — do not mutate a
    row dict after emitting it.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        schema_version: int = 1,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        clock: Any = None,
        max_buffered_rows: int = DEFAULT_MAX_BUFFERED_ROWS,
    ) -> None:
        super().__init__(
            path,
            schema_version=schema_version,
            retention_days=retention_days,
            clock=clock,
        )
        self._buffer: list[dict[str, Any]] = []
        # Separate from the inherited _lock so an emit never blocks on a drain's write.
        self._buffer_lock = threading.Lock()
        self._max_buffered_rows = int(max_buffered_rows)
        self._overflow_logged = False

    def append(self, row: dict[str, Any]) -> None:
        """Stamp ``row`` and park it — the write happens at the next flush."""
        row.setdefault("schema_version", self._schema_version)
        row.setdefault("ts", datetime.now(tz=UTC).isoformat())
        should_log_overflow = False
        with self._buffer_lock:
            if len(self._buffer) < self._max_buffered_rows:
                self._buffer.append(row)
                return
            if not self._overflow_logged:
                self._overflow_logged = True
                should_log_overflow = True
        if should_log_overflow:
            logger.error(
                "audit_log: %s reached %d buffered rows without a flush — "
                "falling back to inline writes",
                self._path,
                self._max_buffered_rows,
            )
        super().append(row)

    def buffered_count(self) -> int:
        """Number of rows parked and not yet written."""
        with self._buffer_lock:
            return len(self._buffer)

    def drain(self) -> int:
        """Write every parked row; return how many. Blocking — call off-loop."""
        with self._buffer_lock:
            if not self._buffer:
                return 0
            rows, self._buffer = self._buffer, []
            self._overflow_logged = False
        for row in rows:
            super().append(row)
        return len(rows)

    async def flush_async(self) -> None:
        """Drain the buffer on a worker thread. Never raises (CancelledError still propagates)."""
        with self._buffer_lock:
            if not self._buffer:
                return
        try:
            await asyncio.to_thread(self.drain)
        except Exception:
            logger.exception("audit_log: buffered flush failed for %s", self._path)


@dataclass(frozen=True)
class AuditSinks:
    """Bundle of audit loggers threaded through the weather pipeline.

    ``None`` fields disable that stream at the emitter site so callers
    don't need to check the config flag themselves.
    """

    eval_log: AuditLogger | None = None
    candidate_log: AuditLogger | None = None
    funnel_log: AuditLogger | None = None


def build_sinks(
    state_dir: Path | str,
    *,
    eval_enabled: bool,
    candidate_enabled: bool,
    funnel_enabled: bool,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    buffered: bool = False,
) -> AuditSinks:
    """Build the three audit loggers under ``state_dir``.

    Paths: ``{state_dir}/eval_audit.jsonl``, ``candidate_audit.jsonl``,
    ``scan_funnel.jsonl``. Disabled streams yield ``None``. Each
    logger is stamped with its stream-specific schema version so a
    future bump to one stream's row shape does not force a bump on
    the others.

    ``buffered=True`` yields ``BufferedAuditLogger`` instances, whose
    rows reach disk only when the caller awaits ``flush_sinks``. Async
    callers on the trading event loop want this; synchronous callers
    (backtest, scripts, tests) want the default.
    """
    root = Path(state_dir)
    cls: type[AuditLogger] = BufferedAuditLogger if buffered else AuditLogger
    return AuditSinks(
        eval_log=cls(
            root / "eval_audit.jsonl",
            schema_version=EVAL_SCHEMA_VERSION,
            retention_days=retention_days,
        )
        if eval_enabled
        else None,
        candidate_log=cls(
            root / "candidate_audit.jsonl",
            schema_version=CANDIDATE_SCHEMA_VERSION,
            retention_days=retention_days,
        )
        if candidate_enabled
        else None,
        funnel_log=cls(
            root / "scan_funnel.jsonl",
            schema_version=FUNNEL_SCHEMA_VERSION,
            retention_days=retention_days,
        )
        if funnel_enabled
        else None,
    )


async def flush_sinks(sinks: AuditSinks) -> None:
    """Drain every buffered stream in ``sinks``; a no-op for plain loggers."""
    for log in (sinks.eval_log, sinks.candidate_log, sinks.funnel_log):
        if isinstance(log, BufferedAuditLogger):
            await log.flush_async()
