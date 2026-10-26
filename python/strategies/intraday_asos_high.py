"""Intraday ASOS HIGH strategy — math functions + strategy class.

Builds the core decision logic for nowcast-based intraday weather trading:

  - ``AsosGuardConfig``         — physical validity filters (dataclass)
  - ``AsosCandidate``           — frozen result carrying all pricing/guard/model fields
  - ``asos_evaluate``           — posterior computation + guard pipeline → candidate
  - ``size_candidate``          — fractional-Kelly sizing + max_contracts clamp
  - ``IntradayAsosHighStrategy``— strategy class skeleton (init/setup/scan/lifecycle)

These functions are hermetic from ``weather_edge`` — do not import from that module.

Notes on EMOS/ASOS posterior sharpness:
    ASOS posteriors (like EMOS) are already sharp estimates; do NOT apply an
    ensemble-confidence discount to ``effective_edge``.  Mirror the EMOS
    ``ec=1.0`` rule in ``intraday_emos_high.py:_apply_depth_adjustment``.

Notes on edge threshold:
    ``asos_evaluate`` computes and stores ``effective_edge`` on the returned
    ``AsosCandidate`` but does NOT itself enforce the threshold filter — the
    edge threshold check belongs in the strategy class's ``generate_signals``
    loop so that audit rows can be written before dropping.  Pass
    ``edge_threshold`` through for the strategy class to compare against
    ``cand.effective_edge``.

Notes on guard_outcome:
    ``asos_evaluate`` returns the FULL ``AsosCandidate`` in ALL cases, with
    ``guard_outcome`` set to the name of the first failing guard (``"stale"``,
    ``"headroom"``, ``"boundary"``, ``"bad_prelim"``) or ``"pass"`` when all
    guards pass.  The ONLY case that returns ``None`` is when
    ``bracket_probability`` raises ``ValueError`` (truly unevaluable / malformed
    geometry).  Callers (``generate_signals``) log a prediction for every
    non-None candidate and only trade ``guard_outcome == "pass"`` ones — matching
    the project convention that ``predictions.jsonl`` captures unconstrained model
    output before filters.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from python.analytics.audit_log import (
    AuditSinks,
    RejectionReason,
    ScanFunnel,
    build_sinks,
    reason_numeric,
)
from python.analytics.boundary_distance import boundary_distance
from python.analytics.fees import fee_per_contract as kalshi_fee
from python.analytics.prediction_log import PredictionRecord
from python.analytics.sizing import SizingResult, dollars_to_contracts, fractional_kelly
from python.common import DailyVariable, ExecutionMode, Side, et_today
from python.data.asos import AsosClient, ClimatologyTable
from python.data.asos_posterior import bracket_probability
from python.data.kalshi_client import KalshiClient
from python.execution.trade_log import is_pnl_relevant_row
from python.notifications.base import AlertLevel, Notifier
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal
from python.strategies.weather_evaluation import _select_one_per_event
from python.strategies.weather_parsing import (
    _DIRECTION_TO_STRIKE_LABEL,
    DepthCheck,
    OpponentSideView,
    ParsedMarket,
    ScanCandidate,
    _orderbook_price_to_probability,
    _parse_fp_int,
    _parse_orderbook_level,
    compute_fill_from_levels,
    extract_market_price_info,
    parse_weather_market,
)
from python.utils.jsonl import iter_jsonl
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date

logger = logging.getLogger(__name__)


@dataclass
class AsosGuardConfig:
    """Physical validity filters for ``asos_evaluate``.

    Attributes:
        boundary_min_f: Minimum distance (°F) from the NWS-rounding boundary
            that would flip bracket resolution.  Candidates with
            ``boundary_distance < boundary_min_f`` are dropped (grazer
            protection).  Set to 0.0 to disable.
        max_staleness_h: Maximum observation age in hours.  Candidates older
            than this are dropped (stale-ASOS guard).
        sigma_floor_f: Minimum posterior spread (°F).  A very small
            ``sigma_gain`` means the posterior is degenerate and unreliable;
            candidates with ``sigma_gain < sigma_floor_f`` are dropped
            (zero-headroom guard).
        bad_prelim_max_drift_f: Maximum acceptable absolute drift (°F) between
            the observed ``M`` and a preliminary NWS reading ``prelim_high``.
            Only active when ``bad_prelim_active=True``.
        bad_prelim_active: Enable the bad-prelim guard.  Default ``False``;
            flip to ``True`` once we have a reliable prelim feed.
    """

    boundary_min_f: float = 1.0
    max_staleness_h: float = 2.0
    sigma_floor_f: float = 0.5
    bad_prelim_max_drift_f: float = 10.0
    bad_prelim_active: bool = False


@dataclass(frozen=True)
class AsosCandidate:
    """Frozen result from ``asos_evaluate`` for a single market/observation pair.

    Returned for EVERY evaluable market — ``guard_outcome`` records whether the
    candidate passed all guards (``"pass"``) or which guard failed first
    (``"stale"``, ``"headroom"``, ``"boundary"``, ``"bad_prelim"``).

    The only case where ``asos_evaluate`` returns ``None`` instead of an
    ``AsosCandidate`` is when ``bracket_probability`` raises ``ValueError``
    (malformed geometry that is truly unevaluable).

    Callers (``generate_signals``) log predictions for ALL non-None candidates
    and only trade those with ``guard_outcome == "pass"``.

    Attributes:
        parsed:         Parsed Kalshi market metadata.
        model_prob:     Posterior P(bracket settles YES) from ``bracket_probability``.
        post_mean_f:    Point estimate of daily HIGH: ``M + mu_gain`` (°F).
        side:           YES if ``model_prob > yes_ask``, else NO.
        exec_price:     Side-aligned execution price: ``yes_ask`` (YES) or
                        ``1 - yes_bid`` (NO).
        raw_edge:       ``model_prob - exec_price`` (YES) or
                        ``(1 - model_prob) - exec_price`` (NO).
        effective_edge: Same as ``raw_edge`` — no ensemble-confidence discount
                        (ASOS posterior is already sharp; mirror EMOS ec=1.0).
        guard_outcome:  ``"pass"`` when all guards pass; name of the FIRST
                        failing guard otherwise: ``"stale"``, ``"headroom"``,
                        ``"boundary"``, or ``"bad_prelim"``.
        mu_gain:        Remaining-gain location parameter used (°F).
        sigma_gain:     Remaining-gain scale parameter used (°F).
        M:              Observed max-so-far at evaluation time (°F).
        obs_age_hours:  Age of the ASOS observation in hours.
        prelim_drift_f: ``abs(M - prelim_high)`` if ``prelim_high`` provided,
                        else ``None``.  For audit only.
    """

    parsed: ParsedMarket
    model_prob: float
    post_mean_f: float
    side: Side
    exec_price: float
    raw_edge: float
    effective_edge: float
    guard_outcome: str
    mu_gain: float
    sigma_gain: float
    M: float  # noqa: N815
    obs_age_hours: float
    prelim_drift_f: float | None
    boundary_distance_f: float  # distance to nearest NWS rounding boundary (°F); NaN if N/A


def asos_evaluate(
    parsed: ParsedMarket,
    *,
    M: float,  # noqa: N803
    mu_gain: float,
    sigma_gain: float,
    nu: float,
    yes_bid: float,
    yes_ask: float,
    obs_age_hours: float = 0.0,
    prelim_high: float | None = None,
    edge_threshold: float = 0.07,
    guards: AsosGuardConfig,
) -> AsosCandidate | None:
    """Evaluate one Kalshi HIGH market against the ASOS intraday posterior.

    Steps
    -----
    1. Compute ``post_mean_f = M + mu_gain``.
    2. Compute ``model_prob`` via ``bracket_probability``.
    3. Compute side/exec_price/raw_edge/effective_edge unconditionally.
    4. Evaluate guards in order; record the name of the FIRST failing guard
       in ``guard_outcome``, else ``"pass"``.  Guard outcome strings:
       ``"stale"``, ``"headroom"``, ``"boundary"``, ``"bad_prelim"``, ``"pass"``.
    5. Return the FULL ``AsosCandidate`` regardless of guard outcome.

    The ONLY case that returns ``None`` is when ``bracket_probability`` raises
    ``ValueError`` (truly unevaluable / malformed geometry).  Every other path
    returns an ``AsosCandidate`` so callers (``generate_signals``) can log a
    prediction for EVERY candidate and only trade ``guard_outcome == "pass"``
    ones — matching the project convention that ``predictions.jsonl`` captures
    unconstrained model output before filters.

    Args:
        parsed:        Parsed Kalshi market struct from ``parse_weather_market``.
        M:             Observed daily-HIGH max-so-far from the ASOS feed (°F).
        mu_gain:       Location of the remaining-gain posterior (°F ≥ 0).
        sigma_gain:    Scale of the remaining-gain posterior (°F > 0).
        nu:            Degrees-of-freedom for the Student-t posterior.
        yes_bid:       Best YES bid price (0–1 scale).
        yes_ask:       Best YES ask price (0–1 scale).
        obs_age_hours: Age of the ASOS observation (hours).  Default 0.
        prelim_high:   Optional preliminary NWS HIGH reading for drift check.
        edge_threshold: Threshold stored on the candidate; NOT enforced here.
        guards:        ``AsosGuardConfig`` instance controlling guard thresholds.

    Returns:
        ``AsosCandidate`` with ``guard_outcome`` set in ALL cases, or ``None``
        only when ``bracket_probability`` raises (malformed geometry).
    """
    post_mean_f = M + mu_gain

    # Malformed geometry (e.g. BELOW with cap=None) makes bracket_probability
    # raise; drop the market rather than crash the scan loop.
    try:
        model_prob = bracket_probability(
            parsed.strike_type,
            floor=parsed.floor_strike,
            cap=parsed.cap_strike,
            M=M,
            mu_gain=mu_gain,
            sigma_gain=sigma_gain,
            nu=nu,
        )
    except ValueError:
        logger.debug(
            "%s: bracket_probability raised (malformed geometry) — drop",
            parsed.ticker,
        )
        return None

    prelim_drift_f: float | None = None
    if prelim_high is not None:
        prelim_drift_f = abs(M - prelim_high)

    # Computed before the guards so guard-dropped candidates still carry full
    # pricing fields for the prediction log.
    if model_prob > yes_ask:
        side = Side.YES
        exec_price = yes_ask
    else:
        side = Side.NO
        exec_price = 1.0 - yes_bid

    # No ensemble-confidence discount: the posterior is already sharp (mirrors
    # the EMOS ec=1.0 rule in intraday_emos_high.py).
    if side == Side.YES:
        raw_edge = model_prob - exec_price
    else:
        raw_edge = (1.0 - model_prob) - exec_price
    effective_edge = raw_edge

    # Always computed (NaN for malformed geometry) so guard-dropped candidates
    # still carry full diagnostic fields.
    strike_label = _DIRECTION_TO_STRIKE_LABEL[parsed.strike_type]
    bd = boundary_distance(
        post_mean_f,
        parsed.floor_strike,
        parsed.cap_strike,
        strike_label,
    )

    # Guards run in order; the first failure wins. ``μ_gain ≥ 0`` and
    # ``σ_gain ≥ 0.5`` are deliberately NOT re-checked: the climatology builder
    # marks violators ``source="unavailable"`` so ``ClimatologyTable.lookup``
    # returns ``None`` and the cell never reaches here. Re-checking would mask
    # a lookup regression.
    guard_outcome = "pass"

    if obs_age_hours > guards.max_staleness_h:
        logger.debug(
            "%s: stale observation (age=%.1fh > max=%.1fh) — guard_outcome=stale",
            parsed.ticker,
            obs_age_hours,
            guards.max_staleness_h,
        )
        guard_outcome = "stale"

    if guard_outcome == "pass" and sigma_gain < guards.sigma_floor_f:
        logger.debug(
            "%s: sigma_gain=%.3f < floor=%.3f — guard_outcome=headroom",
            parsed.ticker,
            sigma_gain,
            guards.sigma_floor_f,
        )
        guard_outcome = "headroom"

    # NaN (malformed geometry) is a conservative drop.
    if guard_outcome == "pass" and (math.isnan(bd) or bd < guards.boundary_min_f):
        logger.debug(
            "%s: boundary_distance=%s < min=%.2f — guard_outcome=boundary",
            parsed.ticker,
            f"{bd:.3f}" if not math.isnan(bd) else "NaN",
            guards.boundary_min_f,
        )
        guard_outcome = "boundary"

    if (
        guard_outcome == "pass"
        and guards.bad_prelim_active
        and prelim_high is not None
        and prelim_drift_f is not None
        and prelim_drift_f > guards.bad_prelim_max_drift_f
    ):
        logger.debug(
            "%s: bad-prelim drift=%.1f°F > max=%.1f°F — guard_outcome=bad_prelim",
            parsed.ticker,
            prelim_drift_f,
            guards.bad_prelim_max_drift_f,
        )
        guard_outcome = "bad_prelim"

    # edge_threshold is NOT enforced here: the strategy class applies it after
    # writing audit rows so every evaluated candidate is logged.

    return AsosCandidate(
        parsed=parsed,
        model_prob=model_prob,
        post_mean_f=post_mean_f,
        side=side,
        exec_price=exec_price,
        raw_edge=raw_edge,
        effective_edge=effective_edge,
        guard_outcome=guard_outcome,
        mu_gain=mu_gain,
        sigma_gain=sigma_gain,
        M=M,
        obs_age_hours=obs_age_hours,
        prelim_drift_f=prelim_drift_f,
        boundary_distance_f=bd,
    )


def size_candidate(
    cand: AsosCandidate,
    *,
    bankroll: float,
    kelly_fraction: float,
    max_single_market_pct: float,
    max_contracts_per_trade: int,
) -> tuple[SizingResult, int]:
    """Fractional-Kelly sizing for a guard-passed ASOS candidate.

    Mirrors the clamp ordering from ``intraday_emos_high.py:_apply_depth_adjustment``
    (spec §6.3 / §7.2 MINOR #3):

        1. ``fractional_kelly`` → ``sizing.dollars``
        2. ``dollars_to_contracts`` → raw contract count
        3. Clamp at ``max_contracts_per_trade`` (AFTER Kelly, BEFORE fee math)

    The per-contract fee is computed from ``cand.exec_price`` (conservative
    taker estimate) and passed to ``fractional_kelly`` so Kelly accounts for
    fee drag.

    Args:
        cand:                  Guard-passed ``AsosCandidate``.
        bankroll:              Total capital available (dollars).
        kelly_fraction:        Safety multiplier on raw Kelly fraction.
        max_single_market_pct: Per-market capital cap as fraction of bankroll.
        max_contracts_per_trade: Hard ceiling on contract count (applied after
                                 Kelly, before fee math).

    Returns:
        ``(SizingResult, contracts)`` where ``contracts`` is the clamped integer
        contract count.  If ``SizingResult.side == NONE`` (zero/negative edge),
        ``contracts`` is 0.
    """
    # cand.exec_price is side-denominated (yes_ask for YES, 1-yes_bid for NO).
    # fractional_kelly + dollars_to_contracts both want the YES-space price and
    # convert NO cost internally as (1 - price) — mirror weather_evaluation.py.
    yes_price = cand.exec_price if cand.side == Side.YES else 1.0 - cand.exec_price
    fee = kalshi_fee(cand.exec_price)

    sizing = fractional_kelly(
        model_prob=cand.model_prob,
        market_price=yes_price,
        kelly_fraction=kelly_fraction,
        max_pct=max_single_market_pct,
        bankroll=bankroll,
        fee_per_contract=fee,
    )

    if sizing.side == Side.NONE or sizing.dollars < 1.0:
        return sizing, 0

    contracts = dollars_to_contracts(sizing.dollars, yes_price, sizing.side)

    # Applied after Kelly and before fee math so recorded fees scale with the
    # clamped count.
    if contracts > max_contracts_per_trade:
        logger.debug(
            "%s: contract cap clamp %d → %d (max_contracts_per_trade)",
            cand.parsed.ticker,
            contracts,
            max_contracts_per_trade,
        )
        contracts = max_contracts_per_trade

    return sizing, contracts


# HIGH series only, mirroring ``WEATHER_SERIES`` (weather_edge.py). Inlined, not
# imported, to keep the module hermetic from weather_edge (see module docstring).
# LAX is informational: kept out of _enabled_cells by default, but its series
# still feeds the scan so shadow predictions accrue.
_ASOS_SERIES: dict[str, tuple[str, DailyVariable]] = {
    "KXHIGHMIA": ("MIA", DailyVariable.TEMP_MAX),
    "KXHIGHAUS": ("AUS", DailyVariable.TEMP_MAX),
    "KXHIGHCHI": ("CHI", DailyVariable.TEMP_MAX),
    "KXHIGHNY": ("NYC", DailyVariable.TEMP_MAX),
    "KXHIGHLAX": ("LAX", DailyVariable.TEMP_MAX),  # informational only
}

# Per-cell rate-limit window for the setup-incomplete WARNING.
_SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS = 30 * 60


def _cell_label(city: str, var: str | DailyVariable) -> str:
    """Canonical cell key — e.g. ``'MIA/high'``.

    Accepts either a ``DailyVariable`` enum or its string ``temp_label``
    (``"high"`` / ``"low"``).
    """
    if isinstance(var, DailyVariable):
        var = "low" if var == DailyVariable.TEMP_MIN else "high"
    return f"{city}/{var}"


class IntradayAsosHighStrategy(Strategy):
    """Same-day KXHIGH trading using the ASOS intraday nowcast posterior.

    Architecture mirrors ``IntradayEmosHighStrategy`` but substitutes the
    ASOS posterior (``asos_evaluate``) for the EMOS posterior — no EMOS
    imports, no hot-reload machinery.

    The strategy is hermetic from ``weather_edge`` (no imports of that module).
    Shared primitives live in ``weather_evaluation`` and ``weather_parsing``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        kalshi_client: KalshiClient,
        weather_client: Any | None = None,  # unused; accepted for API symmetry
        bankroll: float = 10_000.0,
        market_filter: dict[str, Any] | None = None,
        *,
        settings: Settings,
        asos_client: AsosClient,
        climatology: ClimatologyTable,
        prediction_log: Any | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        """Construct the intraday ASOS HIGH strategy.

        ``settings``, ``asos_client``, and ``climatology`` are REQUIRED
        keyword-only arguments (no defaults). A forgotten kwarg in a future
        test factory would otherwise silently produce a misconfigured instance.

        Args:
            config:         Strategy config dict (from ``config.yaml``).
            kalshi_client:  Kalshi REST + WebSocket client.
            weather_client: Unused; accepted for API symmetry with other
                            strategies (may be ``None``).
            bankroll:       Current bankroll for Kelly sizing.
            market_filter:  Optional liquidity filter overrides.
            settings:       Project-wide ``Settings`` instance (REQUIRED).
            asos_client:    Injected ``AsosClient`` for ASOS observation
                            fetching (REQUIRED).
            climatology:    Injected ``ClimatologyTable`` for gain statistics
                            (REQUIRED).
            prediction_log: Optional prediction log for calibration audit.
            notifier:       Optional notifier for Telegram alerts.
        """
        super().__init__(name="intraday_asos_high", config=config, bankroll=bankroll)
        self._kalshi = kalshi_client
        self._settings = settings
        self._market_filter = market_filter or {}
        self._asos_client = asos_client
        self._climatology = climatology
        self._prediction_log = prediction_log
        self._notifier: Notifier | None = notifier

        self._enabled_cells: frozenset[str] = frozenset(config.get("enabled_cells", []) or [])
        self._informational_cells: frozenset[str] = frozenset(
            config.get("informational_cells", []) or []
        )

        self._nu: float = float(config.get("posterior_t_dof", 5))

        self._edge_threshold: float = float(config.get("edge_threshold", 0.07))
        self._max_contracts_per_trade: int = int(config.get("max_contracts_per_trade", 50))
        self._max_trades_per_cell_per_day: int = int(config.get("max_trades_per_cell_per_day", 3))

        self._guards = AsosGuardConfig(
            boundary_min_f=float(config.get("boundary_distance_min_f", 1.0)),
            max_staleness_h=float(config.get("asos_max_staleness_hours", 2.0)),
            sigma_floor_f=float(config.get("sigma_gain_floor_f", 0.5)),
            bad_prelim_max_drift_f=float(config.get("bad_prelim_max_drift_f", 10.0)),
            bad_prelim_active=bool(config.get("bad_prelim_active", False)),
        )

        state_dir = Path(self._settings.state_dir)
        self._audit_sinks: AuditSinks = build_sinks(
            state_dir,
            eval_enabled=bool(config.get("eval_audit_log", True)),
            candidate_enabled=bool(config.get("candidate_audit_log", True)),
            funnel_enabled=bool(config.get("scan_funnel_log", True)),
            retention_days=int(config.get("audit_log_retention_days", 14)),
        )

        # Set only after every setup() precondition passes; scan() and
        # generate_signals() return [] until then.
        self._setup_completed: bool = False

        # Setup-incomplete WARNING throttle: first per (process, cell) fires
        # immediately, then at most one per 30 min.
        self._setup_incomplete_warning_state: dict[str, float] = {}

        self._lead_negative_warning_seen_dates: dict[str, set[date]] = {}

        # Reset via reset_daily_counters() at the orchestrator's daily boundary.
        self._trades_today: dict[tuple[str, date], int] = {}

        # Independent of weather_edge's set.
        self._traded_tickers: set[str] = set()

    async def setup(self) -> None:
        """Refuse-to-start preconditions.

        Two checks in order; each raises ``RuntimeError`` on failure.
        The operational "refuse-to-trade" behavior comes from
        ``self._setup_completed`` staying ``False``.

        Preconditions:
          1. ``_informational_cells`` and ``_enabled_cells`` must be disjoint —
             a cell can be in at most one category.
          2. Every cell in ``_enabled_cells | _informational_cells`` must be a
             HIGH cell (label ends ``"/high"``) and must have at least one
             climatology entry for the current UTC month.
        """
        # Precondition 1: enabled/informational must be disjoint.
        overlap = self._enabled_cells & self._informational_cells
        if overlap:
            raise RuntimeError(
                f"intraday_asos_high: cells {sorted(overlap)} appear in both "
                f"'enabled_cells' and 'informational_cells'. Each cell must be "
                f"in at most one category."
            )

        # Precondition 2: every configured cell must be a HIGH cell with
        # climatology coverage.
        all_cells = self._enabled_cells | self._informational_cells
        current_month = datetime.now(UTC).month
        no_coverage: list[str] = []
        not_high: list[str] = []
        for cell in sorted(all_cells):
            if not cell.endswith("/high"):
                not_high.append(cell)
                continue
            city = cell.split("/")[0]
            # Probes a sample of UTC hours; the builder should leave at least one
            # usable hour_pool cell per city/month.
            covered = any(
                self._climatology.lookup(city, current_month, h, var="high") is not None
                for h in range(0, 24, 3)
            )
            if not covered:
                no_coverage.append(cell)

        if not_high:
            raise RuntimeError(
                f"intraday_asos_high: configured cells {sorted(not_high)} are not "
                f"HIGH cells (must end with '/high'). The ASOS strategy is HIGH-only."
            )
        if no_coverage:
            raise RuntimeError(
                f"intraday_asos_high: configured cells {sorted(no_coverage)} have "
                f"no climatology coverage for the current month ({current_month}). "
                f"Run 'scripts/build_asos_climatology.py' to rebuild the artifact."
            )

        self._setup_completed = True
        logger.info(
            "intraday_asos_high setup complete: enabled_cells=%s informational_cells=%s",
            sorted(self._enabled_cells),
            sorted(self._informational_cells),
        )

    async def aclose(self) -> None:
        """No per-strategy resources to release.

        ``KalshiClient``, ``AsosClient``, ``ClimatologyTable``, and
        ``Notifier`` are orchestrator-owned.
        """
        return None

    def reset_daily_counters(self) -> None:
        """Reset per-cell daily trade counters at the orchestrator's daily
        boundary (called from ``RiskManager.reset_daily()`` fan-out).

        Mirrors ``IntradayEmosHighStrategy.reset_daily_counters``.
        """
        self._trades_today = {}
        # Prune the WARNING-dedup state so a long-lived process doesn't leak.
        today = et_today()
        pruned_lead: dict[str, set[date]] = {}
        for cell, dates in self._lead_negative_warning_seen_dates.items():
            kept = {d for d in dates if d >= today}
            if kept:
                pruned_lead[cell] = kept
        self._lead_negative_warning_seen_dates = pruned_lead
        now = time.monotonic()
        self._setup_incomplete_warning_state = {
            cell: ts
            for cell, ts in self._setup_incomplete_warning_state.items()
            if (now - ts) < _SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS
        }

    def mark_traded(self, ticker: str) -> None:
        """Record a ticker as traded so it's skipped in future scans."""
        self._traded_tickers.add(ticker)

    def unmark_traded(self, ticker: str) -> None:
        """Release a ticker so it can be re-considered this session.

        Idempotent: silently no-ops when the ticker isn't in the set.
        """
        self._traded_tickers.discard(ticker)

    def load_state_from_log(self, trade_log_path: Path) -> None:
        """Rehydrate ``_trades_today`` + ``_traded_tickers`` from ``state/trades.jsonl``.

        Without this, a process restart at e.g. 14:00 ET resets the per-cell
        daily cap counter and re-permits trades on tickers that already filled
        and settled earlier in the same ET day.

        Mirrors ``IntradayEmosHighStrategy.load_state_from_log`` but filters
        rows by ``strategy == "intraday_asos_high"``.
        """
        today = et_today()
        seeded_counts = 0
        seeded_tickers = 0
        try:
            rows = list(iter_jsonl(trade_log_path))
        except OSError:
            logger.warning(
                "intraday_asos_high.load_state_from_log: unreadable trade-log %s "
                "— continuing with empty state",
                trade_log_path,
                exc_info=True,
            )
            return

        for row in rows:
            if not is_pnl_relevant_row(row):
                continue
            if str(row.get("strategy", "")) != self._name:
                continue
            ticker = str(row.get("market_ticker", "") or row.get("ticker", ""))
            if not ticker:
                continue
            ticker_date = parse_ticker_date(ticker)
            if ticker_date is None or ticker_date != today:
                continue
            city_var = parse_ticker_city_var(ticker)
            if city_var is None:
                continue
            cell = _cell_label(city_var[0], city_var[1])
            key = (cell, today)
            self._trades_today[key] = self._trades_today.get(key, 0) + 1
            seeded_counts += 1
            if ticker not in self._traded_tickers:
                self._traded_tickers.add(ticker)
                seeded_tickers += 1

        logger.info(
            "intraday_asos_high.load_state_from_log: hydrated %d cap-counter "
            "increment(s) and %d traded-ticker(s) from %s",
            seeded_counts,
            seeded_tickers,
            trade_log_path,
        )

    async def _emit_setup_incomplete_warning(self, cell: str) -> None:
        """Per-cell rate-limited WARNING when setup() failed.

        First emission per (process, cell) fires immediately; subsequent
        emissions on the same cell are throttled to one per 30 min.
        """
        if self._notifier is None:
            return
        now = time.monotonic()
        last = self._setup_incomplete_warning_state.get(cell)
        if last is not None and (now - last) < _SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS:
            return
        self._setup_incomplete_warning_state[cell] = now
        try:
            await self._notifier.send_alert(
                f"intraday_asos_high SKIPPING TICK on cell={cell} — setup() "
                f"did not complete; strategy is registered but not generating "
                f"signals. Inspect orchestrator startup logs for the setup() "
                f"exception.",
                level=AlertLevel.WARNING,
                failure_bucket=f"intraday_asos_setup_incomplete:{cell}",
            )
        except Exception:
            logger.exception(
                "intraday_asos_high setup-incomplete WARNING dispatch failed (cell=%s)",
                cell,
            )

    async def scan(  # type: ignore[override]
        self,
        *,
        scan_id: str | None = None,
        funnel: ScanFunnel | None = None,
        scan_today: date | None = None,
    ) -> list[ScanCandidate]:
        """Fetch open KXHIGH markets; keep same-day rows on enabled+informational cells.

        Past-date rows are rejected outright. Future-date rows (lead_days > 0)
        are silently skipped — ``weather_edge`` owns those.

        Only processes series in ``_ASOS_SERIES`` (HIGH markets only). A
        market is accepted if its canonical cell label is in
        ``_enabled_cells | _informational_cells``.

        Uses the live-Kalshi dict shape where prices are strings; ``_parse_fp_int``
        coerces the fixed-point integer fields as needed.
        """
        all_cells = self._enabled_cells | self._informational_cells
        if not self._enabled:
            logger.info("intraday_asos_high disabled — skipping scan")
            return []
        if not self._setup_completed:
            for cell in sorted(all_cells):
                await self._emit_setup_incomplete_warning(cell)
            return []

        scan_today = scan_today if scan_today is not None else et_today()
        utc_today = datetime.now(UTC).date()

        candidates: list[ScanCandidate] = []
        total_raw = 0
        skipped_past = 0
        skipped_future = 0

        # Liquidity filter knobs (mirrors weather_edge defaults).
        from python.analytics.audit_log import RejectionReason

        min_volume = self._market_filter.get("min_volume", 0)
        min_oi = self._market_filter.get("min_open_interest", 0)
        min_price = self._market_filter.get("min_price", 0.0)
        max_price = self._market_filter.get("max_price", 1.0)

        for i, (series_ticker, (city_code, daily_var)) in enumerate(_ASOS_SERIES.items()):
            if i > 0:
                # Stagger requests to avoid Kalshi's 429 rate limit.
                await asyncio.sleep(1.0)

            try:
                raw_markets, _truncated = await self._kalshi.iter_all_markets(
                    status="open",
                    series_ticker=series_ticker,
                )
            except Exception:
                logger.exception(
                    "intraday_asos_high: failed to fetch markets for %s",
                    series_ticker,
                )
                continue

            total_raw += len(raw_markets)
            if funnel is not None:
                funnel.n_raw += len(raw_markets)

            for market in raw_markets:
                parsed = parse_weather_market(market, city_code, daily_var)
                if parsed is None:
                    continue

                cell = _cell_label(parsed.city, parsed.temp_label)
                if cell not in all_cells:
                    continue

                if parsed.forecast_date < scan_today:
                    skipped_past += 1
                    if funnel is not None:
                        funnel.increment(RejectionReason.PAST_DATE)
                    continue

                # weather_edge owns future-date markets.
                if parsed.forecast_date > scan_today:
                    skipped_future += 1
                    continue

                # WARN once per (cell, UTC date) when UTC has already passed
                # forecast_date (same as intraday_emos_high).
                lead_days_raw = (parsed.forecast_date - utc_today).days
                if lead_days_raw < 0:
                    seen = self._lead_negative_warning_seen_dates.setdefault(cell, set())
                    if utc_today not in seen:
                        seen.add(utc_today)
                        logger.warning(
                            "intraday_asos_high lead_days_raw=%d on cell=%s "
                            "ticker=%s — treating as same-day. "
                            "ET=%s UTC=%s",
                            lead_days_raw,
                            cell,
                            parsed.ticker,
                            scan_today.isoformat(),
                            utc_today.isoformat(),
                        )

                if min_volume > 0:
                    vol = _parse_fp_int(market.get("volume_fp"))
                    if vol is not None and vol < min_volume:
                        if funnel is not None:
                            funnel.increment(RejectionReason.VOLUME)
                        continue
                if min_oi > 0:
                    oi = _parse_fp_int(market.get("open_interest_fp"))
                    if oi is not None and oi < min_oi:
                        if funnel is not None:
                            funnel.increment(RejectionReason.OPEN_INTEREST)
                        continue

                price_info = extract_market_price_info(market)
                if (min_price > 0.0 or max_price < 1.0) and price_info is not None:
                    if price_info.midpoint < min_price or price_info.midpoint > max_price:
                        if funnel is not None:
                            funnel.increment(RejectionReason.PRICE_GATE)
                        continue

                candidates.append(
                    ScanCandidate(
                        market=market,
                        parsed=parsed,
                        price_info=price_info,
                    )
                )

        logger.info(
            "intraday_asos_high: %d tradeable markets out of %d total "
            "(%d past_date, %d future_date skipped)",
            len(candidates),
            total_raw,
            skipped_past,
            skipped_future,
        )
        return candidates

    async def evaluate(
        self,
        market: dict[str, Any] | ScanCandidate,
    ) -> float | None:
        """Compute the model probability for a single market.

        Fetches the ASOS observation and climatology lookup, then calls
        ``asos_evaluate``.  Returns the model_prob or ``None`` on failure.
        """
        if isinstance(market, ScanCandidate):
            parsed = market.parsed
        else:
            parsed = market.get("_parsed")  # type: ignore[assignment]
            if parsed is None:
                return None

        now_utc = datetime.now(UTC)
        obs = await self._asos_client.max_so_far(
            city=parsed.city,
            local_day=parsed.forecast_date,
        )
        if obs is None:
            return None

        clim = self._climatology.lookup(
            parsed.city,
            parsed.forecast_date.month,
            now_utc.hour,
            var="high",
        )
        if clim is None:
            return None
        mu_gain, sigma_gain, _n, _source = clim

        cand = asos_evaluate(
            parsed,
            M=obs.M,
            mu_gain=mu_gain,
            sigma_gain=sigma_gain,
            nu=self._nu,
            yes_bid=0.0,  # not used for probability only
            yes_ask=1.0,
            obs_age_hours=obs.age_hours,
            guards=self._guards,
        )
        if cand is None:
            return None
        return cand.model_prob

    async def generate_signals(self) -> list[TradeSignal]:
        """Full scan → evaluate → size → signal pipeline.

        Mirrors ``IntradayEmosHighStrategy.generate_signals`` but substitutes
        the ASOS posterior (``asos_evaluate``) for the EMOS posterior path.
        """
        if not self._enabled:
            return []
        if not self._setup_completed:
            for cell in sorted(self._enabled_cells | self._informational_cells):
                await self._emit_setup_incomplete_warning(cell)
            return []

        scan_id = uuid.uuid4().hex[:12]
        scan_ts = datetime.now(tz=UTC).timestamp()
        funnel = ScanFunnel()
        scan_today: date = et_today()

        # Day-rollover defense-in-depth (mirrors intraday_emos_high's
        # self-cleaning of stale per-cell trade counters).
        # Self-clean stale (cell, date) entries so an orphaned counter can't
        # silently brick the strategy if reset_daily_counters() is missed.
        self._trades_today = {
            key: count for key, count in self._trades_today.items() if key[1] == scan_today
        }

        markets = await self.scan(scan_id=scan_id, funnel=funnel, scan_today=scan_today)
        if not markets:
            self._emit_funnel(funnel, scan_id, scan_ts, n_markets=0)
            return []

        # ── Phase 1 (per cell): fetch ASOS observation + climatology ─────────
        # Group markets by cell so we fetch obs once per cell, not per market.
        now_utc = datetime.now(UTC)
        utc_hour = now_utc.hour

        # cell → (M, age_hours, mu_gain, sigma_gain, climo_n, climo_source)
        cell_inputs: dict[str, tuple[float, float, float, float, int, str]] = {}

        # Collect distinct cells from the scan result.
        cell_markets: dict[str, list[ScanCandidate]] = {}
        for sc in markets:
            cell = _cell_label(sc.parsed.city, sc.parsed.temp_label)
            cell_markets.setdefault(cell, []).append(sc)

        # ── Batched ASOS prefetch (Stage 0b: collapse N requests → 1) ─────────
        # Issue ONE IEM request per distinct local_day covering ALL the tick's
        # distinct cities, warming the AsosClient per-(station, local_day) cache.
        # The per-cell ``max_so_far`` calls below then hit the warm cache and
        # make ZERO additional requests within the TTL window. This is the
        # N→1 collapse that prevents the per-station 429 storm at scale
        # (2026-06-06 incident at 5 stations; guaranteed at any expansion).
        # Fail-open: a batch fetch error leaves the cache cold and each per-cell
        # ``max_so_far`` falls back to its own (single-station) fetch, so
        # behavior degrades to exactly today's pre-batch path — never worse.
        cities_by_day: dict[date, set[str]] = {}
        for cell_scs in cell_markets.values():
            rep = cell_scs[0].parsed
            cities_by_day.setdefault(rep.forecast_date, set()).add(rep.city)
        for fdate, day_cities in cities_by_day.items():
            try:
                await self._asos_client.max_so_far_batch(
                    cities=sorted(day_cities),
                    local_day=fdate,
                )
            except Exception:
                # max_so_far_batch is itself fail-open; this is belt-and-suspenders.
                logger.warning(
                    "intraday_asos_high: batched ASOS prefetch raised for day=%s "
                    "cities=%s — falling back to per-cell fetch",
                    fdate,
                    sorted(day_cities),
                    exc_info=True,
                )

        for cell, cell_scs in cell_markets.items():
            rep = cell_scs[0].parsed
            city = rep.city
            forecast_date = rep.forecast_date

            # Fail-open: None means no observation.
            try:
                obs = await self._asos_client.max_so_far(
                    city=city,
                    local_day=forecast_date,
                )
            except Exception:
                logger.warning(
                    "intraday_asos_high: ASOS fetch raised for cell=%s — skipping cell",
                    cell,
                    exc_info=True,
                )
                obs = None

            if obs is None:
                logger.debug(
                    "intraday_asos_high: no ASOS obs for cell=%s — skipping cell",
                    cell,
                )
                continue

            clim = self._climatology.lookup(
                city,
                forecast_date.month,
                utc_hour,
                var="high",
            )
            if clim is None:
                for sc in cell_scs:
                    funnel.increment(RejectionReason.ASOS_CLIMO_UNAVAILABLE)
                    self._emit_eval_reject(
                        sc,
                        RejectionReason.ASOS_CLIMO_UNAVAILABLE,
                        scan_id=scan_id,
                        scan_ts=scan_ts,
                    )
                logger.warning(
                    "intraday_asos_high: no climatology for cell=%s month=%d hour=%d",
                    cell,
                    forecast_date.month,
                    utc_hour,
                )
                continue

            mu_gain, sigma_gain, climo_n, climo_source = clim

            cell_inputs[cell] = (
                obs.M,
                obs.age_hours,
                mu_gain,
                sigma_gain,
                climo_n,
                climo_source,
            )

        # Stage 1: collect depth-check candidates (guard-passed and edge-passed).
        candidates_for_depth: list[
            tuple[ScanCandidate, AsosCandidate, float, float, float, int, str]
        ] = []
        batch_tickers: set[str] = set()

        for sc in markets:
            ticker = sc.parsed.ticker
            if ticker in self._traded_tickers or ticker in batch_tickers:
                continue

            cell = _cell_label(sc.parsed.city, sc.parsed.temp_label)
            if cell not in cell_inputs:
                # Already logged in phase 1.
                continue

            asos_m, age_hours, mu_gain, sigma_gain, climo_n, climo_source = cell_inputs[cell]

            price_info = sc.price_info
            if price_info is None or price_info.yes_bid is None or price_info.yes_ask is None:
                funnel.increment(RejectionReason.NO_PRICE)
                self._emit_eval_reject(
                    sc, RejectionReason.NO_PRICE, scan_id=scan_id, scan_ts=scan_ts
                )
                continue

            yes_bid = float(price_info.yes_bid)
            yes_ask = float(price_info.yes_ask)

            cand = asos_evaluate(
                sc.parsed,
                M=asos_m,
                mu_gain=mu_gain,
                sigma_gain=sigma_gain,
                nu=self._nu,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                obs_age_hours=age_hours,
                prelim_high=None,
                edge_threshold=self._edge_threshold,
                guards=self._guards,
            )
            if cand is None:
                logger.debug(
                    "intraday_asos_high: asos_evaluate returned None for %s (malformed geometry)",
                    ticker,
                )
                continue

            # Logged before any filter: the forward calibration corpus must be
            # unselected.
            self._record_prediction(
                cand=cand,
                sc=sc,
                scan_id=scan_id,
                scan_today=scan_today,
                climo_n=climo_n,
                climo_source=climo_source,
            )

            guard_to_reason = {
                "boundary": RejectionReason.ASOS_BOUNDARY_GUARD,
                "stale": RejectionReason.ASOS_STALE_OBS,
                "headroom": RejectionReason.ASOS_ZERO_HEADROOM,
                "bad_prelim": RejectionReason.ASOS_BAD_PRELIM,
            }
            if cand.guard_outcome != "pass":
                reason = guard_to_reason.get(
                    cand.guard_outcome, RejectionReason.ASOS_BOUNDARY_GUARD
                )
                funnel.increment(reason)
                self._emit_eval_reject(
                    sc,
                    reason,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                    extra={
                        "guard_outcome": cand.guard_outcome,
                        "asos_M": cand.M,
                        "obs_age_hours": cand.obs_age_hours,
                        "model_prob": cand.model_prob,
                        "boundary_distance_f": cand.boundary_distance_f,
                    },
                )
                continue

            if cand.effective_edge < self._edge_threshold:
                funnel.increment(RejectionReason.EDGE_BELOW_THRESHOLD)
                self._emit_eval_reject(
                    sc,
                    RejectionReason.EDGE_BELOW_THRESHOLD,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                    extra={
                        "effective_edge": cand.effective_edge,
                        "model_prob": cand.model_prob,
                        "boundary_distance_f": cand.boundary_distance_f,
                    },
                )
                continue

            # Floor applies outside shadow mode only.
            exec_mode = str(self._settings.execution.mode)
            min_cal_floor: float | None = None
            if exec_mode != ExecutionMode.SHADOW.value:
                raw_floor = self._config.get("min_calibrated_prob")
                if raw_floor is not None:
                    try:
                        min_cal_floor = float(raw_floor)
                    except (TypeError, ValueError):
                        pass

            if min_cal_floor is not None and cand.model_prob < min_cal_floor:
                funnel.increment(RejectionReason.CAL_FLOOR_BLOCK)
                self._emit_eval_reject(
                    sc,
                    RejectionReason.CAL_FLOOR_BLOCK,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                    extra={
                        "model_prob": cand.model_prob,
                        "min_calibrated_prob": min_cal_floor,
                    },
                )
                continue

            kelly_fraction = float(
                self._config.get("kelly_fraction", self._settings.risk.kelly_fraction)
            )
            max_single_market_pct = float(
                self._config.get("max_single_market_pct", self._settings.risk.max_single_market_pct)
            )
            sizing, contracts = size_candidate(
                cand,
                bankroll=self._bankroll,
                kelly_fraction=kelly_fraction,
                max_single_market_pct=max_single_market_pct,
                max_contracts_per_trade=self._max_contracts_per_trade,
            )
            if sizing.side == Side.NONE or contracts == 0:
                funnel.increment(RejectionReason.KELLY_ZERO)
                self._emit_eval_reject(
                    sc, RejectionReason.KELLY_ZERO, scan_id=scan_id, scan_ts=scan_ts
                )
                continue

            batch_tickers.add(ticker)
            candidates_for_depth.append(
                (sc, cand, asos_m, mu_gain, sigma_gain, climo_n, climo_source)
            )

        # Stage 2: concurrent orderbook depth check.
        _orderbook_concurrency = int(self._config.get("orderbook_concurrency", 2))
        _depth_semaphore = asyncio.Semaphore(_orderbook_concurrency)

        kelly_fraction = float(
            self._config.get("kelly_fraction", self._settings.risk.kelly_fraction)
        )
        max_single_market_pct = float(
            self._config.get("max_single_market_pct", self._settings.risk.max_single_market_pct)
        )

        _depth_outcome_t = tuple[
            ScanCandidate,
            AsosCandidate,
            tuple[float, SizingResult, int, bool, DepthCheck | None] | None,
        ]

        async def _run_depth(
            sc: ScanCandidate,
            cand: AsosCandidate,
        ) -> _depth_outcome_t:
            async with _depth_semaphore:
                result = await self._apply_depth_adjustment(
                    sc,
                    cand,
                    kelly_fraction=kelly_fraction,
                    max_single_market_pct=max_single_market_pct,
                    funnel=funnel,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                )
            return sc, cand, result

        depth_outcomes: list[_depth_outcome_t] = await asyncio.gather(
            *[_run_depth(sc, cand) for sc, cand, *_ in candidates_for_depth]
        )

        # n_signals_fired counts pre-cap, as in intraday_emos_high.
        pre_cap_signals: list[TradeSignal] = []
        for sc, cand, result in depth_outcomes:
            if result is None:
                continue
            effective_edge, sizing, contracts, depth_capped, depth_result = result

            signal = self._build_signal(
                sc=sc,
                cand=cand,
                effective_edge=effective_edge,
                sizing=sizing,
                contracts=contracts,
                depth_capped=depth_capped,
                depth_result=depth_result,
                scan_today=scan_today,
            )
            pre_cap_signals.append(signal)
            funnel.n_signals_fired += 1

        collapsed_signals, n_capped = _select_one_per_event(pre_cap_signals, enabled=True)
        if n_capped:
            funnel.n_event_capped += n_capped

        # Informational cells are logged but never traded, regardless of
        # shadow_only.
        live_signals: list[TradeSignal] = []
        for signal in collapsed_signals:
            cell = str(signal.metadata.get("cell", ""))
            if cell in self._informational_cells:
                funnel.n_informational_cell_drop += 1
                logger.debug(
                    "intraday_asos_high: dropping informational-cell signal %s (cell=%s)",
                    signal.market_ticker,
                    cell,
                )
                continue
            live_signals.append(signal)

        # Applied after collapse so the counter measures emissions
        # (intraday_emos_high §9.2(b)). It increments before the orchestrator's
        # shadow-drop, so in shadow_only mode it counts would-have-traded
        # emissions by design, not placed orders.
        signals: list[TradeSignal] = []
        for signal in live_signals:
            cell = str(signal.metadata.get("cell", ""))
            cap_key = (cell, scan_today)
            already = self._trades_today.get(cap_key, 0)
            if already >= self._max_trades_per_cell_per_day:
                logger.info(
                    "intraday_asos_high: per-cell daily cap hit (cell=%s n=%d)"
                    " — dropping candidate %s",
                    cell,
                    already,
                    signal.market_ticker,
                )
                funnel.increment(RejectionReason.TRADES_PER_CELL_CAP)
                continue
            self._trades_today[cap_key] = already + 1
            signals.append(signal)

        logger.info(
            "intraday_asos_high: generated %d signal(s) from %d market(s) "
            "(%d event-capped, %d informational-dropped)",
            len(signals),
            len(markets),
            n_capped,
            funnel.n_informational_cell_drop,
        )
        self._emit_funnel(funnel, scan_id, scan_ts, n_markets=len(markets))
        return signals

    def _emit_funnel(
        self,
        funnel: ScanFunnel,
        scan_id: str,
        scan_ts: float,
        *,
        n_markets: int,
    ) -> None:
        """Write one ``scan_funnel.jsonl`` row. Swallows all errors."""
        if self._audit_sinks.funnel_log is None:
            return
        try:
            self._audit_sinks.funnel_log.emit_funnel(
                funnel,
                scan_id,
                extra={
                    "strategy": self._name,
                    "scan_ts": scan_ts,
                    "n_markets": n_markets,
                },
            )
        except Exception:
            logger.exception("scan_funnel emission failed for scan_id=%s", scan_id)

    def _emit_eval_reject(
        self,
        sc: ScanCandidate,
        reason: RejectionReason,
        *,
        scan_id: str,
        scan_ts: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Emit one ``eval_audit.jsonl`` row for a rejected candidate.

        Swallows all errors — a failed emit must never break signal generation.
        """
        if self._audit_sinks.eval_log is None:
            return
        try:
            price_info = sc.price_info
            row: dict[str, Any] = {
                "ts": datetime.now(tz=UTC).isoformat(),
                "scan_id": scan_id,
                "scan_ts": scan_ts,
                "strategy": self._name,
                "ticker": sc.parsed.ticker,
                "city": sc.parsed.city,
                "var": sc.parsed.temp_label,
                "forecast_date": sc.parsed.forecast_date.isoformat(),
                "strike_type": sc.parsed.strike_type.value,
                "floor": sc.parsed.floor_strike,
                "cap": sc.parsed.cap_strike,
                "yes_bid": price_info.yes_bid if price_info else None,
                "yes_ask": price_info.yes_ask if price_info else None,
                "mid": price_info.midpoint if price_info else None,
                "reason": reason.value,
                "reason_numeric": reason_numeric(reason),
            }
            if extra:
                row.update(extra)
            self._audit_sinks.eval_log.emit_eval(row)
        except Exception:
            logger.exception(
                "eval_audit emission failed for %s reason=%s",
                sc.parsed.ticker,
                reason,
            )

    def _record_prediction(
        self,
        *,
        cand: AsosCandidate,
        sc: ScanCandidate,
        scan_id: str,
        scan_today: date,
        climo_n: int,
        climo_source: str,
    ) -> None:
        """Log a candidate to predictions.jsonl for calibration.

        Fires for EVERY non-None candidate (pass AND guard-dropped) BEFORE
        any filter — the forward calibration corpus is the whole point.
        Fire-and-forget — never breaks signal generation.
        """
        if self._prediction_log is None:
            return
        try:
            price_info = sc.price_info
            market_price = price_info.midpoint if price_info else float("nan")
            exec_price_val: float | None = None
            if not math.isnan(cand.exec_price):
                exec_price_val = cand.exec_price

            rec = PredictionRecord(
                timestamp=datetime.now(tz=UTC),
                strategy=self._name,
                market_ticker=cand.parsed.ticker,
                side=cand.side,
                city=cand.parsed.city,
                daily_variable=cand.parsed.temp_label,
                strike_type=_DIRECTION_TO_STRIKE_LABEL.get(cand.parsed.strike_type),
                floor_strike=(
                    float(cand.parsed.floor_strike)
                    if cand.parsed.floor_strike is not None
                    else None
                ),
                cap_strike=(
                    float(cand.parsed.cap_strike) if cand.parsed.cap_strike is not None else None
                ),
                expiration_date=cand.parsed.forecast_date,
                raw_ensemble_prob=None,  # ASOS has no raw ensemble fraction
                calibrated_prob=cand.model_prob,
                market_price=market_price if not math.isnan(market_price) else 0.0,
                exec_price=exec_price_val,
                confidence=1.0,  # ASOS posterior is already sharp; no ensemble confidence
                isotonic_fit_date=None,
                gfs_prob=None,
                ecmwf_prob=None,
                ensemble_std_pooled=None,
                is_same_day_market=True,  # ASOS strategy is always same-day
                live_calibrator_kind="asos_posterior",
                ens_mean=None,
                ens_std=None,
                emos_prob_raw=None,
                # Schema v6: ASOS-specific fields
                asos_M=cand.M,
                asos_obs_age_h=cand.obs_age_hours,
                mu_gain=cand.mu_gain,
                sigma_gain=cand.sigma_gain,
                climo_n=climo_n,
                climo_source=climo_source,
                posterior_p_raw=cand.model_prob,
                boundary_distance_f=cand.boundary_distance_f,
                post_mean_f=cand.post_mean_f,
                guard_outcome=cand.guard_outcome,
                prelim_drift_f=cand.prelim_drift_f,
                # Schema v7: this row's M / climatology come from the LST
                # (fixed standard-time) climate-day frame (§0a). Frame 1 = legacy
                # civil/DST frame; rows logged after the frame fix are frame 2.
                frame_version=2,
            )
            self._prediction_log.append(rec)
        except Exception:
            logger.exception(
                "intraday_asos_high prediction_log write failed for %s",
                cand.parsed.ticker,
            )

    def _build_signal(
        self,
        *,
        sc: ScanCandidate,
        cand: AsosCandidate,
        effective_edge: float,
        sizing: SizingResult,
        contracts: int,
        depth_capped: bool,
        depth_result: DepthCheck | None,
        scan_today: date | None = None,
    ) -> TradeSignal:
        """Construct the TradeSignal with all per-signal metadata fields."""
        _today = scan_today if scan_today is not None else et_today()
        utc_today = datetime.now(UTC).date()
        lead_days_raw = (cand.parsed.forecast_date - utc_today).days
        lead_days = max(lead_days_raw, 0)
        cell = _cell_label(cand.parsed.city, cand.parsed.temp_label)
        price_info = sc.price_info

        return TradeSignal(
            market_id=sc.market.get("ticker", ""),
            market_ticker=sc.market.get("ticker", ""),
            model_probability=cand.model_prob,
            market_price=price_info.midpoint if price_info else 0.0,
            confidence=1.0,  # ASOS posterior is already sharp
            strategy_name=self._name,
            side=sizing.side,
            suggested_dollars=sizing.dollars,
            suggested_contracts=contracts,
            effective_edge=effective_edge,
            mid_at_signal=price_info.midpoint if price_info else float("nan"),
            exec_price=cand.exec_price,
            metadata={
                "strategy_name": self._name,
                "cell": cell,
                "city": cand.parsed.city,
                "forecast_date": cand.parsed.forecast_date.isoformat(),
                "lead_days": lead_days,
                "lead_days_raw": lead_days_raw,
                "market_type": cand.parsed.temp_label,
                "strike_type": _DIRECTION_TO_STRIKE_LABEL.get(cand.parsed.strike_type),
                "floor_strike": cand.parsed.floor_strike,
                "cap_strike": cand.parsed.cap_strike,
                "description": cand.parsed.description,
                # Pricing
                "yes_bid": price_info.yes_bid if price_info else None,
                "yes_ask": price_info.yes_ask if price_info else None,
                "bid_ask_spread": price_info.spread if price_info else None,
                "midpoint": price_info.midpoint if price_info else None,
                "mid_at_signal": price_info.midpoint if price_info else None,
                "exec_price": cand.exec_price,
                # Edge
                "raw_edge": cand.raw_edge,
                "effective_edge_stage1": cand.effective_edge,
                "fill_edge_stage2": effective_edge,
                "effective_edge": effective_edge,
                # ASOS posterior fields
                "asos_M": cand.M,
                "asos_obs_age_h": cand.obs_age_hours,
                "mu_gain": cand.mu_gain,
                "sigma_gain": cand.sigma_gain,
                "post_mean_f": cand.post_mean_f,
                "posterior_p_raw": cand.model_prob,
                "boundary_distance_f": cand.boundary_distance_f,
                "guard_outcome": cand.guard_outcome,
                # Sizing
                "kelly_raw": sizing.kelly_raw,
                "kelly_capped": sizing.capped,
                "depth_capped": depth_capped,
                "fill_price": (depth_result.fill_price if depth_result else None),
                "fill_slippage": (depth_result.slippage if depth_result else None),
                "book_depth": (depth_result.available_contracts if depth_result else None),
                "book_levels": (depth_result.total_levels if depth_result else None),
                "is_same_day_market": cand.parsed.forecast_date == _today,
                "event_candidates_count": 1,
                "event_dropped_edges": [],
                "event_dropped_tickers": [],
            },
        )

    async def _apply_depth_adjustment(
        self,
        sc: ScanCandidate,
        cand: AsosCandidate,
        *,
        kelly_fraction: float,
        max_single_market_pct: float,
        funnel: ScanFunnel | None = None,
        scan_id: str | None = None,
        scan_ts: float | None = None,
    ) -> tuple[float, SizingResult, int, bool, DepthCheck | None] | None:
        """Stage-2 orderbook depth check + VWAP recompute + max_contracts clamp.

        Mirrors ``IntradayEmosHighStrategy._apply_depth_adjustment`` but uses
        ``ec=1.0`` (no ensemble-confidence discount — ASOS posterior is already
        sharp, same as EMOS). Returns ``(effective_edge, sizing, contracts,
        depth_capped, depth_result)`` on pass, else ``None``.
        """
        sizing, contracts = size_candidate(
            cand,
            bankroll=self._bankroll,
            kelly_fraction=kelly_fraction,
            max_single_market_pct=max_single_market_pct,
            max_contracts_per_trade=self._max_contracts_per_trade,
        )
        if sizing.side == Side.NONE or contracts == 0:
            if funnel is not None:
                funnel.increment(RejectionReason.KELLY_ZERO)
            return None

        effective_edge = cand.effective_edge
        depth_capped = False
        depth_result: DepthCheck | None = None
        opp_view: OpponentSideView | None = None
        fill_fee_final: float | None = None
        fill_edge_final: float | None = None

        depth_check_enabled = bool(self._config.get("orderbook_depth_check", True))

        def _emit_candidate_row(
            reason: RejectionReason,
            fill_edge: float | None = None,
            fill_fee: float | None = None,
        ) -> None:
            if scan_id is None or self._audit_sinks.candidate_log is None:
                return
            try:
                price_info = sc.price_info
                row: dict[str, Any] = {
                    "ts": datetime.now(tz=UTC).isoformat(),
                    "scan_id": scan_id,
                    "scan_ts": scan_ts,
                    "strategy": self._name,
                    "ticker": cand.parsed.ticker,
                    "city": cand.parsed.city,
                    "var": cand.parsed.temp_label,
                    "forecast_date": cand.parsed.forecast_date.isoformat(),
                    "strike_type": cand.parsed.strike_type.value,
                    "floor": cand.parsed.floor_strike,
                    "cap": cand.parsed.cap_strike,
                    "yes_bid": price_info.yes_bid if price_info else None,
                    "yes_ask": price_info.yes_ask if price_info else None,
                    "mid": price_info.midpoint if price_info else None,
                    "exec_price": cand.exec_price,
                    "model_prob": cand.model_prob,
                    "asos_M": cand.M,
                    "mu_gain": cand.mu_gain,
                    "sigma_gain": cand.sigma_gain,
                    "post_mean_f": cand.post_mean_f,
                    "boundary_distance_f": cand.boundary_distance_f,
                    "guard_outcome": cand.guard_outcome,
                    "raw_edge": cand.raw_edge,
                    "effective_edge": effective_edge,
                    "fill_edge": fill_edge,
                    "fill_price": depth_result.fill_price if depth_result else None,
                    "slippage": depth_result.slippage if depth_result else None,
                    "fill_fee": fill_fee,
                    "kelly_dollars": sizing.dollars,
                    "kelly_raw": sizing.kelly_raw,
                    "kelly_capped": sizing.capped,
                    "contracts": contracts,
                    "depth_capped": depth_capped,
                    "side": str(sizing.side),
                    "our_side_available": (
                        depth_result.available_contracts if depth_result else None
                    ),
                    "our_side_levels": depth_result.total_levels if depth_result else None,
                    "opp_side_available": opp_view.available if opp_view else None,
                    "opp_side_levels": opp_view.levels if opp_view else None,
                    "reason": reason.value,
                    "reason_numeric": reason_numeric(reason),
                }
                self._audit_sinks.candidate_log.emit_candidate(row)
            except Exception:
                logger.exception(
                    "candidate_audit emission failed for %s",
                    cand.parsed.ticker,
                )

        if depth_check_enabled:
            depth_result, opp_view = await self._check_orderbook_depth(
                cand.parsed.ticker,
                sizing.side,
                contracts,
            )
            if depth_result is not None:
                if depth_result.is_stale:
                    logger.warning(
                        "%s: orderbook too thin (available=%d, levels=%d) — skipping",
                        cand.parsed.ticker,
                        depth_result.available_contracts,
                        depth_result.total_levels,
                    )
                    if funnel is not None:
                        funnel.n_orderbook_thin += 1
                    _emit_candidate_row(RejectionReason.ORDERBOOK_THIN)
                    return None

                fill_fee = kalshi_fee(depth_result.fill_price)
                ec = 1.0
                if sizing.side == Side.YES:
                    fill_edge = (cand.model_prob - depth_result.fill_price) * ec - fill_fee
                else:
                    fill_edge = ((1.0 - cand.model_prob) - depth_result.fill_price) * ec - fill_fee

                fill_fee_final = fill_fee
                fill_edge_final = fill_edge

                max_fill_slippage = self._config.get("max_fill_slippage")
                if max_fill_slippage is not None and depth_result.slippage > float(
                    max_fill_slippage
                ):
                    if funnel is not None:
                        funnel.n_fill_slippage += 1
                    _emit_candidate_row(
                        RejectionReason.FILL_SLIPPAGE,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                if fill_edge < self._edge_threshold:
                    if funnel is not None:
                        funnel.n_fill_edge_low += 1
                    _emit_candidate_row(
                        RejectionReason.FILL_EDGE_NEGATIVE,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None

                effective_edge = fill_edge

                fill_market_price = (
                    depth_result.fill_price
                    if sizing.side == Side.YES
                    else 1.0 - depth_result.fill_price
                )
                pre_depth_side = sizing.side
                sizing = fractional_kelly(
                    model_prob=cand.model_prob,
                    market_price=fill_market_price,
                    kelly_fraction=kelly_fraction,
                    max_pct=max_single_market_pct,
                    bankroll=self._bankroll,
                    fee_per_contract=fill_fee,
                )
                if sizing.side != Side.NONE and sizing.side != pre_depth_side:
                    logger.critical(
                        "%s: Kelly side flipped post-depth (%s -> %s); "
                        "fill_price=%.4f model_prob=%.4f — skipping",
                        cand.parsed.ticker,
                        pre_depth_side,
                        sizing.side,
                        depth_result.fill_price,
                        cand.model_prob,
                    )
                    if funnel is not None:
                        funnel.n_kelly_side_flip += 1
                    _emit_candidate_row(
                        RejectionReason.KELLY_SIDE_FLIP,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                if sizing.side == Side.NONE or sizing.dollars < 1.0:
                    if funnel is not None:
                        funnel.n_fill_kelly_zero += 1
                    _emit_candidate_row(
                        RejectionReason.KELLY_ZERO,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                contracts = dollars_to_contracts(
                    sizing.dollars,
                    fill_market_price,
                    sizing.side,
                )

                max_take_pct = float(self._market_filter.get("max_orderbook_take_pct", 1.0))
                if max_take_pct < 1.0:
                    max_book_contracts = max(
                        int(depth_result.available_contracts * max_take_pct), 1
                    )
                else:
                    max_book_contracts = depth_result.available_contracts
                if contracts > max_book_contracts:
                    contracts = max_book_contracts
                    depth_capped = True

        # Fallback: best-level depth cap when depth check is disabled or API fails.
        if depth_result is None and self._market_filter.get("max_orderbook_take_pct", 1.0) < 1.0:
            price_info = sc.price_info
            max_take_pct = float(self._market_filter.get("max_orderbook_take_pct", 1.0))
            depth = (
                (price_info.yes_ask_size if sizing.side == Side.YES else price_info.yes_bid_size)
                if price_info
                else None
            )
            if depth is not None and depth > 0:
                max_contracts_best = max(int(depth * max_take_pct), 1)
                if contracts > max_contracts_best:
                    contracts = max_contracts_best
                    depth_capped = True

        if contracts > self._max_contracts_per_trade:
            logger.info(
                "%s: contract cap clamp %d -> %d (max_contracts_per_trade)",
                cand.parsed.ticker,
                contracts,
                self._max_contracts_per_trade,
            )
            contracts = self._max_contracts_per_trade
            depth_capped = True

        per_contract_price: float | None = None
        if depth_result is not None and depth_result.fill_price > 0.0:
            per_contract_price = (
                depth_result.fill_price
                if sizing.side == Side.YES
                else 1.0 - depth_result.fill_price
            )
        else:
            price_info = sc.price_info
            if price_info is not None:
                best_price = price_info.yes_ask if sizing.side == Side.YES else price_info.yes_bid
                if best_price is not None:
                    per_contract_price = best_price if sizing.side == Side.YES else 1.0 - best_price
                elif price_info.midpoint > 0.0:
                    per_contract_price = (
                        price_info.midpoint
                        if sizing.side == Side.YES
                        else 1.0 - price_info.midpoint
                    )
        if contracts > 0 and per_contract_price is not None and per_contract_price > 0.0:
            sizing = SizingResult(
                side=sizing.side,
                dollars=float(contracts) * per_contract_price,
                kelly_raw=sizing.kelly_raw,
                kelly_fraction_used=sizing.kelly_fraction_used,
                capped=sizing.capped,
            )

        _ = opp_view  # captured for informational audit; silences unused warning

        _emit_candidate_row(
            RejectionReason.PASSED,
            fill_edge=fill_edge_final,
            fill_fee=fill_fee_final,
        )
        return effective_edge, sizing, contracts, depth_capped, depth_result

    async def _check_orderbook_depth(
        self,
        ticker: str,
        side: Side,
        target_contracts: int,
    ) -> tuple[DepthCheck | None, OpponentSideView | None]:
        """Fetch the full orderbook and compute realistic fill cost.

        Mirrors ``IntradayEmosHighStrategy._check_orderbook_depth``.
        """
        try:
            data = await self._kalshi.get_orderbook(
                ticker, depth=self._config.get("book_depth_levels", 10)
            )
        except Exception:
            logger.warning(
                "intraday_asos_high: failed to fetch orderbook for %s",
                ticker,
                exc_info=True,
            )
            return None, None

        orderbook = data.get("orderbook_fp", {})
        if side == Side.YES:
            opposite_bids = orderbook.get("no_dollars", [])
            exit_bids = orderbook.get("yes_dollars", [])
        else:
            opposite_bids = orderbook.get("yes_dollars", [])
            exit_bids = orderbook.get("no_dollars", [])

        ask_levels: list[tuple[float, int]] = []
        for level in opposite_bids:
            parsed_level = _parse_orderbook_level(level)
            if parsed_level is None:
                continue
            opp_price, qty = parsed_level
            our_price = _orderbook_price_to_probability(opp_price)
            if our_price <= 0.0 or our_price >= 1.0:
                continue
            ask_levels.append((our_price, qty))
        ask_levels.sort(key=lambda x: x[0])

        opp_available = 0
        opp_levels_n = 0
        opp_best_price: float | None = None
        for level in exit_bids:
            parsed_level = _parse_orderbook_level(level)
            if parsed_level is None:
                continue
            price_prob, qty = parsed_level
            if price_prob <= 0.0 or price_prob >= 1.0:
                continue
            opp_available += qty
            opp_levels_n += 1
            if opp_best_price is None or price_prob > opp_best_price:
                opp_best_price = price_prob

        opp_view = OpponentSideView(
            available=opp_available,
            levels=opp_levels_n,
            best_price=opp_best_price,
        )

        min_depth = self._config.get("min_book_depth", 5)
        depth_check = compute_fill_from_levels(ask_levels, target_contracts, min_depth)
        return depth_check, opp_view


# Hermetic from weather_edge: do NOT import python.strategies.weather_edge here.
__all__ = [
    "AsosGuardConfig",
    "AsosCandidate",
    "asos_evaluate",
    "size_candidate",
    "IntradayAsosHighStrategy",
]
