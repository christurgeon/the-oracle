"""P&L attribution for settled trades.

Decomposes each settled trade's net P&L into four additive components so we
can tell, over any window, how much of the P&L came from each source:

  net_pnl = predicted_alpha + model_error - slippage_cost - fees

where (all in side-denominated space, YES or NO depending on which side we bought):

  side_outcome   = 1 if our side won at settlement else 0
  side_prob      = our model's probability that our side wins
  side_mid       = midpoint price for our side at signal time
  side_exec      = price we actually paid per contract (``fill_price``, or the
                   ``dollars/contracts`` VWAP — NOT ``exec_price``, which is the
                   signal-time ask we submitted as a limit, not the execution)
  fee            = Kalshi order fee (already computed by settlement pipeline)

Then:

  predicted_alpha = (side_prob    - side_mid)  * contracts   # edge at mid
  slippage_cost   = (side_exec    - side_mid)  * contracts   # cost over mid
  model_error     = (side_outcome - side_prob) * contracts   # realized - expected
  fees            = fee

Check: (outcome - prob)*C + (prob - mid)*C - (exec - mid)*C - fee
     = outcome*C - exec*C - fee   ≡   (outcome - exec)*C - fee   ≡   net_pnl  ✓

If a historical record lacks ``mid_at_signal`` (records written before the
field was added), we fall back to a degraded two-way decomposition that rolls
``predicted_alpha`` and ``slippage_cost`` into a single ``predicted_alpha_at_exec``
term. Callers can detect this via ``TradeAttribution.decomposition_mode``.

This module is pure: no I/O, no global state. Callers pass in dicts (from
``data/shadow_settlements.full.json``) and get dataclasses back.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from python.common import Side
from python.utils.ticker_dates import parse_ticker_city_var

logger = logging.getLogger(__name__)

# Tolerance for the decomposition sum identity, in dollars.
#
# Was 0.005 on the premise that "the identity holds to the last cent
# deterministically". That premise assumed ``side_exec`` was a cent-rounded
# price. Since 2026-09-01 it is the realised fill VWAP, which is genuinely
# sub-cent on partial fills across differing prices (e.g. 0.4437 over 7
# contracts), while ``recorded_pnl`` stays cent-rounded — so a residue of up to
# ~1c is correct arithmetic, not a math error. Real defects found by this check
# have been $0.03-$2.66, two to three orders of magnitude above the threshold,
# so 0.011 loses no detection power and stops three permanent false alarms.
RECONCILIATION_TOLERANCE = 0.011


@dataclass(frozen=True)
class TradeAttribution:
    """Per-trade P&L decomposition."""

    order_id: str
    timestamp: datetime | None
    strategy: str
    market_ticker: str
    side: Side
    contracts: int
    side_prob: float
    side_mid: float | None  # None for degraded records
    side_exec: float
    side_outcome: int
    fees: float

    predicted_alpha: float  # (side_prob - side_mid) * C, or NaN in degraded mode
    slippage_cost: float  # (side_exec - side_mid) * C, or NaN in degraded mode
    predicted_alpha_at_exec: float  # (side_prob - side_exec) * C — always defined
    model_error: float  # (side_outcome - side_prob) * C
    net_pnl: float  # decomposition total; reconciled against record["pnl"]
    recorded_pnl: float  # as persisted in settlements file
    reconciliation_diff: float  # net_pnl - recorded_pnl

    # Enrichment from signal metadata (best-effort; None if missing)
    city: str | None
    daily_variable: str | None
    ensemble_std: float | None
    isotonic_calibrated: bool | None

    decomposition_mode: str  # "full" | "degraded" | "recorded_only"


@dataclass(frozen=True)
class AttributionSummary:
    """Aggregated totals across a collection of trades."""

    n: int
    wins: int
    losses: int
    total_contracts: int
    total_wagered: float
    total_net_pnl: float  # decomposition total; diverges from recorded by reconciliation_diff
    total_recorded_pnl: float  # sum of record["pnl"] — realized P&L; use for headlines
    total_predicted_alpha: float  # full-mode trades only
    total_slippage_cost: float  # full-mode trades only
    total_predicted_alpha_at_exec: float  # decomposable trades only (recorded-only rows excluded)
    total_model_error: float  # decomposable trades only (recorded-only rows excluded)
    total_fees: float
    total_reconciliation_diff: float
    n_full_mode: int
    n_degraded_mode: int  # includes recorded_only (any non-full row)
    n_recorded_only: int  # subset of n_degraded_mode that carries no decomposition at all
    total_recorded_only_pnl: float  # recorded P&L of the recorded_only rows (not decomposable)

    @property
    def roi_pct(self) -> float:
        """ROI on the *decomposition* P&L. Prefer :pyattr:`roi_recorded_pct` for reporting."""
        if self.total_wagered <= 0:
            return 0.0
        return 100.0 * self.total_net_pnl / self.total_wagered

    @property
    def roi_recorded_pct(self) -> float:
        """ROI on recorded (realized) P&L — the correct headline ROI."""
        if self.total_wagered <= 0:
            return 0.0
        return 100.0 * self.total_recorded_pnl / self.total_wagered

    @property
    def win_rate_pct(self) -> float:
        if self.n == 0:
            return 0.0
        return 100.0 * self.wins / self.n


# ---------------------------------------------------------------------------
# Per-record attribution
# ---------------------------------------------------------------------------


def _infer_city(record: dict[str, Any]) -> str | None:
    md = record.get("metadata") or {}
    city = md.get("city") if isinstance(md, dict) else None
    if city:
        return str(city)
    ticker = record.get("market_ticker", "") or ""
    city_var = parse_ticker_city_var(ticker)
    return city_var[0] if city_var is not None else None


def _infer_daily_variable(record: dict[str, Any]) -> str | None:
    md = record.get("metadata") or {}
    if isinstance(md, dict):
        label = md.get("market_type") or md.get("daily_variable")
        if label:
            return str(label)
    # Fallback from ticker prefix: KXHIGH* → high, KXLOWT* → low.
    ticker = record.get("market_ticker", "") or ""
    if ticker.startswith("KXHIGH"):
        return "high"
    if ticker.startswith("KXLOWT"):
        return "low"
    return None


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Convert to float, treating None and NaN as missing."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def attribute_trade(record: dict[str, Any]) -> TradeAttribution | None:
    """Decompose a single settled trade record.

    Returns ``None`` if the record is unsettled, malformed, or missing the
    fields required for decomposition.
    """
    if not record.get("settled"):
        return None

    try:
        side_str = str(record["side"]).upper()
        side = Side(side_str)
    except (KeyError, ValueError):
        logger.debug("attribution: skipping record with invalid side: %r", record.get("order_id"))
        return None
    if side not in (Side.YES, Side.NO):
        return None

    # Core settled-trade fields. These define the realized outcome and recorded
    # P&L, so they are genuinely required — a record missing any of them is
    # malformed. ``model_probability`` and ``order_price`` are NOT here: they are
    # optional decomposition inputs (see recorded-only mode below).
    try:
        contracts = int(record["contracts"])
        recorded_pnl = float(record["pnl"])
        fee = float(record["fee"])
        won = bool(record["won"])
    except (KeyError, TypeError, ValueError):
        logger.debug(
            "attribution: skipping record with missing/invalid core fields: %r",
            record.get("order_id"),
        )
        return None

    side_outcome = 1 if won else 0

    # Metadata enrichment — shared by the decomposable and recorded-only paths.
    md = record.get("metadata") or {}
    ensemble_std = _to_float(md.get("ensemble_std_dev")) if isinstance(md, dict) else None
    isotonic_flag: bool | None = None
    if isinstance(md, dict) and "isotonic_calibrated" in md:
        isotonic_flag = bool(md["isotonic_calibrated"])

    # ``fill_price`` is what we actually paid and takes priority. ``exec_price``
    # is deliberately NOT consulted: it records the signal-time ASK (the limit we
    # submitted), not the execution. On the live corpus it disagreed with the
    # realised VWAP on 409 of 707 fill rows (58%) — e.g. order 793df686 shows
    # exec_price=0.59 against a 0.21 fill, a price the trade never paid — and was
    # the sole source of the "P&L reconciliation off" warnings. Using it produces
    # silently-wrong attribution, so a missing price degrades to recorded-only
    # mode instead. ``dollars/contracts`` is the VWAP and so stays correct across
    # partial fills at differing prices; it also covers pre-2026-04-27 rows
    # written before ``fill_price`` existed. nan only if no price is recoverable.
    order_price = _to_float(record.get("order_price"))
    fill_price_raw = _to_float(record.get("fill_price"))
    dollars = _to_float(record.get("dollars"))
    if fill_price_raw is not None:
        side_exec = fill_price_raw
    elif dollars is not None and contracts:
        side_exec = dollars / contracts
    elif order_price is not None:
        side_exec = order_price
    else:
        side_exec = float("nan")

    model_prob_yes = _to_float(record.get("model_probability"))

    # Recorded-only mode: reconciler-backfilled synthetic rows carry no
    # model_probability (and sometimes no price). They cannot be decomposed, but
    # their recorded P&L and win-flag are authoritative — so include them with
    # the decomposition fields marked unavailable rather than dropping them.
    # Dropping previously caused by_strategy to silently undercount both trade
    # counts and P&L (see post-mortem 2026-05-30, order 8db990c0). net_pnl is
    # pinned to recorded_pnl so reconciliation_diff is exactly zero.
    if model_prob_yes is None or not math.isfinite(side_exec):
        return TradeAttribution(
            order_id=str(record.get("order_id", "")),
            timestamp=_parse_timestamp(record.get("timestamp")),
            strategy=str(record.get("strategy", "")),
            market_ticker=str(record.get("market_ticker", "")),
            side=side,
            contracts=contracts,
            side_prob=float("nan"),
            side_mid=None,
            side_exec=side_exec,
            side_outcome=side_outcome,
            fees=fee,
            predicted_alpha=float("nan"),
            slippage_cost=float("nan"),
            predicted_alpha_at_exec=float("nan"),
            model_error=float("nan"),
            net_pnl=recorded_pnl,
            recorded_pnl=recorded_pnl,
            reconciliation_diff=0.0,
            city=_infer_city(record),
            daily_variable=_infer_daily_variable(record),
            ensemble_std=ensemble_std,
            isotonic_calibrated=isotonic_flag,
            decomposition_mode="recorded_only",
        )

    side_prob = model_prob_yes if side == Side.YES else 1.0 - model_prob_yes

    # ``mid_at_signal`` is always YES-space by convention.
    mid_yes = _to_float(record.get("mid_at_signal"))
    if mid_yes is None:
        # Legacy records wrote YES midpoint into market_price when exec wasn't overloaded;
        # we cannot distinguish the two cases, so fall back to degraded mode.
        side_mid: float | None = None
    else:
        side_mid = mid_yes if side == Side.YES else 1.0 - mid_yes

    model_error = (side_outcome - side_prob) * contracts
    predicted_alpha_at_exec = (side_prob - side_exec) * contracts
    # Identity: net_pnl == (side_outcome - side_exec)*C - fee
    #                    == model_error + predicted_alpha_at_exec - fee
    net_pnl = (side_outcome - side_exec) * contracts - fee

    if side_mid is not None:
        predicted_alpha = (side_prob - side_mid) * contracts
        slippage_cost = (side_exec - side_mid) * contracts
        decomposition_mode = "full"
    else:
        predicted_alpha = float("nan")
        slippage_cost = float("nan")
        decomposition_mode = "degraded"

    reconciliation_diff = net_pnl - recorded_pnl
    if abs(reconciliation_diff) > RECONCILIATION_TOLERANCE:
        logger.warning(
            "attribution: P&L reconciliation off for %s: decomposed=%.4f recorded=%.4f diff=%.4f "
            "(contracts=%d exec=%.4f outcome=%d fee=%.4f)",
            record.get("order_id"),
            net_pnl,
            recorded_pnl,
            reconciliation_diff,
            contracts,
            side_exec,
            side_outcome,
            fee,
        )

    return TradeAttribution(
        order_id=str(record.get("order_id", "")),
        timestamp=_parse_timestamp(record.get("timestamp")),
        strategy=str(record.get("strategy", "")),
        market_ticker=str(record.get("market_ticker", "")),
        side=side,
        contracts=contracts,
        side_prob=side_prob,
        side_mid=side_mid,
        side_exec=side_exec,
        side_outcome=side_outcome,
        fees=fee,
        predicted_alpha=predicted_alpha,
        slippage_cost=slippage_cost,
        predicted_alpha_at_exec=predicted_alpha_at_exec,
        model_error=model_error,
        net_pnl=net_pnl,
        recorded_pnl=recorded_pnl,
        reconciliation_diff=reconciliation_diff,
        city=_infer_city(record),
        daily_variable=_infer_daily_variable(record),
        ensemble_std=ensemble_std,
        isotonic_calibrated=isotonic_flag,
        decomposition_mode=decomposition_mode,
    )


def attribute_many(records: Iterable[dict[str, Any]]) -> list[TradeAttribution]:
    """Decompose a batch of settled trade records, skipping unsettled/malformed ones."""
    out: list[TradeAttribution] = []
    for rec in records:
        attr = attribute_trade(rec)
        if attr is not None:
            out.append(attr)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(attrs: Iterable[TradeAttribution]) -> AttributionSummary:
    items = list(attrs)
    n = len(items)
    wins = sum(1 for a in items if a.side_outcome == 1)
    losses = n - wins
    total_contracts = sum(a.contracts for a in items)
    # NaN-safe: recorded-only rows may carry a non-finite side_exec.
    total_wagered = sum(a.side_exec * a.contracts for a in items if math.isfinite(a.side_exec))
    total_net_pnl = sum(a.net_pnl for a in items)
    total_recorded_pnl = sum(a.recorded_pnl for a in items)
    total_fees = sum(a.fees for a in items)
    # Decomposition subtotals exclude recorded-only rows (NaN model_error /
    # predicted_alpha_at_exec), so they are NaN-safe and reflect only rows we
    # could actually decompose.
    total_model_error = sum(a.model_error for a in items if math.isfinite(a.model_error))
    total_predicted_alpha_at_exec = sum(
        a.predicted_alpha_at_exec for a in items if math.isfinite(a.predicted_alpha_at_exec)
    )
    total_recon_diff = sum(a.reconciliation_diff for a in items)

    # Full-mode-only subtotals — NaN-safe: degraded items have NaN here.
    full_items = [a for a in items if a.decomposition_mode == "full"]
    total_predicted_alpha = sum(a.predicted_alpha for a in full_items)
    total_slippage_cost = sum(a.slippage_cost for a in full_items)

    # Recorded-only rows carry no decomposition at all (no model_probability /
    # price). Tracked separately so reports can show a "not decomposable" line
    # that closes the gap between the recorded headline and the decomposition.
    recorded_only_items = [a for a in items if a.decomposition_mode == "recorded_only"]
    n_recorded_only = len(recorded_only_items)
    total_recorded_only_pnl = sum(a.recorded_pnl for a in recorded_only_items)

    return AttributionSummary(
        n=n,
        wins=wins,
        losses=losses,
        total_contracts=total_contracts,
        total_wagered=total_wagered,
        total_net_pnl=total_net_pnl,
        total_recorded_pnl=total_recorded_pnl,
        total_predicted_alpha=total_predicted_alpha,
        total_slippage_cost=total_slippage_cost,
        total_predicted_alpha_at_exec=total_predicted_alpha_at_exec,
        total_model_error=total_model_error,
        total_fees=total_fees,
        total_reconciliation_diff=total_recon_diff,
        n_full_mode=len(full_items),
        n_degraded_mode=n - len(full_items),
        n_recorded_only=n_recorded_only,
        total_recorded_only_pnl=total_recorded_only_pnl,
    )


def by_side(
    attrs: Iterable[TradeAttribution],
) -> dict[Side, AttributionSummary]:
    buckets: dict[Side, list[TradeAttribution]] = defaultdict(list)
    for a in attrs:
        buckets[a.side].append(a)
    return {side: summarize(items) for side, items in buckets.items()}


def by_city(
    attrs: Iterable[TradeAttribution],
) -> dict[str, AttributionSummary]:
    buckets: dict[str, list[TradeAttribution]] = defaultdict(list)
    for a in attrs:
        key = a.city or "unknown"
        buckets[key].append(a)
    return {city: summarize(items) for city, items in buckets.items()}


def by_strategy(
    attrs: Iterable[TradeAttribution],
) -> dict[str, AttributionSummary]:
    buckets: dict[str, list[TradeAttribution]] = defaultdict(list)
    for a in attrs:
        buckets[a.strategy or "unknown"].append(a)
    return {name: summarize(items) for name, items in buckets.items()}


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------


class AttributionEngine:
    """Thin facade that reads the settlement file and produces attribution reports.

    Stateless: each call re-reads the source file. Cheap enough for ad-hoc
    Telegram queries; the settlement file is a few hundred KB.
    """

    def __init__(self, settlements_path: str) -> None:
        self._path = settlements_path

    def _load(self) -> list[dict[str, Any]]:
        path = Path(self._path)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.exception("attribution: failed to read %s", path)
            return []
        return data if isinstance(data, list) else []

    def attributions(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[TradeAttribution]:
        records = self._load()
        attrs = attribute_many(records)
        if since is not None:
            attrs = [a for a in attrs if a.timestamp is not None and a.timestamp >= since]
        if until is not None:
            attrs = [a for a in attrs if a.timestamp is not None and a.timestamp <= until]
        return attrs

    def summary(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> AttributionSummary:
        return summarize(self.attributions(since=since, until=until))

    def report(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a structured report suitable for formatters / JSON export."""
        attrs = self.attributions(since=since, until=until)
        return {
            "window_start": since.isoformat() if since else None,
            "window_end": until.isoformat() if until else None,
            "overall": summarize(attrs),
            "by_side": by_side(attrs),
            "by_city": by_city(attrs),
            "by_strategy": by_strategy(attrs),
        }
