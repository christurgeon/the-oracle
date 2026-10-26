"""
Kelly criterion position sizing for binary prediction markets.

Implements fractional Kelly sizing for both YES and NO sides, with
configurable safety fraction and per-market caps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from python.common import Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    """Result of a Kelly sizing calculation."""

    side: Side
    dollars: float
    kelly_raw: float  # full Kelly fraction (before safety multiplier)
    kelly_fraction_used: float  # fractional Kelly multiplier applied
    capped: bool  # whether the position was capped at max_pct


def same_day_correlation_haircut(n_existing_bets: int, rho: float) -> float:
    """Multiplicative haircut applied to a fresh Kelly dollar size when
    ``n_existing_bets`` correlated bets are already open on the same trading
    day, assuming a pairwise outcome correlation of ``rho``.

    Derivation
    ----------
    Independent fractional-Kelly stacks variances additively: ``n`` bets of
    variance ``σ²`` aggregate to ``n·σ²``. With a constant pairwise
    correlation ``ρ ≥ 0``, the aggregate variance is
    ``n·σ² + n·(n-1)·ρ·σ² = n·σ²·(1 + (n-1)·ρ)``.  Going from ``n`` to ``n+1``
    bets, the *marginal* aggregate-variance ratio relative to the independent
    case is approximately ``1 + n·ρ`` (one new bet correlated with each of the
    ``n`` existing ones).  Shrinking the new bet by ``1/√(1 + n·ρ)`` restores
    the variance budget per the Markowitz square-root rule.

    Boundary conditions:
      - ``n_existing_bets == 0`` → 1.0 (first bet of the day; legacy behavior).
      - ``rho <= 0`` → 1.0 (zero or *negative* correlation provides no
        variance penalty — by symmetry it would justify *increasing* size,
        but we conservatively decline to do so).
      - ``rho == 0`` → 1.0 even for large ``n`` (degenerate independence).
      - ``n_existing_bets < 0`` → 1.0 (defensive; caller bug).

    Pure function, no side effects.

    **Scope:** ``n_existing_bets`` is supplied by the caller. In the production
    weather_edge pipeline (``WeatherEdgeStrategy._same_day_haircut_args``) the
    count comes from ``RiskManager.count_same_day_bets``, which sees only
    positions already *open in the risk-manager state* — i.e. fills emitted by
    PRIOR scans. Within a single scan, multiple candidates evaluated in
    parallel each see the same pre-scan count, so the haircut is **inter-scan
    only, not intra-scan**. At default ``rho == 0.0`` this is moot. If ρ is
    activated, a future enhancement could add an in-scan counter — open
    follow-up if/when ρ is set non-zero.

    See the F4.1 empirical-ρ analysis for the study that motivated this function.
    """
    if n_existing_bets <= 0:
        return 1.0
    if not math.isfinite(rho) or rho <= 0.0:
        return 1.0
    return 1.0 / math.sqrt(1.0 + n_existing_bets * rho)


def fractional_kelly(
    model_prob: float,
    market_price: float,
    kelly_fraction: float = 0.15,
    max_pct: float = 0.05,
    bankroll: float = 10_000.0,
    fee_per_contract: float = 0.0,
    *,
    n_existing_same_day_bets: int = 0,
    correlation_rho: float | None = None,
) -> SizingResult:
    """
    Calculate position size using fractional Kelly criterion.

    For binary prediction markets where contracts pay $1 on YES or $0:
      - If model_prob > market_price -> buy YES at market_price
      - If model_prob < market_price -> buy NO  at (1 - market_price)

    Args:
        model_prob: Model's estimated probability of YES outcome (0-1).
        market_price: Current market YES price (0-1).
        kelly_fraction: Safety multiplier applied to raw Kelly (default 0.15).
        max_pct: Maximum position as fraction of bankroll (default 0.05 = 5%).
        bankroll: Total capital available.
        fee_per_contract: Per-contract exchange fee in dollars (0-1 scale).
            Added to the buy price so Kelly accounts for fee drag.
        n_existing_same_day_bets: Number of trades already opened on the same
            trading day at the time this bet is being sized. Used together
            with ``correlation_rho`` to apply
            :func:`same_day_correlation_haircut` to the dollar size. Defaults
            to 0 (no haircut, legacy behavior).
        correlation_rho: Assumed pairwise outcome correlation across same-day
            trades. ``None`` disables the haircut entirely (legacy
            independent-Kelly behavior). ``0.0`` is also a no-op (independent
            bets). Empirically estimated from
            ``data/shadow_settlements.full.json``; see F4.1 follow-up.

    Returns:
        SizingResult with side, dollar amount, and metadata.
    """
    # Reject non-finite inputs explicitly — NaN slips past the [0, 1] clamp
    # below (NaN comparisons all return False, and `min`/`max` on NaN is
    # order-dependent in CPython) and would silently size a max-conviction
    # trade on poisoned data.
    if not math.isfinite(model_prob) or not math.isfinite(market_price):
        logger.warning(
            "fractional_kelly received non-finite inputs (model_prob=%r, "
            "market_price=%r) — returning NONE (upstream bug?)",
            model_prob,
            market_price,
        )
        return SizingResult(
            side=Side.NONE,
            dollars=0.0,
            kelly_raw=0.0,
            kelly_fraction_used=kelly_fraction,
            capped=False,
        )

    # Clamp model_prob to valid range — out-of-bounds values would produce
    # max-size trades via inflated Kelly fractions.
    if not (0.0 <= model_prob <= 1.0):
        logger.warning(
            "model_prob %.4f outside [0, 1] — clamping (upstream bug?)",
            model_prob,
        )
        model_prob = max(0.0, min(1.0, model_prob))
    if not (0.0 < market_price < 1.0):
        logger.warning(
            "market_price %.4f outside (0, 1) — returning NONE",
            market_price,
        )
        return SizingResult(
            side=Side.NONE,
            dollars=0.0,
            kelly_raw=0.0,
            kelly_fraction_used=kelly_fraction,
            capped=False,
        )

    edge = abs(model_prob - market_price)
    if edge < 0.001:
        return SizingResult(
            side=Side.NONE,
            dollars=0.0,
            kelly_raw=0.0,
            kelly_fraction_used=kelly_fraction,
            capped=False,
        )

    if model_prob > market_price:
        side = Side.YES
        price = market_price + fee_per_contract
        win_prob = model_prob
    else:
        side = Side.NO
        price = (1.0 - market_price) + fee_per_contract
        win_prob = 1.0 - model_prob

    # Odds: payout-to-cost ratio.  Pay `price`, win `1 - price` net.
    if price <= 0 or price >= 1:
        return SizingResult(
            side=Side.NONE,
            dollars=0.0,
            kelly_raw=0.0,
            kelly_fraction_used=kelly_fraction,
            capped=False,
        )

    odds = (1.0 - price) / price
    kelly_raw = (win_prob * odds - (1.0 - win_prob)) / odds
    kelly_raw = max(kelly_raw, 0.0)

    position_dollars = kelly_raw * kelly_fraction * bankroll

    # Same-day correlation haircut (F4.1). Applied BEFORE the per-market cap
    # so the cap remains a hard ceiling regardless of correlation state, and
    # AFTER the kelly_fraction safety multiplier so the haircut composes
    # multiplicatively with the existing fractional-Kelly discount.
    if correlation_rho is not None and n_existing_same_day_bets > 0:
        haircut = same_day_correlation_haircut(n_existing_same_day_bets, correlation_rho)
        position_dollars *= haircut

    max_dollars = max_pct * bankroll
    capped = position_dollars > max_dollars
    final_dollars = min(position_dollars, max_dollars)

    return SizingResult(
        side=side,
        dollars=round(final_dollars, 2),
        kelly_raw=round(kelly_raw, 6),
        kelly_fraction_used=kelly_fraction,
        capped=capped,
    )


def binary_contract_pnl(
    side: Side,
    fill_price: float,
    contracts: int,
    result: str,
    total_fee: float = 0.0,
) -> float:
    """Compute realized P&L for a settled binary contract position.

    Args:
        side: Which side was bought (YES or NO).
        fill_price: Price paid per contract (0-1, side-denominated).
        contracts: Number of contracts held.
        result: Settlement result from Kalshi API ("yes" or "no").
        total_fee: Total exchange fee for the order in dollars.
            Deducted from P&L regardless of outcome (paid at entry).

    Returns:
        Realized P&L in dollars (positive = profit, negative = loss).
    """
    won = (side == Side.YES and result == "yes") or (side == Side.NO and result == "no")
    if won:
        return (1.0 - fill_price) * contracts - total_fee
    return -(fill_price * contracts) - total_fee


def dollars_to_contracts(dollars: float, market_price: float, side: Side) -> int:
    """
    Convert a dollar amount to an integer number of contracts.

    Args:
        dollars: Amount in dollars to invest.
        market_price: Current YES price on the exchange (0-1).
        side: Which side to buy (YES or NO).

    Returns:
        Number of contracts (at least 1 if dollars > 0).
    """
    if side == Side.NONE:
        return 0
    price = market_price if side == Side.YES else (1.0 - market_price)
    if price <= 0:
        return 0
    return max(1, int(dollars / price)) if dollars > 0 else 0
