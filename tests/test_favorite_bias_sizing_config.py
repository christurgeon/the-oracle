from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "python" / "config.yaml"


def _favorite_bias_config() -> dict:
    with _CONFIG.open() as fh:
        return yaml.safe_load(fh)["strategies"]["favorite_bias"]


def test_favorite_bias_sizing_within_pilot_bounds() -> None:
    """Guards against a silent re-doubling of favorite_bias per-trade size.

    kelly 0.36 behaved like ~0.7 effective Kelly on a 2/4-scale-gate edge
    (right side of the Kelly curve). Rolled back 2026-07-03. Editing these
    thresholds must be a DELIBERATE, reviewed size-up.

    Ceiling raised again 2026-08-28 (size-up #10, kelly 0.20->0.30, max_contracts
    25->50) as an explicit operator override of the unmet scale gate — the pooled
    day-clustered ROI CI still included 0 at the time. Not a claim the gate cleared;
    see the kelly_fraction comment in config.yaml for the full rationale and the
    revert trigger.

    Raised again 2026-09-14 (size-up #11, kelly 0.30->0.60, max_contracts 50->100),
    in lockstep with risk.max_single_market_pct 0.05->0.10 — operator directive to
    double per-trade size. Same override status; revert trigger in config.yaml.

    Reads config.yaml directly (the operator-edited artifact) rather than the
    full Settings object, which requires a readable .env not available in every
    environment. These keys are not env-overridden in practice.
    """
    fb = _favorite_bias_config()
    assert fb["kelly_fraction"] <= 0.60, (
        f"favorite_bias.kelly_fraction={fb['kelly_fraction']} exceeds the reviewed "
        "pilot ceiling 0.60 — a size-up past this must be a deliberate override"
    )
    assert fb["max_contracts_per_trade"] <= 100, (
        f"favorite_bias.max_contracts_per_trade={fb['max_contracts_per_trade']} exceeds "
        "the reviewed pilot ceiling 100 — a size-up past this must be a deliberate override"
    )


def _risk_config() -> dict:
    with _CONFIG.open() as fh:
        return yaml.safe_load(fh)["risk"]


def test_risk_side_caps_within_reviewed_bounds() -> None:
    """Guards the knobs that ACTUALLY set per-trade size.

    The 2026-09-06 audit found that no ``strategies.favorite_bias`` knob was
    binding: ``RiskManager.check_signal`` sets every position from
    ``risk.max_single_market_pct x effective_max_total_exposure_pct x bankroll``,
    halved again while ``drawdown >= risk.max_drawdown_pct``. Size-ups #9 and #10
    moved the favorite_bias knobs and changed realized size by +0.7%. So the
    existing favorite_bias guardrail above cannot, on its own, catch a real
    size-up — these are the values that can.
    """
    risk = _risk_config()
    assert risk["max_single_market_pct"] <= 0.10, (
        f"risk.max_single_market_pct={risk['max_single_market_pct']} exceeds the "
        "reviewed bound 0.10 (size-up #11, 2026-09-14). This is the TRUE per-trade size knob, not "
        "strategies.favorite_bias.max_pct — raising it is a real size-up"
    )
    assert risk["max_daily_loss_pct"] <= 0.15, (
        f"risk.max_daily_loss_pct={risk['max_daily_loss_pct']} exceeds the reviewed "
        "bound 0.15 — the daily halt is the backstop on a day of correlated losses"
    )
    assert risk["max_daily_loss_pct"] < risk["max_drawdown_pct"], (
        "risk.max_daily_loss_pct must stay below max_drawdown_pct (settings.py "
        "validates this; asserted here so config.yaml fails fast in CI)"
    )
    assert risk["drawdown_size_reduction"] >= 0.5, (
        f"risk.drawdown_size_reduction={risk['drawdown_size_reduction']} is below the "
        "reviewed 0.50. This is the halving that has governed every live position; "
        "lowering it is a size-up wearing a risk-control costume"
    )
