"""Tests for ``IntradayAsosHighStrategyConfig`` Pydantic model.

Covers:
  (a) ``enabled=True`` + empty ``enabled_cells`` → ``ValidationError``
  (b) A non-HIGH cell (e.g. ``"MIA/low"``) → ``ValidationError``
  (c) Overlap between ``informational_cells`` and ``enabled_cells`` → ``ValidationError``
  (d) ``shadow_only`` defaults to ``True``
  (e) A valid config loads without error

Also covers:
  - ``informational_cells`` null-coercion to ``[]``
  - Deduplication in enabled_cells / informational_cells
  - Field-level validators (cell format)
  - ``intraday_asos_high`` field attached to ``StrategiesConfig``
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.settings import (
    IntradayAsosHighStrategyConfig,
    StrategiesConfig,
)

# ── Invariant (a): enabled=True requires non-empty enabled_cells ─────────────


def test_enabled_true_with_empty_enabled_cells_raises() -> None:
    """enabled=True + no enabled_cells is a silent misconfiguration → reject."""
    with pytest.raises(ValidationError, match="enabled_cells"):
        IntradayAsosHighStrategyConfig(enabled=True, enabled_cells=[])


def test_enabled_true_with_non_empty_enabled_cells_passes() -> None:
    """enabled=True + at least one valid cell should not raise."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True, shadow_only=True, enabled_cells=["MIA/high"]
    )
    assert cfg.enabled is True
    assert cfg.enabled_cells == ["MIA/high"]


def test_enabled_false_with_empty_enabled_cells_passes() -> None:
    """enabled=False + empty enabled_cells is the safe dormant default."""
    cfg = IntradayAsosHighStrategyConfig(enabled=False)
    assert cfg.enabled_cells == []


# ── Invariant (b): all cells must end with /high ──────────────────────────────


def test_non_high_cell_in_enabled_cells_raises() -> None:
    """A low cell in enabled_cells must be rejected (HIGH-only strategy)."""
    with pytest.raises(ValidationError, match="HIGH-only|high"):
        IntradayAsosHighStrategyConfig(enabled_cells=["MIA/low"])


def test_non_high_cell_in_informational_cells_raises() -> None:
    """A low cell in informational_cells must also be rejected."""
    with pytest.raises(ValidationError, match="HIGH-only|high"):
        IntradayAsosHighStrategyConfig(informational_cells=["NYC/low"])


def test_high_cells_pass_both_lists() -> None:
    """Valid /high cells in both lists should not raise."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True,
        enabled_cells=["MIA/high", "AUS/high"],
        informational_cells=["LAX/high"],
    )
    assert "MIA/high" in cfg.enabled_cells
    assert "LAX/high" in cfg.informational_cells


# ── Invariant (c): enabled_cells ∩ informational_cells must be empty ─────────


def test_overlap_between_enabled_and_informational_raises() -> None:
    """A cell in both lists is ambiguous routing → reject."""
    with pytest.raises(ValidationError, match="both enabled_cells and informational_cells"):
        IntradayAsosHighStrategyConfig(
            enabled=True,
            enabled_cells=["MIA/high"],
            informational_cells=["MIA/high"],
        )


def test_disjoint_enabled_and_informational_passes() -> None:
    """Non-overlapping lists should pass."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True,
        enabled_cells=["MIA/high"],
        informational_cells=["LAX/high"],
    )
    assert set(cfg.enabled_cells).isdisjoint(set(cfg.informational_cells))


# ── Invariant (d): shadow_only defaults True ──────────────────────────────────


def test_shadow_only_defaults_true() -> None:
    """The default value for shadow_only must be True (fail-safe)."""
    cfg = IntradayAsosHighStrategyConfig()
    assert cfg.shadow_only is True


def test_shadow_only_can_be_set_false() -> None:
    """shadow_only=False is accepted (explicit operator decision)."""
    cfg = IntradayAsosHighStrategyConfig(shadow_only=False)
    assert cfg.shadow_only is False


# ── Invariant (e): valid config loads ────────────────────────────────────────


def test_valid_full_config_loads() -> None:
    """A complete valid config matching the config.yaml block should load."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True,
        shadow_only=True,
        enabled_cells=["MIA/high", "AUS/high", "CHI/high", "NYC/high"],
        informational_cells=["LAX/high"],
        climatology_path="data/asos/climatology.json",
        posterior_t_dof=5.0,
        boundary_distance_min_f=1.0,
        asos_max_staleness_hours=2.0,
        sigma_gain_floor_f=0.5,
        bad_prelim_max_drift_f=10.0,
        bad_prelim_active=False,
        edge_threshold=0.07,
        min_calibrated_prob=0.110,
        max_contracts_per_trade=15,
        max_trades_per_cell_per_day=3,
        orderbook_depth_check=True,
    )
    assert cfg.enabled is True
    assert cfg.shadow_only is True
    assert cfg.enabled_cells == ["MIA/high", "AUS/high", "CHI/high", "NYC/high"]
    assert cfg.informational_cells == ["LAX/high"]
    assert cfg.climatology_path == "data/asos/climatology.json"
    assert cfg.posterior_t_dof == 5.0
    assert cfg.boundary_distance_min_f == 1.0
    assert cfg.asos_max_staleness_hours == 2.0
    assert cfg.sigma_gain_floor_f == 0.5
    assert cfg.bad_prelim_max_drift_f == 10.0
    assert cfg.bad_prelim_active is False
    assert cfg.edge_threshold == 0.07
    assert cfg.min_calibrated_prob == pytest.approx(0.110)
    assert cfg.max_contracts_per_trade == 15
    assert cfg.max_trades_per_cell_per_day == 3
    assert cfg.orderbook_depth_check is True


# ── Default values ────────────────────────────────────────────────────────────


def test_defaults() -> None:
    """All defaults match the spec and strategy __init__ fallbacks."""
    cfg = IntradayAsosHighStrategyConfig()
    assert cfg.enabled is False
    assert cfg.shadow_only is True
    assert cfg.enabled_cells == []
    assert cfg.informational_cells == []
    assert cfg.climatology_path == "data/asos/climatology.json"
    assert cfg.posterior_t_dof == 5.0
    assert cfg.boundary_distance_min_f == 1.0
    assert cfg.asos_max_staleness_hours == 2.0
    assert cfg.sigma_gain_floor_f == 0.5
    assert cfg.bad_prelim_max_drift_f == 10.0
    assert cfg.bad_prelim_active is False
    assert cfg.edge_threshold == pytest.approx(0.07)
    assert cfg.min_calibrated_prob == pytest.approx(0.110)
    assert cfg.max_contracts_per_trade == 15
    assert cfg.max_trades_per_cell_per_day == 3
    assert cfg.orderbook_depth_check is True
    assert cfg.asos_obs_cache_ttl_seconds == pytest.approx(600.0)


def test_asos_obs_cache_ttl_bounds() -> None:
    """TTL is bounded [60, 3600]: too-short re-hammers IEM (429s), too-long
    serves stale max-so-far. Out-of-range values are rejected."""
    import pydantic

    cfg = IntradayAsosHighStrategyConfig(asos_obs_cache_ttl_seconds=900.0)
    assert cfg.asos_obs_cache_ttl_seconds == 900.0
    with pytest.raises(pydantic.ValidationError):
        IntradayAsosHighStrategyConfig(asos_obs_cache_ttl_seconds=30.0)
    with pytest.raises(pydantic.ValidationError):
        IntradayAsosHighStrategyConfig(asos_obs_cache_ttl_seconds=7200.0)


# ── Null coercion ─────────────────────────────────────────────────────────────


def test_enabled_cells_null_coerces_to_empty() -> None:
    """YAML ``enabled_cells: null`` is shorthand for ``[]``."""
    cfg = IntradayAsosHighStrategyConfig(enabled_cells=None)  # type: ignore[arg-type]
    assert cfg.enabled_cells == []


def test_informational_cells_null_coerces_to_empty() -> None:
    """YAML ``informational_cells: null`` is shorthand for ``[]``."""
    cfg = IntradayAsosHighStrategyConfig(informational_cells=None)  # type: ignore[arg-type]
    assert cfg.informational_cells == []


# ── Cell format validation ────────────────────────────────────────────────────


def test_cell_format_rejects_missing_var_suffix() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayAsosHighStrategyConfig(enabled_cells=["MIA"])


def test_cell_format_rejects_uppercase_var() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayAsosHighStrategyConfig(enabled_cells=["MIA/HIGH"])


def test_cell_format_rejects_wrong_delimiter() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayAsosHighStrategyConfig(enabled_cells=["MIA-high"])


def test_cell_format_dedupes_silently_enabled_cells() -> None:
    """Duplicate entries are a benign typo — deduplicate silently."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True, enabled_cells=["MIA/high", "MIA/high", "AUS/high"]
    )
    assert cfg.enabled_cells == ["MIA/high", "AUS/high"]


def test_cell_format_dedupes_silently_informational_cells() -> None:
    cfg = IntradayAsosHighStrategyConfig(
        informational_cells=["LAX/high", "LAX/high"]
    )
    assert cfg.informational_cells == ["LAX/high"]


# ── Field bounds ──────────────────────────────────────────────────────────────


def test_edge_threshold_bounds() -> None:
    IntradayAsosHighStrategyConfig(edge_threshold=0.02)
    IntradayAsosHighStrategyConfig(edge_threshold=0.30)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(edge_threshold=0.019)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(edge_threshold=0.31)


def test_max_contracts_per_trade_bounds() -> None:
    IntradayAsosHighStrategyConfig(max_contracts_per_trade=1)
    IntradayAsosHighStrategyConfig(max_contracts_per_trade=50)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_contracts_per_trade=0)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_contracts_per_trade=51)


def test_max_trades_per_cell_per_day_bounds() -> None:
    IntradayAsosHighStrategyConfig(max_trades_per_cell_per_day=1)
    IntradayAsosHighStrategyConfig(max_trades_per_cell_per_day=10)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_trades_per_cell_per_day=0)
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_trades_per_cell_per_day=11)


# ── StrategiesConfig attachment ───────────────────────────────────────────────


def test_intraday_asos_high_attached_to_strategies_config() -> None:
    """``intraday_asos_high`` is a field on ``StrategiesConfig``."""
    s = StrategiesConfig()
    assert isinstance(s.intraday_asos_high, IntradayAsosHighStrategyConfig)
    assert s.intraday_asos_high.enabled is False
    assert s.intraday_asos_high.shadow_only is True


# ── Kelly sizing knobs (Fix 1) ────────────────────────────────────────────────


def test_kelly_fraction_default() -> None:
    """``kelly_fraction`` defaults to 0.075 — matches the global risk default."""
    cfg = IntradayAsosHighStrategyConfig()
    assert cfg.kelly_fraction == pytest.approx(0.075)


def test_max_single_market_pct_default() -> None:
    """``max_single_market_pct`` defaults to 0.05 — matches the global risk default."""
    cfg = IntradayAsosHighStrategyConfig()
    assert cfg.max_single_market_pct == pytest.approx(0.05)


def test_kelly_fraction_custom_value() -> None:
    """Operator can set a conservative kelly_fraction for ASOS."""
    cfg = IntradayAsosHighStrategyConfig(kelly_fraction=0.05)
    assert cfg.kelly_fraction == pytest.approx(0.05)


def test_max_single_market_pct_custom_value() -> None:
    """Operator can set a tighter per-market cap for ASOS."""
    cfg = IntradayAsosHighStrategyConfig(max_single_market_pct=0.03)
    assert cfg.max_single_market_pct == pytest.approx(0.03)


def test_kelly_fraction_lower_bound() -> None:
    """``kelly_fraction`` must be >= 0.0."""
    IntradayAsosHighStrategyConfig(kelly_fraction=0.0)  # boundary: allowed
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(kelly_fraction=-0.01)


def test_kelly_fraction_upper_bound() -> None:
    """``kelly_fraction`` must be <= 0.5."""
    IntradayAsosHighStrategyConfig(kelly_fraction=0.5)  # boundary: allowed
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(kelly_fraction=0.51)


def test_max_single_market_pct_lower_bound() -> None:
    """``max_single_market_pct`` must be >= 0.0."""
    IntradayAsosHighStrategyConfig(max_single_market_pct=0.0)  # boundary: allowed
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_single_market_pct=-0.01)


def test_max_single_market_pct_upper_bound() -> None:
    """``max_single_market_pct`` must be <= 1.0."""
    IntradayAsosHighStrategyConfig(max_single_market_pct=1.0)  # boundary: allowed
    with pytest.raises(ValidationError):
        IntradayAsosHighStrategyConfig(max_single_market_pct=1.01)


def test_kelly_fraction_flows_through_model_dump() -> None:
    """``model_dump()`` exposes ``kelly_fraction`` so the strategy can read it via config dict."""
    cfg = IntradayAsosHighStrategyConfig(kelly_fraction=0.04, max_single_market_pct=0.03)
    d = cfg.model_dump()
    assert d["kelly_fraction"] == pytest.approx(0.04)
    assert d["max_single_market_pct"] == pytest.approx(0.03)


def test_valid_full_config_includes_kelly_knobs() -> None:
    """A config block with explicit kelly knobs round-trips correctly."""
    cfg = IntradayAsosHighStrategyConfig(
        enabled=True,
        shadow_only=True,
        enabled_cells=["MIA/high"],
        kelly_fraction=0.075,
        max_single_market_pct=0.05,
    )
    assert cfg.kelly_fraction == pytest.approx(0.075)
    assert cfg.max_single_market_pct == pytest.approx(0.05)
