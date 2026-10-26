"""Tests for ``IntradayEmosHighStrategyConfig`` + cross-config deconfliction.

Covers Pydantic Field constraints (§11.X) and the top-level
``Settings`` model-validator that hard-fails when
``strategies.intraday_emos_high.enabled == True`` overlaps
``weather.same_day_trading.allowed_cities`` on MIA/LAX (§7.3 v1.2
MINOR #5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from python.settings import (
    IntradayEmosHighStrategyConfig,
    Settings,
    StrategiesConfig,
)


def _write_config(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# ── Defaults ────────────────────────────────────────────────────────────


def test_intraday_emos_high_defaults() -> None:
    cfg = IntradayEmosHighStrategyConfig()
    assert cfg.enabled is False
    assert cfg.enabled_cells == []
    assert cfg.edge_threshold == 0.07
    assert cfg.max_edge == 0.30
    assert cfg.min_distance_from_strike_std == 2.0
    assert cfg.max_contracts_per_trade == 50
    assert cfg.max_trades_per_cell_per_day == 3
    assert cfg.emos_hot_reload_pause_seconds == 60
    # Audit-log toggles + retention (bracket-window follow-up #5). Defaults
    # match the hardcoded ``.get(...)`` defaults at
    # ``python/strategies/intraday_emos_high.py:210-213`` so behavior is
    # identical for the dormant config.
    assert cfg.eval_audit_log is True
    assert cfg.candidate_audit_log is True
    assert cfg.scan_funnel_log is True
    assert cfg.audit_log_retention_days == 14


def test_intraday_emos_high_attached_to_strategies_config() -> None:
    s = StrategiesConfig()
    assert isinstance(s.intraday_emos_high, IntradayEmosHighStrategyConfig)
    assert s.intraday_emos_high.enabled is False


def test_enabled_cells_null_coerces_to_empty_list() -> None:
    """``enabled_cells: null`` in YAML is operator shorthand for ``[]``.

    Symmetric with ``WeatherStrategyConfig.emos_enabled_cells`` — without
    the ``mode='before'`` coercion the after-validator raises ``list_type``
    and the bot fails to start.
    """
    cfg = IntradayEmosHighStrategyConfig(enabled_cells=None)  # type: ignore[arg-type]
    assert cfg.enabled_cells == []


# ── Field bound boundaries (§11.X) ──────────────────────────────────────


def test_min_distance_below_2_raises_validation_error() -> None:
    # The v6 audited floor is 2.0; ``Field(ge=2.0)`` enforces it.
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(min_distance_from_strike_std=1.9)
    # Boundary case ``2.0`` accepted; ``10.0`` accepted as sanity cap.
    IntradayEmosHighStrategyConfig(min_distance_from_strike_std=2.0)
    IntradayEmosHighStrategyConfig(min_distance_from_strike_std=10.0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(min_distance_from_strike_std=10.01)


def test_max_contracts_above_50_raises_validation_error() -> None:
    # The v6 audited cap is 50; the value MAY be tightened but MUST NOT exceed 50.
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(max_contracts_per_trade=51)
    IntradayEmosHighStrategyConfig(max_contracts_per_trade=50)
    IntradayEmosHighStrategyConfig(max_contracts_per_trade=1)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(max_contracts_per_trade=0)


def test_edge_threshold_range() -> None:
    # Pair with a max_edge strictly greater so the
    # ``edge_threshold < max_edge`` invariant is satisfied; this test
    # exercises ONLY the field-level ``ge/le`` bound on edge_threshold.
    IntradayEmosHighStrategyConfig(edge_threshold=0.02, max_edge=1.0)
    IntradayEmosHighStrategyConfig(edge_threshold=0.30, max_edge=1.0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(edge_threshold=0.019, max_edge=1.0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(edge_threshold=0.31, max_edge=1.0)


def test_max_edge_range() -> None:
    # Pair with an edge_threshold strictly less so the
    # ``edge_threshold < max_edge`` invariant is satisfied; this test
    # exercises ONLY the field-level ``ge/le`` bound on max_edge.
    IntradayEmosHighStrategyConfig(edge_threshold=0.02, max_edge=0.03)
    IntradayEmosHighStrategyConfig(edge_threshold=0.02, max_edge=1.0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(edge_threshold=0.02, max_edge=0.0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(edge_threshold=0.02, max_edge=1.01)


def test_max_trades_per_cell_per_day_range() -> None:
    IntradayEmosHighStrategyConfig(max_trades_per_cell_per_day=1)
    IntradayEmosHighStrategyConfig(max_trades_per_cell_per_day=10)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(max_trades_per_cell_per_day=0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(max_trades_per_cell_per_day=11)


def test_emos_hot_reload_pause_seconds_range() -> None:
    IntradayEmosHighStrategyConfig(emos_hot_reload_pause_seconds=0)
    IntradayEmosHighStrategyConfig(emos_hot_reload_pause_seconds=600)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(emos_hot_reload_pause_seconds=-1)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(emos_hot_reload_pause_seconds=601)


def test_audit_log_retention_days_range() -> None:
    # Bounds match WeatherStrategyConfig's parity: ge=1, le=365.
    IntradayEmosHighStrategyConfig(audit_log_retention_days=1)
    IntradayEmosHighStrategyConfig(audit_log_retention_days=365)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(audit_log_retention_days=0)
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(audit_log_retention_days=366)


# ── enabled_cells validator (mirrors weather.emos_enabled_cells) ───────


def test_enabled_cells_validator_accepts_valid_cells() -> None:
    cfg = IntradayEmosHighStrategyConfig(enabled_cells=["MIA/high", "LAX/high"])
    assert cfg.enabled_cells == ["MIA/high", "LAX/high"]


def test_enabled_cells_validator_dedupes_silently() -> None:
    # Mirror the weather validator: duplicates are a benign typo, not fatal.
    cfg = IntradayEmosHighStrategyConfig(enabled_cells=["MIA/high", "MIA/high"])
    assert cfg.enabled_cells == ["MIA/high"]


def test_enabled_cells_rejects_missing_var_suffix() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayEmosHighStrategyConfig(enabled_cells=["MIA"])


def test_enabled_cells_rejects_wrong_var() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayEmosHighStrategyConfig(enabled_cells=["MIA/maximum"])


def test_enabled_cells_rejects_lowercase_or_mixed_city() -> None:
    # The pattern ``^[A-Z]{2,5}/(high|low)$`` requires fully uppercase city.
    # "BadCity" mixes case so it fails the pattern.
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayEmosHighStrategyConfig(enabled_cells=["BadCity/high"])


def test_enabled_cells_rejects_uppercase_var() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayEmosHighStrategyConfig(enabled_cells=["MIA/HIGH"])


def test_enabled_cells_rejects_wrong_delimiter() -> None:
    with pytest.raises(ValidationError, match="<CITY>/<high\\|low>"):
        IntradayEmosHighStrategyConfig(enabled_cells=["MIA-high"])


# ── edge_threshold < max_edge invariant (mirrors WeatherStrategyConfig) ─


def test_edge_threshold_below_max_passes() -> None:
    # Defaults: edge_threshold=0.07, max_edge=0.30 — strictly less, passes.
    IntradayEmosHighStrategyConfig()
    # Explicit pair where edge_threshold < max_edge.
    cfg = IntradayEmosHighStrategyConfig(edge_threshold=0.05, max_edge=0.10)
    assert cfg.edge_threshold == 0.05
    assert cfg.max_edge == 0.10


def test_edge_threshold_above_max_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        IntradayEmosHighStrategyConfig(edge_threshold=0.20, max_edge=0.10)
    msg = str(excinfo.value)
    assert "edge_threshold" in msg
    assert "max_edge" in msg
    assert "0.2" in msg
    assert "0.1" in msg
    # Boundary: equal also rejected (strict <).
    with pytest.raises(ValidationError):
        IntradayEmosHighStrategyConfig(edge_threshold=0.10, max_edge=0.10)


# ── Cross-config deconfliction validator (§11.X, §7.3) ──────────────────


def _base_yaml(
    *,
    intraday_enabled: bool,
    intraday_cells: list[str],
    same_day_enabled: bool,
    allowed_cities: list[str],
) -> dict[str, Any]:
    return {
        "strategies": {
            "weather": {
                "edge_threshold": 0.07,
                "same_day_trading": {
                    "enabled": same_day_enabled,
                    "allowed_cities": allowed_cities,
                },
            },
            "intraday_emos_high": {
                "enabled": intraday_enabled,
                "enabled_cells": intraday_cells,
            },
        },
    }


def test_no_conflict_when_intraday_disabled(tmp_path: Path) -> None:
    # Intraday OFF + allowed_cities contains MIA → must NOT raise.
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=False,
            intraday_cells=["MIA/high"],
            same_day_enabled=True,
            allowed_cities=["MIA"],
        ),
    )
    settings = Settings.load(config_path=cfg)
    assert settings.strategies.intraday_emos_high.enabled is False
    assert settings.strategies.weather.same_day_trading.allowed_cities == ["MIA"]


def test_no_conflict_when_intraday_enabled_no_mia_lax_in_allowed_cities(
    tmp_path: Path,
) -> None:
    # Intraday ON, CHI in allowed_cities — no overlap with MIA/LAX intraday cells.
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high", "LAX/high"],
            same_day_enabled=True,
            allowed_cities=["CHI"],
        ),
    )
    settings = Settings.load(config_path=cfg)
    assert settings.strategies.intraday_emos_high.enabled is True
    assert "MIA" not in settings.strategies.weather.same_day_trading.allowed_cities


def test_hard_fail_when_intraday_enabled_and_mia_in_allowed_cities(
    tmp_path: Path,
) -> None:
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high"],
            same_day_enabled=True,
            allowed_cities=["MIA"],
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        Settings.load(config_path=cfg)
    msg = str(excinfo.value)
    assert "strategies.intraday_emos_high" in msg
    assert "weather.same_day_trading.allowed_cities" in msg
    assert "MIA" in msg


def test_hard_fail_when_intraday_enabled_and_lax_in_allowed_cities(
    tmp_path: Path,
) -> None:
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["LAX/high"],
            same_day_enabled=True,
            allowed_cities=["LAX"],
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        Settings.load(config_path=cfg)
    msg = str(excinfo.value)
    assert "strategies.intraday_emos_high" in msg
    assert "weather.same_day_trading.allowed_cities" in msg
    assert "LAX" in msg


def test_hard_fail_mixed_allowed_cities_isolates_offender(tmp_path: Path) -> None:
    # MIA + CHI in allowed_cities, intraday owns MIA/high. CHI is not an
    # intraday cell, so only MIA must be flagged.
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high"],
            same_day_enabled=True,
            allowed_cities=["MIA", "CHI"],
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        Settings.load(config_path=cfg)
    msg = str(excinfo.value)
    assert "MIA" in msg
    # CHI is not an intraday cell so it must not be named as a conflict.
    # (It can appear elsewhere; assert it isn't shown as the offender.)
    # The validator builds ``conflict = allowed & intraday_cities`` so the
    # message lists only MIA. Confirm CHI is absent from the conflict set.
    assert "['MIA']" in msg or '"MIA"' in msg or "{'MIA'}" in msg


def test_no_conflict_chi_only_in_allowed_cities_when_intraday_enabled(
    tmp_path: Path,
) -> None:
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high"],
            same_day_enabled=True,
            allowed_cities=["CHI"],
        ),
    )
    settings = Settings.load(config_path=cfg)
    assert settings.strategies.intraday_emos_high.enabled is True


def test_hard_fail_even_when_same_day_trading_master_switch_off(
    tmp_path: Path,
) -> None:
    """Defense-in-depth: the deconfliction rule is on ``allowed_cities``
    regardless of the ``same_day_trading.enabled`` master switch.

    Rationale: an operator can flip ``same_day_trading.enabled = true`` at
    any time post-deploy without remembering the intraday city overlap.
    Catching the misconfiguration at load — even when same-day is currently
    off — gives the operator one consistent failure surface. This mirrors
    the spec validator which only gates on ``intraday.enabled`` (§11.X).
    """
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high"],
            same_day_enabled=False,
            allowed_cities=["MIA"],
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        Settings.load(config_path=cfg)
    msg = str(excinfo.value)
    assert "MIA" in msg


def test_allowed_cities_field_validator_fires_before_deconfliction(
    tmp_path: Path,
) -> None:
    """``allowed_cities=["mia"]`` with intraday enabled should fail at the
    field-validation level (case mismatch) BEFORE the cross-config
    deconfliction validator runs. The error message must name the
    offending entry ``'mia'`` (not the deconfliction set ``{'MIA'}``),
    proving the field validator caught it first.
    """
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=["MIA/high"],
            same_day_enabled=True,
            allowed_cities=["mia"],
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        Settings.load(config_path=cfg)
    msg = str(excinfo.value)
    # The field validator names the offending entry verbatim.
    assert "'mia'" in msg or '"mia"' in msg
    # The cross-config validator never runs, so its hallmark phrase
    # ``double-execution`` must NOT be in the message.
    assert "double-execution" not in msg


def test_no_conflict_when_intraday_enabled_but_no_cells(tmp_path: Path) -> None:
    # Edge: intraday on with empty enabled_cells. No cells claimed → no
    # conflict possible. The validator's set intersection is empty.
    cfg = _write_config(
        tmp_path,
        _base_yaml(
            intraday_enabled=True,
            intraday_cells=[],
            same_day_enabled=True,
            allowed_cities=["MIA", "LAX"],
        ),
    )
    settings = Settings.load(config_path=cfg)
    assert settings.strategies.intraday_emos_high.enabled is True
    assert settings.strategies.intraday_emos_high.enabled_cells == []


# ── Production config.yaml sanity ──────────────────────────────────────


def test_production_config_intraday_shadow_only_invariant() -> None:
    """Production ``config.yaml`` may enable the intraday strategy only in
    shadow-only mode (spec 2026-05-29-intraday-shadow-only-mode-design.md).
    Live trading still requires a deliberate ``shadow_only: false`` flip
    plus a separate PR. Tripwire: if this fails, someone enabled the
    strategy for live execution without going through the gate.
    """
    settings = Settings.load()
    cfg = settings.strategies.intraday_emos_high
    if cfg.enabled:
        assert cfg.shadow_only is True, (
            "python/config.yaml: intraday_emos_high.enabled=true requires "
            "shadow_only=true. Live execution is gated on the post-shadow "
            "calibration review (§8 of the shadow-only spec) and a separate PR."
        )
    assert "MIA/high" in cfg.enabled_cells


# ── shadow_only flag (§4.1) ─────────────────────────────────────────────


def test_shadow_only_defaults_true() -> None:
    cfg = IntradayEmosHighStrategyConfig()
    assert cfg.shadow_only is True


def test_shadow_only_loads_false() -> None:
    cfg = IntradayEmosHighStrategyConfig(shadow_only=False)
    assert cfg.shadow_only is False


def test_shadow_only_roundtrip() -> None:
    cfg = IntradayEmosHighStrategyConfig(shadow_only=False)
    dumped = cfg.model_dump()
    assert "shadow_only" in dumped
    assert dumped["shadow_only"] is False
    reconstructed = IntradayEmosHighStrategyConfig(**dumped)
    assert reconstructed.shadow_only is False
