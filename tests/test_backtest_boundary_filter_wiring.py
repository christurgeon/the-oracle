"""Backtest threads boundary_filter_cells into the pure _EvalConfig (Path 7a #4)."""

from __future__ import annotations

from python.analytics.backtest import BacktestConfig, _eval_config_from


def test_default_backtest_config_has_filter_off():
    assert BacktestConfig().boundary_filter_cells == {}


def test_eval_config_default_is_empty():
    cfg = _eval_config_from(BacktestConfig())
    assert cfg.boundary_filter_cells == {}


def test_eval_config_threads_boundary_cells():
    bt = BacktestConfig(boundary_filter_cells={"MIA/low": 0.5})
    cfg = _eval_config_from(bt)
    assert cfg.boundary_filter_cells == {"MIA/low": 0.5}


def test_eval_config_copies_dict_not_aliases():
    """The _EvalConfig dict must be a copy so mutating one doesn't leak."""
    bt = BacktestConfig(boundary_filter_cells={"MIA/low": 0.5})
    cfg = _eval_config_from(bt)
    cfg.boundary_filter_cells["NYC/high"] = 1.0
    assert "NYC/high" not in bt.boundary_filter_cells
