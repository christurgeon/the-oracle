"""Unit tests for ``python/data/kxrain_decision.py``."""

from __future__ import annotations

import ast
import inspect
import itertools
import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_decision import (  # noqa: E402
    SKIP_REASONS,
    DecisionOutcome,
    evaluate_decision,
)

_MODULE_PATH = _PROJECT_ROOT / "python" / "data" / "kxrain_decision.py"


# Band edges


def test_band_low_edge_inclusive_eligible():
    out = evaluate_decision(
        yes_bid=0.10, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.eligible is True
    assert out.skip_reason is None
    assert out.in_price_band is True


def test_just_below_band_low_skips_price_band():
    out = evaluate_decision(
        yes_bid=0.0999, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.eligible is False
    assert out.skip_reason == "price_band"
    assert out.in_price_band is False


def test_just_below_band_high_eligible():
    out = evaluate_decision(
        yes_bid=0.8999, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.eligible is True
    assert out.skip_reason is None
    assert out.in_price_band is True


def test_band_high_edge_exclusive_skips_price_band():
    out = evaluate_decision(
        yes_bid=0.90, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.eligible is False
    assert out.skip_reason == "price_band"
    assert out.in_price_band is False


# Skip-reason precedence: one rung at a time


def test_out_of_window_fires_first():
    out = evaluate_decision(
        yes_bid=0.50, in_window=False, no_trades_30m=1, already_entered=False
    )
    assert out.skip_reason == "out_of_window"
    assert out.eligible is False


def test_no_price_when_yes_bid_none():
    out = evaluate_decision(
        yes_bid=None, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.skip_reason == "no_price"
    assert out.eligible is False
    assert out.no_ask is None
    assert out.assumed_entry_price is None


def test_price_band_rung():
    out = evaluate_decision(
        yes_bid=0.95, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.skip_reason == "price_band"
    assert out.eligible is False


def test_already_entered_rung():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=1, already_entered=True
    )
    assert out.skip_reason == "already_entered"
    assert out.eligible is False


def test_trades_unavailable_rung():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=None, already_entered=False
    )
    assert out.skip_reason == "trades_unavailable"
    assert out.eligible is None


def test_no_recent_no_trade_rung():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=0, already_entered=False
    )
    assert out.skip_reason == "no_recent_no_trade"
    assert out.eligible is False


def test_eligible_when_everything_holds():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=3, already_entered=False
    )
    assert out.eligible is True
    assert out.skip_reason is None
    assert out.assumed_entry_price == pytest.approx(0.50)


# The precedence combinations that matter


def test_out_of_window_beats_price_band_and_trades_unavailable():
    out = evaluate_decision(
        yes_bid=0.95, in_window=False, no_trades_30m=None, already_entered=False
    )
    assert out.skip_reason == "out_of_window"
    assert out.eligible is False


def test_no_price_beats_price_band_when_yes_bid_is_none():
    out = evaluate_decision(
        yes_bid=None, in_window=True, no_trades_30m=None, already_entered=False
    )
    assert out.skip_reason == "no_price"
    assert out.eligible is False


def test_price_band_beats_already_entered_and_trades_unavailable():
    out = evaluate_decision(
        yes_bid=0.95, in_window=True, no_trades_30m=None, already_entered=True
    )
    assert out.skip_reason == "price_band"
    assert out.eligible is False


def test_already_entered_beats_trades_unavailable():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=None, already_entered=True
    )
    assert out.skip_reason == "already_entered"
    assert out.eligible is False


def test_trades_unavailable_beats_no_recent_no_trade():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=None, already_entered=False
    )
    assert out.eligible is None
    assert out.skip_reason == "trades_unavailable"


# eligible is None must not be coerced


def test_trades_unavailable_eligible_is_none_not_falsy():
    out = evaluate_decision(
        yes_bid=0.50, in_window=True, no_trades_30m=None, already_entered=False
    )
    assert out.eligible is None
    assert out.eligible is not False


# Derived fields present on skipped rows


def test_derived_fields_present_on_out_of_window_skip():
    out = evaluate_decision(
        yes_bid=0.62, in_window=False, no_trades_30m=None, already_entered=False
    )
    assert out.in_price_band is True
    assert out.no_ask == pytest.approx(0.38)
    assert out.yes_bid_ge_040 is True
    assert out.assumed_entry_price is None


def test_derived_fields_present_on_price_band_skip():
    out = evaluate_decision(
        yes_bid=0.95, in_window=True, no_trades_30m=5, already_entered=False
    )
    assert out.in_price_band is False
    assert out.no_ask == pytest.approx(0.05)
    assert out.yes_bid_ge_040 is True
    assert out.assumed_entry_price is None


def test_assumed_entry_price_none_unless_eligible_true():
    for yes_bid, in_window, trades, entered in [
        (0.50, False, 1, False),
        (0.95, True, 1, False),
        (0.50, True, 1, True),
        (0.50, True, None, False),
        (0.50, True, 0, False),
    ]:
        out = evaluate_decision(
            yes_bid=yes_bid,
            in_window=in_window,
            no_trades_30m=trades,
            already_entered=entered,
        )
        assert out.eligible is not True
        assert out.assumed_entry_price is None


# no_ask arithmetic


@pytest.mark.parametrize(
    "yes_bid",
    [0.0, 0.10, 0.37, 0.5, 0.8999, 0.9, 1.0, 0.1 + 0.2 - 0.3],
)
def test_no_ask_is_one_minus_yes_bid(yes_bid):
    out = evaluate_decision(
        yes_bid=yes_bid, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.no_ask == pytest.approx(1.0 - yes_bid, abs=1e-9)


# Non-finite / malformed yes_bid never raises


@pytest.mark.parametrize("bad_price", [None, math.nan, math.inf, -math.inf])
def test_non_finite_yes_bid_yields_no_price(bad_price):
    out = evaluate_decision(
        yes_bid=bad_price, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert out.skip_reason == "no_price"
    assert out.eligible is False
    assert out.no_ask is None
    assert out.in_price_band is False
    assert out.yes_bid_ge_040 is False


# secondary cut labelling


def test_yes_bid_ge_040_secondary_cut():
    below = evaluate_decision(
        yes_bid=0.39, in_window=True, no_trades_30m=1, already_entered=False
    )
    at = evaluate_decision(
        yes_bid=0.40, in_window=True, no_trades_30m=1, already_entered=False
    )
    assert below.yes_bid_ge_040 is False
    assert at.yes_bid_ge_040 is True


# every returned skip_reason is in the vocabulary


def test_skip_reasons_vocabulary_is_exhaustive():
    expected = {
        "out_of_window",
        "no_price",
        "price_band",
        "already_entered",
        "trades_unavailable",
        "no_recent_no_trade",
        # Never emitted by evaluate_decision; the observer stamps it on the
        # market-absent census row.
        "market_absent",
    }
    assert SKIP_REASONS == frozenset(expected)


def test_every_emitted_skip_reason_is_in_vocabulary():
    yes_bids = [None, math.nan, -1.0, 0.05, 0.10, 0.5, 0.8999, 0.9, 1.5]
    for yes_bid, in_window, trades, entered in itertools.product(
        yes_bids, [True, False], [None, -1, 0, 1], [True, False]
    ):
        out = evaluate_decision(
            yes_bid=yes_bid,
            in_window=in_window,
            no_trades_30m=trades,
            already_entered=entered,
        )
        if out.skip_reason is not None:
            assert out.skip_reason in SKIP_REASONS


# Signature firewall — the point of this module


def test_signature_is_exactly_the_expected_parameter_set():
    sig = inspect.signature(evaluate_decision)
    expected = {
        "yes_bid",
        "in_window",
        "no_trades_30m",
        "already_entered",
        "yes_bid_min",
        "yes_bid_max",
        "secondary_cut",
    }
    assert set(sig.parameters) == expected
    for name, param in sig.parameters.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, name


_FORBIDDEN_NAME_FRAGMENTS = (
    "gauge",
    "precip",
    "p01i",
    "pop",
    "forecast",
    "asos",
    "model",
    "prob",
)


def test_signature_has_no_gauge_or_forecast_parameter():
    sig = inspect.signature(evaluate_decision)
    for name in sig.parameters:
        lowered = name.lower()
        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in lowered, f"parameter {name!r} matches forbidden {fragment!r}"


def test_decision_outcome_is_frozen_dataclass():
    out = evaluate_decision(
        yes_bid=0.5, in_window=True, no_trades_30m=1, already_entered=False
    )
    with pytest.raises(Exception):
        out.eligible = False  # type: ignore[misc]


# Import-surface / AST firewall


def _module_ast() -> ast.Module:
    source = _MODULE_PATH.read_text()
    return ast.parse(source, filename=str(_MODULE_PATH))


def _imported_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_module_imports_nothing_forbidden():
    tree = _module_ast()
    imported = _imported_module_names(tree)
    for name in imported:
        assert not name.startswith("python.execution"), name
        assert not name.startswith("python.risk"), name
        assert not name.startswith("python.strategies"), name
        assert name != "TradeSignal", name


def test_module_has_no_place_order_identifier():
    tree = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "place_order":
            pytest.fail("found identifier 'place_order' in kxrain_decision.py")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "place_order":
            pytest.fail("found function 'place_order' in kxrain_decision.py")
        if isinstance(node, ast.Attribute) and node.attr == "place_order":
            pytest.fail("found attribute access 'place_order' in kxrain_decision.py")


def test_module_has_no_import_time_assert():
    tree = _module_ast()
    for node in tree.body:
        assert not isinstance(node, ast.Assert), "top-level assert would crash orchestrator import"


# Exhaustive sweep: never raises, skip_reason is None iff eligible is True


def test_exhaustive_sweep_never_raises_and_invariant_holds():
    yes_bids = [
        None,
        math.nan,
        math.inf,
        -math.inf,
        -1.0,
        0.0,
        0.0999,
        0.10,
        0.40,
        0.3999,
        0.5,
        0.8999,
        0.90,
        1.0,
        1.5,
    ]
    trades_options = [None, -5, -1, 0, 1, 5]
    for yes_bid, in_window, trades, entered in itertools.product(
        yes_bids, [True, False], trades_options, [True, False]
    ):
        out = evaluate_decision(
            yes_bid=yes_bid,
            in_window=in_window,
            no_trades_30m=trades,
            already_entered=entered,
        )
        assert isinstance(out, DecisionOutcome)
        assert (out.skip_reason is None) == (out.eligible is True)
        if out.skip_reason is not None:
            assert out.skip_reason in SKIP_REASONS


def test_no_trade_recent_none_iff_no_trades_30m_none():
    for trades in [None, 0, 1, 5, -1]:
        out = evaluate_decision(
            yes_bid=0.5, in_window=True, no_trades_30m=trades, already_entered=False
        )
        if trades is None:
            assert out.no_trade_recent is None
        else:
            assert out.no_trade_recent == (trades > 0)
