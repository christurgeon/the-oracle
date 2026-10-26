"""AST-walking drift guard for ``IntradayEmosHighStrategy`` config reads.

Walks ``python/strategies/intraday_emos_high.py`` and collects every
``config.get("...", default)`` or ``self._config.get("...", default)``
string-literal key. Asserts the collected set is a subset of:

* The union of declared Pydantic fields on
  ``IntradayEmosHighStrategyConfig`` and ``WeatherStrategyConfig`` (the
  schema contract — keys must be declared somewhere).
* The merged runtime config dict produced by the orchestrator
  (``{**weather.model_dump(), **intraday_emos_high.model_dump()}``) — the
  runtime contract that every read resolves to a real value with no
  silent fallback to a hardcoded ``.get()`` default.

Catches a future engineer adding ``self._config.get("foo", default)``
without declaring ``foo`` on either model — the exact bug class that bit
``WeatherStrategyConfig`` on 2026-05-21 and motivated bracket-window
follow-up #5.
"""

from __future__ import annotations

import ast
from pathlib import Path

from python.settings import (
    IntradayEmosHighStrategyConfig,
    Settings,
    WeatherStrategyConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INTRADAY_STRATEGY_PATH = REPO_ROOT / "python" / "strategies" / "intraday_emos_high.py"


class _ConfigGetKeyCollector(ast.NodeVisitor):
    """Collect string-literal first-arguments of ``config.get(...)`` and
    ``self._config.get(...)`` calls. Skips calls whose first argument is
    not a string literal (can't audit them statically).
    """

    def __init__(self) -> None:
        self.keys: set[str] = set()
        # Diagnostic — line numbers of skipped non-literal .get(...) reads,
        # for triage if drift surfaces.
        self.non_literal_reads: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 — ast.NodeVisitor naming
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get":
            if self._is_intraday_config_target(func.value) and node.args:
                key_arg = node.args[0]
                if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                    self.keys.add(key_arg.value)
                else:
                    self.non_literal_reads.append(node.lineno)
        self.generic_visit(node)

    @staticmethod
    def _is_intraday_config_target(value: ast.expr) -> bool:
        """True iff ``value`` is ``config`` (constructor scope) or
        ``self._config`` (instance scope) — the two binders introduced at
        ``intraday_emos_high.py:125`` and ``Strategy.__init__`` respectively.
        """
        if isinstance(value, ast.Name) and value.id == "config":
            return True
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "_config"
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            return True
        return False


def _collect_runtime_config_keys() -> set[str]:
    tree = ast.parse(INTRADAY_STRATEGY_PATH.read_text())
    collector = _ConfigGetKeyCollector()
    collector.visit(tree)
    return collector.keys


def test_every_config_read_is_declared_on_a_model() -> None:
    """Every literal key read from the intraday config dict must be declared
    on either ``IntradayEmosHighStrategyConfig`` or ``WeatherStrategyConfig``.
    """
    keys = _collect_runtime_config_keys()
    assert keys, (
        f"AST visitor found no config.get(...) reads in {INTRADAY_STRATEGY_PATH}; "
        "either the visitor regressed or the file was emptied."
    )
    declared = set(IntradayEmosHighStrategyConfig.model_fields.keys()) | set(
        WeatherStrategyConfig.model_fields.keys()
    )
    undeclared = keys - declared
    assert not undeclared, (
        f"intraday strategy reads {sorted(undeclared)} via config.get(...) "
        f"but those keys are not declared on IntradayEmosHighStrategyConfig "
        f"or WeatherStrategyConfig. This is the silent-default bug class "
        f"from bracket-window follow-up #5: declare each on the appropriate "
        f"model with a Pydantic Field()."
    )


def test_runtime_merged_dict_contains_every_config_read() -> None:
    """The orchestrator's merged config dict must contain every literal key
    the strategy reads. Belt-and-suspenders for the per-model check above —
    catches the case where a model declares the field but the merge logic
    in ``python/orchestrator.py`` skips it (e.g. a future refactor that
    forgets one of the two ``model_dump()`` spreads).
    """
    keys = _collect_runtime_config_keys()
    settings = Settings.load()
    merged = {
        **settings.strategies.weather.model_dump(),
        **settings.strategies.intraday_emos_high.model_dump(),
    }
    missing = keys - set(merged.keys())
    assert not missing, (
        f"merged runtime config is missing {sorted(missing)} — "
        f"the strategy would silently fall back to its hardcoded .get() "
        f"defaults. Check the orchestrator merge at "
        f"python/orchestrator.py:_build_intraday_emos_high_strategy."
    )
