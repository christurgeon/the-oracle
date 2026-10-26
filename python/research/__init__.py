"""Hermetic research-only substrate. See `python.research.scaffold`."""

from __future__ import annotations

from python.research.scaffold import (
    EvaluationRow,
    GateConfig,
    GateName,
    LiveCalibratorKind,
    ScaffoldEnvelope,
    SkipReason,
    apply_gate_stack,
    build_envelope,
    iter_evaluation_rows,
    recompute_emos_posterior,
)

__all__ = [
    "EvaluationRow",
    "GateConfig",
    "GateName",
    "LiveCalibratorKind",
    "ScaffoldEnvelope",
    "SkipReason",
    "apply_gate_stack",
    "build_envelope",
    "iter_evaluation_rows",
    "recompute_emos_posterior",
]
