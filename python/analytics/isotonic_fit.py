"""Hermetic isotonic-regression fitting module.

Extracted from scripts/fit_isotonic.py. Pure pair-in / model-or-thresholds-
out math for the per-cell calibrator harness's per-fold refits and for the
production fitter CLI. Mirrors the emos_fit.py hermetic pattern: stdlib +
numpy + sklearn.isotonic only; no I/O, no argparse, no corpus loaders.

Sklearn dependency is intentional and offline-only. This module MUST NOT
be imported by python/data/calibration.py or any orchestrator/strategy
module — runtime calibrator stays sklearn-free per CLAUDE.md's
"No sklearn at runtime" guidance.

The scripts/fit_isotonic.py CLI imports `fit_isotonic_model` to preserve a
single source of truth on the actual isotonic math.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def default_anchor_weight(n_pairs: int) -> int:
    """Default sample-weight applied to synthetic (0,0)/(1,1) anchors.

    Anchors prevent the isotonic curve from flattening near 0 or 1 when
    the training distribution has no rows near those probabilities. The
    weight scales with n_pairs so anchors don't dominate large fits.
    """
    return max(5, n_pairs // 200)


def _apply_anchors(
    features: np.ndarray,
    targets: np.ndarray,
    weight: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate (0,0) and (1,1) anchors with the given sample weight.

    Returns (features_with_anchors, targets_with_anchors, sample_weights).
    Module-private — `fit_isotonic_model` is the only caller.
    """
    anchor_features = np.array([0.0, 1.0], dtype=np.float64)
    anchor_targets = np.array([0.0, 1.0], dtype=np.float64)
    n_pairs = len(features)

    features_combined = np.concatenate([features, anchor_features])
    targets_combined = np.concatenate([targets, anchor_targets])
    sample_weights = np.concatenate([
        np.ones(n_pairs, dtype=np.float64),
        np.array([float(weight), float(weight)], dtype=np.float64),
    ])

    return features_combined, targets_combined, sample_weights


def _validate_pairs(pairs: np.ndarray) -> None:
    """Pre-fit guard. Raises ValueError on degenerate input."""
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must have shape (N, 2); got {pairs.shape}")
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("pairs contains non-finite values")


def fit_isotonic_model(
    pairs: np.ndarray,
    *,
    anchor_endpoints: bool = True,
    anchor_weight: int | None = None,
) -> IsotonicRegression:
    """Fit isotonic regression on (raw_prob, outcome) pairs and return the model.

    Performs exactly ONE sklearn fit. Callers that need the lookup-table form
    should call `fit_isotonic_pairs` (a thin wrapper around this function).

    Args:
        pairs: (N, 2) array of (raw_probability, binary_outcome).
            raw_probability in [0, 1]; binary_outcome in {0, 1}. Inputs are
            cast to float64 internally for sklearn parity.
        anchor_endpoints: Append synthetic (0,0) and (1,1) anchors with
            elevated sample weight. Prevents the curve from flattening
            near the endpoints when training data is sparse there.
        anchor_weight: Manual override for the anchor sample weight.
            Defaults to `default_anchor_weight(N)` when None.

    Returns:
        A fitted sklearn IsotonicRegression with `out_of_bounds="clip"`,
        `y_min=0.0`, `y_max=1.0`. The fitted model exposes `X_thresholds_`
        and `y_thresholds_` for callers that want the lookup table.

    Raises:
        ValueError: pairs is empty, has wrong shape, or contains non-finite values.
    """
    _validate_pairs(pairs)

    features = pairs[:, 0].astype(np.float64)
    targets = pairs[:, 1].astype(np.float64)
    n_pairs = len(pairs)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    if anchor_endpoints:
        weight = anchor_weight if anchor_weight is not None else default_anchor_weight(n_pairs)
        x_w, y_w, sw = _apply_anchors(features, targets, weight)
        iso.fit(x_w, y_w, sample_weight=sw)
    else:
        iso.fit(features, targets)

    return iso


def fit_isotonic_pairs(
    pairs: np.ndarray,
    *,
    anchor_endpoints: bool = True,
    anchor_weight: int | None = None,
) -> tuple[list[float], list[float]]:
    """Fit isotonic regression and return JSON-serializable (breakpoints, values).

    Thin wrapper over `fit_isotonic_model`. The harness's per-fold refits
    use this signature; the runtime IsotonicCalibrator consumes the
    breakpoints/values lookup table directly.

    Args:
        pairs: see `fit_isotonic_model`.
        anchor_endpoints: see `fit_isotonic_model`.
        anchor_weight: see `fit_isotonic_model`.

    Returns:
        (breakpoints, values) lists for JSON serialization.

    Raises:
        ValueError: see `fit_isotonic_model`.
    """
    model = fit_isotonic_model(
        pairs,
        anchor_endpoints=anchor_endpoints,
        anchor_weight=anchor_weight,
    )
    return model.X_thresholds_.tolist(), model.y_thresholds_.tolist()


__all__ = ["fit_isotonic_model", "fit_isotonic_pairs", "default_anchor_weight"]
