"""Shared binning helper for ASOS intraday reliability reports.

Used by:
  - ``scripts/intraday_asos_shadow_report.py`` (forward shadow evaluation)
  - ``research_2026_06_05_asos_reliability.py`` (retrospective research; unpublished)

Extracted to keep the bin boundaries and bin-computation logic DRY.  Both
consumers apply the same bins: ``[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9,
0.95, 1.001]``.  ``1.001`` is the sentinel upper edge that ensures p=1.0 falls
inside the last real bin (``[0.95, 1.001)``).

The ``compute_reliability_bins`` function is pure Python — no pandas, no numpy
— so it is importable without the ``ml`` extras group.
"""

from __future__ import annotations

from typing import Any

# Canonical bin edges used by every ASOS reliability consumer.
ASOS_BINS: list[float] = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.001]


def _bin_label(lo: float, hi: float) -> str:
    """Return a human-readable bin label, e.g. ``'[0.00,0.05)'``."""
    # Use the sentinel edge as ``'1.00'`` in labels.
    hi_str = "1.00" if hi > 1.0 else f"{hi:.2f}"
    return f"[{lo:.2f},{hi_str})"


def bin_index(p: float, bins: list[float] = ASOS_BINS) -> int | None:
    """Return the 0-based index of the bin that ``p`` falls into, or None.

    The last bin uses ``<`` on the sentinel upper edge (which is ``>1``), so
    ``p=1.0`` maps to the last bin.  ``p < 0`` or ``p >= sentinel`` returns
    None (should never occur for valid probabilities).
    """
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if lo <= p < hi:
            return i
    return None


def compute_reliability_bins(
    probs: list[float],
    outcomes: list[int],
    bins: list[float] = ASOS_BINS,
) -> list[dict[str, Any]]:
    """Compute per-bin reliability statistics.

    Parameters
    ----------
    probs:
        Sequence of predicted probabilities in ``[0, 1]``.
    outcomes:
        Sequence of binary outcomes (0 or 1), same length as ``probs``.
    bins:
        Bin edge list (default ``ASOS_BINS``).

    Returns
    -------
    List of dicts, one per non-empty bin, with keys:
        ``bin``    — human-readable label, e.g. ``'[0.00,0.05)'``
        ``n``      — count of predictions in this bin
        ``pred``   — mean predicted probability in this bin
        ``real``   — empirical YES rate (mean of outcomes) in this bin
    """
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs and outcomes must have the same length; got {len(probs)} vs {len(outcomes)}"
        )

    n_bins = len(bins) - 1
    bin_probs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_outcomes: list[list[int]] = [[] for _ in range(n_bins)]

    for p, y in zip(probs, outcomes, strict=True):
        idx = bin_index(p, bins)
        if idx is not None:
            bin_probs[idx].append(p)
            bin_outcomes[idx].append(y)

    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        n = len(bin_probs[i])
        if n == 0:
            continue
        lo = bins[i]
        hi = bins[i + 1]
        rows.append(
            {
                "bin": _bin_label(lo, hi),
                "n": n,
                "pred": sum(bin_probs[i]) / n,
                "real": sum(bin_outcomes[i]) / n,
            }
        )
    return rows
