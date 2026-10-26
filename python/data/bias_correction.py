"""
GFS ensemble bias correction using historical GFS-deterministic vs ERA5 data.

Applies a location-shift correction to ensemble member temperatures before
bracket fraction computation.  Preserves ensemble spread; only corrects
the distribution center.

Safety rails:
  - Shrinkage (default 0.3): applies only 30% of measured bias
  - Max correction cap (default 5°F): prevents catastrophic overcorrection
  - SNR gate: skips correction when bias is not statistically significant
  - Controlled by ``bias_correction_enabled`` in config (enabled in production since 2026-03-21)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from python.common import DailyVariable
from python.data.weather import EnsembleForecast

logger = logging.getLogger(__name__)

# Default path to the bias correction JSON (relative to repo root)
_DEFAULT_BIAS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "gfs_bias" / "bias_correction.json"
)

# DailyVariable is a StrEnum: members compare and hash equal to their string
# values, so a single entry per enum covers both enum and raw-string lookups.
_VARIABLE_TO_BIAS_KEY: dict[str, str] = {
    DailyVariable.TEMP_MAX: "high",
    DailyVariable.TEMP_MIN: "low",
}

# Minimum signal-to-noise ratio to apply a correction.
# SNR = |mean_bias| / (std_bias / sqrt(count)).  Below this threshold,
# the measured bias is not statistically distinguishable from zero.
_MIN_SNR = 2.0


@dataclass(frozen=True)
class BiasEntry:
    """Bias statistics for a single city/variable/month combination."""

    count: int
    mean_bias_f: float
    mae_f: float
    rmse_f: float
    std_bias_f: float

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio: |mean_bias| / standard_error."""
        if self.count <= 0 or self.std_bias_f <= 0:
            return 0.0
        se = self.std_bias_f / math.sqrt(self.count)
        return abs(self.mean_bias_f) / se


class BiasTable:
    """
    Lookup table for GFS bias corrections.

    Loads ``bias_correction.json`` and exposes ``get_bias()`` with a
    fallback chain: monthly → overall → None.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._data: dict[str, Any] = {}
        load_path = path or _DEFAULT_BIAS_PATH
        try:
            with open(load_path) as f:
                self._data = json.load(f)
            logger.info("Loaded bias correction table from %s", load_path)
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Failed to load bias correction table from %s — corrections disabled",
                load_path,
                exc_info=True,
            )

    @property
    def loaded(self) -> bool:
        return bool(self._data)

    def has_cell(self, city: str, variable: str) -> bool:
        """True if the table carries any bias entry for ``city``/``variable``.

        Used by the orchestrator to detect overlap with EMOS-owned cells (EMOS
        already corrects the mean via ``a + b*ens_mean``; a table entry there
        would double-correct). ``variable`` accepts "high"/"low" or a
        ``DailyVariable`` string value (mirrors ``get_bias``).
        """
        bias_key = _VARIABLE_TO_BIAS_KEY.get(variable, variable)
        return bias_key in self._data.get(city, {})

    def get_bias(self, city: str, variable: str, month: int) -> BiasEntry | None:
        """
        Look up bias for a city/variable/month.

        Fallback chain: by_month[month] → overall → None.

        Args:
            city: City code (NYC, CHI, MIA, etc.)
            variable: "high", "low", or a DailyVariable string value.
            month: Calendar month (1-12).

        Returns:
            BiasEntry if data exists, else None.
        """
        bias_key = _VARIABLE_TO_BIAS_KEY.get(variable, variable)
        city_data = self._data.get(city)
        if city_data is None:
            return None

        var_data = city_data.get(bias_key)
        if var_data is None:
            return None

        monthly = var_data.get("by_month", {}).get(str(month))
        if monthly is not None:
            entry = self._build_entry(monthly, city, bias_key, "monthly")
            if entry is not None:
                logger.debug(
                    "Bias lookup %s/%s/month=%d: monthly (bias=%.2f°F, SNR=%.1f)",
                    city,
                    bias_key,
                    month,
                    entry.mean_bias_f,
                    entry.snr,
                )
                return entry

        overall = var_data.get("overall")
        if overall is not None:
            entry = self._build_entry(overall, city, bias_key, "overall")
            if entry is not None:
                logger.debug(
                    "Bias lookup %s/%s/month=%d: overall fallback (bias=%.2f°F, SNR=%.1f)",
                    city,
                    bias_key,
                    month,
                    entry.mean_bias_f,
                    entry.snr,
                )
                return entry

        return None

    @staticmethod
    def _build_entry(
        payload: dict[str, Any],
        city: str,
        bias_key: str,
        scope: str,
    ) -> BiasEntry | None:
        """Construct a ``BiasEntry`` from raw JSON, swallowing schema drift.

        A schema bump that adds/renames fields would otherwise raise
        ``TypeError`` at signal time and crash the strategy.
        """
        try:
            return BiasEntry(**payload)
        except TypeError as exc:
            logger.warning(
                "Bias schema mismatch for %s/%s (%s): %s — corrections skipped",
                city,
                bias_key,
                scope,
                exc,
            )
            return None


def apply_bias_correction(
    forecast: EnsembleForecast,
    bias_table: BiasTable,
    variable: str,
    *,
    shrinkage: float = 0.3,
    max_correction_f: float = 5.0,
) -> tuple[EnsembleForecast, float]:
    """
    Apply bias correction to an ensemble forecast.

    Shifts all ensemble member temperatures by ``-mean_bias * shrinkage``,
    capped at ±max_correction_f.  Skips correction if the measured bias
    is not statistically significant (SNR < 2.0).

    Args:
        forecast: Original ensemble forecast.
        bias_table: Loaded bias correction table.
        variable: "high", "low", or DailyVariable string value.
        shrinkage: Fraction of measured bias to apply (default 0.3).
        max_correction_f: Maximum absolute correction in °F (default 5.0).

    Returns:
        (corrected_forecast, correction_applied_f).
        Returns (original_forecast, 0.0) if no correction is warranted.

    INVARIANT: When a correction IS applied, this function MUST return a new
    ``EnsembleForecast`` whose ``temperatures`` is a freshly-constructed list
    (not aliased to the input). The input forecast's ``temperatures`` list
    must not be mutated. The EMOS shadow path (python/data/emos.py)
    reconstructs the pre-correction members as
    ``[m + shift for m in raw_gfs.temperatures]`` and relies on the input's
    list being unchanged. The no-op return paths (``forecast, 0.0``) return
    the original instance unchanged. See
    tests/test_bias_correction.py::TestApplyBiasCorrectionImmutability for
    the regression check.
    """
    month = forecast.forecast_date.month
    entry = bias_table.get_bias(forecast.city, variable, month)

    if entry is None:
        return forecast, 0.0

    if entry.snr < _MIN_SNR:
        logger.debug(
            "Skipping bias correction for %s/%s/month=%d: SNR=%.1f < %.1f",
            forecast.city,
            variable,
            month,
            entry.snr,
            _MIN_SNR,
        )
        return forecast, 0.0

    correction = -entry.mean_bias_f * shrinkage
    correction = max(-max_correction_f, min(max_correction_f, correction))

    if abs(correction) < 0.01:
        return forecast, 0.0

    corrected = replace(
        forecast,
        temperatures=[t + correction for t in forecast.temperatures],
    )

    logger.debug(
        "Bias correction for %s/%s/month=%d: %.2f°F (raw_bias=%.2f, shrinkage=%.1f, SNR=%.1f)",
        forecast.city,
        variable,
        month,
        correction,
        entry.mean_bias_f,
        shrinkage,
        entry.snr,
    )

    return corrected, correction
