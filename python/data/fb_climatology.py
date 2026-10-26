"""Hermetic strike-climatology math + artifact loader for favorite_bias.

Provides (city,var,month)->(mu,sigma) daily-high/low climatology so the
strategy can score a strike's rarity (z) and climatological base rate (q)
at signal time. Fail-open by construction: a missing/corrupt artifact yields
a disabled table whose lookups all return None. Imports limited to stdlib +
scipy (mirrors emos.py / asos.py); MUST NOT import production strategy code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import norm

_VALID_STRIKE_TYPES = frozenset({"greater", "less"})


@dataclass(frozen=True)
class ClimEntry:
    mu: float
    sigma: float
    n: int
    source: str  # "month" | "pool"


def strike_z(strike: float | None, mu: float | None, sigma: float | None) -> float | None:
    """(strike - mu) / sigma, or None if any input missing or sigma <= 0."""
    if strike is None or mu is None or sigma is None or sigma <= 0:
        return None
    return (strike - mu) / sigma


def clim_yes_prob(
    strike: float | None, strike_type: str, mu: float | None, sigma: float | None
) -> float | None:
    """Coarse climatological P(YES) prior under N(mu, sigma), continuous CDF.

    Not settlement-exact (no NWS integer rounding) -- a logged prior only.
    less: P(high < strike) = Phi(z); greater: P(high > strike) = 1 - Phi(z).
    Returns None for non-greater/less types or unusable stats.
    """
    z = strike_z(strike, mu, sigma)
    if z is None or strike_type not in _VALID_STRIKE_TYPES:
        return None
    cdf = float(norm.cdf(z))
    return cdf if strike_type == "less" else 1.0 - cdf


@dataclass(frozen=True)
class FbClimatologyTable:
    cells: dict[str, ClimEntry]
    artifact_hash: str | None
    is_loaded: bool
    fallback_reason: str | None

    @classmethod
    def _disabled(cls, reason: str) -> FbClimatologyTable:
        return cls(cells={}, artifact_hash=None, is_loaded=False, fallback_reason=reason)

    @classmethod
    def load(cls, path: Path | str | None) -> FbClimatologyTable:
        """Never-raises loader (mirrors EmosTable.load). Returns a disabled
        table on ANY failure so favorite_bias.__init__ can never crash-loop."""
        if path is None:
            return cls._disabled("no-path")
        p = Path(path)
        if not p.exists():
            return cls._disabled("file-missing")
        try:
            raw = p.read_bytes()
        except OSError as exc:
            return cls._disabled(f"io-error:{exc.errno}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return cls._disabled("parse-error")
        try:
            cells_raw = payload["cells"]
            if not isinstance(cells_raw, dict):
                raise ValueError("cells must be a dict")
            cells: dict[str, ClimEntry] = {}
            for key, c in cells_raw.items():
                sigma = float(c["sigma"])
                source = "pool" if str(key).endswith("_pool") else "month"
                cells[key] = ClimEntry(mu=float(c["mu"]), sigma=sigma, n=int(c["n"]), source=source)
        except (KeyError, ValueError, TypeError) as exc:
            return cls._disabled(f"schema-error:{exc}")
        artifact_hash = hashlib.sha256(raw).hexdigest()
        return cls(cells=cells, artifact_hash=artifact_hash, is_loaded=True, fallback_reason=None)

    def lookup(self, city: str, var: str, month: int) -> ClimEntry | None:
        """(city,var,month) -> (city,var) pool fallback -> None. Never raises."""
        entry = self.cells.get(f"{city}_{var}_{month}")
        if entry is not None:
            return entry
        return self.cells.get(f"{city}_{var}_pool")
