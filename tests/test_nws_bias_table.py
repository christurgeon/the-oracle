"""Invariants for the settlement-anchored (NWS CLI) bias table artifact.

Guards the shipped ``data/nws_bias/bias_correction.json`` against regeneration
regressions: schema compatibility with the runtime, EMOS-cell omission (no
double-correction), systematic cells correcting in the right direction, noise
cells SNR-gated to zero, and sparse months falling through to ``overall``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from python.data.bias_correction import BiasTable, apply_bias_correction
from python.data.weather import EnsembleForecast

NWS_TABLE = Path("data/nws_bias/bias_correction.json")
CONFIG_PATH = Path("python/config.yaml")

pytestmark = pytest.mark.skipif(
    not NWS_TABLE.exists(),
    reason="NWS bias table not built (run scripts/build_nws_bias_corpus.py + fit_nws_bias.py)",
)


def _emos_owned_cells() -> list[tuple[str, str]]:
    """``(city, var)`` EMOS owns — same source the fitter and runtime guard read."""
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cells = cfg["strategies"]["weather"].get("emos_enabled_cells") or []
    out = []
    for cell in cells:
        city, _, var = str(cell).partition("/")
        if var:
            out.append((city, var))
    return out


@pytest.fixture
def table() -> BiasTable:
    return BiasTable(NWS_TABLE)


def _corr(table: BiasTable, city: str, var: str, month: int) -> float:
    fc = EnsembleForecast(city=city, forecast_date=date(2026, month, 2), temperatures=[80.0] * 30)
    _, c = apply_bias_correction(fc, table, var, shrinkage=0.3, max_correction_f=5.0)
    return c


def test_table_loads(table: BiasTable) -> None:
    assert table.loaded


def test_emos_cell_omitted(table: BiasTable) -> None:
    # EMOS already absorbs its cells' bias via a + b*ens_mean — no double-count.
    for city, var in _emos_owned_cells():
        assert table.get_bias(city, var, 5) is None, (
            f"{city}/{var} is EMOS-owned but present in the NWS bias table — "
            f"regenerate with --exclude-cells {city}/{var}"
        )


def test_systematic_cold_cell_warms(table: BiasTable) -> None:
    # MIA/high runs ~3.9°F cold (forecast - actual < 0); correction must be +ve
    # (shifts the ensemble warmer toward the settled high).
    c = _corr(table, "MIA", "high", 6)
    assert c > 0.5, f"expected warming correction for cold-biased MIA/high, got {c:+.2f}"


def test_noise_cell_snr_gated(table: BiasTable) -> None:
    # NYC/high bias is not statistically systematic (|t| < 2) -> gated to zero.
    assert _corr(table, "NYC", "high", 6) == 0.0


def test_sparse_month_falls_through_to_overall(table: BiasTable) -> None:
    # June (month 6) is sparse and must be dropped from by_month, so a June
    # lookup returns the well-powered 'overall' entry rather than a low-n stub.
    entry = table.get_bias("MIA", "high", 6)
    assert entry is not None
    assert entry.count >= 50  # overall n, not a 1-2 sample June stub


def test_every_by_month_is_well_powered() -> None:
    # Runtime get_bias returns a present monthly entry immediately (no fallthrough
    # to overall on low SNR). So every shipped by_month entry must be powered
    # enough not to be SNR-gated to a silent zero. Guards against a regeneration
    # with too-low --min-month-count.
    data = json.loads(NWS_TABLE.read_text())
    for city, var_block in data.items():
        for var, entry in var_block.items():
            for month, stat in entry.get("by_month", {}).items():
                assert stat["count"] >= 15, f"{city}/{var} month {month} count={stat['count']} < 15"


def test_no_emos_cell_in_table() -> None:
    # has_cell() backs the orchestrator double-correction guard.
    table = BiasTable(NWS_TABLE)
    for city, var in _emos_owned_cells():
        assert not table.has_cell(city, var), f"{city}/{var} would trip the runtime guard"
    assert table.has_cell("MIA", "high")  # a kept systematic cell
