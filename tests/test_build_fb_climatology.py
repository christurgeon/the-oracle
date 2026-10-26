from __future__ import annotations

from scripts.build_fb_climatology import aggregate_cells, build_artifact


def _obs(city, od, high, low, prelim=False):
    return {
        "city": city,
        "observation_date": od,
        "high_f": high,
        "low_f": low,
        "is_preliminary": prelim,
    }


def test_aggregate_dedups_final_over_prelim() -> None:
    rows = [
        _obs("NYC", "2026-07-01", 90, 70, prelim=True),
        _obs("NYC", "2026-07-01", 88, 68, prelim=False),  # final wins
        _obs("NYC", "2026-07-02", 92, 72, prelim=False),
    ]
    cells = aggregate_cells(rows, min_month_n=1)
    key = "NYC_high_7"
    assert cells[key]["n"] == 2
    assert abs(cells[key]["mu"] - 90.0) < 1e-9  # (88 + 92) / 2


def test_pool_cell_spans_months() -> None:
    rows = [
        _obs("MIA", "2026-05-01", 85, 70),
        _obs("MIA", "2026-06-01", 89, 74),
    ]
    cells = aggregate_cells(rows, min_month_n=1)
    assert "MIA_high_pool" in cells
    assert cells["MIA_high_pool"]["n"] == 2


def test_build_artifact_shape() -> None:
    rows = [_obs("NYC", "2026-06-01", 80, 60), _obs("NYC", "2026-06-02", 84, 64)]
    art = build_artifact(rows, min_month_n=1)
    assert art["schema_version"] == 1
    assert "NYC_high_6" in art["cells"]
    assert set(art["cells"]["NYC_high_6"]) >= {"mu", "sigma", "n"}
