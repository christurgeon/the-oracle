"""Build the slim MIA fixture for the §8.2 integration test.

One-shot helper. Reads the canonical 2026-05-08/1500 snapshot trio,
distills it to the MIA/high `KXHIGHMIA-26MAY08-*` brackets (forecast_date
2026-05-08, settled 93 °F final on 2026-05-09's CLI), and writes the result
under `tests/research/fixtures/snapshots_step0_slim/2026-05-08/1500/`.

Why forecast_date == snapshot_day (2026-05-08): the spike's reported
`(μ=92.736, σ=0.993)` for MIA/high comes from the GFS ensemble on
forecast_date 2026-05-08 (ens_mean=88.7533, ens_std=0.6771 ddof=1).
The settlement is colocated with the run dir (rather than its source
`2026-05-09/2000/settlements.json`) so `load_snapshot_data` joins cleanly
inside the slim fixture's single-day footprint.

Plan §F.1; spec §8.2 + §11 PR 2 / §13 / A15.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SRC_RUN_DIR = _REPO_ROOT / "data" / "snapshots" / "2026-05-08" / "1500"
_SRC_SETTLEMENTS = _REPO_ROOT / "data" / "snapshots" / "2026-05-09" / "2000" / "settlements.json"

_DST_RUN_DIR = (
    _REPO_ROOT / "tests" / "research" / "fixtures" / "snapshots_step0_slim" / "2026-05-08" / "1500"
)

_FORECAST_DATE = "2026-05-08"
_OBSERVATION_DATE = "2026-05-08"
_CITY = "MIA"
_DAILY_VAR = "temperature_2m_max"
_TICKER_PREFIX = "KXHIGHMIA-26MAY08-"

_SIZE_BUDGET_BYTES = 30 * 1024


def _filter_markets(markets: list[dict]) -> list[dict]:
    return [m for m in markets if str(m.get("ticker", "")).startswith(_TICKER_PREFIX)]


def _filter_forecasts(forecasts: list[dict]) -> list[dict]:
    return [
        f
        for f in forecasts
        if f.get("city") == _CITY
        and f.get("forecast_date") == _FORECAST_DATE
        and f.get("daily_variable") == _DAILY_VAR
    ]


def _filter_settlements(settlements: list[dict]) -> list[dict]:
    return [
        s
        for s in settlements
        if s.get("city") == _CITY and s.get("observation_date") == _OBSERVATION_DATE
    ]


def _write_json(path: Path, payload: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text)
    return len(text.encode("utf-8"))


def main() -> None:
    markets_src = json.loads((_SRC_RUN_DIR / "markets.json").read_text())
    forecasts_src = json.loads((_SRC_RUN_DIR / "weather_forecasts.json").read_text())
    settlements_src = json.loads(_SRC_SETTLEMENTS.read_text())

    markets_slim = _filter_markets(markets_src)
    forecasts_slim = _filter_forecasts(forecasts_src)
    settlements_slim = _filter_settlements(settlements_src)

    if len(markets_slim) != 6:
        raise SystemExit(
            f"Expected 6 KXHIGHMIA-26MAY08-* markets in source; found {len(markets_slim)}. "
            "Source snapshot may have rotated."
        )
    if not forecasts_slim:
        raise SystemExit("No MIA temperature_2m_max forecast for 2026-05-08 in source snapshot.")
    if any(len(f.get("temperatures") or []) < 25 for f in forecasts_slim):
        thin = [(f.get("forecast_date"), len(f.get("temperatures") or [])) for f in forecasts_slim]
        raise SystemExit(
            f"Source forecast has < 25 members; cannot pass FORECAST_THIN_MEMBERS gate. {thin}"
        )
    if len(settlements_slim) != 1:
        raise SystemExit(
            f"Expected exactly 1 MIA 2026-05-08 settlement; found {len(settlements_slim)}."
        )
    if settlements_slim[0].get("is_preliminary"):
        raise SystemExit(
            "MIA 2026-05-08 settlement in source is preliminary; "
            "the §8.2 outcome assertions require a final settlement."
        )

    n_markets = _write_json(_DST_RUN_DIR / "markets.json", markets_slim)
    n_forecasts = _write_json(_DST_RUN_DIR / "weather_forecasts.json", forecasts_slim)
    n_settlements = _write_json(_DST_RUN_DIR / "settlements.json", settlements_slim)
    total = n_markets + n_forecasts + n_settlements

    print(f"markets.json:            {n_markets:>6d} bytes  ({len(markets_slim)} markets)")
    print(
        f"weather_forecasts.json:  {n_forecasts:>6d} bytes  ({len(forecasts_slim)} forecasts, "
        f"n_members={len(forecasts_slim[0]['temperatures'])})"
    )
    print(f"settlements.json:        {n_settlements:>6d} bytes  ({len(settlements_slim)} entries)")
    print(f"total:                   {total:>6d} bytes  (budget: {_SIZE_BUDGET_BYTES})")

    if total > _SIZE_BUDGET_BYTES:
        raise SystemExit(
            f"Slim fixture exceeds {_SIZE_BUDGET_BYTES} byte budget (got {total}). "
            "Per plan §F.1 / v1.1 A15: bump budget to 35 KB and document the deviation."
        )


if __name__ == "__main__":
    main()
