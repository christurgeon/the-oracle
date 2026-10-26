# ASOS Intraday Nowcast — Phase 0 De-risking Findings (2026-06-05)

Resolves the four gating investigations from the implementation plan before any
strategy code is built.

## 0.1 — ASOS station == CLI settlement station: PASS

`scripts/research_2026_06_05_asos_station_check.py` →
`data/research/2026-06-05-asos-station-check.json`. The prototype ASOS CSVs use
the **CLI settlement station** for every city, including the CHI trap:

| City | ASOS station | CLI settlement | Match | Trading cell |
|---|---|---|---|---|
| NYC | NYC | NYC | ✓ | yes |
| CHI | **MDW** | **MDW** | ✓ | yes |
| MIA | MIA | MIA | ✓ | yes |
| AUS | AUS | AUS | ✓ | yes |
| LAX | LAX | LAX | ✓ | informational |

**Decision:** no trading cell excluded. `enabled_cells = [MIA/high, AUS/high,
CHI/high, NYC/high]` stands. The runtime `asos.py` station map must use these
(CHI=MDW, NOT ORD); confirm IEM prefix (bare `MDW` vs `KMDW`) during Phase 2.

## 0.2 — Post-guard accrual window: ~10 weeks

From the guarded reliability cohort (36 trades / 22 prototype days at 2000Z):
~1.6 guarded trades/day → **~49 trading days (~10 weeks) to n≥80**; low-p
(`p<0.10`) NO bin ~1.1/day → ~18 days to n≥20 in-bin. This is the 2000Z-only
rate; production evaluates continuously intraday, so the real rate is likely
higher. **Non-blocking; informational.** Re-measure on live shadow data.

## 0.3 — Preliminary-CLI availability at trade time: keep bad-prelim informational

The strategy trades the post-peak window, where a same-day preliminary CLI high
is typically not yet published. **Decision:** ship `bad_prelim_active: false`
(the conservative spec default) — log `prelim_drift_f` as an audit field only.
The forward shadow corpus reveals true trade-time availability empirically;
promote to an active guard later only if availability ≥ ~30%. Zero risk
(shadow-only, informational field).

## 0.4 — Deference open-position key parity: source confirmed

`_open_event_keys()` exists (`orchestrator.py:1353`) and **already feeds
`apply_favorite_bias_deference`** (`orchestrator.py:1601`) for exactly this
purpose (drop a signal sharing an event key with an open position). So the ASOS
deference hook's open-position arm has a proven key source. The Phase-6 build
includes an explicit parity test (Task 0.4 / 6.1) pinning that an intraday
KXHIGH open-position ticker and a matching ASOS `TradeSignal` produce equal
`_event_key_from_signal` keys.
