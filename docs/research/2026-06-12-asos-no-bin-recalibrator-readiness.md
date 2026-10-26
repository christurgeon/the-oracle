# ASOS intraday low-p NO-bin recalibrator — readiness assessment (2026-06-12)

**Assessment only. Nothing fitted or deployed.** Question: do we have enough clean
forward-settled data to TAKE ON building an `intraday_asos_high` low-p NO-bin
recalibrator (gates the live flip + the reopen of the just-closed maker lane)?

**Verdict: NOT YET. The binding shortfall is day-clusters, not row count.** The valid
frame-v2 corpus is effectively **1 clean FINAL settled day (06-10)** plus 1 prelim-only
outlier day (06-11). You cannot fit + out-of-sample-validate a calibrator on 1 day.

Reproduce: `python3 scripts/research/asos_no_bin_readiness.py` (reuses settlement-join +
bracket-outcome logic from `scripts/research/asos_fv_conditioned_markout.py`).

## Method / definitions
- **Dedup**: one row per `market_ticker`, taken at the LAST evaluated timestamp before
  expiration (closest to the would-trade decision point). 81,511 raw ASOS rows →
  **174 distinct markets** → **139 settled** (35 dropped, all unsettled = today 06-12).
- **Frame split**: rows with `frame_version == 2` = valid LST corpus (06-10→06-12);
  rows MISSING `frame_version` = pre-2026-06-09 civil frame, treated as v1 (06-06→06-09).
  Clean split confirmed at the 06-09/06-10 boundary.
- **Low-p NO regime**: NO-favored rows (model `p_yes < 0.5`) binned by `p_yes`. Here
  `p_yes` IS the model's predicted NO-loss probability (YES region hitting = the NO
  contract loses). The 0.02-vs-0.17 defect = "pred ~0.02 NO-loss, realized ~0.17".
- **Settlement**: FINAL (NWS CLI) preferred over preliminary; prelim flagged separately.

## 1. Sample-count table (settled markets)

| frame | settled days | settled mkts | NO-fav | p_yes<.10 | p_yes<.15 |
|-------|--------------|--------------|--------|-----------|-----------|
| v2 (LST, valid) | 06-10, 06-11 | 46 | 37 | **22** | 27 |
| v1 (pre-fix)    | 06-06…06-09  | 93 | 75 | 45 | 49 |

Frame-v2 by cell (settled / NO-fav / p_yes<.10):
AUS 12/10/6 · CHI 10/8/6 · LAX 5/4/2 · MIA 9/7/4 · NYC 10/8/4.

**The number that decides readiness — frame-v2 low-p NO events = 22 — but across only
2 day-clusters, and one of those (06-11) is prelim-only.** Per-day breakdown of the
headline bin (p_yes<0.10):

| exp date | n | NO-losses | realized | settlement |
|----------|---|-----------|----------|------------|
| 06-10 | 10 | 2 | 0.200 | FINAL |
| 06-11 | 12 | 0 | 0.000 | **PRELIM only** (06-11 is the known outlier-settlement day; LAX missing entirely) |

The entire frame-v2 "realized NO-loss" signal is **2 losses on 1 clean FINAL day.**

## 2. NO-side reliability table (NO-favored; pred NO-loss = p_yes vs realized)

**FRAME-V2 (valid LST corpus), n_NOfav = 37**

| p_yes bin | n | pred_mean | realized | Wilson 95% | days |
|-----------|---|-----------|----------|------------|------|
| [0.00,0.05) | 20 | 0.0198 | 0.1000 | [0.028, 0.301] | 2 |
| [0.05,0.10) | 2  | 0.0725 | 0.0000 | [0.000, 0.658] | 1 |
| [0.10,0.15) | 5  | 0.1173 | 0.4000 | [0.118, 0.769] | 2 |
| [0.15,0.25) | 4  | 0.1903 | 0.0000 | [0.000, 0.490] | 2 |
| [0.25,0.50) | 6  | 0.3893 | 0.1667 | [0.030, 0.564] | 2 |

Headline defect bin (frame-v2, p_yes<0.10): **n=22, pred 0.025 vs realized 0.091**,
Wilson95 [0.025, 0.278], **2 day-clusters**.

**FRAME-V1 (pre-fix, shown separately), n_NOfav = 75** — for context only, do not pool:

| p_yes bin | n | pred_mean | realized | Wilson 95% | days |
|-----------|---|-----------|----------|------------|------|
| [0.00,0.05) | 33 | 0.0125 | 0.0303 | [0.005, 0.153] | 4 |
| [0.05,0.10) | 12 | 0.0769 | 0.1667 | [0.047, 0.448] | 3 |
| [0.10,0.15) | 4  | 0.1170 | 0.0000 | [0.000, 0.490] | 2 |
| [0.15,0.25) | 4  | 0.1886 | 0.2500 | [0.046, 0.699] | 3 |
| [0.25,0.50) | 22 | 0.4112 | 0.4091 | [0.233, 0.613] | 4 |

**Is the 0.02-vs-0.17 defect confirmed?** Directionally YES on both frames — the model
is overconfident in the low-p NO tail (predicts the NO contract loses ~2–3% of the time
when it actually loses ~7–17%). But:
- The magnitude is **softer than the 0.02→0.17 prototype claim** with current data
  (frame-v2 p_yes<0.10: 0.025→0.091; frame-v1: 0.030→0.067). It is not "gone."
- **CIs are uninformative.** Wilson on raw rows ignores day-clustering; with only 2
  frame-v2 day-clusters (one prelim), a day-clustered CI would be effectively
  unbounded. We are honest: there is not enough power to estimate the defect magnitude,
  only its sign.

## 3. Readiness verdict — NOT YET

You cannot fit AND out-of-sample-validate a recalibrator when the entire valid corpus
is 1 clean settled day. A fit on 06-10 validated on 06-11 (prelim, outlier) is not a
real OOS test; any calibrator learned here would be a 2-point overfit.

**Estimated requirement to fit + OOS-validate** (low-p NO tail, pooled across cells —
see §4): roughly **8–12 clean FINAL settled days** yielding **~80–120 low-p NO events**
(≈4 fit-days + ≈4 held-out validation-days, plus the prelim/outlier-day haircut).
Frame-v2 currently produces **~22 low-p NO events/day** (06-10 had 10 in the p<0.10
bin alone). At ~1 clean FINAL day/day of calendar (minus prelim-only days like 06-11),
**rough calendar horizon ≈ 2026-06-24 to 2026-07-01** to reach a fittable + validatable
frame-v2 corpus. This lands close to the existing favorite_bias scale gate (~06-23) and
the intraday-lockin V1 deadline (07-21), so no new monitoring cadence is needed.

## 4. Recommended approach (when fittable) + first step

**First step (now):** none on fitting. Keep accruing; watch the count. The number to
watch is **frame-v2 clean-FINAL low-p NO events (target ≥80 across ≥8 FINAL days)**,
reported by `scripts/intraday_asos_shadow_report.py` (per-bin reliability; low-p NO bin
is its kill/flip criterion per CLAUDE.md). This readiness script
(`scripts/research/asos_no_bin_readiness.py`) is the day-cluster-aware companion — rerun
it weekly to track the frame-v2 day-count and the per-exp-date FINAL/prelim split.

**Approach when the count lands:**
- **Pooled NO-tail, not per-cell.** Per-cell would give ~4–8 low-p NO events/cell —
  hopeless. The defect is a structural property of the truncated-Student-t remaining-gain
  posterior's tail (same σ-incompatibility family that killed LAX EMOS), so a single
  pooled NO-tail recalibrator is the right grain.
- **Isotonic over Platt.** The defect is a monotone tail miscalibration (pred too low
  across the whole p<0.15 region), and isotonic is already the house calibrator with a
  numpy-free runtime interpolation path. Platt's 2-parameter sigmoid can't bend a tail
  without distorting the mid-range.
- **Harness fit:** it slots into `python/data/calibration.py` as an `IsotonicCalibrator`
  keyed by strike label, BUT note `intraday_asos_high` does NOT go through
  `evaluate_market_pure` / the per-cell `weather.emos_enabled_cells` dispatcher — it
  uses the standalone `asos_evaluate`. So wiring is a new calibrator-load + apply point
  inside the ASOS path (a small dispatch add, not a free drop-in). Scope this in the
  spec, don't assume the weather harness picks it up automatically.
- **Validation gate before any live flip:** clean FORWARD per-bin reliability with the
  low-p NO bin's day-clustered CI upper bound below the economic break-even, sign-off
  from calibration-analyst + backtesting-analyst. NOT aggregate Brier.

## Caveats
- 06-12 is today: 35 markets unsettled, correctly excluded.
- 06-11 settled on PRELIM only and is the documented outlier-settlement day — it inflates
  the apparent frame-v2 day-count to 2 but contributes 0 NO-losses; do not treat as a
  clean cluster. Recheck once its FINAL CLI lands.
- Frame-v1 is shown for context only — it is the pre-LST-fix civil-frame corpus and must
  not be pooled with v2 for a fit.
