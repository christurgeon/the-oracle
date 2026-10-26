# ASOS-Bayesian Intraday Same-Day Predictor — Prototype Spec (v2)

**Status:** Pre-registered (2026-05-21, v2 after paranoid review).
**Decision artifact:** `docs/research/2026-05-21-asos-bayesian-prototype-results.md`.
**Falsification criteria:** locked below — not relaxable mid-analysis.

## 0. Changelog

**v1 → v2** — Spec-reviewer's adversarial pass surfaced 4 critical +
5 important findings:
- **§2 corpus** restated empirically: 48 of 58 days are slot-complete;
  10 days excluded.
- **§3 mechanism** uses left-truncated Gaussian for the remaining-gain
  posterior (gain ≥ 0 by construction).
- **§5 gates** drop the INCONCLUSIVE soft-out; G3 replaced with
  aggregate net-of-fees P&L + bootstrap CI bound; LAX moved to a
  reported-but-non-gating cell (snapshot-time artifact at 2000Z).
- **§6 baseline** uses the morning RAW ensemble distribution for the
  G1 CRPS gate (no calibrator needed in temperature space). For G3
  P&L only, applies the today-HEAD calibrator artifact symmetrically
  to both arms, disclosed as lookforward bias.
- **§10 timezone** resolved upfront: ASOS column is UTC (verified).
- Per-day cap tie-break, depth haircut sensitivity, integer-cent
  fills, and power analysis all pinned.

**v2 → v2.1** — Second-pass review surfaced 2 additional concerns:
- **§4 σ_climo lower bound** raised from `> 0` to `≥ 0.5 °F` to avoid
  numerical degeneracy of the truncated CDF.
- **§6.1 calibrator-extrapolation diagnostic** added as a gating
  check on G3: if >25% of trade-eligible rows have either arm's raw
  probability outside the calibrator's empirical 5th-95th training
  support, G3 becomes non-falsifying. This blocks the silent failure
  mode where a truncated-Gaussian posterior produces raw probabilities
  near 0/1 outside the isotonic plateau bounds.
- **§5.2 paired-CRPS denominator** corrected from 600 to 384 (the
  unit is `(date, city, var)`, not row). MDE ≈ 0.053 °F is still
  below the 0.1 °F gate — adequate power.
- **§5.2 G2 reporting** clarified to require BOTH pre- and
  post-tie-break counts; post < 20 is the binding G2 fail.
- **§6.3 diagnostic** added: per-cell P&L of discarded 1400Z trades
  for tie-break direction sanity (reported, non-gating).

## 1. Purpose

The 2026-05-20 weather-strategy research doc §6 identifies an
unexploited data advantage: the bot reads zero ASOS hourly
observations, so at any intraday snapshot it trades brackets with a
stale morning forecast against a market that has fresh observations
available. The prototype determines whether updating the morning
ensemble with observed `max_so_far(t)` plus a climatological
"remaining gain" distribution produces a sharper posterior than the
morning-only baseline — sharply enough to flip net-of-fees P&L
positive across the ASOS-covered cells.

This spec is a **go/no-go gate**, not a feature build. Output is a
notebook + a write-up. If the gates pass, formal `ship-feature`
lifecycle begins. If they fail, the idea is killed.

## 2. Empirical data inventory (verified 2026-05-21)

### 2.1 ASOS hourly

| City | File | Span | Rows | Cadence |
|---|---|---|---|---|
| NYC, CHI, LAX, MIA, AUS | `data/research/asos/k{nyc,chi,lax,mia,aus}_asos.csv` | 2024-05-01 → 2026-05-14 | ~17,800 each | hourly, METAR :51 convention |

Columns: `station, valid, tmpf`. **Timezone of `valid` column is
UTC** (empirically verified: NYC mean tmpf peaks at hour 18 UTC =
14 EDT, troughs at hour 09 UTC = 05 EDT — matches local
diurnal cycle).

### 2.2 Snapshot corpus

Eval window: 2026-03-18 → 2026-05-14 = 58 calendar days.

| Bucket | Count | Days |
|---|---|---|
| Slot-complete `{0800, 1400, 2000}` UTC | **48** | clean corpus |
| Legacy top-level (pre-slot-subdir schema) | 8 | 2026-03-18 → 2026-03-25 — **excluded** |
| Partial slots | 2 | 2026-03-26 (only 2000Z), 2026-05-05 (missing 1400Z) — **excluded** |
| Empty | 0 | — |

**Pre-registered eval corpus: 48 days × 5 cities × 2 vars × 2
snapshot times {1400Z, 2000Z}** = nominal 960 evaluation rows
before any filter.

### 2.3 Settlement availability

Pre-checked in `data/snapshots/<date>/<slot>/settlements.json` (newer
layout) or `data/snapshots/<date>/settlements.json` (older).
Settlement record: `{city, observation_date, high_f, low_f,
is_preliminary, issued_at}`.

Settlement priority rule: **prefer `is_preliminary=False`. If only
preliminary available, use it and flag.** This is the same rule
`scripts/build_emos_corpus.py` and `daily_trade_report.py` apply
(see `scripts/_calibration_common.py`).

### 2.4 Forecast row schema

In `weather_forecasts.json`: list of
`{city, forecast_date, daily_variable, temperatures[30 ensemble members],
mean, std_dev, model, model_version}`.

The 30-member ensemble is the production GFS source; ECMWF has its
own row with model="ecmwf_ifs04". Production blends them weighted —
the prototype uses the same blend weights from `python/config.yaml`
`weather.gfs_weight` / `weather.ecmwf_weight`.

## 3. Mechanism (locked)

For a market on city C, observation date D, strike S, snapshot time
t ∈ {1400Z, 2000Z}, with `var ∈ {high, low}`:

### 3.1 Morning baseline (raw ensemble, no calibrator)

Pull `weather_forecasts.json` from `data/snapshots/D/0800/`. Blend
GFS + ECMWF ensemble members per production weights. This yields an
empirical CDF over `daily_max(D, C)` (or `daily_min` for low).

For G1 CRPS, the baseline forecast distribution is this empirical
ensemble CDF directly. **No calibrator is applied to either arm for
G1** — CRPS is in temperature space and the calibrator question
doesn't enter.

### 3.2 Intraday observation

Pull ASOS hourly rows for city C with `valid` in
`[local_day_start_utc(D, C), t]`. For `var=high`,
compute `M = max(tmpf)` over those rows; for `var=low`, `M = min(tmpf)`.

`local_day_start_utc(D, C)` converts the *local* calendar date D
in city C to the UTC instant of 00:00 local. (Kalshi settles on
NWS CLI which uses local-day high/low — production already does
this mapping in `python/data/nws.py` `CITY_TO_CLI_LOCATION`.)

**Missing-hour tolerance.** If the gap between the latest available
ASOS observation and snapshot time t exceeds **2 hours**, drop this
`(D, C, var, t)` row from the eval corpus and log the drop. No
imputation.

### 3.3 Climatology lookup

Look up `(μ_climo, σ_climo, n)` from the climatology table (§4)
keyed on `(city, month_of_local(D), hour_of_t_utc)`.

### 3.4 Posterior (left-truncated Gaussian remaining gain)

Gain is bounded below by 0 for `var=high` (max-so-far cannot exceed
the eventual daily max; remaining gain ≥ 0). The Gaussian assumption
from v1 was wrong near `M ≈ S`, where the strategy actually trades.
v2 uses the **left-truncated Gaussian** with truncation at 0:

```
If M ≥ S (high markets):  P(daily_max ≥ S) = 1
Else:
    z_S = (S - M - μ_climo) / σ_climo
    z_0 = (0 - μ_climo) / σ_climo = -μ_climo / σ_climo
    P(daily_max ≥ S) = (1 - Φ(z_S)) / (1 - Φ(z_0))
```

For `var=low`, mirror: gain ≤ 0; use right-truncated Gaussian at 0.

This posterior is a **raw** model probability. For trade simulation
(G3 only) it is then passed through today's calibrator artifact —
see §6.4.

## 4. Climatology table

**Key:** `(city, month_1_12, hour_of_day_utc_0_23)`. The hour key is
the *snapshot UTC hour*, not local — to match how the posterior is
queried at trade time.

**Values:** `(μ_climo_gain, σ_climo_gain, n_samples)`.

**Definition.**
`gain(d, t) = daily_max(d) − max_so_far(d, t)`
where both are computed in the *city local day frame* against the
local-day daily max from ASOS.

**Fit window:** 2024-05-01 → 2026-03-17 (ASOS coverage MINUS the eval
window). Eval window held out.

**Per-cell sample-size gate (HARD, replaces v1 "log and continue").**
- `n ≥ 30`: use cell directly.
- `n < 30`: fall back to `(city, hour)` pooling across months. Pool
  requires `n_pool ≥ 60`.
- `n_pool < 60`: drop the cell from the eval corpus and report in
  results as `climatology_unavailable`.

**Sanity gates (HARD, drop violators):**
- `μ_climo ≥ 0` for `var=high` and `μ_climo ≤ 0` for `var=low`.
- `σ_climo ≥ 0.5 °F`. A tighter σ is almost certainly a small-sample
  artifact (1°F integer ASOS readings → minimum resolvable σ on a
  full sample is ~0.3°F; below 0.5°F means n is borderline-too-small
  to estimate the variance, *and* produces numerical degeneracy in
  the §3.4 truncated CDF when `μ_climo/σ_climo` is large). Drop
  violators.

**Diagnostic in write-up:** report the fraction of eval rows that
hit fallback pooling. If >25%, the result reflects a different
mechanism than the headline — flag in the decision.

**Output artifact:** `data/research/asos_climatology.json`.

## 5. Falsification gates (pre-registered, NOT relaxable)

All three must pass for GREEN. Any failure → KILL. There is no
INCONCLUSIVE branch.

### G1 — Sharpness in temperature space

CRPS of the intraday posterior at **2000Z** vs the morning ensemble
baseline, computed against actual NWS-final settlement temperature,
per `(date, city, var)`.

**Pass threshold (both must hold):**
- Mean CRPS reduction (baseline − intraday) ≥ **0.1 °F** at 2000Z.
- Bootstrap 95% lower CI bound on the mean ≥ **0.0 °F**
  (B = 1000, paired by `(date, city, var)`).

**Why 2000Z and not 1400Z:** 1400Z is pre-peak in every city
(observed `M` is mostly the overnight low or early-morning warm-up);
the mechanism is *not expected* to clear noise there. Reported but
not gating.

### G2 — Volume

Count of `(date, city, var, snapshot)` rows that would have produced
a trade after applying: production `edge_threshold`, production
`min_calibrated_prob` floor, integer-cent fill rounding, and the
depth haircut from §6.5 at `/10`.

**Pass threshold:** ≥ **20 trades** across the full eval corpus.

### G3 — P&L (aggregate; LAX excluded from primary gate)

Net-of-fees simulated P&L over the eval corpus, aggregated across
cities **excluding LAX** (LAX is structurally disadvantaged at 2000Z
because peak is ~22-23Z — see §5.1).

**Pass threshold (both must hold):**
- Point estimate aggregate P&L ≥ **0**.
- Bootstrap 95% lower CI bound (B = 1000, day-stratified resample)
  ≥ a pre-registered small negative floor.

**LAX is reported separately** as an informational cell. A LAX
negative does not count against the gate; a LAX positive is a
bonus signal.

### 5.1 Why LAX is non-gating

At 2000Z (1 PM PT), LAX is ~3 hours pre-peak (typical peak 22-23Z).
The mechanism specifically claims to update on observed max-so-far;
when `max_so_far(20Z)` reflects pre-peak temperature, the
informational gain is small. A LAX failure at 2000Z is a
snapshot-time artifact, not a mechanism falsification. Production
deployment would use a later snapshot (e.g., a not-yet-collected
2300Z) for LAX; the prototype cannot evaluate that.

### 5.2 Power note

The G1 paired-difference unit is `(date, city, var)` at the 2000Z
snapshot — one paired CRPS observation per combination. For all 5
cities pre-tie-break that's 48 × 5 × 2 = **480 paired observations**;
for the G3-gating 4 cities (LAX excluded from primary) that's 48 × 4
× 2 = **384 pairs**. After expected ~10% post-filter attrition
(climatology unavailable, ASOS gap, settlement unavailable) expect
~340 paired CRPS observations available to G1 gating.

Minimum-detectable-effect for G1: with paired σ_diff ≈ 0.5 °F (rough
prior from EMOS-vs-isotonic comparisons in the codebase),
`MDE = 1.96 · 0.5 / √340 ≈ 0.053 °F` at 95% confidence — below the
0.1 °F gate. Adequate power. **If the empirical σ_diff > 1.5 °F the
gate is underpowered** — flag in the write-up.

G2 post-filter trade count expected in the 20-100 range across the
trade-eligible corpus.

**G2 trade count must be reported BOTH pre- and post-tie-break.**
The §6.3 "later snapshot wins" rule discards 1400Z trades on any
`(C, V, D)` where 2000Z also fires. Pre-tie-break count ≥ 20 with
post-tie-break < 20 is a borderline pass — flag in write-up but
treat post-tie-break < 20 as G2 fail.

## 6. Methodology — locked decisions on sharp edges

### 6.1 Production-fidelity baseline (resolved with reviewer-flagged downgrade)

`state/eval_audit.jsonl` local coverage is only 2026-05-07 → 2026-05-10
+ 2026-05-21. The 38 of 48 eval days before 2026-05-07 have no
recoverable production-side audit row.

**Resolution:**
- **For G1 (CRPS):** baseline is the morning RAW ensemble
  distribution from `weather_forecasts.json` — no calibrator needed.
  This bypasses the artifact-recovery problem entirely. G1
  comparison is fair across all 48 days.
- **For G3 (P&L):** today's calibrator artifacts (EMOS coefficients
  at `data/emos/coefficients.json` HEAD, isotonic at
  `data/calibration/isotonic.json` HEAD) are applied **symmetrically
  to both the morning baseline probability and the intraday
  posterior**. This is a lookforward bias on the absolute P&L level
  but the *delta* between the two arms is preserved *iff* both arms'
  raw probability inputs sit in the calibrator's training-distribution
  support.

**Calibrator-extrapolation diagnostic (NEW v2.1, gating).** Before
running G3, compute the 5th and 95th percentiles of the calibrator's
training input distribution (for isotonic cells: the breakpoints in
`isotonic.json`; for EMOS cells: the 5th/95th of `emos_prob_raw`
across a sample of production audit rows). For each evaluation row,
flag if either `p_morning_raw` or `p_intraday_raw` falls outside
that support — this is calibrator extrapolation.

- If `extrapolation_rate ≤ 25%` across the trade-eligible rows: G3
  proceeds as written.
- If `extrapolation_rate > 25%`: G3 is treated as **non-falsifying**
  (reported but not gating). The intraday posterior is producing
  raw probabilities outside production's empirical training band,
  and the calibrated comparison is no longer apples-to-apples.

Rationale: the isotonic plateau (`block_raw_prob_range = [0.649, 0.967]`)
is fit on the production raw-prob distribution; a truncated-Gaussian
posterior driven by `M ≈ S` will routinely produce raw probabilities
near 0 or 1, far outside that support. Without this guard the
calibrator extrapolation would silently inflate or compress the G3
delta.

### 6.2 Climatology leakage

Fit window 2024-05-01 → 2026-03-17 is disjoint from eval window
2026-03-18 → 2026-05-14. The eval window's seasonal cycle is
represented in the fit corpus by exactly one prior pass
(March–May 2025). Per-cell fallback pooling rate is a primary
diagnostic in the write-up; >25% fallback flags the result as
operating on the *pooled* mechanism, not the headline.

### 6.3 Per-day cap tie-break

Production cap allows 1 bet per `(city, var, forecast_date)` per
UTC day. The prototype evaluates both 1400Z and 2000Z snapshots
independently. **If both fire on the same `(city, var, D)`:
later snapshot wins.** This is pre-registered (NOT
"higher edge", which is endogenous to the calibrator being tested).
Reason: the mechanism's whole claim is that later snapshots are
sharper, so the cap tie-break should not preferentially keep the
weaker early signal.

**Diagnostic (reported, non-gating):** per `(city, var)` pair, report
the count and net P&L of 1400Z trades discarded by the tie-break.
If any cell shows discarded-1400Z P&L > kept-2000Z P&L by a wide
margin, that's evidence the tie-break direction may be wrong and
warrants a follow-up — but does NOT change the v2 gate verdict.

### 6.4 Trade simulation rules

- Source prices: snapshot's `yes_bid_dollars` / `yes_ask_dollars`.
- Fill price: integer-cent rounding (production trades discrete
  cents; non-integer fills inflate apparent edge ~0.5¢ on average).
  YES buys fill at `ceil(ask × 100) / 100`; NO buys fill at
  `ceil((1 - bid) × 100) / 100`.
- Edge: `model_prob - fill_price`, where `model_prob` is the
  posterior probability *after* applying today's calibrator (see §6.1).
- Trade if `edge ≥ edge_threshold` (from config) AND
  `calibrated_prob ≥ min_calibrated_prob` (from config).
- Position size: production Kelly (fractional) at fill price.
  Capped by `max_contracts_per_trade` AND depth haircut (§6.5).
- Settlement: high market `var=high`, strike `(floor, cap)` → win
  if `floor ≤ high_f ≤ cap` (between) or `high_f > floor` (greater)
  or `high_f < floor` (less). Low market mirrors.
- Fee: `python.analytics.fees.order_fee(taker=True)`.

### 6.5 Depth haircut — sensitivity grid

Snapshots have `volume_fp` (cumulative event volume) but not full
orderbook depth. Production has live `get_orderbook`. Prototype
runs G3 at **three haircuts and reports all three**:

| Haircut | `max_contracts` cap |
|---|---|
| `/5` (most conservative) | `min(volume_fp / 5, max_contracts_per_trade)` |
| **`/10` (primary)** | `min(volume_fp / 10, max_contracts_per_trade)` |
| `/20` (least conservative) | `min(volume_fp / 20, max_contracts_per_trade)` |

Primary G3 gate is on `/10`. Sensitivity table shipped in write-up.
If GREEN at `/10` but RED at `/5`, that's a depth-fragility flag —
report but do not auto-kill.

## 7. Evaluation procedure (locked)

```
Step 0: build climatology table from 2024-05-01 → 2026-03-17 ASOS corpus.
         Sanity gates per §4. Write data/research/asos_climatology.json.

Step 1: build eval corpus (one DataFrame).
   For each D in 48-day eval window:
     For each C in {NYC, CHI, LAX, MIA, AUS}:
       For each V in {high, low}:
         morning_8z ← weather_forecasts at (D, 0800Z)
         settled ← preferred-final settlement for (C, D)
         skip if morning_8z OR settled missing
         For each t in {1400Z, 2000Z}:
           snap_t ← snapshot at (D, t)
           asos_through_t ← ASOS for C, local-day-start through t
           skip if ASOS gap > 2h
           M ← max/min over asos_through_t.tmpf
           (μ_climo, σ_climo, n_eff) ← climatology[(C, month, hour_of_t)] with §4 fallback
           skip if climatology_unavailable
           For each market m in snap_t.markets where m.city=C, m.daily_variable matches V,
                                                  m.forecast_date == D (local):
             p_morning ← ensemble CDF on m's strike(s)
             p_intraday ← truncated-Gaussian posterior per §3.4
             record row (D, C, V, t, m.ticker, m.floor, m.cap, m.strike_type,
                         m.yes_bid, m.yes_ask, m.volume_fp,
                         p_morning, p_intraday, M, μ_climo, σ_climo, n_eff,
                         settled.high_f, settled.low_f, settled.is_preliminary)

Step 2: score G1 (CRPS).
   For each row at t=2000Z:
     crps_morning ← CRPS(morning ensemble CDF, settled value)
     crps_intraday ← CRPS(truncated-Gaussian posterior CDF, settled value)
     reduction ← crps_morning - crps_intraday
   aggregate: mean reduction + paired bootstrap 95% lower CI

Step 3: apply calibrator to {p_morning, p_intraday} → {p_morning_cal, p_intraday_cal}.
         Apply edge threshold, min_calibrated_prob, per-day cap (§6.3 tie-break).
         For each haircut in {/5, /10, /20}: run trade sim per §6.4.

Step 4: score G2 (count) and G3 (aggregate P&L + bootstrap CI), LAX-exc.

Step 5: write decision per §5 gates.
```

## 8. Outputs

| Artifact | Path |
|---|---|
| Climatology lookup | `data/research/asos_climatology.json` |
| Per-row evaluations | `data/research/asos_prototype_evaluations.parquet` |
| Notebook | `notebooks/asos_bayesian_prototype.ipynb` (unpublished) |
| Results write-up + decision | `docs/research/2026-05-21-asos-bayesian-prototype-results.md` |

Results write-up sections:
1. Empirical corpus inventory (rows pre/post each filter step).
2. G1 / G2 / G3 results with per-city breakdown (LAX informational).
3. Sensitivity: G3 at `/5`, `/10`, `/20` haircuts.
4. Diagnostics: climatology fallback rate, paired σ_diff for G1
   power, ASOS gap-drop rate, settlement-preliminary rate.
5. Decision: GREEN / KILL.
6. If GREEN: gaps the productionization spec must address (runtime
   ASOS client, `same_day_gate` carve-out, per-day cap expansion,
   later snapshot times for LAX, climatology refit cadence).
7. If KILL: short post-mortem identifying the binding gate.

## 9. Out of scope for prototype

- Runtime ASOS client (HTTP retry, caching, timeout, fail-open).
- `same_day_gate` carve-out around `BLOCK_PEAK_PASSED`.
- Per-day cap expansion to group same-day + next-day on the same
  observable.
- Hyperparameter tuning of edge threshold / Kelly fraction (use
  exact production values).
- Hyperparameter tuning of climatology pooling (use only the
  pre-registered fallback ladder in §4).
- Coverage of DEN / SEA (excluded for other reasons; ASOS not
  collected).
- Later snapshot times for LAX (would need 2300Z snapshots, which
  we don't have).
- Recovering historical calibrator artifacts (resolved by §6.1
  raw-ensemble baseline for G1; symmetric-today's-calibrator for G3).

## 10. Pre-resolved methodological questions

1. **ASOS timezone** — UTC (verified empirically 2026-05-21 against
   NYC diurnal cycle).
2. **Settlement source priority** — final preferred, preliminary
   accepted with flag.
3. **Historical calibrator artifact recovery** — sidestepped per §6.1.
4. **ASOS missing-hour handling** — drop snapshot if gap > 2h.
5. **Per-day cap tie-break** — later snapshot wins.
6. **Integer-cent fills** — apply via `ceil`.
7. **Gain-distribution shape** — left-truncated Gaussian.
8. **LAX 2000Z snapshot-time bias** — reported, non-gating.
9. **Power** — pre-computed in §5.2; flag if empirical σ_diff > 1.5°F.
10. **INCONCLUSIVE branch** — does not exist; binary GREEN / KILL.
