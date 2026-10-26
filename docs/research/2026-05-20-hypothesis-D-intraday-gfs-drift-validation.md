---
title: Hypothesis D — intraday GFS run-to-run drift validation
date: 2026-05-20
parent_workstream: docs/research/2026-05-20-weather-strategy-research.md
type: PRE-REGISTERED PRIMARY VALIDATION (decision rules committed before any data is examined)
owner: Claude Code (operator-directed)
status: INCONCLUSIVE — 2026-05-20. Corpus 14 rows / 2 dates after strict slot pinning; Q4 row-bracket pairs = 0 (need ≥ 200). Re-run after the slot-drift fix backfills ≥ 30 dates with the canonical {0500, 1100, 1600, 2100} slot set.
---

# Hypothesis D — intraday GFS run-to-run drift validation

## Why this exists

the unpublished tracker "Hypothesis D — intraday GFS run-to-run drift" + parent
research doc §4. The edge mechanism: GFS runs every 6h with DA assimilation;
consecutive runs partially correct prior-run state errors. If the latest-slot
ensemble is more accurate than the morning-slot ensemble (in Brier-score terms,
on the between-strike brackets we actually trade), then a same-day "use the
freshest slot" path has measurable value over today's pin-to-morning behaviour.

The kill-test on 2026-05-20 confirmed Open-Meteo IS refreshing GFS between
snapshot slots — only 12% of (date, city, var, forecast_date) keys had all 4
daily slots identical; 46% had all 4 distinct. The hypothesis is alive and now
needs a head-to-head Brier test. Do not redo the kill-test.

This is a cheap diagnostic (operator estimate: ~4 hours) that gates the
intraday-strategy three-way prioritization (Hypothesis D vs. ASOS-Bayesian vs.
Intraday EMOS HIGH). A killed verdict is itself useful — it focuses the
intraday lane on observations (ASOS) rather than model refresh.

## Honest disclosure of peek bias

This is a **primary validation**, not a secondary analysis. The pre-registered
rules below are committed before any Brier number is computed. The author has
not run the analysis script at the time the frontmatter is committed.

The author HAS examined (and these inputs are intentionally pre-committed):

- The kill-test result (12% all-slots-identical, 46% all-distinct, summarised
  in the unpublished tracker entry).
- The structure of `data/snapshots/<date>/<slot>/weather_forecasts.json` and
  `data/snapshots/<date>/settlements.json` (schema, not contents).
- The list of production-active cells (14 minus 4 excluded — see §"Corpus").

The author has NOT examined the per-cell Brier numbers, the per-cell drift
distributions, or any quartile-binned outcome. The rules below name an
absolute Brier-reduction threshold and a CI clause; they are not tuned to a
specific data realization.

## Pre-registered rules (committed BEFORE the script runs)

### Corpus

A (date, city, var, forecast_date) cell enters the corpus iff ALL of the
following hold:

1. All 4 daily slots (`0500`, `1100`, `1600`, `2100`) have populated
   `gfs_seamless` ensembles in `weather_forecasts.json`, AND
2. All 4 daily slots have populated `ecmwf_ifs025` ensembles in
   `weather_forecasts.json` (needed for the production-mirroring blend), AND
3. A settled outcome exists in `data/snapshots/<date>/settlements.json`
   matching `city` and the daily variable (`high_f` for var=high,
   `low_f` for var=low). Prelim/final dedup follows the
   `prefer_latest_settlement` pattern in `scripts/build_emos_corpus.py`:
   prefer `is_preliminary=false` records over `is_preliminary=true` records
   for the same (city, observation_date), AND
4. The cell is one of the 10 production-active (city, var) combinations:
   `{NYC,CHI,MIA,LAX,SEA,AUS} × {high,low}` minus
   `{DEN_low, DEN_high, SEA_high, LAX_high}` (per
   `python/config.yaml:weather.excluded_city_vars` 2026-05-20).

The `forecast_date` is the date the temperature observation occurs, which
matches `settlements.json:observation_date`. The snapshot date can lag the
forecast_date by 0-3 days (same-day brackets exist; brackets that re-open up
to ~3 days ahead also exist).

### Drift metric

For each (date, city, var, forecast_date) cell:

```
blend_mean(slot) = 0.55 * mean(gfs_seamless ensemble members at slot)
                 + 0.45 * mean(ecmwf_ifs025 ensemble members at slot)
drift_F = blend_mean(latest_slot=2100) - blend_mean(earliest_slot=0500)
```

The 0.55/0.45 weights are the production weights from the v6 envelope at
`data/research/workstream1_gate_audit_v6_findings.json:envelope`. Drift is
signed (sign retained); the buckets below operate on `|drift_F|`.

### Buckets

For each (city, var) cell separately, compute the `|drift_F|` quartile
boundaries on the cell's settled rows. Each (date, city, var) row is then
assigned to Q1 (lowest `|drift_F|`) through Q4 (highest `|drift_F|`) **within
its own cell's distribution**. This is the same per-cell-normalization
convention used by the 2026-05-17 σ-ratio rerun (unpublished).

The headline test pools Q4 rows across all 10 cells and compares to pooled
Q1 rows. Per-cell Q4 means different absolute `|drift_F|` cutoffs per cell;
this is intentional.

### Headline test (Brier reduction)

For each (date, city, var) row in the corpus, evaluate Brier on every
between-strike bracket open on the snapshot day that targets this
`(city, var, forecast_date)`:

```
P_in_bracket(slot) = (#ens_members with floor ≤ value ≤ cap) / n_members
realized = 1 if floor ≤ settled_value ≤ cap else 0
brier(slot, bracket) = (P_in_bracket(slot) - realized)^2
```

`value` is computed per-ensemble-member from the blend
`0.55 * gfs_member + 0.45 * ecmwf_member`. If the two ensembles have unequal
member counts, use the smaller of the two (paired index 0..min_n-1). This
matches the `fraction_between` convention in `python/data/weather.py`.

For each (date, city, var) cell, aggregate Brier across its open brackets via
the mean: `cell_brier(slot) = mean over brackets of brier(slot, bracket)`.
Cells with zero open between-strike brackets on the snapshot day are dropped
from the headline test (they contribute no Brier evidence).

The pooled top-quartile Brier reduction is:

```
Δ = mean(cell_brier(0500) for Q4 rows) - mean(cell_brier(2100) for Q4 rows)
```

A positive Δ means latest-slot Brier is LOWER than earliest-slot Brier
(better-calibrated latest-slot forecasts → smaller squared error → smaller
Brier).

### Bootstrap CI

`bootstrap_95_CI(Δ)` is computed by resampling Q4 rows with replacement,
B=2000, seed=20260514 (matches prior Oracle work). The 95% CI is the [2.5,
97.5] percentile of the bootstrapped Δ distribution.

### Verdict rules

Hypothesis D is **SUPPORTED** iff ALL of the following hold:

1. `Δ_top_quartile ≥ 0.02` absolute (pooled across cells).
2. `bootstrap_95_CI_lo(Δ_top_quartile) > 0` (CI excludes zero from below).
3. **Sign holds for ≥ 70% of cells** with `n_top_quartile ≥ 15`. A cell's
   sign is positive iff its own per-cell `cell_Δ = mean(cell_brier(0500) for
   Q4_in_this_cell) - mean(cell_brier(2100) for Q4_in_this_cell) > 0`. Cells
   with `n_top_quartile < 15` are excluded from the percentage; they
   contribute no per-cell sign evidence.

Hypothesis D is **NOT-SUPPORTED** if any of (1), (2), (3) fail.

Hypothesis D is **INCONCLUSIVE** if the corpus contains fewer than 200
between-strike row-bracket pairs in Q4 (insufficient power for the headline
test), OR fewer than 5 cells have `n_top_quartile ≥ 15` (insufficient
cell-level power for clause 3). The inconclusive verdict triggers
"re-run on extended corpus" rather than "kill the hypothesis".

### Pinned parameters (operator-set, locked at frontmatter commit time)

| Knob | Value | Rationale |
|---|---|---|
| Earliest slot | `0500` | Morning snapshot — what the strategy currently reads at scan-time before peak |
| Latest slot | `2100` | Last snapshot before midnight UTC; closest to settlement |
| Blend weights | `(0.55, 0.45)` for (GFS_seamless, ECMWF_ifs025) | Production v6 envelope |
| Bracket type | `between` only | Matches the bulk of weather-edge traffic; greater/less excluded for scope |
| Snapshot date alignment | `forecast_date - snapshot_date ∈ {0,1,2,3}` | All open between-strike brackets at that snapshot for this `forecast_date` |
| Prelim/final dedup | Prefer `is_preliminary=false` | Mirrors `scripts/build_emos_corpus.py:prefer_latest_settlement` |
| Bootstrap B | 2000 | Oracle convention |
| Bootstrap seed | 20260514 | Oracle convention |
| Brier threshold | 0.02 absolute | Operator spec |
| CI lower-bound rule | `> 0` | Operator spec |
| Per-cell sign-hold | `≥ 70%` of cells with n ≥ 15 in Q4 | Operator spec |

### What this validation explicitly does NOT do

- Does not look at greater/less strike types. Scope limited to between.
- Does not vary slot pairs (no 1100-vs-2100, no 1600-vs-2100). The signed
  drift over the full day (0500 → 2100) is the headline; intermediate
  comparisons are deferred.
- Does not look at the directional drift sign (warming-day vs cooling-day).
  Buckets are `|drift_F|` absolute; sign is preserved in the row records for
  future use but not used in the verdict rules.
- Does not measure economic edge / fee-net P&L. The test is Brier-only.
  Operator recommendation in §Results addresses whether the Brier improvement
  is large enough to translate to trading edge.
- Does not interact with the live calibrator (isotonic vs EMOS). Brier is
  computed on raw ensemble fraction `P_in_bracket`, NOT on the live
  calibrator's output. This isolates the "freshness" question from
  calibration-quality.

### Outputs (when results are written)

- Per-cell summary table (10 rows: city/var, n_total, n_Q4, cell_Δ_Q4, sign).
- Pooled Q4 verdict: Δ, bootstrap 95% CI, n_pooled.
- Verdict block: SUPPORTED / NOT-SUPPORTED / INCONCLUSIVE.
- Operator recommendation (concise, ≤200 words).

## Results

Run date: 2026-05-20. Analysis script: tempfile `/tmp/h_d_drift.py` (not
committed, per operator constraint). Markets-source slot:
`data/snapshots/<date>/2100/markets.json` (the latest pre-registered slot —
contains the most complete set of open between-strike brackets for the day).

### Corpus-blocking finding (must read before the numbers below)

Only **2 snapshot dates** in `data/snapshots/` carry the exact pre-registered
slot set `{0500, 1100, 1600, 2100}`: **2026-05-11** and **2026-05-19**. Every
other date drifts to slot names like `{0400, 0900, 1500, 2000}` or `{0500,
1000, 1600, 2000}` — the GitHub Actions queue delay documented in the snapshot-slot drift note
(unpublished).

The pre-reg explicitly pinned `0500` and `2100` in the §"Pinned parameters"
table and required "All 4 daily slots (`0500`, `1100`, `1600`, `2100`)" in
the §Corpus clause 1. Relaxing to "closest-to-canonical-slot" would be a
post-hoc rule change and is not permitted. The strict reading is honoured.

Note: the kill-test cited in the doc body (1,880 keys with ≥ 2 slots present)
used a softer slot-presence rule — it asked "is GFS refreshing between
snapshots?", not "does the 0500-vs-2100 pair exist?". Both readings are
self-consistent given slot-drift; this validation simply cannot answer its
question on the strict-pinning corpus.

### Corpus summary

| Metric | Value |
|---|---|
| Total settled records loaded (final-only) | 445 |
| Snapshot dates with all 4 required slots | 2 (`2026-05-11`, `2026-05-19`) |
| In-corpus rows (10 cells × candidates × dates, all 4 slots populated, settled) | 14 |
| Rows with ≥ 1 open between-strike bracket on the snapshot's `2100/markets.json` | 3 |
| Q4 between-strike row-bracket pairs (pooled across cells) | **0** |
| Cells with `n_Q4 ≥ 15` | **0** |

Most of the 14 corpus rows lost their bracket-source either because the
forecast_date was already past settlement window relative to the snapshot
date's `2100` market list, or because the bracket existed only as a
greater/less strike and was excluded by the between-only scope of the
pre-reg.

### Per-cell table

`cell_Δ_Q4` and sign are undefined when `n_Q4 = 0`.

| cell | n_total | n_Q4 | cell_Δ_Q4 | sign |
|---|---:|---:|---:|:---:|
| NYC/high | 1 | 0 | n/a | n/a |
| NYC/low  | 1 | 0 | n/a | n/a |
| CHI/high | 2 | 0 | n/a | n/a |
| CHI/low  | 2 | 0 | n/a | n/a |
| MIA/high | 1 | 0 | n/a | n/a |
| MIA/low  | 1 | 0 | n/a | n/a |
| LAX/low  | 2 | 0 | n/a | n/a |
| SEA/low  | 2 | 0 | n/a | n/a |
| AUS/high | 1 | 0 | n/a | n/a |
| AUS/low  | 1 | 0 | n/a | n/a |

### Pooled headline

- `Δ_top_quartile` = **undefined** (n_pooled_Q4_rows = 0)
- bootstrap 95% CI = **undefined** (B=2000 not run; no rows to resample)
- n_pooled (Q4 rows with brackets) = **0**

### Verdict

**INCONCLUSIVE.**

Pre-reg §Verdict rules: "INCONCLUSIVE if the corpus contains fewer than 200
between-strike row-bracket pairs in Q4 ... OR fewer than 5 cells have
`n_top_quartile ≥ 15`." Both inconclusiveness clauses fire (0 < 200, 0 < 5);
clause 1 alone is sufficient.

The Brier-reduction threshold (0.02), the bootstrap-CI clause, and the 70%
per-cell sign-hold clause are not tested — they have no rows to operate on.
**The hypothesis is neither SUPPORTED nor NOT-SUPPORTED on this corpus.**

## Operator recommendation

The H-D validation requires a corpus extension before it can produce an
in-or-out verdict. Two paths:

1. **Fix the snapshot-slot drift upstream** (tech-debt §"snapshot-slot
   drift"). Pin the GitHub Actions workflow's effective snapshot slot to
   `{0500, 1100, 1600, 2100}` independent of queue delay — e.g., bucket
   each run to the nearest canonical slot, or rename slot dirs at write
   time. Once that ships, the corpus accumulates one strict-pinning row
   per cell per day; the hypothesis can be re-run with ~30+ in-corpus
   rows per cell after roughly six weeks.

2. **Re-run on or after 2026-07-01** assuming the slot-drift fix lands
   in early June. Earlier re-runs will hit the same Q4 < 200 wall.

In the meantime, keep H-D parked and do **not** reassign the intraday
research bandwidth to H-D's mechanism (freshest-GFS slot). The kill-test
verifies the mechanism exists (Open-Meteo IS refreshing GFS between
slots) but this validation cannot quantify whether that refresh yields a
≥ 0.02 Brier reduction at top-quartile drift. The other two intraday
candidates listed in the unpublished tracker are unchanged in priority.

Specifically: do not change the live calibrator, the `same_day_gate`, or
the slot the strategy reads at scan time. Treat this run as a documented
"awaiting corpus" parking event, not a kill.

When re-running, the implementation script (`/tmp/h_d_drift.py`) is
disposable; the pre-registered rules and the corpus-blocking finding
above are the durable artefacts. Decide the re-run trigger
before scheduling — fold-in versus standalone is undecided.

