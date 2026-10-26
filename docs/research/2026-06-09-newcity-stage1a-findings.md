# Stage 1a — new-city ASOS backfill + draft climatology + lock-profile diagnostics

**Date:** 2026-06-09
**Status:** OFFLINE RESEARCH — no deploy, no production-artifact changes. All
new artifacts live under `data/research/`. Nothing in `python/` or
`python/config.yaml` was touched; the production `data/asos/climatology.json`
was not modified.

**Scripts (committed):**
- `scripts/research_2026_06_09_backfill_newcity_asos.py` — 2-year IEM hourly backfill
- `scripts/research_2026_06_09_build_newcity_climatology.py` — DRAFT climatology (reuses production math)
- `scripts/research_2026_06_09_newcity_lockprofiles.py` — lock profiles + σ_gain + mature Class-A-NO diagnostic

**Artifacts (committed, both < 1 MB):**
- `data/research/newcity_climatology.json` (921 KB) — DRAFT, NOT production
- `data/research/newcity_lockprofiles.json` (690 KB)

**Raw ASOS CSVs** (`data/research/asos/k{city}_asos.csv`) are NOT committed
(`*.csv` is gitignored globally, like the 5 mature-city CSVs). They are
reproducible by re-running the backfill script.

---

## 1. Station-ID verification (gate — PASSED for all 11)

Each ICAO was probed live against IEM before any bulk pull (1-day window,
paced). Every station returned real hourly `tmpf`, and IEM's returned
`station` column equals the NWS CLI settlement code in
`python/data/nws.py:CITY_TO_CLI_LOCATION` (the backfill script also asserts
this per-response and refuses to write a mismatch). The 2-year backfill then
pulled the full window cleanly:

| city | ICAO req | CSV station (=CLI) | rows | first valid (UTC) | last valid (UTC) |
|------|----------|--------------------|-----:|-------------------|------------------|
| ATL  | KATL | ATL | 21,484 | 2024-05-01 00:52 | 2026-06-08 23:52 |
| DAL  | KDFW | DFW | 21,004 | 2024-05-01 00:53 | 2026-06-08 23:53 |
| DC   | KDCA | DCA | 22,365 | 2024-05-01 00:52 | 2026-06-08 23:52 |
| HOU  | KHOU | HOU | 21,990 | 2024-05-01 00:53 | 2026-06-08 23:53 |
| LV   | KLAS | LAS | 19,173 | 2024-05-01 00:56 | 2026-06-08 23:56 |
| MIN  | KMSP | MSP | 22,452 | 2024-05-01 00:16 | 2026-06-08 23:53 |
| NOLA | KMSY | MSY | 21,883 | 2024-05-01 00:53 | 2026-06-08 23:53 |
| OKC  | KOKC | OKC | 21,271 | 2024-05-01 00:52 | 2026-06-08 23:52 |
| PHX  | KPHX | PHX | 19,043 | 2024-05-01 00:51 | 2026-06-08 23:51 |
| SATX | KSAT | SAT | 21,922 | 2024-05-01 00:51 | 2026-06-08 23:51 |
| SFO  | KSFO | SFO | 20,889 | 2024-05-01 00:56 | 2026-06-08 23:56 |

**Total: 233,476 rows.** No anomalies: every station's date span is the full
2024-05-01 → 2026-06-08 window, row counts are uniform (LV/PHX slightly lower
≈ desert stations report fewer SPECI/cloud-driven extra obs — expected, not a
data hole). No wrong-location returns, no missing stations. **Gate PASSED.**

DST note: the backfill is in UTC; the LST climate-day framing is applied
downstream (climatology + lock profiles), matching the production frame.

LST standard offsets (no DST, verified against each city's timezone and the
`CITY_TZ` convention in `python/data/asos.py`): ATL/DC −5 (EST); DAL/HOU/MIN/
NOLA/OKC/SATX −6 (CST); **PHX −7 (MST — Arizona does not observe DST)**; LV/SFO
−8 (PST). PHX is the one to watch: it is the only −7 city and the only one
whose standard offset equals its civil offset year-round.

---

## 2. DRAFT remaining-gain climatology coverage

Built by importing the production builder's pure functions
(`compute_gains`, `aggregate_to_cells`, `build_hour_pool`, `apply_sanity_gates`,
`load_asos_rows`) verbatim — same LST frame, same hourly-frame M (train/serve
parity; never fit on 1-minute data), same FIT window
(2024-05-01 → 2026-04-28 UTC exclusive), same per-cell/pool/sanity gates. The
only differences are the city set, their LST offsets, the new-city station
guard, and the draft output path. Every city passed the station-identity guard.

| city | HIGH filled % | HIGH primary cells | LOW filled % |
|------|--------------:|-------------------:|-------------:|
| ATL  | 64.6 | 185 | 95.8 |
| DAL  | 62.8 | 180 | 95.8 |
| DC   | 75.0 | 197 | 95.8 |
| HOU  | 66.7 | 185 | 95.8 |
| LV   | 63.5 | 179 | 92.4 |
| MIN  | 85.1 | 205 | 95.8 |
| NOLA | 69.4 | 180 | 95.8 |
| OKC  | 69.4 | 189 | 95.8 |
| PHX  | 63.9 | 184 | 92.7 |
| SATX | 67.0 | 193 | 95.8 |
| SFO  | 65.3 | 180 | 88.9 |

The HIGH "filled %" of 63–85 % is **not** a data-quality problem: the
unavailable cells are overnight UTC hours where remaining-gain-to-HIGH is
structurally ≈0 with σ below the 0.5 °F sanity floor — exactly as in the
production 5-city build, and those hours are not HIGH-trading hours anyway.
Daytime HIGH coverage (the cells the strategy would actually use) is dense
(179–205 primary cells per city). LOW coverage is strong everywhere (89–96 %).
**No city is data-starved for the in-scope HIGH variable.**

The DRAFT artifact is in-scope only as a Stage-1 measurement input
(centering test §1b, σ-gate decision); it is **not** wired into any runtime and
must be re-built by the production builder (with the production city set) before
any Stage-2 cell-add — per the spec's hard ordering "config lands only after the
production climatology rebuild ships."

---

## 3. Per-city HIGH lock profiles

`newcity_lockprofiles.json` carries, per `(city, month, LST hour)`:
`mu_gain`, `sigma_gain`, `n`, and `P(final_high − M_t ≤ k)` for k ∈ {0,1,2}.
These come from the 2-year ASOS self-join (M-so-far vs the day's own later
obs) — they need no CLI settlement, so the full 2-year history powers them.

Representative slice (July, 14:00 LST — early-afternoon, the live-decision
window):

| city | n | P(lock ≤ 1 °F) | σ_gain |
|------|--:|---------------:|-------:|
| SFO  | 62 | 1.00 | 0.13 |
| NOLA | 60 | 1.00 | 0.30 |
| LV   | 62 | 0.95 | 0.57 |
| DC   | 62 | 0.94 | 0.65 |
| MIN  | 62 | 0.90 | 0.72 |
| DAL  | 62 | 0.89 | 0.77 |
| ATL  | 62 | 0.89 | 0.78 |
| HOU  | 62 | 0.87 | 0.79 |
| SATX | 62 | 0.87 | 0.93 |
| OKC  | 62 | 0.87 | 0.95 |
| PHX  | 62 | **0.71** | 0.97 |

The lock sharpens through the afternoon: by ~15–17 LST almost every city is
fully locked (σ_gain → 0, P(lock ≤ 1) → 1.0). **PHX is the structural outlier**
— at 14 LST in July only 71 % of days are within 1 °F of the eventual high
(monsoon convection keeps the afternoon high live later than elsewhere),
matching the spec's PHX-monsoon risk row. SFO and NOLA lock earliest (marine /
humid maritime climates with low afternoon volatility).

---

## 4. σ_gain vs `SHARP_SIGMA_GAIN_MAX = 1.2` (per-city recommendation)

The shadow report's sharp band is `σ_gain ≤ 1.2` (5-city calibrated). Computed
on the **unlocked** near-peak cells only (LST 11–15, σ_gain > 0.2 °F — the cells
where a sharp/wide call is genuinely live; later hours are trivially "all
sharp" because the high is already locked, which would make the verdict
meaningless). **The constant was NOT changed** — this is a recommendation for
the later V2→live gate.

| city | median unlocked σ_gain (11–15 LST) | verdict for 1.2 |
|------|-----------------------------------:|-----------------|
| HOU  | 1.01 | APPROPRIATE |
| NOLA | 0.96 | APPROPRIATE |
| LV   | 1.18 | APPROPRIATE |
| ATL  | 1.19 | APPROPRIATE |
| SATX | 1.27 | TOO TIGHT |
| DAL  | 1.29 | TOO TIGHT |
| DC   | 1.32 | TOO TIGHT |
| PHX  | 1.39 | TOO TIGHT |
| OKC  | 1.39 | TOO TIGHT |
| MIN  | 1.49 | TOO TIGHT |
| SFO  | 1.49 | TOO TIGHT |

"TOO TIGHT" means the median live-decision cell has σ_gain ≥ 1.2, so a
1.2 sharp gate would **reject most of the cells** at those cities during the
late-morning/early-afternoon window — not that the gate is wrong, but that at
those cities the *sharp* trade only exists later in the day (closer to peak),
which the post-extremum-band restriction (§6) already enforces.

**Recommendation for V2:** keep `SHARP_SIGMA_GAIN_MAX = 1.2` as the global
default. Do not loosen it. For the continental/high-variance cities (DAL, DC,
MIN, OKC, PHX, SATX, SFO) the implication is that any HIGH trade must wait
until M is close enough to peak that the cell's σ_gain has fallen under 1.2 —
i.e. the σ-gate naturally pushes these cities into the post-peak band rather
than needing a per-city threshold. SFO's σ=1.49 (marine-layer burn-off timing
variance) and PHX's σ=1.39 (monsoon) are the two to re-examine with forward
data before promoting either cell. The coastal/humid cities (HOU, NOLA, LV,
ATL) sit naturally in-band and are the cleanest σ-gate candidates.

---

## 5. New-city CLI settled-day count (forward-join readiness)

New-city `settlements.json` only began accruing ~2026-06-09, so the
forward settled-join set is, as the spec anticipated, **tiny**:

| city | distinct CLI final-HIGH days so far |
|------|-----:|
| ATL  | 2 |
| DAL  | 3 |
| DC   | 3 |
| HOU  | 3 |
| LV   | 3 |
| MIN  | 3 |
| NOLA | 3 |
| OKC  | 3 |
| PHX  | 3 |
| SATX | 2 |
| SFO  | 3 |

≈ 2–3 days each (≈ 31 total). This is far below any of the §1c power floors
(≥20 days / ≥40 trades **per class**), confirming the spec's
UNDERPOWERED-NO-DATA expectation for the immediate window. The lock-profile and
σ_gain work above does NOT depend on these — it uses the 2-year ASOS self-join.
The CLI count is the clock for the eventual centering test (§1b) and V1 premium
measurement (§1c); it will not be measurable until well into the
accrual window (deadline 2026-07-21).

---

## 6. Mature-corpus Class-A NO realized-vs-predicted, by slot / LST hour

On the 5 mature cities, for every same-LST-day HIGH market we formed the
observation-anchored posterior (production climatology + hourly M at slot time,
truncated-t ν=5) and isolated confident-NO candidates (posterior ≤ 0.03 —
betting the HIGH stays below the bracket). We then compared the **predicted**
NO-win rate (1 − posterior) to the **realized** NO-win rate from NWS CLI
finals. n = 1,758 Class-A-NO candidates.

**By snapshot slot:**

| slot (UTC) | n | mean predicted NO-win | realized NO-win | gap (pred − real) |
|------------|--:|----------------------:|----------------:|------------------:|
| 0200 | 1 | 0.980 | 1.000 | −0.020 |
| 0800 | 176 | 0.985 | **0.864** | **+0.121** |
| 1400 | 558 | 0.991 | 0.955 | +0.036 |
| 2000 | 1023 | 0.997 | 0.988 | +0.009 |

**By LST hour (the mechanism, sharply):**

| LST hour | n | predicted NO-win | realized NO-win | gap |
|---------:|--:|-----------------:|----------------:|----:|
| 2 | 9 | 0.985 | 0.778 | **+0.207** |
| 3 | 44 | 0.984 | 0.864 | +0.120 |
| 4 | 65 | 0.987 | 0.862 | +0.125 |
| 5 | 27 | 0.980 | 0.889 | +0.091 |
| 9 | 106 | 0.989 | 0.915 | +0.074 |
| 11 | 104 | 0.994 | 0.981 | +0.013 |
| 13 | 79 | 0.999 | 1.000 | −0.001 |
| 14 | 388 | 0.997 | 0.982 | +0.015 |
| 15 | 279 | 0.997 | 0.993 | +0.005 |
| 16 | 66 | 0.999 | 1.000 | −0.001 |
| 17 | 12 | 0.999 | 1.000 | −0.001 |

**Finding:** confident-NO is materially over-confident *before the peak* and
well-calibrated *after* it. In the pre-dawn / morning hours (LST 2–5) the
posterior says NO wins ~98 % of the time but it actually wins only 78–89 % — a
**+9 to +21 pp** miscalibration — because the hourly-METAR running max
understates the eventual CLI high (memo finding #2, the ~1 °F undercount), so a
later-day gain crosses the bracket the posterior had ruled out. By LST 13–17
the high is essentially locked and the gap collapses to ±0.5 pp. The 0800Z slot
is the single worst offender (+12.1 pp over 176 trades).

**Implication for the spec's Stage-1 rule:** this directly validates "Class-A
NO restricted to the post-extremum band." A confident-NO bet placed before the
day's high is in (roughly before ~LST 11–13, depending on season) is betting
against an understated M and loses far more often than the posterior claims.
The diagnostic gives a concrete cut point: restrict Class-A NO to ~LST ≥ 13 (or
equivalently the post-extremum band where the running max has plateaued), where
the realized-vs-predicted gap is within noise. Pre-peak Class-A NO must not be
trusted as a probe on the new cities either.

---

## 7. Stage-2 structural-exclusion flags

No new city is a hard data-quality exclusion — all 11 verified clean, with full
2-year history and dense daytime HIGH coverage. Two cities carry a **caution
flag** for the eventual V2→live gate (not a Stage-1a exclusion):

- **PHX** — latest/most uncertain afternoon peak (July 14 LST P(lock ≤ 1) = 0.71
  vs 0.87–1.00 elsewhere; σ_gain 1.39). Monsoon convection (≈ July onward) keeps
  the high live later; the spec already flags this. Its (city, month, hour)
  climatology captures the seasonality, but PHX should be the last cell promoted
  and only on a clean *forward* per-bin table that spans the monsoon.
- **SFO** — highest near-peak σ_gain (1.49) despite locking *earliest* on most
  days. The bimodality (marine-layer burn-off either happens early → locked, or
  stalls → big late gain) means the *aggregate* σ is wide even though the median
  day is sharp. The σ-gate will handle this correctly (it sees the per-cell σ),
  but SFO's posterior should be watched for tail miscalibration before promotion.

Everything else (ATL, DAL, DC, HOU, LV, MIN, NOLA, OKC, SATX) is structurally
fine for Stage-2 consideration, gated as always on the Stage-1 centering test
(§1b) and V1 premium measurement (§1c) returning GREEN — neither of which is
measurable yet given the ~2–3 day CLI settled count (§5).

---

## Decisions made that were not specified

- **Backfill window end = 2026-06-09** (not 2026-04-28): I pulled through the
  present so the same CSVs also back forward/eval analyses; the DRAFT climatology
  builder still applies the production FIT window (→ 2026-04-28) internally, so
  the fitted cells are train/serve-parity-identical to what production would
  produce. The extra ~6 weeks of rows are inert for the fit.
- **Near-peak band for the σ-vs-1.2 verdict = LST 11–15** (not a fixed
  "afternoon" guess). An initial 18–22 LST band gave a useless "σ=0, too loose"
  verdict everywhere because the high is already locked by then; 11–15 LST is the
  genuine live-decision window (verified against the per-hour σ profiles). The
  verdict is also computed on *unlocked* cells (σ > 0.2 °F) so a stray
  already-peaked winter morning doesn't bias the median to 0.
- **Mature Class-A NO posterior keyed by run-UTC-hour climatology cell**, matching
  how the production `ClimatologyTable.lookup` is keyed (`(city, month, hour_utc)`),
  so the diagnostic reflects the real runtime lookup rather than an idealized one.
- Copied the 5 mature-city CSVs into the worktree (gitignored, not committed) so
  the mature Class-A-NO diagnostic could run; they are identical to the
  shared-checkout copies.
