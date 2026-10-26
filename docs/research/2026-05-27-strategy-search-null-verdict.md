# Strategy Search — Null Verdict on Current Data

**Date:** 2026-05-27
**Author:** Claude Code (autonomous strategy-search session)
**Scope:** Find a profitable new weather-market trading strategy backed by existing data (snapshots, settlements, NBM, ASOS, shadow_settlements.full.json) under the user's bar: realized ROI with 95% day-clustered bootstrap CI lower bound > 0 vs current baseline.

> **2026-05-27 update:** Initial verdict (NULL) below is correct **on the current 339-trade live corpus**. After Stop-hook prompt that the user's data list explicitly includes GEFS-on-AWS back to 2017 + GHCN-Daily, the search pivoted to a multi-year-backfilled candidate. See the unpublished feature spec for the boundary-distance-aware per-cell calibrator candidate that uses that data path. The null verdict here is preserved as the documentation of *what failed on the existing corpus* — it remains true that no candidate clears the bar on existing data; the unlock is the GEFS+GHCN backfill, not cleverness on the current corpus.

## TL;DR

**Verdict: NULL on the existing 339-trade corpus.** Four candidate strategies were evaluated against the realized-PnL bar. None clears it. The two patterns that look promising in cross-section (low-ensemble-std cohort bleed, raw-prob "overconfidence" bleed) **reverse sign between April and May** — Simpson's reversal within the same corpus — and the post-isotonic_v2 era (since 2026-05-22) currently has **1 settled trade**, so any improvement claim is unfalsifiable today.

**The corpus is fundamentally underpowered.** 339 weather_edge settled trades cluster into **32 distinct trading days** with per-day σ(ROI) ≈ 58.8pp. To detect a +5pp ROI improvement at 80% power / 95% one-sided requires **~1,084 trading days — 34× the current corpus**. Even a +10pp improvement needs ~270 days (~8×). Any candidate without a ≥21pp point-estimate ROI lift cannot in principle clear the 95% CI bar on this corpus.

The honest action items are:
1. **Do not ship a new strategy or filter at this time.** The data does not support a confident "this beats baseline" claim.
2. **Prioritize the in-flight NBM-bracket-position veto** (currently shadow, gate at n≥80 — ETA late June) — it's the highest-confidence near-term P&L lever already scoped.
3. **Investigate the May bleed root cause** — May 2026 realized ROI is **-12.5% on n=225** (weather_edge, settled). This is the single most important diagnostic question that wasn't already on the in-flight list.
4. **Invest in settled-temp backfill** (GHCN-Daily, 666/666 NWS match per an unpublished corpus-sources note) to unblock the **boundary-distance feature** work — that's the largest documented unsolved P&L leak with a clear mechanism, but it is not backtestable on the current corpus because we lack settled temps for the trade window.

**One footnote candidate (84% bootstrap directional probability, does NOT clear the 95% bar):** A May-only "suppress NO-between trades where `raw_ensemble_prob (YES) < 0.10`" filter shows +12.4pp ROI delta, day-clustered CI95 [-11.9pp, +35.6pp]. This is the strongest forward-going candidate in the corpus and is cheap to shadow-validate, but it is one of ~18 informal cuts tested and does not clear the user's bar. See §3.3.

## 1. Goal and bar

User goal (verbatim): "Find a profitable weather-market trading strategy backed by data we already have or can ingest within reasonable storage (<5 GB new). Goal is realized PnL, not just calibration improvement."

User bar:
- Realized ROI beats current per-cell baseline with **95% bootstrap CI lower bound > 0**
- Min n: 100 trades total OR per-cell n ≥ 10 with bootstrap CI on Brier improvement and tail-aware LogSkill
- Fee-adjusted, taker assumptions explicit
- Reject-list provided (SPX, early-exit, between/fade/bracket-arb, excluded cells without structural fix, global calibrators, raising vol/OI filters)

## 2. Landscape survey

A separate Explore-agent report (see session transcript) covered the unpublished tracker, `docs/tech-debt.md`, and `docs/research/` to identify (a) what's actively being built, (b) what's been falsified, (c) underexploited corpora, (d) adjacent market categories. Key constraints carried into this analysis:

- **In flight, do not duplicate:** Favorite-bias pilot (live), Intraday EMOS HIGH (shadow), per-cell calibrator harness (Phase 1A.1 shipped), hierarchical pooling (Phase 2A, corpus-gated), NBM-bracket-position veto (shadow, n→80), LogSkill secondary promotion gate (shipped).
- **Killed, do not re-pitch:** SPX, all early-exit thresholds, monthly/seasonal markets, NBM gap-magnitude filter, ECMWF-GFS disagreement ratio at r<1.0, EMOS on σ-incompatible cells.
- **Parked-not-killed:** Intraday ASOS-Bayesian (G1 mechanism confirmed, G3 untradeable until calibrator redesign).
- **Underexploited corpora:** ASOS hourly (89k rows), NBM forecasts (211 files), NWS preliminary-vs-final, per-city calibration_monitor BSS.

## 3. Candidate strategies evaluated

### Candidate A — NBM-anchored greater/less primary signal

**Mechanism.** Use NBM (NOAA's NWS-official post-processed blend of GFS/ECMWF/HRRR/NAM/MOS) as the primary point forecast for `greater`/`less` weather markets, deriving `p_YES = 1 - Φ((strike − NBM) / σ_NBM)` where `σ_NBM` is fitted per (city, var, lead) from historical NBM error. Distinct from the in-flight NBM veto which only operates on `between` markets and only suppresses, not generates, trades.

**Result: unbacktestable on current data.** The bot trades almost exclusively `between` markets:

| strike_type | n   | win%  | ROI%    |
|-------------|----:|------:|--------:|
| between     | 328 | 64.0% | -4.6%   |
| less        | 8   | 37.5% | -99.8%  |
| greater     | 3   | 100%  | +112%   |

n=11 settled greater+less trades is far below the minimum for any per-cell validation. To backtest NBM-as-primary on greater/less would require replaying the strategy through historical markets with ground-truth settled temps for the trade window. Currently the `data/snapshots/<date>/settlements.json` corpus has only 62 (city, observation_date) pairs spanning 2026-03-15→2026-03-24, with **zero overlap with trade dates 2026-04-21→2026-05-27**. Settled-temp backfill (GHCN-Daily) is the prerequisite.

**Verdict:** Defer. Real mechanism, real data-source advantage, but not backtestable without settled-temp infrastructure.

### Candidate B — Low-ensemble-std overconfidence filter

**Mechanism.** Suppress trades where `ensemble_std_dev < 1.55°F`. Hypothesis: tight ensemble = model overconfident = realized worse than implied.

**May-only result (suggestive):**
| std cohort           | n   | win%   | ROI%    |
|----------------------|----:|-------:|--------:|
| low (<1.55°F)        | 45  | 55.6%  | -30.9%  |
| mid (1.55-2.55°F)    | 91  | 62.6%  | -5.5%   |
| high (≥2.55°F)       | 89  | 62.9%  | -12.6%  |

**Day-clustered bootstrap CI (May, B=2000):**
- low: ROI=-30.9% CI95=[-87.0%, +26.4%]
- mid: -5.5% [-32.2%, +32.0%]
- high: -12.6% [-56.5%, +26.5%]

**CIs span zero by wide margins.** The point estimate ordering is suggestive but n=45 day-clustered into 19 days is too small.

**Per-cell decomposition reveals the cohort is dominated by NYC/high (-93% ROI on n=13, low-std) and LAX/low (-25% on n=8, low-std), both small-n.** When you decompose the bleed by cell, it concentrates in 2 of 12 cells — not a global pattern.

**Verdict:** Not supported.

### Candidate C — Raw-probability overconfidence filter (low raw_p_YES, NO-side)

**Mechanism.** Suppress NO-between trades where `raw_ensemble_prob (YES) < 0.10`. Hypothesis: when the raw ensemble says YES is very unlikely and we bet NO confidently, we systematically over-trust the model.

**May-only (suggestive):**
| raw_p_YES cohort | n   | win%  | ROI%    |
|------------------|----:|------:|--------:|
| [0.00, 0.05)     | 27  | 66.7% | -25.2%  |
| [0.05, 0.10)     | 54  | 59.3% | -35.8%  |
| [0.10, 0.15)     | 68  | 69.1% | +23.5%  |
| [0.15, 0.20)     | 38  | 63.2% | -16.7%  |
| [0.20, 0.30)     | 20  | 65.0% | -0.2%   |

May appears clean: low-p bins bleed, mid-p bins profitable.

**FULL CORPUS reverses the sign:**
| cohort                       | n   | win%  | ROI%    |
|------------------------------|----:|------:|--------:|
| SUPPRESSED (raw_p_YES<0.10)  | 137 | 66.4% | **+21.5%**  |
| KEPT (raw_p_YES≥0.10)        | 169 | 65.1% | **-34.4%**  |
| BASELINE (all)               | 306 | 65.7% | -4.0%   |

Day-clustered bootstrap CI on PnL gain from suppressing low-p_YES trades:
- **Mean PnL gain: negative** (applying the filter *loses* money over the corpus)
- CI95 spans zero

April's [0.05, 0.10) bucket was a **+52% ROI winner**; May's was -36% ROI. **Same cohort definition, opposite sign within two months.**

**Per-cell decomposition of the proposed filter (full corpus):**

| cell      | n_all | n_drop | full_ROI | kept_ROI |
|-----------|------:|-----------------:|---------:|---------:|
| AUS/high  | 38    | 17 | +37.7%   | -13.4%   |
| CHI/high  | 33    | 15 | +41.0%   | -22.1%   |
| SEA/high  | 19    | 12 | +13.8%   | -66.6%   |
| CHI/low   | 27    | 16 | +6.9%    | -82.6%   |
| MIA/low   | 33    | 11 | -33.2%   | -53.0%   |
| NYC/low   | 42    | 18 | -76.0%   | -82.4%   |
| NYC/high  | 47    | 9 | +18.4%   | +50.0%   |

The filter helps NYC/* and MIA/low but devastates the high-side cells. **No global threshold works.**

**Verdict:** Strongly NOT supported. The Simpson's reversal between April and May is itself a finding: aggregate filters built on cross-month windows are unreliable in this regime.

### Candidate C.1 (post-review breakout) — raw_p_YES ∈ [0.05, 0.10) alone

**Backtesting-analyst review (2026-05-27) found that the aggregated raw_p_YES < 0.10 cohort hides one sub-bucket that superficially passes the user's 95% bar on day-clustered bootstrap but fails every robustness check.** Recording it here so future searchers don't rediscover it as a "win":

| metric                                       | value                       |
|----------------------------------------------|-----------------------------|
| n (full corpus, NO-between)                  | 95                          |
| ROI point estimate                           | +43.6%                      |
| Day-clustered bootstrap CI95                 | **[+16.3%, +71.8%]** (passes superficially) |
| Stationary-block bootstrap CI95 (L=7)        | **[−28.7%, +55.6%]** (spans zero) |
| Day-clustered CI95 with Bonferroni (18 cuts) | **[−6.9%, +88.0%]** (spans zero) |
| May-only ROI on same cohort                  | **−38.2%** (sign reversed)  |
| April winner concentration                   | AUS/high (+99% n=12), CHI/low (+92%), DEN/high (+98%), SEA/high (+99%) |

The April winners stack in cells that are now in `excluded_city_vars` (CHI/high, DEN/high, SEA/high) or under separate review (AUS/high). The lever is already pulled by the exclusion list; the [0.05, 0.10) finding is residual signal from cells that don't trade anymore. **Not a viable forward-going strategy.**

### Candidate D — Cross-city residual signal (Hypothesis C, deferred 2026-05-20)

**Mechanism.** Spatially-correlated GFS errors: regress `r_NYC,D` on `r_BOS,D−1 + r_PHIL,D−1 + r_CHI,D−1` (ridge + permutation test). Per `2026-05-20-weather-strategy-research.md` §4 the 67-day corpus is sparse and overfitting risk is high.

**Decision:** Not executed in this session. The estimated 8-10h build cost plus high overfitting risk on a 75-day corpus does not align with the time budget when (a) the simpler candidates already failed and (b) backtesting requires the same settled-temp infrastructure as Candidate A.

**Verdict:** Defer pending settled-temp backfill + permutation-test methodology setup.

## 4. Other findings (incidental)

### 4.1 May 2026 realized ROI is -12.5%

The current strategy bleeds modestly. **n=225 settled trades, -12.5% ROI net of fees**. The largest cell-level losses in May:

| cell      | n   | win%   | ROI%    |
|-----------|----:|-------:|--------:|
| LAX/high  | 15  | 60.0%  | -77.3%  |
| AUS/low   | 14  | 64.3%  | -48.8%  |
| LAX/low   | 9   | 33.3%  | -37.8%  |
| AUS/high  | 24  | 58.3%  | -32.6%  |
| NYC/low   | 34  | 55.9%  | -21.5%  |

Per-cell day-clustered bootstrap CIs (May):
- **Only LAX/high has CI95 upper bound < 0** (confidently losing): ROI=-77.3% CI=[-117%, -15%]
- All other cells span zero. LAX/high is already in `excluded_city_vars`; the losses are pre-exclusion residuals from before the rollback.
- **No cell has CI95 lower bound > 0** (no cell is confidently profitable in May at n levels available).

### 4.2 Post-2026-05-22 isotonic_v2 era: n=1 settled trade

The iso_v2 bounded-exposure pilot went live ~2026-05-22 20:23 UTC. As of corpus pull, **exactly one settled trade exists in the iso_v2 era**. Any claim about whether iso_v2 improved or worsened outcomes is unfalsifiable on this corpus — the pilot's own re-evaluation gate is ~2026-06-05 for the same reason.

### 4.3 Lead-time has no detectable pattern

`lead=0d (same-day)` n=33, ROI=-7.7%. `lead=1d` n=294, ROI=-0.4%. Same-day is a smaller cohort and slightly worse, but neither cohort is large enough to claim a regime effect.

### 4.4 Strike-type concentration

97% of trades are `between` markets, 96% are NO-side. The strategy's edge surface is overwhelmingly "bet temperature won't land in this 1°F window." Any new strategy must either expand this surface (greater/less, but n=11 settled today) or improve the between-NO trades (where Candidates B-C failed).

## 5. Methodology

All analyses on `data/shadow_settlements.full.json` (364 settled rows: 290 live + 74 shadow; 339 weather_edge strategy after filter). Joins to snapshot settlements failed due to a 62-vs-339 settlement-corpus gap (settlements.json collection was thin in the trade window).

**Bootstrap CI methodology:**
- Day-clustered resampling (n_days days resampled with replacement, all same-day trades carried together) — corrects for the within-day correlation that per-trade resampling ignores.
- B=2000 iterations, seed=42 for reproducibility.
- 95% percentile interval from the bootstrap distribution.
- Reference: matches what `scripts/research_2026_05_25_favorite_bias_backtest.py` uses (day-clustered) and what the 2026-05-25 favorite_bias note (unpublished) says is the team's standard.

**Methodology caveats (added 2026-05-27 post backtesting-analyst review):**
- Day-clustered bootstrap does not handle day-to-day serial correlation. The reviewer ran stationary-block bootstrap (L=7) as a robustness check on the one positive sub-cohort surfaced; results in §3.3.1. **Future searches should pair day-clustered with block(L=7) on any positive finding.**
- ~18 informal cuts were tested (6 raw_p bins × 3 month-cuts, plus low/mid/high std and per-cell). No multiple-comparisons correction was applied to the primary analyses. Bonferroni adjustment widens CIs ~1.4×; would not have changed any verdict in this report but is a load-bearing assumption that should be made explicit.

**Power analysis (added 2026-05-27 post-review):**

Per-day standard deviation of ROI in this corpus σ ≈ 58.8 percentage points. Number of trading days required to detect a true ROI improvement Δ at 80% power, 95% one-sided:

| improvement Δ | days needed | × current corpus (32 days) |
|---------------|------------:|---------------------------:|
| +5 pp         | ~1,084      | 34×                        |
| +10 pp        | ~270        | 8.5×                       |
| +20 pp        | ~68         | 2.1×                       |
| +30 pp        | ~30         | within corpus              |

**Implication: any forward-going candidate whose point-estimate ROI improvement is below +20pp cannot in principle clear the user's 95% bar on this corpus.** This explains structurally why all but one of the candidates failed (Candidate C.1 had point +43.6% but failed robustness). It also sharpens recommendation #4: settled-temp backfill is not just nice-to-have; it is a prerequisite for any future strategy-validation work with reasonable power.

**Limitations:**
- Corpus heavily weighted to April + early May (95% of trades pre-2026-05-21).
- No ground-truth settled temps for the trade window; couldn't run boundary-distance analyses against realized temps.
- Day-clustered bootstrap is conservative for daily filters but doesn't account for within-cell autocorrelation across days; block bootstrap was only run as a robustness check on the one positive sub-cohort, not as the primary CI method.

## 6. Recommendations

Ranked by ROI-per-effort:

1. **Do not ship a new filter or strategy.** No candidate clears the bar. Shipping anything on suggestive-only evidence risks the favorite-longshot Simpson's pattern: April's positive becomes May's negative. The strategy is near-breakeven; preserve that.

2. **Let the in-flight NBM veto reach n≥80** (~late June). It's the highest-confidence ship-able improvement already scoped, with proper shadow validation.

3. **Investigate the May -12.5% bleed root cause.** This is the single most important diagnostic question. Candidate explanations:
   - Bracket-window-fix (2026-05-21) over-expanded `between` raw probs → more trades on thinner edges
   - iso_v2 calibrator is producing different signals than iso_v1 — but n=1 settled, so unverifiable today
   - Regime: April warm forecasts vs May warming-trend forecasts hit different sides of brackets
   - Per-cell concentration: 5 cells contributed all the May losses (LAX/high already excluded; AUS/low, LAX/low, AUS/high, NYC/low partially trading)

4. **Invest in GHCN-Daily settled-temp backfill** (per an unpublished corpus-sources note — 666/666 NWS match validated). This unblocks:
   - Boundary-distance feature work (largest documented unsolved P&L leak)
   - NBM-as-primary backtest on greater/less
   - Cross-city residual permutation test
   - All future "is this strategy profitable" claims on multi-year data

5. **Per-cell rolling-window auto-exclusion (consider as risk-control framework, not a strategy).** Daily-computed bootstrap CI on rolling 30/60-day realized ROI per cell; auto-exclude when CI95 upper bound < 0 with n ≥ 10. This is what the in-flight per-cell calibrator harness does offline; a runtime version would be a thin wrapper. Not a "trading strategy" but a forward-going protection that scales with corpus growth. Not specified here — flagged for future scoping.

## 7. What would change the verdict

A new strategy could be specified if any of these conditions are met:

- **Settled-temp backfill ships:** unlocks Candidate A (NBM-direct greater/less) and Candidate D (cross-city residual) backtests on multi-year data.
- **Corpus grows ≥3×** (~600 weather_edge settled trades): the May-only sub-corpus becomes large enough to bootstrap-CI a filter without Simpson's reversal risk.
- **A novel data source ships:** ECMWF ensemble retrieved live (currently we only have GFS deterministic + GFS ensemble + ECMWF deterministic via OpenMeteo); NBM ensemble; HRRR; or a true intraday source like ASOS hourly with a calibrator that handles OOD inputs.

None of these can be conjured from current data + reasonable session budget.

## 8. Files referenced

- `data/shadow_settlements.full.json` — 364 settled rows
- `data/snapshots/<date>/<slot>/{markets,nbm_forecasts,weather_forecasts}.json` — 75 days × 4 slots × 3 files
- the unpublished tracker, `docs/tech-debt.md` (Explore agent landscape report)
- `docs/research/2026-05-20-weather-strategy-research.md` — prior strategy-search session
- `docs/research/2026-05-26-early-exit-losing-positions-research.md`; four other related notes are unpublished
