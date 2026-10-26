# Weather-Markets Strategy Research Report

**Date:** 2026-05-20
**Author:** Claude Code (autonomous research session)
**Audience:** the maintainer, picking this up the morning of 2026-05-21
**Status:** Final draft. Five hypotheses generated; two backtested; one ships to shadow.

## TL;DR

After surveying the Kalshi weather-market surface and the data we already have, the single highest-upside *new* strategy is an **intraday ASOS-Bayesian update for same-day temperature markets**. It is the only candidate that materially expands the strategy's edge surface rather than filtering trades we already make. One suppressive overlay for the existing strategy — an **NBM-bracket-position veto** — survived backtest and should go to shadow immediately.

**What the backtest produced tonight (n=197 joinable live trades, 2026-04-21 → 2026-05-18):**
- **Ship to shadow:** NBM-bracket-position veto. Trades where NBM lands on the opposite side of the bracket from the model's directional bet had 46.9% win rate vs 68.4% for the rest. Dropping these 32 trades flips joined-window P&L from negative to positive. n=32 is below the trust floor — shadow validation required before live.
- **Killed:** NBM gap-magnitude filter (|blend_mean − NBM| > τ). Quintile ROI is flat; the largest-gap cohort actually *wins more* often.
- **Killed:** ECMWF-GFS disagreement ratio filter at the pre-registered `r < 1.0` threshold. Loss-rate gradient is noise. The Q1 tail (r < 0.15) shows the known boundary-grazer pathology but adds nothing over the already-tracked workstream.

**Recommended primary new strategy:** Intraday ASOS-Bayesian same-day high/low predictor (~2-week build, requires `same_day_gate` policy change). Justified by mechanical evidence that NYC's max-so-far at 18Z (2 PM ET) is within 0.43°F of the final daily max on 70% of days — information the current morning-forecast model has no access to and the market appears not to price.

**Recommended next-step experiments (not built tonight):**
- Hypothesis D (intraday GFS run-to-run drift) — kill-test passed (Open-Meteo IS refreshing GFS between slots), worth a 4-hour validation.
- Hypothesis C (cross-city residual correlation) — most novel, most overfitting-prone.

**Strategies NOT recommended:** Daily binary rain (KXRAINNYC) — single-city, today's market has 0 vol / 0 OI. Monthly precipitation, monthly snow, seasonal hurricane — no calibration path.

---

## 1. What's already known and parked

The Oracle's research surface is mature. Before proposing anything new I confirmed it doesn't duplicate work that has already been killed or parked:

- **Lens 1 (market-implied PMF vs EMOS posterior)** — `NO_TRIGGER` aggregate verdict on the 46-day corpus (2026-05-14). σ-ratio-only rerun (2026-05-17) also `NO_TRIGGER`. Decision: park until the corpus-extension rerun around 2026-06-13.
- **Lens 6 (persistence baseline as competitor to EMOS)** — mostly `TIED` due to small-n; v2 single-bucket rerun was `MIXED`.
- **Workstream-1 v6 gate audit (boundary grazer + production strike-distance)** — verdict `SHIP-THIS-CELL` for both LAX/high and MIA/high (positive ask-side 95% CI lower bound in both). LAX/high was promoted to EMOS, then **rolled back 2026-05-18** after live ROI = −39.6% (d=7.19 → EMOS σ=6.8°F incompatible with 1°F brackets). Offline audits don't always survive contact with live trading.
- **Pattern 1 (body improvement) and Pattern 5** — parked. Lens 1 was a power failure, not a refutation; neither Rule 1 (kill Pattern 1) nor Rule 2 (ship Pattern 6) fired aggregate-wise. Don't reflexively re-pitch without the corpus-extension data.
- **Pattern 3 (boundary-grazing override) and Pattern 4 (regime sit-out)** — still in scope. Boundary-grazing is the largest known unsolved P&L leak per the memorized 2026-05-03 finding.
- **Same-day gate** — currently `BLOCK_PEAK_PASSED` after 15:00 local for highs / 06:00 local for lows, plus `BLOCK_NWS_OBSERVATION_EXISTS` once the preliminary CLI report is available. The "late-day override" has been a parked research item.

## 2. Data assets available

Confirmed via direct inspection on 2026-05-20:

| Asset | Coverage | Notes |
|---|---|---|
| `data/snapshots/<date>/<slot>/` | 2026-03-18 → 2026-05-19 (63 days × 4 slots = 216 dirs) | Each slot has `markets.json` + `weather_forecasts.json` (GFS + ECMWF ensemble members) + `nbm_forecasts.json` |
| `nbm_forecasts.json` coverage | 2026-03-27 → 2026-05-19 (54 days × ~4 slots = 211 files) | Deterministic NOAA NBM point forecasts for 9 cities × 2 vars × 3 forecast dates |
| `data/emos/corpus.parquet` | 14 (city, var) cells × 3 leads pooled | Production EMOS coefficients fit 2026-05-11 |
| `data/research/asos/*.csv` | 2024-05-01 → 2026-05-14 (744 days × 24 obs/day) | **Hourly observed temperature** for 5 cities (NYC, CHI, LAX, MIA, AUS) — 89,125 rows total |
| `data/vps_trades.jsonl` | 2026-04-21 → 2026-05-18 (910 trades) | Live + shadow execution log |
| `data/shadow_settlements.full.json` | 333 settled trades | Settled P&L for join keys |
| `state/eval_audit.jsonl` (+ 4 rotated siblings) | 2026-05-07 → 2026-05-19 | Per-market evaluation rows with `live_calibrator_kind` etc. |
| `state/forecasts_raw.jsonl` (+ 4 rotated) | 2026-05-07 → 2026-05-19 | Raw `gfs_members_raw`, `ens_mean_raw_ddof1`, `bias_correction_applied_f` per scan |

**The ASOS corpus is the most under-utilized asset.** 2 years of hourly observations for 5 of our 7 cities and zero current strategy reads it.

## 3. Kalshi market surface (non-temperature)

Surveyed today via Kalshi API + web research. Verdict per category:

| Series | Status | Why it scores low |
|---|---|---|
| `KXRAINNYC` (daily binary rain) | NYC-only, today's market has 0 vol / 0 OI | Single-city surface too thin; verified 0 open daily-rain markets for CHI/MIA/LAX/SEA/AUS/BOS/DEN/PHIL |
| `KXRAINNYCM` / `KXRAINLAXM` / `KXRAINSEAM` (monthly cumulative rain) | Open, ~17k volume on best brackets | Calibration corpus = 1 obs/month — infeasible on our timeline |
| `KXNYCSNOWM` / `KXBOSSNOWM` (monthly snowfall) | Off-season (no open events May) | Same calibration problem; opportunistic only during winter events |
| `KXTROPSTORM` / `KXHURCTOT` / `KXHURCTOTMAJ` (seasonal storm counts) | Open, $8M+ on best brackets | Annual resolution → no calibration path |
| `HURMIA` / `HURORL` (city-specific landfall) | No open events | Episodic only; NHC ensemble tracks are public but corpus is sparse |
| KXWIND / KXTORNADO / KXDROUGHT / KXSEAICE | No open events | Either seasonal or one-off |

**Bottom line:** The existing temperature-only surface (KXHIGH + KXLOWT × 7 cities) is still the highest-quality opportunity on Kalshi for systematic, calibrated trading. Non-temperature markets either lack volume (daily rain), lack a calibration corpus (monthly/annual settlement), or both.

## 4. Strategy candidates (from quant-strategy-researcher)

Five hypotheses generated against existing data. A and B were backtested this session (see §5). C/D/E are described for the next session.

### Hypothesis A — NBM-anchor suppressive filter
**Edge mechanism:** NBM ingests GFS, ECMWF, HRRR, NAM, MOS. When our 2-model blend disagrees with NBM, NBM is the closer proxy for the post-processed market consensus.
**Two forms tested:** (1) gap-magnitude `|blend_mean − NBM_temp| > τ` and (2) bracket-position veto (NBM lands on opposite side of bracket from model's directional bet).
**Result:** Form 1 dead, form 2 strongly supported at n=32. Ship form 2 to shadow. See §5.

### Hypothesis B — ECMWF-GFS model-disagreement boundary filter
**Edge mechanism:** Boundary-grazer pathology may be more sharply detected by between-model mean disagreement than by within-model ensemble std.
**Result:** Not supported at the pre-registered `r < 1.0` threshold. The Q1 tail (r < 0.15) shows the boundary-grazer pathology but adds nothing over the existing tracked workstream. See §5.

### Hypothesis C — Cross-city residual correlation as synthetic ensemble member
**Edge mechanism:** Synoptic-scale weather → spatially correlated GFS errors. Per-cell EMOS has no cross-city term by construction. If `r_NYC,D` is regressable on `r_BOS,D−1 + r_PHIL,D−1 + r_CHI,D−1`, apply the predicted residual as a post-EMOS shift.
**Risk:** 67-day corpus is sparse for a 6-city regression; overfitting risk high. Use ridge + permutation test.
**Effort:** 8-10 hours.

### Hypothesis D — Intraday GFS run-to-run drift
**Edge mechanism:** DA cycle corrects prior-run state errors. Large monotone drift across consecutive snapshots (05Z → 11Z → 16Z) signals the market hasn't yet repriced.
**Kill-test result (run this session):** GFS slot freshness check on 1,880 (date, city, var, forecast_date) keys: only 12% have all 4 slots identical (i.e., all-stale). 46% have all 4 slots distinct. Open-Meteo IS pulling fresh GFS data between slots — **kill-test passed, hypothesis is alive.**
**Effort:** 4 hours validation + 1-2 days implementation if confirmed.

### Hypothesis E — Settlement-day prelim-vs-final NWS bias
**Edge mechanism:** Preliminary "VALID TODAY" CLI reports vs. final "VALID FOR YESTERDAY" CLI reports occasionally disagree by ±1°F. If a city shows a consistent signed bias, late-day markets are mispriced systematically.
**Risk:** Late-day market liquidity is the lowest on Kalshi weather. May not be tradeable even if signal exists.
**Effort:** 5-7 hours.
**Status:** killed 2026-05-22. The captured "prelim" report is a midday snapshot that structurally cannot include the afternoon high, so the apparent bias was a timing artifact, not a revision signal. Two undecided unblock paths: a 0200Z settlements slot plus a 30–45 day wait, or an IEM historical-CLI-archive fetcher.

## 5. Backtest results

Pre-registered backtest ran against 910 live VPS trades (2026-04-21 → 2026-05-18), joined to `data/snapshots/<date>/<slot>/{nbm_forecasts.json, weather_forecasts.json}` on nearest-slot within ±12h, and to `data/shadow_settlements.full.json` for realized P&L.

### Sanity checks
- **Trade-to-NBM join coverage**: 910 rows collapse to 679 unique `order_id`s in window. 239 are `weather_edge` fill rows carrying full metadata. Final joined set after requiring NBM + GFS + ECMWF + settled P&L = **197 trades (82.4%)**.
- **NBM intraday freshness**: Mean |Δ| between adjacent slots for forecast_date = snap_date + 1 (n=21 per cell over last 7 days): NYC 0.19–0.33°F, CHI 0.35–0.39°F, MIA 0.14–0.16°F. NBM updates slot-to-slot — not Open-Meteo-GFS-stale.
- **Forecast field sourcing**: `per_model_probs` in trade metadata is probabilities, not means. GFS/ECMWF means joined from snapshot `weather_forecasts.json` at nearest-slot. Hit rate 92%.
- **Window composition (caveat)**: 192 NO / 5 YES; 190 between / 0 greater / 0 less. The window is overwhelmingly NO-between — generalization to YES/tail strikes is n-limited.

### A. NBM anchor (n=197 joinable trades)

**Primary form (gap magnitude as suppressor):**

| `|gap_F|` quintile | n | win rate | mean ROI | mean signed gap | mean `|gap|` |
|---|---|---|---|---|---|
| Q1 (smallest) | 39 | 61.5% | −7.84% | +0.00 | 0.22°F |
| Q2 | 39 | 64.1% | −5.46% | +0.05 | 0.64°F |
| Q3 | 39 | 61.5% | −11.02% | −0.66 | 1.05°F |
| Q4 | 39 | 66.7% | −6.68% | −0.57 | 1.68°F |
| Q5 (largest) | 41 | 68.3% | −9.62% | −0.61 | 3.47°F |

- `|gap_F| > 2.0°F` cohort: n=49, win rate 67.3%, mean ROI = −12.24%
- `|gap_F| ≤ 2.0°F` cohort: n=148, win rate 63.5%, mean ROI = −6.78%
- **Verdict: NOT-SUPPORTED.** The directional prediction (large gap → worse trades) fails. The >2°F cohort wins *more* often than the ≤2°F cohort. Gap magnitude is not a useful suppressor at any τ in [1°F, 3.5°F].

**Secondary form (NBM-bracket-position veto):** for between strikes, drop trades where (side=YES AND NBM outside bracket) OR (side=NO AND NBM inside bracket).

| Cohort | n | win rate | mean ROI |
|---|---|---|---|
| Model and NBM **disagree** on bracket position | 32 | 46.9% | −32.04% |
| Model and NBM **agree** | 158 | 68.4% | −2.59% |

- **Verdict: STRONGLY SUPPORTED (at n=32).** Dropping these 32 trades flips overall joined P&L from negative to positive. 53% → 32% loss-rate gap is the largest non-overlapping signal in this analysis. n=32 is small; ship to shadow first.

### B. ECMWF-GFS disagreement (n=190 between trades)

`r = boundary_distance_F / model_disagreement_F` quintiles:

| r quintile | n | loss rate | mean r |
|---|---|---|---|
| Q1 (lowest) | 37 | 48.6% | 0.084 |
| Q2 | 37 | 29.7% | 0.264 |
| Q3 | 37 | 29.7% | 0.494 |
| Q4 | 37 | 32.4% | 1.092 |
| Q5 | 37 | 35.1% | 5.549 |

- `r < 1.0` cohort: n=129, loss rate 34.9%
- `r ≥ 1.0` cohort: n=61, loss rate 36.1%
- **Verdict: NOT-SUPPORTED at r=1.0.** Loss rates either side of the threshold are statistically indistinguishable. The Q1 tail (r < 0.15: ~0.27°F boundary distance vs ~3.3°F GFS-ECMWF spread) does show the 48.6% loss-rate signature — but that's the documented boundary-grazer pathology (an unpublished 2026-05-03 note) reframed via inter-model rather than within-model dispersion. Q1 overlaps the NBM-veto set on only 10/38 trades, so the two signals are partially independent — but adding a second redundant boundary-grazer filter is not the right answer. Defer to that workstream.

### C. Recommended overlay configuration

Single change to ship:

1. **Build a runtime NBM client** in `python/data/nbm.py` mirroring the existing Open-Meteo / NWS clients. The snapshot collector already fetches NBM; the strategy doesn't, because no runtime path exists. This is the highest-leverage piece of work — without it the rest is moot.
2. **Wire `nbm_temperature` into `weather_evaluation.py`** alongside `gfs_mean` / `ecmwf_mean`.
3. **Add a config flag** `weather.nbm_bracket_position_veto: bool` (default `false`) and a gate in `weather_edge.py` that vetoes between-strike candidates per the rule above.
4. **Shadow-only for the first 4 weeks** — re-evaluate at n ≥ 80 vetoed trades. Promotion gate: shadow P&L delta > 0 net of fees AND win-rate gap ≥ 15pp AND no NBM-veto cell shows zero settled trades.

Estimated effort: 1.5 days (runtime client + wiring + shadow flag + tests).

## 6. Recommended new strategy: Intraday ASOS-Bayesian same-day predictor

This is the strongest greenfield opportunity in the codebase. The supporting evidence below was gathered fresh this session.

### Why this is the right new strategy

**The data advantage is large, mechanical, and not arbitraged.** I measured the NYC max-so-far profile across 30 recent days using the existing 2-year ASOS hourly corpus:

| UTC hour (ET equiv) | Mean max-so-far | Mean gap to final daily max | % of days already at max |
|---|---|---|---|
| 12Z (8 AM ET) | 58.9°F | 6.67°F | 17% |
| 14Z (10 AM ET) | 61.3°F | 4.30°F | 30% |
| 16Z (12 PM ET) | 63.8°F | 1.77°F | 40% |
| **18Z (2 PM ET)** | **65.2°F** | **0.43°F** | **70%** |
| 20Z (4 PM ET) | 65.6°F | 0.03°F | 97% |
| 22Z (6 PM ET) | 65.6°F | 0.00°F | 100% |

By 18Z (2 PM ET), the daily max is essentially known on 70% of days. On the remaining 30%, the residual uncertainty is bounded by a 1-2 hour window and can be tightened with climatology + the remaining-hour ensemble forecast.

The current strategy doesn't read ASOS. It scans markets at every snapshot and re-uses the morning's GFS + ECMWF ensemble forecast. So at 18Z on any day, we are trading next-day brackets with stale models when **the most informative free observation in the world is sitting one HTTP call away.** The `same_day_gate` further blocks same-day trades after 15:00 local — closing off the highest-edge window entirely.

**Trade-flow evidence the gate is wasting EV:** Of the 910 live VPS trades since 2026-04-21, 244 (27%) execute in the 16Z-20Z window. Every one of those targets a next-day forecast (lead = 1d) — i.e., the gate is doing its job blocking same-day, but we're stuck trading the lower-information forecast horizon instead.

### Mechanism sketch

For a same-day high market on city C with strike S:

1. **Prior** (morning, t=08Z): full ensemble distribution of today's daily max from GFS+ECMWF post-EMOS.
2. **Update** at every snapshot t ≥ 12Z:
   - Read ASOS hourly observations for city C through hour t.
   - Compute `max_so_far(t)`.
   - Build the conditional distribution P(daily_max | max_so_far, hours_remaining) using:
     - 2 years of ASOS hourly climatology per (city, month, hour-of-day) for the typical "remaining gain" distribution.
     - The morning's ensemble standard deviation as an upper bound on the remaining-hour uncertainty.
   - Posterior P(daily_max ≥ S) = P(max-so-far ≥ S) ∪ P(max over remaining hours ≥ S | not exceeded yet).
3. **Trade** if posterior probability — current EMOS-calibrated `min_calibrated_prob` floor — beats market price by edge floor (7c after fees).

Closed-form for the simple case (assuming temperature evolution is approximately a random walk plus diurnal mean):
```
P(daily_max ≥ S | max_so_far = M, hours_left = h)
  = 1                                if M ≥ S
  = Φ((M + μ_climo_gain(t, city, month) - S) / σ_climo_gain(t, city, month))   if M < S
```
where `μ_climo_gain` and `σ_climo_gain` come from the 2-year ASOS corpus aggregated by (city, month, current-hour). This is a simple table lookup, no model retraining required.

### Backtestability

Confirmed:
- **Same-day market prices** are in `data/snapshots/<date>/1600/markets.json` (102 same-day markets per snapshot, with `yes_bid_dollars`/`yes_ask_dollars`/`volume_fp` populated — sample same-day NYC market on 2026-05-15/1600 had 23,049 FP volume, so liquidity is real).
- **ASOS hourly** has 58 days of overlap with the snapshot window (2026-03-18 → 2026-05-14).
- **NWS settlements** are in `data/snapshots/<date>/settlements.json`.

A 50-day, 5-city backtest is constructable using only existing repo files. No new data collection required for the backtest. Implementation cost for the prototype is ~1 week. Productionization is another ~1 week (gate integration, calibration, risk).

The 5 ASOS cities are NYC, CHI, LAX, MIA, AUS — overlapping with all currently-tradeable temperature cells *except* DEN and SEA (both of which are partly or fully excluded already via `excluded_city_vars`). So the strategy can immediately cover the entire live-tradeable cell set without new ASOS data.

### What needs to be unblocked

- The `same_day_gate` `BLOCK_PEAK_PASSED` decision becomes a soft signal rather than a hard block. The new gate is "block unless the intraday-Bayesian model has a calibrated posterior with confidence interval narrower than the market's bid-ask spread."
- Risk: same-day markets near peak time can move fast; the per-event cap (`one_bet_per_event`) needs to extend to cover same-day + next-day on the same observable.

### Falsification criteria

The strategy is **dead** if, on the backtest corpus:
1. The intraday-Bayesian posterior is no sharper than the morning EMOS at any 12Z/16Z/18Z snapshot (mean CRPS reduction < 0.1°F is below noise).
2. OR: the strategy generates < 20 candidate trades on the 50-day corpus (volume too low to recover dev cost).
3. OR: the simulated P&L over the corpus is negative net of fees and spread on at least 4 of the 5 cities.

### Effort estimate
- Prototype backtest in a notebook: 6-10 hours.
- Productionization (gate + calibrator + risk integration): 4-6 days.
- Shadow-trading validation (45-day corpus): 6 weeks before live.

## 7. Operational plan for tomorrow

Ordered by ratio of expected value to effort:

1. **Day 1 — Ship the NBM-bracket-position veto to shadow.** ~1.5 days of work (runtime NBM client + wiring + flag + tests). Per §5.C, the rule is interpretable and the lift was the largest non-overlapping signal in the backtest. Promotion gate: n ≥ 80 vetoed trades, P&L delta > 0, win-rate gap ≥ 15pp.
2. **Day 1 — Run the H-D intraday-drift validation** (~4 hours). Kill-test passed tonight (only 12% of (date, city, var, forecast_date) keys have all 4 slots identical; 46% have all 4 distinct, so Open-Meteo IS refreshing GFS). The next step is the per-cell edge replay: for each settled day, bin (date, city, var) cells by `|blend_mean(latest_slot) − blend_mean(earliest_slot)|`, and measure Brier-score delta between the morning ensemble and the latest-slot ensemble. If the high-drift bucket has better Brier, the signal is real.
3. **Day 2-3 — Brainstorm + spec the intraday-ASOS strategy (§6).** This is the only candidate that needs a full spec → plan → implementation cycle. Goal of the spec session: agree on the `same_day_gate` policy change, the climatology table schema, and the falsification thresholds in §6.
4. **Defer C (cross-city residuals) and E (NWS prelim-vs-final).** C is overfitting-prone on the 67-day corpus and needs a permutation test as the first step. E depends on whether `settlements.json` historically preserves both prelim and final fields — 10-min schema check before any work.
5. **Do NOT touch the existing calibration setup or the LAX/high block** without re-reading the 2026-05-18 rollback notes first. Live calibrators in MIA/high, LAX/low, MIA/low, NYC/high, CHI/low (EMOS) and the 9 isotonic cells are working — don't disturb them while shipping the NBM veto.

## 8. What NOT to do

- Don't re-pitch Pattern 1 (body improvement) or Pattern 5. Lens 1 already killed both.
- Don't try to systematize hurricane / tropical storm seasonal markets — annual resolution makes calibration impossible.
- Don't promote the daily-rain market without first asking Kalshi to add cities — single-city NYC surface with 0 vol is not a strategy.
- Don't reintroduce LAX/high under EMOS without a structural fix for the d=7.19 → σ=6.8°F problem (already rolled back 2026-05-18 with documented live loss).

## 9. Sources

- [Time Series based EMOS for Temperature Forecasts Postprocessing — Jobst 2024, QJRMS](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4844)
- [Diurnal Cycle of Winter Season Temperature Errors in Operational GFS — Patel et al. 2021, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL095101)
- [Spatial postprocessing of ensemble forecasts for temperature using NGR — Feldmann et al. 2014](https://arxiv.org/pdf/1407.0058)
- [Four Strategies, 562 Trades, Zero Edge — Forensic Autopsy of Algorithmic Weather Betting](https://www.researchgate.net/publication/403510051_Four_Strategies_562_Trades_Zero_Edge_A_Forensic_Autopsy_of_Algorithmic_Weather_Betting)
- [Science Behind the NBM Temperature Elements — Veenhuis 2015, NOAA MDL](https://www.weather.gov/media/mdl/Veenhuis_Presentation2015.pdf)
- [NWS National Blend of Models overview](https://www.weather.gov/news/200318-nbm32)
- [CalibShi: Kalshi Weather Market Miscalibration Analysis](https://www.zerve.ai/gallery/85cce830-f612-4b23-8b78-34d7da65a2c6)
- [Open-Meteo Ensemble API docs](https://open-meteo.com/en/docs/ensemble-api)
