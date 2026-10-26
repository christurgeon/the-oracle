# Weather-Edge Deep-Dive: Overconfidence, Edge-Rank Power, and a Market-Anchored Shrinkage Calibrator

**Date:** 2026-06-05
**Corpus:** `data/shadow_settlements.full.json` — 463 settled rows (430 `weather_edge`),
range 2026-04-21 → 2026-06-04. Fresher + larger than the 2026-05-29 Phase-0 audit (347 rows).
**Status:** Diagnostic note. Confirms & extends `2026-05-29-edge-existence-audit.md`. One concrete proposal.

---

## TL;DR

1. **Nothing changed on the newest week.** Last 7 days (≥2026-05-29): weather_edge n=54, **ROI −16.3%**,
   win 46.3%, **100% between markets**. The Phase-0 verdict holds on data it never saw.
2. **The model is overconfident, and it is provable.** Predicted win-prob averages **0.805** vs realized
   **0.591** (+21.4 pt gap). Brier skill vs the base rate is **negative (−0.072)** — the calibrated
   probabilities feeding Kelly are worse than a constant.
3. **Claimed edge has ~zero rank-ordering power.** Pearson corr(model_edge, realized_edge) = **+0.05**
   (n=401, n.s.). Realized edge/contract is pinned near 0 at every edge threshold. We cannot pick winners
   by sharpening or raising the edge bar.
4. **No cell has significant positive realized edge.** Best is MIA/high (t=+1.37); the only significant
   t-stats are *negative* (LAX/low −2.14, NYC/low −1.89).
5. **Optimal model-vs-market blend is λ = 0.00.** `corrected = mkt + λ·(model − mkt)`; Brier rises
   monotonically with any model weight (0.214 → 0.259). Conditional on the market price, the model's
   information is **negative** on the surface it trades.
6. **Additional NWP data will not help at the daily horizon** (corr≈0 and λ=0 mean the market already
   prices every public forecast; three prior nulls agree). The only horizon where new data can pay is
   **intraday (<12h)**, already running in shadow.
7. **A market-anchored shrinkage *calibrator* was proposed and then BACKTEST-FALSIFIED** (§ below):
   recalibrating then re-gating keeps losers / drops winners, because disagreement (what every gate selects
   on) has no predictive power. The fix is structural — **halt the `between` book** — not a better
   calibrator. `between` loses, `greater/less` is positive (n=13).
8. **FORECAST-LAYER root cause found (post-hook): the GFS ensemble is ~8× under-dispersed** (pooled
   RMS(z)=7.85 vs 1.0 calibrated; n=1106), plus large raw μ biases. This *is* the overconfidence, one layer
   up. Three data-backed accuracy levers follow — **L1** per-cell σ (dispersion) calibration, **L2** deploy
   the undeployed NWS-anchored bias table (OOS 3.30→3.00°F MAE), **L3** add intraday observed-temp data
   (hard monotone bound on the daily high — the one lever that makes a 1°F bracket callable). PnL-via-accuracy
   is achievable at the **intraday** horizon (L3); on the daily 1°F surface, accurate = "don't bet."

---

## ⚠️ EXPERT REVIEW CORRECTIONS (2026-06-05 — calibration-analyst, backtesting-analyst, quant-strategy-researcher)

Three specialist agents adversarially re-ran this note against on-disk data. Several headline
claims were overstated or wrong. **Read these corrections before acting on anything below.**

1. **"8× under-dispersed / mechanical root cause" → overstated ~2×.** Production applies a per-cell
   `GFS_DERIVED_SPREAD_INFLATION` (1.0–3.5×, `weather.py:184`) *before* `fraction_between`. The number 7.85
   is RAW σ; the **production-effective RMS(z) ≈ 3.65** (coverage 29%→53%), and ~half of even that is μ-bias
   not dispersion. The "latest run" pairing also injected a nowcast lead-time confound (median lead ≈ −5h);
   at a clean D-1 lead raw RMS(z) ≈ 4.3. Under-dispersion is **real but moderate, and partly already corrected.**
2. **The +21pt traded overconfidence is HALF selection bias, not the σ defect.** The gap GROWS with edge
   (+0.21→+0.26), corr(model_edge, realized) is **−0.25 Spearman**, and the *unconditional* eval stream shows
   only ~+0.10. So it's dominantly optimizer's-curse on a market-priced surface, not "the forecast σ, one layer up."
3. **R1/shrinkage calibrator — RE-TESTED & DEAD (expert-verified 2026-06-05).** My first "falsified (keeps
   losers)" claim was a shadow-sizing artifact. So I re-ran it clean (live-only n=327, 38 days, fixed-stake,
   global λ, day-clustered CIs) and the honest verdict is still NEGATIVE, for the right reason now:
   (a) GATE form — no selection edge (kept realized edge doesn't rise as λ falls; λ=0.25 +0.157 is n=14 noise);
   (b) SIZING form — log-growth improves as λ falls (0.50×→0.93×) but the DECISIVE matched-exposure test shows
   shrinkage ≈ a uniform Kelly cut at every λ (diffs −0.004..−0.000, CI centered 0). Shrinkage scales every bet
   by ~0.46× (CV 0.088) → a near-pure uniform multiplier → zero reallocation value. **It's a complicated way to
   lower `kelly_fraction`. Don't build it.** And on a book with realized edge −0.04, growth-optimal Kelly is the
   corner **f=0** → the honest action is halt/minimize, of which lowering kelly_fraction is only a partial form.
4. **"between loses vs greater/less wins as a structural separator" → REFUTED as evidence.** The greater/less profit is ONE
   oversized shadow trade (CHI/low greater, 415 contracts); remove it → roughly zero, CI spans zero. The
   between book is genuinely bad (stands on its own n + priors); the "greater/less is profitable" half is unsupported.
5. **L1 (per-cell σ calibration) is NOT a new lever** — it's the deployed spread-inflation + the *suspended*
   per-cell EMOS σ stack (`emos.py:91`). Honest σ lands at ~4–6°F → trips the `incompatible_sigma` veto on
   1°F brackets → self-halts the between book. Real, but already pulled; loss-avoidance not PnL-generation.
6. **L3 is ALREADY a PARKED prototype** — `docs/research/2026-05-21-asos-bayesian-prototype-results.md`
   (an unpublished 2026-06-05 note). Mechanism (G1 sharpness) robust; P&L unverifiable (CI
   spans zero), blocked on calibrator extrapolation (30–47% of trade rows) and **adverse selection — the monotone
   bound is PUBLIC** (MMs read the same ASOS feed). Strongest cells: **AUS/high, CHI/high ~2000Z** (NOT MIA/LAX).

**Net corrected priority:** **L2 (NWS bias table) is ALREADY DEPLOYED** — pre-flight on 2026-06-05 found it
shipped 2026-06-02 and running in the author's deployment (`bias_table_path` set,
oracle-bot.service active, re-validated MAE 3.30→3.00°F). The "deploy L2" recommendation was already done 3
days earlier; nothing to flip. The **shrinkage calibrator was re-tested clean (live-only, matched-exposure,
expert-verified) and is DEAD** — no selection edge; sizing benefit = pure exposure reduction = lowering
kelly_fraction; optimal Kelly on this book is f=0 (halt). **L3 = restart the parked ASOS prototype**
(`docs/research/2026-05-21-asos-bayesian-prototype-results.md`) with an obs-fed shadow lane on AUS/high+CHI/high,
gated on calibrator extrapolation <25%. L1 = re-enabling EMOS σ (self-vetoes on 1°F brackets).

---

## 1. The overconfidence is the loss mechanism (answers "don't be overconfident despite losing high %")

Reliability on the *traded* win-probability (YES→model_prob, NO→1−model_prob), weather_edge, n=401:

| pred-bucket | n | avg_pred | realized | gap | ROI |
|---|---|---|---|---|---|
| 0.00–0.20 | 13 | 0.181 | 0.077 | **+0.104** | −92% |
| 0.20–0.35 | 20 | 0.214 | 0.050 | **+0.164** | −31% |
| 0.65–0.80 | 32 | 0.759 | 0.312 | **+0.447** | +32% |
| 0.80–1.01 | **331** | 0.874 | 0.671 | **+0.203** | −1.5% |
| **overall** | 401 | 0.805 | 0.591 | **+0.214** | — |

82% of the book lives in the 0.80+ bucket: the bot pays ~0.87 for a contract it wins 67% of the time
→ EV ≈ −0.18/contract before fees. **Kelly sizes on the inflated 0.87, so it over-bets a negative-EV book.**
This is the literal "overconfident despite a 60% win rate" pathology — the wins are cheap (NO at 0.68,
pays 0.32) and the losses are confident and full-priced.

## 2. Why the edge number is noise — the optimizer's curse (Smith & Winkler 2006)

Selecting max-edge bets from noisy probability estimates *guarantees* realized edge < estimated edge.
Measured here, per edge-bucket, the gap ("shrink needed") is a near-constant **~0.21**:

| edge bin | n | model_edge | realized_edge | shrink |
|---|---|---|---|---|
| 0.08–0.15 | 103 | 0.131 | −0.082 | 0.213 |
| 0.15–0.25 | 233 | 0.190 | −0.028 | 0.218 |
| 0.25–0.40 | 64 | 0.294 | +0.025 | 0.268 |

Claimed edge ≈ a constant +0.21 bias on top of ~zero real edge. Corr(edge, realized) = +0.05 ⇒ no signal.

## 3. The proven fix is to anchor on the market, not the model

`corrected_p = mkt + λ·(model − mkt)`, λ minimizing realized Brier:

| λ (0=market, 1=model) | 0.00 | 0.20 | 0.50 | 1.00 |
|---|---|---|---|---|
| Brier | **0.2141** | 0.2175 | 0.2278 | 0.2591 |

Global optimum **λ=0.00**. Per-cell: every daily-between cell collapses to λ≈0; the lone exception is
**MIA/high (λ≈0.45, n=37 — weak)**. This *is* the per-cell-calibrator philosophy already in the codebase.

---

## Backtest of R1 (run before building) — **R1's gate form is FALSIFIED**

Counterfactual on the corpus: fit λ_cell out-of-sample (leave-one-day-out, Brier objective), recompute
corrected edge `= mkt + λ_cell·(model−mkt) − price`, re-gate at `edge_threshold=0.07`.

| book | n | ROI |
|---|---|---|
| ACTUAL (all traded) | 401 | −1.0% |
| KEPT under R1 | 42 | −33.5% |
| DROPPED by R1 | 359 | +2.9% |

**R1 keeps the losers and drops the winners.** Root cause (independently confirmed): realized ROI by
model-vs-market *disagreement* quartile is −9.6% / +4.4% / −3.7% / +0.6% — non-monotone, all ≈0. Since
**any** edge gate selects on disagreement, and disagreement has zero predictive power, **no
recalibrate-then-re-gate scheme can produce positive PnL on this book.** The overconfidence diagnosis is
correct; the *fix* is not a better calibrator — it is to **not trade the surface at all.**

## Recommendations (ranked, all data-backed — corrected post-backtest)

### R1′ — **Minimize / halt the daily-`between` book** (the structural lever)
The clean separator is strike_type, not any statistic: `between` n=388 loses; `greater/less`
n=13 is positive. There is no rescuable subset of `between` (corr(edge,realized)=+0.05, disagreement
quartiles flat, λ*=0). The only PnL-positive action is to stop the structural bleed — set the between book
to zero / token size. This *is* the 2026-06-05 between-book verdict (unpublished: "hold minimal, focus
intraday"), now re-confirmed on fresher data **and** with the tempting calibration rescue explicitly falsified.

### R2′ — Keep the behavioral `greater/less` surface, sized small
The greater/less profit lives entirely on the directional/behavioral surface, which is what `favorite_bias`
(live pilot, +6.9% ROI n=23) already targets. Continue it at bounded size; do **not** layer a forecasting
edge gate on top.

### R3′ — Put all new accuracy effort at the **intraday (<12h)** horizon
Different horizon, not subject to the daily-efficiency ceiling; the
intraday EMOS shadow is already accruing forward calibration. This is the only place additional *data*
(higher-cadence obs, intraday NWP) can plausibly create edge.

### R-NOT — Do **not** add daily NWP data sources (NBM/ECMWF/GEFS/HRRR) to chase accuracy
corr≈0 and λ=0 prove the market already prices public forecasts; NBM-gap, rain, and the strategy-search
nulls all agree. New-data effort belongs at the **intraday** horizon only.

---

## Forecast-layer analysis — WHERE accuracy can actually improve (added 2026-06-05, post-hook)

The betting-layer analysis says the *traded* probabilities are at ceiling. But that traces to a fixable
defect one layer up. Pairing raw GFS ensemble (μ, σ) against settled NWS temps (n=1106, all cells):

**The ensemble is ~8× under-dispersed.** Pooled RMS of the standardized error z=(T−μ)/σ is **7.85**
(calibrated = 1.0). *Every* cell is under-dispersed (LAX/low 12.1, MIA/low 12.1, DEN/low 11.6, NYC/high 2.6).
The ensemble reports σ≈0.4–2.1°F but realized error is **3–6°F MAE**. **This is the mechanical root cause of
the +21pt betting overconfidence** — bracket probabilities are computed from a σ that is ~8× too narrow.
A single global isotonic curve cannot fix this because per-cell dispersion ranges 5× (z: 2.5→12).

**Large residual μ biases** (raw GFS, pre-correction): MIA/high −3.7°F, DEN/low +5.7°F, CHI/low +3.3,
LAX/low +3.2, MIA/low +3.3, SEA/high +2.3. A separable accuracy lever.

**Error falls with lead, but honest σ stays >> 1°F even same-day:**

| lead (h) | n | MAE | RMSE (honest σ) | ens_σ | RMS(z) |
|---|---|---|---|---|---|
| −12..6 (same-day) | 1600 | 2.88 | **3.96** | 0.79 | 8.08 |
| 6..18 | 1662 | 3.55 | 4.95 | 1.21 | 5.70 |
| 30..48 | 2686 | 4.02 | 5.42 | 1.62 | 4.28 |

**Implication that reconciles everything:** honest σ (≈4°F at D-1, still ~4°F even same-day from NWP alone)
is *much wider* than a 1°F bracket. So a *correctly calibrated* forecast assigns low, flat probabilities
across 1°F daily brackets — i.e., **accurate prediction on this surface means correctly saying "I can't
call it."** That is precisely why the market beats the model and why the between book is unwinnable. The
EMOS "σ-incompatible with 1°F brackets" veto was the model telling the truth. **Making predictions more
accurate and halting the between book are the same conclusion, not opposed.**

### The three data-backed accuracy levers (this is the goal-responsive answer)

- **L1 — Per-cell dispersion calibration (honest σ).** Inflate σ to realized-error dispersion (EMOS done
  per-cell). Cures overconfidence at the source; correctly sizes the bets that survive; reprices 1°F daily
  brackets toward the market (→ self-halt). Mathematically grounded (PIT/coverage; RMS(z)→1).
- **L2 — Deploy the NWS-anchored bias table** (`data/nws_bias/bias_correction.json`, OOS-validated
  3.30→3.00°F, currently **undeployed** behind `weather.bias_table_path`=null). A sitting μ-accuracy gain
  that directly removes the −3.7/+5.7°F residual biases.
- **L3 — NEW DATA SOURCE: intraday observed temperature (METAR/ASOS hourly).** NWP RMSE ~4°F can't call a
  1°F bracket even same-day — but the *realized* high-so-far is a **hard lower bound** on the final daily
  high (monotone): once X°F is observed, P(final high < X)=0, collapsing the lower tail to zero. This is
  information *outside* the NWP ensemble and outside what the morning market priced. It is the **only**
  lever that makes a 1°F bracket genuinely callable and the only place "more accurate → more PnL" is
  achievable. Belongs to the intraday strategy (already in shadow); feed live obs into its posterior.

**Verdict on the goal's premise:** "improve PnL via more accurate predictions" IS achievable — via L1/L2
(cure overconfidence, correct μ) and especially **L3 at the intraday horizon**. It is NOT achievable by
sharpening a *daily* NWP forecast of a 1°F bracket; there the honest, more-accurate answer is to not bet.

## Reproduction
All numbers from `data/shadow_settlements.full.json` via inline scripts (see session 2026-06-05).
Cross-checks: `2026-05-29-edge-existence-audit.md` (T1/T2/T4),
`2026-05-30-nbm-gap-feasibility-spike.md`, and two unpublished notes.
