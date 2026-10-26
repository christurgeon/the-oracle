# NBM-Gap Taker-Viability Feasibility Spike (expiration_ts-anchored)

Date: 2026-05-30
Script: `scripts/nbm_gap_spike.py` (reproducible, seed=20260530)

## The Question

On Kalshi directional (greater/less) weather markets, is there a TAKER-VIABLE
"NBM-gap" edge — do markets where the NBM point forecast (turned into a bracket
probability via a Gaussian) disagrees with the posted price beyond fees+spread
earn positive ROI, net of fees + realistic spread, at a TRUE tradeable lead, with
a day-clustered 95% CI lower bound > 0?

## Pre-Registered Verdict Criteria (committed BEFORE looking at results)

- **TAKER-VIABLE:** ≥1 (σ, gap_threshold) config has day-clustered CI_lo > 0 at a
  TRUE ≤24h lead AND is not a lead-only artifact (positive or flat — not negative
  — at 12h) AND survives a 2¢ assumed half-spread.
- **NULL:** no config clears, OR the only positive configs are long-lead artifacts
  / die to a 2¢ spread.
- **UNDERPOWERED:** n (directional markets joined to NBM) is tiny (<~80); report
  what n would be needed.

n is reported prominently below.

## Method (as run)

1. **p_NBM(bracket)** from the NBM point forecast via a Gaussian. For `greater X`:
   `p_YES = 1 − Φ((X − NBM)/σ)`. For `less X`: `p_YES = Φ((X − NBM)/σ)`.
   σ swept ∈ {2.0, 2.5, 3.0, 3.5}°F.
2. **Anchor = true `expiration_ts`** (settlement), not the last traded candle.
   Entry candle = the close nearest `expiration_ts − lead` (within ±8h tolerance).
   This is the corrected anchor from `scripts/favorite_longshot_verify.py:97-100`,
   adopted to avoid the time-to-expiration confound that RETRACTED the sister
   favorite-longshot analysis.
3. **Candles carry ONLY `yes_price_close`** (verified: candle keys are
   `end_period_ts, open_interest, volume, yes_price_close` — no bid/ask). The
   executable price is therefore modeled with an EXPLICIT assumed half-spread,
   swept {1¢, 2¢, 3¢}. Taker YES fill = price + half_spread; taker NO fill =
   (1 − price) + half_spread.
4. **NBM-gap trade rule:** buy YES when `p_NBM − price_YES > gap + fee + half_spread`;
   buy NO when `(1−p_NBM) − price_NO > gap + fee + half_spread`. gap_threshold swept
   ∈ {0.03, 0.05, 0.08, 0.12}.
5. **Realized P&L** vs settled `result` (authoritative `result` field, Kalshi-settled,
   YES/NO), taker fee via `python.analytics.fees.fee_per_contract`, explicit slippage.
   ROI = Σpnl / Σwagered.
6. **Day-clustered bootstrap 95% CI.** Cluster = (city, var, forecast_date) — the
   shared temperature draw. B=2000, pinned seed.
7. **Lead sweep:** 12h / 24h / 36h / 48h (the mandatory confound check).
8. **NBM leakage guard:** NBM records carry `fetched_at`. For each market+lead the
   join uses the LATEST NBM record with `fetched_at ≤ entry_ts`. No future NBM.

## Dataset / Join Funnel (@ 24h lead)

| Stage | Count |
|---|---|
| Directional settled markets in corpus (greater/less, result∈{yes,no}) | 2255 |
| → with a valid expiration-anchored candle (±8h of target) | 1000 |
| → joined to a leakage-safe NBM record | **776** |
| Clusters (city/var/date) | 629 |
| Event-date span | 2026-03-28 .. 2026-05-28 (62 dates) |

n at the headline 24h lead is **776** — well above the underpowered threshold. The
join shrinks from 2255 because (a) PHIL/BOS have no NBM coverage, (b) many markets
have no candle within ±8h of `expiration_ts − 24h` (the very candle-quiet behavior
that motivated expiration_ts anchoring), and (c) the NBM corpus starts 2026-03-27.
n grows with lead (more candles closer to expiration survive the ±8h gate): 24h
selects ~456–584 trades depending on config; 36h/48h select 500–1000+.

## TAKER Results — σ × gap_threshold × lead (half_spread = 2¢)

ROI [day-clustered 95% CI] (n). **Every cell is negative.** Representative slice
(full grid in script output; pattern is uniform):

| σ | gap | L12h | L24h | L36h | L48h |
|---|---|---|---|---|---|
| 2.0 | 0.03 | −26.7% [−52.5,−8.0] n202 | −18.5% [−29.1,−8.0] n466 | −17.1% [−24.5,−9.9] n774 | −18.5% [−24.8,−12.5] n926 |
| 2.5 | 0.05 | −38.8% [−74.2,−13.9] n200 | −23.0% [−35.4,−10.8] n456 | −22.0% [−30.6,−13.3] n764 | −23.6% [−31.6,−15.6] n860 |
| 3.0 | 0.08 | −39.7% [−74.2,−15.9] n204 | −27.3% [−42.9,−11.5] n436 | −25.5% [−35.2,−15.8] n715 | −30.6% [−39.4,−21.8] n812 |
| 3.5 | 0.12 | −37.7% [−73.4,−12.4] n201 | −28.0% [−47.2,−8.2] n407 | −25.9% [−37.7,−14.5] n638 | −34.8% [−45.1,−24.1] n745 |

**Positive taker configs at lead ≤24h with CI_lo > 0 and n ≥ 30: NONE (0 of 16 σ×gap cells).**

At 24h, every single cell's CI *upper* bound is also negative — the negativity is
not marginal. The ROI gets *worse* at shorter leads (12h is the most negative row
in every config), which is the opposite of a long-lead price-convergence artifact.

## Spread Sensitivity (σ=2.5, 24h)

| gap | hs=1¢ | hs=2¢ | hs=3¢ |
|---|---|---|---|
| 0.03 | −20.9% [−32.8,−9.4] | −22.9% [−34.1,−11.6] | −26.2% [−38.3,−14.6] |
| 0.05 | −18.8% [−31.5,−5.5] | −23.0% [−35.8,−10.0] | −27.6% [−41.0,−13.3] |
| 0.08 | −20.2% [−35.9,−3.3] | −24.1% [−39.7,−8.6] | −28.0% [−43.3,−12.2] |
| 0.12 | −22.6% [−40.2,−5.1] | −25.2% [−43.8,−7.5] | −29.3% [−47.0,−11.6] |

Already negative at the most generous 1¢ assumption; gets worse with spread. There
is no spread level at which this turns positive.

## Maker Upper Bound (spread captured, maker fee, 24h, hs=2¢)

> **Correction 2026-10-01:** weather series charge makers no fee (`fee_type: quadratic`; see the root `CLAUDE.md`, "Kalshi fee integration"). This upper bound deducts a maker fee that is not charged, so it is slightly too pessimistic. Re-run before relying on its sign.

NOT a deployable number — candle data has no bid/ask, no fill-probability, and no
adverse-selection model, so maker fills are unmodeled. This is a pure upper bound:
"is it even worth a maker shadow test?"

| σ | gap | taker ROI | maker ROI (upper bound) |
|---|---|---|---|
| 2.0 | 0.05 | −19.4% [−30.8,−7.9] | −4.9% [−19.1,+9.1] |
| 2.5 | 0.12 | −25.2% [−43.6,−6.9] | −5.1% [−25.4,+16.6] |
| 3.0 | 0.12 | −26.2% [−45.0,−6.9] | −3.1% [−27.7,+20.4] |
| 3.5 | 0.12 | −28.0% [−48.0,−8.7] | −2.8% [−27.9,+23.6] |

Even the maker UPPER BOUND has a negative point estimate in every cell (best
−2.8%) and CI_lo < 0 everywhere. Capturing the full spread for free is not enough
to make the gap rule positive. A maker shadow test is **not** warranted.

## Why It's Negative — The Decisive Control

A sanity control (join @24h, n=776, σ=2.5):

- **NBM has genuine directional skill:** point-forecast directional agreement with
  the settled result = **75.3%**; NBM-Gaussian(σ=2.5) Brier vs outcome = **0.199**
  (< 0.25 coinflip baseline). The gap logic is NOT inverted.
- **But the market is far sharper:** market-price Brier vs outcome = **0.071** —
  nearly 3× better than NBM.

The market is a much better forecaster than an NBM point forecast wrapped in a
Gaussian. Every "gap" the NBM detects is therefore the NBM being wrong relative to
a sharper market, not the market being wrong. Selecting trades *on the size of the
gap* is textbook adverse selection: you trade hardest exactly where your inferior
model most disagrees with the superior price. Among gap-selected trades the settled-
YES rate is 36.6% across a balanced 252 YES-buys / 204 NO-buys — i.e. the rule
systematically lands on the losing side.

## Adversarial Self-Check

- **Is the anchor truly `expiration_ts`?** Yes — `load_joined` anchors on
  `m["expiration_ts"]` and drops any market with no candle within ±8h of
  `expiration_ts − lead`. No last-candle anchoring anywhere.
- **Is it a lead-only artifact?** No — and it fails the confound test in the
  *protective* direction: ROI is most negative at 12h and least negative at 36h.
  A long-lead price-convergence artifact would look positive at long leads and
  decay; this is the opposite. There is no lead at which it's positive.
- **Does it survive σ and spread sweeps?** It's negative across ALL 4 σ values,
  ALL 4 gap thresholds, and ALL 3 half-spreads. No corner of the grid is positive.
- **Concentrated in 1–2 city/dates?** N/A — there is no positive config to
  concentration-check. The negativity is broad (629 clusters, 62 dates, 7 cities).
- **NBM leakage?** Guarded: only NBM records with `fetched_at ≤ entry_ts` are
  joined. If anything this is conservative (uses a slightly staler NBM than a live
  bot might). Relaxing it could only *help* NBM, and it's still negative.
- **Could a different p_NBM mapping rescue it?** Unlikely — the failure is not the
  σ choice (swept) but that NBM Brier (0.199) ≫ market Brier (0.071). No monotone
  reshaping of a forecast that's 3× noisier than the price can produce a gap that
  is the *market's* error rather than NBM's. The ceiling here is the maker upper
  bound, which is also negative.

## VERDICT

**NULL.** No (σ, gap_threshold) config is taker-viable at any lead ≤24h (0/16 cells
have CI_lo > 0; CI upper bounds are negative too). The effect is robust to the σ,
gap, spread, and lead sweeps, is *not* a long-lead artifact (it worsens at short
leads), and the maker upper bound is also negative. Root cause: an NBM-point-forecast
Gaussian (Brier 0.199) is ~3× noisier than the Kalshi price (Brier 0.071), so gap-
selection is adverse selection against a sharper market. n=776 — well-powered, not
underpowered. Do not pursue an NBM-gap directional strategy, taker or maker.
