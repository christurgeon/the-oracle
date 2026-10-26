# 2026-06-12 — New-Strategy Search Dossier

**Goal:** find a NEW, unused, genuinely profitable strategy that supports scaling
capital — not another break-even data-collection exercise.

**Process:** multi-agent fan-out (12 agents, 4 phases): 5 parallel research lenses
(category census via live Kalshi API, settlement-mechanic/cross-market arbitrage on
the 91-day snapshot corpus, intraday whitespace, non-temperature public-feed
categories, microstructure timing) → triage → one adversarial refuter per candidate
(default-refute on fee-math or falsified-overlap doubt) → data-grounded deep dives.
All measurements buy-at-ask, exact taker fees (`python/analytics/fees.py`),
expiration_ts-anchored, day-clustered CIs.

**Headline:** 15 raw candidates → 6 triaged → **0 survived the kill panel**. Two
follow-up deep dives: one DEAD (artifact identified), one YELLOW (real but tiny,
forward-gated). The only robust positive expectancy measured anywhere in the sweep
is the **already-live favorite_bias signal**, whose pre-registered n≥40
revalidation was run today (n=46): **CONTINUE-UNCHANGED**, with a concrete
unlock-to-scale gate (~2026-06-26). Conclusion: there is no deployable new edge on
current data; the capital-scaling path runs through (1) the favorite_bias scale
gate, (2) the intraday_asos_high live-flip gate (in flight), (3) a small gated
new-city favorites expansion.

---

## 1. Kill table (all measured, not argued)

| Candidate | Verdict | Decisive evidence |
|---|---|---|
| **AAA gas-price brackets** (KXAAAGASD/W) | WEAK→dead | Liquidity is real (126k contracts/event measured) but market closes ~03:59Z *before* the leading feeds publish (EIA 7:30am ET) — it's a pure forecast play, not a feed-lag play; market already sharp at close (at-money Brier 0.165); realized day-moves (μ=1.9¢, σ=0.9¢) span ~4 of the 0.5¢ brackets while round-trip costs 3–4¢ → needs >5–8pp edge at-the-money that measurements say doesn't exist. Compliance flag (energy contracts) unresolved — would need clearance before any further work. |
| **Hurricane-count tail NO** (KXHURCTOT) | REFUTED | Market's implied survival curve ≈ NegBin(λ≈6) — the "8–18pt gap" was a free-parameter artifact (asserting λ=4.5 when CSU's 6/10 update says ~5). Market already prices El Niño. The headline >8 strike is a stale 0-volume quote (P(>7)=P(>8)=0.215, structurally impossible curve). n=1/season, uncalibratable, capital locked ~6 months, tail depth 4–41 contracts. |
| **Pre-dawn (0200Z) lead-0 favorite underpricing** | REFUTED | No timing premium: paired same-ticker 0200Z vs 0800Z spread = −2.4pts (CI [−20,+14.5]) vs pre-registered kill at <+6pts. The level is the SAME favorite-bias signal already live at lead-1. |
| **Lead-0 favorites generally** (deep dive, §2) | DEAD | +32%/slot claim was a fill-price artifact. Honest: +5.1%, day-clustered CI [−3.4,+13.9], t=1.19; incremental-only slice +0.9% (CI through zero). |
| **Maritime LOW nowcast** (MIA/LAX same-day) | REFUTED | Lock premise TRUE (100% locked by 1400Z) but already priced: MIA/low 1400Z winner-bracket Brier 0.085, buying favorite at ask nets +0.1% ROI after fees (n=31); mid-range spreads median 4¢/mean 6.5¢; zero volume evidence in 616 snapshot rows. Inherits the σ-vs-sharp-band wall from the 2026-06-07 ASOS-prior panel (unpublished). |
| **Cold-front-day regime overlay** | REFUTED | The front-day Brier gap (0.206 vs 0.150, reproduced) is measured at 2000Z — 8–12h AFTER the low printed; it's a stale-quote afterimage, not live mispricing. Front-day calibration curve flat/inverted in BOTH directions (symmetric variance, no directional counterparty). Only temporally-valid window (pre-dawn) = falsified overnight-stale-quote lane. |
| **Cross-city synoptic repricing lag** | REFUTED | Lead-lag tested on existing 6h corpus: contemporaneous correlation strong (NYC–CHI r=0.47) but LAGGED predictive correlation collapses to noise (PHIL r=0.01, BOS r=−0.04); CHI r=0.24 → partial r=0.19, explained by common forecast drift. The "needs minute data" framing was also wrong (API supports period_interval=1). |
| **MECE partition arbitrage** (6-leg sum) | FALSIFIED in survey | 8,042 complete partitions: 604 gross-positive, **2 net-positive after taker fees** (one an artifact). |
| **Single-leg partition-consensus fade** | FALSIFIED in survey | ROI −2.9% to −5.4% across margins; market threshold Brier (0.047) beats bracket-implied (0.052). |
| **Afternoon-prelim CLI deterministic harvest** | KILLED in survey | Prelim→final HIGH exact match only 92%; AUS/DEN prelims issue ~12Z (pre-peak, useless); gated fee-aware economics negative. |
| **CPI brackets** (KXCPI) | REJECTED in survey | Curve tight, fresh, consistent with Cleveland Fed nowcast — efficient money already there. |
| **TSA throughput** (KXTSA/W) | DEAD on arrival | Series exists in census; zero currently-open tradeable events. |

## 2. Deep dive A — lead-0 favorites: DEAD (artifact identified)

The round-1 refuter's +30–37%/slot numbers do not survive honest fills:
fill-at-ask +5.1% vs fill-at-mid +13.3% vs fill-at-bid +22.8% on the same set —
the claim was mid/bid-fill + non-day-clustered inference. Lead-0 favorite spreads
are wide (median 4.5¢, p90 28¢, 40% >6¢) and the ask drifts **+9¢ median against
you by the next slot** (+1.2% at next-slot ask; −12.4% worst-case). 42% of lead-0
events are correlated double-bets on events already bet at lead-1; the genuinely
incremental slice is **+0.9%, CI [−11.7%, +13.3%]**. The only positive slice is
afternoon-HIGH — the already-falsified lock-in mirage. Provenance reconciled: the
spec's lead-0 = −1.85% (t=−0.33) measurement was correctly anchored; the config
comment `lead: 1 # lead-0 has no edge` is **correct — do not change it**.
Live-fill anchor: the lead-1 pilot's fills run +1.6¢ median above mid (exactly the
modeled half-spread), i.e. snapshot-based lead-1 economics are honest — lead-0's
are not, because its books are wide and adverse-drifting.
Raw tables: `/tmp/lead0_favorites/` (session-scratch; method documented here).

## 3. Deep dive B — new-city favorites expansion: YELLOW, gated

Model-free favorite_bias needs no climatology/bias-table, so the usual new-city
calibration blockers don't apply. But: only 4 days of new-city market snapshots
(since 2026-06-09), **n=2 settled qualifying trades** — unfalsifiable today. The
structural funnel is the real finding: **summer heat compresses 6–8 of the 11
cities out of the [0.40,0.75] band entirely** (PHX/high on 06-11 was a single
`>=112F @ 1¢` strike). Realistic incremental capacity ≈ **0.5–1 trade/day,
DC-dominated** (DC ~0.5/day; LV/MIN/SFO occasional; ATL/DAL/HOU/NOLA/OKC/PHX ≈ 0
in summer). Liquidity itself verifies (19k–80k contracts/city by 1400Z; in-band
spreads 1–2¢, same as mature). SFO flips from expected-fail to plausible-pass
(marine-layer low σ → genuine coin-flip favorites).

**Pre-registered per-city gate (shadow/collection only, no live flip):**
≥30 settled in-band vol≥200 observations per city; forward realized-vs-frozen-g
gap ≥ −1 bootstrap SE; win ≥ break-even-after-fees; kill at win<55% @ n≥40 or
ROI<−10% any n. Never go-live on rank across 11 cities (multiple-comparisons).
Two-stage live per spec §7 (B1 reduced-size ask-divergence ≤1¢ over 10–15 fills →
B2). **Do not expand before the mature pilot's own scale gate resolves** —
injecting 11 unproven cities now would contaminate that gate's corpus.
Code path when a city passes: add its two series to `favorite_bias.py` `SERIES`
(new cities use the extra-T `KXHIGHT<CODE>` form); keep `_CORPUS_CITIES` in
`scripts/fit_favorite_bias_map.py` frozen pre-decision.

## 4. favorite_bias n≥40 revalidation (run 2026-06-12, n=46) — CONTINUE-UNCHANGED

Fresh VPS pull + settlement refresh. **n=46, win 63.0% (Wilson CI [48.6%, 75.5%]),
ROI +10.5%** (day-clustered CI [−31%, +45%]) — point estimates squarely inside the
pre-registered +8–12% forward band; kill criteria all clear; second half of pilot
+22.3% ROI / 73.9% win (edge strengthening, not decaying).

- **The report's WARN ("operator should demote") is a false alarm.** The tripwire
  is a deadband-free sign test; the observed gap (−0.0027) is 0.04 SE from zero
  (significance would need ±14pp at this n) and flips sign under the
  settled-trade counting basis. Calibration is healthy: realized ≈ g is the
  *designed* outcome — the edge is the g-to-fill margin, measured directly at
  **+8.15pp net of fees, 46/46 trades with g > fill**, realized +$0.095/contract
  (day-clustered CI [−$0.063, +$0.234]).
  *Tooling note:* consider adding a magnitude deadband (e.g. |gap| > 1 SE) to
  `tripwire_verdict` so the demote message can't fire on noise.
- **Why not scale today:** win-rate lower bound doesn't exclude the 55% kill
  line; ROI CI spans zero; the aggregate is carried by a city subset
  (ORIG-cohort +38.8% vs PHIL/BOS/DEN net-negative; LOCO: drop LAX → +3.5%).
- **Unlock-to-scale gate (pre-registered, check ~2026-06-26):** n≥75 settled AND
  Wilson win-rate LB > 55% AND day-clustered ROI LB > 0 AND both ORIG/NEW city
  splits ≥ break-even. If met → **kelly 0.05→0.075** as the first step
  (max_contracts 25 unchanged — never hit; liquidity is not the constraint,
  fills are pod-scaled Kelly).
- **At n≥50:** re-fit g(mid) per spec and re-run the gate.
- **Watch:** DEN −27% (n=9), BOS −16% (n=8) — per-cell exclusion candidates if
  still negative at n≥15 each.

## 5. Round 2 — cross-market relative value vs sharper public benchmarks

Round 1 never quantitatively tested the ~1,600 non-weather series. Round 2 ran 4
measurement-first lenses (live executable gaps net of fees, not surveys) + the
same kill panel with mandatory independent re-measurement. **6 measured
candidates → 0 survive.**

| Candidate | Decisive evidence |
|---|---|
| **MLB vs DraftKings devigged line** (KXMLBGAME) | Refuter re-measured from scratch: on the liquid slate (vol>50k, n=24) **0/24 positive net edge**, median gross gap −0.67¢ — Kalshi's MLB book tracks DK devig to <1¢ with 250k–1M contracts at the touch. The book IS the sharp line (pro-MM-made). 70-game settled backtest: DK vs Kalshi Brier diff = noise; gap-bet ROI CIs span zero hugely. Only "positive" gaps were next-day openers with no published benchmark yet — unverifiable and uncapturable. Compliance flag (sports) moot. |
| **BTC/ETH daily brackets vs Deribit chain** | The near-dated Deribit ATM mark (16.6% IV) is tick-quantization noise (physically-impossible deep-V smile; DVOL 40%, realized 45–56%); the Kalshi book prices ~40% vol CORRECTLY — the benchmark/sharpness story inverts. Books also empty until the final ~60min; term mismatch (08:00 UTC expiry vs 21:00 settle) alone mis-prices ATM brackets 9–21pp. Fee class verified full 0.07, not index. |
| **BTC 15M/hourly vs realized-vol model** | Strike resets near spot (fair ≈ 0.5 by construction), no options analog at tenor, 2¢ fee on a coinflip, and the live 15M orderbook was **completely empty** across repeated polls. Measurement trap documented: settled-market `last_price_dollars` returns the 0/1 settlement value — never backtest on it. |
| **FOMC vs CME FedWatch** | JUN-17 hold: Kalshi 0.98/0.99 at 1.9M depth vs FedWatch 97.1% — both sides negative net of 1¢ spread + 1¢ fee. The only "gap" leg (SEP) had **20 contracts** at the ask. The most institutionally-arbitraged category on Kalshi; ~8 events/yr. |
| **Treasury/mortgage RV** | Venue abandoned: every KXUSTYLD book frozen since 2026-04-09, 50–100¢ spreads, ~0 volume; weekly KXFRM discontinued since Feb 2024; weekly KX30YUSTW dark since May. Nothing to trade against. |
| **KXINX/KXNDX brackets vs 0DTE chain** (the permitted SPX revival) | Kalshi wings already embed ~1.29× Gaussian tail mass — the book prices the smile, not GBM. Wing half-spread+fee (2.0–3.5pp) ≈ the entire claimed 2–5pp edge; the index-fee "advantage" is FALSE in the wings (1¢ ceil floor binds both classes); cheap wings traded 0–566 contracts/day (median 0) while volume sits at-the-money where there's no edge; the 5¢ wing ask is a chain-quoting MM and a 15-min-delayed feed makes us the slow side. |

**Class-level conclusion:** on every liquid Kalshi book measured, the book is
made by someone at least as sharp as the free public benchmark; gaps exist only
where there is no depth, no benchmark, or no venue. Cross-platform daily-weather
RV also has no venue (probed 2026-06-12: Polymarket's weather tab is long-horizon
climate/disaster only; daily temp markets discontinued).

## 6. Final ranked candidates

1. **Scale favorite_bias through its pre-registered gate** — the only measured,
   live, fee-aware positive edge across both rounds (g−fill +8.15pp net of fees,
   46/46 trades with g>fill). Action ~2026-06-26: re-run the §4 gate; if
   unlocked, kelly 0.05→0.075, then re-gate for further steps. Capacity verified
   not liquidity-bound (fills 1–9 contracts into 200+-deep books, cap never
   hit). This is the nearest-term legitimate capital-scaling step.
2. **New-city favorite-bias expansion (DC-led)** — the strongest genuinely NEW
   deployable candidate: a live-validated mechanism applied to unproven scope.
   Verified 1–2¢ spreads + real volume in-band; gated per §3 (shadow accrual →
   per-city ≥30 settled → B1 fill-quality ≤1¢ ask-divergence → B2 pilot size).
   Small capacity (~0.5–1 trade/day, DC-dominated in summer); additive, not the
   scaling story. Code path: `favorite_bias.py` SERIES tuple only.
3. ~~**AAA gas leading-feed pre-test (KXAAAGASD/W)**~~ — **EXECUTED 2026-06-12 →
   DEAD, well-powered.** Full offline pre-test run (67 settled daily events
   2026-04-07→06-12, AAA prints read exactly from `expiration_value`; close-time
   at-money benchmark from hourly candles, n=66, median 1.0h pre-close; EIA
   wholesale feed; walk-forward OLS/ridge, n=39 OOS). The model genuinely
   forecasts the daily AAA move (ΔAAA MAE 1.00¢ vs 1.83¢ persistence) but that
   skill does NOT transfer to the 0.5¢ at-money binary: market Brier 0.226 vs
   model 0.27–0.34 across 5 specs; model-favored side at ask = **−$0.124/contract
   (ROI −28%), still −$0.100 at zero spread**; when the model disagrees with the
   market by >2pp it is right only **29–39%** of the time (anti-predictive);
   optimal recalibration shrink k=0 (discard the model). Residual σ≈1¢ spans ~2
   brackets — a 1¢-MAE forecast cannot resolve a 0.5¢ boundary the market
   already prices at p≈0.48. Weekly leg (n=9) underpowered AND structurally
   redundant — it ALSO closes the night before its Monday print (the "stays
   open during the week" premise was false). Pre-registered kill triggered;
   the compliance question is moot. Candidate exhausted.

Behind these: **intraday_asos_high remains the highest-EV path to a sizeable
edge** (σ-below-bracket is the one verified structural advantage; flip gated on
the frame-v2 corpus, per-cell n≥20 ≈ 4–6+ wks). Nothing found in either round
outranks it; do not divert its bandwidth.

**Explicit null (both rounds: 22 agents, ~21 distinct candidates):** no new
category, settlement-mechanic arb, microstructure timing, or cross-market RV
edge survived independent re-measurement at the fee bar. Durable lessons:
(a) a backtest fill below the ask or without day-clustered CIs is not evidence —
both +30%-class mirages (lead-0 favorites, pre-dawn premium) died on exactly
that; (b) on liquid Kalshi books, assume the maker is at least as sharp as the
free benchmark until measured otherwise; (c) never backtest on settled-market
`last_price` fields (they return the 0/1 settlement value).

## 7. Round 3 — cross-category favorite-longshot census (does the live edge generalize?)

The one distinct question rounds 1–2 left unasked: the favorite-longshot bias is
*live-proven* on retail-made weather books (g−fill +8.15pp net) — does the SAME
mid-vs-outcome miscalibration exist on OTHER non-MM-made Kalshi books, deployable
through the existing model-free `favorite_bias` machinery? Censused the full
series list, excluded the proven-efficient MM categories (sports/crypto/FOMC/
index), and pulled hourly candles for 13 long-tail series (~3,950 settled markets:
Spotify daily/global/artist/weekly, album/artist streams, Billboard chart
position, Top Album, Netflix show/movie ranks, Top Model, TSA weekly, + **KXGOLDD
as a pro-MM control**). Anchored at T-24h and T-6h pre-close (TTE discipline);
favorite band [0.40,0.80); fills at candle price + {0, 2}¢; exact taker fees;
day-clustered bootstrap CIs; split-half calendar stability required.

**Verdict: EFFICIENT / DEAD, well-powered.** The apparent edge is a measurement
artifact, not a fadeable bias:
- **Pooled (ex-control) favorite side, T-6h:** n=111, win 78.4%, ROI +7.7% — but
  **day-CI [−11.2%, +19.4%] spans zero** and **split-half +17.3% / −13.5%** (2nd
  half negative; fails the weather pilot's own stability bar). T-24h same shape
  (+7.1%, CI [−21.5%, +23.2%], halves +12.7% / −8.1%).
- **Lock-in discriminator (the kill):** the edge is **−6.7% in the near-coin-flip
  zone** (|p−0.5|≤0.15, n=27) — precisely where a real favorite-longshot bias is
  strongest — and only positive (+7.0%) in already-*leaning* markets. Edge
  concentrates AWAY from coin-flips = stale-candle / lock-in signature
  (`intraday_lockin_mature_dead`), not bias.
- **Pro-MM control reproduces it:** KXGOLDD (market-maker-made, should be
  efficient) shows the same favorite-band "edge" (+10.1% T-24h, +0.3% T-6h),
  proving the construction itself manufactures a positive number on efficient
  books — favorites-win-more × stale-candle-vs-settlement.
- **Structural sparsity:** the ranking ladders (Spotify/Netflix/Billboard "who's
  #1") are ~80% longshots in [0,0.05) with a near-empty favorite band; the only
  series that populate the band are weekly *cumulative* streaming totals — the
  exact lock-in mechanism. KXALBUMSTREAMSU (the lone near-zero-lower-bound series,
  T-6h B_favorite CI [−1.2%, +34.9%]) is that mirage, at candle not ask prices.

The only conceivable resurrection is a live-ask probe on KXALBUMSTREAMSU, but the
MM control + coin-flip-zone negativity say it dies there (the round-2 lead-0
lesson: the real ask is worse than the candle). **Not recommended.** This closes
the search's last open question: the favorite-longshot bias does NOT generalize
beyond retail weather books.
