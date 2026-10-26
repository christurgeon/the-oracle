# favorite_bias out-of-sample test (G0) — results

**Run:** 2026-10-01 → 2026-10-02 on the VPS. **Pre-registration:**
`docs/research/2026-10-01-favorite-bias-oos-g0-prereg.md` (committed before the pull in the project's private
history; this public tree cannot show the ordering). **Code:** `scripts/research/favorite_bias_oos_g0.py` (`fetch`, `analyze`). Research only: zero
orders, no runtime or config change. The favorite_bias config freeze (the unpublished tracker) is
untouched, and this doc proposes no config change.

## Verdicts

- **Q1 (G0): FAIL.** Pooled OOS ROI (conservative fees) is **−0.5%, 95% CI [−8.0%, +7.1%]**,
  n = 448 entries on 347 forecast days. Under the pre-registered table, a point ROI ≤ 0 is FAIL:
  treat the 2026 live result as regime or selection luck, and do not size up. The point is
  close to 0. With realistic fees (+0.1% [−7.4%, +7.7%]) and with no fees (+2.9% [−4.7%, +10.4%]),
  the verdict is INCONCLUSIVE instead. No variant shows an OOS edge.
- **Capacity:** on days with an OOS entry, takers bought a median **$88/day** of YES at prices
  that still cleared the edge floor (p25 $3, p75 $363). In the most recent full year (2025) the
  median was **$239/day** (p25 $60, p75 $552). This is an upper bound.
- **Q2 (maker entry): promising by the pre-registered rule. The best horizon is T = 240 min.**
  Resting at the bid for 240 min, then crossing, beat the pure taker by **+1.5¢ per signal
  [+0.5¢, +2.6¢]** on OOS, and by +2.75¢ [+1.1¢, +4.5¢] in 2025. It improves a strategy that
  has no OOS edge: H(240) ROI per dollar is +2.2% [−5.5%, +10.3%]. With a 0.0175 maker fee,
  only T = 10 min keeps a lower bound above 0.

## Q1 by year

Conservative fee = `fee_per_contract` (decides the verdict). Realistic = 0.07·P·(1−P), no rounding.
CIs are day-clustered (`day_clustered_bootstrap_ci`, B = 4000). Full year = ≥ 9 calendar months of
ladder events.

| Sample | Year | Full? | n | Days | Win | Mean g | Mean ask | ROI conservative | ROI realistic | ROI pre-fee |
|---|---|---|---|---|---|---|---|---|---|---|
| OOS | 2022 | partial (Dec) | 3 | 2 | 100% | 0.745 | 0.653 | +50.0% [+34.2, +59.3] | +50.7% | +53.1% |
| OOS | 2023 | full | 77 | 74 | 58.4% | 0.680 | 0.608 | −7.2% [−25.5, +10.5] | −6.6% [−24.9, +11.1] | −3.9% [−22.2, +13.8] |
| OOS | 2024 | full | 105 | 90 | 55.2% | 0.681 | 0.608 | **−12.5% [−26.9, +2.7]** | −11.9% [−26.4, +3.3] | −9.2% [−23.6, +6.0] |
| OOS | 2025 | full | 149 | 121 | 61.1% | 0.688 | 0.607 | **−2.6% [−15.9, +9.6]** | −2.0% [−15.3, +10.2] | +0.7% [−12.7, +13.0] |
| OOS | 2026 (to 03-17) | partial (3 mo) | 114 | 60 | 70.2% | 0.669 | 0.581 | +17.3% [+2.5, +31.7] | +17.9% [+3.1, +32.3] | +20.8% [+6.0, +35.2] |
| OOS | pooled | — | 448 | 347 | 61.8% | 0.680 | 0.601 | **−0.5% [−8.0, +7.1]** | +0.1% [−7.4, +7.7] | +2.9% [−4.7, +10.4] |
| In-sample | 2026-03-18 → 07-31 | — | 264 | 120 | 71.2% | 0.697 | 0.613 | +12.9% [+4.6, +20.8] | +13.5% [+5.2, +21.5] | +16.2% [+7.8, +24.1] |

The last two full OOS years are 2024 (−12.5%) and 2025 (−2.6%). Both are negative, so PASS is
impossible even before the pooled check.

**Sensitivity (pre-registered, does not decide the verdict).**
- Band [0.55, 0.75): −0.5% [−8.5%, +7.2%], n = 328. No change.
- 1-minute entry on the same 448 entered markets: +1.1% [−6.6%, +8.7%], against −0.5% hourly.
  Hourly bars cost about 1.6 pp, but the sign question stays open.
- Realistic and pre-fee: see above. INCONCLUSIVE instead of FAIL.

## Calibration: realized YES rate vs g(mid)

| Year | Realized YES | Mean g(mid) | Gap |
|---|---|---|---|
| 2023 | 0.584 | 0.680 | −0.095 |
| 2024 | 0.552 | 0.681 | −0.128 |
| 2025 | 0.611 | 0.688 | −0.077 |
| 2026 OOS (to 03-17) | 0.702 | 0.669 | +0.032 |
| 2026 in-sample | 0.712 | 0.697 | +0.015 |

g(mid) overstated the YES rate by 8–13 pp in every full OOS year, and it is about right only in
2026. The map is not stale in the usual sense: the recent period is where it fits. It describes a
2026 market that the earlier years do not resemble.

## Exploratory splits (NOT pre-registered; forking paths, no verdict weight)

These splits suggest where the 2026 edge could come from. None of them is a test.

| OOS slice | n | Win − g | ROI conservative |
|---|---|---|---|
| Spread ≤ 2¢ | 112 | −0.031 | +7.9% [−8.2, +23.1] |
| Spread 3–5¢ | 163 | −0.043 | +2.9% [−8.8, +14.6] |
| Spread > 5¢ | 173 | −0.100 | −8.7% [−20.0, +3.1] |
| 2025-07-01 → 2026-03-17 | 158 | +0.007 | +12.6% [−0.0, +24.4] |
| `less` tails | 128 | +0.019 | +13.9% [+0.2, +26.3] |
| `greater` tails | 320 | −0.094 | −6.2% [−15.1, +2.5] |
| MIA_high | 40 | — | −45.4% [−69.8, −20.1] |

The OOS loss is concentrated in 2023–2024, in wide-spread books (mean spread 5.9¢, against 3.4¢
in 2026), and in `greater` tails. The nine months before the fit window already look like 2026.
One reading is that the lift appeared when the books got tighter in mid-2025. Another is that
it is a recent regime with no history behind it. This data cannot tell them apart. Either way,
the multi-year record that would justify sizing up does not exist.

## Capacity (OOS, upper bound)

Per forecast day with an entry: contracts (and dollars) bought by takers on the YES side, from the
entry bar to the end of the lead-1 window, at prices where g(mid at entry) − price − fee ≥ 0.01.

| Year | Days | Contracts p25 / median / p75 | Dollars p25 / median / p75 |
|---|---|---|---|
| 2023 | 74 | 0 / 10 / 119 | $0 / $6 / $49 |
| 2024 | 90 | 0 / 28 / 250 | $0 / $15 / $106 |
| 2025 | 121 | 113 / 512 / 950 | $60 / $239 / $552 |
| 2026 (to 03-17) | 60 | 194 / 466 / 1005 | $120 / $281 / $574 |
| All OOS | 347 | 5 / 168 / 679 | $3 / $88 / $363 |

## Q2: resting YES bid, then hybrid cross (OOS, 448 signals)

Pessimistic fill = a taker sold YES through our bid (print < bid). H(T) = rest for T, then cross at
the 1-minute ask at T if g(mid_T) − ask_T − fee ≥ 0.01, else skip. Taker per-signal ROI = −0.3¢.

| T | Fill opt. | Fill pess. | Win, filled (pess.) | Win, unfilled | Adverse gap win(filled) − win(all) | H − taker, ¢/signal [CI] | H ROI per $ |
|---|---|---|---|---|---|---|---|
| 10 min | 19.4% | 8.3% | 70.3% (n=37) | 61.1% | +0.084 [−0.048, +0.219] | +1.10 [+0.34, +1.97] | +1.4% [−6.2, +9.0] |
| 60 min | 47.1% | 32.1% | 60.4% (n=144) | 62.5% | −0.014 [−0.077, +0.047] | +1.04 [+0.21, +1.93] | +1.3% [−6.2, +9.1] |
| 240 min | 65.4% | 53.8% | 60.6% (n=241) | 63.3% | −0.012 [−0.052, +0.027] | **+1.54 [+0.51, +2.58]** | +2.2% [−5.5, +10.3] |
| 23:55Z (thin book) | 71.0% | 61.4% | 59.3% (n=275) | 65.9% | −0.026 [−0.060, +0.009] | +1.25 [+0.13, +2.33] | +1.8% [−6.3, +9.6] |

H − taker by year (¢/signal, point [CI]):

| T | 2023 | 2024 | 2025 (most recent full) | 2026 (to 03-17) |
|---|---|---|---|---|
| 10 | +0.18 [−0.52, +0.88] | +1.10 [−0.07, +2.93] | +1.48 [−0.16, +3.44] | +1.25 [−0.22, +2.96] |
| 60 | +0.18 [−2.03, +2.53] | +0.70 [−1.38, +2.77] | +1.05 [−0.39, +2.55] | +1.83 [+0.68, +3.26] |
| 240 | −1.56 [−4.74, +1.60] | +1.81 [−0.44, +4.09] | +2.75 [+1.14, +4.51] | +1.68 [+0.04, +3.40] |
| 23:55Z | −2.01 [−6.39, +2.54] | +2.03 [−0.10, +3.95] | +2.63 [+1.38, +3.81] | +0.78 [−1.00, +2.31] |

Maker-fee sensitivity (0.0175 with the per-contract ceil): H − taker = +1.02¢ [+0.26, +1.88] at 10 min,
+0.72¢ [−0.10, +1.59] at 60, +1.01¢ [−0.01, +2.02] at 240, +0.64¢ [−0.46, +1.69] at 23:55Z.

Fills arrive at about the same win rate as all signals at T ≥ 60 min. The adverse-selection gap
is −1 to −3 pp, and every CI includes 0. The gain is mostly price improvement (bid instead of
ask) plus the skips where the edge was gone at T. Weather series charge makers no fee now
(`MAKER_FEE_FREE_SERIES`), so the zero-fee rows describe the current schedule.

## Replay validation against live (Step 4)

Audit assertions all held: every buy row with forecast_date 2026-08-14 → 09-25 carries map hash
`9ad171bb…`, has mid ≥ 0.50, and is not SEA_high. The first buy with `volume_fp < 200` is
2026-09-06T18:58:03Z; the replay used min_volume 200 before it and 50 after. 16 events (forecast_date
2026-09-07) were excluded for the 18:17–18:58Z deploy gap.

| Measure | Value |
|---|---|
| Events compared | 672 |
| Both entered / live only / replay only / neither | 51 / 8 / 0 / 613 |
| Event-level agreement (all events) | 98.8% |
| Agreement among events either side entered | 86.4% |
| Same ticker when both entered | 100% |
| Entry time, replay − live | median +35 min (p10 +5, p90 +113) |
| Mid at entry, replay hourly − live | median +1.5¢ (p10 −2.0¢, p90 +7.5¢), mean abs 3.4¢ |

No replay-only entries means no gate is missing. The 8 live-only entries are short in-band
windows that hourly bars missed. Three of them came within 20 minutes of the 14:00Z open. The hourly replay enters
later than live and at a slightly firmer mid. That is consistent with the 1-minute sensitivity
being 1.6 pp better.

## Data scope

- Ladder-era start per series and the 54 null-`strike_type` events: see the pre-registration.
  1,041 events failed the ladder definition (pre-ladder single-threshold NYC/CHI 2021–22 plus the
  54). 5 `scalar` tails (KXLOWTMIA) were skipped.
- Universe: 9,344 ladder events (forecast_date ≤ 2026-07-31, plus 2026-08-14 → 09-25 for Step 4).
  Hourly candles for every greater/less tail; 1-minute candles and tape for the 771 entered markets
  only. No `between` data was fetched.
- Cutoff `market_settled_ts` = 2026-08-02T00:00Z: earlier markets from `/historical/*`, later from
  the live endpoints.
- Requests: 20,422 in total (Step 0 probes, pilot and full pull), 283 MB downloaded, 94 MB on disk
  (gzip). Status: 20,330 × 200, 84 × 429, 7 × 404 (probes), 1 × 400 (batch-candle probe). Zero
  cache misses at analysis time.

## Deviations

1. **Rate-limit stop rule (2026-10-02, operator decision).** The brief's rule was "stop and report on
   3 or more 429s within 10 minutes". It fired on 2026-10-01 (22:10:56, 22:12:13, 22:17:23Z), once at
   1 request/s and once after slowing to 1 per 1.5 s. The pull was stopped and reported. The operator
   chose to continue at 1 per 1.5 s, with a stop only if a request still returned 429 after a full
   120 s backoff. That never happened: all 84 429s cleared after the first 15 s backoff. The live
   bot's journal showed no 429s during the pull.
2. **Request spacing.** 1.5 s instead of 1 s, from 2026-10-01T22:17Z. This only made the pull slower.

The replay rules, samples and decision tables were not changed.
