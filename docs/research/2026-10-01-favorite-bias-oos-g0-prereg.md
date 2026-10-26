# favorite_bias out-of-sample test (G0) — pre-registration

**Date:** 2026-10-01. **Committed before the full pull and before any
result was seen**, in the project's private history; this public tree cannot show the ordering. Results: `docs/research/2026-10-01-favorite-bias-oos-g0-results.md`.

Questions (research only, zero orders, no runtime or config change; a favorite_bias config freeze is in
effect, the unpublished tracker):

- **Q1 (G0):** would favorite_bias's exact live rule have made money in years the g(mid) map was never
  fitted on? Is the edge stable year to year, or shrinking? How many dollars a day can it absorb?
- **Q2 (maker entry):** if favorite_bias rested a YES bid instead of crossing to the ask, how often would it
  fill, and do filled trades still win (adverse selection)?

The rules sections below are copied verbatim from the task brief. They are not edited after results are
seen. Any rule that proves impossible gets a dated **Deviation** section at the end; the original text stays.

## Data scope (Step 0 probes, 2026-10-01, ~140 unauthenticated requests, zero 429s)

**Routing.** `GET /historical/cutoff` on 2026-10-01: `market_settled_ts` = `trades_created_ts` =
2026-08-02T00:00Z, `orders_updated_ts` = 2026-09-17T00:00Z. A market whose `close_time` is before
2026-08-02T00:00Z is read from `/historical/*`; later markets from the live endpoints
(`/markets`, `/series/{s}/markets/{t}/candlesticks`, `/markets/trades`). Historical candles use
`yes_bid.close`/`volume`; live candles use `yes_bid.close_dollars`/`volume_fp`. Both are dollars. They are
normalized to one schema.

**Listing.** `/historical/markets?series_ticker=KX…` pages back through the pre-`KX` families
(`HIGHNY-*` appear under `KXHIGHNY`). No bisection needed.

**Ladder-era start per series** (ladder event = exactly one `greater`, exactly one `less`, ≥ 2 `between`):

| Series | Ticker families | Ladder-era start (forecast_date) | Ladder-era events failing the definition |
|---|---|---|---|
| KXHIGHNY | HIGHNY (2021-08 → 2024-10-23), KXHIGHNY | 2022-12-22 | 4 |
| KXHIGHCHI | HIGHCHI, KXHIGHCHI | 2022-12-22 | 9 |
| KXHIGHMIA | HIGHMIA, KXHIGHMIA | 2023-05-11 | 14 |
| KXHIGHAUS | HIGHAUS, KXHIGHAUS | 2023-05-11 | 5 |
| KXHIGHDEN | KXHIGHDEN | 2024-11-20 | 8 |
| KXHIGHPHIL | KXHIGHPHIL | 2024-11-20 | 6 |
| KXHIGHLAX | KXHIGHLAX | 2025-01-05 | 8 |
| KXHIGHTBOS | KXHIGHTBOS | 2026-02-05 | 0 |
| KXLOWT{NYC,CHI,MIA,LAX,DEN,AUS,PHIL} | KXLOWT* only | 2025-12-13 | 0 |
| KXLOWTBOS | KXLOWTBOS | 2026-04-03 | 0 |

- Every ladder-era failure (54 events) falls in 2025-01-16 → 2025-02-10: one tail market carries
  `strike_type: null` (e.g. `KXHIGHMIA-25FEB05-T79`). They are excluded by the event-structure rule.
- NYC/CHI events from 2021-08 to 2022-12 are single-threshold; excluded by event structure.
- No pre-2025-12 low-temperature series exists on `/historical` (`KXLOWNY`, `KXLOWNYC` list zero markets).
  OOS lows therefore cover only 2025-12-13 → 2026-03-17, and BOS is absent from OOS.

**Listing time.** 99.4% of markets open at 14:00 or 15:00 UTC on the lead-1 UTC day; the rest open later on
lead-1 or on lead-0. None open before lead-1, so cumulative volume from `open_time` equals volume inside the
lead-1 window.

**Candles.** Hourly and 1-minute bars both omit intervals with no change; both are forward-filled.

**Taker side.** `taker_book_side` is collinear with `taker_outcome_side` (`yes`↔`bid`, `no`↔`ask`) in every
probed trade. Against the prevailing 1-minute quote, 178/189 `yes`-taker trades print at the ask and 124/134
`no`-taker trades at the bid (9 markets, 2023–2026). So "taker bought YES" = `taker_outcome_side=="yes"` and
"taker sells YES into the bids" = `taker_outcome_side=="no"`.

**Outcomes.** All markets are `finalized`. Greater/less `result` values: `no`, `yes`, and 5 `scalar`
(all KXLOWTMIA). `scalar` markets are excluded and counted.

**Batch candles.** `GET /markets/candlesticks` works on the live tier only; historical markets return `[]`.
Historical candles are one call per market.

**Projection.** ~16,900 hourly-candle calls + ~50 live-tier calls + ~5,800 minute/tape calls for entered
markets ≈ 23k requests, ~200 MB downloaded, at 1 request/s on the VPS. Within the 60k / 2 GB budget.

## Replay rule (mirrors python/strategies/favorite_bias.py)

Write the replay as a function of a **rule config**: band_low, band_high, min_volume, edge_floor,
the g-map path, and excluded city/vars. That lets Step 4 replay the config that was live at the
time. The **study config** is today's live rule:

| Parameter | Value |
|---|---|
| band | [0.50, 0.75) |
| min_volume | 50 |
| edge_floor | 0.01 |
| g map | `data/calibration/favorite_bias_isotonic.json` (fit 2026-07-04) |
| exclude | SEA_high |

- **Universe:**
  - Ladder events only (Step 0.3). Within each event, take only the `greater` and `less` markets.
  - Do not fetch candles or tape for `between` markets.
  - Exclude single-threshold events explicitly, by event structure. Do not rely on
    `strike_type` being null.
- **Lead-1 window:** the UTC calendar day before forecast_date. Live sets
  `scan_today = datetime.now(UTC).date()` and requires `forecast_date - scan_today == 1`.
  - Take forecast_date from the event ticker.
  - Old-format tickers may not parse with `python/utils/ticker_dates.py`. Write a small local
    parser in the research script. Do not modify the util.
- **At each hourly bar close inside the lead-1 window:**
  - Read yes_bid and yes_ask (bar close values, forward-filled), and compute
    mid = (bid + ask) / 2.
  - Cumulative volume = all volume from the market's `open_time` to that bar close.
  - The market qualifies at that bar if all of these hold:
    - 0 < bid < ask < 1
    - band_low ≤ mid < band_high
    - cumulative volume ≥ min_volume
    - edge = g(mid) − ask − `fee_per_contract(ask, is_maker=False)` ≥ edge_floor
  - Load g with `python.data.calibration.IsotonicCalibrator`, the way favorite_bias.py does.
    Import `fee_per_contract` from `python.analytics.fees`. Do not reimplement either.
- **Entry:**
  - Take the first qualifying bar per (city, var, forecast_date) event.
  - If several markets in the event qualify on that bar, take the one with the highest edge. This
    mirrors the live per-day event cap of 1.
  - Taker entry price = that bar's yes_ask close.
  - Stake 1 contract. This is expected-value research, not sizing.
- **Outcome and P&L:**
  - Use the market's settled `result`.
  - P&L per contract = 1{yes} − ask − fee. Report three fee variants:
    - **conservative:** `fee_per_contract`, rounded up per contract as in the live edge math.
      This variant decides the verdicts.
    - **realistic:** 0.07·P·(1−P), with no rounding.
    - **pre-fee:** fee = 0. Kalshi's fee schedule before 2025 is not verified, so this variant
      keeps a fee-regime change from deciding the result.
- **Clustering:** one cluster = one forecast_date across all cities. Compute every ROI CI with
  `python.analytics.edge_audit.day_clustered_bootstrap_ci(..., b=4000)`.

## Samples

- **OOS:** ladder-era events with forecast_date < 2026-03-18. g(mid) was fit on snapshot data
  starting 2026-03-18 (fit date 2026-07-04). The earlier favorite-longshot studies used only 2026
  data, so the pre-2026-03-18 period has never been fitted on.
- **In-sample check:** forecast_date 2026-03-18 → 2026-07-31. Report it separately. Never pool it
  with OOS.
- **Years:** group by the calendar year of forecast_date. A year counts as "full" only if its
  ladder-era coverage spans at least 9 months.

## Q1 decision table (conservative fee variant)

- **INCONCLUSIVE (checked first):** OOS has fewer than 100 entries or fewer than 60 distinct
  forecast days.
- **PASS:** both of these hold:
  - the pooled OOS ROI has a day-clustered 95% lower bound > 0
  - the point ROI is > 0 in each of the last two full OOS years. If only one full year exists, it
    must be > 0, and the partial years are reported but not gated.
- **DECAYING:** the pooled OOS lower bound is > 0, but either the most recent full year's point
  ROI is ≤ 0, or the ROI falls strictly across 3 or more consecutive full years. Meaning: do not
  size up, whatever the pooled number says.
- **FAIL:** pooled OOS point ROI ≤ 0. Meaning: treat the 2026 live result as regime or selection
  luck, and do not size up.
- **INCONCLUSIVE:** anything else.
- **Sensitivity (report it; it does not decide the verdict).** Does the result change with:
  - the realistic or pre-fee variants
  - band [0.55, 0.75)
  - 1-minute entry instead of hourly, on the entered-market subset that has minute data?

## Capacity (Q1)

- For each entered market, sum the contracts from tape trades on the verified "taker bought YES"
  side.
- Count only trades between the entry time and the end of the lead-1 window, at yes prices where
  g(mid_at_entry) − price − fee ≥ 0.01.
- Report, per forecast_date across all cities and by year: the median, p25 and p75 of contracts
  and of dollars.
- This is an upper bound on what we could have bought without moving the price.

## Q2 maker-entry rules

- At the entry time, place a simulated resting YES bid at the entry bar's yes_bid. Maker fee = 0.
- For each horizon T in {10 min, 60 min, 240 min, end of lead-1 window}. "End of lead-1 window"
  means 23:55 UTC on the lead-1 day. Live may trade at any time in that UTC day, but a cross
  after 23:55 is too close to the switch to lead-0. Flag this horizon as thin-book in the report.
  - **Optimistic fill:** some trade where a taker sells YES into the bids (the verified side)
    prints at yes_price ≤ our bid within T.
  - **Pessimistic fill:** some such trade prints at yes_price < our bid. The level was cleared
    through, so our place in the queue does not matter.
  - Report, under both fill definitions:
    - the fill rate
    - win rate and ROI on filled signals, and the same on unfilled signals
    - the adverse-selection gap = win rate(filled) − win rate(all signals), with a day-clustered
      CI.
- **Hybrid policy H(T):**
  - Rest at the bid for T.
  - If it does not fill (pessimistic definition), cross at the ask observed at T, from the 1-minute
    candles, as a taker paying the taker fee. Do this only if g(mid_T) − ask_T − fee ≥ 0.01, where
    mid_T is the mid at T, as live would recompute it. Otherwise skip the signal.
  - Report ROI per signal and per dollar wagered, against the pure-taker baseline, by year, with
    day-clustered CIs.
- **Maker fee sensitivity:** rerun H(T) with the maker multiplier at 0.0175. The maker-fee regime
  for these series before 2025 is unknown, so the zero-fee result is the optimistic case for old
  years.
- **Q2 is promising only if,** for some T, H(T) ROI minus pure-taker ROI has a day-clustered
  lower bound > 0 on OOS, AND its point estimate is > 0 in the most recent full year alone.


## Clarifications (2026-10-01, written before the pilot and before any result)

These resolve ambiguities in the rules above. They are fixed before any data beyond the Step 0 probes
was replayed.

1. **Last hourly bar.** The bar whose close is 00:00 UTC on forecast_date covers 23:00–24:00 on the
   lead-1 day; its close is the last lead-1 quote. It is inside the lead-1 window. Bar-close grid times
   run from the first full hour after `max(open_time, lead-1 start)` through that 00:00 close. Hours with
   no bar are forward-filled from the last bar; before the first bar there is no quote.
2. **DECAYING vs PASS.** When the pooled OOS lower bound is > 0, the DECAYING conditions are checked
   before PASS. "Do not size up, whatever the pooled number says" means a decay finding is never
   overridden by PASS.
3. **Event-cap tie-break.** Several markets qualifying on the same bar are ordered by edge desc, then
   g desc, then ticker asc, matching the live per-day cap sort (`effective_edge`, `confidence = g`,
   ticker).
4. **Q2 horizons are clipped** to 23:55 UTC on the lead-1 day. If the clipped horizon ends at or before
   the entry time (an entry on the last bar), the signal cannot rest, and H(T) crosses at the entry
   bar's ask, the same as the pure taker.
5. **Q2 cross check.** At T, the hybrid crosses only if the 1-minute quote at T is a valid two-sided
   quote (`0 < bid < ask < 1`) and `g(mid_T) − ask_T − taker fee ≥ 0.01`. Band and volume gates are
   not re-applied at T.
6. **ROI definitions.** ROI per dollar = Σ P&L / Σ entry price. "ROI per signal" = Σ P&L / number of
   signals (skipped signals count with P&L 0). The Q2 gate "H(T) ROI minus pure-taker ROI" uses the
   per-signal difference, with the CI from `day_clustered_bootstrap_ci` on per-signal P&L differences
   (wagered = 1 per signal).
7. **Adverse-selection gap CI.** `day_clustered_bootstrap_ci` only computes a ratio of sums, so the gap
   win(filled) − win(all) uses a separate day-clustered bootstrap with the same resampling unit
   (forecast_date) and B = 4000.
8. **Full year.** A year is full when the ladder events in the sample (all events, not entries) cover at
   least 9 distinct calendar months of that year.
9. **Maker fee sensitivity** uses `fee_per_contract(price, is_maker=True)` with no ticker, which applies
   the 0.0175 multiplier with the per-contract ceil.
10. **Step 4 exclusion.** All events whose lead-1 UTC day is 2026-09-06 (forecast_date 2026-09-07) are
    excluded from the agreement statistics, because their lead-1 scans span the 18:17–18:58Z
    deploy gap.

## Deviations

None at pre-registration.
