# New-Data Edge Audit — Can the Recently Collected Corpora Make Money?

Date: 2026-07-19
Scripts: `scratchpad/{asos_reliability,asos_vs_market,asos_roi,asos_confound,newcity_pooled,market_reliability,fb_power}.py`
Seed: 20260719 (all bootstraps day-clustered)

## The Question

Two corpora have been accruing since early June with an explicit promise of
future edge: the `intraday_asos_high` forward shadow corpus (the designated
"intraday frontier") and the 11 new-city collection corpus (Phase B). Can either
enable a new strategy or improve an existing one — provably, net of fees and
spread?

**Answer: no. Neither supports expansion. Both support a subtraction.** A third
finding is that the one strategy we believe works is not yet statistically
distinguishable from zero.

## Corpus Inventory (as of 2026-07-19)

| Corpus | Span | Size | Status |
|---|---|---|---|
| `intraday_asos_high` predictions | 2026-06-06 .. 07-19 (44 days) | 558,305 rows → 12,564 settled (ticker,hour) obs | shadow-only |
| New-city lead-1 in-band settled | 35 distinct settled days | 69 obs across 11 cities | collection-only |
| `favorite_bias` live settled trades | 47 distinct days | 162 trades | **live** |
| NBM forecasts | 2026-03-27 .. (452 runs) | 20 cities | unused (NULL verdict 2026-05-30) |

Snapshot collection is **healthy** — the local checkout was merely 22 commits
behind `origin/main`. No stall.

## Finding 1 — `intraday_asos_high` has no edge. Kill the live flip.

The pre-registered live-flip gate was a clean forward per-bin reliability table,
especially the low-p NO bin. It is not clean; it reproduces the prototype failure
at scale.

**Reliability (all 12,564 settled obs):** the raw posterior is severely
overconfident in *both* tails.

| posterior bin | n | predicted | realized | gap |
|---|---|---|---|---|
| [0.02,0.05) | 1392 | 0.035 | **0.161** | +0.126 |
| [0.05,0.10) | 1941 | 0.075 | **0.196** | +0.121 |
| [0.65,0.80) | 293 | 0.725 | 0.386 | −0.339 |
| [0.80,0.90) | 168 | 0.850 | 0.435 | −0.415 |

Per-cell mean gaps are consistent in sign (AUS +0.025, CHI +0.026, NYC +0.027,
MIA +0.055, LAX +0.023) — a systematic model defect, not noise.

**The decisive test (the NBM test): is a calibrated posterior sharper than the
market?** Isotonic fit on the earlier day-block, scored on the later one.

| | Brier (test, n=6,209) |
|---|---|
| raw ASOS posterior | 0.1834 |
| **climatology (base rate)** | **0.1806** |
| calibrated ASOS (OOS) | 0.1697 |
| **market price** | **0.1185** |

The raw posterior is **worse than climatology**. Calibration helps but the market
remains far sharper: `Brier(market) − Brier(calASOS) = −0.0511`, day-clustered 95%
CI `[−0.0630, −0.0382]`. The CI excludes zero **in every one of the five cells**.

**Realizable P&L confirms it.** An initial simulation showed +12.8% ROI — that was
a methodological error on my part: it selected the *max-edge hour* per market,
which production cannot do. The realizable rule (first qualifying signal) is
negative and degrades with spread:

| half-spread | n | win% | ROI | day-clustered 95% CI |
|---|---|---|---|---|
| 1¢ | 494 | 24.7% | −5.7% | [−15.0%, +4.1%] |
| 2¢ | 489 | — | −10.5% | [−20.5%, −0.3%] |
| 3¢ | 486 | — | −11.8% | [−20.5%, −2.9%] |
| 5¢ | 466 | — | −25.0% | [−34.7%, −15.1%] |

Mechanism: trades costing ≤0.10 (n=233, ~half the book) win **3.0%** of the time
→ **−53.4% ROI**. The calibrator lifts low raw-p to ~0.16, manufacturing "edge" on
contracts that genuinely settle YES ~2–6% of the time. This is the same adverse
selection that killed NBM-gap: gap-selection trades hardest exactly where the
inferior model most disagrees with a sharper price.

**Verdict: NULL.** Do not flip `intraday_asos_high` live. Park it.

## Finding 2 — The favorite-longshot edge does NOT transfer to the new cities.

Scored against the **frozen** original-9 `g(mid)` (no refit → no circularity, per
spec §7).

| cohort | n | mean g(mid) | realized | gap | day-clustered 95% CI |
|---|---|---|---|---|---|
| original 9 (reference) | 192 | 0.647 | 0.677 | +0.030 | [−0.036, +0.091] |
| **11 new cities (OOS)** | **69** | **0.691** | **0.551** | **−0.140** | **[−0.265, −0.022]** |

The original 9 are healthy (CI spans zero). The new cities are **disconfirming**:
the map prices them at 0.691 but they settle YES 55% of the time. Expanding
`favorite_bias` to new cities — the intuitive "double the universe" move — would
have lost money. The pre-registered expected-fail cities behave as predicted
(PHX −0.252, LV −0.167; DAL is worst at −0.341).

**Caveat, stated plainly:** n=69 pooled is modest, and this pooled test is *not*
the pre-registered per-city gate (≥30/city, which no city meets — PHX leads at 13,
ETA ~45 days; most are 70–490 days out). It is an early read, and it points the
cautious direction, which is the safe direction to over-weight. It does not by
itself justify permanently excluding any city.

**Throughput note (the real prize, still unclaimed):** new cities would supply
1.97 eligible signals/day vs the original 9's 1.98/day — a genuine ~2× of the
universe. That is why this matters, and why it should stay in collection rather
than be abandoned.

## Finding 3 — The intraday book is efficient; no model-free edge either.

Testing the market against settlement directly (no forecast model at all), the
same logic that makes `favorite_bias` work on the day-ahead book:

| market price bin (test) | n | mid | realized |
|---|---|---|---|
| [0.00,0.02) | 1073 | 0.010 | 0.002 |
| [0.02,0.05) | 1022 | 0.035 | 0.021 |
| [0.05,0.10) | 751 | 0.075 | 0.060 |

Cheap contracts are overpriced by only ~1–1.5pp. Buying NO wins 96–98% of the
time but ROI is **−1% to −3%** at every threshold, in **both** train and test, and
in all five cells — the mispricing is smaller than fees + spread.

## Finding 4 — `favorite_bias` is not yet proven.

The one live edge, re-tested with day-clustering (trades sharing a forecast day
share a weather draw — the naive per-trade CI overstates precision):

```
n=162 trades over 47 distinct days
ROI = +8.27%
day-clustered 95% CI: [-7.37%, +24.49%]     bootstrap P(ROI <= 0) = 0.152
```

Suggestive, **not significant**. This is quantitative backing for the 2026-07-03
sizing rollback and for the standing analyst HOLD: **no size-up is justified until
CI_lo > 0.** At current throughput (~2 eligible signals/day) that needs roughly
3–4× the current trade count — on the order of six months — *unless* throughput
rises. New cities were the throughput answer, and Finding 2 blocks them. That
tension is the central strategic problem, and it is not solvable by tuning.

## Finding 5 — The throughput lever: lead-2 is impossible; lead-0 is the only candidate.

Findings 1–3 close the *forecasting* paths, leaving `favorite_bias`'s signal
starvation (~2/day) as the binding constraint. The untested lever is more
signals on the **nine cities whose edge is already validated** — sidestepping the
new-city transfer failure entirely.

**Lead-2/3 do not exist.** Kalshi lists these daily-temperature markets only ~1
day ahead: 0 markets at lead ≥2 across the entire 123-day corpus. Not a liquidity
problem — the instrument isn't listed. This lever is structurally unavailable.

**Lead-0 (same-day) is materially better-supplied than lead-1:**

| lead | in-band markets @ vol≥200 | median volume |
|---|---|---|
| 0 | 219 | **1364** |
| 1 | 106 | 112 |

~2× the qualifying signals at ~12× the median volume, on a book `favorite_bias`
does not trade. Scored against the **frozen** lead-1 `g(mid)` (genuinely OOS),
the aggregate looks strong: n=218 over 104 days, gap +0.036 (CI [−0.020,+0.094]),
simulated ROI **+14.8%, CI [+4.6%, +25.2%]** — and stable across a time split
(+18.2% / +11.8%).

**But it does not survive the tradeability split.** The split uses the
peak/trough convention from `same_day_gate.PEAK_LOCAL_TIME` (15:00 local for
high, 06:00 for low) — bets placed after the day's extreme has already occurred
are a materially different proposition (outcome near-determined, worse adverse
selection), so they must be scored separately.

> **Correction (2026-07-20):** an earlier revision described this as "the window
> `same_day_gate` blocks." That is wrong. `same_day_trading.enabled: false` and
> `allowed_cities: []`, so the gate runs as a `disabled()` sentinel that blocks
> **all** same-day markets unconditionally and never reaches its peak-time logic
> or `CITY_TIMEZONE`. The cutoff above is a borrowed *analytical convention*, not
> an active production constraint. Two consequences: (a) `CITY_TIMEZONE` is dead
> code — its missing PHIL/BOS entries should be handled in the analysis script,
> **not** by editing shipped code; (b) that gate belongs to `weather_edge`, so a
> lead-0 `favorite_bias` would **not** inherit it. What should gate a same-day
> favorite_bias is an open design question, not a config flip.

Splitting lead-0 at that cutoff:

| window | n | gap | win% | ROI | day-clustered 95% CI |
|---|---|---|---|---|---|
| **PRE-peak (gate ALLOWS — tradeable)** | 62 | −0.002 | 66.1% | +11.7% | **[−8.2%, +30.7%]** |
| POST-peak (gate BLOCKS) | 82 | +0.049 | 68.3% | +15.6% | [−2.1%, +33.8%] |

Neither CI excludes zero. The aggregate's apparent significance came
substantially from the post-peak window we deliberately do not trade. (54 obs
were excluded for lacking a `CITY_TIMEZONE` entry — PHIL/BOS are not in the
gate's map, itself worth noting.)

### Calibrating my own simulator — a ~3.3× optimism factor

The one place this simulation can be ground-truthed against reality is lead-1,
which `favorite_bias` actually trades:

| | ROI |
|---|---|
| simulated (buy YES at posted ask, real fees) | **+27.2%** |
| **live realized** (n=163) | **+8.3%** |

The snapshot simulator overstates by ~3.3×, because it fills at a posted ask that
live execution does not reliably get. **Any simulated ROI in this document should
be haircut accordingly.** Applying it to the pre-peak +11.7% gives ~+3.5% — well
inside noise, and plausibly below fees. This factor should be applied to any
future snapshot-based ROI claim before it is used to justify capital.

**Verdict: promising, unproven.** Lead-0 pre-peak is the only positive,
mechanism-backed throughput lever found, and it is a **shadow-test candidate, not
a live change**. Pre-registered gate before any live flip: ≥120 pre-peak
tradeable observations across ≥40 forecast-days with a day-clustered ROI CI whose
lower bound clears zero *after* the 3.3× haircut, plus per-cell non-concentration
(CHI/low is −31.4% at n=21 and must not be carried by a single cell).

## Recommended Actions

1. **Park `intraday_asos_high`** (`enabled: false`) — same disposition as
   `intraday_emos_high`. It is provably negative, and parking removes an
   IEM/ASOS API dependency and 558k rows/44 days of log volume. Remove its
   `risk.strategy_allocations` entry in the same change (the 2026-06-07
   crash-loop lesson).
2. **Do not run Phase B.** Record Finding 2 in the new-cities spec. Keep
   collecting — the per-city gate remains the decision rule, and the pooled read
   should be re-run as n grows.
3. **Hold `favorite_bias` sizing flat.** Finding 4 is the justification.
4. **Do not pursue** NBM-gap (NULL, 2026-05-30), ASOS nowcast (NULL, here), or
   intraday model-free (NULL, here). Three independent confirmations that
   out-forecasting this market does not pay.

5. **Shadow-test lead-0 pre-peak `favorite_bias`** (Finding 5) — the single
   remaining candidate. It has a real mechanism (2× signals, 12× liquidity, on
   already-validated cities), it is OOS-positive but not yet significant, and it
   costs nothing to measure in shadow. Do **not** flip it live on current
   evidence; instrument it and let the pre-registered gate decide.

## The Honest Summary

The recently collected data does not enable a new strategy *today*. Its value is
that it **cheaply falsified the expansion paths we were counting on** — before any
took live capital — quantified how thin the one surviving edge actually is, and
narrowed the search to a single testable candidate.

The chain is now explicit: out-forecasting this market does not pay (three
independent nulls: NBM-gap, ASOS nowcast, intraday model-free), so the only edge
we have is the behavioral one, and its binding constraint is **signal starvation**
(~2/day). More cities fails on edge transfer (Finding 2); more lead-days is
structurally impossible (Finding 5). That leaves the same-day book — better
supplied and far more liquid — as the one lever with a real mechanism, pending a
shadow test.

The most transferable result may be the **3.3× simulator-optimism factor**: every
snapshot-based ROI estimate this project has produced, including the encouraging
ones, needs that haircut before it justifies capital. Two of this session's four
promising results (the ASOS +12.8%, the lead-0 aggregate +14.8%) evaporated under
methodological scrutiny rather than under more data.
