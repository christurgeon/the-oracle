# Stage 1b — centering test (the binding gate): built, validated, and a confound exposed

**Date:** 2026-06-09
**Script:** `scripts/research_2026_06_09_centering_test.py`
**Artifact:** `data/research/2026-06-09-centering-test.json`
**Status:** Instrument **BUILT + VALIDATED**; new-city verdict **UNDERPOWERED**
(1 day of overlap). A mature-control confound was discovered that **sharpens
the gate's role** for the rest of the program.

## What the centering test does

For each new-city same-day HIGH market with an active mid, it computes the
observation-anchored posterior — exactly as the live `intraday_asos_high`
strategy would (`ClimatologyTable.lookup` on the **draft** new-city
climatology + LST-frame `M` from ASOS + `bracket_probability`, truncated
Student-t ν=5, NWS rounding) — and measures `deviation = posterior − market_mid`
plus the executable taker edges (`posterior − ask − fee`; `bid − posterior −
fee`). The question (Opus finding 2): does the market *midpoint* sit off our
posterior by more than executable cost? Width alone is anti-taker and is not a
green light.

## The control that mattered: posterior-vs-mid on known-efficient mature books

Because the draft posterior is itself untested, a raw new-city deviation could
be *our model* being wrong rather than the *market* being wrong. So the test
runs the identical computation on the 5 **mature** cities with the
**production** climatology — books we proved efficient on 2026-06-09
(`docs/research/2026-06-09-intraday-lockin-falsification.md`). Result (post-peak, n=280 over
64 days):

| mature-control metric | value | reading |
|---|---|---|
| mean(posterior − mid) | **−0.070** | our posterior runs ~7 pp below the efficient mid (model bias) |
| mean\|posterior − mid\| | **0.243** | our posterior disagrees with truth by ~24 pp on average |
| frac markets flagging executable edge | **0.94** | our posterior "finds" a taker edge on 94% of *efficient* markets |
| median "best edge" | **0.158** | …of ~16 pp — and we know trading it LOSES |

**This is the headline finding.** On books with *no* harvestable edge, the raw
deviation / "executable edge" signals fire almost everywhere. The apparent
edge is dominated by **our posterior's own miscalibration**, not by market
mispricing. Therefore:

- The raw centering signal is **not** a usable gate on its own.
- The only informative quantity is the **new-city deviation NET of the mature
  baseline** (and net of its ~0.24 noise floor), accrued over many independent
  days.
- Even a clean net signal is merely the **NECESSARY** condition — the binding
  edge adjudicator remains **V1 forward settled P&L** (and V2 per-bin
  reliability), not this screen. This *confirms* the spec's sequencing (V1 is
  the real gate) and now quantifies *why* centering cannot be promoted to a
  sufficient test.

## New-city read today: UNDERPOWERED (as the spec predicted)

New-city markets entered the snapshot corpus only on 2026-06-09, and the
2-year ASOS backfill ended 2026-06-08; the single overlapping day yields **10
post-peak evals**. The script's gate requires ≥60 evals over ≥14 days before it
will render any screen, so today's verdict is a hard **UNDERPOWERED — no screen
possible yet**. (Raw numbers for the record: new-city post-peak
mean-deviation −0.19, net of the −0.07 mature baseline = −0.12; mean-abs
0.40 — i.e. *noisier* than the mature 0.24 floor on n=10, fully consistent
with thin data, not a centering signal.)

## Implications for the program

1. **The gate is built and parity-correct** — it will produce a real screen as
   the forward corpus accrues (snapshots + a periodic ASOS refresh).
2. **Centering is demoted from "the binding gate" to "a coarse necessary
   pre-screen."** The mature confound means a positive centering result cannot
   on its own justify Stage 2; it can only *fail fast* (if new ≈ mature, stop)
   or *permit continuation* to V1. The spec's §Stage-1 should read centering and
   V1 as: centering can NULL the lane cheaply, but only V1's forward settled
   P&L can GREEN it.
3. **Operational need:** the new-city ASOS CSVs must be kept current (a periodic
   refresh, same cadence as the mature `research_2026_05_21_refresh_asos.py`)
   so the centering + V1 runs have same-day `M`. Today's run required a manual
   one-day top-up.

## Reproduce

`uv run python scripts/research_2026_06_09_centering_test.py` (reads committed
draft climatology + snapshots; needs the new-city ASOS CSVs in
`data/research/asos/`, which are gitignored — refresh from IEM first).
