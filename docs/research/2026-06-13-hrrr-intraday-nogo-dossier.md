# HRRR-into-intraday — feasibility investigation (NO-GO)

**Date:** 2026-06-13  **Verdict:** NO-GO. Do not build the rigorous (NOAA-S3) version.
**Reproduce:** `PYTHONPATH=. uv run python scripts/research_2026_06_13_hrrr_intraday_nogo.py`

## Question

The one forward-looking survivor of the 2026-06 strategy-brainstorm review was: *condition
the `intraday_asos_high` remaining-gain on a same-day HRRR forecast.* Today that posterior's
remaining-gain `(mu_gain, sigma_gain)` comes from a **static climatology** keyed by
`(city, month, hour)` (`data/asos/climatology.json`). HRRR (NOAA's hourly 0–18h CONUS model)
is the natural tool for the intraday window — does conditioning on it improve the posterior
where it binds: the **low-p NO bin** (the strategy's known defect — predicted ~0.02 vs realized
~0.17 on the prototype cohort)?

The investigation was deliberately **look-ahead-OPTIMISTIC** (Open-Meteo's stitched historical
series, which can include model runs issued after the decision hour) — an **upper bound**. If
HRRR fails even with that help, the clean version is worse.

## Data probe (cleared)

Open-Meteo serves `ncep_hrrr_conus` hourly `temperature_2m` historically back to ≥2021, free,
covering our 2-yr ASOS window. **But** it returns only a stitched "best" series — it cannot pin
a model-run init at hourly granularity (`previous_day1/2` only). A look-ahead-clean study would
need the NOAA HRRR S3 GRIB archive with explicit fixed-run selection. We used the optimistic
series on purpose.

## Method

- **Cells:** MIA / AUS / CHI / NYC high (the `intraday_asos_high` trading cells). HIGH-only.
- **Truth:** ASOS daily max — the climatology's **native** fit target (`build_asos_climatology.py`
  uses the hourly METAR max), so the baseline is *not* handicapped by the METAR-vs-CLI ruler gap.
  CLI settled highs (`data/nws_bias/nws_actuals.csv`) used only in Stage 3.
- **Frame:** fixed-offset LST climate day (`CITY_STD_OFFSET_HOURS`), matching production (no DST bug).
- **OOS:** HRRR per-`(city,hour)` bias and the HRRR-residual σ are fit on TRAIN (`<2025-06-01`);
  everything evaluated on TEST (`≥2025-06-01`).
- **Band split:** post-peak if climo `sigma_gain < 1.2°F` else pre-peak (the production sharp-σ line).

## Stage 1 — daily-HIGH MAE screen (looked green, was a mirage)

Bias-corrected HRRR vs climatology, MAE of the daily-HIGH point forecast (°F), TEST half:

| band | climo MAE | HRRR-BC MAE | Δ (climo−BC) |
|---|---|---|---|
| pre-peak | 1.92 | 0.88 | **+1.04** |
| post-peak | 0.50 | 0.37 | **+0.13** |

Initially read as CONTINUE. An adversarial audit (backtesting-analyst) demolished it:
- **The post-peak win is a `max(M, HRRR_remaining)` floor artifact.** The floor fires ~67% of
  post-peak cells (HRRR predicts `M` = "no new high"), trivially beating a climo baseline that
  always adds `mu_gain>0`. In the 33% where HRRR actually forecasts, it **loses** to climo (−0.05°F).
  A `predict = M always` rule with no HRRR captures the same "win."
- **The pre-peak win is ~20–40% inflated** by Open-Meteo's short-lead stitching, and is in a band
  the strategy does not trade (it gates on post-peak σ).
- **MAE is the wrong metric** for the binding low-p NO bin (calibration-analyst): a better *mean*
  is decoupled from tail coverage, and a sharper-but-miscentered posterior can *empty the tail*.

## Stage 2 — binding re-cut: low-p NO-bin reliability (the right test)

Each posterior pushed through the production `asos_posterior.bracket_probability` over a faithful
tiling of synthetic KXHIGH ABOVE/BELOW markets; bin by predicted P(YES) in the realistic NO-trade
region `[0.01, 0.20)`; measure realized YES (bust) rate. TEST half.

**Post-peak band (the traded band):**

| posterior | n | pred μ | realized | gap (real−pred) |
|---|---|---|---|---|
| CLIMO (baseline) | 3559 | 0.065 | 0.058 | **−0.007** (calibrated) |
| HRRR mean + climo σ | 3750 | 0.061 | 0.054 | −0.006 (adds nothing) |
| HRRR mean + HRRR-resid σ | 2162 | 0.037 | 0.103 | **+0.066 (breaks calibration)** |

The climo low-p NO bin is **already calibrated** on the ASOS frame; HRRR's mean is redundant; and
the only way to inject HRRR as a *sharper* posterior makes it **over-confident** (a symmetric scale
can't capture the right-skewed remaining-gain tail) — manufacturing the very miscalibration we set
out to fix. Pre-peak: nobody is calibrated (HRRR-A over-corrects to −0.043; HRRR-B ≈ climo) and it
isn't traded anyway.

## Stage 3 — CLI confirmatory (the mechanism)

Climo low-p NO bin, same cells, realized rate under ASOS-max truth vs CLI settlement truth
(`n=74` CLI dates):

| band | pred μ | realized (ASOS truth) | realized (CLI truth) |
|---|---|---|---|
| post-peak | 0.060 | 0.024 (gap −0.036, calibrated) | **0.118 (gap +0.058, ~2× — busts)** |

The defect appears **only against CLI settlement truth**. → The binding low-p NO-bin miscalibration
is the **ASOS/METAR → CLI ~1°F settlement gap**, not a forecast-mean error. HRRR forecasts the same
ASOS frame that `M` lives in, so it is structurally blind to the CLI continuous-sensor extreme.

## Conclusion

HRRR is the **wrong tool**: its error source (forecast skill) is orthogonal to the defect's error
source (settlement frame). Holding σ fixed it is redundant; sharpening σ it is harmful. The
expensive Tier-2 (NOAA-S3 fixed-run) build is **not justified**.

**The real fix** — consistent with the 2026-06-07 ASOS-prior panel (unpublished) — is a **recalibrator on
the climatology's own low-p output that learns the CLI bust rate**, gated on the base
`intraday_asos_high` shadow-corpus accrual (which is already collecting posterior→CLI pairs). No new
forecast model required.

**Durable lesson:** judge intraday calibrators by **low-p NO-bin reliability under CLI-settlement
truth**, never daily-HIGH MAE or aggregate score. A point-MAE win can be a
`max(M,·)` floor artifact *and* decorative for the binding bin — both traps appeared here.

## Pointers

- Script (all three stages, reproducible): `scripts/research_2026_06_13_hrrr_intraday_nogo.py`
  (HRRR cache rebuilds to `data/research/hrrr/`, gitignored).
- Base strategy: `python/strategies/intraday_asos_high.py`, posterior math
  `python/data/asos_posterior.py`, the in-flight entry "Intraday ASOS Nowcast" in
  the unpublished tracker.
- Reviews that turned the verdict: backtesting-analyst (floor artifact, look-ahead inflation),
  calibration-analyst (MAE ≠ binding bin), in this session's transcript.
