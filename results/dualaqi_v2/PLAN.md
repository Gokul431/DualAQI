# DualAQI v2: analysis plan (written 2026-11-22, before any v2 result was computed)

## Question
Can DualAQI be improved fairly, and how does the improved model (v2) compare with
DualAQI v1 and with the single-branch LSTM?

## Data split (forward CV, launch hour 03:00 UTC, same 1,816 origins as before)
- Development: forward folds 0-7 (test origins 2020-01-08 to 2021-12-27). All choices are made here.
- Final test: forward folds 8-15 (test origins 2021-12-28 to 2023-12-17). Used once, at the end.
  (The v1 runs already covered these folds; their test-fold results were not used to design v2.)

## Candidates (all on top of DualAQI v1)
- B: v1 + early stopping
- C: B + fire radiative power (FRP): 7 daily lags of log1p(FRP sum)
- D: B + transport: 7 daily means of the north-west -> south-east wind component at Delhi (ERA5),
     and log1p(fire count) x max(transport, 0) for the same 7 days
- E: B + long memory: log1p of the 14-day and 30-day fire-count sums ending at the last usable day
- F: B + C + D + E combined

Early stopping (applied identically to DualAQI and to the single-branch LSTM comparator):
last 15% of each fold's training origins (in time) are validation, with a 3-day gap before them;
up to 40 epochs, patience 5, best validation weights restored. All other settings unchanged
(Adam 1e-3, batch 128, clip 1.0, loss weights 0.25/0.35/0.40).

Causality: every new input for fire day d (FRP, transport, sums) uses only data from day d or
earlier and is used only from 03:00 UTC on day d+1, the same rule as v1. Tested before running.

## Selection rule (fixed in advance)
v2 = the candidate among B, C, D, E, F with the lowest development RMSE, averaged over the three
horizons and seeds 0, 1, 2. Ties (difference < 0.1 AQI) go to the simpler candidate.

## Final test (run once)
On folds 8-15, seeds 0, 1, 2:
1. DualAQI v2 vs DualAQI v1
2. DualAQI v2 vs single-branch LSTM with early stopping (the fair comparator)
3. For reference: DualAQI v1 vs single-branch LSTM v1 on the same folds
Paired dRMSE with 95% CIs (day bootstrap and 7-day block bootstrap, 500 resamples), overall and
Oct-Nov vs rest of the year, plus mean +- SD of RMSE, MAE, R2, bias, CatAcc, POD/FAR (Severe).
An improvement is claimed only where the CI excludes zero. Every candidate's development result
is reported, including those not selected.

## Amendment 1 (2026-11-22, after the development stage, before any test-fold run)
Development result: early stopping lowered accuracy for both models (DualAQI 87.8 -> 93.7,
single-branch LSTM 89.8 -> 98.9 mean RMSE over horizons, folds 0-7). It holds back the most recent
15% of each fold's training data, which is the most informative part in forward CV, and often
stopped near epoch 6. The original rule would therefore select a "v2" (D, 92.9) worse than v1.
Change: early stopping is dropped for both models (the v1 training setup, 15 epochs, is kept for
DualAQI and for the single-branch LSTM comparator). Candidates C', D', E', F' = v1 + the same
feature additions as C, D, E, F, without early stopping, are run on folds 0-7, seeds 0-2.
v2 = the lowest development RMSE among v1, C', D', E', F' (same tie rule; v1 is simplest).
The final test (folds 8-15) is unchanged except that the comparator is the single-branch LSTM v1.
Caveat: the earlier v1 report covered all forward folds, so v1 and LSTM v1 results on the test
years were already seen; no v2 design choice uses them.

## Amendment 2: fresh test on 2024-2025 (written 2026-11-22, before any 2024-2025 data was built)
Reason: folds 8-15 (2022-2023) have been used once. A new, untouched test period is added.
- Data extended to 2025-12-31 with the same code and sources (CPCB folder, Open-Meteo ERA5,
  FIRMS VIIRS S-NPP). 2019-2023 inputs must stay byte-identical (checked).
- Forward folds keep the same 90-day blocks; the new folds after 2023-12-17 (2024-2025) are the
  test set and are run once. Models are fixed: DualAQI v2 (v1 + FRP, 15 epochs), DualAQI v1, and
  the single-branch LSTM (same training setup), seeds 0, 1, 2. No changes after seeing results.
- Primary question: is DualAQI v2's RMSE lower than the LSTM's? Paired dRMSE (v2 - LSTM) per
  horizon with 95% CIs (day bootstrap and 7-day block bootstrap). DualAQI v2 is declared better at
  a horizon only if both CIs are entirely below zero. Secondary: overall vs Oct-Nov vs rest of
  year; MAE, R2, bias, CatAcc, Severe POD/FAR; v2 - v1; v1 - LSTM.
- Whatever the result, it is reported as is.

## Amendment 3 (2026-11-22, data check, before any 2024-2025 model run)
Open-Meteo has no ERA5 boundary-layer height for 2024-01-01 to 2024-06-30 (both `era5` and
`era5_seamless`; a re-download gives the same gap). No other ERA5 source is available (no CDS access)
and values are not filled from another product. Origins needing those hours are dropped for every
model alike, so the fresh test covers July 2024 to December 2025 (two Oct-Nov burning seasons).
Test folds: every forward fold after fold 15 that keeps >= 30 test origins.
