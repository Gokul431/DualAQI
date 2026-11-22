# DualAQI-Net benchmark, CPU-only

Retrospective multi-horizon (24/48/72 h) PM2.5 / AQI forecasting for Delhi, 2019–2023.
Runs on a 16 GB laptop CPU. No GPU needed.

## Files

| File | Purpose |
|---|---|
| `dualaqi.py` | Full pipeline: causal features, Persistence / Ridge / GBM / RandomForest / Ensemble, LSTM, DualAQI-Net, blocked and forward CV, metrics, bootstrap CIs |
| `make_synthetic.py` | Writes fake inputs in the correct schema, for smoke-testing only |
| `requirements.txt` | Dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, for LSTM and DualAQI-Net
```

## Smoke test (no real data needed)

```bash
python make_synthetic.py
python dualaqi.py --data-dir data --out-dir results_test --no-nn   # classical models, ~2 min
python dualaqi.py --data-dir data --out-dir results_test           # adds LSTM and DualAQI-Net
```

## Real data: building the inputs

Use the Python 3.12 environment in `.venv` (`.venv\Scripts\python`). On this machine the system
Python 3.15 has no torch/h5py wheels, and Windows Application Control may block a newly installed
DLL on its first load; re-running the same command clears it.

```bash
.venv\Scripts\python build_inputs.py --pm25 --era5     # data/pm25_hourly.csv, data/era5_hourly.csv
.venv\Scripts\python sanity_checks.py                  # coverage, repeated values, pollutant leakage
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results --launch-hours 3 --n-boot 500
```

- **PM2.5** comes from the CPCB station folder `Delhi PM2.5 Air Quality/` (36 Delhi stations,
  CPCB/DPCC/IITM). Its timestamps are IST and mark the start of the hour; `build_inputs.py` relabels
  each value to the UTC hour at which it becomes available (stamp − 4 h).
- **ERA5** comes from the Open-Meteo archive API (`models=era5`), so no Copernicus CDS account is
  needed. It is the same grid cell and values as the CDS files in `raw/era5/` (checked on 9,504
  overlapping hours of 2023). Responses are cached in `raw/era5_openmeteo/`.
- **MODIS AOD** needs a NASA Earthdata login: `download_modis.py`. Building `aod_daily.csv` and
  `aod_tiles.npz` from the swaths is not written yet. Without those files the AOD features are left
  out and DualAQI-Net, CNN and LateFusion are skipped.
- The files built this way are in **UTC**, so `--launch-hours 3` = 03:00 UTC = 08:30 IST, and the
  calendar features (hour, day of year, weekday, festival day) are computed on UTC time.

## Models

Persistence, Ridge, GBM (HistGradientBoosting), RandomForest, XGBoost, Ensemble (mean of Ridge, GBM,
RandomForest), ARIMA (regression of log PM2.5 on daily and annual Fourier terms with ARMA errors;
order chosen per fold by AIC from (1,0,0), (2,0,1), (2,0,2) on training windows only; each origin is
forecast from its last 168 h, converted with the lognormal mean exp(mu + var/2)), LSTM.

After the runs:

```bash
# seeds 1 and 2 (ARIMA does not use a seed, so it is skipped there)
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results/seed1 --launch-hours 3 --n-boot 500 --seed 1 --no-arima
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results/seed2 --launch-hours 3 --n-boot 500 --seed 2 --no-arima
.venv\Scripts\python seed_summary.py    # results/seed_summary.csv: mean +- SD over seeds 0-2
.venv\Scripts\python case_study.py      # results/case_study/: most polluted 30-day episode, figure + tables
``` With AOD tiles also: CNN (tiles only), LateFusion
(LSTM and CNN predictions mixed by a learned per-horizon weight) and DualAQI-Net (gated fusion).

## DualAQI (sequence + Punjab fire branch) and its ablation

```bash
set FIRMS_MAP_KEY=...                                   # free key from firms.modaps.eosdis.nasa.gov/api/map_key/
.venv\Scripts\python download_firms.py                  # VIIRS S-NPP fires, Punjab box, 2019-2023 -> raw/firms
.venv\Scripts\python build_inputs.py --fire             # Punjab polygon (GADM 4.1), type 0 -> data/punjab_fire_daily.csv
.venv\Scripts\python test_causality.py                  # fire day d used only from d+1 03:00 UTC
# per seed s in 0,1,2 (forward CV only, baselines not re-run):
.venv\Scripts\python dualaqi.py --only-dualaqi --schemes forward --launch-hours 3 --n-boot 500 --seed s --out-dir results/dualaqi/fire/seeds
.venv\Scripts\python dualaqi.py ... --no-fire        --out-dir results/dualaqi/nofire/seeds     # single-branch ablation
.venv\Scripts\python dualaqi.py ... --calendar-only  --out-dir results/dualaqi/calendar/seeds   # event branch without fires
.venv\Scripts\python dualaqi_ablation.py --n-boot 500   # tables, paired dRMSE CIs by season, 2023 case study
```

Event branch: log1p fire counts for the 7 most recent available days plus calendar (hour, weekday,
month as sin/cos; festival flags for the origin and target days) -> MLP -> 32-d. The 32-d vector is
projected to 64-d so the gate g = sigmoid(W[h; e]) can mix it with the LSTM state.
`--no-fire` drops the whole event branch (fire and calendar); `--calendar-only` keeps the branch
without fire counts, which separates the effect of the fires from that of the calendar inputs.

### DualAQI v2 (development/test protocol in results/dualaqi_v2/PLAN.md)

v2 = v1 + 7 daily lags of log1p(fire radiative power), selected on forward folds 0-7 (2020-2021)
and tested once on folds 8-15 (2022-2023). Options: `--event-opts frp|transport|long`,
`--early-stop`, `--fold-range a:b`, `--model-name`. Fire file built with
`build_inputs.py --fire --fire-start 2018-12-01` (adds `frp_sum`). Comparison:
`dualaqi_v2_compare.py --stage dev` then `--stage test`.

## Input schema

Place these in `--data-dir`. Timestamps are UTC when built with `build_inputs.py` (the original
schema below described IST; either works as long as all files share one timezone).

1. **`pm25_hourly.csv`**: columns `datetime, pm25`. One row per hour. `pm25` is the city-level value, the mean of station PM2.5 (µg/m³) computed *before* AQI conversion. Leave gaps as empty cells; they are not imputed for targets.
2. **`era5_hourly.csv`**: columns `datetime, t2m, rh, u10, v10, sp, blh`. Hourly ERA5 at the city point or the station mean. Use the ERA5 archive for retrospective work. ERA5 is not available in real time (see Limitations).
3. **`aod_daily.csv`**: columns `date, aod_terra, aod_aqua`. Daily AOD summary, averaged over the 200 km box around Connaught Place from MODIS MOD04_3K (Terra) and MYD04_3K (Aqua), QA = 3. Leave missing days empty.
4. **`aod_tiles.npz`** (optional, enables DualAQI-Net): `dates` (YYYY-MM-DD strings) and `tiles` with shape `(n_days, 2, 16, 16)`, float32. Channel 0 = Terra, channel 1 = Aqua. Downsample the 67×67 MODIS tile to 16×16 by area averaging. Missing days are omitted from `dates`.

Getting the raw data (for reference):
- PM2.5: OpenAQ v3 API (requires an API key) or CPCB CAAQMS downloads.
- ERA5: Copernicus CDS via `cdsapi` (requires a CDS account and key).
- MODIS AOD: NASA Earthdata via `earthaccess` (requires an Earthdata login).

## Run

```bash
python dualaqi.py --data-dir data --out-dir results --launch-hours 8
```

`--launch-hours` takes comma-separated IST hours. Default is one origin per day at 08:00 IST (1,826 origins, which keeps CPU time low). Use `2,8,14,20` for 6-hourly origins, at a higher cost.

Outputs in `--out-dir`:
- `benchmark_long.csv`: same columns as the earlier benchmark (RMSE, MAE, R², bias, category accuracy, skill vs Persistence, 95% CI for the RMSE difference, per-fold R² mean and SD).
- `oof_predictions.csv.gz`: every out-of-fold prediction, with origin, fold, true AQI, and predicted AQI. Use this for the case study and seasonal analysis.

## Causality rules (check these before reporting)

- Forecast origin t is a launch time. Only data at or before t is used.
- PM2.5 features are forward-filled at most 3 h. Targets are never filled.
- AOD for origin day D comes only from days before D (`shift(1)`), with a 3-day staleness limit. Missing values are filled with the training median inside each CV fold.
- ERA5 features come only from times at or before t.
- Blocked CV: training origins must not overlap the test block's input window or target window (embargo = 168 h input + 72 h horizon).
- Forward CV: expanding window, 90-day test blocks. Trailing blocks with fewer than 30 origins are dropped.

## Things to verify before reporting

- **CPCB breakpoints** in `AQI_KNOTS_C` / `AQI_KNOTS_I`: confirm against the official table. These are my reconstruction.
- **Festival dates**: only Diwali is filled. Add Holi and Eid dates from an official calendar, or the festival feature is incomplete.
- **Tile downsampling**: the 67×67 to 16×16 step should be area averaging, and the result should be checked visually for one day.
- **Seeds**: the neural models use one seed (`--seed`). Rerun with 3 seeds and report mean ± SD before publishing.

## Known simplifications

- The neural models use a fixed 15 epochs with no early stopping. Add a validation split for early stopping if the training loss keeps falling sharply.
- Hyperparameters for GBM, RandomForest, and the networks are fixed, not tuned. Tuning should use only training data.
- The DualAQI-Net fusion is a learned gate between the LSTM state and the flattened CNN features. There is no spatial attention. This matches the design choice of replacing cross-modal attention with gated fusion.
- The synthetic data has no real signal from AOD, so smoke-test results say nothing about model quality.

## Limitations to state in the paper

- This is a retrospective study. ERA5 is not available in real time, so the pipeline cannot run operationally as written. An operational version would need an analysis or forecast product (for example, ECMWF operational fields) and retraining.
- The city-level target averages stations before AQI conversion. The spatial structure of the satellite tiles is therefore not matched by a spatial target.
- One test year (2023) for the earlier results; the forward CV covers 2020–2023 in expanding windows.
