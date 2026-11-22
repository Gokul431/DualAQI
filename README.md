# DualAQI: 24–72 h AQI forecasting for Delhi with a Punjab fire event branch

Code and results for DualAQI, a dual-branch LSTM for retrospective 24, 48 and 72 h AQI forecasting
in Delhi. A sequence branch reads 168 h of city PM2.5 and ERA5 meteorology. An event branch reads
lagged Punjab satellite fire activity (NASA FIRMS VIIRS) and calendar inputs. A learned gate fuses
the two. The model is compared with persistence, ARIMA, Ridge, gradient boosting, random forest,
XGBoost, an ensemble and a single-branch LSTM, under forward (expanding-window) cross-validation.

Everything runs on a laptop CPU; no GPU is needed.

## Files

| File | Purpose |
|---|---|
| `download_firms.py` | Downloads VIIRS S-NPP fire detections for a box around Punjab (NASA FIRMS area API) |
| `build_inputs.py` | Builds the model inputs in `data/`: city-mean PM2.5 (CPCB stations), ERA5 (Open-Meteo), daily Punjab fire counts and FRP |
| `sanity_checks.py` | Coverage, repeated values and pollutant-leakage checks on the built inputs |
| `dualaqi.py` | Features, all models (baselines, LSTM, DualAQI), blocked and forward CV, metrics, bootstrap CIs |
| `test_causality.py` | Tests that fire and calendar inputs use only information available at forecast time |
| `seed_summary.py` | Mean ± SD over seeds 0–2 |
| `dualaqi_v2_compare.py` | DualAQI v2 selection (development folds) and the held-out tests |
| `dualaqi_ablation.py` | Ablation: DualAQI vs single-branch and calendar-only variants, paired dRMSE CIs |
| `case_study.py` | Episode metrics (RMSE, MAE, bias, category accuracy, POD/FAR) for a date window |
| `bias_by_level.py` | Forecast bias by observed AQI category |
| `objective1_visualizations.ipynb`, `objective2_figures.py`, `objective3_figures.py`, `objective4_figures.py` | Figures in `results/figures/` |
| `build_supplementary.py` | Supplementary Material PDF and the Excel tables in `results/supplementary/` |

## Setup

Python 3.12 (the versions used are pinned in `requirements.txt`):

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data

The processed inputs used for the paper are included in `data/`:

| File | Columns | Content |
|---|---|---|
| `pm25_hourly.csv` | `datetime, pm25` | City-mean PM2.5 (µg/m³) over the CPCB Delhi stations reporting in each hour, 2019–2025 |
| `pm25_station_hours.csv` | `station, 2019 … 2025` | Hours with valid PM2.5 per station and year |
| `era5_hourly.csv` | `datetime, t2m, rh, u10, v10, sp, blh` | ERA5 hourly at 28.75° N, 77.25° E |
| `punjab_fire_daily.csv` | `date, fire_count, frp_sum` | Daily VIIRS S-NPP detections (type 0) inside Punjab, Dec 2018–Dec 2025 |

All timestamps are UTC. To rebuild them from the raw sources:

```bash
.venv\Scripts\python build_inputs.py --pm25 --era5 --years 2019-2025
set FIRMS_MAP_KEY=...                       # free key: firms.modaps.eosdis.nasa.gov/api/map_key/
.venv\Scripts\python download_firms.py --start 2018-12-01 --end 2025-12-31
.venv\Scripts\python build_inputs.py --fire --fire-start 2018-12-01 --years 2019-2025
.venv\Scripts\python sanity_checks.py
```

- **PM2.5:** hourly files of the 36 CPCB/DPCC/IITM Delhi stations (one file per station and year,
  folder `Delhi PM2.5 Air Quality/`, not included in this repository). The CPCB timestamps are IST
  and mark the start of each hourly window; each value is relabelled to the UTC hour at which it
  becomes available (stamp − 4 h). Values outside (0, 1000] µg/m³ are removed; gaps are not filled.
  The city value is the mean of the stations reporting in that hour, taken before AQI conversion.
- **ERA5:** Open-Meteo archive API (`models=era5`). Checked against Copernicus CDS ERA5 files for the
  same grid cell on 9,504 overlapping hours (r ≥ 0.999). Boundary-layer height is missing from the
  Open-Meteo record for January–June 2024, so that period is not used for testing.
- **Fires:** FIRMS `VIIRS_SNPP_SP` (S-NPP only, standard processing), kept if inside the GADM 4.1
  Punjab polygon (`raw/boundaries/gadm41_IND_1.json`, not redistributed) and of type 0 (vegetation fire).

## Forecast setup and causality

- One forecast origin per day at 03:00 UTC (08:30 IST); targets are AQI at t + 24, 48 and 72 h.
- Only data at or before the origin is used. PM2.5 and ERA5 features are forward-filled at most 3 h;
  the 168 h LSTM sequence is forward-filled at most 24 h, then set to 0. Targets are never filled.
- The fire count of UTC day d is used only from 03:00 UTC on day d + 1. `test_causality.py` checks
  this at feature level, with a negative control and end to end (logs: `test_causality*.log`).
- PM2.5 is converted to AQI with the CPCB National AQI breakpoints (Supplementary Table S1).
- **Forward CV:** expanding window, 90-day test blocks, training targets end before the test block's
  input window starts. Folds 0–7 (2020–2021) are development, folds 8–15 (2022–2023) and 16–21
  (July 2024–December 2025) are held-out tests.
- **Blocked CV** (baselines only): 10 contiguous blocks with a 240 h embargo (168 h input + 72 h horizon).

## Models

- **Baselines:** Persistence, Ridge, GBM (HistGradientBoosting), RandomForest, XGBoost, Ensemble (mean of
  Ridge, GBM and RandomForest), ARIMA (log PM2.5 regressed on daily and annual Fourier terms with ARMA
  errors; order chosen per fold by AIC among converged fits from (1,0,0), (2,0,1), (2,0,2); lognormal
  mean back-transform), and an LSTM (64 units, linear output layer).
- **DualAQI:** sequence branch LSTM(8 → 64) over 168 h; event branch MLP(24 → 64 → 32) over 7 daily
  lags of log1p fire count, 7 lags of log1p FRP and 10 calendar inputs; gate g = σ(W[h; e]);
  fused z = g·h + (1 − g)·P e; head 64 → 64 → ReLU → Dropout 0.3 → 3 horizons (35,299 parameters).
  DualAQI v1 is the same model without the FRP lags.
- **Ablation controls:** single-branch (DualAQI without the event branch: same LSTM and head,
  23,299 parameters) and calendar-only (event branch without fire inputs).
- Training: Adam (lr 1e-3), 15 epochs, batch 128, gradient clipping 1.0, horizon loss weights
  0.25/0.35/0.40, seeds 0, 1, 2. Neural results are reported as the mean over the three seeds.

## Reproducing the results

```bash
# baselines and LSTM, seeds 0-2 (ARIMA does not depend on the seed and is run once)
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results --launch-hours 3 --n-boot 500
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results/seed1 --launch-hours 3 --n-boot 500 --seed 1 --no-arima
.venv\Scripts\python dualaqi.py --data-dir data --out-dir results/seed2 --launch-hours 3 --n-boot 500 --seed 2 --no-arima
.venv\Scripts\python seed_summary.py

# DualAQI v1 and the ablation variants, forward CV, per seed s in 0,1,2
.venv\Scripts\python test_causality.py
.venv\Scripts\python dualaqi.py --only-dualaqi --schemes forward --launch-hours 3 --n-boot 500 --seed s --out-dir results/dualaqi/fire/seed<s>
.venv\Scripts\python dualaqi.py --only-dualaqi --schemes forward --launch-hours 3 --n-boot 500 --seed s --no-fire       --out-dir results/dualaqi/nofire/seed<s>
.venv\Scripts\python dualaqi.py --only-dualaqi --schemes forward --launch-hours 3 --n-boot 500 --seed s --calendar-only --out-dir results/dualaqi/calendar/seed<s>

# DualAQI v2: selection on folds 0-7, then the held-out tests (protocol: results/dualaqi_v2/PLAN.md)
.venv\Scripts\python dualaqi_v2_compare.py --stage dev
.venv\Scripts\python dualaqi_v2_compare.py --stage test
.venv\Scripts\python dualaqi_v2_compare.py --stage test2024
.venv\Scripts\python dualaqi_ablation.py --version v2
.venv\Scripts\python dualaqi_ablation.py --version v2 --folds 8-15

# case study, bias analysis, figures, supplementary material
.venv\Scripts\python case_study.py --start 2023-10-20 --end 2023-11-20
.venv\Scripts\python bias_by_level.py
.venv\Scripts\python objective2_figures.py
.venv\Scripts\python objective3_figures.py
.venv\Scripts\python objective4_figures.py
.venv\Scripts\python build_supplementary.py
```

`dualaqi.py` writes `benchmark_long.csv` (RMSE, MAE, R², bias, category accuracy, skill vs persistence,
95% day-bootstrap CI of the RMSE difference, per-fold R²) and `oof_predictions.csv.gz` (every
out-of-fold forecast) to `--out-dir`. The run logs of the reported runs are kept next to their outputs
in `results/`. `results/archive_v1_old_arima/` keeps the outputs from before the ARIMA convergence fix.

## Limitations

- Retrospective study: ERA5 is not available in real time, so an operational version would need
  forecast meteorology and retraining.
- The target is the city-mean PM2.5 converted to AQI at the valid hour (not a 24 h average).
- Festival flags cover Diwali 2019–2023 only; Diwali 2024–2025, Holi and Eid are not flagged.
- Hyperparameters are fixed, not tuned. Early stopping was tested on the development folds and not
  adopted (Amendment 1 in `results/dualaqi_v2/PLAN.md`).
- ERA5 is taken from one grid cell over Delhi.

## Licence and data sources

The code in this repository is released under the MIT License (see `LICENSE`). The MIT License covers the code only. The input data in `data/` and the outputs derived from it in `results/` come from the third-party sources below and remain under their own terms; please cite them when you reuse the data.

- **PM2.5:** hourly station data from the Central Pollution Control Board (CPCB) / Delhi Pollution Control Committee (DPCC) network, Government of India open data.
- **Meteorology:** ERA5 hourly reanalysis (Hersbach et al., 2020) obtained through the Open-Meteo archive API (CC BY 4.0). Contains modified Copernicus Climate Change Service information [2026]. Neither the European Commission nor ECMWF is responsible for any use of the Copernicus information or data it contains.
- **Fire activity:** NASA FIRMS VIIRS S-NPP active fire product (standard processing). We acknowledge the use of data and imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://www.earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS).
- **Boundary:** GADM 4.1 Punjab polygon, used only to select fires. It is not redistributed here (GADM data are free for academic use, but redistribution is not allowed).
