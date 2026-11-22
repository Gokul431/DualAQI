# Case study: 20 Oct 2023 to 20 Nov 2023

- Scheme: forward CV (each forecast made by a model trained only on earlier data).
- Episode: 2023-10-20 to 2023-11-20 inclusive (valid dates), fixed by the user. Mean observed AQI 391.
- Forecast origin 03:00 UTC (08:30 IST); one forecast per day per horizon; 32 valid days.
- Observed AQI categories in the episode: Very poor 17, Severe 13, Satisfactory 1, Poor 1.
- Observed AQI range: 62 to 577.
- AQI here is the dualaqi.py mapping applied to the hourly city PM2.5 at the valid time (not a 24-h average).
- POD = share of observed events that were forecast; FAR = share of forecast events that did not occur. Severe = AQI category Severe; Very poor+ = Very poor or Severe.
- DualAQI-Net, CNN and LateFusion are absent: no MODIS AOD yet.

## Metrics inside the episode

|   horizon | model        |   n |    RMSE |     MAE |    Bias |   CatAcc |   POD_vpoor_plus |   FAR_vpoor_plus |   POD_severe |   FAR_severe |
|----------:|:-------------|----:|--------:|--------:|--------:|---------:|-----------------:|-----------------:|-------------:|-------------:|
|        24 | Persistence  |  32 | 116.781 |  75.749 |  -9.243 |    0.656 |            0.933 |            0.034 |        0.692 |        0.308 |
|        24 | ARIMA        |  32 | 106     |  82.24  | -46.904 |    0.5   |            0.967 |            0.065 |        0     |      nan     |
|        24 | Ridge        |  32 | 104.705 |  75.499 | -32.959 |    0.594 |            0.833 |            0.038 |        0.615 |        0.2   |
|        24 | GBM          |  32 |  96.853 |  63.738 |   7.261 |    0.719 |            0.967 |            0.033 |        0.846 |        0.312 |
|        24 | RandomForest |  32 |  98.368 |  60.349 |  -5.912 |    0.688 |            0.9   |            0.036 |        0.846 |        0.267 |
|        24 | XGBoost      |  32 |  97.022 |  59.071 |   4.634 |    0.75  |            0.933 |            0.034 |        0.846 |        0.214 |
|        24 | Ensemble     |  32 |  95.744 |  60.112 | -10.537 |    0.75  |            0.933 |            0.034 |        0.846 |        0.214 |
|        24 | LSTM         |  32 | 106.409 |  74.937 | -28.574 |    0.469 |            0.733 |            0.083 |        0.538 |        0.364 |
|        48 | Persistence  |  32 | 144.093 |  99.119 | -15.933 |    0.594 |            0.867 |            0.071 |        0.615 |        0.385 |
|        48 | ARIMA        |  32 | 110.911 |  87.595 | -51.807 |    0.469 |            0.933 |            0.067 |        0     |      nan     |
|        48 | Ridge        |  32 | 117.952 |  94.089 | -54.948 |    0.438 |            0.767 |            0.08  |        0.231 |        0     |
|        48 | GBM          |  32 | 107.342 |  76.242 |  -3.439 |    0.625 |            0.867 |            0.071 |        0.692 |        0.357 |
|        48 | RandomForest |  32 | 107.366 |  67.585 | -14.423 |    0.719 |            0.867 |            0.071 |        0.692 |        0.182 |
|        48 | XGBoost      |  32 | 104.947 |  68.377 |   2.988 |    0.719 |            0.9   |            0.069 |        0.769 |        0.231 |
|        48 | Ensemble     |  32 | 105.628 |  73.734 | -24.27  |    0.594 |            0.867 |            0.071 |        0.385 |        0.286 |
|        48 | LSTM         |  32 | 117.653 |  90.886 | -48.646 |    0.469 |            0.7   |            0.087 |        0.385 |        0.286 |
|        72 | Persistence  |  32 | 164.94  | 125.067 | -25.384 |    0.438 |            0.833 |            0.074 |        0.462 |        0.538 |
|        72 | ARIMA        |  32 | 111.314 |  88.011 | -52.411 |    0.5   |            0.967 |            0.065 |        0     |      nan     |
|        72 | Ridge        |  32 | 131.38  | 111.484 | -63.81  |    0.188 |            0.667 |            0.091 |        0     |        1     |
|        72 | GBM          |  32 | 118.412 |  77.172 |  -1.071 |    0.625 |            0.8   |            0.077 |        0.846 |        0.312 |
|        72 | RandomForest |  32 | 114.117 |  73.751 | -10.395 |    0.656 |            0.867 |            0.071 |        0.692 |        0.308 |
|        72 | XGBoost      |  32 | 132.207 |  83.807 |   6.923 |    0.594 |            0.833 |            0.074 |        0.846 |        0.389 |
|        72 | Ensemble     |  32 | 114.667 |  82.859 | -25.092 |    0.625 |            0.833 |            0.074 |        0.615 |        0.273 |
|        72 | LSTM         |  32 | 129.66  | 105.811 | -67.582 |    0.344 |            0.6   |            0.1   |        0.231 |        0     |

## LSTM over seeds 0-2 (mean +- SD)

|   horizon | RMSE             | MAE              | CatAcc         | POD_severe     |
|----------:|:-----------------|:-----------------|:---------------|:---------------|
|        24 | 103.010 +- 5.569 | 71.808 +- 4.860  | 0.479 +- 0.048 | 0.744 +- 0.178 |
|        48 | 111.363 +- 6.685 | 82.270 +- 8.185  | 0.531 +- 0.054 | 0.590 +- 0.178 |
|        72 | 119.211 +- 9.324 | 94.347 +- 10.340 | 0.406 +- 0.062 | 0.359 +- 0.118 |
