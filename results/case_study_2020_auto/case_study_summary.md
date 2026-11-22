# Case study: most polluted 30-day episode

- Scheme: forward CV (each forecast made by a model trained only on earlier data).
- Episode: 2020-10-17 to 2020-11-15 (valid dates), chosen as the 30-day window with the highest mean observed AQI (448) over all forward-CV test dates.
- Forecast origin 03:00 UTC (08:30 IST); one forecast per day per horizon; 30 valid days.
- Observed AQI categories in the episode: Severe 16, Very poor 13, Poor 1.
- Observed AQI range: 267 to 912.
- AQI here is the dualaqi.py mapping applied to the hourly city PM2.5 at the valid time (not a 24-h average).
- POD = share of observed events that were forecast; FAR = share of forecast events that did not occur. Severe = AQI category Severe; Very poor+ = Very poor or Severe.
- DualAQI-Net, CNN and LateFusion are absent: no MODIS AOD yet.

## Metrics inside the episode

|   horizon | model        |   n |    RMSE |     MAE |     Bias |   CatAcc |   POD_vpoor_plus |   FAR_vpoor_plus |   POD_severe |   FAR_severe |
|----------:|:-------------|----:|--------:|--------:|---------:|---------:|-----------------:|-----------------:|-------------:|-------------:|
|        24 | Persistence  |  30 | 139.309 |  90.667 |   -7.109 |    0.667 |            0.966 |            0.034 |        0.688 |        0.267 |
|        24 | ARIMA        |  30 | 156.797 | 107.817 | -101.815 |    0.467 |            0.966 |            0.034 |        0.125 |        0     |
|        24 | Ridge        |  30 | 114.987 |  83.086 |  -15.419 |    0.733 |            1     |            0.033 |        0.75  |        0.25  |
|        24 | GBM          |  30 | 105.118 |  73.467 |   -7.878 |    0.767 |            0.931 |            0.036 |        0.875 |        0.176 |
|        24 | RandomForest |  30 | 117.088 |  80.641 |  -46.823 |    0.667 |            0.897 |            0.037 |        0.75  |        0.2   |
|        24 | XGBoost      |  30 | 109.561 |  85.24  |  -11.321 |    0.733 |            0.931 |            0.036 |        0.875 |        0.222 |
|        24 | Ensemble     |  30 | 108.077 |  75.688 |  -23.373 |    0.733 |            0.931 |            0.036 |        0.812 |        0.188 |
|        24 | LSTM         |  30 | 139.422 | 103.121 |  -77.244 |    0.6   |            0.828 |            0.04  |        0.562 |        0.1   |
|        48 | Persistence  |  30 | 171.351 | 119.231 |  -12.585 |    0.633 |            0.966 |            0.034 |        0.625 |        0.286 |
|        48 | ARIMA        |  30 | 166.934 | 119.082 | -113.592 |    0.333 |            0.897 |            0.037 |        0     |      nan     |
|        48 | Ridge        |  30 | 134.927 | 103.093 |  -62.154 |    0.6   |            0.931 |            0.036 |        0.625 |        0.286 |
|        48 | GBM          |  30 | 121.531 |  87.87  |    3.773 |    0.7   |            0.966 |            0.034 |        0.938 |        0.318 |
|        48 | RandomForest |  30 | 140.76  | 100.92  |  -55.829 |    0.533 |            0.897 |            0.037 |        0.562 |        0.308 |
|        48 | XGBoost      |  30 | 123.242 |  90.07  |    0.466 |    0.733 |            0.966 |            0.034 |        0.938 |        0.286 |
|        48 | Ensemble     |  30 | 126.238 |  89.022 |  -38.07  |    0.733 |            0.931 |            0.036 |        0.875 |        0.222 |
|        48 | LSTM         |  30 | 157.18  | 124.042 | -100.56  |    0.3   |            0.724 |            0.045 |        0.312 |        0.375 |
|        72 | Persistence  |  30 | 177.593 | 137.2   |  -16.747 |    0.5   |            0.966 |            0.034 |        0.438 |        0.462 |
|        72 | ARIMA        |  30 | 168.248 | 120.685 | -115.552 |    0.333 |            0.897 |            0.037 |        0     |      nan     |
|        72 | Ridge        |  30 | 150.018 | 107.928 |  -83.822 |    0.5   |            0.828 |            0.04  |        0.5   |        0.2   |
|        72 | GBM          |  30 | 134.5   |  93.322 |  -27.107 |    0.7   |            0.966 |            0.034 |        0.875 |        0.3   |
|        72 | RandomForest |  30 | 139.112 |  96.828 |  -56.689 |    0.7   |            0.897 |            0.037 |        0.75  |        0.25  |
|        72 | XGBoost      |  30 | 142.481 |  99.898 |  -18.344 |    0.667 |            0.931 |            0.036 |        0.75  |        0.294 |
|        72 | Ensemble     |  30 | 136.545 |  93.888 |  -55.873 |    0.733 |            0.897 |            0.037 |        0.812 |        0.235 |
|        72 | LSTM         |  30 | 169.679 | 132.318 | -113.776 |    0.167 |            0.655 |            0.05  |        0.188 |        0.5   |

## LSTM over seeds 0-2 (mean +- SD)

|   horizon | RMSE             | MAE              | CatAcc         | POD_severe     |
|----------:|:-----------------|:-----------------|:---------------|:---------------|
|        24 | 141.253 +- 1.766 | 103.737 +- 0.990 | 0.633 +- 0.033 | 0.542 +- 0.036 |
|        48 | 160.580 +- 3.123 | 124.518 +- 0.849 | 0.256 +- 0.038 | 0.208 +- 0.095 |
|        72 | 174.039 +- 5.503 | 135.016 +- 4.555 | 0.167 +- 0.033 | 0.062 +- 0.108 |
