#!/usr/bin/env python3
"""Writes synthetic inputs in the expected schema, to smoke-test dualaqi.py. NOT real data."""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
os.makedirs('data', exist_ok=True)

idx = pd.date_range('2019-01-01', '2023-12-31 23:00', freq='h')
n = len(idx)
doy = idx.dayofyear.values
hod = idx.hour.values

# PM2.5: seasonal + diurnal + AR(1) noise, with 2% random gaps
seasonal = 130 + 120 * np.cos(2 * np.pi * (doy - 15) / 365.25)
diurnal = 12 * np.sin(2 * np.pi * (hod - 8) / 24)
noise = np.zeros(n)
e = rng.normal(0, 10, n)
for i in range(1, n):
    noise[i] = 0.97 * noise[i - 1] + e[i]
pm = pd.Series(np.clip(seasonal + diurnal + noise, 3, None), index=idx)
pm[rng.random(n) < 0.02] = np.nan
pd.DataFrame({'datetime': idx, 'pm25': pm.values}).to_csv('data/pm25_hourly.csv', index=False)

# ERA5-like hourly meteorology
met = pd.DataFrame({
    'datetime': idx,
    't2m': 288 + 10 * np.cos(2 * np.pi * (doy - 200) / 365.25) + 4 * np.sin(2 * np.pi * (hod - 14) / 24),
    'rh': np.clip(60 + 15 * np.cos(2 * np.pi * (doy - 30) / 365.25) + rng.normal(0, 5, n), 5, 100),
    'u10': rng.normal(0, 2, n),
    'v10': rng.normal(0, 2, n),
    'sp': 1000 + rng.normal(0, 3, n),
    'blh': np.clip(900 + 500 * np.sin(2 * np.pi * (hod - 12) / 24)
                   - 300 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 50, n), 50, None),
})
met.to_csv('data/era5_hourly.csv', index=False)

# Daily AOD summary and 16x16 tiles, with 30% of days missing
days = pd.date_range('2019-01-01', '2023-12-31', freq='D')
m = len(days)
terra = 0.4 + 0.3 * rng.random(m)
aqua = 0.4 + 0.3 * rng.random(m)
terra[rng.random(m) < 0.3] = np.nan
aqua[rng.random(m) < 0.3] = np.nan
pd.DataFrame({'date': days, 'aod_terra': terra, 'aod_aqua': aqua}).to_csv('data/aod_daily.csv', index=False)

tiles = (0.4 + 0.3 * rng.random((m, 2, 16, 16))).astype(np.float32)
present = rng.random(m) >= 0.3
np.savez('data/aod_tiles.npz',
         dates=np.array(days[present].strftime('%Y-%m-%d'), dtype='U10'),
         tiles=tiles[present])
print('synthetic inputs written to ./data')
