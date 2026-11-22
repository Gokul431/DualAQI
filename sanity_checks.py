#!/usr/bin/env python3
"""Sanity checks on the built inputs (run after build_inputs.py). Read-only.

1. Share of hours with city PM2.5 per year (and station-hour completeness).
2. Share of consecutive identical PM2.5 values (city series and per station).
   Above 20% suggests filled or daily values.
3. No other pollutant leaked into PM2.5: PM2.5 is read by column name only; this checks that it
   never equals another pollutant column row-for-row, and that PM2.5 <= PM10 almost always.
4. AOD inputs present or not (DualAQI-Net and the CNN / late-fusion baselines need them).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_inputs import CPCB_DIR, CPCB_PM_COL, station_name  # noqa: E402

DATA = 'data'
YEARS = range(2019, 2024)
OTHER = ['PM10 (µg/m³)', 'NO (µg/m³)', 'NO2 (µg/m³)', 'NOx (ppb)', 'NH3 (µg/m³)', 'SO2 (µg/m³)',
         'CO (mg/m³)', 'Ozone (µg/m³)', 'Benzene (µg/m³)', 'Toluene (µg/m³)']
FLAG = 0.20


def repeat_share(s):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return float((s.values[1:] == s.values[:-1]).mean())


def main():
    pm = pd.read_csv(os.path.join(DATA, 'pm25_hourly.csv'), parse_dates=['datetime'], index_col='datetime')
    met = pd.read_csv(os.path.join(DATA, 'era5_hourly.csv'), parse_dates=['datetime'], index_col='datetime')
    print('== Coverage')
    for name, s in [('pm25_hourly.csv', pm.pm25), ('era5_hourly.csv', met.dropna().t2m)]:
        s = s.dropna()
        print(f'  {name}: first {s.index.min()}, last {s.index.max()}, {len(s)} hours with data')
    full = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31 23:00', freq='h')
    missing_years = [y for y in YEARS if pm.pm25.reindex(full[full.year == y]).notna().sum() == 0]
    print(f'  years with no PM2.5: {missing_years or "none"}')

    print('\n== 1. Share of hours with city PM2.5, per year (UTC)')
    c = pm.pm25.reindex(full)
    share = c.notna().groupby(c.index.year).mean()
    st = pm.n_stations.reindex(full).fillna(0).groupby(full.year).agg(['mean', 'min'])
    for y in YEARS:
        print(f'  {y}: {share[y]:.4f}   stations per hour mean {st.loc[y, "mean"]:.1f}, min {int(st.loc[y, "min"])}')
    sh = pd.read_csv(os.path.join(DATA, 'pm25_station_hours.csv'), index_col='station')
    hrs = pd.Series({y: 8784 if y % 4 == 0 else 8760 for y in YEARS})
    comp = sh[[str(y) for y in YEARS]].div(hrs.values)
    print('  station completeness (share of station-hours valid), median by year:',
          comp.median().round(3).to_dict())

    print('\n== 2. Consecutive identical PM2.5 values')
    r_city = repeat_share(c)
    print(f'  city series: {r_city:.4f}' + ('   <-- ABOVE 20%: FLAG' if r_city > FLAG else ''))
    rows, leak = [], []
    for y in YEARS:
        for f in sorted(glob.glob(os.path.join(CPCB_DIR, str(y), 'site_*_1Hr.csv'))):
            d = pd.read_csv(f, usecols=lambda col: col in ['Timestamp', CPCB_PM_COL] + OTHER,
                            na_values=['NA', 'None', ''])
            p = d[CPCB_PM_COL]
            p = p.where((p > 0) & (p <= 1000))
            rows.append((station_name(f), y, p.notna().sum(), repeat_share(p)))
            both = d.dropna(subset=[CPCB_PM_COL])
            for col in [o for o in OTHER if o in d.columns]:
                m = both[col].notna()
                if m.sum() > 100:
                    eq = float((both.loc[m, CPCB_PM_COL] == both.loc[m, col]).mean())
                    leak.append((station_name(f), y, col, m.sum(), eq))
            if 'PM10 (µg/m³)' in d.columns:
                m = d[[CPCB_PM_COL, 'PM10 (µg/m³)']].dropna()
                if len(m) > 100:
                    leak.append((station_name(f), y, 'PM2.5>PM10', len(m),
                                 float((m[CPCB_PM_COL] > m['PM10 (µg/m³)']).mean())))
    r = pd.DataFrame(rows, columns=['station', 'year', 'n', 'repeat_share']).dropna()
    print(f'  per station-year: median {r.repeat_share.median():.3f}, 90th pct {r.repeat_share.quantile(.9):.3f}, '
          f'max {r.repeat_share.max():.3f}')
    hi = r[r.repeat_share > FLAG].sort_values('repeat_share', ascending=False)
    if len(hi):
        print(f'  !!! {len(hi)} station-years ABOVE 20% repeated values (may be filled or stuck sensors):')
        print(hi.round(3).to_string(index=False))
    else:
        print('  no station-year above 20%')

    print('\n== 3. Other pollutants leaking into PM2.5')
    L = pd.DataFrame(leak, columns=['station', 'year', 'other', 'n', 'share'])
    eq = L[L.other != 'PM2.5>PM10']
    worst = eq.sort_values('share', ascending=False).head(5)
    print(f'  PM2.5 equal to another pollutant column, row-for-row: max share {eq.share.max():.4f}')
    print(worst.round(4).to_string(index=False))
    gt = L[L.other == 'PM2.5>PM10']
    print(f'  share of hours with PM2.5 > PM10: median {gt.share.median():.3f}, max {gt.share.max():.3f}')
    print('  ' + gt.sort_values('share', ascending=False).head(5).round(3).to_string(index=False).replace('\n', '\n  '))
    print('  PM2.5 is read by its column name only ("PM2.5 (µg/m³)"); no other column is read into it.')

    print('\n== 4. AOD inputs')
    for f in ['aod_daily.csv', 'aod_tiles.npz']:
        print(f'  {f}: {"present" if os.path.exists(os.path.join(DATA, f)) else "NOT present"}')
    print('  Without them, AOD features are empty and DualAQI-Net, CNN and LateFusion are skipped.')


if __name__ == '__main__':
    main()
