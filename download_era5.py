#!/usr/bin/env python3
"""Download ERA5 hourly single-level fields for Delhi, one file per month, into raw/era5/.

Needs a Copernicus CDS account and API key in %USERPROFILE%\\.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
and the ERA5 licence accepted once on the dataset's web page.

Default range is 2019-02 to 2022-12 (the months missing from raw/era5/).
Existing files are skipped, so the script can be re-run after an interruption.

    .venv\\Scripts\\python download_era5.py
    .venv\\Scripts\\python download_era5.py --start 2019-02 --end 2019-03   # small test first
"""
import argparse
import os

import pandas as pd

VARIABLES = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind',
             '10m_v_component_of_wind', 'surface_pressure', 'boundary_layer_height']
AREA = [29.0, 76.75, 28.25, 77.5]  # N, W, S, E: same 4x4 box as the 2023 files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2019-02')
    ap.add_argument('--end', default='2022-12')
    ap.add_argument('--out-dir', default=os.path.join('raw', 'era5'))
    args = ap.parse_args()

    import cdsapi
    client = cdsapi.Client()
    os.makedirs(args.out_dir, exist_ok=True)
    for m in pd.period_range(args.start, args.end, freq='M'):
        target = os.path.join(args.out_dir, f'era5_{m.year}_{m.month:02d}.nc')
        if os.path.exists(target):
            print(f'skip {target} (exists)')
            continue
        print(f'requesting {m} ...', flush=True)
        client.retrieve('reanalysis-era5-single-levels', {
            'product_type': ['reanalysis'],
            'variable': VARIABLES,
            'year': [str(m.year)],
            'month': [f'{m.month:02d}'],
            'day': [f'{d:02d}' for d in range(1, m.days_in_month + 1)],
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': AREA,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }, target + '.part')
        os.replace(target + '.part', target)
        print(f'  wrote {target}')


if __name__ == '__main__':
    main()
