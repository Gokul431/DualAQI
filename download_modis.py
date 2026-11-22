#!/usr/bin/env python3
"""Download MODIS 3 km aerosol swaths (MOD04_3K Terra, MYD04_3K Aqua, Collection 6.1) that
intersect the box around Connaught Place, into raw/modis/terra and raw/modis/aqua.

Needs a NASA Earthdata login (https://urs.earthdata.nasa.gov). earthaccess asks for it on
the first run, or reads it from %USERPROFILE%\\_netrc.

Size: roughly 1-2 daytime granules per satellite per day, a few MB each, so the full
2019-2023 range is tens of GB. Test on a short range first:

    .venv\\Scripts\\python download_modis.py --start 2019-01-01 --end 2019-01-07
    .venv\\Scripts\\python download_modis.py                       # full 2019-2023

Re-running skips files that are already downloaded. This script only downloads. Building
aod_daily.csv and aod_tiles.npz from the swaths (QA = 3, box average, 16x16 regrid) is a
separate step.
"""
import argparse
import os

CP_LAT, CP_LON = 28.6315, 77.2167
HALF_LAT, HALF_LON = 0.9, 1.02          # about 100 km each way: a 200 km x 200 km box
PRODUCTS = {'terra': 'MOD04_3K', 'aqua': 'MYD04_3K'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2023-12-31')
    ap.add_argument('--out-dir', default=os.path.join('raw', 'modis'))
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()

    import earthaccess
    earthaccess.login(strategy='interactive', persist=True)
    bbox = (CP_LON - HALF_LON, CP_LAT - HALF_LAT, CP_LON + HALF_LON, CP_LAT + HALF_LAT)  # W, S, E, N
    for sat, short in PRODUCTS.items():
        res = earthaccess.search_data(short_name=short, version='61', bounding_box=bbox,
                                      temporal=(args.start, args.end + 'T23:59:59'), day_night_flag='day')
        out = os.path.join(args.out_dir, sat)
        os.makedirs(out, exist_ok=True)
        print(f'{short}: {len(res)} granules, {args.start} -> {args.end}', flush=True)
        earthaccess.download(res, out, threads=args.threads)
        print(f'  {short}: {len(os.listdir(out))} files in {out}')


if __name__ == '__main__':
    main()
