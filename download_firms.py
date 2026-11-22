#!/usr/bin/env python3
"""Download NASA FIRMS active-fire detections for a box around Punjab (India), via the FIRMS area API.

Source: VIIRS_SNPP_SP (VIIRS S-NPP 375 m, standard processing / archive). S-NPP only, so the
sensor is the same for every year. The box is padded around Punjab; build_inputs.py --fire later
keeps only detections inside the Punjab state boundary.

The MAP_KEY is read from the environment variable FIRMS_MAP_KEY and is never written to disk.
Requests are made in 5-day chunks (the API's maximum). Each chunk is saved as
raw/firms/<source>_<start>.csv; existing chunks are skipped, so the script can be re-run after an
interruption. An empty chunk (header only) is a valid result: no fires.

    set FIRMS_MAP_KEY=...            (PowerShell: $env:FIRMS_MAP_KEY="...")
    .venv\\Scripts\\python download_firms.py --start 2018-12-01 --end 2025-12-31
"""
import argparse
import os
import time
import urllib.request

import pandas as pd

BOX = (73.7, 29.4, 77.1, 32.7)  # west, south, east, north: Punjab (29.5-32.5 N, 73.9-76.9 E) plus padding
HEADER_START = 'latitude,longitude,'
MAX_DAYS = 5  # the area API accepts day_range 1..5


def fetch(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'firms-download/1.0'}),
                                        timeout=120) as r:
                return r.read().decode('utf-8')
        except Exception as e:  # network hiccup or rate limit: back off and retry
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))
            print(f'  retry after {type(e).__name__}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2019-01-01')
    ap.add_argument('--end', default='2023-12-31')
    ap.add_argument('--source', default='VIIRS_SNPP_SP')
    ap.add_argument('--out-dir', default=os.path.join('raw', 'firms'))
    args = ap.parse_args()
    key = os.environ.get('FIRMS_MAP_KEY')
    if not key:
        raise SystemExit('set the FIRMS_MAP_KEY environment variable first')
    os.makedirs(args.out_dir, exist_ok=True)
    box = ','.join(str(v) for v in BOX)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    d, n_new, n_rows = start, 0, 0
    while d <= end:
        days = min(MAX_DAYS, (end - d).days + 1)
        path = os.path.join(args.out_dir, f'{args.source}_{d:%Y-%m-%d}_{days}d.csv')
        if not os.path.exists(path):
            text = fetch(f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{args.source}/{box}/{days}/{d:%Y-%m-%d}')
            if not text.startswith(HEADER_START):
                raise SystemExit(f'unexpected response for {d:%Y-%m-%d}: {text[:200]!r}')
            with open(path + '.part', 'w', encoding='utf-8', newline='') as fh:
                fh.write(text)
            os.replace(path + '.part', path)
            n_new += 1
            n_rows += text.count('\n') - 1
            if n_new % 20 == 0:
                print(f'  {d:%Y-%m-%d}: {n_new} chunks downloaded, {n_rows} detections so far', flush=True)
            time.sleep(0.5)
        d += pd.Timedelta(days=days)
    print(f'done: {n_new} new chunks, {n_rows} detections, files in {args.out_dir}')


if __name__ == '__main__':
    main()
