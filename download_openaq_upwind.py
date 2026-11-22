#!/usr/bin/env python3
"""Upwind PM2.5 (Haryana and Punjab, between the burning region and Delhi) from OpenAQ v3, daily.

Stations are fixed in advance by geography (CPCB stations on the NW corridor with records from 2019),
not by how well they help the model:
  near (Haryana, ~50-150 km from Delhi): Sonipat, Panipat, Karnal, Kurukshetra, Kaithal, Jind
  far  (Punjab, ~200-450 km):           Patiala, Mandi Gobindgarh, Khanna, Ludhiana, Jalandhar,
                                        Bathinda, Amritsar
OpenAQ daily values are IST calendar days (18:30 UTC to 18:30 UTC), so day d is complete before
03:00 UTC on d+1, and the same availability rule as the fire counts applies.

Each location may have two OpenAQ sensor ids (older and newer feed); values are averaged per
location and day. Days with < 50% hourly coverage or values outside (0, 1000] ug/m3 are dropped.
Output data/upwind_daily.csv: date, near_pm25, far_pm25, near_n, far_n (station means and counts;
empty where no station reported). The API key is read from OPENAQ_API_KEY and never stored.

    .venv\\Scripts\\python download_openaq_upwind.py --start 2018-12-01 --end 2025-12-31
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

LOCATIONS = {  # OpenAQ location id: (name, group)
    6951: ('Murthal, Sonipat', 'near'), 6958: ('Sector-18, Panipat', 'near'), 10910: ('Sector-12, Karnal', 'near'),
    7282: ('Sector-7, Kurukshetra', 'near'), 6971: ('Rishi Nagar, Kaithal', 'near'), 6963: ('Police Lines, Jind', 'near'),
    5544: ('Model Town, Patiala', 'far'), 5625: ('RIMT Univ., Mandi Gobindgarh', 'far'), 6940: ('Kalal Majra, Khanna', 'far'),
    5569: ('PAU, Ludhiana', 'far'), 5542: ('Civil Line, Jalandhar', 'far'), 6939: ('Hardev Nagar, Bathinda', 'far'),
    5551: ('Golden Temple, Amritsar', 'far'),
}
API = 'https://api.openaq.org/v3'


def get(path, key):
    for i in range(5):
        try:
            req = urllib.request.Request(API + path, headers={'X-API-Key': key, 'User-Agent': 'dualaqi-research'})
            with urllib.request.urlopen(req, timeout=90) as r:
                if int(r.headers.get('X-Ratelimit-Remaining', 10)) < 3:
                    time.sleep(int(r.headers.get('X-Ratelimit-Reset', 60)) + 1)
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(15 * (i + 1))
                continue
            raise
    raise SystemExit(f'giving up on {path.split("?")[0]}')


def sensor_days(sid, start, end, key, cache_dir):
    path = os.path.join(cache_dir, f'sensor_{sid}_{start}_{end}.json')
    if os.path.exists(path):
        return json.load(open(path))
    rows, page = [], 1
    while True:
        q = urllib.parse.urlencode({'date_from': start, 'date_to': end, 'limit': 1000, 'page': page})
        j = get(f'/sensors/{sid}/days?{q}', key)
        for r in j['results']:
            rows.append({'date': r['period']['datetimeFrom']['local'][:10], 'value': r['value'],
                         'pct': (r.get('coverage') or {}).get('percentComplete')})
        if len(j['results']) < 1000:
            break
        page += 1
        time.sleep(1.1)
    json.dump(rows, open(path + '.part', 'w'))
    os.replace(path + '.part', path)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', default='2018-12-01')
    ap.add_argument('--end', default='2025-12-31')
    ap.add_argument('--cache', default=os.path.join('raw', 'openaq_upwind'))
    ap.add_argument('--out', default=os.path.join('data', 'upwind_daily.csv'))
    args = ap.parse_args()
    key = os.environ.get('OPENAQ_API_KEY')
    if not key:
        raise SystemExit('set OPENAQ_API_KEY first')
    os.makedirs(args.cache, exist_ok=True)

    per_loc = []
    for lid, (name, group) in LOCATIONS.items():
        loc = get(f'/locations/{lid}', key)['results'][0]
        sids = [s['id'] for s in loc['sensors'] if s['parameter']['name'] == 'pm25']
        frames = []
        for sid in sids:
            d = pd.DataFrame(sensor_days(sid, args.start, args.end, key, args.cache))
            if len(d):
                frames.append(d)
            time.sleep(1.1)
        if not frames:
            print(f'  {name}: no data')
            continue
        d = pd.concat(frames)
        n0 = len(d)
        d = d[(d.value > 0) & (d.value <= 1000) & (d.pct.fillna(0) >= 50)]
        s = d.groupby('date').value.mean()
        s.index = pd.to_datetime(s.index)
        per_loc.append(s.rename(lid))
        print(f'  {group:4s} {name:30s} sensors {sids}: {len(s)} valid days '
              f'({s.index.min().date()} -> {s.index.max().date()}), {n0 - len(d)} day-records dropped', flush=True)

    days = pd.date_range(args.start, args.end, freq='D')
    wide = pd.concat(per_loc, axis=1).reindex(days)
    out = pd.DataFrame(index=days)
    for group in ('near', 'far'):
        cols = [lid for lid in wide.columns if LOCATIONS[lid][1] == group]
        out[f'{group}_pm25'] = wide[cols].mean(axis=1)
        out[f'{group}_n'] = wide[cols].notna().sum(axis=1)
    out.index.name = 'date'
    out.to_csv(args.out, date_format='%Y-%m-%d', float_format='%.2f')
    cov = out.assign(year=out.index.year).groupby('year')[['near_n', 'far_n']].agg(lambda x: (x > 0).mean())
    print(f'wrote {args.out}; share of days with at least one station:')
    print(cov.round(3).to_string())


if __name__ == '__main__':
    main()
