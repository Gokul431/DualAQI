#!/usr/bin/env python3
"""Build data/pm25_hourly.csv and data/era5_hourly.csv for dualaqi.py.

All output timestamps are UTC (naive). Each row is labelled by the time its value becomes
available, so with these inputs `dualaqi.py --launch-hours 3` means a 03:00 UTC = 08:30 IST
forecast origin that sees only data available by then.

PM2.5  (--pm25, from the CPCB station folder "Delhi PM2.5 Air Quality/<year>/site_*.csv")
  * CPCB `Timestamp` is IST and marks the START of the 1-hour averaging window. (Checked by
    matching four stations against OpenAQ v3 records, which carry explicit UTC windows.)
  * A window stamped S (IST) ends at S + 1 h IST = S - 4:30 UTC. That falls on a half hour,
    so it is labelled at the next UTC hour, S - 4:00 UTC. The value is never used before it exists.
  * Values outside (0, 1000] ug/m3 are dropped (0 and negatives are invalid readings; 1000 is the
    upper end of the export). Blank / "NA" cells are missing.
  * City value = mean over all stations reporting that hour (before AQI conversion).
    `n_stations` records how many stations contributed. Gaps stay empty; nothing is filled.

ERA5   (--era5)
  * Default source: Open-Meteo archive API, `models=era5`, hourly, UTC. It returns the same
    ERA5 grid cell as the CDS files (28.75 N, 77.25 E) and the same values, rounded
    (checked against raw/era5/era5_2023_jan_mar.nc; see --check-against-cds).
    Raw responses are cached in raw/era5_openmeteo/ so the build is repeatable offline.
  * --era5-source nc reads the CDS NetCDF files in raw/era5/ instead (only 2019-01 and 2023 exist).
  * rh from t2m and d2m (Magnus, Alduchov & Eskridge 1996). t2m in K, sp in hPa, u10/v10 in m/s
    (from Open-Meteo speed and direction: u = -ws sin(dir), v = -ws cos(dir)), blh in m.
"""
import argparse
import glob
import json
import os
import urllib.request

import numpy as np
import pandas as pd

CP_LAT, CP_LON = 28.6315, 77.2167
PM_MIN, PM_MAX = 0.0, 1000.0
CPCB_DIR = 'Delhi PM2.5 Air Quality'
CPCB_PM_COL = 'PM2.5 (µg/m³)'
CPCB_TO_UTC_LABEL = pd.Timedelta(hours=-4)  # IST window start -> next UTC hour after the window ends
ERA5_NEED = ['t2m', 'd2m', 'u10', 'v10', 'sp', 'blh']
OM_VARS = ['temperature_2m', 'dew_point_2m', 'surface_pressure', 'wind_speed_10m',
           'wind_direction_10m', 'boundary_layer_height']
ERA5_ALIASES = {
    '2m_temperature': 't2m', '2m_dewpoint_temperature': 'd2m',
    '10m_u_component_of_wind': 'u10', '10m_v_component_of_wind': 'v10',
    'surface_pressure': 'sp', 'boundary_layer_height': 'blh',
}


def coverage(name, s):
    s = s.dropna()
    print(f'{name}: {len(s)} hours with data, {s.index.min()} -> {s.index.max()}')
    print('  hours with data per year:', s.groupby(s.index.year).size().to_dict())


# ----------------------------------------------------------------------------- PM2.5

def station_name(path):
    return os.path.basename(path)[len('site_'):-len('_1Hr.csv')]


def build_pm25(cpcb_dir, years, out_dir):
    frames = []
    for y in years:
        files = sorted(glob.glob(os.path.join(cpcb_dir, str(y), 'site_*_1Hr.csv')))
        if not files:
            raise SystemExit(f'no station files for {y} in {cpcb_dir}')
        for f in files:
            d = pd.read_csv(f, usecols=['Timestamp', CPCB_PM_COL], na_values=['NA', 'None', ''])
            d.columns = ['ts', 'pm25']
            d['station'] = station_name(f)
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['ts'] = pd.to_datetime(d.ts, format='%Y-%m-%d %H:%M:%S')
    n_rows = len(d)
    d = d.dropna(subset=['pm25'])
    print(f'PM2.5: {n_rows} station-hours in {d.station.nunique()} station files, years {years[0]}-{years[-1]}')
    print(f'  {n_rows - len(d)} blank / NA')
    bad_lo, bad_hi, at_max = (d.pm25 <= PM_MIN), (d.pm25 > PM_MAX), (d.pm25 == PM_MAX)
    print(f'  dropped {bad_lo.sum()} values <= 0 and {bad_hi.sum()} > {PM_MAX:g}; '
          f'{at_max.sum()} values exactly {PM_MAX:g} kept (possible instrument ceiling)')
    d = d[~(bad_lo | bad_hi)]
    dup = d.duplicated(['station', 'ts'])
    if dup.any():
        raise SystemExit(f'{dup.sum()} duplicate station-hours; check the source files')

    d['datetime'] = d.ts + CPCB_TO_UTC_LABEL
    g = d.groupby('datetime')
    city = pd.DataFrame({'pm25': g.pm25.mean(), 'n_stations': g.station.nunique()})
    idx = pd.date_range(city.index.min(), city.index.max(), freq='h')
    city = city.reindex(idx)
    city['n_stations'] = city.n_stations.fillna(0).astype(int)
    city.index.name = 'datetime'
    out = os.path.join(out_dir, 'pm25_hourly.csv')
    city.to_csv(out, float_format='%.3f')

    per_year = (d.assign(year=d.ts.dt.year).groupby(['station', 'year']).size()
                .unstack(fill_value=0))
    per_year.to_csv(os.path.join(out_dir, 'pm25_station_hours.csv'))
    print(f'  stations with PM2.5 per year: {(per_year > 0).sum().to_dict()}')
    print(f'  distinct stations overall: {len(per_year)}')
    print(f'  wrote {out} and pm25_station_hours.csv')
    coverage('pm25', city.pm25)


# ----------------------------------------------------------------------------- ERA5

def rh_from_dewpoint(t_k, td_k):
    t, td = t_k - 273.15, td_k - 273.15
    a, b = 17.625, 243.04
    return np.clip(100 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t)), 0, 100)


def to_model_vars(raw):
    """raw: DataFrame with t2m, d2m (K), u10, v10 (m/s), sp (Pa), blh (m)."""
    out = pd.DataFrame(index=raw.index)
    out['t2m'] = raw.t2m
    out['rh'] = rh_from_dewpoint(raw.t2m, raw.d2m)
    out['u10'] = raw.u10
    out['v10'] = raw.v10
    out['sp'] = raw.sp / 100.0
    out['blh'] = raw.blh
    out.index.name = 'datetime'
    return out.sort_index()


def fetch_openmeteo(year, cache_dir):
    path = os.path.join(cache_dir, f'era5_openmeteo_{year}.json')
    if not os.path.exists(path):
        url = ('https://archive-api.open-meteo.com/v1/archive'
               f'?latitude={CP_LAT}&longitude={CP_LON}&start_date={year}-01-01&end_date={year}-12-31'
               f'&hourly={",".join(OM_VARS)}&models=era5&timezone=GMT&wind_speed_unit=ms')
        req = urllib.request.Request(url, headers={'User-Agent': 'dualaqi-build/1.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        with open(path + '.part', 'wb') as fh:
            fh.write(body)
        os.replace(path + '.part', path)
    with open(path) as fh:
        j = json.load(fh)
    if 'hourly' not in j:
        raise SystemExit(f'{path}: unexpected response: {str(j)[:300]}')
    h = j['hourly']
    missing = [v for v in OM_VARS if v not in h]
    if missing:
        raise SystemExit(f'{path}: variables missing from response: {missing}. Present: {list(h)}')
    df = pd.DataFrame({v: np.array(h[v], dtype=float) for v in OM_VARS},
                      index=pd.DatetimeIndex(pd.to_datetime(h['time'])))
    print(f'  {year}: grid=({j["latitude"]:.2f}N, {j["longitude"]:.2f}E), {len(df)} hours, '
          f'{int(df.isna().any(axis=1).sum())} hours with a missing variable')
    return df, (j['latitude'], j['longitude'])


def era5_from_openmeteo(years, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    parts, grids = [], set()
    print(f'ERA5 from Open-Meteo (models=era5), years {years[0]}-{years[-1]}')
    for y in years:
        df, grid = fetch_openmeteo(y, cache_dir)
        parts.append(df)
        grids.add(grid)
    if len(grids) > 1:
        raise SystemExit(f'responses resolve to different grid cells: {grids}')
    om = pd.concat(parts).sort_index()
    rad = np.deg2rad(om.wind_direction_10m)
    raw = pd.DataFrame({
        't2m': om.temperature_2m + 273.15,
        'd2m': om.dew_point_2m + 273.15,
        'u10': -om.wind_speed_10m * np.sin(rad),
        'v10': -om.wind_speed_10m * np.cos(rad),
        'sp': om.surface_pressure * 100.0,
        'blh': om.boundary_layer_height,
    }, index=om.index)
    return raw


def read_nc_point(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    ds = ds.rename({k: v for k, v in ERA5_ALIASES.items() if k in ds.data_vars})
    tname = 'valid_time' if 'valid_time' in ds.coords else 'time'
    lat = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon = 'longitude' if 'longitude' in ds.coords else 'lon'
    have = [v for v in ERA5_NEED if v in ds.data_vars]
    if not have:
        raise SystemExit(f'{path}: none of {ERA5_NEED} found. Variables in file: {list(ds.data_vars)}')
    pt = ds[have].sel({lat: CP_LAT, lon: CP_LON}, method='nearest')
    grid = (float(pt[lat]), float(pt[lon]))
    df = pt.to_dataframe().reset_index().set_index(tname)[have]
    df.index = pd.DatetimeIndex(df.index)
    return df, grid


def era5_from_nc(era5_dir, verbose=True):
    files = sorted(glob.glob(os.path.join(era5_dir, '*.nc')))
    if not files:
        raise SystemExit(f'no .nc files in {era5_dir}')
    merged, grids = None, set()
    for f in files:
        df, grid = read_nc_point(f)
        grids.add(grid)
        if verbose:
            print(f'  {os.path.basename(f)}: {df.index.min()} -> {df.index.max()}  vars={list(df.columns)}')
        if merged is not None:
            both = merged.index.intersection(df.index)
            for v in set(merged.columns) & set(df.columns):
                a, b = merged.loc[both, v], df.loc[both, v]
                ok = a.isna() | b.isna() | np.isclose(a, b, rtol=1e-4, atol=1e-3)
                if not ok.all():
                    raise SystemExit(f'{f}: {v} conflicts with an earlier file on {(~ok).sum()} hours')
            df = merged.combine_first(df)
        merged = df
    if len(grids) > 1:
        raise SystemExit(f'files resolve to different grid cells: {grids}')
    missing = [v for v in ERA5_NEED if v not in merged.columns]
    if missing:
        raise SystemExit(f'variables missing from every file: {missing}')
    return merged


def check_against_cds(om_raw, era5_dir):
    """Compare Open-Meteo ERA5 with the CDS NetCDF files on every overlapping hour."""
    if not glob.glob(os.path.join(era5_dir, '*.nc')):
        print('  (no CDS files to compare against)')
        return
    cds = era5_from_nc(era5_dir, verbose=False)
    both = om_raw.index.intersection(cds.index)
    print(f'  check vs CDS NetCDF on {len(both)} overlapping hours (Open-Meteo - CDS):')
    for v, scale, unit in [('t2m', 1, 'K'), ('d2m', 1, 'K'), ('u10', 1, 'm/s'), ('v10', 1, 'm/s'),
                           ('sp', 0.01, 'hPa'), ('blh', 1, 'm')]:
        diff = (om_raw.loc[both, v] - cds.loc[both, v]) * scale
        r = np.corrcoef(om_raw.loc[both, v], cds.loc[both, v])[0, 1]
        print(f'    {v:4s} mean|d|={diff.abs().mean():7.3f} {unit:4s} max|d|={diff.abs().max():7.3f}  r={r:.5f}')


def build_era5(args, years):
    if args.era5_source == 'openmeteo':
        raw = era5_from_openmeteo(years, os.path.join(args.raw_dir, 'era5_openmeteo'))
        check_against_cds(raw, os.path.join(args.raw_dir, 'era5'))
    else:
        print(f'ERA5 from NetCDF files in {os.path.join(args.raw_dir, "era5")}')
        raw = era5_from_nc(os.path.join(args.raw_dir, 'era5'))
    out = to_model_vars(raw)
    incomplete = out.isna().any(axis=1)
    if incomplete.any():
        print(f'  {incomplete.sum()} hours have some but not all variables:', out[incomplete].isna().sum().to_dict())
    path = os.path.join(args.out_dir, 'era5_hourly.csv')
    out.to_csv(path, float_format='%.4f')
    print(f'  wrote {path}')
    coverage('era5 (all variables present)', out.dropna().t2m)


# ----------------------------------------------------------------------------- Punjab fires

def punjab_path(boundary_json):
    """matplotlib Path for the Punjab (India) polygon from GADM 4.1 level 1, holes included."""
    from matplotlib.path import Path
    with open(boundary_json, encoding='utf-8') as fh:
        g = json.load(fh)
    feats = [f for f in g['features'] if f['properties']['NAME_1'] == 'Punjab']
    if len(feats) != 1:
        raise SystemExit(f'{boundary_json}: expected one Punjab feature, found {len(feats)}')
    geom = feats[0]['geometry']
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    verts, codes = [], []
    for poly in polys:
        for ring in poly:  # exterior first, then holes; even-odd filling treats holes correctly
            ring = np.asarray(ring, dtype=float)
            verts.append(ring)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
    return Path(np.vstack(verts), codes)


def build_fire(firms_dir, boundary_json, years, out_dir, start=None):
    """Daily Punjab fire counts and FRP sums from FIRMS VIIRS S-NPP detections (acq_date is UTC).

    Keeps detections inside the Punjab state boundary with type 0 (presumed vegetation fire);
    all confidence levels are kept. A day inside a downloaded chunk with no detections is a true 0.
    Columns: date, fire_count, frp_sum (MW, sum of fire radiative power over the day's detections).
    """
    files = sorted(glob.glob(os.path.join(firms_dir, 'VIIRS_SNPP_SP_*_*d.csv')))
    if not files:
        raise SystemExit(f'no FIRMS chunks in {firms_dir}; run download_firms.py first')
    covered = set()
    for f in files:
        chunk_start, days = os.path.basename(f)[len('VIIRS_SNPP_SP_'):-len('d.csv')].rsplit('_', 1)
        covered.update(pd.date_range(chunk_start, periods=int(days), freq='D'))
    want = pd.date_range(start or f'{years[0]}-01-01', f'{years[-1]}-12-31', freq='D')
    gaps = want.difference(pd.DatetimeIndex(sorted(covered)))
    if len(gaps):
        raise SystemExit(f'FIRMS chunks do not cover {len(gaps)} days, e.g. {list(gaps[:5].date)}')

    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d['acq_date'] = pd.to_datetime(d.acq_date)
    n_box = len(d)
    dup = d.duplicated(['latitude', 'longitude', 'acq_date', 'acq_time'])
    d = d[~dup]
    inside = punjab_path(boundary_json).contains_points(d[['longitude', 'latitude']].to_numpy())
    d_in = d[inside]
    veg = d_in.type == 0 if 'type' in d_in.columns else pd.Series(True, index=d_in.index)
    print(f'FIRMS VIIRS S-NPP: {n_box} detections in the download box, {dup.sum()} duplicates dropped')
    print(f'  inside Punjab: {inside.sum()}; type counts {d_in.type.value_counts().to_dict() if "type" in d_in else "n/a"}')
    print(f'  confidence counts (kept): {d_in[veg].confidence.value_counts().to_dict()}')
    d_in = d_in[veg]
    g = d_in.groupby(d_in.acq_date.dt.normalize())
    daily = g.size().reindex(want, fill_value=0)
    frp = g.frp.sum().reindex(want, fill_value=0.0)
    daily.index.name = 'date'
    out = pd.DataFrame({'fire_count': daily.astype(int), 'frp_sum': frp.round(2).values}, index=daily.index)
    if out.frp_sum.lt(0).any():
        raise SystemExit('negative FRP sums')
    if out.fire_count.lt(0).any() or out.fire_count.isna().any() or len(out) != len(want):
        raise SystemExit('fire counts failed the checks (negative, missing or wrong length)')
    path = os.path.join(out_dir, 'punjab_fire_daily.csv')
    out.to_csv(path, date_format='%Y-%m-%d')
    yearly = out.fire_count.groupby(out.index.year).sum().to_dict()
    oct_nov = out[out.index.month.isin([10, 11])].fire_count.groupby(lambda t: t.year).sum().to_dict()
    print(f'  wrote {path}: {len(out)} days, {out.index[0].date()} -> {out.index[-1].date()}, '
          f'no missing days, no negative counts')
    print(f'  fires per year: {yearly}')
    print(f'  of which Oct-Nov: {oct_nov}')
    print(f'  days with zero fires: {(out.fire_count == 0).sum()}; max per day {out.fire_count.max()} '
          f'on {out.fire_count.idxmax().date()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pm25', action='store_true')
    ap.add_argument('--era5', action='store_true')
    ap.add_argument('--fire', action='store_true', help='Punjab daily fire counts from raw/firms (FIRMS VIIRS)')
    ap.add_argument('--boundary', default=os.path.join('raw', 'boundaries', 'gadm41_IND_1.json'))
    ap.add_argument('--fire-start', default=None,
                    help='first day of the fire file (default: Jan 1 of the first year); '
                         'use 2018-12-01 so 30-day fire sums exist for the first 2019 origins')
    ap.add_argument('--years', default='2019-2023', help='inclusive year range, e.g. 2019-2023')
    ap.add_argument('--cpcb-dir', default=CPCB_DIR)
    ap.add_argument('--raw-dir', default='raw')
    ap.add_argument('--era5-source', choices=['openmeteo', 'nc'], default='openmeteo')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()
    if not (args.pm25 or args.era5 or args.fire):
        ap.error('pass --pm25, --era5 and/or --fire')
    y0, y1 = (int(x) for x in args.years.split('-'))
    years = list(range(y0, y1 + 1))
    os.makedirs(args.out_dir, exist_ok=True)
    if args.pm25:
        build_pm25(args.cpcb_dir, years, args.out_dir)
    if args.era5:
        # PM2.5 labels start at 20:00 UTC on 31 Dec of the year before; fetch that year too so the
        # first origins have meteorology history.
        build_era5(args, [y0 - 1] + years if args.era5_source == 'openmeteo' else years)
    if args.fire:
        build_fire(os.path.join(args.raw_dir, 'firms'), args.boundary, years, args.out_dir, args.fire_start)


if __name__ == '__main__':
    main()
