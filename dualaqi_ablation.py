#!/usr/bin/env python3
"""Ablation: DualAQI (sequence + fire branch) vs the single-branch LSTM (--no-fire), forward CV,
seeds 0-2, plus the 2023 case-study export.

Inputs:  results/dualaqi/{fire,nofire}/seed{0,1,2}/oof_predictions.csv.gz
         results/case_study/case_study_forecasts.csv          (existing models, for step 5)
Outputs: results/dualaqi/ablation_metrics.csv     mean +- SD over seeds, by horizon and season
         results/dualaqi/ablation_drmse.csv       paired dRMSE (DualAQI - single-branch) with 95% CIs
         results/case_study_dualaqi_2023.csv      case-study forecasts with DualAQI columns added
         results/case_study_dualaqi_2023_metrics.csv

Season is set by the target (valid) date: Oct-Nov vs the rest of the year.
Severe = AQI >= 400 (CPCB category Severe).

dRMSE confidence intervals (n_boot resamples, same resample for every seed, mean over seeds):
  * day bootstrap: resamples days (one origin per day), as in dualaqi.py's drmse_ci;
  * 7-day block bootstrap: resamples blocks of 7 consecutive days, which allows for day-to-day
    correlation of forecast errors. This is the more conservative interval.

    .venv\\Scripts\\python dualaqi_ablation.py --n-boot 500
"""
import argparse
import os

import numpy as np
import pandas as pd

from dualaqi import HORIZONS, aqi_category
from case_study import episode_metrics

SEEDS = (0, 1, 2)
VARIANTS = {'DualAQI': 'fire', 'DualAQI-noFire': 'nofire', 'DualAQI-calendarOnly': 'calendar'}
LABEL = {'DualAQI': 'DualAQI', 'DualAQI-noFire': 'Single-branch LSTM (--no-fire)',
         'DualAQI-calendarOnly': 'DualAQI, calendar-only event branch'}
COMPARISONS = [  # (a, b, meaning of a - b)
    ('DualAQI', 'DualAQI-noFire', 'event branch (spec ablation): DualAQI - single-branch'),
    ('DualAQI', 'DualAQI-calendarOnly', 'fire counts alone: DualAQI - calendar-only'),
    ('DualAQI-calendarOnly', 'DualAQI-noFire', 'calendar inputs alone: calendar-only - single-branch'),
]
SEASONS = {'all': lambda m: np.ones(len(m), bool), 'Oct-Nov': lambda m: np.isin(m, [10, 11]),
           'rest of year': lambda m: ~np.isin(m, [10, 11])}


VERSION = 'v1'   # set by --version; v2 = the final model selected in results/dualaqi_v2/PLAN.md
FOLDS = None     # set by --folds, e.g. 8-15 for the clean test years only


def load(results, model, seed):
    if model == 'DualAQI' and VERSION == 'v2':
        # v2 = v1 + FRP: development folds 0-7 (candidate C2) + test folds 8-15 (V2), same seeds and folds
        base = os.path.join(results, 'dualaqi_v2')
        a = pd.read_csv(os.path.join(base, 'dev', 'C2', f'seed{seed}', 'oof_predictions.csv.gz'), parse_dates=['origin'])
        b = pd.read_csv(os.path.join(base, 'test', 'V2', f'seed{seed}', 'oof_predictions.csv.gz'), parse_dates=['origin'])
        o = pd.concat([a[(a.model == 'C2') & a.fold.isin(range(8))], b[(b.model == 'DualAQI-v2') & b.fold.isin(range(8, 16))]])
        o = o[o.scheme == 'forward'].assign(model='DualAQI').copy()
    else:
        p = os.path.join(results, 'dualaqi', VARIANTS[model], f'seed{seed}', 'oof_predictions.csv.gz')
        o = pd.read_csv(p, parse_dates=['origin'])
        o = o[(o.scheme == 'forward') & (o.model == model)].copy()
    if FOLDS is not None:
        o = o[o.fold.isin(FOLDS)]
    o['valid'] = o.origin + pd.to_timedelta(o.horizon, unit='h')
    return o.sort_values(['horizon', 'origin']).reset_index(drop=True)


def metrics(y, p):
    e = p - y
    cy, cp = aqi_category(y), aqi_category(p)
    ev, fc = cy >= 5, cp >= 5
    return dict(n=len(y), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), Bias=np.mean(e), CatAcc=np.mean(cy == cp),
                POD_severe=(ev & fc).sum() / ev.sum() if ev.sum() else np.nan,
                FAR_severe=(~ev & fc).sum() / fc.sum() if fc.sum() else np.nan, n_severe=int(ev.sum()))


def boot_idx(n, rng, block):
    if block == 1:
        return rng.integers(0, n, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def drmse(errs_d, errs_n, n_boot, seed, block):
    """errs_*: list over seeds of error arrays on the same origins. Returns point, lo, hi."""
    se_d = np.stack([e ** 2 for e in errs_d])
    se_n = np.stack([e ** 2 for e in errs_n])
    point = np.mean(np.sqrt(se_d.mean(1)) - np.sqrt(se_n.mean(1)))
    rng = np.random.default_rng(seed)
    n = se_d.shape[1]
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = boot_idx(n, rng, block)
        b[i] = np.mean(np.sqrt(se_d[:, idx].mean(1)) - np.sqrt(se_n[:, idx].mean(1)))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results')
    ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--case-start', default='2023-10-20')
    ap.add_argument('--case-end', default='2023-11-20')
    ap.add_argument('--version', choices=['v1', 'v2'], default='v1', help='DualAQI version to ablate')
    ap.add_argument('--folds', default=None, help='inclusive fold range, e.g. 8-15 (default: all forward folds)')
    args = ap.parse_args()
    global VERSION, FOLDS
    VERSION = args.version
    if args.folds:
        a_, b_ = (int(x) for x in args.folds.split('-'))
        FOLDS = list(range(a_, b_ + 1))
    out_dir = os.path.join(args.results, 'dualaqi')
    suffix = '' if (VERSION == 'v1' and FOLDS is None) else f'_{VERSION}' + (f'_folds{args.folds}' if FOLDS else '')

    runs = {(m, s): load(args.results, m, s) for m in VARIANTS for s in SEEDS}
    ref = runs[('DualAQI', 0)][['horizon', 'origin', 'y_true']]
    for k, o in runs.items():
        if not (o[['horizon', 'origin']].values == ref[['horizon', 'origin']].values).all():
            raise SystemExit(f'{k}: origins differ from DualAQI seed 0; the comparison would not be paired')
        if not np.allclose(o.y_true.values, ref.y_true.values):
            raise SystemExit(f'{k}: targets differ')
    print(f'paired check: all 6 runs share the same {ref.origin.nunique()} forward-CV origins and targets '
          f'({ref.origin.min().date()} -> {ref.origin.max().date()})')

    rows, drows = [], []
    for h in HORIZONS:
        mh = ref.horizon.values == h
        months = (ref.origin[mh] + pd.Timedelta(hours=h)).dt.month.values
        for season, sel in SEASONS.items():
            m = sel(months)
            y = ref.y_true.values[mh][m]
            for model in VARIANTS:
                per = pd.DataFrame([metrics(y, runs[(model, s)].y_pred.values[mh][m]) for s in SEEDS])
                row = dict(horizon=h, season=season, model=LABEL[model], n=int(per.n.iloc[0]),
                           n_severe=int(per.n_severe.iloc[0]))
                for c in ['RMSE', 'MAE', 'R2', 'Bias', 'CatAcc', 'POD_severe', 'FAR_severe']:
                    row[f'{c}_mean'], row[f'{c}_sd'] = per[c].mean(), per[c].std(ddof=1)
                rows.append(row)
            for a, b, meaning in COMPARISONS:
                ed = [runs[(a, s)].y_pred.values[mh][m] - y for s in SEEDS]
                en = [runs[(b, s)].y_pred.values[mh][m] - y for s in SEEDS]
                pt, lo, hi = drmse(ed, en, args.n_boot, 0, 1)
                _, blo, bhi = drmse(ed, en, args.n_boot, 0, 7)
                per_seed = [float(np.sqrt(np.mean(x ** 2)) - np.sqrt(np.mean(z ** 2))) for x, z in zip(ed, en)]
                drows.append(dict(comparison=meaning, horizon=h, season=season, n=int(m.sum()), dRMSE=pt,
                                  CI95_day_lo=lo, CI95_day_hi=hi, CI95_7dblock_lo=blo, CI95_7dblock_hi=bhi,
                                  dRMSE_seed0=per_seed[0], dRMSE_seed1=per_seed[1], dRMSE_seed2=per_seed[2],
                                  excludes_zero_day=bool(lo > 0 or hi < 0),
                                  excludes_zero_7dblock=bool(blo > 0 or bhi < 0)))
    met, dr = pd.DataFrame(rows), pd.DataFrame(drows)
    met.to_csv(os.path.join(out_dir, f'ablation_metrics{suffix}.csv'), index=False, float_format='%.4f')
    dr.to_csv(os.path.join(out_dir, f'ablation_drmse{suffix}.csv'), index=False, float_format='%.4f')
    print(f'wrote {out_dir}/ablation_metrics{suffix}.csv and ablation_drmse{suffix}.csv (DualAQI {VERSION})')

    show = met[['season', 'horizon', 'model', 'n']].copy()
    for c, f in [('RMSE', 2), ('MAE', 2), ('R2', 3), ('Bias', 2), ('CatAcc', 3), ('POD_severe', 3), ('FAR_severe', 3)]:
        show[c] = [f'{a:.{f}f} ± {b:.{f}f}' for a, b in zip(met[f'{c}_mean'], met[f'{c}_sd'])]
    pd.set_option('display.width', 250)
    for season in SEASONS:
        print(f'\n== {season}: mean ± SD over seeds 0-2 (forward CV)')
        print(show[show.season == season].drop(columns='season').to_string(index=False))
    for _, _, meaning in COMPARISONS:
        print(f'\n== paired dRMSE, {meaning} (negative = first model better), mean over seeds')
        print(dr[dr.comparison == meaning][['season', 'horizon', 'n', 'dRMSE', 'CI95_day_lo', 'CI95_day_hi',
                                            'CI95_7dblock_lo', 'CI95_7dblock_hi', 'dRMSE_seed0', 'dRMSE_seed1',
                                            'dRMSE_seed2']].round(2).to_string(index=False))

    if suffix:   # the 2023 case-study export (with v1 and v2 columns) is maintained separately
        return
    # ---- step 5: 2023 case study export
    cs = pd.read_csv(os.path.join(args.results, 'case_study', 'case_study_forecasts.csv'),
                     parse_dates=['valid', 'origin'])
    start, end = pd.Timestamp(args.case_start), pd.Timestamp(args.case_end)
    for model in VARIANTS:
        o = runs[(model, 0)]
        o = o[(o.valid >= start) & (o.valid < end + pd.Timedelta(days=1))]
        cs = cs.merge(o[['valid', 'horizon', 'origin', 'y_pred']].rename(columns={'y_pred': model}),
                      on=['valid', 'horizon', 'origin'], how='left')
        obs = o.set_index(['valid', 'horizon']).y_true
        chk = cs.set_index(['valid', 'horizon']).observed.reindex(obs.index)
        if not np.allclose(chk.values, obs.values, atol=0.01):
            raise SystemExit('case study: observed values differ between files')
    if cs[list(VARIANTS)].isna().any().any():
        raise SystemExit('case study: DualAQI forecasts missing for some rows')
    path = os.path.join(args.results, 'case_study_dualaqi_2023.csv')
    cs.sort_values(['horizon', 'valid']).to_csv(path, index=False, float_format='%.2f')
    long = cs.melt(id_vars=['valid', 'horizon', 'origin', 'observed'], var_name='model', value_name='y_pred')
    long = long.rename(columns={'observed': 'y_true'})
    cm = episode_metrics(long, start, end)
    order = ['Persistence', 'ARIMA', 'Ridge', 'GBM', 'RandomForest', 'XGBoost', 'Ensemble', 'LSTM',
             'DualAQI-noFire', 'DualAQI-calendarOnly', 'DualAQI']
    cm = cm.sort_values(['horizon', 'model'], key=lambda c: c.map({m: i for i, m in enumerate(order)})
                        if c.name == 'model' else c)
    cm.to_csv(os.path.join(args.results, 'case_study_dualaqi_2023_metrics.csv'), index=False)
    print(f'\n== case study {start.date()} -> {end.date()} (valid dates), seed 0')
    print(cm[['horizon', 'model', 'n', 'RMSE', 'Bias', 'CatAcc', 'POD_severe', 'FAR_severe', 'peak_obs',
              'peak_pred']].round(3).to_string(index=False))
    print(f'\nwrote {out_dir}/ablation_metrics.csv, ablation_drmse.csv, {path}, '
          f'{args.results}/case_study_dualaqi_2023_metrics.csv')


if __name__ == '__main__':
    main()
