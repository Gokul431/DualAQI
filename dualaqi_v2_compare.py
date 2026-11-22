#!/usr/bin/env python3
"""DualAQI v2: development-stage selection and the final test comparison (see results/dualaqi_v2/PLAN.md).

  --stage dev   folds 0-7: every candidate's RMSE (mean over seeds), the selection rule applied,
                writes results/dualaqi_v2/dev_selection.csv and SELECTED.txt
  --stage test  folds 8-15, run once: v2 vs v1, v2 vs single-branch LSTM (early stopping), and
                v1 vs single-branch LSTM v1 for reference; mean +- SD metrics and paired dRMSE CIs,
                overall and Oct-Nov vs rest of year. Writes results/dualaqi_v2/test_*.csv

    .venv\\Scripts\\python dualaqi_v2_compare.py --stage dev
    .venv\\Scripts\\python dualaqi_v2_compare.py --stage test --n-boot 500
"""
import argparse
import os

import numpy as np
import pandas as pd

from dualaqi import HORIZONS
from dualaqi_ablation import SEASONS, drmse, metrics

SEEDS = (0, 1, 2)
R = os.path.join('results', 'dualaqi_v2')
ES_CANDIDATES = {'B': 'v1 + early stopping', 'C': 'B + FRP', 'D': 'B + wind transport',
                 'E': 'B + 14/30-day fire sums', 'F': 'B + FRP + transport + sums'}
# Amendment 1 (PLAN.md): early stopping dropped; v2 is chosen among these, v1 setup (15 epochs)
CANDIDATES = {'v1': 'DualAQI v1 (15 epochs)', 'C2': 'v1 + FRP', 'D2': 'v1 + wind transport',
              'E2': 'v1 + 14/30-day fire sums', 'F2': 'v1 + FRP + transport + sums'}
EVENT_OPTS_OF = {'v1': '', 'C2': 'frp', 'D2': 'transport', 'E2': 'long', 'F2': 'frp,transport,long'}
SIMPLICITY = ['v1', 'C2', 'D2', 'E2', 'F2']  # tie-break order (simpler first)


def load(path, model, folds):
    o = pd.read_csv(path, parse_dates=['origin'])
    o = o[(o.scheme == 'forward') & (o.model == model) & o.fold.isin(folds)].copy()
    return o.sort_values(['horizon', 'origin']).reset_index(drop=True)


def runs_for(spec, folds):
    """spec: (directory pattern with {s}, model name)."""
    return {s: load(spec[0].format(s=s), spec[1], folds) for s in SEEDS}


def check_paired(runs):
    ref = next(iter(next(iter(runs.values())).values()))
    for name, per_seed in runs.items():
        for s, o in per_seed.items():
            if len(o) != len(ref) or not (o[['horizon', 'origin']].values == ref[['horizon', 'origin']].values).all():
                raise SystemExit(f'{name} seed {s}: origins differ; comparison would not be paired')
    return ref


def stage_dev():
    folds = range(0, 8)
    specs = {c: (os.path.join(R, 'dev', c, 'seed{s}', 'oof_predictions.csv.gz'), c)
             for c in list(ES_CANDIDATES) + ['LSTM-ES', 'C2', 'D2', 'E2', 'F2']}
    specs['v1'] = (os.path.join('results', 'dualaqi', 'fire', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI')
    specs['LSTM-v1'] = (os.path.join('results', 'dualaqi', 'nofire', 'seed{s}', 'oof_predictions.csv.gz'),
                        'DualAQI-noFire')
    runs = {k: runs_for(v, folds) for k, v in specs.items()}
    ref = check_paired(runs)
    desc = {**ES_CANDIDATES, **CANDIDATES, 'LSTM-ES': 'single-branch LSTM + early stopping',
            'LSTM-v1': 'single-branch LSTM v1 (15 epochs)'}
    rows = []
    for name, per_seed in runs.items():
        r = {'model': name, 'description': desc[name]}
        per_h = []
        for h in HORIZONS:
            vals = [metrics(o[o.horizon == h].y_true.values, o[o.horizon == h].y_pred.values)['RMSE']
                    for o in per_seed.values()]
            r[f'RMSE_{h}'] = np.mean(vals)
            r[f'RMSE_{h}_sd'] = np.std(vals, ddof=1)
            per_h.append(np.mean(vals))
        r['RMSE_mean_over_horizons'] = np.mean(per_h)
        rows.append(r)
    t = pd.DataFrame(rows)
    cand = t[t.model.isin(CANDIDATES)].copy()
    best = cand.RMSE_mean_over_horizons.min()
    tied = cand[cand.RMSE_mean_over_horizons < best + 0.1].model.tolist()
    chosen = min(tied, key=SIMPLICITY.index)
    t['selected'] = t.model == chosen
    t.to_csv(os.path.join(R, 'dev_selection.csv'), index=False, float_format='%.3f')
    print(f'development folds 0-7: {ref.origin.nunique()} origins '
          f'({ref.origin.min().date()} -> {ref.origin.max().date()}), seeds 0-2, RMSE (AQI) mean over seeds')
    show = t[['model', 'description', 'RMSE_24', 'RMSE_48', 'RMSE_72', 'RMSE_mean_over_horizons', 'selected']]
    print(show.round(2).to_string(index=False))
    with open(os.path.join(R, 'SELECTED.txt'), 'w') as fh:
        fh.write(f'{chosen}\n{CANDIDATES[chosen]}\n')
    print(f'\nselected by the rule (Amendment 1, among {list(CANDIDATES)}): {chosen} ({CANDIDATES[chosen]}); '
          f'candidates within 0.1 of the best: {tied}')
    print(f'event options for the test run: "{EVENT_OPTS_OF[chosen]}"')


def stage_test(n_boot, stage='test'):
    chosen = open(os.path.join(R, 'SELECTED.txt')).read().split('\n')[0]
    if stage == 'test':
        folds, label, prefix = range(8, 16), 'final test, folds 8-15', 'test'
        specs = {
            'DualAQI v2': (os.path.join(R, 'test', 'V2', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI-v2'),
            'DualAQI v1': (os.path.join('results', 'dualaqi', 'fire', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI'),
            'LSTM': (os.path.join('results', 'dualaqi', 'nofire', 'seed{s}', 'oof_predictions.csv.gz'),
                     'DualAQI-noFire'),
        }
        comps = [('DualAQI v2', 'DualAQI v1'), ('DualAQI v2', 'LSTM'), ('DualAQI v1', 'LSTM')]
    else:  # Amendments 2-3: fresh test, every fold after 15 (July 2024 - December 2025)
        folds, label, prefix = range(16, 100), 'fresh test 2024-2025, folds 16+', 'test2024'
        base = os.path.join(R, 'test2024')
        specs = {
            'DualAQI v2': (os.path.join(base, 'V2', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI-v2'),
            'LSTM': (os.path.join(base, 'LSTM', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI-noFire'),
            'DualAQI v1': (os.path.join(base, 'V1', 'seed{s}', 'oof_predictions.csv.gz'), 'DualAQI'),
        }
        comps = [('DualAQI v2', 'LSTM'), ('DualAQI v2', 'DualAQI v1'), ('DualAQI v1', 'LSTM')]
    runs = {k: runs_for(v, folds) for k, v in specs.items()}
    ref = check_paired(runs)
    print(f'{label}: {ref.origin.nunique()} origins ({ref.origin.min().date()} -> '
          f'{ref.origin.max().date()}), folds {sorted(ref.fold.unique())}, seeds 0-2. '
          f'v2 = candidate {chosen} ({CANDIDATES[chosen]}); LSTM = single-branch, same training setup')
    rows, drows = [], []
    for h in HORIZONS:
        mh = ref.horizon.values == h
        months = (ref.origin[mh] + pd.Timedelta(hours=h)).dt.month.values
        for season, sel in SEASONS.items():
            m = sel(months)
            y = ref.y_true.values[mh][m]
            for name, per_seed in runs.items():
                per = pd.DataFrame([metrics(y, per_seed[s].y_pred.values[mh][m]) for s in SEEDS])
                row = dict(horizon=h, season=season, model=name, n=int(m.sum()), n_severe=int(per.n_severe.iloc[0]))
                for c in ['RMSE', 'MAE', 'R2', 'Bias', 'CatAcc', 'POD_severe', 'FAR_severe']:
                    row[f'{c}_mean'], row[f'{c}_sd'] = per[c].mean(), per[c].std(ddof=1)
                rows.append(row)
            for a, b in comps:
                ea = [runs[a][s].y_pred.values[mh][m] - y for s in SEEDS]
                eb = [runs[b][s].y_pred.values[mh][m] - y for s in SEEDS]
                pt, lo, hi = drmse(ea, eb, n_boot, 0, 1)
                _, blo, bhi = drmse(ea, eb, n_boot, 0, 7)
                drows.append(dict(comparison=f'{a} - {b}', horizon=h, season=season, n=int(m.sum()), dRMSE=pt,
                                  CI95_day_lo=lo, CI95_day_hi=hi, CI95_7dblock_lo=blo, CI95_7dblock_hi=bhi,
                                  excludes_zero_day=bool(lo > 0 or hi < 0),
                                  excludes_zero_7dblock=bool(blo > 0 or bhi < 0)))
    met, dr = pd.DataFrame(rows), pd.DataFrame(drows)
    met.to_csv(os.path.join(R, f'{prefix}_metrics.csv'), index=False, float_format='%.4f')
    dr.to_csv(os.path.join(R, f'{prefix}_drmse.csv'), index=False, float_format='%.4f')
    show = met[['season', 'horizon', 'model']].copy()
    for c, f in [('RMSE', 2), ('MAE', 2), ('R2', 3), ('Bias', 1), ('CatAcc', 3), ('POD_severe', 2), ('FAR_severe', 2)]:
        show[c] = [f'{a:.{f}f} ± {b:.{f}f}' for a, b in zip(met[f'{c}_mean'], met[f'{c}_sd'])]
    pd.set_option('display.width', 250)
    for season in SEASONS:
        print(f'\n== {season}: mean ± SD over seeds (n = {int(met[met.season == season].n.iloc[0])} per horizon)')
        print(show[show.season == season].drop(columns='season').to_string(index=False))
    for a, b in comps:
        print(f'\n== paired dRMSE, {a} - {b} (negative = {a} better)')
        print(dr[dr.comparison == f'{a} - {b}'][['season', 'horizon', 'n', 'dRMSE', 'CI95_day_lo', 'CI95_day_hi',
                                                  'CI95_7dblock_lo', 'CI95_7dblock_hi']].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['dev', 'test', 'test2024'], required=True)
    ap.add_argument('--n-boot', type=int, default=500)
    args = ap.parse_args()
    stage_dev() if args.stage == 'dev' else stage_test(args.n_boot, args.stage)


if __name__ == '__main__':
    main()
