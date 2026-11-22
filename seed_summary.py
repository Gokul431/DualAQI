#!/usr/bin/env python3
"""Mean +- SD over seeds 0-2 for every model and metric (results/, results/seed1, results/seed2).

Seed 0 is the main run in results/. Seeds 1 and 2 were run with --no-arima (ARIMA does not use a
seed). Deterministic models (Persistence, Ridge) should show SD 0, which checks that the runs agree.

    .venv\\Scripts\\python seed_summary.py --results results
"""
import argparse
import os

import pandas as pd

METRICS = ['RMSE', 'MAE', 'R2', 'CatAcc', 'skill_RMSE_vs_persist', 'dRMSE_vs_persist_CI95_lo',
           'dRMSE_vs_persist_CI95_hi']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results')
    args = ap.parse_args()
    runs = []
    for seed, sub in [(0, ''), (1, 'seed1'), (2, 'seed2')]:
        b = pd.read_csv(os.path.join(args.results, sub, 'benchmark_long.csv'))
        b['seed'] = seed
        runs.append(b)
    d = pd.concat(runs)
    shared = set.intersection(*[set(r.model) for r in runs])
    d = d[d.model.isin(shared)]
    agg = d.groupby(['scheme', 'horizon', 'model'])[METRICS].agg(['mean', 'std'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    agg['n_seeds'] = d.groupby(['scheme', 'horizon', 'model']).seed.nunique()
    agg = agg.reset_index()
    path = os.path.join(args.results, 'seed_summary.csv')
    agg.to_csv(path, index=False)
    show = agg[['scheme', 'horizon', 'model', 'n_seeds']].copy()
    for m in ['RMSE', 'MAE', 'R2', 'CatAcc', 'skill_RMSE_vs_persist']:
        show[m] = [f'{a:.3f} +- {s:.3f}' for a, s in zip(agg[f'{m}_mean'], agg[f'{m}_std'])]
    print(show.to_string(index=False))
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
