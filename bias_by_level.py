#!/usr/bin/env python3
"""Forecast bias by observed AQI category, over the whole test set (out-of-fold predictions).

Answers: do models (in particular the LSTM, whose sequence branch DualAQI shares) under-forecast more than
others on high-pollution days?

Grouping is by the OBSERVED category. Note: any model trained on squared error, and Persistence too,
shows negative bias in the highest bins and positive bias in the lowest (regression to the mean),
so compare models with each other and with Persistence, not with zero.

Outputs in <results>/bias_by_level/:
    bias_by_level.csv            scheme, horizon, model, category, n, bias, MAE, share under-forecast
    bias_by_level_lstm_seeds.csv LSTM bias per category, mean +- SD over seeds 0-2 (if seed runs exist)
    bias_by_level.png / .pdf     forward CV, bias per category, one panel per horizon

    .venv\\Scripts\\python bias_by_level.py --results results
"""
import argparse
import os

import numpy as np
import pandas as pd

from dualaqi import HORIZONS, aqi_category

CATS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very poor', 'Severe']
ORDER = ['Persistence', 'ARIMA', 'Ridge', 'GBM', 'RandomForest', 'XGBoost', 'Ensemble', 'LSTM']
PLOT_MODELS = [('Persistence', '#8f8d86', True), ('Ensemble', '#2a78d6', False),
               ('LSTM', '#eb6834', False), ('ARIMA', '#1baf7a', False)]
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def by_level(o):
    o = o.assign(err=o.y_pred - o.y_true, cat=aqi_category(o.y_true.values))
    g = o.groupby(['scheme', 'horizon', 'model', 'cat'])
    t = pd.DataFrame({'n': g.size(), 'bias': g.err.mean(), 'MAE': g.err.apply(lambda e: e.abs().mean()),
                      'share_under': g.err.apply(lambda e: (e < 0).mean()),
                      'mean_observed': g.y_true.mean()}).reset_index()
    t['category'] = t.cat.map(dict(enumerate(CATS)))
    return t


def plot(t, seeds, path_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    f = t[t.scheme == 'forward']
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(9.6, 3.6), sharey=True, facecolor=SURFACE)
    x = np.arange(len(CATS))
    for ax, h in zip(axes, HORIZONS):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if seeds is not None:
            s = seeds[(seeds.scheme == 'forward') & (seeds.horizon == h)].set_index('cat').reindex(range(len(CATS)))
            ax.fill_between(x, s.bias_min, s.bias_max, color='#eb6834', alpha=0.18, lw=0, zorder=2,
                            label='LSTM seed range')
        for model, colour, dashed in PLOT_MODELS:
            s = f[(f.horizon == h) & (f.model == model)].set_index('cat').reindex(range(len(CATS)))
            ax.plot(x, s.bias, color=colour, lw=1.6, ls=(0, (4, 2)) if dashed else '-', marker='o', ms=4,
                    label=model, zorder=3)
        ax.set_title(f'{h} h ahead', loc='left', fontsize=9.5, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(CATS, rotation=35, ha='right')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('Bias (forecast - observed), AQI')
    n = f[(f.horizon == HORIZONS[0]) & (f.model == 'Persistence')].set_index('cat').n.reindex(range(len(CATS)))
    fig.text(0.5, -0.06, 'Observed AQI category (forecasts per category: '
             + ', '.join(f'{c} {int(v) if v == v else 0}' for c, v in zip(CATS, n)) + ')',
             ha='center', fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.suptitle('Forecast bias by observed AQI category, forward CV 2020-2023', x=0.01, ha='left', y=1.1,
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_base}.{ext}', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results')
    args = ap.parse_args()
    out = os.path.join(args.results, 'bias_by_level')
    os.makedirs(out, exist_ok=True)

    t = by_level(pd.read_csv(os.path.join(args.results, 'oof_predictions.csv.gz')))
    t = t.sort_values(['scheme', 'horizon', 'cat', 'model'], key=lambda c: c.map({m: i for i, m in enumerate(ORDER)})
                      if c.name == 'model' else c)
    t.drop(columns='cat').to_csv(os.path.join(out, 'bias_by_level.csv'), index=False, float_format='%.3f')

    seeds = None
    paths = [os.path.join(args.results, 'oof_predictions.csv.gz')] + \
            [os.path.join(args.results, f'seed{s}', 'oof_predictions.csv.gz') for s in (1, 2)]
    if all(os.path.exists(p) for p in paths):
        per = []
        for seed, p in enumerate(paths):
            o = pd.read_csv(p)
            per.append(by_level(o[o.model == 'LSTM']).assign(seed=seed))
        d = pd.concat(per)
        seeds = d.groupby(['scheme', 'horizon', 'cat']).bias.agg(['mean', 'std', 'min', 'max'])
        seeds.columns = ['bias_mean', 'bias_sd', 'bias_min', 'bias_max']
        seeds = seeds.reset_index()
        seeds['n'] = d[d.seed == 0].set_index(['scheme', 'horizon', 'cat']).n.reindex(
            pd.MultiIndex.from_frame(seeds[['scheme', 'horizon', 'cat']])).values
        seeds['category'] = seeds.cat.map(dict(enumerate(CATS)))
        seeds.drop(columns='cat').to_csv(os.path.join(out, 'bias_by_level_lstm_seeds.csv'), index=False,
                                         float_format='%.3f')
    plot(t, seeds, os.path.join(out, 'bias_by_level'))

    f = t[t.scheme == 'forward']
    for h in HORIZONS:
        p = f[f.horizon == h].pivot(index='model', columns='cat', values='bias').reindex(ORDER)
        p.columns = [CATS[c] for c in p.columns]
        print(f'\nforward CV, {h} h: bias by observed category (AQI units)')
        print(p.round(1).to_string())
        if seeds is not None:
            s = seeds[(seeds.scheme == 'forward') & (seeds.horizon == h)]
            print('LSTM seeds 0-2: ' + ', '.join(f'{CATS[c]} {m:.1f} +- {sd:.1f}' for c, m, sd in
                                                  zip(s.cat, s.bias_mean, s.bias_sd)))
    n = f[(f.horizon == 24) & (f.model == 'Persistence')].set_index('category').n
    print('\nforecasts per observed category (forward, each horizon):', n.to_dict())
    print(f'\nwrote {out}/bias_by_level.csv, bias_by_level_lstm_seeds.csv, bias_by_level.png/.pdf')


if __name__ == '__main__':
    main()
