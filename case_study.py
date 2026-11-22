#!/usr/bin/env python3
"""Case study: the most polluted 30-day episode, from the out-of-fold predictions.

* Uses forward-CV predictions (every forecast is made with a model trained only on earlier data).
* The episode is chosen from the data, not by hand: the window of --days consecutive valid dates
  with the highest mean observed AQI at the 24 h horizon.
* Outputs in <out-dir>/case_study/:
    case_study_timeseries.png / .pdf   observed vs forecast AQI for each horizon
    case_study_metrics.csv             per model and horizon, inside the episode
    case_study_summary.md              episode definition and the metrics table
* If results/seed1 and results/seed2 exist, LSTM metrics are also reported as mean +- SD over seeds.

    .venv\\Scripts\\python case_study.py --results results
"""
import argparse
import os

import numpy as np
import pandas as pd

from dualaqi import HORIZONS, aqi_category

CATS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very poor', 'Severe']
PLOT_MODELS = [  # (model, colour, dashed); light-mode reference palette slots 1-3, baseline in grey
    ('Persistence', '#8f8d86', True),
    ('Ensemble', '#2a78d6', False),
    ('LSTM', '#eb6834', False),
    ('ARIMA', '#1baf7a', False),
]
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load_oof(path, scheme):
    o = pd.read_csv(path, parse_dates=['origin'])
    o = o[o.scheme == scheme].copy()
    o['valid'] = o.origin + pd.to_timedelta(o.horizon, unit='h')
    return o


def pick_episode(o, days):
    obs = o[o.horizon == HORIZONS[0]].drop_duplicates('valid').set_index('valid').y_true.sort_index()
    daily = obs.groupby(obs.index.normalize()).mean()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq='D'))
    roll = full.rolling(days, min_periods=int(days * 0.9)).mean()
    end = roll.idxmax()
    return end - pd.Timedelta(days=days - 1), end, roll.max()


def event_scores(y, p, threshold_cat):
    ev, fc = aqi_category(y) >= threshold_cat, aqi_category(p) >= threshold_cat
    hits = np.sum(ev & fc)
    pod = hits / ev.sum() if ev.sum() else np.nan
    far = np.sum(~ev & fc) / fc.sum() if fc.sum() else np.nan
    return pod, far, int(ev.sum())


def episode_metrics(o, start, end):
    m = o[(o.valid >= start) & (o.valid < end + pd.Timedelta(days=1))]
    rows = []
    for (model, h), g in m.groupby(['model', 'horizon']):
        y, p = g.y_true.values, g.y_pred.values
        e = p - y
        pod_s, far_s, n_s = event_scores(y, p, 5)
        pod_v, far_v, n_v = event_scores(y, p, 4)
        rows.append(dict(model=model, horizon=h, n=len(y), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                         Bias=np.mean(e), CatAcc=np.mean(aqi_category(y) == aqi_category(p)),
                         POD_severe=pod_s, FAR_severe=far_s, n_severe=n_s,
                         POD_vpoor_plus=pod_v, FAR_vpoor_plus=far_v, n_vpoor_plus=n_v,
                         peak_obs=y.max(), peak_pred=p.max()))
    return pd.DataFrame(rows)


def plot(o, start, end, path_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    pad = pd.Timedelta(days=3)
    w = o[(o.valid >= start - pad) & (o.valid < end + pad + pd.Timedelta(days=1))]
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(7.2, 7.6), sharex=True, sharey=True,
                             facecolor=SURFACE)
    ymax = max(w.y_true.max(), w[w.model.isin([m for m, _, _ in PLOT_MODELS])].y_pred.max()) * 1.06
    for ax, h in zip(axes, HORIZONS):
        ax.set_facecolor(SURFACE)
        g = w[w.horizon == h]
        ax.axvspan(start, end + pd.Timedelta(days=1), color='#f0efec', zorder=0, lw=0)
        for level in (200, 300, 400):  # CPCB category boundaries
            ax.axhline(level, color=GRID, lw=0.8, zorder=1)
        for mid, name in [(250, 'Poor'), (350, 'Very poor'), (450, 'Severe')]:  # name inside its band
            ax.text(1.005, mid, name, transform=ax.get_yaxis_transform(), va='center', ha='left',
                    fontsize=7.5, color=INK2)
        obs = g.drop_duplicates('valid').sort_values('valid')
        ax.plot(obs.valid, obs.y_true, color=INK, lw=1.8, label='Observed', zorder=4)
        for model, colour, dashed in PLOT_MODELS:
            s = g[g.model == model].sort_values('valid')
            if len(s):
                ax.plot(s.valid, s.y_pred, color=colour, lw=1.5, ls=(0, (4, 2)) if dashed else '-',
                        label=model, zorder=3)
        ax.set_title(f'{h} h ahead', loc='left', fontsize=9.5, color=INK, pad=4)
        ax.set_ylim(0, ymax)
        ax.grid(False)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel('AQI')
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[-1].set_xlabel(f'Valid date (forecast for 08:30 IST), {start.year}')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f'Case study: {start:%d %b} - {end:%d %b %Y} (shaded), forward-CV forecasts',
                 x=0.02, ha='left', y=0.995, fontsize=10.5, color=INK)
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.94, 0.958))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_base}.{ext}', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def seed_rows(results, scheme, start, end):
    paths = [os.path.join(results, 'oof_predictions.csv.gz')] + \
            [os.path.join(results, f'seed{s}', 'oof_predictions.csv.gz') for s in (1, 2)]
    if not all(os.path.exists(p) for p in paths):
        return None
    per_seed = []
    for seed, p in enumerate(paths):
        o = load_oof(p, scheme)
        m = episode_metrics(o[o.model == 'LSTM'], start, end)
        m['seed'] = seed
        per_seed.append(m)
    d = pd.concat(per_seed)
    cols = ['RMSE', 'MAE', 'Bias', 'CatAcc', 'POD_severe', 'FAR_severe', 'POD_vpoor_plus', 'FAR_vpoor_plus']
    agg = d.groupby('horizon')[cols].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    return agg.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results')
    ap.add_argument('--scheme', default='forward', choices=['forward', 'blocked'])
    ap.add_argument('--days', type=int, default=30, help='window length when the episode is picked automatically')
    ap.add_argument('--start', help='fixed first valid date, e.g. 2023-10-20 (use with --end)')
    ap.add_argument('--end', help='fixed last valid date, inclusive, e.g. 2023-11-20')
    args = ap.parse_args()
    out = os.path.join(args.results, 'case_study')
    os.makedirs(out, exist_ok=True)

    o = load_oof(os.path.join(args.results, 'oof_predictions.csv.gz'), args.scheme)
    if args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        obs24 = o[(o.horizon == 24) & (o.valid >= start) & (o.valid < end + pd.Timedelta(days=1))]
        mean_aqi = obs24.drop_duplicates('valid').y_true.mean()
        how = 'fixed by the user'
    elif args.start or args.end:
        ap.error('give both --start and --end')
    else:
        start, end, mean_aqi = pick_episode(o, args.days)
        how = f'chosen as the {args.days}-day window with the highest mean observed AQI over all test dates'
    win = o[(o.valid >= start) & (o.valid < end + pd.Timedelta(days=1))]
    expected_days = (end - start).days + 1
    for h in HORIZONS:
        n = win[win.horizon == h].valid.nunique()
        if n < expected_days:
            print(f'NOTE: {h} h horizon has forecasts for {n} of {expected_days} valid days in the window')
    export = win.pivot_table(index=['valid', 'horizon', 'origin'], columns='model', values='y_pred').reset_index()
    export.insert(3, 'observed', win.groupby(['valid', 'horizon']).y_true.first().values)
    export.columns.name = None
    export.sort_values(['horizon', 'valid']).to_csv(os.path.join(out, 'case_study_forecasts.csv'),
                                                     index=False, float_format='%.2f')
    met = episode_metrics(o, start, end)
    order = {m: i for i, m in enumerate(['Persistence', 'ARIMA', 'Ridge', 'GBM', 'RandomForest', 'XGBoost',
                                         'Ensemble', 'LSTM'])}
    met = met.sort_values(['horizon', 'model'], key=lambda c: c.map(order) if c.name == 'model' else c)
    met.to_csv(os.path.join(out, 'case_study_metrics.csv'), index=False)
    plot(o, start, end, os.path.join(out, 'case_study_timeseries'))
    seeds = seed_rows(args.results, args.scheme, start, end)
    if seeds is not None:
        seeds.to_csv(os.path.join(out, 'case_study_lstm_seeds.csv'), index=False)

    obs = o[(o.horizon == 24) & (o.valid >= start) & (o.valid < end + pd.Timedelta(days=1))].drop_duplicates('valid')
    cat_counts = pd.Series(aqi_category(obs.y_true.values)).map(dict(enumerate(CATS))).value_counts()
    lines = [
        f'# Case study: {start:%d %b %Y} to {end:%d %b %Y}',
        '',
        f'- Scheme: {args.scheme} CV (each forecast made by a model trained only on earlier data).',
        f'- Episode: {start:%Y-%m-%d} to {end:%Y-%m-%d} inclusive (valid dates), {how}. '
        f'Mean observed AQI {mean_aqi:.0f}.',
        f'- Forecast origin 03:00 UTC (08:30 IST); one forecast per day per horizon; {len(obs)} valid days.',
        '- Observed AQI categories in the episode: ' + ', '.join(f'{k} {v}' for k, v in cat_counts.items()) + '.',
        f'- Observed AQI range: {obs.y_true.min():.0f} to {obs.y_true.max():.0f}.',
        '- AQI here is the dualaqi.py mapping applied to the hourly city PM2.5 at the valid time '
        '(not a 24-h average).',
        '- POD = share of observed events that were forecast; FAR = share of forecast events that did not occur. '
        'Severe = AQI category Severe; Very poor+ = Very poor or Severe.',
        '',
        '## Metrics inside the episode',
        '',
    ]
    show = met[['horizon', 'model', 'n', 'RMSE', 'MAE', 'Bias', 'CatAcc', 'POD_vpoor_plus', 'FAR_vpoor_plus',
                'POD_severe', 'FAR_severe']]
    lines.append(show.round(3).to_markdown(index=False))
    if seeds is not None:
        lines += ['', '## LSTM over seeds 0-2 (mean +- SD)', '']
        t = pd.DataFrame({'horizon': seeds.horizon})
        for c in ['RMSE', 'MAE', 'CatAcc', 'POD_severe']:
            t[c] = [f'{m:.3f} +- {s:.3f}' for m, s in zip(seeds[f'{c}_mean'], seeds[f'{c}_std'])]
        lines.append(t.to_markdown(index=False))
    with open(os.path.join(out, 'case_study_summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwrote {out}/case_study_timeseries.png/.pdf, case_study_forecasts.csv, case_study_metrics.csv, '
          f'case_study_summary.md')


if __name__ == '__main__':
    main()
