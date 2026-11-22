#!/usr/bin/env python3
"""Objective 4 (case study, 20 Oct - 20 Nov 2023) figures: 6 PNGs, 300 dpi.
DualAQI v2 is the reported model; neural models are shown as the mean of seeds 0-2.

    .venv\\Scripts\\python objective4_figures.py      -> results/figures/objective4/*.png
"""
import glob
import os

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import dualaqi as D
from case_study import episode_metrics

OUT = os.path.join('results', 'figures', 'objective4')
os.makedirs(OUT, exist_ok=True)
S, E = pd.Timestamp('2023-10-20'), pd.Timestamp('2023-11-20')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8d86', '#e4e3df', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
CAT_COL = ['#2e9e4f', '#9ccc5a', '#f2d94e', '#f0a03c', '#e0453a', '#8c1d18']
CATS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very poor', 'Severe']
H = [24, 48, 72]
mpl.rcParams.update({
    'savefig.dpi': 300, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.family': 'DejaVu Sans',
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 8, 'axes.labelcolor': INK2, 'axes.edgecolor': '#b9b7b0', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 8.5, 'lines.linewidth': 1.6, 'text.color': INK})


def header(fig, title, sub=None, top=None):
    fig.text(0.01, 0.995, title, ha='left', va='top', fontsize=12.5, fontweight='bold')
    if sub:
        fig.text(0.01, 0.948, sub, ha='left', va='top', fontsize=9, color=INK2)
    if top:
        fig.subplots_adjust(top=top)


def save(fig, name):
    path = os.path.join(OUT, f'{name}.png')
    fig.savefig(path, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print('wrote', path)


def aqi_bands(ax, top, labels=True):
    edges = [0, 50, 100, 200, 300, 400, 1000]
    for i in range(6):
        lo, hi = edges[i], min(edges[i + 1], top)
        if lo >= top:
            break
        ax.axhspan(lo, hi, color=CAT_COL[i], alpha=0.09, lw=0, zorder=0)
        if labels and hi - lo > 0.05 * top:
            ax.text(1.005, (lo + hi) / 2, CATS[i], transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, color=INK2)


# ----------------------------------------------------------------------------- forecasts
def window(o):
    o = o.copy()
    o['valid'] = o.origin + pd.to_timedelta(o.horizon, unit='h')
    return o[(o.valid >= S) & (o.valid < E + pd.Timedelta(days=1))][['valid', 'horizon', 'origin', 'y_true', 'y_pred']]


def read(path, model):
    o = pd.read_csv(path, parse_dates=['origin'])
    return window(o[(o.scheme == 'forward') & (o.model == model)])


NEURAL = {   # label: [(path, model name) per seed]
    'DualAQI v2': [(f'results/dualaqi_v2/test/V2/seed{s}/oof_predictions.csv.gz', 'DualAQI-v2') for s in range(3)],
    'LSTM': [('results/oof_predictions.csv.gz', 'LSTM')] +
            [(f'results/seed{s}/oof_predictions.csv.gz', 'LSTM') for s in (1, 2)],
    'Single-branch': [(f'results/dualaqi/nofire/seed{s}/oof_predictions.csv.gz', 'DualAQI-noFire') for s in range(3)],
    'Calendar-only': [(f'results/dualaqi/calendar/seed{s}/oof_predictions.csv.gz', 'DualAQI-calendarOnly') for s in range(3)],
}
SEEDS = {k: [read(p, m) for p, m in v] for k, v in NEURAL.items()}
cs = pd.read_csv('results/case_study_dualaqi_2023.csv', parse_dates=['valid', 'origin'])
OTHERS = ['Persistence', 'ARIMA', 'Ridge', 'GBM', 'RandomForest', 'XGBoost', 'Ensemble']
FC = cs[['valid', 'horizon', 'origin', 'observed'] + OTHERS].copy()   # one forecast table: neural = seed mean
for k, runs in SEEDS.items():
    m = runs[0][['valid', 'horizon', 'origin']].copy()
    m[k] = np.mean([r.y_pred.values for r in runs], axis=0)
    FC = FC.merge(m, on=['valid', 'horizon', 'origin'])
assert len(FC) == 96
MODELS = OTHERS + list(NEURAL)
COL = {'DualAQI v2': BLUE, 'LSTM': ORANGE, 'Ensemble': YELLOW, 'Persistence': '#5f5d57', 'XGBoost': VIOLET,
       'ARIMA': MAGENTA, 'Single-branch': '#c9a27a', 'Calendar-only': AQUA, 'RandomForest': '#7d7b74', 'GBM': '#a3a198',
       'Ridge': '#c9c7c0'}

# metrics: deterministic/tree models from seed 0, neural = mean over seeds
rows = []
for m in OTHERS:
    t = episode_metrics(cs.rename(columns={'observed': 'y_true', m: 'y_pred'})[['valid', 'horizon', 'origin', 'y_true', 'y_pred']]
                        .assign(model=m), S, E)
    rows.append(t)
for k, runs in SEEDS.items():
    t = pd.concat([episode_metrics(r.assign(model=k), S, E) for r in runs])
    rows.append(t.groupby(['model', 'horizon'], as_index=False).mean(numeric_only=True))
MET = pd.concat(rows, ignore_index=True).set_index(['model', 'horizon'])


# ----------------------------------------------------------------------------- 1 context
def fig1():
    pm = pd.read_csv('data/pm25_hourly.csv', parse_dates=['datetime'], index_col='datetime').pm25
    met = pd.read_csv('data/era5_hourly.csv', parse_dates=['datetime'], index_col='datetime')
    fire = pd.read_csv('data/punjab_fire_daily.csv', parse_dates=['date'], index_col='date').fire_count
    a0, a1 = pd.Timestamp('2023-10-10'), pd.Timestamp('2023-11-26')
    ist = pd.Timedelta(hours=5, minutes=30)
    p = pm.copy(); p.index = p.index + ist
    p = p.loc[a0:a1]
    aqi = pd.Series(D.pm25_to_aqi(p.values), index=p.index)
    m = met.copy(); m.index = m.index + ist; m = m.loc[a0:a1]
    rf = []
    for f in glob.glob('Delhi PM2.5 Air Quality/2023/site_*_1Hr.csv'):
        d = pd.read_csv(f, usecols=['Timestamp', 'RF (mm)'], na_values=['NA'])
        rf.append(d.set_index(pd.to_datetime(d.Timestamp))['RF (mm)'])
    rain = pd.concat(rf, axis=1).loc[a0:a1]
    rain_daily = rain.resample('D').sum(min_count=1).mean(axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 0.8, 0.8, 0.7], hspace=0.32))
    a, b, c, d = axes
    a.plot(aqi.index, aqi.values, color='#b9b7b0', lw=0.6, label='hourly')
    dm = aqi.resample('D').mean()
    a.plot(dm.index + pd.Timedelta(hours=12), dm.values, color=INK, lw=1.8, marker='o', ms=3, label='daily mean')
    a.set_ylim(0, 700); aqi_bands(a, 700)
    a.set_ylabel('AQI'); a.legend(loc='upper left', ncol=2)
    a.set_title('a  Delhi city-mean AQI (36 CPCB stations)')
    b.bar(fire.loc[a0:a1].index + pd.Timedelta(hours=12), fire.loc[a0:a1].values, width=0.8, color=ORANGE)
    b.set_ylabel('detections'); b.set_title('b  Punjab fire detections per day (VIIRS)')
    c.plot(m.index, m.blh, color=BLUE, lw=0.9)
    c.set_ylabel('m'); c.set_title('c  Boundary-layer height at Delhi (ERA5, hourly)')
    d.bar(rain_daily.index + pd.Timedelta(hours=12), rain_daily.values, width=0.8, color=AQUA)
    d.set_ylabel('mm/day'); d.set_title('d  Rainfall, mean of CPCB station gauges')
    for ax in axes:
        ax.axvspan(S, E + pd.Timedelta(days=1), color='#eef5fd', zorder=0, lw=0)
        for t, lab in [(pd.Timestamp('2023-11-10 05:00'), 'rain washout'), (pd.Timestamp('2023-11-12 18:00'), 'Diwali')]:
            ax.axvline(t, color=INK2, lw=0.8, ls='--', zorder=1)
    tr = mpl.transforms.blended_transform_factory(a.transData, a.transAxes)
    for t, lab, ha in [(pd.Timestamp('2023-11-10 05:00'), 'rain washout (10 Nov)', 'right'),
                       (pd.Timestamp('2023-11-12 18:00'), 'Diwali (12 Nov)', 'left')]:
        a.text(t, 1.02, lab, transform=tr, ha=ha, va='bottom', fontsize=8, color=INK)
    pk = dm.idxmax()
    a.annotate(f'highest daily mean {dm.max():.0f} ({pk:%d %b})', (pk + pd.Timedelta(hours=12), dm.max()),
               xytext=(pk - pd.Timedelta(days=9), 560), fontsize=8, va='center',
               arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7),
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    d.xaxis.set_major_locator(mdates.DayLocator(interval=5)); d.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    header(fig, 'The October-November 2023 smog episode (case-study window shaded: 20 Oct - 20 Nov)',
           'All times IST. Observed AQI, burning in Punjab, mixing height and rainfall.', top=0.9)
    save(fig, 'fig1_episode_context')


# ----------------------------------------------------------------------------- 2 time series
def fig2():
    show = [('Persistence', '--', 1.1), ('Ensemble', '-', 1.4), ('LSTM', '-', 1.6), ('DualAQI v2', '-', 2.2)]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.4), sharex=True, gridspec_kw=dict(hspace=0.3))
    for ax, h in zip(axes, H):
        f = FC[FC.horizon == h].sort_values('valid')
        x = f.valid + pd.Timedelta(hours=5, minutes=30)
        for k in ('DualAQI v2', 'LSTM'):
            r = np.array([s[s.horizon == h].sort_values('valid').y_pred.values for s in SEEDS[k]])
            ax.fill_between(x, r.min(0), r.max(0), color=COL[k], alpha=0.14, lw=0)
        ax.plot(x, f.observed, color=INK, lw=2.0, marker='o', ms=3.2, label='observed', zorder=5)
        for m, ls, lw in show:
            ax.plot(x, f[m], color=COL[m], ls=ls, lw=lw, label=m + (' (mean of 3 seeds)' if m in NEURAL else ''), zorder=4)
        ax.axhline(400, color='#8c1d18', lw=0.8, ls=':')
        ax.text(1.005, 400, 'Severe', transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, color='#8c1d18')
        ax.set_ylim(0, 650); ax.set_ylabel('AQI'); ax.set_title(f'{h} h ahead')
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
    header(fig, 'Forecasts vs observations through the episode (forecast valid at 08:30 IST)',
           'Shaded bands: range across seeds 0-2 for DualAQI v2 and the LSTM. Dotted line: Severe threshold (AQI 400).', top=0.88)
    fig.subplots_adjust(bottom=0.09)
    save(fig, 'fig2_forecast_timeseries')


# ----------------------------------------------------------------------------- 3 scorecard
def fig3():
    cols = [('RMSE', h) for h in H] + [('Bias', h) for h in H] + [('POD_severe', h) for h in H] + [('CatAcc', h) for h in H]
    order = sorted(MODELS, key=lambda m: MET.loc[(m, slice(None)), 'RMSE'].mean())
    val = pd.DataFrame({c: [MET.loc[(m, c[1]), c[0]] for m in order] for c in cols}, index=order)
    rank = pd.DataFrame(index=order)
    for c in cols:
        v = val[c].abs() if c[0] == 'Bias' else val[c]
        rank[c] = v.rank(ascending=c[0] in ('RMSE', 'Bias'))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('b', ['#0d366b', '#2a78d6', '#86b6ef', '#eef4fc'])
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.grid(False)
    ax.imshow(rank.values, cmap=cmap, vmin=1, vmax=len(order), aspect='auto')
    for i in range(len(order)):
        for j, c in enumerate(cols):
            v, r = val.iloc[i, j], rank.iloc[i, j]
            txt = f'{v:.0f}' if c[0] in ('RMSE', 'Bias') else f'{v:.2f}'
            if c[0] == 'Bias' and v > 0:
                txt = '+' + txt
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.4, color='white' if r <= 4 else INK,
                    fontweight='bold' if r == 1 else 'normal')
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color='white', lw=1.4)
    for i in range(len(order) + 1):
        ax.axhline(i - 0.5, color='white', lw=1.4)
    for k in (2.5, 5.5, 8.5):
        ax.axvline(k, color=INK, lw=1.3)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(['24 h', '48 h', '72 h'] * 4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([m + (' *' if m in NEURAL else '') for m in order])
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith('DualAQI'):
            lbl.set_fontweight('bold')
    for j, t in enumerate(['RMSE (AQI)', 'Bias (AQI)', 'Severe days caught', 'Category accuracy']):
        ax.text(3 * j + 1, -0.95, t, ha='center', fontsize=9.5, fontweight='bold')
    ax.set_ylim(len(order) - 0.5, -1.35); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    header(fig, 'Case-study scorecard, 20 Oct - 20 Nov 2023 (32 forecasts per horizon; 13 Severe days)',
           'Darker = better rank in the column (bias ranked by size). * mean of seeds 0-2; other models seed 0 (seed-independent or nearly).',
           top=0.86)
    save(fig, 'fig3_scorecard')


# ----------------------------------------------------------------------------- 4 event matrix
def fig4():
    show = ['DualAQI v2', 'LSTM', 'Single-branch', 'Calendar-only', 'Ensemble', 'XGBoost', 'GBM', 'RandomForest',
            'Ridge', 'ARIMA', 'Persistence']
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.4), gridspec_kw=dict(hspace=0.42))
    HIT = {1: '#9ec5f4', 2: '#3987e5', 3: '#0d366b'}        # hits by 1, 2 or 3 of 3 seeds
    FA = {1: '#fbe3a3', 2: '#f4c04e', 3: '#c98500'}         # false alarms by 1, 2 or 3 of 3 seeds
    MISS, NEG = ORANGE, '#f0efec'
    for ax, h in zip(axes, (24, 72)):
        f = FC[FC.horizon == h].sort_values('valid').reset_index(drop=True)
        days = (f.valid + pd.Timedelta(hours=5, minutes=30)).dt.strftime('%d\n%b')
        ax.grid(False)
        obs_c = D.aqi_category(f.observed.values)
        for j, k in enumerate(obs_c):
            ax.add_patch(plt.Rectangle((j - 0.45, -1.45), 0.9, 0.9, color=CAT_COL[k], lw=0))
        ev = f.observed.values >= 400
        for i, m in enumerate(show):
            if m in NEURAL:   # number of seeds forecasting Severe each day
                k = np.sum([s[s.horizon == h].sort_values('valid').y_pred.values >= 400 for s in SEEDS[m]], axis=0)
            else:             # single deterministic run counts as all three
                k = (f[m].values >= 400).astype(int) * 3
            for j in range(len(f)):
                c = (HIT[k[j]] if k[j] else MISS) if ev[j] else (FA[k[j]] if k[j] else NEG)
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, color=c, lw=0))
            pod = (k[ev] / 3).mean()   # = mean over seeds of the share of Severe days caught
            ax.text(len(f) - 0.2, i, f'{pod*100:.0f}%', va='center', fontsize=8, color=INK)
        ax.text(len(f) - 0.2, -1, 'caught', va='center', fontsize=7.8, color=INK2)
        ax.set_xlim(-0.6, len(f) + 1.8); ax.set_ylim(len(show) - 0.4, -1.6)
        ax.set_yticks([-1] + list(range(len(show))))
        ax.set_yticklabels(['observed category'] + [m + (' *' if m in NEURAL else '') for m in show], fontsize=8.5)
        ax.set_xticks(range(0, len(f), 2)); ax.set_xticklabels(days[::2], fontsize=7.2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'{h} h ahead', pad=6)
    handles = [Patch(color=HIT[3], label='hit, 3 of 3 seeds'), Patch(color=HIT[2], label='hit, 2 of 3'),
               Patch(color=HIT[1], label='hit, 1 of 3'), Patch(color=MISS, label='missed (no seed)'),
               Patch(color=FA[3], label='false alarm, 3 of 3'), Patch(color=FA[1], label='false alarm, 1-2 of 3'),
               Patch(color=NEG, label='correct: not Severe')]
    handles2 = [Patch(color=CAT_COL[k], label=f'observed: {CATS[k]}') for k in (1, 3, 4, 5)]
    fig.legend(handles=handles, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.005), fontsize=8)
    fig.legend(handles=handles2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    header(fig, 'Day-by-day detection of Severe AQI (>= 400) during the episode',
           '* neural models: shade = how many of seeds 0-2 forecast Severe; single-run models count as 3 of 3. '
           'Right: share of the 13 Severe days caught (mean over seeds).', top=0.89)
    fig.subplots_adjust(bottom=0.1)
    save(fig, 'fig4_severe_event_matrix')


# ----------------------------------------------------------------------------- 5 washout
def fig5():
    fr, rr = [], []
    for f in glob.glob('Delhi PM2.5 Air Quality/2023/site_*_1Hr.csv'):
        d = pd.read_csv(f, usecols=['Timestamp', 'PM2.5 (µg/m³)', 'RF (mm)'], na_values=['NA'])
        d.index = pd.to_datetime(d.Timestamp)
        fr.append(d['PM2.5 (µg/m³)']); rr.append(d['RF (mm)'])
    pm = pd.concat(fr, axis=1).loc['2023-11-09 00:00':'2023-11-11 23:00']
    rf = pd.concat(rr, axis=1).loc['2023-11-09 00:00':'2023-11-11 23:00']
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.22))
    for c in pm.columns:
        a.plot(pm.index, pm[c], color='#c9c7c0', lw=0.6)
    a.plot(pm.index, pm.mean(axis=1), color=INK, lw=2.2, label='city mean')
    nrep = pm.loc['2023-11-10 00:00':'2023-11-10 08:00'].notna().sum(axis=1)
    a.plot([], [], color='#c9c7c0', lw=1, label=f'individual stations ({nrep.min()}-{nrep.max()} reporting on 10 Nov, 00-08 IST)')
    t = pd.Timestamp('2023-11-10 07:00')
    a.axvline(t, color=BLUE, lw=1, ls='--')
    a.text(t + pd.Timedelta(hours=1), 640, 'hour verified by the\n10 Nov forecasts (07:00-08:00 IST)', fontsize=8, color=BLUE, va='top')
    a.set_ylim(0, 680); a.set_ylabel('PM2.5 (µg/m³)'); a.legend(loc='center right', bbox_to_anchor=(1.0, 0.55))
    a.set_title('a  Hourly PM2.5 at every Delhi station')
    n_rain = (rf > 0).sum(axis=1); n_rep = rf.notna().sum(axis=1)
    b.bar(rf.index, n_rain, width=1 / 24 * 0.85, color=AQUA, align='edge')
    b.set_ylabel('gauges with rain'); b.set_ylim(0, max(12, n_rain.max() + 2))
    b.set_title(f'b  Station rain gauges reporting rain each hour (of about {int(n_rep.median())} reporting)')
    b.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    b.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    header(fig, 'The 10 November 2023 washout: a real, city-wide drop, not a data error',
           'Times IST (start of each hourly average). PM2.5 fell from about 400 to 35 µg/m³ in three hours as rain began; '
           'AQI went from 508 to 62.', top=0.87)
    save(fig, 'fig5_washout_10nov')


# ----------------------------------------------------------------------------- 6 scatter
def fig6():
    show = [('Ensemble', 's'), ('LSTM', '^'), ('DualAQI v2', 'o')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, h in zip(axes, (24, 72)):
        f = FC[FC.horizon == h]
        ax.grid(True, axis='both')
        ax.add_patch(plt.Rectangle((400, 0), 300, 400, color='#fde3d4', lw=0, zorder=0))
        ax.add_patch(plt.Rectangle((0, 400), 400, 300, color='#fdf3cf', lw=0, zorder=0))
        ax.text(690, 15, 'missed Severe', ha='right', fontsize=8.5, color='#b9461a')
        ax.text(15, 685, 'false Severe alarm', va='top', fontsize=8.5, color='#9a6b00')
        ax.plot([0, 700], [0, 700], color=INK, lw=0.9, ls='--')
        ax.axvline(400, color='#8c1d18', lw=0.8, ls=':'); ax.axhline(400, color='#8c1d18', lw=0.8, ls=':')
        for m, mk in show:
            r = MET.loc[(m, h), 'RMSE']
            if m in NEURAL:   # every seed's forecasts; RMSE = mean over seeds (as in the scorecard)
                pts = pd.concat([s[s.horizon == h] for s in SEEDS[m]])
                ax.scatter(pts.y_true, pts.y_pred, s=22, marker=mk, color=COL[m], alpha=0.55, edgecolor='none', zorder=3,
                           label=f'{m}, seeds 0-2: RMSE {r:.0f} (seed mean)')
            else:
                ax.scatter(f.observed, f[m], s=40, marker=mk, color=COL[m], edgecolor='white', lw=0.6, zorder=3,
                           label=f'{m}: RMSE {r:.0f}')
        ax.set_xlim(0, 700); ax.set_ylim(0, 700); ax.set_aspect('equal')
        ax.set_xlabel('observed AQI'); ax.set_title(f'{h} h ahead')
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=8)
    axes[0].set_ylabel('forecast AQI')
    header(fig, 'Observed vs forecast AQI in the episode (32 days per panel; neural models: all 3 seeds shown)',
           'Shaded: orange = Severe day forecast below 400 (missed); yellow = Severe forecast on a non-Severe day.', top=0.86)
    save(fig, 'fig6_observed_vs_forecast')


if __name__ == '__main__':
    print(MET.loc[(['DualAQI v2', 'LSTM', 'Single-branch', 'Calendar-only', 'Ensemble'], slice(None)),
                  ['RMSE', 'Bias', 'POD_severe']].round(2).to_string())
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
