#!/usr/bin/env python3
"""Objective 2 (ablation) figures: 3 on the data, 3 on the results. PNG, 300 dpi.

    .venv\\Scripts\\python objective2_figures.py      -> results/figures/objective2/*.png
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import dualaqi as D

OUT = os.path.join('results', 'figures', 'objective2')
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8d86', '#e4e3df', '#ffffff'
BLUE, ORANGE, AQUA, RED = '#2a78d6', '#eb6834', '#1baf7a', '#e34948'
FIRE = '#eb6834'
VAR = {'DualAQI': ('DualAQI v2 (fires + FRP + calendar)', BLUE),
       'DualAQI, calendar-only event branch': ('Calendar-only event branch', AQUA),
       'Single-branch LSTM (--no-fire)': ('Single-branch LSTM', ORANGE)}
MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
mpl.rcParams.update({
    'savefig.dpi': 300, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.family': 'DejaVu Sans',
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 8, 'axes.labelcolor': INK2, 'axes.edgecolor': '#b9b7b0', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 8.5, 'lines.linewidth': 1.6, 'text.color': INK})


def header(fig, title, sub=None):
    fig.text(0.01, 0.995, title, ha='left', va='top', fontsize=12.5, fontweight='bold')
    if sub:
        fig.text(0.01, 0.945, sub, ha='left', va='top', fontsize=9, color=INK2)


def save(fig, name):
    path = os.path.join(OUT, f'{name}.png')
    fig.savefig(path, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print('wrote', path)


# ----------------------------------------------------------------------------- data
pm = pd.read_csv('data/pm25_hourly.csv', parse_dates=['datetime'], index_col='datetime').loc['2019':'2025']
fire = pd.read_csv('data/punjab_fire_daily.csv', parse_dates=['date'], index_col='date').loc['2019':'2025'].fire_count
day = (pm.index + pd.Timedelta(hours=4)).normalize()            # IST calendar day of the hourly window
aqi = pd.Series(D.pm25_to_aqi(pm.pm25.groupby(day).mean().values), index=pm.pm25.groupby(day).mean().index)
aqi = aqi.loc['2019':'2025']


def fig1_seasonal_overlap():
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True, gridspec_kw=dict(hspace=0.32))
    fm = fire.groupby([fire.index.year, fire.index.month]).mean().unstack(0)
    am = aqi.groupby([aqi.index.year, aqi.index.month]).mean().unstack(0)
    for ax in (a, b):
        ax.axvspan(9.5, 11.5, color='#f6efe9', zorder=0, lw=0)
    a.bar(fm.index, fm.mean(1), color=FIRE, width=0.7)
    a.errorbar(fm.index, fm.mean(1), yerr=[fm.mean(1) - fm.min(1), fm.max(1) - fm.mean(1)], fmt='none',
               ecolor=INK2, elinewidth=0.9, capsize=2.5)
    a.set_ylabel('fire detections per day')
    a.set_ylim(0, fm.max(1).max() * 1.28)
    a.set_title('a  Punjab fire detections (mean of 2019-2025; whiskers: lowest and highest year)')
    b.plot(am.index, am.mean(1), color=INK, marker='o', ms=4)
    b.fill_between(am.index, am.min(1), am.max(1), color=MUTED, alpha=0.18, lw=0)
    for lo, hi, col in [(200, 300, '#f0a03c'), (300, 400, '#e0453a'), (400, 500, '#8c1d18')]:
        b.axhspan(lo, hi, color=col, alpha=0.07, lw=0, zorder=0)
    for y, t in [(250, 'Poor'), (350, 'Very poor'), (440, 'Severe')]:
        b.text(12.45, y, t, fontsize=7.8, color=INK2, va='center')
    b.set_ylim(0, 480)
    b.set_ylabel('daily AQI in Delhi')
    b.set_title('b  Delhi daily AQI (mean of 2019-2025; band: lowest to highest year)')
    b.set_xticks(range(1, 13)); b.set_xticklabels(MONTHS); b.set_xlim(0.4, 13.2)
    a.text(10.5, a.get_ylim()[1] * 0.97, 'stubble season\n(Oct-Nov)', ha='center', va='top', fontsize=8.5, color=FIRE,
           fontweight='bold')
    header(fig, 'Why test a fire branch: burning and Delhi pollution peak together in October-November')
    fig.subplots_adjust(top=0.89)
    save(fig, 'fig1_seasonal_overlap')


def fig2_lag_correlation():
    groups = [('2019-2021 (heavy burning)', [2019, 2020, 2021], '#86b6ef'),
              ('2022-2023', [2022, 2023], '#2a78d6'),
              ('2024-2025 (light burning)', [2024, 2025], '#104281')]
    lags = range(0, 11)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(wspace=0.28, width_ratios=[1.35, 1]))
    for name, yrs, col in groups:
        r = []
        for k in lags:
            f = fire.shift(k).reindex(aqi.index)
            m = aqi.index.month.isin([10, 11]) & aqi.index.year.isin(yrs)
            r.append(pd.Series(aqi[m]).corr(f[m], method='spearman'))
        a.plot(list(lags), r, color=col, marker='o', ms=4.5, label=name)
    a.axhline(0, color=INK2, lw=0.8)
    a.axvspan(0.5, 7.5, color='#f0efec', zorder=0, lw=0)
    a.text(4, 0.02, 'days the model sees (D-1 ... D-7)', ha='center', va='bottom', fontsize=8, color=INK2)
    a.set_xticks(list(lags)); a.set_xlabel('lag (days between fire day and Delhi AQI day)')
    a.set_ylabel('Spearman correlation, Oct-Nov days')
    a.set_title('a  Fire count k days earlier vs Delhi daily AQI')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    on = (fire.index.month.isin([10, 11]))
    tot = fire[on].groupby(fire.index[on].year).sum() / 1000
    am = aqi[aqi.index.month.isin([10, 11])].groupby(aqi.index[aqi.index.month.isin([10, 11])].year).mean()
    cols = ['#86b6ef'] * 3 + ['#2a78d6'] * 2 + ['#104281'] * 2
    b.bar(tot.index, tot.values, color=cols, width=0.68)
    for x, v in zip(tot.index, tot.values):
        b.text(x, v + 1.5, f'{v:.0f}k', ha='center', fontsize=8.3, fontweight='bold')
    b.set_ylim(0, tot.max() * 1.15)
    b.set_xticks(tot.index)
    b.set_xticklabels([f'{y}\n\nAQI\n{q:.0f}' for y, q in zip(tot.index, am.values)], fontsize=7.8)
    b.tick_params(axis='x', length=0)
    b.set_ylabel('Oct-Nov detections (thousands)')
    b.set_title('b  Oct-Nov fires per year (under each year: mean Delhi AQI)')
    header(fig, 'The fire signal: how strongly, and how long after, burning in Punjab shows up in Delhi',
           'Fire count on UTC day D-k vs city-mean AQI on IST day D; Oct-Nov only')
    fig.subplots_adjust(top=0.80)
    save(fig, 'fig2_fire_signal')


def fig3_design():
    fig = plt.figure(figsize=(11.5, 5.2))
    a = fig.add_axes([0.02, 0.03, 0.50, 0.84])
    a.axis('off'); a.set_xlim(0, 10); a.set_ylim(-0.3, 6.8)
    variants = [('Single-branch\nLSTM', ORANGE), ('Calendar-only\nevent branch', AQUA), ('DualAQI v2', BLUE)]
    inputs = [('PM2.5 / AQI, past 168 h', [1, 1, 1]), ('ERA5 meteorology, past 168 h', [1, 1, 1]),
              ('Calendar (hour, weekday, month, festival)', [0, 1, 1]), ('Punjab fires (count + FRP), D-1 ... D-7', [0, 0, 1]),
              ('Gated fusion of the two branches', [0, 1, 1])]
    xs = [5.55, 7.3, 9.05]
    for x, (name, col) in zip(xs, variants):
        a.add_patch(FancyBboxPatch((x - 0.82, 5.55), 1.64, 1.1, boxstyle='round,pad=0.05,rounding_size=0.15',
                                   fc=col, ec='none'))
        a.text(x, 6.1, name, ha='center', va='center', color='white', fontsize=7.6, fontweight='bold')
    for i, (name, use) in enumerate(inputs):
        y = 4.75 - i * 1.0
        a.add_patch(FancyBboxPatch((0.05, y - 0.38), 9.85, 0.76, boxstyle='round,pad=0.02,rounding_size=0.1',
                                   fc='#f7f6f3' if i % 2 == 0 else SURF, ec='none'))
        a.text(0.2, y, name, va='center', fontsize=8.8)
        for x, u in zip(xs, use):
            if u:
                a.plot(x, y, 'o', ms=13, color=INK)
                a.text(x, y, '✓', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            else:
                a.plot(x, y, 'o', ms=13, mfc='white', mec='#c9c7c0', mew=1.2)
    a.text(0.2, 6.1, 'Inputs used', fontsize=10, fontweight='bold', va='center')
    a.text(0.2, -0.15, 'Same LSTM, same output head, same initial weights, same minibatch order, same folds, seeds 0-2.',
           fontsize=8, color=INK2)

    b = fig.add_axes([0.58, 0.48, 0.40, 0.40])
    b.axis('off'); b.set_xlim(0, 10); b.set_ylim(0, 4)
    b.text(0, 3.75, 'What each comparison measures', fontsize=10, fontweight='bold')
    for i, (lhs, rhs, what, col) in enumerate([
            ('DualAQI', 'Single-branch', 'effect of the whole event branch (spec)', BLUE),
            ('DualAQI', 'Calendar-only', 'effect of the fire inputs alone (count + FRP)', FIRE),
            ('Calendar-only', 'Single-branch', 'effect of the calendar inputs alone', AQUA)]):
        y = 2.75 - i * 1.05
        b.add_patch(FancyBboxPatch((0, y - 0.36), 10, 0.72, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#f7f6f3', ec='none'))
        b.add_patch(plt.Rectangle((0, y - 0.36), 0.12, 0.72, color=col, lw=0))
        b.text(0.35, y + 0.12, f'{lhs}  -  {rhs}', fontsize=8.8, fontweight='bold', va='center')
        b.text(0.35, y - 0.17, what, fontsize=8.2, color=INK2, va='center')

    fig.text(0.58, 0.38, 'Evaluation periods', fontsize=10, fontweight='bold', va='bottom')
    c = fig.add_axes([0.70, 0.08, 0.27, 0.27])
    c.grid(False)
    for s in ('left', 'top', 'right'):
        c.spines[s].set_visible(False)
    c.set_yticks([])
    periods = [('Ablation: forward CV', '2020-01-08', '2023-12-17', BLUE, '1,440 forecasts / horizon'),
               ('Held-out test 1', '2022-01-01', '2023-12-17', ORANGE, '720'),
               ('Held-out test 2', '2024-07-02', '2025-12-06', AQUA, '523')]
    for i, (lab, s, e, col, n) in enumerate(periods):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        c.barh(2 - i, (e - s).days, left=s, height=0.55, color=col)
        c.text(s - pd.Timedelta(days=25), 2 - i, lab, ha='right', va='center', fontsize=8.2)
        c.text(e + pd.Timedelta(days=25), 2 - i, n, ha='left', va='center', fontsize=7.8, color=INK2)
    c.set_xlim(pd.Timestamp('2019-06-01'), pd.Timestamp('2026-09-01'))
    import matplotlib.dates as mdates
    c.xaxis.set_major_locator(mdates.YearLocator()); c.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    c.tick_params(axis='x', labelsize=8)
    c.set_xticks(pd.to_datetime(['2020-01-01', '2022-01-01', '2024-01-01', '2026-01-01']))
    header(fig, 'Ablation design: three networks that differ only in their event branch')
    save(fig, 'fig3_ablation_design')


# ----------------------------------------------------------------------------- results
met = pd.read_csv('results/dualaqi/ablation_metrics_v2.csv')
drm = pd.read_csv('results/dualaqi/ablation_drmse_v2.csv')
H = [24, 48, 72]


def dodge_points(ax, season, metric, ylabel, title, ref=None):
    for j, (m, (lab, col)) in enumerate(VAR.items()):
        t = met[(met.season == season) & (met.model == m)].sort_values('horizon')
        x = np.arange(3) + (j - 1) * 0.22
        ax.errorbar(x, t[f'{metric}_mean'], yerr=t[f'{metric}_sd'], fmt='o', ms=7, color=col, ecolor=col,
                    elinewidth=1.4, capsize=3, label=lab, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(['24 h', '48 h', '72 h']); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(ylabel); ax.set_title(title)


def fig4_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), gridspec_kw=dict(hspace=0.42, wspace=0.25))
    dodge_points(axes[0, 0], 'Oct-Nov', 'RMSE', 'RMSE (AQI)', 'a  RMSE, stubble season (Oct-Nov)')
    dodge_points(axes[0, 1], 'rest of year', 'RMSE', 'RMSE (AQI)', 'b  RMSE, rest of the year')
    dodge_points(axes[1, 0], 'Oct-Nov', 'Bias', 'bias: forecast - observed (AQI)', 'c  Bias, stubble season', ref=0)
    dodge_points(axes[1, 1], 'Oct-Nov', 'POD_severe', 'share of Severe days caught', 'd  Severe-day detection, stubble season')
    axes[1, 1].set_ylim(0, 1)
    n = met[(met.season == 'Oct-Nov')].n.iloc[0]
    ns = met[(met.season == 'Oct-Nov')].n_severe.iloc[0]
    axes[1, 1].text(0.98, 0.95, f'{ns} Severe days of {n}', transform=axes[1, 1].transAxes, ha='right', va='top',
                    fontsize=8, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    header(fig, 'Ablation results, forward CV 2020-2023 (mean ± SD over seeds 0-2)',
           'Oct-Nov: 244 forecasts per horizon; rest of year: 1,196. Lower RMSE and bias closer to zero are better.')
    fig.subplots_adjust(top=0.86, bottom=0.10)
    save(fig, 'fig4_ablation_metrics')


def fig5_paired():
    comps = [('event branch (spec ablation): DualAQI - single-branch', 'a  Whole event branch\nDualAQI - single-branch'),
             ('fire counts alone: DualAQI - calendar-only', 'b  Fire counts alone\nDualAQI - calendar-only'),
             ('calendar inputs alone: calendar-only - single-branch', 'c  Calendar inputs alone\ncalendar-only - single-branch')]
    seasons = [('all', INK, 'all year'), ('Oct-Nov', RED, 'Oct-Nov'), ('rest of year', MUTED, 'rest of year')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (key, title) in zip(axes, comps):
        ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.axvline(0, color=INK, lw=0.9)
        ax.axvspan(-30, 0, color='#eef5fd', zorder=0, lw=0)
        y = 0
        for h in H:
            for s, col, _ in seasons:
                r = drm[(drm.comparison == key) & (drm.horizon == h) & (drm.season == s)].iloc[0]
                ax.hlines(y, r.CI95_7dblock_lo, r.CI95_7dblock_hi, color=col, lw=1.0)
                ax.hlines(y, r.CI95_day_lo, r.CI95_day_hi, color=col, lw=3.4)
                sig = bool(r.excludes_zero_day and r.excludes_zero_7dblock)
                ax.plot(r.dRMSE, y, 'o', ms=7, mfc=col if sig else 'white', mec=col, mew=1.6, zorder=3)
                y += 1
            y += 0.8
        ax.set_xlim(-18, 9); ax.set_title(title, fontsize=9.6)
        ax.set_xlabel('paired RMSE difference (AQI)')
    ticks, labels, y = [], [], 0
    for h in H:
        for _, _, lab in seasons:
            ticks.append(y); labels.append(f'{h} h  {lab}'); y += 1
        y += 0.8
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    handles = [Line2D([], [], color=INK, lw=3.4, label='95% CI, day bootstrap'),
               Line2D([], [], color=INK, lw=1.0, label='95% CI, 7-day block bootstrap'),
               Line2D([], [], marker='o', color='none', mfc=INK, mec=INK, label='both intervals exclude zero'),
               Line2D([], [], marker='o', color='none', mfc='white', mec=INK, mew=1.6, label='not significant')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04))
    header(fig, 'Paired comparisons, forward CV 2020-2023 (mean over seeds 0-2)',
           'Blue area (negative) = the first model has the lower RMSE. Colour: season (black all year, red Oct-Nov, grey rest of year).')
    fig.subplots_adjust(top=0.82, bottom=0.17)
    save(fig, 'fig5_paired_differences')


def fig6_replication():
    rows = []
    t = drm[drm.comparison.str.startswith('event branch')]
    rows.append(('Forward CV 2020-2023', t))
    for path, lab in [('results/dualaqi_v2/test_drmse.csv', 'Held-out test 2022-2023'),
                      ('results/dualaqi_v2/test2024_drmse.csv', 'Held-out test Jul 2024-2025')]:
        d = pd.read_csv(path)
        rows.append((lab, d[d.comparison == 'DualAQI v2 - LSTM']))
    period_col = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (season, title) in zip(axes, [('all', 'a  All year'), ('Oct-Nov', 'b  Stubble season (Oct-Nov)'),
                                          ('rest of year', 'c  Rest of the year')]):
        ax.axhline(0, color=INK, lw=0.9)
        ax.axhspan(-30, 0, color='#eef5fd', zorder=0, lw=0)
        for j, ((lab, d), col) in enumerate(zip(rows, period_col)):
            d = d[d.season == season].sort_values('horizon')
            x = np.arange(3) + (j - 1) * 0.24
            for xi, (_, r) in zip(x, d.iterrows()):
                ax.vlines(xi, r.CI95_7dblock_lo, r.CI95_7dblock_hi, color=col, lw=1.0)
                ax.vlines(xi, r.CI95_day_lo, r.CI95_day_hi, color=col, lw=3.4)
                sig = bool(r.excludes_zero_day and r.excludes_zero_7dblock)
                ax.plot(xi, r.dRMSE, 'o', ms=7, mfc=col if sig else 'white', mec=col, mew=1.6, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(['24 h', '48 h', '72 h']); ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-18, 13); ax.set_title(title)
    axes[0].set_ylabel('RMSE difference, DualAQI v2 - LSTM (AQI)')
    axes[0].text(-0.45, -16.8, 'DualAQI better', fontsize=8, color=BLUE)
    handles = [Patch(color=c, label=l) for (l, _), c in zip(rows, period_col)]
    handles += [Line2D([], [], marker='o', color='none', mfc=INK, mec=INK, label='both CIs exclude zero'),
                Line2D([], [], marker='o', color='none', mfc='white', mec=INK, mew=1.6, label='not significant')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
    header(fig, 'Does the fire-branch effect hold over time? DualAQI v2 vs single-branch LSTM in three periods',
           'Thick bar: 95% CI (day bootstrap); thin bar: 7-day block bootstrap. Oct-Nov fires fell from ~78k (2021) to ~4k (2025).')
    fig.subplots_adjust(top=0.80, bottom=0.16)
    save(fig, 'fig6_effect_over_time')


if __name__ == '__main__':
    fig1_seasonal_overlap()
    fig2_lag_correlation()
    fig3_design()
    fig4_metrics()
    fig5_paired()
    fig6_replication()
