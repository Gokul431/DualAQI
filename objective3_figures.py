#!/usr/bin/env python3
"""Objective 3 (benchmark) result figures: 10 PNGs, 300 dpi.

    .venv\\Scripts\\python objective3_figures.py      -> results/figures/objective3/*.png
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import dualaqi as D

OUT = os.path.join('results', 'figures', 'objective3')
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8d86', '#e4e3df', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6'
MODELS = ['DualAQI v2', 'LSTM', 'Ensemble', 'RandomForest', 'XGBoost', 'GBM', 'Ridge', 'ARIMA', 'Persistence']
COL = {'DualAQI v2': BLUE, 'DualAQI v1': AQUA, 'LSTM': ORANGE, 'Ensemble': YELLOW, 'RandomForest': VIOLET,
       'ARIMA': MAGENTA, 'XGBoost': '#a3a198', 'GBM': '#7d7b74', 'Ridge': '#c9c7c0', 'Persistence': '#5f5d57'}
MARK = {'DualAQI v2': 'o', 'DualAQI v1': 'o', 'LSTM': 'o', 'Ensemble': 'o', 'RandomForest': 'o', 'ARIMA': 'o',
        'XGBoost': 'D', 'GBM': 's', 'Ridge': '^', 'Persistence': 'v'}
KEY = ['DualAQI v2', 'LSTM', 'Ensemble', 'ARIMA', 'Persistence']
H = [24, 48, 72]
BLUES = LinearSegmentedColormap.from_list('b', ['#0d366b', '#2a78d6', '#86b6ef', '#e3eefc'])
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


def spread_labels(ys, gap):
    """Shift sorted label positions apart by at least `gap` (keeps order)."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = (np.array(ys, dtype=float)[order] - pos).mean()
    out = np.empty_like(pos)
    out[order] = pos + shift
    return out


# ----------------------------------------------------------------------------- data
COLS = ['model', 'horizon', 'origin', 'fold', 'y_true', 'y_pred']


def read_oof(path, model, folds=None, rename=None):
    o = pd.read_csv(path, parse_dates=['origin'])
    o = o[(o.scheme == 'forward') & (o.model == model)]
    if folds is not None:
        o = o[o.fold.isin(list(folds))]
    return o.assign(model=rename or model)[COLS]


def dual_oof(s):
    """DualAQI v2 forward forecasts 2020-2023: development folds 0-7 (C2) + test folds 8-15 (V2)."""
    return pd.concat([read_oof(f'results/dualaqi_v2/dev/C2/seed{s}/oof_predictions.csv.gz', 'C2', range(8), 'DualAQI v2'),
                      read_oof(f'results/dualaqi_v2/test/V2/seed{s}/oof_predictions.csv.gz', 'DualAQI-v2', range(8, 16), 'DualAQI v2')])


# Per-seed forecasts of every model. ARIMA (seed-independent) comes from the seed-0 run.
SEED_RUNS = {0: ('results/oof_predictions.csv.gz', 'results/benchmark_long.csv'),
             1: ('results/seed1/oof_predictions.csv.gz', 'results/seed1/benchmark_long.csv'),
             2: ('results/seed2/oof_predictions.csv.gz', 'results/seed2/benchmark_long.csv')}
OOFS, blk_parts = {}, []
for s, (op, bp) in SEED_RUNS.items():
    b = pd.read_csv(op, parse_dates=['origin'])
    b = b[(b.scheme == 'forward') & b.model.isin(MODELS)][COLS]
    if s > 0:
        b = pd.concat([b, OOFS[0][OOFS[0].model == 'ARIMA'][COLS]])
    o = pd.concat([b, dual_oof(s)], ignore_index=True)
    o['valid'] = o.origin + pd.to_timedelta(o.horizon, unit='h')
    OOFS[s] = o
    t = pd.read_csv(bp)
    t = t[t.scheme == 'blocked']
    if s > 0:
        t0 = pd.read_csv(SEED_RUNS[0][1])
        t = pd.concat([t, t0[(t0.scheme == 'blocked') & (t0.model == 'ARIMA')]])
    blk_parts.append(t)
OOF = OOFS[0]
# metrics: mean over seeds 0-2 (identical across seeds for Persistence, Ridge, GBM, ARIMA)
fwd = pd.concat([D.summarise(OOFS[s][COLS], 'forward', 500, 0) for s in OOFS]).groupby(['model', 'horizon']).mean(numeric_only=True)
blk = pd.concat(blk_parts).groupby(['model', 'horizon']).mean(numeric_only=True)
assert set(fwd.index.get_level_values(0)) >= set(MODELS), set(MODELS) - set(fwd.index.get_level_values(0))


# ----------------------------------------------------------------------------- 1 heatmap
def fig01():
    cols = [('forward', h) for h in H] + [('blocked', h) for h in H]
    order = sorted(MODELS, key=lambda m: fwd.loc[(m, slice(None)), 'RMSE'].mean())
    val = pd.DataFrame({c: [(fwd if c[0] == 'forward' else blk).RMSE.get((m, c[1]), np.nan) for m in order] for c in cols},
                       index=order)
    rank = val.rank(axis=0)
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.grid(False)
    ax.imshow(np.ma.masked_invalid(rank.values), cmap=BLUES, vmin=1, vmax=len(MODELS), aspect='auto')
    for i, m in enumerate(order):
        for j, c in enumerate(cols):
            v, r = val.iloc[i, j], rank.iloc[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#f4f3ef', ec='white', hatch='///', lw=0))
                ax.text(j, i, 'not run', ha='center', va='center', fontsize=7.5, color=MUTED)
                continue
            dark = r <= 4
            ax.text(j, i - 0.12, f'{v:.1f}', ha='center', va='center', fontsize=9.2,
                    fontweight='bold' if r == 1 else 'normal', color='white' if dark else INK)
            ax.text(j, i + 0.26, f'rank {int(r)}', ha='center', va='center', fontsize=6.8,
                    color='#dce9fa' if dark else INK2)
    for j in range(len(cols) + 1):
        ax.axvline(j - 0.5, color='white', lw=1.5)
    for i in range(len(order) + 1):
        ax.axhline(i - 0.5, color='white', lw=1.5)
    ax.axvline(2.5, color=INK, lw=1.4)
    ax.set_xticks(range(6)); ax.set_xticklabels(['24 h', '48 h', '72 h'] * 2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith('DualAQI'):
            lbl.set_fontweight('bold')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1, -0.95, 'Forward CV 2020-2023 (n = 1,440)', ha='center', fontsize=9.5, fontweight='bold')
    ax.text(4, -0.95, 'Blocked CV 2019-2023 (n = 1,816)', ha='center', fontsize=9.5, fontweight='bold')
    ax.set_ylim(len(order) - 0.5, -1.3)
    header(fig, 'RMSE of every model (AQI): darker = better rank within the column',
           'Mean of seeds 0-2 (Persistence, Ridge, GBM and ARIMA do not depend on the seed). Rows sorted by mean forward-CV RMSE. DualAQI was run with forward CV only.', top=0.86)
    save(fig, 'fig01_rmse_heatmap')


# ----------------------------------------------------------------------------- 2 forest
def fig02():
    mods = [m for m in MODELS if m != 'Persistence']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, h in zip(axes, H):
        p = fwd.loc[('Persistence', h), 'RMSE']
        order = sorted(mods, key=lambda m: -fwd.loc[(m, h), 'skill_RMSE_vs_persist'])
        ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.axvline(0, color=INK, lw=0.9)
        for i, m in enumerate(mods):
            r = fwd.loc[(m, h)]
            lo, hi = -r.dRMSE_vs_persist_CI95_hi / p * 100, -r.dRMSE_vs_persist_CI95_lo / p * 100
            sk = r.skill_RMSE_vs_persist * 100
            ax.hlines(i, lo, hi, color=COL[m], lw=3)
            ax.plot(sk, i, MARK[m], ms=7.5, color=COL[m], mec='white', mew=0.8, zorder=3)
            ax.text(hi + 0.6, i, f'{sk:.1f}%', va='center', fontsize=7.6, color=INK2)
        ax.set_xlim(-8, 32); ax.set_xlabel('RMSE reduction vs Persistence (%)')
        ax.set_title(f'{h} h ahead', fontsize=10)
    axes[0].set_yticks(range(len(mods))); axes[0].set_yticklabels(mods); axes[0].invert_yaxis()
    header(fig, 'Skill against Persistence with 95% bootstrap intervals (forward CV 2020-2023)',
           'Mean of seeds 0-2. Right of zero = better than Persistence. Bars: 95% CI resampling whole days (500 resamples), averaged over seeds.', top=0.8)
    save(fig, 'fig02_skill_forest')


# ----------------------------------------------------------------------------- 3 slope
def fig03():
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.grid(True, axis='y')
    for m in MODELS[::-1]:
        y = [fwd.loc[(m, h), 'RMSE'] for h in H]
        hl = m in ('DualAQI v2', 'DualAQI v1', 'LSTM', 'Ensemble', 'RandomForest', 'ARIMA', 'Persistence')
        ax.plot([0, 1, 2], y, color=COL[m], lw=2.4 if m == 'DualAQI v2' else 1.6, marker=MARK[m], ms=6,
                alpha=1 if hl else 0.85, zorder=3 if m == 'DualAQI v2' else 2, mec='white', mew=0.6)
    for x, side in ((0, -1), (2, 1)):
        ys = [fwd.loc[(m, H[x]), 'RMSE'] for m in MODELS]
        lab = spread_labels(ys, 2.05)
        for m, yv, yl in zip(MODELS, ys, lab):
            xt = x + side * 0.12
            ax.plot([x + side * 0.03, xt - side * 0.015], [yv, yl], color=GRID, lw=0.7)
            ax.text(xt, yl, f'{m}  {yv:.1f}' if side > 0 else f'{yv:.1f}  {m}', ha='left' if side > 0 else 'right',
                    va='center', fontsize=8, color=INK, fontweight='bold' if m.startswith('DualAQI') else 'normal')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['24 h', '48 h', '72 h'], fontsize=10)
    ax.set_xlim(-0.95, 2.95); ax.set_ylabel('RMSE (AQI)')
    ax.spines['left'].set_visible(False)
    header(fig, 'How forecast error grows with lead time (forward CV 2020-2023, mean of seeds 0-2)',
           'Learning models degrade slowly; Persistence loses most from 24 h to 72 h.', top=0.88)
    save(fig, 'fig03_horizon_slope')


# ----------------------------------------------------------------------------- 4 bump
def fig04():
    metrics = [('RMSE', True, 'RMSE'), ('MAE', True, 'MAE'), ('R2', False, 'R²'), ('CatAcc', False, 'Category\naccuracy'),
               ('Cat+-1', False, 'Within one\ncategory')]
    avg = {k: {m: fwd.loc[(m, slice(None)), k].mean() for m in MODELS} for k, _, _ in metrics}
    ranks = {k: pd.Series(avg[k]).rank(ascending=asc) for k, asc, _ in metrics}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.grid(False)
    for m in MODELS:
        y = [ranks[k][m] for k, _, _ in metrics]
        ax.plot(range(len(metrics)), y, color=COL[m], lw=3.2 if m == 'DualAQI v2' else 2.0, marker=MARK[m], ms=9,
                mec='white', mew=1, zorder=3 if m.startswith('DualAQI') else 2)
        ax.text(-0.15, y[0], m, ha='right', va='center', fontsize=8.6, fontweight='bold' if m.startswith('DualAQI') else 'normal')
        ax.text(len(metrics) - 1 + 0.15, y[-1], m, ha='left', va='center', fontsize=8.6,
                fontweight='bold' if m.startswith('DualAQI') else 'normal')
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([l for _, _, l in metrics], fontsize=9)
    ax.xaxis.tick_top(); ax.tick_params(axis='x', length=0, pad=8)
    n = len(MODELS)
    ax.set_yticks(range(1, n + 1)); ax.set_ylim(n + 0.6, 0.4); ax.set_yticklabels([f'#{i}' for i in range(1, n + 1)])
    ax.tick_params(axis='y', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-1.6, len(metrics) - 1 + 1.6)
    r2 = [int(ranks[k]['DualAQI v2']) for k, _, _ in metrics]
    header(fig, 'Model ranking across five metrics (forward CV 2020-2023, mean of seeds 0-2, averaged over 24/48/72 h)',
           f'#1 = best. DualAQI v2 ranks #{min(r2)}-#{max(r2)} across the five metrics.', top=0.82)
    save(fig, 'fig04_rank_bump')


# ----------------------------------------------------------------------------- 5 Taylor
def fig05():
    fig = plt.figure(figsize=(12, 6.2))
    tmax_deg, rmin, rmax = 55, 0.55, 1.1
    tmax = np.deg2rad(tmax_deg)
    for k, h in enumerate((24, 72)):
        ax = fig.add_subplot(1, 2, k + 1, projection='polar')
        ax.set_thetamin(0); ax.set_thetamax(tmax_deg)
        ax.set_rorigin(0); ax.set_rlim(rmin, rmax)
        d = OOF[OOF.horizon == h]
        sd_o = d[d.model == 'Persistence'].y_true.std()
        for c in (0.2, 0.35, 0.5, 0.65, 0.8):                  # centred RMS difference arcs around the reference
            phi = np.linspace(0, np.pi, 600)
            x, y = 1 + c * np.cos(phi), c * np.sin(phi)
            th, r = np.arctan2(y, x), np.hypot(x, y)
            ok = (th >= 0) & (th <= tmax) & (r >= rmin) & (r <= rmax)
            ax.plot(np.where(ok, th, np.nan), np.where(ok, r, np.nan), color='#c9c7c0', lw=0.9, ls='--')
            idx = np.where(ok)[0]
            if len(idx):
                j = idx[np.argmin(th[idx])]
                ax.text(th[j] + np.deg2rad(1.5), r[j], f'{c:.2f}', fontsize=7, color=MUTED, ha='center', va='center')
        ax.plot(np.linspace(0, tmax, 100), np.ones(100), color=INK2, lw=0.9)
        ax.plot(0, 1, marker='*', ms=15, color=INK, zorder=6, clip_on=False)
        for m in MODELS[::-1]:
            stats = []
            for o in OOFS.values():   # mean over seeds of correlation and variability ratio
                g = o[(o.horizon == h) & (o.model == m)]
                stats.append((np.corrcoef(g.y_true, g.y_pred)[0, 1], g.y_pred.std() / sd_o))
            corr, ratio = np.mean(stats, axis=0)
            big = m == 'DualAQI v2'
            ax.plot(np.arccos(corr), ratio, MARK[m], ms=12 if big else 9, color=COL[m], mec=INK if big else 'white',
                    mew=1.2 if big else 0.8, zorder=6 if big else 4, label=m)
        cts = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
        ax.set_thetagrids(np.degrees(np.arccos(cts)), [f'{c}' for c in cts], fontsize=7.8)
        ax.set_rgrids([0.6, 0.7, 0.8, 0.9, 1.0, 1.1], fontsize=7.8)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(f'{h} h ahead', fontsize=10.5, loc='center', pad=18)
        ax.text(np.deg2rad(tmax_deg / 2), rmax * 1.075, 'correlation', rotation=-tmax_deg / 2 + 90 - 90,
                ha='center', va='center', fontsize=8.5, color=INK2)
    for k in range(2):
        pos = fig.axes[k].get_position()
        fig.text(pos.x0 + pos.width * 0.55, pos.y0 - 0.035, 'standard deviation / observed standard deviation',
                 ha='center', fontsize=8.5, color=INK2)
    h_, l_ = fig.axes[0].get_legend_handles_labels()
    order = [l_.index(m) for m in MODELS]
    fig.legend([h_[i] for i in order], [l_[i] for i in order], loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5, handletextpad=0.3)
    header(fig, 'Taylor diagram: correlation and variability against the observations (forward CV, mean of seeds 0-2)',
           'Star = observations; closer is better. Dashed arcs: centred RMS difference (normalised). '
           'Points inside the solid arc vary less than the observations.', top=0.84)
    fig.subplots_adjust(bottom=0.14, wspace=0.12)
    save(fig, 'fig05_taylor_diagram')


# ----------------------------------------------------------------------------- 6 dumbbell
def fig06():
    mods = [m for m in MODELS if not m.startswith('DualAQI')]
    order = sorted(mods, key=lambda m: fwd.loc[(m, slice(None)), 'RMSE'].mean())   # one order for all panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, h in zip(axes, H):
        ax.grid(True, axis='x'); ax.grid(False, axis='y')
        for i, m in enumerate(order):
            b, f = blk.loc[(m, h), 'RMSE'], fwd.loc[(m, h), 'RMSE']
            ax.plot([b, f], [i, i], color=GRID, lw=2.5, zorder=1)
            ax.plot(b, i, 'o', ms=8, mfc='white', mec=COL[m], mew=1.8, zorder=3)
            ax.plot(f, i, 'o', ms=8, color=COL[m], mec='white', mew=0.8, zorder=3)
        ax.set_title(f'{h} h ahead', fontsize=10); ax.set_xlabel('RMSE (AQI)')
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order); axes[0].invert_yaxis()
    handles = [Line2D([], [], marker='o', color='none', mfc='white', mec=INK, mew=1.6, ms=8, label='blocked CV (2019-2023)'),
               Line2D([], [], marker='o', color='none', mfc=INK, mec=INK, ms=8, label='forward CV (2020-2023)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    header(fig, 'Robustness to the validation scheme: blocked vs forward cross-validation',
           'Rows sorted by mean forward-CV RMSE. Blocked CV is higher for most models but the ranking is nearly the same.', top=0.8)
    fig.subplots_adjust(bottom=0.2)
    save(fig, 'fig06_blocked_vs_forward')


# ----------------------------------------------------------------------------- 7 seeds
def fig07():
    per = []
    for s, path in [(0, 'results/benchmark_long.csv'), (1, 'results/seed1/benchmark_long.csv'),
                    (2, 'results/seed2/benchmark_long.csv')]:
        b = pd.read_csv(path)
        b = b[b.scheme == 'forward'][['model', 'horizon', 'RMSE']].assign(seed=s)
        per.append(b)
        d = dual_oof(s)
        for (m, h), g in d.groupby(['model', 'horizon']):
            per.append(pd.DataFrame([dict(model=m, horizon=h, RMSE=np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)), seed=s)]))
    per = pd.concat(per)
    per = per[per.model.isin(MODELS)]
    order = per.groupby('model').RMSE.mean().sort_values().index.tolist()   # one order for all panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, h in zip(axes, H):
        p = per[per.horizon == h]
        ax.grid(True, axis='x'); ax.grid(False, axis='y')
        for i, m in enumerate(order):
            v = p[p.model == m].RMSE.values
            ax.hlines(i, v.min(), v.max(), color=COL[m], lw=6, alpha=0.35)
            ax.scatter(v, [i] * len(v), s=26, color=COL[m], edgecolor='white', lw=0.6, zorder=3)
            ax.plot(v.mean(), i, '|', ms=14, mew=2.2, color=INK, zorder=4)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ax.text(v.max() + 0.5, i, f'{v.mean():.1f}' + (f' ± {sd:.2f}' if sd > 1e-6 else ''),
                    va='center', fontsize=7.4, color=INK2)
        ax.set_xlim(p.RMSE.min() - 2, p.RMSE.max() + 11)
        ax.set_title(f'{h} h ahead', fontsize=10); ax.set_xlabel('RMSE (AQI)')
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order); axes[0].invert_yaxis()
    header(fig, 'Sensitivity to the random seed: forward-CV RMSE for seeds 0, 1 and 2',
           'Dots: individual seeds; black tick: mean; label: mean ± SD. Persistence, Ridge, ARIMA and GBM do not depend on the seed.',
           top=0.8)
    save(fig, 'fig07_seed_stability')


# ----------------------------------------------------------------------------- 8 violins
def fig08():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, h in zip(axes, H):
        data = [np.concatenate([(o[(o.horizon == h) & (o.model == m)].y_pred - o[(o.horizon == h) & (o.model == m)].y_true).values
                                for o in OOFS.values()]) for m in KEY]
        vp = ax.violinplot(data, positions=range(len(KEY)), widths=0.8, showextrema=False, bw_method=0.25)
        for body, m in zip(vp['bodies'], KEY):
            body.set_facecolor(COL[m]); body.set_edgecolor('none'); body.set_alpha(0.55)
        for i, e in enumerate(data):
            q1, med, q3 = np.percentile(e, [25, 50, 75])
            ax.vlines(i, q1, q3, color=INK, lw=4)
            ax.plot(i, med, 'o', ms=5, color='white', mec=INK, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-300, 300)
        ax.set_xticks(range(len(KEY))); ax.set_xticklabels(KEY, rotation=25, ha='right')
        ax.set_title(f'{h} h ahead', fontsize=10)
    axes[0].set_ylabel('forecast error: forecast - observed (AQI)')
    header(fig, 'Distribution of forecast errors (forward CV 2020-2023, seeds 0-2 pooled)',
           'Violin: error density; black bar: interquartile range; white dot: median. Axis clipped at ±300.', top=0.82)
    save(fig, 'fig08_error_distribution')


# ----------------------------------------------------------------------------- 9 seasons
def fig09():
    seasons = [('Winter\n(Jan-Feb)', [1, 2]), ('Pre-monsoon\n(Mar-May)', [3, 4, 5]),
               ('Monsoon\n(Jun-Sep)', [6, 7, 8, 9]), ('Post-monsoon\n(Oct-Dec)', [10, 11, 12])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    w = 0.16
    for ax, h in zip(axes, H):
        for j, m in enumerate(KEY):
            vals = []
            for _, ms in seasons:
                per_seed = []
                for o in OOFS.values():
                    g = o[(o.horizon == h) & (o.model == m) & o.valid.dt.month.isin(ms)]
                    per_seed.append(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)))
                vals.append(np.mean(per_seed))
            ax.bar(np.arange(4) + (j - 2) * w, vals, width=w * 0.92, color=COL[m], label=m, edgecolor='white', lw=0.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([s for s, _ in seasons], fontsize=8)
        ax.set_title(f'{h} h ahead', fontsize=10)
    axes[0].set_ylabel('RMSE (AQI)')
    n = OOF[(OOF.horizon == 24) & (OOF.model == 'Persistence')].valid.dt.month
    counts = [int(n.isin(ms).sum()) for _, ms in seasons]
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
    header(fig, 'Forecast error by IMD season (forward CV 2020-2023, mean of seeds 0-2; season of the target date)',
           'Forecasts per season: ' + ', '.join(f'{s.split(chr(10))[0]} {c}' for (s, _), c in zip(seasons, counts)), top=0.82)
    fig.subplots_adjust(bottom=0.22)
    save(fig, 'fig09_season_rmse')


# ----------------------------------------------------------------------------- 10 performance diagram
def fig10():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    sr, pod = np.meshgrid(np.linspace(0.01, 1, 200), np.linspace(0.01, 1, 200))
    csi = 1 / (1 / sr + 1 / pod - 1)
    for ax, h in zip(axes, H):
        ax.grid(False)
        cf = ax.contourf(sr, pod, csi, levels=np.arange(0, 1.01, 0.1), cmap=LinearSegmentedColormap.from_list(
            'c', ['#ffffff', '#cde2fb', '#6da7ec']), alpha=0.8)
        for b in (0.5, 0.8, 1.0, 1.25, 2.0):
            x = np.linspace(0, 1, 50)
            ax.plot(x, b * x, color=MUTED, lw=0.7, ls='--')
            xe = min(1, 1 / b)
            ax.text(xe * 0.98, min(1, b * xe) * 0.98, f'{b:g}', fontsize=7, color=INK2, ha='right', va='top')
        for m in MODELS:
            srs, pods = [], []
            for o in OOFS.values():   # mean over seeds
                g = o[(o.horizon == h) & (o.model == m)]
                ev, fc = g.y_true.values >= 300, g.y_pred.values >= 300
                hit = np.sum(ev & fc)
                srs.append(hit / fc.sum()); pods.append(hit / ev.sum())
            ax.plot(np.mean(srs), np.mean(pods), MARK[m], ms=9, color=COL[m], mec='white', mew=0.8, zorder=4, label=m)
        ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1); ax.set_aspect('equal')
        ax.set_xlabel('success ratio (1 - false-alarm ratio)')
        ax.set_title(f'{h} h ahead', fontsize=10)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(True)
    axes[0].set_ylabel('probability of detection')
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.925, 0.3, 0.011, 0.42])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label('critical success index', color=INK2); cb.outline.set_visible(False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=10, bbox_to_anchor=(0.45, -0.07), fontsize=8, handletextpad=0.2,
               columnspacing=0.9)
    n_ev = int((OOF[(OOF.horizon == 24) & (OOF.model == 'Persistence')].y_true >= 300).sum())
    header(fig, 'Performance diagram for "Very poor or worse" days (AQI >= 300), forward CV 2020-2023, mean of seeds 0-2',
           f'{n_ev} event days of 1,440. Upper right = better; dashed lines: frequency bias (1 = no over/under-forecasting).',
           top=0.82)
    save(fig, 'fig10_performance_diagram')


if __name__ == '__main__':
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10):
        f()
