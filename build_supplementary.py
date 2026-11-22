#!/usr/bin/env python3
"""Builds the Supplementary Material for the DualAQI manuscript.

Outputs (results/supplementary/):
  Supplementary_Material.pdf        title page, Sections S1-S6, Tables S1-S11, Figures S1-S16
  TableS4_benchmark_full.xlsx       full benchmark (seed means, both CV schemes, and per seed)
  TableS7_ablation_full.xlsx        full ablation (metrics and paired differences, 2020-2023 and 2022-2023)

All numbers are read from the result files, so the Supplementary always matches the analysis.

    .venv\\Scripts\\python build_supplementary.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import dualaqi as D

# ----------------------------------------------------------------------------- edit these two lines
TITLE = ('DualAQI: gated fusion of ground observations and satellite fire activity for 24-72 h AQI '
         'forecasting in Delhi')
AUTHORS = '[Author names and affiliations - replace]'
# -----------------------------------------------------------------------------

OUT = os.path.join('results', 'supplementary')
os.makedirs(OUT, exist_ok=True)
FIG1, FIG2, FIG3, FIG4 = (os.path.join('results', 'figures', f'objective{i}') for i in (1, 2, 3, 4))
H = [24, 48, 72]

ttf = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DV', os.path.join(ttf, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVB', os.path.join(ttf, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DVI', os.path.join(ttf, 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')
INK, MUTED, RULE, HEAD = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#c9c7c0'), colors.HexColor('#f0efec')
S = {
    'title': ParagraphStyle('t', fontName='DVB', fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=14),
    'sub': ParagraphStyle('s', fontName='DV', fontSize=11, leading=15, alignment=TA_CENTER, textColor=MUTED),
    'h1': ParagraphStyle('h1', fontName='DVB', fontSize=13, leading=17, spaceBefore=6, spaceAfter=8),
    'h2': ParagraphStyle('h2', fontName='DVB', fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4),
    'body': ParagraphStyle('b', fontName='DV', fontSize=9, leading=13, spaceAfter=5),
    'bul': ParagraphStyle('bl', fontName='DV', fontSize=9, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2),
    'cap': ParagraphStyle('c', fontName='DV', fontSize=8.2, leading=11, textColor=INK, spaceBefore=3, spaceAfter=10),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=7, leading=8.6),
    'cellb': ParagraphStyle('cellb', fontName='DVB', fontSize=7, leading=8.6),
    'toc': ParagraphStyle('toc', fontName='DV', fontSize=9.5, leading=14),
}
FRAME_W = A4[0] - 36 * mm
FRAME_H = A4[1] - 40 * mm


def P(text, style='body'):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S['bul'], bulletText='•') for t in items]


def table(df, widths=None, fs=7, caption=None, label=None):
    """DataFrame -> ReportLab table with a bold caption above (journal style)."""
    st = ParagraphStyle('x', parent=S['cell'], fontSize=fs, leading=fs * 1.25)
    sb = ParagraphStyle('y', parent=S['cellb'], fontSize=fs, leading=fs * 1.25)
    data = [[Paragraph(str(c), sb) for c in df.columns]]
    data += [[Paragraph('' if (isinstance(v, float) and np.isnan(v)) else str(v), st) for v in row] for row in df.values]
    if widths is None:
        widths = [FRAME_W / len(df.columns)] * len(df.columns)
    else:
        widths = [w / sum(widths) * FRAME_W for w in widths]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEAD), ('LINEABOVE', (0, 0), (-1, 0), 0.8, INK),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK), ('LINEBELOW', (0, -1), (-1, -1), 0.8, INK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fafaf8')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    out = []
    if caption:
        out.append(P(f'<b>{label}.</b> {caption}', 'cap'))
    out.append(t)
    out.append(Spacer(1, 10))
    return out


def figure(paths, label, caption):
    """One or more stacked images scaled to the frame, with a caption below."""
    paths = [paths] if isinstance(paths, str) else paths
    sizes = [PILImage.open(p).size for p in paths]
    heights = [FRAME_W * h / w for w, h in sizes]
    scale = min(1.0, (FRAME_H - 60) / sum(heights))
    items = [Image(p, width=FRAME_W * scale, height=hh * scale) for p, hh in zip(paths, heights)]
    return [KeepTogether(items + [P(f'<b>{label}.</b> {caption}', 'cap')])]


def fmt(x, d=2):
    return '' if pd.isna(x) else f'{x:.{d}f}'


def pm(a, b, d=2):
    return f'{a:.{d}f} ± {b:.{d}f}'


# ----------------------------------------------------------------------------- data for the tables
print('loading results ...')
import objective3_figures as F   # seed-averaged benchmark (forward and blocked) and per-seed forecasts
per_seed = {s: D.summarise(F.OOFS[s][F.COLS], 'forward', 500, 0).assign(seed=s) for s in F.OOFS}
PER = pd.concat(per_seed.values())
FWD, BLK = F.fwd.reset_index(), F.blk.reset_index()
ORDER = ['DualAQI v2', 'LSTM', 'Ensemble', 'RandomForest', 'XGBoost', 'GBM', 'Ridge', 'ARIMA', 'Persistence']
okey = lambda c: c.map({m: i for i, m in enumerate(ORDER)}) if c.name == 'model' else c


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DV', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 12 * mm, 'Supplementary Material')
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f'S{doc.page}')
    canvas.restoreState()


story = []
# ----------------------------------------------------------------------------- title page
story += [Spacer(1, 50 * mm), P('Supplementary Material', 'sub'), Spacer(1, 6), P(TITLE, 'title'), P(AUTHORS, 'sub'),
          Spacer(1, 18 * mm), P('<b>Contents</b>', 'h2')]
toc = ['Section S1. Data processing', 'Section S2. AQI conversion', 'Section S3. Causality tests',
       'Section S4. Analysis plan fixed in advance and its amendments', 'Section S5. ARIMA baseline details',
       'Section S6. Model hyperparameters (Table S3 is within this section)',
       'Tables S1-S11 (S4 and S7 are given in full in the accompanying Excel files)', 'Figures S1-S16']
story += [P(t, 'toc') for t in toc]
story += [Spacer(1, 10), P('Notation: RMSE, MAE and bias in AQI units; h = forecast horizon (24, 48, 72 h); '
                           'neural-model values are means over seeds 0-2 unless stated otherwise. All times UTC '
                           'unless marked IST (UTC+05:30).', 'body'), PageBreak()]

# ----------------------------------------------------------------------------- S1 data processing
fire = pd.read_csv('data/punjab_fire_daily.csv', parse_dates=['date'], index_col='date')
yr = fire.fire_count.groupby(fire.index.year).sum()
on = fire[fire.index.month.isin([10, 11])].fire_count.groupby(fire[fire.index.month.isin([10, 11])].index.year).sum()
story += [P('Section S1. Data processing', 'h1'), P('S1.1 Ground PM2.5', 'h2')]
story += bullets([
    'Source: hourly station files of the CPCB/DPCC/IITM network for Delhi (36 stations, one file per station and year).',
    'Time base: CPCB timestamps are IST and mark the start of each hourly averaging window. This was verified by matching '
    'four stations against OpenAQ records with explicit UTC windows. Each value is relabelled to the UTC hour at which it '
    'becomes available: stamp - 4 h (IST window end converted to UTC and rounded up to the next full hour).',
    'Quality control: values outside (0, 1000] µg/m³ are removed; blank and "NA" cells are missing. No gaps are filled.',
    'City value: arithmetic mean of all stations reporting in that hour, computed before conversion to AQI. Station counts per '
    'year are given in Table S11.'])
story += [P('S1.2 Meteorology (ERA5)', 'h2')]
story += bullets([
    'ERA5 hourly single-level reanalysis for the grid cell 28.75° N, 77.25° E, obtained through the Open-Meteo archive '
    'API (models = era5). Variables: 2 m temperature, 2 m dewpoint (converted to relative humidity with the Magnus formula), '
    '10 m u and v wind, surface pressure and boundary-layer height (BLH).',
    'Check against the Copernicus Climate Data Store files on 9,504 overlapping hours of 2023: mean absolute differences '
    '0.03 K (t2m, d2m), 0.03 m/s (u10, v10), 0.7 hPa (pressure) and 1.2 m (BLH); correlations ≥ 0.999.',
    'Gap: the source has no BLH for 1 January - 30 June 2024. Forecast origins that need those hours are dropped for every '
    'model alike (in training and testing); no values were filled from another product.'])
story += [P('S1.3 Satellite fire activity (NASA FIRMS)', 'h2')]
story += bullets([
    'Product: VIIRS S-NPP 375 m active fires, standard processing (VIIRS_SNPP_SP), downloaded through the FIRMS area API '
    'in 5-day requests for a box around Punjab (73.7-77.1° E, 29.4-32.7° N).',
    'Filtering: detections inside the Punjab state polygon (GADM 4.1), presumed vegetation fires only (type 0), all confidence '
    'levels. Daily totals of fire count and fire radiative power (FRP, MW) per UTC day, 1 December 2018 - 31 December 2025; '
    'days without detections are true zeros. No missing days.',
    'Annual detections: ' + '; '.join(f'{y}: {v/1000:.1f}k (Oct-Nov {on.get(y, 0)/1000:.1f}k)' for y, v in yr.items() if y >= 2019) + '.'])
story += [P('S1.4 Calendar inputs', 'h2')]
story += bullets([
    'Hour of day, day of week and month as sine/cosine pairs; a binary festival flag for the origin day and each target day.',
    'Festival dates in the model: Diwali 27 Oct 2019, 14 Nov 2020, 4 Nov 2021, 24 Oct 2022, 12 Nov 2023. '
    '<b>Limitation:</b> Diwali 2024 (1 Nov) and 2025 (20 Oct) were not in the list, so they were not flagged in the '
    'July 2024 - December 2025 held-out test.'])

# ----------------------------------------------------------------------------- S2 AQI
story += [P('Section S2. AQI conversion', 'h1'),
          P('The AQI is a piecewise-linear function of the hourly city-mean PM2.5 concentration (Table S1). Concentrations '
            'above 350 µg/m³ continue the final segment (slope 1 AQI per µg/m³). Categories use the boundaries 50, 100, 200, '
            '300 and 400. Because the input is hourly PM2.5 and not the official 24-hour average, values are not directly '
            'comparable with official CPCB AQI bulletins. The breakpoints should be checked against the official CPCB table '
            'before publication.')]
bp = pd.DataFrame({'PM2.5 (µg/m³)': ['0-30', '30-60', '60-90', '90-120', '120-250', '250-350', '> 350'],
                   'AQI': ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500', '500 + (PM2.5 - 350)'],
                   'Category': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very poor', 'Severe', 'Severe']})
story += table(bp, widths=[1, 1.2, 1], fs=8, caption='PM2.5-to-AQI breakpoints used in the study.', label='Table S1')

# ----------------------------------------------------------------------------- S3 causality
story += [P('Section S3. Causality tests', 'h1'),
          P('Rule tested: the fire count, FRP and wind-transport value for UTC day d may be used only by forecast origins at or '
            'after 03:00 UTC on day d+1; all other inputs end at the origin. Results (test_causality.py):')]
caus = pd.DataFrame([
    ['Feature test, v1 inputs', '300 origins: all fire counts after the last usable day randomised', '0 of 300 changed', '300 of 300 changed'],
    ['Feature test, v2 inputs', '150 origins: future fire counts, FRP and ERA5 u10/v10 randomised', '0 of 150 changed', '150 of 150 changed'],
    ['Timing check', 'Every used fire day d satisfies origin ≥ d+1 03:00 UTC; the next day would not yet be allowed', 'true for all', '-'],
    ['End-to-end test', 'Forward fold 0 retrained after randomising fire counts from 22 Feb 2020', '46 earlier forecasts: max change 0.0',
     '7 of 7 forecasts using the altered day changed'],
], columns=['Test', 'Perturbation', 'Forecasts / features that must not change', 'Negative control (must change)'])
story += table(caus, widths=[1.2, 2.6, 1.5, 1.5], fs=7.5, caption='Causality tests and results.', label='Table S2')

# ----------------------------------------------------------------------------- S4 plan
story += [P('Section S4. Analysis plan fixed in advance and its amendments', 'h1'),
          P('The full plan is archived with the code (results/dualaqi_v2/PLAN.md). Summary:')]
story += bullets([
    '<b>Original plan (before any v2 result):</b> development on forward folds 0-7 (test origins 8 Jan 2020 - 27 Dec 2021); '
    'final test on folds 8-15 (28 Dec 2021 - 17 Dec 2023), run once. Candidates on top of v1 with early stopping: '
    'B (early stopping only), C (+ FRP), D (+ wind transport), E (+ 14/30-day fire sums), F (all three). Selection: lowest '
    'development RMSE averaged over horizons and seeds; ties (< 0.1 AQI) go to the simpler candidate.',
    '<b>Amendment 1 (after the development stage, before any test-fold run):</b> early stopping lowered accuracy for both '
    'DualAQI and the single-branch LSTM (development RMSE 87.8 → 93.7 and 89.8 → 98.9), so it was dropped for both. '
    'Candidates were re-run without it, and v2 was chosen among v1, v1 + FRP, v1 + transport, v1 + sums and all three '
    '(Table S6).',
    '<b>Amendment 2:</b> a second held-out period (July 2024 - December 2025) was added, written before any 2024-2025 data '
    'were built. Models were fixed (v2, v1, single-branch LSTM; seeds 0-2). An improvement is claimed only if both bootstrap '
    'intervals exclude zero.',
    '<b>Amendment 3 (data check, before any 2024-2025 model run):</b> BLH is missing in the source for January-June 2024, '
    'so the second test covers July 2024 - December 2025.',
    'Caveat: v1 and LSTM results on 2020-2023 had been seen before v2 was designed; no v2 design choice used test-fold results.'])

# ----------------------------------------------------------------------------- S5 ARIMA
story += [P('Section S5. ARIMA baseline details', 'h1')]
story += bullets([
    'Model: log PM2.5 = daily Fourier terms (2 harmonics) + annual Fourier terms (2 harmonics, period 365.25 days) + '
    'ARMA(p, q) errors with a constant.',
    'Fitting: once per fold on the training windows only (all other hours set to missing, which the Kalman filter skips). '
    'Order chosen by AIC among (1,0,0), (2,0,1) and (2,0,2), using only fits whose likelihood optimisation converged '
    '(up to 1,000 iterations).',
    'Forecasting: each origin is filtered through its own previous 168 h with the fitted parameters; the 24/48/72 h log '
    'forecasts are converted with the lognormal mean exp(μ + σ²/2), using the model\'s h-step forecast variance, then to AQI.',
    'Convergence: (2,0,2) was chosen in all folds except forward fold 9 (March-June 2022), where (2,0,2) did not converge and '
    'the best converged fit (2,0,1) was used. This changed the forward-CV ARIMA RMSE by at most 0.07 AQI.'])

# ----------------------------------------------------------------------------- S6 hyperparameters
story += [P('Section S6. Model hyperparameters', 'h1'),
          P('All hyperparameters were fixed in advance and were not tuned per fold or horizon.')]
hp = pd.DataFrame([
    ['Persistence', 'Forecast = current AQI at every horizon'],
    ['ARIMA', 'Fourier regression + ARMA errors on log PM2.5; order by AIC from (1,0,0), (2,0,1), (2,0,2) (Section S5)'],
    ['Ridge', 'α = 1; median imputation; standardised features'],
    ['GBM', 'HistGradientBoosting: 300 iterations, learning rate 0.05, 31 leaves, min. 40 samples per leaf, L2 = 1'],
    ['RandomForest', '200 trees, min. 5 samples per leaf, 30% of features per split'],
    ['XGBoost', '300 trees, learning rate 0.05, max. depth 6, 80% row and column subsampling, min. child weight 5, λ = 1'],
    ['Ensemble', 'Equal-weight mean of Ridge, GBM and RandomForest'],
    ['LSTM (baseline)', 'LSTM 64 units over 168 h × 8 inputs; linear output to 3 horizons'],
    ['DualAQI v2', 'LSTM 8→64; event MLP 24→64→32; gate 96→64; projection 32→64; head 64→64→ReLU→Dropout 0.3→3; 35,299 parameters'],
    ['Single-branch LSTM', 'DualAQI without the event branch (LSTM + head); 23,299 parameters'],
    ['Calendar-only', 'DualAQI with only the 10 calendar inputs in the event branch (MLP 10→64→32)'],
    ['Neural training', 'Adam, learning rate 1e-3, 15 epochs, batch 128, gradient-norm clip 1.0, loss weights 0.25/0.35/0.40, '
                        'targets standardised with training-fold statistics, seeds 0, 1, 2, CPU'],
    ['Tabular features', 'PM2.5 lags 0, 1, 2, 3, 6, 12, 24, 48, 72, 168 h; 24/72/168 h means; 24 h SD; ERA5 at the origin; wind '
                         'speed; 24 h mean BLH; current AQI; target-time calendar terms'],
], columns=['Model', 'Settings'])
story += table(hp, widths=[1, 4.2], fs=7.5, caption='Hyperparameters of all models.', label='Table S3')
story += [PageBreak()]

# ----------------------------------------------------------------------------- Table S1 benchmark
story += [P('Supplementary tables', 'h1')]
cols = ['RMSE', 'MAE', 'R2', 'Bias', 'CatAcc', 'Cat+-1', 'skill_RMSE_vs_persist', 'dRMSE_vs_persist_CI95_lo',
        'dRMSE_vs_persist_CI95_hi']
for scheme, src in (('forward', FWD), ('blocked', BLK)):
    t = src[src.model.isin(ORDER)].sort_values(['horizon', 'model'], key=okey)
    show = pd.DataFrame({'h': t.horizon.values, 'Model': t.model.values, 'RMSE': t.RMSE.map(fmt).values,
                         'MAE': t.MAE.map(fmt).values, 'R²': t.R2.map(lambda v: fmt(v, 3)).values,
                         'Bias': t.Bias.map(fmt).values, 'Cat. acc.': t.CatAcc.map(lambda v: fmt(v, 3)).values,
                         'Within 1 cat.': t['Cat+-1'].map(lambda v: fmt(v, 3)).values,
                         'Skill vs pers.': (t.skill_RMSE_vs_persist * 100).map(lambda v: fmt(v, 1) + '%').values,
                         'ΔRMSE vs pers. 95% CI': [f'[{fmt(a, 1)}, {fmt(b, 1)}]' if not pd.isna(a) else '' for a, b in
                                                   zip(t.dRMSE_vs_persist_CI95_lo, t.dRMSE_vs_persist_CI95_hi)]})
    lab = 'Table S4a' if scheme == 'forward' else 'Table S4b'
    cap = ('Benchmark under forward cross-validation, 2020-2023 (1,440 forecasts per horizon). ' if scheme == 'forward' else
           'Benchmark under blocked cross-validation, 2019-2023 (1,816 forecasts per horizon). DualAQI was evaluated with '
           'forward cross-validation only. ')
    cap += ('Seed-dependent models are means over seeds 0-2; Persistence, Ridge, GBM and ARIMA do not depend on the seed. '
            'The CI is for RMSE(model) - RMSE(Persistence), resampling whole days (500 resamples), averaged over seeds. '
            'Full values: TableS4_benchmark_full.xlsx.')
    story += table(show, widths=[0.4, 1.3, 0.7, 0.7, 0.6, 0.6, 0.65, 0.75, 0.8, 1.2], fs=6.8, caption=cap, label=lab)

# ----------------------------------------------------------------------------- Table S2 seeds
seed_dep = ['DualAQI v2', 'LSTM', 'Ensemble', 'RandomForest', 'XGBoost']
g = PER[PER.model.isin(seed_dep)].groupby(['model', 'horizon'])
sd = pd.DataFrame({m: g[m].mean() for m in ['RMSE', 'MAE', 'R2', 'CatAcc']})
sds = pd.DataFrame({m: g[m].std(ddof=1) for m in ['RMSE', 'MAE', 'R2', 'CatAcc']})
sd = sd.reset_index().sort_values(['horizon', 'model'], key=okey)
sds = sds.reset_index().sort_values(['horizon', 'model'], key=okey)
seed_tab = pd.DataFrame({'h': sd.horizon.values, 'Model': sd.model.values,
                         'RMSE': [pm(a, b) for a, b in zip(sd.RMSE, sds.RMSE)],
                         'MAE': [pm(a, b) for a, b in zip(sd.MAE, sds.MAE)],
                         'R²': [pm(a, b, 3) for a, b in zip(sd.R2, sds.R2)],
                         'Category accuracy': [pm(a, b, 3) for a, b in zip(sd.CatAcc, sds.CatAcc)]})
story += table(seed_tab, widths=[0.4, 1.3, 1.2, 1.2, 1.2, 1.3], fs=7.2, label='Table S5',
               caption='Mean ± SD over seeds 0, 1 and 2, forward cross-validation 2020-2023, for the seed-dependent models. '
                       'GBM, Ridge, ARIMA and Persistence give identical results for every seed.')

# ----------------------------------------------------------------------------- Table S3 v2 selection
dev = pd.read_csv('results/dualaqi_v2/dev_selection.csv')
names = {'v1': 'v1 (fire counts + calendar)', 'C2': 'v1 + FRP  → selected as v2', 'D2': 'v1 + wind transport',
         'E2': 'v1 + 14/30-day fire sums', 'F2': 'v1 + FRP + transport + sums', 'B': 'v1 + early stopping', 'C': 'B + FRP',
         'D': 'B + transport', 'E': 'B + fire sums', 'F': 'B + all three', 'LSTM-ES': 'Single-branch LSTM + early stopping',
         'LSTM-v1': 'Single-branch LSTM'}
dev = dev.sort_values('RMSE_mean_over_horizons')
dtab = pd.DataFrame({'Candidate': [names.get(m, m) for m in dev.model],
                     '24 h': [pm(a, b) for a, b in zip(dev.RMSE_24, dev.RMSE_24_sd)],
                     '48 h': [pm(a, b) for a, b in zip(dev.RMSE_48, dev.RMSE_48_sd)],
                     '72 h': [pm(a, b) for a, b in zip(dev.RMSE_72, dev.RMSE_72_sd)],
                     'Mean': dev.RMSE_mean_over_horizons.map(fmt).values})
story += table(dtab, widths=[2.6, 1.1, 1.1, 1.1, 0.7], fs=7.4, label='Table S6',
               caption='Selection of DualAQI v2 on the development period (forward folds 0-7, 8 Jan 2020 - 27 Dec 2021), RMSE '
                       'mean ± SD over seeds 0-2. Early-stopping variants were dropped under Amendment 1 (Section S4).')

# ----------------------------------------------------------------------------- Table S4 ablation
abl_sheets = {}
for tag, path_d, path_m, lab in (('2020-2023 (forward CV, 1,440 forecasts per horizon)', 'results/dualaqi/ablation_drmse_v2.csv',
                                  'results/dualaqi/ablation_metrics_v2.csv', 'Table S7a'),
                                 ('2022-2023 only (clean held-out years, 720 forecasts per horizon)',
                                  'results/dualaqi/ablation_drmse_v2_folds8-15.csv',
                                  'results/dualaqi/ablation_metrics_v2_folds8-15.csv', 'Table S7b')):
    d = pd.read_csv(path_d)
    abl_sheets[lab] = (pd.read_csv(path_m), d)
    short = {'event branch (spec ablation): DualAQI - single-branch': 'v2 - single-branch',
             'fire counts alone: DualAQI - calendar-only': 'v2 - calendar-only (fire inputs)',
             'calendar inputs alone: calendar-only - single-branch': 'calendar-only - single-branch'}
    d['Comparison'] = d.comparison.map(short)
    sig = lambda r: 'yes' if (r.excludes_zero_day and r.excludes_zero_7dblock) else ('day CI only' if r.excludes_zero_day else 'no')
    atab = pd.DataFrame({'Comparison': d.Comparison, 'Season': d.season, 'h': d.horizon, 'ΔRMSE': d.dRMSE.map(fmt),
                         '95% CI (day)': [f'[{fmt(a)}, {fmt(b)}]' for a, b in zip(d.CI95_day_lo, d.CI95_day_hi)],
                         '95% CI (7-day block)': [f'[{fmt(a)}, {fmt(b)}]' for a, b in zip(d.CI95_7dblock_lo, d.CI95_7dblock_hi)],
                         'Significant': [sig(r) for r in d.itertuples()]})
    corder_ = {'v2 - single-branch': 0, 'v2 - calendar-only (fire inputs)': 1, 'calendar-only - single-branch': 2}
    atab = atab.sort_values(['Comparison', 'Season', 'h'], key=lambda c: c.map(corder_) if c.name == 'Comparison' else c.map(
        {'all': 0, 'Oct-Nov': 1, 'rest of year': 2}) if c.name == 'Season' else c)
    story += table(atab, widths=[1.9, 0.9, 0.35, 0.6, 1.1, 1.2, 0.8], fs=7, label=lab,
                   caption=f'Ablation of DualAQI v2, paired RMSE differences, {tag}. Mean over seeds 0-2; negative = first model '
                           'better. "Significant" requires both intervals to exclude zero. Full metrics for each variant: '
                           'TableS7_ablation_full.xlsx.')

# ----------------------------------------------------------------------------- Table S5 held-out tests
rows = []
for per_lab, mp, dp in (('2022-2023', 'results/dualaqi_v2/test_metrics.csv', 'results/dualaqi_v2/test_drmse.csv'),
                        ('Jul 2024 - Dec 2025', 'results/dualaqi_v2/test2024_metrics.csv', 'results/dualaqi_v2/test2024_drmse.csv')):
    m = pd.read_csv(mp)
    m = m[m.season == 'all']
    for r in m.sort_values(['horizon', 'model']).itertuples():
        rows.append([per_lab, r.horizon, r.model.replace('LSTM', 'Single-branch LSTM') if r.model == 'LSTM' else r.model,
                     pm(r.RMSE_mean, r.RMSE_sd), pm(r.MAE_mean, r.MAE_sd), pm(r.R2_mean, r.R2_sd, 3), fmt(r.Bias_mean, 1),
                     pm(r.CatAcc_mean, r.CatAcc_sd, 3)])
htab = pd.DataFrame(rows, columns=['Period', 'h', 'Model', 'RMSE', 'MAE', 'R²', 'Bias', 'Category accuracy'])
story += table(htab, widths=[1.2, 0.35, 1.3, 1.1, 1.1, 1.1, 0.6, 1.2], fs=7, label='Table S8a',
               caption='Held-out tests (each run once): mean ± SD over seeds 0-2.')
rows = []
for per_lab, dp in (('2022-2023', 'results/dualaqi_v2/test_drmse.csv'), ('Jul 2024 - Dec 2025', 'results/dualaqi_v2/test2024_drmse.csv')):
    d = pd.read_csv(dp)
    for r in d[d.season != 'rest of year'].itertuples():
        rows.append([per_lab, r.comparison.replace('- LSTM', '- single-branch LSTM'), r.season, r.horizon, fmt(r.dRMSE),
                     f'[{fmt(r.CI95_day_lo)}, {fmt(r.CI95_day_hi)}]', f'[{fmt(r.CI95_7dblock_lo)}, {fmt(r.CI95_7dblock_hi)}]',
                     'yes' if (r.excludes_zero_day and r.excludes_zero_7dblock) else 'no'])
d5 = pd.DataFrame(rows, columns=['Period', 'Comparison', 'Season', 'h', 'ΔRMSE', '95% CI (day)', '95% CI (7-day block)', 'Both exclude 0'])
story += table(d5, widths=[1.1, 1.8, 0.75, 0.35, 0.55, 1.05, 1.15, 0.7], fs=6.8, label='Table S8b',
               caption='Held-out tests: paired RMSE differences (mean over seeds; negative = first model better). '
                       'Rest-of-year rows are in the result files (results/dualaqi_v2/).')

# ----------------------------------------------------------------------------- Table S6 is in Section S6; Table S6b case study
import objective4_figures as G   # case-study metrics, neural models as seed means
cm = G.MET.reset_index()
corder = ['DualAQI v2', 'LSTM', 'Single-branch', 'Calendar-only', 'Ensemble', 'RandomForest', 'XGBoost', 'GBM', 'Ridge',
          'ARIMA', 'Persistence']
cm = cm[cm.model.isin(corder)].sort_values(['horizon', 'model'], key=lambda c: c.map(
    {m: i for i, m in enumerate(corder)}) if c.name == 'model' else c)
ctab = pd.DataFrame({'h': cm.horizon.values, 'Model': [m + (' *' if m in G.NEURAL else '') for m in cm.model],
                     'RMSE': cm.RMSE.map(lambda v: fmt(v, 1)).values, 'MAE': cm.MAE.map(lambda v: fmt(v, 1)).values,
                     'Bias': cm.Bias.map(lambda v: fmt(v, 1)).values, 'Cat. acc.': cm.CatAcc.map(fmt).values,
                     'Severe POD': cm.POD_severe.map(fmt).values, 'Severe FAR': cm.FAR_severe.map(fmt).values,
                     'V.poor+ POD': cm.POD_vpoor_plus.map(fmt).values, 'V.poor+ FAR': cm.FAR_vpoor_plus.map(fmt).values,
                     'Peak fc.': cm.peak_pred.map(lambda v: fmt(v, 0)).values})
story += table(ctab, widths=[0.35, 1.3, 0.6, 0.6, 0.6, 0.6, 0.65, 0.65, 0.7, 0.7, 0.6], fs=6.6, label='Table S9',
               caption='Case study, 20 October - 20 November 2023 (32 forecasts per horizon; 13 Severe days, observed peak 577 '
                       'at the 08:30 IST target hour). * mean over seeds 0-2; other models single run. POD = share of events '
                       'forecast; FAR = share of forecast events that did not occur; Severe = AQI ≥ 400, V.poor+ = AQI ≥ 300; '
                       'FAR is blank when the model never forecast the event.')

# bias by observed category (forward CV, mean over seeds)
CATS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very poor', 'Severe']
rows = []
for m in ORDER:
    for h in H:
        vals = []
        for o in F.OOFS.values():
            q = o[(o.model == m) & (o.horizon == h)]
            c = D.aqi_category(q.y_true.values)
            vals.append(pd.Series(q.y_pred.values - q.y_true.values).groupby(c).mean().reindex(range(6)))
        rows.append([m, h] + [fmt(v, 1) for v in pd.concat(vals, axis=1).mean(axis=1)])
btab = pd.DataFrame(rows, columns=['Model', 'h'] + CATS)
q = F.OOFS[0][(F.OOFS[0].model == 'Persistence') & (F.OOFS[0].horizon == 24)]
ncat = np.bincount(D.aqi_category(q.y_true.values), minlength=6)
story += table(btab, widths=[1.3, 0.35] + [0.8] * 6, fs=7, label='Table S10',
               caption='Bias (forecast - observed AQI) by observed AQI category, forward cross-validation 2020-2023, mean over '
                       'seeds 0-2. Forecasts per category: ' + ', '.join(f'{c} {n}' for c, n in zip(CATS, ncat)) + '.')

# ----------------------------------------------------------------------------- Table S8 stations
sh = pd.read_csv('data/pm25_station_hours.csv')
yrs = [c for c in sh.columns if c != 'station']
hours = {y: (8784 if int(y) % 4 == 0 else 8760) for y in yrs}
stab = pd.DataFrame({'Station (site_id, name, agency)': sh.station.str.replace('_', ' ')})
for y in yrs:
    stab[y] = (sh[y] / hours[y] * 100).map(lambda v: f'{v:.0f}')
story += table(stab, widths=[3.0] + [0.55] * len(yrs), fs=6.6, label='Table S11',
               caption='The 36 CPCB/DPCC/IITM stations used for city-mean PM2.5 and the share of hours with a valid value (%) '
                       'per year after quality control.')
story += [PageBreak()]

# ----------------------------------------------------------------------------- figures
story += [P('Supplementary figures', 'h1')]
figs = [
    (f'{FIG1}/fig02b_fire_density_by_year.png', 'Paddy-residue burning in Punjab, October-November of each year 2019-2025 '
     '(VIIRS S-NPP detections inside the Punjab boundary; same colour scale in every panel).'),
    (f'{FIG1}/fig02c_source_receptor_map.png', 'Source-receptor geometry: summed fire radiative power in Punjab (Oct-Nov '
     '2019-2025), distance rings from Delhi, the WNW-NW wind sector, and the distribution of Oct-Nov wind directions at Delhi.'),
    (f'{FIG1}/fig15_v2_selection.png', 'Selection of DualAQI v2 on the development period (mean ± SD over seeds 0-2). '
     'Early-stopping variants (grey) were dropped (Section S4).'),
    (f'{FIG1}/fig24_category_confusion.png', 'AQI category confusion matrices for DualAQI v2 at 24 h and 72 h (forward '
     'cross-validation 2020-2023, seed 0).'),
    (f'{FIG3}/fig07_seed_stability.png', 'Forward-CV RMSE of each model for seeds 0, 1 and 2.'),
    (f'{FIG3}/fig04_rank_bump.png', 'Model ranks across five metrics (mean of seeds 0-2, averaged over horizons).'),
    (f'{FIG3}/fig05_taylor_diagram.png', 'Taylor diagram at 24 h and 72 h (mean of seeds 0-2).'),
    (f'{FIG3}/fig06_blocked_vs_forward.png', 'Blocked vs forward cross-validation RMSE for the baseline models.'),
    (f'{FIG3}/fig08_error_distribution.png', 'Distribution of forecast errors for five key models (seeds 0-2 pooled).'),
    (f'{FIG3}/fig09_season_rmse.png', 'RMSE by IMD season of the target date (mean of seeds 0-2).'),
    (f'{FIG3}/fig10_performance_diagram.png', 'Performance diagram for Very-poor-or-worse days (AQI ≥ 300), mean of seeds 0-2.'),
    (f'{FIG1}/fig19_paired_differences.png', 'Paired RMSE differences on the two held-out periods (mean over seeds; thick: '
     'day bootstrap, thin: 7-day block bootstrap).'),
    (f'{FIG4}/fig3_scorecard.png', 'Case-study scorecard, 20 October - 20 November 2023 (neural models: mean of seeds 0-2).'),
    (f'{FIG4}/fig6_observed_vs_forecast.png', 'Case study: observed vs forecast AQI (neural models: all three seeds shown).'),
    ([f'{FIG1}/fig04_pm25_overview.png', f'{FIG1}/fig05_aqi_categories_by_year.png'],
     'Delhi air quality 2019-2025: daily AQI, seasonal and diurnal cycles (top), and share of days per AQI category by year (bottom).'),
    ([f'{FIG1}/fig08_meteorology_climatology.png', f'{FIG1}/fig09_wind_direction.png'],
     'ERA5 meteorology at Delhi: monthly climatology of daily means (top) and daily wind direction in Oct-Nov vs Jun-Sep (bottom).'),
]
for i, (p, cap) in enumerate(figs, 1):
    story += figure(p, f'Figure S{i}', cap)

doc = SimpleDocTemplate(os.path.join(OUT, 'Supplementary_Material.pdf'), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=18 * mm, bottomMargin=20 * mm, title='Supplementary Material', author=AUTHORS)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', os.path.join(OUT, 'Supplementary_Material.pdf'))

# ----------------------------------------------------------------------------- Excel files
with pd.ExcelWriter(os.path.join(OUT, 'TableS4_benchmark_full.xlsx')) as xw:
    FWD[FWD.model.isin(ORDER)].sort_values(['horizon', 'model'], key=okey).to_excel(xw, sheet_name='forward_seed_mean', index=False)
    BLK[BLK.model.isin(ORDER)].sort_values(['horizon', 'model'], key=okey).to_excel(xw, sheet_name='blocked_seed_mean', index=False)
    PER[PER.model.isin(ORDER)].sort_values(['seed', 'horizon', 'model'], key=okey).to_excel(xw, sheet_name='forward_per_seed', index=False)
    pd.DataFrame({'note': ['Forward CV 2020-2023 (1,440 forecasts/horizon), blocked CV 2019-2023 (1,816). Seed means over '
                           'seeds 0-2; Persistence, Ridge, GBM, ARIMA are seed-independent. DualAQI v2 forward CV only.',
                           'skill_RMSE_vs_persist = 1 - RMSE/RMSE_persistence; dRMSE CI: RMSE(model) - RMSE(persistence), '
                           'day bootstrap, 500 resamples.']}).to_excel(xw, sheet_name='README', index=False)
print('wrote', os.path.join(OUT, 'TableS4_benchmark_full.xlsx'))
with pd.ExcelWriter(os.path.join(OUT, 'TableS7_ablation_full.xlsx')) as xw:
    for lab, (m, d) in abl_sheets.items():
        tag = '2020_2023' if lab.endswith('a') else '2022_2023'
        m.to_excel(xw, sheet_name=f'metrics_{tag}', index=False)
        d.to_excel(xw, sheet_name=f'drmse_{tag}', index=False)
    pd.DataFrame({'note': ['Ablation of DualAQI v2: DualAQI v2, calendar-only event branch, single-branch LSTM; forward CV; '
                           'mean +- SD over seeds 0-2; season by target date.',
                           'dRMSE: mean over seeds of RMSE(first) - RMSE(second); CI95_day: day bootstrap; CI95_7dblock: 7-day '
                           'block bootstrap; 500 resamples.']}).to_excel(xw, sheet_name='README', index=False)
print('wrote', os.path.join(OUT, 'TableS7_ablation_full.xlsx'))
