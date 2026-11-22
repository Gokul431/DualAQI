#!/usr/bin/env python3
"""
DualAQI and baseline models for 24/48/72 h AQI forecasting in Delhi (CPU-only, retrospective).

Causality rules (every input is available at forecast time):
  * A forecast origin t is a launch time (03:00 UTC = 08:30 IST in the study). Only data at or
    before t is used.
  * Ground PM2.5 features are forward-filled at most 3 h. Targets are never imputed.
  * ERA5 meteorology features come only from times at or before t.
  * The Punjab fire count of UTC day d is used only from 03:00 UTC on day d+1.
  * The target for horizon h is city PM2.5 at t+h, converted to AQI.

Inputs (CSV in --data-dir, built by build_inputs.py; see README.md for the schema):
  pm25_hourly.csv        datetime, pm25                  city-mean PM2.5 (mean of stations, before AQI conversion)
  era5_hourly.csv        datetime, t2m, rh, u10, v10, sp, blh
  punjab_fire_daily.csv  date, fire_count, frp_sum       daily VIIRS fire detections in Punjab
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError:  # the classical models still run without torch
    torch = None

try:
    from xgboost import XGBRegressor
except ImportError:  # XGBoost baseline skipped
    XGBRegressor = None

HORIZONS = (24, 48, 72)
LOOKBACK_H = 168
MAX_H = max(HORIZONS)
MET_VARS = ['t2m', 'rh', 'u10', 'v10', 'sp', 'blh']
LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]
SEQ_COLS = ['aqi_now', 'ws'] + MET_VARS
LOSS_W = np.array([0.25, 0.35, 0.40], dtype=np.float32)  # loss weights for the 24, 48 and 72 h outputs
ARIMA_ORDERS = [(1, 0, 0), (2, 0, 1), (2, 0, 2)]  # ARMA error orders; one picked per fold by AIC
FIRE_LAGS = 7             # completed fire days seen by the event branch (same 7-day window as the PM2.5 history)
FIRE_AVAILABLE_HOUR = 3   # the fire count for UTC day d may be used from 03:00 UTC on day d+1

# Diwali dates 2019-2023. Diwali 2024-2025, Holi and Eid are not flagged (a stated limitation).
FESTIVAL_DATES = pd.to_datetime(['2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12'])

# Piecewise-linear PM2.5 -> AQI map on the CPCB National AQI breakpoints. The open-ended Severe band
# (> 250 ug/m3) is mapped to 401-500 up to 350 ug/m3 and continued with slope 1 above (Supplementary Table S1).
AQI_KNOTS_C = np.array([0., 30., 60., 90., 120., 250., 350.])
AQI_KNOTS_I = np.array([0., 50., 100., 200., 300., 400., 500.])
CAT_EDGES = np.array([50., 100., 200., 300., 400.])  # category index 0..5


# ----------------------------------------------------------------------------- AQI

def pm25_to_aqi(c):
    c = np.asarray(c, dtype=float)
    aqi = np.interp(c, AQI_KNOTS_C, AQI_KNOTS_I)
    above = c > AQI_KNOTS_C[-1]
    aqi[above] = AQI_KNOTS_I[-1] + (c[above] - AQI_KNOTS_C[-1])
    aqi[np.isnan(c)] = np.nan
    return aqi


def aqi_series(s):
    return pd.Series(pm25_to_aqi(s.values), index=s.index)


def aqi_category(aqi):
    return np.digitize(aqi, CAT_EDGES)


# ----------------------------------------------------------------------------- loading

def load_inputs(data_dir):
    pm = pd.read_csv(os.path.join(data_dir, 'pm25_hourly.csv'), parse_dates=['datetime'],
                     index_col='datetime').sort_index()
    met = pd.read_csv(os.path.join(data_dir, 'era5_hourly.csv'), parse_dates=['datetime'],
                      index_col='datetime').sort_index()
    idx = pd.date_range(pm.index.min(), pm.index.max(), freq='h')
    return pd.concat([pm[['pm25']], met[MET_VARS]], axis=1).reindex(idx)


def load_fire(path, column='fire_count'):
    """One column of the daily Punjab fire file (fire_count or frp_sum). Every day, no negatives."""
    if not path or not os.path.exists(path):
        return None
    f = pd.read_csv(path, parse_dates=['date'], index_col='date').sort_index()[column]
    full = pd.date_range(f.index.min(), f.index.max(), freq='D')
    if len(f) != len(full) or not f.index.equals(full) or f.isna().any():
        raise SystemExit(f'{path}: missing or duplicate days')
    if (f < 0).any():
        raise SystemExit(f'{path}: negative fire counts')
    return f.astype(float)


def last_usable_fire_day(origins):
    """Latest UTC day whose fire count is available at each origin (day d is usable from d+1 03:00 UTC)."""
    return (pd.DatetimeIndex(origins) - pd.Timedelta(hours=FIRE_AVAILABLE_HOUR)).normalize() - pd.Timedelta(days=1)


def fire_lag_matrix(origins, fire):
    """log1p fire counts for the FIRE_LAGS most recent usable days; column 0 is the most recent."""
    last = last_usable_fire_day(origins)
    return np.column_stack([np.log1p(fire.reindex(last - pd.Timedelta(days=k)).values) for k in range(FIRE_LAGS)])


def event_calendar(origins):
    """Calendar inputs of the event branch: hour, day of week, month (sin/cos) and festival flags
    for the origin day and each target day (festival dates are known in advance)."""
    ts = pd.DatetimeIndex(origins)
    hod = ts.hour.values + ts.minute.values / 60.0
    dow = ts.day_of_week.values
    mon = ts.month.values - 1
    cols = [np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
            ts.normalize().isin(FESTIVAL_DATES).astype(float)]
    cols += [(ts + pd.Timedelta(hours=h)).normalize().isin(FESTIVAL_DATES).astype(float) for h in HORIZONS]
    return np.column_stack(cols)


N_CAL = 7 + len(HORIZONS)       # calendar columns, always the last columns of the event features
EVENT_OPTS = ('frp', 'transport', 'long')
TRANSPORT_TO_DEG = 135.0        # air moving towards the south-east, i.e. from Punjab (north-west of Delhi)


def daily_transport(hourly):
    """Daily mean (UTC day) of the wind component blowing from the north-west towards Delhi, m/s.
    Day d uses only ERA5 hours of day d, so it obeys the same availability rule as fire day d."""
    a = np.deg2rad(TRANSPORT_TO_DEG)
    tr = hourly['u10'] * np.sin(a) + hourly['v10'] * np.cos(a)
    return tr.groupby(tr.index.normalize()).mean()


def lag_matrix(series, last, n=FIRE_LAGS):
    return np.column_stack([series.reindex(last - pd.Timedelta(days=k)).values for k in range(n)])


def event_features(origins, fire, opts=(), frp=None, transport=None):
    """Event-branch inputs. opts=() is DualAQI v1: 7 log1p fire lags + calendar.
    'frp': 7 log1p FRP lags; 'transport': 7 daily NW->SE wind lags and log1p(fire) x max(wind, 0);
    'long': log1p of the 14- and 30-day fire sums. All end at the last usable fire day."""
    last = last_usable_fire_day(origins)
    fl = fire_lag_matrix(origins, fire)
    cols = [fl]
    if 'frp' in opts:
        cols.append(np.log1p(lag_matrix(frp, last)))
    if 'transport' in opts:
        T = lag_matrix(transport, last)
        cols += [T, fl * np.clip(T, 0, None)]
    if 'long' in opts:
        for w in (14, 30):
            cols.append(np.log1p(fire.rolling(w, min_periods=w).sum().reindex(last).values)[:, None])
    cols.append(event_calendar(origins))
    return np.hstack(cols).astype(np.float32)


# ----------------------------------------------------------------------------- features

def base_features(hourly):
    pm = hourly['pm25'].ffill(limit=3)
    met = hourly[MET_VARS].ffill(limit=3)
    f = pd.DataFrame(index=hourly.index)
    for L in LAGS:
        f[f'pm_lag{L}'] = pm.shift(L)
    f['pm_mean24'] = pm.rolling(24, min_periods=18).mean()
    f['pm_mean72'] = pm.rolling(72, min_periods=54).mean()
    f['pm_mean168'] = pm.rolling(168, min_periods=126).mean()
    f['pm_std24'] = pm.rolling(24, min_periods=18).std()
    for v in MET_VARS:
        f[v] = met[v]
    f['ws'] = np.hypot(met['u10'], met['v10'])
    f['blh_mean24'] = met['blh'].rolling(24, min_periods=18).mean()
    f['aqi_now'] = aqi_series(pm)
    return f


def calendar_features(ts):
    ts = pd.DatetimeIndex(ts)
    hod = ts.hour.values + ts.minute.values / 60.0
    doy = ts.dayofyear.values
    dow = ts.dayofweek.values
    return pd.DataFrame({
        'hod_s': np.sin(2 * np.pi * hod / 24), 'hod_c': np.cos(2 * np.pi * hod / 24),
        'doy_s': np.sin(2 * np.pi * doy / 365.25), 'doy_c': np.cos(2 * np.pi * doy / 365.25),
        'dow_s': np.sin(2 * np.pi * dow / 7), 'dow_c': np.cos(2 * np.pi * dow / 7),
        'festival': ts.normalize().isin(FESTIVAL_DATES).astype(int),
    }, index=ts)


def build_samples(hourly, launch_hours, fire=None, event_opts=(), frp=None):
    base = base_features(hourly)
    pm = hourly['pm25']
    origins_all = base.index[base.index.hour.isin(launch_hours)]
    Xb_all = base.reindex(origins_all)
    Y_all = pd.DataFrame({h: aqi_series(pm.shift(-h)).reindex(origins_all).values for h in HORIZONS},
                         index=origins_all)
    keep = Xb_all.notna().all(axis=1).values & Y_all.notna().all(axis=1).values
    if fire is not None:  # same origins for DualAQI and --no-fire: only those with all fire lags present
        keep &= np.isfinite(fire_lag_matrix(origins_all, fire)).all(axis=1)
    origins = origins_all[keep]
    Xb = Xb_all[keep]
    Y = Y_all[keep]

    X = {}
    for h in HORIZONS:
        cal = calendar_features(origins + pd.Timedelta(hours=h))
        cal.index = origins
        X[h] = pd.concat([Xb, cal], axis=1)

    # Sequence windows: the 168 hourly rows ending at the origin.
    pos = hourly.index.get_indexer(origins)
    win = pos[:, None] - np.arange(LOOKBACK_H - 1, -1, -1)[None, :]
    seq_arr = base[SEQ_COLS].ffill(limit=24).fillna(0.0).values.astype(np.float32)
    seq = seq_arr[win]

    event = None
    if fire is not None:
        transport = daily_transport(hourly) if 'transport' in event_opts else None
        event = event_features(origins, fire, event_opts, frp=frp, transport=transport)
        if not np.isfinite(event).all():
            raise SystemExit(f'event features incomplete for {int((~np.isfinite(event)).any(axis=1).sum())} origins '
                             f'(opts {event_opts}); extend the fire file further back')
    return {'origins': origins, 'Xb': Xb, 'X': X, 'Y': Y, 'seq': seq, 'event': event}


# ----------------------------------------------------------------------------- models

def fit_predict_classical(Xtr, ytr, Xte, seed):
    out = {}
    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
    out['Ridge'] = ridge.fit(Xtr, ytr).predict(Xte)
    gbm = make_pipeline(SimpleImputer(strategy='median'),
                        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                      min_samples_leaf=40, l2_regularization=1.0,
                                                      random_state=seed))
    out['GBM'] = gbm.fit(Xtr, ytr).predict(Xte)
    rf = make_pipeline(SimpleImputer(strategy='median'),
                       RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.3,
                                             n_jobs=-1, random_state=seed))
    out['RandomForest'] = rf.fit(Xtr, ytr).predict(Xte)
    if XGBRegressor is not None:  # handles missing values natively, no imputer
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                           tree_method='hist', n_jobs=-1, random_state=seed)
        out['XGBoost'] = xgb.fit(Xtr, ytr).predict(Xte)
    return out


def fourier_terms(index):
    """Deterministic calendar regressors: daily (K=2) and annual (K=2) harmonics. Known in advance."""
    ts = pd.DatetimeIndex(index)
    hod = ts.hour.values + ts.minute.values / 60.0
    doy = ts.day_of_year.values + hod / 24.0
    cols = []
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * hod / 24), np.cos(2 * np.pi * k * hod / 24),
                 np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def arima_fold(pm, origins, tr, te):
    """ARIMA baseline: regression on daily + annual Fourier terms with ARIMA errors, on log PM2.5.

    * Fitted once per fold, using only hours inside the training origins' input and target windows
      (all other hours are masked as missing, which the Kalman filter skips).
    * The ARMA order is chosen per fold by AIC from ARIMA_ORDERS, on that fold's training data only,
      among the fits whose likelihood optimisation converged (up to 1000 iterations).
    * Each test origin is forecast from its own previous LOOKBACK_H hours with the fitted parameters.
      The log forecast is converted with the lognormal mean, exp(mu + var/2), using the model's
      h-step forecast variance, then mapped to AQI.
    """
    from statsmodels.tsa.arima.model import ARIMA
    y = np.log(pm.where(pm > 0))
    X = fourier_terms(pm.index)
    pos_tr = pm.index.get_indexer(origins[tr])
    cover = np.zeros(len(pm) + 1, dtype=int)
    np.add.at(cover, np.clip(pos_tr - LOOKBACK_H, 0, None), 1)
    np.add.at(cover, np.clip(pos_tr + MAX_H + 1, None, len(pm)), -1)
    train_y = y.where(np.cumsum(cover)[:len(pm)] > 0)
    a = pm.index.get_loc(train_y.first_valid_index())
    b = pm.index.get_loc(train_y.last_valid_index()) + 1
    fits = [ARIMA(train_y.values[a:b], exog=X[a:b], order=o, trend='c').fit(method_kwargs={'maxiter': 1000})
            for o in ARIMA_ORDERS]
    converged = [r for r in fits if r.mle_retvals.get('converged', True)]  # never pick a fit that did not converge
    res = min(converged or fits, key=lambda r: r.aic)
    pos_te = pm.index.get_indexer(origins[te])
    out = np.full((len(te), len(HORIZONS)), np.nan)
    steps = np.array(HORIZONS) - 1
    for i, p in enumerate(pos_te):
        lo = p - LOOKBACK_H + 1
        fc = res.apply(y.values[lo:p + 1], exog=X[lo:p + 1]).get_forecast(MAX_H, exog=X[p + 1:p + 1 + MAX_H])
        mu, var = fc.predicted_mean[steps], fc.var_pred_mean[steps]
        out[i] = pm25_to_aqi(np.exp(mu + var / 2))
    return out, res.model.order


if torch is not None:
    def mlp_head(hidden):
        return nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, len(HORIZONS)))

    class SeqLSTM(nn.Module):
        """Sequence baseline: LSTM over the 168-hour window with a linear output layer."""

        def __init__(self, n_in, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.head = nn.Linear(hidden, len(HORIZONS))

        def forward(self, seq):
            _, (hn, _) = self.lstm(seq)
            return self.head(hn[-1])

    class DualAQI(nn.Module):
        """Dual-branch LSTM: sequence branch (PM2.5 + ERA5, 168 h) and event branch (lagged Punjab
        fire counts + calendar), fused by a learned gate.

            h = LSTM(seq)                       (64)
            e = MLP(event)                      (32)
            g = sigmoid(W [h; e])               (64)
            fused = g * h + (1 - g) * P e       P: Linear(32 -> 64), so e can be mixed with h
            out = head(fused)                   Linear(64) -> ReLU -> Dropout 0.3 -> 3 horizons

        use_fire=False is the single-branch ablation: fused = h, everything else identical. The LSTM
        and head are created first, so both variants start from the same LSTM and head weights.
        """

        def __init__(self, n_seq, n_event=0, hidden=64, event_dim=32, use_fire=True):
            super().__init__()
            self.use_fire = use_fire
            self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
            self.head = mlp_head(hidden)
            if use_fire:
                self.event = nn.Sequential(nn.Linear(n_event, 64), nn.ReLU(), nn.Linear(64, event_dim))
                self.event_proj = nn.Linear(event_dim, hidden)
                self.gate = nn.Linear(hidden + event_dim, hidden)

        def forward(self, seq, event=None):
            _, (hn, _) = self.lstm(seq)
            h = hn[-1]
            if self.use_fire:
                e = self.event(event)
                g = torch.sigmoid(self.gate(torch.cat([h, e], dim=1)))
                h = g * h + (1 - g) * self.event_proj(e)
            return self.head(h)


def standardize(W_tr, W_te):
    F = W_tr.shape[2]
    flat = W_tr.reshape(-1, F)
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    return ((W_tr - mu) / sd).astype(np.float32), ((W_te - mu) / sd).astype(np.float32)


def train_nn(make_net, Xs_tr, Y_tr, Xs_te, seed, epochs=15, batch=128, lr=1e-3, reseed_after_init=False,
             early_stop=False, max_epochs=40, patience=5, val_frac=0.15, val_gap=3):
    """Train on the fold's training rows (in time order) and predict the test rows.

    early_stop: the last val_frac of the training rows are validation, with val_gap rows (days)
    dropped before them so their targets do not overlap the fitted rows; up to max_epochs,
    stop after `patience` epochs without a lower validation loss, restore the best weights.
    early_stop=False (the setting used for all reported results) trains for a fixed number of epochs.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ymu = Y_tr.mean(0)
    ysd = Y_tr.std(0) + 1e-6
    Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs_tr]
    Xe = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs_te]
    Yt = torch.from_numpy(((Y_tr - ymu) / ysd).astype(np.float32))
    w = torch.from_numpy(LOSS_W)
    net = make_net()
    if reseed_after_init:  # same minibatch order and dropout masks whatever extra layers the net has
        torch.manual_seed(seed + 10_000)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Yt)
    if early_stop:
        n_val = max(1, int(round(val_frac * n)))
        n_fit = n - n_val - val_gap
        val = torch.arange(n - n_val, n)
    else:
        n_fit = n
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    for ep in range(max_epochs if early_stop else epochs):
        net.train()
        perm = torch.randperm(n_fit)  # identical to the fixed-epoch path when early_stop is off
        for i in range(0, n_fit, batch):
            b = perm[i:i + batch]
            pred = net(*[x[b] for x in Xt])
            loss = (w * (pred - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if early_stop:
            net.eval()
            with torch.no_grad():
                vl = float((w * (net(*[x[val] for x in Xt]) - Yt[val]) ** 2).mean())
            if vl < best - 1e-6:
                best, best_ep, bad = vl, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if early_stop:
        net.load_state_dict(best_state)
        print(f'    early stopping: best epoch {best_ep}', flush=True)
    net.eval()
    with torch.no_grad():
        p = net(*Xe).numpy()
    return p * ysd + ymu


# ----------------------------------------------------------------------------- CV

def blocked_folds(times, k):
    """Contiguous time blocks. Training origins must not overlap the test block's input or target windows."""
    times = pd.DatetimeIndex(times)
    lb = pd.Timedelta(hours=LOOKBACK_H)
    mh = pd.Timedelta(hours=MAX_H)
    edges = np.linspace(0, len(times), k + 1).astype(int)
    folds = []
    for i in range(k):
        te = np.arange(edges[i], edges[i + 1])
        t0, t1 = times[te[0]], times[te[-1]]
        tr = np.where(((times + mh) < (t0 - lb)) | ((times - lb) > (t1 + mh)))[0]
        folds.append((tr, te))
    return folds


def forward_folds(times, start, block_days=90):
    """Expanding window: train on everything whose target ends before the test block's input starts."""
    times = pd.DatetimeIndex(times)
    lb = pd.Timedelta(hours=LOOKBACK_H)
    mh = pd.Timedelta(hours=MAX_H)
    folds = []
    s = pd.Timestamp(start)
    while s <= times[-1]:
        e = s + pd.Timedelta(days=block_days)
        te = np.where((times >= s) & (times < e))[0]
        tr = np.where((times + mh) < (s - lb))[0]
        if len(te) >= 30 and len(tr):  # skip tiny trailing blocks
            folds.append((tr, te))
        s = e
    return folds


def _rows(model, h, origin, fold, y, p):
    return pd.DataFrame({'model': model, 'horizon': h, 'origin': origin, 'fold': fold,
                         'y_true': y, 'y_pred': np.clip(p, 0, None)})


def standardize_rows(A_tr, A_te):
    mu = A_tr.mean(0)
    sd = A_tr.std(0) + 1e-6
    return ((A_tr - mu) / sd).astype(np.float32), ((A_te - mu) / sd).astype(np.float32)


def dualaqi_fold(seq, event, Y, tr, te, seed, use_fire, early_stop=False):
    """Train DualAQI (or its --no-fire ablation) on one fold; return test predictions (n_te, 3)."""
    seq_tr, seq_te = standardize(seq[tr], seq[te])
    Ytr = Y.values[tr].astype(np.float32)
    if use_fire:
        ev_tr, ev_te = standardize_rows(event[tr], event[te])
        make = lambda: DualAQI(seq.shape[2], event.shape[1], use_fire=True)
        return train_nn(make, [seq_tr, ev_tr], Ytr, [seq_te, ev_te], seed, reseed_after_init=True,
                        early_stop=early_stop)
    make = lambda: DualAQI(seq.shape[2], use_fire=False)
    return train_nn(make, [seq_tr], Ytr, [seq_te], seed, reseed_after_init=True, early_stop=early_stop)


def run_dualaqi_cv(origins, Xb, Y, folds, seq, event, seed, use_fire, name=None, early_stop=False,
                   fold_ids=None):
    """--only-dualaqi: Persistence (needed for skill scores) and one DualAQI variant only.
    fold_ids keeps the original fold numbering when only some folds are run."""
    name = name or ('DualAQI' if use_fire else 'DualAQI-noFire')
    out = []
    fold_ids = fold_ids if fold_ids is not None else range(len(folds))
    for fold_id, (tr, te) in zip(fold_ids, folds):
        t0 = time.time()
        p = dualaqi_fold(seq, event, Y, tr, te, seed, use_fire, early_stop)
        for j, h in enumerate(HORIZONS):
            out.append(_rows('Persistence', h, origins[te], fold_id, Y[h].values[te], Xb['aqi_now'].values[te]))
            out.append(_rows(name, h, origins[te], fold_id, Y[h].values[te], p[:, j]))
        print(f'  fold {fold_id}: {len(te)} test origins, {time.time() - t0:.1f}s', flush=True)
    return pd.concat(out, ignore_index=True)


def run_cv(origins, Xb, X, Y, folds, seq, use_nn, seed, pm=None):
    out = []
    for fold_id, (tr, te) in enumerate(folds):
        t0 = time.time()
        p_arima, arima_order = arima_fold(pm, origins, tr, te) if pm is not None else (None, None)
        for j, h in enumerate(HORIZONS):
            preds = fit_predict_classical(X[h].iloc[tr], Y[h].values[tr], X[h].iloc[te], seed)
            preds['Persistence'] = Xb['aqi_now'].values[te]
            preds['Ensemble'] = (preds['Ridge'] + preds['GBM'] + preds['RandomForest']) / 3.0
            if p_arima is not None:
                preds['ARIMA'] = p_arima[:, j]
            for m, p in preds.items():
                out.append(_rows(m, h, origins[te], fold_id, Y[h].values[te], p))
        if use_nn:
            seq_tr, seq_te = standardize(seq[tr], seq[te])
            Ytr = Y.values[tr].astype(np.float32)
            p_lstm = train_nn(lambda: SeqLSTM(seq.shape[2]), [seq_tr], Ytr, [seq_te], seed)
            for j, h in enumerate(HORIZONS):
                out.append(_rows('LSTM', h, origins[te], fold_id, Y[h].values[te], p_lstm[:, j]))
        extra = f', ARIMA order {arima_order}' if arima_order else ''
        print(f'  fold {fold_id}: {len(te)} test origins, {time.time() - t0:.1f}s{extra}', flush=True)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- metrics

def metrics(y, p):
    e = p - y
    cy, cp = aqi_category(y), aqi_category(p)
    return dict(n=len(y),
                R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                RMSE=np.sqrt(np.mean(e ** 2)),
                MAE=np.mean(np.abs(e)),
                Bias=np.mean(e),
                CatAcc=np.mean(cy == cp),
                **{'Cat+-1': np.mean(np.abs(cy - cp) <= 1)})


def drmse_ci(g, p, n_boot, seed):
    """95% CI for RMSE(model) - RMSE(persistence), resampling whole days."""
    assert (g.origin.values == p.origin.values).all(), 'origins must match'
    codes, _ = pd.factorize(g.origin.dt.normalize())
    se_m = (g.y_pred.values - g.y_true.values) ** 2
    se_p = (p.y_pred.values - p.y_true.values) ** 2
    D = codes.max() + 1
    sm = np.bincount(codes, se_m, minlength=D)
    sp = np.bincount(codes, se_p, minlength=D)
    cnt = np.bincount(codes, minlength=D)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, D, D)
        n = cnt[idx].sum()
        diffs[b] = np.sqrt(sm[idx].sum() / n) - np.sqrt(sp[idx].sum() / n)
    return np.percentile(diffs, [2.5, 97.5])


def summarise(oof, scheme, n_boot, seed):
    rows = []
    for h in HORIZONS:
        d_h = oof[oof.horizon == h]
        pers = d_h[d_h.model == 'Persistence'].sort_values('origin')
        pm = metrics(pers.y_true.values, pers.y_pred.values)
        for model, g in d_h.groupby('model'):
            g = g.sort_values('origin')
            m = metrics(g.y_true.values, g.y_pred.values)
            fold_r2 = [metrics(gf.y_true.values, gf.y_pred.values)['R2'] for _, gf in g.groupby('fold')]
            row = dict(scheme=scheme, horizon=h, model=model, n=m['n'], R2=m['R2'], RMSE=m['RMSE'],
                       MAE=m['MAE'], Bias=m['Bias'], CatAcc=m['CatAcc'], **{'Cat+-1': m['Cat+-1']},
                       skill_RMSE_vs_persist=(pm['RMSE'] - m['RMSE']) / pm['RMSE'],
                       skill_MAE_vs_persist=(pm['MAE'] - m['MAE']) / pm['MAE'],
                       dRMSE_vs_persist_CI95_lo=np.nan, dRMSE_vs_persist_CI95_hi=np.nan,
                       R2_fold_mean=np.mean(fold_r2),
                       R2_fold_std=np.std(fold_r2, ddof=1) if len(fold_r2) > 1 else np.nan,
                       folds=len(fold_r2))
            if model != 'Persistence':
                lo, hi = drmse_ci(g, pers, n_boot, seed)
                row['dRMSE_vs_persist_CI95_lo'] = lo
                row['dRMSE_vs_persist_CI95_hi'] = hi
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--launch-hours', default='8',
                    help='comma-separated launch hours in the timezone of the input timestamps '
                         '(UTC with build_inputs.py: 3 = 08:30 IST), e.g. 3 or 0,6,12,18')
    ap.add_argument('--no-arima', action='store_true', help='skip the ARIMA baseline')
    ap.add_argument('--k-blocks', type=int, default=10)
    ap.add_argument('--forward-start', default=None, help='first test date; default: first origin + 365 days')
    ap.add_argument('--no-nn', action='store_true', help='skip the LSTM')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--fire-file', default=None, help='daily fire counts; default: <data-dir>/punjab_fire_daily.csv')
    ap.add_argument('--only-dualaqi', action='store_true',
                    help='run only DualAQI (fire + sequence branches) and Persistence; baselines are not re-run')
    ap.add_argument('--no-fire', action='store_true',
                    help='with --only-dualaqi: drop the event branch (single-branch ablation), all else identical')
    ap.add_argument('--calendar-only', action='store_true',
                    help='with --only-dualaqi: event branch gets calendar features but no fire counts '
                         '(isolates the effect of the fire counts from the calendar inputs)')
    ap.add_argument('--schemes', default='blocked,forward', help='CV schemes to run, e.g. forward')
    ap.add_argument('--event-opts', default='', help=f'extra event inputs for DualAQI, comma-separated from {EVENT_OPTS}')
    ap.add_argument('--early-stop', action='store_true',
                    help='DualAQI variants: early stopping on the last 15%% of each fold\'s training origins')
    ap.add_argument('--fold-range', default=None, help='run only folds a:b of each scheme (Python slice), e.g. 0:8')
    ap.add_argument('--model-name', default=None, help='name for the DualAQI variant in the outputs')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hourly = load_inputs(args.data_dir)
    hours = [int(x) for x in args.launch_hours.split(',')]
    fire = load_fire(args.fire_file or os.path.join(args.data_dir, 'punjab_fire_daily.csv'))
    if args.only_dualaqi and fire is None:
        raise SystemExit('--only-dualaqi needs the fire file (also for --no-fire, so both use the same origins)')
    if args.only_dualaqi and torch is None:
        raise SystemExit('--only-dualaqi needs torch')
    opts = tuple(x.strip() for x in args.event_opts.split(',') if x.strip())
    bad = [o for o in opts if o not in EVENT_OPTS]
    if bad:
        raise SystemExit(f'unknown --event-opts {bad}; choose from {EVENT_OPTS}')
    frp = load_fire(args.fire_file or os.path.join(args.data_dir, 'punjab_fire_daily.csv'), 'frp_sum') \
        if 'frp' in opts else None
    s = build_samples(hourly, hours, fire=fire, event_opts=opts, frp=frp)
    origins = s['origins']
    print(f'origins: {len(origins)}  {origins[0]} -> {origins[-1]}  launch hours: {hours}')
    if fire is not None:
        print(f'fire counts: {fire.index[0].date()} -> {fire.index[-1].date()}, {FIRE_LAGS} lagged days per origin; '
              f'event options: {opts or "none (v1)"}; event inputs: {s["event"].shape[1]}; '
              f'early stopping: {"on" if args.early_stop else "off"}')
    print(f"XGBoost: {'on' if XGBRegressor is not None else 'xgboost not installed, skipped'}; "
          f"ARIMA (Fourier regression + ARMA errors, order by AIC from {ARIMA_ORDERS}): "
          f"{'off' if args.no_arima else 'on'}")

    blocked = blocked_folds(origins, args.k_blocks)
    fstart = pd.Timestamp(args.forward_start) if args.forward_start else origins[0] + pd.Timedelta(days=365)
    forward = forward_folds(origins, fstart)

    use_nn = (not args.no_nn) and torch is not None
    if not use_nn and not args.no_nn:
        print('torch not installed: running classical models only')

    schemes = [x.strip() for x in args.schemes.split(',')]
    tables, oofs = [], []
    for scheme, folds in [('blocked', blocked), ('forward', forward)]:
        if scheme not in schemes:
            continue
        fold_ids = list(range(len(folds)))
        if args.fold_range:
            a, b = (int(x) for x in args.fold_range.split(':'))
            fold_ids = fold_ids[a:b]
            folds = [folds[i] for i in fold_ids]
        print(f'== {scheme} CV: {len(folds)} folds (ids {fold_ids[0]}-{fold_ids[-1]})')
        if args.only_dualaqi:
            if args.no_fire and args.calendar_only:
                raise SystemExit('use either --no-fire or --calendar-only')
            event, name = s['event'], None
            if args.calendar_only:
                event, name = s['event'][:, -N_CAL:], 'DualAQI-calendarOnly'
            name = args.model_name or name
            label = name or ('DualAQI-noFire (single-branch ablation)' if args.no_fire else 'DualAQI')
            print(f'   model: {label}, seed {args.seed}')
            oof = run_dualaqi_cv(origins, s['Xb'], s['Y'], folds, s['seq'], event, args.seed,
                                 use_fire=not args.no_fire, name=name, early_stop=args.early_stop,
                                 fold_ids=fold_ids)
        else:
            oof = run_cv(origins, s['Xb'], s['X'], s['Y'], folds, s['seq'], use_nn, args.seed,
                         pm=None if args.no_arima else hourly['pm25'])
        oof['scheme'] = scheme
        oofs.append(oof)
        tables.append(summarise(oof, scheme, args.n_boot, args.seed))

    bench = pd.concat(tables, ignore_index=True)
    bench.to_csv(os.path.join(args.out_dir, 'benchmark_long.csv'), index=False)
    pd.concat(oofs, ignore_index=True).to_csv(os.path.join(args.out_dir, 'oof_predictions.csv.gz'),
                                              index=False, compression='gzip')
    show = ['scheme', 'horizon', 'model', 'RMSE', 'MAE', 'R2', 'CatAcc', 'skill_RMSE_vs_persist']
    print(bench[show].round(3).to_string(index=False))
    print(f'wrote {args.out_dir}/benchmark_long.csv and oof_predictions.csv.gz')


if __name__ == '__main__':
    main()
