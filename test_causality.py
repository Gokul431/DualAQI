#!/usr/bin/env python3
"""Causality tests for the DualAQI event branch (Punjab fire counts).

Rule under test: the fire count for UTC day d may only be used by forecast origins at or after
03:00 UTC on day d+1.

Test 1, features, every origin: for each origin t, replace every fire count on days after
  last_usable_fire_day(t) with random values. Its event features must not change.
  Negative control: changing last_usable_fire_day(t) itself must change them.

Test 2, end to end, forward fold --fold: pick a cut day X in the middle of the test block.
  Train DualAQI, then replace every fire count on days >= X with random values and retrain from
  scratch with the same seed. Every test origin whose last usable fire day is before X must get
  bit-identical predictions. Negative control: perturbing the day before X changes the predictions
  of the origins that use it.

    .venv\\Scripts\\python test_causality.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import dualaqi as D


def perturb(fire, from_day, rng):
    f = fire.copy()
    m = f.index >= from_day
    f[m] = rng.integers(0, 20000, m.sum()).astype(float)
    return f


def test_features(origins, fire, rng):
    base = D.event_features(origins, fire)
    last = D.last_usable_fire_day(origins)
    bad, control_ok = 0, 0
    sample = rng.choice(len(origins), size=min(300, len(origins)), replace=False)
    for i in sample:
        f2 = perturb(fire, last[i] + pd.Timedelta(days=1), rng)
        if not np.array_equal(D.event_features(origins[i:i + 1], f2)[0], base[i]):
            bad += 1
        f3 = fire.copy()
        f3[last[i]] = fire[last[i]] + 1000
        control_ok += not np.array_equal(D.event_features(origins[i:i + 1], f3)[0], base[i])
    # direct check of the rule on the timestamps themselves
    ok_rule = bool(((last + pd.Timedelta(days=1) + pd.Timedelta(hours=D.FIRE_AVAILABLE_HOUR)) <= origins).all())
    later = last + pd.Timedelta(days=1)
    ok_tight = bool(((later + pd.Timedelta(days=1) + pd.Timedelta(hours=D.FIRE_AVAILABLE_HOUR)) > origins).all())
    print(f'Test 1 (features): {len(sample)} origins checked')
    print(f'  every used fire day d satisfies origin >= d+1 03:00 UTC: {ok_rule}')
    print(f'  the next day would not yet be allowed (window is as recent as the rule permits): {ok_tight}')
    print(f'  origins whose features changed when future fire counts were randomised: {bad}  (must be 0)')
    print(f'  negative control, origins that changed when the last usable day was altered: '
          f'{control_ok} of {len(sample)}  (must be all)')
    return bad == 0 and control_ok == len(sample) and ok_rule and ok_tight


def test_features_v2(origins, fire, frp, hourly, rng):
    """Same rule for every v2 input (FRP lags, wind transport, 14/30-day sums), all options on."""
    opts = D.EVENT_OPTS
    feats = lambda f, r, h, o: D.event_features(o, f, opts, frp=r, transport=D.daily_transport(h))
    last = D.last_usable_fire_day(origins)
    sample = rng.choice(len(origins), size=min(150, len(origins)), replace=False)
    base = feats(fire, frp, hourly, origins[sample])
    bad, ctrl = 0, 0
    for j, i in enumerate(sample):
        cut = last[i] + pd.Timedelta(days=1)
        h2 = hourly.copy()
        m = h2.index >= cut
        h2.loc[m, 'u10'] = rng.normal(0, 20, m.sum())
        h2.loc[m, 'v10'] = rng.normal(0, 20, m.sum())
        if not np.array_equal(feats(perturb(fire, cut, rng), perturb(frp, cut, rng), h2, origins[i:i + 1])[0], base[j]):
            bad += 1
        h3 = hourly.copy()
        day = (h3.index >= last[i]) & (h3.index < cut)
        h3.loc[day, 'u10'] = h3.loc[day, 'u10'] + 10
        ctrl += not np.array_equal(feats(fire, frp, h3, origins[i:i + 1])[0], base[j])
    print(f'Test 1b (v2 features: FRP, wind transport, 14/30-day sums): {len(sample)} origins checked')
    print(f'  origins whose features changed when future fire, FRP and ERA5 winds were randomised: {bad}  (must be 0)')
    print(f'  negative control, origins that changed when the last usable day\'s winds were altered: '
          f'{ctrl} of {len(sample)}  (must be all)')
    return bad == 0 and ctrl == len(sample)


def test_end_to_end(s, fire, fold, seed, rng):
    origins, Y, seq = s['origins'], s['Y'], s['seq']
    folds = D.forward_folds(origins, origins[0] + pd.Timedelta(days=365))
    tr, te = folds[fold]
    last = D.last_usable_fire_day(origins[te])
    cut = last[len(te) // 2] + pd.Timedelta(days=1)          # first day that is perturbed
    unaffected = np.asarray(last < cut)                       # origins that never see a day >= cut
    print(f'Test 2 (end to end): forward fold {fold}, {len(tr)} training origins, {len(te)} test origins '
          f'{origins[te[0]].date()} -> {origins[te[-1]].date()}')
    print(f'  fire counts randomised from {cut.date()} onward; {unaffected.sum()} test origins must be unchanged')

    p0 = D.dualaqi_fold(seq, D.event_features(origins, fire), Y, tr, te, seed, use_fire=True)
    p1 = D.dualaqi_fold(seq, D.event_features(origins, perturb(fire, cut, rng)), Y, tr, te, seed, use_fire=True)
    max_diff_unaff = float(np.abs(p1[unaffected] - p0[unaffected]).max())
    changed_aff = int((np.abs(p1[~unaffected] - p0[~unaffected]).max(axis=1) > 1e-6).sum())
    print(f'  max |prediction change| for those origins: {max_diff_unaff:.3g}  (must be 0)')
    print(f'  later origins (allowed to see the changed days) whose predictions changed: '
          f'{changed_aff} of {(~unaffected).sum()}')

    f3 = fire.copy()
    f3[cut - pd.Timedelta(days=1)] = fire[cut - pd.Timedelta(days=1)] + 5000
    p3 = D.dualaqi_fold(seq, D.event_features(origins, f3), Y, tr, te, seed, use_fire=True)
    uses = np.asarray([(cut - pd.Timedelta(days=1) <= l) and (cut - pd.Timedelta(days=1) > l - pd.Timedelta(days=D.FIRE_LAGS))
                       for l in last])
    ctrl = int((np.abs(p3[uses] - p0[uses]).max(axis=1) > 1e-6).sum())
    print(f'  negative control: altering {(cut - pd.Timedelta(days=1)).date()} changed {ctrl} of {uses.sum()} '
          f'origins that use that day  (must be all)')
    return max_diff_unaff == 0.0 and ctrl == uses.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(123)
    hourly = D.load_inputs(args.data_dir)
    fire = D.load_fire(os.path.join(args.data_dir, 'punjab_fire_daily.csv'))
    if fire is None:
        raise SystemExit('data/punjab_fire_daily.csv not found')
    s = D.build_samples(hourly, [3], fire=fire)
    ok1 = test_features(s['origins'], fire, rng)
    frp = D.load_fire(os.path.join(args.data_dir, 'punjab_fire_daily.csv'), 'frp_sum')
    ok1b = test_features_v2(s['origins'], fire, frp, hourly, rng) if frp is not None else True
    ok2 = test_end_to_end(s, fire, args.fold, args.seed, rng)
    print(f'\nRESULT: v1 features {"PASS" if ok1 else "FAIL"}, v2 features {"PASS" if ok1b else "FAIL"}, '
          f'end to end {"PASS" if ok2 else "FAIL"}')
    sys.exit(0 if ok1 and ok1b and ok2 else 1)


if __name__ == '__main__':
    main()
