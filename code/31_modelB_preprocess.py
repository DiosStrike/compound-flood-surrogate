#!/usr/bin/env python3
"""31_modelB_preprocess.py -- Option B data preprocessing (docs/PLAN.md v6 §4 steps 1-5).

Read-only on runs/ and data/<event>/scenarios/split_A.csv; writes only data/modelB_v1/. No normalization, no new splits, no sample windowing.
  Step 1 static.npz: zb_filled (inactive cells filled with the active-cell mean zb), msk, loss_mask (msk > 0)
  Step 2 <event>/<sid>.npz: h_proc (where(dry, 0, h), inactive cells 0), rain_proc (rain_on_model from load_run, not flipped again),
          wl_hourly (bzs[:,0] on-the-hour values), dis_hourly (on-the-hour values of the sum of the four dis columns); on-the-hour rule as in code/30 (nearest record, earlier one on ties)
  Step 3 self-check: max(h_proc[73:241]) == event7d/hmax_event7d_hourlysampled.npy cell by cell; on-the-hour value checks;
          rain_proc event-period cumulative mean / native ampr (ratio reported only); h_proc has no NaN and no negatives. Any self-check failure: stop (exit code 2)
  Step 5 index.csv: one row per run (existing rows overwritten by event+sid)

Usage:
  python3 code/31_modelB_preprocess.py --runs matthew_s041          # trial run
  python3 code/31_modelB_preprocess.py --all                        # batch of 486 (only after user confirmation)
"""
import os, sys, json, time, argparse, hashlib, importlib.util
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from read_run import load_run, run_dir

_spec = importlib.util.spec_from_file_location('step0', os.path.join(ROOT, 'code', '30_modelB_step0_checks.py'))
step0 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(step0)
hourly_values, NATIVE, HOURS = step0.hourly_values, step0.NATIVE, step0.HOURS

OUT = os.path.join(ROOT, 'data', 'modelB_v1')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
T_EVENT = slice(73, 241)          # target times t = 73...240 h
RAIN_EVENT_BLOCKS = slice(72, 240)  # event period [72 h, 240 h) corresponds to ampr blocks 72...239
INDEX_COLS = ['event', 'sid', 'split_A', 'alpha', 'beta', 'gamma', 'tau_h', 'path', 'md5', 'n_hour_subs', 'sub_dt',
              'neg_h_celltimes_zeroed', 'neg_h_cells', 'selfcheck', 'h_target_equals_hmax', 'wl_check', 'dis_check',
              'rain_event_ratio_active', 'rain_event_ratio_grid', 'h_nan', 'h_neg', 'rain_nan', 'rain_neg', 'bytes', 'seconds']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def make_static(r):
    msk = r['msk']; zb = r['zb']; act = msk > 0
    fill = float(np.nanmean(zb[act]))
    zb_filled = np.where(act, zb, fill).astype(np.float32)
    assert np.isfinite(zb_filled).all(), 'zb_filled contains non-finite values'
    assert np.isnan(zb[~act]).all() and np.isfinite(zb[act]).all(), 'zb NaN does not match the inactive cells'
    p = os.path.join(OUT, 'static.npz')
    np.savez_compressed(p, zb_filled=zb_filled, msk=msk.astype(np.int8), loss_mask=act)
    return dict(path=os.path.relpath(p, ROOT), md5=md5(p), bytes=os.path.getsize(p), zb_fill_value=fill,
                n_active=int(act.sum()), n_inactive=int((~act).sum()))


def process(ev, sid):
    t0 = time.time()
    r = load_run(ev, sid)
    msk, act = r['msk'], r['msk'] > 0
    h, dry = r['h'], r['dry']
    # Step 2
    neg = (h < 0) & act[None]
    h_proc = np.where(dry, np.float32(0), h).astype(np.float32)
    h_proc[:, ~act] = 0
    rain_proc = r['rain_on_model'].astype(np.float32)
    wl_hourly, sub_b = hourly_values(r['bzs_t_s'], r['bzs'][:, 0], NATIVE['bzs'])
    dis_hourly, sub_q = hourly_values(r['dis_t_s'], r['dis'].sum(axis=1), NATIVE['dis'])
    subs = [('bzs', s) for s in sub_b] + [('dis', s) for s in sub_q]
    if any(s['exceeds_native'] for _, s in subs):
        raise SystemExit('Stopped: %s on-the-hour substitution time difference exceeds the native step %s' % (sid, [s for _, s in subs if s['exceeds_native']]))
    # Step 3 self-check
    hmax = np.load(os.path.join(run_dir(ev, sid), 'event7d', 'hmax_event7d_hourlysampled.npy'))
    h_eq = bool(np.array_equal(h_proc[T_EVENT].max(axis=0), hmax))
    h_maxdiff = float(np.abs(h_proc[T_EVENT].max(axis=0) - hmax).max())
    bt, qt = r['bzs_t_s'], r['dis_t_s']
    if len(bt) == 2401 and np.array_equal(bt[::10], HOURS):
        wl_ok = bool(np.array_equal(wl_hourly, r['bzs'][::10, 0]))
    else:
        wl_ok = bool(all(wl_hourly[k] == r['bzs'][np.flatnonzero(bt == s['used_t_s'])[0], 0] for k, s in [(s['hour'], s) for s in sub_b]))
    if len(qt) == 961 and np.array_equal(qt[::4], HOURS):
        dis_ok = bool(np.array_equal(dis_hourly, r['dis'][::4].sum(axis=1)))
    else:   # Irma: hours with a record = that record; substituted hours = the record used
        rows = [np.flatnonzero(qt == (HOURS[k] if k not in {s['hour'] for s in sub_q} else next(s['used_t_s'] for s in sub_q if s['hour'] == k)))[0] for k in range(241)]
        dis_ok = bool(np.array_equal(dis_hourly, r['dis'][rows].sum(axis=1)))
    # Event-period rainfall total: model-grid nearest neighbour vs native 1 km ampr (native cells whose centres fall within the model grid bounding box)
    cum_model = rain_proc[RAIN_EVENT_BLOCKS].sum(axis=0)
    A = r['rain'][RAIN_EVENT_BLOCKS].sum(axis=0); xc, yc = r['rain_x'], r['rain_y']
    x0, x1 = r['x'].min() - 100, r['x'].max() + 100; y0, y1 = r['y'].min() - 100, r['y'].max() + 100
    inside = ((xc[None, :] >= x0) & (xc[None, :] <= x1)) & ((yc[:, None] >= y0) & (yc[:, None] <= y1))
    native_mean = float(np.nanmean(A[inside]))
    ratio_grid = float(cum_model.mean() / native_mean) if native_mean > 0 else float('nan')
    ratio_act = float(cum_model[act].mean() / native_mean) if native_mean > 0 else float('nan')
    checks = dict(h_nan=int(np.isnan(h_proc).sum()), h_neg=int((h_proc < 0).sum()),
                  rain_nan=int(np.isnan(rain_proc).sum()), rain_neg=int((rain_proc < 0).sum()))
    ok = h_eq and wl_ok and dis_ok and checks['h_nan'] == 0 and checks['h_neg'] == 0 and checks['rain_nan'] == 0 and checks['rain_neg'] == 0
    # Step 4 storage
    os.makedirs(os.path.join(OUT, ev), exist_ok=True)
    p = os.path.join(OUT, ev, sid + '.npz')
    np.savez_compressed(p, h_proc=h_proc, rain_proc=rain_proc, wl_hourly=wl_hourly.astype(np.float64), dis_hourly=dis_hourly.astype(np.float64))
    sc = r['scenario']
    split = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'split_A.csv')).set_index('scenario').loc[sid, 'split']
    row = dict(event=ev, sid=sid, split_A=split, alpha=sc['alpha'], beta=sc['beta'], gamma=sc['gamma'], tau_h=sc['tau_h'],
               path=os.path.relpath(p, ROOT), md5=md5(p), n_hour_subs=len(subs),
               sub_dt=';'.join('%s:%d/%+d' % (v, s['hour'], int(s['dt_s'])) for v, s in subs),
               neg_h_celltimes_zeroed=int(neg.sum()), neg_h_cells=int(neg.any(axis=0).sum()),
               selfcheck='PASS' if ok else 'FAIL', h_target_equals_hmax=h_eq, wl_check=wl_ok, dis_check=dis_ok,
               rain_event_ratio_active=ratio_act, rain_event_ratio_grid=ratio_grid, **checks,
               bytes=os.path.getsize(p), seconds=round(time.time() - t0, 2))
    extra = dict(h_maxdiff_vs_hmax=h_maxdiff, rain_native_event_mean_mm=native_mean, rain_model_event_mean_mm_active=float(cum_model[act].mean()),
                 raw_bytes=int(h_proc.nbytes + rain_proc.nbytes + wl_hourly.nbytes + dis_hourly.nbytes),
                 dry_neg_all_dry=bool(np.all(dry[neg])), n_native_cells_inside=int(inside.sum()))
    return row, extra, r


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--runs', help='comma-separated sids, e.g. matthew_s041')
    g.add_argument('--all', action='store_true')
    a = ap.parse_args()
    sids = [s.strip() for s in a.runs.split(',')] if a.runs else ['%s_s%03d' % (e, n) for e in EVENTS for n in range(1, 82)]
    os.makedirs(OUT, exist_ok=True)
    idx_path = os.path.join(OUT, 'index.csv')
    idx = pd.read_csv(idx_path) if os.path.exists(idx_path) else pd.DataFrame(columns=INDEX_COLS)
    log = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), runs={})
    t_all = time.time(); static = None
    for sid in sids:
        ev = sid.rsplit('_', 1)[0]
        row, extra, r = process(ev, sid)
        if static is None:
            static = make_static(r); log['static'] = static
        keep = idx[~((idx.event == ev) & (idx.sid == sid))]
        idx = (pd.concat([keep, pd.DataFrame([row])], ignore_index=True) if len(keep) else pd.DataFrame([row]))[INDEX_COLS]
        idx.to_csv(idx_path, index=False)
        log['runs'][sid] = dict(row, **extra)
        print(json.dumps(dict(row, **extra), ensure_ascii=False, default=str), flush=True)
        if row['selfcheck'] != 'PASS':
            json.dump(log, open(os.path.join(OUT, 'preprocess_log.json'), 'w'), indent=1, ensure_ascii=False, default=str)
            print('Stopped: %s failed self-check' % sid); sys.exit(2)
    log['elapsed_s'] = round(time.time() - t_all, 2)
    json.dump(log, open(os.path.join(OUT, 'preprocess_log.json'), 'w'), indent=1, ensure_ascii=False, default=str)
    print('static:', json.dumps(static, ensure_ascii=False)); print('done %.1fs' % log['elapsed_s'])


if __name__ == '__main__':
    main()
