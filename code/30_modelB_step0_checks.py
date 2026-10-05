#!/usr/bin/env python3
"""30_modelB_step0_checks.py -- data part of Scheme B step 0, the pre-training checks (docs/PLAN.md §4 step 0). Read-only; modifies no data.

Checks (all 486 runs):
  2  On-the-hour availability: do every 10th bzs row / every 4th dis row fall exactly on 0, 3600, ..., 864000 s; when an hour is missing, per the PLAN v6 rule
     substitute the record nearest to that hour, taking the earlier record when two are equidistant; list each substituted hour, the record time used and the time offset
     (|offset| > the series' native step is flagged exceeds_native; per PLAN this must stop and be reported)
  3  Hourly-sampling loss: linearly interpolate the hourly series back to the native times and compare with the original, max error / RMSE (recorded only)
  4a Rainfall row order: is load_run()'s rain_on_model element-wise equal to an independent "flip rows + nearest neighbour by coordinates" implementation (s041 of all six events)
  4b bzs 12 columns identical at every time
  Extra: map time axis = 0...864000 s; mean computational time step from sfincs.log
Item 1 (SFINCS time semantics) was checked in source code / manual; conclusions in result/reports/modelB_step0_checks.md, not covered by this script.

Usage: python3 code/30_modelB_step0_checks.py [--out result/manifests/modelB_step0_checks.json]
"""
import os, sys, json, time, re, argparse
import numpy as np, netCDF4 as nc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from read_run import run_dir, load_run

EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
HOURS = np.arange(241) * 3600.0
NATIVE = {'bzs': 360.0, 'dis': 900.0}


def hourly_values(ts, vals, native):
    """On-the-hour instantaneous values: use the on-the-hour record when present; otherwise substitute the nearest record, taking the earlier one when equidistant. Returns values and the substitution list."""
    out = np.empty(len(HOURS)); subs = []
    for k, T in enumerate(HOURS):
        dist = np.abs(ts - T); j = int(np.flatnonzero(dist == dist.min())[0])   # on ties take the smaller index = the earlier record
        out[k] = vals[j]
        if ts[j] != T:
            n_tie = int(np.sum(dist == dist.min()))
            subs.append(dict(hour=k, used_t_s=float(ts[j]), dt_s=float(ts[j] - T), tie=n_tie > 1,
                             exceeds_native=bool(abs(ts[j] - T) > native)))
    return out, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'result', 'manifests', 'modelB_step0_checks.json'))
    a = ap.parse_args()
    t0 = time.time()
    res = dict(hourly_rule='PLAN v6: use the on-the-hour record when present; when missing, substitute the record nearest to that hour, taking the earlier record when equidistant; never use records after the hour. '
                           'On 2026-09-16 this was briefly changed to linear interpolation between the two neighbours per PLAN v5, and reverted to nearest record by the user the same day; this file holds results under the v6 rule.',
               runs={}, events={})
    for ev in EVENTS:
        E = dict(n=0, bzs_rows=set(), dis_rows=set(), bzs_steps=set(), dis_steps=set(), bzs_idx_ok=0, dis_idx_ok=0,
                 n_subs_bzs=0, n_subs_dis=0, runs_with_subs=0, n_ties=0, max_absdt_bzs=0.0, max_absdt_dis=0.0, exceeds_native=0,
                 bzs_cols_identical=0, loss_bzs_max=0.0, loss_dis_max=0.0, loss_bzs_rmse_max=0.0, loss_dis_rmse_max=0.0,
                 loss_bzs_argmax=None, loss_dis_argmax=None, map_time_ok=0, log_avg_dt=[np.inf, -np.inf])
        for n in range(1, 82):
            sid = '%s_s%03d' % (ev, n); d = run_dir(ev, sid)
            bzs = np.loadtxt(os.path.join(d, 'sfincs.bzs')); bt, bv = bzs[:, 0], bzs[:, 1:]   # same reading method as load_run()
            dis = np.loadtxt(os.path.join(d, 'sfincs.dis')); qt, qv = dis[:, 0], dis[:, 1:]
            wl, qsum = bv[:, 0], qv.sum(axis=1)
            r = dict(bzs_shape=list(bv.shape), dis_shape=list(qv.shape))
            E['n'] += 1; E['bzs_rows'].add(len(bt)); E['dis_rows'].add(len(qt))
            E['bzs_steps'].update(np.unique(np.diff(bt)).tolist()); E['dis_steps'].update(np.unique(np.diff(qt)).tolist())
            r['bzs_idx_ok'] = bool(len(bt) == 2401 and np.array_equal(bt[::10], HOURS))
            r['dis_idx_ok'] = bool(len(qt) == 961 and np.array_equal(qt[::4], HOURS))
            E['bzs_idx_ok'] += r['bzs_idx_ok']; E['dis_idx_ok'] += r['dis_idx_ok']
            wl_h, ib = hourly_values(bt, wl, NATIVE['bzs']); q_h, iq = hourly_values(qt, qsum, NATIVE['dis'])
            r['bzs_subs'] = ib; r['dis_subs'] = iq
            E['n_subs_bzs'] += len(ib); E['n_subs_dis'] += len(iq); E['runs_with_subs'] += bool(ib or iq)
            for key, lst in (('bzs', ib), ('dis', iq)):
                for x in lst:
                    E['n_ties'] += x['tie']; E['exceeds_native'] += x['exceeds_native']
                    E['max_absdt_' + key] = max(E['max_absdt_' + key], abs(x['dt_s']))
            eb = np.interp(bt, HOURS, wl_h) - wl; eq = np.interp(qt, HOURS, q_h) - qsum
            r.update(loss_bzs_maxabs=float(np.abs(eb).max()), loss_bzs_rmse=float(np.sqrt(np.mean(eb ** 2))),
                     loss_dis_maxabs=float(np.abs(eq).max()), loss_dis_rmse=float(np.sqrt(np.mean(eq ** 2))))
            if r['loss_bzs_maxabs'] > E['loss_bzs_max']:
                E['loss_bzs_max'] = r['loss_bzs_maxabs']; E['loss_bzs_argmax'] = [sid, float(bt[np.argmax(np.abs(eb))])]
            if r['loss_dis_maxabs'] > E['loss_dis_max']:
                E['loss_dis_max'] = r['loss_dis_maxabs']; E['loss_dis_argmax'] = [sid, float(qt[np.argmax(np.abs(eq))])]
            E['loss_bzs_rmse_max'] = max(E['loss_bzs_rmse_max'], r['loss_bzs_rmse']); E['loss_dis_rmse_max'] = max(E['loss_dis_rmse_max'], r['loss_dis_rmse'])
            rng = float(np.max(bv.max(axis=1) - bv.min(axis=1))); r['bzs_col_maxrange'] = rng
            E['bzs_cols_identical'] += bool(bv.shape[1] == 12 and rng == 0.0)
            with nc.Dataset(os.path.join(d, 'sfincs_map.nc')) as m:
                r['map_time_ok'] = bool(np.array_equal(np.ma.filled(m.variables['time'][:], np.nan).astype(float), HOURS))
            E['map_time_ok'] += r['map_time_ok']
            mm = re.search(r'Average time step \(s\)\s*:\s*([\d.]+)', open(os.path.join(d, 'sfincs.log')).read())
            if mm:
                v = float(mm.group(1)); r['log_avg_dt_s'] = v; E['log_avg_dt'] = [min(E['log_avg_dt'][0], v), max(E['log_avg_dt'][1], v)]
            res['runs'][sid] = r
        for k in ('bzs_rows', 'dis_rows', 'bzs_steps', 'dis_steps'):
            E[k] = sorted(E[k])
        res['events'][ev] = E
        print(ev, {k: v for k, v in E.items()}, '%.0fs' % (time.time() - t0), flush=True)

    rain = {}
    for ev in EVENTS:
        r = load_run(ev, 41)
        A, yc, xc, hdr, y, x = r['rain'], r['rain_y'], r['rain_x'], r['rain_header'], r['y'], r['x']
        yc_f = yc[::-1]
        iy = np.clip(np.round((y - yc_f[0]) / float(hdr['dy'])).astype(int), 0, len(yc) - 1)
        ix = np.clip(np.round((x - xc[0]) / float(hdr['dx'])).astype(int), 0, len(xc) - 1)
        indep = A[:, ::-1, :][:, iy, ix]
        wrong = A[:, iy, ix]                                    # control: no flip but indexed with south-first row numbers
        rain[ev] = dict(rain_y_decreasing=bool(np.all(np.diff(yc) < 0)), model_row0_is_south=bool(y[0, 0] < y[-1, 0]),
                        rain_on_model_equals_flip_then_nn=bool(np.array_equal(r['rain_on_model'], indep, equal_nan=True)),
                        rain_on_model_equals_noflip_southindex=bool(np.array_equal(r['rain_on_model'], wrong, equal_nan=True)),
                        nn_max_dy_m=float(np.abs(yc_f[iy] - y).max()), nn_max_dx_m=float(np.abs(xc[ix] - x).max()),
                        ampr_nt=int(len(r['rain_t_h'])), ampr_times_0_240h=bool(np.array_equal(r['rain_t_h'], np.arange(241.0))))
        print(ev, rain[ev], flush=True)
    res['rain_roworder'] = rain
    res['elapsed_s'] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('done %.0fs -> %s' % (res['elapsed_s'], a.out))


if __name__ == '__main__':
    main()
