#!/usr/bin/env python3
"""27_data_guide.py -- data guide for the 486 simulations (storage location / output variables / meaning, units, interval, size / how to read / spin-up vs main period / missing-data check).
Scans the map/his/forcing files of all 486 runs, read-only; writes result/manifests/data_completeness_486.csv, data/static/meta/data_guide_486.json, result/reports/DATA_GUIDE_486runs.pdf.
Run on the user's Mac (needs netCDF4): python3 code/27_data_guide.py [--quick] (--quick scans only 3 runs per event)"""
import os, sys, json, time, datetime
import numpy as np, pandas as pd
import netCDF4 as nc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_run import ROOT, run_dir, read_ampr
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
EVS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
QUICK = '--quick' in sys.argv
t0 = time.time()

# ---- 1. Variable list (one each from beryl_s003 and irma_s041) ----
def varlist(path):
    d = nc.Dataset(path); rows = []
    for v in d.variables:
        x = d.variables[v]; rows.append(dict(name=v, dims='×'.join('%s(%d)' % (k, len(d.dimensions[k])) for k in x.dimensions), dtype=str(x.dtype),
                                          units=getattr(x, 'units', ''), long_name=getattr(x, 'long_name', getattr(x, 'standard_name', ''))))
    g = {a: str(d.getncattr(a))[:80] for a in d.ncattrs()}; d.close(); return rows, g
map_vars, map_attrs = varlist(os.path.join(run_dir('beryl', 'beryl_s003'), 'sfincs_map.nc'))
his_vars, _ = varlist(os.path.join(run_dir('beryl', 'beryl_s003'), 'sfincs_his.nc'))
map_vars_3600, _ = varlist(os.path.join(run_dir('irma', 'irma_s041'), 'sfincs_map.nc'))
extra_3600 = [r['name'] for r in map_vars_3600 if r['name'] not in {q['name'] for q in map_vars}]

# ---- 2. Per-run completeness scan ----
rows = []
for ev in EVS:
    man = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv'))
    ids = [3, 41, 81] if QUICK else range(1, 82)
    for i in ids:
        sid = '%s_s%03d' % (ev, i); d = run_dir(ev, sid); r = dict(event=ev, scenario=sid)
        m = nc.Dataset(os.path.join(d, 'sfincs_map.nc'))
        msk = np.ma.filled(m.variables['msk'][:], 0); act = msk > 0; zb = np.ma.filled(m.variables['zb'][:], np.nan); t = np.ma.filled(m.variables['time'][:], np.nan)
        zsv = m.variables['zs']; nan_act = 0; nan_act_max = 0; nan_inactive_nonnan = 0
        for k in range(zsv.shape[0]):
            z = np.ma.filled(zsv[k], np.nan); n = int(np.isnan(z[act]).sum()); nan_act += n; nan_act_max = max(nan_act_max, n)
        z0 = np.ma.filled(zsv[0], np.nan)
        r.update(map_n_time=int(len(t)), map_t0_h=float(t[0] / 3600), map_t_last_h=float(t[-1] / 3600), map_dt_all_3600=bool(np.allclose(np.diff(t), 3600)),
                 n_active=int(act.sum()), n_msk2=int((msk == 2).sum()), zb_nan_active=int(np.isnan(zb[act]).sum()), zs_nan_active_total=nan_act, zs_nan_active_max_per_snapshot=nan_act_max,
                 zs_finite_inactive=int(np.isfinite(z0[~act]).sum()), has_cumprcp='cumprcp' in m.variables, has_zsmax='zsmax' in m.variables,
                 cumprcp_all_nan=bool(np.isnan(np.ma.filled(m.variables['cumprcp'][:], np.nan)).all()) if 'cumprcp' in m.variables else None)
        m.close()
        h = nc.Dataset(os.path.join(d, 'sfincs_his.nc')); ht = np.ma.filled(h.variables['time'][:], np.nan)
        pz = np.ma.filled(h.variables['point_zs'][:], np.nan) if 'point_zs' in h.variables else np.array([np.nan])
        r.update(his_n_time=int(len(ht)), his_t_last_h=float(ht[-1] / 3600), his_dt_s=float(np.median(np.diff(ht))), his_n_station=int(pz.shape[1]) if pz.ndim == 2 else 0, his_zs_nan=int(np.isnan(pz).sum())); h.close()
        bzs = np.loadtxt(os.path.join(d, 'sfincs.bzs')); dis = np.loadtxt(os.path.join(d, 'sfincs.dis'))
        r.update(bzs_n=len(bzs), bzs_dt_s=float(np.median(np.diff(bzs[:, 0]))), bzs_t_last_h=float(bzs[-1, 0] / 3600), bzs_nan=int(np.isnan(bzs).sum()), bzs_ncol=bzs.shape[1] - 1,
                 dis_n=len(dis), dis_dt_s=float(np.median(np.diff(dis[:, 0]))), dis_t_last_h=float(dis[-1, 0] / 3600), dis_nan=int(np.isnan(dis).sum()), dis_ncol=dis.shape[1] - 1)
        rt, ra, _, _, hdr = read_ampr(os.path.join(d, 'sfincs.ampr'))
        r.update(ampr_n=len(rt), ampr_dt_s=float(np.median(np.diff(rt))), ampr_t_last_h=float(rt[-1] / 3600), ampr_nan=int(np.isnan(ra).sum()), ampr_neg=int((ra < 0).sum()), ampr_shape='%d×%d' % ra.shape[1:])
        rq = json.load(open(os.path.join(d, 'rain_qc.json'))); r.update(rain_interp_hours=len(json.load(open(os.path.join(ROOT, 'data', ev, 'scenarios', sid, 'scenario.json')))['sources'].get('rain_interp_hours_used', [])),
                                                                    active_outside_ampr=rq['coverage']['active_outside_ampr'])
        rows.append(r); print('%-14s map %d/%.0fh nanZS %d zbNaN %d his %d bzs %d dis %d ampr %d  %.0fs' % (sid, r['map_n_time'], r['map_t_last_h'], nan_act, r['zb_nan_active'], r['his_n_time'], r['bzs_n'], r['dis_n'], r['ampr_n'], time.time() - t0), flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, 'result', 'manifests'), exist_ok=True)
df.to_csv(os.path.join(ROOT, 'result', 'manifests', 'data_completeness_486%s.csv' % ('_quick' if QUICK else '')), index=False)

# ---- 3. Raw-data gaps (interpolated hours inside the 10-day window), classified as spin-up / main period ----
gaps = []
for ev in EVS:
    man = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv')); w0 = pd.Timestamp(man.tstart.iloc[0]); w1 = pd.Timestamp(man.tstop.iloc[0]); sp = pd.Timestamp(man.spinup_until.iloc[0])
    gaps.append(dict(event=ev, window='%s → %s' % (w0.strftime('%Y-%m-%d %H:%M'), w1.strftime('%Y-%m-%d %H:%M')), spinup_until=sp.strftime('%Y-%m-%d %H:%M')))
    qf = os.path.join(ROOT, 'data', ev, 'processed', 'window10d', 'window_quality_%s.json' % ev)
    if os.path.exists(qf):
        q = json.load(open(qf)); gaps[-1]['window_quality'] = q
    gf = os.path.join(ROOT, 'data', ev, 'meta', 'gaps_%s.csv' % ev)
    if os.path.exists(gf):
        g = pd.read_csv(gf); gaps[-1]['gaps_csv_columns'] = list(g.columns); gaps[-1]['gaps_rows'] = g.to_dict('records')

# ---- 4. Summary ----
S = dict(generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), n_runs=len(df), quick=QUICK,
         all_map_241=bool((df.map_n_time == 241).all()), all_t_last_240=bool((df.map_t_last_h == 240).all()), all_dt_3600=bool(df.map_dt_all_3600.all()),
         zs_nan_active_total=int(df.zs_nan_active_total.sum()), zb_nan_active=df.zb_nan_active.unique().tolist(), n_active=df.n_active.unique().tolist(), n_msk2=df.n_msk2.unique().tolist(),
         his=dict(n_time=df.his_n_time.unique().tolist(), dt_s=df.his_dt_s.unique().tolist(), n_station=df.his_n_station.unique().tolist(), zs_nan_total=int(df.his_zs_nan.sum())),
         bzs=dict(n=df.bzs_n.unique().tolist(), dt_s=df.bzs_dt_s.unique().tolist(), ncol=df.bzs_ncol.unique().tolist(), nan=int(df.bzs_nan.sum())),
         dis=dict(n=df.dis_n.unique().tolist(), dt_s=df.dis_dt_s.unique().tolist(), ncol=df.dis_ncol.unique().tolist(), nan=int(df.dis_nan.sum())),
         ampr=dict(n=df.ampr_n.unique().tolist(), dt_s=df.ampr_dt_s.unique().tolist(), shape=df.ampr_shape.unique().tolist(), nan=int(df.ampr_nan.sum()), neg=int(df.ampr_neg.sum()), active_outside=int(df.active_outside_ampr.sum())),
         cumprcp=dict(has=int(df.has_cumprcp.sum()), all_nan=int(df.cumprcp_all_nan.fillna(False).sum())), zsmax=int(df.has_zsmax.sum()),
         rain_interp_hours_by_event={ev: int(df[df.event == ev].rain_interp_hours.max()) for ev in EVS},
         map_vars=map_vars, his_vars=his_vars, map_attrs=map_attrs, extra_vars_dtmaxout3600=extra_3600, windows=gaps)
json.dump(S, open(os.path.join(ROOT, 'data', 'static', 'meta', 'data_guide_486.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: S[k] for k in ('n_runs', 'all_map_241', 'all_t_last_240', 'all_dt_3600', 'zs_nan_active_total', 'zb_nan_active', 'n_active', 'n_msk2', 'his', 'bzs', 'dis', 'ampr', 'cumprcp', 'zsmax', 'rain_interp_hours_by_event', 'extra_vars_dtmaxout3600')}, ensure_ascii=False, indent=1))
print('map vars:', [(r['name'], r['dims'], r['units']) for r in map_vars]); print('his vars:', [(r['name'], r['dims'], r['units']) for r in his_vars])
for g in gaps: print(g['event'], g['window'], 'spinup_until', g['spinup_until'], '| window_quality keys:', list(g.get('window_quality', {}).keys())[:12])
print('elapsed %.1f min' % ((time.time() - t0) / 60))
