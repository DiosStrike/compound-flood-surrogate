#!/usr/bin/env python3
"""read_run.py -- load one SFINCS simulation (6 events x 81) with hourly water depth and the matching forcing. Read-only; modifies no files.

Usage:
    import sys; sys.path.insert(0, 'code'); from read_run import load_run
    r = load_run('beryl', 3)          # or load_run('beryl', 'beryl_s003')
    r['h']        (241, 195, 255) float32  hourly water depth m (SFINCS native h variable, defined at all active cells; dry-cell h <= 0.05 m kept as is); inactive cells NaN; t = 0,1,...,240 h (since tstart)
    r['zs']       (241, 195, 255)          water level m NAVD88; dry cells NaN (SFINCS does not write zs at dry cells), inactive cells NaN
    r['dry']      (241, 195, 255) bool     cell is dry at that hour (h <= huthresh 0.05 m, SFINCS writes no zs); at wet cells zs = zb + h holds exactly
    r['zb'], r['msk']  (195, 255)          bed elevation m NAVD88; mask 0 inactive / 1 active / 2 water-level boundary
    r['t_h']      (241,)                   hours; r['time_utc'] corresponding UTC times; r['is_event'] (241,) bool: t > 72 is the main 7-day period, t <= 72 is spin-up
    r['rain']     (241, 42, 53) float32    ampr hourly rainfall mm/h, 1 km grid, row 0 = north; r['rain_t_h'] hours (TIME unit is hours since tref); r['rain_x'], r['rain_y'] cell-center coordinates (EPSG:26917)
    r['rain_on_model']  (241, 195, 255)    rainfall nearest-neighbour resampled to the 200 m model grid (not identical to SFINCS's internal bilinear interpolation; for aligned samples only)
    r['bzs']      (2401, 12) boundary water level m NAVD88 (6 min, same series at all 12 bnd points); r['bzs_t_s'] seconds
    r['dis']      (961, 4)   discharge at the four source cells m^3/s (15 min, signed); r['dis_t_s'] seconds
    r['scenario'] manifest row (alpha/beta/gamma/tau_h/split/...); r['run_dir']
"""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = {'irma': 'rerun_dtmax3600'}

def run_dir(ev, sid):
    return os.path.join(ROOT, 'runs', ev, SUB[ev], sid) if ev in SUB else os.path.join(ROOT, 'runs', ev, sid)

def read_ampr(path):
    """Read a SFINCS ampr file (meteo_on_equidistant_grid). Returns (times_s, data[t, row, col], x_centers, y_centers, header). Row 0 = north."""
    hdr, blocks, cur = {}, [], None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith('TIME'):
                cur = []; blocks.append((float(s.split('=')[1].split()[0]), cur)); continue
            if cur is None:
                if '=' in s: k, v = s.split('=', 1); hdr[k.strip()] = v.strip()
            else:
                cur.extend(float(x) for x in s.split())
    nr, nc = int(hdr['n_rows']), int(hdr['n_cols']); dx, dy = float(hdr['dx']), float(hdr['dy']); x0, y0 = float(hdr['x_llcorner']), float(hdr['y_llcorner'])
    t = np.array([b[0] for b in blocks]); a = np.array([np.array(b[1], np.float32).reshape(nr, nc) for b in blocks])
    a[a == float(hdr.get('NODATA_value', -999))] = np.nan
    xc = x0 + dx * (np.arange(nc) + 0.5); yc = y0 + dy * (nr - 0.5 - np.arange(nr))     # row 0 = north
    return t, a, xc, yc, hdr

def load_run(ev, sid, with_rain=True):
    import netCDF4 as nc
    if isinstance(sid, int): sid = '%s_s%03d' % (ev, sid)
    d = run_dir(ev, sid); out = dict(event=ev, scenario_id=sid, run_dir=d)
    man = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv')); out['scenario'] = man[man.scenario == sid].iloc[0].to_dict()
    m = nc.Dataset(os.path.join(d, 'sfincs_map.nc'))
    zs = np.ma.filled(m.variables['zs'][:], np.nan).astype(np.float32); zb = np.ma.filled(m.variables['zb'][:], np.nan).astype(np.float32)
    msk = np.ma.filled(m.variables['msk'][:], 0).astype(np.int8); tsec = np.ma.filled(m.variables['time'][:], np.nan).astype(float)
    out.update(zs=zs, zb=zb, msk=msk, x=np.ma.filled(m.variables['x'][:], np.nan), y=np.ma.filled(m.variables['y'][:], np.nan), t_s=tsec, t_h=tsec / 3600.0)
    dry = np.isnan(zs); dry[:, msk == 0] = False; out['dry'] = dry
    if 'h' in m.variables:
        h = np.ma.filled(m.variables['h'][:], np.nan).astype(np.float32)
    else:
        h = zs - zb[None]
    h[:, msk == 0] = np.nan; out['h'] = h
    out['is_event'] = out['t_h'] > 72; out['is_spinup'] = ~out['is_event']
    tref = pd.Timestamp(out['scenario']['tstart']); out['time_utc'] = tref + pd.to_timedelta(tsec, unit='s')
    m.close()
    hs = nc.Dataset(os.path.join(d, 'sfincs_his.nc'))
    out['his'] = {v: np.ma.filled(hs.variables[v][:], np.nan) for v in hs.variables if v not in ('station_name', 'station_id')}
    try: out['his_station_names'] = [b''.join(r).decode().strip() if hasattr(r[0], 'decode') else ''.join(map(str, r)).strip() for r in hs.variables['station_name'][:]]
    except Exception: pass
    hs.close()
    bzs = np.loadtxt(os.path.join(d, 'sfincs.bzs')); out['bzs_t_s'], out['bzs'] = bzs[:, 0], bzs[:, 1:]
    dis = np.loadtxt(os.path.join(d, 'sfincs.dis')); out['dis_t_s'], out['dis'] = dis[:, 0], dis[:, 1:]
    if with_rain:
        t, a, xc, yc, hdr = read_ampr(os.path.join(d, 'sfincs.ampr')); out.update(rain_t_h=t, rain=a, rain_x=xc, rain_y=yc, rain_header=hdr)
        ix = np.clip(np.round((out['x'] - xc[0]) / float(hdr['dx'])).astype(int), 0, len(xc) - 1); iy = np.clip(np.round((yc[0] - out['y']) / float(hdr['dy'])).astype(int), 0, len(yc) - 1)
        out['rain_on_model'] = a[:, iy, ix]
    return out

if __name__ == '__main__':
    import sys
    r = load_run(sys.argv[1] if len(sys.argv) > 1 else 'beryl', int(sys.argv[2]) if len(sys.argv) > 2 else 41)
    print(r['scenario_id'], 'h', r['h'].shape, 'event snapshots', int(r['is_event'].sum()), 'rain', r['rain'].shape, 'bzs', r['bzs'].shape, 'dis', r['dis'].shape)
