#!/usr/bin/env python3
"""
18_validate_obs.py -- validate one SFINCS run against observed water levels at in-domain gauges (model vs reality).
Usage: python3 code/18_validate_obs.py <run_name> [event]   (default baseline irma)

What it does:
  1. Read tstart/tstop from runs/<run>/sfincs.inp, the station table from sfincs.obs, and modelled water levels from sfincs_his.nc.
  2. Download observations for the same window from NOAA CO-OPS (water_level, datum=NAVD, GMT, m) and USGS NWIS IV (63160 NAVD88 water level; if absent, fall back to 00065 local-datum gage height).
     Raw JSON goes to data/raw/obs_validation/<run>/, cleaned UTC/m series to data/processed/obs_validation/<run>/.
     If a station was already downloaded for the same window by an earlier run, the raw JSON is reused (no repeat download).
  3. Per-station comparison: stations on the same datum (NAVD88) get bias / RMSE / correlation / peak difference / peak timing difference; local-datum-only stations get correlation and amplitude only, no absolute water-level comparison.
  4. Figure result/figures/fig_<run>_06_obs_comparison.png, stats table runs/<run>/obs_validation_stats.csv, conclusions printed to the terminal.
No fabricated data: if a download fails or a station lacks the parameter, leave it empty and flag it; observations are not filled or interpolated (observations are only nearest-neighbour matched to model output times for differencing).
"""
import os, sys, json, re, urllib.request, urllib.parse
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
ROOT = PTH.ROOT
EVENT = sys.argv[2] if len(sys.argv) > 2 else 'irma'
EV = PTH.event(EVENT)
RUN = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
RDIR = os.path.join(EV['runs'], RUN)
RAW = os.path.join(EV['obs_raw'], RUN); PROC = os.path.join(EV['obs_proc'], RUN)
FIG = os.path.join(ROOT, 'result/figures'); os.makedirs(RAW, exist_ok=True); os.makedirs(PROC, exist_ok=True)
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .5, 'font.size': 8.5})
C_OBS, C_MOD, C_RES = '#111111', '#d1495b', '#2f6fb0'
FT = 0.3048

# ---------- 1. Run info ----------
inp = dict(re.findall(r'^\s*(\w+)\s*=\s*(.+?)\s*$', open(os.path.join(RDIR, 'sfincs.inp')).read(), re.M))
T0 = pd.Timestamp(inp['tstart'], tz='UTC'); T1 = pd.Timestamp(inp['tstop'], tz='UTC'); TREF = pd.Timestamp(inp['tref'], tz='UTC')
obs_tbl = [(float(a), float(b), c.strip("'")) for a, b, c in (l.split() for l in open(os.path.join(RDIR, 'sfincs.obs')) if l.strip())]
his = xr.open_dataset(os.path.join(RDIR, 'sfincs_his.nc'))
hnames = [s.values.tobytes().decode().strip() for s in his['station_name']]
t_his = pd.DatetimeIndex(his['time'].values).tz_localize('UTC')
print('run %s  window %s -> %s  %d obs stations, %d in his' % (RUN, T0, T1, len(obs_tbl), len(hnames)))

def fetch(url, fn):
    p = os.path.join(RAW, fn)
    if os.path.exists(p):
        return open(p).read(), 'cached'
    req = urllib.request.Request(url, headers={'User-Agent': 'Flood_2.0 validation (research)'})
    txt = urllib.request.urlopen(req, timeout=120).read().decode()
    open(p, 'w').write(txt); return txt, 'downloaded'

# ---------- 2. Download observations ----------
series = {}   # sid -> dict(t=DatetimeIndex UTC, v=array m, datum=str, param=str, src=str, n=int)
for x, y, sid in obs_tbl:
    try:
        if re.fullmatch(r'\d{7}', sid):   # NOAA
            url = ('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=Flood_2.0'
                   '&begin_date=%s&end_date=%s&datum=NAVD&station=%s&time_zone=gmt&units=metric&format=json'
                   % (T0.strftime('%Y%m%d'), T1.strftime('%Y%m%d'), sid))
            txt, how = fetch(url, 'noaa_%s_water_level.json' % sid); d = json.loads(txt)
            if 'data' not in d: series[sid] = dict(err=str(d.get('error', d))[:80]); continue
            df = pd.DataFrame(d['data']); t = pd.DatetimeIndex(pd.to_datetime(df['t'], utc=True)); v = pd.to_numeric(df['v'], errors='coerce').values
            series[sid] = dict(t=t, v=v, datum='NAVD88', param='NOAA water_level (6 min, verified)', src=how, n=int(np.isfinite(v).sum()))
        else:                              # USGS
            url = ('https://waterservices.usgs.gov/nwis/iv/?format=json&sites=%s&startDT=%s&endDT=%s&parameterCd=63160,00065&siteStatus=all'
                   % (sid, T0.strftime('%Y-%m-%dT%H:%MZ'), T1.strftime('%Y-%m-%dT%H:%MZ')))
            txt, how = fetch(url, 'usgs_%s_iv.json' % sid); d = json.loads(txt)
            got = {}
            for ts in d['value']['timeSeries']:
                pc = ts['variable']['variableCode'][0]['value']; vals = ts['values'][0]['value']
                if not vals: continue
                t = pd.to_datetime([r['dateTime'] for r in vals], utc=True); v = np.array([float(r['value']) for r in vals]) * FT
                v[v < -900] = np.nan; got[pc] = (t, v)
            if '63160' in got:
                t, v = got['63160']; series[sid] = dict(t=t, v=v, datum='NAVD88', param='USGS 63160 water level NAVD88 (15 min)', src=how, n=int(np.isfinite(v).sum()))
            elif '00065' in got:
                t, v = got['00065']; series[sid] = dict(t=t, v=v, datum='local gage datum (offset to NAVD88 not published by USGS)', param='USGS 00065 gage height (15 min)', src=how, n=int(np.isfinite(v).sum()))
            else:
                series[sid] = dict(err='NWIS has no 63160/00065')
        if 'v' in series[sid]:
            s = series[sid]; pd.DataFrame({'time_utc': s['t'].strftime('%Y-%m-%dT%H:%M:%SZ'), 'value_m': np.round(s['v'], 4)}).to_csv(
                os.path.join(PROC, '%s_%s.csv' % (sid, 'navd88' if s['datum'] == 'NAVD88' else 'gage_local')), index=False)
            print('  %-9s %-40s %5d records %s' % (sid, s['param'], s['n'], s['src']))
    except Exception as e:
        series[sid] = dict(err='%s: %s' % (type(e).__name__, str(e)[:60])); print('  %-9s download failed %s' % (sid, series[sid]['err']))

# ---------- 3. Comparison ----------
rows = []; panels = []
for x, y, sid in obs_tbl:
    if sid not in hnames: continue
    k = hnames.index(sid); zm = his['point_zs'].values[:, k]; zb = float(his['point_zb'].values[k])
    if not np.isfinite(zm).any(): continue
    s = series.get(sid, {})
    r = dict(station=sid, model_zb_m=round(zb, 2), model_max_m=round(np.nanmax(zm), 3), model_tmax=str(t_his[np.nanargmax(zm)]),
             model_dry_frac=round(float(np.mean(zm - zb < 0.05)), 3), obs_param=s.get('param', ''), obs_datum=s.get('datum', ''), obs_n=s.get('n', 0), note=s.get('err', ''))
    if 'v' in s:
        so = pd.Series(s['v'], index=s['t']).dropna()
        so = so[(so.index >= T0) & (so.index <= T1)]
        # nearest-neighbour match of observations to model times (tolerance = half the obs sampling interval), no interpolation
        step = np.median(np.diff(so.index.values)).astype('timedelta64[s]').astype(int)
        om = so.reindex(t_his, method='nearest', tolerance=pd.Timedelta(seconds=step // 2 + 1)).values
        ok = np.isfinite(om) & np.isfinite(zm)
        r.update(obs_max_m=round(float(so.max()), 3), obs_tmax=str(so.idxmax()), obs_range_m=round(float(so.max() - so.min()), 3),
                 model_range_m=round(float(np.nanmax(zm) - np.nanmin(zm)), 3), n_pairs=int(ok.sum()),
                 corr=round(float(np.corrcoef(om[ok], zm[ok])[0, 1]), 3) if ok.sum() > 10 and np.std(zm[ok]) > 0 else np.nan)
        if s['datum'] == 'NAVD88':
            d = zm[ok] - om[ok]
            r.update(bias_m=round(float(d.mean()), 3), rmse_m=round(float(np.sqrt((d ** 2).mean())), 3), maxabs_m=round(float(np.abs(d).max()), 3),
                     peak_diff_m=round(float(np.nanmax(zm) - so.max()), 3), peak_dt_h=round((t_his[np.nanargmax(zm)] - so.idxmax()) / pd.Timedelta('1h'), 2))
        panels.append((sid, so, zm, zb, s, om, ok))
    rows.append(r)
stats = pd.DataFrame(rows); stats.to_csv(os.path.join(RDIR, 'obs_validation_stats.csv'), index=False)
print(stats.to_string())

# ---------- 4. Figure ----------
names = {'8720218': 'Mayport (boundary forcing station, not independent validation)', '8720219': 'Dames Point, St. Johns main channel', '02246515': 'Pottsburg Creek',
         '02246621': 'Trout River (tidal tributary)', '02246751': 'Broward River (tidal tributary)', '02246804': 'Dunn Creek (tidal tributary)', '02246825': 'Clapboard Creek (tidal tributary)'}
n = len(panels); ncol = 2; nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(13, 3.1 * nrow)); outer = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.22, top=0.95, bottom=0.05)
hh = lambda t: (t - TREF) / pd.Timedelta('1h')
for i, (sid, so, zm, zb, s, om, ok) in enumerate(panels):
    g = outer[i // ncol, i % ncol].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
    a1 = fig.add_subplot(g[0]); a2 = fig.add_subplot(g[1], sharex=a1)
    st = stats.set_index('station').loc[sid]
    if s['datum'] == 'NAVD88':
        a1.plot(hh(so.index), so.values, color=C_OBS, lw=0.9, label='observed (NAVD88)')
        a1.plot(hh(t_his), zm, color=C_MOD, lw=0.9, alpha=0.85, label='modelled')
        lo, hi = min(np.nanmin(zm), so.min()), max(np.nanmax(zm), so.max()); a1.set_ylim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
        if zb > lo - 0.15 * (hi - lo): a1.axhline(zb, color='#888', lw=0.6, ls='--')
        a1.set_title('%s  %s' % (sid, names.get(sid, '')), loc='left', fontsize=9)
        a1.text(0.99, 0.97, 'bias %+.2f  RMSE %.2f  r %.2f\npeak: obs %.2f / model %.2f m, timing diff %+.1f h' % (st['bias_m'], st['rmse_m'], st['corr'], st['obs_max_m'], st['model_max_m'], st['peak_dt_h']),
                transform=a1.transAxes, ha='right', va='top', fontsize=7, bbox=dict(fc='white', ec='#ccc', lw=0.5))
        res = np.where(ok, zm - om, np.nan); a2.fill_between(hh(t_his), res, 0, color=C_RES, alpha=0.35, lw=0); a2.plot(hh(t_his), res, color=C_RES, lw=0.6)
        a2.axhline(0, color='#888', lw=0.6); a2.set_ylabel('model - obs m', fontsize=7)
    else:
        a1.plot(hh(so.index), so.values, color=C_OBS, lw=0.9); a1.set_title('%s  %s' % (sid, names.get(sid, '')), loc='left', fontsize=9)
        a1.text(0.99, 0.97, 'obs: USGS gage height, local datum (offset to NAVD88 unknown)\ntidal range %.2f m; correlation with model r = %s' % (st['obs_range_m'], '%.2f' % st['corr'] if np.isfinite(st['corr']) else 'n/a'),
                transform=a1.transAxes, ha='right', va='top', fontsize=7, bbox=dict(fc='white', ec='#ccc', lw=0.5))
        a2.plot(hh(t_his), zm, color=C_MOD, lw=0.9); a2.axhline(zb, color='#888', lw=0.6, ls='--')
        a2.text(0.99, 0.9, 'model (NAVD88); dashed = cell bed %.2f m; dry %.0f%% of the time' % (zb, 100 * st['model_dry_frac']), transform=a2.transAxes, ha='right', va='top', fontsize=7, color=C_MOD)
    for a in (a1, a2): a.axvspan(0, 72, color='#999', alpha=0.10, lw=0); a.set_xlim(0, hh(T1))
    a1.tick_params(labelbottom=False); a1.set_ylabel('m', fontsize=7)
    if i // ncol == nrow - 1: a2.set_xlabel('h since %s' % TREF.strftime('%Y-%m-%d %H:%M UTC'), fontsize=8)
fig.suptitle('%s: in-domain gauges, observed vs modelled water level   black = obs, red = model, dashed = model cell bed elevation, grey = spin-up; NAVD88 stations overlaid with residuals, local-datum stations plotted in separate panels' % RUN, x=0.02, ha='left', fontsize=10.5, y=0.995)
out = os.path.join(FIG, 'fig_%s_06_obs_comparison.png' % RUN); fig.savefig(out, dpi=150, bbox_inches='tight'); print('->', out)
