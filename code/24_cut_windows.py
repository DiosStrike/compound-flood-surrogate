#!/usr/bin/env python3
"""
24_cut_windows.py -- cut the final 10-day windows (user-confirmed dates) from the 20-day products of the five events, re-plot and report. Does not modify raw data, does not run SFINCS.

Window logic (confirmed by the user 2026-09-15): center day −6 d 00:00 → center day +4 d 00:00, [T0, T1) with 240 hourly intervals;
  spin-up = first 72 labels (T0 → T0+3d), main simulation = last 168 (T0+3d → T1). Label T = [T−1h, T).
Outputs in data/<event>/processed/window10d/:
  mayport_6min_window.csv          native 6 min, T0 ≤ t ≤ T1 (2401 records incl. the T1 endpoint, for SFINCS bzs; the endpoint record belongs to no hourly interval)
  acosta_15min_window.csv          native 15 min, T0 ≤ t ≤ T1 (961 records incl. endpoint, for SFINCS dis)
  mayport_hourly_240.csv / acosta_hourly_240.csv   240 hourly statistics (from the 20-day hourly tables, flags unchanged)
  rain_<product>_hourly_240.nc     240 frames of spatial rainfall (precip_mm raw / precip_mm_filled / flag / lat / lon)
  rain_<product>_areal_240.csv     areal mean (for check plots)
  window_quality_<event>.json      coverage, gaps (assigned to spin-up / event period), peaks, boundary endpoint checks
result/figures/events/fig_<event>_10d_timeseries.{png,svg,_data.csv}
"""
import os, sys, json, datetime, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH, geoio
spec = importlib.util.spec_from_file_location('m23', os.path.join(PTH.CODE, '23_collect_events.py')); m23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m23)
UTC = datetime.timezone.utc; FILL = -9999.0
WINDOWS = {   # user-confirmed: spin-up start / main period start / end
    'matthew': ('2016-10-01', '2016-10-04', '2016-10-11'),
    'ian':     ('2022-09-23', '2022-09-26', '2022-10-03'),
    'milton':  ('2024-10-05', '2024-10-08', '2024-10-15'),   # user decided 2026-09-15 to shift the whole window one day later, to include the 10-14 18Z low-frequency discharge peak
    'dorian':  ('2019-08-29', '2019-09-01', '2019-09-08'),
    'beryl':   ('2012-05-22', '2012-05-25', '2012-06-01'),
}


def runs_of(mask, labels):
    return m23.runs_of(mask, labels)


def cut(ev):
    e = m23.EVENTS[ev]; D = e['dirs']; pro = D['processed']; raw = D['raw']
    out = os.path.join(pro, 'window10d'); os.makedirs(out, exist_ok=True)
    T0 = pd.Timestamp(WINDOWS[ev][0], tz='UTC'); TS = pd.Timestamp(WINDOWS[ev][1], tz='UTC'); T1 = pd.Timestamp(WINDOWS[ev][2], tz='UTC')
    C = pd.Timestamp(e['center'], tz='UTC')
    assert TS - T0 == pd.Timedelta('3D') and T1 - TS == pd.Timedelta('7D'), 'window is not 3 + 7 days'
    Q_offset_days = int((T0 - (C - pd.Timedelta('6D'))) / pd.Timedelta('1D'))   # offset relative to the "center day −6 d" rule (Milton = +1)
    assert T0 >= pd.Timestamp(e['t0']) and T1 <= pd.Timestamp(e['t1']), '10-day window exceeds the 20-day collection range'
    labels = pd.date_range(T0 + pd.Timedelta('1h'), T1, freq='h'); assert len(labels) == 240
    seg = np.where(labels <= TS, 'spinup', 'event')
    Q = dict(event=ev, name=e['name'], center_utc=e['center'], window=dict(T0=str(T0), spinup_end=str(TS), T1=str(T1), n_hours=240, spinup_hours=72, event_hours=168, offset_days_vs_rule=Q_offset_days,
             convention='[T0,T1) left-closed right-open; label T=[T−1h,T); spin-up = labels ≤ T0+3d'), generated_utc=datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), series={}, gaps=[], boundary_endpoints={})
    # ---------------- native series (for SFINCS boundaries): including the T1 endpoint
    wl6 = pd.read_csv(os.path.join(pro, 'mayport_8720218_waterlevel_6min_utc_navd88_m.csv')); wl6['time_utc'] = pd.to_datetime(wl6.time_utc, utc=True)
    w = wl6[(wl6.time_utc >= T0) & (wl6.time_utc <= T1)].reset_index(drop=True); w.to_csv(os.path.join(out, 'mayport_6min_window.csv'), index=False)
    q15 = pd.read_csv(os.path.join(pro, 'acosta_02246500_discharge_15min_utc_m3s.csv')); q15['time_utc'] = pd.to_datetime(q15.time_utc, utc=True)
    q = q15[(q15.time_utc >= T0) & (q15.time_utc <= T1)].reset_index(drop=True); q.to_csv(os.path.join(out, 'acosta_15min_window.csv'), index=False)
    Q['boundary_endpoints'] = dict(
        mayport=dict(records=int(len(w)), expected=2401, missing_records=int(w.record_missing.sum()), first=str(w.time_utc.iloc[0]), last=str(w.time_utc.iloc[-1]),
                     endpoint_T0_valid=bool(np.isfinite(w.water_level_m_navd88.iloc[0])), endpoint_T1_valid=bool(np.isfinite(w.water_level_m_navd88.iloc[-1])),
                     inferred_records=int(w.flag_inferred.sum()), note='T1 endpoint record only serves the SFINCS tstop boundary, not counted in the 240 hourly intervals'),
        acosta=dict(records=int(len(q)), expected=961, missing_records=int(q.record_missing.sum()), first=str(q.time_utc.iloc[0]), last=str(q.time_utc.iloc[-1]),
                    endpoint_T0_valid=bool(np.isfinite(q.discharge_m3s.iloc[0])), endpoint_T1_valid=bool(np.isfinite(q.discharge_m3s.iloc[-1])),
                    estimated_records=int(q.flag_estimated.sum()), gaps_15min=[(str(a), int(d)) for a, d in zip(q.time_utc.values[:-1], np.diff(q.time_utc.values).astype('timedelta64[m]').astype(int)) if d != 15]))
    # ---------------- hourly tables, 240
    def hourly(fn, vcol, name, expected):
        H = pd.read_csv(os.path.join(pro, fn)); H['hour_end_utc'] = pd.to_datetime(H.hour_end_utc, utc=True)
        H = H[(H.hour_end_utc > T0) & (H.hour_end_utc <= T1)].reset_index(drop=True); assert len(H) == 240
        H['segment'] = seg
        for c in ('partial', 'interp', 'unfillable'): H[c] = H[c].astype(bool)
        H.to_csv(os.path.join(out, name + '_hourly_240.csv'), index=False)
        vals = H[vcol].values
        def cnt(mask, s): return int((mask & (H.segment == s)).sum())
        stats = dict(hours=240, native_valid=int(H.n_valid.sum()), native_expected=int(expected * 240), native_coverage=float(H.n_valid.sum() / (expected * 240)),
                     hours_full=int((H.n_valid == expected).sum()), hours_partial=dict(spinup=cnt(H.partial, 'spinup'), event=cnt(H.partial, 'event')),
                     hours_official_est=dict(spinup=cnt(H.official_est_frac > 0, 'spinup'), event=cnt(H.official_est_frac > 0, 'event')),
                     hours_interp=dict(spinup=cnt(H.interp, 'spinup'), event=cnt(H.interp, 'event')), hours_unfillable=int(H.unfillable.sum()),
                     longest_gap_h=max([n for a, b, n in runs_of((H.n_valid == 0).values, list(H.hour_end_utc))] or [0]),
                     completeness_after_fill=float(np.isfinite(vals).mean()))
        for a, b, n in runs_of((H.n_valid == 0).values, list(H.hour_end_utc)):
            Q['gaps'].append(dict(series=name, start=str(a), end=str(b), hours=n, type='whole hour missing -> interpolated', segment='spinup' if a <= TS else 'event'))
        for a, b, n in runs_of(H.partial.values, list(H.hour_end_utc)):
            Q['gaps'].append(dict(series=name, start=str(a), end=str(b), hours=n, type='insufficient samples within hour (mean of available samples)', segment='spinup' if a <= TS else 'event'))
        return H, stats
    W, sw = hourly('mayport_8720218_waterlevel_hourly_utc_navd88_m.csv', 'water_level_m_navd88_filled', 'mayport', 10)
    i = int(np.nanargmax(W.water_level_m_navd88_filled.values)); sw['peak'] = dict(time=str(W.hour_end_utc[i]), value=float(W.water_level_m_navd88_filled[i]), segment=seg[i]); Q['series']['mayport_waterlevel'] = sw
    Wq, sq = hourly('acosta_02246500_discharge_hourly_utc_m3s.csv', 'discharge_m3s_filled', 'acosta', 4)
    i = int(np.nanargmax(Wq.discharge_m3s_filled.values)); j = int(np.nanargmin(Wq.discharge_m3s_filled.values))
    # 25 h low-frequency component peak (check whether the river response falls in the event period)
    ql = pd.Series(Wq.discharge_m3s_filled.values).rolling(25, center=True, min_periods=13).mean().values; k = int(np.nanargmax(ql))
    sq['peak'] = dict(max_time=str(Wq.hour_end_utc[i]), max=float(Wq.discharge_m3s_filled[i]), max_segment=seg[i], min_time=str(Wq.hour_end_utc[j]), min=float(Wq.discharge_m3s_filled[j]), min_segment=seg[j],
                      qlow25h_max_time=str(Wq.hour_end_utc[k]), qlow25h_max=float(ql[k]), qlow25h_segment=seg[k]); Q['series']['acosta_discharge'] = sq
    # ---------------- rainfall: decoded cache -> per-cell interpolation -> 240 frames
    z = np.load(os.path.join(raw, 'rain', 'rain_decoded_cache_%s.npz' % e['rain']))
    P = z['P'].astype(np.float64); FLAG = z['FLAG']; LAT2, LON2 = z['lat'], z['lon']
    zt = pd.to_datetime(z['t'], unit='s', utc=True)
    nT, ny, nx = P.shape; miss = ~np.isfinite(P); PF = P.copy(); KIND = np.zeros_like(FLAG); tt = np.arange(nT, dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            mm = miss[:, iy, ix]
            if not mm.any():
                continue
            ok = np.where(~mm)[0]
            if len(ok) == 0:
                KIND[:, iy, ix] = 3; continue
            fill = np.interp(tt, tt[ok], P[ok, iy, ix]); inside = mm & (tt >= ok[0]) & (tt <= ok[-1])
            PF[inside, iy, ix] = fill[inside]; KIND[inside, iy, ix] = np.where(FLAG[inside, iy, ix] == 2, 2, 1); KIND[mm & ~inside, iy, ix] = 3
    sel = np.array([zt.get_loc(t) for t in labels])
    P240, PF240, K240 = P[sel], PF[sel], KIND[sel]
    ext_j = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))['latlon_bbox_nad83']
    inbox = (LAT2 >= ext_j['latmin']) & (LAT2 <= ext_j['latmax']) & (LON2 >= ext_j['lonmin']) & (LON2 <= ext_j['lonmax'])
    areal_raw = np.array([np.nanmean(P240[k][inbox]) if np.isfinite(P240[k][inbox]).any() else np.nan for k in range(240)])
    areal_fill = np.array([np.nanmean(PF240[k][inbox]) for k in range(240)])
    cov = np.array([np.isfinite(P240[k][inbox]).mean() for k in range(240)])
    A = pd.DataFrame(dict(hour_end_utc=labels, segment=seg, areal_mean_mm_raw=areal_raw, areal_mean_mm_filled=areal_fill, cell_coverage_frac=cov,
                          n_cells_interp=[int(((K240[k] > 0) & (K240[k] < 3) & inbox).sum()) for k in range(240)], n_cells_unfillable=[int(((K240[k] == 3) & inbox).sum()) for k in range(240)]))
    A['cum_mm_filled'] = np.cumsum(A.areal_mean_mm_filled); A['source_class'] = np.select([A.cell_coverage_frac == 0, A.cell_coverage_frac < 1], ['missing_interp', 'partial_cells'], 'obs')
    A.to_csv(os.path.join(out, 'rain_%s_areal_240.csv' % e['rain']), index=False)
    ncpath = os.path.join(out, 'rain_%s_hourly_240.nc' % e['rain'])
    wr = geoio.NC3Writer(ncpath); wr.add_dim('time', 240); wr.add_dim('y', ny); wr.add_dim('x', nx)
    wr.add_gattr('title', '%s hourly precipitation, 10-day window, event %s' % (e['rain'], e['name'])); wr.add_gattr('window', '[%s, %s) UTC; spin-up until %s' % (T0, T1, TS))
    wr.add_gattr('time_convention', 'T = accumulation over [T-1h, T)'); wr.add_gattr('grid', 'native grid, not reprojected; lat/lon of cell centres'); wr.add_gattr('fill_rule', 'flag 0 obs,1 interp(no coverage),2 interp(file missing),3 unfillable')
    wr.add_var('time', ['time'], np.arange(1, 241, dtype=np.float64), {'units': 'hours since %s' % T0.strftime('%Y-%m-%d %H:%M'), 'long_name': 'end of 1-hour interval (UTC)'})
    wr.add_var('lat', ['y', 'x'], LAT2.astype(np.float64), {'units': 'degrees_north'}); wr.add_var('lon', ['y', 'x'], LON2.astype(np.float64), {'units': 'degrees_east'})
    wr.add_var('precip_mm', ['time', 'y', 'x'], np.where(np.isfinite(P240), P240, FILL).astype(np.float32), {'units': 'mm', '_FillValue': np.float32(FILL)})
    wr.add_var('precip_mm_filled', ['time', 'y', 'x'], np.where(np.isfinite(PF240), PF240, FILL).astype(np.float32), {'units': 'mm', '_FillValue': np.float32(FILL)})
    wr.add_var('flag', ['time', 'y', 'x'], K240.astype(np.int8), {'flag_meanings': 'obs interp_nocov interp_filemissing unfillable'})
    wr.add_var('in_study_rect', ['y', 'x'], inbox.astype(np.int8), {}); wr.write()
    missing_h = np.array([cov[k] == 0 for k in range(240)]); k = int(np.nanargmax(areal_fill))
    sr = dict(product=e['rain'], grid_shape=[int(ny), int(nx)], n_cells_in_rect=int(inbox.sum()), hours_missing=dict(spinup=int((missing_h & (seg == 'spinup')).sum()), event=int((missing_h & (seg == 'event')).sum())),
              hours_partial_cells=int(((cov > 0) & (cov < 1)).sum()), cellhours_interp=int((((K240 == 1) | (K240 == 2)) & inbox[None]).sum()), cellhours_unfillable=int(((K240 == 3) & inbox[None]).sum()),
              longest_gap_h=max([n for a, b, n in runs_of(missing_h, list(labels))] or [0]), total_mm_areal=dict(spinup=float(areal_fill[seg == 'spinup'].sum()), event=float(areal_fill[seg == 'event'].sum())),
              peak=dict(time=str(labels[k]), areal_mm=float(areal_fill[k]), segment=seg[k]), completeness_after_fill=float(np.isfinite(PF240[:, inbox]).mean()),
              ampr_note='SFINCS ampr needs 241 time blocks (hourly from 0 s, last block = T1); these 240 frames are labels T0+1h...T1, frame k is the rain rate of [T0+(k−1)h, T0+kh), consistent with Irma')
    for a, b, n in runs_of(missing_h, list(labels)):
        Q['gaps'].append(dict(series='rain', start=str(a), end=str(b), hours=n, type='missing file -> per-cell interpolation', segment='spinup' if a <= TS else 'event'))
    Q['series']['rain'] = sr
    Q['peaks_in_event_segment'] = dict(waterlevel=sw['peak']['segment'] == 'event', discharge_qlow25h=sq['peak']['qlow25h_segment'] == 'event', rain=sr['peak']['segment'] == 'event')
    json.dump(Q, open(os.path.join(out, 'window_quality_%s.json' % ev), 'w'), ensure_ascii=False, indent=1, default=str)
    plot(ev, W, Wq, A, Q, T0, TS, T1)
    print(ev, 'ok', json.dumps(Q['peaks_in_event_segment']))
    return Q


def plot(ev, W, Wq, A, Q, T0, TS, T1):
    import matplotlib; matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    e = m23.EVENTS[ev]; t = pd.to_datetime(W.hour_end_utc)
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    C = dict(obs='#1f4e79', est='#e08a1e', interp='#c0392b', partial='#7f7f7f')
    def series(a, df, ycol, ylabel):
        y = df[ycol].values
        a.plot(t, y, color=C['obs'], lw=1.0, label='hourly value (mean of observations)')
        m = df.official_est_frac > 0; a.plot(t[m], y[m], 'o', ms=4, color=C['est'], label='includes official estimates')
        m = df.partial; a.plot(t[m], y[m], 's', ms=6, mfc='none', mec=C['partial'], label='insufficient samples within hour')
        m = df.interp; a.plot(t[m], y[m], 'x', ms=7, color=C['interp'], label='self-interpolated')
        for a0, b0, n in runs_of((df.n_valid == 0).values, list(t)):
            a.axvspan(a0 - pd.Timedelta('1h'), b0, color=C['interp'], alpha=0.15, lw=0)
        a.set_ylabel(ylabel); a.grid(alpha=0.3)
    series(ax[0], W, 'water_level_m_navd88_filled', 'Mayport water level (m NAVD88)')
    series(ax[1], Wq, 'discharge_m3s_filled', 'Acosta discharge (m³/s, + downstream)'); ax[1].axhline(0, color='k', lw=0.6)
    ta = pd.to_datetime(A.hour_end_utc)
    ax[2].bar(ta, A.areal_mean_mm_filled, width=1 / 24, color=C['obs'], label='areal mean hourly rain (mm)')
    m = A.source_class != 'obs'; ax[2].bar(ta[m], A.areal_mean_mm_filled[m], width=1 / 24, color=C['interp'], label='hours with interpolation / partial coverage')
    for a0, b0, n in runs_of((A.cell_coverage_frac == 0).values, list(ta)):
        ax[2].axvspan(a0 - pd.Timedelta('1h'), b0, color=C['interp'], alpha=0.15, lw=0)
    ax[2].set_ylabel('study-area mean hourly rain (mm)'); ax[2].grid(alpha=0.3)
    ax[3].plot(ta, A.cum_mm_filled, color=C['obs'], lw=1.4, label='areal mean cumulative rain (gap-filled)'); ax[3].set_ylabel('cumulative rain (mm)'); ax[3].grid(alpha=0.3)
    P = Q['series']
    ax[0].plot(pd.Timestamp(P['mayport_waterlevel']['peak']['time']), P['mayport_waterlevel']['peak']['value'], '*', ms=13, color='gold', mec='k', label='peak %.2f m @ %s' % (P['mayport_waterlevel']['peak']['value'], P['mayport_waterlevel']['peak']['time'][5:16]))
    d = P['acosta_discharge']['peak']
    ax[1].plot(pd.Timestamp(d['max_time']), d['max'], '*', ms=13, color='gold', mec='k', label='max %+.0f @ %s' % (d['max'], d['max_time'][5:16]))
    ax[1].plot(pd.Timestamp(d['min_time']), d['min'], '*', ms=13, color='#9ecae1', mec='k', label='min %+.0f @ %s' % (d['min'], d['min_time'][5:16]))
    ax[1].plot(pd.Timestamp(d['qlow25h_max_time']), d['qlow25h_max'], 'D', ms=8, color='#2a7f62', mec='k', label='25 h low-frequency peak %+.0f @ %s' % (d['qlow25h_max'], d['qlow25h_max_time'][5:16]))
    r = P['rain']['peak']; ax[2].plot(pd.Timestamp(r['time']), r['areal_mm'], '*', ms=13, color='gold', mec='k', label='peak %.1f mm/h @ %s' % (r['areal_mm'], r['time'][5:16]))
    c = pd.Timestamp(e['center'], tz='UTC')
    for a in ax:
        a.axvspan(T0, TS, color='#bbbbbb', alpha=0.25, lw=0); a.axvline(TS, color='k', lw=1.0, ls='--'); a.axvline(c, color='#2a7f62', lw=1.2, ls='--')
        h, l = a.get_legend_handles_labels(); a.legend(h, l, fontsize=7.5, loc='upper left', ncol=3)
    y0 = ax[3].get_ylim()[1] * 0.90; ax[3].text(T0 + pd.Timedelta('3h'), y0, 'spin-up 72 h', fontsize=10, color='#333', fontweight='bold'); ax[3].text(TS + pd.Timedelta('3h'), y0, 'main simulation 168 h', fontsize=10, color='#333', fontweight='bold')
    ax[3].set_xlabel('UTC (grey = spin-up; black dashed = main period start %s; green dashed = center day %s; red shading = raw whole-hour missing)' % (TS.strftime('%m-%d'), e['center']))
    ax[3].set_xlim(T0, T1)
    fig.suptitle('%s %d -- final 10-day window [%s, %s) UTC, 240 h = 72 h spin-up + 168 h main; rainfall %s' % (e['name'], e['year'], T0.strftime('%Y-%m-%d'), T1.strftime('%Y-%m-%d'), e['rain']), fontsize=11)
    fig.tight_layout(); fdir = os.path.join(PTH.FIG, 'events'); os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, 'fig_%s_10d_timeseries.png' % ev), dpi=140); fig.savefig(os.path.join(fdir, 'fig_%s_10d_timeseries.svg' % ev)); plt.close(fig)
    pd.DataFrame(dict(hour_end_utc=W.hour_end_utc, segment=W.segment, water_level_m=W.water_level_m_navd88_filled, wl_class=W.source_class, discharge_m3s=Wq.discharge_m3s_filled, q_class=Wq.source_class,
                      rain_areal_mm=A.areal_mean_mm_filled, rain_class=A.source_class, rain_cum_mm=A.cum_mm_filled)).to_csv(os.path.join(fdir, 'fig_%s_10d_timeseries_data.csv' % ev), index=False)


if __name__ == '__main__':
    evs = list(WINDOWS) if len(sys.argv) < 2 or sys.argv[1] == 'all' else sys.argv[1:]
    summ = {ev: cut(ev) for ev in evs}
    json.dump(summ, open(os.path.join(PTH.STATIC_META, 'events_window10d_summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
