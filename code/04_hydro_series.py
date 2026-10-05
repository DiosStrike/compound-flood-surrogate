#!/usr/bin/env python3
"""
04_hydro_series.py -- cleaning and hourly aggregation of the raw Mayport water-level and Acosta discharge series.

Time conventions (UTC throughout):
  * The full window is set by window_config.py (currently 2017-09-06 00:00 -- 09-16 00:00), 240 hourly intervals.
  * Hourly intervals are labelled by their **end time**: label T represents the interval [T-1h, T).
    Endpoints are "left-closed, right-open", so an instantaneous observation at T-1h belongs to label T,
    while an observation at T belongs to the next label. Thus every instantaneous record falls in exactly one interval,
    and 6-minute data have exactly 10 records per hour, 15-minute data exactly 4 records per hour.
  * Spin-up = first 72 labels, main event = last 168 labels (start/end in window_config.py)

This script does **no gap filling, interpolation, smoothing or outlier removal**.
Hours with insufficient coverage are output and flagged as-is; the user decides how to handle them.
"""
import os, sys, json, csv, shutil, datetime
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH

OLD = PTH.OLD
NEW = PTH.ROOT
BR = PTH.OLD_BR

import window_config as WCFG

T0, T1, TSPIN, LABELS = WCFG.pd_window(pd)   # 240 labels
print('window:', WCFG.NOTE)

FT3S_TO_M3S = 0.028316846592


def hour_label(ts):
    """Left-closed, right-open binning -> interval end-time label."""
    return ts.dt.floor('h') + pd.Timedelta('1h')


def aggregate(df, valcol, expected_per_hour, flagcols):
    """Aggregate by hourly interval. Returns a 240-row DataFrame; missing data are not filled."""
    d = df[(df.time_utc >= T0) & (df.time_utc < T1)].copy()
    d['hour_end'] = hour_label(d.time_utc)
    g = d.groupby('hour_end')
    out = pd.DataFrame(index=LABELS)
    out.index.name = 'hour_end_utc'
    out['n_valid'] = g[valcol].count()
    out['n_records'] = g.size()
    out['mean'] = g[valcol].mean()
    out['min'] = g[valcol].min()
    out['max'] = g[valcol].max()
    for c in flagcols:
        out[c] = g[c].max()
    out['n_valid'] = out['n_valid'].fillna(0).astype(int)
    out['n_records'] = out['n_records'].fillna(0).astype(int)
    out['n_expected'] = expected_per_hour
    out['coverage'] = out['n_valid'] / expected_per_hour
    out['complete'] = out['n_valid'] == expected_per_hour
    for c in flagcols:
        out[c] = out[c].fillna(False).astype(bool)
    out['period'] = np.where(out.index <= TSPIN, 'spinup', 'event')
    return out.reset_index()


def gap_report(df, valcol, step_min, label):
    """List the gaps inside the window (spacing between consecutive valid records > nominal step)."""
    d = df[(df.time_utc >= T0) & (df.time_utc <= T1)].copy()
    d = d[d[valcol].notna()].sort_values('time_utc')
    t = list(d.time_utc)
    gaps = []
    step = pd.Timedelta(minutes=step_min)
    if len(t) == 0:
        return gaps
    if t[0] > T0:
        gaps.append(dict(start=str(T0), end=str(t[0]),
                         minutes=float((t[0] - T0) / pd.Timedelta('1m')),
                         kind='missing before window start'))
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt > step:
            gaps.append(dict(start=str(t[i]), end=str(t[i + 1]),
                             minutes=float((dt - step) / pd.Timedelta('1m')),
                             kind='record gap'))
    if t[-1] < T1:
        gaps.append(dict(start=str(t[-1]), end=str(T1),
                         minutes=float((T1 - t[-1]) / pd.Timedelta('1m')),
                         kind='missing after window end'))
    for g in gaps:
        g['series'] = label
    return gaps


def main():
    raw = PTH.IRMA['raw']
    pro = PTH.IRMA['processed']
    meta = PTH.IRMA['meta']
    for d in (raw, pro, meta):
        os.makedirs(d, exist_ok=True)

    report = dict(
        generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        window_utc=[str(T0), str(T1)],
        n_hour_labels=len(LABELS),
        label_convention='label T represents the interval [T-1h, T), left-closed right-open; '
                         'the hourly value is the arithmetic mean of instantaneous records in that interval, not the instantaneous value at T',
        spinup=[str(T0 + pd.Timedelta('1h')), str(TSPIN)],
        event=[str(TSPIN + pd.Timedelta('1h')), str(T1)],
        no_gapfill='this script does no filling, interpolation, smoothing or outlier removal',
        series={}, gaps=[])

    # ---------------------------------------------------------- Mayport water level
    src = os.path.join(BR, 'noaa/8720218_water_level_raw.json')
    shutil.copy2(src, os.path.join(raw, 'noaa_8720218_water_level_raw.json'))
    shutil.copy2(os.path.join(BR, 'noaa/8720218_metadata.json'),
                 os.path.join(raw, 'noaa_8720218_metadata.json'))
    j = json.load(open(src))
    rows = []
    for r in j['data']:
        f = (r.get('f') or '0,0,0,0').split(',')
        f = [x.strip() for x in f] + ['0'] * 4
        try:
            v = float(r['v'])
        except (ValueError, TypeError):
            v = np.nan
        try:
            s = float(r['s'])
        except (ValueError, TypeError):
            s = np.nan
        rows.append(dict(
            time_utc=pd.Timestamp(r['t'], tz='UTC'),
            water_level_m_navd88=v, sigma_m=s,
            flag_inferred=(f[0] == '1'),
            flag_flat_tolerance=(f[1] == '1'),
            flag_rate_of_change=(f[2] == '1'),
            flag_temp_limit=(f[3] == '1'),
            quality=r.get('q', '')))
    wl = pd.DataFrame(rows).sort_values('time_utc')
    wl['source'] = 'NOAA CO-OPS 8720218 product=water_level datum=NAVD units=metric tz=gmt'
    wlw = wl[(wl.time_utc >= T0) & (wl.time_utc <= T1)]
    wlw.to_csv(os.path.join(
        pro, 'mayport_8720218_waterlevel_6min_utc_navd88_m.csv'),
        index=False)

    wl_h = aggregate(wl, 'water_level_m_navd88', 10,
                     ['flag_inferred', 'flag_flat_tolerance',
                      'flag_rate_of_change', 'flag_temp_limit'])
    wl_h = wl_h.rename(columns={
        'mean': 'water_level_m_navd88_hourly_mean',
        'min': 'water_level_m_navd88_hourly_min',
        'max': 'water_level_m_navd88_hourly_max',
        'flag_inferred': 'contains_inferred'})
    wl_h['station'] = '8720218'
    wl_h['datum'] = 'NAVD88'
    wl_h['units'] = 'm'
    wl_h.to_csv(os.path.join(
        pro, 'mayport_8720218_waterlevel_hourly_utc_navd88_m.csv'), index=False)

    report['series']['mayport_waterlevel'] = dict(
        station='NOAA CO-OPS 8720218 Mayport (Bar Pilots Dock)',
        lon=-81.42789, lat=30.398167,
        product='water_level (observed total water level, including astronomical tide and meteorological surge; not a tide prediction)',
        native_step_min=6,
        native_records_in_window=int(len(wlw)),
        native_expected=int((T1 - T0) / pd.Timedelta('6min')) + 1,
        datum='NAVD88', units='m', timezone='UTC',
        hourly_rows=int(len(wl_h)),
        hours_complete=int(wl_h.complete.sum()),
        hours_incomplete=int((~wl_h.complete).sum()),
        hours_with_inferred=int(wl_h.contains_inferred.sum()),
        inferred_records=int(wlw.flag_inferred.sum()),
        inferred_in_event=int(wlw[(wlw.time_utc >= TSPIN)].flag_inferred.sum()),
        quality_codes=dict(wlw.quality.value_counts()),
        hourly_range=[float(wl_h['water_level_m_navd88_hourly_mean'].min()),
                      float(wl_h['water_level_m_navd88_hourly_mean'].max())],
        note='NOAA flag position 1 = Inferred (official estimate); kept record by record and flagged in the hourly table '
             'as contains_inferred; quality=v means verified')
    report['gaps'] += gap_report(wl, 'water_level_m_navd88', 6, 'mayport_waterlevel')

    # ---------------------------------------------------------- Acosta discharge
    src = os.path.join(BR, 'usgs/02246500_00060.csv')
    shutil.copy2(src, os.path.join(raw, 'usgs_02246500_00060_raw.csv'))
    q = pd.read_csv(src)
    q['time_utc'] = pd.to_datetime(q.timestamp_utc, utc=True)
    q = q.sort_values('time_utc')
    # recompute the unit conversion independently to check value_si from the original download script
    q['discharge_m3s'] = q.value_original * FT3S_TO_M3S
    recheck = float(np.nanmax(np.abs(q.discharge_m3s - q.value_si)))
    q['flag_estimated'] = q.qualifiers.astype(str).str.contains('e', case=False)
    q['flag_provisional'] = q.qualifiers.astype(str).str.contains('P')
    q['flag_approved'] = q.qualifiers.astype(str).str.contains('A')
    q['source'] = ('USGS NWIS IV site=02246500 parameter=00060 '
                   '(raw signed total discharge, not de-tided)')
    qw = q[(q.time_utc >= T0) & (q.time_utc <= T1)]
    keep = ['time_utc', 'site_id', 'parameter_code', 'value_original',
            'original_unit', 'discharge_m3s', 'qualifiers', 'flag_estimated',
            'flag_provisional', 'flag_approved', 'source']
    qw[keep].to_csv(os.path.join(
        pro, 'acosta_02246500_discharge_15min_utc_m3s.csv'), index=False)

    q_h = aggregate(q, 'discharge_m3s', 4,
                    ['flag_estimated', 'flag_provisional', 'flag_approved'])
    q_h = q_h.rename(columns={
        'mean': 'discharge_m3s_hourly_mean',
        'min': 'discharge_m3s_hourly_min',
        'max': 'discharge_m3s_hourly_max',
        'flag_estimated': 'contains_estimated'})
    q_h['station'] = '02246500'
    q_h['units'] = 'm3/s'
    q_h['sign_convention'] = ('positive = downstream (seaward), negative = upstream (flood-tide reversal); '
                              'negative values kept as-is, not clipped to zero')
    q_h.to_csv(os.path.join(
        pro, 'acosta_02246500_discharge_hourly_utc_m3s.csv'), index=False)

    neg = int((qw.discharge_m3s < 0).sum())
    report['series']['acosta_discharge'] = dict(
        station='USGS 02246500 ST. JOHNS RIVER AT JACKSONVILLE, FL (Acosta Bridge)',
        lon=-81.6653735, lat=30.3224616,
        parameter='00060 Discharge, cubic feet per second (not tidally filtered)',
        native_step_min=15,
        native_records_in_window=int(len(qw)),
        native_expected=int((T1 - T0) / pd.Timedelta('15min')) + 1,
        units='m3/s', timezone='UTC',
        unit_conversion=f'value_original[ft3/s] x {FT3S_TO_M3S}; '
                        f'max deviation from value_si in the downloaded file {recheck:.3e} m3/s',
        negative_records=neg,
        negative_fraction=round(neg / max(1, len(qw)), 4),
        hourly_rows=int(len(q_h)),
        hours_complete=int(q_h.complete.sum()),
        hours_incomplete=int((~q_h.complete).sum()),
        hours_zero_records=int((q_h.n_valid == 0).sum()),
        hours_with_estimated=int(q_h.contains_estimated.sum()),
        qualifier_counts=dict(qw.qualifiers.astype(str).value_counts()),
        hourly_range=[float(q_h['discharge_m3s_hourly_mean'].min()),
                      float(q_h['discharge_m3s_hourly_mean'].max())],
        note='USGS qualifier A=Approved, P=Provisional, e=Estimated (official estimate). '
             'This is a tidally affected station; 00060 is the signed total discharge, not de-tided. 72137 is the tidally filtered value, '
             'not used, as required.')
    report['gaps'] += gap_report(q, 'discharge_m3s', 15, 'acosta_discharge')

    # ---------------------------------------------------------- list of hours with insufficient coverage
    inc = []
    for lab, d, col in (('mayport_waterlevel', wl_h, 'n_valid'),
                        ('acosta_discharge', q_h, 'n_valid')):
        bad = d[~d.complete]
        for _, r in bad.iterrows():
            inc.append(dict(series=lab,
                            hour_end_utc=str(r.hour_end_utc),
                            n_valid=int(r.n_valid),
                            n_expected=int(r.n_expected),
                            coverage=round(float(r.coverage), 3),
                            period=r.period))
    report['incomplete_hours'] = inc
    pd.DataFrame(inc).to_csv(
        os.path.join(meta, 'hydro_incomplete_hours.csv'), index=False)
    pd.DataFrame(report['gaps']).to_csv(
        os.path.join(meta, 'hydro_gaps.csv'), index=False)
    with open(os.path.join(meta, 'hydro_quality.json'), 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)

    for k, v in report['series'].items():
        print(f"--- {k}")
        for kk in ('native_records_in_window', 'native_expected',
                   'hourly_rows', 'hours_complete', 'hours_incomplete',
                   'hours_with_inferred', 'hours_with_estimated',
                   'hours_zero_records', 'negative_records', 'hourly_range',
                   'unit_conversion'):
            if kk in v:
                print(f'    {kk}: {v[kk]}')
    print(f"\n{len(report['gaps'])} gap periods, {len(inc)} hours with insufficient coverage")
    for g in report['gaps']:
        print('   ', g)


if __name__ == '__main__':
    main()
