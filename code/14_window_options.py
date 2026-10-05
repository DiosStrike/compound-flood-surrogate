#!/usr/bin/env python3
"""
14_window_options.py -- comparison of three candidate simulation time windows (evaluation only; no delivered data is modified)

Options (all 72 h spin-up + 168 h event = 240 hourly labels; label T represents [T-1h, T)):
  W0 current        2017-09-03 00:00 -> 09-13 00:00   spin-up until 09-06
  WA shifted +2 d   2017-09-05 00:00 -> 09-15 00:00   spin-up until 09-08
  WB shifted +3 d   2017-09-06 00:00 -> 09-16 00:00   spin-up until 09-09

Outputs:
  data/interim/mrms_P_0903_0916.npy   (312 h gridded fields, for window cutting)
  data/meta/window_options.json
  result/figures/fig_window_options_overview.png / _WA.png / _WB.png
  result/reports/Flood_2.0_window_options_comparison.pdf
"""
import os, sys, io, json, struct, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OLD = PTH.OLD
NEW = PTH.ROOT
RAWDIR = PTH.OLD_MRMS
BR = PTH.OLD_BR
INT, META = os.path.join(PTH.IRMA['raw'], 'mrms_cache'), PTH.IRMA['meta']
FIG, REP = PTH.FIG, PTH.REP
MISSING = -3.0

# full period: 09-03 01:00 .. 09-16 00:00, 312 hourly labels
ALL0 = pd.Timestamp('2017-09-03 00:00', tz='UTC')
ALL1 = pd.Timestamp('2017-09-16 00:00', tz='UTC')
ALLH = pd.date_range(ALL0 + pd.Timedelta('1h'), ALL1, freq='h')   # 312

WINDOWS = [
    dict(key='W0', name='current', t0='2017-09-03 00:00', color='#777777'),
    dict(key='WA', name='shifted +2 d', t0='2017-09-05 00:00', color='#1b9e77'),
    dict(key='WB', name='shifted +3 d', t0='2017-09-06 00:00', color='#d95f02'),
]
for w in WINDOWS:
    w['T0'] = pd.Timestamp(w['t0'], tz='UTC')
    w['TSPIN'] = w['T0'] + pd.Timedelta('72h')
    w['T1'] = w['T0'] + pd.Timedelta('240h')
    w['labels'] = pd.date_range(w['T0'] + pd.Timedelta('1h'), w['T1'], freq='h')


# ---------------- MRMS: fill in 09-13 01:00 .. 09-16 00:00 ----------------
# GRIB2 decoding reuses the code in 05_mrms_rainfall.py, not rewritten
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'mrms05', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '05_mrms_rainfall.py'))
_m05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m05)
read_grib2_sections = _m05.read_grib2_sections
grid_of = _m05.grid_of


def unpack_values(sec):
    v, _ = _m05.unpack_values(sec)
    return v


def build_mrms():
    cache = os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_P_0903_0916.npy')
    if os.path.exists(cache):
        return np.load(cache), np.load(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_lonlat.npy'))
    P_old = np.load(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_P.npy'))           # 240 x ny x nx
    LL = np.load(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_lonlat.npy'))
    lon0, lon1, lat0, lat1, dlon, dlat = LL
    ny, nx = P_old.shape[1], P_old.shape[2]
    need = ALLH[240:]                                          # 72 hours
    print('Decoding %d additional MRMS files (%s .. %s)' % (len(need), need[0], need[-1]))
    ext = json.load(open(os.path.join(META, 'study_extent.json')))
    ll = ext['latlon_bbox_nad83']
    out = np.full((len(need), ny, nx), np.nan)
    for k, t in enumerate(need):
        f = os.path.join(RAWDIR, 'GaugeCorr_QPE_01H_00.00_%s.grib2'
                         % t.strftime('%Y%m%d-%H%M%S'))
        if not os.path.exists(f):
            print('  missing file', t)
            continue
        sec = read_grib2_sections(f)
        G = grid_of(sec[3])
        la = G['lat0'] + np.arange(G['nj']) * G['dlat']
        lo = G['lon0'] + np.arange(G['ni']) * G['dlon']
        ci = np.where((lo >= ll['lonmin']) & (lo <= ll['lonmax']))[0]
        ri = np.where((la >= ll['latmin']) & (la <= ll['latmax']))[0]
        c0, c1 = max(0, ci[0] - 2), min(G['ni'], ci[-1] + 3)
        r0, r1 = max(0, ri[0] - 2), min(G['nj'], ri[-1] + 3)
        v = np.asarray(unpack_values(sec)).reshape(G['nj'], G['ni'])[r0:r1, c0:c1]
        out[k] = v
        if (k + 1) % 24 == 0:
            print('  %d/%d' % (k + 1, len(need)))
    P = np.concatenate([P_old, out], 0)
    np.save(cache, P)
    return P, LL


P_all, LL = build_mrms()
print('MRMS full period', P_all.shape)
valid = P_all > MISSING / 2
rain = np.where(valid, np.clip(P_all, 0, None), np.nan)
rain_areal = pd.Series(np.nanmean(rain.reshape(len(ALLH), -1), 1), index=ALLH)


# ---------------- water level / discharge ----------------
wl = pd.read_csv(os.path.join(BR, 'noaa/8720218_water_level_utc_navd88_m.csv'))
wl['t'] = pd.to_datetime(wl.time_utc, utc=True)
wl = wl[['t', 'water_level_m_navd88']].dropna().sort_values('t')

q = pd.read_csv(os.path.join(BR, 'usgs/02246500_00060.csv'))
q['t'] = pd.to_datetime(q.timestamp_utc, utc=True)
q = q[q.valid.astype(str).str.lower().isin(['true', '1'])]
q = q[['t', 'value_si']].dropna().sort_values('t')
print('water level %s .. %s (%d records); discharge %s .. %s (%d records)'
      % (wl.t.min(), wl.t.max(), len(wl), q.t.min(), q.t.max(), len(q)))


def hourly(df, col, labels):
    """Hourly mean over the left-closed, right-open interval [T-1h, T)"""
    out = []
    for T in labels:
        m = (df.t >= T - pd.Timedelta('1h')) & (df.t < T)
        s = df.loc[m, col]
        out.append(dict(T=T, n=len(s), mean=s.mean() if len(s) else np.nan,
                        lo=s.min() if len(s) else np.nan,
                        hi=s.max() if len(s) else np.nan))
    return pd.DataFrame(out).set_index('T')


WL_all = hourly(wl, 'water_level_m_navd88', ALLH)
Q_all = hourly(q, 'value_si', ALLH)


# ---------------- statistics per option ----------------
def stats(w):
    lab = w['labels']
    r = rain_areal.reindex(lab)
    h = WL_all.reindex(lab)
    d = Q_all.reindex(lab)
    spin = lab <= w['TSPIN']
    ev = ~spin
    def pk(s):
        i = s.idxmax()
        return (float(s.max()), i)
    rmax, rat = pk(r)
    hmax, hat = pk(h['mean'])
    qmax, qat = pk(d['mean'])
    # "calmness" of the last 48 h before the event ends: water-level and discharge amplitude relative to the full-window maximum
    tail = lab[-48:]
    return dict(
        key=w['key'], name=w['name'],
        t0=str(w['T0']), tspin=str(w['TSPIN']), t1=str(w['T1']),
        rain_hours_missing=int(r.isna().sum()),
        wl_hours_missing=int(h['mean'].isna().sum()),
        q_hours_missing=int(d['mean'].isna().sum()),
        rain_total_mm=round(float(r.sum()), 1),
        rain_spinup_mm=round(float(r[spin].sum()), 1),
        rain_event_mm=round(float(r[ev].sum()), 1),
        rain_peak_mm_h=round(rmax, 2), rain_peak_at=str(rat),
        rain_peak_in_event=bool(rat > w['TSPIN']),
        rain_peak_pos_pct=round(100 * (rat - w['T0']) / (w['T1'] - w['T0']), 1),
        wl_peak_m=round(hmax, 3), wl_peak_at=str(hat),
        wl_peak_pos_pct=round(100 * (hat - w['T0']) / (w['T1'] - w['T0']), 1),
        q_peak_m3s=round(qmax, 1), q_peak_at=str(qat),
        q_peak_pos_pct=round(100 * (qat - w['T0']) / (w['T1'] - w['T0']), 1),
        q_hours_after_peak=int((w['T1'] - qat) / pd.Timedelta('1h')),
        wl_hours_after_peak=int((w['T1'] - hat) / pd.Timedelta('1h')),
        wl_range_tail48=round(float(h['mean'].reindex(tail).max()
                                    - h['mean'].reindex(tail).min()), 3),
        wl_max_tail48=round(float(h['mean'].reindex(tail).max()), 3),
        q_max_tail48=round(float(d['mean'].reindex(tail).max()), 1),
        rain_tail48_mm=round(float(r.reindex(tail).sum()), 1),
        spinup_rain_max_mm_h=round(float(r[spin].max()), 2),
        spinup_wl_max_m=round(float(h['mean'][spin].max()), 3),
    )


S = {w['key']: stats(w) for w in WINDOWS}
for k in ('W0', 'WA', 'WB'):
    s = S[k]
    print('%s %s  rain %.0f mm (spin-up %.0f)  water-level peak %.2f m @%s (%.0f%% into window)  '
          'discharge peak %.0f @%s  %d h left after peak'
          % (k, s['name'], s['rain_total_mm'], s['rain_spinup_mm'], s['wl_peak_m'],
             s['wl_peak_at'][5:16], s['wl_peak_pos_pct'], s['q_peak_m3s'],
             s['q_peak_at'][5:16], s['q_hours_after_peak']))

json.dump(dict(generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
               note='window-selection evaluation only; no delivered data modified', options=S),
          open(os.path.join(META, 'window_options.json'), 'w'),
          ensure_ascii=False, indent=1)


# ================= figures =================
def panel_series(ax, kind):
    if kind == 'rain':
        ax.bar(ALLH, rain_areal.values, width=0.036, color='#1f77b4')
        ax.set_ylabel('Areal-mean rainfall\n(mm/h)', fontsize=10)
    elif kind == 'wl':
        ax.fill_between(ALLH, WL_all['lo'], WL_all['hi'], color='#0057b7', alpha=0.22)
        ax.plot(ALLH, WL_all['mean'], color='#0057b7', lw=1.3)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_ylabel('Mayport water level\n(m, NAVD88)', fontsize=10)
    else:
        ax.fill_between(ALLH, Q_all['lo'], Q_all['hi'], color='#e6194B', alpha=0.20)
        ax.plot(ALLH, Q_all['mean'], color='#e6194B', lw=1.3)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_ylabel('Acosta discharge\n(m³/s)', fontsize=10)
    ax.grid(alpha=0.22, lw=0.5)


def fig_overview(pdf):
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 8.4), sharex=True)
    for ax, k in zip(axes, ('rain', 'wl', 'discharge')):
        panel_series(ax, k)
        y0, y1 = ax.get_ylim()
        for i, w in enumerate(WINDOWS):
            hgt = (y1 - y0) * 0.055
            yy = y1 - hgt * (i + 1) * 1.25
            ax.add_patch(Rectangle((mdates.date2num(w['T0']), yy),
                                   mdates.date2num(w['T1']) - mdates.date2num(w['T0']),
                                   hgt, fc=w['color'], alpha=0.75, ec='k', lw=0.6,
                                   zorder=10))
            ax.plot([mdates.date2num(w['TSPIN'])] * 2, [yy, yy + hgt], color='w',
                    lw=1.6, zorder=11)
            if k == 'rain':
                ax.text(mdates.date2num(w['T0']) + 0.12, yy + hgt * 0.5,
                        '%s %s' % (w['key'], w['name']), va='center', fontsize=8.6,
                        color='w', fontweight='bold', zorder=12)
        ax.set_ylim(y0, y1)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[2].xaxis.set_major_locator(mdates.DayLocator())
    axes[2].set_xlabel('Time (UTC)', fontsize=10)
    axes[0].set_title('Three candidate windows over the full record (white tick = boundary between spin-up and event period)',
                      fontsize=11)
    fig.suptitle('Figure 1   Window options overview -- available record 2017-09-03 -> 09-16 UTC', fontsize=14.5,
                 y=0.985)
    fig.text(0.5, 0.012,
             'Each option is 72 h spin-up + 168 h event = 240 hourly labels. '
             'All three datasets are complete over 09-03 -> 09-16 (raw rainfall files cover up to 09-20, '
             'water level/discharge up to 09-20), so all three windows get the full 240 hours.',
             ha='center', fontsize=9.6)
    fig.tight_layout(rect=[0, 0.028, 1, 0.968])
    fig.savefig(os.path.join(FIG, 'fig_window_options_overview.png'), dpi=170)
    pdf.savefig(fig); plt.close(fig)


def fig_window(pdf, w, idx):
    lab = w['labels']
    r = rain_areal.reindex(lab)
    h = WL_all.reindex(lab)
    d = Q_all.reindex(lab)
    s = S[w['key']]
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 8.6), sharex=True)

    ax = axes[0]
    ax.bar(lab, r.values, width=0.038, color='#1f77b4')
    ax.set_ylabel('Areal-mean rainfall\n(mm/h)', fontsize=10)
    ax.set_title('MRMS GaugeCorr QPE 01H areal mean -- %d/240 hours complete  |  '
                 'window total %.0f mm (spin-up %.0f + event %.0f)  |  peak %.2f mm/h @ %s'
                 % (240 - s['rain_hours_missing'], s['rain_total_mm'],
                    s['rain_spinup_mm'], s['rain_event_mm'], s['rain_peak_mm_h'],
                    s['rain_peak_at'][5:16]), fontsize=10.5)

    ax = axes[1]
    ax.fill_between(lab, h['lo'], h['hi'], color='#0057b7', alpha=0.22,
                    label='within-hour min-max')
    ax.plot(lab, h['mean'], color='#0057b7', lw=1.4, label='hourly mean')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_ylabel('Mayport water level\n(m, NAVD88)', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('NOAA 8720218 observed total water level (downstream boundary forcing) -- %d/240 hours complete  |  '
                 'peak %.2f m @ %s, at %.0f%% of the window, %d h left after peak'
                 % (240 - s['wl_hours_missing'], s['wl_peak_m'], s['wl_peak_at'][5:16],
                    s['wl_peak_pos_pct'], s['wl_hours_after_peak']), fontsize=10.5)

    ax = axes[2]
    ax.fill_between(lab, d['lo'], d['hi'], color='#e6194B', alpha=0.20,
                    label='within-hour min-max')
    ax.plot(lab, d['mean'], color='#e6194B', lw=1.4, label='hourly mean')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_ylabel('Acosta discharge\n(m³/s, signed)', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('USGS 02246500 parameter 00060 total discharge, not de-tided (upstream boundary forcing) -- %d/240 hours complete  |  '
                 'peak %.0f m³/s @ %s, at %.0f%% of the window, %d h left after peak'
                 % (240 - s['q_hours_missing'], s['q_peak_m3s'], s['q_peak_at'][5:16],
                    s['q_peak_pos_pct'], s['q_hours_after_peak']), fontsize=10.5)

    for ax in axes:
        ax.axvspan(w['T0'], w['TSPIN'], color='#999', alpha=0.13)
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_xlim(w['T0'], w['T1'])
    y1 = axes[0].get_ylim()[1]
    axes[0].text(w['T0'] + pd.Timedelta('36h'), y1 * 0.82, 'Spin-up 72 h',
                 ha='center', fontsize=9.5, color='#444')
    axes[0].text(w['TSPIN'] + pd.Timedelta('84h'), y1 * 0.82, 'Event period 168 h',
                 ha='center', fontsize=9.5, color='#444')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    axes[2].xaxis.set_major_locator(mdates.DayLocator())
    axes[2].set_xlabel('Time (UTC; hourly label T represents the interval [T-1h, T))', fontsize=10)
    fig.suptitle('Figure %d   Option %s (%s): %s -> %s UTC, 240 hours'
                 % (idx, w['key'], w['name'], str(w['T0'])[:16], str(w['T1'])[:16]),
                 fontsize=14.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    fig.savefig(os.path.join(FIG, 'fig_window_options_%s.png' % w['key']), dpi=170)
    pdf.savefig(fig); plt.close(fig)


def textpage(pdf, title, blocks, foot=''):
    fig = plt.figure(figsize=(13.6, 8.4))
    fig.text(0.06, 0.945, title, fontsize=17, fontweight='bold', va='top')
    y = 0.875
    for kind, txt in blocks:
        if kind == 'h':
            y -= 0.012
            fig.text(0.06, y, txt, fontsize=12.5, fontweight='bold', va='top',
                     color='#1a3f66')
            y -= 0.045
        elif kind == 'p':
            fig.text(0.075, y, txt, fontsize=10.8, va='top', linespacing=1.75)
            y -= 0.030 * (txt.count('\n') + 1) + 0.014
        elif kind == 't':
            rows, nr = txt, len(txt)
            tb = fig.add_axes([0.075, y - 0.031 * nr, 0.865, 0.031 * nr])
            tb.axis('off')
            tab = tb.table(cellText=[r[1:] for r in rows[1:]], colLabels=rows[0][1:],
                           rowLabels=[r[0] for r in rows[1:]], loc='center',
                           cellLoc='left', rowLoc='left')
            tab.auto_set_font_size(False); tab.set_fontsize(9.3); tab.scale(1, 1.42)
            tab.auto_set_column_width(list(range(len(rows[0]) - 1)))
            for (rr, cc), cell in tab.get_celld().items():
                cell.set_edgecolor('#cccccc')
                if rr == 0:
                    cell.set_facecolor('#e8eef5'); cell.set_text_props(fontweight='bold')
                if cc == -1:
                    cell.set_text_props(fontweight='bold')
            y -= 0.031 * nr + 0.035
    if foot:
        fig.text(0.06, 0.075, foot, fontsize=9.5, va='top', color='#555', linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


out = os.path.join(REP, 'Flood_2.0_window_options_comparison.pdf')
with PdfPages(out) as pdf:
    rows = [['', 'W0 current', 'WA shifted +2 d', 'WB shifted +3 d']]
    def row(lbl, f):
        rows.append([lbl] + [f(S[k]) for k in ('W0', 'WA', 'WB')])
    row('Window (UTC)', lambda s: '%s -> %s' % (s['t0'][5:10], s['t1'][5:10]))
    row('Spin-up end', lambda s: s['tspin'][5:16])
    row('Completeness of 3 datasets', lambda s: '%d/240, %d/240, %d/240'
        % (240 - s['rain_hours_missing'], 240 - s['wl_hours_missing'],
           240 - s['q_hours_missing']))
    row('Window total rainfall', lambda s: '%.0f mm' % s['rain_total_mm'])
    row('  of which spin-up', lambda s: '%.0f mm' % s['rain_spinup_mm'])
    row('  of which event period', lambda s: '%.0f mm' % s['rain_event_mm'])
    row('Rainfall peak', lambda s: '%.1f mm/h @ %s' % (s['rain_peak_mm_h'], s['rain_peak_at'][5:16]))
    row('Water-level peak', lambda s: '%.2f m @ %s' % (s['wl_peak_m'], s['wl_peak_at'][5:16]))
    row('  Peak position in window', lambda s: '%.0f %%' % s['wl_peak_pos_pct'])
    row('  Hours left after peak', lambda s: '%d h' % s['wl_hours_after_peak'])
    row('Discharge peak', lambda s: '%.0f m³/s @ %s' % (s['q_peak_m3s'], s['q_peak_at'][5:16]))
    row('  Peak position in window', lambda s: '%.0f %%' % s['q_peak_pos_pct'])
    row('  Hours left after peak', lambda s: '%d h' % s['q_hours_after_peak'])
    row('Rainfall in last 48 h', lambda s: '%.1f mm' % s['rain_tail48_mm'])
    row('Max water level in last 48 h', lambda s: '%.2f m' % s['wl_max_tail48'])
    row('Spin-up rainfall peak', lambda s: '%.1f mm/h' % s['spinup_rain_max_mm_h'])

    textpage(pdf, 'Simulation time window: comparison of three options', [
        ('p', 'Fixed constraint: 72 h spin-up + 168 h event = 240 hourly labels; label T represents [T-1h, T).\n'
              'All three datasets are complete over 09-03 -> 09-16; every option can be filled completely, with no data gaps.'),
        ('t', rows),
    ], foot='All values computed on the fly by 14_window_options.py; no delivered data modified; '
            'machine-readable version in data/meta/window_options.json.')

    fig_overview(pdf)
    for i, w in enumerate(WINDOWS[1:], start=2):
        fig_window(pdf, w, i)
    fig_window(pdf, WINDOWS[0], 4)

print('PDF ->', out)
