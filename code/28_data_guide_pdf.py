#!/usr/bin/env python3
"""28_data_guide_pdf.py -- render data_guide_486.json + data_completeness_486.csv into a concise data-description PDF (result/reports/DATA_GUIDE_486runs.pdf)."""
import os, json, datetime, textwrap
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, 'data', 'static', 'meta', 'data_guide_486.json')))
df = pd.read_csv(os.path.join(ROOT, 'result', 'manifests', 'data_completeness_486.csv'))
W = {w['event']: w for w in S['windows']}
EVS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']

def wrap(ln, width):
    out, cur, w = [], '', 0; ind = len(ln) - len(ln.lstrip(' '))
    for ch in ln:
        cw = 1.0 if ord(ch) > 0x2e80 else 0.55
        if w + cw > width and cur.strip(): out.append(cur); cur = ' ' * (ind + 2); w = (ind + 2) * 0.55
        cur += ch; w += cw
    out.append(cur); return out
def pages(pdf, title, lines, fs=8.6, mono=False):
    width = 0.86 * 8.27 * 72 / fs; y = None; fig = None
    def new():
        f = plt.figure(figsize=(8.27, 11.69)); f.text(.07, .95, title, fontsize=13, weight='bold'); return f, .915
    fig, y = new()
    for ln in lines:
        for sub in (wrap(ln, width) if ln else ['']):
            if y < .05: pdf.savefig(fig); plt.close(fig); fig, y = new()
            fig.text(.07, y, sub, fontsize=fs, va='top', family='monospace' if mono else None); y -= 0.0165
    pdf.savefig(fig); plt.close(fig)

hm = S['his']; A = S['ampr']
L = []
L += ['Generated: %s. Scope: six hurricanes x 81 Irma-like compound-forcing scenarios = 486 SFINCS simulations (v2.4.2-alpha, 200 m grid). This guide only describes existing files; nothing packed, normalized, split or rerun.' % S['generated'], '',
      '1. Storage location',
      '  runs/<event>/<event>_sNNN/ (event = matthew / ian / milton / dorian / beryl; Irma in runs/irma/rerun_dtmax3600/irma_sNNN/; NNN = 001...081). Each directory carries all of its inputs and is independently reproducible:',
      '    sfincs_map.nc   hourly full-domain fields (main output, ~15 MB; the 6 s041 runs are the dtmaxout=3600 version, ~33-56 MB)',
      '    sfincs_his.nc   10-min time series at 9 observation points',
      '    sfincs.bzs / sfincs.dis / sfincs.ampr   boundary water level / discharge / rainfall forcing of this scenario (byte-identical to data/<event>/scenarios/<sid>/, md5 verified)',
      '    sfincs.inp sfincs.dep sfincs.msk sfincs.ind sfincs.bnd sfincs.src sfincs.obs   settings and static grid (shared by all six events, data/static/sfincs_base/)',
      '    sfincs.log run_status.json rain_qc.json event7d/   log, verification record, rainfall QC, npy of hourly-sampled max depth over the 7-day main period',
      '  Scenario parameters: data/<event>/scenarios/manifest.csv (alpha, beta, gamma, tau_h, split, tstart, tstop, spinup_until, md5 ...) and scenario.json in each directory. Master ledger runs/run_ledger.csv.',
      '  Numbering: s001-s081 in lexicographic order of (α, β, γ, τ), α∈{0.8,1,1.2} (tidal residual) β∈{0.7,1,1.3} (rainfall) γ∈{0.75,1,1.25} (low-frequency discharge) τ∈{−6,0,+6} h (rainfall time shift); s041 = unperturbed baseline.', '',
      '2. Time window: spin-up and main period',
      '  Each event spans 10 days = 240 h, tref = tstart; t = 0 ... 240 h. The first 72 h (t ≤ 72, snapshots 0-72, 73 in total) are spin-up; t = 73 ... 240 h (168 snapshots) is the 7-day main period. 241 hourly snapshots (incl. the t = 0 initial field).',]
for e in EVS: L.append('    %-8s %s   spin-up until %s' % (e, W[e]['window'], W[e]['spinup_until']))
L += ['  Forcing files (bzs/dis/ampr) cover the whole 0-240 h; perturbations apply to all 240 h (incl. spin-up).', '',
      '3. Output variables (sfincs_map.nc, dims time(241) x n(195) x m(255); n rows = south->north, m columns = west->east; x/y are cell centers, EPSG:26917)']
for r in S['map_vars']:
    if r['name'] in ('inp', 'crs', 'sfincsgrid', 'corner_x', 'corner_y', 'total_runtime', 'average_dt', 'status'): continue
    note = {'x': 'cell center x, m', 'y': 'cell center y, m', 'msk': 'mask: 0 inactive (NaN area) / 1 active / 2 water-level boundary (95 cells); 24467 active cells', 'zb': 'bed elevation, m NAVD88; no NaN on active cells',
            'time': 'seconds since tstart; 0, 3600, ..., 864000 (hourly, 241 in total)', 'zs': 'water level, m NAVD88; values only on wet cells (h > 0.05 m), dry cells are _FillValue (NaN when read) -- not missing data', 'h': 'water depth, m; all active cells have values (incl. the 0-0.05 m film on dry cells); h = zs − zb holds exactly on wet cells. This is the direct source of hourly depth',
            'cumprcp': 'cumulative rainfall m (timemax block); all NaN in the 480 dtmaxout=0 runs (SFINCS does not write it), values only in the 6 s041 runs'}.get(r['name'], r['long_name'])
    L.append('    %-9s %-28s %-3s %s' % (r['name'], r['dims'], r['units'], note))
L += ['  The 6 s041 runs (dtmaxout=3600) also have zsmax / hmax / timemax (1 h interval maxima); the other 480 runs do not; no cross-setting comparison is made.',
      '  sfincs_his.nc: time(1441, every 600 s) x stations(9); point_zs / point_h (m), point_zb, station_id / name / x / y. 2 of the 9 stations lie outside the domain (NaN throughout, 1441 x 2 = 2882, expected); the other 7 are complete.', '',
      '4. Corresponding forcing (same directory; time in seconds since tstart, same zero as map)',
      '    sfincs.bzs   2401 rows x 12 columns, every 360 s (6 min), m NAVD88; the 12 bnd points (sfincs.bnd) share one Mayport series h_tide + α·(h_obs − h_tide)',
      '    sfincs.dis   961 rows x 4 columns (Irma 868 rows, see 6), every 900 s (15 min), m³/s signed (negative = flood-tide reversal); 4 source cells (sfincs.src) split by cross-section, sum = γ·Q_low + Q_high of Acosta 02246500',
      '    sfincs.ampr  241 time blocks, TIME unit hours since tref (0 ... 240); 1 km equidistant grid 42 rows x 53 columns (row 0 = north), x_ll 412400 / y_ll 3336200, mm/h; β·P(x, t − τ); SFINCS bilinearly interpolates between cell centers to 200 m',
      '  Label convention: forcing hourly value T represents the interval [T−1h, T); map snapshot t is an instantaneous field. Rainfall block TIME = k holds the rain of hour (k−1, k]; SFINCS interpolates linearly between blocks.', '',
      '5. How to read (code/read_run.py, read-only)',
      '    import sys; sys.path.insert(0, "code"); from read_run import load_run',
      '    r = load_run("beryl", 3)                      # or load_run("beryl", "beryl_s003")',
      '    h = r["h"]            # (241, 195, 255) hourly depth m, NaN on inactive cells; r["is_event"] (241,) bool, t > 72 is the main period',
      '    h_event = h[r["is_event"]]                    # (168, 195, 255) 7-day main period',
      '    r["zs"], r["zb"], r["msk"], r["dry"]          # water level (NaN on dry cells), bed elevation, mask, dry-cell boolean',
      '    r["rain"]  (241, 42, 53) mm/h on the original 1 km grid; r["rain_on_model"] (241, 195, 255) nearest neighbour to 200 m (alignment sample only, not identical to SFINCS internal bilinear)',
      '    r["bzs"] (2401, 12), r["bzs_t_s"]; r["dis"] (961, 4), r["dis_t_s"]; r["rain_t_h"] (241,); r["time_utc"] (241,) pandas timestamps',
      '    r["scenario"]         # manifest row: alpha beta gamma tau_h split ...',
      '  Aligning to hour k (k = 0 ... 240): h[k]; rainfall rain[k] (block k = 0 is the 0 h initial value); boundary water level bzs[bzs_t_s == 3600 k]; discharge dis[dis_t_s == 3600 k]. Requires netCDF4 (already on the Mac).',
      '  Using netCDF4 directly: ds = netCDF4.Dataset(path); h = ds["h"][:] is a masked array, ds["h"][:].filled(nan). Do not use zs to detect missing data.', '',
      '6. Missing-data check (all 486 runs scanned, result/manifests/data_completeness_486.csv)',
      '  · map: 241 snapshots, 3600 s interval, last value 240 h -- 486/486; no NaN in zb on active cells; no NaN in h on active cells; no valid values on inactive cells; 24467 active cells, 95 boundary cells -- identical in 486/486.',
      '  · NaN in zs = dry cells (about 17-21 thousand cells per snapshot), corresponding exactly to h ≤ 0.05 m; h − (zs − zb) = 0 on wet cells; not missing data.',
      '  · his: 1441 steps x 9 stations -- 486/486; NaN only at the 2 out-of-domain stations.',
      '  · bzs 2401 rows / ampr 241 blocks -- 486/486, no NaN, no negative rain, all active cells within ampr coverage; dis 961 rows (Irma 868 rows), no NaN.',
      '  · Raw observation gaps (before forcing generation, over the 10-day window):',
      '      Irma    USGS 02246500 has a 30 min interval from 2017-09-08 19:45 to 09-12 15:45 (93 missing 15 min samples, spanning the end of spin-up and the first 3.6 days of the main period); not filled, the dis file passes the 30 min steps directly to SFINCS (linear interpolation internally), hence 868 rows. MRMS 240/240 hours complete. Mayport 2401/2401, of which 39 hours contain official inferred records, 0 in the main period.',
      '      Matthew native water level / discharge coverage 1.0; officially inferred water-level hours: 2 in spin-up, 1 in the main period; no rainfall gaps.',
      '      Ian     3 rainfall file-hours missing, linearly interpolated per cell: 09-24 06Z, 09-25 21Z (spin-up), 09-28 16Z (main period); water level / discharge 1.0.',
      '      Milton / Dorian / Beryl   native water level / discharge coverage 1.0, no rainfall gaps.',
      '    These are the only non-native observations: the Irma 30 min discharge segment (not filled), the 3 interpolated Ian rain hours, and a few NOAA inferred water-level records. Everything else is native.',
      '  Conclusion: no missing data in the 486 simulation outputs; no missing data in the forcing inputs; raw observation gaps as listed above, checked and compliant (the Irma 30 min discharge segment and the 1 interpolated Ian rain hour in the main period are "usable with conditions" and should be known when used as training data).', '',
      '7. Disk and volume: map 15.5 MB x 480 + 33-56 MB x 6; runs/ total 11.1 GB. All original files kept, no conversion performed.']
os.makedirs(os.path.join(ROOT, 'result', 'reports'), exist_ok=True)
with PdfPages(os.path.join(ROOT, 'result', 'reports', 'DATA_GUIDE_486runs.pdf')) as pdf:
    pages(pdf, 'Flood_2.0 -- data description of 486 SFINCS simulations (2026-09-16)', L, fs=8.2)
    # illustration: h snapshot of one scenario + three forcings
    import netCDF4 as nc, sys; sys.path.insert(0, os.path.join(ROOT, 'code')); from read_run import load_run
    r = load_run('beryl', 41)
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(.07, .95, 'Example: beryl_s041 -- hourly depth and corresponding forcing (grey = spin-up 0-72 h, white = main period 73-240 h)', fontsize=11, weight='bold')
    ax = fig.add_axes([.07, .62, .86, .3]); im = ax.imshow(np.where(r['h'][150] > 0.05, r['h'][150], np.nan), origin='lower', cmap='Blues', vmin=0, vmax=3); ax.set_title('h snapshot t = 150 h (wet cells h > 0.05 m), m', fontsize=9); plt.colorbar(im, ax=ax, fraction=.03)
    wet = np.nanmean(r['h'] > 0.05, axis=(1, 2)) * 100
    for k, (yy, lab, tt) in enumerate([(wet, 'Wet-cell fraction % (h > 0.05 m, active cells)', r['t_h']), (np.nanmean(r['rain'], axis=(1, 2)), 'Domain-mean rainfall mm/h (ampr)', r['rain_t_h']), (r['bzs'][:, 0], 'Boundary water level m NAVD88 (bzs)', r['bzs_t_s'] / 3600), (r['dis'].sum(1), 'Total discharge m³/s (dis)', r['dis_t_s'] / 3600)]):
        ax = fig.add_axes([.07, .47 - .12 * k, .86, .1]); ax.axvspan(0, 72, color='#ddd'); ax.plot(tt, yy, lw=.8, color='#1f5fa8'); ax.set_xlim(0, 240); ax.set_ylabel(lab, fontsize=7); ax.tick_params(labelsize=7)
        if k < 3: ax.set_xticklabels([])
    ax.set_xlabel('t (h since tstart)', fontsize=8); pdf.savefig(fig); plt.close(fig)
print('ok')
