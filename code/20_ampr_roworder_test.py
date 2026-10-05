#!/usr/bin/env python3
"""
20_ampr_roworder_test.py -- verify the row-order assumption of sfincs.ampr (first row of the file = geographic north).

Small test case fully isolated from the production forcing: 50×50 cells @200 m, flat bed zb=5 m, closed with no boundaries, no infiltration; run 6 h.
A set of variants (runs/_tests/ampr_roworder/<variant>/), ruling out causes one by one:
  T1_precip_uniform   precipfile uniform rain 50 mm/hr × 3 h, zsini=zb      -> does the rain mechanism itself work; expect 0.15 m everywhere
  T2_ampr_allrows     ampr 50 mm/hr everywhere, meteo grid extended by 1 km, zsini=zb -> does ampr reading/coverage work; expect 0.15 m everywhere
  T3_ampr_northrows   ampr rains only in the first 2 rows of the file (= northernmost 2 km after extension, i.e. the northernmost 1 km of the model) -> row-order criterion
  T0_orig             first version: zsini=0 (below bed), meteo grid same extent as model, rain only in file row 1 (already run, zero water, kept for the record)
  T4_ampr_row1_nomargin_zsini5  differs from T0 only in zsini (=zb)      -> isolates the effect of zsini
  T5_ampr_allrows_zsini0        differs from T2 only in zsini (=0)       -> same as above
Criterion (computed automatically by check): row order is judged by position only -- T3 rain only in the north -> verified compliant (first row = north); in the south -> non-compliant (16 must reverse row order).
     Water volume is discussed separately (SFINCS temporal interpolation / spatial bilinear / ignores row 1), see REPORT_CURRENT.pdf. "No errors in the log" does not count as passing.

Usage:
  python3 code/20_ampr_roworder_test.py build          # generate all variants
  bash runs/_tests/ampr_roworder/RUN_TEST.command      # run each in Docker (each < 10 s) + check
  python3 code/20_ampr_roworder_test.py check          # interpretation only
ampr is written as in section 5 of 16_build_sfincs.py (13 header lines + TIME line + row blocks).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH

BASE = os.path.join(PTH.ROOT, 'runs', '_tests', 'ampr_roworder')
MMAX = NMAX = 50; DX = 200.0
X0, Y0 = 440000.0, 3350000.0
EPSG = 26917
ZB = 5.0
DXR = 1000.0
RAIN, RAIN_HOURS, TSTOP_H = 50.0, 3, 6
TREF = '20170906 000000'
LX = MMAX * DX                       # 10 km
AREA_ROW = LX * DXR                  # area of the northernmost 1 km strip
VOL_ROW = RAIN / 1000 * RAIN_HOURS * AREA_ROW      # 1.5e6 m³
VOL_ALL = RAIN / 1000 * RAIN_HOURS * LX * LX       # 1.5e7 m³

VARIANTS = {
    'T1_precip_uniform': dict(mode='precip', zsini=ZB, margin=0, rows='all', expect=VOL_ALL, where='uniform'),
    'T2_ampr_allrows':   dict(mode='ampr', zsini=ZB, margin=1, rows='all', expect=VOL_ALL, where='uniform'),
    'T3_ampr_northrows': dict(mode='ampr', zsini=ZB, margin=1, rows='north2', expect=VOL_ROW, where='north'),
    'T4_ampr_row1_nomargin_zsini5': dict(mode='ampr', zsini=ZB, margin=0, rows='first', expect=VOL_ROW, where='north'),   # differs from T0 only in zsini
    'T5_ampr_allrows_zsini0': dict(mode='ampr', zsini=0.0, margin=1, rows='all', expect=VOL_ALL, where='uniform'),        # differs from T2 only in zsini
}


def write_grid(run):
    MSK = np.ones((NMAX, MMAX), np.uint8); Z = np.full((NMAX, MMAX), ZB, np.float32)
    mt = MSK.T; ind = np.where(mt.flatten() > 0)[0]
    np.array(np.hstack([[len(ind)], ind + 1]), dtype='<u4').tofile(os.path.join(run, 'sfincs.ind'))
    np.asarray(mt.flatten()[ind], dtype='u1').tofile(os.path.join(run, 'sfincs.msk'))
    np.asarray(Z.T.flatten()[ind], dtype='<f4').tofile(os.path.join(run, 'sfincs.dep'))


def write_ampr(run, margin, rows):
    import pandas as pd
    ncol = nrow = int(LX / DXR) + 2 * margin
    xmin, ymin = X0 - margin * DXR, Y0 - margin * DXR
    hours = list(range(0, RAIN_HOURS)) + [RAIN_HOURS, TSTOP_H]
    tref = pd.Timestamp('2017-09-06', tz='UTC')
    with open(os.path.join(run, 'sfincs.ampr'), 'w') as f:
        f.write('FileVersion      = 1.03\n'); f.write('filetype         = meteo_on_equidistant_grid\n')
        f.write('NODATA_value     = -999\n'); f.write('n_cols           = %d\n' % ncol); f.write('n_rows           = %d\n' % nrow)
        f.write('grid_unit        = m\n'); f.write('x_llcorner       = %.1f\n' % xmin); f.write('y_llcorner       = %.1f\n' % ymin)
        f.write('dx               = %.1f\n' % DXR); f.write('dy               = %.1f\n' % DXR)
        f.write('n_quantity       = 1\n'); f.write('quantity1        = precipitation\n'); f.write('unit1            = mm/hr\n')
        for h in hours:
            A = np.zeros((nrow, ncol))
            if h < RAIN_HOURS:
                if rows == 'all':
                    A[:, :] = RAIN
                elif rows == 'north2':
                    A[0:2, :] = RAIN          # first 2 rows of the file = northernmost 2 km after extension (northernmost 1 km inside the model + 1 km extension)
                else:
                    A[0, :] = RAIN
            hrs = (tref - pd.Timestamp('1970-01-01', tz='UTC')) / pd.Timedelta('1h') + h
            f.write('TIME = %.4f hours since 1970-01-01 00:00:00 +00:00\n' % hrs)
            for row in A:
                f.write(' '.join('%.2f' % v for v in row) + '\n')


def write_precip(run):
    with open(os.path.join(run, 'sfincs.precip'), 'w') as f:
        for t, v in ((0, RAIN), (RAIN_HOURS * 3600, 0.0), (TSTOP_H * 3600, 0.0)):
            f.write('%10.1f %8.3f\n' % (t, v))


def build():
    for name, v in VARIANTS.items():
        run = os.path.join(BASE, name); os.makedirs(run, exist_ok=True)
        write_grid(run)
        if v['mode'] == 'ampr':
            write_ampr(run, v['margin'], v['rows'])
        else:
            write_precip(run)
        with open(os.path.join(run, 'sfincs.obs'), 'w') as f:
            f.write("%12.2f %13.2f  '%s'\n" % (X0 + LX / 2, Y0 + LX - 500, 'NORTH'))
            f.write("%12.2f %13.2f  '%s'\n" % (X0 + LX / 2, Y0 + 500, 'SOUTH'))
        inp = [('mmax', MMAX), ('nmax', NMAX), ('dx', DX), ('dy', DX), ('x0', X0), ('y0', Y0),
               ('rotation', 0.0), ('epsg', EPSG), ('latitude', 30.35),
               ('tref', TREF), ('tstart', TREF), ('tstop', '20170906 %02d0000' % TSTOP_H),
               ('tspinup', 0), ('dtout', 3600), ('dthisout', 600), ('dtmaxout', TSTOP_H * 3600),
               ('alpha', 0.5), ('theta', 1.0), ('huthresh', 0.05), ('advection', 1),
               ('manning_land', 0.06), ('manning_sea', 0.02), ('rgh_lev_land', 0.0),
               ('zsini', v['zsini']), ('qinf', 0.0),
               ('inputformat', 'bin'), ('outputformat', 'net'),
               ('depfile', 'sfincs.dep'), ('mskfile', 'sfincs.msk'), ('indexfile', 'sfincs.ind'),
               ('amprfile', 'sfincs.ampr') if v['mode'] == 'ampr' else ('precipfile', 'sfincs.precip'),
               ('obsfile', 'sfincs.obs')]
        with open(os.path.join(run, 'sfincs.inp'), 'w') as f:
            for k, val in inp:
                f.write('%-16s= %s\n' % (k, val))
        print('generated %s  (%s, zsini=%.1f, expected volume %.2e m³, location %s)' % (name, v['mode'], v['zsini'], v['expect'], v['where']))
    cmd = os.path.join(BASE, 'RUN_TEST.command')
    with open(cmd, 'w') as f:
        f.write('#!/bin/bash\n# ampr row-order mini test: run SFINCS for each variant (each < 10 s), then interpret. Double-click or bash this file.\n')
        f.write('cd "$(dirname "$0")"\n')
        for name in VARIANTS:
            f.write('echo "=== %s ==="; (cd %s && docker run --rm --platform linux/amd64 -v "$PWD":/data deltares/sfincs-cpu 2>&1 | tail -3)\n' % (name, name))
        f.write('cd ../../.. && python3 code/20_ampr_roworder_test.py check\necho "Done."\n')
    os.chmod(cmd, 0o755)
    open(os.path.join(BASE, 'README.txt'), 'w').write(__doc__)
    print('Next step: bash runs/_tests/ampr_roworder/RUN_TEST.command')


def read_run(run):
    import netCDF4 as nc
    fmap = os.path.join(run, 'sfincs_map.nc')
    if not os.path.exists(fmap):
        return None
    d = nc.Dataset(fmap)
    zb = np.ma.filled(d.variables['zb'][:], np.nan)
    HH = np.ma.filled(d.variables['h'][:], 0.0); HH = np.where(np.isfinite(HH), HH, 0.0); HH[HH < 0] = 0
    y = np.asarray(d.variables['y'][:])
    ax_y = 0 if np.ptp(y[:, 0]) > np.ptp(y[0, :]) else 1
    yy = y.mean(axis=1 - ax_y)
    tt = np.asarray(d.variables['time'][:])
    his = {}
    fh = os.path.join(run, 'sfincs_his.nc')
    if os.path.exists(fh):
        H = nc.Dataset(fh)
        st = [''.join(s.decode() if isinstance(s, bytes) else str(s) for s in row).strip() for row in H.variables['station_name'][:]]
        ph = np.ma.filled(H.variables['point_h'][:], 0.0)
        for i, s in enumerate(st):
            his[s] = float(ph[-1, i])
        his_series = {s: [round(float(v), 4) for v in ph[:, i]] for i, s in enumerate(st)}
    else:
        his_series = {}
    return dict(HH=HH, yy=yy, ax_y=ax_y, tt=tt, his=his, zb=zb, his_series=his_series)


def check():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
    _avail = {f.name for f in _fm.fontManager.ttflist}
    for _f in ('PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei'):
        if _f in _avail:
            plt.rcParams['font.family'] = _f; break
    plt.rcParams['axes.unicode_minus'] = False
    results = {}
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(3.6 * len(VARIANTS), 4.2))
    for ax, (name, v) in zip(np.atleast_1d(axes), VARIANTS.items()):
        run = os.path.join(BASE, name)
        r = read_run(run)
        if r is None:
            results[name] = dict(verdict='not run (no sfincs_map.nc)'); ax.set_title(name + ' not run'); continue
        HH, yy, ax_y = r['HH'], r['yy'], r['ax_y']
        h = HH[-1]
        prof = h.sum(axis=1 - ax_y)
        vol = float(h.sum() * DX * DX)
        y_north0 = Y0 + LX - DXR
        fn = float(prof[yy >= y_north0].sum() / max(prof.sum(), 1e-9))
        fs = float(prof[yy < Y0 + DXR].sum() / max(prof.sum(), 1e-9))
        per_t = [float(HH[i].sum() * DX * DX) for i in range(HH.shape[0])]
        res = dict(volume_m3=vol, expected_m3=v['expect'], volume_ratio=vol / v['expect'],
                   frac_north_1km=fn, frac_south_1km=fs, wet_cells=int((h > 0.001).sum()),
                   h_max=float(h.max()), volume_by_output_step=per_t, his_final_depth_m=r['his'],
                   his_depth_10min=r['his_series'])
        # Criterion (set 2026-09-15): row order judged by position only; volume reported separately, because SFINCS interpolates precip/ampr in time and ampr bilinearly in space,
        # and ignores row 1 of the ampr file, so the volume differs from the theoretical "blocks × area" value -- see REPORT_CURRENT.pdf.
        if v['mode'] == 'precip':
            tri = RAIN / 1000 * RAIN_HOURS / 2 * LX * LX          # precipfile linear interpolation -> triangular integral
            res['expected_if_linear_interp_m3'] = tri
            res['verdict'] = ('verified compliant: uniform rain wets the whole domain, volume = linear-interpolation integral (ratio %.3f)' % (vol / tri)) if abs(vol / tri - 1) < 0.05 and res['wet_cells'] == MMAX * NMAX else \
                             ('pending user confirmation: volume ratio (triangular) %.3f, wet cells %d' % (vol / tri, res['wet_cells']))
        elif v['where'] == 'uniform':
            res['verdict'] = ('conditionally usable: whole domain wet, volume ratio %.3f (no rain in the 2 northern rows + temporal interpolation at the end)' % res['volume_ratio']) if res['wet_cells'] == MMAX * NMAX else \
                             ('non-compliant: wet cells %d/%d' % (res['wet_cells'], MMAX * NMAX))
        else:
            if fs < 0.01 and fn > 0.5 and vol > 0:
                res['verdict'] = 'verified compliant: rain only in the north -> ampr first row = north (north 1 km share %.2f, south 0)' % fn
            elif fs > 0.5:
                res['verdict'] = 'non-compliant: rain falls in the south -> 16_build_sfincs.py must reverse the ampr rows'
            elif vol == 0:
                res['verdict'] = 'non-compliant: no rain (rain only in file row 1 with no extension -> SFINCS ignores row 1)'
            else:
                res['verdict'] = 'pending user confirmation: north %.2f / south %.2f, volume ratio %.3f' % (fn, fs, res['volume_ratio'])
        results[name] = res
        im = ax.imshow(h if ax_y == 0 else h.T, origin='lower', extent=[0, LX / 1000, 0, LX / 1000], cmap='Blues', vmin=0, vmax=0.2)
        ax.set_title('%s\n%s' % (name, res['verdict']), fontsize=8)
        ax.set_xlabel('x (km)'); ax.set_ylabel('y (km, south→north)')
        plt.colorbar(im, ax=ax, fraction=0.045).set_label('water depth at t = 6 h (m)', fontsize=8)
    json.dump(results, open(os.path.join(BASE, 'result.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(results, indent=1, ensure_ascii=False))
    fig.suptitle('ampr row-order mini test (flat closed domain, zb = 5 m, rain 50 mm/hr × 3 h)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, 'check_ampr_roworder.png'), dpi=130)
    PTH.ensure_dirs(PTH.FIG); fig.savefig(os.path.join(PTH.FIG, 'fig_ampr_roworder_check.png'), dpi=130)
    print('figure: %s' % os.path.join(BASE, 'check_ampr_roworder.png'))


def timeqc():
    """T6/T7: does the temporal interpolation scheme conserve volume. Read cumprcp (last frame) and the 10 min depth at the CENTER point in his, compare with the block totals."""
    import netCDF4 as nc
    out = {}
    names = sys.argv[2:] if len(sys.argv) > 2 else ('T6_uniform_real_series', 'T7_alternating_50_0')
    for name in names:
        run = os.path.join(BASE, name)
        if not os.path.exists(os.path.join(run, 'sfincs_map.nc')):
            out[name] = 'not run'; continue
        ser = np.load(os.path.join(run, 'input_series_mm_hr.npy'))
        D = nc.Dataset(os.path.join(run, 'sfincs_map.nc'))
        cp = D.variables['cumprcp']; arr = np.ma.filled(cp[:], np.nan); last = arr[-1] if arr.ndim == 3 else arr
        u = getattr(cp, 'units', '?'); fac = 1000.0 if (u == 'm' or np.nanmax(last) < 5) else 1.0
        cum_sf = float(np.nanmean(last) * fac)
        H = nc.Dataset(os.path.join(run, 'sfincs_his.nc')); ph = np.ma.filled(H.variables['point_h'][:], 0.0)[:, 0] * 1000
        hourly_sf = np.diff(ph[::6])                       # hourly depth increment in mm (closed flat domain = rainfall in that hour)
        block = ser[:-1]
        n = min(len(hourly_sf), len(block))
        # fit SFINCS hourly amount = a*P_k + b*P_{k+1} + c*P_{k-1}
        Pk = block[:n]; Pn1 = np.append(block[1:], block[-1])[:n]; Pm1 = np.insert(block[:-1], 0, block[0])[:n]
        A = np.vstack([Pk, Pn1, Pm1]).T; coef, *_ = np.linalg.lstsq(A, hourly_sf[:n], rcond=None)
        out[name] = dict(cumprcp_final_mm=cum_sf, his_final_depth_mm=float(ph[-1]), block_total_mm=float(block.sum()),
                         ratio_cumprcp=cum_sf / block.sum(), ratio_his=float(ph[-1]) / block.sum(),
                         fit_hourly_eq='sf_k = %.3f P_k + %.3f P_k+1 + %.3f P_k-1' % tuple(coef), fit_rmse_mm=float(np.sqrt(np.mean((A @ coef - hourly_sf[:n]) ** 2))),
                         first_12_sf=[round(float(x), 2) for x in hourly_sf[:12]], first_12_block=[round(float(x), 2) for x in block[:12]])
    fjs = os.path.join(BASE, 'timeqc_result.json')
    old = json.load(open(fjs)) if os.path.exists(fjs) else {}
    old.update(out); json.dump(old, open(fjs, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    {'build': build, 'check': check, 'timeqc': timeqc}[sys.argv[1] if len(sys.argv) > 1 else 'build']()
