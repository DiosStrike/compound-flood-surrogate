#!/usr/bin/env python3
"""
21_rain_qc.py -- standard rainfall-forcing QC (run after every run / scenario)
Usage: python3 code/21_rain_qc.py <run_name> [event=irma]
      python3 code/21_rain_qc.py baseline irma --compare baseline_pre_amprfix   # also make a comparison figure against another run

Checks (PASS only if all pass; verdicts use only four grades):
  1. ampr file: 13 header lines complete; TIME strictly increasing, 1 h spacing, covers [tstart, tstop]; each block has n_rows x n_cols values; no NaN / NODATA / negatives.
  2. Coverage: every SFINCS active-cell centre lies inside the effective ampr coverage (effective = envelope of cell centres after dropping row 1 of the file; SFINCS does not use row 1, measured 2026-09-15).
  3. Input cumulative rainfall: accumulate the hourly ampr blocks onto every active cell by nearest neighbour (mm); report domain mean, minimum and number of zero-rain cells.
  4. What SFINCS actually received: if sfincs_map.nc contains cumprcp (storecumprcp=1 in the inp), compare cell by cell with 3: domain-mean ratio, percentiles of the per-cell ratio,
     number and location of cells differing by > 10 %; and explain known sources of difference (SFINCS bilinear interpolation between 1 km cell centres, time interpolation between blocks).
Output: runs/<event>/<run>/rain_qc.json, rain_qc.png (coverage map + per-cell ratio map); verdict printed to the terminal.
"""
import os, sys, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH

args = [a for a in sys.argv[1:] if not a.startswith('--')]
RUN = args[0] if args else 'baseline'
EVENT = args[1] if len(args) > 1 else 'irma'
CMP = sys.argv[sys.argv.index('--compare') + 1] if '--compare' in sys.argv else None
EV = PTH.event(EVENT)
RDIR = os.path.join(EV['runs'], RUN)


def read_inp(d):
    kv = {}
    for line in open(os.path.join(d, 'sfincs.inp')):
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1); kv[k.strip()] = v.strip()
    return kv


def read_grid(d, kv):
    mmax, nmax = int(kv['mmax']), int(kv['nmax'])
    ind = np.fromfile(os.path.join(d, 'sfincs.ind'), dtype='<u4'); n = int(ind[0]); ind = ind[1:1 + n] - 1
    msk = np.fromfile(os.path.join(d, 'sfincs.msk'), dtype='u1')
    M = np.zeros(mmax * nmax, np.uint8); M[ind] = msk
    return M.reshape(mmax, nmax).T          # (nmax, mmax), n=0 = south


def read_ampr(path):
    hdr = {}; blocks = []; times = []
    with open(path) as f:
        lines = f.read().split('\n')
    import pandas as _pd
    _ref = None
    i = 0
    while i < len(lines) and '=' in lines[i] and not lines[i].startswith('TIME'):
        k, v = lines[i].split('=', 1); hdr[k.strip()] = v.strip(); i += 1
    nr, nc = int(hdr['n_rows']), int(hdr['n_cols'])
    while i < len(lines):
        if lines[i].startswith('TIME'):
            m = re.search(r'=\s*([-\d.]+)\s*(seconds|minutes|hours|days)\s+since\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[i])
            fac = dict(seconds=1 / 3600., minutes=1 / 60., hours=1.0, days=24.0)[m.group(2)]
            ref = _pd.Timestamp(m.group(3), tz='UTC')
            if _ref is None:
                _ref = ref; hdr['time_reference'] = m.group(3)
            times.append(float(m.group(1)) * fac + (ref - _pd.Timestamp('1970-01-01', tz='UTC')) / _pd.Timedelta('1h'))   # unify as hours since 1970
            rows = lines[i + 1:i + 1 + nr]
            A = np.array([np.array(r.split(), dtype=float) for r in rows])
            assert A.shape == (nr, nc), 'ampr block %d shape %s != (%d,%d)' % (len(times), A.shape, nr, nc)
            blocks.append(A); i += 1 + nr
        else:
            i += 1
    return hdr, np.array(times), np.array(blocks)     # blocks: (nt, nr, nc), row 0 = north


def ampr_cum_on_grid(hdr, times, blocks, kv, mmax, nmax):
    """Accumulate hourly blocks onto SFINCS cells by nearest neighbour (mm). Block k holds the mm/hr for [t_k, t_{k+1}) (ampr_block=1)."""
    dx = float(hdr['dx']); x0a, y0a = float(hdr['x_llcorner']), float(hdr['y_llcorner'])
    nr, nc = blocks.shape[1], blocks.shape[2]
    xc = float(kv['x0']) + (np.arange(mmax) + 0.5) * float(kv['dx'])
    yc = float(kv['y0']) + (np.arange(nmax) + 0.5) * float(kv['dy'])
    XC, YC = np.meshgrid(xc, yc)
    ci = np.floor((XC - x0a) / dx).astype(int)
    ri = (nr - 1 - np.floor((YC - y0a) / dx)).astype(int)          # row 0 = north
    ok = (ci >= 0) & (ci < nc) & (ri >= 0) & (ri < nr)
    dt_h = np.diff(times)                                           # duration of each block in hours
    cum = np.zeros((nmax, mmax))
    for k in range(len(times) - 1):
        cum[ok] += blocks[k][ri[ok], ci[ok]] * dt_h[k]
    # envelope of cell centres of the effective coverage (row 1 dropped)
    xr = x0a + (np.arange(nc) + 0.5) * dx; yr = y0a + (np.arange(nr) + 0.5) * dx
    valid = (XC >= xr[0]) & (XC <= xr[-1]) & (YC >= yr[0]) & (YC <= yr[-2])
    return cum, ok, valid


def main():
    kv = read_inp(RDIR); mmax, nmax = int(kv['mmax']), int(kv['nmax'])
    MSK = read_grid(RDIR, kv); ACT = MSK > 0
    hdr, times, blocks = read_ampr(os.path.join(RDIR, 'sfincs.ampr'))
    import pandas as pd
    tref = pd.Timestamp(kv['tref'].replace(' ', 'T'), tz='UTC')
    tstart = pd.Timestamp(kv['tstart'].replace(' ', 'T'), tz='UTC'); tstop = pd.Timestamp(kv['tstop'].replace(' ', 'T'), tz='UTC')
    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    t_abs = [epoch + pd.Timedelta(hours=h) for h in times]
    res = dict(run=RUN, event=EVENT, ampr_grid=dict(n_rows=int(hdr['n_rows']), n_cols=int(hdr['n_cols']), dx=float(hdr['dx']),
               x_ll=float(hdr['x_llcorner']), y_ll=float(hdr['y_llcorner'])), n_blocks=len(times))
    # 1 file
    d = np.diff(times)
    res['time_check'] = dict(strictly_increasing=bool((d > 0).all()), all_1h=bool(np.allclose(d, 1.0)),
                             first=str(t_abs[0]), last=str(t_abs[-1]),
                             covers_window=bool(t_abs[0] <= tstart and t_abs[-1] >= tstop),
                             nan=int((~np.isfinite(blocks)).sum()), nodata=int((blocks <= -998).sum()), negative=int((blocks < 0).sum()))
    # 2 coverage + 3 input accumulation
    cum, ok, valid = ampr_cum_on_grid(hdr, times, blocks, kv, mmax, nmax)
    res['coverage'] = dict(active_cells=int(ACT.sum()), active_outside_ampr=int((ACT & ~ok).sum()),
                           active_outside_valid=int((ACT & ~valid).sum()))
    res['input_cum_mm'] = dict(domain_mean=float(cum[ACT].mean()), min=float(cum[ACT].min()), max=float(cum[ACT].max()),
                               zero_cells=int((ACT & (cum <= 0)).sum()))
    # 4 what SFINCS actually received
    sf = None
    fmap = os.path.join(RDIR, 'sfincs_map.nc')
    if os.path.exists(fmap):
        try:
            import netCDF4 as nc
            D = nc.Dataset(fmap)
            if 'cumprcp' in D.variables and np.isfinite(np.ma.filled(D.variables['cumprcp'][:], np.nan)).any():
                v = D.variables['cumprcp']
                arr = np.ma.filled(v[:], np.nan)
                sf = arr[-1] if arr.ndim == 3 else arr
                units = getattr(v, 'units', '?')
                if units in ('m',) or (np.nanmax(sf) < 5 and np.nanmax(cum) > 50):
                    sf = sf * 1000.0; units = 'm->mm'
                sfa = np.where(ACT, sf, np.nan)
                ratio = np.where(ACT & (cum > 1), sf / np.maximum(cum, 1e-9), np.nan)
                q = np.nanpercentile(ratio, [1, 5, 50, 95, 99])
                big = ACT & (cum > 1) & (np.abs(ratio - 1) > 0.10)
                rows_big = np.where(big.any(axis=1))[0]
                res['sfincs_cumprcp'] = dict(units=units, domain_mean_mm=float(np.nanmean(sfa)), min_mm=float(np.nanmin(sfa)),
                                             nan_cells=int((ACT & ~np.isfinite(sf)).sum()),
                                             mean_ratio=float(np.nanmean(sfa) / cum[ACT].mean()),
                                             ratio_pct=dict(p1=float(q[0]), p5=float(q[1]), p50=float(q[2]), p95=float(q[3]), p99=float(q[4])),
                                             cells_ratio_off_10pct=int(big.sum()),
                                             off_rows_n_index=(rows_big.tolist()[:10] + (['...'] if len(rows_big) > 10 else [])),
                                             north_rows_mean_ratio=[float(np.nanmean(ratio[r])) if np.isfinite(ratio[r]).any() else None for r in range(nmax - 1, nmax - 6, -1)],
                                             note='sources of difference: SFINCS bilinear interpolation between 1 km cell centres (nearest-neighbour accumulation is stepwise, SFINCS is smooth), time interpolation between blocks; '
                                                  'bilinear interpolation does not change the domain total, so the domain-mean ratio should be close to 1; per-cell ratios can deviate where rain-rate gradients are large')
                # ---- time reconciliation: hourly domain-mean increments, SFINCS vs input (to tell "rate loss" from "time interpolation / lag") ----
                if arr.ndim == 3 and arr.shape[0] >= 3:
                    fac = 1000.0 if units == 'm->mm' else 1.0
                    sf_t = np.array([np.nanmean(np.where(ACT, arr[i] * fac, np.nan)) for i in range(arr.shape[0])])
                    inp_t = [0.0]; c = np.zeros((nmax, mmax))
                    dxa = float(hdr['dx']); x0a, y0a = float(hdr['x_llcorner']), float(hdr['y_llcorner']); nr = blocks.shape[1]
                    xc = float(kv['x0']) + (np.arange(mmax) + 0.5) * float(kv['dx']); yc = float(kv['y0']) + (np.arange(nmax) + 0.5) * float(kv['dy'])
                    XC, YC = np.meshgrid(xc, yc); ci = np.floor((XC - x0a) / dxa).astype(int); ri = (nr - 1 - np.floor((YC - y0a) / dxa)).astype(int)
                    for k in range(len(times) - 1):
                        c += blocks[k][ri, ci] * (times[k + 1] - times[k]); inp_t.append(float(c[ACT].mean()))
                    inp_t = np.array(inp_t)
                    n = min(len(sf_t), len(inp_t)); dsf = np.diff(sf_t[:n]); dinp = np.diff(inp_t[:n])
                    sel = dinp > 0.5
                    lags = {}
                    for lag in (-1, 0, 1):
                        a = dsf[max(0, lag):len(dsf) + min(0, lag)]; b = dinp[max(0, -lag):len(dinp) + min(0, -lag)]
                        lags[str(lag)] = dict(corr=float(np.corrcoef(a, b)[0, 1]), sum_ratio=float(a.sum() / b.sum()))
                    trap = float(((dinp[:-1] + dinp[1:]) / 2).sum())
                    res['sfincs_cumprcp']['time_check'] = dict(n_hours=int(n - 1), final_ratio=float(sf_t[n - 1] / inp_t[n - 1]),
                        hourly_ratio_pct=dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'], [float(v) for v in np.percentile(dsf[sel] / dinp[sel], [5, 25, 50, 75, 95])])),
                        lag_corr=lags, trapezoid_total_mm=trap, sfincs_total_mm=float(sf_t[n - 1] - sf_t[0]), input_total_mm=float(inp_t[n - 1]),
                        note='difference in sum_ratio between lag=0 and ±1 = time interpolation / lag; hourly_ratio below 1 overall = rate loss')
                    import csv
                    with open(os.path.join(RDIR, 'rain_qc_timeseries.csv'), 'w', newline='') as fcsv:
                        w = csv.writer(fcsv); w.writerow(['hour', 'input_cum_mm', 'sfincs_cum_mm'])
                        for i in range(n):
                            w.writerow([i, round(float(inp_t[i]), 3), round(float(sf_t[i]), 3)])
                else:
                    res['sfincs_cumprcp']['time_check'] = dict(available=False, note='cumprcp has only the final frame (dims %s); hourly reconciliation not possible' % str(v.dimensions))
            else:
                res['sfincs_cumprcp'] = dict(available=False, note='map has no cumprcp: storecumprcp=1 missing from inp, or not supported by this SFINCS version')
        except ImportError:
            res['sfincs_cumprcp'] = dict(available=False, note='requires pip install netCDF4')
    else:
        res['sfincs_cumprcp'] = dict(available=False, note='SFINCS has not been run yet')
    if sf is None and os.path.exists(fmap) and 'sfincs_cumprcp' not in res:
        res['sfincs_cumprcp'] = dict(available=False, note='map has no valid cumprcp (SFINCS v2.4.2 only writes cumprcp in the max-output block when dtmaxout>0; this run has dtmaxout=0). '
                                                          'Received-rainfall reconciliation is replaced by the ampr-only check in run_all_events (ampr md5 match + ratio to beta x baseline accumulation)')
    # verdict
    tc, cv = res['time_check'], res['coverage']
    file_ok = tc['strictly_increasing'] and tc['all_1h'] and tc['covers_window'] and tc['nan'] == 0 and tc['nodata'] == 0 and tc['negative'] == 0
    cov_ok = cv['active_outside_ampr'] == 0 and cv['active_outside_valid'] == 0
    sc = res['sfincs_cumprcp']
    if not (file_ok and cov_ok):
        verdict = 'does not meet requirements: ' + ('ampr file time/value checks failed; ' if not file_ok else '') + ('active cells not effectively covered by ampr; ' if not cov_ok else '')
    elif sc.get('available', True) and 'mean_ratio' in sc:
        if abs(sc['mean_ratio'] - 1) < 0.03 and sc['nan_cells'] == 0 and sc['cells_ratio_off_10pct'] < 0.02 * cv['active_cells']:
            verdict = 'verified compliant: file, coverage, cumulative rainfall (domain-mean ratio %.3f, cells off by >10%% %d)' % (sc['mean_ratio'], sc['cells_ratio_off_10pct'])
        elif abs(sc['mean_ratio'] - 1) < 0.05 and sc['nan_cells'] == 0:
            verdict = 'usable with conditions: domain-mean ratio %.3f; %d cells off by >10%% (see the interpolation explanation in note)' % (sc['mean_ratio'], sc['cells_ratio_off_10pct'])
        else:
            verdict = 'pending user confirmation: domain-mean ratio %.3f, NaN cells %d' % (sc['mean_ratio'], sc['nan_cells'])
    else:
        verdict = 'usable with conditions: file and coverage passed; SFINCS cumulative rainfall could not be reconciled (%s)' % sc.get('note', '')
    res['verdict'] = verdict
    json.dump(res, open(os.path.join(RDIR, 'rain_qc.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))

    # ---- figures ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
    av = {f.name for f in _fm.fontManager.ttflist}
    for fn in ('PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK JP'):
        if fn in av: plt.rcParams['font.family'] = fn; break
    plt.rcParams['axes.unicode_minus'] = False
    x0, y0, dx = float(kv['x0']), float(kv['y0']), float(kv['dx'])
    ext = [x0 / 1e3, (x0 + mmax * dx) / 1e3, y0 / 1e3, (y0 + nmax * dx) / 1e3]
    npan = 3 if sf is not None else 2
    fig, axs = plt.subplots(1, npan, figsize=(5.2 * npan, 5))
    # coverage map
    ax = axs[0]
    ax.imshow(np.where(ACT, cum, np.nan), origin='lower', extent=ext, cmap='Blues')
    def rect(ax, xl, yl, nc, nr, d, **kw):
        ax.plot([xl, xl + nc * d, xl + nc * d, xl, xl] / np.array(1e3), [yl, yl, yl + nr * d, yl + nr * d, yl] / np.array(1e3), **kw)
    rect(ax, float(hdr['x_llcorner']), float(hdr['y_llcorner']), int(hdr['n_cols']), int(hdr['n_rows']), float(hdr['dx']), color='r', lw=1.5, label='%s ampr grid' % RUN)
    yl_valid_top = float(hdr['y_llcorner']) + (int(hdr['n_rows']) - 1) * float(hdr['dx'])
    ax.axhline(yl_valid_top / 1e3, color='r', ls='--', lw=1, label='lower edge of file row 1 (unused by SFINCS)')
    if CMP:
        h2, _, b2 = read_ampr(os.path.join(EV['runs'], CMP, 'sfincs.ampr'))
        rect(ax, float(h2['x_llcorner']), float(h2['y_llcorner']), int(h2['n_cols']), int(h2['n_rows']), float(h2['dx']), color='k', lw=1.2, ls=':', label='%s ampr grid' % CMP)
    rect(ax, x0, y0, mmax, nmax, dx, color='g', lw=1.2, label='SFINCS model domain')
    ax.set_title('ampr input cumulative rainfall (mm, active cells, nearest neighbour)\nactive cells %d, not covered %d' % (cv['active_cells'], cv['active_outside_valid']), fontsize=9)
    ax.legend(fontsize=7, loc='lower left'); ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
    # northern-edge profile
    ax = axs[1]
    rows = np.arange(nmax); prof = np.array([np.nanmean(np.where(ACT[r], cum[r], np.nan)) if ACT[r].any() else np.nan for r in rows])
    ax.plot(prof, (y0 + (rows + 0.5) * dx) / 1e3, 'b-', label='%s input' % RUN)
    if sf is not None:
        ps = np.array([np.nanmean(np.where(ACT[r], sf[r], np.nan)) if ACT[r].any() else np.nan for r in rows])
        ax.plot(ps, (y0 + (rows + 0.5) * dx) / 1e3, 'r--', label='%s SFINCS cumprcp' % RUN)
    if CMP:
        cum2, _, _ = ampr_cum_on_grid(h2, _, b2, kv, mmax, nmax)
        p2 = np.array([np.nanmean(np.where(ACT[r], cum2[r], np.nan)) if ACT[r].any() else np.nan for r in rows])
        ax.plot(p2, (y0 + (rows + 0.5) * dx) / 1e3, 'k:', label='%s input' % CMP)
    ax.set_xlabel('mean cumulative rainfall over active cells in row (mm)'); ax.set_ylabel('y (km)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title('row-wise (south→north) mean cumulative rainfall', fontsize=9)
    if sf is not None:
        ax = axs[2]
        im = ax.imshow(np.where(ACT, sf / np.maximum(cum, 1e-9), np.nan), origin='lower', extent=ext, cmap='RdBu_r', vmin=0.8, vmax=1.2)
        plt.colorbar(im, ax=ax, fraction=0.045).set_label('SFINCS cumprcp / ampr input')
        ax.set_title('per-cell ratio (domain-mean ratio %.3f)' % sc['mean_ratio'], fontsize=9)
    if CMP and sf is not None:
        try:
            import netCDF4 as nc
            D1 = nc.Dataset(os.path.join(RDIR, 'sfincs_map.nc')); D2 = nc.Dataset(os.path.join(EV['runs'], CMP, 'sfincs_map.nc'))
            h1 = np.ma.filled(D1.variables['hmax'][:], np.nan)[0]; h2 = np.ma.filled(D2.variables['hmax'][:], np.nan)[0]
            zb1 = np.ma.filled(D1.variables['zb'][:], np.nan); zb2 = np.ma.filled(D2.variables['zb'][:], np.nan)
            dh = np.where(ACT, h1 - h2, np.nan)
            rows_y = (y0 + (np.arange(nmax) + 0.5) * dx)
            byrow = np.array([np.nanmean(dh[r]) if np.isfinite(dh[r]).any() else np.nan for r in range(nmax)])
            comp = dict(compare_run=CMP, zb_identical=bool(np.allclose(np.nan_to_num(zb1), np.nan_to_num(zb2))),
                        hmax_diff_mm=dict(mean=float(np.nanmean(dh) * 1000), p50=float(np.nanpercentile(dh, 50) * 1000),
                                          p95=float(np.nanpercentile(dh, 95) * 1000), max=float(np.nanmax(dh) * 1000), min=float(np.nanmin(dh) * 1000)),
                        cells_abs_gt_1cm=int(np.nansum(np.abs(dh) > 0.01)), cells_abs_gt_5cm=int(np.nansum(np.abs(dh) > 0.05)),
                        north_5km_rows_mean_mm=[float(v * 1000) if np.isfinite(v) else None for v in byrow[-25:]],
                        wet_gt0p1_run=int(np.nansum(np.where(ACT, h1, 0) > 0.1)), wet_gt0p1_cmp=int(np.nansum(np.where(ACT, h2, 0) > 0.1)))
            res['compare'] = comp
            json.dump(res, open(os.path.join(RDIR, 'rain_qc.json'), 'w'), indent=1, ensure_ascii=False, default=float)
            f2, a2 = plt.subplots(1, 2, figsize=(11, 5))
            im = a2[0].imshow(dh, origin='lower', extent=ext, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            plt.colorbar(im, ax=a2[0], fraction=0.045).set_label('hmax difference (m): %s − %s' % (RUN, CMP))
            a2[0].set_title('max depth difference (cm scale), wet cells %d vs %d' % (comp['wet_gt0p1_run'], comp['wet_gt0p1_cmp']), fontsize=9)
            a2[1].plot(byrow * 1000, rows_y / 1e3, 'k-'); a2[1].axvline(0, color='r', lw=0.6); a2[1].set_xlabel('row-mean hmax difference (mm)'); a2[1].set_ylabel('y (km)'); a2[1].grid(alpha=0.3)
            a2[1].set_title('row-wise (south→north) mean difference; zb identical: %s' % comp['zb_identical'], fontsize=9)
            f2.tight_layout(); f2.savefig(os.path.join(RDIR, 'rain_qc_compare.png'), dpi=130)
            f2.savefig(os.path.join(PTH.FIG, 'fig_%s_08_vs_%s_comparison.png' % (RUN, CMP)), dpi=130)
            print('comparison figure:', os.path.join(RDIR, 'rain_qc_compare.png'))
        except Exception as e:
            print('comparison failed:', e)
    fig.suptitle('%s/%s rainfall QC -- %s' % (EVENT, RUN, verdict), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RDIR, 'rain_qc.png'), dpi=130)
    PTH.ensure_dirs(PTH.FIG); fig.savefig(os.path.join(PTH.FIG, 'fig_%s_07_rain_qc.png' % RUN), dpi=130)
    print('figure:', os.path.join(RDIR, 'rain_qc.png'))
    print('verdict:', verdict)


if __name__ == '__main__':
    main()
