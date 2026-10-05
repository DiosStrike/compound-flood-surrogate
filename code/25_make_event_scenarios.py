#!/usr/bin/env python3
"""
25_make_event_scenarios.py -- generate 81 compound forcing scenarios for each of the five events using the confirmed Irma rules (code/22_make_scenarios.py / Proposal §4–§5).
Usage: python3 code/25_make_event_scenarios.py [all | matthew ian milton dorian beryl] [--check]

Rules (identical to 22, no new perturbation added):
  water level  r = h_obs − h_tide ;  h_s = h_tide + α·r          (Mayport 6 min observed − NOAA astronomical tide prediction; NAVD88; UTC)
  discharge  Q_low = 25 h centred moving average (mean of available 15 min samples with |t′−t| ≤ 12.5 h; both ends use the 20-day real data); Q_high = Q − Q_low; Q_s = γ·Q_low + Q_high; signed, not clipped at zero
  rainfall  P_s(x,t) = β·P(x, t−τ); τ=±6 uses the ±6 h outside the window from the 20-day product (incl. flagged interpolated hours); per-cell nearest neighbour onto the 1 km UTM grid
  factors  α{0.8,1,1.2} β{0.7,1,1.3} γ{0.75,1,1.25} τ{−6,0,+6}; 81 = lexicographic order; (1,1,1,0) = s041 = unperturbed baseline; perturbations apply to all 240 h incl. spin-up
  split  same as 22: level = idx(α)+idx(β)+idx(γ); low 0–2 / mid 3 / high 4–6; train 23/14/23, val 4/3/4, test 3/4/3; seed 42; τ round-robin; s041 always in test (independent per event)
Differences from Irma (decided 2026-09-15): tref/tstart/tstop in inp are per event; dtmaxout = 3600 (hourly interval maxima, for the main 7-day max depth); all other inp lines are identical to data/static/sfincs_base/sfincs.inp.
Inputs: data/<event>/processed/window10d/ (24_cut_windows.py) + data/<event>/processed/ (20-day 15 min discharge, 6 min tide prediction) + data/<event>/raw/rain/rain_decoded_cache_*.npz
Outputs: data/<event>/scenarios/<event>_sNNN/{sfincs.bzs,sfincs.dis,sfincs.ampr,sfincs.inp,scenario.json} + manifest.csv + split_A.csv + qc_summary.json + discharge_decomposition.csv
      data/<event>/processed/window10d/mayport_residual_6min_window.csv
"""
import os, sys, json, hashlib, itertools, datetime, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH, geoio
spec = importlib.util.spec_from_file_location('m23', os.path.join(PTH.CODE, '23_collect_events.py')); m23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m23)
spec = importlib.util.spec_from_file_location('m24', os.path.join(PTH.CODE, '24_cut_windows.py')); m24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m24)

CHECK_ONLY = '--check' in sys.argv
ALPHAS, BETAS, GAMMAS, TAUS = (0.8, 1.0, 1.2), (0.7, 1.0, 1.3), (0.75, 1.0, 1.25), (-6, 0, 6)
SEED = 42
SPLIT_PLAN = {'low': dict(train=23, val=4, test=3), 'mid': dict(train=14, val=3, test=4), 'high': dict(train=23, val=4, test=3)}
MA_HALF_H = 12.5
DTMAXOUT = 3600
BASE = PTH.SFINCS_BASE
build = json.load(open(os.path.join(BASE, 'sfincs_build.json')))
N_BND = build['boundaries']['n_bnd_points']; SRC_FRAC = [s['frac'] for s in build['boundaries']['sources']]
RG = build['rain']['grid']; DXR = float(RG['dx']); ncol, nrow = int(RG['ncol']), int(RG['nrow']); AX0, AY0 = float(RG['x_ll']), float(RG['y_ll'])
ZONE = 17
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
inp_template = open(os.path.join(BASE, 'sfincs.inp')).read()
INP_KEYS = ('tref', 'tstart', 'tstop', 'zsini', 'dtmaxout')


def write_inp(path, tref, t0, t1, zsini):
    out = []
    for line in inp_template.split('\n'):
        k = line.split('=')[0].strip() if '=' in line else ''
        if k == 'tref': line = '%-16s= %s' % ('tref', tref.strftime('%Y%m%d %H%M%S'))
        elif k == 'tstart': line = '%-16s= %s' % ('tstart', t0.strftime('%Y%m%d %H%M%S'))
        elif k == 'tstop': line = '%-16s= %s' % ('tstop', t1.strftime('%Y%m%d %H%M%S'))
        elif k == 'zsini': line = '%-16s= %s' % ('zsini', round(float(zsini), 3))
        elif k == 'dtmaxout': line = '%-16s= %d' % ('dtmaxout', DTMAXOUT)
        out.append(line)
    open(path, 'w').write('\n'.join(out))


def stratified_split(M, base_id, seed):
    rng = np.random.default_rng(seed); split = {}
    for level, plan in SPLIT_PLAN.items():
        sub = M[M.level == level].copy()
        pool = {tau: list(rng.permutation(sub[sub.tau_h == tau].scenario.values)) for tau in TAUS}
        order = list(rng.permutation(TAUS)); need = dict(test=plan['test'], val=plan['val'])
        if level == 'mid':
            split[base_id] = 'test'; pool[0].remove(base_id); need['test'] -= 1
        for setname in ('test', 'val'):
            k = 0
            while need[setname] > 0:
                tau = order[k % 3]; k += 1
                if pool[tau]:
                    split[pool[tau].pop()] = setname; need[setname] -= 1
        for tau in TAUS:
            for s in pool[tau]:
                split[s] = 'train'
    return split


def make(ev):
    e = m23.EVENTS[ev]; D = e['dirs']; pro = D['processed']; raw = D['raw']; win = os.path.join(pro, 'window10d'); SC = D['scenarios']; os.makedirs(SC, exist_ok=True)
    Wq = json.load(open(os.path.join(win, 'window_quality_%s.json' % ev)))['window']
    T0 = pd.Timestamp(Wq['T0']); TSPIN = pd.Timestamp(Wq['spinup_end']); T1 = pd.Timestamp(Wq['T1']); TREF = T0
    secs = lambda t: (t - TREF) / pd.Timedelta('1s')
    LABELS = pd.date_range(T0 + pd.Timedelta('1h'), T1, freq='h'); assert len(LABELS) == 240
    sid = lambda i: '%s_s%03d' % (ev, i)
    # ------------------------------------------------ 1. water level: residual = observed − tide prediction
    w6 = pd.read_csv(os.path.join(win, 'mayport_6min_window.csv')); w6['t'] = pd.to_datetime(w6.time_utc, utc=True)
    pr = pd.read_csv(os.path.join(pro, 'mayport_8720218_tide_prediction_6min_utc_navd88_m.csv')); pr['t'] = pd.to_datetime(pr.time_utc, utc=True)
    W = w6.merge(pr[['t', 'tide_prediction_m_navd88']], on='t', how='left').sort_values('t').reset_index(drop=True)
    assert len(W) == 2401 and W.water_level_m_navd88.notna().all() and W.tide_prediction_m_navd88.notna().all(), 'water level or tide prediction has gaps in the window'
    assert (np.diff(W.t.values).astype('timedelta64[s]').astype(int) == 360).all()
    H_OBS = W.water_level_m_navd88.values; H_TIDE = W.tide_prediction_m_navd88.values; R_WL = H_OBS - H_TIDE; TS_WL = secs(W.t).values
    pd.DataFrame(dict(time_utc=W.t, water_level_obs_m_navd88=H_OBS, tide_pred_m_navd88=H_TIDE, residual_m=R_WL, flag_inferred=W.flag_inferred.values)).to_csv(
        os.path.join(win, 'mayport_residual_6min_window.csv'), index=False)
    # ------------------------------------------------ 2. discharge: 25 h centred moving average (20 days of real samples)
    q = pd.read_csv(os.path.join(pro, 'acosta_02246500_discharge_15min_utc_m3s.csv')); q['t'] = pd.to_datetime(q.time_utc, utc=True)
    q = q[~q.record_missing.astype(bool)].dropna(subset=['discharge_m3s']).sort_values('t').reset_index(drop=True)
    assert q.t.min() <= T0 - pd.Timedelta(hours=MA_HALF_H) and q.t.max() >= T1 + pd.Timedelta(hours=MA_HALF_H), 'not enough extended discharge data at the ends'
    tq_all = q.t.values.astype('datetime64[s]').astype('int64'); Q_all = q.discharge_m3s.values.astype(float); half = int(MA_HALF_H * 3600)
    lo = np.searchsorted(tq_all, tq_all - half, side='left'); hi = np.searchsorted(tq_all, tq_all + half, side='right')
    csum = np.concatenate([[0.0], np.cumsum(Q_all)]); Q_low_all = (csum[hi] - csum[lo]) / (hi - lo); n_in_win = hi - lo
    inwin = (q.t >= T0) & (q.t <= T1)
    TQ = secs(q.t[inwin]).values; Q = Q_all[inwin]; Q_LOW = Q_low_all[inwin]; Q_HIGH = Q - Q_LOW; NWIN = n_in_win[inwin]
    assert len(Q) == 961, '15 min samples in window %d ≠ 961' % len(Q)
    gaps_q = [(str(q.t[inwin].values[i]), int(d)) for i, d in enumerate(np.diff(q.t[inwin].values).astype('timedelta64[m]').astype(int)) if d != 15]
    est_flags = q.flag_estimated[inwin].astype(bool).values
    decomp = dict(n_samples_window=int(inwin.sum()), expected_15min=961, irregular_steps=len(gaps_q), samples_per_25h_window=dict(min=int(NWIN.min()), max=int(NWIN.max()), nominal=101),
                  reconstruction_max_abs_err=float(np.abs(Q_LOW + Q_HIGH - Q).max()), Q_range=[float(Q.min()), float(Q.max())], Q_low_range=[float(Q_LOW.min()), float(Q_LOW.max())],
                  Q_high_range=[float(Q_HIGH.min()), float(Q_HIGH.max())], neg_frac_Q=float((Q < 0).mean()), Q_low_mean=float(Q_LOW.mean()), Q_mean=float(Q.mean()),
                  official_estimated_samples=int(est_flags.sum()), Q_low_peak_time=str(q.t[inwin].values[int(np.argmax(Q_LOW))]))
    pd.DataFrame(dict(time_utc=q.t[inwin].dt.strftime('%Y-%m-%d %H:%M:%S+00:00').values, Q_m3s=Q, Q_low_m3s=Q_LOW, Q_high_m3s=Q_HIGH, n_samples_25h=NWIN, flag_estimated=est_flags)).to_csv(
        os.path.join(SC, 'discharge_decomposition.csv'), index=False, float_format='%.4f')
    # ------------------------------------------------ 3. rainfall: 20-day decoded cache -> per-cell interpolation -> window ±6 h
    z = np.load(os.path.join(raw, 'rain', 'rain_decoded_cache_%s.npz' % e['rain']))
    P = z['P'].astype(np.float64); FLAG = z['FLAG']; LAT2, LON2 = z['lat'], z['lon']; zt = pd.to_datetime(z['t'], unit='s', utc=True)
    nT, nlat, nlon = P.shape; miss = ~np.isfinite(P); PFa = P.copy(); KIND = np.zeros_like(FLAG); tt = np.arange(nT, dtype=float)
    for iy in range(nlat):
        for ix in range(nlon):
            mm = miss[:, iy, ix]
            if not mm.any(): continue
            ok = np.where(~mm)[0]; fill = np.interp(tt, tt[ok], P[ok, iy, ix]); inside = mm & (tt >= ok[0]) & (tt <= ok[-1])
            PFa[inside, iy, ix] = fill[inside]; KIND[inside, iy, ix] = np.where(FLAG[inside, iy, ix] == 2, 2, 1); KIND[mm & ~inside, iy, ix] = 3
    need = [T0 - pd.Timedelta(hours=h) for h in range(5, -1, -1)] + list(LABELS) + [T1 + pd.Timedelta(hours=h) for h in range(1, 7)]
    PF = {}; interp_hours_used = []
    for T in need:
        k = zt.get_loc(T); v = PFa[k]
        assert np.isfinite(v).all(), '%s has cells that cannot be bracketed' % T
        PF[T] = np.clip(v, 0, None)
        if (KIND[k] > 0).any(): interp_hours_used.append(str(T))
    # 1 km UTM grid -> nearest neighbour on the native grid
    xr_ = AX0 + (np.arange(ncol) + 0.5) * DXR; yr_ = (AY0 + nrow * DXR) - (np.arange(nrow) + 0.5) * DXR
    XR, YR = np.meshgrid(xr_, yr_); LO, LA = geoio.utm_to_ll(XR.ravel(), YR.ravel(), ZONE)
    if e['rain_src'] == 'mrms':
        lons = LON2[0, :]; lats = LAT2[:, 0]; dlon = abs(lons[1] - lons[0]); dlat = abs(lats[1] - lats[0])
        ci_r = np.round((LO - lons[0]) / dlon).astype(int); ri_r = np.round((lats[0] - LA) / dlat).astype(int)
        assert lats[0] > lats[-1], 'MRMS rows should run north to south'
        assert ((ci_r >= 0) & (ci_r < nlon) & (ri_r >= 0) & (ri_r < nlat)).all(), '1 km grid exceeds the MRMS crop extent'
        nn_note = 'MRMS 0.01° regular lat/lon grid, nearest neighbour by index (same as 22/16)'; nn_maxdist_deg = float(max(dlon, dlat) / 2)
    else:
        cosl = np.cos(np.radians(LA)); d2 = (LAT2.ravel()[None, :] - LA[:, None]) ** 2 + ((LON2.ravel()[None, :] - LO[:, None]) * cosl[:, None]) ** 2
        j = np.argmin(d2, axis=1); ri_r, ci_r = np.unravel_index(j, LAT2.shape); nn_maxdist_deg = float(np.sqrt(d2[np.arange(len(j)), j]).max())
        assert nn_maxdist_deg < 0.045, 'Stage IV nearest-neighbour distance too large %.3f° (grid spacing ~0.043°)' % nn_maxdist_deg
        nn_note = 'Stage IV polar-stereographic 2D lat/lon grid, nearest neighbour by approximate spherical distance between cell centres (exhaustive per cell)'
    T_RAIN = [secs(T - pd.Timedelta('1h')) for T in LABELS] + [secs(T1)]

    def ampr_blocks(beta, tau):
        out = [beta * PF[T - pd.Timedelta(hours=tau)][ri_r, ci_r].reshape(nrow, ncol) for T in LABELS]
        out.append(out[-1]); return out

    def write_ampr(path, blocks):
        with open(path, 'w') as f:
            f.write('FileVersion      = 1.03\nfiletype         = meteo_on_equidistant_grid\nNODATA_value     = -999\n')
            f.write('n_cols           = %d\nn_rows           = %d\ngrid_unit        = m\n' % (ncol, nrow))
            f.write('x_llcorner       = %.1f\ny_llcorner       = %.1f\ndx               = %.1f\ndy               = %.1f\n' % (AX0, AY0, DXR, DXR))
            f.write('n_quantity       = 1\nquantity1        = precipitation\nunit1            = mm/hr\n')
            for t, A in zip(T_RAIN, blocks):
                f.write('TIME = %.4f hours since %s +00:00\n' % (t / 3600.0, TREF.strftime('%Y-%m-%d %H:%M:%S')))
                for row in A: f.write(' '.join('%.2f' % v for v in row) + '\n')

    def write_bzs(path, h):
        with open(path, 'w') as f:
            for t, v in zip(TS_WL, h): f.write('%10.1f' % t + ''.join(' %8.4f' % v for _ in range(N_BND)) + '\n')

    def write_dis(path, Qs):
        with open(path, 'w') as f:
            for t, v in zip(TQ, Qs): f.write('%10.1f' % t + ''.join(' %10.3f' % (v * fr) for fr in SRC_FRAC) + '\n')
    # ------------------------------------------------ 4. combinations and split
    combos = list(itertools.product(ALPHAS, BETAS, GAMMAS, TAUS)); assert len(set(combos)) == 81
    rows = []
    for i, (a, b, g, tau) in enumerate(combos, 1):
        score = ALPHAS.index(a) + BETAS.index(b) + GAMMAS.index(g); level = 'low' if score <= 2 else ('mid' if score == 3 else 'high')
        rows.append(dict(scenario=sid(i), alpha=a, beta=b, gamma=g, tau_h=tau, score=score, level=level, is_baseline=(a == 1 and b == 1 and g == 1 and tau == 0)))
    M = pd.DataFrame(rows); BASE_ID = M.loc[M.is_baseline, 'scenario'].item(); assert BASE_ID == sid(41)
    split = stratified_split(M, BASE_ID, SEED); M['split'] = M.scenario.map(split)
    counts = M.groupby(['level', 'split']).size().unstack(fill_value=0)
    for level, plan in SPLIT_PLAN.items():
        assert counts.loc[level, 'train'] == plan['train'] and counts.loc[level, 'val'] == plan['val'] and counts.loc[level, 'test'] == plan['test'], counts
    assert split[BASE_ID] == 'test'; tau_bal = M.groupby(['split', 'tau_h']).size().unstack(fill_value=0)
    # ------------------------------------------------ 5. generate + QC
    qc_all = {}; spin_end = secs(TSPIN); base_blocks = np.array(ampr_blocks(1.0, 0))
    ref_lines = [l for l in inp_template.split('\n') if l.split('=')[0].strip() not in INP_KEYS]
    for _, r in M.iterrows():
        d = os.path.join(SC, r.scenario); os.makedirs(d, exist_ok=True)
        a, b, g, tau = float(r.alpha), float(r.beta), float(r.gamma), int(r.tau_h)
        h_s = H_TIDE + a * R_WL; Q_s = g * Q_LOW + Q_HIGH; blocks = ampr_blocks(b, tau); zsini = round(float(h_s[0]), 3)
        if not CHECK_ONLY:
            write_bzs(os.path.join(d, 'sfincs.bzs'), h_s); write_dis(os.path.join(d, 'sfincs.dis'), Q_s)
            write_ampr(os.path.join(d, 'sfincs.ampr'), blocks); write_inp(os.path.join(d, 'sfincs.inp'), TREF, T0, T1, zsini)
        qc = {}
        bz = np.loadtxt(os.path.join(d, 'sfincs.bzs')); di = np.loadtxt(os.path.join(d, 'sfincs.dis'))
        qc['bzs'] = dict(n=int(bz.shape[0]), cols=int(bz.shape[1] - 1), t0=float(bz[0, 0]), t1=float(bz[-1, 0]), nan=int(np.isnan(bz).sum()), dt_all_360s=bool(np.allclose(np.diff(bz[:, 0]), 360)),
                         covers=bool(bz[0, 0] <= 0 and bz[-1, 0] >= secs(T1)), residual_only_scaled=bool(np.allclose(bz[:, 1] - H_TIDE, a * R_WL, atol=6e-5)), h_max=float(bz[:, 1].max()),
                         official_est_records=int(W.flag_inferred.astype(bool).sum()))
        qc['dis'] = dict(n=int(di.shape[0]), cols=int(di.shape[1] - 1), t0=float(di[0, 0]), t1=float(di[-1, 0]), nan=int(np.isnan(di).sum()), covers=bool(di[0, 0] <= 0 and di[-1, 0] >= secs(T1)),
                         frac_sum=float(sum(SRC_FRAC)), low_only_scaled=bool(np.allclose(di[:, 1:].sum(1) - Q_HIGH, g * Q_LOW, atol=2e-3 * len(SRC_FRAC))),
                         sign_flips_vs_original=int((np.sign(Q_s) != np.sign(Q)).sum()), q_min=float(Q_s.min()), q_max=float(Q_s.max()), neg_frac=float((Q_s < 0).mean()), irregular_steps=len(gaps_q))
        txt = open(os.path.join(d, 'sfincs.ampr')).read().split('\n'); tl = [l for l in txt if l.startswith('TIME')]
        hrs = np.array([float(l.split('=')[1].split('hours')[0]) for l in tl]); nblk = len(tl)
        body = [l for l in txt if l and not l.startswith('TIME') and '=' not in l]; A = np.array([np.array(l.split(), float) for l in body]).reshape(nblk, nrow, ncol)
        ok_shift = np.allclose(A, np.array(blocks), atol=0.0101)   # written precision 0.01 mm (Stage IV raw is 0.01 mm; rounding-boundary difference ≤ 0.01)
        if tau >= 0: lhs, rhs = A[tau:240], b * base_blocks[0:240 - tau]
        else: lhs, rhs = A[0:240 + tau], b * base_blocks[-tau:240]
        qc['ampr'] = dict(n_blocks=nblk, nrow=nrow, ncol=ncol, t0_h=float(hrs[0]), t1_h=float(hrs[-1]), dt_all_1h=bool(np.allclose(np.diff(hrs), 1)), covers=bool(hrs[0] <= 0 and hrs[-1] >= 240),
                          nan=int((~np.isfinite(A)).sum()), nodata=int((A <= -998).sum()), negative=int((A < 0).sum()), beta_and_shift_ok=bool(ok_shift and np.allclose(lhs, rhs, atol=0.0101)),
                          total_mm_gridmean=float(A[:240].sum(0).mean()), ratio_to_tau0_beta1=float(A[:240].sum() / max(base_blocks[:240].sum(), 1e-9)), spinup_mm_gridmean=float(A[:72].sum(0).mean()),
                          event_mm_gridmean=float(A[72:240].sum(0).mean()))
        qc['zsini'] = dict(value=zsini, equals_hs_tstart=bool(abs(zsini - h_s[0]) < 5e-4))
        inp_txt = open(os.path.join(d, 'sfincs.inp')).read(); ip = {l.split('=')[0].strip(): l.split('=', 1)[1].strip() for l in inp_txt.split('\n') if '=' in l and not l.startswith('#')}
        qc['inp'] = dict(only_window_keys_differ=bool([l for l in inp_txt.split('\n') if l.split('=')[0].strip() not in INP_KEYS] == ref_lines),
                         tref=ip['tref'], tstart=ip['tstart'], tstop=ip['tstop'], dtmaxout=int(ip['dtmaxout']), manning_land=ip['manning_land'], manning_sea=ip['manning_sea'])
        qc['spinup'] = dict(residual_abs_max_m=float(np.abs(a * R_WL[TS_WL <= spin_end]).max()), rain_mm=qc['ampr']['spinup_mm_gridmean'])
        files = {k: md5(os.path.join(d, 'sfincs.' + k)) for k in ('bzs', 'dis', 'ampr', 'inp')}
        if r.is_baseline:
            qc['baseline_identity'] = dict(bzs_equals_obs=bool(np.allclose(bz[:, 1], H_OBS, atol=6e-5)), dis_equals_Q=bool(np.allclose(di[:, 1:].sum(1), Q, atol=2e-3 * len(SRC_FRAC))),
                                           ampr_equals_P=bool(np.allclose(A[:240], base_blocks[:240], atol=0.0101)))
        passed = (qc['bzs']['nan'] == 0 and qc['bzs']['dt_all_360s'] and qc['bzs']['covers'] and qc['bzs']['residual_only_scaled'] and qc['bzs']['n'] == 2401
                  and qc['dis']['nan'] == 0 and qc['dis']['covers'] and qc['dis']['low_only_scaled'] and abs(qc['dis']['frac_sum'] - 1) < 1e-6 and qc['dis']['n'] == 961
                  and qc['ampr']['nan'] == 0 and qc['ampr']['nodata'] == 0 and qc['ampr']['negative'] == 0 and qc['ampr']['dt_all_1h'] and qc['ampr']['covers'] and qc['ampr']['n_blocks'] == 241
                  and qc['ampr']['beta_and_shift_ok'] and qc['zsini']['equals_hs_tstart'] and qc['inp']['only_window_keys_differ'] and qc['inp']['dtmaxout'] == DTMAXOUT
                  and qc['inp']['manning_land'] == '0.06' and qc['inp']['manning_sea'] == '0.02' and (all(qc['baseline_identity'].values()) if r.is_baseline else True))
        qc['generation_qc'] = 'PASS' if passed else 'FAIL'; qc_all[r.scenario] = qc
        meta = dict(event=ev, scenario=r.scenario, alpha=a, beta=b, gamma=g, tau_h=tau, score=int(r.score), level=r.level, split=r.split, is_baseline=bool(r.is_baseline),
                    window=dict(tref=str(TREF), tstart=str(T0), tstop=str(T1), spinup_until=str(TSPIN)), zsini=zsini,
                    method=dict(waterlevel='h_s = h_tide + alpha*(h_obs - h_tide)', discharge='Q_s = gamma*Q_low + (Q - Q_low), Q_low = 25 h centered moving mean (available samples)', rain='P_s(x,t) = beta*P(x, t - tau_h)'),
                    sources=dict(waterlevel_obs='NOAA CO-OPS 8720218 water_level (data/%s/raw)' % ev, tide_pred='NOAA CO-OPS 8720218 predictions', discharge='USGS NWIS IV 02246500 00060 (data/%s/raw)' % ev,
                                 rain='%s via IEM (data/%s/raw/rain), 20-day product incl. flagged interpolated hours' % (e['rain'], ev), rain_nearest_neighbour=nn_note, rain_nn_maxdist_deg=nn_maxdist_deg,
                                 rain_interp_hours_used=interp_hours_used),
                    static=dict(base_dir='data/static/sfincs_base', dep_md5=md5(os.path.join(BASE, 'sfincs.dep')), msk_md5=md5(os.path.join(BASE, 'sfincs.msk'))), md5=files, seed=SEED,
                    generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), qc=qc)
        json.dump(meta, open(os.path.join(d, 'scenario.json'), 'w'), indent=1, ensure_ascii=False, default=float)
        M.loc[M.scenario == r.scenario, ['zsini', 'bzs_md5', 'dis_md5', 'ampr_md5', 'inp_md5', 'generation_qc', 'q_sign_flips', 'rain_total_mm', 'h_max_m', 'q_min', 'q_max']] = \
            [zsini, files['bzs'], files['dis'], files['ampr'], files['inp'], qc['generation_qc'], qc['dis']['sign_flips_vs_original'], qc['ampr']['total_mm_gridmean'], qc['bzs']['h_max'], qc['dis']['q_min'], qc['dis']['q_max']]
    # ------------------------------------------------ 6. manifest / split / summary
    M['tstart'] = str(T0); M['tstop'] = str(T1); M['spinup_until'] = str(TSPIN); M['seed'] = SEED
    M['forcing_dir'] = ['data/%s/scenarios/' % ev + s for s in M.scenario]; M['sfincs_run_status'] = 'not_run'; M['sfincs_version'] = ''; M['rain_qc_ratio'] = ''
    cols = ['scenario', 'alpha', 'beta', 'gamma', 'tau_h', 'score', 'level', 'split', 'is_baseline', 'tstart', 'tstop', 'spinup_until', 'zsini', 'forcing_dir',
            'bzs_md5', 'dis_md5', 'ampr_md5', 'inp_md5', 'generation_qc', 'q_sign_flips', 'q_min', 'q_max', 'h_max_m', 'rain_total_mm', 'seed', 'sfincs_run_status', 'sfincs_version', 'rain_qc_ratio']
    M[cols].to_csv(os.path.join(SC, 'manifest.csv'), index=False)
    M[['scenario', 'alpha', 'beta', 'gamma', 'tau_h', 'score', 'level', 'split', 'is_baseline']].to_csv(os.path.join(SC, 'split_A.csv'), index=False)
    summary = dict(event=ev, n_scenarios=81, baseline=BASE_ID, all_pass=bool((M.generation_qc == 'PASS').all()), n_pass=int((M.generation_qc == 'PASS').sum()),
                   window=dict(tref=str(TREF), tstart=str(T0), tstop=str(T1), spinup_until=str(TSPIN)), dtmaxout=DTMAXOUT, level_counts=M.level.value_counts().to_dict(),
                   split_counts=counts.to_dict(), tau_balance={s: tau_bal.loc[s].to_dict() for s in tau_bal.index}, seed=SEED, discharge_decomposition=decomp, discharge_irregular_steps=gaps_q,
                   rain=dict(product=e['rain'], nearest_neighbour=nn_note, nn_maxdist_deg=nn_maxdist_deg, interp_hours_used_in_window_pm6h=interp_hours_used,
                             baseline_total_mm_gridmean=float(base_blocks[:240].sum(0).mean()), baseline_event_mm_gridmean=float(base_blocks[72:240].sum(0).mean())),
                   waterlevel=dict(residual_max_m=float(R_WL.max()), residual_min_m=float(R_WL.min()), official_est_records=int(W.flag_inferred.astype(bool).sum())),
                   baseline_identity=qc_all[BASE_ID].get('baseline_identity'), split_md5=md5(os.path.join(SC, 'split_A.csv')),
                   total_bytes=int(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SC) for f in fs)), generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
    json.dump(summary, open(os.path.join(SC, 'qc_summary.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    print(ev, json.dumps({k: summary[k] for k in ('n_pass', 'all_pass', 'baseline_identity', 'level_counts', 'split_counts', 'total_bytes')}, ensure_ascii=False, default=str))
    print('   rain nn maxdist %.4f°  interp hours used %s  Q_low peak %s  residual max %.3f' % (nn_maxdist_deg, interp_hours_used, decomp['Q_low_peak_time'], R_WL.max()))
    return summary


if __name__ == '__main__':
    evs = [a for a in sys.argv[1:] if not a.startswith('--')] or ['all']
    if evs == ['all']: evs = list(m24.WINDOWS)
    for ev in evs: make(ev)
