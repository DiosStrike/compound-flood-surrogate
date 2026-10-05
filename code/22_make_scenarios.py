#!/usr/bin/env python3
"""
22_make_scenarios.py -- generate 81 Irma-like compound forcing scenarios (Proposal §4–§5); forcing files only, SFINCS is not run.

Usage: python3 code/22_make_scenarios.py            # generate + QC + manifest + split_A + figures
      python3 code/22_make_scenarios.py --check    # redo QC only (files not rewritten)

Method (Proposal §4.3–4.5, verbatim):
  water level  r = h_obs − h_tide ;  h_s = h_tide + α·r                (Mayport 6 min, NAVD88, UTC)
  discharge  Q_low = 25 h centred moving average(Q); Q_high = Q − Q_low; Q_s = γ·Q_low + Q_high  (Acosta 15 min, signed, not clipped at zero)
        the moving average is the sample mean over the time window |t'−t| ≤ 12.5 h (the raw 15 min series has gaps, not filled; both ends use real data 09-03 -> 09-20)
  rainfall  P_s(x,t) = β·P(x, t−τ), positive τ = delay; τ=+6 uses real MRMS of 09-05 19:00–09-06 00:00, τ=−6 uses real MRMS of 09-16 01:00–06:00
Factors: α∈{0.8,1.0,1.2}, β∈{0.7,1.0,1.3}, γ∈{0.75,1.0,1.25}, τ∈{−6,0,+6}; IDs irma_s001–s081 in lexicographic (α,β,γ,τ) order,
      original Irma = (1,1,1,0) = irma_s041.
Split (§5.1): combined forcing level = sum of α/β/γ scores (low/mid/high = 0/1/2), low 0–2 / mid 3 / high 4–6;
      train 23/14/23, val 4/3/4, test 3/4/3; seed 42; τ balanced round-robin within each set; irma_s041 always in the test set.
Outputs: data/irma/scenarios/irma_sNNN/{sfincs.bzs,sfincs.dis,sfincs.ampr,sfincs.inp,scenario.json}
      data/irma/scenarios/{manifest.csv,split_A.csv,qc_summary.json,discharge_decomposition.csv}
      result/figures/fig_irma_25h_discharge_decomposition.png, fig_irma_scenario_spotcheck.png
"""
import os, sys, json, hashlib, itertools, datetime, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH, geoio, window_config as WCFG

CHECK_ONLY = '--check' in sys.argv
ALPHAS, BETAS, GAMMAS, TAUS = (0.8, 1.0, 1.2), (0.7, 1.0, 1.3), (0.75, 1.0, 1.25), (-6, 0, 6)
SEED = 42
SPLIT_PLAN = {'low': dict(train=23, val=4, test=3), 'mid': dict(train=14, val=3, test=4), 'high': dict(train=23, val=4, test=3)}
MA_HALF_H = 12.5

SC = PTH.IRMA['scenarios']; FORC = PTH.IRMA['forcing']; PROC = PTH.IRMA['processed']; RAW = PTH.IRMA['raw']
BASE = PTH.SFINCS_BASE
T0, T1, TSPIN, LABELS = WCFG.pd_window(pd)
TREF = T0
secs = lambda t: (t - TREF) / pd.Timedelta('1s')
build = json.load(open(os.path.join(BASE, 'sfincs_build.json')))
N_BND = build['boundaries']['n_bnd_points']
SRC_FRAC = [s['frac'] for s in build['boundaries']['sources']]
RG = build['rain']['grid']                       # dx, ncol, nrow, x_ll, y_ll (already padded)
EPSG, ZONE = 26917, 17
os.makedirs(SC, exist_ok=True)
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()


def sid(i):
    return 'irma_s%03d' % i


# ---------------------------------------------------------------- 1. water level
res = pd.read_csv(os.path.join(PROC, 'mayport_8720218_residual_6min_utc_navd88_m.csv'))
res['t'] = pd.to_datetime(res.time_utc, utc=True)
W = res[(res.t >= T0) & (res.t <= T1)].sort_values('t').reset_index(drop=True)
assert len(W) == 2401 and W.residual_m.notna().all() and W.tide_pred_m_navd88.notna().all()
assert (np.diff(W.t.values).astype('timedelta64[s]').astype(int) == 360).all()
TS_WL = secs(W.t).values; H_TIDE = W.tide_pred_m_navd88.values; R_WL = W.residual_m.values; H_OBS = W.water_level_obs_m_navd88.values
assert np.allclose(H_TIDE + R_WL, H_OBS, atol=1e-9)

# ---------------------------------------------------------------- 2. discharge: 25 h centred moving average
q = pd.read_csv(os.path.join(RAW, 'usgs_02246500_00060_raw.csv'))
q['t'] = pd.to_datetime(q.timestamp_utc, utc=True)
q = q[q.valid].dropna(subset=['value_si']).sort_values('t').reset_index(drop=True)
assert q.t.min() <= T0 - pd.Timedelta(hours=MA_HALF_H) and q.t.max() >= T1 + pd.Timedelta(hours=MA_HALF_H), 'not enough extended discharge data at the ends'
tq_all = q.t.values.astype('datetime64[s]').astype('int64'); Q_all = q.value_si.values.astype(float)
half = int(MA_HALF_H * 3600)
lo = np.searchsorted(tq_all, tq_all - half, side='left'); hi = np.searchsorted(tq_all, tq_all + half, side='right')
csum = np.concatenate([[0.0], np.cumsum(Q_all)])
Q_low_all = (csum[hi] - csum[lo]) / (hi - lo)
n_in_win = hi - lo
inwin = (q.t >= T0) & (q.t <= T1)
TQ = secs(q.t[inwin]).values; Q = Q_all[inwin]; Q_LOW = Q_low_all[inwin]; Q_HIGH = Q - Q_LOW; NWIN = n_in_win[inwin]
# cross-check with the production 15 min processed table (same source, identical point by point)
qp = pd.read_csv(os.path.join(PROC, 'acosta_02246500_discharge_15min_utc_m3s.csv')); qp['t'] = pd.to_datetime(qp.time_utc, utc=True)
assert len(qp) == inwin.sum() and np.allclose(qp.discharge_m3s.values, Q)
gaps_q = [(str(q.t[inwin].values[i]), int(d)) for i, d in enumerate(np.diff(q.t[inwin].values).astype('timedelta64[m]').astype(int)) if d != 15]
decomp = dict(n_samples_window=int(inwin.sum()), expected_15min=961, irregular_steps=len(gaps_q),
              samples_per_25h_window=dict(min=int(NWIN.min()), max=int(NWIN.max()), nominal=101),
              reconstruction_max_abs_err=float(np.abs(Q_LOW + Q_HIGH - Q).max()),
              Q_range=[float(Q.min()), float(Q.max())], Q_low_range=[float(Q_LOW.min()), float(Q_LOW.max())],
              Q_high_range=[float(Q_HIGH.min()), float(Q_HIGH.max())], neg_frac_Q=float((Q < 0).mean()),
              Q_low_mean=float(Q_LOW.mean()), Q_mean=float(Q.mean()))
pd.DataFrame(dict(time_utc=q.t[inwin].dt.strftime('%Y-%m-%d %H:%M:%S+00:00').values, Q_m3s=Q, Q_low_m3s=Q_LOW, Q_high_m3s=Q_HIGH, n_samples_25h=NWIN)).to_csv(
    os.path.join(SC, 'discharge_decomposition.csv'), index=False, float_format='%.4f')

# ---------------------------------------------------------------- 3. rainfall: window + 6 h of real MRMS at each end
P_win = np.load(os.path.join(RAW, 'mrms_cache/mrms_P.npy'))          # (240, 40, 57) labels 09-06 01:00 ... 09-16 00:00
assert np.isfinite(P_win).all()
LL = np.load(os.path.join(RAW, 'mrms_cache/mrms_lonlat.npy'))
lon_min, lon_max, lat_min, lat_max, dlon, dlat = LL
nlat, nlon = P_win.shape[1], P_win.shape[2]
lons = lon_min + np.arange(nlon) * abs(dlon); lats = lat_max - np.arange(nlat) * abs(dlat)
PF = {}                                                               # label(Timestamp) -> field
for k, T in enumerate(LABELS):
    PF[T] = np.clip(P_win[k], 0, None)
# first 6 h: decoded cache (from 09-03 01:00, float32, raw mm)
z = np.load(os.path.join(RAW, 'mrms_cache/mrms_raw_cache.npz'), allow_pickle=True)
zt = [pd.Timestamp(datetime.datetime.utcfromtimestamp(x), tz='UTC') for x in z['t']]
for T in [T0 - pd.Timedelta(hours=h) for h in range(5, -1, -1)]:
    i = zt.index(T); v = z['P'][i].astype(float)
    assert not np.isclose(v, -3.0, atol=1e-6).any() and np.isfinite(v).all(), 'MRMS %s has no-coverage/missing cells; not filled' % T
    PF[T] = np.clip(v, 0, None)
# last 6 h: decoded directly from grib2 (reusing the decoder and window cut of 05)
spec = importlib.util.spec_from_file_location('m05', os.path.join(PTH.CODE, '05_mrms_rainfall.py')); m05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m05)
ext = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))['latlon_bbox_nad83']
after = [T1 + pd.Timedelta(hours=h) for h in range(0, 7)]              # includes 09-16 00:00 as a consistency check
dec = {}
for T in after:
    f = os.path.join(m05.RAWDIR, 'GaugeCorr_QPE_01H_00.00_%s.grib2' % T.strftime('%Y%m%d-%H%M%S'))
    assert os.path.exists(f), 'missing raw MRMS file ' + f
    sec = m05.read_grib2_sections(f); g = m05.grid_of(sec[3])
    lon_all = g['lon0'] + np.arange(g['ni']) * g['dlon']; lat_all = g['lat0'] + np.arange(g['nj']) * g['dlat']
    ci = np.where((lon_all >= ext['lonmin']) & (lon_all <= ext['lonmax']))[0]; ri = np.where((lat_all >= ext['latmin']) & (lat_all <= ext['latmax']))[0]
    c0, c1 = max(0, ci[0] - 2), min(g['ni'], ci[-1] + 1 + 2); r0, r1 = max(0, ri[0] - 2), min(g['nj'], ri[-1] + 1 + 2)
    assert m05.ref_time(sec[1]) == T.to_pydatetime().replace(tzinfo=None)
    v, _ = m05.unpack_values(sec); v = v.reshape(g['nj'], g['ni'])[r0:r1, c0:c1]
    assert v.shape == (nlat, nlon)
    assert not np.isclose(v, m05.MISSING, atol=1e-6).any(), 'MRMS %s has no-coverage cells; not filled' % T
    dec[T] = np.clip(v.astype(float), 0, None)
assert np.abs(dec[T1] - PF[T1]).max() < 1e-4, 're-decoded 09-16 00:00 does not match the cache'
for T in after[1:]:
    PF[T] = dec[T]
rain_ext = dict(labels_before=[str(T0 - pd.Timedelta(hours=h)) for h in range(5, -1, -1)], labels_after=[str(T) for T in after[1:]],
                decode_consistency_check_max_abs=float(np.abs(dec[T1] - PF[T1]).max()))
# nearest-neighbour index on the 1 km grid (same as 16_build_sfincs.py)
DXR = float(RG['dx']); ncol, nrow = int(RG['ncol']), int(RG['nrow']); AX0, AY0 = float(RG['x_ll']), float(RG['y_ll'])
xr = AX0 + (np.arange(ncol) + 0.5) * DXR; yr = (AY0 + nrow * DXR) - (np.arange(nrow) + 0.5) * DXR
XR, YR = np.meshgrid(xr, yr); LO, LA = geoio.utm_to_ll(XR.ravel(), YR.ravel(), ZONE)
ci_r = np.round((LO - lons[0]) / abs(dlon)).astype(int); ri_r = np.round((lats[0] - LA) / abs(dlat)).astype(int)
assert ((ci_r >= 0) & (ci_r < nlon) & (ri_r >= 0) & (ri_r < nlat)).all()
T_RAIN = [secs(T - pd.Timedelta('1h')) for T in LABELS] + [secs(T1)]


def ampr_blocks(beta, tau):
    out = []
    for T in LABELS:
        src = T - pd.Timedelta(hours=tau)                                # P_s(T) = β·P(T−τ)
        out.append(beta * PF[src][ri_r, ci_r].reshape(nrow, ncol))
    out.append(out[-1])                                                  # append one record at tstop (same as 16)
    return out


def write_ampr(path, blocks):
    with open(path, 'w') as f:
        f.write('FileVersion      = 1.03\nfiletype         = meteo_on_equidistant_grid\nNODATA_value     = -999\n')
        f.write('n_cols           = %d\nn_rows           = %d\ngrid_unit        = m\n' % (ncol, nrow))
        f.write('x_llcorner       = %.1f\ny_llcorner       = %.1f\ndx               = %.1f\ndy               = %.1f\n' % (AX0, AY0, DXR, DXR))
        f.write('n_quantity       = 1\nquantity1        = precipitation\nunit1            = mm/hr\n')
        for t, A in zip(T_RAIN, blocks):
            f.write('TIME = %.4f hours since %s +00:00\n' % (t / 3600.0, TREF.strftime('%Y-%m-%d %H:%M:%S')))
            for row in A:
                f.write(' '.join('%.2f' % v for v in row) + '\n')


def write_bzs(path, h):
    with open(path, 'w') as f:
        for t, v in zip(TS_WL, h):
            f.write('%10.1f' % t + ''.join(' %8.4f' % v for _ in range(N_BND)) + '\n')


def write_dis(path, Qs):
    with open(path, 'w') as f:
        for t, v in zip(TQ, Qs):
            f.write('%10.1f' % t + ''.join(' %10.3f' % (v * fr) for fr in SRC_FRAC) + '\n')


inp_template = open(os.path.join(BASE, 'sfincs.inp')).read()


def write_inp(path, zsini):
    out = []
    for line in inp_template.split('\n'):
        if line.startswith('zsini'):
            line = '%-16s= %s' % ('zsini', round(float(zsini), 3))
        out.append(line)
    open(path, 'w').write('\n'.join(out))


# ---------------------------------------------------------------- 4. combinations and split
combos = list(itertools.product(ALPHAS, BETAS, GAMMAS, TAUS))
assert len(combos) == 81 and len(set(combos)) == 81
rows = []
for i, (a, b, g, tau) in enumerate(combos, 1):
    score = ALPHAS.index(a) + BETAS.index(b) + GAMMAS.index(g)
    level = 'low' if score <= 2 else ('mid' if score == 3 else 'high')
    rows.append(dict(scenario=sid(i), alpha=a, beta=b, gamma=g, tau_h=tau, score=score, level=level, is_baseline=(a == 1 and b == 1 and g == 1 and tau == 0)))
M = pd.DataFrame(rows)
BASE_ID = M.loc[M.is_baseline, 'scenario'].item()
assert M.level.value_counts().to_dict() == {'low': 30, 'mid': 21, 'high': 30}


def stratified_split(M, seed):
    rng = np.random.default_rng(seed)
    split = {}
    for level, plan in SPLIT_PLAN.items():
        sub = M[M.level == level].copy()
        pool = {tau: list(rng.permutation(sub[sub.tau_h == tau].scenario.values)) for tau in TAUS}
        order = list(rng.permutation(TAUS))                                    # random start of the τ round-robin
        need = dict(test=plan['test'], val=plan['val'])
        if level == 'mid':
            split[BASE_ID] = 'test'; pool[0].remove(BASE_ID); need['test'] -= 1
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


split = stratified_split(M, SEED)
M['split'] = M.scenario.map(split)
counts = M.groupby(['level', 'split']).size().unstack(fill_value=0)
for level, plan in SPLIT_PLAN.items():
    assert counts.loc[level, 'train'] == plan['train'] and counts.loc[level, 'val'] == plan['val'] and counts.loc[level, 'test'] == plan['test'], counts
assert M.split.value_counts().to_dict() == {'train': 60, 'val': 11, 'test': 10}
assert split[BASE_ID] == 'test'
tau_bal = M.groupby(['split', 'tau_h']).size().unstack(fill_value=0)

# ---------------------------------------------------------------- 5. generate + QC
ref_md5 = {k: md5(os.path.join(FORC, 'sfincs.' + k)) for k in ('bzs', 'dis', 'ampr')}
ref_inp = open(os.path.join(BASE, 'sfincs.inp')).read()
qc_all = {}
spin_end = secs(TSPIN)
for _, r in M.iterrows():
    d = os.path.join(SC, r.scenario); os.makedirs(d, exist_ok=True)
    a, b, g, tau = float(r.alpha), float(r.beta), float(r.gamma), int(r.tau_h)
    h_s = H_TIDE + a * R_WL
    Q_s = g * Q_LOW + Q_HIGH
    blocks = ampr_blocks(b, tau)
    zsini = round(float(h_s[0]), 3)
    if not CHECK_ONLY:
        write_bzs(os.path.join(d, 'sfincs.bzs'), h_s); write_dis(os.path.join(d, 'sfincs.dis'), Q_s)
        write_ampr(os.path.join(d, 'sfincs.ampr'), blocks); write_inp(os.path.join(d, 'sfincs.inp'), zsini)
    # ---- QC (generation stage)
    qc = {}
    bz = np.loadtxt(os.path.join(d, 'sfincs.bzs')); di = np.loadtxt(os.path.join(d, 'sfincs.dis'))
    qc['bzs'] = dict(n=int(bz.shape[0]), cols=int(bz.shape[1] - 1), t0=float(bz[0, 0]), t1=float(bz[-1, 0]), nan=int(np.isnan(bz).sum()),
                     dt_all_360s=bool(np.allclose(np.diff(bz[:, 0]), 360)), covers=bool(bz[0, 0] <= 0 and bz[-1, 0] >= secs(T1)),
                     residual_only_scaled=bool(np.allclose(bz[:, 1] - H_TIDE, a * R_WL, atol=6e-5)), h_max=float(bz[:, 1].max()))
    qc['dis'] = dict(n=int(di.shape[0]), cols=int(di.shape[1] - 1), t0=float(di[0, 0]), t1=float(di[-1, 0]), nan=int(np.isnan(di).sum()),
                     covers=bool(di[0, 0] <= 0 and di[-1, 0] >= secs(T1)), frac_sum=float(sum(SRC_FRAC)),
                     low_only_scaled=bool(np.allclose(di[:, 1:].sum(1) - Q_HIGH, g * Q_LOW, atol=2e-3 * len(SRC_FRAC))),
                     sign_flips_vs_original=int((np.sign(Q_s) != np.sign(Q)).sum()), q_min=float(Q_s.min()), q_max=float(Q_s.max()),
                     neg_frac=float((Q_s < 0).mean()), irregular_steps=len(gaps_q))
    # ampr: read back and check scaling and shift
    txt = open(os.path.join(d, 'sfincs.ampr')).read().split('\n')
    tl = [l for l in txt if l.startswith('TIME')]
    hrs = np.array([float(l.split('=')[1].split('hours')[0]) for l in tl])
    nblk = len(tl); body = [l for l in txt if l and not l.startswith('TIME') and '=' not in l]
    A = np.array([np.array(l.split(), float) for l in body]).reshape(nblk, nrow, ncol)
    exp = np.array(blocks)
    ok_shift = np.allclose(A, np.round(exp, 2), atol=0.006)
    # independent check: β scaling = ratio to the τ=0 original block; τ shift = comparison with offset original blocks
    base_blocks = np.array(ampr_blocks(1.0, 0))
    k = tau                                                            # P_s[T] = β·P[T−τ]: block j corresponds to original block j−τ
    if tau >= 0:
        lhs, rhs = A[k:240], b * base_blocks[0:240 - k]
    else:
        lhs, rhs = A[0:240 + k], b * base_blocks[-k:240]
    qc['ampr'] = dict(n_blocks=nblk, nrow=nrow, ncol=ncol, t0_h=float(hrs[0]), t1_h=float(hrs[-1]), dt_all_1h=bool(np.allclose(np.diff(hrs), 1)),
                      covers=bool(hrs[0] <= 0 and hrs[-1] >= 240), nan=int((~np.isfinite(A)).sum()), nodata=int((A <= -998).sum()), negative=int((A < 0).sum()),
                      beta_and_shift_ok=bool(ok_shift and np.allclose(lhs, np.round(rhs, 2), atol=0.006)),
                      total_mm_gridmean=float(A[:240].sum(0).mean()), ratio_to_tau0_beta1=float(A[:240].sum() / max(base_blocks[:240].sum(), 1e-9)),
                      spinup_mm_gridmean=float(A[:72].sum(0).mean()))
    qc['zsini'] = dict(value=zsini, equals_hs_tstart=bool(abs(zsini - h_s[0]) < 5e-4))
    inp_txt = open(os.path.join(d, 'sfincs.inp')).read()
    qc['inp_only_zsini_differs'] = bool([l for l in inp_txt.split('\n') if not l.startswith('zsini')] == [l for l in ref_inp.split('\n') if not l.startswith('zsini')])
    qc['spinup'] = dict(residual_abs_max_m=float(np.abs(a * R_WL[TS_WL <= spin_end]).max()), rain_mm=qc['ampr']['spinup_mm_gridmean'])
    files = {k: md5(os.path.join(d, 'sfincs.' + k)) for k in ('bzs', 'dis', 'ampr', 'inp')}
    if r.is_baseline:
        qc['baseline_identity'] = dict(bzs=files['bzs'] == ref_md5['bzs'], dis=files['dis'] == ref_md5['dis'], ampr=files['ampr'] == ref_md5['ampr'],
                                       inp=inp_txt == ref_inp)
    passed = (qc['bzs']['nan'] == 0 and qc['bzs']['dt_all_360s'] and qc['bzs']['covers'] and qc['bzs']['residual_only_scaled'] and qc['bzs']['n'] == 2401
              and qc['dis']['nan'] == 0 and qc['dis']['covers'] and qc['dis']['low_only_scaled'] and abs(qc['dis']['frac_sum'] - 1) < 1e-6
              and qc['ampr']['nan'] == 0 and qc['ampr']['nodata'] == 0 and qc['ampr']['negative'] == 0 and qc['ampr']['dt_all_1h'] and qc['ampr']['covers']
              and qc['ampr']['n_blocks'] == 241 and qc['ampr']['beta_and_shift_ok'] and qc['zsini']['equals_hs_tstart'] and qc['inp_only_zsini_differs']
              and (all(qc['baseline_identity'].values()) if r.is_baseline else True))
    qc['generation_qc'] = 'PASS' if passed else 'FAIL'
    qc_all[r.scenario] = qc
    meta = dict(scenario=r.scenario, alpha=a, beta=b, gamma=g, tau_h=tau, score=int(r.score), level=r.level, split=r.split, is_baseline=bool(r.is_baseline),
                window=dict(tref=str(TREF), tstart=str(T0), tstop=str(T1), spinup_until=str(TSPIN)), zsini=zsini,
                method=dict(waterlevel='h_s = h_tide + alpha*(h_obs - h_tide)', discharge='Q_s = gamma*Q_low + (Q - Q_low), Q_low = 25 h centered moving mean (time window, available samples)',
                            rain='P_s(x,t) = beta*P(x, t - tau_h)'),
                sources=dict(waterlevel_obs='NOAA CO-OPS 8720218 water_level datum=NAVD units=metric tz=gmt 6min', tide_pred='NOAA CO-OPS 8720218 predictions (data/irma/raw/noaa_8720218_predictions_raw.json)',
                             discharge='USGS NWIS IV 02246500 00060 (data/irma/raw/usgs_02246500_00060_raw.csv)', rain='MRMS GaugeCorr_QPE_01H (data/irma/raw/mrms_cache + grib2 for 09-16 01–06)'),
                static=dict(base_dir='data/static/sfincs_base', dep_md5=md5(os.path.join(BASE, 'sfincs.dep')), msk_md5=md5(os.path.join(BASE, 'sfincs.msk'))),
                md5=files, seed=SEED, generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), qc=qc)
    json.dump(meta, open(os.path.join(d, 'scenario.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    M.loc[M.scenario == r.scenario, ['zsini', 'bzs_md5', 'dis_md5', 'ampr_md5', 'inp_md5', 'generation_qc', 'q_sign_flips', 'rain_total_mm', 'h_max_m', 'q_min', 'q_max']] = \
        [zsini, files['bzs'], files['dis'], files['ampr'], files['inp'], qc['generation_qc'], qc['dis']['sign_flips_vs_original'], qc['ampr']['total_mm_gridmean'], qc['bzs']['h_max'], qc['dis']['q_min'], qc['dis']['q_max']]
    print('%s a=%.2f b=%.2f g=%.2f tau=%+d %-4s %-5s %s' % (r.scenario, a, b, g, tau, r.level, r.split, qc['generation_qc']))

# ---------------------------------------------------------------- 6. manifest / split / summary
M['tstart'] = str(T0); M['tstop'] = str(T1); M['spinup_until'] = str(TSPIN); M['seed'] = SEED
M['forcing_dir'] = ['data/irma/scenarios/' + s for s in M.scenario]
M['sfincs_run_status'] = 'not_run'; M['sfincs_version'] = ''; M['rain_qc_ratio'] = ''
cols = ['scenario', 'alpha', 'beta', 'gamma', 'tau_h', 'score', 'level', 'split', 'is_baseline', 'tstart', 'tstop', 'spinup_until', 'zsini', 'forcing_dir',
        'bzs_md5', 'dis_md5', 'ampr_md5', 'inp_md5', 'generation_qc', 'q_sign_flips', 'q_min', 'q_max', 'h_max_m', 'rain_total_mm', 'seed', 'sfincs_run_status', 'sfincs_version', 'rain_qc_ratio']
M[cols].to_csv(os.path.join(SC, 'manifest.csv'), index=False)
M[['scenario', 'alpha', 'beta', 'gamma', 'tau_h', 'score', 'level', 'split', 'is_baseline']].to_csv(os.path.join(SC, 'split_A.csv'), index=False)
summary = dict(n_scenarios=81, unique_combos=len(set(combos)), baseline=BASE_ID, all_pass=bool((M.generation_qc == 'PASS').all()),
               n_pass=int((M.generation_qc == 'PASS').sum()), level_counts=M.level.value_counts().to_dict(),
               split_counts=counts.to_dict(), tau_balance={s: tau_bal.loc[s].to_dict() for s in tau_bal.index}, seed=SEED,
               discharge_decomposition=decomp, discharge_irregular_steps=gaps_q, rain_extension=rain_ext,
               baseline_identity=qc_all[BASE_ID].get('baseline_identity'), split_md5=md5(os.path.join(SC, 'split_A.csv')),
               total_bytes=int(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SC) for f in fs)))
json.dump(summary, open(os.path.join(SC, 'qc_summary.json'), 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: summary[k] for k in ('n_scenarios', 'baseline', 'all_pass', 'n_pass', 'level_counts', 'split_counts', 'tau_balance', 'baseline_identity', 'total_bytes')}, indent=1, ensure_ascii=False, default=str))

# ---------------------------------------------------------------- 7. figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
av = {f.name for f in _fm.fontManager.ttflist}
for fn in ('PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK JP'):
    if fn in av: plt.rcParams['font.family'] = fn; break
plt.rcParams['axes.unicode_minus'] = False
th = TQ / 3600 / 24
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
ax[0].plot(th, Q, color='#1f77b4', lw=0.7, label='Q raw (signed)'); ax[0].plot(th, Q_LOW, color='#d62728', lw=1.8, label='Q_low = 25 h centred moving average')
ax[0].axhline(0, color='k', lw=0.5); ax[0].legend(fontsize=8, loc='upper right'); ax[0].set_ylabel('m³/s'); ax[0].grid(alpha=0.3)
ax[0].set_title('Acosta 02246500 discharge decomposition (negative = flood-tide reverse flow); reconstruction error max |Q_low+Q_high−Q| = %.1e m³/s' % decomp['reconstruction_max_abs_err'], fontsize=10)
ax[1].plot(th, Q_HIGH, color='#2ca02c', lw=0.7, label='Q_high = Q − Q_low'); ax[1].axhline(0, color='k', lw=0.5); ax[1].legend(fontsize=8, loc='upper right'); ax[1].set_ylabel('m³/s'); ax[1].grid(alpha=0.3)
for g, c in zip(GAMMAS, ('#9467bd', '#1f77b4', '#ff7f0e')):
    ax[2].plot(th, g * Q_LOW + Q_HIGH, lw=0.7, color=c, label='γ = %.2f' % g, alpha=0.9)
ax[2].plot(th, [g * v for g, v in zip([1] * len(Q_LOW), Q_LOW)], color='k', lw=0.8, ls=':', label='Q_low(γ=1)')
ax[2].axvspan(0, 3, color='#ddd', alpha=0.5); ax[2].axhline(0, color='k', lw=0.5); ax[2].legend(fontsize=8, loc='upper right', ncol=4); ax[2].set_ylabel('m³/s'); ax[2].grid(alpha=0.3)
ax[2].set_xlabel('Days since 2017-09-06 00:00 UTC (grey = spin-up)'); ax[2].set_title('Q_s = γ·Q_low + Q_high (only the low-frequency component is scaled)', fontsize=10)
fig.tight_layout(); PTH.ensure_dirs(PTH.FIG); fig.savefig(os.path.join(PTH.FIG, 'fig_irma_25h_discharge_decomposition.png'), dpi=140); plt.close()

# spot check: one low / mid (baseline) / high scenario each
pick = [M[(M.level == 'low') & (M.alpha == 0.8) & (M.beta == 0.7) & (M.gamma == 0.75) & (M.tau_h == 6)].scenario.item(), BASE_ID,
        M[(M.level == 'high') & (M.alpha == 1.2) & (M.beta == 1.3) & (M.gamma == 1.25) & (M.tau_h == -6)].scenario.item()]
fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
tw = TS_WL / 86400
for s, c in zip(pick, ('#9467bd', '#1f77b4', '#d62728')):
    r = M[M.scenario == s].iloc[0]; d = os.path.join(SC, s)
    bz = np.loadtxt(os.path.join(d, 'sfincs.bzs')); di = np.loadtxt(os.path.join(d, 'sfincs.dis'))
    txt = open(os.path.join(d, 'sfincs.ampr')).read().split('\n'); body = [l for l in txt if l and not l.startswith('TIME') and '=' not in l]
    A = np.array([np.array(l.split(), float) for l in body]).reshape(241, nrow, ncol)
    lab = '%s α=%.1f β=%.1f γ=%.2f τ=%+d (%s, %s)' % (s, r.alpha, r.beta, r.gamma, r.tau_h, r.level, r.split)
    ax[0].plot(bz[:, 0] / 86400, bz[:, 1], lw=0.8, color=c, label=lab)
    ax[1].plot(di[:, 0] / 86400, di[:, 1:].sum(1), lw=0.7, color=c, label=lab)
    ax[2].plot(np.arange(241) / 24, A.reshape(241, -1).mean(1), lw=0.9, color=c, label=lab)
ax[0].plot(tw, H_TIDE, color='#888', lw=0.6, ls='--', label='astronomical tide h_tide'); ax[0].set_ylabel('Mayport water level m NAVD88'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
ax[1].set_ylabel('Acosta discharge m³/s'); ax[1].axhline(0, color='k', lw=0.5); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
ax[2].set_ylabel('rainfall grid mean mm/hr'); ax[2].set_xlabel('Days since tstart'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
for a_ in ax: a_.axvspan(0, 3, color='#ddd', alpha=0.4)
fig.suptitle('Spot check: the three forcings of a low / mid (original Irma = %s) / high scenario' % BASE_ID, fontsize=10); fig.tight_layout()
fig.savefig(os.path.join(PTH.FIG, 'fig_irma_scenario_spotcheck.png'), dpi=140); plt.close()
print('figures written:', PTH.FIG)
print('all passed' if summary['all_pass'] else 'some scenarios failed QC, see qc_summary.json / scenario.json')
