#!/usr/bin/env python3
"""
run_all_events.py -- batch controller for multi-event x 81-scenario SFINCS runs (runs on the user's Mac; needs Docker and python3 + netCDF4/xarray).

Usage (from the project root):
  python3 run_all_events.py --dry-run                              # only list the runs to do and missing inputs, do not run
  python3 run_all_events.py                                        # all events x all scenarios (only those not yet verified)
  python3 run_all_events.py --events matthew,ian --scenarios 1-10  # given events / scenarios (range 1-10, list 1,5,41, or all)
  python3 run_all_events.py --events irma --scenarios 41           # also works for Irma
  python3 run_all_events.py --force ...                            # re-run even verified runs
  python3 run_all_events.py --postprocess-only ...                 # do not run SFINCS, only redo verification and the 7-day event-period summary

Inputs: data/static/sfincs_base/ (grid, mask, boundary points, source points, observation points, check_run.py) + data/<event>/scenarios/<event>_sNNN/{sfincs.bzs,sfincs.dis,sfincs.ampr,sfincs.inp}
      + data/<event>/scenarios/manifest.csv (produced by the scenario generator; error if missing, never generated here)
Outputs: runs/<event>/<event>_sNNN/  all SFINCS inputs/outputs + run_status.json (config, start/end time, status, verification) + error.txt (on failure)
      runs/<event>/<event>_sNNN/event7d/  hourly-sampled max water depth of the 7-day event period (cell-wise max over the 168 zs snapshots t = 73...240 h): hmax_event7d_hourlysampled.npy, summary.json
      runs/run_ledger.csv (master ledger, one row per run, re-runs overwrite the same row) + runs/run_all_events.log
Verification (not just the exit code): sfincs_map.nc and sfincs_his.nc exist and are non-empty; sfincs.log contains "Simulation finished" and no error/cannot/fail lines;
      map has 241 hourly snapshots (dt=3600, last = tstop - tref) and contains zs / zb / cumprcp (zsmax not required); his last = tstop; check_run.py and 21_rain_qc.py succeed;
      rainfall check: if map has valid cumprcp, use received/input ratio (0.98-1.02); with dtmaxout=0 SFINCS does not write cumprcp, so use an ampr-only check
      (ampr md5 reconciliation + file/coverage checks + input total/(beta x same-tau central scenario s040/s041/s042) within 0.99-1.01); run_status.rain_qc_mode records the mode used (2026-09-16).
      All passed -> verified=true; otherwise status=failed, error.txt kept, re-run next time.
Resume after interruption: runs with verified=true in run_status.json are skipped (unless --force).
"""
import os, sys, json, csv, re, time, hashlib, argparse, datetime, subprocess, shutil
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'data/static/sfincs_base'); IMG = 'deltares/sfincs-cpu'
STATIC = ['sfincs.dep', 'sfincs.msk', 'sfincs.ind', 'sfincs.bnd', 'sfincs.src', 'sfincs.obs', 'check_run.py']
FORCING = ['sfincs.bzs', 'sfincs.dis', 'sfincs.ampr', 'sfincs.inp']
LEDGER = os.path.join(ROOT, 'runs/run_ledger.csv'); LOGF = os.path.join(ROOT, 'runs/run_all_events.log')
LEDGER_COLS = ['event', 'scenario', 'status', 'verified', 'start_utc', 'end_utc', 'elapsed_s', 'docker_exit', 'sfincs_version', 'log_errors', 'map_t_last_s', 'expected_t_last_s',
               'rain_qc_ratio', 'event7d_wet_frac', 'inp_md5', 'bzs_md5', 'dis_md5', 'ampr_md5', 'error']
UTC = datetime.timezone.utc
now = lambda: datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    line = '%s %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg); print(line, flush=True)
    os.makedirs(os.path.dirname(LOGF), exist_ok=True); open(LOGF, 'a').write(line + '\n')


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def parse_scen(s, n=81):
    if s == 'all':
        return list(range(1, n + 1))
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def inp_params(path):
    d = {}
    for line in open(path):
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.split('=', 1); d[k.strip()] = v.strip()
    return d


def t_seconds(inp, key):
    return (datetime.datetime.strptime(inp[key], '%Y%m%d %H%M%S') - datetime.datetime.strptime(inp['tref'], '%Y%m%d %H%M%S')).total_seconds()


def ledger_update(row):
    rows = {}
    if os.path.exists(LEDGER):
        for r in csv.DictReader(open(LEDGER)):
            rows[(r['event'], r['scenario'])] = r
    rows[(row['event'], row['scenario'])] = {c: row.get(c, '') for c in LEDGER_COLS}
    with open(LEDGER, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def verify(run, inp):
    """Return (ok, info, error_msg)"""
    info = {}; errs = []
    for f, mn in (('sfincs_map.nc', 1_000_000), ('sfincs_his.nc', 1000)):
        p = os.path.join(run, f)
        if not os.path.exists(p) or os.path.getsize(p) < mn:
            errs.append('%s missing or too small' % f)
    lg = os.path.join(run, 'sfincs.log')
    if not os.path.exists(lg):
        errs.append('sfincs.log missing'); info['log_errors'] = None; info['finished'] = False
    else:
        txt = open(lg, errors='replace').read()
        info['log_errors'] = sum(1 for l in txt.split('\n') if re.search(r'error|cannot|fail', l, re.I))
        info['finished'] = 'Simulation finished' in txt or 'simulation finished' in txt.lower()
        m = re.search(r'Build-Revision.*?Rev: *(\S+)', txt); info['sfincs_version'] = m.group(1) if m else ''
        if info['log_errors']:
            errs.append('log has %d error/cannot/fail lines' % info['log_errors'])
        if not info['finished']:
            errs.append('log lacks "Simulation finished"')
    exp = t_seconds(inp, 'tstop'); info['expected_t_last_s'] = exp
    try:
        import netCDF4
        if not errs:
            with netCDF4.Dataset(os.path.join(run, 'sfincs_map.nc')) as ds:
                tv = ds.variables['time'][:]; info['map_t_last_s'] = float(tv[-1]); info['map_n_time'] = int(len(tv))
                for v in ('zs', 'cumprcp', 'zb'):
                    if v not in ds.variables:
                        errs.append('map lacks variable %s' % v)
                info['has_zsmax'] = 'zsmax' in ds.variables
                if abs(float(tv[-1]) - exp) > 1:
                    errs.append('map last time %.0f != tstop %.0f' % (tv[-1], exp))
                import numpy as _np
                dt = _np.diff(_np.array(tv, float)); info['map_dt_all_3600'] = bool(len(dt) and _np.allclose(dt, 3600, atol=1))
                if len(tv) != int(exp / 3600) + 1 or not info['map_dt_all_3600']:
                    errs.append('map has %d hourly snapshots / all intervals 3600 s = %s, expected %d' % (len(tv), info['map_dt_all_3600'], int(exp / 3600) + 1))
            with netCDF4.Dataset(os.path.join(run, 'sfincs_his.nc')) as ds:
                tv = ds.variables['time'][:]; info['his_t_last_s'] = float(tv[-1])
                if abs(float(tv[-1]) - exp) > 700:
                    errs.append('his last time %.0f != tstop %.0f' % (tv[-1], exp))
    except ImportError:
        errs.append('netCDF4 missing, cannot verify time coverage (pip install netCDF4)')
    return (len(errs) == 0), info, '; '.join(errs)


def event7d(run, inp, ev, sid):
    """The "hourly-sampled max water depth" of the 7-day event period: cell-wise max(zs - zb, 0) over the 168 hourly instantaneous zs snapshots with t > 72 h (dtout=3600, t = 73 h ... 240 h).
    These are snapshot extremes and may miss short peaks between hours (on 2026-09-15 the user decided not to save the interval-max map zsmax, dtmaxout=0)."""
    import numpy as np, netCDF4
    out = os.path.join(run, 'event7d'); os.makedirs(out, exist_ok=True)
    spin = 72 * 3600.0; tstop = t_seconds(inp, 'tstop')
    with netCDF4.Dataset(os.path.join(run, 'sfincs_map.nc')) as ds:
        tv = np.array(ds.variables['time'][:], float); zb = np.array(ds.variables['zb'][:], float)
        sel = np.where(tv > spin + 1)[0]
        if len(sel) != 168 or abs(tv[sel[-1]] - tstop) > 1:
            raise RuntimeError('event-period hourly snapshots %d != 168 or last snapshot %.0f != tstop' % (len(sel), tv[sel[-1]] if len(sel) else -1))
        hmax = None; hmax_all = None
        for i in range(len(tv)):
            z = np.array(ds.variables['zs'][i], float); h = np.where(np.isfinite(z), z - zb, np.nan); h = np.where(h > 0, h, 0.0)
            hmax_all = h if hmax_all is None else np.fmax(hmax_all, h)
            if i >= sel[0]:
                hmax = h if hmax is None else np.fmax(hmax, h)
        active = np.isfinite(np.array(ds.variables['zs'][sel[0]], float)); zs_last = np.array(ds.variables['zs'][-1], float)
    np.save(os.path.join(out, 'hmax_event7d_hourlysampled.npy'), hmax.astype(np.float32)); np.save(os.path.join(out, 'hmax_full10d_hourlysampled.npy'), hmax_all.astype(np.float32))
    land = active & (zb >= 0)
    s = dict(event=ev, scenario=sid, method='hourly-sampled max water depth: event period = the 168 hourly instantaneous zs snapshots with t > 72 h (t = 73...240 h), hmax = max over snapshots of max(zs - zb, 0); may miss short peaks between hours; the full 10 days (241 snapshots) are also saved as *_full10d_* for comparison',
             n_snapshots_event=int(len(sel)), t_first_snapshot_s=float(tv[sel[0]]), t_last_snapshot_s=float(tv[sel[-1]]), n_snapshots_total=int(len(tv)), active_cells=int(active.sum()),
             wet_frac_h_gt_0p1=float((hmax[active] > 0.1).mean()), hmax_mean_m=float(np.nanmean(hmax[active])), hmax_max_m=float(np.nanmax(hmax[active])),
             hmax_land_max_m=float(np.nanmax(hmax[land])), hmax_land_mean_m=float(np.nanmean(hmax[land])), land_wet_frac=float((hmax[land] > 0.1).mean()), land_cells=int(land.sum()),
             wet_frac_full10d=float((hmax_all[active] > 0.1).mean()), nan_in_zs_active=int((~np.isfinite(hmax[active])).sum()),
             end_state=dict(zs_last_max=float(np.nanmax(zs_last)), zs_last_min=float(np.nanmin(zs_last)), wet_frac_at_tstop_h_gt_0p1=float(((zs_last - zb) > 0.1)[active].mean())))
    json.dump(s, open(os.path.join(out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    return s


def rain_qc_ampr_only(run, ev, sid, status):
    """Rainfall check with dtmaxout=0: (1) sfincs.ampr md5 in the run directory == manifest ampr_md5; (2) all file/coverage checks in rain_qc.json pass;
    (3) input domain-mean total / (beta x this event's s041 domain-mean total) within 0.97-1.03 (with tau!=0, +-6 h of real data shift in at the window ends, so totals differ slightly).
    Pass -> record rain_qc_ratio_vs_beta_baseline; the SFINCS received ratio is no longer required (the six s041 runs were verified with dtmaxout=3600 at 0.9999-1.0001)."""
    import pandas as pd
    errs = []
    man = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv'))
    row = man[man.scenario == sid].iloc[0]
    if md5(os.path.join(run, 'sfincs.ampr')) != row['ampr_md5']:
        errs.append('sfincs.ampr md5 in the run directory differs from the manifest')
    try:
        rq = json.load(open(os.path.join(run, 'rain_qc.json')))
        tc, cv = rq['time_check'], rq['coverage']
        if not (tc['strictly_increasing'] and tc['all_1h'] and tc['covers_window'] and tc['nan'] == 0 and tc['nodata'] == 0 and tc['negative'] == 0):
            errs.append('ampr time/value checks failed')
        if cv['active_outside_ampr'] != 0 or cv['active_outside_valid'] != 0:
            errs.append('active cells not covered by ampr')
        mean_in = rq['input_cum_mm']['domain_mean']
        # Reference = central scenario with the same tau (alpha=beta=gamma=1): tau=-6->s040, 0->s041, +6->s042. With tau!=0, 6 h of real rainfall shift in/out at the window ends, so the total already differs from s041
        # (measured 2026-09-16: Dorian tau=+6 1.0757x, Beryl 0.9595x), so we must not divide by s041 directly; against the same-tau reference the ratio should equal beta to write precision.
        tau = int(round(float(row['tau_h']))); ref_sid = {-6: 's040', 0: 's041', 6: 's042'}[tau]
        ref_run = os.path.join(os.path.dirname(run), '%s_%s' % (ev, ref_sid))
        ref_mean = json.load(open(os.path.join(ref_run, 'rain_qc.json')))['input_cum_mm']['domain_mean']
        beta = float(row['beta'])
        ratio = mean_in / (beta * ref_mean)
        status['rain_qc_ratio_vs_beta_ref'] = ratio; status['rain_qc_ref'] = ref_sid
        if ref_sid != 's041':
            base_mean = json.load(open(os.path.join(os.path.dirname(run), '%s_s041' % ev, 'rain_qc.json')))['input_cum_mm']['domain_mean']
            status['rain_qc_tau_shift_ratio'] = ref_mean / base_mean     # record the total change caused by the time shift itself, for the report
        if not (0.99 <= ratio <= 1.01):
            errs.append('input total/(beta x same-tau reference %s) = %.4f not within 0.99-1.01' % (ref_sid, ratio))
    except Exception as ex:
        errs.append('failed to read rain_qc.json: %s' % ex)
    return (len(errs) == 0), ('; '.join(errs) if errs else '')


def baseline_check(run, status):
    """Extra checks for the baseline (s041): input consistency, rainfall total ratio, Mayport boundary read-in, stability, time coverage, plausibility of results. Returns (ok, dict)."""
    import re
    c = {}; errs = []
    mf = os.path.join(ROOT, 'data', status['event'], 'scenarios', 'manifest.csv')
    try:
        import pandas as pd
        m = pd.read_csv(mf, dtype=str); r = m[m.scenario == status['scenario']].iloc[0]
        c['inputs_match_manifest'] = all(status['md5']['sfincs.' + k] == r[k + '_md5'] for k in ('bzs', 'dis', 'ampr', 'inp'))
        if not c['inputs_match_manifest']: errs.append('input md5 does not match the manifest')
    except Exception as ex:
        errs.append('manifest check failed %s' % ex)
    rr = status.get('rain_qc_ratio'); c['rain_qc_ratio'] = rr
    if status.get('rain_qc_mode') == 'ampr_only':
        c['rain_qc_mode'] = 'ampr_only (dtmaxout=0, no cumprcp; received ratio verified by the dtmaxout=3600 baseline)'
    elif rr is None or not (0.98 <= rr <= 1.02): errs.append('rainfall total ratio %s not within 0.98-1.02' % rr)
    try:
        txt = open(os.path.join(run, 'check_run_stdout.txt'), errors='replace').read()
        mm = re.search(r'RMS ([0-9.]+) m', txt); c['mayport_rms_m'] = float(mm.group(1)) if mm else None
        if c['mayport_rms_m'] is None or c['mayport_rms_m'] > 0.10: errs.append('Mayport simulated-minus-input RMS %s (>0.10 m or missing)' % c['mayport_rms_m'])
    except Exception as ex:
        errs.append('failed to read check_run output %s' % ex)
    e7 = status.get('event7d', {})
    c['nan_in_zs_active'] = e7.get('nan_in_zs_active'); c['hmax_max_m'] = e7.get('hmax_max_m'); c['wet_frac_event7d'] = e7.get('wet_frac_h_gt_0p1'); c['n_snapshots_total'] = e7.get('n_snapshots_total')
    if e7.get('nan_in_zs_active', 1) != 0: errs.append('zs has NaN on active cells')
    c['hmax_land_max_m'] = e7.get('hmax_land_max_m')
    if e7.get('hmax_land_max_m', 99) > 10 or not (0.02 < e7.get('wet_frac_h_gt_0p1', -1) < 0.95): errs.append('implausible result: land-cell (zb>=0) max water depth %s m, wet fraction %s' % (e7.get('hmax_land_max_m'), e7.get('wet_frac_h_gt_0p1')))
    if e7.get('n_snapshots_total') != 241: errs.append('hourly snapshot count %s != 241' % e7.get('n_snapshots_total'))
    v = status.get('verify', {}); c['map_t_last_s'] = v.get('map_t_last_s'); c['log_errors'] = v.get('log_errors')
    c['ok'] = len(errs) == 0; c['errors'] = errs
    return c['ok'], c


INP_OVERRIDE = {'dtmaxout': '0'}             # user decision 2026-09-15: dtmaxout=0 for all six events (interval-max map off), hourly instantaneous fields with dtout=3600 kept; inp is rewritten only inside the run directory, scenario files under data/ untouched
RERUN_SUB = {'irma': 'rerun_dtmax3600'}       # old Irma results runs/irma/irma_sNNN kept; new results go to runs/irma/rerun_dtmax3600/irma_sNNN


def run_dir_of(ev, sid):
    return os.path.join(ROOT, 'runs', ev, RERUN_SUB[ev], sid) if ev in RERUN_SUB else os.path.join(ROOT, 'runs', ev, sid)


def apply_inp_override(path):
    lines = open(path).read().split('\n'); changed = {}
    for j, line in enumerate(lines):
        k = line.split('=')[0].strip() if '=' in line else ''
        if k in INP_OVERRIDE and line.split('=', 1)[1].strip() != INP_OVERRIDE[k]:
            changed[k] = (line.split('=', 1)[1].strip(), INP_OVERRIDE[k]); lines[j] = '%-16s= %s' % (k, INP_OVERRIDE[k])
    if changed:
        open(path, 'w').write('\n'.join(lines))
    return changed


def run_one(ev, i, args, last_elapsed):
    sid = '%s_s%03d' % (ev, i); scd = os.path.join(ROOT, 'data', ev, 'scenarios', sid); run = run_dir_of(ev, sid)
    run_name = os.path.relpath(run, os.path.join(ROOT, 'runs', ev))
    stf = os.path.join(run, 'run_status.json')
    if os.path.exists(stf) and not args.force:
        st = json.load(open(stf))
        if st.get('verified') and not args.postprocess_only:
            log('[%s] already verified, skipping' % sid); return 'skipped', st
    missing = [f for f in FORCING if not os.path.exists(os.path.join(scd, f))]
    if missing:
        log('[%s] missing inputs %s (scenario forcing not generated yet)' % (sid, missing)); return 'missing_input', None
    inp = inp_params(os.path.join(scd, 'sfincs.inp'))
    status = dict(event=ev, scenario=sid, base_dir=BASE, forcing_dir=scd, image=IMG, platform='linux/amd64', created=now(),
                  inp=dict((k, inp[k]) for k in ('tref', 'tstart', 'tstop', 'dtout', 'dtmaxout', 'zsini', 'manning_land', 'manning_sea', 'qinf', 'huthresh', 'advection') if k in inp),
                  md5={f: md5(os.path.join(scd, f)) for f in FORCING}, static_md5={f: md5(os.path.join(BASE, f)) for f in STATIC if f != 'check_run.py'})
    if args.dry_run:
        log('[%s] dry-run: inputs complete' % sid); return 'dry', status
    if not args.postprocess_only:
        os.makedirs(run, exist_ok=True)
        for f in STATIC:
            shutil.copy2(os.path.join(BASE, f), run)
        for f in FORCING:
            shutil.copy2(os.path.join(scd, f), run)
        status['inp_override_applied'] = apply_inp_override(os.path.join(run, 'sfincs.inp')); status['run_inp_md5'] = md5(os.path.join(run, 'sfincs.inp'))
        inp = inp_params(os.path.join(run, 'sfincs.inp'))
        for f in ('sfincs_map.nc', 'sfincs_his.nc', 'sfincs.log', 'error.txt'):
            if os.path.exists(os.path.join(run, f)):
                os.remove(os.path.join(run, f))
        open(os.path.join(run, 'RUN_SOURCE.txt'), 'w').write('event=%s\nscenario=%s\nbase_dir=%s\nforcing_dir=%s\ncreated=%s\n' % (ev, sid, BASE, scd, now()))
        status['start_utc'] = now(); t0 = time.time(); log('[%s] start' % sid)
        with open(os.path.join(run, 'docker_stdout.txt'), 'w') as fo:
            p = subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64', '-v', '%s:/data' % run, IMG], stdout=fo, stderr=subprocess.STDOUT)
        status['docker_exit'] = p.returncode; status['end_utc'] = now(); status['elapsed_s'] = round(time.time() - t0, 1)
        open(os.path.join(os.path.dirname(run), '.last_elapsed_sec'), 'w').write(str(int(status['elapsed_s'])))
    else:
        old = json.load(open(stf)) if os.path.exists(stf) else {}
        for k in ('start_utc', 'end_utc', 'elapsed_s', 'docker_exit', 'inp_override_applied', 'run_inp_md5'):
            status[k] = old.get(k)
        inp = inp_params(os.path.join(run, 'sfincs.inp'))
    ok, info, err = verify(run, inp); status['verify'] = info
    if ok:
        try:
            subprocess.run([sys.executable, 'check_run.py'], cwd=run, stdout=open(os.path.join(run, 'check_run_stdout.txt'), 'w'), stderr=subprocess.STDOUT, check=True, timeout=600)
        except Exception as ex:
            ok = False; err = 'check_run.py failed: %s' % ex
    if ok:
        try:
            if os.path.dirname(run_name):
                os.makedirs(os.path.join(ROOT, 'result/figures', 'fig_' + os.path.dirname(run_name)), exist_ok=True)   # 21_rain_qc saves figures by run name
            r = subprocess.run([sys.executable, os.path.join(ROOT, 'code/21_rain_qc.py'), run_name, ev], capture_output=True, text=True, timeout=900)
            m = re.search(r'"mean_ratio": ([0-9.]+)', r.stdout); status['rain_qc_ratio'] = float(m.group(1)) if m else None
            if r.returncode != 0:
                ok = False; err = '21_rain_qc.py failed: ' + (r.stderr[-300:] if r.stderr else '')
            elif status['rain_qc_ratio'] is None:
                # dtmaxout=0 -> SFINCS does not write cumprcp, so no "received/input" reconciliation; use the ampr-only check instead (user decision 2026-09-16, option B)
                ok, err = rain_qc_ampr_only(run, ev, sid, status)
                status['rain_qc_mode'] = 'ampr_only'
            else:
                status['rain_qc_mode'] = 'cumprcp'
        except Exception as ex:
            ok = False; err = '21_rain_qc.py exception: %s' % ex
    if ok:
        try:
            status['event7d'] = event7d(run, inp, ev, sid)
        except Exception as ex:
            ok = False; err = 'event7d summary failed: %s' % ex
    if ok and sid.endswith('_s041'):
        bok, bc = baseline_check(run, status); status['baseline_check'] = bc
        if not bok:
            ok = False; err = 'baseline check failed: ' + '; '.join(bc['errors'])
    status['verified'] = bool(ok); status['status'] = 'ok' if ok else 'failed'; status['error'] = err
    if not ok:
        open(os.path.join(run, 'error.txt'), 'w').write('%s\n%s\n' % (now(), err))
    elif os.path.exists(os.path.join(run, 'error.txt')):
        os.remove(os.path.join(run, 'error.txt'))
    json.dump(status, open(stf, 'w'), ensure_ascii=False, indent=1)
    ledger_update(dict(event=ev, scenario=sid, status=status['status'], verified=status['verified'], start_utc=status.get('start_utc', ''), end_utc=status.get('end_utc', ''),
                       elapsed_s=status.get('elapsed_s', ''), docker_exit=status.get('docker_exit', ''), sfincs_version=info.get('sfincs_version', ''), log_errors=info.get('log_errors', ''),
                       map_t_last_s=info.get('map_t_last_s', ''), expected_t_last_s=info.get('expected_t_last_s', ''), rain_qc_ratio=status.get('rain_qc_ratio', ''),
                       event7d_wet_frac=status.get('event7d', {}).get('wet_frac_h_gt_0p1', ''), inp_md5=status['md5']['sfincs.inp'], bzs_md5=status['md5']['sfincs.bzs'],
                       dis_md5=status['md5']['sfincs.dis'], ampr_md5=status['md5']['sfincs.ampr'], error=err))
    # back-fill the manifest
    mf = os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv')
    if os.path.exists(mf) and ev not in RERUN_SUB:      # the Irma manifest is frozen (records the old run status); re-run results go only to the ledger and run_status.json
        try:
            import pandas as pd
            m = pd.read_csv(mf, dtype=str)
            for c in ('sfincs_run_status', 'sfincs_version', 'rain_qc_ratio'):
                if c not in m.columns:
                    m[c] = ''
            m.loc[m.scenario == sid, ['sfincs_run_status', 'sfincs_version', 'rain_qc_ratio']] = [status['status'], info.get('sfincs_version', ''), str(status.get('rain_qc_ratio', ''))]
            m.to_csv(mf, index=False)
        except Exception as ex:
            log('[%s] manifest back-fill failed: %s' % (sid, ex))
    log('[%s] %s  %ss  log errors %s  rain ratio %s  %s' % (sid, status['status'], status.get('elapsed_s', '-'), info.get('log_errors', '-'), status.get('rain_qc_ratio', '-'), err))
    return status['status'], status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='irma,matthew,ian,milton,dorian,beryl'); ap.add_argument('--scenarios', default='all')
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--force', action='store_true'); ap.add_argument('--postprocess-only', action='store_true')
    args = ap.parse_args()
    evs = [e.strip() for e in args.events.split(',') if e.strip()]; scen = parse_scen(args.scenarios)
    todo = []
    for ev in evs:
        mf = os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv')
        if not os.path.exists(mf):
            log('event %s lacks manifest.csv (scenarios not generated), skipping this event' % ev); continue
        for i in scen:
            todo.append((ev, i))
    if not args.dry_run and not args.postprocess_only:
        last = {}
        for ev in evs:
            p = os.path.join(os.path.dirname(run_dir_of(ev, 'x')), '.last_elapsed_sec'); last[ev] = int(open(p).read()) if os.path.exists(p) else 39
        n = len(todo); est = sum(last[ev] for ev, _ in todo)
        log('planned %d runs; estimated %d-%d min (basis: %s)' % (n, est * 0.8 / 60, est * 1.5 / 60 + 1, 'last measured ' + ', '.join('%s %ds' % kv for kv in last.items())))
    t_all = time.time(); counts = {}
    for ev, i in todo:
        st, _ = run_one(ev, i, args, None); counts[st] = counts.get(st, 0) + 1
    log('finished: %s; total %.1f min' % (counts, (time.time() - t_all) / 60))
    if not args.dry_run and not args.postprocess_only and counts:
        open(os.path.join(ROOT, 'docs', 'PROGRESS.md'), 'a').write('\n## %s  run_all_events.py %s scenarios %s: %s, total %.0f min\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), args.events, args.scenarios, counts, (time.time() - t_all) / 60))


if __name__ == '__main__':
    main()
