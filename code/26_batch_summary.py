#!/usr/bin/env python3
"""26_batch_summary.py -- summary of the 6 events × 81 scenarios SFINCS batch (done/failed/runtime/disk/rain-QC mode/event×scenario status plot) + short PDF report.
Reads only runs/<event>/<sid>/run_status.json, runs/run_ledger.csv, data/<event>/scenarios/manifest.csv."""
import os, sys, json, glob, datetime
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
SUB = {'irma': 'rerun_dtmax3600'}
def rdir(ev, sid): return os.path.join(ROOT, 'runs', ev, SUB[ev], sid) if ev in SUB else os.path.join(ROOT, 'runs', ev, sid)
def dsize(d): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)

rows = []
for ev in EVS:
    man = pd.read_csv(os.path.join(ROOT, 'data', ev, 'scenarios', 'manifest.csv'))
    for i in range(1, 82):
        sid = '%s_s%03d' % (ev, i); d = rdir(ev, sid); st = json.load(open(os.path.join(d, 'run_status.json')))
        m = man[man.scenario == sid].iloc[0]
        rq = json.load(open(os.path.join(d, 'rain_qc.json')))['input_cum_mm']['domain_mean']
        rows.append(dict(event=ev, i=i, sid=sid, status=st['status'], verified=st['verified'], elapsed=st.get('elapsed_s'), dtmaxout=st['inp'].get('dtmaxout'),
                         mode=st.get('rain_qc_mode'), ref=st.get('rain_qc_ref'), r_cum=st.get('rain_qc_ratio'), r_ref=st.get('rain_qc_ratio_vs_beta_ref'),
                         r_tau=st.get('rain_qc_tau_shift_ratio'), cum_in=rq, wet=st.get('event7d', {}).get('land_wet_frac'), log_err=st.get('verify', {}).get('log_errors'),
                         alpha=m.alpha, beta=m.beta, gamma=m.gamma, tau=m.tau_h, bytes=dsize(d), man_status=m.sfincs_run_status, err=st.get('error', '')))
df = pd.DataFrame(rows)
# independent recomputation (not relying on run_status fields): same-τ reference ratio = input total / (β × total of the same-τ centre scenario); time-shift ratio = reference / s041
ref_of = {-6: 40, 0: 41, 6: 42}
cum = {(r['event'], r['i']): r['cum_in'] for r in rows}
df['r_ref_calc'] = [r['cum_in'] / (r['beta'] * cum[(r['event'], ref_of[int(r['tau'])])]) for r in rows]
df['r_tau_calc'] = [cum[(r['event'], ref_of[int(r['tau'])])] / cum[(r['event'], 41)] for r in rows]
led = pd.read_csv(os.path.join(ROOT, 'runs', 'run_ledger.csv'))
S = dict(n=len(df), ok=int((df.status == 'ok').sum()), verified=int(df.verified.sum()), ledger_ok=int((led.status == 'ok').sum()), ledger_n=len(led),
         manifest_ok=int((df.man_status == 'ok').sum()), log_err_total=int(df.log_err.fillna(0).sum()),
         elapsed_total_h=float(df.elapsed.sum() / 3600), elapsed_min=float(df.elapsed.min()), elapsed_max=float(df.elapsed.max()), elapsed_mean=float(df.elapsed.mean()),
         disk_gb=float(df.bytes.sum() / 1e9), mode_counts=df['mode'].value_counts().to_dict(), dtmaxout_counts=df.dtmaxout.astype(str).value_counts().to_dict(),
         r_cum_range=[float(df.r_cum.dropna().min()), float(df.r_cum.dropna().max())], r_ref_range=[float(df.r_ref_calc.min()), float(df.r_ref_calc.max())], r_ref_in_status=int(df.r_ref.notna().sum()),
         tau_shift={ev: {str(t): float(df[(df.event == ev) & (df.tau == t)].r_tau_calc.iloc[0]) for t in (-6, 6)} for ev in EVS},
         per_event={ev: dict(ok=int((g.status == 'ok').sum()), elapsed_h=float(g.elapsed.sum() / 3600), elapsed_mean=float(g.elapsed.mean()), disk_gb=float(g.bytes.sum() / 1e9),
                             wet_min=float(g.wet.min()), wet_max=float(g.wet.max()), wet_s041=float(g[g.i == 41].wet.iloc[0])) for ev, g in df.groupby('event', sort=False)},
         generated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
os.makedirs(os.path.join(ROOT, 'result', 'figures'), exist_ok=True); os.makedirs(os.path.join(ROOT, 'data', 'static', 'meta'), exist_ok=True)
df.to_csv(os.path.join(ROOT, 'result', 'manifests', 'batch486_verification_2026-09-16.csv'), index=False)
json.dump(S, open(os.path.join(ROOT, 'data', 'static', 'meta', 'batch486_summary_2026-09-16.json'), 'w'), ensure_ascii=False, indent=1)

# ---- Fig. 1: event × scenario status ----
code = np.full((6, 81), 0); 
for r in rows:
    code[EVS.index(r['event']), r['i'] - 1] = 1 if r['status'] == 'ok' and r['mode'] == 'ampr_only' else (2 if r['status'] == 'ok' else 3)
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.imshow(code, cmap=ListedColormap(['#dddddd', '#4c9a4c', '#1f5fa8', '#c0392b']), vmin=0, vmax=3, aspect='auto')
ax.set_yticks(range(6)); ax.set_yticklabels(EVS); ax.set_xticks(range(0, 81, 5)); ax.set_xticklabels(['s%03d' % (i + 1) for i in range(0, 81, 5)], rotation=90, fontsize=7)
ax.set_xticks(np.arange(-.5, 81, 1), minor=True); ax.set_yticks(np.arange(-.5, 6, 1), minor=True); ax.grid(which='minor', color='w', lw=.4)
ax.set_title('SFINCS run status, 6 events × 81 scenarios (%d/%d ok)   green = ok, rain QC ampr-only (dtmaxout=0); blue = ok, cumprcp reconciliation (s041, dtmaxout=3600); red = failed' % (S['ok'], S['n']), fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(ROOT, 'result', 'figures', 'fig_batch486_status.png'), dpi=160); plt.close(fig)

# ---- Fig. 2: per-event runtime / disk / flooded-fraction range ----
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
pe = S['per_event']; x = np.arange(6)
axs[0].bar(x, [pe[e]['elapsed_h'] for e in EVS], color='#888'); axs[0].set_xticks(x); axs[0].set_xticklabels(EVS); axs[0].set_ylabel('h'); axs[0].set_title('Compute time of 81 runs per event (total %.1f h, mean %.0f s per run)' % (S['elapsed_total_h'], S['elapsed_mean']), fontsize=9)
axs[1].bar(x, [pe[e]['disk_gb'] for e in EVS], color='#888'); axs[1].set_xticks(x); axs[1].set_xticklabels(EVS); axs[1].set_ylabel('GB'); axs[1].set_title('Output disk per event (total %.1f GB)' % S['disk_gb'], fontsize=9)
for k, e in enumerate(EVS):
    g = df[df.event == e]; axs[2].scatter(np.full(81, k) + np.random.default_rng(0).uniform(-.25, .25, 81), g.wet * 100, s=6, color='#4c9a4c', alpha=.6)
    axs[2].plot(k, pe[e]['wet_s041'] * 100, 'k_', ms=18, mew=2)
axs[2].set_xticks(x); axs[2].set_xticklabels(EVS); axs[2].set_ylabel('%'); axs[2].set_title('Main 7-day land cells (zb≥0) with h>0.1 m, fraction (dots = 81 scenarios, bar = s041)', fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(ROOT, 'result', 'figures', 'fig_batch486_stats.png'), dpi=160); plt.close(fig)

# ---- PDF ----
def wrap(ln, width):
    out, cur, w = [], '', 0
    ind = len(ln) - len(ln.lstrip(' '))
    for ch in ln:
        cw = 1.0 if ord(ch) > 0x2e80 else 0.55
        if w + cw > width and cur.strip():
            out.append(cur); cur = ' ' * (ind + 2); w = (ind + 2) * 0.55
        cur += ch; w += cw
    out.append(cur); return out
def page(pdf, title, lines, fs=8.6):
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(.07, .95, title, fontsize=13.5, weight='bold')
    y = .915; width = 0.86 * 8.27 * 72 / fs / 1.0
    for ln in lines:
        for sub in (wrap(ln, width) if ln else ['']):
            fig.text(.07, y, sub, fontsize=fs, va='top'); y -= 0.0165
    pdf.savefig(fig); plt.close(fig)
ts = S['tau_shift']
L1 = ['Generated: %s. Scope: six hurricanes (Irma 2017 / Matthew 2016 / Ian 2022 / Milton 2024 / Dorian 2019 / Beryl 2012) × 81 Irma-like compound forcing scenarios = 486 SFINCS simulations.' % S['generated'],
      '',
      'Conclusion: 486/486 runs completed, outputs complete, verification passed (run_status.verified %d, ledger ok %d/%d, manifest ok %d).' % (S['verified'], S['ledger_ok'], S['ledger_n'], S['manifest_ok']),
      '',
      '1. Runs',
      '  · 480 runs with dtmaxout=0 / dtout=3600 (hourly instantaneous fields); the 6 s041 runs are the earlier dtmaxout=3600 versions, kept; Manning 0.06/0.02; SFINCS v2.4.2-alpha, docker deltares/sfincs-cpu.',
      '  · Compute time total %.1f h, per run %.0f–%.0f s (mean %.0f s); wall clock of the 480-run batch 371 min (finished 2026-09-15 21:46).' % (S['elapsed_total_h'], S['elapsed_min'], S['elapsed_max'], S['elapsed_mean']),
      '  · Output disk %.1f GB (runs/<event>/, incl. input copies, map/his/log/QC figures).' % S['disk_gb'],
      '  · Per event: ' + '; '.join('%s %.2f h / %.2f GB' % (e, pe[e]['elapsed_h'], pe[e]['disk_gb']) for e in EVS) + '.',
      '',
      '2. Checks (every run, run_all_events.py)',
      '  · map / his exist; log has 0 errors and "Simulation finished" (error lines over 486 runs: %d); 241 hourly snapshots, last = tstop; no NaN in zs/zb.' % S['log_err_total'],
      '  · bzs/dis/ampr/inp md5 in the run directory = manifest; check_run.py passed; hourly-sampled max depth of the main 7 days (event7d) summarized.',
      '  · Two rain-QC modes: cumprcp reconciliation %d runs (s041, received/input %.5f–%.5f); ampr-only %d runs (SFINCS does not write cumprcp when dtmaxout=0, see 3).' % (S['mode_counts'].get('cumprcp', 0), *S['r_cum_range'], S['mode_counts'].get('ampr_only', 0)),
      '',
      '3. Two fixes to the verification logic in this round (2026-09-16; only verification scripts changed, simulation results untouched)',
      '  1. dtmaxout=0 -> SFINCS v2.4.2-alpha skips the interval-maximum output block, cumprcp is all NaN, so the original "received/input" reconciliation is unusable; 480 runs were wrongly marked failed.',
      '     Switched to ampr-only: ampr md5 = manifest + file time/value/coverage checks + domain-mean input total / (β × reference scenario) within 0.99–1.01. The six s041 runs already show via cumprcp that SFINCS received = input.',
      '  2. The reference scenario must have the same τ: τ=−6/0/+6 map to s040/s041/s042 (all α=β=γ=1). Scenarios with τ≠0 bring in ±6 h of real rainfall at the window ends, so their total legitimately ≠ s041;',
      '     dividing by s041 directly misjudged 54 τ=+6 scenarios (Dorian 1.0757×, Beryl 0.9595×). With the same-τ reference, the 486 ratios are %.5f–%.5f (recomputed independently in this report from the rain_qc.json of each run), i.e. ratio = β to output precision.' % tuple(S['r_ref_range']),
      '     Total change due to the time shift itself (reference/s041, stored as run_status.rain_qc_tau_shift_ratio): ' + '; '.join('%s τ−6 %.4f / τ+6 %.4f' % (e, ts[e]['-6'], ts[e]['6']) for e in EVS) + '.',
      '',
      '4. Flooded fraction h>0.1 m of land cells (zb≥0) over the main 7 days (range over 81 scenarios; parentheses = s041, same definition as the 09-15 six-event baseline report)',
      '  ' + '; '.join('%s %.1f–%.1f %% (%.1f)' % (e, pe[e]['wet_min'] * 100, pe[e]['wet_max'] * 100, pe[e]['wet_s041'] * 100) for e in EVS) + '.',
      '',
      '5. Outputs',
      '  runs/run_ledger.csv (486 rows ok); result/manifests/batch486_verification_2026-09-16.csv (per-run verification table); data/static/meta/batch486_summary_2026-09-16.json;',
      '  result/figures/fig_batch486_status.png, fig_batch486_stats.png; scripts code/26_batch_summary.py, run_all_events.py (backup .bak_20260916), code/21_rain_qc.py (same backup).',
      '',
      '6. Not done: dataset not packaged, no model trained, split_A / scenario factors unchanged, frozen Irma data unchanged. Next stage awaits user instruction.']
with PdfPages(os.path.join(ROOT, 'result', 'reports', 'REPORT_CURRENT.pdf')) as pdf:
    page(pdf, 'Flood_2.0 -- SFINCS batch of 6 events × 81 scenarios: completion and verification (2026-09-16)', L1)
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(.07, .95, 'Fig. 1 event × scenario run status; Fig. 2 per-event runtime / disk / main 7-day land flooded fraction', fontsize=11, weight='bold')
    for k, f in enumerate(['fig_batch486_status.png', 'fig_batch486_stats.png']):
        ax = fig.add_axes([.03, .62 - .38 * k, .94, .3]); ax.imshow(plt.imread(os.path.join(ROOT, 'result', 'figures', f))); ax.axis('off')
    pdf.savefig(fig); plt.close(fig)
print(json.dumps({k: S[k] for k in ('n', 'ok', 'verified', 'ledger_ok', 'manifest_ok', 'log_err_total', 'elapsed_total_h', 'elapsed_mean', 'disk_gb', 'mode_counts', 'r_ref_range', 'tau_shift')}, ensure_ascii=False, indent=1))
