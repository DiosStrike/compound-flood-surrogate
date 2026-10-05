#!/usr/bin/env python3
"""44_exp2_summary.py -- Exp 2 (6-fold leave-one-event-out) summary: reads each fold's training log and eval_test results and builds summary tables and figures in the same format as Exp 1.
Runs locally or on Colab; reads the fold directories only and writes only --out-models and --out-figs. No training, no evaluation.

Inputs (per-fold directory <folds-root>/fold_<event>/):
  train_log.csv, train_stdout.log, fold_info.json, eval_test/{metrics_per_event.csv, metrics_per_run.csv, per_t.csv,
  hmax_rollout_per_run.csv, s041_peak_maps.npz, eval_summary.json} (output of code/39_exp2_eval.py, must include tp/fp/fn count columns)
Outputs:
  <out-models>/metrics_per_event.csv   held-out-event rows of each fold + "all" (6-fold pooled: MAE / RMSE weighted by n, CSI pooled from TP/FP/FN)
  <out-models>/metrics_per_run.csv, per_t.csv, hmax_rollout_per_run.csv, s041_peak_maps.npz (6 folds concatenated, same format as Exp 1)
  <out-models>/folds_training.csv      per fold: epochs, whether max epochs reached, best epoch and val, training hours, median s/step, GPU, sessions, eval seconds
  <out-models>/summary.json
  <out-figs>/fig_exp2_*.png            metric bar charts, rollout error over time, max-depth area comparison, s041 peak triptychs (reusing code/43 plotting functions)
                                       + fig_exp2_training_curves.png (6-fold training curves, marking best and final epoch)
Missing folds or count columns: stop and list the folds to redo; no partial output.

Usage:
  Local (first download Drive MyDrive/Flood2/models/exp2/ to result/models/exp2/folds/):
    python3 code/44_exp2_summary.py
  Colab:
    python3 code/44_exp2_summary.py --folds-root /content/drive/MyDrive/Flood2/models/exp2 \
        --out-models /content/drive/MyDrive/Flood2/models/exp2_summary --out-figs /content/drive/MyDrive/Flood2/figures/exp2
"""
import os, sys, csv, json, re, argparse, importlib.util, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
MODES = ['single', 'rollout', 'persistence']
MASKS = ['active', 'land']
THRS = ['0.1', '0.3']
EVAL_FILES = ['metrics_per_event.csv', 'metrics_per_run.csv', 'per_t.csv', 'hmax_rollout_per_run.csv', 's041_peak_maps.npz', 'eval_summary.json']


def rcsv(p):
    return list(csv.DictReader(open(p)))


def wcsv(p, rows):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def check(folds_root):
    probs = []
    for e in EVENTS:
        d = os.path.join(folds_root, 'fold_%s' % e)
        for f in ['train_log.csv', 'fold_info.json'] + ['eval_test/' + x for x in EVAL_FILES]:
            if not os.path.exists(os.path.join(d, f)):
                probs.append('%s: missing %s' % (e, f))
        p = os.path.join(d, 'eval_test', 'metrics_per_event.csv')
        if os.path.exists(p):
            rows = rcsv(p)
            if 'tp_0.1' not in rows[0]:
                probs.append('%s: eval_test lacks tp/fp/fn count columns (rerun step 8 evaluation with code/39 from 2026-09-17 or later)' % e)
            n_run = len({r['sid'] for r in rcsv(os.path.join(d, 'eval_test', 'metrics_per_run.csv'))})
            if n_run != 10:
                probs.append('%s: number of test runs %d != 10' % (e, n_run))
    return probs


def pooled(rows):
    """rows: event rows from several folds with the same mode / mask -> pooled."""
    n = sum(int(r['n']) for r in rows)
    out = dict(mae=sum(float(r['mae']) * int(r['n']) for r in rows) / n,
               rmse=float(np.sqrt(sum(float(r['rmse']) ** 2 * int(r['n']) for r in rows) / n)), n=n)
    for t in THRS:
        tp, fp, fn = (sum(int(r['%s_%s' % (c, t)]) for r in rows) for c in ('tp', 'fp', 'fn'))
        out['csi_%s' % t] = tp / (tp + fp + fn) if tp + fp + fn else float('nan')
    for t in THRS:
        for c in ('tp', 'fp', 'fn'):
            out['%s_%s' % (c, t)] = sum(int(r['%s_%s' % (c, t)]) for r in rows)
    return out


def fold_training(d, e):
    tl = rcsv(os.path.join(d, 'train_log.csv'))
    best = min(tl, key=lambda r: float(r['val_loss']))
    txt = open(os.path.join(d, 'train_stdout.log'), errors='replace').read() if os.path.exists(os.path.join(d, 'train_stdout.log')) else ''
    sps = [float(x) for x in re.findall(r'\| ([\d.]+) s/step', txt)]
    stop = re.findall(r'training finished \(([^)]*)\)', txt)
    info = json.load(open(os.path.join(d, 'fold_info.json')))
    gpus = sorted({s.get('gpu', s.get('device')) for s in info['sessions']})
    ev = json.load(open(os.path.join(d, 'eval_test', 'eval_summary.json')))
    return dict(fold=e, epochs=len(tl), max_epochs_reached=len(tl) >= 50, stopped=stop[-1] if stop else '',
                best_epoch=int(best['epoch']), best_val_loss=float(best['val_loss']),
                train_hours=round(sum(float(r['epoch_sec']) for r in tl) / 3600, 3), mean_epoch_sec=round(np.mean([float(r['epoch_sec']) for r in tl]), 1),
                sec_per_step_median=round(float(np.median(sps)), 4) if sps else '', gpus=';'.join(map(str, gpus)), sessions=len(info['sessions']),
                eval_seconds=ev.get('seconds', ''), eval_model_epoch=ev.get('model_epoch', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds-root', default=os.path.join(ROOT, 'result', 'models', 'exp2', 'folds'))
    ap.add_argument('--out-models', default=os.path.join(ROOT, 'result', 'models', 'exp2'))
    ap.add_argument('--out-figs', default=os.path.join(ROOT, 'result', 'figures', 'exp2'))
    a = ap.parse_args()
    probs = check(a.folds_root)
    if probs:
        print('Stopped: the following items are missing or incomplete; no partial output:'); [print('  ' + p) for p in probs]; sys.exit(1)
    os.makedirs(a.out_models, exist_ok=True); os.makedirs(a.out_figs, exist_ok=True)
    fd = {e: os.path.join(a.folds_root, 'fold_%s' % e, 'eval_test') for e in EVENTS}

    # event level: take the held-out-event rows of each fold, then pool the 6 folds as 'all'
    ev_rows = []
    for e in EVENTS:
        ev_rows += [r for r in rcsv(os.path.join(fd[e], 'metrics_per_event.csv')) if r['event'] == e]
    all_rows = [dict(event='all', mode=m, mask=k, **pooled([r for r in ev_rows if r['mode'] == m and r['mask'] == k])) for m in MODES for k in MASKS]
    wcsv(os.path.join(a.out_models, 'metrics_per_event.csv'), ev_rows + all_rows)
    for name in ('metrics_per_run.csv', 'hmax_rollout_per_run.csv'):
        wcsv(os.path.join(a.out_models, name), [r for e in EVENTS for r in rcsv(os.path.join(fd[e], name))])
    wcsv(os.path.join(a.out_models, 'per_t.csv'), [r for e in EVENTS for r in rcsv(os.path.join(fd[e], 'per_t.csv')) if r['event'] == e])
    maps = {}; peak = []
    for e in EVENTS:
        z = np.load(os.path.join(fd[e], 's041_peak_maps.npz'))
        for k in ('truth', 'single', 'rollout'):
            maps['%s_%s' % (e, k)] = z['%s_%s' % (e, k)]
        peak.append(int(dict(zip([str(x) for x in z['events']], z['peak_t']))[e]))
    np.savez_compressed(os.path.join(a.out_models, 's041_peak_maps.npz'), **maps, peak_t=np.array(peak), events=np.array(EVENTS))
    ft = [fold_training(os.path.join(a.folds_root, 'fold_%s' % e), e) for e in EVENTS]
    wcsv(os.path.join(a.out_models, 'folds_training.csv'), ft)
    summ = dict(folds_root=os.path.abspath(a.folds_root), pooling='per fold: held-out event, 10 runs x 168 times x cells; all = 6 folds pooled (MAE / RMSE weighted by n, CSI pooled from TP/FP/FN)',
                csi_def='TP/(TP+FP+FN), wet = h > threshold', thresholds_m=[0.1, 0.3], folds=ft,
                note='the irma fold ran the full 50 epochs and was still improving at the end; to keep the 6 folds comparable, the other folds kept the same settings (max 50 epochs, patience 10)',
                events={r['event']: {} for r in ev_rows + all_rows})
    for r in ev_rows + all_rows:
        summ['events'][r['event']].setdefault(r['mode'], {})[r['mask']] = {k: (float(v) if k not in ('event', 'mode', 'mask') else v) for k, v in r.items() if k not in ('event', 'mode', 'mask')}
    json.dump(summ, open(os.path.join(a.out_models, 'summary.json'), 'w'), indent=1, ensure_ascii=False)

    # figures: reuse code/43 functions with changed read/write dirs and title label; file names fig_exp1_* -> fig_exp2_*
    spec = importlib.util.spec_from_file_location('exp1_fig', os.path.join(ROOT, 'code', '43_exp1_figures.py'))
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
    tmp = os.path.join(a.out_figs, '_tmp43'); os.makedirs(tmp, exist_ok=True)
    F.EV, F.FIG, F.LABEL = a.out_models, tmp, 'Exp 2 (6-fold leave-one-out, each event held out)'
    F.fig_bars(); F.fig_time(); F.fig_hmax(); F.fig_maps()
    for f in os.listdir(tmp):
        shutil.move(os.path.join(tmp, f), os.path.join(a.out_figs, f.replace('fig_exp1_', 'fig_exp2_')))
    os.rmdir(tmp)
    fig_training_curves(a.folds_root, a.out_figs, F)

    print('6-fold summary (active-cell RMSE, m; held-out event 10 runs x 168 times):')
    print('  %-8s %8s %8s %8s   %s' % ('event', 'single', 'rollout', 'persist', 'training (epochs / best / hours)'))
    for e in EVENTS + ['all']:
        g = lambda m: summ['events'][e][m]['active']['rmse']
        t = next((x for x in ft if x['fold'] == e), None)
        print('  %-8s %8.4f %8.4f %8.4f   %s' % (e, g('single'), g('rollout'), g('persistence'), '%d / %d / %.2f h' % (t['epochs'], t['best_epoch'], t['train_hours']) if t else ''))
    print('Output:', a.out_models, a.out_figs)


def fig_training_curves(folds_root, out_figs, F):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=F.SURF)
    for ax, e in zip(axes.ravel(), EVENTS):
        F.style(ax)
        tl = rcsv(os.path.join(folds_root, 'fold_%s' % e, 'train_log.csv'))
        ep = [int(r['epoch']) for r in tl]; tr = [float(r['train_loss']) for r in tl]; va = [float(r['val_loss']) for r in tl]
        best = min(tl, key=lambda r: float(r['val_loss'])); be, bv = int(best['epoch']), float(best['val_loss'])
        ax.plot(ep, tr, color=F.CAT[0], lw=1.8, label='Train loss'); ax.plot(ep, va, color=F.CAT[1], lw=1.8, label='Val loss')
        ax.set_yscale('log'); ax.yaxis.set_major_formatter(F.PLAIN)
        ax.axvline(be, color=F.MUTED, lw=1, ls='--'); ax.plot([be], [bv], 'o', ms=7, color=F.CAT[1], mec=F.SURF, mew=2)
        ax.set_title('Held out %s: %d epochs, best epoch %d val %.3g' % (F.EV_CN[e], ep[-1], be, bv), fontsize=9, color=F.INK, loc='left')
        ax.set_xlabel('Epoch', color=F.INK2, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Exp 2 training curves (6-fold leave-one-out; max 50 epochs, patience 10; dashed = best epoch)', fontsize=11, color=F.INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(out_figs, 'fig_exp2_training_curves.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
