#!/usr/bin/env python3
"""39_exp2_eval.py -- Experiment 2 single-fold evaluation (docs/PLAN.md v11 §1 evaluation protocol, §7). Rewritten 2026-09-17 to use the same protocol as code/42_exp1_eval.py.

Does not modify code/32-37: single-step / rollout inference calls code/35_modelB_eval.py; the data directory is set via --data-dir into module 32's DATA.
Three predictions: single (single-step, h(t-1...t-3) from ground truth), rollout (autoregressive, starting from ground-truth h(70-72)), persistence (copy the previous hour, h(t) = ground-truth h(t-1)).
Metrics: one set each on active cells (msk > 0) and land cells (active and zb >= 0): MAE, RMSE (m), CSI (wet = h > 0.1 m / 0.3 m, TP/(TP+FP+FN));
      pooled over run x time x cell within each event, plus an "all" total; rollout also reports per-time error and max depth vs SFINCS hmax.
Only difference from code/42: SFINCS hmax is the per-cell max of h_proc over t = 73...240 (no runs/ on Colab; preprocessing self-check 486/486
      and Experiment 1 evaluation 60/60 both confirmed it equals event7d/hmax_event7d_hourlysampled.npy cell by cell).
Input: <fold dir>/model_best.pt (includes normalization statistics).
Output: <fold dir>/eval_<tag>/ -- metrics_per_run.csv, metrics_per_event.csv, per_t.csv, hmax_rollout_per_run.csv,
      s041_peak_maps.npz (if s041 is included), eval_summary.json. tag: test (default, 10 test runs) / all81 / smoke.

Usage:
  python3 code/39_exp2_eval.py --fold matthew --data-dir /content/Flood_2.0/data/modelB_v1 --out-root /content/drive/MyDrive/Flood2/models [--all-runs]
  Consistency check (local, with the Experiment 1 model): python3 code/39_exp2_eval.py --ckpt result/models/exp1/model_best.pt --runs irma_s041,ian_s005 --out-dir <temp dir>
"""
import os, sys, json, time, argparse, importlib.util, csv
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = ['single', 'rollout', 'persistence']
THRS = [0.1, 0.3]
CELL_KM2 = 0.04


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Acc:
    """Same as Acc in code/42_exp1_eval.py: pooled MAE / RMSE / CSI; additionally outputs TP / FP / FN count columns (absent in 42; metric values identical)."""
    def __init__(self):
        self.sa = 0.0; self.ss = 0.0; self.n = 0; self.tp = {t: 0 for t in THRS}; self.fp = {t: 0 for t in THRS}; self.fn = {t: 0 for t in THRS}

    def add(self, P, T):
        e = P - T; self.sa += float(np.abs(e).sum()); self.ss += float((e.astype(np.float64) ** 2).sum()); self.n += e.size
        for t in THRS:
            pw, tw = P > t, T > t
            self.tp[t] += int((pw & tw).sum()); self.fp[t] += int((pw & ~tw).sum()); self.fn[t] += int((~pw & tw).sum())

    def result(self):
        out = dict(mae=self.sa / self.n, rmse=float(np.sqrt(self.ss / self.n)), n=self.n)
        for t in THRS:
            d = self.tp[t] + self.fp[t] + self.fn[t]
            out['csi_%s' % t] = self.tp[t] / d if d else float('nan')
        for t in THRS:                                   # added 2026-09-17: count columns for pooling CSI across folds (code/44)
            out['tp_%s' % t], out['fp_%s' % t], out['fn_%s' % t] = self.tp[t], self.fp[t], self.fn[t]
        return out


def hms(s):
    s = int(s); return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def evaluate_model(ckpt_path, sids, dev, D, out_dir):
    E35 = _imp('mb_eval35', '35_modelB_eval.py'); U = _imp('mb_unet', '33_modelB_unet.py')
    TS = list(range(D.T_MIN, D.T_MAX + 1))
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = U.UNetV1(); model.load_state_dict(ck['model']); model.to(dev).eval()
    st = D.load_static(); norm = D.Normalizer(ck['stats'], st)
    act = st['loss_mask']; land = act & (st['zb'] >= 0); masks = dict(active=act, land=land)
    events = sorted({s.rsplit('_', 1)[0] for s in sids}, key=lambda e: [x.rsplit('_', 1)[0] for x in sids].index(e))
    os.makedirs(out_dir, exist_ok=True)
    print('[%s] device %s; model epoch %d (val %.6g); eval runs %d; active cells %d, land cells %d -> %s' % (
        time.strftime('%H:%M:%S'), dev, ck['epoch'], ck['val_loss'], len(sids), act.sum(), land.sum(), out_dir), flush=True)
    ev_acc = {(e, m, k): Acc() for e in events + ['all'] for m in MODES for k in masks}
    per_t = {(e, m, k): np.zeros((len(TS), 3)) for e in events for m in MODES for k in masks}
    run_rows, hmax_rows = [], []; peak_maps = {}
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        ev = sid.rsplit('_', 1)[0]; tr = time.time()
        r = D.load_run_arrays(sid); T = r['h'][D.T_MIN:D.T_MAX + 1]
        ps, _ = E35.single_step(model, norm, r, dev)
        pr, _ = E35.rollout(model, norm, r, dev)
        preds = dict(single=np.stack([ps[t] for t in TS]), rollout=np.stack([pr[t] for t in TS]),
                     persistence=r['h'][D.T_MIN - 1:D.T_MAX])
        for m, P in preds.items():
            for k, mk in masks.items():
                Pm, Tm = P[:, mk], T[:, mk]
                a = Acc(); a.add(Pm, Tm); ev_acc[(ev, m, k)].add(Pm, Tm); ev_acc[('all', m, k)].add(Pm, Tm)
                run_rows.append(dict(event=ev, sid=sid, mode=m, mask=k, **a.result()))
                e = Pm - Tm
                per_t[(ev, m, k)][:, 0] += np.abs(e).sum(1); per_t[(ev, m, k)][:, 1] += (e.astype(np.float64) ** 2).sum(1); per_t[(ev, m, k)][:, 2] += e.shape[1]
        hmax = T.max(0)                              # = hmax_event7d_hourlysampled (verified equal cell by cell)
        pmax = preds['rollout'].max(0)
        row = dict(event=ev, sid=sid)
        for k, mk in masks.items():
            e = pmax[mk] - hmax[mk]
            row['mae_%s' % k] = float(np.abs(e).mean()); row['rmse_%s' % k] = float(np.sqrt((e.astype(np.float64) ** 2).mean()))
            row['bias_%s' % k] = float(e.mean())
        for t in THRS:
            pw, tw = pmax[act] > t, hmax[act] > t
            row['csi_%s' % t] = float((pw & tw).sum() / max((pw | tw).sum(), 1))
            row['land_area_km2_sfincs_gt%s' % t] = float((hmax[land] > t).sum() * CELL_KM2)
            row['land_area_km2_pred_gt%s' % t] = float((pmax[land] > t).sum() * CELL_KM2)
        row['max_depth_sfincs'] = float(hmax[act].max()); row['max_depth_pred'] = float(pmax[act].max())
        row['hmax_source'] = 'h_proc max t=73..240'
        hmax_rows.append(row)
        if sid.endswith('_s041'):
            tp = int(np.argmax(T[:, land].sum(1)))
            peak_maps[ev] = dict(t=TS[tp], truth=T[tp], single=preds['single'][tp], rollout=preds['rollout'][tp])
        done = time.time() - t0; eta = done / i * (len(sids) - i)
        rs = [x for x in run_rows if x['sid'] == sid and x['mask'] == 'active']
        print('[%s] %2d/%d %-14s %.1f s | RMSE single %.4f rollout %.4f persistence %.4f | elapsed %s ETA %s' % (
            time.strftime('%H:%M:%S'), i, len(sids), sid, time.time() - tr,
            *[next(x['rmse'] for x in rs if x['mode'] == m) for m in MODES], hms(done), hms(eta)), flush=True)

    def wcsv(name, rows):
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv('metrics_per_run.csv', run_rows)
    wcsv('metrics_per_event.csv', [dict(event=e, mode=m, mask=k, **a.result()) for (e, m, k), a in ev_acc.items()])
    wcsv('per_t.csv', [dict(event=e, mode=m, mask=k, t=t, mae=arr[j, 0] / arr[j, 2], rmse=float(np.sqrt(arr[j, 1] / arr[j, 2])))
                       for (e, m, k), arr in per_t.items() for j, t in enumerate(TS)])
    wcsv('hmax_rollout_per_run.csv', hmax_rows)
    if peak_maps:
        pe = [e for e in events if e in peak_maps]
        np.savez_compressed(os.path.join(out_dir, 's041_peak_maps.npz'),
                            **{'%s_%s' % (e, k): v for e, d in peak_maps.items() for k, v in d.items() if k != 't'},
                            peak_t=np.array([peak_maps[e]['t'] for e in pe]), events=np.array(pe))
    summ = dict(model=os.path.abspath(ckpt_path), model_epoch=ck['epoch'], model_val_loss=ck['val_loss'], device=dev,
                n_runs=len(sids), sids=list(sids), t_range=[D.T_MIN, D.T_MAX], n_t=len(TS), n_active=int(act.sum()), n_land=int(land.sum()),
                thresholds_m=THRS, csi_def='TP/(TP+FP+FN), wet = h > threshold', pooling='pooled over run x time x cell within event', peak_def='time of max summed land-cell depth in s041',
                hmax_source='per-cell max of h_proc over t = 73...240', peak_t={e: d['t'] for e, d in peak_maps.items()}, seconds=round(time.time() - t0, 1),
                events={e: {m: {k: ev_acc[(e, m, k)].result() for k in masks} for m in MODES} for e in events + ['all']})
    json.dump(summ, open(os.path.join(out_dir, 'eval_summary.json'), 'w'), indent=1, ensure_ascii=False)
    print('[%s] evaluation done: %d runs, elapsed %s -> %s' % (time.strftime('%H:%M:%S'), len(sids), hms(time.time() - t0), out_dir), flush=True)
    return summ


def evaluate_fold(out, sids, dev, D, tag='test'):
    """Called by 38 --eval-after: out = fold directory."""
    return evaluate_model(os.path.join(out, 'model_best.pt'), sids, dev, D, os.path.join(out, 'eval_%s' % tag))


def main():
    T = _imp('exp2_train', '38_exp2_train.py')
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='')
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'modelB_v1'))
    ap.add_argument('--out-root', default=os.path.join(ROOT, 'result', 'models'))
    ap.add_argument('--device', default='')
    ap.add_argument('--all-runs', action='store_true', help='evaluate all 81 runs of the held-out event (default: only the 10 test runs)')
    ap.add_argument('--smoke', action='store_true', help='evaluate the smoke-test model under exp2_smoke, 1 run only')
    ap.add_argument('--ckpt', default='', help='specify the model directly (use with --runs and --out-dir, e.g. for a consistency check)')
    ap.add_argument('--runs', default='')
    ap.add_argument('--out-dir', default='')
    a = ap.parse_args()
    T.set_data_dir(a.data_dir)
    dev = T.pick_device(a.device); print('device: %s' % json.dumps(T.device_info(dev), ensure_ascii=False), flush=True)
    if a.ckpt:
        if not (a.runs and a.out_dir):
            sys.exit('--ckpt requires both --runs and --out-dir')
        evaluate_model(a.ckpt, [s for s in a.runs.split(',') if s], dev, T.D, a.out_dir); return
    if not a.fold:
        sys.exit('please give --fold (or --ckpt + --runs + --out-dir)')
    out = os.path.join(a.out_root, 'exp2_smoke' if a.smoke else 'exp2', 'fold_%s' % a.fold)
    _, _, test_sids = T.fold_sids(a.fold)
    if a.all_runs:
        sids, tag = ['%s_s%03d' % (a.fold, n) for n in range(1, 82)], 'all81'
    elif a.smoke:
        sids, tag = test_sids[:1], 'smoke'
    else:
        sids, tag = test_sids, 'test'
    evaluate_fold(out, sids, dev, T.D, tag)


if __name__ == '__main__':
    main()
