#!/usr/bin/env python3
"""42_exp1_eval.py -- Exp. 1 test-set evaluation (split_A test of six events, 60 runs × 168 time steps). Reads data and model only; writes only result/models/exp1/eval/.

Model: result/models/exp1/model_best.pt (includes normalization statistics). Single-step / rollout inference calls code/35_modelB_eval.py directly.
Three predictions:
  single       single-step: h(t−1...t−3) from SFINCS ground truth
  rollout      rollout: start from ground-truth h(70–72); for t ≥ 73 model predictions are fed back (inactive cells 0)
  persistence  previous-hour persistence baseline: h(t) = ground-truth h(t−1)
Metrics (active cells only; land cells = active cells with zb ≥ 0 reported separately): MAE, RMSE (m); flood-extent CSI (h > 0.1 m, h > 0.3 m; CSI = TP / (TP + FP + FN)).
Pooling: all runs × time steps × cells within an event are pooled; "all" pools the 60 runs of the six events.
Rollout also reports: RMSE / MAE per t (pooled over the 10 runs of an event); per-run predicted max depth (cell-wise max over t = 73...240) compared with SFINCS
  event7d/hmax_event7d_hourlysampled.npy.
Outputs: metrics_per_run.csv, metrics_per_event.csv, per_t.csv, hmax_rollout_per_run.csv, s041_peak_maps.npz, eval_summary.json, eval_stdout.log (redirected by the launch command).
"""
import os, sys, json, time, csv, importlib.util
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py'); E35 = _imp('mb_eval35', '35_modelB_eval.py')
from read_run import run_dir

EXP = os.path.join(ROOT, 'result', 'models', 'exp1'); OUT = os.path.join(EXP, 'eval')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
MODES = ['single', 'rollout', 'persistence']
THRS = [0.1, 0.3]
TS = list(range(D.T_MIN, D.T_MAX + 1))
CELL_KM2 = 0.04


class Acc:
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
        return out


def hms(s):
    s = int(s); return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    ck = torch.load(os.path.join(EXP, 'model_best.pt'), map_location='cpu', weights_only=False)
    model = U.UNetV1(); model.load_state_dict(ck['model']); model.to(dev).eval()
    st = D.load_static(); norm = D.Normalizer(ck['stats'], st)
    act = st['loss_mask']; land = act & (st['zb'] >= 0)
    masks = dict(active=act, land=land)
    idx = D.load_index(); test = idx[(idx.split_A == 'test') & idx.event.isin(EVENTS)]
    sids = [s for e in EVENTS for s in test[test.event == e].sid.tolist()]
    assert len(sids) == 60, len(sids)
    print('[%s] device %s; model epoch %d (val %.6g); test runs %d; active cells %d, land cells %d' % (
        time.strftime('%H:%M:%S'), dev, ck['epoch'], ck['val_loss'], len(sids), act.sum(), land.sum()), flush=True)

    ev_acc = {(e, m, k): Acc() for e in EVENTS + ['all'] for m in MODES for k in masks}
    per_t = {(e, m, k): np.zeros((len(TS), 3)) for e in EVENTS for m in MODES for k in masks}   # sum_abs, sum_sq, n
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
        # rollout max depth vs SFINCS hmax
        hmax = np.load(os.path.join(run_dir(ev, sid), 'event7d', 'hmax_event7d_hourlysampled.npy'))
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
        row['maxfile_equals_hproc_max'] = bool(np.array_equal(T.max(0), hmax))
        hmax_rows.append(row)
        if sid.endswith('_s041'):
            tp = int(np.argmax(T[:, land].sum(1)))            # peak time: max summed land-cell depth
            peak_maps[ev] = dict(t=TS[tp], truth=T[tp], single=preds['single'][tp], rollout=preds['rollout'][tp])
        done = time.time() - t0; eta = done / i * (len(sids) - i)
        rs = [x for x in run_rows if x['sid'] == sid and x['mask'] == 'active']
        print('[%s] %2d/%d %-14s %.1f s | RMSE single %.4f rollout %.4f persistence %.4f | elapsed %s remaining %s' % (
            time.strftime('%H:%M:%S'), i, len(sids), sid, time.time() - tr,
            *[next(x['rmse'] for x in rs if x['mode'] == m) for m in MODES], hms(done), hms(eta)), flush=True)

    def wcsv(name, rows):
        with open(os.path.join(OUT, name), 'w', newline='') as f:
            w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv('metrics_per_run.csv', run_rows)
    wcsv('metrics_per_event.csv', [dict(event=e, mode=m, mask=k, **a.result()) for (e, m, k), a in ev_acc.items()])
    trows = []
    for (e, m, k), arr in per_t.items():
        for j, t in enumerate(TS):
            trows.append(dict(event=e, mode=m, mask=k, t=t, mae=arr[j, 0] / arr[j, 2], rmse=float(np.sqrt(arr[j, 1] / arr[j, 2]))))
    wcsv('per_t.csv', trows)
    wcsv('hmax_rollout_per_run.csv', hmax_rows)
    np.savez_compressed(os.path.join(OUT, 's041_peak_maps.npz'),
                        **{'%s_%s' % (e, k): v for e, d in peak_maps.items() for k, v in d.items() if k != 't'},
                        peak_t=np.array([peak_maps[e]['t'] for e in EVENTS]), events=np.array(EVENTS))
    summ = dict(model='result/models/exp1/model_best.pt', model_epoch=ck['epoch'], model_val_loss=ck['val_loss'], device=dev,
                n_runs=len(sids), test_sids=sids, t_range=[D.T_MIN, D.T_MAX], n_t=len(TS), n_active=int(act.sum()), n_land=int(land.sum()),
                thresholds_m=THRS, csi_def='TP/(TP+FP+FN), wet = h > threshold', pooling='pooled over run x time x cell within event', peak_def='time of max summed land-cell depth in s041',
                peak_t={e: peak_maps[e]['t'] for e in EVENTS}, seconds=round(time.time() - t0, 1),
                events={e: {m: {k: ev_acc[(e, m, k)].result() for k in masks} for m in MODES} for e in EVENTS + ['all']})
    json.dump(summ, open(os.path.join(OUT, 'eval_summary.json'), 'w'), indent=1, ensure_ascii=False)
    print('[%s] done: 60 runs, took %s -> %s' % (time.strftime('%H:%M:%S'), hms(time.time() - t0), OUT), flush=True)


if __name__ == '__main__':
    main()
