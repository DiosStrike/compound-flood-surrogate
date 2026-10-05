#!/usr/bin/env python3
"""35_modelB_eval.py -- Plan B evaluation: single-step and rollout (docs/PLAN.md v7 §6). Reads data only; writes only to the --out directory.

Single-step: h_lookback uses SFINCS ground truth; predict each t = 73...240.
Rollout: start from ground-truth h(70), h(71), h(72); predictions for t ≥ 73 are fed back as later inputs (inactive cells filled with 0); rainfall / water level / discharge always use ground truth.
Metrics (per t, active cells only): MSE, MAE, RMSE (m).

Usage: python3 code/35_modelB_eval.py --ckpt <dir>/model.pt --runs matthew_s041 [--device mps]
"""
import os, sys, json, argparse, time, importlib.util
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py')


def _metrics(pred, true, act):
    e = (pred - true)[act]
    return dict(mse=float(np.mean(e ** 2)), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))))


@torch.no_grad()
def single_step(model, norm, r, device, batch_size=8):
    model.eval(); act = norm.st['loss_mask']; ts = list(range(D.T_MIN, D.T_MAX + 1)); preds = {}
    for i in range(0, len(ts), batch_size):
        tb = ts[i:i + batch_size]
        x = np.stack([norm.sample(r, t)[0] for t in tb])
        p = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
        for t, pi in zip(tb, p):
            pi[~act] = 0.0; preds[t] = pi
    per_t = {t: _metrics(preds[t], r['h'][t], act) for t in ts}
    return preds, per_t


@torch.no_grad()
def rollout(model, norm, r, device):
    model.eval(); act = norm.st['loss_mask']
    h = {t: r['h'][t] for t in range(D.T_MIN - D.LOOKBACK, D.T_MIN)}      # ground truth h(70...72)
    per_t = {}
    for t in range(D.T_MIN, D.T_MAX + 1):
        k = [t - 1, t - 2, t - 3]
        x = norm.build(np.stack([h[j] for j in k]), r['rain'][k], r['wl'][k], r['dis'][k])
        p = model(torch.from_numpy(x[None]).to(device)).float().cpu().numpy()[0]
        p[~act] = 0.0; h[t] = p
        per_t[t] = _metrics(p, r['h'][t], act)
    return {t: h[t] for t in range(D.T_MIN, D.T_MAX + 1)}, per_t


def summarize(per_t):
    a = np.array([[v['mse'], v['mae'], v['rmse']] for v in per_t.values()])
    return dict(mse_mean=float(a[:, 0].mean()), mae_mean=float(a[:, 1].mean()), rmse_mean=float(a[:, 2].mean()),
                rmse_max=float(a[:, 2].max()), n_t=len(per_t), any_nan=bool(np.isnan(a).any()))


def load_checkpoint(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = U.UNetV1(); model.load_state_dict(ck['model']); model.to(device)
    st = D.load_static(); norm = D.Normalizer(ck['stats'], st)
    return model, norm, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--runs', required=True)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--modes', default='single,rollout'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    model, norm, ck = load_checkpoint(a.ckpt, a.device)
    out_dir = a.out or os.path.dirname(a.ckpt); res = {}
    for sid in a.runs.split(','):
        r = D.load_run_arrays(sid); res[sid] = {}
        for mode in a.modes.split(','):
            t0 = time.time()
            _, per_t = (single_step if mode == 'single' else rollout)(model, norm, r, a.device)
            res[sid][mode] = dict(summarize(per_t), seconds=round(time.time() - t0, 2), per_t=per_t)
            print(sid, mode, {k: v for k, v in res[sid][mode].items() if k != 'per_t'}, flush=True)
    json.dump(res, open(os.path.join(out_dir, 'eval.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
