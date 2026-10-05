#!/usr/bin/env python3
"""64_infer_timing.py -- measure single-time-step U-Net inference time on the local Mac (MPS). Read-only; writes no project files.

Method:
  load result/models/exp1/model_best.pt (exp1 best weights, epoch 32) -> take one run with split_A = test
  -> warm up with a few forward passes -> run single-step inference on all 168 times t = 73...240 and time it (torch.mps.synchronize() before each timing).
  Both batch 8 (the actual evaluation setting, same basis as the A100 conversion in docs/PLAN.md §8.18) and batch 1 (single-time latency) are reported.

Forward only: no backward pass, no files written; does not affect any existing results.

Data sources: result/models/exp1/model_best.pt, data/modelB_v1/ (index.csv + <event>/<sid>.npz + static.npz)

Usage: python3 code/64_infer_timing.py [--sid <sid>] [--repeat 3]
"""
import os, sys, time, argparse, importlib.util
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def sync(dev):
    if dev.type == 'mps':
        torch.mps.synchronize()
    elif dev.type == 'cuda':
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', default='')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=5)
    a = ap.parse_args()

    D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py')
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device: %s; torch %s' % (dev.type, torch.__version__))

    ck = torch.load(os.path.join(ROOT, 'result/models/exp1/model_best.pt'), map_location='cpu', weights_only=False)
    print('Weights: result/models/exp1/model_best.pt (epoch %s, val %.3g)' % (ck.get('epoch', '?'), ck.get('val_loss', float('nan'))))
    st = D.load_static(); norm = D.Normalizer(ck['stats'], st)
    model = U.UNetV1(); model.load_state_dict(ck['model']); model = model.to(dev).eval()

    idx = D.load_index()                                   # pandas DataFrame
    sid = a.sid or str(idx[idx['split_A'] == 'test']['sid'].iloc[0])
    r = D.load_run_arrays(sid)
    ts = list(range(D.T_MIN, D.T_MAX + 1))
    print('Test run: %s; times t = %d...%d, %d in total' % (sid, ts[0], ts[-1], len(ts)))

    # Inputs are assembled once on the CPU; timing covers only the forward pass itself (same basis as the A100 conversion)
    X = np.stack([norm.sample(r, t)[0] for t in ts])
    print('Input tensor %s, %.1f MB' % (tuple(X.shape), X.nbytes / 1e6))

    def run(bs):
        out = []
        with torch.no_grad():
            for _ in range(a.warmup):                          # warm-up
                model(torch.from_numpy(X[:bs]).to(dev)); sync(dev)
            for _ in range(a.repeat):
                sync(dev); t0 = time.perf_counter()
                for i in range(0, len(ts), bs):
                    model(torch.from_numpy(X[i:i + bs]).to(dev))
                sync(dev); out.append(time.perf_counter() - t0)
        return np.array(out)

    print()
    for bs in (8, 1):
        e = run(bs)
        print('batch %d: total time for %d times %.3f s (%d repeats: %s); %.2f ms per time step'
              % (bs, len(ts), e.mean(), a.repeat, ' / '.join('%.3f' % v for v in e), 1000 * e.mean() / len(ts)))
    print('\nNote: forward inference only; excludes model loading, disk reads and input assembly.')


if __name__ == '__main__':
    main()
