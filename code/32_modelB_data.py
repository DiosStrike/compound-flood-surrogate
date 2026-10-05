#!/usr/bin/env python3
"""32_modelB_data.py -- data loading for Scheme B U-Net v1 (docs/PLAN.md v8 §6). Reads data/modelB_v1/ only; does not modify data.

Input 13 channels (fixed order):
  0 zb | 1-3 h(t-1), h(t-2), h(t-3) | 4-6 rain blocks t-1, t-2, t-3 | 7-9 wl(t-1..t-3) | 10-12 dis(t-1..t-3) (scalars broadcast to constant maps)
Target: h_proc(t), (195, 255), m, not normalized; t = 73...240.
Padding: raw values are first padded to 208 x 256 (north = 13 rows appended at row end, east = 1 column appended at column end); padded cells get h / rain / wl / dis = 0 and zb = zb_fill_value
      (same as inactive cells), then each channel is standardized. loss_mask is False on padded cells.
Standardization statistics: computed from training runs only (compute_stats); zb over active cells; the 3 h channels share one set (active cells, frames 70...239), rainfall likewise
      (active cells, blocks 70...239), water level / discharge one set each (hours 70...239).
Loading: streaming via a run buffer (RunBufferSampler); only buffer_runs decompressed runs are kept in memory at once (about 96 MB each).
"""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'modelB_v1')
NY, NX = 195, 255
PY, PX = 208, 256
T_MIN, T_MAX = 73, 240
LOOKBACK = 3
IN_FRAMES = slice(T_MIN - LOOKBACK, T_MAX)      # frames / blocks / hours that appear as inputs: 70...239


def load_static():
    z = np.load(os.path.join(DATA, 'static.npz'))
    msk = z['msk']; act = z['loss_mask'].astype(bool); zb = z['zb_filled'].astype(np.float32)
    fill = float(np.unique(zb[~act])[0])
    return dict(zb=zb, msk=msk, loss_mask=act, zb_fill=fill)


def load_index():
    return pd.read_csv(os.path.join(DATA, 'index.csv'))


def run_path(sid):
    return os.path.join(DATA, sid.rsplit('_', 1)[0], sid + '.npz')


def load_run_arrays(sid):
    z = np.load(run_path(sid))
    return dict(sid=sid, h=z['h_proc'], rain=z['rain_proc'], wl=z['wl_hourly'].astype(np.float32), dis=z['dis_hourly'].astype(np.float32))


class RunningStat:
    def __init__(self):
        self.n = 0; self.s = 0.0; self.ss = 0.0

    def add(self, x):
        x = np.asarray(x, np.float64).ravel(); self.n += x.size; self.s += x.sum(); self.ss += (x * x).sum()

    def result(self):
        m = self.s / self.n; v = max(self.ss / self.n - m * m, 0.0)
        return dict(mean=float(m), std=float(np.sqrt(v)) if v > 0 else 1.0, n=int(self.n))


def compute_stats(train_sids, static=None):
    """Computed from training runs only. Returns {zb, h, rain, wl, dis: {mean, std, n}}."""
    st = static or load_static(); act = st['loss_mask']
    acc = {k: RunningStat() for k in ('h', 'rain', 'wl', 'dis')}
    zb = RunningStat(); zb.add(st['zb'][act])
    for sid in train_sids:
        r = load_run_arrays(sid)
        acc['h'].add(r['h'][IN_FRAMES][:, act]); acc['rain'].add(r['rain'][IN_FRAMES][:, act])
        acc['wl'].add(r['wl'][IN_FRAMES]); acc['dis'].add(r['dis'][IN_FRAMES])
    out = {k: v.result() for k, v in acc.items()}; out['zb'] = zb.result(); out['train_sids'] = list(train_sids)
    return out


def _pad(a, fill):
    """(..., 195, 255) -> (..., 208, 256); pad with fill on the north (row end) and east (column end)."""
    pad = [(0, 0)] * (a.ndim - 2) + [(0, PY - NY), (0, PX - NX)]
    return np.pad(a, pad, constant_values=fill)


class Normalizer:
    def __init__(self, stats, static):
        self.st = static
        m = np.array([stats['zb']['mean']] + [stats['h']['mean']] * 3 + [stats['rain']['mean']] * 3 + [stats['wl']['mean']] * 3 + [stats['dis']['mean']] * 3, np.float32)
        s = np.array([stats['zb']['std']] + [stats['h']['std']] * 3 + [stats['rain']['std']] * 3 + [stats['wl']['std']] * 3 + [stats['dis']['std']] * 3, np.float32)
        self.mean = m[:, None, None]; self.std = s[:, None, None]
        self.zb_pad = _pad(static['zb'], static['zb_fill'])
        self.mask_pad = _pad(static['loss_mask'], False)

    def build(self, h_prev, rain_prev, wl_prev, dis_prev):
        """h_prev / rain_prev: (3, 195, 255) in order t-1, t-2, t-3; wl_prev / dis_prev: (3,). Returns standardized (13, 208, 256)."""
        x = np.empty((13, PY, PX), np.float32)
        x[0] = self.zb_pad
        x[1:4] = _pad(h_prev, 0.0); x[4:7] = _pad(rain_prev, 0.0)
        x[7:10] = np.asarray(wl_prev, np.float32)[:, None, None]; x[10:13] = np.asarray(dis_prev, np.float32)[:, None, None]
        x[7:13, NY:, :] = 0.0; x[7:13, :, NX:] = 0.0          # padded cells set to 0 (raw value)
        return (x - self.mean) / self.std

    def sample(self, r, t):
        k = [t - 1, t - 2, t - 3]
        return self.build(r['h'][k], r['rain'][k], r['wl'][k], r['dis'][k]), r['h'][t]


def parse_t_list(spec):
    """'73:234:7' -> [73, 80, ..., 234] (inclusive); 'all' or empty -> all of 73...240."""
    if not spec or spec == 'all':
        return list(range(T_MIN, T_MAX + 1))
    a, b, c = (int(v) for v in spec.split(':'))
    ts = list(range(a, b + 1, c))
    assert ts and ts[0] >= T_MIN and ts[-1] <= T_MAX, spec
    return ts


class RunBufferSampler:
    """Run-buffer streaming: shuffle run order each epoch, decompress buffer_runs runs at a time, build (run, t) pairs within the buffer, shuffle, and yield batches.
    t_list: target times taken from every run (default all of 73...240); if samples_per_run>0, that many are drawn at random from t_list each epoch instead.
    With shuffle=False runs and samples stay in order (for validation). Randomness is set by seed + epoch (set_epoch), so interrupted runs resume reproducibly."""

    def __init__(self, sids, norm, batch_size=8, buffer_runs=16, samples_per_run=None, seed=42, drop_last=False, t_list=None, shuffle=True):
        self.sids = list(sids); self.norm = norm; self.bs = batch_size; self.buf = buffer_runs
        self.spr = samples_per_run; self.seed = seed; self.drop_last = drop_last; self.shuffle = shuffle
        self.t_list = list(t_list) if t_list else list(range(T_MIN, T_MAX + 1))
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng([self.seed, epoch])

    def steps_per_epoch(self):
        n = len(self.sids) * (self.spr or len(self.t_list))
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        order = self.rng.permutation(len(self.sids)) if self.shuffle else np.arange(len(self.sids))
        tail = []
        for i in range(0, len(order), self.buf):
            runs = [load_run_arrays(self.sids[j]) for j in order[i:i + self.buf]]
            pairs = []
            for ri in range(len(runs)):
                ts = np.array(self.t_list)
                if self.spr: ts = self.rng.choice(ts, self.spr, replace=False)
                pairs += [(ri, int(t)) for t in ts]
            pairs = [(runs[ri], t) for ri, t in pairs]
            if self.shuffle: self.rng.shuffle(pairs)
            pairs = tail + pairs; tail = []
            nfull = len(pairs) // self.bs * self.bs
            for b in range(0, nfull, self.bs):
                yield self._batch(pairs[b:b + self.bs])
            tail = pairs[nfull:]
            del runs
        if tail and not self.drop_last:
            yield self._batch(tail)

    def _batch(self, pairs):
        xs, ys = zip(*(self.norm.sample(r, t) for r, t in pairs))
        return np.stack(xs), np.stack(ys)
