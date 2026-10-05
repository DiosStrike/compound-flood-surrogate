#!/usr/bin/env python3
"""46_exp3_sampling.py -- Exp. 3 fine-tuning run lists: one nested stratified sample per fold and seed (docs/PLAN.md v16 §8.10 item 5). Reads index.csv only.

Rules:
  pool = the 60 split_A train runs of the held-out event (val 11 / test 10 are never touched);
  combined forcing level = α, β, γ each scored low/mid/high = 0/1/2; total 0–2 low / 3 mid / 4–6 high (same rule as split_A);
  nesting: within a fold and seed, N=1 ⊂ 2 ⊂ 3 ⊂ 5 ⊂ 10 -- one ordered list of length 10 is generated; the first N are the sample for N;
  stratification: levels rotate low/mid/high from a (seed-determined) start; runs within each level are shuffled by seed; the first 10 contain 3–4 of each level;
  B, B′ and C share the same list. Randomness depends only on (seed, fold), so it is reproducible.
Output: result/models/exp3/exp3_samples.csv, columns fold, seed, order, sid, level, alpha, beta, gamma, tau_h (order 1..10).

Usage: python3 code/46_exp3_sampling.py [--data-dir data/modelB_v1] [--out result/models/exp3/exp3_samples.csv] [--seeds 0,1,2]
"""
import argparse, csv, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
LEVELS = ['low', 'mid', 'high']
N_MAX = 10


def level_of(r):
    sc = {0.8: 0, 1.0: 1, 1.2: 2}[float(r['alpha'])] + {0.7: 0, 1.0: 1, 1.3: 2}[float(r['beta'])] + {0.75: 0, 1.0: 1, 1.25: 2}[float(r['gamma'])]
    return 'low' if sc <= 2 else ('mid' if sc == 3 else 'high')


def sample_fold(rows, fold, seed):
    pool = [r for r in rows if r['event'] == fold and r['split_A'] == 'train']
    assert len(pool) == 60, (fold, len(pool))
    rng = np.random.default_rng([seed, EVENTS.index(fold), 2026])
    by = {lv: [r for r in pool if level_of(r) == lv] for lv in LEVELS}
    for lv in LEVELS:
        rng.shuffle(by[lv])
    start = int(rng.integers(3)); order_lv = LEVELS[start:] + LEVELS[:start]
    out = []; k = 0
    while len(out) < N_MAX:
        lv = order_lv[k % 3]; k += 1
        if by[lv]:
            r = by[lv].pop(0); out.append((r, lv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'modelB_v1'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'result', 'models', 'exp3', 'exp3_samples.csv'))
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.data_dir, 'index.csv'))))
    out = []
    for fold in EVENTS:
        for seed in [int(x) for x in a.seeds.split(',')]:
            for i, (r, lv) in enumerate(sample_fold(rows, fold, seed), 1):
                out.append(dict(fold=fold, seed=seed, order=i, sid=r['sid'], level=lv, alpha=r['alpha'], beta=r['beta'], gamma=r['gamma'], tau_h=r['tau_h']))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(out[0].keys())); w.writeheader(); w.writerows(out)
    # self-check: nesting (order unique within fold and seed), no val/test, 3–4 of each level in the first 10
    test_val = {r['sid'] for r in rows if r['split_A'] != 'train'}
    assert not any(o['sid'] in test_val for o in out)
    for fold in EVENTS:
        for seed in [int(x) for x in a.seeds.split(',')]:
            sel = [o for o in out if o['fold'] == fold and o['seed'] == seed]
            assert len(sel) == N_MAX and len({o['sid'] for o in sel}) == N_MAX
            cnt = {lv: sum(1 for o in sel if o['level'] == lv) for lv in LEVELS}
            assert all(3 <= c <= 4 for c in cnt.values()), cnt
            print(fold, 'seed', seed, 'N=5 levels', {lv: sum(1 for o in sel[:5] if o['level'] == lv) for lv in LEVELS}, 'N=10', cnt)
    print('written', a.out, len(out), 'rows')


if __name__ == '__main__':
    main()
