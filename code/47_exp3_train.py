#!/usr/bin/env python3
"""47_exp3_train.py -- Experiment 3 (few-shot fine-tuning), full pipeline for one fold, targeting Colab GPU (docs/PLAN.md v17 §8.10-8.19). Does not modify 32-45.

One session runs one fold: read data / norm_stats / Experiment-2 model_best / 60 evaluation runs once (10 held-out test + 50 source-domain test),
loop over method x N x seed from the list: train for a fixed number of epochs (no early stopping, no validation set) -> evaluate immediately (held-out 10 runs single-step + rollout + per-t + hmax; source-domain 50 runs single-step)
-> write manifest / metrics / per-epoch loss; save weights per §8.15; supports resuming from the manifest.

Methods: B  full-network fine-tuning (start = Experiment-2 model_best, lr 1e-4)
      Bp freeze encoder and bottleneck, tune decoder + output layer only (same start, lr 1e-4)
      C  train from scratch on the same N runs (random init, lr 1e-3)
Shared by all three: the fold's norm_stats.json, the same runs (first N of exp3_samples.csv), t = 73:234:7, batch 8, fixed epochs.
N = 0 (zero-shot) is not trained; --eval-zero evaluates the Experiment-2 model once (incl. the "before fine-tuning" baseline on the 50 source runs).

Output <out-root>/exp3/fold_<event>/:
  manifest.csv        one row per config: method, N, seed, epochs, lr, sids, status, seconds, weight_path, finished_at
  metrics_test.csv    held-out test runs, per run x mode x mask (with sa/ss/n/tp/fp/fn, exactly poolable)
  metrics_source.csv  source-domain 50 test runs, per-run single-step
  per_t.csv, hmax.csv held-out test runs per-t and max-depth comparison
  train_log.csv       loss per config per epoch
  weights/            representative configs only (B/Bp N=5,10 seed 0; C N=10 seed 0)
  trial/              --trial mode: per-epoch single-step RMSE curve csv + png

Usage (Colab):
  main run: python3 -u code/47_exp3_train.py --fold dorian --data-dir ... --exp2-root <.../models/exp2> --out-root <.../models> --samples code/exp3_samples.csv --eval-zero
  pilot trial: python3 -u code/47_exp3_train.py --fold dorian --data-dir ... --exp2-root ... --out-root ... --samples ... --trial --trial-epochs B:60,Bp:60,C:80
"""
import os, sys, csv, json, time, argparse, importlib.util, copy, hashlib
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py'); L = _imp('exp3_lib', '48_exp3_eval_lib.py')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
MAN_COLS = ['config', 'method', 'N', 'seed', 'epochs', 'lr', 'sids', 'status', 'train_sec', 'eval_sec', 'final_loss', 'weight_path', 'finished_at']
KEEP_WEIGHTS = {('B', 5, 0), ('B', 10, 0), ('Bp', 5, 0), ('Bp', 10, 0), ('C', 10, 0)}


def say(msg):
    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg), flush=True)


def hms(s):
    s = int(s); return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def pick_device(req=''):
    return req or ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2)[mask.expand_as(pred)].mean()


def append_rows(path, rows, cols=None):
    """Append to csv. If the file exists, **use the file's header** as column order (otherwise differing key order across batches misaligns columns;
    fixed 2026-09-18: previously the config/event/level/tau_h columns of persistence rows were rotated)."""
    if not rows: return
    new = not os.path.exists(path)
    if new:
        cols = cols or list(rows[0].keys())
    else:
        with open(path, newline='') as f:
            cols = next(csv.reader(f))
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore', restval='')
        if new: w.writeheader()
        w.writerows(rows)


def load_manifest(path):
    return {r['config']: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}


def write_manifest(path, man):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, MAN_COLS); w.writeheader(); w.writerows(man.values())


def build_model(method, base_state, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    model = U.UNetV1()
    if method in ('B', 'Bp'):
        model.load_state_dict(base_state)
    model.to(dev)
    if method == 'Bp':
        for mod in (model.enc, model.bottom):
            for p in mod.parameters(): p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    return model, params


def train_fixed(model, params, lr, epochs, sids, norm, mask, dev, seed, log_cb=None, epoch_cb=None):
    opt = torch.optim.Adam(params, lr=lr)
    sampler = D.RunBufferSampler(sids, norm, 8, 16, None, seed=seed, t_list=D.parse_t_list('73:234:7'))
    hist = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); sampler.set_epoch(ep); losses = []
        for x, y in sampler:
            xt = torch.from_numpy(x).to(dev); yt = torch.from_numpy(y).to(dev)
            loss = masked_mse(model(xt), yt, mask); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            lv = loss.item(); losses.append(lv)
            if not np.isfinite(lv): raise RuntimeError('loss is not finite')
        hist.append(float(np.mean(losses)))
        if epoch_cb: epoch_cb(ep, hist[-1])
    return hist, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True, choices=EVENTS)
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'modelB_v1'))
    ap.add_argument('--exp2-root', default=os.path.join(ROOT, 'result', 'models', 'exp2', 'folds'), help='contains fold_<event>/model_best.pt and norm_stats.json')
    ap.add_argument('--out-root', default=os.path.join(ROOT, 'result', 'models'))
    ap.add_argument('--samples', default=os.path.join(ROOT, 'result', 'models', 'exp3', 'exp3_samples.csv'))
    ap.add_argument('--methods', default='B,Bp,C'); ap.add_argument('--n-list', default='1,2,3,5,10'); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', default='B:20,Bp:20,C:50', help='fixed epochs per method (set after the 2026-09-18 pilot: B 20 / Bp 20 / C 50, PLAN §8.17)')
    ap.add_argument('--lr', default='B:1e-4,Bp:1e-4,C:1e-3')
    ap.add_argument('--eval-zero', action='store_true', help='evaluate N=0 once with the Experiment-2 model (held-out 10 runs + source 50 runs)')
    ap.add_argument('--no-source', action='store_true', help='skip source-domain forgetting evaluation (done by default, for all three seeds)')
    ap.add_argument('--trial', action='store_true', help='epoch pilot trial: N=5 seed 0, per-epoch single-step evaluation curve')
    ap.add_argument('--trial-epochs', default='B:60,Bp:60,C:80')
    ap.add_argument('--ref-exp1-active', type=float, default=None); ap.add_argument('--ref-exp1-land', type=float, default=None)
    ap.add_argument('--device', default='')
    a = ap.parse_args()

    D.DATA = os.path.abspath(a.data_dir); dev = pick_device(a.device)
    say('device %s | torch %s | GPU %s' % (dev, torch.__version__, torch.cuda.get_device_name(0) if dev == 'cuda' else '-'))
    fold = a.fold; out = os.path.join(a.out_root, 'exp3', 'fold_%s' % fold); os.makedirs(out, exist_ok=True)
    st = D.load_static(); act = st['loss_mask']; masks = dict(active=act, land=act & (st['zb'] >= 0))
    mask_t = torch.from_numpy(act).to(dev)
    exp2_dir = os.path.join(a.exp2_root, 'fold_%s' % fold)
    stats = json.load(open(os.path.join(exp2_dir, 'norm_stats.json'))); norm = D.Normalizer(stats, st)
    base = torch.load(os.path.join(exp2_dir, 'model_best.pt'), map_location='cpu', weights_only=False)
    base_state = base['model']; say('start: Experiment-2 fold %s model_best (epoch %d, val %.3g); norm_stats reused from this fold' % (fold, base['epoch'], base['val_loss']))
    idx = list(csv.DictReader(open(os.path.join(D.DATA, 'index.csv'))))
    test_sids = [r['sid'] for r in idx if r['event'] == fold and r['split_A'] == 'test']
    source_sids = [r['sid'] for r in idx if r['event'] != fold and r['split_A'] == 'test']
    assert len(test_sids) == 10 and len(source_sids) == 50
    samples = [r for r in csv.DictReader(open(a.samples)) if r['fold'] == fold]
    sid_list = lambda seed, N: [r['sid'] for r in sorted((x for x in samples if int(x['seed']) == seed), key=lambda x: int(x['order']))][:N]
    t0 = time.time(); cache = L.RunCache(D, test_sids + source_sids + sorted({r['sid'] for r in samples}), idx)
    say('evaluation runs cached in memory: held-out test %d + source test %d + fine-tune pool %d, %.0f s' % (len(test_sids), len(source_sids), len({r['sid'] for r in samples}), time.time() - t0))
    epochs = {k: int(v) for k, v in (x.split(':') for x in (a.trial_epochs if a.trial else a.epochs).split(','))}
    lrs = {k: float(v) for k, v in (x.split(':') for x in a.lr.split(','))}

    # ---------- pilot trial ----------
    if a.trial:
        tdir = os.path.join(out, 'trial'); os.makedirs(tdir, exist_ok=True)
        sids = sid_list(0, 5); say('pilot trial: N=5 seed 0, runs = %s' % sids)
        zero = L.single_rmse_quick(U_load(base_state, dev), norm, cache, test_sids, dev, masks, D)
        curves = {}
        for method in a.methods.split(','):
            model, params = build_model(method, base_state, 0, dev); rows = []
            def cb(ep, loss, method=method, model=model, rows=rows):
                q = L.single_rmse_quick(model, norm, cache, test_sids, dev, masks, D)
                rows.append(dict(method=method, epoch=ep, train_loss=loss, rmse_active=q['active']['rmse'], rmse_land=q['land']['rmse'], mae_active=q['active']['mae'], mae_land=q['land']['mae']))
                if ep % 5 == 0 or ep == 1: say('  %s ep %d loss %.3g | single-step RMSE active %.4f land %.4f' % (method, ep, loss, q['active']['rmse'], q['land']['rmse']))
            say('%s: %d epochs, lr %g, trainable params %d' % (method, epochs[method], lrs[method], sum(p.numel() for p in params)))
            hist, sec = train_fixed(model, params, lrs[method], epochs[method], sids, norm, mask_t, dev, 0, epoch_cb=cb)
            curves[method] = rows; append_rows(os.path.join(tdir, 'trial_curve.csv'), rows); say('%s done, training %s' % (method, hms(sec)))
        json.dump(dict(zero_shot=zero, ref_exp1=dict(active=a.ref_exp1_active, land=a.ref_exp1_land), sids=sids, epochs=epochs, lr=lrs, gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else dev),
                  open(os.path.join(tdir, 'trial_info.json'), 'w'), indent=1)
        plot_trial(curves, zero, a, tdir, fold); say('pilot trial done -> %s' % tdir); return

    # ---------- main run ----------
    man_path = os.path.join(out, 'manifest.csv'); man = load_manifest(man_path)
    def run_eval(cfg, model, do_source):
        te = time.time()
        rows, trows, hrows, srows = L.evaluate_many(model, norm, cache, test_sids, source_sids, dev, masks, D, do_source=do_source, do_persistence=False)
        for x in rows: x['config'] = cfg
        for x in trows: x['config'] = cfg
        for x in hrows: x['config'] = cfg
        for x in srows: x['config'] = cfg
        append_rows(os.path.join(out, 'metrics_test.csv'), rows); append_rows(os.path.join(out, 'per_t.csv'), trows)
        append_rows(os.path.join(out, 'hmax.csv'), hrows); append_rows(os.path.join(out, 'metrics_source.csv'), srows)
        act_s = L.pooled([r for r in rows if r['mode'] == 'single' and r['mask'] == 'active'])['rmse']
        act_r = L.pooled([r for r in rows if r['mode'] == 'rollout' and r['mask'] == 'active'])['rmse']
        return time.time() - te, act_s, act_r
    if a.eval_zero and 'zero' not in man:
        say('N=0 zero-shot evaluation (Experiment-2 model; incl. pre-fine-tuning baseline on 50 source runs)...')
        model = U_load(base_state, dev); es, s1, r1 = run_eval('zero', model, True)
        pers_rows = []
        for s in test_sids:
            rows, _, _ = L.eval_run(model, norm, cache.get(s), dev, masks, D, modes=('persistence',), per_t=False, hmax=False)
            for x in rows: x.update(sid=s, config='persistence', **cache.meta[s]); pers_rows.append(x)
        append_rows(os.path.join(out, 'metrics_test.csv'), pers_rows)
        man['zero'] = dict(config='zero', method='zero', N=0, seed=-1, epochs=0, lr=0, sids='', status='done', train_sec=0, eval_sec=round(es, 1), final_loss='', weight_path='', finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_manifest(man_path, man); say('N=0: single-step RMSE %.4f rollout %.4f (eval %s)' % (s1, r1, hms(es)))
    configs = [(m, N, sd) for m in a.methods.split(',') for N in [int(x) for x in a.n_list.split(',')] for sd in [int(x) for x in a.seeds.split(',')]]
    todo = [c for c in configs if man.get('%s_N%d_s%d' % c, {}).get('status') != 'done']
    say('%d configs, %d to run (%d already done)' % (len(configs), len(todo), len(configs) - len(todo)))
    T0 = time.time(); done_now = 0
    for method, N, seed in todo:
        cfg = '%s_N%d_s%d' % (method, N, seed); sids = sid_list(seed, N); tc = time.time()
        model, params = build_model(method, base_state, seed, dev)
        hist, tsec = train_fixed(model, params, lrs[method], epochs[method], sids, norm, mask_t, dev, seed)
        append_rows(os.path.join(out, 'train_log.csv'), [dict(config=cfg, epoch=i + 1, loss=v) for i, v in enumerate(hist)])
        esec, s1, r1 = run_eval(cfg, model, do_source=(method != 'C' and not a.no_source))
        wpath = ''
        if (method, N, seed) in KEEP_WEIGHTS:
            os.makedirs(os.path.join(out, 'weights'), exist_ok=True); wpath = os.path.join('weights', cfg + '.pt')
            torch.save(dict(model=model.state_dict(), stats=stats, config=cfg, sids=sids, epochs=epochs[method], lr=lrs[method]), os.path.join(out, wpath))
        man[cfg] = dict(config=cfg, method=method, N=N, seed=seed, epochs=epochs[method], lr=lrs[method], sids=';'.join(sids), status='done',
                        train_sec=round(tsec, 1), eval_sec=round(esec, 1), final_loss='%.6g' % hist[-1], weight_path=wpath, finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_manifest(man_path, man); done_now += 1
        el = time.time() - T0; eta = el / done_now * (len(todo) - done_now)
        say('%-12s train %s eval %s | single-step RMSE %.4f rollout %.4f | progress %d/%d elapsed %s ETA %s' % (cfg, hms(tsec), hms(esec), s1, r1, done_now, len(todo), hms(el), hms(eta)))
        del model; torch.cuda.empty_cache() if dev == 'cuda' else None
    say('fold complete: %d configs, this session %s -> %s' % (len(configs), hms(time.time() - T0), out))


def U_load(state, dev):
    m = U.UNetV1(); m.load_state_dict(state); return m.to(dev).eval()


def plot_trial(curves, zero, a, tdir, fold):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L.setup_fonts()                      # CJK font: auto-install / fall back to English (PLAN §8.19)
    T = L.T
    COL = dict(B='#2a78d6', Bp='#1baf7a', C='#eb6834')
    CN = dict(B=T('B full fine-tune', 'B full fine-tune'), Bp=T("B' frozen encoder", "B' frozen encoder"), C=T('C from scratch', 'C from scratch'))
    lrs = dict(x.split(':') for x in a.lr.split(','))
    for k, key, ref in (('active', 'rmse_active', a.ref_exp1_active), ('land', 'rmse_land', a.ref_exp1_land)):
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#fcfcfb')
        for m, rows in curves.items():
            ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], color=COL[m], lw=2, label='%s(lr %g)' % (CN[m], float(lrs[m])))
        ax.axhline(zero[k]['rmse'], color='#898781', ls='--', lw=1, label=T('N=0 zero-shot %.4f', 'N=0 zero-shot %.4f') % zero[k]['rmse'])
        if ref: ax.axhline(ref, color='#4a3aa7', ls=':', lw=1, label=T('Exp-1 same-event %.4f', 'Exp-1 same-event %.4f') % ref)
        ax.set_xlabel(T('epoch', 'epoch')); ax.set_yscale('log')
        ax.set_ylabel(T('single-step RMSE (m) · %s', 'single-step RMSE (m) · %s') % (T('active cells', 'active cells') if k == 'active' else T('land cells', 'land cells')))
        ax.set_title(T('Epoch trial, fold %s, N=5, seed 0 (10 test runs)', 'Epoch trial, fold %s, N=5, seed 0 (10 test runs)') % fold, loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.grid(axis='y', color='#e1e0d9'); ax.set_facecolor('#fcfcfb')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(tdir, 'trial_curve_%s.png' % k), dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
