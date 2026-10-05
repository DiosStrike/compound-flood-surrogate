#!/usr/bin/env python3
"""38_exp2_train.py -- Experiment 2 (6-fold leave-one-event-out cross-event) U-Net v1 training, targeting Colab GPU (docs/PLAN.md v10 §7).

Does not modify code/32-37: calls 32 (data) and 33 (U-Net) directly; the data directory is set via --data-dir on module 32's DATA.
Training logic, sampling, random order, checkpoint keys, train_log.csv columns and stdout progress format match code/34_modelB_train.py (v9).

Fold definition (--fold <event name>, that event is held out):
  train = split_A train of the other 5 events (300 runs); val = split_A val of the other 5 events (55 runs); test = split_A test of the held-out event (10 runs)
  normalization statistics are computed from this fold's 300 training runs only and stored in <fold dir>/norm_stats.json
Outputs: <out-root>/exp2/fold_<event>/ -- fold_info.json, norm_stats.json, checkpoint_last.pt, model_best.pt, train_log.csv
Device: auto cuda -> mps -> cpu (override with --device); prints the device and GPU model at startup.

Usage (Colab):
  smoke test: python3 code/38_exp2_train.py --fold matthew --data-dir /content/Flood_2.0/data/modelB_v1 --out-root /content/smoke --smoke
  training: python3 -u code/38_exp2_train.py --fold matthew --data-dir /content/Flood_2.0/data/modelB_v1 --out-root /content/drive/MyDrive/Flood2/models [--resume]
  evaluate after training: add --eval-after (calls the evaluation in code/39_exp2_eval.py, single-step + rollout, 10 test runs of the held-out event)
"""
import os, sys, json, time, argparse, resource, importlib.util, csv, platform
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
LOG_COLS = ['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'is_best', 'bad_epochs', 'train_sec', 'val_sec', 'epoch_sec',
            'n_train_samples', 'n_val_samples', 'finished_at']


def set_data_dir(path):
    """Point module 32's data directory to path (32's functions read the module variable DATA at call time)."""
    path = os.path.abspath(path)
    for f in ('static.npz', 'index.csv'):
        if not os.path.exists(os.path.join(path, f)):
            raise SystemExit('Data directory %s is missing %s' % (path, f))
    D.DATA = path
    return path


def pick_device(req=''):
    if req:
        return req
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def device_info(dev):
    info = dict(device=dev, torch=torch.__version__, python=platform.python_version(), host=platform.node())
    if dev.startswith('cuda'):
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    return info


def fold_sids(fold):
    """Return (train 300, val 55, test 10); reads the split_A column of index.csv, no new split is created."""
    if fold not in EVENTS:
        raise SystemExit('--fold must be one of %s' % EVENTS)
    idx = D.load_index(); idx = idx[idx.event.isin(EVENTS)]
    others = idx[idx.event != fold]; held = idx[idx.event == fold]
    tr = others[others.split_A == 'train'].sid.tolist(); va = others[others.split_A == 'val'].sid.tolist()
    te = held[held.split_A == 'test'].sid.tolist()
    assert (len(tr), len(va), len(te)) == (300, 55, 10), (len(tr), len(va), len(te))
    return tr, va, te


def masked_mse(pred, target, mask):
    m = mask.expand_as(pred)
    return ((pred - target) ** 2)[m].mean()


def now_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def hms(sec):
    sec = max(int(sec), 0)
    return '%d:%02d:%02d' % (sec // 3600, sec % 3600 // 60, sec % 60)


def say(msg):
    print('[%s] %s' % (now_str(), msg), flush=True)


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == 'darwin' else r * 1024 / 1e9     # bytes on macOS, KB on Linux


def dev_mem_gb(dev):
    if dev.startswith('cuda'):
        return torch.cuda.max_memory_allocated() / 1e9
    if dev == 'mps':
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


@torch.no_grad()
def evaluate_loss(model, sampler, mask, device):
    model.eval(); tot = 0.0; n = 0
    for x, y in sampler:
        p = model(torch.from_numpy(x).to(device))
        tot += float(masked_mse(p, torch.from_numpy(y).to(device), mask)) * len(x); n += len(x)
    return tot / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='exp2', choices=['exp2'])
    ap.add_argument('--fold', required=True, help='held-out event: ' + ' / '.join(EVENTS))
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data', 'modelB_v1'))
    ap.add_argument('--out-root', default=os.path.join(ROOT, 'result', 'models'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default='', help='empty = auto cuda -> mps -> cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=None, help='default 50; default 2 with --smoke')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--t-list', default='73:234:7')
    ap.add_argument('--buffer-runs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--smoke', action='store_true', help='smoke test: train 3 runs / val 1 run / default 2 epochs / one line every 5 steps, output to <out-root>/exp2_smoke/')
    ap.add_argument('--eval-after', action='store_true', help='after training, evaluate the 10 test runs of the held-out event (single-step + rollout)')
    a = ap.parse_args()

    set_data_dir(a.data_dir)
    dev = pick_device(a.device); dinfo = device_info(dev)
    say('Device: %s' % json.dumps(dinfo, ensure_ascii=False))
    if dev == 'cpu':
        say('!! Warning: no GPU detected, running on CPU (in Colab choose GPU under "Runtime -> Change runtime type")')
    train_sids, val_sids, test_sids = fold_sids(a.fold)
    if a.smoke:
        train_sids, val_sids = train_sids[:3], val_sids[:1]; a.log_every = 5
    if a.epochs is None:
        a.epochs = 2 if a.smoke else 50
    out = os.path.join(a.out_root, 'exp2_smoke' if a.smoke else 'exp2', 'fold_%s' % a.fold)
    os.makedirs(out, exist_ok=True)
    st = D.load_static()

    stats_path = os.path.join(out, 'norm_stats.json')
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        assert sorted(stats['train_sids']) == sorted(train_sids), 'training runs of the normalization statistics do not match this fold\'s training set'
        say('Loaded this fold\'s normalization statistics %s' % stats_path)
    else:
        say('Computing this fold\'s normalization statistics (%d training runs)...' % len(train_sids))
        t0 = time.time(); stats = D.compute_stats(train_sids, st); stats['seconds'] = round(time.time() - t0, 1)
        json.dump(stats, open(stats_path, 'w'), indent=1)
        say('Normalization statistics done: %.1f s -> %s' % (stats['seconds'], stats_path))

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    norm = D.Normalizer(stats, st)
    mask = torch.from_numpy(st['loss_mask']).to(dev)
    model = U.UNetV1().to(dev); npar = U.n_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ts = D.parse_t_list(a.t_list)
    train_s = D.RunBufferSampler(train_sids, norm, a.batch_size, a.buffer_runs, None, seed=a.seed, t_list=ts)
    val_s = D.RunBufferSampler(val_sids, norm, a.batch_size, a.buffer_runs, None, seed=a.seed, t_list=ts, shuffle=False)

    last_path = os.path.join(out, 'checkpoint_last.pt'); best_path = os.path.join(out, 'model_best.pt')
    log_path = os.path.join(out, 'train_log.csv')
    start_ep, best, bad, step = 1, float('inf'), 0, 0
    t_run0 = time.time(); prior_sec = 0.0; prior_ep_secs = []
    if a.resume:
        if not os.path.exists(last_path):
            raise SystemExit('--resume given but %s not found' % last_path)
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start_ep, best, bad, step = ck['epoch'] + 1, ck['best_val_loss'], ck['bad_epochs'], ck['step']
        torch.set_rng_state(ck['torch_rng'])
        if ck.get('stopped') and not (ck['stopped'] == 'max_epochs' and a.epochs > ck['epoch']):
            say('finished previously (%s), nothing to resume' % ck['stopped'])
            if a.eval_after:
                run_eval(a, out, test_sids, dev)
            return
        if os.path.exists(log_path):
            rows = [r for r in csv.DictReader(open(log_path)) if int(r['epoch']) <= ck['epoch']]
            with open(log_path, 'w', newline='') as f:
                w = csv.DictWriter(f, LOG_COLS); w.writeheader(); w.writerows(rows)
            prior_ep_secs = [float(r['epoch_sec']) for r in rows if r.get('epoch_sec')]; prior_sec = sum(prior_ep_secs)
        say('resume: starting from epoch %d (best_val_loss %.6g, bad_epochs %d)' % (start_ep, best, bad))
    elif os.path.exists(last_path):
        raise SystemExit('%s already exists: add --resume to continue; to start over, change --out-root or delete the old outputs first' % last_path)
    else:
        with open(log_path, 'w', newline='') as f:
            csv.DictWriter(f, LOG_COLS).writeheader()

    info_path = os.path.join(out, 'fold_info.json')
    info = json.load(open(info_path)) if os.path.exists(info_path) else dict(fold=a.fold, train_sids=train_sids, val_sids=val_sids, test_sids=test_sids, sessions=[])
    info['sessions'].append(dict(started_at=now_str(), resume=a.resume, start_epoch=start_ep, args=vars(a), **dinfo))
    json.dump(info, open(info_path, 'w'), indent=1, ensure_ascii=False)

    print('UNetV1 params %d; device %s; train %d run × %d t = %d samples / %d steps; val %d run; batch %d; epochs ≤ %d; patience %d' % (
        npar, dev, len(train_sids), len(ts), len(train_sids) * len(ts), train_s.steps_per_epoch(), len(val_sids),
        a.batch_size, a.epochs, a.patience), flush=True)
    stopped = None
    n_steps = train_s.steps_per_epoch(); ep_secs = list(prior_ep_secs); val_secs = []
    for ep in range(start_ep, a.epochs + 1):
        te = time.time(); model.train(); train_s.set_epoch(ep); losses = []
        say('epoch %d / %d started (this epoch %d steps)' % (ep, a.epochs, n_steps))
        k = 0
        for x, y in train_s:
            xt = torch.from_numpy(x).to(dev); yt = torch.from_numpy(y).to(dev)
            loss = masked_mse(model(xt), yt, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            lv = loss.item(); step += 1; k += 1; losses.append(lv)
            if k % a.log_every == 0:
                now = time.time(); ep_used = now - te; sps = ep_used / k
                val_est = (sum(val_secs) / len(val_secs)) if val_secs else 0.0
                ep_left = (n_steps - k) * sps + val_est
                per_ep = (sum(ep_secs) / len(ep_secs)) if ep_secs else n_steps * sps + val_est
                all_left = ep_left + (a.epochs - ep) * per_ep
                say('ep %d/%d step %d/%d | epoch elapsed %s remaining %s | total elapsed %s total remaining %s (if all %d epochs run) | loss(last %d steps) %.6f | %.3f s/step | GPU mem peak %.2f GB' % (
                    ep, a.epochs, k, n_steps, hms(ep_used), hms(ep_left), hms(prior_sec + now - t_run0), hms(all_left), a.epochs,
                    min(a.log_every, len(losses)), float(np.mean(losses[-a.log_every:])), sps, dev_mem_gb(dev)))
            if not np.isfinite(lv):
                stopped = 'loss non-finite (step %d)' % step; break
        train_sec = time.time() - te
        say('Epoch %d training part finished (%s); validation started (%d runs)' % (ep, hms(train_sec), len(val_sids)))
        tv = time.time(); val_loss, nval = evaluate_loss(model, val_s, mask, dev); val_sec = time.time() - tv
        val_secs.append(val_sec); say('Validation finished: val_loss %.6f (%d samples, %s)' % (val_loss, nval, hms(val_sec)))
        is_best = val_loss < best
        if is_best:
            best = val_loss; bad = 0
            torch.save(dict(model=model.state_dict(), stats=stats, epoch=ep, val_loss=val_loss, t_list=ts), best_path)
            say('Saved best model %s (epoch %d, val %.6f)' % (best_path, ep, val_loss))
        else:
            bad += 1
        if bad >= a.patience:
            stopped = stopped or 'early_stop(val loss did not improve for %d epochs)' % a.patience
        if ep == a.epochs and not stopped:
            stopped = 'max_epochs'
        row = dict(epoch=ep, train_loss=float(np.mean(losses)) if losses else float('nan'), val_loss=val_loss, best_val_loss=best,
                   is_best=is_best, bad_epochs=bad, train_sec=round(train_sec, 1), val_sec=round(val_sec, 1),
                   epoch_sec=round(time.time() - te, 1), n_train_samples=len(train_sids) * len(ts), n_val_samples=nval,
                   finished_at=now_str())
        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, LOG_COLS).writerow(row)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), stats=stats, epoch=ep, step=step,
                        best_val_loss=best, bad_epochs=bad, torch_rng=torch.get_rng_state(), t_list=ts, args=vars(a),
                        stopped=stopped), last_path)
        ep_secs.append(row['epoch_sec'])
        say('Saved checkpoint %s (epoch %d; this epoch %s; total elapsed %s)' % (last_path, ep, hms(row['epoch_sec']), hms(prior_sec + time.time() - t_run0)))
        print('epoch %d  train %.6f  val %.6f  best %.6f%s  bad %d  %.0f s (train %.0f / val %.0f)  rss_peak %.2f GB  GPU mem peak %.2f GB' % (
            ep, row['train_loss'], val_loss, best, ' *' if is_best else '', bad, row['epoch_sec'], train_sec, val_sec, rss_gb(), dev_mem_gb(dev)), flush=True)
        if stopped:
            if stopped.startswith('early_stop'):
                say('Early stop: val loss has not reached a new low for %d consecutive epochs (best %.6f)' % (a.patience, best))
            break
    bk = torch.load(best_path, map_location='cpu', weights_only=False)
    say('training finished (%s); best model epoch %d val %.6f; this session %s; cumulative %s' % (
        stopped, bk['epoch'], bk['val_loss'], hms(time.time() - t_run0), hms(prior_sec + time.time() - t_run0)))
    if a.eval_after:
        run_eval(a, out, test_sids, dev)


def run_eval(a, out, test_sids, dev):
    E = _imp('exp2_eval', '39_exp2_eval.py')
    E.evaluate_fold(out, test_sids, dev, D)


if __name__ == '__main__':
    main()
