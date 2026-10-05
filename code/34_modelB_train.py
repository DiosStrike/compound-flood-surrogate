#!/usr/bin/env python3
"""34_modelB_train.py -- Plan B U-Net v1 training loop (docs/PLAN.md v9 §6). Reads data only; writes only to the --out directory.

Loss: MSE on active cells only (loss_mask); target h(t) is not normalized (m).
Optimization: Adam lr 1e-3; batch 8; at most 50 epochs; early stop when val loss does not improve for 10 epochs; seed 42.
Sampling: every training and validation run uses the fixed t = 73, 80, ..., 234 (--t-list 73:234:7, 24 in total).
Normalization statistics are computed from training runs only (--stats gives a precomputed file, otherwise computed and saved to norm_stats.json) and stored with the checkpoint.

Outputs (--out):
  norm_stats.json        normalization statistics
  checkpoint_last.pt     saved at the end of every epoch (model, optimizer, epoch, best value, early-stop counter) -> --resume continues from here
  model_best.pt          saved when val loss reaches a new low; at the end of training this is the best model
  train_log.csv          one row per epoch: epoch, train_loss, val_loss, best_val_loss, is_best, bad_epochs, train_sec, val_sec, epoch_sec, n_train_samples, n_val_samples, finished_at
  train_steps.json       per-step records, written only with --max-steps (smoke test)
stdout progress (since 2026-09-16; output only, no change to training logic / random order / checkpoint format):
  one line every 100 steps: "[time] ep cur/max step cur/steps_this_epoch | epoch elapsed ... remaining ... | total elapsed ... total remaining ... | loss(last 100 steps) ... | ... s/step"
  (in the actual output the "last N steps" and "s/step" tokens, the header step count, the "non-finite" marker and the
   start/resume/finished markers stay in the original Chinese, because code/37_watch_training.py, code/40_watch_colab.py
   and code/44_exp2_summary.py parse them)
  one timestamped line each for epoch start, validation start / end, checkpoint save, early stop and training end. Watch: python3 code/37_watch_training.py --exp <name>

Usage:
  Exp. 1: python3 -u code/34_modelB_train.py --exp exp1 --out result/models/exp1 --stats result/models/exp1/norm_stats.json [--resume]
  statistics only: python3 code/34_modelB_train.py --exp exp1 --out result/models/exp1 --stats-only
  smoke test: python3 code/34_modelB_train.py --train matthew_s041 --max-steps 300 --t-list all --out <dir>
"""
import os, sys, json, time, argparse, resource, importlib.util, csv
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


D = _imp('mb_data', '32_modelB_data.py'); U = _imp('mb_unet', '33_modelB_unet.py')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
LOG_COLS = ['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'is_best', 'bad_epochs', 'train_sec', 'val_sec', 'epoch_sec',
            'n_train_samples', 'n_val_samples', 'finished_at']


def masked_mse(pred, target, mask):
    """pred / target (B, 195, 255); mask (195, 255) bool; averaged over active cells only."""
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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9      # on macOS the unit is bytes


def mps_gb():
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def exp_sids(exp):
    """Exp. 1: split_A of the six events combined, train 360 / val 66 (reads the split_A column of data/modelB_v1/index.csv; no new split)."""
    if exp != 'exp1':
        raise SystemExit('only --exp exp1 is supported')
    idx = D.load_index(); idx = idx[idx.event.isin(EVENTS)]
    tr = idx[idx.split_A == 'train'].sid.tolist(); va = idx[idx.split_A == 'val'].sid.tolist()
    assert len(tr) == 360 and len(va) == 66, (len(tr), len(va))
    return tr, va


@torch.no_grad()
def evaluate_loss(model, sampler, mask, device):
    model.eval(); tot = 0.0; n = 0
    for x, y in sampler:
        p = model(torch.from_numpy(x).to(device))
        tot += float(masked_mse(p, torch.from_numpy(y).to(device), mask)) * len(x); n += len(x)
    return tot / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='', help='exp1 = split_A of six events combined (train 360 / val 66)')
    ap.add_argument('--train', default='', help='comma-separated training sids (alternative to --exp)')
    ap.add_argument('--val', default='', help='comma-separated validation sids')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stats', default='', help='precomputed norm_stats.json; if empty, computed from training runs')
    ap.add_argument('--stats-only', action='store_true', help='only compute and save normalization statistics')
    ap.add_argument('--resume', action='store_true', help='resume from --out/checkpoint_last.pt')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--t-list', default='73:234:7', help="training / validation target times, 'start:stop:step' or all")
    ap.add_argument('--max-steps', type=int, default=0, help='if >0, run only this many steps (smoke test)')
    ap.add_argument('--buffer-runs', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    if a.exp:
        train_sids, val_sids = exp_sids(a.exp)
    else:
        train_sids = [s for s in a.train.split(',') if s]; val_sids = [s for s in a.val.split(',') if s]
    st = D.load_static()

    # normalization statistics: training set only
    stats_path = a.stats or os.path.join(a.out, 'norm_stats.json')
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        assert sorted(stats['train_sids']) == sorted(train_sids), 'training runs of the normalization statistics differ from this training set'
        print('loaded normalization statistics %s' % stats_path, flush=True)
    else:
        t0 = time.time(); stats = D.compute_stats(train_sids, st); stats['seconds'] = round(time.time() - t0, 1)
        json.dump(stats, open(stats_path, 'w'), indent=1)
        print('computed normalization statistics: %d training runs, %.1f s -> %s' % (len(train_sids), stats['seconds'], stats_path), flush=True)
    if a.stats_only:
        return

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    norm = D.Normalizer(stats, st)
    mask = torch.from_numpy(st['loss_mask']).to(dev)
    model = U.UNetV1().to(dev); npar = U.n_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ts = D.parse_t_list(a.t_list)
    train_s = D.RunBufferSampler(train_sids, norm, a.batch_size, a.buffer_runs, None, seed=a.seed, t_list=ts)
    val_s = D.RunBufferSampler(val_sids, norm, a.batch_size, a.buffer_runs, None, seed=a.seed, t_list=ts, shuffle=False) if val_sids else None

    last_path = os.path.join(a.out, 'checkpoint_last.pt'); best_path = os.path.join(a.out, 'model_best.pt')
    log_path = os.path.join(a.out, 'train_log.csv')
    start_ep, best, bad, step = 1, float('inf'), 0, 0
    t_run0 = time.time(); prior_sec = 0.0; prior_ep_secs = []      # only for progress display
    if a.resume:
        if not os.path.exists(last_path):
            raise SystemExit('--resume given but %s not found' % last_path)
        ck = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start_ep, best, bad, step = ck['epoch'] + 1, ck['best_val_loss'], ck['bad_epochs'], ck['step']
        torch.set_rng_state(ck['torch_rng'])
        if ck.get('stopped') and not (ck['stopped'] == 'max_epochs' and a.epochs > ck['epoch']):
            print('finished previously (%s), nothing to resume' % ck['stopped']); return
        # keep train_log.csv only up to the checkpoint epoch (drop rows written after an interruption)
        if os.path.exists(log_path):
            rows = [r for r in csv.DictReader(open(log_path)) if int(r['epoch']) <= ck['epoch']]
            with open(log_path, 'w', newline='') as f:
                w = csv.DictWriter(f, LOG_COLS); w.writeheader(); w.writerows(rows)
            prior_ep_secs = [float(r['epoch_sec']) for r in rows if r.get('epoch_sec')]; prior_sec = sum(prior_ep_secs)
        print('resume: starting from epoch %d (best_val_loss %.6g, bad_epochs %d)' % (start_ep, best, bad), flush=True)
    elif os.path.exists(last_path):
        raise SystemExit('%s already exists: add --resume to continue; to start over, change --out or delete the old outputs first' % last_path)
    else:
        with open(log_path, 'w', newline='') as f:
            csv.DictWriter(f, LOG_COLS).writeheader()

    print('UNetV1 params %d; device %s; train %d run × %d t = %d samples / %d steps; val %d run; batch %d; epochs ≤ %d; patience %d' % (
        npar, dev, len(train_sids), len(ts), len(train_sids) * len(ts), train_s.steps_per_epoch(), len(val_sids),
        a.batch_size, a.epochs, a.patience), flush=True)
    steps_log = []; stopped = None
    n_steps = train_s.steps_per_epoch(); ep_secs = list(prior_ep_secs); val_secs = []
    for ep in range(start_ep, a.epochs + 1):
        te = time.time(); model.train(); train_s.set_epoch(ep); losses = []; t_prev = time.time()
        say('epoch %d / %d started (this epoch %d steps)' % (ep, a.epochs, n_steps))
        k = 0
        for x, y in train_s:
            xt = torch.from_numpy(x).to(dev); yt = torch.from_numpy(y).to(dev)
            loss = masked_mse(model(xt), yt, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            lv = loss.item(); step += 1; k += 1; losses.append(lv)
            now = time.time(); dt = now - t_prev; t_prev = now
            if a.max_steps:
                steps_log.append(dict(step=step, epoch=ep, loss=lv, sec=dt, rss_peak_gb=rss_gb(), mps_gb=mps_gb()))
            if k % a.log_every == 0:
                ep_used = now - te; sps = ep_used / k
                val_est = (sum(val_secs) / len(val_secs)) if val_secs else 0.0
                ep_left = (n_steps - k) * sps + val_est
                per_ep = (sum(ep_secs) / len(ep_secs)) if ep_secs else n_steps * sps + val_est
                all_left = ep_left + (a.epochs - ep) * per_ep
                say('ep %d/%d step %d/%d | epoch elapsed %s remaining %s | total elapsed %s total remaining %s (if all %d epochs run) | loss(last %d steps) %.6f | %.3f s/step | mps %.2f GB' % (
                    ep, a.epochs, k, n_steps, hms(ep_used), hms(ep_left), hms(prior_sec + now - t_run0), hms(all_left), a.epochs,
                    min(a.log_every, len(losses)), float(np.mean(losses[-a.log_every:])), sps, mps_gb()))
            if not np.isfinite(lv):
                stopped = 'loss non-finite (step %d)' % step; break
            if a.max_steps and step >= a.max_steps:
                stopped = 'max_steps'; break
        train_sec = time.time() - te
        if val_s:
            say('Epoch %d training part finished (%s); validation started (%d runs)' % (ep, hms(train_sec), len(val_sids)))
        tv = time.time(); val_loss, nval = (evaluate_loss(model, val_s, mask, dev) if val_s else (float('nan'), 0))
        val_sec = time.time() - tv
        if val_s:
            val_secs.append(val_sec); say('Validation finished: val_loss %.6f (%d samples, %s)' % (val_loss, nval, hms(val_sec)))
        is_best = bool(val_s) and val_loss < best
        if is_best:
            best = val_loss; bad = 0
            torch.save(dict(model=model.state_dict(), stats=stats, epoch=ep, val_loss=val_loss, t_list=ts), best_path)
            say('Saved best model %s (epoch %d, val %.6f)' % (best_path, ep, val_loss))
        elif val_s:
            bad += 1
        if val_s and bad >= a.patience:
            stopped = stopped or 'early_stop(val loss did not improve for %d epochs)' % a.patience
        if ep == a.epochs and not stopped:
            stopped = 'max_epochs'
        row = dict(epoch=ep, train_loss=float(np.mean(losses)) if losses else float('nan'), val_loss=val_loss, best_val_loss=best,
                   is_best=is_best, bad_epochs=bad, train_sec=round(train_sec, 1), val_sec=round(val_sec, 1),
                   epoch_sec=round(time.time() - te, 1), n_train_samples=len(losses) and len(train_sids) * len(ts), n_val_samples=nval,
                   finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, LOG_COLS).writerow(row)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), stats=stats, epoch=ep, step=step,
                        best_val_loss=best, bad_epochs=bad, torch_rng=torch.get_rng_state(), t_list=ts, args=vars(a),
                        stopped=stopped), last_path)
        ep_secs.append(row['epoch_sec'])
        say('Saved checkpoint %s (epoch %d; this epoch %s; total elapsed %s)' % (last_path, ep, hms(row['epoch_sec']), hms(prior_sec + time.time() - t_run0)))
        print('epoch %d  train %.6f  val %.6f  best %.6f%s  bad %d  %.0f s(train %.0f / val %.0f)  rss_peak %.2f GB  mps %.2f GB' % (
            ep, row['train_loss'], val_loss, best, ' *' if is_best else '', bad, row['epoch_sec'], train_sec, val_sec, rss_gb(), mps_gb()), flush=True)
        if stopped:
            if stopped.startswith('early_stop'):
                say('Early stop: val loss has not reached a new low for %d consecutive epochs (best %.6f)' % (a.patience, best))
            break
    if steps_log:
        json.dump(steps_log, open(os.path.join(a.out, 'train_steps.json'), 'w'), indent=1)
    if val_s and os.path.exists(best_path):
        bk = torch.load(best_path, map_location='cpu', weights_only=False)
        print('Finished (%s): best model %s (epoch %d, val %.6f)' % (stopped, best_path, bk['epoch'], bk['val_loss']), flush=True)
    else:
        torch.save(dict(model=model.state_dict(), stats=stats, step=step), os.path.join(a.out, 'model.pt'))
        print('Finished (%s): no validation set, saved model.pt' % stopped, flush=True)
    say('training finished (%s); total elapsed %s' % (stopped, hms(prior_sec + time.time() - t_run0)))


if __name__ == '__main__':
    main()
