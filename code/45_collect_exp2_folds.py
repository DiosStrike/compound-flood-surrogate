#!/usr/bin/env python3
"""45_collect_exp2_folds.py -- organize the 6-fold results of Experiment 2 (or Experiment 3 with --exp3), downloaded from Drive and unpacked, into result/models/<exp>/folds/. Reads the source directory only; writes only the target directory.

Source directory (--src): the unpacked folder; the script searches it (up to 4 levels deep) for `fold_<event>` directories, one per event (6 in total).
Copied per fold: train_log.csv, train_stdout.log, fold_info.json, model_best.pt, and the whole eval_test/ directory.
**Skips checkpoint_last.pt** (about 93 MB per fold, not needed for the summary) and other directories such as exp2_smoke/.
Validate first, then copy; if any check fails, list the problems and stop (exit code 1) without copying partial results:
  1. all 6 folds present (irma / matthew / ian / milton / dorian / beryl);
  2. every fold has train_log.csv, fold_info.json, model_best.pt and the 6 files in eval_test/;
  3. eval_test/metrics_per_event.csv has the count columns tp_0.1 / fp_0.1 / fn_0.1 etc. (requires code/39 from 2026-09-17 or later);
  4. every fold's eval_test/metrics_per_run.csv has exactly 10 test runs, all from that held-out event.
When done, reports the number of copied files and disk usage, and suggests running code/44_exp2_summary.py next.

Usage:
  python3 code/45_collect_exp2_folds.py --src ~/Downloads/exp2            # organize
  python3 code/45_collect_exp2_folds.py --src ~/Downloads/exp2 --check-only   # validate only, no copying
"""
import argparse, csv, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
EVAL_FILES = ['metrics_per_event.csv', 'metrics_per_run.csv', 'per_t.csv', 'hmax_rollout_per_run.csv', 's041_peak_maps.npz', 'eval_summary.json']
FOLD_FILES = ['train_log.csv', 'train_stdout.log', 'fold_info.json', 'model_best.pt']
SKIP = {'checkpoint_last.pt'}


def find_folds(src):
    """Find fold_<event> directories up to 4 levels below src."""
    found = {}
    src = os.path.abspath(os.path.expanduser(src))
    base_depth = src.rstrip(os.sep).count(os.sep)
    for d, dirs, _ in os.walk(src):
        if d.count(os.sep) - base_depth > 4:
            dirs[:] = []; continue
        for name in list(dirs):
            if name.startswith('fold_') and name[5:] in EVENTS and 'smoke' not in d:
                found.setdefault(name[5:], os.path.join(d, name))
    return found


def check(folds):
    probs = []
    for e in EVENTS:
        d = folds.get(e)
        if not d:
            probs.append('missing fold fold_%s (not found under --src)' % e); continue
        for f in FOLD_FILES:
            if not os.path.exists(os.path.join(d, f)):
                probs.append('%s: missing %s' % (e, f))
        ed = os.path.join(d, 'eval_test')
        if not os.path.isdir(ed):
            probs.append('%s: missing eval_test/ (training finished but not evaluated? run step 8 in that fold notebook)' % e); continue
        for f in EVAL_FILES:
            if not os.path.exists(os.path.join(ed, f)):
                probs.append('%s: missing eval_test/%s' % (e, f))
        p = os.path.join(ed, 'metrics_per_event.csv')
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            miss = [c for c in ('tp_0.1', 'fp_0.1', 'fn_0.1', 'tp_0.3', 'fp_0.3', 'fn_0.3') if rows and c not in rows[0]]
            if miss:
                probs.append('%s: eval_test lacks count columns %s (re-run step 8 evaluation with code/39 from 2026-09-17 or later)' % (e, ','.join(miss)))
        p = os.path.join(ed, 'metrics_per_run.csv')
        if os.path.exists(p):
            sids = sorted({r['sid'] for r in csv.DictReader(open(p))})
            if len(sids) != 10:
                probs.append('%s: number of test runs %d != 10 (%s)' % (e, len(sids), ','.join(sids[:12])))
            bad = [s for s in sids if not s.startswith(e + '_')]
            if bad:
                probs.append('%s: eval_test contains runs from other events: %s' % (e, ','.join(bad)))
    return probs


def dir_size(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


EXP3_FILES = ['manifest.csv', 'metrics_test.csv', 'metrics_source.csv', 'per_t.csv', 'hmax.csv', 'train_log.csv']


def check_exp3(folds):
    """Experiment 3: all 6 folds present; manifest has zero + B/Bp/C x N(1,2,3,5,10) x seed(0,1,2) = 46 configs done; all files present."""
    probs = []
    for e in EVENTS:
        d = folds.get(e)
        if not d: probs.append('missing fold fold_%s' % e); continue
        for f in EXP3_FILES:
            if not os.path.exists(os.path.join(d, f)): probs.append('%s: missing %s' % (e, f))
        mp = os.path.join(d, 'manifest.csv')
        if os.path.exists(mp):
            man = {r['config']: r for r in csv.DictReader(open(mp))}
            need = ['zero'] + ['%s_N%d_s%d' % (m, n, s) for m in ('B', 'Bp', 'C') for n in (1, 2, 3, 5, 10) for s in (0, 1, 2)]
            miss = [c for c in need if man.get(c, {}).get('status') != 'done']
            if miss: probs.append('%s: manifest missing %d configs (e.g. %s)' % (e, len(miss), ','.join(miss[:4])))
    return probs


def collect_exp3(a, folds):
    t0 = time.time(); n = 0
    for e in EVENTS:
        src = folds[e]; dst = os.path.join(a.dest, 'fold_%s' % e); os.makedirs(dst, exist_ok=True)
        for f in EXP3_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f)); n += 1
        for sub in ('weights', 'trial'):
            sp = os.path.join(src, sub)
            if os.path.isdir(sp):
                os.makedirs(os.path.join(dst, sub), exist_ok=True)
                for f in os.listdir(sp):
                    if not f.startswith('.'): shutil.copy2(os.path.join(sp, f), os.path.join(dst, sub, f)); n += 1
        print('  fold_%-8s %.1f MB' % (e, dir_size(dst) / 1e6), flush=True)
    print('Done: %d files -> %s, total %.1f MB, took %.0f s' % (n, a.dest, dir_size(a.dest) / 1e6, time.time() - t0))
    print('Next: python3 code/49_exp3_summary.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder downloaded from Drive and unpacked (contains fold_<event>/)')
    ap.add_argument('--dest', default=None)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--exp3', action='store_true', help='organize Experiment 3 results (default: Experiment 2)')
    a = ap.parse_args()
    a.dest = a.dest or os.path.join(ROOT, 'result', 'models', 'exp3' if a.exp3 else 'exp2', 'folds')
    folds = find_folds(a.src)
    if a.exp3:
        print('Under %s found %d folds: %s' % (os.path.abspath(os.path.expanduser(a.src)), len(folds), ', '.join(sorted(folds))), flush=True)
        probs = check_exp3(folds)
        if probs:
            print('Stopped: the following items are missing or invalid; no files were copied:'); [print('  ' + p) for p in probs]; sys.exit(1)
        print('Validation passed: all 6 folds present, 46 configs done per fold.')
        if not a.check_only: collect_exp3(a, folds)
        return
    print('Under %s found %d folds: %s' % (os.path.abspath(os.path.expanduser(a.src)), len(folds), ', '.join(sorted(folds))), flush=True)
    probs = check(folds)
    if probs:
        print('Stopped: the following items are missing or invalid; no files were copied:')
        for p in probs:
            print('  ' + p)
        sys.exit(1)
    print('Validation passed: all 6 folds present, eval_test has count columns, 10 test runs per fold.', flush=True)
    if a.check_only:
        return
    t0 = time.time(); n_files = 0
    for e in EVENTS:
        src = folds[e]; dst = os.path.join(a.dest, 'fold_%s' % e)
        os.makedirs(dst, exist_ok=True)
        for f in FOLD_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f)); n_files += 1
        ed = os.path.join(dst, 'eval_test'); os.makedirs(ed, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(src, 'eval_test'))):
            if f in SKIP or f.startswith('.'):
                continue
            sp = os.path.join(src, 'eval_test', f)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(ed, f)); n_files += 1
        print('  fold_%-8s %d files, %.1f MB' % (e, n_files, dir_size(dst) / 1e6), flush=True)
    total = dir_size(a.dest)
    print('Done: %d files -> %s, total %.1f MB, took %.0f s (checkpoint_last.pt skipped)' % (n_files, a.dest, total / 1e6, time.time() - t0))
    print('Next: python3 code/44_exp2_summary.py   # generates result/models/exp2/ and result/figures/exp2/')


if __name__ == '__main__':
    main()
