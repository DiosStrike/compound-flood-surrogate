#!/usr/bin/env python3
"""41_pack_modelB_data.py -- data packing and md5 verification for running Exp. 2 on Colab (docs/PLAN.md v10 §7). Standard library only.

pack (local machine; run after Exp. 1 finishes and with user approval; estimated 2-5 min, ~2.6 GB extra disk, first estimate):
  1. Compute the md5 of every file under data/modelB_v1/ (static.npz, index.csv, preprocess_log.json, <event>/<sid>.npz × 486),
     write modelB_v1_manifest.csv (relpath, bytes, md5); and check that the md5 of the 486 npz files matches the md5 column of index.csv.
  2. Pack into an uncompressed tar: modelB_v1.tar (npz is already compressed, recompressing gains nothing); paths inside the tar are modelB_v1/...
  3. Compute the md5 of the tar itself into modelB_v1.tar.md5; copy the code needed for Exp. 2 (code/32 33 35 38 39 40 41 + colab_exp2.ipynb) to <out>/code/.
  Does not modify data/; the output directory defaults to ~/Flood2_colab_upload/ (outside the project).
verify (on Colab after extraction, or any local extraction directory):
  Check size and md5 of every file against the manifest, and the 486 npz against the md5 column of index.csv; report missing / extra / mismatched files; exit code 0 if all match, otherwise 1.

Usage:
  python3 code/41_pack_modelB_data.py pack [--out ~/Flood2_colab_upload]
  python3 code/41_pack_modelB_data.py verify --data-dir /content/Flood_2.0/data/modelB_v1 --manifest /content/drive/MyDrive/Flood2/modelB_v1_manifest.csv
"""
import argparse, csv, hashlib, os, shutil, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'modelB_v1')
CODE_FILES = ['32_modelB_data.py', '33_modelB_unet.py', '35_modelB_eval.py', '38_exp2_train.py', '39_exp2_eval.py',
              '40_watch_colab.py', '41_pack_modelB_data.py', '43_exp1_figures.py', '44_exp2_summary.py', 'colab_exp2.ipynb',
              'colab_exp2_matthew.ipynb', 'colab_exp2_ian.ipynb', 'colab_exp2_milton.ipynb', 'colab_exp2_dorian.ipynb',
              'colab_exp2_beryl.ipynb', 'colab_exp2_status.ipynb', '45_collect_exp2_folds.py', '47_exp3_train.py', '48_exp3_eval_lib.py',
              '49_exp3_summary.py', 'colab_exp3_trial.ipynb', 'colab_exp3_irma.ipynb', 'colab_exp3_matthew.ipynb', 'colab_exp3_ian.ipynb',
              'colab_exp3_milton.ipynb', 'colab_exp3_dorian.ipynb', 'colab_exp3_beryl.ipynb', 'colab_exp3_status.ipynb']


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def list_files(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.startswith('.'):
                continue
            out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def check_index(data_dir, md5_of):
    """Check that the md5 of the 486 npz files matches the md5 column of index.csv; return a list of problems."""
    probs = []
    rows = list(csv.DictReader(open(os.path.join(data_dir, 'index.csv'))))
    if len(rows) != 486:
        probs.append('index.csv has %d rows ≠ 486' % len(rows))
    for r in rows:
        rel = os.path.join(r['event'], r['sid'] + '.npz')
        if rel not in md5_of:
            probs.append('missing %s' % rel)
        elif md5_of[rel] != r['md5']:
            probs.append('md5 does not match index.csv: %s' % rel)
    return probs


def pack(a):
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    t0 = time.time(); files = list_files(DATA)
    total = sum(os.path.getsize(os.path.join(DATA, f)) for f in files)
    free = shutil.disk_usage(out).free
    print('%d files to pack, %.2f GB; %.1f GB free in output directory' % (len(files), total / 1e9, free / 1e9), flush=True)
    if free < total * 1.1:
        sys.exit('Stopped: not enough free space in output directory')
    md5_of = {}
    man = os.path.join(out, 'modelB_v1_manifest.csv')
    with open(man, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['relpath', 'bytes', 'md5'])
        for i, rel in enumerate(files, 1):
            p = os.path.join(DATA, rel); md5_of[rel] = md5_file(p)
            w.writerow([rel, os.path.getsize(p), md5_of[rel]])
            if i % 50 == 0 or i == len(files):
                print('  md5 %d/%d  %.0f s' % (i, len(files), time.time() - t0), flush=True)
    probs = check_index(DATA, md5_of)
    if probs:
        sys.exit('Stopped: local data does not match index.csv: %s' % probs[:10])
    tar_path = os.path.join(out, 'modelB_v1.tar')
    with tarfile.open(tar_path, 'w') as tar:
        for i, rel in enumerate(files, 1):
            tar.add(os.path.join(DATA, rel), arcname=os.path.join('modelB_v1', rel), recursive=False)
            if i % 50 == 0 or i == len(files):
                print('  tar %d/%d  %.0f s' % (i, len(files), time.time() - t0), flush=True)
    tmd5 = md5_file(tar_path)
    open(tar_path + '.md5', 'w').write('%s  modelB_v1.tar\n' % tmd5)
    cdir = os.path.join(out, 'code'); os.makedirs(cdir, exist_ok=True)
    for f in CODE_FILES:
        shutil.copy2(os.path.join(ROOT, 'code', f), os.path.join(cdir, f))
    print('Done: %s (%.2f GB, md5 %s); manifest %s; %d code files -> %s; elapsed %.0f s' % (
        tar_path, os.path.getsize(tar_path) / 1e9, tmd5, man, len(CODE_FILES), cdir, time.time() - t0), flush=True)


def verify(a):
    t0 = time.time(); data_dir = a.data_dir
    expected = {r['relpath']: r for r in csv.DictReader(open(a.manifest))}
    present = set(list_files(data_dir)); probs = []; md5_of = {}
    for i, (rel, r) in enumerate(sorted(expected.items()), 1):
        p = os.path.join(data_dir, rel)
        if rel not in present:
            probs.append('missing %s' % rel); continue
        if os.path.getsize(p) != int(r['bytes']):
            probs.append('size mismatch %s' % rel); continue
        md5_of[rel] = md5_file(p)
        if md5_of[rel] != r['md5']:
            probs.append('md5 mismatch %s' % rel)
        if i % 100 == 0 or i == len(expected):
            print('  verify %d/%d  %.0f s' % (i, len(expected), time.time() - t0), flush=True)
    extra = sorted(present - set(expected))
    probs += ['extra file %s' % e for e in extra]
    probs += check_index(data_dir, md5_of)
    if probs:
        print('Verification FAILED (%d issues):' % len(probs)); [print('  ' + p) for p in probs[:30]]
        sys.exit(1)
    print('Verification passed: size and md5 of all %d files match, 486 npz match index.csv; elapsed %.0f s' % (len(expected), time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack'); p.add_argument('--out', default='~/Flood2_colab_upload')
    v = sub.add_parser('verify'); v.add_argument('--data-dir', required=True); v.add_argument('--manifest', required=True)
    a = ap.parse_args()
    pack(a) if a.cmd == 'pack' else verify(a)


if __name__ == '__main__':
    main()
