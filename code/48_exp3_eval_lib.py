#!/usr/bin/env python3
"""48_exp3_eval_lib.py -- evaluation library for Experiment 3 (docs/PLAN.md v17 §8.12, §8.19). Does not modify 32-45.

Reuses: single-step / rollout inference from code/35 (single_step, rollout); metric definitions identical to code/39 / 42 (active / land cells, MAE / RMSE / CSI 0.1 / 0.3,
      per-run sa / ss / n / tp / fp / fn counts kept so that pooling is exact).
Adds: setup_fonts / T (automatic CJK font install with English fallback, §8.19), RunCache (evaluation runs kept in memory, read once), evaluate_many (one model in one pass: single-step + rollout + per-t + hmax on held-out test runs,
      single-step on source-domain test runs), pooled (pooling), stratified (stratified by level / tau).
"""
import os, sys, subprocess, importlib.util, csv, json
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRS = [0.1, 0.3]
CELL_KM2 = 0.04

# ---------------- CJK fonts (often missing on Colab; if missing, the whole figure falls back to English, no tofu boxes) ----------------
CJK_FONTS = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC', 'Source Han Sans SC', 'Source Han Sans CN',
             'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Hiragino Sans GB', 'PingFang SC', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
_FONT_READY = None


def _installed_cjk():
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    return [c for c in CJK_FONTS if c in names]


def _register_system_cjk():
    """Use fc-list to find CJK font files on the system and register them with matplotlib (needed after apt install)."""
    from matplotlib import font_manager as fm
    try:
        out = subprocess.run(['fc-list', ':lang=zh', 'file'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    added = []
    for line in out.splitlines():
        f = line.split(':')[0].strip()
        if f.lower().endswith(('.ttf', '.otf', '.ttc')):
            try:
                fm.fontManager.addfont(f); added.append(fm.FontProperties(fname=f).get_name())
            except Exception:
                pass
    return sorted(set(added))


def setup_fonts(auto_install=True, verbose=True):
    """Return True = a CJK font is available (set in matplotlib); False = fall back to English labels. Result is cached."""
    global _FONT_READY
    if _FONT_READY is not None:
        return _FONT_READY
    import matplotlib
    have = _installed_cjk()
    if not have:
        have = [n for n in _register_system_cjk() if n]
    if not have and auto_install and sys.platform.startswith('linux'):
        if verbose:
            print('[font] No CJK font found, trying to install fonts-noto-cjk (about 30-60 s)...', flush=True)
        for cmd in (['apt-get', '-qq', 'update'], ['apt-get', '-qq', '-y', 'install', 'fonts-noto-cjk']):
            try:
                subprocess.run(cmd, capture_output=True, timeout=600)
            except Exception as e:
                if verbose: print('[font] Install failed: %s' % e, flush=True)
        have = [n for n in _register_system_cjk() if n] or _installed_cjk()
    matplotlib.rcParams['axes.unicode_minus'] = False
    if have:
        pref = [c for c in CJK_FONTS if c in have] or have
        matplotlib.rcParams['font.sans-serif'] = pref + ['DejaVu Sans']
        _FONT_READY = True
        if verbose: print('[font] Using CJK font: %s' % pref[0], flush=True)
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        _FONT_READY = False
        if verbose: print('[font] No CJK font, figure labels fall back to English (no tofu boxes)', flush=True)
    return _FONT_READY


def T(cn, en):
    """Figure text: Chinese if a CJK font is available, otherwise English."""
    return cn if _FONT_READY else en



def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E35 = _imp('mb_eval35', '35_modelB_eval.py')


def level_of(alpha, beta, gamma):
    sc = {0.8: 0, 1.0: 1, 1.2: 2}[float(alpha)] + {0.7: 0, 1.0: 1, 1.3: 2}[float(beta)] + {0.75: 0, 1.0: 1, 1.25: 2}[float(gamma)]
    return 'low' if sc <= 2 else ('mid' if sc == 3 else 'high')


def cell_stats(P, T):
    """Return exactly poolable statistics: sa, ss, n, tp/fp/fn (two thresholds)."""
    e = (P - T).astype(np.float64)
    out = dict(sa=float(np.abs(e).sum()), ss=float((e ** 2).sum()), n=int(e.size))
    for t in THRS:
        pw, tw = P > t, T > t
        out['tp_%s' % t] = int((pw & tw).sum()); out['fp_%s' % t] = int((pw & ~tw).sum()); out['fn_%s' % t] = int((~pw & tw).sum())
    return out


def finalize(st):
    out = dict(mae=st['sa'] / st['n'], rmse=float(np.sqrt(st['ss'] / st['n'])), n=st['n'])
    for t in THRS:
        d = st['tp_%s' % t] + st['fp_%s' % t] + st['fn_%s' % t]
        out['csi_%s' % t] = st['tp_%s' % t] / d if d else float('nan')
    out.update({k: st[k] for k in st if k.startswith(('tp_', 'fp_', 'fn_'))})
    return out


def pooled(rows):
    """rows: list of dicts with sa/ss/n/tp/fp/fn (same mode / mask) -> pooled metrics."""
    keys = ['sa', 'ss', 'n'] + ['%s_%s' % (c, t) for t in THRS for c in ('tp', 'fp', 'fn')]
    st = {k: sum(float(r[k]) if k in ('sa', 'ss') else int(float(r[k])) for r in rows) for k in keys}
    return finalize(st)


class RunCache:
    """Evaluation runs kept in memory: sid -> dict(h, rain, wl, dis) + scenario metadata (level, tau)."""

    def __init__(self, D, sids, index_rows):
        self.D = D; self.runs = {}; self.meta = {}
        meta = {r['sid']: r for r in index_rows}
        for s in sids:
            self.runs[s] = D.load_run_arrays(s)
            m = meta[s]; self.meta[s] = dict(event=m['event'], level=level_of(m['alpha'], m['beta'], m['gamma']), tau_h=int(float(m['tau_h'])))

    def get(self, sid):
        return self.runs[sid]


@torch.no_grad()
def eval_run(model, norm, r, dev, masks, D, modes=('single', 'rollout', 'persistence'), per_t=True, hmax=True):
    """One run: return rows (one per mode x mask, with exactly poolable statistics), per_t rows, hmax row."""
    TS = list(range(D.T_MIN, D.T_MAX + 1)); T = r['h'][D.T_MIN:D.T_MAX + 1]
    preds = {}
    if 'single' in modes:
        ps, _ = E35.single_step(model, norm, r, dev); preds['single'] = np.stack([ps[t] for t in TS])
    if 'rollout' in modes:
        pr, _ = E35.rollout(model, norm, r, dev); preds['rollout'] = np.stack([pr[t] for t in TS])
    if 'persistence' in modes:
        preds['persistence'] = r['h'][D.T_MIN - 1:D.T_MAX]
    rows, trows, hrow = [], [], None
    for m, P in preds.items():
        for k, mk in masks.items():
            st = cell_stats(P[:, mk], T[:, mk]); rows.append(dict(mode=m, mask=k, **finalize(st), sa=st['sa'], ss=st['ss']))
            if per_t:
                e = (P[:, mk] - T[:, mk]).astype(np.float64)
                for j, t in enumerate(TS):
                    trows.append(dict(mode=m, mask=k, t=t, mae=float(np.abs(e[j]).mean()), rmse=float(np.sqrt((e[j] ** 2).mean()))))
    if hmax and 'rollout' in preds:
        act, land = masks['active'], masks['land']; hm = T.max(0); pm = preds['rollout'].max(0); hrow = {}
        for k, mk in masks.items():
            e = pm[mk] - hm[mk]; hrow['mae_%s' % k] = float(np.abs(e).mean()); hrow['rmse_%s' % k] = float(np.sqrt((e.astype(np.float64) ** 2).mean())); hrow['bias_%s' % k] = float(e.mean())
        for t in THRS:
            pw, tw = pm[act] > t, hm[act] > t
            hrow['csi_%s' % t] = float((pw & tw).sum() / max((pw | tw).sum(), 1))
            hrow['land_area_km2_sfincs_gt%s' % t] = float((hm[land] > t).sum() * CELL_KM2); hrow['land_area_km2_pred_gt%s' % t] = float((pm[land] > t).sum() * CELL_KM2)
    return rows, trows, hrow


@torch.no_grad()
def single_rmse_quick(model, norm, cache, sids, dev, masks, D):
    """For the per-epoch pilot trial: single-step RMSE (active / land cells), pooled over sids."""
    st = {k: dict(sa=0.0, ss=0.0, n=0) for k in masks}
    for s in sids:
        r = cache.get(s); T = r['h'][D.T_MIN:D.T_MAX + 1]
        ps, _ = E35.single_step(model, norm, r, dev); P = np.stack([ps[t] for t in range(D.T_MIN, D.T_MAX + 1)])
        for k, mk in masks.items():
            e = (P[:, mk] - T[:, mk]).astype(np.float64); st[k]['sa'] += float(np.abs(e).sum()); st[k]['ss'] += float((e ** 2).sum()); st[k]['n'] += e.size
    return {k: dict(mae=v['sa'] / v['n'], rmse=float(np.sqrt(v['ss'] / v['n']))) for k, v in st.items()}


def evaluate_many(model, norm, cache, test_sids, source_sids, dev, masks, D, do_source=True, do_persistence=False):
    """One model in one pass: held-out test runs single-step + rollout (+ optional baseline), per-t, hmax; source-domain test runs single-step. Returns four lists."""
    model.eval(); test_rows, per_t_rows, hmax_rows, src_rows = [], [], [], []
    modes = ('single', 'rollout') + (('persistence',) if do_persistence else ())
    for s in test_sids:
        rows, trows, hrow = eval_run(model, norm, cache.get(s), dev, masks, D, modes=modes)
        for x in rows: x.update(sid=s, **cache.meta[s]); test_rows.append(x)
        for x in trows: x.update(sid=s, event=cache.meta[s]['event']); per_t_rows.append(x)
        if hrow: hrow.update(sid=s, event=cache.meta[s]['event']); hmax_rows.append(hrow)
    if do_source:
        for s in source_sids:
            rows, _, _ = eval_run(model, norm, cache.get(s), dev, masks, D, modes=('single',), per_t=False, hmax=False)
            for x in rows: x.update(sid=s, **cache.meta[s]); src_rows.append(x)
    return test_rows, per_t_rows, hmax_rows, src_rows


def stratified(rows, by):
    """rows (per run, with level / tau_h) pooled by stratum, by in {'level','tau_h'}; returns {group: {mode: {mask: metrics}}}."""
    out = {}
    for g in sorted({r[by] for r in rows}, key=str):
        out[str(g)] = {}
        for m in sorted({r['mode'] for r in rows}):
            out[str(g)][m] = {}
            for k in sorted({r['mask'] for r in rows}):
                sel = [r for r in rows if r[by] == g and r['mode'] == m and r['mask'] == k]
                if sel: out[str(g)][m][k] = pooled(sel)
    return out
