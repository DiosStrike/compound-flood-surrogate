#!/usr/bin/env python3
"""49_exp3_summary.py -- Experiment 3 six-fold summary (docs/PLAN.md v17 §8.6-8.7, §8.19 fonts). Reads the fold directories only; writes result/models/exp3/ and result/figures/exp3/.

Input: <folds-root>/fold_<event>/{manifest.csv, metrics_test.csv, metrics_source.csv, hmax.csv, train_log.csv} (output of code/47)
Output:
  curve_by_fold.csv       fold x method x N: mean / min / max over three seeds (single-step & rollout; active & land cells; MAE / RMSE / CSI)
  curve_mean.csv          six-fold mean; marginal_gain.csv increments between adjacent N (reduction % relative to N=0)
  stratified.csv          metrics stratified by level (low / mid / high) and tau (per fold, method, N; seeds pooled)
  source_forgetting.csv   single-step metrics of B / Bp on the 50 source-domain test runs, per event + total, vs before fine-tuning
  correlates.csv          fold level: improvement (N=5, 10 vs N=0) vs degradation ratio / wetness / rainfall / product
  summary.json
  Figures: fig_exp3_curve_<mask>.png, fig_exp3_marginal.png, fig_exp3_stratified.png, fig_exp3_forgetting.png, fig_exp3_correlates.png, fig_exp3_B_vs_Bp.png
Missing folds or incomplete configs: stop and list them; no partial output.
"""
import os, sys, csv, json, argparse, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
EV_CN = dict(irma='Irma', matthew='Matthew', ian='Ian', milton='Milton', dorian='Dorian', beryl='Beryl')
METHODS = ['B', 'Bp', 'C']; MCN = dict(B='B full fine-tune', Bp="B' frozen encoder", C='C from scratch')
NS = [1, 2, 3, 5, 10]; SEEDS = [0, 1, 2]
COL = dict(B='#2a78d6', Bp='#1baf7a', C='#eb6834'); CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
COV = dict(degrade={'irma': 1.13, 'matthew': 1.18, 'ian': 1.13, 'milton': 1.14, 'dorian': 1.17, 'beryl': 1.13},
           wet_km2={'irma': 479.8, 'matthew': 448.2, 'ian': 94.2, 'milton': 166.4, 'dorian': 107.9, 'beryl': 246.8},
           rain_mm={'irma': 322.1, 'matthew': 259.6, 'ian': 39.3, 'milton': 84.0, 'dorian': 58.8, 'beryl': 120.8},
           product={'irma': 'MRMS', 'matthew': 'MRMS', 'ian': 'MRMS', 'milton': 'MRMS', 'dorian': 'MRMS', 'beryl': 'StageIV'})


def _imp(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, 'code', file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


L = _imp('exp3_lib', '48_exp3_eval_lib.py')
rcsv = lambda p: list(csv.DictReader(open(p)))


def wcsv(p, rows):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def check(root):
    probs = []
    for e in EVENTS:
        d = os.path.join(root, 'fold_%s' % e)
        for f in ('manifest.csv', 'metrics_test.csv', 'metrics_source.csv', 'hmax.csv', 'train_log.csv'):
            if not os.path.exists(os.path.join(d, f)): probs.append('%s: missing %s' % (e, f))
        if os.path.exists(os.path.join(d, 'manifest.csv')):
            man = {r['config']: r for r in rcsv(os.path.join(d, 'manifest.csv'))}
            if man.get('zero', {}).get('status') != 'done': probs.append('%s: missing N=0 zero-shot evaluation (--eval-zero)' % e)
            miss = ['%s_N%d_s%d' % (m, n, s) for m in METHODS for n in NS for s in SEEDS if man.get('%s_N%d_s%d' % (m, n, s), {}).get('status') != 'done']
            if miss: probs.append('%s: missing %d configs, e.g. %s' % (e, len(miss), ','.join(miss[:4])))
    return probs


def metric(rows, cfg, mode, mask):
    sel = [r for r in rows if r['config'] == cfg and r['mode'] == mode and r['mask'] == mask]
    return L.pooled(sel) if sel else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds-root', default=os.path.join(ROOT, 'result', 'models', 'exp3', 'folds'))
    ap.add_argument('--out-models', default=os.path.join(ROOT, 'result', 'models', 'exp3'))
    ap.add_argument('--out-figs', default=os.path.join(ROOT, 'result', 'figures', 'exp3'))
    ap.add_argument('--exp1', default=os.path.join(ROOT, 'result', 'models', 'exp1', 'eval', 'metrics_per_event.csv'))
    a = ap.parse_args()
    probs = check(a.folds_root)
    if probs:
        print('Stopped:'); [print('  ' + p) for p in probs]; sys.exit(1)
    os.makedirs(a.out_models, exist_ok=True); os.makedirs(a.out_figs, exist_ok=True)
    exp1 = {(r['event'], r['mode'], r['mask']): float(r['rmse']) for r in rcsv(a.exp1)} if os.path.exists(a.exp1) else {}

    curve, strat, forget, corr = [], [], [], []
    for e in EVENTS:
        d = os.path.join(a.folds_root, 'fold_%s' % e); T = rcsv(os.path.join(d, 'metrics_test.csv')); S = rcsv(os.path.join(d, 'metrics_source.csv'))
        for mode in ('single', 'rollout'):
            for mask in ('active', 'land'):
                z = metric(T, 'zero', mode, mask)
                # persistence baseline: select by mode == 'persistence' (its config column was rotated in outputs produced before 2026-09-18; the numeric columns are correct)
                pr = [r for r in T if r['mode'] == 'persistence' and r['mask'] == mask]
                p = L.pooled(pr) if (pr and mode == 'single') else None
                curve.append(dict(fold=e, method='zero', N=0, mode=mode, mask=mask, rmse_mean=z['rmse'], rmse_min=z['rmse'], rmse_max=z['rmse'], mae_mean=z['mae'], csi01_mean=z['csi_0.1'], csi03_mean=z['csi_0.3'],
                                  exp1_rmse=exp1.get((e, mode, mask), ''), persistence_rmse=(p or {}).get('rmse', '')))
                for m in METHODS:
                    for n in NS:
                        vals = [metric(T, '%s_N%d_s%d' % (m, n, s), mode, mask) for s in SEEDS]
                        curve.append(dict(fold=e, method=m, N=n, mode=mode, mask=mask, rmse_mean=np.mean([v['rmse'] for v in vals]), rmse_min=min(v['rmse'] for v in vals), rmse_max=max(v['rmse'] for v in vals),
                                          mae_mean=np.mean([v['mae'] for v in vals]), csi01_mean=np.mean([v['csi_0.1'] for v in vals]), csi03_mean=np.mean([v['csi_0.3'] for v in vals]),
                                          exp1_rmse=exp1.get((e, mode, mask), ''), persistence_rmse=(p or {}).get('rmse', '')))
        # stratification (seeds pooled; single-step)
        for by in ('level', 'tau_h'):
            for cfgm, n in [('zero', 0)] + [(m, n) for m in METHODS for n in NS]:
                cfgs = ['zero'] if cfgm == 'zero' else ['%s_N%d_s%d' % (cfgm, n, s) for s in SEEDS]
                rows = [r for r in T if r['config'] in cfgs and r['mode'] == 'single']
                for g, dd in L.stratified(rows, by).items():
                    for mask in ('active', 'land'):
                        strat.append(dict(fold=e, by=by, group=g, method=cfgm, N=n, mask=mask, **{k: dd['single'][mask][k] for k in ('mae', 'rmse', 'csi_0.1', 'csi_0.3', 'n')}))
        # source-domain forgetting (single-step; per event + total; vs zero before fine-tuning)
        for cfgm, n, s in [('zero', 0, -1)] + [(m, n, s) for m in ('B', 'Bp') for n in NS for s in SEEDS]:
            cfg = 'zero' if cfgm == 'zero' else '%s_N%d_s%d' % (cfgm, n, s)
            rows = [r for r in S if r['config'] == cfg and r['mode'] == 'single']
            if not rows: continue
            for mask in ('active', 'land'):
                for ev in sorted({r['event'] for r in rows}) + ['all_source']:
                    sel = [r for r in rows if r['mask'] == mask and (ev == 'all_source' or r['event'] == ev)]
                    forget.append(dict(fold=e, method=cfgm, N=n, seed=s, source_event=ev, mask=mask, **{k: L.pooled(sel)[k] for k in ('mae', 'rmse', 'csi_0.1', 'csi_0.3')}))
        # fold-level correlates
        z = metric(T, 'zero', 'single', 'active')['rmse']
        for m in METHODS:
            for n in (5, 10):
                v = np.mean([metric(T, '%s_N%d_s%d' % (m, n, s), 'single', 'active')['rmse'] for s in SEEDS])
                corr.append(dict(fold=e, method=m, N=n, zero_rmse=z, rmse=v, improvement_pct=100 * (z - v) / z, **{k: COV[k][e] for k in COV}))
    wcsv(os.path.join(a.out_models, 'curve_by_fold.csv'), curve); wcsv(os.path.join(a.out_models, 'stratified.csv'), strat)
    wcsv(os.path.join(a.out_models, 'source_forgetting.csv'), forget); wcsv(os.path.join(a.out_models, 'correlates.csv'), corr)
    # six-fold mean and marginal gain
    mean_rows, marg = [], []
    for mode in ('single', 'rollout'):
        for mask in ('active', 'land'):
            z = np.mean([r['rmse_mean'] for r in curve if r['method'] == 'zero' and r['mode'] == mode and r['mask'] == mask])
            mean_rows.append(dict(method='zero', N=0, mode=mode, mask=mask, rmse_mean6=z))
            for m in METHODS:
                prev = z
                for n in NS:
                    v = np.mean([r['rmse_mean'] for r in curve if r['method'] == m and r['N'] == n and r['mode'] == mode and r['mask'] == mask])
                    mean_rows.append(dict(method=m, N=n, mode=mode, mask=mask, rmse_mean6=v))
                    marg.append(dict(method=m, mode=mode, mask=mask, from_N=([0] + NS)[NS.index(n)], to_N=n, rmse_from=prev, rmse_to=v, gain_pct_vs_zero=100 * (z - v) / z, step_gain_pct=100 * (prev - v) / prev)); prev = v
    wcsv(os.path.join(a.out_models, 'curve_mean.csv'), mean_rows); wcsv(os.path.join(a.out_models, 'marginal_gain.csv'), marg)
    json.dump(dict(folds_root=os.path.abspath(a.folds_root), n_configs_per_fold=len(METHODS) * len(NS) * len(SEEDS) + 1, methods=METHODS, N=NS, seeds=SEEDS,
                   note='six folds give only 6 points; correlates are qualitative observations only', covariates=COV), open(os.path.join(a.out_models, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    figures(curve, mean_rows, strat, forget, corr, a.out_figs)
    print('six-fold mean single-step RMSE, active cells:', {r['N']: round(r['rmse_mean6'], 4) for r in mean_rows if r['mode'] == 'single' and r['mask'] == 'active' and r['method'] in ('zero', 'B')})
    print('output:', a.out_models, a.out_figs)


def figures(curve, mean_rows, strat, forget, corr, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L.setup_fonts()                      # CJK fonts: auto-install / fall back to English (PLAN §8.19)
    T = L.T
    MC = dict(B=T('B full fine-tune', 'B full fine-tune'), Bp=T("B' frozen encoder", "B' frozen encoder"), C=T('C from scratch', 'C from scratch'))
    LAB_N = T('number of fine-tune runs N', 'number of fine-tune runs N')
    LAB_RMSE = T('single-step RMSE (m)', 'single-step RMSE (m)')
    MASK_CN = lambda k: (T('active cells', 'active cells') if k == 'active' else T('land cells', 'land cells'))

    def style(ax):
        ax.set_facecolor(SURF); [ax.spines[sp].set_visible(False) for sp in ('top', 'right')]
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=8)

    xs = [0] + NS
    # 1) per-fold N curves
    for mask in ('active', 'land'):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor=SURF)
        for ax, e in zip(axes.ravel(), EVENTS):
            style(ax); z = next(r for r in curve if r['fold'] == e and r['method'] == 'zero' and r['mode'] == 'single' and r['mask'] == mask)
            for m in METHODS:
                rr = sorted([r for r in curve if r['fold'] == e and r['method'] == m and r['mode'] == 'single' and r['mask'] == mask], key=lambda r: r['N'])
                x = xs if m != 'C' else NS
                y = ([z['rmse_mean']] if m != 'C' else []) + [r['rmse_mean'] for r in rr]
                ax.plot(x, y, '-o', color=COL[m], lw=1.8, ms=5, label=MC[m])
                ax.fill_between(NS, [r['rmse_min'] for r in rr], [r['rmse_max'] for r in rr], color=COL[m], alpha=0.12, lw=0)
            if z['exp1_rmse'] != '': ax.axhline(float(z['exp1_rmse']), color='#4a3aa7', ls=':', lw=1, label=T('Exp-1 same-event', 'Exp-1 same-event'))
            ax.axhline(z['rmse_mean'], color=MUTED, ls='--', lw=1, label=T('N=0 zero-shot', 'N=0 zero-shot'))
            ax.set_xticks(xs); ax.set_xlabel(LAB_N, fontsize=8, color=INK2); ax.set_yscale('log')   # C is an order of magnitude larger than B/B', so a log axis is needed to see the B/B' difference
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%g' % v)); ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: ''))
            ax.set_title(T('held-out %s', 'held-out %s') % EV_CN[e], fontsize=9, color=INK, loc='left')
        axes[0, 0].legend(frameon=False, fontsize=7); axes[0, 0].set_ylabel(LAB_RMSE, fontsize=8, color=INK2)
        fig.suptitle(T('Exp-3: single-step RMSE vs N (%s; line = mean of 3 seeds, band = range)',
                       'Exp-3: single-step RMSE vs N (%s; line = mean of 3 seeds, band = range)') % MASK_CN(mask), fontsize=11, color=INK, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(out, 'fig_exp3_curve_%s.png' % mask), dpi=160); plt.close(fig)

    # 2) six-fold mean
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURF)
    for ax, mask in zip(axes, ('active', 'land')):
        style(ax)
        z = next(r for r in mean_rows if r['method'] == 'zero' and r['mode'] == 'single' and r['mask'] == mask)
        for m in METHODS:
            rr = sorted([r for r in mean_rows if r['method'] == m and r['mode'] == 'single' and r['mask'] == mask], key=lambda r: r['N'])
            x = xs if m != 'C' else NS; y = ([z['rmse_mean6']] if m != 'C' else []) + [r['rmse_mean6'] for r in rr]
            ax.plot(x, y, '-o', color=COL[m], lw=2, ms=5, label=MC[m])
        ax.set_xticks(xs); ax.set_xlabel(LAB_N, fontsize=8, color=INK2); ax.set_ylabel(LAB_RMSE, fontsize=8, color=INK2); ax.set_yscale('log')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%g' % v)); ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: ''))
        ax.set_title(T('mean of 6 folds · %s (log y)', 'mean of 6 folds · %s (log y)') % MASK_CN(mask), fontsize=9, loc='left', color=INK)
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_exp3_marginal.png'), dpi=160); plt.close(fig)

    # 3) B vs B′
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF); style(ax)
    for i, e in enumerate(EVENTS):
        b = {r['N']: r['rmse_mean'] for r in curve if r['fold'] == e and r['method'] == 'B' and r['mode'] == 'single' and r['mask'] == 'active'}
        bp = {r['N']: r['rmse_mean'] for r in curve if r['fold'] == e and r['method'] == 'Bp' and r['mode'] == 'single' and r['mask'] == 'active'}
        ax.plot(NS, [100 * (bp[n] - b[n]) / b[n] for n in NS], '-o', color=CAT[i], lw=1.6, ms=4, label=EV_CN[e])
    ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(NS); ax.set_xlabel(LAB_N, fontsize=8, color=INK2)
    ax.set_ylabel(T("B' minus B, single-step RMSE (%, positive = B' worse)", "B' minus B, single-step RMSE (%, positive = B' worse)"), fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title(T("Frozen encoder (B') vs full fine-tune (B), active cells", "Frozen encoder (B') vs full fine-tune (B), active cells"), fontsize=10, loc='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_exp3_B_vs_Bp.png'), dpi=160); plt.close(fig)

    # 4) stratification
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for ax, by, groups in zip(axes, ('level', 'tau_h'), (['low', 'mid', 'high'], ['-6', '0', '6'])):
        style(ax); w = 0.13
        for gi, g in enumerate(groups):
            for mi, (m, n, lab) in enumerate([('zero', 0, 'N=0'), ('B', 5, 'B N=5'), ('B', 10, 'B N=10')]):
                vals = [r['rmse'] for r in strat if r['by'] == by and r['group'] == g and r['method'] == m and r['N'] == n and r['mask'] == 'active']
                if vals: ax.bar(gi + (mi - 1) * w, np.mean(vals), w - 0.02, color=[MUTED, '#9ec5f4', COL['B']][mi], label=lab if gi == 0 else None)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([T('low', 'low'), T('mid', 'mid'), T('high', 'high')] if by == 'level' else ['tau -6', 'tau 0', 'tau +6'], fontsize=8)
        ax.set_ylabel(LAB_RMSE, fontsize=8, color=INK2)
        ax.set_title(T('stratified by %s (6-fold mean, active cells)', 'stratified by %s (6-fold mean, active cells)') % (T('forcing level', 'forcing level') if by == 'level' else T('rainfall shift tau', 'rainfall shift tau')), fontsize=9, loc='left', color=INK)
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_exp3_stratified.png'), dpi=160); plt.close(fig)

    # 5) source-domain forgetting
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURF); style(ax)
    for i, e in enumerate(EVENTS):
        z = next(r['rmse'] for r in forget if r['fold'] == e and r['method'] == 'zero' and r['source_event'] == 'all_source' and r['mask'] == 'active')
        ys = [np.mean([r['rmse'] for r in forget if r['fold'] == e and r['method'] == 'B' and r['N'] == n and r['source_event'] == 'all_source' and r['mask'] == 'active']) for n in NS]
        ax.plot(xs, [z] + ys, '-o', color=CAT[i], lw=1.6, ms=4, label=T('held-out %s', 'held-out %s') % EV_CN[e])
    ax.set_xticks(xs); ax.set_xlabel(LAB_N, fontsize=8, color=INK2)
    ax.set_ylabel(T('source-domain 50 test runs, single-step RMSE (m, active)', 'source-domain 50 test runs, single-step RMSE (m, active)'), fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title(T('Source forgetting: error on the other 5 events after B fine-tuning (mean of 3 seeds)',
                   'Source forgetting: error on the other 5 events after B fine-tuning (mean of 3 seeds)'), fontsize=10, loc='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_exp3_forgetting.png'), dpi=160); plt.close(fig)

    # 6) fold-level correlates (6 points, qualitative)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURF)
    labs = ((('degrade'), T('Exp-2 degradation ratio (exp2/exp1)', 'Exp-2 degradation ratio (exp2/exp1)')),
            (('wet_km2'), T('event wetness (flooded land area, km^2)', 'event wetness (flooded land area, km^2)')),
            (('rain_mm'), T('event rainfall total (mm)', 'event rainfall total (mm)')))
    for ax, (k, lab) in zip(axes, labs):
        style(ax); ax.grid(axis='x', color=GRID, lw=0.6)
        for r in [r for r in corr if r['method'] == 'B' and r['N'] == 5]:
            ax.scatter(r[k], r['improvement_pct'], s=50, color=COL['B'] if r['product'] == 'MRMS' else '#eda100', edgecolor=SURF, zorder=3)
            ax.annotate(EV_CN[r['fold']], (r[k], r['improvement_pct']), xytext=(4, 4), textcoords='offset points', fontsize=8, color=INK2)
        ax.set_xlabel(lab, fontsize=8, color=INK2)
        ax.set_ylabel(T('RMSE reduction of B N=5 vs N=0 (%)', 'RMSE reduction of B N=5 vs N=0 (%)'), fontsize=8, color=INK2)
    fig.suptitle(T('Improvement vs fold-level covariates (only 6 points, qualitative only; yellow = Stage IV rainfall)',
                   'Improvement vs fold-level covariates (only 6 points, qualitative only; yellow = Stage IV rainfall)'), fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(out, 'fig_exp3_correlates.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
