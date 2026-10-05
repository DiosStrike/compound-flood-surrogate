#!/usr/bin/env python3
"""43_exp1_figures.py -- Exp 1 result figures (reads result/models/exp1/ only, writes result/figures/exp1/).

Figures:
  fig_exp1_training_curve.png        training / validation loss (marks best epoch 32 and early stop at epoch 42)
  fig_exp1_metrics_bars.png          six events + pooled: single-step vs rollout vs persistence baseline (MAE / RMSE on active and land cells, CSI 0.1 / 0.3)
  fig_exp1_rollout_error_time.png    rollout RMSE over time (one line per event, active cells)
  fig_exp1_rollout_hmax_area.png     rollout max depth: land flooded area (h > 0.1 m) model vs SFINCS, one point per run, faceted by event
  fig_exp1_s041_peak_<event>.png     s041 at flood peak: SFINCS truth / model prediction / error (top row single-step, bottom row rollout)
Colors: dataviz reference palette (categorical colors in fixed order; depth single-hue blue ramp; error blue <-> red diverging, grey midpoint; baseline neutral grey).
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'PingFang SC', 'Arial Unicode MS', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.ticker import FuncFormatter
PLAIN = FuncFormatter(lambda v, _: ('%g' % v).replace('\u2212', '-'))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'result', 'models', 'exp1'); EV = os.path.join(EXP, 'eval')
FIG = os.path.join(ROOT, 'result', 'figures', 'exp1')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
EV_CN = dict(irma='Irma', matthew='Matthew', ian='Ian', milton='Milton', dorian='Dorian', beryl='Beryl', all='All six')
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MODE_COL = dict(single=CAT[0], rollout=CAT[1], persistence='#a8a79f')
MODE_CN = dict(single='Single-step', rollout='Rollout', persistence='Persistence (previous hour)')
LABEL = 'Exp 1'        # code/44 reuses this file for Exp 2 and sets this to 'Exp 2 ...'
SEQ = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
DEPTH_CMAP = LinearSegmentedColormap.from_list('depth', SEQ)
ERR_CMAP = LinearSegmentedColormap.from_list('err', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a99', '#b42b2b'])


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)


def rcsv(name):
    return list(csv.DictReader(open(os.path.join(EV, name))))


def fig_training():
    rows = list(csv.DictReader(open(os.path.join(EXP, 'train_log.csv'))))
    ep = [int(r['epoch']) for r in rows]; tr = [float(r['train_loss']) for r in rows]; va = [float(r['val_loss']) for r in rows]
    best = min(rows, key=lambda r: float(r['val_loss'])); be, bv = int(best['epoch']), float(best['val_loss'])
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF); style(ax)
    ax.plot(ep, tr, color=CAT[0], lw=2, label='Train loss'); ax.plot(ep, va, color=CAT[1], lw=2, label='Val loss')
    ax.set_yscale('log'); ax.yaxis.set_major_formatter(PLAIN); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.axvline(be, color=MUTED, lw=1, ls='--'); ax.axvline(ep[-1], color=MUTED, lw=1, ls=':')
    ax.plot([be], [bv], 'o', ms=8, color=CAT[1], mec=SURF, mew=2)
    ax.set_ylim(min(tr + va) / 2, max(tr + va) * 1.6)
    ax.annotate('Best: epoch %d, val %.3g' % (be, bv), (be, bv), xytext=(-150, -4), textcoords='offset points', fontsize=8, color=INK2, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.text(ep[-1] - 0.4, max(tr + va) * 1.3, 'Early stop: epoch %d' % ep[-1], ha='right', va='center', fontsize=8, color=INK2)
    ax.set_xlabel('Epoch', color=INK2, fontsize=9); ax.set_ylabel('Active-cell MSE (m^2, log scale)', color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper center')
    ax.set_title('Exp 1 training curve (batch 8, 24 times per run, patience 10)', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_exp1_training_curve.png'), dpi=170); plt.close(fig)


def fig_bars():
    rows = rcsv('metrics_per_event.csv')
    get = lambda e, m, k, c: float(next(r[c] for r in rows if r['event'] == e and r['mode'] == m and r['mask'] == k))
    panels = [('mae', 'active', 'MAE (m) - active cells'), ('rmse', 'active', 'RMSE (m) - active cells'), ('csi_0.1', 'active', 'CSI (h > 0.1 m) - active cells'),
              ('mae', 'land', 'MAE (m) - land cells'), ('rmse', 'land', 'RMSE (m) - land cells'), ('csi_0.3', 'active', 'CSI (h > 0.3 m) - active cells')]
    groups = EVENTS + ['all']; x = np.arange(len(groups)); w = 0.26
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURF)
    for ax, (c, k, title) in zip(axes.ravel(), panels):
        style(ax)
        for j, m in enumerate(['single', 'rollout', 'persistence']):
            vals = [get(e, m, k, c) for e in groups]
            ax.bar(x + (j - 1) * w, vals, w - 0.03, color=MODE_COL[m], label=MODE_CN[m])
        ax.set_xticks(x); ax.set_xticklabels([EV_CN[g] for g in groups], fontsize=8, rotation=0)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        if c.startswith('csi'): ax.set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc='lower left', bbox_to_anchor=(0, 1.1))
    fig.suptitle('%s test set (10 runs x 168 times per event): single-step vs rollout vs persistence' % LABEL, fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, 'fig_exp1_metrics_bars.png'), dpi=170); plt.close(fig)


def fig_time():
    rows = [r for r in rcsv('per_t.csv') if r['mode'] == 'rollout' and r['mask'] == 'active']
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURF); style(ax)
    for i, e in enumerate(EVENTS):
        rr = [r for r in rows if r['event'] == e]; t = [int(r['t']) for r in rr]; y = [float(r['rmse']) for r in rr]
        ax.plot(t, y, color=CAT[i], lw=2, label=EV_CN[e])
    ax.set_xlim(72, 242); ax.set_xlabel('Target time t (h since tstart)', color=INK2, fontsize=9); ax.set_ylabel('RMSE (m) - active cells', color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc='upper left')
    ax.set_title('Rollout error over time (starting from true h(70-72), 10 test runs pooled per event)', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_exp1_rollout_error_time.png'), dpi=170); plt.close(fig)


def fig_hmax():
    rows = rcsv('hmax_rollout_per_run.csv')
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), facecolor=SURF)
    for ax, e in zip(axes.ravel(), EVENTS):
        style(ax); ax.grid(axis='x', color=GRID, lw=0.6)
        rr = [r for r in rows if r['event'] == e]
        xs = [float(r['land_area_km2_sfincs_gt0.1']) for r in rr]; ys = [float(r['land_area_km2_pred_gt0.1']) for r in rr]
        lo, hi = 0, max(xs + ys) * 1.08
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls='--')
        ax.scatter(xs, ys, s=40, color=CAT[0], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_title(EV_CN[e], fontsize=9, color=INK, loc='left')
    for ax in axes[1]: ax.set_xlabel('SFINCS hmax land flooded area (km^2)', color=INK2, fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel('Rollout max depth land flooded area (km^2)', color=INK2, fontsize=8)
    fig.suptitle('Rollout event max depth vs SFINCS hmax: land-cell area with h > 0.1 m (one point per test run, dashed 1:1)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, 'fig_exp1_rollout_hmax_area.png'), dpi=170); plt.close(fig)


def fig_maps():
    z = np.load(os.path.join(EV, 's041_peak_maps.npz'))
    st = np.load(os.path.join(ROOT, 'data', 'modelB_v1', 'static.npz')); act = st['loss_mask']
    peak_t = dict(zip([str(e) for e in z['events']], z['peak_t']))
    ext = (413300 / 1000, (413300 + 255 * 200) / 1000, 3337100 / 1000, (3337100 + 195 * 200) / 1000)
    for e in EVENTS:
        truth = z['%s_truth' % e]
        vmax = float(np.percentile(truth[act], 99.5))
        errs = [z['%s_%s' % (e, m)] - truth for m in ('single', 'rollout')]
        elim = max(float(np.percentile(np.abs(er[act]), 99.5)) for er in errs) or 0.01
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), facecolor=SURF)
        for i, m in enumerate(('single', 'rollout')):
            pred = z['%s_%s' % (e, m)]
            for j, (arr, cmap, norm, lab) in enumerate([(truth, DEPTH_CMAP, None, 'SFINCS truth h (m)'), (pred, DEPTH_CMAP, None, '%s prediction h (m)' % MODE_CN[m]),
                                                        (pred - truth, ERR_CMAP, TwoSlopeNorm(0, -elim, elim), '%s error, prediction - truth (m)' % MODE_CN[m])]):
                ax = axes[i, j]; ax.set_facecolor('#e1e0d9')
                a = np.ma.masked_where(~act, arr)
                kw = dict(norm=norm) if norm else dict(vmin=0, vmax=vmax)
                im = ax.imshow(a, origin='lower', extent=ext, cmap=cmap, interpolation='nearest', **kw)
                ax.set_title(lab, fontsize=9, color=INK, loc='left'); ax.tick_params(labelsize=7, colors=INK2)
                for s in ax.spines.values(): s.set_color(AXIS)
                cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.yaxis.set_major_formatter(PLAIN); cb.ax.tick_params(labelsize=7, colors=INK2); cb.outline.set_edgecolor(AXIS)
                if j == 0: ax.set_ylabel('UTM 17N northing (km)', fontsize=8, color=INK2)
                if i == 1: ax.set_xlabel('UTM 17N easting (km)', fontsize=8, color=INK2)
        fig.suptitle('%s s041 (original scenario) flood peak t = %d h (max total land-cell depth); grey = inactive cells; color scale capped at 99.5th percentile' % (EV_CN[e], peak_t[e]),
                     fontsize=10, color=INK, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(FIG, 'fig_exp1_s041_peak_%s.png' % e), dpi=150); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_training(); fig_bars(); fig_time(); fig_hmax(); fig_maps()
    print('Figures written to', FIG, sorted(os.listdir(FIG)))


if __name__ == '__main__':
    main()
