#!/usr/bin/env python3
"""52_final_tables.py -- generate metric tables for the final report (draft _v1, csv + md). Reads evaluation results only; every number is read from files, nothing is estimated.

Table 1 Exp. 1: six events + total x single-step / rollout / Persistence MAE, RMSE (active cells, land cells) and CSI (0.1 / 0.3 m) -- [appendix]
Table 2 Exp. 2: same format, six folds (held-out event) + total -- [appendix]
Table 3 (--only main) condensed combined table for the main text: rows = six events + total; columns = single-step RMSE (Exp. 1 / Exp. 2), Persistence RMSE,
      rollout RMSE (Exp. 1 / Exp. 2), single-step CSI 0.1 (Exp. 1 / Exp. 2); all on active cells
Sources: result/models/exp1/eval/metrics_per_event.csv, result/models/exp2/metrics_per_event.csv (both pooled over run x time x cell within each event)
Outputs: final_report_materials/tables/tab_e{1,2}_full_v1.{csv,md}

Note: "Persistence" is the "copy the previous hour" baseline in the notes -- the SFINCS truth at t-1 is used directly as the prediction for t.

Usage: python3 code/52_final_tables.py
"""
import os, csv, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'final_report_materials', 'tables')
EVENTS = ['irma', 'matthew', 'ian', 'milton', 'dorian', 'beryl']
EV_CN = dict(irma='Irma', matthew='Matthew', ian='Ian', milton='Milton', dorian='Dorian', beryl='Beryl', all='All')
MODES = [('single', 'Single-step'), ('rollout', 'Rollout'), ('persistence', 'Persistence')]
COLS = ['event', 'mode', 'mae_active', 'rmse_active', 'csi01_active', 'csi03_active',
        'mae_land', 'rmse_land', 'csi01_land', 'csi03_land', 'n_cellhours_active']


def build(src, label):
    rows = list(csv.DictReader(open(os.path.join(ROOT, src))))
    get = lambda e, m, k, c: next((r[c] for r in rows if r['event'] == e and r['mode'] == m and r['mask'] == k), None)
    out = []
    for e in EVENTS + ['all']:
        for m, mcn in MODES:
            if get(e, m, 'active', 'rmse') is None:
                continue
            out.append(dict(event=EV_CN[e], mode=mcn,
                            mae_active=round(float(get(e, m, 'active', 'mae')), 4),
                            rmse_active=round(float(get(e, m, 'active', 'rmse')), 4),
                            csi01_active=round(float(get(e, m, 'active', 'csi_0.1')), 3),
                            csi03_active=round(float(get(e, m, 'active', 'csi_0.3')), 3),
                            mae_land=round(float(get(e, m, 'land', 'mae')), 4),
                            rmse_land=round(float(get(e, m, 'land', 'rmse')), 4),
                            csi01_land=round(float(get(e, m, 'land', 'csi_0.1')), 3),
                            csi03_land=round(float(get(e, m, 'land', 'csi_0.3')), 3),
                            n_cellhours_active=int(get(e, m, 'active', 'n'))))
    return out


def write(out_rows, name, title, note):
    p_csv = os.path.join(OUT, name + '.csv')
    with open(p_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(out_rows)
    p_md = os.path.join(OUT, name + '.md')
    with open(p_md, 'w') as f:
        f.write('# %s\n\n%s\n\n' % (title, note))
        f.write('| Event | Prediction | MAE active | RMSE active | CSI 0.1 active | CSI 0.3 active | MAE land | RMSE land | CSI 0.1 land | CSI 0.3 land |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for r in out_rows:
            bold = (lambda v: '**%s**' % v) if r['event'] == 'All' else (lambda v: v)
            f.write('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' % (
                bold(r['event']), bold(r['mode']), bold('%.4f' % r['mae_active']), bold('%.4f' % r['rmse_active']),
                bold('%.3f' % r['csi01_active']), bold('%.3f' % r['csi03_active']), bold('%.4f' % r['mae_land']),
                bold('%.4f' % r['rmse_land']), bold('%.3f' % r['csi01_land']), bold('%.3f' % r['csi03_land'])))
        f.write('\nMAE / RMSE in m; CSI dimensionless, wet = h > threshold, CSI = TP / (TP + FP + FN).\n')
        f.write('24,467 active cells, 21,933 land cells (active and zb >= 0); each event row pools that event\'s 10 test runs x 168 time steps x cells,'
                ' the total row pools 60 runs. The number of active cell-hours per row is in the csv column n_cellhours_active.\n')
    return p_csv, p_md


MAIN_COLS = ['event', 'single_rmse_exp1', 'single_rmse_exp2', 'persistence_rmse',
             'rollout_rmse_exp1', 'rollout_rmse_exp2', 'single_csi01_exp1', 'single_csi01_exp2']


def build_main():
    """Condensed combined table for the main text (all active cells; numbers read from the two metrics_per_event.csv files)."""
    r1 = list(csv.DictReader(open(os.path.join(ROOT, 'result/models/exp1/eval/metrics_per_event.csv'))))
    r2 = list(csv.DictReader(open(os.path.join(ROOT, 'result/models/exp2/metrics_per_event.csv'))))
    g = lambda rows, e, m, c: float(next(r[c] for r in rows if r['event'] == e and r['mode'] == m and r['mask'] == 'active'))
    out = []
    for e in EVENTS + ['all']:
        out.append(dict(event=EV_CN[e],
                        single_rmse_exp1=round(g(r1, e, 'single', 'rmse'), 4),
                        single_rmse_exp2=round(g(r2, e, 'single', 'rmse'), 4),
                        persistence_rmse=round(g(r1, e, 'persistence', 'rmse'), 4),
                        rollout_rmse_exp1=round(g(r1, e, 'rollout', 'rmse'), 4),
                        rollout_rmse_exp2=round(g(r2, e, 'rollout', 'rmse'), 4),
                        single_csi01_exp1=round(g(r1, e, 'single', 'csi_0.1'), 3),
                        single_csi01_exp2=round(g(r2, e, 'single', 'csi_0.1'), 3)))
    return out


def write_main(rows, name):
    p_csv = os.path.join(OUT, name + '.csv')
    with open(p_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, MAIN_COLS); w.writeheader(); w.writerows(rows)
    p_md = os.path.join(OUT, name + '.md')
    with open(p_md, 'w') as f:
        f.write('# Table 3 (main text) Main metrics of Exp. 1 vs Exp. 2\n\nAll on **active cells**; RMSE in m; CSI dimensionless.\n\n')
        f.write('| Event | Single-step RMSE Exp. 1 | Single-step RMSE Exp. 2 | Persistence RMSE | Rollout RMSE Exp. 1 | Rollout RMSE Exp. 2 | Single-step CSI 0.1 Exp. 1 | Single-step CSI 0.1 Exp. 2 |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            b = (lambda v: '**%s**' % v) if r['event'] == 'All' else (lambda v: v)
            f.write('| %s | %s | %s | %s | %s | %s | %s | %s |\n' % (
                b(r['event']), b('%.4f' % r['single_rmse_exp1']), b('%.4f' % r['single_rmse_exp2']), b('%.4f' % r['persistence_rmse']),
                b('%.4f' % r['rollout_rmse_exp1']), b('%.4f' % r['rollout_rmse_exp2']),
                b('%.3f' % r['single_csi01_exp1']), b('%.3f' % r['single_csi01_exp2'])))
        f.write('\nTable notes:\n\n')
        f.write('- **Each Exp. 2 row is the result with that event held out** (never seen in training); the test set is the same as Exp. 1 -- the 10 split_A test runs of that event, all 168 time steps.\n')
        f.write('- **Persistence** = the SFINCS truth at t-1 used as the prediction for t; it is model-independent, so Exp. 1 and Exp. 2 share one column.\n')
        f.write('- Each row pools that event\'s 10 test runs x 168 time steps x active cells (24,467); the "total" row pools 60 runs.\n')
        f.write('- Data sources: `result/models/exp1/eval/metrics_per_event.csv`, `result/models/exp2/metrics_per_event.csv`;'
                ' full metrics (incl. land cells, MAE, CSI 0.3) in appendix Tables 1 and 2.\n')
    return p_csv, p_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'full', 'main'],
                    help='full = write only the two full appendix tables; main = write only the condensed main-text table (existing full tables are not overwritten)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    outs = []
    if a.only == 'main':
        outs += write_main(build_main(), 'tab_e12_main_v1')
        for p in outs:
            print('  ', os.path.relpath(p, ROOT))
        return
    outs += write(build('result/models/exp1/eval/metrics_per_event.csv', 'exp1'), 'tab_e1_full_v1',
                  'Table 1 Exp. 1 (same event) test-set metrics',
                  'Data source: `result/models/exp1/eval/metrics_per_event.csv` (generated by `code/42_exp1_eval.py`).'
                  ' Test set: 10 split_A test runs for each of the six events, all 168 time steps. Persistence = the SFINCS truth at t-1 used as the prediction for t.')
    outs += write(build('result/models/exp2/metrics_per_event.csv', 'exp2'), 'tab_e2_full_v1',
                  'Table 2 Exp. 2 (cross-event, 6-fold leave-one-out) test-set metrics',
                  'Data source: `result/models/exp2/metrics_per_event.csv` (aggregated by `code/44_exp2_summary.py`, per fold generated by `code/39_exp2_eval.py`).'
                  ' The event in each row is that fold\'s **held-out event** (never seen in training); same 10 test runs as Exp. 1.'
                  ' Persistence is the same as in Exp. 1 (model-independent); the values in both tables should match.')
    if a.only == 'all':
        outs += write_main(build_main(), 'tab_e12_main_v1')
    for p in outs:
        print('  ', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
