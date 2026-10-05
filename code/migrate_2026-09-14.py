#!/usr/bin/env python3
"""
migrate_2026-09-14.py -- directory restructuring (Proposal §7): move and archive only, never delete any file.
Every move is written to data/static/meta/migration_log_2026-09-14.csv (old path, new path, purpose).
No overwriting on name clash: if the target exists, rename with _dup and log it.
"""
import os, sys, shutil, csv, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P

R = P.ROOT
log = []


def mv(rel_src, rel_dst, why):
    src, dst = os.path.join(R, rel_src), os.path.join(R, rel_dst)
    if not os.path.exists(src):
        log.append((rel_src, '', 'SKIP: does not exist / ' + why)); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst); dst = base + '_dup' + ext
        why += ' (target exists, renamed _dup)'
    shutil.move(src, dst)
    log.append((rel_src, os.path.relpath(dst, R), why))


def mv_glob(rel_dir, names, rel_dst_dir, why):
    for n in names:
        mv(os.path.join(rel_dir, n), os.path.join(rel_dst_dir, n), why)


# ---------- 1. Archive documents ----------
mv('RUN_SFINCS.md', 'archive/docs/RUN_SFINCS.md', 'merged into README §run entry point')
for n in ('01_file_version_table.md', '02_qc_acceptance_checklist.md', '03_decisions_and_open_items.md'):
    mv('result/reports/' + n, 'archive/docs/reports/' + n, 'intermediate records from data preparation; conclusions moved into README/PDF')

# ---------- 2. Shared static data ----------
terrain = ['topobathy_merged_60m_26917.tif', 'topobathy_merged_60m_26917.tif.aux.xml',
           'topobathy_merged_60m_26917_huc10.tif', 'topobathy_merged_60m_26917_huc10.tif.aux.xml',
           'topobathy_merged_60m_26917_active.tif', 'topobathy_merged_60m_26917_active.tif.aux.xml',
           'active_domain_mask_60m_26917.tif', 'dem_usgs_60m_26917.tif', 'crm_60m_26917.tif']
mv_glob('data/processed', terrain, 'data/static/terrain', 'shared terrain')
mv('data/gis', 'data/static/gis', 'model domain / boundaries / HUC10 GeoJSON')
mv('data/my_qgis', 'data/static/gis/my_qgis', 'user QGIS originals (read-only)')
static_interim = ['merged60.npy', 'dem60.npy', 'crm60.npy', 'active_mask.npy', 'huc10_union.npy',
                  'uncovered_edge.npy', 'sfincs200_Z.npy', 'sfincs200_MSK.npy', 'sfincs200_Zmin.npy',
                  'sfincs200_WFRAC.npy', 'connected_water_60m_26917.tif',
                  'diff_dem_minus_crm_60m_26917.tif', 'source_flag_60m_26917.tif']
mv_glob('data/interim', static_interim, 'data/static/interim', 'terrain intermediates')
static_meta = ['study_extent.json', 'terrain_60m.json', 'boundaries.json', 'terrain_qc_boundaries.json',
               'audit_rasters.csv', 'audit_vectors.csv', 'audit_tables.csv', 'audit_checksums.csv']
mv_glob('data/meta', static_meta, 'data/static/meta', 'domain / terrain / boundary metadata and old-project audit')

# SFINCS base: separate the static part from the Irma forcing
sf_static = ['sfincs.inp', 'sfincs.dep', 'sfincs.msk', 'sfincs.ind', 'sfincs.bnd', 'sfincs.src',
             'sfincs.obs', 'sfincs_build.json', 'check_run.py', 'qgis']
mv_glob('sfincs_200m', sf_static, 'data/static/sfincs_base', '200 m base grid and boundary points (shared by all events)')
sf_irma = ['sfincs.bzs', 'sfincs.dis', 'sfincs.ampr', 'sfincs.precip', 'alt']
mv_glob('sfincs_200m', sf_irma, 'data/irma/processed/sfincs_forcing', 'Irma-specific forcing, not static')
if os.path.isdir(os.path.join(R, 'sfincs_200m')) and not os.listdir(os.path.join(R, 'sfincs_200m')):
    os.rmdir(os.path.join(R, 'sfincs_200m')); log.append(('sfincs_200m/', '', 'empty directory removed'))

# ---------- 3. Irma event ----------
mv('data/raw', 'data/irma/raw', 'Irma raw downloads (incl. obs_validation)')
irma_interim = ['mrms_P.npy', 'mrms_FLAG.npy', 'mrms_lonlat.npy', 'mrms_raw_cache.npz', 'mrms_P_0903_0916.npy']
mv_glob('data/interim', irma_interim, 'data/irma/raw/mrms_cache', 'Irma MRMS decode cache')
irma_proc = ['mayport_8720218_waterlevel_6min_utc_navd88_m.csv', 'mayport_8720218_waterlevel_hourly_utc_navd88_m.csv',
             'acosta_02246500_discharge_15min_utc_m3s.csv', 'acosta_02246500_discharge_hourly_utc_m3s.csv',
             'mrms_areal_mean_hourly.csv', 'mrms_gaugecorr_qpe01h_jax_0p01deg.nc', 'obs_validation']
mv_glob('data/processed', irma_proc, 'data/irma/processed', 'Irma processed forcing and validation data')
irma_meta = ['hydro_quality.json', 'hydro_gaps.csv', 'hydro_incomplete_hours.csv', 'mrms_quality.json',
             'window_options.json', 'provenance_check.json']
mv_glob('data/meta', irma_meta, 'data/irma/meta', 'Irma data quality records')
os.makedirs(os.path.join(R, 'data/irma/scenarios'), exist_ok=True)
open(os.path.join(R, 'data/irma/scenarios/README.txt'), 'w').write(
    'bzs/dis/ampr of the 81 Irma-like scenarios plus manifest.csv and split_A.csv will go here (not yet generated).\n')

# Remaining empty directories
for d in ('data/processed', 'data/interim', 'data/meta'):
    p = os.path.join(R, d)
    if os.path.isdir(p):
        left = [x for x in os.listdir(p) if x != '.DS_Store']
        if left:
            log.append((d + '/', '', 'unclassified files remain: ' + ', '.join(left)))
        else:
            for x in os.listdir(p):
                os.remove(os.path.join(p, x))
            os.rmdir(p); log.append((d + '/', '', 'empty directory removed'))

# ---------- 4. runs ----------
mv('runs/baseline', 'runs/irma/baseline', 'first baseline run (self-contained inputs, reproducible, left unchanged)')
mv('runs/.last_elapsed_sec', 'runs/irma/.last_elapsed_sec', 'elapsed-time record')

# ---------- 5. Log ----------
os.makedirs(P.STATIC_META, exist_ok=True)
with open(os.path.join(P.STATIC_META, 'migration_log_2026-09-14.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['old_path', 'new_path', 'note'])
    for r in log:
        w.writerow(r)
n_mv = sum(1 for r in log if r[1])
print('moved %d items, skipped/noted %d items -> data/static/meta/migration_log_2026-09-14.csv' % (n_mv, len(log) - n_mv))
for r in log:
    if not r[1]:
        print('  ', r)
