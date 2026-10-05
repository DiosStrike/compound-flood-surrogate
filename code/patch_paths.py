#!/usr/bin/env python3
"""Replace hard-coded paths in scripts under code/ with paths.py constants. Idempotent; backs up originals to archive/code_backup_2026-09-14/ before editing."""
import os, re, shutil, sys
CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
BK = os.path.join(ROOT, 'archive/code_backup_2026-09-14')
os.makedirs(BK, exist_ok=True)

HDR = "import paths as P\n"

def patch(fn, rules, extra_head=True):
    p = os.path.join(CODE, fn); s = open(p).read(); s0 = s
    if 'import paths as P' not in s and extra_head:
        # insert the import after sys.path.insert(...)
        m = re.search(r"sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\n", s)
        if m:
            s = s[:m.end()] + HDR + s[m.end():]
        else:
            m = re.search(r"^import .*\n", s, re.M)
            s = s[:m.end()] + "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n" + HDR + s[m.end():]
    for a, b in rules:
        if isinstance(a, str):
            s = s.replace(a, b)
        else:
            s = a.sub(b, s)
    if s != s0:
        shutil.copyfile(p, os.path.join(BK, fn))
        open(p, 'w').write(s)
        print('patched', fn)
    else:
        print('unchanged', fn)

NEWLINE = "NEW = os.path.expanduser('~/mnt/Desktop/Flood_2.0')"
OLDLINE = "OLD = os.path.expanduser('~/mnt/Independent_Study_Flood')"
COMMON = [
    (NEWLINE, "NEW = P.ROOT"),
    (OLDLINE, "OLD = P.OLD"),
    ("os.path.join(NEW, 'data/gis')", "P.STATIC_GIS"),
    ("os.path.join(NEW, 'data', 'gis')", "P.STATIC_GIS"),
    ("os.path.join(NEW, 'data/my_qgis')", "P.STATIC_MYQGIS"),
    ("os.path.join(NEW, 'result/figures')", "P.FIG"),
    ("os.path.join(NEW, 'result/reports')", "P.REP"),
    ("os.path.join(NEW, 'result/manifests')", "P.MAN"),
    ("os.path.join(NEW, 'data/gis/huc10_rings_26917.json')", "os.path.join(P.STATIC_GIS, 'huc10_rings_26917.json')"),
    ("os.path.join(NEW, 'data/meta/study_extent.json')", "os.path.join(P.STATIC_META, 'study_extent.json')"),
    ("os.path.join(NEW, 'data/meta/terrain_60m.json')", "os.path.join(P.STATIC_META, 'terrain_60m.json')"),
    ("os.path.join(NEW, 'data/meta/boundaries.json')", "os.path.join(P.STATIC_META, 'boundaries.json')"),
    ("os.path.join(NEW, 'data/meta/terrain_qc_boundaries.json')", "os.path.join(P.STATIC_META, 'terrain_qc_boundaries.json')"),
    ("os.path.join(NEW, 'data/interim/merged60.npy')", "os.path.join(P.STATIC_INTERIM, 'merged60.npy')"),
    ("os.path.join(NEW, 'data/interim/active_mask.npy')", "os.path.join(P.STATIC_INTERIM, 'active_mask.npy')"),
    ("os.path.join(NEW, 'data/interim/huc10_union.npy')", "os.path.join(P.STATIC_INTERIM, 'huc10_union.npy')"),
    ("os.path.join(NEW, 'data/interim/dem60.npy')", "os.path.join(P.STATIC_INTERIM, 'dem60.npy')"),
    ("os.path.join(NEW, 'data/interim/crm60.npy')", "os.path.join(P.STATIC_INTERIM, 'crm60.npy')"),
    ("os.path.join(NEW, 'data/interim/ocean_mask.npy')", "os.path.join(P.STATIC_INTERIM, 'ocean_mask.npy')"),
    ("os.path.join(NEW, 'data/interim/connected_water_60m_26917.tif')", "os.path.join(P.STATIC_INTERIM, 'connected_water_60m_26917.tif')"),
    ("os.path.join(NEW, 'data/interim/source_flag_60m_26917.tif')", "os.path.join(P.STATIC_INTERIM, 'source_flag_60m_26917.tif')"),
    ("os.path.join(NEW, 'data/interim/mrms_P.npy')", "os.path.join(P.IRMA['raw'], 'mrms_cache/mrms_P.npy')"),
    ("os.path.join(NEW, 'data/interim/mrms_FLAG.npy')", "os.path.join(P.IRMA['raw'], 'mrms_cache/mrms_FLAG.npy')"),
    ("os.path.join(NEW, 'data/interim/mrms_lonlat.npy')", "os.path.join(P.IRMA['raw'], 'mrms_cache/mrms_lonlat.npy')"),
    ("os.path.join(NEW, 'data/interim/mrms_raw_cache.npz')", "os.path.join(P.IRMA['raw'], 'mrms_cache/mrms_raw_cache.npz')"),
    ("os.path.join(NEW, 'data/processed/topobathy_merged_60m_26917.tif')", "os.path.join(P.STATIC_TERRAIN, 'topobathy_merged_60m_26917.tif')"),
    ("os.path.join(NEW, 'data/processed/dem_usgs_60m_26917.tif')", "os.path.join(P.STATIC_TERRAIN, 'dem_usgs_60m_26917.tif')"),
    ("os.path.join(NEW, 'data/processed/crm_60m_26917.tif')", "os.path.join(P.STATIC_TERRAIN, 'crm_60m_26917.tif')"),
    ("os.path.join(NEW, 'data/processed/mayport_8720218_waterlevel_hourly_utc_navd88_m.csv')", "os.path.join(P.IRMA['processed'], 'mayport_8720218_waterlevel_hourly_utc_navd88_m.csv')"),
    ("os.path.join(NEW, 'data/processed/mayport_8720218_waterlevel_6min_utc_navd88_m.csv')", "os.path.join(P.IRMA['processed'], 'mayport_8720218_waterlevel_6min_utc_navd88_m.csv')"),
    ("os.path.join(NEW, 'data/processed/acosta_02246500_discharge_hourly_utc_m3s.csv')", "os.path.join(P.IRMA['processed'], 'acosta_02246500_discharge_hourly_utc_m3s.csv')"),
    ("os.path.join(NEW, 'data/processed/acosta_02246500_discharge_15min_utc_m3s.csv')", "os.path.join(P.IRMA['processed'], 'acosta_02246500_discharge_15min_utc_m3s.csv')"),
    ("os.path.join(NEW, 'data/processed/mrms_gaugecorr_qpe01h_jax_0p01deg.nc')", "os.path.join(P.IRMA['processed'], 'mrms_gaugecorr_qpe01h_jax_0p01deg.nc')"),
    ("os.path.join(NEW, 'data/processed/mrms_areal_mean_hourly.csv')", "os.path.join(P.IRMA['processed'], 'mrms_areal_mean_hourly.csv')"),
    ("os.path.join(NEW, 'data/meta/mrms_quality.json')", "os.path.join(P.IRMA['meta'], 'mrms_quality.json')"),
    ("os.path.join(NEW, 'data/meta/provenance_check.json')", "os.path.join(P.IRMA['meta'], 'provenance_check.json')"),
    ("os.path.join(NEW, 'data/raw/usgs_02246500_00060_raw.csv')", "os.path.join(P.IRMA['raw'], 'usgs_02246500_00060_raw.csv')"),
    ("os.path.join(NEW, 'data/raw/noaa_8720218_water_level_raw.json')", "os.path.join(P.IRMA['raw'], 'noaa_8720218_water_level_raw.json')"),
    ("os.path.join(NEW, 'data/raw/noaa_8720218_metadata.json')", "os.path.join(P.IRMA['raw'], 'noaa_8720218_metadata.json')"),
    ("os.path.join(NEW, 'data/raw')", "P.IRMA['raw']"),
    ("os.path.join(OLD, 'outputs/boundary_research')", "P.OLD_BR"),
    ("os.path.join(OLD, 'data/mrms_raw_old')", "P.OLD_MRMS"),
]
STATIC_DIRS = [("os.path.join(NEW, 'data/meta')", "P.STATIC_META"),
               ("os.path.join(NEW, 'data', 'meta')", "P.STATIC_META"),
               ("os.path.join(NEW, 'data/interim')", "P.STATIC_INTERIM"),
               ("os.path.join(NEW, 'data/processed')", "P.STATIC_TERRAIN")]
IRMA_DIRS = [("os.path.join(NEW, 'data/meta')", "P.IRMA['meta']"),
             ("os.path.join(NEW, 'data/interim')", "os.path.join(P.IRMA['raw'], 'mrms_cache')"),
             ("os.path.join(NEW, 'data/processed')", "P.IRMA['processed']")]

# 01 audit of old project -> static meta
patch('01_audit_old_project.py', COMMON + STATIC_DIRS)
patch('02_define_extent.py', COMMON + STATIC_DIRS)
patch('03_build_terrain.py', COMMON + STATIC_DIRS)
# 04 water level / discharge: output to irma processed / meta
patch('04_hydro_series.py', COMMON + IRMA_DIRS)
# 05 rainfall: processed -> irma processed; meta -> irma meta; interim (mrms) -> irma raw/mrms_cache
patch('05_mrms_rainfall.py', COMMON + IRMA_DIRS)
patch('06_terrain_qc_boundaries.py', COMMON + STATIC_DIRS)
patch('08_provenance_check.py', COMMON + IRMA_DIRS)
patch('10_active_domain_boundaries.py', COMMON + STATIC_DIRS)
patch('11_integrate_v2.py', COMMON + STATIC_DIRS)
patch('12_apply_v2.py', COMMON + STATIC_DIRS)
patch('15_boundary_check.py', COMMON + STATIC_DIRS)
# 07 figures: terrain part static, time-series part irma -- global static first; explicit irma files are already replaced one by one in COMMON
patch('07_figures.py', COMMON + [("os.path.join(NEW, 'data/meta')", "P.STATIC_META"),
                                 ("os.path.join(NEW, 'data/interim')", "P.STATIC_INTERIM"),
                                 ("os.path.join(NEW, 'data/processed')", "P.IRMA['processed']")])
# 13 report: PROC only used for csv -> irma; INT/META static
patch('13_report_v2.py', COMMON + [("os.path.join(NEW, 'data/processed')", "P.IRMA['processed']"),
                                   ("os.path.join(NEW, 'data/interim')", "P.STATIC_INTERIM"),
                                   ("os.path.join(NEW, 'data/meta')", "P.STATIC_META")])
# 14 window options: all irma
patch('14_window_options.py', COMMON + IRMA_DIRS)
# 16 SFINCS build: PROC (csv) -> irma; INT/META static; OUT split
patch('16_build_sfincs.py', COMMON + [
    ("os.path.join(NEW, 'data/processed')", "P.IRMA['processed']"),
    ("os.path.join(NEW, 'data/interim')", "P.STATIC_INTERIM"),
    ("os.path.join(NEW, 'data/meta')", "P.STATIC_META"),
    ("OUT = os.path.join(NEW, 'sfincs_200m')", "OUT = P.SFINCS_BASE                      # static: dep/msk/ind/bnd/src/obs/inp\nFORC = P.IRMA['forcing']                # event: bzs/dis/ampr/precip"),
    ("QG, ALT = os.path.join(OUT, 'qgis'), os.path.join(OUT, 'alt')", "QG, ALT = os.path.join(OUT, 'qgis'), os.path.join(FORC, 'alt')"),
    ("for d in (OUT, QG, ALT):", "for d in (OUT, QG, ALT, FORC):"),
    ("os.path.join(OUT, 'sfincs.bzs')", "os.path.join(FORC, 'sfincs.bzs')"),
    ("os.path.join(OUT, 'sfincs.dis')", "os.path.join(FORC, 'sfincs.dis')"),
    ("os.path.join(OUT, 'sfincs.ampr')", "os.path.join(FORC, 'sfincs.ampr')"),
    ("os.path.join(OUT, 'sfincs.precip')", "os.path.join(FORC, 'sfincs.precip')"),
])
# 17 report: SF static, forcing from irma
patch('17_sfincs_report.py', COMMON + [
    ("SF, INT, META = os.path.join(NEW, 'sfincs_200m'), os.path.join(NEW, 'data/interim'), os.path.join(NEW, 'data/meta')",
     "SF, INT, META = P.SFINCS_BASE, P.STATIC_INTERIM, P.STATIC_META\nFORC = P.IRMA['forcing']"),
    ("os.path.join(SF, 'sfincs.bzs')", "os.path.join(FORC, 'sfincs.bzs')"),
    ("os.path.join(SF, 'sfincs.dis')", "os.path.join(FORC, 'sfincs.dis')"),
    ("os.path.join(SF, 'sfincs.precip')", "os.path.join(FORC, 'sfincs.precip')"),
    ("os.path.join(SF, 'sfincs.ampr')", "os.path.join(FORC, 'sfincs.ampr')"),
])
# 09 manifest: directory mapping
patch('09_manifest.py', COMMON + [("os.path.join(NEW, 'data/meta')", "P.STATIC_META")])
# 18 validation: runs/<event>/<run>
patch('18_validate_obs.py', [
    ("ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
     "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport paths as P\nROOT = P.ROOT\nEVENT = sys.argv[2] if len(sys.argv) > 2 else 'irma'\nEV = P.event(EVENT)"),
    ("RDIR = os.path.join(ROOT, 'runs', RUN)", "RDIR = os.path.join(EV['runs'], RUN)"),
    ("RAW = os.path.join(ROOT, 'data/raw/obs_validation', RUN); PROC = os.path.join(ROOT, 'data/processed/obs_validation', RUN)",
     "RAW = os.path.join(EV['obs_raw'], RUN); PROC = os.path.join(EV['obs_proc'], RUN)"),
    ("Usage: python3 code/18_validate_obs.py <run_name>        (default baseline)",
     "Usage: python3 code/18_validate_obs.py <run_name> [event]   (default baseline irma)"),
], extra_head=False)
