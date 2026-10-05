#!/usr/bin/env python3
"""
paths.py -- single source of truth for project paths (after the 2026-09-14 directory restructure)

Conventions (Proposal §7):
  data/static/     shared static data: terrain, GIS, model domain, 200 m SFINCS base grid -- shared by all events, one copy only
  data/<event>/    event-specific: raw / processed / scenarios / meta
  runs/<event>/<run_name>/   one directory per simulation, containing all of its inputs, independently reproducible

ROOT is derived from this file's location (parent of code/), so it works wherever the project is checked out.
The old project OLD is read-only; it is probed from candidate paths and is None if not found (only affects scripts 01-08 that "rerun data preparation").
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, 'code')

_OLD_CANDIDATES = [os.path.expanduser('~/Independent_Study_Flood'),
                   os.path.expanduser('~/mnt/Independent_Study_Flood')]
OLD = next((p for p in _OLD_CANDIDATES if os.path.isdir(p)), None)
OLD_BR = os.path.join(OLD, 'outputs/boundary_research') if OLD else None
OLD_MRMS = os.path.join(OLD, 'data/mrms_raw_old') if OLD else None

# ---- shared static ----
STATIC = os.path.join(ROOT, 'data/static')
STATIC_TERRAIN = os.path.join(STATIC, 'terrain')      # 60 m terrain tif, masks, components
STATIC_GIS = os.path.join(STATIC, 'gis')              # model domain / boundaries / HUC10 geojson; my_qgis/ originals
STATIC_MYQGIS = os.path.join(STATIC_GIS, 'my_qgis')
STATIC_INTERIM = os.path.join(STATIC, 'interim')      # terrain intermediates .npy, diagnostic rasters
STATIC_META = os.path.join(STATIC, 'meta')            # study_extent / terrain_60m / boundaries / old-project audit
SFINCS_BASE = os.path.join(STATIC, 'sfincs_base')     # 200 m dep/msk/ind/bnd/src/obs + inp template (v1 terrain, block average)

# ---- events ----
EVENTS_DIR = os.path.join(ROOT, 'data')


def event(name='irma'):
    e = os.path.join(EVENTS_DIR, name)
    return dict(root=e,
                raw=os.path.join(e, 'raw'),
                processed=os.path.join(e, 'processed'),
                forcing=os.path.join(e, 'processed/sfincs_forcing'),   # bzs / dis / ampr / precip
                scenarios=os.path.join(e, 'scenarios'),
                meta=os.path.join(e, 'meta'),
                obs_raw=os.path.join(e, 'raw/obs_validation'),
                obs_proc=os.path.join(e, 'processed/obs_validation'),
                runs=os.path.join(ROOT, 'runs', name))


IRMA = event('irma')

# ---- results ----
RESULT = os.path.join(ROOT, 'result')
FIG = os.path.join(RESULT, 'figures')
REP = os.path.join(RESULT, 'reports')
MAN = os.path.join(RESULT, 'manifests')
ARCHIVE = os.path.join(ROOT, 'archive')


def ensure_dirs(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    print('ROOT   ', ROOT)
    print('OLD    ', OLD)
    for k in ('STATIC_TERRAIN', 'STATIC_GIS', 'STATIC_INTERIM', 'STATIC_META', 'SFINCS_BASE'):
        p = globals()[k]; print('%-15s %s  %s' % (k, p, 'ok' if os.path.isdir(p) else 'MISSING'))
    for k, p in IRMA.items():
        print('irma.%-10s %s  %s' % (k, p, 'ok' if os.path.isdir(p) else 'MISSING'))
