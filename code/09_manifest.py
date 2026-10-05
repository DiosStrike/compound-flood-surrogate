#!/usr/bin/env python3
"""09_manifest.py -- generate the deliverable file manifest (with SHA256) and the reuse-source mapping table."""
import os, sys, csv, hashlib, datetime, json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH

NEW = PTH.ROOT
OLD = PTH.OLD
MAN = PTH.MAN
os.makedirs(MAN, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


ROLE = {
    'data/static/terrain': 'shared terrain (60 m)',
    'data/static/gis': 'shared GIS (model domain / boundaries / HUC10 / user QGIS originals)',
    'data/static/interim': 'terrain intermediates',
    'data/static/meta': 'domain / terrain / boundary metadata, old-project audit, migration log',
    'data/static/sfincs_base': 'SFINCS 200 m base grid (v1 block average)',
    'data/irma/raw': 'Irma raw downloads and MRMS cache',
    'data/irma/processed': 'Irma processed forcing, SFINCS forcing, observation validation',
    'data/irma/meta': 'Irma data-quality records',
    'data/irma/scenarios': 'Irma scenarios (not generated)',
    'runs/irma': 'Irma simulation runs',
    'result/figures': 'visualization PNGs',
    'result/reports': 'reports',
    'result/manifests': 'manifests',
    'archive': 'archive',
    'code': 'scripts',
}

# reuse-source mapping: new file -> source in the old project (read-only copy or reprocessing)
REUSE = [
    ('data/irma/raw/noaa_8720218_water_level_raw.json',
     'outputs/boundary_research/noaa/8720218_water_level_raw.json',
     'direct copy', 'NOAA CO-OPS API raw response'),
    ('data/irma/raw/noaa_8720218_metadata.json',
     'outputs/boundary_research/noaa/8720218_metadata.json',
     'direct copy', 'NOAA CO-OPS mdapi station metadata (incl. datum table)'),
    ('data/irma/raw/usgs_02246500_00060_raw.csv',
     'outputs/boundary_research/usgs/02246500_00060.csv',
     'direct copy', 'USGS NWIS IV signed total discharge, not de-tided'),
    ('data/irma/processed/mayport_8720218_waterlevel_6min_utc_navd88_m.csv',
     'outputs/boundary_research/noaa/8720218_water_level_raw.json',
     're-parsed raw JSON', 'keeps the 4-digit flag and quality per record; cropped to the 10-day window'),
    ('data/irma/processed/mayport_8720218_waterlevel_hourly_utc_navd88_m.csv',
     'same as above', 'hourly aggregation', '240 records, label = interval end time, no gap filling'),
    ('data/irma/processed/acosta_02246500_discharge_15min_utc_m3s.csv',
     'outputs/boundary_research/usgs/02246500_00060.csv',
     're-parsed + independent unit recomputation', 'sign and qualifier kept'),
    ('data/irma/processed/acosta_02246500_discharge_hourly_utc_m3s.csv',
     'same as above', 'hourly aggregation', '240 records, 93 hours with insufficient coverage flagged'),
    ('data/irma/processed/mrms_gaugecorr_qpe01h_jax_0p01deg.nc',
     'data/mrms_raw_old/GaugeCorr_QPE_01H_00.00_*.grib2 (240 files)',
     'decoded directly with a self-written GRIB2 decoder', 'MRMS native 0.01-degree grid kept, not reprojected'),
    ('data/static/terrain/dem_usgs_60m_26917.tif',
     'data/dem_crm_hu8/merged_jacksonville_dem_1arc.tif',
     'crop + bilinear resampling', 'USGS 3DEP 1 arc-second (about 30 m), EPSG:4269->26917'),
    ('data/static/terrain/crm_60m_26917.tif',
     'data/dem_crm_hu8/crm_3_arc.tiff',
     'crop + bilinear resampling', 'NOAA CRM 3 arc-second; vertical datum unconfirmed, not converted'),
    ('data/static/terrain/topobathy_merged_60m_26917.tif',
     'above two', 'per-cell minimum of valid values', 'no vertical datum conversion -- see open item D1'),
    ('data/static/gis/bnd_downstream_waterlevel.geojson',
     'data/WBD_03_HU2_Shape/Shape/WBDHU10.shp + stations above + merged terrain',
     'newly created', 'HUC10 / rectangle / gauges / candidate boundaries / locations to verify'),
]


def main():
    rows = []
    for dp, dn, fn in os.walk(NEW):
        dn[:] = [d for d in dn if d not in ('__pycache__', '.ipynb_checkpoints')]
        for f in sorted(fn):
            if f.startswith('.'):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, NEW)
            top = '/'.join(rel.split(os.sep)[:2])
            rows.append(dict(
                rel_path=rel,
                role=next((v for k, v in ROLE.items() if rel.startswith(k + os.sep) or rel.startswith(k)), ''),
                bytes=os.path.getsize(p),
                mtime_utc=datetime.datetime.utcfromtimestamp(
                    os.path.getmtime(p)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                sha256=sha256(p)))
    with open(os.path.join(MAN, 'deliverable_manifest.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'Deliverable files: {len(rows)}, total '
          f'{sum(r["bytes"] for r in rows)/1048576:.1f} MB')

    ru = []
    for newp, oldp, how, note in REUSE:
        np_ = os.path.join(NEW, newp)
        op = os.path.join(OLD, oldp) if not oldp.startswith('same') and '*' not in oldp \
            and not oldp.startswith('above') else None
        ru.append(dict(
            new_path=newp,
            new_sha256=sha256(np_) if os.path.exists(np_) else 'MISSING',
            new_bytes=os.path.getsize(np_) if os.path.exists(np_) else 0,
            old_source=oldp,
            old_sha256=sha256(op) if op and os.path.exists(op) else '',
            processing=how, note=note))
    with open(os.path.join(MAN, 'reuse_sources.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ru[0]))
        w.writeheader()
        w.writerows(ru)
    print(f'Reuse-source mappings: {len(ru)}')
    print('Wrote result/manifests/')


if __name__ == '__main__':
    main()
