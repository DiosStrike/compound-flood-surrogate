#!/usr/bin/env python3
"""
01_audit_old_project.py -- full file audit of the old project Independent_Study_Flood.

Read-only on source files. Outputs:
  data/meta/audit_rasters.csv     raster metadata (geometry / CRS / resolution / unit clues)
  data/meta/audit_vectors.csv     vector metadata
  data/meta/audit_tables.csv      CSV / table metadata
  data/meta/audit_checksums.csv   file size and SHA256 (full, in chunks; first 64 MB)
"""
import os, sys, csv, json, hashlib, glob, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

OLD = PTH.OLD
NEW = PTH.ROOT
META = PTH.STATIC_META
os.makedirs(META, exist_ok=True)

SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', 'mrms_raw_old'}


def sha256(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            yield os.path.join(dp, f)


def epsg_of(gk):
    if not gk:
        return ''
    if gk.get('ProjectedCSType'):
        return f"EPSG:{gk['ProjectedCSType']}"
    if gk.get('GeographicType'):
        return f"EPSG:{gk['GeographicType']}"
    return ''


def vertical_of(gk, gdalmeta):
    """Best-effort extraction of vertical datum clues; if none found, write UNKNOWN explicitly, never guess."""
    bits = []
    if gk:
        for k in ('VerticalCSType', 'VerticalDatum', 'VerticalCitation',
                  'VerticalUnits'):
            if gk.get(k) not in (None, ''):
                bits.append(f'{k}={gk[k]}')
    if gdalmeta and isinstance(gdalmeta, str):
        low = gdalmeta.lower()
        for kw in ('navd', 'mllw', 'msl', 'mhw', 'ngvd', 'egm', 'datum'):
            if kw in low:
                bits.append(f'GDALMeta contains "{kw}"')
    return '; '.join(bits) if bits else 'UNKNOWN (file declares no vertical datum)'


def main():
    rasters, vectors, tables, sums = [], [], [], []
    for p in sorted(walk(OLD)):
        rel = os.path.relpath(p, OLD)
        ext = os.path.splitext(p)[1].lower()
        try:
            size = os.path.getsize(p)
            mtime = datetime.datetime.utcfromtimestamp(
                os.path.getmtime(p)).strftime('%Y-%m-%dT%H:%M:%SZ')
        except OSError:
            continue
        if ext in ('.tif', '.tiff'):
            row = {'rel_path': rel, 'bytes': size, 'mtime_utc': mtime}
            try:
                m = geoio.tif_meta(p)
                gk = m.get('geokeys', {})
                b = m.get('bounds', (None,) * 4)
                px = m.get('pixel_size', (None, None))
                row.update({
                    'width': m.get('width'), 'height': m.get('height'),
                    'bands': m.get('bands'),
                    'px_x': px[0], 'px_y': px[1],
                    'xmin': b[0], 'ymin': b[1], 'xmax': b[2], 'ymax': b[3],
                    'crs': epsg_of(gk),
                    'crs_citation': (gk.get('PCSCitation') or
                                     gk.get('GeogCitation') or
                                     gk.get('GTCitation') or ''),
                    'vertical_clue': vertical_of(gk, m.get('GDAL_METADATA')),
                    'nodata': m.get('nodata', ''),
                    'bits': m.get('BitsPerSample'),
                    'sampleformat': m.get('SampleFormat'),
                    'compression': m.get('Compression'),
                    'tiled': m.get('tiled'),
                    'bigtiff': m.get('bigtiff'),
                    'gdal_metadata': (m.get('GDAL_METADATA') or '')
                                     .replace('\n', ' ')[:800],
                    'read_status': 'OK',
                })
            except Exception as e:
                row['read_status'] = f'FAIL: {type(e).__name__}: {e}'
            rasters.append(row)
        elif ext == '.shp':
            try:
                shapes, recs, fields, prj = geoio.read_shapefile(p)
                xs = [s['bbox'] for s in shapes if s and 'bbox' in s and len(s['bbox']) == 4]
                if xs:
                    bb = (min(x[0] for x in xs), min(x[1] for x in xs),
                          max(x[2] for x in xs), max(x[3] for x in xs))
                else:
                    bb = (None,) * 4
                vectors.append({
                    'rel_path': rel, 'bytes': size, 'mtime_utc': mtime,
                    'n_features': len(shapes),
                    'geom_type': shapes[0]['type'] if shapes and shapes[0] else '',
                    'fields': '|'.join(f[0] for f in fields),
                    'xmin': bb[0], 'ymin': bb[1], 'xmax': bb[2], 'ymax': bb[3],
                    'prj': prj.replace('\n', ' ')[:400],
                    'read_status': 'OK'})
            except Exception as e:
                vectors.append({'rel_path': rel, 'bytes': size,
                                'mtime_utc': mtime,
                                'read_status': f'FAIL: {e}'})
        elif ext == '.gpkg':
            try:
                import sqlite3
                db = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
                rows = db.execute(
                    'SELECT table_name, data_type, srs_id, min_x, min_y,'
                    ' max_x, max_y FROM gpkg_contents').fetchall()
                for t, dt, srs, x0, y0, x1, y1 in rows:
                    try:
                        n = db.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                        cols = [c[1] for c in db.execute(
                            f'PRAGMA table_info("{t}")').fetchall()]
                    except Exception:
                        n, cols = -1, []
                    vectors.append({
                        'rel_path': f'{rel}::{t}', 'bytes': size,
                        'mtime_utc': mtime, 'n_features': n,
                        'geom_type': dt, 'fields': '|'.join(cols),
                        'xmin': x0, 'ymin': y0, 'xmax': x1, 'ymax': y1,
                        'prj': f'srs_id={srs}', 'read_status': 'OK'})
                db.close()
            except Exception as e:
                vectors.append({'rel_path': rel, 'bytes': size,
                                'mtime_utc': mtime,
                                'read_status': f'FAIL: {e}'})
        elif ext in ('.csv',):
            try:
                with open(p, 'r', errors='replace') as f:
                    head = f.readline().strip()
                    first = f.readline().strip()
                    n = 2 + sum(1 for _ in f)
                tables.append({'rel_path': rel, 'bytes': size,
                               'mtime_utc': mtime, 'n_lines': n,
                               'header': head[:400], 'first_row': first[:400]})
            except Exception as e:
                tables.append({'rel_path': rel, 'bytes': size,
                               'mtime_utc': mtime, 'n_lines': -1,
                               'header': f'FAIL: {e}', 'first_row': ''})
        elif ext == '.nc':
            tables.append({'rel_path': rel, 'bytes': size, 'mtime_utc': mtime,
                           'n_lines': -1, 'header': 'NetCDF (see audit_netcdf)',
                           'first_row': ''})
        # checksums: only for data files < 300 MB
        if ext in ('.tif', '.tiff', '.nc', '.csv', '.gpkg', '.shp', '.asc') \
                and size < 300 * (1 << 20):
            sums.append({'rel_path': rel, 'bytes': size,
                         'sha256': sha256(p)})

    def dump(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(META, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'{name}: {len(rows)} rows')

    dump('audit_rasters.csv', rasters)
    dump('audit_vectors.csv', vectors)
    dump('audit_tables.csv', tables)
    dump('audit_checksums.csv', sums)


if __name__ == '__main__':
    main()
