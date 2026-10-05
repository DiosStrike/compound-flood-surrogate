#!/usr/bin/env python3
"""
02_define_extent.py -- define the study rectangle and verify the in-house UTM projection.

Steps:
 1. Verify geoio.ll_to_utm against the EPSG:26917 station GeoPackage produced by QGIS in the old project
    (QGIS uses PROJ, so it serves as an independent reference).
 2. Extract the two HUC10 polygons from WBD and project them to EPSG:26917.
 3. Compute the common bounding rectangle of HUC10 + Acosta + Mayport.
 4. Round outward to a multiple of 600 m (least common multiple of the 60 m delivery grid and the 200 m model grid,
    so both nest cleanly) and record the actual extent.
Outputs: data/meta/study_extent.json, data/gis/huc10_26917.json (intermediate)
"""
import os, sys, json, struct, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

OLD = PTH.OLD
NEW = PTH.ROOT
ZONE = 17

STATIONS = {
    'USGS_02246500': dict(
        name='ST. JOHNS RIVER AT JACKSONVILLE, FL (Acosta Bridge)',
        agency='USGS', sid='02246500', lon=-81.6653735, lat=30.3224616,
        role='upstream_discharge_reference',
        src='NWIS IV sourceInfo.geoLocation (srs reported as EPSG:4326; '
            'NWIS actually stores NAD83; difference <1 m)'),
    'NOAA_8720218': dict(
        name='Mayport (Bar Pilots Dock)',
        agency='NOAA CO-OPS', sid='8720218', lon=-81.42789, lat=30.398167,
        role='sea_side_water_level_reference',
        src='CO-OPS mdapi stations/8720218 (noaa_stations.geojson)'),
}

HUC10S = ['0308010315', '0308010316']


def gpkg_points(path):
    """Read coordinates of a GeoPackage point layer (for verification)."""
    out = []
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    for (t,) in db.execute("SELECT table_name FROM gpkg_contents"
                           " WHERE data_type='features'"):
        cols = [c[1] for c in db.execute(f'PRAGMA table_info("{t}")')]
        gcol = 'geom' if 'geom' in cols else cols[1]
        for row in db.execute(f'SELECT * FROM "{t}"'):
            rec = dict(zip(cols, row))
            blob = rec[gcol]
            if not blob or blob[:2] != b'GP':
                continue
            flags = blob[3]
            env = (flags >> 1) & 0x07
            envsz = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
            wkb = blob[8 + envsz:]
            bo = '<' if wkb[0] == 1 else '>'
            gt = struct.unpack(bo + 'I', wkb[1:5])[0] % 1000
            if gt == 1:
                x, y = struct.unpack(bo + 'dd', wkb[5:21])
                attrs = {k: v for k, v in rec.items() if k != gcol}
                out.append((t, x, y, attrs))
    db.close()
    return out


def validate_projection():
    print('=== 1. projection check (against the 26917 point layer produced by QGIS/PROJ) ===')
    checks = []
    for g in ['QGIS/hu8_study_area/noaa_station.gpkg',
              'QGIS/hu8_study_area/usgs_station.gpkg']:
        p = os.path.join(OLD, g)
        if not os.path.exists(p):
            continue
        for t, x, y, attrs in gpkg_points(p):
            print(f'  {g}::{t}  x={x:.3f} y={y:.3f}  attrs={attrs}')
            checks.append((g, t, x, y, attrs))
    # forward projection of known stations for comparison
    for key, s in STATIONS.items():
        x, y = geoio.ll_to_utm(s['lon'], s['lat'], ZONE)
        print(f'  this implementation {key}: x={float(x):.3f} y={float(y):.3f}')
        # round-trip accuracy
        lo, la = geoio.utm_to_ll(x, y, ZONE)
        d = np.hypot((float(lo) - s['lon']) * 96000, (float(la) - s['lat']) * 111000)
        print(f'       round-trip residual ≈ {d*1000:.3f} mm')
    return checks


def main():
    checks = validate_projection()

    print('\n=== 2. extract HUC10 ===')
    shp = os.path.join(OLD, 'data/WBD_03_HU2_Shape/Shape/WBDHU10.shp')
    shapes, recs, fields, prj = geoio.read_shapefile(shp)
    hu = {}
    for i, r in enumerate(recs):
        code = str(r.get('huc10', '')).strip()
        if code in HUC10S:
            s = shapes[i]
            rings_ll = s['parts']
            rings_utm = []
            for ring in rings_ll:
                a = np.array(ring)
                x, y = geoio.ll_to_utm(a[:, 0], a[:, 1], ZONE)
                rings_utm.append(np.column_stack([x, y]))
            hu[code] = dict(
                name=r.get('name'), areasqkm=r.get('areasqkm'),
                areaacres=r.get('areaacres'), states=r.get('states'),
                hutype=r.get('hutype'), humod=r.get('humod'),
                loaddate=r.get('loaddate'), tnmid=r.get('tnmid'),
                sourcedata=r.get('sourcedata'),
                referenceg=r.get('referenceg'),
                n_rings=len(rings_ll),
                n_vertices=int(sum(len(x) for x in rings_ll)),
                bbox_ll=[float(v) for v in s['bbox']],
                rings_utm=[r_.tolist() for r_ in rings_utm])
            b = np.vstack(rings_utm)
            print(f"  {code} {r.get('name')}  area={r.get('areasqkm')} km2"
                  f"  vertices={hu[code]['n_vertices']}  loaddate={r.get('loaddate')}")
            print(f"     UTM bbox = {b[:,0].min():.1f} {b[:,1].min():.1f}"
                  f" {b[:,0].max():.1f} {b[:,1].max():.1f}")
    missing = [c for c in HUC10S if c not in hu]
    if missing:
        raise SystemExit(f'HUC10 not found in WBD: {missing}')

    print('\n=== 3. common bounding rectangle ===')
    allpts = [np.array(r) for c in hu for r in hu[c]['rings_utm']]
    stat_xy = []
    for key, s in STATIONS.items():
        x, y = geoio.ll_to_utm(s['lon'], s['lat'], ZONE)
        s['x_26917'], s['y_26917'] = float(x), float(y)
        stat_xy.append([float(x), float(y)])
    P = np.vstack(allpts + [np.array(stat_xy)])
    raw = dict(xmin=float(P[:, 0].min()), ymin=float(P[:, 1].min()),
               xmax=float(P[:, 0].max()), ymax=float(P[:, 1].max()))
    print('  unrounded:', {k: round(v, 2) for k, v in raw.items()})

    ALIGN = 600.0     # least common multiple of 60 m and 200 m
    ext = dict(
        xmin=float(np.floor(raw['xmin'] / ALIGN) * ALIGN),
        ymin=float(np.floor(raw['ymin'] / ALIGN) * ALIGN),
        xmax=float(np.ceil(raw['xmax'] / ALIGN) * ALIGN),
        ymax=float(np.ceil(raw['ymax'] / ALIGN) * ALIGN))
    W = ext['xmax'] - ext['xmin']
    H = ext['ymax'] - ext['ymin']
    print('  rounded:', {k: round(v, 1) for k, v in ext.items()})
    print(f'  width {W/1000:.1f} km x height {H/1000:.1f} km'
          f'  -> 60 m: {int(W/60)} x {int(H/60)} cells'
          f' | 200 m: {int(W/200)} x {int(H/200)} cells')

    # lon/lat bounding box (for cropping 4269/4326 source rasters; corners + edge sampling ensure full coverage)
    xs = np.linspace(ext['xmin'], ext['xmax'], 200)
    ys = np.linspace(ext['ymin'], ext['ymax'], 200)
    gx, gy = np.meshgrid(xs, ys)
    lo, la = geoio.utm_to_ll(gx.ravel(), gy.ravel(), ZONE)
    ll = dict(lonmin=float(lo.min()), lonmax=float(lo.max()),
              latmin=float(la.min()), latmax=float(la.max()))
    print('  corresponding lon/lat bounding box (NAD83):',
          {k: round(v, 6) for k, v in ll.items()})

    out = dict(
        generated_utc=__import__('datetime').datetime.utcnow()
            .strftime('%Y-%m-%dT%H:%M:%SZ'),
        crs='EPSG:26917 (NAD83 / UTM zone 17N)',
        align_m=ALIGN,
        align_rationale='600 m = LCM(60 m delivery grid, 200 m later model grid), '
                        'so both grids align cleanly with the rectangle boundary',
        rect_raw=raw, rect_aligned=ext,
        width_m=W, height_m=H,
        n_cells_60m=[int(W / 60), int(H / 60)],
        n_cells_200m=[int(W / 200), int(H / 200)],
        latlon_bbox_nad83=ll,
        huc10={c: {k: v for k, v in hu[c].items() if k != 'rings_utm'}
               for c in hu},
        stations=STATIONS,
        note='rectangle sheet != HUC10 boundary != final model active domain; the three are maintained separately in this project')

    os.makedirs(PTH.STATIC_META, exist_ok=True)
    os.makedirs(PTH.STATIC_GIS, exist_ok=True)
    with open(os.path.join(PTH.STATIC_META, 'study_extent.json'), 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(PTH.STATIC_GIS, 'huc10_rings_26917.json'), 'w') as f:
        json.dump({c: hu[c]['rings_utm'] for c in hu}, f)
    print('\nwritten data/meta/study_extent.json and data/gis/huc10_rings_26917.json')


if __name__ == '__main__':
    main()
