#!/usr/bin/env python3
"""
12_apply_v2.py -- replace the v1 active domain and boundaries with v2 (redone by the user in QGIS)

Kept unchanged: huc10.geojson, study_rectangle.geojson, stations_forcing.geojson,
        topobathy_merged_60m_26917.tif, ..._huc10.tif, dem/crm components,
        all water-level / discharge / rainfall data.
Archived: v1 active-domain mask, active-domain clipped terrain, both boundary GeoJSONs, boundary JSON -> data/archive_v1/
Created: v2 active-domain polygon / mask / clipped terrain, the two cleaned boundary GeoJSONs, boundaries_v2.json
"""
import os, sys, json, shutil, datetime, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

NEW = PTH.ROOT
QD, GIS = PTH.STATIC_MYQGIS, PTH.STATIC_GIS
PROC, INT = PTH.STATIC_TERRAIN, PTH.STATIC_INTERIM
META = PTH.STATIC_META
ZONE, DX, NODATA, WET = 17, 60.0, -9999.0, 0.0

ext = json.load(open(os.path.join(META, 'study_extent.json')))
R = ext['rect_aligned']
z = np.load(os.path.join(INT, 'merged60.npy'))
NY, NX = z.shape
TR = (R['xmin'], DX, 0.0, R['ymax'], 0.0, -DX)
ACT2 = np.load(os.path.join(INT, 'active_mask.npy'))

# ---------- 1. Active-domain polygon / mask / clipped terrain ----------
print('1. Write model domain')
shutil.copyfile(os.path.join(QD, 'huc10_extend_active.geojson'),
                os.path.join(GIS, 'active_domain.geojson'))
shutil.copyfile(os.path.join(QD, 'huc10_extend.geojson'),
                os.path.join(GIS, 'huc10_extend_parts.geojson'))

geoio.write_geotiff(os.path.join(PROC, 'active_domain_mask_60m_26917.tif'),
                    ACT2.astype(np.float32), TR, 26917, nodata=None,
                    citation='Flood_2.0 active domain mask v2 (1=active, 0=inactive)')
zc = np.where(ACT2, z, NODATA).astype(np.float32)
geoio.write_geotiff(os.path.join(PROC, 'topobathy_merged_60m_26917_active.tif'),
                    zc, TR, 26917, nodata=NODATA,
                    citation='Flood_2.0 topobathy 60m clipped to active domain v2, m NAVD88')
print('   active_domain_mask_60m_26917.tif / topobathy_merged_60m_26917_active.tif')

# ---------- 2. Clean boundaries and recompute attributes ----------
print('2. Clean boundaries and recompute attributes')


def to_utm(ring):
    a = np.asarray(ring, float)
    x, y = geoio.ll_to_utm(a[:, 0], a[:, 1], ZONE)
    return np.c_[x, y]


def densify(p, step=20.0):
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(float(np.hypot(*(b - a))) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out)


pad = np.zeros((NY + 2, NX + 2), bool)
pad[1:-1, 1:-1] = ACT2
EDGE = ACT2 & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
EDGEW = EDGE & (z <= WET)


def measure(pts):
    """Return statistics of the active-domain wet edge cells covered by this boundary line"""
    p = densify(pts)
    ii = np.floor((p[:, 0] - R['xmin']) / DX).astype(int)
    jj = np.floor((R['ymax'] - p[:, 1]) / DX).astype(int)
    cells = set()
    for j, i in zip(jj, ii):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                a, b = j + dj, i + di
                if 0 <= a < NY and 0 <= b < NX and EDGEW[a, b]:
                    cells.add((a, b))
    if not cells:
        return dict(n_cells=0, width_m=0.0, z_min_m=None, flow_area_m2=0.0)
    zz = np.array([z[j, i] for j, i in cells])
    return dict(n_cells=len(cells), width_m=round(len(cells) * DX, 1),
                z_min_m=round(float(zz.min()), 2),
                flow_area_m2=round(float(np.clip(-zz, 0, None).sum() * DX), 1))


out = {}
for key, fn in [('upstream', 'bnd_upstream_discharge.geojson'),
                ('downstream', 'bnd_downstream_waterlevel.geojson')]:
    d = json.load(open(os.path.join(QD, fn)))
    kept, dropped, newly = [], [], []
    # Features newly digitized in QGIS have all-null attributes: inherit the common attributes of existing features in the same layer and assign IDs sequentially
    have = [f for f in d['features']
            if f['geometry'] and f['geometry'].get('coordinates')
            and f['properties'].get('bnd_id')]
    tmpl = dict(have[0]['properties']) if have else {}
    used = {f['properties'].get('bnd_id') for f in have}   # count only kept features; IDs of empty geometries can be reused
    pre = 'UP' if key == 'upstream' else 'DW'
    nxt = 1
    for f in d['features']:
        g = f['geometry']
        if not g or not g.get('coordinates'):
            dropped.append(f['properties'].get('bnd_id'))
            continue
        if not f['properties'].get('bnd_id'):
            while '%s%02d' % (pre, nxt) in used:
                nxt += 1
            bid = '%s%02d' % (pre, nxt)
            used.add(bid)
            for k_, v_ in tmpl.items():
                if f['properties'].get(k_) is None:
                    f['properties'][k_] = v_
            f['properties']['bnd_id'] = bid
            f['properties']['kind'] = 'water-body cut added by the user in QGIS'
            newly.append(bid)
        pts = to_utm(g['coordinates']) if g['type'] == 'LineString' else np.vstack(
            [to_utm(c) for c in g['coordinates']])
        st = measure(pts)
        L = float(np.hypot(*(pts[1:] - pts[:-1]).T).sum())
        p = f['properties']
        p['width_m'] = st['width_m']
        p['z_min_m'] = st['z_min_m']
        p['flow_area_m2'] = st['flow_area_m2']
        p['length_m'] = round(L, 1)
        p['n_edge_cells'] = st['n_cells']
        p['status'] = 'user_edited_v2'
        p['note'] = 'redrawn by the user in QGIS on 2026-09-14; attributes recomputed by 12_apply_v2.py on the 60 m grid'
        p.pop('along_channel_km_to_acosta', None)
        kept.append(f)
    if key == 'upstream':
        tot = sum(f['properties']['flow_area_m2'] for f in kept) or 1.0
        for f in kept:
            f['properties']['Q_fraction'] = round(f['properties']['flow_area_m2'] / tot, 4)
    d['features'] = kept
    json.dump(d, open(os.path.join(GIS, fn), 'w'), ensure_ascii=False, indent=1)
    out[key] = dict(kept=[f['properties']['bnd_id'] for f in kept], dropped=dropped,
                    newly_digitized=newly,
                    features=[f['properties'] for f in kept])
    print('   %s: kept %s%s, removed %d empty geometries'
          % (fn, out[key]['kept'],
             (' (of which %s added in QGIS, IDs auto-assigned)' % ','.join(newly)) if newly else '',
             len(dropped)))

# ---------- 3. Edge-closure check record ----------
cov = set()
for key, fn in [('upstream', 'bnd_upstream_discharge.geojson'),
                ('downstream', 'bnd_downstream_waterlevel.geojson')]:
    d = json.load(open(os.path.join(GIS, fn)))
    for f in d['features']:
        g = f['geometry']
        pts = to_utm(g['coordinates']) if g['type'] == 'LineString' else np.vstack(
            [to_utm(c) for c in g['coordinates']])
        p = densify(pts)
        ii = np.floor((p[:, 0] - R['xmin']) / DX).astype(int)
        jj = np.floor((R['ymax'] - p[:, 1]) / DX).astype(int)
        for j, i in zip(jj, ii):
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    a, b = j + dj, i + di
                    if 0 <= a < NY and 0 <= b < NX and EDGEW[a, b]:
                        cov.add((a, b))
UNC = np.zeros(z.shape, bool)
for j, i in zip(*np.where(EDGEW)):
    if (j, i) not in cov:
        UNC[j, i] = True
np.save(os.path.join(INT, 'uncovered_edge.npy'), UNC)


def comps(mask):
    ny, nx = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    off = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sj, si in zip(*np.where(mask)):
        if lab[sj, si]:
            continue
        cur += 1
        st = [(sj, si)]
        lab[sj, si] = cur
        while st:
            j, i = st.pop()
            for dj, di in off:
                a, b = j + dj, i + di
                if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not lab[a, b]:
                    lab[a, b] = cur
                    st.append((a, b))
    return lab, cur


lab, n = comps(UNC)
cl = []
for c in range(1, n + 1):
    jj, ii = np.where(lab == c)
    zz = z[jj, ii]
    cl.append(dict(n_cells=int(jj.size), width_m=round(float(jj.size) * DX, 1),
                   x=round(float(R['xmin'] + (ii.mean() + .5) * DX)),
                   y=round(float(R['ymax'] - (jj.mean() + .5) * DX)),
                   z_min_m=round(float(zz.min()), 2),
                   flow_area_m2=round(float(np.clip(-zz, 0, None).sum() * DX), 1)))
cl.sort(key=lambda c: -c['flow_area_m2'])

summary = collections.OrderedDict(
    generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
    version='user QGIS version (2026-09-14)',
    active_domain=dict(km2=round(ACT2.sum() * DX * DX / 1e6, 2),
                       cells=int(ACT2.sum()),
                       z_min=round(float(z[ACT2].min()), 2),
                       z_max=round(float(z[ACT2].max()), 2)),
    boundaries=out,
    edge_cells=dict(total=int(EDGE.sum()), wet=int(EDGEW.sum()),
                    covered=len(cov), uncovered=int(UNC.sum())),
    uncovered_clusters=cl)
json.dump(summary, open(os.path.join(META, 'boundaries.json'), 'w'),
          ensure_ascii=False, indent=1)
print('3. Edge closure: wet edge cells %d, covered %d, uncovered %d (%d sites)'
      % (EDGEW.sum(), len(cov), UNC.sum(), n))
print('Done -> data/meta/boundaries.json')
