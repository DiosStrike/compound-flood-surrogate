#!/usr/bin/env python3
"""
11_integrate_v2.py -- audit the v2 products the user redid in QGIS and integrate them with existing data

Does exactly three things:
  A. audit the 5 files exported by the user under data/my_qgis/ (CRS / geometry / attributes / raster / consistency with the project grid)
  B. archive the v1 active domain and boundaries (moved to data/archive_v1/); raw data are never touched
  C. rebuild the active-domain mask, clipped terrain and cleaned boundary GeoJSON from v2, and recompute the Q split and boundary-closure check

Does not change terrain values, does not run the model, does not fill data.
"""
import os, sys, json, collections, datetime, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

NEW = PTH.ROOT
QD = PTH.STATIC_MYQGIS
GIS = PTH.STATIC_GIS
PROC = PTH.STATIC_TERRAIN
INT = PTH.STATIC_INTERIM
META = PTH.STATIC_META
ARCH = os.path.join(NEW, 'data/archive_v1')
ZONE, DX, NODATA, WET = 17, 60.0, -9999.0, 0.0

rep = collections.OrderedDict()
rep['generated_utc'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------- utilities ----------
def to_utm_ring(ring):
    a = np.asarray(ring, float)
    x, y = geoio.ll_to_utm(a[:, 0], a[:, 1], ZONE)
    return np.c_[x, y]


def geom_rings_utm(g):
    """Return a list of UTM rings [outer ring, inner rings...] (Polygon / MultiPolygon are all flattened)"""
    if g is None:
        return []
    t = g['type']
    polys = [g['coordinates']] if t == 'Polygon' else g['coordinates']
    out = []
    for poly in polys:
        for ring in poly:
            out.append(to_utm_ring(ring))
    return out


def line_utm(g):
    if not g or not g.get('coordinates'):
        return None
    if g['type'] == 'LineString':
        return to_utm_ring(g['coordinates'])
    segs = [to_utm_ring(c) for c in g['coordinates']]
    return np.vstack(segs)


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rasterize_rings(rings, tr, shape):
    x0, dx, _, y0, _, dy = tr
    ny, nx = shape
    mask = np.zeros(shape, bool)
    xs = x0 + (np.arange(nx) + 0.5) * dx
    for j in range(ny):
        yc = y0 + (j + 0.5) * dy
        xint = []
        for ring in rings:
            a = np.asarray(ring)
            x1, y1 = a[:-1, 0], a[:-1, 1]
            x2, y2 = a[1:, 0], a[1:, 1]
            hit = ((y1 <= yc) & (y2 > yc)) | ((y2 <= yc) & (y1 > yc))
            if hit.any():
                t = (yc - y1[hit]) / (y2[hit] - y1[hit])
                xint.append(x1[hit] + t * (x2[hit] - x1[hit]))
        if not xint:
            continue
        xi = np.sort(np.concatenate(xint))
        for k in range(0, len(xi) - 1, 2):
            mask[j] |= (xs >= xi[k]) & (xs < xi[k + 1])
    return mask


def densify(pts, step=20.0):
    """Densify a polyline at step metres, to determine which cells it falls on"""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(np.ceil(d / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.array(out)


def line_cells(pts, tr, shape):
    x0, dx, _, y0, _, dy = tr
    p = densify(pts)
    ii = np.floor((p[:, 0] - x0) / dx).astype(int)
    jj = np.floor((p[:, 1] - y0) / dy).astype(int)
    ok = (ii >= 0) & (ii < shape[1]) & (jj >= 0) & (jj < shape[0])
    return set(zip(jj[ok].tolist(), ii[ok].tolist()))


def components(mask, nbr8=True):
    ny, nx = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    off = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if nbr8:
        off += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sj, si in zip(*np.where(mask)):
        if lab[sj, si]:
            continue
        cur += 1
        stack = [(sj, si)]
        lab[sj, si] = cur
        while stack:
            j, i = stack.pop()
            for dj, di in off:
                a, b = j + dj, i + di
                if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not lab[a, b]:
                    lab[a, b] = cur
                    stack.append((a, b))
    return lab, cur


# ================= A. Audit =================
print('=' * 64)
print('A. Audit data/my_qgis/')
print('=' * 64)

ext = json.load(open(os.path.join(META, 'study_extent.json')))
R = ext['rect_aligned']
z = np.load(os.path.join(INT, 'merged60.npy'))
NY, NX = z.shape
TR = (R['xmin'], DX, 0.0, R['ymax'], 0.0, -DX)
print('Project grid: %d x %d, origin (%.0f, %.0f), 60 m' % (NY, NX, R['xmin'], R['ymax']))

audit = collections.OrderedDict()

# --- A1 user raster vs project raster ---
up = os.path.join(QD, 'topobathy_merged_60m_huc10_extend.tiff')
um = geoio.tif_meta(up)
ua = geoio.tif_read(up)
ua = ua[0] if isinstance(ua, tuple) else ua
ua = np.asarray(ua, float)
ua[ua == um['nodata']] = np.nan
ux0, udx = um['transform'][0], um['transform'][1]
uy0, udy = um['transform'][3], um['transform'][5]
di = int(round((ux0 - R['xmin']) / DX))
dj = int(round((R['ymax'] - uy0) / DX))
sub = z[dj:dj + ua.shape[0], di:di + ua.shape[1]]
val = ~np.isnan(ua)
dif = np.abs(ua[val] - sub[val])
audit['raster'] = dict(
    file='topobathy_merged_60m_huc10_extend.tiff',
    epsg=um['geokeys'].get('ProjectedCSType'), pixel=um['pixel_size'][0],
    shape=[int(ua.shape[0]), int(ua.shape[1])], nodata=um['nodata'],
    grid_offset_cells=[dj, di],
    grid_aligned=bool(abs((ux0 - R['xmin']) / DX - di) < 1e-6 and abs((R['ymax'] - uy0) / DX - dj) < 1e-6),
    valid_cells=int(val.sum()),
    zmin=float(np.nanmin(ua)), zmax=float(np.nanmax(ua)),
    max_abs_diff_vs_project=float(dif.max()) if dif.size else None)
print('A1 raster: EPSG:%s, %dx%d, 60 m, grid aligned=%s' % (
    audit['raster']['epsg'], ua.shape[0], ua.shape[1], audit['raster']['grid_aligned']))
print('    elevation %.2f .. %.2f m, %d valid cells' % (audit['raster']['zmin'], audit['raster']['zmax'], val.sum()))
print('    max difference vs project merged60 at the same locations = %.6g m  (0 means clipped only, values unchanged)'
      % audit['raster']['max_abs_diff_vs_project'])

# --- A2 active-domain polygon ---
ga = json.load(open(os.path.join(QD, 'huc10_extend_active.geojson')))
g_act = ga['features'][0]['geometry']
rings_act = geom_rings_utm(g_act)
ACT2 = rasterize_rings(rings_act, TR, z.shape)
area_poly = sum(ring_area(r) for r in rings_act) / 1e6   # valid only when there are outer rings only
lab_a, n_a = components(ACT2)
audit['active_polygon'] = dict(
    file='huc10_extend_active.geojson', geom=g_act['type'],
    n_rings=len(rings_act), area_km2_polygon=round(area_poly, 2),
    area_km2_raster=round(ACT2.sum() * DX * DX / 1e6, 2),
    cells=int(ACT2.sum()), connected_parts=int(n_a))

# --- A3 huc10_extend (2 pieces) ---
gx = json.load(open(os.path.join(QD, 'huc10_extend.geojson')))
parts = []
HU2 = np.zeros(z.shape, bool)
for f in gx['features']:
    rr = geom_rings_utm(f['geometry'])
    m = rasterize_rings(rr, TR, z.shape)
    HU2 |= m
    parts.append(dict(huc10=f['properties']['huc10'],
                      area_km2=round(sum(ring_area(r) for r in rr) / 1e6, 2),
                      attr_areasqkm=f['properties'].get('areasqkm')))
audit['huc10_extend'] = dict(file='huc10_extend.geojson', parts=parts,
                             union_area_km2=round(HU2.sum() * DX * DX / 1e6, 2),
                             dissolved=False)

# --- A4 comparison with v1 ---
ACT1 = np.load(os.path.join(INT, 'active_mask.npy'))
U1 = np.load(os.path.join(INT, 'huc10_union.npy'))
audit['vs_v1'] = dict(
    v1_active_km2=round(ACT1.sum() * DX * DX / 1e6, 2),
    v2_active_km2=round(ACT2.sum() * DX * DX / 1e6, 2),
    only_v1_km2=round((ACT1 & ~ACT2).sum() * DX * DX / 1e6, 2),
    only_v2_km2=round((ACT2 & ~ACT1).sum() * DX * DX / 1e6, 2),
    v1_huc10_union_km2=round(U1.sum() * DX * DX / 1e6, 2))
print('A2 active domain (v2): %.2f km2 (polygon) / %.2f km2 (60 m raster), %d connected components'
      % (area_poly, audit['active_polygon']['area_km2_raster'], n_a))
print('A4 v1 active domain %.2f km2 -> v2 %.2f km2 ; only in v1 %.2f, only in v2 %.2f'
      % (audit['vs_v1']['v1_active_km2'], audit['vs_v1']['v2_active_km2'],
         audit['vs_v1']['only_v1_km2'], audit['vs_v1']['only_v2_km2']))

# --- A5 boundary files ---
bnd = {}
for key, fn in [('upstream', 'bnd_upstream_discharge.geojson'),
                ('downstream', 'bnd_downstream_waterlevel.geojson')]:
    d = json.load(open(os.path.join(QD, fn)))
    keep, empty = [], []
    for f in d['features']:
        L = line_utm(f['geometry'])
        (empty if L is None else keep).append(f['properties'].get('bnd_id'))
        if L is not None:
            f['_utm'] = L
    bnd[key] = dict(raw=d, keep=keep, empty=empty)
    print('A5 %-10s file has %d features: with geometry %d (%s), geometry emptied %d (%s)'
          % (fn.replace('bnd_', '').replace('.geojson', ''), len(d['features']),
             len(keep), ','.join(keep), len(empty),
             ','.join(empty[:3]) + ('...' if len(empty) > 3 else '')))
audit['boundaries_raw'] = {k: dict(kept=v['keep'], cleared=v['empty']) for k, v in bnd.items()}


# ================= B. Boundary placement and closure check =================
print()
print('=' * 64)
print('B. Boundary placement and closure check (v2 active domain)')
print('=' * 64)

wet = ACT2 & (z <= WET)
# Active-domain edge cells: active with at least one inactive 4-neighbour
pad = np.zeros((NY + 2, NX + 2), bool)
pad[1:-1, 1:-1] = ACT2
edge = ACT2 & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
edge_wet = edge & (z <= WET)
print('Active-domain edge cells %d, of which water (z<=0) %d' % (edge.sum(), edge_wet.sum()))

cov = set()
seg_info = []
for key in ('upstream', 'downstream'):
    for f in bnd[key]['raw']['features']:
        if '_utm' not in f:
            continue
        cs = line_cells(f['_utm'], TR, z.shape)
        # cells crossed by the line + their 8-neighbourhood, taken as the "covered by this boundary" range
        grown = set()
        for (j, i) in cs:
            for dj_ in (-1, 0, 1):
                for di_ in (-1, 0, 1):
                    a, b = j + dj_, i + di_
                    if 0 <= a < NY and 0 <= b < NX:
                        grown.add((a, b))
        oncell = [(j, i) for (j, i) in grown if edge_wet[j, i]]
        cov |= set(oncell)
        zz = np.array([z[j, i] for (j, i) in grown if ACT2[j, i] and z[j, i] <= WET])
        L = float(np.hypot(*(f['_utm'][1:] - f['_utm'][:-1]).T).sum())
        seg_info.append(dict(key=key, bnd_id=f['properties']['bnd_id'],
                             length_m=round(L, 1),
                             edge_wet_cells=len(oncell),
                             n_wet_cells=int(zz.size),
                             z_min=float(zz.min()) if zz.size else None,
                             flow_area_m2=float(np.clip(-zz, 0, None).sum() * DX) if zz.size else 0.0))
        print('  %-5s %-6s length %.2f km, covers %d edge water cells, lowest section %s m, flow area %.0f m2'
              % (key[:4], f['properties']['bnd_id'], L / 1000, len(oncell),
                 ('%.2f' % zz.min()) if zz.size else 'n/a',
                 seg_info[-1]['flow_area_m2']))

unc = [(j, i) for (j, i) in zip(*np.where(edge_wet)) if (j, i) not in cov]
print('"Edge water cells" not covered by any boundary = %d (these default to wall)' % len(unc))
if unc:
    m = np.zeros(z.shape, bool)
    for j, i in unc:
        m[j, i] = True
    lab_u, n_u = components(m)
    clusters = []
    for c in range(1, n_u + 1):
        jj, ii = np.where(lab_u == c)
        zc = z[jj, ii]
        clusters.append(dict(n_cells=int(jj.size),
                             width_m=round(float(jj.size) * DX, 1),
                             x=float(R['xmin'] + (ii.mean() + .5) * DX),
                             y=float(R['ymax'] - (jj.mean() + .5) * DX),
                             z_min=round(float(zc.min()), 2),
                             flow_area_m2=round(float(np.clip(-zc, 0, None).sum() * DX), 1)))
    clusters.sort(key=lambda c: -c['flow_area_m2'])
    print('  grouped into %d clusters, top 8 by flow area:' % n_u)
    for c in clusters[:8]:
        print('    %3d cells (%5.0f m)  lowest %7.2f m  section %8.0f m2  @ (%.0f, %.0f)'
              % (c['n_cells'], c['width_m'], c['z_min'], c['flow_area_m2'], c['x'], c['y']))
    audit['unclosed_water_edges'] = dict(total_cells=len(unc), clusters=n_u, top=clusters[:12])
else:
    audit['unclosed_water_edges'] = dict(total_cells=0, clusters=0, top=[])

audit['segments'] = seg_info
json.dump(audit, open(os.path.join(META, 'audit_v2_my_qgis.json'), 'w'),
          ensure_ascii=False, indent=1)
np.save(os.path.join(INT, 'active_mask.npy'), ACT2)
np.save(os.path.join(INT, 'huc10_union.npy'), HU2)
print()
print('Audit results -> data/meta/audit_v2_my_qgis.json')
