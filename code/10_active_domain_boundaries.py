#!/usr/bin/env python3
"""
10_active_domain_boundaries.py -- active domain and boundary definition

Active domain = union of two HUC10 basins + seaward extension to the -2 m depth contour.
Based on Leijnse et al. (2021) Coastal Engineering 163:103796 (the original SFINCS paper,
whose §4.1 application case is precisely Hurricane Irma in Jacksonville):
  §2.3 "For coastal applications, SFINCS models are typically forced along the
        2-m depth contour ... Inactive cells can be cells located below or above
        certain elevation thresholds (e.g. cells in deeper water beyond the
        seaward edge of the swash zone)."
  §4.1 "The active SFINCS grid starts from the MSL-2m contour line at the
        seaward side of the domain."
The paper uses m+MSL, this project uses NAVD88 (local MSL is 0.1585 m below NAVD88, so the paper's MSL-2 m
is about NAVD88 -2.16 m). Per user instruction, -2.0 m in this project's datum is used directly.

Implementation of the seaward extension: first define the connected body "reachable from the east edge of the rectangle, outside HUC10, elevation <= -2 m"
as the **open ocean**; then, row by row, advance east from the easternmost HUC10 cell until **hitting the open ocean**.
If a row reaches the map edge without hitting the open ocean (i.e. the row is not coastal), that row is not extended.
This avoids the earlier problem of "advance row by row until z<=-2" running out more than ten kilometres in non-coastal rows.

Boundary classification (user instruction 2026-09-14):
  * water-body cross-sections near the discharge point (upstream of Acosta, <= 10 km along the channel) -> upstream discharge boundary
  * all other water-body cross-sections (incl. the seaward -2 m contour and the tributaries) -> downstream water-level boundary
  * all other domain boundaries                                       -> default wall, no file written
If there are several parallel upstream cross-sections, Q is split in proportion to flow cross-section area; the full discharge is not applied repeatedly.
"""
import os, sys, json, collections, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

NEW = PTH.ROOT
ZONE = 17
DX = 60.0
SEAWARD_CONTOUR = -2.0      # m, project datum (NAVD88)
WET = 0.0                   # m, elevation threshold for "water body"
MAX_EXT_CELLS = 80          # safety cap for seaward extension per row (4.8 km); beyond it the row is considered non-coastal
NODATA = -9999.0

ACOSTA = dict(id='02246500', agency='USGS',
              name='ST. JOHNS RIVER AT JACKSONVILLE, FL (Acosta Bridge)',
              lon=-81.6653735, lat=30.3224616,
              role='data source of the upstream discharge boundary (gauge)',
              param='00060 un-detided signed total discharge')
MAYPORT = dict(id='8720218', agency='NOAA CO-OPS', name='Mayport (Bar Pilots Dock)',
               lon=-81.42789, lat=30.398167,
               role='data source of the downstream water-level boundary (gauge)',
               param='water_level observed total water level (NAVD88)')
OTHER_STATIONS = [
    dict(id='8720219', agency='NOAA', name='Dames Point',
         lon=-81.5583, lat=30.3867, data='2017-09-03..09-20 6-minute NAVD88 water level, incl. 47 official estimated records during the event'),
    dict(id='8720226', agency='NOAA', name='Southbank Riverwalk, St Johns River',
         lon=-81.6583, lat=30.32, data='2017-09-03..09-20 6-minute NAVD88 water level, incl. 8 official estimated records during the event'),
    dict(id='02246515', agency='USGS', name='POTTSBURG CREEK NR S JACKSONVILLE',
         lon=-81.5900917, lat=30.2644071, data='00065 stage / 00060 discharge / 63160'),
    dict(id='02246621', agency='USGS', name='TROUT R NR JACKSONVILLE',
         lon=-81.69648758, lat=30.41746058, data='00065 stage / 00060 discharge / 72137'),
    dict(id='02246751', agency='USGS', name='BROWARD RIVER BL BISCAYNE BLVD',
         lon=-81.6682, lat=30.44333889, data='00065 stage / 00060 discharge / 72137'),
    dict(id='02246804', agency='USGS', name='DUNN CREEK AT DUNN CREEK RD',
         lon=-81.59688889, lat=30.45494167, data='00065 stage / 00060 discharge / 72137'),
    dict(id='02246825', agency='USGS', name='CLAPBOARD CREEK NR JACKSONVILLE',
         lon=-81.5182611, lat=30.44838056, data='00065 stage / 00060 discharge / 72137'),
]

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def components(mask):
    ny, nx = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    for sj, si in zip(*np.where(mask)):
        if lab[sj, si]:
            continue
        cur += 1
        q = collections.deque([(sj, si)])
        lab[sj, si] = cur
        while q:
            j, i = q.popleft()
            for dj, di in NB8:
                a, b = j + dj, i + di
                if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not lab[a, b]:
                    lab[a, b] = cur
                    q.append((a, b))
    return lab, cur


def geodesic(mask, seeds):
    ny, nx = mask.shape
    d = np.full(mask.shape, -1, np.int32)
    q = collections.deque()
    for j, i in seeds:
        if 0 <= j < ny and 0 <= i < nx and mask[j, i]:
            d[j, i] = 0
            q.append((j, i))
    while q:
        j, i = q.popleft()
        nd = d[j, i] + 1
        for dj, di in NB8:
            a, b = j + dj, i + di
            if 0 <= a < ny and 0 <= b < nx and mask[a, b] and d[a, b] < 0:
                d[a, b] = nd
                q.append((a, b))
    return d


def order_path(cells):
    """Order a set of cells into an as-continuous-as-possible polyline (greedy nearest neighbour)."""
    pts = list(cells)
    if len(pts) <= 2:
        return pts
    arr = np.array(pts, float)
    used = np.zeros(len(pts), bool)
    start = int(np.lexsort((arr[:, 1], arr[:, 0]))[0])
    order = [start]
    used[start] = True
    for _ in range(len(pts) - 1):
        cur = arr[order[-1]]
        d = ((arr - cur) ** 2).sum(1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
    return [pts[i] for i in order]


def main():
    ext = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))
    R = ext['rect_aligned']
    tr = (R['xmin'], DX, 0.0, R['ymax'], 0.0, -DX)
    z = np.load(os.path.join(PTH.STATIC_INTERIM, 'merged60.npy'))
    ny, nx = z.shape

    def rc2xy(r, c):
        return R['xmin'] + (c + 0.5) * DX, R['ymax'] - (r + 0.5) * DX

    def xy2rc(x, y):
        return int((R['ymax'] - y) / DX), int((x - R['xmin']) / DX)

    rings = json.load(open(os.path.join(PTH.STATIC_GIS, 'huc10_rings_26917.json')))
    HU = {c: rasterize_rings(r, tr, z.shape) for c, r in rings.items()}
    U = np.zeros(z.shape, bool)
    for m in HU.values():
        U |= m
    print(f'HUC10 union {U.sum()} cells = {U.sum()*DX*DX/1e6:.1f} km2')

    # ---------------- open ocean ----------------
    deep_out = (~U) & (z <= SEAWARD_CONTOUR)
    dlab, dn = components(deep_out)
    east = set(int(v) for v in dlab[:, nx - 1] if v > 0)
    OCEAN = np.isin(dlab, sorted(east)) if east else np.zeros(z.shape, bool)
    print(f'Open ocean (outside HUC10, z<={SEAWARD_CONTOUR:+.1f} m, connected to the east edge of the rectangle)'
          f' {OCEAN.sum()} cells = {OCEAN.sum()*DX*DX/1e6:.1f} km2')

    # ---------------- seaward extension strip ----------------
    EXT = np.zeros(z.shape, bool)
    rows_ok, rows_capped = 0, 0
    for j in range(ny):
        cols = np.where(U[j])[0]
        if cols.size == 0:
            continue
        i0 = cols.max() + 1
        run = []
        i = i0
        while i < nx and not OCEAN[j, i] and len(run) < MAX_EXT_CELLS:
            run.append(i)
            i += 1
        if i < nx and OCEAN[j, i]:
            EXT[j, run] = True
            rows_ok += 1
        else:
            rows_capped += 1
    ACT = U | EXT
    wid = [int(EXT[j].sum()) for j in range(ny) if EXT[j].any()]
    print(f'Seaward extension {EXT.sum()} cells = {EXT.sum()*DX*DX/1e6:.1f} km2;'
          f' coastal rows {rows_ok}, non-coastal rows {rows_capped}')
    if wid:
        print(f'  width per row: median {int(np.median(wid))} cells '
              f'({np.median(wid)*DX/1000:.2f} km), max {max(wid)} cells '
              f'({max(wid)*DX/1000:.2f} km)')
    print(f'Active domain total {ACT.sum()} cells = {ACT.sum()*DX*DX/1e6:.1f} km2')
    print(f'Sea area deeper than {SEAWARD_CONTOUR:+.1f} m set inactive: {OCEAN.sum()} cells')

    # ---------------- seaward water-level boundary = outer edge of the extension strip (-2 m contour) ----------------
    sea_edge = np.zeros(z.shape, bool)
    for j in range(ny):
        cols = np.where(EXT[j])[0]
        if cols.size:
            sea_edge[j, cols.max()] = True
    print(f'\nSeaward boundary (-2 m contour) {sea_edge.sum()} cells = '
          f'{sea_edge.sum()*DX/1000:.1f} km')

    # ---------------- water-body cross-sections on the remaining domain boundary ----------------
    inner = np.zeros_like(ACT)
    inner[1:-1, 1:-1] = (ACT[1:-1, 1:-1] & ACT[:-2, 1:-1] & ACT[2:, 1:-1]
                         & ACT[1:-1, :-2] & ACT[1:-1, 2:])
    edge = ACT & ~inner & ~sea_edge
    wet_edge = edge & (z <= WET)
    lab, n = components(wet_edge)
    print(f'Remaining domain boundary {edge.sum()} cells, of which water {wet_edge.sum()} cells, grouped into {n} segments')

    # ---------------- along-channel distance ----------------
    wet = (z <= WET) & ACT
    ax, ay = [float(v) for v in geoio.ll_to_utm(ACOSTA['lon'], ACOSTA['lat'], ZONE)]
    ar, ac = xy2rc(ax, ay)
    # The Acosta gauge lies about 2.5 km **outside** the HUC10 union (on the upstream side of the junction of the two HUC10s),
    # so a small search radius cannot be used; instead snap to the nearest "water cell inside the active domain" over the whole map.
    jj, ii = np.where(wet)
    if jj.size == 0:
        raise SystemExit('!! no water cells in the active domain; terrain or active domain is wrong')
    d2 = (jj - ar) ** 2 + (ii - ac) ** 2
    kmin = int(np.argmin(d2))
    snap = (int(jj[kmin]), int(ii[kmin]))
    snap_m = float(np.sqrt(d2[kmin]) * DX)
    seeds = [snap]
    sx, sy = rc2xy(*snap)
    print(f'Acosta gauge ({ax:.0f}, {ay:.0f}) snapped to the nearest water cell in the active domain '
          f'({sx:.0f}, {sy:.0f}), straight-line distance {snap_m/1000:.2f} km')
    d_acosta = geodesic(wet, seeds)
    print(f'Water cells reachable from Acosta along the channel {int((d_acosta>=0).sum())} / {int(wet.sum())}')

    # ---------------- classification ----------------
    segs = []
    for k in range(1, n + 1):
        jj, ii = np.where(lab == k)
        cx, cy = rc2xy(jj.mean(), ii.mean())
        dd = d_acosta[jj, ii]
        dd = dd[dd >= 0]
        along = float(dd.min()) * DX / 1000.0 if dd.size else None
        depth = np.clip(WET - z[jj, ii], 0, None)
        segs.append(dict(k=int(k), n_cells=int(len(jj)),
                         width_m=float(len(jj) * DX),
                         cx=float(cx), cy=float(cy),
                         z_min=float(z[jj, ii].min()),
                         along_channel_km_to_acosta=along,
                         flow_area_m2=float(depth.sum() * DX),
                         south_of_acosta=bool(cy < ay),
                         rows=jj.tolist(), cols=ii.tolist()))
    segs.sort(key=lambda s: -s['n_cells'])
    for s in segs:
        up = (s['south_of_acosta']
              and s['along_channel_km_to_acosta'] is not None
              and s['along_channel_km_to_acosta'] <= 10.0
              and s['n_cells'] >= 2)
        s['bnd_type'] = 'upstream_discharge' if up else 'downstream_waterlevel'

    ups = [s for s in segs if s['bnd_type'] == 'upstream_discharge']
    downs = [s for s in segs if s['bnd_type'] == 'downstream_waterlevel']
    tot = sum(s['flow_area_m2'] for s in ups)
    for s in ups:
        s['Q_fraction'] = round(s['flow_area_m2'] / tot, 4) if tot else 0.0
    print(f'\nUpstream discharge boundary {len(ups)} segments | tributary downstream water-level boundary {len(downs)} segments'
          f' | seaward downstream water-level boundary 1 segment')
    print('\n--- Upstream discharge boundary (Q split by flow cross-section area) ---')
    for i, s in enumerate(ups, 1):
        print(f"  UP{i:02d}  ({s['cx']:.0f}, {s['cy']:.0f})  width {s['width_m']:6.0f} m"
              f"  min {s['z_min']:7.2f} m  section area {s['flow_area_m2']:9.0f} m2"
              f"  -> Q {s['Q_fraction']*100:5.1f}%"
              f"  along-channel to Acosta {s['along_channel_km_to_acosta']:.2f} km")
    print('\n--- Tributary downstream water-level boundary ---')
    for i, s in enumerate(downs, 1):
        a = s['along_channel_km_to_acosta']
        print(f"  DW{i:02d}  ({s['cx']:.0f}, {s['cy']:.0f})  width {s['width_m']:6.0f} m"
              f"  min {s['z_min']:7.2f} m"
              f"  to Acosta {('%.2f km' % a) if a is not None else 'not connected'}")

    # ---------------- write outputs ----------------
    pro = PTH.STATIC_TERRAIN
    gis = PTH.STATIC_GIS
    os.makedirs(pro, exist_ok=True)
    geoio.write_geotiff(os.path.join(pro, 'active_domain_mask_60m_26917.tif'),
                        np.where(ACT, 1, 0).astype(np.uint8), tr, 26917, nodata=0,
                        citation='1=active (HUC10 union + seaward strip to -2 m)')
    geoio.write_geotiff(os.path.join(pro, 'topobathy_merged_60m_26917_huc10.tif'),
                        np.where(U, z, NODATA).astype(np.float32), tr, 26917,
                        nodata=NODATA, citation='clipped to HUC10 union')
    geoio.write_geotiff(os.path.join(pro, 'topobathy_merged_60m_26917_active.tif'),
                        np.where(ACT, z, NODATA).astype(np.float32), tr, 26917,
                        nodata=NODATA, citation='clipped to active domain')
    np.save(os.path.join(PTH.STATIC_INTERIM, 'active_mask.npy'), ACT)
    np.save(os.path.join(PTH.STATIC_INTERIM, 'huc10_union.npy'), U)
    np.save(os.path.join(PTH.STATIC_INTERIM, 'ocean_mask.npy'), OCEAN)

    G = geoio.GeoJSONWriter(gis, 26917, ZONE)

    def line_from(cells):
        pts = [rc2xy(r, c) for r, c in order_path(cells)]
        if len(pts) == 1:
            pts = pts * 2
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return geoio._wkb_linestring(pts), (min(xs), min(ys), max(xs), max(ys))

    upf = []
    for i, s in enumerate(ups, 1):
        wkb, bb = line_from(list(zip(s['rows'], s['cols'])))
        upf.append((wkb, bb, dict(
            bnd_id=f'UP{i:02d}', bnd_type='upstream_discharge',
            station=ACOSTA['id'], station_name=ACOSTA['name'],
            forcing='USGS 02246500 parameter 00060 un-detided signed total discharge Q(t), m3/s',
            width_m=round(s['width_m'], 1), z_min_m=round(s['z_min'], 2),
            flow_area_m2=round(s['flow_area_m2'], 1),
            Q_fraction=s['Q_fraction'],
            along_channel_km_to_acosta=round(s['along_channel_km_to_acosta'], 2),
            note='Q split by share of flow cross-section area; full discharge not applied repeatedly',
            status='candidate - editable in QGIS')))
    G.add_layer('bnd_upstream_discharge', 'LINESTRING', upf,
                [('bnd_id', 'TEXT'), ('bnd_type', 'TEXT'), ('station', 'TEXT'),
                 ('station_name', 'TEXT'), ('forcing', 'TEXT'),
                 ('width_m', 'REAL'), ('z_min_m', 'REAL'),
                 ('flow_area_m2', 'REAL'), ('Q_fraction', 'REAL'),
                 ('along_channel_km_to_acosta', 'REAL'), ('note', 'TEXT'),
                 ('status', 'TEXT')], 'Upstream discharge boundary')

    dwf = []
    sj, si = np.where(sea_edge)
    wkb, bb = line_from(list(zip(sj.tolist(), si.tolist())))
    dwf.append((wkb, bb, dict(
        bnd_id='DW00', bnd_type='downstream_waterlevel',
        station=MAYPORT['id'], station_name=MAYPORT['name'],
        forcing='NOAA 8720218 observed total water level h(t), m NAVD88',
        kind=f'seaward {SEAWARD_CONTOUR:+.1f} m depth contour (Leijnse et al. 2021 §2.3/§4.1)',
        width_m=round(float(sea_edge.sum() * DX), 1),
        z_min_m=round(float(z[sj, si].min()), 2),
        along_channel_km_to_acosta=None,
        note='SFINCS interpolates each boundary cell water level weighted by the two nearest boundary points',
        status='candidate - editable in QGIS')))
    for i, s in enumerate(downs, 1):
        wkb, bb = line_from(list(zip(s['rows'], s['cols'])))
        a = s['along_channel_km_to_acosta']
        dwf.append((wkb, bb, dict(
            bnd_id=f'DW{i:02d}', bnd_type='downstream_waterlevel',
            station=MAYPORT['id'], station_name=MAYPORT['name'],
            forcing='NOAA 8720218 observed total water level h(t), m NAVD88',
            kind='tributary cut',
            width_m=round(s['width_m'], 1), z_min_m=round(s['z_min'], 2),
            along_channel_km_to_acosta=(round(a, 2) if a is not None else None),
            note='SFINCS interpolates each boundary cell water level weighted by the two nearest boundary points',
            status='candidate - editable in QGIS')))
    G.add_layer('bnd_downstream_waterlevel', 'LINESTRING', dwf,
                [('bnd_id', 'TEXT'), ('bnd_type', 'TEXT'), ('station', 'TEXT'),
                 ('station_name', 'TEXT'), ('forcing', 'TEXT'), ('kind', 'TEXT'),
                 ('width_m', 'REAL'), ('z_min_m', 'REAL'),
                 ('along_channel_km_to_acosta', 'REAL'), ('note', 'TEXT'),
                 ('status', 'TEXT')], 'Downstream water-level boundary')

    sf = []
    for s in (ACOSTA, MAYPORT):
        x, y = geoio.ll_to_utm(s['lon'], s['lat'], ZONE)
        x, y = float(x), float(y)
        sf.append((geoio._wkb_point(x, y), (x, y, x, y),
                   dict(station_id=s['id'], agency=s['agency'], name=s['name'],
                        role=s['role'], parameter=s['param'],
                        lon=s['lon'], lat=s['lat'])))
    G.add_layer('stations_forcing', 'POINT', sf,
                [('station_id', 'TEXT'), ('agency', 'TEXT'), ('name', 'TEXT'),
                 ('role', 'TEXT'), ('parameter', 'TEXT'),
                 ('lon', 'REAL'), ('lat', 'REAL')], 'The two forcing stations actually used in this round')
    G.close()

    for old in ('candidate_boundary_sections.geojson', 'review_locations.geojson',
                'stations_observed.geojson'):
        p = os.path.join(gis, old)
        if os.path.exists(p):
            os.remove(p)
            print(f'  deleted obsolete layer {old}')

    meta = dict(
        generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        crs='EPSG:26917', cellsize_m=DX,
        reference='Leijnse et al. (2021) Coastal Engineering 163:103796 §2.3, §4.1',
        active_domain=dict(
            definition='HUC10 union + seaward extension to the -2 m depth contour',
            seaward_contour_m=SEAWARD_CONTOUR,
            datum_note='The paper uses m+MSL; local MSL is 0.1585 m below NAVD88, '
                       'so the paper\'s MSL-2 m is about NAVD88 -2.16 m; -2.0 m used per user instruction',
            huc10_union_km2=round(U.sum() * DX * DX / 1e6, 2),
            seaward_extension_km2=round(EXT.sum() * DX * DX / 1e6, 2),
            active_km2=round(ACT.sum() * DX * DX / 1e6, 2),
            ocean_inactive_km2=round(OCEAN.sum() * DX * DX / 1e6, 2),
            ext_row_width_cells=dict(median=int(np.median(wid)) if wid else 0,
                                     max=int(max(wid)) if wid else 0),
            rows_coastal=rows_ok, rows_not_coastal=rows_capped),
        boundary_rule=dict(
            upstream='water-body cross-sections upstream (south) of Acosta, <=10 km along the channel',
            downstream='seaward -2 m depth contour + all other water-body cross-sections',
            wall='remaining domain boundaries default to wall, no separate file',
            Q_split='multiple upstream cross-sections split Q by share of flow cross-section area'),
        acosta_snap=dict(station_xy=[round(ax, 1), round(ay, 1)],
                         snapped_xy=[round(sx, 1), round(sy, 1)],
                         snap_distance_m=round(snap_m, 1),
                         note='The Acosta gauge is outside the HUC10 union; along-channel distance measured from the snapped point'),
        sea_boundary=dict(cells=int(sea_edge.sum()),
                          length_km=round(sea_edge.sum() * DX / 1000, 2)),
        upstream_segments=[{k: v for k, v in s.items() if k not in ('rows', 'cols')}
                           for s in ups],
        downstream_tributary_segments=[
            {k: v for k, v in s.items() if k not in ('rows', 'cols')} for s in downs],
        stations_used=[ACOSTA, MAYPORT],
        stations_not_used=OTHER_STATIONS,
        disclaimer='All boundaries are candidates; they must not be used as final model boundaries until confirmed by the user in QGIS')
    with open(os.path.join(PTH.STATIC_META, 'active_domain_boundaries.json'), 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print('\nWrote data/meta/active_domain_boundaries.json')


if __name__ == '__main__':
    main()
