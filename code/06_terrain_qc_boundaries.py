#!/usr/bin/env python3
"""
06_terrain_qc_boundaries.py -- terrain QC + candidate boundary identification + GeoPackage output.

This stage only produces **candidate lines** and **problem locations**; it does not produce the final model mask and does not modify the terrain.
Everything that needs human judgement is output as a to-be-confirmed feature; nothing is decided automatically.

Checks (items 9 and 10 of section 4 of the task spec):
  1. Land-sea seam: elevation jumps where the source switches between DEM and CRM
  2. Anomalous lows: isolated pits much lower than the minimum of their 8 neighbours
  3. Channel continuity: whether the sea-connected flood area reaches the Acosta section along the main river
  4. Bridge/road blockage: places where high ground cuts the connected channel corridor

Candidate boundary identification:
  Cut sections are located where "water connected to the open sea" crosses the rectangle border, not by straight-line distance.
  For each cut, the nearest available water-level station is found by **geodesic distance along water** (not straight-line distance).
"""
import os, sys, json, struct, datetime, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

NEW = PTH.ROOT
OLD = PTH.OLD
ZONE = 17
DX = 60.0

# Candidate water-level stations (all from official downloads already saved by the old project; coordinates in data/meta/stations.json)
WL_STATIONS = [
    dict(id='8720218', agency='NOAA', name='Mayport (Bar Pilots Dock)',
         lon=-81.42789, lat=30.398167, param='water_level(NAVD88)',
         available='2017-09-03..09-20 6-min complete'),
    dict(id='8720219', agency='NOAA', name='Dames Point',
         lon=-81.5583, lat=30.3867, param='water_level(NAVD88)',
         available='2017-09-03..09-20 6-min complete (incl. 47 official inferred values during the event)'),
    dict(id='8720226', agency='NOAA', name='Southbank Riverwalk, St Johns River',
         lon=-81.6583, lat=30.32, param='water_level(NAVD88)',
         available='2017-09-03..09-20 6-min complete (incl. 8 official inferred values during the event)'),
    dict(id='02246515', agency='USGS', name='POTTSBURG CREEK NR S JACKSONVILLE',
         lon=-81.5900917, lat=30.2644071, param='00065 gage height',
         available='downloaded by old project 00065/00060/63160'),
    dict(id='02246621', agency='USGS', name='TROUT R NR JACKSONVILLE',
         lon=-81.69648758, lat=30.41746058, param='00065 gage height',
         available='downloaded by old project 00065/00060/72137'),
    dict(id='02246751', agency='USGS', name='BROWARD RIVER BL BISCAYNE BLVD',
         lon=-81.6682, lat=30.44333889, param='00065 gage height',
         available='downloaded by old project 00065/00060/72137'),
    dict(id='02246804', agency='USGS', name='DUNN CREEK AT DUNN CREEK RD',
         lon=-81.59688889, lat=30.45494167, param='00065 gage height',
         available='downloaded by old project 00065/00060/72137'),
    dict(id='02246825', agency='USGS', name='CLAPBOARD CREEK NR JACKSONVILLE',
         lon=-81.5182611, lat=30.44838056, param='00065 gage height',
         available='downloaded by old project 00065/00060/72137'),
]

ACOSTA = dict(id='02246500', agency='USGS',
              name='ST. JOHNS RIVER AT JACKSONVILLE (Acosta Bridge)',
              lon=-81.6653735, lat=30.3224616)


# ------------------------------------------------------------------ utilities

def rasterize_rings(rings, tr, shape):
    """Rasterize polygon rings to a boolean mask by scanline (cell-centre test)."""
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


def label_components(mask):
    """8-connected component labelling (BFS, pure numpy/collections)."""
    ny, nx = mask.shape
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for sj in range(ny):
        for si in range(nx):
            if not mask[sj, si] or lab[sj, si]:
                continue
            cur += 1
            q = collections.deque([(sj, si)])
            lab[sj, si] = cur
            while q:
                j, i = q.popleft()
                for dj, di in nbrs:
                    a, b = j + dj, i + di
                    if 0 <= a < ny and 0 <= b < nx and mask[a, b] and not lab[a, b]:
                        lab[a, b] = cur
                        q.append((a, b))
    return lab, cur


def geodesic_cells(mask, seeds):
    """Multi-source BFS inside mask; returns each cell's **number of cells along water** to the nearest seed."""
    ny, nx = mask.shape
    dist = np.full(mask.shape, -1, np.int32)
    q = collections.deque()
    for j, i in seeds:
        if 0 <= j < ny and 0 <= i < nx and mask[j, i]:
            dist[j, i] = 0
            q.append((j, i))
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        j, i = q.popleft()
        d = dist[j, i] + 1
        for dj, di in nbrs:
            a, b = j + dj, i + di
            if 0 <= a < ny and 0 <= b < nx and mask[a, b] and dist[a, b] < 0:
                dist[a, b] = d
                q.append((a, b))
    return dist


def rc_to_xy(tr, r, c):
    x0, dx, _, y0, _, dy = tr
    return x0 + (c + 0.5) * dx, y0 + (r + 0.5) * dy


def xy_to_rc(tr, x, y):
    x0, dx, _, y0, _, dy = tr
    return int((y - y0) / dy), int((x - x0) / dx)


# ------------------------------------------------------------------ main

def main():
    ext = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))
    R = ext['rect_aligned']
    tr = (R['xmin'], DX, 0.0, R['ymax'], 0.0, -DX)
    z = np.load(os.path.join(PTH.STATIC_INTERIM, 'merged60.npy'))
    flag, _ = geoio.tif_read(os.path.join(
        NEW, 'data/interim/source_flag_60m_26917.tif'))
    ny, nx = z.shape
    print(f'terrain {nx} x {ny} @ {DX} m')

    rings = json.load(open(os.path.join(PTH.STATIC_GIS, 'huc10_rings_26917.json')))
    hu_mask = {}
    for code, rgs in rings.items():
        hu_mask[code] = rasterize_rings(rgs, tr, z.shape)
        print(f'  HUC10 {code} rasterized cells {hu_mask[code].sum()}'
              f'  = {hu_mask[code].sum()*DX*DX/1e6:.1f} km2'
              f'  (WBD attribute {ext["huc10"][code]["areasqkm"]} km2)')
    hu_any = np.zeros(z.shape, bool)
    for m in hu_mask.values():
        hu_any |= m

    qc = {}

    # ---- check 1: land-sea seam
    print('\n[check 1] land-sea seam / elevation jumps at data-source switches')
    src_dem = (flag == 1) | (flag == 3)
    src_crm = (flag == 2) | (flag == 4)
    edge = np.zeros(z.shape, bool)
    edge[:, :-1] |= src_dem[:, :-1] & src_crm[:, 1:]
    edge[:, 1:] |= src_dem[:, 1:] & src_crm[:, :-1]
    edge[:-1] |= src_dem[:-1] & src_crm[1:]
    edge[1:] |= src_dem[1:] & src_crm[:-1]
    gx = np.zeros(z.shape); gy = np.zeros(z.shape)
    gx[:, 1:-1] = np.abs(z[:, 2:] - z[:, :-2]) / 2
    gy[1:-1] = np.abs(z[2:] - z[:-2]) / 2
    grad = np.hypot(gx, gy)
    seam = edge & (grad > 3.0)
    qc['seam'] = dict(
        n_source_switch_cells=int(edge.sum()),
        n_seam_jump_gt_3m=int(seam.sum()),
        seam_grad_p99=float(np.percentile(grad[edge], 99)) if edge.any() else None,
        interior_grad_p99=float(np.percentile(grad[~edge], 99)))
    print(f'  data-source switch edge cells {edge.sum()}, of which elevation jump >3 m: {seam.sum()}')

    # ---- check 2: anomalous lows (isolated pits)
    print('[check 2] anomalous lows (lower than the 8-neighbour minimum)')
    nb_min = np.full(z.shape, np.inf)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj == 0 and di == 0:
                continue
            s = np.roll(np.roll(z, dj, 0), di, 1)
            nb_min = np.minimum(nb_min, s)
    pit_depth = nb_min - z
    pit_depth[0] = pit_depth[-1] = 0
    pit_depth[:, 0] = pit_depth[:, -1] = 0
    pits = pit_depth > 2.0
    qc['pits'] = dict(n_pits_gt_2m=int(pits.sum()),
                      max_pit_depth_m=float(pit_depth.max()),
                      n_pits_in_huc10=int((pits & hu_any).sum()))
    print(f'  isolated pits (depth >2 m): {pits.sum()}, deepest {pit_depth.max():.2f} m')

    # ---- check 3: channel connectivity
    print('[check 3] channel connectivity (low ground connected to the open sea)')
    results_conn = {}
    for thr in (0.0, -0.5, 0.5, 1.0):
        wet = z <= thr
        seeds = [(j, nx - 1) for j in range(ny) if wet[j, nx - 1]]
        seeds += [(0, i) for i in range(nx) if wet[0, i]]
        d = geodesic_cells(wet, seeds)
        conn = d >= 0
        ar, ac = xy_to_rc(tr, *geoio.ll_to_utm(ACOSTA['lon'], ACOSTA['lat'], ZONE))
        ar, ac = int(ar), int(ac)
        win = conn[max(0, ar - 3):ar + 4, max(0, ac - 3):ac + 4]
        results_conn[thr] = dict(
            n_wet=int(wet.sum()), n_connected=int(conn.sum()),
            acosta_reachable=bool(win.any()),
            acosta_along_channel_cells=int(
                d[max(0, ar - 3):ar + 4, max(0, ac - 3):ac + 4].max())
            if win.any() else None)
        print(f'  threshold {thr:+.1f} m: wet cells {wet.sum():7d} connected {conn.sum():7d}'
              f'  Acosta reachable={results_conn[thr]["acosta_reachable"]}'
              f'  cells along channel={results_conn[thr]["acosta_along_channel_cells"]}')
    qc['channel_connectivity'] = {str(k): v for k, v in results_conn.items()}

    # use the 0 m threshold as the working water area
    WET_THR = 0.0
    wet = z <= WET_THR
    seeds = [(j, nx - 1) for j in range(ny) if wet[j, nx - 1]]
    seeds += [(0, i) for i in range(nx) if wet[0, i]]
    dsea = geodesic_cells(wet, seeds)
    conn = dsea >= 0

    # ---- check 4: high-ground blockages inside the connected corridor (suspected bridges/roads)
    print('[check 4] suspected blockages inside connected water (bridges/embankments)')
    block = np.zeros(z.shape, bool)
    for dj, di in ((0, 1), (1, 0)):
        a = np.roll(conn, dj, 0) if dj else np.roll(conn, di, 1)
        b = np.roll(conn, -dj, 0) if dj else np.roll(conn, -di, 1)
        block |= (~conn) & a & b & (z > WET_THR) & (z < 12.0)
    block[0] = block[-1] = False
    block[:, 0] = block[:, -1] = False
    blab, nblk = label_components(block)
    blocks = []
    for k in range(1, nblk + 1):
        jj, ii = np.where(blab == k)
        if len(jj) < 2:
            continue
        x, y = rc_to_xy(tr, jj.mean(), ii.mean())
        blocks.append(dict(id=f'BLK{k:03d}', n_cells=int(len(jj)),
                           x=float(x), y=float(y),
                           z_min=float(z[jj, ii].min()),
                           z_max=float(z[jj, ii].max()),
                           in_huc10=bool(hu_any[jj, ii].any())))
    blocks.sort(key=lambda b: -b['n_cells'])
    qc['blockages'] = dict(n_candidates=len(blocks), top=blocks[:25])
    print(f'  suspected blockage clusters: {len(blocks)} (>=2 cells), largest {blocks[0]["n_cells"] if blocks else 0} cells')

    # ---- candidate boundaries: where sea-connected water crosses the rectangle border
    print('\n[candidate boundaries] where connected water crosses the rectangle border')
    border = np.zeros(z.shape, bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    cut = conn & border
    clab, ncut = label_components(cut)
    cuts = []
    for k in range(1, ncut + 1):
        jj, ii = np.where(clab == k)
        side = []
        if (jj == 0).any():
            side.append('N')
        if (jj == ny - 1).any():
            side.append('S')
        if (ii == 0).any():
            side.append('W')
        if (ii == nx - 1).any():
            side.append('E')
        x0_, y0_ = rc_to_xy(tr, jj.min(), ii.min())
        x1_, y1_ = rc_to_xy(tr, jj.max(), ii.max())
        cx, cy = rc_to_xy(tr, jj.mean(), ii.mean())
        width_m = max(abs(x1_ - x0_), abs(y1_ - y0_)) + DX
        cuts.append(dict(k=int(k), side='/'.join(side), n_cells=int(len(jj)),
                         width_m=float(width_m),
                         x0=float(x0_), y0=float(y0_), x1=float(x1_), y1=float(y1_),
                         cx=float(cx), cy=float(cy),
                         z_min=float(z[jj, ii].min()),
                         rows=jj.tolist(), cols=ii.tolist(),
                         dist_to_sea_cells=int(dsea[jj, ii].min())))
    cuts.sort(key=lambda c: -c['n_cells'])
    print(f'  {len(cuts)} in total; top 12 by size:')
    for c in cuts[:12]:
        print(f"    {c['side']} side  width {c['width_m']:6.0f} m  {c['n_cells']:3d} cells"
              f"  centre ({c['cx']:.0f},{c['cy']:.0f})  min {c['z_min']:6.2f} m"
              f"  to sea along channel {c['dist_to_sea_cells']} cells")

    # ---- geodesic distance along water -> station assignment suggestions
    print('\n[station assignment] geodesic distance along water (not straight-line)')
    station_dist = {}
    for s in WL_STATIONS:
        sx, sy = geoio.ll_to_utm(s['lon'], s['lat'], ZONE)
        sr, sc = xy_to_rc(tr, float(sx), float(sy))
        s['x'], s['y'] = float(sx), float(sy)
        # stations often sit on land; use the nearest connected water cell as the entry point
        best = None
        for rad in range(0, 12):
            j0, j1 = max(0, sr - rad), min(ny, sr + rad + 1)
            i0, i1 = max(0, sc - rad), min(nx, sc + rad + 1)
            sub = conn[j0:j1, i0:i1]
            if sub.any():
                jj, ii = np.where(sub)
                d2 = (jj + j0 - sr) ** 2 + (ii + i0 - sc) ** 2
                m = np.argmin(d2)
                best = (int(jj[m] + j0), int(ii[m] + i0), float(np.sqrt(d2[m]) * DX))
                break
        if best is None:
            s['snap'] = None
            continue
        s['snap'] = dict(row=best[0], col=best[1], snap_dist_m=round(best[2], 1))
        station_dist[s['id']] = geodesic_cells(conn, [(best[0], best[1])])

    for c in cuts:
        rows, cols = np.array(c['rows']), np.array(c['cols'])
        cand = []
        for sid, dmap in station_dist.items():
            d = dmap[rows, cols]
            d = d[d >= 0]
            if d.size:
                cand.append((sid, int(d.min()) * DX / 1000.0))
        cand.sort(key=lambda t: t[1])
        c['station_ranking_along_channel_km'] = [
            dict(station=s, along_channel_km=round(v, 2)) for s, v in cand]
        # straight-line ranking, for comparison (shows why straight-line distance alone is not enough)
        st = []
        for s in WL_STATIONS:
            st.append((s['id'], np.hypot(c['cx'] - s['x'], c['cy'] - s['y']) / 1000))
        st.sort(key=lambda t: t[1])
        c['station_ranking_straight_km'] = [
            dict(station=s, straight_km=round(v, 2)) for s, v in st[:4]]
        c['nearest_along_channel'] = cand[0][0] if cand else None
        c['nearest_straight'] = st[0][0]
        c['ranking_disagrees'] = (c['nearest_along_channel'] != c['nearest_straight'])

    dis = [c for c in cuts if c.get('ranking_disagrees')]
    print(f'  {len(dis)}/{len(cuts)}  cuts where the "nearest station along water" differs from the "nearest station in straight line"')

    # ---- classify candidate boundaries
    acx, acy = geoio.ll_to_utm(ACOSTA['lon'], ACOSTA['lat'], ZONE)
    for c in cuts:
        if 'E' in c['side'] and c['width_m'] > 500:
            c['bnd_type'] = 'sea_waterlevel'
            c['proposed_station'] = '8720218'
            c['reason'] = 'east side towards the open sea, wide inlet; proposed: Mayport observed total water level'
        elif 'S' in c['side'] and c['n_cells'] >= 8:
            c['bnd_type'] = 'upstream_discharge'
            c['proposed_station'] = '02246500'
            c['reason'] = ('south-side main-river section, same St. Johns River main channel as Acosta; '
                           'proposed: apply Acosta Q(t) approximately')
        else:
            c['bnd_type'] = 'tidal_tributary_waterlevel'
            c['proposed_station'] = c.get('nearest_along_channel')
            c['reason'] = ('truncated tidal tributary; candidate stations ranked by geodesic distance along water; '
                           'select after the connectivity is confirmed manually')
        c['status'] = 'candidate - to be confirmed'

    up = [c for c in cuts if c['bnd_type'] == 'upstream_discharge']
    if len(up) > 1:
        for c in up:
            c['warning'] = (f'{len(up)} main-river/tributary sections identified on the south side. '
                            'The same full Acosta discharge must not be copied to several cuts; '
                            'a single upstream discharge section must be chosen manually.')
    print(f'\n  classes: sea side {sum(1 for c in cuts if c["bnd_type"]=="sea_waterlevel")}'
          f' | upstream discharge candidates {len(up)}'
          f' | tidal tributaries {sum(1 for c in cuts if c["bnd_type"]=="tidal_tributary_waterlevel")}')

    # ---- land-side edges suspected not to carry water (flag only, no wall)
    dry_border = border & ~conn
    qc['dry_border_cells'] = int(dry_border.sum())
    qc['wet_border_cells'] = int(cut.sum())
    print(f'  rectangle border: connected water cells {cut.sum()}, dry land cells {dry_border.sum()}'
          f' (flag only, no automatic wall)')

    # ---- write GeoPackage
    gjdir = PTH.STATIC_GIS
    G = geoio.GeoJSONWriter(gjdir, 26917, ZONE)

    feats = []
    for code, rgs in rings.items():
        wkb = geoio._wkb_polygon([[tuple(p) for p in r] for r in rgs])
        a = np.vstack([np.array(r) for r in rgs])
        h = ext['huc10'][code]
        feats.append((wkb, (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()),
                      dict(huc10=code, name=h['name'], areasqkm=h['areasqkm'],
                           states=h['states'], hutype=h['hutype'],
                           loaddate=str(h['loaddate']),
                           source='USGS WBD WBD_03_HU2_Shape/Shape/WBDHU10.shp',
                           note='study core area, not the model active domain')))
    G.add_layer('huc10', 'POLYGON', feats,
                [('huc10', 'TEXT'), ('name', 'TEXT'), ('areasqkm', 'REAL'),
                 ('states', 'TEXT'), ('hutype', 'TEXT'), ('loaddate', 'TEXT'),
                 ('source', 'TEXT'), ('note', 'TEXT')],
                'two complete HUC10')

    # study rectangle
    rect = [[(R['xmin'], R['ymin']), (R['xmax'], R['ymin']),
             (R['xmax'], R['ymax']), (R['xmin'], R['ymax']),
             (R['xmin'], R['ymin'])]]
    G.add_layer('study_rectangle', 'POLYGON',
                [(geoio._wkb_polygon(rect),
                  (R['xmin'], R['ymin'], R['xmax'], R['ymax']),
                  dict(name='rectangular terrain sheet', width_m=R['xmax'] - R['xmin'],
                       height_m=R['ymax'] - R['ymin'],
                       note='sheet extent, not the model active domain; outer terrain kept, not clipped'))],
                [('name', 'TEXT'), ('width_m', 'REAL'), ('height_m', 'REAL'),
                 ('note', 'TEXT')], 'study rectangle sheet')

    # observation stations
    sfeats = []
    allst = [dict(ACOSTA, role='upstream discharge reference station (observed)',
                  param='00060 signed total discharge, tide not removed')] + [
        dict(s, role=('sea-side water-level reference station (observed)' if s['id'] == '8720218'
                      else 'candidate water-level station (observed)')) for s in WL_STATIONS]
    for s in allst:
        x, y = geoio.ll_to_utm(s['lon'], s['lat'], ZONE)
        x, y = float(x), float(y)
        sfeats.append((geoio._wkb_point(x, y), (x, y, x, y),
                       dict(station_id=s['id'], agency=s['agency'],
                            name=s['name'], role=s['role'],
                            parameter=s.get('param', ''),
                            availability=s.get('available', ''),
                            lon=s['lon'], lat=s['lat'],
                            kind='observation station location')))
    G.add_layer('stations_observed', 'POINT', sfeats,
                [('station_id', 'TEXT'), ('agency', 'TEXT'), ('name', 'TEXT'),
                 ('role', 'TEXT'), ('parameter', 'TEXT'),
                 ('availability', 'TEXT'), ('lon', 'REAL'), ('lat', 'REAL'),
                 ('kind', 'TEXT')],
                'actual observation station locations (distinct from candidate input sections)')

    # candidate boundary lines
    lfeats = []
    for n, c in enumerate(cuts, 1):
        rows, cols = np.array(c['rows']), np.array(c['cols'])
        pts = [rc_to_xy(tr, r, cc) for r, cc in
               sorted(zip(rows.tolist(), cols.tolist()))]
        if len(pts) == 1:
            pts = pts * 2
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rank = '; '.join(f"{d['station']}={d['along_channel_km']}km"
                         for d in c['station_ranking_along_channel_km'][:4])
        rank_s = '; '.join(f"{d['station']}={d['straight_km']}km"
                           for d in c['station_ranking_straight_km'][:4])
        lfeats.append((geoio._wkb_linestring(pts),
                       (min(xs), min(ys), max(xs), max(ys)),
                       dict(bnd_id=f'CAND{n:03d}', bnd_type=c['bnd_type'],
                            side=c['side'], width_m=round(c['width_m'], 1),
                            n_cells=c['n_cells'], z_min_m=round(c['z_min'], 2),
                            proposed_station=c.get('proposed_station') or '',
                            reason=c['reason'],
                            rank_along_channel=rank,
                            rank_straight_line=rank_s,
                            ranking_disagrees=int(bool(c.get('ranking_disagrees'))),
                            open_question=c.get('warning',
                                                'connectivity and section position need confirmation'),
                            status=c['status'],
                            is_candidate=1)))
    G.add_layer('candidate_boundary_sections', 'LINESTRING', lfeats,
                [('bnd_id', 'TEXT'), ('bnd_type', 'TEXT'), ('side', 'TEXT'),
                 ('width_m', 'REAL'), ('n_cells', 'INTEGER'),
                 ('z_min_m', 'REAL'), ('proposed_station', 'TEXT'),
                 ('reason', 'TEXT'), ('rank_along_channel', 'TEXT'),
                 ('rank_straight_line', 'TEXT'),
                 ('ranking_disagrees', 'INTEGER'), ('open_question', 'TEXT'),
                 ('status', 'TEXT'), ('is_candidate', 'INTEGER')],
                'candidate boundary cuts (all candidates; must not be used as final model boundaries without confirmation)')

    # locations to check
    qfeats = []
    for b in blocks[:60]:
        qfeats.append((geoio._wkb_point(b['x'], b['y']),
                       (b['x'], b['y'], b['x'], b['y']),
                       dict(qid=b['id'], issue='suspected bridge/embankment blocking connected water',
                            n_cells=b['n_cells'], z_min_m=round(b['z_min'], 2),
                            z_max_m=round(b['z_max'], 2),
                            in_huc10=int(b['in_huc10']),
                            action='manual decision needed: lower deck / carve channel? terrain not modified in this stage',
                            status='to be confirmed')))
    pj, pi = np.where(pits)
    order = np.argsort(-pit_depth[pj, pi])[:40]
    for k in order:
        x, y = rc_to_xy(tr, pj[k], pi[k])
        qfeats.append((geoio._wkb_point(x, y), (x, y, x, y),
                       dict(qid=f'PIT{k:03d}', issue='anomalous isolated low (pit)',
                            n_cells=1, z_min_m=round(float(z[pj[k], pi[k]]), 2),
                            z_max_m=round(float(pit_depth[pj[k], pi[k]]), 2),
                            in_huc10=int(hu_any[pj[k], pi[k]]),
                            action='manual decision needed: fill the pit? terrain not modified in this stage',
                            status='to be confirmed')))
    sj, si = np.where(seam)
    _ord = np.argsort(-grad[sj, si])[:40]
    sj, si = sj[_ord], si[_ord]
    for k in range(len(sj)):
        x, y = rc_to_xy(tr, sj[k], si[k])
        qfeats.append((geoio._wkb_point(x, y), (x, y, x, y),
                       dict(qid=f'SEAM{k:03d}', issue='elevation jump >3 m at DEM/CRM seam',
                            n_cells=1, z_min_m=round(float(z[sj[k], si[k]]), 2),
                            z_max_m=round(float(grad[sj[k], si[k]]), 2),
                            in_huc10=int(hu_any[sj[k], si[k]]),
                            action='manual check needed: is this real terrain? terrain not modified in this stage',
                            status='to be confirmed')))
    G.add_layer('review_locations', 'POINT', qfeats,
                [('qid', 'TEXT'), ('issue', 'TEXT'), ('n_cells', 'INTEGER'),
                 ('z_min_m', 'REAL'), ('z_max_m', 'REAL'),
                 ('in_huc10', 'INTEGER'), ('action', 'TEXT'),
                 ('status', 'TEXT')],
                'boundary and terrain locations needing further checks')
    G.close()
    print(f'\nGeoJSON written to {gjdir}')

    # connectivity / water raster for mapping and QGIS checks
    geoio.write_geotiff(
        os.path.join(PTH.STATIC_INTERIM, 'connected_water_60m_26917.tif'),
        np.where(conn, 1, np.where(wet, 2, 0)).astype(np.uint8), tr, 26917,
        nodata=0, citation='1=water connected to open sea (z<=0m) 2=isolated low 0=other')

    qc['cuts'] = [{k: v for k, v in c.items() if k not in ('rows', 'cols')}
                  for c in cuts]
    qc['generated_utc'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    qc['wet_threshold_m'] = WET_THR
    qc['huc10_raster_km2'] = {c: float(m.sum() * DX * DX / 1e6)
                              for c, m in hu_mask.items()}
    qc['disclaimer'] = ('This file only records candidates and items to check. Without user confirmation, '
                        'no candidate line may be used as a final model boundary, and the terrain must not be modified on this basis.')
    with open(os.path.join(PTH.STATIC_META, 'terrain_qc_boundaries.json'), 'w') as f:
        json.dump(qc, f, indent=1, ensure_ascii=False)
    print('written data/meta/terrain_qc_boundaries.json')


if __name__ == '__main__':
    main()
