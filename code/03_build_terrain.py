#!/usr/bin/env python3
"""
03_build_terrain.py -- build the 60 m merged DEM/CRM terrain + datum diagnostics.

Method follows Sebastian et al. (2021) §3.2.1: the USGS DEM and NOAA CRM are merged on a common grid
by taking the lower elevation where they overlap. As required by this project, the USGS DEM is ~30 m (1 arc-second),
not the 10 m used in the paper.

Resampling follows the bilinear method of the old report, in two stages (same as the old workflow):
    source (geographic) --bilinear--> 30 m EPSG:26917 reference grid --bilinear--> 60 m delivery grid
60 m cell centers fall exactly midway between 30 m cell centers, so the second bilinear stage equals a 2x2 average, which avoids
aliasing from direct point sampling.

Important: this script **performs no vertical datum conversion** and adds or subtracts no constant.
The vertical datum of the CRM is recorded inconsistently in the old project (see the conflict table in result/reports),
and the file itself has no VerticalCSType GeoKey. The script only outputs quantitative diagnostics (onshore overlap
DEM-CRM difference statistics and distribution plot) for manual judgment.
"""
import os, sys, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np

OLD = PTH.OLD
NEW = PTH.ROOT
ZONE = 17

DEM_SRC = os.path.join(OLD, 'data/dem_crm_hu8/merged_jacksonville_dem_1arc.tif')
DEM_TILES = [
    'data/dem_crm_hu8/USGS_1_n30w082_20251208.tif',
    'data/dem_crm_hu8/USGS_1_n30w083_20221103.tif',
    'data/dem_crm_hu8/USGS_1_n31w082_20220725.tif',
    'data/dem_crm_hu8/USGS_1_n31w083_20220725.tif',
]
CRM_SRC = os.path.join(OLD, 'data/dem_crm_hu8/crm_3_arc.tiff')
CRM_ALT = os.path.join(OLD, 'data/crm_1arc_90m.tiff')

NODATA = -9999.0

# Merge threshold (confirmed by the user 2026-09-14 = 0.5 m).
# Why not 0: over water USGS 3DEP gives a **hydro-flattened water-surface elevation**;
# this reach of the St. Johns River at Jacksonville is flattened to +0.10 m NAVD88. With DEM<=0,
# 0.10 > 0 would be classed as land and the 15 m deep main channel filled as +0.10 m flat ground.
# Measured: among cells CRM considers deeper than 3 m, DEM<=0 captures only 78.4%, DEM<=0.5 captures 97.9%;
# both have +0.0000 m effect on land (DEM>1 m), and 0 land cells are lowered to <=0 in either case.
LAND_THRESHOLD = 0.5


# ------------------------------------------------------------------ sampling

def load_window_ll(path, lon0, lon1, lat0, lat1, pad=4):
    """Read a geographic raster within a lon/lat window; returns (arr (float64, NaN = invalid),
    lons, lats) -- lons/lats are cell-center coordinates, lats decrease from north to south."""
    m = geoio.tif_meta(path)
    x0, dx, _, y0, _, dy = m['transform']
    W, H = m['width'], m['height']
    c0 = int(np.floor((lon0 - x0) / dx)) - pad
    c1 = int(np.ceil((lon1 - x0) / dx)) + pad
    r0 = int(np.floor((lat1 - y0) / dy)) - pad
    r1 = int(np.ceil((lat0 - y0) / dy)) + pad
    c0, c1 = max(0, c0), min(W, c1)
    r0, r1 = max(0, r0), min(H, r1)
    if c1 <= c0 or r1 <= r0:
        return None, None, None, m
    with geoio.TiffFile(path) as t:
        a = t.read(window=(r0, r1, c0, c1)).astype(np.float64)
    nd = m.get('nodata')
    if nd is not None and not isinstance(nd, str):
        a[a == nd] = np.nan
    a[a < -1e5] = np.nan          # -999999 / -3.4e38 and the like
    a[a > 1e5] = np.nan
    lons = x0 + (np.arange(c0, c1) + 0.5) * dx
    lats = y0 + (np.arange(r0, r1) + 0.5) * dy
    return a, lons, lats, m


def bilinear(arr, lons, lats, qlon, qlat, min_weight=0.5):
    """Bilinear sampling; weights are renormalized over valid neighbours; returns NaN if valid weight is insufficient.
    lats decrease from north to south."""
    if arr is None:
        return np.full(qlon.shape, np.nan)
    dx = lons[1] - lons[0]
    dy = lats[1] - lats[0]          # negative
    fx = (qlon - lons[0]) / dx
    fy = (qlat - lats[0]) / dy
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    tx = fx - i0
    ty = fy - j0
    ny, nx = arr.shape
    ok = (i0 >= 0) & (i0 < nx - 1) & (j0 >= 0) & (j0 < ny - 1)
    i0c = np.clip(i0, 0, nx - 2)
    j0c = np.clip(j0, 0, ny - 2)
    out = np.full(qlon.shape, np.nan)
    acc = np.zeros(qlon.shape)
    wsum = np.zeros(qlon.shape)
    for di, dj, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                      (0, 1, (1 - tx) * ty),       (1, 1, tx * ty)):
        v = arr[j0c + dj, i0c + di]
        good = np.isfinite(v)
        acc = acc + np.where(good, v * w, 0.0)
        wsum = wsum + np.where(good, w, 0.0)
    valid = ok & (wsum >= min_weight)
    out[valid] = acc[valid] / wsum[valid]
    return out


# ------------------------------------------------------------------ main workflow

def main():
    ext = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))
    R = ext['rect_aligned']
    ll = ext['latlon_bbox_nad83']
    print('Rectangle extent (EPSG:26917):', R)

    # ---- 30 m intermediate grid
    d30 = 30.0
    nx30 = int(round((R['xmax'] - R['xmin']) / d30))
    ny30 = int(round((R['ymax'] - R['ymin']) / d30))
    x30 = R['xmin'] + (np.arange(nx30) + 0.5) * d30
    y30 = R['ymax'] - (np.arange(ny30) + 0.5) * d30
    GX, GY = np.meshgrid(x30, y30)
    print(f'30 m reference grid {nx30} x {ny30}')
    qlon, qlat = geoio.utm_to_ll(GX, GY, ZONE)

    pad_deg = 0.02
    print('Reading USGS 1 arc-second DEM ...')
    dem_a, dem_lo, dem_la, dem_m = load_window_ll(
        DEM_SRC, ll['lonmin'] - pad_deg, ll['lonmax'] + pad_deg,
        ll['latmin'] - pad_deg, ll['latmax'] + pad_deg)
    print(f'  window {dem_a.shape}, valid {np.isfinite(dem_a).mean()*100:.2f}%')
    dem30 = bilinear(dem_a, dem_lo, dem_la, qlon, qlat)
    del dem_a

    print('Reading NOAA CRM ...')
    crm_a, crm_lo, crm_la, crm_m = load_window_ll(
        CRM_SRC, ll['lonmin'] - pad_deg, ll['lonmax'] + pad_deg,
        ll['latmin'] - pad_deg, ll['latmax'] + pad_deg)
    print(f'  window {crm_a.shape}, valid {np.isfinite(crm_a).mean()*100:.2f}%')
    crm30 = bilinear(crm_a, crm_lo, crm_la, qlon, qlat)
    del crm_a

    # ---- 30 m -> 60 m bilinear (equivalent to a 2x2 average when cell centers are aligned)
    def to60(a):
        ny, nx = a.shape
        b = a[:ny // 2 * 2, :nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2)
        with np.errstate(invalid='ignore'):
            return np.nanmean(b, axis=(1, 3))

    dem60 = to60(dem30)
    crm60 = to60(crm30)
    ny60, nx60 = dem60.shape
    tr60 = (R['xmin'], 60.0, 0.0, R['ymax'], 0.0, -60.0)
    print(f'60 m delivery grid {nx60} x {ny60}')

    # ---- diagnostics: onshore overlap DEM - CRM
    both = np.isfinite(dem60) & np.isfinite(crm60)
    diff = np.where(both, dem60 - crm60, np.nan)
    land = both & (dem60 > 1.0)          # clearly land, avoiding intertidal zone and water
    hi = both & (dem60 > 5.0)            # more conservative land
    stats = {}
    for lab, msk in (('overlap_all', both), ('land_dem_gt_1m', land),
                     ('land_dem_gt_5m', hi)):
        v = diff[msk]
        v = v[np.isfinite(v)]
        stats[lab] = dict(
            n=int(v.size),
            median=float(np.median(v)) if v.size else None,
            mean=float(v.mean()) if v.size else None,
            p05=float(np.percentile(v, 5)) if v.size else None,
            p95=float(np.percentile(v, 95)) if v.size else None,
            std=float(v.std()) if v.size else None)
        print(f'  DEM-CRM {lab:16s} n={stats[lab]["n"]:8d} '
              f'median={stats[lab]["median"]}')

    # ---- minimum merge (NoData excluded)
    # D2-b (confirmed by the user 2026-09-14): CRM participates in the merge only underwater.
    # The lower value is taken only where DEM <= LAND_THRESHOLD (0.5 m).
    # This threshold sits above the 3DEP flattened water surface (+0.10 m) and below real land (>1 m):
    # it preserves the riverbed while CRM never touches land.
    # Reason: CRM is a 90 m product smoothed over land; taking the minimum everywhere would replace the correct
    # 30 m DEM elevations with coarse low values (measured: land lowered by 0.246 m on average,
    # 3333 land cells originally >0 m pushed to <=0 m).
    underwater = both & (dem60 <= LAND_THRESHOLD)
    merged = np.where(underwater, np.fmin(dem60, crm60),
                      np.where(np.isfinite(dem60), dem60, crm60))
    src_flag = np.zeros(dem60.shape, dtype=np.uint8)
    src_flag[np.isfinite(dem60) & ~np.isfinite(crm60)] = 1
    src_flag[~np.isfinite(dem60) & np.isfinite(crm60)] = 2
    src_flag[both & ~underwater] = 3                    # land: keep DEM
    src_flag[underwater & (dem60 <= crm60)] = 3         # underwater but DEM is lower
    src_flag[underwater & (dem60 > crm60)] = 4          # underwater, use CRM

    valid = np.isfinite(merged)
    print(f'Merged result: valid {valid.sum()} / {merged.size}'
          f' ({valid.mean()*100:.2f}%)'
          f'  elevation {np.nanmin(merged):.2f} .. {np.nanmax(merged):.2f} m')

    # ---- write outputs
    pro = PTH.STATIC_TERRAIN
    inter = PTH.STATIC_INTERIM
    os.makedirs(pro, exist_ok=True)
    os.makedirs(inter, exist_ok=True)

    def w(path, arr, nd=NODATA, dtype=np.float32, cite=''):
        a = np.where(np.isfinite(arr), arr, nd).astype(dtype)
        geoio.write_geotiff(path, a, tr60, 26917, nodata=nd, citation=cite)
        return path

    w(os.path.join(pro, 'topobathy_merged_60m_26917.tif'), merged,
      cite='USGS 3DEP 1 arc-sec + NOAA CRM 3 arc-sec, min-merge restricted to '
           'underwater cells (D2-b), NO vertical transform applied')
    w(os.path.join(pro, 'dem_usgs_60m_26917.tif'), dem60,
      cite='USGS 3DEP 1 arc-second, bilinear to 60 m')
    w(os.path.join(pro, 'crm_60m_26917.tif'), crm60,
      cite='NOAA CRM 3 arc-second, bilinear to 60 m, datum UNVERIFIED')
    w(os.path.join(inter, 'diff_dem_minus_crm_60m_26917.tif'), diff)
    geoio.write_geotiff(os.path.join(inter, 'source_flag_60m_26917.tif'),
                        src_flag, tr60, 26917, nodata=0,
                        citation='0=none 1=DEM only 2=CRM only 3=DEM kept 4=CRM used (underwater)')

    np.save(os.path.join(inter, 'merged60.npy'), merged)
    np.save(os.path.join(inter, 'dem60.npy'), dem60)
    np.save(os.path.join(inter, 'crm60.npy'), crm60)

    meta = dict(
        generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        crs='EPSG:26917', cellsize_m=60.0,
        transform=tr60, shape=[int(ny60), int(nx60)], nodata=NODATA,
        units='m', vertical_datum='not unified / not converted -- see datum_conflict note',
        sources=dict(
            usgs_dem=dict(
                file=os.path.relpath(DEM_SRC, OLD),
                tiles=DEM_TILES,
                native_px_deg=dem_m['pixel_size'],
                native_arcsec=round(dem_m['pixel_size'][0] * 3600, 3),
                crs='EPSG:4269 (NAD83 geographic)',
                declared_vertical='no VerticalCSType in file; the official datum of the USGS 3DEP 1 arc-second'
                                  ' product is NAVD88 (not declared in the file)',
                nodata=dem_m.get('nodata')),
            noaa_crm=dict(
                file=os.path.relpath(CRM_SRC, OLD),
                alt_file=os.path.relpath(CRM_ALT, OLD),
                native_px_deg=crm_m['pixel_size'],
                native_arcsec=round(crm_m['pixel_size'][0] * 3600, 3),
                crs='EPSG:4326 (file declares WGS 84)',
                declared_vertical='no VerticalCSType in file -- datum unknown',
                nodata=crm_m.get('nodata'))),
        method=dict(
            resample='bilinear (source geographic -> 30 m EPSG:26917 -> 60 m)',
            merge=f'lower value taken only where DEM<={LAND_THRESHOLD} m, 30 m DEM kept everywhere else;'
                  ' NoData excluded; where only one source is valid, that value is kept',
            vertical_transform='not performed (datum not confirmed)',
            horizontal_note='CRM declares WGS84, DEM is NAD83; the difference here is'
                            ' about 1 m, far smaller than the 90 m source pixel, so no datum shift applied'),
        diff_stats=stats,
        valid_fraction=float(valid.mean()),
        elev_range=[float(np.nanmin(merged)), float(np.nanmax(merged))])
    with open(os.path.join(PTH.STATIC_META, 'terrain_60m.json'), 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print('Wrote data/meta/terrain_60m.json')


if __name__ == '__main__':
    main()
