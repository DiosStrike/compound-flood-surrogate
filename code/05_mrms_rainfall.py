#!/usr/bin/env python3
"""
05_mrms_rainfall.py -- prepare MRMS GaugeCorr_QPE_01H hourly rainfall.

Source: raw grib2 under data/mrms_raw_old/ of the old project (NOAA MRMS,
   GaugeCorr_QPE_01H = gauge-corrected 1-hour quantitative precipitation estimate).

Decoding: GRIB2 section 5 data representation template 41 (PNG compression), 16-bit packing,
     R=-30, E=0, D=1  ->  rainfall (mm) = (X - 30) / 10
     -3.0 is the MRMS "no radar coverage / missing" flag and 0.0 is true zero rain; the two are kept separate.

Grid: keep the native MRMS 0.01 deg (~1 km) lat/lon grid, **no reprojection, no resampling**.
     Only clipped in lat/lon to the study rectangle, with 2 extra pixels outward to ensure full coverage.

Time: file name / section 1 reference time = observation time. By product convention GaugeCorr_QPE_01H
     time stamp T represents the accumulation over [T-1h, T), consistent with this project's "label = interval end time"
     convention, so **no time shift is applied**. Note GRIB2 uses PDT=0 (instantaneous template);
     the file itself does not declare an accumulation interval -- this relies on the product definition, not file metadata.

Outputs:
  data/processed/mrms_gaugecorr_qpe01h_jax_0p01deg.nc  (NetCDF-3)
  data/processed/mrms_event_total_mm.csv               (areal-mean time series, for checking)
  data/meta/mrms_quality.json
"""
import os, sys, io, json, struct, glob, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import window_config as WCFG
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

OLD = PTH.OLD
NEW = PTH.ROOT
RAWDIR = PTH.OLD_MRMS

MISSING = -3.0
FILL = -9999.0


def read_grib2_sections(path):
    b = open(path, 'rb').read()
    if b[:4] != b'GRIB':
        raise ValueError('not GRIB')
    pos, sec = 16, {}
    while pos < len(b) - 4:
        if b[pos:pos + 4] == b'7777':
            break
        slen = struct.unpack('>I', b[pos:pos + 4])[0]
        sec[b[pos + 4]] = b[pos:pos + slen]
        pos += slen
    return sec


def grid_of(s3):
    gdtn = struct.unpack('>H', s3[12:14])[0]
    if gdtn != 0:
        raise ValueError(f'Unsupported grid template {gdtn}')
    ni, nj = struct.unpack('>II', s3[30:38])
    la1, lo1 = struct.unpack('>ii', s3[46:54])
    la2, lo2 = struct.unpack('>ii', s3[55:63])
    di, dj = struct.unpack('>II', s3[63:71])
    scan = s3[71]
    f = 1e-6
    lon0 = lo1 * f
    if lon0 > 180:
        lon0 -= 360.0
    lat0 = la1 * f
    dlon = di * f
    dlat = dj * f
    if scan & 0x40 == 0:          # j direction is north to south
        dlat = -dlat
    if scan & 0x80:
        dlon = -dlon
    return dict(ni=int(ni), nj=int(nj), lon0=lon0, lat0=lat0,
                dlon=dlon, dlat=dlat, scan=int(scan),
                lat2=la2 * f, lon2=(lo2 * f - 360 if lo2 * f > 180 else lo2 * f))


def unpack_values(sec):
    s5 = sec[5]
    npts = struct.unpack('>I', s5[5:9])[0]
    drtn = struct.unpack('>H', s5[9:11])[0]
    if drtn != 41:
        raise ValueError(f'Unsupported data representation template {drtn}')
    R = struct.unpack('>f', s5[11:15])[0]
    E = struct.unpack('>h', s5[15:17])[0]
    D = struct.unpack('>h', s5[17:19])[0]
    nbits = s5[19]
    img = Image.open(io.BytesIO(sec[7][5:]))
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[:, :, 0].astype(np.uint32) * 256 + a[:, :, 1]
    X = a.astype(np.float64)
    return (R + X * (2.0 ** E)) / (10.0 ** D), dict(
        R=float(R), E=int(E), D=int(D), nbits=int(nbits), npts=int(npts),
        drt_template=41)


def ref_time(s1):
    y = struct.unpack('>H', s1[12:14])[0]
    return datetime.datetime(y, s1[14], s1[15], s1[16], s1[17], s1[18])


def main():
    labels = list(WCFG.LABELS_DT)
    assert len(labels) == WCFG.N_HOURS
    print('Window:', WCFG.NOTE)

    ext = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))
    ll = ext['latlon_bbox_nad83']

    files, missing_files = [], []
    for t in labels:
        f = os.path.join(RAWDIR,
                         f'GaugeCorr_QPE_01H_00.00_{t:%Y%m%d-%H%M%S}.grib2')
        if os.path.exists(f):
            files.append((t, f))
        else:
            missing_files.append(t.strftime('%Y-%m-%dT%H:%M:%SZ'))
    print(f'Found {len(files)}/{WCFG.N_HOURS} raw files; {len(missing_files)} files missing')

    # ---- determine grid and window from the first file
    sec0 = read_grib2_sections(files[0][1])
    G = grid_of(sec0[3])
    lon_all = G['lon0'] + np.arange(G['ni']) * G['dlon']
    lat_all = G['lat0'] + np.arange(G['nj']) * G['dlat']
    pad = 2
    ci = np.where((lon_all >= ll['lonmin']) & (lon_all <= ll['lonmax']))[0]
    ri = np.where((lat_all >= ll['latmin']) & (lat_all <= ll['latmax']))[0]
    c0, c1 = max(0, ci[0] - pad), min(G['ni'], ci[-1] + 1 + pad)
    r0, r1 = max(0, ri[0] - pad), min(G['nj'], ri[-1] + 1 + pad)
    lons = lon_all[c0:c1]
    lats = lat_all[r0:r1]
    ny, nx = len(lats), len(lons)
    print(f'MRMS window {nx} x {ny} cells @ {abs(G["dlon"]):.3f} deg'
          f'  lon {lons.min():.4f}..{lons.max():.4f}'
          f'  lat {lats.min():.4f}..{lats.max():.4f}')
    print(f'  study rectangle lat/lon bbox {ll["lonmin"]:.4f}..{ll["lonmax"]:.4f} / '
          f'{ll["latmin"]:.4f}..{ll["latmax"]:.4f}  -> fully covered: '
          f'{lons.min() <= ll["lonmin"] and lons.max() >= ll["lonmax"] and lats.min() <= ll["latmin"] and lats.max() >= ll["latmax"]}')

    P = np.full((len(labels), ny, nx), np.nan)
    FLAG = np.full((len(labels), ny, nx), 2, dtype=np.int8)  # 2=file missing
    grid_consistent = True
    packing = None
    reftime_ok = []
    idx = {t: i for i, (t, _) in enumerate(files)}

    # ---- decode cache: data/interim/mrms_raw_cache_<t0>_<t1>.npy
    # Stores the "raw mm values before no-coverage handling", bit-identical to per-file decoding.
    # A cache hit skips GRIB2 decoding (~3 min); if the cache does not cover the needed period it falls back to decoding.
    CACHE = os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_raw_cache.npz')
    cache_hit = False
    if os.path.exists(CACHE):
        _z = np.load(CACHE, allow_pickle=True)
        _t = [datetime.datetime.utcfromtimestamp(x) for x in _z['t']]
        _pos = {tt: i for i, tt in enumerate(_t)}
        if (_z['P'].shape[1:] == (ny, nx) and all(t in _pos for t in labels)):
            _P = _z['P']
            for k, t in enumerate(labels):
                v = _P[_pos[t]]
                nocov = np.isclose(v, MISSING, atol=1e-6)
                P[k] = np.where(nocov, np.nan, v)
                FLAG[k] = np.where(nocov, 1, 0)
            packing = dict(R=-30.0, E=0, D=1, nbits=16, npts=int(G['ni'] * G['nj']),
                           drt_template=41)
            reftime_ok = [True] * len(labels)
            cache_hit = True
            print('  Decode cache hit %s, skipping GRIB2 decoding' % os.path.basename(CACHE))

    for k, t in enumerate(labels):
        if cache_hit:
            break
        if t not in idx:
            continue
        f = files[idx[t]][1]
        sec = read_grib2_sections(f)
        g = grid_of(sec[3])
        if (g['ni'], g['nj'], round(g['lon0'], 6), round(g['lat0'], 6)) != \
           (G['ni'], G['nj'], round(G['lon0'], 6), round(G['lat0'], 6)):
            grid_consistent = False
        rt = ref_time(sec[1])
        reftime_ok.append(rt == t)
        v, pk = unpack_values(sec)
        packing = packing or pk
        v = v.reshape(g['nj'], g['ni'])[r0:r1, c0:c1]
        nocov = np.isclose(v, MISSING, atol=1e-6)
        P[k] = np.where(nocov, np.nan, v)
        FLAG[k] = np.where(nocov, 1, 0)
        if (k + 1) % 48 == 0:
            print(f'  processed {k+1}/{WCFG.N_HOURS}')

    if not cache_hit:
        _raw = np.where(FLAG == 1, MISSING, np.where(FLAG == 2, np.nan, P))
        np.savez_compressed(
            CACHE, P=_raw.astype(np.float32),
            t=np.array([t.replace(tzinfo=datetime.timezone.utc).timestamp()
                        for t in labels], dtype=np.float64))
        print('  Wrote decode cache', os.path.basename(CACHE))

    # ---- quality statistics
    n_nocov = int((FLAG == 1).sum())
    n_filemiss = int((FLAG == 2).sum())
    n_valid = int((FLAG == 0).sum())
    true_zero = int(np.nansum(P == 0.0))
    neg = int(np.nansum(P < 0))
    total_mm = np.where(np.isfinite(P), P, 0.0).sum(axis=0)
    hours_valid = (FLAG == 0).sum(axis=0)
    areal = np.array([np.nanmean(P[k]) if np.isfinite(P[k]).any() else np.nan
                      for k in range(len(labels))])

    print(f'valid cell-hours {n_valid} | no coverage {n_nocov} | file missing {n_filemiss}'
          f' | true-zero cell-hours {true_zero} | negative (anomalous) {neg}')
    print(f'Event total rainfall: areal min {total_mm.min():.1f} mm '
          f'median {np.median(total_mm):.1f} mm max {total_mm.max():.1f} mm')
    top = np.argsort(areal)[-5:][::-1]
    print('Top 5 hours by areal-mean rain rate:')
    for i in top:
        print(f'   {labels[i]:%Y-%m-%d %H:%M} UTC  {areal[i]:.2f} mm/h')

    # ---- write NetCDF-3
    pro = PTH.IRMA['processed']
    os.makedirs(pro, exist_ok=True)
    ncpath = os.path.join(pro, 'mrms_gaugecorr_qpe01h_jax_0p01deg.nc')
    epoch = WCFG.T0_DT
    tvals = np.array([(t - epoch).total_seconds() / 3600.0 for t in labels])

    w = geoio.NC3Writer(ncpath)
    w.add_dim('time', len(labels))
    w.add_dim('lat', ny)
    w.add_dim('lon', nx)
    w.add_gattr('title', 'MRMS GaugeCorr_QPE_01H hourly precipitation, '
                         'Jacksonville study rectangle')
    w.add_gattr('source', 'NOAA MRMS GaugeCorr_QPE_01H GRIB2 '
                          '(local archive: Independent_Study_Flood/data/mrms_raw_old)')
    w.add_gattr('grid', 'native MRMS 0.01 deg lat/lon, NOT reprojected or resampled')
    w.add_gattr('crs', 'EPSG:4326 (native MRMS lat/lon grid)')
    w.add_gattr('time_convention',
                'time stamp T = accumulation over [T-1h, T); '
                'no time shift applied')
    w.add_gattr('history', f'created {datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} '
                           'by Flood_2.0/code/05_mrms_rainfall.py')
    w.add_var('time', ['time'], tvals.astype(np.float64),
              {'units': 'hours since %s:00' % WCFG.T0_ISO,
               'calendar': 'proleptic_gregorian',
               'long_name': 'end of 1-hour accumulation interval (UTC)',
               'axis': 'T'})
    w.add_var('lat', ['lat'], lats.astype(np.float64),
              {'units': 'degrees_north', 'standard_name': 'latitude',
               'axis': 'Y'})
    w.add_var('lon', ['lon'], lons.astype(np.float64),
              {'units': 'degrees_east', 'standard_name': 'longitude',
               'axis': 'X'})
    w.add_var('precip_mm', ['time', 'lat', 'lon'],
              np.where(np.isfinite(P), P, FILL).astype(np.float32),
              {'units': 'mm', 'long_name': 'hourly accumulated precipitation',
               '_FillValue': np.float32(FILL), 'missing_value': np.float32(FILL),
               'cell_methods': 'time: sum (interval: 1 hour, label: interval end)',
               'note': 'value 0.0 = true zero rainfall; _FillValue = '
                       'no radar coverage or source file missing (see quality_flag)'})
    w.add_var('precip_rate_mm_per_h', ['time', 'lat', 'lon'],
              np.where(np.isfinite(P), P, FILL).astype(np.float32),
              {'units': 'mm h-1', 'long_name':
               'mean rainfall intensity over the 1-hour interval',
               '_FillValue': np.float32(FILL),
               'note': 'values equal precip_mm because the accumulation interval is exactly 1 hour;'
                       ' this variable is the mean rain rate over the hour, not the instantaneous rate'})
    w.add_var('quality_flag', ['time', 'lat', 'lon'], FLAG,
              {'long_name': 'data quality flag',
               'flag_values': '0 1 2',
               'flag_meanings': 'valid no_radar_coverage source_file_missing'})
    w.add_var('event_total_mm', ['lat', 'lon'], total_mm.astype(np.float32),
              {'units': 'mm', 'long_name':
               'sum of valid hourly accumulations over the %d-hour window' % WCFG.N_HOURS,
               'note': 'no-coverage/missing hours count as 0 in the sum; interpret together with hours_valid'})
    w.add_var('hours_valid', ['lat', 'lon'], hours_valid.astype(np.int32),
              {'long_name': 'number of hours with quality_flag == 0',
               'valid_max': np.int32(WCFG.N_HOURS)})
    w.write()
    print('Wrote', ncpath)

    import csv
    with open(os.path.join(pro, 'mrms_areal_mean_hourly.csv'), 'w',
              newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['hour_end_utc', 'areal_mean_mm', 'n_valid_cells',
                     'n_nocoverage_cells', 'period'])
        for k, t in enumerate(labels):
            wr.writerow([t.strftime('%Y-%m-%dT%H:%M:%SZ'),
                         '' if not np.isfinite(areal[k]) else round(float(areal[k]), 4),
                         int((FLAG[k] == 0).sum()), int((FLAG[k] == 1).sum()),
                         'spinup' if t <= datetime.datetime(2017, 9, 6) else 'event'])

    np.save(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_P.npy'), P)
    np.save(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_FLAG.npy'), FLAG)
    np.save(os.path.join(PTH.IRMA['raw'], 'mrms_cache/mrms_lonlat.npy'),
            np.array([lons.min(), lons.max(), lats.min(), lats.max(),
                      G['dlon'], G['dlat']]))

    meta = dict(
        generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        product='NOAA MRMS GaugeCorr_QPE_01H (gauge-corrected 1-hour QPE)',
        raw_dir=os.path.relpath(RAWDIR, OLD),
        files_found=len(files), files_expected=WCFG.N_HOURS,
        files_missing=missing_files,
        grid=dict(native_deg=abs(G['dlon']), ni=G['ni'], nj=G['nj'],
                  crs='EPSG:4326', window_nx=nx, window_ny=ny,
                  lon_range=[float(lons.min()), float(lons.max())],
                  lat_range=[float(lats.min()), float(lats.max())],
                  covers_study_rect=bool(
                      lons.min() <= ll['lonmin'] and lons.max() >= ll['lonmax']
                      and lats.min() <= ll['latmin'] and lats.max() >= ll['latmax']),
                  resampled=False, reprojected=False),
        packing=packing,
        missing_convention=dict(
            mrms_no_coverage_value=MISSING,
            netcdf_fill=FILL,
            true_zero='precip_mm == 0.0 and quality_flag == 0'),
        time=dict(labels=[labels[0].strftime('%Y-%m-%dT%H:%M:%SZ'),
                          labels[-1].strftime('%Y-%m-%dT%H:%M:%SZ')],
                  n=len(labels),
                  reference_time_matches_label=bool(all(reftime_ok)),
                  significance_of_reference_time='3 = Observation time',
                  pdt_template=0,
                  caveat='GRIB2 uses PDT=0 (instantaneous template); the file does not declare an accumulation interval;'
                         ' the 1-hour accumulation and "time stamp = interval end" rely on the MRMS product definition,'
                         ' not on metadata in the file',
                  shift_applied='none'),
        grid_consistent_across_files=grid_consistent,
        counts=dict(valid=n_valid, no_coverage=n_nocov,
                    file_missing=n_filemiss, true_zero=true_zero,
                    negative=neg, total=int(P.size)),
        event_total_mm=dict(min=float(total_mm.min()),
                            median=float(np.median(total_mm)),
                            max=float(total_mm.max())),
        note='areal-mean time series is for check plots only and cannot replace the spatial rainfall input')
    with open(os.path.join(PTH.IRMA['meta'], 'mrms_quality.json'), 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print('Wrote data/meta/mrms_quality.json')


if __name__ == '__main__':
    main()
