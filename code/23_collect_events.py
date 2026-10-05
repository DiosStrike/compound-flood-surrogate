#!/usr/bin/env python3
"""
23_collect_events.py -- multi-event (Matthew / Ian / Milton / Dorian / Beryl) 20-day data collection, verification, organization, gap filling and plotting.

Usage (on the user's Mac, from the project root):
  python3 code/23_collect_events.py download all      # download raw data only (NOAA / USGS / MRMS / Stage IV); existing files are skipped
  python3 code/23_collect_events.py process  all      # process only (hourly statistics, gaps, interpolation, nc, figures)
  python3 code/23_collect_events.py all               # both steps
  all can be replaced by event names: matthew ian milton dorian beryl

Scope: each event is centered on the "main impact day over the study area", 10 days on each side, [T0, T1) = 480 hourly intervals; label T = [T-1h, T).
Raw data are fetched with 1 extra day on each side (for bracketing endpoint interpolation), used only for bracketing and never entering the 480-hour product.

Principles:
  * Raw responses are saved byte-for-byte (json / grib) and never rewritten; processed results are stored separately, raw values are never overwritten by interpolation.
  * Water level / discharge: keep native 6 min / 15 min records and raw gaps; hourly value = arithmetic mean of valid observations in the interval, with sample count and coverage recorded.
  * Rainfall: hourly accumulation in mm, spatial distribution kept (MRMS native 0.01 deg; Stage IV native ~4 km polar-stereographic grid); the areal mean is only for check plots.
  * Internal gaps are linearly interpolated in time (1 h missing = mean of neighbours; multi-hour runs by time distance); rainfall is interpolated per cell, no-coverage / missing files are not treated as zero rain.
  * Missing values at window endpoints: first look outside the window (±1 day raw data) for valid bracketing values; if none -> mark as unfillable and report, no extrapolation.
  * Each hour keeps source fractions: obs_frac (observed), official_est_frac (official estimate: NOAA inferred / USGS 'e'), partial (insufficient samples), interp (own interpolation).
"""
import os, sys, io, json, struct, gzip, hashlib, datetime, time, csv, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH, geoio

UTC = datetime.timezone.utc
FT3S_TO_M3S = 0.028316846592
FILL = -9999.0
MRMS_MISSING = -3.0

EVENTS = {
    'matthew': dict(name='Matthew', year=2016, center='2016-10-07', rain='GaugeCorr_QPE_01H', rain_src='mrms'),
    'ian':     dict(name='Ian',     year=2022, center='2022-09-29', rain='MultiSensor_QPE_01H_Pass2', rain_src='mrms'),
    'milton':  dict(name='Milton',  year=2024, center='2024-10-10', rain='MultiSensor_QPE_01H_Pass2', rain_src='mrms'),
    'dorian':  dict(name='Dorian',  year=2019, center='2019-09-04', rain='GaugeCorr_QPE_01H', rain_src='mrms'),
    'beryl':   dict(name='Beryl',   year=2012, center='2012-05-28', rain='StageIV_01h', rain_src='stage4'),
}
for k, e in EVENTS.items():
    c = datetime.datetime.fromisoformat(e['center']).replace(tzinfo=UTC)
    e['t0'] = c - datetime.timedelta(days=10)
    e['t1'] = c + datetime.timedelta(days=10)
    e['labels'] = [e['t0'] + datetime.timedelta(hours=h) for h in range(1, 481)]
    e['ext_labels'] = [e['t0'] + datetime.timedelta(hours=h) for h in range(-23, 505)]   # ±24 h bracketing segment
    e['dirs'] = PTH.event(k)

NOAA = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
USGS = 'https://nwis.waterservices.usgs.gov/nwis/iv/'
MT = 'https://mtarchive.geol.iastate.edu/{y}/{m}/{d}/mrms/ncep/{p}/{p}_00.00_{y}{m}{d}-{h}0000.grib2.gz'
ST4 = 'https://mesonet.agron.iastate.edu/archive/data/{y}/{m}/{d}/stage4/ST4.{y}{m}{d}{h}.01h.grib'


def log(*a):
    print(datetime.datetime.now().strftime('%H:%M:%S'), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ============================================================ download
def _get(url, retries=4, timeout=120):
    import requests
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Flood_2.0 data collection (research)'})
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return r
            last = 'HTTP %d' % r.status_code
        except Exception as ex:
            last = str(ex)
        time.sleep(2 + 3 * i)
    raise RuntimeError('%s -> %s' % (url, last))


def download(ev):
    e = EVENTS[ev]; D = e['dirs']
    raw = D['raw']; os.makedirs(raw, exist_ok=True)
    logf = os.path.join(raw, 'download_log.json')
    LOG = json.load(open(logf)) if os.path.exists(logf) else {'items': []}
    def rec(kind, url, path, params=None, note=''):
        LOG['items'].append(dict(kind=kind, url=url, file=os.path.relpath(path, PTH.ROOT), params=params,
                                 downloaded_utc=datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                 bytes=os.path.getsize(path), sha256=sha256(path), note=note))
    b = (e['t0'] - datetime.timedelta(days=1)).strftime('%Y%m%d'); en = (e['t1'] + datetime.timedelta(days=1)).strftime('%Y%m%d')
    # ---- NOAA observations + astronomical tide predictions (6 min, NAVD, metric, gmt)
    for prod in ('water_level', 'predictions'):
        out = os.path.join(raw, 'noaa_8720218_%s_raw.json' % prod)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            log(ev, prod, 'already exists, skipped'); continue
        params = dict(product=prod, application='Flood_2.0', begin_date=b, end_date=en, datum='NAVD', station='8720218',
                      time_zone='gmt', units='metric', interval='6', format='json')
        url = NOAA + '?' + '&'.join('%s=%s' % kv for kv in params.items())
        r = _get(url); open(out, 'wb').write(r.content)
        j = r.json(); key = 'data' if prod == 'water_level' else 'predictions'
        if key not in j:
            raise RuntimeError('NOAA %s returned no data: %s' % (prod, j))
        rec('noaa_' + prod, url, out, params, '%d records' % len(j[key])); log(ev, prod, len(j[key]), 'records')
    # ---- USGS IV 00060
    out = os.path.join(raw, 'usgs_02246500_00060_raw.json')
    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
        params = dict(format='json', sites='02246500', parameterCd='00060',
                      startDT=(e['t0'] - datetime.timedelta(days=1)).strftime('%Y-%m-%dT00:00Z'),
                      endDT=(e['t1'] + datetime.timedelta(days=1)).strftime('%Y-%m-%dT00:00Z'))
        url = USGS + '?' + '&'.join('%s=%s' % kv for kv in params.items())
        r = _get(url); open(out, 'wb').write(r.content)
        n = len(r.json()['value']['timeSeries'][0]['values'][0]['value'])
        rec('usgs_iv', url, out, params, '%d records' % n); log(ev, 'usgs', n, 'records')
    else:
        log(ev, 'usgs already exists, skipped')
    # ---- rainfall files (extended segment, 528 hours)
    rdir = os.path.join(raw, 'rain', e['rain']); os.makedirs(rdir, exist_ok=True)
    missing = []
    n_new = 0
    for t in e['ext_labels']:
        y, m, d, h = t.strftime('%Y'), t.strftime('%m'), t.strftime('%d'), t.strftime('%H')
        if e['rain_src'] == 'mrms':
            url = MT.format(y=y, m=m, d=d, h=h, p=e['rain'])
        else:
            url = ST4.format(y=y, m=m, d=d, h=h)
        out = os.path.join(rdir, url.rsplit('/', 1)[1])
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        r = _get(url)
        if r.status_code == 404:
            missing.append(t.strftime('%Y-%m-%dT%H:%MZ')); continue
        open(out, 'wb').write(r.content); n_new += 1
        if n_new % 48 == 0:
            log(ev, 'rainfall downloaded', n_new)
    LOG['rain'] = dict(product=e['rain'], source='IEM ' + ('mtarchive' if e['rain_src'] == 'mrms' else 'mesonet archive stage4'),
                       url_pattern=MT if e['rain_src'] == 'mrms' else ST4, dir=os.path.relpath(rdir, PTH.ROOT),
                       n_hours_requested=len(e['ext_labels']), n_files=len(os.listdir(rdir)), missing_hours_404=missing,
                       downloaded_utc=datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'))
    json.dump(LOG, open(logf, 'w'), ensure_ascii=False, indent=1)
    log(ev, 'rainfall files %d, %d missing (404)' % (LOG['rain']['n_files'], len(missing)))


# ============================================================ GRIB decoding
def read_grib2_sections(b):
    if b[:4] != b'GRIB':
        raise ValueError('not GRIB')
    pos, sec = 16, {}
    while pos < len(b) - 4:
        if b[pos:pos + 4] == b'7777':
            break
        slen = struct.unpack('>I', b[pos:pos + 4])[0]
        sec[b[pos + 4]] = b[pos:pos + slen]; pos += slen
    return sec


def mrms_decode(path):
    from PIL import Image
    b = gzip.open(path, 'rb').read()
    sec = read_grib2_sections(b)
    s3, s5 = sec[3], sec[5]
    gdtn = struct.unpack('>H', s3[12:14])[0]; assert gdtn == 0, gdtn
    ni, nj = struct.unpack('>II', s3[30:38]); la1, lo1 = struct.unpack('>ii', s3[46:54]); di, dj = struct.unpack('>II', s3[63:71]); scan = s3[71]
    lon0 = lo1 * 1e-6; lon0 = lon0 - 360 if lon0 > 180 else lon0; lat0 = la1 * 1e-6
    dlon = di * 1e-6 * (-1 if scan & 0x80 else 1); dlat = dj * 1e-6 * (1 if scan & 0x40 else -1)
    drtn = struct.unpack('>H', s5[9:11])[0]; assert drtn == 41, drtn
    R = struct.unpack('>f', s5[11:15])[0]; E = struct.unpack('>h', s5[15:17])[0]; Dd = struct.unpack('>h', s5[17:19])[0]
    assert sec[6][5] == 255, 'bitmap present (indicator %d)' % sec[6][5]
    a = np.asarray(Image.open(io.BytesIO(sec[7][5:])))
    if a.ndim == 3:
        a = a[:, :, 0].astype(np.uint32) * 256 + a[:, :, 1]
    v = (R + a.astype(np.float64) * 2.0 ** E) / 10.0 ** Dd
    s1 = sec[1]; rt = datetime.datetime(struct.unpack('>H', s1[12:14])[0], s1[14], s1[15], s1[16], s1[17], s1[18], tzinfo=UTC)
    return v.reshape(nj, ni), dict(ni=int(ni), nj=int(nj), lon0=lon0, lat0=lat0, dlon=dlon, dlat=dlat, R=R, E=E, D=Dd, reftime=rt)


def _ibm2float(b4):
    a = struct.unpack('>I', b4)[0]
    if a == 0:
        return 0.0
    s = -1.0 if a & 0x80000000 else 1.0
    ex = (a >> 24) & 0x7f; mant = a & 0xffffff
    return s * mant * 16.0 ** (ex - 64) / 2 ** 24


def _s3(b, p, signed=False):
    v = (b[p] << 16) | (b[p + 1] << 8) | b[p + 2]
    if signed and v & 0x800000:
        v = -(v & 0x7fffff)
    return v


def st4_decode(path):
    """GRIB1 simple packing + bitmap, polar-stereographic grid (NCEP HRAP, true latitude 60N, LoV=-105). Returns (values[ny,nx] south->north, meta)"""
    b = open(path, 'rb').read()
    assert b[:4] == b'GRIB' and b[7] == 1
    p = 8; pl = _s3(b, p)
    flags = b[p + 7]; param = b[p + 8]
    yy, mo, dd, hh = b[p + 12], b[p + 13], b[p + 14], b[p + 15]; P1, P2, tri, cent = b[p + 18], b[p + 19], b[p + 20], b[p + 24]
    Dd = struct.unpack('>h', b[p + 26:p + 28])[0]; Dd = -(Dd & 0x7fff) if Dd < 0 else Dd
    year = (cent - 1) * 100 + yy
    ref = datetime.datetime(year, mo, dd, hh, tzinfo=UTC)
    p += pl
    assert flags & 128, 'no GDS'
    gl = _s3(b, p); drt = b[p + 5]; assert drt == 5, drt
    nx = (b[p + 6] << 8) | b[p + 7]; ny = (b[p + 8] << 8) | b[p + 9]
    la1 = _s3(b, p + 10, True) / 1000; lo1 = _s3(b, p + 13, True) / 1000; lov = _s3(b, p + 17, True) / 1000
    dx = _s3(b, p + 20); dy = _s3(b, p + 23); scan = b[p + 27]
    p += gl
    bitmap = None
    if flags & 64:
        bl = _s3(b, p); unused = b[p + 3]
        bits = np.unpackbits(np.frombuffer(b[p + 6:p + bl], dtype=np.uint8))
        bitmap = bits[:nx * ny].astype(bool); p += bl
    bl = _s3(b, p); bf = b[p + 3]
    assert bf & 0xC0 == 0, 'not simple packing'
    E = struct.unpack('>h', b[p + 4:p + 6])[0]; E = -(E & 0x7fff) if E < 0 else E
    R = _ibm2float(b[p + 6:p + 10]); nb = b[p + 10]
    data = np.frombuffer(b[p + 11:p + bl], dtype=np.uint8)
    npts = int(bitmap.sum()) if bitmap is not None else nx * ny
    bits = np.unpackbits(data)[:npts * nb].reshape(npts, nb)
    X = bits.dot(1 << np.arange(nb - 1, -1, -1, dtype=np.int64)).astype(np.float64)
    vals = (R + X * 2.0 ** E) / 10.0 ** Dd
    full = np.full(nx * ny, np.nan)
    if bitmap is not None:
        full[bitmap] = vals
    else:
        full[:] = vals
    assert scan == 64, scan   # +i, +j (south to north)
    return full.reshape(ny, nx), dict(nx=nx, ny=ny, la1=la1, lo1=lo1, lov=lov, dx=dx, dy=dy, R=R, E=E, D=Dd,
                                      reftime=ref, P1=P1, P2=P2, tri=tri, param=param, nbits=nb)


def st4_latlon(meta):
    """Cell-center lat/lon of a GRIB1 polar-stereographic grid (north pole, true latitude 60N, sphere R=6371229 m). dx=4763 is rounded to integer m; the exact NCEP grid 240 value 4762.5 m is used."""
    Rm = 6371229.0; lat_ts = math.radians(60.0)
    dx = 4762.5 if abs(meta['dx'] - 4762.5) < 1 else float(meta['dx']); dy = dx
    de = (1 + math.sin(lat_ts)) * Rm
    la1, lo1, lov = math.radians(meta['la1']), math.radians(meta['lo1']), math.radians(meta['lov'])
    k = de * math.cos(la1) / (1 + math.sin(la1))
    x0 = k * math.sin(lo1 - lov); y0 = -k * math.cos(lo1 - lov)
    i = np.arange(meta['nx']); j = np.arange(meta['ny'])
    X = x0 + i[None, :] * dx; Y = y0 + j[:, None] * dy
    r2 = X ** 2 + Y ** 2
    lat = np.degrees(np.arcsin((de ** 2 - r2) / (de ** 2 + r2)))
    lon = np.degrees(lov + np.arctan2(X, -Y))
    return lat, lon


# ============================================================ hourly statistics and interpolation (generic)
def hourly_stats(df, tcol, vcol, labels, expected, est_col):
    """df: native records (NaN = missing value). Returns n_valid / mean / min / max / official_est_frac for each label."""
    t = df[tcol].values.astype('datetime64[ns]')
    lab = np.array([np.datetime64(l.replace(tzinfo=None)) for l in labels], dtype='datetime64[ns]')
    # a record falls into label T's interval [T-1h, T)
    idx = np.searchsorted(lab, t, side='right')    # t < lab[idx] ; and t >= lab[idx]-1h ?
    out = []
    v = df[vcol].values.astype(float); est = df[est_col].values.astype(bool)
    for k, L in enumerate(lab):
        m = (t >= L - np.timedelta64(1, 'h')) & (t < L)
        vv = v[m]; ok = np.isfinite(vv)
        n = int(ok.sum())
        out.append(dict(hour_end_utc=pd.Timestamp(L, tz='UTC'), n_records=int(m.sum()), n_valid=n, n_expected=expected,
                        coverage=n / expected, mean=float(vv[ok].mean()) if n else np.nan,
                        min=float(vv[ok].min()) if n else np.nan, max=float(vv[ok].max()) if n else np.nan,
                        official_est_frac=float(est[m][ok].mean()) if n else 0.0))
    return pd.DataFrame(out)


def interp_hours(vals):
    """vals: hourly values of the extended series (incl. the ±24 h bracketing segment) (NaN = missing); internal gaps are linearly interpolated in time.
    Returns filled, kind (0 observed/statistic, 1 interpolated, 3 cannot be bracketed)"""
    v = np.array(vals, float); n = len(v)
    kind = np.zeros(n, int); filled = v.copy()
    ok = np.isfinite(v); idx = np.where(ok)[0]
    for i in np.where(~ok)[0]:
        left = idx[idx < i]; right = idx[idx > i]
        if len(left) and len(right):
            a, b = left[-1], right[0]
            w = (i - a) / (b - a)
            filled[i] = v[a] * (1 - w) + v[b] * w; kind[i] = 1
        else:
            kind[i] = 3
    return filled, kind


def runs_of(mask, labels):
    """runs of consecutive True -> [(start_label, end_label, n)]"""
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((labels[i], labels[j], j - i + 1)); i = j + 1
        else:
            i += 1
    return out


# ============================================================ processing
def _files_sig(file_of):
    h = hashlib.sha256()
    for t in sorted(file_of):
        f = file_of[t]; h.update(('%s:%s:%d' % (t.isoformat(), os.path.basename(f) if f else '-', os.path.getsize(f) if f else 0)).encode())
    return h.hexdigest()


def process(ev):
    e = EVENTS[ev]; D = e['dirs']; raw, pro, meta = D['raw'], D['processed'], D['meta']
    for d in (pro, meta):
        os.makedirs(d, exist_ok=True)
    labels = e['labels']; ext = e['ext_labels']
    L = pd.DatetimeIndex([pd.Timestamp(t) for t in labels]); LX = pd.DatetimeIndex([pd.Timestamp(t) for t in ext])
    in_win = np.array([(t in set(labels)) for t in ext])
    Q = dict(event=ev, name=e['name'], center_utc=e['center'], window_utc=[e['t0'].isoformat(), e['t1'].isoformat()], n_hours=480,
             label_convention='label T = [T-1h, T), left-closed right-open; hourly value = arithmetic mean of valid observations in the interval', clamp='endpoint interpolation takes bracketing values from raw data up to ±24 h outside the window',
             generated_utc=datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), series={}, gaps=[])
    GAPS = []

    def gap_rows(series, mask_missing, mask_partial, kind, peak_hours, unit_note):
        for a, b, n in runs_of(mask_missing, ext):
            inwin = any((t in set(labels)) for t in [a, b])
            GAPS.append(dict(series=series, start_utc=a.isoformat(), end_utc=b.isoformat(), hours=n, type='whole hour missing',
                             method='linear interpolation in time' if all(k == 1 for k in kind[[ext.index(a), ext.index(b)]]) else 'cannot be bracketed (not filled)',
                             in_window=inwin, covers_peak=any(a <= p <= b for p in peak_hours), note=unit_note))
        for a, b, n in runs_of(mask_partial, ext):
            GAPS.append(dict(series=series, start_utc=a.isoformat(), end_utc=b.isoformat(), hours=n, type='insufficient samples within hour',
                             method='mean of available samples, no interpolation', in_window=any((t in set(labels)) for t in [a, b]),
                             covers_peak=any(a <= p <= b for p in peak_hours), note=unit_note))

    # ---------------------------------------------------------- water level
    j = json.load(open(os.path.join(raw, 'noaa_8720218_water_level_raw.json')))
    rows = []
    for r in j['data']:
        f = ((r.get('f') or '0,0,0,0').split(',') + ['0'] * 4)[:4]
        try: v = float(r['v'])
        except (ValueError, TypeError): v = np.nan
        rows.append(dict(time_utc=pd.Timestamp(r['t'], tz='UTC'), water_level_m_navd88=v, sigma_m=r.get('s', ''),
                         flag_inferred=f[0].strip() == '1', flag_flat_tolerance=f[1].strip() == '1',
                         flag_rate_of_change=f[2].strip() == '1', flag_temp_limit=f[3].strip() == '1', quality=r.get('q', '')))
    wl = pd.DataFrame(rows).sort_values('time_utc').drop_duplicates('time_utc')
    # missing native timestamps (6 min step)
    full6 = pd.date_range(e['t0'] - datetime.timedelta(days=1), e['t1'] + datetime.timedelta(days=1), freq='6min', tz='UTC')
    wl = wl.set_index('time_utc').reindex(full6).reset_index().rename(columns={'index': 'time_utc'})
    wl['record_missing'] = wl.water_level_m_navd88.isna()
    for c in ('flag_inferred', 'flag_flat_tolerance', 'flag_rate_of_change', 'flag_temp_limit'):
        wl[c] = wl[c].fillna(False).astype(bool)
    wl['source'] = 'NOAA CO-OPS 8720218 water_level datum=NAVD units=metric tz=gmt interval=6'
    wl[(wl.time_utc >= pd.Timestamp(e['t0'])) & (wl.time_utc <= pd.Timestamp(e['t1']))].to_csv(
        os.path.join(pro, 'mayport_8720218_waterlevel_6min_utc_navd88_m.csv'), index=False)
    H = hourly_stats(wl.assign(time_utc=wl.time_utc.dt.tz_convert(None)), 'time_utc', 'water_level_m_navd88', ext, 10, 'flag_inferred')
    filled, kind = interp_hours(H['mean'].values)
    H['value_filled'] = filled; H['interp'] = kind == 1; H['unfillable'] = kind == 3
    H['partial'] = (H.n_valid > 0) & (H.n_valid < H.n_expected)
    H['obs_frac'] = np.where(H.n_valid > 0, 1 - H.official_est_frac, 0.0)
    H['source_class'] = np.select([H.n_valid == 0, H.partial, H.official_est_frac > 0], ['interp', 'partial', 'contains_official_est'], 'obs')
    H.loc[(H.n_valid == 0) & H.unfillable, 'source_class'] = 'unfillable'
    H['in_window'] = in_win
    wl_h = H.rename(columns={'mean': 'water_level_m_navd88_hourly_mean', 'min': 'hourly_min', 'max': 'hourly_max', 'value_filled': 'water_level_m_navd88_filled'})
    wl_h['station'] = '8720218'; wl_h['datum'] = 'NAVD88'; wl_h['units'] = 'm'
    wl_h[wl_h.in_window].drop(columns='in_window').to_csv(os.path.join(pro, 'mayport_8720218_waterlevel_hourly_utc_navd88_m.csv'), index=False)
    W = wl_h[wl_h.in_window].reset_index(drop=True)
    pk_wl = [labels[int(np.nanargmax(W.water_level_m_navd88_filled.values))]]
    Q['series']['mayport_waterlevel'] = dict(
        station='NOAA CO-OPS 8720218 Mayport (Bar Pilots Dock)', product='water_level (observed total water level)', datum='NAVD88', units='m', native_step_min=6,
        native_expected=4800, native_valid=int((~wl.record_missing[(wl.time_utc >= pd.Timestamp(e['t0'])) & (wl.time_utc < pd.Timestamp(e['t1']))]).sum()),
        native_coverage=None, hours_total=480, hours_full=int((W.n_valid == 10).sum()), hours_partial=int(W.partial.sum()),
        hours_missing=int((W.n_valid == 0).sum()), hours_interp=int(W.interp.sum()), hours_unfillable=int(W.unfillable.sum()),
        hours_with_official_est=int((W.official_est_frac > 0).sum()), official_est_records=int(wl.flag_inferred[(wl.time_utc >= pd.Timestamp(e['t0'])) & (wl.time_utc < pd.Timestamp(e['t1']))].sum()),
        longest_gap_h=max([n for a, b, n in runs_of((W.n_valid == 0).values, labels)] or [0]),
        peak=dict(time_utc=pk_wl[0].isoformat(), value=float(np.nanmax(W.water_level_m_navd88_filled))),
        completeness_after_fill=float(np.isfinite(W.water_level_m_navd88_filled).mean()))
    Q['series']['mayport_waterlevel']['native_coverage'] = Q['series']['mayport_waterlevel']['native_valid'] / 4800
    gap_rows('mayport_waterlevel', (H.n_valid == 0).values, H.partial.values, kind, pk_wl, 'm NAVD88')
    # astronomical tide prediction (organized only, not used in statistics)
    jp = json.load(open(os.path.join(raw, 'noaa_8720218_predictions_raw.json')))
    pr = pd.DataFrame([dict(time_utc=pd.Timestamp(r['t'], tz='UTC'), tide_prediction_m_navd88=float(r['v'])) for r in jp['predictions']])
    pr[(pr.time_utc >= pd.Timestamp(e['t0'])) & (pr.time_utc <= pd.Timestamp(e['t1']))].to_csv(
        os.path.join(pro, 'mayport_8720218_tide_prediction_6min_utc_navd88_m.csv'), index=False)

    # ---------------------------------------------------------- discharge
    ju = json.load(open(os.path.join(raw, 'usgs_02246500_00060_raw.json')))
    ts = ju['value']['timeSeries'][0]; vals = ts['values'][0]['value']
    q = pd.DataFrame([dict(time_utc=pd.Timestamp(x['dateTime']).tz_convert('UTC'), value_original=float(x['value']) if x['value'] not in ('', None) else np.nan,
                           qualifiers='|'.join(x['qualifiers'])) for x in vals]).sort_values('time_utc').drop_duplicates('time_utc')
    nodata = float(ts['variable'].get('noDataValue', -999999))
    q.loc[q.value_original == nodata, 'value_original'] = np.nan
    full15 = pd.date_range(e['t0'] - datetime.timedelta(days=1), e['t1'] + datetime.timedelta(days=1), freq='15min', tz='UTC')
    q = q.set_index('time_utc').reindex(full15).reset_index().rename(columns={'index': 'time_utc'})
    q['record_missing'] = q.value_original.isna(); q['qualifiers'] = q.qualifiers.fillna('')
    q['original_unit'] = 'ft3/s'; q['discharge_m3s'] = q.value_original * FT3S_TO_M3S
    q['flag_estimated'] = q.qualifiers.str.contains('e', case=True); q['flag_provisional'] = q.qualifiers.str.contains('P'); q['flag_approved'] = q.qualifiers.str.contains('A')
    q['site_id'] = '02246500'; q['parameter_code'] = '00060'; q['source'] = 'USGS NWIS IV 02246500 00060 (raw signed total discharge, not de-tided; negative = flood-tide reversal)'
    q[(q.time_utc >= pd.Timestamp(e['t0'])) & (q.time_utc <= pd.Timestamp(e['t1']))].to_csv(os.path.join(pro, 'acosta_02246500_discharge_15min_utc_m3s.csv'), index=False)
    Hq = hourly_stats(q.assign(time_utc=q.time_utc.dt.tz_convert(None)), 'time_utc', 'discharge_m3s', ext, 4, 'flag_estimated')
    filled, kindq = interp_hours(Hq['mean'].values)
    Hq['value_filled'] = filled; Hq['interp'] = kindq == 1; Hq['unfillable'] = kindq == 3
    Hq['partial'] = (Hq.n_valid > 0) & (Hq.n_valid < Hq.n_expected); Hq['obs_frac'] = np.where(Hq.n_valid > 0, 1 - Hq.official_est_frac, 0.0)
    Hq['source_class'] = np.select([Hq.n_valid == 0, Hq.partial, Hq.official_est_frac > 0], ['interp', 'partial', 'contains_official_est'], 'obs')
    Hq.loc[(Hq.n_valid == 0) & Hq.unfillable, 'source_class'] = 'unfillable'
    Hq['in_window'] = in_win
    q_h = Hq.rename(columns={'mean': 'discharge_m3s_hourly_mean', 'min': 'hourly_min', 'max': 'hourly_max', 'value_filled': 'discharge_m3s_filled'})
    q_h['station'] = '02246500'; q_h['units'] = 'm3/s'; q_h['sign_convention'] = 'positive = downstream (seaward), negative = upstream (flood-tide reversal); negative values kept as-is'
    q_h[q_h.in_window].drop(columns='in_window').to_csv(os.path.join(pro, 'acosta_02246500_discharge_hourly_utc_m3s.csv'), index=False)
    Wq = q_h[q_h.in_window].reset_index(drop=True)
    pk_q = [labels[int(np.nanargmax(Wq.discharge_m3s_filled.values))], labels[int(np.nanargmin(Wq.discharge_m3s_filled.values))]]
    qwin = q[(q.time_utc >= pd.Timestamp(e['t0'])) & (q.time_utc < pd.Timestamp(e['t1']))]
    Q['series']['acosta_discharge'] = dict(
        station='USGS 02246500 St. Johns River at Acosta Bridge', parameter='00060 discharge, signed', units='m3/s (raw ft3/s × 0.028316846592)', native_step_min=15,
        native_expected=1920, native_valid=int((~qwin.record_missing).sum()), native_coverage=float((~qwin.record_missing).mean()),
        hours_total=480, hours_full=int((Wq.n_valid == 4).sum()), hours_partial=int(Wq.partial.sum()), hours_missing=int((Wq.n_valid == 0).sum()),
        hours_interp=int(Wq.interp.sum()), hours_unfillable=int(Wq.unfillable.sum()), hours_with_official_est=int((Wq.official_est_frac > 0).sum()),
        official_est_records=int(qwin.flag_estimated.sum()), qualifier_counts={k: int(v) for k, v in qwin.qualifiers.value_counts().items()},
        longest_gap_h=max([n for a, b, n in runs_of((Wq.n_valid == 0).values, labels)] or [0]),
        peak=dict(max_time_utc=pk_q[0].isoformat(), max=float(np.nanmax(Wq.discharge_m3s_filled)), min_time_utc=pk_q[1].isoformat(), min=float(np.nanmin(Wq.discharge_m3s_filled))),
        completeness_after_fill=float(np.isfinite(Wq.discharge_m3s_filled).mean()))
    gap_rows('acosta_discharge', (Hq.n_valid == 0).values, Hq.partial.values, kindq, pk_q, 'm3/s')

    # ---------------------------------------------------------- rainfall
    ext_j = json.load(open(os.path.join(PTH.STATIC_META, 'study_extent.json')))['latlon_bbox_nad83']
    rdir = os.path.join(raw, 'rain', e['rain'])
    P = None; FLAGF = None
    file_of = {}
    for t in ext:
        if e['rain_src'] == 'mrms':
            f = os.path.join(rdir, '%s_00.00_%s.grib2.gz' % (e['rain'], t.strftime('%Y%m%d-%H0000')))
        else:
            f = os.path.join(rdir, 'ST4.%s.01h.grib' % t.strftime('%Y%m%d%H'))
        file_of[t] = f if os.path.exists(f) and os.path.getsize(f) > 0 else None
    first = next(f for f in file_of.values() if f)
    if e['rain_src'] == 'mrms':
        v0, G = mrms_decode(first)
        lon_all = G['lon0'] + np.arange(G['ni']) * G['dlon']; lat_all = G['lat0'] + np.arange(G['nj']) * G['dlat']
        pad = 2
        ci = np.where((lon_all >= ext_j['lonmin']) & (lon_all <= ext_j['lonmax']))[0]; ri = np.where((lat_all >= ext_j['latmin']) & (lat_all <= ext_j['latmax']))[0]
        c0, c1 = ci[0] - pad, ci[-1] + 1 + pad; r0, r1 = ri[0] - pad, ri[-1] + 1 + pad
        lats = lat_all[r0:r1]; lons = lon_all[c0:c1]; LAT2, LON2 = np.meshgrid(lats, lons, indexing='ij')
        grid_note = 'MRMS native 0.01° lat/lon grid, cropped to the study rectangle with a 2-cell margin; not reprojected'
        crop = lambda v: v[r0:r1, c0:c1]
    else:
        v0, G = st4_decode(first)
        LATF, LONF = st4_latlon(G)
        pad = 0.06
        m = (LATF >= ext_j['latmin'] - pad) & (LATF <= ext_j['latmax'] + pad) & (LONF >= ext_j['lonmin'] - pad) & (LONF <= ext_j['lonmax'] + pad)
        rr, cc = np.where(m); r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        LAT2, LON2 = LATF[r0:r1, c0:c1], LONF[r0:r1, c0:c1]; lats = lons = None
        grid_note = 'NCEP Stage IV native HRAP polar-stereographic grid (4762.5 m, true latitude 60N, LoV −105°), cropped to the study rectangle plus 0.06°; not reprojected; lat/lon computed from GRIB1 GDS parameters on a sphere R=6371229 m'
        crop = lambda v: v[r0:r1, c0:c1]
    ny, nx = LAT2.shape
    nT = len(ext)
    P = np.full((nT, ny, nx), np.nan); FLAG = np.full((nT, ny, nx), 2, np.int8)   # 0 observed 1 no coverage 2 file missing
    reft_ok = []; decode_meta = None
    cache = os.path.join(raw, 'rain', 'rain_decoded_cache_%s.npz' % e['rain'])
    cache_hit = False
    if os.path.exists(cache):
        z = np.load(cache)
        if z['P'].shape == (nT, ny, nx) and np.allclose(z['t'], [t.timestamp() for t in ext]) and 'files_sha' in z and z['files_sha'] == _files_sig(file_of):
            P = z['P'].astype(np.float64); FLAG = z['FLAG']; decode_meta = json.loads(str(z['decode_meta'])); reft_ok = [bool(z['reft_ok'])]; cache_hit = True
            log(ev, 'rainfall decode cache hit, skipping GRIB decoding')
    for k, t in enumerate(ext):
        if cache_hit:
            break
        f = file_of[t]
        if f is None:
            continue
        if e['rain_src'] == 'mrms':
            v, g = mrms_decode(f); v = crop(v)
            nocov = np.isclose(v, MRMS_MISSING, atol=1e-6) | (v < -0.5) | (v > 5000)
            reft_ok.append(g['reftime'] == t)
            decode_meta = decode_meta or dict(R=g['R'], E=g['E'], D=g['D'], drt_template=41, missing_rule='value ≈ -3 or < -0.5 or > 5000 treated as no coverage')
        else:
            v, g = st4_decode(f); v = crop(v)
            nocov = ~np.isfinite(v)
            reft_ok.append(g['reftime'] + datetime.timedelta(hours=g['P2']) == t)
            decode_meta = decode_meta or dict(R=g['R'], E=g['E'], D=g['D'], nbits=g['nbits'], P1=g['P1'], P2=g['P2'], tri=g['tri'], param=g['param'],
                                              missing_rule='cells with GRIB1 bitmap 0 = no coverage', time_rule='file-name hour = reference time + P2 (1 h) = end of accumulation interval')
        P[k] = np.where(nocov, np.nan, v); FLAG[k] = np.where(nocov, 1, 0)
        if (k + 1) % 96 == 0:
            log(ev, 'rainfall decoded %d/%d' % (k + 1, nT))
    if not cache_hit:
        np.savez_compressed(cache, P=P.astype(np.float32), FLAG=FLAG, t=np.array([t.timestamp() for t in ext]), lat=LAT2, lon=LON2,
                            files_sha=_files_sig(file_of), decode_meta=json.dumps(decode_meta), reft_ok=bool(all(reft_ok)))
    # ---- per-cell interpolation in time (missing file / no coverage)
    PF = P.copy(); KIND = np.zeros_like(FLAG)   # 0 observed 1 interpolated (no coverage) 2 interpolated (file missing) 3 cannot be bracketed
    miss = ~np.isfinite(P)
    tt = np.arange(nT, dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            col = P[:, iy, ix]; mm = miss[:, iy, ix]
            if not mm.any():
                continue
            ok = np.where(~mm)[0]
            if len(ok) == 0:
                KIND[:, iy, ix] = 3; continue
            fill = np.interp(tt, tt[ok], col[ok])
            inside = mm & (tt >= ok[0]) & (tt <= ok[-1])
            PF[inside, iy, ix] = fill[inside]
            KIND[inside, iy, ix] = np.where(FLAG[inside, iy, ix] == 2, 2, 1)
            KIND[mm & ~inside, iy, ix] = 3
    # ---- areal mean (cell centers inside the study rectangle)
    inbox = (LAT2 >= ext_j['latmin']) & (LAT2 <= ext_j['latmax']) & (LON2 >= ext_j['lonmin']) & (LON2 <= ext_j['lonmax'])
    areal_raw = np.array([np.nanmean(P[k][inbox]) if np.isfinite(P[k][inbox]).any() else np.nan for k in range(nT)])
    areal_fill = np.array([np.nanmean(PF[k][inbox]) if np.isfinite(PF[k][inbox]).any() else np.nan for k in range(nT)])
    cov_cells = np.array([np.isfinite(P[k][inbox]).mean() for k in range(nT)])
    A = pd.DataFrame(dict(hour_end_utc=LX, areal_mean_mm_raw=areal_raw, areal_mean_mm_filled=areal_fill, cell_coverage_frac=cov_cells,
                          file_present=[file_of[t] is not None for t in ext], n_cells_interp=[int(((KIND[k] > 0) & (KIND[k] < 3) & inbox).sum()) for k in range(nT)],
                          n_cells_unfillable=[int(((KIND[k] == 3) & inbox).sum()) for k in range(nT)], in_window=in_win))
    A['cum_mm_filled'] = np.where(A.in_window, np.nancumsum(np.where(A.in_window, A.areal_mean_mm_filled, 0)), np.nan)
    A['source_class'] = np.select([~A.file_present, A.cell_coverage_frac == 0, A.cell_coverage_frac < 1], ['file_missing_interp', 'nocov_interp', 'partial_cells'], 'obs')
    A[A.in_window].drop(columns='in_window').to_csv(os.path.join(pro, 'rain_%s_areal_mean_hourly.csv' % e['rain']), index=False)
    # ---- NetCDF (480 hours)
    sel = np.where(in_win)[0]
    ncpath = os.path.join(pro, 'rain_%s_hourly_native_grid.nc' % e['rain'])
    w = geoio.NC3Writer(ncpath)
    w.add_dim('time', len(sel)); w.add_dim('y', ny); w.add_dim('x', nx)
    w.add_gattr('title', '%s hourly precipitation, Jacksonville study rectangle, event %s' % (e['rain'], e['name']))
    w.add_gattr('source', 'IEM archive (%s); files in %s' % ('mtarchive MRMS' if e['rain_src'] == 'mrms' else 'mesonet stage4', os.path.relpath(rdir, PTH.ROOT)))
    w.add_gattr('grid', grid_note); w.add_gattr('crs', 'EPSG:4326 lat/lon of cell centres given in lat/lon variables')
    w.add_gattr('time_convention', 'T = accumulation over [T-1h, T)'); w.add_gattr('units', 'mm per hour interval')
    w.add_gattr('fill_rule', 'precip_mm = raw (missing = FILL); precip_mm_filled = after per-cell linear interpolation in time; flag: 0 obs, 1 interp(no coverage), 2 interp(file missing), 3 unfillable')
    w.add_gattr('history', 'created %s by code/23_collect_events.py' % datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'))
    tv = np.array([(labels[i] - e['t0']).total_seconds() / 3600 for i in range(480)])
    w.add_var('time', ['time'], tv, {'units': 'hours since %s' % e['t0'].strftime('%Y-%m-%d %H:%M'), 'calendar': 'proleptic_gregorian', 'long_name': 'end of 1-hour interval (UTC)'})
    w.add_var('lat', ['y', 'x'], LAT2.astype(np.float64), {'units': 'degrees_north'}); w.add_var('lon', ['y', 'x'], LON2.astype(np.float64), {'units': 'degrees_east'})
    w.add_var('precip_mm', ['time', 'y', 'x'], np.where(np.isfinite(P[sel]), P[sel], FILL).astype(np.float32), {'units': 'mm', '_FillValue': np.float32(FILL), 'missing_value': np.float32(FILL)})
    w.add_var('precip_mm_filled', ['time', 'y', 'x'], np.where(np.isfinite(PF[sel]), PF[sel], FILL).astype(np.float32), {'units': 'mm', '_FillValue': np.float32(FILL), 'missing_value': np.float32(FILL)})
    w.add_var('flag', ['time', 'y', 'x'], KIND[sel].astype(np.int8), {'flag_values': '0 1 2 3', 'flag_meanings': 'obs interp_nocov interp_filemissing unfillable'})
    w.add_var('in_study_rect', ['y', 'x'], inbox.astype(np.int8), {'long_name': 'cell centre inside study rectangle (used for areal mean)'})
    w.write()
    Aw = A[A.in_window].reset_index(drop=True)
    pk_r = [labels[int(np.nanargmax(Aw.areal_mean_mm_filled.values))]]
    fm = [t for t in labels if file_of[t] is None]
    Q['series']['rain'] = dict(product=e['rain'], source=('IEM mtarchive' if e['rain_src'] == 'mrms' else 'IEM mesonet archive stage4'), grid=grid_note,
                               grid_shape=[int(ny), int(nx)], n_cells_in_rect=int(inbox.sum()), decode=decode_meta, reftime_check_all_ok=bool(all(reft_ok)),
                               hours_total=480, hours_file_missing=len(fm), file_missing_hours=[t.isoformat() for t in fm],
                               hours_with_nocov_cells=int(((FLAG[sel] == 1) & inbox[None]).any(axis=(1, 2)).sum()),
                               cellhours_obs=int(((KIND[sel] == 0) & inbox[None]).sum()), cellhours_interp=int((((KIND[sel] == 1) | (KIND[sel] == 2)) & inbox[None]).sum()),
                               cellhours_unfillable=int(((KIND[sel] == 3) & inbox[None]).sum()),
                               longest_gap_h=max([n for a, b, n in runs_of(np.array([file_of[t] is None or cov_cells[ext.index(t)] == 0 for t in labels]), labels)] or [0]),
                               event_total_mm_areal=float(np.nansum(Aw.areal_mean_mm_filled)), peak=dict(time_utc=pk_r[0].isoformat(), areal_mm=float(np.nanmax(Aw.areal_mean_mm_filled))),
                               areal_mean_method='hourly arithmetic mean (nanmean) over cell centers inside the study rectangle (study_extent.json latlon bbox); for check plots only',
                               completeness_after_fill=float(np.isfinite(PF[sel][:, inbox]).mean()))
    rain_missing_mask = np.array([(file_of[t] is None) or (cov_cells[k] == 0) for k, t in enumerate(ext)])
    rain_partial = np.array([(0 < cov_cells[k] < 1) for k in range(nT)])
    kind_r = np.where(rain_missing_mask, np.where(np.array([(KIND[k][inbox] == 3).any() for k in range(nT)]), 3, 1), 0)
    gap_rows('rain_%s' % e['rain'], rain_missing_mask, rain_partial, kind_r, pk_r, 'mm/h areal mean')

    # ---------------------------------------------------------- gap list / quality json
    G = pd.DataFrame(GAPS, columns=['series', 'start_utc', 'end_utc', 'hours', 'type', 'method', 'in_window', 'covers_peak', 'note'])
    G.to_csv(os.path.join(meta, 'gaps_%s.csv' % ev), index=False)
    Q['gaps'] = GAPS
    Q['peaks'] = dict(waterlevel=Q['series']['mayport_waterlevel']['peak'], discharge=Q['series']['acosta_discharge']['peak'], rain=Q['series']['rain']['peak'])
    json.dump(Q, open(os.path.join(meta, 'collection_quality_%s.json' % ev), 'w'), ensure_ascii=False, indent=1, default=str)
    # ---------------------------------------------------------- figures
    plot_event(ev, W, Wq, Aw, Q)
    write_readme(ev, Q)
    log(ev, 'processing finished')
    return Q


def plot_event(ev, W, Wq, Aw, Q):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    e = EVENTS[ev]; t = pd.to_datetime(W.hour_end_utc)
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    C = dict(obs='#1f4e79', est='#e08a1e', interp='#c0392b', partial='#7f7f7f')
    def series(a, df, ycol, ylabel):
        y = df[ycol].values
        a.plot(t, y, color=C['obs'], lw=1.0, label='hourly value (mean of observations)')
        m = df.official_est_frac > 0; a.plot(t[m], y[m], 'o', ms=3.5, color=C['est'], label='contains official estimate')
        m = df.partial; a.plot(t[m], y[m], 's', ms=5, mfc='none', mec=C['partial'], label='insufficient samples within hour')
        m = df.interp; a.plot(t[m], y[m], 'x', ms=6, color=C['interp'], label='own interpolation')
        m = df.unfillable; a.plot(t[m], np.full(m.sum(), np.nanmin(y)), 'v', color='k', label='cannot be bracketed')
        for a0, b0, n in runs_of((df.n_valid == 0).values, list(t)):
            a.axvspan(a0 - pd.Timedelta('1h'), b0, color=C['interp'], alpha=0.12, lw=0)
        a.set_ylabel(ylabel); a.grid(alpha=0.3)
    series(ax[0], W, 'water_level_m_navd88_filled', 'Mayport water level (m NAVD88)')
    series(ax[1], Wq, 'discharge_m3s_filled', 'Acosta discharge (m³/s, + downstream)')
    ax[1].axhline(0, color='k', lw=0.6)
    ta = pd.to_datetime(Aw.hour_end_utc)
    ax[2].bar(ta, Aw.areal_mean_mm_filled, width=1 / 24, color=C['obs'], label='areal-mean hourly rain (mm)')
    m = Aw.source_class != 'obs'
    ax[2].bar(ta[m], Aw.areal_mean_mm_filled[m], width=1 / 24, color=C['interp'], label='hours with interpolation / partial coverage')
    for a0, b0, n in runs_of((~Aw.file_present | (Aw.cell_coverage_frac == 0)).values, list(ta)):
        ax[2].axvspan(a0 - pd.Timedelta('1h'), b0, color=C['interp'], alpha=0.12, lw=0)
    ax[2].set_ylabel('study-area mean hourly rain (mm)'); ax[2].grid(alpha=0.3)
    ax[3].plot(ta, Aw.cum_mm_filled, color=C['obs'], lw=1.4, label='areal-mean cumulative rain (after filling)')
    ax[3].plot(ta, np.nancumsum(Aw.areal_mean_mm_raw.fillna(0)), color=C['partial'], lw=1, ls='--', label='cumulative (raw, missing counted as 0, reference only)')
    ax[3].set_ylabel('cumulative rain (mm)'); ax[3].grid(alpha=0.3)
    c = pd.Timestamp(e['center'])
    for a in ax:
        a.axvline(c, color='#2a7f62', lw=1.2, ls='--'); a.axvline(c + pd.Timedelta('1D'), color='#2a7f62', lw=0.6, ls=':')
    P = Q['peaks']
    ax[0].plot(pd.Timestamp(P['waterlevel']['time_utc']), P['waterlevel']['value'], '*', ms=12, color='gold', mec='k', label='peak %.2f m @ %s' % (P['waterlevel']['value'], P['waterlevel']['time_utc'][5:16]))
    ax[1].plot(pd.Timestamp(P['discharge']['max_time_utc']), P['discharge']['max'], '*', ms=12, color='gold', mec='k', label='max %.0f @ %s' % (P['discharge']['max'], P['discharge']['max_time_utc'][5:16]))
    ax[1].plot(pd.Timestamp(P['discharge']['min_time_utc']), P['discharge']['min'], '*', ms=12, color='#9ecae1', mec='k', label='min %.0f @ %s' % (P['discharge']['min'], P['discharge']['min_time_utc'][5:16]))
    ax[2].plot(pd.Timestamp(P['rain']['time_utc']), P['rain']['areal_mm'], '*', ms=12, color='gold', mec='k', label='peak %.1f mm/h @ %s' % (P['rain']['areal_mm'], P['rain']['time_utc'][5:16]))
    for a in ax:
        h, l = a.get_legend_handles_labels(); a.legend(h, l, fontsize=7.5, loc='upper left', ncol=3)
    ax[3].set_xlabel('UTC (green dashed line = center day %s 00:00; red shading = raw whole-hour gaps)' % e['center'])
    fig.suptitle('%s %d -- 20-day collection range %s → %s UTC (480 h, label T=[T−1h,T))  rainfall %s' % (e['name'], e['year'], e['t0'].strftime('%Y-%m-%d'), e['t1'].strftime('%Y-%m-%d'), e['rain']), fontsize=11)
    fig.tight_layout()
    fdir = os.path.join(PTH.FIG, 'events'); os.makedirs(fdir, exist_ok=True)
    fig.savefig(os.path.join(fdir, 'fig_%s_20d_timeseries.png' % ev), dpi=140); fig.savefig(os.path.join(fdir, 'fig_%s_20d_timeseries.svg' % ev))
    plt.close(fig)
    pd.DataFrame(dict(hour_end_utc=W.hour_end_utc, water_level_m=W.water_level_m_navd88_filled, wl_class=W.source_class, discharge_m3s=Wq.discharge_m3s_filled, q_class=Wq.source_class,
                      rain_areal_mm=Aw.areal_mean_mm_filled, rain_class=Aw.source_class, rain_cum_mm=Aw.cum_mm_filled)).to_csv(os.path.join(fdir, 'fig_%s_20d_timeseries_data.csv' % ev), index=False)


def write_readme(ev, Q):
    e = EVENTS[ev]; D = e['dirs']
    s = Q['series']
    txt = f"""data/{ev}/ -- {e['name']} {e['year']} 20-day collected data (generated {Q['generated_utc']}, code/23_collect_events.py)
Range [{e['t0']:%Y-%m-%d %H:%M}, {e['t1']:%Y-%m-%d %H:%M}) UTC, 480 hourly intervals, label T=[T-1h,T); center day {e['center']}
raw/        raw responses (byte-for-byte): noaa_8720218_water_level_raw.json / predictions; usgs_02246500_00060_raw.json; rain/{e['rain']}/ hourly files; download_log.json (URL, parameters, time, sha256)
processed/  mayport_*_6min (native + gaps + NOAA quality flags) / mayport_*_hourly (hourly mean, n_valid, coverage, official_est_frac, interp, value_filled, source_class)
            acosta_*_15min / acosta_*_hourly (same as above; USGS 'e' = official estimate)
            rain_{e['rain']}_hourly_native_grid.nc (precip_mm raw / precip_mm_filled interpolated / flag; native grid, not reprojected)
            rain_{e['rain']}_areal_mean_hourly.csv (mean over cell centers in the study rectangle; for check plots only)
meta/       collection_quality_{ev}.json (coverage, gaps, peaks, decoding parameters) gaps_{ev}.csv (gap list)
Quality: water level  native coverage {s['mayport_waterlevel']['native_coverage']:.4f}, partial hours {s['mayport_waterlevel']['hours_partial']}, official-estimate hours {s['mayport_waterlevel']['hours_with_official_est']}, interpolated hours {s['mayport_waterlevel']['hours_interp']}, longest gap {s['mayport_waterlevel']['longest_gap_h']} h
         discharge    native coverage {s['acosta_discharge']['native_coverage']:.4f}, partial hours {s['acosta_discharge']['hours_partial']}, official-estimate hours {s['acosta_discharge']['hours_with_official_est']}, interpolated hours {s['acosta_discharge']['hours_interp']}, longest gap {s['acosta_discharge']['longest_gap_h']} h
         rainfall     missing-file hours {s['rain']['hours_file_missing']}, hours with uncovered cells {s['rain']['hours_with_nocov_cells']}, interpolated cell-hours {s['rain']['cellhours_interp']}, unbracketable cell-hours {s['rain']['cellhours_unfillable']}
Raw values are never overwritten by interpolation; interpolated values are written only to *_filled columns/variables. The final 10-day window is not yet fixed (awaiting user-specified start day).
"""
    open(os.path.join(D['meta'], 'README_%s.txt' % ev), 'w', encoding='utf-8').write(txt)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    evs = list(EVENTS) if (len(sys.argv) < 3 or sys.argv[2] == 'all') else sys.argv[2:]
    t_start = time.time()
    for ev in evs:
        if mode in ('download', 'all'):
            download(ev)
        if mode in ('process', 'all'):
            process(ev)
    log('All done, elapsed %.1f min' % ((time.time() - t_start) / 60))
