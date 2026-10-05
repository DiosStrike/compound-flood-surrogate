"""
geoio.py -- minimal geospatial IO toolkit in pure Python / numpy.

The only reason it exists: the egress network policy of the original session blocked pypi / apt,
so GDAL / rasterio / geopandas / pyproj could not be installed.
This module implements only what this project needs for review and processing; it is not meant to be general.

Implemented:
  * TIFF/GeoTIFF IFD parsing (incl. BigTIFF), reading geometry and GeoKey metadata
  * GeoTIFF pixel reading (strips/tiles; uncompressed, LZW, Deflate)
  * GeoTIFF writing (single band, Deflate, with GeoKey / nodata)
  * ESRI Shapefile (.shp/.dbf/.prj) reading
  * GeoPackage (.gpkg) writing
  * NAD83/GRS80 lon/lat <-> UTM transverse Mercator, forward and inverse
  * NetCDF-3 classic writing
All functions only read source files and never modify them in place.
"""
import struct, zlib, io, os, math, sqlite3, datetime
import numpy as np

# ---------------------------------------------------------------- TIFF parsing

_TIFF_TYPES = {
    1: ('B', 1), 2: ('c', 1), 3: ('H', 2), 4: ('I', 4), 5: ('II', 8),
    6: ('b', 1), 7: ('B', 1), 8: ('h', 2), 9: ('i', 4), 10: ('ii', 8),
    11: ('f', 4), 12: ('d', 8), 16: ('Q', 8), 17: ('q', 8), 18: ('Q', 8),
}

TAG_NAMES = {
    256: 'ImageWidth', 257: 'ImageLength', 258: 'BitsPerSample',
    259: 'Compression', 262: 'Photometric', 273: 'StripOffsets',
    277: 'SamplesPerPixel', 278: 'RowsPerStrip', 279: 'StripByteCounts',
    284: 'PlanarConfig', 317: 'Predictor', 322: 'TileWidth',
    323: 'TileLength', 324: 'TileOffsets', 325: 'TileByteCounts',
    339: 'SampleFormat', 33550: 'ModelPixelScale', 33922: 'ModelTiepoint',
    34264: 'ModelTransformation', 34735: 'GeoKeyDirectory',
    34736: 'GeoDoubleParams', 34737: 'GeoAsciiParams',
    42112: 'GDAL_METADATA', 42113: 'GDAL_NODATA',
}

# GeoTIFF GeoKey ID -> name (only those used in this project)
GEOKEY_NAMES = {
    1024: 'GTModelType', 1025: 'GTRasterType', 1026: 'GTCitation',
    2048: 'GeographicType', 2049: 'GeogCitation', 2050: 'GeogGeodeticDatum',
    2054: 'GeogAngularUnits', 2056: 'GeogEllipsoid',
    3072: 'ProjectedCSType', 3073: 'PCSCitation', 3076: 'ProjLinearUnits',
    4096: 'VerticalCSType', 4097: 'VerticalCitation',
    4098: 'VerticalDatum', 4099: 'VerticalUnits',
}


class TiffFile:
    """Minimal TIFF/BigTIFF reader."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, 'rb')
        head = self.fh.read(8)
        if head[:2] == b'II':
            self.e = '<'
        elif head[:2] == b'MM':
            self.e = '>'
        else:
            raise ValueError(f'{path}: not a TIFF')
        magic = struct.unpack(self.e + 'H', head[2:4])[0]
        self.big = (magic == 43)
        if self.big:
            self.fh.seek(8)
            off = struct.unpack(self.e + 'Q', self.fh.read(8))[0]
        else:
            off = struct.unpack(self.e + 'I', head[4:8])[0]
        self.ifds = []
        while off:
            ifd, off = self._read_ifd(off)
            self.ifds.append(ifd)
            if len(self.ifds) > 64:
                break

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _read_ifd(self, off):
        f, e = self.fh, self.e
        f.seek(off)
        if self.big:
            n = struct.unpack(e + 'Q', f.read(8))[0]
            entry_size, cnt_fmt, off_fmt, off_sz = 20, 'Q', 'Q', 8
        else:
            n = struct.unpack(e + 'H', f.read(2))[0]
            entry_size, cnt_fmt, off_fmt, off_sz = 12, 'I', 'I', 4
        raw = f.read(entry_size * n)
        tags = {}
        for i in range(n):
            b = raw[i * entry_size:(i + 1) * entry_size]
            tag, typ = struct.unpack(e + 'HH', b[:4])
            cnt = struct.unpack(e + cnt_fmt, b[4:4 + off_sz])[0]
            valbytes = b[4 + off_sz:]
            tags[tag] = self._read_value(typ, cnt, valbytes, off_sz)
        nxt = struct.unpack(e + off_fmt, f.read(off_sz))[0]
        return tags, nxt

    def _read_value(self, typ, cnt, valbytes, off_sz):
        if typ not in _TIFF_TYPES:
            return None
        fmt, size = _TIFF_TYPES[typ]
        total = size * cnt
        if total <= off_sz:
            data = valbytes[:total]
        else:
            off = struct.unpack(self.e + ('Q' if off_sz == 8 else 'I'),
                                valbytes[:off_sz])[0]
            cur = self.fh.tell()
            self.fh.seek(off)
            data = self.fh.read(total)
            self.fh.seek(cur)
        if typ == 2:
            return data.rstrip(b'\x00').decode('latin-1', 'replace')
        if typ in (5, 10):
            nums = struct.unpack(self.e + ('I' if typ == 5 else 'i') * (2 * cnt), data)
            return [nums[2 * i] / nums[2 * i + 1] if nums[2 * i + 1] else 0.0
                    for i in range(cnt)]
        vals = list(struct.unpack(self.e + fmt * cnt, data))
        return vals[0] if cnt == 1 else vals

    # -------------------------------------------------------- metadata

    def meta(self, idx=0):
        t = self.ifds[idx]
        m = {'path': self.path, 'bigtiff': self.big,
             'byteorder': 'little' if self.e == '<' else 'big'}
        for tag, name in TAG_NAMES.items():
            if tag in t and tag not in (273, 279, 324, 325):
                m[name] = t[tag]
        m['width'] = t.get(256)
        m['height'] = t.get(257)
        m['bands'] = t.get(277, 1)
        m['tiled'] = 322 in t
        # affine transform
        if 33550 in t and 33922 in t:
            sx, sy, sz = t[33550]
            tp = t[33922]
            i, j, k, x, y, z = tp[:6]
            m['transform'] = (x - i * sx, sx, 0.0, y + j * sy, 0.0, -sy)
            m['pixel_size'] = (sx, sy)
        elif 34264 in t:
            a = t[34264]
            m['transform'] = (a[3], a[0], a[1], a[7], a[4], a[5])
            m['pixel_size'] = (abs(a[0]), abs(a[5]))
        # GeoKeys
        m['geokeys'] = self._geokeys(t)
        # extent
        if 'transform' in m and m['width']:
            x0, dx, _, y0, _, dy = m['transform']
            m['bounds'] = (x0, y0 + dy * m['height'],
                           x0 + dx * m['width'], y0)
        # nodata
        if 42113 in t:
            try:
                m['nodata'] = float(t[42113])
            except (TypeError, ValueError):
                m['nodata'] = t[42113]
        return m

    def _geokeys(self, t):
        if 34735 not in t:
            return {}
        d = t[34735]
        dbl = t.get(34736, [])
        asc = t.get(34737, '')
        if isinstance(dbl, float):
            dbl = [dbl]
        out = {}
        n = d[3]
        for i in range(1, n + 1):
            kid, loc, cnt, val = d[4 * i:4 * i + 4]
            name = GEOKEY_NAMES.get(kid, f'Key{kid}')
            if loc == 0:
                out[name] = val
            elif loc == 34736:
                out[name] = dbl[val] if val < len(dbl) else None
            elif loc == 34737:
                out[name] = asc[val:val + cnt].rstrip('|\x00')
        return out

    # -------------------------------------------------------- pixel reading

    def read(self, idx=0, window=None):
        """Read the full image or a window (row0, row1, col0, col1). Returns a 2D numpy array."""
        t = self.ifds[idx]
        w, h = t[256], t[257]
        bps = t.get(258, 8)
        if isinstance(bps, list):
            bps = bps[0]
        sf = t.get(339, 1)
        if isinstance(sf, list):
            sf = sf[0]
        comp = t.get(259, 1)
        pred = t.get(317, 1)
        spp = t.get(277, 1)
        if spp != 1:
            raise NotImplementedError('only single band is supported')
        dt = self._dtype(bps, sf)
        if window is None:
            window = (0, h, 0, w)
        r0, r1, c0, c1 = window
        out = np.zeros((r1 - r0, c1 - c0), dtype=dt)

        if 322 in t:  # tiles
            tw, th = t[322], t[323]
            offs, cnts = self._as_list(t[324]), self._as_list(t[325])
            ntx = (w + tw - 1) // tw
            for ty in range(r0 // th, (r1 - 1) // th + 1):
                for tx in range(c0 // tw, (c1 - 1) // tw + 1):
                    k = ty * ntx + tx
                    if k >= len(offs):
                        continue
                    buf = self._decode(offs[k], cnts[k], comp, pred, dt, tw, th)
                    if buf is None:
                        continue
                    tile = buf.reshape(th, tw)
                    ys, xs = ty * th, tx * tw
                    a0, a1 = max(r0, ys), min(r1, ys + th)
                    b0, b1 = max(c0, xs), min(c1, xs + tw)
                    if a1 > a0 and b1 > b0:
                        out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = \
                            tile[a0 - ys:a1 - ys, b0 - xs:b1 - xs]
        else:  # strips
            rps = t.get(278, h)
            if isinstance(rps, list):
                rps = rps[0]
            offs, cnts = self._as_list(t[273]), self._as_list(t[279])
            for s in range(r0 // rps, (r1 - 1) // rps + 1):
                if s >= len(offs):
                    continue
                rows = min(rps, h - s * rps)
                buf = self._decode(offs[s], cnts[s], comp, pred, dt, w, rows)
                if buf is None:
                    continue
                strip = buf.reshape(rows, w)
                ys = s * rps
                a0, a1 = max(r0, ys), min(r1, ys + rows)
                if a1 > a0:
                    out[a0 - r0:a1 - r0, :] = strip[a0 - ys:a1 - ys, c0:c1]
        return out

    @staticmethod
    def _as_list(v):
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _dtype(bps, sf):
        if sf == 3:
            return {32: np.float32, 64: np.float64}[bps]
        if sf == 2:
            return {8: np.int8, 16: np.int16, 32: np.int32}[bps]
        return {1: np.uint8, 8: np.uint8, 16: np.uint16, 32: np.uint32}[bps]

    def _decode(self, off, cnt, comp, pred, dt, w, h):
        self.fh.seek(off)
        raw = self.fh.read(cnt)
        if comp == 1:
            pass
        elif comp in (8, 32946):
            raw = zlib.decompress(raw)
        elif comp == 5:
            raw = _lzw(raw)
        else:
            raise NotImplementedError(f'unsupported compression {comp}')
        need = w * h * np.dtype(dt).itemsize
        if len(raw) < need:
            raw = raw + b'\x00' * (need - len(raw))
        a = np.frombuffer(raw[:need], dtype=dt)
        if self.e == '>':
            a = a.byteswap().view(a.dtype.newbyteorder())
        a = a.copy()
        if pred == 2:
            a = a.reshape(h, w)
            np.cumsum(a, axis=1, dtype=a.dtype, out=a)
            a = a.ravel()
        elif pred == 3:
            a = a.reshape(h, w)
            # floating-point predictor: byte-level reordering
            b = a.view(np.uint8).reshape(h, -1)
            np.cumsum(b, axis=1, dtype=np.uint8, out=b)
            n = np.dtype(dt).itemsize
            b = b.reshape(h, n, w).transpose(0, 2, 1).copy()
            a = b.view(dt).reshape(h, w).ravel()
        return a


def _lzw(data):
    """TIFF LZW decoding (MSB-first, early change)."""
    out = bytearray()
    dic = [bytes([i]) for i in range(256)] + [b'', b'']
    bitpos, nbits, prev = 0, 9, None
    total = len(data) * 8
    while bitpos + nbits <= total:
        byte = bitpos >> 3
        chunk = data[byte:byte + 3]
        if len(chunk) < 3:
            chunk = chunk + b'\x00' * (3 - len(chunk))
        v = (chunk[0] << 16) | (chunk[1] << 8) | chunk[2]
        code = (v >> (24 - (bitpos & 7) - nbits)) & ((1 << nbits) - 1)
        bitpos += nbits
        if code == 256:
            dic = dic[:258]
            nbits, prev = 9, None
            continue
        if code == 257:
            break
        if prev is None:
            entry = dic[code]
        elif code < len(dic):
            entry = dic[code]
            dic.append(prev + entry[:1])
        else:
            entry = prev + prev[:1]
            dic.append(entry)
        out += entry
        prev = entry
        if len(dic) + 1 >= (1 << nbits) and nbits < 12:
            nbits += 1
    return bytes(out)


def tif_meta(path):
    with TiffFile(path) as t:
        return t.meta()


def tif_read(path, window=None):
    with TiffFile(path) as t:
        return t.read(window=window), t.meta()


# ---------------------------------------------------------------- projection

# GRS80 (NAD83)
_A = 6378137.0
_F = 1 / 298.257222101
_E2 = _F * (2 - _F)


def ll_to_utm(lon, lat, zone=17, north=True):
    """NAD83/GRS80 lon/lat -> UTM (metres). Input may be scalar or ndarray."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    k0 = 0.9996
    lon0 = math.radians(zone * 6 - 183)
    phi = np.radians(lat)
    lam = np.radians(lon)
    e2 = _E2
    ep2 = e2 / (1 - e2)
    N = _A / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T = np.tan(phi) ** 2
    C = ep2 * np.cos(phi) ** 2
    A = np.cos(phi) * (lam - lon0)
    e4, e6 = e2 * e2, e2 ** 3
    M = _A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
              - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
              + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
              - (35 * e6 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                  + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A ** 2 / 2
              + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
              + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    if not north:
        y = y + 10000000.0
    return x, y


def utm_to_ll(x, y, zone=17, north=True):
    """UTM (metres) -> NAD83/GRS80 lon/lat."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k0 = 0.9996
    lon0 = math.radians(zone * 6 - 183)
    e2 = _E2
    ep2 = e2 / (1 - e2)
    xx = x - 500000.0
    yy = y - (0.0 if north else 10000000.0)
    M = yy / k0
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    mu = M / (_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    C1 = ep2 * np.cos(phi1) ** 2
    T1 = np.tan(phi1) ** 2
    N1 = _A / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
    R1 = _A * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
    D = xx / (N1 * k0)
    lat = phi1 - (N1 * np.tan(phi1) / R1) * (
        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon = lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2)
                  * D ** 5 / 120) / np.cos(phi1)
    return np.degrees(lon), np.degrees(lat)


# ---------------------------------------------------------------- Shapefile

def read_shapefile(shp_path):
    """Read an ESRI Shapefile. Returns (geoms, records, fields, prj_text).
    geoms: each feature is [[(x,y),...], ...] (multiple rings/parts).
    """
    base = os.path.splitext(shp_path)[0]
    geoms, shapes = [], []
    with open(shp_path, 'rb') as f:
        f.seek(100)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            num, clen = struct.unpack('>ii', hdr)
            body = f.read(clen * 2)
            shapes.append(_parse_shape(body))
    records, fields = _read_dbf(base + '.dbf')
    prj = ''
    if os.path.exists(base + '.prj'):
        prj = open(base + '.prj', 'r', errors='replace').read()
    return shapes, records, fields, prj


def _parse_shape(b):
    if len(b) < 4:
        return None
    (typ,) = struct.unpack('<i', b[:4])
    if typ == 0:
        return None
    if typ == 1:  # Point
        x, y = struct.unpack('<dd', b[4:20])
        return {'type': 'Point', 'parts': [[(x, y)]],
                'bbox': (x, y, x, y)}
    if typ in (3, 5, 13, 15, 23, 25):  # PolyLine / Polygon (+Z/M)
        bbox = struct.unpack('<4d', b[4:36])
        nparts, npoints = struct.unpack('<ii', b[36:44])
        parts = struct.unpack('<%di' % nparts, b[44:44 + 4 * nparts])
        off = 44 + 4 * nparts
        pts = np.frombuffer(b[off:off + 16 * npoints],
                            dtype='<f8').reshape(npoints, 2)
        rings = []
        for i in range(nparts):
            a = parts[i]
            z = parts[i + 1] if i + 1 < nparts else npoints
            rings.append([tuple(p) for p in pts[a:z]])
        kind = 'Polygon' if typ in (5, 15, 25) else 'PolyLine'
        return {'type': kind, 'parts': rings, 'bbox': bbox}
    return None


def _read_dbf(path):
    with open(path, 'rb') as f:
        hdr = f.read(32)
        nrec, hlen, rlen = struct.unpack('<IHH', hdr[4:12])
        nf = (hlen - 33) // 32
        fields = []
        for _ in range(nf):
            fd = f.read(32)
            name = fd[:11].split(b'\x00')[0].decode('latin-1')
            ftyp = fd[11:12].decode('latin-1')
            flen = fd[16]
            fdec = fd[17]
            fields.append((name, ftyp, flen, fdec))
        f.seek(hlen)
        recs = []
        for _ in range(nrec):
            raw = f.read(rlen)
            if not raw or raw[:1] == b'\x1a':
                break
            pos, rec = 1, {}
            for name, ftyp, flen, fdec in fields:
                v = raw[pos:pos + flen].decode('latin-1').strip()
                pos += flen
                if ftyp in 'NF':
                    try:
                        v = float(v) if fdec else (int(v) if v else None)
                    except ValueError:
                        v = None
                rec[name] = v
            recs.append(rec)
    return recs, fields


# ---------------------------------------------------------------- GeoTIFF writing

def write_geotiff(path, arr, transform, epsg, nodata=None,
                  citation='', compress=True):
    """Write a single-band GeoTIFF (Deflate). transform=(x0,dx,0,y0,0,dy)."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    dt = arr.dtype
    if dt == np.float32:
        bps, sf = 32, 3
    elif dt == np.float64:
        bps, sf = 64, 3
    elif dt == np.int16:
        bps, sf = 16, 2
    elif dt == np.uint8:
        bps, sf = 8, 1
    elif dt == np.int32:
        bps, sf = 32, 2
    else:
        arr = arr.astype(np.float32)
        dt, bps, sf = np.float32, 32, 3

    rows_per_strip = max(1, min(h, int(4 * 1024 * 1024 / (w * bps // 8))))
    strips, offs, cnts = [], [], []
    for s in range(0, h, rows_per_strip):
        buf = arr[s:s + rows_per_strip].tobytes()
        strips.append(zlib.compress(buf, 6) if compress else buf)
    comp = 8 if compress else 1

    x0, dx, _, y0, _, dy = transform
    pixscale = struct.pack('<3d', abs(dx), abs(dy), 0.0)
    tiepoint = struct.pack('<6d', 0.0, 0.0, 0.0, x0, y0, 0.0)

    gk = [1, 1, 0, 0]   # version header; key count filled in later
    dbl, asc = [], []

    def addkey(kid, val):
        gk.extend([kid, 0, 1, val])

    def addasc(kid, s):
        nonlocal asc
        pos = sum(len(a.encode('utf-8', 'replace')) for a in asc)
        asc.append(s + '|')
        gk.extend([kid, 34737, len(s.encode('utf-8', 'replace')) + 1, pos])

    addkey(1024, 1)          # ModelType = projected
    addkey(1025, 1)          # RasterType = PixelIsArea
    addkey(3072, epsg)       # ProjectedCSType
    addkey(3076, 9001)       # LinearUnits = metre
    if citation:
        addasc(1026, citation)
    gk[3] = (len(gk) - 4) // 4
    geokey_bytes = struct.pack('<%dH' % len(gk), *gk)
    ascii_bytes = ''.join(asc).encode('utf-8', 'replace')

    entries = []   # (tag, type, count, payload_bytes_or_None, inline_value)
    extra = bytearray()
    HEADER = 8

    def plan():
        # two passes: estimate lengths first, then fix offsets
        pass

    # assemble: compute fixed block sizes first
    tags = []

    def add(tag, typ, count, data=None, inline=None):
        tags.append([tag, typ, count, data, inline])

    add(256, 4, 1, inline=w)
    add(257, 4, 1, inline=h)
    add(258, 3, 1, inline=bps)
    add(259, 3, 1, inline=comp)
    add(262, 3, 1, inline=1)
    add(273, 4, len(strips), data=b'')      # StripOffsets, filled in later
    add(277, 3, 1, inline=1)
    add(278, 4, 1, inline=rows_per_strip)
    add(279, 4, len(strips),
        data=struct.pack('<%dI' % len(strips), *[len(s) for s in strips]))
    add(284, 3, 1, inline=1)
    add(339, 3, 1, inline=sf)
    add(33550, 12, 3, data=pixscale)
    add(33922, 12, 6, data=tiepoint)
    add(34735, 3, len(gk), data=geokey_bytes)
    if ascii_bytes:
        add(34737, 2, len(ascii_bytes), data=ascii_bytes)
    if nodata is not None:
        nd = ('%r' % float(nodata)).encode('ascii') + b'\x00'
        add(42113, 2, len(nd), data=nd)

    tags.sort(key=lambda t: t[0])
    ifd_size = 2 + 12 * len(tags) + 4
    cursor = HEADER + ifd_size
    # first assign offsets to non-strip external data
    for t in tags:
        tag, typ, count, data, inline = t
        if data is None:
            continue
        if tag == 273:
            continue
        if len(data) <= 4:
            t[4] = int.from_bytes(data.ljust(4, b'\x00'), 'little')
            t[3] = None
        else:
            t[4] = cursor
            cursor += len(data) + (len(data) & 1)
    # the StripOffsets array itself
    so_arr_off = cursor
    cursor += 4 * len(strips) + (4 * len(strips) & 1)
    strip_offsets = []
    for s in strips:
        strip_offsets.append(cursor)
        cursor += len(s)
    so_bytes = struct.pack('<%dI' % len(strips), *strip_offsets)
    for t in tags:
        if t[0] == 273:
            if len(so_bytes) <= 4:
                t[4] = int.from_bytes(so_bytes.ljust(4, b'\x00'), 'little')
                t[3] = None
            else:
                t[3] = so_bytes
                t[4] = so_arr_off

    with open(path, 'wb') as f:
        f.write(b'II' + struct.pack('<HI', 42, HEADER))
        f.write(struct.pack('<H', len(tags)))
        for tag, typ, count, data, inline in tags:
            f.write(struct.pack('<HHI', tag, typ, count))
            if data is None:
                f.write(struct.pack('<I', inline))
            else:
                f.write(struct.pack('<I', inline))
        f.write(struct.pack('<I', 0))
        for tag, typ, count, data, inline in tags:
            if data is None or tag == 273:
                continue
            f.write(data)
            if len(data) & 1:
                f.write(b'\x00')
        f.write(so_bytes)
        if len(so_bytes) & 1:
            f.write(b'\x00')
        for s in strips:
            f.write(s)
    return path


# ---------------------------------------------------------------- GeoPackage

def _wkb_point(x, y):
    return struct.pack('<BIdd', 1, 1, x, y)


def _wkb_linestring(pts):
    b = struct.pack('<BII', 1, 2, len(pts))
    return b + b''.join(struct.pack('<dd', *p) for p in pts)


def _wkb_polygon(rings):
    b = struct.pack('<BII', 1, 3, len(rings))
    for r in rings:
        b += struct.pack('<I', len(r))
        b += b''.join(struct.pack('<dd', *p) for p in r)
    return b


def _gpkg_blob(wkb, srs_id, bbox=None):
    flags = 0b00000001  # little endian
    if bbox:
        flags |= 0b00000010
    hdr = b'GP' + bytes([0, flags]) + struct.pack('<i', srs_id)
    if bbox:
        hdr += struct.pack('<4d', bbox[0], bbox[2], bbox[1], bbox[3])
    return hdr + wkb


class GeoPackage:
    """Minimal GeoPackage writer (points/lines/polygons, attribute table)."""

    def __init__(self, path, srs_id=26917, srs_name='NAD83 / UTM zone 17N',
                 srs_wkt=''):
        if os.path.exists(path):
            os.remove(path)
        self.path = path
        self.srs_id = srs_id
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA application_id = 1196444487;')
        self.db.execute('PRAGMA user_version = 10200;')
        c = self.db.cursor()
        c.executescript("""
        CREATE TABLE gpkg_spatial_ref_sys (
          srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
          organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
          definition TEXT NOT NULL, description TEXT);
        CREATE TABLE gpkg_contents (
          table_name TEXT PRIMARY KEY, data_type TEXT NOT NULL,
          identifier TEXT UNIQUE, description TEXT DEFAULT '',
          last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
          srs_id INTEGER REFERENCES gpkg_spatial_ref_sys(srs_id));
        CREATE TABLE gpkg_geometry_columns (
          table_name TEXT NOT NULL, column_name TEXT NOT NULL,
          geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
          z TINYINT NOT NULL, m TINYINT NOT NULL,
          CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name));
        """)
        c.executemany(
            'INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)',
            [('Undefined cartesian SRS', -1, 'NONE', -1, 'undefined', ''),
             ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', ''),
             ('WGS 84 geodetic', 4326, 'EPSG', 4326,
              'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
              'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', ''),
             (srs_name, srs_id, 'EPSG', srs_id,
              srs_wkt or _UTM17N_WKT, '')])
        self.db.commit()

    def add_layer(self, name, geom_type, features, fields, description=''):
        """features: [(wkb_bytes, bbox, {field: value}), ...]
           fields: [(name, sqltype), ...]"""
        cols = ', '.join(f'"{n}" {t}' for n, t in fields)
        c = self.db.cursor()
        c.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, '
                  f'geom BLOB{", " + cols if cols else ""})')
        minx = miny = 1e30
        maxx = maxy = -1e30
        for wkb, bbox, attrs in features:
            blob = _gpkg_blob(wkb, self.srs_id, bbox)
            names = ['geom'] + [n for n, _ in fields]
            vals = [blob] + [attrs.get(n) for n, _ in fields]
            q = ', '.join(f'"{n}"' for n in names)
            p = ', '.join('?' * len(names))
            c.execute(f'INSERT INTO "{name}" ({q}) VALUES ({p})', vals)
            if bbox:
                minx, miny = min(minx, bbox[0]), min(miny, bbox[1])
                maxx, maxy = max(maxx, bbox[2]), max(maxy, bbox[3])
        c.execute('INSERT INTO gpkg_contents (table_name, data_type, identifier,'
                  ' description, min_x, min_y, max_x, max_y, srs_id)'
                  ' VALUES (?,?,?,?,?,?,?,?,?)',
                  (name, 'features', name, description,
                   minx, miny, maxx, maxy, self.srs_id))
        c.execute('INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)',
                  (name, 'geom', geom_type, self.srs_id, 0, 0))
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()


_UTM17N_WKT = (
    'PROJCS["NAD83 / UTM zone 17N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
    'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],'
    'AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-81],'
    'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26917"]]')


# ---------------------------------------------------------------- NetCDF-3 writing

class NC3Writer:
    """Minimal NetCDF-3 classic / 64-bit offset writer."""
    _TYPES = {np.dtype('int8'): 1, np.dtype('int16'): 3, np.dtype('int32'): 4,
              np.dtype('float32'): 5, np.dtype('float64'): 6}
    _SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}

    def __init__(self, path):
        self.path = path
        self.dims = []        # (name, size)  size=0 -> unlimited
        self.gattrs = []
        self.vars = []        # dict(name, dims, dtype, attrs, data)

    def add_dim(self, name, size):
        self.dims.append((name, size))

    def add_gattr(self, name, value):
        self.gattrs.append((name, value))

    def add_var(self, name, dims, data, attrs=None):
        data = np.ascontiguousarray(data)
        self.vars.append({'name': name, 'dims': list(dims), 'data': data,
                          'attrs': list((attrs or {}).items())})

    # -- encoding primitives
    @staticmethod
    def _pad(b):
        return b + b'\x00' * ((4 - len(b) % 4) % 4)

    def _name(self, s):
        b = s.encode('utf-8')
        return struct.pack('>i', len(b)) + self._pad(b)

    def _attr(self, name, val):
        out = self._name(name)
        if isinstance(val, str):
            b = val.encode('utf-8')
            out += struct.pack('>ii', 2, len(b)) + self._pad(b)
        else:
            a = np.asarray(val)
            if a.dtype.kind == 'i':
                a = a.astype('>i4'); t = 4
            else:
                a = a.astype('>f8'); t = 6
            a = np.atleast_1d(a)
            out += struct.pack('>ii', t, a.size) + self._pad(a.tobytes())
        return out

    def _attrs(self, items):
        if not items:
            return struct.pack('>ii', 0, 0)
        out = struct.pack('>ii', 12, len(items))
        for k, v in items:
            out += self._attr(k, v)
        return out

    def write(self):
        dimmap = {n: i for i, (n, _) in enumerate(self.dims)}
        hdr = b'CDF\x02'                       # 64-bit offset
        hdr += struct.pack('>i', 0)            # numrecs = 0
        hdr += struct.pack('>ii', 10, len(self.dims))
        for n, s in self.dims:
            hdr += self._name(n) + struct.pack('>i', s)
        hdr += self._attrs(self.gattrs)
        hdr += struct.pack('>ii', 11, len(self.vars))
        # two passes are needed to fix the begin offsets
        var_hdrs = []
        for v in self.vars:
            nt = self._TYPES[v['data'].dtype]
            vsize = int(v['data'].nbytes)
            vsize += (4 - vsize % 4) % 4
            b = self._name(v['name'])
            b += struct.pack('>i', len(v['dims']))
            for d in v['dims']:
                b += struct.pack('>i', dimmap[d])
            b += self._attrs(v['attrs'])
            b += struct.pack('>ii', nt, vsize)
            var_hdrs.append([b, vsize, nt])
        base = len(hdr) + sum(len(b) + 8 for b, _, _ in var_hdrs)
        off = base
        with open(self.path, 'wb') as f:
            f.write(hdr)
            for vh in var_hdrs:
                f.write(vh[0] + struct.pack('>q', off))
                off += vh[1]
            for v, vh in zip(self.vars, var_hdrs):
                a = v['data']
                a = a.astype(a.dtype.newbyteorder('>'))
                b = a.tobytes()
                f.write(b + b'\x00' * (vh[1] - len(b)))
        return self.path


# ---------------------------------------------------------------- GeoJSON writing

class GeoJSONWriter:
    """Same add_layer API as the GeoPackage class, but writes GeoJSON.

    Why GeoJSON: GDAL could not be installed in the original session, so there was no way to verify that the hand-written
    GeoPackage binary can be read by QGIS (in practice it could not). GeoJSON is plain text, natively readable and editable
    in QGIS, and cannot be corrupted by a bad binary write.

    Coordinates are written as EPSG:4326 lon/lat per the GeoJSON spec; QGIS reprojects them to the project CRS automatically.
    """

    def __init__(self, outdir, srs_id=26917, zone=17):
        self.outdir = outdir
        self.srs_id = srs_id
        self.zone = zone
        os.makedirs(outdir, exist_ok=True)
        self.written = []

    def _ll(self, xs, ys):
        lo, la = utm_to_ll(np.asarray(xs, float), np.asarray(ys, float), self.zone)
        lo = np.atleast_1d(lo); la = np.atleast_1d(la)
        return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lo, la)]

    def _geom(self, wkb):
        bo = '<' if wkb[0] == 1 else '>'
        gt = struct.unpack(bo + 'I', wkb[1:5])[0] % 1000
        if gt == 1:
            x, y = struct.unpack(bo + 'dd', wkb[5:21])
            return {'type': 'Point', 'coordinates': self._ll([x], [y])[0]}
        if gt == 2:
            n = struct.unpack(bo + 'I', wkb[5:9])[0]
            pts = np.frombuffer(wkb[9:9 + 16 * n], dtype=bo + 'f8').reshape(n, 2)
            return {'type': 'LineString',
                    'coordinates': self._ll(pts[:, 0], pts[:, 1])}
        if gt == 3:
            nr = struct.unpack(bo + 'I', wkb[5:9])[0]
            pos, rings = 9, []
            for _ in range(nr):
                n = struct.unpack(bo + 'I', wkb[pos:pos + 4])[0]
                pos += 4
                pts = np.frombuffer(wkb[pos:pos + 16 * n],
                                    dtype=bo + 'f8').reshape(n, 2)
                pos += 16 * n
                ring = self._ll(pts[:, 0], pts[:, 1])
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)
            return {'type': 'Polygon', 'coordinates': rings}
        raise ValueError(f'unsupported geometry type {gt}')

    def add_layer(self, name, geom_type, features, fields, description=''):
        import json as _json
        feats = []
        for wkb, bbox, attrs in features:
            props = {}
            for k, _t in fields:
                v = attrs.get(k)
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                props[k] = v
            feats.append({'type': 'Feature', 'geometry': self._geom(wkb),
                          'properties': props})
        path = os.path.join(self.outdir, name + '.geojson')
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump({'type': 'FeatureCollection', 'name': name,
                        'features': feats}, f, ensure_ascii=False, indent=1)
        self.written.append((path, len(feats), description))
        return path

    def close(self):
        for path, n, d in self.written:
            print(f'  {os.path.basename(path):42s} {n:4d} features  {d}')
