"""One-off check: whether the NaN/fill in zs of map.nc marks "dry cells" (h is 0 or equally filled), and whether h agrees with zs-zb."""
import sys, numpy as np, netCDF4 as nc
sys.path.insert(0, 'code'); from read_run import run_dir
for ev, i in (('beryl', 3), ('irma', 41)):
    d = nc.Dataset(run_dir(ev, '%s_s%03d' % (ev, i)) + '/sfincs_map.nc')
    msk = np.ma.filled(d.variables['msk'][:], 0); act = msk > 0; zb = np.ma.filled(d.variables['zb'][:], np.nan)
    print(ev, i, 'zs _FillValue', getattr(d.variables['zs'], '_FillValue', None), 'h _FillValue', getattr(d.variables['h'], '_FillValue', None), 'h attrs', {a: d.variables['h'].getncattr(a) for a in d.variables['h'].ncattrs()})
    for k in (0, 100, 240):
        zs = np.ma.filled(d.variables['zs'][k], np.nan); h = np.ma.filled(d.variables['h'][k], np.nan)
        zsnan = np.isnan(zs) & act; hnan = np.isnan(h) & act
        print('  t=%dh: zs NaN active cells %d, h NaN active cells %d, co-located %d; min h where zs valid %.4f; cells with h valid but zs NaN %d, their max h %.4f; max |h-(zs-zb)| where valid %.5f' % (
            k, zsnan.sum(), hnan.sum(), (zsnan & hnan).sum(), np.nanmin(h[~zsnan & act]) if (~zsnan & act).any() else -1, (zsnan & ~hnan).sum(), np.nanmax(h[zsnan & ~hnan]) if (zsnan & ~hnan).any() else -1,
            np.nanmax(np.abs(h - (zs - zb))[~zsnan & ~hnan & act])))
