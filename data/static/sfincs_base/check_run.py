#!/usr/bin/env python3
"""
check_run.py -- three quick checks after a SFINCS run (run inside the sfincs_200m/ directory)
Requires: pip install xarray netCDF4 matplotlib numpy
Outputs: check_his_mayport.png, check_zsmax.png, and conclusions printed to the terminal
"""
import os, sys, re
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# ---------- 1. log ----------
print('=' * 60); print('1. sfincs.log')
if not os.path.exists('sfincs.log'):
    print('   no sfincs.log -- SFINCS has not run yet, or was not run in this directory'); sys.exit(1)
log = open('sfincs.log', errors='replace').read()
bad = [l for l in log.splitlines() if re.search(r'error|cannot|fail|not found', l, re.I)]
print('   lines %d; lines containing error/cannot/fail %d' % (log.count('\n'), len(bad)))
for l in bad[:10]:
    print('     !', l.strip())
for key in ('mmax', 'nmax', 'bndfile', 'srcfile', 'amprfile', 'obsfile'):
    m = [l for l in log.splitlines() if key in l]
    print('   %-9s %s' % (key, m[0].strip()[:90] if m else '(not mentioned in log)'))

# ---------- 2. his: Mayport vs bzs ----------
print('=' * 60); print('2. sfincs_his.nc -- Mayport observation point vs input bzs')
try:
    import xarray as xr, matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
except ImportError:
    print('   requires pip install xarray netCDF4 matplotlib'); sys.exit(1)
if not os.path.exists('sfincs_his.nc'):
    print('   no sfincs_his.nc'); sys.exit(1)
his = xr.open_dataset('sfincs_his.nc')
names = [str(s.values).strip().strip("'").strip('b').strip("'") for s in his['station_name']] \
    if 'station_name' in his else [str(i) for i in range(his.sizes.get('stations', 0))]
try:
    k = [i for i, nm in enumerate(names) if '8720218' in nm][0]
except IndexError:
    k = 1
    print('   no station with 8720218 in its name found in his; using station 2 instead; station names:', names)
zs = his['point_zs'][:, k].values if 'point_zs' in his else his['zs'][:, k].values
t_his = (his['time'].values - his['time'].values[0]) / np.timedelta64(1, 'h')
bzs = np.loadtxt('sfincs.bzs'); t_b, h_b = bzs[:, 0] / 3600., bzs[:, 1]
hb_on_his = np.interp(t_his, t_b, h_b)
d = zs - hb_on_his
print('   simulated - input water level: mean %.3f m, RMS %.3f m, max |diff| %.3f m' % (np.nanmean(d), np.sqrt(np.nanmean(d ** 2)), np.nanmax(np.abs(d))))
print('   interpretation: RMS < 0.05 is normal; constant offset -> datum/zsini; no variation at all -> boundary cells not recognized')
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t_b, h_b, 'k', lw=1, label='sfincs.bzs input (Mayport observed)')
ax.plot(t_his, zs, 'r', lw=1, alpha=0.8, label='sfincs_his.nc simulated (%s)' % names[k])
ax.set_xlabel('hours since tref'); ax.set_ylabel('m NAVD88'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('check_his_mayport.png', dpi=140); print('   -> check_his_mayport.png')

# ---------- 3. map: zsmax ----------
print('=' * 60); print('3. sfincs_map.nc -- max water level field')
if not os.path.exists('sfincs_map.nc'):
    print('   no sfincs_map.nc'); sys.exit(0)
mp = xr.open_dataset('sfincs_map.nc')
var = 'zsmax' if 'zsmax' in mp else ('zs' if 'zs' in mp else None)
if var is None:
    print('   no zsmax/zs in map; variables:', list(mp.data_vars)); sys.exit(0)
A = mp[var]
if 'time' in A.dims or 'timemax' in A.dims:
    A = A.max(dim=[d for d in A.dims if 'time' in d])
A = A.values
wet = np.isfinite(A)
zb = mp['zb'].values if 'zb' in mp else None
if zb is not None:
    depth = A - zb
    print('   active cells %d; max water level %.2f … %.2f m; cells with max depth > 0.1 m %d (%.1f%%)'
          % (wet.sum(), np.nanmin(A), np.nanmax(A), (depth > 0.1).sum(), 100 * (depth > 0.1).sum() / max(wet.sum(), 1)))
    print('   interpretation: flooded fraction 10%–40%, concentrated along river banks -> normal; ≈100% or ≈0% -> dep/msk orientation flipped')
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(np.ma.masked_invalid(A), origin='lower', cmap='Blues')
fig.colorbar(im, ax=ax, label='%s (m)' % var); ax.set_title('SFINCS %s' % var)
fig.tight_layout(); fig.savefig('check_zsmax.png', dpi=140); print('   -> check_zsmax.png')
