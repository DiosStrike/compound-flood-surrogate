#!/usr/bin/env python3
"""15_boundary_check.py -- final boundary check figure (river-mouth extension + new tributary cut + remaining wall openings). Plotting only."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as PTH
import geoio
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

NEW = PTH.ROOT
GIS, INT = PTH.STATIC_GIS, PTH.STATIC_INTERIM
META, FIG = PTH.STATIC_META, PTH.FIG
DX, ZONE = 60.0, 17

R = json.load(open(os.path.join(META, 'study_extent.json')))['rect_aligned']
BD = json.load(open(os.path.join(META, 'boundaries.json')))
z = np.load(os.path.join(INT, 'merged60.npy'))
ACT = np.load(os.path.join(INT, 'active_mask.npy'))
UNC = np.load(os.path.join(INT, 'uncovered_edge.npy'))
NY, NX = z.shape
EXTENT = [R['xmin'], R['xmax'], R['ymin'], R['ymax']]


def gj(path):
    d = json.load(open(path)); out = []
    for f in d['features']:
        g = f['geometry']
        if not g or not g.get('coordinates'):
            continue
        segs = [g['coordinates']] if g['type'] == 'LineString' else g['coordinates']
        for s in segs:
            a = np.asarray(s, float)
            x, y = geoio.ll_to_utm(a[:, 0], a[:, 1], ZONE)
            out.append((np.c_[x, y], f['properties']))
    return out


DW = gj(os.path.join(GIS, 'bnd_downstream_waterlevel.geojson'))
UP = gj(os.path.join(GIS, 'bnd_upstream_discharge.geojson'))
sta = {}
for f in json.load(open(os.path.join(GIS, 'stations_forcing.geojson')))['features']:
    lon, lat = f['geometry']['coordinates']
    sta[f['properties']['station_id']] = [float(v) for v in geoio.ll_to_utm(lon, lat, ZONE)]

jj, ii = np.where(UNC)
UX = R['xmin'] + (ii + .5) * DX
UY = R['ymax'] - (jj + .5) * DX

cmap = LinearSegmentedColormap.from_list('tb', [
    (0.00, '#08306b'), (0.22, '#2171b5'), (0.38, '#6baed6'), (0.44, '#c6dbef'),
    (0.4501, '#e8f6d0'), (0.55, '#a1d99b'), (0.70, '#d9c27a'),
    (0.85, '#b07b52'), (1.00, '#f2f2f2')])
cmap.set_bad('#d9d9d9')
norm = TwoSlopeNorm(vmin=-25., vcenter=0., vmax=31.)
OLD_N = np.array([461670.0, 3363630.0])     # north end of DW00 before the extension


def draw(ax, cx, cy, rr, ms=1.0):
    ax.imshow(np.ma.masked_invalid(z), extent=EXTENT, origin='upper', cmap=cmap,
              norm=norm, interpolation='nearest')
    ax.imshow(np.ma.masked_where(ACT, np.ones_like(z)), extent=EXTENT, origin='upper',
              cmap=ListedColormap(['white']), alpha=0.62, interpolation='nearest', zorder=2)
    X = R['xmin'] + (np.arange(NX) + .5) * DX
    Y = R['ymax'] - (np.arange(NY) + .5) * DX
    ax.contour(X, Y, ACT.astype(float), levels=[.5], colors=['#222'],
               linewidths=1.4, zorder=7)
    for g, p in DW:
        ax.plot(g[:, 0], g[:, 1], color='#0057b7', lw=4.6 * ms, solid_capstyle='round',
                zorder=10)
    for g, p in UP:
        ax.plot(g[:, 0], g[:, 1], color='#e6194B', lw=5.2 * ms, solid_capstyle='round',
                zorder=11)
    ax.scatter(UX, UY, s=130 * ms, marker='s', facecolors='none', edgecolors='#ff8c00',
               lw=2.2 * ms, zorder=14)
    for sid, mk, c in (('02246500', '^', '#e6194B'), ('8720218', 'o', '#0057b7')):
        if sid in sta:
            ax.plot(*sta[sid], marker=mk, ms=13 * ms, mfc=c, mec='k', mew=1.4, zorder=16)
    ax.set_xlim(cx - rr, cx + rr); ax.set_ylim(cy - rr * 0.80, cy + rr * 0.80)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])


fig = plt.figure(figsize=(14.2, 8.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.75, 1, 1], wspace=0.10, hspace=0.20,
                      left=0.015, right=0.955, top=0.885, bottom=0.115)

ax = fig.add_subplot(gs[:, 0])
draw(ax, (R['xmin'] + R['xmax']) / 2, (R['ymin'] + R['ymax']) / 2 + 3000, 25500, 0.75)
dw0 = BD['boundaries']['downstream']['features'][0]
dw1 = BD['boundaries']['downstream']['features'][1]
up0 = BD['boundaries']['upstream']['features'][0]
ax.set_title('Model domain %.0f km^2 -- 1 upstream segment + 2 downstream segments' % BD['active_domain']['km2'],
             fontsize=12)
cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.035, pad=0.012, extend='both')
cb.set_label('Elevation (m, NAVD88)', fontsize=9.5)

zooms = [
    (460700, 3363500, 2500, '(1) River mouth: DW00 extended to the NW\ncovers the former group-A 11 cells', '#0057b7'),
    (456230, 3367160, 1500, '(2) New DW01 tributary cut\nwidth 360 m, min -3.58 m', '#0057b7'),
    (439100, 3353700, 1900, '(3) Remaining wall: second Acosta channel\n6 cells / 1390 m^2', '#ff8c00'),
    (461220, 3338900, 1500, '(4) Remaining wall: small southern bay\n2 cells / 291 m^2', '#ff8c00'),
]
for k, (cx, cy, rr, ttl, col) in enumerate(zooms):
    a = fig.add_subplot(gs[k // 2, 1 + k % 2])
    draw(a, cx, cy, rr, 1.0)
    if k == 0:
        a.plot(*OLD_N, marker='*', ms=19, mfc='white', mec='#0057b7', mew=2.2, zorder=18)
        a.annotate('End before extension', OLD_N, xytext=(12, -34), textcoords='offset points',
                   fontsize=8.6, color='#0057b7', fontweight='bold', zorder=19,
                   arrowprops=dict(arrowstyle='->', color='#0057b7', lw=1.2),
                   bbox=dict(fc='white', alpha=0.92, ec='#0057b7', boxstyle='round,pad=0.26'))
    a.set_title(ttl, fontsize=9.4, color=col)
    for sp in a.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(1.8)

h = [Line2D([], [], color='#e6194B', lw=5, label='Upstream discharge boundary UP01 (Acosta Q(t), 100%)'),
     Line2D([], [], color='#0057b7', lw=4.6, label='Downstream water-level boundary DW00 + DW01 (Mayport h(t))'),
     Line2D([], [], color='#222', lw=1.4, label='Model domain boundary (uncovered = wall)'),
     Line2D([], [], ls='', marker='s', mfc='none', mec='#ff8c00', mew=2.2, ms=10,
            label='Remaining wet cells without boundary (8 cells / 2 sites, set to wall)'),
     Line2D([], [], ls='', marker='^', ms=11, mfc='#e6194B', mec='k', label='USGS 02246500 Acosta'),
     Line2D([], [], ls='', marker='o', ms=10, mfc='#0057b7', mec='k', label='NOAA 8720218 Mayport')]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=9.2, frameon=True,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle('Final model boundary check (2026-09-14 user QGIS version)', fontsize=14.5, y=0.965)
fig.text(0.5, 0.082,
         'DW00 %.2f km (section %.0f m^2) + DW01 %.0f m (section %.0f m^2) + UP01 %.0f m (section %.0f m^2); '
         'of %d wet edge cells %d are covered, remaining %d (%.0f m^2, %.1f%% of downstream section) set to wall.'
         % (dw0['length_m'] / 1000, dw0['flow_area_m2'], dw1['length_m'], dw1['flow_area_m2'],
            up0['length_m'], up0['flow_area_m2'], BD['edge_cells']['wet'],
            BD['edge_cells']['covered'], BD['edge_cells']['uncovered'],
            sum(c['flow_area_m2'] for c in BD['uncovered_clusters']),
            100 * sum(c['flow_area_m2'] for c in BD['uncovered_clusters']) / dw0['flow_area_m2']),
         ha='center', fontsize=9.5)
p = os.path.join(FIG, 'fig_model_boundary_final_check.png')
fig.savefig(p, dpi=175)
print(p)
