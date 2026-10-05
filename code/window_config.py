#!/usr/bin/env python3
"""
window_config.py -- single source of truth for the simulation time window.

2026-09-14 the user chose option WB (3 days later than the original option):
    full window  2017-09-06 00:00 -> 2017-09-16 00:00 UTC, 240 hourly labels
    spin-up      09-06 01:00 .. 09-09 00:00   (72 labels)
    event period 09-09 01:00 .. 09-16 00:00   (168 labels)

Rationale (see the window-option comparison PDF in result/reports/, produced by code/14_window_options.py):
    The original window 09-03 -> 09-13 cut off before the heavy rain and flood peak had ended -- the last 48 h still had 206.8 mm of rain,
    and only 22 h remained after the discharge peak (6400 m³/s @ 09-12 02:00). WB fully contains the storm and leaves
    a 94 h recession period; the flood peak falls at 61% of the window, and spin-up rainfall is only 6 mm (quiet enough).

To change the window, only edit this file, then rerun in order 04_hydro_series.py, 05_mrms_rainfall.py,
07_figures.py, 13_report_v2.py, 09_manifest.py.
"""
import datetime

# ---- Window definition (UTC) ----
T0_STR = '2017-09-06 00:00'      # window start (exclusive; the first label is T0+1h)
SPIN_HOURS = 72
EVENT_HOURS = 168
N_HOURS = SPIN_HOURS + EVENT_HOURS          # 240

_T0 = datetime.datetime(2017, 9, 6, 0)
T0_DT = _T0
TSPIN_DT = _T0 + datetime.timedelta(hours=SPIN_HOURS)
T1_DT = _T0 + datetime.timedelta(hours=N_HOURS)

T0_ISO = T0_DT.strftime('%Y-%m-%d %H:%M')
TSPIN_ISO = TSPIN_DT.strftime('%Y-%m-%d %H:%M')
T1_ISO = T1_DT.strftime('%Y-%m-%d %H:%M')

# 240 hourly labels; label T represents the interval [T-1h, T)
LABELS_DT = [_T0 + datetime.timedelta(hours=h) for h in range(1, N_HOURS + 1)]

VERSION = 'WB'
NOTE = ('window %s -> %s UTC; spin-up until %s; 240 hourly labels, label T represents [T-1h, T)'
        % (T0_ISO, T1_ISO, TSPIN_ISO))


def pd_window(pd):
    """For scripts using pandas: return tz-aware versions of (T0, T1, TSPIN, LABELS)."""
    T0 = pd.Timestamp(T0_ISO, tz='UTC')
    T1 = pd.Timestamp(T1_ISO, tz='UTC')
    TSPIN = pd.Timestamp(TSPIN_ISO, tz='UTC')
    LABELS = pd.date_range(T0 + pd.Timedelta('1h'), T1, freq='h')
    assert len(LABELS) == N_HOURS, len(LABELS)
    return T0, T1, TSPIN, LABELS


if __name__ == '__main__':
    print(NOTE)
    print('first label', LABELS_DT[0], ' last label', LABELS_DT[-1],
          ' total', len(LABELS_DT))
