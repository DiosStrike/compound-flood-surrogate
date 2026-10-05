data/dorian/ -- Dorian 2019 20-day collected data (generated 2026-09-15T15:29:30Z, code/23_collect_events.py)
Range [2019-08-25 00:00, 2019-09-14 00:00) UTC, 480 hourly intervals, label T=[T-1h,T); center day 2019-09-04
raw/        raw responses (byte-for-byte): noaa_8720218_water_level_raw.json / predictions; usgs_02246500_00060_raw.json; rain/GaugeCorr_QPE_01H/ hourly files; download_log.json (URL, parameters, time, sha256)
processed/  mayport_*_6min (native + gaps + NOAA quality flags) / mayport_*_hourly (hourly mean, n_valid, coverage, official_est_frac, interp, value_filled, source_class)
            acosta_*_15min / acosta_*_hourly (same as above; USGS 'e' = official estimate)
            rain_GaugeCorr_QPE_01H_hourly_native_grid.nc (precip_mm raw / precip_mm_filled interpolated / flag; native grid, not reprojected)
            rain_GaugeCorr_QPE_01H_areal_mean_hourly.csv (mean over cell centers in the study rectangle; for check plots only)
meta/       collection_quality_dorian.json (coverage, gaps, peaks, decoding parameters) gaps_dorian.csv (gap list)
Quality: water level  native coverage 1.0000, partial hours 0, official-estimate hours 0, interpolated hours 0, longest gap 0 h
         discharge    native coverage 1.0000, partial hours 0, official-estimate hours 0, interpolated hours 0, longest gap 0 h
         rainfall     missing-file hours 0, hours with uncovered cells 0, interpolated cell-hours 0, unbracketable cell-hours 0
Raw values are never overwritten by interpolation; interpolated values are written only to *_filled columns/variables. The final 10-day window is not yet fixed (awaiting user-specified start day).
