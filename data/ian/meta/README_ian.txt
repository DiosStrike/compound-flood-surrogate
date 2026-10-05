data/ian/ -- Ian 2022 20-day collected data (generated 2026-09-15T15:20:16Z, code/23_collect_events.py)
Range [2022-09-19 00:00, 2022-10-09 00:00) UTC, 480 hourly intervals, label T=[T-1h,T); center day 2022-09-29
raw/        raw responses (byte-for-byte): noaa_8720218_water_level_raw.json / predictions; usgs_02246500_00060_raw.json; rain/MultiSensor_QPE_01H_Pass2/ hourly files; download_log.json (URL, parameters, time, sha256)
processed/  mayport_*_6min (native + gaps + NOAA quality flags) / mayport_*_hourly (hourly mean, n_valid, coverage, official_est_frac, interp, value_filled, source_class)
            acosta_*_15min / acosta_*_hourly (same as above; USGS 'e' = official estimate)
            rain_MultiSensor_QPE_01H_Pass2_hourly_native_grid.nc (precip_mm raw / precip_mm_filled interpolated / flag; native grid, not reprojected)
            rain_MultiSensor_QPE_01H_Pass2_areal_mean_hourly.csv (mean over cell centers in the study rectangle; for check plots only)
meta/       collection_quality_ian.json (coverage, gaps, peaks, decoding parameters) gaps_ian.csv (gap list)
Quality: water level  native coverage 1.0000, partial hours 0, official-estimate hours 0, interpolated hours 0, longest gap 0 h
         discharge    native coverage 0.9969, partial hours 1, official-estimate hours 0, interpolated hours 1, longest gap 1 h
         rainfall     missing-file hours 3, hours with uncovered cells 0, interpolated cell-hours 5724, unbracketable cell-hours 0
Raw values are never overwritten by interpolation; interpolated values are written only to *_filled columns/variables. The final 10-day window is not yet fixed (awaiting user-specified start day).
