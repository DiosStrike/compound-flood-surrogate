# Proposal -- Flood_2.0: Scenario Generalization, Cross-Event Generalization, and Few-Shot Fine-Tuning of a Compound-Flood Surrogate Model for Jacksonville

Version 2026-09-16. This document covers only the research plan and methods; results obtained, progress, and revision history are in the PDF reports of each round (`result/reports/`) and `docs/PROGRESS.md`.
Execution details follow `docs/PLAN.md`; data specifications follow `docs/DATA_SPEC.md`.

---

## 1. Background and Objectives

Existing surrogate models for compound flooding (storm surge + river + rainfall) are mostly trained and tested on a single event and a single forcing combination, which makes it hard to answer "has the model learned this storm, or this system?" This project takes Jacksonville / the lower St. Johns River as the study area, uses SFINCS as a synthetic data generator, and poses questions through three experiments:

- **Experiment 1, same-event**: Can the surrogate generalize across forcing variants of the same hurricane?
- **Experiment 2, cross-event**: Trained on the other events, can it predict a held-out event? How large are the differences between each event and the rest? (Whether "event diversity vs. number of scenarios" is more valuable is an optional extension, not done in this round; see §6.3.)
- **Experiment 3, few-shot fine-tuning**: For a held-out event, fine-tuning with only N = 1 / 3 / 5 runs: does pretraining help, how much target data is needed, and how much improvement over zero-shot?

The surrogate task is described in Section 5: predict the water-depth field h(t) at target time t; the preferred model is **U-Net**. One model is first carried through the whole project; other models are added once this succeeds.

The paper reports two things separately: **the surrogate's ability to reproduce SFINCS** (main line) and **the accuracy of SFINCS against real floods** (recorded only, no parameter calibration).

---

## 2. Study Area and Fixed Model Setup

| Item | Setting |
|---|---|
| Study area | Union of HUC10 0308010315 + 0308010316 (about 977 km²), bounding rectangle 51 × 39 km |
| Model | SFINCS (Deltares), `deltares/sfincs-cpu` Docker image; version recorded for every run |
| Grid | 200 m, 195 rows (south→north) × 255 columns (west→east), origin (413400, 3337200), EPSG:26917; active cells determined from a 60 m active-domain mask by ≥ 50 % coverage; mask msk: 0 inactive / 1 active / 2 water-level boundary |
| Topography | USGS 3DEP 30 m + NOAA CRM 3″ merged to 60 m (minimum of the two where DEM ≤ 0.5 m), 60 m → 200 m block average (v1) |
| Boundaries | Downstream: Mayport 8720218 water level (msk = 2 cells, 12 boundary points share one series); upstream: Acosta 02246500 signed discharge, split across 4 source cells by cross-section; remaining domain boundaries are wall |
| Rainfall | Hourly QPE → 1 km UTM grid by nearest neighbor, `sfincs.ampr` spatial rainfall; products per event in Section 3 |
| Time | Each event is a **10-day continuous window = 72 h spin-up + 168 h main period**, all in UTC; hour label T denotes interval [T−1h, T); spin-up frames serve only as model inputs, not training targets |
| Parameters | manning_land 0.06 / manning_sea 0.02 / rgh_lev_land 0; huthresh 0.05; advection 1; **qinf = 0 (no infiltration)**; zsini = value of each scenario's water-level boundary at tstart |
| Format and output | inputformat = bin (ind column-major), outputformat = net; `sfincs_map.nc` hourly instantaneous fields (dtout = 3600, 241 snapshots), `sfincs_his.nc` observation points; batch runs set dtmaxout = 0 inside the run directory, the six s041 runs are dtmaxout = 3600 versions, and the two conventions are not compared |
| Coordinate reference | Horizontal EPSG:26917, vertical NAVD88, units m; discharge m³/s (signed, negative = flood-tide reverse flow, not clipped at zero); rainfall mm/hr |
| Rainfall file | First ampr row = geographic north; rainfall grid extends beyond the model domain (west / east / south 1 km, north 2 km; SFINCS does not use the first row of the file); timestamps referenced to tref; inp sets dtwnd = 60 |

The above settings are fixed for all events and all scenarios; the only things that vary by scenario are the three forcing files (bzs / dis / ampr) and zsini. All six events share the same topography, grid, and static files.

---

## 3. Events and Data Sources

### 3.1 Six Events

| Event | 10-day window (UTC, [tstart, tstop]) | Rainfall product |
|---|---|---|
| Irma 2017 | 2017-09-06 00:00 → 09-16 00:00 | MRMS GaugeCorr_QPE_01H |
| Matthew 2016 | 2016-10-01 00:00 → 10-11 00:00 | MRMS GaugeCorr_QPE_01H |
| Ian 2022 | 2022-09-23 00:00 → 10-03 00:00 | MRMS MultiSensor_QPE_01H_Pass2 |
| Milton 2024 | 2024-10-05 00:00 → 10-15 00:00 | MRMS MultiSensor_QPE_01H_Pass2 |
| Dorian 2019 | 2019-08-29 00:00 → 09-08 00:00 | MRMS GaugeCorr_QPE_01H |
| Beryl 2012 | 2012-05-22 00:00 → 06-01 00:00 | NCEP Stage IV 01h (4 km native grid) |

- Event selection: data completeness (Mayport observations and predictions, Acosta discharge, study-area rainfall) first, local impact second; the five additional events were selected by the user.
- Window rule (the five events other than Irma): the day of main impact on the study area is the center day; window = center day −6 d → +4 d.
- 81 scenarios per event, **6 × 81 = 486** SFINCS simulations in total.

### 3.2 Data Sources

| Data | Source | Processing |
|---|---|---|
| Topography | USGS 3DEP 1″ DEM (water surfaces hydro-flattened); NOAA CRM 3″ | Reprojected to EPSG:26917, merged at 60 m, block-averaged to 200 m; produced once and shared by all events (`data/static/`) |
| Observed water level | NOAA CO-OPS 8720218 Mayport, `product=water_level`, `datum=NAVD`, `units=metric`, `time_zone=gmt`, 6 min native | Unit / time checks |
| Astronomical tide prediction | NOAA CO-OPS Data API, 8720218, `product=predictions`, same datum / units / time_zone / 6 min | Subtracted only after three checks against observations: same length, same timestamps, same datum |
| Discharge | USGS NWIS 02246500 Acosta Bridge, parameter 00060 (signed), 15 min native | Units cfs → m³/s; negative values kept |
| Rainfall | MRMS hourly QPE (GaugeCorr_QPE_01H before 2020-10, MultiSensor_QPE_01H_Pass2 after); Beryl uses NCEP Stage IV 01h (GRIB1, HRAP polar stereographic grid) | Clipped to study area, nearest neighbor to 1 km UTM grid; units mm/hr; different events may use different rainfall products, with product / version differences recorded and event totals compared with local rain gauges |
| Validation stations (not forcing) | NOAA 8720219 Dames Point; USGS 02246515 / 02246621 / 02246751 / 02246804 / 02246825 | Used only for the "SFINCS vs. real floods" layer |

**Data principles**: missing data are recorded as missing and not substituted with other products; every download records source URL, request parameters, time, and checksum.

- Irma: original gaps are not filled (e.g., the 30 min interval segment in Acosta discharge is kept as is, so the number of `dis` rows differs from other events).
- Other five events: hourly data distinguish four source types -- observed / officially estimated (NOAA inferred, USGS e) / self-interpolated / partially observed hourly statistics; short gaps inside the window are linearly interpolated in time (rainfall cell by cell), with interpolation written only to separate filled products and logged entry by entry; missing endpoints are bracketed only by looking outside the window, without extrapolation.
- The 25 h filter prescribed by the method and SFINCS's own spatiotemporal interpolation are not counted as "filling".

---

## 4. Scenario Generation (81 per Event)

### 4.1 Naming and Nature

These data are called **"<event>-like compound-forcing scenarios"** (e.g., Irma-like): based on the three observed forcings of that event, perturbed by scaling and timing. They are **not** "hurricanes with a changed hurricane category" -- wind fields, pressure, and tracks are not recomputed, and physical consistency among the three forcings is not reconstructed. The four factors and their values are **this project's sensitivity-experiment design**, not copied from any paper. All six events use the same factors and rules.

### 4.2 Four Factors (3 × 3 × 3 × 3 = 81)

| Factor | Values | Applied to |
|---|---|---|
| α non-tidal water-level residual scale | 0.8 / 1.0 / 1.2 | Mayport observed − astronomical tide prediction |
| β rainfall scale | 0.7 / 1.0 / 1.3 | Whole rainfall field multiplied cell by cell, spatial pattern unchanged |
| γ low-frequency discharge scale | 0.75 / 1.0 / 1.25 | Low-frequency component of Acosta signed discharge |
| τ rainfall time shift | −6 / 0 / +6 h | Rainfall shifted as a whole, positive = delayed; water level and discharge unchanged |

- Scenario IDs `<event>_s001` … `s081`, in lexicographic order of (α, β, γ, τ); α = β = γ = 1, τ = 0 is the unperturbed baseline = s041, and the generation script must reproduce the original forcing point by point.
- Perturbations apply to the full 240 h (including spin-up).

### 4.3 Water Level

```
Non-tidal residual    r(t)   = h_obs(t) − h_tide(t)
Scenario water level  h_s(t) = h_tide(t) + α · r(t)
```
- `h_obs`: Mayport observed water level; `h_tide`: the "astronomical-tide-only water level" computed by NOAA from the station's multi-year harmonic constants. Both are NAVD88, UTC, 6 min.
- `r` is what remains after subtracting the prediction from the observation. It **is not pure storm surge**: it also contains river outflow, seasonal water level, datum error, and prediction error. The paper uses this wording.
- Scenarios keep the same time-varying tide curve and scale only the residual; α = 1 reproduces the observation.

### 4.4 Discharge

```
Q_low(t)  = 25 h centered moving average(Q)(t)
Q_high(t) = Q(t) − Q_low(t)
Q_s(t)    = γ · Q_low(t) + Q_high(t)
```
- `Q`: Acosta 02246500 parameter 00060, 15 min, signed (negative = flood-tide reverse flow), not clipped at zero.
- Why 25 h: the semidiurnal tide is about 12.4 h; averaging over about two tidal cycles (25 h) largely cancels flood and ebb. "Centered" means 12.5 h on each side of t, introducing no time shift.
- **`Q_low` is the low-frequency component and `Q_high` is the retained high-frequency fluctuation**; they must not be called "river discharge" and "pure tide".
- γ multiplies only the low-frequency part; total `Q_s` may change sign at individual times, which is allowed, recorded, and not intervened.
- The filter needs ≥ 13 h of real data on both ends of the window: extra data are taken outward, without shortening the window or extrapolating.
- If the original series has gaps, the moving average uses the available samples within the |t′ − t| ≤ 12.5 h time window, without filling; `Q_low + Q_high` reproduces the original series point by point.
- USGS 72137 (tidally filtered discharge) is used only as a cross-check, not as forcing.

### 4.5 Rainfall

```
P_s(x, t) = β · P(x, t − τ)
```
τ = +6 requires real rainfall for 6 h before tstart, and τ = −6 requires 6 h after tstop. Stage IV uses nearest neighbor on its native 2D grid.

### 4.6 Checks

At generation (results written to scenario.json / manifest):
1. bzs / dis / ampr fully cover [tstart, tstop], with no extrapolated gap filling.
2. zsini = the scenario's `h_s` at tstart.
3. Record the number of sign flips and extremes of `Q_s`; the baseline scenario equals the original forcing point by point.

After simulation:
1. Log has no errors and ends normally, map / his complete, final time-axis value = tstop, input file md5 = manifest.
2. Rainfall verification: with dtmaxout = 0 SFINCS does not write cumprcp, so ampr-only verification is used -- ampr md5 = manifest + file time / value / coverage checks + input total / (β × **the center scenario with the same τ**, s040 / s041 / s042). For τ ≠ 0 the rainfall total inherently differs from s041, so s041 must not be used directly as the reference.
3. After each event's simulations, compare with observation stations (`code/18_validate_obs.py`); report differences only, without changing parameters on our own.

---

## 5. Surrogate Model Task

### 5.1 Task Definition

- Predict the water-depth field h(t) at target time t; **event maximum depth hmax is not done for now**.
- Target times t = 73 … 240 h (main period), 168 samples per run, 486 × 168 = 81,648 samples in total.
- These samples are not mutually independent: adjacent hours within a run are autocorrelated, and the 81 scenarios within an event differ only in 4 scalars.

### 5.2 Inputs and Output

| Quantity | Shape | Description |
|---|---|---|
| `zb` | (195, 255) | Bed elevation; inactive cells filled with the mean zb of active cells (default, see Section 9) |
| `h_lookback` | (3, 195, 255) | h(t−1), h(t−2), h(t−3); at t = 73 the spin-up frames h(70…72) are used |
| `rainfall` | (3, 195, 255) | Rainfall for hours t−1, t−2, t−3, mm/hr |
| `waterlevel` | (3,) | Boundary water level at t−1, t−2, t−3, m NAVD88 |
| `discharge` | (3,) | Discharge at t−1, t−2, t−3, m³/s, signed |
| **Output** `h_tgt(t)` | (195, 255) | float32, m |

- U-Net input has 13 channels: zb 1 + h 3 + rainfall 3 + waterlevel 3 + discharge 3; scalars are broadcast to constant maps.
- **msk is not an input**; it is used only for the loss mask (`loss_mask = msk > 0`) and evaluation.

### 5.3 Processing Rules for Targets and Inputs

- **Target**: wet cells take h, dry cells take 0. Dry cells = SFINCS's `dry` flag, i.e., active cells with h ≤ 0.05 m (huthresh). Dry cells may have negative h; these are also set to 0, and the number of negative cells set to zero is recorded per run. Inactive cells are filled with 0 and excluded from the loss.
- `h_lookback` uses the same rule as the target.
- **Water level**: column 1 of `bzs` (the 12 columns are identical at every time, to be verified); **discharge**: sum of the four columns of `dis`. Both take the **instantaneous value at the top of each hour**; when a top-of-hour record is missing, the nearest record to that hour is used instead, with the substituted time and time offset recorded; if the offset exceeds the series' native step, stop and report.
- **Rainfall**: use `rain_on_model` from `load_run()` (nearest neighbor to the 200 m grid, already spatially aligned, not flipped again); time s corresponds to ampr block s. If block s acts on [s, s+1h) in SFINCS, then rainfall(t−k) takes block s = t−k (k = 1, 2, 3), covering [t−3, t); this time semantics must be confirmed against the SFINCS manual or source code during the pre-training checks, and if it does not hold, stop and report.
- **Grid padding** is done only in the training code, without changing stored data: inputs are padded from 195 × 255 to 208 × 256 (13 rows on the north side, 1 column on the east side, divisible by 16), outputs are cropped back to 195 × 255 before computing the loss; padded cells are filled with 0 (zb with the same fill value) and excluded from the loss.
- **Normalization** statistics are computed only from the respective training set, per fold at training time, not in preprocessing.
- Samples are not pre-windowed; at training time the data loader takes inputs from [t−3, t−1] and the target at t.

### 5.4 Model

The preferred model is **U-Net**; this single model is first carried through Experiments 1-3, and other models are added once this succeeds.

### 5.5 Evaluation Modes

Each experiment reports both:
- **Single-step**: h_lookback uses SFINCS ground truth.
- **Rollout**: starting from h(70…72), the model's own predictions are fed forward for 168 consecutive steps.
- **Persistence baseline (copy previous hour)**: h(t) = ground-truth h(t−1), used to judge whether single-step prediction beats simple persistence.

**Target positioning**: this project focuses on single-step prediction; rollout prediction is evaluated and reported as usual, but improving rollout accuracy is not a goal. **Future direction, not done in this round**: improving rollout prediction (e.g., multi-step training).

### 5.6 U-Net Version 1 (v1) Architecture and Hyperparameters

Version 1 settings; batch size and per-epoch sampling were fixed after the smoke test.

| Item | Setting |
|---|---|
| Architecture | 4 downsampling stages; encoder channels 32 → 64 → 128 → 256, bottleneck 512 |
| Each level | 2 × (3×3 convolution + BatchNorm + ReLU) |
| Downsampling / upsampling | 2×2 max pooling / 2×2 transposed convolution |
| Skip connections | Channel concatenation |
| Output layer | 1×1 convolution + ReLU (ensures h ≥ 0) |
| Input | 13 channels: zb 1, h 3, rainfall 3, waterlevel 3, discharge 3 (scalars broadcast to constant maps) |
| Grid padding | 195 × 255 → 208 × 256 (13 rows north, 1 column east), output cropped back to 195 × 255 |
| Input standardization | Computed from the training set only, per experiment / per fold; zb over active cells; the 3 h channels share one set of statistics, likewise for rainfall, water level, and discharge; not done in preprocessing |
| Target | h(t), not normalized (m) |
| Loss | MSE over active cells only (loss_mask) |
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 8 |
| Epochs and early stopping | At most 50 epochs; early stop if validation loss does not improve for 10 epochs |
| Random seed | 42 |
| Per-epoch sampling | Every run in the training and validation sets uses fixed t = 73, 80, 87, …, 234 (one every 7 hours, 24 times) |
| Testing | All 168 times evaluated, single-step and rollout |
| Evaluation | Single-step (h_lookback uses ground truth); rollout (from ground-truth h(70…72), for t ≥ 73 model predictions are fed back, inactive cells fed back as 0) |

Note (sampling): water depth changes little between adjacent hours, so taking one time every 7 hours greatly shortens training. Comparing full training with every-7-hour training is listed as a future direction (§6.2) and not done in this round.

![U-Net version 1 (v1) architecture](figures/unet_modelB_v1.png)

---

## 6. Experimental Design and Data Splits

### 6.1 Shared Split split_A

- The 81 scenarios of each event are split by **whole run**: all cells, time slices, and derived samples of a run go into only one set. Train 60 / validation 11 / test 10; once generated the split table is frozen (`split_A.csv`), shared by all three experiments, and no new split is created.
- All six events are split by the same rule: random seed **42**; **composite forcing level** (defined by this project, not real severity) scores α, β, γ each as low / medium / high = 0 / 1 / 2, total 0-6: low = 0-2 points (30 scenarios), medium = 3 points (21), high = 4-6 points (30); τ is not scored but is balanced across sets as far as possible. Stratification is below; each event's unperturbed baseline s041 is always placed in the test set. The factor combinations are identical across the six events, so the same scenario ID belongs to the same set in all six:

| Set | Low | Medium | High | Total |
|---|---|---|---|---|
| Train | 23 | 14 | 23 | 60 |
| Validation | 4 | 3 | 4 | 11 |
| Test | 3 | 4 | 3 | 10 |

### 6.2 Experiment 1 -- Same-Event

- split_A of all six events merged: 360 training runs / 66 validation runs.
- The 10 test runs of each event are **reported separately**, giving 6 sets of results.
- **Future directions, not done in this round** (written in the discussion): (1) compare full training (168 times per run) with every-7-hour training (24 times per run); only every-7-hour sampling is used in this round; (2) improve rollout prediction (e.g., multi-step training); (3) raise the epoch limit -- Experiment 1 and several folds of Experiment 2 ended at the epoch limit while still improving, so a higher limit may further improve accuracy; it is not done in this round because current results suffice to answer the three experiments' questions, all folds use the same settings so results are mutually comparable, and changing the limit would require rerunning all folds.

### 6.3 Experiment 2 -- Cross-Event (6-Fold Leave-One-Out)

- Train 6 leave-one-out models: each uses the split_A training / validation sets of 5 events and is tested on the 10 test runs of the held-out event (all 81 runs of that event may also be reported).
- **No run of the held-out event may enter that model's training or tuning.**
- The results of the 6 models are used to compare each event with the rest.
- Note: all six events share the same topography and grid, so **"cross-event" means cross-forcing, not cross-basin**; there are only 6 real events, so the effective independent sample size across events is about 6.
- **Optional extension, not done in this round**: fixed-budget comparison -- e.g., "1 event × 50 scenarios" vs. "5 events × 10 scenarios" with the same total number of simulations, evaluated on the same held-out events, to answer whether "event diversity vs. number of scenarios" is more valuable; the number of scenarios per training event is kept balanced.

### 6.4 Experiment 3 -- Few-Shot Fine-Tuning

- Done for every leave-one-out model (fixed procedure, not triggered by performance).
- Fine-tune with N = 1 / 3 / 5 runs from the held-out event's split_A training pool; the test set is unchanged (i.e., the Experiment 2 test set).
- 3 random seeds per N (may be extended to 5 later).
- Controls: zero-shot (Experiment 2 results); training from scratch on only these N runs.
- Questions to answer:
  - Does pretraining help: fine-tuning vs. training from scratch on the same N runs;
  - How much target data is needed: how performance changes with N = 1 → 3 → 5 (N is the number of runs from the held-out event, not the number of events);
  - How much improvement over zero-shot, and which held-out events benefit more.
- Reports distinguish "zero-shot" from "after fine-tuning": once target-event data are used for fine-tuning, that event is no longer an "unseen event".

### 6.5 Data Storage and Reproducibility

- Each scenario keeps its complete inputs (inp / dep / msk / ind / bnd / bzs / src / dis / ampr / obs), `sfincs_map.nc`, `sfincs_his.nc`, logs, QC results, and derived results; shared static data are kept only once (`data/static/`).
- `manifest.csv` records at least: α, β, γ, τ, composite forcing level, dataset assignment (train / val / test), time window, data sources (URL and request parameters), input file checksums (md5), SFINCS version, random seed, run status.
- Surrogate preprocessing output `data/modelB_v1/`: one copy of static quantities, one file per run (h_proc, rain_proc, wl_hourly, dis_hourly), plus a run-level index table (event, sid, split_A label, α β γ τ, path, md5, number of top-of-hour substitutions, number of negative-h cells set to zero, self-check result); the sample-level index is generated by the data loader and not written to disk.
- Every scenario must be independently reproducible from the manifest and fixed scripts.

---

## 7. Evaluation Metrics

Experiments 1, 2, and 3 use the same conventions, reported per experiment and per event, plus a multi-event total; one set each for single-step, rollout, and the persistence baseline; testing evaluates all 168 times:
- Water-depth MAE / RMSE (m), reported separately over **active cells** (msk > 0) and **land cells** (active and zb ≥ 0);
- Inundation extent CSI = TP / (TP + FP + FN), wet = h > threshold, **thresholds 0.1 m and 0.3 m**, one set each for active and land cells;
- Aggregation: pooled over runs × times × cells within each event;
- Rollout additionally reports: error over time; per-run comparison of predicted maximum depth (cell-wise maximum over t = 73…240) with SFINCS hmax (MAE / RMSE, CSI, land inundation area error).

The two layers "surrogate reproducing SFINCS" and "SFINCS vs. real floods" are kept distinct; the latter uses the validation stations of Section 3.2 and is recorded only.

---

## 8. Limitations, Implementation Steps, and References

### 8.1 Limitations

- The 200 m grid does not represent four tributaries (Pottsburg / Trout / Broward / Dunn); they are already absent in the 30 m source DEM, and related stations are excluded from quantitative validation.
- qinf = 0: infiltration is not considered, so pluvial flooding may be overestimated; this belongs to the "SFINCS vs. real floods" layer and does not affect the internal consistency of the surrogate experiments.
- Scenarios are not physically self-consistent hurricanes (no recomputed wind field, pressure, or track); the non-tidal residual is not pure storm surge.
- No wind / pressure forcing; wind-driven setup enters only indirectly through Mayport observations.
- The CRM vertical datum is not declared in the files; rainfall products differ across events (two generations of MRMS products, Stage IV 4 km).
- A small number of water cells on the domain boundary are treated as wall.
- All six events share the same topography and grid, so cross-event conclusions are cross-forcing rather than cross-basin; there are only 6 real events.
- Model inputs and outputs are hourly snapshots; transient processes between hours are not in the data.
- Rainfall input is nearest-neighbor resampled to 200 m and does not equal SFINCS's internal bilinear interpolation.

### 8.2 Implementation Steps

1. **Pre-training checks** (read-only): time semantics of SFINCS rainfall / boundaries / output (checked against the manual or source code, with sources cited); top-of-hour availability of bzs / dis for all 486 runs; hourly-sampling loss; rainfall row order and bzs duplicate-column checks.
2. **Preprocessing**: static quantities → per-run processing (first try 1 run, report size and time, then batch after confirmation) → per-run self-checks (main-period max h_proc equals `hmax_event7d_hourlysampled.npy` cell by cell, top-of-hour value checks, rainfall total ratio, no NaN / negative values) → index table. No training, no normalization, no new split.
3. **Experiment 1** same-event.
4. **Experiment 2** cross-event (6-fold leave-one-out).
5. **Experiment 3** few-shot fine-tuning.

If any check fails: stop and report, without patching or guessing. Report the expected duration before launching any time-consuming operation.

### 8.3 References

- Leijnse, T., van Ormondt, M., Nederhoff, K., van Dongeren, A. (2021). Modeling compound flooding in coastal systems using a computationally efficient reduced-physics solver: including fluvial, pluvial, tidal, wind- and wave-driven processes. *Coastal Engineering*, 163, 103796. https://doi.org/10.1016/j.coastaleng.2020.103796 -- SFINCS model; §4.1 is the Jacksonville / Irma case.
- Lee, J., et al. (2025). Probabilistic storm surge and flood-inundation modeling of the Texas gulf coast using SFINCS. *Coastal Engineering*, 198, 104721. https://doi.org/10.1016/j.coastaleng.2025.104721 -- runs perturbed-forcing ensembles with SFINCS (81 / 189 / 1000 members), without an ML surrogate; this project borrows only the "ensemble" idea, with different perturbation targets.
- Zahura, F. T., Goodall, J. L., Sadler, J. M., Shen, Y., Morsy, M. M., Behl, M. (2020). Training machine learning surrogate models from a high-fidelity physics-based model: application for real-time street-scale flood prediction in an urban coastal community. *Water Resources Research*, 56, e2019WR027038. https://doi.org/10.1029/2019WR027038 -- physics model as ground truth, split by whole events.
- Xu, Q., et al. (2025). Urban flood modeling and forecasting with deep neural operator and transfer learning. *Journal of Hydrology*, 661, 133705. https://doi.org/10.1016/j.jhydrol.2025.133705 -- source-domain pretraining → fine-tuning with little target data → comparison with training on target data only.
- Guo, L., et al. (2015). River-tide dynamics: Exploration of nonstationary and nonlinear tidal behavior in the Yangtze River estuary. *Journal of Geophysical Research: Oceans*, 120. https://doi.org/10.1002/2014JC010491 -- reference only for the idea of low-frequency / tidal separation (checked at abstract level only), not a direct basis for this project's discharge scaling method.
- Other citations from the surrogate-model selection survey are in `Model_paper/PAPERS.csv`.
- NOAA CO-OPS Data API documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
- USGS NWIS Instantaneous Values Web Service: https://waterservices.usgs.gov/docs/instantaneous-values/
- NOAA MRMS QPE products (GaugeCorr_QPE_01H; MultiSensor_QPE_01H_Pass2) and NCEP Stage IV 01h, archive: Iowa State University MTArchive, https://mtarchive.geol.iastate.edu/
- SFINCS user manual: https://sfincs.readthedocs.io/

---

## 9. Open Items

- **Fine-tuning method** (Experiment 3): full-network fine-tuning / freezing some layers / tuning only the output layer.
- Two further items have defaults and do not block preprocessing: fill value for `zb` at inactive cells (default: mean over active cells); storage format (default: one npz per run, decided after trying 1 run).
