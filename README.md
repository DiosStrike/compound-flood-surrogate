# Cross-Event Generalization and Few-Shot Fine-Tuning of a Compound-Flood Surrogate Model: A Systematic Measurement with SFINCS

*Independent research project.*

Physics-based compound-flood models are reliable but expensive when many runs are needed. This project uses the SFINCS hydrodynamic model to generate 81 compound-forcing scenarios for each of six tropical cyclones affecting Jacksonville and the lower St. Johns River, Florida (Irma, Matthew, Ian, Milton, Dorian, Beryl; 486 ten-day simulations on a 200 m grid). A U-Net is trained to predict the hourly water-depth field. Three experiments share one data split: (1) same-event accuracy, (2) six-fold leave-one-event-out cross-event generalization, and (3) few-shot fine-tuning on the held-out event, including a frozen-encoder variant and a train-from-scratch control. Results and figures are in the [report](report/report.pdf).

## Repository structure

```
compound-flood-surrogate/
├── proposal/proposal.md     research plan and methods
├── report/report.pdf        final report
├── code/                    all scripts and Colab notebooks (numbered in pipeline order)
│   ├── paths.py             single source of truth for project paths
│   ├── read_run.py          load_run(): read one SFINCS simulation
│   ├── 01–16                data audit, study extent, terrain, hydro/rain forcing, SFINCS model build
│   ├── 18, 20, 21           observation validation, ampr row-order test, rainfall QC
│   ├── 22–25                Irma scenarios, multi-event data collection, 10-day windows, event scenarios
│   ├── 26–29                batch verification, data completeness, summary reports
│   ├── 30–35, 37            surrogate (U-Net) checks, preprocessing, data loader, model, training, eval, monitor
│   ├── 38–41, 44, 45        Exp. 2 (cross-event, Colab) training/eval/monitoring/packing/summary
│   ├── 42–43                Exp. 1 evaluation (43 also provides plotting helpers used by 44)
│   ├── 46–49                Exp. 3 (few-shot fine-tuning) sampling/training/eval library/summary
│   ├── 52, 55, 62, 66, 68, 70–72   report table generation
│   └── colab_*.ipynb        Colab notebooks for Exp. 2 and Exp. 3 (one per fold)
├── run_sfincs.sh            run one SFINCS simulation in Docker
├── run_scenarios.sh         batch-run the Irma scenarios
├── run_all_events.py        driver for all 6 events × 81 scenarios
├── data/
│   ├── static/              terrain, GIS boundaries, metadata, SFINCS base model (grid, mask, boundaries)
│   └── <event>/             meta/ (data-quality records) and processed/ (water level, discharge, rainfall series)
├── results/                 small result files: evaluation metrics (CSV/JSON) and summary tables for Exp. 1–3, manifests
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
```

SFINCS runs in Docker (`deltares/sfincs-cpu`). Run all commands from the repository root. Scripts write outputs to `result/` and `runs/`, which are created as needed.

1. **Forcing and scenarios.** `python3 code/23_collect_events.py all` downloads and processes the raw data for the five additional events. `python3 code/24_cut_windows.py all` cuts the 10-day windows. `python3 code/22_make_scenarios.py` (Irma) and `python3 code/25_make_event_scenarios.py all` generate the 81 scenarios per event.
2. **SFINCS simulations.** Use `bash run_sfincs.sh <name> [event]` for one run, or `python3 run_all_events.py` for all 486 runs (`--dry-run` lists them first). Then validate with `python3 code/18_validate_obs.py <run> <event>` and `python3 code/21_rain_qc.py <run> <event>`.
3. **Preprocessing.** `python3 code/31_modelB_preprocess.py --all` writes `data/modelB_v1/`.
4. **Experiment 1 (same event).** `python3 code/34_modelB_train.py --exp exp1 --out result/models/exp1 --stats-only` (normalization statistics), then `python3 code/34_modelB_train.py --exp exp1 --out result/models/exp1 --stats result/models/exp1/norm_stats.json`, then `python3 code/42_exp1_eval.py`.
5. **Experiment 2 (cross-event, 6 folds).** Pack the data with `python3 code/41_pack_modelB_data.py pack`, run `code/colab_exp2_<event>.ipynb` on Colab (or `code/38_exp2_train.py --fold <event> --eval-after`), then summarize with `python3 code/44_exp2_summary.py`.
6. **Experiment 3 (few-shot fine-tuning).** `python3 code/46_exp3_sampling.py`, then `code/colab_exp3_<event>.ipynb` (`code/47_exp3_train.py`), then `python3 code/49_exp3_summary.py`.

Each script's docstring lists its options.

## Data

This repository includes the static model inputs, processed forcing series, quality metadata, and small result tables. The following large files are **not** included; they can be regenerated with the scripts listed:

| Not included | Size | What it is | How to obtain / regenerate |
|---|---|---|---|
| `data/<event>/raw/` | ~1.8 GB | Raw downloads: NOAA CO-OPS Mayport 8720218 water level and tide predictions, USGS NWIS 02246500 (Acosta Bridge) discharge, and hourly rainfall grids (MRMS GaugeCorr / MultiSensor QPE; NCEP Stage IV for Beryl) | Public sources. Download with `code/23_collect_events.py download all` (five events), and `code/04_hydro_series.py` + `code/05_mrms_rainfall.py` for Irma |
| `data/<event>/scenarios/` | ~1.4 GB | 81 SFINCS forcing sets per event (`sfincs.inp`, `.bzs`, `.dis`, `.ampr`, `manifest.csv`, frozen `split_A.csv`). Scenarios vary non-tidal residual scale α, rainfall scale β, low-frequency discharge scale γ, and rainfall time shift τ | `code/22_make_scenarios.py` (Irma) and `code/25_make_event_scenarios.py all` (fixed seed 42) |
| `runs/` | ~12 GB | Output of the 486 SFINCS simulations (`sfincs_map.nc`, `sfincs_his.nc`, hourly max-depth arrays) | `run_all_events.py` / `run_sfincs.sh` with Docker `deltares/sfincs-cpu` |
| `data/modelB_v1/` | ~2.4 GB | Preprocessed surrogate training data: one `.npz` per run (hourly water depth and rainfall on the 195×255 grid, hourly water level and discharge), plus `static.npz` and `index.csv` | `code/31_modelB_preprocess.py --all` (reads `runs/`) |
| Model weights and large outputs | ~2.3 GB | Trained U-Net checkpoints (`*.pt`), per-time-step metric tables, peak-depth maps (`*.npz`), figures | Retrain with steps 4–6 above |

## Report

The full write-up is in [report/report.pdf](report/report.pdf).
