# Satellite Image Segmentation for Land Cover and Deforestation Mapping Using Deep Learning

Final-year project. Study area: Kaziranga / Karbi Anglong region, Assam, India (EPSG:32646, 30 m, 3009 x 3704 px, about 90 x 111 km).

> **Status:** complete and verified. The data is validated, the pipeline is implemented with 57 passing tests, the model is trained and evaluated on a held-out region, and full-scene maps are generated. The full `./run_pipeline.sh` run finished all 7 stages with exit code 0.

## Overview

A multi-task U-Net (TensorFlow/Keras) segments bi-temporal Sentinel-2 composites into two pixel-level outputs:

- **Task A, land cover:** 5 classes from ESA WorldCover 2021 (tree cover, grassland, cropland, built-up, water).
- **Task B, forest loss 2021-2023:** binary, from Hansen Global Forest Change v1.12. This is *gross stand-replacement forest-cover loss*, not strictly deforestation.

**Input:** pre composite (Nov 2020 - Feb 2021) and post composite (Nov 2023 - Feb 2024), bands B2, B3, B4, B8, B11, B12, stacked into **12 channels**. `nobs` is dropped; see below.

### Key decisions (all in [`configs/config.yaml`](configs/config.yaml))

| Topic | Decision |
|---|---|
| Validation | `inspect_rasters.py` checks every raster (grid, bands, dtypes, statistics, codes, recomputes `loss_target`, AOI metadata). Later stages refuse to run unless the latest validation passed for byte-identical files (SHA-256). |
| `nobs` band | Dropped (12 channels). Validation found no missing pixels, and `nobs` (4-84) mostly reflects orbit overlap, a location artefact. Configurable (`s2.nobs_mode`). |
| Normalisation | Per channel: clip to the train 1st-99th percentiles, then standardise with train mean/std. Computed from TRAIN patches only. |
| Land-cover labels | Raw IDs 10/30/40/50/80 map to classes 0-4. IDs 0/20/60/90 map to 255 (ignore). Pixels with `loss_target == 1` are also ignored, because their 2021 label is stale in the 2023-24 image. |
| Forest-loss labels | `loss_target` used as exported: 1 = loss 2021-23 on >= 30 % canopy (2000), 0 = no loss, 255 = ignore. |
| Spatial split | 512 px blocks (4 x 4 patches) assigned whole to train/val/test **before** patching, with a 1-patch buffer. The seeded search (seed 42) requires >= 10,000 loss pixels in val and test and >= 10,000 pixels of every class in every split, and matches 70/15/15 for both patches and loss pixels. |
| Patches | 128 x 128, stride 128 (no overlap). |
| Model | Shared encoder (16-32-64-128, bottleneck 256), separate decoders, softmax(5) and sigmoid(1) heads. A lighter shared-decoder variant and a single-task U-Net baseline are also implemented. |
| Losses | Land cover: class-weighted CE (sqrt inverse frequency, clipped 0.2-10). Forest loss: positive-weighted BCE (sqrt(neg/pos), capped at 10) + soft Dice, or focal + Dice. All ignore 255, normalised by valid pixels. |
| Threshold | Tuned on **validation only** (max F1), then applied unchanged to test and to the full-scene maps. |
| Reproducibility | Fixed seeds. Every stage writes a timestamped run directory with `metadata.json` (sources, hashes, config, lineage). Downstream stages refuse stale or incomplete upstream runs. |

## Current data split (measured)

| Split | Patches | Forest-loss px | Tree | Grass | Crop | Built-up | Water |
|---|---|---|---|---|---|---|---|
| train | 317 | 79,703 | 3,645,486 | 194,775 | 902,276 | 33,276 | 314,369 |
| val | 71 | 16,898 | 762,491 | 51,619 | 282,192 | 15,074 | 28,189 |
| test | 108 | 17,504 | 1,121,364 | 52,582 | 511,235 | 12,307 | 44,609 |

148 patch cells were dropped as buffer between regions. The map is in `outputs/plots/split_map.png`.

## Folder structure

```
satellite_landcover_deforestation/
├── configs/config.yaml        # every path, band, class, split, model and training setting
├── data/
│   ├── raw/                   # Earth Engine exports (validated, never modified)
│   ├── processed/prep_*/      # full-scene image/label arrays + metadata
│   ├── splits/split_*/        # block assignment, patch windows, split map
│   └── patches/patches_*/     # train/val/test .npz patches, manifest.csv, band_stats.json
├── gee/                       # Earth Engine scripts
├── docs/                      # dataset, methodology, results, report support
├── demo/app.py                # optional Streamlit demo
├── outputs/
│   ├── checkpoints/           # train_*/ (best.keras, last.keras, history.csv, ...)
│   ├── plots/                 # split map, eval_*/ figures
│   ├── predictions/           # pred_*/ GeoTIFFs, preview.png, area summary
│   └── reports/               # validation/ and evaluation/ reports
├── scripts/                   # one CLI per pipeline stage (exit code 1 on any failure)
│   ├── inspect_rasters.py  prepare_dataset.py  split_dataset.py  create_patches.py
│   └── train.py  evaluate.py  predict.py
├── src/
│   ├── validation.py  preprocessing.py  spatial_split.py  dataset.py
│   ├── losses.py  metrics.py  inference.py  visualization.py  runs.py  cli.py
│   └── models/  unet.py  multitask_unet.py
├── tests/                     # 57 unit tests (synthetic arrays only inside tests)
├── requirements.txt  requirements-demo.txt  pytest.ini
├── run_pipeline.sh            # Linux/macOS runner
└── run_pipeline.ps1           # Windows PowerShell runner
```

## Setup

Requires 64-bit **Python 3.11-3.13** (3.12 recommended). TensorFlow 2.21 has no wheels for Python 3.14.

**Linux** (this machine; `uv` is installed):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import tensorflow as tf, rasterio, numpy; print(tf.__version__, rasterio.__version__, numpy.__version__)"
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Stage order: `inspect_rasters -> prepare_dataset -> split_dataset -> create_patches -> train -> evaluate -> predict`.

```bash
python -m pytest -q                       # 57 tests
./run_pipeline.sh --smoke-test            # tiny end-to-end run (must pass first)
./run_pipeline.sh                         # full run
```

On Windows use `.\run_pipeline.ps1 -SmokeTest` and `.\run_pipeline.ps1`. Each runner checks the exit code after every stage and stops at the first failure.

Individual stages: `python scripts/<stage>.py --config configs/config.yaml [--smoke-test]`.
The runner uses `.venv/bin/python` automatically if it exists, so the venv does not need to be activated for it.
The data stages reuse their latest run when the raw files and relevant config are unchanged (content fingerprint), so re-running the pipeline does not duplicate data or invalidate a trained model. Pass `--force` to a data stage to rebuild anyway.
Resume training: `python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/train_<timestamp> --epochs N`.

## Results (held-out geographic test region)

From the final full pipeline run (`eval_20261124_171655`); details in [docs/RESULTS.md](docs/RESULTS.md).

| Task | Metric | Model | Trivial baseline |
|---|---|---|---|
| Land cover | Overall accuracy | **0.877** | 0.644 (majority class) |
| Land cover | Mean IoU / mean Dice | **0.561 / 0.669** | 0.129 / n/a |
| Land cover | IoU tree / crop / water / built-up / grass | 0.890 / 0.720 / 0.771 / 0.266 / 0.159 | |
| Forest loss | Precision / Recall / F1 / IoU | **0.391 / 0.702 / 0.502 / 0.335** | F1 0 (all "no loss") |
| Forest loss | PR-AUC | **0.558** | 0.0105 (random) |
| Forest loss | Threshold (tuned on validation) | 0.43 | |

- Training took 8.8 min on CPU (best epoch 24 of 32).
- Retraining the same configuration changed the scores by about ±0.01.
- Grassland and built-up are the weak classes.
- The full-scene maps over-estimate forest-loss area (258.6 km² vs 135.6 km² in Hansen) and built-up area (about 2x). Use them to show *where*, not *how much*.

**These scores measure agreement with the reference labels (ESA WorldCover 2021, Hansen GFC v1.12), not independently verified real-world accuracy.**

## Documentation

| Document | Content |
|---|---|
| [docs/DATASET.md](docs/DATASET.md) | Dataset documentation, measured statistics, data dictionary, temporal strategy, split, citations |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Workflow and data flow, preprocessing, model architecture, losses, training, evaluation metrics, inference |
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results, reproducibility, full-scene areas, interpretation |
| [docs/REPORT_SUPPORT.md](docs/REPORT_SUPPORT.md) | Problem statement, objectives, existing vs proposed system, architecture, algorithms, limitations, future scope |

## Demo (optional)

The pipeline does not need it. After a full pipeline run:

```bash
pip install -r requirements-demo.txt      # Linux venv created with uv: uv pip install -r requirements-demo.txt
streamlit run demo/app.py
```

It has three tabs: full-scene maps with GeoTIFF downloads, test metrics and figures, and live prediction on any 512 x 512 px window compared with the reference labels.

## Data sources

- Copernicus Sentinel-2 L2A (harmonized) and Cloud Score+, via Google Earth Engine.
- ESA WorldCover 2021 v200 (CC-BY-4.0). Zanaga et al. 2022, doi:10.5281/zenodo.7254221.
- Hansen et al. 2013, *Science* 342:850-853, doi:10.1126/science.1244693 (GFC v1.12, 2024).
