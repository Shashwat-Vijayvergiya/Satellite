# Methodology: workflow, model, training and evaluation

## 1. Workflow and data flow

```
 data/raw/ (GEE exports)
     │
     ▼
 [1] inspect_rasters ──► outputs/reports/validation/run_*/validation_report.json
     │  hard checks (grid, bands, dtypes, codes, constant bands, loss_target recompute, AOI)
     │  any error ─► exit 1, pipeline stops
     │  records SHA-256 of every raw file (latest.json)
     ▼
 [2] prepare_dataset ──► data/processed/prep_*/
     │  stack pre+post (12 ch), clip DN, valid mask,
     │  WorldCover remap (10/30/40/50/80 → 0..4, rest → 255),
     │  land cover ignored where loss_target == 1, forest-loss target
     ▼
 [3] split_dataset ──► data/splits/split_*/
     │  512 px blocks → train/val/test (seeded, constrained), 1-patch buffer
     │  decides the patch windows BEFORE any patch is cut
     ▼
 [4] create_patches ──► data/patches/patches_*/
     │  128 x 128 patches (stride 128) per split, manifest.csv,
     │  normalisation stats from TRAIN patches only
     ▼
 [5] train ──► outputs/checkpoints/train_*/
     │  multi-task U-Net, masked losses, early stopping, best/last checkpoints
     ▼
 [6] evaluate ──► outputs/reports/evaluation/eval_*/ + outputs/plots/eval_*/
     │  threshold tuned on VALIDATION, metrics on TEST region only
     ▼
 [7] predict ──► outputs/predictions/pred_*/
        full scene: land-cover GeoTIFF, loss probability GeoTIFF, loss mask GeoTIFF,
        preview PNG, area summary
```

**Guarantees built into the pipeline**
- **Fail fast:** every script exits with code 1 on any exception. `run_pipeline.sh` and `run_pipeline.ps1` check the exit code after every stage and stop immediately.
- **No stale data:** each stage writes a new timestamped run folder and only then updates `latest.json`. Every stage checks the parent IDs of its inputs (patches built from the current split, split built from the current prep, prep built from the currently validated raw files) and refuses mismatches or incomplete runs.
- **No needless rebuilds:** `prepare_dataset`, `split_dataset` and `create_patches` store a fingerprint (SHA-256 of their parent run and the config sections they depend on). If the latest run has the same fingerprint, it is reused, so a trained model stays linked to its exact data. `--force` rebuilds.
- **Reproducibility:** seed 42 for Python, NumPy and TensorFlow (`keras.utils.set_random_seed`) and for the split search. Each training run saves its exact `run_config.yaml`.

## 2. Preprocessing

| Step | Method |
|---|---|
| Input stack | pre [B2, B3, B4, B8, B11, B12] + post [same] = 12 channels |
| `nobs` | Dropped (configurable). No missing pixels exist, and `nobs` reflects orbit overlap. Validity still uses `nobs > 0` and non-zero reflectance in both dates. |
| No data | Pixels invalid in either date get label 255 in both tasks (none in this scene) |
| Clipping | reflectance DN capped at 10000 (14 pixels affected) |
| Normalisation | per channel: clip to the train 1st-99th percentile, then (x - mean) / std of the clipped train values. Patches are stored as raw DN and normalised at load time. |
| Augmentation | train only: random horizontal/vertical flip and 90° rotation, applied identically to image and both label maps |

## 3. Model architecture

**Why U-Net:** segmentation needs one label per pixel. The encoder downsamples to gather context (is this hillside forest or jhum fallow?), and the decoder upsamples back to full resolution. Skip connections carry fine detail lost by pooling (field edges, river banks, clearing outlines), so boundaries stay sharp. U-Nets train well from a few hundred patches, which suits a single-scene dataset, and are small enough for a CPU. No claim is made that a more complex architecture would or would not do better; that was not tested.

**Multi-task design:** one shared encoder feeds two task-specific decoders. Land cover and forest loss both depend on the same spectral and textural cues (canopy, bare soil, moisture), so sharing the encoder lets each task benefit from the other's supervision and halves the encoder cost.

```
 image (128,128,12)
   enc0  2x[3x3 conv 16 + BN + ReLU] ──────────────────────────┐ skip
   pool → enc1  2x[conv 32] ──────────────────────────────┐    │
   pool → enc2  2x[conv 64] ─────────────────────────┐    │    │
   pool → enc3  2x[conv 128] ───────────────────┐    │    │    │
   pool → bottleneck 2x[conv 256] + dropout 0.2 │    │    │    │
            │                                   │    │    │    │
            ├─► land-cover decoder: up 2x2 → concat skip → 2x[conv] (128, 64, 32, 16) → 1x1 conv, softmax(5)
            └─► forest-loss decoder: same structure                                  → 1x1 conv, sigmoid(1)
```

| Part | Parameters |
|---|---|
| Shared encoder (4 levels) | 296,256 |
| Bottleneck (8 x 8 x 256) | 886,784 |
| Land-cover decoder + head | 763,845 |
| Forest-loss decoder + head | 763,777 |
| **Total** | **2,710,662** |

Variants also implemented (`src/models/`): a lighter **shared-decoder** variant (`model.decoder: shared`, 1.95 M parameters) and a **single-task U-Net** baseline (`build_unet`). Only the separate-decoder multi-task model has been trained and evaluated.

**CPU cost (measured on this machine, 12 threads):** about 0.6 s per training step of 8 patches, so roughly 0.5 min per epoch on 317 train + 71 val patches.

## 4. Losses and class imbalance

All losses skip pixels labelled 255 and divide by the number of valid pixels in the batch (or, for weighted CE, by the summed weight of valid pixels). A batch with many ignored pixels is not diluted, and a fully ignored batch gives exactly 0.

**Land cover:** class-weighted sparse categorical cross-entropy. Weights are smoothed inverse frequencies from the train pixels, w_c = (1/f_c)^0.5, rescaled so the pixel-weighted mean is 1, then clipped to [0.2, 10]. Raw 1/f weights would give built-up about 110 x tree cover and destabilise training, which is why they are smoothed.

| Class | Train pixels | Weight |
|---|---|---|
| tree_cover | 3,645,486 | 0.659 |
| grassland | 194,775 | 2.852 |
| cropland | 902,276 | 1.325 |
| built_up | 33,276 | 6.901 |
| water | 314,369 | 2.245 |

**Forest loss:** positive-weighted binary cross-entropy + soft Dice (configurable to focal + Dice). Positives are 1.70% of train pixels (79,703 of 4,676,214). pos_weight = sqrt(neg/pos) = **7.59** (cap 10). Dice directly optimises overlap and is insensitive to the huge number of true negatives.

**Total:** 1.0 x land-cover loss + 1.0 x forest-loss loss (task weights configurable).

## 5. Training

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 1e-3, ReduceLROnPlateau (x0.5 after 3 stagnant epochs, min 1e-5) |
| Batch / epochs | 8 / up to 40 |
| Early stopping | val_loss, patience 8, restore best weights |
| Checkpoints | `best.keras` (lowest val_loss), `last.keras` every epoch, `state.json` for `--resume` |
| Logged each epoch | total and per-task losses, land-cover mIoU, forest-loss F1 and recall at 0.5 (train and val), in `history.csv` |
| Smoke test | `--smoke-test`: 16 patches per split, 1 epoch, batch 4, 512 px prediction window; must pass before full training |

## 6. Evaluation

Evaluation uses only the held-out spatial **test** blocks. The **validation** blocks are used for early stopping and to choose the decision threshold.

**Threshold:** the forest-loss probability threshold is swept from 0.05 to 0.95 (step 0.01) on **validation** pixels, and the F1-maximising value is fixed. It is then applied unchanged to test and to the full-scene map. The test-split sweep is saved for analysis only (`threshold_test_analysis_only.csv`) and is never used for selection.

**Land-cover metrics** (per class and averaged, ignoring 255):
- Overall accuracy: correct pixels / all valid pixels. Dominated by tree cover, so it is reported next to the majority-class baseline.
- IoU = TP / (TP + FP + FN) per class. mIoU is the unweighted mean, so rare classes count equally.
- Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R), which equals the per-class pixel Dice.
- Confusion matrix: rows are reference, columns are prediction.

**Forest-loss metrics:** precision, recall, F1 (= Dice) and IoU at the tuned threshold, plus a confusion matrix. **PR-AUC** (average precision) is threshold-free and appropriate for rare positives; a random classifier scores the positive prevalence. Accuracy is reported but is not meaningful at 1.5% prevalence.

**Reference baselines:** majority class for land cover (always "tree cover"), and prevalence for forest-loss PR-AUC. They show what "no learning" scores.

**Interpretation:** every score measures **agreement with the reference labels** (WorldCover 2021, Hansen GFC), which are themselves model-derived products with their own errors. They are not independently verified real-world accuracy.

## 7. Inference

`predict.py` reads the pre and post GeoTIFFs and applies the same preprocessing with the training normalisation stats. It cuts the scene into 128 px tiles with 32 px overlap (reflect-padding the edges), averages overlapping probabilities, and writes GeoTIFFs with the input CRS and transform:
- `landcover.tif`: uint8 classes 0-4 with a colour table, 255 = no data.
- `deforestation_prob.tif`: float32, -1 = no data.
- `deforestation_mask.tif`: uint8, 1 = loss at the validation-tuned threshold, 255 = no data.

It also writes a preview PNG and the area summary (km² and ha; 30 m pixel = 0.09 ha).
