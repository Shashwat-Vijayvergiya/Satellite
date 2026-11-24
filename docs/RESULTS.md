# Results

All numbers below were produced by runs on 2026-11-24 and read from the files listed. Nothing is estimated or rounded beyond what is shown.

> **Scope of these scores.** They measure **agreement with the reference labels** (ESA WorldCover 2021 and Hansen GFC v1.12) on a held-out geographic test region. Those references are themselves model-derived products with their own errors. This is **not** independently verified real-world accuracy.

**Final model = experiment E3** (bilinear upsampling, class-weight cap 3, NDVI/NBR change channels, overlapping train patches). It was chosen over the baseline and E1/E2 on **validation only** (§1), then evaluated on the test region **once**. The original baseline (git tag `baseline-v1`) is kept for comparison in §6.

| Item | Run |
|---|---|
| Prepared data / split | `data/processed/prep_20261124_170750`, `data/splits/split_20261124_170752` (shared by all models) |
| Patches | `data/patches/patches_20261124_200011` (496 base + 765 overlapping train patches) |
| Model | `outputs/checkpoints/exp_e3_20261124_200023` (best epoch 14 of 22, early stopping) |
| Evaluation | `outputs/reports/evaluation/eval_20261124_203857`, figures in `outputs/plots/eval_20261124_203857/` |
| Full-scene prediction | `outputs/predictions/pred_20261124_204028` |

## 1. Model selection (validation only)

Rule, fixed before any experiment ran (`configs/experiments/README.md`): the highest **validation land-cover mIoU + validation forest-loss PR-AUC** wins, and only the winner is evaluated on test. Each experiment builds on the previous one.

| Model | Change | Val mIoU | Val PR-AUC | Val loss F1 | **Score** |
|---|---|---|---|---|---|
| Baseline | 12 bands, transposed-conv decoder, class-weight cap 10 | 0.581 | 0.462 | 0.473 | 1.044 |
| E1 | + bilinear upsampling, class-weight cap 3 | 0.573 | 0.485 | 0.490 | 1.057 |
| E2 | + NDVI and NBR (pre, post, change) as 6 extra channels = 18 inputs | 0.591 | 0.478 | 0.486 | 1.069 |
| **E3** | + overlapping train patches (stride 64; 1,082 train patches instead of 317) | **0.628** | **0.502** | **0.501** | **1.130** |

Reports: `val_eval_20261124_191919` (baseline), `_194159` (E1), `_200009` (E2), `_203812` (E3).

- E1 helped forest loss (PR-AUC +0.022) but slightly lowered mIoU, mostly via built-up IoU (0.362 → 0.338).
- E2's index channels recovered land cover (+0.019 mIoU).
- E3 gave the largest step: more varied training crops, especially for the small built-up and water classes (val IoU 0.378 → 0.494 and 0.844 → 0.871).
- Differences of about ±0.01 are within run-to-run noise (§7). The E3 gain (+0.086 over the baseline) is well outside it.

## 2. Training (E3)

| | Value |
|---|---|
| Train / val patches | 1,082 (317 base + 765 overlapping) / 71 (128 x 128 px) |
| Parameters | 3,146,726 |
| Epochs run / best | 22 / 14 (early stopping, patience 8) |
| Training time | 37.6 min on a 12-thread CPU |
| Best epoch, train | total loss 1.133; land-cover mIoU 0.638; forest-loss F1@0.5 0.511 |
| Best epoch, validation | total loss 1.321; land-cover mIoU 0.628; forest-loss F1@0.5 0.492 |
| Class weights (tree, grass, crop, built, water) | 0.66, 2.85, 1.33, 3.00 (capped), 2.25 |

The overlapping patches lie fully inside kept train cells, so they never touch the buffer, validation or test areas. The normalisation statistics and class weights are computed from the 317 non-overlapping train patches only. Overlapping crops are not independent samples; they act as stronger translation augmentation.

## 3. Land cover (test region: 108 patches, 1,742,097 labelled pixels)

| Metric | E3 (final) | Baseline | Majority-class baseline |
|---|---|---|---|
| Overall accuracy | **0.889** | 0.877 | 0.644 |
| Mean IoU | **0.600** | 0.561 | 0.129 |
| Mean Dice (= macro F1) | **0.708** | 0.669 | n/a |

| Class | IoU | Precision | Recall | F1 / Dice | Baseline IoU | Reference pixels |
|---|---|---|---|---|---|---|
| Tree cover | 0.900 | 0.962 | 0.934 | 0.948 | 0.890 | 1,121,364 |
| Cropland | 0.744 | 0.866 | 0.841 | 0.853 | 0.720 | 511,235 |
| Water | 0.789 | 0.810 | 0.968 | 0.882 | 0.771 | 44,609 |
| Built-up | 0.398 | 0.506 | 0.650 | 0.569 | 0.266 | 12,307 |
| Grassland | 0.169 | 0.231 | 0.387 | 0.290 | 0.159 | 52,582 |

Validation region (71 patches): OA 0.876, mIoU 0.628, mean Dice 0.738.

**Reading the results**
- **Every class improved on test.** The largest gain is in built-up (IoU +0.13): precision rose from 0.29 to 0.51 once its class weight was capped at 3 instead of 6.9.
- **Strong classes:** tree cover, cropland and water (IoU 0.74–0.90).
- **Grassland is still the weakest class.** Of the reference grassland pixels, 35.0% are predicted as cropland and 16.5% as tree cover; only 38.7% are correct (`confusion_landcover_test.png`). The classes look alike in dry-season composites, and WorldCover's own grassland label is uncertain here. This is a label/data limit more than a model limit.

## 4. Forest loss 2021-2023 (test region)

Threshold tuned on **validation only** (max F1): **0.70**. The baseline's was 0.43. E3 is more confident, so its F1-optimal cut is higher.

| Metric | E3 test | E3 validation | Baseline test |
|---|---|---|---|
| Precision | **0.449** | 0.503 | 0.391 |
| Recall | **0.694** | 0.499 | 0.702 |
| F1 = Dice | **0.545** | 0.501 | 0.502 |
| IoU | **0.375** | 0.334 | 0.335 |
| PR-AUC (average precision) | **0.584** | 0.502 | 0.558 |
| Random PR-AUC (prevalence) | 0.0105 | 0.0166 | 0.0105 |
| Positive (loss) pixels | 17,504 | 16,898 | 17,504 |
| TP / FP / FN | 12,140 / 14,911 / 5,364 | 8,435 / 8,328 / 8,463 | 12,284 / 19,128 / 5,220 |

- PR-AUC is about **56x the random level**.
- Compared with the baseline, E3 makes 22% fewer false alarms (14,911 vs 19,128) at nearly the same recall.
- Accuracy (0.988) is reported only for completeness; it is meaningless at 1% prevalence.
- Test and validation differ (PR-AUC 0.584 vs 0.502). Different hill areas have different loss patterns, so expect roughly ±0.05 between regions.

**Qualitative inspection** (`examples_test.png`): the model finds the same clearings as Hansen and as visible in the post-image RGB. It draws them as smooth, compact blobs, whereas the reference is speckled (nearest-neighbour resampled from about 25 m). Some of the "false positives" are blob edges and pixels the reference ignores, so true precision against reality cannot be read from these numbers.

## 5. Full-scene maps (all 11,145,336 pixels, about 10,031 km²)

| Class | E3 predicted (km²) | Baseline predicted (km²) | WorldCover 2021, mapped classes (km²) |
|---|---|---|---|
| Tree cover | 6,790.2 | 6,718.4 | 6,779.8 |
| Cropland | 2,164.6 | 2,146.3 | 2,118.2 |
| Water | 550.8 | 518.3 | 487.3 |
| Grassland | 407.2 | 464.0 | 368.8 |
| Built-up | 118.1 | 183.8 | 93.1 |

| Forest loss | Value |
|---|---|
| Predicted (p >= 0.70) | 276,100 px = **248.5 km²** (24,849 ha); baseline 258.6 km² |
| Hansen `loss_target == 1` | 150,715 px = 135.6 km² |
| Predicted loss on pixels Hansen ignores (older loss, low canopy, water) | 70,353 px = 63.3 km² |

**Important:** the full scene includes the training region, so these maps are not an evaluation.
- **Built-up:** its over-prediction fell from about 2.0x to 1.3x WorldCover.
- **Forest loss:** the predicted area is still about **1.8x the Hansen area**, because the F1-optimal threshold favours recall (test recall 0.69 > precision 0.45). Use the maps to show *where* loss occurs. Do not report these areas as area estimates without calibration (for example, a sample-based area estimate).

Outputs in `outputs/predictions/pred_20261124_204028/`:
- `landcover.tif`, `deforestation_prob.tif`, `deforestation_mask.tif`: EPSG:32646, 30 m, transform identical to the inputs.
- `preview.png`, `area_summary.json/.csv`.

## 6. Baseline model (for comparison)

The baseline is `outputs/checkpoints/train_20261124_170757`; reproduce it with `configs/experiments/baseline.yaml` or git tag `baseline-v1`.
- **Inputs and decoder:** 12 bands, transposed-conv decoder, class-weight cap 10.
- **Data:** 317 train patches.
- **Training:** best epoch 24 of 32, 8.8 min.
- **Test results:**
  - Land cover: OA 0.877, mIoU 0.561.
  - Forest loss: F1 0.502, PR-AUC 0.558 (threshold 0.43).
- **Files:** evaluation `eval_20261124_171655`, prediction `pred_20261124_171720`.

## 7. Run-to-run reproducibility

The baseline configuration was trained twice (Phase C run `train_20261124_164209` vs `train_20261124_170757`):

| | Phase C run | Second run |
|---|---|---|
| Best epoch / val loss | 24 / 1.4485 | 24 / 1.4512 |
| Threshold (val) | 0.46 | 0.43 |
| Test land-cover OA / mIoU | 0.877 / 0.559 | 0.877 / 0.561 |
| Test forest-loss F1 / PR-AUC | 0.511 / 0.558 | 0.502 / 0.558 |

The data, split and patches are bit-identical between runs: all 496 patch files and the normalisation statistics were compared byte for byte. Training differs slightly because TensorFlow CPU kernels are not fully deterministic. Variance is about ±0.01 on these metrics. E3 was trained once; the split itself (one geographic split) was not varied.

## 8. Limitations specific to these results

- One region, one geographic split, one training run per experiment. Results may not transfer elsewhere.
- Three experiments were compared on the same validation region. Selecting the best of three slightly flatters its validation score. The test score was measured once afterwards and is not affected.
- Reference labels are imperfect: WorldCover grassland/cropland confusion, Hansen speckle at 30 m, and gross loss (jhum, plantation, river change) rather than deforestation.
- Only the multi-task model was trained. Single-task and shared-decoder baselines are implemented but not compared.
- Forest-loss area is biased upwards by the F1-tuned threshold.

See [REPORT_SUPPORT.md](REPORT_SUPPORT.md) §13 for the full limitation list.
