# Results

All numbers below were produced by the final end-to-end run of `./run_pipeline.sh` (full mode) on 2026-11-24 and read from the files listed. Nothing is estimated or rounded beyond what is shown.

> **Scope of these scores.** They measure **agreement with the reference labels** (ESA WorldCover 2021 and Hansen GFC v1.12) on a held-out geographic test region. Those references are themselves model-derived products with their own errors. This is **not** independently verified real-world accuracy.

| Item | Run |
|---|---|
| Prepared data / split / patches | `data/processed/prep_*`, `data/splits/split_*`, `data/patches/patches_20261124_170753` (latest) |
| Model | `outputs/checkpoints/train_20261124_170757` (best epoch 24 of 32, early stopping) |
| Evaluation | `outputs/reports/evaluation/eval_20261124_171655`, figures in `outputs/plots/eval_20261124_171655/` |
| Full-scene prediction | `outputs/predictions/pred_20261124_171720` |

## 1. Training

| | Value |
|---|---|
| Train / val patches | 317 / 71 (128 x 128 px) |
| Epochs run / best | 32 / 24 (early stopping, patience 8) |
| Training time | 8.8 min on a 12-thread CPU (529.9 s) |
| Best epoch, train | total loss 1.268; land-cover mIoU 0.603; forest-loss F1@0.5 0.473 |
| Best epoch, validation | total loss 1.451; land-cover mIoU 0.581; forest-loss F1@0.5 0.473 |

Train and validation scores stay close, so there is no strong overfitting. Curves are in `outputs/plots/eval_20261124_171655/training_curves.png`.

## 2. Land cover (test region: 108 patches, 1,742,097 labelled pixels)

| Metric | Model | Majority-class baseline |
|---|---|---|
| Overall accuracy | **0.877** | 0.644 |
| Mean IoU | **0.561** | 0.129 |
| Mean Dice (= macro F1) | **0.669** | n/a |

| Class | IoU | Precision | Recall | F1 / Dice | Reference pixels |
|---|---|---|---|---|---|
| Tree cover | 0.890 | 0.956 | 0.929 | 0.942 | 1,121,364 |
| Cropland | 0.720 | 0.863 | 0.813 | 0.837 | 511,235 |
| Water | 0.771 | 0.815 | 0.933 | 0.870 | 44,609 |
| Built-up | 0.266 | 0.293 | 0.741 | 0.420 | 12,307 |
| Grassland | 0.159 | 0.219 | 0.369 | 0.275 | 52,582 |

Validation region (71 patches): OA 0.861, mIoU 0.581, mean Dice 0.695.

**Reading the results**
- **Strong:** tree cover, cropland and water (IoU 0.72-0.89) clearly beat the baseline.
- **Grassland is the weakest class.** Of the reference grassland pixels, 36.4% are predicted as cropland and 18.2% as tree cover; only 36.9% are correct (`confusion_landcover_test.png`). In dry-season composites the classes look alike, and WorldCover's own grassland class is uncertain in this region.
- **Built-up is over-predicted:** high recall (0.74) but low precision (0.29). This is a side effect of its large class weight (6.9), which pushes the model to find every town pixel at the cost of false alarms on bright bare surfaces.

## 3. Forest loss 2021-2023 (test region)

Threshold tuned on **validation only** (max F1): **0.43**.

| Metric | Test | Validation |
|---|---|---|
| Precision | **0.391** | 0.487 |
| Recall | **0.702** | 0.460 |
| F1 = Dice | **0.502** | 0.473 |
| IoU | **0.335** | 0.310 |
| PR-AUC (average precision) | **0.558** | 0.462 |
| Random PR-AUC (prevalence) | 0.0105 | 0.0166 |
| Positive (loss) pixels | 17,504 | 16,898 |
| TP / FP / FN | 12,284 / 19,128 / 5,220 | 7,775 / 8,202 / 9,123 |

- PR-AUC is about **53x the random level**, so the model has clearly learned to rank loss pixels.
- Accuracy (0.985) is reported only for completeness; it is meaningless at 1% prevalence.
- At threshold 0.5 the test F1 would have been 0.512. It was not chosen, because thresholds are only tuned on validation. The small gap is the normal cost of an honest hold-out.
- Test and validation differ (PR-AUC 0.558 vs 0.462). Different hill areas have different loss patterns, so expect roughly ±0.05 between regions.

**Qualitative inspection** (`examples_test.png`): the model finds the same clearings as Hansen and as visible in the post-image RGB. It draws them as smooth, compact blobs, whereas the reference is speckled (nearest-neighbour resampled from about 25 m). Part of the "false positives" are blob edges and pixels the reference ignores, so the true precision against reality is not known from these numbers.

## 4. Run-to-run reproducibility

Training the same configuration twice (Phase C run `train_20261124_164209` vs the final run) gave:

| | Phase C run | Final run |
|---|---|---|
| Best epoch / val loss | 24 / 1.4485 | 24 / 1.4512 |
| Threshold (val) | 0.46 | 0.43 |
| Test land-cover OA / mIoU | 0.877 / 0.559 | 0.877 / 0.561 |
| Test forest-loss F1 / PR-AUC | 0.511 / 0.558 | 0.502 / 0.558 |

The data, split and patches are bit-identical between runs: all 496 patch files and the normalisation statistics were compared byte for byte. Training differs slightly because TensorFlow CPU kernels are not fully deterministic. Variance is about ±0.01 on these metrics; the split itself (one geographic split) was not varied.

## 5. Full-scene maps (all 11,145,336 pixels, about 10,031 km²)

| Class | Predicted area (km²) | WorldCover 2021, mapped classes (km²) |
|---|---|---|
| Tree cover | 6,718.4 | 6,779.8 |
| Cropland | 2,146.3 | 2,118.2 |
| Water | 518.3 | 487.3 |
| Grassland | 464.0 | 368.8 |
| Built-up | 183.8 | 93.1 |

| Forest loss | Value |
|---|---|
| Predicted (p >= 0.43) | 287,290 px = **258.6 km²** (25,856 ha) |
| Hansen `loss_target == 1` | 150,715 px = 135.6 km² |
| Predicted loss on pixels Hansen ignores (older loss, low canopy, water) | 69,776 px = 62.8 km² |

**Important:** the full scene includes the training region, so these maps are not an evaluation. The F1-optimal threshold trades precision for recall (test recall 0.70 > precision 0.39), so the **predicted loss area is about 1.9x the Hansen area**, and built-up is about 2x. Use the maps to show *where* loss and land cover occur. Do not report these areas as area estimates without calibration (for example, a sample-based area estimate).

Outputs in `outputs/predictions/pred_20261124_171720/`: `landcover.tif`, `deforestation_prob.tif`, `deforestation_mask.tif` (EPSG:32646, 30 m, transform identical to the inputs), `preview.png`, `area_summary.json/.csv`.

## 6. Limitations specific to these results

- One region, one geographic split, one architecture. Results may not transfer elsewhere.
- Reference labels are imperfect: WorldCover grassland/cropland confusion, Hansen speckle at 30 m, and gross loss (jhum, plantation, river change) rather than deforestation.
- Only the multi-task model was trained. Single-task and shared-decoder baselines are implemented but not yet compared.
- Rare-class areas are biased upwards by class weighting and the F1-tuned threshold.

See [REPORT_SUPPORT.md](REPORT_SUPPORT.md) §13 for the full limitation list.
