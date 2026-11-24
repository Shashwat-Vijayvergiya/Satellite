# Report support: Satellite Image Segmentation for Land Cover and Deforestation Mapping Using Deep Learning

Material for the project report and viva. Technical details are in [DATASET.md](DATASET.md), [METHODOLOGY.md](METHODOLOGY.md) and [RESULTS.md](RESULTS.md); measured numbers come only from those documents.

## 1. Proposed work

A reproducible deep-learning system that, from two dry-season Sentinel-2 composites (2020-21 and 2023-24), produces two pixel-level maps for a 90 x 111 km region of Assam: a **5-class land-cover map** and a **forest-loss map** for 2021-2023. Both come from one multi-task U-Net running on an ordinary laptop CPU.

## 2. Problem statement

Assam's forests, especially in the Karbi Anglong hills, are under pressure from shifting cultivation, plantation expansion, logging and river dynamics. Official forest-cover reports are biennial and aggregate to districts. Global products (ESA WorldCover, Hansen GFC) exist but are produced separately, for different years, with different methods. They do not give a single, consistent, regionally adapted view of *what the land is* and *where forest was lost*.

The problem is to learn, from freely available satellite imagery, a model that segments land cover and recent forest loss at the pixel level for a specific Indian region. The model must be validated honestly against spatially independent data and run without cloud infrastructure.

## 3. Objectives

1. Assemble and **strictly validate** an aligned 30 m dataset: Sentinel-2 composites, ESA WorldCover 2021 and Hansen GFC.
2. Design a leakage-free **geographic train/validation/test split** with enough forest-loss examples in each region.
3. Implement a **multi-task U-Net** in TensorFlow/Keras for land-cover and forest-loss segmentation.
4. Handle **severe class imbalance** (forest loss is 1.5% of labelled pixels) and **unreliable labels**, using masked, weighted losses.
5. **Evaluate per task** on a held-out region with appropriate metrics (IoU, F1, PR-AUC), tuning the decision threshold on validation only.
6. Produce **georeferenced maps** (GeoTIFF) and area statistics for the whole region, plus a demonstration interface.

## 4. Existing system

| Existing approach | Limitation for this problem |
|---|---|
| Manual visual interpretation of imagery / field surveys | Slow, costly, not repeatable over 10,000 km² |
| India State of Forest Report (biennial) | District aggregates, coarse timing, no per-pixel change map for users |
| ESA WorldCover (2020, 2021) | Land cover only, global model, single year; no change information |
| Hansen Global Forest Change | Forest loss only, global model, no land-cover context |
| Classical pixel classifiers (Random Forest, SVM, NDVI thresholds) | Treat pixels independently and ignore spatial context; one model per task; sensitive to thresholds |

## 5. Proposed system

- **Input:** bi-temporal Sentinel-2 composites (pre 2020-21, post 2023-24), 6 spectral bands each, 12 channels.
- **Model:** multi-task U-Net with a shared encoder and two decoders. It learns spatial context (texture, shape, neighbourhood) rather than isolated pixel values, and predicts both tasks in one pass.
- **Training signal:** the existing global products act as reference labels, with explicit masking where they are unreliable (rare classes, stale labels after forest loss, pre-2021 loss, water).
- **Engineering:**
  - strict validation that blocks bad data;
  - versioned, lineage-checked datasets;
  - leakage-free spatial split;
  - fail-fast pipeline runners;
  - unit tests;
  - georeferenced outputs;
  - an optional Streamlit demo.

## 6. Methodology (summary)

1. **Data acquisition:** Google Earth Engine exports on a common 30 m UTM grid (done before this work).
2. **Validation:** grid alignment, band counts and types, statistics, codes, and a pixel-wise recomputation of the forest-loss target. Hard errors stop the pipeline.
3. **Preprocessing:** stack and clip bands; robust normalisation from training data only; remap WorldCover to 5 classes; set ignore masks.
4. **Spatial split:** 512 px blocks assigned to train/val/test before patching, with a buffer and constraints on forest-loss and class counts.
5. **Patching:** 128 x 128 non-overlapping patches with a manifest.
6. **Training:** multi-task U-Net with masked weighted cross-entropy (land cover) and masked weighted BCE + Dice (forest loss), early stopping and checkpointing.
7. **Evaluation:** threshold tuned on validation; per-task metrics on the test region; comparison with trivial baselines.
8. **Inference:** overlapping-tile prediction of the full scene, written as GeoTIFFs and area summaries.

## 7. System architecture

```
┌───────────────────── Data layer ─────────────────────┐
│ GEE exports → validation → prepared arrays → spatial │
│ split → patches (+ manifest, train-only stats)       │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────── Model layer ─────────────────────┐
│ tf.data loader (lazy, augmentation) → multi-task     │
│ U-Net → masked losses → Keras training with callbacks│
└──────────────────────────┬───────────────────────────┘
                           ▼
┌───────────────── Evaluation & output ────────────────┐
│ threshold on val → test metrics + plots → full-scene │
│ tiled inference → GeoTIFFs, PNG, area summary → demo │
└──────────────────────────────────────────────────────┘
Cross-cutting: config.yaml (single source of truth), run folders with
metadata/lineage, fail-fast CLIs, unit tests.
```

## 8. Data flow

Raw GeoTIFFs (4 rasters + AOI) → `validation_report.json` → `image.npy` (12 ch) + `landcover.npy` + `deforest.npy` → `windows.csv` (split) → `*.npz` patches + `manifest.csv` + `band_stats.json` → `best.keras` → `metrics.json` and plots → `landcover.tif`, `deforestation_prob.tif`, `deforestation_mask.tif`, `area_summary.csv`. The diagram is in METHODOLOGY.md §1.

## 9. Algorithms

- **Spatial block assignment:** seeded random search over whole-block assignments; reject infeasible ones (val/test forest-loss < 10,000 px, any class < 10,000 px in any split); minimise the deviation from 70/15/15 in both patch and forest-loss shares.
- **Robust normalisation:** per-channel percentile clipping (1-99%) and standardisation, using training pixels only.
- **U-Net forward pass:** 4 encoder levels (3x3 convolutions, batch norm, ReLU, max-pooling), bottleneck, 4 decoder levels (transposed-convolution upsampling + skip concatenation), with a 1x1 softmax head and a 1x1 sigmoid head.
- **Masked weighted cross-entropy:** L = -Σ w_y log p_y over valid pixels / Σ w_y over valid pixels.
- **Masked BCE + soft Dice:** L = mean(-(w⁺ y log p + (1-y) log(1-p))) + 1 - (2 Σ p y + 1)/(Σ p + Σ y + 1), over valid pixels.
- **Adam optimisation** with early stopping and learning-rate reduction on plateau.
- **Threshold selection:** maximise F1 over a grid of thresholds on validation pixels.
- **Overlapping-tile inference:** reflect padding, 128 px tiles with 32 px overlap, probability averaging.

## 10. Model architecture

Shared encoder 16-32-64-128 filters, bottleneck 256 with dropout 0.2, and two decoders 128-64-32-16. Outputs: softmax(5) land cover and sigmoid(1) forest loss. Total 2,710,662 parameters. Full table in METHODOLOGY.md §3.

## 11. Evaluation metrics

- **Land cover:** overall accuracy, per-class IoU, mIoU, mean Dice, per-class precision/recall/F1, confusion matrix, and the majority-class baseline.
- **Forest loss:** precision, recall, F1, IoU, Dice, PR-AUC, confusion matrix, and threshold analysis. Accuracy alone is not used, because positives are rare.

Definitions are in METHODOLOGY.md §6.

## 12. Expected and obtained outcomes

- **Expected:** a validated, reproducible pipeline; land-cover maps that clearly beat a majority-class baseline; forest-loss detection far better than chance despite 1.5% prevalence; GeoTIFF maps for GIS use.
- **Obtained:** see [RESULTS.md](RESULTS.md) for the measured test-region scores, the full-scene area estimates, and their interpretation.

## 13. Limitations

1. **Temporal mismatch:** WorldCover is a 2021 product, closest to the pre composite and not matched to the post composite. Forest loss within about 2 months of the composite windows leaks into the "before" or "after" image.
2. **Resolution of the forest-loss labels:** Hansen was resampled nearest-neighbour from about 25 x 28 m geographic pixels to 30 m UTM. The labels are speckled and no finer than 30 m.
3. **Gross loss, not deforestation:** the target includes plantation harvest, fire, flood, river-bank change and shifting cultivation, not only permanent conversion.
4. **Reference labels are model-derived:** WorldCover and Hansen have their own errors, so the scores measure agreement with them, not ground truth.
5. **Strong class imbalance:** forest loss is 1.49% of labelled pixels; built-up 0.93% and grassland 3.7% of the scene. The rare classes are the hardest to learn.
6. **Single region and single split:** one 90 x 111 km scene, with one geographic split. Results may not transfer to other regions, and run-to-run variance is only partly measured.
7. **Classes left out:** shrubland, bare/sparse vegetation and herbaceous wetland (0.48% together) are ignored, so the model cannot predict them.
8. **Dry-season composites only:** grassland and cropland look alike in winter imagery, which limits the separation of these classes.

## 14. Future scope

- Add more regions and years, and repeat with several split seeds to report confidence intervals.
- Use time series (monthly composites or Sentinel-1 radar, which sees through cloud) to separate grassland from cropland and to date loss events.
- Validate against independently interpreted samples (high-resolution imagery or field data) to estimate real-world accuracy.
- Compare against Random Forest and single-task U-Net baselines, and test lighter or pretrained encoders.
- Distinguish loss drivers (jhum, plantation, river erosion) as a multi-class change task.
- Deploy the demo as a small web map with GeoTIFF tiles.
