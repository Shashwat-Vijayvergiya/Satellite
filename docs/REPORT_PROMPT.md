# Prompt for an LLM: write the project report

Copy everything below the line into the LLM. Before sending, replace the `<<...>>` placeholders on the title page (name, roll number, guide, department, institution, year). The LLM cannot see the project figures: it inserts numbered figure placeholders, and you paste the PNG files listed in Part D afterwards.

---

## PART A: ROLE AND TASK

You are an expert technical writer and remote-sensing / deep-learning academic. Write a complete **final-year undergraduate project report** for the project described in Part C. The report must be ready to paste into a word processor (use Markdown headings, numbered sections, tables and figure/table captions).

**Title:** Satellite Image Segmentation for Land Cover and Deforestation Mapping Using Deep Learning

## PART B: NON-NEGOTIABLE RULES

1. **Use only the facts in Part C.** Do not invent numbers, results, datasets, experiments, hardware, comparisons with other papers, accuracy claims or citations. If something is not in Part C, do not state it. Do not add results for experiments that were not run (e.g. Random Forest, single-task U-Net, other architectures).
2. **Results measure agreement with reference labels** (ESA WorldCover 2021 and Hansen Global Forest Change), **not independently verified real-world accuracy.** State this in the Abstract, Results and wherever accuracy is discussed.
3. Call the forest-loss target "gross forest-cover loss" (stand-replacement loss); explain once that it is not strictly deforestation. The project title keeps the word "deforestation".
4. **Page budget (A4, 12 pt, 1.5 spacing, about 450-500 words per text page; tables/figures count toward pages). Maximum 21 pages in total:**

| # | Section | Pages |
|---|---|---|
| 1 | Title page | 1 |
| 2 | Table of contents | 1 |
| 3 | Abstract | 1 |
| 4 | Problem Statement | 1 |
| 5 | Proposed Work (with system design and workflow) | 4-5 |
| 6 | Detailed Description (every step of the workflow, how it was implemented) | 7 |
| 7 | Datasets and Exploratory Data Analysis (EDA) | 1-3 (at least 1 page of EDA) |
| 8 | Results | 1 |
| 9 | References | 1 |

5. **Figures:** insert placeholders exactly in this form, at the right place, with a caption: `[Figure N: <caption> — insert file <path>]`. Use only the files listed in Part D for real results. For design diagrams (system architecture, workflow flowchart, U-Net architecture) provide the diagram as **Mermaid code** inside a code block, followed by a caption, so the student can render it. Number figures and tables consecutively.
6. **References:** use only the references in Part E, in IEEE numeric style ([1], [2], ...), cited in the text where relevant. Do not add any other references.
7. Write in formal academic English, third person or "we", past tense for work done. No marketing language. Keep the Results section to one page: put the most important tables there and refer to other numbers briefly.
8. Use the exact numbers from Part C (keep the given decimals). Report areas in km² and ha as given.

## PART C: COMPLETE PROJECT FACT SHEET

### C1. Overview
- Goal: a reproducible deep-learning system that segments **two pixel-level outputs** from bi-temporal Sentinel-2 imagery: (A) **land cover**, 5 classes; (B) **forest-cover loss 2021-2023**, binary.
- One **multi-task U-Net** (shared encoder, two task-specific decoders), implemented in **Python 3.12, TensorFlow 2.21 / Keras 3.15**, trained on a **CPU** (no GPU, no cloud).
- Study area: **Kaziranga / Karbi Anglong region, Assam, India.** North: Brahmaputra floodplain (river, sandbars, Kaziranga grasslands, tea and rice). South: forested Karbi Anglong hills, where most forest loss occurs (shifting cultivation "jhum" is common there).
- Extent: lon 92.9998°E–93.9080°E, lat 25.7970°N–26.8031°N; **EPSG:32646 (UTM 46N), 30 m pixels, 3009 x 3704 px = 11,145,336 pixels, about 90.3 x 111.1 km ≈ 10,031 km².**

### C2. Problem context (for Problem Statement)
- Assam's forests (especially Karbi Anglong) face pressure from shifting cultivation, plantations, logging and river dynamics.
- National forest reporting (India State of Forest Report) is biennial and summarised at district level, with no per-pixel change map for users.
- Global products exist but are separate and not regionally adapted: **ESA WorldCover** gives land cover for a single year with no change information; **Hansen Global Forest Change** gives forest loss but no land-cover context. Manual interpretation and field surveys are slow and costly over 10,000 km². Classical pixel classifiers (Random Forest, SVM, NDVI thresholds) ignore spatial context and need one model per task.
- Problem: learn from free satellite imagery a model that maps land cover and recent forest loss per pixel for this region, validated honestly on spatially independent data, runnable on an ordinary laptop.

### C3. Objectives
1. Assemble and strictly validate an aligned 30 m dataset (Sentinel-2 composites, ESA WorldCover 2021, Hansen GFC).
2. Create a leakage-free geographic train/validation/test split with enough forest-loss examples in each region.
3. Implement a multi-task U-Net in TensorFlow/Keras.
4. Handle severe class imbalance (forest loss ≈ 1.5% of labelled pixels) and unreliable labels with masked, weighted losses.
5. Evaluate each task on a held-out region with suitable metrics (IoU, F1, PR-AUC), tuning the decision threshold on validation only.
6. Produce georeferenced GeoTIFF maps and area statistics for the whole region, plus a demonstration interface.

### C4. Data sources (exported from Google Earth Engine on one common grid)
| File | Content | Type | Bands |
|---|---|---|---|
| s2_pre_2020-21.tif | Sentinel-2 L2A Harmonized **median composite, 1 Nov 2020 – 28 Feb 2021** | UInt16 | B2, B3, B4, B8, B11, B12, nobs |
| s2_post_2023-24.tif | Same, **1 Nov 2023 – 29 Feb 2024** | UInt16 | same |
| worldcover_2021_30m.tif | **ESA WorldCover 2021 v200**, 10 m aggregated to 30 m by **mode** | UInt8 | class map |
| hansen_labels_30m.tif | **Hansen GFC v1.12 (2000–2024)** bands + derived target, resampled **nearest-neighbour** to 30 m | UInt8 | treecover2000, lossyear, datamask, loss_target |
| aoi.geojson | study-area polygon and export metadata | GeoJSON | — |

- Sentinel-2 values = surface reflectance × 10000. B2 blue, B3 green, B4 red, B8 NIR (native 10 m); B11, B12 SWIR (native 20 m); all averaged to 30 m. Cloud masking with **Cloud Score+ (cs_cdf ≥ 0.60)**, then per-pixel median. `nobs` = number of valid observations per pixel.
- Dry-season (winter) windows were chosen for low cloud cover.
- Hansen native grid is geographic (0.00025°, ≈25 x 28 m here); labels are therefore only reliable at 30 m (no 10 m claims).
- **loss_target** definition: **1** = land pixel with loss year 2021–2023 and tree cover in 2000 ≥ 30%; **0** = land with no loss; **255 = ignore** (water/no data, loss 2001–2020, loss 2024, and 2021–23 loss on pixels < 30% canopy).

### C5. Temporal strategy (must be explained)
- WorldCover (2021) is not exactly time-matched to either composite; the pre composite (dry season 2020/21) is the closest match.
- Forest loss 2021–2023 lies between the two composites. Loss within about 2 months of the composite windows can leak (Jan–Feb 2021 into the pre image; Jan–Feb 2024 changes, labelled 2024 and ignored, into the post image).
- Land-cover labels are stale where forest was lost after 2021, so those pixels (loss_target = 1) are **ignored in the land-cover loss and metrics**.

### C6. Data validation results (all measured; the pipeline stops with an error if any hard check fails)
Checks: file existence and full read (truncation), split-export tiles, band counts (7, 7, 1, 4), data types, CRS, transform, bounds and **exact grid alignment** of all rasters, NaN/inf, no-data conventions, per-band statistics (a constant band is a hard error), reflectance plausibility, WorldCover class IDs and shares, Hansen code ranges, a **pixel-by-pixel recomputation of loss_target**, and consistency with aoi.geojson. SHA-256 hashes of raw files are recorded; later stages refuse to run if files change.

Findings:
- All 4 rasters share exactly EPSG:32646, transform (30, 0, 499980, 0, −30, 2964630), 3009 x 3704 px. AOI metadata matches; AOI bounding box equals raster footprint (difference 1e-13°).
- **100% of pixels valid in both dates** (no zero-reflectance pixels, nobs never 0; nobs min 9 pre, 4 post).
- Only **14 pixels** in the whole scene exceed 10000 DN (bright targets / residual cloud) → clipped to 10000. No constant band.
- loss_target recomputation: **0 disagreeing pixels**. Counts exactly matched the Earth Engine diagnostics.

### C7. EDA facts (use for the Datasets/EDA section; build tables and suggest charts)

**Sentinel-2 band statistics (all pixels):**
| Band | Pre mean | Pre std | Pre p1–p99 | Post mean | Post std | Post p1–p99 |
|---|---|---|---|---|---|---|
| B2 | 539.7 | 306.9 | 180–1970 | 496.8 | 310.2 | 162–2123 |
| B3 | 732.0 | 335.1 | 303–2264 | 689.5 | 337.1 | 275–2398 |
| B4 | 663.5 | 442.8 | 186–2419 | 590.3 | 443.5 | 155–2565 |
| B8 | 2606.6 | 650.4 | 441–4601 | 2753.9 | 702.2 | 347–4818 |
| B11 | 2003.2 | 621.7 | 181–3869 | 1978.1 | 610.9 | 159–3926 |
| B12 | 1191.6 | 681.5 | 148–4140 | 1148.2 | 662.8 | 140–4187 |
| nobs | 22.6 | 10.3 | 15–68 | 20.7 | 10.0 | 13–65 |

**WorldCover 2021 class distribution (whole scene):**
| ID | Class | Pixels | Share | Used as |
|---|---|---|---|---|
| 10 | Tree cover | 7,671,037 | 68.83% | class 0 |
| 40 | Cropland | 2,359,033 | 21.17% | class 2 |
| 80 | Permanent water | 541,483 | 4.86% | class 4 |
| 30 | Grassland | 417,084 | 3.74% | class 1 |
| 50 | Built-up | 103,453 | 0.93% | class 3 |
| 60 | Bare/sparse vegetation | 44,606 | 0.40% | ignored |
| 20 | Shrubland | 5,060 | 0.05% | ignored |
| 90 | Herbaceous wetland | 3,580 | 0.03% | ignored |
(No class 0/no-data, 70, 95 or 100 present.)

**Hansen / forest-loss target:**
- treecover2000: 0–100%, mean 43.5%. datamask: land 11,004,848 px; permanent water 140,488 px.
- loss_target: **positive 150,715 (1.49% of labelled pixels); negative 9,953,974; ignore 1,040,647.**
- Positives by year: 2021: 54,386; 2022: 48,965; 2023: 47,364.
- Ignore reasons: water/no data 140,488; loss 2001–2020 819,303; loss 2024 46,256; 2021–23 loss below 30% canopy 34,600.
- Of the loss pixels, 137,871 (91.5%) are WorldCover tree cover, 7,271 grassland, 5,514 cropland.
- WorldCover water (541,483 px) vs Hansen water (140,488 px): 435,337 WorldCover-water pixels are Hansen land — the seasonally shifting Brahmaputra channels and sandbars.
- Key EDA insights: extreme imbalance for forest loss (≈1:66 positive:negative) and for built-up (0.93%); loss concentrated in the southern hills; dry-season grassland and cropland are spectrally similar; NIR (B8) is slightly higher in the post image.

### C8. Proposed system design and workflow (for Proposed Work; draw as flowchart + architecture diagram)
Pipeline of 7 command-line stages, all settings in one `configs/config.yaml`:

1. **inspect_rasters** – strict validation (C6); any error → exit code 1, pipeline stops.
2. **prepare_dataset** – stack pre + post bands (12 channels; NDVI/NBR index channels are added at load time → 18 model inputs), clip reflectance, validity mask, remap WorldCover to 5 classes, build both label maps with ignore rules.
3. **split_dataset** – geographic block split into train/val/test **before** cutting patches.
4. **create_patches** – 128 x 128 patches (plus overlapping training patches at stride 64 inside the training region), normalisation statistics from the training region only, manifest.
5. **train** – multi-task U-Net with masked, weighted losses, early stopping, checkpoints. Improvement experiments are trained separately and compared on validation only (model selection).
6. **evaluate** – threshold tuned on validation, metrics on the held-out test region.
7. **predict** – full-scene inference to GeoTIFF maps, preview image and area summary.

Engineering guarantees (part of the design): fail-fast scripts (exit code 1 on any error) and pipeline runners that stop at the first failed stage; every stage writes a timestamped, versioned run folder with metadata (source files and hashes, configuration, parent run IDs); later stages refuse stale, incomplete or mismatched inputs; unchanged stages are reused via a content fingerprint; fixed random seed 42; 62 automated unit tests (synthetic data only inside tests); optional Streamlit web demo.

System architecture layers: Data layer (validation → prepared arrays → spatial split → patches) → Model layer (tf.data loader with augmentation → multi-task U-Net → masked losses → training) → Evaluation and output layer (validation threshold → test metrics and plots → full-scene tiled inference → GeoTIFFs, PNG, area summary → demo). Cross-cutting: single config file, run metadata/lineage, tests.

### C9. Implementation details per step (for Detailed Description)

**Preprocessing**
- Input = pre [B2, B3, B4, B8, B11, B12] + post [same] = **12 channels**. The `nobs` band was **dropped** as an input (configurable): validation found no missing pixels, and nobs (4–84) mainly reflects satellite orbit overlap, a location artefact the model could exploit. Validity still requires non-zero reflectance and nobs > 0 in both dates.
- **Spectral indices** (added in experiment E2, kept in the final model): NDVI = (B8−B4)/(B8+B4) and NBR = (B8−B12)/(B8+B12), each for the pre date, the post date and their change (post − pre) = 6 extra channels, computed from raw values when patches are loaded. The model input is therefore **18 channels** (12 normalised bands + 6 indices). NBR change (dNBR) is a standard burn/clearing indicator; NDVI change captures canopy loss.
- Reflectance clipped at 10000 DN.
- **Normalisation:** per channel, clip to the 1st–99th percentile of the **training region**, then standardise with the training mean and standard deviation (computed on 5,193,728 training pixels only, to avoid leakage). Patches are stored as raw values and normalised when loaded.
- **Label remapping:** WorldCover 10, 30, 40, 50, 80 → classes 0–4 (tree cover, grassland, cropland, built-up, water); 0, 20, 60, 90 → 255 (ignore; too rare: 0.48% together). Land-cover label also set to 255 where loss_target = 1. Forest-loss label = loss_target (1/0/255).
- Prepared label counts: tree 7,533,166; grassland 409,813; cropland 2,353,519; built-up 103,450; water 541,474; ignore 203,914.

**Spatial split (prevents spatial leakage)**
- Scene divided into a 28 x 23 grid of 128-px patch cells (644 cells; partial edge cells dropped), grouped into **42 blocks of 512 px (4 x 4 cells)**. Whole blocks are assigned to train/val/test before any patch is extracted.
- **Buffer of 1 patch:** train cells adjacent to val/test blocks and val cells adjacent to test blocks are dropped (148 cells).
- Assignment by a seeded search (seed 42, 5,000 candidates, 386 feasible) with constraints: ≥ 10,000 forest-loss pixels in validation and in test; ≥ 10,000 pixels of every land-cover class in every split; closest to 70/15/15 for both patch counts and forest-loss pixels. (A first version that balanced only patch counts gave train just 58% of loss pixels; adding the loss-pixel balance fixed this.)

| Split | Patches | Loss px | Tree | Grass | Crop | Built-up | Water |
|---|---|---|---|---|---|---|---|
| Train | 317 | 79,703 | 3,645,486 | 194,775 | 902,276 | 33,276 | 314,369 |
| Validation | 71 | 16,898 | 762,491 | 51,619 | 282,192 | 15,074 | 28,189 |
| Test | 108 | 17,504 | 1,121,364 | 52,582 | 511,235 | 12,307 | 44,609 |

**Patches:** 128 x 128 px (3.84 km), stride 128 (no overlap), 496 base patches stored as .npz with a manifest CSV (patch ID, split, location, per-class pixel counts, forest-loss pixel count, valid pixels). **Final model (experiment E3):** 765 extra *overlapping* training patches at stride 64 (50% overlap), each lying entirely inside kept training cells so they never touch the buffer, validation or test regions → 1,082 training patches. Normalisation statistics and class weights are computed from the 317 non-overlapping training patches only. Validation and test patches are unchanged.

**Data pipeline:** tf.data reads patches lazily (memory-efficient); training augmentation = random horizontal/vertical flips and 90° rotations applied identically to image and both label maps.

**Model: multi-task U-Net**
- Why U-Net: per-pixel labels need both context (encoder downsampling) and fine detail (decoder upsampling with skip connections that restore field edges, river banks, clearing outlines); works well with small datasets (hundreds of patches) and is CPU-friendly.
- Why multi-task: both tasks rely on the same spectral/texture cues; a shared encoder lets each task benefit from the other and halves encoder cost.
- Architecture: input 128 x 128 x 18 → encoder 4 levels, each two 3x3 convolutions + batch normalisation + ReLU, then 2x2 max-pooling, filters **16, 32, 64, 128** → bottleneck **256 filters** (8 x 8) with dropout 0.2 → **two separate decoders**, each 4 levels of **bilinear upsampling x2 + 3x3 convolution** (avoids checkerboard artefacts of transposed convolutions) + skip concatenation + two 3x3 convolutions (128, 64, 32, 16 filters) → heads: **1x1 conv softmax (5 classes)** for land cover and **1x1 conv sigmoid (1 channel)** for forest loss.
- Parameters: encoder 297,120; bottleneck 886,784; land-cover decoder + head 981,445; forest-loss decoder + head 981,377; **total 3,146,726**. (The baseline model — 12 inputs, transposed-convolution upsampling — had 2,710,662.) (A lighter shared-decoder variant — 1,946,902 parameters with the baseline settings — and a single-task U-Net are implemented but were not trained/evaluated.)

**Losses and class imbalance**
- All losses ignore label 255 and are normalised by the number of valid pixels in the batch.
- Land cover: **class-weighted cross-entropy**; weights = (1/frequency)^0.5 from training pixels, rescaled to mean 1, clipped to [0.2, 3] (raw inverse frequency would weight built-up about 110x tree cover and destabilise training). Weights: tree 0.659, grassland 2.852, cropland 1.325, built-up 3.000 (capped), water 2.245. The baseline used a cap of 10 (built-up weight 6.901), which made built-up heavily over-predicted (precision 0.29); lowering the cap to 3 was experiment E1.
- Forest loss: **positive-weighted binary cross-entropy + soft Dice loss**; positives are 1.70% of training pixels (79,703 of 4,676,214); positive weight = sqrt(neg/pos) = **7.59** (cap 10). Focal + Dice is available as an option.
- Total loss = 1.0 x land-cover loss + 1.0 x forest-loss loss.
- Formulas: weighted CE L = −Σ w_y log p_y / Σ w_y over valid pixels; BCE+Dice L = mean(−(w⁺ y log p + (1−y) log(1−p))) + 1 − (2Σpy + 1)/(Σp + Σy + 1).

**Training**
- Adam, learning rate 0.001, ReduceLROnPlateau (x0.5 after 3 stagnant epochs, min 1e-5), batch size 8, up to 40 epochs, early stopping on validation loss (patience 8, best weights restored), checkpoints of best and last model, resume support, per-epoch logging of per-task losses and metrics.
- A **smoke test** (16 patches per split, 1 epoch) runs the whole pipeline end to end in about 1 minute before full training.
- Hardware: ordinary 12-thread laptop CPU, no GPU. Speed ≈ 0.6 s per batch of 8 (≈ 0.5 min per epoch for the baseline, ≈ 1.7 min per epoch for the final model with 1,082 training patches).

**Model selection (improvement experiments)**
- Starting from the baseline, three cumulative experiments were run: **E1** = bilinear upsampling + class-weight cap 3; **E2** = E1 + NDVI/NBR channels; **E3** = E2 + overlapping training patches. Each uses an overlay config file and is stored as a separate versioned run, so the baseline was never overwritten.
- **Selection rule, fixed before training:** highest validation land-cover mIoU + validation forest-loss PR-AUC. The test region was **not** used to choose between models; only the winner was evaluated on test, once.

| Model | Val mIoU | Val PR-AUC | Score |
|---|---|---|---|
| Baseline | 0.581 | 0.462 | 1.044 |
| E1 | 0.573 | 0.485 | 1.057 |
| E2 | 0.591 | 0.478 | 1.069 |
| **E3 (final)** | **0.628** | **0.502** | **1.130** |

**Evaluation method**
- Only the held-out test blocks are used for final scores; validation blocks are used for early stopping and threshold selection.
- Forest-loss threshold: swept 0.05–0.95 (step 0.01) on **validation**; F1-maximising value **0.70** fixed (baseline model: 0.43) and applied unchanged to test and to the full-scene map.
- Land-cover metrics: overall accuracy (OA), per-class IoU = TP/(TP+FP+FN), mean IoU, precision, recall, F1 (= Dice), confusion matrix.
- Forest-loss metrics: precision, recall, F1/Dice, IoU, **PR-AUC** (average precision; a random model scores the positive prevalence), confusion matrix, threshold analysis. Accuracy is not meaningful at ~1% prevalence.
- Reference baselines: majority class ("always tree cover") for land cover; prevalence for PR-AUC.

**Inference:** full scene processed in 128-px tiles with 32-px overlap (reflect padding at borders, overlapping probabilities averaged to avoid seams); outputs keep the input CRS and geotransform: land-cover GeoTIFF (classes 0–4 with colour table), forest-loss probability GeoTIFF (float), binary forest-loss mask GeoTIFF, preview PNG, area summary (km², ha; one 30 m pixel = 0.09 ha).

**Demo:** optional Streamlit web app with three tabs: full-scene maps with GeoTIFF downloads; test metrics and figures; live prediction on any 512 x 512 px window compared side by side with reference labels.

### C10. Results (all measured on the held-out test region; final model = E3)
- Training (E3): 1,082 training patches, best epoch 14 of 22 (early stopping), **37.6 min** on CPU. Best epoch: train loss 1.133 / validation loss 1.321; land-cover mIoU train 0.638 / validation 0.628; forest-loss F1@0.5 train 0.511 / validation 0.492 (small gap → no strong overfitting).

**Land cover (test, 108 patches, 1,742,097 labelled pixels):**
| Metric | Final model (E3) | Baseline model | Majority-class baseline |
|---|---|---|---|
| Overall accuracy | **0.889** | 0.877 | 0.644 |
| Mean IoU | **0.600** | 0.561 | 0.129 |
| Mean Dice | **0.708** | 0.669 | — |

| Class | IoU | Precision | Recall | F1 | Baseline-model IoU | Reference px |
|---|---|---|---|---|---|---|
| Tree cover | 0.900 | 0.962 | 0.934 | 0.948 | 0.890 | 1,121,364 |
| Cropland | 0.744 | 0.866 | 0.841 | 0.853 | 0.720 | 511,235 |
| Water | 0.789 | 0.810 | 0.968 | 0.882 | 0.771 | 44,609 |
| Built-up | 0.398 | 0.506 | 0.650 | 0.569 | 0.266 | 12,307 |
| Grassland | 0.169 | 0.231 | 0.387 | 0.290 | 0.159 | 52,582 |
Validation (E3): OA 0.876, mIoU 0.628, mean Dice 0.738.

**Forest-cover loss 2021–2023 (threshold 0.70 from validation):**
| Metric | Test (E3) | Validation (E3) | Test (baseline model) |
|---|---|---|---|
| Precision | **0.449** | 0.503 | 0.391 |
| Recall | **0.694** | 0.499 | 0.702 |
| F1 = Dice | **0.545** | 0.501 | 0.502 |
| IoU | **0.375** | 0.334 | 0.335 |
| PR-AUC | **0.584** | 0.502 | 0.558 |
| Random PR-AUC (prevalence) | 0.0105 | 0.0166 | 0.0105 |
| TP / FP / FN | 12,140 / 14,911 / 5,364 | 8,435 / 8,328 / 8,463 | 12,284 / 19,128 / 5,220 |

- PR-AUC is about **56 times** the random level.
- Confusion: 99.1% of no-loss pixels correctly rejected; 69% of loss pixels detected. Compared with the baseline model, 22% fewer false alarms at almost the same recall.
- Every land-cover class improved; the largest gain is built-up (IoU 0.266 → 0.398, precision 0.29 → 0.51) from capping its class weight at 3.
- Grassland is still weak: of reference grassland, 38.7% correct, 35.0% predicted as cropland, 16.5% as tree cover (dry-season spectral similarity; WorldCover grassland is uncertain).
- Qualitatively, predicted loss areas match clearings visible in the post image and in Hansen, but are smoother and more blob-like than the speckled Hansen reference. Some of the "false positives" are blob edges and pixels the reference ignores.
- At threshold 0.5 the test F1 would have been 0.517 (lower than the 0.545 at the validation-tuned 0.70). Thresholds are chosen on validation only.
- **Reproducibility:** two independent trainings of the baseline configuration differed by about ±0.01 on all test metrics (TensorFlow CPU kernels are not fully deterministic). The data and patches were byte-identical between runs. The E3 gain over the baseline (+0.039 test mIoU, +0.043 test F1, +0.086 validation selection score) is well outside this noise. E3 itself was trained once.

**Full-scene maps (whole 10,031 km² scene, includes the training region, so not an evaluation):**
| Class | Predicted, E3 (km²) | WorldCover 2021, mapped classes (km²) |
|---|---|---|
| Tree cover | 6,790.2 | 6,779.8 |
| Cropland | 2,164.6 | 2,118.2 |
| Water | 550.8 | 487.3 |
| Grassland | 407.2 | 368.8 |
| Built-up | 118.1 | 93.1 |
- Predicted forest loss: 276,100 px = **248.5 km² (24,849 ha)** vs Hansen reference 135.6 km²; 63.3 km² of the prediction falls on pixels Hansen ignores.
- The F1-optimal threshold favours recall, so the maps over-estimate loss area (≈1.8x). Built-up over-prediction fell from ≈2.0x (baseline model) to ≈1.3x. Use the maps to show **where** change occurs, not as calibrated area estimates.
- The maps visibly show:
  - the Brahmaputra and its sandbars (water/grassland) in the north;
  - the forested Karbi Anglong hills;
  - valley cropland;
  - towns as built-up in the south/south-east;
  - scattered forest loss in the hills.

### C11. Limitations (include in Results discussion or a short closing paragraph)
1. Temporal mismatch between labels (WorldCover 2021) and imagery; loss near composite windows can leak.
2. Hansen resampled to 30 m: speckled labels, no finer than 30 m.
3. Gross forest-cover loss is not strictly deforestation (includes harvest, fire, flood/river change, shifting cultivation).
4. Reference labels are model-derived products with their own errors; scores = agreement, not ground truth.
5. Strong class imbalance (loss 1.49%, built-up 0.93%, grassland 3.7%).
6. Single region and one geographic split; transfer to other regions untested.
7. Rare classes (shrubland, bare, wetland) ignored and cannot be predicted.
8. Dry-season imagery only, limiting grassland/cropland separation.

### C12. Future scope
More regions/years, several split seeds and repeated training runs (confidence intervals); time series and Sentinel-1 radar; further indices or pretrained encoders; validation against independently interpreted samples; comparison with Random Forest and single-task baselines; distinguishing loss drivers (jhum, plantation, river erosion); calibrated area estimation; web-map deployment.

## PART D: FIGURES AVAILABLE (real outputs; reference them as placeholders)
| File | Shows | Suggested section |
|---|---|---|
| outputs/plots/split_map.png | train/val/test blocks with buffer + forest-loss pixels per patch cell | Detailed Description (split) or EDA |
| outputs/plots/eval_20261124_203857/training_curves.png | loss, mIoU, F1 per epoch (train vs val) | Detailed Description (training) or Results |
| outputs/plots/eval_20261124_203857/confusion_landcover_test.png | land-cover confusion matrix, test | Results |
| outputs/plots/eval_20261124_203857/confusion_deforestation_test.png | forest-loss confusion matrix, test | Results |
| outputs/plots/eval_20261124_203857/pr_curve.png | precision-recall curves val/test with random baseline | Results |
| outputs/plots/eval_20261124_203857/threshold_val.png | threshold tuning on validation | Detailed Description (evaluation) |
| outputs/plots/eval_20261124_203857/examples_test.png | test patches: pre/post RGB, reference vs predicted land cover and loss | Results |
| outputs/predictions/pred_20261124_204028/preview.png | full-scene post RGB, predicted land cover, loss probability, loss overlay | Results or Proposed Work |

For EDA, also recommend 2 charts the student can create from the tables in C7 (e.g. bar chart of WorldCover class shares; bar chart of forest-loss pixels by year and ignore reasons) — mark them as `[Figure N: ... — create from Table M]`.
Diagrams to provide as Mermaid code: (1) end-to-end workflow flowchart (7 stages with inputs/outputs and the fail-fast checks), (2) system architecture (three layers), (3) multi-task U-Net architecture.

## PART E: REFERENCES (use only these; IEEE style)
1. M. C. Hansen et al., "High-resolution global maps of 21st-century forest cover change," *Science*, vol. 342, no. 6160, pp. 850–853, 2013, doi:10.1126/science.1244693.
2. D. Zanaga et al., "ESA WorldCover 10 m 2021 v200," Zenodo, 2022, doi:10.5281/zenodo.7254221.
3. M. Drusch et al., "Sentinel-2: ESA's optical high-resolution mission for GMES operational services," *Remote Sensing of Environment*, vol. 120, pp. 25–36, 2012, doi:10.1016/j.rse.2011.11.026.
4. N. Gorelick et al., "Google Earth Engine: Planetary-scale geospatial analysis for everyone," *Remote Sensing of Environment*, vol. 202, pp. 18–27, 2017, doi:10.1016/j.rse.2017.06.031.
5. V. J. Pasquarella et al., "Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning," in *Proc. IEEE/CVF CVPR Workshops*, 2023, pp. 2125–2135. (Cloud Score+)
6. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, LNCS vol. 9351, 2015, pp. 234–241, doi:10.1007/978-3-319-24574-4_28.
7. F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in *Proc. 3DV*, 2016, pp. 565–571. (Dice loss)
8. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.
9. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015, arXiv:1412.6980.
10. D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi:10.1111/ecog.02881. (spatial split / leakage)
11. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *Proc. 12th USENIX OSDI*, 2016, pp. 265–283.
12. F. Chollet et al., "Keras," 2015. [Online]. Available: https://keras.io

## PART F: SECTION-BY-SECTION INSTRUCTIONS

1. **Title page (1 page):** title; "A project report submitted in partial fulfilment of the requirements for the degree of <<DEGREE, e.g. Bachelor of Technology in ...>>"; by <<STUDENT NAME(S)>> (<<ROLL NUMBER(S)>>); under the guidance of <<GUIDE NAME, DESIGNATION>>; <<DEPARTMENT>>, <<INSTITUTION>>, <<CITY>>; <<MONTH YEAR>>. Keep the placeholders as written.
2. **Table of contents (1 page):** all numbered sections and subsections with page-number placeholders, plus a list of figures and tables.
3. **Abstract (1 page, 250–300 words):** problem, data (Sentinel-2 bi-temporal composites, WorldCover 2021, Hansen GFC; 10,031 km² of Assam at 30 m), method (validated pipeline, spatial split, multi-task U-Net, masked weighted losses, validation-tuned threshold), key test results of the final model (OA 0.889, mIoU 0.600; loss F1 0.545, PR-AUC 0.584 vs 0.0105 random; improved over the baseline model via validation-only model selection), CPU training in under 40 min, georeferenced maps, and the agreement-not-ground-truth caveat. Add 5–6 keywords.
4. **Problem Statement (1 page):** use C2 and C3; end with the numbered objectives.
5. **Proposed Work (4–5 pages):** overview of the proposed system vs existing approaches (short comparison table); **system architecture diagram (Mermaid)**; **workflow flowchart (Mermaid)** of the 7 stages with their inputs/outputs and fail-fast checks; design decisions with brief justifications (multi-task U-Net, 12-band bi-temporal input plus NDVI/NBR change channels (18 inputs), nobs dropped, spatial block split before patching with buffer, masked weighted losses, validation-only threshold, versioned runs and lineage, CPU-friendly sizing); multi-task U-Net architecture diagram (Mermaid) with the layer/parameter table; tools and technologies table (Python 3.12, TensorFlow 2.21/Keras 3.15, NumPy, rasterio/GDAL, scikit-learn, pandas, matplotlib, pytest, Streamlit, Google Earth Engine).
6. **Detailed Description (7 pages):** one subsection per step of the flowchart, in order, explaining **what is done, why, and how it was implemented** (with the exact parameters from C9): data acquisition (GEE, done beforehand); data validation; preprocessing and normalisation; label remapping and ignore masks (with the temporal strategy C5); spatial split (with split table and split_map figure); patch generation and tf.data loading with augmentation; model architecture; loss functions and imbalance handling (with formulas and weights table); training procedure (with training_curves figure); model selection experiments (E1–E3 table, validation-only rule); evaluation method and threshold tuning (with threshold_val figure); full-scene inference and GeoTIFF outputs; reproducibility, testing and demo.
7. **Datasets and EDA (1–3 pages, at least one full page of EDA):** dataset table (C4); study area description; validation summary (C6); EDA tables from C7 (band statistics, class distribution, forest-loss statistics, split statistics) and their interpretation (imbalance, spatial concentration of loss, water disagreement, spectral similarity); 2 recommended chart placeholders.
8. **Results (1 page):** land-cover metrics table (overall and per class, with baseline), forest-loss metrics table (test and validation, with random baseline), 1–2 figure placeholders (confusion matrices and/or examples), and a short discussion: strengths, weak classes and why, improvement over the baseline model, area-bias caveat, reproducibility, and a 3–4 line limitations paragraph (C11). Keep the agreement-not-ground-truth statement.
9. **References (1 page):** the list in Part E, numbered in order of first citation in the text.

Begin writing the report now, section by section, following Parts B–F exactly.
