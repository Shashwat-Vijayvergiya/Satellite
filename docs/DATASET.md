# Dataset documentation and data dictionary

All numbers in this document were **measured on the actual files** by `scripts/inspect_rasters.py` (Phase A) and `scripts/prepare_dataset.py`. The full machine-readable report is `outputs/reports/validation/run_*/validation_report.json`.

## 1. Study area

| Property | Value |
|---|---|
| Region | Kaziranga / Karbi Anglong, Assam, India |
| Extent (lon/lat) | 92.9998 E - 93.9080 E, 25.7970 N - 26.8031 N |
| CRS | EPSG:32646 (WGS 84 / UTM zone 46N) |
| Pixel size | 30 m |
| Grid | 3009 columns x 3704 rows = 11,145,336 pixels |
| Origin (x0, ytop) | (499980, 2964630) m; transform (30, 0, 499980, 0, -30, 2964630) |
| Size | about 90.3 km x 111.1 km (about 10,031 km²) |

The north of the scene is the Brahmaputra floodplain (river, Kaziranga grasslands, tea and rice); the south is the forested Karbi Anglong hills, where most forest loss occurs.

## 2. Source files (`data/raw/`)

All five files were exported from Google Earth Engine as single tiles (no split exports). All four rasters share **exactly** the same CRS, transform and size, checked to 1e-6 m.

| File | Content | Dtype | Bands | Compression |
|---|---|---|---|---|
| `s2_pre_2020-21.tif` | Sentinel-2 L2A Harmonized median composite, 2020-11-01 to 2021-02-28 | UInt16 | 7: B2, B3, B4, B8, B11, B12, nobs (names embedded) | LZW |
| `s2_post_2023-24.tif` | Same, 2023-11-01 to 2024-02-29 | UInt16 | 7: as above (names embedded) | LZW |
| `worldcover_2021_30m.tif` | ESA WorldCover 2021 v200, 10 m aggregated to 30 m by **mode** | UInt8 | 1: Map (no embedded name) | LZW |
| `hansen_labels_30m.tif` | Hansen GFC 2024 v1.12 bands + derived target, nearest-neighbour to 30 m | UInt8 | 4: treecover2000, lossyear, datamask, loss_target (names embedded) | LZW |
| `aoi.geojson` | Study-area polygon + export metadata | GeoJSON | n/a | n/a |

`aoi.geojson` properties (crs, width/height, x0/ytop, scale, band orders, tree threshold 30, loss years 21-23, 1 x 1 tiles) all match the rasters. Its bounding box matches the raster footprint to within 1e-13°. The AOI lists the windows as end-exclusive (`2020-11-01..2021-03-01`, `2023-11-01..2024-03-01`), which covers the same dates as above.

The SHA-256 hash of every raw file is recorded at validation. Every later stage refuses to run if a file changed after validation.

## 3. Sentinel-2 composites

Values are surface reflectance x 10000 (raw DN of the harmonized collection). B2, B3, B4 and B8 are native 10 m; B11 and B12 are native 20 m; all were averaged to 30 m. Clouds were masked with Cloud Score+ (`cs_cdf >= 0.60`), then a per-pixel median was taken over each window.

**Validity:** 100% of pixels are valid in both dates. No pixel has all reflectance = 0, `nobs` is never 0, and the two no-data conventions never disagree.

Statistics over all pixels:

| Band | Pre mean | Pre std | Pre p1-p99 | Post mean | Post std | Post p1-p99 |
|---|---|---|---|---|---|---|
| B2 (blue) | 539.7 | 306.9 | 180-1970 | 496.8 | 310.2 | 162-2123 |
| B3 (green) | 732.0 | 335.1 | 303-2264 | 689.5 | 337.1 | 275-2398 |
| B4 (red) | 663.5 | 442.8 | 186-2419 | 590.3 | 443.5 | 155-2565 |
| B8 (NIR) | 2606.6 | 650.4 | 441-4601 | 2753.9 | 702.2 | 347-4818 |
| B11 (SWIR1) | 2003.2 | 621.7 | 181-3869 | 1978.1 | 610.9 | 159-3926 |
| B12 (SWIR2) | 1191.6 | 681.5 | 148-4140 | 1148.2 | 662.8 | 140-4187 |
| nobs | 22.6 | 10.3 | 15-68 (min 9, max 82) | 20.7 | 10.0 | 13-65 (min 4, max 84) |

Only **14 pixels** in the whole scene exceed 10000 DN (B8, B11 or B12; bright targets or residual cloud). They are clipped to 10000. No band is constant.

## 4. ESA WorldCover 2021 (land-cover reference)

The raster contains only IDs 10, 20, 30, 40, 50, 60, 80 and 90. There is no ID 0 (no data), and none of 70, 95 or 100. Every share is within 0.05 percentage points of the Earth Engine diagnostics.

| Raw ID | Class | Pixels | Share | Model class |
|---|---|---|---|---|
| 10 | Tree cover | 7,671,037 | 68.83% | **0 tree_cover** |
| 40 | Cropland | 2,359,033 | 21.17% | **2 cropland** |
| 80 | Permanent water | 541,483 | 4.86% | **4 water** |
| 30 | Grassland | 417,084 | 3.74% | **1 grassland** |
| 50 | Built-up | 103,453 | 0.93% | **3 built_up** |
| 60 | Bare / sparse vegetation | 44,606 | 0.40% | 255 ignore |
| 20 | Shrubland | 5,060 | 0.05% | 255 ignore |
| 90 | Herbaceous wetland | 3,580 | 0.03% | 255 ignore |
| 0 | No data | 0 | 0% | 255 ignore |

The explicit raw-to-model mapping is in `configs/config.yaml` (`worldcover.raw_to_model`), and every prepared dataset saves it as `class_mapping.json`. Classes 20, 60 and 90 are ignored because they are too rare (0.48% together) to learn reliably from one scene.

## 5. Hansen Global Forest Change (forest-loss reference)

The native Hansen grid is geographic (EPSG:4326, 0.00025°, about 25 x 28 m here). It was resampled **nearest-neighbour** to the 30 m UTM grid. Its accuracy is therefore limited to 30 m, and no 10 m claim is made.

| Band | Values found | Meaning |
|---|---|---|
| `treecover2000` | 0-100 (mean 43.5) | % canopy closure in 2000 |
| `lossyear` | 0-24 | 0 = no loss; n = loss in 2000 + n |
| `datamask` | 1: 11,004,848 px; 2: 140,488 px | 1 = land, 2 = permanent water (no code 0 present) |
| `loss_target` | 0, 1, 255 | the training target (below) |

`loss_target` definition: **1** = land, `lossyear` in 21-23, and `treecover2000` >= 30%. **0** = land with `lossyear` = 0. **255** = everything else. The validator recomputes it pixel by pixel from the other three bands: **0 pixels disagree**.

| loss_target | Pixels | Note |
|---|---|---|
| 1 (loss 2021-2023) | **150,715** | 1.49% of labelled pixels; 2021: 54,386, 2022: 48,965, 2023: 47,364 |
| 0 (no loss) | **9,953,974** | |
| 255 (ignore) | **1,040,647** | water/no data 140,488; loss 2001-2020: 819,303; loss 2024: 46,256; 2021-23 loss below 30% canopy: 34,600 |

These counts exactly match the Earth Engine diagnostics.

Of the loss pixels, 137,871 (91.5%) are WorldCover tree cover, 7,271 grassland and 5,514 cropland.

**What the target is:** gross stand-replacement forest-cover loss. It is **not** strictly deforestation: it can include plantation harvest, fire, flood and river-bank change, and shifting cultivation (jhum), which is common in Karbi Anglong.

## 6. Temporal strategy

```
2020-11 ---- 2021-02         2021 ............ 2023          2023-11 ---- 2024-02
[ pre composite ]       WorldCover 2021   Hansen loss 2021-23   [ post composite ]
```

- **WorldCover 2021** is not exactly time-matched to either composite. The pre composite (dry season 2020/21) is the closest match.
- **Forest loss 2021-2023** sits between the two composites. Loss events within about 2 months of the window edges leak: Jan-Feb 2021 loss may already show in the pre image, and Jan-Feb 2024 changes (labelled lossyear 24, ignored) can show in the post image.
- **Stale labels:** land-cover labels on pixels that lost forest after 2021 are stale for the post image. Those pixels (`loss_target == 1`) are therefore ignored in the land-cover loss and metrics.
- **Water disagreement:** WorldCover marks 541,483 px as water but Hansen marks only 140,488 px. 435,337 WorldCover-water pixels are Hansen land, mostly the seasonally shifting Brahmaputra channels and sandbars.

## 7. Derived datasets (data dictionary)

Every stage writes a new timestamped run folder with a `metadata.json` (status, creation time, sources and SHA-256, config, parent run IDs) and updates `latest.json`.

### `data/processed/prep_<ts>/`

| File | Type / shape | Content |
|---|---|---|
| `image.npy` | uint16 (3704, 3009, 12) | channels `pre_B2, pre_B3, pre_B4, pre_B8, pre_B11, pre_B12, post_B2 ... post_B12`, DN clipped to 10000 |
| `landcover.npy` | uint8 (3704, 3009) | 0 tree_cover, 1 grassland, 2 cropland, 3 built_up, 4 water, 255 ignore |
| `deforest.npy` | uint8 (3704, 3009) | 1 loss 2021-23, 0 no loss, 255 ignore |
| `valid.npy` | bool (3704, 3009) | input valid in both dates |
| `class_mapping.json` | JSON | raw-to-model mapping and ignore rules |

Prepared label counts: land cover tree 7,533,166; grass 409,813; crop 2,353,519; built-up 103,450; water 541,474; ignore 203,914 (IDs 20/60/90 plus loss pixels). Forest loss: 150,715 / 9,953,974 / 1,040,647.

`nobs` is not an input channel (`s2.nobs_mode: drop`). It carries no validity information here (no missing pixels), and its range (4-84) mostly reflects Sentinel-2 orbit overlap, a location artefact.

### `data/splits/split_<ts>/`

| File | Content |
|---|---|
| `block_assignment.csv` | block_id, block_row/col, split, pixel and map bounds, kept patches |
| `windows.csv` | patch_id (`rRR_cCC`), split, cell row/col, row_off/col_off (px), block_id, x_min/y_max (m) |
| `cell_split.npy` | int8 (28, 23): 0 train, 1 val, 2 test, -1 buffer |
| `summary.json`, `split_map.png` | per-split counts; map |

### `data/patches/patches_<ts>/`

| File | Content |
|---|---|
| `{train,val,test}/<patch_id>.npz` | `image` uint16 (128,128,12), `landcover` uint8 (128,128), `deforest` uint8 (128,128) |
| `manifest.csv` | patch_id, split, path, source_prep_run, source_files, row_off, col_off, block_id, x_min, y_max, valid_pixels, lc_tree_cover ... lc_water, lc_ignore, df_positive, df_negative, df_ignore |
| `band_stats.json` | per-channel p1, p99, mean and std of clipped values, from **train patches only** (5,193,728 pixels) |

## 8. Spatial split

- The scene is cut into a 28 x 23 grid of 128 px patch cells. Partial cells at the right and bottom edges are dropped, so 644 cells remain.
- Cells are grouped into 42 blocks of 512 px (4 x 4 cells), and whole blocks are assigned to train, val or test **before** any patch is extracted.
- A 1-cell buffer drops train cells next to val/test blocks, and val cells next to test blocks; test cells are never dropped.
- A seeded search (seed 42, 5000 candidates, 386 feasible) picks the assignment closest to 70/15/15 for both patch counts and loss pixels. It must give >= 10,000 loss pixels to val and test, and >= 10,000 pixels of every class to every split.

| Split | Patches | Loss px | Tree | Grass | Crop | Built-up | Water |
|---|---|---|---|---|---|---|---|
| train | 317 | 79,703 | 3,645,486 | 194,775 | 902,276 | 33,276 | 314,369 |
| val | 71 | 16,898 | 762,491 | 51,619 | 282,192 | 15,074 | 28,189 |
| test | 108 | 17,504 | 1,121,364 | 52,582 | 511,235 | 12,307 | 44,609 |
| buffer (dropped) | 148 | | | | | | |

The split map is in `outputs/plots/split_map.png`.

## 9. Citations and attribution

- **Sentinel-2:** contains modified Copernicus Sentinel data (2020-2024), processed in Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED). Cloud masking used Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED); Pasquarella et al. 2023, CVPR Workshops.
- **ESA WorldCover 2021 v200:** © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by the ESA WorldCover consortium. Licence CC-BY-4.0. Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200.* doi:10.5281/zenodo.7254221.
- **Hansen Global Forest Change v1.12 (2000-2024):** Hansen, M. C. et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science* 342: 850-853. doi:10.1126/science.1244693.
