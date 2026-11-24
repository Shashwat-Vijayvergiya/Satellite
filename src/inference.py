"""Full-scene tiled inference and georeferenced GeoTIFF output.

The scene is reflect-padded, cut into overlapping tiles (tile_size, stride =
tile_size - overlap), predicted in batches, and the probabilities of
overlapping tiles are averaged. This removes the seams a non-overlapping
U-Net leaves at tile edges. Outputs keep the CRS and transform of the input
rasters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

# ESA WorldCover legend colours for model classes 0..4
LANDCOVER_COLORS = {0: (0, 100, 0), 1: (255, 255, 76), 2: (240, 150, 255), 3: (250, 0, 0), 4: (0, 100, 200)}


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Window starts that cover [0, length) with the last window flush to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def predict_scene(predict_fn: Callable[[np.ndarray], dict[str, np.ndarray]], image: np.ndarray, tile: int, overlap: int,
                  batch_size: int, n_classes: int, progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """image: (H, W, C) float32 normalised. Returns (landcover probs (H, W, K), deforestation prob (H, W))."""
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile_size)")
    h, w, _ = image.shape
    ph, pw = max(h, tile), max(w, tile)
    padded = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect") if (ph, pw) != (h, w) else image
    stride = tile - overlap
    windows = [(r, c) for r in tile_origins(ph, tile, stride) for c in tile_origins(pw, tile, stride)]
    lc_sum = np.zeros((ph, pw, n_classes), np.float32)
    df_sum = np.zeros((ph, pw), np.float32)
    count = np.zeros((ph, pw), np.float32)
    for i in range(0, len(windows), batch_size):
        batch_win = windows[i: i + batch_size]
        batch = np.stack([padded[r: r + tile, c: c + tile] for r, c in batch_win])
        out = predict_fn(batch)
        for j, (r, c) in enumerate(batch_win):
            lc_sum[r: r + tile, c: c + tile] += out["landcover"][j]
            df_sum[r: r + tile, c: c + tile] += out["deforestation"][j, ..., 0]
            count[r: r + tile, c: c + tile] += 1
        if progress:
            progress(min(i + batch_size, len(windows)), len(windows))
    lc = lc_sum[:h, :w] / count[:h, :w, None]
    df = df_sum[:h, :w] / count[:h, :w]
    return lc, df


def write_geotiff(path: Path, array: np.ndarray, crs: Any, transform: Any, nodata: float | int | None = None,
                  colormap: dict[int, tuple[int, int, int]] | None = None, tags: dict[str, str] | None = None,
                  band_description: str | None = None) -> None:
    """Write a single-band GeoTIFF on the given grid (LZW compressed, tiled)."""
    import rasterio

    if array.ndim != 2:
        raise ValueError("write_geotiff expects a 2-D array")
    profile = {"driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
               "dtype": array.dtype.name, "crs": crs, "transform": transform, "nodata": nodata,
               "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256}
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        if colormap:
            dst.write_colormap(1, {k: (*v, 255) for k, v in colormap.items()})
        if tags:
            dst.update_tags(**tags)
        if band_description:
            dst.set_band_description(1, band_description)


def area_summary(landcover: np.ndarray, deforest_mask: np.ndarray, valid: np.ndarray, class_names: list[str],
                 pixel_area_m2: float) -> dict[str, Any]:
    """Predicted class areas and loss pixels over valid pixels (km2 and ha)."""
    km2 = pixel_area_m2 / 1e6
    out: dict[str, Any] = {"pixel_area_m2": pixel_area_m2, "valid_pixels": int(valid.sum()),
                           "valid_area_km2": float(valid.sum() * km2), "landcover": {}}
    n_valid = max(int(valid.sum()), 1)
    for k, name in enumerate(class_names):
        n = int(((landcover == k) & valid).sum())
        out["landcover"][name] = {"pixels": n, "area_km2": n * km2, "area_ha": n * km2 * 100, "share_pct": 100 * n / n_valid}
    n_loss = int(((deforest_mask == 1) & valid).sum())
    out["deforestation"] = {"loss_pixels": n_loss, "area_km2": n_loss * km2, "area_ha": n_loss * km2 * 100,
                            "share_pct": 100 * n_loss / n_valid}
    return out
