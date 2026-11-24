"""Raster preprocessing shared by training-data preparation and inference.

- builds the model input stack (pre + post, nobs dropped or kept, reflectance
  clipped), always as raw uint16 DN; normalisation happens at load time
- computes the validity mask (no-data in either date -> invalid)
- remaps WorldCover IDs to model classes with an explicit lookup table
- builds both label maps with the ignore rules from the config
- computes / applies robust per-channel normalisation statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def channel_layout(cfg: dict[str, Any]) -> tuple[list[int], list[str]]:
    """Band indices kept from each date and the resulting channel names (pre first)."""
    bands = cfg["s2"]["bands"]
    if cfg["s2"]["nobs_mode"] == "drop":
        keep = [bands.index(b) for b in cfg["s2"]["spectral_bands"]]
    else:
        keep = list(range(len(bands)))
    names = [f"pre_{bands[i]}" for i in keep] + [f"post_{bands[i]}" for i in keep]
    return keep, names


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read all bands as (bands, H, W) plus the rasterio profile."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(), dict(src.profile)


def compute_valid_mask(pre: np.ndarray, post: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """True where both dates have data: not all reflectance bands 0 and nobs > 0."""
    bands = cfg["s2"]["bands"]
    spec = [bands.index(b) for b in cfg["s2"]["spectral_bands"]]
    nobs = bands.index(cfg["s2"]["nobs_band"])
    valid = np.ones(pre.shape[1:], dtype=bool)
    for arr in (pre, post):
        valid &= ~(arr[spec] == 0).all(axis=0)
        valid &= arr[nobs] > 0
    return valid


def build_input_stack(pre: np.ndarray, post: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Return (H, W, C) uint16: [pre channels, post channels], clipped. C = 12 (drop) or 14."""
    if pre.shape != post.shape:
        raise ValueError(f"pre {pre.shape} and post {post.shape} composites differ in shape")
    keep, names = channel_layout(cfg)
    bands = cfg["s2"]["bands"]
    nobs = bands.index(cfg["s2"]["nobs_band"])
    rmax = np.uint16(cfg["s2"]["reflectance_clip_max"])
    nmax = np.uint16(cfg["s2"]["nobs_clip_max"])
    h, w = pre.shape[1:]
    out = np.empty((h, w, 2 * len(keep)), dtype=np.uint16)
    for d, arr in enumerate((pre, post)):
        for j, b in enumerate(keep):
            out[..., d * len(keep) + j] = np.minimum(arr[b], nmax if b == nobs else rmax)
    return out, names


def worldcover_lut(cfg: dict[str, Any]) -> np.ndarray:
    """256-entry lookup table raw ID -> model class (unmapped IDs -> ignore_index)."""
    ignore = cfg["worldcover"]["ignore_index"]
    lut = np.full(256, ignore, dtype=np.uint8)
    for raw_id, cls in cfg["worldcover"]["raw_to_model"].items():
        lut[int(raw_id)] = int(cls)
    return lut


def remap_worldcover(wc: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Map raw WorldCover IDs to model classes 0..4 / 255 with the explicit config mapping."""
    if wc.dtype != np.uint8:
        raise ValueError(f"WorldCover must be uint8, got {wc.dtype}")
    mapped = {int(k) for k in cfg["worldcover"]["raw_to_model"]}
    present = set(np.unique(wc).tolist())
    unknown = sorted(present - mapped)
    if unknown and cfg["worldcover"]["unknown_id_policy"] == "error":
        raise ValueError(f"WorldCover IDs {unknown} are not in worldcover.raw_to_model")
    return worldcover_lut(cfg)[wc]


def build_landcover_target(lc_model: np.ndarray, loss_target: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Land-cover labels with 255 where the input is invalid or (optionally) loss_target == 1."""
    ignore = cfg["worldcover"]["ignore_index"]
    out = lc_model.copy()
    if cfg["worldcover"]["ignore_where_loss_target_is_loss"]:
        out[loss_target == cfg["hansen"]["loss_target_values"]["loss"]] = ignore
    out[~valid] = ignore
    return out


def build_deforestation_target(loss_target: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Deforestation labels {0, 1, 255}; 255 also where the input is invalid."""
    out = loss_target.copy()
    out[~valid] = cfg["hansen"]["loss_target_values"]["ignore"]
    return out


def compute_band_stats(pixels: np.ndarray, names: list[str], percentiles: tuple[float, float]) -> dict[str, Any]:
    """Robust stats from (N, C) TRAIN pixels: percentile clip bounds, then mean/std of clipped values."""
    if pixels.ndim != 2 or pixels.shape[1] != len(names) or pixels.shape[0] == 0:
        raise ValueError(f"expected (N>0, {len(names)}) pixels, got {pixels.shape}")
    lo, hi = np.percentile(pixels, percentiles, axis=0)
    clipped = np.clip(pixels.astype(np.float64), lo, hi)
    mean = clipped.mean(axis=0)
    std = np.maximum(clipped.std(axis=0), 1e-6)
    return {"channels": list(names), "p_low": lo.tolist(), "p_high": hi.tolist(),
            "mean": mean.tolist(), "std": std.tolist(), "percentiles": list(percentiles),
            "n_pixels": int(pixels.shape[0])}


def normalize(image: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """(..., C) uint16 DN -> float32 robust-standardised values."""
    lo = np.asarray(stats["p_low"], np.float32)
    hi = np.asarray(stats["p_high"], np.float32)
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    if image.shape[-1] != lo.size:
        raise ValueError(f"image has {image.shape[-1]} channels, stats have {lo.size}")
    x = np.clip(image.astype(np.float32), lo, hi)
    return (x - mean) / std
