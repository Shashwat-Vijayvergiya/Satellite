"""Patch manifests and memory-aware tf.data pipelines.

Each patch is an uncompressed .npz under the patch run directory:
    image      uint16 (128, 128, C)  raw DN, clipped (normalised at load time)
    landcover  uint8  (128, 128)     0..4, 255 = ignore
    deforest   uint8  (128, 128)     0/1,  255 = ignore
Patches are read lazily by tf.data, so only a few batches are in RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.preprocessing import model_input

if TYPE_CHECKING:
    import pandas as pd
    import tensorflow as tf

IGNORE = 255


def load_manifest(patch_run: Path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(patch_run / "manifest.csv")
    if df.empty:
        raise RuntimeError(f"manifest in {patch_run.name} is empty")
    if "overlap_extra" not in df.columns:  # manifests written before overlapping train patches existed
        df["overlap_extra"] = False
    return df


def split_paths(manifest: "pd.DataFrame", split: str, patch_run: Path) -> list[str]:
    rows = manifest[manifest["split"] == split]
    paths = [str(patch_run / p) for p in rows["path"]]
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} patch files missing for split {split}, e.g. {missing[0]}")
    return paths


def limit_for_smoke_test(paths: list[str], n: int, seed: int) -> list[str]:
    if len(paths) <= n:
        return list(paths)
    idx = np.random.default_rng(seed).choice(len(paths), size=n, replace=False)
    return [paths[i] for i in sorted(idx)]


def load_patch(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["image"], z["landcover"], z["deforest"]


def compute_class_weights(lc_counts: np.ndarray, power: float, clip: tuple[float, float]) -> np.ndarray:
    """Smoothed inverse-frequency weights, rescaled so the pixel-weighted mean is 1, then clipped."""
    counts = np.asarray(lc_counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"every class needs > 0 training pixels, got {counts.tolist()}")
    freq = counts / counts.sum()
    w = (1.0 / freq) ** power
    w = w / (w * freq).sum()
    return np.clip(w, clip[0], clip[1]).astype(np.float32)


def compute_pos_weight(pos: int, neg: int, power: float, max_weight: float) -> float:
    if pos <= 0 or neg <= 0:
        raise ValueError(f"need positive and negative training pixels, got pos={pos} neg={neg}")
    return float(min((neg / pos) ** power, max_weight))


def augment(x: "tf.Tensor", lc: "tf.Tensor", df: "tf.Tensor") -> tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
    """Identical random horizontal/vertical flip and 90-degree rotation for image and labels."""
    import tensorflow as tf

    c = x.shape[-1]
    stack = tf.concat([x, tf.cast(lc, tf.float32), df], axis=-1)
    stack = tf.cond(tf.random.uniform(()) < 0.5, lambda: tf.reverse(stack, [1]), lambda: stack)
    stack = tf.cond(tf.random.uniform(()) < 0.5, lambda: tf.reverse(stack, [0]), lambda: stack)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    stack = tf.image.rot90(stack, k)
    return stack[..., :c], tf.cast(stack[..., c: c + 1], tf.int32), stack[..., c + 1: c + 2]


def make_dataset(paths: list[str], stats: dict[str, Any], batch_size: int, shuffle: bool, augment_data: bool,
                 seed: int, patch_size: int, indices: list[str] | None = None) -> "tf.data.Dataset":
    """Yield (image float32 (B,P,P,C), {"landcover": int32 (B,P,P,1), "deforestation": float32 (B,P,P,1)})."""
    import tensorflow as tf

    if not paths:
        raise ValueError("no patch paths given")
    n_raw = len(stats["channels"])
    n_ch = n_raw + 3 * len(indices or [])

    def _load(p: bytes):
        img, lc, df = load_patch(p.decode())
        if img.shape != (patch_size, patch_size, n_raw):
            raise ValueError(f"{p.decode()}: image shape {img.shape} != {(patch_size, patch_size, n_raw)}")
        return model_input(img, stats, indices), lc[..., None].astype(np.int32), df[..., None].astype(np.float32)

    def _tf_load(p):
        x, lc, df = tf.numpy_function(_load, [p], [tf.float32, tf.int32, tf.float32])
        x.set_shape((patch_size, patch_size, n_ch))
        lc.set_shape((patch_size, patch_size, 1))
        df.set_shape((patch_size, patch_size, 1))
        return x, lc, df

    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(lambda x, lc, df: (x, {"landcover": lc, "deforestation": df}))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
