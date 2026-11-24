"""Core library for satellite land-cover and deforestation segmentation (Assam).

Submodules:
    preprocessing  -- raster I/O, grid checks, label remapping, normalisation
    dataset        -- patch index and tf.data pipelines
    losses         -- masked multi-task losses
    metrics        -- masked segmentation metrics and threshold tuning
    visualization  -- plots of inputs, labels, predictions and training curves
    models         -- U-Net and multi-task U-Net builders

Heavy dependencies (tensorflow, rasterio) are imported lazily inside the
functions that need them, so `import src` stays cheap.

Other modules: validation (raw-data checks), runs (versioned run dirs and
lineage), spatial_split (geographic block split), inference (tiled
prediction and GeoTIFF writing), cli (shared stage runner).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_config", "set_global_seed", "PROJECT_ROOT"]

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configs/config.yaml and validate cross-field consistency (fail fast).

    Raises FileNotFoundError / ValueError with a clear message on any problem.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} is empty or not a mapping")

    required = ["paths", "grid", "s2", "hansen", "worldcover", "split", "patching",
                "model", "loss", "training", "evaluation", "validation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")

    n_bands = len(cfg["s2"]["bands"])
    mode = cfg["s2"]["nobs_mode"]
    if mode not in ("as_channel", "drop"):
        raise ValueError(f"s2.nobs_mode must be 'as_channel' or 'drop', got {mode!r}")
    channels = 2 * (n_bands if mode == "as_channel" else n_bands - 1)
    if cfg["model"]["input_channels"] != channels:
        raise ValueError(f"model.input_channels={cfg['model']['input_channels']} but s2.nobs_mode={mode} gives {channels}")
    if cfg["split"]["block_size_px"] % cfg["patching"]["patch_size"]:
        raise ValueError("split.block_size_px must be a multiple of patching.patch_size")
    if cfg["evaluation"]["deforestation_threshold"]["tune_on"] != "val":
        raise ValueError("evaluation.deforestation_threshold.tune_on must be 'val' (never tune on test)")
    mapping = {int(k): int(v) for k, v in cfg["worldcover"]["raw_to_model"].items()}
    n_cls = cfg["model"]["heads"]["landcover"]["num_classes"]
    kept = sorted(v for v in mapping.values() if v != cfg["worldcover"]["ignore_index"])
    if kept != list(range(n_cls)):
        raise ValueError(f"worldcover.raw_to_model must map to classes 0..{n_cls - 1} exactly once, got {kept}")
    cfg["worldcover"]["raw_to_model"] = mapping
    return cfg


def set_global_seed(seed: int, tensorflow: bool = True) -> None:
    """Seed Python, NumPy and (if tensorflow=True) TensorFlow via keras.utils.set_random_seed."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not tensorflow:
        return
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:  # data stages do not need TensorFlow
        pass
