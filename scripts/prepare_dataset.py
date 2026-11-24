r"""Build aligned, model-ready full-scene arrays from the validated raw rasters.

Outputs (data/processed/prep_<timestamp>/):
    image.npy       uint16 (H, W, C)  pre + post channels (C = 12 with nobs dropped), reflectance clipped
    landcover.npy   uint8  (H, W)     model classes 0..4, 255 = ignore
    deforest.npy    uint8  (H, W)     1 = loss 2021-23, 0 = no loss, 255 = ignore
    valid.npy       bool   (H, W)     input valid in both dates
    class_mapping.json, metadata.json (sources, hashes, config, counts)

Refuses to run unless the latest raw-data validation passed for the current files.

Usage:  python scripts/prepare_dataset.py --config configs/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "prepare_dataset"


def add_args(p) -> None:
    p.add_argument("--force", action="store_true", help="Rebuild even if inputs and config are unchanged")


def run(cfg, args) -> None:
    from src import preprocessing as P
    from src.runs import finish_run, fingerprint, new_run_dir, reusable_run, write_json
    from src.validation import require_validated_raw_data

    latest = require_validated_raw_data(cfg, ROOT)
    fp = fingerprint(latest["sha256"], cfg["grid"], cfg["s2"], cfg["hansen"], cfg["worldcover"])
    base = ROOT / cfg["paths"]["processed_dir"]
    reuse = None if args.force else reusable_run(base, ROOT, fp)
    if reuse:
        log(STAGE, f"inputs and config unchanged: reusing {reuse.name} (use --force to rebuild)")
        return
    raw = cfg["paths"]["raw"]
    log(STAGE, f"validation {latest['run_dir']} passed; reading rasters")
    pre, prof = P.read_raster(ROOT / raw["s2_pre"])
    post, _ = P.read_raster(ROOT / raw["s2_post"])
    wc, _ = P.read_raster(ROOT / raw["worldcover"])
    hz, _ = P.read_raster(ROOT / raw["hansen"])

    valid = P.compute_valid_mask(pre, post, cfg)
    image, names = P.build_input_stack(pre, post, cfg)
    del pre, post
    loss_target = hz[cfg["hansen"]["bands"].index(cfg["hansen"]["loss_target_band"])]
    lc = P.build_landcover_target(P.remap_worldcover(wc[0], cfg), loss_target, valid, cfg)
    df = P.build_deforestation_target(loss_target, valid, cfg)

    run_dir = new_run_dir(ROOT / cfg["paths"]["processed_dir"], "prep")
    for name, arr in (("image", image), ("landcover", lc), ("deforest", df), ("valid", valid)):
        np.save(run_dir / f"{name}.npy", arr)
    names_cls = cfg["worldcover"]["class_names"]
    mapping = {
        "raw_to_model": {str(k): v for k, v in cfg["worldcover"]["raw_to_model"].items()},
        "model_classes": {i: n for i, n in enumerate(names_cls)},
        "ignore_index": cfg["worldcover"]["ignore_index"],
        "landcover_ignored_where_loss_target_is_1": cfg["worldcover"]["ignore_where_loss_target_is_loss"],
        "deforestation": {"1": "forest loss 2021-2023", "0": "no loss", "255": "ignore"},
    }
    write_json(run_dir / "class_mapping.json", mapping)

    lc_counts = {names_cls[k] if k < len(names_cls) else "ignore": int(c) for k, c in enumerate(np.bincount(lc.ravel(), minlength=256)) if c}
    df_counts = {int(k): int(c) for k, c in enumerate(np.bincount(df.ravel(), minlength=256)) if c}
    meta = {
        "stage": STAGE,
        "fingerprint": fp,
        "validation_run": latest["run_dir"],
        "raw_sha256": latest["sha256"],
        "source_files": raw,
        "shape": list(image.shape),
        "channel_names": names,
        "dtype": str(image.dtype),
        "crs": str(prof["crs"]),
        "transform": list(prof["transform"])[:6],
        "valid_pixels": int(valid.sum()),
        "landcover_counts": lc_counts,
        "deforestation_counts": df_counts,
        "class_mapping": mapping,
        "config": cfg,
    }
    finish_run(run_dir, ROOT, meta)
    log(STAGE, f"wrote {run_dir.relative_to(ROOT)}: image {image.shape} {image.dtype}, channels {names}")
    log(STAGE, f"valid pixels {meta['valid_pixels']:,}; land cover {lc_counts}; deforestation {df_counts}")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args))
