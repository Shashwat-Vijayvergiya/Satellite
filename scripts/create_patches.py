r"""Cut the 128 x 128 patches listed by the current split run (stride 128, no overlap).

Writes data/patches/patches_<timestamp>/{train,val,test}/<patch_id>.npz plus:
    manifest.csv     patch ID, source, split, path, land-cover class counts,
                     forest-loss positive count, valid pixel count, ...
    band_stats.json  robust per-channel normalisation stats from TRAIN patches only
    metadata.json    lineage (split + prep run IDs), parameters, counts

Usage:  python scripts/create_patches.py --config configs/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "create_patches"


def add_args(p) -> None:
    p.add_argument("--force", action="store_true", help="Rebuild even if inputs and config are unchanged")


def run(cfg, args) -> None:
    import pandas as pd

    from src.preprocessing import compute_band_stats
    from src.runs import data_lineage, finish_run, fingerprint, new_run_dir, reusable_run, write_json

    lin = data_lineage(cfg, ROOT, "split")
    prep_run, prep_meta = lin["prep"]
    split_run, split_meta = lin["split"]
    log(STAGE, f"using {prep_run.name} and {split_run.name}")
    fp = fingerprint(prep_run.name, split_run.name, cfg["patching"], cfg["s2"]["normalization"])
    reuse = None if args.force else reusable_run(ROOT / cfg["paths"]["patches_dir"], ROOT, fp)
    if reuse:
        log(STAGE, f"inputs and config unchanged: reusing {reuse.name} (use --force to rebuild)")
        return
    image = np.load(prep_run / "image.npy", mmap_mode="r")
    lc = np.load(prep_run / "landcover.npy", mmap_mode="r")
    df = np.load(prep_run / "deforest.npy", mmap_mode="r")
    valid = np.load(prep_run / "valid.npy", mmap_mode="r")
    windows = pd.read_csv(split_run / "windows.csv")
    p = int(split_meta["patch_size"])
    names = prep_meta["channel_names"]
    classes = cfg["worldcover"]["class_names"]

    run_dir = new_run_dir(ROOT / cfg["paths"]["patches_dir"], "patches")
    rows, train_pixels = [], []
    sources = ";".join(Path(v).name for v in prep_meta["source_files"].values())
    for w in windows.itertuples(index=False):
        sl = (slice(w.row_off, w.row_off + p), slice(w.col_off, w.col_off + p))
        img, l, d, v = np.array(image[sl]), np.array(lc[sl]), np.array(df[sl]), np.array(valid[sl])
        if img.shape[:2] != (p, p):
            raise RuntimeError(f"window {w.patch_id} is not {p}x{p}")
        rel_path = f"{w.split}/{w.patch_id}.npz"
        (run_dir / w.split).mkdir(exist_ok=True)
        np.savez(run_dir / rel_path, image=img, landcover=l, deforest=d)
        lc_bins = np.bincount(l.ravel(), minlength=256)
        row = {"patch_id": w.patch_id, "split": w.split, "path": rel_path, "source_prep_run": prep_run.name,
               "source_files": sources, "row_off": w.row_off, "col_off": w.col_off, "block_id": w.block_id,
               "x_min": w.x_min, "y_max": w.y_max, "valid_pixels": int(v.sum())}
        row.update({f"lc_{n}": int(lc_bins[i]) for i, n in enumerate(classes)})
        row["lc_ignore"] = int(lc_bins[255])
        row["df_positive"], row["df_negative"], row["df_ignore"] = int((d == 1).sum()), int((d == 0).sum()), int((d == 255).sum())
        rows.append(row)
        if w.split == "train":
            train_pixels.append(img[v])
    manifest = pd.DataFrame(rows)
    manifest.to_csv(run_dir / "manifest.csv", index=False)
    if not train_pixels:
        raise RuntimeError("no train patches")
    stats = compute_band_stats(np.concatenate(train_pixels), names, tuple(cfg["s2"]["normalization"]["percentiles"]))
    stats["computed_on"] = "train patches only (valid pixels)"
    write_json(run_dir / "band_stats.json", stats)

    counts = {}
    for s, g in manifest.groupby("split"):
        counts[s] = {"patches": int(len(g)), "df_positive": int(g["df_positive"].sum()), "df_negative": int(g["df_negative"].sum()),
                     "landcover": {n: int(g[f"lc_{n}"].sum()) for n in classes}}
    finish_run(run_dir, ROOT, {"stage": STAGE, "fingerprint": fp, "split_run_id": split_run.name, "prep_run_id": prep_run.name,
                               "patch_size": p, "stride": int(cfg["patching"]["stride"]), "channel_names": names,
                               "n_patches": int(len(manifest)), "counts": counts, "band_stats_file": "band_stats.json",
                               "config": cfg})
    for s in ("train", "val", "test"):
        c = counts.get(s, {"patches": 0, "df_positive": 0})
        log(STAGE, f"{s:5s}: {c['patches']} patches, loss pixels {c['df_positive']:,}")
    log(STAGE, f"wrote {len(manifest)} patches to {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args))
