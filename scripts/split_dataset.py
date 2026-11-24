r"""Assign geographic blocks to train/val/test BEFORE any patch is extracted.

Uses the latest prepare_dataset run. Writes data/splits/split_<timestamp>/:
    block_assignment.csv   one row per 512 px block (split, pixel/map bounds, kept patches)
    windows.csv            the patch windows that create_patches will cut, with their split
    summary.json, split_map.png, metadata.json
Fails (exit 1) if no assignment gives val and test enough forest-loss pixels
and every split every land-cover class.

Usage:  python scripts/split_dataset.py --config configs/config.yaml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "split_dataset"


def add_args(p) -> None:
    p.add_argument("--force", action="store_true", help="Rebuild even if inputs and config are unchanged")


def run(cfg, args) -> None:
    import pandas as pd

    from src import spatial_split as S
    from src.runs import data_lineage, finish_run, new_run_dir, point_latest, reusable_run, split_fingerprint, write_json
    from src.visualization import plot_split_map

    prep_run, prep_meta = data_lineage(cfg, ROOT, "prep")["prep"]
    log(STAGE, f"using {prep_run.name}")
    fp = split_fingerprint(cfg, prep_run.name)
    reuse = None if args.force else reusable_run(ROOT / cfg["paths"]["splits_dir"], ROOT, fp)
    if reuse:
        point_latest(reuse, ROOT)
        log(STAGE, f"inputs and config unchanged: reusing {reuse.name} (use --force to rebuild)")
        return
    lc = np.load(prep_run / "landcover.npy", mmap_mode="r")
    df = np.load(prep_run / "deforest.npy", mmap_mode="r")
    valid = np.load(prep_run / "valid.npy", mmap_mode="r")
    h, w = lc.shape
    p = int(cfg["patching"]["patch_size"])
    if int(cfg["patching"]["stride"]) != p:
        raise ValueError("this pipeline cuts non-overlapping patches: patching.stride must equal patch_size")
    grid = S.make_grid(h, w, p, int(cfg["split"]["block_size_px"]))
    k = len(cfg["worldcover"]["class_names"])
    stats = S.cell_label_stats(grid, np.asarray(df), np.asarray(lc), np.asarray(valid), k)
    px = p * p
    labelled = np.maximum(stats["lc"].sum(-1), stats["pos"] + stats["neg"]) / px
    usable = (stats["valid"] / px >= cfg["patching"]["min_valid_fraction"]) & (labelled >= cfg["patching"]["min_labelled_fraction"])
    log(STAGE, f"grid {grid.n_rows} x {grid.n_cols} patch cells, {grid.n_blocks} blocks of {grid.block_cells}x{grid.block_cells}; "
               f"usable cells {int(usable.sum())}/{usable.size}")

    seed = int(cfg["split"]["seed"])
    block_split, info = S.search_assignment(grid, stats, usable, cfg["split"], seed)
    cell_block = grid.cell_block_ids()
    cs = S.cell_splits(block_split, cell_block)
    keep = S.buffer_keep(cs, int(cfg["split"]["buffer_patches"]))
    summary = S.summarize(cs, keep, stats, usable)
    summary["unusable_cells"] = int((~usable).sum())
    summary["search"] = info

    names = {v: k_ for k_, v in S.SPLIT_CODE.items()}
    g = cfg["grid"]
    res = g["pixel_size_m"]
    rows = []
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            if not (usable[r, c] and keep[r, c]):
                continue
            rows.append({"patch_id": f"r{r:02d}_c{c:02d}", "split": names[int(cs[r, c])], "cell_row": r, "cell_col": c,
                         "row_off": r * p, "col_off": c * p, "block_id": int(cell_block[r, c]),
                         "x_min": g["x0"] + c * p * res, "y_max": g["ytop"] - r * p * res})
    windows = pd.DataFrame(rows)
    blocks = []
    bc = grid.block_cells
    for b in range(grid.n_blocks):
        br, bcol = divmod(b, grid.n_block_cols)
        r0, c0 = br * bc * p, bcol * bc * p
        r1, c1 = min(r0 + bc * p, grid.n_rows * p), min(c0 + bc * p, grid.n_cols * p)
        blocks.append({"block_id": b, "block_row": br, "block_col": bcol, "split": names[int(block_split[b])],
                       "row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1,
                       "x_min": g["x0"] + c0 * res, "x_max": g["x0"] + c1 * res,
                       "y_min": g["ytop"] - r1 * res, "y_max": g["ytop"] - r0 * res,
                       "kept_patches": int((windows["block_id"] == b).sum())})

    run_dir = new_run_dir(ROOT / cfg["paths"]["splits_dir"], "split")
    windows.to_csv(run_dir / "windows.csv", index=False)
    pd.DataFrame(blocks).to_csv(run_dir / "block_assignment.csv", index=False)
    np.save(run_dir / "cell_split.npy", np.where(usable & keep, cs, -1).astype(np.int8))
    write_json(run_dir / "summary.json", summary)
    plot_split_map(cs, keep & usable, stats["pos"], grid.block_cells, run_dir / "split_map.png")
    plots = ROOT / cfg["evaluation"]["plots_dir"]
    plots.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "split_map.png", plots / "split_map.png")
    finish_run(run_dir, ROOT, {"stage": STAGE, "fingerprint": fp, "prep_run_id": prep_run.name, "seed": seed, "split_config": cfg["split"],
                               "patch_size": p, "grid_cells": [grid.n_rows, grid.n_cols], "summary": summary, "config": cfg})
    for n in S.SPLITS:
        s = summary[n]
        log(STAGE, f"{n:5s}: {s['patches']:4d} patches ({s['patch_share']:.1%}), loss pixels {s['positive_pixels']:,}, "
                   f"no-loss pixels {s['negative_pixels']:,}, land-cover pixels per class {s['landcover_pixels']}")
    log(STAGE, f"buffer dropped {summary['dropped_by_buffer']} cells; feasible candidates {info['n_feasible']}/{info['n_candidates']}")
    log(STAGE, f"wrote {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args))
