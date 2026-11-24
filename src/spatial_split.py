"""Geographic train/val/test split on square blocks, decided BEFORE patching.

Layout: the scene is cut into a fixed grid of patch cells (patch_size px,
from the top-left corner, partial cells at the right/bottom dropped). Cells
are grouped into square blocks of block_size_px. Whole blocks are assigned to
a split, so neighbouring patches always share a split unless they sit on a
block boundary, where a buffer drops cells:

    train cell within `buffer` cells of a val/test block -> dropped
    val cell within `buffer` cells of a test block       -> dropped
    test cells                                           -> always kept

Assignment: `n_candidates` seeded random block orders are cut into
test/val/train by cumulative patch count. Infeasible ones are rejected:
val/test need >= min_positive_pixels forest-loss pixels, and every split
needs >= min_class_pixels of every land-cover class. Among feasible
candidates, the one whose post-buffer shares of patches AND of forest-loss
pixels are closest to the target fractions wins. (Loss is concentrated in
the southern hills; matching patch shares alone tended to put most of it in
val/test and starve training of positives.)
Block assignment is therefore geographic (whole blocks), reproducible
(seed), and never a random split of neighbouring patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class CellGrid:
    height: int
    width: int
    patch: int
    block_cells: int

    @property
    def n_rows(self) -> int:
        return self.height // self.patch

    @property
    def n_cols(self) -> int:
        return self.width // self.patch

    @property
    def n_block_rows(self) -> int:
        return -(-self.n_rows // self.block_cells)

    @property
    def n_block_cols(self) -> int:
        return -(-self.n_cols // self.block_cells)

    @property
    def n_blocks(self) -> int:
        return self.n_block_rows * self.n_block_cols

    def cell_block_ids(self) -> np.ndarray:
        r = np.arange(self.n_rows)[:, None] // self.block_cells
        c = np.arange(self.n_cols)[None, :] // self.block_cells
        return (r * self.n_block_cols + c).astype(np.int32)


def make_grid(height: int, width: int, patch: int, block_size_px: int) -> CellGrid:
    if block_size_px % patch:
        raise ValueError("block_size_px must be a multiple of patch_size")
    return CellGrid(height, width, patch, block_size_px // patch)


def cell_label_stats(grid: CellGrid, deforest: np.ndarray, landcover: np.ndarray, valid: np.ndarray,
                     n_classes: int) -> dict[str, np.ndarray]:
    """Per-cell pixel counts: positives, negatives, valid input, labelled LC, LC per class."""
    p, nr, nc = grid.patch, grid.n_rows, grid.n_cols

    def per_cell(mask: np.ndarray) -> np.ndarray:
        return mask[: nr * p, : nc * p].reshape(nr, p, nc, p).sum(axis=(1, 3), dtype=np.int64)

    lc_counts = np.stack([per_cell(landcover == k) for k in range(n_classes)], axis=-1)
    return {
        "pos": per_cell(deforest == 1),
        "neg": per_cell(deforest == 0),
        "valid": per_cell(valid),
        "lc": lc_counts,
    }


def cell_splits(block_split: np.ndarray, cell_block: np.ndarray) -> np.ndarray:
    return block_split[cell_block]


def _near(mask: np.ndarray, radius: int) -> np.ndarray:
    """True where any True cell of `mask` is within Chebyshev distance `radius`."""
    if radius <= 0:
        return mask.copy()
    pad = np.pad(mask, radius)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            out |= pad[dr: dr + h, dc: dc + w]
    return out


def buffer_keep(cell_split: np.ndarray, buffer: int) -> np.ndarray:
    keep = np.ones(cell_split.shape, dtype=bool)
    if buffer <= 0:
        return keep
    near_val_or_test = _near(cell_split >= 1, buffer)
    near_test = _near(cell_split == 2, buffer)
    keep[(cell_split == 0) & near_val_or_test] = False
    keep[(cell_split == 1) & near_test] = False
    return keep


def summarize(cell_split: np.ndarray, keep: np.ndarray, stats: dict[str, np.ndarray], usable: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total = int((keep & usable).sum())
    for name, code in SPLIT_CODE.items():
        m = (cell_split == code) & keep & usable
        out[name] = {
            "patches": int(m.sum()),
            "patch_share": float(m.sum() / max(total, 1)),
            "positive_pixels": int(stats["pos"][m].sum()),
            "negative_pixels": int(stats["neg"][m].sum()),
            "landcover_pixels": [int(x) for x in stats["lc"][m].sum(axis=0)],
        }
    out["dropped_by_buffer"] = int((~keep & usable).sum())
    return out


def search_assignment(grid: CellGrid, stats: dict[str, np.ndarray], usable: np.ndarray, split_cfg: dict[str, Any],
                      seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (block_split codes per block, info). Raises RuntimeError if no candidate is feasible."""
    fr = split_cfg["fractions"]
    target = np.array([fr["train"], fr["val"], fr["test"]], dtype=float)
    if abs(target.sum() - 1) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fr}")
    cell_block = grid.cell_block_ids()
    block_patches = np.bincount(cell_block[usable], minlength=grid.n_blocks)
    total = block_patches.sum()
    if total == 0:
        raise RuntimeError("no usable patch cells")
    buffer = int(split_cfg["buffer_patches"])
    min_pos = split_cfg["min_positive_pixels"]
    min_cls = int(split_cfg["min_class_pixels"])
    rng = np.random.default_rng(seed)
    best, best_score, n_feasible = None, np.inf, 0
    for _ in range(int(split_cfg["n_candidates"])):
        order = rng.permutation(grid.n_blocks)
        start = (np.cumsum(block_patches[order]) - block_patches[order]) / total
        codes = np.where(start < target[2], 2, np.where(start < target[2] + target[1], 1, 0))
        block_split = np.empty(grid.n_blocks, dtype=np.int8)
        block_split[order] = codes
        cs = cell_splits(block_split, cell_block)
        keep = buffer_keep(cs, buffer)
        s = summarize(cs, keep, stats, usable)
        if s["val"]["positive_pixels"] < min_pos["val"] or s["test"]["positive_pixels"] < min_pos["test"]:
            continue
        if any(min(s[n]["landcover_pixels"]) < min_cls for n in SPLITS):
            continue
        n_feasible += 1
        shares = np.array([s[n]["patch_share"] for n in SPLITS])
        pos = np.array([s[n]["positive_pixels"] for n in SPLITS], dtype=float)
        pos_shares = pos / max(pos.sum(), 1.0)
        # match the target fractions for patches AND for forest-loss pixels
        score = float(np.abs(shares - target).sum() + np.abs(pos_shares - target).sum())
        if score < best_score:
            best, best_score = block_split.copy(), score
    if best is None:
        raise RuntimeError(
            f"No feasible block assignment among {split_cfg['n_candidates']} candidates "
            f"(min positives {min_pos}, min class pixels {min_cls}). Use smaller blocks or relax constraints."
        )
    return best, {"n_candidates": int(split_cfg["n_candidates"]), "n_feasible": n_feasible,
                  "score_patch_plus_positive_share_error": best_score}


def overlapping_train_windows(cell_split: np.ndarray, patch: int, stride: int) -> list[tuple[int, int]]:
    """Pixel offsets (row, col) of extra train windows on a `stride` grid.

    A window is kept only if every patch cell it touches is a kept TRAIN cell
    (code 0 in cell_split; dropped/val/test cells are -1/1/2), so extra windows
    never reach into buffer, validation or test areas. Windows that coincide
    with the regular patch grid are skipped (they already exist).
    """
    if stride <= 0 or patch % stride:
        raise ValueError("stride must be a positive divisor of patch")
    nr, nc = cell_split.shape
    out = []
    for r in range(0, nr * patch - patch + 1, stride):
        for c in range(0, nc * patch - patch + 1, stride):
            if r % patch == 0 and c % patch == 0:
                continue
            cells = cell_split[r // patch:(r + patch - 1) // patch + 1, c // patch:(c + patch - 1) // patch + 1]
            if (cells == 0).all():
                out.append((r, c))
    return out
