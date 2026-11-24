"""Plots for reports (matplotlib, non-interactive Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.inference import LANDCOVER_COLORS  # noqa: E402

IGNORE_COLOR = (0.55, 0.55, 0.55)


def _lc_cmap(n: int) -> ListedColormap:
    return ListedColormap([np.array(LANDCOVER_COLORS[i]) / 255 for i in range(n)] + [IGNORE_COLOR])


def landcover_rgb(labels: np.ndarray, n_classes: int, ignore_index: int = 255) -> np.ndarray:
    lut = np.array([np.array(LANDCOVER_COLORS[i]) / 255 for i in range(n_classes)] + [IGNORE_COLOR])
    idx = np.where(labels == ignore_index, n_classes, np.clip(labels, 0, n_classes - 1))
    return lut[idx]


def stretch_rgb(image: np.ndarray, rgb_idx: Sequence[int], pct: tuple[float, float] = (2, 98)) -> np.ndarray:
    """(H, W, C) DN -> percentile-stretched RGB in [0, 1]."""
    rgb = image[..., list(rgb_idx)].astype(np.float32)
    lo, hi = np.percentile(rgb, pct, axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_split_map(cell_split: np.ndarray, keep: np.ndarray, pos_per_cell: np.ndarray, block_cells: int, out: Path) -> None:
    code = np.where(keep, cell_split, 3)
    cmap = ListedColormap(["#4c9be8", "#f2a541", "#d1495b", "#dddddd"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    axes[0].imshow(code, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    nr, nc = cell_split.shape
    for b in range(0, nr + 1, block_cells):
        axes[0].axhline(b - 0.5, color="k", lw=0.6)
    for b in range(0, nc + 1, block_cells):
        axes[0].axvline(b - 0.5, color="k", lw=0.6)
    axes[0].set_title("Spatial split (one cell = one 128 px patch)")
    axes[0].legend(handles=[Patch(color=c, label=l) for c, l in
                            zip(["#4c9be8", "#f2a541", "#d1495b", "#dddddd"], ["train", "val", "test", "buffer (dropped)"])],
                   loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=8)
    im = axes[1].imshow(pos_per_cell, cmap="magma", interpolation="nearest")
    axes[1].set_title("Forest-loss pixels per patch cell")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    _save(fig, out)


def plot_training_history(history: dict[str, list[float]], out: Path) -> None:
    keys = [("loss", "Total loss"), ("landcover_loss", "Land-cover loss"), ("deforestation_loss", "Deforestation loss"),
            ("landcover_miou", "Land-cover mIoU"), ("deforestation_f1", "Deforestation F1 @0.5")]
    keys = [(k, t) for k, t in keys if k in history]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (k, title) in zip(axes, keys):
        ep = np.arange(1, len(history[k]) + 1)
        ax.plot(ep, history[k], "o-", ms=3, label="train")
        if f"val_{k}" in history:
            ax.plot(ep, history[f"val_{k}"], "o-", ms=3, label="val")
        ax.set_title(title), ax.set_xlabel("epoch"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out: Path, title: str) -> None:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.1 * len(class_names) + 1.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"), ax.set_ylabel("Reference"), ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    _save(fig, out)


def plot_threshold_curve(rows: list[dict[str, float]], best: float, out: Path, title: str) -> None:
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ("precision", "recall", "f1", "iou"):
        ax.plot(t, [r[k] for r in rows], label=k)
    ax.axvline(best, color="k", ls="--", lw=1, label=f"chosen {best:.2f}")
    ax.set_xlabel("threshold"), ax.set_ylim(0, 1), ax.grid(alpha=0.3), ax.legend(), ax.set_title(title)
    _save(fig, out)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float, float]], out: Path) -> None:
    """curves: name -> (precision, recall, average_precision, prevalence)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (p, r, ap, prev) in curves.items():
        line, = ax.plot(r, p, label=f"{name} (AP {ap:.3f})")
        ax.axhline(prev, color=line.get_color(), ls=":", lw=1)
    ax.set_xlabel("recall"), ax.set_ylabel("precision"), ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_title("Deforestation precision-recall (dotted = random baseline)"), ax.grid(alpha=0.3), ax.legend()
    _save(fig, out)


def plot_patch_examples(samples: list[dict[str, Any]], class_names: Sequence[str], threshold: float, out: Path) -> None:
    """samples: dicts with pre_rgb, post_rgb, lc_true, lc_pred, df_true, df_prob, title."""
    n = len(samples)
    cols = ["Pre RGB", "Post RGB", "LC reference", "LC predicted", "Loss reference", "Loss probability", f"Loss pred (>= {threshold:.2f})"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.3 * n), squeeze=False)
    k = len(class_names)
    for i, s in enumerate(samples):
        df_ref = np.where(s["df_true"] == 255, 0.5, s["df_true"]).astype(float)
        pred = (s["df_prob"] >= threshold).astype(float)
        imgs = [s["pre_rgb"], s["post_rgb"], landcover_rgb(s["lc_true"], k), landcover_rgb(s["lc_pred"], k)]
        for j, im in enumerate(imgs):
            axes[i, j].imshow(im)
        axes[i, 4].imshow(df_ref, cmap="gray", vmin=0, vmax=1)
        axes[i, 5].imshow(s["df_prob"], cmap="magma", vmin=0, vmax=1)
        overlay = s["post_rgb"].copy()
        overlay[pred == 1] = [1, 0, 1]
        axes[i, 6].imshow(overlay)
        axes[i, 0].set_ylabel(s.get("title", ""), fontsize=7)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    handles = [Patch(color=np.array(LANDCOVER_COLORS[i]) / 255, label=n) for i, n in enumerate(class_names)]
    handles.append(Patch(color=IGNORE_COLOR, label="ignored (255)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out)


def plot_scene_maps(post_rgb: np.ndarray, landcover: np.ndarray, df_prob: np.ndarray, df_mask: np.ndarray,
                    class_names: Sequence[str], out: Path, step: int = 2) -> None:
    """Scene overview: post RGB, predicted land cover, loss probability, loss overlay (subsampled by `step`)."""
    rgb = post_rgb[::step, ::step]
    lc = landcover[::step, ::step]
    prob = df_prob[::step, ::step]
    mask = df_mask[::step, ::step]
    overlay = rgb.copy()
    overlay[mask == 1] = [1, 0, 1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5))
    axes[0].imshow(rgb), axes[0].set_title("Post composite (2023-24) RGB")
    axes[1].imshow(landcover_rgb(lc, len(class_names))), axes[1].set_title("Predicted land cover")
    im = axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1); axes[2].set_title("Forest-loss probability")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(overlay), axes[3].set_title("Predicted loss (magenta) on post RGB")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    handles = [Patch(color=np.array(LANDCOVER_COLORS[i]) / 255, label=n) for i, n in enumerate(class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out)
