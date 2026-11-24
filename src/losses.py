"""Masked losses for the multi-task U-Net (TensorFlow/Keras 3).

Every loss drops pixels whose label == ignore_index (255) and divides by the
VALID pixel count (or the summed class weight of valid pixels) of the whole
batch. A batch with many ignored pixels therefore does not dilute the loss,
and a fully ignored batch gives exactly 0.

Land cover:    class-weighted sparse categorical cross-entropy.
Deforestation: (positive-weighted BCE | focal) + soft Dice.
The functions return a scalar per batch; Keras passes scalars through
unchanged, so normalisation is global over the batch rather than per image.
"""

from __future__ import annotations

from typing import Any, Sequence

import keras
import tensorflow as tf

EPS = 1e-7


@keras.saving.register_keras_serializable(package="sat")
class MaskedWeightedCategoricalCrossentropy(keras.losses.Loss):
    """y_true int (B,H,W,1) in 0..K-1 or ignore; y_pred softmax (B,H,W,K)."""

    def __init__(self, class_weights: Sequence[float], ignore_index: int = 255, name: str = "masked_wce", **kw: Any):
        super().__init__(name=name, **kw)
        self.class_weights = [float(w) for w in class_weights]
        self.ignore_index = int(ignore_index)

    def call(self, y_true, y_pred):
        k = len(self.class_weights)
        y = tf.cast(tf.squeeze(y_true, -1), tf.int32)
        mask = tf.not_equal(y, self.ignore_index)
        y_safe = tf.where(mask, y, tf.zeros_like(y))
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), EPS, 1.0)
        log_p = tf.reduce_sum(tf.one_hot(y_safe, k) * tf.math.log(p), axis=-1)
        w = tf.gather(tf.constant(self.class_weights, tf.float32), y_safe) * tf.cast(mask, tf.float32)
        return -tf.reduce_sum(w * log_p) / tf.maximum(tf.reduce_sum(w), EPS)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "class_weights": self.class_weights, "ignore_index": self.ignore_index}


@keras.saving.register_keras_serializable(package="sat")
class MaskedBinarySegmentationLoss(keras.losses.Loss):
    """y_true float (B,H,W,1) in {0, 1, ignore}; y_pred sigmoid (B,H,W,1).

    total = base_weight * (weighted BCE or focal) + dice_weight * soft Dice loss
    """

    def __init__(self, kind: str = "bce", pos_weight: float = 1.0, base_weight: float = 1.0, dice_weight: float = 1.0,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0, smooth: float = 1.0, ignore_index: int = 255,
                 name: str = "masked_binary_seg", **kw: Any):
        super().__init__(name=name, **kw)
        if kind not in ("bce", "focal"):
            raise ValueError(f"kind must be 'bce' or 'focal', got {kind!r}")
        self.kind, self.pos_weight, self.base_weight, self.dice_weight = kind, float(pos_weight), float(base_weight), float(dice_weight)
        self.focal_alpha, self.focal_gamma, self.smooth, self.ignore_index = float(focal_alpha), float(focal_gamma), float(smooth), int(ignore_index)

    def call(self, y_true, y_pred):
        y = tf.cast(y_true, tf.float32)
        mask = tf.cast(tf.not_equal(y, float(self.ignore_index)), tf.float32)
        y = y * mask  # ignored -> 0, excluded by mask anyway
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), EPS, 1.0 - EPS)
        n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
        if self.kind == "bce":
            per_px = -(self.pos_weight * y * tf.math.log(p) + (1.0 - y) * tf.math.log(1.0 - p))
        else:
            p_t = y * p + (1.0 - y) * (1.0 - p)
            alpha_t = y * self.focal_alpha + (1.0 - y) * (1.0 - self.focal_alpha)
            per_px = -alpha_t * tf.pow(1.0 - p_t, self.focal_gamma) * tf.math.log(p_t)
        base = tf.reduce_sum(per_px * mask) / n_valid
        inter = tf.reduce_sum(p * y * mask)
        denom = tf.reduce_sum(p * mask) + tf.reduce_sum(y * mask)
        dice = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        has_valid = tf.cast(tf.reduce_sum(mask) > 0, tf.float32)
        return has_valid * (self.base_weight * base + self.dice_weight * dice)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "kind": self.kind, "pos_weight": self.pos_weight, "base_weight": self.base_weight,
                "dice_weight": self.dice_weight, "focal_alpha": self.focal_alpha, "focal_gamma": self.focal_gamma,
                "smooth": self.smooth, "ignore_index": self.ignore_index}


def build_losses(loss_cfg: dict[str, Any], class_weights: Sequence[float], pos_weight: float) -> tuple[dict[str, keras.losses.Loss], dict[str, float]]:
    """Return ({"landcover": loss, "deforestation": loss}, task weights) for model.compile."""
    d = loss_cfg["deforestation"]
    kinds = {"bce_dice": "bce", "focal_dice": "focal"}
    if d["type"] not in kinds:
        raise ValueError(f"loss.deforestation.type must be one of {list(kinds)}, got {d['type']!r}")
    losses = {
        "landcover": MaskedWeightedCategoricalCrossentropy(class_weights, loss_cfg["ignore_index"]),
        "deforestation": MaskedBinarySegmentationLoss(
            kind=kinds[d["type"]], pos_weight=pos_weight, base_weight=d["base_weight"], dice_weight=d["dice_weight"],
            focal_alpha=d["focal_alpha"], focal_gamma=d["focal_gamma"], ignore_index=loss_cfg["ignore_index"]),
    }
    tw = loss_cfg["task_weights"]
    return losses, {"landcover": float(tw["landcover"]), "deforestation": float(tw["deforestation"])}
