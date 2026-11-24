"""Masked segmentation metrics.

NumPy functions are used for the final evaluation; the Keras Metric classes
give per-task progress during training. Every metric skips pixels whose
label == ignore_index (255). The deforestation threshold is tuned on the
VALIDATION split only (tune_threshold) and then applied unchanged to test.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import tensorflow as tf

IGNORE = 255


# ----------------------------------------------------------------------------
# NumPy (evaluation)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_index: int = IGNORE) -> np.ndarray:
    """(K, K) int64, rows = reference, cols = prediction; ignored pixels skipped."""
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    m = t != ignore_index
    t, p = t[m], p[m]
    if t.size and (t.max() >= num_classes or p.max() >= num_classes or p.min() < 0):
        raise ValueError("labels/predictions outside 0..K-1")
    return np.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def overall_accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    return _safe_div(tp, cm.sum(0) + cm.sum(1) - tp)


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    return _safe_div(np.diag(cm), cm.sum(0))


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    return _safe_div(np.diag(cm), cm.sum(1))


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """Per-class F1, identical to the per-class pixel Dice coefficient."""
    tp = np.diag(cm)
    return _safe_div(2 * tp, cm.sum(0) + cm.sum(1))


def landcover_report(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    iou, f1 = per_class_iou(cm), per_class_f1(cm)
    prec, rec = per_class_precision(cm), per_class_recall(cm)
    return {
        "overall_accuracy": overall_accuracy(cm),
        "mean_iou": float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
        "mean_dice": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
        "macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
        "per_class": {n: {"iou": float(iou[i]), "dice": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "reference_pixels": int(cm[i].sum()), "predicted_pixels": int(cm[:, i].sum())}
                      for i, n in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "n_pixels": int(cm.sum()),
    }


def _valid_binary(y_true: np.ndarray, prob: np.ndarray, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    m = t != ignore_index
    return t[m].astype(np.int8), p[m]


def binary_report(y_true: np.ndarray, prob: np.ndarray, threshold: float, ignore_index: int = IGNORE) -> dict[str, Any]:
    t, p = _valid_binary(y_true, prob, ignore_index)
    pred = p >= threshold
    tp = int((pred & (t == 1)).sum())
    fp = int((pred & (t == 0)).sum())
    fn = int((~pred & (t == 1)).sum())
    tn = int((~pred & (t == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "dice": f1, "iou": iou,
            "accuracy": (tp + tn) / max(t.size, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "confusion_matrix": [[tn, fp], [fn, tp]], "n_pixels": int(t.size), "positive_pixels": int((t == 1).sum())}


def pr_auc(y_true: np.ndarray, prob: np.ndarray, ignore_index: int = IGNORE) -> float:
    """Average precision (area under the precision-recall curve) over valid pixels."""
    from sklearn.metrics import average_precision_score

    t, p = _valid_binary(y_true, prob, ignore_index)
    if t.size == 0 or t.min() == t.max():
        return float("nan")
    return float(average_precision_score(t, p))


def pr_curve(y_true: np.ndarray, prob: np.ndarray, ignore_index: int = IGNORE) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import precision_recall_curve

    t, p = _valid_binary(y_true, prob, ignore_index)
    precision, recall, _ = precision_recall_curve(t, p)
    return precision, recall


def threshold_sweep(y_true: np.ndarray, prob: np.ndarray, thresholds: np.ndarray, ignore_index: int = IGNORE) -> list[dict[str, float]]:
    """Precision/recall/F1/IoU at every threshold (pred = prob >= t), via sorted cumulative counts."""
    t, p = _valid_binary(y_true, prob, ignore_index)
    order = np.argsort(p)
    p_sorted, t_sorted = p[order], t[order]
    pos_total = int((t == 1).sum())
    cum_pos_below = np.concatenate([[0], np.cumsum(t_sorted == 1)])
    rows = []
    for thr in np.asarray(thresholds, dtype=float):
        k = int(np.searchsorted(p_sorted, thr, side="left"))  # count with prob < thr
        n_pred = p.size - k
        tp = pos_total - int(cum_pos_below[k])
        fp, fn = n_pred - tp, int(cum_pos_below[k])
        rows.append({"threshold": float(thr),
                     "precision": tp / n_pred if n_pred else float("nan"),
                     "recall": tp / pos_total if pos_total else float("nan"),
                     "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
                     "iou": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
                     "predicted_positive_pixels": int(n_pred)})
    return rows


def tune_threshold(y_true_val: np.ndarray, prob_val: np.ndarray, start: float, stop: float, step: float,
                   metric: str = "f1", ignore_index: int = IGNORE) -> tuple[float, list[dict[str, float]]]:
    """Threshold maximising `metric` on VALIDATION data (ties -> lowest threshold)."""
    t, _ = _valid_binary(y_true_val, prob_val, ignore_index)
    if not (t == 1).any():
        raise RuntimeError("threshold tuning needs positive (forest-loss) pixels in the validation data")
    thresholds = np.round(np.arange(start, stop + step / 2, step), 6)
    rows = threshold_sweep(y_true_val, prob_val, thresholds, ignore_index)
    scores = np.array([r[metric] for r in rows], dtype=float)
    if not np.isfinite(scores).any():
        raise RuntimeError(f"threshold tuning failed: {metric} undefined at every threshold (no positives?)")
    return float(rows[int(np.nanargmax(scores))]["threshold"]), rows


# ----------------------------------------------------------------------------
# Keras (training progress)
# ----------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="sat")
class MaskedMeanIoU(keras.metrics.Metric):
    """Streaming mean IoU over classes seen so far; ignores label 255."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE, name: str = "miou", **kw: Any):
        super().__init__(name=name, **kw)
        self.num_classes, self.ignore_index = int(num_classes), int(ignore_index)
        self.cm = self.add_variable(shape=(self.num_classes, self.num_classes), initializer="zeros", name="cm", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        pred = tf.reshape(tf.argmax(y_pred, axis=-1, output_type=tf.int32), [-1])
        mask = tf.not_equal(y, self.ignore_index)
        y, pred = tf.boolean_mask(y, mask), tf.boolean_mask(pred, mask)
        self.cm.assign_add(tf.math.confusion_matrix(y, pred, num_classes=self.num_classes, dtype=tf.float32))

    def result(self):
        tp = tf.linalg.diag_part(self.cm)
        union = tf.reduce_sum(self.cm, 0) + tf.reduce_sum(self.cm, 1) - tp
        present = tf.cast(union > 0, tf.float32)
        iou = tp / tf.maximum(union, 1.0)
        return tf.math.divide_no_nan(tf.reduce_sum(iou * present), tf.reduce_sum(present))

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "num_classes": self.num_classes, "ignore_index": self.ignore_index}


@keras.saving.register_keras_serializable(package="sat")
class MaskedBinaryScore(keras.metrics.Metric):
    """Streaming precision / recall / f1 / iou at a fixed threshold; ignores label 255."""

    def __init__(self, score: str = "f1", threshold: float = 0.5, ignore_index: int = IGNORE, name: str | None = None, **kw: Any):
        if score not in ("precision", "recall", "f1", "iou"):
            raise ValueError(f"unknown score {score!r}")
        super().__init__(name=name or score, **kw)
        self.score, self.threshold, self.ignore_index = score, float(threshold), int(ignore_index)
        self.tp = self.add_variable(shape=(), initializer="zeros", name="tp", dtype="float32")
        self.fp = self.add_variable(shape=(), initializer="zeros", name="fp", dtype="float32")
        self.fn = self.add_variable(shape=(), initializer="zeros", name="fn", dtype="float32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        mask = tf.not_equal(y, float(self.ignore_index))
        y, pred = tf.boolean_mask(y, mask), tf.cast(tf.boolean_mask(p, mask) >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(pred * y))
        self.fp.assign_add(tf.reduce_sum(pred * (1.0 - y)))
        self.fn.assign_add(tf.reduce_sum((1.0 - pred) * y))

    def result(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        if self.score == "precision":
            return tp / tf.maximum(tp + fp, 1.0)
        if self.score == "recall":
            return tp / tf.maximum(tp + fn, 1.0)
        if self.score == "iou":
            return tp / tf.maximum(tp + fp + fn, 1.0)
        return 2 * tp / tf.maximum(2 * tp + fp + fn, 1.0)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "score": self.score, "threshold": self.threshold, "ignore_index": self.ignore_index}
