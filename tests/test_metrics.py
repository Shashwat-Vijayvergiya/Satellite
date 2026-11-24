"""Metric and threshold-tuning tests (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest

import src.metrics as M


def test_confusion_matrix_skips_ignore_index() -> None:
    t = np.array([0, 1, 1, 2, 255, 255])
    p = np.array([0, 1, 2, 2, 0, 1])
    cm = M.confusion_matrix(t, p, 3)
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    assert cm.sum() == 4


def test_overall_accuracy_iou_dice_on_known_matrix() -> None:
    cm = np.array([[8, 2], [1, 9]])
    assert M.overall_accuracy(cm) == pytest.approx(17 / 20)
    np.testing.assert_allclose(M.per_class_iou(cm), [8 / 11, 9 / 12])
    np.testing.assert_allclose(M.per_class_f1(cm), [16 / 19, 18 / 21])
    np.testing.assert_allclose(M.per_class_precision(cm), [8 / 9, 9 / 11])
    np.testing.assert_allclose(M.per_class_recall(cm), [8 / 10, 9 / 10])


def test_per_class_iou_nan_for_absent_class_and_mean_ignores_nan() -> None:
    cm = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]])
    iou = M.per_class_iou(cm)
    assert np.isnan(iou[2]) and iou[0] == 1.0
    rep = M.landcover_report(cm, ["a", "b", "c"])
    assert rep["mean_iou"] == 1.0


def test_binary_report_at_threshold() -> None:
    t = np.array([1, 1, 0, 0, 255, 1])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.99, 0.7])
    r = M.binary_report(t, p, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3) and r["iou"] == pytest.approx(0.5) and r["dice"] == r["f1"]


def test_threshold_sweep_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 2000)
    t[:100] = 255
    p = np.clip(t * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)
    thr = np.round(np.arange(0.05, 0.96, 0.05), 6)
    rows = M.threshold_sweep(t, p, thr)
    for row in rows:
        ref = M.binary_report(t, p, row["threshold"])
        for k in ("precision", "recall", "f1", "iou"):
            assert row[k] == pytest.approx(ref[k], rel=1e-9, nan_ok=True)


def test_tune_threshold_returns_argmax_on_given_validation_arrays() -> None:
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 5000)
    p = np.clip(0.35 * t + rng.uniform(0, 0.65, 5000), 0, 1)
    best, rows = M.tune_threshold(t, p, 0.05, 0.95, 0.01)
    f1s = np.array([r["f1"] for r in rows])
    assert best == rows[int(np.nanargmax(f1s))]["threshold"]
    assert 0.05 <= best <= 0.95


def test_tune_threshold_without_positives_raises() -> None:
    with pytest.raises(RuntimeError):
        M.tune_threshold(np.zeros(10, int), np.linspace(0, 1, 10), 0.1, 0.9, 0.1)


def test_pr_auc_ignores_masked_pixels() -> None:
    from sklearn.metrics import average_precision_score

    t = np.array([1, 0, 1, 0, 255, 255])
    p = np.array([0.8, 0.3, 0.6, 0.7, 0.0, 1.0])
    assert M.pr_auc(t, p) == pytest.approx(average_precision_score(t[:4], p[:4]))


def test_keras_metrics_match_numpy() -> None:
    tf = pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=(2, 8, 8, 1)).astype(np.int32)
    y[0, :2] = 255
    probs = rng.dirichlet(np.ones(4), size=(2, 8, 8)).astype(np.float32)
    m = M.MaskedMeanIoU(4)
    m.update_state(tf.constant(y), tf.constant(probs))
    cm = M.confusion_matrix(y[..., 0], probs.argmax(-1), 4)
    assert float(m.result()) == pytest.approx(float(np.nanmean(M.per_class_iou(cm))), rel=1e-5)

    yd = rng.integers(0, 2, size=(2, 8, 8, 1)).astype(np.float32)
    yd[1, :3] = 255
    pd_ = rng.uniform(size=(2, 8, 8, 1)).astype(np.float32)
    f1 = M.MaskedBinaryScore("f1")
    f1.update_state(tf.constant(yd), tf.constant(pd_))
    assert float(f1.result()) == pytest.approx(M.binary_report(yd, pd_, 0.5)["f1"], rel=1e-5)
