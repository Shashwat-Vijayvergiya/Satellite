"""Masked loss tests (synthetic tensors only)."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.losses import MaskedBinarySegmentationLoss, MaskedWeightedCategoricalCrossentropy  # noqa: E402

RNG = np.random.default_rng(0)


def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def test_weighted_ce_matches_manual_and_ignores_masked_pixels() -> None:
    y = RNG.integers(0, 3, size=(2, 4, 4, 1)).astype(np.int32)
    y[0, 0, 0, 0] = 255
    p = _softmax(RNG.normal(size=(2, 4, 4, 3)))
    w = np.array([1.0, 2.0, 0.5], np.float32)
    loss = float(MaskedWeightedCategoricalCrossentropy(w)(y, p))
    m = y[..., 0] != 255
    ys = y[..., 0][m]
    ll = np.log(p[m][np.arange(ys.size), ys])
    expected = -(w[ys] * ll).sum() / w[ys].sum()
    assert loss == pytest.approx(expected, rel=1e-5)
    p2 = p.copy()
    p2[0, 0, 0] = [0.98, 0.01, 0.01]          # change prediction only at the ignored pixel
    assert float(MaskedWeightedCategoricalCrossentropy(w)(y, p2)) == pytest.approx(loss, rel=1e-6)


def test_weighted_ce_normalised_by_valid_pixels() -> None:
    y = np.zeros((1, 2, 2, 1), np.int32)
    p = _softmax(np.tile([[2.0, 0.0]], (1, 2, 2, 1)))
    base = float(MaskedWeightedCategoricalCrossentropy([1.0, 1.0])(y, p))
    y_pad = np.concatenate([y, np.full((1, 2, 2, 1), 255, np.int32)], axis=2)  # add ignored pixels
    p_pad = np.concatenate([p, _softmax(np.zeros((1, 2, 2, 2)))], axis=2)
    assert float(MaskedWeightedCategoricalCrossentropy([1.0, 1.0])(y_pad, p_pad)) == pytest.approx(base, rel=1e-6)


def test_weighted_ce_class_weight_changes_loss() -> None:
    y = np.array([[[[0], [1]]]], np.int32)
    p = _softmax(np.array([[[[3.0, 0.0], [3.0, 0.0]]]]))  # class 1 is badly predicted
    a = float(MaskedWeightedCategoricalCrossentropy([1.0, 1.0])(y, p))
    b = float(MaskedWeightedCategoricalCrossentropy([1.0, 5.0])(y, p))
    assert b > a


def test_bce_pos_weight_and_ignore() -> None:
    y = np.array([1, 0, 255, 0], np.float32).reshape(1, 2, 2, 1)
    p = np.array([0.6, 0.3, 0.99, 0.2], np.float32).reshape(1, 2, 2, 1)
    loss = MaskedBinarySegmentationLoss("bce", pos_weight=4.0, base_weight=1.0, dice_weight=0.0)
    expected = -(4 * np.log(0.6) + np.log(0.7) + np.log(0.8)) / 3
    assert float(loss(y, p)) == pytest.approx(expected, rel=1e-5)
    p2 = p.copy()
    p2[0, 1, 0, 0] = 0.01
    assert float(loss(y, p2)) == pytest.approx(expected, rel=1e-5)


def test_dice_perfect_prediction_near_zero_and_wrong_near_one() -> None:
    y = np.zeros((1, 8, 8, 1), np.float32)
    y[0, :4] = 1
    dice_only = MaskedBinarySegmentationLoss("bce", base_weight=0.0, dice_weight=1.0, smooth=1e-6)
    assert float(dice_only(y, np.clip(y, 1e-7, 1 - 1e-7))) < 1e-4
    assert float(dice_only(y, np.clip(1 - y, 1e-7, 1 - 1e-7))) > 0.99


def test_focal_gamma_zero_alpha_half_is_half_bce() -> None:
    y = RNG.integers(0, 2, size=(1, 4, 4, 1)).astype(np.float32)
    p = RNG.uniform(0.05, 0.95, size=(1, 4, 4, 1)).astype(np.float32)
    bce = MaskedBinarySegmentationLoss("bce", pos_weight=1.0, dice_weight=0.0)(y, p)
    focal = MaskedBinarySegmentationLoss("focal", focal_alpha=0.5, focal_gamma=0.0, dice_weight=0.0)(y, p)
    assert float(focal) == pytest.approx(0.5 * float(bce), rel=1e-5)


def test_all_ignored_batch_gives_zero_finite_loss() -> None:
    y_df = np.full((1, 4, 4, 1), 255, np.float32)
    p_df = np.full((1, 4, 4, 1), 0.3, np.float32)
    assert float(MaskedBinarySegmentationLoss("bce", pos_weight=5)(y_df, p_df)) == 0.0
    y_lc = np.full((1, 4, 4, 1), 255, np.int32)
    v = float(MaskedWeightedCategoricalCrossentropy([1, 1, 1])(y_lc, _softmax(np.zeros((1, 4, 4, 3)))))
    assert np.isfinite(v) and v == 0.0


def test_losses_round_trip_through_config() -> None:
    a = MaskedBinarySegmentationLoss("focal", pos_weight=3.0, focal_alpha=0.7)
    b = MaskedBinarySegmentationLoss.from_config(a.get_config())
    y = RNG.integers(0, 2, size=(1, 4, 4, 1)).astype(np.float32)
    p = RNG.uniform(0.1, 0.9, size=(1, 4, 4, 1)).astype(np.float32)
    assert float(a(y, p)) == pytest.approx(float(b(y, p)))
    c = MaskedWeightedCategoricalCrossentropy.from_config(MaskedWeightedCategoricalCrossentropy([1, 2]).get_config())
    assert c.class_weights == [1.0, 2.0]
