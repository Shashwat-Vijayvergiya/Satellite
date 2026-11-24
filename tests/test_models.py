"""Model output shapes and .keras save/load round trip (synthetic inputs only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from src.losses import build_losses  # noqa: E402
from src.metrics import MaskedBinaryScore, MaskedMeanIoU  # noqa: E402
from src.models.multitask_unet import build_multitask_unet  # noqa: E402
from src.models.unet import build_unet  # noqa: E402


@pytest.mark.parametrize("mode", ["separate", "shared"])
def test_multitask_output_shapes(mode: str) -> None:
    m = build_multitask_unet((32, 32, 12), 5, base_filters=4, depth=2, decoder_mode=mode)
    out = m.predict_on_batch(np.zeros((2, 32, 32, 12), np.float32))
    assert set(out) == {"landcover", "deforestation"}
    assert out["landcover"].shape == (2, 32, 32, 5) and out["deforestation"].shape == (2, 32, 32, 1)
    np.testing.assert_allclose(out["landcover"].sum(-1), 1.0, rtol=1e-5)
    assert (out["deforestation"] >= 0).all() and (out["deforestation"] <= 1).all()


def test_unet_baseline_shape_and_bad_size() -> None:
    m = build_unet((32, 32, 12), 5, base_filters=4, depth=2)
    assert m.output_shape == (None, 32, 32, 5)
    with pytest.raises(ValueError):
        build_unet((30, 30, 12), 5, depth=2)


def test_compiled_model_save_load_round_trip(tmp_path: Path) -> None:
    import src.metrics  # noqa: F401

    loss_cfg = {"ignore_index": 255, "task_weights": {"landcover": 1.0, "deforestation": 1.0},
                "deforestation": {"type": "bce_dice", "base_weight": 1.0, "dice_weight": 1.0, "focal_alpha": 0.75, "focal_gamma": 2.0}}
    losses, tw = build_losses(loss_cfg, [1.0, 2.0, 1.0, 3.0, 1.0], 5.0)
    m = build_multitask_unet((32, 32, 12), 5, base_filters=4, depth=2)
    m.compile(optimizer="adam", loss=losses, loss_weights=tw,
              metrics={"landcover": [MaskedMeanIoU(5)], "deforestation": [MaskedBinaryScore("f1")]})
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 32, 32, 12)).astype(np.float32)
    y = {"landcover": rng.integers(0, 5, (4, 32, 32, 1)).astype(np.int32),
         "deforestation": rng.integers(0, 2, (4, 32, 32, 1)).astype(np.float32)}
    hist = m.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(hist.history["loss"][0])
    assert "landcover_miou" in hist.history and "deforestation_f1" in hist.history
    path = tmp_path / "m.keras"
    m.save(path)
    m2 = keras.models.load_model(path)
    a, b = m.predict_on_batch(x), m2.predict_on_batch(x)
    np.testing.assert_allclose(a["landcover"], b["landcover"], atol=1e-6)
    assert m2.loss["deforestation"].pos_weight == 5.0
