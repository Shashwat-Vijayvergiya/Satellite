"""Tiled inference and georeferenced output tests (synthetic arrays only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from src.inference import area_summary, predict_scene, tile_origins, write_geotiff  # noqa: E402


def test_tile_origins_cover_whole_axis() -> None:
    for length in (128, 129, 300, 3704):
        starts = tile_origins(length, 128, 96)
        assert starts[0] == 0 and starts[-1] == max(length - 128, 0)
        assert all(b - a <= 96 for a, b in zip(starts, starts[1:]))
    assert tile_origins(50, 128, 96) == [0]


def _pixelwise(batch: np.ndarray) -> dict[str, np.ndarray]:
    """A per-pixel 'model': output depends only on the pixel itself, so tiling must not change it."""
    s = batch.sum(-1, keepdims=True)
    lc = np.concatenate([np.tanh(s) + 2, np.ones_like(s)], -1)
    lc = lc / lc.sum(-1, keepdims=True)
    return {"landcover": lc, "deforestation": 1 / (1 + np.exp(-s))}


@pytest.mark.parametrize("shape,overlap", [((300, 260, 3), 32), ((128, 128, 3), 0), ((100, 90, 3), 16)])
def test_tiled_prediction_equals_direct_application(shape, overlap) -> None:
    img = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    lc, df = predict_scene(_pixelwise, img, 128, overlap, 4, 2)
    ref = _pixelwise(img[None])
    assert lc.shape == shape[:2] + (2,) and df.shape == shape[:2]
    np.testing.assert_allclose(lc, ref["landcover"][0], rtol=1e-5)
    np.testing.assert_allclose(df, ref["deforestation"][0, ..., 0], rtol=1e-5)


def test_geotiff_preserves_crs_transform_and_values(tmp_path: Path) -> None:
    t = from_origin(499980, 2964630, 30, 30)
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
    p = tmp_path / "lc.tif"
    write_geotiff(p, arr, "EPSG:32646", t, nodata=255, colormap={0: (0, 100, 0)}, tags={"model_run": "x"})
    with rasterio.open(p) as src:
        assert src.crs.to_string() == "EPSG:32646"
        assert tuple(src.transform)[:6] == tuple(t)[:6]
        assert src.nodata == 255 and src.tags()["model_run"] == "x"
        np.testing.assert_array_equal(src.read(1), arr)
        assert src.colormap(1)[0][:3] == (0, 100, 0)


def test_area_summary_units() -> None:
    lc = np.array([[0, 0], [1, 255]], np.uint8)
    mask = np.array([[1, 0], [0, 255]], np.uint8)
    valid = np.array([[True, True], [True, False]])
    s = area_summary(lc, mask, valid, ["tree", "grass"], 900.0)
    assert s["landcover"]["tree"]["pixels"] == 2 and s["landcover"]["tree"]["area_km2"] == pytest.approx(0.0018)
    assert s["landcover"]["tree"]["area_ha"] == pytest.approx(0.18)
    assert s["deforestation"]["loss_pixels"] == 1 and s["valid_pixels"] == 3
