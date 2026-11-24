"""Raw-data validation tests.

Synthetic rasters are written to pytest's tmp_path ONLY to prove the
validator rejects bad inputs; they are never used as project data.
The last test checks the REAL data in data/raw/ against the latest
validation report (skipped if data/raw/ or the report is absent).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

from src import load_config  # noqa: E402
from src import validation as V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W, H = 24, 20


def _write(path: Path, arr: np.ndarray, cfg: dict, names=None, transform=None) -> None:
    g = cfg["grid"]
    t = transform or from_origin(g["x0"], g["ytop"], g["pixel_size_m"], g["pixel_size_m"])
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=arr.shape[0],
                       dtype=arr.dtype, crs=g["crs"], transform=t) as dst:
        dst.write(arr)
        if names:
            dst.descriptions = tuple(names)


@pytest.fixture()
def synth(tmp_path: Path):
    """Build a small, internally consistent synthetic dataset; return (cfg, writer)."""
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "config.yaml"))
    cfg["grid"].update(width=W, height=H)
    for k in cfg["paths"]["raw"]:
        cfg["paths"]["raw"][k] = str(tmp_path / Path(cfg["paths"]["raw"][k]).name)
    cfg["validation"]["report_root"] = str(tmp_path / "reports")
    rng = np.random.default_rng(0)

    s2 = rng.integers(100, 4000, size=(7, H, W), dtype=np.uint16)
    s2[6] = rng.integers(10, 30, size=(H, W), dtype=np.uint16)
    wc = rng.choice(np.array([10, 30, 40, 50, 80], dtype=np.uint8), size=(1, H, W))
    tc = rng.integers(0, 101, size=(H, W)).astype(np.uint8)
    ly = rng.choice(np.array([0, 0, 0, 5, 21, 22, 23, 24], dtype=np.uint8), size=(H, W))
    dm = np.where(rng.random((H, W)) < 0.95, 1, 2).astype(np.uint8)
    lt = np.full((H, W), 255, np.uint8)
    lt[(dm == 1) & (ly == 0)] = 0
    lt[(dm == 1) & np.isin(ly, [21, 22, 23]) & (tc >= 30)] = 1
    data = {"s2_pre": s2, "s2_post": s2.copy(), "worldcover": wc, "hansen": np.stack([tc, ly, dm, lt])}

    def write(mutate=None, transform_for=None):
        d = copy.deepcopy(data)
        if mutate:
            mutate(d)
        names = {"s2_pre": cfg["s2"]["bands"], "s2_post": cfg["s2"]["bands"], "hansen": cfg["hansen"]["bands"]}
        for k, arr in d.items():
            t = None
            if transform_for == k:
                t = from_origin(cfg["grid"]["x0"] + 30, cfg["grid"]["ytop"], 30, 30)
            _write(Path(cfg["paths"]["raw"][k]), arr, cfg, names.get(k), t)
        g = cfg["grid"]
        b = transform_bounds(g["crs"], "EPSG:4326", g["x0"], g["ytop"] - H * 30, g["x0"] + W * 30, g["ytop"], densify_pts=101)
        props = {"crs": g["crs"], "width_px": W, "height_px": H, "x0": g["x0"], "ytop": g["ytop"], "scale_m": 30,
                 "s2_bands": ",".join(cfg["s2"]["bands"]), "hansen_bands": ",".join(cfg["hansen"]["bands"]),
                 "tree_thresh": 30, "loss_years": "21-23", "tiles_x": 1, "tiles_y": 1}
        ring = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]
        Path(cfg["paths"]["raw"]["aoi"]).write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
        return V.validate_raw_data(cfg, tmp_path)

    return cfg, write


def _has(rep, text: str) -> bool:
    return any(text in e for e in rep.errors)


def test_consistent_synthetic_dataset_passes(synth) -> None:
    _, write = synth
    rep = write()
    assert rep.passed, rep.errors


def test_constant_band_is_hard_error(synth) -> None:
    _, write = synth
    rep = write(lambda d: d["s2_post"].__setitem__(2, 777))  # B4 constant
    assert _has(rep, "s2_post/B4: constant")


def test_constant_band_gives_nonzero_exit(synth, tmp_path: Path) -> None:
    import scripts.inspect_rasters as cli

    cfg, write = synth
    write(lambda d: d["s2_pre"].__setitem__(0, 500))
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    assert cli.main(["--config", str(cfg_file)]) == 1
    latest = json.loads((Path(cfg["validation"]["report_root"]) / "latest.json").read_text())
    assert latest["status"] == "failed"


def test_misaligned_transform_is_hard_error(synth) -> None:
    _, write = synth
    rep = write(transform_for="worldcover")
    assert _has(rep, "worldcover: transform")


def test_loss_target_inconsistent_with_definition_is_hard_error(synth) -> None:
    _, write = synth

    def flip(d):
        lt = d["hansen"][3]
        idx = np.argwhere(lt == 0)[0]
        lt[tuple(idx)] = 1

    rep = write(flip)
    assert _has(rep, "hansen/loss_target: 1 pixels disagree")


def test_unexpected_worldcover_id_is_hard_error(synth) -> None:
    _, write = synth
    rep = write(lambda d: d["worldcover"].__setitem__((0, 0, 0), 70))
    assert _has(rep, "worldcover: unexpected class IDs [70]")


def test_invalid_loss_target_value_is_hard_error(synth) -> None:
    _, write = synth
    rep = write(lambda d: d["hansen"].__setitem__((3, 0, 0), 7))
    assert _has(rep, "hansen/loss_target: unexpected values [7]")


def test_missing_file_is_hard_error(synth) -> None:
    cfg, write = synth
    write()
    Path(cfg["paths"]["raw"]["hansen"]).unlink()
    rep = V.validate_raw_data(cfg, Path("/"))
    assert _has(rep, "hansen: file not found")


def test_split_tiles_are_detected(synth) -> None:
    cfg, write = synth
    write()
    p = Path(cfg["paths"]["raw"]["s2_pre"])
    (p.parent / f"{p.stem}-0000000000-0000000000.tif").write_bytes(b"x")
    rep = V.validate_raw_data(cfg, Path("/"))
    assert _has(rep, "s2_pre: both")


def test_guard_rejects_changed_raw_file(synth) -> None:
    cfg, write = synth
    rep = write()
    V.write_report(rep, cfg, Path("/"), Path("cfg.yaml"))
    V.require_validated_raw_data(cfg, Path("/"))  # passes
    with open(cfg["paths"]["raw"]["aoi"], "a") as f:
        f.write(" ")
    with pytest.raises(RuntimeError, match="changed or is missing"):
        V.require_validated_raw_data(cfg, Path("/"))


def test_real_raw_data_matches_latest_passed_validation() -> None:
    cfg = load_config(ROOT / "configs" / "config.yaml")
    latest = ROOT / cfg["validation"]["report_root"] / "latest.json"
    if not latest.is_file() or not (ROOT / cfg["paths"]["raw"]["s2_pre"]).is_file():
        pytest.skip("real data or validation report not present")
    V.require_validated_raw_data(cfg, ROOT)
