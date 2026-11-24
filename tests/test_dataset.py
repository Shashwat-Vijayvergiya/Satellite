"""Preprocessing, spatial split, patch windows and tf.data tests (synthetic arrays only)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src import load_config
from src import preprocessing as P
from src import spatial_split as S
from src.dataset import augment, compute_class_weights, compute_pos_weight, make_dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "config.yaml")


# ---------------------------------------------------------------- remapping
def test_worldcover_remap_matches_config_mapping(cfg) -> None:
    raw = np.array([[0, 10, 20, 30], [40, 50, 60, 80], [90, 10, 40, 80]], dtype=np.uint8)
    out = P.remap_worldcover(raw, cfg)
    expected = np.array([[255, 0, 255, 1], [2, 3, 255, 4], [255, 0, 2, 4]], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_unknown_worldcover_id_raises(cfg) -> None:
    with pytest.raises(ValueError, match=r"\[70\]"):
        P.remap_worldcover(np.array([[10, 70]], dtype=np.uint8), cfg)


def test_landcover_ignored_where_loss_target_is_1_or_invalid(cfg) -> None:
    lc = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    lt = np.array([[1, 0], [255, 0]], dtype=np.uint8)
    valid = np.array([[True, True], [True, False]])
    out = P.build_landcover_target(lc, lt, valid, cfg)
    np.testing.assert_array_equal(out, [[255, 1], [2, 255]])
    df = P.build_deforestation_target(lt, valid, cfg)
    np.testing.assert_array_equal(df, [[1, 0], [255, 255]])


def test_valid_mask_requires_data_in_both_dates(cfg) -> None:
    pre = np.full((7, 2, 2), 500, np.uint16)
    post = pre.copy()
    pre[:6, 0, 0] = 0          # all reflectance 0 in pre
    post[6, 0, 1] = 0          # nobs 0 in post
    pre[0, 1, 0] = 0           # only one band 0: still valid
    v = P.compute_valid_mask(pre, post, cfg)
    np.testing.assert_array_equal(v, [[False, False], [True, True]])


# ---------------------------------------------------------------- input stack
def test_stack_drop_nobs_gives_12_channels_pre_then_post(cfg) -> None:
    pre = np.arange(7, dtype=np.uint16)[:, None, None] * np.ones((7, 3, 4), np.uint16)
    post = pre + 100
    img, names = P.build_input_stack(pre, post, cfg)
    assert img.shape == (3, 4, 12) and img.dtype == np.uint16
    assert names[:6] == ["pre_B2", "pre_B3", "pre_B4", "pre_B8", "pre_B11", "pre_B12"]
    assert names[6:] == [n.replace("pre", "post") for n in names[:6]]
    np.testing.assert_array_equal(img[0, 0], [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105])


def test_stack_as_channel_gives_14_and_clips(cfg) -> None:
    c = copy.deepcopy(cfg)
    c["s2"]["nobs_mode"] = "as_channel"
    pre = np.full((7, 2, 2), 12000, np.uint16)
    pre[6] = 80
    img, names = P.build_input_stack(pre, pre.copy(), c)
    assert img.shape == (2, 2, 14) and names[6] == "pre_nobs"
    assert img[..., 0].max() == c["s2"]["reflectance_clip_max"]
    assert img[..., 6].max() == c["s2"]["nobs_clip_max"]


def test_band_stats_and_normalize() -> None:
    rng = np.random.default_rng(0)
    px = rng.integers(0, 3000, size=(10000, 2)).astype(np.uint16)
    px[:10, 0] = 60000  # outliers are clipped by the percentiles
    st = P.compute_band_stats(px, ["a", "b"], (1, 99))
    z = P.normalize(px, st)
    assert z.dtype == np.float32
    assert abs(float(z[:, 1].mean())) < 0.05 and abs(float(z[:, 1].std()) - 1) < 0.05
    assert z[:10, 0].max() <= (st["p_high"][0] - st["mean"][0]) / st["std"][0] + 1e-5


# ---------------------------------------------------------------- spatial split
def test_cell_grid_and_block_ids() -> None:
    g = S.make_grid(3704, 3009, 128, 512)
    assert (g.n_rows, g.n_cols, g.n_block_rows, g.n_block_cols) == (28, 23, 7, 6)
    ids = g.cell_block_ids()
    assert ids[0, 0] == 0 and ids[0, 4] == 1 and ids[4, 0] == 6 and ids[27, 22] == 41


def test_cell_label_stats_counts_per_cell() -> None:
    g = S.make_grid(8, 8, 4, 8)
    df = np.zeros((8, 8), np.uint8)
    df[0, 0] = 1
    df[5, 6] = 1
    df[4:, :4] = 255
    lc = np.zeros((8, 8), np.uint8)
    st = S.cell_label_stats(g, df, lc, np.ones((8, 8), bool), 2)
    np.testing.assert_array_equal(st["pos"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(st["neg"], [[15, 16], [0, 15]])
    assert st["lc"].shape == (2, 2, 2) and st["lc"][..., 0].sum() == 64


def test_buffer_drops_train_next_to_val_and_val_next_to_test() -> None:
    cs = np.array([[0, 0, 0, 1, 1, 2]], dtype=np.int8)
    keep = S.buffer_keep(cs, 1)
    np.testing.assert_array_equal(keep, [[True, True, False, True, False, True]])
    assert S.buffer_keep(cs, 0).all()


def _toy_stats(rng, nr=16, nc=16):
    pos = rng.integers(0, 400, size=(nr, nc))
    return {"pos": pos, "neg": 16384 - pos, "valid": np.full((nr, nc), 16384),
            "lc": rng.integers(100, 3000, size=(nr, nc, 5))}


def test_block_assignment_is_deterministic_and_whole_blocks() -> None:
    g = S.make_grid(16 * 128, 16 * 128, 128, 512)
    stats = _toy_stats(np.random.default_rng(1))
    usable = np.ones((16, 16), bool)
    sc = {"fractions": {"train": 0.7, "val": 0.15, "test": 0.15}, "buffer_patches": 1,
          "min_positive_pixels": {"val": 1000, "test": 1000}, "min_class_pixels": 10, "n_candidates": 200}
    a, info = S.search_assignment(g, stats, usable, sc, seed=42)
    b, _ = S.search_assignment(g, stats, usable, sc, seed=42)
    np.testing.assert_array_equal(a, b)
    assert info["n_feasible"] > 0
    cs = S.cell_splits(a, g.cell_block_ids())
    for blk in range(g.n_blocks):  # every cell of a block has the block's split
        assert len(np.unique(cs[g.cell_block_ids() == blk])) == 1
    assert set(np.unique(a)) == {0, 1, 2}


def test_no_cell_in_two_splits_and_infeasible_raises() -> None:
    g = S.make_grid(16 * 128, 16 * 128, 128, 512)
    stats = _toy_stats(np.random.default_rng(2))
    sc = {"fractions": {"train": 0.7, "val": 0.15, "test": 0.15}, "buffer_patches": 1,
          "min_positive_pixels": {"val": 10 ** 9, "test": 10 ** 9}, "min_class_pixels": 10, "n_candidates": 50}
    with pytest.raises(RuntimeError, match="No feasible"):
        S.search_assignment(g, stats, np.ones((16, 16), bool), sc, seed=0)


# ---------------------------------------------------------------- weights
def test_class_weights_smoothed_and_clipped() -> None:
    w = compute_class_weights(np.array([7_000_000, 400_000, 2_300_000, 100_000, 500_000]), 0.5, (0.2, 10.0))
    assert w.dtype == np.float32 and w.shape == (5,)
    assert w[3] > w[1] > w[0]                      # rarer -> larger
    assert w.max() <= 10.0 and w.min() >= 0.2
    with pytest.raises(ValueError):
        compute_class_weights(np.array([10, 0]), 0.5, (0.2, 10))


def test_pos_weight_capped() -> None:
    assert compute_pos_weight(1, 10_000, 0.5, 10.0) == 10.0
    assert abs(compute_pos_weight(100, 6_400, 0.5, 10.0) - 8.0) < 1e-9


# ---------------------------------------------------------------- tf.data
def test_augment_applies_same_transform_to_image_and_labels() -> None:
    import tensorflow as tf

    rng = np.random.default_rng(0)
    lc = rng.integers(0, 5, size=(8, 8, 1)).astype(np.int32)
    df = rng.integers(0, 2, size=(8, 8, 1)).astype(np.float32)
    x = np.concatenate([lc.astype(np.float32), df * 10], axis=-1)   # channels encode the labels
    tf.random.set_seed(3)
    for _ in range(10):
        xa, la, da = augment(tf.constant(x), tf.constant(lc), tf.constant(df))
        np.testing.assert_array_equal(xa.numpy()[..., 0], la.numpy()[..., 0])
        np.testing.assert_array_equal(xa.numpy()[..., 1], da.numpy()[..., 0] * 10)


def test_make_dataset_shapes_and_normalisation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(5):
        p = tmp_path / f"p{i}.npz"
        np.savez(p, image=rng.integers(100, 3000, (16, 16, 12)).astype(np.uint16),
                 landcover=rng.integers(0, 5, (16, 16)).astype(np.uint8), deforest=np.full((16, 16), 255, np.uint8))
        paths.append(str(p))
    names = [f"{d}_{b}" for d in ("pre", "post") for b in ("B2", "B3", "B4", "B8", "B11", "B12")]
    stats = {"channels": names, "p_low": [0] * 12, "p_high": [10000] * 12,
             "mean": [1000] * 12, "std": [500] * 12}
    ds_idx = make_dataset(paths, stats, 2, shuffle=False, augment_data=False, seed=0, patch_size=16, indices=["ndvi", "nbr"])
    assert next(iter(ds_idx))[0].shape == (2, 16, 16, 18)
    ds = make_dataset(paths, stats, 2, shuffle=False, augment_data=True, seed=0, patch_size=16)
    batches = list(ds)
    assert len(batches) == 3
    x, y = batches[0]
    assert x.shape == (2, 16, 16, 12) and x.dtype.name == "float32"
    assert y["landcover"].shape == (2, 16, 16, 1) and y["landcover"].dtype.name == "int32"
    assert y["deforestation"].shape == (2, 16, 16, 1) and float(y["deforestation"].numpy().min()) == 255.0
    assert -2.5 < float(x.numpy().mean()) < 4.5


# ---------------------------------------------------------------- lineage
def test_stale_downstream_run_is_rejected(tmp_path: Path) -> None:
    from src.runs import finish_run, new_run_dir, require_parent, resolve_latest

    prep1 = new_run_dir(tmp_path / "processed", "prep")
    finish_run(prep1, tmp_path, {})
    split = new_run_dir(tmp_path / "splits", "split")
    finish_run(split, tmp_path, {"prep_run_id": prep1.name})
    prep2 = new_run_dir(tmp_path / "processed", "prep")   # data re-prepared later
    finish_run(prep2, tmp_path, {})
    latest_prep, _ = resolve_latest(tmp_path / "processed", tmp_path, "prep")
    assert latest_prep == prep2
    _, split_meta = resolve_latest(tmp_path / "splits", tmp_path, "split")
    with pytest.raises(RuntimeError, match="Stale split"):
        require_parent(split_meta, "prep_run_id", latest_prep.name, "split", "prep")


def test_incomplete_run_is_rejected(tmp_path: Path) -> None:
    from src.runs import load_run, new_run_dir

    run = new_run_dir(tmp_path, "train")   # no metadata.json written (crashed run)
    with pytest.raises(RuntimeError, match="no metadata.json"):
        load_run(run, "train")


def test_unchanged_inputs_reuse_run_and_changes_do_not(tmp_path: Path) -> None:
    from src.runs import finish_run, fingerprint, new_run_dir, reusable_run

    base = tmp_path / "processed"
    cfg = {"s2": {"nobs_mode": "drop"}}
    fp = fingerprint("sha-abc", cfg)
    assert reusable_run(base, tmp_path, fp) is None            # nothing built yet
    run = new_run_dir(base, "prep")
    finish_run(run, tmp_path, {"fingerprint": fp})
    assert reusable_run(base, tmp_path, fingerprint("sha-abc", {"s2": {"nobs_mode": "drop"}})) == run
    assert reusable_run(base, tmp_path, fingerprint("sha-abc", {"s2": {"nobs_mode": "as_channel"}})) is None  # config changed
    assert reusable_run(base, tmp_path, fingerprint("sha-NEW", cfg)) is None                                   # raw data changed


# ---------------------------------------------------------------- experiments
def test_config_overlay_deep_merges_onto_base(tmp_path: Path) -> None:
    import shutil

    shutil.copy(ROOT / "configs" / "config.yaml", tmp_path / "config.yaml")
    (tmp_path / "exp.yaml").write_text(
        "base: config.yaml\nmodel:\n  upsampling: bilinear\n  input_channels: 18\nfeatures:\n  indices: [ndvi, nbr]\n")
    c = load_config(tmp_path / "exp.yaml")
    assert c["model"]["upsampling"] == "bilinear" and c["model"]["input_channels"] == 18
    assert c["model"]["base_filters"] == 16                 # untouched keys come from the base
    assert c["features"]["indices"] == ["ndvi", "nbr"]
    (tmp_path / "bad.yaml").write_text("base: config.yaml\nfeatures:\n  indices: [ndvi]\n")  # 12 + 3 != 12
    with pytest.raises(ValueError, match="input_channels"):
        load_config(tmp_path / "bad.yaml")


def test_indices_values_and_model_input_shape(cfg) -> None:
    names = P.channel_layout(cfg)[1]
    img = np.zeros((2, 2, 12), np.uint16)
    img[..., names.index("pre_B8")], img[..., names.index("pre_B4")], img[..., names.index("pre_B12")] = 3000, 1000, 2000
    img[..., names.index("post_B8")], img[..., names.index("post_B4")], img[..., names.index("post_B12")] = 1000, 1000, 3000
    ind = P.compute_indices(img, names, ["ndvi", "nbr"])
    assert ind.shape == (2, 2, 6) and P.index_channel_names(["ndvi", "nbr"])[2] == "dndvi"
    np.testing.assert_allclose(ind[0, 0], [0.5, 0.0, -0.5, 0.2, -0.5, -0.7], atol=1e-6)
    stats = {"channels": names, "p_low": [0] * 12, "p_high": [10000] * 12, "mean": [0] * 12, "std": [1] * 12}
    assert P.model_input(img, stats, ["ndvi", "nbr"]).shape == (2, 2, 18)
    assert P.model_input(img, stats, []).shape == (2, 2, 12)
    zero = P.compute_indices(np.zeros((1, 1, 12), np.uint16), names, ["ndvi"])
    assert np.isfinite(zero).all() and (zero == 0).all()     # 0/0 -> 0, no NaN


def test_overlapping_train_windows_stay_inside_train_cells() -> None:
    cs = np.array([[0, 0, 1],
                   [0, 0, 0],
                   [-1, 0, 0]], dtype=np.int8)
    wins = S.overlapping_train_windows(cs, 4, 2)
    assert (2, 2) in wins and (0, 2) in wins                 # spans only train cells
    assert (0, 6) not in wins                                 # would touch the val cell
    assert (6, 2) not in wins                                 # would touch the dropped cell
    assert all((r % 4, c % 4) != (0, 0) for r, c in wins)    # regular grid not duplicated
    for r, c in wins:
        assert (cs[r // 4:(r + 3) // 4 + 1, c // 4:(c + 3) // 4 + 1] == 0).all()
    with pytest.raises(ValueError):
        S.overlapping_train_windows(cs, 4, 3)


def test_fingerprint_lookup_finds_matching_run_not_only_latest(tmp_path: Path) -> None:
    from src.runs import finish_run, new_run_dir, reusable_run

    base = tmp_path / "patches"
    a = new_run_dir(base, "patches")
    finish_run(a, tmp_path, {"fingerprint": "fp-baseline"})
    b = new_run_dir(base, "patches")
    finish_run(b, tmp_path, {"fingerprint": "fp-experiment"})   # b is now "latest"
    assert reusable_run(base, tmp_path, "fp-baseline") == a     # baseline data still found
    assert reusable_run(base, tmp_path, "fp-experiment") == b
    assert reusable_run(base, tmp_path, "fp-unknown") is None
