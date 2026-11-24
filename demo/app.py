"""Streamlit demo (optional). The pipeline runs without it.

Run from the project root, after the pipeline has completed:
    streamlit run demo/app.py

Tabs:
    Scene maps       full-scene predictions written by predict.py (preview, areas, GeoTIFF downloads)
    Test metrics     held-out test-region scores and figures written by evaluate.py
    Live prediction  pick any 512 x 512 px window, run the model now, compare with the reference labels
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import load_config  # noqa: E402
from src.runs import read_json, resolve_latest  # noqa: E402

st.set_page_config(page_title="Assam land cover & forest loss", layout="wide")
CFG = load_config(ROOT / "configs" / "config.yaml")
CLASSES = CFG["worldcover"]["class_names"]
NOTE = ("Scores and maps show agreement with reference labels (ESA WorldCover 2021, Hansen GFC v1.12), "
        "not independently verified ground truth. Forest loss = gross stand-replacement loss 2021-2023 at 30 m.")


def latest(base: Path, what: str):
    try:
        return resolve_latest(base, ROOT, what)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()


train_run, tmeta = latest(ROOT / CFG["training"]["checkpoint_dir"], "train")
eval_run, emeta = latest(ROOT / CFG["evaluation"]["report_dir"] / "evaluation", "evaluate")
pred_run, pmeta = latest(ROOT / CFG["prediction"]["output_dir"], "predict")

st.title("Land cover and forest-loss mapping, Kaziranga / Karbi Anglong (Assam)")
st.caption(f"Model `{train_run.name}` (best epoch {tmeta.get('best_epoch')}), evaluation `{eval_run.name}`, "
           f"prediction `{pred_run.name}`. {NOTE}")
if emeta["train_run_id"] != train_run.name or pmeta["train_run_id"] != train_run.name:
    st.warning("The latest evaluation/prediction runs were made with a different model than the latest training run.")

tab_maps, tab_metrics, tab_live = st.tabs(["Scene maps", "Test metrics", "Live prediction"])

with tab_maps:
    st.image(str(pred_run / "preview.png"), width="stretch")
    summ = read_json(pred_run / "area_summary.json")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted land cover")
        st.dataframe(pd.DataFrame(summ["landcover"]).T[["area_km2", "area_ha", "share_pct"]].round(2), width="stretch")
    with c2:
        st.subheader("Predicted forest loss 2021-2023")
        d = summ["deforestation"]
        st.metric("Loss area", f"{d['area_km2']:.1f} km²", f"{d['loss_pixels']:,} pixels at 30 m")
        st.write(f"Threshold {summ['threshold']:.2f}, from {summ['threshold_source']}.")
    st.subheader("Download GeoTIFFs (EPSG:32646, 30 m)")
    cols = st.columns(3)
    for col, name in zip(cols, ("landcover.tif", "deforestation_prob.tif", "deforestation_mask.tif")):
        col.download_button(name, (pred_run / name).read_bytes(), file_name=name)

with tab_metrics:
    m = read_json(eval_run / "metrics.json")
    test = m["splits"]["test"]
    lc, df = test["landcover"], test["deforestation"]
    base = m["baselines_test"]
    a, b, c, d_ = st.columns(4)
    a.metric("Land cover OA", f"{lc['overall_accuracy']:.3f}", f"baseline {base['landcover_majority_class']['overall_accuracy']:.3f}")
    b.metric("Land cover mIoU", f"{lc['mean_iou']:.3f}", f"baseline {base['landcover_majority_class']['mean_iou']:.3f}")
    c.metric("Forest-loss F1", f"{df['f1']:.3f}", f"threshold {m['threshold']['value']:.2f} (val)")
    d_.metric("Forest-loss PR-AUC", f"{df['pr_auc']:.3f}", f"random {base['deforestation_random']['pr_auc_equals_prevalence']:.3f}")
    st.dataframe(pd.DataFrame(lc["per_class"]).T[["iou", "precision", "recall", "f1", "reference_pixels"]].round(3),
                 width="stretch")
    plots = ROOT / emeta["plots_dir"]
    for row in (("confusion_landcover_test.png", "confusion_deforestation_test.png"), ("pr_curve.png", "threshold_val.png")):
        cols = st.columns(2)
        for col, name in zip(cols, row):
            if (plots / name).is_file():
                col.image(str(plots / name), width="stretch")
    if (plots / "examples_test.png").is_file():
        st.image(str(plots / "examples_test.png"), caption="Test patches: reference vs prediction", width="stretch")


@st.cache_resource
def load_model():
    import keras

    import src.losses  # noqa: F401
    import src.metrics  # noqa: F401

    return keras.models.load_model(train_run / "best.keras", compile=False)


with tab_live:
    import rasterio
    from rasterio.windows import Window

    from src.inference import predict_scene
    from src.preprocessing import build_input_stack, compute_valid_mask, model_input
    from src.visualization import landcover_rgb, stretch_rgb

    g = CFG["grid"]
    size = 512
    st.write("Choose a 512 x 512 px window (15.4 x 15.4 km). The model runs on the raw composites now. "
             "The default window lies entirely in the held-out TEST region (8,766 reference loss pixels).")
    c1, c2 = st.columns(2)
    row = c1.slider("Window top row (px)", 0, g["height"] - size, 2048, step=128)
    col = c2.slider("Window left column (px)", 0, g["width"] - size, 1536, step=128)
    split_run, _ = latest(ROOT / CFG["paths"]["splits_dir"], "split")
    cell_split = np.load(split_run / "cell_split.npy")
    p = CFG["patching"]["patch_size"]
    cells = cell_split[row // p:(row + size) // p, col // p:(col + size) // p]
    names = {0: "train", 1: "val", 2: "test", -1: "buffer/unused"}
    shares = {names[int(k)]: int(v) for k, v in zip(*np.unique(cells, return_counts=True))}
    st.caption(f"Patch cells in this window by split: {shares}. Only test-region windows are a fair check.")
    if st.button("Run model on this window", type="primary"):
        win = Window(col, row, size, size)
        raw = CFG["paths"]["raw"]
        with rasterio.open(ROOT / raw["s2_pre"]) as a, rasterio.open(ROOT / raw["s2_post"]) as b:
            pre, post = a.read(window=win), b.read(window=win)
        stats = read_json(ROOT / CFG["paths"]["patches_dir"] / tmeta["patch_run_id"] / "band_stats.json")
        valid = compute_valid_mask(pre, post, CFG)
        stack, ch = build_input_stack(pre, post, CFG)
        model = load_model()
        with st.spinner("Predicting..."):
            lc_prob, df_prob = predict_scene(lambda bt: model.predict_on_batch(bt), model_input(stack, stats, list(tmeta.get("indices") or [])), 128,
                                             CFG["prediction"]["overlap"], 16, len(CLASSES))
        thr = float(emeta["threshold"])
        lc_pred = np.argmax(lc_prob, -1).astype(np.uint8)
        lc_pred[~valid] = 255
        prep_run, _ = latest(ROOT / CFG["paths"]["processed_dir"], "prepare_dataset")
        lc_ref = np.load(prep_run / "landcover.npy", mmap_mode="r")[row:row + size, col:col + size]
        df_ref = np.load(prep_run / "deforest.npy", mmap_mode="r")[row:row + size, col:col + size]
        post_rgb = stretch_rgb(stack, [ch.index(f"post_{x}") for x in ("B4", "B3", "B2")])
        pre_rgb = stretch_rgb(stack, [ch.index(f"pre_{x}") for x in ("B4", "B3", "B2")])
        overlay = post_rgb.copy()
        overlay[df_prob >= thr] = [1, 0, 1]
        r1 = st.columns(3)
        r1[0].image(pre_rgb, caption="Pre composite 2020-21", clamp=True, width="stretch")
        r1[1].image(post_rgb, caption="Post composite 2023-24", clamp=True, width="stretch")
        r1[2].image(overlay, caption=f"Predicted loss (magenta, p >= {thr:.2f})", clamp=True, width="stretch")
        r2 = st.columns(3)
        r2[0].image(landcover_rgb(np.asarray(lc_ref), len(CLASSES)), caption="Land cover reference (WorldCover 2021; grey = ignored)", width="stretch")
        r2[1].image(landcover_rgb(lc_pred, len(CLASSES)), caption="Land cover predicted", width="stretch")
        ref = np.where(np.asarray(df_ref) == 255, 0.5, np.asarray(df_ref)).astype(float)
        r2[2].image(ref, caption="Loss reference (Hansen 2021-23; grey = ignored)", clamp=True, width="stretch")
        st.caption("Legend: dark green tree cover, yellow grassland, pink cropland, red built-up, blue water.")
