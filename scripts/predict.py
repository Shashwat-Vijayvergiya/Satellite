r"""Full-scene inference from the pre and post Sentinel-2 composites.

Writes outputs/predictions/pred_<timestamp>/ (smoke_pred_<timestamp>/ with --smoke-test):
    landcover.tif            uint8, model classes 0..4 (colour table), 255 = no data
    deforestation_prob.tif   float32 forest-loss probability, -1 = no data
    deforestation_mask.tif   uint8 1 = predicted loss, 0 = no loss, 255 = no data
    preview.png              post RGB, land cover, loss probability, loss overlay
    area_summary.json/.csv   predicted class areas (km2, ha) and detected loss pixels
CRS and geotransform are copied from the input rasters. The binary mask uses
the threshold tuned on VALIDATION by evaluate.py for the same model.

Usage:
    python scripts/predict.py --config configs/config.yaml
    python scripts/predict.py --config configs/config.yaml --pre other_pre.tif --post other_post.tif
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "predict"


def add_args(p) -> None:
    p.add_argument("--run", type=Path, help="Train run dir (default: latest complete run)")
    p.add_argument("--pre", type=Path, help="Pre composite GeoTIFF (default: config paths.raw.s2_pre)")
    p.add_argument("--post", type=Path, help="Post composite GeoTIFF (default: config paths.raw.s2_post)")
    p.add_argument("--threshold", type=float, help="Override the validation-tuned threshold")


def run(cfg, args) -> None:
    import keras
    import pandas as pd
    import rasterio
    from rasterio.windows import Window

    import src.losses  # noqa: F401  (registers custom objects)
    import src.metrics  # noqa: F401
    from src.inference import LANDCOVER_COLORS, area_summary, predict_scene, write_geotiff
    from src.preprocessing import build_input_stack, compute_valid_mask, normalize
    from src.runs import finish_run, load_run, new_run_dir, read_json, resolve_latest, write_json
    from src.validation import require_validated_raw_data
    from src.visualization import plot_scene_maps, stretch_rgb

    smoke = args.smoke_test
    ck = ROOT / cfg["training"]["checkpoint_dir"]
    if args.run:
        train_run, tmeta = load_run(args.run if args.run.is_absolute() else ROOT / args.run, "train")
    else:
        train_run, tmeta = resolve_latest(ck, ROOT, "train", "latest_smoke.json" if smoke else "latest.json")
    patch_run = ROOT / cfg["paths"]["patches_dir"] / tmeta["patch_run_id"]
    stats = read_json(patch_run / "band_stats.json")

    if args.threshold is not None:
        thr, thr_src = float(args.threshold), "command line --threshold"
    else:
        eval_base = ROOT / cfg["evaluation"]["report_dir"] / "evaluation"
        eval_run, emeta = resolve_latest(eval_base, ROOT, "evaluate", "latest_smoke.json" if smoke else "latest.json")
        if emeta["train_run_id"] != train_run.name:
            raise RuntimeError(f"latest evaluation {eval_run.name} is for {emeta['train_run_id']}, not {train_run.name}. "
                               "Run evaluate.py for this model first (it tunes the threshold on validation).")
        thr, thr_src = float(emeta["threshold"]), f"{eval_run.name} (tuned on validation)"

    raw = cfg["paths"]["raw"]
    pre_path = args.pre or ROOT / raw["s2_pre"]
    post_path = args.post or ROOT / raw["s2_post"]
    if args.pre is None and args.post is None:
        require_validated_raw_data(cfg, ROOT)
    win = None
    if smoke:
        s = int(cfg["prediction"]["smoke_window_px"])
        win = Window(0, 0, s, s)
    with rasterio.open(pre_path) as a, rasterio.open(post_path) as b:
        for src in (a, b):
            if src.count != len(cfg["s2"]["bands"]):
                raise ValueError(f"{src.name}: {src.count} bands, expected {len(cfg['s2']['bands'])} ({cfg['s2']['bands']})")
        if (a.crs, a.transform, a.shape) != (b.crs, b.transform, b.shape):
            raise ValueError("pre and post composites are not on the same grid")
        crs = a.crs
        transform = a.window_transform(win) if win else a.transform
        pre, post = a.read(window=win), b.read(window=win)
    valid = compute_valid_mask(pre, post, cfg)
    stack, names = build_input_stack(pre, post, cfg)
    if names != stats["channels"]:
        raise RuntimeError(f"input channels {names} differ from the training channels {stats['channels']}")
    del pre, post
    x = normalize(stack, stats)
    log(STAGE, f"input {stack.shape} from {Path(pre_path).name} + {Path(post_path).name}; valid {valid.mean():.2%}; threshold {thr:.2f} ({thr_src})")

    model = keras.models.load_model(train_run / "best.keras", compile=False)
    pc = cfg["prediction"]
    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 10 or done == total:
            last[0] = time.time()
            log(STAGE, f"tiles {done}/{total}")

    k = len(cfg["worldcover"]["class_names"])
    lc_prob, df_prob = predict_scene(lambda bt: model.predict_on_batch(bt), x, int(pc["tile_size"]), int(pc["overlap"]),
                                     int(pc["batch_size"]), k, progress)
    del x
    lc = np.argmax(lc_prob, axis=-1).astype(np.uint8)
    lc[~valid] = 255
    df_mask = (df_prob >= thr).astype(np.uint8)
    df_mask[~valid] = 255
    df_out = np.where(valid, df_prob, -1).astype(np.float32)

    out = new_run_dir(ROOT / pc["output_dir"], "smoke_pred" if smoke else "pred")
    classes = cfg["worldcover"]["class_names"]
    tags = {"model_run": train_run.name, "threshold": f"{thr:.4f}", "threshold_source": thr_src,
            "classes": ",".join(f"{i}={n}" for i, n in enumerate(classes)), "labels_reference": "ESA WorldCover 2021 / Hansen GFC v1.12"}
    write_geotiff(out / "landcover.tif", lc, crs, transform, nodata=255, colormap=LANDCOVER_COLORS, tags=tags, band_description="landcover_class")
    write_geotiff(out / "deforestation_prob.tif", df_out, crs, transform, nodata=-1, tags=tags, band_description="forest_loss_probability")
    write_geotiff(out / "deforestation_mask.tif", df_mask, crs, transform, nodata=255, tags=tags, band_description="forest_loss_mask")
    post_idx = [names.index(f"post_{b}") for b in ("B4", "B3", "B2")]
    plot_scene_maps(stretch_rgb(stack, post_idx), lc, df_prob, df_mask, classes, out / "preview.png",
                    step=1 if smoke else 2)
    px_area = abs(transform.a * transform.e)
    summary = area_summary(lc, df_mask, valid, classes, px_area)
    summary.update({"threshold": thr, "threshold_source": thr_src, "model_run": train_run.name})
    write_json(out / "area_summary.json", summary)
    rows = [{"layer": "landcover", "class": n, **v} for n, v in summary["landcover"].items()]
    rows.append({"layer": "deforestation", "class": "forest_loss", "pixels": summary["deforestation"]["loss_pixels"],
                 **{k_: v for k_, v in summary["deforestation"].items() if k_ != "loss_pixels"}})
    pd.DataFrame(rows).to_csv(out / "area_summary.csv", index=False)
    finish_run(out, ROOT, {"stage": STAGE, "train_run_id": train_run.name, "inputs": [str(pre_path), str(post_path)],
                           "threshold": thr, "threshold_source": thr_src, "smoke_test": smoke, "shape": list(lc.shape),
                           "seconds": round(time.time() - t0, 1)},
               latest_name="latest_smoke.json" if smoke else "latest.json")
    d = summary["deforestation"]
    log(STAGE, "predicted land cover: " + ", ".join(f"{n} {v['area_km2']:.1f} km2" for n, v in summary["landcover"].items()))
    log(STAGE, f"predicted forest loss: {d['loss_pixels']:,} px = {d['area_km2']:.2f} km2 ({d['area_ha']:.0f} ha)")
    log(STAGE, f"wrote {out.relative_to(ROOT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args, needs_tf=True))
