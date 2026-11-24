r"""Evaluate a trained model on the held-out spatial TEST region, separately per task.

1. Predict the VALIDATION patches and tune the deforestation threshold there (max F1).
2. Apply that fixed threshold to the TEST patches.
3. Land cover: OA, mIoU, per-class IoU / Dice / precision / recall / F1, confusion matrix.
   Deforestation: precision, recall, F1, IoU, Dice, PR-AUC, confusion matrix, threshold analysis.
4. Trivial reference baselines (majority class, prevalence) so the numbers can be judged.

Scores measure agreement with the reference labels (ESA WorldCover 2021, Hansen GFC),
not independently verified real-world accuracy.

Writes outputs/reports/evaluation/eval_<timestamp>/ (metrics.json, metrics_summary.csv,
landcover_per_class.csv, threshold_*.csv, threshold.json) and outputs/plots/eval_<timestamp>/.

Usage:  python scripts/evaluate.py --config configs/config.yaml [--run outputs/checkpoints/train_<ts>]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "evaluate"


def add_args(p) -> None:
    p.add_argument("--run", type=Path, help="Train run dir (default: latest complete run)")


def resolve_train_run(cfg, args):
    from src.runs import load_run, resolve_latest

    if args.run:
        return load_run(args.run if args.run.is_absolute() else ROOT / args.run, "train")
    return resolve_latest(ROOT / cfg["training"]["checkpoint_dir"], ROOT, "train",
                          "latest_smoke.json" if args.smoke_test else "latest.json")


def predict_patches(model, paths, stats, batch, patch):
    from src.dataset import make_dataset

    ds = make_dataset(paths, stats, batch, shuffle=False, augment_data=False, seed=0, patch_size=patch)
    lc_t, lc_p, df_t, df_p = [], [], [], []
    for x, y in ds:
        out = model.predict_on_batch(x)
        lc_t.append(y["landcover"].numpy()[..., 0].astype(np.uint8))
        lc_p.append(np.argmax(out["landcover"], axis=-1).astype(np.uint8))
        df_t.append(y["deforestation"].numpy()[..., 0].astype(np.uint8))
        df_p.append(np.asarray(out["deforestation"])[..., 0].astype(np.float32))
    return np.concatenate(lc_t), np.concatenate(lc_p), np.concatenate(df_t), np.concatenate(df_p)


def run(cfg, args) -> None:
    import keras
    import pandas as pd

    import src.losses  # noqa: F401  (registers custom objects for load_model)
    import src.metrics as M
    from src.dataset import limit_for_smoke_test, load_manifest, load_patch, split_paths
    from src.runs import data_lineage, finish_run, new_run_dir, read_json, write_json
    from src.visualization import (plot_confusion_matrix, plot_patch_examples, plot_pr_curves, plot_threshold_curve,
                                   stretch_rgb)

    smoke = args.smoke_test
    train_run, tmeta = resolve_train_run(cfg, args)
    patch_run, patch_meta = data_lineage(cfg, ROOT, "patches")["patches"]
    if tmeta["patch_run_id"] != patch_run.name:
        raise RuntimeError(f"{train_run.name} was trained on {tmeta['patch_run_id']}, but the current patch run is "
                           f"{patch_run.name}. Retrain, or the split/test region may not match.")
    log(STAGE, f"model {train_run.relative_to(ROOT)} (best epoch {tmeta.get('best_epoch')}), patches {patch_run.name}")
    model = keras.models.load_model(train_run / "best.keras", compile=False)
    stats = read_json(patch_run / "band_stats.json")
    manifest = load_manifest(patch_run)
    classes = cfg["worldcover"]["class_names"]
    k = len(classes)
    p = int(patch_meta["patch_size"])
    batch = int(cfg["prediction"]["batch_size"])
    seed = int(cfg["project"]["seed"])

    paths = {s: split_paths(manifest, s, patch_run) for s in ("val", "test")}
    if smoke:
        n = int(cfg["training"]["smoke_test"]["max_patches_per_split"])
        paths = {s: limit_for_smoke_test(v, n, seed) for s, v in paths.items()}
    preds = {}
    for s in ("val", "test"):
        log(STAGE, f"predicting {len(paths[s])} {s} patches")
        preds[s] = predict_patches(model, paths[s], stats, batch, p)

    # --- threshold: VALIDATION ONLY ---------------------------------------
    tcfg = cfg["evaluation"]["deforestation_threshold"]
    assert tcfg["tune_on"] == "val"
    _, _, df_t_val, df_p_val = preds["val"]
    sr = tcfg["search"]
    thr, val_rows = M.tune_threshold(df_t_val, df_p_val, sr["start"], sr["stop"], sr["step"], tcfg["metric"])
    log(STAGE, f"threshold tuned on VALIDATION ({tcfg['metric']}): {thr:.2f}")

    results = {"threshold": {"value": thr, "tuned_on": "val", "metric": tcfg["metric"]}, "splits": {}}
    for s in ("val", "test"):
        lc_t, lc_p, df_t, df_p = preds[s]
        cm = M.confusion_matrix(lc_t, lc_p, k)
        lc_rep = M.landcover_report(cm, classes)
        df_rep = M.binary_report(df_t, df_p, thr)
        df_rep["pr_auc"] = M.pr_auc(df_t, df_p)
        df_rep["at_0.5"] = {key: v for key, v in M.binary_report(df_t, df_p, 0.5).items() if key in ("precision", "recall", "f1", "iou")}
        valid_pred = lc_p[lc_t != 255]
        pred_share = np.bincount(valid_pred, minlength=k) / max(valid_pred.size, 1)
        lc_rep["predicted_share"] = dict(zip(classes, pred_share.round(4).tolist()))
        results["splits"][s] = {"landcover": lc_rep, "deforestation": df_rep, "n_patches": len(paths[s])}

    # --- trivial baselines on TEST (context for the scores) ----------------
    tr = manifest[manifest["split"] == "train"]
    majority = int(np.argmax([tr[f"lc_{n}"].sum() for n in classes]))
    lc_t, _, df_t, _ = preds["test"]
    cm_base = M.confusion_matrix(lc_t, np.full_like(lc_t, majority), k)
    base = M.landcover_report(cm_base, classes)
    prevalence = float((df_t == 1).sum() / max((df_t != 255).sum(), 1))
    results["baselines_test"] = {
        "landcover_majority_class": {"class": classes[majority], "overall_accuracy": base["overall_accuracy"], "mean_iou": base["mean_iou"]},
        "deforestation_random": {"pr_auc_equals_prevalence": prevalence, "f1_all_negative": 0.0},
    }

    # --- sanity warnings -------------------------------------------------
    warnings = []
    test = results["splits"]["test"]
    if max(test["landcover"]["predicted_share"].values()) > 0.99:
        warnings.append("land-cover predictions are >99% one class (degenerate model)")
    if test["deforestation"]["tp"] + test["deforestation"]["fp"] == 0:
        warnings.append("no forest-loss pixels predicted on test at the tuned threshold")
    if test["landcover"]["overall_accuracy"] <= base["overall_accuracy"]:
        warnings.append("land-cover OA does not beat the majority-class baseline")
    results["warnings"] = warnings
    for w in warnings:
        log(STAGE, f"WARNING: {w}")

    # --- write reports ---------------------------------------------------
    rep_base = ROOT / cfg["evaluation"]["report_dir"] / "evaluation"
    eval_dir = new_run_dir(rep_base, "smoke_eval" if smoke else "eval")
    plot_dir = ROOT / cfg["evaluation"]["plots_dir"] / eval_dir.name
    write_json(eval_dir / "threshold.json", results["threshold"])
    pd.DataFrame(val_rows).to_csv(eval_dir / "threshold_val.csv", index=False)
    test_rows = M.threshold_sweep(df_t, preds["test"][3], np.array([r["threshold"] for r in val_rows]))
    pd.DataFrame(test_rows).to_csv(eval_dir / "threshold_test_analysis_only.csv", index=False)
    rows = []
    for s, r in results["splits"].items():
        for m_ in ("overall_accuracy", "mean_iou", "mean_dice", "macro_f1"):
            rows.append({"task": "landcover", "split": s, "metric": m_, "value": r["landcover"][m_]})
        for m_ in ("precision", "recall", "f1", "iou", "dice", "pr_auc"):
            rows.append({"task": "deforestation", "split": s, "metric": m_, "value": r["deforestation"][m_]})
        rows.append({"task": "deforestation", "split": s, "metric": "threshold", "value": thr})
    pd.DataFrame(rows).to_csv(eval_dir / "metrics_summary.csv", index=False)
    pc = []
    for s, r in results["splits"].items():
        for n, v in r["landcover"]["per_class"].items():
            pc.append({"split": s, "class": n, **v})
    pd.DataFrame(pc).to_csv(eval_dir / "landcover_per_class.csv", index=False)

    plot_confusion_matrix(np.array(test["landcover"]["confusion_matrix"]), classes, plot_dir / "confusion_landcover_test.png",
                          "Land cover, test region")
    plot_confusion_matrix(np.array(test["deforestation"]["confusion_matrix"]), ["no loss", "loss"],
                          plot_dir / "confusion_deforestation_test.png", f"Forest loss, test (threshold {thr:.2f})")
    plot_threshold_curve(val_rows, thr, plot_dir / "threshold_val.png", "Threshold tuning on VALIDATION")
    curves = {}
    for s in ("val", "test"):
        _, _, t_, p_ = preds[s]
        prec, rec = M.pr_curve(t_, p_)
        curves[s] = (prec, rec, results["splits"][s]["deforestation"]["pr_auc"],
                     float((t_ == 1).sum() / max((t_ != 255).sum(), 1)))
    plot_pr_curves(curves, plot_dir / "pr_curve.png")

    # examples: test patches with the most loss pixels + a few random ones
    test_df = manifest[manifest["split"] == "test"].copy()
    test_df["full_path"] = [str(patch_run / x) for x in test_df["path"]]
    test_df = test_df[test_df["full_path"].isin(paths["test"])]
    n_ex = int(cfg["evaluation"]["n_example_patches"])
    top = test_df.nlargest(n_ex // 2, "df_positive")
    rest = test_df.drop(top.index)
    rnd = rest.sample(min(n_ex - len(top), len(rest)), random_state=seed) if len(rest) else rest
    names = stats["channels"]
    pre_idx = [names.index(f"pre_{b}") for b in ("B4", "B3", "B2")]
    post_idx = [names.index(f"post_{b}") for b in ("B4", "B3", "B2")]
    order = {pp: i for i, pp in enumerate(paths["test"])}
    samples = []
    for row in list(top.itertuples()) + list(rnd.itertuples()):
        i = order[row.full_path]
        img, _, _ = load_patch(row.full_path)
        lc_t, lc_p, df_t_, df_p_ = (a[i] for a in preds["test"])
        samples.append({"pre_rgb": stretch_rgb(img, pre_idx), "post_rgb": stretch_rgb(img, post_idx), "lc_true": lc_t,
                        "lc_pred": lc_p, "df_true": df_t_, "df_prob": df_p_, "title": f"{row.patch_id}\nloss px {row.df_positive}"})
    if samples:
        plot_patch_examples(samples, classes, thr, plot_dir / "examples_test.png")
    if (train_run / "training_curves.png").is_file():
        shutil.copy(train_run / "training_curves.png", plot_dir / "training_curves.png")

    results["note"] = ("Scores measure agreement with reference labels (ESA WorldCover 2021, Hansen GFC v1.12), "
                       "not independently verified ground truth.")
    write_json(eval_dir / "metrics.json", results)
    finish_run(eval_dir, ROOT, {"stage": STAGE, "train_run_id": train_run.name, "train_run_dir": str(train_run.relative_to(ROOT)),
                                "patch_run_id": patch_run.name, "smoke_test": smoke, "threshold": thr,
                                "plots_dir": str(plot_dir.relative_to(ROOT)), "warnings": warnings},
               latest_name="latest_smoke.json" if smoke else "latest.json")

    lc_r, df_r = test["landcover"], test["deforestation"]
    log(STAGE, f"TEST land cover: OA {lc_r['overall_accuracy']:.3f}, mIoU {lc_r['mean_iou']:.3f}, mean Dice {lc_r['mean_dice']:.3f} "
               f"(majority baseline OA {base['overall_accuracy']:.3f}, mIoU {base['mean_iou']:.3f})")
    for n, v in lc_r["per_class"].items():
        log(STAGE, f"    {n:11s} IoU {v['iou']:.3f}  P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  ref px {v['reference_pixels']:,}")
    log(STAGE, f"TEST forest loss @ {thr:.2f}: P {df_r['precision']:.3f}, R {df_r['recall']:.3f}, F1 {df_r['f1']:.3f}, "
               f"IoU {df_r['iou']:.3f}, PR-AUC {df_r['pr_auc']:.3f} (random PR-AUC {prevalence:.3f})")
    log(STAGE, f"reports: {eval_dir.relative_to(ROOT)}; plots: {plot_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args, needs_tf=True))
