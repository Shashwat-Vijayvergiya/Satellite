r"""Train the multi-task U-Net (land cover + deforestation) on the current patch run.

Writes outputs/checkpoints/train_<timestamp>/ (smoke_<timestamp>/ with --smoke-test):
    best.keras      lowest val_loss          last.keras   end of latest epoch (for --resume)
    history.csv     per-epoch losses and per-task metrics (train + val)
    state.json      epochs completed (resume)     run_config.yaml   exact config used
    training_curves.png, metadata.json (lineage, class weights, pos_weight, timings)

Usage:
    python scripts/train.py --config configs/config.yaml --smoke-test
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/train_<timestamp>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cli import ROOT, log, run_stage  # noqa: E402

STAGE = "train"


def add_args(p) -> None:
    p.add_argument("--resume", type=Path, help="Existing train run dir to continue from last.keras")
    p.add_argument("--epochs", type=int, help="Override training.epochs")
    p.add_argument("--experiment", help="Experiment name: run dir exp_<name>_<ts>, does NOT replace the main model "
                                       "(updates latest_experiments.json instead of latest.json)")


def run(cfg, args) -> None:
    import keras
    import pandas as pd
    import yaml

    from src.dataset import compute_class_weights, compute_pos_weight, limit_for_smoke_test, load_manifest, make_dataset, split_paths
    from src.losses import build_losses
    from src.metrics import MaskedBinaryScore, MaskedMeanIoU
    from src.models.multitask_unet import build_from_config
    from src.runs import data_lineage, finish_run, new_run_dir, read_json, write_json
    from src.visualization import plot_training_history

    t = cfg["training"]
    smoke = args.smoke_test
    seed = int(cfg["project"]["seed"])
    patch_run, patch_meta = data_lineage(cfg, ROOT, "patches")["patches"]
    stats = read_json(patch_run / "band_stats.json")
    indices = list(cfg["features"].get("indices") or [])
    n_in = len(stats["channels"]) + 3 * len(indices)
    if n_in != int(cfg["model"]["input_channels"]):
        raise RuntimeError(f"patch run has {len(stats['channels'])} band channels + {3 * len(indices)} index channels = {n_in}, "
                           f"but model.input_channels={cfg['model']['input_channels']}")
    manifest = load_manifest(patch_run)
    train_paths, val_paths = split_paths(manifest, "train", patch_run), split_paths(manifest, "val", patch_run)
    batch, epochs = int(t["batch_size"]), int(args.epochs or t["epochs"])
    if smoke:
        sm = t["smoke_test"]
        train_paths = limit_for_smoke_test(train_paths, int(sm["max_patches_per_split"]), seed)
        val_paths = limit_for_smoke_test(val_paths, int(sm["max_patches_per_split"]), seed)
        batch, epochs = int(sm["batch_size"]), int(args.epochs or sm["epochs"])

    # Imbalance handling from TRAIN pixel counts (full train split, also in smoke mode).
    # Overlapping extra patches are excluded so no pixel is counted twice.
    tr = manifest[(manifest["split"] == "train") & ~manifest["overlap_extra"]]
    classes = cfg["worldcover"]["class_names"]
    lc_counts = np.array([tr[f"lc_{n}"].sum() for n in classes], dtype=np.int64)
    lcfg, dcfg = cfg["loss"]["landcover"], cfg["loss"]["deforestation"]
    class_weights = (compute_class_weights(lc_counts, float(lcfg["class_weight_power"]), tuple(lcfg["class_weight_clip"]))
                     if lcfg["class_weights"] == "auto" else np.asarray(lcfg["class_weights"], np.float32))
    pos, neg = int(tr["df_positive"].sum()), int(tr["df_negative"].sum())
    pos_weight = (compute_pos_weight(pos, neg, float(dcfg["pos_weight_power"]), float(dcfg["pos_weight_max"]))
                  if dcfg["pos_weight"] == "auto" else float(dcfg["pos_weight"]))
    log(STAGE, f"train pixels per class {dict(zip(classes, lc_counts.tolist()))}")
    log(STAGE, f"class weights {dict(zip(classes, np.round(class_weights, 3).tolist()))}")
    log(STAGE, f"train loss/no-loss pixels {pos:,}/{neg:,} ({pos / (pos + neg):.3%}); pos_weight {pos_weight:.2f}")

    p = int(patch_meta["patch_size"])
    train_ds = make_dataset(train_paths, stats, batch, shuffle=True, augment_data=bool(t["augment"]), seed=seed, patch_size=p,
                            indices=indices)
    val_ds = make_dataset(val_paths, stats, batch, shuffle=False, augment_data=False, seed=seed, patch_size=p, indices=indices)

    initial_epoch = 0
    if args.resume:
        run_dir = args.resume if args.resume.is_absolute() else ROOT / args.resume
        state = read_json(run_dir / "state.json")
        info = read_json(run_dir / "train_info.json")
        if info["patch_run_id"] != patch_run.name:
            raise RuntimeError(f"cannot resume: run used patches {info['patch_run_id']}, current is {patch_run.name}")
        model = keras.models.load_model(run_dir / "last.keras")
        initial_epoch = int(state["epochs_completed"])
        if initial_epoch >= epochs:
            raise RuntimeError(f"run already completed {initial_epoch} epochs (target {epochs}); pass --epochs N with N > {initial_epoch}")
        log(STAGE, f"resuming {run_dir.name} from epoch {initial_epoch}")
    else:
        prefix = "smoke" if smoke else (f"exp_{args.experiment}" if args.experiment else "train")
        run_dir = new_run_dir(ROOT / t["checkpoint_dir"], prefix)
        model = build_from_config(cfg["model"], p)
        losses, task_weights = build_losses(cfg["loss"], class_weights.tolist(), pos_weight)
        k = len(classes)
        model.compile(
            optimizer=keras.optimizers.Adam(float(t["learning_rate"])),
            loss=losses, loss_weights=task_weights,
            metrics={"landcover": [MaskedMeanIoU(k, name="miou")],
                     "deforestation": [MaskedBinaryScore("f1", name="f1"), MaskedBinaryScore("recall", name="recall")]},
        )
        state = {"epochs_completed": 0, "best_val_loss": None}
        write_json(run_dir / "train_info.json", {"patch_run_id": patch_run.name, "split_run_id": patch_meta["split_run_id"],
                                                  "prep_run_id": patch_meta["prep_run_id"], "smoke_test": smoke,
                                                  "class_weights": class_weights.tolist(), "pos_weight": pos_weight,
                                                  "experiment": args.experiment, "indices": indices,
                                                  "upsampling": cfg["model"]["upsampling"],
                                                  "config_chain": cfg.get("config_chain"),
                                                  "train_patches": len(train_paths), "val_patches": len(val_paths),
                                                  "batch_size": batch, "params": int(model.count_params())})
        (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(STAGE, f"run dir {run_dir.relative_to(ROOT)}; model params {model.count_params():,}; "
               f"train {len(train_paths)} / val {len(val_paths)} patches; batch {batch}; epochs {epochs}")

    class StateCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            vl = float(logs.get("val_loss", np.nan))
            best = state.get("best_val_loss")
            state["epochs_completed"] = epoch + 1
            state["best_val_loss"] = vl if best is None or vl < best else best
            write_json(run_dir / "state.json", state)

    es, rl = t["early_stopping"], t["reduce_lr_on_plateau"]
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(run_dir / "best.keras"), monitor="val_loss", mode="min", save_best_only=True,
                                        initial_value_threshold=state.get("best_val_loss")),
        keras.callbacks.ModelCheckpoint(str(run_dir / "last.keras"), save_best_only=False),
        StateCallback(),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=bool(args.resume)),
        keras.callbacks.EarlyStopping(monitor=es["monitor"], patience=int(es["patience"]), mode="min", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=rl["monitor"], factor=float(rl["factor"]), patience=int(rl["patience"]),
                                          min_lr=float(rl["min_lr"]), verbose=1),
        keras.callbacks.TerminateOnNaN(),
    ]
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)
    elapsed = time.time() - t0

    hist = pd.read_csv(run_dir / "history.csv")
    if hist.empty:
        raise RuntimeError("no epoch completed")
    if not np.isfinite(hist["loss"].to_numpy()).all() or not np.isfinite(hist["val_loss"].to_numpy()).all():
        raise RuntimeError("loss became NaN/inf; training aborted (see history.csv)")
    if not (run_dir / "best.keras").is_file():
        raise RuntimeError("best.keras was not written")
    plot_training_history({c: hist[c].tolist() for c in hist.columns}, run_dir / "training_curves.png")
    best = hist.loc[hist["val_loss"].idxmin()]
    final = {k: float(v) for k, v in best.items()}
    finish_run(run_dir, ROOT, {"stage": STAGE, **read_json(run_dir / "train_info.json"), "epochs_run": int(len(hist)),
                               "best_epoch": int(best["epoch"]) + 1, "best_epoch_metrics": final,
                               "train_seconds_this_session": round(elapsed, 1), "config": cfg},
               latest_name="latest_smoke.json" if smoke else ("latest_experiments.json" if args.experiment else "latest.json"))
    log(STAGE, f"best epoch {int(best['epoch']) + 1}: val_loss {best['val_loss']:.4f}, "
               f"val LC mIoU {best.get('val_landcover_miou', float('nan')):.3f}, val loss-F1@0.5 {best.get('val_deforestation_f1', float('nan')):.3f}")
    log(STAGE, f"trained {len(hist)} epochs in {elapsed / 60:.1f} min; checkpoints in {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(run_stage(STAGE, __doc__.splitlines()[0], run, add_args, needs_tf=True))
