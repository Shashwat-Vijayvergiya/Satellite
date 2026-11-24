# Model-improvement experiments

Each file overlays `configs/config.yaml` (via `base:`); experiments are cumulative.

| Config | Change vs previous | Rationale |
|---|---|---|
| `e1_weights_bilinear.yaml` | class-weight cap 10 -> 3; bilinear upsampling | built-up over-predicted (weight 6.9, precision 0.29); checkerboard artefacts |
| `e2_indices.yaml` | + NDVI and NBR (pre, post, difference): 12 -> 18 channels | dNBR/dNDVI are standard clearing indicators; helps a small dataset |
| `e3_overlap.yaml` | + overlapping train patches (stride 64) | about 4x more (correlated) training samples; val/test unchanged |

**Selection rule (fixed before any run):** the model with the highest
*validation* land-cover mIoU + *validation* forest-loss PR-AUC wins
(`evaluate.py --val-only`). Baseline: 0.5812 + 0.4625 = 1.0437. Only the
winner is evaluated on the test region, once.

Run, e.g.:

    python scripts/create_patches.py --config configs/experiments/e3_overlap.yaml   # only e3 needs new patches
    python scripts/train.py --config configs/experiments/e1_weights_bilinear.yaml --experiment e1
    python scripts/evaluate.py --config configs/config.yaml --run outputs/checkpoints/exp_e1_<ts> --val-only
