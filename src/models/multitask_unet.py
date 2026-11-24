"""Multi-task U-Net: one shared encoder, two task outputs.

    landcover      5-class softmax, (H, W, 5)
    deforestation  1-channel sigmoid, (H, W, 1)

decoder="separate" (default) gives each task its own decoder, so both reuse
the shared encoder features but reconstruct detail independently.
decoder="shared" is a lighter variant: one decoder with two 1x1 heads.
Output names MUST stay "landcover" and "deforestation"; the dataset
labels, losses and metrics are keyed by them.
"""

from __future__ import annotations

from typing import Any

import keras
from keras import layers

from src.models.unet import decoder, encoder


def build_multitask_unet(input_shape: tuple[int, int, int], num_landcover_classes: int = 5, base_filters: int = 16,
                         depth: int = 4, dropout: float = 0.2, batch_norm: bool = True,
                         decoder_mode: str = "separate") -> keras.Model:
    if input_shape[0] % 2 ** depth or input_shape[1] % 2 ** depth:
        raise ValueError(f"input size {input_shape[:2]} must be divisible by 2**depth={2 ** depth}")
    if decoder_mode not in ("separate", "shared"):
        raise ValueError(f"decoder must be 'separate' or 'shared', got {decoder_mode!r}")
    inputs = keras.Input(input_shape, name="image")
    bottleneck, skips = encoder(inputs, base_filters, depth, dropout, batch_norm)
    if decoder_mode == "separate":
        x_lc = decoder(bottleneck, skips, base_filters, batch_norm, prefix="lc_dec")
        x_df = decoder(bottleneck, skips, base_filters, batch_norm, prefix="df_dec")
    else:
        x_lc = x_df = decoder(bottleneck, skips, base_filters, batch_norm, prefix="dec")
    lc = layers.Conv2D(num_landcover_classes, 1, activation="softmax", dtype="float32", name="landcover")(x_lc)
    df = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="deforestation")(x_df)
    return keras.Model(inputs, {"landcover": lc, "deforestation": df}, name=f"multitask_unet_{decoder_mode}")


def build_from_config(model_cfg: dict[str, Any], patch_size: int) -> keras.Model:
    return build_multitask_unet(
        (patch_size, patch_size, int(model_cfg["input_channels"])),
        num_landcover_classes=int(model_cfg["heads"]["landcover"]["num_classes"]),
        base_filters=int(model_cfg["base_filters"]), depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]), batch_norm=bool(model_cfg["batch_norm"]),
        decoder_mode=model_cfg.get("decoder", "separate"),
    )
