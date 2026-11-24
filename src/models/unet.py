"""U-Net building blocks and a single-task baseline (TensorFlow/Keras).

Why U-Net: segmentation needs a label for every pixel. The encoder
downsamples to gather context (is this patch forest or cropland?), and the
decoder upsamples back to full resolution. Skip connections carry the
fine spatial detail lost in pooling (field edges, river banks), so
boundaries stay sharp. It trains well from a few hundred patches, which
suits a single 90 x 111 km scene.

CPU sizing: base_filters=16, depth=4 -> 16/32/64/128 filters, bottleneck 256.
"""

from __future__ import annotations

import keras
from keras import layers


def conv_block(x, filters: int, batch_norm: bool = True, dropout: float = 0.0, name: str = "blk"):
    """Two 3x3 Conv2D -> (BatchNorm) -> ReLU, optional dropout at the end."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not batch_norm, kernel_initializer="he_normal",
                          name=f"{name}_conv{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def encoder_block(x, filters: int, batch_norm: bool = True, name: str = "enc"):
    """conv_block then 2x2 max-pool. Returns (skip, pooled)."""
    skip = conv_block(x, filters, batch_norm, name=name)
    return skip, layers.MaxPooling2D(2, name=f"{name}_pool")(skip)


def decoder_block(x, skip, filters: int, batch_norm: bool = True, name: str = "dec", upsampling: str = "transpose"):
    """Upsample x2, concatenate the skip, conv_block.

    upsampling: "transpose" = learned 2x2 transposed convolution;
                "bilinear"  = bilinear resize + 3x3 conv (avoids checkerboard artefacts).
    """
    if upsampling == "bilinear":
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_resize")(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_up")(x)
    elif upsampling == "transpose":
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
    else:
        raise ValueError(f"upsampling must be 'transpose' or 'bilinear', got {upsampling!r}")
    x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    return conv_block(x, filters, batch_norm, name=name)


def encoder(inputs, base_filters: int, depth: int, dropout: float, batch_norm: bool):
    """Return (bottleneck, skips) where skips[0] is full resolution."""
    skips, x = [], inputs
    for d in range(depth):
        s, x = encoder_block(x, base_filters * 2 ** d, batch_norm, name=f"enc{d}")
        skips.append(s)
    x = conv_block(x, base_filters * 2 ** depth, batch_norm, dropout, name="bottleneck")
    return x, skips


def decoder(x, skips, base_filters: int, batch_norm: bool, prefix: str, upsampling: str = "transpose"):
    for d in reversed(range(len(skips))):
        x = decoder_block(x, skips[d], base_filters * 2 ** d, batch_norm, name=f"{prefix}{d}", upsampling=upsampling)
    return x


def build_unet(input_shape: tuple[int, int, int], num_classes: int, activation: str = "softmax", base_filters: int = 16,
               depth: int = 4, dropout: float = 0.2, batch_norm: bool = True, output_name: str = "output",
               upsampling: str = "transpose") -> keras.Model:
    """Single-task U-Net baseline."""
    if input_shape[0] % 2 ** depth or input_shape[1] % 2 ** depth:
        raise ValueError(f"input size {input_shape[:2]} must be divisible by 2**depth={2 ** depth}")
    inputs = keras.Input(input_shape, name="image")
    x, skips = encoder(inputs, base_filters, depth, dropout, batch_norm)
    x = decoder(x, skips, base_filters, batch_norm, prefix="dec", upsampling=upsampling)
    out = layers.Conv2D(num_classes, 1, activation=activation, dtype="float32", name=output_name)(x)
    return keras.Model(inputs, out, name="unet")
