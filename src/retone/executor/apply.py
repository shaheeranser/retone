"""apply(): the one function that edits pixels, running every operation in a fixed order."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor.color_space import lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear
from retone.executor.exposure_wb import channel_gains
from retone.executor.params import Params
from retone.executor.pixels import from_float, to_float
from retone.executor.saturation import adjust_chroma
from retone.executor.sharpen import unsharp_lightness
from retone.executor.tone import tone_curve

IntImage = NDArray[np.integer[Any]]
Float32Image = NDArray[np.float32]


def _exposure_and_white_balance(rgb: Float32Image, params: Params) -> Float32Image:
    # Gains must act on linear light; on encoded values they would shift hue as well as brightness.
    linear = srgb_to_linear(rgb) * channel_gains(params.exposure, params.temperature, params.tint)
    return np.clip(linear_to_srgb(linear), 0.0, 1.0)


def _lab_stage(rgb: Float32Image, params: Params) -> Float32Image:
    lab = rgb_to_lab(rgb)
    if params.saturation != 0 or params.vibrance != 0:
        lab = adjust_chroma(lab, params.saturation, params.vibrance)
    if params.sharpen_amount != 0:
        lab = unsharp_lightness(lab, params.sharpen_amount)
    return lab_to_rgb(lab)


def apply(image: IntImage, params: Params) -> IntImage:
    """Return a new image with the parameters applied; the input is never modified.

    Order: exposure and white balance in linear light, tone curve, then chroma and
    sharpening in LAB. Each stage is skipped at identity, so identity parameters
    return the input exactly. 8-bit input gives 8-bit output, 16-bit gives 16-bit.
    """
    rgb = to_float(image)
    if params.exposure != 0 or params.temperature != 0 or params.tint != 0:
        rgb = _exposure_and_white_balance(rgb, params)
    rgb = tone_curve(rgb, params.contrast, params.highlights, params.shadows)
    if params.saturation != 0 or params.vibrance != 0 or params.sharpen_amount != 0:
        rgb = _lab_stage(rgb, params)
    return from_float(rgb, image.dtype)
