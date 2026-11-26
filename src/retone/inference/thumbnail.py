"""Downscale a photo to the model's fixed thumbnail size, averaging in linear light."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.executor.color_space import linear_to_srgb, srgb_to_linear
from retone.executor.pixels import from_float, to_float

IntImage = NDArray[np.integer[Any]]

THUMBNAIL_LONG_EDGE = 448


def thumbnail(image: IntImage, long_edge: int = THUMBNAIL_LONG_EDGE) -> NDArray[np.uint8]:
    """Return an RGB uint8 thumbnail whose longer side is `long_edge`, never upscaled.

    Area averaging runs on linear-light values, so a thumbnail's tones match what the
    full-resolution image averages to; averaging gamma-encoded values would darken bright
    regions. The model only ever sees this result, so it must be a faithful summary.
    """
    height, width = image.shape[:2]
    scale = min(1.0, long_edge / max(height, width))
    if scale == 1.0:
        return cast(NDArray[np.uint8], from_float(to_float(image), np.dtype(np.uint8)))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    linear = srgb_to_linear(to_float(image))
    resized = np.asarray(cv2.resize(linear, size, interpolation=cv2.INTER_AREA), dtype=np.float32)
    encoded = np.clip(linear_to_srgb(resized), 0.0, 1.0)
    return cast(NDArray[np.uint8], from_float(encoded, np.dtype(np.uint8)))
