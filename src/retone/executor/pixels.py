"""Conversion between integer pixel arrays (8 or 16 bit RGB) and float32 in [0, 1]."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor.errors import ImageError

_MAX_BY_DTYPE = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


def to_float(image: NDArray[np.integer[Any]]) -> NDArray[np.float32]:
    """Scale an HxWx3 uint8/uint16 RGB image to float32 in [0, 1], returning a new array."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    peak = _MAX_BY_DTYPE.get(image.dtype)
    if peak is None:
        raise ImageError(f"expected uint8 or uint16 pixels, got {image.dtype}")
    return (image.astype(np.float32) / np.float32(peak)).astype(np.float32, copy=False)


def from_float(
    rgb: NDArray[np.float32], dtype: np.dtype[np.integer[Any]]
) -> NDArray[np.integer[Any]]:
    """Clip float32 RGB to [0, 1] and quantise to the given integer dtype."""
    peak = _MAX_BY_DTYPE[np.dtype(dtype)]
    scaled = np.clip(rgb, 0.0, 1.0) * np.float32(peak)
    quantised: NDArray[np.integer[Any]] = np.rint(scaled).astype(dtype)
    return quantised
