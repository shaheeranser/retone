"""A classical sharpening amount from edge energy and noise on a full-resolution crop."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

IntImage = NDArray[np.integer[Any]]

CROP_SIZE = 512
_PEAK = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


def _unit_float(image: IntImage) -> NDArray[np.float32]:
    return (image.astype(np.float32) / np.float32(_PEAK[np.dtype(image.dtype)])).astype(np.float32)


def _centre_crop(image: NDArray[Any], size: int) -> NDArray[Any]:
    height, width = image.shape[:2]
    half = size // 2
    cy, cx = height // 2, width // 2
    top, bottom = max(0, cy - half), min(height, cy + half)
    left, right = max(0, cx - half), min(width, cx + half)
    return image[top:bottom, left:right]


def classical_sharpen_amount(image: IntImage, crop_size: int = CROP_SIZE) -> int:
    """Pick 0..100: low edge energy raises the amount, visible noise lowers it.

    Full-resolution pixels carry the detail the amount is meant to protect, so the
    measurement uses a full-res crop rather than the thumbnail the model sees.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    crop = _unit_float(_centre_crop(image, crop_size))
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    edge_energy = float(cv2.Laplacian(grey, cv2.CV_32F, ksize=3).var())
    noise = float(np.median(np.abs(grey - cv2.medianBlur(grey, 3))))

    amount = 100.0 - 22.0 * np.log10(1.0 + edge_energy) - min(40.0, noise * 1000.0)
    return int(np.clip(round(amount), 0, 100))
