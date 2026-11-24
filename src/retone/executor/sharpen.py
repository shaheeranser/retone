"""Unsharp mask on LAB lightness with a radius defined relative to the image diagonal."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

Float32Image = NDArray[np.float32]

# Sigma as a fraction of the image diagonal, so the same setting looks alike at any resolution.
RADIUS_FRACTION_OF_DIAGONAL = 0.0003
MAX_STRENGTH = 1.5
_LIGHTNESS_MAX = 100.0


def sharpen_sigma(height: int, width: int) -> float:
    return RADIUS_FRACTION_OF_DIAGONAL * math.hypot(height, width)


def unsharp_lightness(lab: Float32Image, amount: int) -> Float32Image:
    """Sharpen only L, so colour is never shifted; chroma channels pass through."""
    lightness = lab[..., 0]
    sigma = sharpen_sigma(lab.shape[0], lab.shape[1])
    blurred = cv2.GaussianBlur(lightness, (0, 0), sigma)
    boosted = lightness + MAX_STRENGTH * amount / 100 * (lightness - blurred)
    out = lab.copy()
    out[..., 0] = np.clip(boosted, 0.0, _LIGHTNESS_MAX)
    return out
