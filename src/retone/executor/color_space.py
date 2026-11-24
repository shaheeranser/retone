"""Colour-space maths on float32 RGB in [0, 1]: sRGB transfer function and CIE LAB."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Float32Image = NDArray[np.float32]

_LINEAR_CUTOFF = 0.04045
_ENCODED_CUTOFF = 0.0031308


def srgb_to_linear(rgb: Float32Image) -> Float32Image:
    low = rgb / np.float32(12.92)
    high = ((rgb + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4)
    return np.where(rgb <= _LINEAR_CUTOFF, low, high).astype(np.float32)


def linear_to_srgb(linear: Float32Image) -> Float32Image:
    clipped = np.maximum(linear, np.float32(0.0))
    low = clipped * np.float32(12.92)
    high = np.float32(1.055) * clipped ** np.float32(1 / 2.4) - np.float32(0.055)
    return np.where(clipped <= _ENCODED_CUTOFF, low, high).astype(np.float32)


def rgb_to_lab(rgb: Float32Image) -> Float32Image:
    """sRGB-encoded float RGB to LAB with L in [0, 100]."""
    return np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), dtype=np.float32)


def lab_to_rgb(lab: Float32Image) -> Float32Image:
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), dtype=np.float32)
