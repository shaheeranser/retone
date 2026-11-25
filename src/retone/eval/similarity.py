"""PSNR and SSIM between a restored image and its target, for diagnostics."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.executor.errors import ImageError

_PEAK = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}
_C1 = 0.01**2
_C2 = 0.03**2


def _validate(image_a: NDArray[Any], image_b: NDArray[Any]) -> float:
    if image_a.shape != image_b.shape:
        raise ImageError(f"image shapes differ: {image_a.shape} vs {image_b.shape}")
    if image_a.ndim != 3 or image_a.shape[2] != 3:
        raise ImageError(f"expected an HxWx3 RGB image, got shape {image_a.shape}")
    peak = _PEAK.get(np.dtype(image_a.dtype))
    if peak is None:
        raise ImageError(f"expected uint8 or uint16 pixels, got {image_a.dtype}")
    return peak


def psnr(image_a: NDArray[Any], image_b: NDArray[Any]) -> float:
    """Peak signal-to-noise ratio in dB; identical images report 100 dB."""
    peak = _validate(image_a, image_b)
    difference = image_a.astype(np.float64) - image_b.astype(np.float64)
    mse = float(np.mean(difference**2))
    if mse == 0.0:
        return 100.0
    return 10.0 * math.log10(peak**2 / mse)


def _grey_unit(image: NDArray[Any], peak: float) -> NDArray[np.float64]:
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / peak
    return grey


def ssim(image_a: NDArray[Any], image_b: NDArray[Any]) -> float:
    """Mean structural similarity on lightness, with the standard Gaussian window."""
    peak = _validate(image_a, image_b)
    first = _grey_unit(image_a, peak)
    second = _grey_unit(image_b, peak)
    mu_first = cv2.GaussianBlur(first, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_second = cv2.GaussianBlur(second, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_product = mu_first * mu_second

    var_first = cv2.GaussianBlur(first * first, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    var_first = np.maximum(var_first - mu_first_sq, 0.0)
    var_second = cv2.GaussianBlur(second * second, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    var_second = np.maximum(var_second - mu_second_sq, 0.0)
    covariance = cv2.GaussianBlur(first * second, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT)
    covariance = covariance - mu_product

    numerator = (2.0 * mu_product + _C1) * (2.0 * covariance + _C2)
    denominator = (mu_first_sq + mu_second_sq + _C1) * (var_first + var_second + _C2)
    return float(np.mean(numerator / denominator))
