"""Source-image quality filters: resolution, exposure, clipping and sharpness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

_PEAK = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}
_SHARPNESS_EDGE = 512


@dataclass(frozen=True)
class QualityThresholds:
    min_side: int = 512
    min_luma: float = 25.0
    max_luma: float = 230.0
    max_clipped_fraction: float = 0.05
    min_sharpness: float = 20.0
    max_aspect: float = 4.0

    def __post_init__(self) -> None:
        if self.min_side < 1 or not 0 <= self.max_clipped_fraction <= 1:
            raise ValueError("invalid quality thresholds")
        if not 0 <= self.min_luma < self.max_luma <= 255:
            raise ValueError("luma thresholds must lie in 0..255")
        if self.max_aspect < 1:
            raise ValueError("max_aspect must be at least 1")


@dataclass(frozen=True)
class QualityMetrics:
    width: int
    height: int
    mean_luma: float
    clipped_fraction: float
    sharpness: float


def measure_quality(image: NDArray[Any]) -> QualityMetrics:
    """Measure the properties that make an image a useful 'correct' target."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    peak = _PEAK.get(np.dtype(image.dtype))
    if peak is None:
        raise ValueError(f"expected uint8 or uint16 pixels, got {image.dtype}")
    rgb = image.astype(np.float32) * (255.0 / peak)
    height, width = image.shape[:2]

    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = float(grey.mean())
    clipped = float(np.mean(np.any((image == 0) | (image == peak), axis=2)))

    scale = min(1.0, _SHARPNESS_EDGE / max(height, width))
    small = cv2.resize(
        grey,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    sharpness = float(cv2.Laplacian(small, cv2.CV_32F, ksize=3).var())
    return QualityMetrics(
        width=width,
        height=height,
        mean_luma=luma,
        clipped_fraction=clipped,
        sharpness=sharpness,
    )


def acceptable(metrics: QualityMetrics, thresholds: QualityThresholds) -> bool:
    """Reject images that are tiny, badly exposed, clipped beyond repair or blurry."""
    long_side = max(metrics.width, metrics.height)
    short_side = min(metrics.width, metrics.height)
    aspect = long_side / max(1, short_side)
    return (
        short_side >= thresholds.min_side
        and thresholds.min_luma <= metrics.mean_luma <= thresholds.max_luma
        and metrics.clipped_fraction <= thresholds.max_clipped_fraction
        and metrics.sharpness >= thresholds.min_sharpness
        and aspect <= thresholds.max_aspect
    )
