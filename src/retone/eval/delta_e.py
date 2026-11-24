"""CIEDE2000 colour difference between sRGB images, computed in bounded row blocks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from retone.executor.errors import ImageError

_LAB_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_LAB_EPSILON = (6.0 / 29.0) ** 3
_PEAK = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


def _as_unit_float(image: NDArray[Any]) -> NDArray[np.float64]:
    peak = _PEAK.get(image.dtype)
    if peak is not None:
        return image.astype(np.float64) / peak
    if image.dtype.kind == "f":
        return np.clip(image.astype(np.float64), 0.0, 1.0)
    raise ImageError(f"expected uint8, uint16 or float pixels, got {image.dtype}")


def _lab_f(t: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(t > _LAB_EPSILON, np.cbrt(t), t / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)


def srgb_to_lab(rgb: NDArray[Any]) -> NDArray[np.float64]:
    """Convert HxWx3 sRGB (integer or float in [0, 1]) to CIELAB with a D65 white point."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")
    linear = _as_unit_float(rgb)
    linear = np.where(linear <= 0.04045, linear / 12.92, ((linear + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _LAB_WHITE
    fx, fy, fz = (_lab_f(xyz[..., i]) for i in range(3))
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def ciede2000_from_lab(lab1: NDArray[np.float64], lab2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise CIEDE2000 difference (Sharma, Wu & Dalal, 2005)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p

    chroma_product = c1p * c2p
    hue_delta = h2p - h1p
    dhp = np.where(
        chroma_product == 0.0,
        0.0,
        np.where(
            np.abs(hue_delta) <= 180.0,
            hue_delta,
            np.where(hue_delta > 180.0, hue_delta - 360.0, hue_delta + 360.0),
        ),
    )
    dhp_rad = np.radians(dhp)
    dhp_rad = np.where(chroma_product == 0.0, 0.0, dhp_rad)
    dhp_term = 2.0 * np.sqrt(chroma_product) * np.sin(dhp_rad / 2.0)

    l_bar = (l1 + l2) / 2.0
    c_barp = (c1p + c2p) / 2.0
    hue_sum = h1p + h2p
    hue_abs = np.abs(h1p - h2p)
    h_barp = np.where(
        chroma_product == 0.0,
        hue_sum,
        np.where(
            hue_abs <= 180.0,
            hue_sum / 2.0,
            np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
        ),
    )

    h_barp_rad = np.radians(h_barp)
    t = (
        1.0
        - 0.17 * np.cos(h_barp_rad - np.radians(30.0))
        + 0.24 * np.cos(2.0 * h_barp_rad)
        + 0.32 * np.cos(3.0 * h_barp_rad + np.radians(6.0))
        - 0.20 * np.cos(4.0 * h_barp_rad - np.radians(63.0))
    )
    dtheta = np.radians(30.0 * np.exp(-(((h_barp - 275.0) / 25.0) ** 2)))
    c_barp7 = c_barp**7
    rc = 2.0 * np.sqrt(c_barp7 / (c_barp7 + 25.0**7))
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_barp
    sh = 1.0 + 0.015 * c_barp * t
    rt = -np.sin(2.0 * dtheta) * rc

    dl_term = dlp / sl
    dc_term = dcp / sc
    dh_term = dhp_term / sh
    distance = dl_term**2 + dc_term**2 + dh_term**2 + rt * dc_term * dh_term
    return cast(NDArray[np.float64], np.sqrt(distance))


def mean_delta_e(image_a: NDArray[Any], image_b: NDArray[Any], row_block: int = 256) -> float:
    """Mean CIEDE2000 over an image pair, processing rows in blocks to bound memory."""
    if image_a.shape != image_b.shape:
        raise ImageError(f"image shapes differ: {image_a.shape} vs {image_b.shape}")
    if image_a.ndim != 3 or image_a.shape[2] != 3:
        raise ImageError(f"expected an HxWx3 RGB image, got shape {image_a.shape}")
    if row_block < 1:
        raise ImageError(f"row_block must be positive, got {row_block}")

    total = 0.0
    for start in range(0, image_a.shape[0], row_block):
        rows = slice(start, start + row_block)
        lab_a = srgb_to_lab(image_a[rows])
        lab_b = srgb_to_lab(image_b[rows])
        total += float(np.sum(ciede2000_from_lab(lab_a, lab_b)))
    pixels = image_a.shape[0] * image_a.shape[1]
    return float(total / pixels)
