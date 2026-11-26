"""Build before/after comparison images from an original and its enhanced render."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor.errors import ImageError

IntImage = NDArray[np.integer[Any]]

GAP = 8
GAP_VALUE = 255


def side_by_side(before: IntImage, after: IntImage) -> IntImage:
    """Place the original left and the enhanced render right, separated by a light gap."""
    height = before.shape[0]
    gap = np.full((height, GAP, before.shape[2]), GAP_VALUE, dtype=before.dtype)
    return np.concatenate([before, gap, after], axis=1)


def split(before: IntImage, after: IntImage) -> IntImage:
    """Keep the original's left half and the render's right half with a divider line."""
    width = before.shape[1]
    combined = after.copy()
    half = width // 2
    combined[:, :half] = before[:, :half]
    combined[:, half : half + 1] = GAP_VALUE
    return combined


def compare(before: IntImage, after: IntImage, style: str) -> IntImage:
    """Dispatch on a comparison style, rejecting shapes that cannot be compared."""
    if before.shape != after.shape:
        raise ImageError(
            f"cannot compare a {before.shape[1]}x{before.shape[0]} original with a "
            f"{after.shape[1]}x{after.shape[0]} render"
        )
    if style == "side-by-side":
        return side_by_side(before, after)
    if style == "split":
        return split(before, after)
    raise ValueError(f"unknown comparison style {style!r}")
