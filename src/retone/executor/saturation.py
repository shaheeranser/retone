"""Saturation and vibrance as a scaling of LAB chroma, leaving lightness untouched."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32Image = NDArray[np.float32]

# Chroma at which a colour counts as fully saturated for vibrance's low-chroma weighting.
VIBRANCE_CHROMA_REFERENCE = 80.0


def adjust_chroma(lab: Float32Image, saturation: int, vibrance: int) -> Float32Image:
    """Scale chroma by saturation everywhere and by vibrance mostly where chroma is low."""
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.hypot(a, b)
    muted = 1.0 - np.minimum(chroma / VIBRANCE_CHROMA_REFERENCE, 1.0)
    scale = (1.0 + saturation / 100) * (1.0 + vibrance / 100 * muted)
    scale = np.maximum(scale, 0.0).astype(np.float32)
    out = lab.copy()
    out[..., 1] = a * scale
    out[..., 2] = b * scale
    return out
