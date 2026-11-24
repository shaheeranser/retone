"""Tone curves for contrast, highlights and shadows on sRGB-encoded float values."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32Image = NDArray[np.float32]

# Strengths are capped so the sum of curves keeps a positive slope everywhere (monotonic).
HIGHLIGHT_SHADOW_STRENGTH = 0.12
_BUMP_NORMALISER = 27.0 / 4.0


def _bump(x: Float32Image) -> Float32Image:
    """Smooth lobe peaking at 1 near x = 2/3 and vanishing at 0 and 1."""
    return np.asarray(_BUMP_NORMALISER * x * x * (1.0 - x), dtype=np.float32)


def apply_contrast(x: Float32Image, contrast: int) -> Float32Image:
    """S-curve about mid-grey; positive contrast steepens the middle."""
    power = np.float32(2.0 ** (contrast / 100))
    below = 0.5 * (2.0 * x) ** power
    above = 1.0 - 0.5 * (2.0 * (1.0 - x)) ** power
    return np.where(x < 0.5, below, above).astype(np.float32)


def apply_highlights_shadows(x: Float32Image, highlights: int, shadows: int) -> Float32Image:
    """Lift or lower the bright and dark lobes; positive values brighten them."""
    lift_highlights = HIGHLIGHT_SHADOW_STRENGTH * highlights / 100 * _bump(x)
    lift_shadows = HIGHLIGHT_SHADOW_STRENGTH * shadows / 100 * _bump(1.0 - x)
    return np.asarray(x + lift_highlights + lift_shadows, dtype=np.float32)


def tone_curve(rgb: Float32Image, contrast: int, highlights: int, shadows: int) -> Float32Image:
    """Apply contrast, then highlights and shadows, skipping each at identity."""
    out = np.clip(rgb, 0.0, 1.0)
    if contrast != 0:
        out = apply_contrast(out, contrast)
    if highlights != 0 or shadows != 0:
        out = apply_highlights_shadows(out, highlights, shadows)
    return out
