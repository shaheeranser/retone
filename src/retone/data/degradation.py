"""Sampling of bad parameters and the guards that keep a pair invertible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor.params import Params
from retone.executor.schema import SCHEMA

# Sharpening cannot be undone by the executor, so it is never degraded.
DEGRADABLE = tuple(spec.name for spec in SCHEMA if spec.name != "sharpen_amount")
_PEAK = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}


@dataclass(frozen=True)
class Magnitude:
    low: int
    high: int

    def __post_init__(self) -> None:
        if not 0 < self.low <= self.high <= 100:
            raise ValueError(f"invalid magnitude range {self.low}..{self.high}")


def sample_params(rng: np.random.Generator, magnitude: Magnitude, max_params: int = 4) -> Params:
    """Choose 1..max_params distinct degradable parameters with random signs and magnitudes."""
    count = int(rng.integers(1, max_params + 1))
    names = rng.choice(DEGRADABLE, size=count, replace=False)
    values = {name: 0 for name in DEGRADABLE}
    for name in names:
        size = int(rng.integers(magnitude.low, magnitude.high + 1))
        sign = -1 if rng.random() < 0.5 else 1
        values[str(name)] = sign * size
    return Params(**values)


def invert(params: Params) -> Params:
    """The correction that best undoes a degradation, clamped to the schema."""
    inverse = {spec.name: spec.clamp(-getattr(params, spec.name)) for spec in SCHEMA}
    inverse["sharpen_amount"] = 0
    return Params(**inverse)


def _peak(image: NDArray[Any]) -> float:
    return _PEAK[np.dtype(image.dtype)]


def _clipped_mask(image: NDArray[Any]) -> NDArray[np.bool_]:
    peak = _peak(image)
    mask: NDArray[np.bool_] = (image == 0) | (image == peak)
    # numpy 2.4 stubs type ndarray.any() as Any, so the cast is explicit.
    clipped: NDArray[np.bool_] = np.asarray(mask.any(axis=2), dtype=np.bool_)
    return clipped


def newly_clipped_fraction(target: NDArray[Any], degraded: NDArray[Any]) -> float:
    """Fraction of pixels clipped in the degraded image that were not clipped in the target."""
    newly = _clipped_mask(degraded) & ~_clipped_mask(target)
    return float(np.count_nonzero(newly) / newly.size)


def normalized_rms(image_a: NDArray[Any], image_b: NDArray[Any]) -> float:
    """Root-mean-square difference as a fraction of full scale, for residual filtering."""
    if image_a.shape != image_b.shape:
        raise ValueError(f"shapes differ: {image_a.shape} vs {image_b.shape}")
    peak = _peak(image_a)
    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.sqrt(np.mean((diff / peak) ** 2)))
