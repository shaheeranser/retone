"""Small deterministic RGB images for tests, without any network or fixture files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor import save_image


def make_image(height: int = 64, width: int = 96, seed: int = 0) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    red = (xs * 255 / max(1, width - 1)).astype(np.float64)
    green = (ys * 255 / max(1, height - 1)).astype(np.float64)
    blue = np.full((height, width), 96.0)
    base = np.stack([red, green, blue], axis=2) + rng.normal(0.0, 6.0, (height, width, 3))
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


def write_image(path: Path, image: NDArray[np.integer[Any]]) -> Path:
    save_image(path, image)
    return path
