"""Reads and writes image files; the only place OpenCV's BGR order meets our RGB order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.executor.errors import ImageError


def load_image(path: Path) -> NDArray[np.integer[Any]]:
    """Load an image as HxWx3 RGB, keeping 8 or 16 bit depth."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if bgr is None:
        raise ImageError(f"cannot read an image from {path}")
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def save_image(path: Path, image: NDArray[np.integer[Any]]) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise ImageError(f"cannot write an image to {path}")
