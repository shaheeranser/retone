"""Linear-light thumbnailing: sizing, dtype, and brightness fidelity."""

from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import make_image
from retone.inference.thumbnail import THUMBNAIL_LONG_EDGE, thumbnail


def test_caps_the_long_edge_and_returns_uint8() -> None:
    image = make_image(1200, 800)
    shot = thumbnail(image)
    assert max(shot.shape[:2]) == THUMBNAIL_LONG_EDGE
    assert shot.dtype == np.uint8


def test_never_upscales_small_images() -> None:
    image = make_image(40, 60)
    assert thumbnail(image).shape == image.shape


def test_linear_average_keeps_brightness() -> None:
    # A 2x2 checker of black and white averages to mid-grey in linear light (~188),
    # not to the gamma-encoded midpoint of 128.
    checker = np.zeros((2, 2, 3), dtype=np.uint8)
    checker[0, 0] = 255
    checker[1, 1] = 255
    grey = thumbnail(checker, long_edge=1)
    assert grey.shape == (1, 1, 3)
    assert 180 <= int(grey[0, 0, 0]) <= 194


def test_flat_colour_is_preserved() -> None:
    flat = np.full((10, 10, 3), 120, dtype=np.uint8)
    shots = thumbnail(flat, long_edge=5)
    assert shots.shape == (5, 5, 3)
    assert pytest.approx(120, abs=1) == float(np.median(shots))


def test_sixteen_bit_input_is_scaled_not_clipped() -> None:
    image = np.full((8, 8, 3), 65535, dtype=np.uint16)
    shot = thumbnail(image, long_edge=4)
    assert shot.dtype == np.uint8
    assert int(shot.max()) == 255
