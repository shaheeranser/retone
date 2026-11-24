"""The classical sharpening rule reacts to edge energy and noise."""

from __future__ import annotations

import cv2
import numpy as np

from retone.data.sharpen_rule import classical_sharpen_amount


def _checkerboard(size: int = 256, square: int = 8) -> np.ndarray:
    rows = np.arange(size) // square
    board = ((rows[:, None] + rows[None, :]) % 2) * 220 + 16
    return np.stack([board] * 3, axis=2).astype(np.uint8)


def test_blurrier_images_get_more_sharpening() -> None:
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    assert classical_sharpen_amount(blurred) > classical_sharpen_amount(sharp)


def test_noise_reduces_the_amount() -> None:
    clean = _checkerboard()
    rng = np.random.default_rng(0)
    noisy = np.clip(clean.astype(np.int16) + rng.integers(-25, 26, clean.shape), 0, 255)
    assert classical_sharpen_amount(noisy.astype(np.uint8)) <= classical_sharpen_amount(clean)


def test_amount_is_in_range() -> None:
    for image in (_checkerboard(), np.zeros((64, 64, 3), np.uint8)):
        assert 0 <= classical_sharpen_amount(image) <= 100
