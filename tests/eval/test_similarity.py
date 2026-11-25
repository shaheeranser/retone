"""PSNR and SSIM diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import make_image
from retone.eval.similarity import psnr, ssim
from retone.executor.errors import ImageError


def test_identical_images_are_perfect() -> None:
    image = make_image(40, 60, seed=1)
    assert psnr(image, image) == 100.0
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_noise_lowers_both_metrics() -> None:
    image = make_image(40, 60, seed=1)
    rng = np.random.default_rng(2)
    noisy = np.clip(image.astype(np.int16) + rng.integers(-30, 31, image.shape), 0, 255)
    noisy = noisy.astype(np.uint8)
    assert psnr(noisy, image) < 100.0
    assert ssim(noisy, image) < 1.0
    assert ssim(noisy, image) >= 0.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ImageError):
        psnr(make_image(40, 60), make_image(40, 61))
    with pytest.raises(ImageError):
        ssim(make_image(40, 60), make_image(40, 61))
