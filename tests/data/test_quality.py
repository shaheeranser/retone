"""Source-quality measurements and thresholds."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_fixtures import make_image
from retone.data.quality import QualityThresholds, acceptable, measure_quality


def _thresholds(**overrides: object) -> QualityThresholds:
    values: dict[str, object] = {"min_side": 128, "min_sharpness": 5.0}
    values.update(overrides)
    return QualityThresholds(**values)  # type: ignore[arg-type]


def test_sharp_well_exposed_image_passes() -> None:
    image = make_image(300, 400, seed=1)
    assert acceptable(measure_quality(image), _thresholds())


def test_blurred_image_fails_sharpness() -> None:
    blurred = cv2.GaussianBlur(make_image(300, 400, seed=2), (0, 0), 5.0)
    metrics = measure_quality(blurred)
    assert metrics.sharpness < 5.0
    assert not acceptable(metrics, _thresholds())


def test_dark_and_clipped_images_fail() -> None:
    dark = np.full((300, 400, 3), 3, dtype=np.uint8)
    white = np.full((300, 400, 3), 255, dtype=np.uint8)
    assert not acceptable(measure_quality(dark), _thresholds())
    assert not acceptable(measure_quality(white), _thresholds())


def test_small_image_fails_resolution() -> None:
    assert not acceptable(measure_quality(make_image(100, 100, seed=3)), _thresholds())


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        QualityThresholds(min_side=0)
    with pytest.raises(ValueError):
        QualityThresholds(min_luma=200, max_luma=100)
