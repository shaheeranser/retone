"""Thumbnail-validity: full-resolution and thumbnail fits applied at full size."""

from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import make_image
from retone.eval.thumbnail_validity import thumbnail_validity
from retone.executor import Params
from retone.executor.errors import ImageError


def test_a_fit_that_ignores_its_input_gives_zero_difference() -> None:
    image = make_image(40, 60, seed=1)
    result = thumbnail_validity(
        [image], lambda _image: Params(exposure=8), lambda source: source[::2, ::2]
    )
    assert result.count == 1
    assert result.mean_delta_e == 0.0


def test_a_fit_that_depends_on_size_reports_a_difference() -> None:
    image = make_image(40, 60, seed=1)

    def fit(passed: np.ndarray) -> Params:
        return Params(exposure=int(passed.mean() // 4))

    result = thumbnail_validity([image], fit, lambda source: source[::2, ::2])
    assert result.mean_delta_e > 0.0


def test_both_stages_receive_the_expected_images() -> None:
    image = make_image(8, 9)
    shapes: list[tuple[int, ...]] = []

    def fit(passed: np.ndarray) -> Params:
        shapes.append(passed.shape)
        return Params()

    thumbnail_validity([image], fit, lambda source: source[:, :4])
    assert shapes == [(8, 9, 3), (8, 4, 3)]


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ImageError):
        thumbnail_validity([], lambda _image: Params(), lambda source: source)
