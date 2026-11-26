"""Benchmark measurement with an injected clock, model step and memory probe."""

from __future__ import annotations

import numpy as np
import pytest

from image_fixtures import make_image
from retone.eval.benchmark import measure
from retone.executor import Params
from retone.executor.errors import ImageError


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_model_and_render_are_timed_separately() -> None:
    image = make_image(1200, 2000)
    clock = _Clock([10.0, 10.4, 11.0])
    result = measure(image, lambda _image: Params(exposure=5), clock=clock, peak=lambda: 1234)
    assert result.mean_model_seconds == pytest.approx(0.4)
    assert result.mean_render_seconds == pytest.approx(0.6)
    assert result.mean_total_seconds == pytest.approx(1.0)
    assert result.peak_rss_bytes == 1234
    assert result.megapixels == pytest.approx(2.4)


def test_repeats_are_averaged() -> None:
    image = make_image(10, 10)
    clock = _Clock([0.0, 0.2, 0.5, 1.0, 1.4, 2.0])
    result = measure(image, lambda _image: Params(), repeats=2, clock=clock, peak=lambda: 0)
    assert result.runs == 2
    assert result.mean_model_seconds == pytest.approx((0.2 + 0.4) / 2)
    assert result.mean_render_seconds == pytest.approx((0.3 + 0.6) / 2)


def test_fit_receives_the_full_resolution_image() -> None:
    image = make_image(8, 9)
    seen: list[tuple[int, ...]] = []

    def fit(passed: np.ndarray) -> Params:
        seen.append(passed.shape)
        return Params()

    measure(image, fit, clock=_Clock([0.0, 0.0, 0.0]), peak=lambda: 0)
    assert seen == [(8, 9, 3)]


def test_invalid_repeats_are_rejected() -> None:
    with pytest.raises(ValueError):
        measure(make_image(4, 4), lambda _image: Params(), repeats=0)


def test_non_image_input_is_rejected() -> None:
    with pytest.raises(ImageError):
        measure(np.zeros((4, 4), dtype=np.uint8), lambda _image: Params())
