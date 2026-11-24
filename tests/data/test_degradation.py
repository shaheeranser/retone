"""Seeded parameter sampling, inversion and clipping measurement."""

from __future__ import annotations

import numpy as np
import pytest

from retone.data.degradation import (
    DEGRADABLE,
    Magnitude,
    invert,
    newly_clipped_fraction,
    normalized_rms,
    sample_params,
)
from retone.executor import Params


def test_sampling_respects_range_and_parameter_cap() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        params = sample_params(rng, Magnitude(20, 70), max_params=4)
        values = params.as_dict()
        assert params.sharpen_amount == 0
        non_zero = [name for name in DEGRADABLE if values[name] != 0]
        assert 1 <= len(non_zero) <= 4
        assert all(20 <= abs(values[name]) <= 70 for name in non_zero)


def test_invert_negates_and_clamps() -> None:
    params = Params(exposure=70, temperature=-30, sharpen_amount=40)
    inverse = invert(params)
    assert inverse.exposure == -70
    assert inverse.temperature == 30
    assert inverse.sharpen_amount == 0


def test_newly_clipped_counts_only_new_clipping() -> None:
    target = np.full((2, 2, 3), 128, dtype=np.uint8)
    degraded = target.copy()
    degraded[0, 0] = 0
    degraded[1, 1] = 255
    assert newly_clipped_fraction(target, degraded) == 0.5


def test_no_new_clipping_when_target_was_already_clipped() -> None:
    target = np.zeros((2, 2, 3), dtype=np.uint8)
    degraded = np.zeros((2, 2, 3), dtype=np.uint8)
    assert newly_clipped_fraction(target, degraded) == 0.0


def test_rms_is_zero_for_identical_images() -> None:
    image = np.full((4, 4, 3), 90, dtype=np.uint8)
    assert normalized_rms(image, image) == 0.0


def test_magnitude_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        Magnitude(0, 10)
    with pytest.raises(ValueError):
        Magnitude(20, 10)
