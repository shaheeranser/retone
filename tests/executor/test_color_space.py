from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retone.executor.color_space import lab_to_rgb, linear_to_srgb, rgb_to_lab, srgb_to_linear

unit_rgb = arrays(
    np.float32,
    st.tuples(st.integers(1, 5), st.integers(1, 5), st.just(3)),
    elements=st.floats(0.0, 1.0, width=32),
)


@pytest.mark.parametrize(
    ("encoded", "linear"),
    [(0.0, 0.0), (1.0, 1.0), (0.5, 0.21404114), (0.04, 0.04 / 12.92)],
    ids=["black", "white", "mid-grey", "linear-segment"],
)
def test_srgb_to_linear_known_values(encoded: float, linear: float) -> None:
    result = srgb_to_linear(np.full((1, 1, 3), encoded, dtype=np.float32))
    assert np.allclose(result, linear, atol=1e-6)


@given(unit_rgb)
def test_srgb_linear_round_trip(rgb: np.ndarray) -> None:
    assert np.allclose(linear_to_srgb(srgb_to_linear(rgb)), rgb, atol=1e-5)


@given(unit_rgb)
def test_srgb_to_linear_is_monotonic_per_channel(rgb: np.ndarray) -> None:
    flat = np.sort(rgb.reshape(-1))
    assert np.all(np.diff(srgb_to_linear(flat.reshape(1, -1, 1)).reshape(-1)) >= -1e-7)


def test_lab_of_white_and_black() -> None:
    white = rgb_to_lab(np.ones((1, 1, 3), dtype=np.float32))[0, 0]
    black = rgb_to_lab(np.zeros((1, 1, 3), dtype=np.float32))[0, 0]
    assert white[0] == pytest.approx(100.0, abs=0.01)
    assert abs(white[1]) < 0.05
    assert abs(white[2]) < 0.05
    assert black[0] == pytest.approx(0.0, abs=0.01)


@given(unit_rgb)
def test_lab_round_trip_stays_within_one_8_bit_level(rgb: np.ndarray) -> None:
    # OpenCV's float LAB uses lookup tables, so the round trip is accurate to about 4e-3.
    assert np.allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=5e-3)
