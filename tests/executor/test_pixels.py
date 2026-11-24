from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retone.executor.errors import ImageError
from retone.executor.pixels import from_float, to_float

images_u8 = arrays(np.uint8, st.tuples(st.integers(1, 6), st.integers(1, 6), st.just(3)))
images_u16 = arrays(np.uint16, st.tuples(st.integers(1, 6), st.integers(1, 6), st.just(3)))


@given(images_u8)
def test_uint8_round_trips_exactly(image: np.ndarray) -> None:
    assert np.array_equal(from_float(to_float(image), np.dtype(np.uint8)), image)


@given(images_u16)
def test_uint16_round_trips_exactly(image: np.ndarray) -> None:
    assert np.array_equal(from_float(to_float(image), np.dtype(np.uint16)), image)


def test_to_float_scales_to_unit_range_and_returns_a_new_array() -> None:
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = to_float(image)
    assert result.dtype == np.float32
    assert result[0, 0, 0] == 0.0
    assert result[0, 0, 2] == 1.0
    result[0, 0, 0] = 0.5
    assert image[0, 0, 0] == 0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.int32),
    ],
    ids=["grey", "four-channel", "float", "int32"],
)
def test_to_float_rejects_unsupported_arrays(image: np.ndarray) -> None:
    with pytest.raises(ImageError):
        to_float(image)


def test_from_float_clips_out_of_range_values() -> None:
    rgb = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    assert from_float(rgb, np.dtype(np.uint8)).tolist() == [[[0, 128, 255]]]
