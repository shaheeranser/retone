from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retone.executor.apply import apply
from retone.executor.errors import ImageError
from retone.executor.params import Params
from retone.executor.schema import SCHEMA

shapes = st.tuples(st.integers(1, 8), st.integers(1, 8), st.just(3))
images_u8 = arrays(np.uint8, shapes)
images_u16 = arrays(np.uint16, shapes)
params_strategy = st.builds(
    Params, **{spec.name: st.integers(spec.minimum, spec.maximum) for spec in SCHEMA}
)


def patch(value: int, dtype: type = np.uint8) -> np.ndarray:
    return np.full((3, 3, 3), value, dtype=dtype)


@given(images_u8)
def test_identity_returns_the_input_exactly_for_8_bit(image: np.ndarray) -> None:
    assert np.array_equal(apply(image, Params()), image)


@given(images_u16)
def test_identity_returns_the_input_exactly_for_16_bit(image: np.ndarray) -> None:
    assert np.array_equal(apply(image, Params()), image)


@settings(max_examples=60)
@given(images_u8, params_strategy)
def test_apply_never_modifies_its_input(image: np.ndarray, params: Params) -> None:
    before = image.copy()
    apply(image, params)
    assert np.array_equal(image, before)


@settings(max_examples=60)
@given(images_u8, params_strategy)
def test_apply_is_deterministic(image: np.ndarray, params: Params) -> None:
    assert np.array_equal(apply(image, params), apply(image, params))


@settings(max_examples=60)
@given(st.one_of(images_u8, images_u16), params_strategy)
def test_output_keeps_shape_and_bit_depth(image: np.ndarray, params: Params) -> None:
    out = apply(image, params)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_exposure_acts_in_linear_light() -> None:
    # +1 EV on mid grey: linear 0.214 doubles to 0.428, which encodes to about 0.686.
    out = apply(patch(128), Params(exposure=50))
    assert out[0, 0, 0] == pytest.approx(0.686 * 255, abs=2)


def test_exposure_is_clipped_at_white_and_black() -> None:
    assert np.all(apply(patch(255), Params(exposure=100)) == 255)
    assert np.all(apply(patch(0), Params(exposure=-100)) == 0)


def test_warm_temperature_shifts_grey_toward_red() -> None:
    out = apply(patch(128), Params(temperature=60))
    assert out[0, 0, 0] > out[0, 0, 2]


def test_minus_100_saturation_gives_grey() -> None:
    colour = np.zeros((3, 3, 3), dtype=np.uint8)
    colour[..., 0], colour[..., 1], colour[..., 2] = 200, 80, 40
    out = apply(colour, Params(saturation=-100)).astype(int)
    assert np.ptp(out[0, 0]) <= 3


def test_16_bit_output_is_not_quantised_to_8_bit_steps() -> None:
    ramp = np.linspace(0, 65535, 4096, dtype=np.uint16).reshape(1, -1, 1).repeat(3, axis=2)
    out = apply(ramp, Params(contrast=40))
    assert np.count_nonzero(out % 257) > out.size // 2


@pytest.mark.parametrize(
    "image",
    [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.float32)],
    ids=["grey", "float"],
)
def test_unsupported_images_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(ImageError):
        apply(image, Params())
