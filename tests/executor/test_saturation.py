from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from retone.executor.saturation import adjust_chroma

signed = st.integers(-100, 100)
labs = arrays(
    np.float32,
    st.tuples(st.integers(1, 4), st.integers(1, 4), st.just(3)),
    elements=st.floats(-100.0, 100.0, width=32),
)


def chroma(lab: np.ndarray) -> np.ndarray:
    return np.hypot(lab[..., 1], lab[..., 2])


def lab_pixel(lightness: float, a: float, b: float) -> np.ndarray:
    return np.array([[[lightness, a, b]]], dtype=np.float32)


def test_identity_returns_equal_values() -> None:
    lab = lab_pixel(50, 20, -30)
    assert np.array_equal(adjust_chroma(lab, 0, 0), lab)


def test_minus_100_saturation_removes_all_colour() -> None:
    assert np.allclose(chroma(adjust_chroma(lab_pixel(50, 40, -60), -100, 0)), 0.0)


def test_positive_saturation_scales_chroma_proportionally() -> None:
    result = adjust_chroma(lab_pixel(50, 30, 40), 50, 0)
    assert chroma(result)[0, 0] == pytest.approx(75.0, rel=1e-5)


def test_vibrance_boosts_muted_colours_more_than_vivid_ones() -> None:
    muted = adjust_chroma(lab_pixel(50, 5, 0), 0, 100)
    vivid = adjust_chroma(lab_pixel(50, 80, 0), 0, 100)
    assert chroma(muted)[0, 0] / 5 > chroma(vivid)[0, 0] / 80
    assert chroma(vivid)[0, 0] == pytest.approx(80.0, rel=1e-5)


@given(labs, signed, signed)
def test_lightness_is_never_changed(lab: np.ndarray, saturation: int, vibrance: int) -> None:
    assert np.array_equal(adjust_chroma(lab, saturation, vibrance)[..., 0], lab[..., 0])


@given(labs, signed, signed)
def test_hue_direction_is_preserved(lab: np.ndarray, saturation: int, vibrance: int) -> None:
    out = adjust_chroma(lab, saturation, vibrance)
    cross = lab[..., 1] * out[..., 2] - lab[..., 2] * out[..., 1]
    assert np.allclose(cross, 0.0, atol=1e-2)


@given(labs, signed, signed)
def test_input_is_not_modified(lab: np.ndarray, saturation: int, vibrance: int) -> None:
    before = lab.copy()
    adjust_chroma(lab, saturation, vibrance)
    assert np.array_equal(lab, before)
