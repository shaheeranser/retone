from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from retone.executor.tone import tone_curve

signed = st.integers(-100, 100)
RAMP = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(1, -1, 1).repeat(3, axis=2)


def curve(contrast: int = 0, highlights: int = 0, shadows: int = 0) -> np.ndarray:
    return tone_curve(RAMP, contrast, highlights, shadows)[0, :, 0]


def at(values: np.ndarray, x: float) -> float:
    return float(values[round(x * 1000)])


def test_identity_leaves_the_ramp_unchanged() -> None:
    assert np.array_equal(curve(), RAMP[0, :, 0])


@given(signed, signed, signed)
def test_black_and_white_stay_fixed(contrast: int, highlights: int, shadows: int) -> None:
    values = curve(contrast, highlights, shadows)
    assert at(values, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert at(values, 1.0) == pytest.approx(1.0, abs=1e-6)


@given(signed, signed, signed)
def test_curve_is_monotonic_and_stays_in_range(
    contrast: int, highlights: int, shadows: int
) -> None:
    values = curve(contrast, highlights, shadows)
    assert np.all(np.diff(values) >= -1e-6)
    assert values.min() >= -1e-6 and values.max() <= 1.0 + 1e-6


@pytest.mark.parametrize(
    ("kwargs", "x", "direction"),
    [
        ({"contrast": 50}, 0.25, -1),
        ({"contrast": 50}, 0.75, +1),
        ({"contrast": -50}, 0.25, +1),
        ({"contrast": -50}, 0.75, -1),
        ({"highlights": 80}, 0.7, +1),
        ({"highlights": -80}, 0.7, -1),
        ({"shadows": 80}, 0.3, +1),
        ({"shadows": -80}, 0.3, -1),
    ],
    ids=[
        "contrast-up-darkens-shadows",
        "contrast-up-brightens-highlights",
        "contrast-down-lifts-shadows",
        "contrast-down-lowers-highlights",
        "highlights-up",
        "highlights-down",
        "shadows-up",
        "shadows-down",
    ],
)
def test_direction_of_effect(kwargs: dict[str, int], x: float, direction: int) -> None:
    assert (at(curve(**kwargs), x) - x) * direction > 0.01


def test_highlights_barely_touch_the_darks_and_shadows_barely_touch_the_brights() -> None:
    assert abs(at(curve(highlights=100), 0.1) - 0.1) < 0.01
    assert abs(at(curve(shadows=100), 0.9) - 0.9) < 0.01


def test_input_is_not_modified() -> None:
    before = RAMP.copy()
    tone_curve(RAMP, 40, 30, -20)
    assert np.array_equal(RAMP, before)
