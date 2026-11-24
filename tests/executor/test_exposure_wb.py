from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from retone.executor.exposure_wb import channel_gains

signed = st.integers(-100, 100)


def test_identity_parameters_give_unit_gains() -> None:
    assert np.allclose(channel_gains(0, 0, 0), [1.0, 1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    ("exposure", "gain"),
    [(50, 2.0), (-50, 0.5), (100, 4.0), (-100, 0.25)],
    ids=["+1EV", "-1EV", "+2EV", "-2EV"],
)
def test_exposure_scales_every_channel_by_powers_of_two(exposure: int, gain: float) -> None:
    assert np.allclose(channel_gains(exposure, 0, 0), gain, rtol=1e-5)


def test_warmer_boosts_red_over_blue_and_cooler_does_the_opposite() -> None:
    warm = channel_gains(0, 50, 0)
    cool = channel_gains(0, -50, 0)
    assert warm[0] > warm[2]
    assert cool[0] < cool[2]


def test_positive_tint_lowers_green() -> None:
    assert channel_gains(0, 0, 50)[1] < 1.0


@given(signed, signed, signed)
def test_white_balance_never_changes_luminance(exposure: int, temperature: int, tint: int) -> None:
    luma = np.array([0.2126, 0.7152, 0.0722])
    with_wb = channel_gains(exposure, temperature, tint) @ luma
    exposure_only = channel_gains(exposure, 0, 0) @ luma
    assert with_wb == pytest.approx(exposure_only, rel=1e-5)


@given(signed, signed, signed)
def test_gains_are_positive_and_finite(exposure: int, temperature: int, tint: int) -> None:
    gains = channel_gains(exposure, temperature, tint)
    assert np.all(gains > 0) and np.all(np.isfinite(gains))
