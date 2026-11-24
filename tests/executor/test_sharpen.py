from __future__ import annotations

import numpy as np
import pytest

from retone.executor.sharpen import sharpen_sigma, unsharp_lightness


def step_edge(size: int = 1200) -> np.ndarray:
    lab = np.zeros((size, size, 3), dtype=np.float32)
    lab[..., 0] = 30.0
    lab[:, size // 2 :, 0] = 70.0
    lab[..., 1] = 10.0
    lab[..., 2] = -10.0
    return lab


def test_zero_amount_changes_nothing() -> None:
    lab = step_edge()
    assert np.array_equal(unsharp_lightness(lab, 0), lab)


def test_sharpening_overshoots_both_sides_of_an_edge() -> None:
    lab = step_edge()
    out = unsharp_lightness(lab, 100)
    edge = lab.shape[1] // 2
    assert out[600, edge, 0] > 70.0
    assert out[600, edge - 1, 0] < 30.0


def test_flat_regions_are_untouched() -> None:
    lab = step_edge()
    out = unsharp_lightness(lab, 100)
    assert out[600, 10, 0] == pytest.approx(30.0, abs=1e-4)
    assert out[600, -10, 0] == pytest.approx(70.0, abs=1e-4)


def test_chroma_channels_pass_through_unchanged() -> None:
    lab = step_edge()
    out = unsharp_lightness(lab, 100)
    assert np.array_equal(out[..., 1:], lab[..., 1:])


def test_more_amount_overshoots_more() -> None:
    lab = step_edge()
    edge = lab.shape[1] // 2
    mild = unsharp_lightness(lab, 30)[600, edge, 0]
    strong = unsharp_lightness(lab, 100)[600, edge, 0]
    assert strong > mild > 70.0


def test_lightness_stays_in_range() -> None:
    out = unsharp_lightness(step_edge(), 100)
    assert out[..., 0].min() >= 0.0 and out[..., 0].max() <= 100.0


def test_input_is_not_modified() -> None:
    lab = step_edge()
    before = lab.copy()
    unsharp_lightness(lab, 100)
    assert np.array_equal(lab, before)


@pytest.mark.parametrize("scale", [2, 3, 5], ids=["x2", "x3", "x5"])
def test_radius_scales_linearly_with_the_diagonal(scale: int) -> None:
    assert sharpen_sigma(400 * scale, 600 * scale) == pytest.approx(scale * sharpen_sigma(400, 600))
