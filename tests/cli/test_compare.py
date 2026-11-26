"""Comparison image construction: shapes, divider, and shape-mismatch rejection."""

from __future__ import annotations

import numpy as np
import pytest

from retone.cli.compare import GAP, compare, side_by_side, split
from retone.executor.errors import ImageError


def _pair() -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((10, 20, 3), dtype=np.uint8)
    after = np.full((10, 20, 3), 200, dtype=np.uint8)
    return before, after


def test_side_by_side_widens_by_the_gap() -> None:
    before, after = _pair()
    combined = side_by_side(before, after)
    assert combined.shape == (10, 20 * 2 + GAP, 3)
    assert combined[0, 0, 0] == 0
    assert combined[0, -1, 0] == 200
    assert combined[0, 20, 0] == 255  # the gap column


def test_split_keeps_left_original_right_render() -> None:
    before, after = _pair()
    combined = split(before, after)
    assert combined.shape == before.shape
    assert combined[0, 0, 0] == 0
    assert combined[0, -1, 0] == 200


@pytest.mark.parametrize("style", ["side-by-side", "split"])
def test_style_dispatch(style: str) -> None:
    before, after = _pair()
    assert compare(before, after, style).shape[0] == 10


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ImageError):
        compare(np.zeros((10, 20, 3), np.uint8), np.zeros((10, 21, 3), np.uint8), "split")


def test_unknown_style_is_rejected() -> None:
    before, after = _pair()
    with pytest.raises(ValueError):
        compare(before, after, "diagonal")
