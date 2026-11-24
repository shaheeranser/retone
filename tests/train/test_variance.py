"""Coarse bins, per-parameter variance and collapse detection."""

from __future__ import annotations

import pytest

from retone.executor import Params
from retone.train.variance import (
    coarsen_params,
    collapsed,
    prediction_variance,
)


def test_coarsen_rounds_and_clamps() -> None:
    coarse = coarsen_params(Params(exposure=23, contrast=-13, sharpen_amount=97), 5)
    assert coarse.exposure == 25
    assert coarse.contrast == -15
    assert coarse.sharpen_amount == 95


def test_coarsen_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError):
        coarsen_params(Params(), 0)


def test_variance_is_zero_for_identical_predictions() -> None:
    variance = prediction_variance([Params(exposure=10), Params(exposure=10)])
    assert variance["exposure"] == 0.0
    assert collapsed(variance, min_std=1.0)


def test_spread_predictions_are_not_collapsed() -> None:
    variance = prediction_variance([Params(exposure=-40), Params(exposure=40)])
    assert variance["exposure"] == 1600.0
    assert not collapsed(variance, min_std=1.0)


def test_empty_variance_is_not_collapsed() -> None:
    assert prediction_variance([]) == {}
    assert not collapsed({})
