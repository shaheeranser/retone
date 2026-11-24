"""Prediction variance, collapse detection and coarse binning of parameter labels."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from retone.executor import SCHEMA, Params

DEFAULT_COARSEN_STEP = 5
DEFAULT_MIN_STD = 1.0


def coarsen_params(params: Params, step: int) -> Params:
    """Round every parameter to a multiple of `step`, clamped to the schema."""
    if step < 1:
        raise ValueError(f"coarsen step must be positive, got {step}")
    values = {
        spec.name: spec.clamp(round(getattr(params, spec.name) / step) * step) for spec in SCHEMA
    }
    return Params(**values)


def prediction_variance(predictions: Sequence[Params]) -> dict[str, float]:
    """Population variance per parameter over a sample of predictions."""
    if not predictions:
        return {}
    matrix = np.array(
        [[getattr(params, spec.name) for spec in SCHEMA] for params in predictions],
        dtype=np.float64,
    )
    return {
        spec.name: float(value) for spec, value in zip(SCHEMA, np.var(matrix, axis=0), strict=True)
    }


def collapsed(variance: dict[str, float], min_std: float = DEFAULT_MIN_STD) -> bool:
    """True when the average per-parameter standard deviation is below `min_std`."""
    if not variance:
        return False
    mean_std = sum(math.sqrt(value) for value in variance.values()) / len(variance)
    return mean_std < min_std
