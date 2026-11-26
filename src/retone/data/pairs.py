"""Constructing (input, target, label) pairs by degrading a good photo and inverting it."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.data.degradation import (
    Magnitude,
    invert,
    newly_clipped_fraction,
    normalized_rms,
    sample_params,
)
from retone.executor import Params, apply

IntImage = NDArray[np.integer[Any]]


class PairKind(StrEnum):
    IDENTITY = "identity"
    SYNTHETIC = "synthetic"
    MILD = "mild"


@dataclass(frozen=True)
class PairPolicy:
    identity_weight: float = 0.05
    synthetic_weight: float = 0.55
    mild_weight: float = 0.4
    synthetic_magnitude: Magnitude = field(default_factory=lambda: Magnitude(20, 70))
    mild_magnitude: Magnitude = field(default_factory=lambda: Magnitude(3, 15))
    min_params: int = 2
    max_params: int = 4
    label_step: int = 5
    max_newly_clipped: float = 0.02
    max_residual: float = 0.02
    max_attempts: int = 12

    def __post_init__(self) -> None:
        weights = (self.identity_weight, self.synthetic_weight, self.mild_weight)
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(f"pair weights must be non-negative and sum to 1, got {weights}")
        if not 1 <= self.label_step <= 100:
            raise ValueError(f"label_step must be in 1..100, got {self.label_step}")
        if not 0.0 <= self.max_newly_clipped <= 1.0:
            raise ValueError(f"max_newly_clipped must be in [0, 1], got {self.max_newly_clipped}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}")
        if not 1 <= self.min_params <= self.max_params <= 8:
            raise ValueError(
                f"need 1 <= min_params <= max_params <= 8, got {self.min_params}..{self.max_params}"
            )


@dataclass(frozen=True)
class Pair:
    input_image: IntImage
    label: Params
    degradation: Params
    kind: PairKind
    residual: float
    newly_clipped: float

    def accepted(self, policy: PairPolicy) -> bool:
        return self.residual <= policy.max_residual and (
            self.newly_clipped <= policy.max_newly_clipped
        )


def choose_kind(rng: np.random.Generator, policy: PairPolicy) -> PairKind:
    """Draw a pair kind from the policy weights using a caller-seeded generator."""
    draw = rng.random()
    if draw < policy.identity_weight:
        return PairKind.IDENTITY
    if draw < policy.identity_weight + policy.synthetic_weight:
        return PairKind.SYNTHETIC
    return PairKind.MILD


def _nonzero(params: Params) -> int:
    return sum(1 for value in params.as_dict().values() if value != 0)


def _degraded_pair(
    target: IntImage, rng: np.random.Generator, kind: PairKind, policy: PairPolicy
) -> Pair:
    base = policy.synthetic_magnitude if kind is PairKind.SYNTHETIC else policy.mild_magnitude
    for scale in (1.0, 0.5, 0.25):
        for _ in range(policy.max_attempts):
            magnitude = Magnitude(max(1, round(base.low * scale)), max(1, round(base.high * scale)))
            degradation = sample_params(rng, magnitude, policy.max_params, policy.min_params)
            input_image = apply(target, degradation)
            clipped = newly_clipped_fraction(target, input_image)
            if clipped > policy.max_newly_clipped:
                continue
            label = invert(degradation).coarsen(policy.label_step)
            # Rounding to the label step can erase small values; keep retrying
            # until the label still shows the required number of corrections.
            if _nonzero(label) < policy.min_params:
                continue
            corrected = apply(input_image, label)
            residual = normalized_rms(target, corrected)
            # Retry inside the attempt budget: returning a pair the caller then
            # discards loses the heavy multi-parameter pairs that fail most often.
            if residual > policy.max_residual:
                continue
            return Pair(
                input_image=input_image,
                label=label,
                degradation=degradation,
                kind=kind,
                residual=residual,
                newly_clipped=clipped,
            )
    # No attempt passed both guards; return a small pair whose label still
    # survives rounding, and let the caller filter it if it does not.
    degradation = sample_params(rng, Magnitude(3, 5), policy.max_params, policy.min_params)
    input_image = apply(target, degradation)
    label = invert(degradation).coarsen(policy.label_step)
    return Pair(
        input_image=input_image,
        label=label,
        degradation=degradation,
        kind=kind,
        residual=normalized_rms(target, apply(input_image, label)),
        newly_clipped=newly_clipped_fraction(target, input_image),
    )


def build_pair(
    target: IntImage, rng: np.random.Generator, kind: PairKind, policy: PairPolicy
) -> Pair:
    """Build one pair. `target` is the good image the correction should approach."""
    if kind is PairKind.IDENTITY:
        identity = Params()
        return Pair(
            input_image=target.copy(),
            label=identity,
            degradation=identity,
            kind=kind,
            residual=0.0,
            newly_clipped=0.0,
        )
    return _degraded_pair(target, rng, kind, policy)
