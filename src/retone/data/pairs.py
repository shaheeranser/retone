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
    identity_weight: float = 0.3
    synthetic_weight: float = 0.5
    mild_weight: float = 0.2
    synthetic_magnitude: Magnitude = field(default_factory=lambda: Magnitude(20, 70))
    mild_magnitude: Magnitude = field(default_factory=lambda: Magnitude(3, 15))
    max_params: int = 4
    max_newly_clipped: float = 0.02
    max_residual: float = 0.02
    max_attempts: int = 12

    def __post_init__(self) -> None:
        weights = (self.identity_weight, self.synthetic_weight, self.mild_weight)
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(f"pair weights must be non-negative and sum to 1, got {weights}")
        if not 0.0 <= self.max_newly_clipped <= 1.0:
            raise ValueError(f"max_newly_clipped must be in [0, 1], got {self.max_newly_clipped}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}")


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


def _degraded_pair(
    target: IntImage, rng: np.random.Generator, kind: PairKind, policy: PairPolicy
) -> Pair:
    base = policy.synthetic_magnitude if kind is PairKind.SYNTHETIC else policy.mild_magnitude
    for scale in (1.0, 0.5, 0.25):
        for _ in range(policy.max_attempts):
            magnitude = Magnitude(max(1, round(base.low * scale)), max(1, round(base.high * scale)))
            degradation = sample_params(rng, magnitude, policy.max_params)
            input_image = apply(target, degradation)
            clipped = newly_clipped_fraction(target, input_image)
            if clipped > policy.max_newly_clipped:
                continue
            label = invert(degradation)
            corrected = apply(input_image, label)
            return Pair(
                input_image=input_image,
                label=label,
                degradation=degradation,
                kind=kind,
                residual=normalized_rms(target, corrected),
                newly_clipped=clipped,
            )
    # Even an extreme image keeps a pair; the caller filters it if the guards fail.
    degradation = sample_params(rng, Magnitude(1, 3), policy.max_params)
    input_image = apply(target, degradation)
    label = invert(degradation)
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
