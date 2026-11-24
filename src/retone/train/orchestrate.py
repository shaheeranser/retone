"""Training retry loop: detect prediction collapse and retrain with coarser bins."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retone.data.errors import DatasetError
from retone.executor import Params
from retone.train.formatting import TrainingExample
from retone.train.variance import (
    DEFAULT_COARSEN_STEP,
    DEFAULT_MIN_STD,
    collapsed,
    prediction_variance,
)

TrainOnce = Callable[[int, Sequence[TrainingExample]], object]
Predict = Callable[[object, Sequence[TrainingExample], int], Sequence[Params]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    initial_bin_step: int = 1
    coarsen_step: int = DEFAULT_COARSEN_STEP
    min_std: float = DEFAULT_MIN_STD
    sample_size: int = 16

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}")
        if self.initial_bin_step < 1 or self.coarsen_step < 1:
            raise ValueError("bin steps must be positive")
        if self.sample_size < 2:
            raise ValueError(f"sample_size must be at least 2, got {self.sample_size}")


@dataclass(frozen=True)
class TrainingOutcome:
    attempts: int
    bin_step: int
    collapsed: bool
    variance: dict[str, float]
    examples: int


def run_training(
    policy: RetryPolicy,
    records: Sequence[TrainingExample],
    *,
    train_once: TrainOnce,
    predict: Predict,
) -> TrainingOutcome:
    """Train, then judge prediction variance; on collapse retrain with coarser labels."""
    if not records:
        raise DatasetError(("no training examples",))

    bin_step = policy.initial_bin_step
    outcome = TrainingOutcome(0, bin_step, False, {}, 0)
    for attempt in range(1, policy.max_attempts + 1):
        coarsened = [record.coarsened(bin_step) for record in records]
        handle = train_once(bin_step, coarsened)
        sample = coarsened[: policy.sample_size]
        predictions = predict(handle, sample, len(sample))
        variance = prediction_variance(predictions)
        is_collapsed = len(predictions) >= 2 and collapsed(variance, policy.min_std)
        outcome = TrainingOutcome(
            attempts=attempt,
            bin_step=bin_step,
            collapsed=is_collapsed,
            variance=variance,
            examples=len(coarsened),
        )
        if not is_collapsed:
            return outcome
        bin_step *= policy.coarsen_step
    return outcome
