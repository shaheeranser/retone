"""The train/measure/retrain loop and its coarser-bin fallback."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retone.data.errors import DatasetError
from retone.executor import Params
from retone.train.formatting import TrainingExample
from retone.train.orchestrate import RetryPolicy, run_training


def _example(exposure: int) -> TrainingExample:
    return TrainingExample(
        example_id=f"e{exposure}",
        image="image",
        instruction="instruction",
        answer="{}",
        params=Params(exposure=exposure),
    )


def test_spread_predictions_finish_in_one_attempt() -> None:
    steps: list[int] = []

    def train_once(step: int, records: Sequence[TrainingExample]) -> object:
        steps.append(step)
        return object()

    def predict(
        handle: object, records: Sequence[TrainingExample], limit: int
    ) -> tuple[Params, ...]:
        return (Params(exposure=-20), Params(exposure=20))

    outcome = run_training(
        RetryPolicy(max_attempts=2, sample_size=2),
        [_example(23)],
        train_once=train_once,
        predict=predict,
    )
    assert steps == [1]
    assert outcome.attempts == 1
    assert outcome.bin_step == 1
    assert not outcome.collapsed


def test_collapse_retrains_with_coarser_labels() -> None:
    steps: list[int] = []
    seen: list[int] = []

    def train_once(step: int, records: Sequence[TrainingExample]) -> object:
        steps.append(step)
        seen.extend(record.params.exposure for record in records)
        return object()

    def predict(
        handle: object, records: Sequence[TrainingExample], limit: int
    ) -> tuple[Params, ...]:
        return (Params(), Params())

    outcome = run_training(
        RetryPolicy(max_attempts=2, coarsen_step=5, sample_size=2),
        [_example(23)],
        train_once=train_once,
        predict=predict,
    )
    assert steps == [1, 5]
    assert seen == [23, 25]
    assert outcome.attempts == 2
    assert outcome.bin_step == 5
    assert outcome.collapsed


def test_empty_records_are_rejected() -> None:
    with pytest.raises(DatasetError):
        run_training(
            RetryPolicy(),
            [],
            train_once=lambda step, records: object(),
            predict=lambda handle, records, limit: (),
        )


def test_sample_size_must_allow_a_comparison() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(sample_size=1)
