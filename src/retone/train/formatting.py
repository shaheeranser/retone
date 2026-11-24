"""Training examples and the Unsloth vision conversation format."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from retone.data.errors import DatasetError
from retone.executor import ParamError, Params, parse_params
from retone.train.variance import coarsen_params


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    # A Path for local JSONL, or a PIL image supplied by the Hugging Face dataset.
    image: Any
    instruction: str
    answer: str
    params: Params

    @classmethod
    def from_mapping(
        cls, row: Mapping[str, Any], image_root: Path | None = None
    ) -> TrainingExample:
        problems: list[str] = [
            f"record has no {name!r} field"
            for name in ("id", "image", "instruction", "answer")
            if name not in row
        ]
        if problems:
            raise DatasetError(tuple(problems))
        example_id, instruction, answer = row["id"], row["instruction"], row["answer"]
        if not all(isinstance(value, str) and value for value in (example_id, instruction, answer)):
            raise DatasetError(("id, instruction and answer must be non-empty strings",))

        image = row["image"]
        if isinstance(image, str):
            path = Path(image)
            image = path if image_root is None else image_root / path
            if not image.is_file():
                raise DatasetError((f"image not found at {image}",))

        params = _params(row, answer)
        return cls(
            example_id=example_id,
            image=image,
            instruction=instruction,
            answer=answer,
            params=params,
        )

    def coarsened(self, step: int) -> TrainingExample:
        """Return this example with its label and answer rounded to `step` bins."""
        if step <= 1:
            return self
        params = coarsen_params(self.params, step)
        return replace(self, params=params, answer=json.dumps(params.as_dict()))


def _params(row: Mapping[str, Any], answer: str) -> Params:
    raw = row.get("params")
    if raw is None:
        try:
            raw = json.loads(answer)
        except json.JSONDecodeError as err:
            raise DatasetError((f"answer is not valid JSON: {err.msg}",)) from err
    if not isinstance(raw, Mapping):
        raise DatasetError(("params must be an object",))
    try:
        return parse_params(raw).params
    except ParamError as err:
        raise DatasetError(err.problems) from err


def load_examples(
    path: Path,
    *,
    image_root: Path | None = None,
    split: str | None = None,
    bin_step: int = 1,
) -> tuple[TrainingExample, ...]:
    """Read examples.jsonl, validate every row and optionally coarsen the labels."""
    examples: list[TrainingExample] = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as err:
            raise DatasetError((f"line {index}: invalid JSON ({err.msg})",)) from err
        if not isinstance(row, Mapping):
            raise DatasetError((f"line {index}: expected a JSON object",))
        if split is not None and row.get("split") != split:
            continue
        try:
            examples.append(TrainingExample.from_mapping(row, image_root))
        except DatasetError as err:
            raise DatasetError(tuple(f"line {index}: {p}" for p in err.problems)) from err
    if not examples:
        raise DatasetError((f"{path} has no examples for split {split!r}",))
    return tuple(example.coarsened(bin_step) for example in examples)


def to_vision_record(example: TrainingExample) -> dict[str, Any]:
    """One Unsloth vision example: user text plus image, assistant parameter JSON."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example.instruction},
                    {"type": "image", "image": example.image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.answer}],
            },
        ]
    }
