"""Example loading, label coarsening and the Unsloth conversation record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from golden_support import assert_golden
from image_fixtures import make_image, write_image
from retone.data.errors import DatasetError
from retone.executor import Params
from retone.train.formatting import TrainingExample, load_examples, to_vision_record


def _row(photo_id: str, split: str, exposure: int = 0) -> dict[str, object]:
    params = Params(exposure=exposure, contrast=7)
    return {
        "id": f"{photo_id}-00",
        "split": split,
        "image": f"{photo_id}.png",
        "instruction": "Make it better.",
        "answer": json.dumps(params.as_dict()),
        "params": params.as_dict(),
    }


def _corpus(tmp_path: Path) -> Path:
    write_image(tmp_path / "a.png", make_image(32, 40, seed=1))
    write_image(tmp_path / "b.png", make_image(32, 40, seed=2))
    manifest = tmp_path / "examples.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in (_row("a", "train", 23), _row("b", "test"))) + "\n"
    )
    return manifest


def test_load_examples_filters_split_and_coarsens(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    train = load_examples(manifest, image_root=tmp_path, split="train", bin_step=5)
    assert len(train) == 1
    assert train[0].params.exposure == 25
    assert json.loads(train[0].answer)["exposure"] == 25


def test_invalid_rows_report_the_problem() -> None:
    with pytest.raises(DatasetError):
        TrainingExample.from_mapping({"id": "x", "image": "x.png"})
    with pytest.raises(DatasetError):
        TrainingExample.from_mapping(
            {"id": "x", "image": "x.png", "instruction": "i", "answer": "not json"}
        )


def test_conversation_golden(update_golden: bool) -> None:
    example = TrainingExample(
        example_id="a-00",
        image="image-object",
        instruction="Make it better.",
        answer='{"exposure": 24}',
        params=Params(exposure=24),
    )
    rendered = json.dumps(to_vision_record(example), indent=2, sort_keys=True) + "\n"
    assert_golden("dataset/conversation.json", rendered, update=update_golden)
