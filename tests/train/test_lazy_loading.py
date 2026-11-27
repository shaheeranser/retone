"""Lazy image loading and rank guards, proven on a fake in-memory dataset.

No Hub download and no real images: the dataset object counts decodes so the
test can assert that construction decodes nothing and one sample decodes one.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from retone.executor import Params
from retone.train import qlora


class FakeImage:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.size = (448, 336)

    def convert(self, mode: str) -> FakeImage:
        return self


class FakeRows:
    """What `remove_columns("image")` returns: the same rows without images."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return self.rows[index]
        return self.rows[index]


class FakeDataset:
    """Stands in for a datasets.Dataset: decodes an image per __getitem__."""

    def __init__(self) -> None:
        self.decodes = 0
        self.rows = [
            {
                "id": f"p{i}",
                "split": "test" if i % 10 == 0 else "train",
                "instruction": "Make it better.",
                "answer": json.dumps(Params(exposure=10, contrast=5).as_dict()),
                "kind": "synthetic",
            }
            for i in range(30)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        self.decodes += 1
        row = self.rows[index]
        return {**row, "image": FakeImage(index)}

    def remove_columns(self, names: str | list[str]) -> FakeRows:
        # Mirrors datasets.Dataset: the returned view has no image column, so
        # reading it never decodes an image.
        return FakeRows(self.rows)


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> FakeDataset:
    dataset = FakeDataset()
    module = types.ModuleType("datasets")

    def fake_load_dataset(repo: str, **kwargs: Any) -> dict[str, FakeDataset]:
        return {"train": dataset}

    module.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    # `datasets` is a train-extra dependency, absent in CI's `.[dev]` install,
    # so inject a stub instead of importing the real package.
    monkeypatch.setitem(sys.modules, "datasets", module)
    return dataset


def _settings(tmp_path: Path) -> qlora.TrainSettings:
    return qlora.TrainSettings(
        model_name="fake", dataset_repo="fake/dataset", output_repo="fake/out", output_dir=tmp_path
    )


def test_loading_decodes_no_images(fake_loader: FakeDataset, tmp_path: Path) -> None:
    """The whole point: no image is decoded until a sample is requested."""
    train, test = qlora.load_examples_from_hub(_settings(tmp_path), "token", select_train=True)
    assert fake_loader.decodes == 0
    assert train and test


def test_indexing_decodes_exactly_one_image(fake_loader: FakeDataset, tmp_path: Path) -> None:
    train, _ = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    records = qlora._as_vision_records(train)
    assert fake_loader.decodes == 0
    sample = records[0]
    assert fake_loader.decodes == 1
    assert sample["messages"][0]["content"][1]["type"] == "image"


def test_records_support_len_and_slice(fake_loader: FakeDataset, tmp_path: Path) -> None:
    train, _ = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    records = qlora._as_vision_records(train)
    assert len(records) == len(train)
    assert records.column_names == ["messages"]
    sliced = records[:2]
    assert len(sliced) == 2
    assert fake_loader.decodes == 2


def test_split_is_respected(fake_loader: FakeDataset, tmp_path: Path) -> None:
    train, test = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    assert all("p" in example.example_id for example in train)
    assert len(test) == 3  # every tenth of 30 rows


def test_rank_guards_default_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert qlora.is_main() is True


def test_select_and_shuffle_decode_nothing(fake_loader: FakeDataset, tmp_path: Path) -> None:
    train, _ = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    records = qlora._as_vision_records(train)
    subset = records.select([0, 2])
    shuffled = records.shuffle(seed=1)
    assert fake_loader.decodes == 0
    assert len(subset) == 2
    assert len(shuffled) == len(records)
    shuffled[0]
    assert fake_loader.decodes == 1
