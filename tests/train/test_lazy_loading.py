"""Lazy image loading, proven on a fake Hub dataset.

No Hub download and no real images: the fakes count decodes and record the
subset passed to `map`, so the test can assert that loading decodes nothing, the
build is deferred to a streaming `map` over a memmapped view, and the labels
come from the (possibly coarsened) records rather than the raw rows.
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


class FakeView:
    """A `datasets.Dataset`-like view: __getitem__ decodes one image."""

    def __init__(self, rows: list[dict[str, Any]], counter: dict[str, int]) -> None:
        self.rows = rows
        self.counter = counter
        self.column_names = ["id", "split", "instruction", "answer", "kind", "image"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        self.counter["decodes"] += 1
        return {**self.rows[index], "image": FakeImage(index)}


class FakeHubDataset:
    def __init__(self) -> None:
        self.counter = {"decodes": 0, "mapped": 0, "selected": 0}
        self.rows = [
            {
                "id": f"p{i}",
                "split": "test" if i % 10 == 0 else "train",
                "instruction": "Make it better.",
                "answer": json.dumps(Params(exposure=10, contrast=5).as_dict()),
                "kind": "synthetic",
            }
            for i in range(600)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        return FakeView(self.rows, self.counter)[index]

    def remove_columns(self, names: str | list[str]) -> FakeRows:
        return FakeRows(self.rows, self.counter)

    def select(self, indices: list[int]) -> FakeSelected:
        self.counter["selected"] += 1
        return FakeSelected([self.rows[i] for i in indices], self.counter)

    def map(self, fn: Any, **kwargs: Any) -> FakeMapped:
        self.counter["mapped"] += 1
        rows = [fn(row, i) for i, row in enumerate(self.rows)]
        return FakeMapped(rows)


class FakeRows:
    """What `remove_columns("image")` returns: rows without images."""

    def __init__(self, rows: list[dict[str, Any]], counter: dict[str, int]) -> None:
        self.rows = rows
        self.counter = counter

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        return self.rows[index]


class FakeSelected(FakeView):
    """A memory-mapped subset: indexing it decodes just that row, and `map`
    streams over it exactly as datasets.Dataset does."""

    def __init__(self, rows: list[dict[str, Any]], counter: dict[str, int]) -> None:
        super().__init__(rows, counter)
        self.column_names = ["id", "split", "instruction", "answer", "kind", "image"]

    def map(self, fn: Any, **kwargs: Any) -> FakeMapped:
        self.counter["mapped"] += 1
        return FakeMapped([fn(self[i], i) for i in range(len(self))])


class FakeMapped:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def column_names(self) -> list[str]:
        return ["messages"]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHubDataset:
    dataset = FakeHubDataset()
    module = types.ModuleType("datasets")

    def fake_load_dataset(repo: str, **kwargs: Any) -> dict[str, FakeHubDataset]:
        return {"train": dataset}

    module.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    module.Dataset = types.SimpleNamespace(from_list=lambda rows: FakeMapped(rows))
    # `datasets` is a train-extra dependency, absent in CI's `.[dev]` install,
    # so inject a stub instead of importing the real package.
    monkeypatch.setitem(sys.modules, "datasets", module)
    return dataset


def _settings(tmp_path: Path) -> qlora.TrainSettings:
    return qlora.TrainSettings(
        model_name="fake", dataset_repo="fake/dataset", output_repo="fake/out", output_dir=tmp_path
    )


def test_loading_decodes_no_images(hub: FakeHubDataset, tmp_path: Path) -> None:
    """The whole point: no image is decoded at load time."""
    train, test = qlora.load_examples_from_hub(_settings(tmp_path), "token", select_train=True)
    assert hub.counter["decodes"] == 0
    assert train and test


def test_build_uses_a_streaming_map_not_a_list(hub: FakeHubDataset, tmp_path: Path) -> None:
    train, _ = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    built = qlora._as_vision_records(train)
    assert hub.counter["selected"] == 1  # memmapped subset, not a decoded copy
    assert hub.counter["mapped"] == 1  # one streaming pass writes the Arrow file
    assert built.column_names == ["messages"]
    assert len(built) == len(train)


def test_labels_come_from_records_not_rows(hub: FakeHubDataset, tmp_path: Path) -> None:
    """A retry coarsens the label; the dataset must carry the coarsened value."""
    train, _ = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    coarsened = [record.coarsened(50) for record in train]
    built = qlora._as_vision_records(coarsened)
    first = built.rows[0]["messages"][-1]["content"][0]["text"]
    assert json.loads(first)["exposure"] == 0  # 10 rounded to the step-50 grid


def test_split_is_respected(hub: FakeHubDataset, tmp_path: Path) -> None:
    train, test = qlora.load_examples_from_hub(_settings(tmp_path), "token")
    assert len(test) == 60  # every tenth of 600 rows
    assert len(train) == 540


def test_rank_guards_default_to_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert qlora.is_main() is True
