"""Manifest parsing, provenance and the deterministic split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_fixtures import make_image, write_image
from retone.data.corpus import derive_seed, load_manifest, split_items
from retone.data.errors import DatasetError


def _manifest(tmp_path: Path, ids: list[str]) -> Path:
    lines: list[str] = []
    for photo_id in ids:
        filename = f"{photo_id}.png"
        write_image(tmp_path / filename, make_image(seed=len(photo_id)))
        lines.append(
            json.dumps(
                {
                    "photo_id": photo_id,
                    "file": filename,
                    "photographer": "Tester",
                    "url": f"https://example.org/{photo_id}",
                    "license": "CC0-1.0",
                }
            )
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = _manifest(tmp_path, ["one", "two"])
    items = load_manifest(path)
    assert [item.photo_id for item in items] == ["one", "two"]
    assert items[0].path == tmp_path / "one.png"
    assert items[0].license == "CC0-1.0"


def test_bad_entries_are_reported_together(tmp_path: Path) -> None:
    write_image(tmp_path / "ok.png", make_image())
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "photo_id": "a",
                        "file": "ok.png",
                        "photographer": "T",
                        "url": "u",
                        "license": "l",
                    }
                ),
                json.dumps(
                    {
                        "photo_id": "b",
                        "file": "missing.png",
                        "photographer": "T",
                        "url": "u",
                        "license": "l",
                    }
                ),
                json.dumps(
                    {
                        "photo_id": "a",
                        "file": "ok.png",
                        "photographer": "T",
                        "url": "u",
                        "license": "l",
                    }
                ),
                json.dumps({"photo_id": "c", "file": "ok.png"}),
            ]
        )
        + "\n"
    )
    with pytest.raises(DatasetError) as caught:
        load_manifest(path)
    assert len(caught.value.problems) == 3


def test_split_is_independent_of_order(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [f"id{index}" for index in range(40)])
    items = load_manifest(path)
    train, test = split_items(items, seed=7, test_fraction=0.25)
    shuffled = tuple(reversed(items))
    train_again, test_again = split_items(shuffled, seed=7, test_fraction=0.25)
    assert {item.photo_id for item in train} == {item.photo_id for item in train_again}
    assert {item.photo_id for item in test} == {item.photo_id for item in test_again}
    assert len(train) + len(test) == 40


def test_split_fraction_must_be_a_fraction() -> None:
    with pytest.raises(DatasetError):
        split_items((), seed=0, test_fraction=1.0)


def test_seed_is_stable_and_photo_specific() -> None:
    assert derive_seed(3, "abc") == derive_seed(3, "abc")
    assert derive_seed(3, "abc") != derive_seed(4, "abc")
    assert derive_seed(3, "abc") != derive_seed(3, "abd")
