"""Pair-name parsing, recursive discovery and copying."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retone.data.collect import (
    EditedName,
    copy_pairs,
    discover_pairs,
    parse_edited,
)
from retone.data.errors import DatasetError


def _touch(path: Path, data: bytes = b"image") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("IMG_1234", None),
        ("IMG_1234(1)", None),
        ("IMG_1234(edited)", EditedName("IMG_1234", None, None)),
        ("IMG_1234(edit)", EditedName("IMG_1234", None, None)),
        ("IMG_1234(EDITED)", EditedName("IMG_1234", None, None)),
        ("IMG_1234(edited_krita)", EditedName("IMG_1234", "krita", None)),
        ("IMG_1234(edit-krita2)", EditedName("IMG_1234", "krita", 2)),
        ("IMG_1234(edited 2)", EditedName("IMG_1234", None, 2)),
        ("IMG_1234(edit)2", EditedName("IMG_1234", None, 2)),
        ("IMG_1234(edited)2", EditedName("IMG_1234", None, 2)),
        ("IMG_1234(edit)(2)", EditedName("IMG_1234", None, 2)),
        ("IMG_1234(edited) (3)", EditedName("IMG_1234", None, 3)),
        ("IMG_1234(edited_krita)2", EditedName("IMG_1234", "krita", 2)),
        ("foo(bar)(edited)", EditedName("foo(bar)", None, None)),
        ("(edited)", None),
    ],
)
def test_parse_edited(stem: str, expected: EditedName | None) -> None:
    assert parse_edited(stem) == expected


def _tree(root: Path) -> None:
    _touch(root / "a" / "IMG_1.JPG", b"before-a")
    _touch(root / "a" / "IMG_1(edited).JPG", b"edit-a")
    _touch(root / "a" / "IMG_1(edit)2.JPG", b"edit-a2")
    _touch(root / "a" / "IMG_1(edited_krita)2.JPG", b"krita-a2")
    _touch(root / "a" / "IMG_1(edited)(3).JPG", b"edit-a3")
    _touch(root / "b" / "DSC_2.jpg", b"before-b")
    _touch(root / "b" / "DSC_2(edited).jpeg", b"edit-b")
    _touch(root / "c" / "ORPHAN(edited).jpg")
    _touch(root / "d" / "notes.txt")


def test_discovery_finds_nested_pairs_and_variants(tmp_path: Path) -> None:
    _tree(tmp_path)
    discovery = discover_pairs(tmp_path)
    assert len(discovery.pairs) == 5
    assert len(discovery.unmatched) == 1
    variants = sorted(pair.edited.variant for pair in discovery.pairs)
    assert variants == ["edit", "edit", "edit v2", "edit v3", "krita v2"]
    assert all(pair.before.is_file() and pair.after.is_file() for pair in discovery.pairs)


def test_pairs_do_not_cross_directories(tmp_path: Path) -> None:
    _touch(tmp_path / "g" / "Z.jpg")
    _touch(tmp_path / "h" / "Z(edited).jpg")
    discovery = discover_pairs(tmp_path)
    assert not discovery.pairs
    assert len(discovery.unmatched) == 1


def test_multiple_originals_are_flagged(tmp_path: Path) -> None:
    _touch(tmp_path / "f" / "Y.jpg")
    _touch(tmp_path / "f" / "Y.png")
    _touch(tmp_path / "f" / "Y(edited).webp")
    discovery = discover_pairs(tmp_path)
    assert len(discovery.pairs) == 1
    assert len(discovery.ambiguous) == 1


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        discover_pairs(tmp_path / "absent")


def test_copy_writes_images_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    _tree(source)
    destination = tmp_path / "collected"
    summary = copy_pairs(discover_pairs(source), destination)

    assert summary.pairs == 5
    assert summary.unmatched == 1
    assert sum(summary.by_variant.values()) == 5
    rows = [json.loads(line) for line in (destination / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    first = rows[0]
    assert (destination / first["before"]).is_file()
    assert (destination / first["after"]).is_file()
    assert first["before_source"] == "a/IMG_1.JPG"
    assert (destination / "summary.json").is_file()


def test_dry_run_copies_nothing(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    _tree(source)
    destination = tmp_path / "nothing"
    summary = copy_pairs(discover_pairs(source), destination, dry_run=True)
    assert summary.pairs == 5
    assert not destination.exists()
