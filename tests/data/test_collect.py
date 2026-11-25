"""Pair-name parsing, recursive discovery, orientation matching and copying."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_fixtures import make_image, write_image
from retone.data.collect import (
    EditedName,
    copy_pairs,
    discover_pairs,
    parse_edited,
)
from retone.data.errors import DatasetError
from retone.executor import load_image


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
    write_image(root / "a" / "IMG_1.jpg", make_image(24, 32, seed=1))
    write_image(root / "a" / "IMG_1(edited).jpg", make_image(24, 32, seed=2))
    write_image(root / "a" / "IMG_1(edit)2.jpg", make_image(24, 32, seed=3))
    write_image(root / "a" / "IMG_1(edited_krita)2.jpg", make_image(24, 32, seed=4))
    write_image(root / "a" / "IMG_1(edited)(3).jpg", make_image(24, 32, seed=5))
    write_image(root / "b" / "DSC_2.jpg", make_image(24, 32, seed=6))
    write_image(root / "b" / "DSC_2(edited).jpeg", make_image(24, 32, seed=7))
    write_image(root / "c" / "ORPHAN(edited).jpg", make_image(24, 32, seed=8))
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
    assert summary.rotated == 0
    assert summary.rotations == {"90": 0, "270": 0}
    assert summary.skipped_directories == 0
    assert sum(summary.by_variant.values()) == 5
    rows = [json.loads(line) for line in (destination / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    first = rows[0]
    assert (destination / first["before"]).is_file()
    assert (destination / first["after"]).is_file()
    assert first["before_source"] == "a/IMG_1.jpg"
    assert first["rotation_after"] == 0
    assert first["before_size"] == [32, 24]
    assert (destination / "summary.json").is_file()


def test_dry_run_copies_nothing(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    _tree(source)
    destination = tmp_path / "nothing"
    summary = copy_pairs(discover_pairs(source), destination, dry_run=True)
    assert summary.pairs == 5
    assert not destination.exists()


def test_rotated_edit_is_restored_to_the_original_orientation(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    source.mkdir()
    before = make_image(40, 60, seed=11)
    after = cv2.rotate(before, cv2.ROTATE_90_CLOCKWISE)
    write_image(source / "IMG_9.png", before)
    write_image(source / "IMG_9(edited).png", after)

    destination = tmp_path / "out"
    summary = copy_pairs(discover_pairs(source), destination)
    assert summary.rotated == 1
    assert summary.rotations["90"] + summary.rotations["270"] == 1

    row = json.loads((destination / "pairs.jsonl").read_text().splitlines()[0])
    assert row["rotation_after"] in (90, 270)
    corrected = load_image(destination / row["after"])
    assert corrected.shape == before.shape
    assert np.array_equal(corrected, before)


def test_matching_orientation_is_copied_untouched(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    source.mkdir()
    before = make_image(40, 60, seed=12)
    after = np.clip(before.astype(np.int16) + 15, 0, 255).astype(np.uint8)
    write_image(source / "IMG_9.png", before)
    write_image(source / "IMG_9(edited).png", after)

    destination = tmp_path / "out"
    summary = copy_pairs(discover_pairs(source), destination)
    assert summary.rotated == 0
    row = json.loads((destination / "pairs.jsonl").read_text().splitlines()[0])
    assert row["after"] == "images/p0001__after.png"
    assert load_image(destination / row["after"]).shape == before.shape


def test_square_original_skips_orientation_matching(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    source.mkdir()
    write_image(source / "SQ.jpg", make_image(40, 40, seed=13))
    write_image(source / "SQ(edited).jpg", make_image(50, 40, seed=14))

    summary = copy_pairs(discover_pairs(source), tmp_path / "out")
    assert summary.rotated == 0
    assert summary.orientation_skipped == 1


def test_unreadable_images_are_copied_and_counted(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    source.mkdir()
    _touch(source / "BAD.heic", b"not a decodable image")
    _touch(source / "BAD(edited).heic", b"also not decodable")

    destination = tmp_path / "out"
    summary = copy_pairs(discover_pairs(source), destination)
    assert summary.unreadable == 1
    assert summary.unreadable_files == ("BAD.heic",)
    assert (destination / "images" / "p0001__before.heic").is_file()
    assert (destination / "images" / "p0001__after.heic").is_file()


def test_underscore_directories_are_skipped_by_default(tmp_path: Path) -> None:
    source = tmp_path / "pictures"
    write_image(source / "_corrupt" / "BAD.jpg", make_image(16, 20, seed=15))
    write_image(source / "_corrupt" / "BAD(edited).jpg", make_image(16, 20, seed=16))
    write_image(source / "good" / "OK.jpg", make_image(16, 20, seed=17))
    write_image(source / "good" / "OK(edited).jpg", make_image(16, 20, seed=18))

    default = discover_pairs(source)
    assert len(default.pairs) == 1
    assert len(default.skipped) == 1

    included = discover_pairs(source, exclude=())
    assert len(included.pairs) == 2
    assert not included.skipped
