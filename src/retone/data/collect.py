"""Discover human-edited before/after pairs, orient them, and copy them."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.data.errors import DatasetError
from retone.executor import ImageError, load_image, save_image

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}
)
DEFAULT_EXCLUDES = ("_*",)
# Accepts "(edit)", "(edited)", "(edit_krita)", "(edited 2)", "(edit)2",
# "(edited)2", "(edited)(2)" and spaced variants of the numeric suffix.
_EDITED = re.compile(
    r"^(?P<base>.*)\((?P<kind>edit(?:ed)?)(?P<tags>[^()]*)\)"
    r"(?P<tail>[ _-]?\d+)?(?P<counter>[ _-]*\(\d+\))?$",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"^(?P<name>[a-z][a-z _-]*?)(?:[ _-]?(?P<number>\d+))?$")
_FEATURE_SIZE = 64


@dataclass(frozen=True)
class EditedName:
    base: str
    software: str | None
    version: int | None

    @property
    def variant(self) -> str:
        label = self.software or "edit"
        return f"{label} v{self.version}" if self.version is not None else label


@dataclass(frozen=True)
class DiscoveredPair:
    before: Path
    after: Path
    edited: EditedName


@dataclass(frozen=True)
class Discovery:
    root: Path
    pairs: tuple[DiscoveredPair, ...]
    unmatched: tuple[Path, ...]
    ambiguous: tuple[Path, ...]
    skipped: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Orientation:
    rotation: int
    before_size: tuple[int, int] | None
    after_size: tuple[int, int] | None
    skipped: bool = False
    unreadable: bool = False


@dataclass(frozen=True)
class CollectionSummary:
    pairs: int
    unmatched: int
    ambiguous: int
    rotated: int
    rotations: dict[str, int]
    orientation_skipped: int
    unreadable: int
    unreadable_files: tuple[str, ...]
    skipped_directories: int
    by_variant: dict[str, int]
    destination: str


def parse_edited(stem: str) -> EditedName | None:
    """Return the base name and edit variant for `name(edit...)`, else None."""
    match = _EDITED.match(stem)
    if match is None:
        return None
    base = match.group("base").rstrip()
    if not base:
        return None
    software, tag_version = _parse_tags(match.group("tags"))
    version = _pick_version(match.group("tail"), match.group("counter"), tag_version)
    return EditedName(base=base, software=software, version=version)


def _parse_tags(tags: str) -> tuple[str | None, int | None]:
    text = tags.strip().lower().strip("_- ")
    if not text:
        return None, None
    if text.isdigit():
        return None, int(text)
    match = _TAG.match(text)
    if match is None:
        return text, None
    name = re.sub(r"[ _-]+", "_", match.group("name").strip(" _-")) or None
    number = match.group("number")
    return name, int(number) if number is not None else None


def _pick_version(tail: str | None, counter: str | None, tag_version: int | None) -> int | None:
    if counter is not None:
        return int(counter.strip(" ()"))
    if tail is not None:
        return int(re.sub(r"[ _-]", "", tail))
    return tag_version


def _image_files(root: Path, exclude: Sequence[str]) -> tuple[list[Path], list[Path]]:
    found: list[Path] = []
    skipped: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        kept: list[str] = []
        for name in sorted(subdirectories):
            if name.startswith(".") or any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
                skipped.append(Path(directory) / name)
            else:
                kept.append(name)
        subdirectories[:] = kept
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(directory) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                found.append(path)
    return found, skipped


def _pick_before(candidates: list[Path], after: Path) -> tuple[Path | None, bool]:
    if not candidates:
        return None, False
    same_suffix = [path for path in candidates if path.suffix.lower() == after.suffix.lower()]
    chosen = sorted(same_suffix or candidates)[0]
    return chosen, len(candidates) > 1


def discover_pairs(root: Path, *, exclude: Sequence[str] = DEFAULT_EXCLUDES) -> Discovery:
    """Pair every `name(edit...)` with the same folder's `name`, ignoring extension."""
    if not root.is_dir():
        raise DatasetError((f"source directory not found: {root}",))
    files, skipped = _image_files(root, exclude)
    by_directory: dict[Path, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        by_directory[path.parent][path.stem].append(path)

    pairs: list[DiscoveredPair] = []
    unmatched: list[Path] = []
    ambiguous: list[Path] = []
    for directory in sorted(by_directory):
        stems = by_directory[directory]
        for stem in sorted(stems):
            edited = parse_edited(stem)
            if edited is None:
                continue
            candidates = [
                path
                for key in (edited.base, edited.base + " ")
                for path in stems.get(key, [])
                if parse_edited(path.stem) is None
            ]
            for after in stems[stem]:
                before, is_ambiguous = _pick_before(candidates, after)
                if before is None:
                    unmatched.append(after)
                    continue
                if is_ambiguous:
                    ambiguous.append(after)
                pairs.append(DiscoveredPair(before=before, after=after, edited=edited))
    pairs.sort(key=lambda pair: (pair.before.relative_to(root), pair.after.relative_to(root)))
    return Discovery(
        root=root,
        pairs=tuple(pairs),
        unmatched=tuple(unmatched),
        ambiguous=tuple(ambiguous),
        skipped=tuple(skipped),
    )


def _shape_sign(image: NDArray[Any]) -> int:
    height, width = image.shape[:2]
    return 1 if width > height else -1 if height > width else 0


def _structure_features(image: NDArray[Any]) -> NDArray[np.float32]:
    """Small gradient-magnitude map, standardised so tone edits do not dominate."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(grey, (_FEATURE_SIZE, _FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    grad_x = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    spread = float(magnitude.std())
    return (magnitude - float(magnitude.mean())) / (spread if spread > 1e-6 else 1.0)


def _orientation(before: NDArray[Any], after: NDArray[Any]) -> Orientation:
    """Match the edited image to the original's orientation, choosing 90 or 270 by structure."""
    before_size = (int(before.shape[1]), int(before.shape[0]))
    after_size = (int(after.shape[1]), int(after.shape[0]))
    before_sign, after_sign = _shape_sign(before), _shape_sign(after)
    if before_sign == 0 or after_sign == 0 or before_sign == after_sign:
        return Orientation(0, before_size, after_size, skipped=before_sign != after_sign)

    candidates = {
        90: cv2.rotate(after, cv2.ROTATE_90_CLOCKWISE),
        270: cv2.rotate(after, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    reference = _structure_features(before)
    rotation = min(
        candidates,
        key=lambda angle: float(np.mean((reference - _structure_features(candidates[angle])) ** 2)),
    )
    return Orientation(rotation, before_size, after_size)


def _orient_pair(pair: DiscoveredPair) -> tuple[Orientation, NDArray[Any] | None]:
    try:
        before = load_image(pair.before)
        after = load_image(pair.after)
    except ImageError:
        return Orientation(0, None, None, unreadable=True), None
    orientation = _orientation(before, after)
    if orientation.rotation == 0:
        return orientation, None
    rotated = cv2.rotate(
        after,
        cv2.ROTATE_90_CLOCKWISE if orientation.rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE,
    )
    return orientation, rotated


def copy_pairs(
    discovery: Discovery, destination: Path, *, dry_run: bool = False
) -> CollectionSummary:
    """Orient, copy and describe every pair; the original is always the reference."""
    by_variant: dict[str, int] = {}
    rotations = {"90": 0, "270": 0}
    unreadable_files: list[str] = []
    rows: list[dict[str, object]] = []
    rotated = 0
    skipped = 0
    unreadable = 0
    if not dry_run:
        (destination / "images").mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(discovery.pairs, start=1):
        pair_id = f"p{index:04d}"
        orientation, rotated_after = _orient_pair(pair)
        before_name = f"{pair_id}__before{pair.before.suffix.lower()}"
        if orientation.rotation:
            after_name = f"{pair_id}__after.png"
            rotated += 1
            rotations[str(orientation.rotation)] += 1
        else:
            after_name = f"{pair_id}__after{pair.after.suffix.lower()}"
        if orientation.skipped:
            skipped += 1
        if orientation.unreadable:
            unreadable += 1
            unreadable_files.append(str(pair.before.relative_to(discovery.root)))

        if not dry_run:
            shutil.copy2(pair.before, destination / "images" / before_name)
            if rotated_after is not None:
                save_image(destination / "images" / after_name, rotated_after)
            else:
                shutil.copy2(pair.after, destination / "images" / after_name)
        rows.append(
            {
                "pair_id": pair_id,
                "before": f"images/{before_name}",
                "after": f"images/{after_name}",
                "before_source": str(pair.before.relative_to(discovery.root)),
                "after_source": str(pair.after.relative_to(discovery.root)),
                "variant": pair.edited.variant,
                "software": pair.edited.software,
                "version": pair.edited.version,
                "rotation_after": orientation.rotation,
                "before_size": orientation.before_size,
                "after_size": orientation.after_size,
            }
        )
        by_variant[pair.edited.variant] = by_variant.get(pair.edited.variant, 0) + 1

    if not dry_run:
        with (destination / "pairs.jsonl").open("w", encoding="utf-8") as manifest:
            for row in rows:
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": str(discovery.root),
            "pairs": len(rows),
            "unmatched": [str(path.relative_to(discovery.root)) for path in discovery.unmatched],
            "ambiguous": [str(path.relative_to(discovery.root)) for path in discovery.ambiguous],
            "skipped_directories": [
                str(path.relative_to(discovery.root)) for path in discovery.skipped
            ],
            "rotated": rotated,
            "rotations": rotations,
            "orientation_skipped": skipped,
            "unreadable": unreadable,
            "unreadable_files": unreadable_files,
            "by_variant": by_variant,
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    return CollectionSummary(
        pairs=len(rows),
        unmatched=len(discovery.unmatched),
        ambiguous=len(discovery.ambiguous),
        rotated=rotated,
        rotations=rotations,
        orientation_skipped=skipped,
        unreadable=unreadable,
        unreadable_files=tuple(unreadable_files),
        skipped_directories=len(discovery.skipped),
        by_variant=by_variant,
        destination=str(destination),
    )
