"""Discover human-edited before/after pairs and copy them into a work folder."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from retone.data.errors import DatasetError

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"}
)
# Accepts "(edit)", "(edited)", "(edit_krita)", "(edited 2)", "(edit)2",
# "(edited)2", "(edited)(2)" and spaced variants of the numeric suffix.
_EDITED = re.compile(
    r"^(?P<base>.*)\((?P<kind>edit(?:ed)?)(?P<tags>[^()]*)\)"
    r"(?P<tail>[ _-]?\d+)?(?P<counter>[ _-]*\(\d+\))?$",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"^(?P<name>[a-z][a-z _-]*?)(?:[ _-]?(?P<number>\d+))?$")


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


@dataclass(frozen=True)
class CollectionSummary:
    pairs: int
    unmatched: int
    ambiguous: int
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


def _image_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith("."))
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(directory) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                found.append(path)
    return found


def _pick_before(candidates: list[Path], after: Path) -> tuple[Path | None, bool]:
    if not candidates:
        return None, False
    same_suffix = [path for path in candidates if path.suffix.lower() == after.suffix.lower()]
    chosen = sorted(same_suffix or candidates)[0]
    return chosen, len(candidates) > 1


def discover_pairs(root: Path) -> Discovery:
    """Pair every `name(edit...)` with the same folder's `name`, ignoring extension."""
    if not root.is_dir():
        raise DatasetError((f"source directory not found: {root}",))
    by_directory: dict[Path, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in _image_files(root):
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
    )


def copy_pairs(
    discovery: Discovery, destination: Path, *, dry_run: bool = False
) -> CollectionSummary:
    """Copy each pair into `destination/images` and write pairs.jsonl plus a summary."""
    by_variant: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    if not dry_run:
        (destination / "images").mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(discovery.pairs, start=1):
        pair_id = f"p{index:04d}"
        before_name = f"{pair_id}__before{pair.before.suffix.lower()}"
        after_name = f"{pair_id}__after{pair.after.suffix.lower()}"
        if not dry_run:
            shutil.copy2(pair.before, destination / "images" / before_name)
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
            "by_variant": by_variant,
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    return CollectionSummary(
        pairs=len(rows),
        unmatched=len(discovery.unmatched),
        ambiguous=len(discovery.ambiguous),
        by_variant=by_variant,
        destination=str(destination),
    )
