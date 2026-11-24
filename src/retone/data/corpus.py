"""The Unsplash source manifest: provenance per photo and a deterministic split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retone.data.errors import DatasetError

MANIFEST_FIELDS = ("photo_id", "file", "photographer", "url", "license")
UNSPLASH_LICENSE = "Unsplash License"


@dataclass(frozen=True)
class SourceImage:
    photo_id: str
    path: Path
    photographer: str
    source_url: str
    license: str


def _parse_entry(raw: object, base_dir: Path, index: int) -> SourceImage:
    if not isinstance(raw, dict):
        raise DatasetError((f"line {index}: expected a JSON object",))
    missing = [field for field in MANIFEST_FIELDS if field not in raw]
    if missing:
        raise DatasetError((f"line {index}: missing fields {missing}",))
    values = {field: raw[field] for field in MANIFEST_FIELDS}
    if not all(isinstance(value, str) and value for value in values.values()):
        raise DatasetError((f"line {index}: every field must be a non-empty string",))
    path = Path(values["file"])
    return SourceImage(
        photo_id=values["photo_id"],
        path=path if path.is_absolute() else base_dir / path,
        photographer=values["photographer"],
        source_url=values["url"],
        license=values["license"],
    )


def load_manifest(path: Path) -> tuple[SourceImage, ...]:
    """Read a JSONL source manifest and reject duplicates or malformed entries together."""
    entries: list[SourceImage] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as err:
            problems.append(f"line {index}: invalid JSON ({err.msg})")
            continue
        try:
            entry = _parse_entry(raw, path.parent, index)
        except DatasetError as err:
            problems.extend(err.problems)
            continue
        if entry.photo_id in seen:
            problems.append(f"line {index}: duplicate photo_id {entry.photo_id!r}")
            continue
        if not entry.path.is_file():
            problems.append(f"line {index}: image not found at {entry.path}")
            continue
        seen.add(entry.photo_id)
        entries.append(entry)
    if problems:
        raise DatasetError(tuple(problems))
    if not entries:
        raise DatasetError((f"manifest {path} contains no photos",))
    return tuple(entries)


def derive_seed(seed: int, photo_id: str) -> int:
    """A per-photo seed independent of order, worker count and platform hash randomisation."""
    digest = hashlib.sha256(f"{seed}:{photo_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_items(
    items: tuple[SourceImage, ...], *, seed: int, test_fraction: float
) -> tuple[tuple[SourceImage, ...], tuple[SourceImage, ...]]:
    """Partition photos into (train, test) by a hash, so file order cannot change membership."""
    if not 0.0 < test_fraction < 1.0:
        raise DatasetError((f"test_fraction must be in (0, 1), got {test_fraction}",))
    train: list[SourceImage] = []
    test: list[SourceImage] = []
    for item in items:
        bucket = derive_seed(seed, item.photo_id) / 2**64
        (test if bucket < test_fraction else train).append(item)
    return tuple(train), tuple(test)


def source_metadata(item: SourceImage) -> dict[str, Any]:
    return {
        "photo_id": item.photo_id,
        "photographer": item.photographer,
        "url": item.source_url,
        "license": item.license,
    }
