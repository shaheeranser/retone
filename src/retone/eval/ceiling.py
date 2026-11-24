"""Reconstruction ceiling: re-render test pairs at full resolution and measure ΔE."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retone.eval.delta_e import mean_delta_e
from retone.eval.records import params, photo_id
from retone.executor import ParamError, apply, load_image
from retone.executor.errors import ImageError


@dataclass(frozen=True)
class CeilingResult:
    count: int
    mean_delta_e: float
    per_image: tuple[float, ...]


def reconstruction_residual(
    records: Iterable[Mapping[str, Any]],
    path_for: Mapping[str, Path],
    *,
    split: str = "test",
    limit: int | None = None,
) -> CeilingResult:
    """Mean ΔE between `apply(input, label)` and the target, one full-resolution image at a time."""
    scores: list[float] = []
    for record in records:
        if record.get("split") != split:
            continue
        if limit is not None and len(scores) >= limit:
            break
        photo_id_value = photo_id(record)
        path = path_for.get(photo_id_value)
        if path is None:
            raise ImageError(f"no source path for photo {photo_id_value!r}")
        target = load_image(path)
        input_image = apply(target, params(record, "degradation"))
        corrected = apply(input_image, params(record, "params"))
        scores.append(mean_delta_e(corrected, target))
    if not scores:
        raise ImageError(f"no records for split {split!r}")
    return CeilingResult(
        count=len(scores),
        mean_delta_e=sum(scores) / len(scores),
        per_image=tuple(scores),
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ParamError((f"{path}: every line must be a JSON object",))
            rows.append(raw)
    return rows


def load_source_paths(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ParamError((f"{path}: expected an object mapping photo ids to paths",))
    return {str(photo_id): Path(str(value)) for photo_id, value in raw.items()}
