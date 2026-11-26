"""Check that fitting on a 448 px thumbnail matches fitting on the full image.

Both parameter sets are applied to the full-resolution image, so the reported
mean delta-E is render cost at full size and measures only the difference the
model's smaller view makes. The pass threshold is a reporting decision, not
enforced here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.eval.delta_e import mean_delta_e
from retone.executor import Params, apply
from retone.executor.errors import ImageError

Fit = Callable[[NDArray[np.integer[Any]]], Params]
MakeThumbnail = Callable[[NDArray[np.integer[Any]]], NDArray[np.integer[Any]]]


@dataclass(frozen=True)
class ThumbnailValidity:
    count: int
    mean_delta_e: float
    per_image: tuple[float, ...]


def thumbnail_validity(
    images: Iterable[NDArray[np.integer[Any]]],
    fit: Fit,
    make_thumbnail: MakeThumbnail,
) -> ThumbnailValidity:
    """For each image, compare the full-resolution fit against the thumbnail fit."""
    deltas: list[float] = []
    for image in images:
        from_full = apply(image, fit(image))
        from_thumbnail = apply(image, fit(make_thumbnail(image)))
        deltas.append(mean_delta_e(from_full, from_thumbnail))
    if not deltas:
        raise ImageError("thumbnail validity needs at least one image")
    return ThumbnailValidity(
        count=len(deltas),
        mean_delta_e=sum(deltas) / len(deltas),
        per_image=tuple(deltas),
    )
