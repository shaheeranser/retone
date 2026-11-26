"""The seam between the pipeline and the model: hand a thumbnail, get raw JSON back."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from retone.inference.errors import ModelUnavailableError


@runtime_checkable
class InferenceClient(Protocol):
    """Chooses adjustment values for one thumbnail.

    Returns the model's answer as a raw mapping without trusting its shape; the
    caller validates it against the schema. Raises `ModelUnavailable` when no
    answer can be obtained.
    """

    def choose_params(self, thumbnail_rgb: NDArray[np.uint8]) -> Mapping[str, object]: ...


__all__ = ["InferenceClient", "ModelUnavailableError"]
