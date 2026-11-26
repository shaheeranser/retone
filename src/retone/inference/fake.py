"""A deterministic fake client so the pipeline and CLI test without a real model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


class FakeClient:
    """Returns a fixed mapping, or raises whatever it was constructed with."""

    def __init__(self, answer: Mapping[str, object] | None = None) -> None:
        self._answer: Mapping[str, object] = dict(answer or {})
        self.calls = 0

    def choose_params(self, thumbnail_rgb: NDArray[np.uint8]) -> Mapping[str, object]:
        del thumbnail_rgb
        self.calls += 1
        return dict(self._answer)


class RaisingClient:
    """A client that always fails, to exercise the model-unavailable path."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def choose_params(self, thumbnail_rgb: NDArray[np.uint8]) -> Mapping[str, object]:
        del thumbnail_rgb
        raise self._error
