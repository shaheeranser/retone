"""Training examples: the fixed instruction, the 448 px thumbnail and the JSON record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.data.pairs import PairKind
from retone.executor import Params

IntImage = NDArray[np.integer[Any]]

THUMBNAIL_LONG_EDGE = 448

INSTRUCTION = (
    "You are a photo retoucher. Look at the photo and choose adjustment values that "
    "make it look better without changing its content. Reply with one JSON object whose "
    "keys are exposure, temperature, tint, contrast, highlights, shadows, saturation, "
    "vibrance and sharpen_amount. Each value is an integer from -100 to 100, except "
    "sharpen_amount which is 0 to 100. Zero means no change."
)


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    image_file: str
    params: Params
    kind: PairKind
    source: dict[str, Any]
    degradation: Params
    residual: float

    def record(self) -> dict[str, Any]:
        """A JSON-serialisable row; key order is fixed so the golden file stays stable."""
        return {
            "id": self.example_id,
            "split": self.split,
            "image": self.image_file,
            "instruction": INSTRUCTION,
            "answer": json.dumps(self.params.as_dict(), ensure_ascii=False),
            "params": self.params.as_dict(),
            "kind": str(self.kind),
            "residual": self.residual,
            "source": dict(self.source),
            "degradation": self.degradation.as_dict(),
        }


def thumbnail(image: IntImage, long_edge: int = THUMBNAIL_LONG_EDGE) -> IntImage:
    """Downscale so the longer side is `long_edge`, never upscaling, preserving dtype."""
    height, width = image.shape[:2]
    scale = min(1.0, long_edge / max(height, width))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cast(IntImage, cv2.resize(image, size, interpolation=cv2.INTER_AREA))
