"""Thumbnail sizing, the fixed instruction and the deterministic golden record."""

from __future__ import annotations

import json

import numpy as np

from golden_support import assert_golden
from image_fixtures import make_image
from retone.data.examples import INSTRUCTION, Example, thumbnail
from retone.data.pairs import PairKind
from retone.executor import Params


def test_thumbnail_caps_the_long_edge() -> None:
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    assert thumbnail(image).shape == (448, 336, 3)


def test_thumbnail_never_upscales() -> None:
    image = make_image(40, 60)
    assert thumbnail(image).shape == image.shape


def test_instruction_golden(update_golden: bool) -> None:
    assert_golden("dataset/instruction.txt", INSTRUCTION + "\n", update=update_golden)


def test_record_golden(update_golden: bool) -> None:
    example = Example(
        example_id="photo-00-synthetic",
        split="train",
        image_file="images/photo-00-synthetic.jpg",
        params=Params(
            exposure=24,
            temperature=-8,
            tint=0,
            contrast=15,
            highlights=-20,
            shadows=10,
            saturation=5,
            vibrance=12,
        ),
        kind=PairKind.SYNTHETIC,
        source={
            "photo_id": "photo",
            "photographer": "Tester",
            "url": "https://example.org/photo",
            "license": "CC0-1.0",
        },
        degradation=Params(
            exposure=-24,
            temperature=8,
            contrast=-15,
            highlights=20,
            shadows=-10,
            saturation=-5,
            vibrance=-12,
        ),
        residual=0.0012,
    )
    rendered = json.dumps(example.record(), indent=2, sort_keys=True) + "\n"
    assert_golden("dataset/example_record.json", rendered, update=update_golden)
