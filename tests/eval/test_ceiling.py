"""The full-resolution reconstruction ceiling and its manifest loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_fixtures import make_image, write_image
from retone.data.degradation import invert
from retone.eval.ceiling import (
    load_records,
    load_source_paths,
    reconstruction_residual,
)
from retone.executor import Params
from retone.executor.errors import ImageError


def _record(photo_id: str, degradation: Params, params: Params) -> dict[str, object]:
    return {
        "split": "test",
        "source": {"photo_id": photo_id},
        "degradation": degradation.as_dict(),
        "params": params.as_dict(),
    }


def test_identity_pair_has_zero_residual(tmp_path: Path) -> None:
    image = make_image(40, 50)
    path = write_image(tmp_path / "photo.png", image)
    record = _record("photo", Params(), Params())
    result = reconstruction_residual([record], {"photo": path})
    assert result.count == 1
    assert result.mean_delta_e == 0.0


def test_degraded_pair_reports_a_measured_residual(tmp_path: Path) -> None:
    image = make_image(40, 50)
    path = write_image(tmp_path / "photo.png", image)
    degradation = Params(exposure=12, contrast=6)
    record = _record("photo", degradation, invert(degradation))
    result = reconstruction_residual([record], {"photo": path})
    assert result.count == 1
    assert result.mean_delta_e >= 0.0


def test_missing_split_is_an_error(tmp_path: Path) -> None:
    image = make_image(8, 8)
    path = write_image(tmp_path / "photo.png", image)
    with pytest.raises(ImageError):
        reconstruction_residual(
            [_record("photo", Params(), Params())], {"photo": path}, split="train"
        )


def test_missing_source_path_is_an_error() -> None:
    with pytest.raises(ImageError):
        reconstruction_residual([_record("photo", Params(), Params())], {})


def test_manifest_loaders_round_trip(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps(_record("photo", Params(), Params())) + "\n")
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({"photo": "/tmp/photo.png"}))
    assert load_records(examples)[0]["split"] == "test"
    assert load_source_paths(paths) == {"photo": Path("/tmp/photo.png")}
