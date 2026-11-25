"""Human-pair loading, rendering and diagnostic scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_fixtures import make_image, write_image
from retone.eval.human import (
    load_human_pairs,
    load_predictions_by_label,
    render_predictions,
)
from retone.executor import ParamError, Params


def _pairs(tmp_path: Path, after_offset: int = 0) -> Path:
    before = make_image(40, 60, seed=1)
    after = np.clip(before.astype(np.int16) + after_offset, 0, 255).astype(np.uint8)
    write_image(tmp_path / "before" / "p0001.jpg", before)
    write_image(tmp_path / "after" / "p0001.jpg", after)
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "pair_id": "p0001",
                "role": "evaluation",
                "before": "before/p0001.jpg",
                "after": "after/p0001.jpg",
                "variant": "edit",
            }
        )
        + "\n"
    )
    return path


def test_load_human_pairs(tmp_path: Path) -> None:
    pairs = load_human_pairs(_pairs(tmp_path))
    assert len(pairs) == 1
    assert pairs[0].pair_id == "p0001"
    assert pairs[0].before == tmp_path / "before" / "p0001.jpg"
    assert pairs[0].variant == "edit"


def test_missing_or_duplicate_pairs_are_reported(tmp_path: Path) -> None:
    path = _pairs(tmp_path)
    row = json.loads(path.read_text())
    row["pair_id"] = "p0002"
    row["before"] = "before/absent.jpg"
    path.write_text(path.read_text() + json.dumps(row) + "\n")
    with pytest.raises(ParamError) as caught:
        load_human_pairs(path)
    assert len(caught.value.problems) == 1

    path = _pairs(tmp_path)
    path.write_text(path.read_text() + path.read_text())
    with pytest.raises(ParamError):
        load_human_pairs(path)


def test_identity_prediction_scores_zero(tmp_path: Path) -> None:
    pairs = load_human_pairs(_pairs(tmp_path))
    report = render_predictions(pairs, {"p0001": Params()}, tmp_path / "model")
    assert report.count == 1
    assert report.mean_delta_e == 0.0
    assert (tmp_path / "model" / "p0001.jpg").is_file()


def test_changed_prediction_has_a_delta(tmp_path: Path) -> None:
    pairs = load_human_pairs(_pairs(tmp_path, after_offset=20))
    report = render_predictions(pairs, {"p0001": Params(exposure=10)}, tmp_path / "model")
    assert report.mean_delta_e > 0.0


def test_missing_prediction_is_rejected(tmp_path: Path) -> None:
    pairs = load_human_pairs(_pairs(tmp_path))
    with pytest.raises(ParamError):
        render_predictions(pairs, {}, tmp_path / "model")


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _pairs(tmp_path)
    write_image(tmp_path / "after" / "p0001.jpg", make_image(50, 60, seed=2))
    pairs = load_human_pairs(path)
    with pytest.raises(ParamError):
        render_predictions(pairs, {"p0001": Params()}, tmp_path / "model")


def test_prediction_labels_parse_and_reject_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(json.dumps({"id": "p0001", "params": {}}) + "\n")
    labels = load_predictions_by_label([f"tuned={path}"])
    assert labels["tuned"]["p0001"] == Params()

    with pytest.raises(ParamError):
        load_predictions_by_label([f"tuned={path}", f"tuned={path}"])
    with pytest.raises(ParamError):
        load_predictions_by_label(["not-a-label"])
