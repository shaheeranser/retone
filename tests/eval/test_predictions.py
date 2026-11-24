"""Prediction scoring: parameter MAE, prediction files and full-resolution ΔE."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_fixtures import make_image, write_image
from retone.eval.__main__ import main as eval_main
from retone.eval.predictions import (
    load_predictions,
    parameter_mae,
    score_predictions,
)
from retone.executor import ParamError, Params


def test_parameter_mae_is_per_parameter() -> None:
    mae = parameter_mae(
        [Params(exposure=10, contrast=-4), Params(exposure=30, contrast=4)],
        [Params(exposure=0, contrast=0), Params(exposure=20, contrast=0)],
    )
    assert mae["exposure"] == 10.0
    assert mae["contrast"] == 4.0
    assert mae["sharpen_amount"] == 0.0


def test_parameter_mae_rejects_mismatched_sets() -> None:
    with pytest.raises(ValueError):
        parameter_mae([Params()], [])


def _record() -> dict[str, object]:
    return {
        "id": "photo-00",
        "split": "test",
        "source": {"photo_id": "photo"},
        "degradation": Params().as_dict(),
        "params": Params().as_dict(),
    }


def test_identity_prediction_scores_zero(tmp_path: Path) -> None:
    path = write_image(tmp_path / "photo.png", make_image(40, 50))
    score = score_predictions([_record()], {"photo-00": Params()}, {"photo": path})
    assert score.count == 1
    assert score.mean_delta_e == 0.0
    assert score.overall_mae == 0.0


def test_wrong_prediction_has_error_and_delta(tmp_path: Path) -> None:
    path = write_image(tmp_path / "photo.png", make_image(40, 50))
    score = score_predictions([_record()], {"photo-00": Params(exposure=10)}, {"photo": path})
    assert score.overall_mae > 0.0
    assert score.mean_delta_e > 0.0


def test_load_predictions_round_trip_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"id": "a", "params": Params(exposure=5).as_dict()}) + "\n")
    assert load_predictions(path) == {"a": Params(exposure=5)}

    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "params": {}}),
                json.dumps({"id": "a", "params": {}}),
            ]
        )
        + "\n"
    )
    with pytest.raises(ParamError):
        load_predictions(path)


def test_eval_cli_scores_prediction_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = write_image(tmp_path / "photo.png", make_image(40, 50))
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps(_record()) + "\n")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "photo-00", "params": Params().as_dict()}) + "\n")
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({"photo": str(image)}))

    code = eval_main(
        [
            "--examples",
            str(examples),
            "--source-paths",
            str(paths),
            "--predictions",
            str(predictions),
        ]
    )
    assert code == 0
    score = json.loads(capsys.readouterr().out)
    assert score["count"] == 1
    assert score["mean_delta_e"] == 0.0
