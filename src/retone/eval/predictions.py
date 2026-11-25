"""Score predicted parameters: full-resolution ΔE and per-parameter error."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retone.eval.delta_e import mean_delta_e
from retone.eval.records import field, params, photo_id
from retone.eval.similarity import psnr, ssim
from retone.executor import SCHEMA, ParamError, Params, apply, load_image, parse_params
from retone.executor.errors import ImageError


@dataclass(frozen=True)
class PredictionScore:
    count: int
    mean_delta_e: float
    parameter_mae: dict[str, float]
    overall_mae: float
    mean_psnr_db: float
    mean_ssim: float
    baseline_delta_e: float
    baseline_psnr_db: float
    baseline_ssim: float


def parameter_mae(predicted: Sequence[Params], expected: Sequence[Params]) -> dict[str, float]:
    """Mean absolute error per parameter between two equally sized prediction sets."""
    if not predicted or len(predicted) != len(expected):
        raise ValueError(
            f"need equally sized, non-empty sets, got {len(predicted)} and {len(expected)}"
        )
    return {
        spec.name: sum(
            abs(getattr(guess, spec.name) - getattr(truth, spec.name))
            for guess, truth in zip(predicted, expected, strict=True)
        )
        / len(predicted)
        for spec in SCHEMA
    }


def load_predictions(path: Path) -> dict[str, Params]:
    predictions: dict[str, Params] = {}
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as err:
            raise ParamError((f"line {index}: invalid JSON ({err.msg})",)) from err
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise ParamError((f"line {index}: expected an object with an id",))
        if not isinstance(raw.get("params"), Mapping):
            raise ParamError((f"line {index}: expected a params object",))
        if raw["id"] in predictions:
            raise ParamError((f"line {index}: duplicate id {raw['id']!r}",))
        predictions[raw["id"]] = parse_params(raw["params"]).params
    if not predictions:
        raise ParamError((f"{path} contains no predictions",))
    return predictions


def score_predictions(
    records: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, Params],
    path_for: Mapping[str, Path],
    *,
    split: str = "test",
) -> PredictionScore:
    """Render each prediction on the full-resolution input and score it against the target."""
    deltas: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    baseline_deltas: list[float] = []
    baseline_psnrs: list[float] = []
    baseline_ssims: list[float] = []
    guessed: list[Params] = []
    expected: list[Params] = []
    for record in records:
        if record.get("split") != split:
            continue
        example_id = field(record, "id")
        if not isinstance(example_id, str):
            raise ParamError(("dataset record id must be a string",))
        source = path_for.get(photo_id(record))
        if source is None:
            raise ImageError(f"no source path for photo {photo_id(record)!r}")
        target = load_image(source)
        input_image = apply(target, params(record, "degradation"))
        prediction = predictions.get(example_id, Params())
        corrected = apply(input_image, prediction)
        deltas.append(mean_delta_e(corrected, target))
        psnrs.append(psnr(corrected, target))
        ssims.append(ssim(corrected, target))
        baseline_deltas.append(mean_delta_e(input_image, target))
        baseline_psnrs.append(psnr(input_image, target))
        baseline_ssims.append(ssim(input_image, target))
        guessed.append(prediction)
        expected.append(params(record, "params"))
    if not deltas:
        raise ImageError(f"no records for split {split!r}")
    mae = parameter_mae(guessed, expected)
    return PredictionScore(
        count=len(deltas),
        mean_delta_e=sum(deltas) / len(deltas),
        parameter_mae=mae,
        overall_mae=sum(mae.values()) / len(mae),
        mean_psnr_db=sum(psnrs) / len(psnrs),
        mean_ssim=sum(ssims) / len(ssims),
        baseline_delta_e=sum(baseline_deltas) / len(baseline_deltas),
        baseline_psnr_db=sum(baseline_psnrs) / len(baseline_psnrs),
        baseline_ssim=sum(baseline_ssims) / len(baseline_ssims),
    )
