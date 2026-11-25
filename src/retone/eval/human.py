"""Human-edited evaluation pairs: render predictions on the before image and score them."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from retone.eval.delta_e import mean_delta_e
from retone.executor import ParamError, Params, apply, load_image, save_image


@dataclass(frozen=True)
class HumanPair:
    pair_id: str
    before: Path
    after: Path
    variant: str | None = None


@dataclass(frozen=True)
class PairScore:
    pair_id: str
    delta_e: float


@dataclass(frozen=True)
class HumanReport:
    count: int
    mean_delta_e: float
    median_delta_e: float
    per_pair: tuple[PairScore, ...]


def load_human_pairs(path: Path) -> tuple[HumanPair, ...]:
    """Read the collector's pairs.jsonl, resolving paths relative to the manifest."""
    root = path.parent
    pairs: list[HumanPair] = []
    seen: set[str] = set()
    problems: list[str] = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as err:
            problems.append(f"line {index}: invalid JSON ({err.msg})")
            continue
        if not isinstance(row, Mapping):
            problems.append(f"line {index}: expected an object")
            continue
        pair_id = row.get("pair_id")
        before, after = row.get("before"), row.get("after")
        if (
            not isinstance(pair_id, str)
            or not isinstance(before, str)
            or not isinstance(after, str)
        ):
            problems.append(f"line {index}: pair_id, before and after must be strings")
            continue
        if pair_id in seen:
            problems.append(f"line {index}: duplicate pair_id {pair_id!r}")
            continue
        before_path = Path(before) if Path(before).is_absolute() else root / before
        after_path = Path(after) if Path(after).is_absolute() else root / after
        if not before_path.is_file() or not after_path.is_file():
            problems.append(f"line {index}: image missing for pair {pair_id!r}")
            continue
        variant = row.get("variant")
        seen.add(pair_id)
        pairs.append(
            HumanPair(
                pair_id=pair_id,
                before=before_path,
                after=after_path,
                variant=variant if isinstance(variant, str) else None,
            )
        )
    if problems:
        raise ParamError(tuple(problems))
    if not pairs:
        raise ParamError((f"{path} contains no human pairs",))
    return tuple(pairs)


def render_predictions(
    pairs: tuple[HumanPair, ...],
    predictions: Mapping[str, Params],
    output_dir: Path,
) -> HumanReport:
    """Render each prediction on the full-resolution before image and score ΔE to the after."""
    missing = [pair.pair_id for pair in pairs if pair.pair_id not in predictions]
    if missing:
        raise ParamError((f"missing predictions for {missing}",))
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[PairScore] = []
    for pair in pairs:
        before = load_image(pair.before)
        after = load_image(pair.after)
        if before.shape != after.shape:
            raise ParamError(
                (
                    f"pair {pair.pair_id!r} has mismatched sizes "
                    f"{before.shape[1]}x{before.shape[0]} and {after.shape[1]}x{after.shape[0]}",
                )
            )
        rendered = apply(before, predictions[pair.pair_id])
        save_image(output_dir / f"{pair.pair_id}.jpg", rendered)
        scores.append(PairScore(pair.pair_id, mean_delta_e(rendered, after)))
    values = [score.delta_e for score in scores]
    return HumanReport(
        count=len(scores),
        mean_delta_e=sum(values) / len(values),
        median_delta_e=float(statistics.median(values)),
        per_pair=tuple(scores),
    )


def load_predictions_by_label(specs: list[str]) -> dict[str, dict[str, Params]]:
    """Parse repeated `label=path` arguments into prediction sets."""
    from retone.eval.predictions import load_predictions

    labels: dict[str, dict[str, Params]] = {}
    problems: list[str] = []
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            problems.append(f"prediction {spec!r} must look like label=path")
            continue
        if label in labels:
            problems.append(f"duplicate prediction label {label!r}")
            continue
        labels[label] = load_predictions(Path(raw_path))
    if problems:
        raise ParamError(tuple(problems))
    if not labels:
        raise ParamError(("at least one --prediction label=path is required",))
    return labels
