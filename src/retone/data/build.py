"""Building the dataset: parallel pair construction, thumbnail writing and manifests."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.data.corpus import SourceImage, derive_seed, source_metadata, split_items
from retone.data.examples import THUMBNAIL_LONG_EDGE, Example, thumbnail
from retone.data.pairs import PairPolicy, build_pair, choose_kind
from retone.executor import load_image, save_image


@dataclass(frozen=True)
class BuildConfig:
    output_dir: Path
    seed: int = 0
    test_fraction: float = 0.1
    workers: int = 1
    examples_per_photo: int = 4
    long_edge: int = THUMBNAIL_LONG_EDGE
    policy: PairPolicy = field(default_factory=PairPolicy)

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")
        if self.examples_per_photo < 1:
            raise ValueError(f"examples_per_photo must be positive, got {self.examples_per_photo}")
        if self.long_edge < 16:
            raise ValueError(f"long_edge must be at least 16, got {self.long_edge}")


@dataclass(frozen=True)
class BuildSummary:
    train_photos: int
    test_photos: int
    examples: int
    filtered: int
    mean_residual: float
    mean_newly_clipped: float
    label_variance: dict[str, float]


@dataclass(frozen=True)
class _Job:
    photo_id: str
    path: str
    split: str
    source: dict[str, Any]
    seed: int
    output_dir: str
    examples_per_photo: int
    long_edge: int
    policy: PairPolicy


@dataclass(frozen=True)
class _PhotoResult:
    rows: tuple[dict[str, Any], ...]
    residuals: tuple[float, ...]
    clipped: tuple[float, ...]
    filtered: int


def _image_extension(image: NDArray[np.integer[Any]]) -> str:
    return ".png" if image.dtype == np.uint16 else ".jpg"


def _process_photo(job: _Job) -> _PhotoResult:
    output_dir = Path(job.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    image = load_image(Path(job.path))
    thumb = thumbnail(image, job.long_edge)

    rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    clipped_values: list[float] = []
    filtered = 0
    for index in range(job.examples_per_photo):
        rng = np.random.default_rng(derive_seed(job.seed, f"{job.photo_id}:{index}"))
        kind = choose_kind(rng, job.policy)
        pair = build_pair(thumb, rng, kind, job.policy)
        if not pair.accepted(job.policy):
            filtered += 1
            continue
        example_id = f"{job.photo_id}-{index:02d}-{kind.value}"
        image_file = f"images/{example_id}{_image_extension(thumb)}"
        save_image(output_dir / image_file, pair.input_image)
        example = Example(
            example_id=example_id,
            split=job.split,
            image_file=image_file,
            params=pair.label,
            kind=kind,
            source=job.source,
            degradation=pair.degradation,
            residual=pair.residual,
        )
        rows.append(example.record())
        residuals.append(pair.residual)
        clipped_values.append(pair.newly_clipped)
    return _PhotoResult(tuple(rows), tuple(residuals), tuple(clipped_values), filtered)


def _jobs_for(items: tuple[SourceImage, ...], split: str, config: BuildConfig) -> tuple[_Job, ...]:
    return tuple(
        _Job(
            photo_id=item.photo_id,
            path=str(item.path),
            split=split,
            source=source_metadata(item),
            seed=config.seed,
            output_dir=str(config.output_dir),
            examples_per_photo=config.examples_per_photo,
            long_edge=config.long_edge,
            policy=config.policy,
        )
        for item in sorted(items, key=lambda entry: entry.photo_id)
    )


def _run_jobs(jobs: tuple[_Job, ...], workers: int) -> tuple[_PhotoResult, ...]:
    if workers == 1:
        return tuple(_process_photo(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(_process_photo, jobs))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_variance(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    names = tuple(rows[0]["params"])
    return {name: float(np.var([float(row["params"][name]) for row in rows])) for name in names}


def build_dataset(items: tuple[SourceImage, ...], config: BuildConfig) -> BuildSummary:
    """Build all pairs, write the manifests and return measured statistics."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train, test = split_items(items, seed=config.seed, test_fraction=config.test_fraction)
    jobs = _jobs_for(train, "train", config) + _jobs_for(test, "test", config)
    results = _run_jobs(jobs, config.workers)

    rows = [row for result in results for row in result.rows]
    rows.sort(key=lambda row: row["id"])
    residuals = [value for result in results for value in result.residuals]
    clipped = [value for result in results for value in result.clipped]

    _write_jsonl(config.output_dir / "examples.jsonl", rows)
    test_ids = {item.photo_id for item in test}
    sources = [
        {**source_metadata(item), "split": "test" if item.photo_id in test_ids else "train"}
        for item in sorted(items, key=lambda entry: entry.photo_id)
    ]
    _write_jsonl(config.output_dir / "sources.jsonl", sources)
    # Local-only map for the full-resolution ceiling check; never uploaded.
    (config.output_dir / "source_paths.json").write_text(
        json.dumps(
            {item.photo_id: str(item.path.resolve()) for item in items}, indent=2, sort_keys=True
        )
        + "\n"
    )

    summary = BuildSummary(
        train_photos=len(train),
        test_photos=len(test),
        examples=len(rows),
        filtered=sum(result.filtered for result in results),
        mean_residual=float(np.mean(residuals)) if residuals else 0.0,
        mean_newly_clipped=float(np.mean(clipped)) if clipped else 0.0,
        label_variance=_label_variance(rows),
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps({**asdict(summary), "config": _config_dict(config)}, indent=2, sort_keys=True)
        + "\n"
    )
    return summary


def _config_dict(config: BuildConfig) -> dict[str, Any]:
    policy = config.policy
    return {
        "seed": config.seed,
        "test_fraction": config.test_fraction,
        "examples_per_photo": config.examples_per_photo,
        "long_edge": config.long_edge,
        "policy": {
            "identity_weight": policy.identity_weight,
            "synthetic_weight": policy.synthetic_weight,
            "mild_weight": policy.mild_weight,
            "synthetic_magnitude": [
                policy.synthetic_magnitude.low,
                policy.synthetic_magnitude.high,
            ],
            "mild_magnitude": [policy.mild_magnitude.low, policy.mild_magnitude.high],
            "max_params": policy.max_params,
            "min_params": policy.min_params,
            "label_step": policy.label_step,
            "max_newly_clipped": policy.max_newly_clipped,
            "max_residual": policy.max_residual,
            "max_attempts": policy.max_attempts,
        },
    }
