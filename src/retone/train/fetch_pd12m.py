"""Fetch a small PD12M subset and write the source manifest the builder consumes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from retone.data.quality import QualityThresholds, acceptable, measure_quality

PD12M_DATASET = "Spawning/PD12M"
USER_AGENT = "retone-dataset/0.1 (hackathon project)"
DEFAULT_OUT = Path("artifacts/data/raw/pd12m")
_CHUNK = 128
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


@dataclass(frozen=True)
class FetchSummary:
    written: int
    candidate_rows: int
    rejected_metadata: int
    rejected_license: int
    rejected_download: int
    rejected_quality: int
    rejected_hash: int
    destination: str


@dataclass(frozen=True)
class _Candidate:
    photo_id: str
    url: str
    suffix: str
    row: Mapping[str, Any]


def extension_for(mime_type: str) -> str | None:
    return _EXTENSIONS.get(mime_type.lower())


def is_public_domain(license_url: str) -> bool:
    """PD12M should contain only public domain/CC0 rows; refuse anything else."""
    lower = license_url.lower()
    return "publicdomain" in lower or "/cc0/" in lower or "zero/1.0" in lower


def _download(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def _decode(data: bytes) -> Any:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _safe_name(photo_id: str, suffix: str) -> str:
    clean = _SAFE_ID.sub("_", photo_id).strip("._-") or "image"
    return f"{clean}{suffix}"


def _metadata(
    row: Mapping[str, Any], thresholds: QualityThresholds
) -> tuple[str, _Candidate | None]:
    photo_id = row.get("id")
    url = row.get("url")
    mime_type = row.get("mime_type")
    width, height = row.get("width"), row.get("height")
    license_url = row.get("license")
    if (
        not isinstance(photo_id, str)
        or not isinstance(url, str)
        or not isinstance(mime_type, str)
        or not isinstance(license_url, str)
    ):
        return "metadata", None
    if not isinstance(width, int) or not isinstance(height, int):
        return "metadata", None
    if not is_public_domain(license_url):
        return "license", None
    suffix = extension_for(mime_type)
    if suffix is None or min(width, height) < thresholds.min_side:
        return "metadata", None
    return "ok", _Candidate(photo_id, url, suffix, row)


def _fetch_one(
    candidate: _Candidate,
    download: Callable[[str], bytes],
    thresholds: QualityThresholds,
) -> tuple[str, dict[str, Any] | None, bytes | None]:
    try:
        data = download(candidate.url)
    except OSError:
        return "download", None, None
    expected = candidate.row.get("hash")
    if isinstance(expected, str) and expected and hashlib.md5(data).hexdigest() != expected:
        return "hash", None, None
    image = _decode(data)
    if image is None or not acceptable(measure_quality(image), thresholds):
        return "quality", None, None
    record = {
        "photo_id": candidate.photo_id,
        "file": _safe_name(candidate.photo_id, candidate.suffix),
        "photographer": str(candidate.row.get("source", "PD12M")),
        "url": candidate.url,
        "license": str(candidate.row.get("license", "")),
        "source": str(candidate.row.get("source", "PD12M")),
        "width": candidate.row.get("width"),
        "height": candidate.row.get("height"),
        "hash": expected if isinstance(expected, str) else None,
    }
    return "ok", record, data


def _consume(
    candidates: list[_Candidate],
    pool: ThreadPoolExecutor,
    manifest: Any,
    destination: Path,
    counts: dict[str, int],
    used_files: set[str],
    download: Callable[[str], bytes],
    thresholds: QualityThresholds,
    limit: int,
) -> bool:
    for status, record, data in pool.map(
        lambda candidate: _fetch_one(candidate, download, thresholds), candidates
    ):
        counts["candidate_rows"] += 1
        if status != "ok" or record is None or data is None:
            counts[f"rejected_{status}"] += 1
            continue
        if record["file"] in used_files:
            counts["rejected_metadata"] += 1
            continue
        used_files.add(record["file"])
        (destination / record["file"]).write_bytes(data)
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts["written"] += 1
        if counts["written"] >= limit:
            return True
    return False


def fetch(
    rows: Iterable[Mapping[str, Any]],
    destination: Path,
    *,
    limit: int,
    thresholds: QualityThresholds,
    download: Callable[[str], bytes] = _download,
    workers: int = 4,
) -> FetchSummary:
    """Stream metadata, download valid images, verify hashes and write a manifest."""
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.jsonl"
    counts = {
        "written": 0,
        "candidate_rows": 0,
        "rejected_metadata": 0,
        "rejected_license": 0,
        "rejected_download": 0,
        "rejected_quality": 0,
        "rejected_hash": 0,
    }
    seen: set[str] = set()
    used_files: set[str] = set()

    with (
        manifest_path.open("w", encoding="utf-8") as manifest,
        ThreadPoolExecutor(max_workers=max(1, workers)) as pool,
    ):
        batch: list[_Candidate] = []
        for row in rows:
            status, candidate = _metadata(row, thresholds)
            if candidate is None:
                counts[f"rejected_{status}"] += 1
                continue
            if candidate.photo_id in seen:
                counts["rejected_metadata"] += 1
                continue
            seen.add(candidate.photo_id)
            batch.append(candidate)
            if len(batch) < _CHUNK:
                continue
            if _consume(
                batch, pool, manifest, destination, counts, used_files, download, thresholds, limit
            ):
                batch.clear()
                break
            batch.clear()
        if batch and counts["written"] < limit:
            _consume(
                batch, pool, manifest, destination, counts, used_files, download, thresholds, limit
            )

    summary = FetchSummary(destination=str(destination), **counts)
    (destination / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _rows() -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    return cast(
        Iterable[Mapping[str, Any]],
        load_dataset(
            PD12M_DATASET,
            split="train",
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a CC0/public-domain PD12M subset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--min-sharpness", type=float, default=20.0)
    parser.add_argument("--max-clipped", type=float, default=0.05)
    args = parser.parse_args(argv)

    thresholds = QualityThresholds(
        min_side=args.min_side,
        min_sharpness=args.min_sharpness,
        max_clipped_fraction=args.max_clipped,
    )
    try:
        summary = fetch(
            _rows(),
            args.out,
            limit=args.limit,
            thresholds=thresholds,
            workers=args.workers,
        )
    except ImportError:
        print("install the training extra for Hugging Face datasets", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0 if summary.written else 1


if __name__ == "__main__":
    raise SystemExit(main())
