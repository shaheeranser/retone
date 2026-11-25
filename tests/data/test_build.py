"""End-to-end build: manifests, images, statistics and worker-count reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

from image_fixtures import make_image, write_image
from retone.data.build import BuildConfig, build_dataset
from retone.data.corpus import load_manifest
from retone.train.build_dataset import main as build_cli


def _corpus(tmp_path: Path, count: int = 6) -> Path:
    lines: list[str] = []
    for index in range(count):
        photo_id = f"photo{index}"
        filename = f"{photo_id}.png"
        write_image(tmp_path / filename, make_image(80, 100, seed=index))
        lines.append(
            json.dumps(
                {
                    "photo_id": photo_id,
                    "file": filename,
                    "photographer": "Tester",
                    "url": f"https://example.org/{photo_id}",
                    "license": "CC0-1.0",
                }
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def test_build_is_identical_for_one_or_many_workers(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    items = load_manifest(manifest)

    single = build_dataset(
        items,
        BuildConfig(output_dir=tmp_path / "single", seed=11, workers=1, examples_per_photo=3),
    )
    many = build_dataset(
        items,
        BuildConfig(output_dir=tmp_path / "many", seed=11, workers=2, examples_per_photo=3),
    )

    assert (tmp_path / "single" / "examples.jsonl").read_text() == (
        tmp_path / "many" / "examples.jsonl"
    ).read_text()
    assert (tmp_path / "single" / "summary.json").read_text() == (
        tmp_path / "many" / "summary.json"
    ).read_text()
    assert single == many
    assert single.train_photos + single.test_photos == 6
    assert set(single.label_variance) == {
        "exposure",
        "temperature",
        "tint",
        "contrast",
        "highlights",
        "shadows",
        "saturation",
        "vibrance",
        "sharpen_amount",
    }


def test_examples_reference_existing_images(tmp_path: Path) -> None:
    manifest = _corpus(tmp_path)
    build_dataset(
        load_manifest(manifest),
        BuildConfig(output_dir=tmp_path / "out", seed=1, examples_per_photo=2),
    )
    rows = [
        json.loads(line) for line in (tmp_path / "out" / "examples.jsonl").read_text().splitlines()
    ]
    assert rows
    for row in rows:
        assert (tmp_path / "out" / row["image"]).is_file()
        assert json.loads(row["answer"]) == row["params"]
        assert row["source"]["license"] == "CC0-1.0"
        assert row["residual"] >= 0.0


def test_cli_reports_a_missing_corpus(tmp_path: Path) -> None:
    assert build_cli(["--manifest", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path)]) == 2
