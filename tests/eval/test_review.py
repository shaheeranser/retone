"""Review bundle structure, blind labels, HTML, CSV and metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from image_fixtures import make_image, write_image
from retone.eval.__main__ import main as eval_main
from retone.eval.human import load_human_pairs
from retone.eval.review import build_review_bundle
from retone.executor import Params


def _pairs(tmp_path: Path) -> tuple[Path, Params]:
    before = make_image(40, 60, seed=3)
    after = np.clip(before.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    write_image(tmp_path / "before" / "p0001.jpg", before)
    write_image(tmp_path / "after" / "p0001.jpg", after)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
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
    return manifest, Params()


def test_single_label_bundle(tmp_path: Path) -> None:
    manifest, _ = _pairs(tmp_path)
    pairs = load_human_pairs(manifest)
    out = tmp_path / "review"
    bundle = build_review_bundle(pairs, {"tuned": {"p0001": Params()}}, out)

    assert bundle.pairs == 1
    assert bundle.labels == {"tuned": "model"}
    assert (out / "original" / "p0001.jpg").is_file()
    assert (out / "reference" / "p0001.jpg").is_file()
    assert (out / "model" / "p0001.jpg").is_file()
    page = (out / "review.html").read_text()
    assert "p0001" in page
    assert "Human reference" in page
    assert "Original" in page
    assert "model" in page

    with (out / "ratings.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "pair_id"
    assert len(rows) == 2
    assert rows[1][2] == "model"

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["by_label"]["tuned"]["count"] == 1
    manifest_data = json.loads((out / "manifest.json").read_text())
    assert manifest_data["labels"] == {"tuned": "model"}


def test_two_labels_are_blind_and_deterministic(tmp_path: Path) -> None:
    manifest, _ = _pairs(tmp_path)
    pairs = load_human_pairs(manifest)
    first = build_review_bundle(
        pairs, {"base": {"p0001": Params()}, "tuned": {"p0001": Params(exposure=5)}}, tmp_path / "a"
    )
    second = build_review_bundle(
        pairs, {"base": {"p0001": Params()}, "tuned": {"p0001": Params(exposure=5)}}, tmp_path / "b"
    )
    assert first.labels == second.labels
    assert set(first.labels.values()) == {"A", "B"}
    assert (tmp_path / "a" / "model" / first.labels["base"] / "p0001.jpg").is_file()
    assert (tmp_path / "a" / "model" / first.labels["tuned"] / "p0001.jpg").is_file()

    with (tmp_path / "a" / "ratings.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 3
    assert {rows[1][2], rows[2][2]} == {"A", "B"}

    manifest_data = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest_data["display_to_actual"][first.labels["base"]] == "base"


def test_review_cli(tmp_path: Path) -> None:
    manifest, _ = _pairs(tmp_path)
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(json.dumps({"id": "p0001", "params": {}}) + "\n")
    out = tmp_path / "review"
    code = eval_main(
        [
            "review",
            "--pairs",
            str(manifest),
            "--prediction",
            f"tuned={predictions}",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "review.html").is_file()
