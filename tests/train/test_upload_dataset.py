"""The upload entry script builds rows and refuses to run without an environment token."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retone.train.upload_dataset import _rows, main


def _examples(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "photo-00.jpg").write_bytes(b"not-an-image")
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "photo-00",
                "split": "train",
                "image": "images/photo-00.jpg",
                "instruction": "instruction",
                "answer": "{}",
                "kind": "identity",
                "source": {"photo_id": "photo"},
            }
        )
        + "\n"
    )
    return path


def test_rows_carry_the_image_path_and_metadata(tmp_path: Path) -> None:
    rows = _rows(_examples(tmp_path), tmp_path)
    assert rows[0]["photo_id"] == "photo"
    assert Path(rows[0]["image"]).is_file()


def test_main_requires_a_token_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert main(["--examples", str(tmp_path), "--image-root", str(tmp_path), "--repo-id", "x"]) == 2
