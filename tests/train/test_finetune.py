"""Offline checks of the fine-tune entry point: dry run and token handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_fixtures import make_image, write_image
from retone.executor import Params
from retone.train.finetune import main


def _corpus(tmp_path: Path) -> Path:
    write_image(tmp_path / "a.png", make_image(32, 40, seed=1))
    write_image(tmp_path / "b.png", make_image(32, 40, seed=2))
    rows = [
        {
            "id": "a-00",
            "split": "train",
            "image": "a.png",
            "instruction": "Make it better.",
            "answer": json.dumps(Params(exposure=23).as_dict()),
            "params": Params(exposure=23).as_dict(),
        },
        {
            "id": "b-00",
            "split": "test",
            "image": "b.png",
            "instruction": "Make it better.",
            "answer": json.dumps(Params(exposure=-12).as_dict()),
            "params": Params(exposure=-12).as_dict(),
        },
    ]
    path = tmp_path / "examples.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_dry_run_validates_local_examples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = _corpus(tmp_path)
    code = main(
        ["--examples", str(examples), "--image-root", str(tmp_path), "--dry-run", "--bin-step", "5"]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["train"] == 1
    assert summary["test"] == 1
    assert summary["bin_step"] == 5


def test_training_requires_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    examples = _corpus(tmp_path)
    assert main(["--examples", str(examples), "--output-repo", "someone/model"]) == 2


def test_training_requires_an_output_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "token")
    examples = _corpus(tmp_path)
    assert main(["--examples", str(examples)]) == 2


def test_dry_run_needs_a_local_source(tmp_path: Path) -> None:
    assert main(["--dataset-repo", "someone/data", "--dry-run"]) == 2
