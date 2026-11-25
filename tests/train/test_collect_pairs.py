"""The collector CLI copies by default and reports a summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_fixtures import make_image, write_image
from retone.train.collect_pairs import main


def test_cli_copies_pairs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "pictures"
    source.mkdir()
    write_image(source / "IMG_1.png", make_image(16, 16, seed=1))
    write_image(source / "IMG_1(edit)2.png", make_image(16, 16, seed=2))
    destination = tmp_path / "collected"

    assert main(["--source", str(source), "--dest", str(destination)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pairs"] == 1
    assert summary["destination"] == str(destination)
    assert (destination / "pairs.jsonl").is_file()
    assert (destination / "images" / "p0001__before.png").is_file()
    assert (destination / "images" / "p0001__after.png").is_file()


def test_cli_reports_a_missing_source(tmp_path: Path) -> None:
    assert main(["--source", str(tmp_path / "absent"), "--dest", str(tmp_path)]) == 2
