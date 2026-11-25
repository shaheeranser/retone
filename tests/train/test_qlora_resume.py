"""Checkpoint discovery for resumable training."""

from __future__ import annotations

from pathlib import Path

from retone.train.qlora import _latest_checkpoint


def test_latest_checkpoint_uses_the_highest_step(tmp_path: Path) -> None:
    for step in (100, 200, 50):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-bad").mkdir()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint-200"


def test_no_checkpoint_returns_none(tmp_path: Path) -> None:
    assert _latest_checkpoint(tmp_path) is None
