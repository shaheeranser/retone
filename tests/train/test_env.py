"""Local .env parsing: comments, quotes and real-environment precedence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from retone.train.env import load_env


def test_load_env_parses_values_and_does_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RETONE_KEEP", "original")
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "RETONE_ADDED=value",
                'RETONE_QUOTED="two words"',
                "export RETONE_EXPORTED=yes",
                "RETONE_KEEP=replacement",
                "",
            ]
        )
    )
    try:
        load_env(path)
        assert os.environ["RETONE_ADDED"] == "value"
        assert os.environ["RETONE_QUOTED"] == "two words"
        assert os.environ["RETONE_EXPORTED"] == "yes"
        assert os.environ["RETONE_KEEP"] == "original"
    finally:
        for name in ("RETONE_ADDED", "RETONE_QUOTED", "RETONE_EXPORTED"):
            os.environ.pop(name, None)


def test_missing_file_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETONE_ABSENT", raising=False)
    load_env(tmp_path / "absent")
    assert "RETONE_ABSENT" not in os.environ
