"""Rank guards, which must hold for a plain single-process run too."""

from __future__ import annotations

import pytest

from retone.train.distributed import is_distributed, is_main, rank, world_size


def test_unset_environment_is_a_single_main_process(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert world_size() == 1
    assert rank() == 0
    assert is_distributed() is False
    assert is_main() is True


def test_rank_zero_of_a_distributed_run_is_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    assert is_distributed() is True
    assert is_main() is True


def test_non_zero_rank_is_not_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    assert is_main() is False
    assert rank() == 1
