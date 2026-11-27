"""Multi-GPU launch helpers: rank discovery without importing torch."""

from __future__ import annotations

import os


def world_size() -> int:
    """Number of processes torchrun launched; 1 for a plain `python -m` run."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_main() -> bool:
    """True on rank 0 and when running single-process (unset RANK defaults to 0)."""
    return rank() == 0
