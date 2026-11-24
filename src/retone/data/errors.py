"""Typed errors raised while building the dataset."""

from __future__ import annotations


class DatasetError(ValueError):
    """Raised when a manifest, image or policy is invalid; lists every problem at once."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
