"""Typed errors raised by the executor."""

from __future__ import annotations


class ParamError(ValueError):
    """Raised for malformed parameter input; lists every offending field at once."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class ImageError(ValueError):
    """Raised when an image cannot be read, written, or is not a supported array."""
