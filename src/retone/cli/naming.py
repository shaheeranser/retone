"""Default output paths: a suffix beside the input, never the input itself."""

from __future__ import annotations

from pathlib import Path

SUFFIX = "_retone"


def default_output(input_path: Path) -> Path:
    """`photo.jpg` becomes `photo_retone.jpg` in the same directory, so the original is safe."""
    return input_path.with_name(f"{input_path.stem}{SUFFIX}{input_path.suffix}")
