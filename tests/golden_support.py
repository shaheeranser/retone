"""Golden-file comparison with an explicit update switch, so a diff is reviewable."""

from __future__ import annotations

from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"


def assert_golden(name: str, actual: str, *, update: bool) -> None:
    path = GOLDEN_DIR / name
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
    assert path.read_text() == actual, f"golden mismatch for {name}; rerun with --update-golden"
