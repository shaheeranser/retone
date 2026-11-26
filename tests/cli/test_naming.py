"""Default output naming never collides with the input."""

from __future__ import annotations

from pathlib import Path

import pytest

from retone.cli.naming import default_output


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", "photo_retone.jpg"),
        ("holiday.PNG", "holiday_retone.PNG"),
        ("no-extension", "no-extension_retone"),
        ("archive.photo.jpeg", "archive.photo_retone.jpeg"),
    ],
)
def test_suffix_is_added_before_the_extension(name: str, expected: str) -> None:
    assert default_output(Path("/photos") / name) == Path("/photos") / expected


def test_output_stays_beside_the_input() -> None:
    result = default_output(Path("/a/b/c.jpg"))
    assert result.parent == Path("/a/b")
    assert result != Path("/a/b/c.jpg")
