"""ICC detection: sRGB is quiet, non-sRGB warns, and plain images carry no profile."""

from __future__ import annotations

from pathlib import Path

from image_fixtures import make_image, write_image, write_png_with_profile
from retone.executor.icc import embedded_profile, is_non_srgb


def test_plain_png_has_no_profile(tmp_path: Path) -> None:
    path = write_image(tmp_path / "plain.png", make_image())
    assert embedded_profile(path) is None
    assert not is_non_srgb(path)


def test_srgb_profile_is_not_flagged(tmp_path: Path) -> None:
    path = write_png_with_profile(tmp_path / "srgb.png", make_image(), "sRGB IEC61966-2.1")
    assert embedded_profile(path) is not None
    assert not is_non_srgb(path)


def test_non_srgb_profile_is_flagged(tmp_path: Path) -> None:
    path = write_png_with_profile(tmp_path / "adobe.png", make_image(), "Adobe RGB (1998)")
    assert is_non_srgb(path)


def test_missing_file_is_treated_as_no_profile(tmp_path: Path) -> None:
    assert embedded_profile(tmp_path / "absent.png") is None
