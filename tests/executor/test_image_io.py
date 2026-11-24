from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from retone.executor.errors import ImageError
from retone.executor.image_io import load_image, save_image


def test_save_writes_bgr_so_red_stays_red(tmp_path: Path) -> None:
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 255
    path = tmp_path / "red.png"
    save_image(path, red)
    on_disk_bgr = cv2.imread(str(path))
    assert on_disk_bgr[0, 0].tolist() == [0, 0, 255]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16], ids=["8-bit", "16-bit"])
def test_round_trip_keeps_pixels_and_bit_depth(tmp_path: Path, dtype: type[np.integer]) -> None:
    rng = np.random.default_rng(0)
    peak = np.iinfo(dtype).max
    image = rng.integers(0, peak + 1, size=(5, 7, 3), dtype=dtype)
    path = tmp_path / "image.png"
    save_image(path, image)
    loaded = load_image(path)
    assert loaded.dtype == image.dtype
    assert np.array_equal(loaded, image)


def test_load_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageError):
        load_image(tmp_path / "absent.png")


def test_load_reports_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageError):
        load_image(path)


def test_save_reports_unwritable_destination(tmp_path: Path) -> None:
    with pytest.raises(ImageError):
        save_image(tmp_path / "missing-dir" / "out.png", np.zeros((1, 1, 3), dtype=np.uint8))
