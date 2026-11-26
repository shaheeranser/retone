"""Small deterministic RGB images for tests, without any network or fixture files."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retone.executor import save_image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_image(height: int = 64, width: int = 96, seed: int = 0) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    red = (xs * 255 / max(1, width - 1)).astype(np.float64)
    green = (ys * 255 / max(1, height - 1)).astype(np.float64)
    blue = np.full((height, width), 96.0)
    base = np.stack([red, green, blue], axis=2) + rng.normal(0.0, 6.0, (height, width, 3))
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


def write_image(path: Path, image: NDArray[np.integer[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(path, image)
    return path


def icc_profile(description: str) -> bytes:
    """A minimal RGB ICC profile whose `desc` tag holds the given text."""
    header = bytearray(128)
    header[16:20] = b"RGB "
    tag = b"desc" + b"\x00" * 4 + struct.pack(">I", len(description)) + description.encode()
    entry = b"desc" + struct.pack(">II", 128 + 4 + 12, len(tag))
    return bytes(header) + struct.pack(">I", 1) + entry + tag


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def write_png_with_profile(path: Path, image: NDArray[np.integer[Any]], description: str) -> Path:
    """Write a real image as PNG with an iCCP chunk inserted after the IHDR chunk."""
    plain = write_image(path.with_suffix(".plain.png"), image).read_bytes()
    assert plain.startswith(PNG_SIGNATURE)
    ihdr_length = struct.unpack(">I", plain[8:12])[0]
    insert_at = 8 + 4 + 4 + ihdr_length + 4  # signature + IHDR chunk + its CRC
    iccp = _png_chunk(b"iCCP", b"icc\x00\x00" + zlib.compress(icc_profile(description)))
    path.write_bytes(plain[:insert_at] + iccp + plain[insert_at:])
    return path
