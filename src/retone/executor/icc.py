"""Detect an embedded ICC profile so the CLI can warn when it is not sRGB.

The executor assumes sRGB and never converts (01-executor design OD-04). A non-sRGB
input is therefore colour-managed incorrectly, which is worth a warning but must not
change the pipeline. Detection only reads file headers and the profile's description
tag; no pixels are decoded. Only JPEG (APP2) and PNG (iCCP) carry profiles we can read
without extra dependencies; other containers are treated as sRGB.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

_JPEG_SOI = b"\xff\xd8"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ICC_JPEG_MARKER = b"ICC_PROFILE\x00"
_HEADER_SIZE = 128
_RGB = b"RGB "
_SRGB = "srgb"


def embedded_profile(path: Path) -> bytes | None:
    """Return the raw embedded ICC profile bytes, or None when there is none."""
    try:
        head = path.read_bytes()[: 8 * 1024 * 1024]
    except OSError:
        return None
    if head.startswith(_PNG_SIGNATURE):
        return _png_iccp(head)
    if head.startswith(_JPEG_SOI):
        return _jpeg_app2(head)
    return None


def is_non_srgb(path: Path) -> bool:
    """True when the file embeds a profile that is not an RGB sRGB profile."""
    profile = embedded_profile(path)
    if profile is None:
        return False
    if len(profile) < _HEADER_SIZE or profile[16:20] != _RGB:
        return True
    return _SRGB not in _description(profile).lower()


def _description(profile: bytes) -> str:
    """Read the profile's `desc` tag, tolerating both ICC v2 and v4 layouts."""
    try:
        count = struct.unpack(">I", profile[_HEADER_SIZE : _HEADER_SIZE + 4])[0]
    except struct.error:
        return ""
    for index in range(count):
        entry = _HEADER_SIZE + 4 + index * 12
        signature, offset, size = struct.unpack(">4sII", profile[entry : entry + 12])
        if signature != b"desc":
            continue
        return _decode_desc(profile[offset : offset + size])
    return ""


def _decode_desc(tag: bytes) -> str:
    kind = tag[:4]
    if kind == b"desc":
        length = struct.unpack(">I", tag[8:12])[0]
        return tag[12 : 12 + length].split(b"\x00", 1)[0].decode("latin-1", "replace")
    if kind == b"mluc":
        records = struct.unpack(">I", tag[8:12])[0]
        if records == 0:
            return ""
        length, offset = struct.unpack(">II", tag[20:28])
        raw = tag[offset : offset + length]
        return raw.decode("utf-16-be", "replace").replace("\x00", "").strip()
    return ""


def _png_iccp(data: bytes) -> bytes | None:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        body_start = offset + 8
        body_end = body_start + length
        if body_end > len(data):
            return None
        if chunk_type == b"iCCP":
            body = data[body_start:body_end]
            name_end = body.find(b"\x00")
            if name_end < 0 or name_end + 1 >= len(body):
                return None
            try:
                return zlib.decompress(body[name_end + 2 :])
            except zlib.error:
                return None
        offset = body_end + 4  # skip the 4-byte CRC
    return None


def _jpeg_app2(data: bytes) -> bytes | None:
    offset = 2
    chunks: dict[int, bytes] = {}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        if length < 2:
            return None
        segment = data[offset + 4 : offset + 2 + length]
        if marker == 0xDA:  # start of scan: no more metadata segments follow
            break
        if marker == 0xE2 and segment.startswith(_ICC_JPEG_MARKER):
            header = len(_ICC_JPEG_MARKER)
            sequence = segment[header]
            chunks[sequence] = segment[header + 2 :]
        offset += 2 + length
    if not chunks:
        return None
    return b"".join(chunks[index] for index in sorted(chunks))
