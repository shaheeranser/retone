"""PD12M metadata filtering, downloads, hashes and quality gates (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from image_fixtures import make_image
from retone.data.quality import QualityThresholds
from retone.train.fetch_pd12m import (
    extension_for,
    fetch,
    is_public_domain,
)

CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
BY = "https://creativecommons.org/licenses/by/4.0/"


def _encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return buffer.tobytes()


def _row(
    photo_id: str,
    url: str,
    *,
    width: int = 800,
    height: int = 600,
    license_url: str = CC0,
    digest: str | None = None,
) -> dict[str, Any]:
    return {
        "id": photo_id,
        "url": url,
        "width": width,
        "height": height,
        "mime_type": "image/jpeg",
        "license": license_url,
        "source": "test-source",
        "hash": digest,
    }


def test_public_domain_filter_and_extensions() -> None:
    assert is_public_domain(CC0)
    assert is_public_domain("https://creativecommons.org/publicdomain/mark/1.0/")
    assert not is_public_domain(BY)
    assert extension_for("image/jpeg") == ".jpg"
    assert extension_for("IMAGE/PNG") == ".png"
    assert extension_for("image/gif") is None


def test_fetch_filters_licenses_hashes_quality_and_errors(tmp_path: Path) -> None:
    image = make_image(600, 800, seed=1)
    data = _encode(image)
    digest = hashlib.md5(data).hexdigest()

    def download(url: str) -> bytes:
        if url == "error":
            raise OSError("offline")
        if url == "garbage":
            return b"not an image"
        return data

    rows = [
        _row("ok", "ok", digest=digest),
        _row("by", "ok", license_url=BY),
        _row("small", "ok", width=100, height=100),
        _row("badhash", "ok", digest="0" * 32),
        _row("offline", "error"),
        _row("garbage", "garbage"),
    ]
    thresholds = QualityThresholds(min_side=256, min_sharpness=1.0)
    summary = fetch(
        rows,
        tmp_path / "out",
        limit=10,
        thresholds=thresholds,
        download=download,
        workers=2,
    )

    assert summary.written == 1
    assert summary.rejected_license == 1
    assert summary.rejected_hash == 1
    assert summary.rejected_download == 1
    assert summary.rejected_quality == 1
    rows_out = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    assert len(rows_out) == 1
    record = json.loads(rows_out[0])
    assert record["photo_id"] == "ok"
    assert (tmp_path / "out" / record["file"]).is_file()


def test_fetch_rejects_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fetch([], tmp_path, limit=0, thresholds=QualityThresholds())
    with pytest.raises(ValueError):
        fetch([], tmp_path, limit=1, thresholds=QualityThresholds(), workers=0)
