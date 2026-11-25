"""Minimal .env loading for local entry points; real environment variables win."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    env_path = Path(".env") if path is None else path
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
            cleaned = cleaned[1:-1]
        os.environ.setdefault(key, cleaned)
