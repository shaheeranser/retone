"""Stable process exit codes, part of the CLI's public contract."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Documented exit statuses so scripts can branch on failure kind."""

    SUCCESS = 0
    USAGE = 2
    INPUT = 3
    MODEL = 4
    WRITE = 5
