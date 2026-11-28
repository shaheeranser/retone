"""Exports the parameter schema as JSON Schema, the grammar that constrains model output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from retone.executor.schema import SCHEMA


def json_schema(limits: Mapping[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    """The schema grammar; `limits` narrows named parameters inside their schema range."""
    bounds = {
        spec.name: (
            max(spec.minimum, limits[spec.name][0])
            if limits and spec.name in limits
            else spec.minimum,
            min(spec.maximum, limits[spec.name][1])
            if limits and spec.name in limits
            else spec.maximum,
        )
        for spec in SCHEMA
    }
    return {
        "type": "object",
        "properties": {
            name: {"type": "integer", "minimum": low, "maximum": high}
            for name, (low, high) in bounds.items()
        },
        "required": [spec.name for spec in SCHEMA],
        "additionalProperties": False,
    }
