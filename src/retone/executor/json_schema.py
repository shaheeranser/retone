"""Exports the parameter schema as JSON Schema, the grammar that constrains model output."""

from __future__ import annotations

from typing import Any

from retone.executor.schema import SCHEMA


def json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            spec.name: {"type": "integer", "minimum": spec.minimum, "maximum": spec.maximum}
            for spec in SCHEMA
        },
        "required": [spec.name for spec in SCHEMA],
        "additionalProperties": False,
    }
