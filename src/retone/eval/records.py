"""Field access for dataset records read by evaluation, validated at the boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from retone.executor import ParamError, Params, parse_params


def field(record: Mapping[str, Any], name: str) -> object:
    if name not in record:
        raise ParamError((f"dataset record has no {name!r} field",))
    return record[name]


def photo_id(record: Mapping[str, Any]) -> str:
    source = field(record, "source")
    if not isinstance(source, Mapping) or "photo_id" not in source:
        raise ParamError(("dataset record source has no photo_id",))
    value = source["photo_id"]
    if not isinstance(value, str):
        raise ParamError(("dataset record photo_id must be a string",))
    return value


def params(record: Mapping[str, Any], name: str) -> Params:
    raw = field(record, name)
    if not isinstance(raw, Mapping):
        raise ParamError((f"dataset record {name!r} must be an object",))
    return parse_params(raw).params
