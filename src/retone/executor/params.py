"""Validated parameter values and parsing of untrusted input (model output, user flags)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from retone.executor.errors import ParamError
from retone.executor.schema import SCHEMA, SPEC_BY_NAME


@dataclass(frozen=True)
class Params:
    exposure: int = 0
    temperature: int = 0
    tint: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    saturation: int = 0
    vibrance: int = 0
    sharpen_amount: int = 0

    def as_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def coarsen(self, step: int) -> Params:
        """Round every value to a multiple of `step`, clamped to the schema."""
        if step < 1:
            raise ValueError(f"coarsen step must be positive, got {step}")
        values = {
            spec.name: spec.clamp(round(getattr(self, spec.name) / step) * step) for spec in SCHEMA
        }
        return Params(**values)


@dataclass(frozen=True)
class ParsedParams:
    params: Params
    clamped: tuple[str, ...]


def _is_integer(value: object) -> bool:
    # bool is an int subclass, but "true" is not an adjustment value.
    return isinstance(value, int) and not isinstance(value, bool)


def parse_params(raw: Mapping[str, object]) -> ParsedParams:
    """Turn untrusted input into valid Params, clamping out-of-range values.

    Unknown names and non-integer values are rejected together so the caller sees
    every problem at once; missing names take their schema default.
    """
    problems = [f"unknown parameter {name!r}" for name in raw if name not in SPEC_BY_NAME]
    problems += [
        f"{spec.name} must be an integer, got {raw[spec.name]!r}"
        for spec in SCHEMA
        if spec.name in raw and not _is_integer(raw[spec.name])
    ]
    if problems:
        raise ParamError(tuple(problems))

    values: dict[str, int] = {}
    clamped: list[str] = []
    for spec in SCHEMA:
        value = raw.get(spec.name, spec.default)
        assert isinstance(value, int)
        values[spec.name] = spec.clamp(value)
        if not spec.contains(value):
            clamped.append(spec.name)
    return ParsedParams(Params(**values), tuple(clamped))
