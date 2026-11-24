"""The parameter schema: every parameter's name, integer range, default and identity value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    minimum: int
    maximum: int
    default: int
    identity: int

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum

    def clamp(self, value: int) -> int:
        return max(self.minimum, min(self.maximum, value))


def _signed(name: str) -> ParamSpec:
    return ParamSpec(name, -100, 100, 0, 0)


SCHEMA: tuple[ParamSpec, ...] = (
    _signed("exposure"),
    _signed("temperature"),
    _signed("tint"),
    _signed("contrast"),
    _signed("highlights"),
    _signed("shadows"),
    _signed("saturation"),
    _signed("vibrance"),
    ParamSpec("sharpen_amount", 0, 100, 0, 0),
)

SPEC_BY_NAME: dict[str, ParamSpec] = {spec.name: spec for spec in SCHEMA}
