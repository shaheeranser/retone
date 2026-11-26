"""Turn an untrusted model answer into safe parameters, falling back to defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from retone.executor import ParamError, Params, parse_params


@dataclass(frozen=True)
class ValidatedParams:
    """Safe parameters plus the human-readable warnings to print on stderr."""

    params: Params
    warnings: tuple[str, ...]


def validate_model_output(raw: Mapping[str, object]) -> ValidatedParams:
    """Clamp in-range values and fall back to defaults when the answer is unusable.

    The server is a process boundary, so a malformed answer must not crash the run.
    Clamped values keep the model's intent; a structurally invalid answer (unknown
    names, non-integers) discards it entirely and renders the identity parameters.
    """
    try:
        parsed = parse_params(raw)
    except ParamError as err:
        rejected = (f"model output rejected, using defaults: {err}",)
        return ValidatedParams(Params(), rejected)
    warnings = tuple(
        f"model value for {name} was out of range and has been clamped" for name in parsed.clamped
    )
    return ValidatedParams(parsed.params, warnings)
