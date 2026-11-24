"""Parsing of model output into parameters, falling back to defaults on any defect."""

from __future__ import annotations

import json
import re

from retone.executor import ParamError, Params, parse_params

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_generated_params(text: str) -> Params:
    """Take the first JSON object in the text; malformed or invalid output means identity."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        return Params()
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Params()
    if not isinstance(raw, dict):
        return Params()
    try:
        return parse_params(raw).params
    except ParamError:
        return Params()
