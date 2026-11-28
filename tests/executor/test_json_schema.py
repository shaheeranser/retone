from __future__ import annotations

import json

import pytest

from golden_support import assert_golden
from retone.executor.json_schema import json_schema
from retone.executor.params import parse_params
from retone.executor.schema import SCHEMA, ParamSpec


def test_json_schema_matches_golden(update_golden: bool) -> None:
    rendered = json.dumps(json_schema(), indent=2) + "\n"
    assert_golden("executor/json_schema.json", rendered, update=update_golden)


@pytest.mark.parametrize("spec", SCHEMA, ids=lambda spec: spec.name)
def test_json_schema_ranges_agree_with_the_schema(spec: ParamSpec) -> None:
    prop = json_schema()["properties"][spec.name]
    assert (prop["minimum"], prop["maximum"]) == (spec.minimum, spec.maximum)


def test_a_document_valid_for_the_schema_always_parses_unclamped() -> None:
    document = {spec.name: spec.maximum for spec in SCHEMA}
    assert parse_params(document).clamped == ()


def test_request_limits_narrow_the_grammar_but_stay_inside_the_schema() -> None:
    from retone.executor import REQUEST_LIMITS

    properties = json_schema(REQUEST_LIMITS)["properties"]
    for spec in SCHEMA:
        low, high = REQUEST_LIMITS[spec.name]
        assert spec.minimum <= low <= high <= spec.maximum
        assert (properties[spec.name]["minimum"], properties[spec.name]["maximum"]) == (low, high)
    assert json_schema()["properties"]["saturation"]["maximum"] == 100


def test_the_instruction_states_every_request_limit() -> None:
    from retone.executor import INSTRUCTION, REQUEST_LIMITS

    for name, (low, high) in REQUEST_LIMITS.items():
        assert f"{name} {low} to {high}" in INSTRUCTION
