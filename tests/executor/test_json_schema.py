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
