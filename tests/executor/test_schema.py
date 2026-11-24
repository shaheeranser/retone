from __future__ import annotations

import pytest

from retone.executor.schema import SCHEMA, SPEC_BY_NAME, ParamSpec


def test_schema_names_are_unique_and_indexed() -> None:
    names = [spec.name for spec in SCHEMA]
    assert len(names) == len(set(names))
    assert set(SPEC_BY_NAME) == set(names)


@pytest.mark.parametrize("spec", SCHEMA, ids=lambda spec: spec.name)
def test_default_and_identity_lie_inside_the_range(spec: ParamSpec) -> None:
    assert spec.contains(spec.default)
    assert spec.contains(spec.identity)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-500, -100), (-100, -100), (0, 0), (100, 100), (101, 100)],
    ids=["far-below", "at-minimum", "inside", "at-maximum", "just-above"],
)
def test_clamp_limits_to_the_range(value: int, expected: int) -> None:
    assert SPEC_BY_NAME["exposure"].clamp(value) == expected


def test_sharpen_amount_is_never_negative() -> None:
    assert SPEC_BY_NAME["sharpen_amount"].clamp(-5) == 0
