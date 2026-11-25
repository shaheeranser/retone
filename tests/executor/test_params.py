from __future__ import annotations

import pytest

from retone.executor.errors import ParamError
from retone.executor.params import Params, parse_params
from retone.executor.schema import SCHEMA


def test_params_fields_match_the_schema() -> None:
    assert tuple(Params().as_dict()) == tuple(spec.name for spec in SCHEMA)


def test_params_default_to_schema_defaults() -> None:
    assert Params().as_dict() == {spec.name: spec.default for spec in SCHEMA}


def test_coarsen_rounds_and_clamps() -> None:
    coarse = Params(exposure=23, contrast=-13, sharpen_amount=97).coarsen(5)
    assert coarse.exposure == 25
    assert coarse.contrast == -15
    assert coarse.sharpen_amount == 95


def test_coarsen_rejects_invalid_step() -> None:
    with pytest.raises(ValueError):
        Params().coarsen(0)


@pytest.mark.parametrize(
    ("raw", "expected", "clamped"),
    [
        ({}, {}, ()),
        ({"exposure": 20}, {"exposure": 20}, ()),
        ({"exposure": 500}, {"exposure": 100}, ("exposure",)),
        ({"contrast": -101, "tint": 7}, {"contrast": -100, "tint": 7}, ("contrast",)),
        ({"sharpen_amount": -1}, {"sharpen_amount": 0}, ("sharpen_amount",)),
    ],
    ids=[
        "empty-uses-defaults",
        "in-range",
        "clamp-high",
        "clamp-low-with-valid",
        "no-negative-sharpen",
    ],
)
def test_parse_fills_defaults_and_reports_clamping(
    raw: dict[str, object], expected: dict[str, int], clamped: tuple[str, ...]
) -> None:
    parsed = parse_params(raw)
    assert parsed.params == Params(**expected)
    assert parsed.clamped == clamped


@pytest.mark.parametrize(
    ("raw", "fragments"),
    [
        ({"bogus": 1}, ["unknown parameter 'bogus'"]),
        ({"exposure": 1.5}, ["exposure must be an integer"]),
        ({"exposure": "3"}, ["exposure must be an integer"]),
        ({"exposure": True}, ["exposure must be an integer"]),
        (
            {"bogus": 1, "tint": 2.0, "shadows": None},
            ["unknown parameter 'bogus'", "tint must be an integer", "shadows must be an integer"],
        ),
    ],
    ids=["unknown", "float", "string", "bool", "all-problems-reported-together"],
)
def test_parse_rejects_bad_input_naming_every_field(
    raw: dict[str, object], fragments: list[str]
) -> None:
    with pytest.raises(ParamError) as info:
        parse_params(raw)
    assert len(info.value.problems) == len(fragments)
    for fragment in fragments:
        assert any(fragment in problem for problem in info.value.problems)
