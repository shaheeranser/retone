"""Model-output validation: clamping keeps intent, garbage falls back to defaults."""

from __future__ import annotations

import pytest

from retone.executor import Params
from retone.inference.validation import validate_model_output


def test_valid_output_is_used_verbatim() -> None:
    result = validate_model_output({"exposure": 12, "contrast": -5})
    assert result.params == Params(exposure=12, contrast=-5)
    assert result.warnings == ()


def test_missing_names_take_defaults() -> None:
    result = validate_model_output({"saturation": 10})
    assert result.params == Params(saturation=10)
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ({"exposure": 500}, "exposure"),
        ({"sharpen_amount": -20}, "sharpen_amount"),
        ({"temperature": 101, "tint": -101}, "temperature"),
    ],
    ids=["exposure-high", "sharpen-negative", "two-clamped"],
)
def test_out_of_range_values_clamp_and_warn(raw: dict[str, int], name: str) -> None:
    result = validate_model_output(raw)
    assert any(name in warning for warning in result.warnings)
    assert result.params.as_dict()[name] in (-100, 0, 100)


def test_unknown_name_falls_back_to_defaults() -> None:
    result = validate_model_output({"exposure": 20, "brightness": 5})
    assert result.params == Params()
    assert len(result.warnings) == 1
    assert "defaults" in result.warnings[0]


def test_non_integer_value_falls_back_to_defaults() -> None:
    result = validate_model_output({"exposure": "ten"})
    assert result.params == Params()
    assert "defaults" in result.warnings[0]
