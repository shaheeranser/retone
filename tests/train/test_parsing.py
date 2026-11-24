"""Generated-text parsing falls back to defaults on every defect."""

from __future__ import annotations

import pytest

from retone.executor import Params
from retone.train.parsing import parse_generated_params


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"exposure": 12, "contrast": -4}', Params(exposure=12, contrast=-4)),
        ('```json\n{"exposure": 12}\n```', Params(exposure=12)),
        ('Sure: {"exposure": 12} done', Params(exposure=12)),
        ("no json here", Params()),
        ('{"exposure": "12"}', Params()),
        ('{"not_a_param": 12}', Params()),
        ("{broken", Params()),
    ],
)
def test_parse_generated_params(text: str, expected: Params) -> None:
    assert parse_generated_params(text) == expected
