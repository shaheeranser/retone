from __future__ import annotations

import pytest

from retone.train.gate import verdict


@pytest.mark.parametrize(
    ("model", "identity", "expected"),
    [(5.0, 10.0, True), (8.5, 10.0, True), (8.6, 10.0, False), (12.0, 10.0, False)],
)
def test_gate_verdict(model: float, identity: float, expected: bool) -> None:
    assert verdict(model, identity, 0.85) is expected
