"""Row selection keeps synthetic and mild rows and thins identity deterministically."""

from __future__ import annotations

from retone.train.selection import select_rows

ROWS = (
    [{"id": f"s{i}", "kind": "synthetic"} for i in range(10)]
    + [{"id": f"m{i}", "kind": "mild"} for i in range(10)]
    + [{"id": f"i{i}", "kind": "identity"} for i in range(1000)]
)


def test_synthetic_and_mild_rows_are_kept() -> None:
    kinds = [row["kind"] for row in select_rows(ROWS)]
    assert kinds.count("synthetic") == 10
    assert kinds.count("mild") == 10


def test_identity_rows_are_thinned_to_about_the_fraction() -> None:
    identity = [row for row in select_rows(ROWS, identity_keep=0.1) if row["kind"] == "identity"]
    assert 60 < len(identity) < 140


def test_selection_is_deterministic() -> None:
    assert select_rows(ROWS, seed=1) == select_rows(ROWS, seed=1)
