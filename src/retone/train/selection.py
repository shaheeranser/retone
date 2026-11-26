"""Choose which dataset rows to train on: visible corrections, with few identity rows."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

# v1 trained on 33% identity and a fifth of mild 3-15 point edits, and greedy
# decoding answered all zeros. v2 dropped both identity and mild and the model
# never learned a small correction, so it could only answer identity or one
# large value. Keep every visible correction (synthetic and mild) and thin
# identity hard: it is a restraint signal, not a third of the data.
KEPT_KINDS = frozenset({"synthetic", "mild"})
IDENTITY_KEEP = 0.05


def select_rows(
    rows: Iterable[Mapping[str, Any]], *, identity_keep: float = IDENTITY_KEEP, seed: int = 3407
) -> list[Mapping[str, Any]]:
    """Keep every visible-correction row and a seeded fraction of identity rows."""
    rng = random.Random(seed)
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        kind = row.get("kind")
        if kind in KEPT_KINDS or (kind == "identity" and rng.random() < identity_keep):
            kept.append(row)
    return kept
