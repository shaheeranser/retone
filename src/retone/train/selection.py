"""Choose which dataset rows to train on: visible corrections, with few identity rows."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

# The first run trained on rows where 33% were identity and 20% were 3-15 point
# changes that are invisible once rounded; greedy decoding then answered all zeros.
KEPT_KINDS = frozenset({"synthetic"})
IDENTITY_KEEP = 0.1


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
