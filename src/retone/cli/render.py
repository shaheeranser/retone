"""Render chosen parameters for a human (a table) or a machine (JSON), to an injected sink."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TextIO

from retone.executor import Params


def render_json(params: Params, out: TextIO) -> None:
    """One JSON object on a single line, so `retone ... --json | jq` works."""
    out.write(json.dumps(params.as_dict(), sort_keys=True) + "\n")


def render_table(params: Params, out: TextIO, *, names: Sequence[str] | None = None) -> None:
    """A fixed-width `name value` table of the non-identity parameters, or a note if none."""
    shown = [(name, value) for name, value in params.as_dict().items() if value != 0]
    if names is not None:
        wanted = set(names)
        shown = [(name, value) for name, value in shown if name in wanted]
    if not shown:
        out.write("no adjustments\n")
        return
    width = max(len(name) for name, _ in shown)
    for name, value in shown:
        out.write(f"{name:<{width}} {value:+d}\n")
