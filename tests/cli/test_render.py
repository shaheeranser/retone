"""Parameter rendering: the human table and the machine JSON, to an injected sink."""

from __future__ import annotations

import json
from collections.abc import Sequence
from io import StringIO

from retone.cli.render import render_json, render_table
from retone.executor import Params


def _render_table(params: Params, names: Sequence[str] | None = None) -> str:
    out = StringIO()
    render_table(params, out, names=names)
    return out.getvalue()


def test_identity_parameters_render_a_note() -> None:
    assert _render_table(Params()) == "no adjustments\n"


def test_non_identity_parameters_render_signed_values() -> None:
    text = _render_table(Params(exposure=25, contrast=-10, sharpen_amount=40))
    assert "+25" in text and "exposure" in text
    assert "-10" in text and "contrast" in text
    assert "+40" in text and "sharpen_amount" in text
    assert "saturation" not in text  # zero values are hidden


def test_names_filter_restricts_rows() -> None:
    text = _render_table(Params(exposure=5, contrast=7), names=["exposure"])
    assert "exposure" in text
    assert "contrast" not in text


def test_json_is_single_line_and_complete() -> None:
    out = StringIO()
    render_json(Params(exposure=5), out)
    line = out.getvalue()
    assert line.endswith("\n")
    assert line.count("\n") == 1
    payload = json.loads(line)
    assert payload["exposure"] == 5
    assert payload["sharpen_amount"] == 0
    assert set(payload) == set(Params().as_dict())
