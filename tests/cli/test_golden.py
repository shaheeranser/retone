"""Golden CLI output: the parameter table and JSON are reviewable artifacts."""

from __future__ import annotations

from io import StringIO

from golden_support import assert_golden
from retone.cli.render import render_json, render_table
from retone.executor import Params

SAMPLE = Params(
    exposure=18,
    temperature=-6,
    tint=0,
    contrast=12,
    highlights=-15,
    shadows=8,
    saturation=4,
    vibrance=10,
    sharpen_amount=25,
)


def test_table_golden(update_golden: bool) -> None:
    out = StringIO()
    render_table(SAMPLE, out)
    assert_golden("cli/params_table.txt", out.getvalue(), update=update_golden)


def test_json_golden(update_golden: bool) -> None:
    out = StringIO()
    render_json(SAMPLE, out)
    assert_golden("cli/params.json", out.getvalue(), update=update_golden)
