"""Rendering of the measured results table as markdown and CSV."""

from __future__ import annotations

import csv
import io

from retone.eval.results import ResultRow, results_markdown, write_results_csv


def _rows() -> list[ResultRow]:
    return [
        ResultRow("fitted parameters", mean_delta_e=1.2, parameter_mae=0.0),
        ResultRow(
            "fine-tuned f16",
            mean_delta_e=5.25,
            parameter_mae=6.5,
            latency_seconds=1.234,
            peak_rss_bytes=1500 * 1024 * 1024,
        ),
    ]


def test_markdown_has_the_fixed_columns_and_order() -> None:
    table = results_markdown(_rows())
    lines = table.splitlines()
    assert lines[0] == "| Model | Mean ΔE | Parameter MAE | Latency | Peak RAM |"
    assert lines[2].startswith("| fitted parameters | 1.20 | 0.00 |")
    assert "1.23 s" in lines[3]
    assert "1500 MB" in lines[3]


def test_absent_metrics_render_as_an_em_dash() -> None:
    table = results_markdown([ResultRow("Q4_K_M")])
    assert table.splitlines()[2] == "| Q4_K_M | — | — | — | — |"


def test_csv_writes_numbers_and_leaves_absent_metrics_empty() -> None:
    buffer = io.StringIO()
    write_results_csv(_rows(), buffer)
    parsed = list(csv.reader(io.StringIO(buffer.getvalue())))
    assert parsed[0] == [
        "model",
        "mean_delta_e",
        "parameter_mae",
        "latency_seconds",
        "peak_rss_bytes",
    ]
    assert parsed[1] == ["fitted parameters", "1.2", "0.0", "", ""]
    assert parsed[2][3] == "1.234"
