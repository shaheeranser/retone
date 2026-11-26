"""Render the measured results table: configurations down, metrics across.

Rows follow the order the design fixes (fitted parameters, base, fine-tuned f16,
Q8_0, Q4_K_M); the caller supplies a row per configuration it actually measured,
and a missing metric prints as an em dash rather than an invented number.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

MISSING = "—"
CSV_HEADER = ("model", "mean_delta_e", "parameter_mae", "latency_seconds", "peak_rss_bytes")


@dataclass(frozen=True)
class ResultRow:
    label: str
    mean_delta_e: float | None = None
    parameter_mae: float | None = None
    latency_seconds: float | None = None
    peak_rss_bytes: int | None = None


def _number(value: float | None, digits: int = 2) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def _latency(seconds: float | None) -> str:
    return MISSING if seconds is None else f"{seconds:.2f} s"


def _memory(byte_count: int | None) -> str:
    return MISSING if byte_count is None else f"{byte_count / (1024 * 1024):.0f} MB"


def results_markdown(rows: Sequence[ResultRow]) -> str:
    """A GitHub-flavoured markdown table with one row per measured configuration."""
    lines = [
        "| Model | Mean ΔE | Parameter MAE | Latency | Peak RAM |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        delta_e = _number(row.mean_delta_e)
        mae = _number(row.parameter_mae)
        latency = _latency(row.latency_seconds)
        memory = _memory(row.peak_rss_bytes)
        lines.append(f"| {row.label} | {delta_e} | {mae} | {latency} | {memory} |")
    return "\n".join(lines) + "\n"


def write_results_csv(rows: Sequence[ResultRow], out: TextIO) -> None:
    """The same rows as machine-readable CSV; absent metrics stay empty."""
    writer = csv.writer(out)
    writer.writerow(CSV_HEADER)
    for row in rows:
        writer.writerow(
            [
                row.label,
                "" if row.mean_delta_e is None else row.mean_delta_e,
                "" if row.parameter_mae is None else row.parameter_mae,
                "" if row.latency_seconds is None else row.latency_seconds,
                "" if row.peak_rss_bytes is None else row.peak_rss_bytes,
            ]
        )
