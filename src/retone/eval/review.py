"""Build a self-contained human-review bundle: images, HTML, ratings CSV and metrics."""

from __future__ import annotations

import csv
import html
import json
import os
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from retone.eval.human import HumanPair, HumanReport, render_predictions
from retone.executor import Params

CSV_HEADER = (
    "pair_id",
    "rater",
    "model_label",
    "looks_better_than_original",
    "closeness_to_reference_1_5",
    "would_use",
    "notes",
)


@dataclass(frozen=True)
class ReviewBundle:
    out_dir: Path
    pairs: int
    labels: dict[str, str]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _display_labels(labels: Sequence[str], seed: int) -> dict[str, str]:
    if len(labels) == 1:
        return {labels[0]: "model"}
    ordered = list(labels)
    random.Random(seed).shuffle(ordered)
    return {label: chr(ord("A") + index) for index, label in enumerate(ordered)}


def _image(relative: str, alt: str) -> str:
    return (
        f'<figure><img src="{html.escape(relative)}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


def _review_html(
    pairs: Sequence[HumanPair],
    display: Mapping[str, str],
    reports: Mapping[str, HumanReport],
    seed: int,
) -> str:
    models = list(display)
    blind = len(models) > 1
    scores = {
        label: {score.pair_id: score.delta_e for score in report.per_pair}
        for label, report in reports.items()
    }
    headers = "".join(f"<th>{html.escape(display[label])}</th>" for label in models)
    rows: list[str] = []
    for pair in pairs:
        before_name = f"{pair.pair_id}{pair.before.suffix.lower()}"
        after_name = f"{pair.pair_id}{pair.after.suffix.lower()}"
        cells = [
            _image(f"original/{before_name}", "Original"),
            _image(f"reference/{after_name}", "Human reference"),
        ]
        for label in models:
            model_path = (
                f"model/{pair.pair_id}.jpg"
                if len(models) == 1
                else f"model/{display[label]}/{pair.pair_id}.jpg"
            )
            delta = scores[label][pair.pair_id]
            cells.append(_image(model_path, f"{display[label]} (ΔE {delta:.2f})"))
        meta = html.escape(pair.pair_id) + (
            f"<br><small>{html.escape(pair.variant)}</small>" if pair.variant else ""
        )
        rows.append(f"<tr><td>{meta}</td>{''.join(f'<td>{cell}</td>' for cell in cells)}</tr>")

    blind_note = (
        "<p>Columns A and B are shuffled and do not reveal the model. "
        "The mapping is in <code>manifest.json</code>.</p>"
        if blind
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>retone human review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #111; color: #eee; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #444; padding: 0.5rem; vertical-align: top; }}
img {{ max-width: 280px; height: auto; display: block; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 0.85rem; color: #aaa; margin-top: 0.3rem; }}
code {{ background: #222; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>retone human review</h1>
<p>For each row, compare the original, your edit, and the model output. Fill
<code>ratings.csv</code>; the ΔE values are diagnostics, not a pass/fail gate.</p>
<p><a href="ratings.csv">Open ratings.csv</a> · run seed {seed}</p>
{blind_note}
<table>
<thead><tr><th>Pair</th><th>Original</th><th>Human reference</th>{headers}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</body>
</html>
"""


def build_review_bundle(
    pairs: Sequence[HumanPair],
    predictions_by_label: Mapping[str, Mapping[str, Params]],
    out_dir: Path,
    *,
    seed: int = 0,
) -> ReviewBundle:
    """Write original/reference/model images and the HTML, CSV and metrics files."""
    if not predictions_by_label:
        raise ValueError("at least one prediction set is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    display = _display_labels(list(predictions_by_label), seed)

    for pair in pairs:
        _link_or_copy(
            pair.before, out_dir / "original" / f"{pair.pair_id}{pair.before.suffix.lower()}"
        )
        _link_or_copy(
            pair.after, out_dir / "reference" / f"{pair.pair_id}{pair.after.suffix.lower()}"
        )

    reports: dict[str, HumanReport] = {}
    for label, predictions in predictions_by_label.items():
        model_dir = out_dir / "model" if len(display) == 1 else out_dir / "model" / display[label]
        reports[label] = render_predictions(tuple(pairs), predictions, model_dir)

    (out_dir / "review.html").write_text(
        _review_html(pairs, display, reports, seed), encoding="utf-8"
    )
    with (out_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for pair in pairs:
            for label in predictions_by_label:
                writer.writerow([pair.pair_id, "", display[label], "", "", "", ""])

    metrics = {
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "scores": {
                    label: {score.pair_id: score.delta_e for score in reports[label].per_pair}[
                        pair.pair_id
                    ]
                    for label in reports
                },
            }
            for pair in pairs
        ],
        "by_label": {
            label: {
                "display": display[label],
                "count": report.count,
                "mean_delta_e": report.mean_delta_e,
                "median_delta_e": report.median_delta_e,
            }
            for label, report in reports.items()
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "run": str(out_dir),
        "seed": seed,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "before": str(pair.before),
                "after": str(pair.after),
                "variant": pair.variant,
            }
            for pair in pairs
        ],
        "labels": display,
        "display_to_actual": {shown: label for label, shown in display.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReviewBundle(out_dir=out_dir, pairs=len(pairs), labels=dict(display))
