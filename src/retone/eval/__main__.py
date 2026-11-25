"""Entry point: ceiling, prediction scoring, or human review bundles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from retone.eval.ceiling import (
    load_records,
    load_source_paths,
    reconstruction_residual,
)
from retone.eval.human import load_human_pairs, load_predictions_by_label
from retone.eval.predictions import load_predictions, score_predictions
from retone.eval.review import build_review_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="retone evaluation commands")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ceiling = subcommands.add_parser("ceiling", help="measure synthetic reconstruction ΔE")
    ceiling.add_argument("--examples", type=Path, required=True)
    ceiling.add_argument("--source-paths", type=Path, required=True)
    ceiling.add_argument("--split", default="test")
    ceiling.add_argument("--limit", type=int, default=None)

    score = subcommands.add_parser("score", help="score prediction files on synthetic records")
    score.add_argument("--examples", type=Path, required=True)
    score.add_argument("--source-paths", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--split", default="test")

    review = subcommands.add_parser("review", help="build a human review bundle")
    review.add_argument("--pairs", type=Path, required=True)
    review.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="prediction file for a model label; repeat for blind A/B",
    )
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ceiling":
        result = reconstruction_residual(
            load_records(args.examples),
            load_source_paths(args.source_paths),
            split=args.split,
            limit=args.limit,
        )
        print(json.dumps(asdict(result), indent=2))
        return 0
    if args.command == "score":
        score = score_predictions(
            load_records(args.examples),
            load_predictions(args.predictions),
            load_source_paths(args.source_paths),
            split=args.split,
        )
        print(json.dumps(asdict(score), indent=2))
        return 0

    bundle = build_review_bundle(
        load_human_pairs(args.pairs),
        load_predictions_by_label(args.prediction),
        args.out,
        seed=args.seed,
    )
    print(json.dumps({"out": str(bundle.out_dir), "pairs": bundle.pairs, "labels": bundle.labels}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
