"""Entry point: measure the reconstruction ceiling or score prediction files."""

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
from retone.eval.predictions import load_predictions, score_predictions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the reconstruction ceiling or score predicted parameters"
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--source-paths", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    records = load_records(args.examples)
    paths = load_source_paths(args.source_paths)
    if args.predictions is not None:
        score = score_predictions(
            records, load_predictions(args.predictions), paths, split=args.split
        )
        print(json.dumps(asdict(score), indent=2))
        return 0
    result = reconstruction_residual(records, paths, split=args.split, limit=args.limit)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
