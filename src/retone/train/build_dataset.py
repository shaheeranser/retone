"""Entry point: build the training dataset from a photo source manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from retone.data import BuildConfig, DatasetError, build_dataset, load_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build retone training examples")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--examples-per-photo", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        items = load_manifest(args.manifest)
        summary = build_dataset(
            items,
            BuildConfig(
                output_dir=args.out,
                seed=args.seed,
                test_fraction=args.test_fraction,
                workers=args.workers,
                examples_per_photo=args.examples_per_photo,
            ),
        )
    except (DatasetError, OSError, ValueError) as err:
        print(f"build failed: {err}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
