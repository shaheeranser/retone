"""Entry point: measure the reconstruction ceiling from a built dataset."""

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the full-resolution reconstruction ceiling"
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--source-paths", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    result = reconstruction_residual(
        load_records(args.examples),
        load_source_paths(args.source_paths),
        split=args.split,
        limit=args.limit,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
