"""Copy human-edited before/after pairs from a folder tree into a work directory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from retone.data.collect import copy_pairs, discover_pairs
from retone.data.errors import DatasetError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect before/after photo pairs")
    parser.add_argument("--source", type=Path, required=True, help="folder to search recursively")
    parser.add_argument(
        "--dest", type=Path, default=Path("data/raw/pairs"), help="copy destination"
    )
    parser.add_argument("--dry-run", action="store_true", help="report without copying")
    args = parser.parse_args(argv)

    try:
        discovery = discover_pairs(args.source)
        summary = copy_pairs(discovery, args.dest, dry_run=args.dry_run)
    except DatasetError as err:
        print(f"collect failed: {err}", file=sys.stderr)
        return 2

    for path in discovery.unmatched[:20]:
        print(f"no original for {path.relative_to(args.source)}", file=sys.stderr)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
