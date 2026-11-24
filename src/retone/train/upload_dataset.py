"""Upload the built dataset to a private Hugging Face dataset; token from the environment only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _rows(examples_path: Path, image_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in examples_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            {
                "id": record["id"],
                "split": record["split"],
                "image": str((image_root / record["image"]).resolve()),
                "instruction": record["instruction"],
                "answer": record["answer"],
                "kind": record["kind"],
                "photo_id": record["source"]["photo_id"],
            }
        )
    if not rows:
        raise ValueError(f"{examples_path} contains no examples")
    return rows


def upload_dataset(
    examples_path: Path,
    image_root: Path,
    repo_id: str,
    token: str,
    *,
    private: bool = True,
) -> str:
    """Push the examples with encoded thumbnails; returns the repository id."""
    from datasets import Dataset, Image

    dataset = Dataset.from_list(_rows(examples_path, image_root)).cast_column("image", Image())
    dataset.push_to_hub(repo_id, private=private, token=token)
    return repo_id


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload the retone dataset to Hugging Face")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true", help="push as public (default private)")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2
    try:
        repo_id = upload_dataset(
            args.examples, args.image_root, args.repo_id, token, private=not args.public
        )
    except (OSError, ValueError) as err:
        print(f"upload failed: {err}", file=sys.stderr)
        return 1
    print(repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
