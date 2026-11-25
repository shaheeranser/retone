"""Baseline inference: run the prompted base model on held-out examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from retone.data.errors import DatasetError
from retone.train import qlora
from retone.train.env import load_env
from retone.train.formatting import TrainingExample, load_examples


def _records(args: argparse.Namespace, token: str | None) -> tuple[TrainingExample, ...]:
    if args.examples is not None:
        return load_examples(
            args.examples, image_root=args.image_root, split=args.split, bin_step=1
        )
    if not token:
        raise DatasetError(("HF_TOKEN is required to pull the private dataset",))
    settings = qlora.TrainSettings(
        model_name=args.model, dataset_repo=args.dataset_repo or "", output_repo="baseline"
    )
    train, test = qlora.load_examples_from_hub(settings, token)
    return test if args.split == "test" else train


def main(argv: Sequence[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Score the prompted base model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-repo")
    source.add_argument("--examples", type=Path)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=qlora.DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    try:
        records = _records(args, token)
    except DatasetError as err:
        print(f"dataset failed: {err}", file=sys.stderr)
        return 2

    loaded = qlora.load_model(args.model, max_seq_length=args.max_seq_length, lora=False)
    limit = len(records) if args.limit is None else min(args.limit, len(records))
    predictions = qlora.predict(loaded, records, limit, max_new_tokens=args.max_new_tokens)
    args.output.write_text(
        "".join(
            json.dumps({"id": record.example_id, "params": guess.as_dict()}) + "\n"
            for record, guess in zip(records[:limit], predictions, strict=True)
        )
    )
    print(json.dumps({"predictions": len(predictions), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
