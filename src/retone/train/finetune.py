"""Host-agnostic fine-tuning entry: run on Colab, Kaggle, or an SSH GPU box."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from retone.data.errors import DatasetError
from retone.train import qlora
from retone.train.env import load_env
from retone.train.formatting import TrainingExample, load_examples
from retone.train.orchestrate import RetryPolicy, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune retone on a GPU host")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-repo", help="private HF dataset id to pull")
    source.add_argument("--examples", type=Path, help="local examples.jsonl instead")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-repo", help="HF model repo for adapter and merged model")
    parser.add_argument("--model", default=qlora.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument(
        "--resume", action="store_true", help="resume from the latest local or HF checkpoint"
    )
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--min-std", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bin-step", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--public", action="store_true", help="push public (default private)")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate local examples without a GPU or token"
    )
    return parser


def _load_local(
    examples: Path, image_root: Path | None, bin_step: int
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    train = load_examples(examples, image_root=image_root, split="train", bin_step=bin_step)
    try:
        test = load_examples(examples, image_root=image_root, split="test", bin_step=bin_step)
    except DatasetError:
        test = train
    return train, test


def _settings(args: argparse.Namespace) -> qlora.TrainSettings:
    return qlora.TrainSettings(
        model_name=args.model,
        dataset_repo=args.dataset_repo or "local",
        output_repo=args.output_repo or "",
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        resume=args.resume,
        output_dir=args.output_dir,
        private=not args.public,
    )


def _dry_run(args: argparse.Namespace) -> int:
    if args.examples is None:
        print("--dry-run needs --examples and --image-root for an offline check", file=sys.stderr)
        return 2
    try:
        train, test = _load_local(args.examples, args.image_root, args.bin_step)
    except DatasetError as err:
        print(f"dataset check failed: {err}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "train": len(train),
                "test": len(test),
                "bin_step": args.bin_step,
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_env()
    args = _parser().parse_args(argv)
    if args.dry_run:
        return _dry_run(args)
    if not args.output_repo:
        print("--output-repo is required when training", file=sys.stderr)
        return 2
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2

    settings = _settings(args)
    try:
        if args.examples is not None:
            train, _test = _load_local(args.examples, args.image_root, args.bin_step)
        else:
            train, _test = qlora.load_examples_from_hub(settings, token)
    except DatasetError as err:
        print(f"dataset failed: {err}", file=sys.stderr)
        return 2

    policy = RetryPolicy(
        max_attempts=args.max_attempts,
        initial_bin_step=args.bin_step,
        min_std=args.min_std,
        sample_size=args.sample_size,
    )
    outcome = run_training(
        policy,
        train,
        train_once=lambda _step, records: qlora.train_once(settings, records, token),
        predict=lambda handle, records, limit: qlora.predict(
            handle, records, limit, max_new_tokens=settings.max_new_tokens
        ),
    )
    print(json.dumps(asdict(outcome), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
