"""Host-agnostic fine-tuning entry: run on Colab, Kaggle, or an SSH GPU box."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from retone.data.errors import DatasetError
from retone.executor import Params
from retone.train import qlora
from retone.train.distributed import is_distributed, is_main
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
    parser.add_argument(
        "--private",
        action="store_true",
        help="push model repos private (default public, which has free storage)",
    )
    parser.add_argument("--no-push", action="store_true", help="keep results local; no Hub traffic")
    parser.add_argument(
        "--pilot", action="store_true", help="short gate run: no Hub traffic and no merge"
    )
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
        private=args.private,
        push=not (args.no_push or args.pilot),
        merge=not args.pilot,
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


def _strip_torchrun_args(argv: Sequence[str]) -> list[str]:
    """Drop the launcher's own flags, which argparse would reject.

    `torchrun` appends `--local-rank=<n>` (newer torch) or `--local_rank=<n>`
    (older) to the module's argv. Without this every DDP run exits 2 with
    "unrecognized arguments".
    """
    return [arg for arg in argv if not re.match(r"--local[-_]rank(=|\s|$)", arg)]


def main(argv: Sequence[str] | None = None) -> int:
    load_env()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(_strip_torchrun_args(raw))
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
            train, _test = qlora.load_examples_from_hub(settings, token, select_train=True)
    except DatasetError as err:
        print(f"dataset failed: {err}", file=sys.stderr)
        return 2

    policy = RetryPolicy(
        max_attempts=args.max_attempts,
        initial_bin_step=args.bin_step,
        min_std=args.min_std,
        sample_size=args.sample_size,
    )
    if is_distributed() and policy.max_attempts > 1:
        # A retry re-enters trainer.train() on every rank. If the ranks judged
        # collapse differently one would loop and the others would hang, so DDP
        # runs the single attempt and the collapse flag is read on rank 0 only.
        print(
            "distributed run: forcing a single attempt (no collapse retry)",
            file=sys.stderr,
        )

    def predict(handle: object, records: Sequence[TrainingExample], limit: int) -> Sequence[Params]:
        if not is_main():
            # Non-zero ranks must not generate; they report spread labels so the
            # collapse check does not trip and force a divergent retry.
            return [Params(exposure=-1), Params(exposure=1)]
        return qlora.predict(handle, records, limit, max_new_tokens=settings.max_new_tokens)

    attempts = 1 if is_distributed() else policy.max_attempts
    outcome = run_training(
        RetryPolicy(
            max_attempts=attempts,
            initial_bin_step=policy.initial_bin_step,
            min_std=policy.min_std,
            sample_size=policy.sample_size,
        ),
        train,
        train_once=lambda _step, records: qlora.train_once(settings, records, token),
        predict=predict,
    )
    print(json.dumps(asdict(outcome), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
