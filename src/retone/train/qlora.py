"""Unsloth QLoRA adapter: loading, training, checkpointing, generation and upload.

The training extra is optional and GPU-only, so every heavy import is local to the
function that needs it and PyTorch objects are typed as `Any` at this seam.
"""

from __future__ import annotations

import inspect
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from retone.data.errors import DatasetError
from retone.executor import Params, json_schema
from retone.train.distributed import is_main
from retone.train.formatting import TrainingExample, to_vision_record
from retone.train.parsing import parse_generated_params
from retone.train.selection import select_rows

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Rows decoded per Dataset chunk. Bounds peak memory while building the training
# set; large enough that concatenation is a handful of Arrow files, not 100.
VISION_CHUNK = 512


@dataclass(frozen=True)
class TrainSettings:
    model_name: str
    dataset_repo: str
    output_repo: str
    max_seq_length: int = 2048
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8
    lora_rank: int = 16
    save_steps: int = 200
    max_steps: int = -1
    max_new_tokens: int = 128
    seed: int = 3407
    resume: bool = False
    output_dir: Path = Path("outputs")
    private: bool = False
    push: bool = True
    merge: bool = True


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class ImageHandle:
    """A position in the Hub dataset, decoded on demand instead of held decoded."""

    dataset: Any
    index: int


def _messages(record: TrainingExample, image: Any) -> list[dict[str, Any]]:
    """The Unsloth vision conversation for one example."""
    return list(to_vision_record(replace(record, image=image))["messages"])


def _vision_dataset_from_handles(records: Sequence[TrainingExample]) -> Any:
    """A real Dataset with correctly encoded images, built in bounded chunks.

    Two constraints pull in opposite directions:

    - TRL only accepts a `datasets.Dataset`, so a list of plain dicts is not
      enough;
    - building one eagerly decodes every thumbnail into RAM (~12 GB for 26k
      rows) and the host OOM-killer ends the run.

    `Dataset.map` was the obvious way to stream it, but `datasets` re-serialises
    the *nested* image value inside `messages` as the PIL object's `__dict__`,
    which is not a usable image. `Dataset.from_list` encodes it correctly as
    `{path, bytes}`. So build in chunks and concatenate: each chunk holds only
    `chunk_size` decoded images, and the result is one genuine Dataset.
    """
    from datasets import Dataset, concatenate_datasets

    parts = []
    total = len(records)
    step = max(1, total // 10)
    for start in range(0, total, VISION_CHUNK):
        rows = [
            {"messages": _messages(record, _as_pil(record.image))}
            for record in records[start : start + VISION_CHUNK]
        ]
        parts.append(Dataset.from_list(rows))
        # Visible progress: a silent 25k-row build looks hung from a notebook.
        done = min(start + VISION_CHUNK, total)
        if done % step < VISION_CHUNK or done == total:
            print(f"  encoded {done}/{total} rows", file=sys.stderr, flush=True)
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _vision_dataset_from_list(records: Sequence[TrainingExample]) -> Any:
    from datasets import Dataset

    rows = [{"messages": _messages(record, _as_pil(record.image))} for record in records]
    return Dataset.from_list(rows)


def _as_vision_records(records: Sequence[TrainingExample]) -> Any:
    """A real datasets.Dataset for TRL, built without an in-memory image list."""
    if records and all(isinstance(record.image, ImageHandle) for record in records):
        return _vision_dataset_from_handles(records)
    return _vision_dataset_from_list(records)


def _example(dataset: Any, index: int, row: Any) -> TrainingExample:
    return TrainingExample.from_mapping({**row, "image": ImageHandle(dataset, index)})


def load_examples_from_hub(
    settings: TrainSettings, token: str, *, select_train: bool = False
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    """Pull the private dataset and split rows by their recorded `split` column.

    The image column is never materialised here: `list(dataset)` would decode
    every thumbnail up front and exhaust the host's RAM before training starts.
    """
    from datasets import load_dataset

    # Pull every shard once and reuse the local cache on restart. The earlier
    # `force_redownload` re-fetched ~477 MB on every run (and per rank under
    # DDP); it was added for a stale empty revision, so that case is retried
    # explicitly below instead of paying the download every time.
    def _load(download_mode: str) -> Any:
        return load_dataset(settings.dataset_repo, token=token, download_mode=download_mode)

    try:
        bundle = _load("reuse_dataset_if_exists")
        cached = bundle["train"] if "train" in bundle else bundle[next(iter(bundle))]
        if len(cached) == 0:
            raise DatasetError(("cached revision is empty",))
    except Exception:  # dataset is an untrusted external boundary
        try:
            bundle = _load("force_redownload")
        except Exception as err:
            raise DatasetError(
                (
                    f"cannot load dataset {settings.dataset_repo!r}: {err}. "
                    "Run `python -m retone.train.upload_dataset ...` first and check "
                    "that the repo contains parquet files.",
                )
            ) from err
    dataset = bundle["train"] if "train" in bundle else bundle[next(iter(bundle))]
    if len(dataset) == 0:
        raise DatasetError(
            (
                f"dataset {settings.dataset_repo!r} has no rows. "
                "Re-run `python -m retone.train.upload_dataset ...`.",
            )
        )
    rows = list(dataset.remove_columns("image"))
    train_positions = [(index, row) for index, row in enumerate(rows) if row.get("split") != "test"]
    if select_train:
        candidates = [row for _, row in train_positions]
        keep = {id(row) for row in select_rows(candidates, seed=settings.seed)}
        train_positions = [(index, row) for index, row in train_positions if id(row) in keep]
    train = [_example(dataset, index, row) for index, row in train_positions]
    test = [
        _example(dataset, index, row)
        for index, row in enumerate(rows)
        if row.get("split") == "test"
    ]
    if not train:
        raise DatasetError(("the Hugging Face dataset has no training rows",))
    return tuple(train), tuple(test or train)


def load_model(
    model_name: str,
    *,
    max_seq_length: int = 2048,
    lora: bool = True,
    lora_rank: int = 16,
    seed: int = 3407,
) -> LoadedModel:
    """Load the base model in 4-bit and wrap it in a QLoRA adapter when training."""
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name, max_seq_length=max_seq_length, load_in_4bit=True
    )
    if lora:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
            target_modules="all-linear",
        )
    return LoadedModel(model=model, tokenizer=tokenizer)


def _as_pil(image: Any) -> Any:
    """Normalise to a PIL image, keeping its on-disk format where possible.

    `convert("RGB")` on an image that is already RGB returns a new in-memory
    image with no `format`, so `datasets` re-encodes it as PNG: ~450 KB and
    ~340 ms per row instead of ~85 KB and ~100 ms from the source JPEG. Only
    convert when the mode actually differs.
    """
    if isinstance(image, ImageHandle):
        decoded = image.dataset[image.index]["image"]
        # The Hub Image() column yields a decoded image that still knows its
        # format; converting an already-RGB image would drop that and force a
        # PNG re-encode.
        if getattr(decoded, "mode", "RGB") != "RGB":
            return decoded.convert("RGB")
        return decoded
    if isinstance(image, Path):
        from PIL import Image

        with Image.open(image) as handle:
            return handle.convert("RGB")
    if type(image).__module__.startswith("PIL."):
        return image.convert("RGB") if image.mode != "RGB" else image
    return image


def _bf16_supported() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _latest_checkpoint(directory: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if path.is_dir() and step.isdigit():
            candidates.append((int(step), path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _restore_checkpoint(settings: TrainSettings, token: str) -> Path | None:
    """Use a local checkpoint, or pull the latest checkpoint folder from HF."""
    local = _latest_checkpoint(settings.output_dir)
    if local is not None or not settings.resume:
        return local

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        snapshot_download(
            repo_id=settings.output_repo,
            repo_type="model",
            token=token,
            local_dir=str(settings.output_dir),
            allow_patterns=["checkpoint-*/*"],
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return _latest_checkpoint(settings.output_dir)


def _ensure_repo(api: Any, repo_id: str, private: bool) -> None:
    """Create the repo and enforce its visibility.

    `create_repo(..., exist_ok=True)` leaves an existing repo's visibility
    alone, so a repo first created while private stays private and keeps
    counting against the free private-storage quota. Flip it explicitly.
    """
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    update = getattr(api, "update_repo_settings", None)
    if update is not None:
        update(repo_id=repo_id, private=private)


def _upload_folder(folder: Path, repo_id: str, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _ensure_repo(api, repo_id, private)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), token=token)


def train_once(
    settings: TrainSettings, records: Sequence[TrainingExample], token: str
) -> LoadedModel:
    """Train one QLoRA run, push checkpoints, then push the adapter and merged model."""
    import unsloth  # noqa: F401  # must be imported before trl/transformers
    from huggingface_hub import HfApi
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    api = HfApi(token=token)
    if is_main() and settings.push:
        _ensure_repo(api, settings.output_repo, settings.private)
        _ensure_repo(api, f"{settings.output_repo}-merged", settings.private)

    loaded = load_model(
        settings.model_name,
        max_seq_length=settings.max_seq_length,
        lora=True,
        lora_rank=settings.lora_rank,
        seed=settings.seed,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _restore_checkpoint(settings, token) if is_main() else None
    if checkpoint is not None:
        print(f"resuming from {checkpoint}", file=sys.stderr)
    bf16 = _bf16_supported()
    arguments = SFTConfig(
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps,
        learning_rate=settings.learning_rate,
        warmup_steps=5,
        logging_steps=1,
        save_steps=settings.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=settings.seed,
        output_dir=str(settings.output_dir),
        report_to="none",
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        # Only rank 0 pushes; every rank uploading the same checkpoint would
        # race on the Hub and multiply the egress.
        push_to_hub=is_main() and settings.push,
        hub_model_id=settings.output_repo if is_main() and settings.push else None,
        hub_strategy="every_save",
        hub_private_repo=settings.private,
        hub_token=token if is_main() and settings.push else None,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": loaded.model,
        # Without response masking the loss covers the constant instruction and the model
        # memorises it; these markers are Qwen2.5-VL's chat template.
        "data_collator": UnslothVisionDataCollator(
            loaded.model,
            loaded.tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        "train_dataset": _as_vision_records(records),
        "args": arguments,
    }
    tokenizer_keyword = (
        "processing_class"
        if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs[tokenizer_keyword] = loaded.tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    output = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    metrics = getattr(output, "metrics", {}) or {}
    runtime = metrics.get("train_runtime")
    steps_per_second = metrics.get("train_steps_per_second")
    if runtime is not None:
        print(
            f"training metrics: runtime={runtime:.1f}s, "
            f"steps_per_second={steps_per_second}, steps={metrics.get('train_steps')}",
            file=sys.stderr,
        )

    if not is_main():
        # Secondary ranks train and stop; rank 0 owns the push, the merge and
        # the local directory, which would otherwise be written twice at once.
        return loaded

    if not settings.merge:
        return loaded
    if settings.push:
        loaded.model.push_to_hub(settings.output_repo, private=settings.private, token=token)
    merged_dir = settings.output_dir / "merged"
    loaded.model.save_pretrained_merged(
        str(merged_dir), loaded.tokenizer, save_method="merged_16bit"
    )
    if settings.push:
        _upload_folder(merged_dir, f"{settings.output_repo}-merged", token, settings.private)
        shutil.rmtree(merged_dir, ignore_errors=True)
    return loaded


def _json_prefix(tokenizer: Any) -> Any:
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError:
        print(
            "lm-format-enforcer is not installed; generating without a JSON constraint",
            file=sys.stderr,
        )
        return None
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(json_schema()))


def _generate_one(
    loaded: LoadedModel, example: TrainingExample, max_new_tokens: int, prefix_fn: Any
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example.instruction},
            ],
        }
    ]
    prompt = loaded.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = loaded.tokenizer(
        _as_pil(example.image), prompt, add_special_tokens=False, return_tensors="pt"
    )
    inputs = inputs.to(loaded.model.device)
    generated = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=False,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    prompt_length = inputs["input_ids"].shape[1]
    decoded = loaded.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
    return str(decoded[0])


def predict(
    handle: object, records: Sequence[TrainingExample], limit: int, *, max_new_tokens: int = 128
) -> tuple[Params, ...]:
    """Generate and parse parameters for at most `limit` examples."""
    from unsloth import FastVisionModel

    loaded = handle
    if not isinstance(loaded, LoadedModel):
        raise TypeError(f"expected a LoadedModel, got {type(handle).__name__}")
    FastVisionModel.for_inference(loaded.model)
    prefix_fn = _json_prefix(loaded.tokenizer)
    predictions: list[Params] = []
    for example in records[:limit]:
        text = _generate_one(loaded, example, max_new_tokens, prefix_fn)
        predictions.append(parse_generated_params(text))
    return tuple(predictions)
