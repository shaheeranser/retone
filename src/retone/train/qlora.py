"""Unsloth QLoRA adapter: loading, training, checkpointing, generation and upload.

The training extra is optional and GPU-only, so every heavy import is local to the
function that needs it and PyTorch objects are typed as `Any` at this seam.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from retone.data.errors import DatasetError
from retone.executor import Params, json_schema
from retone.train.formatting import TrainingExample, to_vision_record
from retone.train.parsing import parse_generated_params

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


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
    private: bool = True


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_examples_from_hub(
    settings: TrainSettings, token: str
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    """Pull the private dataset and split rows by their recorded `split` column."""
    from datasets import load_dataset

    try:
        bundle = load_dataset(
            settings.dataset_repo,
            token=token,
            download_mode="force_redownload",
        )
    except Exception as err:  # dataset is an untrusted external boundary
        raise DatasetError(
            (
                f"cannot load dataset {settings.dataset_repo!r}: {err}. "
                "Run `python -m retone.train.upload_dataset ...` first and check "
                "that the repo contains parquet files.",
            )
        ) from err
    dataset = bundle["train"] if "train" in bundle else bundle[next(iter(bundle))]
    train: list[TrainingExample] = []
    test: list[TrainingExample] = []
    for row in dataset:
        example = TrainingExample.from_mapping(row)
        (test if row.get("split") == "test" else train).append(example)
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
            modules_to_save=["lm_head", "embed_tokens"],
        )
    return LoadedModel(model=model, tokenizer=tokenizer)


def _as_pil(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Path):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


def _as_vision_records(records: Sequence[TrainingExample]) -> list[dict[str, Any]]:
    return [to_vision_record(replace(record, image=_as_pil(record.image))) for record in records]


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
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

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


def _upload_folder(folder: Path, repo_id: str, token: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
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
    api.create_repo(repo_id=settings.output_repo, private=settings.private, exist_ok=True)
    api.create_repo(
        repo_id=f"{settings.output_repo}-merged", private=settings.private, exist_ok=True
    )

    loaded = load_model(
        settings.model_name,
        max_seq_length=settings.max_seq_length,
        lora=True,
        lora_rank=settings.lora_rank,
        seed=settings.seed,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _restore_checkpoint(settings, token)
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
        push_to_hub=True,
        hub_model_id=settings.output_repo,
        hub_strategy="every_save",
        hub_private_repo=settings.private,
        hub_token=token,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": loaded.model,
        "data_collator": UnslothVisionDataCollator(loaded.model, loaded.tokenizer),
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
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    loaded.model.push_to_hub(settings.output_repo, private=settings.private, token=token)
    merged_dir = settings.output_dir / "merged"
    loaded.model.save_pretrained_merged(
        str(merged_dir), loaded.tokenizer, save_method="merged_16bit"
    )
    _upload_folder(merged_dir, f"{settings.output_repo}-merged", token, settings.private)
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
