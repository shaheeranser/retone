# 03 — Fine-tune: Design

Read [`../architecture.md`](../architecture.md) §6 first.

## 1. Purpose

Teach a small vision-language model to map a thumbnail to the parameter JSON
whose application gets closest to the expert edit. This milestone carries the
most schedule risk (new architecture support, Colab availability), so it begins
with a gate that can still change the base model.

## 2. Model choice

Primary Qwen3.5-2B (natively multimodal, Apache 2.0 per model card); speed
fallback Qwen3.5-0.8B; tooling fallback Qwen2.5-VL-3B, which Unsloth and
llama.cpp have supported longer. Qwen3.5 uses a Gated DeltaNet hybrid
architecture, so Unsloth fine-tuning and llama.cpp vision inference must both be
confirmed before committing. Qwen3.5 small models are verbose by default, so
thinking is disabled. Unofficial community distills are avoided.

Optional distillation: a larger Qwen model generates varied instruction
phrasings (for `07`), or a 2B model is distilled into 0.8B if speed requires it.

## 3. Training recipe

QLoRA through Unsloth: rank 16, learning rate 2e-4, 1-2 epochs as the starting
point. The example format is fixed: 448 px thumbnail (optionally plus a 1:1
detail crop), a fixed short instruction, and the parameter JSON as integers.
Checkpoints are pushed to Hugging Face every few hundred steps.

## 4. Execution model

Code is edited locally and pushed to GitHub. A self-contained shell entry,
`scripts/train_remote.sh`, clones the repo on any GPU host, installs the `train`
extra, pulls the private dataset from Hugging Face, and runs
`python -m retone.train.finetune`. The same command works in a Colab free
notebook, a Kaggle notebook, or an SSH GPU box; no Colab-specific API or CLI is
assumed. It ends by pushing checkpoints, the LoRA adapter and the merged model;
GGUF conversion and quantization are handled in `04`.

## 5. Failure modes

- Prediction collapse to dataset averages: variance check, coarser bins, more
  augmentation.
- Colab session lost: frequent checkpoints.
- A100/L4 unavailable: everything sized for a T4.
- Qwen3.5 unsupported in tooling: gate falls back to Qwen2.5-VL-3B.
- Invalid JSON from the model: constrained decoding at inference, schema
  validation and default fallback in `inference`.

## Open decisions

- **OD-01** — Final base model after the gate (Qwen3.5-2B vs 0.8B vs
  Qwen2.5-VL-3B).
- **OD-02** — Whether to include the 1:1 detail crop in the prompt (cost vs
  benefit); links to `02` OD-04.
- **OD-03** — Hyperparameters beyond the starting point (rank, LR, epochs,
  bins), decided from validation behaviour.
- **OD-04** — Compute. Resolved 2026-11-24: target a 16 GB T4 on any available
  host (Colab free, Kaggle, or an SSH GPU box). The entry script is
  host-agnostic; Colab Pro is not required, and Google AI Pro is unrelated.
  A paid hourly host is the fallback if free capacity is unavailable.
