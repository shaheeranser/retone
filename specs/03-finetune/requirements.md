# 03 — Fine-tune: Requirements

Depends on `01-executor` and `02-dataset`.

## Toolchain gate

- **FT-GATE-1** — BEFORE committing to a base model, THE SYSTEM SHALL demonstrate
  that a QLoRA step with image inputs runs in Unsloth for Qwen2.5-VL-3B and then
  for Qwen3.5-2B.
- **FT-GATE-2** — THE SYSTEM SHALL demonstrate that the model converts to GGUF
  (language model plus `mmproj`) and answers an image prompt in llama.cpp.
- **FT-GATE-3** — THE SYSTEM SHALL time one inference on the target laptop and
  accept the model only if it takes a few seconds per image.
- **FT-GATE-4** — WHEN any gate fails, THE SYSTEM SHALL record the failure and
  switch to the next candidate (Qwen3.5-2B, then Qwen3.5-0.8B) before further
  work.

## Baseline

- **FT-BASE-1** — THE SYSTEM SHALL evaluate the base model, prompted, with
  JSON-constrained output on the test set, using the `04` metrics.

## Training

- **FT-TRN-1** — THE SYSTEM SHALL train on examples of thumbnail plus fixed short
  instruction to parameter JSON, with thinking disabled and integer values
  rounded to a step of 5.
- **FT-TRN-2** — THE SYSTEM SHALL fine-tune with QLoRA via Unsloth, starting
  from LoRA rank 16, learning rate 2e-4, 1-2 epochs.
- **FT-TRN-3** — THE SYSTEM SHALL push checkpoints to Hugging Face every few
  hundred steps so a lost Colab session loses little work.
- **FT-TRN-4** — THE SYSTEM SHALL fit within a 16 GB T4 (small images, short
  outputs) so GPU availability does not block the run.
- **FT-TRN-5** — THE SYSTEM SHALL check prediction variance across images, and
  WHEN predictions barely vary, THE SYSTEM SHALL retrain with coarser bins
  (steps of 5).
- **FT-TRN-6** — THE SYSTEM SHALL resume from the latest checkpoint, local or on
  Hugging Face, when explicitly asked, so a lost GPU session does not restart
  from zero.

## Execution

- **FT-RUN-1** — THE SYSTEM SHALL provide a self-contained entry script that
  clones the repo, pulls the dataset from Hugging Face, trains, and pushes
  artifacts back, with the token read only from the environment or Colab
  Secrets.
- **FT-RUN-2** — THE SYSTEM SHALL be runnable with one command on a Colab free
  notebook, a Kaggle notebook, or an SSH GPU host.
- **FT-RUN-3** — THE SYSTEM SHALL NOT depend on a Colab-specific API or CLI.

## Quality

- **FT-QLT-1** — THE SYSTEM SHALL show a fine-tuned mean ΔE that clearly beats
  the baseline on the test set.
