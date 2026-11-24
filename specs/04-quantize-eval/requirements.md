# 04 — Quantize and evaluate: Requirements

Depends on `01-executor`, `02-dataset`, `03-finetune`.

## Quantization

- **QE-QNT-1** — THE SYSTEM SHALL merge the LoRA, convert to GGUF, and produce
  Q4_K_M and Q8_0 variants, keeping `mmproj` at f16.
- **QE-QNT-2** — THE SYSTEM MAY build an importance matrix from task data.
- **QE-QNT-3** — THE SYSTEM SHALL serve the model with `llama-server` using the
  parameter JSON Schema so output is always valid JSON.

## Metrics

- **QE-MET-1** — THE SYSTEM SHALL compute mean CIEDE2000 ΔE against the held-out
  target at full resolution over the test split.
- **QE-MET-2** — THE SYSTEM SHALL compute per-parameter mean absolute error
  against fitted parameters.
- **QE-MET-3** — THE SYSTEM SHALL use the `01` executor for every render.

## Benchmark

- **QE-BEN-1** — THE SYSTEM SHALL measure latency and peak RAM by running the
  real model on the target laptop, not by estimate.
- **QE-BEN-2** — THE SYSTEM SHALL report model time and full-resolution render
  time separately, on a 24 MP photo.
- **QE-BEN-3** — THE SYSTEM SHALL produce the results table with rows: fitted
  parameters (ceiling), base prompted, fine-tuned f16, Q8_0, Q4_K_M; columns:
  mean ΔE, parameter MAE, latency, peak RAM.

## Thumbnail validity

- **QE-THM-1** — THE SYSTEM SHALL fit parameters on the full-resolution image
  and on its 448 px copy, apply both to the full-resolution image, and report
  the mean ΔE between the two results.
