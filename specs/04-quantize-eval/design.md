# 04 — Quantize and evaluate: Design

Read [`../architecture.md`](../architecture.md) §8 first.

## 1. Purpose

Show three things in one table: fine-tuning helped, quantization cost little, and
the model runs on the weak device. Every number is measured, never estimated.

## 2. Quantization path

Merge the LoRA into the base, convert to GGUF on the GPU host, quantize to
Q4_K_M and Q8_0, keep `mmproj` at f16. Optionally compare OpenVINO INT4 on the
laptop.
`llama-server` runs with the parameter JSON Schema as its grammar, so malformed
output cannot occur at generation time; `inference` still validates and falls
back to defaults, because the server is a boundary.

## 3. Metrics

ΔE is CIEDE2000 against the held-out target at full resolution. Parameter MAE is
against the constructed labels (the model's actual training target). Metrics live
in `eval/` and depend only on the executor; they never import training code.

## 4. Benchmark protocol

Latency and peak RAM are measured on the target laptop (i7-1185G7, 16 GB, no
CUDA) with the real quantized model. Model time and render time are reported
separately because they scale differently: the model sees a fixed-size
thumbnail, while rendering scales with megapixels.

## 5. Thumbnail validity

A mean ΔE of about 1-2 or below between thumbnail-fitted and full-res-fitted
renders shows reasoning on a thumbnail costs essentially nothing; the threshold
is stated in the report, not enforced by a test.

## 6. Failure modes

- Quantization degrades quality: Q8_0 retained as the fallback row.
- Too slow on the laptop: smaller thumbnail, Q4, Qwen3.5-0.8B, OpenVINO.
- Server not available in tests: fake inference client; real server only in
  integration tests.

## 7. Human-edited evaluation

The owner's 23 before/after edits are a separate held-out set in
`artifacts/data/human/` (see `../data-layout.md`). They are never trained on and
no labels are fitted. For each pair, a prediction file supplies parameters; the
executor renders them on the full-resolution before image, and CIEDE2000 to the
human after image is recorded as a diagnostic.

The review bundle under `artifacts/eval/human/<run>/` keeps `original/`, `model/`
and `reference/` side by side with `review.html`, `ratings.csv`, `metrics.json`
and `manifest.json`. When two models are compared, their columns are labelled A/B in
a seeded order and `manifest.json` holds the mapping, so the reviewer does not
judge the model by name.

Human review is the acceptance signal: the owner records, per pair and model,
whether it looks better than the original, how close it is to the reference, and
whether they would use it. ΔE is evidence, never the sole gate; a different but
pleasing edit must not be marked a failure by one number.

## Open decisions

- **OD-01** — Whether to build an imatrix and whether to compare OpenVINO INT4.
- **OD-02** — The ΔE threshold at which the thumbnail check is called a pass,
  and the minimum fine-tune-vs-baseline margin counted as "clearly beats".
- **OD-03** — Which iGPU backend (Vulkan or SYCL) or CPU-only to use for
  reported latency.
