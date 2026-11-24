# 04 — Quantize and evaluate: Design

Read [`../architecture.md`](../architecture.md) §8 first.

## 1. Purpose

Show three things in one table: fine-tuning helped, quantization cost little, and
the model runs on the weak device. Every number is measured, never estimated.

## 2. Quantization path

Merge the LoRA into the base, convert to GGUF on Colab, quantize to Q4_K_M and
Q8_0, keep `mmproj` at f16. Optionally compare OpenVINO INT4 on the laptop.
`llama-server` runs with the parameter JSON Schema as its grammar, so malformed
output cannot occur at generation time; `inference` still validates and falls
back to defaults, because the server is a boundary.

## 3. Metrics

ΔE is CIEDE2000 against Expert C at full resolution. Parameter MAE is against the
fitted labels (the model's actual training target). Metrics live in `eval/` and
depend only on the executor; they never import training code.

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

## Open decisions

- **OD-01** — Whether to build an imatrix and whether to compare OpenVINO INT4.
- **OD-02** — The ΔE threshold at which the thumbnail check is called a pass,
  and the minimum fine-tune-vs-baseline margin counted as "clearly beats".
- **OD-03** — Which iGPU backend (Vulkan or SYCL) or CPU-only to use for
  reported latency.
