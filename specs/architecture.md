# Architecture

System overview, module map and dependency rules for `retone`. Read this before
any milestone design.

## 1. What retone is

`retone` is a command-line tool that makes a photo look better without
inventing content. A small fine-tuned, quantized Qwen vision-language model
looks at a thumbnail of the photo and outputs a short JSON of adjustment
values (exposure, white balance, contrast, saturation and so on). A
deterministic OpenCV executor applies those values to the full-resolution
original. The model chooses numbers; code applies them.

It runs fully offline on a weak laptop (target: Intel i7-1185G7, 16 GB RAM, no
CUDA) through llama.cpp. Training happens on Colab; the laptop only runs the
finished GGUF model.

Contrast with cloud "AI enhance": only global parametric operations are allowed,
so every change is visible and reversible, and nothing can be hallucinated.

Prior art: HDRNet (Gharbi et al., 2017) predicts adjustments at low resolution
and applies them at full resolution; image-adaptive 3D-LUTs (Zeng et al., 2020)
do the same with a lookup table.

## 2. Design principles

1. The model chooses numbers; deterministic code applies them.
2. No hallucination by construction: only global, parametric operations
   (exposure, white balance, tone curve, saturation, unsharp mask). No AI
   denoise, upscaling or inpainting.
3. Reason on a thumbnail, render at full resolution. The model sees a 448 px
   thumbnail (plus an optional 1:1 detail crop).
4. Resolution-dependent operations use relative units: sharpening radius is a
   fraction of the image diagonal, converted to pixels at apply time.
5. One source of truth: the executor defines every formula; fitting,
   evaluation and the CLI all call it.

## 3. Data flow

```text
training (GPU host) Unsplash photos -> constructed pairs (executor) -> dataset on HF
                   -> QLoRA fine-tune -> merge -> GGUF + mmproj -> HF

inference (laptop) photo -> 448 px thumbnail -> llama-server (GGUF, JSON schema)
                   -> params -> validate/clamp -> executor.apply(full-res) -> output
```

## 4. Module map and dependency direction

```text
src/retone/executor/   apply(), parameter schema, ranges, colour-space maths
src/retone/data/       Unsplash corpus loading, pair construction, examples
src/retone/train/      fine-tuning and export entry scripts (run on any GPU host)
src/retone/eval/       delta-E and parameter metrics, benchmark runner
src/retone/inference/  llama-server client, thumbnailing, output validation
src/retone/cli/        argument parsing, command handlers, rendering, wiring
tests/                 mirrors src/retone/
specs/                 committed design record
reports/               gitignored run reports
```

Dependency direction:

```text
cli -> inference -> executor
data  -> executor
eval  -> executor
train -> data, executor
```

- `executor` imports nothing else from the project and no ML framework.
- `inference` knows how to talk to the model and validate its output; it does
  not edit images.
- `cli` composes the others and owns flags, exit codes and output formatting;
  business logic never lives in an argument-parsing callback.
- `train` and `data` may use training-only dependencies; `executor`, `eval`,
  `inference` and `cli` must not import them.
- The CLI is a thin layer over a library so a later GUI (see `07-extensions`)
  can reuse it without rework.

## 5. Load-bearing invariants

1. The model only emits numbers; no generative operation exists anywhere.
2. One `apply()` and one parameter schema serve fitting, evaluation and the CLI.
3. Parameters are clamped to schema ranges; the original is never modified;
   invalid model output falls back to defaults.
4. stdout carries only results; diagnostics go to stderr.
5. Executor internals: float32, RGB order after loading (OpenCV loads BGR),
   linear light for exposure and white balance, fixed operation order.

## 6. Models, data and artifacts

Code lives in GitHub; data and models live in Hugging Face repos while working;
GPU hosts are disposable compute (Colab free, Kaggle, or an SSH box) and the
training scripts make no host-specific assumption. The hackathon submission
requires a public model link, so what becomes public is decided in
`06-submission` OD-01.

| Artifact | Approx. size (2B) | Stored on |
|---|---|---|
| LoRA adapter | tens of MB | Hugging Face (backup) |
| Merged model, f16 | ~4-5 GB | Hugging Face (for re-quantizing) |
| GGUF, Q4_K_M | ~1.5 GB | Hugging Face + laptop |
| `mmproj`, f16 | hundreds of MB | Hugging Face + laptop |

Primary model: Qwen3.5-2B (Apache 2.0 per its model card; to verify, see §7).
Speed fallback Qwen3.5-0.8B; tooling fallback Qwen2.5-VL-3B. Thinking mode is
disabled for this task. Avoid unofficial community distills.

## 7. Licensing and dependency register

The project is MIT. That holds only if nothing forces otherwise. Each entry is
verified in `00-scaffolding` before the first release and recorded here with the
date and source.

| Component | Role | License | Status |
|---|---|---|---|
| Qwen3.5 weights | base model | Apache 2.0 (model card) | to verify |
| Unsloth | fine-tuning | to record | to verify |
| llama.cpp | inference | to record | to verify |
| OpenCV, NumPy | executor | to record | to verify |
| Unsplash photos | training corpus | [Unsplash License](https://unsplash.com/license): irrevocable, worldwide, free commercial use, modification allowed; bars compiling into a competing stock service | verified 2026-11-24 |
| `datasets`, `huggingface-hub`, Pillow (optional upload extra) | dataset upload | Apache-2.0, Apache-2.0, HPND | to verify |
| Unsloth, TRL, transformers, PEFT, accelerate (optional train extra) | fine-tuning | Apache-2.0 per project metadata | to verify |
| lm-format-enforcer (optional train extra) | constrained JSON generation | MIT | to verify |
| MIT-Adobe FiveK | **not used** | research-only, non-commercial | rejected 2026-11-24; see `02-dataset` design §1 |

## 8. Evaluation contract

All runs use the held-out Unsplash test split; delta-E is CIEDE2000 against the
held-out target at full resolution. The results table covers: reconstruction
ceiling (the analytic labels), base Qwen prompted, fine-tuned f16, Q8_0 and
Q4_K_M, each with mean delta-E, parameter MAE, laptop latency and peak RAM. A
thumbnail-validity check reports the mean delta-E between rendering from
thumbnail-sized and full-resolution measurements. Qwen time and full-resolution
render time are reported separately on a 24 MP photo.
