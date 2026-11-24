# retone

Offline photo enhancement that never invents content.

A small fine-tuned, quantized Qwen vision-language model looks at a thumbnail of
your photo and answers with a short list of numbers: exposure, white balance,
contrast, saturation and so on. A deterministic OpenCV executor applies those
numbers to the full-resolution original. The model chooses values. Code edits
pixels.

```text
photo -> 448 px thumbnail -> Qwen (GGUF, llama.cpp) -> parameters
      -> clamp and validate -> OpenCV at full resolution -> enhanced photo
```

Built for the [Qwen Local Hackathon 2026](https://github.com/junaiddshaukat/finetune-anywhere/tree/main/showcase/qwen-local-hackathon-2026).

## Why it is different

Cloud "AI enhance" tools use generative models that can add detail that was never
in the photo. retone only changes adjustment values, and only with global
operations (exposure, white balance, tone curve, saturation, unsharp mask). It has
no AI denoise, upscaling or inpainting, so every change is visible and
reversible, and the original file is never touched.

It runs on a weak laptop. The target is an Intel i7-1185G7 with 16 GB RAM and no
CUDA, through llama.cpp. Training happens on Colab; the laptop only runs the
finished model.

## Intended usage

```sh
retone photo.jpg                 # writes an enhanced copy beside the original
retone photo.jpg --params        # shows the chosen parameters without writing an image
retone photo.jpg --json          # parameters as JSON on stdout
retone photo.jpg --compare       # also writes a before/after image
```

Results go to stdout and diagnostics to stderr, so the output pipes cleanly.
Linux is the primary target, and the tool should work on macOS and Windows too.

## Status

Early. Today the repository holds the project standards, the design specs and an
empty package skeleton. **The `retone` command does not exist yet.** The usage
above describes the intended tool, not what runs now.

| Milestone | State |
|---|---|
| `00-scaffolding` | in progress |
| `01-executor` | specified |
| `02-dataset` | specified |
| `03-finetune` | specified |
| `04-quantize-eval` | specified |
| `05-cli` | specified |
| `06-submission` | specified |
| `07-extensions` (GUI, sliders, learned sharpening) | deferred |

Planned results (fitted-parameter ceiling, base versus fine-tuned, f16 / Q8_0 /
Q4_K_M, laptop latency and memory) will be published here once measured.

## How the project is built

The work is spec-driven. Each milestone in [`specs/`](./specs/) has
`requirements.md` (testable criteria with stable IDs), `design.md` (rationale,
interfaces, open decisions) and `tasks.md`. Read
[`specs/architecture.md`](./specs/architecture.md) first, then
[`specs/style-guide.md`](./specs/style-guide.md) and
[`specs/agent-practices.md`](./specs/agent-practices.md). [`AGENTS.md`](./AGENTS.md)
is the entry point for AI coding agents, and [`CONTRIBUTING.md`](./CONTRIBUTING.md)
covers contributing.

## Development

Python 3.11 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
git config core.hooksPath scripts/hooks   # labelled commits, no agent trailers
```

The lint, type-check and test commands are listed in `AGENTS.md`.

## License

MIT, see [`LICENSE`](./LICENSE). The licenses of the base model, training data and
dependencies are tracked in `specs/architecture.md` section 7. The MIT-Adobe FiveK
dataset terms are still being verified and may limit what derived data and model
weights can be published.
