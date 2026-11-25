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
CUDA, through llama.cpp. Training happens on a GPU host; the laptop only runs the
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

Early. The deterministic executor and the dataset builder are implemented and
tested. **The `retone` command does not exist yet**; it arrives in `05-cli`. The
usage above describes the intended tool, not what runs now.

| Milestone | State |
|---|---|
| `00-scaffolding` | in progress (dependency licences still being recorded) |
| `01-executor` | implemented and tested |
| `02-dataset` | implemented; first real PD12M build and upload pending |
| `03-finetune` | implemented; GPU gate and training run pending |
| `04-quantize-eval` | human review implemented; quantization and benchmarks pending |
| `05-cli` | specified |
| `06-submission` | specified |
| `07-extensions` (GUI, sliders, learned sharpening) | deferred |

Planned results (reconstruction ceiling, base versus fine-tuned, f16 / Q8_0 /
Q4_K_M, laptop latency and memory) will be published here once measured.

## How the project is built

The work is spec-driven. Each milestone in [`specs/`](./specs/) has
`requirements.md` (testable criteria with stable IDs), `design.md` (rationale,
interfaces, open decisions) and `tasks.md`. Read
[`specs/architecture.md`](./specs/architecture.md) first, then
[`specs/style-guide.md`](./specs/style-guide.md) and
[`specs/agent-practices.md`](./specs/agent-practices.md). [`AGENTS.md`](./AGENTS.md)
is the entry point for AI coding agents, and [`CONTRIBUTING.md`](./CONTRIBUTING.md)
covers contributing. [`specs/data-layout.md`](./specs/data-layout.md) defines the
private input and generated-output trees.

## Development

Python 3.11 or newer.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
git config core.hooksPath scripts/hooks   # labelled commits, no agent trailers
```

The lint, type-check and test commands are listed in `AGENTS.md`.

### Training on a remote GPU

Training needs a CUDA GPU with about 16 GB of VRAM. `scripts/train_remote.sh` is
host-agnostic: it clones the repo, installs the `train` extra, pulls the private
dataset and runs the fine-tune. The same command works in a Colab free notebook,
a Kaggle notebook, or an SSH GPU box.

```sh
export DATASET_REPO=you/retone-dataset
export OUTPUT_REPO=you/retone-qwen
export HF_TOKEN=...                 # never commit this
bash scripts/train_remote.sh        # add --max-steps 60 for a smoke run
```

Validate the dataset and formatter without a GPU or token:

```sh
python -m retone.train.fetch_pd12m --limit 5000 --out artifacts/data/raw/pd12m
python -m retone.train.build_dataset \
  --manifest artifacts/data/raw/pd12m/manifest.jsonl \
  --out artifacts/data/processed/synthetic
python -m retone.train.finetune --examples artifacts/data/processed/examples.jsonl \
  --image-root artifacts/data/processed --dry-run --bin-step 5
```

Private before/after edits from the owner's library are collected into
gitignored storage, never published:

```sh
python -m retone.train.collect_pairs --source ~/pictures --dest artifacts/data/human
```

Directories matching `_*` (backup or corrupt folders) are skipped by default;
repeat `--exclude` to choose other patterns. Edits are rotated back to the
original's orientation. These pairs are **evaluation only**: no labels are fitted
and they never enter training.

Build a human-review bundle from prediction files:

```sh
python -m retone.eval review \
  --pairs artifacts/data/human/pairs.jsonl \
  --prediction tuned=predictions-tuned.jsonl \
  --out artifacts/eval/human/2026-11-25
```

Open `review.html`, then fill `ratings.csv`. ΔE in `metrics.json` is a diagnostic;
human judgement is the acceptance signal. See
[`specs/data-layout.md`](./specs/data-layout.md).

Checkpoints, the adapter and the merged model are pushed to
`$OUTPUT_REPO` and `$OUTPUT_REPO-merged`. Baseline and fine-tuned predictions
are scored on the held-out split with:

```sh
python -m retone.eval score --examples artifacts/data/processed/examples.jsonl \
  --source-paths artifacts/data/processed/source_paths.json --predictions predictions.jsonl
```

## License

MIT, see [`LICENSE`](./LICENSE). The licenses of the base model, training data and
dependencies are tracked in `specs/architecture.md` section 7. The training corpus
is PD12M: public domain/CC0 images under the CDLA-Permissive-2.0 dataset licence,
with Megalith-10M (MIT dataset, CC0/public-domain Flickr links) as a fallback.
MIT-Adobe FiveK is research-only, and Unsplash's Terms of Service forbids using
their images in machine-learning datasets, so neither is used.
