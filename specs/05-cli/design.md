# 05 — CLI: Design

Read [`../architecture.md`](../architecture.md) §3-4 first.

## 1. Purpose

A one-command "enhance" that anyone familiar with a terminal can run offline.
The CLI owns flags, exit codes and output formatting and nothing else; model
access lives in `inference`, pixels in `executor`.

## 2. Flow

```text
retone photo.jpg
  -> load (RGB, bit depth kept)
  -> thumbnail 448 px (linear-light downscale)
  -> inference client -> llama-server -> JSON
  -> validate/clamp (executor schema) or defaults
  -> executor.apply(full-res)
  -> write output; print result on stdout
```

## 3. Boundaries

`cli` imports `inference` and `executor`. Argument-parsing callbacks only
translate arguments into a call; business logic is in library functions so a
later GUI (see `07`) can reuse it. The inference client is injected so tests use
a fake that returns fixed JSON.

## 4. Output contract

stdout carries only the result (paths written, or JSON when `--json`). Warnings
about clamping or fallback go to stderr. Exit codes are stable and documented:
success, usage error, input error, model-unavailable, write error.

## 5. Failure modes

- Model server down or model missing: clear message, non-zero exit.
- Invalid output: clamp/default, warn, continue.
- Unsupported or corrupt image: input error exit code.
- Existing output file: refuse unless an explicit overwrite flag is given.

## Open decisions

- **OD-01** — CLI framework. Resolved 2026-11-26 (owner): `argparse`. The command
  is deliberately small (one positional list plus flags), so a third-party
  framework adds a dependency for no gain; stdlib keeps the license register
  short and matches the existing `retone.eval` entry point. The command installs
  as a `retone` console script (`pyproject.toml`) and also runs as
  `python -m retone`.
- **OD-02** — How the model runs. Resolved 2026-11-26 (owner wanted the model "in
  the tool"): the model is shipped as a GGUF beside the tool, not reimplemented in
  Python; the client talks to a running `llama-server` over HTTP, and
  `scripts/serve.sh` launches it. This keeps the heavy ML stack out of the Python
  package (the laptop needs only llama.cpp), keeps the client a swappable seam for
  tests, and still gives one logical tool: `retone photo.jpg` works once the
  server is up. True in-process bindings (`llama-cpp-python`) remain a later swap
  behind the same `InferenceClient` interface; they add a compiled dependency and
  are not needed for the submission. The schema is sent per request as the
  response grammar, so it stays a single source of truth.
- **OD-03** — Distribution: PyPI via `uv tool`/`pipx`, a single binary, or both;
  how the GGUF is fetched (`hf download` vs bundled). Deferred to `06-submission`.
- **OD-04** — Output naming and overwrite policy. Resolved 2026-11-26 (owner):
  default is a `_retone` suffix beside the input (`photo.jpg` -> `photo_retone.jpg`),
  an explicit `--out` overrides it, and an existing output is refused unless
  `--force` is given. The input file is never written.
- **OD-05** — Before/after output. Resolved 2026-11-26 (owner): both layouts via
  `--compare side-by-side|split|both`; `both` writes two images. Default is off so
  the one-command path writes only the enhanced image.
