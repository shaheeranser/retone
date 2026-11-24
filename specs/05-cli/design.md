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

- **OD-01** — CLI framework; links to `00-scaffolding` OD-02.
- **OD-02** — How the model runs: the CLI starts and stops `llama-server` itself,
  talks to an already-running one, or embeds llama.cpp via bindings.
- **OD-03** — Distribution: PyPI via `uv tool`/`pipx`, a single binary, or both;
  how the GGUF is fetched (`hf download` vs bundled).
- **OD-04** — Output naming and overwrite policy defaults.
- **OD-05** — Whether the before/after output is a side-by-side image, a split,
  or both.
