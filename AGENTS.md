# AGENTS.md

Instructions for AI coding agents working in this repository. This file is
tool-agnostic: nothing in it assumes a particular agent product, vendor, or
editor integration.

The committed, authoritative standards are
[`specs/style-guide.md`](./specs/style-guide.md) (how code should read) and
[`specs/agent-practices.md`](./specs/agent-practices.md) (how work is done).
This file is the entry point that points at them. Start with
[`specs/architecture.md`](./specs/architecture.md) for the system overview and
module map.

## Read the spec before you implement

`specs/` is the project's design documentation and is committed. Before
implementing a milestone's tasks, read that milestone's documents:

- `specs/<milestone>/requirements.md` — testable acceptance criteria (EARS-style
  "WHEN X, THE SYSTEM SHALL Y") with stable requirement IDs.
- `specs/<milestone>/design.md` — rationale, interfaces, failure modes. **Read
  this first.**
- `specs/<milestone>/tasks.md` — the checklist, each task citing the
  requirement IDs it satisfies.

Milestones are numbered and build on each other: `00-scaffolding`,
`01-executor`, `02-dataset`, `03-finetune`, `04-quantize-eval`, `05-cli`,
`06-submission`, `07-extensions`. A later design may reference an earlier one;
earlier ones never depend on later ones.

Do not silently resolve an open decision. Each `design.md` ends with a numbered
`OD-xx` list. If your work depends on one, surface it rather than picking a
value you cannot justify.

## Build, test, lint

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/ruff format --check .      # must report nothing to reformat
.venv/bin/mypy                       # strict
.venv/bin/pytest                     # offline; integration tests skipped by default
```

All of these must pass before a change is done. CI runs the same commands on
push and pull request. The CLI framework and distribution are still open
(`00-scaffolding` OD-02), so there is no console script or build artifact yet.

## Python code style

Project-wide rules are in [`specs/style-guide.md`](./specs/style-guide.md); the
essentials:

- One concern per module; run the neatcode placement protocol before adding code.
- Full type annotations, strict type checking, no `Any` without a reason.
- Comments explain why, never what; each module starts with a one-line
  responsibility docstring; no banners.
- Inject inference clients, writers, clocks and RNGs; never reach for globals.
- Validate untrusted data (model output, files, CLI input) at the boundary.
- No dependency where the standard library, NumPy or OpenCV suffice. Record
  additions and their licenses in `specs/architecture.md` §7.

## Package layout

```text
src/retone/executor/    apply(), parameter schema, ranges
src/retone/data/        FiveK prep, label fitting, augmentation
src/retone/train/       fine-tuning and export scripts (run on Colab)
src/retone/eval/        delta-E and parameter metrics, benchmarks
src/retone/inference/   llama-server client, thumbnailing, output validation
src/retone/cli/         flags, commands, rendering, wiring
```

Dependency direction is `cli → inference → executor`; `data`, `eval` and `train`
call the executor. The authoritative map is
[`specs/architecture.md`](./specs/architecture.md) §4.

## Invariants you must respect

1. The model only emits numbers; deterministic code applies them. No generative
   operation, ever.
2. One `apply()` and one parameter schema serve fitting, evaluation and the CLI.
3. Parameters are clamped to the schema, the original is never modified, and
   invalid model output falls back to defaults.
4. stdout carries only results; diagnostics go to stderr.

## Testing expectations

- Table-driven tests by default; hypothesis property tests for the executor.
- Golden files for rendered text and JSON.
- Deterministic and offline: fixed seeds, fake inference client, no network, no
  GPU. Tests needing the real model are marked `integration` and skipped by
  default.
- Bug fixes include a regression test that fails before the fix.

## Commit hygiene

Do not add `Co-authored-by` trailers for agents or tooling. Commits should
reflect the humans responsible for the change. Use small, labelled commits
(`feat(scope):`, `fix(scope):`, `docs:`, `test:`, `chore:`, `refactor:`, `ci:`)
and leave the build green at every commit. Enable the enforcing hook with
`git config core.hooksPath scripts/hooks`.

## Reports

Every milestone run ends with `reports/<milestone>/<run>.md` (gitignored). See
`specs/agent-practices.md` §5.
