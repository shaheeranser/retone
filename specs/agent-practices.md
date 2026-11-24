# Agent practices

How work is done in this repository. These are project rules, not tool or
environment settings: they say what this codebase expects of anyone working on
it, and they are written for AI coding agents because that is often who is
implementing a milestone.

Normative code shape lives in [`style-guide.md`](./style-guide.md); the system
overview and module map live in [`architecture.md`](./architecture.md); the
design rationale lives in each milestone's `design.md`. Anything specific to one
person's machine or agent tooling is not a project rule and does not belong
here.

## 1. Read the spec before you implement

Each milestone in this directory is the design record: `requirements.md`
(testable acceptance criteria with stable IDs), `design.md` (rationale,
interfaces, failure modes — read this first), and `tasks.md` (the checklist,
each task citing the requirement IDs it satisfies). Implement against the spec,
not against a guess about what the feature should be.

Milestones build in order: a later `design.md` may reference an earlier one, but
an earlier one never depends on a later one. Preserve that direction when
editing.

## 2. Do not silently resolve an open decision

Every `design.md` ends with a numbered `OD-xx` list. These are choices the
project owner has reserved. Surface them — ask — rather than picking a value you
cannot justify. When one *is* decided, record the resolution and its reasoning
in the design, so the next reader does not have to rediscover it.

## 3. Ask when implementation demands it

If a task needs a decision the spec does not make, ask. Guessing produces work
that has to be undone, and buries a design choice inside an implementation.

## 4. Commits

- Small, focused commits made along the way, not one large commit at the end.
- Labelled, human-readable messages — `feat`, `fix`, `chore`, `docs`, `test`,
  `refactor`, `ci` — with an optional scope, e.g. `feat(executor): …`.
- One concern per commit, and a commit should build and pass tests on its own.
- No `Co-authored-by` trailers for agents or tooling. Commits reflect the
  humans responsible for the change. `scripts/hooks/commit-msg` enforces both
  the label and the missing trailer; enable it with
  `git config core.hooksPath scripts/hooks`.

## 5. Every milestone run leaves a report

A development run is not finished until its report exists:

```text
reports/<milestone>/<run>.md
```

For example `reports/01-executor/2026-11-25.md`. A report records what was
built, how it is shaped, how to run and test it, how it was validated on real
inputs, the commit walkthrough, and the known gaps. `reports/` is gitignored:
reports are working records, while `specs/` is the committed, normative
documentation.

## 6. Validate on real inputs

Green unit tests are necessary but not sufficient. A run is validated by driving
the feature end to end on genuine inputs with the real dependencies — real
photos through the executor, the real quantized model through the CLI — and
checking the result against expected behaviour. Always show before and after on
images the model never saw during training. Record the commands and results in
the run report.

## 7. Report status honestly

The README describes the product's *full intended scope*; it is not softened to
match what is implemented today. What exists versus what is still a gap belongs
in the run report's "known gaps". Never describe a capability as achieved when a
later milestone owns it, and never report a metric that was not measured.

## 8. Change as little as the spec requires

Make the smallest change that satisfies the requirement. Do not add features,
configurability, or abstractions the task did not ask for, and do not pre-build
for hypothetical future milestones. Delete code that is no longer used rather
than leaving compatibility shims. The milestone that needs something adds it.

## 9. Respect the boundaries

The layers are described in [`architecture.md`](./architecture.md) §4. In short:
`executor` depends on nothing else in the project; `data`, `train` and `eval`
call the executor; the CLI composes the inference client and the executor. Do
not let a feature reach across a boundary.

Four invariants are load-bearing and must not be bypassed:

1. **The model only emits numbers; deterministic code applies them.** No
   generative operation (AI denoise, upscaling, inpainting) exists anywhere.
2. **One source of truth.** Fitting, evaluation and the CLI import the same
   `apply()` and the same parameter schema. Never re-implement a formula.
3. **Parameters are always clamped to the schema, and the original is never
   modified.** Invalid model output falls back to defaults rather than crashing.
4. **stdout carries only results.** Diagnostics go to stderr, so
   `retone … --json | jq` always works.

## 10. Leave the build green

Before considering work done, the commands in `AGENTS.md` ("Build, test, lint")
all pass. Run them yourself; do not rely on CI to catch a failure you could have
seen locally.

## 11. Refactoring

Follow the neatcode procedure: pin behaviour with tests, extract one module at a
time, run the tests, and keep refactoring separate from behaviour changes. Never
rewrite a large module in a single pass, never weaken a failing test to make a
refactor pass, and do not add an abstraction for a hypothetical third case.

## 12. Colab and secrets

Training scripts run on disposable Colab VMs, so each one is self-contained:
it clones the repo, pulls data from Hugging Face, and pushes checkpoints and
artifacts back. Tokens come from the environment or Colab Secrets, never from
code, config files, or logs. Models and datasets never enter git.
