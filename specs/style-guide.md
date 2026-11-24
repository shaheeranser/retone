# Style Guide

Project-wide rules, applying to every milestone. Milestone-specific rationale
lives in `specs/<milestone>/design.md`; this file covers how code should *read*,
not what it should do. It adapts the project owner's Go and TypeScript standards
to Python, with the [neatcode](https://github.com/shaheeranser/neatcode)
placement and refactoring practices folded in.

Tooling named here (ruff, mypy, pytest, hypothesis, uv) is the working
assumption recorded as OD-01 in `00-scaffolding/tasks.md`; the rules themselves
do not depend on it.

## 1. One concern per file

Each module holds one type, one adapter, or one clearly bounded piece of logic.
If you cannot name a file for what it does without reaching for "utils",
"helpers", or "common", it holds more than one concern.

**Bad** — a grab-bag whose only shared trait is that the functions are small:

```python
# src/retone/utils.py
def srgb_to_linear(img): ...
def load_fivek_pair(path): ...
def delta_e(a, b): ...
def format_params_table(params): ...
```

**Good** — one module per concern, named for the concern:

```python
src/retone/executor/color_space.py   # sRGB <-> linear light
src/retone/data/fivek.py             # FiveK pair loading
src/retone/eval/delta_e.py           # CIEDE2000 metric
src/retone/cli/render.py             # parameter table rendering
```

A type and the functions that belong to it are still one concern. The test is
whether a reader can describe the module's job in one sentence without saying
"and".

## 2. Placement before writing

Before adding code, run the neatcode placement protocol:

1. State the responsibility in one sentence.
2. Search for the module that already owns it; the module map is
   `specs/architecture.md` §4.
3. Match by responsibility, not by the file that happens to be open.
4. If nothing fits, create a module with a concrete name and a one-line
   responsibility docstring.

State the decision before writing: "Adding X to `path` because it owns Y." If
the sentence needs an "and", split the work.

## 3. Comments describe behaviour and intent

A comment says what the code accomplishes for the system and why — especially
when the reason is not obvious. It must not restate operations the reader can
already see.

**Bad** — narrating the code back to the reader:

```python
# Loop over the channels and multiply each by its gain.
for i in range(3):
    out[..., i] *= gains[i]
```

**Good** — explaining why, which the code cannot convey:

```python
# Gains apply in linear light; applying them to gamma-encoded values would shift
# hue as well as brightness.
out *= gains
```

Every module starts with a one-line responsibility docstring. That is the only
block comment allowed. No banners, ASCII rules, or section headers.

## 4. Prefer self-explanatory code

If a better name or a clearer structure can carry the intent, do that instead of
writing a comment. A comment that exists only to explain a name is a signal to
rename.

```python
# Bad
if abs(ev) > 3.0:  # exposure is out of range

# Good
if not schema.exposure.contains(ev):
```

## 5. Python shape

- Python 3.11+. Full type annotations on every function; strict type checking
  must pass. No `Any` without a comment saying why; no blanket `# type: ignore`.
- `from __future__ import annotations` at the top of each module, right after the
  module docstring.
- Public API of a package is what its `__init__.py` re-exports and nothing else;
  `__init__.py` holds no logic. Import from the module that owns the symbol.
- Prefer plain functions and frozen dataclasses (or pydantic models at I/O
  boundaries). Add a class only when it owns state or a lifetime.
- Arrays: say the dtype and layout in the name or type (`rgb_f32`, `bgr_u8`).
  Convert at the boundary once; never carry BGR past `load_image`.
- No mutation of inputs. `apply()` returns a new array; the original image is
  never modified.
- Soft thresholds, used as signals rather than verdicts: 400 code lines per
  file, 50 lines per function, cyclomatic complexity 10, nesting depth 4. Run
  neatcode's `audit.py` to rank files.

## 6. Tests are part of the design

Tests are written with the code, not bolted on afterwards.

- **Table-driven by default.** `pytest.mark.parametrize` rows carry a behaviour
  id and fixed inputs.
- **Property-based tests for the executor.** Use hypothesis for invariants:
  identity parameters leave the image unchanged, output stays in range, results
  are deterministic, every parameter clamps to its schema range, the original
  array is never mutated, radii scale with image diagonal.
- **Golden files for anything that renders text or JSON** (CLI output, parameter
  JSON), stored in `tests/golden/` and regenerated with an explicit flag
  (`pytest --update-golden`), so a diff is a reviewable artifact.
- **Deterministic and offline.** Fixed seeds, frozen inputs, no network, no model
  download, no GPU. A fake inference client stands in for `llama-server`. A test
  that needs the real model or network is an integration test, marked
  `@pytest.mark.integration`, and skipped by default.
- **Regression tests.** A bug fix includes a test that fails before the fix.
- Tests mirror the package layout: `tests/executor/test_color_space.py`.

## 7. Inject what you depend on

A function takes what it needs — an inference client, a clock, a writer, a
random generator — so a test can pass a fake and capture output. Reaching for
global `time.time()`, `sys.stdout`, `os.environ` or an unseeded RNG inside logic
makes behaviour untestable.

```python
# Good: the sink is a seam a test can fill.
def render_params(params: Params, out: TextIO) -> None: ...

# Bad: the destination is welded in.
def render_params(params: Params) -> None:
    print(...)
```

Construction and wiring belong in `cli/` and in the `train/` entry scripts.

## 8. Errors and validation

- Validate untrusted external data at the boundary — model output against the
  parameter JSON schema, files on disk, CLI arguments — and trust the validated
  type inward.
- Model output is untrusted: invalid or out-of-range output is clamped or falls
  back to defaults and is reported on stderr; it never crashes the CLI.
- Raise typed errors with a stable code and a human message; chain with
  `raise … from err` so the original cause is never discarded.
- When several independent things can be wrong, collect and report them
  together rather than failing on the first.
- No bare `except:` and no `except Exception` outside the top-level CLI handler.

## 9. Dependencies

The standard library and NumPy/OpenCV come first; add a dependency only when it
does something they genuinely cannot, and record the reason and its license in
`specs/architecture.md`. Training-only dependencies (Unsloth, TRL, transformers)
live in an optional extra and are never imported by `executor`, `eval` or `cli`.

## 10. Concurrency and performance

- Parallel label fitting uses process pools with a bounded worker count and
  per-item seeds, so a run is reproducible regardless of core count.
- Never load a full dataset into memory; stream or memory-map.
- Report latency and peak memory by measurement, not estimate.
