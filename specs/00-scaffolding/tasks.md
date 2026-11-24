# 00 — Scaffolding: Tasks

This milestone has no requirements or design file; it only creates the project
skeleton and settles the toolchain. Read
[`../architecture.md`](../architecture.md) and
[`../agent-practices.md`](../agent-practices.md) first.

## 1. Repository

- [x] **T-1.1** — Initialise git, MIT license, `.gitignore`.
- [x] **T-1.2** — Add the commit-message hook enforcing labelled commits and no
      agent trailers.
- [x] **T-1.3** — Add `AGENTS.md`, `CONTRIBUTING.md`, the PR template, and the
      `specs/` standards.
- [ ] **T-1.4** — Create the public GitHub repository `shaheeranser/retone` and
      push (needs owner confirmation before the first push).

## 2. Toolchain

- [ ] **T-2.1** — Add `pyproject.toml` with the `src/retone` package layout, the
      `retone` console script, and a `train` optional extra for training-only
      dependencies. *(OD-01, OD-02)*
- [ ] **T-2.2** — Configure lint, format and strict type checking; fix the exact
      commands and write them into `AGENTS.md` "Build, test, lint". *(OD-01)*
- [ ] **T-2.3** — Configure pytest with `integration` marker skipped by default,
      and the `--update-golden` option.
- [ ] **T-2.4** — Add a CI workflow running the commands from T-2.2 on push and
      pull request.
- [ ] **T-2.5** — Add empty package skeletons for the modules in
      `architecture.md` §4, each with a one-line responsibility docstring and an
      import test proving `executor` imports nothing else from the project.

## 3. Licensing

- [ ] **T-3.1** — Verify and record each license in `architecture.md` §7 with
      date and source, in particular the MIT-Adobe FiveK terms and whether the
      derived dataset and fine-tuned weights may be published. *(OD-03)*
- [ ] **T-3.2** — Add a `NOTICE` or README section stating third-party licenses.

## 4. Environment checks

- [x] **T-4.1** — Read the hackathon starter repo's CONTRIBUTING and
      showcase rules and record the required submission contents in
      `06-submission`. *(OD-05, resolved by `06-submission/requirements.md`)*
- [x] **T-4.2** — VLM base model confirmed in scope by the owner on
      2026-11-24. *(OD-04)*

## Open decisions

- **OD-01** — Python tooling. Working assumption: Python 3.11+, uv, ruff
  (lint + format), mypy strict, pytest, hypothesis. Not yet confirmed.
- **OD-02** — CLI framework (argparse vs click vs typer) and distribution
  (`uv tool` / `pipx` from PyPI vs a PyInstaller single binary). Linux first,
  but the tool should be usable on as many platforms as possible.
- **OD-03** — Whether the FiveK terms permit publishing the derived dataset and
  fine-tuned weights, and if not, what is published instead (code and scripts
  only, or a different dataset).
- **OD-04** — Whether a vision-language base model (Qwen3.5-2B) is acceptable
  for judging, given the hackathon recommends Qwen3 0.6B, 1.7B or 4B.
  Resolved 2026-11-24: vision-language models are in scope (owner decision).
- **OD-05** — Exact submission contents for the PR to the starter repo.
  Resolved 2026-11-24: see `06-submission/requirements.md`. Remaining
  consequence: the rules require a public model link and public demo video.
