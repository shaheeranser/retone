# 05 — CLI: Tasks

Read [`design.md`](./design.md) first.

## 1. Inference client

- [x] **T-1.1** — Implement the `llama-server` client with a fake for tests.
      *(CLI-PIP-1, CLI-PIP-3; OD-02)*
- [x] **T-1.2** — Implement thumbnailing in linear light. *(CLI-PIP-1)*
- [x] **T-1.4** — Warn on stderr when the input carries a non-sRGB ICC profile and
      never convert it (`01-executor` design, resolved OD-04). Detection reads the
      profile header in `retone.executor.icc` with no extra dependency. *(CLI-PIP-2)*
- [x] **T-1.3** — Validate and clamp model output with fallback to defaults.
      *(CLI-PIP-2)*

## 2. Command

- [x] **T-2.1** — Wire the `retone` command: single and multiple inputs, default
      output naming, no overwrite of input. *(CLI-CMD-1, CLI-CMD-2, CLI-CMD-3,
      CLI-PIP-4; OD-04)*
- [x] **T-2.2** — Add `--params` and apply-explicit-params (`--set`) modes.
      *(CLI-CMD-4, CLI-CMD-5)*
- [x] **T-2.3** — Add `--json` and the before/after output (`--compare`).
      *(CLI-OUT-2, CLI-OUT-3; OD-05)*
- [x] **T-2.4** — Enforce stdout/stderr split and stable exit codes. *(CLI-OUT-1,
      CLI-OUT-4)*

## 3. Quality

- [x] **T-3.1** — Golden tests for rendered output and JSON; tests for each exit
      code with the fake client.
- [ ] **T-3.2** — Validate end to end with the real quantized model on photos the
      model never saw; record before/after in the run report. *(blocked: needs the
      trained GGUF from `03`/`04`)*
- [ ] **T-3.3** — Verify behaviour on Linux, macOS and Windows where available,
      and document. Linux verified; macOS and Windows pending. *(CLI-PRT-1,
      CLI-PRT-2; OD-03)*
