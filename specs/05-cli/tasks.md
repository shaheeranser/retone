# 05 — CLI: Tasks

Read [`design.md`](./design.md) first.

## 1. Inference client

- [ ] **T-1.1** — Implement the `llama-server` client with a fake for tests.
      *(CLI-PIP-1, CLI-PIP-3; OD-02)*
- [ ] **T-1.2** — Implement thumbnailing in linear light. *(CLI-PIP-1)*
- [ ] **T-1.3** — Validate and clamp model output with fallback to defaults.
      *(CLI-PIP-2)*

## 2. Command

- [ ] **T-2.1** — Wire the `retone` command: single and multiple inputs, default
      output naming, no overwrite of input. *(CLI-CMD-1, CLI-CMD-2, CLI-CMD-3,
      CLI-PIP-4; OD-04)*
- [ ] **T-2.2** — Add `--params` and apply-explicit-params modes. *(CLI-CMD-4,
      CLI-CMD-5)*
- [ ] **T-2.3** — Add `--json` and the before/after output. *(CLI-OUT-2,
      CLI-OUT-3; OD-05)*
- [ ] **T-2.4** — Enforce stdout/stderr split and stable exit codes. *(CLI-OUT-1,
      CLI-OUT-4)*

## 3. Quality

- [ ] **T-3.1** — Golden tests for rendered output and JSON; tests for each exit
      code with the fake client.
- [ ] **T-3.2** — Validate end to end with the real quantized model on photos the
      model never saw; record before/after in the run report.
- [ ] **T-3.3** — Verify behaviour on Linux, macOS and Windows where available,
      and document. *(CLI-PRT-1, CLI-PRT-2; OD-03)*
