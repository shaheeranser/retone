# 01 — Executor: Tasks

Read [`design.md`](./design.md) first. Each task cites the requirements it
satisfies.

## 1. Schema

- [x] **T-1.1** — Define the parameter schema (names, ranges, defaults, identity)
      once. *(EXE-SCH-1; OD-01)*
- [x] **T-1.2** — Implement `Params.from_mapping` with defaults, clamping with a
      report, and rejection of unknown or non-integer fields, aggregating errors.
      *(EXE-SCH-2, EXE-SCH-3, EXE-SCH-4)*
- [x] **T-1.3** — Export the JSON Schema. *(EXE-SCH-5)*
- [x] **T-1.4** — Tests: table-driven clamping, defaults, rejection; golden JSON
      Schema.

## 2. Colour pipeline

- [x] **T-2.1** — Implement load/save with BGR to RGB conversion isolated to the
      boundary, 8- and 16-bit. *(EXE-APP-5, EXE-APP-6)*
- [x] **T-2.2** — Implement sRGB to linear and back. *(EXE-APP-4)*
- [x] **T-2.3** — Tests: round-trip property, golden colour patches.

## 3. Operations

- [x] **T-3.1** — Exposure and white balance in linear light. *(EXE-APP-4)*
- [x] **T-3.2** — Tone curve for contrast, highlights, shadows. *(OD-02)*
- [x] **T-3.3** — Saturation and vibrance. *(OD-03)*
- [x] **T-3.4** — Unsharp mask on lightness with diagonal-relative radius.
      *(EXE-APP-7)*
- [x] **T-3.5** — Tests per operation: identity is a no-op, direction of effect,
      range stays valid.

## 4. apply

- [x] **T-4.1** — Compose operations in the fixed order in `apply`. *(EXE-APP-1,
      EXE-APP-4)*
- [x] **T-4.2** — Property tests with hypothesis: identity equals input within
      one step, never mutates input, deterministic, output in range, clamp
      holds, radius scales with diagonal. *(EXE-APP-2, EXE-APP-3, EXE-APP-8,
      EXE-APP-7)*
- [x] **T-4.3** — Test that the executor imports nothing else from the project
      and no ML framework. *(EXE-BND-1, EXE-APP-9)*

## 5. Validation

- [ ] **T-5.1** — Hand-tune parameters on 20 test photos; record before/after and
      the owner's judgement in the run report. *(EXE-QLT-1)*
- [x] **T-5.2** — Measure apply time on a 24 MP photo on the target laptop.
      Measured on a synthetic 24 MP image: 11 to 31 s for the full pipeline,
      see the design's performance note and OD-01.
