# 02 — Dataset: Design

Read [`../01-executor/design.md`](../01-executor/design.md) first.

## 1. Purpose

Turn FiveK's (original, expert-edit) pairs into (thumbnail, parameter JSON)
training examples, using the executor as the definition of what a parameter
means. The fitted ΔE on the test set is the ceiling of the whole approach: a
model cannot beat what the parameter set can express.

## 2. Pipeline

```text
FiveK pair -> 256 px copies -> optimiser over Params (calls executor.apply)
           -> fitted Params + residual ΔE -> filter -> training example
           -> augmentation -> HF dataset
```

The optimiser is `scipy.optimize` or CMA-ES over the integer-valued parameters
(relaxed to floats during search, rounded at the end). If too slow, port the
executor to PyTorch and fit with gradients, guarded by an agreement test.

## 3. Augmentation

Apply random bad parameters to already well-edited images; the label is the
inverse correction. This enlarges the set and teaches large corrections, but it
can drift from real error distributions, so it is capped and heavy clipping is
avoided.

## 4. Sharpening

FiveK edits are mainly tone and colour, so v1 sharpening is a classical rule
(e.g. Laplacian variance on a full-resolution crop to pick amount) and is not
part of the model's learned output. A learned variant is deferred to
`07-extensions`.

## 5. Failure modes

- Poor fits (high residual ΔE): filtered by threshold; threshold chosen from the
  residual distribution and recorded.
- Label collapse (all labels near average): checked via per-parameter variance.
- License restrictions on redistribution: see OD-01.
- Non-reproducible fits across machines: per-image seeds and bounded workers.

## Open decisions

- **OD-01** — What may be published given FiveK terms (dataset, weights, or code
  only); links to `00-scaffolding` OD-03.
- **OD-02** — Optimiser: scipy vs CMA-ES vs gradient fit via PyTorch port.
- **OD-03** — Residual-ΔE threshold for dropping fits, and the augmentation
  ratio.
- **OD-04** — Whether to include the optional 1:1 detail crop in v1 examples or
  only the thumbnail.
