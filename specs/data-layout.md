# Data and artifacts layout

This file is the contract for everything the repository reads and writes outside
`src/`, `tests/`, `specs/` and `reports/`. Both trees below are gitignored: no
input photo, derived dataset, render or review sheet is ever committed.

## Inputs: `data/`

```text
data/
  raw/
    unsplash/        fetched Unsplash corpus plus its source manifest
    photos/          owner's unpaired photos (formerly eval/images)
  human/             evaluation-only before/after pairs
    before/          p0001.jpg
    after/           p0001.png when rotated losslessly, else p0001.jpg
    pairs.jsonl      pair id, paths, provenance, variant, rotation, role
    summary.json     collection counts and skipped/unreadable reports
  processed/
    synthetic/       examples.jsonl, images/, source_paths.json (02 output)
  cache/             downloads and transient files
```

- `data/` is local input only. It is fully ignored by git.
- Small test fixtures live in `tests/fixtures/`, never in `data/`.
- Provenance for published corpora is recorded in the manifests; the private
  owner pairs are never published.
- `data/human/` is **evaluation only** in v1: no labels are fitted and nothing
  from it enters training. If that changes, the milestone design must reopen
  OD-05 first.

## Outputs: `artifacts/`

```text
artifacts/
  executor/          milestone 01 renders, contact sheets, params_*.json
  human/
    <run>/           one review bundle, named by date or run id
      original/      hardlinks or copies of data/human/before
      model/         rendered model output, one file per pair and model label
      reference/     hardlinks or copies of data/human/after
      review.html    side-by-side review page
      ratings.csv    human rating template
      metrics.json   ΔE diagnostics per pair and aggregate
      manifest.json  pair ids, prediction labels, blind-label mapping
  synthetic/         future model-versus-target sheets
```

- `artifacts/` is generated output only. It is fully ignored by git.
- A review bundle is self-contained: `original/`, `model/` and `reference/`
  sit beside each other inside it.
- Runs are immutable once reviewed; a new model or prediction set gets a new
  run directory rather than overwriting an older review.

## Ignore rules

`.gitignore` ignores `/data/` and `/artifacts/` at the repository root. The
leading slash matters: an unanchored `data/` would also ignore
`src/retone/data/` and `tests/data/`. There are no tracked files in either
top-level tree, so a private photo cannot be committed by mistake and
`git status` stays clean.

## Lifecycle

1. Collect raw inputs under `data/raw/` or `data/human/`.
2. Build training examples under `data/processed/` and upload them to Hugging
   Face (never commit them).
3. Render evaluation outputs under `artifacts/`.
4. Record what was run and what was concluded under `reports/` (also ignored).
5. Keep `specs/` as the committed design record.

## Invariants

1. No binary input, derived dataset, model, render or review bundle is committed.
2. `data/human/` is evaluation-only in v1; no leakage into training.
3. Human review is the acceptance signal; ΔE is a diagnostic, not a gate.
4. Collector-managed folders contain only files referenced by the current
   manifest; stale outputs are pruned.
